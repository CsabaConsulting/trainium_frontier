#!/usr/bin/env python3
"""Screen FP8 only for a square projection's input-gradient product.

Forward and weight-gradient products retain BF16 operands. The input-gradient
algorithm quantizes G per row and W per matrix with the already characterized
legacy E4M3 policy. All current FP32-master casts, calibration and dual packing
are charged once per measured three-product call; no free forward FP8 cache is
assumed. This is a component screen, not an autograd/model or quality gate.
"""
import argparse
import gc
import json

import torch
import torch.nn.functional as F

from fp8_dual_weight_cache_nki import build_kernel as build_prepare
from fp8_gradient_gemm_nki import build_kernel as build_gradient
from fp8_mlp_island_probe import pair_gate, row_reference
from fp8_packed_weight_cache_probe import decode_legacy, selected_lanes
from fp8_weight_cache_probe import calibration_reference, require_exact, quantization_diagnostic
from fresh_start_fp8_probe import e4m3_legacy
from mlp_fused_bf16_probe import timing
from serial_qkv_probe import audit, product_reference


def emit(marker, **record):
    print(marker + ' ' + json.dumps(record), flush=True)


def functions(prepare, gradient):
    def native(x, master, g):
        weight = master.bfloat16()
        return F.linear(x, weight), g @ weight, (g.T @ x).float()

    def candidate(x, master, g):
        weight = master.bfloat16()
        groups = weight.shape[0] // min(512, weight.shape[0])
        lanes = 2 if groups % 2 == 0 else 1
        # The current producer also writes two unused representations. Keep
        # those costs in this first conservative screen; no free packing.
        _, meta, _, backward = prepare[lanes](weight, lanes, True)
        dx = gradient[2](g, backward, meta, 2, False)
        return F.linear(x, weight), dx, (g.T @ x).float()

    return {name: torch.compile(call, backend='neuron', fullgraph=True, dynamic=False)
            for name, call in [('native', native), ('fp8_dx', candidate)]}


def inputs(tokens, channels, case):
    gen = torch.Generator().manual_seed(158013 + channels)
    x = F.rms_norm(torch.randn(tokens, channels, generator=gen), (channels,)).bfloat16()
    w = torch.randn(channels, channels, generator=gen) / channels**.5
    g = (torch.randn(tokens, channels, generator=gen) * .01).bfloat16()
    if case == 'zero_gradient': g.zero_()
    if case == 'zero_weight': w.zero_()
    if case == 'tiny_gradient': g = (g.float() * 2.0**-14).bfloat16()
    if case == 'outliers':
        g[127, 0] = 2.0**15; g[128, -1] = -(2.0**15)
        w[0, 127] = 8; w[-1, 128] = -8
    if case == 'changed_weight': w = w * -.75 + .03125
    if case == 'cancellation':
        g[:, 128:256] = g[:, :128]
        w[128:256] = -w[:128]
    return x, w, g


def run_case(tokens, channels, case, compiled, prepare, gradient):
    host = inputs(tokens, channels, case)
    x, w, g = host
    device = tuple(v.to('neuron') for v in host)
    results = {name: tuple(v.detach().cpu() for v in fn(*device))
               for name, fn in compiled.items()}
    for name, values in results.items():
        assert values[0].dtype == values[1].dtype == torch.bfloat16
        assert values[2].dtype == torch.float32
        for index, left, right in [(0, x, w.bfloat16().T), (2, g.T, x)]:
            # These products use the original BF16 operands. The weight
            # derivative preserves its BF16 product/FP32 destination contract.
            truth, bound = product_reference(left, right)
            audit(f'{case}:{name}:product{index}', values[index], truth, bound)
        repeated = compiled[name](*device)
        for index in range(3):
            require_exact(f'{case}:{name}:repeat{index}', repeated[index], values[index])
    truth, bound = product_reference(g, w.bfloat16())
    audit(f'{case}:native:dx', results['native'][1], truth, bound)
    require_exact(case + ':unchanged_forward', results['fp8_dx'][0], results['native'][0])
    require_exact(case + ':unchanged_weight_product', results['fp8_dx'][2], results['native'][2])

    lanes = selected_lanes(channels)
    restored, meta, forward, backward = prepare[lanes](device[1].bfloat16(), lanes, True)
    expected_meta, expected_qdq = calibration_reference(w.bfloat16())
    require_exact(case + ':weight_metadata', meta, expected_meta)
    require_exact(case + ':weight_quantizer', restored, expected_qdq)
    raw_w = e4m3_legacy(w.bfloat16().float() * expected_meta[0, 0])
    width = min(512, channels)
    expected_forward = raw_w.reshape(channels // width, width, channels // 128, 128).permute(0, 3, 2, 1)
    expected_backward = raw_w.reshape(channels // 128, 128, channels // width, width).permute(2, 1, 0, 3)
    require_exact(case + ':forward_packing', decode_legacy(forward), expected_forward)
    require_exact(case + ':backward_packing', decode_legacy(backward), expected_backward)
    dx, gq, gm, dot = (v.cpu() for v in gradient[2](device[2], backward, meta, 2, True))
    expected_gm, expected_gq = row_reference(g)
    require_exact(case + ':gradient_metadata', gm, expected_gm)
    require_exact(case + ':gradient_quantizer', gq, expected_gq)
    pair_gate(case + ':fp8_dx_dot', dot, gq.float() * gm[:, 0:1], raw_w,
              gm[:, 1:2], expected_meta[0, 1], full=tokens <= 512)
    require_exact(case + ':dx_rounding', dx, dot.bfloat16())
    require_exact(case + ':compiled_dx', results['fp8_dx'][1], dx)
    quantization_diagnostic(case + ':dx_quantization_error', dx, results['native'][1])
    for index in range(3): require_exact(f'{case}:input{index}_unchanged', device[index], host[index])
    if case in ('random', 'real_random'):
        changed = g.clone(); changed[127:129].mul_(-7)
        varied = compiled['fp8_dx'](device[0], device[1], changed.to('neuron'))
        keep = torch.ones(tokens, dtype=torch.bool); keep[127:129] = False
        require_exact(case + ':other_dx_rows_unchanged', varied[1].cpu()[keep], dx[keep])
        require_exact(case + ':forward_independent_of_gradient', varied[0], results['fp8_dx'][0])
    emit('FP8_SQUARE_DX_CASE', case=case, tokens=tokens, channels=channels, passed=True,
         native_forward_and_weight_product_exact=True, declared_fp8_dx_independently_gated=True,
         all_master_casts_calibration_and_packing_included=True,
         autograd_or_model_gate=False, whole_model_or_quality_claim=False)
    del results, device, host
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    assert args.steps > 0
    import torch_neuronx  # noqa: F401
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    prepare, gradient = wrap_nki(build_prepare()), wrap_nki(build_gradient())
    compiled = functions(prepare, gradient)
    emit('FP8_SQUARE_DX_CONTRACT', forward='original BF16', weight_derivative='original BF16 product to FP32',
         input_derivative='per-row G E4M3 times per-matrix W E4M3, double-row pair rule, BF16 output',
         changed_vjp=True, no_free_shared_forward_fp8_cache=True,
         exact_original_backward=False, quality_gate=False)
    cases = ['random', 'zero_gradient', 'zero_weight', 'tiny_gradient', 'outliers',
             'changed_weight', 'cancellation']
    for case in cases: run_case(512, 256, case, compiled, prepare, gradient)
    run_case(32768, 1024, 'real_random', compiled, prepare, gradient)
    device = tuple(v.to('neuron') for v in inputs(32768, 1024, 'random'))
    calls = {name: (lambda fn=fn: fn(*device)) for name, fn in compiled.items()}
    result = timing(calls, args.steps, torch.neuron.synchronize)
    emit('FP8_SQUARE_DX_TIMING', tokens=32768, channels=1024, **result,
         units='ms per forward, input gradient and weight gradient with fresh cache preparation',
         master_cast_calibration_and_all_dual_packing_charged=True,
         autograd_accumulation_optimizer_and_full_model_excluded=True)
    emit('FP8_SQUARE_DX_RESULT', passed=True, cases=len(cases)+1,
         native_forward_and_weight_product_exact=True,
         independent_declared_fp8_dx_gates=True, real_tokens=32768, real_channels=1024,
         autograd_or_model_gate=False, whole_model_or_quality_claim=False)


if __name__ == '__main__':
    main()
