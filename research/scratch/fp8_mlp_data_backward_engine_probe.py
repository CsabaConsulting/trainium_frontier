#!/usr/bin/env python3
"""Transfer the gated sequential-scale fusion to both MLP data-gradient dots.

Run the unchanged eleven independent MLP stage cases plus the current 32768
token shape. The original kernel remains the exact-output authority for the
declared FP8 algorithm. BF16 native is a separately bounded cost comparator.
"""
import argparse
import json

import torch

from fp8_gradient_gemm_nki import build_kernel as build_dot
from fp8_mlp_data_backward_engine_nki import build_kernel as build_candidate
from fp8_mlp_data_backward_nki import build_kernel as build_original
from fp8_mlp_data_backward_probe import native_gate, repack_gate, run
from fp8_packed_weight_cache_nki import build_kernel as build_packer
from fp8_packed_weight_cache_probe import pack_gate, selected_lanes
from fp8_weight_cache_nki import build_kernels
from fp8_weight_cache_probe import require_exact
from mlp_fused_bf16_probe import timing


class PairedKernel:
    """Keep the original probe's no-fused-abs branch an actual old-kernel call."""
    def __init__(self, original, candidate):
        self.original, self.candidate = original, candidate

    def __getitem__(self, grid):
        def call(g, r, pd, pu, dm, um, lanes, tile, diagnostic, fused_abs):
            kernel = self.candidate if fused_abs else self.original
            return kernel[grid](g, r, pd, pu, dm, um, lanes, tile, diagnostic, fused_abs)
        return call


def benchmark(original, candidate, calibrate, quantize, prepare, steps):
    tokens, channels, hidden = 32768, 1024, 4096
    generator = torch.Generator().manual_seed(9391 + channels + tokens)
    g = (torch.randn(tokens, channels, generator=generator) * .03125).bfloat16()
    r = torch.randn(tokens, hidden, generator=generator).relu().bfloat16()
    dt = (torch.randn(hidden, channels, generator=generator) / hidden**.5).bfloat16()
    ut = (torch.randn(channels, hidden, generator=generator) / channels**.5).bfloat16()
    dm, dq_t, pd = pack_gate('timing:down_transpose', dt, calibrate, quantize, prepare)
    um, uq_t, pu = pack_gate('timing:up_transpose', ut, calibrate, quantize, prepare)
    gd, rd = g.to('neuron'), r.to('neuron')
    down, up = dq_t.T.contiguous(), uq_t.T.contiguous()
    dl, ul = selected_lanes(hidden), selected_lanes(channels)
    rebuilt_d, packed_d = prepare[dl](down.T.contiguous(), dm, dl)
    rebuilt_u, packed_u = prepare[ul](up.T.contiguous(), um, ul)
    require_exact('timing:down_repack_qdq', rebuilt_d, dq_t)
    require_exact('timing:up_repack_qdq', rebuilt_u, uq_t)
    repack_gate('timing:down', packed_d, pd)
    repack_gate('timing:up', packed_u, pu)

    def native_data(gg, rr, dd, uu):
        pre = (gg @ dd) * (rr * 2.0)
        return pre @ uu, pre

    native = torch.compile(native_data, backend='neuron', fullgraph=True, dynamic=False)
    native_gate(g, down.cpu(), up.cpu(), r, native(gd, rd, down, up))

    def update(kernel, fresh):
        if fresh:
            _, current_d = prepare[dl](down.T.contiguous(), dm, dl)
            _, current_u = prepare[ul](up.T.contiguous(), um, ul)
        else:
            current_d, current_u = pd, pu
        return kernel[2](gd, rd, current_d, current_u, dm, um, 2, 128, False, True)

    expected = [v.cpu() for v in update(original, False)]
    calls = {
        'native_bf16_surrogate_chain': lambda: native(gd, rd, down, up),
        'old_fp8_fused_abs_extra_cache_and_chain': lambda: update(original, True),
        'candidate_fp8_extra_cache_and_chain': lambda: update(candidate, True),
        'old_fp8_fused_abs_cached_chain': lambda: update(original, False),
        'candidate_fp8_cached_chain': lambda: update(candidate, False),
    }
    for name, fn in calls.items():
        if name.startswith('native'):
            continue
        for field, value, truth in zip(('dx', 'dpre'), fn(), expected):
            require_exact('timing:' + name + ':' + field, value, truth)
    measured = timing(calls, steps, torch.neuron.synchronize)
    assert all(bool(torch.isfinite(torch.tensor(v)).all()) and min(v) > 0
               for v in measured['samples_ms'].values())
    for field, value, truth in zip(('dx', 'dpre'), update(candidate, True), expected):
        require_exact('timing:repeat_after_measurement:' + field, value, truth)
    return dict(**measured, tokens=tokens, channels=channels, hidden=hidden,
                data_gradient_chains_per_call=1,
                extra_backward_transposes_and_packing_charged=True,
                shared_forward_calibration_already_paid=True,
                native_control_uses_same_qdq_weights=True,
                cached_control_preparation_excluded=True,
                independent_native_chain_bound=True, separate_times_not_additive=True,
                weight_gradients_fp32_accumulation_optimizer_model_excluded=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    assert args.steps > 0
    import torch_neuronx  # noqa: F401
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    original, candidate, dot, prepare = [wrap_nki(b()) for b in
                                        (build_original, build_candidate, build_dot, build_packer)]
    calibrate, quantize = [wrap_nki(k) for k in build_kernels()]
    paired = PairedKernel(original, candidate)
    print('FP8_MLP_ENGINE_CONTRACT ' + json.dumps(dict(
        unchanged_independent_references_and_bounds=True,
        original_kernel_is_exact_algorithm_control=True,
        abs_reduction_already_present=True, original_scale_order=True,
        fp32_intermediate_operations=True, precombined_scale_product=False,
        original_bf16_vjp_exact=False, model_or_quality_claim=False)), flush=True)
    for case in ('random', 'zero_gradient', 'zero_relu', 'zero_down', 'zero_up',
                 'tiny_gradient', 'subnormal_gradient', 'signed_zero_gradient',
                 'row_scale_range', 'outliers'):
        run((512, 256, 512), case, paired, dot, calibrate, quantize, prepare,
            args.steps, False, True)
    for tokens in (16384, 32768):
        run((tokens, 1024, 4096), 'real_random', paired, dot, calibrate, quantize,
            prepare, args.steps, False, True)
    measured = benchmark(original, candidate, calibrate, quantize, prepare, args.steps)
    print('FP8_MLP_ENGINE_RESULT ' + json.dumps(dict(
        passed=True, original_cases=11, cases=12, production_tokens=32768,
        all_diagnostic_and_normal_outputs_exact_original=True,
        original_independent_stage_gates=True,
        row_independence_at_both_real_shapes=True,
        **measured, autograd_or_model_gate=False, whole_model_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
