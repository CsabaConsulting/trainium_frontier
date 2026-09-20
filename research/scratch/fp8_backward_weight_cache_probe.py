#!/usr/bin/env python3
"""Charge backward-only weight preparation with the source161 gated FP8 dX.

The earlier free-preparation control lost before the dX operation fusions.
Those now improve cached dX, so this reopens preparation on a changed premise.
"""
import argparse
import json

import torch
import torch.nn.functional as F

from fp8_backward_weight_cache_nki import build_kernel as build_prepare
from fp8_dual_weight_cache_nki import build_kernel as build_dual
from fp8_gradient_engine_probe import ConfiguredGradient
from fp8_gradient_engine_profile import engine_prerequisite
from fp8_gradient_gemm_engine_nki import build_kernel as build_gradient
from fp8_packed_weight_cache_probe import decode_legacy, selected_lanes
from fp8_square_dx_probe import functions, inputs, run_case
from fp8_weight_cache_probe import calibration_reference, require_exact
from fresh_start_fp8_probe import e4m3_legacy
from mlp_fused_bf16_probe import timing


def weight_case(channels, case, prepare, dual):
    generator = torch.Generator().manual_seed(164011 + channels)
    w = (torch.randn(channels, channels, generator=generator) / channels**.5).bfloat16()
    if case == 'zero':
        w.zero_()
    elif case == 'tiny':
        w = (w.float() * 2.0**-110).bfloat16()
    elif case == 'large':
        w = (w.float() * 2.0**60).bfloat16()
    elif case == 'signed_zero':
        bits = torch.zeros(w.shape, dtype=torch.int16)
        bits[1::2] = -32768
        w = bits.view(torch.bfloat16)
    elif case == 'subnormal':
        bits = (torch.arange(w.numel()) % 127 + 1).short().reshape(w.shape)
        bits[1::2] |= -32768
        w = bits.view(torch.bfloat16)
    elif case == 'positive':
        w.fill_(.03125)
    elif case == 'negative':
        w.fill_(-.0625)
    elif case == 'outlier':
        w[127, 0] = 2.0**20
        w[128, -1] = -(2.0**20)
    elif case == 'row_scale_range':
        w = torch.ldexp(w.float(), torch.linspace(-60, 30, channels).round().int()[:, None]).bfloat16()
    device = w.to('neuron')
    lanes = selected_lanes(channels)
    metadata, packed = prepare[lanes](device, lanes, True)
    expected, _ = calibration_reference(w)
    raw = e4m3_legacy(w.float() * expected[0, 0])
    width = min(512, channels)
    layout = raw.reshape(channels // 128, 128, channels // width, width).permute(2, 1, 0, 3)
    require_exact(case + ':metadata_cpu', metadata, expected)
    require_exact(case + ':packed_cpu', decode_legacy(packed), layout)
    _, old_metadata, _, old_packed = dual[lanes](device, lanes, True)
    require_exact(case + ':metadata_old', metadata, old_metadata)
    require_exact(case + ':bytes_old', packed, old_packed)
    repeated = prepare[lanes](device, lanes, True)
    for i, (a, b) in enumerate(zip(repeated, (metadata, packed))):
        require_exact(case + ':repeat' + str(i), a, b)
    require_exact(case + ':input_unchanged', device, w)
    print('FP8_BACKWARD_CACHE_WEIGHT_GATE ' + json.dumps(dict(
        passed=True, case=case, channels=channels, independent_metadata_and_layout=True,
        exact_original_metadata_and_bytes=True, repeated_and_input_unchanged=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-root', required=True)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    assert args.steps > 0
    engine_prerequisite(args.engine_root)
    import torch_neuronx  # noqa: F401
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    prepare, dual = [wrap_nki(b()) for b in (build_prepare, build_dual)]
    gradient = ConfiguredGradient(wrap_nki(build_gradient()), 'both')
    weight_cases = ('random', 'zero', 'tiny', 'large', 'signed_zero', 'subnormal',
                    'positive', 'negative', 'outlier', 'row_scale_range')
    for case in weight_cases:
        weight_case(256, case, prepare, dual)
    weight_case(1024, 'real_random', prepare, dual)

    original = functions(dual, gradient)

    def candidate(x, master, g):
        weight = master.bfloat16()
        groups = weight.shape[0] // min(512, weight.shape[0])
        lanes = 2 if groups % 2 == 0 else 1
        metadata, packed = prepare[lanes](weight, lanes, True)
        dx = gradient[2](g, packed, metadata, 2, False)
        return F.linear(x, weight), dx, (g.T @ x).float()

    compiled = dict(native=original['native'], fp8_dx=torch.compile(
        candidate, backend='neuron', fullgraph=True, dynamic=False))
    squares = [(512, 256, case) for case in ('random', 'zero_gradient', 'zero_weight',
               'tiny_gradient', 'outliers', 'changed_weight', 'cancellation')]
    squares.append((32768, 1024, 'real_random'))
    for tokens, channels, case in squares:
        # Original producer is the independent diagnostic/packing authority;
        # only the compiled candidate uses the new backward-only producer.
        run_case(tokens, channels, case, compiled, dual, gradient)
        values = [v.to('neuron') for v in inputs(tokens, channels, case)]
        before, after = original['fp8_dx'](*values), compiled['fp8_dx'](*values)
        for i in range(3):
            require_exact(case + ':original_product' + str(i), after[i], before[i])
    values = [v.to('neuron') for v in inputs(32768, 1024, 'random')]
    calls = {
        'native_three_fresh': lambda: original['native'](*values),
        'fused_dx_dual_preparation_three_fresh': lambda: original['fp8_dx'](*values),
        'fused_dx_backward_preparation_three_fresh': lambda: compiled['fp8_dx'](*values),
    }
    measured = timing(calls, args.steps, torch.neuron.synchronize)
    assert all(bool(torch.isfinite(torch.tensor(v)).all()) and min(v) > 0
               for v in measured['samples_ms'].values())
    print('FP8_BACKWARD_CACHE_RESULT ' + json.dumps(dict(
        passed=True, weight_cases=11, square_cases=8, cases=19,
        independent_cpu_cache_and_product_gates=True, exact_original_three_products=True,
        **measured, production_tokens=32768, production_channels=1024,
        all_master_casts_calibration_and_required_packing_charged=True,
        no_forward_fp8_cache_borrowed=True, autograd_or_model_gate=False,
        whole_model_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
