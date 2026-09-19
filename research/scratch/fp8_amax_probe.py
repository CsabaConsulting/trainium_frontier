#!/usr/bin/env python3
"""Gate exact amax alternatives before timing their real calibration cost.

The original max/min implementation and an independent CPU scale/quantizer
reference are both retained. Include pure subnormal inputs so the exhaustive
mixed-code case cannot conceal a maximum-reduction underflow discrepancy.
"""
import argparse
import json
import torch

from fp8_weight_cache_nki import build_kernels
from fp8_weight_cache_probe import calibration_reference, require_exact
from mlp_fused_bf16_probe import timing


def gate(name, value, mode, calibrate, quantize):
    expected, quantized = calibration_reference(value)
    device = value.to('neuron')
    original = calibrate[1](device, False, 0)
    candidate = calibrate[1](device, True, mode)
    records = [
        require_exact(name + ':original_cpu_metadata', original, expected, fatal=False),
        require_exact(name + ':candidate_cpu_metadata', candidate, expected, fatal=False),
        require_exact(name + ':candidate_original_metadata', candidate, original, fatal=False),
        require_exact(name + ':repeat', calibrate[1](device, True, mode), candidate, fatal=False),
        require_exact(name + ':input_unchanged', device, value, fatal=False),
        require_exact(name + ':quantized_cpu', quantize[2](device, candidate, 2), quantized, fatal=False),
    ]
    passed = all(r['passed'] for r in records)
    print('FP8_AMAX_GATE ' + json.dumps(dict(name=name, mode=mode, shape=list(value.shape),
          expected_metadata=expected[0].tolist(), original_metadata=original.cpu()[0].tolist(),
          candidate_metadata=candidate.cpu()[0].tolist(), passed=passed)), flush=True)
    assert passed, 'Exact calibration contract failed; inspect original and candidate separately'
    return device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--steps', type=int, default=12)
    args = parser.parse_args()
    assert not args.benchmark or args.real_shape
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    calibrate, quantize = (wrap_nki(k) for k in build_kernels())
    gen = torch.Generator().manual_seed(12719)
    base = torch.randn(256, 256, generator=gen).bfloat16()
    cases = dict(random=base, zero=torch.zeros_like(base),
                 negative=-base.abs(), positive=base.abs(),
                 tiny=(base.float() * 2.0**-110).bfloat16(),
                 large=(base.float() * 2.0**90).bfloat16())
    boundary = torch.tensor([0.0, -0.0, 1.0, -1.0, 2.0, -2.0,
                             2.0**-100, -2.0**-100, 2.0**100, -2.0**100,
                             0.99609375, 1.9921875, -0.99609375, -1.9921875,
                             2.0**-126, -2.0**-126], dtype=torch.bfloat16)
    cases['exponent_boundaries'] = boundary.repeat(4096).reshape(256, 256)
    codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    codes[~torch.isfinite(codes) | (codes.float().abs() > 2.0**100)] = 0
    cases['all_declared_finite_codes'] = codes.reshape(256, 256)
    cases['smallest_normal'] = torch.full_like(base, 2.0**-126)
    cases['subnormal_only'] = torch.full_like(base, 2.0**-130)
    cases['smallest_subnormal'] = torch.full_like(base, -(2.0**-133))
    count = 0
    for name, value in cases.items():
        gate(name, value, args.mode, calibrate, quantize)
        count += 1
    if args.real_shape:
        for cols in (1024, 4096):
            value = (torch.randn(16384, cols, generator=gen) * .03125).bfloat16()
            device = gate('real_' + str(cols), value, args.mode, calibrate, quantize)
            count += 1
            if args.benchmark:
                measured = timing({
                    'original_sequential': lambda: calibrate[1](device, False, 0),
                    'original_parallel': lambda: calibrate[1](device, True, 0),
                    'candidate': lambda: calibrate[1](device, True, args.mode),
                }, args.steps, torch.neuron.synchronize)
                print('FP8_AMAX_TIMING ' + json.dumps(dict(mode=args.mode,
                      shape=list(value.shape), **measured,
                      calibration_only=True, full_pipeline_or_quality_gain=False)), flush=True)
    print('FP8_AMAX_RESULT ' + json.dumps(dict(mode=args.mode, cases=count,
          passed=True, full_pipeline_or_quality_gain=False)), flush=True)


if __name__ == '__main__':
    main()
