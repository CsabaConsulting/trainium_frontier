#!/usr/bin/env python3
"""Gate the two source160-profile-guided FP8 dX fusions independently.

Retain both existing independent CPU-reference suites and their bounds. Exact
old-kernel comparisons additionally protect each square-projection case. Only
after all checks compare fresh three-product work and cached dX costs.
"""
import argparse
import json

import torch

from fp8_dual_weight_cache_nki import build_kernel as build_dual
from fp8_gradient_gemm_engine_nki import build_kernel as build_candidate
from fp8_gradient_gemm_nki import build_kernel as build_original
from fp8_gradient_gemm_probe import run as gradient_case
from fp8_packed_weight_cache_nki import build_kernel as build_packer
from fp8_square_dx_probe import functions, inputs, run_case
from fp8_weight_cache_nki import build_kernels as build_calibration
from fp8_weight_cache_probe import require_exact
from mlp_fused_bf16_probe import timing


class ConfiguredGradient:
    def __init__(self, wrapped, variant):
        self.wrapped = wrapped
        self.abs_reduction = variant in ('abs', 'both')
        self.fused_scaling = variant in ('scale', 'both')

    def __getitem__(self, grid):
        def call(g, packed, metadata, lanes, diagnostic):
            return self.wrapped[grid](g, packed, metadata, lanes, diagnostic,
                                      self.abs_reduction, self.fused_scaling)
        return call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('abs', 'scale', 'both'), required=True)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    assert args.steps > 0
    import torch_neuronx  # noqa: F401
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    old, new, dual, pack = [wrap_nki(builder()) for builder in
                            (build_original, build_candidate, build_dual, build_packer)]
    candidate = ConfiguredGradient(new, args.variant)
    calibrate, quantize = [wrap_nki(k) for k in build_calibration()]
    print('FP8_DX_ENGINE_CONTRACT ' + json.dumps(dict(
        variant=args.variant, original_scale_order=True,
        precombined_scale_product=False, original_pair_bound=True,
        original_cpu_references=True, whole_model_or_quality_claim=False)), flush=True)
    domain_cases = ('random', 'zero_gradient', 'zero_weight', 'tiny_gradient',
                    'row_scale_range', 'outliers', 'signed_cancellation')
    for case in domain_cases:
        gradient_case((512, 256, 512), case, candidate, calibrate, quantize, pack, args.steps, False)
    compiled, previous = functions(dual, candidate), functions(dual, old)
    square_cases = [(512, 256, c) for c in
                    ('random', 'zero_gradient', 'zero_weight', 'tiny_gradient',
                     'outliers', 'changed_weight', 'cancellation')]
    square_cases.append((32768, 1024, 'real_random'))
    for tokens, channels, case in square_cases:
        run_case(tokens, channels, case, compiled, dual, candidate)
        values = [v.to('neuron') for v in inputs(tokens, channels, case)]
        before, after = previous['fp8_dx'](*values), compiled['fp8_dx'](*values)
        for i in range(3):
            require_exact(args.variant + ':' + case + ':original_product' + str(i), after[i], before[i])
    x, master, g = [v.to('neuron') for v in inputs(32768, 1024, 'random')]
    weight = master.bfloat16()
    _, metadata, _, packed = dual[2](weight, 2, True)

    def native_dx(gg, ww):
        return gg @ ww

    def old_dx(gg, pp, mm):
        return old[2](gg, pp, mm, 2, False)

    def new_dx(gg, pp, mm):
        return candidate[2](gg, pp, mm, 2, False)

    ndx, odx, qdx = [torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
                     for fn in (native_dx, old_dx, new_dx)]
    require_exact(args.variant + ':cached_dx', qdx(g, packed, metadata), odx(g, packed, metadata))
    calls = {
        'native_three_fresh': lambda: compiled['native'](x, master, g),
        'old_fp8_three_fresh': lambda: previous['fp8_dx'](x, master, g),
        'candidate_fp8_three_fresh': lambda: compiled['fp8_dx'](x, master, g),
        'native_dx_cached': lambda: ndx(g, weight),
        'old_fp8_dx_cached': lambda: odx(g, packed, metadata),
        'candidate_fp8_dx_cached': lambda: qdx(g, packed, metadata),
    }
    measured = timing(calls, args.steps, torch.neuron.synchronize)
    assert all(bool(torch.isfinite(torch.tensor(v)).all()) and min(v) > 0
               for v in measured['samples_ms'].values())
    print('FP8_DX_ENGINE_RESULT ' + json.dumps(dict(
        passed=True, variant=args.variant, domain_cases=len(domain_cases),
        square_cases=len(square_cases), square_outputs_exact_old_fp8=True,
        independent_declared_fp8_gates=True, cases=len(domain_cases) + len(square_cases),
        real_tokens=32768, real_channels=1024, **measured,
        preparation_charged_in_three_product_calls=True, cached_controls_exclude_preparation=True,
        separate_times_not_additive=True, autograd_or_model_gate=False,
        whole_model_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
