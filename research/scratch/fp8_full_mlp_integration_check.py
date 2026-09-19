#!/usr/bin/env python3
"""Gate real Linear masters, refreshed caches, padding and native checkpoint load.

The implementation/rounding authority is the source135 independent all-four-product gate.
This check uses the prior separate FP8 cache producer, CPU cache references and full VJP as
an integration reference. Quantized training is intentionally different from
BF16; exactness is required against the same declared FP8 algorithm, not against
unquantized training. No quality or complete-model speed claim follows.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T

from fp8_full_mlp_integration import install
from fp8_complete_mlp_probe import build_functions, prepare_caches
from fp8_compiled_cache_probe import cache_gate
from fp8_weight_cache_nki import build_kernels as build_calibration
from fp8_packed_weight_cache_nki import build_kernel as build_pack
from fp8_mlp_data_backward_nki import build_kernel as build_backward
from fp8_gradient_prepare_nki import build_kernels as build_gradient
from fp8_weight_gradient_nki import build_kernels as build_product
from fp8_mlp_token_rows_nki import build_kernel
from fp8_weight_cache_probe import calibration_reference, require_exact
from fp8_rms_scale_contract import normalize, policy, require_domain
from mlp_fused_integration_check import Probe, refresh


def compiled_nodes(output):
    """Check the intended inner compiled graph across explicit view boundaries."""
    pending, seen, names = [output.grad_fn], set(), []
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.append(type(node).__name__)
        pending.extend(child for child, _ in node.next_functions)
    return names


def run(channels, separate_region=False):
    from torch_neuronx import wrap_nki
    torch.manual_seed(8801 + channels)
    T.USE_WEIGHT_CACHE = T.USE_BF16_NORM_OUTPUT = True
    T.USE_GATED_MLP = T.USE_COALESCED_SWIGLU = T.USE_NKI_RELU2 = False
    T.USE_PARALLEL_BLOCK = False
    T.RELU2_TAU = 0.0
    reference = Probe(channels).to('neuron')
    candidate = copy.deepcopy(reference)
    original_keys = tuple(candidate.state_dict())
    original_ids = tuple(id(p) for p in candidate.parameters())
    rng = torch.get_rng_state().clone()
    contract = install(candidate, 'neuron', T, separate_region=separate_region)
    assert torch.equal(rng, torch.get_rng_state())
    assert tuple(candidate.state_dict()) == original_keys
    assert tuple(id(p) for p in candidate.parameters()) == original_ids
    # Only the explicit boundary variant permits a caller graph break. Its
    # inner complete MLP is still fullgraph=True; every numerical gate remains.
    optimized = torch.compile(candidate, backend='neuron', fullgraph=not separate_region, dynamic=False)
    kernel = wrap_nki(build_kernel())
    calibrate = wrap_nki(build_calibration()[0])
    pack = wrap_nki(build_pack())
    backward = wrap_nki(build_backward())
    partial, finish = (wrap_nki(k) for k in build_gradient())
    product = wrap_nki(build_product()[2])
    reference_apply = build_functions(channels, kernel, backward, partial, finish, product)[1]
    scale, bound = policy(channels)
    fixed = (torch.tensor([scale, 1.0 / scale, bound], dtype=torch.float32).repeat(128, 1).to('neuron'),
             torch.ones((128, 3), dtype=torch.float32).to('neuron'))
    generator = torch.Generator().manual_seed(8873 + channels)
    inputs = [normalize(torch.randn(4, 128, channels, generator=generator)) for _ in range(2)]
    gradients = [(torch.randn(4, 128, channels, generator=generator) * .03125).bfloat16()
                 for _ in range(2)]
    cmlp, rmlp = candidate.transformer.h[0].mlp, reference.transformer.h[0].mlp
    previous_cache = None
    changed_cache_values = []
    for cached in (False, True):
        for step in range(2):
            candidate.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            # Independent CPU scale selection and legacy-format QDQ; never
            # reuse the candidate's prepared tensors as reference inputs.
            cpu_caches = [calibration_reference(linear.weight.detach().cpu().bfloat16())
                          for linear in (rmlp.c_fc, rmlp.c_proj)]
            (um, uq), (dm, dq) = cpu_caches
            if cached:
                refresh(candidate)
                for name, linear, expected in zip(('up', 'down'), (cmlp.c_fc, cmlp.c_proj), cpu_caches):
                    q, meta, packed, packed_transpose = linear._fp8_training_cache
                    assert not any(v.requires_grad for v in (q, meta, packed, packed_transpose))
                    require_exact(f'D{channels}:{step}:{name}:fresh_qdq', q, expected[1])
                    require_exact(f'D{channels}:{step}:{name}:fresh_metadata', meta, expected[0])
                current = cmlp.c_fc._fp8_training_cache[0].cpu()
                if previous_cache is not None:
                    changed = int((current != previous_cache).sum())
                    assert changed > 0, 'Reference updates failed to exercise stale-cache detection'
                    changed_cache_values.append(changed)
                previous_cache = current.clone()
            else:
                assert cmlp.c_fc._fp8_training_cache is None
                assert cmlp.c_proj._fp8_training_cache is None
            device_caches = prepare_caches((rmlp.c_fc.weight, rmlp.c_proj.weight),
                                            calibrate, pack, 'full', fixed)
            cache_gate(f'D{channels}:{cached}:{step}:reference_cpu_cache', device_caches,
                       (rmlp.c_fc.weight, rmlp.c_proj.weight), fixed)
            if cached:
                cu, cmu, cfu, cbu = cmlp.c_fc._fp8_training_cache
                cd, cmd, cfd, cbd = cmlp.c_proj._fp8_training_cache
                candidate_caches = (cu, cd, cmu, cmd, cfu, cfd, cbu, cbd,
                                    cmlp._fp8_activation_meta, cmlp._fp8_one_meta)
                cache_gate(f'D{channels}:{cached}:{step}:candidate_cpu_cache', candidate_caches,
                           (cmlp.c_fc.weight, cmlp.c_proj.weight), fixed)
                for i, (value, truth) in enumerate(zip(candidate_caches, device_caches)):
                    require_exact(f'D{channels}:{step}:all_cache_bytes:{i}', value, truth)
            cache_snapshot = [tuple(v.cpu().clone() for v in linear._fp8_training_cache)
                              for linear in (cmlp.c_fc, cmlp.c_proj)] if cached else None
            for micro, (x, gradient) in enumerate(zip(inputs, gradients)):
                bx = x.to('neuron').requires_grad_()
                cx = x.to('neuron').requires_grad_()
                expected = reference_apply(bx.reshape(512, channels), rmlp.c_fc.weight,
                                           rmlp.c_proj.weight, *device_caches).reshape(x.shape)
                output = optimized(cx)
                if separate_region:
                    assert 'CompiledFunctionBackward' in compiled_nodes(output)
                else:
                    assert type(output.grad_fn).__name__ == 'CompiledFunctionBackward'
                require_exact(f'D{channels}:{cached}:{step}:{micro}:output', output, expected)
                g = gradient.to('neuron')
                expected.backward(g)
                output.backward(g)
                require_exact(f'D{channels}:{cached}:{step}:{micro}:input_gradient', cx.grad, bx.grad)
                require_exact(f'D{channels}:{cached}:{step}:{micro}:input_unchanged', cx, x)
                for (name, p), (other, q) in zip(candidate.named_parameters(), reference.named_parameters()):
                    assert name == other and p.dtype == q.dtype == torch.float32
                    assert p.grad is not None and p.grad.dtype == q.grad.dtype == torch.float32
                    require_exact(f'D{channels}:{cached}:{step}:{micro}:master_gradient:{name}', p.grad, q.grad)
            if cached:
                for linear, snapshot in zip((cmlp.c_fc, cmlp.c_proj), cache_snapshot):
                    for value, before in zip(linear._fp8_training_cache, snapshot):
                        require_exact(f'D{channels}:{step}:cache_unchanged_by_microbatches', value, before)
            # Same real reference update on both copies. This checks fresh
            # caches without confusing compounded training drift with wiring.
            with torch.no_grad():
                for p, q in zip(candidate.parameters(), reference.parameters()):
                    q.add_(q.grad, alpha=-.001)
                    p.copy_(q)
    refresh(candidate)
    refresh(reference)
    # Exercise the FP8 TRAINING path, including non-aligned lengths and
    # changes to future rows / the other batch. Model eval alone cannot prove
    # prediction independence of a training-only quantized implementation.
    with torch.no_grad():
        x = inputs[0][:2, :31].contiguous()
        require_domain(x)
        original = optimized(x.to('neuron')).cpu()
        repeated = optimized(x.to('neuron')).cpu()
        require_exact(f'D{channels}:training_repeat', repeated, original)
        changed = x.clone()
        changed[:, 16:] = inputs[1][:2, 16:31]
        future = optimized(changed.to('neuron')).cpu()
        require_exact(f'D{channels}:training_future_independence', future[:, :16], original[:, :16])
        changed = x.clone(); changed[1] = inputs[1][1, :31]
        other = optimized(changed.to('neuron')).cpu()
        require_exact(f'D{channels}:training_other_batch_independence', other[0], original[0])
        alone = optimized(x[:1].to('neuron')).cpu()
        require_exact(f'D{channels}:training_single_batch', alone[0], original[0])
        # Same real rows on either side of the512-token pad boundary.
        full = optimized(inputs[0].to('neuron')).cpu()
        short = optimized(inputs[0][:, :127].contiguous().to('neuron')).cpu()
        require_exact(f'D{channels}:training_padding_independence', short, full[:, :127])
    candidate.eval(); reference.eval()
    restored = Probe(channels).to('neuron')
    restored.load_state_dict(candidate.state_dict(), strict=True)
    restored.eval(); refresh(restored)
    with torch.no_grad():
        xx = inputs[0][:2, :31].to('neuron')
        require_exact(f'D{channels}:native_eval', candidate(xx), reference(xx))
        require_exact(f'D{channels}:strict_native_restore', restored(xx), candidate(xx))
    assert tuple(candidate.state_dict()) == original_keys
    assert tuple(id(p) for p in candidate.parameters()) == original_ids
    print('FP8_FULL_MLP_INTEGRATION_GATE ' + json.dumps(dict(
        channels=channels, tokens=512, cached_and_uncached=True, microbatches=2,
        separate_compiled_region=separate_region,
        reference_updates=4, exact_outputs_input_and_fp32_master_gradients=True,
        changed_cache_values=changed_cache_values, training_padding_and_row_independence=True,
        native_eval_and_checkpoint_restore_exact=True, contract=contract,
        independent_algorithm_authority='source135 four-product bounds and source137 exact dual-cache gate',
        whole_model_speed_or_quality_established=False, passed=True)), flush=True)
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-width', action='store_true')
    parser.add_argument('--separate-region', action='store_true')
    args = parser.parse_args()
    import torch_neuronx
    torch.set_num_threads(4)
    run(256, args.separate_region)
    if args.real_width:
        # Each independent model width exercises cached/uncached training and
        # four no-grad prediction shapes. Their shared Probe.forward code object
        # otherwise exhausts Dynamo's eight-specialization limit across widths.
        # Drop only this process's inactive Python graph guards between models;
        # keep the default limit, all numerical checks and device caches.
        torch._dynamo.reset()
        print('FP8_INTEGRATION_HARNESS_RESET ' + json.dumps(dict(
            completed_width=256, next_width=1024, process_local_dynamo_guards_only=True,
            numerical_contract_changed=False, shared_device_cache_purged=False)), flush=True)
        run(1024, args.separate_region)
    print('FP8_FULL_MLP_INTEGRATION_RESULT ' + json.dumps(dict(passed=True, real_width=args.real_width,
          separate_compiled_region=args.separate_region,
          full_fp8_backward=True, full_model_or_quality_promotion=False)), flush=True)


if __name__ == '__main__':
    main()
