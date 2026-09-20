#!/usr/bin/env python3
"""Locate the first complete-attention difference without changing its gate.

The original native forward is the authority for the failed paired comparison.
Partial replacements isolate Q, K and V. Separately materialized projections
and a CPU-FP64-derived unrounded V are explicit precision hypotheses, not new
acceptance references. Returning/fixing intermediates can itself change compiler
fusion; these observations therefore never promote an integration or quality run.
"""
import argparse
import copy
import json
import types

import torch
import torch.nn.functional as F

from serial_qkv_integration_check import Probe, T, refresh, require_component
from serial_qkv_integration import install
from serial_qkv_nki import build_kernel, make_autograd


def emit(marker, **record):
    print(marker + ' ' + json.dumps(record), flush=True)


def comparison(name, actual, expected):
    a, b = (x.detach().cpu().double() for x in (actual, expected))
    error = (a - b).abs()
    rms = b.square().mean().sqrt()
    relative = float(error.square().mean().sqrt() / rms.clamp_min(1e-30))
    failures = int((error > .03 * b.abs() + .008 * rms + 1e-30).sum())
    record = dict(name=name, exact=bool(torch.equal(a, b)),
                  finite=bool(torch.isfinite(a).all()), bound_failures=failures,
                  relative_rms=relative, max_abs=float(error.max()),
                  unchanged_paired_gate_passed=(failures == 0 and relative < .004))
    emit('SERIAL_QKV_BOUNDARY_COMPARE', **record)
    assert record['finite'], record
    return record


def setup_partial(model, layout, replacement):
    """Diagnostic only: choose which of three original projections to replace."""
    attn = model.transformer.h[0].attn
    original = type(attn).forward
    from torch_neuronx import wrap_nki
    kernel = wrap_nki(build_kernel())
    token_major = layout == 'tokens'

    def call(xq, xk, xv, wq, wk, wv):
        tile = 128 if token_major else (512 if xq.shape[0] % 1024 == 0 else 256)
        return kernel[2](xq, xk, xv, wq, wk, wv, 2, tile, token_major)

    apply = make_autograd(call)

    def forward(self, xq, xk, xv, ve, cos, sin, mask):
        values = (xq, xk, xv)
        linears = (self.c_q, self.c_k, self.c_v)
        weights = tuple(m._cached_weight if m._cached_weight is not None
                        else m.weight.bfloat16() for m in linears)
        rows = tuple(x.reshape(-1, self.n_embd) for x in values)
        if replacement == 'native_preprojected':
            result = tuple(F.linear(x, w) for x, w in zip(rows, weights))
        else:
            result = apply(*rows, *weights)
        selections = {'nki_q': (0,), 'nki_k': (1,), 'nki_v': (2,),
                      'nki_qk': (0, 1), 'nki_qv': (0, 2), 'nki_kv': (1, 2)}
        if replacement in selections:
            selected = selections[replacement]
            result = tuple(result[i] if i in selected else F.linear(rows[i], weights[i])
                           for i in range(3))
        shaped = tuple(x.reshape(xq.shape[0], xq.shape[1], self.n_head, self.head_dim)
                       for x in result)
        assert all(x.is_contiguous() for x in shaped)
        return original(self, xq, xk, xv, ve, cos, sin, mask, qkv=shaped)

    attn.forward = types.MethodType(forward, attn)


def run(channels, layout, cached, with_ve, batch=4, length=128, seed=None,
        include_fp32_arm=False, arm_set='standard', gate_init='random',
        skip_materialized=False):
    seed = 151031 + channels if seed is None else seed
    torch.manual_seed(seed)
    torch._dynamo.reset()
    baseline = Probe(channels, length)
    if gate_init == 'zero':
        # GPT.init_weights zeros this gate, whereas the original Probe uses
        # Linear's random initializer. Keep the output projection nonzero:
        # zeroing it would hide upstream errors. This is a diagnostic control,
        # never a substitute for the random/trained-gate acceptance fixture.
        with torch.no_grad():
            baseline.transformer.h[0].attn.ve_gate.weight.zero_()
    baseline = baseline.to('neuron')
    # deepcopy cannot copy differentiable nonleaf weight-cache tensors.
    # Retain a cache-free template before the optional cache refresh.
    template = copy.deepcopy(baseline)
    host = [F.rms_norm(torch.randn(batch, length, channels), (channels,)).bfloat16()
            for _ in range(3)]
    host.append((torch.randn(batch, length, channels) * .1).bfloat16())
    values = [v.to('neuron').requires_grad_() for v in host]
    if not with_ve: values[-1] = None
    angles = torch.outer(torch.arange(length).float(),
                         100000.0**(-torch.arange(0, 128, 2).float() / 128))
    cos, sin = [v.reshape(1, length, 1, 64).to('neuron') for v in (angles.cos(), angles.sin())]
    if cached: refresh(baseline)
    tag = f'{channels}:batch{batch}:length{length}:cache{cached}:ve{with_ve}'
    native = torch.compile(baseline, backend='neuron', fullgraph=True, dynamic=False)
    y = native(*values, cos, sin)
    assert type(y.grad_fn).__name__ == 'CompiledFunctionBackward'
    expected = y.detach().cpu()
    assert comparison(tag + ':native_repeat', native(*values, cos, sin), expected)['exact']
    records = {}
    modes = ('native_preprojected', 'nki_all', 'nki_q', 'nki_k', 'nki_v')
    if arm_set == 'pairs':
        modes = ('native_preprojected', 'nki_qk', 'nki_qv', 'nki_kv',
                 'nki_all_direct', 'nki_all')
    if include_fp32_arm: modes += ('nki_all_fp32v',)
    for mode in modes:
        # Fresh native initialization/weights remain the authority. Reset only
        # Python compile guards, not shared device/compiler caches.
        torch._dynamo.reset()
        candidate = copy.deepcopy(template)
        if cached: refresh(candidate)
        if mode in ('nki_all', 'nki_all_fp32v'):
            install(candidate, 'neuron', T, layout, value_fp32=(mode == 'nki_all_fp32v'))
        else: setup_partial(candidate, layout, mode)
        compiled = torch.compile(candidate, backend='neuron', fullgraph=True, dynamic=False)
        result = compiled(*values, cos, sin)
        assert type(result.grad_fn).__name__ == 'CompiledFunctionBackward'
        records[mode] = comparison(tag + ':' + mode, result, expected)

    if skip_materialized:
        emit('SERIAL_QKV_BOUNDARY_CASE', channels=channels, layout=layout, cached=cached,
             batch=batch, length=length, seed=seed, all_inputs_require_grad=True,
             include_fp32_arm=include_fp32_arm, arm_set=arm_set, gate_init=gate_init,
             materialized_controls=False, with_ve=with_ve, diagnostic_complete=True,
             observations=records, case_id=f'{cached}:{with_ve}',
             acceptance_bounds_unchanged=True, integration_promoted=False)
        return

    attn = baseline.transformer.h[0].attn

    def project(xq, xk, xv):
        return attn.c_q(xq), attn.c_k(xk), attn.c_v(xv)

    torch._dynamo.reset()
    projection = torch.compile(project, backend='neuron', fullgraph=True, dynamic=False)
    # Host round trips deliberately enforce stored BF16 intermediates.
    frozen = [x.detach().cpu().to('neuron').reshape(batch, length, channels // 128, 128).requires_grad_()
              for x in projection(*values[:3])]
    weight = attn.c_v.weight.detach().cpu().bfloat16().double()
    exact_v = (host[2].double() @ weight.T).float()
    unrounded_v = exact_v.to('neuron').reshape(frozen[2].shape).requires_grad_()

    def consume(xq, xk, xv, ve, c, s, q, k, v):
        return type(attn).forward(attn, xq, xk, xv, ve, c, s, None, qkv=(q, k, v))

    torch._dynamo.reset()
    consumer = torch.compile(consume, backend='neuron', fullgraph=True, dynamic=False)
    rounded = consumer(*values, cos, sin, *frozen)
    records['materialized_bf16'] = comparison(tag + ':materialized_bf16', rounded, expected)
    retained = consumer(*values, cos, sin, frozen[0], frozen[1], unrounded_v)
    records['materialized_fp32_v_hypothesis'] = comparison(tag + ':materialized_fp32_v_hypothesis', retained, expected)
    emit('SERIAL_QKV_BOUNDARY_CASE', channels=channels, layout=layout, cached=cached,
         batch=batch, length=length, seed=seed, all_inputs_require_grad=True,
         include_fp32_arm=include_fp32_arm,
         arm_set=arm_set, gate_init=gate_init, materialized_controls=True,
         with_ve=with_ve, diagnostic_complete=True, observations=records,
         case_id=f'{cached}:{with_ve}',
         acceptance_bounds_unchanged=True, integration_promoted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', required=True, choices=('tokens', 'channels'))
    parser.add_argument('--channels', required=True, type=int, choices=(256, 1024))
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--cache-mode', choices=('both', 'cached', 'uncached'), default='both')
    parser.add_argument('--ve-mode', choices=('both', 'on', 'off'), default='both')
    parser.add_argument('--include-fp32-arm', action='store_true')
    parser.add_argument('--arm-set', choices=('standard', 'pairs'), default='standard')
    parser.add_argument('--gate-init', choices=('random', 'zero'), default='random')
    parser.add_argument('--skip-materialized', action='store_true')
    parser.add_argument('--component-root', required=True)
    parser.add_argument('--component-label')
    parser.add_argument('--fp32-component-label')
    parser.add_argument('--component-commit', default='85addadd2ade1d0eb37fa9a5294b598bfe2ba9b4')
    args = parser.parse_args()
    assert 1 <= args.batch <= 32 and args.length in (128, 1024)
    assert args.batch * args.length % 256 == 0
    require_component(args.component_root, args.layout, args.component_label, args.component_commit)
    if args.include_fp32_arm:
        assert args.fp32_component_label
        require_component(args.component_root, args.layout, args.fp32_component_label,
                          args.component_commit, value_fp32=True)
    import torch_neuronx  # noqa: F401
    torch.set_num_threads(4)
    T.USE_WEIGHT_CACHE = T.USE_BF16_NORM_OUTPUT = T.USE_NKI_ROPE_NORM = T.COMPILE_SDPA_DIRECT = True
    T.USE_FUSED_QKV = T.USE_PARALLEL_BLOCK = T.USE_NGRAM_VE = T.USE_HEAD_GATE = T.USE_NKI_ATTN = False
    T.USE_VE = True
    assert T._HAS_NKI
    caches = (False, True) if args.cache_mode == 'both' else (args.cache_mode == 'cached',)
    ve_modes = (False, True) if args.ve_mode == 'both' else (args.ve_mode == 'on',)
    for cached in caches:
        for with_ve in ve_modes:
            run(args.channels, args.layout, cached, with_ve, args.batch, args.length,
                args.seed, args.include_fp32_arm, args.arm_set, args.gate_init,
                args.skip_materialized)
    emit('SERIAL_QKV_BOUNDARY_RESULT', layout=args.layout, channels=args.channels,
         batch=args.batch, length=args.length, all_inputs_require_grad=True,
         include_fp32_arm=args.include_fp32_arm, cases=len(caches) * len(ve_modes),
         arm_set=args.arm_set, gate_init=args.gate_init,
         materialized_controls=not args.skip_materialized,
         diagnostic_complete=True, acceptance_bounds_unchanged=True,
         integration_promoted=False, whole_model_or_quality_claim=False)


if __name__ == '__main__':
    main()
