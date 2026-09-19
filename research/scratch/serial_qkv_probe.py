#!/usr/bin/env python3
"""Independent product/gradient gates and charged QKV component timings.

No attention replacement or whole-model/quality promotion. Both native and
candidate are audited against FP64 products of identical BF16 inputs, with
the existing FP32 gamma_K plus one BF16 rounding error model. Negative controls
must reject halved nonzero products; tolerances do not depend on observations.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from serial_qkv_nki import build_kernel, make_autograd, native, native_value_fp32


def emit(marker, **record):
    print(marker + ' ' + json.dumps(record), flush=True)


def product_reference(left, right, output_dtype=torch.bfloat16):
    left, right = (v.detach().cpu().double() for v in (left, right))
    truth = left @ right
    contraction = left.shape[1]
    gamma = contraction * 2.0**-24 / (1 - contraction * 2.0**-24)
    accumulation = gamma * (left.abs() @ right.abs())
    unit = 2.0**(-24 if output_dtype == torch.float32 else -8)
    bound = accumulation + unit * (truth.abs() + accumulation) + 1e-30
    return truth, bound


def audit(name, actual, truth, bound):
    value = actual.detach().cpu().double()
    error = (value - truth).abs()
    failures = int((error > bound).sum())
    negative = int(((value * .5 - truth).abs() > bound).sum())
    finite = bool(torch.isfinite(value).all())
    record = dict(name=name, shape=list(value.shape), finite=finite,
                  bound_failures=failures, max_abs=float(error.max()),
                  relative_rms=float(error.norm() / truth.norm().clamp_min(1e-30)),
                  half_value_negative_control_failures=negative,
                  passed=finite and failures == 0)
    emit('SERIAL_QKV_PRODUCT_GATE', **record)
    assert record['passed'], record
    assert not bool(truth.abs().any()) or negative > 0, record


def exact(name, actual, expected):
    a, b = actual.detach().cpu(), expected.detach().cpu()
    passed = torch.equal(a, b) and bool(torch.isfinite(a).all())
    emit('SERIAL_QKV_EXACT_GATE', name=name, shape=list(a.shape), passed=passed)
    assert passed, name


def functions(token_major, value_fp32=False):
    from torch_neuronx import wrap_nki
    kernel = wrap_nki(build_kernel())

    def call(xq, xk, xv, wq, wk, wv):
        tile = 128 if token_major else min(512, xq.shape[0] // 2)
        return kernel[2](xq, xk, xv, wq, wk, wv, 2, tile, token_major, value_fp32)

    reference = native_value_fp32 if value_fp32 else native
    return (torch.compile(reference, backend='neuron', fullgraph=True, dynamic=False),
            torch.compile(make_autograd(call), backend='neuron', fullgraph=True, dynamic=False))


def data(tokens, channels, kind):
    generator = torch.Generator().manual_seed(149013 + tokens + channels)
    inputs = [torch.randn(tokens, channels, generator=generator).bfloat16() for _ in range(3)]
    weights = [torch.randn(channels, channels, generator=generator) / channels**.5 for _ in range(3)]
    gradients = [(torch.randn(tokens, channels, generator=generator) * .01).bfloat16() for _ in range(3)]
    if kind == 'zero_input': inputs[1].zero_()
    if kind == 'zero_weight': weights[2].zero_()
    if kind == 'zero_gradient': gradients[0].zero_()
    if kind == 'tiny_gradient': gradients = [g * 2.0**-14 for g in gradients]
    if kind == 'unrounded_v_gradient': gradients[2] = gradients[2].float() + 1e-5
    if kind == 'outliers':
        inputs[0][0, 0] = 64
        weights[1][0, 1] = -16
        gradients[2][-1, -1] = 4
    if kind == 'changed_weights':
        weights[0].mul_(-.75); weights[1].mul_(1.5); weights[2].add_(.03125)
    if kind == 'cancellation':
        inputs = [v.reshape(tokens, channels // 2, 2) for v in inputs]
        for value in inputs: value[:, :, 1] = value[:, :, 0]
        inputs = [v.reshape(tokens, channels) for v in inputs]
        for weight in weights: weight[:, 1::2] = -weight[:, 0::2]
    return inputs, weights, gradients


def leaves(values):
    return tuple(v.to('neuron').detach().clone().requires_grad_() for v in values)


def run_case(shape, kind, compiled, value_fp32=False):
    tokens, channels = shape
    xs, ws, gs = data(tokens, channels, kind)
    original = xs + ws
    operands = [leaves(original) for _ in compiled]
    device_gradients = tuple(g.to('neuron') for g in gs)
    results = []
    for arm, call, args in zip(('native', 'candidate'), compiled, operands):
        outputs = call(*args)
        assert len(outputs) == 3
        dtypes = (torch.bfloat16, torch.bfloat16, torch.float32 if value_fp32 else torch.bfloat16)
        assert all(y.dtype == dtype and y.shape == xs[0].shape and y.is_contiguous()
                   for y, dtype in zip(outputs, dtypes))
        assert all(type(y.grad_fn).__name__ == 'CompiledFunctionBackward' for y in outputs)
        repeated = call(*args)
        for index, (output, repeat) in enumerate(zip(outputs, repeated)):
            exact(f'{kind}:{arm}:repeat{index}', repeat, output)
        torch.autograd.backward(outputs, device_gradients)
        assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in args[3:])
        results.append((tuple(y.detach().cpu() for y in outputs), tuple(v.grad.detach().cpu() for v in args)))
        for index, (value, initial) in enumerate(zip(args, original)):
            exact(f'{kind}:{arm}:nonmutation{index}', value, initial)
    for index in range(3):
        cached = ws[index].bfloat16()
        for field, left, right, values in (
            ('output', xs[index], cached.T, [v[0][index] for v in results]),
            ('input_gradient', gs[index].bfloat16(), cached, [v[1][index] for v in results]),
            ('master_gradient', gs[index].bfloat16().T, xs[index], [v[1][3 + index] for v in results])):
            dtype = torch.float32 if value_fp32 and index == 2 and field == 'output' else torch.bfloat16
            truth, bound = product_reference(left, right, dtype)
            for arm, actual in zip(('native', 'candidate'), values):
                audit(f'{kind}:{index}:{field}:{arm}', actual, truth, bound)
            emit('SERIAL_QKV_NATIVE_COMPARISON', case=kind, stream=index, field=field,
                 exact=torch.equal(values[0], values[1]),
                 different=int((values[0] != values[1]).sum()))
    # Changing only a suffix of the V stream cannot affect Q/K or earlier V
    # rows. Deliberately distinct stream inputs also gate mistaken input sharing.
    with torch.no_grad():
        changed = [v.detach().clone() for v in operands[1]]
        changed[2][-128:].add_(.5)
        changed_outputs = compiled[1](*changed)
        exact(f'{kind}:independent_q', changed_outputs[0], results[1][0][0])
        exact(f'{kind}:independent_k', changed_outputs[1], results[1][0][1])
        exact(f'{kind}:earlier_v_rows', changed_outputs[2][:-128], results[1][0][2][:-128])
    emit('SERIAL_QKV_CASE', kind=kind, tokens=tokens, channels=channels,
         value_fp32=value_fp32, backward_gemm_operands_bf16=True,
         independent_fp64_forward_and_six_gradients=True,
         distinct_inputs=True, contiguous_outputs=True, fp32_master_gradients=True,
         repeat_and_nonmutation=True, stream_and_row_independence=True, passed=True)


def lifecycle(compiled, value_fp32=False):
    """Refresh actual cast caches after an update; accumulate two microbatches."""
    xs, ws, gs = data(512, 256, 'random')
    master_sets = [leaves(ws) for _ in compiled]
    gdev = tuple(g.to('neuron') for g in gs)
    for step in range(2):
        cache_sets = [tuple(w.bfloat16() for w in masters) for masters in master_sets]
        for masters in master_sets:
            for w in masters: w.grad = None
        truths = [torch.zeros_like(w, dtype=torch.float64) for w in ws]
        bounds = [torch.zeros_like(w, dtype=torch.float64) for w in ws]
        magnitudes = [torch.zeros_like(w, dtype=torch.float64) for w in ws]
        for micro in range(2):
            inputs = [x + micro * .0625 for x in xs]
            for arm, call, caches in zip(('native', 'candidate'), compiled, cache_sets):
                dev_inputs = leaves(inputs)
                outputs = call(*dev_inputs, *caches)
                for index, output in enumerate(outputs):
                    dtype = torch.float32 if value_fp32 and index == 2 else torch.bfloat16
                    truth, bound = product_reference(inputs[index], caches[index].detach().cpu().T, dtype)
                    audit(f'lifecycle:{step}:{micro}:{arm}:output{index}', output, truth, bound)
                torch.autograd.backward(outputs, gdev)
            for index in range(3):
                truth, bound = product_reference(gs[index].T, inputs[index])
                truths[index] += truth; bounds[index] += bound
                magnitudes[index] += truth.abs() + bound
        for arm, masters in zip(('native', 'candidate'), master_sets):
            for index, w in enumerate(masters):
                assert w.grad.dtype == torch.float32
                audit(f'lifecycle:{step}:{arm}:accumulated_master{index}', w.grad,
                      truths[index], bounds[index] + 2.0**-23 * magnitudes[index])
        with torch.no_grad():
            for index, reference in enumerate(master_sets[0]):
                reference.add_(reference.grad, alpha=-.02)
                master_sets[1][index].copy_(reference)
        for old, new in zip(cache_sets[0], master_sets[0]):
            assert not torch.equal(old.cpu(), new.bfloat16().cpu()), 'cache fixture must actually change'
    emit('SERIAL_QKV_LIFECYCLE', cached_and_uncached=True, master_updates=2,
         value_fp32=value_fp32,
         microbatches_per_update=2, fp32_accumulation=True, passed=True)


def benchmark(compiled, steps, value_fp32=False):
    xs, ws, gs = data(32768, 1024, 'random')
    operands = [leaves(xs + ws) for _ in compiled]
    gdev = tuple(g.to('neuron') for g in gs)
    calls = []
    for function, args in zip(compiled, operands):
        def call(function=function, args=args):
            for value in args: value.grad = None
            torch.autograd.backward(function(*args), gdev)
        calls.append(call)
    for call in calls:
        for _ in range(3): call()
        torch.neuron.synchronize()
    samples = {name: [] for name in ('native', 'candidate')}
    for block in range(5):
        for index in ((0, 1) if block % 2 == 0 else (1, 0)):
            torch.neuron.synchronize()
            begin = time.perf_counter()
            for _ in range(steps): calls[index]()
            torch.neuron.synchronize()
            samples[('native', 'candidate')[index]].append((time.perf_counter() - begin) * 1000 / steps)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    emit('SERIAL_QKV_COMPONENT_TIMING', tokens=32768, channels=1024, steps_per_block=steps,
         value_fp32=value_fp32, native_control_returns_fp32_v=value_fp32,
         samples_ms=samples, median_ms=medians,
         improvement_pct=100 * (1 - medians['candidate'] / medians['native']),
         includes_three_forward_six_backward_products=True,
         includes_fp32_master_cache_cast_and_accumulation=True,
         all_layout_and_dispatch_costs_timed=True, whole_model_or_quality_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', choices=('channels', 'tokens'), required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--value-fp32', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.steps <= 50
    import torch_neuronx  # noqa: F401
    torch.set_num_threads(4)
    compiled = functions(args.layout == 'tokens', args.value_fp32)
    kinds = ('random', 'zero_input', 'zero_weight', 'zero_gradient',
             'tiny_gradient', 'outliers', 'changed_weights', 'cancellation')
    if args.value_fp32: kinds += ('unrounded_v_gradient',)
    for kind in kinds: run_case((512, 256), kind, compiled, args.value_fp32)
    lifecycle(compiled, args.value_fp32)
    torch._dynamo.reset()
    emit('SERIAL_QKV_PROCESS_GUARD_RESET', completed_width=256, next_width=1024,
         shared_device_cache_purged=False)
    compiled = functions(args.layout == 'tokens', args.value_fp32)
    run_case((32768, 1024), 'real_random', compiled, args.value_fp32)
    benchmark(compiled, args.steps, args.value_fp32)
    emit('SERIAL_QKV_RESULT', layout=args.layout, cases=len(kinds) + 1, passed=True,
         value_fp32=args.value_fp32, fp32_value_output_bounds=args.value_fp32,
         real_tokens=32768, real_channels=1024, independent_fp64_gates=True,
         lifecycle_passed=True, whole_model_or_quality_claim=False)


if __name__ == '__main__':
    main()
