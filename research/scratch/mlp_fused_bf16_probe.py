#!/usr/bin/env python3
"""Stagewise numerical and training-cost gates for the fused BF16 MLP.

CPU checks use ordinary autograd as an independent derivative control. Device
checks additionally audit each fused GEMM against FP64 sums of BF16 inputs,
then compare compiled input/weight gradients against the native implementation.
The forward benchmark includes training's saved ReLU tensor, FP32 weight reads
and conversion. The forward/backward benchmark includes gradient production.
No optimizer, whole-model or BPB claim follows from these component timings.
"""
import argparse
import json
import statistics
import time

import torch

from mlp_fused_bf16 import build_kernel, make_autograd, reference_forward


def emit(kind, value):
    print(kind + ' ' + json.dumps(value), flush=True)


def gemm_gate(name, actual, left, right, relu=False):
    left, right, actual = (x.detach().cpu().double() for x in (left, right, actual))
    exact = left @ right
    l1_products = left.abs() @ right.abs()
    k = left.shape[1]
    gamma = k * 2.0**-24 / (1 - k * 2.0**-24)
    accumulation = gamma * l1_products
    # FP32 accumulation and one BF16 output rounding. ReLU is1-Lipschitz.
    bound = accumulation + 2.0**-8 * (exact.abs() + accumulation) + 1e-30
    if relu:
        exact = exact.clamp_min(0)
    error = (actual - exact).abs()
    record = dict(name=name, shape=list(actual.shape), finite=bool(torch.isfinite(actual).all()),
                  bound_failures=int((error > bound).sum()), max_abs=float(error.max()),
                  relative_rms=float(error.square().mean().sqrt() / exact.square().mean().sqrt().clamp_min(1e-30)))
    record['passed'] = record['finite'] and record['bound_failures'] == 0
    emit('FUSED_MLP_STAGE_GATE', record)
    assert record['passed'], record


def compare(name, actual, expected):
    actual, expected = (x.detach().cpu().double() for x in (actual, expected))
    error = (actual - expected).abs()
    rms = expected.square().mean().sqrt()
    # Near-zero values need an absolute term; also bound total and maximum
    # drift. This tolerates BF16 operation-order rounding, not indexing errors.
    bound = 0.03 * expected.abs() + 0.008 * rms + 1e-30
    relative = float(error.square().mean().sqrt() / rms.clamp_min(1e-30))
    record = dict(name=name, shape=list(actual.shape), finite=bool(torch.isfinite(actual).all()),
                  bound_failures=int((error > bound).sum()), max_abs=float(error.max()),
                  reference_rms=float(rms), relative_rms=relative,
                  exact=bool(torch.equal(actual, expected)))
    record['passed'] = record['finite'] and record['bound_failures'] == 0 and relative < 0.004
    emit('FUSED_MLP_NATIVE_GATE', record)
    assert record['passed'], record


def timing(arms, steps, sync):
    for fn in arms.values():
        for _ in range(3):
            fn()
        sync()
    samples = {name: [] for name in arms}
    for repeat in range(3):
        order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
        for name in order:
            sync()
            start = time.perf_counter()
            for _ in range(steps):
                arms[name]()
            sync()
            samples[name].append(1000 * (time.perf_counter() - start) / steps)
    return dict(samples_ms=samples, median_ms={name: statistics.median(v) for name, v in samples.items()})


def run_case(device, shape, kind, kernel, steps, contiguous_weights=False, tiled_io=False,
             cached_bf16_weights=False, transposed_relu=False):
    tokens, channels, hidden = shape
    tile = min(512, tokens // 2)
    torch.manual_seed(941 + tokens + channels + (kind == 'changed'))
    x = torch.randn(tokens, channels).bfloat16()
    up = torch.randn(hidden, channels) / channels**0.5
    down = torch.randn(channels, hidden) / hidden**0.5
    if kind == 'zero':
        x.zero_()
    if kind == 'changed':
        up *= 0.5
        down *= 1.5
    if cached_bf16_weights:
        # Model training already refreshes BF16 Linear caches once per step.
        # This arm checks the kernel with that input dtype, with normal weight
        # layouts. It does not include master parameter updates/cache refresh.
        up, down = up.bfloat16(), down.bfloat16()
    g = (torch.randn(tokens, channels) * 0.01).bfloat16()
    cpu_inputs = (x, up, down)
    if device == 'neuron':
        import torch_neuronx
        wrapped = torch_neuronx.wrap_nki(kernel)

        def kernel_call(xx, uu, dd):
            return wrapped[2](xx, uu, dd, 2, tile, contiguous_weights, tiled_io, transposed_relu)

        torch._dynamo.reset()
        fused_forward = torch.compile(kernel_call, backend='neuron', fullgraph=True, dynamic=False)
        native_forward = torch.compile(reference_forward, backend='neuron', fullgraph=True, dynamic=False)
    else:
        native_forward = reference_forward
        def kernel_call(xx, uu, dd):
            output, saved = reference_forward(xx, uu, dd)
            return output, saved.T if transposed_relu else saved
        fused_forward = kernel_call
    inputs = tuple(t.to(device) for t in cpu_inputs)
    if device == 'neuron':
        eager_y, eager_saved = kernel[2](*inputs, 2, tile, contiguous_weights, tiled_io,
                                         transposed_relu)
        eager_saved_cpu = eager_saved.cpu()
        eager = (eager_y, eager_saved_cpu.T if transposed_relu else eager_saved_cpu)
        gemm_gate(f'eager:up:{shape}:{kind}', eager[1], x, up.bfloat16().T, relu=True)
        gemm_gate(f'eager:down:{shape}:{kind}', eager[0], eager[1].cpu().square(), down.bfloat16().T)
    output, relu = fused_forward(*inputs)
    expected_output, expected_relu = native_forward(*inputs)
    # Canonicalize ONLY the detached CPU comparison, never the timed forward.
    relu_cpu = relu.detach().cpu()
    if transposed_relu:
        relu_cpu = relu_cpu.T
    if device == 'neuron':
        gemm_gate(f'compiled:up:{shape}:{kind}', relu_cpu, x, up.bfloat16().T, relu=True)
        gemm_gate(f'compiled:down:{shape}:{kind}', output, relu_cpu.square(), down.bfloat16().T)
    compare(f'forward:{shape}:{kind}', output, expected_output)
    compare(f'saved_relu:{shape}:{kind}', relu_cpu, expected_relu)
    fused = make_autograd(kernel_call, transposed_relu)

    def native(xx, uu, dd):
        return reference_forward(xx, uu, dd)[0]

    if device == 'neuron':
        fused = torch.compile(fused, backend='neuron', fullgraph=True, dynamic=False)
        native = torch.compile(native, backend='neuron', fullgraph=True, dynamic=False)
    controls = tuple(t.to(device).detach().clone().requires_grad_() for t in cpu_inputs)
    candidates = tuple(t.to(device).detach().clone().requires_grad_() for t in cpu_inputs)
    gd = g.to(device)

    def train_call(fn, values):
        for value in values:
            value.grad = None
        y = fn(*values)
        y.backward(gd)
        return y

    expected_y = train_call(native, controls)
    actual_y = train_call(fused, candidates)
    compare(f'train_forward:{shape}:{kind}', actual_y, expected_y)
    for key, candidate, control in zip(('input', 'up_weight', 'down_weight'), candidates, controls):
        assert candidate.grad.dtype == control.grad.dtype == candidate.dtype
        compare(f'gradient:{key}:{shape}:{kind}', candidate.grad, control.grad)
    emit('FUSED_MLP_CASE', dict(shape=shape, kind=kind, device=device,
                              contiguous_weights=contiguous_weights, tiled_io=tiled_io,
                              cached_bf16_weights=cached_bf16_weights,
                              transposed_relu=transposed_relu,
                              saved_layout_consumed_directly_in_backward=True,
                              token_tile=tile, all_gates_passed=True,
                              saved_activation_bytes=2*tokens*hidden,
                              weight_input_bytes=2*up.element_size()*channels*hidden,
                              master_parameter_updates_and_cache_refresh_included=False))
    if device == 'neuron' and kind == 'random':
        arms = dict(native_forward_saved_relu=lambda: native_forward(*inputs),
                    fused_forward_saved_relu=lambda: fused_forward(*inputs),
                    native_forward_backward=lambda: train_call(native, controls),
                    fused_forward_native_backward=lambda: train_call(fused, candidates))
        emit('FUSED_MLP_BENCH', dict(shape=shape, **timing(arms, steps, torch.neuron.synchronize),
                                   contiguous_weights=contiguous_weights, tiled_io=tiled_io,
                                   cached_bf16_weights=cached_bf16_weights,
                                   transposed_relu=transposed_relu,
                                   saved_layout_consumed_directly_in_backward=True,
                                   weight_reads_casts_layout_and_relu_save_included=True,
                                   includes_optimizer_or_whole_model=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=('cpu', 'neuron'), default='neuron')
    ap.add_argument('--real-shape', action='store_true')
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--contiguous-weights', action='store_true',
                    help='Fix scalar weight DMA packets via contiguous reads and on-chip transpose.')
    ap.add_argument('--tiled-io', action='store_true',
                    help='Contiguous activation transfers with explicit128x128 on-chip transposes.')
    ap.add_argument('--cached-bf16-weights', action='store_true',
                    help='Compare BF16 cached weight inputs; master updates/refresh excluded from both arms.')
    ap.add_argument('--transposed-relu', action='store_true',
                    help='Save ReLU in HxT kernel layout and charge its view/layout use in backward.')
    args = ap.parse_args()
    torch.set_num_threads(4)
    kernel = None
    if args.device == 'neuron':
        import torch_neuronx  # noqa:F401
        kernel = build_kernel()
    for kind in ('random', 'changed', 'zero'):
        run_case(args.device, (256, 256, 512), kind, kernel, args.steps,
                 args.contiguous_weights, args.tiled_io, args.cached_bf16_weights, args.transposed_relu)
    if args.real_shape:
        run_case(args.device, (8192, 1024, 4096), 'random', kernel, args.steps,
                 args.contiguous_weights, args.tiled_io, args.cached_bf16_weights, args.transposed_relu)
    print('PASS: fused BF16 MLP component gates; no full-model claim', flush=True)


if __name__ == '__main__':
    main()
