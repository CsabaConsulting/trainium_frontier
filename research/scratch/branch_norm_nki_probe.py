#!/usr/bin/env python3
"""Analytic, compiled-autograd, independence and timing gates for branch RMSNorm."""
import argparse
import json

import torch

from branch_norm_nki import EPS, build_kernels, make_function, reference
from mlp_fused_bf16_probe import timing
from mlp_separate_diagnose import autograd_nodes


def numerical(name, output, gradient, x, g):
    xd, gd = x.double(), g.double()
    inverse = (xd.square().mean(-1, keepdim=True) + EPS).rsqrt()
    yd = xd * inverse
    expected = inverse * (gd - xd * inverse.square() * (gd * xd).mean(-1, keepdim=True))
    rounded = yd.bfloat16()
    up = torch.nextafter(rounded, torch.full_like(rounded, float("inf")))
    down = torch.nextafter(rounded, torch.full_like(rounded, -float("inf")))
    ulp = torch.maximum((up.float() - rounded.float()).abs(), (down.float() - rounded.float()).abs())
    forward_failures = int(((output.float() - rounded.float()).abs() > ulp).sum())
    error = (gradient.double() - expected).abs()
    relative = float(error.norm() / expected.norm().clamp_min(1e-30))
    u = 2.0 ** -24
    gamma = (x.shape[-1] + 8) * u / (1 - (x.shape[-1] + 8) * u)
    envelope = (gd * inverse).abs() + xd.abs() * inverse.pow(3) * (gd * xd).abs().mean(-1, keepdim=True)
    backward_failures = int((error > 4 * gamma * envelope + 1e-30).sum())
    finite = bool(torch.isfinite(output).all() and torch.isfinite(gradient).all())
    record = dict(name=name, finite=finite, forward_one_ulp_failures=forward_failures,
                  backward_bound_failures=backward_failures, backward_relative_rms=relative,
                  backward_max_abs=float(error.max()),
                  passed=finite and not forward_failures and not backward_failures and relative < 3e-6)
    print("BRANCH_NORM_NUMERICAL " + json.dumps(record), flush=True)
    assert record["passed"], record


def diagnose_inverse(packed_inverse=False):
    """Read the actual saved statistic; do not infer accuracy from its dtype."""
    from torch_neuronx import wrap_nki
    forward, _ = build_kernels()
    kernel = wrap_nki(forward)
    generator = torch.Generator().manual_seed(391)
    base = torch.randn(256, 128, generator=generator)
    for scale in (0.0, 1e-6, 1.0, 1e4):
        x = base * scale
        expected = (x.double().square().mean(-1, keepdim=True) + EPS).rsqrt()
        device_x = x.to("neuron")
        records = []
        for mode, name in ((0, "scalar_raw"), (1, "scalar_newton"), (2, "gpsimd")):
            _, inverse = kernel[2](device_x, 2, mode, packed_inverse)
            actual = inverse.cpu().double()
            if packed_inverse:
                actual = actual.reshape(128, 2, -1).permute(2, 1, 0).reshape(-1, 1)
            relative = ((actual - expected) / expected).abs()
            records.append(dict(mode=name, actual_first=float(actual[0, 0]),
                                expected_first=float(expected[0, 0]),
                                max_relative=float(relative.max()),
                                relative_rms=float((actual - expected).norm() / expected.norm())))
            if mode:
                assert bool(torch.isfinite(actual).all()) and float(relative.max()) < 3e-6
        print("BRANCH_NORM_INVERSE_DIAGNOSIS " + json.dumps(dict(
            shape=list(x.shape), scale=scale, packed_inverse=packed_inverse, results=records)), flush=True)


def run(rows, channels, steps, inverse_mode, packed_inverse=False):
    torch.manual_seed(871 + rows + channels)
    base = torch.randn(rows, channels)
    grad = torch.randn(rows, channels).bfloat16()
    native = torch.compile(reference, backend="neuron", fullgraph=True, dynamic=False)
    candidate = torch.compile(make_function(inverse_mode, packed_inverse), backend="neuron",
                              fullgraph=True, dynamic=False)
    gd = grad.to("neuron")
    for scale in (0.0, 1e-6, 1e-3, 1.0, 1e4):
        xx = base * scale
        outputs = {}
        for name, function in (("native", native), ("nki", candidate)):
            leaf = xx.to("neuron").requires_grad_()
            output = function(leaf)
            nodes = autograd_nodes(output)
            assert nodes.get("CompiledFunctionBackward", 0) >= 1, nodes
            output.backward(gd)
            assert output.dtype == torch.bfloat16 and leaf.grad.dtype == torch.float32
            cpu_output, cpu_gradient = output.detach().cpu(), leaf.grad.detach().cpu()
            numerical(f"{name}:{rows}:{channels}:{scale}", cpu_output, cpu_gradient, xx, grad)
            outputs[name] = cpu_output
        print("BRANCH_NORM_NATIVE_COMPARISON " + json.dumps(dict(
            rows=rows, channels=channels, scale=scale,
            output_max_abs=float((outputs["nki"].float() - outputs["native"].float()).abs().max()))), flush=True)
    # Changes elsewhere must not affect any untouched token, or later repeats.
    original = base.to("neuron").requires_grad_()
    first = candidate(original).detach().cpu()
    changed = base.clone()
    changed[rows // 2:] += 19.0
    altered = candidate(changed.to("neuron").requires_grad_()).detach().cpu()
    torch.testing.assert_close(first[:rows // 2], altered[:rows // 2], rtol=0, atol=0)
    repeated = candidate(original).detach().cpu()
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    # The real model supplies upstream BF16 gradients. Both arms include their
    # required saved statistics and FP32 gradient production in this benchmark.
    leaves = {name: base.to("neuron").requires_grad_() for name in ("native", "nki")}
    def call(name, function):
        leaves[name].grad = None
        y = function(leaves[name])
        y.backward(gd)

    measured = timing({"native_forward_backward": lambda: call("native", native),
                       "nki_forward_backward": lambda: call("nki", candidate)},
                      steps, torch.neuron.synchronize)
    print("BRANCH_NORM_BENCH " + json.dumps(dict(
        rows=rows, channels=channels, inverse_mode=inverse_mode, packed_inverse=packed_inverse,
        **measured, same_precision_policy=True,
        untouched_tokens_and_repeated_calls_exact=True, whole_model=False)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--diagnose-inverse", action="store_true")
    ap.add_argument("--packed-inverse", action="store_true",
                    help="Batch saved token statistics into one DMA per core in each direction.")
    ap.add_argument("--inverse-mode", type=int, choices=(1, 2), default=2,
                    help="1=Scalar rsqrt plus FP32 Newton step; 2=higher-precision GpSIMD rsqrt")
    args = ap.parse_args()
    import torch_neuronx  # noqa:F401
    torch.set_num_threads(4)
    if args.diagnose_inverse:
        diagnose_inverse(args.packed_inverse)
    run(256, 128, args.steps, args.inverse_mode, args.packed_inverse)
    run(256, 1024, args.steps, args.inverse_mode, args.packed_inverse)
    if args.real_shape:
        run(8192, 1024, args.steps, args.inverse_mode, args.packed_inverse)
    print("PASS: branch RMS normalization numerical and independence gates", flush=True)


if __name__ == "__main__":
    main()
