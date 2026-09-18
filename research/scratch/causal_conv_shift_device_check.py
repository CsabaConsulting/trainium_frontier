#!/usr/bin/env python3
"""Native arithmetic/causality gate for the two-tap convolution variants.

The default normalized input is FP32; BF16 normalization changes that input to
BF16 while leaving the residual stream FP32. Check either input precision against
an independent FP64 convolution and arbitrary-gradient reference. Report
the old compiled convolution too, but do not use it as an exact arithmetic oracle.
This script makes no latency claim; all timings belong in the full phase harness.
"""
import argparse
import json
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from causal_conv_shift import shift_causal_conv


def normal(x, w):
    return F.conv1d(F.pad(x.transpose(1, 2), (w.shape[-1] - 1, 0)),
                    w.to(x.dtype), groups=x.shape[-1]).transpose(1, 2)


def shifted(x, w):
    return shift_causal_conv(SimpleNamespace(weight=w, kernel_size=w.shape[-1]), x)


def cpu_reference(x, w, grad, absolute=False):
    xx = (x.abs() if absolute else x).double().requires_grad_()
    # The production forward casts the master weights to the input precision.
    rounded_w = w.to(x.dtype)
    ww = (rounded_w.abs() if absolute else rounded_w).double().requires_grad_()
    gg = (grad.abs() if absolute else grad).double()
    y = normal(xx, ww)
    dx, dw = torch.autograd.grad(y, (xx, ww), gg)
    return tuple(t.detach() for t in (y, dx, dw))


def errors(actual, expected, absolute_sum, terms, arithmetic_dtype):
    delta = (actual.double() - expected).abs()
    # Allow one rounding for each product and addition. The bound remains useful
    # near cancellation; the independent RMS gate rejects a loosely bounded loss.
    ku = 2 * terms * 2.0 ** -24
    bound = ku / (1 - ku) * absolute_sum + torch.finfo(torch.float32).tiny
    if arithmetic_dtype == torch.bfloat16:
        # Each output and the cast-weight gradient round once to BF16. Keep the
        # reference in FP64, and include this *specified* final rounding in the
        # absolute error budget. Products and partial reductions remain FP32.
        bound = bound + (2.0 ** -8) * expected.abs() + torch.finfo(torch.bfloat16).tiny
    rounded_reference = expected.to(arithmetic_dtype).to(actual.dtype)
    return dict(finite=bool(torch.isfinite(actual).all()),
                max_abs=float(delta.max()),
                relative_rms=float(delta.norm() / expected.norm().clamp_min(1e-30)),
                bound_failures=int((delta > bound).sum()),
                max_error_over_bound=float((delta / bound).max()),
                rounded_reference_max_abs=float((actual - rounded_reference).abs().max()),
                terms=terms)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--large", action="store_true")
    ap.add_argument("--kernel", choices=["shift", "nki", "train-nki"], default="shift")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--zero-weights", action="store_true",
                    help="Include the zero-initialized branch used by the first training forward.")
    ap.add_argument("--eager", action="store_true",
                    help="Isolate kernel execution from whole-function graph compilation.")
    args = ap.parse_args()
    import torch_neuronx  # noqa: F401
    torch.set_num_threads(2)
    torch.manual_seed(481)
    arithmetic_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    cases = [(2, 128, 128, 1), (2, 128, 128, 2), (2, 128, 128, 4)]
    variant = shifted
    if args.kernel in ("nki", "train-nki"):
        if args.kernel == "nki":
            from causal_conv_nki2 import Conv2NKI
            variant = lambda x, w: Conv2NKI.apply(x, w)
        else:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            import train as T
            T.USE_NKI_LOCAL_CONV = True
            assert T._HAS_NKI, "integrated native kernels were not imported"
            # Compile the production autograd operation directly. Constructing
            # SimpleNamespace inside a fullgraph lambda is unsupported by this
            # Dynamo version and fails before reaching the operation. The real
            # CausalConv module/shape guard is exercised in the full-model probe.
            variant = lambda x, w: T._Conv2NKI.apply(x, w)
        cases = [(1, 128, 128, 2), (2, 128, 128, 2), (2, 256, 256, 2)]
    if args.large:
        cases.append((4, 2048, 1024, 2))
    compiled = {"normal": normal, args.kernel: variant}
    if not args.eager:
        compiled = {name: torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)
                    for name, fn in compiled.items()}
    cases = [(shape, False) for shape in cases]
    if args.zero_weights:
        cases += [(shape, True) for shape, _ in cases]
    for (batch, length, width, taps), zero_weights in cases:
        x = torch.randn(batch, length, width, dtype=torch.float32).to(arithmetic_dtype)
        w = torch.randn(width, 1, taps, dtype=torch.float32) * 0.1
        if zero_weights:
            w.zero_()
        grad = torch.randn_like(x)
        expected = cpu_reference(x, w, grad)
        absolute = cpu_reference(x, w, grad, absolute=True)
        results = {}
        for name, fn in compiled.items():
            print(json.dumps(dict(stage="forward", implementation=name, eager=args.eager,
                                  dtype=args.dtype, zero_weights=zero_weights,
                                  shape=[batch, length, width, taps])), flush=True)
            xx = x.to("neuron").requires_grad_()
            ww = w.to("neuron").requires_grad_()
            y = fn(xx, ww)
            # Synchronize at the logical boundary so an asynchronous compiler
            # failure is attributed to forward or backward rather than a later CPU copy.
            y_cpu = y.detach().cpu()
            print(json.dumps(dict(stage="backward", implementation=name, eager=args.eager,
                                  dtype=args.dtype, zero_weights=zero_weights,
                                  shape=[batch, length, width, taps])), flush=True)
            dx, dw = torch.autograd.grad(y, (xx, ww), grad.to("neuron"))
            actual = (y_cpu, dx.detach().cpu(), dw.detach().cpu())
            report = [errors(a, e, ab, n, arithmetic_dtype) for a, e, ab, n in
                      zip(actual, expected, absolute, [taps, taps, batch * length])]
            results[name] = report
            print(json.dumps(dict(check="conv_native_arithmetic", implementation=name,
                                  dtype=args.dtype, zero_weights=zero_weights,
                                  shape=[batch, length, width, taps],
                                  forward_dx_dw=report)), flush=True)
            if name == args.kernel:
                assert y.dtype == arithmetic_dtype and dx.dtype == arithmetic_dtype
                assert dw.dtype == w.dtype
                rms_limit = 2e-5 if arithmetic_dtype == torch.float32 else 0.004
                assert all(r["finite"] and r["bound_failures"] == 0 and
                           r["relative_rms"] < rms_limit for r in report), report
                cut = length // 2
                future = x.clone()
                future[:, cut:] += 100
                changed = fn(future.to("neuron"), ww.detach()).detach().cpu()
                torch.testing.assert_close(changed[:, :cut], actual[0][:, :cut], rtol=0, atol=0)
                other_batch = x.clone()
                other_batch[0] += 100
                changed = fn(other_batch.to("neuron"), ww.detach()).detach().cpu()
                torch.testing.assert_close(changed[1:], actual[0][1:], rtol=0, atol=0)
                print(json.dumps(dict(check="conv_native_causality", shape=[batch, length, width, taps],
                                      dtype=args.dtype, zero_weights=zero_weights,
                                      future_and_cross_batch_exact=True)), flush=True)
    print(f"PASS: native {args.dtype} {args.kernel} convolution arithmetic and causality; no speed or BPB claim", flush=True)


if __name__ == "__main__":
    main()
