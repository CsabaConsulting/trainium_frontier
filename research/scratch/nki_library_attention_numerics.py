#!/usr/bin/env python3
"""Diagnose the original FP64 attention screen; never promote a candidate.

The source167 bundled control failed an empirical per-coordinate envelope.
Retain that exact envelope and count/report EVERY failure, rather than silently
loosening it. Separate staged-rounding estimates explain potential cancellation
and BF16 intermediate effects; they are diagnostics, not acceptance gates.
The original new-versus-bundled campaign gate and exact repeats still assert.
No timing, full-model experiment or quality promotion is performed here.
"""
import torch

from mlp_fused_bf16_probe import compare
from nki_library_attention_probe import cpu_reference, forward_backward, make_forward
from nki_library_probe_common import NAMESPACE, emit, exact, verify_sources


FIELDS = ("output", "dq", "dk", "dv")


def rounding_estimate(values, grad, scale):
    """Independent dense algebra with documented BF16 boundaries, no NKI code.

    FP64 dot products isolate BF16 storage effects; this intentionally does not
    emulate the kernel's tiling, FP32 summation order or approximate exp unit.
    Thus closeness supports a diagnosis, but does not prove equivalence.
    """
    q, k, v = (value.double() for value in values)
    g = grad.double()
    score = (q @ k.transpose(-2, -1)) * scale
    mask = torch.ones(score.shape[-2:], dtype=torch.bool).tril()
    score = score.masked_fill(~mask, -torch.inf)
    exp = (score - score.amax(dim=-1, keepdim=True)).exp()
    normalizer = exp.sum(dim=-1, keepdim=True)
    output = ((exp.bfloat16().double() @ v) / normalizer).bfloat16().double()
    probability = (exp / normalizer).bfloat16().double()
    delta = (g * output).sum(dim=-1, keepdim=True)
    ds = (probability * (g @ v.transpose(-2, -1) - delta)).bfloat16().double()
    dq = ((ds @ k) * scale).bfloat16().double()
    dk = ((ds.transpose(-2, -1) @ q) * scale).bfloat16().double()
    dv = (probability.transpose(-2, -1) @ g).bfloat16().double()
    absolute_pv_sum = (exp / normalizer) @ v.abs()
    return (output, dq, dk, dv), absolute_pv_sum


def report(name, actual, reference, staged, absolute_pv_sum=None):
    actual, reference, staged = (x.detach().cpu().double() for x in (actual, reference, staged))
    error = (actual - reference).abs()
    rms = reference.square().mean().sqrt()
    # Byte-for-byte the same formula as source166/167 attention_cpu_check.
    bound = .03 * reference.abs() + .008 * rms + 1e-30
    fail = error > bound
    relative = float(error.norm() / reference.norm().clamp_min(1e-30))
    ratios = error / bound.clamp_min(1e-300)
    flat = ratios.flatten().topk(min(8, ratios.numel())).indices
    worst = []
    for pos in flat:
        idx = torch.unravel_index(pos, actual.shape)
        idx = tuple(int(v) for v in idx)
        row = dict(index=idx, actual=float(actual[idx]), fp64=float(reference[idx]),
                   staged=float(staged[idx]), original_bound=float(bound[idx]),
                   error_over_bound=float(ratios[idx]))
        if absolute_pv_sum is not None:
            row["absolute_pv_sum"] = float(absolute_pv_sum[idx])
            row["bf16_pv_roundoff_scale_not_gate"] = float(2**-8 * absolute_pv_sum[idx])
        worst.append(row)
    original_passed = bool(torch.isfinite(actual).all()) and not bool(fail.any()) and relative < .01
    emit("NKILIB_ATTENTION_NUMERICS", dict(
        name=name, shape=list(actual.shape), original_bound_failures=int(fail.sum()),
        original_relative_rms=relative, original_passed=original_passed,
        staged_relative_rms=float((actual - staged).norm() / reference.norm().clamp_min(1e-30)),
        staged_exact_fraction=float((actual == staged).double().mean()),
        worst=worst, diagnostic_only=True, acceptance_threshold_changed=False))
    return int(fail.sum()), original_passed


def main():
    torch.set_num_threads(4)
    verify_sources()
    from nki_library_attention_compat import install
    install()
    from nki_library_attention_readout_compat import install as install_readouts
    install_readouts()
    cases = [((2, 2, 512, 128), kind) for kind in ("random", "zero_qk", "small", "outlier")]
    cases += [((2, 8, 1024, 128), "random"), ((32, 8, 1024, 128), "random")]
    failures, all_original_passed = 0, True
    for shape, kind in cases:
        torch._dynamo.reset()
        emit("NKILIB_CASE_START", dict(shape=shape, kind=kind, diagnostic_only=True))
        gen = torch.Generator().manual_seed(166022 + shape[0] + shape[2])
        values = [torch.randn(shape, generator=gen).bfloat16() for _ in range(3)]
        grad = (torch.randn(shape, generator=gen) * .03).bfloat16()
        scale = shape[-1] ** -.5
        if kind == "zero_qk":
            values[0].zero_(); values[1].zero_()
        elif kind == "small":
            values[0] *= .03125; values[1] *= .03125
        elif kind == "outlier":
            values[0][:, :, ::127, ::31] *= 4
            values[1][:, :, ::127, ::31] *= 4
        functions = dict(bundled=make_forward("nkilib", scale), upstream=make_forward(NAMESPACE, scale))
        dev = [value.to("neuron").requires_grad_(True) for value in values]
        gd = grad.to("neuron")
        results = {}
        for name, fn in functions.items():
            got = forward_backward(fn, dev, gd)
            assert all(torch.isfinite(value.detach().cpu()).all() for value in got), name
            saved = tuple(value.detach().cpu() for value in got)
            exact(f"{shape}:{kind}:{name}:repeat", forward_backward(fn, dev, gd), saved)
            results[name] = saved
        for field, new, old in zip(FIELDS, results["upstream"], results["bundled"]):
            compare(f"{shape}:{kind}:{field}:unchanged_native_gate", new, old)
        for batch, head in sorted({(0, 0), (shape[0] - 1, shape[1] - 1)}):
            sliced = [value[batch:batch + 1, head:head + 1] for value in values]
            grad_slice = grad[batch:batch + 1, head:head + 1]
            reference = cpu_reference(sliced, grad_slice, scale)
            staged, absolute_pv_sum = rounding_estimate(sliced, grad_slice, scale)
            for name, got in results.items():
                for field, actual, expected, rounded in zip(FIELDS, got, reference, staged):
                    count, passed = report(f"{shape}:{kind}:{name}:{batch}:{head}:{field}",
                                           actual[batch:batch + 1, head:head + 1], expected, rounded,
                                           absolute_pv_sum if field == "output" else None)
                    failures += count
                    all_original_passed &= passed
        emit("NKILIB_ATTENTION_DIAGNOSTIC_CASE", dict(shape=shape, kind=kind, completed=True,
            unchanged_native_gate_passed=True, diagnostic_only=True, promotion_allowed=False))
    emit("NKILIB_ATTENTION_DIAGNOSTIC_RESULT", dict(completed=True, cases=len(cases),
        original_bound_failures=failures, all_original_cpu_gates_passed=all_original_passed,
        diagnostic_only=True, promotion_allowed=False, training_or_bpb_claim=False))


if __name__ == "__main__":
    main()
