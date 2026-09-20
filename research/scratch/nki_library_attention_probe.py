#!/usr/bin/env python3
"""Compare pinned upstream and bundled NKI attention forward AND backward.

Matches the installed SDPA layout/LSE/precision contract. No backend global or
installed package is patched. CPU references use independent FP64 autograd.
"""
import argparse
import importlib

import torch
import torch.nn.functional as F
import torch_neuronx

from mlp_fused_bf16_probe import compare, timing
from nki_library_probe_common import NAMESPACE, attention_cpu_check, emit, exact, verify_sources


def make_forward(prefix, scale):
    fwd = torch_neuronx.wrap_nki(importlib.import_module(prefix + ".core.attention.attention_cte").attention_cte)
    bwd = torch_neuronx.wrap_nki(importlib.import_module(prefix + ".core.attention.attention_bwd").attention_bwd)

    class Attention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v):
            batch, heads, tokens, dim = q.shape
            out, neg, recip = fwd[2](q.reshape(batch * heads, tokens, dim),
                                     k.reshape(batch * heads, tokens, dim),
                                     v.reshape(batch * heads, tokens, dim),
                                     tp_q=True, tp_k=True, scale=scale,
                                     causal_mask=True, cache_softmax=True)
            out = out.reshape(batch, heads, tokens, dim)
            lse = (-(neg + torch.log(recip))).float().reshape(batch, heads, 128, tokens // 128)
            ctx.save_for_backward(q, k, v, out, lse)
            return out

        @staticmethod
        def backward(ctx, grad):
            q, k, v, out, lse = ctx.saved_tensors
            dq, dk, dv = bwd[2](q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1),
                                out.transpose(-2, -1), grad.transpose(-2, -1), lse,
                                use_causal_mask=True, mixed_precision=True, softmax_scale=scale)
            return dq.transpose(-2, -1), dk.transpose(-2, -1), dv.transpose(-2, -1)

    return torch.compile(Attention.apply, backend="neuron", fullgraph=True, dynamic=False)


def forward_backward(fn, values, grad):
    for value in values:
        value.grad = None
    output = fn(*values)
    output.backward(grad)
    return (output, *(value.grad for value in values))


def cpu_reference(values, grad, scale):
    values = [v.double().requires_grad_(True) for v in values]
    output = F.scaled_dot_product_attention(*values, is_causal=True, scale=scale)
    output.backward(grad.double())
    return (output.detach(), *(v.grad for v in values))


def run_case(shape, kind, steps):
    torch._dynamo.reset()
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
    dev = [v.to("neuron").requires_grad_(True) for v in values]
    gd = grad.to("neuron")
    results = {}
    for name, fn in functions.items():
        got = forward_backward(fn, dev, gd)
        assert all(torch.isfinite(v.detach().cpu()).all() for v in got), name
        saved = tuple(v.detach().cpu() for v in got)
        exact(f"{shape}:{kind}:{name}:repeat", forward_backward(fn, dev, gd), saved)
        results[name] = saved
    for field, new, old in zip(("output", "dq", "dk", "dv"), results["upstream"], results["bundled"]):
        compare(f"{shape}:{kind}:{field}:unchanged_native_gate", new, old)

    # Heads/batches are mathematically independent. The complete key/query
    # sequence is retained for each sampled head, including all causal rows.
    selected = sorted({(0, 0), (shape[0] - 1, shape[1] - 1)})
    for batch, head in selected:
        sliced = [v[batch:batch + 1, head:head + 1] for v in values]
        reference = cpu_reference(sliced, grad[batch:batch + 1, head:head + 1], scale)
        for name, output in results.items():
            for field, actual, expected in zip(("output", "dq", "dk", "dv"), output, reference):
                attention_cpu_check(f"{shape}:{kind}:{name}:{batch}:{head}:{field}:fp64",
                                    actual[batch:batch + 1, head:head + 1], expected)

    fn = functions["upstream"]
    # Perturb only future K/V, check the prefix output exactly. Then backprop
    # a prefix-only loss and require zero K/V gradients in all future positions.
    cutoff = shape[2] // 2
    altered = [v.clone() for v in values]
    for value in altered[1:]:
        value[:, :, cutoff:] *= -2
    changed = [v.to("neuron").requires_grad_(True) for v in altered]
    prefix_grad = grad.clone(); prefix_grad[:, :, cutoff:] = 0
    changed_result = forward_backward(fn, changed, prefix_grad.to("neuron"))
    exact(f"{shape}:{kind}:causal_prefix", changed_result[0][:, :, :cutoff],
          results["upstream"][0][:, :, :cutoff])
    for field, value in zip(("dk", "dv"), changed_result[2:]):
        actual = value.detach().cpu()[:, :, cutoff:]
        exact(f"{shape}:{kind}:causal_future_{field}", actual, torch.zeros_like(actual))
    if shape[0] > 1:
        altered = [v.clone() for v in values]
        for value in altered:
            value[1:] *= -.5
        changed = [v.to("neuron").requires_grad_(True) for v in altered]
        changed_result = forward_backward(fn, changed, gd)
        exact(f"{shape}:{kind}:batch_independent", tuple(v[0] for v in changed_result),
              tuple(v[0] for v in results["upstream"]))
    measured = {}
    if kind == "random":
        measured = timing({name: lambda fn=fn: forward_backward(fn, dev, gd)
                           for name, fn in functions.items()}, steps, torch.neuron.synchronize)
    emit("NKILIB_ATTENTION_CASE", dict(shape=list(shape), kind=kind, passed=True, **measured,
                                      both_passes_and_layouts_charged=True,
                                      independent_cpu_fp64_gradients=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify_sources()
    from nki_library_attention_compat import install
    install()
    cases = [((2, 2, 512, 128), kind) for kind in ("random", "zero_qk", "small", "outlier")]
    cases += [((2, 8, 1024, 128), "random"), ((32, 8, 1024, 128), "random")]
    for shape, kind in cases:
        emit("NKILIB_CASE_START", dict(shape=shape, kind=kind))
        run_case(shape, kind, args.steps)
    emit("NKILIB_ATTENTION_RESULT", dict(passed=True, cases=len(cases), full_backward=True,
                                        independent_cpu_fp64_gradients=True, training_or_bpb_claim=False))


if __name__ == "__main__":
    main()
