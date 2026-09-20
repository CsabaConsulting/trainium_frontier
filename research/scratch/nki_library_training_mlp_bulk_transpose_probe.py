#!/usr/bin/env python3
"""Gate a compiled bulk transpose of library saves against native HxT training.

Original native equivalence and stagewise FP64 tests are retained. Independent
FP64 autograd/propagated roundoff bounds cover every gradient coordinate.
Two distinct microbatches exercise exact FP32 master-gradient accumulation.
This component includes master casts, layouts, saved activation, full backward
and gradient allocation. It excludes optimizer, communication and whole model.
"""
import argparse

import torch
import torch_neuronx

from mlp_backward_rounding_bounds import audit, reference_and_bounds
from mlp_fused_bf16 import make_autograd, reference_forward as row_reference_forward
from mlp_fused_bf16_probe import compare, gemm_gate, timing
from nki_library_probe_common import emit, exact, verify_sources
from nki_library_training_mlp import install, saved_relu_mlp


def reference_forward(x, up, down):
    output, relu = row_reference_forward(x, up, down)
    return output, relu.T.contiguous()


native = make_autograd(reference_forward, transposed_relu=True)


def run_case(shape, kind, steps):
    torch._dynamo.reset()
    tokens, channels = shape
    hidden = channels * 4
    gen = torch.Generator().manual_seed(171021 + tokens + channels)
    x = torch.randn(shape, generator=gen).bfloat16()
    up = torch.randn(hidden, channels, generator=gen) / channels ** .5
    down = torch.randn(channels, hidden, generator=gen) / hidden ** .5
    grad = (torch.randn(shape, generator=gen) * .01).bfloat16()
    if kind == "zero_input":
        x.zero_()
    elif kind == "zero_gradient":
        grad.zero_()
    elif kind == "tiny_gradient":
        grad *= 2**-12
    elif kind == "outliers":
        x[::127, ::63] *= 8
    elif kind == "changed_weights":
        up *= .5; down *= 1.5
    elif kind == "zero_up":
        up.zero_()
    elif kind == "zero_down":
        down.zero_()
    wrapped = torch_neuronx.wrap_nki(saved_relu_mlp)

    def call(xx, uu, dd):
        output, row_relu = wrapped[2](xx, uu.bfloat16().T.contiguous(), dd.bfloat16().T.contiguous())
        # The HBM read/write and compiler-selected transpose run inside the
        # compiled forward and every complete-component timing sample.
        return output, row_relu.T.contiguous()

    forward = torch.compile(call, backend="neuron", fullgraph=True, dynamic=False)
    reference = torch.compile(reference_forward, backend="neuron", fullgraph=True, dynamic=False)
    inputs = tuple(value.to("neuron") for value in (x, up, down))
    output, saved = forward(*inputs)
    expected_output, expected_saved = reference(*inputs)
    compare(f"{shape}:{kind}:forward", output, expected_output)
    compare(f"{shape}:{kind}:saved_relu", saved, expected_saved)
    exact(f"{shape}:{kind}:saved_repeat", forward(*inputs), (output, saved))
    indices = torch.unique(torch.cat((torch.linspace(0, tokens - 1, 33).long(),
        torch.tensor([0, 1, 127, 128, tokens // 2 - 1, tokens // 2, tokens - 1]))))
    saved_cpu = saved.detach().cpu().T
    gemm_gate(f"{shape}:{kind}:independent_saved_relu", saved_cpu[indices], x[indices], up.bfloat16().T, relu=True)
    gemm_gate(f"{shape}:{kind}:independent_down", output.detach().cpu()[indices],
              saved_cpu[indices].square(), down.bfloat16().T)
    del saved_cpu

    candidate = torch.compile(make_autograd(call, transposed_relu=True), backend="neuron", fullgraph=True, dynamic=False)
    control = torch.compile(native, backend="neuron", fullgraph=True, dynamic=False)
    cv = tuple(value.to("neuron").detach().clone().requires_grad_() for value in (x, up, down))
    rv = tuple(value.to("neuron").detach().clone().requires_grad_() for value in (x, up, down))
    gd = grad.to("neuron")

    def train(fn, values):
        for value in values:
            value.grad = None
        y = fn(*values)
        y.backward(gd)
        return (y, *(value.grad for value in values))

    got = train(candidate, cv)
    expected = train(control, rv)
    for field, new, old in zip(("train_forward", "dx", "dup", "ddown"), got, expected):
        compare(f"{shape}:{kind}:{field}", new, old)
    assert cv[0].grad.dtype == torch.bfloat16
    assert all(value.grad.dtype == torch.float32 for value in cv[1:])
    snapshot = tuple(value.detach().cpu() for value in got)
    exact(f"{shape}:{kind}:train_repeat", train(candidate, cv), snapshot)

    # The existing independently derived, unfitted error model is used as-is.
    # Full production gradients are checked too; no sampled weight-gradient shortcut.
    truth, bounds = reference_and_bounds(x, up.bfloat16(), down.bfloat16(), grad)
    audit(f"{shape}:{kind}:native_fp64", tuple(value.grad for value in rv), truth, bounds)
    audit(f"{shape}:{kind}:library_fp64", tuple(value.grad for value in cv), truth, bounds)
    del truth, bounds

    # Fresh leaf copies yield independent second-microbatch derivatives. The
    # accumulation target is ordinary FP32 addition of the two first-order VJPs.
    first = [value.grad.detach().cpu().clone() for value in cv[1:]]
    x2 = (-x * .5).to("neuron").requires_grad_()
    g2 = (grad * -.75).to("neuron")
    params2 = tuple(value.detach().clone().requires_grad_() for value in cv[1:])
    second_output = candidate(x2, *params2)
    second_output.backward(g2)
    second = [value.grad.detach().cpu().clone() for value in params2]
    candidate(x2.detach().clone().requires_grad_(), *cv[1:]).backward(g2)
    for field, value, a, b in zip(("up", "down"), cv[1:], first, second):
        exact(f"{shape}:{kind}:{field}:fp32_accumulation", value.grad, a + b)

    # Row-local MLP must not couple the two logical halves of this token batch.
    altered = x.clone(); altered[tokens // 2:] *= -2
    altered_output, altered_saved = forward(altered.to("neuron"), inputs[1], inputs[2])
    exact(f"{shape}:{kind}:other_rows_independent", altered_output[:tokens // 2], output[:tokens // 2])
    exact(f"{shape}:{kind}:saved_rows_independent", altered_saved[:, :tokens // 2], saved[:, :tokens // 2])
    measured = {}
    if kind == "random":
        measured = timing(dict(native_forward_backward=lambda: train(control, rv),
                               library_forward_backward=lambda: train(candidate, cv)),
                          steps, torch.neuron.synchronize)
    emit("NKILIB_TRAINING_BULK_TRANSPOSE_CASE", dict(shape=shape, kind=kind, passed=True,
        full_independent_fp64_autograd_gradients=True, unchanged_native_comparison=True,
        matched_native_hxt_control=True, saved_relu_layout="HxT", explicit_activation_transpose=True, compiler_bulk_transpose=True,
        exact_repeat_rows_and_fp32_accumulation=True, saved_relu_bytes=tokens * hidden * 2,
        master_casts_layouts_activation_stores_and_full_backward_charged=True,
        optimizer_communication_and_model_excluded=True, **measured))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify_sources()
    install()
    cases = [((1024, 256), kind) for kind in (
        "random", "zero_input", "zero_gradient", "tiny_gradient", "outliers", "changed_weights",
        "zero_up", "zero_down")]
    cases += [((2048, 1024), "random"), ((32768, 1024), "random")]
    for shape, kind in cases:
        emit("NKILIB_CASE_START", dict(shape=shape, kind=kind, training_component=True, saved_relu_layout="HxT"))
        run_case(shape, kind, args.steps)
    emit("NKILIB_TRAINING_BULK_TRANSPOSE_RESULT", dict(passed=True, cases=len(cases),
        saved_relu_and_full_backward=True, all_fp64_and_native_gates=True,
        matched_native_hxt_control=True, saved_relu_layout="HxT", compiler_bulk_transpose=True,
        two_microbatch_fp32_accumulation=True, training_or_bpb_claim=False))


if __name__ == "__main__":
    main()
