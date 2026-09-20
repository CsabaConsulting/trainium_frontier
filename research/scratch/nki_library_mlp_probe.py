#!/usr/bin/env python3
"""Screen adapted pinned upstream BF16 MLP forward, with layout costs.

No forward-only result is a training speed result. The upstream kernel does not
return the ReLU activation needed by our backward, so any promotion must add and
charge saved activations or recomputation and independently validate gradients.
"""
import argparse
import importlib

import torch
import torch.nn.functional as F
import torch_neuronx

from mlp_fused_bf16_probe import compare, timing
from nki_library_probe_common import NAMESPACE, bounded, emit, exact, verify_sources


def make_forward(prefix):
    mlp = importlib.import_module(prefix + ".core.mlp.mlp").mlp
    types = importlib.import_module(prefix + ".core.utils.common_types")
    wrapped = torch_neuronx.wrap_nki(mlp)

    def call(x, up, down):
        # Ordinary Linear parameter orientation at the external boundary.
        # Both transposes/contiguous conversions are inside the timed function.
        up_layout = up.T.contiguous()
        # Upstream validates a non-None gate operand even when skip_gate_proj
        # is True. Alias the up operand; no gate GEMM or extra cache is needed.
        # CTE's intermediate-sharding loop addresses flattened B*S rows, but
        # its DMA-transpose input path only flattens for token sharding. Give
        # it the explicit row view for either strategy. MLP rows are independent.
        rows = x.reshape(1, -1, x.shape[-1])
        result = wrapped[2](rows, up_layout, up_layout, down.T.contiguous(),
                            activation_fn=types.ActFnType.SquaredReLU,
                            normalization_type=types.NormType.NO_NORM,
                            quantization_type=types.QuantizationType.NONE,
                            skip_gate_proj=True, force_cte_mode=True)
        # Native wrap_nki unwraps a single kernel output. Only index when the
        # wrapper actually returns a collection; indexing a Tensor drops B.
        output = result if isinstance(result, torch.Tensor) else result[0]
        return output.reshape(x.shape)

    return torch.compile(call, backend="neuron", fullgraph=True, dynamic=False)


def native(x, up, down):
    return F.linear(F.relu(F.linear(x, up)).square(), down)


def cpu_reference(x, up, down):
    """FP64 ReLU-squared, with propagated accumulation/BF16-rounding bounds."""
    x, up, down = (t.double() for t in (x, up, down))
    channels, hidden = x.shape[-1], up.shape[0]
    unit = 2.0 ** -8
    gamma_up = channels * 2.0 ** -24 / (1 - channels * 2.0 ** -24)
    gamma_down = hidden * 2.0 ** -24 / (1 - hidden * 2.0 ** -24)
    pre = x @ up.T
    pre_error = gamma_up * (x.abs() @ up.abs().T)
    # Both native and the explicit library adapter round the preactivation
    # through BF16 ReLU before squaring. Preserve that rounding in the bound.
    pre_error = pre_error + unit * (pre.abs() + pre_error)
    relu = pre.clamp_min(0)
    activated = relu.square()
    activated_error = 2 * relu * pre_error + pre_error.square()
    activated_error += unit * (activated.abs() + activated_error)
    out = activated @ down.T
    propagated = activated_error @ down.abs().T
    accumulation = gamma_down * ((activated.abs() + activated_error) @ down.abs().T)
    out_error = propagated + accumulation
    bound = out_error + unit * (out.abs() + out_error) + 1e-30
    return out, bound


def case(shape, kind, steps, token_sharding=False):
    torch._dynamo.reset()  # no live graphs between distinct shape screens
    gen = torch.Generator().manual_seed(166021 + shape[1] + shape[2])
    batch, tokens, channels = shape
    hidden = channels * 4
    x = torch.randn(shape, generator=gen).bfloat16()
    up = (torch.randn(hidden, channels, generator=gen) / channels ** .5).bfloat16()
    down = (torch.randn(channels, hidden, generator=gen) / hidden ** .5).bfloat16()
    if kind == "zero":
        x.zero_()
    elif kind == "outlier":
        x[:, ::127, ::63] *= 8
    elif kind == "changed":
        up *= .5
        down *= 1.5
    arms = dict(native=torch.compile(native, backend="neuron", fullgraph=True, dynamic=False),
                upstream_relu2_adapter=make_forward(NAMESPACE))
    dev = tuple(v.to("neuron") for v in (x, up, down))
    # All output channels for 33 distinct rows spanning batch/sequence, or all
    # rows when smaller. This avoids a production-size CPU GEMM while checking
    # batch and tile boundaries. Finite and repeat gates still cover all rows.
    idx = torch.unique(torch.linspace(0, batch * tokens - 1, 33).long())
    truth, bound = cpu_reference(x.reshape(-1, channels)[idx], up, down)
    outputs = {}
    for name, fn in arms.items():
        with torch.no_grad():
            output = fn(*dev)
        emit("NKILIB_OUTPUT_CONTRACT", dict(name=name, expected_shape=shape,
                                           actual_shape=list(output.shape), dtype=str(output.dtype)))
        assert tuple(output.shape) == shape and output.dtype == torch.bfloat16, (
            name, tuple(output.shape), output.dtype, shape)
        cpu = output.detach().cpu()
        assert torch.isfinite(cpu).all(), name
        bounded(f"{shape}:{kind}:{name}:fp64", cpu.reshape(-1, channels)[idx], truth, bound)
        with torch.no_grad():
            exact(f"{shape}:{kind}:{name}:repeat", fn(*dev), cpu)
        outputs[name] = cpu
    if token_sharding:
        # The row-sharded path also has to satisfy the retained campaign
        # native-equivalence gate before timing or considering training work.
        compare(f"{shape}:{kind}:token_sharding_native_gate",
                outputs["upstream_relu2_adapter"], outputs["native"])
    if batch > 1:
        altered = x.clone()
        altered[1:] *= -2
        other = arms["upstream_relu2_adapter"](altered.to("neuron"), dev[1], dev[2]).detach().cpu()
        exact(f"{shape}:{kind}:batch_independent", other[0], outputs["upstream_relu2_adapter"][0])
    measured = {}
    if kind == "random":
        with torch.no_grad():
            measured = timing({name: lambda fn=fn: fn(*dev) for name, fn in arms.items()},
                              steps, torch.neuron.synchronize)
    emit("NKILIB_MLP_CASE", dict(shape=list(shape), kind=kind, passed=True, **measured,
                                upstream_vs_native_relative_rms=float(
                                    (outputs["upstream_relu2_adapter"].float() - outputs["native"].float()).norm()
                                    / outputs["native"].float().norm().clamp_min(1e-30)),
                                forward_only=True, adapted_relu2_cte=True, saved_training_activation_excluded=True,
                                input_weight_layout_work_charged=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--token-sharding", action="store_true",
                        help="Override only isolated CTE shard selection with independent token rows")
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify_sources()
    from nki_library_relu2_adapter import install
    install(token_sharding=args.token_sharding)
    emit("NKILIB_RELU2_ADAPTER", dict(installed_in_isolated_namespace=True,
                                     upstream_single_op_helper_rejects_squared_relu=True,
                                     installed_compiler_unchanged=True,
                                     token_sharding=args.token_sharding))
    cases = [((2, 512, 256), kind) for kind in ("random", "zero", "outlier", "changed")]
    cases += [((2, 1024, 1024), "random"), ((32, 1024, 1024), "random")]
    for shape, kind in cases:
        emit("NKILIB_CASE_START", dict(shape=shape, kind=kind))
        case(shape, kind, args.steps, token_sharding=args.token_sharding)
    emit("NKILIB_MLP_RESULT", dict(passed=True, cases=len(cases), forward_only=True, adapted_relu2_cte=True,
                                  token_sharding=args.token_sharding, training_or_bpb_claim=False))


if __name__ == "__main__":
    main()
