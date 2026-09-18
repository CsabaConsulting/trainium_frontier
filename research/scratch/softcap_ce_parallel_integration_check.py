#!/usr/bin/env python3
"""Gate the installed public CE/autograd path in eager and compiled execution."""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
from softcap_ce_parallel_integration import install, selected_lanes


def native_loss_factory(forward, backward):
    # Capture the original primitives so installing the candidate cannot
    # silently change the reference through train.py's global lookups.
    class NativeLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z, target):
            loss, lse = forward(z, target)
            ctx.save_for_backward(z, target, lse)
            return loss.reshape(-1)

        @staticmethod
        def backward(ctx, gradient):
            z, target, lse = ctx.saved_tensors
            return backward(z, target, lse, gradient.reshape(-1, 1).float().contiguous()), None

    def loss(z, target, reduction):
        value = NativeLoss.apply(z, target.reshape(-1, 1).float())
        return value.mean() if reduction == "mean" else value.view_as(target)
    return loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-shape", action="store_true")
    args = parser.parse_args()
    import torch_neuronx
    torch.set_num_threads(4)
    # This gate intentionally compiles six static shapes and two reductions.
    # Raising the diagnostic specialization cap does not enable graph fallback.
    torch._dynamo.config.recompile_limit = 32
    assert not T.LOSS_BYTE_WEIGHT and T.LOGIT_SOFTCAP == 15.0
    native = native_loss_factory(T._softcap_ce_fwd, T._softcap_ce_bwd)
    before_rng = torch.get_rng_state().clone()
    install(T)
    assert torch.equal(before_rng, torch.get_rng_state())

    def candidate(z, target, reduction):
        return T.softcap_cross_entropy(z, target, reduction)

    functions = {
        "eager": (native, candidate),
        "compiled": tuple(torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
                          for fn in (native, candidate)),
    }
    generator = torch.Generator().manual_seed(6023)
    cases = [(rows, "random") for rows in (128, 256, 384, 512, 1024)]
    cases += [(256, name) for name in ("tiny", "saturation", "uniform", "confident_correct",
                                      "confident_wrong", "zero_gradient")]
    if args.real_shape:
        cases.append((16384, "real_bf16_values"))
    records = []
    for rows, case in cases:
        z = torch.randn(rows, 8192, generator=generator) * 6
        target = torch.randint(8192, (rows,), generator=generator)
        target[:4] = torch.tensor([0, 8191, 4095, 4096])
        gradient = torch.randn(rows, generator=generator) * .5
        if case == "tiny":
            z *= .0001
        elif case == "saturation":
            z *= 1000
        elif case == "uniform":
            z.fill_(.7)
        elif case.startswith("confident"):
            z.zero_()
            z[torch.arange(rows), target] = 500 if case == "confident_correct" else -500
            if case == "confident_wrong":
                z[torch.arange(rows), (target + 1) % 8192] = 500
        elif case == "zero_gradient":
            gradient.zero_()
        elif case == "real_bf16_values":
            z = z.bfloat16().float()
        dz = z.to("neuron").requires_grad_()
        dt, dg = target.to("neuron"), gradient.to("neuron")
        reductions = ("none", "mean") if case in ("random", "real_bf16_values") else ("none",)
        for mode, pair in functions.items():
            for reduction in reductions:
                values = []
                for function in pair:
                    loss = function(dz, dt, reduction)
                    upstream = dg if reduction == "none" else dg[0]
                    derivative, = torch.autograd.grad(loss, dz, grad_outputs=upstream)
                    values.append((loss.detach().cpu(), derivative.cpu()))
                exact = [torch.equal(a, b) for a, b in zip(*values)]
                finite = all(bool(torch.isfinite(value).all()) for pair_values in values for value in pair_values)
                record = dict(case=case, rows=rows, lanes=selected_lanes(rows), mode=mode,
                              reduction=reduction, exact_loss_gradient=exact, finite=finite,
                              max_abs_loss_gradient=[float((a - b).abs().max()) for a, b in zip(*values)],
                              passed=all(exact) and finite)
                records.append(record)
                print("CE_INTEGRATION_GATE " + json.dumps(record), flush=True)
                assert record["passed"], record
                del values, loss, derivative
        assert torch.equal(dz.detach().cpu(), z) and torch.equal(dt.cpu(), target)
        del dz, dt, dg
        gc.collect()
    print("CE_INTEGRATION_RESULT " + json.dumps(dict(
        passed=all(record["passed"] for record in records), checks=len(records),
        real_shape=args.real_shape, compiled_fullgraph=True,
        original_autograd_interface=True, integrated_into_training=False)), flush=True)


if __name__ == "__main__":
    main()
