#!/usr/bin/env python3
"""Gate cached head against compiled original Linear/master-weight gradients.

Exact equality is required, including FP32 master accumulation through two
microbatches and cache refresh after a deterministic update. Hardware reference
is needed because eager Linear introduces a different BF16 logits boundary.
Independent projection/CE gates are retained in head_loss_fused_probe.py.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
from head_loss_fused_autograd import loss_factory
from softcap_ce_parallel_integration import install


def emit(kind, record):
    print(kind + " " + json.dumps(record), flush=True)


def exact_gate(name, native, candidate):
    left = tuple(v.detach().cpu() for v in native)
    right = tuple(v.detach().cpu() for v in candidate)
    equal = [torch.equal(a, b) for a, b in zip(left, right)]
    finite = all(bool(torch.isfinite(v).all()) for row in (left, right) for v in row)
    record = dict(name=name, exact_loss_input_master_gradient=equal,
                  max_abs=[float((a.float() - b.float()).abs().max()) for a, b in zip(left, right)],
                  fp32_master_gradient=right[-1].dtype == torch.float32,
                  finite=finite, passed=all(equal) and finite and right[-1].dtype == torch.float32)
    emit("FUSED_HEAD_AUTOGRAD_GATE", record)
    assert record["passed"], record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-shape", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--saved-softcap", action="store_true")
    parser.add_argument("--fp32-carrier", action="store_true")
    args = parser.parse_args()
    assert not args.benchmark or args.real_shape
    import torch_neuronx
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 32
    T.USE_WEIGHT_CACHE = True
    before_rng = torch.get_rng_state().clone()
    install(T)
    fused_loss = loss_factory(save_softcap=args.saved_softcap, fp32_carrier=args.fp32_carrier)
    assert torch.equal(before_rng, torch.get_rng_state())
    generator = torch.Generator().manual_seed(9417)
    shapes = [(512, 256)] + ([(16384, 1024)] if args.real_shape else [])
    count = 0
    for rows, channels in shapes:
        head = T.Linear(channels, 8192, bias=False).to("neuron")

        def original(x, target, reduction):
            if args.fp32_carrier:
                x = x.bfloat16()
            return T.softcap_cross_entropy(head(x).float(), target, reduction)

        def candidate(x, target, reduction):
            return fused_loss(x, head._cached_weight, target, reduction)

        functions = [torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
                     for fn in (original, candidate)]
        cases = ("zero_weight", "tiny_weight", "random", "outliers", "zero_gradient",
                 "confident_correct", "confident_wrong") if rows == 512 else ("random",)
        for case in cases:
            xc = torch.randn(rows, channels, generator=generator).bfloat16()
            wc = torch.randn(8192, channels, generator=generator) * .03
            tc = torch.randint(8192, (rows,), generator=generator)
            tc[:4] = torch.tensor([0, 8191, 511, 512])
            gcpu = torch.randn(rows, generator=generator) * .5
            if case == "zero_weight":
                wc.zero_()
            elif case == "tiny_weight":
                wc *= 2.0**-12
            elif case == "outliers":
                xc[:, 0] *= 64
                wc[:, 127] *= 64
            elif case == "zero_gradient":
                gcpu.zero_()
            elif case.startswith("confident_"):
                xc.zero_(); xc[:, 0] = 1
                wc.zero_(); wc[:, 0] = -80; wc[0, 0] = 80
                tc.fill_(0 if case == "confident_correct" else 8191)
            x = xc.to(device="neuron", dtype=torch.float32 if args.fp32_carrier
                      else torch.bfloat16).requires_grad_()
            target, gradient = tc.to("neuron"), gcpu.to("neuron")
            with torch.no_grad():
                head.weight.copy_(wc.to("neuron"))
            for reduction in ("none", "mean"):
                outputs = []
                for function in functions:
                    head.refresh_weight_cache(torch.bfloat16)
                    value = function(x, target, reduction)
                    derivs = torch.autograd.grad(value, (x, head.weight),
                        grad_outputs=gradient if reduction == "none" else gradient[0])
                    outputs.append((value, *derivs))
                exact_gate(f"{rows}x{channels}:{case}:{reduction}", *outputs)
                count += 1
                del outputs, value, derivs
            assert torch.equal(head.weight.detach().cpu(), wc)
            assert torch.equal(x.detach().cpu(), xc) and torch.equal(target.cpu(), tc)
        # Two distinct microbatches accumulate into the same FP32 master.
        x2 = (xc * .75 + .125).to(device="neuron", dtype=x.dtype).requires_grad_()
        initial_master = head.weight.detach().cpu().clone()
        accumulated, updated = [], []
        for function in functions:
            with torch.no_grad():
                head.weight.copy_(initial_master.to("neuron"))
            head.weight.grad = None
            x.grad = x2.grad = None
            head.refresh_weight_cache(torch.bfloat16)
            first = function(x, target, "mean")
            # Each microbatch has a fresh compiled graph. Match the actual
            # trainer's ordinary backward; retain_graph is incompatible with
            # AOT donated buffers and is unnecessary for the shared cast.
            first.backward()
            second = function(x2, target, "mean")
            second.backward()
            accumulated.append((first.detach(), second.detach(), x.grad.clone(), x2.grad.clone(),
                                head.weight.grad.clone()))
            with torch.no_grad():
                head.weight.add_(head.weight.grad, alpha=-.001)
            head.refresh_weight_cache(torch.bfloat16)
            refreshed = function(x2, target, "mean")
            dx, dw = torch.autograd.grad(refreshed, (x2, head.weight))
            updated.append((refreshed, dx, dw))
        exact_gate(f"{rows}x{channels}:two_microbatch_accumulation", *accumulated)
        exact_gate(f"{rows}x{channels}:updated_master_refreshed_cache", *updated)
        count += 2
        del accumulated, updated
        if args.benchmark and rows == 16384:
            from mlp_fused_bf16_probe import timing
            head.weight.grad = None
            head.refresh_weight_cache(torch.bfloat16)

            def step(function):
                value = function(x, target, "mean")
                return value, torch.autograd.grad(value, (x, head.weight))

            pair = [step(fn) for fn in functions]
            exact_gate("benchmark_boundary", (pair[0][0], *pair[0][1]),
                       (pair[1][0], *pair[1][1]))
            del pair
            measured = timing({name: (lambda fn=fn: step(fn))
                               for name, fn in zip(("native", "cached"), functions)},
                              args.steps, torch.neuron.synchronize)
            emit("FUSED_HEAD_AUTOGRAD_BENCH", dict(shape=[rows, channels, 8192], **measured,
                units="ms per head/loss forward and backward",
                saved_softcap=args.saved_softcap,
                fp32_carrier=args.fp32_carrier,
                transpose_per_forward_charged=True, fp32_master_gradient=True,
                optimizer_or_whole_model_included=False))
        del head, x, x2, target, gradient
        gc.collect()
    emit("FUSED_HEAD_AUTOGRAD_RESULT", dict(passed=True, comparisons=count,
        real_shape=args.real_shape, compiled_original_linear=True,
        saved_softcap=args.saved_softcap,
        fp32_carrier=args.fp32_carrier,
        cache_refresh_and_master_accumulation=True, rng_unchanged=True,
        integrated_into_model=False, quality_or_whole_step_promotion=False))


if __name__ == "__main__":
    main()
