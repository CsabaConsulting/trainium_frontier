#!/usr/bin/env python3
"""Non-promoting paired timing after source169's complete diagnostic.

The original CPU coordinate screen is retained and reported, including its
failures in the bundled control. This is NOT its successful acceptance run.
Timing answers whether a corrected independent rounding contract is worth
further work. Candidate/bundled comparisons, repeats and structural checks
still assert. No full-model or quality promotion is permitted by this script.
"""
import argparse

import torch

from mlp_fused_bf16_probe import compare, timing
from nki_library_attention_numerics import FIELDS, report, rounding_estimate
from nki_library_attention_probe import cpu_reference, forward_backward, make_forward
from nki_library_probe_common import NAMESPACE, emit, exact, verify_sources


def case(shape, steps):
    torch._dynamo.reset()
    gen = torch.Generator().manual_seed(166022 + shape[0] + shape[2])
    values = [torch.randn(shape, generator=gen).bfloat16() for _ in range(3)]
    grad = (torch.randn(shape, generator=gen) * .03).bfloat16()
    scale = shape[-1] ** -.5
    functions = dict(bundled=make_forward("nkilib", scale), upstream=make_forward(NAMESPACE, scale))
    dev = [value.to("neuron").requires_grad_(True) for value in values]
    gd = grad.to("neuron")
    results = {}
    for name, fn in functions.items():
        got = forward_backward(fn, dev, gd)
        assert all(torch.isfinite(value.detach().cpu()).all() for value in got), name
        saved = tuple(value.detach().cpu() for value in got)
        exact(f"{shape}:{name}:repeat", forward_backward(fn, dev, gd), saved)
        results[name] = saved
    for field, new, old in zip(FIELDS, results["upstream"], results["bundled"]):
        compare(f"{shape}:{field}:unchanged_native_gate", new, old)

    # Exact behavioral checks are unchanged from the original component probe.
    fn = functions["upstream"]
    cutoff = shape[2] // 2
    changed_values = [value.clone() for value in values]
    for value in changed_values[1:]:
        value[:, :, cutoff:] *= -2
    changed = [value.to("neuron").requires_grad_(True) for value in changed_values]
    prefix_grad = grad.clone(); prefix_grad[:, :, cutoff:] = 0
    changed_result = forward_backward(fn, changed, prefix_grad.to("neuron"))
    exact(f"{shape}:causal_prefix", changed_result[0][:, :, :cutoff],
          results["upstream"][0][:, :, :cutoff])
    for field, value in zip(("dk", "dv"), changed_result[2:]):
        actual = value.detach().cpu()[:, :, cutoff:]
        exact(f"{shape}:causal_future_{field}", actual, torch.zeros_like(actual))
    changed_values = [value.clone() for value in values]
    for value in changed_values:
        value[1:] *= -.5
    changed = [value.to("neuron").requires_grad_(True) for value in changed_values]
    changed_result = forward_backward(fn, changed, gd)
    exact(f"{shape}:batch_independent", tuple(value[0] for value in changed_result),
          tuple(value[0] for value in results["upstream"]))

    original_passed = True
    original_failures = 0
    for batch, head in sorted({(0, 0), (shape[0] - 1, shape[1] - 1)}):
        sliced = [value[batch:batch + 1, head:head + 1] for value in values]
        gs = grad[batch:batch + 1, head:head + 1]
        reference = cpu_reference(sliced, gs, scale)
        staged, absolute_pv_sum = rounding_estimate(sliced, gs, scale)
        for name, got in results.items():
            for field, actual, expected, rounded in zip(FIELDS, got, reference, staged):
                count, passed = report(f"{shape}:{name}:{batch}:{head}:{field}",
                                       actual[batch:batch + 1, head:head + 1], expected, rounded,
                                       absolute_pv_sum if field == "output" else None)
                original_failures += count
                original_passed &= passed
    measured = timing({name: lambda fn=fn: forward_backward(fn, dev, gd)
                       for name, fn in functions.items()}, steps, torch.neuron.synchronize)
    emit("NKILIB_ATTENTION_TIMING_CASE", dict(shape=shape, completed=True,
        unchanged_native_gate_passed=True, exact_repeat_causality_and_batch_gates=True,
        original_cpu_screen_passed=original_passed, original_coordinate_failures=original_failures,
        diagnostic_only=True, promotion_allowed=False, forward_and_backward_and_layouts_charged=True,
        intermediate_readout_compatibility_cost_charged=True, **measured))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify_sources()
    from nki_library_attention_compat import install
    install()
    from nki_library_attention_readout_compat import install as install_readouts
    install_readouts()
    for shape in [(2, 8, 1024, 128), (32, 8, 1024, 128)]:
        emit("NKILIB_CASE_START", dict(shape=shape, diagnostic_only=True))
        case(shape, args.steps)
    emit("NKILIB_ATTENTION_TIMING_RESULT", dict(completed=True, cases=2,
        diagnostic_only=True, promotion_allowed=False, training_or_bpb_claim=False))


if __name__ == "__main__":
    main()
