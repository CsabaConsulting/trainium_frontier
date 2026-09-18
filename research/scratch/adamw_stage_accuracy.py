"""Independent FP64 gates for the stages of a stored-FP32-state AdamW update.

The original whole-chain weight envelope omitted amplified moment rounding near
cancellation. Check both states against ideal FP64, then the weight computation
against FP64 using the actual stored FP32 states. Keep the original constants,
global ideal-FP64 RMS gate, and separate accumulated-trajectory gate.
"""
import json

import torch

from adamw_fused_fp32 import scalar_values
from adamw_fused_fp32_probe import check, expected64


def stage_reference(actual, inputs, values):
    p, g, old_m, old_v = inputs
    ideal, state_envelopes = expected64(p, g, old_m, old_v, values)
    weight, m, v = [x.detach().cpu().double() for x in actual]
    assert bool(torch.isfinite(m).all()) and bool(torch.isfinite(v).all())
    assert bool((v >= 0).all()), "second moment is negative"
    lr, b1, b2, _, _, decay, eps = values.double().unbind()
    update = lr * (m / b1) / ((v / b2).sqrt() + eps)
    conditional_weight = p.double() * decay - update
    weight_envelope = (p.double() * decay).abs() + update.abs()
    return ((conditional_weight, ideal[1], ideal[2]),
            (weight_envelope, state_envelopes[1], state_envelopes[2]),
            ideal, state_envelopes)


def check_stages(name, actual, inputs, values):
    expected, envelopes, ideal, original_envelopes = stage_reference(actual, inputs, values)
    check("staged:" + name, actual, expected, envelopes, inputs=inputs)
    weight = actual[0].detach().cpu().double()
    relative = float((weight - ideal[0]).norm() / ideal[0].norm().clamp_min(1e-30))
    assert relative < 2e-6, ("ideal-FP64 whole-update RMS", name, relative)
    u = 2.0 ** -24
    old_bound = 4 * (32 * u / (1 - 32 * u)) * original_envelopes[0] + 32 * 2.0 ** -149
    print("ADAMW_STAGE_PROPAGATION " + json.dumps(dict(
        name=name, whole_update_ideal_fp64_relative_rms=relative,
        original_unpropagated_bound_failures=int(((weight - ideal[0]).abs() > old_bound).sum()),
        maximum_weight_effect_of_stored_state_rounding=float((expected[0] - ideal[0]).abs().max()),
        maximum_weight_arithmetic_error=float((weight - expected[0]).abs().max()),
        state_and_arithmetic_constants_unchanged=True)), flush=True)


def cpu_counterexamples():
    # Exact original inputs from the independently preserved wave58 receipt.
    p = torch.tensor([-.0002836883068084717, .00014795362949371338], dtype=torch.float32)
    g = torch.tensor([-.005310865584760904, .0046853311359882355], dtype=torch.float32)
    m = torch.tensor([.000591130112297833, -.0005207399954088032], dtype=torch.float32)
    v = torch.tensor([6.050793217582395e-7, 4.700442559624207e-7], dtype=torch.float32)
    values = scalar_values(.1125, 4, .9, .96, .01)
    nm = m + (g - m) * values[3]
    nv = v + (g.square() - v) * values[4]
    w = p * values[5] - values[0] * (nm / values[1]) / ((nv / values[2]).sqrt() + values[6])
    ideal, envelopes = expected64(p, g, m, v, values)
    original = check("cpu_counterexample:original", (w, nm, nv), ideal, envelopes, enforce=False)
    assert original["results"][0]["bound_failures"] == 2
    def fixture_check(name, actual):
        expected, bounds, _, _ = stage_reference(actual, (p, g, m, v), values)
        return check(name, actual, expected, bounds, enforce=False)

    # These two deliberately selected near-cancellation values are an
    # elementwise-bound regression fixture, not an RMS sample of a full table.
    # Their large relative moment errors are expected; the complete 8M-element
    # integration checks still enforce the unchanged RMS limits in check_stages.
    staged = fixture_check("cpu_counterexample:correct_fp32", (w, nm, nv))
    assert all(r["finite"] and r["bound_failures"] == 0 for r in staged["results"])
    # Conditioning on stored states cannot excuse inaccurate states: corrupt a
    # moment and recompute a consistent weight; the independent state gate fails.
    bad_m = nm + .001
    bad_w = p * values[5] - values[0] * (bad_m / values[1]) / ((nv / values[2]).sqrt() + values[6])
    corrupt_state = fixture_check("cpu_counterexample:corrupt_state", (bad_w, bad_m, nv))
    assert corrupt_state["results"][1]["bound_failures"] == 2, "corrupted moment escaped"
    corrupt_weight = fixture_check("cpu_counterexample:corrupt_weight", (w + .001, nm, nv))
    assert corrupt_weight["results"][0]["bound_failures"] == 2, "corrupted weight escaped"
    print("PASS: original-bound counterexamples and elementwise negative controls; "
          "selected cancellation fixture is not a full-table RMS sample", flush=True)


if __name__ == "__main__":
    cpu_counterexamples()
