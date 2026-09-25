#!/usr/bin/env python3
"""Deterministic verdicts for the v2 research loop. No LLM judgment in the critical path.

Input: the append-only ledger (JSON Lines, one record per finished+evaluated run; schema in
workflow/templates/run-record.json). Output: a verdict JSON for one arm, or a control-drift report.

  python workflow/tools/decide.py --ledger research/v2/ledger.jsonl --arm A2
  python workflow/tools/decide.py --ledger research/v2/ledger.jsonl --drift

Rules (change only with human approval, and bump RULES_VERSION):
  valid run      : status=="evaluated", over_budget False, causality_passed not False, finite BPB,
                   eval_tokens==20_971_520, eval_seq_len==1024, git_dirty False,
                   explicit consistent 1800-second budget declaration; not a diagnostic/smoke
  control        : kind=="control", same host, same base_sha256, valid; pooled mean of up to the
                   last 3 controls on that host for that base (>=1 required, >=2 preferred)
  step count     : reported only for EVERY arm; <60% of same-host control is a diagnostic alarm,
                   as are recorded timing spikes/stalls; none changes the BPB verdict
  big-win hold   : delta <= -0.010 or public_20m < 0.975 -> HOLD (independent cold replay + audit)
  one host       : delta <= -0.0006 -> NOMINATE (replicate on the other host)
                   delta >= +0.0005 -> DISCARD
                   otherwise        -> REPLICATE if kind in HIGH_PRIOR else DISCARD
  two hosts      : both < 0, mean <= -0.0010, |d21-d22| <= 0.0015 -> ADOPT_PENDING_HUMAN_ACK
                   both < 0, mean <= -0.0010, disagreement > 0.0015 -> REPLICATE (third run)
                   mean > -0.0003 -> DISCARD
                   otherwise -> REPLICATE once if HIGH_PRIOR and replicates < 3, else DISCARD
  drift alarm    : a control more than 0.0010 from the mean of the previous controls (same host+base)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

RULES_VERSION = "v2-2026-09-25-general-fixed-budget"
EVAL_TOKENS, EVAL_SEQ = 20_971_520, 1024
TRAINING_BUDGET = 1800
NOMINATE, DISCARD_SINGLE = -0.0006, 0.0005
ADOPT_MEAN, ADOPT_DISAGREE, DISCARD_MEAN = -0.0010, 0.0015, -0.0003
HOLD_DELTA, HOLD_ABS = -0.010, 0.975
STEP_DIAGNOSTIC_RATIO = 0.60
DRIFT = 0.0010
HIGH_PRIOR = {"table_arm", "data_order_arm"}


def budget_declarations(run):
    """Read explicit declarations in the archived record, never a source default.

    Receipt authentication remains the Operator/result-review boundary, as for
    source hashes and over_budget. All present declarations must agree, including
    CLI overrides; an env-only authenticated archival declaration is sufficient.
    """
    declarations = {}
    if "training_budget_seconds" in run:
        declarations["training_budget_seconds"] = run["training_budget_seconds"]
    env = run.get("env", {})
    if not isinstance(env, dict):
        return declarations, "invalid archived environment for budget declaration"
    key = "NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS"
    if key in env:
        declarations["env." + key] = env[key]
    argv = run.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(v, str) for v in argv):
        return declarations, "invalid archived argv for budget declaration"
    for i, arg in enumerate(argv):
        if arg == "--max-train-seconds":
            if i + 1 == len(argv):
                return declarations, "missing --max-train-seconds value"
            declarations[f"argv[{i}]"] = argv[i + 1]
        elif arg.startswith("--max-train-seconds="):
            declarations[f"argv[{i}]"] = arg.split("=", 1)[1]
    if not declarations:
        return declarations, "missing explicit 1800-second training budget declaration"
    values = []
    for value in declarations.values():
        try:
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError
            value = float(value)
            if not math.isfinite(value):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            return declarations, "invalid training budget declaration"
        values.append(value)
    if len(set(values)) != 1:
        return declarations, "conflicting training budget declarations"
    if values[0] != TRAINING_BUDGET:
        return declarations, "declared training budget is not 1800 seconds"
    return declarations, None


def finite_nonnegative(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def load(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_bytes().replace(b"\x00", b"").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def invalid_reason(r: dict) -> str | None:
    if r.get("status") != "evaluated":
        return f"status={r.get('status')}"
    if r.get("over_budget") is not False:
        return "over_budget not False"
    if (r.get("kind") in {"diagnostic", "smoke", "timing", "compile_smoke", "native_smoke"}
            or ("run_purpose" in r and r["run_purpose"] != "quality")):
        return "diagnostic/smoke is not a quality run"
    _, budget_error = budget_declarations(r)
    if budget_error:
        return budget_error
    if r.get("causality_passed") is False:  # None = not reported by this evaluator; False = failed
        return "causality check failed"
    if r.get("git_dirty") is not False:
        return "git_dirty not False"
    if r.get("eval_tokens") != EVAL_TOKENS or r.get("eval_seq_len") != EVAL_SEQ:
        return "not a public-20M context-1024 eval"
    b = r.get("public_20m")
    if not isinstance(b, (int, float)) or not math.isfinite(b):
        return "non-finite public_20m"
    return None


def controls_for(rows, host, base, before_ts=None, k=3):
    c = [r for r in rows if r.get("kind") == "control" and r.get("host") == host
         and r.get("base_sha256") == base and invalid_reason(r) is None
         and (before_ts is None or r["timestamp_utc"] <= before_ts)]
    c.sort(key=lambda r: r["timestamp_utc"])
    return c[-k:]


def run_diagnostics(rows, run):
    """Observations only: never consulted by the BPB decision branches."""
    alarms = []
    ctrls = controls_for(rows, run.get("host"), run.get("base_sha256"))
    control_steps = [c.get("steps") for c in ctrls]
    steps, mean_steps, ratio = run.get("steps"), None, None
    if control_steps and all(finite_nonnegative(s) for s in control_steps):
        mean_steps = sum(control_steps) / len(control_steps)
    if finite_nonnegative(steps) and mean_steps is not None and mean_steps > 0:
        ratio = steps / mean_steps
        if ratio < STEP_DIAGNOSTIC_RATIO:
            alarms.append("steps_below_60_percent_of_same_host_control")
    elif not finite_nonnegative(steps):
        alarms.append("step_count_missing_or_invalid")
    timing = run.get("timing_diagnostics", {})
    timing = timing if isinstance(timing, dict) else {}
    for name, flag in (("step_time_spikes", "step_time_spike_detected"), ("stalls", "stall_detected")):
        events = timing.get(name, run.get(name))
        if (isinstance(events, list) and events) or timing.get(flag) is True or run.get(flag) is True:
            alarms.append("recorded_" + name)
    measured = run.get("startup_measured_seconds")
    if measured is None:
        measured = run.get("harness_startup_s")
    if not finite_nonnegative(measured):
        measured = None
        alarms.append("measured_startup_missing")
    elif measured > 870:
        alarms.append("startup_above_870_second_planning_warning")
    cold = run.get("cold_startup_seconds")
    cold_evidence = run.get("cold_startup_evidence")
    cold_proven = (run.get("cold_startup_proven") is True and finite_nonnegative(cold)
                   and cold > 0 and isinstance(cold_evidence, str) and bool(cold_evidence.strip()))
    if not cold_proven:
        alarms.append("cold_startup_unproven_submission_gate")
    declarations, budget_error = budget_declarations(run)
    return {"steps": steps, "control_mean_steps": mean_steps,
            "step_ratio": round(ratio, 4) if ratio is not None else None,
            "diagnostic_alarms": alarms, "affects_bpb_verdict": False,
            "budget_declarations": declarations, "budget_declaration_error": budget_error,
            "training_seconds": run.get("training_seconds"),
            "charged_through_exit_seconds": run.get("charged_through_exit_seconds", run.get("harness_charged_through_exit_s")),
            "charged_source_estimate_seconds": run.get("charged_seconds_est"),
            "startup_measured_seconds": measured,
            "startup_allowance_seconds": run.get("startup_allowance_seconds", run.get("startup_allowance_s")),
            "startup_source_estimate_seconds": run.get("startup_source_estimate_seconds"),
            "legacy_startup_s_untyped": run.get("startup_s"),
            "cold_startup_seconds": cold,
            "cold_startup_submission_gate": "EVIDENCE_PRESENT_NOT_SUBMISSION_APPROVAL" if cold_proven else "HOLD_MISSING_PROVEN_COLD_STARTUP"}


def host_delta(rows, run):
    ctrls = controls_for(rows, run["host"], run["base_sha256"])
    if not ctrls:
        return {"host": run["host"], "label": run["label"], "error": "no valid same-host control for this base"}
    cm = sum(c["public_20m"] for c in ctrls) / len(ctrls)
    diagnostic = run_diagnostics(rows, run)
    return {"host": run["host"], "label": run["label"], "public_20m": run["public_20m"],
            "controls": [c["label"] for c in ctrls], "control_mean": round(cm, 7),
            "delta": round(run["public_20m"] - cm, 7), "steps": run.get("steps"),
            "step_ratio": diagnostic["step_ratio"], "control_mean_steps": diagnostic["control_mean_steps"],
            "pooled_controls": len(ctrls)}


def decide_arm(rows, arm_id):
    runs = [r for r in rows if r.get("arm_id") == arm_id and r.get("kind") != "control"]
    diagnostics = {r["label"]: run_diagnostics(rows, r) for r in runs}
    bad = {r["label"]: invalid_reason(r) for r in runs if invalid_reason(r)}
    runs = [r for r in runs if r["label"] not in bad]
    out = {"rules": RULES_VERSION, "arm_id": arm_id, "invalid_runs": bad, "run_diagnostics": diagnostics}
    if not runs:
        out["verdict"] = "INSUFFICIENT"
        out["why"] = "no valid evaluated runs for this arm"
        return out
    kind = runs[-1].get("kind")
    latest = {}
    for r in sorted(runs, key=lambda r: r["timestamp_utc"]):
        latest[r["host"]] = r
    per_host = [host_delta(rows, r) for r in latest.values()]
    out["per_host"] = per_host
    out["replicates"] = len(runs)
    if any("error" in h for h in per_host):
        out["verdict"] = "INSUFFICIENT"
        out["why"] = "; ".join(h["error"] for h in per_host if "error" in h)
        return out
    deltas = [h["delta"] for h in per_host]
    if any(d <= HOLD_DELTA for d in deltas) or any(h["public_20m"] < HOLD_ABS for h in per_host):
        out["verdict"] = "HOLD"
        out["why"] = "big-win hold: independent cold replay and audit before reporting"
        return out
    if len(per_host) == 1:
        d = deltas[0]
        if d <= NOMINATE:
            v, why = "NOMINATE", "single-host delta <= -0.0006: replicate on the other host"
        elif d >= DISCARD_SINGLE:
            v, why = "DISCARD", "single-host delta >= +0.0005"
        elif kind in HIGH_PRIOR:
            v, why = "REPLICATE", "ambiguous band, high-prior kind: one cross-host replicate"
        else:
            v, why = "DISCARD", "ambiguous band, low-prior kind"
    else:
        mean = sum(deltas) / len(deltas)
        spread = max(deltas) - min(deltas)
        out["mean_delta"], out["disagreement"] = round(mean, 7), round(spread, 7)
        if all(d < 0 for d in deltas) and mean <= ADOPT_MEAN:
            if spread <= ADOPT_DISAGREE:
                v, why = "ADOPT_PENDING_HUMAN_ACK", "both hosts negative, mean <= -0.0010, hosts agree"
            else:
                v, why = "REPLICATE", "passes mean rule but hosts disagree by > 0.0015: third replicate"
        elif mean > DISCARD_MEAN:
            v, why = "DISCARD", "two-host mean > -0.0003"
        elif kind in HIGH_PRIOR and len(runs) < 3:
            v, why = "REPLICATE", "between discard and adopt, high-prior kind: one more replicate"
        else:
            v, why = "DISCARD", "does not meet adoption rule"
    out["verdict"], out["why"] = v, why
    return out


def drift(rows):
    alarms = []
    ctrls = sorted((r for r in rows if r.get("kind") == "control" and invalid_reason(r) is None),
                   key=lambda r: r["timestamp_utc"])
    for i, c in enumerate(ctrls):
        prev = [p for p in ctrls[:i] if p["host"] == c["host"] and p["base_sha256"] == c["base_sha256"]]
        if prev:
            m = sum(p["public_20m"] for p in prev) / len(prev)
            if abs(c["public_20m"] - m) > DRIFT:
                alarms.append({"label": c["label"], "host": c["host"], "value": c["public_20m"],
                               "previous_mean": round(m, 7), "diff": round(c["public_20m"] - m, 7)})
    return {"rules": RULES_VERSION, "drift_alarms": alarms,
            "run_diagnostics": {r["label"]: run_diagnostics(rows, r) for r in rows if r.get("kind") == "control"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--arm")
    g.add_argument("--drift", action="store_true")
    ns = ap.parse_args()
    rows = load(ns.ledger)
    res = drift(rows) if ns.drift else decide_arm(rows, ns.arm)
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
