#!/usr/bin/env python3
"""Deterministic verdicts for the v2 research loop. No LLM judgment in the critical path.

Input: the append-only ledger (JSON Lines, one record per finished+evaluated run; schema in
workflow/templates/run-record.json). Output: a verdict JSON for one arm, or a control-drift report.

  python workflow/tools/decide.py --ledger research/v2/ledger.jsonl --arm A2
  python workflow/tools/decide.py --ledger research/v2/ledger.jsonl --drift

Rules (v2-2026-09-24; change only with human approval, and bump RULES_VERSION):
  valid run      : status=="evaluated", over_budget False, causality_passed not False, finite BPB,
                   eval_tokens==20_971_520, eval_seq_len==1024, git_dirty False
  control        : kind=="control", same host, same base_sha256, valid; pooled mean of up to the
                   last 3 controls on that host for that base (>=1 required, >=2 preferred)
  step ratio     : arm/control steps in [0.90, 1.10] unless kind=="throughput_arm" (else CONFOUNDED)
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

RULES_VERSION = "v2-2026-09-24"
EVAL_TOKENS, EVAL_SEQ = 20_971_520, 1024
NOMINATE, DISCARD_SINGLE = -0.0006, 0.0005
ADOPT_MEAN, ADOPT_DISAGREE, DISCARD_MEAN = -0.0010, 0.0015, -0.0003
HOLD_DELTA, HOLD_ABS = -0.010, 0.975
STEP_WINDOW = (0.90, 1.10)
DRIFT = 0.0010
HIGH_PRIOR = {"table_arm", "data_order_arm"}


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


def host_delta(rows, run):
    ctrls = controls_for(rows, run["host"], run["base_sha256"])
    if not ctrls:
        return {"host": run["host"], "label": run["label"], "error": "no valid same-host control for this base"}
    cm = sum(c["public_20m"] for c in ctrls) / len(ctrls)
    cs = sum(c["steps"] for c in ctrls) / len(ctrls)
    ratio = run["steps"] / cs if cs else float("nan")
    return {"host": run["host"], "label": run["label"], "public_20m": run["public_20m"],
            "controls": [c["label"] for c in ctrls], "control_mean": round(cm, 7),
            "delta": round(run["public_20m"] - cm, 7), "step_ratio": round(ratio, 4),
            "pooled_controls": len(ctrls)}


def decide_arm(rows, arm_id):
    runs = [r for r in rows if r.get("arm_id") == arm_id and r.get("kind") != "control"]
    bad = {r["label"]: invalid_reason(r) for r in runs if invalid_reason(r)}
    runs = [r for r in runs if r["label"] not in bad]
    out = {"rules": RULES_VERSION, "arm_id": arm_id, "invalid_runs": bad}
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
    if kind != "throughput_arm" and any(not (STEP_WINDOW[0] <= h["step_ratio"] <= STEP_WINDOW[1]) for h in per_host):
        out["verdict"] = "CONFOUNDED"
        out["why"] = f"step ratio outside {STEP_WINDOW}; not a same-budget comparison"
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
    return {"rules": RULES_VERSION, "drift_alarms": alarms}


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
