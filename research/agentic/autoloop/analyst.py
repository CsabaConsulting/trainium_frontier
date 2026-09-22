"""Deterministic matched-control analysis. An optional LLM Analyst may add prose, never a verdict."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from . import config as C

CORRECTNESS_KEYS = ("status", "over_budget", "causality_passed", "public_val_bpb", "eval_over_timeout", "num_steps", "training_seconds")


@dataclass
class Verdict:
    decision: str  # KEEP | DISCARD | REPEAT | CONFOUNDED | FAILED
    tier: str
    reasons: list[str]
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _f(row: dict, key: str) -> float | None:
    v = row.get(key)
    if isinstance(v, bool) or v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def correctness_failures(row: dict | None, checkpoint_present: bool, runner_rc: int | None) -> list[str]:
    fails: list[str] = []
    if row is None:
        return ["no ledger row retrieved for the label"]
    if runner_rc not in (0, None):
        fails.append(f"runner exit code {runner_rc}")
    if row.get("status") != "keep":
        fails.append(f"ledger status {row.get('status')!r}")
    if row.get("over_budget") is True:
        fails.append("over_budget")
    if row.get("causality_passed") is not True:
        fails.append("causality_passed is not True")
    if row.get("eval_over_timeout") is True:
        fails.append("eval_over_timeout")
    if _f(row, "public_val_bpb") is None:
        fails.append("public_val_bpb missing or non-finite")
    if row.get("git_dirty") is True and not row.get("_dirt_explained", False):
        fails.append("git_dirty=True and not explained by the run's own log/ledger writes")
    if not checkpoint_present:
        fails.append("final.pt missing")
    steps = _f(row, "num_steps")
    if not steps or steps <= 0:
        fails.append("num_steps missing")
    return fails


def step_rate(row: dict) -> float | None:
    steps, secs = _f(row, "num_steps"), _f(row, "training_seconds")
    if steps and secs and secs > 0:
        return steps / secs
    return None


def compare(
    *,
    tier: str,
    candidate: dict | None,
    control: dict | None,
    candidate_ckpt: bool,
    control_ckpt: bool,
    candidate_rc: int | None,
    caps: C.Caps,
    smoke_gate: dict | None = None,
) -> Verdict:
    reasons: list[str] = []
    metrics: dict = {}
    cand_fail = correctness_failures(candidate, candidate_ckpt, candidate_rc)
    if cand_fail:
        return Verdict("FAILED", tier, [f"candidate: {f}" for f in cand_fail], metrics)
    ctrl_fail = correctness_failures(control, control_ckpt, None)
    if ctrl_fail:
        return Verdict("CONFOUNDED", tier, [f"control: {f}" for f in ctrl_fail], metrics)
    assert candidate and control
    for key in ("world_size", "lnc", "seq_len", "max_train_seconds"):
        if candidate.get(key) != control.get(key):
            reasons.append(f"{key} differs between candidate and control")
    if reasons:
        return Verdict("CONFOUNDED", tier, reasons, metrics)
    cand_rate, ctrl_rate = step_rate(candidate), step_rate(control)
    metrics.update(
        {
            "candidate_bpb": _f(candidate, "public_val_bpb"),
            "control_bpb": _f(control, "public_val_bpb"),
            "bpb_delta": (_f(candidate, "public_val_bpb") or 0) - (_f(control, "public_val_bpb") or 0),
            "candidate_steps": _f(candidate, "num_steps"),
            "control_steps": _f(control, "num_steps"),
            "candidate_step_rate": cand_rate,
            "control_step_rate": ctrl_rate,
            "candidate_params_M": _f(candidate, "num_params_M"),
            "control_params_M": _f(control, "num_params_M"),
        }
    )
    if tier == "smoke":
        if cand_rate is None or ctrl_rate is None:
            return Verdict("CONFOUNDED", tier, ["step rate unavailable"], metrics)
        smoke_gate = smoke_gate or {}
        gate_ratio = float(smoke_gate.get("min_step_rate_ratio", 0.0))
        floor = max(ctrl_rate * (1 - caps.smoke_step_rate_tolerance), ctrl_rate * gate_ratio)
        metrics["step_rate_floor"] = floor
        metrics["smoke_gate"] = smoke_gate
        min_steps = smoke_gate.get("min_candidate_steps")
        candidate_steps = metrics["candidate_steps"]
        if min_steps is not None and candidate_steps < int(min_steps):
            return Verdict(
                "DISCARD",
                tier,
                [f"smoke steps {candidate_steps:.0f} < proposal minimum {int(min_steps)}"],
                metrics,
            )
        if cand_rate < floor:
            return Verdict("DISCARD", tier, [f"smoke step rate {cand_rate:.4f} < floor {floor:.4f} steps/s"], metrics)
        max_bpb_delta = smoke_gate.get("max_bpb_delta")
        if max_bpb_delta is not None and metrics["bpb_delta"] > float(max_bpb_delta):
            return Verdict(
                "DISCARD",
                tier,
                [f"smoke public_val_bpb delta {metrics['bpb_delta']:+.5f} exceeds proposal maximum {float(max_bpb_delta):+.5f}"],
                metrics,
            )
        return Verdict("KEEP", tier, ["correctness gates passed; step rate within tolerance; eligible for full run"], metrics)
    # full tier
    cs, ks = metrics["candidate_steps"], metrics["control_steps"]
    if ks and abs(cs - ks) / ks > caps.steps_comparability_frac:
        return Verdict("CONFOUNDED", tier, [f"steps differ by more than {caps.steps_comparability_frac:.0%} ({cs} vs {ks})"], metrics)
    delta = metrics["bpb_delta"]
    if delta <= caps.keep_bpb_delta:
        return Verdict("KEEP", tier, [f"public_val_bpb improved by {-delta:.5f} (threshold {-caps.keep_bpb_delta:.5f})"], metrics)
    return Verdict("DISCARD", tier, [f"public_val_bpb delta {delta:+.5f} did not meet threshold {caps.keep_bpb_delta:+.5f}"], metrics)


def verdict_markdown(proposal: dict, manifest: dict, verdict: Verdict, control_label: str | None, hostname: str) -> str:
    m = verdict.metrics
    lines = [
        f"# Verdict: {verdict.decision} — {proposal['proposal_id']} ({verdict.tier})",
        "",
        f"- Hypothesis: {proposal['hypothesis']}",
        f"- Single causal change: {proposal['single_causal_change']}",
        f"- Flags: `{' '.join(manifest['train_flags'])}`",
        f"- Commit: `{manifest['commit']}`",
        f"- Box: {manifest['box']} (`{hostname}`), shape LNC={manifest['lnc']} nproc={manifest['nproc_per_node']}, seed {manifest['seed']}",
        f"- Label: `{manifest['label']}`; control label: `{control_label}`",
        f"- Budget: num_steps={manifest['num_steps']} max_train_seconds={manifest['max_train_seconds']}",
        f"- NEURON_CC_FLAGS: `{manifest['env'].get('NEURON_CC_FLAGS')}`",
        "",
        "## Reasons",
        *[f"- {r}" for r in verdict.reasons],
        "",
        "## Metrics (public dev shard, evaluate_bpb, same box/day/commit)",
        "",
        "| metric | candidate | control |",
        "|---|---:|---:|",
        f"| public_val_bpb | {m.get('candidate_bpb')} | {m.get('control_bpb')} |",
        f"| num_steps | {m.get('candidate_steps')} | {m.get('control_steps')} |",
        f"| steps/s | {m.get('candidate_step_rate')} | {m.get('control_step_rate')} |",
        f"| params (M) | {m.get('candidate_params_M')} | {m.get('control_params_M')} |",
        "",
        "These are development measurements, not leaderboard scores.",
    ]
    return "\n".join(lines) + "\n"
