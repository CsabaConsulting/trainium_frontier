#!/usr/bin/env python3
"""Measure whether short-smoke BPB deltas predict full-run BPB deltas.

This is an evidence report, not an automatic gate editor.  It only joins durable proposal and
verdict JSON already archived under ``research/agentic/experiments``.  Historical proposals that
predate mechanism metadata are classified heuristically and marked as inferred.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

CLASSES = ("throughput", "per_step_quality", "schedule", "capacity", "data_regime")


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def infer_class(proposal: dict) -> tuple[str, bool]:
    declared = str(proposal.get("mechanism_class", ""))
    if declared in CLASSES:
        return declared, False
    proposal_id = str(proposal.get("_proposal_id", "")).lower()
    if any(term in proposal_id for term in ("batch-ramp", "epoch", "shuffle", "context")):
        return "data_regime", True
    if any(term in proposal_id for term in ("bf16", "fp8", "nki", "kernel", "fused")):
        return "throughput", True
    if any(term in proposal_id for term in ("dim64", "no-ngram", "width", "depth", "capacity")):
        return "capacity", True
    if any(term in proposal_id for term in ("warmdown", "cooldown", "flat-lr", "decay", "freeze", "ema", "demon")):
        return "schedule", True
    return "per_step_quality", True


def _finite_delta(verdict: dict) -> float | None:
    value = (verdict.get("metrics") or {}).get("bpb_delta")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def collect(experiments_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not experiments_dir.is_dir():
        return rows
    for directory in sorted(p for p in experiments_dir.iterdir() if p.is_dir()):
        proposal = _load(directory / "proposal.json")
        smoke = _load(directory / "smoke-verdict.json")
        full = _load(directory / "full-verdict.json")
        smoke_delta, full_delta = _finite_delta(smoke), _finite_delta(full)
        if smoke_delta is None or full_delta is None:
            continue
        mechanism, inferred = infer_class({**proposal, "_proposal_id": directory.name})
        rows.append({
            "proposal_id": directory.name,
            "mechanism_class": mechanism,
            "mechanism_class_inferred": inferred,
            "smoke_bpb_delta": smoke_delta,
            "full_bpb_delta": full_delta,
            "smoke_decision": smoke.get("decision"),
            "full_decision": full.get("decision"),
            "full_comparable": full.get("decision") in {"KEEP", "DISCARD"},
        })
    return rows


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        average = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = average
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx <= 0 or dy <= 0:
        return None
    return numerator / math.sqrt(dx * dy)


def summarize(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    groups["all"] = list(rows)
    for row in rows:
        groups[row["mechanism_class"]].append(row)
    result: dict[str, dict] = {}
    for name in ("all", *CLASSES):
        group = groups.get(name, [])
        comparable = [row for row in group if row["full_comparable"]]
        smoke = [row["smoke_bpb_delta"] for row in comparable]
        full = [row["full_bpb_delta"] for row in comparable]
        agreements = [
            (s < 0) == (f < 0) if s != 0 and f != 0 else (s == 0 and f == 0)
            for s, f in zip(smoke, full)
        ]
        rho = _pearson(_rank(smoke), _rank(full)) if group else None
        agreement = (sum(agreements) / len(agreements)) if agreements else None
        if len(comparable) < 6:
            routing = "insufficient_evidence"
        elif agreement is not None and agreement < 0.70:
            routing = "do_not_gate_on_smoke"
        else:
            routing = "smoke_gate_supported"
        result[name] = {
            "n": len(group),
            "comparable_full_n": len(comparable),
            "sign_agreement": agreement,
            "spearman_rho": rho,
            "routing_evidence": routing,
        }
    return {"summary": result, "rows": rows}


def markdown(report: dict) -> str:
    lines = [
        "# Smoke-to-full proxy calibration",
        "",
        "Negative BPB delta is better. Historical mechanism labels may be inferred. This report",
        "does not change any gate automatically; at least six paired outcomes are required before",
        "a class is considered calibrated.",
        "",
        "| Class | Pairs | Comparable full | Clean sign agreement | Clean Spearman rho | Evidence |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, stats in report["summary"].items():
        agreement = "—" if stats["sign_agreement"] is None else f'{stats["sign_agreement"]:.1%}'
        rho = "—" if stats["spearman_rho"] is None else f'{stats["spearman_rho"]:.3f}'
        lines.append(
            f'| {name} | {stats["n"]} | {stats["comparable_full_n"]} | '
            f'{agreement} | {rho} | {stats["routing_evidence"]} |'
        )
    lines.extend(["", "## Paired experiments", ""])
    for row in report["rows"]:
        inferred = " (inferred)" if row["mechanism_class_inferred"] else ""
        lines.append(
            f'- `{row["proposal_id"]}` — {row["mechanism_class"]}{inferred}; '
            f'smoke {row["smoke_bpb_delta"]:+.6f}, full {row["full_bpb_delta"]:+.6f}; '
            f'full verdict `{row["full_decision"]}`.'
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "experiments",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()
    report = summarize(collect(args.experiments_dir))
    print(markdown(report) if args.format == "markdown" else json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
