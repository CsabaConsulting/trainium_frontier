#!/usr/bin/env python3
"""Read-only K2 timing arithmetic, not a native/quality/launch approval.

Usage: python3 compare.py CONTROL/rehearsal.json ON/rehearsal.json
The control is the actual completed e1e2 trn22 parent (>=45 boundaries);
ON is the frozen 63925519 candidate (exactly 45). Source/host/argv/runtime,
finite loss, stall/recompile and cold-cache proofs require external receipts.
The historical 105-step batch receipt documents schema only, not a baseline.
JSON decimal metrics are strings; decisions use exact rational arithmetic on
the JSON numbers, without rounding or an epsilon at the 1.03 threshold.
Exit 0 means structurally valid receipts, including a below-threshold result;
exit 2 means invalid input. This helper never writes files or launches jobs.
"""

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re


TOKENS = 35 * 131072
THRESHOLD = Fraction(103, 100)
SOURCE = "63925519ac6c0aab1a1d598e96fef3f4413fbd083d9b6d177901a2e972060db5"


def decimal_text(value):
    with localcontext() as context:
        context.prec = 28
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def inspect(path, candidate=False):
    """Validate one receipt; return serializable report and exact interval."""
    report = {"path": str(Path(path).resolve()), "raw_input_sha256": None,
              "receipt_valid": False, "errors": [], "diagnostics": []}
    errors = report["errors"]
    try:
        raw = Path(path).read_bytes()
        report["raw_input_sha256"] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw, parse_float=Decimal, parse_constant=Decimal,
                          object_pairs_hook=unique_keys)
        if not isinstance(data, dict):
            raise ValueError("receipt must be a JSON object")
    except (OSError, ValueError, UnicodeError) as exc:
        errors.append(str(exc))
        return report, None

    def number(value, label):
        if (isinstance(value, bool) or not isinstance(value, (int, Decimal))
                or (isinstance(value, Decimal) and not value.is_finite())):
            errors.append(label + " must be a finite JSON number")
            return None
        return Fraction(value)

    if data.get("mode") != "train":
        errors.append("mode must be train")
    if type(data.get("returncode")) is not int or data["returncode"] != 0:
        errors.append("returncode must be integer 0")
    budget_ok = data.get("charged_through_process_exit_within_budget") is True
    report["official_budget_receipt_true"] = budget_ok
    if not budget_ok:
        errors.append("official through-exit budget flag must be true")

    observations = data.get("observations")
    times = []
    if not isinstance(observations, list):
        errors.append("observations must be a list")
        observations = []
    count = len(observations)
    report["observed_boundary_count"] = count
    if count < 45 or (candidate and count != 45):
        errors.append("candidate needs exactly 45 boundaries" if candidate
                      else "control needs at least 45 boundaries")
    if type(data.get("step_boundaries")) is not int or data["step_boundaries"] != count:
        errors.append("step_boundaries must match the observation count")
    ties = []
    for index, item in enumerate(observations):
        if not isinstance(item, dict):
            errors.append("observation %d must be an object" % index)
            times.append(None)
            continue
        if type(item.get("step")) is not int or item["step"] != index:
            errors.append("observation %d must have sequential step %d" % (index, index))
        value = number(item.get("observed_seconds"), "time%d" % index)
        times.append(value)
        if value is None:
            continue
        if value < 0:
            errors.append("time%d must be nonnegative" % index)
        if index and times[index - 1] is not None:
            if value < times[index - 1]:
                errors.append("timestamps decrease at step %d" % index)
            elif value == times[index - 1]:
                ties.append(index)
    report["tied_boundary_steps"] = ties
    if ties:
        report["diagnostics"].append(
            "Zero/tied adjacent durations require observer-resolution/stall review; "
            "the arithmetic threshold alone is not a speed qualification.")

    interval = None
    if count >= 45 and times[9] is not None and times[44] is not None:
        interval = times[44] - times[9]
        report.update(t9_seconds=decimal_text(times[9]),
                      t44_seconds=decimal_text(times[44]),
                      selected_interval_seconds=decimal_text(interval))
        if interval <= 0:
            errors.append("selected t44-t9 interval must be positive")
            interval = None

    wall = number(data.get("wall_seconds"), "wall_seconds")
    allowance = number(data.get("startup_allowance_observed"), "startup_allowance_observed")
    charged = number(data.get("charged_through_process_exit"), "charged_through_process_exit")
    if wall is not None and (wall < 0 or any(t is not None and t > wall for t in times)):
        errors.append("wall_seconds must be nonnegative and cover every boundary")
    if count >= 5 and times[4] is not None:
        startup = times[4]
        capped = min(startup, Fraction(900))
        report.update(first_five_uncapped_startup_seconds=decimal_text(startup),
                      capped_startup_allowance_seconds=decimal_text(capped),
                      startup_excess_seconds=decimal_text(max(startup - 900, 0)))
        if startup <= 0:
            errors.append("first-five startup must be positive")
        # Receipt subtraction used binary floats; this tolerance is accounting
        # consistency only, never applied to timing threshold or budget limit.
        if allowance is not None and abs(allowance - capped) > Fraction(1, 1000000):
            errors.append("reported startup allowance differs from min(t4,900)")
        if wall is not None:
            derived_charge = wall - capped
            report["derived_charged_through_exit_seconds"] = decimal_text(derived_charge)
            if not 0 <= derived_charge <= 1800:
                errors.append("derived through-exit charge outside 0..1800 seconds")
            if charged is not None and abs(charged - derived_charge) > Fraction(1, 1000000):
                errors.append("reported through-exit charge disagrees with wall minus allowance")
    if charged is not None:
        report["receipt_charged_through_exit_seconds"] = decimal_text(charged)
        if not 0 <= charged <= 1800:
            errors.append("reported through-exit charge outside 0..1800 seconds")
    checkpoint_size = data.get("checkpoint_bytes")
    checkpoint_sha = data.get("checkpoint_sha256")
    if type(checkpoint_size) is not int or checkpoint_size <= 0:
        errors.append("checkpoint_bytes must be a positive integer")
    if not isinstance(checkpoint_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha):
        errors.append("checkpoint_sha256 must contain 64 hexadecimal digits")
    report["checkpoint_receipt"] = {"bytes": checkpoint_size if type(checkpoint_size) is int else None,
                                    "sha256": checkpoint_sha if isinstance(checkpoint_sha, str) else None,
                                    "checkpoint_file_independently_verified": False}
    report["receipt_valid"] = not errors
    if not errors and interval is not None:
        report["fixed_window_tokens_per_second"] = decimal_text(Fraction(TOKENS) / interval)
        return report, interval
    return report, None


def compare(control_path, candidate_path):
    control, control_interval = inspect(control_path)
    candidate, candidate_interval = inspect(candidate_path, candidate=True)
    valid = control["receipt_valid"] and candidate["receipt_valid"]
    errors = []
    if Path(control_path).resolve() == Path(candidate_path).resolve():
        errors.append("control and candidate must be distinct actual receipts")
        valid = False
    ratio = control_interval / candidate_interval if valid else None
    return {
        "schema": "k2_fixed_window_timing_v1", "receipts_valid": valid,
        "errors": errors, "statistic": "35*131072/(t44-t9)",
        "window_selection": "precommitted; no fastest-window selection",
        "control": control, "candidate": candidate,
        "candidate_to_control_rate_ratio": decimal_text(ratio) if ratio is not None else None,
        "threshold": "1.03", "timing_threshold_pass": ratio >= THRESHOLD if ratio is not None else None,
        "timing_diagnostic_review_required": bool(control.get("tied_boundary_steps")
                                                 or candidate.get("tied_boundary_steps")),
        "context_only_expected_candidate_source_sha256": SOURCE,
        "source_identity_verified_by_this_helper": False,
        "native_correctness": "NOT_ESTABLISHED", "finite_loss": "NOT_ESTABLISHED",
        "cold_cache_proof": "NOT_ESTABLISHED: require packet/runtime receipts; "
                            "fresh_empty_sdk_caches alone is not proof. Parent control uses reused namespaces.",
        "full_launch_approval": "NOT_ESTABLISHED", "final_bpb": None,
        "external_checks_required": ["actual matched host/runtime/seed42/shape/argv/source",
                                     "native correctness and finite loss",
                                     "no recompile/stall in fixed interval",
                                     "cold-cache packet, creation and rank-environment evidence",
                                     "checkpoint retrieval/authentication and applicable launch gates"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    result = compare(args.control, args.candidate)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["receipts_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
