"""Fail-closed final-week public-BPB screen and confirmation decisions.

This module deliberately does not schedule evaluation or change the active fixed queue.  It defines
the exact artifact schema and the only legal transitions once a separately reviewed canonical base
enables :class:`config.EvaluationPolicy`.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation

from . import config as C


class SelectionPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PairedEvaluation:
    schema_version: int
    population: str  # screen2m | full20m
    context_length: int
    num_tokens: int
    num_bytes: int
    candidate_label: str
    candidate_sha256: str
    candidate_host: str
    candidate_bpb: str
    control_label: str
    control_sha256: str
    control_host: str
    control_bpb: str
    active_sweep_id: str | None

    @classmethod
    def from_dict(cls, raw: dict) -> "PairedEvaluation":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(raw, dict) or set(raw) != expected:
            missing = sorted(expected - set(raw)) if isinstance(raw, dict) else sorted(expected)
            extra = sorted(set(raw) - expected) if isinstance(raw, dict) else []
            raise SelectionPolicyError(f"paired-evaluation schema mismatch; missing={missing} extra={extra}")
        value = cls(**raw)
        value.validate_identity()
        return value

    def validate_identity(self) -> None:
        if self.schema_version != 1:
            raise SelectionPolicyError("paired-evaluation schema_version must be 1")
        if self.population not in {"screen2m", "full20m"}:
            raise SelectionPolicyError("paired-evaluation population is invalid")
        for name in ("candidate_sha256", "control_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise SelectionPolicyError(f"{name} is not a lowercase SHA-256")
        for name in ("candidate_label", "control_label", "candidate_host", "control_host"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise SelectionPolicyError(f"{name} must be nonempty")
        if self.active_sweep_id is not None and (
                not isinstance(self.active_sweep_id, str) or not self.active_sweep_id.strip()):
            raise SelectionPolicyError("active_sweep_id must be null or a nonempty identifier")
        _decimal(self.candidate_bpb, "candidate_bpb")
        _decimal(self.control_bpb, "control_bpb")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SelectionDecision:
    decision: str  # NOMINATE_CONFIRMATION | DISCARD | ADOPT
    reason: str
    deltas: tuple[str, ...]
    mean_delta: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _decimal(value, name: str) -> Decimal:
    if isinstance(value, bool):
        raise SelectionPolicyError(f"{name} must be finite decimal text")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise SelectionPolicyError(f"{name} must be finite decimal text") from exc
    if not parsed.is_finite() or not math.isfinite(float(parsed)):
        raise SelectionPolicyError(f"{name} must be finite")
    return parsed


def _validate_pair(record: PairedEvaluation, policy: C.EvaluationPolicy, population: str) -> Decimal:
    errors = policy.validate()
    if errors:
        raise SelectionPolicyError("invalid evaluation policy: " + "; ".join(errors))
    if not policy.enabled:
        raise SelectionPolicyError("evaluation policy is disabled pending canonical-root authorization")
    expected_tokens = (
        policy.screen_eval_tokens if population == "screen2m" else policy.confirmation_eval_tokens
    )
    expected_bytes = (
        policy.screen_eval_bytes if population == "screen2m" else policy.confirmation_eval_bytes
    )
    if record.population != population:
        raise SelectionPolicyError(f"expected {population} evidence")
    if (record.context_length != policy.context_length or record.num_tokens != expected_tokens
            or record.num_bytes != expected_bytes):
        raise SelectionPolicyError("evaluation context/token/byte population differs from policy")
    if record.control_sha256 != policy.current_base_sha256:
        raise SelectionPolicyError("control is not the current canonical base")
    if record.candidate_sha256 == record.control_sha256:
        raise SelectionPolicyError("candidate and control source identities are identical")
    if record.candidate_host != record.control_host:
        raise SelectionPolicyError("candidate is not paired with a same-host control")
    if policy.current_control_labels_by_host.get(record.control_host) != record.control_label:
        raise SelectionPolicyError("control label is not the authorized current control for this host")
    return _decimal(record.candidate_bpb, "candidate_bpb") - _decimal(record.control_bpb, "control_bpb")


def assess_screen(record: PairedEvaluation, policy: C.EvaluationPolicy) -> SelectionDecision:
    """A screen may nominate a confirmation; it can never return ADOPT or KEEP."""

    delta = _validate_pair(record, policy, "screen2m")
    nominate = _decimal(policy.screen_nominate_delta, "screen_nominate_delta")
    discard = _decimal(policy.screen_discard_delta, "screen_discard_delta")
    rendered = str(delta)
    if delta <= nominate:
        return SelectionDecision(
            "NOMINATE_CONFIRMATION",
            "screen met the nomination boundary; a separate full20m confirmation is still required",
            (rendered,),
        )
    if delta >= discard:
        return SelectionDecision("DISCARD", "screen met the regression discard boundary", (rendered,))
    if (record.active_sweep_id is not None
            and record.active_sweep_id in policy.active_sweep_ids):
        return SelectionDecision(
            "NOMINATE_CONFIRMATION",
            "middle-band screen belongs to an explicitly active sweep; full20m confirmation required",
            (rendered,),
        )
    return SelectionDecision(
        "DISCARD", "middle-band screen is discarded outside a control-plane-authorized active sweep",
        (rendered,),
    )


def assess_confirmation(
    records: list[PairedEvaluation], policy: C.EvaluationPolicy
) -> SelectionDecision:
    """Require two independent negative full20m pairs before ADOPT is possible."""

    if len(records) != policy.adoption_required_full20m_runs:
        raise SelectionPolicyError(
            f"exactly {policy.adoption_required_full20m_runs} full20m runs are required"
        )
    labels = {record.candidate_label for record in records}
    if len(labels) != len(records):
        raise SelectionPolicyError("confirmation candidate labels are not independent")
    candidate_shas = {record.candidate_sha256 for record in records}
    if len(candidate_shas) != 1:
        raise SelectionPolicyError("confirmation runs do not use identical candidate bytes")
    deltas = tuple(_validate_pair(record, policy, "full20m") for record in records)
    rendered = tuple(str(delta) for delta in deltas)
    if any(delta >= 0 for delta in deltas):
        return SelectionDecision(
            "DISCARD", "every full20m confirmation must be strictly negative", rendered,
            str(sum(deltas) / Decimal(len(deltas))),
        )
    mean = sum(deltas) / Decimal(len(deltas))
    threshold = _decimal(policy.adoption_mean_delta, "adoption_mean_delta")
    if mean <= threshold:
        return SelectionDecision(
            "ADOPT",
            "two negative full20m same-host pairs meet the predeclared mean threshold",
            rendered,
            str(mean),
        )
    return SelectionDecision(
        "DISCARD", "full20m mean did not meet the adoption threshold", rendered, str(mean)
    )
