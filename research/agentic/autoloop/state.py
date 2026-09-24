"""Restart-safe state: one atomic JSON file plus one directory per proposal.

Stages move only forward:

    proposed -> verified | rejected
    verified -> staged:<tier> -> launched:<tier> -> collected:<tier> -> analyzed:<tier>
    analyzed:smoke -> (promote) staged:full | closed
    analyzed:full  -> closed

On restart the controller reads the stage and resumes: a ``launched`` tier is polled, never
relaunched; a ``collected`` tier is analyzed; an ``analyzed`` tier is archived/promoted.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config as C

SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text())


@dataclass
class Store:
    root: Path = C.AGENT_DIR
    state_path: Path = C.STATE_FILE
    experiments_dir: Path = C.EXPERIMENTS_DIR
    utilization_journal_path: Path | None = None
    state: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.utilization_journal_path is None:
            self.utilization_journal_path = self.root / "runtime" / "idle-utilization.jsonl"
        self.state = read_json(self.state_path, None) or self._fresh()
        if self.state.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"state schema {self.state.get('schema_version')} != {SCHEMA_VERSION}; refusing to guess")

    def _fresh(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "controller_id": f"autoloop-{uuid.uuid4().hex[:12]}",
            "proposals": {},
            "usage": {},
            "box_failures": {},
            "runs_total": 0,
            "breaker": None,
            "cycles": 0,
            "last_cycle": None,
        }

    def save(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def record_host_utilization(
        self, *, box: str, host: str, status: str, reason: str, observed_at: str
    ) -> float | None:
        """Append one fsynced utilization sample and return the current idle streak in seconds."""

        if status not in {"busy", "idle", "unknown"}:
            raise ValueError(f"invalid utilization status {status!r}")
        if not box or not host or not reason:
            raise ValueError("utilization samples require box, host, and reason")
        try:
            observed = datetime.strptime(observed_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError as exc:
            raise ValueError("utilization timestamp must be UTC YYYY-MM-DDTHH:MM:SSZ") from exc
        tracking = self.state.setdefault("host_utilization", {}).setdefault(box, {})
        if status == "idle":
            if tracking.get("last_status") != "idle" or not tracking.get("idle_since"):
                tracking["idle_since"] = observed_at
            idle_since = datetime.strptime(
                tracking["idle_since"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
            idle_seconds: float | None = max(0.0, (observed - idle_since).total_seconds())
        else:
            tracking["idle_since"] = None
            idle_seconds = None
        tracking.update({
            "host": host,
            "last_status": status,
            "last_reason": reason,
            "last_observed_at": observed_at,
        })
        sample = {
            "schema_version": 1,
            "timestamp": observed_at,
            "box": box,
            "host": host,
            "inventory_status": status,
            "reason": reason,
            "idle_since": tracking.get("idle_since"),
            "idle_seconds": idle_seconds,
        }
        assert self.utilization_journal_path is not None
        path = self.utilization_journal_path
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "ab") as journal:
            payload = (json.dumps(sample, sort_keys=True) + "\n").encode()
            journal.write(payload)
            journal.flush()
            os.fsync(journal.fileno())
        self.save()
        return idle_seconds

    # ---- proposals ----
    @property
    def controller_id(self) -> str:
        return self.state["controller_id"]

    def proposal_dir(self, proposal_id: str) -> Path:
        return self.experiments_dir / proposal_id

    def proposals(self) -> dict:
        return self.state["proposals"]

    def add_proposal(self, proposal: dict) -> None:
        pid = proposal["proposal_id"]
        if pid in self.proposals():
            raise RuntimeError(f"proposal {pid} already tracked")
        self.proposals()[pid] = {"stage": "proposed", "tiers": {}, "verdict": None, "box": None, "updated_at": utc_now()}
        atomic_write_json(self.proposal_dir(pid) / "proposal.json", proposal)
        self.save()

    def track_existing_proposal(self, proposal_id: str, *, box: str, candidate_commit: str,
                                comparison_role: str) -> None:
        """Track a reviewed, already-versioned fixed-queue proposal without rewriting it."""
        if proposal_id in self.proposals():
            return
        proposal_path = self.proposal_dir(proposal_id) / "proposal.json"
        if proposal_path.is_symlink() or not proposal_path.is_file():
            raise RuntimeError(f"fixed proposal artifact missing: {proposal_id}")
        proposal = json.loads(proposal_path.read_text())
        if proposal.get("proposal_id") != proposal_id:
            raise RuntimeError(f"fixed proposal identity mismatch: {proposal_id}")
        self.proposals()[proposal_id] = {
            "stage": "verified", "tiers": {}, "verdict": None, "box": box,
            "candidate_commit": candidate_commit, "comparison_role": comparison_role,
            "fixed_queue": True, "updated_at": utc_now(),
        }
        self.save()

    def activate_fixed_plan(self, *, plan_id: str, plan_sha256: str,
                            parked_proposals: list[dict]) -> None:
        """Losslessly migrate the runtime to the fixed queue and durably park old work.

        Existing breakers, usage, run counts, histories, and unrelated proposal records are kept.
        A parked proposal can never become launchable merely because a breaker is later cleared.
        """
        requested = {"plan_id": plan_id, "plan_sha256": plan_sha256, "mode": "fixed_full_only"}
        active = self.state.get("active_plan")
        if active is not None and active != requested:
            raise RuntimeError("runtime is already bound to a different fixed plan")
        for parked in parked_proposals:
            pid = parked["proposal_id"]
            expected = parked["expected_previous_stage"]
            rec = self.proposals().get(pid)
            if rec is None:
                rec = {
                    "stage": expected, "tiers": {}, "verdict": None, "box": None,
                    "candidate_commit": parked["candidate_commit"], "updated_at": utc_now(),
                    "reconstructed_for_plan_migration": True,
                }
                self.proposals()[pid] = rec
            if rec.get("stage") == "parked":
                if rec.get("parked_by_plan") != plan_id or rec.get("parked_from") != expected:
                    raise RuntimeError(f"parked record for {pid} does not match fixed plan")
                continue
            if rec.get("stage") != expected:
                raise RuntimeError(
                    f"cannot park {pid}: stage {rec.get('stage')!r} != expected {expected!r}"
                )
            if rec.get("candidate_commit") not in (None, parked["candidate_commit"]):
                raise RuntimeError(f"cannot park {pid}: candidate commit changed")
            rec.update({
                "stage": "parked", "parked_from": expected, "parked_by_plan": plan_id,
                "launch_authorized": False, "candidate_commit": parked["candidate_commit"],
                "updated_at": utc_now(),
            })
        self.state["active_plan"] = requested
        self.save()

    def set_stage(self, proposal_id: str, stage: str, **extra) -> None:
        rec = self.proposals()[proposal_id]
        rec["stage"] = stage
        rec["updated_at"] = utc_now()
        rec.update(extra)
        self.save()

    def tier(self, proposal_id: str, tier: str) -> dict:
        return self.proposals()[proposal_id]["tiers"].setdefault(tier, {})

    def write_artifact(self, proposal_id: str, name: str, data) -> Path:
        path = self.proposal_dir(proposal_id) / name
        atomic_write_json(path, data)
        return path

    def read_artifact(self, proposal_id: str, name: str):
        return read_json(self.proposal_dir(proposal_id) / name)

    def used_proposal_ids(self) -> set[str]:
        ids = set(self.proposals())
        if self.experiments_dir.is_dir():
            ids |= {p.name for p in self.experiments_dir.iterdir() if p.is_dir()}
        if C.PROPOSALS_DIR.is_dir():
            ids |= {p.stem for p in C.PROPOSALS_DIR.glob("*.json")}
        return ids

    def used_labels(self) -> set[str]:
        labels: set[str] = set()
        for rec in self.proposals().values():
            for t in rec["tiers"].values():
                if t.get("label"):
                    labels.add(t["label"])
        return labels

    # ---- usage / caps ----
    def usage_today(self) -> dict:
        return self.state["usage"].setdefault(
            today(), {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0, "runs": 0}
        )

    def record_llm(self, role: str, input_tokens: int, output_tokens: int, caps: C.Caps) -> None:
        u = self.usage_today()
        u["llm_calls"] += 1
        u["input_tokens"] += input_tokens
        u["output_tokens"] += output_tokens
        u["usd"] += input_tokens / 1000 * caps.usd_per_1k_input_tokens + output_tokens / 1000 * caps.usd_per_1k_output_tokens
        u.setdefault("by_role", {}).setdefault(role, 0)
        u["by_role"][role] += 1
        self.save()

    def record_run(self) -> None:
        self.usage_today()["runs"] += 1
        self.state["runs_total"] += 1
        self.save()

    def llm_calls_allowed(self, caps: C.Caps, cycle_calls: int, needed: int = 1) -> tuple[bool, str]:
        if needed < 1:
            raise ValueError("needed must be positive")
        u = self.usage_today()
        if u["llm_calls"] + needed > caps.max_llm_calls_per_day:
            return False, f"daily LLM call cap reached or has fewer than {needed} calls remaining"
        if u["usd"] >= caps.max_llm_usd_per_day:
            return False, "daily LLM spend cap reached"
        if cycle_calls + needed > caps.max_llm_calls_per_cycle:
            return False, f"per-cycle LLM call cap reached or has fewer than {needed} calls remaining"
        return True, ""

    def runs_allowed(self, caps: C.Caps) -> tuple[bool, str]:
        if self.usage_today()["runs"] >= caps.max_runs_per_day:
            return False, "daily run cap reached"
        if self.state["runs_total"] >= caps.max_runs_total:
            return False, "total run cap reached"
        return True, ""

    # ---- breakers ----
    def open_breaker(self, reason: str, **detail) -> None:
        self.state["breaker"] = {"reason": reason, "at": utc_now(), **detail}
        self.save()

    def breaker(self) -> dict | None:
        return self.state.get("breaker")

    def clear_breaker_for_fixed_plan(self, *, receipt_commit: str, receipt_path: str,
                                     receipt_sha256: str, cleared_by_role: str) -> None:
        """Explicit Operator recovery transition; never called automatically by the controller."""
        if self.breaker() is None:
            raise RuntimeError("fixed-plan breaker clearance requires an existing breaker")
        expected = (
            C.BREAKER_CLEANUP_RECEIPT_COMMIT, C.BREAKER_CLEANUP_RECEIPT_PATH,
            C.BREAKER_CLEANUP_RECEIPT_SHA256, "operator",
        )
        if (receipt_commit, receipt_path, receipt_sha256, cleared_by_role) != expected:
            raise RuntimeError("fixed-plan breaker clearance receipt differs from pinned evidence")
        self.state["fixed_breaker_clearance"] = {
            "receipt_commit": receipt_commit, "receipt_path": receipt_path,
            "receipt_sha256": receipt_sha256, "cleared_by_role": cleared_by_role,
            "cleared_at": utc_now(), "previous_breaker": self.state["breaker"],
        }
        self.state["breaker"] = None
        self.save()

    def box_failure(self, box: str) -> int:
        n = self.state["box_failures"].get(box, 0) + 1
        self.state["box_failures"][box] = n
        self.save()
        return n

    def box_success(self, box: str) -> None:
        self.state["box_failures"][box] = 0
        self.save()
