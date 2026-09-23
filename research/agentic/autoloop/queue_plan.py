"""Strict loader for the human-authorized dense1024 full-only queue.

This is deliberately not a general scheduler.  Any missing field, reordered arm, unverified
proposal, or byte mismatch stops the queue before remote inventory or launch.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import config as C


class QueuePlanError(RuntimeError):
    pass


EXPECTED_PLAN_ID = "20260923-dense6x1024-lean-public-bpb-c1024"
EXPECTED_IDS = (
    "20260923-dense6x1024-lean-control-c1024",
    "20260923-dense6x1024-matrix-lr-0025",
    "20260923-dense6x1024-matrix-lr-0015",
    "20260923-dense6x1024-warmdown-060",
    "20260923-dense6x1024-warmdown-085",
    "20260923-dense6x1024-total-batch-262144",
    "20260923-dense6x1024-max-train-seconds-1080",
)
EXPECTED_CHANGES = (
    None,
    ("MATRIX_LR", 0.02, 0.025),
    ("MATRIX_LR", 0.02, 0.015),
    ("WARMDOWN_RATIO", 0.75, 0.6),
    ("WARMDOWN_RATIO", 0.75, 0.85),
    ("TOTAL_BATCH_SIZE", 131072, 262144),
    ("MAX_TRAIN_SECONDS", 1800, 1080),
)
EXPECTED_REFS = (
    ("4ed89f36c437ab75b0bf63cea80a3871f547de34", "39e38e9fc3a9107fe933414286852d98e81c2723", C.AUTHORITATIVE_TRAIN_SHA256),
    ("1fe54026f16f0f4af79da30b3106a9873a11a24d", C.AUTHORITATIVE_COMMIT, "abdc8372808489a2bdf94885922aa12e7173b5a1dc4d91cab1bd874d2573add8"),
    ("f84dfad3538f5b5c04dc50a9ec52ba62574493c8", C.AUTHORITATIVE_COMMIT, "78643cfec5af607f9d2dccc33698c5b3ff9a48ec1fa0a872547bf11b9fbdeb19"),
    ("3ab583509e553fd3ed4b1b42dfe49e8b9bc2a23d", C.AUTHORITATIVE_COMMIT, "983754ec3453fea07f7fb66dc44614d3e38d3b80e8a5160c1e570110bc8c869d"),
    ("69ff20a007cd2f841ccd99a7429808d2d4513d49", C.AUTHORITATIVE_COMMIT, "4eee680d95130f054ebd2191fd544ce6abc93e32f9563340e91c067907b1c183"),
    ("e488dca2d29ce306d0c71d2ce3e0ebe5133777c6", C.AUTHORITATIVE_COMMIT, "b0de2e77b02bea775f4cef8c39a88b44bee9a671d39fae9b0ee64c374c158cc2"),
    ("4c6ec085b7161c96bddaa1d80f516d3c7b3d6c99", C.AUTHORITATIVE_COMMIT, "b2a2b8fa5d3b41723ae89fddb7f929a902f161bb2bd41f3f47144b48f499eb18"),
)


def _regular_under_root(root: Path, value: str) -> Path:
    path = root / value
    if path.is_symlink() or not path.is_file():
        raise QueuePlanError(f"plan path is not a regular file: {value}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise QueuePlanError(f"plan path escapes repository: {value}") from exc
    return path


@dataclass(frozen=True)
class QueueEntry:
    position: int
    proposal_id: str
    role: str
    box: str | None
    proposal_path: str
    verification_path: str
    commit: str
    parent_commit: str
    train_sha256: str
    constant_change: dict | None


@dataclass(frozen=True)
class FixedQueuePlan:
    path: Path
    sha256: str
    raw: dict
    entries: tuple[QueueEntry, ...]

    @classmethod
    def load(cls, path: Path = C.QUEUE_PLAN_FILE, root: Path = C.ROOT) -> "FixedQueuePlan":
        if path.is_symlink() or not path.is_file():
            raise QueuePlanError("fixed queue plan must be a regular file")
        payload = path.read_bytes()
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise QueuePlanError(f"invalid queue JSON: {exc}") from exc
        if raw.get("schema_version") != 1 or raw.get("plan_id") != EXPECTED_PLAN_ID:
            raise QueuePlanError("unknown fixed queue schema/identity")
        if raw.get("mode") != "fixed_full_only":
            raise QueuePlanError("queue is not fixed_full_only")
        control = raw.get("authoritative_control", {})
        if control.get("commit") != C.AUTHORITATIVE_COMMIT:
            raise QueuePlanError("queue control commit differs from pinned config")
        if control.get("train_sha256") != C.AUTHORITATIVE_TRAIN_SHA256:
            raise QueuePlanError("queue control train hash differs from pinned bytes")
        if raw.get("exact_train_argv") != list(C.EXACT_DENSE1024_TRAIN_ARGV):
            raise QueuePlanError("queue train argv differs from authorized argv")
        ev = raw.get("evaluation", {})
        if (ev.get("selection_metric"), ev.get("seq_len"), ev.get("max_tokens"),
                ev.get("expected_num_tokens"), ev.get("expected_num_bytes")) != (
                    "public_val_bpb", C.EXACT_DENSE1024_EVAL_CONTEXT,
                    C.EXACT_DENSE1024_EVAL_TOKENS, C.EXACT_DENSE1024_EVAL_TOKENS,
                    C.EXACT_DENSE1024_EVAL_BYTES):
            raise QueuePlanError("queue evaluation population differs from pinned contract")
        policy = raw.get("launch_policy", {})
        required_policy = {
            "tiers": ["full"], "general_research_disabled": True, "smoke_disabled": True,
            "first_cohort_requires_all_verified": True,
            "first_cohort_requires_both_preflight_clean": True,
            "first_cohort_launch": "concurrent_across_assigned_hosts",
            "cross_host_throughput_comparison_allowed": False,
            "later_entries_strictly_ordered": True,
        }
        if policy != required_policy:
            raise QueuePlanError("fixed launch policy changed")
        rows = raw.get("queue")
        if not isinstance(rows, list) or tuple(r.get("proposal_id") for r in rows) != EXPECTED_IDS:
            raise QueuePlanError("queue membership/order changed")
        if raw.get("first_cohort") != list(EXPECTED_IDS[:2]):
            raise QueuePlanError("first cohort changed")
        recovery = raw.get("breaker_recovery", {})
        if recovery != {
            "automatic_clear_allowed": False,
            "operator_receipt_commit": C.BREAKER_CLEANUP_RECEIPT_COMMIT,
            "operator_receipt_path": C.BREAKER_CLEANUP_RECEIPT_PATH,
            "operator_receipt_sha256": C.BREAKER_CLEANUP_RECEIPT_SHA256,
            "pushed_branch": "agent/control-plane-20260921",
            "requires_explicit_operator_state_transition": True,
        }:
            raise QueuePlanError("fixed breaker recovery contract changed")
        if raw.get("compile_cache_cleanup") != {
            "scope": "exact_unique_label_owned_train_and_eval_cache_paths_only",
            "only_after_archive_commit_verified_on_configured_remote": True,
            "reject_symlink_nonregular_or_live_process_reference": True,
            "immutable_cleanup_receipt_must_be_retrieved_hashed_committed_pushed_and_verified": True,
        }:
            raise QueuePlanError("fixed compile-cache cleanup contract changed")
        entries: list[QueueEntry] = []
        for position, (row, expected_change, expected_ref) in enumerate(
                zip(rows, EXPECTED_CHANGES, EXPECTED_REFS)):
            if row.get("position") != position:
                raise QueuePlanError("queue positions must be contiguous")
            for key in ("proposal_path", "verification_path", "commit", "parent_commit", "train_sha256"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise QueuePlanError(f"queue entry {position} lacks {key}")
            if not re.fullmatch(r"[0-9a-f]{40}", row["commit"]):
                raise QueuePlanError(f"queue entry {position} commit is not a full SHA")
            if not re.fullmatch(r"[0-9a-f]{64}", row["train_sha256"]):
                raise QueuePlanError(f"queue entry {position} train hash is not SHA-256")
            if (row["commit"], row["parent_commit"], row["train_sha256"]) != expected_ref:
                raise QueuePlanError(f"queue entry {position} commit/parent/train fingerprint changed")
            expected_path = f"research/agentic/experiments/{row['proposal_id']}/proposal.json"
            expected_verification = f"research/agentic/experiments/{row['proposal_id']}/verification.json"
            if (row["proposal_path"], row["verification_path"]) != (expected_path, expected_verification):
                raise QueuePlanError(f"queue entry {position} proposal/verification path changed")
            proposal_path = _regular_under_root(root, row["proposal_path"])
            proposal = json.loads(proposal_path.read_text())
            if (proposal.get("proposal_id") != row["proposal_id"]
                    or proposal.get("status") != "human_approved_pending_independent_verification"
                    or proposal.get("tier_policy") != "full_only"
                    or proposal.get("assigned_box") != row.get("box")
                    or proposal.get("selection_metric") != "public_val_bpb"):
                raise QueuePlanError(f"queue proposal contract changed: {row['proposal_id']}")
            change = row.get("constant_change")
            if expected_change is None:
                if change is not None or row.get("role") != "control" or row.get("box") != "trn22":
                    raise QueuePlanError("control entry changed")
                if proposal.get("kind") != "control" or proposal.get("change_type") != "control":
                    raise QueuePlanError("fixed control is not represented through the control-only schema")
            else:
                name, before, after = expected_change
                if not isinstance(change, dict) or (change.get("name"), change.get("from"), change.get("to")) != (name, before, after):
                    raise QueuePlanError(f"queue entry {position} constant change changed")
                if row.get("role") != "candidate":
                    raise QueuePlanError(f"queue entry {position} is not a candidate")
                if (proposal.get("change_type") != "code" or proposal.get("flags") != []
                        or proposal.get("files_allowed") != ["train.py"]
                        or len(str(proposal.get("implementation_plan", ""))) < 30):
                    raise QueuePlanError(f"fixed candidate proposal is not a one-file code proposal: {row['proposal_id']}")
            if position == 1 and row.get("box") != "trn21":
                raise QueuePlanError("first candidate must run on trn21")
            if position > 1 and row.get("box") not in C.PINNED_HOSTS:
                raise QueuePlanError("later fixed entries require explicit box affinity")
            entries.append(QueueEntry(
                position, row["proposal_id"], row["role"], row.get("box"),
                row["proposal_path"], row["verification_path"], row["commit"],
                row["parent_commit"], row["train_sha256"], change,
            ))
        batch = rows[5]["constant_change"]
        if batch.get("legal_multiple") != 131072 or batch["to"] % batch["legal_multiple"]:
            raise QueuePlanError("batch arm is not the nearest authorized legal changed batch")
        return cls(path, hashlib.sha256(payload).hexdigest(), raw, tuple(entries))

    def verification(self, entry: QueueEntry, root: Path = C.ROOT) -> dict | None:
        path = root / entry.verification_path
        if not path.exists():
            return None
        path = _regular_under_root(root, entry.verification_path)
        data = json.loads(path.read_text())
        expected_proposal_sha = hashlib.sha256((root / entry.proposal_path).read_bytes()).hexdigest()
        expected = {
            "verdict": "PASS", "proposal_id": entry.proposal_id, "commit": entry.commit,
            "train_sha256": entry.train_sha256, "proposal_sha256": expected_proposal_sha,
            "verified_by_role": "verifier",
        }
        if any(data.get(k) != v for k, v in expected.items()):
            raise QueuePlanError(f"invalid independent verification receipt for {entry.proposal_id}")
        return data

    def all_verified(self, entries: tuple[QueueEntry, ...], root: Path = C.ROOT) -> bool:
        return all(self.verification(entry, root) is not None for entry in entries)
