"""The unattended loop. Deterministic; models only propose and review.

One ``cycle()`` performs at most one hardware action (stage+launch, or collect) per box and at
most ``caps.max_llm_calls_per_cycle`` model calls. ``daemon()`` repeats cycles with sleeps. Every
transition is persisted before the next remote call so a restart resumes instead of repeating.

Ladder per box (all on the authoritative commit, seed 42, full-chip shape):

    control smoke -> control full -> candidate smoke -> [KEEP] candidate full -> verdict

Circuit breakers (open => no new launches; running training is observed to completion):
STOP file, competition cutoff, config invalid, caps reached, archive failure, unknown lock owner,
two consecutive infrastructure failures on a box, missing artifacts, state schema mismatch.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config as C
from .analyst import Verdict, compare, verdict_markdown
from .archive import ArchiveError, GitArchiver
from .implementer import ImplementationError, implement_code_proposal
from .ledger import Ledger
from .manifest import ManifestError, Proposal, RunManifest, build_manifest, known_train_flags
from .models import ModelClient, ModelError, ModelReply, parse_json_block
from .remote import LocalGit, RemoteError, RemoteOperator, StagedRun, LaunchReceipt
from .queue_plan import FixedQueuePlan, QueueEntry, QueuePlanError
from .state import Store, utc_now
from .verifier import closed_direction_terms, load_context, run_verifier

RESEARCHER_SYSTEM = """You are the Researcher in an unattended Trainium Phase 1 research loop.
Goal: lower official val_bpb after a 30-minute full-chip Trn2 run. First compare 4 to 6 distinct
candidate directions, then select exactly ONE evidence-backed, falsifiable causal change for the
controller to execute. Include the rejected shortlist in the output; it is planning evidence, not
authorization to bundle or run those alternatives. Source-level bug/wiring audits are the first
idea stream, followed by measured prior outcomes and primary-source techniques. It may be an
existing train.py flag experiment or a small code change
confined to train.py (architecture, optimizer, schedule, numerical format, or inline NKI kernel).
Prefer a bounded code proposal when the remaining flag ideas are already heavily explored; use a
flag only when the supplied evidence establishes that it is a real, isolated, non-duplicate gap.
Seed 42, full-chip shape, data, evaluator, and budget are fixed by the controller.
Search the provided outcome memory, ledger summary and decision log; never repeat a closed or already-run flag set
at this commit. A flag proposal must set one causal option only; do not bundle a second precision,
kernel, optimizer, or architecture switch. If two switches are truly inseparable, propose a code
change that implements one explicit boundary instead. The supplied authoritative train.py evidence
index contains measured/closed source comments: do not call a flag untested or infer its scope/cost
from its name when that index contradicts the claim. Absence from the bounded index means unknown,
not untested. Cite repository evidence. Return ONLY a JSON object:
{"proposal_id": "YYYYMMDD-short-slug", "hypothesis": "...", "single_causal_change": "...",
 "flags": ["--flag", "value", ...], "evidence": ["path or label ..."],
 "duplicate_search_terms": ["..."], "expected_effect": "predeclared BPB direction and size",
 "mechanism_class":"throughput"|"per_step_quality"|"schedule"|"capacity"|"data_regime",
 "predicted_bpb_delta":-0.002, "predicted_bpb_interval":[-0.004,0.001],
 "candidate_shortlist":[{"idea":"...","mechanism_class":"schedule",
   "predicted_bpb_delta":-0.002,"why_ranked":"evidence, transfer likelihood, and cost"}, ...],
 "selection_rationale":"why the selected idea has the best evidence-adjusted value per host-hour",
 "abort_conditions": ["..."], "change_type":"flags"|"code",
 "smoke_gate":{"min_candidate_steps":123,"min_step_rate_ratio":1.0,"max_bpb_delta":0.0},
 "implementation_plan":"exact bounded edit plan, empty for flags",
 "files_allowed":[]|["train.py"]}
BPB deltas are always candidate minus matched control, so negative is better. Predictions are
priors, not measurements: give an honest interval that includes regression when uncertainty
warrants it. The shortlist must contain 4 to 6 candidates and no mechanism class may occupy more
than half the list. Rank transfer likelihood, information value, and host-hours explicitly.
smoke_gate must contain only thresholds justified by the proposal; use {} when the controller's
global smoke gate is sufficient. Proposal thresholds can only tighten the global gate. For flags,
files_allowed must be [] and implementation_plan empty. For code, flags must be [],
files_allowed must be ["train.py"], and the plan must be precise enough for another isolated agent
to implement without inventing a second change. The pre-run implementation pipeline can edit only
train.py and perform deterministic syntax/diff/static review. Do not require named pre-edit
artifacts, CPU/unit execution, compilation output, dtype logs, smoke metrics, or hardware results
before the implementation gate can pass; express runtime checks as smoke abort conditions instead.
Do not propose evaluator/tokenizer/data changes,
external data, pretrained assets, or private validation access. If no non-duplicate hypothesis exists, return
{"proposal_id": "none", "reason": "..."}."""

ANALYST_SYSTEM = """You are the Analyst. You receive a deterministic verdict plus the candidate and
control ledger rows. Write at most 12 lines of plain prose explaining the comparison, naming any
confounder. You may not change the decision. Return plain text."""


def _clip_text(value, max_chars: int) -> str:
    """Bound model context fields without changing the durable source artifact."""
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _clip_findings(value, *, max_items: int = 3, max_chars: int = 350) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_clip_text(item, max_chars) for item in value[:max_items]]


@dataclass
class CycleReport:
    started_at: str
    actions: list[str] = field(default_factory=list)
    stopped_reason: str | None = None
    llm_calls: int = 0

    def add(self, msg: str) -> None:
        self.actions.append(msg)

    def to_dict(self) -> dict:
        return self.__dict__


class Controller:
    def __init__(
        self,
        *,
        cfg: C.LoopConfig,
        store: Store,
        operator: RemoteOperator,
        model: ModelClient | None,
        archiver: GitArchiver,
        local_git: LocalGit,
        stop_file: Path = C.STOP_FILE,
        now=None,
        train_py_text: str | None = None,
    ):
        self.cfg = cfg
        self.store = store
        self.op = operator
        self.model = model
        self.archiver = archiver
        self.git = local_git
        self.stop_file = stop_file
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._train_py_text = train_py_text
        self._cycle_llm_calls = 0
        self.fixed_plan = (
            FixedQueuePlan.load(C.QUEUE_PLAN_FILE, C.ROOT)
            if cfg.plan_mode == "fixed_full_only" else None
        )

    # ---- helpers ----------------------------------------------------------------------------

    def now_iso(self) -> str:
        return self._now().strftime("%Y-%m-%dT%H:%M:%SZ")

    def known_flags(self) -> set[str]:
        if self._train_py_text is None:
            self._train_py_text = self.git.file_at_commit(C.AUTHORITATIVE_COMMIT, "train.py").decode(errors="replace")
        return known_train_flags(self._train_py_text)

    def _record_usage(self, role: str, reply: ModelReply) -> None:
        self._cycle_llm_calls += 1
        self.store.record_llm(role, reply.input_tokens, reply.output_tokens, self.cfg.caps)

    def _llm_ok(self) -> tuple[bool, str]:
        if self.model is None:
            return False, "no model client"
        return self.store.llm_calls_allowed(self.cfg.caps, self._cycle_llm_calls)

    def _llm_calls_available(self, needed: int) -> tuple[bool, str]:
        """Reserve an atomic multi-agent stage before its first model call.

        Code implementation needs both the Implementer and an independent implementation
        Verifier.  Starting when only one call remains strands a valid proposal halfway through
        the stage and used to record a quota condition as a technical rejection.
        """
        if self.model is None:
            return False, "no model client"
        return self.store.llm_calls_allowed(self.cfg.caps, self._cycle_llm_calls, needed=needed)

    def _box_has_ready_work(self, box: str) -> bool:
        """Return whether *box* can advance durable work without asking for a new proposal.

        The configured box order is intentionally stable, but it must not let an idle first box
        start a potentially long Researcher call while a later box already owns a verified smoke
        or full candidate.  Controls and deferred code proposals use the same ready stages.
        """
        return any(
            rec.get("stage") in {"proposal_verified", "verified"}
            and rec.get("box") in (None, box)
            for rec in self.store.proposals().values()
        )

    def breakers(self) -> str | None:
        if self.stop_file.exists():
            return "STOP file present"
        b = self.store.breaker()
        if b:
            return f"breaker open: {b['reason']}"
        if self._now() >= C.COMPETITION_CUTOFF:
            return "competition cutoff reached"
        errs = self.cfg.validate()
        if errs:
            return "config invalid: " + "; ".join(errs)
        return None

    def all_ledgers(self) -> list[Ledger]:
        ledgers = [Ledger.load(C.LOCAL_LEDGER)]
        if self.store.experiments_dir.is_dir():
            for p in self.store.experiments_dir.glob("*/artifacts/*/experiment-log.jsonl"):
                ledgers.append(Ledger.load(p))
        return ledgers

    def all_labels(self) -> set[str]:
        labels = self.store.used_labels()
        for l in self.all_ledgers():
            labels |= l.labels()
        return labels

    # ---- cycle ------------------------------------------------------------------------------

    def cycle(self) -> CycleReport:
        report = CycleReport(started_at=self.now_iso())
        self._cycle_llm_calls = 0
        reason = self.breakers()
        if reason:
            if self.fixed_plan is not None and any(
                    rec.get("stage", "").startswith("launched:")
                    for rec in self.store.proposals().values()):
                report.add("fixed queue: breaker open; observing already-launched work only")
                try:
                    self._reconcile(report)
                except Exception as exc:
                    report.add(f"fixed observation failed: {type(exc).__name__}: {exc}")
            report.stopped_reason = reason
            return report
        self.store.state["cycles"] += 1
        self.store.state["last_cycle"] = {"started_at": report.started_at}
        self.store.save()

        try:
            if self.fixed_plan is not None:
                self._cycle_fixed(report)
                self.store.state["last_cycle"]["finished_at"] = self.now_iso()
                self.store.state["last_cycle"]["actions"] = report.actions
                self.store.save()
                return report
            busy_boxes = self._reconcile(report)
            if self.breakers():
                report.stopped_reason = self.breakers()
                return report
            # Advance already-verified work before generating a fresh hypothesis on an earlier
            # idle box.  ``sorted`` is stable, so the configured order is preserved among boxes
            # with equal priority.
            boxes = sorted(self.cfg.boxes, key=lambda box: not self._box_has_ready_work(box))
            for box in boxes:
                if box in busy_boxes:
                    continue
                if self._advance_box(box, report):
                    break  # one launch per cycle
        except (RemoteError, ManifestError, ArchiveError, QueuePlanError, RuntimeError) as exc:
            self.store.open_breaker(type(exc).__name__, detail=str(exc)[:1000])
            report.stopped_reason = f"{type(exc).__name__}: {exc}"
        self.store.state["last_cycle"]["finished_at"] = self.now_iso()
        self.store.state["last_cycle"]["actions"] = report.actions
        self.store.save()
        report.llm_calls = self._cycle_llm_calls
        return report

    # ---- human-authorized fixed full-only queue ---------------------------------------------

    def _validate_fixed_entry(self, entry: QueueEntry) -> None:
        if not self.git.commit_exists(entry.commit):
            raise QueuePlanError(f"fixed commit does not exist: {entry.commit}")
        candidate = self.git.file_at_commit(entry.commit, "train.py")
        if hashlib.sha256(candidate).hexdigest() != entry.train_sha256:
            raise QueuePlanError(f"fixed train hash mismatch: {entry.proposal_id}")
        if self.git.parent_of(entry.commit) != entry.parent_commit:
            raise QueuePlanError(f"fixed commit parent mismatch: {entry.proposal_id}")
        expected_paths = ["research/run_experiment.py"] if entry.position == 0 else ["train.py"]
        if self.git.changed_paths(entry.parent_commit, entry.commit) != expected_paths:
            raise QueuePlanError(f"fixed commit changed unauthorized paths: {entry.proposal_id}")
        protected = {
            "research/run_experiment.py": C.AUTHORITATIVE_RUNNER_SHA256,
            "prepare.py": C.AUTHORITATIVE_PREPARE_SHA256,
        }
        for path, expected_sha in protected.items():
            actual_sha = hashlib.sha256(self.git.file_at_commit(entry.commit, path)).hexdigest()
            if actual_sha != expected_sha:
                raise QueuePlanError(f"fixed protected source mismatch: {entry.proposal_id}:{path}")
        control = self.git.file_at_commit(C.AUTHORITATIVE_COMMIT, "train.py")
        if entry.position == 0:
            if candidate != control:
                raise QueuePlanError("authoritative control train bytes changed")
            return
        change = entry.constant_change or {}
        before = f"{change['name']} = {change['from']}".encode()
        after = f"{change['name']} = {change['to']}".encode()
        if control.count(before) != 1 or candidate != control.replace(before, after, 1):
            raise QueuePlanError(f"{entry.proposal_id} is not the declared one-constant diff")

    def _cycle_fixed(self, report: CycleReport) -> None:
        assert self.fixed_plan is not None
        parked = self.fixed_plan.raw.get("parked_proposals", [])
        self.store.activate_fixed_plan(
            plan_id=self.fixed_plan.raw["plan_id"], plan_sha256=self.fixed_plan.sha256,
            parked_proposals=parked,
        )
        clearance = self.store.state.get("fixed_breaker_clearance")
        expected_clearance = {
            "receipt_commit": C.BREAKER_CLEANUP_RECEIPT_COMMIT,
            "receipt_path": C.BREAKER_CLEANUP_RECEIPT_PATH,
            "receipt_sha256": C.BREAKER_CLEANUP_RECEIPT_SHA256,
            "cleared_by_role": "operator",
        }
        if not isinstance(clearance, dict) or any(clearance.get(k) != v for k, v in expected_clearance.items()):
            report.add("fixed queue: waiting for explicit Operator breaker-clearance receipt")
            return
        receipt_bytes = self.git.file_at_commit(
            C.BREAKER_CLEANUP_RECEIPT_COMMIT, C.BREAKER_CLEANUP_RECEIPT_PATH
        )
        if hashlib.sha256(receipt_bytes).hexdigest() != C.BREAKER_CLEANUP_RECEIPT_SHA256:
            raise QueuePlanError("pushed trn21 cleanup receipt bytes differ from pinned evidence")
        for entry in self.fixed_plan.entries:
            self._validate_fixed_entry(entry)
        busy = self._reconcile(report)
        if self.store.breaker():
            report.add("fixed queue: breaker opened during reconciliation; no launch eligible")
            return
        launchable = set(self.fixed_plan.entries[i].proposal_id for i in range(len(self.fixed_plan.entries)))
        for pid, rec in self.store.proposals().items():
            if pid not in launchable and rec.get("stage") in {
                    "proposed", "proposal_verified", "verified", "implemented"}:
                raise QueuePlanError(f"non-queue proposal remains launchable: {pid}")

        first = self.fixed_plan.entries[:2]
        first_records = [self.store.proposals().get(entry.proposal_id) for entry in first]
        if not all(first_records):
            if not self.fixed_plan.all_verified(first, C.ROOT):
                report.add("fixed queue: waiting for both independent first-cohort verifications")
                return
            for entry in first:
                self.store.track_existing_proposal(
                    entry.proposal_id, box=entry.box or "", candidate_commit=entry.commit,
                    comparison_role=entry.role,
                )
        if (all(self.store.proposals()[entry.proposal_id]["stage"] == "verified" for entry in first)
                and not self.store.state.get("fixed_first_cohort_preflight")):
            if not self.fixed_plan.all_verified(first, C.ROOT):
                raise QueuePlanError("first-cohort verification receipts changed before preflight")
            inventories = {entry.box: self.op.inventory(entry.box) for entry in first}
            if not all(inv.clean_and_idle for inv in inventories.values()):
                report.add("fixed queue: both first-cohort preflights must be clean in the same cycle")
                return
            self.store.state["fixed_first_cohort_preflight"] = {
                "at": self.now_iso(), "inventories": {box: inv.to_dict() for box, inv in inventories.items()}
            }
            self.store.save()
            for entry in first:
                launched = self._stage_and_launch(
                    entry.proposal_id, "full", entry.box or "", [], report,
                    execution_mode="exact-dense1024-cold-eval",
                    comparison_role=entry.role, expected_train_sha256=entry.train_sha256,
                    control_label_override=(None if entry.role == "control" else
                                            f"al-{first[0].proposal_id}-full-r1"),
                )
                if not launched:
                    raise QueuePlanError("first-cohort launch failed after paired preflight")
            return
        # If a crash occurred between first-cohort launches, the durable paired preflight permits
        # only the missing member to be resumed; no unrelated work can enter the gap.
        if self.store.state.get("fixed_first_cohort_preflight"):
            for entry in first:
                rec = self.store.proposals()[entry.proposal_id]
                if rec["stage"] == "verified" and entry.box not in busy:
                    self._stage_and_launch(
                        entry.proposal_id, "full", entry.box or "", [], report,
                        execution_mode="exact-dense1024-cold-eval",
                        comparison_role=entry.role, expected_train_sha256=entry.train_sha256,
                        control_label_override=(None if entry.role == "control" else
                                                f"al-{first[0].proposal_id}-full-r1"),
                    )
                    return
        if any(self.store.proposals()[entry.proposal_id]["stage"] != "closed" for entry in first):
            report.add("fixed queue: first cohort still active")
            return
        # Later arms are strictly serial and require their own independent receipt.
        for entry in self.fixed_plan.entries[2:]:
            rec = self.store.proposals().get(entry.proposal_id)
            if rec is not None:
                if rec["stage"] != "closed":
                    report.add(f"fixed queue: {entry.proposal_id} still active")
                    return
                continue
            if self.fixed_plan.verification(entry, C.ROOT) is None:
                report.add(f"fixed queue: waiting for independent verification of {entry.proposal_id}")
                return
            self.store.track_existing_proposal(
                entry.proposal_id, box=entry.box or "", candidate_commit=entry.commit,
                comparison_role=entry.role,
            )
            self._stage_and_launch(
                entry.proposal_id, "full", entry.box or "", [], report,
                execution_mode="exact-dense1024-cold-eval", comparison_role=entry.role,
                expected_train_sha256=entry.train_sha256,
                control_label_override=f"al-{first[0].proposal_id}-full-r1",
            )
            return
        report.add("fixed queue: complete")

    # ---- reconcile running/collected work -------------------------------------------------

    def _reconcile(self, report: CycleReport) -> set[str]:
        busy: set[str] = set()
        fixed_ids = ({entry.proposal_id for entry in self.fixed_plan.entries}
                     if self.fixed_plan is not None else None)
        for pid, rec in list(self.store.proposals().items()):
            stage = rec["stage"]
            if fixed_ids is not None and pid not in fixed_ids:
                if stage not in {"parked", "closed", "rejected"}:
                    raise QueuePlanError(f"non-queue runtime work is not inert: {pid} ({stage})")
                continue
            if self.store.breaker() and stage.startswith("staged:"):
                report.add(f"{pid}: breaker open; persisted stage not launched")
                continue
            if stage.startswith("launched:"):
                tier = stage.split(":", 1)[1]
                if self._poll_and_collect(pid, tier, report):
                    stage = self.store.proposals()[pid]["stage"]
                else:
                    busy.add(rec["box"])
                    continue
            if stage.startswith("failure_collected:"):
                tier = stage.split(":", 1)[1]
                try:
                    self._archive_fixed_failure(pid, tier, report)
                except Exception as exc:
                    self._persist_fixed_lifecycle_failure(pid, tier, "failure_archive", exc)
                continue
            if stage.startswith("collected:"):
                tier = stage.split(":", 1)[1]
                self._analyze(pid, tier, report)
                stage = self.store.proposals()[pid]["stage"]
            if stage.startswith("analyzed:"):
                tier = stage.split(":", 1)[1]
                try:
                    self._archive_and_promote(pid, tier, report)
                except Exception as exc:
                    if self.store.proposals()[pid].get("fixed_queue"):
                        self._persist_fixed_lifecycle_failure(pid, tier, "success_archive", exc)
                        continue
                    raise
            if stage.startswith("staged:"):
                # The immutable checkout, manifest, and staged hashes were persisted before the
                # launch attempt.  Reuse that exact label after a restart: op.launch re-checks the
                # controller lock, idle/device/disk inventory, HEAD, porcelain, and train.py hash.
                # Its unique launcher directory still fails closed if the first launch became
                # ambiguous before the receipt was persisted.
                tier = stage.split(":", 1)[1]
                self._launch_persisted_stage(pid, tier, report, resumed=True)
                busy.add(rec["box"])
        return busy

    def _launch_persisted_stage(self, pid: str, tier: str, report: CycleReport, *, resumed: bool = False) -> bool:
        manifest = RunManifest.from_dict(self.store.read_artifact(pid, f"{tier}-manifest.json"))
        staged_data = self.store.read_artifact(pid, f"{tier}-staged.json")
        if not staged_data:
            raise RemoteError(f"{pid}/{tier}: staged state has no durable staged receipt")
        staged = StagedRun(**staged_data)
        receipt = self.op.launch(manifest, staged)
        self.store.write_artifact(pid, f"{tier}-launch.json", receipt.to_dict())
        self.store.tier(pid, tier).update({"pid": receipt.pid, "launched_at": receipt.launched_at})
        self.store.set_stage(pid, f"launched:{tier}", box=manifest.box)
        self.store.record_run()
        prefix = "resumed staged launch" if resumed else "launched"
        report.add(f"{manifest.box}: {prefix} {manifest.label} pid={receipt.pid} dir={manifest.remote_run_dir}")
        return True

    def _poll_and_collect(self, pid: str, tier: str, report: CycleReport) -> bool:
        trec = self.store.tier(pid, tier)
        manifest = RunManifest.from_dict(self.store.read_artifact(pid, f"{tier}-manifest.json"))
        receipt = LaunchReceipt(**self.store.read_artifact(pid, f"{tier}-launch.json"))
        status = self.op.poll(manifest, receipt)
        trec["last_poll"] = status.to_dict()
        self.store.save()
        launched = datetime.strptime(receipt.launched_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        elapsed = (self._now() - launched).total_seconds()
        fixed_timeout = (
            self.store.proposals()[pid].get("fixed_queue")
            and (elapsed > self.cfg.caps.run_timeout_seconds or trec.get("timeout_observed_at"))
        )
        if fixed_timeout:
            if status.pid_alive:
                observation = {
                    "schema_version": 1, "final": False,
                    "reason": "run exceeded controller timeout; awaiting process-group quiescence",
                    "manifest_id": manifest.manifest_id, "label": manifest.label,
                    "status": status.to_dict(), "observed_at": self.now_iso(),
                }
                self.store.write_artifact(pid, f"{tier}-timeout-observation.json", observation)
                trec["timeout_observed_at"] = trec.get("timeout_observed_at") or self.now_iso()
                self.store.save()
                report.add(
                    f"{pid}/{tier}: past timeout ({elapsed:.0f}s); process group still alive, "
                    "host remains busy"
                )
                return False
            self._record_fixed_failure(
                pid, tier, manifest, receipt, status,
                "run exceeded controller timeout; process group is now quiescent", report,
            )
            return True
        if not status.done:
            if status.pid_alive:
                if elapsed > self.cfg.caps.run_timeout_seconds:
                    report.add(f"{pid}/{tier}: still running past timeout ({elapsed:.0f}s)")
                    self._fail_tier(pid, tier, manifest, "run exceeded controller timeout", infra=True)
                    return True
                report.add(f"{pid}/{tier}: running ({elapsed:.0f}s)")
                return False
            if elapsed < self.cfg.caps.launch_grace_seconds:
                report.add(f"{pid}/{tier}: pid not visible yet, within grace")
                return False
            if self.store.proposals()[pid].get("fixed_queue"):
                self._record_fixed_failure(
                    pid, tier, manifest, receipt, status,
                    "runner and descendants exited without a done marker", report,
                )
                return True
            self._fail_tier(pid, tier, manifest, "runner died without done marker", infra=True)
            return True
        if status.pid_alive:
            report.add(f"{pid}/{tier}: done marker present; awaiting process-group quiescence")
            return False
        # done: collect
        local_dir = self.store.proposal_dir(pid) / "artifacts" / tier
        try:
            collected = self.op.collect(manifest, receipt, local_dir)
        except Exception as exc:
            if not self.store.proposals()[pid].get("fixed_queue"):
                raise
            self._record_fixed_failure(
                pid, tier, manifest, receipt, status,
                f"completed run evidence collection failed: {exc}", report,
            )
            return True
        row = collected.ledger_row
        if row is not None:
            row = dict(row)
            row["_dirt_explained"] = collected.post_run_dirt_explained
        result = {
            "manifest_id": manifest.manifest_id,
            "label": manifest.label,
            "returncode": status.returncode,
            "collected": collected.to_dict(),
            "ledger_row": row,
            "hostname": receipt.pre_launch_inventory.get("hostname"),
        }
        self.store.write_artifact(pid, f"{tier}-result.json", result)
        self.store.set_stage(pid, f"collected:{tier}")
        try:
            self.op.release_lock(manifest.box)
        except RemoteError as exc:
            report.add(f"lock release warning: {exc}")
        report.add(f"{pid}/{tier}: collected rc={status.returncode} label={manifest.label}")
        return True

    def _record_fixed_failure(
        self,
        pid: str,
        tier: str,
        manifest: RunManifest,
        receipt: LaunchReceipt,
        status,
        reason: str,
        report: CycleReport,
    ) -> None:
        if tier != "full" or status.pid_alive or getattr(status, "group_members", []):
            raise QueuePlanError("fixed failure can be finalized only for a quiescent full run")
        local_dir = self.store.proposal_dir(pid) / "artifacts" / "full-failure"
        try:
            evidence = self.op.collect_failure_evidence(
                manifest, receipt, status, local_dir, reason
            )
        except Exception as exc:
            evidence = {
                "schema_version": 1, "outcome": "failed", "final": True,
                "reason": reason, "manifest_id": manifest.manifest_id,
                "label": manifest.label, "box": manifest.box,
                "manifest": manifest.to_dict(), "launch_receipt": receipt.to_dict(),
                "status": status.to_dict(), "preflight": receipt.pre_launch_inventory,
                "postflight": {"available": False, "error": str(exc)[:500]},
                "process": {
                    "pid": receipt.pid, "process_group": receipt.process_group or receipt.pid,
                    "quiescent": True, "remaining_members": [],
                },
                "files": {}, "available_result": None,
                "errors": [f"failure evidence collection: {exc}"[:1000]],
                "collected_at": self.now_iso(),
            }
        if (evidence.get("final") is not True
                or (evidence.get("process") or {}).get("quiescent") is not True):
            raise QueuePlanError("failure evidence did not prove process-group quiescence")
        self.store.write_artifact(pid, f"{tier}-failure.json", evidence)
        self.store.set_stage(
            pid, f"failure_collected:{tier}", verdict="FAILED", failure=reason,
            host_busy=False,
        )
        try:
            self.op.release_lock(manifest.box)
        except Exception as exc:
            evidence["errors"].append(f"lock release: {exc}"[:1000])
            self.store.write_artifact(pid, f"{tier}-failure.json", evidence)
        report.add(f"{pid}/{tier}: final failure evidence collected after quiescence")

    def _persist_fixed_lifecycle_failure(
        self, pid: str, tier: str, phase: str, exc: Exception
    ) -> None:
        rec = self.store.proposals()[pid]
        artifact = {
            "schema_version": 1, "outcome": "lifecycle_failure", "final": True,
            "proposal_id": pid, "tier": tier, "phase": phase,
            "stage_before": rec.get("stage"), "error_type": type(exc).__name__,
            "error": str(exc)[:2000], "recorded_at": self.now_iso(),
        }
        self.store.write_artifact(pid, f"{tier}-lifecycle-failure.json", artifact)
        self.store.set_stage(pid, rec.get("stage", f"analyzed:{tier}"), lifecycle_failure=artifact)
        self.store.open_breaker(
            "fixed lifecycle failure", proposal_id=pid, tier=tier,
            phase=phase, error=str(exc)[:1000],
        )

    def _fail_tier(self, pid: str, tier: str, manifest: RunManifest, reason: str, infra: bool) -> None:
        verdict = Verdict("FAILED", tier, [reason])
        self.store.write_artifact(pid, f"{tier}-verdict.json", verdict.to_dict())
        self.store.set_stage(pid, f"analyzed:{tier}", verdict="FAILED", failure=reason)
        if infra:
            n = self.store.box_failure(manifest.box)
            if n >= self.cfg.caps.max_consecutive_infra_failures_per_box:
                self.store.open_breaker("consecutive infrastructure failures", box=manifest.box, count=n)

    # ---- analysis ---------------------------------------------------------------------------

    def _control_id(self, box: str) -> str:
        return f"control-{box}-lnc{self.cfg.lnc}x{self.cfg.nproc_per_node}-{C.AUTHORITATIVE_COMMIT[:8]}"

    def _control_result(self, box: str, tier: str) -> dict | None:
        cid = self._control_id(box)
        rec = self.store.proposals().get(cid)
        if not rec:
            return None
        res = self.store.read_artifact(cid, f"{tier}-result.json")
        ver = self.store.read_artifact(cid, f"{tier}-verdict.json")
        if res and ver and ver.get("decision") == "KEEP":
            return res
        return None

    def _analyze(self, pid: str, tier: str, report: CycleReport) -> None:
        rec = self.store.proposals()[pid]
        result = self.store.read_artifact(pid, f"{tier}-result.json")
        manifest = self.store.read_artifact(pid, f"{tier}-manifest.json")
        proposal = self.store.read_artifact(pid, "proposal.json")
        if rec.get("fixed_queue"):
            self._analyze_fixed(pid, tier, result, manifest, proposal, report)
            return
        is_control = proposal.get("kind") == "control"
        if is_control:
            from .analyst import correctness_failures

            fails = correctness_failures(result["ledger_row"], result["collected"]["checkpoint_present"], result["returncode"])
            verdict = Verdict("KEEP" if not fails else "FAILED", tier, fails or ["control run correct; usable as matched control"])
            if result["ledger_row"]:
                verdict.metrics = {"control_bpb": result["ledger_row"].get("public_val_bpb"), "control_steps": result["ledger_row"].get("num_steps")}
        else:
            ctrl = self._control_result(rec["box"], tier)
            verdict = compare(
                tier=tier,
                candidate=result["ledger_row"],
                control=ctrl["ledger_row"] if ctrl else None,
                candidate_ckpt=result["collected"]["checkpoint_present"],
                control_ckpt=bool(ctrl and ctrl["collected"]["checkpoint_present"]),
                candidate_rc=result["returncode"],
                caps=self.cfg.caps,
                smoke_gate=proposal.get("smoke_gate", {}),
            )
        if verdict.decision == "FAILED" and any("runner exit code" in r or "final.pt missing" in r for r in verdict.reasons):
            self.store.box_failure(rec["box"])
        elif verdict.decision != "FAILED":
            self.store.box_success(rec["box"])
        prose = ""
        ok, _ = self._llm_ok()
        if ok and self.cfg.analyst_model_id and not is_control:
            try:
                reply = self.model.complete(
                    role="analyst", model_id=self.cfg.analyst_model_id, system=ANALYST_SYSTEM,
                    user=json.dumps({"verdict": verdict.to_dict(), "candidate": result["ledger_row"], "control": (self._control_result(rec["box"], tier) or {}).get("ledger_row")}, indent=2),
                    max_tokens=800,
                )
                self._record_usage("analyst", reply)
                prose = reply.text
            except ModelError as exc:
                prose = f"(analyst unavailable: {exc})"
        self.store.write_artifact(pid, f"{tier}-verdict.json", {**verdict.to_dict(), "analyst_prose": prose})
        md = verdict_markdown(proposal, manifest, verdict, manifest.get("control_label"), result.get("hostname") or "")
        (self.store.proposal_dir(pid) / f"{tier}-verdict.md").write_text(md + ("\n## Analyst\n\n" + prose + "\n" if prose else ""))
        self.store.set_stage(pid, f"analyzed:{tier}", verdict=verdict.decision)
        report.add(f"{pid}/{tier}: {verdict.decision} — {verdict.reasons[0]}")

    def _analyze_fixed(self, pid: str, tier: str, result: dict, manifest: dict,
                       proposal: dict, report: CycleReport) -> None:
        if tier != "full":
            raise QueuePlanError("fixed queue attempted a non-full analysis")
        rec = self.store.proposals()[pid]
        row = result.get("ledger_row") or {}
        collected = result.get("collected") or {}
        failures = self._fixed_correctness_failures(rec, result, manifest)
        bpb = row.get("public_val_bpb")
        role = rec.get("comparison_role")
        control_bpb = None
        if role == "candidate":
            control_id = self.fixed_plan.entries[0].proposal_id if self.fixed_plan else ""
            control = self.store.read_artifact(control_id, "full-result.json") or {}
            control_verdict = self.store.read_artifact(control_id, "full-verdict.json") or {}
            control_bpb = (control.get("ledger_row") or {}).get("public_val_bpb")
            control_state = self.store.proposals().get(control_id, {})
            if (control_state.get("stage") != "closed"
                    or not (control_state.get("archive_verified") or {}).get("local_verified")
                    or control_verdict.get("decision") != "KEEP"
                    or not isinstance(control_bpb, (int, float))):
                report.add(f"{pid}/full: waiting for archived usable fixed control")
                return
        if failures:
            verdict = Verdict("FAILED", tier, failures)
        elif role == "control":
            verdict = Verdict("KEEP", tier, ["fixed control passed exact public-BPB correctness gates"])
        elif float(bpb) < float(control_bpb):
            verdict = Verdict("KEEP", tier, ["public_val_bpb is lower than the fixed control; throughput was not compared"])
        else:
            verdict = Verdict("DISCARD", tier, ["public_val_bpb did not beat the fixed control; throughput was not compared"])
        verdict.metrics = {"candidate_bpb": bpb, "control_bpb": control_bpb,
                           "selection_metric": "public_val_bpb"}
        self.store.write_artifact(pid, f"{tier}-verdict.json", verdict.to_dict())
        md = verdict_markdown(proposal, manifest, verdict, manifest.get("control_label"), result.get("hostname") or "")
        (self.store.proposal_dir(pid) / f"{tier}-verdict.md").write_text(md + "\n")
        self.store.set_stage(pid, f"analyzed:{tier}", verdict=verdict.decision)
        report.add(f"{pid}/{tier}: {verdict.decision} — {verdict.reasons[0]}")

    @staticmethod
    def _fixed_correctness_failures(rec: dict, result: dict, manifest: dict) -> list[str]:
        row = result.get("ledger_row") or {}
        collected = result.get("collected") or {}
        failures: list[str] = []
        if result.get("returncode") != 0:
            failures.append("runner exit code is nonzero")
        if not collected.get("checkpoint_present") or not collected.get("checkpoint_sha256"):
            failures.append("final.pt missing or unhashed")
        if row.get("_dirt_explained") is not True:
            failures.append("post-run dirt is unexplained")
        if row.get("status") != "keep" or row.get("over_budget") is True:
            failures.append("runner status/budget gate failed")
        bpb = row.get("public_val_bpb")
        if (isinstance(bpb, bool) or not isinstance(bpb, (int, float))
                or not math.isfinite(float(bpb))):
            failures.append("public_val_bpb is missing/nonfinite")
        steps = row.get("num_steps")
        if (isinstance(steps, bool) or not isinstance(steps, (int, float))
                or not math.isfinite(float(steps)) or steps <= 0):
            failures.append("num_steps is missing/nonpositive/nonfinite")
        required = {
            "eval_context": C.EXACT_DENSE1024_EVAL_CONTEXT,
            "eval_tokens_requested": C.EXACT_DENSE1024_EVAL_TOKENS,
            "eval_num_tokens": C.EXACT_DENSE1024_EVAL_TOKENS,
            "eval_num_bytes": C.EXACT_DENSE1024_EVAL_BYTES,
            "eval_counts_match": True,
            "eval_over_timeout": False,
            "selection_metric": "public_val_bpb",
            "exact_train_argv": list(C.EXACT_DENSE1024_TRAIN_ARGV),
            "git_commit": rec["candidate_commit"],
            "source_dirty": False,
        }
        for key, expected in required.items():
            if row.get(key) != expected:
                failures.append(f"{key} differs from fixed contract")
        if row.get("source_sha256", {}).get("train.py") != manifest.get("expected_train_sha256"):
            failures.append("runner-observed train.py hash differs from manifest")
        external = row.get("external_eval") or {}
        if (external.get("fresh_process") is not True or external.get("returncode") != 0
                or external.get("seq_len") != C.EXACT_DENSE1024_EVAL_CONTEXT
                or external.get("eval_tokens") != C.EXACT_DENSE1024_EVAL_TOKENS):
            failures.append("fresh public eval did not complete under the exact population")
        return failures

    def _archive_and_promote(self, pid: str, tier: str, report: CycleReport) -> None:
        rec = self.store.proposals()[pid]
        verdict = self.store.read_artifact(pid, f"{tier}-verdict.json")
        decision = verdict["decision"]
        proposal = self.store.read_artifact(pid, "proposal.json")
        if rec.get("fixed_queue"):
            self._archive_fixed(pid, tier, proposal, verdict, report)
            return
        is_control = proposal.get("kind") == "control"
        paths = [self.store.proposal_dir(pid)]
        closing = False
        if is_control:
            if tier == "smoke" and decision == "KEEP":
                next_stage = "verified"  # control full still needed
            elif tier == "full" and decision == "KEEP":
                next_stage, closing = "closed", True
            else:
                next_stage, closing = "closed", True
        else:
            if tier == "smoke" and decision == "KEEP":
                next_stage = "verified"  # eligible for full
            else:
                next_stage, closing = "closed", True
        if closing:
            self._append_decision_log(pid, tier, proposal, verdict)
            paths.append(C.DECISION_LOG)
        sha = self.archiver.commit(paths, f"autoloop: {pid} {tier} {decision}")
        self.store.set_stage(pid, next_stage, last_archive_commit=sha, closed=closing)
        report.add(f"{pid}/{tier}: archived ({sha or 'no changes'}) -> {next_stage}")
        # Once the complete result/verdict is durably pushed, prune disposable checkpoints to
        # keep both fixed-size Trn2 roots usable. Preserve a candidate full KEEP for human review;
        # smoke checkpoints, controls, and non-winning full checkpoints are reproducible scratch.
        if tier == "smoke" or is_control or decision != "KEEP":
            result = self.store.read_artifact(pid, f"{tier}-result.json") or {}
            expected_sha = (result.get("collected") or {}).get("checkpoint_sha256")
            if expected_sha:
                manifest = RunManifest.from_dict(self.store.read_artifact(pid, f"{tier}-manifest.json"))
                pruned = self.op.prune_checkpoint(manifest, expected_sha)
                self.store.write_artifact(pid, f"{tier}-checkpoint-pruned.json", pruned)
                prune_sha = self.archiver.commit(
                    [self.store.proposal_dir(pid)],
                    f"autoloop: prune preserved checkpoint for {pid} {tier}",
                )
                self.store.set_stage(pid, next_stage, last_archive_commit=prune_sha or sha, closed=closing)
                report.add(f"{pid}/{tier}: checkpoint hash-verified and pruned ({prune_sha or 'record local'})")

    def _archive_fixed(self, pid: str, tier: str, proposal: dict, verdict: dict,
                       report: CycleReport) -> None:
        if tier != "full":
            raise QueuePlanError("fixed queue attempted a non-full archive")
        result = self.store.read_artifact(pid, "full-result.json") or {}
        pdir = self.store.proposal_dir(pid)
        artifacts = self.store.proposal_dir(pid) / "artifacts" / "full"
        required = [
            pdir / "proposal.json",
            C.ROOT / next(e.verification_path for e in self.fixed_plan.entries if e.proposal_id == pid),
            pdir / "full-manifest.json", pdir / "full-staged.json",
            pdir / "full-launch.json", pdir / "full-result.json",
            pdir / "full-verdict.json", pdir / "full-verdict.md",
        ]
        expected_artifacts = {
            "experiment-log.jsonl", f"{result.get('label')}.log",
            f"{result.get('label')}.eval.log", f"{result.get('label')}.launch.json",
            "runner.log", "done.json", "launch.json",
        }
        if artifacts.is_symlink() or not artifacts.is_dir():
            raise ArchiveError("fixed run retrieved-artifact directory is missing/non-directory")
        observed_artifacts = {path.name for path in artifacts.iterdir()
                              if path.is_file() and not path.is_symlink()}
        if observed_artifacts != expected_artifacts:
            raise ArchiveError("fixed run archive lacks the exact complete retrieved evidence set")
        required.extend(artifacts / name for name in sorted(expected_artifacts))
        metrics = verdict.get("metrics") or {}
        decision_row = (
                f"| {self.now_iso()[:10]} | fixed queue {pid} | "
                f"{self.store.proposals()[pid]['candidate_commit'][:8]} | "
                f"{self.fixed_plan.entries[0].proposal_id} | public BPB "
                f"{metrics.get('candidate_bpb')} vs {metrics.get('control_bpb')}; "
                f"throughput excluded | {verdict['decision']} (full) | "
                f"`research/agentic/experiments/{pid}/` |\n"
        )
        if f"fixed queue {pid}" not in C.DECISION_LOG.read_text():
            with C.DECISION_LOG.open("a") as fh:
                fh.write(decision_row)
        journal_path = pdir / "full-lifecycle-journal.json"
        journal = self._fixed_journal(pid, "success", journal_path)
        if journal.get("phase") == "complete":
            completion_required = [journal_path, pdir / "full-cache-cleanup.json"]
            if expected_checkpoint := (result.get("collected") or {}).get("checkpoint_sha256"):
                completion_required.append(pdir / "full-checkpoint-pruned.json")
            completion_commit, completion_verification = self._commit_fixed_phase(
                [pdir], f"autoloop: complete fixed lifecycle for {pid} full",
                completion_required,
            )
            if not completion_verification.get("remote_verified"):
                raise ArchiveError("completed fixed lifecycle is not on configured remote")
            self.store.set_stage(
                pid, "closed", last_archive_commit=completion_commit, closed=True,
                archive_verified=journal.get("archive_verified", {}),
                archive_receipt_verified=journal.get("archive_receipt_verified", {}),
            )
            report.add(f"{pid}/full: reconciled already-complete lifecycle -> closed")
            return
        archive_commit = journal.get("archive_commit")
        if archive_commit:
            verification = self.archiver.verify_commit(archive_commit, required)
        else:
            journal["phase"] = "archive_pending"
            self.store.write_artifact(pid, "full-lifecycle-journal.json", journal)
            archive_commit, verification = self._commit_fixed_phase(
                [pdir, C.DECISION_LOG], f"autoloop: {pid} full {verdict['decision']}", required
            )
            journal["archive_commit"] = archive_commit
            journal["archive_verified"] = verification
            journal["phase"] = "archive_verified"
            self.store.write_artifact(pid, "full-lifecycle-journal.json", journal)
        if not verification.get("remote_verified"):
            raise ArchiveError("fixed evidence archive is not verified on the configured remote")
        archive_verification_path = self.store.write_artifact(
            pid, "full-archive-verification.json", verification
        )
        receipt_sha = journal.get("archive_receipt_commit")
        if receipt_sha:
            receipt_verification = self.archiver.verify_commit(
                receipt_sha, [archive_verification_path]
            )
        else:
            receipt_sha, receipt_verification = self._commit_fixed_phase(
                [pdir], f"autoloop: verify archive for {pid} full",
                [archive_verification_path],
            )
            journal["archive_receipt_commit"] = receipt_sha
            journal["archive_receipt_verified"] = receipt_verification
            journal["phase"] = "archive_receipt_verified"
            self.store.write_artifact(pid, "full-lifecycle-journal.json", journal)
        if not receipt_verification.get("remote_verified"):
            raise ArchiveError("fixed archive-verification receipt is not on configured remote")
        report.add(f"{pid}/full: archived and verified ({receipt_sha})")
        expected_sha = (result.get("collected") or {}).get("checkpoint_sha256")
        final_sha = receipt_sha
        manifest = RunManifest.from_dict(self.store.read_artifact(pid, "full-manifest.json"))
        if expected_sha:
            prune_path = pdir / "full-checkpoint-pruned.json"
            pruned = self.store.read_artifact(pid, "full-checkpoint-pruned.json")
            recorded_prune_commit = journal.get("checkpoint_receipt_commit")
            if recorded_prune_commit and pruned is not None:
                if pruned.get("sha256") != expected_sha:
                    raise ArchiveError("persisted checkpoint-prune receipt hash changed")
                self.op.confirm_checkpoint_pruned(manifest, pruned)
                prune_verification = self.archiver.verify_commit(
                    recorded_prune_commit, [prune_path]
                )
                if not prune_verification.get("remote_verified"):
                    raise ArchiveError("checkpoint prune receipt is not on configured remote")
                final_sha = recorded_prune_commit
            elif pruned is None:
                journal.update({
                    "phase": "checkpoint_delete_authorized",
                    "checkpoint_intent": {
                        "remote_checkpoint": manifest.remote_checkpoint,
                        "sha256": expected_sha,
                    },
                })
                self.store.write_artifact(pid, "full-lifecycle-journal.json", journal)
                intent_sha, intent_verification = self._commit_fixed_phase(
                    [pdir], f"autoloop: authorize checkpoint prune for {pid} full",
                    [journal_path, archive_verification_path],
                )
                if not intent_verification.get("remote_verified"):
                    raise ArchiveError("checkpoint deletion intent is not on configured remote")
                pruned = self.op.prune_checkpoint(
                    manifest, expected_sha, authorization_commit=intent_sha
                )
                if pruned.get("authorization_commit") != intent_sha:
                    raise ArchiveError("checkpoint prune receipt authorization commit mismatch")
                self.store.write_artifact(pid, "full-checkpoint-pruned.json", pruned)
            else:
                if pruned.get("sha256") != expected_sha:
                    raise ArchiveError("persisted checkpoint-prune receipt hash changed")
                self.op.confirm_checkpoint_pruned(manifest, pruned)
                intent_sha = pruned.get("authorization_commit")
                if not intent_sha:
                    raise ArchiveError("checkpoint receipt lacks its deletion-intent commit")
            if not recorded_prune_commit:
                journal.update({
                    "phase": "checkpoint_pruned", "checkpoint_intent_commit": intent_sha,
                    "checkpoint_receipt": {
                        "path": str(prune_path.relative_to(C.ROOT)), "sha256": expected_sha,
                        "reconciled_after_restart": bool(pruned.get("reconciled_after_restart")),
                    },
                })
                self.store.write_artifact(pid, "full-lifecycle-journal.json", journal)
                prune_sha, prune_verification = self._commit_fixed_phase(
                    [pdir], f"autoloop: prune archived checkpoint for {pid} full",
                    [prune_path, journal_path],
                )
                if not prune_verification["remote_verified"]:
                    raise ArchiveError("checkpoint prune receipt is not on configured remote")
                journal["checkpoint_receipt_commit"] = prune_sha
                final_sha = prune_sha
                report.add(
                    f"{pid}/full: checkpoint hash-verified, pruned, and "
                    f"receipt-verified ({prune_sha})"
                )
        cache_receipt_path = pdir / "full-cache-cleanup.json"
        cache_receipt = self.store.read_artifact(pid, "full-cache-cleanup.json")
        cache_root = f"{C.REMOTE_AUTOLOOP_DIR}/caches/{manifest.label}"
        cache_paths = [manifest.env[key] for key in C.CACHE_ENV_KEYS]
        cache_paths += [manifest.env[f"AUTOLOOP_EVAL_{key}"] for key in C.CACHE_ENV_KEYS]
        if cache_receipt is None:
            journal.update({
                "phase": "cache_delete_authorized",
                "cache_intent": {"cache_root": cache_root, "cache_paths": cache_paths},
            })
            self.store.write_artifact(pid, "full-lifecycle-journal.json", journal)
            cache_intent_sha, cache_intent_verification = self._commit_fixed_phase(
                [pdir], f"autoloop: authorize compile-cache cleanup for {pid} full",
                [journal_path, archive_verification_path],
            )
            if not cache_intent_verification.get("remote_verified"):
                raise ArchiveError("compile-cache deletion intent is not on configured remote")
            cache_receipt = self.op.cleanup_compile_caches(
                manifest, cache_receipt_path, authorization_commit=cache_intent_sha
            )
        else:
            cache_intent_sha = cache_receipt.get("authorization_commit")
            if not cache_intent_sha:
                intent = journal.get("cache_intent") or {}
                if (journal.get("phase") != "cache_delete_authorized"
                        or intent.get("cache_root") != cache_root
                        or intent.get("cache_paths") != cache_paths):
                    raise ArchiveError("cache-cleanup receipt lacks a reconcilable deletion intent")
                finder = getattr(self.archiver, "commit_for_path", None)
                cache_intent_sha = finder(journal_path) if finder else self.archiver.current_head()
                intent_verification = self.archiver.verify_commit(
                    cache_intent_sha, [journal_path, archive_verification_path]
                )
                if not intent_verification.get("remote_verified"):
                    raise ArchiveError("recovered cache deletion intent is not on configured remote")
                cache_receipt = self.op.cleanup_compile_caches(
                    manifest, cache_receipt_path, authorization_commit=cache_intent_sha
                )
            else:
                self.op.confirm_compile_caches_clean(manifest, cache_receipt)
        journal.update({
            "phase": "cache_cleaned", "cache_intent_commit": cache_intent_sha,
            "cache_receipt": {
                "path": str(cache_receipt_path.relative_to(C.ROOT)),
                "cache_root": cache_root,
                "reconciled_after_restart": bool(cache_receipt.get("reconciled_after_restart")),
            },
        })
        self.store.write_artifact(pid, "full-lifecycle-journal.json", journal)
        cache_sha, cache_verification = self._commit_fixed_phase(
            [pdir], f"autoloop: clean archived compile caches for {pid} full",
            [cache_receipt_path, journal_path],
        )
        if not cache_verification["remote_verified"]:
            raise ArchiveError("compile-cache cleanup receipt is not on configured remote")
        journal["cache_receipt_commit"] = cache_sha
        final_sha = cache_sha
        report.add(f"{pid}/full: exact job-owned compile caches pruned and receipt-verified ({cache_sha})")
        journal["phase"] = "complete"
        journal["completed_at"] = self.now_iso()
        self.store.write_artifact(pid, "full-lifecycle-journal.json", journal)
        completion_sha, completion_verification = self._commit_fixed_phase(
            [pdir], f"autoloop: complete fixed lifecycle for {pid} full",
            [journal_path, cache_receipt_path] + ([pdir / "full-checkpoint-pruned.json"] if expected_sha else []),
        )
        if not completion_verification["remote_verified"]:
            raise ArchiveError("fixed lifecycle completion journal is not on configured remote")
        final_sha = completion_sha
        self.store.set_stage(
            pid, "closed", last_archive_commit=final_sha, closed=True,
            archive_verified=verification, archive_receipt_verified=receipt_verification,
        )
        report.add(f"{pid}/full: definition-of-done complete -> closed")

    def _fixed_journal(self, pid: str, mode: str, path: Path) -> dict:
        journal = self.store.read_artifact(pid, path.name)
        if journal is None:
            return {
                "schema_version": 1, "proposal_id": pid, "tier": "full", "mode": mode,
                "phase": "initialized", "created_at": self.now_iso(),
            }
        if (journal.get("schema_version") != 1 or journal.get("proposal_id") != pid
                or journal.get("tier") != "full" or journal.get("mode") != mode):
            raise ArchiveError("fixed lifecycle journal identity/schema mismatch")
        return journal

    def _commit_fixed_phase(
        self, paths: list[Path], message: str, required: list[Path]
    ) -> tuple[str, dict]:
        commit = self.archiver.commit(paths, message)
        if commit is None:
            if not required:
                raise ArchiveError("cannot reconcile an empty fixed archive phase")
            head = getattr(self.archiver, "current_head", None)
            if head is None:
                raise ArchiveError("archiver cannot reconcile an already-committed phase")
            commit = head()
        verification = self.archiver.verify_commit(commit, required)
        return commit, verification

    def _archive_fixed_failure(self, pid: str, tier: str, report: CycleReport) -> None:
        if tier != "full":
            raise QueuePlanError("fixed failure archive is full-only")
        pdir = self.store.proposal_dir(pid)
        failure_path = pdir / "full-failure.json"
        failure = self.store.read_artifact(pid, "full-failure.json") or {}
        if (failure.get("outcome") != "failed" or failure.get("final") is not True
                or (failure.get("process") or {}).get("quiescent") is not True):
            raise ArchiveError("fixed failure evidence is incomplete or not quiescent")
        required = [
            pdir / "proposal.json",
            C.ROOT / next(e.verification_path for e in self.fixed_plan.entries if e.proposal_id == pid),
            pdir / "full-manifest.json", pdir / "full-staged.json", pdir / "full-launch.json",
            failure_path,
        ]
        for item in (failure.get("files") or {}).values():
            if item.get("available") is True:
                path = Path(item.get("local", ""))
                if path.is_symlink() or not path.is_file():
                    raise ArchiveError("declared failure artifact is missing/non-regular")
                try:
                    path.resolve().relative_to((pdir / "artifacts" / "full-failure").resolve())
                except ValueError as exc:
                    raise ArchiveError("declared failure artifact escaped its evidence directory") from exc
                if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
                    raise ArchiveError("declared failure artifact hash changed")
                required.append(path)
        journal_path = pdir / "full-failure-lifecycle-journal.json"
        journal = self._fixed_journal(pid, "failure", journal_path)
        if journal.get("phase") == "complete":
            receipt_commit, completion = self._commit_fixed_phase(
                [pdir], f"autoloop: complete failed fixed run {pid} full",
                [journal_path, failure_path],
            )
            if not completion.get("remote_verified"):
                raise ArchiveError("completed fixed failure lifecycle is not on configured remote")
            verification = journal.get("archive_verified") or {}
        elif journal.get("archive_commit"):
            commit = journal["archive_commit"]
            verification = self.archiver.verify_commit(commit, required)
        else:
            journal["phase"] = "failure_archive_pending"
            self.store.write_artifact(pid, journal_path.name, journal)
            commit, verification = self._commit_fixed_phase(
                [pdir], f"autoloop: archive failed fixed run {pid} full",
                required + [journal_path],
            )
            journal.update({
                "phase": "failure_archive_verified", "archive_commit": commit,
                "archive_verified": verification,
            })
            self.store.write_artifact(pid, journal_path.name, journal)
        if not verification.get("remote_verified"):
            raise ArchiveError("fixed failure evidence is not verified on configured remote")
        if journal.get("phase") != "complete":
            archive_verification = self.store.write_artifact(
                pid, "full-failure-archive-verification.json", verification
            )
            receipt_commit = journal.get("archive_receipt_commit")
            if receipt_commit:
                receipt_verification = self.archiver.verify_commit(
                    receipt_commit, [archive_verification]
                )
            else:
                receipt_commit, receipt_verification = self._commit_fixed_phase(
                    [pdir], f"autoloop: verify failed fixed run {pid} full",
                    [archive_verification],
                )
                journal.update({
                    "phase": "failure_archive_receipt_verified",
                    "archive_receipt_commit": receipt_commit,
                })
                self.store.write_artifact(pid, journal_path.name, journal)
            if not receipt_verification.get("remote_verified"):
                raise ArchiveError("fixed failure verification receipt is not on configured remote")
            journal.update({"phase": "complete", "completed_at": self.now_iso()})
            self.store.write_artifact(pid, journal_path.name, journal)
            receipt_commit, completion = self._commit_fixed_phase(
                [pdir], f"autoloop: complete failed fixed run {pid} full",
                [journal_path, failure_path, archive_verification],
            )
            if not completion.get("remote_verified"):
                raise ArchiveError("fixed failure completion journal is not on configured remote")
        self.store.set_stage(
            pid, "failed:full", closed=True, verdict="FAILED",
            last_archive_commit=receipt_commit, archive_verified=verification,
        )
        self.store.open_breaker(
            "fixed run failed", proposal_id=pid, tier="full",
            failure_reason=failure.get("reason"),
        )
        report.add(f"{pid}/full: distinct failure evidence archived -> failed")

    def _append_decision_log(self, pid: str, tier: str, proposal: dict, verdict: dict) -> None:
        date = self.now_iso()[:10]
        rec = self.store.proposals()[pid]
        if proposal.get("kind") == "control":
            change = "(control)"
        elif proposal.get("change_type") == "code":
            change = "code: " + proposal.get("single_causal_change", "")[:80]
        else:
            change = " ".join(proposal.get("flags", []))
        commit = rec.get("candidate_commit") or C.AUTHORITATIVE_COMMIT
        m = verdict.get("metrics", {})
        result = f"cand {m.get('candidate_bpb')} vs ctrl {m.get('control_bpb')} bpb, steps {m.get('candidate_steps')}/{m.get('control_steps')}" if m else "; ".join(verdict.get("reasons", []))[:120]
        row = f"| {date} | autoloop {pid} (`{change}`) | {commit[:8]} | {self._control_id(rec['box'])} | {result} | {verdict['decision']} ({tier}) | `research/agentic/experiments/{pid}/` |\n"
        with C.DECISION_LOG.open("a") as fh:
            fh.write(row)

    # ---- advancing a box ---------------------------------------------------------------------

    def _advance_box(self, box: str, report: CycleReport) -> bool:
        """Returns True if a launch happened on this box."""
        # Never launch while any proposal is analyzed-but-unarchived (evidence must be in git first).
        for pid, rec in self.store.proposals().items():
            if rec["stage"].startswith(("analyzed:", "collected:")):
                report.add(f"{box}: waiting; {pid} not yet archived")
                return False
        ok, why = self.store.runs_allowed(self.cfg.caps)
        if not ok:
            report.add(f"{box}: {why}")
            return False
        # 1. control ladder first
        cid = self._control_id(box)
        crec = self.store.proposals().get(cid)
        if crec is None:
            self.store.add_proposal({"proposal_id": cid, "kind": "control", "hypothesis": "matched control at authoritative commit", "single_causal_change": "none", "flags": [], "evidence": ["research/experiment-protocol.md §5"]})
            self.store.set_stage(cid, "verified", box=box, created_day=self.now_iso()[:10].replace("-", ""))
            crec = self.store.proposals()[cid]
        if crec["stage"] != "closed":
            if crec["stage"] == "verified":
                tier = "full" if "smoke" in crec["tiers"] else "smoke"
                return self._stage_and_launch(cid, tier, box, [], report)
            report.add(f"{box}: control in stage {crec['stage']}")
            return False
        if crec.get("verdict") != "KEEP":
            report.add(f"{box}: control failed; box unusable without human review")
            return False
        # 2. A proposal that passed the proposal verifier but was deliberately deferred until a
        # complete Implementer + independent implementation-Verifier pair can run atomically.
        for pid, rec in self.store.proposals().items():
            if rec["stage"] != "proposal_verified" or rec.get("box") not in (None, box):
                continue
            data = self.store.read_artifact(pid, "proposal.json")
            prop = Proposal.from_model_json(
                data,
                str(data.get("source_model_id", self.cfg.researcher_model_id)),
                str(data.get("created_at", self.now_iso())),
            )
            if prop.change_type == "code":
                ok, why = self._llm_calls_available(2)
                if not ok:
                    report.add(f"implement: {pid} deferred; requires two agent calls: {why}")
                    return False
                if not self._implement(prop, report):
                    return False
            self.store.set_stage(pid, "verified", box=box)
            return self._stage_and_launch(pid, "smoke", box, prop.flags, report)

        # 3. an already-verified candidate?
        for pid, rec in self.store.proposals().items():
            if rec["stage"] != "verified" or rec.get("box") not in (None, box):
                continue
            prop = self.store.read_artifact(pid, "proposal.json")
            if prop.get("kind") == "control":
                continue
            smoke_verdict = self.store.read_artifact(pid, "smoke-verdict.json") or {}
            tier = "full" if smoke_verdict.get("decision") == "KEEP" else "smoke"
            self.store.set_stage(pid, "verified", box=box)
            return self._stage_and_launch(pid, tier, box, prop["flags"], report)
        # 4. propose + verify a new one
        prop = self._propose(report)
        if prop is None:
            return False
        if not self._verify(prop, report):
            return False
        sha = self.archiver.commit(
            [self.store.proposal_dir(prop.proposal_id)],
            f"autoloop: {prop.proposal_id} verified proposal",
        )
        self.store.set_stage(prop.proposal_id, "proposal_verified", last_archive_commit=sha)
        if prop.change_type == "code":
            ok, why = self._llm_calls_available(2)
            if not ok:
                report.add(
                    f"implement: {prop.proposal_id} deferred; requires two agent calls: {why}"
                )
                return False
            if not self._implement(prop, report):
                return False
        self.store.set_stage(prop.proposal_id, "verified", box=box)
        return self._stage_and_launch(prop.proposal_id, "smoke", box, prop.flags, report)

    def _stage_and_launch(self, pid: str, tier: str, box: str, flags: list[str], report: CycleReport,
                          *, execution_mode: str = "standard", comparison_role: str | None = None,
                          expected_train_sha256: str = "",
                          control_label_override: str | None = None) -> bool:
        rec = self.store.proposals()[pid]
        attempts = rec.setdefault("attempts", {})
        r = attempts.get(tier, 0) + 1
        attempts[tier] = r
        self.store.save()
        if pid.startswith("control-"):
            label = f"al-{rec['created_day']}-{pid}-{tier}-r{r}"
        else:
            label = f"al-{pid}-{tier}-r{r}"
        if label in self.all_labels():
            raise ManifestError(f"label {label} already used")
        inv = self.op.inventory(box)
        self.store.write_artifact(pid, f"{tier}-inventory.json", inv.to_dict())
        if not inv.clean_and_idle:
            report.add(f"{box}: not clean/idle: {inv.findings}")
            n = self.store.box_failure(box)
            if n >= self.cfg.caps.max_consecutive_infra_failures_per_box:
                self.store.open_breaker("box repeatedly not clean/idle", box=box, findings=inv.findings)
            return False
        if not self.op.acquire_lock(box):
            owner = self.op.lock_owner(box)
            if owner != self.store.controller_id:
                self.store.open_breaker("box lock held by another controller", box=box, owner=owner)
                report.add(f"{box}: lock held by {owner}")
                return False
        candidate_commit = rec.get("candidate_commit") or C.AUTHORITATIVE_COMMIT
        remote_run_dir = f"{C.REMOTE_RUNS_DIR}/{candidate_commit[:8]}--{pid}--{label}"
        control_label = control_label_override
        if execution_mode == "standard" and not pid.startswith("control-"):
            ctrl = self._control_result(box, tier)
            if ctrl is None:
                report.add(f"{box}: no usable {tier} control; cannot launch candidate")
                self.op.release_lock(box)
                return False
            control_label = ctrl["label"]
        proposal = self.store.read_artifact(pid, "proposal.json")
        manifest = build_manifest(
            proposal_id=pid, tier=tier, box=box, remote_run_dir=remote_run_dir, label=label, train_flags=flags,
            cfg=self.cfg, frozen_at=self.now_iso(), control_label=control_label,
            abort_conditions=proposal.get("abort_conditions", []), description_note=proposal.get("single_causal_change", "")[:160],
            candidate_commit=candidate_commit,
            execution_mode=execution_mode, comparison_role=comparison_role,
            expected_train_sha256=expected_train_sha256,
        )
        self.store.write_artifact(pid, f"{tier}-manifest.json", manifest.to_dict())
        staged = self.op.stage(box, pid, label, candidate_commit)
        if staged.run_dir != remote_run_dir:
            raise RemoteError("staged dir differs from manifest dir")
        self.store.write_artifact(pid, f"{tier}-staged.json", staged.to_dict())
        self.store.tier(pid, tier).update({"label": label, "box": box})
        self.store.set_stage(pid, f"staged:{tier}", box=box)
        return self._launch_persisted_stage(pid, tier, report)

    # ---- propose / verify ---------------------------------------------------------------------

    def _implement(self, prop: Proposal, report: CycleReport) -> bool:
        if self.model is None:
            report.add(f"implement: {prop.proposal_id} no model client")
            return False
        try:
            result = implement_code_proposal(
                prop,
                experiment_dir=self.store.proposal_dir(prop.proposal_id),
                cfg=self.cfg,
                client=self.model,
                now=self.now_iso(),
                can_call=self._llm_ok,
                record_usage=self._record_usage,
            )
        except (ImplementationError, ModelError) as exc:
            result_data = {
                "passed": False,
                "candidate_commit": None,
                "error": str(exc),
                "implemented_at": self.now_iso(),
            }
            self.store.write_artifact(prop.proposal_id, "implementation.json", result_data)
            self.store.set_stage(prop.proposal_id, "rejected", verdict="REJECTED", failure=str(exc)[:500])
            self.archiver.commit(
                [self.store.proposal_dir(prop.proposal_id)],
                f"autoloop: {prop.proposal_id} implementation failed",
            )
            report.add(f"implement: {prop.proposal_id} failed: {str(exc)[:240]}")
            return False
        self.store.write_artifact(prop.proposal_id, "implementation.json", result.to_dict())
        if not result.passed or not result.candidate_commit:
            self.store.set_stage(prop.proposal_id, "rejected", verdict="REJECTED")
            self.archiver.commit(
                [self.store.proposal_dir(prop.proposal_id)],
                f"autoloop: {prop.proposal_id} implementation rejected",
            )
            report.add(
                f"implement: {prop.proposal_id} rejected: det={result.deterministic_findings} "
                f"llm={result.verifier_verdict} {result.verifier_findings[:3]}"
            )
            return False
        sha = self.archiver.commit(
            [self.store.proposal_dir(prop.proposal_id)],
            f"autoloop: {prop.proposal_id} implementation verified",
        )
        self.store.set_stage(
            prop.proposal_id,
            "implemented",
            candidate_commit=result.candidate_commit,
            candidate_branch=result.branch,
            last_archive_commit=sha,
        )
        report.add(f"implement: {prop.proposal_id} PASS commit={result.candidate_commit[:12]}")
        return True

    def _ledger_summary(self, limit: int = 60) -> str:
        """Return a bounded recent-outcome view; the complete ledger remains authoritative on disk."""
        rows = []
        for l in self.all_ledgers():
            rows.extend(l.rows)
        rows.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)
        out = []
        keys = (
            "timestamp", "label", "git_commit", "status", "public_val_bpb", "num_steps",
            "world_size", "lnc", "seq_len", "num_params_M", "compiled", "causality_passed",
        )
        for r in rows[: min(max(limit, 0), 60)]:
            compact = {k: r.get(k) for k in keys if r.get(k) is not None}
            if r.get("launch_command"):
                compact["launch_command"] = _clip_text(r["launch_command"], 480)
            if r.get("description"):
                compact["description"] = _clip_text(r["description"], 240)
            out.append(json.dumps(compact))
        return "\n".join(out)

    def _recent_rejection_feedback(self, limit: int = 5) -> str:
        """Return compact, durable verifier feedback so Researcher iterations can improve."""
        rejected = [
            (pid, rec) for pid, rec in self.store.proposals().items()
            if rec.get("stage") == "rejected"
        ]
        rejected.sort(key=lambda item: str(item[1].get("updated_at", "")), reverse=True)
        rows: list[str] = []
        for pid, _rec in rejected[:limit]:
            proposal = self.store.read_artifact(pid, "proposal.json") or {}
            verification = self.store.read_artifact(pid, "verification.json") or {}
            implementation = self.store.read_artifact(pid, "implementation.json") or {}
            rows.append(json.dumps({
                "proposal_id": pid,
                "change_type": proposal.get("change_type"),
                "single_causal_change": _clip_text(proposal.get("single_causal_change", ""), 700),
                "flags": proposal.get("flags", []),
                "implementation_plan": _clip_text(proposal.get("implementation_plan", ""), 1000),
                "proposal_verifier_verdict": verification.get("llm_verdict"),
                "proposal_verifier_findings": _clip_findings(verification.get("llm_findings", [])),
                "deterministic_findings": _clip_findings(verification.get("deterministic_findings", [])),
                "implementation_verifier_verdict": implementation.get("verifier_verdict"),
                "implementation_verifier_findings": _clip_findings(
                    implementation.get("verifier_findings", [])
                ),
            }))
        return "\n".join(rows) if rows else "(none yet)"

    @staticmethod
    def _historical_mechanism_class(proposal: dict) -> str:
        """Best-effort class for pre-metadata proposals; never used as a run verdict."""
        declared = str(proposal.get("mechanism_class", ""))
        if declared in {"throughput", "per_step_quality", "schedule", "capacity", "data_regime"}:
            return declared
        proposal_id = str(proposal.get("_proposal_id", "")).lower()
        if any(term in proposal_id for term in ("batch-ramp", "epoch", "shuffle", "context")):
            return "data_regime"
        if any(term in proposal_id for term in ("bf16", "fp8", "nki", "kernel", "fused")):
            return "throughput"
        if any(term in proposal_id for term in ("dim64", "no-ngram", "width", "depth", "capacity")):
            return "capacity"
        if any(term in proposal_id for term in ("warmdown", "cooldown", "flat-lr", "decay", "freeze", "ema", "demon")):
            return "schedule"
        return "per_step_quality"

    def _outcome_memory(self, limit: int = 12) -> str:
        """Compact proposal-to-outcome memory, diversified across mechanism classes.

        Raw ledger rows make it hard for the Researcher to learn which causal claim produced a
        result.  This joins the durable proposal and verdict artifacts.  Historical mechanism
        classes are explicitly marked inferred and are planning hints only.
        """
        records: list[dict] = []
        for pid, state_record in self.store.proposals().items():
            proposal = self.store.read_artifact(pid, "proposal.json") or {}
            smoke = self.store.read_artifact(pid, "smoke-verdict.json") or {}
            full = self.store.read_artifact(pid, "full-verdict.json") or {}
            if not smoke and not full:
                continue
            declared = proposal.get("mechanism_class")
            mechanism = self._historical_mechanism_class({**proposal, "_proposal_id": pid})
            records.append({
                "proposal_id": pid,
                "updated_at": state_record.get("updated_at"),
                "mechanism_class": mechanism,
                "mechanism_class_inferred": declared not in {
                    "throughput", "per_step_quality", "schedule", "capacity", "data_regime"
                },
                "single_causal_change": _clip_text(proposal.get("single_causal_change", ""), 650),
                "flags": proposal.get("flags", []),
                "predicted_bpb_delta": proposal.get("predicted_bpb_delta"),
                "smoke_decision": smoke.get("decision"),
                "smoke_bpb_delta": (smoke.get("metrics") or {}).get("bpb_delta"),
                "full_decision": full.get("decision"),
                "full_bpb_delta": (full.get("metrics") or {}).get("bpb_delta"),
            })
        records.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
        selected: list[dict] = []
        class_counts: dict[str, int] = {}
        for record in records:
            cls = str(record["mechanism_class"])
            if class_counts.get(cls, 0) >= 3:
                continue
            selected.append(record)
            class_counts[cls] = class_counts.get(cls, 0) + 1
            if len(selected) >= limit:
                break
        return "\n".join(json.dumps(row) for row in selected) if selected else "(none yet)"

    def _control_evidence(self) -> str:
        """Expose the exact pinned full controls instead of asking a model to infer defaults."""
        rows: list[str] = []
        for box in self.cfg.boxes:
            pid = self._control_id(box)
            result = self.store.read_artifact(pid, "full-result.json") or {}
            row = result.get("ledger_row", {})
            if row:
                rows.append(json.dumps({
                    "box": box,
                    "label": row.get("label"),
                    "git_commit": row.get("git_commit"),
                    "launch_command": row.get("launch_command"),
                    "environment": row.get("environment"),
                    "num_steps": row.get("num_steps"),
                    "public_val_bpb": row.get("public_val_bpb"),
                    "world_size": row.get("world_size"),
                    "lnc": row.get("lnc"),
                    "causality_passed": row.get("causality_passed"),
                }))
        return "\n".join(rows) if rows else "(controls not complete)"

    def _train_evidence(self, prop: Proposal, train_text: str, max_regions: int = 16) -> str:
        """Select source regions relevant to a proposal, including backing flag constants."""
        lines = train_text.splitlines()
        centers: set[int] = set()
        constant_names: set[str] = set()
        for flag in prop.flag_names():
            needle = f'"{flag}"'
            for i, line in enumerate(lines):
                if needle in line or f"'{flag}'" in line:
                    centers.add(i)
                    snippet = "\n".join(lines[i:min(len(lines), i + 10)])
                    constant_names.update(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", snippet))
                    break
        for name in constant_names:
            pattern = re.compile(rf"^{re.escape(name)}\s*=")
            for i, line in enumerate(lines):
                if pattern.match(line):
                    centers.add(i)
                    break

        # Code proposals have no flags. Ground their plan in bounded lexical source excerpts.
        if prop.change_type == "code":
            raw = " ".join([
                prop.hypothesis, prop.single_causal_change, prop.implementation_plan,
                *prop.duplicate_search_terms,
            ]).lower()
            stop = {
                "about", "after", "before", "change", "code", "current", "exact", "from",
                "into", "only", "proposal", "should", "single", "that", "their", "this",
                "train", "with", "without",
            }
            terms = {t for t in re.findall(r"[a-z_][a-z0-9_]{3,}", raw) if t not in stop}
            scored: list[tuple[int, int]] = []
            for i, line in enumerate(lines):
                low = line.lower()
                score = sum(1 for term in terms if term in low)
                if score:
                    scored.append((score, i))
            for _score, i in sorted(scored, key=lambda item: (-item[0], item[1]))[:max_regions]:
                centers.add(i)

        regions: list[tuple[int, int]] = []
        for center in sorted(centers)[:max_regions]:
            lo, hi = max(0, center - 5), min(len(lines), center + 11)
            if regions and lo <= regions[-1][1] + 2:
                regions[-1] = (regions[-1][0], max(regions[-1][1], hi))
            else:
                regions.append((lo, hi))
        if not regions:
            return "(no lexical source excerpt found; require a more precise implementation plan)"
        return "\n".join(
            f"--- train.py lines {lo + 1}-{hi} ---\n" + "\n".join(lines[lo:hi])
            for lo, hi in regions
        )

    def _research_train_evidence_index(self, train_text: str, max_chars: int = 28000) -> str:
        """Bounded source excerpts around durable measured/closed comments for the Researcher.

        The Researcher cannot inspect the repository interactively.  A list of flag names alone
        encouraged it to call historically measured flags untested and to invent their scope.
        Keep the independent Verifier's complete-source review, while exposing enough authoritative
        source evidence to avoid wasting proposal calls on already-closed axes.
        """
        lines = train_text.splitlines()
        marker = re.compile(
            r"\b(?:MEASURED|CLOSED|EXACT NULL|STAYS OFF|RETAINED OFF|PARKED|ON HOLD|"
            r"NEGATIVE|REGRESSION|NO WIN|DO NOT REOPEN)\b",
            re.IGNORECASE,
        )
        centers = [i for i, line in enumerate(lines) if marker.search(line)]
        regions: list[tuple[int, int]] = []
        for center in centers:
            lo, hi = max(0, center - 12), min(len(lines), center + 7)
            if regions and lo <= regions[-1][1] + 2:
                regions[-1] = (regions[-1][0], max(regions[-1][1], hi))
            else:
                regions.append((lo, hi))

        chunks: list[str] = []
        used = 0
        for lo, hi in regions:
            chunk = f"--- train.py lines {lo + 1}-{hi} ---\n" + "\n".join(lines[lo:hi])
            if used + len(chunk) + 1 > max_chars:
                break
            chunks.append(chunk)
            used += len(chunk) + 1
        return "\n".join(chunks) if chunks else "(no measured/closed source comments found)"

    def _propose(self, report: CycleReport) -> Proposal | None:
        backoff = self.store.state.get("research_backoff_until")
        if backoff:
            try:
                until = datetime.strptime(backoff, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if self._now() < until:
                    report.add(f"propose: backoff until {backoff}")
                    return None
            except ValueError:
                self.store.state.pop("research_backoff_until", None)
                self.store.save()
        ok, why = self._llm_ok()
        if not ok:
            report.add(f"propose: {why}")
            return None
        context = load_context([C.AGENT_DIR / "CURRENT_STATE.md", C.DECISION_LOG, C.ROOT / "research" / "experiment-protocol.md"])
        flags = sorted(self.known_flags())
        assert self._train_py_text is not None
        source_evidence = self._research_train_evidence_index(self._train_py_text)
        user = (
            f"Pinned commit: {C.AUTHORITATIVE_COMMIT}\nShape: LNC={self.cfg.lnc} nproc={self.cfg.nproc_per_node}, seed 42, 1800 s.\n"
            f"Proposal ids already used: {sorted(self.store.used_proposal_ids())}\n\n"
            f"Available train.py flags at this commit:\n{' '.join(flags)}\n\n"
            f"Authoritative measured/closed train.py evidence excerpts "
            f"(bounded; absence does not mean untested):\n{source_evidence}\n\n"
            f"Exact pinned full controls:\n{self._control_evidence()}\n\n"
            f"Recent rejected proposals and verifier feedback (do not repeat these defects):\n"
            f"{self._recent_rejection_feedback()}\n\n"
            f"Proposal-to-outcome memory (calibrate predictions; inferred classes are hints):\n"
            f"{self._outcome_memory()}\n\n"
            f"Recent ledger rows (newest first):\n{self._ledger_summary()}\n\n{context}"
        )
        reply = None
        try:
            # Fable uses an always-on reasoning budget. A 16K cap returned empty text, while 64K
            # plus an oversized evidence view exceeded the observed five-minute idle connection
            # window. The bounded context and 32K cap retain reasoning headroom without inviting
            # another transport-length failure.
            reply = self.model.complete(role="researcher", model_id=self.cfg.researcher_model_id, system=RESEARCHER_SYSTEM, user=user, max_tokens=32000)
            self._record_usage("researcher", reply)
            data = parse_json_block(reply.text)
        except ModelError as exc:
            until = self._now() + timedelta(minutes=30)
            self.store.state["research_backoff_until"] = until.strftime("%Y-%m-%dT%H:%M:%SZ")
            self.store.save()
            if reply is not None:
                d = C.RUNTIME_DIR / "model-replies"
                d.mkdir(parents=True, exist_ok=True)
                stamp = self.now_iso().replace(":", "")
                (d / f"researcher-failed-{stamp}.json").write_text(json.dumps({
                    "model_id": reply.model_id,
                    "stop_reason": reply.stop_reason,
                    "input_tokens": reply.input_tokens,
                    "output_tokens": reply.output_tokens,
                    "text": reply.text,
                    "error": str(exc),
                }, indent=2) + "\n")
            report.add(f"propose: model error {exc}; backing off 30m")
            return None
        if str(data.get("proposal_id", "none")) == "none":
            self.store.state["research_backoff_until"] = (self._now() + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.store.save()
            report.add(f"propose: researcher declined: {data.get('reason', '')[:200]}")
            return None
        try:
            prop = Proposal.from_model_json(
                data,
                self.cfg.researcher_model_id,
                self.now_iso(),
                require_research_metadata=True,
            )
        except ManifestError as exc:
            self.store.state["research_backoff_until"] = (self._now() + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.store.save()
            report.add(f"propose: rejected malformed proposal: {exc}")
            return None
        if prop.proposal_id in self.store.used_proposal_ids():
            self.store.state["research_backoff_until"] = (self._now() + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.store.save()
            report.add("propose: duplicate proposal id")
            return None
        self.store.state.pop("research_backoff_until", None)
        self.store.save()
        self.store.add_proposal(prop.to_dict())
        report.add(f"propose: {prop.proposal_id} type={prop.change_type} flags={' '.join(prop.flags)}")
        return prop

    def _verify(self, prop: Proposal, report: CycleReport) -> bool:
        ok, why = self._llm_ok()
        client = self.model if ok else None
        current_state = (C.AGENT_DIR / "CURRENT_STATE.md").read_text() if (C.AGENT_DIR / "CURRENT_STATE.md").is_file() else ""
        train_text = self._train_py_text or self.git.file_at_commit(C.AUTHORITATIVE_COMMIT, "train.py").decode(errors="replace")
        verifier_context = load_context([
            C.ROOT / "README.md",
            C.ROOT / "AGENTS.md",
            C.AGENT_DIR / "CURRENT_STATE.md",
            C.ROOT / "research" / "experiment-protocol.md",
            C.DECISION_LOG,
            C.ROOT / "research" / "trn2-experiment-results.md",
        ])
        verifier_context += "\n--- EXACT PINNED FULL CONTROLS ---\n" + self._control_evidence()
        verifier_context += "\n--- RECENT LEDGER ROWS ---\n" + self._ledger_summary(200)
        verifier_context += "\n--- RELEVANT AUTHORITATIVE train.py EVIDENCE ---\n" + self._train_evidence(prop, train_text)
        # A lexical excerpt is useful for orientation, but it cannot prove transitive flag
        # semantics (argument -> global -> helper -> module construction -> forward/optimizer).
        # The 2026-09-22 trigram proposals demonstrated the failure mode: the excerpt showed the
        # CLI definition but omitted the downstream table construction, so an otherwise bounded
        # proposal repeatedly returned NEEDS_WORK for missing evidence that existed in the pinned
        # source.  The verifier is the only role that receives this complete source; the Researcher
        # still works from compact evidence, and no model receives tools.
        verifier_context += (
            "\n--- COMPLETE AUTHORITATIVE train.py (pinned commit; source of truth) ---\n"
            + train_text
            + "\n--- END COMPLETE AUTHORITATIVE train.py ---\n"
        )
        verification = run_verifier(
            prop, cfg=self.cfg, client=client,
            context_text=verifier_context,
            known_flags=self.known_flags(), ledgers=self.all_ledgers(), used_labels=self.all_labels(),
            used_proposal_ids=self.store.used_proposal_ids() - {prop.proposal_id},
            closed_terms=closed_direction_terms(current_state), now=self.now_iso(), record_usage=self._record_usage,
        )
        self.store.write_artifact(prop.proposal_id, "verification.json", verification.to_dict())
        if not verification.passed:
            self.store.set_stage(prop.proposal_id, "rejected", verdict="REJECTED")
            self.archiver.commit([self.store.proposal_dir(prop.proposal_id)], f"autoloop: {prop.proposal_id} rejected by verifier")
            report.add(f"verify: {prop.proposal_id} rejected: det={verification.deterministic_findings} llm={verification.llm_verdict}")
            return False
        report.add(f"verify: {prop.proposal_id} PASS by {verification.verifier_model_id}")
        return True

    # ---- daemon -----------------------------------------------------------------------------

    def daemon(self, *, max_cycles: int = 0, sleep=time.sleep) -> int:
        n = 0
        while max_cycles == 0 or n < max_cycles:
            rep = self.cycle()
            n += 1
            for a in rep.actions:
                print(a)
            if rep.stopped_reason:
                print(f"STOPPED: {rep.stopped_reason}")
                return 3
            sleep(self.cfg.caps.poll_interval_seconds)
        return 0
