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
from .state import Store, utc_now
from .verifier import closed_direction_terms, load_context, run_verifier

RESEARCHER_SYSTEM = """You are the Researcher in an unattended Trainium Phase 1 research loop.
Goal: lower official val_bpb after a 30-minute full-chip Trn2 run. Propose ONE evidence-backed,
falsifiable causal change. It may be an existing train.py flag experiment or a small code change
confined to train.py (architecture, optimizer, schedule, numerical format, or inline NKI kernel).
Prefer a bounded code proposal when the remaining flag ideas are already heavily explored; use a
flag only when the supplied evidence establishes that it is a real, isolated, non-duplicate gap.
Seed 42, full-chip shape, data, evaluator, and budget are fixed by the controller.
Search the provided ledger summary and decision log; never repeat a closed or already-run flag set
at this commit. A flag proposal must set one causal option only; do not bundle a second precision,
kernel, optimizer, or architecture switch. If two switches are truly inseparable, propose a code
change that implements one explicit boundary instead. Cite repository evidence. Return ONLY a JSON object:
{"proposal_id": "YYYYMMDD-short-slug", "hypothesis": "...", "single_causal_change": "...",
 "flags": ["--flag", "value", ...], "evidence": ["path or label ..."],
 "duplicate_search_terms": ["..."], "expected_effect": "predeclared BPB direction and size",
 "abort_conditions": ["..."], "change_type":"flags"|"code",
 "smoke_gate":{"min_candidate_steps":123,"min_step_rate_ratio":1.0,"max_bpb_delta":0.0},
 "implementation_plan":"exact bounded edit plan, empty for flags",
 "files_allowed":[]|["train.py"]}
smoke_gate must contain only thresholds justified by the proposal; use {} when the controller's
global smoke gate is sufficient. Proposal thresholds can only tighten the global gate. For flags,
files_allowed must be [] and implementation_plan empty. For code, flags must be [],
files_allowed must be ["train.py"], and the plan must be precise enough for another isolated agent
to implement without inventing a second change. Do not propose evaluator/tokenizer/data changes,
external data, pretrained assets, or private validation access. If no non-duplicate hypothesis exists, return
{"proposal_id": "none", "reason": "..."}."""

ANALYST_SYSTEM = """You are the Analyst. You receive a deterministic verdict plus the candidate and
control ledger rows. Write at most 12 lines of plain prose explaining the comparison, naming any
confounder. You may not change the decision. Return plain text."""


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
            report.stopped_reason = reason
            return report
        self.store.state["cycles"] += 1
        self.store.state["last_cycle"] = {"started_at": report.started_at}
        self.store.save()

        try:
            busy_boxes = self._reconcile(report)
            if self.breakers():
                report.stopped_reason = self.breakers()
                return report
            for box in self.cfg.boxes:
                if box in busy_boxes:
                    continue
                if self._advance_box(box, report):
                    break  # one launch per cycle
        except (RemoteError, ManifestError, ArchiveError) as exc:
            self.store.open_breaker(type(exc).__name__, detail=str(exc)[:1000])
            report.stopped_reason = f"{type(exc).__name__}: {exc}"
        self.store.state["last_cycle"]["finished_at"] = self.now_iso()
        self.store.state["last_cycle"]["actions"] = report.actions
        self.store.save()
        report.llm_calls = self._cycle_llm_calls
        return report

    # ---- reconcile running/collected work -------------------------------------------------

    def _reconcile(self, report: CycleReport) -> set[str]:
        busy: set[str] = set()
        for pid, rec in list(self.store.proposals().items()):
            stage = rec["stage"]
            if stage.startswith("launched:"):
                tier = stage.split(":", 1)[1]
                if self._poll_and_collect(pid, tier, report):
                    stage = self.store.proposals()[pid]["stage"]
                else:
                    busy.add(rec["box"])
                    continue
            if stage.startswith("collected:"):
                tier = stage.split(":", 1)[1]
                self._analyze(pid, tier, report)
                stage = self.store.proposals()[pid]["stage"]
            if stage.startswith("analyzed:"):
                tier = stage.split(":", 1)[1]
                self._archive_and_promote(pid, tier, report)
            if stage.startswith("staged:"):
                # staged but never launched (crash between stage and launch): the run dir is
                # unused; stage again under a new r-number rather than trusting it.
                self.store.set_stage(pid, "verified", note="re-stage after interrupted launch")
        return busy

    def _poll_and_collect(self, pid: str, tier: str, report: CycleReport) -> bool:
        trec = self.store.tier(pid, tier)
        manifest = RunManifest.from_dict(self.store.read_artifact(pid, f"{tier}-manifest.json"))
        receipt = LaunchReceipt(**self.store.read_artifact(pid, f"{tier}-launch.json"))
        status = self.op.poll(manifest, receipt)
        trec["last_poll"] = status.to_dict()
        self.store.save()
        launched = datetime.strptime(receipt.launched_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        elapsed = (self._now() - launched).total_seconds()
        if not status.done:
            if status.pid_alive:
                if elapsed > self.cfg.caps.run_timeout_seconds:
                    report.add(f"{pid}/{tier}: still running past timeout ({elapsed:.0f}s); observing, branch marked FAILED")
                    self._fail_tier(pid, tier, manifest, "run exceeded controller timeout", infra=True)
                    return True
                report.add(f"{pid}/{tier}: running ({elapsed:.0f}s)")
                return False
            if elapsed < self.cfg.caps.launch_grace_seconds:
                report.add(f"{pid}/{tier}: pid not visible yet, within grace")
                return False
            self._fail_tier(pid, tier, manifest, "runner died without done marker", infra=True)
            return True
        # done: collect
        local_dir = self.store.proposal_dir(pid) / "artifacts" / tier
        collected = self.op.collect(manifest, receipt, local_dir)
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

    def _archive_and_promote(self, pid: str, tier: str, report: CycleReport) -> None:
        rec = self.store.proposals()[pid]
        verdict = self.store.read_artifact(pid, f"{tier}-verdict.json")
        decision = verdict["decision"]
        proposal = self.store.read_artifact(pid, "proposal.json")
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
        # 2. an already-verified candidate?
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
        # 3. propose + verify a new one
        prop = self._propose(report)
        if prop is None:
            return False
        if not self._verify(prop, report):
            return False
        if prop.change_type == "code":
            if not self._implement(prop, report):
                return False
        else:
            sha = self.archiver.commit(
                [self.store.proposal_dir(prop.proposal_id)],
                f"autoloop: {prop.proposal_id} verified proposal",
            )
            self.store.set_stage(prop.proposal_id, "proposal_verified", last_archive_commit=sha)
        self.store.set_stage(prop.proposal_id, "verified", box=box)
        return self._stage_and_launch(prop.proposal_id, "smoke", box, prop.flags, report)

    def _stage_and_launch(self, pid: str, tier: str, box: str, flags: list[str], report: CycleReport) -> bool:
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
        control_label = None
        if not pid.startswith("control-"):
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
        )
        self.store.write_artifact(pid, f"{tier}-manifest.json", manifest.to_dict())
        staged = self.op.stage(box, pid, label, candidate_commit)
        if staged.run_dir != remote_run_dir:
            raise RemoteError("staged dir differs from manifest dir")
        self.store.write_artifact(pid, f"{tier}-staged.json", staged.to_dict())
        self.store.tier(pid, tier).update({"label": label, "box": box})
        self.store.set_stage(pid, f"staged:{tier}", box=box)
        receipt = self.op.launch(manifest, staged)
        self.store.write_artifact(pid, f"{tier}-launch.json", receipt.to_dict())
        self.store.tier(pid, tier).update({"pid": receipt.pid, "launched_at": receipt.launched_at})
        self.store.set_stage(pid, f"launched:{tier}", box=box)
        self.store.record_run()
        report.add(f"{box}: launched {label} pid={receipt.pid} dir={remote_run_dir}")
        return True

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

    def _ledger_summary(self, limit: int = 120) -> str:
        rows = []
        for l in self.all_ledgers():
            rows.extend(l.rows)
        rows.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)
        out = []
        keys = (
            "timestamp", "label", "git_commit", "status", "public_val_bpb", "num_steps",
            "world_size", "lnc", "seq_len", "num_params_M", "compiled", "causality_passed",
            "launch_command", "environment", "description", "oneliners", "ngram_ve",
            "out_bigram", "engram", "sched_tricks", "batch_ramp", "weight_ema",
        )
        for r in rows[:limit]:
            out.append(json.dumps({k: r.get(k) for k in keys if r.get(k) is not None}))
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
                "single_causal_change": proposal.get("single_causal_change"),
                "flags": proposal.get("flags", []),
                "implementation_plan": proposal.get("implementation_plan", ""),
                "proposal_verifier_verdict": verification.get("llm_verdict"),
                "proposal_verifier_findings": verification.get("llm_findings", []),
                "deterministic_findings": verification.get("deterministic_findings", []),
                "implementation_verifier_verdict": implementation.get("verifier_verdict"),
                "implementation_verifier_findings": implementation.get("verifier_findings", []),
            }))
        return "\n".join(rows) if rows else "(none yet)"

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
        user = (
            f"Pinned commit: {C.AUTHORITATIVE_COMMIT}\nShape: LNC={self.cfg.lnc} nproc={self.cfg.nproc_per_node}, seed 42, 1800 s.\n"
            f"Proposal ids already used: {sorted(self.store.used_proposal_ids())}\n\n"
            f"Available train.py flags at this commit:\n{' '.join(flags)}\n\n"
            f"Exact pinned full controls:\n{self._control_evidence()}\n\n"
            f"Recent rejected proposals and verifier feedback (do not repeat these defects):\n"
            f"{self._recent_rejection_feedback()}\n\n"
            f"Recent ledger rows (newest first):\n{self._ledger_summary()}\n\n{context}"
        )
        reply = None
        try:
            # Fable uses an always-on reasoning budget.  The 2026-09-22 live call consumed all
            # 16,000 output tokens in hidden reasoning and returned empty text, so the structured
            # answer needs enough headroom to exist after reasoning completes.
            reply = self.model.complete(role="researcher", model_id=self.cfg.researcher_model_id, system=RESEARCHER_SYSTEM, user=user, max_tokens=64000)
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
            prop = Proposal.from_model_json(data, self.cfg.researcher_model_id, self.now_iso())
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
