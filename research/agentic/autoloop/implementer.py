"""Isolated code Implementer and independent implementation-verification gate.

The Implementer receives no tools and returns exact text replacements.  Replacements are applied
only inside a fresh local git worktree at the authoritative commit.  A deterministic gate and an
independent model-family review must both pass before a candidate commit is created.  The remote
Operator never sees a dirty tree and never receives model-authored shell commands.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config as C
from .manifest import Proposal
from .models import ModelClient, ModelError, ModelReply, parse_json_block


IMPLEMENTER_SYSTEM = """You are the isolated Implementer for one AWS Trainium experiment.
Implement exactly the approved single causal change in train.py. Do not change the evaluator,
tokenizer, data source, seed, time budget, output paths, or competition protocol. Do not add
network access, external data, pretrained assets, shell commands, subprocesses, or hidden scoring
logic. Prefer the smallest auditable implementation. You cannot run code or launch hardware.

Return ONLY one JSON object with this exact shape:
{"summary":"...","replacements":[{"old":"exact unique text copied from train.py","new":"replacement text"}],"notes":["..."]}
Every old string must occur exactly once. Include enough surrounding text to make it unique.
Use at most 12 replacements. Do not return a markdown diff or commentary outside JSON."""


IMPLEMENTATION_VERIFIER_SYSTEM = """You are the independent implementation Verifier. You did not
propose or implement this experiment. Review the approved proposal, the exact train.py patch, and
the deterministic check results and complete candidate train.py. This is a pre-run implementation
gate: do not require smoke, BPB, causality, checkpoint, or hardware results that can only exist after
this gate passes; the controller runs and records those gates later. The implementation pipeline
intentionally edits only train.py and does not create proposal-specific named pre-edit artifacts or
run CPU/unit/compile checks. Judge pre-edit claims from the authoritative source and deterministic
checks supplied here; do not reject solely because a separately named pre-edit artifact is absent.
PASS only if the patch implements exactly one causal change, is
competition-legal, preserves the 30-minute/full-chip/seed-42 evaluation contract, does not touch
data/evaluation/tokenizer behavior, has no external-data or pretrained path, preserves causal
load_for_eval behavior, and is small enough to attribute. You cannot edit or launch anything.
Return ONLY JSON: {"verdict":"PASS"|"FAIL"|"NEEDS_WORK","findings":["..."]}."""


FORBIDDEN_ADDED_RE = re.compile(
    r"(?i)(https?://|\brequests\b|\burllib\b|\bboto3\b|\bsubprocess\b|\bsocket\b|"
    r"prepare\.py|eval-public|private[_ -]?val|core[_ -]?(data|eval)|pretrained)"
)


@dataclass
class ImplementationResult:
    passed: bool
    candidate_commit: str | None
    branch: str
    diff_sha256: str | None
    deterministic_findings: list[str]
    verifier_verdict: str
    verifier_findings: list[str]
    implementer_model_id: str
    verifier_model_id: str
    implemented_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class ImplementationError(RuntimeError):
    pass


def _run(root: Path, argv: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=str(cwd or root), text=True, capture_output=True, check=False)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return _run(root, ["git", "-C", str(root), *args])


def _added_lines(diff: str) -> list[str]:
    return [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]


def _deterministic_checks(worktree: Path, original: str, candidate: str, diff: str) -> list[str]:
    findings: list[str] = []
    if candidate == original:
        findings.append("implementer made no change")
    status = _git(worktree, "status", "--porcelain").stdout.splitlines()
    if status != [" M train.py"]:
        findings.append(f"worktree changed files other than exactly train.py: {status}")
    changed = sum(1 for line in diff.splitlines() if line[:1] in {"+", "-"} and not line.startswith(("+++", "---")))
    if changed > 1500:
        findings.append(f"patch is too large for one attributable experiment: {changed} changed lines")
    if "def load_for_eval(" not in candidate:
        findings.append("load_for_eval hook was removed")
    try:
        ast.parse(candidate, filename="train.py")
    except SyntaxError as exc:
        findings.append(f"train.py syntax error: {exc}")
    bad_added = [line.strip() for line in _added_lines(diff) if FORBIDDEN_ADDED_RE.search(line)]
    if bad_added:
        findings.append("forbidden external/evaluation-related added code: " + repr(bad_added[:8]))
    check = _git(worktree, "diff", "--check")
    if check.returncode:
        findings.append("git diff --check failed: " + check.stderr.strip()[:500])
    return findings


def implement_code_proposal(
    proposal: Proposal,
    *,
    experiment_dir: Path,
    cfg: C.LoopConfig,
    client: ModelClient,
    now: str,
    can_call,
    record_usage,
) -> ImplementationResult:
    if proposal.change_type != "code" or proposal.files_allowed != ["train.py"]:
        raise ImplementationError("implementer received a non-code or out-of-scope proposal")
    branch = f"autoloop/{proposal.proposal_id}"
    worktree = C.RUNTIME_DIR / "worktrees" / proposal.proposal_id
    worktree.parent.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    reply_dir = C.RUNTIME_DIR / "model-replies" / proposal.proposal_id
    reply_dir.mkdir(parents=True, exist_ok=True)
    if worktree.exists():
        raise ImplementationError(f"generated worktree already exists: {worktree}")
    if _git(C.ROOT, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}").returncode == 0:
        raise ImplementationError(f"candidate branch already exists: {branch}")

    add = _git(C.ROOT, "worktree", "add", "-b", branch, str(worktree), C.AUTHORITATIVE_COMMIT)
    if add.returncode:
        raise ImplementationError("git worktree add failed: " + add.stderr.strip()[:500])

    passed = False
    implementer_model_id = cfg.implementer_model_id or cfg.researcher_model_id
    try:
        train_path = worktree / "train.py"
        original = train_path.read_text()
        ok, why = can_call()
        if not ok:
            raise ImplementationError(f"implementer model unavailable: {why}")
        user = (
            "APPROVED PROPOSAL:\n" + json.dumps(proposal.to_dict(), indent=2)
            + "\n\nAUTHORITATIVE train.py:\n```python\n" + original + "\n```"
        )
        reply = client.complete(
            role="implementer", model_id=implementer_model_id,
            system=IMPLEMENTER_SYSTEM, user=user, max_tokens=24000,
        )
        record_usage("implementer", reply)
        (reply_dir / "implementer.txt").write_text(reply.text)
        try:
            data = parse_json_block(reply.text)
        except ModelError as exc:
            raise ImplementationError(f"implementer returned invalid JSON ({reply.stop_reason}): {exc}") from exc
        replacements = data.get("replacements")
        if not isinstance(replacements, list) or not (1 <= len(replacements) <= 12):
            raise ImplementationError("implementer replacements must contain 1..12 entries")
        candidate = original
        total_new = 0
        for i, item in enumerate(replacements):
            if not isinstance(item, dict) or not isinstance(item.get("old"), str) or not isinstance(item.get("new"), str):
                raise ImplementationError(f"replacement {i} is malformed")
            old, new = item["old"], item["new"]
            if not old or candidate.count(old) != 1:
                raise ImplementationError(f"replacement {i} old text occurs {candidate.count(old)} times, expected once")
            total_new += len(new)
            if total_new > 250_000:
                raise ImplementationError("replacement output exceeds 250 KB")
            candidate = candidate.replace(old, new, 1)
        train_path.write_text(candidate)
        diff_proc = _git(worktree, "diff", "--", "train.py")
        diff = diff_proc.stdout
        (experiment_dir / "implementation.patch").write_text(diff)
        findings = _deterministic_checks(worktree, original, candidate, diff)
        (experiment_dir / "implementation-checks.json").write_text(json.dumps({"findings": findings}, indent=2) + "\n")
        verifier_verdict, verifier_findings = "SKIPPED", []
        if not findings:
            ok, why = can_call()
            if not ok:
                verifier_findings = [why]
            else:
                review = client.complete(
                    role="implementation_verifier", model_id=cfg.verifier_model_id,
                    system=IMPLEMENTATION_VERIFIER_SYSTEM,
                    user=("PROPOSAL:\n" + json.dumps(proposal.to_dict(), indent=2)
                          + "\n\nDETERMINISTIC CHECKS: PASS\n\nPATCH:\n```diff\n" + diff + "\n```"
                          + "\n\nCOMPLETE CANDIDATE train.py (source of truth):\n```python\n"
                          + candidate + "\n```"),
                    max_tokens=12000,
                )
                record_usage("implementation_verifier", review)
                (reply_dir / "implementation-verifier.txt").write_text(review.text)
                try:
                    vd = parse_json_block(review.text)
                    verifier_verdict = str(vd.get("verdict", "ERROR")).upper()
                    verifier_findings = [str(x) for x in vd.get("findings", [])][:30]
                except ModelError as exc:
                    verifier_verdict, verifier_findings = "ERROR", [f"invalid verifier JSON ({review.stop_reason}): {exc}"]
        if findings or verifier_verdict != "PASS":
            return ImplementationResult(
                False, None, branch, hashlib.sha256(diff.encode()).hexdigest() if diff else None,
                findings, verifier_verdict, verifier_findings, implementer_model_id,
                cfg.verifier_model_id, now,
            )
        add_train = _git(worktree, "add", "--", "train.py")
        if add_train.returncode:
            raise ImplementationError("git add train.py failed: " + add_train.stderr.strip()[:500])
        commit = _git(worktree, "commit", "-m", f"experiment: {proposal.proposal_id}")
        if commit.returncode:
            raise ImplementationError("candidate commit failed: " + commit.stderr.strip()[:500])
        rev = _git(worktree, "rev-parse", "HEAD")
        sha = rev.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ImplementationError("candidate commit SHA is invalid")
        changed_files = _git(worktree, "diff-tree", "--no-commit-id", "--name-only", "-r", sha).stdout.splitlines()
        if changed_files != ["train.py"]:
            raise ImplementationError(f"candidate commit changed unexpected files: {changed_files}")
        passed = True
        return ImplementationResult(
            True, sha, branch, hashlib.sha256(diff.encode()).hexdigest(), findings,
            verifier_verdict, verifier_findings, implementer_model_id, cfg.verifier_model_id, now,
        )
    finally:
        # Only this controller-created disposable worktree is removed.  The branch and the full
        # patch/review evidence remain durable in git/research/agentic.
        _git(C.ROOT, "worktree", "remove", *( [] if passed else ["--force"] ), str(worktree))
