"""Independent verifier gate.

Two layers, both mandatory:

1. Deterministic checks the model cannot argue with: flag existence at the pinned commit, forbidden
   flags, duplicate search against every ledger the controller has (local, archived, retrieved
   remote), label uniqueness, and proposal completeness.
2. An LLM Verifier from a *different model family* than the Researcher reviews the proposal against
   AGENTS.md / DECISION_LOG.md / CURRENT_STATE.md and must return ``{"verdict": "PASS"}``.

Only PASS + zero deterministic failures promotes a proposal. There is no override path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config as C
from .ledger import Ledger
from .manifest import ManifestError, Proposal, validate_flag_tokens
from .models import ModelClient, ModelError, parse_json_block

VERIFIER_SYSTEM = """You are the independent Verifier for a Trainium competition research loop.
You did not write this proposal. Review it against the repository contract and decision log.
Check: (1) single causal change; (2) flags or the code implementation plan describe exactly that change; (3) evidence is cited from
the repository; (4) it is not a duplicate of a closed direction; (5) it is competition-legal
(no pretrained weights, no external data, no evaluator changes); (6) the expected effect is
falsifiable with a seed-42 full-chip run against a matched control.
The deterministic gate has already validated exact flag-set duplication, legal flag names, unique
labels, and proposal structure before you are called. The controller—not the proposal—pins seed 42,
the authoritative base commit, LNC/world size, 1800-second budget, clean staging, exact evaluation,
and a same-box full control. Do not demand that the proposal restate those controller guarantees,
and do not reject merely because it does not narrate the deterministic duplicate search. For a code
proposal, judge the bounded causal plan now; a separate independent verifier reviews its exact patch.
Public BPB is the local proxy and official BPB is the final target; requiring a predicted official
number before a legal experiment is not a valid rejection reason.
For code proposals, the implementation stage may edit only train.py and perform deterministic
syntax/diff/static review before the smoke run. Return NEEDS_WORK if the proposal makes a named
pre-edit artifact, CPU/unit execution, compilation output, dtype log, or hardware result a
prerequisite for implementation approval; those executable checks belong in smoke abort conditions.
Return ONLY a JSON object: {"verdict": "PASS"|"FAIL"|"NEEDS_WORK", "findings": ["..."]}.
Mark PASS only with evidence. You cannot launch, edit, or approve hardware use."""


@dataclass
class Verification:
    proposal_id: str
    deterministic_ok: bool
    deterministic_findings: list[str]
    llm_verdict: str  # PASS | FAIL | NEEDS_WORK | SKIPPED | ERROR
    llm_findings: list[str]
    verifier_model_id: str
    researcher_model_id: str
    independent: bool
    verified_at: str
    duplicates_found: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.deterministic_ok and self.independent and self.llm_verdict == "PASS"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def deterministic_checks(
    proposal: Proposal,
    *,
    known_flags: set[str],
    ledgers: list[Ledger],
    used_labels: set[str],
    used_proposal_ids: set[str],
    closed_terms: set[str],
) -> tuple[list[str], list[dict]]:
    findings: list[str] = []
    duplicates: list[dict] = []
    if proposal.change_type == "flags":
        try:
            validate_flag_tokens(proposal.flags, known_flags=known_flags)
        except ManifestError as exc:
            findings.append(f"flags: {exc}")
    elif proposal.change_type == "code":
        if proposal.files_allowed != ["train.py"]:
            findings.append("code proposal may edit only train.py")
        if proposal.flags:
            findings.append("code proposal unexpectedly contains launch flags")
    else:
        findings.append("unknown change_type")
    if proposal.proposal_id in used_proposal_ids:
        findings.append("proposal_id already used")
    for tier in ("smoke", "full"):
        label = f"al-{proposal.proposal_id}-{tier}-r1"
        if label in used_labels:
            findings.append(f"label {label} already exists in a ledger")
    if proposal.change_type == "flags":
        names = proposal.flag_names()
        for ledger in ledgers:
            for row in ledger.rows_with_flags(names, commit_prefix=C.AUTHORITATIVE_COMMIT):
                duplicates.append({"ledger": ledger.source, "label": row.get("label"), "status": row.get("status")})
        if duplicates:
            findings.append("identical flag set already run at the authoritative commit")
    lowered = " ".join([proposal.hypothesis, proposal.single_causal_change, *proposal.duplicate_search_terms]).lower()
    hits = sorted(t for t in closed_terms if t and t.lower() in lowered)
    if hits:
        findings.append(f"touches closed directions without new evidence: {hits}")
    if len(proposal.evidence) < 1:
        findings.append("no evidence")
    return findings, duplicates


def closed_direction_terms(current_state_text: str) -> set[str]:
    """Pull the 'Do not reopen ...' list out of CURRENT_STATE.md (best effort, lowercase terms)."""
    terms: set[str] = set()
    for line in current_state_text.splitlines():
        if line.lower().startswith("do not reopen"):
            body = line.split("reopen", 1)[1]
            for chunk in body.replace(" or ", ",").split(","):
                t = chunk.strip(" .").lower()
                t = t.replace("the parked ", "").replace("previously tested ", "")
                if 4 <= len(t) <= 60 and "without" not in t:
                    terms.add(t)
    return terms


def run_verifier(
    proposal: Proposal,
    *,
    cfg: C.LoopConfig,
    client: ModelClient | None,
    context_text: str,
    known_flags: set[str],
    ledgers: list[Ledger],
    used_labels: set[str],
    used_proposal_ids: set[str],
    closed_terms: set[str],
    now: str,
    record_usage,
) -> Verification:
    findings, duplicates = deterministic_checks(
        proposal,
        known_flags=known_flags,
        ledgers=ledgers,
        used_labels=used_labels,
        used_proposal_ids=used_proposal_ids,
        closed_terms=closed_terms,
    )
    independent = C.model_family(cfg.verifier_model_id) != C.model_family(proposal.source_model_id or cfg.researcher_model_id)
    if not independent:
        findings.append("verifier model family equals researcher family; independence not met")
    verification = Verification(
        proposal_id=proposal.proposal_id,
        deterministic_ok=not findings,
        deterministic_findings=findings,
        llm_verdict="SKIPPED",
        llm_findings=[],
        verifier_model_id=cfg.verifier_model_id,
        researcher_model_id=proposal.source_model_id or cfg.researcher_model_id,
        independent=independent,
        verified_at=now,
        duplicates_found=duplicates,
    )
    if findings or client is None:
        if client is None:
            verification.llm_findings = ["no model client configured"]
        return verification
    user = (
        "Proposal under review (JSON):\n" + json.dumps(proposal.to_dict(), indent=2)
        + "\n\nRepository context:\n" + context_text
    )
    try:
        reply = client.complete(role="verifier", model_id=cfg.verifier_model_id, system=VERIFIER_SYSTEM, user=user, max_tokens=12000)
        record_usage("verifier", reply)
        data = parse_json_block(reply.text)
        verdict = str(data.get("verdict", "")).upper()
        if verdict not in {"PASS", "FAIL", "NEEDS_WORK"}:
            verdict = "ERROR"
        verification.llm_verdict = verdict
        verification.llm_findings = [str(f) for f in data.get("findings", [])][:20]
    except ModelError as exc:
        verification.llm_verdict = "ERROR"
        verification.llm_findings = [str(exc)]
    return verification


def load_context(paths: list[Path], max_bytes: int = 400_000) -> str:
    chunks, total = [], 0
    for p in paths:
        if not p.is_file():
            continue
        text = p.read_text(errors="replace")
        total += len(text)
        if total > max_bytes:
            break
        chunks.append(f"--- BEGIN {p.name} ---\n{text}\n--- END {p.name} ---")
    return "\n".join(chunks)
