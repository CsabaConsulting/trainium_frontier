"""Proposal and RunManifest schemas plus every launch-shape rule.

A *Proposal* is what the Researcher model returns (validated JSON). A *RunManifest* is the frozen
argv/env/box/label record the Operator will execute verbatim. Models never produce a manifest; the
controller derives it from a verified proposal and pinned constants.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from dataclasses import asdict, dataclass, field
from typing import Iterable

from . import config as C

PROPOSAL_ID_RE = re.compile(r"^\d{8}-[a-z0-9][a-z0-9-]{3,60}$")
LABEL_RE = re.compile(r"^al-\d{8}-[a-z0-9][a-z0-9-]{3,80}-(smoke|full)-r\d+$")
FLAG_RE = re.compile(r"^--[a-z][a-z0-9-]*$")
VALUE_RE = re.compile(r"^[A-Za-z0-9_.,:=+-]{1,64}$")

TIERS = ("smoke", "full")  # controls run the same two tiers with an empty flag list
MECHANISM_CLASSES = (
    "throughput",
    "per_step_quality",
    "schedule",
    "capacity",
    "data_regime",
)


class ManifestError(ValueError):
    """Any rule violation. The loop treats these as a stopped branch, never as retry-able."""


@dataclass
class Proposal:
    proposal_id: str
    hypothesis: str
    single_causal_change: str
    flags: list[str]  # train.py argv tokens, e.g. ["--weight-ema", "--weight-ema-decay", "0.995"]
    evidence: list[str]
    duplicate_search_terms: list[str]
    expected_effect: str
    abort_conditions: list[str] = field(default_factory=list)
    status: str = "proposed"
    source_model_id: str = ""
    created_at: str = ""
    change_type: str = "flags"  # flags | code
    implementation_plan: str = ""
    files_allowed: list[str] = field(default_factory=list)
    smoke_gate: dict[str, float | int] = field(default_factory=dict)
    # Research-design metadata.  Defaults keep pre-v2 proposals resumable after a controller
    # upgrade; fresh model output is required to populate them by ``from_model_json``.
    mechanism_class: str = "unclassified"
    predicted_bpb_delta: float | None = None  # candidate - control; negative is better
    predicted_bpb_interval: list[float] = field(default_factory=list)
    candidate_shortlist: list[dict] = field(default_factory=list)
    selection_rationale: str = ""

    @classmethod
    def from_model_json(
        cls,
        raw: dict,
        source_model_id: str,
        created_at: str,
        *,
        require_research_metadata: bool = False,
    ) -> "Proposal":
        required = (
            "proposal_id", "hypothesis", "single_causal_change", "flags", "evidence",
            "duplicate_search_terms", "expected_effect",
        )
        missing = [k for k in required if k not in raw]
        if missing:
            raise ManifestError(f"proposal missing fields: {missing}")
        flags = raw["flags"]
        if not isinstance(flags, list) or not all(isinstance(t, str) for t in flags):
            raise ManifestError("proposal.flags must be a list of strings")
        prop = cls(
            proposal_id=str(raw["proposal_id"]).strip(),
            hypothesis=str(raw["hypothesis"]).strip(),
            single_causal_change=str(raw["single_causal_change"]).strip(),
            flags=[t.strip() for t in flags],
            evidence=[str(e) for e in raw["evidence"]],
            duplicate_search_terms=[str(e) for e in raw["duplicate_search_terms"]],
            expected_effect=str(raw["expected_effect"]).strip(),
            abort_conditions=[str(e) for e in raw.get("abort_conditions", [])],
            source_model_id=source_model_id,
            created_at=created_at,
            change_type=str(raw.get("change_type", "flags")).strip().lower(),
            implementation_plan=str(raw.get("implementation_plan", "")).strip(),
            files_allowed=[str(p) for p in raw.get("files_allowed", [])],
            smoke_gate=raw.get("smoke_gate", {}),
            mechanism_class=str(raw.get("mechanism_class", "unclassified")).strip(),
            predicted_bpb_delta=raw.get("predicted_bpb_delta"),
            predicted_bpb_interval=raw.get("predicted_bpb_interval", []),
            candidate_shortlist=raw.get("candidate_shortlist", []),
            selection_rationale=str(raw.get("selection_rationale", "")).strip(),
        )
        prop.validate_shape(require_research_metadata=require_research_metadata)
        return prop

    def validate_shape(self, *, require_research_metadata: bool = False) -> None:
        if not PROPOSAL_ID_RE.match(self.proposal_id):
            raise ManifestError(f"bad proposal_id {self.proposal_id!r}")
        if len(self.hypothesis) < 20 or len(self.single_causal_change) < 10:
            raise ManifestError("hypothesis/single_causal_change too short to be falsifiable")
        if not self.evidence:
            raise ManifestError("proposal cites no evidence")
        if self.change_type not in {"flags", "code"}:
            raise ManifestError("change_type must be flags or code")
        if self.mechanism_class != "unclassified" and self.mechanism_class not in MECHANISM_CLASSES:
            raise ManifestError(
                f"mechanism_class must be one of {MECHANISM_CLASSES}, got {self.mechanism_class!r}"
            )
        if self.predicted_bpb_delta is not None and (
            isinstance(self.predicted_bpb_delta, bool)
            or not isinstance(self.predicted_bpb_delta, (int, float))
            or not math.isfinite(float(self.predicted_bpb_delta))
        ):
            raise ManifestError("predicted_bpb_delta must be a finite number")
        if not isinstance(self.predicted_bpb_interval, list) or len(self.predicted_bpb_interval) not in {0, 2}:
            raise ManifestError("predicted_bpb_interval must be [] or [low, high]")
        if self.predicted_bpb_interval:
            low, high = self.predicted_bpb_interval
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v))
                for v in (low, high)
            ):
                raise ManifestError("predicted_bpb_interval values must be finite numbers")
            if float(low) > float(high):
                raise ManifestError("predicted_bpb_interval low must not exceed high")
            if self.predicted_bpb_delta is not None and not (
                float(low) <= float(self.predicted_bpb_delta) <= float(high)
            ):
                raise ManifestError("predicted_bpb_delta must lie inside predicted_bpb_interval")
        if not isinstance(self.candidate_shortlist, list):
            raise ManifestError("candidate_shortlist must be a list")
        for i, item in enumerate(self.candidate_shortlist):
            if not isinstance(item, dict):
                raise ManifestError(f"candidate_shortlist[{i}] must be an object")
            required_item = {"idea", "mechanism_class", "predicted_bpb_delta", "why_ranked"}
            missing_item = required_item - set(item)
            if missing_item:
                raise ManifestError(
                    f"candidate_shortlist[{i}] missing fields: {sorted(missing_item)}"
                )
            if str(item["mechanism_class"]) not in MECHANISM_CLASSES:
                raise ManifestError(
                    f"candidate_shortlist[{i}].mechanism_class must be one of {MECHANISM_CLASSES}"
                )
            delta = item["predicted_bpb_delta"]
            if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
                raise ManifestError(
                    f"candidate_shortlist[{i}].predicted_bpb_delta must be finite"
                )
        if require_research_metadata:
            if self.mechanism_class not in MECHANISM_CLASSES:
                raise ManifestError("fresh proposal requires a classified mechanism_class")
            if self.predicted_bpb_delta is None or len(self.predicted_bpb_interval) != 2:
                raise ManifestError("fresh proposal requires a BPB point prediction and interval")
            if not 4 <= len(self.candidate_shortlist) <= 6:
                raise ManifestError("fresh proposal requires a shortlist of 4 to 6 candidates")
            class_counts: dict[str, int] = {}
            for item in self.candidate_shortlist:
                cls_name = str(item["mechanism_class"])
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            if any(count * 2 > len(self.candidate_shortlist) for count in class_counts.values()):
                raise ManifestError("one mechanism class may not occupy more than half the shortlist")
            if not any(
                item["mechanism_class"] == self.mechanism_class
                and float(item["predicted_bpb_delta"]) == float(self.predicted_bpb_delta)
                for item in self.candidate_shortlist
            ):
                raise ManifestError("selected prediction must appear in candidate_shortlist")
            if len(self.selection_rationale) < 20:
                raise ManifestError("fresh proposal requires a concrete selection_rationale")
        if not isinstance(self.smoke_gate, dict):
            raise ManifestError("proposal.smoke_gate must be an object")
        allowed_gate_keys = {"min_candidate_steps", "min_step_rate_ratio", "max_bpb_delta"}
        unknown_gate_keys = set(self.smoke_gate) - allowed_gate_keys
        if unknown_gate_keys:
            raise ManifestError(f"proposal.smoke_gate has unknown fields: {sorted(unknown_gate_keys)}")
        min_steps = self.smoke_gate.get("min_candidate_steps")
        if min_steps is not None and (isinstance(min_steps, bool) or not isinstance(min_steps, int) or min_steps < 1):
            raise ManifestError("smoke_gate.min_candidate_steps must be a positive integer")
        for key in ("min_step_rate_ratio", "max_bpb_delta"):
            value = self.smoke_gate.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ManifestError(f"smoke_gate.{key} must be a finite number")
        ratio = self.smoke_gate.get("min_step_rate_ratio")
        if ratio is not None and float(ratio) <= 0:
            raise ManifestError("smoke_gate.min_step_rate_ratio must be positive")
        if self.change_type == "flags":
            if not self.flags:
                raise ManifestError("flag-only proposal has no flags (that is the control)")
            if self.files_allowed:
                raise ManifestError("flag-only proposal may not name editable files")
            validate_flag_tokens(self.flags)
        else:
            if self.flags:
                raise ManifestError("code proposal must encode its single change in code, not flags")
            if self.files_allowed != ["train.py"]:
                raise ManifestError("code proposal files_allowed must be exactly ['train.py']")
            if len(self.implementation_plan) < 30:
                raise ManifestError("code proposal implementation_plan is too short")

    def flag_names(self) -> list[str]:
        return sorted({t for t in self.flags if t.startswith("--")})

    def canonical_flags(self) -> str:
        return " ".join(self.flags)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_flag_tokens(tokens: Iterable[str], known_flags: set[str] | None = None) -> None:
    tokens = list(tokens)
    if not tokens:
        return
    if not tokens[0].startswith("--"):
        raise ManifestError(f"flag list must start with a --flag, got {tokens[0]!r}")
    prev_flag = None
    seen: set[str] = set()
    for tok in tokens:
        if tok.startswith("--"):
            if not FLAG_RE.match(tok):
                raise ManifestError(f"illegal flag token {tok!r}")
            if tok in C.FORBIDDEN_TRAIN_FLAGS:
                raise ManifestError(f"flag {tok} is protocol-owned and may not be proposed")
            base = tok[5:] if tok.startswith("--no-") else tok[2:]
            if base in seen:
                raise ManifestError(f"flag {tok} set twice (or with its --no- form)")
            seen.add(base)
            if known_flags is not None and tok not in known_flags:
                raise ManifestError(f"flag {tok} does not exist in train.py at the pinned commit")
            prev_flag = tok
        else:
            if prev_flag is None:
                raise ManifestError(f"value {tok!r} without a preceding flag")
            if not VALUE_RE.match(tok):
                raise ManifestError(f"illegal flag value {tok!r} (shell-unsafe or too long)")
            prev_flag = None


def known_train_flags(train_py_text: str) -> set[str]:
    """Extract every --flag name from train.py's argparse block, including --no-* forms."""
    flags: set[str] = set()
    for m in re.finditer(r'add_argument\(\s*"(--[a-z][a-z0-9-]*)"', train_py_text):
        flags.add(m.group(1))
    # BooleanOptionalAction generates --no-<flag>; be permissive on detection, strict on use.
    for m in re.finditer(
        r'add_argument\(\s*"(--[a-z][a-z0-9-]*)"[^)]*?BooleanOptionalAction', train_py_text, re.S
    ):
        flags.add("--no-" + m.group(1)[2:])
    # Flags registered through a helper (train.py:5170 pattern: parser.add_argument(flag, ...))
    for m in re.finditer(r'\(\s*"(--[a-z][a-z0-9-]*)"\s*,\s*"NEURON_COMPETITION_R1_', train_py_text):
        flags.add(m.group(1))
    return flags


@dataclass
class RunManifest:
    manifest_id: str
    proposal_id: str
    tier: str  # control | smoke | full
    box: str
    host: str
    remote_run_dir: str
    commit: str
    base_commit: str
    label: str
    description: str
    lnc: int
    nproc_per_node: int
    num_steps: int
    max_train_seconds: int
    seed: int
    train_flags: list[str]
    argv: list[str]  # exact remote argv (no shell)
    env: dict[str, str]
    remote_log: str
    remote_ledger: str
    remote_checkpoint: str
    abort_conditions: list[str]
    control_label: str | None
    frozen_at: str
    frozen_by: str = "autoloop-controller"
    launch_command_string: str = ""  # informational; never executed
    execution_mode: str = "standard"  # standard | exact-dense1024-cold-eval
    comparison_role: str = "candidate"  # control | candidate
    expected_train_sha256: str = ""
    exact_train_argv: list[str] = field(default_factory=list)
    selection_metric: str = "public_val_bpb"
    eval_context: int | None = None
    eval_tokens: int | None = None
    eval_bytes: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "RunManifest":
        data = dict(raw)
        data.setdefault("base_commit", C.AUTHORITATIVE_COMMIT)
        return cls(**data)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def build_manifest(
    *,
    proposal_id: str,
    tier: str,
    box: str,
    remote_run_dir: str,
    label: str,
    train_flags: list[str],
    cfg: C.LoopConfig,
    frozen_at: str,
    control_label: str | None,
    abort_conditions: list[str],
    description_note: str,
    candidate_commit: str | None = None,
    execution_mode: str = "standard",
    comparison_role: str | None = None,
    expected_train_sha256: str = "",
) -> RunManifest:
    if tier not in TIERS:
        raise ManifestError(f"unknown tier {tier!r}")
    if not LABEL_RE.match(label):
        raise ManifestError(f"label {label!r} violates the autoloop label grammar")
    if box not in C.PINNED_HOSTS:
        raise ManifestError(f"box {box!r} is not pinned")
    if not remote_run_dir.startswith(C.REMOTE_RUNS_DIR + "/"):
        raise ManifestError("remote run dir must live under the autoloop runs base")
    for tok in C.REMOTE_FORBIDDEN_PATH_TOKENS:
        if tok in remote_run_dir:
            raise ManifestError(f"remote run dir references forbidden path token {tok!r}")
    shape = (cfg.lnc, cfg.nproc_per_node)
    if shape not in C.VALID_SHAPES:
        raise ManifestError(f"shape {shape} is not full-chip")
    if C.REQUIRED_CC_FLAG not in cfg.neuron_cc_flags.split():
        raise ManifestError("NEURON_CC_FLAGS lacks --optlevel=1")
    comparison_role = comparison_role or ("control" if proposal_id.startswith("control-") else "candidate")
    if comparison_role not in {"control", "candidate"}:
        raise ManifestError("comparison_role must be control or candidate")
    is_control = comparison_role == "control"
    if is_control and train_flags:
        raise ManifestError("control run must have no flags")
    commit = candidate_commit or C.AUTHORITATIVE_COMMIT
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ManifestError("candidate commit must be a full lowercase git SHA")
    if execution_mode == "standard" and not is_control and not train_flags and commit == C.AUTHORITATIVE_COMMIT:
        raise ManifestError("candidate must have flags or a verified code commit")
    validate_flag_tokens(train_flags)

    if execution_mode not in {"standard", "exact-dense1024-cold-eval"}:
        raise ManifestError(f"unknown execution_mode {execution_mode!r}")
    if execution_mode == "exact-dense1024-cold-eval" and tier != "full":
        raise ManifestError("exact dense1024 queue is full-only")
    if tier == "smoke":
        num_steps, max_train = cfg.caps.smoke_num_steps, cfg.caps.smoke_max_train_seconds
    else:
        num_steps, max_train = cfg.caps.full_num_steps, C.PHASE1_BUDGET_SECONDS
    if max_train > C.PHASE1_BUDGET_SECONDS:
        raise ManifestError("max_train_seconds exceeds the Phase 1 budget")

    description_note = re.sub(r"[^A-Za-z0-9 _.,:=+/()'-]", "", description_note)[:160]
    description = (
        f"autoloop {proposal_id} {tier}: {description_note} flags='{' '.join(train_flags) or 'none'}' "
        f"[{box}] CC='{cfg.neuron_cc_flags}' seed={C.SEED} commit={commit[:8]}"
    )
    if execution_mode == "exact-dense1024-cold-eval":
        if train_flags:
            raise ManifestError("exact dense1024 mode encodes variants in one-constant commits, not flags")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_train_sha256):
            raise ManifestError("exact dense1024 mode requires an expected train.py SHA-256")
        num_steps = C.EXACT_DENSE1024_NUM_STEPS
        argv = [
            C.REMOTE_PYTHON, "research/run_experiment.py",
            "--label", label,
            "--solution", f"autoloop:{proposal_id}",
            "--description", description,
            "--nproc-per-node", str(cfg.nproc_per_node),
            "--lnc", str(cfg.lnc),
            "--no-commit",
            "--exact-dense1024",
            "--expected-train-sha256", expected_train_sha256,
        ]
    else:
        argv = [
            C.REMOTE_PYTHON,
            "research/run_experiment.py",
            "--label", label,
            "--solution", f"autoloop:{proposal_id}",
            "--description", description,
            "--num-steps", str(num_steps),
            "--max-train-seconds", str(max_train),
            "--nproc-per-node", str(cfg.nproc_per_node),
            "--lnc", str(cfg.lnc),
            "--no-commit",
        ]
        if train_flags:
            argv += ["--extra-args", *train_flags, "--seed", str(C.SEED)]
        else:
            argv += ["--extra-args", "--seed", str(C.SEED)]
    env = {
        # Non-interactive SSH sessions do not source the venv activation script.  The
        # repository's run_experiment.py launches `torchrun` by name, so make the pinned
        # native venv explicit instead of depending on a login-shell PATH.
        "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "NEURON_CC_FLAGS": cfg.neuron_cc_flags,
        "NEURON_LOGICAL_NC_CONFIG": str(cfg.lnc),
        "NEURON_COMPETITION_R1_SEED": str(C.SEED),
        "PYTHONUNBUFFERED": "1",
    }
    if execution_mode == "exact-dense1024-cold-eval":
        env["NEURON_CC_FLAGS"] = C.EXACT_DENSE1024_CC_FLAGS
        cache_root = f"{C.REMOTE_AUTOLOOP_DIR}/caches/{label}"
        for key in C.CACHE_ENV_KEYS:
            suffix = key.lower()
            env[key] = f"{cache_root}/train/{suffix}"
            env[f"AUTOLOOP_EVAL_{key}"] = f"{cache_root}/eval/{suffix}"
    manifest = RunManifest(
        manifest_id=f"{proposal_id}--{tier}--{label}",
        proposal_id=proposal_id,
        tier=tier,
        box=box,
        host=C.PINNED_HOSTS[box],
        remote_run_dir=remote_run_dir,
        commit=commit,
        base_commit=C.AUTHORITATIVE_COMMIT,
        label=label,
        description=description,
        lnc=cfg.lnc,
        nproc_per_node=cfg.nproc_per_node,
        num_steps=num_steps,
        max_train_seconds=max_train,
        seed=C.SEED,
        train_flags=list(train_flags),
        argv=argv,
        env=env,
        remote_log=f"{remote_run_dir}/research/run-logs/{label}.log",
        remote_ledger=f"{remote_run_dir}/research/experiment-log.jsonl",
        remote_checkpoint=(
            f"{remote_run_dir}/out/final.pt"
            if execution_mode == "exact-dense1024-cold-eval"
            else f"{remote_run_dir}/out/{label}/final.pt"
        ),
        abort_conditions=list(abort_conditions),
        control_label=control_label,
        frozen_at=frozen_at,
        launch_command_string=f"NEURON_LOGICAL_NC_CONFIG={cfg.lnc} " + shlex.join(argv),
        execution_mode=execution_mode,
        comparison_role=comparison_role,
        expected_train_sha256=expected_train_sha256,
        exact_train_argv=(list(C.EXACT_DENSE1024_TRAIN_ARGV)
                          if execution_mode == "exact-dense1024-cold-eval" else []),
        selection_metric="public_val_bpb",
        eval_context=(C.EXACT_DENSE1024_EVAL_CONTEXT
                      if execution_mode == "exact-dense1024-cold-eval" else None),
        eval_tokens=(C.EXACT_DENSE1024_EVAL_TOKENS
                     if execution_mode == "exact-dense1024-cold-eval" else None),
        eval_bytes=(C.EXACT_DENSE1024_EVAL_BYTES
                    if execution_mode == "exact-dense1024-cold-eval" else None),
    )
    assert_manifest_legal(manifest)
    return manifest


def assert_manifest_legal(m: RunManifest) -> None:
    """Re-run every invariant on a loaded manifest (used right before launch and after restart)."""
    if m.base_commit != C.AUTHORITATIVE_COMMIT:
        raise ManifestError("manifest base commit is not authoritative")
    if not re.fullmatch(r"[0-9a-f]{40}", m.commit):
        raise ManifestError("manifest commit is not a full lowercase git SHA")
    if m.comparison_role not in {"control", "candidate"}:
        raise ManifestError("manifest comparison role is invalid")
    is_control = m.comparison_role == "control"
    if is_control and m.commit != C.AUTHORITATIVE_COMMIT:
        raise ManifestError("control manifest must use the authoritative commit")
    if m.train_flags and m.commit != C.AUTHORITATIVE_COMMIT:
        raise ManifestError("flag candidate must use the authoritative commit")
    if m.execution_mode == "standard" and not is_control and not m.train_flags and m.commit == C.AUTHORITATIVE_COMMIT:
        raise ManifestError("code candidate must use an independently verified candidate commit")
    if m.seed != C.SEED:
        raise ManifestError("seed must be 42 in manifest")
    if (m.lnc, m.nproc_per_node) not in C.VALID_SHAPES:
        raise ManifestError("manifest shape is not full-chip")
    if "--no-commit" not in m.argv:
        raise ManifestError("--no-commit missing")
    if m.argv[0] != C.REMOTE_PYTHON or m.argv[1] != "research/run_experiment.py":
        raise ManifestError("argv does not start with the pinned runner invocation")
    if m.env.get("NEURON_LOGICAL_NC_CONFIG") != str(m.lnc):
        raise ManifestError("env LNC disagrees with manifest lnc")
    if C.REQUIRED_CC_FLAG not in m.env.get("NEURON_CC_FLAGS", "").split():
        raise ManifestError("env NEURON_CC_FLAGS lacks --optlevel=1")
    if m.max_train_seconds > C.PHASE1_BUDGET_SECONDS:
        raise ManifestError("over budget")
    if m.host != C.PINNED_HOSTS.get(m.box):
        raise ManifestError("host/box mismatch")
    if not LABEL_RE.match(m.label):
        raise ManifestError("label grammar")
    if not m.remote_run_dir.startswith(C.REMOTE_RUNS_DIR + "/"):
        raise ManifestError("run dir outside autoloop base")
    for tok in C.REMOTE_FORBIDDEN_PATH_TOKENS:
        if any(tok in a for a in m.argv) or tok in m.remote_run_dir:
            raise ManifestError(f"forbidden path token {tok!r} in manifest")
    for a in m.argv:
        if any(ch in a for ch in "|;&`$<>\n"):
            raise ManifestError(f"shell metacharacter in argv element {a!r}")
    if m.execution_mode == "exact-dense1024-cold-eval":
        expected_runner_tail = [
            "--no-commit", "--exact-dense1024", "--expected-train-sha256",
            m.expected_train_sha256,
        ]
        if m.argv[-4:] != expected_runner_tail:
            raise ManifestError("exact runner argv tail differs from the pinned schema")
        if m.exact_train_argv != list(C.EXACT_DENSE1024_TRAIN_ARGV):
            raise ManifestError("exact train argv differs from the human-authorized command")
        if m.train_flags or not re.fullmatch(r"[0-9a-f]{64}", m.expected_train_sha256):
            raise ManifestError("exact mode train hash/flags invalid")
        if (m.eval_context != C.EXACT_DENSE1024_EVAL_CONTEXT
                or m.eval_tokens != C.EXACT_DENSE1024_EVAL_TOKENS
                or m.eval_bytes != C.EXACT_DENSE1024_EVAL_BYTES):
            raise ManifestError("exact mode cold-eval contract changed")
        if m.remote_checkpoint != f"{m.remote_run_dir}/out/final.pt":
            raise ManifestError("exact mode checkpoint is not the submission default out/final.pt")
        if m.env.get("NEURON_CC_FLAGS") != C.EXACT_DENSE1024_CC_FLAGS:
            raise ManifestError("exact mode compiler flags differ from the package effective default")
        cache_values: list[str] = []
        expected_root = f"{C.REMOTE_AUTOLOOP_DIR}/caches/{m.label}"
        for key in C.CACHE_ENV_KEYS:
            train_path = m.env.get(key, "")
            eval_path = m.env.get(f"AUTOLOOP_EVAL_{key}", "")
            if not train_path.startswith(expected_root + "/train/"):
                raise ManifestError(f"exact train cache path for {key} is not unique to the label")
            if not eval_path.startswith(expected_root + "/eval/"):
                raise ManifestError(f"exact eval cache path for {key} is not unique to the label")
            cache_values.extend((train_path, eval_path))
        if len(cache_values) != len(set(cache_values)):
            raise ManifestError("exact train/eval cache paths are not pairwise unique")
    else:
        if "--seed" not in m.argv or m.argv[m.argv.index("--seed") + 1] != str(C.SEED):
            raise ManifestError("seed must be 42 in standard runner argv")
        extra = m.argv.index("--extra-args") + 1 if "--extra-args" in m.argv else None
        if extra is None:
            raise ManifestError("argv lacks --extra-args block")
        flags_in_argv = m.argv[extra:]
        if flags_in_argv[:-2] != m.train_flags or flags_in_argv[-2:] != ["--seed", str(C.SEED)]:
            raise ManifestError("train_flags and argv disagree")
        validate_flag_tokens(m.train_flags)
