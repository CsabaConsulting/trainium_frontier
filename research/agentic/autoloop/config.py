"""Pinned constants and JSON-overridable caps for the autoloop.

Everything a model must never influence lives here: hosts, key path, authoritative commit, legal
launch shapes, forbidden flags, forbidden remote tokens, and hard caps. Overrides are read from
``research/agentic/autoloop.json`` (optional). Pinned values cannot be overridden by that file.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AGENT_DIR = ROOT / "research" / "agentic"
RUNTIME_DIR = AGENT_DIR / "runtime"
EXPERIMENTS_DIR = AGENT_DIR / "experiments"
PROPOSALS_DIR = AGENT_DIR / "proposals"
STOP_FILE = AGENT_DIR / "STOP"
STATE_FILE = RUNTIME_DIR / "autoloop_state.json"
LOCAL_LOCK_DIR = RUNTIME_DIR / "locks"
BUNDLE_DIR = RUNTIME_DIR / "bundles"
CONFIG_FILE = AGENT_DIR / "autoloop.json"
QUEUE_PLAN_FILE = AGENT_DIR / "frontier-full-only-queue.json"
DECISION_LOG = AGENT_DIR / "DECISION_LOG.md"
LOCAL_LEDGER = ROOT / "research" / "experiment-log.jsonl"
UTILIZATION_JOURNAL = RUNTIME_DIR / "idle-utilization.jsonl"
FINAL_WEEK_RESEARCH_ADDENDUM = AGENT_DIR / "FINAL_WEEK_RESEARCH_ADDENDUM_2026-09-24.md"

# ---- pinned (not overridable) -------------------------------------------------------------------

# Filled by the reviewed frontier-control preparation commit. The controller additionally checks
# the exact train.py byte hash before staging and again on the host.
AUTHORITATIVE_COMMIT = "4ed89f36c437ab75b0bf63cea80a3871f547de34"
AUTHORITATIVE_TRAIN_SHA256 = "9173f62617789b058083c3a4722a906e2b165f4986716281f1915db3fb9fa344"
AUTHORITATIVE_RUNNER_SHA256 = "88243a74f0f080929f93d0b88ddedbe6f5351ad37bf272229d9ae6d26efd5d2d"
AUTHORITATIVE_PREPARE_SHA256 = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
BREAKER_CLEANUP_RECEIPT_COMMIT = "563f83ec3f5b68c8bc609d11956bdd2072720aee"
BREAKER_CLEANUP_RECEIPT_PATH = "research/agentic/operations/TRN21_DETERMINISTIC_CLEANUP_2026-09-23_R8.md"
BREAKER_CLEANUP_RECEIPT_SHA256 = "ac801b82ffd711a9efa269e824cdc53c8a4e0fe2c1037dd3b3ecf1978f5df5e1"
EXACT_DENSE1024_TRAIN_ARGV: tuple[str, ...] = (
    "torchrun", "--standalone", "--nproc_per_node=4", "train.py",
    "--n-embd", "1024", "--no-ngram-ve", "--no-qk-shift",
    "--compile-sdpa-direct", "--nki-local-conv", "--bf16-norm-output",
    "--pack-factor", "32", "--clip-after-reduce", "--seq-len", "1024",
    "--no-eval-public",
)
EXACT_DENSE1024_NUM_STEPS = 1_000_000
EXACT_DENSE1024_EVAL_CONTEXT = 1024
EXACT_DENSE1024_EVAL_TOKENS = 20_971_520
EXACT_DENSE1024_EVAL_BYTES = 83_462_186
EXACT_DENSE1024_CC_FLAGS = "--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
CACHE_ENV_KEYS: tuple[str, ...] = (
    "TORCH_NEURONX_NEFF_CACHE_DIR",
    "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
    "TORCH_NEURONX_HLO_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "NEURON_COMPILE_CACHE_DIR",
)

# Order given by the human operator on 2026-09-22. The trn21/trn22 labels are the protocol's box
# ids; the mapping to these two DNS names is an assumption recorded in AUTOLOOP.md and re-checked
# by `inventory` (remote hostname is written into every manifest).
PINNED_HOSTS: dict[str, str] = {
    "trn21": "ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com",
    "trn22": "ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com",
}
SSH_KEY_PATH = ROOT / "trn2.pem"
REMOTE_PYTHON = "/home/ubuntu/workspace/native_venv/bin/python"
REMOTE_HOME = "/home/ubuntu"
REMOTE_AUTOLOOP_DIR = f"{REMOTE_HOME}/autoloop"  # controller scratch: bundles, launchers, locks
REMOTE_RUNS_DIR = f"{REMOTE_HOME}/workspace/autoloop_runs"  # NEW immutable clean run dirs only
REMOTE_DATA_CACHE = f"{REMOTE_HOME}/.cache/neuron_competition_r1"

# Directories that must never appear in any remote argv (dirty, evidence-bearing, or foreign).
REMOTE_FORBIDDEN_PATH_TOKENS = (
    "wave195",
    "trainium_freshstart_20260918",
    "trainium_frontier",
)
# Any remote argv containing one of these tokens is rejected before it reaches ssh.
REMOTE_FORBIDDEN_COMMAND_TOKENS = (
    "aws",
    "ec2",
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
    "submit",
    "leaderboard",
    "rm",
    "rmdir",
    "unlink",
    "mv",
    "kill",
    "pkill",
    "killall",
    "sudo",
    "su",
    "curl",
    "wget",
    "ssh",
    "scp",
    "rsync",
    "bash",
    "sh",
    "zsh",
    "eval",
    "exec",
    "python",  # remote python is allowed only as the pinned REMOTE_PYTHON absolute path
    "|",
    ";",
    "&&",
    "||",
    ">",
    "<",
    "`",
    "$(",
)
# git subcommands the operator may run remotely (only ever with -C <fresh run dir>).
REMOTE_GIT_ALLOWED_SUBCOMMANDS = ("clone", "checkout", "rev-parse", "status", "log", "show")
# git subcommands that are never allowed anywhere.
REMOTE_GIT_FORBIDDEN_SUBCOMMANDS = (
    "push", "clean", "reset", "rebase", "commit", "merge", "branch", "worktree", "stash", "gc",
    "prune", "filter-branch", "update-ref", "reflog", "rm", "add", "am", "apply", "cherry-pick",
)

VALID_SHAPES: tuple[tuple[int, int], ...] = ((2, 4), (1, 8))  # (lnc, nproc_per_node) full chip
SEED = 42
PHASE1_BUDGET_SECONDS = 1800
REQUIRED_CC_FLAG = "--optlevel=1"
TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS = 900

# Final-week public-development selection populations.  These constants describe the policy but do
# not rebase the active controller; ``EvaluationPolicy.enabled`` remains false until a separately
# reviewed canonical-root change supplies the exact current base hash.
SCREEN_EVAL_TOKENS = 2_097_152
CONFIRMATION_EVAL_TOKENS = 20_971_520
SCREEN_EVAL_BYTES = 8_418_678
CONFIRMATION_EVAL_BYTES = 83_462_186
EVAL_CONTEXT = 1024
SCREEN_NOMINATE_DELTA = "-0.0005"
SCREEN_DISCARD_DELTA = "0.0005"
ADOPTION_MEAN_DELTA = "-0.0008"
ADOPTION_REQUIRED_FULL20M_RUNS = 2

# train.py flags a flag-only proposal may never set (they define the protocol, not the hypothesis).
FORBIDDEN_TRAIN_FLAGS = frozenset(
    {
        "--out-dir",
        "--seed",
        "--init-from",
        "--num-steps",
        "--max-train-seconds",
        "--eval-tokens",
        "--eval-public",
        "--no-eval-public",
        "--compile-only",
        "--device-type",
    }
)

COMPETITION_CUTOFF = datetime(2026, 9, 30, 23, 59, tzinfo=timezone(timedelta(hours=-7)))  # PT


# ---- overridable caps ---------------------------------------------------------------------------


@dataclass
class Caps:
    max_llm_calls_per_day: int = 60
    max_llm_calls_per_cycle: int = 4
    max_llm_usd_per_day: float = 10.0
    max_runs_per_day: int = 4
    max_runs_total: int = 12
    max_consecutive_infra_failures_per_box: int = 2
    min_free_gib: float = 8.0  # protocol says ~5G; the 2026-09-21 inventory adopted 8 GiB
    poll_interval_seconds: int = 120
    launch_grace_seconds: int = 900  # compile + startup before "no progress" counts as a failure
    run_timeout_seconds: int = PHASE1_BUDGET_SECONDS + 1500  # training + eval + compile + margin
    smoke_max_train_seconds: int = 300
    smoke_num_steps: int = 5000
    full_num_steps: int = 5000
    smoke_step_rate_tolerance: float = 0.02  # smoke must be >= control * (1 - tol) steps/s
    keep_bpb_delta: float = -0.002  # candidate must beat control by at least this much
    steps_comparability_frac: float = 0.10  # protocol §5
    usd_per_1k_input_tokens: float = 0.005
    usd_per_1k_output_tokens: float = 0.025


@dataclass
class EvaluationPolicy:
    """Fail-closed two-population selection policy for the next canonical root.

    Decimal thresholds are stored as strings so boundary decisions never depend on binary-float
    rounding.  The active fixed queue is intentionally not changed by this schema.
    """

    schema_version: int = 1
    enabled: bool = False
    current_base_sha256: str = ""
    current_control_labels_by_host: dict[str, str] = field(default_factory=dict)
    active_sweep_ids: list[str] = field(default_factory=list)
    context_length: int = EVAL_CONTEXT
    screen_eval_tokens: int = SCREEN_EVAL_TOKENS
    confirmation_eval_tokens: int = CONFIRMATION_EVAL_TOKENS
    screen_eval_bytes: int = SCREEN_EVAL_BYTES
    confirmation_eval_bytes: int = CONFIRMATION_EVAL_BYTES
    screen_nominate_delta: str = SCREEN_NOMINATE_DELTA
    screen_discard_delta: str = SCREEN_DISCARD_DELTA
    middle_band_requires_active_sweep: bool = True
    adoption_required_full20m_runs: int = ADOPTION_REQUIRED_FULL20M_RUNS
    adoption_mean_delta: str = ADOPTION_MEAN_DELTA
    adoption_requires_each_negative: bool = True
    adoption_requires_same_host_current_control: bool = True

    def validate(self) -> list[str]:
        expected = {
            "schema_version": 1,
            "context_length": EVAL_CONTEXT,
            "screen_eval_tokens": SCREEN_EVAL_TOKENS,
            "confirmation_eval_tokens": CONFIRMATION_EVAL_TOKENS,
            "screen_eval_bytes": SCREEN_EVAL_BYTES,
            "confirmation_eval_bytes": CONFIRMATION_EVAL_BYTES,
            "screen_nominate_delta": SCREEN_NOMINATE_DELTA,
            "screen_discard_delta": SCREEN_DISCARD_DELTA,
            "middle_band_requires_active_sweep": True,
            "adoption_required_full20m_runs": ADOPTION_REQUIRED_FULL20M_RUNS,
            "adoption_mean_delta": ADOPTION_MEAN_DELTA,
            "adoption_requires_each_negative": True,
            "adoption_requires_same_host_current_control": True,
        }
        errors = [
            f"evaluation_policy.{key} must remain {value!r}"
            for key, value in expected.items()
            if getattr(self, key) != value
        ]
        if self.enabled and not re.fullmatch(r"[0-9a-f]{64}", self.current_base_sha256):
            errors.append("enabled evaluation_policy requires a full lowercase current_base_sha256")
        if self.enabled and not self.current_control_labels_by_host:
            errors.append("enabled evaluation_policy requires authorized current control labels")
        for host, label in self.current_control_labels_by_host.items():
            if host not in PINNED_HOSTS or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
                errors.append("evaluation_policy current control label mapping is invalid")
        if len(self.active_sweep_ids) != len(set(self.active_sweep_ids)) or any(
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
                for value in self.active_sweep_ids):
            errors.append("evaluation_policy active_sweep_ids must be unique safe identifiers")
        if not self.enabled and self.current_base_sha256:
            errors.append("disabled evaluation_policy must not pre-pin a base hash")
        if not self.enabled and (self.current_control_labels_by_host or self.active_sweep_ids):
            errors.append("disabled evaluation_policy must not authorize controls or sweeps")
        return errors


@dataclass
class LoopConfig:
    lnc: int = 2
    nproc_per_node: int = 4
    neuron_cc_flags: str = "--optlevel=1"
    researcher_model_id: str = ""
    implementer_model_id: str = ""
    verifier_model_id: str = ""
    analyst_model_id: str = ""
    aws_region: str = "us-west-2"
    aws_profile: str = ""
    bedrock_read_timeout_seconds: int = 1800
    git_push: bool = False
    git_remote: str = "origin"
    plan_mode: str = ""
    boxes: list[str] = field(default_factory=lambda: ["trn22", "trn21"])
    caps: Caps = field(default_factory=Caps)
    evaluation_policy: EvaluationPolicy = field(default_factory=EvaluationPolicy)

    # ---- derived / validation ----
    def shape(self) -> tuple[int, int]:
        return (self.lnc, self.nproc_per_node)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.shape() not in VALID_SHAPES:
            errors.append(f"launch shape {self.shape()} is not a full-chip Phase 1 shape {VALID_SHAPES}")
        if REQUIRED_CC_FLAG not in self.neuron_cc_flags.split():
            errors.append(f"NEURON_CC_FLAGS must contain {REQUIRED_CC_FLAG}")
        if not self.researcher_model_id:
            errors.append("researcher_model_id is empty")
        if not self.verifier_model_id:
            errors.append("verifier_model_id is empty")
        if self.researcher_model_id and self.verifier_model_id:
            if model_family(self.researcher_model_id) == model_family(self.verifier_model_id):
                errors.append("verifier must be a different model family from the researcher")
        for box in self.boxes:
            if box not in PINNED_HOSTS:
                errors.append(f"unknown box {box!r}; pinned boxes are {sorted(PINNED_HOSTS)}")
        if self.plan_mode not in {"", "fixed_full_only"}:
            errors.append("plan_mode must be empty or fixed_full_only")
        if self.plan_mode == "fixed_full_only":
            if self.boxes != ["trn22", "trn21"]:
                errors.append("fixed_full_only requires boxes [trn22, trn21] in authorized order")
            if self.neuron_cc_flags != EXACT_DENSE1024_CC_FLAGS:
                errors.append("fixed_full_only requires the exact dense1024 compiler flags")
        if self.caps.min_free_gib < 5.0:
            errors.append("min_free_gib may not be lowered below the protocol's 5 GiB")
        if self.caps.max_runs_per_day < 1 or self.caps.max_runs_total < 1:
            errors.append("run caps must be positive")
        if not 60 <= self.bedrock_read_timeout_seconds <= 3600:
            errors.append("bedrock_read_timeout_seconds must be between 60 and 3600")
        errors.extend(self.evaluation_policy.validate())
        return errors

    def to_dict(self) -> dict:
        return asdict(self)


def model_family(model_id: str) -> str:
    """'us.anthropic.claude-fable-5-1' -> 'anthropic'; 'us.openai.gpt-6-astra' -> 'openai'."""
    parts = model_id.split(".")
    for part in parts:
        if part in {"anthropic", "openai", "amazon", "meta", "mistral", "cohere", "ai21", "deepseek"}:
            return part
    return parts[1] if len(parts) > 1 else model_id


def load_config(path: Path | None = None) -> LoopConfig:
    path = path or CONFIG_FILE
    cfg = LoopConfig()
    if path.is_file():
        raw = json.loads(path.read_text())
        for pinned in ("hosts", "ssh_key_path", "commit", "remote_python", "seed"):
            if pinned in raw:
                raise ValueError(f"{path.name} may not override pinned value {pinned!r}")
        caps_raw = raw.pop("caps", {})
        evaluation_raw = raw.pop("evaluation_policy", {})
        for key, value in raw.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown config key {key!r}")
            setattr(cfg, key, value)
        for key, value in caps_raw.items():
            if not hasattr(cfg.caps, key):
                raise ValueError(f"unknown caps key {key!r}")
            setattr(cfg.caps, key, value)
        for key, value in evaluation_raw.items():
            if not hasattr(cfg.evaluation_policy, key):
                raise ValueError(f"unknown evaluation_policy key {key!r}")
            setattr(cfg.evaluation_policy, key, value)
    return cfg
