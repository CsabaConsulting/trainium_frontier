"""Pinned constants and JSON-overridable caps for the autoloop.

Everything a model must never influence lives here: hosts, key path, authoritative commit, legal
launch shapes, forbidden flags, forbidden remote tokens, and hard caps. Overrides are read from
``research/agentic/autoloop.json`` (optional). Pinned values cannot be overridden by that file.
"""

from __future__ import annotations

import json
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
DECISION_LOG = AGENT_DIR / "DECISION_LOG.md"
LOCAL_LEDGER = ROOT / "research" / "experiment-log.jsonl"

# ---- pinned (not overridable) -------------------------------------------------------------------

AUTHORITATIVE_COMMIT = "9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a"

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
    git_push: bool = False
    git_remote: str = "origin"
    boxes: list[str] = field(default_factory=lambda: ["trn22", "trn21"])
    caps: Caps = field(default_factory=Caps)

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
        if self.caps.min_free_gib < 5.0:
            errors.append("min_free_gib may not be lowered below the protocol's 5 GiB")
        if self.caps.max_runs_per_day < 1 or self.caps.max_runs_total < 1:
            errors.append("run caps must be positive")
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
        for key, value in raw.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown config key {key!r}")
            setattr(cfg, key, value)
        for key, value in caps_raw.items():
            if not hasattr(cfg.caps, key):
                raise ValueError(f"unknown caps key {key!r}")
            setattr(cfg.caps, key, value)
    return cfg
