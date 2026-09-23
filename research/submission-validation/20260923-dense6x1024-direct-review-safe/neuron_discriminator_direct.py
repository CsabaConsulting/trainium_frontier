#!/usr/bin/env python3
"""Fail-closed source-versus-direct Neuron discriminator.

The top-level process authenticates an externally finalized request and then runs
each implementation in a different process and compile cache.  Worker entry
points are private implementation details; they require a parent nonce and are
never an authorization boundary.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import types
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
HARNESS = Path(__file__).resolve()
SOURCE = ROOT / "submissions/dense6x1024_lean_review/train.py"
DIRECT = ROOT / "submissions/dense6x1024_direct_review/train.py"
PREPARE = ROOT / "prepare.py"
PROPOSAL = EVIDENCE / "proposal.json"
REQUEST = EVIDENCE / "neuron-discriminator-authorization-request.json"
RUNBOOK = EVIDENCE / "neuron-discriminator-runbook.md"
SOURCE_SHA256 = "9173f62617789b058083c3a4722a906e2b165f4986716281f1915db3fb9fa344"
DIRECT_SHA256 = "c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537"
PREPARE_SHA256 = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
AUTH_SCHEMA = "dense6x1024-direct-neuron-discriminator-authorization-v1"
REPORT_SCHEMA = "dense6x1024-direct-neuron-discriminator-report-v1"
WORKER_SCHEMA = "dense6x1024-direct-neuron-discriminator-worker-v1"
CHILD_PROCESS_SCHEMA = "dense6x1024-direct-neuron-discriminator-child-process-v1"
CHILD_ENV_SCHEMA = "dense6x1024-direct-neuron-discriminator-child-environment-v1"
CHILD_GROUP_SCHEMA = "dense6x1024-direct-neuron-discriminator-process-group-v1"
PROCESS_REVALIDATION_SCHEMA = "dense6x1024-direct-neuron-discriminator-process-revalidation-v1"
PENDING_HEAD = "PENDING_EXTERNAL_VERIFIER_HEAD"
ROWS = (128, 512, 32768)
REPETITIONS = 2
WORLD_SIZE = 4
SEED = 42
HOST_THREADS = 4
OUTER_TIMEOUT = ["timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "5400s"]
NATIVE_PYTHON = "/home/ubuntu/workspace/native_venv/bin/python"
TORCHRUN = "/home/ubuntu/workspace/native_venv/bin/torchrun"
NEURON_CC_FLAGS = "--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
EXPECTED_HOSTNAME = "ip-172-31-32-253"
EXPECTED_HOST = "trn22"
EXPECTED_REMOTE = "/home/ubuntu/trainium_freshstart_submission_validation_20260923_direct_discriminator"
EXPECTED_PURPOSE = (
    "Source/direct Neuron component discriminator only; no training, submission, or scientific verdict."
)
PROCESS_GROUP_TERM_GRACE_SECONDS = 5.0
PROCESS_GROUP_KILL_GRACE_SECONDS = 5.0
PROCESS_GROUP_POLL_SECONDS = 0.05
STATIC_ENVIRONMENT = {
    "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
    "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv",
    "NEURON_LOGICAL_NC_CONFIG": "2",
    "WORLD_SIZE": "4",
    "LOCAL_WORLD_SIZE": "4",
    "NEURON_CC_FLAGS": NEURON_CC_FLAGS,
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "PYTHONUNBUFFERED": "1",
    "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "1",
}
RECOMPILE_PATTERNS = (
    re.compile(r"recompil(?:e|ation|ing).*(?:limit|too many|exceed)", re.I),
    re.compile(r"(?:limit|too many|exceed).*recompil", re.I),
)
PROTECTED_PATHS = (
    "train.py",
    "prepare.py",
    "research/fresh_start_guard.py",
    "research/fresh_start_host.py",
    "research/fresh_start_launch.py",
    "research/fresh_start_backup.py",
    "research/fresh_start_sync.py",
    "research/run_experiment.py",
    "research/scratch/optimizer_ownership_check.py",
    "research/run-logs/fs-w62-optimizer-ownership-padded-neuron-gate-r1.job.json",
    "submissions/dense6x1024_lean_review/train.py",
    "submissions/dense6x1024_direct_review/train.py",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/cpu_equivalence_direct.py",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron_discriminator_direct.py",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/test_neuron_discriminator_static.py",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron-discriminator-runbook.md",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/proposal.json",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/fs-20260923-dense6x1024-direct-neuron-component-20260923T103500000001Z-result.json",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/independent-neuron-discriminator-verifier.json",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/independent-neuron-discriminator-fix-harness-verifier.json",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/independent-analyst-neuron-discriminator-r2.json",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/independent-neuron-discriminator-prepare-boundary-verifier.json",
)
LAUNCHER_DEFAULT_FINGERPRINT_PATHS = (
    "train.py", "prepare.py", "research/run_experiment.py",
    "research/fresh_start_host.py", "research/fresh_start_guard.py",
    "research/fresh_start_backup.py",
)
AUTHORIZATION_KEYS = {
    "schema", "authorization_state", "purpose", "expected_sha256", "request_contract",
}
FIXTURE_SPECS = {
    "tiny_fixed": {"steps": 2, "parameter_count": 40, "element_count": 43788},
    "tiny_changed": {"steps": 4, "parameter_count": 40, "element_count": 43788},
    "production_width1024_fs_w62": {
        "steps": 4, "parameter_count": 62, "element_count": 117453580,
    },
}
DISTRIBUTED_ENV_NAMES = {
    "WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK",
    "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
    "MASTER_ADDR", "MASTER_PORT", "NODE_RANK",
}
DISTRIBUTED_ENV_PREFIXES = (
    "TORCHELASTIC_", "TORCHRUN_", "PET_", "RDZV_", "RENDEZVOUS_",
    "PMI_", "PMIX_", "OMPI_COMM_WORLD_",
)
SAFE_CHILD_ENV_NAMES = (
    "PATH", "NKI_VENV_PATH", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
    "NEURON_COMPILE_CACHE_DIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "PYTHONUNBUFFERED", "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION",
    "WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK",
    "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
    "MASTER_ADDR", "MASTER_PORT", "NODE_RANK", "TRAINIUM_DISCRIMINATOR_LABEL",
    "TRAINIUM_DISCRIMINATOR_EXPECTED_COMMIT", "TRAINIUM_DISCRIMINATOR_AUTHORIZED_HEAD",
    "TRAINIUM_DISCRIMINATOR_AUTHORIZATION_SHA256", "TRAINIUM_FRESH_START_RUN_ID",
    "TRAINIUM_DISCRIMINATOR_PREPARE_SHA256",
)
CHILD_PROCESS_KEYS = {
    "schema", "name", "argv", "environment_identity", "timeout_seconds",
    "started_at_utc", "completed_at_utc", "duration_seconds", "returncode",
    "timed_out", "error", "recompile_limit_warnings", "log", "artifacts",
    "cache", "process_group", "passed", "record_path",
}
CHILD_LOG_KEYS = {"path", "stream", "bytes", "sha256", "complete"}
CHILD_ARTIFACT_KEYS = {"path", "exists", "regular", "symlink", "bytes", "sha256"}
CHILD_CACHE_KEYS = {"path", "exists", "directory", "symlink", "entry_count"}
CHILD_RECEIPT_KEYS = {"path", "sha256", "bytes"}
CHILD_GROUP_KEYS = {
    "schema", "start_new_session", "launched", "leader_pid", "process_group_id",
    "session_id", "term_grace_seconds", "kill_grace_seconds", "term_sent",
    "kill_sent", "primary_reaped", "quiescent", "quiescence_checks", "cleanup_errors", "proof",
}
BASE_CHILD_ENV_KEYS = {
    "schema", "mode", "retained", "elastic_names_present",
    "removed_distributed_names", "parent_nonce_sha256",
}
CE_CHILD_ENV_KEYS = BASE_CHILD_ENV_KEYS | {
    "world_size", "local_world_size", "rank", "local_rank", "rendezvous_configured",
}
OPTIMIZER_CHILD_ENV_KEYS = BASE_CHILD_ENV_KEYS | {
    "world_size", "local_world_size", "rank_injected_by_torchrun",
    "rendezvous_injected_by_torchrun",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def exclusive_json(path: Path, value: Any) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"refusing to replace result: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(json_bytes(value))
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def static_source_checks() -> dict[str, Any]:
    source_text = SOURCE.read_text()
    direct_text = DIRECT.read_text()
    ast.parse(source_text, filename=str(SOURCE))
    ast.parse(direct_text, filename=str(DIRECT))
    ast.parse(HARNESS.read_text(), filename=str(HARNESS))
    checks = {
        "source_sha256": sha256(SOURCE) == SOURCE_SHA256,
        "direct_sha256": sha256(DIRECT) == DIRECT_SHA256,
        "prepare_sha256": sha256(PREPARE) == PREPARE_SHA256,
        "source_ce_adapter": "def _submission_ce_install(" in source_text,
        "source_optimizer_adapter": "def _submission_optimizer_install(" in source_text,
        "direct_native_ce": "class _SoftcapCrossEntropyNKI" in direct_text,
        "direct_explicit_owner": "owned_optimizer = OwnedOptimizer(optimizer)" in direct_text,
        "same_lane_rule": source_text[source_text.index("def selected_lanes(rows):"):].split("\n\n", 1)[0]
        == direct_text[direct_text.index("def selected_lanes(rows):"):].split("\n", 4)[0]
        + "\n" + "\n".join(direct_text[direct_text.index("def selected_lanes(rows):"):].split("\n", 4)[1:4]),
        "ce_rows_frozen": ROWS == (128, 512, 32768),
        "ce_repetitions_frozen": REPETITIONS == 2,
        "world_size_frozen": WORLD_SIZE == 4,
        "seed_frozen": SEED == 42,
    }
    # The textual comparison above is intentionally supplemented by semantic
    # pins because comments/EOF placement are not a numerical contract.
    checks["lane_rule_source"] = (
        "return 2 if rows >= 512 and rows % 256 == 0 else 1" in source_text
    )
    checks["lane_rule_direct"] = (
        "return 2 if rows >= 512 and rows % 256 == 0 else 1" in direct_text
    )
    # Do not make harmless formatting a gate.
    checks.pop("same_lane_rule")
    return {"passed": all(checks.values()), "checks": checks}


def expected_contract() -> dict[str, Any]:
    return {
        "seed": SEED,
        "host": EXPECTED_HOST,
        "expected_hostname": EXPECTED_HOSTNAME,
        "remote": EXPECTED_REMOTE,
        "host_threads": HOST_THREADS,
        "world_size": WORLD_SIZE,
        "logical_nc_config": "2",
        "source_path": rel(SOURCE),
        "source_sha256": SOURCE_SHA256,
        "direct_path": rel(DIRECT),
        "direct_sha256": DIRECT_SHA256,
        "prepare_path": rel(PREPARE),
        "prepare_sha256": PREPARE_SHA256,
        "ce_rows": list(ROWS),
        "ce_repetitions": REPETITIONS,
        "optimizer_fixtures": ["tiny_fixed", "tiny_changed", "production_width1024_fs_w62"],
        "outer_timeout_argv": OUTER_TIMEOUT,
        "native_python": NATIVE_PYTHON,
        "torchrun_argv_prefix": [TORCHRUN, "--standalone", "--nproc_per_node=4"],
        "environment_static": STATIC_ENVIRONMENT,
        "recompile_limit_warning_policy": "invalidate",
        "worker_isolation": "one variant per process; one fresh compile cache per variant",
        "ce_child_environment": {
            "distributed_exact_names_removed": sorted(DISTRIBUTED_ENV_NAMES),
            "distributed_prefixes_removed": list(DISTRIBUTED_ENV_PREFIXES),
            "logical_singleton": {"world_size": 1, "local_world_size": 1, "rank": 0, "local_rank": 0},
            "rendezvous_configured": False,
        },
        "optimizer_child_environment": {
            "retained_outer": {"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"},
            "rank_and_rendezvous": "injected by pinned four-rank torchrun",
        },
        "child_evidence": {
            "log": "immutable complete combined stdout/stderr with SHA-256",
            "record": "immutable per-child JSON before orchestration decision",
            "top_failure": "all accumulated child records and receipts retained",
        },
        "child_process_group": {
            "start_new_session": True,
            "term_grace_seconds": PROCESS_GROUP_TERM_GRACE_SECONDS,
            "kill_grace_seconds": PROCESS_GROUP_KILL_GRACE_SECONDS,
            "poll_seconds": PROCESS_GROUP_POLL_SECONDS,
            "quiescence_required_before_snapshot": True,
            "pre_top_log_artifact_cache_revalidation": True,
        },
        "prepare_import_boundary": {
            "repository_root_first": True,
            "explicit_canonical_module_load": True,
            "compile_authenticated_source_bytes": True,
            "bytecode_cache_consulted": False,
            "rehash_source_after_execution": True,
            "reject_preloaded_prepare": True,
            "reject_shadow_or_wrong_path": True,
            "regular_non_symlink_required": True,
            "authorized_sha256_required": PREPARE_SHA256,
        },
    }


def require_request_integrity(requested: Path) -> dict[str, Any]:
    requested = requested.absolute()
    if requested != REQUEST or requested.is_symlink() or not requested.is_file():
        raise ValueError(f"authorization request must be canonical, regular and non-symlink: {REQUEST}")
    relative = rel(requested)
    try:
        tracked = git("ls-files", "--error-unmatch", "--", relative)
    except subprocess.CalledProcessError as error:
        raise ValueError("authorization request is not tracked") from error
    if tracked != relative:
        raise ValueError("authorization request tracking identity mismatch")
    dirty = git("status", "--porcelain=v1", "--untracked-files=all", "--", relative)
    if dirty:
        raise ValueError("authorization request is dirty/untracked")
    return {"path": str(requested), "relative": relative, "tracked": True, "clean": True}


def authenticate(args: argparse.Namespace, full: bool) -> dict[str, Any]:
    request_integrity = require_request_integrity(args.authorization_request)
    request = json.loads(REQUEST.read_text())
    if set(request) != AUTHORIZATION_KEYS:
        raise ValueError("authorization top-level key schema mismatch")
    if request.get("schema") != AUTH_SCHEMA:
        raise ValueError("authorization schema mismatch")
    if request.get("purpose") != EXPECTED_PURPOSE:
        raise ValueError("authorization purpose mismatch")
    if request.get("request_contract") != expected_contract():
        raise ValueError("authorization contract differs from hard-coded harness contract")
    actual_request_sha = sha256(REQUEST)
    if not re.fullmatch(r"[0-9a-f]{64}", args.authorization_sha256):
        raise ValueError("--authorization-sha256 must be a literal lowercase SHA-256")
    if args.authorization_sha256 != actual_request_sha:
        raise ValueError("external authorization SHA does not match request bytes")
    if args.static_only and args.authorized_head == PENDING_HEAD:
        if request.get("authorization_state") != "pending_external_verifier":
            raise ValueError("sentinel is only valid for pending static review")
        full = False
    else:
        if not re.fullmatch(r"[0-9a-f]{40}", args.authorized_head):
            raise ValueError("--authorized-head must be an external literal 40-hex commit")
        if request.get("authorization_state") != "finalized_by_external_verifier":
            raise ValueError("pending authorization request cannot launch")
    expected_hashes = request.get("expected_sha256")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise ValueError("authorization expected_sha256 is missing")
    if tuple(expected_hashes) != PROTECTED_PATHS:
        raise ValueError("authorization protected path set/order differs from hard-coded harness")
    actual_hashes = {}
    for name, expected in expected_hashes.items():
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"fingerprint target is not a regular file: {name}")
        actual_hashes[name] = sha256(path)
        if actual_hashes[name] != expected:
            raise ValueError(f"fingerprint mismatch: {name}")
    if sha256(SOURCE) != SOURCE_SHA256 or sha256(DIRECT) != DIRECT_SHA256:
        raise ValueError("frozen source/direct hash mismatch")
    head = git("rev-parse", "HEAD")
    if not args.static_only and head != args.authorized_head:
        raise ValueError("checkout is not the exact externally authorized commit")
    dirty = git("status", "--porcelain=v1", "--untracked-files=all", "--", *expected_hashes)
    if dirty:
        raise ValueError("fingerprinted paths are dirty/untracked")
    return {
        "request_sha256": actual_request_sha,
        "authorized_head": args.authorized_head,
        "head": head,
        "expected_sha256": expected_hashes,
        "actual_sha256": actual_hashes,
        "state": request.get("authorization_state"),
        "request_integrity": request_integrity,
    }


def path_from_sys_path(entry: str) -> Path:
    return Path(entry if entry else os.getcwd()).absolute().resolve()


def load_pinned_prepare(
    root: Path | None = None, expected_sha256: str | None = None,
    authorized_sha256: str | None = None,
) -> dict[str, Any]:
    root = ROOT if root is None else root
    expected_sha256 = PREPARE_SHA256 if expected_sha256 is None else expected_sha256
    root = root.absolute()
    prepare_path = root / "prepare.py"
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"repository root is not a regular non-symlink directory: {root}")
    if prepare_path.is_symlink() or not prepare_path.is_file():
        raise ValueError(f"prepare dependency is not a regular non-symlink file: {prepare_path}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("pinned prepare SHA-256 is not a lowercase literal")
    authorized_sha256 = authorized_sha256 or os.environ.get(
        "TRAINIUM_DISCRIMINATOR_PREPARE_SHA256"
    )
    if authorized_sha256 != expected_sha256:
        raise ValueError("authorized prepare SHA-256 differs from the pinned harness value")
    source_bytes = prepare_path.read_bytes()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("prepare.py SHA-256 differs from the authorized protected hash")
    if root.resolve() == ROOT.resolve():
        request = json.loads(REQUEST.read_text())
        request_prepare_sha = request.get("expected_sha256", {}).get("prepare.py")
        if request_prepare_sha != authorized_sha256:
            raise ValueError("authorization request does not pin the active prepare.py SHA-256")
    if "prepare" in sys.modules:
        module = sys.modules["prepare"]
        module_path = getattr(module, "__file__", None) if module is not None else None
        raise RuntimeError(f"refusing preloaded prepare module: {module_path!r}")
    importlib.invalidate_caches()
    try:
        discovered = importlib.util.find_spec("prepare")
    except (ImportError, AttributeError, ValueError) as error:
        raise RuntimeError(f"cannot safely resolve prepare before controlled load: {error}") from error
    if discovered is not None:
        if not discovered.origin:
            raise RuntimeError("shadow prepare module has no regular file origin")
        discovered_path = Path(discovered.origin).absolute()
        if (discovered_path.is_symlink() or not discovered_path.is_file()
                or discovered_path.resolve() != prepare_path.resolve()):
            raise RuntimeError(f"shadow/wrong-path prepare module would resolve from: {discovered.origin}")
    filtered = []
    for entry in sys.path:
        try:
            same_root = path_from_sys_path(entry) == root.resolve()
        except (OSError, RuntimeError, ValueError):
            same_root = False
        if not same_root:
            filtered.append(entry)
    sys.path[:] = [str(root), *filtered]
    importlib.invalidate_caches()
    controlled = importlib.util.find_spec("prepare")
    if (controlled is None or not controlled.origin
            or Path(controlled.origin).absolute().resolve() != prepare_path.resolve()):
        raise RuntimeError("repository-root-first prepare resolution is not canonical")
    try:
        code = compile(source_bytes, str(prepare_path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError, TypeError) as error:
        raise RuntimeError(f"cannot compile authenticated prepare.py source bytes: {error}") from error
    spec = importlib.machinery.ModuleSpec("prepare", loader=None, origin=str(prepare_path))
    module = types.ModuleType("prepare")
    module.__file__ = str(prepare_path)
    module.__cached__ = None
    module.__loader__ = None
    module.__package__ = ""
    module.__spec__ = spec
    sys.modules["prepare"] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        if sys.modules.get("prepare") is module:
            sys.modules.pop("prepare", None)
        raise
    loaded_path = Path(getattr(module, "__file__", "")).absolute()
    if prepare_path.is_symlink() or not prepare_path.is_file():
        sys.modules.pop("prepare", None)
        raise RuntimeError("prepare.py stopped being a regular non-symlink after execution")
    try:
        post_source_bytes = prepare_path.read_bytes()
    except OSError as error:
        sys.modules.pop("prepare", None)
        raise RuntimeError(f"cannot re-read prepare.py after execution: {error}") from error
    post_sha256 = hashlib.sha256(post_source_bytes).hexdigest()
    if (sys.modules.get("prepare") is not module or loaded_path.is_symlink()
            or not loaded_path.is_file() or loaded_path.resolve() != prepare_path.resolve()
            or post_source_bytes != source_bytes or post_sha256 != expected_sha256
            or module.__spec__ is not spec or module.__spec__.name != "prepare"
            or Path(module.__spec__.origin or "").absolute().resolve() != prepare_path.resolve()
            or module.__loader__ is not None or module.__cached__ is not None
            or module.__package__ != ""
            or not sys.path or path_from_sys_path(sys.path[0]) != root.resolve()):
        sys.modules.pop("prepare", None)
        raise RuntimeError("loaded prepare module identity changed across controlled import")
    return {
        "module": "prepare",
        "path": str(prepare_path),
        "sha256": actual_sha256,
        "repository_root": str(root),
        "repository_root_sys_path_index": 0,
        "execution": "compile-exact-authenticated-source-bytes",
        "source_rehashed_after_execution": True,
    }


def import_train(path: Path, name: str):
    load_pinned_prepare()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def is_distributed_environment_name(name: str) -> bool:
    return name in DISTRIBUTED_ENV_NAMES or name.startswith(DISTRIBUTED_ENV_PREFIXES)


def safe_child_environment_identity(
    environment: dict[str, str], mode: str, removed_names: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    retained = {name: environment[name] for name in SAFE_CHILD_ENV_NAMES if name in environment}
    elastic_names = sorted(name for name in environment if name.startswith(DISTRIBUTED_ENV_PREFIXES))
    parent_nonce = environment.get("TRAINIUM_DISCRIMINATOR_PARENT_NONCE", "")
    return {
        "schema": CHILD_ENV_SCHEMA,
        "mode": mode,
        "retained": retained,
        "elastic_names_present": elastic_names,
        "removed_distributed_names": sorted(removed_names),
        "parent_nonce_sha256": hashlib.sha256(parent_nonce.encode()).hexdigest() if parent_nonce else None,
    }


def ce_singleton_environment_identity(environment: dict[str, str]) -> dict[str, Any]:
    present = sorted(name for name in environment if is_distributed_environment_name(name))
    if present:
        raise RuntimeError(f"CE worker inherited distributed/rendezvous variables: {present}")
    identity = safe_child_environment_identity(environment, "ce-singleton")
    identity.update({
        "world_size": 1,
        "local_world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "rendezvous_configured": False,
    })
    return identity


def prepare_ce_environment(environment: dict[str, str], cache: Path) -> tuple[dict[str, str], dict[str, Any]]:
    child = dict(environment)
    removed = sorted(name for name in child if is_distributed_environment_name(name))
    for name in removed:
        child.pop(name)
    child["NEURON_COMPILE_CACHE_DIR"] = str(cache)
    singleton = ce_singleton_environment_identity(child)
    launch_identity = safe_child_environment_identity(child, "ce-singleton-launch", removed)
    launch_identity.update({
        "world_size": 1,
        "local_world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "rendezvous_configured": False,
    })
    if singleton["retained"] != launch_identity["retained"]:
        raise AssertionError("CE singleton environment identity is incoherent")
    return child, launch_identity


def prepare_optimizer_environment(
    environment: dict[str, str], cache: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    child = dict(environment)
    if child.get("WORLD_SIZE") != str(WORLD_SIZE) or child.get("LOCAL_WORLD_SIZE") != str(WORLD_SIZE):
        raise RuntimeError("optimizer torchrun launch requires retained WORLD_SIZE=LOCAL_WORLD_SIZE=4")
    removed = sorted(
        name for name in child
        if is_distributed_environment_name(name) and name not in {"WORLD_SIZE", "LOCAL_WORLD_SIZE"}
    )
    for name in removed:
        child.pop(name)
    child["NEURON_COMPILE_CACHE_DIR"] = str(cache)
    identity = safe_child_environment_identity(child, "optimizer-torchrun-four-rank-launch", removed)
    identity.update({
        "world_size": WORLD_SIZE,
        "local_world_size": WORLD_SIZE,
        "rank_injected_by_torchrun": True,
        "rendezvous_injected_by_torchrun": True,
    })
    return child, identity


def tensor_digest(torch, tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous().cpu()
    raw = value.view(torch.uint8).numpy().tobytes()
    finite = bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    magnitude = value.float()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": finite,
        "nonzero": bool(torch.count_nonzero(value)),
        "l2": float(torch.linalg.vector_norm(magnitude)) if value.numel() else 0.0,
        "max_abs": float(magnitude.abs().max()) if value.numel() else 0.0,
    }


def ce_fixture(torch, rows: int, repetition: int, device):
    generator = torch.Generator().manual_seed(SEED + rows * 17 + repetition)
    logits = torch.randn(rows, 8192, generator=generator, dtype=torch.float32)
    targets = torch.randint(0, 8192, (rows,), generator=generator, dtype=torch.int64)
    upstream = torch.randn(rows, generator=generator, dtype=torch.float32)
    return logits.to(device).requires_grad_(True), targets.to(device), upstream.to(device)


def run_ce_worker(args: argparse.Namespace) -> int:
    if os.environ.get("TRAINIUM_DISCRIMINATOR_PARENT_NONCE") != args.parent_nonce:
        raise RuntimeError("CE worker lacks its authenticated parent nonce")
    environment_identity = ce_singleton_environment_identity(dict(os.environ))
    import torch
    import torch.nn.functional as F
    import torch_neuronx  # noqa: F401

    torch.set_num_threads(HOST_THREADS)
    module = import_train(SOURCE if args.variant == "source" else DIRECT,
                          f"discriminator_{args.variant}_{os.getpid()}")
    module._HAS_NKI = True
    device = torch.device("neuron")
    calls: list[dict[str, Any]] = []
    if args.variant == "source":
        module._submission_ce_install(module)
        original_forward, original_backward = module._softcap_ce_fwd, module._softcap_ce_bwd

        def observed_forward(z, target):
            calls.append({"op": "forward", "rows": z.shape[0], "lanes": module.selected_lanes(z.shape[0])})
            return original_forward(z, target)

        def observed_backward(z, target, lse, gradient):
            calls.append({"op": "backward", "rows": z.shape[0], "lanes": module.selected_lanes(z.shape[0])})
            return original_backward(z, target, lse, gradient)

        module._softcap_ce_fwd = observed_forward
        module._softcap_ce_bwd = observed_backward
    else:
        class Probe:
            def __init__(self, wrapped, operation):
                self.wrapped, self.operation = wrapped, operation

            def __getitem__(self, grid):
                launch = self.wrapped[grid]

                def call(*values):
                    rows = values[0].shape[0]
                    calls.append({"op": self.operation, "rows": rows, "lanes": grid})
                    return launch(*values)
                return call

        module._softcap_ce_fwd_native = Probe(module._softcap_ce_fwd_native, "forward")
        module._softcap_ce_bwd_native = Probe(module._softcap_ce_bwd_native, "backward")

    source_payload = None
    source_pt = args.result_dir / "ce-source.pt"
    if args.variant == "direct":
        source_payload = torch.load(source_pt, map_location="cpu", weights_only=True)
    saved: dict[str, Any] = {}
    cases = []
    for rows in ROWS:
        for repetition in range(REPETITIONS):
            key = f"r{rows}-n{repetition}"
            logits, targets, upstream = ce_fixture(torch, rows, repetition, device)
            loss = module.softcap_cross_entropy(logits, targets, "none")
            loss.backward(upstream)
            module.synchronize(device)
            loss_cpu = loss.detach().cpu()
            grad_cpu = logits.grad.detach().cpu()
            record: dict[str, Any] = {
                "key": key,
                "rows": rows,
                "repetition": repetition,
                "selected_lanes": module.selected_lanes(rows),
                "loss": tensor_digest(torch, loss_cpu),
                "gradient": tensor_digest(torch, grad_cpu),
                "autograd_node": type(loss.grad_fn).__name__,
            }
            if args.variant == "source":
                saved[key] = {"loss": loss_cpu, "gradient": grad_cpu}
            else:
                expected = source_payload[key]
                loss_equal = bool(torch.equal(loss_cpu, expected["loss"]))
                grad_equal = bool(torch.equal(grad_cpu, expected["gradient"]))
                loss_max = float((loss_cpu - expected["loss"]).abs().max())
                grad_max = float((grad_cpu - expected["gradient"]).abs().max())
                # Independent PyTorch is diagnostic only.  It deliberately
                # cannot veto byte-exact source/direct equivalence.
                eager_logits, eager_targets, eager_upstream = ce_fixture(torch, rows, repetition, device)
                eager_loss = F.cross_entropy(15.0 * torch.tanh(eager_logits / 15.0), eager_targets,
                                             reduction="none")
                eager_loss.backward(eager_upstream)
                module.synchronize(device)
                record["source_direct"] = {
                    "torch_equal_loss": loss_equal,
                    "torch_equal_gradient": grad_equal,
                    "max_abs_loss": loss_max,
                    "max_abs_gradient": grad_max,
                    "passed": loss_equal and grad_equal and loss_max == 0.0 and grad_max == 0.0,
                }
                record["pytorch_diagnostic_numerically_non_gating"] = {
                    "loss_max_abs": float((loss_cpu - eager_loss.detach().cpu()).abs().max()),
                    "gradient_max_abs": float((grad_cpu - eager_logits.grad.detach().cpu()).abs().max()),
                }
                del eager_logits, eager_targets, eager_upstream, eager_loss
            cases.append(record)
            del logits, targets, upstream, loss, loss_cpu, grad_cpu
    if args.variant == "source":
        if source_pt.exists() or source_pt.is_symlink():
            raise FileExistsError(source_pt)
        torch.save(saved, source_pt)
    expected_calls = expected_ce_calls()
    passed = calls == expected_calls
    if args.variant == "direct":
        passed = passed and all(case["source_direct"]["passed"] for case in cases)
    result = {
        "schema": WORKER_SCHEMA,
        "worker": "ce",
        "variant": args.variant,
        "passed": passed,
        "cases": cases,
        "calls": calls,
        "expected_calls": expected_calls,
        "compile_cache": os.environ.get("NEURON_COMPILE_CACHE_DIR"),
        "environment_identity": environment_identity,
    }
    exclusive_json(args.result_dir / f"ce-{args.variant}.json", result)
    return 0 if passed else 1


def canonical(value: Any, torch) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_digest(torch, value)
    if isinstance(value, dict):
        return {str(k): canonical(v, torch) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v, torch) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported snapshot value: {type(value)}")


def configure_optimizer_module(module, torch) -> None:
    module.USE_FOREACH_OPTIM = True
    module.USE_TENSOR_LR_SCALARS = True
    module.USE_COMPILE_MUON = True
    module.USE_NKI_MUON = False
    module.USE_CAUTIOUS_WD = False
    module.COMPUTE_DTYPE = torch.bfloat16


def group_templates(module, torch, fixture: str) -> list[dict[str, Any]]:
    if fixture == "production_width1024_fs_w62":
        module.DEPTH, module.N_EMBD = 6, 1024
        module.USE_NGRAM_VE = module.USE_QK_SHIFT = False
        module.QK_SHIFT_BETA, module.QK_SHIFT_FREEZE = 0.0, True
        module.USE_LOCAL_CONV = module.TRAIN_LOCAL_CONV = True
        with torch.device("meta"):
            model = module.GPT(module.build_config())
            groups = model.setup_optimizer().param_groups
        if sum(p.numel() for g in groups for p in g["params"]) != 117453580:
            raise AssertionError("production width-1024 inventory drifted from fs-w62")
        return groups
    groups = [
        dict(kind="adamw", params=[torch.empty(64, 32, device="meta") for _ in range(5)],
             lr=.02, betas=(.8, .96), eps=1e-10, weight_decay=.01),
        dict(kind="adamw", params=[torch.empty(1, device="meta") for _ in range(12)],
             lr=.002, betas=(.9, .95), eps=1e-10, weight_decay=0.0),
    ]
    for shape, count in (((8, 32), 3), ((32, 32), 8), ((32, 64), 6), ((64, 32), 6)):
        groups.append(dict(kind="muon", params=[torch.empty(shape, device="meta") for _ in range(count)],
                           lr=.02, momentum=.95, beta2=.9, ns_steps=5,
                           weight_decay=.2, transposed=False))
    return groups


def instantiate_optimizer(module, torch, templates, device):
    generator = torch.Generator().manual_seed(9407)
    groups, parameters = [], []
    for template in templates:
        copied = {k: v for k, v in template.items() if k != "params"}
        copied["params"] = []
        for meta in template["params"]:
            p = torch.nn.Parameter((torch.randn(meta.shape, generator=generator) * .03).to(device))
            copied["params"].append(p)
            parameters.append(p)
        groups.append(copied)
    return parameters, module.MuonAdamW(groups), generator


def plan_snapshot(module, optimizer, owner) -> dict[str, Any]:
    entries, _, summary = module.make_plan(optimizer.param_groups, WORLD_SIZE)
    expected = {"entries": entries, "summary": summary}
    if owner is not None:
        actual = {"entries": owner.entries, "summary": owner.summary}
        if actual != expected:
            raise AssertionError("owner plan differs from canonical make_plan")
    return expected


def persistent_state(torch, optimizer, owner) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if owner is None:
        for gi, group in enumerate(optimizer.param_groups):
            if group["kind"] == "adamw":
                for pi, parameter in enumerate(group["params"]):
                    fields = optimizer.state.get(parameter, {})
                    result[f"g{gi}:p{pi}"] = canonical(fields, torch)
            else:
                fields = optimizer.state.get(group["params"][0], {})
                for pi in range(len(group["params"])):
                    mapped = {}
                    for key, value in fields.items():
                        if key in ("stacked_params", "stacked_grads"):
                            continue
                        mapped[key] = value[pi] if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == len(group["params"]) else value
                    result[f"g{gi}:p{pi}"] = canonical(mapped, torch)
        return result
    for gi, (source, owned, positions) in enumerate(zip(optimizer.param_groups, owner.groups, owner.positions)):
        if source["kind"] == "adamw":
            for local, position in enumerate(positions):
                result[f"g{gi}:p{position}"] = canonical(optimizer.state.get(owned["params"][local], {}), torch)
        elif positions:
            fields = optimizer.state.get(owned["params"][0], {})
            for local, position in enumerate(positions):
                mapped = {}
                for key, value in fields.items():
                    if key in ("stacked_params", "stacked_grads"):
                        continue
                    mapped[key] = value[local] if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == len(owned["params"]) else value
                result[f"g{gi}:p{position}"] = canonical(mapped, torch)
    return result


def snapshot_optimizer(torch, module, optimizer, owner, parameters, device, step: int) -> dict[str, Any]:
    parameter_records = {str(i): tensor_digest(torch, p) for i, p in enumerate(parameters)}
    group_scalars = []
    for group in optimizer.param_groups:
        group_scalars.append(canonical({k: v for k, v in group.items() if k != "params"}, torch))
    digest = hashlib.sha256()
    for record in parameter_records.values():
        digest.update(bytes.fromhex(record["sha256"]))
    local = torch.tensor(list(digest.digest()), device=device, dtype=torch.float32)
    gathered = torch.empty(32 * WORLD_SIZE, device=device, dtype=torch.float32)
    module.dist.all_gather_into_tensor(gathered, local)
    module.synchronize(device)
    rank_digests = [bytes(int(v) for v in row).hex() for row in gathered.cpu().view(WORLD_SIZE, 32)]
    return {
        "step": step,
        "parameters": parameter_records,
        "state": persistent_state(torch, optimizer, owner),
        "group_scalars": group_scalars,
        "owner_calls": None if owner is None else owner.calls,
        "cross_rank_parameter_sha256": rank_digests,
        "cross_rank_exact": len(set(rank_digests)) == 1,
    }


def run_optimizer_fixture(torch, module, variant: str, fixture: str, device) -> dict[str, Any]:
    templates = group_templates(module, torch, fixture)
    parameters, optimizer, generator = instantiate_optimizer(module, torch, templates, device)
    if variant == "source":
        owner = module._submission_optimizer_install(optimizer, module)
        step_call = optimizer.step
    elif variant == "direct":
        owner = module.OwnedOptimizer(optimizer)
        step_call = owner.step
    else:
        owner = None
        step_call = optimizer.step
    plan = plan_snapshot(module, optimizer, owner)
    initial_parameters = {str(i): tensor_digest(torch, parameter)
                          for i, parameter in enumerate(parameters)}
    linear = projection = linear_input = cache_parameter = None
    if fixture == "production_width1024_fs_w62":
        cache_parameter = next(parameter for parameter in parameters if parameter.ndim == 2)
        linear = module.Linear(cache_parameter.shape[1], cache_parameter.shape[0], bias=False)
        linear.weight = cache_parameter
        projection = torch.compile(linear, backend="neuron", fullgraph=True, dynamic=False)
        linear_input = torch.randn(2, cache_parameter.shape[1], generator=generator).to(device).bfloat16()
    step_count = 2 if fixture == "tiny_fixed" else 4
    snapshots = []
    for iteration in range(1, step_count + 1):
        gradients = [torch.randn(p.shape, generator=generator) * .003 for p in parameters]
        update_adamw = True
        for group_index, group in enumerate(optimizer.param_groups):
            if fixture != "tiny_fixed":
                group["lr"] = 0.0 if iteration == 4 else templates[group_index]["lr"] / iteration
                if group["kind"] == "muon":
                    group["momentum"] = .85 + iteration * .02
                else:
                    group["betas"] = (.8 if iteration < 3 else .9, .96)
        if fixture != "tiny_fixed" and iteration == 2:
            for index in range(0, len(gradients), 5):
                gradients[index] = None
        if fixture != "tiny_fixed" and iteration == 3:
            update_adamw = False
            for index in range(1, len(gradients), 4):
                gradients[index].zero_()
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = None if gradient is None else gradient.to(device)
        step_call(update_adamw=update_adamw)
        module.synchronize(device)
        snapshot = snapshot_optimizer(torch, module, optimizer, owner, parameters, device, iteration)
        if linear is not None:
            with torch.no_grad():
                linear.refresh_weight_cache(torch.bfloat16)
                compiled = projection(linear_input)
                eager = torch.nn.functional.linear(linear_input, cache_parameter.bfloat16())
                module.synchronize(device)
                cache_exact = bool(torch.equal(linear._cached_weight.cpu(), cache_parameter.detach().cpu().bfloat16()))
                output_exact = bool(torch.equal(compiled.cpu(), eager.cpu()))
                snapshot["linear_cache"] = {
                    "cache_exact": cache_exact,
                    "output_exact": output_exact,
                    "compiled_output": tensor_digest(torch, compiled),
                    "eager_output": tensor_digest(torch, eager),
                }
                if not cache_exact or not output_exact:
                    raise AssertionError("production Linear cache refresh/output differs from fs-w62 reference")
        snapshots.append(snapshot)
        del gradients
    passed = all(item["cross_rank_exact"] for item in snapshots)
    result = {
        "fixture": fixture,
        "passed": passed,
        "parameter_count": len(parameters),
        "element_count": sum(p.numel() for p in parameters),
        "plan": plan,
        "initial_parameters": initial_parameters,
        "snapshots": snapshots,
    }
    del parameters, optimizer, owner
    return result


def run_optimizer_worker(args: argparse.Namespace) -> int:
    if os.environ.get("TRAINIUM_DISCRIMINATOR_PARENT_NONCE") != args.parent_nonce:
        raise RuntimeError("optimizer worker lacks its authenticated parent nonce")
    import torch
    import torch.distributed as dist
    import torch_neuronx  # noqa: F401

    torch.set_num_threads(HOST_THREADS)
    path = SOURCE if args.variant in ("raw", "source") else DIRECT
    module = import_train(path, f"optimizer_{args.variant}_{os.getpid()}")
    configure_optimizer_module(module, torch)
    _, rank, _, world_size, device = module.init_runtime("neuron")
    if world_size != WORLD_SIZE or rank != int(os.environ["LOCAL_RANK"]):
        raise RuntimeError("optimizer worker requires exact four-rank torchrun")
    try:
        fixtures = [
            run_optimizer_fixture(torch, module, args.variant, name, device)
            for name in ("tiny_fixed", "tiny_changed", "production_width1024_fs_w62")
        ]
        result = {
            "schema": WORKER_SCHEMA,
            "worker": "optimizer",
            "variant": args.variant,
            "rank": rank,
            "world_size": world_size,
            "passed": all(item["passed"] for item in fixtures),
            "fixtures": fixtures,
            "compile_cache": os.environ.get("NEURON_COMPILE_CACHE_DIR"),
        }
        exclusive_json(args.result_dir / f"optimizer-{args.variant}-rank{rank}.json", result)
        return 0 if result["passed"] else 1
    finally:
        dist.destroy_process_group()


def scan_recompile_warning(text: str) -> list[str]:
    return [line for line in text.splitlines() if any(pattern.search(line) for pattern in RECOMPILE_PATTERNS)]


def artifact_observation(path: Path) -> dict[str, Any]:
    exists = path.exists()
    symlink = path.is_symlink()
    regular = path.is_file() and not symlink
    return {
        "path": str(path),
        "exists": exists,
        "regular": regular,
        "symlink": symlink,
        "bytes": path.stat().st_size if regular else None,
        "sha256": sha256(path) if regular else None,
    }


def cache_observation(path: Path) -> dict[str, Any]:
    exists = path.exists()
    symlink = path.is_symlink()
    directory = path.is_dir() and not symlink
    return {
        "path": str(path),
        "exists": exists,
        "directory": directory,
        "symlink": symlink,
        "entry_count": len(list(path.iterdir())) if directory else None,
    }


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process_group_quiescence(
    process: subprocess.Popen[bytes], process_group_id: int, grace_seconds: float,
) -> tuple[bool, int]:
    deadline = time.monotonic() + grace_seconds
    checks = 0
    while True:
        process.poll()  # Reap the direct child as soon as it exits.
        checks += 1
        if not process_group_exists(process_group_id):
            return True, checks
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, checks
        time.sleep(min(PROCESS_GROUP_POLL_SECONDS, remaining))


def quiesce_process_group(
    process: subprocess.Popen[bytes], process_group_id: int, terminate: bool,
) -> dict[str, Any]:
    term_sent = False
    kill_sent = False
    checks = 0
    cleanup_errors = []
    if terminate and process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGTERM)
            term_sent = True
        except ProcessLookupError:
            pass
        except OSError as error:
            cleanup_errors.append(f"SIGTERM: {type(error).__name__}: {error}")
    quiescent, count = wait_for_process_group_quiescence(
        process, process_group_id,
        PROCESS_GROUP_TERM_GRACE_SECONDS if terminate else 0.0,
    )
    checks += count
    if not quiescent:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
            kill_sent = True
        except ProcessLookupError:
            pass
        except OSError as error:
            cleanup_errors.append(f"SIGKILL: {type(error).__name__}: {error}")
        quiescent, count = wait_for_process_group_quiescence(
            process, process_group_id, PROCESS_GROUP_KILL_GRACE_SECONDS,
        )
        checks += count
    process.poll()
    primary_reaped = process.returncode is not None
    return {
        "schema": CHILD_GROUP_SCHEMA,
        "start_new_session": True,
        "launched": True,
        "leader_pid": process.pid,
        "process_group_id": process_group_id,
        "session_id": process_group_id,
        "term_grace_seconds": PROCESS_GROUP_TERM_GRACE_SECONDS,
        "kill_grace_seconds": PROCESS_GROUP_KILL_GRACE_SECONDS,
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "primary_reaped": primary_reaped,
        "quiescent": quiescent,
        "quiescence_checks": checks,
        "cleanup_errors": cleanup_errors,
        "proof": (
            "killpg(process_group_id, 0) raised ProcessLookupError after direct-child reap"
            if quiescent else "process group remained observable after bounded SIGTERM/SIGKILL"
        ),
    }


def unlaunched_process_group_record() -> dict[str, Any]:
    return {
        "schema": CHILD_GROUP_SCHEMA,
        "start_new_session": True,
        "launched": False,
        "leader_pid": None,
        "process_group_id": None,
        "session_id": None,
        "term_grace_seconds": PROCESS_GROUP_TERM_GRACE_SECONDS,
        "kill_grace_seconds": PROCESS_GROUP_KILL_GRACE_SECONDS,
        "term_sent": False,
        "kill_sent": False,
        "primary_reaped": False,
        "quiescent": True,
        "quiescence_checks": 0,
        "cleanup_errors": [],
        "proof": "no process group was launched",
    }


def validate_child_process_record(record: Any, include_receipt: bool = False) -> None:
    expected = CHILD_PROCESS_KEYS | ({"record_receipt"} if include_receipt else set())
    require_exact_keys(record, expected, "child_process")
    if record["schema"] != CHILD_PROCESS_SCHEMA or not isinstance(record["name"], str):
        raise ValueError("child_process: identity/schema mismatch")
    if (not isinstance(record["argv"], list) or not record["argv"]
            or any(not isinstance(value, str) for value in record["argv"])):
        raise ValueError("child_process: argv is not an exact nonempty string array")
    if type(record["timeout_seconds"]) is not int or record["timeout_seconds"] <= 0:
        raise ValueError("child_process: invalid timeout")
    if (not isinstance(record["started_at_utc"], str)
            or not isinstance(record["completed_at_utc"], str)
            or type(record["duration_seconds"]) not in (int, float)
            or not math.isfinite(record["duration_seconds"])
            or record["duration_seconds"] < 0):
        raise ValueError("child_process: invalid timing evidence")
    if type(record["timed_out"]) is not bool:
        raise ValueError("child_process: invalid timeout state")
    if record["returncode"] is not None and type(record["returncode"]) is not int:
        raise ValueError("child_process: invalid returncode")
    if record["error"] is not None:
        require_exact_keys(record["error"], {"type", "message"}, "child_process.error")
        if not all(isinstance(value, str) for value in record["error"].values()):
            raise ValueError("child_process: invalid error evidence")
    if (not isinstance(record["recompile_limit_warnings"], list)
            or any(not isinstance(value, str) for value in record["recompile_limit_warnings"])):
        raise ValueError("child_process: invalid warning evidence")
    log = record["log"]
    require_exact_keys(log, CHILD_LOG_KEYS, "child_process.log")
    if (not isinstance(log["path"], str) or log["stream"] != "combined-stdout-stderr"
            or type(log["bytes"]) is not int or log["bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", log["sha256"] or "")
            or log["complete"] is not True):
        raise ValueError("child_process: invalid immutable log evidence")
    log_path = Path(log["path"])
    if (log_path.is_symlink() or not log_path.is_file() or log_path.stat().st_size != log["bytes"]
            or sha256(log_path) != log["sha256"]):
        raise ValueError("child_process: immutable log bytes differ from evidence")
    if not isinstance(record["artifacts"], dict):
        raise ValueError("child_process: artifact observations missing")
    for name, observation in record["artifacts"].items():
        if not isinstance(name, str):
            raise ValueError("child_process: artifact name is not a string")
        require_exact_keys(observation, CHILD_ARTIFACT_KEYS, f"child_process.artifacts.{name}")
        if (type(observation["exists"]) is not bool or type(observation["regular"]) is not bool
                or type(observation["symlink"]) is not bool):
            raise ValueError("child_process: invalid artifact observation")
        if observation["regular"]:
            if (type(observation["bytes"]) is not int or observation["bytes"] < 0
                    or not re.fullmatch(r"[0-9a-f]{64}", observation["sha256"] or "")):
                raise ValueError("child_process: invalid regular artifact observation")
        elif observation["bytes"] is not None or observation["sha256"] is not None:
            raise ValueError("child_process: absent/unsafe artifact has digest evidence")
    cache = record["cache"]
    require_exact_keys(cache, CHILD_CACHE_KEYS, "child_process.cache")
    if (type(cache["exists"]) is not bool or type(cache["directory"]) is not bool
            or type(cache["symlink"]) is not bool
            or (cache["directory"] and (type(cache["entry_count"]) is not int or cache["entry_count"] < 0))
            or (not cache["directory"] and cache["entry_count"] is not None)):
        raise ValueError("child_process: invalid cache observation")
    group = record["process_group"]
    require_exact_keys(group, CHILD_GROUP_KEYS, "child_process.process_group")
    if (group["schema"] != CHILD_GROUP_SCHEMA or group["start_new_session"] is not True
            or group["term_grace_seconds"] != PROCESS_GROUP_TERM_GRACE_SECONDS
            or group["kill_grace_seconds"] != PROCESS_GROUP_KILL_GRACE_SECONDS
            or type(group["launched"]) is not bool or type(group["term_sent"]) is not bool
            or type(group["kill_sent"]) is not bool or type(group["primary_reaped"]) is not bool
            or type(group["quiescent"]) is not bool or type(group["quiescence_checks"]) is not int
            or group["quiescence_checks"] < 0 or not isinstance(group["cleanup_errors"], list)
            or any(not isinstance(value, str) for value in group["cleanup_errors"])
            or not isinstance(group["proof"], str)):
        raise ValueError("child_process: invalid process-group evidence")
    if group["launched"]:
        if (type(group["leader_pid"]) is not int or group["leader_pid"] <= 0
                or group["process_group_id"] != group["leader_pid"]
                or group["session_id"] != group["leader_pid"]
                or group["primary_reaped"] is not True or group["quiescent"] is not True
                or group["quiescence_checks"] <= 0):
            raise ValueError("child_process: launched process group is not proven quiescent/reaped")
    elif (group["leader_pid"] is not None or group["process_group_id"] is not None
            or group["session_id"] is not None or group["term_sent"] or group["kill_sent"]
            or group["primary_reaped"] or group["quiescent"] is not True
            or group["quiescence_checks"] != 0 or group["cleanup_errors"]):
        raise ValueError("child_process: unlaunched process-group evidence is incoherent")
    environment = record["environment_identity"]
    if not isinstance(environment, dict) or environment.get("schema") != CHILD_ENV_SCHEMA:
        raise ValueError("child_process: sanitized environment identity missing")
    if environment.get("mode") == "ce-singleton-launch":
        require_exact_keys(environment, CE_CHILD_ENV_KEYS, "child_process.environment_identity")
        if (environment["world_size"], environment["local_world_size"], environment["rank"],
                environment["local_rank"], environment["rendezvous_configured"]) != (1, 1, 0, 0, False):
            raise ValueError("child_process: CE launch is not a coherent singleton")
        if (environment["elastic_names_present"]
                or any(is_distributed_environment_name(name) for name in environment["retained"])):
            raise ValueError("child_process: CE launch retained distributed environment")
    elif environment.get("mode") == "optimizer-torchrun-four-rank-launch":
        require_exact_keys(environment, OPTIMIZER_CHILD_ENV_KEYS, "child_process.environment_identity")
        if (environment["world_size"] != WORLD_SIZE or environment["local_world_size"] != WORLD_SIZE
                or environment["rank_injected_by_torchrun"] is not True
                or environment["rendezvous_injected_by_torchrun"] is not True
                or environment["retained"].get("WORLD_SIZE") != str(WORLD_SIZE)
                or environment["retained"].get("LOCAL_WORLD_SIZE") != str(WORLD_SIZE)
                or environment["elastic_names_present"]):
            raise ValueError("child_process: optimizer launch is not coherent four-rank torchrun")
    else:
        raise ValueError("child_process: unknown environment mode")
    if (not re.fullmatch(r"[0-9a-f]{64}", environment["parent_nonce_sha256"] or "")
            or not isinstance(environment["removed_distributed_names"], list)
            or environment["removed_distributed_names"] != sorted(set(environment["removed_distributed_names"]))
            or any(not is_distributed_environment_name(name)
                   for name in environment["removed_distributed_names"])):
        raise ValueError("child_process: invalid sanitized environment identity")
    expected_passed = (
        record["returncode"] == 0 and record["timed_out"] is False
        and record["error"] is None and not record["recompile_limit_warnings"]
        and group["launched"] is True and group["primary_reaped"] is True
        and group["quiescent"] is True and not group["cleanup_errors"]
    )
    if record["passed"] is not expected_passed:
        raise ValueError("child_process: pass state differs from recorded process evidence")
    if include_receipt:
        receipt = record["record_receipt"]
        require_exact_keys(receipt, CHILD_RECEIPT_KEYS, "child_process.record_receipt")
        record_path = Path(receipt["path"])
        if (receipt["path"] != record["record_path"] or record_path.is_symlink()
                or not record_path.is_file() or type(receipt["bytes"]) is not int
                or record_path.stat().st_size != receipt["bytes"]
                or not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"] or "")
                or sha256(record_path) != receipt["sha256"]):
            raise ValueError("child_process: invalid immutable record receipt")
        if json.loads(record_path.read_text()) != {
            key: value for key, value in record.items() if key != "record_receipt"
        }:
            raise ValueError("child_process: persisted record differs from top-level evidence")


def run_child(
    name: str,
    command: list[str],
    env: dict[str, str],
    environment_identity: dict[str, Any],
    timeout: int,
    log_path: Path,
    record_path: Path,
    cache_path: Path,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    if log_path.exists() or log_path.is_symlink():
        raise FileExistsError(f"refusing to replace child log: {log_path}")
    if record_path.exists() or record_path.is_symlink():
        raise FileExistsError(f"refusing to replace child record: {record_path}")
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        record_fd = os.open(record_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except BaseException:
        os.close(log_fd)
        log_path.unlink()
        raise
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    returncode = None
    timed_out = False
    error_record = None
    process = None
    group_record = unlaunched_process_group_record()
    try:
        with os.fdopen(log_fd, "wb") as output:
            try:
                process = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as error:  # Persist spawn/exec failures before returning fail-closed.
                error_record = {"type": type(error).__name__, "message": str(error)}
            if process is not None:
                process_group_id = process.pid
                terminate_group = False
                try:
                    returncode = process.wait(timeout=timeout)
                    terminate_group = process_group_exists(process_group_id)
                    if terminate_group:
                        error_record = {
                            "type": "LingeringProcessGroup",
                            "message": "direct child exited while descendants remained in its process group",
                        }
                except subprocess.TimeoutExpired as error:
                    timed_out = True
                    error_record = {"type": type(error).__name__, "message": str(error)}
                    terminate_group = True
                except Exception as error:
                    error_record = {"type": type(error).__name__, "message": str(error)}
                    terminate_group = True
                group_record = quiesce_process_group(
                    process, process_group_id, terminate=terminate_group,
                )
                returncode = process.returncode
            if process is not None and not group_record["quiescent"]:
                raise RuntimeError("child process group did not become quiescent after bounded TERM/KILL")
            output.flush()
            os.fsync(output.fileno())
        duration = time.monotonic() - started
        completed_at = datetime.now(timezone.utc).isoformat()
        log_bytes = log_path.read_bytes()
        warnings = scan_recompile_warning(log_bytes.decode("utf-8", errors="replace"))
        record = {
            "schema": CHILD_PROCESS_SCHEMA,
            "name": name,
            "argv": list(command),
            "environment_identity": environment_identity,
            "timeout_seconds": timeout,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "duration_seconds": duration,
            "returncode": returncode,
            "timed_out": timed_out,
            "error": error_record,
            "recompile_limit_warnings": warnings,
            "log": {
                "path": str(log_path),
                "stream": "combined-stdout-stderr",
                "bytes": len(log_bytes),
                "sha256": hashlib.sha256(log_bytes).hexdigest(),
                "complete": True,
            },
            "artifacts": {
                artifact_name: artifact_observation(path)
                for artifact_name, path in artifact_paths.items()
            },
            "cache": cache_observation(cache_path),
            "process_group": group_record,
            "passed": (
                returncode == 0 and not timed_out and error_record is None and not warnings
                and group_record["launched"] and group_record["primary_reaped"]
                and group_record["quiescent"] and not group_record["cleanup_errors"]
            ),
            "record_path": str(record_path),
        }
        validate_child_process_record(record)
        with os.fdopen(record_fd, "wb") as output:
            output.write(json_bytes(record))
            output.flush()
            os.fsync(output.fileno())
        receipt = {
            "path": str(record_path),
            "sha256": sha256(record_path),
            "bytes": record_path.stat().st_size,
        }
        result = {**record, "record_receipt": receipt}
        validate_child_process_record(result, include_receipt=True)
        return result
    except BaseException:
        try:
            os.close(record_fd)
        except OSError:
            pass
        raise


def revalidate_process_evidence(processes: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for process in processes:
        failures = []
        try:
            validate_child_process_record(process, include_receipt=True)
        except Exception as error:
            failures.append(f"record/log/receipt: {type(error).__name__}: {error}")
        for name, expected in process.get("artifacts", {}).items():
            try:
                actual = artifact_observation(Path(expected["path"]))
                if actual != expected:
                    failures.append(f"artifact snapshot changed: {name}")
            except Exception as error:
                failures.append(f"artifact revalidation failed {name}: {type(error).__name__}: {error}")
        expected_cache = process.get("cache")
        if isinstance(expected_cache, dict) and isinstance(expected_cache.get("path"), str):
            try:
                if cache_observation(Path(expected_cache["path"])) != expected_cache:
                    failures.append("compile-cache snapshot changed")
            except Exception as error:
                failures.append(f"compile-cache revalidation failed: {type(error).__name__}: {error}")
        group = process.get("process_group")
        if isinstance(group, dict) and group.get("launched") is True:
            group_id = group.get("process_group_id")
            if type(group_id) is not int or process_group_exists(group_id):
                failures.append("process group is observable immediately before top-result write")
        results.append({
            "name": process.get("name"),
            "passed": not failures,
            "failures": failures,
            "record_path": process.get("record_path"),
            "record_sha256": process.get("record_receipt", {}).get("sha256"),
        })
    return {
        "schema": PROCESS_REVALIDATION_SCHEMA,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(result["passed"] for result in results),
        "process_count": len(processes),
        "results": results,
    }


def attach_prewrite_process_revalidation(report: dict[str, Any]) -> None:
    if report.get("mode") != "neuron":
        return
    processes = report.get("processes", [])
    if not isinstance(processes, list):
        report["passed"] = False
        report["process_evidence_revalidation"] = {
            "schema": PROCESS_REVALIDATION_SCHEMA,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "passed": False,
            "process_count": 0,
            "results": [],
        }
        report["evidence_failure"] = {
            "type": "ProcessEvidenceSchemaError", "message": "top report processes is not a list",
        }
        return
    revalidation = revalidate_process_evidence(processes)
    report["process_evidence_revalidation"] = revalidation
    if not revalidation["passed"]:
        report["passed"] = False
        report["evidence_failure"] = {
            "type": "ProcessEvidenceMutation",
            "message": "child log, record, artifact, cache, or quiescence evidence changed before top write",
        }


def expected_ce_calls() -> list[dict[str, Any]]:
    return [
        {"op": operation, "rows": rows, "lanes": 2 if rows >= 512 else 1}
        for rows in ROWS for _ in range(REPETITIONS) for operation in ("forward", "backward")
    ]


def validate_ce_report(report: Any, variant: str) -> None:
    require_exact_keys(
        report,
        {"schema", "worker", "variant", "passed", "cases", "calls", "expected_calls",
         "compile_cache", "environment_identity"},
        f"ce.{variant}",
    )
    if (report["schema"] != WORKER_SCHEMA or report["worker"] != "ce"
            or report["variant"] != variant or report["passed"] is not True):
        raise ValueError(f"ce.{variant}: worker identity/schema/pass mismatch")
    if not isinstance(report["compile_cache"], str) or not Path(report["compile_cache"]).is_absolute():
        raise ValueError(f"ce.{variant}: compile cache missing/non-absolute")
    environment = report["environment_identity"]
    require_exact_keys(
        environment,
        {"schema", "mode", "retained", "elastic_names_present", "removed_distributed_names",
         "parent_nonce_sha256", "world_size", "local_world_size", "rank", "local_rank",
         "rendezvous_configured"},
        f"ce.{variant}.environment_identity",
    )
    if (environment["schema"] != CHILD_ENV_SCHEMA or environment["mode"] != "ce-singleton"
            or environment["world_size"] != 1 or environment["local_world_size"] != 1
            or environment["rank"] != 0 or environment["local_rank"] != 0
            or environment["rendezvous_configured"] is not False
            or environment["elastic_names_present"] != []
            or environment["removed_distributed_names"] != []
            or any(is_distributed_environment_name(name) for name in environment["retained"])
            or not re.fullmatch(r"[0-9a-f]{64}", environment["parent_nonce_sha256"] or "")):
        raise ValueError(f"ce.{variant}: singleton environment proof mismatch")
    calls = expected_ce_calls()
    if report["calls"] != calls or report["expected_calls"] != calls:
        raise ValueError(f"ce.{variant}: exact lane/call trace mismatch")
    cases = report["cases"]
    if not isinstance(cases, list) or len(cases) != len(ROWS) * REPETITIONS:
        raise ValueError(f"ce.{variant}: case cardinality mismatch")
    index = 0
    for rows in ROWS:
        for repetition in range(REPETITIONS):
            case = cases[index]
            keys = {"key", "rows", "repetition", "selected_lanes", "loss", "gradient", "autograd_node"}
            if variant == "direct":
                keys |= {"source_direct", "pytorch_diagnostic_numerically_non_gating"}
            require_exact_keys(case, keys, f"ce.{variant}.cases[{index}]")
            if (case["key"] != f"r{rows}-n{repetition}" or case["rows"] != rows
                    or case["repetition"] != repetition
                    or case["selected_lanes"] != (2 if rows >= 512 else 1)
                    or not isinstance(case["autograd_node"], str) or not case["autograd_node"]):
                raise ValueError(f"ce.{variant}.cases[{index}]: case identity/wiring mismatch")
            validate_tensor_record(case["loss"], f"ce.{variant}.cases[{index}].loss")
            validate_tensor_record(case["gradient"], f"ce.{variant}.cases[{index}].gradient")
            if variant == "direct":
                exact = case["source_direct"]
                require_exact_keys(
                    exact,
                    {"torch_equal_loss", "torch_equal_gradient", "max_abs_loss", "max_abs_gradient", "passed"},
                    f"ce.direct.cases[{index}].source_direct",
                )
                if exact != {
                    "torch_equal_loss": True, "torch_equal_gradient": True,
                    "max_abs_loss": 0.0, "max_abs_gradient": 0.0, "passed": True,
                }:
                    raise ValueError(f"ce.direct.cases[{index}]: source/direct is not exact")
                diagnostic = case["pytorch_diagnostic_numerically_non_gating"]
                require_exact_keys(
                    diagnostic, {"loss_max_abs", "gradient_max_abs"},
                    f"ce.direct.cases[{index}].pytorch_diagnostic",
                )
                if any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in diagnostic.values()):
                    raise ValueError(f"ce.direct.cases[{index}]: invalid diagnostic values")
            index += 1


TENSOR_RECORD_KEYS = {"sha256", "shape", "dtype", "finite", "nonzero", "l2", "max_abs"}
WORKER_RECORD_KEYS = {
    "schema", "worker", "variant", "rank", "world_size", "passed", "fixtures", "compile_cache",
}
FIXTURE_RECORD_KEYS = {
    "fixture", "passed", "parameter_count", "element_count", "plan", "initial_parameters", "snapshots",
}
SNAPSHOT_KEYS = {
    "step", "parameters", "state", "group_scalars", "owner_calls",
    "cross_rank_parameter_sha256", "cross_rank_exact",
}


def require_exact_keys(value: Any, expected: set[str], path: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{path}: key schema mismatch")


def validate_tensor_record(value: Any, path: str, require_nonzero: bool = True) -> None:
    require_exact_keys(value, TENSOR_RECORD_KEYS, path)
    if not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
        raise ValueError(f"{path}: invalid tensor digest")
    if (not isinstance(value["shape"], list)
            or any(type(dimension) is not int or dimension <= 0 for dimension in value["shape"])):
        raise ValueError(f"{path}: invalid tensor shape")
    if not isinstance(value["dtype"], str) or not value["dtype"].startswith("torch."):
        raise ValueError(f"{path}: invalid tensor dtype")
    if value["finite"] is not True:
        raise ValueError(f"{path}: nonfinite tensor evidence")
    for field in ("l2", "max_abs"):
        if type(value[field]) not in (int, float) or not math.isfinite(value[field]) or value[field] < 0:
            raise ValueError(f"{path}: invalid {field}")
    if require_nonzero and (value["nonzero"] is not True or value["l2"] <= 0 or value["max_abs"] <= 0):
        raise ValueError(f"{path}: zero/no-op tensor evidence")


def validate_canonical_finite(value: Any, path: str) -> None:
    if isinstance(value, dict):
        if set(value) == TENSOR_RECORD_KEYS:
            validate_tensor_record(value, path)
        else:
            for key, item in value.items():
                validate_canonical_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_canonical_finite(item, f"{path}[{index}]")
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: nonfinite scalar")
    elif not isinstance(value, (str, int, bool)) and value is not None:
        raise ValueError(f"{path}: unsupported canonical value")


def validate_parameter_records(records: Any, count: int, path: str) -> None:
    expected = {str(index) for index in range(count)}
    if not isinstance(records, dict) or set(records) != expected:
        raise ValueError(f"{path}: parameter record cardinality/index mismatch")
    for index in range(count):
        validate_tensor_record(records[str(index)], f"{path}.{index}")


def validate_plan(plan: Any, parameter_count: int, element_count: int, path: str) -> list[dict[str, Any]]:
    require_exact_keys(plan, {"entries", "summary"}, path)
    entries = plan["entries"]
    if not isinstance(entries, list) or len(entries) != parameter_count:
        raise ValueError(f"{path}: plan entry cardinality mismatch")
    positions = set()
    elements_by_owner = [0] * WORLD_SIZE
    for index, entry in enumerate(entries):
        require_exact_keys(
            entry,
            {"index", "group", "position", "kind", "shape", "elements",
             "estimated_ms", "owner", "offset"},
            f"{path}.entries[{index}]",
        )
        if entry["index"] != index or type(entry["group"]) is not int or entry["group"] < 0:
            raise ValueError(f"{path}: invalid plan index/group")
        if type(entry["position"]) is not int or entry["position"] < 0:
            raise ValueError(f"{path}: invalid plan position")
        position = (entry["group"], entry["position"])
        if position in positions:
            raise ValueError(f"{path}: duplicate plan group/position")
        positions.add(position)
        if entry["kind"] not in ("adamw", "muon"):
            raise ValueError(f"{path}: invalid plan optimizer kind")
        if (not isinstance(entry["shape"], list) or not entry["shape"]
                or any(type(dimension) is not int or dimension <= 0 for dimension in entry["shape"])):
            raise ValueError(f"{path}: invalid plan shape")
        shape_elements = math.prod(entry["shape"])
        if entry["elements"] != shape_elements:
            raise ValueError(f"{path}: plan element/shape mismatch")
        if type(entry["owner"]) is not int or entry["owner"] not in range(WORLD_SIZE):
            raise ValueError(f"{path}: invalid plan owner")
        if type(entry["offset"]) is not int or entry["offset"] < 0:
            raise ValueError(f"{path}: invalid plan offset")
        if type(entry["estimated_ms"]) not in (int, float) or not math.isfinite(entry["estimated_ms"]) or entry["estimated_ms"] <= 0:
            raise ValueError(f"{path}: invalid plan cost")
        elements_by_owner[entry["owner"]] += entry["elements"]
    if sum(entry["elements"] for entry in entries) != element_count:
        raise ValueError(f"{path}: plan total element mismatch")
    group_ids = {entry["group"] for entry in entries}
    if group_ids != set(range(max(group_ids) + 1)):
        raise ValueError(f"{path}: plan groups are not contiguous")
    for group in group_ids:
        group_entries = [entry for entry in entries if entry["group"] == group]
        if ({entry["position"] for entry in group_entries} != set(range(len(group_entries)))
                or len({entry["kind"] for entry in group_entries}) != 1):
            raise ValueError(f"{path}: group positions/kind are inconsistent")
    summary = plan["summary"]
    require_exact_keys(
        summary,
        {"elements_by_owner", "estimated_ms_by_owner", "shard_elements", "total_padded_bytes"},
        f"{path}.summary",
    )
    if summary["elements_by_owner"] != elements_by_owner:
        raise ValueError(f"{path}: owner element summary mismatch")
    costs = summary["estimated_ms_by_owner"]
    if (not isinstance(costs, list) or len(costs) != WORLD_SIZE
            or any(type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0 for cost in costs)):
        raise ValueError(f"{path}: invalid owner cost summary")
    stride = summary["shard_elements"]
    if type(stride) is not int or stride % 128 or stride < max(elements_by_owner):
        raise ValueError(f"{path}: invalid shard stride")
    if summary["total_padded_bytes"] != stride * WORLD_SIZE * 4:
        raise ValueError(f"{path}: invalid padded byte total")
    return entries


def expected_state_keys(entries: list[dict[str, Any]], variant: str, rank: int) -> dict[str, str]:
    return {
        f"g{entry['group']}:p{entry['position']}": entry["kind"]
        for entry in entries if variant == "raw" or entry["owner"] == rank
    }


def validate_state(state: Any, expected: dict[str, str], path: str) -> None:
    if not isinstance(state, dict) or set(state) != set(expected) or not state:
        raise ValueError(f"{path}: optimizer state coverage mismatch/empty")
    for key, kind in expected.items():
        fields = state[key]
        required = {"step", "exp_avg", "exp_avg_sq"} if kind == "adamw" else {
            "momentum_buffer", "second_momentum_buffer"
        }
        require_exact_keys(fields, required, f"{path}.{key}")
        if kind == "adamw":
            if type(fields["step"]) is not int or fields["step"] <= 0:
                raise ValueError(f"{path}.{key}: invalid AdamW step")
            validate_tensor_record(fields["exp_avg"], f"{path}.{key}.exp_avg")
            validate_tensor_record(fields["exp_avg_sq"], f"{path}.{key}.exp_avg_sq")
        else:
            validate_tensor_record(fields["momentum_buffer"], f"{path}.{key}.momentum_buffer")
            validate_tensor_record(fields["second_momentum_buffer"], f"{path}.{key}.second_momentum_buffer")


def validate_snapshot(
    snapshot: Any, fixture: str, variant: str, rank: int, step: int,
    parameter_count: int, expected_state: dict[str, str], group_kinds: list[str], path: str,
) -> None:
    keys = set(SNAPSHOT_KEYS)
    if fixture == "production_width1024_fs_w62":
        keys.add("linear_cache")
    require_exact_keys(snapshot, keys, path)
    if snapshot["step"] != step:
        raise ValueError(f"{path}: nonsequential step")
    validate_parameter_records(snapshot["parameters"], parameter_count, f"{path}.parameters")
    validate_state(snapshot["state"], expected_state, f"{path}.state")
    if (not isinstance(snapshot["group_scalars"], list)
            or len(snapshot["group_scalars"]) != len(group_kinds)):
        raise ValueError(f"{path}: group scalar evidence missing")
    validate_canonical_finite(snapshot["group_scalars"], f"{path}.group_scalars")
    if any(not isinstance(group, dict) or group.get("kind") != group_kinds[index]
           for index, group in enumerate(snapshot["group_scalars"])):
        raise ValueError(f"{path}: group scalar kind/order mismatch")
    expected_calls = None if variant == "raw" else step
    if snapshot["owner_calls"] != expected_calls:
        raise ValueError(f"{path}: owner call count mismatch")
    digests = snapshot["cross_rank_parameter_sha256"]
    combined = hashlib.sha256(b"".join(
        bytes.fromhex(snapshot["parameters"][str(index)]["sha256"])
        for index in range(parameter_count)
    )).hexdigest()
    if (snapshot["cross_rank_exact"] is not True or not isinstance(digests, list)
            or digests != [combined] * WORLD_SIZE):
        raise ValueError(f"{path}: cross-rank parameter digest mismatch")
    if fixture == "production_width1024_fs_w62":
        cache = snapshot["linear_cache"]
        require_exact_keys(
            cache, {"cache_exact", "output_exact", "compiled_output", "eager_output"},
            f"{path}.linear_cache",
        )
        if cache["cache_exact"] is not True or cache["output_exact"] is not True:
            raise ValueError(f"{path}: Linear cache/output proof failed")
        validate_tensor_record(cache["compiled_output"], f"{path}.compiled_output")
        validate_tensor_record(cache["eager_output"], f"{path}.eager_output")
        if cache["compiled_output"] != cache["eager_output"]:
            raise ValueError(f"{path}: compiled/eager Linear output record mismatch")


def parameter_sha_map(records: dict[str, Any]) -> tuple[str, ...]:
    return tuple(records[str(index)]["sha256"] for index in range(len(records)))


def validate_fixture(fixture: Any, variant: str, rank: int, index: int) -> None:
    path = f"{variant}.rank{rank}.fixtures[{index}]"
    require_exact_keys(fixture, FIXTURE_RECORD_KEYS, path)
    expected_name = tuple(FIXTURE_SPECS)[index]
    spec = FIXTURE_SPECS[expected_name]
    if fixture["fixture"] != expected_name or fixture["passed"] is not True:
        raise ValueError(f"{path}: fixture identity/pass mismatch")
    if (fixture["parameter_count"] != spec["parameter_count"]
            or fixture["element_count"] != spec["element_count"]):
        raise ValueError(f"{path}: parameter/element count drift")
    validate_parameter_records(
        fixture["initial_parameters"], spec["parameter_count"], f"{path}.initial_parameters"
    )
    entries = validate_plan(
        fixture["plan"], spec["parameter_count"], spec["element_count"], f"{path}.plan"
    )
    expected_state = expected_state_keys(entries, variant, rank)
    group_kinds = [
        next(entry["kind"] for entry in entries if entry["group"] == group)
        for group in range(max(entry["group"] for entry in entries) + 1)
    ]
    snapshots = fixture["snapshots"]
    if not isinstance(snapshots, list) or len(snapshots) != spec["steps"]:
        raise ValueError(f"{path}: snapshot cardinality mismatch")
    previous_parameters = parameter_sha_map(fixture["initial_parameters"])
    previous_state = None
    for step, snapshot in enumerate(snapshots, 1):
        validate_snapshot(
            snapshot, expected_name, variant, rank, step, spec["parameter_count"],
            expected_state, group_kinds, f"{path}.snapshots[{step - 1}]",
        )
        current_parameters = parameter_sha_map(snapshot["parameters"])
        should_change = expected_name == "tiny_fixed" or step < 4
        if should_change == (current_parameters == previous_parameters):
            raise ValueError(f"{path}: unexpected no-op/change at step {step}")
        if previous_state is not None and snapshot["state"] == previous_state:
            raise ValueError(f"{path}: optimizer state did not update at step {step}")
        previous_parameters = current_parameters
        previous_state = snapshot["state"]


def validate_optimizer_worker(report: Any, variant: str, rank: int) -> None:
    path = f"{variant}.rank{rank}"
    require_exact_keys(report, WORKER_RECORD_KEYS, path)
    if (report["schema"] != WORKER_SCHEMA or report["worker"] != "optimizer"
            or report["variant"] != variant or report["rank"] != rank
            or report["world_size"] != WORLD_SIZE or report["passed"] is not True):
        raise ValueError(f"{path}: worker identity/schema/pass mismatch")
    if not isinstance(report["compile_cache"], str) or not Path(report["compile_cache"]).is_absolute():
        raise ValueError(f"{path}: compile cache identity missing/non-absolute")
    fixtures = report["fixtures"]
    if not isinstance(fixtures, list) or len(fixtures) != len(FIXTURE_SPECS):
        raise ValueError(f"{path}: fixture cardinality mismatch")
    for index, fixture in enumerate(fixtures):
        validate_fixture(fixture, variant, rank, index)


def compare_optimizer_results(result_dir: Path) -> dict[str, Any]:
    reports: dict[str, list[dict[str, Any]]] = {}
    failures = []
    for variant in ("raw", "source", "direct"):
        reports[variant] = []
        for rank in range(WORLD_SIZE):
            path = result_dir / f"optimizer-{variant}-rank{rank}.json"
            try:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("worker report missing/unsafe")
                report = json.loads(path.read_text())
                reports[variant].append(report)
                validate_optimizer_worker(report, variant, rank)
            except Exception as error:
                failures.append(f"{variant} rank {rank}: {type(error).__name__}: {error}")
    if failures:
        return {"passed": False, "failures": failures, "variants": {}, "worker_reports": reports}
    caches = {variant: {report["compile_cache"] for report in ranks} for variant, ranks in reports.items()}
    if any(len(paths) != 1 for paths in caches.values()) or len({next(iter(paths)) for paths in caches.values()}) != 3:
        failures.append("optimizer variants do not have three distinct rank-shared compile caches")
    for rank in range(1, WORLD_SIZE):
        if reports["raw"][rank]["fixtures"] != reports["raw"][0]["fixtures"]:
            failures.append(f"raw rank {rank} snapshots differ from raw rank 0")
    for rank in range(WORLD_SIZE):
        source = reports["source"][rank]
        direct = reports["direct"][rank]
        if source["fixtures"] != direct["fixtures"]:
            failures.append(f"source/direct rank {rank} snapshots differ")
    for fixture_index, fixture_name in enumerate(FIXTURE_SPECS):
        raw = reports["raw"][0]["fixtures"][fixture_index]
        for rank in range(WORLD_SIZE):
            direct_fixture = reports["direct"][rank]["fixtures"][fixture_index]
            if direct_fixture["initial_parameters"] != raw["initial_parameters"]:
                failures.append(f"{fixture_name}: direct rank {rank} initial parameters differ from raw")
            if direct_fixture["plan"] != raw["plan"]:
                failures.append(f"{fixture_name}: direct rank {rank} ownership plan differs from raw")
        for step_index in range(FIXTURE_SPECS[fixture_name]["steps"]):
            raw_step = raw["snapshots"][step_index]
            merged_state = {}
            for rank in range(WORLD_SIZE):
                direct_step = reports["direct"][rank]["fixtures"][fixture_index]["snapshots"][step_index]
                if direct_step["parameters"] != raw_step["parameters"]:
                    failures.append(f"{fixture_name} step {step_index + 1}: direct rank {rank} parameters differ from raw")
                if direct_step["group_scalars"] != raw_step["group_scalars"]:
                    failures.append(f"{fixture_name} step {step_index + 1}: direct rank {rank} scalars differ from raw")
                if direct_step["cross_rank_parameter_sha256"] != raw_step["cross_rank_parameter_sha256"]:
                    failures.append(f"{fixture_name} step {step_index + 1}: direct rank {rank} cross-rank digests differ from raw")
                overlap = set(merged_state) & set(direct_step["state"])
                if overlap:
                    failures.append(f"{fixture_name} step {step_index + 1}: duplicated owned state {sorted(overlap)[:3]}")
                merged_state.update(direct_step["state"])
            if merged_state != raw_step["state"]:
                failures.append(f"{fixture_name} step {step_index + 1}: merged direct state differs from raw")
    return {
        "passed": not failures,
        "failures": failures,
        "variants": {variant: {"rank_count": len(ranks), "all_workers_passed": all(r["passed"] for r in ranks)}
                     for variant, ranks in reports.items()},
        "worker_reports": reports,
    }


def expected_outer_environment(args: argparse.Namespace) -> dict[str, str]:
    return {**STATIC_ENVIRONMENT,
        "TRAINIUM_DISCRIMINATOR_LABEL": args.run_label,
        "TRAINIUM_DISCRIMINATOR_EXPECTED_COMMIT": args.expected_commit,
        "TRAINIUM_DISCRIMINATOR_AUTHORIZED_HEAD": args.authorized_head,
        "TRAINIUM_DISCRIMINATOR_AUTHORIZATION_SHA256": args.authorization_sha256,
    }


def validate_full_environment(args: argparse.Namespace) -> None:
    exact = expected_outer_environment(args)
    for key, expected in exact.items():
        if os.environ.get(key) != expected:
            raise ValueError(f"environment mismatch for {key}")
    if args.expected_hostname != EXPECTED_HOSTNAME or socket.gethostname() != EXPECTED_HOSTNAME:
        raise ValueError("hostname differs from external authorization")
    if args.expected_commit != args.authorized_head or git("rev-parse", "HEAD") != args.expected_commit:
        raise ValueError("expected commit, authorized HEAD, and checkout must be identical")
    if not re.fullmatch(r"fs-[a-z0-9][a-z0-9-]{15,160}", args.run_label):
        raise ValueError("run label is not unique/canonical")
    if str(Path(sys.executable).resolve()) != str(Path(NATIVE_PYTHON).resolve()):
        raise ValueError("harness is not running under the pinned native Python")
    if os.environ.get("TRAINIUM_FRESH_START_RUN_ID") != args.run_label:
        raise ValueError("fresh_start_guard run identity is missing/mismatched")
    if args.report.resolve() != (ROOT / "research/run-logs" / f"{args.run_label}-result.json").resolve():
        raise ValueError("report path is not the unique label-derived run-log target")


def exact_launch_request(args: argparse.Namespace, authorization: dict[str, Any]) -> dict[str, Any]:
    environment = expected_outer_environment(args)
    command = [
        *OUTER_TIMEOUT,
        NATIVE_PYTHON, str(HARNESS),
        "--authorization-request", str(REQUEST),
        "--authorized-head", args.authorized_head,
        "--authorization-sha256", args.authorization_sha256,
        "--expected-commit", args.expected_commit,
        "--run-label", args.run_label,
        "--report", str(args.report.resolve()),
        "--expected-hostname", args.expected_hostname,
    ]
    return {
        "host": EXPECTED_HOST,
        "remote": EXPECTED_REMOTE,
        "expected_commit": args.expected_commit,
        "authorized_head": args.authorized_head,
        "label": args.run_label,
        "purpose": EXPECTED_PURPOSE,
        "environment": environment,
        "command": command,
        "fingerprint_paths": [*PROTECTED_PATHS, rel(REQUEST)],
        "authorization_request": rel(REQUEST),
        "authorization_sha256": args.authorization_sha256,
    }


def launcher_fingerprint_paths() -> tuple[str, ...]:
    return tuple(dict.fromkeys((*LAUNCHER_DEFAULT_FINGERPRINT_PATHS, *PROTECTED_PATHS, rel(REQUEST))))


def validate_launcher_payload(
    payload: dict[str, Any], authorized_request: dict[str, Any],
    required_hashes: dict[str, str], args: argparse.Namespace, suffix: str,
) -> None:
    augmented = {"started_at", "hostname", "git_commit", "source_dirty",
                 "preflight_free_bytes", "source_sha256"}
    if set(payload) != set(authorized_request) | augmented:
        raise ValueError(f"{suffix} manifest key set differs from exact schema")
    if {key: payload[key] for key in authorized_request} != authorized_request:
        raise ValueError(f"{suffix} parsed request/argv/environment is not exact")
    if tuple(payload["source_sha256"]) != launcher_fingerprint_paths():
        raise ValueError(f"{suffix} fingerprint path union/order differs from launcher contract")
    if (payload.get("hostname") != EXPECTED_HOSTNAME
            or payload.get("git_commit") != args.expected_commit
            or payload.get("remote") != EXPECTED_REMOTE
            or payload.get("source_dirty") is not False
            or payload.get("source_sha256") != required_hashes):
        raise ValueError(f"{suffix} checkout/fingerprint identity mismatch")
    if not isinstance(payload.get("started_at"), str) or not isinstance(
            payload.get("preflight_free_bytes"), int):
        raise ValueError(f"{suffix} launcher augmentation type mismatch")


def validate_outer_receipts(args: argparse.Namespace, authorization: dict[str, Any]) -> dict[str, Any]:
    run_logs = ROOT / "research/run-logs"
    authorized_request = exact_launch_request(args, authorization)
    hash_values = dict(authorization["expected_sha256"])
    hash_values[rel(REQUEST)] = authorization["request_sha256"]
    required_hashes = {path: hash_values[path] for path in launcher_fingerprint_paths()}
    if tuple(required_hashes) != launcher_fingerprint_paths():
        raise ValueError("authorization hash map differs from exact launcher fingerprint union")
    receipts = {}
    for suffix in ("launch.json", "job.json"):
        path = run_logs / f"{args.run_label}.{suffix}"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required fresh-start receipt missing/unsafe: {path}")
        payload = json.loads(path.read_text())
        validate_launcher_payload(payload, authorized_request, required_hashes, args, suffix)
        receipts[suffix] = {"path": str(path), "sha256": sha256(path)}
    preflight = run_logs / f"{args.run_label}.preflight.json"
    if preflight.is_symlink() or not preflight.is_file():
        raise ValueError("required preflight receipt missing/unsafe")
    payload = json.loads(preflight.read_text())
    if (payload.get("label") != args.run_label or payload.get("remote") != EXPECTED_REMOTE
            or payload.get("hostname") != EXPECTED_HOSTNAME or payload.get("passed") is not True):
        raise ValueError("preflight receipt does not prove the exact passing run")
    receipts["preflight.json"] = {"path": str(preflight), "sha256": sha256(preflight)}
    return receipts


def orchestrator_failure_report(
    authorization: dict[str, Any], result_dir: Path, processes: list[dict[str, Any]],
    outer_receipts: dict[str, Any], error: Exception,
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "mode": "neuron",
        "passed": False,
        "authorization": authorization,
        "source_sha256": sha256(SOURCE),
        "direct_sha256": sha256(DIRECT),
        "failure": {"type": type(error).__name__, "message": str(error)},
        "processes": list(processes),
        "outer_receipts": dict(outer_receipts),
        "result_directory": str(result_dir),
    }


def run_orchestrator(args: argparse.Namespace, authorization: dict[str, Any]) -> dict[str, Any]:
    result_dir = Path(f"/tmp/{args.run_label}-discriminator-results")
    run_logs = ROOT / "research/run-logs"
    processes: list[dict[str, Any]] = []
    outer_receipts: dict[str, Any] = {}

    try:
        validate_full_environment(args)
        outer_receipts = validate_outer_receipts(args, authorization)
        if result_dir.exists() or result_dir.is_symlink():
            raise FileExistsError(f"fresh result directory already exists: {result_dir}")
        result_dir.mkdir(mode=0o700)
        nonce = hashlib.sha256(f"{args.run_label}:{args.authorization_sha256}".encode()).hexdigest()
        common_env = os.environ.copy()
        common_env["TRAINIUM_DISCRIMINATOR_PARENT_NONCE"] = nonce
        prepare_sha = authorization["expected_sha256"].get("prepare.py")
        if prepare_sha != PREPARE_SHA256:
            raise ValueError("authenticated protected map does not authorize pinned prepare.py")
        common_env["TRAINIUM_DISCRIMINATOR_PREPARE_SHA256"] = prepare_sha
        python = NATIVE_PYTHON
        for variant in ("source", "direct"):
            name = f"ce-{variant}"
            cache = Path(f"/tmp/{args.run_label}-{name}-cache")
            if cache.exists() or cache.is_symlink():
                raise FileExistsError(f"compile cache is not fresh: {cache}")
            cache.mkdir(mode=0o700)
            env, environment_identity = prepare_ce_environment(common_env, cache)
            command = [python, str(HARNESS), "--worker", "ce", "--variant", variant,
                       "--result-dir", str(result_dir), "--parent-nonce", nonce]
            artifact_paths = {"worker_json": result_dir / f"ce-{variant}.json"}
            if variant == "source":
                artifact_paths["source_tensors"] = result_dir / "ce-source.pt"
            index = len(processes)
            process = run_child(
                name, command, env, environment_identity, 1800,
                run_logs / f"{args.run_label}.child-{index:02d}-{name}.log",
                run_logs / f"{args.run_label}.child-{index:02d}-{name}.json",
                cache, artifact_paths,
            )
            processes.append(process)
            if not process["passed"]:
                raise RuntimeError(f"child failed or emitted recompile-limit warning: {name}")
        ce_source = json.loads((result_dir / "ce-source.json").read_text())
        ce_direct = json.loads((result_dir / "ce-direct.json").read_text())
        validate_ce_report(ce_source, "source")
        validate_ce_report(ce_direct, "direct")
        ce_records_exact = all(
            source_case["loss"] == direct_case["loss"]
            and source_case["gradient"] == direct_case["gradient"]
            for source_case, direct_case in zip(ce_source["cases"], ce_direct["cases"])
        )
        ce_passed = (
            ce_source["passed"] and ce_direct["passed"]
            and ce_source["calls"] == ce_direct["calls"]
            and ce_source["compile_cache"] != ce_direct["compile_cache"]
            and ce_records_exact
            and len(ce_direct["cases"]) == len(ROWS) * REPETITIONS
            and all(case["source_direct"]["passed"] for case in ce_direct["cases"])
        )
        for variant in ("raw", "source", "direct"):
            name = f"optimizer-{variant}"
            cache = Path(f"/tmp/{args.run_label}-{name}-cache")
            if cache.exists() or cache.is_symlink():
                raise FileExistsError(f"compile cache is not fresh: {cache}")
            cache.mkdir(mode=0o700)
            env, environment_identity = prepare_optimizer_environment(common_env, cache)
            command = [TORCHRUN, "--standalone", "--nproc_per_node=4", str(HARNESS),
                       "--worker", "optimizer", "--variant", variant,
                       "--result-dir", str(result_dir), "--parent-nonce", nonce]
            artifact_paths = {
                f"rank{rank}_json": result_dir / f"optimizer-{variant}-rank{rank}.json"
                for rank in range(WORLD_SIZE)
            }
            index = len(processes)
            process = run_child(
                name, command, env, environment_identity, 2400,
                run_logs / f"{args.run_label}.child-{index:02d}-{name}.log",
                run_logs / f"{args.run_label}.child-{index:02d}-{name}.json",
                cache, artifact_paths,
            )
            processes.append(process)
            if not process["passed"]:
                raise RuntimeError(f"child failed or emitted recompile-limit warning: {name}")
        optimizer = compare_optimizer_results(result_dir)
        return {
            "schema": REPORT_SCHEMA,
            "mode": "neuron",
            "passed": ce_passed and optimizer["passed"] and all(p["passed"] for p in processes),
            "authorization": authorization,
            "source_sha256": sha256(SOURCE),
            "direct_sha256": sha256(DIRECT),
            "ce": {
                "passed": ce_passed,
                "source_cases": ce_source["cases"],
                "direct_cases": ce_direct["cases"],
                "source_direct_exact": [case["source_direct"] for case in ce_direct["cases"]],
                "source_calls": ce_source["calls"],
                "direct_calls": ce_direct["calls"],
                "pytorch_diagnostic_numerically_non_gating": [
                    case["pytorch_diagnostic_numerically_non_gating"] for case in ce_direct["cases"]
                ],
            },
            "optimizer": optimizer,
            "processes": processes,
            "outer_receipts": outer_receipts,
            "result_directory": str(result_dir),
        }
    except Exception as error:
        return orchestrator_failure_report(authorization, result_dir, processes, outer_receipts, error)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", choices=("ce", "optimizer"))
    p.add_argument("--variant", choices=("raw", "source", "direct"))
    p.add_argument("--result-dir", type=Path)
    p.add_argument("--parent-nonce")
    p.add_argument("--static-only", action="store_true")
    p.add_argument("--authorization-request", type=Path)
    p.add_argument("--authorized-head")
    p.add_argument("--authorization-sha256")
    p.add_argument("--run-label")
    p.add_argument("--expected-commit")
    p.add_argument("--report", type=Path)
    p.add_argument("--expected-hostname")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.worker:
        if not args.variant or not args.result_dir or not args.parent_nonce:
            raise ValueError("worker requires variant, result-dir, and parent-nonce")
        if args.worker == "ce" and args.variant == "raw":
            raise ValueError("CE has only source/direct variants")
        return run_ce_worker(args) if args.worker == "ce" else run_optimizer_worker(args)
    required = (args.authorization_request, args.authorized_head, args.authorization_sha256, args.report)
    if any(value is None for value in required):
        raise ValueError("top level requires authorization request/head/SHA and report")
    if args.report.parent.resolve() != (ROOT / "research/run-logs").resolve():
        raise ValueError("report must be a direct child of research/run-logs")
    if args.report.exists() or args.report.is_symlink():
        raise FileExistsError("report exists; overwrite is forbidden")
    static = static_source_checks()
    authorization = authenticate(args, full=not args.static_only)
    if args.static_only:
        report = {
            "schema": REPORT_SCHEMA,
            "mode": "static-only",
            "passed": static["passed"],
            "authorization": authorization,
            "static": static,
            "hardware_executed": False,
        }
    else:
        if not args.run_label or not args.expected_hostname or not args.expected_commit:
            raise ValueError("full mode requires run-label, expected-hostname, and expected-commit")
        try:
            report = run_orchestrator(args, authorization)
        except Exception as error:
            report = orchestrator_failure_report(
                authorization, Path(f"/tmp/{args.run_label}-discriminator-results"), [], {}, error,
            )
        report["static"] = static
        report["passed"] = report["passed"] and static["passed"]
    attach_prewrite_process_revalidation(report)
    exclusive_json(args.report, report)
    print(json.dumps({"report": str(args.report), "passed": report["passed"], "mode": report["mode"]}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
