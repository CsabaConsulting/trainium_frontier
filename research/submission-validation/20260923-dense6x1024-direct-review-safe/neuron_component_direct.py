#!/usr/bin/env python3
"""Fail-closed Neuron component gate for the direct-review dense6x1024 package.

The full mode must run under four-rank LNC2 ``torchrun`` on real Neuron.  It exercises the
candidate's direct fused-MLP autograd path, both adaptive CE grids, and explicit optimizer
ownership.  It does not load data, train the model, score a checkpoint, or make a quality claim.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HARNESS_PATH = Path(__file__).resolve()
RUNBOOK_PATH = HARNESS_PATH.with_name("neuron-component-runbook.md")
PROPOSAL_PATH = HARNESS_PATH.with_name("proposal.json")
AUTHORIZATION_PATH = HARNESS_PATH.with_name("neuron-component-authorization-request.json")
CANONICAL_TRAIN = (ROOT / "submissions/dense6x1024_direct_review/train.py").resolve()
RUN_LOGS = ROOT / "research/run-logs"
TRUST_ROOT_VERIFIER_COMMIT = "c6d521ab068a4392da1d67bacc55e081335b42bf"
TRUST_ROOT_REPORT = HARNESS_PATH.with_name(
    "independent-neuron-component-authorization-review.json"
)
TRUST_ROOT_REPORT_SHA256 = "860b844a4a8e7a600df6d96ef4a42abdf139acb5faeed883c23ceb7edaff4a86"
EXPECTED_TRAIN_SHA256 = "c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537"
EXPECTED_WORLD_SIZE = 4
EXPECTED_LNC = "2"
EXPECTED_SEED = 42
EXPECTED_HOST_THREADS = 4
EXPECTED_CC_FLAGS = "--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
EXPECTED_HARDWARE_HOSTNAME = "ip-172-31-32-253"
FULL_LABEL_PATTERN = r"fs-20260923-dense6x1024-direct-neuron-component-[0-9]{8}T[0-9]{12}Z"
AUTHORIZATION_SCHEMA = "dense6x1024-direct-neuron-component-authorization-v1"
EXPECTED_HOST = "trn22"
EXPECTED_REMOTE = "/home/ubuntu/trainium_freshstart_submission_validation_20260923_direct_component"
EXPECTED_PURPOSE = (
    "Standalone authorized direct-wiring Neuron component gate: repeated fused MLP, repeated "
    "adaptive CE one/two lane, and two-step explicit OwnedOptimizer semantics. No training, "
    "submission, or scientific verdict."
)
EXPECTED_STATIC_ENVIRONMENT = {
    "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
    "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv",
    "NEURON_LOGICAL_NC_CONFIG": "2",
    "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "PYTHONUNBUFFERED": "1",
    "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "1",
}
EXPECTED_TIMEOUT_ARGV = [
    "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "1200s",
]
EXPECTED_TORCHRUN_ARGV_PREFIX = [
    "/home/ubuntu/workspace/native_venv/bin/torchrun", "--standalone", "--nproc_per_node=4",
]
VOCAB_SIZE = 8192
MLP_SHAPE = (32, 1024, 1024)
MLP_HIDDEN = 4096

# Declared before execution.  Candidate and independent reference were exact in the preserved
# source-component evidence.  The bounds below allow BF16 matmul rounding for MLP while requiring
# close FP32 agreement for CE/optimizer.  Norm gates independently reject zero/attenuated output.
TOLERANCES = {
    "mlp_output": {"atol": 0.015625, "rtol": 0.01, "max_error_norm_ratio": 0.01,
                   "min_reference_norm": 1.0, "min_candidate_norm_ratio": 0.98,
                   "max_candidate_norm_ratio": 1.02},
    "mlp_input_gradient": {"atol": 0.00025, "rtol": 0.02, "max_error_norm_ratio": 0.02,
                           "min_reference_norm": 1e-6, "min_candidate_norm_ratio": 0.90,
                           "max_candidate_norm_ratio": 1.10},
    "mlp_up_gradient": {"atol": 0.002, "rtol": 0.02, "max_error_norm_ratio": 0.02,
                        "min_reference_norm": 1e-5, "min_candidate_norm_ratio": 0.90,
                        "max_candidate_norm_ratio": 1.10},
    "mlp_down_gradient": {"atol": 0.002, "rtol": 0.02, "max_error_norm_ratio": 0.02,
                          "min_reference_norm": 1e-5, "min_candidate_norm_ratio": 0.90,
                          "max_candidate_norm_ratio": 1.10},
    "ce_loss": {"atol": 2e-5, "rtol": 2e-5, "max_error_norm_ratio": 1e-4,
                "min_reference_norm": 1.0, "min_candidate_norm_ratio": 0.999,
                "max_candidate_norm_ratio": 1.001},
    "ce_gradient": {"atol": 2e-6, "rtol": 2e-4, "max_error_norm_ratio": 5e-4,
                    "min_reference_norm": 1e-5, "min_candidate_norm_ratio": 0.99,
                    "max_candidate_norm_ratio": 1.01},
    "optimizer_parameter": {"atol": 1e-7, "rtol": 1e-6, "max_error_norm_ratio": 1e-6,
                            "min_reference_norm": 1e-3, "min_candidate_norm_ratio": 0.99999,
                            "max_candidate_norm_ratio": 1.00001},
    "optimizer_update": {"atol": 1e-7, "rtol": 1e-5, "max_error_norm_ratio": 1e-4,
                         "min_reference_norm": 1e-8, "min_candidate_norm_ratio": 0.99,
                         "max_candidate_norm_ratio": 1.01},
}

EXPECTED_HASH_PATHS = (
    "train.py",
    "prepare.py",
    "research/run_experiment.py",
    "research/fresh_start_host.py",
    "research/fresh_start_guard.py",
    "research/fresh_start_launch.py",
    "research/fresh_start_backup.py",
    "research/fresh_start_sync.py",
    "submissions/dense6x1024_direct_review/train.py",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron_component_direct.py",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron-component-runbook.md",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/proposal.json",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/independent-neuron-component-authorization-review.json",
)
AUTHORIZATION_RELATIVE_PATH = (
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/"
    "neuron-component-authorization-request.json"
)
FINGERPRINT_PATHS = (*EXPECTED_HASH_PATHS, AUTHORIZATION_RELATIVE_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_report_path(label: str) -> Path:
    return RUN_LOGS / f"{label}.component.json"


def validate_report_path(path: Path, label: str) -> Path:
    if not re.fullmatch(r"fs-[A-Za-z0-9_.-]+", label):
        raise ValueError("run label must be unique-looking and start with fs-")
    absolute = path.absolute()
    if absolute.parent != RUN_LOGS or absolute.name != f"{label}.component.json":
        raise ValueError(f"report must be exactly {expected_report_path(label)}")
    if not RUN_LOGS.is_dir() or RUN_LOGS.is_symlink() or RUN_LOGS.resolve() != RUN_LOGS:
        raise RuntimeError("approved run-log parent must be an existing real directory")
    if os.path.lexists(absolute):
        raise FileExistsError(f"refusing to overwrite or follow existing report: {absolute}")
    return absolute


def write_report(path: Path, label: str, report: dict[str, Any]) -> None:
    path = validate_report_path(path, label)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def functions_named(tree: ast.AST, name: str) -> list[ast.FunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]


def classes_named(tree: ast.AST, name: str) -> list[ast.ClassDef]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name]


def static_wiring_checks(train_py: Path) -> dict[str, Any]:
    text = train_py.read_text()
    tree = ast.parse(text)
    mlp = classes_named(tree, "MLP")
    ce = classes_named(tree, "_SoftcapCrossEntropyNKI")
    main = functions_named(tree, "main")
    if len(mlp) != 1 or len(ce) != 1 or len(main) != 1:
        raise AssertionError("expected one MLP, _SoftcapCrossEntropyNKI, and main definition")
    mlp_text = ast.unparse(mlp[0])
    ce_text = ast.unparse(ce[0])
    main_text = ast.unparse(main[0])
    trigger_hits = [
        trigger
        for trigger in ("_submission_", "_install_submission_training", "MethodType", "SUBMISSION_")
        if trigger in text
    ]
    checks = {
        "candidate_sha256": sha256(train_py),
        "candidate_hash_pinned": sha256(train_py) == EXPECTED_TRAIN_SHA256,
        "mlp_calls_direct_fused_autograd": "_FusedMLP.apply" in mlp_text,
        "ce_selects_lanes": ce_text.count("selected_lanes") == 2,
        "ce_calls_wrapped_forward": "_softcap_ce_fwd_native[lanes]" in ce_text,
        "ce_calls_wrapped_backward": "_softcap_ce_bwd_native[lanes]" in ce_text,
        "main_constructs_explicit_owner": "owned_optimizer = OwnedOptimizer(optimizer)" in main_text,
        "main_steps_explicit_owner": "owned_optimizer.step(update_adamw=True)" in main_text,
        "review_trigger_hits": trigger_hits,
        "runtime_forward_assignment_absent": ".forward =" not in text,
        "runtime_step_assignment_absent": ".step =" not in text,
        "runtime_ce_global_assignment_absent": (
            "train_module._softcap_ce_fwd =" not in text
            and "train_module._softcap_ce_bwd =" not in text
        ),
    }
    checks["passed"] = all(
        value is True or value == [] or key in {"candidate_sha256", "passed"}
        for key, value in checks.items()
    )
    return checks


def import_candidate(train_py: Path):
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("dense6x1024_direct_neuron_component", train_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {train_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synchronize(torch, device) -> None:
    if not hasattr(torch, "neuron"):
        raise RuntimeError("torch.neuron is unavailable")
    torch.neuron.synchronize()


def graph_node_names(tensor) -> list[str]:
    pending = [tensor.grad_fn]
    seen: set[int] = set()
    names: list[str] = []
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        names.append(type(node).__name__)
        pending.extend(parent for parent, _index in getattr(node, "next_functions", ()))
    return names


def tensor_metrics(torch, actual, reference, tolerance: dict[str, float]) -> dict[str, Any]:
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        return {
            "passed": False,
            "shape_equal": actual.shape == reference.shape,
            "dtype_equal": actual.dtype == reference.dtype,
            "actual_shape": list(actual.shape),
            "reference_shape": list(reference.shape),
            "actual_dtype": str(actual.dtype),
            "reference_dtype": str(reference.dtype),
            **tolerance,
        }
    left = actual.detach().float()
    right = reference.detach().float()
    finite = bool((torch.isfinite(left).all() & torch.isfinite(right).all()).cpu().item())
    signed_difference = left - right
    difference = signed_difference.abs()
    maximum = float(difference.max().cpu().item())
    mean = float(difference.mean().cpu().item())
    candidate_norm = float(torch.linalg.vector_norm(left.reshape(-1)).cpu().item())
    reference_norm = float(torch.linalg.vector_norm(right.reshape(-1)).cpu().item())
    error_norm = float(torch.linalg.vector_norm(signed_difference.reshape(-1)).cpu().item())
    error_norm_ratio = error_norm / reference_norm if reference_norm > 0.0 else math.inf
    candidate_norm_ratio = candidate_norm / reference_norm if reference_norm > 0.0 else math.inf
    limit = tolerance["atol"] + tolerance["rtol"] * right.abs()
    within = bool((difference <= limit).all().cpu().item())
    norm_gate = (
        math.isfinite(candidate_norm)
        and math.isfinite(reference_norm)
        and math.isfinite(error_norm_ratio)
        and reference_norm >= tolerance["min_reference_norm"]
        and candidate_norm > 0.0
        and tolerance["min_candidate_norm_ratio"]
        <= candidate_norm_ratio
        <= tolerance["max_candidate_norm_ratio"]
        and error_norm_ratio <= tolerance["max_error_norm_ratio"]
    )
    return {
        "passed": finite and within and norm_gate,
        "finite": finite,
        "shape_equal": True,
        "dtype_equal": True,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_abs": maximum,
        "mean_abs": mean,
        "candidate_l2_norm": candidate_norm,
        "reference_l2_norm": reference_norm,
        "error_l2_norm": error_norm,
        "error_norm_ratio": error_norm_ratio,
        "candidate_norm_ratio": candidate_norm_ratio,
        "elementwise_within_tolerance": within,
        "norm_gate_passed": norm_gate,
        **tolerance,
    }


def host_metric_passes(actual: list[float], reference: list[float], tolerance: dict[str, float]) -> bool:
    if len(actual) != len(reference):
        return False
    finite = all(math.isfinite(item) for item in actual + reference)
    candidate_norm = math.sqrt(sum(item * item for item in actual))
    reference_norm = math.sqrt(sum(item * item for item in reference))
    error_norm = math.sqrt(sum((left - right) ** 2 for left, right in zip(actual, reference)))
    if reference_norm <= 0.0:
        return False
    candidate_ratio = candidate_norm / reference_norm
    error_ratio = error_norm / reference_norm
    elementwise = all(
        abs(left - right) <= tolerance["atol"] + tolerance["rtol"] * abs(right)
        for left, right in zip(actual, reference)
    )
    return bool(
        finite
        and elementwise
        and reference_norm >= tolerance["min_reference_norm"]
        and candidate_norm > 0.0
        and tolerance["min_candidate_norm_ratio"] <= candidate_ratio
        <= tolerance["max_candidate_norm_ratio"]
        and error_ratio <= tolerance["max_error_norm_ratio"]
    )


def metric_adversarial_self_test() -> dict[str, Any]:
    """Prove each declared metric rejects degenerate and materially wrong tensors."""
    cases: dict[str, Any] = {}
    for index, (name, tolerance) in enumerate(TOLERANCES.items()):
        scale = max(1.0, tolerance["min_reference_norm"])
        reference = [scale * (1.0 + ((position * 17 + index * 11) % 31) / 17.0) for position in range(257)]
        corrupt = list(reference)
        corrupt[17] += max(10.0, 1000.0 * tolerance["atol"])
        variants = {
            "identical": list(reference),
            "zero": [0.0] * len(reference),
            "scaled_half": [item * 0.5 for item in reference],
            "permuted": list(reversed(reference)),
            "single_corruption": corrupt,
            "nonfinite": list(reference),
        }
        variants["nonfinite"][3] = float("nan")
        results = {
            variant: host_metric_passes(value, reference, tolerance)
            for variant, value in variants.items()
        }
        passed = results["identical"] and not any(
            results[key] for key in results if key != "identical"
        )
        cases[name] = {"passed": passed, "results": results}
    return {"passed": all(case["passed"] for case in cases.values()), "cases": cases}


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def git_output(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def require_clean_fingerprint_scope() -> dict[str, Any]:
    for relative in FINGERPRINT_PATHS:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    modified = git_output(["diff", "--name-only", "HEAD", "--", *FINGERPRINT_PATHS])
    staged = git_output(["diff", "--cached", "--name-only", "HEAD", "--", *FINGERPRINT_PATHS])
    untracked = git_output(
        ["ls-files", "--others", "--exclude-standard", "--", *FINGERPRINT_PATHS]
    )
    if modified or staged or untracked:
        raise RuntimeError(
            "fingerprint scope is dirty: "
            + json.dumps({"modified": modified, "staged": staged, "untracked": untracked})
        )
    return {"tracked": list(FINGERPRINT_PATHS), "modified": [], "staged": [], "untracked": []}


def load_authorization(args, full: bool) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", args.authorized_head):
        raise ValueError("--authorized-head must be a literal full lowercase commit SHA")
    if git_head() != args.authorized_head:
        raise ValueError("checkout HEAD differs from the externally authorized exact HEAD")
    if args.expected_commit != args.authorized_head:
        raise ValueError("--expected-commit must equal the externally authorized exact HEAD")
    if not re.fullmatch(r"[0-9a-f]{64}", args.authorization_sha256):
        raise ValueError("--authorization-sha256 must be a literal lowercase SHA-256")
    requested = args.authorization_request.absolute()
    if requested != AUTHORIZATION_PATH or requested.is_symlink() or not requested.is_file():
        raise ValueError(f"authorization request must be exactly {AUTHORIZATION_PATH}")
    clean = require_clean_fingerprint_scope()
    payload = json.loads(requested.read_text())
    if payload.get("schema") != AUTHORIZATION_SCHEMA:
        raise ValueError("authorization request schema mismatch")
    if payload.get("trust_root_verifier_commit") != TRUST_ROOT_VERIFIER_COMMIT:
        raise ValueError("authorization trust-root commit mismatch")
    if (
        sha256(TRUST_ROOT_REPORT) != TRUST_ROOT_REPORT_SHA256
        or payload.get("trust_root_report_sha256") != TRUST_ROOT_REPORT_SHA256
    ):
        raise ValueError("authorization trust-root report mismatch")
    expected_hashes = payload.get("expected_sha256")
    if not isinstance(expected_hashes, dict) or tuple(expected_hashes) != EXPECTED_HASH_PATHS:
        raise ValueError("authorization expected-hash path set/order mismatch")
    actual_hashes = {relative: sha256(ROOT / relative) for relative in EXPECTED_HASH_PATHS}
    mismatches = {
        relative: {"expected": expected_hashes.get(relative), "actual": actual_hashes[relative]}
        for relative in EXPECTED_HASH_PATHS
        if expected_hashes.get(relative) != actual_hashes[relative]
    }
    if mismatches:
        raise RuntimeError(f"authorized source hash mismatch: {json.dumps(mismatches, sort_keys=True)}")
    authorization_sha = sha256(requested)
    if args.authorization_sha256 != authorization_sha:
        raise ValueError("request bytes differ from the externally authorized SHA-256")
    template_state = payload.get("template_state")
    if template_state not in {
        "pending_external_verifier_literals", "finalized_by_external_verifier"
    }:
        raise ValueError("authorization template state mismatch")
    if full and template_state != "finalized_by_external_verifier":
        raise RuntimeError("pending authorization template cannot launch hardware")
    request_contract = payload.get("request_contract")
    expected_contract = {
        "host": EXPECTED_HOST,
        "remote": EXPECTED_REMOTE,
        "expected_hostname": EXPECTED_HARDWARE_HOSTNAME,
        "purpose": EXPECTED_PURPOSE,
        "environment_static": EXPECTED_STATIC_ENVIRONMENT,
        "timeout_argv": EXPECTED_TIMEOUT_ARGV,
        "torchrun_argv_prefix": EXPECTED_TORCHRUN_ARGV_PREFIX,
    }
    if request_contract != expected_contract:
        raise ValueError("authorization request contract differs from hard-coded harness constants")
    return {
        "path": str(requested),
        "sha256": authorization_sha,
        "schema": payload["schema"],
        "trust_root_verifier_commit": payload["trust_root_verifier_commit"],
        "expected_sha256": expected_hashes,
        "authorized_head": args.authorized_head,
        "template_state": template_state,
        "request_contract": request_contract,
        "clean_scope": clean,
    }


def identity_manifest(args, train_py: Path, authorization: dict[str, Any]) -> dict[str, Any]:
    environment_names = (
        "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS", "NEURON_COMPILE_CACHE_DIR",
        "WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK", "OMP_NUM_THREADS",
        "MKL_NUM_THREADS", "PYTHONUNBUFFERED", "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION",
        "TRAINIUM_COMPONENT_RUN_LABEL", "TRAINIUM_COMPONENT_EXPECTED_COMMIT",
        "TRAINIUM_COMPONENT_EXPECTED_HOSTNAME", "TRAINIUM_COMPONENT_AUTHORIZED_HEAD",
        "TRAINIUM_COMPONENT_AUTHORIZATION_SHA256",
        "TRAINIUM_FRESH_START_RUN_ID",
    )
    return {
        "argv": sys.argv,
        "run_label": args.run_label,
        "git_commit": args.expected_commit,
        "hostname": args.expected_hostname,
        "seed": args.seed,
        "host_threads": args.host_threads,
        "candidate": {"path": str(train_py), "sha256": sha256(train_py)},
        "files": {
            "harness": {"path": str(HARNESS_PATH), "sha256": sha256(HARNESS_PATH)},
            "runbook": {"path": str(RUNBOOK_PATH), "sha256": sha256(RUNBOOK_PATH)},
            "proposal": {"path": str(PROPOSAL_PATH), "sha256": sha256(PROPOSAL_PATH)},
            "authorization_request": {
                "path": str(AUTHORIZATION_PATH), "sha256": authorization["sha256"]
            },
        },
        "authorization": authorization,
        "environment": {name: os.environ.get(name) for name in environment_names},
    }


def validate_identity(args, train_py: Path, full: bool) -> dict[str, Any]:
    authorization = load_authorization(args, full)
    if args.seed != EXPECTED_SEED:
        raise ValueError(f"seed must be exactly {EXPECTED_SEED}")
    if args.host_threads != EXPECTED_HOST_THREADS:
        raise ValueError(f"host threads must be exactly {EXPECTED_HOST_THREADS}")
    if train_py != CANONICAL_TRAIN:
        raise ValueError(f"candidate must resolve exactly to {CANONICAL_TRAIN}")
    if sha256(train_py) != EXPECTED_TRAIN_SHA256:
        raise ValueError("candidate hash mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
        raise ValueError("expected commit must be a full lowercase git SHA")
    if git_head() != args.expected_commit:
        raise ValueError("checked-out commit does not equal --expected-commit")
    if socket.gethostname() != args.expected_hostname:
        raise ValueError("hostname does not equal --expected-hostname")
    if full and args.expected_hostname != EXPECTED_HARDWARE_HOSTNAME:
        raise ValueError(f"full mode is pinned to hostname {EXPECTED_HARDWARE_HOSTNAME}")
    if full and not re.fullmatch(FULL_LABEL_PATTERN, args.run_label):
        raise ValueError("full-mode label does not match the authorized unique-label pattern")
    expected_cache = f"/tmp/neuron-cache-{args.run_label}"
    required = {
        "NEURON_LOGICAL_NC_CONFIG": EXPECTED_LNC,
        "NEURON_CC_FLAGS": EXPECTED_CC_FLAGS,
        "NEURON_COMPILE_CACHE_DIR": expected_cache,
        "WORLD_SIZE": str(EXPECTED_WORLD_SIZE),
        "LOCAL_WORLD_SIZE": str(EXPECTED_WORLD_SIZE),
        "OMP_NUM_THREADS": str(EXPECTED_HOST_THREADS),
        "MKL_NUM_THREADS": str(EXPECTED_HOST_THREADS),
        "PYTHONUNBUFFERED": "1",
        "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "1",
        "TRAINIUM_COMPONENT_RUN_LABEL": args.run_label,
        "TRAINIUM_COMPONENT_EXPECTED_COMMIT": args.expected_commit,
        "TRAINIUM_COMPONENT_EXPECTED_HOSTNAME": args.expected_hostname,
        "TRAINIUM_COMPONENT_AUTHORIZED_HEAD": args.authorized_head,
        "TRAINIUM_COMPONENT_AUTHORIZATION_SHA256": authorization["sha256"],
    }
    mismatches = {
        name: {"expected": expected, "actual": os.environ.get(name)}
        for name, expected in required.items() if os.environ.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"pinned environment mismatch: {json.dumps(mismatches, sort_keys=True)}")
    if full:
        static_environment = authorization.get("request_contract", {}).get(
            "environment_static", {}
        )
        static_mismatches = {
            name: {"expected": expected, "actual": os.environ.get(name)}
            for name, expected in static_environment.items()
            if os.environ.get(name) != expected
        }
        if static_mismatches:
            raise RuntimeError(
                "authorized static environment mismatch: "
                + json.dumps(static_mismatches, sort_keys=True)
            )
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if rank not in range(EXPECTED_WORLD_SIZE) or local_rank not in range(EXPECTED_WORLD_SIZE):
        raise RuntimeError("RANK and LOCAL_RANK must each be in [0, 3]")
    if full and os.environ.get("TRAINIUM_FRESH_START_RUN_ID") != args.run_label:
        raise RuntimeError("missing/mismatched fresh_start_guard run identity")
    validate_report_path(args.report, args.run_label)
    return identity_manifest(args, train_py, authorization)


def nested_exact(torch, left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(nested_exact(torch, left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(nested_exact(torch, a, b) for a, b in zip(left, right))
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and bool(left.detach().cpu().equal(right.detach().cpu()))
        )
    return type(left) is type(right) and left == right


def nested_finite(torch, value) -> bool:
    if isinstance(value, dict):
        return all(nested_finite(torch, item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(nested_finite(torch, item) for item in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().cpu().item())
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def fused_mlp_gate(torch, candidate, device, seed: int) -> dict[str, Any]:
    import torch.nn.functional as F

    batch, sequence, channels = MLP_SHAPE
    tokens = batch * sequence
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    up_host = torch.randn(MLP_HIDDEN, channels, generator=generator).mul_(0.02)
    down_host = torch.randn(channels, MLP_HIDDEN, generator=generator).mul_(0.02)
    config = candidate.GPTConfig(
        sequence_len=sequence,
        vocab_size=VOCAB_SIZE,
        n_layer=1,
        n_head=channels // 128,
        n_embd=channels,
        mlp_ratio=4,
        attn_scale=0.0,
        n_kv_heads=0,
    )
    mlp = candidate.MLP(config)
    with torch.no_grad():
        mlp.c_fc.weight.copy_(up_host)
        mlp.c_proj.weight.copy_(down_host)
    mlp.to(device)
    mlp.train()
    mlp.c_fc.refresh_weight_cache(torch.bfloat16)
    mlp.c_proj.refresh_weight_cache(torch.bfloat16)
    reference_up = up_host.to(device).requires_grad_(True)
    reference_down = down_host.to(device).requires_grad_(True)
    executions: list[dict[str, Any]] = []
    cache_refresh: dict[str, Any] = {"performed_after_first_execution": False}
    for execution in range(2):
        x_host = torch.randn(MLP_SHAPE, generator=generator).mul_(0.5).bfloat16()
        upstream_host = torch.randn(MLP_SHAPE, generator=generator).mul_(0.25).bfloat16()
        x = x_host.to(device).requires_grad_(True)
        upstream = upstream_host.to(device)
        output = mlp(x)
        fused_node_names = [name for name in graph_node_names(output) if "FusedMLP" in name]
        loss = (output.float() * upstream.float()).sum() / tokens
        loss.backward()
        synchronize(torch, device)
        actual = {
            "output": output.detach(),
            "input_gradient": x.grad.detach(),
            "up_gradient": mlp.c_fc.weight.grad.detach(),
            "down_gradient": mlp.c_proj.weight.grad.detach(),
        }

        # Independent eager/autograd expression: no candidate helper/custom Function/saved layout.
        reference_x = x_host.to(device).requires_grad_(True)
        hidden = F.relu(F.linear(reference_x, reference_up.to(torch.bfloat16)))
        reference_output = F.linear(hidden.square(), reference_down.to(torch.bfloat16))
        reference_loss = (reference_output.float() * upstream.float()).sum() / tokens
        reference_loss.backward()
        synchronize(torch, device)
        reference = {
            "output": reference_output.detach(),
            "input_gradient": reference_x.grad.detach(),
            "up_gradient": reference_up.grad.detach(),
            "down_gradient": reference_down.grad.detach(),
        }
        metrics = {
            name: tensor_metrics(torch, actual[name], reference[name], TOLERANCES[f"mlp_{name}"])
            for name in actual
        }
        executions.append({
            "execution": execution + 1,
            "after_weight_update_and_cache_refresh": execution == 1,
            "fused_autograd_nodes": fused_node_names,
            "fused_branch_exercised": bool(fused_node_names),
            "metrics": metrics,
            "passed": bool(fused_node_names) and all(metric["passed"] for metric in metrics.values()),
        })
        if execution == 0:
            before_cache = {
                "up": mlp.c_fc._cached_weight.detach().clone(),
                "down": mlp.c_proj._cached_weight.detach().clone(),
            }
            with torch.no_grad():
                mlp.c_fc.weight.add_(mlp.c_fc.weight.grad, alpha=-0.01)
                mlp.c_proj.weight.add_(mlp.c_proj.weight.grad, alpha=-0.01)
                reference_up.add_(reference_up.grad, alpha=-0.01)
                reference_down.add_(reference_down.grad, alpha=-0.01)
            mlp.c_fc.weight.grad = None
            mlp.c_proj.weight.grad = None
            reference_up.grad = None
            reference_down.grad = None
            mlp.c_fc.refresh_weight_cache(torch.bfloat16)
            mlp.c_proj.refresh_weight_cache(torch.bfloat16)
            synchronize(torch, device)
            cache_refresh = {
                "performed_after_first_execution": True,
                "up_cache_changed": not bool(before_cache["up"].equal(mlp.c_fc._cached_weight)),
                "down_cache_changed": not bool(before_cache["down"].equal(mlp.c_proj._cached_weight)),
                "up_cache_matches_updated_master": bool(
                    mlp.c_fc._cached_weight.equal(mlp.c_fc.weight.detach().to(torch.bfloat16))
                ),
                "down_cache_matches_updated_master": bool(
                    mlp.c_proj._cached_weight.equal(mlp.c_proj.weight.detach().to(torch.bfloat16))
                ),
            }
            del before_cache
        del x, upstream, output, loss, actual, reference_x, hidden, reference_output
        del reference_loss, reference
        gc.collect()

    branch = {
        "training": mlp.training,
        "tokens": tokens,
        "tile": min(512, tokens // 2),
        "candidate_trigger_shape": list(MLP_SHAPE),
        "compiled_execution_count": len(executions),
        "later_execution_after_update": executions[1]["after_weight_update_and_cache_refresh"],
        "cache_refresh": cache_refresh,
    }
    passed = (
        len(executions) == 2
        and all(item["passed"] for item in executions)
        and branch["training"]
        and tokens == 32768
        and branch["tile"] == 512
        and all(cache_refresh.values())
    )
    result = {"passed": passed, "branch": branch, "executions": executions}
    del mlp, reference_up, reference_down
    gc.collect()
    synchronize(torch, device)
    return result


class GridProbe:
    """Record grid selection while forwarding every call to the real wrapped NKI kernel."""

    def __init__(self, wrapped, label: str):
        self.wrapped = wrapped
        self.label = label
        self.calls: list[dict[str, Any]] = []

    def __getitem__(self, lanes: int):
        real_call = self.wrapped[lanes]

        def invoke(*args, **kwargs):
            self.calls.append(
                {
                    "label": self.label,
                    "lanes": int(lanes),
                    "shapes": [list(value.shape) for value in args if hasattr(value, "shape")],
                }
            )
            return real_call(*args, **kwargs)

        return invoke


def adaptive_ce_case(torch, candidate, device, seed: int, rows: int) -> dict[str, Any]:
    import torch.nn.functional as F

    expected_lanes = 1 if rows == 128 else 2
    if candidate.selected_lanes(rows) != expected_lanes:
        raise AssertionError(f"selected_lanes({rows}) did not return {expected_lanes}")
    generator = torch.Generator(device="cpu").manual_seed(seed + 200 + rows)
    original_forward = candidate._softcap_ce_fwd_native
    original_backward = candidate._softcap_ce_bwd_native
    forward_probe = GridProbe(original_forward, "forward")
    backward_probe = GridProbe(original_backward, "backward")
    candidate._softcap_ce_fwd_native = forward_probe
    candidate._softcap_ce_bwd_native = backward_probe
    executions: list[dict[str, Any]] = []
    try:
        for execution in range(2):
            logits_host = torch.randn(rows, VOCAB_SIZE, generator=generator).mul_(6.0)
            targets_host = torch.randint(VOCAB_SIZE, (rows,), generator=generator)
            targets_host[:4] = torch.tensor(
                [0, VOCAB_SIZE - 1, VOCAB_SIZE // 2 - 1, VOCAB_SIZE // 2]
            )
            upstream_host = torch.randn(rows, generator=generator).mul_(0.25)
            logits = logits_host.to(device).requires_grad_(True)
            targets = targets_host.to(device)
            upstream = upstream_host.to(device)
            loss = candidate.softcap_cross_entropy(logits, targets, "none")
            direct_nodes = [
                name for name in graph_node_names(loss) if "SoftcapCrossEntropyNKI" in name
            ]
            gradient, = torch.autograd.grad(loss, logits, grad_outputs=upstream)
            synchronize(torch, device)

            # Independent native PyTorch softcap15 CE/autograd reference.
            reference_logits = logits_host.to(device).requires_grad_(True)
            reference_targets = targets_host.to(device)
            capped = 15.0 * torch.tanh(reference_logits / 15.0)
            reference_loss = F.cross_entropy(capped, reference_targets, reduction="none")
            reference_gradient, = torch.autograd.grad(
                reference_loss, reference_logits, grad_outputs=upstream_host.to(device)
            )
            synchronize(torch, device)
            metrics = {
                "loss": tensor_metrics(
                    torch, loss.detach(), reference_loss.detach(), TOLERANCES["ce_loss"]
                ),
                "gradient": tensor_metrics(
                    torch, gradient.detach(), reference_gradient.detach(), TOLERANCES["ce_gradient"]
                ),
            }
            executions.append({
                "execution": execution + 1,
                "direct_autograd_nodes": direct_nodes,
                "direct_autograd_exercised": bool(direct_nodes),
                "metrics": metrics,
                "passed": bool(direct_nodes) and all(metric["passed"] for metric in metrics.values()),
            })
            del logits, targets, upstream, loss, gradient, reference_logits, reference_targets
            del capped, reference_loss, reference_gradient
            gc.collect()
    finally:
        candidate._softcap_ce_fwd_native = original_forward
        candidate._softcap_ce_bwd_native = original_backward
    wiring = {
        "rows": rows,
        "vocab": VOCAB_SIZE,
        "expected_lanes": expected_lanes,
        "selected_lanes": candidate.selected_lanes(rows),
        "compiled_execution_count": len(executions),
        "later_execution_exercised": len(executions) == 2 and executions[1]["passed"],
        "wrapped_forward_calls": forward_probe.calls,
        "wrapped_backward_calls": backward_probe.calls,
        "wrapped_forward_exercised_twice": len(forward_probe.calls) == 2,
        "wrapped_backward_exercised_twice": len(backward_probe.calls) == 2,
        "wrapped_forward_selected_expected_grid": [call["lanes"] for call in forward_probe.calls]
        == [expected_lanes, expected_lanes],
        "wrapped_backward_selected_expected_grid": [call["lanes"] for call in backward_probe.calls]
        == [expected_lanes, expected_lanes],
    }
    passed = len(executions) == 2 and all(item["passed"] for item in executions) and all(
        (
            wiring["selected_lanes"] == expected_lanes,
            wiring["later_execution_exercised"],
            wiring["wrapped_forward_exercised_twice"],
            wiring["wrapped_backward_exercised_twice"],
            wiring["wrapped_forward_selected_expected_grid"],
            wiring["wrapped_backward_selected_expected_grid"],
        )
    )
    result = {"passed": passed, "wiring": wiring, "executions": executions}
    gc.collect()
    synchronize(torch, device)
    return result


def adaptive_ce_gate(torch, candidate, device, seed: int) -> dict[str, Any]:
    if not candidate._HAS_NKI:
        raise RuntimeError("candidate imported without the native NKI stack")
    cases = {str(rows): adaptive_ce_case(torch, candidate, device, seed, rows) for rows in (128, 512)}
    return {"passed": all(case["passed"] for case in cases.values()), "cases": cases}


def optimizer_state_coverage(torch, owner) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    passed = True
    for group_index, (source, positions, owned) in enumerate(
        zip(owner.optimizer.param_groups, owner.positions, owner.groups)
    ):
        actual_parameters = [source["params"][position] for position in positions]
        prefix_matches = all(a is b for a, b in zip(actual_parameters, owned["params"]))
        state_ok = True
        detail: dict[str, Any] = {}
        if positions and source["kind"] == "adamw":
            state_ok = all(
                parameter in owner.optimizer.state
                and {"step", "exp_avg", "exp_avg_sq"}.issubset(owner.optimizer.state[parameter])
                for parameter in actual_parameters
            )
            detail["adamw_state_parameters"] = len(actual_parameters)
        elif positions and source["kind"] == "muon":
            state = owner.optimizer.state[owned["params"][0]]
            required = {
                "momentum_buffer",
                "second_momentum_buffer",
                "stacked_grads",
                "stacked_params",
            }
            state_ok = required.issubset(state) and all(
                state[name].shape[0] == len(owned["params"])
                for name in required
            )
            detail.update(
                muon_actual_parameters=len(actual_parameters),
                muon_stacked_rows=len(owned["params"]),
                singleton_padding=group_index in owner.padding,
            )
        check = {
            "group": group_index,
            "kind": source["kind"],
            "positions": positions,
            "prefix_matches_source": prefix_matches,
            "state_complete": state_ok,
            **detail,
        }
        checks.append(check)
        passed = passed and prefix_matches and state_ok
    return {"passed": passed, "groups": checks}


def optimizer_gate(torch, candidate, device, seed: int, rank: int, world_size: int) -> dict[str, Any]:
    config = candidate.GPTConfig(
        sequence_len=128,
        vocab_size=VOCAB_SIZE,
        n_layer=2,
        n_head=1,
        n_embd=128,
        mlp_ratio=4,
        attn_scale=0.0,
        n_kv_heads=0,
    )
    torch.manual_seed(seed + 301)
    reference_model = candidate.GPT(config)
    reference_model.init_weights()
    torch.manual_seed(seed + 301)
    owned_model = candidate.GPT(config)
    owned_model.init_weights()
    if not all(
        left.equal(right)
        for left, right in zip(reference_model.state_dict().values(), owned_model.state_dict().values())
    ):
        raise AssertionError("optimizer fixtures did not initialize identically")
    reference_model.to(device)
    owned_model.to(device)
    candidate.COMPUTE_DTYPE = torch.bfloat16
    reference_optimizer = reference_model.setup_optimizer()
    base_optimizer = owned_model.setup_optimizer()
    owner = candidate.OwnedOptimizer(base_optimizer)
    api = {
        "base_param_groups_retained": hasattr(base_optimizer, "param_groups"),
        "base_zero_grad_callable": callable(getattr(base_optimizer, "zero_grad", None)),
        "owner_has_no_param_groups_api": not hasattr(owner, "param_groups"),
        "owner_has_no_zero_grad_api": not hasattr(owner, "zero_grad"),
        "explicit_owner_is_distinct_from_base": owner is not base_optimizer,
        "base_state_empty_before_first_update": not base_optimizer.state,
    }
    expected_slots = {
        (group_index, position)
        for group_index, group in enumerate(base_optimizer.param_groups)
        for position in range(len(group["params"]))
    }
    actual_slots = {(entry["group"], entry["position"]) for entry in owner.entries}
    plan = {
        "entries": len(owner.entries),
        "parameters": len(owner.parameters),
        "expected_slots": len(expected_slots),
        "slot_set_exact": actual_slots == expected_slots,
        "entry_indices_exact": [entry["index"] for entry in owner.entries]
        == list(range(len(owner.entries))),
        "parameter_ids_exact": owner.parameter_ids == [id(parameter) for parameter in owner.parameters],
        "all_four_owners_present": {entry["owner"] for entry in owner.entries} == set(range(world_size)),
        "local_positions_exact": all(
            positions
            == [
                entry["position"]
                for entry in owner.entries
                if entry["group"] == group_index and entry["owner"] == rank
            ]
            for group_index, positions in enumerate(owner.positions)
        ),
        "digest": hashlib.sha256(
            json.dumps(owner.entries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "summary": owner.summary,
    }
    reference_named = list(reference_model.named_parameters())
    owned_named = list(owned_model.named_parameters())
    if [name for name, _parameter in reference_named] != [name for name, _parameter in owned_named]:
        raise AssertionError("optimizer fixture parameter names differ")
    initial_reference = {name: parameter.detach().clone() for name, parameter in reference_named}
    initial_owned = {name: parameter.detach().clone() for name, parameter in owned_named}
    gradient_generator = torch.Generator(device="cpu").manual_seed(seed + 302)
    schedules = (
        {"lr_factor": 0.5, "momentum": 0.90, "weight_decay": 0.10, "beta2": 0.90},
        {"lr_factor": 0.35, "momentum": 0.92, "weight_decay": 0.05, "beta2": 0.88},
    )
    executions: list[dict[str, Any]] = []
    for step_index, schedule in enumerate(schedules):
        for (_name_a, reference_parameter), (_name_b, owned_parameter) in zip(
            reference_named, owned_named
        ):
            host_gradient = torch.randn(
                reference_parameter.shape, generator=gradient_generator, dtype=torch.float32
            ).mul_(0.01 * (step_index + 1))
            device_gradient = host_gradient.to(device)
            reference_parameter.grad = device_gradient
            owned_parameter.grad = device_gradient.clone()
        for reference_group, owned_group in zip(
            reference_optimizer.param_groups, base_optimizer.param_groups
        ):
            for group in (reference_group, owned_group):
                group["lr"] = group["initial_lr"] * schedule["lr_factor"]
                if group["kind"] == "muon":
                    for key in ("momentum", "weight_decay", "beta2"):
                        group[key] = schedule[key]
        reference_optimizer.step(update_adamw=True)
        owner.step(update_adamw=True)
        synchronize(torch, device)
        parameter_metrics = {}
        update_metrics = {}
        for (name, reference_parameter), (_owned_name, owned_parameter) in zip(
            reference_named, owned_named
        ):
            parameter_metrics[name] = tensor_metrics(
                torch, owned_parameter, reference_parameter, TOLERANCES["optimizer_parameter"]
            )
            update_metrics[name] = tensor_metrics(
                torch,
                owned_parameter.detach() - initial_owned[name],
                reference_parameter.detach() - initial_reference[name],
                TOLERANCES["optimizer_update"],
            )
        executions.append({
            "step": step_index + 1,
            "schedule": schedule,
            "parameter_metrics_passed": all(item["passed"] for item in parameter_metrics.values()),
            "update_delta_metrics_passed": all(item["passed"] for item in update_metrics.values()),
            "max_parameter_abs": max(item["max_abs"] for item in parameter_metrics.values()),
            "max_update_error_norm_ratio": max(
                item["error_norm_ratio"] for item in update_metrics.values()
            ),
        })
    worst_parameter = max(parameter_metrics, key=lambda name: parameter_metrics[name]["max_abs"])
    replicated_parameter_hasher = hashlib.sha256()
    for name, parameter in owned_named:
        replicated_parameter_hasher.update(name.encode())
        replicated_parameter_hasher.update(str(parameter.dtype).encode())
        replicated_parameter_hasher.update(str(tuple(parameter.shape)).encode())
        replicated_parameter_hasher.update(parameter.detach().float().cpu().numpy().tobytes())
    parameter_summary = {
        "passed": all(item["parameter_metrics_passed"] for item in executions),
        "update_deltas_passed": all(item["update_delta_metrics_passed"] for item in executions),
        "executions": executions,
        "later_update_exercised": len(executions) == 2,
        "count": len(parameter_metrics),
        "worst_parameter": worst_parameter,
        "max_abs": parameter_metrics[worst_parameter]["max_abs"],
        "replicated_parameter_digest": replicated_parameter_hasher.hexdigest(),
        **TOLERANCES["optimizer_parameter"],
    }
    schedule_propagated = all(
        all(owned_group[key] == source[key] for key in source if key != "params")
        for source, owned_group in zip(base_optimizer.param_groups, owner.groups)
    )
    coverage = optimizer_state_coverage(torch, owner)
    payload = owner.state_dict()
    schema = {
        "schema": payload.get("schema"),
        "schema_exact": payload.get("schema") == "fresh-start-owned-optimizer-v1",
        "rank_exact": payload.get("rank") == rank,
        "world_size_exact": payload.get("world_size") == world_size,
        "entries_exact": payload.get("entries") == owner.entries,
        "pad_singleton_muon_exact": payload.get("pad_singleton_muon") is True,
        "optimizer_state_finite": nested_finite(torch, payload["optimizer"]),
        "stacked_transient_fields_absent": all(
            name not in fields
            for fields in payload["optimizer"]["state"].values()
            for name in ("stacked_params", "stacked_grads")
        ),
    }
    torch.manual_seed(seed + 303)
    reload_model = candidate.GPT(config)
    reload_model.init_weights()
    reload_model.to(device)
    reload_optimizer = reload_model.setup_optimizer()
    reload_owner = candidate.OwnedOptimizer(reload_optimizer)
    reload_owner.load_state_dict(payload)
    roundtrip_exact = nested_exact(torch, payload, reload_owner.state_dict())
    bad_schema_rejected = False
    try:
        reload_owner.load_state_dict({**payload, "schema": "invalid-schema"})
    except ValueError:
        bad_schema_rejected = True
    state = {
        "coverage": coverage,
        "schema": schema,
        "roundtrip_exact": roundtrip_exact,
        "bad_schema_rejected": bad_schema_rejected,
        "owner_calls": owner.calls,
        "schedule_propagated": schedule_propagated,
        "send_finite": bool(torch.isfinite(owner.send).all().cpu().item()),
        "received_finite": bool(torch.isfinite(owner.received).all().cpu().item()),
    }
    passed = (
        all(api.values())
        and all(value is True or key in {"entries", "parameters", "expected_slots", "digest", "summary"}
                for key, value in plan.items())
        and parameter_summary["passed"]
        and parameter_summary["update_deltas_passed"]
        and parameter_summary["later_update_exercised"]
        and coverage["passed"]
        and all(value is True or key == "schema" for key, value in schema.items())
        and roundtrip_exact
        and bad_schema_rejected
        and owner.calls == 2
        and schedule_propagated
        and state["send_finite"]
        and state["received_finite"]
    )
    result = {
        "passed": passed,
        "api_split": api,
        "ownership_plan": plan,
        "parameter_reference": parameter_summary,
        "state": state,
    }
    del reference_model, owned_model, reload_model, reference_optimizer, base_optimizer
    del owner, reload_optimizer, reload_owner, payload, parameter_metrics, update_metrics
    del initial_reference, initial_owned
    gc.collect()
    synchronize(torch, device)
    return result


def guarded_gate(name: str, function) -> dict[str, Any]:
    try:
        result = function()
        if not isinstance(result, dict) or not isinstance(result.get("passed"), bool):
            raise TypeError(f"{name} did not return a boolean passed field")
        return result
    except Exception as error:  # Fail closed while allowing every rank to reach global reduction.
        return {
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=12),
        }
    finally:
        gc.collect()


def numeric_metric(result: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = result
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def provision_fresh_cache(args, identity: dict[str, Any]) -> dict[str, Any]:
    cache = Path(os.environ["NEURON_COMPILE_CACHE_DIR"])
    marker = cache / ".direct-component-identity.json"
    payload = {
        "run_label": args.run_label,
        "git_commit": args.expected_commit,
        "hostname": args.expected_hostname,
        "candidate_sha256": EXPECTED_TRAIN_SHA256,
        "harness_sha256": identity["files"]["harness"]["sha256"],
    }
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        if os.path.lexists(cache):
            raise FileExistsError(f"fresh per-run compile cache already exists: {cache}")
        os.mkdir(cache, 0o700)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(marker, flags, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
    deadline = time.monotonic() + 30.0
    observed = None
    while time.monotonic() < deadline:
        try:
            if marker.is_file() and not marker.is_symlink():
                observed = json.loads(marker.read_text())
                if observed == payload:
                    break
        except (json.JSONDecodeError, OSError):
            pass
        time.sleep(0.1)
    if not cache.is_dir() or cache.is_symlink() or marker.is_symlink() or not marker.is_file():
        raise RuntimeError("fresh compile-cache identity was not safely created")
    if observed != payload:
        raise RuntimeError("fresh compile-cache identity mismatch")
    return {"path": str(cache), "marker": str(marker), "identity": payload, "fresh": True}


def exact_request(args, identity: dict[str, Any]) -> dict[str, Any]:
    authorization = identity["authorization"]
    environment = dict(EXPECTED_STATIC_ENVIRONMENT)
    environment.update({
        "NEURON_COMPILE_CACHE_DIR": f"/tmp/neuron-cache-{args.run_label}",
        "TRAINIUM_COMPONENT_RUN_LABEL": args.run_label,
        "TRAINIUM_COMPONENT_EXPECTED_COMMIT": args.expected_commit,
        "TRAINIUM_COMPONENT_EXPECTED_HOSTNAME": args.expected_hostname,
        "TRAINIUM_COMPONENT_AUTHORIZED_HEAD": args.authorized_head,
        "TRAINIUM_COMPONENT_AUTHORIZATION_SHA256": authorization["sha256"],
    })
    command = [
        *EXPECTED_TIMEOUT_ARGV,
        *EXPECTED_TORCHRUN_ARGV_PREFIX,
        str(HARNESS_PATH),
        "--train-py", str(CANONICAL_TRAIN),
        "--report", str(args.report),
        "--run-label", args.run_label,
        "--expected-commit", args.expected_commit,
        "--expected-hostname", args.expected_hostname,
        "--authorized-head", args.authorized_head,
        "--authorization-request", str(AUTHORIZATION_PATH),
        "--authorization-sha256", args.authorization_sha256,
        "--seed", str(EXPECTED_SEED),
        "--host-threads", str(EXPECTED_HOST_THREADS),
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
        "fingerprint_paths": list(FINGERPRINT_PATHS),
        "authorization_request": AUTHORIZATION_RELATIVE_PATH,
        "authorization_sha256": args.authorization_sha256,
    }


def validate_launcher_manifest(
    payload: dict[str, Any], authorized_request: dict[str, Any],
    required_hashes: dict[str, str], args, suffix: str,
) -> None:
    augmented = {
        "started_at", "hostname", "git_commit", "source_dirty",
        "preflight_free_bytes", "source_sha256",
    }
    if set(payload) != set(authorized_request) | augmented:
        raise RuntimeError(f"{suffix} manifest schema/key mismatch")
    observed_request = {key: payload[key] for key in authorized_request}
    if observed_request != authorized_request:
        raise RuntimeError(f"{suffix} exact authorized request mismatch")
    if (
        payload.get("git_commit") != args.expected_commit
        or payload.get("hostname") != args.expected_hostname
        or payload.get("remote") != str(ROOT)
        or payload.get("source_dirty") is not False
    ):
        raise RuntimeError(f"{suffix} immutable checkout identity mismatch")
    if payload.get("source_sha256") != required_hashes:
        raise RuntimeError(f"{suffix} fingerprint mismatch")
    if not isinstance(payload["started_at"], str) or not isinstance(
        payload["preflight_free_bytes"], int
    ):
        raise RuntimeError(f"{suffix} launcher augmentation type mismatch")


def outer_receipts(args, identity: dict[str, Any]) -> dict[str, Any]:
    authorized_request = exact_request(args, identity)
    suffixes = ("launch.json", "job.json", "preflight.json")
    receipts: dict[str, Any] = {}
    for suffix in suffixes:
        path = RUN_LOGS / f"{args.run_label}.{suffix}"
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"required fresh-start receipt missing/unsafe: {path}")
        payload = json.loads(path.read_text())
        receipts[suffix] = {"path": str(path), "sha256": sha256(path)}
        if suffix != "preflight.json":
            required_hashes = dict(identity["authorization"]["expected_sha256"])
            required_hashes[AUTHORIZATION_RELATIVE_PATH] = identity["authorization"]["sha256"]
            validate_launcher_manifest(
                payload, authorized_request, required_hashes, args, suffix
            )
        else:
            if (
                payload.get("label") != args.run_label
                or payload.get("remote") != str(ROOT)
                or payload.get("passed") is not True
            ):
                raise RuntimeError("preflight receipt does not prove a passing matching preflight")
            if payload.get("hostname") != args.expected_hostname:
                raise RuntimeError("preflight hostname mismatch")
    receipts["process.json"] = {
        "path": str(RUN_LOGS / f"{args.run_label}.process.json"),
        "status": "written_by_guard_after_workload_exit",
    }
    receipts["postflight.json"] = {
        "path": str(RUN_LOGS / f"{args.run_label}.postflight.json"),
        "status": "pending_until_guard_cleanup_after_workload_exit",
    }
    receipts["log"] = {"path": str(RUN_LOGS / f"{args.run_label}.log")}
    return receipts


def run_neuron(
    args, train_py: Path, static: dict[str, Any], identity: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        raise RuntimeError("process group was already initialized")
    torch.set_num_threads(args.host_threads)
    receipts = outer_receipts(args, identity)
    cache_identity = provision_fresh_cache(args, identity)
    candidate = import_candidate(train_py)
    if not getattr(candidate, "_HAS_NKI", False):
        raise RuntimeError("candidate did not import the native NKI stack")
    initialized = False
    cleanup_succeeded = False
    rank = int(os.environ.get("RANK", "-1"))
    local_report: dict[str, Any] = {}
    global_report: dict[str, Any] = {}
    try:
        dist.init_process_group("neuron")
        initialized = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != EXPECTED_WORLD_SIZE:
            raise RuntimeError(f"expected {EXPECTED_WORLD_SIZE} ranks, got {world_size}")
        device = torch.device("neuron")
        torch.manual_seed(args.seed)
        environment = {
            "rank": rank,
            "world_size": world_size,
            "device": str(device),
            "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
            "torch_version": torch.__version__,
            "torch_neuron_available": hasattr(torch, "neuron"),
            "candidate_has_nki": candidate._HAS_NKI,
            "seed": args.seed,
        }
        mlp = guarded_gate("fused_mlp", lambda: fused_mlp_gate(torch, candidate, device, args.seed))
        ce = guarded_gate("adaptive_ce", lambda: adaptive_ce_gate(torch, candidate, device, args.seed))
        optimizer = guarded_gate(
            "owned_optimizer",
            lambda: optimizer_gate(torch, candidate, device, args.seed, rank, world_size),
        )
        local_report = {
            "rank": rank,
            "environment": environment,
            "fused_mlp": mlp,
            "adaptive_ce": ce,
            "owned_optimizer": optimizer,
            "passed": mlp["passed"] and ce["passed"] and optimizer["passed"],
        }
        stage_names = ("fused_mlp", "adaptive_ce", "owned_optimizer")
        status = torch.tensor(
            [int(local_report[name]["passed"]) for name in stage_names],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(status, op=dist.ReduceOp.SUM)
        status_counts = [int(value) for value in status.cpu().tolist()]
        plan_digest = optimizer.get("ownership_plan", {}).get("digest")
        plan_available = isinstance(plan_digest, str) and len(plan_digest) == 64
        plan_bytes = bytes.fromhex(plan_digest) if plan_available else bytes(32)
        plan_input = torch.tensor(list(plan_bytes), dtype=torch.float32, device=device)
        gathered_plan = torch.empty(32 * world_size, dtype=torch.float32, device=device)
        dist.all_gather_into_tensor(gathered_plan, plan_input)
        gathered_plan_rows = gathered_plan.cpu().reshape(world_size, 32)
        plan_digests_agree = plan_available and all(
            bool(gathered_plan_rows[index].equal(gathered_plan_rows[0]))
            for index in range(1, world_size)
        )
        parameter_digest = optimizer.get("parameter_reference", {}).get(
            "replicated_parameter_digest"
        )
        parameter_digest_available = isinstance(parameter_digest, str) and len(parameter_digest) == 64
        parameter_bytes = bytes.fromhex(parameter_digest) if parameter_digest_available else bytes(32)
        parameter_input = torch.tensor(list(parameter_bytes), dtype=torch.float32, device=device)
        gathered_parameters = torch.empty(32 * world_size, dtype=torch.float32, device=device)
        dist.all_gather_into_tensor(gathered_parameters, parameter_input)
        parameter_rows = gathered_parameters.cpu().reshape(world_size, 32)
        parameter_digests_agree = parameter_digest_available and all(
            bool(parameter_rows[index].equal(parameter_rows[0])) for index in range(1, world_size)
        )
        common_identity = json.loads(json.dumps(identity))
        common_identity["environment"].pop("RANK", None)
        common_identity["environment"].pop("LOCAL_RANK", None)
        identity_digest = hashlib.sha256(
            json.dumps(common_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        identity_input = torch.tensor(list(bytes.fromhex(identity_digest)), dtype=torch.float32, device=device)
        gathered_identity = torch.empty(32 * world_size, dtype=torch.float32, device=device)
        dist.all_gather_into_tensor(gathered_identity, identity_input)
        identity_rows = gathered_identity.cpu().reshape(world_size, 32)
        identity_agrees = all(
            bool(identity_rows[index].equal(identity_rows[0])) for index in range(1, world_size)
        )
        metric_paths = (
            ("fused_mlp", "executions", "1", "metrics", "output", "max_abs"),
            ("fused_mlp", "executions", "1", "metrics", "input_gradient", "max_abs"),
            ("adaptive_ce", "cases", "128", "executions", "1", "metrics", "loss", "max_abs"),
            ("adaptive_ce", "cases", "128", "executions", "1", "metrics", "gradient", "max_abs"),
            ("adaptive_ce", "cases", "512", "executions", "1", "metrics", "loss", "max_abs"),
            ("adaptive_ce", "cases", "512", "executions", "1", "metrics", "gradient", "max_abs"),
            ("owned_optimizer", "parameter_reference", "max_abs"),
        )
        values = [numeric_metric(local_report, path) for path in metric_paths]
        available = torch.tensor(
            [int(value is not None) for value in values], dtype=torch.float32, device=device
        )
        maxima = torch.tensor(
            [0.0 if value is None else value for value in values],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(available, op=dist.ReduceOp.SUM)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        metric_names = [".".join(path) for path in metric_paths]
        global_report = {
            "rank_pass_counts": dict(zip(stage_names, status_counts)),
            "required_pass_count_per_stage": world_size,
            "max_abs_across_ranks": {
                name: float(value)
                for name, value, count in zip(metric_names, maxima.cpu().tolist(), available.cpu().tolist())
                if int(count) == world_size
            },
            "metrics_available_on_all_ranks": all(int(count) == world_size for count in available.cpu().tolist()),
            "ownership_plan_digest_all_ranks": plan_digests_agree,
            "replicated_parameter_digest": parameter_digest,
            "replicated_parameter_digest_all_ranks": parameter_digests_agree,
            "identity_digest": identity_digest,
            "identity_digest_all_ranks": identity_agrees,
            "all_ranks_passed": (
                all(count == world_size for count in status_counts)
                and all(int(count) == world_size for count in available.cpu().tolist())
                and plan_digests_agree
                and parameter_digests_agree
                and identity_agrees
            ),
        }
        print(
            "DIRECT_NEURON_COMPONENT_RANK "
            + json.dumps(
                {
                    "rank": rank,
                    "passed": local_report["passed"],
                    "stages": {name: local_report[name]["passed"] for name in stage_names},
                    "errors": {
                        name: {
                            "type": local_report[name].get("error_type"),
                            "error": local_report[name].get("error"),
                        }
                        for name in stage_names
                        if not local_report[name]["passed"]
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
        dist.barrier()
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()
            cleanup_succeeded = not dist.is_initialized()
    passed = bool(global_report.get("all_ranks_passed")) and cleanup_succeeded
    report = {
        "schema": "dense6x1024-direct-neuron-component-v1",
        "scope": "standalone real-Neuron component gate; no data, training, scoring, or quality verdict",
        "identity": identity,
        "fresh_compile_cache": cache_identity,
        "outer_receipts": receipts,
        "static_wiring": static,
        "tolerances": TOLERANCES,
        "expected_environment": {
            "world_size": EXPECTED_WORLD_SIZE,
            "NEURON_LOGICAL_NC_CONFIG": EXPECTED_LNC,
            "device": "neuron",
        },
        "distributed_initialized": initialized,
        "distributed_cleanup_succeeded": cleanup_succeeded,
        "rank0": local_report if rank == 0 else {"rank": rank},
        "global": global_report,
        "passed": passed,
    }
    return report, 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-py", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--authorized-head", required=True)
    parser.add_argument("--authorization-request", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host-threads", type=int, default=4)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Validate pins, source wiring, and CPU adversarial metric rejection; no Neuron use.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_py = args.train_py.resolve()
    args.report = args.report.absolute()
    if not train_py.is_file():
        raise FileNotFoundError(train_py)
    identity = validate_identity(args, train_py, full=not args.static_only)
    static = static_wiring_checks(train_py)
    adversarial = metric_adversarial_self_test()
    static["adversarial_metric_rejection"] = adversarial
    static["passed"] = static["passed"] and adversarial["passed"]
    if not static["passed"]:
        report = {
            "schema": "dense6x1024-direct-neuron-component-v1",
            "mode": "static-only" if args.static_only else "neuron",
            "static_wiring": static,
            "passed": False,
        }
        write_report(args.report, args.run_label, report)
        print("DIRECT_NEURON_COMPONENT " + json.dumps(report, sort_keys=True), flush=True)
        return 1
    if args.static_only:
        report = {
            "schema": "dense6x1024-direct-neuron-component-v1",
            "mode": "static-only",
            "identity": identity,
            "static_wiring": static,
            "tolerances": TOLERANCES,
            "passed": True,
            "qualification": "No Neuron operation was executed.",
        }
        write_report(args.report, args.run_label, report)
        print("DIRECT_NEURON_COMPONENT " + json.dumps(report, sort_keys=True), flush=True)
        return 0
    report, exit_code = run_neuron(args, train_py, static, identity)
    if int(os.environ.get("RANK", "-1")) == 0:
        write_report(args.report, args.run_label, report)
        print("DIRECT_NEURON_COMPONENT " + json.dumps(report, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        failure = {
            "schema": "dense6x1024-direct-neuron-component-v1",
            "rank": int(os.environ.get("RANK", "-1")),
            "passed": False,
            "fatal_error_type": type(error).__name__,
            "fatal_error": str(error),
            "traceback": traceback.format_exc(limit=16),
        }
        if failure["rank"] in (-1, 0) and "--report" in sys.argv and "--run-label" in sys.argv:
            try:
                report_index = sys.argv.index("--report") + 1
                label_index = sys.argv.index("--run-label") + 1
                write_report(
                    Path(sys.argv[report_index]).absolute(), sys.argv[label_index], failure
                )
            except Exception as report_error:
                failure["report_write_error"] = f"{type(report_error).__name__}: {report_error}"
        print("DIRECT_NEURON_COMPONENT_FATAL " + json.dumps(failure, sort_keys=True), flush=True)
        raise SystemExit(2)
