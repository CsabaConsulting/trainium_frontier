#!/usr/bin/env python3
"""Fail-closed G0 request materializer and runtime.

This file is not authority.  ``materialize`` requires three exact literals supplied by an
independent finalizer and an authorization blob whose current pending template deliberately cannot
pass.  ``run`` validates the launcher-produced runtime job, starts only the exact G0 torchrun in a
private process group, and hashes artifacts only after bounded TERM/KILL/reap quiescence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "g0-request.template.json"
EXPECTED_BRANCH = "experiment/20260924-v2-a1-hybrid-native5-g0-finalization"
CORE = "7528ca8dd9d9915097929217c2a2b0d9cfc17107"
VERIFIER_COMMIT = "aeb85c2cee3481901c827d7a2313b9178906fc84"
VERIFIER_SHA = "3e2ee9548202d134bc71b9d87b6bf9808857393aa45ec0fd9f6c65ff7fecbacf"
FINALIZATION_PATHS = [
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g0_finalization.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g0-request.template.json",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/EXPERIMENT_PROPOSAL.json",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/RUNBOOK.md",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/requests.pending.json",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/test_gate_package.py",
]
EXPECTED_SOURCE_SHA256 = {
    "prepare.py": "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb",
    "submissions/dense6x1024_public0992851/train.py": "9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae",
    "research/a1_native_gate_observer.py": "89ad30056bf3fe207c43ea61f8c8a94583232f9f21e5158a80fd37cce42d02e7",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_hybrid.py": "908431d7e1da8a9110e6e6005de5ecbe3dba13d798bf2f37f2443404dd8d7d94",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_ordinary.py": "6ed69d2356c28e0dd8f88f494d988c0b87f7f86426a001945bfd958ed6f8f6d0",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/compare_gate_reports.py": "aa2d7c084ee17c5c639bd5ff7b6bc0cbf6e9575d5f802a056328eb4e387f5ca4",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/verify_model_checkpoints.py": "e1ead481f9786ec02a3c60d88e72820f084274200bccd1f7eeb16da108cb6e2e",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/verification.json": VERIFIER_SHA,
    "research/fresh_start_launch.py": "872c29d6703be64e3bcfdae4deff99acb459eca804f5a7bb574ac6d788e4a80c",
    "research/fresh_start_guard.py": "ac6f7d23dbe998b1d9c8ae14e209c6ffc21fac5d927895b1b92b15dc493f6de6",
    "research/fresh_start_host.py": "f5bc5034b4ac77e2af3234a9630ab8e4d0e3f1b24f7404ac22778d30016e6d16",
    "research/fresh_start_backup.py": "c978fd20b6620d0d80c25f8982974e8c4b2ad009edd9da376d784f5cecc3ead3",
    "research/fresh_start_sync.py": "ea428d33f4de18376950abbd81cfda33d3a969f75b60e3648ff552bea04fc60c",
    "research/run_experiment.py": "494e283f2464f1d08059eb69423165cbd40ff3fbf9ba9c89f746e0ce45096705",
}
EXPECTED_PATHS = {
    "compile_cache": "/home/ubuntu/neuron_cache_20260924_v2_a1_native5_g0_trn21_r1",
    "output": "out/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1",
    "report": "research/run-logs/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/native-gate",
    "child_log": "research/run-logs/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1.child.log",
    "runtime_result": "research/run-logs/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1.runtime-result.json",
}
EXPECTED_ENVIRONMENT = {
    "HOME": "/home/ubuntu", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    "LOGNAME": "ubuntu",
    "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
    "NEURON_COMPETITION_R1_SEED": "42", "NEURON_LOGICAL_NC_CONFIG": "2",
    "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv",
    "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
    "PYTHONUNBUFFERED": "1", "TMPDIR": "/tmp", "USER": "ubuntu",
}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


class ContractError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"not a regular non-symlink file: {path}")
    stat = path.stat()
    if stat.st_nlink != 1:
        raise ContractError(f"file has unexpected hard links: {path}")


def _load(path: Path) -> dict:
    _regular(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ContractError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root is not an object: {path}")
    return value


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise ContractError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _validate_checkout(head: str, protected: list[str]) -> None:
    if not HEX40.fullmatch(head):
        raise ContractError("authorized head is not a lowercase full commit")
    if _git("rev-parse", "HEAD") != head:
        raise ContractError("checkout HEAD differs from external authorized head")
    if _git("symbolic-ref", "--short", "HEAD") != EXPECTED_BRANCH:
        raise ContractError("checkout is detached or on the wrong symbolic branch")
    dirty = _git("diff", "--name-only", "HEAD", "--", *protected)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *protected)
    if dirty or untracked:
        raise ContractError("protected finalization checkout is dirty")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", CORE, head], cwd=ROOT
    ).returncode:
        raise ContractError("reviewed implementation is not an ancestor of authorized head")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", VERIFIER_COMMIT, head], cwd=ROOT
    ).returncode:
        raise ContractError("verifier receipt commit is not an ancestor of authorized head")


def validate_template(template: dict) -> None:
    required = {
        "schema", "state", "executable", "arm", "host", "expected_hostname", "label",
        "remote", "origin", "purpose", "exact_symbolic_branch", "verified_core_commit", "verifier_receipt_commit",
        "verifier_receipt_sha256", "external_literals", "authorization_path",
        "request_template_path", "runtime_driver_path", "runtime_job_path", "finalization_paths", "paths",
        "environment", "child_environment", "outer_timeout_argv", "child_timeout_seconds",
        "term_grace_seconds", "kill_grace_seconds", "min_free_bytes", "hard_abort_bytes",
        "world_size", "lnc", "seed", "optimizer_updates", "in_process_max_seconds",
        "source", "train_argv", "source_sha256", "lifecycle", "downstream_locks",
    }
    if set(template) != required:
        raise ContractError("G0 request-template schema mismatch")
    expected = {
        "schema": "a1-native-five-g0-request-template-v1",
        "state": "pending-external-finalization",
        "executable": False,
        "arm": "G0",
        "host": "trn21",
        "expected_hostname": "ip-172-31-34-85",
        "label": "fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1",
        "exact_symbolic_branch": EXPECTED_BRANCH,
        "remote": "/home/ubuntu/trainium_freshstart_20260924_v2_a1_native5_g0_hybrid_observation_off_trn21_r1",
        "origin": "https://github.com/CsabaConsulting/trainium_frontier.git",
        "authorization_path": "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g0-authorization.json",
        "request_template_path": "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g0-request.template.json",
        "runtime_driver_path": "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g0_finalization.py",
        "runtime_job_path": "research/run-logs/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1.job.json",
        "paths": EXPECTED_PATHS,
        "environment": EXPECTED_ENVIRONMENT,
        "child_environment": {
            **EXPECTED_ENVIRONMENT,
            "NEURON_COMPILE_CACHE_DIR": EXPECTED_PATHS["compile_cache"],
            "TRAINIUM_FRESH_START_RUN_ID": "fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1",
        },
        "source": "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_hybrid.py",
        "verified_core_commit": CORE,
        "verifier_receipt_commit": VERIFIER_COMMIT,
        "verifier_receipt_sha256": VERIFIER_SHA,
        "outer_timeout_argv": [
            "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "4200s"
        ],
        "child_timeout_seconds": 3600,
        "term_grace_seconds": 60,
        "kill_grace_seconds": 60,
        "min_free_bytes": 6_979_321_856,
        "hard_abort_bytes": 5_368_709_120,
        "world_size": 4, "lnc": 2, "seed": 42, "optimizer_updates": 5,
        "in_process_max_seconds": 1800,
    }
    for key, value in expected.items():
        if template.get(key) != value:
            raise ContractError(f"G0 template {key} drift")
    if template["external_literals"] != {
        "request_bearing_head": "PENDING_EXTERNAL_VERIFIER",
        "request_sha256": "PENDING_EXTERNAL_VERIFIER",
        "authorization_sha256": "PENDING_EXTERNAL_VERIFIER",
    }:
        raise ContractError("G0 template must remain pending external literals")
    if template["finalization_paths"] != FINALIZATION_PATHS:
        raise ContractError("G0 finalization protected-path set drift")
    if template["source_sha256"] != EXPECTED_SOURCE_SHA256:
        raise ContractError("G0 frozen source hash map drift")
    if template["downstream_locks"] != {
        "G1": "PENDING_AUTHENTICATED_G0_ARCHIVE_CLEAN_POSTFLIGHT_AND_NEW_EXTERNAL_AUTHORIZATION",
        "G2": "PENDING_AUTHENTICATED_G1_ARCHIVE_CLEAN_POSTFLIGHT_AND_NEW_EXTERNAL_AUTHORIZATION",
    }:
        raise ContractError("G1/G2 locks changed")
    if template["source"].endswith("train_gate_hybrid.py") is False:
        raise ContractError("G0 source is not the hybrid gate")
    argv = template["train_argv"]
    required_sequence = [
        "--num-steps", "5", "--max-train-seconds", "1800", "--n-embd", "1024",
        "--ngram-ve", "--ngram-ve-layers=-1", "--ngram-ve-table-mult", "64",
        "--ngram-ve-dim", "128", "--ngram-ve-trigram", "--ngram-ve-opt", "rmsprop",
        "--ngram-ve-flat-lr", "--ngram-ve-bf16", "--ngram-ve-clip-exclude",
        "--no-qk-shift", "--compile-sdpa-direct", "--nki-local-conv",
        "--bf16-norm-output", "--pack-factor", "32", "--clip-after-reduce",
        "--seq-len", "1024", "--no-eval-public", "--matrix-lr", "0.015",
        "--out-dir", template["paths"]["output"], "--native-gate-arm", "G0",
        "--no-native-gate-observation", "--native-gate-label", template["label"],
        "--native-gate-host", "trn21", "--native-gate-report-dir",
        template["paths"]["report"],
    ]
    if argv != required_sequence:
        raise ContractError("G0 train argv drift")
    if template["child_environment"].get("NEURON_COMPILE_CACHE_DIR") != template["paths"]["compile_cache"]:
        raise ContractError("compile cache is not exact")
    if template["environment"] != {
        key: value for key, value in template["child_environment"].items()
        if key not in {"NEURON_COMPILE_CACHE_DIR", "TRAINIUM_FRESH_START_RUN_ID"}
    }:
        raise ContractError("outer and child environment contract differs")
    lifecycle = template["lifecycle"]
    if lifecycle != {
        "private_child_process_group": True,
        "bounded_term_kill_reap_quiescence": True,
        "hash_only_after_quiescence": True,
        "no_cleanup_or_prune": True,
        "checkpoint_prune_authorized": False,
        "cache_cleanup_authorized": False,
    }:
        raise ContractError("G0 lifecycle contract drift")


def validate_authorization(auth: dict, template: dict, request_sha: str) -> None:
    required = {
        "schema", "state", "executable", "arm", "host", "label",
        "verified_core_commit", "verifier_receipt_commit", "verifier_receipt_sha256",
        "request_template_path", "request_sha256", "head_binding",
        "protected_finalization_sha256", "source_sha256", "native_execution_authorized",
        "g1_authorized", "g2_authorized", "checkpoint_prune_authorized",
        "cache_cleanup_authorized",
    }
    if set(auth) != required:
        raise ContractError("authorization schema mismatch")
    expected = {
        "schema": "a1-native-five-g0-authorization-v1",
        "state": "independent-verifier-finalized-g0-only",
        "executable": True,
        "arm": "G0", "host": "trn21", "label": template["label"],
        "verified_core_commit": CORE,
        "verifier_receipt_commit": VERIFIER_COMMIT,
        "verifier_receipt_sha256": VERIFIER_SHA,
        "request_template_path": template["request_template_path"],
        "request_sha256": request_sha,
        "head_binding": "external-literals-only",
        "native_execution_authorized": True,
        "g1_authorized": False, "g2_authorized": False,
        "checkpoint_prune_authorized": False, "cache_cleanup_authorized": False,
    }
    for key, value in expected.items():
        if auth.get(key) != value:
            raise ContractError(f"authorization {key} mismatch")
    if auth["source_sha256"] != template["source_sha256"]:
        raise ContractError("authorization source identities differ from request template")
    pins = auth["protected_finalization_sha256"]
    if not isinstance(pins, dict) or set(pins) != set(FINALIZATION_PATHS):
        raise ContractError("authorization finalization identity set is not exact")
    for relative, expected_sha in {**template["source_sha256"], **pins}.items():
        if not HEX64.fullmatch(expected_sha):
            raise ContractError(f"invalid protected hash for {relative}")
        path = ROOT / relative
        _regular(path)
        if _sha(path) != expected_sha:
            raise ContractError(f"protected hash mismatch: {relative}")


def build_concrete_request(
    template: dict, *, head: str, request_sha: str, authorization_sha: str,
    authorization_path: str,
) -> dict:
    driver_args = [
        "/home/ubuntu/workspace/native_venv/bin/python", template["runtime_driver_path"], "run",
        "--authorized-head", head, "--request-sha256", request_sha,
        "--authorization-sha256", authorization_sha,
        "--request-template", template["request_template_path"],
        "--authorization", authorization_path,
        "--runtime-job", template["runtime_job_path"],
    ]
    fingerprints = sorted(set(template["source_sha256"]) | set(template["finalization_paths"]) | {
        template["runtime_driver_path"], template["request_template_path"], authorization_path,
    })
    return {
        "schema": "a1-native-five-g0-concrete-request-v1",
        "arm": "G0", "host": "trn21", "expected_hostname": template["expected_hostname"],
        "label": template["label"], "remote": template["remote"],
        "expected_commit": head, "origin": template["origin"], "purpose": template["purpose"],
        "command": template["outer_timeout_argv"] + driver_args,
        "environment": template["environment"], "min_free_bytes": template["min_free_bytes"],
        "runner_owns_manifest": False, "non_overwrite": True,
        "fingerprint_paths": fingerprints, "verified_core_commit": CORE,
        "verifier_receipt_commit": VERIFIER_COMMIT,
        "request_template_path": template["request_template_path"],
        "request_sha256": request_sha, "authorization_path": authorization_path,
        "authorization_sha256": authorization_sha,
    }


def _write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        data = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def materialize(args: argparse.Namespace) -> int:
    pending_template = _load(TEMPLATE_PATH)
    if args.request_template != pending_template["request_template_path"]:
        raise ContractError("request-template path differs from fixed contract")
    if args.authorization != pending_template["authorization_path"]:
        raise ContractError("authorization path differs from fixed contract")
    template_path = (ROOT / args.request_template).resolve()
    auth_path = (ROOT / args.authorization).resolve()
    template = _load(template_path)
    validate_template(template)
    if _sha(template_path) != args.request_sha256:
        raise ContractError("request template differs from external request SHA256")
    auth = _load(auth_path)
    if _sha(auth_path) != args.authorization_sha256:
        raise ContractError("authorization differs from external authorization SHA256")
    validate_authorization(auth, template, args.request_sha256)
    protected = sorted(set(template["source_sha256"]) | set(auth["protected_finalization_sha256"]) |
                       {args.request_template, args.authorization})
    _validate_checkout(args.authorized_head, protected)
    concrete = build_concrete_request(
        template, head=args.authorized_head, request_sha=args.request_sha256,
        authorization_sha=args.authorization_sha256, authorization_path=args.authorization,
    )
    _write_exclusive(args.output, concrete)
    print(json.dumps({"status": "MATERIALIZED_NOT_LAUNCHED", "output": str(args.output),
                      "sha256": _sha(args.output)}, sort_keys=True))
    return 0


def _group_members(pgid: int) -> list[int]:
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(members)


def _quiesce_group(pgid: int, term_s: int, kill_s: int) -> list[dict]:
    actions: list[dict] = []
    for sig, grace in ((signal.SIGTERM, term_s), (signal.SIGKILL, kill_s)):
        members = _group_members(pgid)
        if not members:
            break
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        actions.append({"signal": sig.name, "members": members})
        deadline = time.monotonic() + grace
        while _group_members(pgid) and time.monotonic() < deadline:
            time.sleep(0.25)
    survivors = _group_members(pgid)
    if survivors:
        raise ContractError(f"private child process group survived cleanup: {survivors}")
    return actions


def _artifact_snapshot(paths: list[Path]) -> dict:
    rows: dict[str, dict] = {}
    for root in paths:
        if root.is_symlink() or not root.exists():
            raise ContractError(f"required artifact path missing or symlinked: {root}")
        files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        if not files:
            raise ContractError(f"required artifact path is empty: {root}")
        for path in files:
            _regular(path)
            stat = path.stat()
            rows[str(path.relative_to(ROOT))] = {
                "bytes": stat.st_size, "sha256": _sha(path), "mtime_ns": stat.st_mtime_ns,
            }
    return rows


def _validate_runtime_job(runtime: dict, concrete: dict, template: dict, auth: dict) -> None:
    for key, value in concrete.items():
        if runtime.get(key) != value:
            raise ContractError(f"launcher runtime job changed {key}")
    if runtime.get("hostname") != template["expected_hostname"]:
        raise ContractError("runtime hostname differs from trn21 identity")
    if runtime.get("git_commit") != concrete["expected_commit"] or runtime.get("source_dirty") is not False:
        raise ContractError("runtime checkout identity is not clean and exact")
    observed = runtime.get("source_sha256")
    if not isinstance(observed, dict):
        raise ContractError("runtime job lacks launcher source hashes")
    for path, expected in template["source_sha256"].items():
        if observed.get(path) != expected:
            raise ContractError(f"launcher-observed source mismatch: {path}")
    for path, expected in auth["protected_finalization_sha256"].items():
        if observed.get(path) != expected:
            raise ContractError(f"launcher-observed finalization mismatch: {path}")


def run(args: argparse.Namespace) -> int:
    pending_template = _load(TEMPLATE_PATH)
    if args.request_template != pending_template["request_template_path"]:
        raise ContractError("runtime request-template path differs from fixed contract")
    if args.authorization != pending_template["authorization_path"]:
        raise ContractError("runtime authorization path differs from fixed contract")
    if args.runtime_job != pending_template["runtime_job_path"]:
        raise ContractError("runtime job path differs from fixed contract")
    template_path = ROOT / args.request_template
    auth_path = ROOT / args.authorization
    template = _load(template_path)
    validate_template(template)
    if _sha(template_path) != args.request_sha256:
        raise ContractError("runtime request-template SHA mismatch")
    auth = _load(auth_path)
    if _sha(auth_path) != args.authorization_sha256:
        raise ContractError("runtime authorization SHA mismatch")
    validate_authorization(auth, template, args.request_sha256)
    protected = sorted(set(template["source_sha256"]) | set(auth["protected_finalization_sha256"]) |
                       {args.request_template, args.authorization})
    _validate_checkout(args.authorized_head, protected)
    concrete = build_concrete_request(
        template, head=args.authorized_head, request_sha=args.request_sha256,
        authorization_sha=args.authorization_sha256, authorization_path=args.authorization,
    )
    runtime = _load(ROOT / args.runtime_job)
    _validate_runtime_job(runtime, concrete, template, auth)
    if os.environ.get("TRAINIUM_FRESH_START_RUN_ID") != template["label"]:
        raise ContractError("fresh-start ownership marker missing or changed")

    paths = {key: Path(value) for key, value in template["paths"].items()}
    for key in ("output", "report", "child_log", "runtime_result"):
        if (ROOT / paths[key]).exists() or (ROOT / paths[key]).is_symlink():
            raise ContractError(f"G0 {key} path was not initially absent")
    cache = Path(template["paths"]["compile_cache"])
    if cache.exists() or cache.is_symlink():
        raise ContractError("G0 compile cache path was not initially absent")
    free = shutil.disk_usage(ROOT).free
    if free < template["hard_abort_bytes"]:
        raise ContractError("hard abort: less than 5 GiB free")
    if free < template["min_free_bytes"]:
        raise ContractError("launch blocked: less than 6.5 GiB free")
    cache.mkdir(mode=0o700)

    child_command = [
        "/home/ubuntu/workspace/native_venv/bin/torchrun", "--standalone", "--nproc_per_node=4",
        template["source"], *template["train_argv"],
    ]
    log_path = ROOT / paths["child_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    log_fd = os.open(log_path, flags, 0o600)
    started = time.monotonic()
    process = None
    timed_out = False
    interrupted: list[int] = []

    def forward(sig, _frame):
        interrupted.append(sig)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    actions: list[dict] = []
    try:
        process = subprocess.Popen(
            child_command, cwd=ROOT, env=dict(template["child_environment"]),
            stdout=log_fd, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=template["child_timeout_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
        actions = _quiesce_group(
            process.pid, template["term_grace_seconds"], template["kill_grace_seconds"]
        )
        process.wait(timeout=5)
    finally:
        os.fsync(log_fd)
        os.close(log_fd)
    if process is None:
        raise ContractError("G0 child did not start")
    if _group_members(process.pid):
        raise ContractError("G0 child process group is not quiescent")

    success = returncode == 0 and not timed_out and not interrupted
    artifact_paths = [log_path]
    if success:
        artifact_paths.extend([
            ROOT / paths["output"] / "final.pt", ROOT / paths["report"],
        ])
    artifacts = _artifact_snapshot(artifact_paths)
    report_files = sorted((ROOT / paths["report"]).glob("rank-*.jsonl")) if success else []
    if success and len(report_files) != 4:
        raise ContractError("G0 did not emit exactly four rank reports")
    result = {
        "schema": "a1-native-five-g0-runtime-result-v1", "arm": "G0",
        "status": "PASS" if success else "FAILED", "complete": success,
        "label": template["label"], "host": "trn21", "authorized_head": args.authorized_head,
        "request_sha256": args.request_sha256,
        "authorization_sha256": args.authorization_sha256,
        "runtime_job_sha256": _sha(ROOT / args.runtime_job), "command": child_command,
        "environment": template["child_environment"], "returncode": returncode,
        "timed_out": timed_out, "interrupted_signals": interrupted,
        "duration_seconds": time.monotonic() - started, "process_group_actions": actions,
        "process_group_quiescent": True, "artifacts": artifacts,
        "compile_cache": {"path": str(cache), "preserved": True},
        "checkpoint_pruned": False, "cache_cleaned": False,
    }
    _write_exclusive(ROOT / paths["runtime_result"], result)
    return 0 if success else 2


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--authorized-head", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--request-template", required=True)
    parser.add_argument("--authorization", required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    mat = sub.add_parser("materialize")
    _common(mat)
    mat.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run")
    _common(execute)
    execute.add_argument("--runtime-job", required=True)
    args = parser.parse_args()
    if not HEX64.fullmatch(args.request_sha256):
        parser.error("--request-sha256 must be lowercase SHA-256")
    if not HEX64.fullmatch(args.authorization_sha256):
        parser.error("--authorization-sha256 must be lowercase SHA-256")
    try:
        return materialize(args) if args.action == "materialize" else run(args)
    except ContractError as exc:
        print(f"G0_FINALIZATION_INVALID: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
