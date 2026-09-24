#!/usr/bin/env python3
"""Fail-closed G1 request materializer and runtime.

This file is not authority.  ``materialize`` requires three exact literals supplied by an
independent finalizer and an authorization blob whose current pending template deliberately cannot
pass.  ``run`` validates the launcher-produced runtime job, starts only the exact G1 torchrun in a
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
import stat
import subprocess
import sys
import time
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "g1-request.template.json"
EXPECTED_BRANCH = "experiment/20260924-v2-a1-hybrid-native5-g1-finalization"
CORE = "994d302b0036c56aaab63797f0561797bc4cf431"
VERIFIER_COMMIT = "8add2585d30b11ac2c50fb2701957657a1a6f30d"
VERIFIER_SHA = "c4da83d6fe19fddab548ef49cb1a2d2eb69146e37bc36f452bcf37b446fc8c68"
ANALYST_COMMIT = "9749a572fbf2344be02c7a1e16ba01e963f33a82"
ANALYST_SHA = "a530256c81a51ce5635e6c240eddffd03160955f02a57d5c88c0596b54a3be17"
R4_COMPARATOR_SHA = "00f1a7263bf7367fb7ab7568822396ff04839b726c9565bb1fd6d0d70ecca17c"
G0_EVIDENCE_COMMIT = "37aa3644e63e20a1427cfa69b44fd6de7e25cd81"
G0_REPLAY_SHA = "7198d39b9c21f1eb7e47f84e52398ca85d239ad564a36e6d524674d83355ab4a"
FINALIZATION_PATHS = [
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1_finalization.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-request.template.json",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/EXPERIMENT_PROPOSAL.json",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/RUNBOOK.md",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/requests.pending.json",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/test_g1_package.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/adjudicate_g0_g1.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-authorization.pending.json",
]
EXPECTED_SOURCE_SHA256 = {
    "prepare.py": "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb",
    "submissions/dense6x1024_public0992851/train.py": "9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae",
    "research/a1_native_gate_observer.py": "89ad30056bf3fe207c43ea61f8c8a94583232f9f21e5158a80fd37cce42d02e7",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_hybrid.py": "908431d7e1da8a9110e6e6005de5ecbe3dba13d798bf2f37f2443404dd8d7d94",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_ordinary.py": "6ed69d2356c28e0dd8f88f494d988c0b87f7f86426a001945bfd958ed6f8f6d0",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/compare_gate_reports.py": R4_COMPARATOR_SHA,
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/a1_ownership_manifest.json": "8119df41c8a9ecbfc3f9e69067c7e6ac162c2d5d8ba0a15635265304dcf865a2",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/verify_model_checkpoints.py": "e1ead481f9786ec02a3c60d88e72820f084274200bccd1f7eeb16da108cb6e2e",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/verification.json": "3e2ee9548202d134bc71b9d87b6bf9808857393aa45ec0fd9f6c65ff7fecbacf",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g0-comparator-r4-independent-verification.json": VERIFIER_SHA,
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/G1_DRAFTING_CHECKLIST_ANALYSIS.md": ANALYST_SHA,
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/OPERATOR_RECEIPT.json": "4cca77e78ee510a21c693a0ca285c259dab14e0ba69e89b8b63c555cf0b58d14",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/runtime/research/run-logs/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1.runtime-result.json": "d2d3dcafd5d31a3cc69a030f1db3798ba22aa77f69878a8be00919a803180eff",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/runtime/research/run-logs/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1.job.json": "400b339d4cc913a0bacabc8fd7c657cf3795f07bbd56b44ffde48d8a1e8c3d1a",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/runtime/research/run-logs/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1.postflight.json": "425be4b5357410a3bd8da350ab7862f435490ab706a3ada425ca9a9254c31c6b",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/runtime/research/run-logs/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/native-gate/rank-0.jsonl": "8aa51461a8cfba1efd1467d97d3ff4788728870b1809b93c2f27c8c8a1054064",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/runtime/research/run-logs/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/native-gate/rank-1.jsonl": "bb72db43603c324eb6d8b4a54a7a50d58a6186b9763dec661a1681d7dc88cf63",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/runtime/research/run-logs/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/native-gate/rank-2.jsonl": "db6e73e08a1154f4a62b488578cf36565b9accf951eeb40ebe4259e7ca069f7c",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/runtime/research/run-logs/fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1/native-gate/rank-3.jsonl": "897e2fea281755d9ab655670bd56d7d36fae4d90353c98f5e05f29cfb5289aee",
    "research/fresh_start_launch.py": "872c29d6703be64e3bcfdae4deff99acb459eca804f5a7bb574ac6d788e4a80c",
    "research/fresh_start_guard.py": "ac6f7d23dbe998b1d9c8ae14e209c6ffc21fac5d927895b1b92b15dc493f6de6",
    "research/fresh_start_host.py": "f5bc5034b4ac77e2af3234a9630ab8e4d0e3f1b24f7404ac22778d30016e6d16",
    "research/fresh_start_backup.py": "c978fd20b6620d0d80c25f8982974e8c4b2ad009edd9da376d784f5cecc3ead3",
    "research/fresh_start_sync.py": "ea428d33f4de18376950abbd81cfda33d3a969f75b60e3648ff552bea04fc60c",
    "research/run_experiment.py": "494e283f2464f1d08059eb69423165cbd40ff3fbf9ba9c89f746e0ce45096705",
}
EXPECTED_PATHS = {
    "compile_cache": "/home/ubuntu/neuron_cache_20260924_v2_a1_native5_g1_trn21_r1",
    "output": "out/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1",
    "report": "research/run-logs/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1/native-gate",
    "child_log": "research/run-logs/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1.child.log",
    "runtime_result": "research/run-logs/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1.runtime-result.json",
}
MATERIALIZATION_ROOT = Path("/tmp")
MATERIALIZATION_OUTPUT = MATERIALIZATION_ROOT / (
    "fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1.request.json"
)
COMPILE_CACHE_ROOT = Path("/home/ubuntu")
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


FileIdentity = tuple[int, int, int, int, int, int]
DirectoryIdentity = tuple[int, int, int]


class SealedWrite(NamedTuple):
    sha256: str
    identity: FileIdentity
    bytes: int


def _relative_parts(trusted_root: Path, path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    """Return a lexical path below an explicit root without following any component."""
    root = Path(os.path.abspath(os.fspath(trusted_root)))
    target = Path(os.path.abspath(os.fspath(path)))
    try:
        common = Path(os.path.commonpath((os.fspath(root), os.fspath(target))))
    except ValueError as exc:
        raise ContractError(f"path is outside trusted root {root}: {path}") from exc
    if common != root:
        raise ContractError(f"path escapes trusted root {root}: {path}")
    relative = target.relative_to(root)
    parts = relative.parts
    if any(part in {"", ".", ".."} or "/" in part for part in parts):
        raise ContractError(f"invalid lexical path below {root}: {path}")
    return root, target, parts


def _identity(value: os.stat_result) -> FileIdentity:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_size, value.st_mtime_ns,
    )


def _validate_chain(trusted_root: Path, path: Path, terminal: str) -> os.stat_result | None:
    """lstat every existing component and enforce the requested terminal state.

    ``terminal`` is one of ``absent``, ``file``, ``directory`` or ``existing``. Missing
    ancestors are permitted only for an absent terminal subsequently created with dirfd/openat.
    """
    if terminal not in {"absent", "file", "directory", "existing"}:
        raise ContractError(f"invalid terminal-state request: {terminal}")
    root, target, parts = _relative_parts(trusted_root, path)
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise ContractError(f"trusted root is unavailable: {root}: {exc}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ContractError(f"trusted root is not a real directory: {root}")
    if not parts:
        if terminal != "directory":
            raise ContractError(f"terminal aliases trusted root: {target}")
        return root_stat

    current = root
    missing = False
    terminal_stat: os.stat_result | None = None
    for index, part in enumerate(parts):
        current = current / part
        final = index == len(parts) - 1
        if missing:
            continue
        try:
            observed = os.lstat(current)
        except FileNotFoundError:
            missing = True
            continue
        except OSError as exc:
            raise ContractError(f"cannot lstat path component {current}: {exc}") from exc
        if stat.S_ISLNK(observed.st_mode):
            raise ContractError(f"symlink path component rejected: {current}")
        if not final:
            if not stat.S_ISDIR(observed.st_mode):
                raise ContractError(f"non-directory path ancestor rejected: {current}")
        else:
            terminal_stat = observed

    if terminal == "absent":
        if terminal_stat is not None:
            raise ContractError(f"terminal path was not absent: {target}")
        return None
    if terminal_stat is None:
        raise ContractError(f"required terminal path is missing: {target}")
    if terminal in {"file", "existing"} and stat.S_ISREG(terminal_stat.st_mode):
        if terminal_stat.st_nlink != 1:
            raise ContractError(f"file has unexpected hard links: {target}")
    elif terminal == "file":
        raise ContractError(f"terminal is not a regular file: {target}")
    elif terminal == "existing" and not stat.S_ISDIR(terminal_stat.st_mode):
        raise ContractError(f"terminal is neither a regular file nor directory: {target}")
    elif terminal == "directory" and not stat.S_ISDIR(terminal_stat.st_mode):
        raise ContractError(f"terminal is not a directory: {target}")
    return terminal_stat


def _open_directory_chain(trusted_root: Path, directory: Path, *, create: bool) -> int:
    root, target, parts = _relative_parts(trusted_root, directory)
    root_stat = _validate_chain(root, root, "directory")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(root, flags)
    except OSError as exc:
        raise ContractError(f"cannot open trusted root {root}: {exc}") from exc
    try:
        opened_root = os.fstat(fd)
        if not stat.S_ISDIR(opened_root.st_mode):
            raise ContractError(f"opened trusted root is not a directory: {root}")
        if root_stat is None or (opened_root.st_dev, opened_root.st_ino) != (
            root_stat.st_dev, root_stat.st_ino,
        ):
            raise ContractError(f"trusted root identity changed while opening: {root}")
        walked = root
        for part in parts:
            walked = walked / part
            try:
                next_fd = os.open(part, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise ContractError(f"required directory is missing: {target}")
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                    next_fd = os.open(part, flags, dir_fd=fd)
                except OSError as exc:
                    raise ContractError(f"cannot safely create directory {walked}: {exc}") from exc
            except OSError as exc:
                raise ContractError(f"cannot safely open directory {walked}: {exc}") from exc
            os.close(fd)
            fd = next_fd
            if not stat.S_ISDIR(os.fstat(fd).st_mode):
                raise ContractError(f"opened path component is not a directory: {walked}")
        return fd
    except Exception:
        os.close(fd)
        raise


def _open_file_no_follow(trusted_root: Path, path: Path) -> tuple[int, os.stat_result]:
    root, target, parts = _relative_parts(trusted_root, path)
    if not parts:
        raise ContractError(f"file path aliases trusted root: {target}")
    _validate_chain(root, target, "file")
    parent_fd = _open_directory_chain(root, target.parent, create=False)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(parts[-1], flags, dir_fd=parent_fd)
    except OSError as exc:
        os.close(parent_fd)
        raise ContractError(f"cannot safely open file {target}: {exc}") from exc
    os.close(parent_fd)
    observed = os.fstat(fd)
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        os.close(fd)
        raise ContractError(f"opened file is not regular single-link: {target}")
    return fd, observed


def _assert_file_identity(
    trusted_root: Path, path: Path, expected: FileIdentity,
) -> None:
    fd, observed = _open_file_no_follow(trusted_root, path)
    try:
        if _identity(observed) != expected:
            raise ContractError(f"file identity changed during validation: {path}")
    finally:
        os.close(fd)


def _read_file_no_follow(trusted_root: Path, path: Path) -> tuple[bytes, FileIdentity]:
    fd, before = _open_file_no_follow(trusted_root, path)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        if _identity(after) != _identity(before):
            raise ContractError(f"file changed while being read: {path}")
    finally:
        os.close(fd)
    identity = _identity(before)
    _assert_file_identity(trusted_root, path, identity)
    return b"".join(chunks), identity


def _sha_with_identity(
    path: Path, trusted_root: Path = ROOT
) -> tuple[str, FileIdentity]:
    data, identity = _read_file_no_follow(trusted_root, path)
    digest = hashlib.sha256(data).hexdigest()
    # The lexical chain and inode are checked once more after hashing, not merely after reading.
    _assert_file_identity(trusted_root, path, identity)
    return digest, identity


def _sha(path: Path, trusted_root: Path = ROOT) -> str:
    return _sha_with_identity(path, trusted_root)[0]


def _regular(path: Path, trusted_root: Path = ROOT) -> None:
    _validate_chain(trusted_root, path, "file")


def _load(path: Path, trusted_root: Path = ROOT) -> dict:
    data, _ = _read_file_no_follow(trusted_root, path)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, ValueError) as exc:
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
        "accepted_contract", "g0_prerequisite",
    }
    if set(template) != required:
        raise ContractError("G1 request-template schema mismatch")
    expected = {
        "schema": "a1-native-five-g1-request-template-v1",
        "state": "pending-external-finalization",
        "executable": False,
        "arm": "G1",
        "host": "trn21",
        "expected_hostname": "ip-172-31-34-85",
        "label": "fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1",
        "exact_symbolic_branch": EXPECTED_BRANCH,
        "remote": "/home/ubuntu/trainium_freshstart_20260924_v2_a1_native5_g1_hybrid_observation_on_trn21_r1",
        "origin": "https://github.com/CsabaConsulting/trainium_frontier.git",
        "purpose": "Human-approved G1-only A1 native five-update hybrid observation-on equivalence gate",
        "authorization_path": "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-authorization.json",
        "request_template_path": "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-request.template.json",
        "runtime_driver_path": "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1_finalization.py",
        "runtime_job_path": "research/run-logs/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1.job.json",
        "paths": EXPECTED_PATHS,
        "environment": EXPECTED_ENVIRONMENT,
        "child_environment": {
            **EXPECTED_ENVIRONMENT,
            "NEURON_COMPILE_CACHE_DIR": EXPECTED_PATHS["compile_cache"],
            "TRAINIUM_FRESH_START_RUN_ID": "fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1",
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
        "accepted_contract": {
            "r4_commit": CORE,
            "r4_comparator_sha256": R4_COMPARATOR_SHA,
            "ownership_manifest_sha256": "8119df41c8a9ecbfc3f9e69067c7e6ac162c2d5d8ba0a15635265304dcf865a2",
            "pass_receipt_commit": VERIFIER_COMMIT,
            "pass_receipt_sha256": VERIFIER_SHA,
            "analyst_commit": ANALYST_COMMIT,
            "analyst_checklist_sha256": ANALYST_SHA,
        },
        "g0_prerequisite": {
            "evidence_commit": G0_EVIDENCE_COMMIT,
            "corrected_replay_status": "PASS",
            "corrected_replay_sha256": G0_REPLAY_SHA,
            "corrected_replay_bytes": 326,
            "checkpoint_sha256": "2d0891c569e6046bd4a3635c62c1ebdb4c6ea6ed0d9788212910a26e61ad3b13",
            "checkpoint_bytes": 739_322_371,
        },
    }
    for key, value in expected.items():
        if template.get(key) != value:
            raise ContractError(f"G1 template {key} drift")
    if template["external_literals"] != {
        "request_bearing_head": "PENDING_EXTERNAL_VERIFIER",
        "request_sha256": "PENDING_EXTERNAL_VERIFIER",
        "authorization_sha256": "PENDING_EXTERNAL_VERIFIER",
    }:
        raise ContractError("G1 template must remain pending external literals")
    if template["finalization_paths"] != FINALIZATION_PATHS:
        raise ContractError("G1 finalization protected-path set drift")
    if template["source_sha256"] != EXPECTED_SOURCE_SHA256:
        raise ContractError("G1 frozen source hash map drift")
    if template["downstream_locks"] != {
        "G2": "PENDING_AUTHENTICATED_G1_ADJUDICATION_ARCHIVE_AND_NEW_EXTERNAL_AUTHORIZATION",
        "full_a1": False, "cleanup": False, "checkpoint_prune": False,
        "cache_delete": False, "submission": False,
    }:
        raise ContractError("post-G1 locks changed")
    if template["source"].endswith("train_gate_hybrid.py") is False:
        raise ContractError("G1 source is not the hybrid gate")
    argv = template["train_argv"]
    required_sequence = [
        "--num-steps", "5", "--max-train-seconds", "1800", "--n-embd", "1024",
        "--ngram-ve", "--ngram-ve-layers=-1", "--ngram-ve-table-mult", "64",
        "--ngram-ve-dim", "128", "--ngram-ve-trigram", "--ngram-ve-opt", "rmsprop",
        "--ngram-ve-flat-lr", "--ngram-ve-bf16", "--ngram-ve-clip-exclude",
        "--no-qk-shift", "--compile-sdpa-direct", "--nki-local-conv",
        "--bf16-norm-output", "--pack-factor", "32", "--clip-after-reduce",
        "--seq-len", "1024", "--no-eval-public", "--matrix-lr", "0.015",
        "--out-dir", template["paths"]["output"], "--native-gate-arm", "G1",
        "--native-gate-observation", "--native-gate-label", template["label"],
        "--native-gate-host", "trn21", "--native-gate-report-dir",
        template["paths"]["report"],
    ]
    if argv != required_sequence:
        raise ContractError("G1 train argv drift")
    if template["child_environment"].get("NEURON_COMPILE_CACHE_DIR") != template["paths"]["compile_cache"]:
        raise ContractError("compile cache is not exact")
    if template["environment"] != {
        key: value for key, value in template["child_environment"].items()
        if key not in {"NEURON_COMPILE_CACHE_DIR", "TRAINIUM_FRESH_START_RUN_ID"}
    }:
        raise ContractError("outer and child environment contract differs")
    lifecycle = template["lifecycle"]
    if lifecycle != {
        "one_run_per_host": True,
        "private_child_process_group": True,
        "bounded_term_kill_reap_quiescence": True,
        "hash_only_after_quiescence": True,
        "no_cleanup_or_prune": True,
        "checkpoint_prune_authorized": False,
        "cache_cleanup_authorized": False,
    }:
        raise ContractError("G1 lifecycle contract drift")


def validate_authorization(auth: dict, template: dict, request_sha: str) -> None:
    required = {
        "schema", "state", "executable", "arm", "host", "label",
        "verified_core_commit", "verifier_receipt_commit", "verifier_receipt_sha256",
        "request_template_path", "request_sha256", "head_binding",
        "protected_finalization_sha256", "source_sha256", "native_execution_authorized",
        "g1_authorized", "g2_authorized", "checkpoint_prune_authorized",
        "cache_cleanup_authorized", "full_a1_authorized", "submission_authorized",
        "g0_prerequisite",
    }
    if set(auth) != required:
        raise ContractError("authorization schema mismatch")
    expected = {
        "schema": "a1-native-five-g1-authorization-v1",
        "state": "independent-verifier-finalized-g1-only",
        "executable": True,
        "arm": "G1", "host": "trn21", "label": template["label"],
        "verified_core_commit": CORE,
        "verifier_receipt_commit": VERIFIER_COMMIT,
        "verifier_receipt_sha256": VERIFIER_SHA,
        "request_template_path": template["request_template_path"],
        "request_sha256": request_sha,
        "head_binding": "external-literals-only",
        "native_execution_authorized": True,
        "g1_authorized": True, "g2_authorized": False,
        "checkpoint_prune_authorized": False, "cache_cleanup_authorized": False,
        "full_a1_authorized": False, "submission_authorized": False,
        "g0_prerequisite": template["g0_prerequisite"],
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
        "schema": "a1-native-five-g1-concrete-request-v1",
        "arm": "G1", "host": "trn21", "expected_hostname": template["expected_hostname"],
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


def _safe_mkdir(trusted_root: Path, path: Path) -> None:
    root, target, parts = _relative_parts(trusted_root, path)
    if not parts:
        raise ContractError(f"directory terminal aliases trusted root: {target}")
    _validate_chain(root, target, "absent")
    parent_fd = _open_directory_chain(root, target.parent, create=True)
    try:
        os.mkdir(parts[-1], 0o700, dir_fd=parent_fd)
    except OSError as exc:
        raise ContractError(f"cannot safely create exclusive directory {target}: {exc}") from exc
    finally:
        os.close(parent_fd)
    _validate_chain(root, target, "directory")


def _directory_identity(
    trusted_root: Path, path: Path,
) -> DirectoryIdentity:
    _validate_chain(trusted_root, path, "directory")
    fd = _open_directory_chain(trusted_root, path, create=False)
    try:
        observed = os.fstat(fd)
        if not stat.S_ISDIR(observed.st_mode):
            raise ContractError(f"opened terminal is not a directory: {path}")
        # Directory size, mtime and link count legitimately change when the child creates
        # compiler entries, final.pt or rank reports. Device/inode/type remain the stable,
        # no-follow identity of the exact directory created during preflight.
        identity = (observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode))
    finally:
        os.close(fd)
    _validate_chain(trusted_root, path, "directory")
    return identity


def _assert_directory_identity(
    trusted_root: Path, path: Path, expected: DirectoryIdentity,
) -> None:
    if _directory_identity(trusted_root, path) != expected:
        raise ContractError(f"directory identity changed during validation: {path}")


def _open_exclusive(trusted_root: Path, path: Path) -> int:
    root, target, parts = _relative_parts(trusted_root, path)
    if not parts:
        raise ContractError(f"file terminal aliases trusted root: {target}")
    _validate_chain(root, target, "absent")
    parent_fd = _open_directory_chain(root, target.parent, create=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(parts[-1], flags, 0o600, dir_fd=parent_fd)
    except OSError as exc:
        raise ContractError(f"cannot safely create exclusive file {target}: {exc}") from exc
    finally:
        os.close(parent_fd)
    opened = os.fstat(fd)
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
        os.close(fd)
        raise ContractError(f"new file is not regular single-link: {target}")
    return fd


def _assert_sealed_file(trusted_root: Path, path: Path, sealed: SealedWrite) -> None:
    digest, identity = _sha_with_identity(path, trusted_root)
    if identity != sealed.identity:
        raise ContractError(f"exclusive output inode changed after creation: {path}")
    if digest != sealed.sha256 or identity[4] != sealed.bytes:
        raise ContractError(f"exclusive output content changed after creation: {path}")


def _write_exclusive(trusted_root: Path, path: Path, payload: dict) -> SealedWrite:
    data = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    expected_digest = hashlib.sha256(data).hexdigest()
    fd = _open_exclusive(trusted_root, path)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ContractError(f"short write while creating {path}")
            view = view[written:]
        os.fsync(fd)
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise ContractError(f"exclusive output lost its regular single-link inode: {path}")
        if observed.st_size != len(data):
            raise ContractError(f"exclusive output has an unexpected size: {path}")
        sealed = SealedWrite(
            sha256=expected_digest, identity=_identity(observed), bytes=len(data),
        )
    finally:
        os.close(fd)
    # The path must still name the exact inode whose fd was written and fsynced, and those
    # bytes—not a same-shaped replacement—must remain at that name through successful return.
    _assert_sealed_file(trusted_root, path, sealed)
    return sealed


def materialize(args: argparse.Namespace) -> int:
    pending_template = _load(TEMPLATE_PATH)
    if args.request_template != pending_template["request_template_path"]:
        raise ContractError("request-template path differs from fixed contract")
    if args.authorization != pending_template["authorization_path"]:
        raise ContractError("authorization path differs from fixed contract")
    template_path = ROOT / args.request_template
    auth_path = ROOT / args.authorization
    _relative_parts(ROOT, template_path)
    _relative_parts(ROOT, auth_path)
    if args.output != MATERIALIZATION_OUTPUT:
        raise ContractError(f"materialization output differs from exact fixed path: {args.output}")
    _validate_chain(MATERIALIZATION_ROOT, args.output, "absent")
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
    # Revalidate all consumed authority immediately before the only write.
    if _sha(template_path) != args.request_sha256 or _sha(auth_path) != args.authorization_sha256:
        raise ContractError("materialization authority changed before output creation")
    _validate_chain(MATERIALIZATION_ROOT, args.output, "absent")
    sealed_output = _write_exclusive(MATERIALIZATION_ROOT, args.output, concrete)
    print(json.dumps({"status": "MATERIALIZED_NOT_LAUNCHED", "output": str(args.output),
                      "sha256": sealed_output.sha256}, sort_keys=True))
    _assert_sealed_file(MATERIALIZATION_ROOT, args.output, sealed_output)
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


def _directory_files_no_follow(trusted_root: Path, directory: Path) -> list[Path]:
    """Enumerate a directory through pinned dirfds, rejecting every non-regular leaf."""
    root, target, _ = _relative_parts(trusted_root, directory)
    _validate_chain(root, target, "directory")
    start_fd = _open_directory_chain(root, target, create=False)
    files: list[Path] = []

    def walk(fd: int, lexical: Path) -> None:
        try:
            entries = sorted(os.scandir(fd), key=lambda entry: entry.name)
        except OSError as exc:
            raise ContractError(f"cannot safely enumerate artifact directory {lexical}: {exc}") from exc
        for entry in entries:
            child = lexical / entry.name
            try:
                observed = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ContractError(f"cannot lstat artifact {child}: {exc}") from exc
            if stat.S_ISLNK(observed.st_mode):
                raise ContractError(f"artifact symlink rejected: {child}")
            if stat.S_ISDIR(observed.st_mode):
                flags = os.O_RDONLY | os.O_DIRECTORY
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                try:
                    child_fd = os.open(entry.name, flags, dir_fd=fd)
                except OSError as exc:
                    raise ContractError(f"cannot safely open artifact directory {child}: {exc}") from exc
                try:
                    walk(child_fd, child)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
                files.append(child)
            else:
                raise ContractError(f"artifact is not a regular single-link file: {child}")

    try:
        walk(start_fd, target)
    finally:
        os.close(start_fd)
    _validate_chain(root, target, "directory")
    return files


def _artifact_snapshot(
    paths: list[Path], trusted_root: Path = ROOT,
) -> tuple[dict[str, dict], dict[Path, FileIdentity]]:
    rows: dict[str, dict] = {}
    identities: dict[Path, FileIdentity] = {}
    for artifact_root in paths:
        root_stat = _validate_chain(trusted_root, artifact_root, "existing")
        if root_stat is not None and stat.S_ISREG(root_stat.st_mode):
            files = [artifact_root]
        else:
            # Retry as a directory; the file validation above can fail only on its type.
            files = _directory_files_no_follow(trusted_root, artifact_root)
        if not files:
            raise ContractError(f"required artifact path is empty: {artifact_root}")
        for path in files:
            digest, identity = _sha_with_identity(path, trusted_root)
            rows[str(path.relative_to(trusted_root))] = {
                "bytes": identity[4], "sha256": digest, "mtime_ns": identity[5],
            }
            identities[path] = identity
        # This explicitly rechecks each lexical root after all hashes below it are complete.
        terminal = "file" if len(files) == 1 and files[0] == artifact_root else "directory"
        _validate_chain(trusted_root, artifact_root, terminal)
    _revalidate_snapshot(trusted_root, identities)
    return rows, identities


def _revalidate_snapshot(
    trusted_root: Path,
    identities: dict[Path, FileIdentity],
) -> None:
    for path, identity in identities.items():
        _assert_file_identity(trusted_root, path, identity)


def _immediate_entry_schema_no_follow(
    trusted_root: Path, directory: Path,
) -> list[tuple[str, str]]:
    """Return exact immediate names/types; never omit directories or follow aliases."""
    _validate_chain(trusted_root, directory, "directory")
    fd = _open_directory_chain(trusted_root, directory, create=False)
    rows: list[tuple[str, str]] = []
    try:
        try:
            entries = sorted(os.scandir(fd), key=lambda entry: entry.name)
        except OSError as exc:
            raise ContractError(f"cannot enumerate exact artifact membership {directory}: {exc}") from exc
        for entry in entries:
            try:
                observed = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ContractError(f"cannot lstat artifact member {directory / entry.name}: {exc}") from exc
            if stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
                kind = "file"
            elif stat.S_ISDIR(observed.st_mode):
                kind = "directory"
            elif stat.S_ISLNK(observed.st_mode):
                kind = "symlink"
            else:
                kind = "other"
            rows.append((entry.name, kind))
    finally:
        os.close(fd)
    _validate_chain(trusted_root, directory, "directory")
    return rows


def _validate_success_artifact_allowlist(
    output: Path, report: Path, trusted_root: Path = ROOT,
) -> None:
    if _immediate_entry_schema_no_follow(trusted_root, output) != [("final.pt", "file")]:
        raise ContractError("G1 output directory artifact allowlist is not exact")
    expected_reports = [(f"rank-{rank}.jsonl", "file") for rank in range(4)]
    if _immediate_entry_schema_no_follow(trusted_root, report) != expected_reports:
        raise ContractError("G1 rank-report artifact allowlist is not exact")


def _immediate_file_identities_no_follow(
    trusted_root: Path, directory: Path,
) -> dict[str, FileIdentity]:
    """Bind every immediate member name to its regular single-link inode identity."""
    _validate_chain(trusted_root, directory, "directory")
    fd = _open_directory_chain(trusted_root, directory, create=False)
    rows: dict[str, FileIdentity] = {}
    try:
        entries = sorted(os.scandir(fd), key=lambda entry: entry.name)
        for entry in entries:
            observed = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise ContractError(
                    f"final artifact member is not regular single-link: {directory / entry.name}"
                )
            rows[entry.name] = _identity(observed)
    except OSError as exc:
        raise ContractError(f"cannot bind final artifact membership {directory}: {exc}") from exc
    finally:
        os.close(fd)
    _validate_chain(trusted_root, directory, "directory")
    return rows


def _validate_success_artifact_bindings(
    output: Path, report: Path, identities: dict[Path, FileIdentity],
    trusted_root: Path = ROOT,
) -> None:
    expected = {
        output: [output / "final.pt"],
        report: [report / f"rank-{rank}.jsonl" for rank in range(4)],
    }
    for directory, paths in expected.items():
        observed = _immediate_file_identities_no_follow(trusted_root, directory)
        if list(observed) != [path.name for path in paths]:
            raise ContractError(f"final artifact membership changed: {directory}")
        for path in paths:
            if identities.get(path) != observed[path.name]:
                raise ContractError(f"final artifact inode changed after hashing: {path}")


def _final_success_artifact_seal(
    output: Path, report: Path, identities: dict[Path, FileIdentity],
    trusted_root: Path = ROOT,
) -> None:
    # Keep these three checks adjacent: names/types, all hashed inode identities, then the
    # exact name-to-inode binding. The caller writes the sealed result immediately afterward.
    _validate_success_artifact_allowlist(output, report, trusted_root)
    _revalidate_snapshot(trusted_root, identities)
    _validate_success_artifact_bindings(output, report, identities, trusted_root)


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


def _revalidate_authority(
    template_path: Path, auth_path: Path, template: dict, auth: dict,
    request_sha: str, authorization_sha: str,
) -> None:
    if _sha(template_path) != request_sha:
        raise ContractError("request template changed during runtime validation")
    if _sha(auth_path) != authorization_sha:
        raise ContractError("authorization changed during runtime validation")
    validate_authorization(auth, template, request_sha)


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
    runtime_path = ROOT / args.runtime_job
    runtime = _load(runtime_path)
    _validate_runtime_job(runtime, concrete, template, auth)
    runtime_job_sha, runtime_job_identity = _sha_with_identity(runtime_path)
    if os.environ.get("TRAINIUM_FRESH_START_RUN_ID") != template["label"]:
        raise ContractError("fresh-start ownership marker missing or changed")

    paths = {key: Path(value) for key, value in template["paths"].items()}
    rooted = {key: ROOT / paths[key] for key in ("output", "report", "child_log", "runtime_result")}
    for key in ("output", "report", "child_log", "runtime_result"):
        _validate_chain(ROOT, rooted[key], "absent")
    cache = Path(template["paths"]["compile_cache"])
    _validate_chain(COMPILE_CACHE_ROOT, cache, "absent")
    free = shutil.disk_usage(ROOT).free
    if free < template["hard_abort_bytes"]:
        raise ContractError("hard abort: less than 5 GiB free")
    if free < template["min_free_bytes"]:
        raise ContractError("launch blocked: less than 6.5 GiB free")
    _safe_mkdir(COMPILE_CACHE_ROOT, cache)
    _safe_mkdir(ROOT, rooted["output"])
    _safe_mkdir(ROOT, rooted["report"])
    directory_identities = {
        cache: _directory_identity(COMPILE_CACHE_ROOT, cache),
        rooted["output"]: _directory_identity(ROOT, rooted["output"]),
        rooted["report"]: _directory_identity(ROOT, rooted["report"]),
    }

    child_command = [
        "/home/ubuntu/workspace/native_venv/bin/torchrun", "--standalone", "--nproc_per_node=4",
        template["source"], *template["train_argv"],
    ]
    log_path = rooted["child_log"]
    log_fd = _open_exclusive(ROOT, log_path)
    log_identity = _identity(os.fstat(log_fd))
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
        # Final no-follow gate immediately before Popen. Nothing below may silently recreate paths.
        _revalidate_authority(
            template_path, auth_path, template, auth,
            args.request_sha256, args.authorization_sha256,
        )
        _assert_file_identity(ROOT, runtime_path, runtime_job_identity)
        _assert_file_identity(ROOT, log_path, log_identity)
        _assert_directory_identity(COMPILE_CACHE_ROOT, cache, directory_identities[cache])
        _assert_directory_identity(ROOT, rooted["output"], directory_identities[rooted["output"]])
        _assert_directory_identity(ROOT, rooted["report"], directory_identities[rooted["report"]])
        _validate_chain(ROOT, rooted["runtime_result"], "absent")
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
        raise ContractError("G1 child did not start")
    if _group_members(process.pid):
        raise ContractError("G1 child process group is not quiescent")

    success = returncode == 0 and not timed_out and not interrupted
    artifact_paths = [log_path]
    if success:
        _validate_success_artifact_allowlist(rooted["output"], rooted["report"])
        artifact_paths.extend([
            rooted["output"] / "final.pt", rooted["report"],
        ])
    artifacts, artifact_identities = _artifact_snapshot(artifact_paths)
    if success:
        _validate_success_artifact_allowlist(rooted["output"], rooted["report"])
    result = {
        "schema": "a1-native-five-g1-runtime-result-v1", "arm": "G1",
        "status": "PASS" if success else "FAILED", "complete": success,
        "label": template["label"], "host": "trn21", "authorized_head": args.authorized_head,
        "request_sha256": args.request_sha256,
        "authorization_sha256": args.authorization_sha256,
        "runtime_job_sha256": runtime_job_sha, "command": child_command,
        "environment": template["child_environment"], "returncode": returncode,
        "timed_out": timed_out, "interrupted_signals": interrupted,
        "duration_seconds": time.monotonic() - started, "process_group_actions": actions,
        "process_group_quiescent": True, "artifacts": artifacts,
        "compile_cache": {"path": str(cache), "preserved": True},
        "checkpoint_pruned": False, "cache_cleaned": False,
    }
    # No artifact or authority may change between hashing and the top-level result write.
    _revalidate_snapshot(ROOT, artifact_identities)
    _revalidate_authority(
        template_path, auth_path, template, auth,
        args.request_sha256, args.authorization_sha256,
    )
    _assert_file_identity(ROOT, runtime_path, runtime_job_identity)
    _assert_directory_identity(COMPILE_CACHE_ROOT, cache, directory_identities[cache])
    _assert_directory_identity(ROOT, rooted["output"], directory_identities[rooted["output"]])
    _assert_directory_identity(ROOT, rooted["report"], directory_identities[rooted["report"]])
    _validate_chain(ROOT, rooted["runtime_result"], "absent")
    if success:
        _final_success_artifact_seal(
            rooted["output"], rooted["report"], artifact_identities,
        )
    sealed_result = _write_exclusive(ROOT, rooted["runtime_result"], result)
    _assert_sealed_file(ROOT, rooted["runtime_result"], sealed_result)
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
        print(f"G1_FINALIZATION_INVALID: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
