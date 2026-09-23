#!/usr/bin/env python3
"""Fail-closed full-budget c595 WARMDOWN_RATIO=0.60 train/eval driver."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PYTHON = "/home/ubuntu/workspace/native_venv/bin/python"
LABEL = "fs-20260924-c595-warmdown060-trn22-r3"
HOST = "trn22"
HOSTNAME = "ip-172-31-32-253"
REMOTE = "/home/ubuntu/trainium_freshstart_20260924_c595_warmdown060_trn22_r3"
CONCRETE_REQUEST = Path("/tmp/fs-20260924-c595-warmdown060-trn22-r3.request.json")
PACKAGE_REL = "submissions/dense6x1024_direct_review_warmdown060"
SOURCE_REL = "submissions/dense6x1024_direct_review/train.py"
SOURCE_SHA = "c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537"
CANDIDATE_SHA = "52cf99df3e241ab8399c104b7ad4b717571e0987781b6b835303a8500baff38c"
PREPARE_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
LAUNCH_SHA = "b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd"
CONTROL_BPB = Decimal("0.993593002710418")
KEEP_DELTA = Decimal("0.0015")
REPLICATE_BELOW_DELTA = Decimal("0.0025")
OUTPUT = ROOT / "out" / LABEL
EVIDENCE = OUTPUT / "evidence"
METRIC_KEYS = {
    "val_bpb", "num_tokens", "num_bytes", "total_nats",
    "eval_seconds", "eval_timeout", "eval_over_timeout",
}
BASE_ARTIFACTS = {
    "training.log", "training-process.json", "training-rehearsal.json",
    "checkpoint-finite.log", "checkpoint-finite-process.json", "checkpoint-finite.json",
    "fresh-public20m.log", "fresh-public20m-process.json", "fresh-public20m.metrics.json",
    "evaluation-rehearsal.json", "result.json",
}
HEX64 = set("0123456789abcdef")
EXACT_CC_FLAGS = "--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
PINNED_LD_LIBRARY_PATH = "/usr/local/lib:/usr/lib"
PINNED_LD_LIBRARY_PATH_SHA256 = "ac04b79dfecb27e2f78a97194a6d6cc7158f45a16ab3544df4c2b0ee5be3d3c0"
AUTH_REL = "research/submission-validation/20260924-c595-warmdown060-trn22/authorization.json"
REQUEST_REL = "research/submission-validation/20260924-c595-warmdown060-trn22/final-request.json"
DRIVER_REL = "research/submission-validation/20260924-c595-warmdown060-trn22/retune_job.py"
ADAPTER_REL = "research/submission-validation/20260924-c595-warmdown060-trn22/launch_adapter.py"
PROPOSAL_REL = "research/submission-validation/20260924-c595-warmdown060-trn22/EXPERIMENT_PROPOSAL.json"
RUNTIME_JOB_REL = f"research/run-logs/{LABEL}.job.json"
DISPATCH_REL = f"research/run-logs/{LABEL}.dispatch.json"
LAUNCHER_SOURCE_PREFIX = (
    "train.py",
    "prepare.py",
    "research/run_experiment.py",
    "research/fresh_start_host.py",
    "research/fresh_start_guard.py",
    "research/fresh_start_backup.py",
)
RUNTIME_DYNAMIC_KEYS = (
    "request_file_sha256",
    "started_at",
    "hostname",
    "git_commit",
    "source_dirty",
    "preflight_free_bytes",
    "source_sha256",
)
CORE_FINGERPRINT_PATHS = {
    "train.py",
    "prepare.py",
    "research/run_experiment.py",
    "research/scratch/submission_rehearsal.py",
    "research/fresh_start_launch.py",
    "research/fresh_start_sync.py",
    "research/fresh_start_guard.py",
    "research/fresh_start_host.py",
    "research/fresh_start_backup.py",
    PROPOSAL_REL,
    DRIVER_REL,
    ADAPTER_REL,
    SOURCE_REL,
    f"{PACKAGE_REL}/train.py",
    f"{PACKAGE_REL}/provenance.json",
    f"{PACKAGE_REL}/launch-command.txt",
}
MINIMAL_ENVIRONMENT = {
    "HOME": "/home/ubuntu",
    "USER": "ubuntu",
    "LOGNAME": "ubuntu",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LD_LIBRARY_PATH": PINNED_LD_LIBRARY_PATH,
    "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
    "TMPDIR": "/tmp",
    "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv",
    "NEURON_LOGICAL_NC_CONFIG": "2",
    "NEURON_CC_FLAGS": EXACT_CC_FLAGS,
    "PYTHONUNBUFFERED": "1",
    "TRAINIUM_FRESH_START_RUN_ID": LABEL,
}


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_directory(path: Path) -> int:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SystemExit(f"unsafe directory: {path}")
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def create_output(root: Path = ROOT, label: str = LABEL) -> tuple[Path, Path]:
    if not root.is_absolute() or not label or "/" in label or label in {".", ".."}:
        raise SystemExit("unsafe output identity")
    root_fd = _open_directory(root)
    try:
        try:
            os.mkdir("out", 0o700, dir_fd=root_fd)
            os.fsync(root_fd)
        except FileExistsError:
            pass
        out_fd = os.open("out", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            out_info = os.stat("out", dir_fd=root_fd, follow_symlinks=False)
            if not stat.S_ISDIR(out_info.st_mode):
                raise SystemExit("unsafe output parent")
            try:
                os.mkdir(label, 0o700, dir_fd=out_fd)
                os.fsync(out_fd)
            except FileExistsError:
                raise SystemExit("run output already exists")
            label_fd = os.open(label, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=out_fd)
            try:
                os.mkdir("evidence", 0o700, dir_fd=label_fd)
                os.fsync(label_fd)
            finally:
                os.close(label_fd)
        finally:
            os.close(out_fd)
    finally:
        os.close(root_fd)
    return root / "out" / label, root / "out" / label / "evidence"


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        count = os.write(fd, view[offset:])
        if type(count) is not int or count <= 0 or count > len(view) - offset:
            raise SystemExit("invalid durable write count")
        offset += count


def durable(evidence: Path, name: str, data: bytes) -> None:
    if not name or "/" in name or name in {".", ".."}:
        raise SystemExit("artifact path escape")
    directory = _open_directory(evidence)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            try:
                _write_all(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException:
            os.unlink(name, dir_fd=directory)
            os.fsync(directory)
            raise
        os.fsync(directory)
    finally:
        os.close(directory)


def durable_json(evidence: Path, name: str, value: object) -> None:
    durable(evidence, name, canonical_json(value))


def artifact_bytes(evidence: Path, name: str) -> bytes:
    if not name or "/" in name or name in {".", ".."}:
        raise SystemExit("artifact path escape")
    directory = _open_directory(evidence)
    try:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise SystemExit(f"artifact not regular: {name}")
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != info.st_size:
                raise SystemExit(f"artifact changed while opening: {name}")
            chunks = []
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            data = b"".join(chunks)
            if len(data) != info.st_size:
                raise SystemExit(f"artifact changed while reading: {name}")
            return data
        finally:
            os.close(fd)
    finally:
        os.close(directory)


def persisted_json(evidence: Path, name: str, expected: dict, schema: str) -> dict:
    data = artifact_bytes(evidence, name)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"persisted JSON invalid: {name}") from error
    if value != expected or value.get("schema") != schema or data != canonical_json(expected):
        raise SystemExit(f"persisted JSON schema/content mismatch: {name}")
    return value


def inspect_artifacts(evidence: Path, expected: set[str]) -> dict[str, dict[str, object]]:
    directory = _open_directory(evidence)
    try:
        names = set(os.listdir(directory))
    finally:
        os.close(directory)
    if names != expected:
        raise SystemExit("artifact allowlist mismatch")
    result = {}
    for name in sorted(names):
        data = artifact_bytes(evidence, name)
        result[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return result


def revalidate_members(evidence: Path, members: dict[str, dict[str, object]]) -> None:
    for name, identity in members.items():
        data = artifact_bytes(evidence, name)
        if identity != {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}:
            raise SystemExit(f"manifest member changed: {name}")


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_group(process: subprocess.Popen[bytes]) -> None:
    pgid = process.pid
    if _group_alive(pgid):
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + 15.0
        while _group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
    if _group_alive(pgid):
        os.killpg(pgid, signal.SIGKILL)
        deadline = time.monotonic() + 15.0
        while _group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    if _group_alive(pgid):
        raise SystemExit("child process group did not quiesce")


def run_process(
    argv: list[str], env: dict[str, str], timeout: int, evidence: Path, stem: str,
) -> tuple[bytes, dict]:
    started = time.monotonic()
    process = subprocess.Popen(
        argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop_group(process)
        output, _ = process.communicate()
    descendants_after_parent = _group_alive(process.pid)
    if descendants_after_parent:
        _stop_group(process)
    elapsed = time.monotonic() - started
    receipt = {
        "schema": "c595-warmdown060-process-v1",
        "label": LABEL,
        "name": stem,
        "argv": argv,
        "timeout_seconds": timeout,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "duration_seconds": elapsed,
        "process_group_quiescent": not _group_alive(process.pid),
        "descendants_after_parent_exit": descendants_after_parent,
        "log_sha256": hashlib.sha256(output).hexdigest(),
        "log_bytes": len(output),
    }
    durable(evidence, f"{stem}.log", output)
    durable_json(evidence, f"{stem}-process.json", receipt)
    persisted_json(evidence, f"{stem}-process.json", receipt, "c595-warmdown060-process-v1")
    if timed_out or process.returncode != 0 or descendants_after_parent or not receipt["process_group_quiescent"]:
        raise SystemExit(f"{stem} failed, timed out, or left descendants")
    return output, receipt


def _json_objects(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    found = []
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            found.append(value)
    return found


def parse_finite_receipt(output: str) -> dict:
    matches = [value for value in _json_objects(output) if set(value).intersection({"weights_only", "finite", "tensor_count"})]
    if len(matches) != 1:
        raise SystemExit("finite receipt count mismatch")
    value = matches[0]
    expected_keys = {"weights_only", "map_location", "finite", "tensor_count"}
    if set(value) != expected_keys:
        raise SystemExit("finite receipt schema mismatch")
    if value["weights_only"] is not True or value["map_location"] != "cpu" or value["finite"] is not True:
        raise SystemExit("finite receipt contract mismatch")
    if type(value["tensor_count"]) is not int or value["tensor_count"] <= 0:
        raise SystemExit("finite receipt tensor count invalid")
    return value


def parse_metrics(output: str) -> dict:
    matches = [value for value in _json_objects(output) if METRIC_KEYS.intersection(value)]
    if len(matches) != 1 or set(matches[0]) != METRIC_KEYS:
        raise SystemExit("evaluation metric schema/count mismatch")
    value = matches[0]
    for key in ("val_bpb", "total_nats", "eval_seconds"):
        if type(value[key]) is not float or not math.isfinite(value[key]) or value[key] <= 0:
            raise SystemExit(f"invalid {key}")
    if type(value["num_tokens"]) is not int or value["num_tokens"] != 20971520:
        raise SystemExit("token count mismatch")
    if type(value["num_bytes"]) is not int or value["num_bytes"] != 83462186:
        raise SystemExit("byte count mismatch")
    if type(value["eval_timeout"]) is not int or value["eval_timeout"] != 1200:
        raise SystemExit("eval timeout mismatch")
    if value["eval_over_timeout"] is not False or value["eval_seconds"] > 1200.0:
        raise SystemExit("evaluation exceeded timeout")
    derived = value["total_nats"] / (value["num_bytes"] * math.log(2.0))
    if not math.isclose(value["val_bpb"], derived, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit("BPB math mismatch")
    return value


def decide_bpb(candidate_bpb: str) -> dict[str, object]:
    try:
        candidate = Decimal(candidate_bpb)
    except InvalidOperation as error:
        raise SystemExit("candidate BPB is not decimal") from error
    if not candidate.is_finite() or candidate <= 0:
        raise SystemExit("candidate BPB is nonfinite or nonpositive")
    improvement = CONTROL_BPB - candidate
    keep = improvement >= KEEP_DELTA
    return {
        "candidate_bpb_decimal": str(candidate),
        "control_bpb_decimal": str(CONTROL_BPB),
        "improvement_decimal": str(improvement),
        "decision": "KEEP" if keep else "DISCARD",
        "replication_required": keep and improvement < REPLICATE_BELOW_DELTA,
    }


def _safe_regular(path: Path) -> os.stat_result:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise SystemExit(f"symlink rejected: {current}")
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"not a regular file: {path}")
    return info


def _nofollow_identity(path: Path) -> tuple[str, int]:
    _safe_regular(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise SystemExit("opened checkpoint is not regular")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        if size != opened.st_size:
            raise SystemExit("checkpoint changed while hashing")
        return digest.hexdigest(), size
    finally:
        os.close(fd)


def _nofollow_bytes(path: Path) -> bytes:
    _safe_regular(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise SystemExit("opened path is not regular")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        data = b"".join(chunks)
        if len(data) != opened.st_size:
            raise SystemExit("path changed while reading")
        return data
    finally:
        os.close(fd)


def _nofollow_json(path: Path) -> dict:
    try:
        value = json.loads(_nofollow_bytes(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"generated JSON invalid: {path}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"generated JSON is not an object: {path}")
    return value


def build_subprocess_environment(source: dict[str, str]) -> dict[str, str]:
    allowed_sensitive = {
        "LD_LIBRARY_PATH": PINNED_LD_LIBRARY_PATH,
        "NEURON_LOGICAL_NC_CONFIG": "2",
        "NEURON_CC_FLAGS": EXACT_CC_FLAGS,
        "TRAINIUM_FRESH_START_RUN_ID": LABEL,
    }
    for key, value in source.items():
        sensitive = key.startswith(("LD_", "XLA_", "NEURON_", "TORCH_"))
        if sensitive and (key not in allowed_sensitive or value != allowed_sensitive[key]):
            raise SystemExit(f"hostile or unpinned inherited environment: {key}")
    for key, value in allowed_sensitive.items():
        if source.get(key) != value:
            raise SystemExit(f"required outer environment mismatch: {key}")
    return dict(MINIMAL_ENVIRONMENT)


def validate_candidate() -> None:
    source_path = ROOT / SOURCE_REL
    package = ROOT / PACKAGE_REL
    candidate_path = package / "train.py"
    if sha256(source_path) != SOURCE_SHA or sha256(candidate_path) != CANDIDATE_SHA:
        raise SystemExit("source or candidate hash mismatch")
    source = source_path.read_text()
    candidate = candidate_path.read_text()
    if candidate.replace("WARMDOWN_RATIO = 0.60", "WARMDOWN_RATIO = 0.75") != source:
        raise SystemExit("candidate is not the exact warmdown one-line change")
    for exact in (
        "WARMDOWN_RATIO = 0.60", "WEIGHT_DECAY = 0.2", "MAX_TRAIN_SECONDS = 1800",
        "default=int(os.environ.get('NEURON_COMPETITION_R1_SEED', '42'))",
    ):
        if candidate.count(exact) != 1:
            raise SystemExit(f"candidate config mismatch: {exact}")
    if sha256(ROOT / "prepare.py") != PREPARE_SHA:
        raise SystemExit("prepare.py hash mismatch")
    for launch in (ROOT / "submissions/dense6x1024_direct_review/launch-command.txt", package / "launch-command.txt"):
        if sha256(launch) != LAUNCH_SHA:
            raise SystemExit("launch command hash mismatch")
    provenance = json.loads((package / "provenance.json").read_text())
    if provenance.get("train_sha256") != CANDIDATE_SHA or provenance.get("derived_from", {}).get("train_py_sha256") != SOURCE_SHA:
        raise SystemExit("candidate provenance mismatch")


def _lower_sha(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in HEX64 for character in value):
        raise SystemExit(f"invalid {name}")
    return value


def validate_authorization_request(auth: dict, request: dict, verified_core: str) -> str:
    request_sha = hashlib.sha256(canonical_json(request)).hexdigest()
    expected_auth_keys = {
        "schema", "arm", "host", "expected_hostname", "label", "package",
        "base_train_sha256", "train_sha256", "single_causal_change", "unchanged_weight_decay",
        "seed", "max_train_seconds", "smoke_authorized", "training_authorized",
        "evaluation_authorized", "checkpoint_prune_authorized", "verified_core_commit",
        "request_path", "request_sha256", "request_content", "state",
    }
    if set(auth) != expected_auth_keys:
        raise SystemExit("authorization schema mismatch")
    expected_auth = {
        "schema": "c595-warmdown060-authorization-v2",
        "arm": "candidate", "host": HOST, "expected_hostname": HOSTNAME, "label": LABEL,
        "package": PACKAGE_REL, "base_train_sha256": SOURCE_SHA, "train_sha256": CANDIDATE_SHA,
        "single_causal_change": "WARMDOWN_RATIO 0.75 -> 0.60", "unchanged_weight_decay": 0.2,
        "seed": 42, "max_train_seconds": 1800, "smoke_authorized": False,
        "training_authorized": True, "evaluation_authorized": True,
        "checkpoint_prune_authorized": False, "verified_core_commit": verified_core,
        "request_path": REQUEST_REL, "request_sha256": request_sha, "request_content": request,
        "state": "human_approved_pending_operator",
    }
    if auth != expected_auth:
        raise SystemExit("authorization content mismatch")
    expected_request_keys = {
        "schema", "host", "expected_hostname", "remote", "label", "purpose",
        "verified_core_commit", "authorization_path", "request_path", "source_sha256",
        "environment", "command", "runtime_job_schema", "runner_owns_manifest", "non_overwrite",
    }
    if set(request) != expected_request_keys:
        raise SystemExit("final request schema mismatch")
    if (
        request["schema"] != "c595-warmdown060-final-request-v1"
        or request["host"] != HOST
        or request["expected_hostname"] != HOSTNAME
        or request["remote"] != REMOTE
        or request["label"] != LABEL
        or request["verified_core_commit"] != verified_core
        or request["authorization_path"] != AUTH_REL
        or request["request_path"] != REQUEST_REL
        or request["runtime_job_schema"] != "c595-warmdown060-runtime-job-v1"
        or request["runner_owns_manifest"] is not False
        or request["non_overwrite"] is not True
        or request["environment"] != MINIMAL_ENVIRONMENT
        or set(request["source_sha256"]) != CORE_FINGERPRINT_PATHS
    ):
        raise SystemExit("final request content mismatch")
    expected_command = [
        "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "4200s", PYTHON,
        DRIVER_REL, "--authorization", AUTH_REL, "--request", REQUEST_REL,
        "--runtime-job", RUNTIME_JOB_REL,
    ]
    if request["command"] != expected_command:
        raise SystemExit("final request command mismatch")
    for path, digest in request["source_sha256"].items():
        _lower_sha(digest, f"source digest {path}")
    return request_sha


def build_concrete_request(
    template: dict, authorization_sha: str, template_sha: str, finalization_head: str,
) -> dict:
    _lower_sha(authorization_sha, "authorization sha")
    _lower_sha(template_sha, "template sha")
    if type(finalization_head) is not str or len(finalization_head) != 40 or any(c not in HEX64 for c in finalization_head):
        raise SystemExit("invalid finalization commit")
    expected_sha = {
        **template["source_sha256"],
        AUTH_REL: authorization_sha,
        REQUEST_REL: template_sha,
    }
    return {
        "authorization_path": AUTH_REL,
        "authorization_sha256": authorization_sha,
        "command": template["command"],
        "environment": template["environment"],
        "expected_commit": finalization_head,
        "expected_hostname": HOSTNAME,
        "expected_sha256": expected_sha,
        "finalization_head": finalization_head,
        "fingerprint_paths": sorted(expected_sha),
        "host": HOST,
        "label": LABEL,
        "non_overwrite": True,
        "origin": "https://github.com/CsabaConsulting/trainium_frontier.git",
        "purpose": template["purpose"],
        "remote": template["remote"],
        "runner_owns_manifest": False,
        "schema": "fresh-start-request-v2",
        "template_content": template,
        "template_path": REQUEST_REL,
        "template_sha256": template_sha,
        "verified_core_commit": template["verified_core_commit"],
    }


def standard_launcher_job_bytes(runtime: dict, concrete: dict) -> bytes:
    """Reproduce fresh_start_launch.py's deterministic remote json.dump bytes."""
    # The adapter writes the concrete request with recursively sorted keys.  The
    # standard launcher loads those bytes, appends request_file_sha256 locally,
    # then appends the remaining dynamic fields remotely.  Its json.dump call
    # preserves that insertion order and deliberately writes no final newline.
    ordered_concrete = json.loads(canonical_json(concrete))
    source_paths = list(dict.fromkeys((*LAUNCHER_SOURCE_PREFIX, *concrete["fingerprint_paths"])))
    if set(source_paths) != set(runtime["source_sha256"]):
        raise SystemExit("runtime source serialization paths mismatch")
    expected = dict(ordered_concrete)
    for key in RUNTIME_DYNAMIC_KEYS[:-1]:
        expected[key] = runtime[key]
    expected["source_sha256"] = {path: runtime["source_sha256"][path] for path in source_paths}
    return json.dumps(expected, indent=2).encode()


def validate_runtime_contract(
    runtime: dict, template: dict, template_sha: str, authorization_sha: str,
    verified_core: str, finalization_head: str, runtime_bytes: bytes | None = None,
) -> str:
    concrete = build_concrete_request(template, authorization_sha, template_sha, finalization_head)
    concrete_sha = hashlib.sha256(canonical_json(concrete)).hexdigest()
    if set(runtime) != set(concrete) | set(RUNTIME_DYNAMIC_KEYS):
        raise SystemExit("runtime job contract mismatch")
    for key, value in concrete.items():
        if runtime.get(key) != value:
            raise SystemExit(f"runtime job concrete request mismatch: {key}")
    if (
        runtime["request_file_sha256"] != concrete_sha
        or runtime["hostname"] != HOSTNAME
        or runtime["git_commit"] != finalization_head
        or runtime["source_dirty"] is not False
        or type(runtime["preflight_free_bytes"]) is not int
        or runtime["preflight_free_bytes"] < 8 * 1024**3
        or runtime["source_sha256"] != concrete["expected_sha256"]
        or type(runtime["started_at"]) is not str
        or not runtime["started_at"].endswith("+00:00")
    ):
        raise SystemExit("runtime job dynamic metadata mismatch")
    if runtime["verified_core_commit"] != verified_core:
        raise SystemExit("runtime verified core mismatch")
    if runtime_bytes is not None and runtime_bytes != standard_launcher_job_bytes(runtime, concrete):
        raise SystemExit("runtime job standard-launcher serialization mismatch")
    return concrete_sha


def validate_runtime_job_bytes(
    runtime_bytes: bytes, template: dict, template_sha: str, authorization_sha: str,
    verified_core: str, finalization_head: str,
) -> tuple[dict, str]:
    try:
        runtime = json.loads(runtime_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("runtime job JSON invalid") from error
    if not isinstance(runtime, dict):
        raise SystemExit("runtime job is not an object")
    concrete_sha = validate_runtime_contract(
        runtime, template, template_sha, authorization_sha, verified_core,
        finalization_head, runtime_bytes,
    )
    return runtime, concrete_sha


def _committed_bytes(head: str, relative: str) -> bytes:
    path = ROOT / relative
    actual = _nofollow_bytes(path)
    committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    if actual != committed:
        raise SystemExit(f"working path differs from finalization HEAD: {relative}")
    return actual


def validate_finalization(auth_path: Path, request_path: Path, runtime_path: Path) -> dict[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    parents = subprocess.check_output(["git", "rev-list", "--parents", "-n", "1", head], cwd=ROOT, text=True).split()
    if len(parents) != 2:
        raise SystemExit("finalization HEAD must have exactly one parent")
    verified_core = parents[1]
    changed = set(subprocess.check_output(["git", "diff", "--name-only", verified_core, head], cwd=ROOT, text=True).splitlines())
    if changed != {AUTH_REL, REQUEST_REL}:
        raise SystemExit("finalization child changed paths beyond authorization and request")
    if subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True).strip():
        raise SystemExit("tracked working tree is not clean")
    if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True).strip():
        raise SystemExit("index is not clean")
    if auth_path != ROOT / AUTH_REL or request_path != ROOT / REQUEST_REL or runtime_path != ROOT / RUNTIME_JOB_REL:
        raise SystemExit("runtime path contract mismatch")
    auth_bytes = _committed_bytes(head, AUTH_REL)
    request_bytes = _committed_bytes(head, REQUEST_REL)
    try:
        auth = json.loads(auth_bytes)
        request = json.loads(request_bytes)
    except json.JSONDecodeError as error:
        raise SystemExit("authorization or request JSON invalid") from error
    if auth_bytes != canonical_json(auth) or request_bytes != canonical_json(request):
        raise SystemExit("authorization or request is not canonical JSON")
    request_sha = validate_authorization_request(auth, request, verified_core)
    if hashlib.sha256(request_bytes).hexdigest() != request_sha:
        raise SystemExit("request bytes/hash mismatch")
    for relative, expected_sha in request["source_sha256"].items():
        core_blob = subprocess.check_output(["git", "show", f"{verified_core}:{relative}"], cwd=ROOT)
        head_blob = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        actual = _nofollow_bytes(ROOT / relative)
        if core_blob != head_blob or head_blob != actual or hashlib.sha256(actual).hexdigest() != expected_sha:
            raise SystemExit(f"core tree/source identity mismatch: {relative}")
    authorization_sha = hashlib.sha256(auth_bytes).hexdigest()
    runtime_bytes = _nofollow_bytes(runtime_path)
    _runtime, concrete_sha = validate_runtime_job_bytes(
        runtime_bytes, request, request_sha, authorization_sha, verified_core, head
    )
    return {
        "verified_core_commit": verified_core,
        "finalization_head": head,
        "authorization_sha256": authorization_sha,
        "template_sha256": request_sha,
        "concrete_request_sha256": concrete_sha,
        "runtime_job_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
        "launch_adapter_sha256": request["source_sha256"][ADAPTER_REL],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--runtime-job", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != HOSTNAME:
        raise SystemExit("wrong host")
    identity = validate_finalization(
        ROOT / args.authorization, ROOT / args.request, ROOT / args.runtime_job
    )
    validate_candidate()
    output, evidence = create_output()
    env = build_subprocess_environment(os.environ)
    package = ROOT / PACKAGE_REL
    rehearsal = ROOT / "research/scratch/submission_rehearsal.py"
    train_box = output / "train"
    eval_box = output / "eval"

    training_argv = [PYTHON, str(rehearsal), "--package", str(package), "--sandbox", str(train_box), "--mode", "train", "--fresh-caches"]
    run_process(training_argv, env, 2400, evidence, "training")
    checkpoint = train_box / "out/final.pt"
    checkpoint_sha, checkpoint_bytes = _nofollow_identity(checkpoint)
    train_rehearsal = _nofollow_json(train_box / "rehearsal.json")
    if (
        train_rehearsal.get("mode") != "train"
        or train_rehearsal.get("returncode") != 0
        or train_rehearsal.get("charged_through_process_exit_within_budget") is not True
        or train_rehearsal.get("checkpoint_sha256") != checkpoint_sha
        or train_rehearsal.get("checkpoint_bytes") != checkpoint_bytes
    ):
        raise SystemExit("training rehearsal receipt mismatch")
    durable_json(evidence, "training-rehearsal.json", train_rehearsal)
    if artifact_bytes(evidence, "training-rehearsal.json") != canonical_json(train_rehearsal):
        raise SystemExit("training rehearsal persistence mismatch")

    finite_source = """import json,sys,torch
value=torch.load(sys.argv[1],map_location='cpu',weights_only=True); stack=[value]; count=0
while stack:
 item=stack.pop()
 if isinstance(item,dict): stack.extend(item.values())
 elif isinstance(item,(list,tuple)): stack.extend(item)
 elif isinstance(item,torch.Tensor):
  count+=1
  if (item.is_floating_point() or item.is_complex()) and not torch.isfinite(item).all().item(): raise SystemExit('nonfinite')
print(json.dumps({'weights_only':True,'map_location':'cpu','finite':True,'tensor_count':count},sort_keys=True))
"""
    finite_output, _ = run_process([PYTHON, "-c", finite_source, str(checkpoint)], env, 600, evidence, "checkpoint-finite")
    finite_receipt = parse_finite_receipt(finite_output.decode("utf-8"))
    durable_json(evidence, "checkpoint-finite.json", {"schema": "c595-warmdown060-finite-v1", **finite_receipt})
    persisted_json(evidence, "checkpoint-finite.json", {"schema": "c595-warmdown060-finite-v1", **finite_receipt}, "c595-warmdown060-finite-v1")

    eval_argv = [
        PYTHON, str(rehearsal), "--package", str(package), "--sandbox", str(eval_box), "--mode", "eval",
        "--checkpoint", str(checkpoint), "--eval-tokens", "20971520", "--fresh-caches",
    ]
    eval_output, eval_process = run_process(eval_argv, env, 1500, evidence, "fresh-public20m")
    eval_text = eval_output.decode("utf-8")
    metrics = parse_metrics(eval_text)
    command_lines = [line for line in eval_text.splitlines() if line.startswith("REHEARSAL_COMMAND ")]
    if len(command_lines) != 1:
        raise SystemExit("fresh evaluation command receipt mismatch")
    prepare_argv = json.loads(command_lines[0].removeprefix("REHEARSAL_COMMAND "))
    expected_prepare_argv = [
        PYTHON, "prepare.py", "eval-public", "--train-py", "train.py", "--checkpoint", str(checkpoint),
        "--device", "neuron", "--batch-size", "1", "--seq-len", "1024", "--eval-tokens", "20971520",
        "--timeout-seconds", "1200",
    ]
    if prepare_argv != expected_prepare_argv:
        raise SystemExit("fresh evaluation argv mismatch")
    metrics_receipt = {
        "schema": "c595-warmdown060-metrics-v1", "label": LABEL, "wrapper_argv": eval_argv,
        "prepare_argv": prepare_argv, "metrics": metrics, "wall_seconds": eval_process["duration_seconds"],
    }
    durable_json(evidence, "fresh-public20m.metrics.json", metrics_receipt)
    persisted_json(evidence, "fresh-public20m.metrics.json", metrics_receipt, "c595-warmdown060-metrics-v1")
    eval_rehearsal = _nofollow_json(eval_box / "rehearsal.json")
    if eval_rehearsal.get("mode") != "eval" or eval_rehearsal.get("returncode") != 0:
        raise SystemExit("evaluation rehearsal receipt mismatch")
    durable_json(evidence, "evaluation-rehearsal.json", eval_rehearsal)
    if artifact_bytes(evidence, "evaluation-rehearsal.json") != canonical_json(eval_rehearsal):
        raise SystemExit("evaluation rehearsal persistence mismatch")

    after_sha, after_bytes = _nofollow_identity(checkpoint)
    if (after_sha, after_bytes) != (checkpoint_sha, checkpoint_bytes):
        raise SystemExit("checkpoint changed during finite scan or evaluation")
    validate_candidate()
    decision = decide_bpb(str(metrics["val_bpb"]))
    result = {
        "schema": "c595-warmdown060-result-v1", "label": LABEL, "host": HOST, "hostname": HOSTNAME,
        **identity,
        "source_sha256": SOURCE_SHA, "candidate_sha256": CANDIDATE_SHA, "prepare_sha256": PREPARE_SHA,
        "seed": 42, "max_train_seconds": 1800, "smoke": False,
        "single_causal_change": "WARMDOWN_RATIO 0.75 -> 0.60", "unchanged_weight_decay": 0.2,
        "environment_contract": {
            "LD_LIBRARY_PATH": PINNED_LD_LIBRARY_PATH,
            "LD_LIBRARY_PATH_sha256": PINNED_LD_LIBRARY_PATH_SHA256,
            "included_in_strict_child_environment": True,
        },
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": after_sha, "checkpoint_bytes": after_bytes,
        "checkpoint_preserved": True, "finite_scan": finite_receipt, "metrics": metrics, "decision_gate": decision,
        "checkpoint_prune_authorized": False, "cache_cleanup_authorized": False,
    }
    durable_json(evidence, "result.json", result)
    persisted_json(evidence, "result.json", result, "c595-warmdown060-result-v1")
    members = inspect_artifacts(evidence, BASE_ARTIFACTS)
    manifest = {"schema": "c595-warmdown060-artifact-manifest-v1", "label": LABEL, "members": members}
    durable_json(evidence, "artifact-manifest.json", manifest)
    persisted_json(evidence, "artifact-manifest.json", manifest, "c595-warmdown060-artifact-manifest-v1")
    manifest_sha = hashlib.sha256(artifact_bytes(evidence, "artifact-manifest.json")).hexdigest()
    revalidate_members(evidence, members)
    completion = {
        "schema": "c595-warmdown060-completion-v1", "label": LABEL, "host": HOST, "hostname": HOSTNAME,
        **identity,
        "candidate_sha256": CANDIDATE_SHA, "prepare_sha256": PREPARE_SHA,
        "environment_contract": {
            "LD_LIBRARY_PATH": PINNED_LD_LIBRARY_PATH,
            "LD_LIBRARY_PATH_sha256": PINNED_LD_LIBRARY_PATH_SHA256,
            "included_in_strict_child_environment": True,
        },
        "artifact_manifest_sha256": manifest_sha, "artifact_members": members,
        "checkpoint_path": str(checkpoint), "checkpoint_pre_eval_sha256": checkpoint_sha,
        "checkpoint_pre_eval_bytes": checkpoint_bytes, "checkpoint_post_eval_sha256": after_sha,
        "checkpoint_post_eval_bytes": after_bytes, "checkpoint_preserved": True,
        "ready_to_retrieve": True, "checkpoint_prune_executed": False, "cache_cleanup_executed": False,
    }
    durable_json(evidence, "ready-to-retrieve.json", completion)
    persisted_json(evidence, "ready-to-retrieve.json", completion, "c595-warmdown060-completion-v1")
    completion_sha = hashlib.sha256(artifact_bytes(evidence, "ready-to-retrieve.json")).hexdigest()
    final = inspect_artifacts(evidence, BASE_ARTIFACTS | {"artifact-manifest.json", "ready-to-retrieve.json"})
    if final["artifact-manifest.json"]["sha256"] != manifest_sha:
        raise SystemExit("manifest changed before completion")
    revalidate_members(evidence, members)
    if final["ready-to-retrieve.json"]["sha256"] != completion_sha:
        raise SystemExit("completion changed before final report")
    print(json.dumps({
        "result": result, "artifact_manifest_sha256": manifest_sha,
        "completion_receipt_sha256": completion_sha, "ready_to_retrieve": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
