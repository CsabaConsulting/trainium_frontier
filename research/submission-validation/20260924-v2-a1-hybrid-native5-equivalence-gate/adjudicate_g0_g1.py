#!/usr/bin/env python3
"""Fail-closed G0/G1-r3-only evidence adjudication.

This is gate infrastructure, not execution authority.  It authenticates the immutable G0
archive, an externally hash-bound G1 evidence manifest, and exact detailed G1 reports.  It
normalizes only ``arm``, ``detailed`` and ``label`` in the header and otherwise requires exact
G0/G1 shared-record equality.  Publication delegates to the independently reviewed r4
comparator's sealed writer; no second result writer exists here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import types
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
COMPARATOR_PATH = HERE / "compare_gate_reports.py"
COMPARATOR_SHA256 = "4b199343a1cecfc6be4b86242da421127c39e8dcf9f67b378ef56f75927fdd32"
MANIFEST_SCHEMA = "a1-native-five-g0-g1-r3-evidence-manifest-v1"
RESULT_SCHEMA = "a1-native-five-g0-g1-r3-adjudication-v1"
G0_LABEL = "fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1"
G1_LABEL = "fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3"
G0_BASE = (
    HERE / "operator-evidence" / G0_LABEL / "runtime" / "research" / "run-logs" / G0_LABEL
)
G0_REPORT_DIR = G0_BASE / "native-gate"
G1_ARCHIVE = HERE / "operator-evidence" / G1_LABEL / "runtime"
G1_REPORT_DIR = G1_ARCHIVE / "research" / "run-logs" / G1_LABEL / "native-gate"
G1_CHECKPOINT = G1_ARCHIVE / "out" / G1_LABEL / "final.pt"
G0_PASS_RECEIPT = HERE / "g0-comparator-r4-independent-verification.json"
G0_PASS_RECEIPT_SHA256 = "c4da83d6fe19fddab548ef49cb1a2d2eb69146e37bc36f452bcf37b446fc8c68"
G0_REPLAY_SHA256 = "7198d39b9c21f1eb7e47f84e52398ca85d239ad564a36e6d524674d83355ab4a"
G0_CHECKPOINT_SHA256 = "2d0891c569e6046bd4a3635c62c1ebdb4c6ea6ed0d9788212910a26e61ad3b13"
G0_CHECKPOINT_BYTES = 739_322_371
EXPECTED_GATHER_INDEXES = [68, 138, 208, 278, 348]
EXPECTED_FIXED_SOURCES = {
    "prepare.py": "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb",
    "submissions/dense6x1024_public0992851/train.py": "9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae",
    "research/a1_native_gate_observer.py": "25bc72a2c4a70948915e0449d6dbc6fbcfa35e519b06a82137b55d755c88e225",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_hybrid.py": "908431d7e1da8a9110e6e6005de5ecbe3dba13d798bf2f37f2443404dd8d7d94",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_ordinary.py": "6ed69d2356c28e0dd8f88f494d988c0b87f7f86426a001945bfd958ed6f8f6d0",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/a1_ownership_manifest.json": "5fd3b2f075191210b7e373c8321c68b2d8f880db8cc7a788d60a90db4a382007",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/compare_gate_reports.py": COMPARATOR_SHA256,
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/verify_model_checkpoints.py": "e1ead481f9786ec02a3c60d88e72820f084274200bccd1f7eeb16da108cb6e2e",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g0-comparator-r4-independent-verification.json": G0_PASS_RECEIPT_SHA256,
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/G1_DRAFTING_CHECKLIST_ANALYSIS.md": "a530256c81a51ce5635e6c240eddffd03160955f02a57d5c88c0596b54a3be17",
    "research/fresh_start_launch.py": "7620fea10bb51c54481b8347ec9e97ebd40ce8f5ee206dc3f203c9483c35ee2f",
    "research/fresh_start_guard.py": "ac6f7d23dbe998b1d9c8ae14e209c6ffc21fac5d927895b1b92b15dc493f6de6",
    "research/fresh_start_host.py": "f5bc5034b4ac77e2af3234a9630ab8e4d0e3f1b24f7404ac22778d30016e6d16",
    "research/fresh_start_backup.py": "c978fd20b6620d0d80c25f8982974e8c4b2ad009edd9da376d784f5cecc3ead3",
    "research/fresh_start_sync.py": "ea428d33f4de18376950abbd81cfda33d3a969f75b60e3648ff552bea04fc60c",
    "research/run_experiment.py": "494e283f2464f1d08059eb69423165cbd40ff3fbf9ba9c89f746e0ce45096705",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/G1_R1_TRANSPORT_FAILURE_ANALYSIS.md": "c457bb70c161d1db5f048581cbfbe321acc260bd249e131eac847ae8be7b676b",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-draft-independent-verification.json": "15fb8d9396693ba70ee1c181bc1060196a616987c4b1908dc4a5a58e0b6c07cd",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-authorization.json": "f3cee7fd7b5f3a10eebaf6b457b5f58b13ea5fe9e4a8a9b6f912285d8a67f41e",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1-pre-dispatch-failure/OPERATOR_FAILURE_RECEIPT.json": "e67e153f48f23b684fc3d041cecafbce82125767fd5e7bd25c28e82768907943",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1-pre-dispatch-failure/concrete-request.json": "b1d7d037c34c9ef21b9ad093633e2d710b130c366bc214a912bfb2592a3318f0",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1-pre-dispatch-failure/dispatch.json": "6bc760190b86ab95f2c6844794063d34d682e2b609679360071d63b98c69fa44",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/G1_R2_CORRECTNESS_FAILURE_ANALYSIS.md": "014288adbbcfaa516cbdcf141e5f70a7dc7418fa63191f6797014ee0b5b20c3a",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-r2-transport-independent-verification.json": "987d103735959d11759c05180369ac87061abd718af0f821bcb81cb7906d2be0",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-r2-authorization.json": "3d98b81c75c020418c2b61349b8f1b42ff937416b8b605c27542eb097f877771",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2/OPERATOR_RECEIPT.json": "c6db31c323c545c851c63fb1c8313b46a0b2d0aeafc1e2a1ea8045e43acd8fb5",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2/runtime/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2.child.log": "87dedc4345bb1605e115acf995ceb5dedb56c20bfddb423343405a85632e9f4a",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/operator-evidence/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2/runtime/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2.runtime-result.json": "705547ff56258ac69232b0dd515d07a52056445c32ce5e513ac15613ffba8596",
}
REQUIRED_RUNTIME_SOURCE_PATHS = {
    "prepare.py", "submissions/dense6x1024_public0992851/train.py",
    "research/a1_native_gate_observer.py", "research/fresh_start_launch.py",
    "research/fresh_start_guard.py", "research/fresh_start_host.py",
    "research/fresh_start_backup.py", "research/fresh_start_sync.py",
    "research/run_experiment.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_hybrid.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_ordinary.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/a1_ownership_manifest.json",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/compare_gate_reports.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/verify_model_checkpoints.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1_finalization.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-request.template.json",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-r3-authorization.json",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-r3-authorization.pending.json",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/adjudicate_g0_g1.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/test_g1_package.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/test_g1_r3_package.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/test_gate_package.py",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/EXPERIMENT_PROPOSAL.json",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/G1_R3_CORRECTNESS_RETRY_AMENDMENT.json",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/RUNBOOK.md",
    "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/requests.pending.json",
}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


class InvalidEvidence(ValueError):
    pass


def _safe_module(path: Path, expected_sha256: str) -> types.ModuleType:
    try:
        parts = path.relative_to(ROOT).parts
    except ValueError as exc:
        raise InvalidEvidence("r4 comparator path escapes repository root") from exc
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise InvalidEvidence("r4 comparator path is not lexical and rooted")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fds: list[int] = []
    directory_identities: list[tuple[int, int, int]] = []
    fd = -1
    try:
        parent_fd = os.open(ROOT, directory_flags)
        directory_fds.append(parent_fd)
        root_info = os.fstat(parent_fd)
        directory_identities.append((root_info.st_dev, root_info.st_ino, root_info.st_mode))
        for part in parts[:-1]:
            child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            named = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            child_info = os.fstat(child_fd)
            if ((named.st_dev, named.st_ino, named.st_mode)
                    != (child_info.st_dev, child_info.st_ino, child_info.st_mode)):
                os.close(child_fd)
                raise InvalidEvidence("r4 comparator ancestor identity changed")
            directory_fds.append(child_fd)
            directory_identities.append((child_info.st_dev, child_info.st_ino,
                                         child_info.st_mode))
            parent_fd = child_fd
        fd = os.open(parts[-1], file_flags, dir_fd=parent_fd)
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.getuid() or before.st_gid != os.getgid()
                or before.st_size < 1 or before.st_size > 4 * 1024 * 1024):
            raise InvalidEvidence("unsafe r4 comparator source identity")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
        named = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                                  value.st_uid, value.st_gid, value.st_size,
                                  value.st_mtime_ns, value.st_ctime_ns)
        if identity(before) != identity(after) or identity(before) != identity(named):
            raise InvalidEvidence("r4 comparator source changed during authentication")
        root_named = os.lstat(ROOT)
        if (root_named.st_dev, root_named.st_ino, root_named.st_mode) != directory_identities[0]:
            raise InvalidEvidence("repository root changed during comparator authentication")
        for index, part in enumerate(parts[:-1]):
            observed = os.stat(part, dir_fd=directory_fds[index], follow_symlinks=False)
            if (observed.st_dev, observed.st_ino, observed.st_mode) != directory_identities[index + 1]:
                raise InvalidEvidence("r4 comparator ancestor changed during authentication")
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise InvalidEvidence("r4 comparator SHA256 mismatch")
    finally:
        if fd >= 0:
            os.close(fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
    module = types.ModuleType("a1_g0_g1_pinned_r4_comparator")
    module.__file__ = os.fspath(path)
    exec(compile(data, os.fspath(path), "exec"), module.__dict__)
    return module


CMP = _safe_module(COMPARATOR_PATH, COMPARATOR_SHA256)


def _binding(value: Any, label: str) -> dict[str, Any]:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}
            or type(value["path"]) is not str or Path(value["path"]).is_absolute()
            or ".." in Path(value["path"]).parts
            or HEX64.fullmatch(value["sha256"]) is None
            or type(value["bytes"]) is not int or value["bytes"] < 1):
        raise InvalidEvidence(f"malformed {label} binding")
    return value


def _capture_binding(binding: dict[str, Any], *, parser=None, max_bytes=64 * 1024 * 1024):
    binding = _binding(binding, "file")
    captured = CMP._capture_file(
        ROOT, ROOT / binding["path"], max_bytes=max_bytes,
        expected_sha256=binding["sha256"], parser=parser,
    )
    if captured["identity"][6] != binding["bytes"]:
        raise InvalidEvidence(f"authenticated size mismatch: {binding['path']}")
    return captured


def _checkpoint(path: Path, expected_sha256: str, expected_bytes: int) -> dict[str, Any]:
    if not path.is_absolute():
        path = ROOT / path
    _, parts = CMP._path_parts(ROOT, path)
    directory_fds, name, opened_root = CMP._open_directory_chain(ROOT, path)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = -1
    try:
        fd = os.open(name, flags, dir_fd=directory_fds[-1])
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.getuid() or before.st_gid != os.getgid()
                or before.st_size != expected_bytes):
            raise InvalidEvidence("unsafe G1 checkpoint identity/size")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise InvalidEvidence("G1 checkpoint digest mismatch")
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb", closefd=True) as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
        after = os.fstat(fd)
        named = os.stat(name, dir_fd=directory_fds[-1], follow_symlinks=False)
        ident = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                               value.st_uid, value.st_gid, value.st_size,
                               value.st_mtime_ns, value.st_ctime_ns)
        if ident(before) != ident(after) or ident(before) != ident(named):
            raise InvalidEvidence("G1 checkpoint changed during same-fd validation")
        CMP._revalidate_directory_chain(directory_fds, opened_root, parts[:-1])
    finally:
        if fd >= 0:
            os.close(fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
    if not isinstance(payload, dict) or set(payload) != {
            "model_state", "model_config", "toggles", "meta"}:
        raise InvalidEvidence("G1 checkpoint is not exact model-only schema")
    state = payload["model_state"]
    if not isinstance(state, dict) or not state:
        raise InvalidEvidence("G1 checkpoint model state is empty")
    state_digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise InvalidEvidence("G1 checkpoint model state entry malformed")
        cpu = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(cpu).all().item()):
            raise InvalidEvidence("G1 checkpoint contains nonfinite tensor")
        state_digest.update(name.encode() + b"\0" + str(cpu.dtype).encode() + b"\0")
        state_digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode() + b"\0")
        state_digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return {"sha256": expected_sha256, "bytes": expected_bytes,
            "model_state_sha256": state_digest.hexdigest(), "tensor_count": len(state)}


def _expected_command(source: str, output: str, report: str) -> list[str]:
    return [
        "/home/ubuntu/workspace/native_venv/bin/torchrun", "--standalone", "--nproc_per_node=4",
        source, "--num-steps", "5", "--max-train-seconds", "1800", "--n-embd", "1024",
        "--ngram-ve", "--ngram-ve-layers=-1", "--ngram-ve-table-mult", "64",
        "--ngram-ve-dim", "128", "--ngram-ve-trigram", "--ngram-ve-opt", "rmsprop",
        "--ngram-ve-flat-lr", "--ngram-ve-bf16", "--ngram-ve-clip-exclude",
        "--no-qk-shift", "--compile-sdpa-direct", "--nki-local-conv",
        "--bf16-norm-output", "--pack-factor", "32", "--clip-after-reduce",
        "--seq-len", "1024", "--no-eval-public", "--matrix-lr", "0.015",
        "--out-dir", output, "--native-gate-arm", "G1", "--native-gate-observation",
        "--native-gate-label", G1_LABEL, "--native-gate-host", "trn21",
        "--native-gate-report-dir", report,
    ]


def _expected_environment(cache: str) -> dict[str, str]:
    return {
        "HOME": "/home/ubuntu", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "LOGNAME": "ubuntu",
        "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
        "NEURON_COMPETITION_R1_SEED": "42", "NEURON_LOGICAL_NC_CONFIG": "2",
        "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv",
        "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
        "PYTHONUNBUFFERED": "1", "TMPDIR": "/tmp", "USER": "ubuntu",
        "NEURON_COMPILE_CACHE_DIR": cache, "TRAINIUM_FRESH_START_RUN_ID": G1_LABEL,
    }


def _parse_g1_r3_rank_bytes(data: bytes, rank: int) -> list[dict[str, Any]]:
    """Adapt only the fresh retry label to the immutable r4 G1 science parser."""
    lines = data.splitlines(keepends=True)
    if not lines or not lines[0].endswith(b"\n"):
        raise InvalidEvidence(f"G1 r3 rank {rank}: missing terminated header")
    try:
        header = json.loads(lines[0])
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InvalidEvidence(f"G1 r3 rank {rank}: malformed header") from error
    if not isinstance(header, dict) or header.get("label") != G1_LABEL:
        raise InvalidEvidence(f"G1 r3 rank {rank}: retry label mismatch")
    header["label"] = CMP.LABELS["G1"]
    adapted = json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    adapted += b"".join(lines[1:])
    return CMP._parse_rank_bytes(adapted, "G1", rank)


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    required = {"schema", "state", "evidence_head", "host", "hostname", "label",
                "source", "paths", "source_sha256", "runtime_source_sha256",
                "request_sha256", "authorization_sha256",
                "runtime_job", "runtime_result", "postflight", "process", "checkpoint",
                "rank_reports", "g0_dependency", "cleanup", "checkpoint_pruned",
                "cache_cleaned", "g2_authorized", "full_a1_authorized", "submission_authorized"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise InvalidEvidence("G1 evidence manifest schema mismatch")
    if (manifest["schema"] != MANIFEST_SCHEMA or manifest["state"] != "independent-evidence-finalized"
            or HEX40.fullmatch(manifest["evidence_head"]) is None
            or (manifest["host"], manifest["hostname"], manifest["label"])
            != ("trn21", "ip-172-31-34-85", G1_LABEL)
            or manifest["source"] !=
            "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_hybrid.py"
            or HEX64.fullmatch(manifest["request_sha256"]) is None
            or HEX64.fullmatch(manifest["authorization_sha256"]) is None):
        raise InvalidEvidence("G1 evidence identity mismatch")
    paths = manifest["paths"]
    expected_paths = {
        "remote": "/home/ubuntu/trainium_freshstart_20260925_v2_a1_native5_g1_hybrid_observation_on_trn21_r3",
        "compile_cache": "/home/ubuntu/neuron_cache_20260925_v2_a1_native5_g1_trn21_r3",
        "output": f"out/{G1_LABEL}",
        "report": f"research/run-logs/{G1_LABEL}/native-gate",
    }
    if paths != expected_paths:
        raise InvalidEvidence("G1 evidence path contract mismatch")
    if manifest["source_sha256"] != EXPECTED_FIXED_SOURCES:
        raise InvalidEvidence("G1 scientific/source pin map mismatch")
    runtime_sources = manifest["runtime_source_sha256"]
    if (not isinstance(runtime_sources, dict) or not runtime_sources
            or any(type(path) is not str or HEX64.fullmatch(digest) is None
                   for path, digest in runtime_sources.items())
            or any(runtime_sources.get(path) != digest
                   for path, digest in manifest["source_sha256"].items())):
        raise InvalidEvidence("G1 launcher-observed source map is incomplete or malformed")
    if not REQUIRED_RUNTIME_SOURCE_PATHS.issubset(runtime_sources):
        raise InvalidEvidence("G1 launcher-observed protected source set is incomplete")
    expected_g0 = {
        "evidence_commit": "37aa3644e63e20a1427cfa69b44fd6de7e25cd81",
        "pass_receipt_commit": "8add2585d30b11ac2c50fb2701957657a1a6f30d",
        "pass_receipt_sha256": G0_PASS_RECEIPT_SHA256,
        "corrected_replay_sha256": G0_REPLAY_SHA256,
        "corrected_replay_bytes": 326,
        "checkpoint_sha256": G0_CHECKPOINT_SHA256,
        "checkpoint_bytes": G0_CHECKPOINT_BYTES,
    }
    if manifest["g0_dependency"] != expected_g0:
        raise InvalidEvidence("G0 prerequisite identity drift")
    for key in ("runtime_job", "runtime_result", "postflight", "process", "checkpoint"):
        _binding(manifest[key], key)
    archived_logs = G1_ARCHIVE / "research" / "run-logs"
    expected_bound_paths = {
        "runtime_job": archived_logs / f"{G1_LABEL}.job.json",
        "runtime_result": archived_logs / f"{G1_LABEL}.runtime-result.json",
        "postflight": archived_logs / f"{G1_LABEL}.postflight.json",
        "process": archived_logs / f"{G1_LABEL}.process.json",
        "checkpoint": G1_ARCHIVE / "out" / G1_LABEL / "final.pt",
    }
    for key, expected_path in expected_bound_paths.items():
        if Path(manifest[key]["path"]) != expected_path.relative_to(ROOT):
            raise InvalidEvidence(f"G1 {key} archive path binding mismatch")
    reports = manifest["rank_reports"]
    if not isinstance(reports, dict) or set(reports) != {f"rank-{rank}.jsonl" for rank in range(4)}:
        raise InvalidEvidence("G1 rank report binding set mismatch")
    for name, binding in reports.items():
        _binding(binding, name)
        if Path(binding["path"]) != (
                G1_ARCHIVE / "research" / "run-logs" / G1_LABEL / "native-gate" / name
        ).relative_to(ROOT):
            raise InvalidEvidence(f"G1 {name} archive path binding mismatch")
    if any(manifest[key] is not False for key in (
            "cleanup", "checkpoint_pruned", "cache_cleaned", "g2_authorized",
            "full_a1_authorized", "submission_authorized")):
        raise InvalidEvidence("G1 evidence claims forbidden downstream authority/action")
    return manifest


def _validate_lifecycle(manifest: dict[str, Any]) -> dict[str, Any]:
    runtime = _capture_binding(manifest["runtime_result"], parser=json.loads)["parsed"]
    postflight = _capture_binding(manifest["postflight"], parser=json.loads)["parsed"]
    process = _capture_binding(manifest["process"], parser=json.loads)["parsed"]
    runtime_job = _capture_binding(manifest["runtime_job"], parser=json.loads)["parsed"]
    paths = manifest["paths"]
    command = _expected_command(manifest["source"], paths["output"], paths["report"])
    environment = _expected_environment(paths["compile_cache"])
    outer_environment = {key: value for key, value in environment.items()
                         if key not in {"NEURON_COMPILE_CACHE_DIR", "TRAINIUM_FRESH_START_RUN_ID"}}
    request_template = (
        "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/"
        "g1-request.template.json"
    )
    authorization = (
        "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/"
        "g1-r3-authorization.json"
    )
    driver = (
        "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/"
        "g1_finalization.py"
    )
    runtime_job_relative = f"research/run-logs/{G1_LABEL}.job.json"
    outer_command = [
        "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "4200s",
        "/home/ubuntu/workspace/native_venv/bin/python", driver, "run",
        "--authorized-head", manifest["evidence_head"],
        "--request-sha256", manifest["request_sha256"],
        "--authorization-sha256", manifest["authorization_sha256"],
        "--request-template", request_template,
        "--authorization", authorization,
        "--runtime-job", runtime_job_relative,
    ]
    expected_job = {
        "schema": "a1-native-five-g1-r3-concrete-request-v1", "arm": "G1",
        "host": "trn21", "expected_hostname": "ip-172-31-34-85", "label": G1_LABEL,
        "remote": paths["remote"], "expected_commit": manifest["evidence_head"],
        "origin": "https://github.com/CsabaConsulting/trainium_frontier.git",
        "purpose": "Human-approved G1-only A1 native five-update hybrid observation-on correctness retry r3",
        "command": outer_command, "environment": outer_environment,
        "min_free_bytes": 6_979_321_856, "runner_owns_manifest": False,
        "non_overwrite": True, "verified_core_commit":
        "994d302b0036c56aaab63797f0561797bc4cf431",
        "verifier_receipt_commit": "8add2585d30b11ac2c50fb2701957657a1a6f30d",
        "request_template_path": request_template,
        "request_sha256": manifest["request_sha256"],
        "authorization_path": authorization,
        "authorization_sha256": manifest["authorization_sha256"],
        "hostname": "ip-172-31-34-85", "git_commit": manifest["evidence_head"],
        "source_dirty": False, "source_sha256": manifest["runtime_source_sha256"],
    }
    if not isinstance(runtime_job, dict) or any(runtime_job.get(key) != value
                                                 for key, value in expected_job.items()):
        raise InvalidEvidence("G1 launcher runtime-job contract mismatch")
    expected_fingerprints = sorted(set(manifest["runtime_source_sha256"]) - {"train.py"})
    if runtime_job.get("fingerprint_paths") != expected_fingerprints:
        raise InvalidEvidence("G1 launcher fingerprint path set mismatch")
    if (type(runtime_job.get("preflight_free_bytes")) is not int
            or runtime_job["preflight_free_bytes"] < 6_979_321_856
            or type(runtime_job.get("started_at")) is not str
            or not runtime_job["started_at"]):
        raise InvalidEvidence("G1 launcher preflight identity/budget mismatch")
    required_runtime = {
        "schema": "a1-native-five-g1-r3-runtime-result-v1", "arm": "G1", "status": "PASS",
        "complete": True, "label": G1_LABEL, "host": "trn21",
        "authorized_head": manifest["evidence_head"],
        "request_sha256": manifest["request_sha256"],
        "authorization_sha256": manifest["authorization_sha256"],
        "command": command, "environment": environment, "returncode": 0,
        "timed_out": False, "interrupted_signals": [], "process_group_quiescent": True,
        "checkpoint_pruned": False, "cache_cleaned": False,
    }
    if not isinstance(runtime, dict) or any(runtime.get(key) != value
                                            for key, value in required_runtime.items()):
        raise InvalidEvidence("G1 runtime result contract mismatch")
    if (type(runtime.get("duration_seconds")) is not float
            or not math.isfinite(runtime["duration_seconds"])
            or not 0 < runtime["duration_seconds"] <= 3600
            or runtime.get("compile_cache") != {"path": paths["compile_cache"], "preserved": True}
            or type(runtime.get("runtime_job_sha256")) is not str
            or runtime["runtime_job_sha256"] != manifest["runtime_job"]["sha256"]):
        raise InvalidEvidence("G1 runtime budget/cache/job identity mismatch")
    expected_artifacts = {
        f"{paths['output']}/final.pt": manifest["checkpoint"],
        f"research/run-logs/{G1_LABEL}.child.log": None,
        **{f"{paths['report']}/{name}": binding for name, binding in manifest["rank_reports"].items()},
    }
    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
        raise InvalidEvidence("G1 sealed artifact membership mismatch")
    for relative, binding in expected_artifacts.items():
        if binding is not None and (artifacts[relative].get("sha256"), artifacts[relative].get("bytes")) != (
                binding["sha256"], binding["bytes"]):
            raise InvalidEvidence(f"G1 sealed artifact identity mismatch: {relative}")
    if (not isinstance(postflight, dict) or postflight.get("label") != G1_LABEL
            or postflight.get("remote") != paths["remote"] or postflight.get("stage") != "postflight"
            or postflight.get("source_commit") != manifest["evidence_head"]
            or postflight.get("command_returncode") != 0 or postflight.get("guard_error") is not None
            or postflight.get("interrupted_signals") != [] or postflight.get("failures") != []
            or postflight.get("host_clean") is not True
            or postflight.get("cleanup", {}).get("remaining") != []
            or postflight.get("cleanup", {}).get("artifacts_deleted") != []
            or postflight.get("cleanup", {}).get("caches_deleted") != []
            or postflight.get("health", {}).get("hostname") != "ip-172-31-34-85"
            or postflight.get("health", {}).get("neuron_process_count") != 0
            or postflight.get("health", {}).get("device_holders", {}).get("stdout") != ""
            or postflight.get("health", {}).get("relevant_processes") != []):
        raise InvalidEvidence("G1 clean postflight contract mismatch")
    if (not isinstance(process, dict) or process.get("label") != G1_LABEL
            or type(process.get("guard_pid")) is not int or process["guard_pid"] <= 0
            or type(process.get("child_pid")) is not int or process["child_pid"] <= 0
            or process.get("child_session") != process["child_pid"]
            or process.get("command") != outer_command):
        raise InvalidEvidence("G1 process identity contract mismatch")
    return runtime


def _validate_g1_records(g0_dir: Path, g1_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    g0_summary = CMP.validate_immutable_g0(g0_dir)
    ownership = CMP._load_ownership_manifest()
    report_pins = ownership["source_pins"]["immutable_g0_report_sha256"]
    g0 = [CMP._load_rank(g0_dir, "G0", rank,
                         expected_sha256=report_pins[f"rank-{rank}.jsonl"])
          for rank in range(4)]
    g1 = []
    for rank in range(4):
        binding = manifest["rank_reports"][f"rank-{rank}.jsonl"]
        expected_path = g1_dir / f"rank-{rank}.jsonl"
        if Path(binding["path"]) != expected_path.relative_to(ROOT):
            raise InvalidEvidence(f"G1 rank {rank} path binding mismatch")
        captured = CMP._capture_file(
            g1_dir, expected_path, expected_sha256=binding["sha256"],
            parser=lambda data, rank=rank: _parse_g1_r3_rank_bytes(data, rank),
        )
        if captured["identity"][6] != binding["bytes"]:
            raise InvalidEvidence(f"G1 rank {rank} byte-size binding mismatch")
        records = captured["parsed"]
        CMP._validate_exact_groups(records, ownership, "G1", rank)
        CMP._ownership_shard_elements(records, "G1", rank, ownership)
        CMP._validate_owner_exchange_records(records, "G1", rank, ownership)
        g1.append(records)
    failures = []
    for rank in range(4):
        CMP._exact(CMP._strip(CMP._by(g0[rank], "header")[0], "arm", "detailed", "label"),
                   CMP._strip(CMP._by(g1[rank], "header")[0], "arm", "detailed", "label"),
                   f"rank {rank}: observation changed header contract", failures)
        for kind in ("inventory", "loss", "update", "collectives", "complete"):
            CMP._exact(CMP._by(g0[rank], kind), CMP._by(g1[rank], kind),
                       f"rank {rank}: observation changed {kind}", failures)
        inventory = CMP._by(g1[rank], "inventory")[0]
        entries = inventory["ownership"]["entries"]
        if any(item.get("kind") == "rmsprop" for item in entries):
            failures.append(f"rank {rank}: RMS table entered ownership")
        calls = [item["rms_call_count"] for item in CMP._by(g1[rank], "update")]
        if calls != [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]:
            failures.append(f"rank {rank}: exact RMS call sequence mismatch")
        gathers = [item for item in CMP._by(g1[rank], "collectives")[0]["sequence"]
                   if item.get("op") == "all_gather_into_tensor"]
        if [item["index"] for item in gathers] != EXPECTED_GATHER_INDEXES:
            failures.append(f"rank {rank}: owner exchange position mismatch")
        for gradient in CMP._by(g1[rank], "gradients"):
            if set(gradient) != {"kind", "step", "stage", "tables"}:
                failures.append(f"rank {rank}: gradient record schema is not exact")
            for name, value in gradient["tables"].items():
                if (value.get("shape") != [524_288, 128]
                        or value.get("numel") != 67_108_864
                        or value.get("dtype") != "torch.bfloat16"):
                    failures.append(f"rank {rank}: {name} gradient identity mismatch")
        for clip in CMP._by(g1[rank], "clip"):
            if (set(clip) != {"kind", "step", "main_norm", "table_norm", "max_norm",
                             "scalar_bytes_sha256"}
                    or any(type(clip.get(key)) is not float or not math.isfinite(clip[key])
                           or clip[key] < 0 for key in ("main_norm", "table_norm", "max_norm"))):
                failures.append(f"rank {rank}: clip record schema/value is not exact")
        updates = CMP._by(g1[rank], "update")
        for index, update in enumerate(updates):
            completed = (index + 1) // 2
            state = CMP._table_state(update)
            if set(state) != CMP.TABLES:
                failures.append(f"rank {rank} update {index}: table state missing")
                continue
            expected_fields = set() if completed == 0 else {"step", "exp_avg_sq", "bias2"}
            for name, row in state.items():
                fields = row.get("fields")
                if (row.get("kind") != "rmsprop" or not isinstance(fields, dict)
                        or set(fields) != expected_fields
                        or (completed and fields.get("step") != completed)
                        or (completed and (fields["exp_avg_sq"].get("shape") != [524_288, 128]
                                           or fields["exp_avg_sq"].get("numel") != 67_108_864
                                           or fields["exp_avg_sq"].get("dtype") != "torch.bfloat16"))):
                    failures.append(f"rank {rank} update {index}: malformed {name} RMS state")
    reference = g1[0]
    reference_post = [row for row in CMP._by(reference, "gradients")
                      if row["stage"] != "local_pre_avg"]
    for rank in range(1, 4):
        CMP._exact(reference_post, [row for row in CMP._by(g1[rank], "gradients")
                                    if row["stage"] != "local_pre_avg"],
                   f"rank {rank}: post-AVG/clip gradients disagree", failures)
        CMP._exact(CMP._by(reference, "clip"), CMP._by(g1[rank], "clip"),
                   f"rank {rank}: clip scalar bytes disagree", failures)
        for index, (left, right) in enumerate(zip(CMP._by(reference, "update"),
                                                  CMP._by(g1[rank], "update"))):
            CMP._exact(left["tables"], right["tables"],
                       f"rank {rank} update {index}: table params disagree", failures)
            CMP._exact(CMP._table_state(left), CMP._table_state(right),
                       f"rank {rank} update {index}: table state disagrees", failures)
            CMP._exact(left["model_state"], right["model_state"],
                       f"rank {rank} update {index}: persistent model disagrees", failures)
        CMP._exact(CMP._by(reference, "complete")[0]["final_model_state"],
                   CMP._by(g1[rank], "complete")[0]["final_model_state"],
                   f"rank {rank}: final model disagrees", failures)
    if failures:
        raise InvalidEvidence("; ".join(failures))
    return {
        "ownership": g0_summary,
        "final_model_state": CMP._by(reference, "complete")[0]["final_model_state"],
    }


def adjudicate(g0_dir: Path, g1_dir: Path, manifest_path: Path,
               manifest_sha256: str, checkpoint_path: Path) -> dict[str, Any]:
    if HEX64.fullmatch(manifest_sha256) is None:
        raise InvalidEvidence("evidence manifest SHA256 is not exact")
    captured = CMP._capture_file(ROOT, manifest_path, max_bytes=1024 * 1024,
                                 expected_sha256=manifest_sha256, parser=json.loads)
    manifest = _validate_manifest(captured["parsed"])
    for relative, expected in EXPECTED_FIXED_SOURCES.items():
        CMP._capture_file(ROOT, ROOT / relative, expected_sha256=expected)
    if g0_dir.resolve() != G0_REPORT_DIR.resolve():
        raise InvalidEvidence("G0 archive path is not the immutable package archive")
    if g1_dir.resolve() != G1_REPORT_DIR.resolve():
        raise InvalidEvidence("G1 report path is not the exact immutable archive path")
    receipt = CMP._capture_file(ROOT, G0_PASS_RECEIPT,
                                expected_sha256=G0_PASS_RECEIPT_SHA256,
                                parser=json.loads)["parsed"]
    if (receipt.get("verdict") != "PASS" or receipt.get("reviewed_commit") !=
            "994d302b0036c56aaab63797f0561797bc4cf431"
            or receipt.get("archived_g0_replay", {}).get("result_sha256") != G0_REPLAY_SHA256
            or receipt.get("archived_g0_replay", {}).get("result_bytes") != 326):
        raise InvalidEvidence("immutable G0 PASS receipt contract mismatch")
    runtime = _validate_lifecycle(manifest)
    record_summary = _validate_g1_records(g0_dir, g1_dir, manifest)
    checkpoint_binding = manifest["checkpoint"]
    if checkpoint_path.resolve() != G1_CHECKPOINT.resolve():
        raise InvalidEvidence("G1 checkpoint path is not the exact immutable archive path")
    if Path(checkpoint_binding["path"]) != checkpoint_path.relative_to(ROOT):
        raise InvalidEvidence("G1 checkpoint path binding mismatch")
    checkpoint = _checkpoint(checkpoint_path, checkpoint_binding["sha256"],
                             checkpoint_binding["bytes"])
    if (checkpoint["model_state_sha256"] != record_summary["final_model_state"]["sha256"]
            or checkpoint["tensor_count"] != record_summary["final_model_state"]["tensor_count"]):
        raise InvalidEvidence("G1 checkpoint state differs from sealed final model record")
    return {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        "verdict": "G1_OBSERVATION_NONPERTURBING",
        "g0_replay_sha256": G0_REPLAY_SHA256,
        "g1_evidence_manifest_sha256": manifest_sha256,
        "g1_runtime_result_sha256": manifest["runtime_result"]["sha256"],
        "g1_checkpoint": checkpoint,
        "g0_ownership": record_summary["ownership"],
        "shared_records_exact": True,
        "g1_detailed_records_complete": True,
        "process_group_quiescent": runtime["process_group_quiescent"],
        "g2_authorized": False,
        "full_a1_authorized": False,
        "submission_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--evidence-manifest-sha256", required=True)
    parser.add_argument("--g1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = adjudicate(args.g0, args.g1, args.evidence_manifest,
                             args.evidence_manifest_sha256, args.g1_checkpoint)
        rc = 0
    except Exception as exc:
        payload = {"schema": RESULT_SCHEMA, "status": "INVALID",
                   "error": f"{type(exc).__name__}: {exc}"}
        rc = 2
    receipt = CMP._write_exclusive(args.output, payload)
    CMP._verify_publication_receipt(payload, receipt)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
