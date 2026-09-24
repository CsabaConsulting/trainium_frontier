#!/usr/bin/env python3
"""Read-only A2 smoke observer with an external, preflighted STEP_LOG parent.

The reviewed A0 observer accepts separate boundary-file and guard-log paths in
its observe() function, but its CLI requires one shared parent. This adapter
authenticates a label-owned scratch directory outside the pinned source tree,
then delegates process, guard, boundary-stream and postflight observation to
the unchanged, hash-pinned A0 implementation. It never launches or stops work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import types


BASE_OBSERVER_SHA256 = "a6f437fe8a9d0a14ec3025bc822f1afa12e3f355fca730e4ae0db7ae00263592"


def _absolute_lexically_clean(path: Path) -> Path:
    if not path.is_absolute() or str(path) != os.path.abspath(path):
        raise ValueError(f"noncanonical absolute path: {path}")
    return path


def _directory_without_symlinks(path: Path) -> os.stat_result:
    _absolute_lexically_clean(path)
    current = Path("/")
    for part in path.parts[1:]:
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"directory component is missing, symlinked or not a directory: {current}")
    return current.stat()


def _read_single_link_regular(path: Path) -> tuple[bytes, os.stat_result]:
    _directory_without_symlinks(path.parent)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"not a single-link regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), info
    finally:
        os.close(fd)


def _inside(path: Path, directory: Path) -> bool:
    return os.path.commonpath((str(path), str(directory))) == str(directory)


def validate(a: argparse.Namespace) -> dict:
    if not re.fullmatch(r"fs-[A-Za-z0-9_.-]+", a.label):
        raise ValueError("noncanonical fs-* label")
    for name in (
        "source_checkout", "scratch_root", "base_observer", "guard_script",
        "process_json", "postflight_json", "boundary_file", "write_probe",
    ):
        _absolute_lexically_clean(getattr(a, name))
    source = a.source_checkout
    scratch = a.scratch_root
    source_info = _directory_without_symlinks(source)
    scratch_info = _directory_without_symlinks(scratch)
    parent = scratch / a.label
    parent_info = _directory_without_symlinks(parent)
    if scratch_info.st_uid != a.expected_run_uid or stat.S_IMODE(scratch_info.st_mode) & 0o022:
        raise ValueError("scratch root is not owned by run UID or is group/world writable")
    if stat.S_IMODE(parent_info.st_mode) & 0o077:
        raise ValueError("label STEP_LOG parent must be private to run UID")
    if a.boundary_file != parent / f"{a.label}.steps.jsonl":
        raise ValueError("STEP_LOG is not the exact label file in its dedicated scratch directory")
    if _inside(parent, source) or _inside(source, parent):
        raise ValueError("STEP_LOG parent overlaps the clean source checkout")
    if os.path.lexists(a.boundary_file):
        raise ValueError("STEP_LOG file already exists before observer start")
    if a.expected_run_uid != os.geteuid() or parent_info.st_uid != a.expected_run_uid:
        raise ValueError("observer/run UID or STEP_LOG parent owner mismatch")
    if not os.access(parent, os.W_OK | os.X_OK, effective_ids=True):
        raise ValueError("STEP_LOG parent is not writable/searchable by run UID")
    expected_parent = (
        a.parent_device, a.parent_inode, a.parent_uid, a.parent_gid, a.parent_mode
    )
    observed_parent = (
        parent_info.st_dev, parent_info.st_ino, parent_info.st_uid,
        parent_info.st_gid, stat.S_IMODE(parent_info.st_mode),
    )
    if observed_parent != expected_parent:
        raise ValueError("STEP_LOG parent identity/ownership/mode changed since stage")
    if a.write_probe != parent / f"{a.label}.write-probe":
        raise ValueError("positive write probe is not the exact label-owned sibling")
    probe, probe_info = _read_single_link_regular(a.write_probe)
    if not probe or probe_info.st_uid != a.expected_run_uid:
        raise ValueError("positive write probe is empty or not owned by run UID")
    probe_sha = hashlib.sha256(probe).hexdigest()
    if probe_sha != a.write_probe_sha256:
        raise ValueError("positive write probe hash mismatch")
    logs = source / "research" / "run-logs"
    logs_info = _directory_without_symlinks(logs)
    if logs_info.st_uid != a.expected_run_uid or not os.access(
        logs, os.W_OK | os.X_OK, effective_ids=True
    ):
        raise ValueError("guard receipt parent is not owned and writable/searchable by run UID")
    observed_logs = (
        logs_info.st_dev, logs_info.st_ino, logs_info.st_uid,
        logs_info.st_gid, stat.S_IMODE(logs_info.st_mode),
    )
    expected_logs = (
        a.guard_parent_device, a.guard_parent_inode, a.guard_parent_uid,
        a.guard_parent_gid, a.guard_parent_mode,
    )
    if observed_logs != expected_logs:
        raise ValueError("guard receipt parent identity/ownership/mode changed since stage")
    if a.process_json != logs / f"{a.label}.process.json":
        raise ValueError("guard process receipt path mismatch")
    if a.postflight_json != logs / f"{a.label}.postflight.json":
        raise ValueError("guard postflight receipt path mismatch")
    if a.guard_script != source / "research" / "fresh_start_guard.py":
        raise ValueError("guard script is not in the pinned source checkout")
    guard_bytes, _ = _read_single_link_regular(a.guard_script)
    guard_sha = hashlib.sha256(guard_bytes).hexdigest()
    if guard_sha != a.guard_script_sha256:
        raise ValueError("staged guard script hash mismatch")
    if _inside(a.base_observer, source):
        raise ValueError("reviewed observer must be staged outside the clean source checkout")
    base_bytes, _ = _read_single_link_regular(a.base_observer)
    if hashlib.sha256(base_bytes).hexdigest() != BASE_OBSERVER_SHA256:
        raise ValueError("reviewed A0 observer hash mismatch")
    preflight = {
        "source_device": source_info.st_dev,
        "source_inode": source_info.st_ino,
        "scratch_root": str(scratch),
        "step_parent": str(parent),
        "step_parent_device": parent_info.st_dev,
        "step_parent_inode": parent_info.st_ino,
        "step_parent_uid": parent_info.st_uid,
        "step_parent_gid": parent_info.st_gid,
        "step_parent_mode": stat.S_IMODE(parent_info.st_mode),
        "guard_receipt_parent_device": logs_info.st_dev,
        "guard_receipt_parent_inode": logs_info.st_ino,
        "guard_receipt_parent_uid": logs_info.st_uid,
        "guard_receipt_parent_gid": logs_info.st_gid,
        "guard_receipt_parent_mode": stat.S_IMODE(logs_info.st_mode),
        "run_uid": a.expected_run_uid,
        "write_probe_sha256": probe_sha,
        "guard_script_sha256": guard_sha,
        "reviewed_base_observer_sha256": BASE_OBSERVER_SHA256,
    }
    return preflight, base_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--base-observer", required=True, type=Path)
    parser.add_argument("--guard-script", required=True, type=Path)
    parser.add_argument("--guard-script-sha256", required=True)
    parser.add_argument("--process-json", required=True, type=Path)
    parser.add_argument("--postflight-json", required=True, type=Path)
    parser.add_argument("--boundary-file", required=True, type=Path)
    parser.add_argument("--write-probe", required=True, type=Path)
    parser.add_argument("--write-probe-sha256", required=True)
    parser.add_argument("--expected-run-uid", required=True, type=int)
    parser.add_argument("--parent-device", required=True, type=int)
    parser.add_argument("--parent-inode", required=True, type=int)
    parser.add_argument("--parent-uid", required=True, type=int)
    parser.add_argument("--parent-gid", required=True, type=int)
    parser.add_argument("--parent-mode", required=True, type=int)
    parser.add_argument("--guard-parent-device", required=True, type=int)
    parser.add_argument("--guard-parent-inode", required=True, type=int)
    parser.add_argument("--guard-parent-uid", required=True, type=int)
    parser.add_argument("--guard-parent-gid", required=True, type=int)
    parser.add_argument("--guard-parent-mode", required=True, type=int)
    parser.add_argument("--poll-seconds", type=float, default=0.02)
    parser.add_argument("--timeout-seconds", type=float, default=4800.0)
    a = parser.parse_args()
    if a.poll_seconds <= 0 or a.timeout_seconds <= 0:
        parser.error("poll and timeout must be positive")
    try:
        preflight, base_bytes = validate(a)
        module = types.ModuleType("reviewed_a0_boundary_observer")
        module.__file__ = str(a.base_observer)
        exec(compile(base_bytes, str(a.base_observer), "exec"), module.__dict__)
        result = module.observe(a)
        parent_final = _directory_without_symlinks(a.boundary_file.parent)
        final_identity = (
            parent_final.st_dev, parent_final.st_ino, parent_final.st_uid,
            parent_final.st_gid, stat.S_IMODE(parent_final.st_mode),
        )
        expected_identity = (
            a.parent_device, a.parent_inode, a.parent_uid,
            a.parent_gid, a.parent_mode,
        )
        parent_stable = final_identity == expected_identity
        logs_final = _directory_without_symlinks(a.process_json.parent)
        logs_stable = (
            logs_final.st_dev == a.guard_parent_device
            and logs_final.st_ino == a.guard_parent_inode
            and logs_final.st_uid == a.guard_parent_uid
            and logs_final.st_gid == a.guard_parent_gid
            and stat.S_IMODE(logs_final.st_mode) == a.guard_parent_mode
        )
        probe_final, _ = _read_single_link_regular(a.write_probe)
        probe_stable = hashlib.sha256(probe_final).hexdigest() == a.write_probe_sha256
        values = [row.get("value") for row in result.get("boundaries", [])]
        passed = (
            result.get("scored_timing_proof") is True
            and values == list(range(30)) and parent_stable
            and logs_stable and probe_stable
        )
        result.update({
            "purpose": "non-scored c5c A2 native-30 smoke",
            "step_parent_preflight": preflight,
            "step_parent_postflight": {
                "device": parent_final.st_dev, "inode": parent_final.st_ino,
                "uid": parent_final.st_uid, "gid": parent_final.st_gid,
                "mode": stat.S_IMODE(parent_final.st_mode),
                "identity_unchanged": parent_stable,
                "write_probe_unchanged": probe_stable,
            },
            "guard_receipt_parent_unchanged": logs_stable,
            "smoke_boundary_proof": passed,
        })
        if not parent_stable or not logs_stable or not probe_stable:
            result.setdefault("invalid_reasons", []).append("receipt_parent_or_probe_changed_during_smoke")
        if not passed:
            result.setdefault("invalid_reasons", []).append("smoke_requires_exactly_30_boundaries_0_to_29")
        print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
        return 0 if passed else 2
    except (OSError, ValueError) as exc:
        print(json.dumps({
            "schema": "a2-native30-boundary-adapter-v1",
            "label": a.label,
            "purpose": "non-scored c5c A2 native-30 smoke",
            "smoke_boundary_proof": False,
            "invalid_reasons": [f"observer_preflight:{exc}"],
        }, sort_keys=True, separators=(",", ":")), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
