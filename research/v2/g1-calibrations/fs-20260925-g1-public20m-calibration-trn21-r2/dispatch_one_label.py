#!/usr/bin/env python3
"""Controller-side adapter for one independently authorized trn21 G1 eval.

This script is intentionally inert for the packaged pending request. It checks
the separate Supervisor authorization and loads the already-proven e7 launcher
only from exact hashed local files; it never reads an SSH key's contents.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys


HERE = Path(__file__).resolve().parent
LABEL = "fs-20260925-g1-public20m-calibration-trn21-r2"
STAGE_REVIEW_SHA256 = "bf3c78a7db65ea7829386129969e7bb06cc700f11c2d4b2f251c448f04878a21"
LAUNCHER_SHA256 = "06eefe82e04a810474614295228e1cef84923373e653a9c62f97f5dca3456b94"
SYNC_SHA256 = "ea428d33f4de18376950abbd81cfda33d3a969f75b60e3648ff552bea04fc60c"
HOST = "ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--operator-checkout", type=Path, required=True)
    parser.add_argument("--ssh-key-path", type=Path, required=True)
    args = parser.parse_args()

    request_path = args.request.resolve(strict=True)
    if request_path == (HERE / "request.guard-pending.json").resolve():
        raise RuntimeError("pending request is never dispatchable")
    request = json.loads(request_path.read_text())
    pending_path = HERE / "request.guard-pending.json"
    pending = json.loads(pending_path.read_text())
    pending_sha = sha256(pending_path)
    authorized_payload = dict(request)
    pending_payload = dict(pending)
    for key in ("status", "launch_authorized", "authorization"):
        authorized_payload.pop(key, None)
        pending_payload.pop(key, None)
    if authorized_payload != pending_payload:
        raise RuntimeError("authorized request changed science, guard, or source fields")
    auth = request.get("authorization")
    if request.get("launch_authorized") is not True or not isinstance(auth, dict):
        raise RuntimeError("separate one-label Supervisor authorization is missing")
    if (request.get("label") != LABEL or request.get("host") != "trn21"
            or auth.get("decision") != "AUTHORIZE_ONE_LABEL_G1_EVALUATOR"
            or auth.get("pending_package_sha256") != pending_sha
            or auth.get("stage_review_sha256") != STAGE_REVIEW_SHA256):
        raise RuntimeError("authorization does not bind this reviewed package and stage")
    verifier = auth.get("guard_package_verifier", {})
    verifier_path = Path(verifier.get("path", ""))
    verifier_sha = verifier.get("sha256", "")
    if (not verifier_path.is_absolute() or not verifier_path.is_file()
            or not re.fullmatch(r"[0-9a-f]{64}", verifier_sha)
            or sha256(verifier_path) != verifier_sha):
        raise RuntimeError("independent guard-package Verifier receipt is absent or changed")
    if not auth.get("supervisor_authorized_at_utc"):
        raise RuntimeError("Supervisor authorization timestamp is absent")
    if request.get("source_stage_binding", {}).get("independent_stage_review_sha256") != STAGE_REVIEW_SHA256:
        raise RuntimeError("stage review binding changed")
    monitor_path = HERE / "guarded_eval.py"
    if request.get("monitor", {}).get("script_sha256") != sha256(monitor_path):
        raise RuntimeError("monitor source does not match authorized manifest")
    if request.get("status") != "AUTHORIZED_ONE_LABEL_G1_EVALUATOR":
        raise RuntimeError("authorized status absent")

    checkout = args.operator_checkout.resolve(strict=True)
    launch_path = checkout / "research/fresh_start_launch.py"
    sync_path = checkout / "research/fresh_start_sync.py"
    if sha256(launch_path) != LAUNCHER_SHA256 or sha256(sync_path) != SYNC_SHA256:
        raise RuntimeError("proven launcher or transport module differs")
    key_path = args.ssh_key_path.resolve(strict=True)
    key_stat = key_path.stat()
    if (not stat.S_ISREG(key_stat.st_mode) or key_stat.st_uid != os.getuid()
            or (key_stat.st_mode & 0o077)):
        raise RuntimeError("SSH key path is not an owned private regular file")
    # The only transport change is the user's configured key path and strict
    # host-key policy. No key bytes or environment secrets are inspected.
    sys.path.insert(0, str(checkout / "research"))
    launcher = importlib.import_module("fresh_start_launch")
    if Path(launcher.__file__).resolve() != launch_path.resolve():
        raise RuntimeError("unexpected launcher import path")
    launcher.HOSTS = {"trn21": HOST}
    launcher.SSH = [
        "ssh", "-i", str(key_path), "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
    ]
    sys.argv = [str(launch_path), str(request_path)]
    launcher.main()


if __name__ == "__main__":
    main()
