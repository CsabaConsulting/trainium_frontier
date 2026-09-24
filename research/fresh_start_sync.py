#!/usr/bin/env python3
"""Retrieve this campaign's live logs and merge completed records without loss.

Run locally; no remote training is launched or interrupted. Repeated invocations
are safe. Host copies of the old research log are never allowed to overwrite the
canonical local log. Preserve full differing records, keyed by source provenance.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/home/ubuntu/trainium_freshstart_20260918"
HOSTS = {
    "trn21": "ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com",
    "trn22": "ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com",
}
SSH = ["ssh", "-i", "/home/csaba/.ssh/trn2.pem", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=12"]


def rows(path):
    if not path.exists():
        return []
    parsed = []
    for number, raw in enumerate(path.read_bytes().replace(b"\0", b"").splitlines(), 1):
        try:
            parsed.append(json.loads(raw))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            print(f"WARNING: {path}:{number}: preserved source, skipped invalid row: {error}", flush=True)
    return parsed


def identity(row):
    return tuple(str(row.get(k, "")) for k in
                 ("label", "timestamp", "hostname", "git_commit", "launch_command"))


def sync_once(remote=REMOTE, prefix="fs-", hosts=None):
    incoming = ROOT / "out" / "fresh-start-sync"
    incoming.mkdir(parents=True, exist_ok=True)
    index = ROOT / "research" / "experiment-log.jsonl"
    count = 0
    failures = []
    for label, host in (HOSTS if hosts is None else hosts).items():
        try:
            subprocess.run(
                ["rsync", "-az", "--timeout=45", "-e", " ".join(SSH),
                 f"--include={prefix}*", "--exclude=*", f"{host}:{remote}/research/run-logs/",
                 str(ROOT / "research" / "run-logs") + "/"], check=True, timeout=55)
            remote_index = incoming / f"{label}-{Path(remote).name}-experiment-log.jsonl"
            subprocess.run(["rsync", "-az", "--timeout=45", "-e", " ".join(SSH),
                            f"{host}:{remote}/research/experiment-log.jsonl", str(remote_index)],
                           check=True, timeout=55)
            with (incoming / "merge.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                known = {identity(row) for row in rows(index)}
                with index.open("a") as output:
                    for row in rows(remote_index):
                        if not row.get("label", "").startswith(prefix) or identity(row) in known:
                            continue
                        output.write(json.dumps(row) + "\n")
                        known.add(identity(row))
                        count += 1
                    output.flush()
                    os.fsync(output.fileno())
        except (subprocess.SubprocessError, OSError) as error:
            failures.append(label)
            print(f"WARNING: sync {label}: {error}", flush=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"{stamp}: live logs synchronized; {count} completed records added; failures={failures}", flush=True)
    return not failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--remote", default=REMOTE,
                        help="immutable remote checkout to retrieve")
    parser.add_argument("--prefix", default="fs-",
                        help="labels owned by this checkout; use a new prefix each wave")
    parser.add_argument("--host", action="append", choices=tuple(HOSTS), default=None,
                        help="retrieve only these hosts; repeatable, defaults to both")
    args = parser.parse_args()
    if args.rounds < 1 or args.interval < 1:
        parser.error("rounds and interval must be positive")
    selected = {key: HOSTS[key] for key in args.host} if args.host else HOSTS
    for round_number in range(args.rounds):
        sync_once(args.remote, args.prefix, selected)
        if round_number + 1 < args.rounds:
            time.sleep(min(args.interval, 60))


if __name__ == "__main__":
    main()
