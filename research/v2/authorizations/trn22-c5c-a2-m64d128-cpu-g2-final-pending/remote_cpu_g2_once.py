"""Transported trn22 A2 CPU-only audit/one-attempt helper; no standalone authority.

These bytes may only be transported by a separately reviewed, assigned local
Operator wrapper. The helper has no AWS or native-training path. Its `audit`
mode reads only; `run` repeats the full audit in the same SSH process before
one exact contained synthetic CPU G2 attempt. The local wrapper owes an
unconditional independent-process postflight after any host contact.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys


SOURCE = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2"
ROOT = "/home/ubuntu/trainium-v2-g2-a2-c5c-m64d128-20260925-trn22-r4"
TMP = ROOT + "/tmp"
PROCESS = ROOT + "/process"
STEP = ROOT + "/steps.jsonl"
RECEIPT = ROOT + "/g2.json"
LABEL = "fs-20260925-g2-a2-c5c-m64d128-trn22-r4"
HOSTNAME = "ip-172-31-32-253"
COMMIT = "e65a3097a5b00f6178188a935bb2acb9a5042c6c"
TRAIN_SHA = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREPARE_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
G2_SHA = "6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8"
CONTAINED_SHA = "0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d"
PROBE_SHA = "4a8b8ed413240c229ce943dd9bd8d6781bbddf7c348aa76b5b496f5c5d14f70a"
REQUEST_CANONICAL_SHA = "11501dd6f9cac2f825079111fc2e2a72cfff79d269f9e640831d9074b61d8f47"
COMMANDS_CANONICAL_SHA = "c86b171d5fbed2664dd63304f560dcad892ac12f6f01686b2d9d0edb557fae0d"
KEYS = ["ngram_embeds.b1.weight", "ngram_embeds.t1.weight",
        "ngram_embeds.b5.weight", "ngram_embeds.t5.weight"]
MAX_PAYLOAD = 2_000_000
MAX_RESULT = 3_000_000


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def unique_json(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, "duplicate JSON key: " + key)
        out[key] = value
    return out


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def authenticate_payload(payload: dict) -> None:
    """Pin both request and executable command data inside the remote process."""
    require(type(payload) is dict and
            set(payload) == {"request", "commands", "probe_source"} and
            type(payload["probe_source"]) is str and
            sha(payload["probe_source"].encode()) == PROBE_SHA and
            sha(canonical(payload["request"])) == REQUEST_CANONICAL_SHA and
            sha(canonical(payload["commands"])) == COMMANDS_CANONICAL_SHA,
            "CPU-only request, command arrays or read-only probe changed")


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            h.update(block)
    finally:
        os.close(fd)
    return h.hexdigest()


def snapshot(probe_source: str) -> dict:
    """Execute the pinned read-only probe in-process and add exact A2 bindings."""
    require(sha(probe_source.encode()) == PROBE_SHA, "read-only probe source drift")
    namespace = {"__name__": "a2_pinned_readonly_probe"}
    exec(compile(probe_source, "<pinned-readonly-probe>", "exec"), namespace)
    old_argv = sys.argv
    output = io.StringIO()
    try:
        sys.argv = ["remote_snapshot.py", "trn22"]
        with contextlib.redirect_stdout(output):
            namespace["main"]()
    finally:
        sys.argv = old_argv
    lines = output.getvalue().splitlines()
    require(len(lines) == 1, "read-only probe did not emit one snapshot")
    base = json.loads(lines[0], object_pairs_hook=unique_json)
    mem = None
    with open("/proc/meminfo", encoding="ascii") as source:
        for line in source:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                require(len(fields) == 3 and fields[2] == "kB", "bad MemAvailable field")
                mem = int(fields[1]) * 1024
                break
    require(type(mem) is int and mem > 0, "MemAvailable missing")
    pointer_path = SOURCE + "/.git"
    pointer = namespace["bounded_hash"](pointer_path)
    require(pointer.get("type") == "regular" and pointer.get("bytes", 0) < 4096,
            "linked .git pointer absent or unsafe")
    fd = os.open(pointer_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        raw_pointer = os.read(fd, 4096)
    finally:
        os.close(fd)
    require(sha(raw_pointer) == pointer.get("sha256"), "git pointer raced")
    match = re.fullmatch(rb"gitdir: ([^\n]+)\n", raw_pointer)
    require(match is not None, "unexpected linked gitdir pointer")
    linked_path = os.path.normpath(os.path.join(SOURCE, os.fsdecode(match.group(1))))
    linked = namespace["named"](linked_path)
    paths = (ROOT, TMP, PROCESS, STEP, RECEIPT)
    scratch = {path: namespace["named"](path) for path in paths}
    if scratch[RECEIPT].get("type") == "regular":
        scratch[RECEIPT] = namespace["bounded_hash"](RECEIPT)
    find = subprocess.run(["/usr/bin/find", "/home/ubuntu", "-xdev", "-name",
                           "*" + LABEL + "*", "-print"], capture_output=True,
                          timeout=60, check=False)
    require(find.returncode == 0 and not find.stderr and len(find.stdout) < 131072,
            "unique-label read-only scan failed")
    return {"base": base, "memory_available_bytes": mem,
            "git_pointer": pointer, "linked_gitdir_path": linked_path,
            "linked_gitdir": linked, "scratch": scratch,
            "g2_tool": namespace["bounded_hash"](SOURCE + "/workflow/tools/preflight_dryrun.py"),
            "contained_tool": namespace["bounded_hash"](SOURCE + "/workflow/tools/contained.py"),
            "label_matches": find.stdout.decode().splitlines()}


def validate_snapshot(item: dict, request: dict, *, phase: str) -> dict:
    """Fail closed on the full live snapshot; postflight cannot certify G2."""
    require(phase in ("before", "prepared", "after"), "unknown snapshot phase")
    base = item["base"]
    identity, space, activity = base["identity"], base["root_space"], base["activity"]
    require(base.get("schema") == "dual-host-readonly-v1" and base.get("host") == "trn22" and
            identity.get("hostname") == HOSTNAME and identity.get("uid") == 1000 and
            re.fullmatch(r"[0-9a-f-]{36}", identity.get("boot_id", "")) is not None and
            identity.get("root_mount") == identity.get("home_mount") and
            identity.get("root_mount", {}).get("filesystem") == "ext4" and
            identity.get("root_device", {}).get("device") == 66305,
            "host, boot or root mount identity failed")
    capacity = request["capacity"]
    require(type(space.get("free_bytes")) is int and
            type(space.get("total_bytes")) is int and
            240 * 1024 ** 3 <= space["total_bytes"] <= 260 * 1024 ** 3 and
            space["free_bytes"] >= capacity["minimum_fresh_root_free_bytes"] and
            space["free_bytes"] >= capacity["scratch_peak_projection_bytes"] +
                                    capacity["operational_root_floor_bytes"] and
            type(space.get("free_inodes")) is int and
            space["free_inodes"] >= capacity["minimum_fresh_root_free_inodes"] and
            item["memory_available_bytes"] >=
                capacity["minimum_fresh_memory_available_bytes"] and
            item["memory_available_bytes"] >= capacity["projected_peak_memory_bytes"],
            "fresh disk/inode/memory capacity gate failed")
    require(base["stops"].get("present") == {} and
            base["checkouts"].get("discovery_truncated") is False and
            activity["processes"] == {"matched": [], "unreadable_cmdlines": 0} and
            activity["neuron"].get("rc") == 0 and
            activity["neuron"].get("json_valid") is True and
            type(activity["neuron"].get("device_count")) is int and
            activity["neuron"]["device_count"] >= 1 and
            activity["neuron"].get("non_diagnostic_pids") == [] and
            activity["host_lock_fuser"].get("path_exists") is True and
            activity["host_lock_fuser"].get("rc") == 1 and
            activity["host_lock_fuser"].get("pids") == [] and
            activity["neuron0_fuser"].get("path_exists") is True and
            activity["neuron0_fuser"].get("rc") == 1 and
            activity["neuron0_fuser"].get("pids") == [],
            "STOP, process, lock or Neuron occupancy gate failed")
    source = base["checkouts"]["entries"]["e65_exact"]
    host = request["host"]
    entry = source["entry"]
    require(source.get("path") == SOURCE and entry.get("type") == "directory" and
            (entry.get("device"), entry.get("inode"), entry.get("uid"), entry.get("mode")) ==
            (66305, 2432738, 1000, "0775") and
            source["git"].get("commit") == COMMIT and source["git"].get("rc") == 0 and
            source["git_status"].get("clean") is True and
            item["git_pointer"].get("sha256") == host["git_pointer_sha256"] and
            item["git_pointer"].get("inode") == host["git_pointer_inode"] and
            item["git_pointer"].get("mode") == host["git_pointer_mode"] and
            item["linked_gitdir"].get("type") == "directory" and
            item["linked_gitdir"].get("inode") == host["linked_gitdir_inode"] and
            item["linked_gitdir"].get("mode") == host["linked_gitdir_mode"],
            "linked checkout or Git identity changed")
    hashes = source["source_hashes"]
    require(hashes["train.py"].get("sha256") == TRAIN_SHA and
            hashes["prepare.py"].get("sha256") == PREPARE_SHA and
            item["g2_tool"].get("sha256") == G2_SHA and
            item["contained_tool"].get("sha256") == CONTAINED_SHA,
            "source, evaluator or CPU tool bytes changed")
    scratch = item["scratch"]
    require(set(scratch) == {ROOT, TMP, PROCESS, STEP, RECEIPT},
            "exact scratch inventory missing")
    if phase == "before":
        require(all(value.get("exists") is False for value in scratch.values()) and
                item["label_matches"] == [], "new label or scratch already used")
    elif phase == "prepared":
        require(all(scratch[path].get("exists") is False for path in (PROCESS, STEP, RECEIPT)) and
                all(scratch[path].get("type") == "directory" and
                    scratch[path].get("device") == 66305 and
                    scratch[path].get("uid") == 1000 and
                    scratch[path].get("mode") == "0700" for path in (ROOT, TMP)),
                "prepared scratch is not exact and private")
    else:
        require(scratch[ROOT].get("exists") is False or
                (scratch[ROOT].get("type") == "directory" and
                 scratch[ROOT].get("uid") == 1000), "postflight root is unsafe")
    return item


def _preexec_bridge(prepared: dict) -> None:
    """Last no-subprocess source/boot/STOP check in launch child before exec."""
    identity = prepared["base"]["identity"]
    require(socket.gethostname() == HOSTNAME and os.geteuid() == 1000 and
            Path("/proc/sys/kernel/random/boot_id").read_text().strip() ==
                identity["boot_id"] and
            os.readlink("/proc/self/ns/mnt") == identity["mount_namespace"],
            "pre-exec host boot/namespace changed")
    root_mount = None
    with open("/proc/self/mountinfo", encoding="utf-8") as mounts:
        for line in mounts:
            left, sep, right = line.partition(" - ")
            if sep and len(left.split()) >= 5 and left.split()[4] == "/":
                fields, fs = left.split(), right.split()
                root_mount = {"mount_id": fields[0], "major_minor": fields[2],
                              "point": "/", "filesystem": fs[0], "source": fs[1]}
    require(root_mount == identity["root_mount"], "pre-exec root mount changed")
    source = os.lstat(SOURCE)
    pointer = os.lstat(SOURCE + "/.git")
    linked = os.lstat(prepared["linked_gitdir_path"])
    require((source.st_dev, source.st_ino, stat.S_IMODE(source.st_mode)) ==
                (66305, 2432738, 0o775) and
            (pointer.st_ino, stat.S_IMODE(pointer.st_mode)) ==
                (2432740, 0o664) and
            stat.S_ISDIR(linked.st_mode) and
            (linked.st_ino, stat.S_IMODE(linked.st_mode)) == (2432736, 0o775) and
            file_sha(SOURCE + "/.git") ==
                "dc53f6887bc5f9c2c1bbdbbd60911cdbeb076f7c57bafd796ee5b96f69ecbf1b",
            "pre-exec linked checkout changed")
    require(file_sha(SOURCE + "/train.py") == TRAIN_SHA and
            file_sha(SOURCE + "/prepare.py") == PREPARE_SHA and
            file_sha(SOURCE + "/workflow/tools/preflight_dryrun.py") == G2_SHA and
            file_sha(SOURCE + "/workflow/tools/contained.py") == CONTAINED_SHA,
            "pre-exec source or CPU tool bytes changed")
    stop_paths = ["/home/ubuntu/research/agentic/STOP", "/home/ubuntu/research/v2/STOP"]
    for entry in os.scandir("/home/ubuntu"):
        if entry.name.startswith("trainium_freshstart_") and entry.is_dir(follow_symlinks=False):
            stop_paths.extend((entry.path + "/research/agentic/STOP",
                               entry.path + "/research/v2/STOP"))
    require(all(not os.path.lexists(path) for path in stop_paths),
            "host STOP appeared immediately before launch")


def _run(argv: list[str], timeout: int, *, preexec_fn=None) -> dict:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          check=False, preexec_fn=preexec_fn)
    require(len(done.stdout) < MAX_RESULT and len(done.stderr) < MAX_RESULT,
            "CPU command output too large")
    return {"argv_sha256": sha(json.dumps(argv, separators=(",", ":")).encode()),
            "returncode": done.returncode, "stdout": done.stdout, "stderr": done.stderr}


def _raw_artifacts() -> dict:
    receipt = Path(RECEIPT)
    require(receipt.is_file() and not receipt.is_symlink(), "G2 receipt missing")
    raw = receipt.read_bytes()
    require(len(raw) < MAX_RESULT, "G2 receipt too large")
    parsed = json.loads(raw, object_pairs_hook=unique_json)
    require(parsed.get("verdict") == "PASS" and parsed.get("steps") == 2 and
            parsed.get("nproc") == 4 and parsed.get("train_py_sha256") == TRAIN_SHA,
            "G2 tool did not return exact two-step CPU PASS")
    workdirs = sorted(Path(TMP).glob("g2-*"))
    require(len(workdirs) == 1 and workdirs[0].is_dir() and not workdirs[0].is_symlink(),
            "retained G2 workdir not unique")
    work = workdirs[0]
    ranks = {}
    for index in range(4):
        path = work / f"rank{index}.json"
        require(path.is_file() and not path.is_symlink(), "raw rank record missing")
        rank_raw = path.read_bytes()
        require(len(rank_raw) < MAX_RESULT, "raw rank record too large")
        rank = json.loads(rank_raw, object_pairs_hook=unique_json)
        checks = rank.get("rmsprop_trace", {}).get("checks")
        require(type(checks) is list and len(checks) == 8,
                "raw eight-record RMS trace missing")
        ranks[str(index)] = {"sha256": sha(rank_raw), "json": rank}
    require(parsed.get("ranks") == [ranks[str(i)]["json"] for i in range(4)],
            "aggregate/raw rank receipt mismatch")
    require(Path(STEP).is_file() and not Path(STEP).is_symlink(),
            "step log absent or symlinked")
    step_raw = Path(STEP).read_bytes()
    require(step_raw == b"0\n1\n", "two ordered optimizer boundaries missing")
    checkpoint = work / "out/final.pt"
    require(checkpoint.is_file() and not checkpoint.is_symlink(), "retained checkpoint missing")
    return {"g2_receipt_sha256": sha(raw), "g2_receipt_json": parsed,
            "rank_records": ranks, "step_log_sha256": sha(step_raw),
            "checkpoint_sha256": file_sha(str(checkpoint)),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "torchrun_log_sha256": file_sha(str(work / "torchrun.log")),
            "contained_run_log_sha256": file_sha(PROCESS + "/run.log")}


def run_once(payload: dict) -> dict:
    """One reviewed CPU attempt; preserve partial scratch and raw records."""
    request, commands = payload["request"], payload["commands"]
    records = []
    launch_attempted = False
    try:
        first = snapshot(payload["probe_source"])
        validate_snapshot(first, request, phase="before")
        prep = _run(commands["prepare_scratch_after_marker"], 120)
        records.append(prep)
        require(prep["returncode"] == 0 and not prep["stderr"].strip(),
                "exact scratch/TMPDIR creation failed")
        prepared = snapshot(payload["probe_source"])
        validate_snapshot(prepared, request, phase="prepared")
        require(prepared["base"]["identity"]["boot_id"] ==
                first["base"]["identity"]["boot_id"] and
                prepared["base"]["identity"]["root_mount"] ==
                first["base"]["identity"]["root_mount"],
                "boot/mount changed before launch")
        # The exact contained child rechecks TMPDIR and c5c/prepare/G2 bytes
        # in its own interpreter before the pinned tool's mkdtemp call.
        launch_attempted = True
        launch = _run(commands["launch"], 180,
                      preexec_fn=lambda: _preexec_bridge(prepared))
        records.append(launch)
        require(launch["returncode"] == 0 and not launch["stderr"].strip(),
                "contained launch failed")
        wait = _run(commands["wait"], 4500)
        records.append(wait)
        require(wait["returncode"] == 0 and not wait["stderr"].strip(),
                "contained wait failed")
        quiesce = _run(commands["quiesce"], 180)
        records.append(quiesce)
        require(quiesce["returncode"] == 0 and not quiesce["stderr"].strip(),
                "contained quiescence failed")
        artifacts = _raw_artifacts()
        return {"status": "CPU_G2_RAW_EVIDENCE_PENDING_INDEPENDENT_NUMERIC_REVIEW",
                "cpu_only": True, "native_training_authorized": False,
                "evaluation_authorized": False, "records": records,
                "artifacts": artifacts}
    except BaseException as error:
        if launch_attempted:
            for name in ("stop_if_needed", "quiesce"):
                try:
                    records.append(_run(commands[name], 180))
                except BaseException as stop_error:
                    records.append({"containment_error": type(stop_error).__name__ +
                                    ": " + str(stop_error)[:300]})
        return {"status": "HOLD_PRESERVE_PARTIAL_NO_RETRY", "cpu_only": True,
                "native_training_authorized": False,
                "evaluation_authorized": False, "records": records,
                "error_type": type(error).__name__, "error": str(error)[:500]}


def main() -> int:
    require(len(sys.argv) == 3 and sys.argv[1] in ("audit", "run") and
            re.fullmatch(r"[0-9a-f]{64}", sys.argv[2]) is not None,
            "exact mode and payload SHA required")
    raw = sys.stdin.buffer.read(MAX_PAYLOAD + 1)
    require(len(raw) <= MAX_PAYLOAD and sha(raw) == sys.argv[2],
            "remote payload SHA/size mismatch")
    payload = json.loads(raw, object_pairs_hook=unique_json)
    authenticate_payload(payload)
    if sys.argv[1] == "audit":
        result = {"status": "READ_ONLY_A2_G2_AUDIT", "snapshot": snapshot(payload["probe_source"]),
                  "native_training_authorized": False, "evaluation_authorized": False}
    else:
        result = run_once(payload)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0 if result["status"] != "HOLD_PRESERVE_PARTIAL_NO_RETRY" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
        print(json.dumps({"status": "HOLD_PRESERVE_PARTIAL_NO_RETRY",
                          "error_type": type(error).__name__,
                          "error": str(error)[:500]}, sort_keys=True), flush=True)
        raise SystemExit(2) from error
