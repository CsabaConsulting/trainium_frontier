"""Transported CPU-only G2 audit/one-attempt adapter; no standalone authority.

The reviewed local wrapper transports these exact bytes with ``python3 -c``.
This module does not import train.py or prepare.py and never runs native training
or public evaluation. A full read-only audit precedes the one synthetic CPU
attempt, and a same-process identity/STOP bridge runs immediately before the
frozen contained.py launch. The local wrapper performs a separate unconditional
read-only postflight SSH call after any attempted host contact.
"""

import contextlib
import datetime
import hashlib
import io
import json
import os
import re
import socket
import stat
import subprocess
import sys


SOURCE = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2"
ROOT = "/home/ubuntu/trainium-v2-g2-20260925-trn21-stage-r2"
LABEL = "fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2"
PROCESS_DIR = ROOT + "/process-" + LABEL
STEP_LOG = ROOT + "/" + LABEL + ".steps.jsonl"
CPU_RECEIPT = ROOT + "/c5c-g2.json"
STOP_PATHS = (
    "/home/ubuntu/research/agentic/STOP",
    "/home/ubuntu/research/v2/STOP",
    "/home/ubuntu/trainium_freshstart_20260918/research/agentic/STOP",
    "/home/ubuntu/trainium_freshstart_20260918/research/v2/STOP",
    SOURCE + "/research/agentic/STOP",
    SOURCE + "/research/v2/STOP",
)
FLOOR = 10737418240
MAX_PAYLOAD = 262144
MAX_RECEIPT = 1048576


def digest(data):
    return hashlib.sha256(data).hexdigest()


def argv_digest(argv):
    return digest(json.dumps(argv, separators=(",", ":")).encode("utf-8"))


def unique_json(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise RuntimeError("duplicate JSON key: " + key)
        obj[key] = value
    return obj


def stat_path(path, *, hash_file=False):
    try:
        named = os.lstat(path)
    except FileNotFoundError:
        return {"exists": False}
    kind = ("regular" if stat.S_ISREG(named.st_mode) else
            "directory" if stat.S_ISDIR(named.st_mode) else "other")
    result = {"exists": True, "type": kind, "device": named.st_dev,
              "inode": named.st_ino, "uid": named.st_uid, "gid": named.st_gid,
              "mode": oct(stat.S_IMODE(named.st_mode)), "nlink": named.st_nlink,
              "bytes": named.st_size}
    if hash_file:
        if kind != "regular" or named.st_nlink != 1:
            raise RuntimeError("hashed target is not one regular file: " + path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            before = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns):
                raise RuntimeError("hashed file changed before read: " + path)
            h = hashlib.sha256()
            count = 0
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
                count += len(chunk)
            after = os.fstat(fd)
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                    before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
                raise RuntimeError("hashed file changed during read: " + path)
            if count != named.st_size:
                raise RuntimeError("hashed file byte count changed: " + path)
            result["sha256"] = h.hexdigest()
        finally:
            os.close(fd)
    return result


def checked_command(argv, *, timeout=60):
    process = subprocess.run(argv, capture_output=True, text=True, check=False,
                             timeout=timeout)
    if process.returncode != 0 or process.stderr.strip():
        raise RuntimeError("read-only preflight command failed: " + argv[0])
    return process.stdout


def fresh_git():
    prefix = ["git", "-C", SOURCE]
    commands = {
        "head": ["rev-parse", "HEAD"],
        "count": ["rev-list", "--all", "--count"],
        "shallow": ["rev-parse", "--is-shallow-repository"],
        "status": ["status", "--porcelain", "--untracked-files=all"],
        "remote": ["remote"],
    }
    return {name: checked_command(prefix + argv, timeout=30)
            for name, argv in commands.items()}


def archived_full_probe(source):
    out = io.StringIO()
    namespace = {"__name__": "reviewed_readonly_preflight",
                 "__file__": "<sha256-pinned-full-readonly-probe>"}
    with contextlib.redirect_stdout(out):
        exec(compile(source, namespace["__file__"], "exec"), namespace)
    lines = out.getvalue().splitlines()
    if len(lines) != 1:
        raise RuntimeError("full read-only probe did not emit one JSON snapshot")
    return json.loads(lines[0], object_pairs_hook=unique_json)


def new_label_matches():
    output = checked_command(["find", "/home/ubuntu", "-xdev", "-name",
                              "*" + LABEL + "*", "-print"], timeout=90)
    return output.splitlines()


def collect_snapshot(probe_source):
    """Full reviewed probe plus fresh exact source/tool/new-G2 observations."""
    old = archived_full_probe(probe_source)
    paths = (ROOT, PROCESS_DIR, STEP_LOG, CPU_RECEIPT)
    return {
        "schema": "trn21-c5c-stage-r2-cpu-g2-full-readonly-v1",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "old": old,
        "source_checkout": stat_path(SOURCE),
        "source_train": stat_path(SOURCE + "/train.py", hash_file=True),
        "source_prepare": stat_path(SOURCE + "/prepare.py", hash_file=True),
        "git": fresh_git(),
        "tools": {
            "preflight": stat_path("/home/ubuntu/trainium-v2-g2-preflight-dryrun-20260925-trn21-stage-r2.py", hash_file=True),
            "contained": stat_path("/home/ubuntu/trainium-v2-g2-contained-20260925-trn21-stage-r2.py", hash_file=True),
        },
        "scratch": {path: stat_path(path, hash_file=(path == CPU_RECEIPT and
                                                    os.path.isfile(path))) for path in paths},
        "cpu_label_matches": new_label_matches(),
    }


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def same_fields(observed, expected):
    return isinstance(observed, dict) and all(observed.get(key) == value
                                              for key, value in expected.items())


def validate_snapshot(snapshot, request, checkout, *, phase):
    """Fail closed on a full live preflight or separate postflight snapshot."""
    require(phase in ("before", "after") and
            snapshot.get("schema") == "trn21-c5c-stage-r2-cpu-g2-full-readonly-v1",
            "wrong CPU G2 snapshot phase/schema")
    old = snapshot["old"]
    host = checkout["expected_host_identity"]
    time = datetime.datetime.fromisoformat(snapshot["utc"])
    now = datetime.datetime.now(datetime.timezone.utc)
    require(time.tzinfo is not None and abs((now - time).total_seconds()) <= 180,
            "CPU G2 snapshot is stale")
    require(old.get("schema") == "trn21-c5c-stage-r2-readonly-prewrite-v1" and
            old.get("hostname") == host["hostname"] == request["expected_hostname"] and
            old.get("uid") == host["uid"] == 1000 and
            old.get("boot_id") == host["boot_id"] and
            old.get("mount_namespace") == host["mount_namespace"] and
            old.get("root_mount") == host["root_mount"] and
            old.get("home_mount") == host["root_mount"] and
            old.get("root_free_bytes", 0) >= FLOOR,
            "host/boot/mount/disk identity failed")
    for path, expected in host["ancestry"].items():
        require(same_fields(old["ancestry"].get(path), expected),
                "host ancestry changed: " + path)
    for path, expected in checkout["persistent_targets"].items():
        require(same_fields(old["targets"].get(path), expected),
                "capsule or CPU tool changed: " + path)
    expected_checkout = request["source_checkout_required_identity"]
    source_expected = {"exists": True, "type": "directory",
                       "device": expected_checkout["device"],
                       "inode": expected_checkout["inode"],
                       "uid": expected_checkout["uid"],
                       "mode": "0o700"}
    require(same_fields(snapshot["source_checkout"], source_expected),
            "accepted source checkout identity changed")
    for name, inode, digest_value in (
        ("source_train", request["source_train_inode"], request["source_train_sha256"]),
        ("source_prepare", request["source_prepare_inode"], request["source_prepare_sha256"]),
    ):
        require(same_fields(snapshot[name], {"exists": True, "type": "regular",
                                             "device": 66305, "uid": 1000,
                                             "inode": inode, "sha256": digest_value}),
                "fresh source file changed: " + name)
    tools = request["persistent_cpu_tools"]
    for name in ("preflight", "contained"):
        require(same_fields(snapshot["tools"][name], {
                    "exists": True, "type": "regular", "device": 66305,
                    "uid": 1000, "nlink": 1, "mode": "0o600",
                    "inode": tools[name + "_inode"], "bytes": tools[name + "_bytes"],
                    "sha256": tools[name + "_sha256"]}),
                "fresh CPU helper changed: " + name)
    git = snapshot["git"]
    require(git == {"head": request["source_commit"] + "\n", "count": "1\n",
                    "shallow": "true\n", "status": "", "remote": ""},
            "source checkout is not clean one-commit shallow/no-remote")
    for expected in checkout["protected_checkpoints"]:
        matching = [item for item in old["checkpoints"]
                    if item.get("path") == expected["path"]]
        require(len(matching) == 1 and same_fields(matching[0], expected) and
                same_fields(matching[0].get("fuser"),
                            {"rc": 1, "pids": [], "stderr_empty": True}),
                "protected checkpoint changed or has a holder")
    require(same_fields(old["cache_root"], checkout["warm_cache_root"]),
            "warm cache root changed")
    for name, expected in checkout["warm_cache_children"].items():
        require(same_fields(old["caches"].get(name), expected),
                "warm cache changed: " + name)
    require(set(old["stops"]) == set(STOP_PATHS) and
            all(item.get("exists") is False for item in old["stops"].values()),
            "host STOP is present or unobserved")
    labels = old["science_and_stage_labels"]
    require(set(labels) == {checkout["science_label_unconsumed"],
                            checkout["stage_label"]} and
            all(item.get("rc") == 0 and item.get("matches") == [] and
                item.get("stderr_empty") is True for item in labels.values()),
            "science or checkout label was consumed")
    for path in checkout["forbidden_paths_must_remain_absent"]:
        require(old["targets"].get(path, {}).get("exists") is False,
                "science output or old G2 root appeared")
    require(old["processes"] == {"hits": [], "unreadable_cmdline_count": 0} and
            same_fields(old["host_lock_fuser"],
                        {"rc": 1, "pids": [], "stderr_empty": True}) and
            same_fields(old["neuron0_fuser"],
                        {"rc": 1, "pids": [], "stderr_empty": True}) and
            same_fields(old["neuron_ls"],
                        {"rc": 0, "json_valid": True, "non_diagnostic_pids": []}),
            "host process, lock or Neuron holder present")
    scratch = snapshot["scratch"]
    require(set(scratch) == {ROOT, PROCESS_DIR, STEP_LOG, CPU_RECEIPT},
            "exact CPU scratch inventory missing")
    if phase == "before":
        require(all(item.get("exists") is False for item in scratch.values()) and
                snapshot["cpu_label_matches"] == [],
                "new CPU root, step log, process directory or label already exists")
    else:
        root = scratch[ROOT]
        require(root.get("exists") is False or same_fields(root, {
                    "exists": True, "type": "directory", "device": 66305,
                    "uid": 1000, "mode": "0o700"}),
                "CPU scratch root is unsafe after attempt")
        if root.get("exists") is True:
            process = scratch[PROCESS_DIR]
            require(process.get("exists") is False or same_fields(process, {
                        "exists": True, "type": "directory", "device": 66305,
                        "uid": 1000, "mode": "0o700"}),
                    "CPU process directory is unsafe")
    return snapshot


def current_identity(checkout):
    expected = checkout["expected_host_identity"]
    boot = open("/proc/sys/kernel/random/boot_id", encoding="ascii").read().strip()
    namespace = os.readlink("/proc/self/ns/mnt")
    root_stat = os.statvfs("/")
    mounts = []
    with open("/proc/self/mountinfo", encoding="utf-8") as stream:
        for line in stream:
            left, sep, right = line.partition(" - ")
            if not sep:
                continue
            words = left.split()
            tail = right.split()
            if len(words) >= 5 and len(tail) >= 2 and words[4] == "/":
                mounts.append({"mount_id": words[0], "major_minor": words[2],
                               "mount_point": words[4], "filesystem": tail[0],
                               "source": tail[1]})
    parent = stat_path("/home/ubuntu")
    parent_expected = {key: value for key, value in
                       expected["ancestry"]["/home/ubuntu"].items()
                       if key != "nonwritable_by_group_or_world"}
    require(socket.gethostname() == expected["hostname"] and
            os.geteuid() == expected["uid"] and
            boot == expected["boot_id"] and
            namespace == expected["mount_namespace"] and
            mounts == [expected["root_mount"]] and
            same_fields(parent, parent_expected) and
            root_stat.f_bavail * root_stat.f_frsize >= FLOOR,
            "prelaunch host/boot/namespace/disk bridge failed")
    return (boot, namespace, mounts[0]["mount_id"])


def bridge_gate(request, checkout, *, scratch_created):
    """Recheck state in the launch SSH process, immediately before spawn.

    This function is also used as ``preexec_fn`` in the launch child, so a
    different boot or a STOP appearing after the outer full preflight blocks
    exec of the contained launcher. It performs no subprocess call itself.
    """
    first = current_identity(checkout)
    expected_source = request["source_checkout_required_identity"]
    require(same_fields(stat_path(SOURCE), {
                "exists": True, "type": "directory", "device": 66305,
                "inode": expected_source["inode"], "uid": 1000, "mode": "0o700"}),
            "source checkout changed at launch bridge")
    for name, path in (("train", SOURCE + "/train.py"),
                       ("prepare", SOURCE + "/prepare.py")):
        require(same_fields(stat_path(path, hash_file=True), {
                    "exists": True, "type": "regular", "device": 66305,
                    "inode": request["source_" + name + "_inode"], "uid": 1000,
                    "sha256": request["source_" + name + "_sha256"]}),
                "source file changed at launch bridge")
    tools = request["persistent_cpu_tools"]
    for name in ("preflight", "contained"):
        require(same_fields(stat_path(tools[name + "_path"], hash_file=True), {
                    "exists": True, "type": "regular", "device": 66305,
                    "inode": tools[name + "_inode"], "uid": 1000,
                    "mode": "0o600", "nlink": 1,
                    "bytes": tools[name + "_bytes"],
                    "sha256": tools[name + "_sha256"]}),
                "CPU helper changed at launch bridge")
    if scratch_created:
        require(same_fields(stat_path(ROOT), {
                    "exists": True, "type": "directory", "device": 66305,
                    "uid": 1000, "mode": "0o700"}) and
                same_fields(stat_path(PROCESS_DIR), {
                    "exists": True, "type": "directory", "device": 66305,
                    "uid": 1000, "mode": "0o700"}) and
                stat_path(STEP_LOG)["exists"] is False and
                stat_path(CPU_RECEIPT)["exists"] is False,
                "fresh CPU scratch paths changed before launch")
    else:
        require(all(stat_path(path)["exists"] is False for path in
                    (ROOT, PROCESS_DIR, STEP_LOG, CPU_RECEIPT)),
                "CPU scratch was consumed before launch")
    # STOP is checked last, then boot/namespace again; no subprocess, network
    # call, sleep, or CPU operation intervenes between this gate and Popen.
    require(all(stat_path(path)["exists"] is False for path in STOP_PATHS),
            "host STOP appeared at launch bridge")
    require(current_identity(checkout) == first,
            "host boot or namespace changed inside launch bridge")


def create_private_scratch(checkout):
    home = os.open("/home/ubuntu", os.O_RDONLY | os.O_DIRECTORY |
                   os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        identity = os.fstat(home)
        expected = checkout["expected_host_identity"]
        require((identity.st_dev, identity.st_ino, identity.st_uid) == (
                    expected["parent_device"], expected["parent_inode"], 1000),
                "CPU scratch parent changed")
        os.mkdir(os.path.basename(ROOT), 0o700, dir_fd=home)
        os.fsync(home)
        root = os.open(os.path.basename(ROOT), os.O_RDONLY | os.O_DIRECTORY |
                       os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=home)
        try:
            record = os.fstat(root)
            require(record.st_dev == 66305 and record.st_uid == 1000 and
                    stat.S_IMODE(record.st_mode) == 0o700,
                    "new CPU scratch root not private ext4")
            os.mkdir(os.path.basename(PROCESS_DIR), 0o700, dir_fd=root)
            os.fsync(root)
        finally:
            os.close(root)
    finally:
        os.close(home)


def run_exact(argv, *, timeout, preexec_fn=None):
    process = subprocess.run(argv, text=True, capture_output=True, check=False,
                             timeout=timeout, preexec_fn=preexec_fn)
    require(len(process.stdout) <= MAX_RECEIPT and
            len(process.stderr) <= MAX_RECEIPT,
            "CPU command output exceeds archived bound")
    return {"argv_sha256": argv_digest(argv), "returncode": process.returncode,
            "stdout": process.stdout, "stderr": process.stderr}


def read_cpu_receipt():
    record = stat_path(CPU_RECEIPT, hash_file=True)
    require(record.get("exists") is True and record.get("type") == "regular" and
            record.get("uid") == 1000 and record.get("device") == 66305 and
            record.get("nlink") == 1 and record.get("bytes", 0) <= MAX_RECEIPT,
            "CPU receipt is absent or unsafe")
    fd = os.open(CPU_RECEIPT, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        raw = os.read(fd, MAX_RECEIPT + 1)
    finally:
        os.close(fd)
    require(len(raw) == record["bytes"] and digest(raw) == record["sha256"],
            "CPU receipt changed after quiescence")
    parsed = json.loads(raw, object_pairs_hook=unique_json)
    require(parsed.get("schema") == "g2-preflight-v1" and
            parsed.get("verdict") == "PASS" and
            parsed.get("steps") == 3 and parsed.get("nproc") == 4 and
            parsed.get("train_py") == SOURCE + "/train.py" and
            parsed.get("train_py_sha256") ==
                "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586" and
            parsed.get("prepare", {}).get("dir") == SOURCE and
            parsed.get("prepare", {}).get("sha256") ==
                "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb" and
            parsed.get("prepare", {}).get("stub") is False and
            parsed.get("timed_out") is False and
            parsed.get("torchrun_rc") == 0,
            "CPU receipt is not a comparable three-step four-rank synthetic PASS")
    return {"stat": record, "json": parsed}


def run_cpu_once(payload):
    """Single exact CPU G2 sequence; failures preserve scratch and raw results."""
    request = payload["request"]
    checkout = payload["checkout_request"]
    commands = payload["commands"]
    records = []
    launch_attempted = False
    scratch_created = False
    try:
        pre = collect_snapshot(payload["archived_probe_source"])
        validate_snapshot(pre, request, checkout, phase="before")
        bridge_gate(request, checkout, scratch_created=False)
        create_private_scratch(checkout)
        scratch_created = True
        bridge_gate(request, checkout, scratch_created=True)
        launch_attempted = True
        # The child rechecks boot, namespace and STOP immediately before exec
        # of the exact frozen contained.py launch argv. A remote preflight in
        # a separate SSH call is insufficient by itself.
        launch = run_exact(commands["launch_argv"], timeout=180,
                           preexec_fn=lambda: bridge_gate(
                               request, checkout, scratch_created=True))
        records.append(launch)
        require(launch["returncode"] == 0 and not launch["stderr"].strip(),
                "exact contained CPU launch failed")
        wait = run_exact(commands["wait_argv"], timeout=1250)
        records.append(wait)
        require(wait["returncode"] == 0 and not wait["stderr"].strip(),
                "contained CPU wait failed or timed out")
        quiesce = run_exact(commands["quiesce_argv"], timeout=180)
        records.append(quiesce)
        require(quiesce["returncode"] == 0 and not quiesce["stderr"].strip(),
                "CPU G2 did not quiesce")
        receipt = read_cpu_receipt()
        return {"status": "CPU_G2_RAW_EVIDENCE_PENDING_INDEPENDENT_NUMERIC_REVIEW",
                "schema": "trn21-c5c-stage-r2-cpu-g2-attempt-v1", "cpu_only": True,
                "native_training_authorized": False, "evaluation_authorized": False,
                "launch_attempted": True, "scratch_created": True,
                "commands": records, "cpu_receipt": receipt,
                "preflight_utc": pre["utc"]}
    except BaseException as error:
        if launch_attempted:
            try:
                records.append(run_exact(commands["stop_on_failure_argv"], timeout=180))
                records.append(run_exact(commands["quiesce_argv"], timeout=180))
            except BaseException as stop_error:
                records.append({"stop_or_quiesce_error": type(stop_error).__name__ +
                                ": " + str(stop_error)[:300]})
        return {"status": "HOLD_PRESERVE_PARTIAL_NO_RETRY",
                "schema": "trn21-c5c-stage-r2-cpu-g2-attempt-v1", "cpu_only": True,
                "native_training_authorized": False, "evaluation_authorized": False,
                "launch_attempted": launch_attempted,
                "scratch_created": scratch_created,
                "commands": records, "error_type": type(error).__name__,
                "error": str(error)[:500]}


def check_payload(payload):
    require(isinstance(payload, dict) and set(payload) == {
                "request_text", "checkout_request_text", "archived_probe_source",
                "commands"}, "wrong exact CPU payload shape")
    request = json.loads(payload["request_text"], object_pairs_hook=unique_json)
    checkout = json.loads(payload["checkout_request_text"], object_pairs_hook=unique_json)
    commands = payload["commands"]
    require(request["schema"] == "trn21-c5c-a0-cpu-g2-final-request-v1" and
            request["cpu_g2_intent"] is True and
            request["native_training_authorized"] is False and
            request["evaluation_authorized"] is False and
            request["submission_authorized"] is False and
            request["source_checkout"] == SOURCE and
            request["cpu_root"] == ROOT and request["cpu_label"] == LABEL and
            request["g2_options"] ==
                ["--steps", "3", "--nproc", "4", "--timeout", "900", "--keep"] and
            request["synthetic_cpu_only"] is True and
            request["archived_probe_source_sha256"] ==
                digest(payload["archived_probe_source"].encode("utf-8")) and
            request["checkout_request_sha256"] ==
                digest(payload["checkout_request_text"].encode("utf-8")) and
            checkout["checkout_path"] == SOURCE and
            checkout["cpu_g2_authorized"] is False,
            "CPU-only request or source checkout payload drift")
    mapping = (("launch_argv", "exact_launch_argv_sha256"),
               ("wait_argv", "exact_wait_argv_sha256"),
               ("stop_on_failure_argv", "exact_stop_on_failure_argv_sha256"),
               ("quiesce_argv", "exact_quiesce_argv_sha256"))
    require(isinstance(commands, dict) and set(commands) == {a for a, _ in mapping} and
            all(argv_digest(commands[a]) == request[b] for a, b in mapping),
            "frozen four-rank CPU commands drift")
    launch = commands["launch_argv"]
    require(len(launch) == 64 and launch[2] == "launch" and
            "--nproc" in launch and launch[launch.index("--nproc") + 1] == "4" and
            "--steps" in launch and launch[launch.index("--steps") + 1] == "3" and
            "--real-data" not in launch and "--no-eval-public" in launch,
            "CPU G2 was not the exact synthetic four-rank three-step command")
    return {"request": request, "checkout_request": checkout,
            "archived_probe_source": payload["archived_probe_source"],
            "commands": commands}


def main():
    if (len(sys.argv) != 3 or sys.argv[1] not in ("audit", "run") or
            re.fullmatch(r"[0-9a-f]{64}", sys.argv[2]) is None):
        raise SystemExit("exact audit/run mode and payload SHA required")
    raw = sys.stdin.buffer.read(MAX_PAYLOAD + 1)
    if len(raw) > MAX_PAYLOAD or digest(raw) != sys.argv[2]:
        raise SystemExit("CPU G2 payload size or SHA drift")
    try:
        payload = check_payload(json.loads(raw, object_pairs_hook=unique_json))
        if sys.argv[1] == "audit":
            snapshot = collect_snapshot(payload["archived_probe_source"])
            result = {"status": "READ_ONLY_G2_AUDIT", "snapshot": snapshot,
                      "native_training_authorized": False,
                      "evaluation_authorized": False}
        else:
            result = run_cpu_once(payload)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
        return 0 if result["status"] != "HOLD_PRESERVE_PARTIAL_NO_RETRY" else 2
    except BaseException as error:
        print(json.dumps({"status": "HOLD_PRESERVE_PARTIAL_NO_RETRY",
                          "error_type": type(error).__name__,
                          "error": str(error)[:500]},
                         sort_keys=True, separators=(",", ":")), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
