"""Disabled one-attempt trn22 r10 read-only boundary timing collector.

The checked-in HOLD contract is not authority. Only a separate exact enabled
request, independent PASS and Supervisor assignment can reach SSH. No remote
operation accepts transfer stdin or calls a source-stage writer.
"""

import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE_R10 = HERE.parent / "trn22-c5c-a0-r10-incremental-stage-r1-hold"
STAGE_R10_REMOTE = STAGE_R10 / "remote_incremental.py"
STAGE_R10_OPERATOR = STAGE_R10 / "operator_once.py"
STAGE_R10_HOLD = STAGE_R10 / "REQUEST.HOLD.json"
REMOTE = HERE / "remote_timing_readonly.py"
HOLD = HERE / "GATE_CONTRACT.HOLD.json"
ENABLED = HERE / "TIMING_REQUEST.ENABLED.json"
REVIEW = HERE / "INDEPENDENT_TIMING_REVIEW.PASS.json"
ASSIGNMENT = HERE / "SUPERVISOR_TIMING_ASSIGNMENT.json"
OUTPUT = HERE / "OPERATOR_RAW_R10_TIMING_RESULT.json"
CONTROL_ROOT = Path("/home/poojith-devan/Desktop/Trainium")
CONTROL_STATE = CONTROL_ROOT / "research/v2/state.json"
KNOWN_HOSTS = Path("/home/poojith-devan/.ssh/known_hosts")
IDENTITY_FILE = Path("/home/poojith-devan/Desktop/Trainium/trn2.pem")
ENDPOINT = "ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com"
HOST_FINGERPRINT = "SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U"
KNOWN_HOSTS_SHA256 = "e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3"
STAGE_R10_COMMIT = "020bfa7ebd048f947d5207cbedaf028a29a4b159"
STAGE_R10_REMOTE_SHA256 = "da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f"
STAGE_R10_OPERATOR_SHA256 = "7c04b0288de04a514b3da0a6a9c01f525551f2cfc92f69d4ba63f8941aa2d0b5"
STAGE_R10_HOLD_SHA256 = "625b4a53c0f67fba0a7b4ef5a6705f64f3935eb8f5841ef5e2212f6c8e820b6e"
REMOTE_SHA256 = "a9bc253680403883dacc601b35cb1935c81672c512fc9a04cff9ce12e7944ed4"
TRIALS = 20
MAX_AGE_NS = 1_000_000_000
PRE_FREE = 13_177_973_911
MIN_INODES = 10_000
SCOPE = "TRN22_R10_STAGE_R1_READ_ONLY_FINAL_BOUNDARY_TIMING"
SCHEMA = "trn22-r10-stage-r1-timing-v1"
MAX_REMOTE_LINE = 8192
MAX_REMOTE_STDOUT = 256 * 1024
STABLE_IDENTITY = ("hostname", "uid", "root_device", "boot_id", "mount_namespace", "source")
STAGE_R10_PIN = {"commit": STAGE_R10_COMMIT,
                "remote_sha256": STAGE_R10_REMOTE_SHA256,
                "operator_sha256": STAGE_R10_OPERATOR_SHA256,
                "hold_sha256": STAGE_R10_HOLD_SHA256}
STEP_SEQUENCE = ("activity_pre_scan", "label_scan_1", "label_scan_2",
                 "activity_post_scan", "space_pre_open", "direct_pre_open",
                 "label_scan_3", "activity_post_would_open",
                 "space_post_would_open", "direct_post_would_open")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_regular(path, *, private=False, limit=2 * 1024 * 1024):
    named = os.lstat(path)
    if (not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or
            named.st_uid != os.getuid() or
            (private and stat.S_IMODE(named.st_mode) & 0o077)):
        raise RuntimeError("unsafe local evidence metadata: " + path.name)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(opened, field) != getattr(named, field) for field in fields):
            raise RuntimeError("local evidence changed during open: " + path.name)
        data = bytearray()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            data.extend(block)
            if len(data) > limit:
                raise RuntimeError("local evidence unexpectedly large: " + path.name)
        after, named_after = os.fstat(fd), os.lstat(path)
        if any(getattr(other, field) != getattr(opened, field)
               for other in (after, named_after) for field in fields):
            raise RuntimeError("local evidence changed during read: " + path.name)
        return bytes(data)
    finally:
        os.close(fd)


def _pinned(path, expected):
    raw = _read_regular(path)
    if _sha(raw) != expected:
        raise RuntimeError("pinned source hash changed: " + path.name)
    return raw


def _exact_json(path):
    raw = _read_regular(path)
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != _canonical(value) + b"\n":
        raise RuntimeError("noncanonical fixed authority: " + path.name)
    return raw, value


def _local_gates():
    for root in (CONTROL_ROOT, REPO):
        for suffix in ("research/agentic/STOP", "research/v2/STOP"):
            if os.path.lexists(root / suffix):
                raise RuntimeError("local STOP present")
    raw = _read_regular(CONTROL_STATE)
    state = json.loads(raw)
    breakers = state.get("breakers") if isinstance(state, dict) else None
    if (not isinstance(breakers, dict) or
            any(breakers.get(key, {}).get("tripped") is not False
                for key in ("global_correctness", "trn22")) or
            state.get("hosts", {}).get("trn22", {}).get("launch_allowed") is not False):
        raise RuntimeError("local trn22 STOP/breaker/launch gate is unsafe")
    if _read_regular(CONTROL_STATE) != raw:
        raise RuntimeError("local controller state changed during gate")
    for root in (CONTROL_ROOT, REPO):
        for suffix in ("research/agentic/STOP", "research/v2/STOP"):
            if os.path.lexists(root / suffix):
                raise RuntimeError("local STOP appeared during gate")
    return {"controller_state_sha256": _sha(raw), "trn22_launch_allowed": False,
            "breakers_clear": True, "stops_absent": True}


def _ssh_identity():
    parent, key = os.lstat(KNOWN_HOSTS.parent), os.lstat(IDENTITY_FILE)
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or
            stat.S_IMODE(parent.st_mode) & 0o077 or
            not stat.S_ISREG(key.st_mode) or key.st_uid != os.getuid() or
            key.st_nlink != 1 or stat.S_IMODE(key.st_mode) != 0o600):
        raise RuntimeError("local SSH identity metadata is unsafe")
    if _sha(_read_regular(KNOWN_HOSTS, private=True)) != KNOWN_HOSTS_SHA256:
        raise RuntimeError("pinned known-hosts bytes changed")
    hostname = ENDPOINT.split("@", 1)[1]
    found = subprocess.run(["/usr/bin/ssh-keygen", "-F", hostname, "-f",
                            str(KNOWN_HOSTS)], capture_output=True,
                           timeout=5, check=False)
    if found.returncode != 0:
        raise RuntimeError("pinned trn22 host-key entry missing")
    entries = [line for line in found.stdout.splitlines()
               if line and not line.startswith(b"#")]
    ed = [line for line in entries if len(line.split()) >= 3 and
          line.split()[1] == b"ssh-ed25519"]
    if len(ed) != 1:
        raise RuntimeError("pinned trn22 ED25519 entry is not unique")
    fp = subprocess.run(["/usr/bin/ssh-keygen", "-l", "-E", "sha256", "-f", "-"],
                        input=ed[0] + b"\n", capture_output=True,
                        timeout=5, check=False)
    if fp.returncode != 0 or HOST_FINGERPRINT.encode("ascii") not in fp.stdout.split():
        raise RuntimeError("trn22 host fingerprint changed")
    if _sha(_read_regular(KNOWN_HOSTS, private=True)) != KNOWN_HOSTS_SHA256:
        raise RuntimeError("known-hosts changed during pin check")
    return {"algorithm": "ssh-ed25519", "fingerprint": HOST_FINGERPRINT,
            "known_hosts_sha256": KNOWN_HOSTS_SHA256}


def _authorize():
    """Fail locally before SSH or a local receipt unless all distinct gates exist."""
    _pinned(STAGE_R10_REMOTE, STAGE_R10_REMOTE_SHA256)
    _pinned(STAGE_R10_OPERATOR, STAGE_R10_OPERATOR_SHA256)
    _pinned(STAGE_R10_HOLD, STAGE_R10_HOLD_SHA256)
    _pinned(REMOTE, REMOTE_SHA256)
    hold_raw, hold = _exact_json(HOLD)
    own_hash = _sha(_read_regular(Path(__file__)))
    expected_hold = {
        "schema": SCHEMA + "-hold", "dispatch_authorized": False,
        "scope": SCOPE, "host": "trn22", "hostname": "ip-172-31-32-253",
        "endpoint": ENDPOINT, "fingerprint": HOST_FINGERPRINT,
        "known_hosts_sha256": KNOWN_HOSTS_SHA256,
        "stage_r10_commit": STAGE_R10_COMMIT,
        "stage_r10_remote_sha256": STAGE_R10_REMOTE_SHA256,
        "stage_r10_operator_sha256": STAGE_R10_OPERATOR_SHA256,
        "stage_r10_hold_sha256": STAGE_R10_HOLD_SHA256,
        "diagnostic_sha256": REMOTE_SHA256,
        "collector_sha256": own_hash, "trial_count": TRIALS,
        "remote_operations": ["probe", "timing", "probe"],
        "modeled_boundary": "THREE_ACTIVITY_THREE_LABEL_TWO_SPACE_CLOCK_ONLY_WOULD_OPEN",
        "output_basename": OUTPUT.name, "one_attempt": True,
        "remote_writes_authorized": False, "host_process_signals_authorized": False,
        "archive_target_open_authorized": False, "training_eval_aws_authorized": False}
    if hold != expected_hold:
        raise RuntimeError("disabled timing contract changed")
    request_raw, request = _exact_json(ENABLED)
    expected_request = {**expected_hold,
                        "schema": SCHEMA + "-enabled",
                        "dispatch_authorized": True,
                        "hold_contract_sha256": _sha(hold_raw)}
    if request != expected_request:
        raise RuntimeError("exact enabled read-only request missing or mismatched")
    review_raw, review = _exact_json(REVIEW)
    if review != {"schema": SCHEMA + "-review", "verdict": "PASS_READ_ONLY_TIMING_ONLY",
                  "role": "independent_verifier", "host": "trn22", "scope": SCOPE,
                  "request_sha256": _sha(request_raw),
                  "stage_r10_commit": STAGE_R10_COMMIT,
                  "stage_r10_remote_sha256": STAGE_R10_REMOTE_SHA256,
                  "stage_r10_operator_sha256": STAGE_R10_OPERATOR_SHA256,
                  "stage_r10_hold_sha256": STAGE_R10_HOLD_SHA256,
                  "diagnostic_sha256": REMOTE_SHA256,
                  "collector_sha256": own_hash}:
        raise RuntimeError("independent exact timing review missing or mismatched")
    assignment_raw, assignment = _exact_json(ASSIGNMENT)
    if assignment != {"schema": SCHEMA + "-assignment",
                      "status": "ASSIGNED_ONE_ATTEMPT_READ_ONLY_TIMING",
                      "role": "supervisor", "host": "trn22", "scope": SCOPE,
                      "request_sha256": _sha(request_raw),
                      "review_sha256": _sha(review_raw),
                      "output_basename": OUTPUT.name, "one_attempt": True}:
        raise RuntimeError("Supervisor one-attempt assignment missing or mismatched")
    local_gates = _local_gates()
    ssh_identity = _ssh_identity()
    if os.path.lexists(OUTPUT):
        raise RuntimeError("fixed raw timing result exists; no retry")
    return {"request_sha256": _sha(request_raw),
            "review_sha256": _sha(review_raw),
            "assignment_sha256": _sha(assignment_raw),
            "hold_contract_sha256": _sha(hold_raw),
            "local_gates": local_gates, "ssh_identity": ssh_identity}


_BOOTSTRAP = (
    "import base64,hashlib,json,sys\n"
    "if sys.flags.optimize: raise SystemExit('optimized Python HOLD')\n"
    "if len(sys.argv) not in (3,4): raise SystemExit('wrong read-only shape')\n"
    "raw=base64.b64decode(sys.argv[1],validate=True)\n"
    "if hashlib.sha256(raw).hexdigest()!=%r: raise SystemExit('diagnostic SHA drift')\n"
    "ns={'__name__':'trn22_r10_r1_readonly_timing','__file__':'<pinned-readonly-timing>'}\n"
    "exec(compile(raw,ns['__file__'],'exec'),ns)\n"
    "operation=sys.argv[2]\n"
    "if operation=='probe' and len(sys.argv)==3:\n"
    " result=ns['probe']()\n"
    " print(json.dumps(result,sort_keys=True,separators=(',',':')),flush=True)\n"
    "elif operation=='timing' and len(sys.argv)==4:\n"
    " pin_raw=base64.b64decode(sys.argv[3],validate=True)\n"
    " pin=json.loads(pin_raw)\n"
    " if json.dumps(pin,sort_keys=True,separators=(',',':')).encode()!=pin_raw: raise SystemExit('noncanonical pin')\n"
    " result=ns['timing'](pin)\n"
    " if result!='COMPLETE_READ_ONLY': raise SystemExit(3)\n"
    "else: raise SystemExit('only probe or timing allowed')\n"
) % (REMOTE_SHA256,)


def _ssh_argv(operation, *, pin=None):
    if operation not in ("probe", "timing") or (operation == "probe") != (pin is None):
        raise RuntimeError("wrong read-only operation shape")
    raw = _pinned(REMOTE, REMOTE_SHA256)
    words = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s",
             "120s" if operation == "timing" else "30s", "/usr/bin/env", "-i",
             "PATH=/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
             "HOME=/nonexistent", "/usr/bin/python3", "-I", "-B", "-c",
             _BOOTSTRAP, base64.b64encode(raw).decode("ascii"), operation]
    if operation == "timing":
        words.append(base64.b64encode(_canonical(pin)).decode("ascii"))
    command = " ".join(shlex.quote(word) for word in words)
    if shlex.split(command) != words or len(command.encode("utf-8")) > 120000:
        raise RuntimeError("read-only SSH command quoting/size changed")
    return ["/usr/bin/ssh", "-T", "-F", "/dev/null", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes", "-o",
            "UserKnownHostsFile=" + str(KNOWN_HOSTS), "-o",
            "GlobalKnownHostsFile=/dev/null", "-o", "HostKeyAlgorithms=ssh-ed25519",
            "-o", "UpdateHostKeys=no", "-o", "VerifyHostKeyDNS=no",
            "-o", "ControlMaster=no", "-o", "ControlPath=none",
            "-o", "ForwardAgent=no", "-o", "ConnectTimeout=10", "-o",
            "IdentitiesOnly=yes", "-i", str(IDENTITY_FILE), ENDPOINT, command]


def _parse_jsonl(stdout, operation):
    """Return the longest strict complete-line prefix, including on timeout."""
    result = {"records": [], "trailing_base64": "", "error": None}
    if len(stdout) > MAX_REMOTE_STDOUT:
        result["error"] = "remote stdout exceeded fixed byte bound"
        return result
    chunks = stdout.split(b"\n")
    complete, trailing = chunks[:-1], chunks[-1]
    if trailing:
        result["trailing_base64"] = base64.b64encode(trailing).decode("ascii")
        result["error"] = "partial trailing JSONL record"
    elif not stdout.endswith(b"\n"):
        result["error"] = "no complete JSONL line"
    for index, line in enumerate(complete):
        if not line or len(line) > MAX_REMOTE_LINE:
            result["error"] = "empty or overlong JSONL line"
            break
        try:
            value = json.loads(line.decode("utf-8"))
            if not isinstance(value, dict) or _canonical(value) != line:
                raise ValueError("noncanonical or nonobject JSONL line")
        except (UnicodeError, ValueError) as error:
            result["error"] = "malformed JSONL line %d: %s" % (index + 1, error)
            break
        if operation == "probe":
            if value.get("schema") != SCHEMA + "-probe" or index != 0:
                result["error"] = "wrong or repeated probe JSONL record"
                break
        else:
            if value.get("schema") != SCHEMA:
                result["error"] = "wrong timing JSONL schema"
                break
            expected_kind = ("start" if index == 0 else
                             "end" if value.get("kind") == "end" else "trial")
            if value.get("kind") != expected_kind:
                result["error"] = "out-of-order JSONL kind"
                break
            if value["kind"] == "trial" and value.get("index") != index:
                result["error"] = "nonconsecutive trial index"
                break
            if any(item.get("kind") == "end" for item in result["records"]):
                result["error"] = "JSONL record after end"
                break
            if len(result["records"]) >= TRIALS + 2:
                result["error"] = "too many JSONL records"
                break
        result["records"].append(value)
    return result


def _run_ssh(operation, *, pin=None, postflight=False):
    if not postflight:
        _local_gates()
    _ssh_identity()
    argv = _ssh_argv(operation, pin=pin)
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(argv, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=135 if operation == "timing" else 45,
                                   check=False)
        stdout, stderr, rc, timed_out = (completed.stdout, completed.stderr,
                                          completed.returncode, False)
    except subprocess.TimeoutExpired as error:
        stdout, stderr, rc, timed_out = (error.stdout or b"", error.stderr or b"",
                                          None, True)
    redacted = list(argv)
    redacted[-3] = "<identity-file-path-redacted>"  # -i path precedes endpoint and command.
    redacted[-1] = "<remote-command-sha256:" + _sha(argv[-1].encode()) + ">"
    raw = {"operation": operation, "returncode": rc, "timed_out": timed_out,
           "started_monotonic_ns": started, "completed_monotonic_ns": time.monotonic_ns(),
           "argv_redacted": redacted, "argv_sha256": _sha(_canonical(argv)),
           "remote_command_sha256": _sha(argv[-1].encode()), "stdin": "/dev/null",
           "stdout_bytes": len(stdout), "stdout_sha256": _sha(stdout),
           "stdout_base64": base64.b64encode(stdout).decode("ascii"),
           "stderr_bytes": len(stderr), "stderr_sha256": _sha(stderr),
           "stderr_base64": base64.b64encode(stderr).decode("ascii")}
    raw["parsed"] = _parse_jsonl(stdout, operation)
    return raw


def _require_probe(raw):
    parsed = raw["parsed"]
    if (raw["returncode"] != 0 or raw["timed_out"] or raw["stderr_bytes"] or
            parsed["error"] is not None or len(parsed["records"]) != 1):
        raise RuntimeError("read-only trn22 probe failed or was incomplete")
    probe = parsed["records"][0]
    if (probe.get("schema") != SCHEMA + "-probe" or
            probe.get("identity", {}).get("hostname") != "ip-172-31-32-253" or
            probe.get("identity", {}).get("uid") != 1000 or
            not isinstance(probe.get("process_identity", {}).get("pid"), int) or
            not isinstance(probe.get("process_identity", {}).get("start_ticks"), int) or
            probe.get("identity", {}).get("source", {}).get("commit") !=
            "e65a3097a5b00f6178188a935bb2acb9a5042c6c" or
            probe.get("stage_r10") != STAGE_R10_PIN or
            probe.get("stop_paths_absent") is not True or
            probe.get("label") != "fs-20260925-trn22-c5c-a0-r10-incremental-stage-r1" or
            probe.get("bundle") !=
            "/home/ubuntu/trainium-c5c-e65-to-r10-20260925.trn22-stage-r1.bundle" or
            probe.get("target") !=
            "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r10_stage_r1"):
        raise RuntimeError("read-only probe host/label/target identity malformed")
    return probe


def _exact_nonnegative_int(value):
    return type(value) is int and value >= 0


def _space_observation(value):
    return (isinstance(value, dict) and
            set(value) == {"root_free_bytes", "root_free_inodes"} and
            _exact_nonnegative_int(value["root_free_bytes"]) and
            _exact_nonnegative_int(value["root_free_inodes"]) and
            value["root_free_bytes"] >= PRE_FREE and
            value["root_free_inodes"] >= MIN_INODES)


def _checked_steps(trial, count):
    steps = trial.get("steps")
    if not isinstance(steps, list) or len(steps) != count:
        raise RuntimeError("exact r10 read-only step prefix incomplete")
    for index, (step, name) in enumerate(zip(steps, STEP_SEQUENCE)):
        is_space = index in (4, 8)
        keys = {"name", "started_ns", "ended_ns", "duration_ns"}
        if is_space:
            keys.add("observation")
        if (not isinstance(step, dict) or set(step) != keys or
                step["name"] != name or
                any(not _exact_nonnegative_int(step[key]) for key in
                    ("started_ns", "ended_ns", "duration_ns")) or
                step["ended_ns"] < step["started_ns"] or
                step["duration_ns"] != step["ended_ns"] - step["started_ns"]):
            raise RuntimeError("r10 monotonic step evidence malformed")
        if is_space and not _space_observation(step["observation"]):
            raise RuntimeError("r10 root byte/inode observation below floor")
    if any(left["ended_ns"] > right["started_ns"] for left, right in
           zip(steps, steps[1:])):
        raise RuntimeError("r10 read-only step order malformed")
    return steps


def _checked_ages(trial, field, now, steps, *, label, activity, space):
    if not _exact_nonnegative_int(now):
        raise RuntimeError("r10 monotonic boundary missing")
    observed = trial.get(field)
    expected = {"label_age_ns": now - steps[label]["ended_ns"],
                "activity_age_ns": now - steps[activity]["ended_ns"],
                "space_age_ns": now - steps[space]["ended_ns"]}
    if (not isinstance(observed, dict) or set(observed) != set(expected) or
            any(not _exact_nonnegative_int(value) for value in observed.values()) or
            any(value < 0 for value in expected.values()) or
            observed != expected):
        raise RuntimeError("r10 boundary age evidence malformed")
    return any(value > MAX_AGE_NS for value in expected.values())


def _require_trial(trial, index):
    lengths = {"double_scan": 3, "pre_open": 6, "would_open": 6,
               "would_payload": 10, "none": 10}
    if (not isinstance(trial, dict) or trial.get("schema") != SCHEMA or
            trial.get("kind") != "trial" or trial.get("index") != index or
            trial.get("model") != "CLOCK_ONLY_NO_OPEN" or
            trial.get("stopped_at") not in lengths or
            type(trial.get("max_age_ns")) is not int or
            trial["max_age_ns"] != MAX_AGE_NS or
            trial.get("status") != ("WITHIN_MODEL_NO_OPEN" if
                                    trial.get("stopped_at") == "none" else "TIMING_MISS")):
        raise RuntimeError("r10 timing trial malformed or aborted")
    stopped = trial["stopped_at"]
    keys = {"schema", "kind", "index", "model", "status", "stopped_at",
            "max_age_ns", "scan_pair_duration_ns", "steps"}
    if stopped != "double_scan":
        keys |= {"pre_open_check_ns", "pre_open_ages_ns"}
    if stopped in ("would_open", "would_payload", "none"):
        keys |= {"would_open_ns", "would_open_ages_ns"}
    if stopped in ("would_payload", "none"):
        keys |= {"would_payload_ns", "post_would_open_ages_ns"}
    if set(trial) != keys:
        raise RuntimeError("r10 timing trial has missing or extra boundary fields")
    steps = _checked_steps(trial, lengths[stopped])
    pair = steps[2]["ended_ns"] - steps[1]["started_ns"]
    if (not _exact_nonnegative_int(trial["scan_pair_duration_ns"]) or
            trial["scan_pair_duration_ns"] != pair):
        raise RuntimeError("r10 double-scan duration malformed")
    if stopped == "double_scan":
        if pair <= MAX_AGE_NS:
            raise RuntimeError("r10 double-scan miss classification malformed")
        return True
    if pair > MAX_AGE_NS:
        raise RuntimeError("r10 double-scan success classification malformed")
    pre = trial["pre_open_check_ns"]
    if not _exact_nonnegative_int(pre) or pre < steps[5]["ended_ns"]:
        raise RuntimeError("r10 pre-open check time malformed")
    pre_miss = _checked_ages(trial, "pre_open_ages_ns", pre, steps,
                             label=2, activity=3, space=4)
    if stopped == "pre_open":
        if not pre_miss:
            raise RuntimeError("r10 pre-open miss classification malformed")
        return True
    if pre_miss:
        raise RuntimeError("r10 stale pre-open gate classified as fresh")
    would = trial["would_open_ns"]
    if not _exact_nonnegative_int(would) or would < pre:
        raise RuntimeError("r10 would-open clock boundary malformed")
    would_miss = _checked_ages(trial, "would_open_ages_ns", would, steps,
                               label=2, activity=3, space=4)
    if stopped == "would_open":
        if not would_miss:
            raise RuntimeError("r10 would-open miss classification malformed")
        return True
    if would_miss or steps[6]["started_ns"] < would:
        raise RuntimeError("r10 stale or reordered would-open gate")
    payload = trial["would_payload_ns"]
    if not _exact_nonnegative_int(payload) or payload < steps[9]["ended_ns"]:
        raise RuntimeError("r10 would-payload clock boundary malformed")
    post_miss = _checked_ages(trial, "post_would_open_ages_ns", payload, steps,
                              label=6, activity=7, space=8)
    if (stopped == "would_payload") != post_miss:
        raise RuntimeError("r10 post-would-open miss classification malformed")
    return post_miss


def _require_timing(raw, pre, post):
    parsed = raw["parsed"]
    records = parsed["records"]
    if (raw["returncode"] != 0 or raw["timed_out"] or raw["stderr_bytes"] or
            parsed["error"] is not None or len(records) != TRIALS + 2):
        raise RuntimeError("read-only r10 timing incomplete; partial JSONL retained")
    start, end = records[0], records[-1]
    if (start.get("kind") != "start" or start.get("trial_count") != TRIALS or
            start.get("preflight_sha256") != _sha(_canonical(pre)) or
            start.get("stage_r10") != STAGE_R10_PIN or
            end.get("kind") != "end" or end.get("trial_count") != TRIALS or
            end.get("completed_trials") != TRIALS or
            end.get("status") != "COMPLETE_READ_ONLY"):
        raise RuntimeError("read-only r10 timing start/end binding malformed")
    if (pre.get("stage_r10") != STAGE_R10_PIN or
            post.get("stage_r10") != STAGE_R10_PIN or
            pre["lock"] != post["lock"] or pre["lock"] != start.get("lock")):
        raise RuntimeError("before/timing/after r10 source or lock pin drift")
    for key in STABLE_IDENTITY:
        if (pre["identity"][key] != post["identity"][key] or
                pre["identity"][key] != start["identity"][key]):
            raise RuntimeError("before/timing/after host/source identity changed")
    if pre["process_identity"] == post["process_identity"]:
        raise RuntimeError("before/after timing process identity drift")
    if (start.get("process_identity") in (pre["process_identity"],
                                            post["process_identity"]) or
            not isinstance(start.get("process_identity"), dict)):
        raise RuntimeError("timing did not use a distinct SSH process")
    trials = records[1:-1]
    misses = sum(_require_trial(trial, index) for index, trial in
                 enumerate(trials, 1))
    if type(end.get("timing_misses")) is not int or end["timing_misses"] != misses:
        raise RuntimeError("r10 timing miss total does not match trials")
    return {"status": "COMPLETE_READ_ONLY_PENDING_INDEPENDENT_REVIEW",
            "timing_misses": misses,
            "within_model_trials": TRIALS - misses,
            "max_double_scan_ns": max(t["scan_pair_duration_ns"] for t in trials),
            "max_pre_open_label_age_ns": max(
                (t["pre_open_ages_ns"]["label_age_ns"] for t in trials
                 if "pre_open_ages_ns" in t), default=None),
            "max_would_open_label_age_ns": max(
                (t["would_open_ages_ns"]["label_age_ns"] for t in trials
                 if "would_open_ages_ns" in t), default=None),
            "max_post_would_open_age_ns": max(
                (max(t["post_would_open_ages_ns"].values()) for t in trials
                 if "post_would_open_ages_ns" in t), default=None)}


def collect_once(authority):
    """Consume a single fixed local raw receipt; no retries or host mutation."""
    fd = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    record = {"schema": SCHEMA + "-raw-local", "status": "FAILED_CLOSED",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "authority": authority, "phases": [], "comparison": None, "errors": []}
    before = after = raw_timing = None
    timing_started = False
    try:
        if _local_gates() != authority["local_gates"]:
            raise RuntimeError("local gates changed before read-only preflight")
        raw_pre = _run_ssh("probe")
        record["phases"].append(raw_pre)
        before = _require_probe(raw_pre)
        if _local_gates() != authority["local_gates"]:
            raise RuntimeError("local gates changed before read-only timing")
        timing_started = True
        raw_timing = _run_ssh("timing", pin=before)
        record["phases"].append(raw_timing)
    except BaseException as error:
        record["errors"].append({"type": type(error).__name__, "message": str(error)[:512]})
    finally:
        if timing_started:
            try:
                raw_post = _run_ssh("probe", postflight=True)
                record["phases"].append(raw_post)
                after = _require_probe(raw_post)
            except BaseException as error:
                record["errors"].append({"type": type(error).__name__,
                                         "message": str(error)[:512]})
        if before is not None and after is not None and raw_timing is not None:
            try:
                record["comparison"] = _require_timing(raw_timing, before, after)
                if not record["errors"]:
                    record["status"] = "RAW_READ_ONLY_TIMING_RETRIEVED_PENDING_REVIEW"
            except BaseException as error:
                record["errors"].append({"type": type(error).__name__,
                                         "message": str(error)[:512]})
        record["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        payload = _canonical(record) + b"\n"
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise RuntimeError("local raw receipt write failed")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
    return record["status"]


def main():
    authority = _authorize()  # HOLD package has no enabled request/review/assignment.
    status = collect_once(authority)
    print(status + ": " + str(OUTPUT))
    if status != "RAW_READ_ONLY_TIMING_RETRIEVED_PENDING_REVIEW":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
