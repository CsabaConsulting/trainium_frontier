"""Disabled one-attempt trn22 read-only final-continuity timing collector.

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
STAGE_R2 = HERE.parent / "trn22-c5c-a0-r9-incremental-stage-r2-hold/remote_incremental.py"
REMOTE = HERE / "remote_timing_readonly.py"
HOLD = HERE / "GATE_CONTRACT.HOLD.json"
ENABLED = HERE / "TIMING_REQUEST.ENABLED.json"
REVIEW = HERE / "INDEPENDENT_TIMING_REVIEW.PASS.json"
ASSIGNMENT = HERE / "SUPERVISOR_TIMING_ASSIGNMENT.json"
OUTPUT = HERE / "OPERATOR_RAW_TIMING_RESULT.json"
CONTROL_ROOT = Path("/home/poojith-devan/Desktop/Trainium")
CONTROL_STATE = CONTROL_ROOT / "research/v2/state.json"
KNOWN_HOSTS = Path("/home/poojith-devan/.ssh/known_hosts")
IDENTITY_FILE = Path("/home/poojith-devan/Desktop/Trainium/trn2.pem")
ENDPOINT = "ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com"
HOST_FINGERPRINT = "SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U"
KNOWN_HOSTS_SHA256 = "e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3"
STAGE_R2_COMMIT = "a7643f05c5468da6537c2d7b403a05ac048679f0"
STAGE_R2_SHA256 = "2b84ef670811b3f0015b30e619cb3ec5b23e914326cdd3e74523fdc5b68ee7a7"
REMOTE_SHA256 = "09b811018356dca32e52c82db38840f31626e53005f596d222ed3560c7b0f0fe"
TRIALS = 20
SCOPE = "TRN22_R9_STAGE_R2_READ_ONLY_FINAL_CONTINUITY_TIMING"
SCHEMA = "trn22-r9-final-continuity-timing-r4-v1"
MAX_REMOTE_LINE = 8192
MAX_REMOTE_STDOUT = 256 * 1024
STABLE_IDENTITY = ("hostname", "uid", "root_device", "boot_id", "mount_namespace", "source")


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
    _pinned(STAGE_R2, STAGE_R2_SHA256)
    _pinned(REMOTE, REMOTE_SHA256)
    hold_raw, hold = _exact_json(HOLD)
    own_hash = _sha(_read_regular(Path(__file__)))
    expected_hold = {
        "schema": SCHEMA + "-hold", "dispatch_authorized": False,
        "scope": SCOPE, "host": "trn22", "hostname": "ip-172-31-32-253",
        "endpoint": ENDPOINT, "fingerprint": HOST_FINGERPRINT,
        "known_hosts_sha256": KNOWN_HOSTS_SHA256,
        "stage_r2_commit": STAGE_R2_COMMIT,
        "stage_r2_remote_sha256": STAGE_R2_SHA256,
        "diagnostic_sha256": REMOTE_SHA256,
        "collector_sha256": own_hash, "trial_count": TRIALS,
        "remote_operations": ["probe", "timing", "probe"],
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
                  "stage_r2_remote_sha256": STAGE_R2_SHA256,
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
    "ns={'__name__':'trn22_r9_readonly_timing','__file__':'<pinned-readonly-timing>'}\n"
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
            probe.get("stop_paths_absent") is not True or
            probe.get("label") != "fs-20260925-trn22-c5c-a0-r9-incremental-stage-r1" or
            probe.get("bundle") !=
            "/home/ubuntu/trainium-c5c-e65-to-r9-20260925.trn22-stage-r1.bundle" or
            probe.get("target") !=
            "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r9_stage_r1"):
        raise RuntimeError("read-only probe host/label/target identity malformed")
    return probe


def _require_timing(raw, pre, post):
    parsed = raw["parsed"]
    records = parsed["records"]
    if (raw["returncode"] != 0 or raw["timed_out"] or raw["stderr_bytes"] or
            parsed["error"] is not None or len(records) != TRIALS + 2):
        raise RuntimeError("read-only timing incomplete; partial JSONL retained")
    start, end = records[0], records[-1]
    if (start.get("kind") != "start" or start.get("trial_count") != TRIALS or
            start.get("preflight_sha256") != _sha(_canonical(pre)) or
            end.get("kind") != "end" or end.get("trial_count") != TRIALS or
            end.get("completed_trials") != TRIALS or
            end.get("status") != "COMPLETE_READ_ONLY"):
        raise RuntimeError("read-only timing start/end binding malformed")
    for key in STABLE_IDENTITY:
        if (pre["identity"][key] != post["identity"][key] or
                pre["identity"][key] != start["identity"][key]):
            raise RuntimeError("before/timing/after host/source identity changed")
    if (pre["lock"] != post["lock"] or pre["process_identity"] == post["process_identity"]):
        raise RuntimeError("before/after lock or process identity drift")
    if (start.get("process_identity") in (pre["process_identity"],
                                            post["process_identity"]) or
            not isinstance(start.get("process_identity"), dict)):
        raise RuntimeError("timing did not use a distinct SSH process")
    misses = 0
    for trial in records[1:-1]:
        if trial.get("kind") != "trial" or trial.get("status") not in (
                "TIMING_MISS", "WITHIN_WINDOW_NO_OPEN"):
            raise RuntimeError("timing trial aborted or missing")
        names = [step.get("name") for step in trial.get("steps", [])]
        if names != ["activity", "label_scan_1", "label_scan_2", "direct_stop_target"]:
            raise RuntimeError("exact proposed read-only sequence incomplete")
        if any(step["started_ns"] > step["ended_ns"] or
               step["duration_ns"] != step["ended_ns"] - step["started_ns"]
               for step in trial["steps"]):
            raise RuntimeError("monotonic substage timing malformed")
        if any(left["ended_ns"] > right["started_ns"] for left, right in
               zip(trial["steps"], trial["steps"][1:])):
            raise RuntimeError("read-only final sequence time order malformed")
        if (trial["max_age_ns"] != 1_000_000_000 or
                trial["activity_age_ns"] != trial["would_open_ns"] -
                trial["steps"][0]["ended_ns"] or
                trial["label_age_ns"] != trial["would_open_ns"] -
                trial["steps"][2]["ended_ns"] or
                trial["would_open_ns"] < trial["steps"][3]["ended_ns"]):
            raise RuntimeError("final read-only age evidence malformed")
        missed = max(trial["activity_age_ns"], trial["label_age_ns"]) > 1_000_000_000
        if (trial["status"] == "TIMING_MISS") != missed:
            raise RuntimeError("timing miss classification does not match ages")
        misses += missed
    if end.get("timing_misses") != misses:
        raise RuntimeError("timing miss total does not match trials")
    return {"status": "COMPLETE_READ_ONLY_PENDING_INDEPENDENT_REVIEW",
            "timing_misses": misses,
            "max_activity_age_ns": max(t["activity_age_ns"] for t in records[1:-1]),
            "max_label_age_ns": max(t["label_age_ns"] for t in records[1:-1])}


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
