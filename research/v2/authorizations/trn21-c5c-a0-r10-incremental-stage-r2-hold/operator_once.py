"""One-attempt trn21 r10 incremental source stage, currently DISABLED/HOLD.

No enabled request, independent final PASS, or Supervisor assignment is in
this package. ``main`` therefore exits before a marker, journal, or SSH.
Only a later exact separately reviewed request can enter ``stage_once``.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REMOTE = HERE / "remote_incremental.py"
BUNDLE = HERE / "e65-to-r10.bundle"
DISABLED_REQUEST = HERE / "REQUEST.HOLD.json"
ENABLED_REQUEST = HERE / "REQUEST.ENABLED_PENDING_FINAL_REVIEW.json"
FINAL_REVIEW = HERE / "INDEPENDENT_FINAL_SOURCE_STAGE_REVIEW.PASS.json"
ASSIGNMENT = HERE / "SUPERVISOR_ONE_ATTEMPT_SOURCE_STAGE_ASSIGNMENT.json"
MARKER = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/"
              "trn21-c5c-a0-r10-incremental-stage-r2.one-attempt.jsonl")
CONTROL_ROOT = Path("/home/poojith-devan/Desktop/Trainium")
LEGACY_CONTROL_ROOT = Path("/home/poojith-devan/Documents/ChatGPT/Trainium")
PACKAGE_ROOT = HERE.parents[3]
KNOWN_HOSTS = Path("/home/poojith-devan/.ssh/known_hosts")
IDENTITY_FILE = Path("/home/poojith-devan/Desktop/Trainium/trn2.pem")
ENDPOINT = "ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com"
HOST_FINGERPRINT = "SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc"
KNOWN_HOSTS_SHA256 = "e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3"
BUNDLE_SHA256 = "b191ba0b5272d062c61ed9e35cfbb7efcb67e1df49cbe3959be28c906073d20e"
BUNDLE_BYTES = 109800
BASE_COMMIT = "e65a3097a5b00f6178188a935bb2acb9a5042c6c"
TARGET_COMMIT = "22bfbab5a77d6feb670d2d737504f8b70b08ddc1"
R10_SOURCE_REVIEW_COMMIT = "ee9a4b14cc30ab372162fd434692e59b13ac44ab"
R10_SOURCE_REVIEW_SHA256 = "5e6fbfe6fd46608d0b220b99402185c78c809523026df7153882d3637de20c96"
SOURCE = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2"
TARGET = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10"
REMOTE_BUNDLE = "/home/ubuntu/trainium-c5c-e65-to-r10-20260925.trn21-stage-r2.bundle"
LABEL = "fs-20260925-trn21-c5c-a0-r10-incremental-stage-r2"
SCOPE = "TRN21_R10_INCREMENTAL_SOURCE_STAGE_ONLY_ONE_ATTEMPT"
PRE_FREE = 13_177_973_911
MIN_INODES = 10_000
CHRONOLOGY_TOLERANCE_SECONDS = 1e-6


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _regular(path, *, private=False):
    named = os.lstat(path)
    if (not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or
            named.st_uid != os.getuid() or
            (private and stat.S_IMODE(named.st_mode) & 0o077)):
        raise RuntimeError("unsafe local evidence file metadata: " + str(path))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
                named.st_dev, named.st_ino, named.st_size):
            raise RuntimeError("local evidence inode drift: " + str(path))
        data = bytearray()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            data.extend(block)
            if len(data) > 2 * 1024 * 1024:
                raise RuntimeError("local evidence file unexpectedly large")
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise RuntimeError("local evidence changed while reading")
        return bytes(data)
    finally:
        os.close(fd)


def _json(path):
    raw = _regular(path)
    value = json.loads(raw)
    if not isinstance(value, dict) or raw.strip() != _canonical(value):
        raise RuntimeError("authority file is not canonical JSON: " + str(path))
    return value, _sha(raw)


def _source_bytes():
    bundle = _regular(BUNDLE)
    if len(bundle) != BUNDLE_BYTES or _sha(bundle) != BUNDLE_SHA256:
        raise RuntimeError("exact incremental Git bundle changed")
    remote = _regular(REMOTE)
    return bundle, remote


def _local_stop_and_breakers():
    # Only the active Desktop controller decides breaker/launch state. The
    # historical Documents checkout can still STOP the attempt, never enable it.
    for root in (CONTROL_ROOT, LEGACY_CONTROL_ROOT, PACKAGE_ROOT):
        for suffix in ("research/agentic/STOP", "research/v2/STOP"):
            if os.path.lexists(root / suffix):
                raise RuntimeError("local STOP present")
    state = json.loads(_regular(CONTROL_ROOT / "research/v2/state.json"))
    if not isinstance(state, dict):
        raise RuntimeError("controller state is malformed")
    breakers = state.get("breakers")
    if (not isinstance(breakers, dict) or
            any(breakers.get(name, {}).get("tripped") is not False
                for name in ("global_correctness", "trn21")) or
            state.get("launch_authorized") is not False or
            state.get("hosts", {}).get("trn21", {}).get("launch_allowed") is not False):
        raise RuntimeError("controller STOP/breaker/launch state is not a safe source-stage HOLD")


def _authority(request, request_hash, review, review_hash, assignment,
               remote_hash, operator_hash):
    expected_request = {
        "schema": "trn21-r10-incremental-stage-enabled-request-v2",
        "dispatch_authorized": True,
        "source_stage_authorized": True,
        "scope": SCOPE,
        "host": "trn21",
        "hostname": "ip-172-31-34-85",
        "source_commit": BASE_COMMIT,
        "target_commit": TARGET_COMMIT,
        "independent_r10_source_review_commit": R10_SOURCE_REVIEW_COMMIT,
        "independent_r10_source_review_sha256": R10_SOURCE_REVIEW_SHA256,
        "source": SOURCE,
        "target": TARGET,
        "remote_bundle": REMOTE_BUNDLE,
        "bundle_sha256": BUNDLE_SHA256,
        "bundle_bytes": BUNDLE_BYTES,
        "remote_source_sha256": remote_hash,
        "operator_source_sha256": operator_hash,
        "attempt_label": LABEL,
        "marker": str(MARKER),
        "one_attempt": True,
        "native_training_authorized": False,
        "evaluation_authorized": False,
    }
    if request != expected_request:
        raise RuntimeError("no exact enabled source-stage request")
    if review != {
            "schema": "trn21-r10-incremental-stage-independent-final-review-v2",
            "verdict": "PASS_SOURCE_STAGE_ONLY",
            "request_sha256": request_hash,
            "remote_source_sha256": remote_hash,
            "operator_source_sha256": operator_hash,
            "bundle_sha256": BUNDLE_SHA256,
            "independent_r10_source_review_sha256": R10_SOURCE_REVIEW_SHA256,
            "scope": SCOPE,
            "host": "trn21",
            "attempt_label": LABEL,
            "native_training_authorized": False}:
        raise RuntimeError("actual independent final source-stage PASS is absent or mismatched")
    if assignment != {
            "schema": "trn21-r10-incremental-stage-supervisor-assignment-v2",
            "assignment_id": "trn21-r10-incremental-stage-r2-one-attempt",
            "request_sha256": request_hash,
            "review_sha256": review_hash,
            "scope": SCOPE,
            "host": "trn21",
            "attempt_label": LABEL,
            "marker": str(MARKER),
            "one_attempt": True,
            "native_training_authorized": False}:
        raise RuntimeError("actual Supervisor one-attempt assignment absent or mismatched")


def _ssh_identity():
    parent = os.lstat(KNOWN_HOSTS.parent)
    key = os.lstat(IDENTITY_FILE)  # Never read or print the PEM bytes.
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or
            stat.S_IMODE(parent.st_mode) & 0o077 or
            not stat.S_ISREG(key.st_mode) or key.st_uid != os.getuid() or
            key.st_nlink != 1 or stat.S_IMODE(key.st_mode) != 0o600):
        raise RuntimeError("SSH identity metadata is unsafe")
    if _sha(_regular(KNOWN_HOSTS, private=True)) != KNOWN_HOSTS_SHA256:
        raise RuntimeError("known-hosts bytes changed")
    hostname = ENDPOINT.split("@", 1)[1]
    found = subprocess.run(["/usr/bin/ssh-keygen", "-F", hostname, "-f",
                            str(KNOWN_HOSTS)], capture_output=True,
                           timeout=20, check=False)
    if found.returncode != 0:
        raise RuntimeError("pinned SSH host entry not found")
    entries = [line for line in found.stdout.splitlines()
               if line and not line.startswith(b"#")]
    ed = [line for line in entries if len(line.split()) >= 3 and
          line.split()[1] == b"ssh-ed25519"]
    if len(ed) != 1:
        raise RuntimeError("pinned ED25519 host entry is not unique")
    fp = subprocess.run(["/usr/bin/ssh-keygen", "-l", "-E", "sha256", "-f", "-"],
                        input=ed[0] + b"\n", capture_output=True, timeout=20,
                        check=False)
    if fp.returncode != 0 or HOST_FINGERPRINT.encode("ascii") not in fp.stdout.split():
        raise RuntimeError("trn21 host fingerprint changed")
    if _sha(_regular(KNOWN_HOSTS, private=True)) != KNOWN_HOSTS_SHA256:
        raise RuntimeError("known-hosts changed during identity check")


def _ssh_argv(remote_command):
    """Use the same pinned transport for the read-only probe and each phase."""
    return ["/usr/bin/ssh", "-T", "-F", "/dev/null", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes", "-o",
            "UserKnownHostsFile=" + str(KNOWN_HOSTS), "-o",
            "GlobalKnownHostsFile=/dev/null", "-o", "HostKeyAlgorithms=ssh-ed25519",
            "-o", "UpdateHostKeys=no", "-o", "IdentitiesOnly=yes", "-o",
            "ConnectTimeout=10", "-i", str(IDENTITY_FILE), ENDPOINT,
            remote_command]


def _remote_timeout_identity():
    """Check the remote absolute timeout that actually wraps the phase."""
    version = subprocess.run(_ssh_argv("/usr/bin/timeout --version"),
                             input=b"", capture_output=True, timeout=20,
                             check=False)
    if (version.returncode != 0 or version.stderr or len(version.stdout) > 1024 or
            not version.stdout.startswith(b"timeout (GNU coreutils) ")):
        raise RuntimeError("remote absolute GNU timeout identity check failed")


def _ssh_phase(phase, prior, data, remote_source):
    _local_stop_and_breakers()
    _ssh_identity()
    # A fresh isolated Python process executes exact, hash-checked source.
    program = ("import json,sys\n"
               "if sys.flags.optimize: raise SystemExit('optimized Python HOLD')\n"
               "scope={'__name__':'trn21_incremental_remote'}\n"
               "exec(" + repr(remote_source.decode("utf-8")) + ",scope)\n"
               "result=scope['perform'](sys.argv[1],json.loads(sys.argv[2]),"
               "sys.stdin.buffer.read())\n"
               "print(json.dumps(result,sort_keys=True,separators=(',',':')))\n")
    command = ["/usr/bin/timeout", "--foreground", "--signal=TERM",
               "--kill-after=15s", "320s", "/usr/bin/env", "-i",
               "PATH=/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
               "/usr/bin/python3", "-I", "-B", "-c", program, phase,
               json.dumps(prior, sort_keys=True, separators=(",", ":"))]
    remote_command = " ".join(shlex.quote(item) for item in command)
    _remote_timeout_identity()
    # STOP may have appeared during the SSH probe; it cannot be overridden by
    # the earlier clean controller snapshot, and must block this phase.
    _local_stop_and_breakers()
    result = subprocess.run(_ssh_argv(remote_command), input=data, capture_output=True,
                            timeout=350, check=False)
    if result.returncode != 0 or result.stderr or len(result.stdout) > 2 * 1024 * 1024:
        raise RuntimeError("SSH phase failed or emitted stderr: " + phase)
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or value.get("phase") != phase:
        raise RuntimeError("remote phase receipt malformed: " + phase)
    return value


def _append(fd, event):
    data = _canonical(event) + b"\n"
    count = 0
    while count < len(data):
        count += os.write(fd, data[count:])
    os.fsync(fd)


def _marker(request_hash, review_hash, assignment_hash):
    parent = os.lstat(MARKER.parent)
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or
            stat.S_IMODE(parent.st_mode) != 0o700 or os.path.lexists(MARKER)):
        raise RuntimeError("one-attempt marker parent or absence gate failed")
    fd = os.open(MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    _append(fd, {"event": "ONE_ATTEMPT_CONSUMED", "label": LABEL,
                 "request_sha256": request_hash, "review_sha256": review_hash,
                 "assignment_sha256": assignment_hash,
                 "scope": SCOPE})
    os.fsync(fd)
    return fd


def _checked_final_gate(final):
    """Reject a write receipt that omits an actual-open or post-open gate."""
    if (not isinstance(final, dict) or
            final.get("pre_open_broad_label_scan") is not True or
            final.get("post_label_activity_check") is not True or
            final.get("post_open_broad_label_scan") is not True or
            final.get("post_open_label_and_activity_check") is not True):
        raise RuntimeError("missing pre/post-open continuity proof")
    for name in ("activity_age_seconds", "label_age_seconds", "space_age_seconds",
                 "label_scan_duration_seconds", "open_call_elapsed_seconds",
                 "post_open_label_age_seconds", "post_open_activity_age_seconds",
                 "post_open_space_age_seconds"):
        value = final.get(name)
        if (type(value) not in (int, float) or not math.isfinite(value) or
                not 0 <= value <= 1.0):
            raise RuntimeError("invalid final continuity age: " + name)
    for older, newer in (("label_age_seconds", "activity_age_seconds"),
                         ("activity_age_seconds", "space_age_seconds"),
                         ("space_age_seconds", "open_call_elapsed_seconds"),
                         ("post_open_label_age_seconds", "post_open_activity_age_seconds"),
                         ("post_open_activity_age_seconds", "post_open_space_age_seconds")):
        if final[older] + CHRONOLOGY_TOLERANCE_SECONDS < final[newer]:
            raise RuntimeError("contradictory final monotonic chronology")
    for name in ("root_space_before_open", "root_space_after_open"):
        space = final.get(name)
        if (not isinstance(space, dict) or
                type(space.get("root_free_bytes")) is not int or
                type(space.get("root_free_inodes")) is not int or
                space["root_free_bytes"] < PRE_FREE or
                space["root_free_inodes"] < MIN_INODES):
            raise RuntimeError("invalid final root byte/inode proof: " + name)


def stage_once(remote_call=_ssh_phase):
    """Enter only after a distinct enabled request/review/assignment exists.

    The checked-in HOLD request cannot pass this function. All SSH phases are
    separate processes. A full postflight is ATTEMPTED after each entered
    marker, but local STOP, remote STOP or transport failure can prevent its
    completion. Such a case is HOLD with the marker and partials retained.
    """
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE") is not None:
        raise RuntimeError("optimized Python or inherited optimization is forbidden")
    # Current disabled package contains no enabled file; this is first gate.
    request, request_hash = _json(ENABLED_REQUEST)
    if request.get("dispatch_authorized") is not True:
        raise RuntimeError("source staging not enabled")
    bundle, remote_source = _source_bytes()
    remote_hash = _sha(remote_source)
    operator_hash = _sha(_regular(Path(__file__)))
    review, review_hash = _json(FINAL_REVIEW)
    assignment, assignment_hash = _json(ASSIGNMENT)
    _authority(request, request_hash, review, review_hash, assignment,
               remote_hash, operator_hash)
    _local_stop_and_breakers()
    _ssh_identity()
    _remote_timeout_identity()
    # A STOP arriving during the remote identity probe must not consume the
    # one-attempt marker or enter any staging phase.
    _local_stop_and_breakers()
    marker_fd = _marker(request_hash, review_hash, assignment_hash)
    preflight = write = readback = postflight = None
    failure = None
    try:
        try:
            preflight = remote_call("preflight", None, b"", remote_source)
            _append(marker_fd, {"event": "FRESH_PREFLIGHT", "receipt": preflight})
            if (preflight.get("snapshot", {}).get("bundle_exists") or
                    preflight.get("snapshot", {}).get("target_exists")):
                raise RuntimeError("preflight found a prior target")
            write = remote_call("write", preflight["snapshot"], bundle, remote_source)
            _append(marker_fd, {"event": "STAGE_WRITE", "receipt": write})
            _checked_final_gate(write.get("final_no_write_continuity"))
            readback = remote_call("readback", write, b"", remote_source)
            _append(marker_fd, {"event": "DISTINCT_SESSION_READBACK", "receipt": readback})
            if (readback.get("bundle", {}).get("sha256") != BUNDLE_SHA256 or
                    readback.get("target", {}).get("commit") != TARGET_COMMIT or
                    readback.get("target", {}).get("clean") is not True or
                    readback.get("snapshot", {}).get("process_identity") ==
                    write.get("snapshot", {}).get("process_identity")):
                raise RuntimeError("cross-session source-stage readback mismatch")
        except Exception as error:
            failure = type(error).__name__ + ": " + str(error)
            _append(marker_fd, {"event": "STAGE_HOLD", "reason": failure})
        finally:
            try:
                postflight = remote_call("postflight", None, b"", remote_source)
                _append(marker_fd, {"event": "FULL_POSTFLIGHT_COMPLETED",
                                    "receipt": postflight})
            except Exception as error:
                _append(marker_fd, {"event": "POSTFLIGHT_HOLD",
                                    "full_remote_postflight_completed": False,
                                    "reason": type(error).__name__ + ": " + str(error)})
                failure = failure or "full remote postflight did not complete"
        if failure:
            raise RuntimeError(failure + "; marker and all partials retained")
        if (postflight is None or postflight.get("phase") != "postflight" or
                postflight.get("scope") != "COMPLETED_READONLY_POSTFLIGHT" or
                not postflight.get("bundle_exists") or
                not postflight.get("target_exists") or
                postflight.get("snapshot", {}).get("source") !=
                preflight.get("snapshot", {}).get("source") or
                postflight.get("snapshot", {}).get("protected") !=
                preflight.get("snapshot", {}).get("protected")):
            raise RuntimeError("terminal full postflight changed; all partials retained")
        return {"status": "STAGE_EVIDENCE_READY_FOR_INDEPENDENT_RESULT_REVIEW_ONLY",
                "marker": str(MARKER), "label": LABEL,
                "source_commit": BASE_COMMIT, "target_commit": TARGET_COMMIT,
                "bundle_sha256": BUNDLE_SHA256,
                "native_training_authorized": False}
    finally:
        os.close(marker_fd)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("HOLD: no ad-hoc arguments or alternate request paths")
    if not ENABLED_REQUEST.exists():
        raise SystemExit("HOLD: no enabled source-stage request, final review or assignment")
    try:
        print(json.dumps(stage_once(), sort_keys=True))
    except Exception as error:
        raise SystemExit("HOLD: " + type(error).__name__ + ": " + str(error))
