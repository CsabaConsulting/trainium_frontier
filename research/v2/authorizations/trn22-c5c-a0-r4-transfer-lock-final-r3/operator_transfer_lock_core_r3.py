"""Local trn22 transfer-only evidence, breaker, journal and SSH helpers.

There is no CLI dispatch here. The sole final wrapper must authenticate its
review and Supervisor assignment before it calls these helpers. No checkout,
native run, evaluation, overwrite, cleanup or retry exists in this module.
"""

import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

import stage_r4_transport_lock_final_r3 as transport


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REQUEST = HERE / "TRANSFER_REQUEST.HOLD.json"
INDEPENDENT_REVIEW = HERE / "INDEPENDENT_ENABLEMENT_REVIEW.json"
ASSIGNMENT = HERE / "SUPERVISOR_TRANSFER_ASSIGNMENT.json"
SOURCE_CONTRACT = REPO / "research/v2/authorizations/trn22-c5c-a0-r4-source-stage/GATE_CONTRACT.json"
SOURCE_REVIEW = REPO / "research/v2/authorizations/trn22-c5c-a0-r4-source-stage/INDEPENDENT_STATIC_VERIFIER_RECEIPT.md"
CONTROL_STATE = Path("/home/poojith-devan/Desktop/Trainium/research/v2/state.json")
JOURNAL = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r4-fifo-final-r3-one-attempt.jsonl")
SOURCE_CONTRACT_SHA256 = "79878b16f459d4c12b32cb0ebb85d205dad86d584ae09654e0e8161920deab69"
SOURCE_REVIEW_SHA256 = "d49e8c03567210283074c44a2f2937594ada39f43e399f97d176c373c1a32c1e"
FROZEN_HOLD_REQUEST_SHA256 = "2b7249002ec301962153efdfc3b003454dff9c2ba6fe80a09b780e10adb8ab0c"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _read_evidence(path):
    named = path.lstat()
    if (not stat.S_ISREG(named.st_mode) or named.st_uid != os.getuid() or
            named.st_nlink != 1 or stat.S_IMODE(named.st_mode) & 0o002 or
            named.st_size > 1024 * 1024):
        raise RuntimeError("unsafe fixed-path review or assignment evidence")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
                named.st_dev, named.st_ino, named.st_size):
            raise RuntimeError("fixed-path evidence changed on open")
        chunks = []
        count = 0
        while True:
            part = os.read(fd, 65536)
            if not part:
                break
            count += len(part)
            if count > 1024 * 1024:
                raise RuntimeError("oversized fixed-path evidence")
            chunks.append(part)
        linked = path.lstat()
        opened_after = os.fstat(fd)
        if ((opened_after.st_dev, opened_after.st_ino, opened_after.st_size) != (
                named.st_dev, named.st_ino, named.st_size) or
                (linked.st_dev, linked.st_ino, linked.st_size) != (
                named.st_dev, named.st_ino, named.st_size)):
            raise RuntimeError("fixed-path evidence changed during read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _pinned(path, digest):
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("missing exact review or assignment SHA-256")
    data = _read_evidence(path)  # Actual fixed-path content, not a string claim.
    if _sha(data) != digest:
        raise RuntimeError("reviewed fixed-path evidence drift: " + str(path))
    return data


def _validate_authority_bundle(request_bytes, review_bytes, assignment_bytes):
    """One-way hash chain: request <- independent PASS <- Supervisor assignment.

    The request never supplies its own PASS or assignment hash. Both later
    files are separately issued, read from fixed paths, and bind actual bytes.
    """
    request = json.loads(request_bytes)
    review = json.loads(review_bytes)
    assignment = json.loads(assignment_bytes)
    request_sha = _sha(request_bytes)
    operator_sha = _sha(Path(__file__).read_bytes())
    transport_sha = _sha((HERE / "stage_r4_transport.py").read_bytes())
    if any(key in request for key in ("independent_review_status",
                                      "independent_review_sha256", "assignment_sha256")):
        raise RuntimeError("request attempted to self-attest review or assignment")
    if (request.get("source_commit") != "446b657695f086f9b484fa601c9a3b61c2495912" or
            request.get("schema") != "trn22-c5c-a0-r4-transfer-only-request-v1" or
            request.get("dispatch_authorized") is not True or
            request.get("host") != "trn22" or
            request.get("capsule_source_outside_git") != str(transport.SOURCE) or
            request.get("capsule_sha256") != transport.SOURCE_SHA256 or
            request.get("capsule_bytes") != transport.SOURCE_BYTES or
            request.get("target") != transport.TARGET or
            request.get("journal") != str(JOURNAL) or
            request.get("source_stage_contract_sha256") != SOURCE_CONTRACT_SHA256 or
            request.get("source_stage_independent_static_review_sha256") != SOURCE_REVIEW_SHA256 or
            request.get("remote_sha256") != transport.REMOTE_SHA256 or
            request.get("transport_sha256") != transport_sha or
            request.get("assignment_scope") != "TRN22_R4_CAPSULE_TRANSFER_ONLY_ONE_ATTEMPT" or
            request.get("ssh_identity") != {
                "known_hosts_sha256": transport.KNOWN_HOSTS_SHA256,
                "host_fingerprint": transport.HOST_FINGERPRINT,
                "host_key_algorithm": "ssh-ed25519",
                "config_file": "/dev/null",
            } or
            request.get("one_attempt") is not True or
            request.get("fresh_probe_per_attempt_required") is not True or
            request.get("preflight_to_write_identity_binding_required") is not True or
            request.get("live_remote_stop_process_safety_before_o_excl_required") is not True or
            request.get("raw_ssh_argv_exit_stdout_stderr_journal_required") is not True or
            request.get("final_read_only_postflight_even_on_failure_required") is not True or
            any(request.get(key) is not False for key in (
                "same_path_overwrite_delete_rename_or_retry_authorized",
                "source_checkout_authorized", "cpu_g2_authorized",
                "native_run_authorized", "eval_authorized",
                "monitor_operation_authorized", "submission_authorized"))):
        raise RuntimeError("transfer request/code/binding drift")
    if (review.get("schema") != "trn22-r4-independent-transfer-enable-review-v1" or
            review.get("verdict") != "PASS" or
            review.get("reviewer_role") != "independent_verifier" or
            review.get("request_sha256") != request_sha or
            review.get("operator_sha256") != operator_sha or
            review.get("transport_sha256") != transport_sha or
            review.get("remote_sha256") != transport.REMOTE_SHA256 or
            review.get("capsule_sha256") != transport.SOURCE_SHA256 or
            review.get("source_commit") != request["source_commit"] or
            review.get("host") != "trn22" or
            review.get("scope") != request["assignment_scope"] or
            review.get("no_host_action_by_reviewer") is not True):
        raise RuntimeError("actual independent review is not an exact structured PASS")
    if (assignment.get("schema") != "trn22-r4-supervisor-transfer-only-one-attempt-v1" or
            assignment.get("issuer_role") != "supervisor" or
            assignment.get("dispatch_authorized") is not True or
            assignment.get("scope") != request["assignment_scope"] or
            assignment.get("host") != "trn22" or
            assignment.get("request_sha256") != request_sha or
            assignment.get("independent_review_sha256") != _sha(review_bytes) or
            assignment.get("operator_sha256") != operator_sha or
            assignment.get("transport_sha256") != transport_sha or
            assignment.get("remote_sha256") != transport.REMOTE_SHA256 or
            assignment.get("source_commit") != request["source_commit"] or
            assignment.get("capsule_sha256") != transport.SOURCE_SHA256 or
            assignment.get("target") != transport.TARGET or
            assignment.get("journal") != str(JOURNAL) or
            assignment.get("one_attempt") is not True):
        raise RuntimeError("Supervisor one-attempt assignment mismatch")
    return request, assignment


def authorize():
    # This frozen disabled request cannot be enabled by changing a JSON flag.
    request_bytes = _pinned(REQUEST, FROZEN_HOLD_REQUEST_SHA256)
    request = json.loads(request_bytes)
    if request.get("dispatch_authorized") is not True:
        raise RuntimeError("DISPATCH HOLD: frozen transfer request is disabled")
    _pinned(SOURCE_CONTRACT, SOURCE_CONTRACT_SHA256)
    _pinned(SOURCE_REVIEW, SOURCE_REVIEW_SHA256)
    review_bytes = _read_evidence(INDEPENDENT_REVIEW)
    assignment_bytes = _read_evidence(ASSIGNMENT)
    return _validate_authority_bundle(request_bytes, review_bytes, assignment_bytes)


def _check_controller_breakers():
    stop_paths = {CONTROL_STATE.parent / "STOP",
                  CONTROL_STATE.parent.parent / "agentic/STOP",
                  REPO / "research/agentic/STOP",
                  REPO / "research/v2/STOP"}
    for path in sorted(stop_paths):
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise RuntimeError("local STOP is present: " + str(path))
    first = _read_evidence(CONTROL_STATE)
    state = json.loads(first)
    breakers = state["breakers"]
    if any(breakers[key].get("tripped") is not False
           for key in ("global_correctness", "trn22")):
        raise RuntimeError("local controller global or trn22 breaker is tripped/unknown")
    if _read_evidence(CONTROL_STATE) != first:
        raise RuntimeError("local controller state raced during breaker read")
    for path in sorted(stop_paths):
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise RuntimeError("local STOP appeared during breaker check: " + str(path))
    return {"path": str(CONTROL_STATE), "sha256": _sha(first),
            "global_tripped": False, "trn22_tripped": False,
            "local_stops_clear": True}


class Journal:
    """Fixed-path exclusive private append-only evidence, consumed before SSH."""

    def __init__(self):
        parent = JOURNAL.parent.lstat()
        if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or
                stat.S_IMODE(parent.st_mode) & 0o077):
            raise RuntimeError("fixed journal parent is not private")
        self.parent_fd = os.open(JOURNAL.parent, os.O_RDONLY | os.O_DIRECTORY |
                                 os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            self.fd = os.open(JOURNAL.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                              os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=self.parent_fd)
            os.fsync(self.parent_fd)
        except BaseException:
            os.close(self.parent_fd)
            raise

    def emit(self, event, **details):
        payload = _json_bytes({"utc": _utc(), "event": event, **details}) + b"\n"
        while payload:
            written = os.write(self.fd, payload)
            if written <= 0:
                raise RuntimeError("journal append failed")
            payload = payload[written:]
        os.fsync(self.fd)

    def close(self):
        os.fsync(self.fd)
        os.close(self.fd)
        os.fsync(self.parent_fd)
        os.close(self.parent_fd)


def _run_ssh(operation, journal, *, preflight=None, source_binding=None):
    if operation == "write" and not isinstance(source_binding, dict):
        raise RuntimeError("write requires exact previously hashed source binding")
    if operation != "write" and source_binding is not None:
        raise RuntimeError("source binding is only legal for the sole write")
    argv = transport.build_ssh_argv(operation, preflight=preflight)
    started = _utc()
    source_fd = None
    if operation == "write":
        source_fd = os.open(transport.SOURCE, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(source_fd)
            named = transport.SOURCE.lstat()
            identity = ("st_dev", "st_ino", "st_size", "st_mode",
                        "st_mtime_ns", "st_ctime_ns")
            if (info.st_size != transport.SOURCE_BYTES or info.st_nlink != 1 or
                    not stat.S_ISREG(info.st_mode) or
                    stat.S_IMODE(info.st_mode) != 0o600 or
                    any(getattr(info, key) != getattr(named, key)
                        for key in identity)):
                raise RuntimeError("source metadata drift before SSH write")
            if source_binding != {
                    "path": str(transport.SOURCE),
                    "bytes": transport.SOURCE_BYTES,
                    "sha256": transport.SOURCE_SHA256,
                    "device": info.st_dev, "inode": info.st_ino,
                    "mode": "0o600"}:
                raise RuntimeError("write source FD differs from hashed source binding")
            digest, count = hashlib.sha256(), 0
            while True:
                block = os.read(source_fd, 4 * 1024 * 1024)
                if not block:
                    break
                count += len(block)
                digest.update(block)
            if (count != transport.SOURCE_BYTES or
                    digest.hexdigest() != transport.SOURCE_SHA256 or
                    any(getattr(snapshot, key) != getattr(info, key)
                        for snapshot in (os.fstat(source_fd),
                                         transport.SOURCE.lstat())
                        for key in identity)):
                raise RuntimeError("source FD bytes/inode drift before SSH write")
            os.lseek(source_fd, 0, os.SEEK_SET)
            journal.emit("SOURCE_FD_REHASH_PASS", bytes=count,
                         sha256=digest.hexdigest(), device=info.st_dev,
                         inode=info.st_ino)
        except BaseException:
            os.close(source_fd)
            raise
    proc = None
    timeout = False
    try:
        try:
            if operation == "write":
                # Last local gate after host-key setup, source rehash and
                # journal fsync, directly before the sole SSH spawn.
                _check_controller_breakers()
            proc = subprocess.Popen(argv, stdin=source_fd if source_fd is not None else subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    shell=False, close_fds=True)
        except BaseException as error:
            journal.emit("RAW_SSH_TRANSPORT", operation=operation, argv=argv,
                         argv_sha256=transport.argv_sha256(argv),
                         local_ssh_pid=None, started_utc=started,
                         finished_utc=_utc(), exit_code=None, timed_out=False,
                         stdout_bytes=0, stdout_sha256=_sha(b""), stdout_b64="",
                         stderr_bytes=0, stderr_sha256=_sha(b""), stderr_b64="",
                         spawn_error=repr(error))
            raise
        try:
            stdout, stderr = proc.communicate(timeout=600 if operation == "write" else 300)
        except subprocess.TimeoutExpired:
            timeout = True
            proc.kill()  # Local SSH only; an incomplete remote target is never cleaned up.
            stdout, stderr = proc.communicate()
        event = {"operation": operation, "argv": argv,
                 "argv_sha256": transport.argv_sha256(argv),
                 "local_ssh_pid": proc.pid, "started_utc": started,
                 "finished_utc": _utc(), "exit_code": proc.returncode,
                 "timed_out": timeout,
                 "stdout_bytes": len(stdout), "stdout_sha256": _sha(stdout),
                 "stdout_b64": base64.b64encode(stdout).decode("ascii"),
                 "stderr_bytes": len(stderr), "stderr_sha256": _sha(stderr),
                 "stderr_b64": base64.b64encode(stderr).decode("ascii")}
        journal.emit("RAW_SSH_TRANSPORT", **event)
        if timeout or proc.returncode != 0:
            raise RuntimeError("SSH " + operation + " returned nonzero/timeout; raw transport archived")
        parsed = json.loads(stdout)
        if not isinstance(parsed, dict):
            raise RuntimeError("remote receipt is not a JSON object")
        journal.emit("REMOTE_RECEIPT", operation=operation, receipt=parsed)
        return parsed
    finally:
        if source_fd is not None:
            os.close(source_fd)


def run_once():
    """Not callable under this frozen HOLD request; future review must enable."""
    request, assignment = authorize()  # Current HOLD stops here before all SSH.
    source = transport.checked_local_source()
    initial_breakers = _check_controller_breakers()
    journal = Journal()  # O_EXCL and fsync consume exactly one assignment attempt.
    pre = None
    persistence = None
    failure = None
    post_failure = None
    try:
        try:
            journal.emit("ONE_ATTEMPT_CONSUMED", request_sha256=assignment["request_sha256"],
                         assignment_sha256=_sha(_read_evidence(ASSIGNMENT)),
                         source=source, controller=initial_breakers)
            pre = _run_ssh("probe", journal)
            transport.validate_probe(pre, before_write=True)
            journal.emit("FRESH_PREFLIGHT_PASS", old_inventory=pre["old_inventory"])
            _check_controller_breakers()
            written = _run_ssh("write", journal, preflight=pre)
            continuity = transport.compare_preflight_write(pre, written)
            journal.emit("PREFLIGHT_TO_WRITE_CONTINUITY_PASS", result=continuity)
            verified = _run_ssh("verify", journal)  # A distinct SSH process.
            persistence = transport.compare_sessions(written, verified,
                                                     preflight=pre)
            journal.emit("CROSS_SESSION_PERSISTENCE_PASS", result=persistence)
        except BaseException as error:
            failure = repr(error)
            journal.emit("TRANSFER_HOLD", error=failure)
        finally:
            # Mandatory even after a failed preflight or a partial remote write.
            try:
                post = _run_ssh("probe", journal)
                transport.validate_probe(post, before_write=False)
                if pre is not None:
                    transport.compare_postflight(pre, post, persistence=persistence)
                journal.emit("FINAL_READ_ONLY_POSTFLIGHT", snapshot=post,
                             compared_with_preflight=pre is not None)
            except BaseException as error:
                post_failure = repr(error)
                journal.emit("POSTFLIGHT_HOLD", error=post_failure)
        if failure or post_failure or persistence is None:
            raise RuntimeError("transfer HOLD; raw journal retained at fixed path")
        journal.emit("TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW",
                     persistence=persistence)
        return persistence
    finally:
        journal.close()


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: source transfer is disabled pending independent review and a new one-shot assignment")
