"""Disabled one-attempt trn22 r4 transfer wrapper with raw SSH evidence.

This implementation is deliberately non-dispatchable in the frozen HOLD
request. No CLI executes run_once. A later *new* reviewed enablement package
and exact Supervisor assignment are required. No checkout or training exists.
"""

import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

import stage_r4_transport as transport


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REQUEST = HERE / "TRANSFER_REQUEST.HOLD.json"
INDEPENDENT_REVIEW = HERE / "INDEPENDENT_IMPLEMENTATION_REVIEW.md"
ASSIGNMENT = HERE / "SUPERVISOR_TRANSFER_ASSIGNMENT.json"
SOURCE_CONTRACT = REPO / "research/v2/authorizations/trn22-c5c-a0-r4-source-stage/GATE_CONTRACT.json"
SOURCE_REVIEW = REPO / "research/v2/authorizations/trn22-c5c-a0-r4-source-stage/INDEPENDENT_STATIC_VERIFIER_RECEIPT.md"
CONTROL_STATE = Path("/home/poojith-devan/Desktop/Trainium/research/v2/state.json")
JOURNAL = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r4-transfer-only-one-attempt.jsonl")
SOURCE_CONTRACT_SHA256 = "79878b16f459d4c12b32cb0ebb85d205dad86d584ae09654e0e8161920deab69"
SOURCE_REVIEW_SHA256 = "d49e8c03567210283074c44a2f2937594ada39f43e399f97d176c373c1a32c1e"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _pinned(path, digest):
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("missing exact review or assignment SHA-256")
    data = path.read_bytes()  # Actual fixed-path content, not a length-only claim.
    if _sha(data) != digest:
        raise RuntimeError("reviewed fixed-path evidence drift: " + str(path))
    return data


def authorize():
    request_bytes = REQUEST.read_bytes()
    request = json.loads(request_bytes)
    if (request.get("schema") != "trn22-c5c-a0-r4-transfer-only-request-v1" or
            request.get("dispatch_authorized") is not True):
        raise RuntimeError("DISPATCH HOLD: frozen transfer request is disabled")
    _pinned(SOURCE_CONTRACT, SOURCE_CONTRACT_SHA256)
    _pinned(SOURCE_REVIEW, SOURCE_REVIEW_SHA256)
    if (request.get("source_commit") != "446b657695f086f9b484fa601c9a3b61c2495912" or
            request.get("capsule_sha256") != transport.SOURCE_SHA256 or
            request.get("capsule_bytes") != transport.SOURCE_BYTES or
            request.get("target") != transport.TARGET or
            request.get("journal") != str(JOURNAL) or
            request.get("remote_sha256") != transport.REMOTE_SHA256 or
            request.get("transport_sha256") != _sha((HERE / "stage_r4_transport.py").read_bytes()) or
            request.get("operator_sha256") != _sha(Path(__file__).read_bytes()) or
            request.get("independent_review_status") != "PASS" or
            request.get("assignment_scope") != "TRN22_R4_CAPSULE_TRANSFER_ONLY_ONE_ATTEMPT"):
        raise RuntimeError("transfer request/code/binding drift")
    review = _pinned(INDEPENDENT_REVIEW, request.get("independent_review_sha256"))
    if b"PASS" not in review or b"HOLD" in review[:200]:
        raise RuntimeError("independent review is not a clean PASS")
    assignment = json.loads(_pinned(ASSIGNMENT, request.get("assignment_sha256")))
    if (assignment.get("dispatch_authorized") is not True or
            assignment.get("scope") != request["assignment_scope"] or
            assignment.get("host") != "trn22" or
            assignment.get("source_commit") != request["source_commit"] or
            assignment.get("capsule_sha256") != transport.SOURCE_SHA256 or
            assignment.get("target") != transport.TARGET or
            assignment.get("one_attempt") is not True):
        raise RuntimeError("Supervisor one-attempt assignment mismatch")
    return request, assignment


def _check_controller_breakers():
    first = CONTROL_STATE.read_bytes()
    state = json.loads(first)
    breakers = state["breakers"]
    if any(breakers[key].get("tripped") is not False
           for key in ("global_correctness", "trn22")):
        raise RuntimeError("local controller global or trn22 breaker is tripped/unknown")
    if CONTROL_STATE.read_bytes() != first:
        raise RuntimeError("local controller state raced during breaker read")
    return {"path": str(CONTROL_STATE), "sha256": _sha(first),
            "global_tripped": False, "trn22_tripped": False}


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


def _run_ssh(operation, journal):
    argv = transport.build_ssh_argv(operation)
    started = _utc()
    source_fd = None
    if operation == "write":
        source_fd = os.open(transport.SOURCE, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        info = os.fstat(source_fd)
        if (info.st_size != transport.SOURCE_BYTES or info.st_nlink != 1 or
                not stat.S_ISREG(info.st_mode) or
                stat.S_IMODE(info.st_mode) != 0o600):
            os.close(source_fd)
            raise RuntimeError("source drift before SSH write")
    proc = None
    timeout = False
    try:
        try:
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
            journal.emit("ONE_ATTEMPT_CONSUMED", request_sha256=_sha(REQUEST.read_bytes()),
                         assignment_sha256=request["assignment_sha256"],
                         source=source, controller=initial_breakers)
            pre = _run_ssh("probe", journal)
            transport.validate_probe(pre, before_write=True)
            journal.emit("FRESH_PREFLIGHT_PASS", old_inventory=pre["old_inventory"])
            _check_controller_breakers()
            written = _run_ssh("write", journal)
            verified = _run_ssh("verify", journal)  # A distinct SSH process.
            persistence = transport.compare_sessions(written, verified)
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
