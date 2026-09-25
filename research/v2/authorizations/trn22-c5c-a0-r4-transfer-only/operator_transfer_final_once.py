"""DISABLED trn22 r4 continuity-repair draft; no dispatch authority exists.

This source now has different core/transport/remote hashes from the prior
static-PASS package. A new independent static review, successor final request
and separate Supervisor one-shot assignment are required. The frozen request
below explicitly disables dispatch, including if old review files reappear.
There is no checkout, G2, training, evaluation, cleanup, overwrite or retry.
"""

import json
import os
from pathlib import Path
import stat
import sys

import operator_transfer_once as core


HERE = Path(__file__).resolve().parent
REQUEST = HERE / "TRANSFER_REQUEST.ENABLEMENT_CANDIDATE.json"
FINAL_REVIEW = HERE / "INDEPENDENT_FINAL_ENABLEMENT_REVIEW.json"
ASSIGNMENT = HERE / "SUPERVISOR_FINAL_TRANSFER_ASSIGNMENT.json"
REPAIRED_STATIC_REVIEW = HERE / "INDEPENDENT_REPAIRED_STATIC_REVIEW.md"
CORE_WRAPPER = HERE / "operator_transfer_once.py"
TRANSPORT = HERE / "stage_r4_transport.py"
REMOTE = HERE / "stage_r4_remote.py"
MARKER = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/"
              "trn22-c5c-a0-r4-transfer-final-r1-one-attempt.marker")
REQUEST_SHA256 = "833737bbe76a7b7acc287c247616527b199b5f515a46fcf8de0190595600ee57"
CORE_SHA256 = "7590e275f94f79ffac9098bfd3c0345d4c9cb5ce10a26624065b0e04d6189736"
TRANSPORT_SHA256 = "ea14fd19f6992400ad56e59c8dcbec620c784f616c4ff60f6d0568aa721a4ec0"
REMOTE_SHA256 = "023e578693c3fb826b81cfc645190d79e409757dba0bd8df29adad6e3f90a69d"
REPAIRED_STATIC_REVIEW_SHA256 = "7dcfc65cde4dd29a409e8541ace24ada72465b15964d89af5852a5fd89f896db"
SOURCE_COMMIT = "446b657695f086f9b484fa601c9a3b61c2495912"
ASSIGNMENT_ID = "trn22-c5c-a0-r4-transfer-final-r1"
SCOPE = "TRN22_R4_CAPSULE_TRANSFER_ONLY_ONE_ATTEMPT"


def _read_required(path, purpose):
    try:
        return core._read_evidence(path)
    except FileNotFoundError as error:
        raise RuntimeError(purpose + " HOLD: exact fixed-path file is absent") from error


def _read_frozen_request_and_code():
    request_bytes = core._pinned(REQUEST, REQUEST_SHA256)
    request = json.loads(request_bytes)
    for path, digest in ((CORE_WRAPPER, CORE_SHA256),
                         (TRANSPORT, TRANSPORT_SHA256),
                         (REMOTE, REMOTE_SHA256),
                         (REPAIRED_STATIC_REVIEW, REPAIRED_STATIC_REVIEW_SHA256),
                         (core.SOURCE_CONTRACT, core.SOURCE_CONTRACT_SHA256),
                         (core.SOURCE_REVIEW, core.SOURCE_REVIEW_SHA256)):
        core._pinned(path, digest)
    if (request.get("schema") != "trn22-c5c-a0-r4-transfer-final-request-v1" or
            request.get("status") != "CONTINUITY_REPAIR_DISABLED_PENDING_NEW_STATIC_AND_FINAL_REVIEWS_AND_SUPERVISOR_ASSIGNMENT" or
            request.get("dispatch_authorized") is not False or
            request.get("dispatch_meaning") !=
            "NO_DISPATCH_AUTHORITY_PRIOR_STATIC_REVIEW_INVALIDATED_BY_CODE_REPAIR" or
            request.get("prepared_by_role") != "implementer" or
            request.get("assignment_id") != ASSIGNMENT_ID or
            request.get("scope") != SCOPE or
            request.get("host") != "trn22" or
            request.get("hostname") != "ip-172-31-32-253" or
            request.get("uid") != 1000 or
            request.get("endpoint") != core.transport.ENDPOINT or
            request.get("source_commit") != SOURCE_COMMIT or
            request.get("capsule_source_outside_git") != str(core.transport.SOURCE) or
            request.get("capsule_bytes") != core.transport.SOURCE_BYTES or
            request.get("capsule_sha256") != core.transport.SOURCE_SHA256 or
            request.get("target") != core.transport.TARGET or
            request.get("one_shot_marker") != str(MARKER) or
            request.get("journal") != str(core.JOURNAL) or
            request.get("source_stage_contract_sha256") != core.SOURCE_CONTRACT_SHA256 or
            request.get("source_stage_independent_review_sha256") != core.SOURCE_REVIEW_SHA256 or
            request.get("historical_repaired_static_review_sha256_not_authority") !=
            REPAIRED_STATIC_REVIEW_SHA256 or
            request.get("core_wrapper_sha256") != CORE_SHA256 or
            request.get("transport_sha256") != TRANSPORT_SHA256 or
            request.get("remote_sha256") != REMOTE_SHA256 or
            request.get("ssh_identity") != {
                "known_hosts_sha256": core.transport.KNOWN_HOSTS_SHA256,
                "host_fingerprint": core.transport.HOST_FINGERPRINT,
                "host_key_algorithm": "ssh-ed25519",
                "config_file": "/dev/null",
            } or
            any(request.get(key) is not True for key in (
                "required_fresh_read_only_preflight",
                "preflight_to_write_identity_binding_required",
                "live_remote_stop_process_safety_before_o_excl_required",
                "local_stop_and_breaker_gates_before_marker_and_write",
                "raw_ssh_transport_journal_required", "distinct_ssh_readback_required",
                "final_read_only_postflight_even_on_failure_required", "one_attempt")) or
            any(request.get(key) is not False for key in (
                "same_path_overwrite_delete_rename_or_retry_authorized",
                "source_checkout_authorized", "cpu_g2_authorized",
                "native_run_authorized", "eval_authorized",
                "monitor_operation_authorized", "submission_authorized")) or
            any(key in request for key in (
                "final_independent_review_sha256", "supervisor_assignment_sha256"))):
        raise RuntimeError("FINAL REQUEST HOLD: scope, source, safety or code binding drift")
    wrapper_sha = core._sha(core._read_evidence(HERE / "operator_transfer_final_once.py"))
    return request, {"request_sha256": core._sha(request_bytes),
                     "final_wrapper_sha256": wrapper_sha}


def _expected_final_review(request, binding):
    return {
        "schema": "trn22-r4-final-transfer-independent-review-v1",
        "verdict": "PASS",
        "reviewer_role": "independent_verifier",
        "request_sha256": binding["request_sha256"],
        "final_wrapper_sha256": binding["final_wrapper_sha256"],
        "core_wrapper_sha256": CORE_SHA256,
        "transport_sha256": TRANSPORT_SHA256,
        "remote_sha256": REMOTE_SHA256,
        "historical_static_review_sha256_not_authority": REPAIRED_STATIC_REVIEW_SHA256,
        "source_stage_contract_sha256": core.SOURCE_CONTRACT_SHA256,
        "source_stage_independent_review_sha256": core.SOURCE_REVIEW_SHA256,
        "source_commit": SOURCE_COMMIT,
        "capsule_bytes": core.transport.SOURCE_BYTES,
        "capsule_sha256": core.transport.SOURCE_SHA256,
        "target": core.transport.TARGET,
        "journal": str(core.JOURNAL),
        "one_shot_marker": str(MARKER),
        "assignment_id": ASSIGNMENT_ID,
        "scope": SCOPE,
        "host": "trn22",
        "endpoint": core.transport.ENDPOINT,
        "no_host_action_by_reviewer": True,
    }


def _expected_assignment(request, binding, final_review_sha256):
    return {
        "schema": "trn22-r4-final-transfer-supervisor-assignment-v1",
        "issuer_role": "supervisor",
        "dispatch_authorized": True,
        "assignment_id": ASSIGNMENT_ID,
        "scope": SCOPE,
        "host": "trn22",
        "endpoint": core.transport.ENDPOINT,
        "request_sha256": binding["request_sha256"],
        "final_independent_review_sha256": final_review_sha256,
        "final_wrapper_sha256": binding["final_wrapper_sha256"],
        "core_wrapper_sha256": CORE_SHA256,
        "transport_sha256": TRANSPORT_SHA256,
        "remote_sha256": REMOTE_SHA256,
        "historical_static_review_sha256_not_authority": REPAIRED_STATIC_REVIEW_SHA256,
        "source_stage_contract_sha256": core.SOURCE_CONTRACT_SHA256,
        "source_stage_independent_review_sha256": core.SOURCE_REVIEW_SHA256,
        "source_commit": SOURCE_COMMIT,
        "capsule_bytes": core.transport.SOURCE_BYTES,
        "capsule_sha256": core.transport.SOURCE_SHA256,
        "target": core.transport.TARGET,
        "journal": str(core.JOURNAL),
        "one_shot_marker": str(MARKER),
        "one_attempt": True,
        "source_checkout_authorized": False,
        "cpu_g2_authorized": False,
        "native_run_authorized": False,
        "eval_authorized": False,
        "submission_authorized": False,
    }


def _validate_authority_files(request, binding, review_bytes, assignment_bytes):
    try:
        review = json.loads(review_bytes)
        assignment = json.loads(assignment_bytes)
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("FINAL AUTHORITY HOLD: malformed review or assignment") from error
    if review != _expected_final_review(request, binding):
        raise RuntimeError("FINAL REVIEW HOLD: actual independent file is not exact PASS")
    review_sha = core._sha(review_bytes)
    if assignment != _expected_assignment(request, binding, review_sha):
        raise RuntimeError("SUPERVISOR ASSIGNMENT HOLD: exact review/request/marker binding mismatch")
    return {**binding, "final_review_sha256": review_sha,
            "supervisor_assignment_sha256": core._sha(assignment_bytes)}


def authorize_final():
    """Explicit HOLD even if stale files or crafted role labels are present."""
    request, binding = _read_frozen_request_and_code()
    if request["dispatch_authorized"] is not True:
        raise RuntimeError("DISPATCH HOLD: continuity repair requires a new reviewed successor")
    review_bytes = _read_required(FINAL_REVIEW, "FINAL REVIEW")
    assignment_bytes = _read_required(ASSIGNMENT, "SUPERVISOR ASSIGNMENT")
    authority = _validate_authority_files(request, binding, review_bytes,
                                          assignment_bytes)
    core.transport.checked_ssh_identity()  # Local host-key proof only; no SSH.
    return authority


def _open_marker_parent():
    """Resolve every fixed directory component without following a symlink."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/", flags)
    try:
        for name in MARKER.parent.parts[1:]:
            child = os.open(name, flags, dir_fd=fd)
            info = os.fstat(child)
            if (not stat.S_ISDIR(info.st_mode) or
                    info.st_uid not in (0, os.getuid()) or
                    stat.S_IMODE(info.st_mode) & 0o022):
                os.close(child)
                raise RuntimeError("unsafe one-shot marker ancestor")
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _consume_fixed_marker(authority):
    """Durably consume this assignment once; preserve even a partial marker."""
    named = MARKER.parent.lstat()
    if (not stat.S_ISDIR(named.st_mode) or named.st_uid != os.getuid() or
            stat.S_IMODE(named.st_mode) & 0o077):
        raise RuntimeError("one-shot marker parent is not private")
    parent_fd = _open_marker_parent()
    fd = None
    try:
        parent = os.fstat(parent_fd)
        if (parent.st_dev, parent.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError("one-shot marker parent changed")
        fd = os.open(MARKER.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent_fd)
        payload = core._json_bytes({"schema": "trn22-r4-final-transfer-one-shot-marker-v1",
                                    "utc": core._utc(), "assignment_id": ASSIGNMENT_ID,
                                    "host": "trn22", "scope": SCOPE,
                                    "target": core.transport.TARGET,
                                    **authority}) + b"\n"
        remaining = payload
        while remaining:
            count = os.write(fd, remaining)
            if count <= 0:
                raise RuntimeError("one-shot marker write failed")
            remaining = remaining[count:]
        os.fsync(fd)
        os.fsync(parent_fd)
        return core._sha(payload)
    finally:
        if fd is not None:
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)


def run_once():
    """Only a separately reviewed and assigned transfer can pass the first gate."""
    authority = authorize_final()  # Both future fixed-path files are absent now.
    source = core.transport.checked_local_source()
    first_breakers = core._check_controller_breakers()
    marker_sha = _consume_fixed_marker(authority)  # O_EXCL consumes one attempt.
    journal = core.Journal()  # A second fixed O_EXCL gate before any SSH.
    pre = None
    persistence = None
    failure = None
    post_failure = None
    try:
        try:
            journal.emit("FINAL_ONE_ATTEMPT_CONSUMED", authority=authority,
                         marker_sha256=marker_sha, source=source,
                         controller=first_breakers)
            pre = core._run_ssh("probe", journal)
            core.transport.validate_probe(pre, before_write=True)
            journal.emit("FRESH_PREFLIGHT_PASS", old_inventory=pre["old_inventory"])
            if authorize_final() != authority:
                raise RuntimeError("final review or assignment drifted after preflight")
            core._pinned(MARKER, marker_sha)
            core._check_controller_breakers()  # Immediately before the only write.
            written = core._run_ssh("write", journal, preflight=pre)
            continuity = core.transport.compare_preflight_write(pre, written)
            journal.emit("PREFLIGHT_TO_WRITE_CONTINUITY_PASS", result=continuity)
            verified = core._run_ssh("verify", journal)  # Distinct SSH process.
            persistence = core.transport.compare_sessions(written, verified,
                                                          preflight=pre)
            journal.emit("CROSS_SESSION_PERSISTENCE_PASS", result=persistence)
        except BaseException as error:
            failure = repr(error)
            journal.emit("TRANSFER_HOLD", error=failure)
        finally:
            # Mandatory distinct read-only probe even after failed preflight/write.
            try:
                post = core._run_ssh("probe", journal)
                core.transport.validate_probe(post, before_write=False)
                if pre is not None:
                    core.transport.compare_postflight(pre, post,
                                                      persistence=persistence)
                journal.emit("FINAL_READ_ONLY_POSTFLIGHT", snapshot=post,
                             compared_with_preflight=pre is not None)
            except BaseException as error:
                post_failure = repr(error)
                journal.emit("POSTFLIGHT_HOLD", error=post_failure)
        if failure or post_failure or persistence is None:
            raise RuntimeError("transfer HOLD; marker and raw journal retained")
        journal.emit("TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW",
                     persistence=persistence)
        return persistence
    finally:
        journal.close()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("FINAL AUTHORITY HOLD: no alternate target, marker or operation")
    run_once()
