"""Exact trn22 FIFO-lock capsule transfer-only final request candidate.

Request intent is not authority. No marker, journal or SSH is reachable until
the future fixed-path independent final PASS and a separate Supervisor
one-shot assignment authenticate these actual bytes. No checkout or native run.
"""

import json
import os
from pathlib import Path
import stat
import sys

import operator_transfer_lock_core_r2 as core


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SUBJECT = (REPO / "research/v2/authorizations/"
           "trn22-c5c-a0-r4-transfer-lock-nonblock-hold")
REQUEST = HERE / "TRANSFER_REQUEST.FIFO_FINAL_R2.json"
STATIC_REVIEW = (REPO / "research/v2/verification/"
                 "20260925-trn22-fifo-nonblock-5740-independent-static-review.md")
HISTORICAL_R1_HOLD = (REPO / "research/v2/verification/"
                      "20260925-trn22-fifo-final-r1-independent-HOLD.md")
FINAL_REVIEW = HERE / "INDEPENDENT_FIFO_FINAL_R2_REVIEW.json"
ASSIGNMENT = HERE / "SUPERVISOR_FIFO_FINAL_R2_ASSIGNMENT.json"
MARKER = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/"
              "trn22-c5c-a0-r4-fifo-final-r2-one-attempt.marker")
JOURNAL = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/"
               "trn22-c5c-a0-r4-fifo-final-r2-one-attempt.jsonl")
REQUEST_SHA256 = "b8bcfbb5d898bfb6c3b6308600b9c2e91d474fdccee66620bd6ff055f30f2ec2"
STATIC_REVIEW_SHA256 = "67c3e430bd14623b4f11a305f9a445ad6d3276e6e63833a693d004aa94988f6d"
HISTORICAL_R1_HOLD_SHA256 = "8d1bf8caa6fa0643b067c9242cbeed1e8adb756592bbedc274fefcc328ca04ad"
CORE_SHA256 = "42d21f959ca2c7bab156bb5bdc320203c1278971092fd40e95da27e24e0846b8"
TRANSPORT_SHA256 = "fde41bbe3ab5367840d1b225316b2d961a562a2dee4e6ca9509120e83b060d84"
REMOTE_SHA256 = "cdf06f700722e144dcb304e517a7e000fe0c6b2bcb210e4d2bb38f92757d5e27"
SUBJECT_SHA256 = {
    "GATE_CONTRACT.HOLD.json": "df3a8a776b10fdf31bf307ff4b036131395b8a744c88699dc80d7e0f64ea0b03",
    "RUNBOOK.md": "18342e895e1ecb38032dd7d965a3820abf29ded152161ba381d8824bc0cbe431",
    "stage_r4_remote_lock_hold.py": REMOTE_SHA256,
    "stage_r4_transport_lock_hold.py": "4a8bfcc7ca091288e69b395cc28fe86607d257eef5f6fdbf2ba6d54a55781dfb",
    "test_lock_final_scan_hold.py": "c73a1f8b813d40e0ca4b2a2b110f931522376ccd99221a157d98e449cedf2f6b",
}
ASSIGNMENT_ID = "trn22-c5c-a0-r4-fifo-lock-transfer-final-r2"
SCOPE = "TRN22_R4_CAPSULE_TRANSFER_ONLY_ONE_ATTEMPT"


def _read_required(path, purpose):
    try:
        return core._read_evidence(path)
    except FileNotFoundError as error:
        raise RuntimeError(purpose + " HOLD: exact fixed-path file is absent") from error


def _frozen_binding():
    """Pin static PASS, exact disabled subject and new executable bytes."""
    request_bytes = core._pinned(REQUEST, REQUEST_SHA256)
    request = json.loads(request_bytes)
    core._pinned(STATIC_REVIEW, STATIC_REVIEW_SHA256)
    core._pinned(HISTORICAL_R1_HOLD, HISTORICAL_R1_HOLD_SHA256)
    for filename, digest in SUBJECT_SHA256.items():
        core._pinned(SUBJECT / filename, digest)
    core._pinned(HERE / "operator_transfer_lock_core_r2.py", CORE_SHA256)
    core._pinned(HERE / "stage_r4_transport_lock_final_r2.py", TRANSPORT_SHA256)
    core._pinned(core.SOURCE_CONTRACT, core.SOURCE_CONTRACT_SHA256)
    core._pinned(core.SOURCE_REVIEW, core.SOURCE_REVIEW_SHA256)
    if (request.get("schema") !=
            "trn22-c5c-a0-r4-fifo-lock-transfer-final-r2-request-v1" or
            request.get("dispatch_authorized") is not True or
            request.get("dispatch_meaning") !=
            "REQUEST_INTENT_ONLY_NO_HOST_ACTION_WITHOUT_ACTUAL_FINAL_PASS_AND_SUPERVISOR_ASSIGNMENT" or
            request.get("prepared_by_role") != "implementer" or
            request.get("assignment_id") != ASSIGNMENT_ID or
            request.get("scope") != SCOPE or
            request.get("host") != "trn22" or
            request.get("hostname") != "ip-172-31-32-253" or
            request.get("uid") != 1000 or
            request.get("endpoint") != core.transport.ENDPOINT or
            request.get("source_commit") !=
            "446b657695f086f9b484fa601c9a3b61c2495912" or
            request.get("capsule_source_outside_git") !=
            str(core.transport.SOURCE) or
            request.get("capsule_bytes") != core.transport.SOURCE_BYTES or
            request.get("capsule_sha256") != core.transport.SOURCE_SHA256 or
            request.get("target") != core.transport.TARGET or
            request.get("one_shot_marker") != str(MARKER) or
            request.get("journal") != str(JOURNAL) or
            request.get("source_stage_contract_sha256") !=
            core.SOURCE_CONTRACT_SHA256 or
            request.get("source_stage_independent_review_sha256") !=
            core.SOURCE_REVIEW_SHA256 or
            request.get("independent_disabled_static_review_sha256") !=
            STATIC_REVIEW_SHA256 or
            request.get("historical_final_r1_hold_sha256_not_authority") !=
            HISTORICAL_R1_HOLD_SHA256 or
            request.get("disabled_subject_file_sha256") != SUBJECT_SHA256 or
            request.get("core_wrapper_sha256") != CORE_SHA256 or
            request.get("enabled_transport_sha256") != TRANSPORT_SHA256 or
            request.get("remote_sha256") != REMOTE_SHA256 or
            request.get("final_label_to_o_excl_max_ns") != 1000000000 or
            request.get("ssh_identity") != {
                "known_hosts_sha256": core.transport.KNOWN_HOSTS_SHA256,
                "host_fingerprint": core.transport.HOST_FINGERPRINT,
                "host_key_algorithm": "ssh-ed25519", "config_file": "/dev/null"} or
            core.transport.REMOTE_SHA256 != REMOTE_SHA256 or
            core.transport.REMOTE != SUBJECT / "stage_r4_remote_lock_hold.py" or
            core.transport.remote_constants().MAX_FINAL_SCAN_TO_OPEN_NS !=
            1000000000 or
            core.JOURNAL != JOURNAL or
            any(request.get(key) is not True for key in (
                "required_fresh_read_only_preflight",
                "preflight_to_write_host_boot_mount_lock_identity_binding_required",
                "complete_old_r1_r2_r3_file_backed_git_inventory_required",
                "private_cache_0700_no_acl_existing_lock_held_final_label_and_live_safety_required",
                "local_stop_and_breaker_gates_before_marker_and_write",
                "local_stop_and_breaker_recheck_immediately_before_marker_o_excl",
                "local_stop_and_breaker_recheck_after_source_fd_hash_before_write_popen",
                "write_source_fd_matches_pre_marker_hashed_binding",
                "raw_ssh_argv_exit_stdout_stderr_journal_required",
                "distinct_ssh_readback_required",
                "final_read_only_postflight_even_on_failure_required", "one_attempt")) or
            any(request.get(key) is not False for key in (
                "same_path_overwrite_delete_rename_or_retry_authorized",
                "source_checkout_authorized", "cpu_g2_authorized",
                "native_run_authorized", "eval_authorized",
                "monitor_operation_authorized", "submission_authorized")) or
            any(key in request for key in (
                "final_independent_review_sha256", "supervisor_assignment_sha256"))):
        raise RuntimeError("FIFO FINAL REQUEST HOLD: exact scope/code binding drift")
    wrapper_sha = core._sha(core._read_evidence(Path(__file__).resolve()))
    return {
        "request_sha256": core._sha(request_bytes),
        "final_wrapper_sha256": wrapper_sha,
        "disabled_subject_commit": request["disabled_subject_commit"],
        "canonical_disabled_subject_commit":
            request["canonical_disabled_subject_commit"],
        "independent_disabled_static_review_sha256": STATIC_REVIEW_SHA256,
        "historical_final_r1_hold_sha256_not_authority":
            HISTORICAL_R1_HOLD_SHA256,
        "disabled_subject_file_sha256": SUBJECT_SHA256,
        "core_wrapper_sha256": CORE_SHA256,
        "enabled_transport_sha256": TRANSPORT_SHA256,
        "remote_sha256": REMOTE_SHA256,
        "source_stage_contract_sha256": core.SOURCE_CONTRACT_SHA256,
        "source_stage_independent_review_sha256": core.SOURCE_REVIEW_SHA256,
        "source_commit": request["source_commit"],
        "capsule_bytes": core.transport.SOURCE_BYTES,
        "capsule_sha256": core.transport.SOURCE_SHA256,
        "target": core.transport.TARGET,
        "one_shot_marker": str(MARKER), "journal": str(JOURNAL),
        "assignment_id": ASSIGNMENT_ID, "scope": SCOPE,
        "host": "trn22", "endpoint": core.transport.ENDPOINT,
    }


def _expected_final_review(binding):
    return {**binding,
            "schema": "trn22-r4-fifo-lock-final-r2-independent-review-v1",
            "verdict": "PASS", "reviewer_role": "independent_verifier",
            "no_host_action_by_reviewer": True}


def _expected_assignment(binding, review_sha256):
    return {**binding,
            "schema": "trn22-r4-fifo-lock-final-r2-supervisor-assignment-v1",
            "issuer_role": "supervisor", "dispatch_authorized": True,
            "final_independent_review_sha256": review_sha256,
            "one_attempt": True, "source_checkout_authorized": False,
            "cpu_g2_authorized": False, "native_run_authorized": False,
            "eval_authorized": False, "monitor_operation_authorized": False,
            "submission_authorized": False}


def _validate_authority_files(binding, review_bytes, assignment_bytes):
    try:
        review = json.loads(review_bytes)
        assignment = json.loads(assignment_bytes)
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("FIFO FINAL AUTHORITY HOLD: malformed files") from error
    if review != _expected_final_review(binding):
        raise RuntimeError("FIFO FINAL REVIEW HOLD: actual file is not exact PASS")
    review_sha = core._sha(review_bytes)
    if assignment != _expected_assignment(binding, review_sha):
        raise RuntimeError("FIFO SUPERVISOR HOLD: assignment binding mismatch")
    return {**binding, "final_review_sha256": review_sha,
            "supervisor_assignment_sha256": core._sha(assignment_bytes)}


def authorize_final():
    """Both future fixed files must authenticate before local host-key checks."""
    binding = _frozen_binding()
    review_bytes = _read_required(FINAL_REVIEW, "FIFO FINAL REVIEW")
    assignment_bytes = _read_required(ASSIGNMENT, "FIFO SUPERVISOR ASSIGNMENT")
    authority = _validate_authority_files(binding, review_bytes, assignment_bytes)
    core.transport.checked_ssh_identity()  # Local-only, never SSH.
    return authority


def _open_marker_parent():
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
    """One O_EXCL/fsync marker; any partial remains as a consumed attempt."""
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
        # This is the last local gate after slow authority and ancestry work,
        # directly before the exclusive attempt marker is created.
        core._check_controller_breakers()
        fd = os.open(MARKER.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent_fd)
        payload = core._json_bytes({
            "schema": "trn22-r4-fifo-lock-final-r2-one-shot-marker-v1",
            "utc": core._utc(), "assignment_id": ASSIGNMENT_ID,
            "host": "trn22", "scope": SCOPE,
            "target": core.transport.TARGET, **authority}) + b"\n"
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
    """Exactly one transfer-only attempt after separate review and assignment."""
    authority = authorize_final()  # Both future fixed files are absent now.
    source = core.transport.checked_local_source()
    first_breakers = core._check_controller_breakers()
    if authorize_final() != authority:
        raise RuntimeError("FIFO final review or assignment drifted before marker")
    core._check_controller_breakers()  # Slow reauthorization may have raced STOP.
    marker_sha = _consume_fixed_marker(authority)
    journal = core.Journal()  # Second fixed O_EXCL before any SSH.
    pre = None
    persistence = None
    failure = None
    post_failure = None
    try:
        try:
            journal.emit("FIFO_FINAL_ONE_ATTEMPT_CONSUMED", authority=authority,
                         marker_sha256=marker_sha, source=source,
                         controller=first_breakers)
            pre = core._run_ssh("probe", journal)
            core.transport.validate_probe(pre, before_write=True)
            journal.emit("FRESH_PREFLIGHT_PASS", old_inventory=pre["old_inventory"])
            if authorize_final() != authority:
                raise RuntimeError("FIFO final review or assignment drifted after preflight")
            core._pinned(MARKER, marker_sha)
            core._check_controller_breakers()  # Immediately before sole write.
            written = core._run_ssh("write", journal, preflight=pre,
                                    source_binding=source)
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
            # A failed or partial write still receives a distinct read-only probe.
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
            raise RuntimeError("transfer HOLD; fixed marker and raw journal retained")
        journal.emit("TRANSFER_ONLY_DONE_PENDING_INDEPENDENT_RESULT_REVIEW",
                     persistence=persistence)
        return persistence
    finally:
        journal.close()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("FIFO FINAL AUTHORITY HOLD: no alternate host or operation")
    run_once()
