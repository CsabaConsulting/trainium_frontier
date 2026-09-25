"""Exact trn21 r4 transfer request; final review and assignment are absent.

The request expresses transfer-only intent. This entry point cannot create its
one-attempt marker or open SSH until a different Verifier writes the exact
fixed-path final PASS and a Supervisor separately writes the exact one-shot
assignment. The disabled core, transport and remote writer remain byte-identical;
the final remote safety delta is in separate pinned files. No checkout, G2,
native run or cleanup is here.
"""

import base64
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import operator_transfer_once as core
import stage_r4_transport_final as final_transport


HERE = Path(__file__).resolve().parent
REQUEST = HERE / "TRANSFER_REQUEST.FINAL_PENDING_REVIEW.json"
FINAL_REVIEW = HERE / "INDEPENDENT_FINAL_ENABLEMENT_REVIEW.json"
ASSIGNMENT = HERE / "SUPERVISOR_FINAL_TRANSFER_ONLY_ASSIGNMENT.json"
DISABLED_REQUEST = HERE / "TRANSFER_REQUEST.HOLD.json"
CORE_WRAPPER = HERE / "operator_transfer_once.py"
TRANSPORT = HERE / "stage_r4_transport.py"
REMOTE = HERE / "stage_r4_remote.py"
FINAL_TRANSPORT = HERE / "stage_r4_transport_final.py"
FINAL_REMOTE = HERE / "stage_r4_remote_final.py"
STATIC_REVIEW = HERE / "INDEPENDENT_STATIC_TRANSFER_REVIEW.md"
SOURCE_STAGE_HOLD_RUNBOOK = core.REPO / "research/v2/authorizations/trn21-c5c-a0-r4-source-stage-hold/RUNBOOK.md"
MARKER = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/"
              "trn21-c5c-a0-r4-transfer-final-r1-one-attempt.marker")

REQUEST_SHA256 = "04a6b215ac796927677a29aebdf414ffe0f361495cca8a45d2dec070bc60f136"
DISABLED_REQUEST_SHA256 = "5af3241bf07ee3275685f1f8881be1a297e36e4542c695149269f2cdc2c1079b"
CORE_SHA256 = "81f69b8f15837fd34aced5463c605c2a91936a36330a33ca6fc32f9fcc6e4621"
TRANSPORT_SHA256 = "fac4a840a43d7addb0ee3579330fbd02b5b49d759b4e7e7b90ee904e14a01647"
REMOTE_SHA256 = "be2e0aca21aaf5e29b330c5c0b004dd8a013e99c2aaaa29e820be53b0a15dd68"
FINAL_TRANSPORT_SHA256 = "8c1e362fd586a00137016d7e0fad6f6ed758124cf7f66716a143895c91b512ba"
FINAL_REMOTE_SHA256 = "9b73cd84a46af46e9ce7e8729e328878a7b29c16ac1556e9adeb93785fd9b4b8"
STATIC_REVIEW_SHA256 = "3803dc589c7ebf7ffd870a3d2eaddeecec77496fd24fc3a1cc92ccba5a482ab3"
SOURCE_STAGE_HOLD_RUNBOOK_SHA256 = "e31b9df3b4b1b2ac0f0fef355d773ef980875c4a3841bc0355387af6af9627f5"
SOURCE_COMMIT = "446b657695f086f9b484fa601c9a3b61c2495912"
CAPSULE_MANIFEST_SHA256 = "7ec8798c3e6d0284a4514a15a0caeb17bc5eb2cb5df935fb7dd6e8f3aabee42a"
DISABLED_CANONICAL_COMMIT = "96f02153fc65e6236c84b3a76983ab91301d1da7"
STATIC_CANONICAL_COMMIT = "3d5e66806c98e3a635c51b595c1cdcc0579eb866"
STATIC_ORIGIN_COMMIT = "9cdc545e92cf65bea08aeb912b9b21a8f30e07f5"
ASSIGNMENT_ID = "trn21-c5c-a0-r4-transfer-final-r1"
SCOPE = "TRN21_R4_CAPSULE_TRANSFER_ONLY_ONE_ATTEMPT"
FINAL_REVIEW_RELATIVE = "research/v2/authorizations/trn21-c5c-a0-r4-transfer-only/INDEPENDENT_FINAL_ENABLEMENT_REVIEW.json"
ASSIGNMENT_RELATIVE = "research/v2/authorizations/trn21-c5c-a0-r4-transfer-only/SUPERVISOR_FINAL_TRANSFER_ONLY_ASSIGNMENT.json"


def _expected_request():
    remote = core.transport.remote_constants()  # Hash checked before this is called.
    return {
        "schema": "trn21-c5c-a0-r4-transfer-final-request-v1",
        "status": "FINAL_REQUEST_PENDING_INDEPENDENT_REVIEW_FRESH_HOST_PREFLIGHT_AND_SUPERVISOR_ASSIGNMENT",
        "dispatch_authorized": True,
        "dispatch_meaning": "TRANSFER_ONLY_REQUEST_INTENT_REQUIRES_ACTUAL_FINAL_PASS_AND_SUPERVISOR_ONE_ATTEMPT_ASSIGNMENT",
        "prepared_by_role": "implementer",
        "assignment_id": ASSIGNMENT_ID,
        "scope": SCOPE,
        "host": "trn21",
        "hostname": remote.HOST,
        "uid": remote.UID,
        "endpoint": core.transport.ENDPOINT,
        "source_commit": SOURCE_COMMIT,
        "capsule_source_outside_git": str(core.transport.SOURCE),
        "capsule_bytes": core.transport.SOURCE_BYTES,
        "capsule_sha256": core.transport.SOURCE_SHA256,
        "capsule_manifest_sha256": CAPSULE_MANIFEST_SHA256,
        "target": core.transport.TARGET,
        "one_shot_marker": str(MARKER),
        "journal": str(core.JOURNAL),
        "source_stage_hold_contract_sha256": core.TRN21_HOLD_PLAN_SHA256,
        "source_stage_hold_runbook_sha256": SOURCE_STAGE_HOLD_RUNBOOK_SHA256,
        "local_capsule_review_sha256": core.CAPSULE_REVIEW_SHA256,
        "disabled_implementation_canonical_commit": DISABLED_CANONICAL_COMMIT,
        "disabled_static_review_canonical_commit": STATIC_CANONICAL_COMMIT,
        "disabled_static_review_origin_commit": STATIC_ORIGIN_COMMIT,
        "disabled_request_sha256": DISABLED_REQUEST_SHA256,
        "disabled_core_wrapper_sha256": CORE_SHA256,
        "transport_sha256": TRANSPORT_SHA256,
        "remote_sha256": REMOTE_SHA256,
        "final_transport_sha256": FINAL_TRANSPORT_SHA256,
        "final_remote_sha256": FINAL_REMOTE_SHA256,
        "disabled_static_review_sha256_not_final_authority": STATIC_REVIEW_SHA256,
        "final_review_fixed_path": FINAL_REVIEW_RELATIVE,
        "supervisor_assignment_fixed_path": ASSIGNMENT_RELATIVE,
        "ssh_identity": {
            "known_hosts_sha256": core.transport.KNOWN_HOSTS_SHA256,
            "host_fingerprint": core.transport.HOST_FINGERPRINT,
            "host_key_algorithm": "ssh-ed25519",
            "config_file": "/dev/null",
        },
        "root_mount_source": remote.MOUNT_SOURCE,
        "root_major_minor": remote.MOUNT_MAJOR_MINOR,
        "root_device": 66305,
        "pretransfer_free_floor_bytes": core.transport.PRE_FREE,
        "posttransfer_free_floor_bytes": core.transport.POST_FREE,
        "minimum_free_inodes": core.transport.MIN_INODES,
        "existing_e65_checkout_preserved": remote.E65_CHECKOUT,
        "future_clean_checkout_remains_absent": remote.CHECKOUT,
        "live_path_absences_verified": False,
        "required_fresh_read_only_preflight": True,
        "preflight_to_writer_boot_mount_parent_binding_required": True,
        "writer_full_live_protected_asset_and_quiescence_probe_required": True,
        "writer_final_stop_process_neuron_target_gate_before_o_excl_required": True,
        "strict_pinned_trn21_host_identity_required": True,
        "local_stop_and_breaker_gates_before_marker_and_write_required": True,
        "raw_ssh_transport_journal_required": True,
        "distinct_ssh_readback_required": True,
        "unconditional_distinct_full_read_only_postflight_required": True,
        "one_attempt": True,
        "same_path_overwrite_delete_rename_or_retry_authorized": False,
        "source_checkout_authorized": False,
        "cpu_g2_authorized": False,
        "native_run_authorized": False,
        "eval_authorized": False,
        "monitor_operation_authorized": False,
        "cleanup_authorized": False,
        "submission_authorized": False,
    }


def _read_frozen_request_and_code():
    """Read actual reviewed files and reject any substitution before SSH."""
    if (Path(core.__file__).resolve() != CORE_WRAPPER.resolve() or
            Path(core.transport.__file__).resolve() != TRANSPORT.resolve() or
            Path(final_transport.__file__).resolve() != FINAL_TRANSPORT.resolve() or
            final_transport.base is not core.transport or
            core.REQUEST != DISABLED_REQUEST or
            core.FROZEN_HOLD_REQUEST_SHA256 != DISABLED_REQUEST_SHA256 or
            core.transport.REMOTE != REMOTE or
            core.transport.REMOTE_SHA256 != REMOTE_SHA256 or
            final_transport.REMOTE != FINAL_REMOTE or
            final_transport.REMOTE_SHA256 != FINAL_REMOTE_SHA256):
        raise RuntimeError("FINAL REQUEST HOLD: imported disabled implementation changed")
    request_bytes = core._pinned(REQUEST, REQUEST_SHA256)
    for path, digest in (
            (DISABLED_REQUEST, DISABLED_REQUEST_SHA256),
            (CORE_WRAPPER, CORE_SHA256),
            (TRANSPORT, TRANSPORT_SHA256),
            (REMOTE, REMOTE_SHA256),
            (FINAL_TRANSPORT, FINAL_TRANSPORT_SHA256),
            (FINAL_REMOTE, FINAL_REMOTE_SHA256),
            (STATIC_REVIEW, STATIC_REVIEW_SHA256),
            (core.TRN21_HOLD_PLAN, core.TRN21_HOLD_PLAN_SHA256),
            (SOURCE_STAGE_HOLD_RUNBOOK, SOURCE_STAGE_HOLD_RUNBOOK_SHA256),
            (core.CAPSULE_REVIEW, core.CAPSULE_REVIEW_SHA256)):
        core._pinned(path, digest)
    plan = json.loads(core._pinned(core.TRN21_HOLD_PLAN,
                                   core.TRN21_HOLD_PLAN_SHA256))
    source = plan["source_capsule"]
    if (source["frozen_source_commit"] != SOURCE_COMMIT or
            source["local_path"] != str(core.transport.SOURCE) or
            source["local_bytes"] != core.transport.SOURCE_BYTES or
            source["local_sha256"] != core.transport.SOURCE_SHA256 or
            source["manifest_sha256"] != CAPSULE_MANIFEST_SHA256 or
            plan["proposed_new_trn21_paths"]["persistent_ext4_capsule"] != core.transport.TARGET or
            plan["pretransfer_safety_floor"]["minimum_root_free_bytes_before_transfer"] !=
            core.transport.PRE_FREE):
        raise RuntimeError("FINAL REQUEST HOLD: frozen source-stage plan disagrees with transfer")
    try:
        request = json.loads(request_bytes)
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("FINAL REQUEST HOLD: malformed frozen request") from error
    if request != _expected_request():
        raise RuntimeError("FINAL REQUEST HOLD: request/source/scope/safety binding drift")
    final_remote = final_transport.remote_constants()
    if (final_remote.SCHEMA != core.transport.SCHEMA or
            final_remote.HOST != "ip-172-31-34-85" or
            final_remote.TARGET != core.transport.TARGET or
            final_remote.BYTES != core.transport.SOURCE_BYTES or
            final_remote.SHA256 != core.transport.SOURCE_SHA256 or
            final_transport.SOURCE != core.transport.SOURCE or
            final_transport.ENDPOINT != core.transport.ENDPOINT or
            final_transport.TARGET != core.transport.TARGET):
        raise RuntimeError("FINAL REQUEST HOLD: final payload/source/host binding drift")
    wrapper_sha = core._sha(core._read_evidence(HERE / "operator_transfer_final_once.py"))
    return request, {"request_sha256": core._sha(request_bytes),
                     "final_wrapper_sha256": wrapper_sha}


def _expected_final_review(binding):
    return {
        "schema": "trn21-r4-final-transfer-independent-review-v1",
        "verdict": "PASS",
        "reviewer_role": "independent_verifier",
        "no_host_action_by_reviewer": True,
        "request_sha256": binding["request_sha256"],
        "final_wrapper_sha256": binding["final_wrapper_sha256"],
        "disabled_request_sha256": DISABLED_REQUEST_SHA256,
        "disabled_core_wrapper_sha256": CORE_SHA256,
        "transport_sha256": TRANSPORT_SHA256,
        "remote_sha256": REMOTE_SHA256,
        "final_transport_sha256": FINAL_TRANSPORT_SHA256,
        "final_remote_sha256": FINAL_REMOTE_SHA256,
        "disabled_static_review_sha256_not_final_authority": STATIC_REVIEW_SHA256,
        "source_stage_hold_contract_sha256": core.TRN21_HOLD_PLAN_SHA256,
        "source_stage_hold_runbook_sha256": SOURCE_STAGE_HOLD_RUNBOOK_SHA256,
        "local_capsule_review_sha256": core.CAPSULE_REVIEW_SHA256,
        "source_commit": SOURCE_COMMIT,
        "capsule_source_outside_git": str(core.transport.SOURCE),
        "capsule_bytes": core.transport.SOURCE_BYTES,
        "capsule_sha256": core.transport.SOURCE_SHA256,
        "target": core.transport.TARGET,
        "one_shot_marker": str(MARKER),
        "journal": str(core.JOURNAL),
        "assignment_id": ASSIGNMENT_ID,
        "scope": SCOPE,
        "host": "trn21",
        "hostname": "ip-172-31-34-85",
        "endpoint": core.transport.ENDPOINT,
        "one_attempt": True,
    }


def _expected_assignment(binding, final_review_sha256):
    return {
        "schema": "trn21-r4-final-transfer-supervisor-assignment-v1",
        "issuer_role": "supervisor",
        "dispatch_authorized": True,
        "assignment_id": ASSIGNMENT_ID,
        "scope": SCOPE,
        "host": "trn21",
        "hostname": "ip-172-31-34-85",
        "endpoint": core.transport.ENDPOINT,
        "request_sha256": binding["request_sha256"],
        "final_independent_review_sha256": final_review_sha256,
        "final_wrapper_sha256": binding["final_wrapper_sha256"],
        "disabled_request_sha256": DISABLED_REQUEST_SHA256,
        "disabled_core_wrapper_sha256": CORE_SHA256,
        "transport_sha256": TRANSPORT_SHA256,
        "remote_sha256": REMOTE_SHA256,
        "final_transport_sha256": FINAL_TRANSPORT_SHA256,
        "final_remote_sha256": FINAL_REMOTE_SHA256,
        "disabled_static_review_sha256_not_final_authority": STATIC_REVIEW_SHA256,
        "source_stage_hold_contract_sha256": core.TRN21_HOLD_PLAN_SHA256,
        "source_stage_hold_runbook_sha256": SOURCE_STAGE_HOLD_RUNBOOK_SHA256,
        "local_capsule_review_sha256": core.CAPSULE_REVIEW_SHA256,
        "source_commit": SOURCE_COMMIT,
        "capsule_source_outside_git": str(core.transport.SOURCE),
        "capsule_bytes": core.transport.SOURCE_BYTES,
        "capsule_sha256": core.transport.SOURCE_SHA256,
        "target": core.transport.TARGET,
        "one_shot_marker": str(MARKER),
        "journal": str(core.JOURNAL),
        "one_attempt": True,
        "same_path_overwrite_delete_rename_or_retry_authorized": False,
        "source_checkout_authorized": False,
        "cpu_g2_authorized": False,
        "native_run_authorized": False,
        "eval_authorized": False,
        "monitor_operation_authorized": False,
        "cleanup_authorized": False,
        "submission_authorized": False,
    }


def _validate_authority_files(binding, review_bytes, assignment_bytes):
    try:
        review = json.loads(review_bytes)
        assignment = json.loads(assignment_bytes)
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("FINAL AUTHORITY HOLD: malformed review or assignment") from error
    if review != _expected_final_review(binding):
        raise RuntimeError("FINAL REVIEW HOLD: actual independent file is not exact PASS")
    review_sha = core._sha(review_bytes)
    if assignment != _expected_assignment(binding, review_sha):
        raise RuntimeError("SUPERVISOR ASSIGNMENT HOLD: exact review/request/marker binding mismatch")
    return {**binding, "final_review_sha256": review_sha,
            "supervisor_assignment_sha256": core._sha(assignment_bytes)}


def _read_required(path, purpose):
    try:
        return core._read_evidence(path)
    except FileNotFoundError as error:
        raise RuntimeError(purpose + " HOLD: exact fixed-path file is absent") from error


def authorize_final():
    """Require two future actual files; neither exists in this package."""
    request, binding = _read_frozen_request_and_code()
    if request["dispatch_authorized"] is not True:
        raise RuntimeError("DISPATCH HOLD: frozen final request is disabled")
    review_bytes = _read_required(FINAL_REVIEW, "FINAL REVIEW")
    assignment_bytes = _read_required(ASSIGNMENT, "SUPERVISOR ASSIGNMENT")
    authority = _validate_authority_files(binding, review_bytes, assignment_bytes)
    core.transport.checked_ssh_identity()  # Local host-key proof only; no SSH.
    return authority


def _consume_fixed_marker(authority):
    """Durably consume this exact assignment; a failed attempt keeps the file."""
    parent_fd = core._open_fixed_parent(MARKER)
    fd = None
    try:
        fd = os.open(MARKER.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent_fd)
        os.fchmod(fd, 0o600)
        payload = core._json_bytes({"schema": "trn21-r4-final-transfer-one-shot-marker-v1",
                                    "utc": core._utc(), "assignment_id": ASSIGNMENT_ID,
                                    "host": "trn21", "scope": SCOPE,
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


def _pinned_marker(digest):
    named = MARKER.lstat()
    if (not stat.S_ISREG(named.st_mode) or named.st_uid != os.getuid() or
            named.st_nlink != 1 or stat.S_IMODE(named.st_mode) != 0o600):
        raise RuntimeError("final one-shot marker metadata changed")
    core._pinned(MARKER, digest)


def _run_ssh_final(operation, journal, *, preflight=None):
    """Use the exact final payload while preserving the reviewed raw journal."""
    argv = final_transport.build_ssh_argv(operation, preflight=preflight)
    started = core._utc()
    source_fd = core._open_checked_source_fd() if operation == "write" else None
    proc = None
    timed_out = False
    try:
        if operation == "write":
            core._check_controller_breakers()
        try:
            proc = subprocess.Popen(argv, stdin=source_fd if source_fd is not None
                                    else subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    shell=False, close_fds=True)
        except BaseException as error:
            journal.emit("RAW_SSH_TRANSPORT", operation=operation, argv=argv,
                         argv_sha256=final_transport.argv_sha256(argv),
                         local_ssh_pid=None, local_ssh_start_ticks=None,
                         local_ssh_birth_error=None, started_utc=started,
                         finished_utc=core._utc(), exit_code=None, timed_out=False,
                         stdout_bytes=0, stdout_sha256=core._sha(b""), stdout_b64="",
                         stderr_bytes=0, stderr_sha256=core._sha(b""), stderr_b64="",
                         spawn_error=repr(error))
            raise
        birth_error = None
        try:
            local_start_ticks = core._local_ssh_birth_ticks(proc.pid)
        except (OSError, ValueError, IndexError) as error:
            local_start_ticks = None
            birth_error = repr(error)
        try:
            stdout, stderr = proc.communicate(timeout=600 if operation == "write" else 300)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()  # Local SSH only; never clean a partial remote target.
            stdout, stderr = proc.communicate()
        journal.emit("RAW_SSH_TRANSPORT", operation=operation, argv=argv,
                     argv_sha256=final_transport.argv_sha256(argv),
                     local_ssh_pid=proc.pid, local_ssh_start_ticks=local_start_ticks,
                     local_ssh_birth_error=birth_error, started_utc=started,
                     finished_utc=core._utc(), exit_code=proc.returncode,
                     timed_out=timed_out, stdout_bytes=len(stdout),
                     stdout_sha256=core._sha(stdout),
                     stdout_b64=base64.b64encode(stdout).decode("ascii"),
                     stderr_bytes=len(stderr), stderr_sha256=core._sha(stderr),
                     stderr_b64=base64.b64encode(stderr).decode("ascii"))
        if local_start_ticks is None:
            raise RuntimeError("SSH process birth identity unavailable; raw transport archived")
        if timed_out or proc.returncode != 0:
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
    """One transfer attempt only after independent final PASS and assignment."""
    authority = authorize_final()  # Absent actual files stop before marker or SSH.
    source = final_transport.checked_local_source()
    ssh_identity = final_transport.checked_ssh_identity()
    first_breakers = core._check_controller_breakers()
    marker_sha = _consume_fixed_marker(authority)
    journal = core.Journal()  # Separate fixed O_EXCL journal, fsynced before SSH.
    pre = None
    persistence = None
    failure = None
    post_failure = None
    try:
        journal.emit("FINAL_ONE_ATTEMPT_CONSUMED", authority=authority,
                     marker_sha256=marker_sha, source=source,
                     ssh_identity=ssh_identity, controller=first_breakers)
        try:
            pre = _run_ssh_final("probe", journal)
            final_transport.validate_probe(pre, before_write=True)
            journal.emit("FRESH_PREFLIGHT_PASS", old_inventory=pre["old_inventory"])
            if authorize_final() != authority:
                raise RuntimeError("final review or assignment drifted after preflight")
            _pinned_marker(marker_sha)
            core._check_controller_breakers()
            if authorize_final() != authority:
                raise RuntimeError("final authority drifted immediately before write")
            written = _run_ssh_final("write", journal, preflight=pre)
            continuity = final_transport.compare_preflight_to_write(pre, written)
            journal.emit("PREFLIGHT_TO_WRITE_CONTINUITY_PASS", result=continuity)
            verified = _run_ssh_final("verify", journal)  # Distinct SSH process.
            persistence = final_transport.compare_sessions(pre, written, verified)
            journal.emit("CROSS_SESSION_PERSISTENCE_PASS", result=persistence)
        except BaseException as error:
            failure = repr(error)
            journal.emit("TRANSFER_HOLD", error=failure)
        finally:
            # Full, distinct, read-only probe even after failed preflight/write.
            try:
                post = _run_ssh_final("probe", journal)
                final_transport.validate_probe(post, before_write=False)
                comparison = (final_transport.compare_postflight(pre, post,
                                                                persistence=persistence)
                              if pre is not None else None)
                journal.emit("FINAL_READ_ONLY_POSTFLIGHT", snapshot=post,
                             comparison=comparison, compared_with_preflight=pre is not None)
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
