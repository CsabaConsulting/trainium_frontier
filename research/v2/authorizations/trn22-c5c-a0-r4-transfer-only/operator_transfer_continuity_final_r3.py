"""Exact trn22 r4 continuity-r3 transfer-only final request.

Request intent is not authority. No marker, journal or SSH is reachable until
an independent final PASS at the fixed path and a distinct Supervisor
one-attempt assignment bind this request, this wrapper and the actual reviewed
r3 bytes. There is no checkout, G2, native run, evaluation or cleanup path.
"""

import json
from pathlib import Path
import sys

import operator_transfer_once as core
import operator_transfer_final_once as marker_support


HERE = Path(__file__).resolve().parent
REQUEST = HERE / "TRANSFER_REQUEST.CONTINUITY_FINAL_R3.json"
STATIC_REVIEW = HERE / "INDEPENDENT_CONTINUITY_R3_STATIC_REVIEW.md"
DISABLED_R3_REQUEST = HERE / "TRANSFER_REQUEST.CONTINUITY_R3_HOLD.json"
DISABLED_R3_WRAPPER = HERE / "operator_transfer_continuity_r3_hold.py"
MARKER_HELPER = HERE / "operator_transfer_final_once.py"
HISTORICAL_FINAL_R2_HOLD = HERE / "INDEPENDENT_CONTINUITY_FINAL_R2_HOLD.md"
CORE_WRAPPER = HERE / "operator_transfer_once.py"
TRANSPORT = HERE / "stage_r4_transport.py"
REMOTE = HERE / "stage_r4_remote.py"
FINAL_REVIEW = HERE / "INDEPENDENT_CONTINUITY_FINAL_R3_REVIEW.json"
ASSIGNMENT = HERE / "SUPERVISOR_CONTINUITY_TRANSFER_R3_ASSIGNMENT.json"
MARKER = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/"
              "trn22-c5c-a0-r4-transfer-final-r1-one-attempt.marker")
JOURNAL = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/"
               "trn22-c5c-a0-r4-transfer-only-one-attempt.jsonl")

REQUEST_SHA256 = "a2b1ad44023f32c010c8d169e25986830fa60a26d0162d281a4c0154b447e7a0"
STATIC_REVIEW_SHA256 = "f71e81cb1ac079593ccc5f0e2b06a3436ce0a1c9d45d6850ac46487a3eed3fed"
DISABLED_R3_REQUEST_SHA256 = "247c4eb25e9667276e83ecaba5cb570ef2d400464185b54be073e640dd773fa1"
DISABLED_R3_WRAPPER_SHA256 = "6160246baba29e525545299bb28db9a4310873b4033926e72655dc382f7358a5"
MARKER_HELPER_SHA256 = "b3ad1290f5e9cc6655b57380b269b9c2e4082391ea27d4f81545388b892beb1d"
HISTORICAL_FINAL_R2_HOLD_SHA256 = "873bf131898e014adbb79a9cb6dd5cefd5f653897f8c80ae63b13e296f1e38f1"
CORE_SHA256 = "7590e275f94f79ffac9098bfd3c0345d4c9cb5ce10a26624065b0e04d6189736"
TRANSPORT_SHA256 = "1fa3afb41dfbf8288e53c4aa43dd16b348013416e708d8432e97364b31556b51"
REMOTE_SHA256 = "e93efd99c0a58aa32537d945ddc8fc29f9d7ce0dabb89ed93526da30421f4086"
R3_IMPLEMENTER_COMMIT = "76485b9623d84351433cc3d2cd350f830c624d1e"
R3_CANONICAL_CODE_COMMIT = "e21135cc84fa941784e6a52fa142dadd0973de3f"
SOURCE_COMMIT = "446b657695f086f9b484fa601c9a3b61c2495912"
ASSIGNMENT_ID = "trn22-c5c-a0-r4-transfer-final-r1"
SCOPE = "TRN22_R4_CAPSULE_TRANSFER_ONLY_ONE_ATTEMPT"


def _read_required(path, purpose):
    try:
        return core._read_evidence(path)
    except FileNotFoundError as error:
        raise RuntimeError(purpose + " HOLD: exact fixed-path file is absent") from error


def _read_frozen_request_and_code():
    """Pin actual r3 STATIC PASS and code, never a prior review claim."""
    request_bytes = core._pinned(REQUEST, REQUEST_SHA256)
    request = json.loads(request_bytes)
    for path, digest in (
            (STATIC_REVIEW, STATIC_REVIEW_SHA256),
            (DISABLED_R3_REQUEST, DISABLED_R3_REQUEST_SHA256),
            (DISABLED_R3_WRAPPER, DISABLED_R3_WRAPPER_SHA256),
            (MARKER_HELPER, MARKER_HELPER_SHA256),
            (HISTORICAL_FINAL_R2_HOLD, HISTORICAL_FINAL_R2_HOLD_SHA256),
            (CORE_WRAPPER, CORE_SHA256), (TRANSPORT, TRANSPORT_SHA256),
            (REMOTE, REMOTE_SHA256),
            (core.SOURCE_CONTRACT, core.SOURCE_CONTRACT_SHA256),
            (core.SOURCE_REVIEW, core.SOURCE_REVIEW_SHA256)):
        core._pinned(path, digest)
    if (request.get("schema") !=
            "trn22-c5c-a0-r4-transfer-continuity-final-r3-request-v1" or
            request.get("status") !=
            "EXACT_TRANSFER_ONLY_R3_REQUEST_PENDING_INDEPENDENT_FINAL_REVIEW_AND_SUPERVISOR_ASSIGNMENT" or
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
            request.get("reviewed_r3_implementer_commit") != R3_IMPLEMENTER_COMMIT or
            request.get("canonical_r3_code_commit") != R3_CANONICAL_CODE_COMMIT or
            request.get("source_commit") != SOURCE_COMMIT or
            request.get("capsule_source_outside_git") != str(core.transport.SOURCE) or
            request.get("capsule_bytes") != core.transport.SOURCE_BYTES or
            request.get("capsule_sha256") != core.transport.SOURCE_SHA256 or
            request.get("target") != core.transport.TARGET or
            request.get("one_shot_marker") != str(MARKER) or
            request.get("journal") != str(JOURNAL) or
            request.get("source_stage_contract_sha256") != core.SOURCE_CONTRACT_SHA256 or
            request.get("source_stage_independent_review_sha256") !=
            core.SOURCE_REVIEW_SHA256 or
            request.get("independent_r3_static_review_sha256") !=
            STATIC_REVIEW_SHA256 or
            request.get("reviewed_disabled_r3_request_sha256") !=
            DISABLED_R3_REQUEST_SHA256 or
            request.get("reviewed_disabled_r3_wrapper_sha256") !=
            DISABLED_R3_WRAPPER_SHA256 or
            request.get("reviewed_marker_helper_sha256") !=
            MARKER_HELPER_SHA256 or
            request.get("historical_final_r2_hold_sha256_not_authority") !=
            HISTORICAL_FINAL_R2_HOLD_SHA256 or
            request.get("core_wrapper_sha256") != CORE_SHA256 or
            request.get("transport_sha256") != TRANSPORT_SHA256 or
            request.get("remote_sha256") != REMOTE_SHA256 or
            core.transport.REMOTE_SHA256 != REMOTE_SHA256 or
            request.get("ssh_identity") != {
                "known_hosts_sha256": core.transport.KNOWN_HOSTS_SHA256,
                "host_fingerprint": core.transport.HOST_FINGERPRINT,
                "host_key_algorithm": "ssh-ed25519", "config_file": "/dev/null"} or
            any(request.get(key) is not True for key in (
                "required_fresh_read_only_preflight",
                "preflight_to_write_identity_binding_required",
                "complete_old_file_backed_git_inventory_required",
                "final_live_pre_o_excl_neuron_lock_label_stop_process_required",
                "local_stop_and_breaker_gates_before_marker_and_write",
                "raw_ssh_transport_journal_required",
                "distinct_ssh_readback_required",
                "final_read_only_postflight_even_on_failure_required",
                "one_attempt")) or
            any(request.get(key) is not False for key in (
                "same_path_overwrite_delete_rename_or_retry_authorized",
                "source_checkout_authorized", "cpu_g2_authorized",
                "native_run_authorized", "eval_authorized",
                "monitor_operation_authorized", "submission_authorized")) or
            any(key in request for key in (
                "final_independent_review_sha256", "supervisor_assignment_sha256")) or
            marker_support.MARKER != MARKER or
            marker_support.ASSIGNMENT_ID != ASSIGNMENT_ID or
            marker_support.SCOPE != SCOPE or
            core.JOURNAL != JOURNAL):
        raise RuntimeError("CONTINUITY FINAL R3 REQUEST HOLD: exact scope/code binding drift")
    wrapper_sha = core._sha(core._read_evidence(Path(__file__).resolve()))
    return request, {"request_sha256": core._sha(request_bytes),
                     "final_wrapper_sha256": wrapper_sha}


def _expected_final_review(binding):
    return {
        "schema": "trn22-r4-continuity-final-r3-independent-review-v1",
        "verdict": "PASS", "reviewer_role": "independent_verifier",
        "request_sha256": binding["request_sha256"],
        "final_wrapper_sha256": binding["final_wrapper_sha256"],
        "independent_r3_static_review_sha256": STATIC_REVIEW_SHA256,
        "reviewed_r3_implementer_commit": R3_IMPLEMENTER_COMMIT,
        "canonical_r3_code_commit": R3_CANONICAL_CODE_COMMIT,
        "reviewed_disabled_r3_request_sha256": DISABLED_R3_REQUEST_SHA256,
        "reviewed_disabled_r3_wrapper_sha256": DISABLED_R3_WRAPPER_SHA256,
        "reviewed_marker_helper_sha256": MARKER_HELPER_SHA256,
        "historical_final_r2_hold_sha256_not_authority": HISTORICAL_FINAL_R2_HOLD_SHA256,
        "core_wrapper_sha256": CORE_SHA256,
        "transport_sha256": TRANSPORT_SHA256,
        "remote_sha256": REMOTE_SHA256,
        "source_stage_contract_sha256": core.SOURCE_CONTRACT_SHA256,
        "source_stage_independent_review_sha256": core.SOURCE_REVIEW_SHA256,
        "source_commit": SOURCE_COMMIT,
        "capsule_bytes": core.transport.SOURCE_BYTES,
        "capsule_sha256": core.transport.SOURCE_SHA256,
        "target": core.transport.TARGET,
        "one_shot_marker": str(MARKER), "journal": str(JOURNAL),
        "assignment_id": ASSIGNMENT_ID, "scope": SCOPE,
        "host": "trn22", "endpoint": core.transport.ENDPOINT,
        "no_host_action_by_reviewer": True,
    }


def _expected_assignment(binding, review_sha256):
    return {
        "schema": "trn22-r4-continuity-final-r3-supervisor-assignment-v1",
        "issuer_role": "supervisor", "dispatch_authorized": True,
        "request_sha256": binding["request_sha256"],
        "final_independent_review_sha256": review_sha256,
        "final_wrapper_sha256": binding["final_wrapper_sha256"],
        "independent_r3_static_review_sha256": STATIC_REVIEW_SHA256,
        "reviewed_r3_implementer_commit": R3_IMPLEMENTER_COMMIT,
        "canonical_r3_code_commit": R3_CANONICAL_CODE_COMMIT,
        "reviewed_disabled_r3_request_sha256": DISABLED_R3_REQUEST_SHA256,
        "reviewed_disabled_r3_wrapper_sha256": DISABLED_R3_WRAPPER_SHA256,
        "reviewed_marker_helper_sha256": MARKER_HELPER_SHA256,
        "historical_final_r2_hold_sha256_not_authority": HISTORICAL_FINAL_R2_HOLD_SHA256,
        "core_wrapper_sha256": CORE_SHA256,
        "transport_sha256": TRANSPORT_SHA256,
        "remote_sha256": REMOTE_SHA256,
        "source_stage_contract_sha256": core.SOURCE_CONTRACT_SHA256,
        "source_stage_independent_review_sha256": core.SOURCE_REVIEW_SHA256,
        "source_commit": SOURCE_COMMIT,
        "capsule_bytes": core.transport.SOURCE_BYTES,
        "capsule_sha256": core.transport.SOURCE_SHA256,
        "target": core.transport.TARGET,
        "one_shot_marker": str(MARKER), "journal": str(JOURNAL),
        "assignment_id": ASSIGNMENT_ID, "scope": SCOPE,
        "host": "trn22", "endpoint": core.transport.ENDPOINT,
        "one_attempt": True,
        "source_checkout_authorized": False, "cpu_g2_authorized": False,
        "native_run_authorized": False, "eval_authorized": False,
        "monitor_operation_authorized": False, "submission_authorized": False,
    }


def _validate_authority_files(binding, review_bytes, assignment_bytes):
    """Require actual fixed-path final PASS bound by the Supervisor's bytes."""
    try:
        review = json.loads(review_bytes)
        assignment = json.loads(assignment_bytes)
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("CONTINUITY FINAL R3 AUTHORITY HOLD: malformed files") from error
    if review != _expected_final_review(binding):
        raise RuntimeError("CONTINUITY FINAL R3 REVIEW HOLD: actual file is not exact PASS")
    review_sha = core._sha(review_bytes)
    if assignment != _expected_assignment(binding, review_sha):
        raise RuntimeError("CONTINUITY FINAL R3 SUPERVISOR HOLD: assignment binding mismatch")
    return {**binding, "final_review_sha256": review_sha,
            "supervisor_assignment_sha256": core._sha(assignment_bytes)}


def authorize_final():
    """No marker, journal or SSH until both *future* fixed files are valid."""
    _request, binding = _read_frozen_request_and_code()
    review_bytes = _read_required(FINAL_REVIEW, "FINAL R3 REVIEW")
    assignment_bytes = _read_required(ASSIGNMENT, "SUPERVISOR R3 ASSIGNMENT")
    authority = _validate_authority_files(binding, review_bytes, assignment_bytes)
    core.transport.checked_ssh_identity()  # Pinned local host-key metadata; not SSH.
    return authority


def run_once():
    """Exactly one transfer-only attempt after separate final authorization."""
    authority = authorize_final()
    source = core.transport.checked_local_source()
    initial_breakers = core._check_controller_breakers()
    if authorize_final() != authority:
        raise RuntimeError("actual final r3 review or assignment drifted before marker")
    marker_sha = marker_support._consume_fixed_marker(authority)
    journal = core.Journal()
    pre = None
    persistence = None
    failure = None
    post_failure = None
    try:
        try:
            journal.emit("FINAL_R3_ONE_ATTEMPT_CONSUMED", authority=authority,
                         marker_sha256=marker_sha, source=source,
                         controller=initial_breakers)
            pre = core._run_ssh("probe", journal)
            core.transport.validate_probe(pre, before_write=True)
            journal.emit("FRESH_PREFLIGHT_PASS", old_inventory=pre["old_inventory"])
            if authorize_final() != authority:
                raise RuntimeError("actual final r3 review or assignment drifted after preflight")
            core._pinned(MARKER, marker_sha)
            core._check_controller_breakers()
            written = core._run_ssh("write", journal, preflight=pre)
            continuity = core.transport.compare_preflight_write(pre, written)
            journal.emit("PREFLIGHT_TO_WRITE_CONTINUITY_PASS", result=continuity)
            verified = core._run_ssh("verify", journal)
            persistence = core.transport.compare_sessions(written, verified,
                                                          preflight=pre)
            journal.emit("CROSS_SESSION_PERSISTENCE_PASS", result=persistence)
        except BaseException as error:
            failure = repr(error)
            journal.emit("TRANSFER_HOLD", error=failure)
        finally:
            # Preserve partial targets; always attempt a distinct read-only probe.
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
        raise SystemExit("FINAL R3 AUTHORITY HOLD: no alternate host or operation")
    run_once()
