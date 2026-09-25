"""Exact trn22 r4 transfer-only request; no final review or assignment exists.

An enabled request records intent, never standalone authority. This wrapper
cannot create its one-shot marker or open SSH until a different Verifier's
actual fixed-path final PASS and a separate Supervisor one-attempt assignment
bind these exact bytes. The prior disabled repair and static review remain
unchanged. No checkout, CPU G2, native training, evaluation or cleanup exists.
"""

import json
from pathlib import Path
import sys

import operator_transfer_once as core
import operator_transfer_final_once as marker_support


HERE = Path(__file__).resolve().parent
REQUEST = HERE / "TRANSFER_REQUEST.CONTINUITY_FINAL_R2.json"
STATIC_REVIEW = HERE / "INDEPENDENT_CONTINUITY_REPAIR_STATIC_REVIEW.md"
DISABLED_CORE_REQUEST = HERE / "TRANSFER_REQUEST.HOLD.json"
DISABLED_FINAL_REQUEST = HERE / "TRANSFER_REQUEST.ENABLEMENT_CANDIDATE.json"
DISABLED_FINAL_WRAPPER = HERE / "operator_transfer_final_once.py"
CORE_WRAPPER = HERE / "operator_transfer_once.py"
TRANSPORT = HERE / "stage_r4_transport.py"
REMOTE = HERE / "stage_r4_remote.py"
FINAL_REVIEW = HERE / "INDEPENDENT_CONTINUITY_FINAL_R2_REVIEW.json"
ASSIGNMENT = HERE / "SUPERVISOR_CONTINUITY_TRANSFER_R2_ASSIGNMENT.json"
MARKER = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/"
              "trn22-c5c-a0-r4-transfer-final-r1-one-attempt.marker")
JOURNAL = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/"
               "trn22-c5c-a0-r4-transfer-only-one-attempt.jsonl")
REQUEST_SHA256 = "99f46c558c49ba4a404daf8908e387b7bc31deaff8ab72e26142596efddaafd5"
STATIC_REVIEW_SHA256 = "669d99837e25ee634e4db418125b1b8bf13862a1907baad10ac213b5e89c7c57"
DISABLED_CORE_REQUEST_SHA256 = "2b7249002ec301962153efdfc3b003454dff9c2ba6fe80a09b780e10adb8ab0c"
DISABLED_FINAL_REQUEST_SHA256 = "833737bbe76a7b7acc287c247616527b199b5f515a46fcf8de0190595600ee57"
DISABLED_FINAL_WRAPPER_SHA256 = "b3ad1290f5e9cc6655b57380b269b9c2e4082391ea27d4f81545388b892beb1d"
CORE_SHA256 = "7590e275f94f79ffac9098bfd3c0345d4c9cb5ce10a26624065b0e04d6189736"
TRANSPORT_SHA256 = "ea14fd19f6992400ad56e59c8dcbec620c784f616c4ff60f6d0568aa721a4ec0"
REMOTE_SHA256 = "023e578693c3fb826b81cfc645190d79e409757dba0bd8df29adad6e3f90a69d"
REPAIR_COMMIT = "d682a90b1e75bdac128938feda884484ea87a8d5"
SOURCE_COMMIT = "446b657695f086f9b484fa601c9a3b61c2495912"
ASSIGNMENT_ID = "trn22-c5c-a0-r4-transfer-final-r1"
SCOPE = "TRN22_R4_CAPSULE_TRANSFER_ONLY_ONE_ATTEMPT"


def _read_required(path, purpose):
    try:
        return core._read_evidence(path)
    except FileNotFoundError as error:
        raise RuntimeError(purpose + " HOLD: exact fixed-path file is absent") from error


def _read_frozen_request_and_code():
    """Pin real static-PASS and code bytes before reading future authority."""
    request_bytes = core._pinned(REQUEST, REQUEST_SHA256)
    request = json.loads(request_bytes)
    for path, digest in (
            (STATIC_REVIEW, STATIC_REVIEW_SHA256),
            (DISABLED_CORE_REQUEST, DISABLED_CORE_REQUEST_SHA256),
            (DISABLED_FINAL_REQUEST, DISABLED_FINAL_REQUEST_SHA256),
            (DISABLED_FINAL_WRAPPER, DISABLED_FINAL_WRAPPER_SHA256),
            (CORE_WRAPPER, CORE_SHA256),
            (TRANSPORT, TRANSPORT_SHA256),
            (REMOTE, REMOTE_SHA256),
            (core.SOURCE_CONTRACT, core.SOURCE_CONTRACT_SHA256),
            (core.SOURCE_REVIEW, core.SOURCE_REVIEW_SHA256)):
        core._pinned(path, digest)
    if (request.get("schema") !=
            "trn22-c5c-a0-r4-transfer-continuity-final-r2-request-v1" or
            request.get("status") !=
            "EXACT_TRANSFER_ONLY_REQUEST_PENDING_INDEPENDENT_FINAL_REVIEW_AND_SUPERVISOR_ASSIGNMENT" or
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
            request.get("reviewed_repair_commit") != REPAIR_COMMIT or
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
            request.get("independent_continuity_static_review_sha256") !=
            STATIC_REVIEW_SHA256 or
            request.get("reviewed_disabled_core_request_sha256") !=
            DISABLED_CORE_REQUEST_SHA256 or
            request.get("reviewed_disabled_final_request_sha256") !=
            DISABLED_FINAL_REQUEST_SHA256 or
            request.get("reviewed_disabled_final_wrapper_sha256") !=
            DISABLED_FINAL_WRAPPER_SHA256 or
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
                "final_independent_review_sha256", "supervisor_assignment_sha256")) or
            marker_support.MARKER != MARKER or
            marker_support.ASSIGNMENT_ID != ASSIGNMENT_ID or
            marker_support.SCOPE != SCOPE or
            core.JOURNAL != JOURNAL):
        raise RuntimeError("CONTINUITY FINAL REQUEST HOLD: exact scope/code binding drift")
    wrapper_sha = core._sha(core._read_evidence(HERE / Path(__file__).name))
    return request, {"request_sha256": core._sha(request_bytes),
                     "final_wrapper_sha256": wrapper_sha}


def _expected_final_review(binding):
    return {
        "schema": "trn22-r4-continuity-final-r2-independent-review-v1",
        "verdict": "PASS",
        "reviewer_role": "independent_verifier",
        "request_sha256": binding["request_sha256"],
        "final_wrapper_sha256": binding["final_wrapper_sha256"],
        "independent_continuity_static_review_sha256": STATIC_REVIEW_SHA256,
        "reviewed_repair_commit": REPAIR_COMMIT,
        "reviewed_disabled_final_wrapper_sha256": DISABLED_FINAL_WRAPPER_SHA256,
        "core_wrapper_sha256": CORE_SHA256,
        "transport_sha256": TRANSPORT_SHA256,
        "remote_sha256": REMOTE_SHA256,
        "source_stage_contract_sha256": core.SOURCE_CONTRACT_SHA256,
        "source_stage_independent_review_sha256": core.SOURCE_REVIEW_SHA256,
        "source_commit": SOURCE_COMMIT,
        "capsule_bytes": core.transport.SOURCE_BYTES,
        "capsule_sha256": core.transport.SOURCE_SHA256,
        "target": core.transport.TARGET,
        "one_shot_marker": str(MARKER),
        "journal": str(JOURNAL),
        "assignment_id": ASSIGNMENT_ID,
        "scope": SCOPE,
        "host": "trn22",
        "endpoint": core.transport.ENDPOINT,
        "no_host_action_by_reviewer": True,
    }


def _expected_assignment(binding, review_sha256):
    return {
        "schema": "trn22-r4-continuity-final-r2-supervisor-assignment-v1",
        "issuer_role": "supervisor",
        "dispatch_authorized": True,
        "request_sha256": binding["request_sha256"],
        "final_independent_review_sha256": review_sha256,
        "final_wrapper_sha256": binding["final_wrapper_sha256"],
        "independent_continuity_static_review_sha256": STATIC_REVIEW_SHA256,
        "reviewed_repair_commit": REPAIR_COMMIT,
        "reviewed_disabled_final_wrapper_sha256": DISABLED_FINAL_WRAPPER_SHA256,
        "core_wrapper_sha256": CORE_SHA256,
        "transport_sha256": TRANSPORT_SHA256,
        "remote_sha256": REMOTE_SHA256,
        "source_stage_contract_sha256": core.SOURCE_CONTRACT_SHA256,
        "source_stage_independent_review_sha256": core.SOURCE_REVIEW_SHA256,
        "source_commit": SOURCE_COMMIT,
        "capsule_bytes": core.transport.SOURCE_BYTES,
        "capsule_sha256": core.transport.SOURCE_SHA256,
        "target": core.transport.TARGET,
        "one_shot_marker": str(MARKER),
        "journal": str(JOURNAL),
        "assignment_id": ASSIGNMENT_ID,
        "scope": SCOPE,
        "host": "trn22",
        "endpoint": core.transport.ENDPOINT,
        "one_attempt": True,
        "source_checkout_authorized": False,
        "cpu_g2_authorized": False,
        "native_run_authorized": False,
        "eval_authorized": False,
        "monitor_operation_authorized": False,
        "submission_authorized": False,
    }


def _validate_authority_files(binding, review_bytes, assignment_bytes):
    """Exact actual-file review -> assignment hash chain; no role self-claim."""
    try:
        review = json.loads(review_bytes)
        assignment = json.loads(assignment_bytes)
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("CONTINUITY FINAL AUTHORITY HOLD: malformed review/assignment") from error
    if (review != _expected_final_review(binding) or
            review.get("no_host_action_by_reviewer") is not True):
        raise RuntimeError("CONTINUITY FINAL REVIEW HOLD: actual file is not exact PASS")
    review_sha = core._sha(review_bytes)
    if (assignment != _expected_assignment(binding, review_sha) or
            assignment.get("dispatch_authorized") is not True or
            assignment.get("one_attempt") is not True or
            any(assignment.get(key) is not False for key in (
                "source_checkout_authorized", "cpu_g2_authorized",
                "native_run_authorized", "eval_authorized",
                "monitor_operation_authorized", "submission_authorized"))):
        raise RuntimeError("CONTINUITY SUPERVISOR HOLD: exact review/assignment binding mismatch")
    return {**binding, "final_review_sha256": review_sha,
            "supervisor_assignment_sha256": core._sha(assignment_bytes)}


def authorize_final():
    """No marker/journal/SSH until both future fixed-path authority files pass."""
    _request, binding = _read_frozen_request_and_code()
    review_bytes = _read_required(FINAL_REVIEW, "FINAL REVIEW")
    assignment_bytes = _read_required(ASSIGNMENT, "SUPERVISOR ASSIGNMENT")
    authority = _validate_authority_files(binding, review_bytes, assignment_bytes)
    core.transport.checked_ssh_identity()  # Local host-key proof, not SSH.
    return authority


def run_once():
    """One transfer-only attempt after independent PASS and Supervisor binding."""
    authority = authorize_final()
    source = core.transport.checked_local_source()
    initial_breakers = core._check_controller_breakers()
    marker_sha = marker_support._consume_fixed_marker(authority)
    journal = core.Journal()
    pre = None
    persistence = None
    failure = None
    post_failure = None
    try:
        try:
            journal.emit("FINAL_ONE_ATTEMPT_CONSUMED", authority=authority,
                         marker_sha256=marker_sha, source=source,
                         controller=initial_breakers)
            pre = core._run_ssh("probe", journal)
            core.transport.validate_probe(pre, before_write=True)
            journal.emit("FRESH_PREFLIGHT_PASS", old_inventory=pre["old_inventory"])
            if authorize_final() != authority:
                raise RuntimeError("actual final review or assignment drifted after preflight")
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
            # Even an unsuccessful write or preflight gets a distinct read-only probe.
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
        raise SystemExit("FINAL AUTHORITY HOLD: no alternate host, target or operation")
    run_once()
