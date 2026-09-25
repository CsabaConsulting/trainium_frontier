"""Non-dispatchable trn22 r4 continuity-r3 transfer repair inspection.

This file only checks local, exact-byte bindings. It has no enablement path:
even a future review or Supervisor assignment cannot make run_once contact a
host. A separately named, independently reviewed successor is required.
"""

import json
from pathlib import Path
import sys

import operator_transfer_once as core
import stage_r4_transport as transport


HERE = Path(__file__).resolve().parent
REQUEST = HERE / "TRANSFER_REQUEST.CONTINUITY_R3_HOLD.json"
HISTORICAL_STATIC_PASS = HERE / "INDEPENDENT_CONTINUITY_REPAIR_STATIC_REVIEW.md"
HISTORICAL_FINAL_HOLD = HERE / "INDEPENDENT_CONTINUITY_FINAL_R2_HOLD.md"
HISTORICAL_FINAL_REQUEST = HERE / "TRANSFER_REQUEST.CONTINUITY_FINAL_R2.json"
HISTORICAL_FINAL_WRAPPER = HERE / "operator_transfer_continuity_final_r2.py"
CORE = HERE / "operator_transfer_once.py"
TRANSPORT = HERE / "stage_r4_transport.py"
REMOTE = HERE / "stage_r4_remote.py"

REQUEST_SHA256 = "247c4eb25e9667276e83ecaba5cb570ef2d400464185b54be073e640dd773fa1"
HISTORICAL_STATIC_PASS_SHA256 = "669d99837e25ee634e4db418125b1b8bf13862a1907baad10ac213b5e89c7c57"
HISTORICAL_FINAL_HOLD_SHA256 = "873bf131898e014adbb79a9cb6dd5cefd5f653897f8c80ae63b13e296f1e38f1"
HISTORICAL_FINAL_REQUEST_SHA256 = "99f46c558c49ba4a404daf8908e387b7bc31deaff8ab72e26142596efddaafd5"
HISTORICAL_FINAL_WRAPPER_SHA256 = "14dac3c538d3d08e2876c1327e177f2a4c4696cfadb9884004725305383ccf23"
CORE_SHA256 = "7590e275f94f79ffac9098bfd3c0345d4c9cb5ce10a26624065b0e04d6189736"
TRANSPORT_SHA256 = "1fa3afb41dfbf8288e53c4aa43dd16b348013416e708d8432e97364b31556b51"
REMOTE_SHA256 = "e93efd99c0a58aa32537d945ddc8fc29f9d7ce0dabb89ed93526da30421f4086"
SOURCE_COMMIT = "446b657695f086f9b484fa601c9a3b61c2495912"
MARKER = "/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r4-transfer-final-r1-one-attempt.marker"
JOURNAL = "/home/poojith-devan/Desktop/Trainium-source-artifacts/trn22-c5c-a0-r4-transfer-only-one-attempt.jsonl"


def inspect_hold():
    """Read-only local consistency check; historical reviews confer no authority."""
    request_bytes = core._pinned(REQUEST, REQUEST_SHA256)
    for path, expected in (
            (HISTORICAL_STATIC_PASS, HISTORICAL_STATIC_PASS_SHA256),
            (HISTORICAL_FINAL_HOLD, HISTORICAL_FINAL_HOLD_SHA256),
            (HISTORICAL_FINAL_REQUEST, HISTORICAL_FINAL_REQUEST_SHA256),
            (HISTORICAL_FINAL_WRAPPER, HISTORICAL_FINAL_WRAPPER_SHA256),
            (CORE, CORE_SHA256), (TRANSPORT, TRANSPORT_SHA256),
            (REMOTE, REMOTE_SHA256),
            (core.SOURCE_CONTRACT, core.SOURCE_CONTRACT_SHA256),
            (core.SOURCE_REVIEW, core.SOURCE_REVIEW_SHA256)):
        core._pinned(path, expected)
    request = json.loads(request_bytes)
    remote = transport.remote_constants()  # Pinned local source; never SSH.
    if (request.get("schema") != "trn22-c5c-a0-r4-transfer-continuity-r3-hold-v1" or
            request.get("status") !=
            "DISPATCH_HOLD_PENDING_NEW_INDEPENDENT_STATIC_AND_FINAL_REVIEWS_AND_SUPERVISOR_ASSIGNMENT" or
            request.get("dispatch_authorized") is not False or
            request.get("prepared_by_role") != "implementer" or
            request.get("host") != "trn22" or
            request.get("source_commit") != SOURCE_COMMIT or
            request.get("capsule_source_outside_git") != str(transport.SOURCE) or
            request.get("capsule_bytes") != transport.SOURCE_BYTES or
            request.get("capsule_sha256") != transport.SOURCE_SHA256 or
            request.get("target") != transport.TARGET or
            request.get("one_shot_marker") != MARKER or
            request.get("journal") != JOURNAL or
            request.get("source_stage_contract_sha256") != core.SOURCE_CONTRACT_SHA256 or
            request.get("source_stage_independent_review_sha256") != core.SOURCE_REVIEW_SHA256 or
            request.get("historical_continuity_static_pass_sha256_not_authority") !=
            HISTORICAL_STATIC_PASS_SHA256 or
            request.get("historical_final_r2_hold_sha256") != HISTORICAL_FINAL_HOLD_SHA256 or
            request.get("historical_final_r2_request_sha256") !=
            HISTORICAL_FINAL_REQUEST_SHA256 or
            request.get("historical_final_r2_wrapper_sha256") !=
            HISTORICAL_FINAL_WRAPPER_SHA256 or
            request.get("core_wrapper_sha256") != CORE_SHA256 or
            request.get("transport_sha256") != TRANSPORT_SHA256 or
            request.get("remote_sha256") != REMOTE_SHA256 or
            transport.REMOTE_SHA256 != REMOTE_SHA256 or
            remote.SHA256 != transport.SOURCE_SHA256 or
            remote.TARGET != transport.TARGET or
            request.get("ssh_identity") != {
                "known_hosts_sha256": transport.KNOWN_HOSTS_SHA256,
                "host_fingerprint": transport.HOST_FINGERPRINT,
                "host_key_algorithm": "ssh-ed25519", "config_file": "/dev/null"} or
            any(request.get(key) is not True for key in (
                "file_backed_old_git_inventory_classification_required",
                "fresh_read_only_preflight_required",
                "preflight_to_write_identity_binding_required",
                "final_live_pre_o_excl_neuron_lock_label_stop_process_required",
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
                "independent_final_review_sha256", "supervisor_assignment_sha256"))):
        raise RuntimeError("CONTINUITY R3 DISPATCH HOLD: exact scope or SHA drift")
    return {"request_sha256": core._sha(request_bytes),
            "wrapper_sha256": core._sha(core._read_evidence(Path(__file__).resolve())),
            "transport_sha256": TRANSPORT_SHA256,
            "remote_sha256": REMOTE_SHA256,
            "dispatch_authorized": False}


def run_once():
    inspect_hold()
    raise RuntimeError("CONTINUITY R3 DISPATCH HOLD: new independent reviews and a "
                       "separate enabled successor are required; no SSH is permitted")


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("CONTINUITY R3 DISPATCH HOLD: no alternate operation or target")
    run_once()
