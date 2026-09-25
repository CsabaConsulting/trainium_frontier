"""Conditional trn21 r4 source-transfer request atop reviewed disabled r5.

The request's intent flags are not dispatch authority by themselves. The
distinct exact independent final PASS and Supervisor one-shot assignment must
both exist at fixed paths and match this subject before its marker, journal,
or SSH can be touched. No checkout, CPU G2, native launch or evaluation exists
here. The reviewed disabled r5 and all earlier files remain byte-identical.
"""

import base64
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import operator_transfer_once as core
import operator_transfer_final_r3_once as prior_r3
import operator_transfer_final_r4_once as prior_r4
import operator_transfer_final_r5_once as prior_r5
import stage_r4_transport_final_r5 as final_transport


HERE = Path(__file__).resolve().parent
REQUEST = HERE / "TRANSFER_REQUEST.FINAL_R5_ENABLED_PENDING_REVIEW.json"
ENABLED_TESTS = HERE / "test_operator_transfer_final_r5_enabled_once.py"
ENABLED_RUNBOOK = HERE / "FINAL_R5_ENABLED_TRANSFER_ONLY_PENDING_REVIEW_RUNBOOK.md"
FINAL_REVIEW = HERE / "INDEPENDENT_FINAL_R5_ENABLED_REQUEST_REVIEW.json"
ASSIGNMENT = HERE / "SUPERVISOR_FINAL_R5_ENABLED_TRANSFER_ONLY_ASSIGNMENT.json"
DISABLED_REQUEST = HERE / "TRANSFER_REQUEST.HOLD.json"
CORE_WRAPPER = HERE / "operator_transfer_once.py"
TRANSPORT = HERE / "stage_r4_transport.py"
REMOTE = HERE / "stage_r4_remote.py"
FINAL_TRANSPORT = HERE / "stage_r4_transport_final_r5.py"
FINAL_REMOTE = HERE / "stage_r4_remote_final_r4.py"
PRIOR_R5_REQUEST = HERE / "TRANSFER_REQUEST.FINAL_R5_HOLD.json"
PRIOR_R5_WRAPPER = HERE / "operator_transfer_final_r5_once.py"
PRIOR_R5_TRANSPORT = FINAL_TRANSPORT
PRIOR_R5_REMOTE = FINAL_REMOTE
PRIOR_R5_TESTS = HERE / "test_operator_transfer_final_r5_once.py"
PRIOR_R5_RUNBOOK = HERE / "FINAL_R5_DISABLED_INODE_FLOOR_RUNBOOK.md"
PRIOR_R5_INDEPENDENT_PASS = (core.REPO / "research/v2/verification/"
                             "20260925-trn21-r5-inode-floor-disabled-independent-static-PASS.md")
PRIOR_R5_INDEPENDENT_TESTS = (core.REPO / "research/v2/verification/"
                              "test_trn21_r5_inode_floor_independent.py")
PRIOR_R4_REQUEST = HERE / "TRANSFER_REQUEST.FINAL_R4_HOLD.json"
PRIOR_R4_WRAPPER = HERE / "operator_transfer_final_r4_once.py"
PRIOR_R4_TRANSPORT = HERE / "stage_r4_transport_final_r4.py"
PRIOR_R4_REMOTE = FINAL_REMOTE
PRIOR_R4_TESTS = HERE / "test_operator_transfer_final_r4_once.py"
PRIOR_R4_RUNBOOK = HERE / "FINAL_R4_DISABLED_RUNBOOK.md"
PRIOR_R4_INDEPENDENT_HOLD = (core.REPO / "research/v2/verification/"
                             "20260925-trn21-r4-private-lock-independent-static-HOLD.md")
PRIOR_R4_INDEPENDENT_TESTS = (core.REPO / "research/v2/verification/"
                              "test_trn21_r4_private_lock_independent.py")
PRIOR_R3_REQUEST = HERE / "TRANSFER_REQUEST.FINAL_R3_HOLD.json"
PRIOR_R3_WRAPPER = HERE / "operator_transfer_final_r3_once.py"
PRIOR_R3_TRANSPORT = HERE / "stage_r4_transport_final_r3.py"
PRIOR_R3_REMOTE = HERE / "stage_r4_remote_final_r3.py"
PRIOR_R3_TESTS = HERE / "test_operator_transfer_final_r3_once.py"
PRIOR_R3_RUNBOOK = HERE / "FINAL_R3_DISABLED_RUNBOOK.md"
PRIOR_R3_STATIC_REVIEW = HERE / "INDEPENDENT_FINAL_R3_DISABLED_STATIC_REVIEW.md"
LOCK_MODE_RECEIPT = (core.REPO / "research/v2/authorizations/"
                     "trn21-c5c-a0-r4-source-feasibility/"
                     "OPERATOR_LOCK_READONLY_FEASIBILITY_20260925T053029Z.md")
LOCK_MODE_RAW = (core.REPO / "research/v2/authorizations/"
                 "trn21-c5c-a0-r4-source-feasibility/"
                 "OPERATOR_RAW_LOCK_READONLY_20260925T053029Z.json")
LOCK_ACL_RECEIPT = (core.REPO / "research/v2/authorizations/"
                    "trn21-c5c-a0-r4-source-feasibility/"
                    "OPERATOR_GROUP_ACL_READONLY_20260925T053728Z.md")
LOCK_ACL_RAW = (core.REPO / "research/v2/authorizations/"
                "trn21-c5c-a0-r4-source-feasibility/"
                "OPERATOR_RAW_GROUP_ACL_READONLY_20260925T053728Z.json")
HELD_R2_REQUEST = HERE / "TRANSFER_REQUEST.FINAL_R2_PENDING_REVIEW.json"
HELD_R2_WRAPPER = HERE / "operator_transfer_final_r2_once.py"
HELD_R2_TRANSPORT = HERE / "stage_r4_transport_final_r2.py"
HELD_R2_REMOTE = HERE / "stage_r4_remote_final_r2.py"
HELD_R2_TESTS = HERE / "test_operator_transfer_final_r2_once.py"
HELD_R2_RUNBOOK = HERE / "FINAL_R2_ENABLEMENT_RUNBOOK.md"
SUPERVISOR_R2_HOLD = HERE / "SUPERVISOR_FINAL_R2_HOLD.md"
HELD_FINAL_REQUEST = HERE / "TRANSFER_REQUEST.FINAL_PENDING_REVIEW.json"
HELD_FINAL_WRAPPER = HERE / "operator_transfer_final_once.py"
HELD_FINAL_TRANSPORT = HERE / "stage_r4_transport_final.py"
HELD_FINAL_REMOTE = HERE / "stage_r4_remote_final.py"
HELD_FINAL_REVIEW = HERE / "INDEPENDENT_FINAL_ENABLEMENT_HOLD.md"
HELD_FINAL_NEGATIVE_TESTS = HERE / "independent_final_enablement_negative_tests.py"
STATIC_REVIEW = HERE / "INDEPENDENT_STATIC_TRANSFER_REVIEW.md"
SOURCE_STAGE_HOLD_RUNBOOK = core.REPO / "research/v2/authorizations/trn21-c5c-a0-r4-source-stage-hold/RUNBOOK.md"
MARKER = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/"
              "trn21-c5c-a0-r4-transfer-final-r5-enabled-r1-one-attempt.marker")

REQUEST_SHA256 = "b611a7546d1c73d338dea14d083432b16fbea5616bd3002c5dd57850d1f0710c"
DISABLED_REQUEST_SHA256 = "5af3241bf07ee3275685f1f8881be1a297e36e4542c695149269f2cdc2c1079b"
CORE_SHA256 = "81f69b8f15837fd34aced5463c605c2a91936a36330a33ca6fc32f9fcc6e4621"
TRANSPORT_SHA256 = "fac4a840a43d7addb0ee3579330fbd02b5b49d759b4e7e7b90ee904e14a01647"
REMOTE_SHA256 = "be2e0aca21aaf5e29b330c5c0b004dd8a013e99c2aaaa29e820be53b0a15dd68"
FINAL_TRANSPORT_SHA256 = "edcd84c6269b5e78dd1440ea6a565913e33d078350dab574e5a89e4265080db1"
FINAL_REMOTE_SHA256 = "ad30d93b464175c6df1550654076bc868a8bd8cd652fedd08bd5bfe91f055e2b"
PRIOR_R5_REQUEST_SHA256 = "6cdd7e966238e20eed16ab0a5abcf8ad89ec324318de539b230b7513ac01d65e"
PRIOR_R5_WRAPPER_SHA256 = "75b1ad77476d3963208674b5f285efc5e78cca91d895c071c3c8eec9c1f0bdb7"
PRIOR_R5_TRANSPORT_SHA256 = FINAL_TRANSPORT_SHA256
PRIOR_R5_REMOTE_SHA256 = FINAL_REMOTE_SHA256
PRIOR_R5_TESTS_SHA256 = "ce787cfc5cc933cc66dfc592e4dda84c7b423bc959aa3489633bcbb432aba716"
PRIOR_R5_RUNBOOK_SHA256 = "9b8c6239fc40b2d19b2ad09c98fdd48b4013156e23ebe66eb021955d1c8397a5"
PRIOR_R5_INDEPENDENT_PASS_SHA256 = "be212cbbd50ac525049602c87b584fb2e8d21cca6c3e03ed29fcc587e67a58b4"
PRIOR_R5_INDEPENDENT_TESTS_SHA256 = "22997ceb8b89bb93340a227da7f790c40079c587aba6b5854fc1818f3ba72515"
PRIOR_R5_ORIGIN_COMMIT = "0bacf0d009b4894264a2f8097b7e5cad8286bf5a"
PRIOR_R5_PASS_CANONICAL_COMMIT = "73be2973dda3a0c1ff2d62a37612aa95a08f0027"
PRIOR_R4_REQUEST_SHA256 = "58390b53ae09c666b705d153355baa12dd6392d8e3ea550bd44451fa41ab09e6"
PRIOR_R4_WRAPPER_SHA256 = "e36742d05bc517e4da0357b47144c329ec666cf197f8673f1b5f9570928bbf04"
PRIOR_R4_TRANSPORT_SHA256 = "8907b6fe1e9f775cb1b14bbfe0be867be54fbe071098bee507762c54146e405f"
PRIOR_R4_REMOTE_SHA256 = FINAL_REMOTE_SHA256
PRIOR_R4_TESTS_SHA256 = "802481a6c303a4c1ce3aadd9ef92656a80b584e066d8cca5bbae5074ae4e0563"
PRIOR_R4_RUNBOOK_SHA256 = "b1ff4dff41fd4fbb158ec3af49b4f30e51c072d7dd051ee3f7b9f6cc96950bc1"
PRIOR_R4_INDEPENDENT_HOLD_SHA256 = "7a7029e30a7a9193bd3e5fd8d241ec28f277ff1bd1966c864d41019a98c196d6"
PRIOR_R4_INDEPENDENT_TESTS_SHA256 = "b9798cd4d805fa6a1103f61fbaf06fb1fe55e99bdb78c9e1a0735c1d5a8f63da"
PRIOR_R4_CANONICAL_COMMIT = "bca55b2cbc10a4d22184b7308e2c8b9c2624dc5c"
PRIOR_R4_HOLD_CANONICAL_COMMIT = "30ff004e7ded31fe881982d3b0533c7f86f94407"
PRIOR_R3_REQUEST_SHA256 = "3763e3c6c469745a92547d27e46aa01b05874d9b7911cd7ce8f6dcdfb49dd21e"
PRIOR_R3_WRAPPER_SHA256 = "6bf2aea104e9dbdefc69ee48a5bfe095e5b8f46c3d99bdd9ba9ba9aecf913a1a"
PRIOR_R3_TRANSPORT_SHA256 = "0b676d2105010a6d5e6770dc507133ca958dc677ae21079cf673713a9894a083"
PRIOR_R3_REMOTE_SHA256 = "3a738cff02b26824ea0be1fb50ee16b6cacce96941a9e2162506c7cf608ead32"
PRIOR_R3_TESTS_SHA256 = "d404c81111c38a298f897af1026b08b22dd96163400df60fde0723175969225d"
PRIOR_R3_RUNBOOK_SHA256 = "46f4b5c39e1ab3db200455f730bdb6bde9371ef4b2b8a186edf7cc40b0b9dcb2"
PRIOR_R3_STATIC_REVIEW_SHA256 = "57e50f32bb13b29a4a9b7e41f88c09c09d26dc0eb29efe38a3e81538d53d6000"
LOCK_MODE_RECEIPT_SHA256 = "0557d5b940808fa3549123fe881c92f6c611d08f1a66a1401b22eeb344781f3a"
LOCK_MODE_RAW_SHA256 = "687b4d7b57c4de041adad37352a385e15ead7cfba0b83c4d3a3df5bd12058d14"
LOCK_ACL_RECEIPT_SHA256 = "065636a908c2b8b5aa8feb99e952a14a2dc63c2803140228b5d85cd496e515c4"
LOCK_ACL_RAW_SHA256 = "484ea4e76b6a59d5e5ec6757db6936b0bc3e7fc3ad89f69aa304a84da483443c"
HELD_R2_REQUEST_SHA256 = "097b4e904c3d4cb0584f8973843110928388649f35f31efcd481e1863780e31e"
HELD_R2_WRAPPER_SHA256 = "494180db56ab0bbdf34e68a78b8d25fc6a240c8a02b8ca76fcb80a06bb906729"
HELD_R2_TRANSPORT_SHA256 = "d7d3b279929ae9cb764033178b5b48fd5aa8ded77e8e9111a6e1c148ddb03b58"
HELD_R2_REMOTE_SHA256 = "6b1cfe6b0b0d178a48bc881c389a2b2c3949736d7e9457c2aa4fa840581803c7"
HELD_R2_TESTS_SHA256 = "7f88d74d08d28669b28ba5e21b4b0d7dc3d0ee8ec5238ad70aef53e75b84986c"
HELD_R2_RUNBOOK_SHA256 = "15756ef78d5ba823ac54219568ff8467db7e7758c8dbf1c4465323bee5ebd4ec"
SUPERVISOR_R2_HOLD_SHA256 = "1f82c8d13c4abd8099a4a13a778628398088ba19dd6f8e7718ff0837b50e5639"
FROZEN_LAUNCHER_SHA256 = "9488efa918119439a33cbe68cb5638bd1af3b15f1f8f61ec790a216e2bc1e3f4"
FROZEN_HOST_LOCK_MODULE_SHA256 = "b34c368c12175615514e63daebc296ec09332ef801e176e0d19cc3edc045fed3"
HELD_FINAL_REQUEST_SHA256 = "04a6b215ac796927677a29aebdf414ffe0f361495cca8a45d2dec070bc60f136"
HELD_FINAL_WRAPPER_SHA256 = "4c02dd1891009effe00d22b01966943ae85041a8ccbc0006792f3b76859feec1"
HELD_FINAL_TRANSPORT_SHA256 = "8c1e362fd586a00137016d7e0fad6f6ed758124cf7f66716a143895c91b512ba"
HELD_FINAL_REMOTE_SHA256 = "9b73cd84a46af46e9ce7e8729e328878a7b29c16ac1556e9adeb93785fd9b4b8"
HELD_FINAL_REVIEW_SHA256 = "368c39aabd5437a0b2937773de3ad7aee4548e93a7255aa1a5012f792bbc49d0"
HELD_FINAL_NEGATIVE_TESTS_SHA256 = "ec73e4a510d5b4010d92804b98285a128f967a01623063395e3c00f26cff59d4"
STATIC_REVIEW_SHA256 = "3803dc589c7ebf7ffd870a3d2eaddeecec77496fd24fc3a1cc92ccba5a482ab3"
SOURCE_STAGE_HOLD_RUNBOOK_SHA256 = "e31b9df3b4b1b2ac0f0fef355d773ef980875c4a3841bc0355387af6af9627f5"
SOURCE_COMMIT = "446b657695f086f9b484fa601c9a3b61c2495912"
CAPSULE_MANIFEST_SHA256 = "7ec8798c3e6d0284a4514a15a0caeb17bc5eb2cb5df935fb7dd6e8f3aabee42a"
DISABLED_CANONICAL_COMMIT = "96f02153fc65e6236c84b3a76983ab91301d1da7"
STATIC_CANONICAL_COMMIT = "3d5e66806c98e3a635c51b595c1cdcc0579eb866"
STATIC_ORIGIN_COMMIT = "9cdc545e92cf65bea08aeb912b9b21a8f30e07f5"
HELD_FINAL_SOURCE_COMMIT = "2cda547669059f49b173a08e273bff7f81bd734c"
HELD_FINAL_CANONICAL_COMMIT = "0a5fc72cb77161f902feeee2620d69cb93740c05"
HELD_REVIEW_ORIGIN_COMMIT = "b7bef672b2372e93a8658950f6866dbd34abc232"
HELD_REVIEW_CANONICAL_COMMIT = "25fb21d9002e063bf3750e2d2e0bb34c93959f28"
ASSIGNMENT_ID = "trn21-c5c-a0-r4-transfer-final-r5-enabled-r1"
TRANSFER_ATTEMPT_LABEL = "trn21-c5c-a0-r4-source-transfer-r5-enabled-r1-20260925"
SCIENCE_LABEL = "fs-20260925-v2-c5c-a0-control-trn21-r4"
SCOPE = "TRN21_R4_CAPSULE_TRANSFER_ONLY_ONE_ATTEMPT"
FINAL_REVIEW_RELATIVE = "research/v2/authorizations/trn21-c5c-a0-r4-transfer-only/INDEPENDENT_FINAL_R5_ENABLED_REQUEST_REVIEW.json"
ASSIGNMENT_RELATIVE = "research/v2/authorizations/trn21-c5c-a0-r4-transfer-only/SUPERVISOR_FINAL_R5_ENABLED_TRANSFER_ONLY_ASSIGNMENT.json"


def _expected_request():
    remote = final_transport.remote_constants()  # Hash checked before this is called.
    return {
        "schema": "trn21-c5c-a0-r4-transfer-final-request-v5-enabled-candidate-r1",
        "status": "ENABLED_REQUEST_CANDIDATE_PENDING_EXACT_FINAL_REVIEW_AND_SUPERVISOR_ASSIGNMENT",
        "dispatch_authorized": True,
        "dispatch_meaning": "TRANSFER_INTENT_ONLY_REQUIRES_EXACT_FINAL_PASS_AND_SUPERVISOR_ONE_SHOT_ASSIGNMENT",
        "source_transfer_authorized": True,
        "prepared_by_role": "implementer",
        "assignment_id": ASSIGNMENT_ID,
        "transfer_attempt_label": TRANSFER_ATTEMPT_LABEL,
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
        "prior_final_r5_origin_commit": PRIOR_R5_ORIGIN_COMMIT,
        "prior_final_r5_independent_pass_canonical_commit": PRIOR_R5_PASS_CANONICAL_COMMIT,
        "prior_final_r5_request_sha256": PRIOR_R5_REQUEST_SHA256,
        "prior_final_r5_wrapper_sha256": PRIOR_R5_WRAPPER_SHA256,
        "prior_final_r5_transport_sha256": PRIOR_R5_TRANSPORT_SHA256,
        "prior_final_r5_remote_sha256": PRIOR_R5_REMOTE_SHA256,
        "prior_final_r5_tests_sha256": PRIOR_R5_TESTS_SHA256,
        "prior_final_r5_runbook_sha256": PRIOR_R5_RUNBOOK_SHA256,
        "prior_final_r5_independent_pass_sha256": PRIOR_R5_INDEPENDENT_PASS_SHA256,
        "prior_final_r5_independent_tests_sha256": PRIOR_R5_INDEPENDENT_TESTS_SHA256,
        "prior_final_r4_candidate_canonical_commit": PRIOR_R4_CANONICAL_COMMIT,
        "prior_final_r4_independent_hold_canonical_commit": PRIOR_R4_HOLD_CANONICAL_COMMIT,
        "prior_final_r4_request_sha256": PRIOR_R4_REQUEST_SHA256,
        "prior_final_r4_wrapper_sha256": PRIOR_R4_WRAPPER_SHA256,
        "prior_final_r4_transport_sha256": PRIOR_R4_TRANSPORT_SHA256,
        "prior_final_r4_remote_sha256": PRIOR_R4_REMOTE_SHA256,
        "prior_final_r4_tests_sha256": PRIOR_R4_TESTS_SHA256,
        "prior_final_r4_runbook_sha256": PRIOR_R4_RUNBOOK_SHA256,
        "prior_final_r4_independent_hold_sha256": PRIOR_R4_INDEPENDENT_HOLD_SHA256,
        "prior_final_r4_independent_tests_sha256": PRIOR_R4_INDEPENDENT_TESTS_SHA256,
        "prior_final_r3_request_sha256": PRIOR_R3_REQUEST_SHA256,
        "prior_final_r3_wrapper_sha256": PRIOR_R3_WRAPPER_SHA256,
        "prior_final_r3_transport_sha256": PRIOR_R3_TRANSPORT_SHA256,
        "prior_final_r3_remote_sha256": PRIOR_R3_REMOTE_SHA256,
        "prior_final_r3_tests_sha256": PRIOR_R3_TESTS_SHA256,
        "prior_final_r3_runbook_sha256": PRIOR_R3_RUNBOOK_SHA256,
        "prior_final_r3_disabled_static_review_sha256": PRIOR_R3_STATIC_REVIEW_SHA256,
        "trn21_lock_mode_readonly_receipt_sha256": LOCK_MODE_RECEIPT_SHA256,
        "trn21_lock_mode_readonly_raw_sha256": LOCK_MODE_RAW_SHA256,
        "trn21_lock_acl_readonly_receipt_sha256": LOCK_ACL_RECEIPT_SHA256,
        "trn21_lock_acl_readonly_raw_sha256": LOCK_ACL_RAW_SHA256,
        "host_lock_private_parent_exception": {
            "cache_inode": remote.LOCK_OBSERVED_INODES["cache"],
            "host_directory_inode": remote.LOCK_OBSERVED_INODES["host_directory"],
            "file_inode": remote.LOCK_OBSERVED_INODES["file"],
            "cache_mode": oct(remote.LOCK_OBSERVED_MODES["cache"]),
            "host_directory_mode": oct(remote.LOCK_OBSERVED_MODES["host_directory"]),
            "file_mode": oct(remote.LOCK_OBSERVED_MODES["file"]),
            "owner_uid_gid": [remote.UID, remote.LOCK_OBSERVED_GID],
            "device": remote.LOCK_OBSERVED_DEVICE,
            "acl_xattrs_absent_required": True,
            "same_namespace_relevant_mount_alias_absent_required": True,
            "exact_named_and_opened_identity_required": True,
        },
        "held_final_r2_request_sha256": HELD_R2_REQUEST_SHA256,
        "held_final_r2_wrapper_sha256": HELD_R2_WRAPPER_SHA256,
        "held_final_r2_transport_sha256": HELD_R2_TRANSPORT_SHA256,
        "held_final_r2_remote_sha256": HELD_R2_REMOTE_SHA256,
        "held_final_r2_tests_sha256": HELD_R2_TESTS_SHA256,
        "held_final_r2_runbook_sha256": HELD_R2_RUNBOOK_SHA256,
        "supervisor_final_r2_hold_sha256": SUPERVISOR_R2_HOLD_SHA256,
        "frozen_446b_launcher_sha256": FROZEN_LAUNCHER_SHA256,
        "frozen_446b_host_lock_module_sha256": FROZEN_HOST_LOCK_MODULE_SHA256,
        "held_final_candidate_source_commit": HELD_FINAL_SOURCE_COMMIT,
        "held_final_candidate_canonical_commit": HELD_FINAL_CANONICAL_COMMIT,
        "held_final_review_origin_commit": HELD_REVIEW_ORIGIN_COMMIT,
        "held_final_review_canonical_commit": HELD_REVIEW_CANONICAL_COMMIT,
        "held_final_request_sha256": HELD_FINAL_REQUEST_SHA256,
        "held_final_wrapper_sha256": HELD_FINAL_WRAPPER_SHA256,
        "held_final_transport_sha256": HELD_FINAL_TRANSPORT_SHA256,
        "held_final_remote_sha256": HELD_FINAL_REMOTE_SHA256,
        "held_final_review_sha256": HELD_FINAL_REVIEW_SHA256,
        "held_final_negative_tests_sha256": HELD_FINAL_NEGATIVE_TESTS_SHA256,
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
        "full_probe_before_and_after_inode_floor_required": True,
        "existing_e65_checkout_preserved": remote.E65_CHECKOUT,
        "future_clean_checkout_remains_absent": remote.CHECKOUT,
        "future_science_label_reserved_not_launched": SCIENCE_LABEL,
        "live_path_absences_verified": False,
        "required_fresh_read_only_preflight": True,
        "preflight_to_writer_boot_mount_parent_binding_required": True,
        "writer_full_live_protected_asset_and_quiescence_probe_required": True,
        "writer_final_stop_process_neuron_target_gate_before_o_excl_required": True,
        "writer_final_science_label_scan_after_all_slow_activity_checks_required": True,
        "writer_receipt_final_label_evidence_required": True,
        "existing_host_wide_lock_path": "/home/ubuntu/.cache/trainium_freshstart/device0.lock",
        "existing_host_wide_lock_readonly_nofollow_nonblocking_flock_required": True,
        "host_lock_private_parent_exception_requires_fresh_live_probe": True,
        "host_lock_file_create_modify_delete_authorized": False,
        "preflight_writer_readback_postflight_lock_identity_continuity_required": True,
        "readback_and_postflight_clean_science_label_evidence_required": True,
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
        "frozen_446b_native_launcher_separate_hold_acknowledged": True,
        "eval_authorized": False,
        "monitor_operation_authorized": False,
        "cleanup_authorized": False,
        "submission_authorized": False,
    }


def _read_frozen_request_and_code():
    """Read actual reviewed files and reject any substitution before SSH."""
    if Path(prior_r5.__file__).resolve() != PRIOR_R5_WRAPPER.resolve():
        raise RuntimeError("FINAL REQUEST HOLD: reviewed r5 wrapper import changed")
    prior_r5_request, _ = prior_r5._read_frozen_request_and_code()
    if (prior_r5_request["dispatch_authorized"] is not False or
            prior_r5_request["source_transfer_authorized"] is not False):
        raise RuntimeError("FINAL REQUEST HOLD: reviewed r5 predecessor is not disabled")
    if Path(prior_r4.__file__).resolve() != PRIOR_R4_WRAPPER.resolve():
        raise RuntimeError("FINAL REQUEST HOLD: held r4 wrapper import changed")
    prior_r4_request, _ = prior_r4._read_frozen_request_and_code()
    if (prior_r4_request["dispatch_authorized"] is not False or
            prior_r4_request["source_transfer_authorized"] is not False):
        raise RuntimeError("FINAL REQUEST HOLD: held r4 predecessor is not disabled")
    if Path(prior_r3.__file__).resolve() != PRIOR_R3_WRAPPER.resolve():
        raise RuntimeError("FINAL REQUEST HOLD: reviewed r3 wrapper import changed")
    prior_request, _ = prior_r3._read_frozen_request_and_code()
    if (prior_request["dispatch_authorized"] is not False or
            prior_request["source_transfer_authorized"] is not False):
        raise RuntimeError("FINAL REQUEST HOLD: reviewed r3 predecessor is not disabled")
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
            (PRIOR_R5_REQUEST, PRIOR_R5_REQUEST_SHA256),
            (PRIOR_R5_WRAPPER, PRIOR_R5_WRAPPER_SHA256),
            (PRIOR_R5_TRANSPORT, PRIOR_R5_TRANSPORT_SHA256),
            (PRIOR_R5_REMOTE, PRIOR_R5_REMOTE_SHA256),
            (PRIOR_R5_TESTS, PRIOR_R5_TESTS_SHA256),
            (PRIOR_R5_RUNBOOK, PRIOR_R5_RUNBOOK_SHA256),
            (PRIOR_R5_INDEPENDENT_PASS, PRIOR_R5_INDEPENDENT_PASS_SHA256),
            (PRIOR_R5_INDEPENDENT_TESTS, PRIOR_R5_INDEPENDENT_TESTS_SHA256),
            (PRIOR_R4_REQUEST, PRIOR_R4_REQUEST_SHA256),
            (PRIOR_R4_WRAPPER, PRIOR_R4_WRAPPER_SHA256),
            (PRIOR_R4_TRANSPORT, PRIOR_R4_TRANSPORT_SHA256),
            (PRIOR_R4_REMOTE, PRIOR_R4_REMOTE_SHA256),
            (PRIOR_R4_TESTS, PRIOR_R4_TESTS_SHA256),
            (PRIOR_R4_RUNBOOK, PRIOR_R4_RUNBOOK_SHA256),
            (PRIOR_R4_INDEPENDENT_HOLD, PRIOR_R4_INDEPENDENT_HOLD_SHA256),
            (PRIOR_R4_INDEPENDENT_TESTS, PRIOR_R4_INDEPENDENT_TESTS_SHA256),
            (PRIOR_R3_REQUEST, PRIOR_R3_REQUEST_SHA256),
            (PRIOR_R3_WRAPPER, PRIOR_R3_WRAPPER_SHA256),
            (PRIOR_R3_TRANSPORT, PRIOR_R3_TRANSPORT_SHA256),
            (PRIOR_R3_REMOTE, PRIOR_R3_REMOTE_SHA256),
            (PRIOR_R3_TESTS, PRIOR_R3_TESTS_SHA256),
            (PRIOR_R3_RUNBOOK, PRIOR_R3_RUNBOOK_SHA256),
            (PRIOR_R3_STATIC_REVIEW, PRIOR_R3_STATIC_REVIEW_SHA256),
            (LOCK_MODE_RECEIPT, LOCK_MODE_RECEIPT_SHA256),
            (LOCK_MODE_RAW, LOCK_MODE_RAW_SHA256),
            (LOCK_ACL_RECEIPT, LOCK_ACL_RECEIPT_SHA256),
            (LOCK_ACL_RAW, LOCK_ACL_RAW_SHA256),
            (HELD_R2_REQUEST, HELD_R2_REQUEST_SHA256),
            (HELD_R2_WRAPPER, HELD_R2_WRAPPER_SHA256),
            (HELD_R2_TRANSPORT, HELD_R2_TRANSPORT_SHA256),
            (HELD_R2_REMOTE, HELD_R2_REMOTE_SHA256),
            (HELD_R2_TESTS, HELD_R2_TESTS_SHA256),
            (HELD_R2_RUNBOOK, HELD_R2_RUNBOOK_SHA256),
            (SUPERVISOR_R2_HOLD, SUPERVISOR_R2_HOLD_SHA256),
            (HELD_FINAL_REQUEST, HELD_FINAL_REQUEST_SHA256),
            (HELD_FINAL_WRAPPER, HELD_FINAL_WRAPPER_SHA256),
            (HELD_FINAL_TRANSPORT, HELD_FINAL_TRANSPORT_SHA256),
            (HELD_FINAL_REMOTE, HELD_FINAL_REMOTE_SHA256),
            (HELD_FINAL_REVIEW, HELD_FINAL_REVIEW_SHA256),
            (HELD_FINAL_NEGATIVE_TESTS, HELD_FINAL_NEGATIVE_TESTS_SHA256),
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
            core.transport.PRE_FREE or
            plan["frozen_source_file_sha256"]["research/fresh_start_launch.py"] !=
            FROZEN_LAUNCHER_SHA256 or
            plan["frozen_source_file_sha256"]["research/fresh_start_host.py"] !=
            FROZEN_HOST_LOCK_MODULE_SHA256):
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
            final_remote.LABEL != SCIENCE_LABEL or
            final_remote.LOCK_PATH !=
            "/home/ubuntu/.cache/trainium_freshstart/device0.lock" or
            final_remote.LOCK_OBSERVED_INODES !=
            {"cache": 295736, "host_directory": 596822, "file": 596823} or
            final_remote.LOCK_OBSERVED_MODES !=
            {"cache": 0o700, "host_directory": 0o775, "file": 0o664} or
            final_remote.LOCK_OBSERVED_GID != 1000 or
            final_remote.LOCK_OBSERVED_DEVICE != 66305 or
            final_transport.SOURCE != core.transport.SOURCE or
            final_transport.ENDPOINT != core.transport.ENDPOINT or
            final_transport.TARGET != core.transport.TARGET):
        raise RuntimeError("FINAL REQUEST HOLD: final payload/source/host binding drift")
    wrapper_sha = core._sha(core._read_evidence(HERE / "operator_transfer_final_r5_enabled_once.py"))
    tests_sha = core._sha(core._read_evidence(ENABLED_TESTS))
    runbook_sha = core._sha(core._read_evidence(ENABLED_RUNBOOK))
    return request, {"request_sha256": core._sha(request_bytes),
                     "final_wrapper_sha256": wrapper_sha,
                     "candidate_tests_sha256": tests_sha,
                     "candidate_runbook_sha256": runbook_sha}


def _expected_final_review(binding):
    return {
        "schema": "trn21-r4-final-transfer-independent-review-v5-enabled-candidate-r1",
        "verdict": "PASS",
        "reviewer_role": "independent_verifier",
        "no_host_action_by_reviewer": True,
        "source_transfer_only_reviewed": True,
        "native_launch_remains_hold": True,
        "request_sha256": binding["request_sha256"],
        "final_wrapper_sha256": binding["final_wrapper_sha256"],
        "candidate_tests_sha256": binding["candidate_tests_sha256"],
        "candidate_runbook_sha256": binding["candidate_runbook_sha256"],
        "disabled_request_sha256": DISABLED_REQUEST_SHA256,
        "disabled_core_wrapper_sha256": CORE_SHA256,
        "transport_sha256": TRANSPORT_SHA256,
        "remote_sha256": REMOTE_SHA256,
        "final_transport_sha256": FINAL_TRANSPORT_SHA256,
        "final_remote_sha256": FINAL_REMOTE_SHA256,
        "prior_final_r5_origin_commit": PRIOR_R5_ORIGIN_COMMIT,
        "prior_final_r5_independent_pass_canonical_commit": PRIOR_R5_PASS_CANONICAL_COMMIT,
        "prior_final_r5_request_sha256": PRIOR_R5_REQUEST_SHA256,
        "prior_final_r5_wrapper_sha256": PRIOR_R5_WRAPPER_SHA256,
        "prior_final_r5_transport_sha256": PRIOR_R5_TRANSPORT_SHA256,
        "prior_final_r5_remote_sha256": PRIOR_R5_REMOTE_SHA256,
        "prior_final_r5_tests_sha256": PRIOR_R5_TESTS_SHA256,
        "prior_final_r5_runbook_sha256": PRIOR_R5_RUNBOOK_SHA256,
        "prior_final_r5_independent_pass_sha256": PRIOR_R5_INDEPENDENT_PASS_SHA256,
        "prior_final_r5_independent_tests_sha256": PRIOR_R5_INDEPENDENT_TESTS_SHA256,
        "prior_final_r4_request_sha256": PRIOR_R4_REQUEST_SHA256,
        "prior_final_r4_wrapper_sha256": PRIOR_R4_WRAPPER_SHA256,
        "prior_final_r4_transport_sha256": PRIOR_R4_TRANSPORT_SHA256,
        "prior_final_r4_remote_sha256": PRIOR_R4_REMOTE_SHA256,
        "prior_final_r4_tests_sha256": PRIOR_R4_TESTS_SHA256,
        "prior_final_r4_runbook_sha256": PRIOR_R4_RUNBOOK_SHA256,
        "prior_final_r4_independent_hold_sha256": PRIOR_R4_INDEPENDENT_HOLD_SHA256,
        "prior_final_r4_independent_tests_sha256": PRIOR_R4_INDEPENDENT_TESTS_SHA256,
        "full_probe_before_and_after_inode_floor_reviewed": True,
        "prior_final_r3_request_sha256": PRIOR_R3_REQUEST_SHA256,
        "prior_final_r3_wrapper_sha256": PRIOR_R3_WRAPPER_SHA256,
        "prior_final_r3_transport_sha256": PRIOR_R3_TRANSPORT_SHA256,
        "prior_final_r3_remote_sha256": PRIOR_R3_REMOTE_SHA256,
        "prior_final_r3_tests_sha256": PRIOR_R3_TESTS_SHA256,
        "prior_final_r3_runbook_sha256": PRIOR_R3_RUNBOOK_SHA256,
        "prior_final_r3_disabled_static_review_sha256": PRIOR_R3_STATIC_REVIEW_SHA256,
        "trn21_lock_mode_readonly_receipt_sha256": LOCK_MODE_RECEIPT_SHA256,
        "trn21_lock_mode_readonly_raw_sha256": LOCK_MODE_RAW_SHA256,
        "trn21_lock_acl_readonly_receipt_sha256": LOCK_ACL_RECEIPT_SHA256,
        "trn21_lock_acl_readonly_raw_sha256": LOCK_ACL_RAW_SHA256,
        "host_lock_private_parent_exception_reviewed": True,
        "held_final_r2_request_sha256": HELD_R2_REQUEST_SHA256,
        "held_final_r2_wrapper_sha256": HELD_R2_WRAPPER_SHA256,
        "held_final_r2_transport_sha256": HELD_R2_TRANSPORT_SHA256,
        "held_final_r2_remote_sha256": HELD_R2_REMOTE_SHA256,
        "supervisor_final_r2_hold_sha256": SUPERVISOR_R2_HOLD_SHA256,
        "held_final_request_sha256": HELD_FINAL_REQUEST_SHA256,
        "held_final_wrapper_sha256": HELD_FINAL_WRAPPER_SHA256,
        "held_final_transport_sha256": HELD_FINAL_TRANSPORT_SHA256,
        "held_final_remote_sha256": HELD_FINAL_REMOTE_SHA256,
        "held_final_review_sha256": HELD_FINAL_REVIEW_SHA256,
        "held_final_negative_tests_sha256": HELD_FINAL_NEGATIVE_TESTS_SHA256,
        "final_label_scan_after_slow_gate_and_lock_reviewed": True,
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
        "transfer_attempt_label": TRANSFER_ATTEMPT_LABEL,
        "future_science_label_reserved_not_launched": SCIENCE_LABEL,
        "scope": SCOPE,
        "host": "trn21",
        "hostname": "ip-172-31-34-85",
        "endpoint": core.transport.ENDPOINT,
        "one_attempt": True,
    }


def _expected_assignment(binding, final_review_sha256):
    return {
        "schema": "trn21-r4-final-transfer-supervisor-assignment-v5-enabled-candidate-r1",
        "issuer_role": "supervisor",
        "dispatch_authorized": True,
        "source_transfer_authorized": True,
        "authorized_operation": "source_capsule_transfer_only",
        "assignment_id": ASSIGNMENT_ID,
        "transfer_attempt_label": TRANSFER_ATTEMPT_LABEL,
        "future_science_label_reserved_not_launched": SCIENCE_LABEL,
        "scope": SCOPE,
        "host": "trn21",
        "hostname": "ip-172-31-34-85",
        "endpoint": core.transport.ENDPOINT,
        "request_sha256": binding["request_sha256"],
        "final_independent_review_sha256": final_review_sha256,
        "final_wrapper_sha256": binding["final_wrapper_sha256"],
        "candidate_tests_sha256": binding["candidate_tests_sha256"],
        "candidate_runbook_sha256": binding["candidate_runbook_sha256"],
        "disabled_request_sha256": DISABLED_REQUEST_SHA256,
        "disabled_core_wrapper_sha256": CORE_SHA256,
        "transport_sha256": TRANSPORT_SHA256,
        "remote_sha256": REMOTE_SHA256,
        "final_transport_sha256": FINAL_TRANSPORT_SHA256,
        "final_remote_sha256": FINAL_REMOTE_SHA256,
        "prior_final_r5_origin_commit": PRIOR_R5_ORIGIN_COMMIT,
        "prior_final_r5_independent_pass_canonical_commit": PRIOR_R5_PASS_CANONICAL_COMMIT,
        "prior_final_r5_request_sha256": PRIOR_R5_REQUEST_SHA256,
        "prior_final_r5_wrapper_sha256": PRIOR_R5_WRAPPER_SHA256,
        "prior_final_r5_transport_sha256": PRIOR_R5_TRANSPORT_SHA256,
        "prior_final_r5_remote_sha256": PRIOR_R5_REMOTE_SHA256,
        "prior_final_r5_tests_sha256": PRIOR_R5_TESTS_SHA256,
        "prior_final_r5_runbook_sha256": PRIOR_R5_RUNBOOK_SHA256,
        "prior_final_r5_independent_pass_sha256": PRIOR_R5_INDEPENDENT_PASS_SHA256,
        "prior_final_r5_independent_tests_sha256": PRIOR_R5_INDEPENDENT_TESTS_SHA256,
        "prior_final_r4_request_sha256": PRIOR_R4_REQUEST_SHA256,
        "prior_final_r4_wrapper_sha256": PRIOR_R4_WRAPPER_SHA256,
        "prior_final_r4_transport_sha256": PRIOR_R4_TRANSPORT_SHA256,
        "prior_final_r4_remote_sha256": PRIOR_R4_REMOTE_SHA256,
        "prior_final_r4_tests_sha256": PRIOR_R4_TESTS_SHA256,
        "prior_final_r4_runbook_sha256": PRIOR_R4_RUNBOOK_SHA256,
        "prior_final_r4_independent_hold_sha256": PRIOR_R4_INDEPENDENT_HOLD_SHA256,
        "prior_final_r4_independent_tests_sha256": PRIOR_R4_INDEPENDENT_TESTS_SHA256,
        "full_probe_before_and_after_inode_floor_reviewed": True,
        "prior_final_r3_request_sha256": PRIOR_R3_REQUEST_SHA256,
        "prior_final_r3_wrapper_sha256": PRIOR_R3_WRAPPER_SHA256,
        "prior_final_r3_transport_sha256": PRIOR_R3_TRANSPORT_SHA256,
        "prior_final_r3_remote_sha256": PRIOR_R3_REMOTE_SHA256,
        "prior_final_r3_tests_sha256": PRIOR_R3_TESTS_SHA256,
        "prior_final_r3_runbook_sha256": PRIOR_R3_RUNBOOK_SHA256,
        "prior_final_r3_disabled_static_review_sha256": PRIOR_R3_STATIC_REVIEW_SHA256,
        "trn21_lock_mode_readonly_receipt_sha256": LOCK_MODE_RECEIPT_SHA256,
        "trn21_lock_mode_readonly_raw_sha256": LOCK_MODE_RAW_SHA256,
        "trn21_lock_acl_readonly_receipt_sha256": LOCK_ACL_RECEIPT_SHA256,
        "trn21_lock_acl_readonly_raw_sha256": LOCK_ACL_RAW_SHA256,
        "host_lock_private_parent_exception_reviewed": True,
        "held_final_r2_request_sha256": HELD_R2_REQUEST_SHA256,
        "held_final_r2_wrapper_sha256": HELD_R2_WRAPPER_SHA256,
        "held_final_r2_transport_sha256": HELD_R2_TRANSPORT_SHA256,
        "held_final_r2_remote_sha256": HELD_R2_REMOTE_SHA256,
        "supervisor_final_r2_hold_sha256": SUPERVISOR_R2_HOLD_SHA256,
        "held_final_request_sha256": HELD_FINAL_REQUEST_SHA256,
        "held_final_wrapper_sha256": HELD_FINAL_WRAPPER_SHA256,
        "held_final_transport_sha256": HELD_FINAL_TRANSPORT_SHA256,
        "held_final_remote_sha256": HELD_FINAL_REMOTE_SHA256,
        "held_final_review_sha256": HELD_FINAL_REVIEW_SHA256,
        "held_final_negative_tests_sha256": HELD_FINAL_NEGATIVE_TESTS_SHA256,
        "final_label_scan_after_slow_gate_and_lock_reviewed": True,
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


def _decode_authority_json(raw):
    """Reject ambiguous duplicate keys before exact authority comparison."""
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate authority key: " + key)
            obj[key] = value
        return obj

    return json.loads(raw, object_pairs_hook=unique_object)


def _validate_authority_files(binding, review_bytes, assignment_bytes):
    try:
        review = _decode_authority_json(review_bytes)
        assignment = _decode_authority_json(assignment_bytes)
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
    if (request["dispatch_authorized"] is not True or
            request["source_transfer_authorized"] is not True):
        raise RuntimeError("DISPATCH HOLD: source-transfer intent is not enabled")
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
        payload = core._json_bytes({"schema": "trn21-r4-final-transfer-one-shot-marker-v5-enabled-r1",
                                    "utc": core._utc(), "assignment_id": ASSIGNMENT_ID,
                                    "transfer_attempt_label": TRANSFER_ATTEMPT_LABEL,
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
