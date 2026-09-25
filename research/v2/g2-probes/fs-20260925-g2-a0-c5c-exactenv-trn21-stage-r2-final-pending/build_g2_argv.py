"""Offline, disabled successor to the independently reviewed stage-r2 G2 plan.

This module constructs and checks command *data*. It never contacts a host or
executes the command. The old HOLD object remains byte-for-byte immutable.
"""

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CONTRACT = HERE / "GATE_CONTRACT.json"
FROZEN = REPO / "research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2"

PINNED_FILES = {
    "frozen_draft": {
        "contract": ("GATE_CONTRACT.json", "9867b33cb250c6851f46c00bada89e7b4b730c0c861770dab182e994240db2f1"),
        "builder": ("build_g2_argv.py", "74b1d49cda3b5898ccb66539239f7d3ab78f489e1ef23e80fbd7b616cc65f5eb"),
        "independent_static_review": ("INDEPENDENT_STATIC_REVIEW.md", "4712413bf8dd3da14f55f131426d1662b9061fe660a811c271c25e67ef222d9b"),
    },
    "transfer_lineage": {
        "operator_terminal_receipt": ("research/v2/authorizations/trn21-c5c-a0-stage-r2/operator-transfer-only-terminal-receipt.json", "9ab90252e5e21812f5aad2af7fd0bfd1e71205e2f8f740f6f41faf6606fe6a2e"),
        "independent_result_review": ("research/v2/authorizations/trn21-c5c-a0-stage-r2/INDEPENDENT_TRANSFER_RESULT_RECEIPT.md", "bfafcb56ff449708dc31a1ebbe9023448018ea505e37a15847ae78bc3a3a2916"),
    },
    "checkout_lineage": {
        "final_request": ("research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/CHECKOUT_REQUEST.FINAL_PENDING_REVIEW.json", "be97460e0fa4cd20c00d29c2eefbe0923e511b9327998629f7962fee07c29010"),
        "independent_request_review": ("research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/INDEPENDENT_FINAL_REQUEST_REVIEW.PASS.json", "acfc9036cf0864a2abf019e961e7680de976aba99b5ffcd656d2d4acf39f04eb"),
        "supervisor_checkout_only_assignment": ("research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/SUPERVISOR_CHECKOUT_ONLY_ASSIGNMENT.json", "4c1d1774bd2ac9070334e1bdad5d5fe4fd9f9f71b6c72b41272c0d211243e4db"),
    },
}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path):
    return json.loads(path.read_text())


def _frozen_builder():
    spec = importlib.util.spec_from_file_location("frozen_stage_r2_g2", FROZEN / "build_g2_argv.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pin_local_files(contract):
    """Authenticate every earlier review/receipt before inspecting its claims."""
    loaded = {}
    for section, entries in PINNED_FILES.items():
        binding = contract[section]
        for name, (relative, expected_sha) in entries.items():
            actual_path = FROZEN / relative if section == "frozen_draft" else REPO / relative
            _require(binding[name + "_path"] == str(actual_path.relative_to(REPO)),
                     "pinned path drift: " + section + "." + name)
            _require(binding[name + "_sha256"] == expected_sha,
                     "pinned digest drift: " + section + "." + name)
            _require(_sha256(actual_path) == expected_sha,
                     "reviewed file bytes drift: " + section + "." + name)
            if actual_path.suffix == ".json":
                loaded[name] = _json(actual_path)
    return loaded


def _check_identity(contract, old, transfer, checkout, review, assignment):
    source = contract["source_checkout"]
    commit = contract["source_commit"]
    train = contract["source_train_sha256"]
    prepare = contract["source_prepare_sha256"]
    _require(contract["host"] == old["host"] == checkout["host"] == review["host"] == assignment["host"] == "trn21",
             "host lineage drift")
    _require(source == old["source_checkout"] == checkout["checkout_path"] == review["checkout_path"] == assignment["checkout_path"],
             "checkout path lineage drift")
    _require(commit == old["source_commit"] == checkout["source_commit"] == review["source_commit"] == assignment["source_commit"],
             "source commit lineage drift")
    _require(train == old["source_train_sha256"] == checkout["train_sha256"] == review["train_sha256"] == assignment["train_sha256"],
             "train.py lineage drift")
    _require(prepare == old["source_prepare_sha256"] == checkout["prepare_sha256"] == review["prepare_sha256"] == assignment["prepare_sha256"],
             "prepare.py lineage drift")
    _require(review["verdict"] == "PASS_EXACT_CHECKOUT_REQUEST_ONLY" and
             review["reviewed_request_sha256"] == contract["checkout_lineage"]["final_request_sha256"] and
             assignment["request_sha256"] == contract["checkout_lineage"]["final_request_sha256"] and
             assignment["independent_request_review_sha256"] == contract["checkout_lineage"]["independent_request_review_sha256"],
             "independent checkout-request review binding drift")
    _require(assignment["checkout_only_authorized"] is True and
             assignment["one_attempt_only"] is True and
             assignment["cpu_g2_authorized"] is False and
             checkout["operator_one_attempt_only"] is True and
             review["cpu_g2_authorized"] is False,
             "checkout-only authority drift")
    _require(transfer["artifacts"]["capsule"]["operator_comparator_result"]["sha256"] ==
             checkout["source_capsule_sha256"] == review["source_capsule_sha256"] ==
             assignment["source_capsule_sha256"], "source capsule lineage drift")


def _check_safety_pins(contract, old, transfer, checkout):
    safety = contract["safety_pins"]
    tools = old["persistent_cpu_tools"]
    g2 = old["cpu_g2_isolation"]
    _require(safety["one_attempt_per_label"] is True and
             g2["one_attempt_per_label"] is True,
             "CPU label must allow exactly one attempt")
    _require(safety["cpu_label"] == g2["new_cpu_label"] and
             safety["cpu_root"] == g2["root_must_be_absent"] and
             safety["minimum_fresh_root_free_bytes"] == g2["minimum_fresh_root_free_bytes"] == 10737418240,
             "CPU scratch or disk gate drift")
    _require(safety["persistent_cpu_tool_required_mode"] == tools["required_mode"] == "0600" and
             safety["persistent_cpu_tool_required_uid"] == tools["required_uid"] == 1000 and
             safety["persistent_cpu_tool_required_device"] == tools["required_device"] == 66305 and
             safety["persistent_cpu_tool_required_nlink"] == tools["required_nlink"] == 1,
             "persistent CPU tool identity-policy drift")
    for name in ("preflight", "contained"):
        path = tools[name + "_path"]
        inode = safety[name + "_inode"]
        # Inode 0 (and bool) is not a valid authenticated identity.
        _require(type(inode) is int and inode > 0 and
                 type(tools[name + "_inode"]) is int and tools[name + "_inode"] == inode,
                 "invalid or unauthenticated " + name + " inode")
        transferred = transfer["artifacts"][name]["operator_comparator_result"]
        final = transfer["final_postflight_snapshot"]["targets"][path]
        request = checkout["persistent_targets"][path]
        _require(transferred["status"] == "CROSS_SESSION_PERSISTENCE_HASH_PASS" and
                 transferred["path"] == path and transferred["inode"] == inode and
                 transferred["device"] == 66305 and transferred["bytes"] == tools[name + "_bytes"] and
                 transferred["sha256"] == tools[name + "_sha256"],
                 "cross-session " + name + " identity drift")
        for observed in (final, request):
            _require(observed["exists"] is True and observed["type"] == "regular" and
                     observed["device"] == 66305 and observed["inode"] == inode and
                     observed["uid"] == 1000 and observed["nlink"] == 1 and
                     observed["bytes"] == tools[name + "_bytes"] and
                     observed["sha256"] == tools[name + "_sha256"] and
                     int(observed["mode"], 8) == 0o600,
                     "final reviewed " + name + " identity drift")


def build_plan(contract=None, *, hold_contract=None, transfer_receipt=None,
               checkout_request=None, checkout_review=None, checkout_assignment=None):
    """Return authenticated argv data; always HOLD for dispatch.

    Overrides exist solely for synthetic adversarial tests. Real use hashes
    and reloads every reviewed local artifact before trusting its fields.
    """
    c = _json(CONTRACT) if contract is None else contract
    _require(c["schema"] == "trn21-c5c-a0-exactenv-cpu-g2-stage-r2-final-pending-v1" and
             c["status"] == "IMPLEMENTER_FINAL_PENDING_INDEPENDENT_REVIEW_AND_CHECKOUT_RESULT_HOLD" and
             c["prepared_by_role"] == "implementer" and
             all(c[key] is False for key in ("cpu_g2_dispatch_authorized", "native_training_authorized",
                                             "evaluation_authorized", "submission_authorized")),
             "disabled-only G2 authority drift")
    lineage = c["checkout_lineage"]
    _require(lineage["source_checkout_result_independently_accepted"] is False and
             all(lineage[key] is None for key in (
                 "operator_checkout_terminal_receipt_path", "operator_checkout_terminal_receipt_sha256",
                 "independent_checkout_result_review_path", "independent_checkout_result_review_sha256")),
             "unreviewed checkout result cannot enable this package")
    pinned = _pin_local_files(c)
    legacy = _frozen_builder()
    legacy.check_local_evidence()
    old = _json(legacy.CONTRACT) if hold_contract is None else hold_contract
    transfer = pinned["operator_terminal_receipt"] if transfer_receipt is None else transfer_receipt
    checkout = pinned["final_request"] if checkout_request is None else checkout_request
    review = pinned["independent_request_review"] if checkout_review is None else checkout_review
    assignment = pinned["supervisor_checkout_only_assignment"] if checkout_assignment is None else checkout_assignment
    _check_identity(c, old, transfer, checkout, review, assignment)
    _check_safety_pins(c, old, transfer, checkout)
    plan = legacy.build_plan(old)
    frozen = c["frozen_draft"]
    for name, argv_key in (("exact_launch_argv_sha256", "launch_argv"),
                           ("exact_wait_argv_sha256", "wait_argv"),
                           ("exact_stop_on_failure_argv_sha256", "stop_on_failure_argv"),
                           ("exact_quiesce_argv_sha256", "quiesce_argv")):
        _require(frozen[name] == old[name] == legacy.argv_digest(plan[argv_key]),
                 "frozen command digest drift: " + name)
    return {"contract": c, "launch_argv": plan["launch_argv"],
            "wait_argv": plan["wait_argv"], "stop_on_failure_argv": plan["stop_on_failure_argv"],
            "quiesce_argv": plan["quiesce_argv"]}


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: this offline final-pending package never runs CPU G2")
