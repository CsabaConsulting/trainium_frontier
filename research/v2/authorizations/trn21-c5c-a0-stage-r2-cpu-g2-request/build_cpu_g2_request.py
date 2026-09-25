"""Offline binding of the accepted trn21 checkout to a disabled CPU G2 request.

This module never contacts a host, makes a filesystem write, or executes the
constructed command arrays. A separate Verifier and Supervisor are required.
"""

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REQUEST = HERE / "CPU_G2_REQUEST.HOLD.json"
BINDING = HERE / "INVOCATION_BINDING.HOLD.json"
EXPECTED_REQUEST_SHA256 = "20bb8c7349e7d98a2ff6dd05ff7570326c8e25da9b44871600ec321434bf1a78"

PINNED_FILES = {
    "hardened_offline_g2": {
        "contract": ("research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2-final-pending/GATE_CONTRACT.json", "743d75e489831d21cd682a3a7a2b84becd8f0dc8f8cfc914dc4aa27123c9d905"),
        "builder": ("research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2-final-pending/build_g2_argv.py", "f308730af1e98811d609c00e0bf348bc78c04354661e53674379184c71a96bdf"),
        "independent_static_review": ("research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2-final-pending/INDEPENDENT_STATIC_REVIEW.md", "ad19dc6a648977fbad4ce0b8d57de05256a68da0723160114fa5ac23ea60d455"),
    },
    "accepted_checkout_result": {
        "operator_terminal": ("research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-result/OPERATOR_CHECKOUT_ONLY_TERMINAL_RECEIPT.md", "1e11e59f1fe4ceecd393c43552d0f4105961e3d5fcc5e03709f07abc46d3dc3d"),
        "independent_result_review": ("research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-result/INDEPENDENT_CHECKOUT_RESULT_REVIEW.PASS.json", "b3c82bd594ae312142925cc3888f1de532473854f246a17b4ac4251baf37b89c"),
        "raw_checkout_stage": ("research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-result/CHECKOUT_STAGE_RECEIPT_20260925T031732Z.json", "097256565cfdf31f2bd5f17531e404f8911c22d35eb0d631c1b0a4d99c54ed37"),
        "raw_unconditional_postflight": ("research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-result/UNCONDITIONAL_POSTFLIGHT_20260925T031738Z.json", "4dea161fedcee76d335c053144bc7a7b36f3f20299ced2412151c3170c6d2fba"),
    },
}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path):
    return json.loads(path.read_text())


def _pin_files(request):
    loaded = {}
    for section, entries in PINNED_FILES.items():
        for name, (relative, expected_sha) in entries.items():
            _require(request[section][name + "_path"] == relative and
                     request[section][name + "_sha256"] == expected_sha,
                     "request file pin drift: " + section + "." + name)
            path = REPO / relative
            _require(_sha256(path) == expected_sha,
                     "reviewed evidence bytes drift: " + section + "." + name)
            if path.suffix == ".json":
                loaded[name] = _load_json(path)
    return loaded


def _hardened_plan():
    path = REPO / PINNED_FILES["hardened_offline_g2"]["builder"][0]
    spec = importlib.util.spec_from_file_location("hardened_trn21_g2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.build_plan()


def _check_source(request, hard, review, stage, post):
    source = request["source_checkout"]
    commit = request["source_commit"]
    train = request["source_train_sha256"]
    prepare = request["source_prepare_sha256"]
    stat = request["source_checkout_required_identity"]
    _require(request["host"] == "trn21" and request["expected_hostname"] == "ip-172-31-34-85" and
             review["host"] == request["host"] and review["expected_hostname"] == request["expected_hostname"] and
             post["hostname"] == request["expected_hostname"] and post["uid"] == 1000,
             "host identity drift")
    _require(source == hard["source_checkout"] == review["checkout_path"] ==
             stage["checkout"]["checkout_path"], "source checkout path drift")
    _require(commit == hard["source_commit"] == review["source_commit"] ==
             stage["checkout"]["git_checks"]["head"]["stdout"].strip(), "source commit drift")
    _require(train == hard["source_train_sha256"] == review["train_sha256"] ==
             stage["checkout"]["source_files"]["train.py"]["sha256"], "train.py drift")
    _require(prepare == hard["source_prepare_sha256"] == review["prepare_sha256"] ==
             stage["checkout"]["source_files"]["prepare.py"]["sha256"], "prepare.py drift")
    expected = {"device": 66305, "inode": 3168319, "uid": 1000, "mode": "0700",
                "type": "directory", "one_commit_shallow_clean_no_remote": True}
    _require(stat == expected and review["checkout_device"] == stat["device"] and
             review["checkout_inode"] == stat["inode"] and review["checkout_uid"] == stat["uid"] and
             int(review["checkout_mode"], 8) == int(stat["mode"], 8),
             "independently accepted checkout identity drift")
    staged = stage["checkout"]["checkout_stat"]
    later = post["targets"][source]
    for observed in (staged, later):
        _require(observed["device"] == stat["device"] and observed["inode"] == stat["inode"] and
                 observed["uid"] == stat["uid"] and int(observed["mode"], 8) == 0o700,
                 "checkout stage/postflight inode or mode drift")
    _require(later["exists"] is True and later["type"] == "directory" and
             type(request["source_train_inode"]) is int and request["source_train_inode"] == 3174933 and
             type(request["source_prepare_inode"]) is int and request["source_prepare_inode"] == 3168417 and
             request["source_train_inode"] == stage["checkout"]["source_files"]["train.py"]["inode"] and
             request["source_prepare_inode"] == stage["checkout"]["source_files"]["prepare.py"]["inode"],
             "source file identity drift")
    checks = stage["checkout"]["git_checks"]
    for key in ("all_count", "all_untracked", "fsck", "head", "head_count", "remotes", "shallow", "status"):
        _require(checks[key]["returncode"] == 0 and not checks[key]["stderr"],
                 "checkout Git check failed: " + key)
    _require(checks["all_count"]["stdout"] == checks["head_count"]["stdout"] == "1\n" and
             checks["all_untracked"]["stdout"] == checks["remotes"]["stdout"] == checks["status"]["stdout"] == "" and
             checks["shallow"]["stdout"] == "true\n" and stat["one_commit_shallow_clean_no_remote"] is True,
             "checkout Git source not the reviewed clean one-commit shallow source")


def _check_checkout_evidence(request, review, stage, post):
    accepted = request["accepted_checkout_result"]
    hashes = review["evidence_sha256"]
    _require(review["verdict"] == "PASS_EXACT_SOURCE_CHECKOUT_RESULT_ONLY" and
             review["prepared_by_role"] == "verifier" and
             review["cpu_g2_authorized_by_this_receipt"] is False and
             review["native_training_authorized_by_this_receipt"] is False and
             review["evaluation_authorized_by_this_receipt"] is False,
             "checkout result is not a valid limited-scope PASS")
    _require(hashes["operator_terminal_summary"] == accepted["operator_terminal_sha256"] and
             hashes["raw_checkout_stage_stdout"] == accepted["raw_checkout_stage_sha256"] and
             hashes["raw_unconditional_postflight_stdout"] == accepted["raw_unconditional_postflight_sha256"],
             "independent checkout-result evidence binding drift")
    _require(stage["status"] == "SOURCE_CHECKOUT_VERIFIED_FOR_INDEPENDENT_REVIEW_ONLY" and
             stage["cpu_g2_authorized"] is False and stage["native_training_authorized"] is False and
             stage["evaluation_authorized"] is False and
             post["root_free_bytes"] >= 10737418240 and
             review["postflight_root_free_bytes"] == post["root_free_bytes"],
             "checkout stage or postflight drift")
    # The accepted postflight scanned an older G2 path, not the new stage-r2
    # scratch root. Exact new-root absence is therefore a *fresh* live gate.
    root = request["cpu_g2_isolation"]["root_must_be_absent_at_fresh_preflight"]
    if root in post["targets"]:
        _require(post["targets"][root]["exists"] is False,
                 "CPU scratch root was already present at checkout postflight")
    _require(post["processes"]["hits"] == [] and
             post["processes"]["unreadable_cmdline_count"] == 0 and
             post["neuron0_fuser"]["pids"] == [] and
             post["host_lock_fuser"]["pids"] == [],
             "checkout postflight had a process or device holder")


def _check_tools(request, hard, stage, post):
    expected = hard["persistent_cpu_tools"]
    tools = request["persistent_cpu_tools"]
    for key in ("required_uid", "required_device", "required_mode", "required_nlink",
                "preflight_path", "preflight_bytes", "preflight_inode", "preflight_sha256",
                "contained_path", "contained_bytes", "contained_inode", "contained_sha256",
                "rehash_with_no_follow_before_and_after"):
        _require(tools[key] == expected[key], "CPU tool pin drift: " + key)
    _require(tools["retransfer_substitution_or_copy_into_checkout_authorized"] is False,
             "CPU tool replacement authority drift")
    for name in ("preflight", "contained"):
        inode = tools[name + "_inode"]
        path = tools[name + "_path"]
        _require(type(inode) is int and inode > 0, "invalid CPU tool inode")
        observed = (stage["transferred_tools_after"][name], post["targets"][path])
        for item in observed:
            value = item["target_stat"] if "target_stat" in item else item
            _require(value["device"] == 66305 and value["inode"] == inode and value["uid"] == 1000 and
                     value["nlink"] == 1 and value["bytes"] == tools[name + "_bytes"] and
                     int(value["mode"], 8) == 0o600,
                     "CPU tool stage/postflight identity drift")
            if "sha256" in item:
                _require(item["sha256"] == tools[name + "_sha256"], "CPU tool hash drift")
            else:
                _require(value["sha256"] == tools[name + "_sha256"], "CPU tool postflight hash drift")


def _check_cpu_plan(request, hard, plan, digest):
    cpu = request["cpu_g2_isolation"]
    old = hard["cpu_g2_isolation"]
    _require(cpu["one_attempt_per_label"] is True and old["one_attempt_per_label"] is True and
             cpu["new_cpu_label"] == old["new_cpu_label"] and
             cpu["root_must_be_absent_at_fresh_preflight"] == old["root_must_be_absent"] and
             cpu["root_parent_filesystem"] == "same persistent root ext4 as source checkout" and
             cpu["root_mode_if_created"] == old["root_mode_if_created"] == "0700" and
             cpu["minimum_fresh_root_free_bytes"] == old["minimum_fresh_root_free_bytes"] == 10737418240 and
             cpu["g2_options"] == old["g2_options"] and cpu["synthetic_cpu_only"] is True and
             cpu["real_data_native_neuron_or_public_eval_authorized"] is False,
             "CPU-only G2 isolation drift")
    _require(request["science_run_label_reserved_not_launched"] ==
             hard["selected_science"]["science_run_label_unconsumed"] and
             cpu["new_cpu_label"] != request["science_run_label_reserved_not_launched"],
             "science label collision or substitution")
    for name, argv_key in (("exact_launch_argv_sha256", "launch_argv"),
                           ("exact_wait_argv_sha256", "wait_argv"),
                           ("exact_stop_on_failure_argv_sha256", "stop_on_failure_argv"),
                           ("exact_quiesce_argv_sha256", "quiesce_argv")):
        _require(request[name] == hard[name] == digest(plan[argv_key]),
                 "exact CPU command drift: " + name)
    preflight = request["fresh_preflight_required"]
    preflight_keys = {
        "exact_source_checkout_git_hash_inode_uid_mode_and_clean_shallow_one_commit",
        "exact_cpu_tools_no_follow_hash_inode_mode_uid_device_and_nlink",
        "both_protected_checkpoints_and_six_warm_caches_unchanged",
        "exact_new_g2_root_process_dir_step_log_and_cpu_label_absent",
        "science_output_log_and_step_paths_absent",
        "host_boot_mount_namespace_ext4_and_parent_identity",
        "local_and_host_STOP_breakers_untripped",
        "no_matched_training_eval_process_host_lock_or_neuron_holder",
        "minimum_root_free_bytes",
    }
    _require(set(preflight) == preflight_keys and
             preflight["minimum_root_free_bytes"] == 10737418240 and
             all(value is True for key, value in preflight.items() if key != "minimum_root_free_bytes") and
             request["terminal_postflight_required_even_on_failure"] is True and
             request["result_requires_independent_cpu_numeric_comparison"] is True and
             request["argv_digest_rule"] == "SHA-256 of UTF-8 compact JSON array with comma/colon separators",
             "fresh live preflight or terminal postflight gate drift")


def _check_binding(request, binding, plan, digest):
    _require(binding["schema"] == "trn21-c5c-a0-stage-r2-exact-cpu-g2-invocation-hold-v1" and
             binding["status"] == "HOLD_PENDING_INDEPENDENT_CPU_REQUEST_STATIC_REVIEW" and
             binding["dispatch_authorized"] is False and
             binding["cpu_g2_dispatch_authorized"] is False and
             binding["request_path"] == str(REQUEST.relative_to(REPO)) and
             binding["request_sha256"] == EXPECTED_REQUEST_SHA256 == _sha256(REQUEST) and
             binding["offline_builder_path"] == str(Path(__file__).resolve().relative_to(REPO)) and
             binding["offline_builder_sha256"] == _sha256(Path(__file__).resolve()) and
             binding["hardened_g2_independent_static_review_sha256"] ==
                 request["hardened_offline_g2"]["independent_static_review_sha256"] and
             binding["independent_checkout_result_review_sha256"] ==
                 request["accepted_checkout_result"]["independent_result_review_sha256"] and
             binding["native_training_authorized"] is False and
             binding["evaluation_authorized"] is False and
             binding["one_attempt_only"] is True and
             binding["fresh_full_live_preflight_required"] is True and
             binding["unconditional_separate_full_read_only_postflight_required"] is True and
             binding["independent_static_review_sha256"] is None and
             binding["supervisor_cpu_only_assignment_sha256"] is None,
             "disabled exact invocation binding drift")
    _require(binding["source_checkout"] == request["source_checkout"] and
             binding["source_commit"] == request["source_commit"] and
             binding["cpu_label"] == request["cpu_g2_isolation"]["new_cpu_label"] and
             binding["cpu_root"] == request["cpu_g2_isolation"]["root_must_be_absent_at_fresh_preflight"],
             "invocation source or CPU label drift")
    for name, argv_key in (("exact_launch_argv_sha256", "launch_argv"),
                           ("exact_wait_argv_sha256", "wait_argv"),
                           ("exact_stop_on_failure_argv_sha256", "stop_on_failure_argv"),
                           ("exact_quiesce_argv_sha256", "quiesce_argv")):
        _require(binding[name] == request[name] == digest(plan[argv_key]),
                 "invocation command digest drift: " + name)


def build_plan(request=None, binding=None, *, review=None, stage=None, postflight=None):
    """Authenticate the frozen evidence and return command data, never authority."""
    r = _load_json(REQUEST) if request is None else request
    _require(r["schema"] == "trn21-c5c-a0-stage-r2-exact-cpu-g2-request-hold-v1" and
             r["status"] == "IMPLEMENTER_EXACT_CPU_ONLY_REQUEST_PENDING_INDEPENDENT_STATIC_REVIEW_HOLD" and
             r["prepared_by_role"] == "implementer" and
             all(r[key] is False for key in ("cpu_g2_dispatch_authorized", "native_training_authorized",
                                            "evaluation_authorized", "submission_authorized")) and
             r["independent_cpu_g2_request_static_review_sha256"] is None and
             r["supervisor_cpu_only_one_attempt_assignment_sha256"] is None,
             "CPU-only request authority must remain HOLD")
    _require(_sha256(REQUEST) == EXPECTED_REQUEST_SHA256, "exact HOLD request bytes drift")
    loaded = _pin_files(r)
    accepted_review = loaded["independent_result_review"] if review is None else review
    raw_stage = loaded["raw_checkout_stage"] if stage is None else stage
    raw_post = loaded["raw_unconditional_postflight"] if postflight is None else postflight
    hard_module, hard_plan = _hardened_plan()
    # The reviewed hardening sidecar authenticates the immutable earlier G2
    # contract; command science/tool fields remain in that frozen contract.
    hard = _load_json(hard_module._frozen_builder().CONTRACT)
    _check_source(r, hard, accepted_review, raw_stage, raw_post)
    _check_checkout_evidence(r, accepted_review, raw_stage, raw_post)
    _check_tools(r, hard, raw_stage, raw_post)
    _check_cpu_plan(r, hard, hard_plan, hard_module._frozen_builder().argv_digest)
    b = _load_json(BINDING) if binding is None else binding
    _check_binding(r, b, hard_plan, hard_module._frozen_builder().argv_digest)
    return {"request": r, "binding": b, "launch_argv": hard_plan["launch_argv"],
            "wait_argv": hard_plan["wait_argv"],
            "stop_on_failure_argv": hard_plan["stop_on_failure_argv"],
            "quiesce_argv": hard_plan["quiesce_argv"]}


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: CPU G2 request construction is offline only")
