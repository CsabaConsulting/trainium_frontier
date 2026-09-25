"""Offline A2 exact-size CPU G2 plan and receipt checks. Never contacts a host.

This is deliberately NOT an Operator launcher. The frozen request remains
disabled even when every local assertion here succeeds. A distinct reviewed
enabled request, fresh host preflight and Supervisor assignment are required.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REQUEST = HERE / "G2_REQUEST.HOLD.json"
LABEL = "fs-20260925-g2-a2-c5c-m64d128-trn22-r3"
CHECKOUT = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2"
SCRATCH = "/home/ubuntu/trainium-v2-g2-a2-c5c-m64d128-20260925-trn22-r3"
TRAIN_SHA = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREPARE_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
G2_SHA = "6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8"
CONTAINED_SHA = "0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d"
QUEUE_SHA = "1e9dcc4fd5f38e62829dc19460d2337662184ecfbd4d21d8a20f15993be9dd00"
SCRATCH_GUARD_SHA = "663788533e4aa10f8a1e9d8286ce1e51c6c6b4d17e802394206ce709a64a63c6"
KEYS = [
    "ngram_embeds.b1.weight", "ngram_embeds.t1.weight",
    "ngram_embeds.b5.weight", "ngram_embeds.t5.weight",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _pinned(path: str, expected_sha: str) -> dict | None:
    target = REPO / path
    require(target.is_file() and not target.is_symlink(), "missing or symlinked pin: " + path)
    require(sha256_file(target) == expected_sha, "pinned bytes changed: " + path)
    return json.loads(target.read_text()) if target.suffix == ".json" else None


def _queue_argv(queue: dict) -> list[str]:
    base = shlex.split(queue["base_command"])
    require(base[:5] == ["NEURON_LOGICAL_NC_CONFIG=2", "torchrun", "--standalone",
                         "--nproc_per_node=4", "train.py"], "queue launch shape drift")
    train_argv = base[5:]
    require(train_argv.count("--no-ngram-ve") == 1, "queue must have one table-off token")
    arms = [arm for arm in queue["queue"] if arm.get("id") == "A2"]
    require(len(arms) == 1 and arms[0].get("screen_host") == "trn22", "queue A2 host drift")
    delta = arms[0]["argv_delta"]
    require(delta["replace"] == "--no-ngram-ve", "queue A2 replacement drift")
    at = train_argv.index("--no-ngram-ve")
    return train_argv[:at] + delta["with"] + train_argv[at + 1:]


def build_plan() -> dict:
    """Authenticate the local static plan; produce command *data*, never execute it."""
    request = json.loads(REQUEST.read_text())
    require(request["schema"] == "c5c-a2-m64d128-cpu-g2-disabled-v2" and
            request["status"] == "IMPLEMENTER_DRAFT_PENDING_INDEPENDENT_STATIC_REVIEW_NO_DISPATCH" and
            request["prepared_by_role"] == "implementer", "request identity/status drift")
    authority = request["authority"]
    for key in ("cpu_g2_dispatch_authorized", "host_contact_authorized", "source_stage_authorized",
                "native_training_authorized", "evaluation_authorized", "aws_action_authorized",
                "cleanup_authorized", "submission_authorized"):
        require(authority[key] is False, "disabled authority drift: " + key)
    for key in ("independent_static_review", "supervisor_one_attempt_assignment",
                "operator_fresh_preflight", "operator_result"):
        require(authority[key] is None, "unreviewed authority injected: " + key)

    lineage = request["lineage"]
    for stem, expected_path, expected_sha in (
        ("proposal", "research/submission-validation/20260925-v2-c5c-a2-m64d128-trn22-r1/EXPERIMENT_PROPOSAL.json",
         "9f406cc97e4d0ce8fbc1e5f01a623dd4441a05bec02f90560cbe068f6b4d9d8f"),
        ("a2_gate", "research/submission-validation/20260925-v2-c5c-a2-m64d128-trn22-r1/GATE_CONTRACT.json",
         "a40723151099e0fec912d670b702ef9f2774fe14853025660a9ffc116d2fc364"),
        ("independent_static_pass", "research/agentic/verification/20260925-v2-c5c-a2-m64d128-trn22-r1-static-r2-review.md",
         "c1f4665335c3af3201593ff3c1770b21c6430abf821be078422e890f4a16d85d"),
        ("readonly_snapshot", "research/v2/observations/20260925-dual-host-readonly-operator/OPERATOR_READONLY_DUAL_HOST_SNAPSHOT_20260925T0733Z.md",
         "3665847467803f05f1a61a040f0b38e7a88e486bc7af52b2fbfcf24472343777"),
    ):
        require(lineage[stem + "_path"] == expected_path and
                lineage[stem + "_sha256"] == expected_sha, "lineage pin drift: " + stem)
        value = _pinned(expected_path, expected_sha)
        if stem == "a2_gate":
            old_gate = value
    require(lineage["snapshot_is_not_fresh_preflight"] is True, "snapshot scope drift")

    source = request["source"]
    for key, path, digest in (
        ("root_train_py_sha256", "submissions/research_v2/train.py", TRAIN_SHA),
        ("prepare_py_sha256", "prepare.py", PREPARE_SHA),
        ("g2_tool_sha256", "workflow/tools/preflight_dryrun.py", G2_SHA),
        ("containment_tool_sha256", "workflow/tools/contained.py", CONTAINED_SHA),
        ("queue_sha256", "research/v2/queue.json", QUEUE_SHA),
    ):
        require(source[key] == digest, "source pin drift: " + key)
        _pinned(path, digest)
    require(source["model_or_evaluator_edit_authorized"] is False and
            source["root_train_py_copy_or_transfer_authorized"] is False, "source authority drift")

    host = request["host"]
    require(host["name"] == "trn22" and host["expected_hostname"] == "ip-172-31-32-253" and
            host["source_checkout"] == CHECKOUT and
            host["source_commit"] == "e65a3097a5b00f6178188a935bb2acb9a5042c6c" and
            host["checkout_directory_device"] == 66305 and
            host["checkout_directory_inode"] == 2432738 and
            host["checkout_directory_uid"] == 1000 and
            host["checkout_directory_mode"] == "0775" and
            host["git_pointer_inode"] == 2432740 and
            host["git_pointer_mode"] == "0664" and
            host["git_pointer_sha256"] ==
            "dc53f6887bc5f9c2c1bbdbbd60911cdbeb076f7c57bafd796ee5b96f69ecbf1b" and
            host["linked_gitdir_inode"] == 2432736 and
            host["linked_gitdir_mode"] == "0775" and
            host["same_uid_or_privileged_race_not_excluded"] is True and
            host["require_fresh_clean_source_and_gitdir_identity_in_run_session"] is True,
            "host/source binding drift")

    queue = _pinned("research/v2/queue.json", QUEUE_SHA)
    g2 = request["g2"]
    argv = _queue_argv(queue)
    require(g2["label"] == LABEL and g2["scratch_root"] == SCRATCH and
            g2["python"] == "/home/ubuntu/workspace/native_venv/bin/python" and
            g2["train_argv"] == argv == old_gate["a2_argv"], "exact A2 argv drift")
    require(g2["cpu_only"] is True and g2["synthetic_data"] is True and
            g2["real_prepare_import"] is True and g2["real_data_flag_allowed"] is False and
            g2["steps"] == 2 and g2["nproc"] == 4 and
            (g2["micro_rows"], g2["micro_tokens"]) == (2, 256) and
            g2["timeout_seconds"] == 3600 and g2["trace_rmsprop"] is True and
            g2["keep_workdir_and_checkpoint"] is True and
            g2["expected_table_keys"] == KEYS and
            g2["expected_table_elements_each"] == 67108864 and
            g2["expected_rms_group_calls_per_rank"] == 2 and
            g2["expected_rms_checks_per_rank"] == 8 and
            g2["maximum_abs_error_over_lr"] == 0.03 and
            g2["expected_step_boundaries"] == [0, 1] and
            g2["checkpoint_required"] is True, "G2 geometry or acceptance drift")

    env = request["environment"]
    expected_env = {
        "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/bin:/bin",
        "HOME": "/home/ubuntu", "LANG": "C.UTF-8", "PYTHONPATH": CHECKOUT,
        "OMP_NUM_THREADS": "1", "TMPDIR": SCRATCH + "/tmp", "TRN_EXP_LABEL": LABEL,
        "NEURON_LOGICAL_NC_CONFIG": "2",
        "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
        "NEURON_COMPETITION_R1_SEED": "42", "NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS": "1800",
        "NEURON_COMPETITION_R1_STEP_LOG": SCRATCH + "/steps.jsonl",
    }
    require(env["clean_env_i_required"] is True and
            env["no_shell_expansion_or_inherited_science_overrides"] is True and
            env["exact_assignments"] == expected_env and
            env["additional_neuron_competition_r1_overrides_allowed"] is False and
            env["neuron_device_use_allowed"] is False, "CPU environment drift")

    capacity = request["capacity_and_preflight"]
    require(capacity["minimum_fresh_root_free_bytes"] == 34359738368 and
            capacity["minimum_fresh_memory_available_bytes"] == 68719476736 and
            capacity["minimum_fresh_root_free_inodes"] == 10000 and
            capacity["operational_root_floor_bytes"] == 6979321856 and
            capacity["scratch_peak_projection_bytes"] is None and
            capacity["require_projection_plus_floor_below_fresh_free"] is True and
            capacity["fresh_preflight_receipt"] is None and
            all(capacity[k] is True for k in (
                "require_exact_scratch_tmp_receipt_process_step_paths_absent",
                "require_unique_label_unused_in_local_and_host_ledgers",
                "require_no_host_stop_breaker_or_global_stop",
                "require_no_train_eval_g2_or_neuron_holder",
                "require_no_host_lock_holder", "require_clean_source_checkout_and_no_follow_rehash",
                "require_source_and_gitdir_identity_bridge_in_same_ssh_session",
                "require_protected_checkpoint_and_six_cache_unchanged", "one_cpu_job_per_host")),
            "preflight gate drift")
    scratch = request["scratch_contract"]
    require(scratch["parent"] == "/home/ubuntu" and
            scratch["guard_sha256"] == SCRATCH_GUARD_SHA and
            scratch["required_directory_mode"] == "0700" and
            all(scratch[k] is True for k in (
                "exact_new_root_and_tmp_from_absent",
                "no_follow_directory_identity_checks",
                "prelaunch_guard_in_g2_launcher_process",
                "tempfile_tempdir_must_equal_exact_tmp",
                "tempfile_fallback_forbidden",
                "creation_only_after_fresh_preflight_and_one_attempt_marker")) and
            scratch["prepare_receipt"] is None and
            scratch["prelaunch_guard_receipt"] is None, "scratch gate drift")
    post = request["postflight_and_result"]
    require(all(post[k] is True for k in (
                "contained_launch_wait_stop_if_needed_quiesce_required",
                "unconditional_distinct_session_readonly_postflight_required",
                "preserve_partial_scratch_on_failure",
                "retrieve_raw_run_log_rank_json_g2_receipt_step_log_checkpoint_sha_and_containment_receipts",
                "independent_numeric_result_review_required",
                "independent_raw_per_table_attribution_proof_required")) and
            all(post[k] is None for k in (
                "cpu_g2_receipt", "checkpoint_sha256", "operator_postflight_receipt",
                "independent_numeric_review",
                "independent_raw_per_table_attribution_proof")), "postflight/result gate drift")

    py = g2["python"]
    process_dir = SCRATCH + "/process"
    g2_receipt = SCRATCH + "/g2.json"
    g2_command = [
        py, CHECKOUT + "/workflow/tools/preflight_dryrun.py",
        "--train-py", CHECKOUT + "/train.py", "--prepare-dir", CHECKOUT,
        "--steps", "2", "--nproc", "4", "--micro-rows", "2", "--micro-tokens", "256",
        "--timeout", "3600", "--trace-rmsprop", "--keep", "--out", g2_receipt,
        "--", *argv,
    ]
    guard_path = HERE / "scratch_guard.py"
    require(guard_path.is_file() and not guard_path.is_symlink() and
            sha256_file(guard_path) == SCRATCH_GUARD_SHA,
            "scratch guard source bytes changed")
    guard_code = guard_path.read_text()
    clean_env = ["/usr/bin/env", "-i", *[f"{k}={expected_env[k]}" for k in sorted(expected_env)]]
    # The projection remains UNBOUND until an independent review and fresh
    # preflight supply it. This disabled command data is deliberately unable
    # to create scratch or dispatch as-is.
    guard_common = [SCRATCH, SCRATCH + "/tmp", "34359738368", "10000",
                    "UNBOUND_PROJECTION", "6979321856"]
    scratch_prepare = [*clean_env, py, "-B", "-c", guard_code,
                       "prepare", *guard_common, "--end"]
    clean_command = [*clean_env, py, "-B", "-c", guard_code,
                     "run", *guard_common, CHECKOUT + "/workflow/tools/preflight_dryrun.py",
                     *g2_command[2:]]
    contained = [py, CHECKOUT + "/workflow/tools/contained.py"]
    launch = [*contained, "launch", "--label", LABEL, "--dir", process_dir,
              "--cwd", CHECKOUT, "--cgroup", "--", *clean_command]
    wait = [*contained, "wait", "--label", LABEL, "--dir", process_dir, "--timeout", "4200"]
    stop = [*contained, "stop", "--label", LABEL, "--dir", process_dir, "--grace", "60"]
    quiesce = [*contained, "quiesce", "--label", LABEL, "--dir", process_dir,
               "--files", process_dir + "/launch.json", process_dir + "/run.log", g2_receipt]
    commands = {"prepare_scratch_after_marker": scratch_prepare,
                "launch": launch, "wait": wait, "stop_if_needed": stop, "quiesce": quiesce}
    digest = hashlib.sha256(json.dumps(commands, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"label": LABEL, "source_checkout": CHECKOUT, "scratch_root": SCRATCH,
            "train_argv": argv, "environment": expected_env, "commands": commands,
            "scratch_projection_binding": "UNBOUND_PROJECTION",
            "scratch_guard_sha256": SCRATCH_GUARD_SHA,
            "commands_sha256": digest, "dispatch_authorized": False}


def validate_step_log(raw: bytes) -> None:
    require(raw == b"0\n1\n", "expected exactly two ordered optimizer boundaries")


def validate_g2_receipt(receipt: dict, *, plan: dict | None = None) -> dict:
    """Check receipt structure; never grant per-table or final numeric PASS.

    The pinned G2 tool emits RMS check records without table identities. A
    later independent reviewer must establish an auditable raw trace-to-table
    mapping, or a separately reviewed tool change is needed. Eight same-size
    checks are not themselves four independently proven table updates.
    """
    plan = build_plan() if plan is None else plan
    require(receipt.get("schema") == "g2-preflight-v1" and receipt.get("verdict") == "PASS",
            "G2 is not a PASS receipt")
    require(receipt.get("train_py") == CHECKOUT + "/train.py" and
            receipt.get("train_py_sha256") == TRAIN_SHA and
            receipt.get("argv") == plan["train_argv"], "G2 source or argv mismatch")
    prepare = receipt.get("prepare") or {}
    require(prepare.get("dir") == CHECKOUT and prepare.get("sha256") == PREPARE_SHA and
            prepare.get("stub") is False, "G2 did not import the real pinned prepare.py")
    require(type(receipt.get("steps")) is int and receipt["steps"] == 2 and
            type(receipt.get("nproc")) is int and receipt["nproc"] == 4 and
            receipt.get("micro_batch") == [2, 256] and
            receipt.get("host") == "ip-172-31-32-253" and
            type(receipt.get("torchrun_rc")) is int and receipt["torchrun_rc"] == 0 and
            receipt.get("timed_out") is False, "G2 topology, host, or process failure")
    appended = receipt.get("appended_argv") or []
    require(appended[:6] == ["--device-type", "cpu", "--no-compile", "--num-steps", "2", "--out-dir"] and
            len(appended) == 7 and type(appended[6]) is str and
            re.fullmatch(re.escape(SCRATCH) + r"/tmp/g2-[A-Za-z0-9_]+/out", appended[6]) is not None,
            "G2 CPU-only appended argv mismatch")
    ranks = receipt.get("ranks")
    require(type(ranks) is list and len(ranks) == 4 and
            all(type(r.get("rank")) is int for r in ranks) and
            sorted(r.get("rank") for r in ranks) == [0, 1, 2, 3], "missing or repeated G2 rank")
    plan_hash = receipt.get("plan_sha256")
    require(type(plan_hash) is str and re.fullmatch(r"[0-9a-f]{64}", plan_hash) is not None,
            "missing ownership plan digest")
    summary = ranks[0].get("plan_summary")
    require(type(summary) is dict and set(summary) == {
                "elements_by_owner", "estimated_ms_by_owner", "shard_elements",
                "total_padded_bytes"}, "ownership summary fields differ from pinned source")
    counts, costs, stride = (summary["elements_by_owner"],
                             summary["estimated_ms_by_owner"], summary["shard_elements"])
    require(type(counts) is list and len(counts) == 4 and
            all(type(n) is int and n >= 0 for n in counts) and
            type(costs) is list and len(costs) == 4 and
            all(type(n) in (int, float) and math.isfinite(n) and n >= 0 for n in costs) and
            type(stride) is int and stride > 0 and stride % 128 == 0 and
            stride >= max(counts) and
            type(summary["total_padded_bytes"]) is int and
            summary["total_padded_bytes"] == stride * 4 * 4,
            "invalid ownership summary geometry")
    for rank in ranks:
        require(rank.get("status") == "PASS" and rank.get("appended_argv") == appended and
                rank.get("plan_sha256") == plan_hash and
                rank.get("plan_summary") == summary,
                "rank status, appended argv, or ownership plan mismatch")
        losses = rank.get("losses")
        require(type(losses) is list and len(losses) == 2 and
                all(type(loss) in (int, float) and math.isfinite(loss) for loss in losses),
                "missing or nonfinite rank loss")
        require(rank.get("tables") == KEYS and rank.get("unchanged_tables") == [] and
                rank.get("cross_rank_mismatches") == [] and
                type(rank.get("n_changed")) is int and rank["n_changed"] >= 4,
                "table set, update, or rank equality failed")
        trace = rank.get("rmsprop_trace") or {}
        require(type(trace.get("calls")) is int and trace["calls"] == 2 and
                type(trace.get("optimizer_steps")) is int and trace["optimizer_steps"] == 2 and
                type(trace.get("checks")) is list and len(trace["checks"]) == 8,
                "missing exact two-step/four-table RMSProp trace")
        steps_seen = []
        for check in trace["checks"]:
            require(type(check) is dict and set(check) == {
                        "step_t", "numel", "rows_touched", "max_abs_err_over_lr",
                        "untouched_rows_unchanged", "grad_abs_median_touched"},
                    "RMS trace has missing or unauthenticated table identity fields")
            err, median = check.get("max_abs_err_over_lr"), check.get("grad_abs_median_touched")
            require(type(check.get("numel")) is int and check["numel"] == 67108864 and
                    type(check.get("rows_touched")) is int and
                    0 < check["rows_touched"] <= 524288 and
                    check.get("untouched_rows_unchanged") is True and
                    type(err) in (int, float) and math.isfinite(err) and 0 <= err <= 0.03 and
                    type(median) in (int, float) and math.isfinite(median) and median >= 0,
                    "RMSProp table size, touched rows, or numeric check failed")
            steps_seen.append(check.get("step_t") if type(check.get("step_t")) is int else None)
        require(steps_seen == [1] * 4 + [2] * 4,
                "RMSProp per-table step counters mismatch")
    r0 = next(rank for rank in ranks if rank["rank"] == 0)
    ck = r0.get("checkpoint_check") or {}
    require(ck.get("ok") is True and ck.get("path_exists") is True and
            ck.get("deterministic") is True and ck.get("logits_shape") == [1, 256, 8192] and
            all(type(value) is int for value in ck["logits_shape"]) and
            all(key in r0.get("final_digests", {}) for key in KEYS),
            "model-only checkpoint strict-load/logits or table digest failed")
    return {"structural_checks": "PASS", "per_table_attribution":
            "HOLD_INDEPENDENT_RAW_PROOF_REQUIRED", "numeric_gate": "HOLD"}


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: offline A2 G2 plan only; no host or CPU action")
