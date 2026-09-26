"""Authenticate fresh20M and unchanged checkpoint; preserve comparator roles."""
import argparse
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RAW_REF = "e5a29999"
ROOT = "research/v2/operator/20260926-trn22-startup-clock-full/completed/"
checks = []

def check(name, ok):
    if not ok: raise RuntimeError(name)
    checks.append(name)

def sha(raw): return hashlib.sha256(raw).hexdigest()

def git(path): return subprocess.check_output(["git", "show", RAW_REF + ":" + ROOT + path], cwd=REPO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--row", type=Path)
    args = parser.parse_args()
    saved_argv = sys.argv
    sys.argv = [str(HERE / "audit_training.py")]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path(str(HERE / "audit_training.py"), run_name="__main__")
    finally:
        sys.argv = saved_argv
    training = json.loads(out.getvalue())
    check("all65 training checks rerun", training["check_count"] == 65 and training["status"] == "PASS_TRAINING_AUTHENTICATION_ONLY")
    raw = git("results-bundle.json")
    check("canonical final bundle exact", raw == (REPO / ROOT / "results-bundle.json").read_bytes())
    bundle = json.loads(raw)
    files = bundle["files"]
    for name, item in files.items():
        data = item["text"].encode()
        check("final artifact " + name, sha(data) == item["sha256"] and len(data) == item["bytes"])
    check("complete26files410104bytes", len(files) == bundle["file_count"] == 26 and sum(x["bytes"] for x in files.values()) == bundle["raw_bytes"] == 410104)
    record = lambda n: json.loads(files[n]["text"])
    binding = record("NATIVE_BINDING.disabled.json")
    old = json.loads((REPO / "research/v2/operator/20260926-trn22-startup-clock-full/training-completed/results-bundle.json").read_text())["files"]
    for name in ("NATIVE_BINDING.disabled.json", "MANIFEST.sha256.json", "checkpoint.json", "training.log", "training-launch.json", "train/rehearsal.json", "train/steps.jsonl", "fresh-public20m-launch.json", "eval-under-lock-idle.json"):
        check("unchanged training/handoff " + name, files[name] == old[name])
    log = files["fresh-public20m.log"]["text"]
    def tagged(prefix):
        values = [json.loads(x[len(prefix):]) for x in log.splitlines() if x.startswith(prefix)]
        check("one " + prefix, len(values) == 1)
        return values[0]
    isolation = tagged("ISOLATED_SUBMISSION ")
    command = tagged("REHEARSAL_COMMAND ")
    check("isolated exact eval source/prepare", isolation["mode"] == "eval" and isolation["sandbox"] == binding["eval_sandbox"] and isolation["files"] == ["prepare.py", "train.py"] and isolation["train_sha256"] == binding["source_sha256"] and isolation["prepare_sha256"] == binding["prepare_sha256"])
    check("trained science argv preserved", shlex.split(isolation["launch_command"]) == binding["training_command"])
    expected = [binding["evaluation_command"][5], "prepare.py", "eval-public", "--train-py", "train.py", "--checkpoint", binding["sandbox"] + "/out/final.pt", "--device", "neuron", "--batch-size", "1", "--seq-len", "1024", "--eval-tokens", "20971520", "--timeout-seconds", "1200"]
    check("exact fresh20M context1024 B1", command == expected)
    metrics = []
    for match in re.finditer(r"\{", log):
        try: value, _ = json.JSONDecoder().raw_decode(log[match.start():])
        except ValueError: continue
        if isinstance(value, dict) and "total_nats" in value and "val_bpb" in value: metrics.append(value)
    check("one complete raw eval result", len(metrics) == 1)
    ev = metrics[0]
    bpb = ev["total_nats"] / (ev["num_bytes"] * math.log(2))
    check("raw totals and own BPB arithmetic", ev["num_tokens"] == 20971520 and ev["num_bytes"] == 83462186 and ev["total_nats"] == 57179468.0 and bpb == ev["val_bpb"] == 0.9883821509811542 and math.isfinite(bpb))
    check("eval timeout respected", ev["eval_over_timeout"] is False and ev["eval_seconds"] < ev["eval_timeout"] == 1200)
    receipt = record("eval/rehearsal.json")
    check("fresh process exit0", receipt == tagged("SUBMISSION_REHEARSAL ") and receipt["returncode"] == 0 and files["eval-exit-code.txt"]["text"].strip() == "0")
    terminal, outer = record("terminal.json"), record("outer-flock-exit.json")
    check("final complete without next/upload", terminal["label"] == binding["label"] and terminal["eval_exit_code"] == outer["returncode"] == 0 and terminal["no_automatic_next_job"] is True and terminal["submission_authorized"] is False and outer["command"] == binding["outer_command"])
    parse = datetime.datetime.fromisoformat
    start = re.search(r"^FRESH-PUBLIC20M_START_UTC=(.+)$", log, re.M).group(1)
    end = re.search(r"^FRESH-PUBLIC20M_END_UTC=(.+)$", log, re.M).group(1)
    check("ordered training eval outer times", parse(training["train_end_utc"]) < parse(training["fresh20M_start_utc"]) <= parse(start) < parse(end) <= parse(terminal["utc"]) < parse(outer["ended_utc"]))
    check("eval durations distinguished", ev["eval_seconds"] < receipt["wall_seconds"] <= (parse(end) - parse(start)).total_seconds())
    post_bytes, ck_bytes = git("explicit-process-postflight.json"), git("post-eval-checkpoint-hash.json")
    check("canonical external postflight", post_bytes == (REPO / ROOT / "explicit-process-postflight.json").read_bytes() and ck_bytes == (REPO / ROOT / "post-eval-checkpoint-hash.json").read_bytes())
    post, ck = json.loads(post_bytes), json.loads(ck_bytes)
    check("15 recorded identities gone", len(post["observed_pids"]) == len(post["pid_rows"]) == 15 and sorted(post["observed_pids"]) == sorted(x["pid"] for x in post["pid_rows"]) and all(x["exists"] is False for x in post["pid_rows"]) and post["all_observed_pids_absent"] is True and not post["related_rows"])
    identities = post["all_process_identity_rows"]
    check("unfiltered identity list preserved", len(identities) == 254 and all(set(("pid", "ppid", "pgid", "sid", "comm")) <= set(r) for r in identities))
    observed = set(post["observed_pids"])
    check("no surviving observed identity relationship", all(not (observed & {r[k] for k in ("pid", "ppid", "pgid", "sid")}) for r in identities))
    check("compiler and evaluator identities explicitly checked", {3191465,3191631,3213857,3213858,3213931} <= observed)
    check("postflight device quiet and lock free", post["ps_rc"] == post["lock_competitor_rc"] == 0 and post["neuron_device_idle"] is True and not post["ps_stderr"] and not post["neuron_ls_stderr"] and all(not d["neuron_processes"] for d in post["neuron_ls"]))
    check("retrieval also quiet on original lock", not bundle["job_lines"] and not bundle["workers"] and bundle["lock_competitor_rc"] == bundle["neuron_ls_rc"] == 0 and [bundle["lock_device"], bundle["lock_inode"]] == [66305, 832589] and not any(x["exists"] for x in bundle["old_queue_pids"]))
    check("checkpoint authenticated after eval", ck["status"] == "PASS" and ck["unchanged_after_eval"] is True and ck["before"] == ck["after"] and ck["sha256"] == ck["expected_training_sha256"] == training["checkpoint"]["sha256"] and ck["bytes_hashed"] == ck["before"]["size"] == training["checkpoint"]["bytes"] and ck["path"] == binding["sandbox"] + "/out/final.pt")
    check("hash held original lock and idle", [ck["lock"]["device"], ck["lock"]["inode"]] == [66305, 832589] and ck["lock"]["exclusive_held_through_hash"] and not ck["idle_checks"]["ps"]["busy"] and not ck["idle_checks"]["neuron_ls"]["process_pids"] and ck["idle_checks"]["ps"]["returncode"] == ck["idle_checks"]["neuron_ls"]["returncode"] == 0)
    check("postflight follows complete process", parse(outer["ended_utc"]) < parse(bundle["utc"]) < parse(post["utc"]) < parse(ck["started_utc"]) < parse(ck["ended_utc"]))
    ledger_path, selector_path = REPO / "research/v2/ledger.jsonl", REPO / "workflow/tools/decide.py"
    ledger_before, selector_before = ledger_path.read_bytes(), selector_path.read_bytes()
    check("selector unchanged", sha(selector_before) == "fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac")
    rows = [json.loads(x) for x in ledger_before.splitlines() if x.strip()]
    primary = next(r for r in rows if r["label"] == binding["decision_comparator"]["label"])
    parent = next(r for r in rows if r["label"] == binding["raw_parent_contrast"]["label"])
    check("original comparator roles preserved", primary["kind"] == "schedule_arm" and primary["public_20m"] == binding["decision_comparator"]["public20m_bpb"] and parent["public_20m"] == binding["raw_parent_contrast"]["public20m_bpb"] and parent["kind"] != "control")
    delta = bpb - primary["public_20m"]
    check("observed distinct-stack improvement within replicate tier", -0.005 < delta <= -0.001 and training["startup_charge_above_900_seconds"] == 0)
    row = dict(schema="trn-run-record-v2", label=binding["label"], arm_id="L2-architecture-startup-clock-36a862", kind="architecture_startup_clock_arm", host="trn22", timestamp_utc=terminal["utc"], base_sha256=binding["decision_comparator"]["source_sha256"], train_py_sha256=binding["source_sha256"], source_commit=binding["source_commit"], status="evaluated", git_dirty=False, git_dirty_scope="Frozen committed scientific source and bound launch, not unrelated working tree files", argv=binding["training_command"][4:], env=binding["training_environment"], training_budget_seconds=1800, over_budget=False, steps=training["steps"], training_seconds=1784.8, charged_seconds_est=1785.2, charged_through_exit_seconds=training["charged_through_process_exit_seconds"], process_through_exit_wall_seconds=training["process_wall_seconds"], startup_measured_seconds=training["uncapped_fifth_seconds"], startup_allowance_seconds=training["uncapped_fifth_seconds"], startup_allowance_est=805.5, cold_startup_proven=False, cold_scope="Existing research caches; separate cold-five845.6273 does not make this full run cold", causality_passed=None, run_purpose="quality", eval_tokens=20971520, eval_seq_len=1024, public_20m=bpb, eval_seconds=ev["eval_seconds"], eval_over_timeout=False, checkpoint_sha256=ck["sha256"], checkpoint_bytes=ck["bytes_hashed"], checkpoint_path=ck["path"], raw_archive=ROOT+"results-bundle.json", raw_archive_commit=RAW_REF, raw_archive_sha256=sha(raw), eval_log_bundle_key="fresh-public20m.log", eval_log_sha256=files["fresh-public20m.log"]["sha256"], training_log_bundle_key="training.log", training_log_sha256=files["training.log"]["sha256"], independent_review=str(HERE.relative_to(REPO)/"FINAL_REVIEW.md"), primary_comparator_label=primary["label"], primary_comparator_public20m=primary["public_20m"], delta_vs_primary=delta, secondary_comparator_label=parent["label"], secondary_comparator_public20m=parent["public_20m"], delta_vs_secondary=bpb-parent["public_20m"], historical_comparators_relabelled=False, formal_control_eligible=False, submission_eligible=False, submission_authorized=False, supervisor_decision="NO_ADOPTION_PREPARE_GENUINE_CONTROL_AND_CROSS_HOST_CONFIRMATION")
    spec = importlib.util.spec_from_file_location("selector_readonly", selector_path)
    selector = importlib.util.module_from_spec(spec); spec.loader.exec_module(selector)
    check("valid fixed-budget quality record", selector.invalid_reason(row) is None)
    query = [r for r in rows if r["label"] != row["label"]] + [row]
    verdict = selector.decide_arm(query, row["arm_id"])
    controls = selector.controls_for(rows, "trn22", row["base_sha256"])
    check("no retrospective formal control", not controls)
    check("unchanged selector requires actual control", verdict["verdict"] == "INSUFFICIENT" and verdict["per_host"][0]["error"] == "no valid same-host control for this base")
    row["selector_verdict"] = verdict["verdict"]
    row["formal_control_label"] = None
    row["formal_control_public20m"] = None
    row["delta_vs_formal_control"] = None
    check("ledger and selector untouched", ledger_path.read_bytes() == ledger_before and selector_path.read_bytes() == selector_before)
    result = dict(status="PASS_RAW_RESULT_AUTHENTICATION_NOT_ADOPTED", training_checks=65, final_checks=len(checks), checks=checks, public20m=bpb, delta_vs_primary=delta, delta_vs_secondary=row["delta_vs_secondary"], delta_vs_formal_control=row["delta_vs_formal_control"], evaluation=ev, selector_result=verdict, primary_role="Confirmed same-host different-stack schedule arm, not kind=control; no valid formal control exists yet", adopted=False, repeat_now=False, replication_warranted_by_human_tree=True, submission_ready=False, checkpoint_sha256=ck["sha256"], raw_commit=RAW_REF)
    if args.output: args.output.write_text(json.dumps(result, indent=2)+"\n")
    if args.row: args.row.write_text(json.dumps(row, separators=(",", ":"))+"\n")
    print(json.dumps({k: v for k,v in result.items() if k not in ("checks", "selector_result")}, indent=2))

if __name__ == "__main__": main()
