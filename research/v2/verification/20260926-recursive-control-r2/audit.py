#!/usr/bin/env python3
"""Read-only, tensor-free authentication of the completed same-config control."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "research/v2/operator/20260925-trn21-l3-base-repeat-r2"
SOURCE = "6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3"
PREPARE = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
COMMIT = "a4aa4351789115a2d0aa947d9ca00ff689f9bd0b"
LABEL = "fs-20260925-l3-base-repeat-trn21-r2-full"
SELECTOR = "fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac"
checks = []


def check(condition, description):
    if not condition:
        raise RuntimeError(description)
    checks.append(description)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(relative):
    return json.loads((RAW / relative).read_text())


def option(argv, key):
    check(argv.count(key) == 1, "one " + key)
    return argv[argv.index(key) + 1]


terminal = read("completed-eval/terminal.json")
train_launch = read("completed-training/training-launch.json")
eval_launch = read("completed-eval/fresh-public20m-launch.json")
train_exit = read("completed-eval/training-exit.json")
eval_exit = read("completed-eval/fresh-public20m-exit.json")
preflight = read("launched/preflight.json")
proposal = read("EXPERIMENT_PROPOSAL.priority.json")
observation = read("completed-eval/observation.json")
for name, metadata in observation["files"].items():
    data = (RAW / "completed-eval" / name).read_bytes()
    check(sha(data) == metadata["sha256"], "retrieved hash " + name)
    check(len(data) == metadata["bytes"], "retrieved bytes " + name)

committed = subprocess.check_output(
    ["git", "show", COMMIT + ":workflow/proposals/research-v2/train.py"], cwd=ROOT
)
check(sha(committed) == SOURCE, "science source matches committed bytes")
check(sha((ROOT / "prepare.py").read_bytes()) == PREPARE, "local immutable evaluator hash")
check(preflight["train_sha256"] == SOURCE, "host source hash")
check(preflight["prepare_sha256"] == PREPARE, "host evaluator hash")
check(preflight["idle"] is True and preflight["device_lock_held"] is True,
      "historical preflight idle and lock receipt")
check(preflight["disk_free_bytes"] >= 6.5 * 1024**3, "historical disk gate")
check(terminal["label"] == LABEL and terminal["host"] == "trn21", "label and host")
check(terminal["kind"] == "control", "prospectively designated control")
check(terminal["train_py_sha256"] == terminal["base_sha256"] == SOURCE, "base/source identity")
check(terminal["prepare_sha256"] == PREPARE, "terminal evaluator identity")
check(train_exit["exit_code"] == eval_exit["exit_code"] == 0, "train and eval exit zero")
check(train_exit["utc"] < eval_launch["utc"] < eval_exit["utc"], "fresh eval after train exit")
train_argv = train_launch["command"]
check(train_argv[:4] == ["/home/ubuntu/workspace/native_venv/bin/torchrun",
      "--standalone", "--nproc_per_node=4", "train.py"], "full chip four-rank launch")
check(train_argv[4:] == terminal["argv"] == terminal["base_science_argv"], "exact training argv")
check(option(terminal["argv"], "--max-train-seconds") == "1800", "declared 1800 budget")
check(option(terminal["argv"], "--seed") == "42", "seed42")
check("--recursive-schedule" in terminal["argv"], "adopted Recursive schedule")
check(option(terminal["argv"], "--num-steps") == "1000000", "not a step-limited smoke")
check(terminal["argv"] == terminal["same_host_control"]["argv"], "same-host science argv match")
check(terminal["same_host_control"]["source_sha256"] == SOURCE, "same-host source match")
check(train_launch["environment"]["NEURON_LOGICAL_NC_CONFIG"] == "2", "LNC2")
check(train_launch["environment"]["TRN_EXP_LABEL"] == LABEL, "environment label")
check(sha((RAW / "EXPERIMENT_PROPOSAL.priority.json").read_bytes()) == terminal["proposal_sha256"],
      "proposal bound in terminal")

eval_argv = eval_launch["command"]
check(eval_argv[2] == "eval-public", "unaltered public evaluator entrypoint")
check(option(eval_argv, "--checkpoint") == terminal["checkpoint_uri"], "checkpoint path binding")
check(option(eval_argv, "--train-py").endswith("/" + LABEL + "/train.py"), "eval exact run source path")
check(option(eval_argv, "--seq-len") == "1024", "public context1024")
check(option(eval_argv, "--eval-tokens") == "20971520", "public20M argv")
eval_text = (RAW / "completed-eval/fresh-public20m.log").read_text()
evaluation = json.loads(eval_text[eval_text.index("{\n"):])
check(evaluation == terminal["evaluation"], "raw eval JSON matches terminal")
check(evaluation["num_tokens"] == 20971520 and terminal["eval_tokens"] == 20971520,
      "actual public20M token count")
check(terminal["eval_seq_len"] == 1024, "terminal context1024")
check(evaluation["eval_over_timeout"] is False, "eval within timeout")
check(math.isclose(evaluation["total_nats"] / evaluation["num_bytes"] / math.log(2),
                   evaluation["val_bpb"], rel_tol=0, abs_tol=1e-14), "BPB arithmetic")
check(terminal["public_20m"] == evaluation["val_bpb"], "terminal BPB")
check(re.fullmatch("[0-9a-f]{64}", terminal["checkpoint_sha256"]) is not None,
      "archived host checkpoint SHA present (not a local checkpoint rehash)")

log = (RAW / "completed-training/training.log").read_text()
metrics = terminal["metrics"]
for key in ("training_seconds", "charged_seconds_est", "startup_allowance_est", "num_steps"):
    found = re.findall(r"^" + key + r":\s+([0-9.]+)\s*$", log, re.M)
    check(len(found) == 1 and float(found[0]) == metrics[key], "raw training metric " + key)
check(re.search(r"^over_budget:\s+False\s*$", log, re.M) is not None,
      "raw over_budget false")
check(metrics["over_budget"] is False and 0 < metrics["charged_seconds_est"] <= 1800,
      "source-accounted budget pass")
check(metrics["num_steps"] == terminal["steps"] == 3662, "completed steps")
check(metrics["world_size"] == 4 and metrics["seq_len"] == 1024, "actual world size and sequence")

row = {
    "schema": "trn-run-record-v2", "label": LABEL, "arm_id": terminal["arm_id"],
    "kind": "control", "run_purpose": "quality", "host": "trn21",
    "timestamp_utc": eval_exit["utc"], "base_sha256": SOURCE, "train_py_sha256": SOURCE,
    "git_commit": COMMIT, "git_dirty": False,
    "git_dirty_scope": "Exact committed science bytes, not a globally clean host checkout",
    "argv": terminal["argv"], "env": train_launch["environment"],
    "status": "evaluated", "training_budget_seconds": 1800,
    "budget_declaration_evidence": "completed-training/training-launch.json and raw training summary",
    "over_budget": False, "steps": metrics["num_steps"],
    "training_seconds": metrics["training_seconds"], "charged_seconds_est": metrics["charged_seconds_est"],
    "charged_through_exit_seconds": None, "startup_measured_seconds": None,
    "startup_allowance_est": metrics["startup_allowance_est"], "cold_startup_proven": False,
    "causality_passed": None, "eval_tokens": evaluation["num_tokens"], "eval_seq_len": 1024,
    "public_20m": evaluation["val_bpb"], "checkpoint_sha256": terminal["checkpoint_sha256"],
    "checkpoint_uri": terminal["checkpoint_uri"],
    "checkpoint_status": "Retained on host; SHA from retrieved terminal receipt, no local rehash",
    "same_config_prior_public_20m": terminal["same_host_control"]["public_val_bpb"],
    "same_config_delta": evaluation["val_bpb"] - terminal["same_host_control"]["public_val_bpb"],
    "verdict": "CONTROL_RECORDED_NO_NEW_TECHNIQUE", "submission_eligible": False,
    "independent_review": "research/v2/verification/20260926-recursive-control-r2/REVIEW.md",
    "raw_evidence": "research/v2/operator/20260925-trn21-l3-base-repeat-r2/",
    "notes": "Same configuration repeat, not a new adopted gain. Warm research cache; actual uncapped startup unknown. Do not transplant the separate 855.072-second cold diagnostic. Frozen confirmed base mean unchanged."
}
selector_path = ROOT / "workflow/tools/decide.py"
check(sha(selector_path.read_bytes()) == SELECTOR, "unchanged fixed-budget selector hash")
spec = importlib.util.spec_from_file_location("control_selector", selector_path)
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)
check(selector.invalid_reason(row) is None, "selector accepts authenticated control record")
print(json.dumps({"status": "PASS", "checks": checks, "row": row,
                  "scope": "Offline result authentication only; no new native execution or cold-start proof"}, indent=2))
