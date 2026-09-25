"""Offline result authentication and in-memory unchanged-selector query.

No ledger/history mutation, model import, host call or training.
"""
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
FINAL_COMMIT = "14f8cd6f"
BASE = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREP = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
SELECTOR = "fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac"
LEDGER = "a6ea564fd159ab165673286ba6806ef224a9f8373d18e60189981cbecf3069fa"
PREFIX = "research/v2/operator/20260925-trn21-scalar-matrixlr0175-r1/"
LABEL = "fs-20260925-fp-c5c-matrixlr0175-trn21-r1"
checks = []
evidence = {}


def require(ok, name):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)


def raw(path, commit=None):
    if commit:
        return subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT)
    return (ROOT / path).read_bytes()


def j(path, commit=None):
    return json.loads(raw(path, commit))


for subdir, commit in [("completed-training/", None), ("final-evaluation/", FINAL_COMMIT)]:
    receipt = j(PREFIX + subdir + "retrieval.json", commit)
    if commit:
        require(receipt["missing"] == [], "complete_final_retrieval")
    for filename, expected in receipt["files"].items():
        path = PREFIX + subdir + filename
        data = raw(path, commit)
        digest = hashlib.sha256(data).hexdigest()
        require(len(data) == expected["bytes"] and digest == expected["sha256"], "hash:" + subdir + filename)
        evidence[path] = digest

terminal = j(PREFIX + "final-evaluation/terminal.json", FINAL_COMMIT)
launch = j(PREFIX + "completed-training/training-launch.json")
train_exit = j(PREFIX + "completed-training/training-exit.json")
eval_launch = j(PREFIX + "final-evaluation/fresh-public20m-launch.json", FINAL_COMMIT)
eval_exit = j(PREFIX + "final-evaluation/fresh-public20m-exit.json", FINAL_COMMIT)
preflight = j(PREFIX + "completed-training/preflight.json")
proposal_path = "research/v2/proposals/20260925-final-push-overnight/" + LABEL + "/EXPERIMENT_PROPOSAL.json"
proposal = j(proposal_path)
require(hashlib.sha256(raw(proposal_path)).hexdigest() == terminal["proposal_sha256"], "bound_proposal_hash")
require(terminal["label"] == launch["label"] == LABEL and terminal["status"] == "evaluated_pending_review", "unique_label_terminal")
require(terminal["base_sha256"] == terminal["train_py_sha256"] == preflight["train_sha256"] == BASE,
        "old_c5c_not_recursive_source")
require(terminal["prepare_sha256"] == preflight["prepare_sha256"] == PREP, "unchanged_prepare_binding")
require(hashlib.sha256(raw("submissions/research_v2/train.py")).hexdigest() == BASE and
        hashlib.sha256(raw("prepare.py")).hexdigest() == PREP, "exact_local_science_hashes")
require(launch["command"][1:4] == ["--standalone", "--nproc_per_node=4", "train.py"] and
        launch["command"][4:] == terminal["argv"] == proposal["queue_binding"]["argv"], "exact_training_argv")
base_args = terminal["base_science_argv"]
expected_science = base_args.copy()
expected_science[expected_science.index("--matrix-lr") + 1] = "0.0175"
require(terminal["argv"] == expected_science + ["--seed", "42", "--max-train-seconds", "1800",
        "--num-steps", "1000000", "--out-dir", "out"], "sole_matrix_lr_science_change")
require(preflight["idle"] is True and preflight["device_lock_held"] is True, "archived_idle_locked_preflight")
require(train_exit["exit_code"] == eval_exit["exit_code"] == 0, "both_process_exits_zero")
for key, value in [("NEURON_LOGICAL_NC_CONFIG", "2"), ("NEURON_COMPETITION_R1_SEED", "42"),
                   ("NEURON_CC_FLAGS", "--optlevel=1 --auto-cast matmult --auto-cast-type bf16")]:
    require(launch["environment"][key] == value, "training_env:" + key)

train_log = raw(PREFIX + "completed-training/training.log").decode()
for key, value in terminal["metrics"].items():
    line = re.search(r"^" + re.escape(key) + r":\s*(\S+)", train_log, re.M)
    require(line is not None and line.group(1) == str(value if not isinstance(value, bool) else value), "raw_train_metric:" + key)
losses = [float(x) for x in re.findall(r"^step \d+ \| loss (\S+)", train_log, re.M)]
require(bool(losses) and all(math.isfinite(x) for x in losses) and "Traceback" not in train_log,
        "finite_logged_losses_no_traceback")
eval_text = raw(PREFIX + "final-evaluation/fresh-public20m.log", FINAL_COMMIT).decode()
evaluation = json.loads(eval_text[eval_text.index("{\n"):])
require(evaluation == terminal["evaluation"], "raw_eval_matches_terminal")
require(evaluation["num_tokens"] == 20971520 and evaluation["num_bytes"] == 83462186 and
        evaluation["eval_over_timeout"] is False and evaluation["eval_seconds"] <= 1200,
        "full_finite_public_20m_no_timeout")
bpb = evaluation["total_nats"] / (evaluation["num_bytes"] * math.log(2))
require(math.isfinite(bpb) and abs(bpb - evaluation["val_bpb"]) < 1e-14 and
        bpb == terminal["public_20m"], "independent_bpb_arithmetic")
eargv = eval_launch["command"]
require(eargv[eargv.index("--checkpoint") + 1] == terminal["checkpoint_uri"] and
        eargv[eargv.index("--seq-len") + 1] == "1024" and
        eargv[eargv.index("--eval-tokens") + 1] == "20971520" and
        eargv[eargv.index("--batch-size") + 1] == "1" and
        eargv[eargv.index("--timeout-seconds") + 1] == "1200" and "eval-public" in eargv,
        "fresh_eval_checkpoint_and_shape")
require(bool(re.fullmatch("[0-9a-f]{64}", terminal["checkpoint_sha256"])), "checkpoint_hash_present")
parse_time = dt.datetime.fromisoformat
require(parse_time(eval_launch["utc"]) > parse_time(train_exit["utc"]), "separate_eval_after_training_exit")
process_wall = (parse_time(train_exit["utc"]) - parse_time(launch["utc"])).total_seconds()

selector_path = ROOT / "workflow/tools/decide.py"
ledger_path = ROOT / "research/v2/ledger.jsonl"
require(hashlib.sha256(selector_path.read_bytes()).hexdigest() == SELECTOR, "unchanged_selector_hash")
require(hashlib.sha256(ledger_path.read_bytes()).hexdigest() == LEDGER, "frozen_full_ledger_hash")
spec = importlib.util.spec_from_file_location("unchanged_decide", selector_path)
decide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decide)
rows = decide.load(str(ledger_path))
require(len(rows) == 10 and all(r["label"] != LABEL for r in rows), "all_current_rows_without_candidate")
controls = decide.controls_for(rows, "trn21", BASE)
require(len(controls) == 1 and controls[0]["label"] == terminal["same_host_control"]["run_label"], "correct_old_base_control_only")
control = controls[0]
require(base_args == control["argv"], "control_science_argv_exact")
control_text = raw(control["eval_log"]).decode()
require(hashlib.sha256(raw(control["eval_log"])).hexdigest() == control["eval_log_sha256"], "control_raw_eval_hash")
control_eval, _ = json.JSONDecoder().raw_decode(control_text[control_text.index("{\n"):])
require(control_eval["total_nats"] / (control_eval["num_bytes"] * math.log(2)) == control["public_20m"], "control_bpb_arithmetic")

# Construct one reviewed query input in memory. No historical row is changed and
# nothing is appended to the ledger. Actual external charged time stays null.
candidate = {"label": LABEL, "arm_id": terminal["arm_id"], "kind": terminal["kind"], "host": "trn21",
             "timestamp_utc": terminal["utc"], "base_sha256": BASE, "train_py_sha256": BASE,
             "status": "evaluated", "git_dirty": False,
             "git_dirty_scope": "Exact committed science bytes, not whole host checkout",
             "argv": terminal["argv"], "env": launch["environment"], "training_budget_seconds": 1800,
             "over_budget": terminal["metrics"]["over_budget"], "steps": terminal["steps"],
             "training_seconds": terminal["metrics"]["training_seconds"],
             "charged_seconds_est": terminal["metrics"]["charged_seconds_est"],
             "charged_through_exit_seconds": None, "startup_measured_seconds": None,
             "startup_allowance_est": terminal["metrics"]["startup_allowance_est"],
             "process_through_exit_wall_seconds": process_wall,
             "cold_startup_proven": False, "causality_passed": None,
             "eval_tokens": evaluation["num_tokens"], "eval_seq_len": terminal["eval_seq_len"],
             "public_20m": bpb, "checkpoint_sha256": terminal["checkpoint_sha256"]}
require(decide.invalid_reason(candidate) is None, "selector_accepts_reviewed_query_scope")
verdict = decide.decide_arm(rows + [candidate], candidate["arm_id"])
delta = bpb - control["public_20m"]
require(verdict["verdict"] == "DISCARD" and verdict["per_host"][0]["delta"] == round(delta, 7), "selector_discard_worse_than_threshold")
require(hashlib.sha256(ledger_path.read_bytes()).hexdigest() == LEDGER and
        hashlib.sha256(selector_path.read_bytes()).hexdigest() == SELECTOR, "ledger_selector_untouched")
print(json.dumps({"audit_status": "PASS_RESEARCH_RESULT_AUDIT_NOT_OFFICIAL_TIMING_CERTIFICATION",
                  "check_count": len(checks), "checks": checks, "source_sha256": BASE,
                  "prepare_sha256": PREP, "selector_sha256": SELECTOR, "ledger_sha256": LEDGER,
                  "historical_rows": len(rows), "new_query_rows": 1, "candidate": candidate,
                  "control_bpb": control["public_20m"], "delta_unrounded": delta,
                  "training_launch_to_exit_utc_span_seconds": process_wall,
                  "actual_external_charged_seconds": None, "selector_result": verdict,
                  "evidence_sha256": evidence}, indent=2))
