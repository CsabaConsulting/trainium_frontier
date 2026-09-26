"""Authenticate cold training accounting and live attached eval, not BPB."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research/v2/operator"
PATH = BASE / "20260926-trn22-training-handoff-0543/trn22.json"
count = 0


def check(value, message):
    global count
    if not value:
        raise RuntimeError(message)
    count += 1


raw = PATH.read_bytes()
check(hashlib.sha256(raw).hexdigest() == "f701bc30e2b16c7ac26c40e4fe9bb4b464c41069e8dcd365dd6e73c2cce20075", "snapshot pin")
record = json.loads(raw)
check(record["returncode"] == 0 and record["stderr"] == "", "successful read-only observation")
r = record["remote"]
check(r["no_changes"] is True and r["ps_rc"] == r["neuron_ls_rc"] == 0, "successful readonly process/device observation")
check((r["lock"]["device"], r["lock"]["inode"], r["lock"]["held"]) == (66305, 832589, True), "original lifetime lock retained")
check(len(r["files"]) == 19, "19 complete files")
for name, f in r["files"].items():
    b = f["text"].encode()
    check(f["capture"] == "complete_at_observation" and len(b) == f["captured_bytes"] == f["bytes_at_stat"] and hashlib.sha256(b).hexdigest() == f["sha256_captured"], name)


def get(name):
    return json.loads(r["files"][name]["text"])


t = get("train/rehearsal.json")
check(t["returncode"] == 0 and t["mode"] == "train", "successful training exit")
check(r["files"]["run-exit-code.txt"]["text"].strip() == "0", "separate zero exit receipt")
markers = [int(x) for x in r["files"]["train/steps.jsonl"]["text"].splitlines()]
obs = t["observations"]
check(markers == list(range(3628)) and t["step_boundaries"] == 3628, "all3628 completed steps")
check([x["step"] for x in obs] == markers, "all timed boundary identities match")
times = [x["observed_seconds"] for x in obs]
check(all(math.isfinite(x) and x >= 0 for x in times) and all(a <= b for a, b in zip(times, times[1:])), "finite monotonic boundary times")
startup = times[4]
check(startup == t["startup_allowance_observed"] == 859.4735238229623, "measured actual fifth boundary")
check(startup < 870 < 900, "actual startup below warning and cap")
charge = t["wall_seconds"] - min(startup, 900.0)
check(abs(charge - t["charged_through_process_exit"]) < 1e-9, "independent official through-exit arithmetic")
check(0 < charge <= 1800 and t["charged_through_process_exit_within_budget"] is True, "budget pass")
check(t["wall_seconds"] > times[-1], "exit wall includes checkpoint/teardown after last step")
ck = get("checkpoint.json")
check(ck["bytes"] == t["checkpoint_bytes"] == r["checkpoint"]["bytes"] == 453044626, "actual checkpoint size")
check(ck["sha256"] == t["checkpoint_sha256"] == "4d024cce5120189010d0017fcbf6aaac22a064506eaeb8ca98b453972997d2f2", "training-end checkpoint SHA")
check(len(t["fresh_empty_sdk_caches"]) == 6 and all(p.startswith(r["root"] + "/train/caches/") for p in t["fresh_empty_sdk_caches"].values()), "phase-private cold cache paths")
idle = get("eval-under-lock-idle.json")
check(idle["status"] == "PASS" and idle["ps"]["returncode"] == idle["neuron_ls"]["returncode"] == 0, "fresh eval phase idle checks")
check(idle["ps"]["busy"] == idle["neuron_ls"]["process_pids"] == [], "training device idle before attached eval")
check((idle["lock"]["device"], idle["lock"]["inode"], idle["lock"]["verified_existing_exclusive_lock"]) == (66305, 832589, True), "same lock through phase gap")
launch = get("fresh-public20m-launch.json")
check(launch["source_sha256"] == "e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43", "clean candidate eval source")
check(launch["prepare_sha256"] == "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb", "unchanged evaluator source")
check(launch["harness_sha256"] == "7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b", "unchanged accounting harness")
now = {p["pid"]: p for p in r["processes"]}
for pid, parent, start in ((3281909, 3281908, 101254226), (3307829, 3281909, 101519461), (3307830, 3307829, 101519461), (3307844, 3307830, 101519716)):
    p = now[pid]
    check(p["exists"] and p["identity_stable"] and (p["ppid"], p["starttime"]) == (parent, start), "actual live eval lineage " + str(pid))
argv = now[3307844]["argv"]
check(argv[1:3] == ["prepare.py", "eval-public"], "actual fresh ordinary public eval")
for key, value in (("--train-py", "train.py"), ("--checkpoint", r["checkpoint"]["path"]), ("--batch-size", "1"), ("--seq-len", "1024"), ("--eval-tokens", "20971520"), ("--device", "neuron")):
    check(argv[argv.index(key) + 1] == value, "actual eval argument " + key)
for pid in (3281925, 3281926, 3281940, 3281955, 3281956, 3281957, 3281958):
    check(now[pid]["exists"] is False, "original training process absent " + str(pid))
for name in ("training.log", "fresh-public20m.log"):
    check(not re.search(r"Traceback|RuntimeError|AssertionError|FATAL", r["files"][name]["text"]), "no observed error " + name)
check("terminal.json" not in r["files"] and "eval/rehearsal.json" not in r["files"], "eval completion not inferred")
print(json.dumps({"verdict": "ACCEPT_COLD_TRAINING_BUDGET_AND_LIVE_FRESH20M_HANDOFF_ONLY", "checks_passed": count, "optimized": bool(sys.flags.optimize), "observed_utc": r["utc"], "training_steps": 3628, "cold_startup_seconds": startup, "wall_seconds": t["wall_seconds"], "charged_through_exit_seconds": charge, "over_budget": False, "checkpoint_sha256_before_eval": ck["sha256"], "fresh_eval_started_utc": launch["utc"], "actual_eval_pid": 3307844, "public20m": None, "full_job_terminal": False, "submission_ready": False}, sort_keys=True))
