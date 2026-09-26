"""Authenticate frozen full-run training and evaluator handoff, not quality."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess

REPO = Path(__file__).resolve().parents[4]
RAW_REF = "6b292621"
RAW = "research/v2/operator/20260926-trn22-startup-clock-full/training-completed/results-bundle.json"
PACKET_REF = "59d21dde"
PACKET = "research/v2/proposals/20260926-startup-adjusted-full-trn22/"
checks = []

def check(name, ok):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)

def git(ref, path):
    return subprocess.check_output(["git", "show", ref + ":" + path], cwd=REPO)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = git(RAW_REF, RAW)
    check("canonical raw byte identity", raw == (REPO / RAW).read_bytes())
    check("pinned full36 training bundle", sha(raw) == "34cf54017543997d7d5673e023601d02ecb6069cfb3a17219970bcf2cee08911")
    bundle = json.loads(raw)
    files = bundle["files"]
    for name, item in files.items():
        data = item["text"].encode()
        check("embedded artifact " + name, len(data) == item["bytes"] and sha(data) == item["sha256"])
    check("exact 22-file 408769-byte archive", len(files) == bundle["file_count"] == 22 and sum(x["bytes"] for x in files.values()) == bundle["raw_bytes"] == 408769)
    text = lambda name: files[name]["text"]
    record = lambda name: json.loads(text(name))
    binding = record("NATIVE_BINDING.disabled.json")
    check("frozen binding bytes", text("NATIVE_BINDING.disabled.json").encode() == git(PACKET_REF, PACKET + "NATIVE_BINDING.disabled.json"))
    check("frozen manifest bytes", text("MANIFEST.sha256.json").encode() == git(PACKET_REF, PACKET + "MANIFEST.sha256.json"))
    check("source Git pin", sha(git(binding["source_commit"], binding["source_git_path"])) == binding["source_sha256"])
    check("driver Git pin", sha(git(PACKET_REF, PACKET + "host_driver.py")) == binding["driver_sha256"])
    check("prepare unchanged", sha((REPO / "prepare.py").read_bytes()) == binding["prepare_sha256"])
    check("harness unchanged", sha(git("00f2ca7f", "research/scratch/submission_rehearsal.py")) == binding["harness_sha256"])
    for name, field in (("idle_guard.py", "idle_guard_sha256"), ("dispatch_driver.py", "dispatch_driver_sha256")):
        check("shared support " + name, sha((REPO / "research/v2/operator/20260926-common-host-idle-guard" / name).read_bytes()) == binding[field])
    log = text("training.log")
    def prefixed(prefix):
        values = [json.loads(x[len(prefix):]) for x in log.splitlines() if x.startswith(prefix)]
        check("one " + prefix, len(values) == 1)
        return values[0]
    isolated = prefixed("ISOLATED_SUBMISSION ")
    command = prefixed("REHEARSAL_COMMAND ")
    summary = prefixed("SUBMISSION_REHEARSAL ")
    receipt = record("train/rehearsal.json")
    check("summary equals raw receipt", summary == {k: v for k, v in receipt.items() if k != "observations"})
    check("exact effective scientific command", command == binding["training_command"] == shlex.split(isolated["launch_command"]))
    check("isolated source and prepare identity", isolated["train_sha256"] == binding["source_sha256"] and isolated["prepare_sha256"] == binding["prepare_sha256"])
    check("Startup clock only, no Recursive", "--startup-adjusted-schedule" in command and "--recursive-schedule" not in command)
    check("seed42 MAX1800 four ranks", command[command.index("--seed") + 1] == "42" and command[command.index("--max-train-seconds") + 1] == "1800" and "--nproc_per_node=4" in command)
    check("existing research cache not cold claim", receipt["fresh_empty_sdk_caches"] == isolated["fresh_empty_sdk_caches"] == {} and "--fresh-caches" not in binding["command"])
    observations = receipt["observations"]
    steps = [int(x) for x in text("train/steps.jsonl").splitlines()]
    check("3630 consecutive observed boundaries", steps == list(range(3630)) == [x["step"] for x in observations] and receipt["step_boundaries"] == 3630)
    check("monotonic finite observations", all(math.isfinite(x["observed_seconds"]) for x in observations) and all(0 <= a["observed_seconds"] <= b["observed_seconds"] for a, b in zip(observations, observations[1:])))
    fifth = observations[4]["observed_seconds"]
    allowance = min(fifth, 900.0)
    charged = receipt["wall_seconds"] - allowance
    check("actual fifth and capped allowance", fifth == 807.6938910679892 and allowance == receipt["startup_allowance_observed"])
    check("official accounting through process exit", charged == receipt["charged_through_process_exit"] == 1789.505140539026 and charged <= 1800 and receipt["charged_through_process_exit_within_budget"] is True)
    check("training exit0", receipt["returncode"] == 0 and text("run-exit-code.txt").strip() == "0")
    checkpoint = record("checkpoint.json")
    check("retained checkpoint identity", checkpoint["sha256"] == receipt["checkpoint_sha256"] == "f22d8287c6fb29a7d97344d6c3b3d63492bb7d4b58fc4fbf91039e483bf6e20e" and checkpoint["bytes"] == receipt["checkpoint_bytes"] == 453048594)
    check("source over_budget False", re.search(r"^over_budget:\s*False$", log, re.M) is not None)
    for name, expected in (("num_steps", 3630), ("world_size", 4), ("lnc", 2), ("seq_len", 1024)):
        match = re.search(r"^" + name + r":\s*(\d+)$", log, re.M)
        check("source " + name, match is not None and int(match.group(1)) == expected)
    rows = re.findall(r"^step\s+(\d+)\s+\| loss\s+(\S+)\s+\| lrm\s+(\S+)\s+\| dt\s+(\S+)s\s+\| tok/s\s+([\d,]+)", log, re.M)
    check("printed losses LR time finite", len(rows) > 360 and all(all(math.isfinite(float(v)) for v in row[1:4]) for row in rows))
    throughput = [int(row[4].replace(",", "")) for row in rows if int(row[0]) >= 100]
    parse = datetime.datetime.fromisoformat
    start = re.search(r"^TRAINING_START_UTC=(.*)$", log, re.M).group(1)
    end = re.search(r"^TRAINING_END_UTC=(.*)$", log, re.M).group(1)
    check("training UTC encloses harness", (parse(end) - parse(start)).total_seconds() >= receipt["wall_seconds"])
    for name, command_field, env_field in (("training-launch.json", "command", "training_environment"), ("fresh-public20m-launch.json", "evaluation_command", "evaluation_environment")):
        launch = record(name)
        check("bound launch " + name, launch["command"] == binding[command_field] and launch["environment"] == binding[env_field] and all(launch[k] == binding[k] for k in ("label", "source_sha256", "prepare_sha256", "harness_sha256")))
    train_guard, eval_guard = record("under-lock-idle.json"), record("eval-under-lock-idle.json")
    check("unchanged original lifetime lock", train_guard["lock"] == eval_guard["lock"] and [train_guard["lock"][k] for k in ("device", "inode")] == [66305, 832589] and train_guard["lock"]["path"] == binding["device_lock"])
    for name, guard in (("train", train_guard), ("eval", eval_guard)):
        check("fresh lock+ps+neuron-ls " + name, guard["status"] == "PASS" and guard["lock"]["verified_existing_exclusive_lock"] and guard["ps"]["returncode"] == guard["neuron_ls"]["returncode"] == 0 and not guard["ps"]["busy"] and not guard["neuron_ls"]["process_pids"])
    launch = record("fresh-public20m-launch.json")
    check("ordered eval handoff", parse(end) < parse(checkpoint["utc"]) < parse(eval_guard["started_utc"]) <= parse(eval_guard["ended_utc"]) < parse(launch["utc"]))
    check("fresh20M intended binding", binding["evaluation"]["tokens"] == 20971520 and binding["evaluation"]["seq_len"] == 1024 and binding["evaluation"]["batch_size"] == 1)
    workers = bundle["workers"]
    expected = ["/home/ubuntu/workspace/native_venv/bin/python", "prepare.py", "eval-public", "--train-py", "train.py", "--checkpoint", binding["sandbox"] + "/out/final.pt", "--device", "neuron", "--batch-size", "1", "--seq-len", "1024", "--eval-tokens", "20971520", "--timeout-seconds", "1200"]
    check("actual isolated evaluator command", len(workers) == 1 and shlex.split(workers[0]["argv"]) == expected and workers[0]["cwd"] == binding["eval_sandbox"])
    check("actual held lock during evaluation", bundle["lock_competitor_rc"] == 1 and [bundle["lock_device"], bundle["lock_inode"]] == [66305, 832589])
    check("selector unchanged", sha((REPO / "workflow/tools/decide.py").read_bytes()) == "fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac")
    graphs = {}
    for rank in range(4):
        graph_rows = list(csv.DictReader(io.StringIO(text(f"metrics-training/metric_table_graph_stats_rank{rank}.csv"))))
        graphs[str(rank)] = {"rows": len(graph_rows), "columns": list(graph_rows[0]) if graph_rows else []}
    result = {"status": "PASS_TRAINING_AUTHENTICATION_ONLY", "quality": "PENDING_FRESH20M", "check_count": len(checks), "checks": checks, "raw_commit": RAW_REF, "bundle_sha256": sha(raw), "label": binding["label"], "source_sha256": binding["source_sha256"], "steps": len(steps), "uncapped_fifth_seconds": fifth, "startup_charge_above_900_seconds": max(0.0, fifth - 900), "process_wall_seconds": receipt["wall_seconds"], "charged_through_process_exit_seconds": charged, "budget_headroom_seconds": 1800 - charged, "printed_steady_tok_s_median": statistics.median(throughput), "printed_steady_samples": len(throughput), "train_end_utc": end, "fresh20M_start_utc": launch["utc"], "checkpoint": checkpoint, "checkpoint_downloaded": False, "graph_csv_inventory_only": graphs, "cache_hits_proven": False, "cold_full_replay": False, "job_quiescence": "NOT_YET: fresh evaluator active at retrieval", "submission_ready": False}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "graph_csv_inventory_only")}, indent=2))

if __name__ == "__main__":
    main()
