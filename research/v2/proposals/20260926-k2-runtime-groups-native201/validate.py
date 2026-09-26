#!/usr/bin/env python3
"""Check already accepted local proposal bindings; stdlib, read-only, no gates."""
import hashlib
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
proposal = json.loads((HERE / "EXPERIMENT_PROPOSAL.json").read_text())
inputs = json.loads((HERE / "INPUTS.sha256.json").read_text())
checks = []


def check(ok, name):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)


for name, expected in inputs.items():
    check(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, "input " + name)
candidate = proposal["candidate"]
check(inputs[candidate["source"]] == candidate["source_sha256"], "candidate source")
check(inputs[candidate["parent_source"]] == candidate["parent_source_sha256"], "parent source")
command = candidate["launch_command"]
check(hashlib.sha256(command.encode()).hexdigest() == candidate["launch_command_sha256"], "exact native201 argv")
args = shlex.split(command)
check(args[args.index("--num-steps") + 1] == "201" and "--k2-embedding" in args, "201 ON argv")
check(candidate["total_batch_size"] == 131072 and candidate["seed"] == 42 and candidate["max_train_seconds"] == 1800, "science constants")
check(proposal["dispatch"]["max_total_native_runs"] == 1 and not proposal["dispatch"]["launch_authorized"] and not proposal["dispatch"]["stage_authorized"], "one run and preparation only")
check(proposal["timing"]["continuation_threshold"] == 1.03, "net threshold")
check(proposal["accounting"]["training_budget_seconds"] == 1800 and proposal["accounting"]["startup_cap_seconds"] == 900, "unchanged budget and cap")
for key in ("primary", "alternative"):
    control = proposal["control"][key]
    rate = 35 * 131072 / (control["t44_seconds"] - control["t9_seconds"])
    check(abs(rate - control["primary_tokens_per_second"]) < 1e-9, control["host"] + " fixed rate")
    check(abs(1800 * rate - control["baseline_net_tokens"]) < 1e-6, control["host"] + " base net")
    check(abs(1.03 * control["baseline_net_tokens"] - control["three_percent_net_threshold"]) < 1e-6, control["host"] + " continuation net")
    bundle = json.loads((ROOT / control["raw_archive"]).read_text())
    files = bundle["files"]
    check(files["package/train.py"]["sha256"] == control["source_sha256"], control["host"] + " exact source")
    rehearsal = json.loads(files["train/rehearsal.json"]["text"])
    times = {r["step"]: r["observed_seconds"] for r in rehearsal["observations"]}
    check(times[9] == control["t9_seconds"] and times[44] == control["t44_seconds"], control["host"] + " raw boundaries")
print(json.dumps({"status": "PASS_LOCAL_BINDINGS_ONLY_NO_LAUNCH", "checks": len(checks),
                  "proposal_sha256": hashlib.sha256((HERE / "EXPERIMENT_PROPOSAL.json").read_bytes()).hexdigest(),
                  "bound_input_count": len(inputs), "source_sha256": candidate["source_sha256"],
                  "launch_command_sha256": candidate["launch_command_sha256"],
                  "heavy_processes": False, "host_calls": False, "checks_passed": checks}, indent=2))
