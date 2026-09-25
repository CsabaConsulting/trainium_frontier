"""Mechanical Operator checks only; not independent review or launch authority."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
OLD = REPO / "research/v2/operator/20260926-trn22-arch3e79-full"
checks = 0

def check(ok, message):
    global checks
    if not ok:
        raise RuntimeError(message)
    checks += 1

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

b = json.loads((ROOT / "NATIVE_BINDING.disabled.json").read_text())
q = json.loads((ROOT / "EXPERIMENT_PROPOSAL.json").read_text())
oldb = json.loads((OLD / "NATIVE_BINDING.disabled.json").read_text())
oldlabel = oldb["label"]
newlabel = "fs-20260926-arch3e79-lr012-trn22-r1-full"
check(b["label"] == q["label"] == newlabel, "LABEL")
check(b["host"] == q["host"] == "trn22", "HOST")
check(b["stage_authorized"] is False and b["launch_authorized"] is False and q["launch_authorized"] is False, "DISABLED")
check(digest(OLD / "host_driver.py") == "c8957e5dbd67bbf017510d6110bcd0e0ab09d2183db098d18654a54f5498e7e0", "ORIGINAL_DRIVER")
check((ROOT / "host_driver.py").read_text() == (OLD / "host_driver.py").read_text().replace(oldlabel, newlabel), "DRIVER_ONLY_LABEL_PATHS")
check(digest(ROOT / "host_driver.py") == b["driver_sha256"] == q["driver_sha256"], "DRIVER_PIN")
expected = oldb["training_command"].copy()
expected[expected.index("--matrix-lr") + 1] = "0.012"
check(b["training_command"] == q["argv"] == expected, "SOLE_MATRIX_LR_DELTA")
check("--recursive-schedule" not in expected, "NO_RECURSIVE")
for flag, value in [("--seed", "42"), ("--nproc_per_node=4", None), ("--max-train-seconds", "1800"), ("--num-steps", "1000000"), ("--attention-layers", "0,2,4,6,7"), ("--seq-len", "1024"), ("--pack-factor", "32")]:
    check(flag in expected, "MISSING_" + flag)
    if value is not None:
        check(expected[expected.index(flag) + 1] == value, "VALUE_" + flag)
for name in ("command", "evaluation_command", "training_environment", "evaluation_environment", "outer_command"):
    actual = json.dumps(b[name], sort_keys=True)
    reference = json.dumps(oldb[name], sort_keys=True).replace(oldlabel, newlabel)
    check(actual == reference, "BINDING_ONLY_LABEL_PATHS_" + name)
for name in ("source_commit", "source_git_path", "source_sha256", "prepare_path", "prepare_sha256", "harness_path", "harness_sha256", "idle_guard_sha256", "dispatch_driver_sha256", "device_lock", "budgets", "evaluation"):
    check(b[name] == oldb[name], "PIN_UNCHANGED_" + name)
source = subprocess.check_output(["git", "show", b["source_commit"] + ":" + b["source_git_path"]], cwd=REPO)
check(hashlib.sha256(source).hexdigest() == b["source_sha256"], "SOURCE_GIT_PIN")
check(b["source_sha256"] == "3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c", "EXACT_3E79")
check(q["primary_comparator"] == b["same_host_control"], "COMPARATOR_BINDING")
check(q["primary_comparator"]["public20m_bpb"] == 0.9863403052402583, "ARCH_LR_CONTROL_SCORE")
check(q["primary_comparator"]["label"] == oldlabel, "ARCH_LR_CONTROL_LABEL")
check(q["secondary_comparator"]["label"] == "fs-20260925-v2-c5c-a0-control-trn22-r10", "HISTORICAL_C5C_ACTUAL_LABEL")
check("NOT" in q["secondary_comparator"]["scope"], "NO_C5C_LR_COMPARISON")
check(b["no_auto_next_job"] is True, "NO_AUTO_NEXT")
tree = ast.parse((ROOT / "host_driver.py").read_text())
values = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        v = node.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "Path":
            v = v.args[0]
        values[node.targets[0].id] = ast.literal_eval(v)
for k, name in [("ROOT", "root"), ("PREP", "prepare_path"), ("HARNESS", "harness_path"), ("LABEL", "label"), ("SOURCE_SHA", "source_sha256"), ("PREP_SHA", "prepare_sha256"), ("HARNESS_SHA", "harness_sha256"), ("TRAIN_COMMAND", "command"), ("EVAL_COMMAND", "evaluation_command"), ("TRAIN_ENV", "training_environment"), ("EVAL_ENV", "evaluation_environment")]:
    check(values[k] == b[name], "DRIVER_BINDING_" + k)
guard = REPO / "research/v2/operator/20260926-common-host-idle-guard"
check(digest(guard / "idle_guard.py") == b["idle_guard_sha256"], "GUARD_BYTES")
check(digest(guard / "dispatch_driver.py") == b["dispatch_driver_sha256"], "DISPATCH_BYTES")
print(json.dumps({"status": "OPERATOR_MECHANICAL_PASS", "checks": checks, "not_independent_review": True, "no_host_actions": True, "binding_sha256": digest(ROOT / "NATIVE_BINDING.disabled.json"), "proposal_sha256": digest(ROOT / "EXPERIMENT_PROPOSAL.json"), "driver_sha256": digest(ROOT / "host_driver.py")}, sort_keys=True))
