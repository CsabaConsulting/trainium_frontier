"""Implementer-only, stdlib packet checks; never imports the effectful host driver."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT = "f35ac4947cfcb9a06a5833d00afe098ed79f07f9"
PARENT_PATH = "research/v2/proposals/20260926-startup-adjusted-cold-five-trn22/"
SOURCE = "4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea"
PARENT_SOURCE = "36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb"
PROPOSAL_SHA = "96ac22424bf802f9884c63a6c105ee86ac8b4ea3663f861ab5bdfc422c18aacb"
checks = []


def require(condition, name):
    if not condition:
        raise RuntimeError(name)
    checks.append(name)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(commit, path):
    return subprocess.check_output(["git", "show", commit + ":" + path], cwd=REPO)


def value(argv, flag):
    return argv[argv.index(flag) + 1]


def contract(b):
    """Fixture validator, not a new runtime launcher or result acceptance gate."""
    if any(b[k] is not False for k in ("stage_authorized", "launch_authorized", "native_launch_authorized", "source_adoption", "full_run_automatic", "host_reserved")):
        raise ValueError("AUTHORITY")
    if b["source_sha256"] != SOURCE or b["host"] != "trn22" or b["expected_hostname"] != "ip-172-31-32-253":
        raise ValueError("SOURCE_HOST")
    if b["original_lock"] ["inode"] != 832589 or b["original_lock"]["device"] != 66305:
        raise ValueError("ORIGINAL_LOCK")
    argv = b["training_command"]
    if argv != b["base_training_command"] + ["--recursive-schedule"]:
        raise ValueError("ONE_FLAG")
    for key, expected in (("--max-train-seconds", "1800"), ("--num-steps", "5"), ("--seed", "42"), ("--matrix-lr", "0.015"), ("--pack-factor", "32"), ("--seq-len", "1024")):
        if value(argv, key) != expected:
            raise ValueError(key)
    if b["training_environment"]["NEURON_COMPILE_CACHE_URL"] != b["harness_injected_environment"]["NEURON_COMPILE_CACHE_DIR"]:
        raise ValueError("CACHE_ALIAS")
    if b["cold_decision"]["maximum_seconds"] != 900 or b["target_startup_seconds"] != 870:
        raise ValueError("COLD_THRESHOLD")
    if not b["source_gate"]["fullsize_cpu"].startswith("HOLD:"):
        raise ValueError("CPU_HOLD")


def main():
    b = json.loads((HERE / "CT_BINDING.disabled.json").read_text())
    p = json.loads((HERE / "EXPERIMENT_PROPOSAL.json").read_text())
    old = json.loads(blob(PARENT, PARENT_PATH + "CT_BINDING.disabled.json"))
    driver = (HERE / "host_driver.py").read_bytes()
    proposal_bytes = (HERE / "EXPERIMENT_PROPOSAL.json").read_bytes()
    require(sha(proposal_bytes) == PROPOSAL_SHA == b["proposal_sha256"], "original_proposal_bytes_preserved")
    require(proposal_bytes == blob(b["proposal_first_commit"], str((HERE / "EXPERIMENT_PROPOSAL.json").relative_to(REPO))), "proposal_first_exact_commit")
    require(sha(blob(b["source_commit"], b["source_git_path"])) == SOURCE, "child_source_blob_pin")
    require(sha(blob(p["control"]["source_commit"], b["source_git_path"])) == PARENT_SOURCE, "parent_source_blob_pin")
    old_driver = blob(PARENT, PARENT_PATH + "host_driver.py")
    require(sha(old_driver) == b["driver_derivation"]["sha256"], "parent_driver_pin")
    expected_driver = old_driver.replace(old["label"].encode(), b["label"].encode()).replace(PARENT_SOURCE.encode(), SOURCE.encode())
    require(driver == expected_driver, "entire_driver_only_label_and_source_literals")
    require(sha(driver) == b["driver_sha256"], "child_driver_pin")
    constants = {}
    tree = ast.parse(driver)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = node.value.args[0] if isinstance(node.value, ast.Call) else node.value
            constants[node.targets[0].id] = ast.literal_eval(v)
    for key, field in (("ROOT", "root"), ("LABEL", "label"), ("SOURCE_SHA", "source_sha256"), ("PREP", "prepare_path"), ("PREP_SHA", "prepare_sha256"), ("HARNESS", "harness_path"), ("HARNESS_SHA", "harness_sha256"), ("COMMAND", "command"), ("ENV", "training_environment")):
        require(constants[key] == b[field], "driver_binding_" + key)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    guards = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "require_idle_under_inherited_lock"]
    launches = [n for n in calls if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess" and n.func.attr == "call"]
    require(len(guards) == len(launches) == 1 and guards[0].lineno < launches[0].lineno, "one_guard_before_one_child")
    fd = next(k.value for k in launches[0].keywords if k.arg == "pass_fds")
    require(ast.dump(fd) == "Tuple(elts=[Name(id='lock_fd', ctx=Load())], ctx=Load())", "live_lock_fd_inherited_by_timeout")
    require(ast.literal_eval(guards[0].args[0]) == b["device_lock"] == b["original_lock"]["path"], "one_common_lock")
    require(b["outer_command"][:6] == ["/usr/bin/flock", "-n", b["device_lock"], "/usr/bin/python3", "-I", "-B"], "outer_normal_python_no_optimized_assert_bypass")
    require(b["detached_command"] == ["/usr/bin/python3", "-I", "-B", b["root"] + "/dispatch_driver.py"], "sole_existing_dispatch")
    for field, path in (("idle_guard_sha256", b["idle_guard_source"]), ("dispatch_driver_sha256", "research/v2/operator/20260926-common-host-idle-guard/dispatch_driver.py")):
        require(sha((REPO / path).read_bytes()) == b[field] == old[field], field)
    require(sha(blob(b["harness_source_commit"], b["harness_source_git_path"])) == b["harness_sha256"] == old["harness_sha256"], "unchanged_rehearsal_pin")
    require(b["prepare_sha256"] == p["prepare_sha256"] == old["prepare_sha256"], "unchanged_prepare_pin_no_host_claim")
    require(b["training_command"] == p["argv"] == old["training_command"] + ["--recursive-schedule"], "proposal_and_parent_one_science_flag")
    require(shlex.split((HERE / "package/launch-command.txt").read_text()) == p["argv"], "package_exact_argv")
    for file, pin in b["package_files"].items():
        require(sha((HERE / file).read_bytes()) == pin, "package_pin_" + file)
    provenance = json.loads((HERE / "package/provenance.json").read_text())
    require(provenance["train_sha256"] == SOURCE and provenance["source_commit"] == b["source_commit"], "package_source_provenance")
    cpu = json.loads((REPO / "research/v2/proposals/20260926-clock36-recursive-exact-cpu-r2-trn22/on/request.json").read_text())
    cpu_argv = list(cpu["argv"])
    cpu_argv[cpu_argv.index("--num-steps") + 1] = "5"
    require(cpu["source_sha256"] == SOURCE and cpu_argv == p["argv"][4:], "cpu_request_config_only_not_cpu_result")
    require(b["command"][:5] == ["/usr/bin/timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "1800s"], "unchanged_outer_1800_plus_60")
    require(b["command"][-3:] == ["--mode", "train", "--fresh-caches"], "fresh_train_only_harness")
    require(b["training_environment"]["NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS"] == "1800" and b["training_environment"]["NEURON_LOGICAL_NC_CONFIG"] == "2", "budget_and_lnc")
    cache_keys = ("TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "TORCH_NEURONX_HLO_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "NEURON_COMPILE_CACHE_DIR", "TORCH_NEURONX_DEBUG_DIR")
    cache_values = [b["harness_injected_environment"][k] for k in cache_keys]
    require(set(cache_values) == {b["sandbox"] + "/caches/" + n for n in ("neff", "local", "hlo", "inductor", "compiler", "debug")}, "six_disjoint_private_caches")
    require(b["no_eval"] and b["no_automatic_full"] and b["cold_decision"]["automatic_full"] is False, "no_eval_no_automatic_full")
    require(b["resource_floors"]["operator_preflight_disk_bytes"] == int(10.5 * 1024**3) and b["resource_floors"]["unchanged_inner_driver_disk_bytes"] == int(6.5 * 1024**3), "operator_vs_inherited_resource_floors_truthful")
    require("no internal hostname assertion" in b["host_identity"] and "mandatory" in b["host_identity"], "host_identity_operator_obligation_explicit")
    require(all(s in b["postflight"] for s in ("PPID/PGID/SID", "compiler", "832589", "66305", "HOLD")), "explicit_descendant_postflight_obligation")
    for key in ("command", "training_command", "outer_command", "detached_command"):
        require(sha(json.dumps(b[key], separators=(",", ":"), ensure_ascii=True).encode()) == b[key + "_sha256"], key + "_canonical_pin")
    contract(b)
    require(True, "positive_contract")
    mutations = {
        "authority": lambda x: x.update(launch_authorized=True),
        "source": lambda x: x.update(source_sha256=PARENT_SOURCE),
        "host": lambda x: x.update(expected_hostname="wrong-host"),
        "lock": lambda x: x["original_lock"].update(inode=832590),
        "flag": lambda x: x["training_command"].remove("--recursive-schedule"),
        "steps": lambda x: x["training_command"].__setitem__(x["training_command"].index("--num-steps") + 1, "6"),
        "cache": lambda x: x["training_environment"].update(NEURON_COMPILE_CACHE_URL="/inherited-cache"),
        "cap": lambda x: x["cold_decision"].update(maximum_seconds=901),
        "cpu_gate": lambda x: x["source_gate"].update(fullsize_cpu="PASS"),
    }
    for name, mutate in mutations.items():
        bad = copy.deepcopy(b)
        mutate(bad)
        try:
            contract(bad)
        except ValueError:
            require(True, "negative_refused_" + name)
        else:
            raise RuntimeError("NEGATIVE_ACCEPTED_" + name)
    print(json.dumps({"status": "IMPLEMENTER_STATIC_CHECKS_PASS_NOT_INDEPENDENT_REVIEW", "checks": len(checks), "names": checks, "host_actions": False, "tensor_allocations": False, "cpu_qualification": "HOLD", "native_launch_authorized": False}, indent=2))


if __name__ == "__main__":
    main()
