#!/usr/bin/env python3
"""Verify actual receipts and emit a selector input; never modify the ledger."""
import copy
import hashlib
import json
import math
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RAW = "research/v2/operator/20260926-final-sprint-successors/trn22-batch-lr14-completed/results-bundle.json"
RAW_SHA = "c05eaa57272719cb7ba3e2d24dae67f11ba6ca6e2de170ba7d0fbffe163bd91f"
SOURCE = "8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee"
BASE = "e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43"
LABEL = "fs-20260926-sprint-clean-batch-lr14-trn22-r1"
ARM = "e1e2-batch262144-lr021"
SELECTOR_SHA = "fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    raw = (ROOT / RAW).read_bytes()
    assert sha(raw) == RAW_SHA
    bundle = json.loads(raw)
    files = bundle["files"]
    assert len(files) == 39
    for name, member in files.items():
        data = member["text"].encode()
        assert len(data) == member["bytes"] and sha(data) == member["sha256"], name
    get = lambda name: json.loads(files[name]["text"])
    binding, proposal = get("NATIVE_BINDING.json"), get("EXPERIMENT_PROPOSAL.json")
    train, evaluation = get("training-launch.json"), get("fresh-public20m-launch.json")
    rehearsal, terminal = get("train/rehearsal.json"), get("terminal.json")
    preflight = get("operator-preflight.json")
    assert proposal["candidate_sha256"] == SOURCE and proposal["base_sha256"] == BASE
    assert "batch-lr14-trn22" in proposal["variants"]
    assert files["EXPERIMENT_PROPOSAL.json"]["text"].encode() == (ROOT / "research/v2/proposals/20260926-sprint-p0-clean-batch/EXPERIMENT_PROPOSAL.json").read_bytes()
    assert preflight["files"]["EXPERIMENT_PROPOSAL.json"] == files["EXPERIMENT_PROPOSAL.json"]["sha256"]
    assert binding["label"] == LABEL
    assert sha(files["package/train.py"]["text"].encode()) == SOURCE
    source_path = "research/v2/proposals/20260926-sprint-p0-clean-batch/variants/batch-lr14-trn22/package/train.py"
    assert (ROOT / source_path).read_bytes() == files["package/train.py"]["text"].encode()
    command = files["package/launch-command.txt"]["text"].strip()
    assert files["package/launch-command.txt"]["sha256"] == "c74a4cc45690d395827ae95fe12bb21f628adb4a83ab2cf0c15f7445bdb15b5a"
    argv = shlex.split(command)[1:]
    trainlog, evallog = files["training.log"]["text"], files["fresh-public20m.log"]["text"]
    actual_argv = json.loads(next(line.split(" ", 1)[1] for line in trainlog.splitlines() if line.startswith("REHEARSAL_COMMAND ")))
    assert actual_argv == argv
    for option, expected in (("--total-batch-size", "262144"), ("--matrix-lr", "0.021"), ("--seq-len", "1024")):
        assert argv[argv.index(option) + 1] == expected
    assert "--k2-embedding" not in argv and "--num-steps" not in argv
    for launch, mode in ((train, "train"), (evaluation, "eval")):
        assert launch["source_sha256"] == SOURCE == binding["source_sha256"]
        assert launch["label"] == LABEL
        assert launch["prepare_sha256"] == binding["prepare_sha256"] == sha((ROOT / "prepare.py").read_bytes())
        assert launch["harness_sha256"] == binding["harness_sha256"]
        assert launch["command"] == binding[mode + "_command"]
        assert launch["environment"] == binding[mode + "_environment"]
    env = train["environment"]
    assert env["NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS"] == "1800"
    assert env["NEURON_COMPETITION_R1_SEED"] == "42"
    assert env["NEURON_LOGICAL_NC_CONFIG"] == "2"
    assert "--nproc_per_node=4" in argv and "world_size:        4" in trainlog
    assert env["NEURON_CC_FLAGS"] == "--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
    assert rehearsal["returncode"] == 0 and files["run-exit-code.txt"]["text"].strip() == "0"
    assert terminal["eval_exit_code"] == 0 and files["eval-exit-code.txt"]["text"].strip() == "0"
    assert get("eval/rehearsal.json")["returncode"] == 0
    assert [int(x) for x in files["train/steps.jsonl"]["text"].splitlines()] == list(range(1921))
    assert rehearsal["step_boundaries"] == 1921
    observed = {row["step"]: row["observed_seconds"] for row in rehearsal["observations"]}
    startup = observed[4]
    assert startup == rehearsal["startup_allowance_observed"] < 900
    charged = rehearsal["wall_seconds"] - min(startup, 900)
    assert charged == rehearsal["charged_through_process_exit"] <= 1800
    assert rehearsal["charged_through_process_exit_within_budget"] is True and "over_budget:       False" in trainlog
    eval_argv = json.loads(next(line.split(" ", 1)[1] for line in evallog.splitlines() if line.startswith("REHEARSAL_COMMAND ")))
    for option, expected in (("--seq-len", "1024"), ("--batch-size", "1"), ("--eval-tokens", "20971520"), ("--device", "neuron")):
        assert eval_argv[eval_argv.index(option) + 1] == expected
    assert eval_argv[1:3] == ["prepare.py", "eval-public"]
    score = json.loads(re.search(r'^\{\n  "eval_over_timeout".*?^\}', evallog, re.M | re.S).group())
    assert score["eval_over_timeout"] is False and score["num_tokens"] == 20971520
    bpb = score["total_nats"] / (math.log(2) * score["num_bytes"])
    assert bpb == score["val_bpb"] == 0.9867730678453942
    checkpoint = get("checkpoint.json")
    for other in (get("checkpoint-after-postflight.json"), bundle["checkpoint_after_terminal"], rehearsal):
        assert other.get("sha256", other.get("checkpoint_sha256")) == checkpoint["sha256"]
        assert other.get("bytes", other.get("checkpoint_bytes")) == checkpoint["bytes"] == 453044626
    assert bundle["checkpoint_after_terminal"]["stable"] is True
    assert all(value["stable"] for value in bundle["retained_input_identities"].values())
    assert {value["sha256"] for value in bundle["retained_input_identities"].values()} == {SOURCE, binding["prepare_sha256"], binding["harness_sha256"]}
    for name in ("under-lock-idle.json", "eval-under-lock-idle.json", "postflight-under-lock-idle.json"):
        idle = get(name)
        assert idle["status"] == "PASS" and idle["lock"]["verified_existing_exclusive_lock"] is True
        assert [idle["lock"]["device"], idle["lock"]["inode"]] == [66305, 832589]
        assert idle["ps"]["returncode"] == 0 and not idle["ps"]["busy"]
        assert idle["neuron_ls"]["returncode"] == 0 and not idle["neuron_ls"]["process_pids"]
    post = bundle["postflight"]
    assert post["all_observed_pids_absent"] is True and post["neuron_device_idle"] is True
    assert post["ps_rc"] == post["neuron_ls_rc"] == post["lock_competitor_rc"] == 0
    assert not post["busy_job_rows"] and post["fuser_rc"] == 1
    assert [post["lock_device"], post["lock_inode"]] == [66305, 832589]
    assert get("outer-flock-exit.json")["returncode"] == 0
    losses = [float(x) for x in re.findall(r"^step \d+ \| loss (\S+)", trainlog, re.M)]
    assert losses and all(math.isfinite(x) for x in losses)
    assert sha((ROOT / "workflow/tools/decide.py").read_bytes()) == SELECTOR_SHA
    ledger_bytes = (ROOT / "research/v2/ledger.jsonl").read_bytes()
    rows = [json.loads(x) for x in ledger_bytes.splitlines() if x.strip()]
    prior = next(r for r in rows if r["label"] == "fs-20260926-sprint-clean-batch-lr14-trn21-r1")
    assert prior["train_py_sha256"] == SOURCE and prior["argv"] == argv and prior["launch_command"] == command
    record = copy.deepcopy(prior)
    record.update(label=LABEL, host="trn22", timestamp_utc=terminal["utc"], env=env, steps=1921, charged_through_exit_seconds=charged, startup_measured_seconds=startup, startup_allowance_seconds=startup, cold_startup_seconds=None, cold_startup_proven=False, cold_startup_evidence="Retained same-host batch/LR.015 research cache namespace; measured startup is not cold proof.", public_20m=bpb, eval_num_bytes=score["num_bytes"], eval_total_nats=score["total_nats"], checkpoint_sha256=checkpoint["sha256"], checkpoint_bytes=checkpoint["bytes"], raw_archive=RAW, raw_archive_sha256=RAW_SHA, submission_eligible=False, notes="Actual prospectively declared cross-host confirmation of joint batch262144/matrixLR.021, exact same source/argv as trn21. Complete fixed1800 train and fresh public20M/context1024; compare with genuine same-host e1e2 controls. Warm-cache research evidence, not cold submission qualification or official score.")
    relevant = [r for r in rows if (r.get("arm_id") == ARM or (r.get("kind") == "control" and r.get("base_sha256") == BASE)) and r.get("label") != LABEL]
    assert len(relevant) == 3 and len({r["label"] for r in relevant}) == 3
    relevant.append(record)
    controls = {r["host"]: r for r in relevant if r["kind"] == "control"}
    arms = {r["host"]: r for r in relevant if r["arm_id"] == ARM}
    deltas = {host: arms[host]["public_20m"] - controls[host]["public_20m"] for host in ("trn21", "trn22")}
    print(json.dumps({"verification": "PASS_ACTUAL_TWO_HOST_RECEIPTS", "verified_new_archive_members": 39, "ledger_sha256_at_analysis": sha(ledger_bytes), "canonical_ledger_modified": False, "new_actual_record": record, "selector_input_rows": relevant, "exact_comparison": {"deltas": deltas, "candidate_mean": sum(r["public_20m"] for r in arms.values()) / 2, "control_mean": sum(r["public_20m"] for r in controls.values()) / 2, "mean_delta": sum(deltas.values()) / 2, "disagreement": abs(deltas["trn21"] - deltas["trn22"])}, "source_path": source_path, "selector_sha256": SELECTOR_SHA, "sequential_completed_step_boundaries": len(observed), "printed_loss_values_checked_finite": len(losses), "measured_training_wall_seconds": rehearsal["wall_seconds"], "eval_start_utc": evaluation["utc"], "eval_seconds": score["eval_seconds"], "postflight_utc": post["utc"], "limits": ["Two-host same-seed confirmation, not independent-seed generalization or an official score.", "Warm research cache on both arm runs; exact cold full replay remains required for a submission.", "Only actually printed losses checked finite; no claim that all1921 loss values or every native tensor was observed.", "CPU/source and remote checkpoint receipts are reused at their original scope, not a new native tensor exactness test or copied-checkpoint check.", "Eight-layer schedule-only delta remains -0.00007142415259708823; stacking schedule with this batch/LR base is a new experiment, not an assumed additive gain.", "Joint batch/LR comparison does not isolate their separate causal effects."]}, indent=2))


if __name__ == "__main__":
    main()
