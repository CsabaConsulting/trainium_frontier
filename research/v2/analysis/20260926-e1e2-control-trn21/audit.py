#!/usr/bin/env python3
"""Read-only, stdlib-only verification of the actual trn21 control receipts."""
import hashlib
import json
import math
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RAW = "research/v2/operator/20260926-final-sprint-successors/trn21-e1e2-control-completed/results-bundle.json"
RAW_SHA = "9289bf8a94f164dccedbc1898408e7b8a02a4e3ed7cb927a44ad501b203e22d9"
BASE = "e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43"
LABEL = "fs-20260926-sprint-e1e2-control-trn21-r1"
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
        encoded = member["text"].encode()
        assert len(encoded) == member["bytes"], name
        assert sha(encoded) == member["sha256"], name
    get = lambda name: json.loads(files[name]["text"])
    binding, proposal = get("NATIVE_BINDING.json"), get("EXPERIMENT_PROPOSAL.json")
    train, evaluation = get("training-launch.json"), get("fresh-public20m-launch.json")
    rehearsal, terminal = get("train/rehearsal.json"), get("terminal.json")
    preflight = get("operator-preflight.json")
    assert proposal["proposal_id"] == "20260926-e1e2-prospective-controls"
    assert {"host": "trn21", "label": LABEL, "kind": "control"} in proposal["runs"]
    assert (ROOT / "research/v2/proposals/20260926-e1e2-prospective-controls/EXPERIMENT_PROPOSAL.json").read_bytes() == files["EXPERIMENT_PROPOSAL.json"]["text"].encode()
    assert preflight["files"]["EXPERIMENT_PROPOSAL.json"] == files["EXPERIMENT_PROPOSAL.json"]["sha256"]
    assert binding["kind"] == "control" and binding["label"] == LABEL
    assert sha(files["package/train.py"]["text"].encode()) == BASE
    assert (ROOT / "workflow/candidates/confirmed-clock36-116188/train.py").read_bytes() == files["package/train.py"]["text"].encode()
    command = files["package/launch-command.txt"]["text"].strip()
    assert command == binding["launch_command"]
    assert sha(files["package/launch-command.txt"]["text"].encode()) == proposal["candidate"]["launch_sha256"]
    assert (ROOT / "workflow/candidates/confirmed-clock36-116188/launch-command.txt").read_bytes() == files["package/launch-command.txt"]["text"].encode()
    argv = shlex.split(command)[1:]
    trainlog, evallog = files["training.log"]["text"], files["fresh-public20m.log"]["text"]
    actual_argv = json.loads(next(line.split(" ", 1)[1] for line in trainlog.splitlines() if line.startswith("REHEARSAL_COMMAND ")))
    assert actual_argv == argv
    for launch, mode in ((train, "train"), (evaluation, "eval")):
        assert launch["source_sha256"] == BASE == binding["source_sha256"]
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
    steps = [int(line) for line in files["train/steps.jsonl"]["text"].splitlines()]
    assert steps == list(range(3631)) and rehearsal["step_boundaries"] == 3631
    observed = {row["step"]: row["observed_seconds"] for row in rehearsal["observations"]}
    startup = observed[4]
    assert startup == rehearsal["startup_allowance_observed"] < 900
    charged = rehearsal["wall_seconds"] - min(startup, 900)
    assert charged == rehearsal["charged_through_process_exit"] <= 1800
    assert rehearsal["charged_through_process_exit_within_budget"] is True
    assert "over_budget:       False" in trainlog
    eval_argv = json.loads(next(line.split(" ", 1)[1] for line in evallog.splitlines() if line.startswith("REHEARSAL_COMMAND ")))
    for option, expected in (("--seq-len", "1024"), ("--batch-size", "1"), ("--eval-tokens", "20971520"), ("--device", "neuron")):
        assert eval_argv[eval_argv.index(option) + 1] == expected
    assert eval_argv[1:3] == ["prepare.py", "eval-public"]
    score = json.loads(re.search(r'^\{\n  "eval_over_timeout".*?^\}', evallog, re.M | re.S).group())
    assert score["eval_over_timeout"] is False and score["num_tokens"] == 20971520
    bpb = score["total_nats"] / (math.log(2) * score["num_bytes"])
    assert bpb == score["val_bpb"] == 0.9880953480915388
    checkpoint = get("checkpoint.json")
    for other in (get("checkpoint-after-postflight.json"), bundle["checkpoint_after_terminal"], rehearsal):
        assert other.get("sha256", other.get("checkpoint_sha256")) == checkpoint["sha256"]
        assert other.get("bytes", other.get("checkpoint_bytes")) == checkpoint["bytes"] == 453044626
    assert bundle["checkpoint_after_terminal"]["stable"] is True
    for identity in bundle["retained_input_identities"].values():
        assert identity["stable"] is True
    assert {v["sha256"] for v in bundle["retained_input_identities"].values()} == {BASE, binding["prepare_sha256"], binding["harness_sha256"]}
    assert binding["lock_identity"] == [66305, 596823]
    for name in ("under-lock-idle.json", "eval-under-lock-idle.json", "postflight-under-lock-idle.json"):
        idle = get(name)
        assert idle["status"] == "PASS" and idle["lock"]["verified_existing_exclusive_lock"] is True
        assert [idle["lock"]["device"], idle["lock"]["inode"]] == [66305, 596823]
        assert idle["ps"]["returncode"] == 0 and not idle["ps"]["busy"]
        assert idle["neuron_ls"]["returncode"] == 0 and not idle["neuron_ls"]["process_pids"]
    post = bundle["postflight"]
    assert post["all_observed_pids_absent"] is True and post["neuron_device_idle"] is True
    assert post["ps_rc"] == post["neuron_ls_rc"] == post["lock_competitor_rc"] == 0
    assert not post["busy_job_rows"] and post["fuser_rc"] == 1
    assert [post["lock_device"], post["lock_inode"]] == [66305, 596823]
    assert get("outer-flock-exit.json")["returncode"] == 0
    assert sha((ROOT / "workflow/tools/decide.py").read_bytes()) == SELECTOR_SHA
    losses = [{"step": int(step), "loss": float(loss)} for step, loss in re.findall(r"^step (\d+) \| loss (\S+)", trainlog, re.M)]
    assert losses and all(math.isfinite(row["loss"]) for row in losses)
    record = dict(schema="trn-run-record-v2", label=LABEL, arm_id="e1e2-control", kind="control", run_purpose="quality", host="trn21", timestamp_utc=terminal["utc"], base_sha256=BASE, train_py_sha256=BASE, status="evaluated", git_dirty=False, git_dirty_scope="Authenticated exact staged scientific source; no blanket checkout cleanliness claim", argv=argv, launch_command=command, env=env, training_budget_seconds=1800, over_budget=False, steps=3631, charged_through_exit_seconds=charged, startup_measured_seconds=startup, startup_allowance_seconds=startup, cold_startup_seconds=None, cold_startup_proven=False, cold_startup_evidence="Retained same-host research namespace; measured startup is not cold proof.", causality_passed=None, eval_tokens=score["num_tokens"], eval_seq_len=1024, eval_batch_size=1, public_20m=bpb, eval_num_bytes=score["num_bytes"], eval_total_nats=score["total_nats"], checkpoint_sha256=checkpoint["sha256"], checkpoint_bytes=checkpoint["bytes"], raw_archive_sha256=RAW_SHA, raw_archive=RAW, submission_eligible=False, notes="Actual prospectively declared unchanged e1e2 same-host control, exact accepted source/scored argv; complete train+fresh20M and original-lock quiescence. Warm research startup, not cold submission proof; no new official score. Control role declared in original proposal before launch, never relabelled from a candidate.")
    print(json.dumps({"verification": "PASS_ACTUAL_CONTROL_RECEIPTS", "verified_members": 39, "selector_sha256": SELECTOR_SHA, "control": record, "first_200_step_logged_losses": [row for row in losses if row["step"] <= 200], "prospective_k2_timing_reference": {"statistic": "35*131072/(t44-t9)", "t9_seconds": observed[9], "t44_seconds": observed[44], "tokens_per_second": 35 * 131072 / (observed[44] - observed[9]), "three_percent_threshold_tokens_per_second": 1.03 * 35 * 131072 / (observed[44] - observed[9])}, "limits": ["Checkpoint bytes retained remotely; local verification authenticates archived remote hash/size and stable identity, not a copied checkpoint.", "All logged losses finite; this is not a new full-tensor exactness test or causality test.", "Shared-cache startup is measured, but a cold start and actual cache hits are not proved.", "Training/eval source and launch bindings authenticated; no blanket surrounding-checkout cleanliness claim.", "Historical candidates/controls retain their original roles; only this prospectively declared actual control is appended."]}, indent=2))


if __name__ == "__main__":
    main()
