#!/usr/bin/env python3
"""Local-only contract check for a pending warm S0 native request. Never dispatches."""
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
PKG = Path(__file__).resolve().parent
LABEL = "fs-20260925-s0-warm-native-trn22-r1"
S0_SHA = "ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb"
PREPARE_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
FLOOR = 13 * 1024**3 // 2
PRIOR_DENSE_GROWTH = 1_854_160_896
CACHE_ROOT = (
    "/home/ubuntu/trainium_freshstart_20260924_current_base_control_trn22_r1/"
    "out/fs-20260924-current-base-control-trn22-r1-work/train/caches"
)
CACHE_ENV = {
    "TORCH_NEURONX_NEFF_CACHE_DIR": "neff",
    "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": "local",
    "TORCH_NEURONX_HLO_CACHE_DIR": "hlo",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "NEURON_COMPILE_CACHE_DIR": "compiler",
    "TORCH_NEURONX_DEBUG_DIR": "debug",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha(commit, relative):
    data = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    return hashlib.sha256(data).hexdigest()


proposal = json.loads((PKG / "EXPERIMENT_PROPOSAL.json").read_text())
request = json.loads((PKG / "request-template.json").read_text())

assert sha(ROOT / "train.py") == S0_SHA
assert sha(ROOT / "submissions/candidates/S0/train.py") == S0_SHA
assert sha(ROOT / "prepare.py") == PREPARE_SHA
assert proposal["candidate"]["run_label"] == LABEL
assert proposal["status"].endswith("pending-independent-verification-and-operator-authorization")
assert proposal["verdict"] is None
assert proposal["single_causal_change"].startswith("No science change")
assert proposal["control"]["public_val_bpb"] is None
assert request["state"] == "PENDING_INDEPENDENT_VERIFICATION_AND_OPERATOR_AUTHORIZATION"
assert request["dispatch_authorized"] is False
assert request["host"] == "trn22" and request["expected_hostname"] == "ip-172-31-32-253"
assert request["label"] == LABEL
assert request["remote"] == "/home/ubuntu/trainium_freshstart_20260925_s0_warm_native_trn22_r1"
assert request["origin"] == "https://github.com/CsabaConsulting/trainium_frontier.git"
assert request["min_free_bytes"] == FLOOR
assert request["pre_launch_required_free_bytes"] == FLOOR + PRIOR_DENSE_GROWTH
assert request["runner_owns_manifest"] is False and request["non_overwrite"] is True
assert request["checkpoint_prune_authorized"] is False
assert request["warm_cache_policy"]["cold_cache_authorized"] is False
assert request["warm_cache_policy"]["cache_cleanup_authorized"] is False
assert set(request["fingerprint_paths"]) == set(request["expected_sha256"])
subprocess.run(["git", "merge-base", "--is-ancestor", request["expected_commit"], "HEAD"], cwd=ROOT, check=True)
for relative, expected in request["expected_sha256"].items():
    assert object_sha(request["expected_commit"], relative) == expected, relative

environment = request["environment"]
command = request["command"]
assert environment["NEURON_LOGICAL_NC_CONFIG"] == "2"
assert environment["NEURON_CC_FLAGS"] == "--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
assert request["warm_cache_policy"]["reuse_only"] == [
    f"{CACHE_ROOT}/{name}" for name in CACHE_ENV.values()
]
for key, name in CACHE_ENV.items():
    expected = f"{CACHE_ROOT}/{name}"
    assert environment[key] == expected
    assert command.count(f"{key}={expected}") == 1
assert command[:7] == ["timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "3600s", "env", "-i"]
assert "NEURON_LOGICAL_NC_CONFIG=2" in command
assert "NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16" in command
assert f"TRAINIUM_FRESH_START_RUN_ID={LABEL}" in command
assert command[-2:] == ["--nproc_per_node=4", "train.py"]
assert command[command.index("train.py") + 1:] == []
assert not any(x.startswith("NEURON_RT_VISIBLE_CORES=") or x.startswith("NEURON_RT_NUM_CORES=") for x in command)
assert not any(x.startswith("PYTHONPATH=") for x in command)
assert not any(x.startswith("NEURON_COMPETITION_R1_OUT_DIR=") for x in command)
assert f"NEURON_COMPETITION_R1_STEP_LOG={request['remote']}/research/run-logs/{LABEL}.steps.jsonl" in command

s0_source = (ROOT / "train.py").read_text()
assert "MAX_TRAIN_SECONDS = 1800" in s0_source
assert "parser.add_argument('--seed', type=int, default=42" in s0_source
assert "parser.add_argument('--out-dir', type=str, default='out')" in s0_source
assert "make_dataloader(tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, 'train', device)" in s0_source
assert "record_step_boundary(step)" in s0_source
assert "torch.load(checkpoint_path, map_location='cpu', weights_only=True)" in s0_source
assert "def load_for_eval(" in s0_source
assert "TRAIN_TIME_BUDGET_SECONDS = 30 * 60" in (ROOT / "prepare.py").read_text()

g1 = json.loads((ROOT / "research/v2/g1.json").read_text())
assert g1["evaluation"]["eval_args_verified"] is True
assert g1["evaluation"]["verification_scope"].startswith("trn22 only")
g2 = json.loads((ROOT / "research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/remote-run/receipts/fs-20260925-g2-s0-bare-trn22-r1.json").read_text())
assert g2["argv"] == [] and g2["train_py_sha256"] == S0_SHA
assert all(rank["status"] == "PASS" for rank in g2["ranks"])
comparison = json.loads((ROOT / "research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/remote-run/comparisons/116188-vs-s0.json").read_text())
assert comparison["identical"] is True and comparison["b_sha256"] == S0_SHA

runbook = (PKG / "RUNBOOK.md").read_text()
for required in (
    "not a cold submission replay", "8,833,482,752", "current existence, inode identity and size are unverified",
    "does not\nenforce", "only fsynced decimal step", "sparse stdout log is not a complete 100-step lockstep",
    "two cold", "Only the human submits",
):
    assert required in runbook, required
for forbidden in ("rm -", "rmtree", "unlink", "cache_cleanup_authorized\": true", "checkpoint_prune_authorized\": true"):
    assert forbidden not in runbook

print("PASS pending warm-S0 static contract; no host action or dispatch authorization")
