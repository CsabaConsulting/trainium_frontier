#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PKG = Path(__file__).resolve().parent
LABEL = "fs-20260925-v2-c5c-a0-control-trn22-r1"
TRAIN_SHA = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREPARE_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def core_sha(commit, relative):
    data = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    return hashlib.sha256(data).hexdigest()


proposal = json.loads((PKG / "EXPERIMENT_PROPOSAL.json").read_text())
request = json.loads((PKG / "request-template.json").read_text())

assert sha(ROOT / "train.py") == TRAIN_SHA
assert sha(ROOT / "prepare.py") == PREPARE_SHA
assert proposal["status"] == "human-approved-pending-independent-verification"
assert proposal["candidate"]["run_label"] == LABEL
assert proposal["gates"]["human_run_approval"] is True
assert proposal["verdict"] is None
assert proposal["control"]["public_val_bpb"] is None
assert proposal["control"]["public20m_status"].startswith("unknown")
assert proposal["external_quality_reference"] == {
    "public20m_bpb": 0.99195,
    "absolute_tolerance": 0.001,
    "acceptable_interval": [0.99095, 0.99295],
    "meaning": "A separately declared global numerical-equivalence reference band, not a result attributed to the 499f753c control and not evidence of a quality win",
}

assert request["state"] == "PENDING_INDEPENDENT_VERIFICATION_AND_OPERATOR_AUTHORIZATION"
assert request["dispatch_authorized"] is False
subprocess.run(["git", "merge-base", "--is-ancestor", request["expected_commit"], "HEAD"], cwd=ROOT, check=True)
assert set(request["fingerprint_paths"]) == set(request["expected_sha256"])
for relative, expected in request["expected_sha256"].items():
    assert core_sha(request["expected_commit"], relative) == expected, relative
assert request["host"] == "trn22" and request["expected_hostname"] == "ip-172-31-32-253"
assert request["label"] == LABEL
assert request["remote"] == "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r1"
assert request["min_free_bytes"] == 13 * 1024**3 // 2
assert request["pre_launch_required_free_bytes"] == request["min_free_bytes"] + 644245095
assert request["runner_owns_manifest"] is False and request["non_overwrite"] is True
assert request["checkpoint_prune_authorized"] is False
cache_root = (
    "/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/"
    "out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train"
)
cache_env = {
    "TORCH_NEURONX_NEFF_CACHE_DIR": "neff",
    "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": "local",
    "TORCH_NEURONX_HLO_CACHE_DIR": "hlo",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "NEURON_COMPILE_CACHE_DIR": "compiler",
    "TORCH_NEURONX_DEBUG_DIR": "debug",
}
expected_cache_entries = [f"{key}={cache_root}/{name}" for key, name in cache_env.items()]
assert request["warm_cache_policy"] == {
    "reuse_only": [f"{cache_root}/{name}" for name in cache_env.values()],
    "cold_cache_authorized": False,
    "cache_cleanup_authorized": False,
}
for key, name in cache_env.items():
    assert request["environment"][key] == f"{cache_root}/{name}"

command = request["command"]
assert command[:7] == ["timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "3600s", "env", "-i"]
assert "TRAINIUM_FRESH_START_RUN_ID=" + LABEL in command
for entry in expected_cache_entries:
    assert command.count(entry) == 1, entry
assert not any(item.startswith(tuple(f"{key}=" for key in cache_env)) and item not in expected_cache_entries for item in command)
assert not any(x.startswith("NEURON_RT_VISIBLE_CORES=") or x.startswith("NEURON_RT_NUM_CORES=") for x in command)
train_index = command.index("train.py")
assert command[train_index + 1:] == [
    "--n-embd", "1024", "--no-ngram-ve", "--no-qk-shift", "--compile-sdpa-direct",
    "--nki-local-conv", "--bf16-norm-output", "--pack-factor", "32",
    "--clip-after-reduce", "--seq-len", "1024", "--no-eval-public", "--matrix-lr", "0.015",
]
assert "--seed" not in command and "--max-train-seconds" not in command and "--num-steps" not in command

source = (ROOT / "train.py").read_text()
assert "os.environ.setdefault('NEURON_COMPILE_CACHE_DIR', '/tmp/neuron_cache')" in source
assert "NEURON_COMPETITION_R1_SEED', '42'" in source
assert "MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS" in source
prepare = (ROOT / "prepare.py").read_text()
assert "TRAIN_TIME_BUDGET_SECONDS = 30 * 60" in prepare
assert 'STEP_LOG_PATH = os.environ.get("NEURON_COMPETITION_R1_STEP_LOG", "")' in prepare
assert 'fh.write(f"{int(step)}\\n")' in prepare

runbook = (PKG / "RUNBOOK.md").read_text()
assert "contains no loss or timestamp" in runbook
assert "Use the training stdout log" in runbook
assert "external" in runbook and "runtime observer's boundary-arrival timestamps" in runbook
assert "first c5c baseline cannot be called a" in runbook

for token in ("rm -", "rm -r", "unlink", "rmtree", "cache_cleanup_authorized\": true", "checkpoint_prune_authorized\": true"):
    assert token not in runbook

print("PASS warm-c5c-a0 static contract (pending/non-executable)")
