"""Non-host, non-Torch checks for the pending exact-source A1 package."""

from __future__ import annotations

import ast
import hashlib
import json
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    contract = json.loads((HERE / "gate-contract.json").read_text())
    proposal = json.loads((HERE / "EXPERIMENT_PROPOSAL.json").read_text())
    assert contract["schema"] == "pending-exact-source-a1-gates-v1"
    assert contract["dispatch_authorized"] is False
    assert contract["leaderboard_submission_authorized"] is False
    assert "PENDING" in contract["status"]
    assert "pending" in proposal["status"]
    assert proposal["verdict"] is None
    assert all(value is False for value in proposal["gates"].values())
    assert proposal["control"]["run_label"] is None
    assert proposal["control"]["public_val_bpb"] is None

    source = ROOT / contract["source_path"]
    assert digest(source) == contract["source_sha256"]
    for rel, expected in contract["fingerprints"].items():
        assert digest(ROOT / rel) == expected, rel
    ast.parse(source.read_text())
    src = source.read_text()
    for flag in (
        "--ngram-ve", "--ngram-ve-layers", "--ngram-ve-table-mult", "--ngram-ve-dim",
        "--ngram-ve-trigram", "--ngram-ve-opt", "--ngram-ve-flat-lr",
        "--ngram-ve-bf16", "--ngram-ve-clip-exclude", "--num-steps",
    ):
        assert flag in src, flag
    assert "def has_ngram_bigram(" in src and "def has_ngram_trigram(" in src
    assert "kind='rmsprop'" in src and "_rmsprop_step(source)" in src

    queue = json.loads((ROOT / "research/v2/queue.json").read_text())
    arm = next(x for x in queue["queue"] if x["id"] == "A1")
    assert arm["argv_delta"]["replace"] == "--no-ngram-ve"
    base = contract["control_argv"]
    assert base.count("--no-ngram-ve") == 1
    splice_at = base.index("--no-ngram-ve")
    assert contract["a1_argv"] == (
        base[:splice_at] + arm["argv_delta"]["with"] + base[splice_at + 1:]
    )
    assert shlex.split(proposal["candidate"]["launch_command"].split("train.py ", 1)[1]) == contract["a1_argv"]
    assert contract["table_expectation"]["keys"] == ["b5", "t5"]
    assert contract["table_expectation"]["rows_per_table"] == 64 * 8192
    assert contract["table_expectation"]["table_parameters_total"] == 2 * 64 * 8192 * 128
    assert contract["g2"]["steps"] == 2 and contract["g2"]["nproc"] == 4
    assert contract["g2"]["trace_rmsprop"] is True and contract["g2"]["stub_prepare"] is False
    assert contract["g2"]["receipt"] is None and contract["native_smoke"]["receipt"] is None
    assert contract["full"]["receipt"] is None
    assert contract["full"]["requires_valid_same_host_control_label"] is None
    assert contract["native_smoke"]["argv_extension"] == ["--num-steps", "30"]
    assert contract["full"]["argv_extension"] == []
    assert contract["disk"]["projected_table_run_headroom_bytes"] is None
    assert contract["topology"] == {
        "neuron_logical_nc_config": 2, "nproc_per_node": 4, "seed": 42,
        "max_train_seconds": 1800,
    }
    print("PASS: exact c5c source, A1 queue flags, two-table dimensions and no-launch gates")


if __name__ == "__main__":
    main()
