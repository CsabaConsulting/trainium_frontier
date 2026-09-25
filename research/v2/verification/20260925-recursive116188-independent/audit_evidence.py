"""Independent stdlib-only launch-profile and raw reduced-evidence audit."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PACKET = ROOT / "workflow/candidates/recursive-116188"
BASE = ROOT / "submissions/dense6x1024_direct_review_matrixlr015"
PORT_SHA = "4083da774ca0f8e26b69309203b07daceca1b559a1418eacc8371f2ff5021409"
REF_SHA = "6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parsed(path, argv):
    tree = ast.parse(path.read_text())
    constants = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            constants.append(node)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    stop = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "args" for t in n.targets))
    ns = dict(argparse=argparse, os=SimpleNamespace(environ={}))
    exec(compile(ast.Module(body=constants, type_ignores=[]), "<constants AST>", "exec"), ns)
    exec(compile(ast.Module(body=main.body[:stop], type_ignores=[]), "<safe parser AST>", "exec"), ns)
    result = vars(ns["parser"].parse_args(argv))
    ns.update(N_EMBD=result["n_embd"], SEQ_LEN=result["seq_len"], PACK_FACTOR=result["pack_factor"],
              GPTConfig=lambda **kwargs: kwargs, TOKENIZER_VOCAB_SIZE=8192)
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_config")
    exec(compile(ast.Module(body=[build], type_ignores=[]), "<config AST>", "exec"), ns)
    return result, ns["build_config"]()


def receipt(name, expected_sha):
    data = json.loads((PACKET / name).read_text())
    assert data["train_py_sha256"] == expected_sha
    assert data["verdict"] == "PASS"
    assert (data["nproc"], data["steps"], data["micro_batch"]) == (4, 3, [1, 16])
    assert data["prepare"]["stub"] is True
    assert sorted(r["rank"] for r in data["ranks"]) == [0, 1, 2, 3]
    for rank in data["ranks"]:
        assert rank["status"] == "PASS" and not rank["cross_rank_mismatches"]
        assert rank["n_params"] == rank["n_changed"] == 62
        assert not rank["unchanged_params"] and not rank["unchanged_tables"]
        assert rank["plan_sha256"] == data["plan_sha256"]
        assert len(rank["losses"]) == 3 and all(math.isfinite(v) for v in rank["losses"])
    rank0 = next(r for r in data["ranks"] if r["rank"] == 0)
    assert len(rank0["final_digests"]) == 62
    ckpt = rank0["checkpoint_check"]
    assert ckpt["ok"] and ckpt["path_exists"] and ckpt["deterministic"]
    assert ckpt["logits_shape"] == [1, 16, 8192]
    return data, rank0


def main():
    assert digest(PACKET / "train.py") == PORT_SHA
    assert digest(BASE / "train.py") == "116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d"
    assert digest(ROOT / "workflow/proposals/research-v2/train.py") == REF_SHA
    hashes = {}
    for line in (PACKET / "SHA256SUMS").read_text().splitlines():
        expected, path = line.split(None, 1)
        assert digest(ROOT / path) == expected
        hashes[path] = expected
    assert len(hashes) == 14
    # Derive the proven argv from preserved evidence rather than a duplicated test list.
    old_command = shlex.split((BASE / "launch-command.txt").read_text())
    old_argv = old_command[old_command.index("train.py") + 1:]
    provenance = json.loads((BASE / "provenance.json").read_text())
    assert shlex.split(provenance["launch_command"]) == old_command
    old_args, old_config = parsed(BASE / "train.py", old_argv)
    new_args, new_config = parsed(PACKET / "train.py", [])
    assert new_args == old_args and new_config == old_config
    assert new_config == dict(sequence_len=1024, vocab_size=8192, n_layer=6, n_head=8,
                              n_embd=1024, mlp_ratio=4, attn_scale=0.0, n_kv_heads=0)
    adoption = json.loads((ROOT / "research/v2/adoptions/20260925-recursive-schedule.json").read_text())
    proposal = json.loads((PACKET / "proposal.json").read_text())
    assert adoption["source_sha256"] == REF_SHA
    assert proposal["cpu_g2"]["research_reference_argv"] == adoption["argv"]
    ref, rr = receipt("tiny-g2-research-on-r2.json", REF_SHA)
    port, pr = receipt("tiny-g2-port-r2.json", PORT_SHA)
    assert port["argv"] == ["--n-embd", "128"]
    small_argv = adoption["argv"].copy()
    small_argv[small_argv.index("--n-embd") + 1] = "128"
    assert ref["argv"] == small_argv
    losses = lambda r: {rank["rank"]: rank["losses"] for rank in r["ranks"]}
    assert losses(port) == losses(ref)
    assert pr["final_digests"] == rr["final_digests"]
    assert port["plan_sha256"] == ref["plan_sha256"]
    failures = {}
    for name, sha in (("tiny-g2-port.json", PORT_SHA), ("tiny-g2-research-on.json", REF_SHA)):
        data = json.loads((PACKET / name).read_text())
        assert data["verdict"] == "FAIL" and data["train_py_sha256"] == sha
        assert sorted(r["rank"] for r in data["ranks"]) == [0, 1, 2, 3]
        assert all(r["status"] == "FAIL" and r["error"] == "ModuleNotFoundError: No module named 'requests'"
                   and not r.get("losses") for r in data["ranks"])
        failures[name] = "PRESERVED_IMPORT_DEPENDENCY_FAILURE_NO_TRAINING_LOSSES"
    compare = subprocess.run([
        "python3", "-B", "workflow/tools/preflight_dryrun.py", "--compare",
        "workflow/candidates/recursive-116188/tiny-g2-research-on-r2.json",
        "workflow/candidates/recursive-116188/tiny-g2-port-r2.json",
    ], cwd=ROOT, text=True, capture_output=True, check=True)
    comparison = json.loads(compare.stdout)
    assert comparison["identical"]
    output = dict(verdict="PASS_STATIC_PROFILE_AND_REDUCED_RECEIPT_AUDIT_ONLY",
                  candidate_commit="abe15ac29cc5ec277f9fe1c96012fd8cb5eaa04a",
                  candidate_sha256=PORT_SHA, reference_sha256=REF_SHA,
                  manifest_sha256=hashes, proven_argv=old_argv, all_bare_parsed_arguments=new_args,
                  effective_model_config=new_config, complete_argument_profile_equal=True,
                  raw_loss_arrays_equal=losses(port), identical_parameter_digest_count=62,
                  plan_sha256=port["plan_sha256"], rank_count=4, steps=3,
                  observed_micro_batch=[1, 16], actual_receipt_model_width=128,
                  strict_reload_and_rank_consistency=True, raw_import_failures=failures,
                  recomputed_harness_comparison=comparison,
                  new_tensor_execution=False, exact_size_G2=False, native_executed=False,
                  cold_proof=False, submission_approval=False)
    (HERE / "evidence-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
