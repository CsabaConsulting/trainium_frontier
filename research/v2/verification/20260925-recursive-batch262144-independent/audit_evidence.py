"""Independent tensor-free lineage and three-receipt audit."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PACKET = ROOT / "research/v2/lanes/recursive-batch262144"
COMMIT = "4e361d15baef94f041501f558620422c55ae8c88"
SOURCE = "workflow/proposals/research-v2/train.py"
BASE = "6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3"
CANDIDATE = "a489f7f9e935e4fe9ecd2ad9df5f0f327afcbe9475b71e7fbe0f4e71a48925d9"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parent = git("show", f"{COMMIT}^:{SOURCE}")
    adopted = git("show", f"a4aa4351789115a2d0aa947d9ca00ff689f9bd0b:{SOURCE}")
    candidate = (ROOT / SOURCE).read_bytes()
    assert parent == adopted and sha(parent) == BASE and sha(candidate) == CANDIDATE
    diff = git("diff", f"{COMMIT}^", COMMIT, "--", SOURCE).decode().splitlines(keepends=True)
    added = [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")]
    deleted = [line for line in diff if line.startswith("-") and not line.startswith("---")]
    assert len(added) == 5 and not deleted
    restored = candidate.decode()
    for line in added:
        assert restored.count(line) == 1
        restored = restored.replace(line, "", 1)
    assert restored.encode() == parent
    changed = git("diff", "--name-only", f"{COMMIT}^", COMMIT).decode().splitlines()
    assert all(path == SOURCE or path.startswith("research/v2/lanes/recursive-batch262144/") for path in changed)
    hashes = {}
    for line in (PACKET / "SHA256SUMS").read_text().splitlines():
        expected, path = line.split(None, 1)
        assert sha((ROOT / path).read_bytes()) == expected
        hashes[path] = expected
    assert len(hashes) == 12
    receipts, rank0s, summaries = {}, {}, {}
    for mode, expected, micro_count in (("base", BASE, 3), ("default", CANDIDATE, 3), ("262144", CANDIDATE, 6)):
        receipt = json.loads((PACKET / f"tiny-g2-{mode}.json").read_text())
        assert receipt["verdict"] == "PASS" and receipt["train_py_sha256"] == expected
        assert (receipt["nproc"], receipt["steps"], receipt["micro_batch"]) == (4, 3, [1, 16])
        assert receipt["prepare"]["stub"]
        assert sorted(r["rank"] for r in receipt["ranks"]) == [0, 1, 2, 3]
        for rank in receipt["ranks"]:
            assert rank["status"] == "PASS" and not rank["cross_rank_mismatches"]
            assert rank["n_params"] == rank["n_changed"] == 62
            assert not rank["unchanged_params"] and not rank["unchanged_tables"]
            assert rank["plan_sha256"] == receipt["plan_sha256"]
            assert len(rank["losses"]) == micro_count and all(math.isfinite(v) for v in rank["losses"])
        rank0 = next(r for r in receipt["ranks"] if r["rank"] == 0)
        assert len(rank0["final_digests"]) == 62
        ckpt = rank0["checkpoint_check"]
        assert ckpt["ok"] and ckpt["path_exists"] and ckpt["deterministic"]
        assert ckpt["logits_shape"] == [1, 16, 8192]
        receipts[mode], rank0s[mode] = receipt, rank0
        summaries[mode] = dict(source_sha256=expected, rank_count=4, optimizer_steps=3,
                               finite_forward_losses_per_rank=micro_count,
                               all_loss_arrays={r["rank"]: r["losses"] for r in receipt["ranks"]},
                               updated_parameter_count=62, checkpoint_and_rank_consistency=True)
    assert receipts["base"]["argv"] == receipts["default"]["argv"]
    assert receipts["262144"]["argv"] == receipts["base"]["argv"] + ["--total-batch-size", "262144"]
    assert summaries["base"]["all_loss_arrays"] == summaries["default"]["all_loss_arrays"]
    assert rank0s["base"]["final_digests"] == rank0s["default"]["final_digests"]
    plans = {r["plan_sha256"] for r in receipts.values()}
    assert len(plans) == 1
    proposal = json.loads((PACKET / "proposal.json").read_text())
    adoption = json.loads((ROOT / "research/v2/adoptions/20260925-recursive-schedule.json").read_text())
    assert proposal["base"]["sha256"] == BASE
    assert proposal["candidate"]["argv"] == adoption["argv"] + ["--total-batch-size", "262144"]
    reduced_argv = adoption["argv"].copy()
    reduced_argv[reduced_argv.index("--n-embd") + 1] = "128"
    assert receipts["base"]["argv"] == reduced_argv
    assert not proposal["launch_authorized"] and not proposal["submission_authorized"]
    assert all(v["label"] is None and v["public_20m"] is None
               for v in proposal["control"]["fresh_new_base_controls"].values())
    comparison = subprocess.run(["python3", "-B", "workflow/tools/preflight_dryrun.py", "--compare",
        "research/v2/lanes/recursive-batch262144/tiny-g2-base.json",
        "research/v2/lanes/recursive-batch262144/tiny-g2-default.json"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    comparison = json.loads(comparison.stdout)
    assert comparison["identical"]
    output = dict(verdict="PASS_LINEAGE_AND_REDUCED_RECEIPT_AUDIT_ONLY", commit=COMMIT,
                  candidate_sha256=CANDIDATE, actual_parent_source_sha256=BASE,
                  five_insertions_reconstruct_exact_adopted_source=True, manifest_hashes=hashes,
                  receipts=summaries, default_comparison=comparison, shared_plan_sha256=plans.pop(),
                  unchanged_model_microshape=True,
                  physical_G2_tokens_per_optimizer_step=dict(base=64, default=64, arm262144=128),
                  production_tokens_per_optimizer_step=dict(control=131072, arm=262144),
                  production_arithmetic_is_not_G2_physical_token_evidence=True,
                  fresh_controls_pending=True, ledger_mutated=False,
                  full_size=False, native_executed=False, quality_claim=False)
    (HERE / "evidence-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
