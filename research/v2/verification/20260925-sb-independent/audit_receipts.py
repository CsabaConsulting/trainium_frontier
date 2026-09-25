"""Independent, tensor-free audit of the bounded S-B receipts; no launch authority."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "research/v2/sb-architecture-fp8"
HERE = Path(__file__).resolve().parent
SOURCE = "workflow/proposals/research-v2/train.py"
COMBINED = "a2f6b188db8418f4bccf4f2704db1a68bd9c3588bcc69d1fc7819701db39a238"
REFS = {
    "dense": ("b18fb3586494c21f10008a9c63fd2056d1e6c00a", "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"),
    "arch": ("1e21c3d797eed764f108c41546b270065ad5650c", "93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5"),
    "fp8": ("50a901374ae6f76b35b8556d32fd72b1ece28fc6", "921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def check_receipt(path, expected_sha):
    raw = path.read_bytes()
    receipt = json.loads(raw)
    assert receipt["train_py_sha256"] == expected_sha
    assert receipt["verdict"] == "PASS"
    assert receipt["steps"] == 3 and receipt["nproc"] == 4
    assert receipt["micro_batch"] == [1, 16] and receipt["prepare"]["stub"]
    ranks = receipt["ranks"]
    assert sorted(r["rank"] for r in ranks) == [0, 1, 2, 3]
    for rank in ranks:
        assert rank["status"] == "PASS" and not rank["cross_rank_mismatches"]
        assert len(rank["losses"]) == 3 and all(math.isfinite(v) for v in rank["losses"])
        assert rank["n_params"] == rank["n_changed"]
        assert not rank["unchanged_params"] and not rank["unchanged_tables"]
        assert rank["plan_sha256"] == receipt["plan_sha256"]
    rank0 = next(r for r in ranks if r["rank"] == 0)
    assert len(rank0["final_digests"]) == rank0["n_params"]
    checkpoint = rank0["checkpoint_check"]
    assert checkpoint["ok"] and checkpoint["path_exists"] and checkpoint["deterministic"]
    assert checkpoint["logits_shape"] == [1, 16, 8192]
    return {
        "file": str(path.relative_to(ROOT)), "receipt_sha256": digest(raw),
        "source_sha256": expected_sha, "rank_count": len(ranks),
        "finite_losses": sum(len(r["losses"]) for r in ranks),
        "all_parameters_updated": rank0["n_params"],
        "consistent_ownership_and_checkpoint": True,
    }


def compare(left, right):
    result = subprocess.run(
        [sys.executable, "-B", "workflow/tools/preflight_dryrun.py", "--compare",
         str(left.relative_to(ROOT)), str(right.relative_to(ROOT))],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout)
    assert report["identical"]
    return report


def main():
    assert digest((ROOT / SOURCE).read_bytes()) == COMBINED
    assert digest(git("show", f"ab7a007ed2f9ae8104b83632e772b3921639fb84:{SOURCE}")) == COMBINED
    for commit, expected in REFS.values():
        assert digest(git("show", f"{commit}:{SOURCE}")) == expected
    changed = git("diff", "--name-only", REFS["arch"][0], "ab7a007ed2f9ae8104b83632e772b3921639fb84").decode().splitlines()
    assert all(p == SOURCE or p.startswith("research/v2/sb-architecture-fp8/") for p in changed)
    harness = "workflow/tools/preflight_dryrun.py"
    assert (ROOT / harness).read_bytes() == git("show", f"{REFS['arch'][0]}:{harness}")
    audit = []
    for name, (_, expected) in REFS.items():
        audit.append(check_receipt(EVIDENCE / f"g2-ref-{name}.json", expected))
    for name in ("off", "arch", "fp8", "both"):
        audit.append(check_receipt(EVIDENCE / f"g2-sb-{name}.json", COMBINED))
    audit.append(check_receipt(HERE / "g2-combined-rerun.json", COMBINED))
    comparisons = {}
    for reference, candidate in (("dense", "off"), ("arch", "arch"), ("fp8", "fp8")):
        comparisons[reference] = compare(EVIDENCE / f"g2-ref-{reference}.json", EVIDENCE / f"g2-sb-{candidate}.json")
    comparisons["independent_combined_rerun"] = compare(EVIDENCE / "g2-sb-both.json", HERE / "g2-combined-rerun.json")
    focused = json.loads((HERE / "interactions-rerun.json").read_text())
    assert focused["source_sha256"] == COMBINED
    assert focused["verdict"] == "PASS_IMPLEMENTER_SELF_CHECK_ONLY"
    assert len(focused["checks"]) == 10 and all(focused["checks"].values())
    assert not focused["native_executed"] and not focused["exact_size_g2"]
    output = {
        "verdict": "PASS_SOURCE_IDENTITIES_AND_REDUCED_RECEIPT_AUDIT_ONLY",
        "source_sha256": COMBINED,
        "reviewed_commit": "ab7a007ed2f9ae8104b83632e772b3921639fb84",
        "only_production_file_changed": SOURCE,
        "unchanged_harness_sha256": digest((ROOT / harness).read_bytes()),
        "raw_receipts": audit, "recomputed_comparisons": comparisons,
        "independently_rerun_focused_checks": list(focused["checks"]),
        "native_executed": False, "exact_size_g2": False,
        "performance_or_BPB_claim": False, "launch_authorization": False,
    }
    (HERE / "receipt-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
