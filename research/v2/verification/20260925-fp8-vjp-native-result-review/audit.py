"""Read-only stdlib audit of the four-rank native diagnostic archive."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
PREFIX = ROOT / "research/v2/operator/20260925-trn21-l1-vjpdiag-r1/completed"
RUN = PREFIX / "runs/fs-20260925-l1-921271-vjpdiag-r1"
SOURCE = "921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858"
CHECKER = "d8187ad2f3e68d204203a6252e16062e52521e8b93f119c7f0a9fbe82e4367fd"
checks = []


def require(ok, description):
    if not ok:
        raise RuntimeError(description)
    checks.append(description)


def read(path):
    return json.loads(path.read_bytes())


def git_bytes(commit, path):
    return subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT)


require(hashlib.sha256(git_bytes("50a901374ae6f76b35b8556d32fd72b1ece28fc6",
        "workflow/proposals/research-v2/train.py")).hexdigest() == SOURCE, "exact_source")
require(hashlib.sha256(git_bytes("fe781c226a7c700a71324783b70018613c1dbcaf",
        "research/v2/lanes/fp8-projections/check_native.py")).hexdigest() == CHECKER, "exact_reviewed_checker")
retrieval = read(PREFIX / "retrieval.json")
require(retrieval["missing"] == [], "archive_complete")
for filename, info in retrieval["files"].items():
    data = (PREFIX / filename).read_bytes()
    require(len(data) == info["bytes"] and hashlib.sha256(data).hexdigest() == info["sha256"], "hash:" + filename)
launch, exit_record, terminal = [read(RUN / (name + ".json")) for name in ["launch", "exit", "terminal"]]
require(launch["command"][:4] == ["/usr/bin/timeout", "--signal=TERM", "--kill-after=60s", "300s"] and
        "--nproc_per_node=4" in launch["command"] and "--diagnose-first-case" in launch["command"], "bounded_diagnostic_argv")
require(exit_record["exit_code"] == 0 and 0 < exit_record["wall_seconds"] < 300 and
        abs(exit_record["monotonic"] - launch["monotonic"] - exit_record["wall_seconds"]) < 1e-8,
        "clean_bounded_monotonic_exit")
require(terminal["status"] == "DIAGNOSTIC_COMPLETE_NOT_COMPONENT_PASS", "terminal_not_acceptance")
postflight = read(RUN / "postflight.json")
require(postflight["active_workloads"] == [] and postflight["device_holders"] == [] and
        postflight["source_sha256"] == SOURCE, "recorded_quiescent_postflight")
expected_cases = {"original_compiled_bound_apply", "compiled_enclosing_wrapper",
                  "eager_apply_inspected", "compiled_bound_apply_inspected_replay"}
compiled_cases = expected_cases - {"eager_apply_inspected"}
rank_records = [read(PREFIX / f"receipts/diagnostic/diagnostic-rank{i}.json") for i in range(4)]
require(terminal["ranks"] == rank_records, "terminal_matches_individual_raw_ranks")
for rank, r in enumerate(rank_records):
    require(r["rank"] == rank and r["world_size"] == 4 and r["shape"] == [256,256,256] and
            r["seed"] == 42 and r["train_sha256"] == SOURCE and r["checker_sha256"] == CHECKER,
            f"rank{rank}:identity")
    require(r["verdict"] == "DIAGNOSTIC_ONLY_NOT_COMPONENT_PASS" and
            r["full_component_gate_passed"] is False and r["whole_step_measured"] is False and
            r["native_kernel_executed"] is True and r["original_rms_ceiling"] == .01,
            f"rank{rank}:unchanged_gate_scope")
    require(set(r["cases"]) == expected_cases and all(c["status"] == "OBSERVED" for c in r["cases"].values()),
            f"rank{rank}:all_cases_observed")
    for name, c in r["cases"].items():
        require(c["forward"]["within_original_1_percent_rms"] is True and
                c["compiled_or_eager_vs_direct"]["relative_rms"] == 0 and
                c["upstream_roundtrip"]["relative_rms"] == 0 and c["upstream_roundtrip"]["nonzero"] == 65536,
                f"rank{rank}:{name}:forward_and_python_upstream")
    for name in compiled_cases:
        c = r["cases"][name]
        require(c["grad_fn"] == "CompiledFunctionBackward" and all(
                c[g]["present"] and c[g]["finite"] and c[g]["nonzero"] == 0 and
                c[g]["rms"] == c[g]["l2"] == c[g]["max_abs"] == 0 and
                c[g]["relative_rms"] == 1 and c[g]["reference"]["nonzero"] == 65536 and
                c[g]["error_max_abs"] == c[g]["reference"]["max_abs"] and
                c[g]["within_original_1_percent_rms"] is False for g in ["dx", "dw"]),
                f"rank{rank}:{name}:actual_zero_vjp")
    require(not r["cases"]["original_compiled_bound_apply"]["pre_backward_saved_tensor_inspection"] and
            not r["cases"]["compiled_enclosing_wrapper"]["pre_backward_saved_tensor_inspection"],
            f"rank{rank}:original_wrapper_uninspected")
    eager = r["cases"]["eager_apply_inspected"]
    for path in [eager, r["manual_native_vjp"], r["ordinary_bf16_control"]]:
        require(all(path[g]["finite"] and path[g]["nonzero"] == 65536 and
                    path[g]["within_original_1_percent_rms"] and path[g]["relative_rms"] < .01
                    for g in ["dx", "dw"]), f"rank{rank}:nonzero_eager_control_vjp")
    require(all(eager[g] == r["manual_native_vjp"][g] for g in ["dx", "dw"]), f"rank{rank}:eager_manual_stats_identical")
    for name in ["eager_apply_inspected", "compiled_bound_apply_inspected_replay"]:
        saved = r["cases"][name]["saved_context"]
        tensors = saved["tensors"]
        require(saved["inspection_errors"] == [] and len(tensors) == 2 and
                all(t["summary"]["finite"] and t["summary"]["dtype"] == "torch.bfloat16" and
                    t["summary"]["nonzero"] > 0 for t in tensors), f"rank{rank}:{name}:nonzero_saved_context")
        for operand in ["effective_x", "effective_w"]:
            require(sum(t["comparisons"][operand]["relative_rms"] < .01 for t in tensors) == 1,
                    f"rank{rank}:{name}:saved_{operand}_matches")
    require(r["direct_operands"]["reserved_exponent_present"] is False and
            all(r["direct_operands"][k]["relative_rms"] < .01 for k in ["packed", "scales", "effective_x", "effective_w"]),
            f"rank{rank}:direct_operands_within_gate")
require(all({k:v for k,v in r.items() if k != "rank"} ==
            {k:v for k,v in rank_records[0].items() if k != "rank"} for r in rank_records),
        "all_four_rank_observations_identical_except_rank")
r = rank_records[0]
summary = {}
for name, c in r["cases"].items():
    summary[name] = {"status": c["status"], "pre_backward_inspection": c["pre_backward_saved_tensor_inspection"],
                     "forward_relative_rms": c["forward"]["relative_rms"],
                     "dx_relative_rms": c["dx"]["relative_rms"], "dw_relative_rms": c["dw"]["relative_rms"],
                     "dx_nonzero": c["dx"]["nonzero"], "dw_nonzero": c["dw"]["nonzero"]}
print(json.dumps({"audit_status": "PASS_DIAGNOSTIC_AUTHENTICATION_COMPONENT_HOLD",
                  "check_count": len(checks), "checks": checks, "source_sha256": SOURCE,
                  "checker_sha256": CHECKER, "wall_seconds": exit_record["wall_seconds"],
                  "all_rank_stats_identical": True, "cases_rank0_representative": summary,
                  "manual_native_vjp": r["manual_native_vjp"], "ordinary_bf16_control": r["ordinary_bf16_control"],
                  "archive_files": retrieval["files"]}, indent=2))
