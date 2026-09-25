"""Read-only source/manifest checks; no model import, tensors, staging or launch."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANONICAL = "e9109dec72e41f0d481bd55df49aaedf7655bb6a"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(commit, path):
    return subprocess.run(["git", "show", commit + ":" + path], cwd=ROOT,
                          check=True, capture_output=True).stdout


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(name):
    return json.loads((HERE / name).read_text())


def value(argv, flag):
    require(argv.count(flag) == 1, "missing/duplicate " + flag)
    return argv[argv.index(flag) + 1]


def main():
    manifest = read("sources.json")
    source = manifest["candidate"]
    require(source["sha256"] == "93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5", "candidate pin")
    for spec in [source, *manifest["references"], *manifest["tools"]]:
        require(sha(git_bytes(spec["git_commit"], spec["git_path"])) == spec["sha256"], spec["stage_name"])
    require(sha((ROOT / source["git_path"]).read_bytes()) == source["sha256"], "working source changed")
    reviews = []
    for spec in manifest["source_reviews"]:
        reviews.append({**spec, "sha256": sha(git_bytes(spec["git_commit"], spec["path"]))})
    all_labels = []
    for host in ["trn21", "trn22"]:
        proposal = read(host + "/EXPERIMENT_PROPOSAL.json")
        plan = read(host + "/cpu-g2-plan.json")
        job = read(host + "/native-job.disabled.json")
        require(not proposal["launch_authorized"] and not proposal["armed"], "proposal must stay disabled")
        require(not plan["cpu_execution_authorized"] and not plan["native_training_authorized"], "CPU plan must stay disabled")
        require(not job["armed"] and not job["launch_authorized"], "native job must stay disabled")
        require("NOT_READY" in job["readiness"], "native readiness")
        require(proposal["candidate"]["train_sha256"] == source["sha256"] == job["source_sha256"], "source binding")
        require(job["host"] == plan["host"] == proposal["queue_binding"]["host"] == host, "host binding")
        require(job["argv"] == proposal["candidate"]["argv"] == proposal["queue_binding"]["argv"], "native argv binding")
        require(job["argv"].count("--unet-8x1024") == 1, "architecture ON")
        require(value(job["argv"], "--attention-layers") == "0,2,4,6,7", "attention pattern")
        for flag, expected in [("--n-embd", "1024"), ("--seq-len", "1024"), ("--pack-factor", "32"),
                               ("--seed", "42"), ("--max-train-seconds", "1800"), ("--num-steps", "1000000")]:
            require(value(job["argv"], flag) == expected, flag)
        require("--no-ngram-ve" in job["argv"] and "--recursive-schedule" not in job["argv"], "standalone architecture")
        require(job["command"][1:3] == ["--standalone", "--nproc_per_node=4"], "full chip command")
        require(job["environment"]["NEURON_LOGICAL_NC_CONFIG"] == "2", "LNC2")
        require("--optlevel=1" in job["environment"]["NEURON_CC_FLAGS"], "compiler policy")
        require(value(job["fresh_evaluation"]["command"], "--eval-tokens") == "20971520", "20M")
        require(value(job["fresh_evaluation"]["command"], "--seq-len") == "1024", "eval context")
        require(plan["minimum_available_ram_gib"] == 64 and plan["minimum_free_disk_gib"] == 6.5, "preflight floors")
        require(plan["run_serially"] and len(plan["probes"]) == 4, "serial probe plan")
        probes = {p["name"]: p for p in plan["probes"]}
        require(probes["candidate-on"]["argv"] == job["argv"], "G2/native exact argv")
        require(probes["candidate-on"]["shape"] == "8x1024", "full ON shape")
        require(probes["candidate-on"]["source_sha256"] == probes["candidate-off"]["source_sha256"] == source["sha256"], "candidate G2 source")
        off = probes["candidate-off"]["argv"]
        require("--unet-8x1024" not in off and "--unet-skip" not in off and "--partial-attn-8" not in off, "default OFF")
        require(probes["c5c-off"]["argv"] == probes["6ef7-off"]["argv"] == off, "same reference/OFF argv")
        require("--recursive-schedule" not in off, "Recursive must be OFF")
        for p in probes.values():
            require((p["steps"], p["nproc"], p["micro_rows"], p["micro_tokens"]) == (3, 4, 2, 256), "G2 profile")
            require(p["keep"] and p["timeout_seconds"] == 3600, "G2 evidence retention")
            require(p["command"][p["command"].index("--") + 1:] == p["argv"], "G2 command argv")
            require("--real-data" not in p["command"], "synthetic G2 only")
            all_labels.append(p["label"])
        all_labels += [job["label"], job["fresh_evaluation"]["label"]]
        require(len(plan["comparisons"]) == 2, "both identity comparisons")
        # Authenticate the historical control, without claiming a fresh host check.
        control = proposal["control"]
        git_bytes(CANONICAL, control["review"])
        ledger = git_bytes(CANONICAL, "research/v2/ledger.jsonl").decode()
        records = [json.loads(line) for line in ledger.splitlines() if line.strip()]
        require(any((r.get("label") == control["run_label"] or r.get("run_label") == control["run_label"])
                    for r in records), "historical control label missing")
    require(len(set(all_labels)) == len(all_labels), "duplicate packet labels")
    old_queue = git_bytes(CANONICAL, "research/v2/queue.json").decode()
    old_ledger = git_bytes(CANONICAL, "research/v2/ledger.jsonl").decode()
    require(all(label not in old_queue and label not in old_ledger for label in all_labels), "label already recorded at canonical snapshot")
    # Static parameter arithmetic for this pinned no-table profile, not a peak-RAM forecast.
    width, vocab, heads, ratio = 1024, 8192, 8, 4
    def parameter_count(depth, attention, ve_sites, skips):
        return (2 * vocab * width + depth * 2 * ratio * width**2 +
                attention * 4 * width**2 + ve_sites * vocab * width +
                attention * width * 2 + ve_sites * 32 * heads + depth * 2 + skips)
    dense = parameter_count(6, 6, 3, 0)
    architecture = parameter_count(8, 5, 1, 4)
    prior = json.loads(git_bytes(CANONICAL, "research/v2/operator/20260925-trn21-l3-timer-dense-g2-r1/g2.json"))
    require(dense == sum(prior["ranks"][0]["plan_summary"]["elements_by_owner"]), "static count fails dense receipt cross-check")
    result = {
        "verdict": "PASS_STATIC_PACKET_ONLY", "candidate_sha256": source["sha256"],
        "ready": False, "stage_authorized": False, "cpu_g2_completed": False, "native_launch_authorized": False,
        "unique_labels_at_canonical_snapshot": len(all_labels), "canonical_snapshot": CANONICAL,
        "source_reviews_hashed": reviews,
        "memory_arithmetic": {
            "dense_parameters_receipt_cross_checked": dense,
            "architecture_parameters_static_estimate": architecture,
            "architecture_fp32_parameters_bytes_per_rank": architecture * 4,
            "four_rank_parameter_plus_gradient_bytes_only": architecture * 4 * 2 * 4,
            "peak_ram_bytes": None, "peak_hbm_bytes": None,
            "minimum_available_ram_bytes_policy": 64 * 1024**3,
            "prior_dense_preflight_available_ram_bytes": 131432075264,
            "scope": "Arithmetic lower bound only; excludes optimizer/broadcast buffers, activations, cached weights, Python/runtime, checkpoint reload and eval masks. Prior availability is historical, not current headroom."
        },
        "limitations": ["No exact-size model allocated or executed", "No host access/staging/launch", "Queue currently rejects 93ae until separately reviewed support exists"],
        "files": {str(p.relative_to(HERE)): sha(p.read_bytes()) for p in sorted(HERE.rglob("*"))
                  if p.is_file() and p.name != "packet-check.json" and "__pycache__" not in p.parts},
    }
    (HERE / "packet-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["verdict", "candidate_sha256", "ready", "unique_labels_at_canonical_snapshot", "memory_arithmetic"]}, indent=2))


if __name__ == "__main__":
    main()
