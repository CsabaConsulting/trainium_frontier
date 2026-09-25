"""Offline, non-dispatchable construction of the trn21 stage-r2 CPU G2 argv.

This file never executes a host command. The request remains HOLD even when
the deterministic command hashes match; a later Verifier and Supervisor must
materialize and assign a fresh request after an independently accepted checkout.
"""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CONTRACT = HERE / "GATE_CONTRACT.json"
PYTHON = "/home/ubuntu/workspace/native_venv/bin/python"
ENV = "/usr/bin/env"
CACHE_ROOT = ("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1"
              "/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train")
SCIENCE_LABEL = "fs-20260925-v2-c5c-a0-control-trn21-r1"
TRAIN_ARGV = ["--n-embd", "1024", "--no-ngram-ve", "--no-qk-shift",
              "--compile-sdpa-direct", "--nki-local-conv", "--bf16-norm-output",
              "--pack-factor", "32", "--clip-after-reduce", "--seq-len", "1024",
              "--no-eval-public", "--matrix-lr", "0.015"]
EFFECTIVE = {
    "NEURON_LOGICAL_NC_CONFIG": "2",
    "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
    "NEURON_COMPETITION_R1_SEED": "42",
    "NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS": "1800",
}
LOCAL_HASHES = {
    "workflow/tools/preflight_dryrun.py": "6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8",
    "workflow/tools/contained.py": "0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d",
    "research/v2/authorizations/trn21-c5c-a0-stage-r2-extraction/GATE_CONTRACT.json": "cb034ffea544c560104a4d558a8a18119f50a4db5088045e0c5d711635bfbd65",
    "research/v2/authorizations/trn21-c5c-a0-stage-r2-extraction/stage_r2_checkout.py": "e66afb7a1e3368ed7854eaa679e2bad9a6b8529e3b793e0f8fff2c69c496785d",
    "research/v2/authorizations/trn21-c5c-a0-stage-r2-extraction/INDEPENDENT_IMPLEMENTATION_REVIEW.md": "1a05c6dab4cbae6f49bd7e58760155d26562ba39df6a85524fe4a50289679df4",
    "research/v2/authorizations/trn21-c5c-a0-stage-r2/INDEPENDENT_TRANSFER_RESULT_RECEIPT.md": "bfafcb56ff449708dc31a1ebbe9023448018ea505e37a15847ae78bc3a3a2916",
    "research/v2/authorizations/trn21-c5c-a0-stage-r2/operator-transfer-only-terminal-receipt.json": "9ab90252e5e21812f5aad2af7fd0bfd1e71205e2f8f740f6f41faf6606fe6a2e",
    "research/submission-validation/20260925-v2-c5c-a0-control-trn21-r1/EXPERIMENT_PROPOSAL.json": "55c6d500fbc535a45ad3afb6ef32882e30726b8ec90ac412e2ba987b5c71c1b9",
    "research/submission-validation/20260925-v2-c5c-a0-control-trn21-r1/GATE_CONTRACT.json": "8d6a6a5cc03f30c05de80f5581cfe628a727d7b0be62e91b18cbed09c3d8dbdf",
    "research/v2/authorizations/20260925-trn21-c5c-a0-r1-proposal-ratification.json": "c483e2b00653ede82226a73cd9d9b7f3b01fb2988828c1f749f66e416641551d",
    "research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/operator-receipt.json": "a0896536999087e19f1f2e781b79763b9c6ec9adb9b971d452a1d16e095c2c12",
    "research/agentic/verification/20260925-g1-trn21-r2-calibration-result-review.md": "2c0674fe4eb0e3ad64cf07bbcf6a6d24bbc89595d6e5d1b488c93b0e06f84ed0",
    "research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/remote-run/receipts/fs-20260925-g2-a0-research-v2-trn22-r1.json": "17ce1ec84ddc0d585ecd103237ecfcd86d3820cdd9aaf1c4fe8b60ed308b90e3",
    "research/agentic/verification/20260925-trn22-c5c-a0-r2-g2-r3-numeric-review.md": "e76654c7882c2ede9e382914aebb672b10841a083fc0eb2271e15b39965a1de4",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def argv_digest(argv):
    return digest(json.dumps(argv, separators=(",", ":")).encode("utf-8"))


def check_local_evidence():
    for relative, expected in LOCAL_HASHES.items():
        if digest((REPO / relative).read_bytes()) != expected:
            raise RuntimeError("frozen local G2 evidence or tool drifted: " + relative)


def build_plan(contract=None, *, check_argv_hashes=True):
    c = contract if contract is not None else json.loads(CONTRACT.read_text())
    if contract is None:
        check_local_evidence()
        for name, key in (("build_g2_argv.py", "offline_builder_sha256"),
                          ("test_g2_argv.py", "synthetic_tests_sha256"),
                          ("RUNBOOK.md", "runbook_sha256")):
            if digest((HERE / name).read_bytes()) != c["draft_file_pins"][key]:
                raise RuntimeError("frozen G2 draft file drift: " + name)
    source = c["source_checkout"]
    science = c["selected_science"]
    tools = c["persistent_cpu_tools"]
    g2 = c["cpu_g2_isolation"]
    label = g2["new_cpu_label"]
    root = g2["root_must_be_absent"]
    process_dir = g2["contained_process_dir_must_be_absent"]
    if (c["schema"] != "trn21-c5c-a0-exactenv-cpu-g2-stage-r2-hold-v1" or
            c["host"] != "trn21" or c["expected_hostname"] != "ip-172-31-34-85" or
            source != "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2" or
            c["source_commit"] != "e65a3097a5b00f6178188a935bb2acb9a5042c6c" or
            c["source_train_sha256"] != "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586" or
            c["source_prepare_sha256"] != "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb" or
            c["source_checkout_required_mode"] != "0700" or
            c["source_checkout_clean_shallow_one_commit_and_independently_reviewed"] is not False or
            any(c[name] is not False for name in ("cpu_g2_dispatch_authorized",
                  "native_training_authorized", "evaluation_authorized", "submission_authorized")) or
            any(value is not None for value in c["missing_gates"].values()) or
            c["source_stage_lineage"]["disabled_checkout_request_draft_accepted_for_dispatch"] is not False or
            any(c["source_stage_lineage"][name] is not None for name in (
                "repaired_checkout_request_and_independent_review_sha256",
                "operator_checkout_terminal_receipt_sha256", "independent_checkout_result_review_sha256"))):
        raise RuntimeError("source, authority or checkout gate drift")
    if (science["science_run_label_unconsumed"] != SCIENCE_LABEL or
            science["train_argv"] != TRAIN_ARGV or
            science["four_effective_environment_values"] != EFFECTIVE):
        raise RuntimeError("immutable science argv or environment drift")
    if (label != "fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2" or
            root != "/home/ubuntu/trainium-v2-g2-20260925-trn21-stage-r2" or
            process_dir != root + "/process-" + label or
            g2["step_log_parent_required_before_launch"] != root or
            g2["step_log_must_be_absent_before_launch"] != root + "/" + label + ".steps.jsonl" or
            g2["cpu_receipt"] != root + "/c5c-g2.json" or
            g2["cpu_tempdir"] != root or
            g2["root_mode_if_created"] != "0700" or
            g2["g2_options"] != ["--steps", "3", "--nproc", "4", "--timeout", "900", "--keep"] or
            g2["synthetic_data_only"] is not True or
            g2["real_data_option_allowed"] is not False or
            g2["native_neuron_work_or_public_evaluation_allowed"] is not False):
        raise RuntimeError("unique CPU-only G2 isolation drift")
    if (tools["preflight_path"] != "/home/ubuntu/trainium-v2-g2-preflight-dryrun-20260925-trn21-stage-r2.py" or
            tools["preflight_sha256"] != LOCAL_HASHES["workflow/tools/preflight_dryrun.py"] or
            tools["contained_path"] != "/home/ubuntu/trainium-v2-g2-contained-20260925-trn21-stage-r2.py" or
            tools["contained_sha256"] != LOCAL_HASHES["workflow/tools/contained.py"] or
            tools["rehash_with_no_follow_before_and_after"] is not True or
            tools["tool_retransfer_or_substitution_authorized"] is not False):
        raise RuntimeError("persistent reviewed CPU tools drift")
    environment = [
        "HOME=/home/ubuntu",
        "PATH=/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
        "NKI_VENV_PATH=/home/ubuntu/workspace/native_venv",
        "PYTHONUNBUFFERED=1", "OMP_NUM_THREADS=1",
        "TMPDIR=" + root, "TRN_EXP_LABEL=" + label,
        "NEURON_LOGICAL_NC_CONFIG=2",
        "NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
        "NEURON_COMPETITION_R1_SEED=42",
        "NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800",
        "TORCH_NEURONX_NEFF_CACHE_DIR=" + CACHE_ROOT + "/neff",
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=" + CACHE_ROOT + "/local",
        "TORCH_NEURONX_HLO_CACHE_DIR=" + CACHE_ROOT + "/hlo",
        "TORCHINDUCTOR_CACHE_DIR=" + CACHE_ROOT + "/inductor",
        "NEURON_COMPILE_CACHE_DIR=" + CACHE_ROOT + "/compiler",
        "TORCH_NEURONX_DEBUG_DIR=" + CACHE_ROOT + "/debug",
        "NEURON_COMPETITION_R1_OUT_DIR=" + source + "/out/" + SCIENCE_LABEL,
        "NEURON_COMPETITION_R1_STEP_LOG=" + g2["step_log_must_be_absent_before_launch"],
        "TRAINIUM_FRESH_START_RUN_ID=" + SCIENCE_LABEL,
    ]
    probe = [PYTHON, tools["preflight_path"], "--train-py", source + "/train.py",
             "--prepare-dir", source] + g2["g2_options"] + ["--out", g2["cpu_receipt"],
                                                       "--"] + TRAIN_ARGV
    launch = [PYTHON, tools["contained_path"], "launch", "--label", label,
              "--dir", process_dir, "--cwd", source, "--cgroup", "--", ENV,
              "-i"] + environment + probe
    wait = [PYTHON, tools["contained_path"], "wait", "--label", label,
            "--dir", process_dir, "--timeout", "1200"]
    stop = [PYTHON, tools["contained_path"], "stop", "--label", label,
            "--dir", process_dir, "--grace", "60"]
    quiesce = [PYTHON, tools["contained_path"], "quiesce", "--label", label,
               "--dir", process_dir, "--files", process_dir + "/launch.json",
               process_dir + "/run.log"]
    if check_argv_hashes and (c["exact_launch_argv_sha256"] != argv_digest(launch) or
                              c["exact_wait_argv_sha256"] != argv_digest(wait) or
                              c["exact_stop_on_failure_argv_sha256"] != argv_digest(stop) or
                              c["exact_quiesce_argv_sha256"] != argv_digest(quiesce)):
        raise RuntimeError("exact G2 argv hash drift")
    return {"contract": c, "launch_argv": launch, "wait_argv": wait,
            "stop_on_failure_argv": stop, "quiesce_argv": quiesce}


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: this offline builder never runs CPU G2")
