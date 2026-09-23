#!/usr/bin/env python3
import hashlib, importlib.util, json, math, pathlib, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cohort_job", HERE / "cohort_job.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

good = json.dumps({"val_bpb": 0.99, "num_tokens": 20971520, "num_bytes": 83462186})
assert module.extract_eval_metrics("noise\n" + good)["val_bpb"] == 0.99
multiple = json.dumps({"irrelevant": 1}) + "\n" + good
assert module.extract_eval_metrics(multiple)["num_tokens"] == 20971520
for bad in ("no json", json.dumps({"val_bpb": 1.0}), "{broken"):
    try:
        module.extract_eval_metrics(bad)
    except SystemExit:
        pass
    else:
        raise AssertionError("malformed/missing metrics accepted")
module.validate_eval_metrics(module.extract_eval_metrics(good))
for bad_metrics in (
    {"val_bpb": math.nan, "num_tokens": 20971520, "num_bytes": 83462186},
    {"val_bpb": 1.0, "num_tokens": 1, "num_bytes": 83462186},
    {"val_bpb": 1.0, "num_tokens": 20971520, "num_bytes": 1},
):
    try:
        module.validate_eval_metrics(bad_metrics)
    except SystemExit:
        pass
    else:
        raise AssertionError("invalid metrics accepted")
dirty_env = {"RANK": "7", "WORLD_SIZE": "8", "MASTER_ADDR": "bad",
             "TORCHELASTIC_RUN_ID": "bad", "NEURON_COMPETITION_R1_SEED": "7",
             "SAFE": "yes"}
clean = module.scrub_environment(dirty_env)
assert clean["SAFE"] == "yes" and clean["NEURON_LOGICAL_NC_CONFIG"] == "2"
assert not any(key in clean for key in dirty_env if key != "SAFE")
source = (HERE / "cohort_job.py").read_text()
for required in ("socket.gethostname()", "tarfile.open(args.archive, \"x:gz\")",
                 "actual_members != expected_members", "ready_to_prune_after_retrieval_push",
                 "NEURON_COMPETITION_R1_", "TORCHELASTIC_RUN_ID"):
    assert required in source
assert "checkpoint.unlink()" not in source
prune = (HERE / "prune_checkpoint.py").read_text()
assert prune.index("retrieved_verified_committed_pushed") < prune.index("os.O_EXCL") < prune.index("checkpoint.unlink()")
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory); out = root / "run/out"; out.mkdir(parents=True)
    checkpoint = out / "final.pt"; checkpoint.write_bytes(b"checkpoint")
    archive = root / "evidence.tar.gz"; archive.write_bytes(b"archive")
    proof = root / "proof.json"; proof.write_text(json.dumps({"retrieved_verified_committed_pushed": True}))
    command = [sys.executable, str(HERE / "prune_checkpoint.py"), "--arm", "control",
               "--label", "fs-forged", "--checkpoint", str(checkpoint),
               "--expected-sha256", hashlib.sha256(b"checkpoint").hexdigest(),
               "--expected-bytes", str(len(b"checkpoint")), "--archive", str(archive),
               "--archive-sha256", hashlib.sha256(b"archive").hexdigest(),
               "--evidence-pushed-receipt", str(proof),
               "--evidence-pushed-receipt-sha256", hashlib.sha256(proof.read_bytes()).hexdigest(),
               "--receipt", str(root / "receipt.json")]
    failed = subprocess.run(command, capture_output=True)
    assert failed.returncode != 0 and checkpoint.is_file() and not (root / "receipt.json").exists()
print("cohort adversarial static tests: PASS")
