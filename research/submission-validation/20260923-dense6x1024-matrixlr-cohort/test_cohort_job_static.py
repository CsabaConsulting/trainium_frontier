#!/usr/bin/env python3
import importlib.util, json, math, pathlib

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
print("cohort adversarial static tests: PASS")
