#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
path=HERE/"request.json";request=json.loads(path.read_text())
assert request["expected_commit"]=="5076552ef594aaca932cf7b57aa250bd1a9d55a7" and request["host"]=="trn21"
assert set(request["fingerprint_paths"])==set(request["expected_sha256"])
assert request["environment"]["NEURON_CC_FLAGS"]=="--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
assert request["command"][:5]==["timeout","--foreground","--signal=TERM","--kill-after=60s","4200s"]
assert not any("smoke" in item.lower() for item in request["command"])
assert request["command"][-2:]==["--core-commit",request["expected_commit"]]
assert "research/submission-validation/20260923-c595-dual-host-seed42-controls/eval_recovery.py" in request["fingerprint_paths"]
for item,want in request["expected_sha256"].items():
 blob=subprocess.check_output(["git","show",request["expected_commit"]+":"+item],cwd=ROOT)
 assert hashlib.sha256(blob).hexdigest()==want,(item,hashlib.sha256(blob).hexdigest(),want)
assert not any(request["label"] in name for name in subprocess.check_output(["git","ls-files","research/run-logs"],cwd=ROOT,text=True).splitlines())
job=json.loads((HERE/"job.json").read_text())
assert job["request_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest() and job["expected_commit"]==request["expected_commit"]
assert job["seed"]==42 and job["max_train_seconds"]==1800 and job["smoke"] is False
print("c595 MATRIX_LR 0.025 immutable request: PASS (no dispatch)")
