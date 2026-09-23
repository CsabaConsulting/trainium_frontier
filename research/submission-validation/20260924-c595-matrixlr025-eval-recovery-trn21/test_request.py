#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
request=json.loads((HERE/"request.json").read_text());job=json.loads((HERE/"job.json").read_text())
assert request["expected_commit"]=="9c9647d9a9befea78053ff3c42c5fb945e46754d" and request["host"]=="trn21"
assert request["label"]=="fs-20260924-c595-matrixlr025-eval-recovery-trn21-r1"
assert set(request["fingerprint_paths"])==set(request["expected_sha256"])
for path,want in request["expected_sha256"].items():
 blob=subprocess.check_output(["git","show",request["expected_commit"]+":"+path],cwd=ROOT);assert hashlib.sha256(blob).hexdigest()==want
assert request["command"][-2:]==["--core-commit",request["expected_commit"]]
assert "NEURON_CC_FLAGS" not in request["environment"] and all("torchrun" not in x for x in request["command"])
assert job["request_sha256"]==hashlib.sha256((HERE/"request.json").read_bytes()).hexdigest()
assert job["training_authorized"] is False and job["checkpoint_prune_authorized"] is False and job["cache_cleanup_authorized"] is False
assert not any(request["label"] in x for x in subprocess.check_output(["git","ls-files","research/run-logs"],cwd=ROOT,text=True).splitlines())
print("LR evaluator-only immutable request: PASS (no dispatch)")
