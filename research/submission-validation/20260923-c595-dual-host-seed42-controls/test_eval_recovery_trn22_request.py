#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
path=HERE/"eval-recovery-trn22-r1-request.json";request=json.loads(path.read_text())
assert request["host"]=="trn22" and request["expected_hostname"]=="ip-172-31-32-253"
assert request["expected_commit"]=="4cb00f7611c95aa541fc9ad6b6a892e8dbf853f6"
assert request["remote"]=="/home/ubuntu/trainium_freshstart_20260924_c595_eval_recovery_trn22_r1"
assert request["command"][-2:]==["--core-commit",request["expected_commit"]]
assert set(request["fingerprint_paths"])==set(request["expected_sha256"])
for item,want in request["expected_sha256"].items():
 blob=subprocess.check_output(["git","show",request["expected_commit"]+":"+item],cwd=ROOT)
 assert hashlib.sha256(blob).hexdigest()==want
launcher=(ROOT/"research/fresh_start_launch.py").read_text()
assert "free >= 8 * 1024**3" in launcher and "less than 8 GiB free" in launcher
assert request["environment"]["NEURON_LOGICAL_NC_CONFIG"]=="2" and request["environment"]["NEURON_CC_FLAGS"]=="--optlevel=1"
assert not any(word in request["command"] for word in ("torchrun","train","submission_rehearsal.py","run_experiment.py","prune","cleanup"))
assert not any(request["label"] in name for name in subprocess.check_output(["git","ls-files","research/run-logs"],cwd=ROOT,text=True).splitlines())
job=json.loads((HERE/"eval-recovery-trn22-r1-job.json").read_text())
assert job["request_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest() and job["expected_commit"]==request["expected_commit"]
assert job["minimum_free_bytes"]==8*1024**3 and job["training_authorized"] is False and job["checkpoint_prune_authorized"] is False
print("c595 trn22 evaluator recovery immutable request/disk gate: PASS (no dispatch)")
