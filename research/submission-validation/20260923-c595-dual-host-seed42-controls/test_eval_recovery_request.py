#!/usr/bin/env python3
import hashlib,json,pathlib,re,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
path=HERE/"eval-recovery-trn21-r1-request.json";request=json.loads(path.read_text())
assert request["expected_commit"]=="9b0952aab6465ed10737e37abae417c47ea9d80a"
assert request["host"]=="trn21" and request["expected_hostname"]=="ip-172-31-34-85"
assert re.fullmatch(r"fs-[A-Za-z0-9_.-]+",request["label"])
assert request["remote"]=="/home/ubuntu/trainium_freshstart_20260924_c595_eval_recovery_trn21_r1"
assert set(request["fingerprint_paths"])==set(request["expected_sha256"])
assert request["command"][:5]==["timeout","--foreground","--signal=TERM","--kill-after=60s","3300s"]
assert request["environment"]["NEURON_LOGICAL_NC_CONFIG"]=="2" and request["environment"]["NEURON_CC_FLAGS"]=="--optlevel=1"
assert request["command"][-2:]==["--core-commit",request["expected_commit"]]
assert not any(word in request["command"] for word in ("torchrun","train","submission_rehearsal.py","run_experiment.py"))
tracked=set(subprocess.check_output(["git","ls-tree","-r","--name-only",request["expected_commit"]],cwd=ROOT,text=True).splitlines())
for item,want in request["expected_sha256"].items():
 assert item in tracked
 blob=subprocess.check_output(["git","show",request["expected_commit"]+":"+item],cwd=ROOT)
 assert hashlib.sha256(blob).hexdigest()==want
# This trn21 label is historical and may now have immutable dispatch evidence.
launcher=(ROOT/"research/fresh_start_launch.py").read_text()
assert 'request["request_file_sha256"] = hashlib.sha256(request_bytes).hexdigest()' in launcher
job=json.loads((HERE/"eval-recovery-trn21-r1-job.json").read_text())
assert job["request_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest() and job["expected_commit"]==request["expected_commit"]
assert job["mode"]=="evaluator_only" and job["training_authorized"] is False and job["checkpoint_prune_authorized"] is False
print("c595 evaluator recovery immutable request: PASS (no dispatch)")
