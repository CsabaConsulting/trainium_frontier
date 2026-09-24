#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
for host,hostname in (("trn21","ip-172-31-34-85"),("trn22","ip-172-31-32-253")):
 path=HERE/f"{host}-request.json";request=json.loads(path.read_text());job=json.loads((HERE/f"{host}-job.json").read_text())
 assert request["host"]==host and request["expected_hostname"]==hostname and request["expected_commit"]=="8635a2ed2bb88876e31579e89da66aa2f9466bcf"
 assert set(request["fingerprint_paths"])==set(request["expected_sha256"])
 for item,want in request["expected_sha256"].items():
  blob=subprocess.check_output(["git","show",request["expected_commit"]+":"+item],cwd=ROOT);assert hashlib.sha256(blob).hexdigest()==want,(host,item)
 assert request["command"][:5]==["timeout","--foreground","--signal=TERM","--kill-after=60s","4200s"]
 assert request["command"][request["command"].index("--arm")+1]==host
 assert request["environment"]["NEURON_LOGICAL_NC_CONFIG"]=="2" and request["environment"]["NEURON_CC_FLAGS"]=="--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
 assert job["request_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest() and job["checkpoint_prune_authorized"] is False
 assert not any(request["label"] in x for x in subprocess.check_output(["git","ls-files","research/run-logs"],cwd=ROOT,text=True).splitlines())
print("current-base dual-host immutable requests: PASS (no dispatch)")
