#!/usr/bin/env python3
import copy,hashlib,importlib.util,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("control_job",HERE/"control_job.py");driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
for host,hostname in (("trn21","ip-172-31-34-85"),("trn22","ip-172-31-32-253")):
 path=HERE/f"{host}-request.json";request=json.loads(path.read_text());job=json.loads((HERE/f"{host}-job.json").read_text())
 assert request["host"]==host and request["expected_hostname"]==hostname and request["expected_commit"]=="0cd4811430bd8b22a9c1359175b73f26ffec0bd8"
 assert request["min_free_bytes"]==13*1024**3//2
 assert set(request["fingerprint_paths"])==set(request["expected_sha256"])
 for item,want in request["expected_sha256"].items():
  blob=subprocess.check_output(["git","show",request["expected_commit"]+":"+item],cwd=ROOT);assert hashlib.sha256(blob).hexdigest()==want,(host,item)
 assert request["command"][:5]==["timeout","--foreground","--signal=TERM","--kill-after=60s","4200s"]
 assert request["command"][request["command"].index("--arm")+1]==host
 assert request["environment"]["NEURON_LOGICAL_NC_CONFIG"]=="2" and request["environment"]["NEURON_CC_FLAGS"]=="--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
 contract=request["command"][request["command"].index("--request-contract-sha256")+1]
 auth_path=HERE/f"{host}-authorization.json";auth=json.loads(auth_path.read_text());assert auth["request_contract_sha256"]==contract==driver.request_contract(request)
 assert request["command"][request["command"].index("--authorization-sha256")+1]==hashlib.sha256(auth_path.read_bytes()).hexdigest()
 # This is the exact mutation performed by fresh_start_launch.REMOTE_SCRIPT before it writes .job.json.
 runtime=copy.deepcopy(request);runtime.update(started_at="2026-09-24T00:00:00+00:00",hostname=hostname,git_commit=request["expected_commit"],source_dirty=False,preflight_free_bytes=9*1024**3,source_sha256=dict(request["expected_sha256"]))
 assert len(driver.verify_runtime(runtime,host,request["label"],request["expected_commit"],contract))==64
 low=copy.deepcopy(runtime);low["preflight_free_bytes"]=request["min_free_bytes"]-1
 try:driver.verify_runtime(low,host,request["label"],request["expected_commit"],contract)
 except SystemExit:pass
 else:raise AssertionError("6.49 GiB runtime accepted")
 boundary=copy.deepcopy(runtime);boundary["preflight_free_bytes"]=request["min_free_bytes"];driver.verify_runtime(boundary,host,request["label"],request["expected_commit"],contract)
 attacks=[]
 for section,key,value in ((None,"environment",dict(request["environment"],NEURON_LOGICAL_NC_CONFIG="1")),(None,"command",request["command"][:-1]+["0"*64]),(None,"expected_commit","0"*40),("expected_sha256","prepare.py","0"*64),("source_sha256","prepare.py","0"*64)):
  bad=copy.deepcopy(runtime)
  if section is None:bad[key]=value
  else:bad[section][key]=value
  attacks.append(bad)
 for bad in attacks:
  try:driver.verify_runtime(bad,host,request["label"],request["expected_commit"],contract)
  except SystemExit:pass
  else:raise AssertionError("mutated launcher runtime accepted")
 assert job["request_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest() and job["checkpoint_prune_authorized"] is False
 assert not any(request["label"] in x for x in subprocess.check_output(["git","ls-files","research/run-logs"],cwd=ROOT,text=True).splitlines())
print("current-base dual-host immutable requests: PASS (no dispatch)")
