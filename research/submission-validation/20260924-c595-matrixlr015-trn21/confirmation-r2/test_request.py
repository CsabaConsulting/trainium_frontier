#!/usr/bin/env python3
"""No-SSH exact-core authentication test for LR0.015 confirmation r2."""
import hashlib,json,pathlib,subprocess,sys,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).resolve().parent
path=HERE/"request.json";request=json.loads(path.read_text());job=json.loads((HERE/"job.json").read_text())
assert request["expected_commit"]=="e7f11c38dc3b5e68b373322e10cfef580f1e8d36"
assert request["label"]=="fs-20260924-c595-matrixlr015-confirm-trn21-r2" and request["remote"].endswith("_r2")
assert set(request["fingerprint_paths"])==set(request["expected_sha256"]) and "request.json" not in "\n".join(request["fingerprint_paths"])
tracked=set(subprocess.check_output(["git","ls-tree","-r","--name-only",request["expected_commit"]],cwd=ROOT,text=True).splitlines())
for item,want in request["expected_sha256"].items():
 assert item in tracked,item
 blob=subprocess.check_output(["git","show",request["expected_commit"]+":"+item],cwd=ROOT);assert hashlib.sha256(blob).hexdigest()==want
assert request["command"][:5]==["timeout","--foreground","--signal=TERM","--kill-after=60s","4200s"]
assert request["environment"]["NEURON_CC_FLAGS"]=="--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
assert job["request_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest() and job["expected_commit"]==request["expected_commit"]
sys.path.insert(0,str(ROOT/"research"));import fresh_start_launch as launcher
with tempfile.TemporaryDirectory() as directory:
 checkout=pathlib.Path(directory)/"checkout";subprocess.run(["git","clone","--quiet","--no-checkout",str(ROOT),str(checkout)],check=True);subprocess.run(["git","checkout","--quiet",request["expected_commit"]],cwd=checkout,check=True)
 probe=dict(request);probe["remote"]=str(checkout);prefix=launcher.REMOTE_SCRIPT.split("# CHECKOUT_AUTHENTICATION_BOUNDARY",1)[0]+"\nprint('AUTHENTICATED_TO_GUARD')\n"
 done=subprocess.run([sys.executable,"-c",prefix],input=json.dumps(probe),text=True,capture_output=True);assert done.returncode==0,done.stderr;assert done.stdout.strip()=="AUTHENTICATED_TO_GUARD"
assert "job = fdir / (m['label'] + '.job.json')" in launcher.REMOTE_SCRIPT and "fresh_start_guard.py" in launcher.REMOTE_SCRIPT
assert not any(request["label"] in x for x in subprocess.check_output(["git","ls-files","research/run-logs"],cwd=ROOT,text=True).splitlines())
print("LR0.015 confirmation r2 immutable launcher request: PASS (no SSH)")
