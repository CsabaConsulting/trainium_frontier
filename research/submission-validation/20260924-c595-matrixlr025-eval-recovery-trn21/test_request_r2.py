#!/usr/bin/env python3
"""No-SSH immutable request and authenticated-checkout launcher boundary test."""
import hashlib,importlib.util,json,pathlib,subprocess,sys,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
request_path=HERE/"request-r2.json";request=json.loads(request_path.read_text());job=json.loads((HERE/"job-r2.json").read_text())
assert request["expected_commit"]=="68b1df78e65f9543300717949e316699c07d2549"
assert request["label"]=="fs-20260924-c595-matrixlr025-eval-recovery-trn21-r2"
assert request["remote"].endswith("_r2") and set(request["fingerprint_paths"])==set(request["expected_sha256"])
assert "request-r2.json" not in "\n".join(request["fingerprint_paths"])
tracked=set(subprocess.check_output(["git","ls-tree","-r","--name-only",request["expected_commit"]],cwd=ROOT,text=True).splitlines())
for path,want in request["expected_sha256"].items():
 assert path in tracked,path
 blob=subprocess.check_output(["git","show",request["expected_commit"]+":"+path],cwd=ROOT);assert hashlib.sha256(blob).hexdigest()==want
assert request["environment"]["NEURON_CC_FLAGS"]=="--optlevel=1" and request["command"][0]=="timeout"
assert job["request_sha256"]==hashlib.sha256(request_path.read_bytes()).hexdigest() and job["expected_commit"]==request["expected_commit"]
# Execute the real remote authentication prefix against a complete temporary checkout.
sys.path.insert(0,str(ROOT/"research"));import fresh_start_launch as launcher
with tempfile.TemporaryDirectory() as directory:
 checkout=pathlib.Path(directory)/"checkout"
 subprocess.run(["git","clone","--quiet","--no-checkout",str(ROOT),str(checkout)],check=True)
 subprocess.run(["git","checkout","--quiet",request["expected_commit"]],cwd=checkout,check=True)
 probe=dict(request);probe["remote"]=str(checkout)
 prefix=launcher.REMOTE_SCRIPT.split("# CHECKOUT_AUTHENTICATION_BOUNDARY",1)[0]+"\nprint('AUTHENTICATED_TO_GUARD')\n"
 done=subprocess.run([sys.executable,"-c",prefix],input=json.dumps(probe),text=True,capture_output=True)
 assert done.returncode==0,done.stderr
 assert done.stdout.strip()=="AUTHENTICATED_TO_GUARD"
# The authenticated payload is then written as the exact guard job schema.
assert "job = fdir / (m['label'] + '.job.json')" in launcher.REMOTE_SCRIPT
assert "str(r / 'research/fresh_start_guard.py')" in launcher.REMOTE_SCRIPT
assert not any(request["label"] in x for x in subprocess.check_output(["git","ls-files","research/run-logs"],cwd=ROOT,text=True).splitlines())
print("LR evaluator recovery r2 launcher authentication dry-run: PASS (no SSH)")
