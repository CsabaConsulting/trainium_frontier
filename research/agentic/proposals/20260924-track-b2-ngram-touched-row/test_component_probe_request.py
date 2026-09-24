#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[3];path=HERE/"component-probe-request.json";request=json.loads(path.read_text());core=request["expected_commit"]
assert request["state"]=="blocked_non_dispatchable" and request["command"]==[] and request["dispatch_authorized"] is False and request["training_authorized"] is False
assert request["expected_hostname"] is None and len(request["blocked_fields"])==6
assert set(request["fingerprint_paths"])==set(request["expected_sha256"])
for rel,want in request["expected_sha256"].items():
 data=subprocess.check_output(["git","show",core+":"+rel],cwd=ROOT);assert hashlib.sha256(data).hexdigest()==want
auth=(ROOT/request["authorization"]).read_bytes();assert hashlib.sha256(auth).hexdigest()==request["authorization_sha256"]
assert request["planned_command"][:3]==["torchrun","--standalone","--nproc_per_node=4"]
print("Track B2 blocked component request fingerprints: PASS (non-dispatchable)")
