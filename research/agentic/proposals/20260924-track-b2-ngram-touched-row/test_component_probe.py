#!/usr/bin/env python3
import copy,hashlib,importlib.util,json,os,pathlib,tempfile
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("probe",HERE/"component_probe.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base=json.loads((HERE/"component-probe-authorization.json").read_text())
try:m.validate_authorization(base,hashlib.sha256(b"pending").hexdigest())
except SystemExit:pass
else:raise AssertionError("pending authorization accepted")
valid=copy.deepcopy(base);valid.update(state="authorized_component_probe_only",expected_hostname="host",expected_core_commit="a"*40,expected_request_sha256="b"*64,candidate_train_py="train-candidate.py",candidate_train_sha256="c"*64,track_a_receipt_path="receipt.json",track_a_receipt_sha256="d"*64,checkpoint_path="/exact/final.pt",checkpoint_sha256="e"*64,checkpoint_bytes=123)
m.validate_authorization(valid,"f"*64)
attacks=[]
for key,value in (("state","draft"),("training_authorized",True),("control_train_sha256","0"*64),("candidate_train_py",valid["control_train_py"]),("candidate_train_sha256",m.CONTROL_SHA),("checkpoint_bytes",0),("world_size",1),("lnc",1),("expected_request_sha256","not-a-digest")):
 bad=copy.deepcopy(valid);bad[key]=value;attacks.append(bad)
bad=copy.deepcopy(valid);bad["extra"]=1;attacks.append(bad)
for bad in attacks:
 try:m.validate_authorization(bad,"f"*64)
 except SystemExit:pass
 else:raise AssertionError("forged authorization accepted")
with tempfile.TemporaryDirectory() as td:
 root=pathlib.Path(td);target=root/"target";target.write_text("x");link=root/"link";link.symlink_to(target)
 try:m.nofollow_bytes(link)
 except SystemExit:pass
 else:raise AssertionError("symlink accepted")
print("Track B2 native component probe fail-closed contracts: PASS (no dispatch)")
