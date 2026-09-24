#!/usr/bin/env python3
import hashlib,importlib.util,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ROOT/"train.py")==sha(ROOT/"submissions/current_base_matrixlr015_control/train.py")=="116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d"
assert sha(ROOT/"prepare.py")=="5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
spec=importlib.util.spec_from_file_location("job",HERE/"control_job.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def value(tokens,bytes_):
 nats=5800000.0;return {"val_bpb":nats/(bytes_*math.log(2)),"num_tokens":tokens,"num_bytes":bytes_,"total_nats":nats,"eval_seconds":1.0,"eval_timeout":1200,"eval_over_timeout":False}
assert m.metrics(json.dumps(value(2097152,8418678)),2097152)["num_bytes"]==8418678
assert m.metrics(json.dumps(value(20971520,83462186)),20971520,83462186)["num_bytes"]==83462186
for bad in (value(1,8418678),{**value(20971520,83462186),"num_bytes":1},{**value(20971520,83462186),"val_bpb":True}):
 try:m.metrics(json.dumps(bad),20971520,83462186)
 except SystemExit:pass
 else:raise AssertionError("bad metrics accepted")
receipt={"schema":"c595-strict-finite-v1","finite":True,"weights_only":True,"map_location":"cpu","tensor_count":62,"checkpoint_sha256":"a"*64,"checkpoint_bytes":99}
warning="\n".join(["[W923 23:48:21.812477753 OperatorEntry.cpp:208] Warning: Warning only once for all operators,  other operators may also be overridden.",*m.WARNING_TAIL,json.dumps(receipt,sort_keys=True)])
assert m.parse_finite(warning,"a"*64,99)==receipt
for attack in (warning+"\nextra",warning.replace("aten::gather.out","other"),json.dumps({**receipt,"tensor_count":61},sort_keys=True)):
 try:m.parse_finite(attack,"a"*64,99)
 except SystemExit:pass
 else:raise AssertionError("bad finite output accepted")
proposal=json.loads((HERE/"EXPERIMENT_PROPOSAL.json").read_text());assert proposal["single_causal_change"]=="none" and proposal["hosts"]==["trn21","trn22"]
assert set(m.ARMS)=={"trn21","trn22"} and "primary-public1024.log" in m.BASE_ARTIFACTS and "fresh-public20m.log" in m.BASE_ARTIFACTS
print("current-base dual control static gates: PASS")
