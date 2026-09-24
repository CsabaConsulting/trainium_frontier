#!/usr/bin/env python3
import copy,hashlib,importlib.util,json,os,pathlib,tempfile,types,numpy as np
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("probe",HERE/"component_probe.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base=json.loads((HERE/"component-probe-authorization.json").read_text())
try:m.validate_authorization(base,hashlib.sha256(b"pending").hexdigest())
except SystemExit:pass
else:raise AssertionError("pending authorization accepted")
valid=copy.deepcopy(base);valid.update(state="authorized_component_probe_only",expected_hostname="host",expected_core_commit="a"*40,expected_request_contract_sha256="b"*64,candidate_train_py="train-candidate.py",candidate_train_sha256="c"*64,track_a_receipt_path="receipt.json",track_a_receipt_sha256="d"*64,checkpoint_path="/exact/final.pt",checkpoint_sha256="e"*64,checkpoint_bytes=123)
m.validate_authorization(valid,"f"*64)
request=json.loads((HERE/"component-probe-request.json").read_text());output=pathlib.Path("/tmp/track-b2-probe-test/result.json");request.update(state="authorized_component_probe_only",dispatch_authorized=True,expected_hostname="host",expected_commit="a"*40,authorization_sha256="f"*64,output_root=str(output.parent),output=str(output))
args=types.SimpleNamespace(request_file=HERE/"component-probe-request.json",authorization=HERE/"component-probe-authorization.json",authorization_sha256="f"*64,core_commit="a"*40,output=output)
request["command"]=["torchrun","--standalone","--nproc_per_node=4",str((HERE/"component_probe.py").relative_to(m.ROOT)),"--request-file",str(args.request_file.relative_to(m.ROOT)),"--authorization",str(args.authorization.relative_to(m.ROOT)),"--authorization-sha256",args.authorization_sha256,"--core-commit",args.core_commit,"--output",str(output)]
valid["expected_request_contract_sha256"]=m.request_contract(request);data=(json.dumps(request,sort_keys=True)+"\n").encode();m.validate_request(request,data,args,valid)
mutated=copy.deepcopy(request);mutated["environment"]["PYTHONUNBUFFERED"]="0"
try:m.validate_request(mutated,(json.dumps(mutated)+"\n").encode(),args,valid)
except SystemExit:pass
else:raise AssertionError("semantically mutated request accepted")
attacks=[]
for key,value in (("state","draft"),("training_authorized",True),("control_train_sha256","0"*64),("candidate_train_py",valid["control_train_py"]),("candidate_train_sha256",m.CONTROL_SHA),("checkpoint_bytes",0),("world_size",1),("lnc",1),("expected_request_contract_sha256","not-a-digest")):
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
 # Symlinked parent traversal and output creation must fail before writing.
 real=root/"real";real.mkdir();parent_link=root/"parent-link";parent_link.symlink_to(real,target_is_directory=True)
 try:m.secure_write_new_tree(parent_link/"owned",parent_link/"owned"/"result.json",b"x")
 except SystemExit:pass
 else:raise AssertionError("symlinked output parent accepted")
 checkpoint=root/"checkpoint";checkpoint.write_bytes(b"before");before=m.nofollow_identity(checkpoint);checkpoint.write_bytes(b"after")
 assert before!=m.nofollow_identity(checkpoint)
# Candidate cannot forge correctness with equal self-reported tensors/booleans.
for forged in ({"schema":"track-b2-native-bf16-ordering-v1","duplicates_present":True,"dense_param":1,"touched_param":1},{"schema":"track-b2-native-touched-result-v1","param":1,"logical_moment":1,"physical_untouched_before":1,"physical_untouched_after":1,"passed":True}):
 try:m.validate_candidate_schema(forged)
 except SystemExit:pass
 else:raise AssertionError("self-reported correctness accepted")
class FakeTensor:
 def __init__(self,value):self.value=np.asarray(value,dtype=np.float32).copy()
 def div_(self,value):self.value/=np.float32(value);return self
class FakeDist:
 class ReduceOp:SUM="sum"
 def __init__(self,sums):self.sums=sums;self.index=0
 def all_reduce(self,value,op):assert op==self.ReduceOp.SUM;value.value[:]=self.sums[self.index];self.index+=1
def fake_bytes(value):return value.value.tobytes()
# Integration: the same production helper hashes each real local fixture before FakeDist mutates
# it, then hashes the identical four-rank SUM/AVG result after mutation.
raw=[[FakeTensor(np.arange(12,dtype=np.float32).reshape(3,4)+rank*17+table) for table in range(4)] for rank in range(4)]
sums=[sum((raw[rank][table].value for rank in range(4)),np.zeros((3,4),dtype=np.float32)) for table in range(4)]
local_hashes=[];global_hashes=[]
for rank in range(4):
 local,global_=m.reduce_clipped_with_evidence(raw[rank],FakeDist(sums),fake_bytes);local_hashes.append(local);global_hashes.append(global_)
assert len(set(local_hashes))==4 and len(set(global_hashes))==1
valid_ranks=[{"rank":rank,"local_contribution_sha256":local_hashes[rank],"global_gradient_sha256":global_hashes[rank],"param_sha256":"a"*64,"logical_moment_sha256":"b"*64,"untouched_state_sha256":str(rank+4)*64} for rank in range(4)]
m.validate_rank_evidence(valid_ranks)
for forged in (valid_ranks[:3],[valid_ranks[0]]*4,[dict(valid_ranks[i],rank=0) for i in range(4)],[dict(item,global_gradient_sha256=("d"*64 if i==3 else item["global_gradient_sha256"])) for i,item in enumerate(valid_ranks)],[dict(item,param_sha256=("e"*64 if i==2 else item["param_sha256"])) for i,item in enumerate(valid_ranks)]):
 try:m.validate_rank_evidence(forged)
 except SystemExit:pass
 else:raise AssertionError("missing/mismatched rank contributions accepted")
print("Track B2 native component probe fail-closed contracts: PASS (no dispatch)")
