#!/usr/bin/env python3
import importlib.util,json,math,os,pathlib,tempfile
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("eval_recovery",HERE/"eval_recovery.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def metric(tokens=2097152,num_bytes=8418678,**updates):
 total=5776000.0
 value={"val_bpb":float(total/(num_bytes*math.log(2.0))),"num_tokens":tokens,"num_bytes":num_bytes,
  "total_nats":total,"eval_seconds":1.0,"eval_timeout":1200,"eval_over_timeout":False}
 value.update(updates);return json.dumps(value)
assert m.parse_metrics(metric(),2097152,8418678)["num_tokens"]==2097152
assert m.parse_metrics(metric(20971520,83462186),20971520,83462186)["num_bytes"]==83462186
for bad in ("",metric()+metric(),metric(val_bpb=1),metric(val_bpb=True),metric(val_bpb=float("nan")),
 metric(total_nats=1),metric(eval_seconds=1200.1),metric(eval_timeout=1199),metric(eval_over_timeout=True),
 metric(num_tokens=2097153),metric(num_bytes=8418679),metric(val_bpb=0.9),json.dumps({"val_bpb":1.0})):
 try:m.parse_metrics(bad,2097152,8418678)
 except SystemExit:pass
 else:raise AssertionError("invalid recovery metric accepted")

dirty={"RANK":"1","LOCAL_RANK":"1","WORLD_SIZE":"4","MASTER_ADDR":"bad","MASTER_PORT":"1",
 "TORCHELASTIC_RUN_ID":"bad","NEURON_COMPETITION_R1_SEED":"9","PYTHONPATH":"bad","PYTHONHOME":"bad",
 "NEURON_RT_VISIBLE_CORES":"0","NEURON_RT_NUM_CORES":"1","NEURON_CC_FLAGS":"bad","KEEP":"yes"}
clean=m.scrub(dirty)
assert clean["KEEP"]=="yes" and clean["NEURON_LOGICAL_NC_CONFIG"]=="2" and clean["NKI_VENV_PATH"]=="/home/ubuntu/workspace/native_venv"
assert not (set(dirty)-{"KEEP"})&set(clean)

authorization=json.loads((HERE/"eval-recovery-trn21-authorization.json").read_text())
assert authorization["training_authorized"] is False and authorization["checkpoint_prune_authorized"] is False
assert authorization["checkpoint_path"]==str(m.CHECKPOINT) and authorization["checkpoint_sha256"]==m.CHECKPOINT_SHA
source=(HERE/"eval_recovery.py").read_text()
assert "submission_rehearsal" not in source and "torchrun" not in source and "run_experiment" not in source
assert '[PYTHON,"prepare.py","eval-public","--train-py",TRAIN_REL,"--checkpoint",str(CHECKPOINT)]' in source
assert '"--eval-tokens","20971520","--timeout-seconds","1200"' in source
assert "weights_only=True" in source and "map_location='cpu'" in source

with tempfile.TemporaryDirectory(prefix="c595-eval-artifacts-") as temporary:
 root=pathlib.Path(temporary)/"checkout";root.mkdir();outside=pathlib.Path(temporary)/"outside";outside.mkdir()
 (root/"out").symlink_to(outside,target_is_directory=True)
 try:m.create_output(root,root/"out"/m.LABEL)
 except (OSError,SystemExit):pass
 else:raise AssertionError("symlinked output ancestor accepted")
 assert list(outside.iterdir())==[]
 (root/"out").unlink();m.create_output(root,root/"out"/m.LABEL)
 old_root,old_output=m.ROOT,m.OUTPUT;m.ROOT=root;m.OUTPUT=root/"out"/m.LABEL
 try:
  for name in m.BASE_ARTIFACTS:m.durable(m.OUTPUT/name,b"evidence-"+name.encode())
  members=m.inspect_artifacts(m.BASE_ARTIFACTS)
  m.validate_manifest({"schema":"c595-eval-recovery-artifact-manifest-v1","label":m.LABEL,"members":members})
  (m.OUTPUT/"extra").write_text("x")
  try:m.inspect_artifacts(m.BASE_ARTIFACTS)
  except SystemExit:pass
  else:raise AssertionError("extra artifact accepted")
  (m.OUTPUT/"extra").unlink();victim=m.OUTPUT/next(iter(m.BASE_ARTIFACTS));victim.unlink();victim.symlink_to(outside)
  try:m.inspect_artifacts(m.BASE_ARTIFACTS)
  except (OSError,SystemExit):pass
  else:raise AssertionError("artifact symlink accepted")
  victim.unlink();victim.write_bytes(b"mutated")
  try:m.validate_manifest({"schema":"c595-eval-recovery-artifact-manifest-v1","label":m.LABEL,"members":members})
  except SystemExit:pass
  else:raise AssertionError("mutated artifact accepted")
  for forged in ({"schema":"wrong","label":m.LABEL,"members":members},{"schema":"c595-eval-recovery-artifact-manifest-v1","label":m.LABEL,"members":{}},{}):
   try:m.validate_manifest(forged)
   except SystemExit:pass
   else:raise AssertionError("forged/truncated manifest accepted")
  completion={"candidate_sha256":"a"*64,"prepare_sha256":"b"*64,"request_file_sha256":"c"*64,"runtime_job_sha256":"d"*64,
   "artifact_manifest_sha256":"e"*64,"checkpoint_pre_sha256":"f"*64,"checkpoint_post_sha256":"0"*64,
   "ready_to_retrieve":True,"checkpoint_preserved":True}
  m.validate_completion(completion,completion)
  for forged in ({**completion,"request_file_sha256":"z"*64},{**completion,"ready_to_retrieve":False},{key:value for key,value in completion.items() if key!="runtime_job_sha256"}):
   try:m.validate_completion(forged,completion)
   except SystemExit:pass
   else:raise AssertionError("forged/truncated completion accepted")
 finally:m.ROOT,m.OUTPUT=old_root,old_output

all_durable=sorted(m.BASE_ARTIFACTS|{"artifact-manifest.json","ready-to-retrieve.json"})
with tempfile.TemporaryDirectory(prefix="c595-short-write-") as temporary:
 root=pathlib.Path(temporary)/"checkout";root.mkdir();m.create_output(root,root/"out"/m.LABEL)
 old_root,old_output,original_write=m.ROOT,m.OUTPUT,m.os.write;m.ROOT=root;m.OUTPUT=root/"out"/m.LABEL
 def short_write(fd,view):return original_write(fd,view[:max(1,min(3,len(view)))])
 m.os.write=short_write
 try:
  for name in all_durable:
   payload=("repeated-short-write-"+name).encode();m.durable(m.OUTPUT/name,payload)
   assert m.artifact_bytes(name)==payload
 finally:m.os.write=original_write;m.ROOT,m.OUTPUT=old_root,old_output

for failed_name in all_durable:
 with tempfile.TemporaryDirectory(prefix="c595-zero-write-") as temporary:
  root=pathlib.Path(temporary)/"checkout";root.mkdir();m.create_output(root,root/"out"/m.LABEL)
  old_root,old_output,original_write=m.ROOT,m.OUTPUT,m.os.write;m.ROOT=root;m.OUTPUT=root/"out"/m.LABEL
  m.os.write=lambda _fd,_view:0
  try:
   try:m.durable(m.OUTPUT/failed_name,b"must-not-persist")
   except SystemExit:pass
   else:raise AssertionError("zero write accepted for "+failed_name)
   assert not (m.OUTPUT/failed_name).exists() and not (m.OUTPUT/"ready-to-retrieve.json").exists()
  finally:m.os.write=original_write;m.ROOT,m.OUTPUT=old_root,old_output
print("c595 evaluator-only recovery static/adversarial gates: PASS")
