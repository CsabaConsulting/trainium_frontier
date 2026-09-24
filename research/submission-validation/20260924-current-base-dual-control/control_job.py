#!/usr/bin/env python3
"""Fail-closed dual-host current-base control driver; never submits or prunes."""
import argparse,hashlib,importlib.util,io,json,math,os,pathlib,re,socket,stat,subprocess,sys,tarfile
from decimal import Decimal,InvalidOperation
ROOT=pathlib.Path(__file__).resolve().parents[3]; HERE=pathlib.Path(__file__).resolve().parent
PACKAGE_REL="submissions/current_base_matrixlr015_control"
ARMS={"trn21":{"hostname":"ip-172-31-34-85","label":"fs-20260924-current-base-control-trn21-r1"},"trn22":{"hostname":"ip-172-31-32-253","label":"fs-20260924-current-base-control-trn22-r1"}}
BASE_SHA=TRAIN_SHA="116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d"
LAUNCH_SHA="b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd"
PREPARE_SHA="5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
CONTROL=Decimal("0.993231940827686"); KEEP=Decimal("0.0015"); REPLICATE=Decimal("0.0025")
METRIC_KEYS={"val_bpb","num_tokens","num_bytes","total_nats","eval_seconds","eval_timeout","eval_over_timeout"}
FINITE_KEYS={"schema","finite","weights_only","map_location","tensor_count","checkpoint_sha256","checkpoint_bytes"}
BASE_ARTIFACTS={"training.log","training-rehearsal.json","training-steps.jsonl","checkpoint-finite.log","checkpoint-finite.json","primary-public1024.log","primary-eval-rehearsal.json","primary.metrics.json","fresh-public20m.log","eval-rehearsal.json","full20m.metrics.json","result.json","evidence.tar.gz"}
HEX64=set("0123456789abcdef")
RUNTIME_ADDITIONS={"started_at","hostname","git_commit","source_dirty","preflight_free_bytes","source_sha256"}
WARNING_TAIL=["  Overriding a previously registered kernel for the same operator and the same dispatch key","  operator: aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)","    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6","  dispatch key: PrivateUse1","  previous kernel: registered at /pytorch/build/aten/src/ATen/RegisterCPU_3.cpp:7637","       new kernel: registered at NeuronDispatcher.cpp:0 (function operator())"]
WARNING_FIRST=re.compile(r"^\[W\d+ \d{2}:\d{2}:\d{2}\.\d+ OperatorEntry\.cpp:208\] Warning: Warning only once for all operators,  other operators may also be overridden\.$")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def request_contract(data):
 """Digest exact operator intent without creating a commit/request self-reference."""
 base={k:data[k] for k in ("schema","host","expected_hostname","remote","origin","label","purpose","fingerprint_paths","environment","command","runner_owns_manifest","non_overwrite")}
 command=list(base["command"])
 for flag in ("--core-commit","--authorization-sha256","--request-contract-sha256"):
  try: command[command.index(flag)+1]="<"+flag[2:].upper().replace("-","_")+">"
  except (ValueError,IndexError): raise SystemExit("runtime command binding missing")
 base["command"]=command
 return hashlib.sha256(json.dumps(base,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def verify_runtime(data,host,label,core,contract):
 required={"schema","host","expected_hostname","remote","expected_commit","origin","label","purpose","fingerprint_paths","expected_sha256","environment","command","runner_owns_manifest","non_overwrite"}
 if set(data)!=required|RUNTIME_ADDITIONS: raise SystemExit("runtime job schema mismatch")
 if data["host"]!=host or data["label"]!=label or data["expected_commit"]!=core or data["git_commit"]!=core: raise SystemExit("runtime job identity mismatch")
 if data["hostname"]!=ARMS[host]["hostname"] or data["expected_hostname"]!=ARMS[host]["hostname"] or data["source_dirty"] is not False: raise SystemExit("runtime host/dirty mismatch")
 if request_contract(data)!=contract: raise SystemExit("runtime request contract mismatch")
 if set(data["fingerprint_paths"])!=set(data["expected_sha256"]) or not set(data["fingerprint_paths"])<=set(data["source_sha256"]): raise SystemExit("runtime fingerprint schema mismatch")
 for path in data["fingerprint_paths"]:
  want=data["expected_sha256"][path]
  if type(want) is not str or len(want)!=64 or any(c not in HEX64 for c in want) or data["source_sha256"][path]!=want: raise SystemExit("runtime fingerprint mismatch")
 return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def artifact_module(label):
 path=HERE/"artifact_io.py"
 spec=importlib.util.spec_from_file_location("verified_artifacts",path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 m.ROOT=ROOT; m.LABEL=label; m.OUTPUT=ROOT/"out"/label
 return m
def scrub(source):
 blocked={"RANK","LOCAL_RANK","WORLD_SIZE","LOCAL_WORLD_SIZE","GROUP_RANK","ROLE_RANK","ROLE_WORLD_SIZE","MASTER_ADDR","MASTER_PORT","PYTHONPATH","PYTHONHOME","NEURON_RT_VISIBLE_CORES","NEURON_RT_NUM_CORES","NEURON_CC_FLAGS"}
 return {k:v for k,v in source.items() if k not in blocked and not k.startswith("TORCHELASTIC_") and not k.startswith("NEURON_COMPETITION_R1_")}
def metrics(text,expected_tokens,expected_bytes=None):
 decoder=json.JSONDecoder(); found=[]
 for offset,char in enumerate(text):
  if char!="{": continue
  try: value,_=decoder.raw_decode(text[offset:])
  except json.JSONDecodeError: continue
  if isinstance(value,dict) and METRIC_KEYS.intersection(value): found.append(value)
 if len(found)!=1 or set(found[0])!=METRIC_KEYS: raise SystemExit("evaluation metric schema/count mismatch")
 value=found[0]
 for key in ("val_bpb","total_nats","eval_seconds"):
  if type(value[key]) is not float or value[key]<=0 or not math.isfinite(value[key]): raise SystemExit("invalid "+key)
 if type(value["num_tokens"]) is not int or value["num_tokens"]!=expected_tokens: raise SystemExit("token count mismatch")
 if type(value["num_bytes"]) is not int or value["num_bytes"]<=0 or (expected_bytes is not None and value["num_bytes"]!=expected_bytes): raise SystemExit("byte count mismatch")
 if type(value["eval_timeout"]) is not int or value["eval_timeout"]!=1200 or value["eval_over_timeout"] is not False or value["eval_seconds"]>1200.0: raise SystemExit("eval timeout mismatch")
 if not math.isclose(value["val_bpb"],value["total_nats"]/(value["num_bytes"]*math.log(2)),rel_tol=1e-12,abs_tol=1e-12): raise SystemExit("BPB math mismatch")
 return value
def parse_finite(text,checkpoint_sha,checkpoint_bytes):
 lines=text.splitlines()
 if len(lines)==1:receipt=lines[0]
 elif len(lines)==8 and WARNING_FIRST.fullmatch(lines[0]) and lines[1:7]==WARNING_TAIL:receipt=lines[7]
 else:raise SystemExit("unknown finite receipt output")
 try: value=json.loads(receipt)
 except json.JSONDecodeError: raise SystemExit("finite receipt is not JSON")
 expected={"schema":"c595-strict-finite-v1","finite":True,"weights_only":True,"map_location":"cpu","tensor_count":62,"checkpoint_sha256":checkpoint_sha,"checkpoint_bytes":checkpoint_bytes}
 if type(value) is not dict or set(value)!=FINITE_KEYS or value!=expected or receipt!=json.dumps(expected,sort_keys=True): raise SystemExit("finite receipt invalid")
 if len(checkpoint_sha)!=64 or any(c not in HEX64 for c in checkpoint_sha): raise SystemExit("finite digest invalid")
 return value
def decision_from_serialized(candidate):
 try: value=Decimal(candidate)
 except (InvalidOperation,TypeError): raise SystemExit("candidate BPB is not decimal")
 if not value.is_finite() or value<=0: raise SystemExit("candidate BPB invalid")
 improvement=CONTROL-value; keep=improvement>=KEEP
 return {"control_full20m_bpb":str(CONTROL),"candidate_full20m_bpb":str(value),"improvement":str(improvement),"decision_gate":"KEEP" if keep else "DISCARD","replication_required":keep and improvement<REPLICATE}
def make_archive(a,names):
 output=io.BytesIO()
 with tarfile.open(fileobj=output,mode="w:gz") as tar:
  for name in sorted(names):
   data=a.artifact_bytes(name); info=tarfile.TarInfo(name); info.size=len(data); info.mode=0o600; tar.addfile(info,io.BytesIO(data))
 return output.getvalue()
def verify_archive(data,a,names):
 with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as tar:
  members=tar.getmembers()
  if [x.name for x in members]!=sorted(names) or any(not x.isfile() for x in members): raise SystemExit("archive member mismatch")
  for member in members:
   if tar.extractfile(member).read()!=a.artifact_bytes(member.name): raise SystemExit("archive member mutation")
def revalidate_manifest(a,manifest,members):
 persisted=a.artifact_bytes("artifact-manifest.json")
 if json.loads(persisted)!=manifest: raise SystemExit("manifest content mutation")
 current=a.inspect_artifacts(BASE_ARTIFACTS|{"artifact-manifest.json"})
 if {name:current[name] for name in BASE_ARTIFACTS}!=members: raise SystemExit("manifest member mutation")
 return hashlib.sha256(persisted).hexdigest()
def copy_artifact(a,source,name):
 a.safe_regular(source); data,_=a.nofollow_bytes(source); a.durable(a.OUTPUT/name,data)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--arm",choices=ARMS,required=True); ap.add_argument("--authorization-sha256",required=True); ap.add_argument("--sandbox-root",type=pathlib.Path,required=True); ap.add_argument("--core-commit",required=True); ap.add_argument("--request-contract-sha256",required=True); args=ap.parse_args()
 host=args.arm;hostname=ARMS[host]["hostname"];label=ARMS[host]["label"]
 auth_path=HERE/(host+"-authorization.json"); auth=json.loads(auth_path.read_text())
 exact={"schema":"current-base-control-authorization-v1","arm":host,"host":host,"expected_hostname":hostname,"label":label,"package":PACKAGE_REL,"train_sha256":TRAIN_SHA,"single_causal_change":"none","seed":42,"max_train_seconds":1800,"smoke_authorized":False,"checkpoint_prune_authorized":False,"request_contract_sha256":args.request_contract_sha256,"state":"human_approved_pending_operator"}
 if sha(auth_path)!=args.authorization_sha256 or auth!=exact: raise SystemExit("authorization mismatch")
 if socket.gethostname()!=hostname or subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=args.core_commit: raise SystemExit("host/core mismatch")
 evidence=ROOT/"out"/label; work=ROOT/"out"/(label+"-work")
 if args.sandbox_root!=evidence or evidence.exists() or work.exists(): raise SystemExit("unsafe or existing sandbox")
 package=ROOT/PACKAGE_REL
 if sha(package/"train.py")!=TRAIN_SHA or sha(package/"launch-command.txt")!=LAUNCH_SHA or sha(ROOT/"prepare.py")!=PREPARE_SHA or sha(ROOT/"train.py")!=TRAIN_SHA or (package/"train.py").read_bytes()!=(ROOT/"train.py").read_bytes(): raise SystemExit("protected hash mismatch")
 a=artifact_module(label); runtime=ROOT/"research/run-logs"/(label+".job.json")
 try: a.safe_regular(runtime); runtime_bytes,_=a.nofollow_bytes(runtime)
 except (FileNotFoundError,OSError): raise SystemExit("runtime job missing or unsafe")
 runtime_data=json.loads(runtime_bytes)
 if args.request_contract_sha256!=auth["request_contract_sha256"]: raise SystemExit("request contract authorization mismatch")
 runtime_job_sha=verify_runtime(runtime_data,host,label,args.core_commit,args.request_contract_sha256)
 a.create_output(ROOT,evidence); work.mkdir(parents=True)
 env=scrub(os.environ); env.update(PYTHONUNBUFFERED="1",NEURON_LOGICAL_NC_CONFIG="2",NEURON_CC_FLAGS="--optlevel=1 --auto-cast matmult --auto-cast-type bf16")
 rehearsal=ROOT/"research/scratch/submission_rehearsal.py"; train_box=work/"train"; primary_box=work/"eval-primary"; eval_box=work/"eval-full20m"
 trained=subprocess.run([sys.executable,str(rehearsal),"--package",str(package),"--sandbox",str(train_box),"--mode","train","--fresh-caches"],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 a.durable(evidence/"training.log",trained.stdout.encode())
 if trained.returncode: raise SystemExit("full training failed")
 copy_artifact(a,train_box/"rehearsal.json","training-rehearsal.json"); copy_artifact(a,train_box/"steps.jsonl","training-steps.jsonl")
 checkpoint=train_box/"out/final.pt"; a.safe_regular(checkpoint); before_sha,before_bytes=a.nofollow_sha(checkpoint)
 finite="""import hashlib,json,os,sys,torch\np=sys.argv[1];fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);h=hashlib.sha256();z=0\nwhile True:\n b=os.read(fd,8388608)\n if not b:break\n h.update(b);z+=len(b)\nos.close(fd);v=torch.load(p,map_location='cpu',weights_only=True);s=[v];n=0\nwhile s:\n x=s.pop()\n if isinstance(x,dict):s.extend(x.values())\n elif isinstance(x,(list,tuple)):s.extend(x)\n elif isinstance(x,torch.Tensor):\n  n+=1\n  if (x.is_floating_point() or x.is_complex()) and not torch.isfinite(x).all().item():raise SystemExit('nonfinite')\nprint(json.dumps({'schema':'c595-strict-finite-v1','finite':True,'weights_only':True,'map_location':'cpu','tensor_count':n,'checkpoint_sha256':h.hexdigest(),'checkpoint_bytes':z},sort_keys=True))"""
 checked=subprocess.run([sys.executable,"-c",finite,str(checkpoint)],cwd=ROOT,env=scrub(env),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT); a.durable(evidence/"checkpoint-finite.log",checked.stdout.encode())
 if checked.returncode: raise SystemExit("checkpoint finite scan failed")
 finite_value=parse_finite(checked.stdout,before_sha,before_bytes); a.durable_json(evidence/"checkpoint-finite.json",finite_value); a.persisted_json("checkpoint-finite.json",finite_value,"c595-strict-finite-v1")
 primary_done=subprocess.run([sys.executable,str(rehearsal),"--package",str(package),"--sandbox",str(primary_box),"--mode","eval","--checkpoint",str(checkpoint),"--eval-tokens","2097152","--fresh-caches"],cwd=ROOT,env=scrub(env),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);a.durable(evidence/"primary-public1024.log",primary_done.stdout.encode())
 if primary_done.returncode:raise SystemExit("fresh primary evaluation failed")
 primary=metrics(primary_done.stdout,2097152);primary_line=next((x for x in primary_done.stdout.splitlines() if x.startswith("REHEARSAL_COMMAND ")),None);primary_command=json.loads(primary_line.removeprefix("REHEARSAL_COMMAND ")) if primary_line else []
 primary_tail=["--device","neuron","--batch-size","1","--seq-len","1024","--eval-tokens","2097152","--timeout-seconds","1200"]
 if not any(primary_command[i:i+len(primary_tail)]==primary_tail for i in range(len(primary_command))):raise SystemExit("fresh primary command mismatch")
 copy_artifact(a,primary_box/"rehearsal.json","primary-eval-rehearsal.json");primary_receipt={"schema":"current-base-primary-metrics-v1","argv":primary_command,"metrics":primary};a.durable_json(evidence/"primary.metrics.json",primary_receipt);a.persisted_json("primary.metrics.json",primary_receipt,"current-base-primary-metrics-v1")
 if a.nofollow_sha(checkpoint)!=(before_sha,before_bytes):raise SystemExit("checkpoint changed after primary")
 evaluated=subprocess.run([sys.executable,str(rehearsal),"--package",str(package),"--sandbox",str(eval_box),"--mode","eval","--checkpoint",str(checkpoint),"--eval-tokens","20971520","--fresh-caches"],cwd=ROOT,env=scrub(env),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT); a.durable(evidence/"fresh-public20m.log",evaluated.stdout.encode())
 if evaluated.returncode: raise SystemExit("fresh full20m evaluation failed")
 value=metrics(evaluated.stdout,20971520,83462186); line=next((x for x in evaluated.stdout.splitlines() if x.startswith("REHEARSAL_COMMAND ")),None); command=json.loads(line.removeprefix("REHEARSAL_COMMAND ")) if line else []
 tail=["--device","neuron","--batch-size","1","--seq-len","1024","--eval-tokens","20971520","--timeout-seconds","1200"]
 if not any(command[i:i+len(tail)]==tail for i in range(len(command))): raise SystemExit("fresh eval command mismatch")
 copy_artifact(a,eval_box/"rehearsal.json","eval-rehearsal.json")
 metric_receipt={"schema":"current-base-full20m-metrics-v1","argv":command,"metrics":value}; a.durable_json(evidence/"full20m.metrics.json",metric_receipt); a.persisted_json("full20m.metrics.json",metric_receipt,"current-base-full20m-metrics-v1")
 after_sha,after_bytes=a.nofollow_sha(checkpoint)
 if (after_sha,after_bytes)!=(before_sha,before_bytes): raise SystemExit("checkpoint changed")
 result={"schema":"current-base-control-result-v1","label":label,"host":host,"hostname":hostname,"candidate_sha256":TRAIN_SHA,"seed":42,"world_size":4,"lnc":2,"compiler_flags":"--optlevel=1 --auto-cast matmult --auto-cast-type bf16","max_train_seconds":1800,"smoke":False,"checkpoint_path":str(checkpoint),"checkpoint_sha256":after_sha,"checkpoint_bytes":after_bytes,"checkpoint_preserved":True,"finite_scan":finite_value,"primary":primary,"full20m":value,"ready_to_retrieve":False,"prune_authorized":False,"cache_cleanup_authorized":False}
 a.durable_json(evidence/"result.json",result); a.persisted_json("result.json",result,"current-base-control-result-v1")
 prearchive=BASE_ARTIFACTS-{"evidence.tar.gz"}; archive=make_archive(a,prearchive); a.durable(evidence/"evidence.tar.gz",archive); verify_archive(a.artifact_bytes("evidence.tar.gz"),a,prearchive)
 post_sha,post_bytes=a.nofollow_sha(checkpoint)
 if (post_sha,post_bytes)!=(before_sha,before_bytes): raise SystemExit("checkpoint changed after archive")
 members=a.inspect_artifacts(BASE_ARTIFACTS); manifest={"schema":"current-base-control-artifact-manifest-v1","label":label,"members":members}; a.durable_json(evidence/"artifact-manifest.json",manifest); manifest_sha=revalidate_manifest(a,manifest,members)
 completion={"schema":"current-base-control-completion-v1","label":label,"host":host,"hostname":hostname,"core_commit":args.core_commit,"candidate_sha256":TRAIN_SHA,"prepare_sha256":PREPARE_SHA,"request_contract_sha256":args.request_contract_sha256,"runtime_job_sha256":runtime_job_sha,"artifact_manifest_sha256":manifest_sha,"artifact_members":members,"checkpoint_path":str(checkpoint),"checkpoint_pre_sha256":before_sha,"checkpoint_pre_bytes":before_bytes,"checkpoint_post_sha256":post_sha,"checkpoint_post_bytes":post_bytes,"checkpoint_preserved":True,"archive_sha256":hashlib.sha256(archive).hexdigest(),"archive_bytes":len(archive),"prune_authorized":False,"cache_cleanup_authorized":False,"ready_to_retrieve":True}
 a.durable_json(evidence/"ready-to-retrieve.json",completion); completion_bytes=a.artifact_bytes("ready-to-retrieve.json"); completion_sha=hashlib.sha256(completion_bytes).hexdigest()
 if json.loads(completion_bytes)!=completion or a.inspect_artifacts(BASE_ARTIFACTS|{"artifact-manifest.json","ready-to-retrieve.json"})["ready-to-retrieve.json"]["sha256"]!=completion_sha: raise SystemExit("completion revalidation failed")
 print(json.dumps({"result":result,"artifact_manifest_sha256":manifest_sha,"completion_receipt_sha256":completion_sha,"ready_to_retrieve":True},sort_keys=True))
if __name__=="__main__": main()
