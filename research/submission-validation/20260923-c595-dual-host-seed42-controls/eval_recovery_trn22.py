#!/usr/bin/env python3
"""Fail-closed evaluator-only recovery for the retained trn22 c595 checkpoint."""
import argparse, hashlib, json, math, os, pathlib, socket, stat, subprocess, time

ROOT=pathlib.Path(__file__).resolve().parents[3]
PYTHON="/home/ubuntu/workspace/native_venv/bin/python"
SOURCE_COMMIT="067ff2326bbad1d4ed7060a802e7cee3e8448545"
TRAIN_REL="submissions/dense6x1024_direct_review/train.py"
TRAIN_SHA="c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537"
PREPARE_SHA="5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
CHECKPOINT=pathlib.Path("/home/ubuntu/trainium_freshstart_20260924_c595_control_trn22_r2/out/fs-20260924-c595-control-trn22-r2/train/out/final.pt")
CHECKPOINT_SHA="b12fbc758f63aeedec09752bdaa43b5267d3d5a2eb410ab6fbe8f80f641435ea"
CHECKPOINT_BYTES=469831009
HOSTNAME="ip-172-31-32-253"
LABEL="fs-20260924-c595-eval-recovery-trn22-r1"
OUTPUT=ROOT/"out"/LABEL
METRIC_KEYS={"val_bpb","num_tokens","num_bytes","total_nats","eval_seconds","eval_timeout","eval_over_timeout"}
BASE_ARTIFACTS={"checkpoint-finite.log","primary-public1024.log","primary-public1024.metrics.json",
 "secondary-public20m.log","secondary-public20m.metrics.json","result.json"}
HEX64=set("0123456789abcdef")

def open_directory(path):
 info=os.lstat(path)
 if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):raise SystemExit("unsafe output ancestor: "+str(path))
 return os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
def create_output(root=ROOT,output=OUTPUT):
 if not root.is_absolute() or output!=root/"out"/LABEL:raise SystemExit("output path escape")
 root_fd=open_directory(root)
 try:
  try:os.mkdir("out",0o700,dir_fd=root_fd);os.fsync(root_fd)
  except FileExistsError:pass
  out_fd=os.open("out",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=root_fd)
  try:
   out_info=os.stat("out",dir_fd=root_fd,follow_symlinks=False)
   if not stat.S_ISDIR(out_info.st_mode):raise SystemExit("unsafe output parent")
   try:os.mkdir(LABEL,0o700,dir_fd=out_fd);os.fsync(out_fd)
   except FileExistsError:raise SystemExit("recovery output already exists")
  finally:os.close(out_fd)
 finally:os.close(root_fd)
def output_fd():
 root_fd=open_directory(ROOT)
 try:
  out_fd=os.open("out",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=root_fd)
  try:return os.open(LABEL,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=out_fd)
  finally:os.close(out_fd)
 finally:os.close(root_fd)
def write_all(fd,data,writer=os.write):
 view=memoryview(data);offset=0
 while offset<len(view):
  count=writer(fd,view[offset:])
  if type(count) is not int or count<=0 or count>len(view)-offset:raise SystemExit("invalid durable write count")
  offset+=count
 return offset
def durable(path,data):
 if path.parent!=OUTPUT or path.name in ("",".",".."):raise SystemExit("artifact path escape")
 directory=output_fd()
 try:
  fd=os.open(path.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
  try:
   try:
    if write_all(fd,data,os.write)!=len(data):raise SystemExit("incomplete durable write")
    os.fsync(fd)
   finally:os.close(fd)
  except BaseException:
   os.unlink(path.name,dir_fd=directory);os.fsync(directory);raise
  os.fsync(directory)
 finally:os.close(directory)
def durable_json(path,value):durable(path,(json.dumps(value,indent=2,sort_keys=True)+"\n").encode())
def artifact_bytes(name):
 directory=output_fd()
 try:
  info=os.stat(name,dir_fd=directory,follow_symlinks=False)
  if not stat.S_ISREG(info.st_mode):raise SystemExit("artifact not regular: "+name)
  fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
  try:
   if os.fstat(fd).st_size!=info.st_size:raise SystemExit("artifact changed while opening")
   chunks=[]
   while True:
    chunk=os.read(fd,1024*1024)
    if not chunk:break
    chunks.append(chunk)
   return b"".join(chunks)
  finally:os.close(fd)
 finally:os.close(directory)
def persisted_json(name,expected,schema):
 data=artifact_bytes(name)
 try:value=json.loads(data)
 except (UnicodeDecodeError,json.JSONDecodeError):raise SystemExit("persisted JSON invalid: "+name)
 if value!=expected or value.get("schema")!=schema:raise SystemExit("persisted JSON schema/content mismatch: "+name)
 canonical=(json.dumps(expected,indent=2,sort_keys=True)+"\n").encode()
 if data!=canonical:raise SystemExit("persisted JSON bytes mismatch: "+name)
 return value
def inspect_artifacts(expected):
 directory=output_fd()
 try:names=set(os.listdir(directory))
 finally:os.close(directory)
 if names!=set(expected):raise SystemExit("artifact allowlist mismatch")
 result={}
 for name in sorted(names):
  data=artifact_bytes(name);result[name]={"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
 return result
def validate_manifest(value):
 if set(value)!={"schema","label","members"} or value["schema"]!="c595-eval-recovery-artifact-manifest-v1" or value["label"]!=LABEL:raise SystemExit("manifest schema mismatch")
 if value["members"]!=inspect_artifacts(BASE_ARTIFACTS):raise SystemExit("manifest member mutation")
 return value
def validate_completion(value,expected):
 if set(value)!=set(expected) or value!=expected:raise SystemExit("completion receipt mismatch")
 for key in ("candidate_sha256","prepare_sha256","request_file_sha256","runtime_job_sha256","artifact_manifest_sha256","checkpoint_pre_sha256","checkpoint_post_sha256"):
  digest=value[key]
  if type(digest) is not str or len(digest)!=64 or any(c not in HEX64 for c in digest):raise SystemExit("completion digest invalid")
 if value["ready_to_retrieve"] is not True or value["checkpoint_preserved"] is not True:raise SystemExit("completion not ready")
 return value
def safe_regular(path):
 current=pathlib.Path(path.anchor)
 for part in path.parts[1:]:
  current/=part;info=os.lstat(current)
  if stat.S_ISLNK(info.st_mode):raise SystemExit("symlink rejected: "+str(current))
 info=os.lstat(path)
 if not stat.S_ISREG(info.st_mode):raise SystemExit("checkpoint is not regular")
 return info
def nofollow_bytes(path):
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  info=os.fstat(fd)
  if not stat.S_ISREG(info.st_mode):raise SystemExit("not regular")
  chunks=[]
  while True:
   block=os.read(fd,8*1024*1024)
   if not block:break
   chunks.append(block)
  return b"".join(chunks),info.st_size
 finally:os.close(fd)
def nofollow_sha(path):
 data,size=nofollow_bytes(path);return hashlib.sha256(data).hexdigest(),size
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def scrub(source):
 exact={"RANK","LOCAL_RANK","WORLD_SIZE","LOCAL_WORLD_SIZE","GROUP_RANK","ROLE_RANK","ROLE_WORLD_SIZE",
  "MASTER_ADDR","MASTER_PORT","PYTHONPATH","PYTHONHOME","NEURON_RT_VISIBLE_CORES","NEURON_RT_NUM_CORES","NEURON_CC_FLAGS"}
 result={}
 for key,value in source.items():
  if key in exact or key.startswith("TORCHELASTIC_") or key.startswith("NEURON_COMPETITION_R1_"):continue
  result[key]=value
 result.update(PATH="/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
  NKI_VENV_PATH="/home/ubuntu/workspace/native_venv",NEURON_LOGICAL_NC_CONFIG="2",PYTHONUNBUFFERED="1")
 return result
def parse_metrics(output,tokens,num_bytes):
 decoder=json.JSONDecoder();found=[]
 for offset,char in enumerate(output):
  if char!="{":continue
  try:value,_=decoder.raw_decode(output[offset:])
  except json.JSONDecodeError:continue
  if isinstance(value,dict) and METRIC_KEYS.intersection(value):found.append(value)
 if len(found)!=1:raise SystemExit(f"expected one metric object, got {len(found)}")
 value=found[0]
 if set(value)!=METRIC_KEYS:raise SystemExit("metric schema mismatch")
 for key in ("val_bpb","total_nats","eval_seconds"):
  if type(value[key]) is not float or not math.isfinite(value[key]) or value[key]<=0:raise SystemExit("invalid "+key)
 if type(value["num_tokens"]) is not int or value["num_tokens"]!=tokens:raise SystemExit("token mismatch")
 if type(value["num_bytes"]) is not int or value["num_bytes"]!=num_bytes:raise SystemExit("byte mismatch")
 if type(value["eval_timeout"]) is not int or value["eval_timeout"]!=1200:raise SystemExit("timeout contract mismatch")
 if value["eval_over_timeout"] is not False or value["eval_seconds"]>1200.0:raise SystemExit("evaluation timeout")
 derived=value["total_nats"]/(value["num_bytes"]*math.log(2.0))
 if not math.isclose(value["val_bpb"],derived,rel_tol=1e-12,abs_tol=1e-12):raise SystemExit("BPB math mismatch")
 return value
def run_eval(name,argv,env,tokens,num_bytes):
 started=time.monotonic()
 try:done=subprocess.run(argv,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1200)
 except subprocess.TimeoutExpired as error:
  durable(OUTPUT/(name+".log"),(error.stdout or "").encode() if isinstance(error.stdout,str) else (error.stdout or b""));raise SystemExit(name+" exceeded 1200 seconds")
 elapsed=time.monotonic()-started;durable(OUTPUT/(name+".log"),done.stdout.encode())
 if done.returncode!=0 or elapsed>1200.0:raise SystemExit(name+" failed or exceeded wall timeout")
 metrics=parse_metrics(done.stdout,tokens,num_bytes)
 receipt={"schema":"c595-eval-recovery-metrics-v1","name":name,"argv":argv,"metrics":metrics,"wall_seconds":elapsed}
 durable_json(OUTPUT/(name+".metrics.json"),receipt)
 persisted_json(name+".metrics.json",receipt,"c595-eval-recovery-metrics-v1")
 return metrics,elapsed
def finite_scan(env):
 source="""import json,sys,torch\nv=torch.load(sys.argv[1],map_location='cpu',weights_only=True); stack=[v]; n=0\nwhile stack:\n x=stack.pop()\n if isinstance(x,dict): stack.extend(x.values())\n elif isinstance(x,(list,tuple)): stack.extend(x)\n elif isinstance(x,torch.Tensor):\n  n+=1\n  if (x.is_floating_point() or x.is_complex()) and not torch.isfinite(x).all().item(): raise SystemExit('nonfinite')\nprint(json.dumps({'weights_only':True,'map_location':'cpu','finite':True,'tensor_count':n}))\n"""
 done=subprocess.run([PYTHON,"-c",source,str(CHECKPOINT)],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
 durable(OUTPUT/"checkpoint-finite.log",done.stdout.encode())
 if done.returncode:raise SystemExit("finite checkpoint scan failed")
 value=json.loads(done.stdout.strip().splitlines()[-1])
 if value!={"weights_only":True,"map_location":"cpu","finite":True,"tensor_count":value.get("tensor_count")} or type(value["tensor_count"]) is not int or value["tensor_count"]<1:raise SystemExit("finite scan receipt invalid")
 return value

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--authorization",type=pathlib.Path,required=True);ap.add_argument("--authorization-sha256",required=True);ap.add_argument("--core-commit",required=True);args=ap.parse_args()
 if socket.gethostname()!=HOSTNAME:raise SystemExit("wrong host")
 if subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=args.core_commit:raise SystemExit("core commit mismatch")
 authorization=ROOT/args.authorization
 if file_sha(authorization)!=args.authorization_sha256:raise SystemExit("authorization hash mismatch")
 expected_auth={"schema":"c595-eval-recovery-authorization-v1","host":"trn22","expected_hostname":HOSTNAME,"label":LABEL,
  "source_commit":SOURCE_COMMIT,"candidate_sha256":TRAIN_SHA,"prepare_sha256":PREPARE_SHA,"checkpoint_path":str(CHECKPOINT),
  "checkpoint_sha256":CHECKPOINT_SHA,"checkpoint_bytes":CHECKPOINT_BYTES,"training_authorized":False,"checkpoint_prune_authorized":False,"state":"human_approved_evaluator_only"}
 if json.loads(authorization.read_text())!=expected_auth:raise SystemExit("authorization contract mismatch")
 if file_sha(ROOT/TRAIN_REL)!=TRAIN_SHA or file_sha(ROOT/"prepare.py")!=PREPARE_SHA:raise SystemExit("working evaluator source mismatch")
 for rel,want in ((TRAIN_REL,TRAIN_SHA),("prepare.py",PREPARE_SHA)):
  blob=subprocess.check_output(["git","show",SOURCE_COMMIT+":"+rel],cwd=ROOT)
  if hashlib.sha256(blob).hexdigest()!=want:raise SystemExit("source commit object mismatch")
 info=safe_regular(CHECKPOINT);before_sha,before_bytes=nofollow_sha(CHECKPOINT)
 if (before_sha,before_bytes)!=(CHECKPOINT_SHA,CHECKPOINT_BYTES) or info.st_size!=CHECKPOINT_BYTES:raise SystemExit("checkpoint identity mismatch")
 runtime_job=ROOT/"research"/"run-logs"/(LABEL+".job.json");safe_regular(runtime_job)
 runtime_job_bytes,_=nofollow_bytes(runtime_job);runtime_job_data=json.loads(runtime_job_bytes)
 if runtime_job_data.get("label")!=LABEL or runtime_job_data.get("expected_commit")!=args.core_commit or runtime_job_data.get("host")!="trn22":raise SystemExit("runtime job identity mismatch")
 request_sha=runtime_job_data.get("request_file_sha256")
 if type(request_sha) is not str or len(request_sha)!=64 or any(c not in HEX64 for c in request_sha):raise SystemExit("runtime request identity missing")
 runtime_job_sha=hashlib.sha256(runtime_job_bytes).hexdigest()
 create_output()
 env=scrub(os.environ);scan=finite_scan(env)
 primary=[PYTHON,"prepare.py","eval-public","--train-py",TRAIN_REL,"--checkpoint",str(CHECKPOINT)]
 secondary=[PYTHON,"prepare.py","eval-public","--train-py",TRAIN_REL,"--checkpoint",str(CHECKPOINT),"--device","neuron","--batch-size","1","--seq-len","1024","--eval-tokens","20971520","--timeout-seconds","1200"]
 primary_metrics,primary_wall=run_eval("primary-public1024",primary,env,2097152,8418678)
 secondary_metrics,secondary_wall=run_eval("secondary-public20m",secondary,env,20971520,83462186)
 after_sha,after_bytes=nofollow_sha(CHECKPOINT)
 if (after_sha,after_bytes)!=(CHECKPOINT_SHA,CHECKPOINT_BYTES):raise SystemExit("checkpoint changed during evaluation")
 result={"schema":"c595-eval-recovery-result-v1","label":LABEL,"host":"trn22","hostname":HOSTNAME,"source_commit":SOURCE_COMMIT,
  "candidate_sha256":TRAIN_SHA,"prepare_sha256":PREPARE_SHA,"checkpoint_path":str(CHECKPOINT),"checkpoint_sha256":after_sha,
  "checkpoint_bytes":after_bytes,"checkpoint_preserved":True,"training_executed":False,"checkpoint_prune_executed":False,
  "finite_scan":scan,"primary":{"argv":primary,"metrics":primary_metrics,"wall_seconds":primary_wall},
  "secondary":{"argv":secondary,"metrics":secondary_metrics,"wall_seconds":secondary_wall},
  "environment":{"python":PYTHON,"lnc":"2","inherited_neuron_cc_flags_removed":True,"distributed_and_competition_variables_scrubbed":True}}
 durable_json(OUTPUT/"result.json",result)
 persisted_json("result.json",result,"c595-eval-recovery-result-v1")
 members=inspect_artifacts(BASE_ARTIFACTS)
 manifest={"schema":"c595-eval-recovery-artifact-manifest-v1","label":LABEL,"members":members}
 durable_json(OUTPUT/"artifact-manifest.json",manifest)
 manifest_bytes=artifact_bytes("artifact-manifest.json");manifest_sha=hashlib.sha256(manifest_bytes).hexdigest()
 if json.loads(manifest_bytes)!=manifest or inspect_artifacts(BASE_ARTIFACTS|{"artifact-manifest.json"})["artifact-manifest.json"]["sha256"]!=manifest_sha:raise SystemExit("manifest revalidation failed")
 # Validate against the exact pre-manifest allowlist by temporarily checking
 # the already captured member map and each no-follow byte identity.
 for name,identity in members.items():
  data=artifact_bytes(name)
  if identity!={"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}:raise SystemExit("manifest member changed")
 completion={"schema":"c595-eval-recovery-completion-v1","label":LABEL,"host":"trn22","hostname":HOSTNAME,
  "core_commit":args.core_commit,"source_commit":SOURCE_COMMIT,"candidate_sha256":TRAIN_SHA,"prepare_sha256":PREPARE_SHA,
  "request_file_sha256":request_sha,"runtime_job_sha256":runtime_job_sha,"artifact_manifest_sha256":manifest_sha,
  "artifact_members":members,"checkpoint_path":str(CHECKPOINT),"checkpoint_pre_sha256":before_sha,"checkpoint_pre_bytes":before_bytes,
  "checkpoint_post_sha256":after_sha,"checkpoint_post_bytes":after_bytes,"checkpoint_preserved":True,
  "training_executed":False,"checkpoint_prune_executed":False,"ready_to_retrieve":True}
 durable_json(OUTPUT/"ready-to-retrieve.json",completion)
 completion_bytes=artifact_bytes("ready-to-retrieve.json");completion_sha=hashlib.sha256(completion_bytes).hexdigest()
 validate_completion(json.loads(completion_bytes),completion)
 final_set=BASE_ARTIFACTS|{"artifact-manifest.json","ready-to-retrieve.json"}
 final_artifacts=inspect_artifacts(final_set)
 if final_artifacts["artifact-manifest.json"]["sha256"]!=manifest_sha or final_artifacts["ready-to-retrieve.json"]["sha256"]!=completion_sha:raise SystemExit("final artifact mutation detected")
 print(json.dumps({"result":result,"artifact_manifest_sha256":manifest_sha,"completion_receipt_sha256":completion_sha,"ready_to_retrieve":True},sort_keys=True))
if __name__=="__main__":main()
