#!/usr/bin/env python3
"""Fail-closed full-budget single-factor MATRIX_LR candidate driver."""
import argparse,hashlib,json,math,os,pathlib,re,socket,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
LABEL="fs-20260924-c595-matrixlr025-trn21-r1";HOSTNAME="ip-172-31-34-85"
PACKAGE_REL="submissions/dense6x1024_direct_review_matrixlr025"
BASE_SHA="c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537"
TRAIN_SHA="4b36bb0da7950299c09f8bbbfc8d4a77c6bb73415a85f3bcf413cc2c4bab2994"
LAUNCH_SHA="b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd"
PREPARE_SHA="5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
METRIC_KEYS={"val_bpb","num_tokens","num_bytes","total_nats","eval_seconds","eval_timeout","eval_over_timeout"}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def scrub(source):
 blocked={"RANK","LOCAL_RANK","WORLD_SIZE","LOCAL_WORLD_SIZE","GROUP_RANK","ROLE_RANK","ROLE_WORLD_SIZE","MASTER_ADDR","MASTER_PORT","PYTHONPATH","PYTHONHOME","NEURON_RT_VISIBLE_CORES","NEURON_RT_NUM_CORES"}
 return {k:v for k,v in source.items() if k not in blocked and not k.startswith("TORCHELASTIC_") and not k.startswith("NEURON_COMPETITION_R1_")}
def metrics(text):
 decoder=json.JSONDecoder();found=[]
 for offset,char in enumerate(text):
  if char!="{":continue
  try:value,_=decoder.raw_decode(text[offset:])
  except json.JSONDecodeError:continue
  if isinstance(value,dict) and METRIC_KEYS.intersection(value):found.append(value)
 if len(found)!=1 or set(found[0])!=METRIC_KEYS:raise SystemExit("evaluation metric schema/count mismatch")
 value=found[0]
 for key in ("val_bpb","total_nats","eval_seconds"):
  if type(value[key]) is not float or value[key]<=0 or not math.isfinite(value[key]):raise SystemExit("invalid "+key)
 if type(value["num_tokens"]) is not int or value["num_tokens"]!=20971520:raise SystemExit("token count mismatch")
 if type(value["num_bytes"]) is not int or value["num_bytes"]!=83462186:raise SystemExit("byte count mismatch")
 if type(value["eval_timeout"]) is not int or value["eval_timeout"]!=1200 or value["eval_over_timeout"] is not False or value["eval_seconds"]>1200.0:raise SystemExit("eval timeout mismatch")
 if not math.isclose(value["val_bpb"],value["total_nats"]/(value["num_bytes"]*math.log(2.0)),rel_tol=1e-12,abs_tol=1e-12):raise SystemExit("BPB math mismatch")
 return value
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--authorization-sha256",required=True);ap.add_argument("--sandbox-root",type=pathlib.Path,required=True);args=ap.parse_args()
 auth_path=HERE/"authorization.json";auth=json.loads(auth_path.read_text())
 if sha(auth_path)!=args.authorization_sha256:raise SystemExit("authorization hash mismatch")
 exact={"arm":"candidate","host":"trn21","expected_hostname":HOSTNAME,"label":LABEL,"package":PACKAGE_REL,"base_train_sha256":BASE_SHA,"train_sha256":TRAIN_SHA,"single_causal_change":"MATRIX_LR 0.020 -> 0.025","seed":42,"max_train_seconds":1800,"smoke_authorized":False,"checkpoint_prune_authorized":False,"state":"human_approved_pending_operator"}
 if auth!={"schema":"c595-matrixlr025-authorization-v1",**exact}:raise SystemExit("authorization mismatch")
 if socket.gethostname()!=HOSTNAME:raise SystemExit("wrong host")
 sandbox=ROOT/"out"/LABEL
 if str(args.sandbox_root)!=str(sandbox) or sandbox.exists():raise SystemExit("unsafe or existing sandbox")
 package=ROOT/PACKAGE_REL;base=ROOT/"submissions/dense6x1024_direct_review/train.py"
 if sha(package/"train.py")!=TRAIN_SHA or sha(package/"launch-command.txt")!=LAUNCH_SHA or sha(ROOT/"prepare.py")!=PREPARE_SHA or sha(base)!=BASE_SHA:raise SystemExit("protected hash mismatch")
 base_lines=base.read_text().splitlines();candidate_lines=(package/"train.py").read_text().splitlines()
 changes=[(a,b) for a,b in zip(base_lines,candidate_lines) if a!=b]
 if len(base_lines)!=len(candidate_lines) or changes!=[("MATRIX_LR = 0.02","MATRIX_LR = 0.025")]:raise SystemExit("candidate is not exact one-line causal change")
 source=(package/"train.py").read_text()
 if "MAX_TRAIN_SECONDS = 1800" not in source or "default=int(os.environ.get('NEURON_COMPETITION_R1_SEED', '42'))" not in source:raise SystemExit("seed/time contract mismatch")
 env=scrub(os.environ);env.update(PYTHONUNBUFFERED="1",NEURON_LOGICAL_NC_CONFIG="2")
 rehearsal=ROOT/"research/scratch/submission_rehearsal.py";train_box=sandbox/"train";eval_box=sandbox/"eval"
 trained=subprocess.run([sys.executable,str(rehearsal),"--package",str(package),"--sandbox",str(train_box),"--mode","train","--fresh-caches"],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 sandbox.mkdir(parents=True,exist_ok=True);(sandbox/"training.log").write_text(trained.stdout)
 if trained.returncode:raise SystemExit("full training failed")
 checkpoint=train_box/"out/final.pt"
 if not checkpoint.is_file():raise SystemExit("checkpoint missing")
 checkpoint_sha,checkpoint_bytes=sha(checkpoint),checkpoint.stat().st_size
 finite="""import json,sys,torch\nv=torch.load(sys.argv[1],map_location='cpu',weights_only=True);s=[v];n=0\nwhile s:\n x=s.pop()\n if isinstance(x,dict):s.extend(x.values())\n elif isinstance(x,(list,tuple)):s.extend(x)\n elif isinstance(x,torch.Tensor):\n  n+=1\n  if (x.is_floating_point() or x.is_complex()) and not torch.isfinite(x).all().item():raise SystemExit('nonfinite')\nprint(json.dumps({'finite':True,'weights_only':True,'tensor_count':n}))\n"""
 checked=subprocess.run([sys.executable,"-c",finite,str(checkpoint)],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(sandbox/"checkpoint-finite.log").write_text(checked.stdout)
 if checked.returncode:raise SystemExit("checkpoint finite scan failed")
 evaluated=subprocess.run([sys.executable,str(rehearsal),"--package",str(package),"--sandbox",str(eval_box),"--mode","eval","--checkpoint",str(checkpoint),"--eval-tokens","20971520","--fresh-caches"],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (sandbox/"fresh-public20m.log").write_text(evaluated.stdout)
 if evaluated.returncode:raise SystemExit("fresh full20m evaluation failed")
 value=metrics(evaluated.stdout)
 command_line=next((line for line in evaluated.stdout.splitlines() if line.startswith("REHEARSAL_COMMAND ")),None)
 command=json.loads(command_line.removeprefix("REHEARSAL_COMMAND ")) if command_line else []
 tail=["--device","neuron","--batch-size","1","--seq-len","1024","--eval-tokens","20971520","--timeout-seconds","1200"]
 if not any(command[i:i+len(tail)]==tail for i in range(len(command))):raise SystemExit("fresh eval command mismatch")
 if sha(checkpoint)!=checkpoint_sha or checkpoint.stat().st_size!=checkpoint_bytes:raise SystemExit("checkpoint changed")
 improvement=0.993231940827686-value["val_bpb"]
 decision="KEEP" if improvement>=0.0015 else "DISCARD"
 result={"schema":"c595-matrixlr025-result-v1","label":LABEL,"host":"trn21","candidate_sha256":TRAIN_SHA,"base_sha256":BASE_SHA,"seed":42,"max_train_seconds":1800,"smoke":False,"checkpoint_path":str(checkpoint),"checkpoint_sha256":checkpoint_sha,"checkpoint_bytes":checkpoint_bytes,"checkpoint_preserved":True,"metrics":value,"control_full20m_bpb":0.993231940827686,"improvement":improvement,"decision_gate":decision,"replication_required":decision=="KEEP" and improvement<0.0025,"ready_to_retrieve":True,"prune_authorized":False,"cache_cleanup_authorized":False}
 (sandbox/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
