#!/usr/bin/env python3
"""Evaluator-only recovery for the retained c595 MATRIX_LR=.025 checkpoint."""
import importlib.util,json,pathlib,re,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3]
LIB=ROOT/"research/submission-validation/20260923-c595-dual-host-seed42-controls/eval_recovery.py"
spec=importlib.util.spec_from_file_location("proven_eval_recovery",LIB);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ROOT=ROOT;m.SOURCE_COMMIT="3e74e30fd1b0e856ddea321f08735bbf4df37a18"
m.TRAIN_REL="submissions/dense6x1024_direct_review_matrixlr025/train.py"
m.TRAIN_SHA="4b36bb0da7950299c09f8bbbfc8d4a77c6bb73415a85f3bcf413cc2c4bab2994"
m.CHECKPOINT=pathlib.Path("/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr025_trn21_r1/out/fs-20260924-c595-matrixlr025-trn21-r1-work/train/out/final.pt")
m.CHECKPOINT_SHA="b2c4818e95e4ae25643ab48eb9dd199edebfc4b065290d6c601244b758af69ba";m.CHECKPOINT_BYTES=469831009
m.HOSTNAME="ip-172-31-34-85";m.LABEL="fs-20260924-c595-matrixlr025-eval-recovery-trn21-r1";m.OUTPUT=ROOT/"out"/m.LABEL
_create_output=m.create_output
m.create_output=lambda:_create_output(ROOT,m.OUTPUT)
WARNING_TAIL=[
"  Overriding a previously registered kernel for the same operator and the same dispatch key",
"  operator: aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)",
"    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6",
"  dispatch key: PrivateUse1",
"  previous kernel: registered at /pytorch/build/aten/src/ATen/RegisterCPU_3.cpp:7637",
"       new kernel: registered at NeuronDispatcher.cpp:0 (function operator())"]
FIRST=re.compile(r"^\[W\d+ \d{2}:\d{2}:\d{2}\.\d+ OperatorEntry\.cpp:208\] Warning: Warning only once for all operators,  other operators may also be overridden\.$")
def parse_finite_output(output):
 lines=output.splitlines()
 if len(lines)==1:warning=[];receipt_line=lines[0]
 elif len(lines)==8 and FIRST.fullmatch(lines[0]) and lines[1:7]==WARNING_TAIL:warning=lines[:7];receipt_line=lines[7]
 else:raise SystemExit("unknown finite-scan stdout/warning")
 try:value=json.loads(receipt_line)
 except json.JSONDecodeError:raise SystemExit("finite receipt is not JSON")
 expected={"schema":"c595-strict-finite-v1","finite":True,"weights_only":True,"map_location":"cpu","tensor_count":62,"checkpoint_sha256":m.CHECKPOINT_SHA,"checkpoint_bytes":m.CHECKPOINT_BYTES}
 if value!=expected or type(value.get("tensor_count")) is not int or set(value)!=set(expected):raise SystemExit("finite receipt mismatch")
 if receipt_line!=json.dumps(expected,sort_keys=True):raise SystemExit("finite receipt is not canonical")
 return value,len(warning)
def finite_scan(env):
 source="""import hashlib,json,os,sys,torch\np=sys.argv[1];fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);h=hashlib.sha256();z=0\nwhile True:\n b=os.read(fd,8388608)\n if not b:break\n h.update(b);z+=len(b)\nos.close(fd);v=torch.load(p,map_location='cpu',weights_only=True);s=[v];n=0\nwhile s:\n x=s.pop()\n if isinstance(x,dict):s.extend(x.values())\n elif isinstance(x,(list,tuple)):s.extend(x)\n elif isinstance(x,torch.Tensor):\n  n+=1\n  if (x.is_floating_point() or x.is_complex()) and not torch.isfinite(x).all().item():raise SystemExit('nonfinite')\nprint(json.dumps({'schema':'c595-strict-finite-v1','finite':True,'weights_only':True,'map_location':'cpu','tensor_count':n,'checkpoint_sha256':h.hexdigest(),'checkpoint_bytes':z},sort_keys=True))"""
 done=subprocess.run([m.PYTHON,"-c",source,str(m.CHECKPOINT)],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
 m.durable(m.OUTPUT/"checkpoint-finite.log",done.stdout.encode())
 if done.returncode:raise SystemExit("finite checkpoint scan failed")
 value,warning_count=parse_finite_output(done.stdout)
 return {**value,"accepted_known_warning_lines":warning_count}
m.finite_scan=finite_scan
if __name__=="__main__":m.main()
