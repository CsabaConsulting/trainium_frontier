"""Bounded CPU author check; no native or distributed claims."""
import os
os.environ['OMP_NUM_THREADS']='1'
import ast, hashlib, importlib.util, json, sys, types, subprocess
from pathlib import Path
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
root=Path(__file__).resolve().parent
stub=Path('/home/poojith-devan/Desktop/Trainium/workflow/tools/stub_prepare')
sys.path.insert(0,str(stub))
pa=types.ModuleType('pyarrow');pa.__version__='0.0.0';pq=types.ModuleType('pyarrow.parquet');pa.parquet=pq
sys.modules['pyarrow']=pa;sys.modules['pyarrow.parquet']=pq
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
mods=[load('base',root/'base.py'),load('recursive',root/'train.py'),load('batch',root.parent/'20260926-sprint-p0-clean-batch/train.py')]
def run(m):
 m.COMPUTE_DTYPE=torch.bfloat16;torch.manual_seed(42)
 model=m.GPT(m.GPTConfig(sequence_len=32,vocab_size=256,n_layer=8,n_head=1,n_embd=128));model.init_weights();opt=model.setup_optimizer();out=[]
 for step in range(3):
  opt.zero_grad(set_to_none=True)
  for layer in model.modules():
   if isinstance(layer,m.Linear):layer.refresh_weight_cache(m.COMPUTE_DTYPE)
  x=(torch.arange(64).reshape(2,32)*13+step*19)%256;y=(x*3+7)%256
  loss=model(x,y);loss.backward();m.clip_and_sync_gradients(model,list(model.parameters()),max_norm=1.)
  elapsed=[0.,900.,1764.][step]
  for g in opt.param_groups:
   g['lr']=g['initial_lr']*m.lr_multiplier(step,elapsed,1800)
   if g['kind']=='muon':g.update(momentum=m.muon_momentum(step,elapsed/1800),weight_decay=.2*(1-elapsed/1800),beta2=.9)
  opt.step();h=hashlib.sha256()
  for n,p in model.state_dict().items():h.update(n.encode());h.update(p.contiguous().view(torch.uint8).numpy().tobytes())
  out.append(dict(loss=float(loss),state_sha256=h.hexdigest()))
 return out
results=[run(m) for m in mods]
assert results[0]==results[1]==results[2],results
m=mods[1];m.RECURSIVE_SCHEDULE=True
donor_text=subprocess.check_output(['git','show','44926a6f:workflow/proposals/research-v2/train.py'],text=True)
donor_ast=ast.parse(donor_text)
names={'lr_multiplier','muon_momentum','muon_beta2'}
donor=dict(vars(m))
exec(compile(ast.Module(body=[n for n in donor_ast.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'<exact4bfaec-schedule>','exec'),donor)
schedule=[]
for step in [0,1,299,300,2000]:
 for elapsed in [-1.,0.,90.,630.,900.,1799.,1800.,1801.]:
  assert m.muon_momentum(step,elapsed/1800)==donor['muon_momentum'](step,elapsed/1800)
  assert m.muon_beta2(elapsed/1800)==donor['muon_beta2'](elapsed/1800)
  for kind,ratio in [('muon',.95),('adamw',.65)]:
   expected=donor['lr_multiplier'](step,elapsed,1800,kind)
   assert m.lr_multiplier(step,elapsed,1800,kind)==expected,(step,elapsed,kind,m.lr_multiplier(step,elapsed,1800,kind),expected)
  schedule.append([step,elapsed])
print(json.dumps(dict(passed=True,steps=3,atol=0,rtol=0,base_recursive_off_batch_default=results,schedule_cases=len(schedule),native=False,distributed=False),indent=2))
