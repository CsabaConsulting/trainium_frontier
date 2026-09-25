"""Independent packet-only CPU exactness probe. No compilation/distributed workers.
Unmodified reconstructed modules are imported with fail-closed prepare/pyarrow stubs.
main() executes only through CLI setup, stopping BEFORE init_runtime.
Tiny models: D=128,L=8,V=256,T<=32,B<=2. Eval table floor is test-only 32.
Reference training MLP adapter installed explicitly with forward_backend='native'.
BF16 compute is explicitly selected to model LONG arithmetic on CPU, not native execution.
All tensor comparisons demand exact equality (atol=rtol=0); no threshold retries.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['NEURON_LOGICAL_NC_CONFIG']='2'
import ast,copy,dataclasses,gc,hashlib,importlib.util,inspect,json,sys,types,traceback
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import weakref
W=Path('/work'); A=W/'test_artifacts'; A.mkdir(exist_ok=True)
torch.set_num_threads(1);torch.set_num_interop_threads(1)
RESULTS=[]
def record(name,**kw):
 row=dict(case=name,**kw);RESULTS.append(row);print(json.dumps(row),flush=True)
 (A/'results.json').write_text(json.dumps(RESULTS,indent=2)+'\n')
def save(name,data):torch.save(data,A/(name+'.pt'))
def blocked(*a,**k):raise RuntimeError('Unexercised external prepare/pyarrow operation')
prepare=types.ModuleType('prepare')
for n in ['Tokenizer','_document_batches','_row_groups_for_split','causality_check','ensure_tokenizer','evaluate_bpb','get_dist_info','get_token_bytes','make_dataloader','record_step_boundary']:setattr(prepare,n,blocked)
# Budget/startup constants are import placeholders, not a tested organizer contract.
for n,v in dict(TOKENIZER_VOCAB_SIZE=8192,TRAIN_TIME_BUDGET_SECONDS=1800,TRAIN_STARTUP_STEPS_EXCLUDED=2,TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=600,EVAL_TIMEOUT_SECONDS=600).items():setattr(prepare,n,v)
prepare.print0=print
sys.modules['prepare']=prepare
pa=types.ModuleType('pyarrow');pa.__version__='0.0.0';pq=types.ModuleType('pyarrow.parquet');pq.ParquetFile=blocked;pa.parquet=pq
sys.modules['pyarrow']=pa;sys.modules['pyarrow.parquet']=pq
COMMON='--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public'.split()
REF=COMMON+'--matrix-lr 0.015 --seed 42 --max-train-seconds 1800 --num-steps 1000000 --out-dir out --unet-8x1024 --attention-layers 0,2,4,6,7 --initial-cache-refresh'.split()
class CLIStop(Exception):pass
def load(name):
 spec=importlib.util.spec_from_file_location('probe_'+name,W/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 original=m.init_runtime
 captured={}
 def stop(device_type,seed=42):
  captured.update(vars(inspect.currentframe().f_back.f_locals['args']));captured.update(runtime_seed=seed,runtime_device_type=device_type)
  raise CLIStop()
 m.init_runtime=stop;sys.argv=['train.py']+(REF if name=='scored_reference' else COMMON)
 try:m.main()
 except CLIStop:pass
 finally:m.init_runtime=original
 config=dataclasses.asdict(m.build_config())
 globals_snapshot={k:v for k,v in vars(m).items() if k.isupper() and isinstance(v,(str,int,float,bool,tuple,list,type(None)))}
 (A/(name+'_effective_config.json')).write_text(json.dumps(dict(argv=sys.argv[1:],args=captured,model_config=config,globals=globals_snapshot),indent=2)+'\n')
 assert config==dict(sequence_len=1024,vocab_size=8192,n_layer=8,n_head=8,n_embd=1024,mlp_ratio=4,attn_scale=0.0,n_kv_heads=0)
 assert captured['seed']==42 and captured['max_train_seconds']==1800 and m.MATRIX_LR==.015
 record(name+'_long_configuration',passed=True,model_config=config,seed=captured['seed'],matrix_lr=m.MATRIX_LR,budget=captured['max_train_seconds'])
 return m

def storage_bytes(obj):
 seen={}
 def walk(x):
  if isinstance(x,torch.Tensor):seen[x.untyped_storage()._cdata]=x.untyped_storage().nbytes()
  elif isinstance(x,dict):
   for v in x.values():walk(v)
  elif isinstance(x,(list,tuple)):
   for v in x:walk(v)
 walk(obj);return sum(seen.values())

class MemoryWatch(TorchDispatchMode):
 """Track live Python tensor handles incl. saved-for-backward tensors, dedup storages.
 Backstop: 20 MiB analytical extra workspace beyond all retained model/optim state.
 """
 def __init__(self):self.refs={};self.peak=0
 def add(self,x):
  if isinstance(x,torch.Tensor):self.refs[id(x)]=weakref.ref(x)
  elif isinstance(x,dict):
   for y in x.values():self.add(y)
  elif isinstance(x,(list,tuple)):
   for y in x:self.add(y)
 def sample(self):
  live={};dead=[]
  for i,r in self.refs.items():
   x=r()
   if x is None:dead.append(i)
   else:
    s=x.untyped_storage();live[s._cdata]=s.nbytes()
  for i in dead:del self.refs[i]
  self.peak=max(self.peak,sum(live.values()));assert self.peak<=64*1024**2,self.peak
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  out=func(*args,**(kwargs or {}));self.add(args);self.add(out);self.sample();return out
 def pack(self,x):self.add(x);self.sample();return x

def compare(name,a,b,expected_equal=True):
 if a is None or b is None:
  equal=a is b; ma=mr=None;shape=None
 else:
  shape=list(a.shape);equal=a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
  if a.shape==b.shape:
   af=a.double();bf=b.double();d=(af-bf).abs();ma=d.max().item() if d.numel() else 0.;mr=(d/bf.abs().clamp_min(1e-30)).max().item() if d.numel() else 0.
  else:ma=mr=None
 record(name,passed=equal if expected_equal else not equal,expected_equal=expected_equal,equal=equal,max_abs=ma,max_rel=mr,shape=shape,atol=0,rtol=0)

def refresh(m,model,detach=False):
 for lin in model.modules():
  if isinstance(lin,m.Linear):
   lin.refresh_weight_cache(m.COMPUTE_DTYPE)
   if detach:lin._cached_weight=lin._cached_weight.detach()

def run_training(name,m):
 m.COMPUTE_DTYPE=torch.bfloat16;torch.manual_seed(42)
 model=m.GPT(m.GPTConfig(sequence_len=32,vocab_size=256,n_layer=8,n_head=1,n_embd=128));model.init_weights()
 if name=='scored_reference':m._submission_mlp_install(model,'cpu',m,forward_backend='native')
 assert sum(p.numel() for p in model.parameters())==1475892
 assert [i for i,b in enumerate(model.transformer.h) if b.attn is not None]==[0,2,4,6,7]
 assert list(model.value_embeds)==['7'] and len(model.unet_skip_gates)==4
 save(name+'_initial',model.state_dict())
 opt=model.setup_optimizer()
 pn={id(p):n for n,p in model.named_parameters()}
 groups=[{**{k:v for k,v in g.items() if k!='params'},'params':[pn[id(p)] for p in g['params']]} for g in opt.param_groups]
 entries,_,summary=m.make_plan(opt.param_groups,4)
 (A/(name+'_optimizer_plan.json')).write_text(json.dumps(dict(groups=groups,entries=entries,summary=summary),indent=2)+'\n')
 for step in range(3):
  opt.zero_grad(set_to_none=True);refresh(m,model)
  tokens=(torch.arange(64).reshape(2,32)*13+step*19)%256;targets=(tokens*3+7)%256
  save(name+f'_step{step}_inputs',dict(tokens=tokens,targets=targets))
  loss=model(tokens,targets);loss.backward()
  save(name+f'_step{step}_before_clip',dict(loss=loss.detach(),grads={n:p.grad for n,p in model.named_parameters()}))
  assert all(p.grad is not None for p in model.parameters())
  if name=='candidate':m.clip_and_sync_gradients(model,list(model.parameters()),max_norm=1.)
  else:m.clip_and_sync_gradients(model,list(model.parameters()),[],max_norm=1.,clip_after_reduce=True)
  elapsed=[0.,900.,1764.][step]
  for g in opt.param_groups:
   g['lr']=g['initial_lr']*m.lr_multiplier(step,elapsed,1800)
   if g['kind']=='muon':g.update(momentum=m.muon_momentum(step,elapsed/1800),weight_decay=.2*(1-elapsed/1800),beta2=.9)
  opt.step()
  save(name+f'_step{step}_after_update',model.state_dict())
  retained=storage_bytes([list(model.parameters()),[p.grad for p in model.parameters()],opt.state,opt._scalar_cache,[x._cached_weight for x in model.modules() if isinstance(x,m.Linear)],list(model.buffers())])
  record(name+f'_step{step}_memory',passed=retained+20*1024**2<64*1024**2,retained_tensor_bytes=retained,conservative_workspace_bytes=20*1024**2)
 m.save_checkpoint(A/(name+'_trained.pt'),model,{'independent_test':True,'step':3})
 del opt,model;gc.collect()

def eval_tests(name,m):
 # Keep construction context within lane bound; this overrides only nonpersistent table floor.
 m.EVAL_MIN_SEQ_LEN=32
 original=m.compute_dtype_for
 for dtype in [torch.float32,torch.bfloat16]:
  tag=str(dtype).split('.')[-1]
  m.compute_dtype_for=(lambda device: dtype)
  model=m.load_for_eval(str(A/(name+'_trained.pt')),torch.device('cpu'))
  assert not model.training and all(not p.requires_grad for p in model.parameters())
  for length in [1,7,16,31,32]:
   x=(torch.arange(2*length).reshape(2,length)*17+19)%256
   visits=[]
   handle=model.transformer.h[0].register_forward_pre_hook(lambda module,args:visits.append(list(args[0].shape)))
   with torch.no_grad():batch=model(x)
   handle.remove()
   with torch.no_grad():
    rows=torch.cat([model(row[None]) for row in x],dim=0)
    altered=x.clone();cut=max(1,length//2);altered[:,cut:]=(altered[:,cut:]+87)%256
    suffix=model(altered)
   base=f'{name}_eval_{tag}_t{length}'
   save(base,dict(tokens=x,altered_tokens=altered,batch=batch,independent_rows=rows,changed_suffix_output=suffix))
   record(base+'_route',passed=True,model_class=type(model).__name__,block0_input_shapes=visits)
   compare(base+'_batch_rows',batch,rows)
   compare(base+'_causal_prefix',batch[:,:cut],suffix[:,:cut])
  del model;gc.collect()
 m.compute_dtype_for=original

def negative_cache(m):
 torch.manual_seed(912);lin=m.Linear(128,128,bias=False);x=torch.randn(2,7,128,dtype=torch.bfloat16,requires_grad=True)
 lin.refresh_weight_cache(torch.bfloat16);good=lin(x);good.float().square().mean().backward()
 grad=lin.weight.grad.detach().clone();out=good.detach();lin.weight.grad=None;x.grad=None
 lin._cached_weight=lin._cached_weight.detach();bad=lin(x);bad.float().square().mean().backward()
 save('negative_detached_cache',dict(input=x.detach(),weight=lin.weight.detach(),good_output=out,bad_output=bad.detach(),good_gradient=grad,bad_gradient=lin.weight.grad))
 compare('negative_detached_cache_outputs',out,bad.detach())
 compare('negative_detached_cache_gradient',grad,lin.weight.grad,expected_equal=False)
 record('negative_detached_cache_detected',passed=lin.weight.grad is None and grad.count_nonzero().item()>0,good_nonzero_gradient_elements=grad.count_nonzero().item(),bad_gradient_missing=lin.weight.grad is None)

def negative_pairing(m):
 torch.manual_seed(42);m.COMPUTE_DTYPE=torch.bfloat16
 model=m.GPT(m.GPTConfig(sequence_len=32,vocab_size=256,n_layer=8,n_head=1,n_embd=128));model.init_weights();model.load_state_dict(torch.load(A/'candidate_trained.pt',weights_only=True)['model_state']);model.eval()
 # Unequal explicit nonzero gates prevent zero-initialized skips from hiding wrong pairing.
 with torch.no_grad():
  for i,g in enumerate(model.unet_skip_gates):g.fill_([.3,-.2,.5,-.4][i])
 x=(torch.arange(32).reshape(1,32)*11)%256
 original=m.GPT.forward;node=ast.parse(inspect.getsource(m.GPT)).body[0]
 method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
 for n in ast.walk(method):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='pop':n.args=[ast.Constant(0)]
 method.name='wrong_pairing';ast.fix_missing_locations(method);ns={};exec(compile(ast.Module(body=[method],type_ignores=[]),'<negative_fifo_skip>','exec'),m.__dict__,ns)
 with torch.no_grad():good=model(x);bad=ns['wrong_pairing'](model,x)
 save('negative_fifo_pairing',dict(tokens=x,model_state=model.state_dict(),good_output=good,bad_output=bad))
 compare('negative_fifo_pairing',good,bad,expected_equal=False)
 del model;gc.collect()

def isolated_mlp(name,m):
 m.COMPUTE_DTYPE=torch.bfloat16;torch.manual_seed(233)
 mlp=m.MLP(m.GPTConfig(sequence_len=32,vocab_size=256,n_layer=8,n_head=1,n_embd=128))
 if name=='scored_reference':mlp.forward=types.MethodType(m.make_forward('cpu',type(mlp).forward,True,forward_backend='native'),mlp)
 x=torch.randn(1,256,128,dtype=torch.bfloat16,requires_grad=True);refresh(m,mlp)
 y=mlp(x);y.float().square().mean().backward()
 save(name+'_isolated_mlp',dict(input=x.detach(),output=y.detach(),dx=x.grad,up=mlp.c_fc.weight.detach(),down=mlp.c_proj.weight.detach(),dup=mlp.c_fc.weight.grad,ddown=mlp.c_proj.weight.grad))

def compare_saved(stem):
 a=torch.load(A/('candidate_'+stem+'.pt'),weights_only=True);b=torch.load(A/('scored_reference_'+stem+'.pt'),weights_only=True)
 def walk(x,y,path):
  if isinstance(x,dict):
   assert x.keys()==y.keys(),path
   for k in x:walk(x[k],y[k],path+'/'+k)
  else:compare(stem+path,x,y)
 walk(a,b,'')

def main():
 mods={n:load(n) for n in ['scored_reference','candidate']}
 # Pure integer arithmetic; no full-sized model/tensor allocation.
 parts=dict(token_embedding=8192*1024,lm_head=8192*1024,value_embedding=8192*1024,attention=5*4*1024**2,mlp=8*8*1024**2,local_conv=5*1024*2,ve_gate=8*32,residual_scalars=16,unet_scalars=4)
 record('full_parameter_count_static',passed=sum(parts.values())==113256724,components=parts,total=sum(parts.values()))
 for name,m in mods.items():run_training(name,m);eval_tests(name,m);isolated_mlp(name,m)
 for stem in ['initial']+[f'step{s}_{k}' for s in range(3) for k in ['before_clip','after_update']]+['isolated_mlp']+[f'eval_{d}_t{t}' for d in ['float32','bfloat16'] for t in [1,7,16,31,32]]:compare_saved(stem)
 plans=[json.loads((A/(n+'_optimizer_plan.json')).read_text()) for n in ['candidate','scored_reference']]
 record('optimizer_group_and_ownership_plan',passed=plans[0]==plans[1],equal=plans[0]==plans[1],actual_world_size=1,planned_world_size=4,collectives_executed=False)
 for s,e in [(0,0),(300,450),(600,900),(2000,1719),(2100,1800)]:
  a=mods['candidate'];b=mods['scored_reference']
  record(f'schedule_{s}_{e}',passed=a.lr_multiplier(s,e,1800)==b.lr_multiplier(s,e,1800) and a.muon_momentum(s,e/1800)==b.muon_momentum(s,e/1800),candidate=[a.lr_multiplier(s,e,1800),a.muon_momentum(s,e/1800)],reference=[b.lr_multiplier(s,e,1800),b.muon_momentum(s,e/1800)],max_abs=0,max_rel=0)
 negative_cache(mods['candidate']);negative_pairing(mods['candidate'])

if __name__=='__main__':
 watcher=MemoryWatch()
 try:
  with watcher,torch.autograd.graph.saved_tensors_hooks(watcher.pack,lambda x:x):main()
  record('tracked_live_tensor_storage',passed=watcher.peak<=64*1024**2,peak_bytes=watcher.peak,limit_bytes=64*1024**2)
  record('environment',passed=True,torch=torch.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),native_executed=False,full_size_cpu_executed=False)
 except BaseException:
  traceback.print_exc();record('harness_exception',passed=False,traceback=traceback.format_exc());raise
 failures=[r['case'] for r in RESULTS if not r['passed']]
 print(json.dumps(dict(summary='FAIL' if failures else 'PASS',cases=len(RESULTS),failures=failures)),flush=True)
 sys.exit(bool(failures))
