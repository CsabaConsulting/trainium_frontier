import ast, contextlib, copy, io, json, runpy
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
with contextlib.redirect_stdout(io.StringIO()):
 r=runpy.run_path('/tmp/trainium-clock36-recursive-fresh-review-oFZnOI/review-output/independent_reference.py')
env=dict(torch=torch,POLAR_EXPRESS_COEFFS=r['c']['POLAR_EXPRESS_COEFFS'])
for name in ('_muon_lr_g','_muon_math'):
 exec(compile(ast.Module(body=[r['cd'][name]],type_ignores=[]),'unchanged-muon-core','exec'),env)
torch.manual_seed(42)
initial=torch.randn(1,3,4)
x=torch.randn(4,2); target=torch.randn(1,3,2)
def run(clock,recursive,reference=False,base=False):
 w=initial.clone(); mom=torch.zeros_like(w); second=torch.zeros(1,1,4)
 records=[]
 e=r['b'] if base else r['c']
 e.update(RECURSIVE_SCHEDULE=recursive,WARMUP_STEPS=0,_PROCESS_STARTED=0.)
 budget=17.
 for step,now in enumerate((790.,1000.,1001.,1002.,1003.,1004.)):
  elapsed=now-budget
  selected=(max(0.,now-870.) if step<5 else elapsed) if clock else elapsed
  if not reference:
   selected=e['startup_adjusted_schedule_elapsed'](step,elapsed,budget) if clock else elapsed
  p=min(1.,selected/1800.)
  lr=r['oracle'](step,selected,0,'muon') if reference else e['lr_multiplier'](step,selected,1800)
  m=r['om'](step,p) if reference else e['muon_momentum'](step,p)
  beta=r['ob'](p) if reference else e['muon_beta2'](p)
  w.requires_grad_(True)
  loss=((w@x-target)**2).mean(); grad=torch.autograd.grad(loss,w)[0]
  with torch.no_grad():
   # These are actual unchanged production Muon math operations, supplied
   # with independent schedule values in reference arms. No model adapter.
   w,mom,second=env['_muon_math'](w,grad,mom,second,torch.tensor(m),torch.tensor(1-m),torch.tensor(.015*lr),torch.tensor(.015*lr*.1*(1-p)),False,5,-2,3,torch.tensor(1-beta) if recursive else 1-beta,True,False)
  assert all(torch.isfinite(t).all() for t in (w,mom,second,grad,loss))
  records.append(tuple(t.detach().clone() for t in (loss,grad,w,mom,second)))
  if step==4: budget=now-max(0.,now+30.-900.)
 return records
for clock in (False,True):
 for left,right in ((run(clock,False,base=True),run(clock,False)),(run(clock,True),run(clock,True,reference=True))):
  assert len(left)==6
  for a,b in zip(left,right):
   assert all(torch.equal(x,y) for x,y in zip(a,b))
print(json.dumps({'status':'PASS_TINY_MUON_CORE_SCHEDULE_INTEGRATION','seed':42,'threads':torch.get_num_threads(),'torch':torch.__version__,'shape':[1,3,4],'updates':6,'startup_steps_excluded':5,'comparisons':'both clock modes; child OFF versus base; child ON versus independent schedule formulas; exact loss, gradients, weights, momentum and second moment','scope':'AST-extracted unchanged production Muon math with tiny differentiable matrix loss; does not execute full main, AdamW, model, checkpoint or causal evaluation'},indent=2))
