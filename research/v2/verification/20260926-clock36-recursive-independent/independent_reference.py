import ast, hashlib, json, math, os, sys, textwrap
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/tmp/trainium-clock36-recursive-fresh-review-oFZnOI')
def sha(b): return hashlib.sha256(b).hexdigest()
diff = (ROOT/'review-input/source-full-context.diff').read_text().splitlines(keepends=True)[5:]
base = ''.join(s[1:] for s in diff if s[:1] in (' ', '-'))
child = ''.join(s[1:] for s in diff if s[:1] in (' ', '+'))
assert sha(base.encode()) == '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'
assert sha(child.encode()) == '4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea'
trees = [ast.parse(s) for s in (base,child)]
def defs(t): return {n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
bd,cd=map(defs,trees)
changed=[k for k in bd if ast.dump(bd[k],include_attributes=False)!=ast.dump(cd[k],include_attributes=False)]
assert set(changed)=={'lr_multiplier','muon_momentum','muon_beta2','MuonAdamW','main'}
assert ast.get_source_segment(base,bd['startup_adjusted_schedule_elapsed'])==ast.get_source_segment(child,cd['startup_adjusted_schedule_elapsed'])
proposal=json.loads((ROOT/'review-input/EXPERIMENT_PROPOSAL.json').read_text())
for name,h in proposal['source_inputs']['donor_function_ast_sha256'].items():
 assert sha(ast.dump(cd[name],include_attributes=False).encode())==h

def environment(tree):
 e={}
 for n in tree.body:
  if isinstance(n,ast.Assign):
   try: value=ast.literal_eval(n.value)
   except (ValueError,TypeError): continue
   for t in n.targets:
    if isinstance(t,ast.Name): e[t.id]=value
 e.update(TRAIN_STARTUP_STEPS_EXCLUDED=5,TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900,STARTUP_LAUNCH_MARGIN_SECONDS=30,_PROCESS_STARTED=0)
 for name in ['lr_multiplier','muon_momentum','muon_beta2','startup_adjusted_schedule_elapsed','demon_beta1_scale']:
  exec(compile(ast.Module(body=[defs(tree)[name]],type_ignores=[]),'extracted','exec'),e)
 return e
b,c=map(environment,trees)
assert c['RECURSIVE_SCHEDULE'] is False
def oracle(step,elapsed,warmup,kind):
 if step<warmup: return (step+1)/max(1,warmup)
 p=min(1.,max(0.,elapsed/1800.))
 r={'muon':.95}.get(kind,.65)
 if p<1.-r: return 1.
 q=(1.-p)/r
 return q*1.+(1-q)*.05
def om(step,p):
 p=min(1.,max(0.,p)); f=min(step/300,1.)
 if p>1.-.95: return .95+((p-(1.-.95))/.95)**2*(.79-.95)
 return (1-f)*.85+f*.95
def ob(p):
 p=min(1.,max(0.,p))
 return .95+max(0.,(p-(1.-.95))/.95)*(.97-.95)
count=0
for rec in (False,True):
 c['RECURSIVE_SCHEDULE']=rec
 for warmup in (0,1,5,300):
  b['WARMUP_STEPS']=c['WARMUP_STEPS']=warmup
  for step in (0,1,4,5,6,299,300,301):
   for p in (-.2,0.,math.nextafter(.05,0),.05,math.nextafter(.05,1),1.-.95,math.nextafter(1.-.95,1),math.nextafter(.35,0),.35,math.nextafter(.35,1),1.,1.2):
    for kind in ('muon','adamw','other'):
     actual=c['lr_multiplier'](step,p*1800,1800,kind)
     expected=oracle(step,p*1800,warmup,kind) if rec else b['lr_multiplier'](step,p*1800,1800)
     assert actual==expected,(step,p,kind,actual,expected)
     count+=1
    assert c['muon_momentum'](step,p)==(om(step,p) if rec else b['muon_momentum'](step,p))
    assert c['muon_beta2'](p)==(ob(p) if rec else b['muon_beta2'](p))

# Execute the actual production routing slice, including flat LR and WD, without training.
start=child.index('        schedule_elapsed = startup_adjusted_schedule_elapsed(')
end=child.index('        if not skip_step:',start)
routing=compile(textwrap.dedent(child[start:end]),'production-routing','exec')
groups=[dict(kind=k,initial_lr=.015,no_lr_decay=flat) for k in ('muon','adamw','other') for flat in (False,True)]
import copy
clockcases=0
for rec in (False,True):
 for clock in (False,True):
  for offset in (0.,1234567.125):
   for age in (779.,790.,869.999,870.,870.001,900.,1000.):
    for step in (0,1,4,5,6,299,300,301):
     c.update(RECURSIVE_SCHEDULE=rec,_PROCESS_STARTED=offset,WARMUP_STEPS=0,DEMON_BETA1=False,WD_DECAY_TO_ZERO=True,WEIGHT_DECAY=.1)
     # After the real fifth-boundary formula, use rounded observable age.
     now=offset+age
     charged=max(0.,(now-offset)+30.-900.)
     budget=offset+17. if step<5 else now-charged
     elapsed=now-budget
     selected=max(0.,elapsed+(budget-offset)+30.-900.) if clock and step<5 else elapsed
     e=dict(c,step=step,elapsed=elapsed,budget_started=budget,args=SimpleNamespace(startup_adjusted_schedule=clock,max_train_seconds=1800),optimizer=SimpleNamespace(param_groups=copy.deepcopy(groups)),batch_lr_scale=1.)
     exec(routing,e)
     assert e['schedule_elapsed']==selected
     for g in e['optimizer'].param_groups:
      lr=1. if g['no_lr_decay'] else (oracle(step,selected,0,g['kind']) if rec else b['lr_multiplier'](step,selected,1800))
      # base warmup must match this slice's explicit zero warmup.
      if not rec and not g['no_lr_decay']:
       prev=b['WARMUP_STEPS']; b['WARMUP_STEPS']=0; lr=b['lr_multiplier'](step,selected,1800); b['WARMUP_STEPS']=prev
      assert g['lr']==.015*lr
      if g['kind']=='muon':
       p=min(1.,selected/1800)
       assert g['momentum']==c['muon_momentum'](step,p)
       assert g['beta2']==c['muon_beta2'](p)
       assert g['weight_decay']==.1*(1.-p)
     clockcases+=1

# All five required wrong outcomes are distinguishable from the oracle.
negative={}
negative['raw_elapsed_nonmuon']=oracle(6,1000.,0,'adamw')!=oracle(6,130.,0,'adamw')
negative['freeze_charged_startup']=oracle(1,130.,0,'muon')!=oracle(1,0.,0,'muon')
c['_PROCESS_STARTED']=0
negative['adjust_again_after_five']=c['startup_adjusted_schedule_elapsed'](5,130.,870.)!=max(0.,130.+30.-900.)
negative['ignore_flat_precedence']=oracle(6,1000.,0,'adamw')!=1.
route_node=next(n for n in ast.walk(cd['MuonAdamW']) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='one_beta2' for t in n.targets))
sentinel=object(); route_env=dict(RECURSIVE_SCHEDULE=True,MUON_BETA2_FINAL=0.,one_b2_t=sentinel,group={'beta2':.96})
exec(compile(ast.Module(body=[route_node],type_ignores=[]),'beta2-route','exec'),route_env)
negative['scalar_beta2']=route_env['one_beta2'] is sentinel and route_env['one_beta2']!=(1.-.96)
assert all(negative.values())

# Independently compare the raw receipts' per-step tensors/gradients/state hashes.
receipts={p.stem:json.loads(p.read_text()) for p in (ROOT/'raw').glob('clock*.json')}
for clock in (0,1):
 for left,right in [('base','default'),('base','off'),('on','reference')]:
  a,z=[receipts[f'clock{clock}-{x}'] for x in (left,right)]
  assert len(a['ranks'])==len(z['ranks'])==4
  for x,y in zip(a['ranks'],z['ranks']):
   assert x['rank']==y['rank']
   for key in ('losses','final_digests'):
    assert (key in x)==(key in y)
    assert x.get(key)==y.get(key)
   assert len(x['cache_observations']['steps'])==6
   assert x['cache_observations']['steps']==y['cache_observations']['steps']
   assert x['recursive_observations']['startup_steps_excluded']==5
   if x['rank']==0: assert x['checkpoint_check']['ok'] and y['checkpoint_check']['ok']
 for q in receipts[f'clock{clock}-on']['ranks']:
  assert all(r['is_tensor'] for r in q['recursive_observations']['beta2_routes'])
  for call in q['recursive_observations']['lr_calls']:
   assert call['result']==oracle(call['step'],call['elapsed'],0,call['kind'])

print(json.dumps(dict(status='PASS_SCOPED_SCALAR_CLOCK_ROUTING_AND_RAW_COMPARISON',source_sha256=sha(child.encode()),base_sha256=sha(base.encode()),changed_definitions=changed,scalar_lr_cases=count,clock_routing_cases=clockcases,negative_controls=negative,raw_tensor_comparisons='six arm pairs x four ranks x six updates; exact gradients, parameters and optimizer state digest structures',thread_environment={k:os.getenv(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},python=sys.version,limitations=['No independently executed full model/update path in this scalar reference; actual updates are raw supplied evidence.','No independent organizer causality or eval state-invariance evidence.','No native capture, exact-size G2, cold or quality approval.'],input_manifest={str(p.relative_to(ROOT)):sha(p.read_bytes()) for d in ('review-input','raw') for p in sorted((ROOT/d).iterdir()) if p.is_file()}),indent=2))
