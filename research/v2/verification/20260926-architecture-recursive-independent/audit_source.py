"""Independent schedule numerics, exact source transport and reduced raw receipt audit."""
import ast
from decimal import Decimal,localcontext
import hashlib
import json
import math
from pathlib import Path
import subprocess

REPO=Path('/home/poojith-devan/Desktop/Trainium')
PATH='workflow/proposals/research-v2/train.py'
REF='b4d83b5db66971f3c348837f04f53cba65aa6d33'
CAND='fe0969fd311d654e143dc6113614416c8970a424'
OLD='a4aa4351789115a2d0aa947d9ca00ff689f9bd0b'
RAW='d1ceed83f0cc38605fc262eecf3784e0d9d50fd0'
checks=[]
def check(name,value):
 if not value:raise RuntimeError('FAIL: '+name)
 checks.append(name)
def blob(ref,path=PATH):return subprocess.check_output(['git','-C',str(REPO),'show',ref+':'+path])
def sha(data):return hashlib.sha256(data).hexdigest()
sources={name:blob(ref) for name,ref in [('baseline',REF),('candidate',CAND),('adopted',OLD)]}
trees={k:ast.parse(v) for k,v in sources.items()}
check('exact source pins',{k:sha(v) for k,v in sources.items()}=={'baseline':'3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c','candidate':'75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f','adopted':'6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'})
names=('lr_multiplier','muon_momentum','muon_beta2','demon_beta1_scale','current_toggles','restore_eval_toggles')
def functions(tree):return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
defs={k:functions(v) for k,v in trees.items()}
def extract(tree):
 ns={}
 for n in tree.body:
  if isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) for t in n.targets):
   try:value=ast.literal_eval(n.value)
   except (ValueError,TypeError):continue
   for t in n.targets:ns[t.id]=value
 fs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 exec(compile(ast.Module(body=fs,type_ignores=[]),'exact-schedule-functions','exec',optimize=0),ns)
 return ns
nss={k:extract(t) for k,t in trees.items()}
for name in names[:3]:check(name+' exact adopted AST',ast.dump(defs['candidate'][name])==ast.dump(defs['adopted'][name]))
for name in ('GPT','GPTConfig','_SubmissionEvalGPT','load_for_eval','save_checkpoint','_install_submission_training'):
 check(name+' unchanged baseline AST',ast.dump(defs['candidate'][name])==ast.dump(defs['baseline'][name]))
old_opt={n.name:n for n in defs['baseline']['MuonAdamW'].body if isinstance(n,ast.FunctionDef)}
new_opt={n.name:n for n in defs['candidate']['MuonAdamW'].body if isinstance(n,ast.FunctionDef)}
check('device scalar fill unchanged',ast.dump(old_opt['_muon_compile_scalars'])==ast.dump(new_opt['_muon_compile_scalars']))
assignment=next(n for n in ast.walk(new_opt['_muon_step']) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='one_beta2' for t in n.targets))
transport=compile(ast.Module(body=[assignment],type_ignores=[]),'exact-beta2-transport','exec',optimize=0)
for on,ramp,use_tensor in [(False,-1.,False),(True,-1.,True),(False,.97,True)]:
 sentinel=object();ctx={'RECURSIVE_SCHEDULE':on,'MUON_BETA2_FINAL':ramp,'one_b2_t':sentinel,'group':{'beta2':.95}}
 exec(transport,ctx);check('beta2 scalar transport '+str((on,ramp)),(ctx['one_beta2'] is sentinel) if use_tensor else ctx['one_beta2']==1-.95)

grid=[-180.,0.,89.999999,90.,90.000001,450.,630.,900.,1710.,1800.,1980.]
max_decimal_error=0.;numeric_cases=0
for warmup in (0,3,300):
 for step in (0,1,2,3,299,300,600):
  for elapsed in grid:
   for ns in nss.values():ns['WARMUP_STEPS']=warmup
   nss['candidate']['RECURSIVE_SCHEDULE']=False
   for kind in ('muon','adamw'):
    a=nss['candidate']['lr_multiplier'](step,elapsed,1800.,kind)
    check('OFF exact lr '+str((warmup,step,elapsed,kind)),a==nss['baseline']['lr_multiplier'](step,elapsed,1800.))
   for fn,args in [('muon_momentum',(step,elapsed/1800.)),('muon_beta2',(elapsed/1800.,))]:
    check('OFF exact '+fn+str((warmup,step,elapsed)),nss['candidate'][fn](*args)==nss['baseline'][fn](*args))
   nss['candidate']['RECURSIVE_SCHEDULE']=True;nss['adopted']['RECURSIVE_SCHEDULE']=True
   for kind in ('muon','adamw'):
    actual=nss['candidate']['lr_multiplier'](step,elapsed,1800.,kind)
    check('ON adopted exact lr '+str((warmup,step,elapsed,kind)),actual==nss['adopted']['lr_multiplier'](step,elapsed,1800.,kind))
    with localcontext() as ctx:
     ctx.prec=50;D=Decimal;p=min(D(1),max(D(0),D(str(elapsed))/D(1800)))
     wd=D('.95') if kind=='muon' else D('.65')
     # Independent high-precision piecewise line through (1-wd,1),(1,.05).
     expected=D(step+1)/D(max(1,warmup)) if step<warmup else D(1) if p<1-wd else D('.05')+(1-p)*D('.95')/wd
    error=abs(actual-float(expected));max_decimal_error=max(max_decimal_error,error);numeric_cases+=1
    check('ON decimal lr '+str((warmup,step,elapsed,kind)),error<=2e-15)
   for fn,args in [('muon_momentum',(step,elapsed/1800.)),('muon_beta2',(elapsed/1800.,))]:
    check('ON adopted exact '+fn+str((warmup,step,elapsed)),nss['candidate'][fn](*args)==nss['adopted'][fn](*args))
   p=min(1.,max(0.,elapsed/1800.))
   # Independent algebraic momentum/beta2 references, tolerance covers float reassociation only.
   frac=min(step/300.,1.);m=.85+.10*frac if p<=1.-.95 else .95-.16*((p-(1.-.95))/.95)**2
   beta=.95+.02*max(0.,(p-(1.-.95))/.95)
   check('ON numeric momentum '+str((warmup,step,elapsed)),abs(nss['candidate']['muon_momentum'](step,elapsed/1800.)-m)<=2e-15)
   check('ON numeric beta2 '+str((warmup,step,elapsed)),abs(nss['candidate']['muon_beta2'](elapsed/1800.)-beta)<=2e-15)

# Execute the actual training-loop LR branch, including flat-table precedence.
main=defs['candidate']['main']
group_loop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and ast.unparse(n.iter)=='optimizer.param_groups' and any(isinstance(v,ast.Assign) and ast.unparse(v.targets[0])=="group['lr']" for v in n.body))
assigns=[n for n in group_loop.body if isinstance(n,ast.Assign)]
for kind in ('muon','adamw','rmsprop'):
 for flat in (False,True):
  group={'kind':kind,'initial_lr':.015,'no_lr_decay':flat};ctx={'group':group,'batch_lr_scale':1.,'lrm':.6,'lrm_adam':.9,'lrm_flat':1.}
  exec(compile(ast.Module(body=assigns,type_ignores=[]),'actual-group-lr-assignments','exec',optimize=0),ctx)
  check('group kind/flat precedence '+str((kind,flat)),group['lr']==.015*(1. if flat else .6 if kind=='muon' else .9))
for payload,expected in [({},False),({'toggles':{'RECURSIVE_SCHEDULE':False}},False),({'toggles':{'RECURSIVE_SCHEDULE':True}},True)]:
 ns=nss['candidate'];ns.update(ARCH_TOGGLE_NAMES=['RECURSIVE_SCHEDULE'],_EVAL_TOGGLE_DEFAULTS={'RECURSIVE_SCHEDULE':False},RECURSIVE_SCHEDULE=True)
 ns['restore_eval_toggles'](payload);check('checkpoint toggle restoration '+str(payload),ns['RECURSIVE_SCHEDULE'] is expected)
check('CLI default false',any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--recursive-schedule' and any(k.arg=='default' and isinstance(k.value,ast.Constant) and k.value.value is False for k in n.keywords) for n in ast.walk(main)))

raw={};raw_pins={}
for arm in ('reference','default','off','on'):
 data=blob(RAW,'research/v2/lanes/architecture-recursive-stack/tiny-g2-'+arm+'.json');r=json.loads(data);raw[arm]=r;raw_pins[arm]=sha(data)
 check(arm+' source pin',r['train_py_sha256']==sha(sources['baseline' if arm=='reference' else 'candidate']))
 check(arm+' actual CPU scope',r['nproc']==4 and r['steps']==3 and r['micro_batch']==[1,16] and r['prepare']['stub'] is True and r['torchrun_rc']==0 and r['timed_out'] is False)
 ranks=sorted(r['ranks'],key=lambda x:x['rank']);check(arm+' all rank ids',[x['rank'] for x in ranks]==[0,1,2,3])
 for rank in ranks:
  check(arm+' rank finite/update/consistency '+str(rank['rank']),rank['status']=='PASS' and rank['n_changed']==rank['n_params']==65 and not rank['cross_rank_mismatches'] and len(rank['losses'])==3 and all(math.isfinite(x) for x in rank['losses']))
 check(arm+' rank0 all65 retained digests',len(ranks[0]['final_digests'])==65)
 check(arm+' actual retained reload',ranks[0]['checkpoint_check']==dict(ranks[0]['checkpoint_check'],ok=True,deterministic=True,path_exists=True,logits_shape=[1,16,8192]))
for arm in ('default','off'):
 a,b=raw['reference'],raw[arm]
 check(arm+' exact actual12losses/all65rank0finaldigests/plan',a['plan_sha256']==b['plan_sha256'] and a['ranks'][0]['final_digests']==b['ranks'][0]['final_digests'] and all(x['losses']==y['losses'] and x['plan_summary']==y['plan_summary'] for x,y in zip(a['ranks'],b['ranks'])))
changes=sum(raw['on']['ranks'][0]['final_digests'][k]!=v for k,v in raw['off']['ranks'][0]['final_digests'].items())
check('ON intentionally differs',changes>0)
ns=nss['candidate'];ns.update(RECURSIVE_SCHEDULE=True,WARMUP_STEPS=0)
examples=[{'progress':p,'step':600,'muon_lr':ns['lr_multiplier'](600,p*1800,1800,'muon'),'adam_lr':ns['lr_multiplier'](600,p*1800,1800,'adamw'),'momentum':ns['muon_momentum'](600,p),'beta2':ns['muon_beta2'](p)} for p in (0,.05,.25,.35,.5,.95,1)]
print(json.dumps({'verdict':'PASS_SOURCE_AND_REDUCED_NUMERICAL_SCOPE_ONLY_FRESH_REVIEW_PENDING','checks':len(checks),'decimal_lr_cases':numeric_cases,'maximum_decimal_lr_error':max_decimal_error,'on_changed_final_tensors':changes,'source_hashes':{k:sha(v) for k,v in sources.items()},'raw_receipt_hashes':raw_pins,'raw_plan_sha256':raw['reference']['plan_sha256'],'test_name_sha256':sha(json.dumps(checks).encode()),'static_and_receipt_tests':checks[:14]+checks[-46:],'numeric_grid':{'warmup_steps':[0,3,300],'steps':[0,1,2,3,299,300,600],'elapsed_seconds':grid,'max_train_seconds':1800,'kinds':['muon','adamw'],'independent_reference_tolerance':2e-15,'exact_adopted_and_OFF_comparisons':True},'representative_ON_values':examples,'fullsize_cpu':'PENDING','native':'PENDING','cold':'PENDING','quality':'PENDING','production_source_edited':False,'tensor_allocations':False,'hosts':False},indent=2))
