"""Independent bounded CPU review; no full train module imports or harness reuse.
Execute: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B /work/reference_test.py --attempt attempt1
"""
import argparse, ast, collections, copy, hashlib, json, math, pathlib, struct, sys, textwrap, traceback, types, weakref
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
W=pathlib.Path('/work')
ap=argparse.ArgumentParser(); ap.add_argument('--attempt',required=True); cli=ap.parse_args()
OUT=W/cli.attempt; OUT.mkdir(exist_ok=False)
torch.set_num_threads(1); torch.set_num_interop_threads(1)
assert torch.get_num_threads()==torch.get_num_interop_threads()==1
log=(OUT/'results.jsonl').open('w'); failures=[]; counts=collections.Counter(); extraction=[]
def record(kind,name,passed,**data):
    row=dict(kind=kind,name=name,passed=bool(passed),**data)
    log.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');log.flush()
    counts[kind]+=1
    if not passed: failures.append(row)
def chk(name,condition,**data): record('check',name,condition,**data)
def negative(name,expected,wrong):
    try:
        assert expected==wrong, f'expected {expected!r}; deliberately wrong actual {wrong!r}'
    except AssertionError:
        record('negative_control',name,True,expected=expected,wrong_actual=wrong,expected_failure=traceback.format_exc())
    else: record('negative_control',name,False,expected=expected,wrong_actual=wrong)
def bits(x): return struct.pack('<d',x).hex()
def exact_structure(x):
    if isinstance(x,float):return ('binary64',bits(x))
    if isinstance(x,dict):return tuple((k,exact_structure(v)) for k,v in sorted(x.items()))
    if isinstance(x,(list,tuple)):return tuple(exact_structure(v) for v in x)
    return x
def fp32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def bytes32(xs): return b''.join(struct.pack('<f',x) for x in xs).hex()
def source(role): return (W/f'{role}_train.py').read_text()
def tree(role): return ast.parse(source(role))
def compile_nodes(role,label,nodes,ns):
    mod=ast.Module(body=copy.deepcopy(nodes),type_ignores=[]);ast.fix_missing_locations(mod)
    code=ast.unparse(mod)+'\n'; path=OUT/f'{role}_{label}.py';path.write_text(code)
    extraction.append(dict(role=role,label=label,original_lines=[[n.lineno,n.end_lineno] for n in nodes],artifact=path.name,sha256=hashlib.sha256(code.encode()).hexdigest()))
    exec(compile(mod,str(W/f'{role}_train.py'),'exec'),ns)
def getfn(role,name): return next(n for n in tree(role).body if isinstance(n,ast.FunctionDef) and n.name==name)
def main_nodes(role): return getfn(role,'main').body
def getloop(role): return next(n for n in main_nodes(role) if isinstance(n,ast.While))
def isassign(n,name): return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
def schedule_nodes(role):
    body=getloop(role).body; start=next(i for i,n in enumerate(body) if isassign(n,'schedule_elapsed' if role=='candidate' else 'lrm'))
    end=next(i for i,n in enumerate(body[start:],start) if isinstance(n,ast.For) and ast.unparse(n.iter)=='optimizer.param_groups')
    return body[start:end+1]
def transition_node(role):
    return next(n for n in getloop(role).body if isinstance(n,ast.If) and ast.unparse(n.test)=='step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED')
class NoClock:
    calls=0
    def monotonic(self):
        self.calls+=1
        raise AssertionError('unexpected extra clock sample')
class ScriptClock:
    def __init__(self,values):self.values=iter(values);self.calls=0
    def monotonic(self):self.calls+=1;return next(self.values)
NAMES='STARTUP_LAUNCH_MARGIN_SECONDS WARMUP_STEPS WARMDOWN_RATIO FINAL_LR_FRAC LRM_QUANTUM WEIGHT_DECAY WD_DECAY_TO_ZERO MUON_MOMENTUM_WARMUP_STEPS MUON_MOMENTUM_COOLDOWN_FRAC MUON_MOMENTUM_FINAL DEMON_BETA1 DEMON_BETA1_START_FRAC DEMON_BETA1_FINAL DEMON_BETA1_REF MUON_BETA2 MUON_BETA2_FINAL USE_TENSOR_LR_SCALARS USE_CAUTIOUS_WD'.split()
def namespace(role):
    ns={'TRAIN_STARTUP_STEPS_EXCLUDED':5,'TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS':900.0,'_PROCESS_STARTED':1024.0,'torch':torch,'time':NoClock()}
    for n in tree(role).body:
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name) and t.id in NAMES: ns[t.id]=ast.literal_eval(n.value)
    names=['lr_multiplier','muon_momentum','demon_beta1_scale','muon_beta2']
    if role=='candidate':names+=['startup_adjusted_schedule_elapsed']
    compile_nodes(role,'schedule_functions',[getfn(role,n) for n in names],ns)
    return ns

def groups(): return [dict(kind='muon',initial_lr=.015),dict(kind='muon',initial_lr=.015,no_lr_decay=True),dict(kind='adamw',initial_lr=.02,initial_betas=(.8,.95),betas=(.8,.95),weight_decay=.01)]
def schedule(role,ns,step,elapsed,budget,enabled,overrides=None):
    g=groups(); env=ns.copy(); env.update(step=step,elapsed=elapsed,budget_started=budget,args=types.SimpleNamespace(startup_adjusted_schedule=enabled,max_train_seconds=1800.),batch_lr_scale=1.,optimizer=types.SimpleNamespace(param_groups=g))
    if overrides: env.update(overrides)
    # Extracted functions resolve global settings from their original namespace.
    save={k:ns.get(k) for k in (overrides or {})};ns.update(overrides or {})
    lr_inputs=[]; original=ns['lr_multiplier']
    def spy(step,elapsed=0.,max_train_seconds=1.):
        lr_inputs.append(elapsed);return original(step,elapsed,max_train_seconds)
    env['lr_multiplier']=spy
    exec(SCHEDULE_CODE[role],env)
    ns.update(save)
    return dict(schedule_elapsed=env.get('schedule_elapsed',elapsed),wd_progress=env['wd_progress'],lrm=env['lrm'],lrm_flat=env['lrm_flat'],groups=g,lr_inputs=lr_inputs)

def independent_lr(step,seconds,warmup=0):
    if step<warmup:return (step+1)/max(1,warmup)
    p=min(1.,seconds/1800.)
    return 1. if p<.25 else 1.-.95*((p-.25)/.75)

def scalar_clock_reference(step,now,process,origin,enabled):
    if not enabled or step>=5:return now-origin
    return max(0.,now-(process+870.))

class Tracker(TorchDispatchMode):
    """Track every live output tensor, deduplicate CPU storages; include intermediates."""
    def __init__(self):super().__init__();self.refs={};self.peak=0;self.max_tensor=0;self.calls=0
    def track(self,obj):
        for t in tree_flatten(obj)[0]:
            if isinstance(t,torch.Tensor):
                assert t.device.type=='cpu'
                self.refs[id(t)]=weakref.ref(t)
                self.max_tensor=max(self.max_tensor,t.numel()*t.element_size())
        live={};dead=[]
        for key,ref in self.refs.items():
            t=ref()
            if t is None:dead.append(key);continue
            s=t.untyped_storage();live[s._cdata]=s.nbytes()
        for key in dead:del self.refs[key]
        total=sum(live.values());self.peak=max(self.peak,total)
        assert total<64*1024*1024, f'live tensor storage budget exceeded: {total}'
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.calls+=1;self.track((args,kwargs));out=func(*args,**(kwargs or {}));self.track(out);return out

try:
    NS={role:namespace(role) for role in ['baseline','candidate']}
    SCHEDULE_CODE={}
    for role in NS:
        nodes=schedule_nodes(role);mod=ast.Module(body=nodes,type_ignores=[])
        SCHEDULE_CODE[role]=compile(mod,str(W/f'{role}_train.py'),'exec')
        (OUT/f'{role}_callsite.py').write_text(ast.unparse(mod)+'\n')
        extraction.append(dict(role=role,label='callsite',original_lines=[[n.lineno,n.end_lineno] for n in nodes]))
    # Actual CLI declaration; no environment-dependent full argument parser.
    argnode=next(n for n in main_nodes('candidate') if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and n.value.args and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value=='--startup-adjusted-schedule')
    parser=argparse.ArgumentParser();compile_nodes('candidate','cli',[argnode],dict(parser=parser,argparse=argparse))
    parsed=[parser.parse_args(x).startup_adjusted_schedule for x in [[],['--no-startup-adjusted-schedule'],['--startup-adjusted-schedule']]]
    chk('real_boolean_cli',parsed==[False,False,True],argv=[[],['--no-startup-adjusted-schedule'],['--startup-adjusted-schedule']],expected=[False,False,True],actual=parsed)
    # Exhaustive finite grid, dyadic timestamps: exact binary64 clock equality.
    for init in [0.,37.25,1024.]:
      for age in [37.25,869.75,870.,870.25,899.75,900.,900.25,1000.,1500.,2670.,3000.]:
        if age<init:continue
        for step in [0,1,2,3,4,5,6,300]:
          process=1024.;now=process+age;origin=process+init;elapsed=now-origin
          outputs={}
          for mode,enabled in [('default',parsed[0]),('off',parsed[1]),('on',parsed[2])]:
            out=schedule('candidate',NS['candidate'],step,elapsed,origin,enabled)
            exp=scalar_clock_reference(step,now,process,origin,enabled);lr=independent_lr(step,exp)
            ok=bits(out['schedule_elapsed'])==bits(exp) and out['lr_inputs']==[exp] and bits(out['wd_progress'])==bits(min(1.,exp/1800.)) and abs(out['lrm']-lr)<=4e-16
            chk('clock_and_lr_wd_grid',ok,inputs=dict(init=init,process=process,now=now,step=step,elapsed=elapsed,mode=mode),expected=dict(seconds=exp,wd_progress=min(1.,exp/1800.),lr=lr),actual=out)
            outputs[mode]=out
          base=schedule('baseline',NS['baseline'],step,elapsed,origin,False)
          chk('default_off_baseline_binary_identity',exact_structure(outputs['default'])==exact_structure(outputs['off'])==exact_structure(base),inputs=dict(step=step,init=init,age=age),expected=base,actual=dict(default=outputs['default'],off=outputs['off']))
    for step in [0,1,2,3,4,5]:
        out=schedule('candidate',NS['candidate'],step,2000.,1061.25,True,{'WARMUP_STEPS':3})
        e=out['schedule_elapsed'];normal=independent_lr(step,e,3);flat=min(1.,(step+1)/3)
        chk('step_warmup_and_flat_lr_precedence',abs(out['lrm']-normal)<=4e-16 and out['groups'][1]['lr']==.015*flat,inputs=dict(step=step,elapsed=2000.,warmup=3),expected=dict(normal=normal,flat_lr=.015*flat),actual=out)
    chk('no_extra_clock_reads',all(ns['time'].calls==0 for ns in NS.values()),actual={k:v['time'].calls for k,v in NS.items()},expected=0)
    # Inactive helper must not be evaluated, including for weird OFF inputs.
    helper=NS['candidate']['startup_adjusted_schedule_elapsed']
    def forbidden(*a):raise AssertionError('OFF called helper')
    NS['candidate']['startup_adjusted_schedule_elapsed']=forbidden
    for value in [-0.0,0.001,790.006,1e10]:
        a=schedule('candidate',NS['candidate'],0,value,1061.25,False);b=schedule('baseline',NS['baseline'],0,value,1061.25,False)
        chk('off_bypasses_helper_preserves_elapsed_bits',a==b and bits(a['schedule_elapsed'])==bits(value),expected=b,actual=a,elapsed_bits=bits(value))
    NS['candidate']['startup_adjusted_schedule_elapsed']=helper
    # Execute unchanged entire fifth-boundary block against scripted end samples.
    for role,ns in NS.items():
      trans=transition_node(role);(OUT/f'{role}_fifth_boundary.py').write_text(ast.unparse(trans)+'\n')
      code=compile(ast.Module(body=[trans],type_ignores=[]),str(W/f'{role}_train.py'),'exec')
      for step in [3,4,5]:
        for boundary_age in [869.75,870.,870.25,900.,1000.,1500.]:
          process=1024.;old_origin=1061.25;now=process+boundary_age;clock=ScriptClock([now]);recent=[790.,.25,.5]
          env=dict(ns,step=step,budget_started=old_origin,startup_allowance_est=0.,recent_step_times=recent,time=clock)
          exec(code,env)
          expected_origin=min(now,process+870.) if step==4 else old_origin
          expected_recent=[] if step==4 else [790.,.25,.5]
          ok=bits(env['budget_started'])==bits(expected_origin) and recent==expected_recent and clock.calls==int(step==4) and env['startup_allowance_est']==(min(boundary_age,900.) if step==4 else 0.)
          chk('fifth_boundary_reference',ok,inputs=dict(role=role,step=step,now=now,old_origin=old_origin,recent_before=[790.,.25,.5]),expected=dict(origin=expected_origin,recent=expected_recent,clock_calls=int(step==4)),actual=dict(origin=env['budget_started'],recent=recent,allowance_est=env['startup_allowance_est'],clock_calls=clock.calls))
          if step==4:
            next_now=now+.25;seconds=next_now-expected_origin
            out=schedule('candidate',NS['candidate'],5,next_now-env['budget_started'],env['budget_started'],True)
            chk('step5_uses_charged_elapsed_exactly',bits(out['schedule_elapsed'])==bits(seconds),inputs=dict(boundary_age=boundary_age,post_boundary_delta=.25),expected=seconds,actual=out)
    # Executed semantic mutants: source files themselves remain untouched.
    ns=NS['candidate'];correct=ns['startup_adjusted_schedule_elapsed']
    mutant_inputs=dict(step=1,elapsed=962.75,budget_started=1061.25,process_origin=1024.,now=2024.)
    (OUT/'mutant_inputs.json').write_text(json.dumps(mutant_inputs,indent=2)+'\n')
    def run_mutant(name,fn):
        ns['startup_adjusted_schedule_elapsed']=fn
        actual=schedule('candidate',ns,1,962.75,1061.25,True)
        negative(name,scalar_clock_reference(1,2024.,1024.,1061.25,True),actual['schedule_elapsed'])
        ns['startup_adjusted_schedule_elapsed']=correct
    run_mutant('uncapped_startup_forgiveness_discards_excess',lambda step,elapsed,budget: 0. if step<5 else elapsed)
    run_mutant('missing_30_second_launch_margin',lambda step,elapsed,budget: max(0.,elapsed+budget-1024.-900.))
    run_mutant('ignoring_process_to_loop_initialization',lambda step,elapsed,budget: max(0.,elapsed-870.))
    trans=copy.deepcopy(transition_node('candidate'))
    for n in trans.body:
        if isassign(n,'budget_started'):n.value=ast.Name(id='now',ctx=ast.Load())
    ast.fix_missing_locations(trans)
    (OUT/'mutant_reset_to_now.py').write_text(ast.unparse(trans)+'\n')
    env=dict(ns,step=4,budget_started=1061.25,startup_allowance_est=0.,recent_step_times=[790.],time=ScriptClock([2024.]))
    exec(compile(ast.Module(body=[trans],type_ignores=[]),'mutant_reset_to_now','exec'),env)
    wrong=schedule('candidate',ns,5,2024.25-env['budget_started'],env['budget_started'],True)
    negative('resetting_origin_to_now_loses_excess_at_step5',130.25,wrong['schedule_elapsed'])
    env=dict(ns,step=4,budget_started=1061.25,startup_allowance_est=0.,recent_step_times=[790.],time=ScriptClock([2024.]),TRAIN_STARTUP_STEPS_EXCLUDED=6)
    exec(compile(ast.Module(body=[transition_node('candidate')],type_ignores=[]),'mutant_sixth_boundary','exec'),env)
    negative('delaying_reset_to_sixth_boundary',1894.,env['budget_started'])
    nodes=copy.deepcopy(schedule_nodes('candidate'))
    for n in nodes:
        if isassign(n,'wd_progress'):
            for k in ast.walk(n):
                if isinstance(k,ast.Name) and k.id=='schedule_elapsed':k.id='elapsed'
    env=dict(ns,step=1,elapsed=790.006,budget_started=1061.25,args=types.SimpleNamespace(startup_adjusted_schedule=True,max_train_seconds=1800.),batch_lr_scale=1.,optimizer=types.SimpleNamespace(param_groups=groups()))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'mutant_split_LR_WD','exec'),env)
    (OUT/'mutant_split_LR_WD.py').write_text(ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n')
    negative('LR_WD_clock_split',0.,env['wd_progress'])
    # Prove all other source operations unchanged, using complete ASTs.
    btree=tree('baseline');ctree=tree('candidate')
    ctree.body=[n for n in ctree.body if not(isinstance(n,ast.FunctionDef) and n.name=='startup_adjusted_schedule_elapsed')]
    cm=next(n for n in ctree.body if isinstance(n,ast.FunctionDef) and n.name=='main');cm.body=[n for n in cm.body if not(isinstance(n,ast.Expr) and ast.dump(n)==ast.dump(argnode))]
    cloop=next(n for n in cm.body if isinstance(n,ast.While));cloop.body=[n for n in cloop.body if not isassign(n,'schedule_elapsed')]
    for n in cloop.body:
        if isassign(n,'lrm') or isassign(n,'wd_progress'):
            for k in ast.walk(n):
                if isinstance(k,ast.Name) and k.id=='schedule_elapsed':k.id='elapsed'
    chk('complete_source_AST_identical_except_requested_delta',ast.dump(btree)==ast.dump(ctree),baseline_sha256=hashlib.sha256(source('baseline').encode()).hexdigest(),candidate_sha256=hashlib.sha256(source('candidate').encode()).hexdigest())

    # Tiny exactness fixture. Source methods are bound to a minimal state container;
    # no optimizer base constructor, model, compiler, distributed runtime, or device import.
    def make_opt(role):
        cls=next(n for n in tree(role).body if isinstance(n,ast.ClassDef) and n.name=='MuonAdamW')
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_device_scalar','_adamw_state','_adamw_step_foreach']]
        ns=dict(torch=torch,USE_TENSOR_LR_SCALARS=True,USE_CAUTIOUS_WD=False)
        compile_nodes(role,'adamw_methods',methods,ns)
        obj=types.SimpleNamespace(state=collections.defaultdict(dict),_scalar_cache={})
        for n in methods:setattr(obj,n.name,types.MethodType(ns[n.name],obj))
        return obj
    def scalar_lerp(a,b,w):
        w=fp32(w);d=fp32(b-a)
        # CPU scalar lerp's numerically stable branch, one final float32 rounding.
        return fp32(math.fma(w,d,a)) if w<.5 else fp32(math.fma(-fp32(1.-w),d,b))
    def ref_update(p,g,state,lr,wd,betas,eps):
        if g is None:return p,state,[0.]*len(p)
        n=state['step']+1;b1,b2=betas
        m=[scalar_lerp(a,b,1.-b1) for a,b in zip(state['m'],g)]
        v=[scalar_lerp(a,fp32(b*b),1.-b2) for a,b in zip(state['v'],g)]
        # foreach scalar division is a float32 reciprocal followed by multiplication.
        c1=fp32(1./(1.-b1**n));c2=fp32(1./(1.-b2**n))
        denom=[fp32(fp32(math.sqrt(fp32(x*c2)))+fp32(eps)) for x in v]
        update=[fp32(fp32(x*c1)/d) for x,d in zip(m,denom)]
        lr32=fp32(lr);decay=fp32(1.-fp32(lr32*fp32(wd)))
        out=[fp32(fp32(x*decay)-fp32(u*lr32)) for x,u in zip(p,update)]
        return out,dict(step=n,m=m,v=v),[fp32(x-y) for x,y in zip(out,p)]
    INITIAL=[[1.,-.5,.25,-2.],[.75,-1.25],[3.]]
    ages=[100.,1500.,1520.,1600.,1800.,1800.25,2200.,2670.]
    fixture=dict(parameters=INITIAL,ages=ages,process_origin=1024.,loop_initialization=37.25,betas=[.5,.75],eps=.125,weight_decay=.125,initial_lr=.02,dtype='float32',missing_gradient_rule='p1 has None at steps 0,2,5; p2 always None',gradient_rule='p0: [(-1)**step * (step+1)/8, -.25, .5, -(step+2)/16]; p1: [.125*(step+1), -.375] when present')
    (OUT/'tensor_inputs.json').write_text(json.dumps(fixture,indent=2)+'\n')
    tracker=Tracker();runs={}
    with tracker:
      for mode in ['baseline','default','off','on']:
        role='baseline' if mode=='baseline' else 'candidate';opt=make_opt(role)
        params=[torch.nn.Parameter(torch.tensor(p,dtype=torch.float32)) for p in INITIAL]
        expected=copy.deepcopy(INITIAL);states=[dict(step=0,m=[0.]*len(p),v=[0.]*len(p)) for p in INITIAL]
        group=dict(params=params,kind='adamw',lr=.02,initial_lr=.02,betas=(.5,.75),weight_decay=.125,eps=.125)
        run=[];origin=1061.25
        for step,age in enumerate(ages):
          if step==5:origin=1024.+870. # independently calculated fifth boundary at age 1800
          now=1024.+age;elapsed=now-origin
          sched=schedule(role,NS[role],step,elapsed,origin,mode=='on');group['lr']=sched['groups'][2]['lr']
          seconds=scalar_clock_reference(step,now,1024.,origin,mode=='on');lr_ref=.02*independent_lr(step,seconds)
          # Independent algebraic LR reference must round to same optimizer fp32 scalar.
          chk('tensor_lr_scalar_fp32',bytes32([lr_ref])==bytes32([group['lr']]),mode=mode,step=step,expected=lr_ref,actual=group['lr'])
          grad_values=[[(-1)**step*(step+1)/8.,-.25,.5,-(step+2)/16.],None if step in [0,2,5] else [.125*(step+1),-.375],None]
          for p,g in zip(params,grad_values):p.grad=None if g is None else torch.tensor(g,dtype=torch.float32)
          before=[p.detach().tolist() for p in params]
          opt._adamw_step_foreach(group)
          row=dict(mode=mode,step=step,seconds=seconds,lr_expected=lr_ref,lr_actual=group['lr'],gradients=grad_values,parameters=[])
          for i,(p,g) in enumerate(zip(params,grad_values)):
            exp,new_state,delta=ref_update(expected[i],g,states[i],lr_ref,.125,(.5,.75),.125);expected[i]=exp;states[i]=new_state
            actual=p.detach().tolist();st=opt.state.get(p,{})
            actual_state=dict(step=st.get('step',0),m=st['exp_avg'].tolist() if st else [0.]*len(actual),v=st['exp_avg_sq'].tolist() if st else [0.]*len(actual))
            actual_delta=(p.detach()-torch.tensor(before[i],dtype=torch.float32)).tolist()
            result=dict(index=i,gradient_missing=g is None,state_present=bool(st),expected_parameter=exp,actual_parameter=actual,expected_bytes=bytes32(exp),actual_bytes=bytes32(actual),expected_delta=delta,actual_delta=actual_delta,expected_state=new_state,actual_state=actual_state)
            exact=bytes32(exp)==bytes32(actual) and new_state['step']==actual_state['step'] and all(bytes32(new_state[k])==bytes32(actual_state[k]) for k in ['m','v']) and bytes32(delta)==bytes32(actual_delta)
            chk('tensor_update_state_byte_exact',exact,mode=mode,step=step,**result);row['parameters'].append(result)
          run.append(row)
        chk('always_missing_gradient_no_state',params[2] not in opt.state,mode=mode,expected=False,actual=params[2] in opt.state)
        runs[mode]=run
        del params,opt,group,p,st
    def numerical_projection(rows):return [[{k:r[k] for k in ['actual_bytes','actual_state','actual_delta','gradient_missing','state_present']} for r in row['parameters']] for row in rows]
    chk('tiny_baseline_default_off_exact',numerical_projection(runs['baseline'])==numerical_projection(runs['default'])==numerical_projection(runs['off']))
    chk('tiny_ON_intentional_difference',numerical_projection(runs['on'])!=numerical_projection(runs['off']))
    negative('tensor_one_bit_mutation',runs['on'][0]['parameters'][0]['actual_bytes'],bytes.fromhex(runs['on'][0]['parameters'][0]['actual_bytes'])[0:1].hex()+'ff'+runs['on'][0]['parameters'][0]['actual_bytes'][4:])
    (OUT/'tensor_results.json').write_text(json.dumps(runs,indent=2)+'\n')
    chk('tensor_memory_bound',tracker.peak<64*1024*1024,tracked_peak_storage_bytes=tracker.peak,max_tensor_bytes=tracker.max_tensor,dispatch_calls=tracker.calls,limit_bytes=64*1024*1024)
except Exception:
    record('unexpected_error','uncaught',False,traceback=traceback.format_exc())
finally:
    summary=dict(status='PASS' if not failures else 'REFUSE',counts=dict(counts),failures=len(failures),python=sys.version,torch=torch.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),scope='Extracted source functions/callsite/transition and tiny FP32 AdamW fixture only; no full model or distributed/native execution',stubs={'prepare_constants':{'TRAIN_STARTUP_STEPS_EXCLUDED':5,'TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS':900.},'time':'NoClock or scripted values','optimizer_container':'SimpleNamespace/defaultdict, binding actual methods; no math stub'})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'extractions.json').write_text(json.dumps(extraction,indent=2)+'\n');(OUT/'failures.json').write_text(json.dumps(failures,indent=2)+'\n');log.close();print(json.dumps(summary,indent=2))
    sys.exit(0 if not failures else 1)
