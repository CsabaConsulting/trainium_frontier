"""Own exact-source/fake-clock reference; stdlib only, no model imports."""
import argparse
import ast
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import types

REPO=Path(__file__).resolve().parents[4]
BASE='b4d83b5db66971f3c348837f04f53cba65aa6d33'
CANDIDATE='7419927c7417b5666a984ec9c2b849feadabaaa9'
SOURCE='workflow/proposals/research-v2/train.py'
def blob(commit):return subprocess.check_output(['git','show',commit+':'+SOURCE],cwd=REPO)
def sha(raw):return hashlib.sha256(raw).hexdigest()
checks=[]
def check(name,value):
    if not value:raise RuntimeError(name)
    checks.append(name)
oldraw,newraw=blob(BASE),blob(CANDIDATE)
check('exact baseline',sha(oldraw)=='3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c')
check('exact candidate',sha(newraw)=='36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb')
old,new=ast.parse(oldraw),ast.parse(newraw)
def dump(node):return ast.dump(node,include_attributes=False)
functions={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
oldfunctions={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
for name in oldfunctions:
    if name!='main':check('unchanged function '+name,dump(functions[name])==dump(oldfunctions[name]))
main=functions['main'];oldmain=oldfunctions['main']
parser_stmt=next(n for n in ast.walk(main) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and any(isinstance(a,ast.Constant) and a.value=='--startup-adjusted-schedule' for a in n.value.args))
schedule_assign=next(n for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='schedule_elapsed' for t in n.targets))
check('no new clock read in helper',not any(isinstance(n,ast.Attribute) and n.attr in ('monotonic','time','perf_counter') for n in ast.walk(functions['startup_adjusted_schedule_elapsed'])))
class Undo(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name=='startup_adjusted_schedule_elapsed':return None
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if dump(node)==dump(parser_stmt):return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if dump(node)==dump(schedule_assign):return None
        return self.generic_visit(node)
    def visit_Name(self,node):
        if node.id=='schedule_elapsed':node.id='elapsed'
        return node
restored=Undo().visit(copy.deepcopy(new))
check('entire candidate AST equals baseline after only intended edits',dump(restored)==dump(old))
scope={'argparse':argparse,'parser':argparse.ArgumentParser()}
exec(compile(ast.Module(body=[parser_stmt],type_ignores=[]),'exact-parser','exec'),scope)
for argv,expected in [([],False),(['--startup-adjusted-schedule'],True),(['--no-startup-adjusted-schedule'],False)]:
    check('flag '+repr(argv),scope['parser'].parse_args(argv).startup_adjusted_schedule is expected)
check('helper used at only two progress inputs',sum(isinstance(n,ast.Name) and n.id=='schedule_elapsed' for n in ast.walk(main))==3)
env={'TRAIN_STARTUP_STEPS_EXCLUDED':5,'TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS':900.,'STARTUP_LAUNCH_MARGIN_SECONDS':30.,'_PROCESS_STARTED':0.,'WARMUP_STEPS':0,'WARMDOWN_RATIO':.75,'FINAL_LR_FRAC':.05,'LRM_QUANTUM':0}
exec(compile(ast.Module(body=[functions['startup_adjusted_schedule_elapsed'],functions['lr_multiplier']],type_ignores=[]),'exact-clock-functions','exec'),env)
rows=[]
for origin in (0.,100000.):
    env['_PROCESS_STARTED']=origin
    for init_delay in (0.,.25,50.):
        budget_started=origin+init_delay
        for age in (50.,300.,844.,869.75,870.,870.25,899.75,900.,900.25,1200.,1800.,2700.):
            now=origin+age;elapsed=now-budget_started
            for step in range(8):
                # Independent physical-time definition, not the helper's reconstruction.
                expected=float(max(Fraction(0),Fraction(now)-Fraction(origin)+30-900)) if step<5 else elapsed
                actual=env['startup_adjusted_schedule_elapsed'](step,elapsed,budget_started)
                check('physical fakeclock '+str((origin,init_delay,age,step)),actual==expected)
                if step in (0,4,5):rows.append({'origin':origin,'initialization_seconds':init_delay,'process_age':age,'step':step,'elapsed':elapsed,'expected_schedule_elapsed':expected,'actual_schedule_elapsed':actual})
check('30-second margin onset',env['startup_adjusted_schedule_elapsed'](4,870.,100000.)==0. and env['startup_adjusted_schedule_elapsed'](4,871.,100000.)==1.)
# Exact unchanged reset AST, then next-step physical-time schedule reference.
boundary=next(n for n in ast.walk(main) if isinstance(n,ast.If) and ast.unparse(n.test)=='step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED')
oldboundary=next(n for n in ast.walk(oldmain) if isinstance(n,ast.If) and ast.unparse(n.test)=='step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED')
check('fifth reset AST exact',dump(boundary)==dump(oldboundary))
reset=compile(ast.Module(body=[boundary],type_ignores=[]),'exact-fifth-reset','exec')
transitions=[]
for age in (50.,844.,869.75,870.,870.25,900.,1543.3):
    env.update(step=4,_PROCESS_STARTED=100000.,time=types.SimpleNamespace(monotonic=lambda age=age:100000.+age),recent_step_times=[844.,600.,.5],budget_started=100050.)
    exec(reset,env)
    sampled_age=(100000.+age)-100000.
    excess=max(0.,sampled_age+30.-900.)
    after=(100000.+age+.5)-env['budget_started']
    check('reset excess preserved '+str(age),abs(after-(excess+.5))<1e-10 and env['recent_step_times']==[])
    check('step5 identity '+str(age),env['startup_adjusted_schedule_elapsed'](5,after,env['budget_started'])==after)
    transitions.append({'fifth_process_age_input':age,'sampled_age':sampled_age,'charged_excess':excess,'next_step_elapsed':after,'schedule_elapsed':env['startup_adjusted_schedule_elapsed'](5,after,env['budget_started'])})
# Actual LR and WD source expressions share one helper result and OFF does not call it.
loop=next(n for n in ast.walk(main) if isinstance(n,ast.While))
assigns={n.targets[0].id:n for n in loop.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
progress_code=compile(ast.Module(body=[assigns[k] for k in ('schedule_elapsed','lrm','wd_progress')],type_ignores=[]),'exact-progress-block','exec')
helper=env['startup_adjusted_schedule_elapsed']
for enabled in (False,True):
    for step in (0,4,5,6):
        for elapsed in (0.,844.,900.,1500.):
            env.update(step=step,elapsed=elapsed,budget_started=100000.,_PROCESS_STARTED=100000.,args=types.SimpleNamespace(startup_adjusted_schedule=enabled,max_train_seconds=1800.))
            if not enabled:
                def forbidden(*args):raise RuntimeError('OFF called helper')
                env['startup_adjusted_schedule_elapsed']=forbidden
            else:env['startup_adjusted_schedule_elapsed']=helper
            exec(progress_code,env)
            expected=elapsed if not enabled or step>=5 else max(0.,elapsed+30.-900.)
            check('shared scalar input '+str((enabled,step,elapsed)),env['schedule_elapsed']==expected and env['lrm']==env['lr_multiplier'](step,elapsed=expected,max_train_seconds=1800.) and env['wd_progress']==min(1.,expected/1800.))
result={'status':'PASS_STATIC_AND_FAKE_CLOCK_ONLY','candidate_commit':CANDIDATE,'baseline_commit':BASE,'source_sha256':sha(newraw),'checks':len(checks),'physical_clock_cases':576,'fifth_reset_cases':len(transitions),'progress_cases':32,'full_source_ast_restored':True,'raw_tiny_gate_evidence_pending':True,'fresh_review_pending':True,'native_acceptance':False,'quality_effect':'UNKNOWN','no_tensors':True,'no_hosts':True,'clock_rows':rows,'fifth_transitions':transitions,'checks_passed':checks}
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
if args.output:args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('checks_passed','clock_rows','fifth_transitions')},indent=2))
