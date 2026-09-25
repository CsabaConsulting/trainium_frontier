"""Independent frozen K2 binding/receipt/refusal checks; stdlib, no hardware."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import types

REPO=Path(__file__).resolve().parents[4]
COMMIT='0abd35a071508679e4d184ca2aeb3ab1295cba3f'
PACKET='research/v2/proposals/20260926-k2-name-binding-native/'
OLD='research/v2/proposals/20260926-k2-reporting-native-binding/'
OLD_COMMIT='df48361a619baa65fe9864cd815a1cd864d7a789'
checks=[]
def ok(name,value):
    if not value:raise RuntimeError(name)
    checks.append(name)
def refusal(name,fn):
    try:fn()
    except (RuntimeError,KeyError,FileNotFoundError,ValueError,TypeError):checks.append(name);return
    raise RuntimeError('accepted bad fixture '+name)
def git(path,commit=COMMIT):return subprocess.check_output(['git','show',commit+':'+path],cwd=REPO)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def data(path):return git(PACKET+path)
P=json.loads(data('NATIVE_BINDING.disabled.json'));Q=json.loads(git(OLD+'NATIVE_BINDING.disabled.json',OLD_COMMIT))
manifest=json.loads(data('MANIFEST.sha256.json'))
for name,pin in manifest['files'].items():ok('manifest '+name,sha(data(name))==pin)
ok('exact manifest',sha(data('MANIFEST.sha256.json'))=='3991cd06963355513f49693e7fb432eb32c09d0d540a4e44459c7503faef9458')
ok('exact plan',sha(data('NATIVE_BINDING.disabled.json'))=='127fbb6a9b4339a3331da76ab70b484cdb92c3573735703a99a4bd91b3a05f72')
ok('exact driver',sha(data('host_driver.py'))==P['driver_sha256']=='438139440fca22e55dc097cd59d301d6f69d6e5794705dc3c8015eb31ffaa2ba')
def mapped(raw):
    for old,new in [(Q['label'],P['label']),(Q['support_sha256']['consumer_nki.py'],P['support_sha256']['consumer_nki.py']),(Q['support_sha256']['check_native.py'],P['support_sha256']['check_native.py']),(Q['support_sha256']['rank_entry.py'],P['support_sha256']['rank_entry.py'])]:raw=raw.replace(old,new)
    return raw
for name in ('host_driver.py','rank_entry.py','bound-command.txt'):
    ok('constant-only rebind '+name,mapped(git(OLD+name,OLD_COMMIT).decode())==data(name).decode())
ok('validator unchanged',git(OLD+'validate_receipts.py',OLD_COMMIT)==data('validate_receipts.py'))
old_files=subprocess.check_output(['git','ls-tree','-r','--name-only',OLD_COMMIT,'--',OLD],cwd=REPO,text=True).splitlines()
for file in old_files:ok('old packet preserved '+file,git(file)==git(file,OLD_COMMIT))
for field in ('command','checker_args','environment','dispatch_command','outer_command','expected_hostname','cache_keys','timeout_seconds','term_grace_seconds'):
    ok('unchanged bound '+field,json.loads(mapped(json.dumps(Q[field])))==P[field])
source={}
for name,pin in P['support_sha256'].items():
    if name in ('consumer_nki.py','check_native.py','reference.py'):
        raw=git(P['source_module_path']+'/'+name,P['source_module_commit']);source[name]=raw
        if name!='consumer_nki.py':ok('unchanged checker/reference '+name,raw==git(Q['source_module_path']+'/'+name,Q['source_module_commit']))
    elif name in ('idle_guard.py','dispatch_driver.py'):raw=git('research/v2/operator/20260926-common-host-idle-guard/'+name)
    else:raw=data(name)
    ok('support '+name,sha(raw)==pin)
ok('source exact namespace successor',P['source_module_commit']=='d0f0d7fbdbe07ddee39680ee1a2b79fdfac388ca' and sha(source['check_native.py'])=='89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850')
ok('original REFUSE preserved',P['rebind']['original_source_verdict']=='NATIVE_FRONTEND_REFUSE_PRESERVED_94fb52c4')

proposal=json.loads(data('EXPERIMENT_PROPOSAL.json'))
ok('proposal exact source/label',proposal['candidate_commit']==P['source_module_commit'] and proposal['label']==P['label'])
ok('proposal disabled',proposal['stage_authorized'] is proposal['launch_authorized'] is False and proposal['operator_assignment']=='NONE')
ok('only one diagnosed retry still disabled',P['retries']==P['retry_classification']['automatic_retries']==0 and P['retry_classification']['authorization'] is False and P['retry_classification']['one_diagnosed_retry_if_separately_assigned'] is True)
ok('explicit postflight required outside unchanged driver',P['postflight']==proposal['postflight'] and P['postflight']['required'] is True and P['postflight']['artifact']=='postflight-explicit-pids.json' and all(x in P['postflight']['method'] for x in ('PPID/PGID/SID','unfiltered','ps AND neuron-ls')))
ok('no new driver postflight claimed','postflight-explicit-pids.json' not in data('host_driver.py').decode())
ok('source pin only not source approval',P['support_sha256']['consumer_nki.py']=='4939265bf73b3d5f42276e72e48f022df338226a40a9be5b79ed63bde6a4f5b8' and 'PENDING' in P['rebind']['successor_review'])

ok('disabled scope',all(P[k] is False for k in ('stage_authorized','launch_authorized','operator_assigned','host_reserved','source_adoption','automatic_full','automatic_eval','automatic_next_job')))
ok('single bounded native command',P['command'][:8]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','600s','/home/ubuntu/workspace/native_venv/bin/torchrun','--standalone','--nproc_per_node=4'] and P['command'][8]==P['root']+'/rank_entry.py' and P['command'][9:]==P['checker_args'] and shlex.split(data('bound-command.txt').decode())==P['command'])
ok('original common lock',P['outer_command']==['/usr/bin/flock','-n','/home/ubuntu/.cache/trainium_freshstart/device0.lock','/usr/bin/python3','-I','-B',P['root']+'/host_driver.py'])
ok('private distinct cache roots',len(set(P['environment'][k] for k in P['cache_keys']))==6 and all(P['environment'][k].startswith(P['root']+'/caches/') for k in P['cache_keys']) and P['environment']['NEURON_COMPILE_CACHE_URL']==P['environment']['NEURON_COMPILE_CACHE_DIR'])
ok('LNC2/seed42/threads1',P['environment']['NEURON_LOGICAL_NC_CONFIG']=='2' and P['environment']['NEURON_COMPETITION_R1_SEED']=='42' and all(P['environment'][k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')))
ok('no import or optimize override',not {'PYTHONPATH','PYTHONHOME','PYTHONOPTIMIZE'}&P['environment'].keys())

vns={};exec(compile(data('validate_receipts.py'),'exact-validator','exec'),vns)
# Own literal expected case inventory, independently checked against frozen checker.
cases=('cross_tile_large','cross_tile_unit','unique','all_same','counts_253_1_1_1','boundary_rows','signed_zero','bf16_subnormal','fp32_subnormal','empty')
dtypes=('torch.bfloat16','torch.float32')
common=('upstream_roundtrip','direct_consumer','direct_repeated_call')
grad=('dense_CPU_oracle','eager_autograd_forward','eager_autograd_backward','compiled_enclosing_autograd_forward','compiled_enclosing_autograd_backward')
checker_tree=ast.parse(source['check_native.py'])
literal_strings={n.value for n in ast.walk(checker_tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
derived_names={mode+suffix for mode in ('eager_autograd','compiled_enclosing_autograd') for suffix in ('_forward','_backward')}
ok('checker generated mode/suffix labels',{'eager_autograd','compiled_enclosing_autograd','_forward','_backward'}<=literal_strings)
for name in common+grad:ok('prescribed checker label '+name,name in literal_strings or name in derived_names)
def files():
    result={}
    for rank in range(4):
        inventory=[]
        for case in cases:
            for dtype in dtypes:
                comparisons=[]
                for name in common+(grad if case!='empty' else ()):
                    comparisons.append({'name':name,'exact':True,'finite':True,'shape':[256,128],'actual_dtype':dtype if name in ('upstream_roundtrip','eager_autograd_forward','compiled_enclosing_autograd_forward') else 'torch.float32','actual_sha256':'a'*64,'expected_sha256':'a'*64,'max_abs':0.})
                inventory.append({'name':case,'dtype':dtype,'status':'EXACT','direct_first_call_seconds':.2,'checks':comparisons})
        result[f'receipts/rank{rank}.json']={'scope':'TINY_CONSUMER_FALSIFIER_ONLY','status':'TINY_CONSUMER_EXACT_ONLY','rank':rank,'world_size':4,'label':P['label'],'shape':{'V':256,'C':128,'T':256},'seed':42,'argv':P['checker_args'],'kernel_sha256':P['support_sha256']['consumer_nki.py'],'checker_sha256':P['support_sha256']['check_native.py'],'reference_sha256':P['support_sha256']['reference.py'],'tolerance':{'atol':0,'rtol':0,'criterion':'identical bytes'},'accepted_speedup':False,'whole_step_speedup':'UNMEASURED','diagnostic_elapsed_seconds':1.,'cases':inventory,'timings':[{'case':name,'dtype':dtype,'seconds':[.1,.2,.1],'scope':'consumer_only_excludes_routing_and_transfer'} for name in ('unique','all_same') for dtype in dtypes]}
        result[f'rank-environment-r{rank}.json']={'rank':rank,'local_rank':rank,'world_size':4,'environment':P['environment'],'checker_sha256':P['support_sha256']['check_native.py'],'checker_argv':P['checker_args']}
        result[f'rank-wrapper-exit-r{rank}.json']={'rank':rank,'returncode':0,'error':None}
    return copy.deepcopy(result)
def validate(fixture):
    class FakePath:
        def __init__(self,path):self.path=str(path)
        def __truediv__(self,suffix):return FakePath(self.path+'/'+str(suffix))
        def read_text(self):return json.dumps(fixture[self.path.removeprefix('/fixture/')])
    old=vns['Path'];vns['Path']=FakePath
    try:return vns['validate']('/fixture',P['label'],P['environment'],P['support_sha256'],P['checker_args'])
    finally:vns['Path']=old
good=files();complete=validate(good)
ok('all four 20/150/12 completion',[r['complete_cases'] for r in complete['ranks']]==[20]*4 and [r['exact_comparisons'] for r in complete['ranks']]==[150]*4 and [r['timing_samples'] for r in complete['ranks']]==[12]*4 and complete['accepted_speedup'] is complete['source_adoption'] is complete['full_run_authorized'] is False)
key='receipts/rank2.json'
mutations=[('missing rank',lambda f:f.pop('receipts/rank3.json')),('wrapper fail',lambda f:f['rank-wrapper-exit-r1.json'].update(returncode=1)),('wrapper bool',lambda f:f['rank-wrapper-exit-r1.json'].update(returncode=False)),('wrong label',lambda f:f[key].update(label='wrong')),('old kernel',lambda f:f[key].update(kernel_sha256=Q['support_sha256']['consumer_nki.py'])),('REFUSE',lambda f:f[key].update(status='REFUSE_OR_UNSUPPORTED')),('claimed speedup',lambda f:f[key].update(accepted_speedup=True)),('loose atol',lambda f:f[key]['tolerance'].update(atol=.01)),('wrong shape',lambda f:f[key].update(shape={'V':256,'C':128,'T':255})),('missing case',lambda f:f[key]['cases'].pop()),('duplicate case',lambda f:f[key]['cases'].__setitem__(1,copy.deepcopy(f[key]['cases'][0]))),('missing gradient',lambda f:f[key]['cases'][0]['checks'].pop()),('incorrect',lambda f:f[key]['cases'][0]['checks'][0].update(exact=False)),('nonfinite',lambda f:f[key]['cases'][0]['checks'][0].update(finite=False)),('nonzero error',lambda f:f[key]['cases'][0]['checks'][0].update(max_abs=1e-45)),('different bytes',lambda f:f[key]['cases'][0]['checks'][0].update(expected_sha256='b'*64)),('null nan metric',lambda f:f[key]['cases'][0]['checks'][0].update(max_abs=None,max_abs_nonfinite='nan')),('signedzero failure',lambda f:f[key]['cases'][0]['checks'][0].update(exact=False,max_abs=0.)),('nan elapsed',lambda f:f[key].update(diagnostic_elapsed_seconds=float('nan'))),('nan timing',lambda f:f[key]['timings'][0].update(seconds=[.1,float('nan'),.1])),('missing sample',lambda f:f[key]['timings'][0].update(seconds=[.1,.1])),('wrong rank environment',lambda f:f['rank-environment-r1.json']['environment'].update(NEURON_LOGICAL_NC_CONFIG='1'))]
for name,change in mutations:
    fixture=copy.deepcopy(good);change(fixture);refusal('receipt '+name,lambda fixture=fixture:validate(fixture))

# Actual driver definitions, with only host/filesystem/process dependencies stubbed.
dns={'__name__':'not_main','__file__':P['root']+'/host_driver.py'};exec(compile(data('host_driver.py'),'exact-driver','exec'),dns)
for name,key in [('LABEL','label'),('COMMAND','command'),('CHECKER_ARGS','checker_args'),('ENV','environment'),('PINS','support_sha256'),('CACHE_KEYS','cache_keys')]:ok('driver '+name,dns[name]==P[key])
def drive(fault):
    n=dict(dns)
    # Rebind main globals by executing the function definition in the isolated fixture namespace.
    with tempfile.TemporaryDirectory(prefix='k2-driver-review-') as d:
        root=Path(d);env=copy.deepcopy(P['environment'])
        for k in P['cache_keys']:env[k]=str(root/'caches'/Path(env[k]).name)
        env['NEURON_COMPILE_CACHE_URL']=env['NEURON_COMPILE_CACHE_DIR'];env['TORCH_NEURONX_METRICS_DIR']=str(root/'metrics')
        events=[];records={}
        def guard(*args):
            events.append('guard')
            if fault=='guard':raise RuntimeError('BUSY_OR_UNKNOWN')
            return 11
        def child(command,**kwargs):
            events.append('child');ok('fd/command '+fault,kwargs['pass_fds']==(11,) and command==P['command'] and kwargs['env']==env)
            return 124 if fault=='timeout' else 0
        def completion(*args):
            events.append('validation')
            if fault=='missing-receipt':raise RuntimeError('MISSING_RANK')
            return complete
        def record(name,value):
            if name in records:raise RuntimeError('duplicate record')
            records[name]=value
        n.update(ROOT=root,PROJECT=root,__file__=str(root/'host_driver.py'),ENV=env,record=record,sha=lambda p:'wrong' if fault=='hash' else P['support_sha256'][p.name],socket=types.SimpleNamespace(gethostname=lambda:'wrong' if fault=='host' else P['expected_hostname']),os=types.SimpleNamespace(path=types.SimpleNamespace(lexists=lambda p:os.path.lexists(p) or (fault=='stop' and p==root/'STOP')),statvfs=lambda p:types.SimpleNamespace(f_bavail=0 if fault=='disk' else 12*1024**3,f_frsize=1)),load_module=lambda name,p:types.SimpleNamespace(require_idle_under_inherited_lock=guard) if name=='bound_idle_guard' else types.SimpleNamespace(validate=completion),subprocess=types.SimpleNamespace(call=child,DEVNULL=subprocess.DEVNULL,STDOUT=subprocess.STDOUT))
        if fault=='existing-cache':(root/'caches').mkdir()
        main=next(x for x in ast.parse(data('host_driver.py')).body if isinstance(x,ast.FunctionDef) and x.name=='main')
        exec(compile(ast.Module(body=[main],type_ignores=[]),'exact-main','exec'),n)
        error=None
        try:rc=n['main']()
        except Exception as exc:rc=None;error=str(exc)
        return {'returncode':rc,'error':error,'events':events,'terminal':records['terminal.json']}
driver_results={}
for fault in ('success','host','stop','disk','hash','existing-cache','guard','timeout','missing-receipt'):
    result=drive(fault);driver_results[fault]=result
    if fault=='success':ok('driver complete only after validator',result['events']==['guard','child','validation'] and result['terminal']['complete'] is True and result['terminal']['accepted_speedup'] is False)
    else:ok('driver refuses '+fault,result['terminal']['complete'] is False and result['events'].count('child')==(1 if fault in ('timeout','missing-receipt') else 0))

# Exact wrapper main with fake runpy; no checker/native import.
ens={'__name__':'not_main','__file__':P['root']+'/rank_entry.py'};exec(compile(data('rank_entry.py'),'exact-rank-entry','exec'),ens)
ok('environment whitelist complete',set(ens['KEYS'])==set(P['environment']))
ok('rank checker pin',ens['CHECKER_SHA']==P['support_sha256']['check_native.py'])
wrapper_results={}
for scenario in ('success','systemexit1','exception','wrong-rank','wrong-checker'):
    with tempfile.TemporaryDirectory(prefix='k2-wrapper-review-') as d:
        root=Path(d);(root/'check_native.py').write_bytes(source['check_native.py'] if scenario!='wrong-checker' else b'bad')
        records={};calls=[];n=dict(ens);n.update(ROOT=root,__file__=str(root/'rank_entry.py'),record=lambda name,value:records.__setitem__(name,value),os=types.SimpleNamespace(environ={**P['environment'],'RANK':'2','LOCAL_RANK':'1' if scenario=='wrong-rank' else '2','WORLD_SIZE':'4'},getpid=lambda:123),sys=types.SimpleNamespace(argv=[str(root/'rank_entry.py'),*P['checker_args']]))
        def run(path,run_name):
            calls.append((path,run_name));ok('wrapper argv0 '+scenario,n['sys'].argv==[str(root/'check_native.py'),*P['checker_args']])
            if scenario=='systemexit1':raise SystemExit(1)
            if scenario=='exception':raise RuntimeError('injected checker error')
        n['runpy']=types.SimpleNamespace(run_path=run)
        main=next(x for x in ast.parse(data('rank_entry.py')).body if isinstance(x,ast.FunctionDef) and x.name=='main')
        exec(compile(ast.Module(body=[main],type_ignores=[]),'exact-entry-main','exec'),n)
        caught=None
        try:n['main']()
        except BaseException as exc:caught=type(exc).__name__
        wrapper_results[scenario]={'calls':len(calls),'caught':caught,'records':records}
        if scenario in ('wrong-rank','wrong-checker'):ok('wrapper pre-run refuse '+scenario,caught=='RuntimeError' and not calls)
        else:ok('wrapper terminal '+scenario,len(calls)==1 and records['rank-wrapper-exit-r2.json']['returncode']==(0 if scenario=='success' else 1))

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
result={'verdict':'PASS_DISABLED_K2_NAMESPACE_BINDING_MECHANICS_ONLY','candidate_commit':COMMIT,'binding_sha256':sha(data('NATIVE_BINDING.disabled.json')),'driver_sha256':sha(data('host_driver.py')),'checks':len(checks),'driver_fixture_results':driver_results,'wrapper_fixture_results':wrapper_results,'no_hosts':True,'no_tensors':True,'no_checker_execution':True,'launch_authorized':False,'native_correctness':'UNPROVEN','source_adoption':False,'checks_passed':checks}
if args.output:args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('checks_passed','driver_fixture_results','wrapper_fixture_results')},indent=2))
