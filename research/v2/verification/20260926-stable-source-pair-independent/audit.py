"""Independent stdlib-only frozen-binding and refusal fixtures; never launches a job."""
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
COMMIT='3edbc186353891983ed8d02005fa16d4583830c6'
PACKET='research/v2/proposals/20260926-architecture-stable-source-pair/'
checks=[]
def ok(name,value):
    if not value:raise RuntimeError(name)
    checks.append(name)
def refused(name,fn,match):
    try:fn()
    except Exception as exc:ok(name,match in str(exc));return
    raise RuntimeError('accepted negative: '+name)
def git(path,commit=COMMIT):return subprocess.check_output(['git','show',commit+':'+path],cwd=REPO)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def data(name):return git(PACKET+name)
def parsed(name):return json.loads(data(name))
plan=parsed('PAIR_BINDING.disabled.json')
manifest=parsed('MANIFEST.sha256.json')
driver=data('host_driver.py').decode()
helper=data('identity_checks.py').decode()
ns={};exec(compile(helper,'frozen-identity_checks.py','exec'),ns)
M=types.SimpleNamespace(**ns)
tree=ast.parse(driver)
driver_try=next(n for n in tree.body if isinstance(n,ast.Try))
body=compile(ast.Module(body=[driver_try],type_ignores=[]),'frozen-driver-try','exec')
guard_raw=git('research/v2/operator/20260926-common-host-idle-guard/idle_guard.py')
gns={};exec(compile(guard_raw,'frozen-idle-guard','exec'),gns)

for path,digest in manifest['files'].items():ok('manifest '+path,sha(data(path))==digest)
ok('manifest pin',sha(data('MANIFEST.sha256.json'))=='623406e0303c1f619f4415e2e33cb8478a6ab2fe337c769c65e48b36b9d9eea6')
ok('driver/helper pins',sha(data('host_driver.py'))=='25d154da94f03751ad8c962d750d05eeec90219c739acf6d1c4740d087bd49b6' and sha(data('identity_checks.py'))==plan['helper_sha256'])
pin=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PLAN_SHA' for t in n.targets))
ok('compiled plan pin',pin==sha(data('PAIR_BINDING.disabled.json'))=='518c039d570d02b5ac34a4649b866465e18df834061a992949841ca0b8981cca')
ok('source git bytes',sha(git(plan['source_git_path'],plan['source_commit']))==plan['source_sha256']=='3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c')
obj=plan['harness_git_object'].split(':',1)
harness=git(obj[1],obj[0])
ok('harness immutable',sha(harness)==plan['harness_sha256'])
ok('prepare immutable',sha(git('prepare.py'))==plan['prepare_sha256'])
ok('guard immutable',sha(guard_raw)==plan['idle_guard_sha256'])
ok('dispatch immutable',sha(git('research/v2/operator/20260926-common-host-idle-guard/dispatch_driver.py'))==plan['dispatch_driver_sha256'])
old=json.loads(git(plan['source_binding_parent']))
for phase in plan['phases']:
    args=phase['training_command'].copy();args[3]='train.py'
    ok(phase['name']+' science unchanged',args==old['training_command'])
    launch=data(phase['name'].lower()+'/package/launch-command.txt')
    provenance=parsed(phase['name'].lower()+'/package/provenance.json')
    ok(phase['name']+' exact launch',sha(launch)==phase['launch_sha256'] and shlex.split(launch.decode())==phase['training_command'] and provenance['launch_command']==launch.decode().strip() and provenance['train_sha256']==plan['source_sha256'])
    ok(phase['name']+' native settings',phase['environment']['NEURON_LOGICAL_NC_CONFIG']=='2' and phase['environment']['NEURON_CC_FLAGS']==old['training_environment']['NEURON_CC_FLAGS'] and phase['environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS']=='1800')
    ok(phase['name']+' exact shared caches',all(phase['environment'][k]==v for k,v in plan['cache_environment'].items()) and phase['environment']['NEURON_COMPILE_CACHE_URL']==plan['cache_environment']['NEURON_COMPILE_CACHE_DIR'])
    ok(phase['name']+' timeout and no eval',phase['command'][:5]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','1800s'] and phase['command'][phase['command'].index('--mode')+1]=='train')
    ok(phase['name']+' clean import env',not {'PYTHONPATH','PYTHONHOME','PYTHONOPTIMIZE'}&phase['environment'].keys())
ok('phase order/fresh distinction',[p['name'] for p in plan['phases']]==['S1','S2'] and '--fresh-caches' in plan['phases'][0]['command'] and '--fresh-caches' not in plan['phases'][1]['command'])
ok('exact original outer lock',plan['outer_command']==['/usr/bin/flock','-n','/home/ubuntu/.cache/trainium_freshstart/device0.lock','/usr/bin/python3','-I','-B',plan['root']+'/host_driver.py'])
ok('disabled/no broad claims',all(plan[k] is False for k in ('stage_authorized','launch_authorized','host_reserved','source_adoption','submission_authorized')) and plan['source_identity']['actual_import_origin_observed'] is False and all(plan[k] is True for k in ('no_retry','no_fallback','no_eval','no_automatic_full','no_quality_or_adoption')))

# Own receipt/filesystem fixtures, not the Implementer's tests. No model bytes.
with tempfile.TemporaryDirectory(prefix='pair-review-receipts-') as d:
    root=Path(d);stable=root/'stable';stable.mkdir();sandbox=root/'sandbox';sandbox.mkdir();(sandbox/'out').mkdir()
    package=root/'package';package.mkdir()
    source=b'fixture_source=1\n';prep=b'raise RuntimeError("must never execute prepare")\n';ckpt=b'fixture, not a model checkpoint'
    for directory in (stable,sandbox,package):
        (directory/'train.py').write_bytes(source);(directory/'train.py').chmod(0o444)
    for directory in (stable,sandbox):(directory/'prepare.py').write_bytes(prep)
    (stable/'prepare.py').chmod(0o444);stable.chmod(0o555)
    protected=root/'protected.py';protected.write_bytes(prep)
    harness_file=root/'harness.py';harness_file.write_bytes(b'fixture harness')
    checkpoint=sandbox/'out/final.pt';checkpoint.write_bytes(ckpt)
    phase=copy.deepcopy(plan['phases'][0]);phase.update(root=str(root),sandbox=str(sandbox),package=str(package))
    p=copy.deepcopy(plan);p.update(stable_directory=str(stable),source_sha256=sha(source),prepare_sha256=sha(prep),prepare_path=str(protected),harness_path=str(harness_file),harness_sha256=sha(harness_file.read_bytes()))
    phase['training_command'][3]=str(stable/'train.py')
    launch=shlex.join(phase['training_command'])+'\n';provenance=json.dumps({'train_sha256':sha(source),'launch_command':launch.strip()})
    for name,text in [('launch-command.txt',launch),('provenance.json',provenance)]:
        (package/name).write_text(text);(package/name).chmod(0o444)
    phase.update(launch_sha256=sha(launch.encode()),provenance_sha256=sha(provenance.encode()))
    identity=M.inputs(p,phase);ok('immutable source identity stable/no import',identity==M.inputs(p,phase) and identity['expected_imported_prepare']['sha256']==sha(prep))
    changed=copy.deepcopy(phase);changed['environment']['PYTHONHOME']='bad'
    refused('import override',lambda:M.inputs(p,changed),'IMPORT_PATH_OVERRIDE')
    (stable/'train.py').chmod(0o644)
    refused('writable source',lambda:M.inputs(p,phase),'WRITABLE_IMMUTABLE_FILE');(stable/'train.py').chmod(0o444)
    alias=root/'alias';alias.symlink_to(stable,target_is_directory=True)
    refused('symlink ancestor',lambda:M.identity(alias/'train.py'),'SYMLINK')
    os.link(stable/'train.py',root/'hardlink')
    refused('hardlink source',lambda:M.inputs(p,phase),'HARDLINK');(root/'hardlink').unlink()
    stable.chmod(0o755);replacement=stable/'replace';replacement.write_bytes(source);replacement.chmod(0o444);os.replace(replacement,stable/'train.py');stable.chmod(0o555)
    ok('same-byte replacement differs',identity!=M.inputs(p,phase))
    log=root/'training.log';goodlog=''.join(f'step {i:05} | loss {5-i/10:.4f} | lrm 1.000 | dt 0.50s |\n' for i in range(5));log.write_text(goodlog)
    def receipt(fifth=900.,wall=901.,name='S1'):
        return {'returncode':0,'mode':'train','step_boundaries':5,'observations':[{'step':i,'observed_seconds':fifth-4+i} for i in range(5)],'wall_seconds':wall,'startup_allowance_observed':min(fifth,900.),'charged_through_process_exit':wall-min(fifth,900.),'charged_through_process_exit_within_budget':True,'fresh_empty_sdk_caches':p['cache_environment'] if name=='S1' else {},'checkpoint_sha256':sha(ckpt),'checkpoint_bytes':len(ckpt)}
    def validate(r,name='S1'):
        phase['name']=name;(sandbox/'rehearsal.json').write_text(json.dumps(r));return M.validate_phase(p,phase)
    ok('S1 inclusive900',validate(receipt())['fifth_boundary_seconds']==900)
    refused('S1 just over900',lambda:validate(receipt(900.000001,901.)),'S1_COLD_FIFTH_OVER_900')
    ok('S2 inclusive charged1800',validate(receipt(900.,2700.,'S2'),'S2')['charged_through_exit_seconds']==1800.)
    refused('S2 over charged1800',lambda:validate(receipt(900.,2700.000001,'S2'),'S2'),'SOURCE_BUDGET')
    ok('S2 startup cap applied',validate(receipt(901.,2700.,'S2'),'S2')['charged_through_exit_seconds']==1800.)
    for field,value,error in [('returncode',False,'HARNESS_FAILED'),('step_boundaries',5.,'FIVE_BOUNDARIES'),('charged_through_process_exit',2.,'CHARGED_MISMATCH'),('startup_allowance_observed',901.,'ALLOWANCE_MISMATCH'),('charged_through_process_exit_within_budget',False,'SOURCE_BUDGET'),('wall_seconds',float('nan'),'INVALID_EXIT_WALL'),('checkpoint_sha256','0'*64,'SHA_MISMATCH'),('checkpoint_bytes',0,'CHECKPOINT_SIZE'),('fresh_empty_sdk_caches',{},'S1_FRESH_CACHE')]:
        bad=receipt();bad[field]=value;refused('receipt '+field,lambda bad=bad:validate(bad),error)
    bad=receipt();bad['observations'][1]['step']=0
    refused('unordered steps',lambda:validate(bad),'STEP_SEQUENCE')
    bad=receipt();bad['observations'][0]['observed_seconds']=float('inf')
    refused('nonfinite boundary',lambda:validate(bad),'INVALID_BOUNDARY_TIME')
    log.write_text(goodlog.replace('loss 5.0000','loss nan'))
    refused('nonfinite loss',lambda:validate(receipt()),'NONFINITE_LOGGED_VALUE');log.write_text(goodlog)
    # Exact-byte graph fixture; no compiler or native interpretation.
    metrics=root/'metrics';metrics.mkdir();phase['environment']['TORCH_NEURONX_METRICS_DIR']=str(metrics)
    debug=root/'debug/neuronx-cc';debug.mkdir(parents=True);p['cache_environment']['TORCH_NEURONX_DEBUG_DIR']=str(root/'debug')
    rows=[]
    for name in ('forward','backward'):
        folder=debug/name;folder.mkdir();raw=('exact binary fixture '+name).encode();(folder/'module.mlir').write_bytes(raw)
        rows.append(sha(raw)+':'+'a'*64+',model_default_0_'+name+'_1,model_default\n')
    for rank in range(4):(metrics/f'metric_table_graph_stats_rank{rank}.csv').write_text('graph_id,graph_name,model_name\n'+''.join(rows))
    graph=M.capture_inputs(p,phase)
    ok('all rank exact input fixture',graph['exact_input_bytes_complete'] and len(graph['graphs'])==4 and len(graph['principal_keys'])==2 and graph['native_neff_reuse']=='UNPROVEN' and graph['cache_hit'] is False)
    for key in graph['principal_keys']:ok('copied exact compiler fixture '+key,sha((root/'compiler-evidence'/(key+'.mlir')).read_bytes())==key)
    phase['root']=str(root/'second');Path(phase['root']).mkdir();(debug/'forward/module.mlir').write_bytes(b'wrong')
    graph=M.capture_inputs(p,phase);ok('missing exact compiler bytes refuses completeness',graph['exact_input_bytes_complete'] is False and len(graph['missing_keys'])==1)
    stable.chmod(0o755)

# Execute the exact outer try body with explicit I/O/child/guard stubs.
# No actual subprocess, host, compiler or tensor can be dispatched by these fixtures.
def drive(fault=None):
    with tempfile.TemporaryDirectory(prefix='pair-review-driver-') as d:
        root=Path(d);p=copy.deepcopy(plan);p['root']=str(root);p['stable_directory']=str(root/'stable');(root/'stable').mkdir()
        for i,phase in enumerate(p['phases']):
            r=root/phase['name'];r.mkdir();metrics=r/'metrics';metrics.mkdir()
            phase.update(root=str(r),sandbox=str(r/'absent-sandbox'))
            phase['environment']['TORCH_NEURONX_METRICS_DIR']=str(metrics)
        for key in p['cache_environment']:
            directory=root/key;directory.mkdir();p['cache_environment'][key]=str(directory)
        raw=json.dumps(p).encode();(root/'PAIR_BINDING.disabled.json').write_bytes(raw)
        records={};calls=[];timeline=[];counter=[0];inputs_count=[0]
        def record(name,value):
            if name in records:raise RuntimeError('duplicate record')
            records[name]=value
        def guard(*args):
            counter[0]+=1;timeline.append('guard'+str(counter[0]))
            if fault=='guard'+str(counter[0]):raise RuntimeError('GUARD_FIXTURE_BUSY_OR_ERROR')
            return 8 if fault=='changed-fd' and counter[0]==2 else 7
        def inputs(*args):
            inputs_count[0]+=1
            result={k:1 for k in ('executed_train','expected_imported_prepare','stable_directory','protected_prepare','harness')}
            if fault=='source-changed' and inputs_count[0]==2:result['executed_train']=2
            return result
        def validate(*args):
            timeline.append('validate'+args[1]['name'])
            if fault=='S1-acceptance' and args[1]['name']=='S1':raise RuntimeError('S1_COLD_FIFTH_OVER_900')
            return {'status':'FIVE_STEP_MECHANICAL_GATES_PASS','first_five_logged':[]}
        def capture(*args):
            timeline.append('capture'+args[1]['name']);return {'exact_input_bytes_complete':fault!='missing-inputs'}
        mockchecks=types.SimpleNamespace(no_symlink=lambda p:Path(p),inputs=inputs,validate_phase=validate,capture_inputs=capture,compare_losses=lambda *args:{})
        def load(name,path):return types.SimpleNamespace(require_idle_under_inherited_lock=guard) if 'idle_guard' in name else mockchecks
        def digest(path):
            if path.name=='PAIR_BINDING.disabled.json':return sha(path.read_bytes())
            return p[{'idle_guard.py':'idle_guard_sha256','dispatch_driver.py':'dispatch_driver_sha256','identity_checks.py':'helper_sha256'}[path.name]]
        def call(command,**kwargs):
            ok('child inherited original fd '+str(len(calls)),kwargs['pass_fds']==(7,) and kwargs['stdin']==subprocess.DEVNULL)
            calls.append(command);timeline.append('child'+str(len(calls)));return 3 if fault=='exit' else 0
        env={'ROOT':root,'PLAN_SHA':sha(raw),'__file__':str(root/'host_driver.py'),'Path':Path,'json':json,'os':os,'require':M.require,'sha':digest,'load':load,'record':record,'utc':lambda:'fixture','resources':lambda p:{},'subprocess':types.SimpleNamespace(call=call,DEVNULL=subprocess.DEVNULL,STDOUT=subprocess.STDOUT)}
        error=None
        try:exec(body,env)
        except Exception as exc:error=str(exc)
        return {'children':len(calls),'error':error,'timeline':timeline,'terminal':records.get('terminal.json')}

good=drive();ok('exact two synchronous phases',good['children']==2 and good['error'] is None and good['timeline']==['guard1','guard2','child1','guard3','validateS1','captureS1','guard4','child2','guard5','validateS2','captureS2'] and good['terminal']['native_cache_reuse']=='UNPROVEN')
driver_failures={}
for fault,count in [('guard1',0),('guard2',0),('changed-fd',0),('guard3',1),('guard4',1),('guard5',2),('exit',1),('source-changed',1),('S1-acceptance',1),('missing-inputs',1)]:
    result=drive(fault);driver_failures[fault]=result
    ok('fail closed '+fault,result['children']==count and result['error'] and result['terminal']['status']=='PAIR_REFUSED_OR_INCOMPLETE')

# Actual unchanged ps/neuron parser with independently constructed stdout fixtures.
device={'neuron_device':0,'instance_type':'trn2.3xlarge','nc_count':4,'logical_neuroncore_config':2,'neuroncore_ids':[0,1,2,3],'neuron_processes':[]}
def idle(ps_extra='',neuron=None,ps_rc=0,nl_stderr=''):
    commands=[];evidence={}
    def run(command,**kwargs):
        commands.append(command)
        if command==gns['PS']:return types.SimpleNamespace(returncode=ps_rc,stdout='101 100 python3 /packet/host_driver.py\n100 1 flock /usr/bin/flock -n lock driver\n'+ps_extra,stderr='')
        return types.SimpleNamespace(returncode=0,stdout=json.dumps([device]) if neuron is None else neuron,stderr=nl_stderr)
    try:gns['check_idle'](run,101,100,evidence)
    except Exception as exc:return str(exc),commands
    return None,commands
ok('guard both fresh checks',idle()[0] is None and idle()[1]==[gns['PS'],gns['NEURON_LS']])
busy=idle('102 1 python3 /else/host_driver.py\n');ok('other driver not excluded',busy[0]=='PS_NATIVE_JOB_BUSY' and len(busy[1])==2)
busy=idle('102 1 python3 /venv/bin/torchrun --standalone train.py\n');ok('torchrun refuses',busy[0]=='PS_NATIVE_JOB_BUSY' and len(busy[1])==2)
ok('malformed neuron refuses',idle(neuron='{}')[0]=='NEURON_LS_DEVICE_SCHEMA_UNKNOWN')
ok('ps error refuses',idle(ps_rc=1)[0]=='PS_COMMAND_FAILED')
ok('neuron stderr refuses',idle(nl_stderr='warning')[0]=='NEURON_LS_COMMAND_FAILED')
with tempfile.TemporaryDirectory(prefix='pair-direct-lock-') as d:
    lock=Path(d)/'device0.lock';lock.touch()
    refused('direct invocation lacks authenticated outer flock',lambda:gns['inherited_lock'](lock,Path(d)/'host_driver.py'),'OUTER_FLOCK_COMMAND_MISMATCH')

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
result={'verdict':'PASS_DISABLED_PAIR_MECHANICS_ONLY','checks':len(checks),'candidate_commit':COMMIT,'plan_sha256':sha(data('PAIR_BINDING.disabled.json')),'driver_sha256':sha(data('host_driver.py')),'helper_sha256':sha(data('identity_checks.py')),'driver_faults':driver_failures,'no_hosts':True,'no_tensors':True,'no_training':True,'stage_authorized':False,'launch_authorized':False,'reuse_proven':False,'actual_import_origin_observed':False,'causal_path_only_proven':False,'checks_passed':checks}
encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
if args.output:args.output.write_text(encoded)
print(json.dumps({k:v for k,v in result.items() if k not in ('driver_faults','checks_passed')},indent=2))
