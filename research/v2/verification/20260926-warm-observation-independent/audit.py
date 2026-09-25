"""Independent frozen W1 mechanics audit: stdlib, synthetic files/effects only."""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace

REF='52c31b73bbd5d76de35f32f55a66b6125aa8effb'
FIRST='ae77965ca71d44acb81f173a02af0717980c9a02'
PACK='research/v2/proposals/20260926-stablepath-warm-observation-trn22/'
REPO=Path(__file__).resolve().parents[4]
CHECKS=[]


def check(ok,name):
    if not ok: raise RuntimeError(name)
    CHECKS.append(name)


def blob(path,ref=REF):
    return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)


def digest(raw):return hashlib.sha256(raw).hexdigest()


PLAN=json.loads(blob(PACK+'WARM_BINDING.disabled.json'))
PROPOSAL=json.loads(blob(PACK+'EXPERIMENT_PROPOSAL.json'))
ANCHOR=json.loads(blob(PACK+'S1-pre-source-identity.json'))
DRIVER=ast.parse(blob(PACK+'host_driver.py'))
HELPER={}
exec(compile(blob(PACK+'identity_checks.py'),'<frozen-helper>','exec'),HELPER)
TARGET={}
exec(compile(ast.Module(body=[n for n in DRIVER.body if isinstance(n,ast.FunctionDef) and n.name=='classify_target'],type_ignores=[]),'<frozen-classifier>','exec'),TARGET)


def raises(call,name):
    try:call()
    except (RuntimeError,FileNotFoundError,ValueError,TypeError,KeyError):
        CHECKS.append(name)
    else:raise RuntimeError('Unexpected acceptance:'+name)


def static():
    manifest=json.loads(blob(PACK+'MANIFEST.sha256.json'))
    for path,h in manifest['files'].items():check(digest(blob(PACK+path))==h,'manifest:'+path)
    first=json.loads(blob(PACK+'EXPERIMENT_PROPOSAL.json',FIRST))
    first.pop('prepared_utc');now=copy.deepcopy(PROPOSAL);now.pop('prepared_utc')
    check(first==now,'proposal-first science unchanged')
    source=blob(PLAN['source_git_path'],PLAN['source_commit'])
    check(digest(source)==PLAN['source_sha256'],'exact3e79 source')
    check(digest(blob('prepare.py'))==PLAN['prepare_sha256'],'exact prepare')
    href,hpath=PLAN['harness_git_object'].split(':',1)
    harness=blob(hpath,href)
    check(digest(harness)==PLAN['harness_sha256'],'exact unedited harness')
    for file,key in [('identity_checks.py','helper_sha256'),('EXPERIMENT_PROPOSAL.json','proposal_sha256'),('S1-pre-source-identity.json','s1_identity_sha256')]:
        check(digest(blob(PACK+file))==PLAN[key],'driver support:'+file)
    for file,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
        check(digest(blob('research/v2/operator/20260926-common-host-idle-guard/'+file))==PLAN[key],'shared:'+file)
    parent='research/v2/proposals/20260926-architecture-stable-source-pair/'
    check(blob(PACK+'identity_checks.py')==blob(parent+'identity_checks.py'),'historical helper byte-identical')
    old=json.loads(blob(parent+'PAIR_BINDING.disabled.json'))
    phase=PLAN['phases'][0]
    check(len(PLAN['phases'])==1 and phase['name']=='W1','one W1')
    check(phase['training_command']==old['phases'][0]['training_command'],'S1 science argv identical')
    exempt={'TRN_EXP_LABEL','TORCH_NEURONX_METRICS_DIR'}
    check({k:v for k,v in phase['environment'].items() if k not in exempt}==
          {k:v for k,v in old['phases'][0]['environment'].items() if k not in exempt},'all other S1 environment identical')
    check(PLAN['cache_environment']==old['cache_environment'],'all six S1 paths identical')
    check(len(set(PLAN['cache_environment'].values()))==6,'six distinct caches')
    for key,value in PLAN['cache_environment'].items():check(phase['environment'][key]==value,'cache env:'+key)
    check(phase['environment']['NEURON_COMPILE_CACHE_URL']==PLAN['cache_environment']['NEURON_COMPILE_CACHE_DIR'],'compile URL equals retained DIR')
    check(phase['environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS']=='1800','source env1800')
    for flag,value in [('--seed','42'),('--num-steps','5'),('--max-train-seconds','1800'),('--seq-len','1024'),('--matrix-lr','0.015')]:
        check(phase['training_command'][phase['training_command'].index(flag)+1]==value,'fixed science:'+flag)
    check('--fresh-caches' not in phase['command'],'no cache recreation')
    check(phase['command'][:5]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','1800s'],'exact whole-process watchdog')
    check(phase['command'][-2:]==['--mode','train'],'one train no eval')
    for file,key in [('launch-command.txt','launch_sha256'),('provenance.json','provenance_sha256')]:
        check(digest(blob(PACK+'w1/package/'+file))==phase[key],'package pin:'+file)
    check(shlex.split(blob(PACK+'w1/package/launch-command.txt').decode())==phase['training_command'],'package exact argv')
    historical=json.loads(blob('research/v2/operator/20260926-trn22-stable-source-pair/completed/results-bundle.json'))
    raw=historical['files']['S1-pre-source-identity.json']
    check(raw['text'].encode()==blob(PACK+'S1-pre-source-identity.json'),'raw anchor byte equality incl nanoseconds')
    check(digest(raw['text'].encode())==raw['sha256'],'raw anchor authenticated')
    check(type(ANCHOR['executed_train']['st_mtime_ns']) is int,'integer ns retained')
    calls=[n for n in ast.walk(DRIVER) if isinstance(n,ast.Call)]
    jobs=[n for n in calls if ast.unparse(n.func)=='subprocess.call']
    check(len(jobs)==1,'one subprocess.call only')
    check(any(k.arg=='pass_fds' and ast.unparse(k.value)=='(lock_fd,)' for k in jobs[0].keywords),'same inherited FD')
    guard=[n for n in calls if ast.unparse(n.func)=='guard.require_idle_under_inherited_lock']
    check(len(guard)==3 and guard[0].lineno<guard[1].lineno<jobs[0].lineno<guard[2].lineno,'three in-lock checks ordered')
    check(not any(isinstance(n,ast.Assert) for n in ast.walk(DRIVER)),'driver gates survive O')
    check(all(not PLAN[k] for k in ('launch_authorized','stage_authorized','host_reserved','source_adoption','submission_authorized')),'disabled')
    check(PLAN['original_S2_launched'] is False and PLAN['no_retry'] is True,'no S2 resurrection')
    check('PYTHONPATH' not in phase['environment'] and 'PYTHONHOME' not in phase['environment'],'no import shim')
    check('PYTHONOPTIMIZE' not in phase['environment'] and '-O' not in phase['command'],'real harness assertions retained')
    terminal=json.loads(historical['files']['terminal.json']['text'])
    check(terminal['status']=='PAIR_REFUSED_OR_INCOMPLETE' and 'MATCHED_PRINCIPAL_COMPILER_BYTES_MISSING' in terminal['error'],'original raw pair failure preserved')
    check(not any(k.startswith('S2-launch') for k in historical['files']),'original S2 launch absent')


def runner(scenario):
    plan=copy.deepcopy(PLAN)
    phase=plan['phases'][0]
    recorded,events={},[]
    if scenario=='two_phases':plan['phases'].append(copy.deepcopy(phase))
    if scenario=='wrong_plan_root':plan['root']+='-wrong'
    class VirtualPath:
        def __init__(self,p):self.p=str(p)
        def __str__(self):return self.p
        def __truediv__(self,q):return VirtualPath(self.p+'/'+str(q))
        def __eq__(self,q):return self.p==str(q)
        def resolve(self):return self
        @property
        def parent(self):return VirtualPath(self.p.rsplit('/',1)[0])
        def glob(self,p):return []
        def is_dir(self):return not(scenario=='missing_cache' and self.p==next(iter(plan['cache_environment'].values())))
        def iterdir(self):
            return iter(['unexpected'] if scenario=='dirty_metrics' and self.p==phase['environment']['TORCH_NEURONX_METRICS_DIR'] else [])
        def exists(self):return self.p.rsplit('/',1)[-1] in recorded
        def read_text(self):
            return json.dumps(plan if self.p.endswith('WARM_BINDING.disabled.json') else ANCHOR)
        def open(self,mode):return io.StringIO()
    def require(ok,message):
        if not ok:raise RuntimeError(message)
    ninputs=0
    def inputs(*args):
        nonlocal ninputs
        ninputs+=1
        x=copy.deepcopy(ANCHOR)
        x['package_train']={'new':1};x['launch']={'new':2};x['provenance']={'new':3}
        if scenario=='pre_identity' or scenario=='post_identity' and ninputs==2:x['executed_train']['sha256']='changed'
        return x
    def validate(*args):
        events.append('validate')
        if scenario=='invalid_result':raise RuntimeError('INVALID_RESULT')
        return dict(fifth_boundary_seconds=121 if scenario=='slow' else 120,first_five_logged=[])
    def capture(*args):
        events.append('capture')
        if scenario=='forensics_error':raise RuntimeError('OPTIONAL_BYTES_MISSING')
        return dict(exact_input_bytes_complete=scenario!='forensics_missing',cache_hit=False,native_neff_reuse='UNPROVEN')
    def guarded(*args):
        number=1+len([e for e in events if e.startswith('guard')])
        events.append('guard'+str(number))
        if scenario=='guard'+str(number):raise RuntimeError('BUSY_OR_UNKNOWN')
        if scenario=='fd_change' and number==2:return 10
        if scenario=='post_fd_change' and number==3:return 10
        return 9
    guard=SimpleNamespace(require_idle_under_inherited_lock=guarded)
    helper=SimpleNamespace(inputs=inputs,no_symlink=VirtualPath,validate_phase=validate,capture_inputs=capture)
    def spawn(command,**kwargs):
        events.append('child')
        check(command==phase['command'],'mock child exact command:'+scenario)
        check(kwargs['pass_fds']==(9,) and kwargs['env']==phase['environment'],'mock FD/env:'+scenario)
        return 124 if scenario=='child_timeout' else 1 if scenario=='child_fail' else 0
    hashes={'WARM_BINDING.disabled.json':digest(blob(PACK+'WARM_BINDING.disabled.json')),
            'idle_guard.py':plan['idle_guard_sha256'],'dispatch_driver.py':plan['dispatch_driver_sha256'],
            'identity_checks.py':plan['helper_sha256'],'EXPERIMENT_PROPOSAL.json':plan['proposal_sha256'],
            'S1-pre-source-identity.json':plan['s1_identity_sha256']}
    namespace=dict(ROOT=VirtualPath(PLAN['root']),PLAN_SHA=hashes['WARM_BINDING.disabled.json'],
                   __file__=PLAN['root']+'/host_driver.py',Path=VirtualPath,json=json,
                   sha=lambda p:'bad' if scenario=='support_hash' else hashes[str(p).rsplit('/',1)[-1]],
                   require=require,record=lambda name,value:recorded.__setitem__(name,value),
                   load=lambda name,path:guard if 'idle_guard' in name else helper,utc=lambda:'fixed',
                   classify_target=TARGET['classify_target'],
                   socket=SimpleNamespace(gethostname=lambda:'bad' if scenario=='host' else plan['expected_hostname']),
                   os=SimpleNamespace(path=SimpleNamespace(lexists=lambda p:scenario=='sandbox_exists' and str(p)==phase['sandbox'] or scenario=='stop' and str(p).endswith('/STOP')),
                                      statvfs=lambda p:SimpleNamespace(f_bavail=0 if scenario=='disk' else 10*1024**3,f_frsize=1)),
                   subprocess=SimpleNamespace(call=spawn,DEVNULL=-3,STDOUT=-2))
    resourcefn=next(n for n in DRIVER.body if isinstance(n,ast.FunctionDef) and n.name=='resources')
    exec(compile(ast.Module(body=[resourcefn],type_ignores=[]),'<frozen-resources>','exec'),namespace)
    top=next(n for n in DRIVER.body if isinstance(n,ast.Try))
    try:exec(compile(ast.Module(body=[top],type_ignores=[]),'<frozen-runner>','exec'),namespace)
    except RuntimeError:pass
    terminal=recorded['terminal.json']
    good=scenario in ('good','slow','forensics_error','forensics_missing')
    check((terminal['status']=='WARM_OBSERVATION_COMPLETE_PENDING_INDEPENDENT_REVIEW')==good,'runner outcome:'+scenario)
    check(events.count('child')<=1,'no second child:'+scenario)
    check(terminal['original_S2_launched'] is False and terminal['no_retry'] is True,'original preserved:'+scenario)
    if good:
        check(events==['guard1','guard2','child','guard3','validate','capture'],'one locked observed child order:'+scenario)
        check(terminal['native_cache_reuse']=='UNPROVEN' and terminal['path_only_causality']=='UNPROVEN','no cache proof:'+scenario)
        check(terminal['target_disposition']==('WARM_TARGET_MISSED' if scenario=='slow' else 'USEFUL_WARM_START_OBSERVED'),'target outcome:'+scenario)
    if scenario in ('guard1','guard2','fd_change','pre_identity','host','stop','disk','support_hash','wrong_plan_root','two_phases','dirty_metrics','missing_cache','sandbox_exists'):
        check('child' not in events,'preflight no child:'+scenario)
    return dict(scenario=scenario,events=events,status=terminal['status'])


def finite_fixture():
    with tempfile.TemporaryDirectory(prefix='warm-independent-tiny-') as temp:
        base=Path(temp);sandbox=base/'train';(sandbox/'out').mkdir(parents=True)
        for name,raw in [('train.py',b'fixture-source'),('prepare.py',b'fixture-prepare'),('out/final.pt',b'fixture-checkpoint')]:
            (sandbox/name).write_bytes(raw)
        (sandbox/'train.py').chmod(0o444)
        plan=dict(source_sha256=digest(b'fixture-source'),prepare_sha256=digest(b'fixture-prepare'),cache_environment=PLAN['cache_environment'])
        phase=dict(root=str(base),sandbox=str(sandbox),name='W1')
        good=dict(returncode=0,mode='train',step_boundaries=5,
                  observations=[dict(step=i,observed_seconds=i+116.0) for i in range(5)],
                  wall_seconds=123.0,startup_allowance_observed=120.0,charged_through_process_exit=3.0,
                  charged_through_process_exit_within_budget=True,fresh_empty_sdk_caches={},
                  checkpoint_sha256=digest(b'fixture-checkpoint'),checkpoint_bytes=len(b'fixture-checkpoint'))
        log=''.join(f'step {i} | loss 1.0 | lrm 1.0 | dt 0.5s | synthetic\n' for i in range(5))
        def call(value=good,text=log):
            (sandbox/'rehearsal.json').write_text(json.dumps(value))
            (base/'training.log').write_text(text)
            return HELPER['validate_phase'](plan,phase)
        check(call()['fifth_boundary_seconds']==120,'tiny exact120 validated')
        for fifth,wall,disposition in [(120.001,123,'WARM_TARGET_MISSED'),(1000,1003,'WARM_TARGET_MISSED'),(900,2700,'WARM_TARGET_MISSED')]:
            r=copy.deepcopy(good);r['observations']=[dict(step=i,observed_seconds=fifth-4+i) for i in range(5)]
            r.update(wall_seconds=wall,startup_allowance_observed=min(fifth,900),charged_through_process_exit=wall-min(fifth,900))
            value=call(r)
            check(TARGET['classify_target'](value)==disposition,'actual fifth not capped:'+str(fifth))
        variants=[]
        for key,value in [('returncode',False),('returncode',1),('mode','eval'),('step_boundaries',4),
                          ('wall_seconds',float('inf')),('startup_allowance_observed',900),('charged_through_process_exit',float('nan')),
                          ('charged_through_process_exit_within_budget',False),('fresh_empty_sdk_caches',{'new':'cache'}),
                          ('checkpoint_sha256','0'*64),('checkpoint_bytes',0)]:
            r=copy.deepcopy(good);r[key]=value;variants.append((key+repr(value),r))
        for name,value in [('nan',float('nan')),('negative',-1),('reverse',0)]:
            r=copy.deepcopy(good);r['observations'][-1]['observed_seconds']=value;variants.append(('time'+name,r))
        r=copy.deepcopy(good);r['observations'][2]['step']=True;variants.append(('boolean-step',r))
        r=copy.deepcopy(good);r['wall_seconds']=1920.001;r['charged_through_process_exit']=1800.001;variants.append(('overbudget',r))
        for name,r in variants:raises(lambda r=r:call(r),'receipt negative:'+name)
        raises(lambda:call(text=log.replace('loss 1.0','loss nan',1)),'nonfinite log')
        raises(lambda:call(text=log.replace('step 4','step 3')),'duplicate logged step')
        (sandbox/'train.py').chmod(0o644)
        raises(call,'writable sandbox source')
        (sandbox/'train.py').chmod(0o444)
        (sandbox/'out/final.pt').unlink()
        raises(call,'missing checkpoint')


def identity_negatives():
    with tempfile.TemporaryDirectory(prefix='warm-independent-identity-') as temp:
        p=Path(temp)/'tiny';p.write_bytes(b'123');p.chmod(0o444)
        record=HELPER['identity'](p,digest(b'123'),True)
        check(record['st_size']==3,'tiny read-only identity')
        raises(lambda:HELPER['identity'](p,'0'*64,True),'wrong immutable hash')
        link=Path(temp)/'symlink';link.symlink_to(p)
        raises(lambda:HELPER['identity'](link),'symlink refused')
        hard=Path(temp)/'hard';os.link(p,hard)
        raises(lambda:HELPER['identity'](p),'hardlink refused')


def idle_guard_negatives():
    guard={}
    exec(compile(blob('research/v2/operator/20260926-common-host-idle-guard/idle_guard.py'),'<frozen-idle-guard>','exec'),guard)
    device=dict(neuron_device=0,instance_type='trn2.3xlarge',nc_count=4,
                logical_neuroncore_config=2,neuroncore_ids=[0,1,2,3],neuron_processes=[])
    for scenario in ('idle','busy_job','busy_device','bad_ps','ps_error','ps_stderr','nls_error','nls_stderr','nls_malformed','bad_schema','bad_pid','timeout'):
        calls=[]
        def run(command,**kwargs):
            calls.append(command)
            check(kwargs['timeout']==15 and kwargs['env']==guard['CHECK_ENV'],'guard bounded clean-env:'+scenario+':'+str(len(calls)))
            if scenario=='timeout':raise subprocess.TimeoutExpired(command,15)
            if command==guard['PS']:
                rows='100 99 python3 own-driver\n99 98 flock own-flock\n'
                if scenario=='busy_job':rows+='200 1 python3 python /tmp/train.py\n'
                if scenario=='bad_ps':rows='malformed'
                return SimpleNamespace(stdout=rows,stderr='bad' if scenario=='ps_stderr' else '',returncode=1 if scenario=='ps_error' else 0)
            value=copy.deepcopy(device)
            if scenario=='busy_device':value['neuron_processes']=[{'pid':200}]
            if scenario=='bad_pid':value['neuron_processes']=[{'pid':True}]
            if scenario=='bad_schema':value['nc_count']=8
            return SimpleNamespace(stdout='{' if scenario=='nls_malformed' else json.dumps([value]),stderr='bad' if scenario=='nls_stderr' else '',returncode=1 if scenario=='nls_error' else 0)
        evidence={}
        if scenario=='idle':
            guard['check_idle'](run,100,99,evidence)
            check(calls==[guard['PS'],guard['NEURON_LS']],'both actual idle checks')
        else:raises(lambda:guard['check_idle'](run,100,99,evidence),'shared guard refuses:'+scenario)
        if scenario=='busy_job':check(calls==[guard['PS'],guard['NEURON_LS']],'busy ps still collects device check')


if __name__=='__main__':
    static()
    outcomes=[runner(s) for s in ('good','slow','forensics_error','forensics_missing','support_hash','wrong_plan_root',
              'two_phases','sandbox_exists','dirty_metrics','missing_cache','pre_identity','post_identity',
              'host','stop','disk','guard1','guard2','guard3','fd_change','post_fd_change','child_fail','child_timeout','invalid_result')]
    finite_fixture();identity_negatives();idle_guard_negatives()
    print(json.dumps(dict(status='PASS_INDEPENDENT_W1_MECHANICS_ONLY',candidate=REF,
                          count=len(CHECKS),checks=CHECKS,runner_scenarios=outcomes,
                          host_actions=False,native_imports=False,tensors=False,
                          actual_runner_process_executed=False,cleanup_proven=False,
                          cleanup_scope='Explicit Operator PID/PPID/PGID/SID/compiler/device postflight remains mandatory; mocked driver checks do not prove OS cleanup.',
                          fresh_helper_source_review=False,
                          authorship='Reviewer did not author new W1 packet; historically authored reused frozen stable-pair identity helper.'),indent=2,allow_nan=False))
