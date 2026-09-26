"""Disabled anchored-source full control composition; source/harness/support unchanged."""
import datetime as dt
import hashlib
import importlib.util
import json
import math
from importlib.machinery import PathFinder
import re
import socket
import stat
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-clock36-anchored-warm-control-trn21-r1-packet")
PREP = Path("/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/prepare.py")
HARNESS = Path("/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/research/scratch/submission_rehearsal.py")
LABEL = "fs-20260926-clock36-anchored-warm-control-trn21-r1"
SOURCE_SHA = "36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb"
PREP_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
HARNESS_SHA = "7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b"
TRAIN_COMMAND = ["/usr/bin/timeout","--foreground","--signal=TERM","--kill-after=60s","2730s","/home/ubuntu/workspace/native_venv/bin/python","/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/research/scratch/submission_rehearsal.py","--package","/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-clock36-anchored-warm-control-trn21-r1-packet/package","--sandbox","/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-clock36-anchored-warm-control-trn21-r1-packet/train","--mode","train"]
EVAL_COMMAND = ["/usr/bin/timeout","--foreground","--signal=TERM","--kill-after=60s","1800s","/home/ubuntu/workspace/native_venv/bin/python","/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/research/scratch/submission_rehearsal.py","--package","/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-clock36-anchored-warm-control-trn21-r1-packet/package","--sandbox","/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-clock36-anchored-warm-control-trn21-r1-packet/eval","--mode","eval","--checkpoint","/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-clock36-anchored-warm-control-trn21-r1-packet/train/out/final.pt","--eval-tokens","20971520"]
TRAIN_ENV = {"HOME":"/home/ubuntu","NEURON_CC_FLAGS":"--optlevel=1 --auto-cast matmult --auto-cast-type bf16","NEURON_COMPETITION_R1_CACHE_DIR":"/home/ubuntu/.cache/neuron_competition_r1","NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS":"1800","NEURON_COMPETITION_R1_SEED":"42","NEURON_LOGICAL_NC_CONFIG":"2","NKI_VENV_PATH":"/home/ubuntu/workspace/native_venv","PATH":"/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin","PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","TORCH_NEURONX_ENABLED_METRIC_TABLES":"graph_stats","TORCH_NEURONX_METRICS_DIR":"/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-clock36-anchored-warm-control-trn21-r1-packet/metrics-training","TRN_EXP_LABEL":"fs-20260926-clock36-anchored-warm-control-trn21-r1","NEURON_COMPILE_CACHE_URL":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/compiler","TORCH_NEURONX_NEFF_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/neff","TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/local","TORCH_NEURONX_HLO_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/hlo","TORCHINDUCTOR_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/inductor","NEURON_COMPILE_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/compiler","TORCH_NEURONX_DEBUG_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/debug"}
EVAL_ENV = {"HOME":"/home/ubuntu","NEURON_CC_FLAGS":"--optlevel=1","NEURON_COMPETITION_R1_CACHE_DIR":"/home/ubuntu/.cache/neuron_competition_r1","NEURON_COMPETITION_R1_SEED":"42","NEURON_LOGICAL_NC_CONFIG":"2","NKI_VENV_PATH":"/home/ubuntu/workspace/native_venv","PATH":"/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin","PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","TRN_EXP_LABEL":"fs-20260926-clock36-anchored-warm-control-trn21-r1","NEURON_COMPILE_CACHE_URL":"/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/eval/caches/compiler","TORCH_NEURONX_NEFF_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/eval/caches/neff","TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/eval/caches/local","TORCH_NEURONX_HLO_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/eval/caches/hlo","TORCHINDUCTOR_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/eval/caches/inductor","NEURON_COMPILE_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/eval/caches/compiler","TORCH_NEURONX_DEBUG_DIR":"/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/eval/caches/debug"}

def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def record(name,obj):
    with (ROOT/name).open('x') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n')
def preflight(env):
    for base in (Path('/home/ubuntu'),PREP.parent,ROOT,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
        for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):
            assert not os.path.lexists(base/rel),'STOP: '+str(base/rel)
    v=os.statvfs('/')
    assert v.f_bavail*v.f_frsize>=10.5*1024**3,'DISK_HEADROOM'
    assert sha(ROOT/'package/train.py')==SOURCE_SHA,'SOURCE_SHA'
    assert sha(PREP)==PREP_SHA,'PREPARE_SHA'
    assert sha(HARNESS)==HARNESS_SHA,'HARNESS_SHA'
    for key in ('TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'):
        assert Path(env[key]).is_dir(),'CACHE_MISSING:'+key
def phase(name,command,env,lock_fd):
    record(name+'-launch.json',{'utc':utc(),'label':LABEL,'command':command,'environment':env,'source_sha256':SOURCE_SHA,'prepare_sha256':PREP_SHA,'harness_sha256':HARNESS_SHA})
    with (ROOT/(name+'.log')).open('x') as out:
        out.write(name.upper()+'_START_UTC='+utc()+'\n'); out.flush()
        rc=subprocess.call(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,pass_fds=(lock_fd,))
        out.write(name.upper()+'_END_UTC='+utc()+'\n'); out.flush()
    with (ROOT/('run-exit-code.txt' if name=='training' else 'eval-exit-code.txt')).open('x') as f: f.write(str(rc)+'\n')
    return rc
PLAN_SHA = 'e71a92a55043871b7dd5660c191fad4135d20384e6fa1be3b6edc91398d83e10'
STAT_FIELDS = ('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_mtime_ns','st_ctime_ns')
ANCESTOR_STABLE_FIELDS = ('st_dev','st_ino','st_mode','st_uid','st_gid')

def require(ok,message):
    if not ok:
        raise RuntimeError(message)

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def directory_record(checks,path):
    path=checks.no_symlink(path)
    item=path.stat()
    require(stat.S_ISDIR(item.st_mode),'NOT_DIRECTORY:'+str(path))
    return {key:getattr(item,key) for key in STAT_FIELDS}

def anchored_inputs(checks,plan):
    source=Path(plan['executed_source'])
    imported=Path(plan['expected_prepare_import'])
    require(imported==source.parent/'prepare.py','SCRIPT_PREPARE_DIRECTORY')
    resolved={}
    for name in plan['source_import_roots']+['sitecustomize','usercustomize']:
        spec=PathFinder.find_spec(name,[str(source.parent)])
        if name=='prepare':
            require(spec is not None and spec.origin==str(imported),'EXPECTED_PREPARE_RESOLUTION')
            resolved[name]=spec.origin
        else:
            require(spec is None,'SCRIPT_DIRECTORY_IMPORT_SHADOW:'+name)
            resolved[name]=None
    files={'executed_train':checks.identity(source,plan['source_sha256'],readonly=True),
           'expected_imported_prepare':checks.identity(imported,plan['prepare_sha256'],readonly=False)}
    mapped={'device':'st_dev','inode':'st_ino','mode':'st_mode','uid':'st_uid','size':'st_size',
            'mtime_ns':'st_mtime_ns','ctime_ns':'st_ctime_ns','sha256':'sha256'}
    require(set(plan['anchor_observation'])=={str(source),str(imported)},'ANCHOR_FILE_SET')
    for key,path in (('executed_train',source),('expected_imported_prepare',imported)):
        expected=plan['anchor_observation'][str(path)]
        require(expected['stable_during_read'] is True,'UNSTABLE_CAPTURED_ANCHOR')
        for raw,field in mapped.items():
            require(files[key][field]==expected[raw],'ANCHORED_FIELD_CHANGED:'+key+':'+raw)
    package=Path(plan['package'])
    files['package_train']=checks.identity(package/'train.py',plan['source_sha256'],readonly=True)
    for name,pin in plan['package_pins'].items():
        files['package_'+name]=checks.identity(package/name,pin,readonly=True)
    files['protected_prepare']=checks.identity(plan['prepare_path'],plan['prepare_sha256'])
    files['harness']=checks.identity(plan['harness_path'],plan['harness_sha256'])
    files['driver']=checks.identity(ROOT/'host_driver.py')
    for name,pin in plan['support'].items():
        files['support_'+name]=checks.identity(ROOT/name,pin)
    files['binding']=checks.identity(ROOT/'ANCHORED_BINDING.disabled.json',PLAN_SHA)
    manifest=json.loads((package/'provenance.json').read_text())
    launch=(package/'launch-command.txt').read_text().strip()
    require(manifest['train_sha256']==plan['source_sha256'] and manifest['launch_command']==launch,'PACKAGE_PROVENANCE')
    import shlex
    require(shlex.split(launch)==plan['training_command'],'PACKAGE_ARGV')
    # Never call W1 inputs(): neither this old directory nor prepare0600 is
    # claimed read-only. Only its unchanged individual-file operation is reused.
    return {'files':files,'import_resolution_expected_only':resolved,
            'historical_directories':{str(path):directory_record(checks,path) for path in (source.parent,source.parent.parent)},
            'ancestors':{str(path):directory_record(checks,path) for path in source.parents},
            'actual_worker_import_origin_observed':False}

def unchanged(before,after):
    for key in ('files','import_resolution_expected_only','historical_directories'):
        require(after[key]==before[key],'ANCHORED_SNAPSHOT_CHANGED:'+key)
    require(before['ancestors'].keys()==after['ancestors'].keys(),'ANCESTOR_SET_CHANGED')
    for path in before['ancestors']:
        for field in ANCESTOR_STABLE_FIELDS:
            require(after['ancestors'][path][field]==before['ancestors'][path][field],'ANCESTOR_IDENTITY_CHANGED:'+path+':'+field)

def check_lock(fd,plan):
    st=os.fstat(fd)
    require((st.st_dev,st.st_ino)==(plan['original_lock']['device'],plan['original_lock']['inode']),'ORIGINAL_LOCK_IDENTITY')

def account_training(receipt,log):
    require(type(receipt['returncode']) is int and receipt['returncode']==0 and receipt['mode']=='train','TRAIN_EXIT_OR_MODE')
    observed=receipt['observations']
    require(type(receipt['step_boundaries']) is int and len(observed)==receipt['step_boundaries'] and len(observed)>=5,'OFFICIAL_BOUNDARIES_REQUIRED')
    require(all(type(v['step']) is int for v in observed) and [v['step'] for v in observed]==list(range(len(observed))),'BOUNDARY_SEQUENCE')
    times=[v['observed_seconds'] for v in observed]
    require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in times) and times==sorted(times),'BOUNDARY_TIMES')
    wall=receipt['wall_seconds']
    require(type(wall) in (int,float) and math.isfinite(wall) and wall>=times[-1],'PROCESS_WALL')
    allowance=min(times[4],900.0)
    charged=wall-allowance
    require(type(receipt['startup_allowance_observed']) in (int,float) and receipt['startup_allowance_observed']==allowance,'ALLOWANCE')
    require(type(receipt['charged_through_process_exit']) in (int,float) and receipt['charged_through_process_exit']==charged and 0<=charged<=1800 and receipt['charged_through_process_exit_within_budget'] is True,'CHARGED_THROUGH_EXIT')
    require(receipt['fresh_empty_sdk_caches']=={},'NO_COLD_CACHE_FLAG')
    rows=re.findall(r'^step (\d+) \| loss (\S+) \| lrm (\S+) \| dt (\S+)s \|',log,re.MULTILINE)
    require([int(v[0]) for v in rows[:5]]==list(range(5)),'FIRST_FIVE_LOGGED_LOSSES')
    values=[dict(step=int(step),loss=float(loss),lrm=float(lrm),step_seconds=float(duration)) for step,loss,lrm,duration in rows]
    require(all(math.isfinite(v[k]) for v in values for k in ('loss','lrm','step_seconds')),'NONFINITE_LOGGED_TRAINING')
    return {'fifth_boundary_seconds':times[4],'process_wall_seconds':wall,'charged_through_exit_seconds':charged,
            'observed_steps':len(observed),'first_five_logged':values[:5],
            'warm_target_seconds':180,'warm_target_disposition':'WARM_TARGET_MET' if times[4]<=180 else 'WARM_TARGET_MISSED',
            'cache_hit':'UNPROVEN','source_path_only_causality':'UNPROVEN','quality_result':'PENDING_ATTACHED_EVAL'}

def copied_inputs(checks,plan,sandbox):
    return {'train':checks.identity(sandbox/'train.py',plan['source_sha256'],readonly=True),
            'prepare':checks.identity(sandbox/'prepare.py',plan['prepare_sha256'])}

def graph_inventory(checks):
    result={}
    for rank in range(4):
        path=Path(TRAIN_ENV['TORCH_NEURONX_METRICS_DIR'])/('metric_table_graph_stats_rank'+str(rank)+'.csv')
        try:result[str(rank)]=checks.identity(path,max_bytes=2*1024**2)
        except Exception as error:result[str(rank)]={'unavailable':str(error)}
    return {'closed_files':result,'zero_compile_us_is_cache_hit_proof':False,'no_cache_walk':True}

def main():
    guard=checks=plan=baseline=None
    lock_fd=None
    error=None
    result=None
    try:
        require(sys.flags.optimize==0,'NORMAL_PYTHON_REQUIRED_FOR_UNCHANGED_PREFLIGHT')
        require(Path(__file__).resolve()==ROOT/'host_driver.py','EXACT_DRIVER_PATH')
        require(sha(ROOT/'ANCHORED_BINDING.disabled.json')==PLAN_SHA,'PLAN_SHA')
        plan=json.loads((ROOT/'ANCHORED_BINDING.disabled.json').read_text())
        require(plan['root']==str(ROOT) and plan['label']==LABEL,'PLAN_ROOT_LABEL')
        for name,pin in plan['support'].items():
            require(sha(ROOT/name)==pin,'SUPPORT_SHA:'+name)
        guard=load_module('anchored_idle_guard',ROOT/'idle_guard.py')
        checks=load_module('anchored_identity_checks',ROOT/'identity_checks.py')
        lock_fd=guard.require_idle_under_inherited_lock(plan['device_lock'],ROOT/'host_driver.py',record)
        check_lock(lock_fd,plan)
        require(socket.gethostname()==plan['expected_hostname'],'HOSTNAME')
        require(TRAIN_COMMAND==plan['train_command'] and EVAL_COMMAND==plan['eval_command'],'BOUND_HARNESS_COMMANDS')
        require(TRAIN_ENV==plan['training_environment'] and EVAL_ENV==plan['evaluation_environment'],'BOUND_ENVIRONMENTS')
        require(all('PYTHONPATH' not in env and 'PYTHONHOME' not in env and env['PYTHONDONTWRITEBYTECODE']=='1' for env in (TRAIN_ENV,EVAL_ENV)),'NO_IMPORT_OVERRIDE_OR_BYTECODE')
        preflight(TRAIN_ENV)
        require(not os.path.lexists(ROOT/'train') and not os.path.lexists(ROOT/'eval'),'ABSENT_NEW_CWDS')
        metrics=checks.no_symlink(TRAIN_ENV['TORCH_NEURONX_METRICS_DIR'])
        require(metrics.is_dir() and not any(metrics.iterdir()),'EMPTY_NEW_METRICS')
        baseline=anchored_inputs(checks,plan)
        record('anchor-pre-training.json',baseline)
        fd=guard.require_idle_under_inherited_lock(plan['device_lock'],ROOT/'host_driver.py',lambda n,o:record('pre-training-'+n,o))
        require(fd==lock_fd,'LOCK_FD_CHANGED')
        check_lock(fd,plan)
        rc=phase('training',TRAIN_COMMAND,TRAIN_ENV,lock_fd)
        after=anchored_inputs(checks,plan)
        record('anchor-post-training.json',after)
        unchanged(baseline,after)
        require(rc==0,'TRAIN_EXIT_'+str(rc))
        rehearsal=json.loads((ROOT/'train/rehearsal.json').read_text())
        result=account_training(rehearsal,(ROOT/'training.log').read_text())
        checkpoint=checks.identity(ROOT/'train/out/final.pt',rehearsal['checkpoint_sha256'])
        require(checkpoint['st_size']==rehearsal['checkpoint_bytes'] and checkpoint['st_size']>0,'CHECKPOINT_BYTES')
        record('checkpoint.json',checkpoint)
        train_copies=copied_inputs(checks,plan,ROOT/'train')
        record('training-copies.json',train_copies)
        record('training-accounting.json',result)
        record('closed-graph-inventory.json',graph_inventory(checks))
        preflight(EVAL_ENV)
        before_eval=anchored_inputs(checks,plan)
        record('anchor-pre-eval.json',before_eval)
        unchanged(baseline,before_eval)
        eval_fd=guard.require_idle_under_inherited_lock(plan['device_lock'],ROOT/'host_driver.py',lambda n,o:record('eval-'+n,o))
        require(eval_fd==lock_fd,'LOCK_FD_CHANGED')
        check_lock(eval_fd,plan)
        rc=phase('fresh-public20m',EVAL_COMMAND,EVAL_ENV,lock_fd)
        after_eval=anchored_inputs(checks,plan)
        record('anchor-post-eval.json',after_eval)
        unchanged(baseline,after_eval)
        require(rc==0,'EVAL_EXIT_'+str(rc))
        evaluation=json.loads((ROOT/'eval/rehearsal.json').read_text())
        require(type(evaluation['returncode']) is int and evaluation['returncode']==0 and evaluation['mode']=='eval','EVAL_REHEARSAL_FAILED')
        require(checks.identity(ROOT/'train/out/final.pt',checkpoint['sha256'])==checkpoint,'CHECKPOINT_CHANGED_DURING_EVAL')
        require(copied_inputs(checks,plan,ROOT/'train')==train_copies,'TRAINING_COPIES_CHANGED')
        record('eval-copies.json',copied_inputs(checks,plan,ROOT/'eval'))
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    finally:
        final_errors=[]
        if baseline is not None:
            try:
                final=anchored_inputs(checks,plan)
                record('anchor-final.json',final)
                unchanged(baseline,final)
            except Exception as exc:final_errors.append('ANCHOR_FINAL:'+type(exc).__name__+': '+str(exc))
        if lock_fd is not None:
            try:
                fd=guard.require_idle_under_inherited_lock(plan['device_lock'],ROOT/'host_driver.py',lambda n,o:record('final-'+n,o))
                require(fd==lock_fd,'LOCK_FD_CHANGED')
                check_lock(fd,plan)
            except Exception as exc:final_errors.append('FINAL_IDLE:'+type(exc).__name__+': '+str(exc))
        failed=error is not None or bool(final_errors)
        record('terminal.json',{'utc':utc(),'label':LABEL,'status':'ANCHORED_CONTROL_REFUSED_OR_INCOMPLETE' if failed else 'ANCHORED_CONTROL_COMPLETE_PENDING_INDEPENDENT_RESULT_REVIEW',
               'error':error,'final_errors':final_errors,'training_observation':result,
               'explicit_observed_PID_postflight':'REQUIRED_FROM_SOLE_OPERATOR_BEFORE_RESULT_USE_OR_NEXT_JOB',
               'actual_worker_import_origin_observed':False,'native_cache_hit':'UNPROVEN','path_only_causality':'UNPROVEN',
               'control_eligibility':'PENDING_INDEPENDENT_RAW_BUDGET_EVAL_POSTFLIGHT_REVIEW',
               'no_automatic_next_job':True,'submission_authorized':False})
    return 1 if failed else 0

if __name__=='__main__':
    sys.exit(main())
