"""Disabled clean architecture true-cold full replay then fresh public20M under one lifetime lock."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import socket
import sys

ROOT = Path("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-clean-cold-trn22-r1-packet")
PREP = Path("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py")
HARNESS = Path("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/research/scratch/submission_rehearsal.py")
LABEL = "fs-20260926-arch3e79-clean-cold-trn22-r1"
SOURCE_SHA = "3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f"
PREP_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
HARNESS_SHA = "7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b"
TRAIN_COMMAND = ["/usr/bin/timeout","--foreground","--signal=TERM","--kill-after=60s","2730s","/home/ubuntu/workspace/native_venv/bin/python","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/research/scratch/submission_rehearsal.py","--package","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-clean-cold-trn22-r1-packet/package","--sandbox","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-clean-cold-trn22-r1-packet/train","--mode","train","--fresh-caches"]
EVAL_COMMAND = ["/usr/bin/timeout","--foreground","--signal=TERM","--kill-after=60s","1800s","/home/ubuntu/workspace/native_venv/bin/python","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/research/scratch/submission_rehearsal.py","--package","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-clean-cold-trn22-r1-packet/package","--sandbox","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-clean-cold-trn22-r1-packet/eval","--mode","eval","--checkpoint","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-clean-cold-trn22-r1-packet/train/out/final.pt","--eval-tokens","20971520","--fresh-caches"]
TRAIN_ENV = {"HOME":"/home/ubuntu","NEURON_CC_FLAGS":"--optlevel=1 --auto-cast matmult --auto-cast-type bf16","NEURON_COMPETITION_R1_CACHE_DIR":"/home/ubuntu/.cache/neuron_competition_r1","NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS":"1800","NEURON_COMPETITION_R1_SEED":"42","NEURON_LOGICAL_NC_CONFIG":"2","NKI_VENV_PATH":"/home/ubuntu/workspace/native_venv","PATH":"/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin","PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","TORCH_NEURONX_ENABLED_METRIC_TABLES":"graph_stats","TORCH_NEURONX_METRICS_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-clean-cold-trn22-r1-packet/metrics-training","TRN_EXP_LABEL":"fs-20260926-arch3e79-clean-cold-trn22-r1","NEURON_COMPILE_CACHE_URL":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-clean-cold-trn22-r1-packet/train/caches/compiler"}
EVAL_ENV = {"HOME":"/home/ubuntu","NEURON_CC_FLAGS":"--optlevel=1","NEURON_COMPETITION_R1_CACHE_DIR":"/home/ubuntu/.cache/neuron_competition_r1","NEURON_COMPETITION_R1_SEED":"42","NEURON_LOGICAL_NC_CONFIG":"2","NKI_VENV_PATH":"/home/ubuntu/workspace/native_venv","PATH":"/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin","PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","TRN_EXP_LABEL":"fs-20260926-arch3e79-clean-cold-trn22-r1","NEURON_COMPILE_CACHE_URL":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-clean-cold-trn22-r1-packet/eval/caches/compiler","TORCH_NEURONX_ENABLED_METRIC_TABLES":"graph_stats","TORCH_NEURONX_METRICS_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-clean-cold-trn22-r1-packet/metrics-eval"}

def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def record(name,obj):
    with (ROOT/name).open('x') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n')
def preflight(env):
    if sys.flags.optimize:
        raise RuntimeError('OPTIMIZED_DRIVER_FORBIDDEN')
    assert socket.gethostname()=='ip-172-31-32-253','HOSTNAME'
    for base in (Path('/home/ubuntu'),PREP.parent,ROOT,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
        for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):
            assert not os.path.lexists(base/rel),'STOP: '+str(base/rel)
    v=os.statvfs('/')
    assert v.f_bavail*v.f_frsize>=10.5*1024**3,'DISK_HEADROOM'
    assert sha(ROOT/'package/train.py')==SOURCE_SHA,'SOURCE_SHA'
    assert sha(ROOT/'package/launch-command.txt')=='b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd','LAUNCH_SHA'
    assert sha(ROOT/'package/provenance.json')=='9f917597ed0f6e48e32b029e0b5e158011620eee832bb4d204f7d97fea4f685c','PROVENANCE_SHA'
    assert sha(PREP)==PREP_SHA,'PREPARE_SHA'
    assert sha(HARNESS)==HARNESS_SHA,'HARNESS_SHA'
    # The pinned rehearsal creates these six SDK roots fresh before its training/eval child.
    phase_root = ROOT/('train' if env is TRAIN_ENV else 'eval')
    assert not os.path.lexists(phase_root),'PHASE_SANDBOX_ALREADY_EXISTS'
    assert env['NEURON_COMPILE_CACHE_URL']==str(phase_root/'caches/compiler'),'FRESH_URL_BINDING'
def phase(name,command,env,lock_fd):
    record(name+'-launch.json',{'utc':utc(),'label':LABEL,'command':command,'environment':env,'source_sha256':SOURCE_SHA,'prepare_sha256':PREP_SHA,'harness_sha256':HARNESS_SHA})
    with (ROOT/(name+'.log')).open('x') as out:
        out.write(name.upper()+'_START_UTC='+utc()+'\n'); out.flush()
        rc=subprocess.call(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,pass_fds=(lock_fd,))
        out.write(name.upper()+'_END_UTC='+utc()+'\n'); out.flush()
    with (ROOT/('run-exit-code.txt' if name=='training' else 'eval-exit-code.txt')).open('x') as f: f.write(str(rc)+'\n')
    return rc
try:
    preflight(TRAIN_ENV)
    assert not os.path.lexists(ROOT/'train'),'TRAIN_SANDBOX_ALREADY_EXISTS'
    assert not os.path.lexists(ROOT/'eval'),'EVAL_SANDBOX_ALREADY_EXISTS'
    assert sha(ROOT/'dispatch_driver.py')=='c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a','DISPATCH_SHA'
    assert sha(ROOT/'idle_guard.py')=='adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737','IDLE_GUARD_SHA'
    spec=importlib.util.spec_from_file_location('bound_idle_guard',ROOT/'idle_guard.py')
    guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    lock_fd=guard.require_idle_under_inherited_lock('/home/ubuntu/.cache/trainium_freshstart/device0.lock',ROOT/'host_driver.py',record)
    rc=phase('training',TRAIN_COMMAND,TRAIN_ENV,lock_fd)
    assert rc==0,'TRAIN_EXIT_'+str(rc)
    rehearsal=json.loads((ROOT/'train/rehearsal.json').read_text())
    assert rehearsal['returncode']==0 and rehearsal['charged_through_process_exit_within_budget'] is True,'TRAIN_BUDGET_FAILED'
    assert (ROOT/'train/out/final.pt').is_file(),'MISSING_CHECKPOINT'
    record('checkpoint.json',{'utc':utc(),'sha256':rehearsal['checkpoint_sha256'],'bytes':rehearsal['checkpoint_bytes']})
    preflight(EVAL_ENV)
    eval_fd=guard.require_idle_under_inherited_lock('/home/ubuntu/.cache/trainium_freshstart/device0.lock',ROOT/'host_driver.py',lambda name,obj:record('eval-'+name,obj))
    assert eval_fd==lock_fd,'LOCK_FD_CHANGED'
    rc=phase('fresh-public20m',EVAL_COMMAND,EVAL_ENV,lock_fd)
    record('terminal.json',{'utc':utc(),'label':LABEL,'status':'COMPLETE_PENDING_INDEPENDENT_REVIEW' if rc==0 else 'EVAL_FAILED','eval_exit_code':rc,'no_automatic_next_job':True,'submission_authorized':False})
    sys.exit(rc)
except Exception as exc:
    record('terminal.json',{'utc':utc(),'label':LABEL,'status':'FULL_DRIVER_FAILED','error':str(exc),'no_automatic_next_job':True})
    raise
