"""Exact cold-qualified architecture full research then fresh public20M under one lifetime lock."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-base-control-trn22-r1-full-packet")
PREP = Path("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py")
HARNESS = Path("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/research/scratch/submission_rehearsal.py")
LABEL = "fs-20260926-arch3e79-base-control-trn22-r1-full"
SOURCE_SHA = "3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c"
PREP_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
HARNESS_SHA = "7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b"
TRAIN_COMMAND = ["/usr/bin/timeout","--foreground","--signal=TERM","--kill-after=60s","2730s","/home/ubuntu/workspace/native_venv/bin/python","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/research/scratch/submission_rehearsal.py","--package","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-base-control-trn22-r1-full-packet/package","--sandbox","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-base-control-trn22-r1-full-packet/train","--mode","train"]
EVAL_COMMAND = ["/usr/bin/timeout","--foreground","--signal=TERM","--kill-after=60s","1800s","/home/ubuntu/workspace/native_venv/bin/python","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/research/scratch/submission_rehearsal.py","--package","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-base-control-trn22-r1-full-packet/package","--sandbox","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-base-control-trn22-r1-full-packet/eval","--mode","eval","--checkpoint","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-base-control-trn22-r1-full-packet/train/out/final.pt","--eval-tokens","20971520"]
TRAIN_ENV = {"HOME":"/home/ubuntu","NEURON_CC_FLAGS":"--optlevel=1 --auto-cast matmult --auto-cast-type bf16","NEURON_COMPETITION_R1_CACHE_DIR":"/home/ubuntu/.cache/neuron_competition_r1","NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS":"1800","NEURON_COMPETITION_R1_SEED":"42","NEURON_LOGICAL_NC_CONFIG":"2","NKI_VENV_PATH":"/home/ubuntu/workspace/native_venv","PATH":"/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin","PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","TORCH_NEURONX_ENABLED_METRIC_TABLES":"graph_stats","TORCH_NEURONX_METRICS_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-base-control-trn22-r1-full-packet/metrics-training","TRN_EXP_LABEL":"fs-20260926-arch3e79-base-control-trn22-r1-full","NEURON_COMPILE_CACHE_URL":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch3e79-cold-trn22-r1-packet/train/caches/compiler","TORCH_NEURONX_NEFF_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch3e79-cold-trn22-r1-packet/train/caches/neff","TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch3e79-cold-trn22-r1-packet/train/caches/local","TORCH_NEURONX_HLO_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch3e79-cold-trn22-r1-packet/train/caches/hlo","TORCHINDUCTOR_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch3e79-cold-trn22-r1-packet/train/caches/inductor","NEURON_COMPILE_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch3e79-cold-trn22-r1-packet/train/caches/compiler","TORCH_NEURONX_DEBUG_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch3e79-cold-trn22-r1-packet/train/caches/debug"}
EVAL_ENV = {"HOME":"/home/ubuntu","NEURON_CC_FLAGS":"--optlevel=1","NEURON_COMPETITION_R1_CACHE_DIR":"/home/ubuntu/.cache/neuron_competition_r1","NEURON_COMPETITION_R1_SEED":"42","NEURON_LOGICAL_NC_CONFIG":"2","NKI_VENV_PATH":"/home/ubuntu/workspace/native_venv","PATH":"/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin","PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","TRN_EXP_LABEL":"fs-20260926-arch3e79-base-control-trn22-r1-full","NEURON_COMPILE_CACHE_URL":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/eval/compiler","TORCH_NEURONX_NEFF_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/eval/neff","TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/eval/local","TORCH_NEURONX_HLO_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/eval/hlo","TORCHINDUCTOR_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/eval/inductor","NEURON_COMPILE_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/eval/compiler","TORCH_NEURONX_DEBUG_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/eval/debug"}

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
try:
    preflight(TRAIN_ENV)
    assert not os.path.lexists(ROOT/'train'),'TRAIN_SANDBOX_ALREADY_EXISTS'
    assert not os.path.lexists(ROOT/'eval'),'EVAL_SANDBOX_ALREADY_EXISTS'
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
