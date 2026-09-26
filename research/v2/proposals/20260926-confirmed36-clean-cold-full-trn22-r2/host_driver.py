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

ROOT = Path("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet")
PREP = Path("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py")
HARNESS = Path("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/research/scratch/submission_rehearsal.py")
LABEL = "fs-20260926-confirmed36-clean-coldfull-trn22-r2"
SOURCE_SHA = "e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43"
PREP_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
HARNESS_SHA = "7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b"
TRAIN_COMMAND = ["/usr/bin/timeout","--foreground","--signal=TERM","--kill-after=60s","2730s","/home/ubuntu/workspace/native_venv/bin/python","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/research/scratch/submission_rehearsal.py","--package","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet/package","--sandbox","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet/train","--mode","train","--fresh-caches"]
EVAL_COMMAND = ["/usr/bin/timeout","--foreground","--signal=TERM","--kill-after=60s","1800s","/home/ubuntu/workspace/native_venv/bin/python","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/research/scratch/submission_rehearsal.py","--package","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet/package","--sandbox","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet/eval","--mode","eval","--checkpoint","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet/train/out/final.pt","--eval-tokens","20971520","--fresh-caches"]
TRAIN_ENV = {"HOME":"/home/ubuntu","NEURON_CC_FLAGS":"--optlevel=1 --auto-cast matmult --auto-cast-type bf16","NEURON_COMPETITION_R1_CACHE_DIR":"/home/ubuntu/.cache/neuron_competition_r1","NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS":"1800","NEURON_COMPETITION_R1_SEED":"42","NEURON_LOGICAL_NC_CONFIG":"2","NKI_VENV_PATH":"/home/ubuntu/workspace/native_venv","PATH":"/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin","PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","TORCH_NEURONX_ENABLED_METRIC_TABLES":"graph_stats","TORCH_NEURONX_METRICS_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet/metrics-training","TRN_EXP_LABEL":"fs-20260926-confirmed36-clean-coldfull-trn22-r2","NEURON_COMPILE_CACHE_URL":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet/train/caches/compiler"}
EVAL_ENV = {"HOME":"/home/ubuntu","NEURON_CC_FLAGS":"--optlevel=1","NEURON_COMPETITION_R1_CACHE_DIR":"/home/ubuntu/.cache/neuron_competition_r1","NEURON_COMPETITION_R1_SEED":"42","NEURON_LOGICAL_NC_CONFIG":"2","NKI_VENV_PATH":"/home/ubuntu/workspace/native_venv","PATH":"/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin","PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","TRN_EXP_LABEL":"fs-20260926-confirmed36-clean-coldfull-trn22-r2","NEURON_COMPILE_CACHE_URL":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet/eval/caches/compiler","TORCH_NEURONX_ENABLED_METRIC_TABLES":"graph_stats","TORCH_NEURONX_METRICS_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet/metrics-eval"}

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
    minimum_gib = 13.5 if env is TRAIN_ENV else 10.5
    assert v.f_bavail*v.f_frsize>=minimum_gib*1024**3,'DISK_HEADROOM'
    assert sha(ROOT/'package/train.py')==SOURCE_SHA,'SOURCE_SHA'
    assert sha(ROOT/'package/launch-command.txt')=='b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd','LAUNCH_SHA'
    assert sha(ROOT/'package/provenance.json')=='dcec405e7b4ee2fd78d60382aaebadc0ab0f114c6c6abfae11bc2efea236683f','PROVENANCE_SHA'
    assert sha(PREP)==PREP_SHA,'PREPARE_SHA'
    assert sha(HARNESS)==HARNESS_SHA,'HARNESS_SHA'
    # The pinned rehearsal creates these six SDK roots fresh before its training/eval child.
    phase_root = ROOT/('train' if env is TRAIN_ENV else 'eval')
    assert not os.path.lexists(phase_root),'PHASE_SANDBOX_ALREADY_EXISTS'
    assert env['NEURON_COMPILE_CACHE_URL']==str(phase_root/'caches/compiler'),'FRESH_URL_BINDING'
def phase(name,command,env,guardian):
    record(name+'-launch.json',{'utc':utc(),'label':LABEL,'command':command,'environment':env,'source_sha256':SOURCE_SHA,'prepare_sha256':PREP_SHA,'harness_sha256':HARNESS_SHA})
    with (ROOT/(name+'.log')).open('x') as out:
        out.write(name.upper()+'_START_UTC='+utc()+'\n'); out.flush()
        rc=guardian.run(command,cwd=ROOT,env=env,out=out)
        out.write(name.upper()+'_END_UTC='+utc()+'\n'); out.flush()
    with (ROOT/('run-exit-code.txt' if name=='training' else 'eval-exit-code.txt')).open('x') as f: f.write(str(rc)+'\n')
    return rc
def device_clear(stage):
    if not Path('/dev/neuron0').is_char_device():
        raise RuntimeError(stage.upper()+'_DEVICE_NODE_MISSING')
    holders=subprocess.run(['/usr/bin/fuser','/dev/neuron0'],capture_output=True,text=True,timeout=15,env=guard.CHECK_ENV)
    record(stage+'-fuser.json',{'returncode':holders.returncode,'stdout':holders.stdout,'stderr':holders.stderr})
    if holders.returncode!=1 or holders.stdout.strip() or holders.stderr.strip():
        raise RuntimeError(stage.upper()+'_DEVICE_HOLDERS_OR_CHECK_FAILED')
def postflight():
    if not guardian.job_started:
        record('postflight-no-job.json',{'utc':utc(),'status':'NO_JOB_STARTED','external_idle_not_claimed':True})
        return  # A rejected initial idle/disk check must not wait for someone else's job.
    guard.require_idle_under_inherited_lock('/home/ubuntu/.cache/trainium_freshstart/device0.lock',ROOT/'host_driver.py',lambda name,obj:record('postflight-'+name,obj))
    device_clear('postflight')
    checkpoint=ROOT/'train/out/final.pt'
    if checkpoint.is_file():
        digest=sha(checkpoint)
        record('checkpoint-after-postflight.json',{'utc':utc(),'sha256':digest,'bytes':checkpoint.stat().st_size})
        if (ROOT/'checkpoint.json').is_file() and digest!=json.loads((ROOT/'checkpoint.json').read_text())['sha256']:
            raise RuntimeError('CHECKPOINT_CHANGED_AFTER_TRAINING')

guardian=None
exit_code=1
try:
    if sys.flags.optimize:
        raise RuntimeError('OPTIMIZED_DRIVER_FORBIDDEN')
    assert not os.path.lexists(ROOT/'train'),'TRAIN_SANDBOX_ALREADY_EXISTS'
    assert not os.path.lexists(ROOT/'eval'),'EVAL_SANDBOX_ALREADY_EXISTS'
    assert sha(ROOT/'dispatch_driver.py')=='c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a','DISPATCH_SHA'
    assert sha(ROOT/'idle_guard.py')=='adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737','IDLE_GUARD_SHA'
    spec=importlib.util.spec_from_file_location('bound_idle_guard',ROOT/'idle_guard.py')
    guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    assert sha(ROOT/'lifetime_guard.py')=='e461ec5c76c784fd51471c9e8c268cdde42fe1def68b31bfc9524480f980c312','LIFETIME_GUARD_SHA'
    spec=importlib.util.spec_from_file_location('bound_lifetime_guard',ROOT/'lifetime_guard.py')
    lifetime=importlib.util.module_from_spec(spec); spec.loader.exec_module(lifetime)
    # Authenticate the original inode at the FINAL lifetime acquisition, before
    # idle checks or training, not only during the earlier staging acquisition.
    lock=guard.inherited_lock('/home/ubuntu/.cache/trainium_freshstart/device0.lock',ROOT/'host_driver.py')
    lifetime.require_original_lock(lock['fd'],lock['path'],(66305,832589))
    guardian=lifetime.Guardian(lock['fd'],lock['path'],(66305,832589),ROOT/'lifetime-events.jsonl')
    lock_fd=guard.require_idle_under_inherited_lock('/home/ubuntu/.cache/trainium_freshstart/device0.lock',ROOT/'host_driver.py',record)
    device_clear('training-preflight')
    preflight(TRAIN_ENV)
    lifetime.require_original_lock(lock_fd,lock['path'],(66305,832589))
    rc=phase('training',TRAIN_COMMAND,TRAIN_ENV,guardian)
    exit_code=rc if rc>=0 else 128-rc
    assert rc==0,'TRAIN_EXIT_'+str(rc)
    rehearsal=json.loads((ROOT/'train/rehearsal.json').read_text())
    assert rehearsal['returncode']==0 and rehearsal['charged_through_process_exit_within_budget'] is True,'TRAIN_BUDGET_FAILED'
    assert (ROOT/'train/out/final.pt').is_file(),'MISSING_CHECKPOINT'
    record('checkpoint.json',{'utc':utc(),'sha256':rehearsal['checkpoint_sha256'],'bytes':rehearsal['checkpoint_bytes']})
    eval_fd=guard.require_idle_under_inherited_lock('/home/ubuntu/.cache/trainium_freshstart/device0.lock',ROOT/'host_driver.py',lambda name,obj:record('eval-'+name,obj))
    assert eval_fd==lock_fd,'LOCK_FD_CHANGED'
    device_clear('eval-preflight')
    preflight(EVAL_ENV)
    lifetime.require_original_lock(eval_fd,lock['path'],(66305,832589))
    rc=phase('fresh-public20m',EVAL_COMMAND,EVAL_ENV,guardian)
    exit_code=rc if rc>=0 else 128-rc
    record('terminal.json',{'utc':utc(),'label':LABEL,'status':'COMPLETE_PENDING_INDEPENDENT_REVIEW' if rc==0 else 'EVAL_FAILED','eval_exit_code':rc,'no_automatic_next_job':True,'submission_authorized':False})
except BaseException as exc:
    if exit_code==0:
        exit_code=1
    if guardian is not None and guardian.signal_received is not None:
        exit_code=128+guardian.signal_received
    record('terminal.json',{'utc':utc(),'label':LABEL,'status':'FULL_DRIVER_FAILED','error':str(exc),'no_automatic_next_job':True})
finally:
    if guardian is not None:
        guardian.finish(postflight)
sys.exit(exit_code)
