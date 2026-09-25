"""Bounded CT startup-only binding of the unchanged Sep24 rehearsal harness."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import socket
import sys

ROOT = Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-archclean3f50-cold-trn22-r1-packet')
PREP = Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py')
HARNESS = Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/research/scratch/submission_rehearsal.py')
LABEL = 'fs-20260926-ct-archclean3f50-cold-trn22-r1'
SOURCE_SHA = '3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f'
PREP_SHA = '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb'
HARNESS_SHA = '7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b'
COMMAND = ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=60s', '1800s', '/home/ubuntu/workspace/native_venv/bin/python', '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/research/scratch/submission_rehearsal.py', '--package', '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-archclean3f50-cold-trn22-r1-packet/package', '--sandbox', '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-archclean3f50-cold-trn22-r1-packet/train', '--mode', 'train', '--fresh-caches']
ENV = {'HOME': '/home/ubuntu', 'NEURON_CC_FLAGS': '--optlevel=1 --auto-cast matmult --auto-cast-type bf16', 'NEURON_COMPETITION_R1_CACHE_DIR': '/home/ubuntu/.cache/neuron_competition_r1', 'NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS': '1800', 'NEURON_COMPETITION_R1_SEED': '42', 'NEURON_LOGICAL_NC_CONFIG': '2', 'NKI_VENV_PATH': '/home/ubuntu/workspace/native_venv', 'PATH': '/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'TORCH_NEURONX_ENABLED_METRIC_TABLES': 'graph_stats', 'TORCH_NEURONX_METRICS_DIR': '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-archclean3f50-cold-trn22-r1-packet/metrics-training', 'TRN_EXP_LABEL': 'fs-20260926-ct-archclean3f50-cold-trn22-r1', 'NEURON_COMPILE_CACHE_URL': '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-archclean3f50-cold-trn22-r1-packet/train/caches/compiler'}

def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def record(name, obj):
    with (ROOT/name).open('x') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n')
try:
    if sys.flags.optimize: raise RuntimeError('OPTIMIZED_DRIVER_FORBIDDEN')
    assert socket.gethostname() == 'ip-172-31-32-253', 'HOSTNAME'
    for base in (Path('/home/ubuntu'),PREP.parent,ROOT,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
        for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):
            assert not os.path.lexists(base/rel), 'STOP: '+str(base/rel)
    v=os.statvfs('/')
    assert v.f_bavail*v.f_frsize>=6.5*1024**3, 'DISK_BELOW_6_5_GIB'
    assert sha(ROOT/'package/train.py')==SOURCE_SHA,'SOURCE_SHA'
    assert sha(ROOT/'package/launch-command.txt')=='2cebf5a4e51436739bb4c293915ea07184484824478301e9246074319874f9f6','LAUNCH_SHA'
    assert sha(ROOT/'package/provenance.json')=='d40a3437902531b563a0db1c8bfc2a8fd07d0f3b7b4071b787680a0505d574c2','PROVENANCE_SHA'
    assert sha(ROOT/'dispatch_driver.py')=='c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a','DISPATCH_SHA'
    assert ENV['NEURON_COMPILE_CACHE_URL']==str(ROOT/'train/caches/compiler'),'COMPILER_URL'
    assert sha(PREP)==PREP_SHA,'PREPARE_SHA'
    assert sha(HARNESS)==HARNESS_SHA,'HARNESS_SHA'
    assert not os.path.lexists(ROOT/'train'),'SANDBOX_ALREADY_EXISTS'
    assert sha(ROOT/'idle_guard.py')=='adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737','IDLE_GUARD_SHA'
    spec=importlib.util.spec_from_file_location('bound_idle_guard',ROOT/'idle_guard.py')
    guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    lock_fd=guard.require_idle_under_inherited_lock('/home/ubuntu/.cache/trainium_freshstart/device0.lock',ROOT/'host_driver.py',record)
    record('training-launch.json',{'utc':utc(),'label':LABEL,'command':COMMAND,'environment':ENV,'source_sha256':SOURCE_SHA,'prepare_sha256':PREP_SHA,'harness_sha256':HARNESS_SHA})
    with (ROOT/'training.log').open('x') as out:
        out.write('HARNESS_START_UTC='+utc()+'\n'); out.flush()
        rc=subprocess.call(COMMAND,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,pass_fds=(lock_fd,))
        out.write('HARNESS_END_UTC='+utc()+'\n'); out.flush()
    with (ROOT/'run-exit-code.txt').open('x') as f: f.write(str(rc)+'\n')
    record('terminal.json',{'utc':utc(),'label':LABEL,'status':'CT_HARNESS_EXITED' if rc==0 else 'CT_HARNESS_FAILED','returncode':rc,'no_eval':True,'no_automatic_full':True})
    sys.exit(rc)
except Exception as exc:
    record('terminal.json',{'utc':utc(),'label':LABEL,'status':'CT_DRIVER_FAILED','error':str(exc),'no_eval':True,'no_automatic_full':True})
    raise
