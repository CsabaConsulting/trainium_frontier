"""One bounded P actual-loop profile, exact reviewed source/wrapper binding only."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/ubuntu/trainium_profile_recursive6ef7_20260926_trn21_r1')
PREP = Path('/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/prepare.py')
PROFILE_API = Path('/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx/profiling.py')
LABEL = 'fs-20260926-profile-recursive6ef7-trn21-r1'
SOURCE_SHA = '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'
WRAPPER_SHA = 'eaad6aff2e18a825eb8ddca91aed5efa72c91a385eb02cbdaee36b01c2644f7d'
PREP_SHA = '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb'
API_SHA = 'ac8a9a43e5d8326cf532dfd2916cf6b7d7833d3a1d67e8ebe0e075aa1dfea3d6'
COMMAND = ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=60s', '2640s', '/home/ubuntu/workspace/native_venv/bin/torchrun', '--standalone', '--nproc_per_node=4', '/home/ubuntu/trainium_profile_recursive6ef7_20260926_trn21_r1/profile_actual_loop.py', '--train-py', '/home/ubuntu/trainium_profile_recursive6ef7_20260926_trn21_r1/train.py', '--capture', '--out', '/home/ubuntu/trainium_profile_recursive6ef7_20260926_trn21_r1/profile', '--target-step', '20', '--device-ranks', '0', '--capture-nc', '0-7', '--max-events-per-nc', '200000', '--', '--n-embd', '1024', '--no-ngram-ve', '--no-qk-shift', '--compile-sdpa-direct', '--nki-local-conv', '--bf16-norm-output', '--pack-factor', '32', '--clip-after-reduce', '--seq-len', '1024', '--no-eval-public', '--matrix-lr', '0.015', '--seed', '42', '--max-train-seconds', '1800', '--num-steps', '22', '--out-dir', '/home/ubuntu/trainium_profile_recursive6ef7_20260926_trn21_r1/out', '--recursive-schedule']
ENV = {'HOME': '/home/ubuntu', 'PATH': '/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin', 'NKI_VENV_PATH': '/home/ubuntu/workspace/native_venv', 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': '/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2', 'NEURON_COMPETITION_R1_CACHE_DIR': '/home/ubuntu/.cache/neuron_competition_r1', 'NEURON_LOGICAL_NC_CONFIG': '2', 'NEURON_CC_FLAGS': '--optlevel=1 --auto-cast matmult --auto-cast-type bf16', 'NEURON_COMPETITION_R1_SEED': '42', 'NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS': '1800', 'NEURON_COMPETITION_R1_OUT_DIR': '/home/ubuntu/trainium_profile_recursive6ef7_20260926_trn21_r1/out', 'TRN_EXP_LABEL': 'fs-20260926-profile-recursive6ef7-trn21-r1', 'TORCH_NEURONX_ENABLED_METRIC_TABLES': 'graph_stats', 'TORCH_NEURONX_METRICS_DIR': '/home/ubuntu/trainium_profile_recursive6ef7_20260926_trn21_r1/metrics-training', 'TORCH_NEURONX_NEFF_CACHE_DIR': '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/neff', 'TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR': '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/local', 'TORCH_NEURONX_HLO_CACHE_DIR': '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/hlo', 'TORCHINDUCTOR_CACHE_DIR': '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/inductor', 'NEURON_COMPILE_CACHE_DIR': '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/compiler', 'TORCH_NEURONX_DEBUG_DIR': '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/debug'}

def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def record(name,obj):
    with (ROOT/name).open('x') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n')
try:
    for base in (Path('/home/ubuntu'),PREP.parent,ROOT,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
        for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):
            assert not os.path.lexists(base/rel),'STOP: '+str(base/rel)
    v=os.statvfs('/')
    assert v.f_bavail*v.f_frsize>=10.5*1024**3,'DISK_HEADROOM'
    assert sha(ROOT/'train.py')==SOURCE_SHA,'SOURCE_SHA'
    assert sha(ROOT/'profile_actual_loop.py')==WRAPPER_SHA,'WRAPPER_SHA'
    assert sha(PREP)==PREP_SHA,'PREPARE_SHA'
    assert sha(PROFILE_API)==API_SHA,'PROFILING_API_SHA'
    assert (ROOT/'profile').is_dir() and not any((ROOT/'profile').iterdir()),'PROFILE_PARENT_NOT_FRESH'
    assert not os.path.lexists(ROOT/'out'),'OUT_ALREADY_EXISTS'
    for key in ('TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'):
        assert Path(ENV[key]).is_dir(),'CACHE_MISSING:'+key
    assert sha(ROOT/'idle_guard.py')=='5d735b4baf1b68bab2e9528adc4b1d09a930cb2beafdd069f1a6a513f65c34fb','IDLE_GUARD_SHA'
    spec=importlib.util.spec_from_file_location('bound_idle_guard',ROOT/'idle_guard.py')
    guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    guard.require_idle_under_inherited_lock('/home/ubuntu/.cache/trainium_freshstart/device0.lock',ROOT/'host_driver.py',record)
    record('profile-launch.json',{'utc':utc(),'label':LABEL,'command':COMMAND,'environment':ENV,'source_sha256':SOURCE_SHA,'wrapper_sha256':WRAPPER_SHA,'prepare_sha256':PREP_SHA,'no_eval':True})
    with (ROOT/'profile.log').open('x') as out:
        out.write('PROFILE_START_UTC='+utc()+'\n'); out.flush()
        rc=subprocess.call(COMMAND,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
        out.write('PROFILE_END_UTC='+utc()+'\n'); out.flush()
    with (ROOT/'run-exit-code.txt').open('x') as f: f.write(str(rc)+'\n')
    record('terminal.json',{'utc':utc(),'label':LABEL,'status':'PROFILE_PROCESS_EXITED_PENDING_ARTIFACT_REVIEW' if rc==0 else 'PROFILE_PROCESS_FAILED','returncode':rc,'quality_evidence':False,'no_eval':True,'no_automatic_next_job':True})
    sys.exit(rc)
except Exception as exc:
    record('terminal.json',{'utc':utc(),'label':LABEL,'status':'PROFILE_DRIVER_FAILED','error':str(exc),'quality_evidence':False,'no_eval':True})
    raise
