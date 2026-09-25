"""Host-resident exact 93ae architecture train then fresh public-20M eval; science bytes unchanged."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/ubuntu/trainium_l2_unet8x1024_93ae_20260925_trn22_r1/native/fs-20260925-l2-unet8x1024-93ae-trn22-full-r1')
PREP = Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py')
TRAIN_SHA = '93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5'
PREP_SHA = '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb'
LABEL = 'fs-20260925-l2-unet8x1024-93ae-trn22-full-r1'
VENV = '/home/ubuntu/workspace/native_venv/bin/'
CACHE = Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train')
EVAL_CACHE = Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/eval')
FLAGS = ['--n-embd', '1024', '--no-ngram-ve', '--no-qk-shift', '--compile-sdpa-direct', '--nki-local-conv', '--bf16-norm-output', '--pack-factor', '32', '--clip-after-reduce', '--seq-len', '1024', '--no-eval-public', '--matrix-lr', '0.015', '--seed', '42', '--max-train-seconds', '1800', '--num-steps', '1000000', '--out-dir', 'out', '--unet-8x1024', '--attention-layers', '0,2,4,6,7']

def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def record(name,value):
    with (ROOT/name).open('x') as f: json.dump(value,f,indent=2,sort_keys=True); f.write('\n')
def stop_check():
    for base in (Path('/home/ubuntu'),PREP.parent,ROOT,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
        for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):
            assert not os.path.lexists(base/rel), 'STOP: '+str(base/rel)
    st=os.statvfs('/')
    assert st.f_bavail*st.f_frsize>=6.5*1024**3,'DISK_BELOW_6_5_GIB'
    assert sha(ROOT/'train.py')==TRAIN_SHA,'TRAIN_SHA'
    assert sha(PREP)==PREP_SHA,'PREPARE_SHA'
def environment(cache):
    env={'HOME':'/home/ubuntu','PATH':VENV.rstrip('/')+':/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin','NKI_VENV_PATH':'/home/ubuntu/workspace/native_venv','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(PREP.parent),'NEURON_COMPETITION_R1_CACHE_DIR':'/home/ubuntu/.cache/neuron_competition_r1','NEURON_LOGICAL_NC_CONFIG':'2','NEURON_CC_FLAGS':'--optlevel=1 --auto-cast matmult --auto-cast-type bf16','NEURON_COMPETITION_R1_SEED':'42','TRN_EXP_LABEL':LABEL}
    for key,name in [('TORCH_NEURONX_NEFF_CACHE_DIR','neff'),('TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','local'),('TORCH_NEURONX_HLO_CACHE_DIR','hlo'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('NEURON_COMPILE_CACHE_DIR','compiler'),('TORCH_NEURONX_DEBUG_DIR','debug')]:
        assert (cache/name).is_dir(),'CACHE_MISSING: '+str(cache/name)
        env[key]=str(cache/name)
    return env
def phase(name,argv,env):
    record(name+'-launch.json',{'utc':utc(),'label':LABEL,'command':argv,'environment':env,'train_sha256':TRAIN_SHA,'prepare_sha256':PREP_SHA})
    with (ROOT/(name+'.log')).open('x') as out:
        out.write(name.upper()+'_START_UTC='+utc()+'\n'); out.flush()
        rc=subprocess.call(['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','5400s' if name=='training' else '1800s',*argv],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
        out.write(name.upper()+'_END_UTC='+utc()+'\n'); out.flush()
    with (ROOT/('run-exit-code.txt' if name=='training' else 'eval-exit-code.txt')).open('x') as f: f.write(str(rc)+'\n')
    return rc

try:
    stop_check()
    env=environment(CACHE)
    env.update({'NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS':'1800','NEURON_COMPETITION_R1_OUT_DIR':str(ROOT/'out')})
    rc=phase('training',[VENV+'torchrun','--standalone','--nproc_per_node=4','train.py',*FLAGS],env)
    if rc: raise RuntimeError('TRAIN_EXIT_'+str(rc))
    stop_check()
    ckpt=ROOT/'out/final.pt'
    assert ckpt.is_file(),'MISSING_CHECKPOINT'
    record('checkpoint.json',{'sha256':sha(ckpt),'bytes':ckpt.stat().st_size,'utc':utc()})
    rc=phase('fresh-public20m',[VENV+'python',str(PREP),'eval-public','--train-py',str(ROOT/'train.py'),'--checkpoint',str(ckpt),'--device','neuron','--batch-size','1','--seq-len','1024','--eval-tokens','20971520','--timeout-seconds','1200'],environment(EVAL_CACHE))
    record('terminal.json',{'utc':utc(),'status':'COMPLETE' if rc==0 else 'EVAL_FAILED','eval_exit_code':rc,'label':LABEL})
    sys.exit(rc)
except Exception as exc:
    record('terminal.json',{'utc':utc(),'status':'FAILED','error':str(exc),'label':LABEL})
    raise
