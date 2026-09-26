"""Ten sequential tiny CPU G2 arms; no native/host operation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SUPPORT=Path('/home/poojith-devan/Desktop/Trainium')
ARGV=['--n-embd','128','--depth','8','--unet-skip','--partial-attn-8','--attention-layers','0,2,4,6,7','--initial-cache-refresh','--no-ngram-ve','--no-qk-shift','--compile-sdpa-direct','--nki-local-conv','--bf16-norm-output','--pack-factor','32','--clip-after-reduce','--seq-len','1024','--no-eval-public','--matrix-lr','0.015','--seed','42','--max-train-seconds','1800']
if __name__=='__main__':
    from test_static import BASE, SOURCE_SHA
    candidate=ROOT/'workflow/proposals/research-v2/train.py'
    if hashlib.sha256(candidate.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise RuntimeError('candidate pin')
    if next(int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))<6*1024**2:
        raise RuntimeError('requires 6 GiB available; no fullsize fallback')
    tmp=Path(tempfile.mkdtemp(prefix='clock-recursive-source-'))
    baseline=tmp/'train.py'
    baseline.write_text(BASE)
    raw=HERE/'raw'
    raw.mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(SUPPORT/'research/v2/lanes/architecture-recursive-stack/test_support'))
    commands=[]
    for clock in (False,True):
        for arm in ('base','default','off','on','reference'):
            name=f'clock{int(clock)}-{arm}'
            out=raw/f'{name}.json'
            if out.exists():
                raise RuntimeError('refuse overwrite: '+str(out))
            flags=['--startup-adjusted-schedule'] if clock else ['--no-startup-adjusted-schedule']
            if arm=='off': flags+=['--no-recursive-schedule']
            if arm in ('on','reference'): flags+=['--recursive-schedule']
            source=baseline if arm=='base' else candidate
            command=[sys.executable,'-B',str(HERE/'g2_observe.py'),'--train-py',str(source),'--stub-prepare','--steps','6','--nproc','4','--micro-rows','2','--micro-tokens','128','--timeout','150','--out',str(out),'--']+ARGV+flags
            task_env=dict(env,CLOCK_RECURSIVE_REFERENCE='1' if arm=='reference' else '0')
            commands.append(dict(name=name,argv=command,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),env={k:task_env[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONPATH','CLOCK_RECURSIVE_REFERENCE')}))
            (raw/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            print('Starting '+name,flush=True)
            subprocess.run(command,env=task_env,cwd=ROOT,check=True,timeout=180)
