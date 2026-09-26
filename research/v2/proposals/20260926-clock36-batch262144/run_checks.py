"""Author-only tiny CPU receipts, never host/native qualification."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / ('observed-receipts' if os.environ.get('OBSERVE') == '1' else 'receipts')
OUT.mkdir(exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RUN_BATCH_TENSOR_CHECKS='1', PYTHONPATH=str(ROOT/'research/v2/lanes/recursive-schedule/test_support'))
argv = ['--n-embd','128','--depth','8','--unet-skip','--partial-attn-8','--attention-layers','0,2,4,6,7','--no-ngram-ve','--no-qk-shift','--compile-sdpa-direct','--nki-local-conv','--bf16-norm-output','--pack-factor','32','--clip-after-reduce','--seq-len','1024','--no-eval-public','--matrix-lr','0.015','--seed','42','--max-train-seconds','1800','--initial-cache-refresh','--startup-adjusted-schedule']
records=[]
def run(label, args, optimized):
    selected_env = dict(env, PYTHONOPTIMIZE='1' if optimized else '0')
    cmd=[sys.executable]+(['-O'] if optimized else [])+args
    p=subprocess.run(cmd,cwd=ROOT,env=selected_env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
    (OUT/(label+'.stdout')).write_text(p.stdout)
    (OUT/(label+'.stderr')).write_text(p.stderr)
    records.append({'label':label,'argv':cmd,'exit_code':p.returncode,'optimized':optimized})
    (OUT/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
    print(label,p.returncode,flush=True)
for optimized in (False,True):
    suffix='optimized' if optimized else 'normal'
    run('unit-'+suffix,[str(HERE/'test_batch.py')],optimized)
    run('cache-'+suffix,[str(HERE/'test_cache.py')],optimized)
    for arm, source, delta in [('parent','parent.py',[]),('default','train.py',[]),('batch262144','train.py',['--total-batch-size','262144'])]:
        label=arm+'-'+suffix
        driver = HERE/'observe_tiny.py' if os.environ.get('OBSERVE') == '1' else ROOT/'workflow/tools/preflight_dryrun.py'
        run(label,[str(driver),'--train-py',str(HERE/source),'--stub-prepare','--steps','3','--nproc','4','--micro-rows','2','--micro-tokens','128','--timeout','180','--out',str(OUT/(label+'.json')),'--']+argv+delta,optimized)
    run('identity-'+suffix,[str(ROOT/'workflow/tools/preflight_dryrun.py'),'--compare',str(OUT/('parent-'+suffix+'.json')),str(OUT/('default-'+suffix+'.json'))],optimized)
(OUT/'source.patch').write_text(''.join(difflib.unified_diff((HERE/'parent.py').read_text().splitlines(True),(HERE/'train.py').read_text().splitlines(True),fromfile='exact36/train.py',tofile='batch-port/train.py')))
(OUT/'sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'parent.py',HERE/'train.py',HERE/'test_batch.py',HERE/'test_cache.py',Path(__file__),ROOT/'workflow/tools/preflight_dryrun.py',ROOT/'workflow/tools/stub_prepare/prepare.py']},indent=2)+'\n')
