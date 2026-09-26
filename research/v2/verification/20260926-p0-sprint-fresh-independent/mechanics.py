import ast, fcntl, hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path
ROOT=Path('/home/poojith-devan/p0-sprint-fresh-review-8kEym4')
P=ROOT/'research/v2/proposals'
OUT=Path(__file__).parent
variants=sorted(P.glob('*/variants/*'))
assert len(variants)==6
for v in variants:
    b=json.loads((v/'NATIVE_BINDING.disabled.json').read_text())
    tree=ast.parse((v/'host_driver.py').read_text())
    values={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try: values[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError): pass
    for upper,lower in [('LABEL','label'),('TRAIN_COMMAND','train_command'),('EVAL_COMMAND','eval_command'),('TRAIN_ENV','train_environment'),('EVAL_ENV','eval_environment'),('SOURCE_SHA','source_sha256')]:
        assert values[upper]==b[lower],(v,upper)
    assert hashlib.sha256((v/'package/train.py').read_bytes()).hexdigest()==b['source_sha256']
    assert (v/'package/train.py').read_bytes()==(v.parent.parent/'train.py').read_bytes()
    assert b['budget']['source_seconds']==1800 and b['budget']['startup_cap_seconds']==900
    assert b['train_environment']['NEURON_COMPETITION_R1_SEED']=='42'
    assert b['launch_authorized'] is False and b['release_file_created'] is False
    assert '--eval-tokens' in b['eval_command'] and b['eval_command'][-1]=='20971520'
    assert not (v/'RELEASE.json').exists()
    for f in v.glob('*.py'): compile(f.read_text(),str(f),'exec')
print('Six frozen bindings: internal command/environment/source/budget consistency PASS')

v=P/'20260926-sprint-p0-clean-recursive/variants/recursive-trn21'
spec=importlib.util.spec_from_file_location('bounded_guard',v/'lifetime_guard.py')
gm=importlib.util.module_from_spec(spec); spec.loader.exec_module(gm)
path=OUT/'mechanics.lock'; fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600); fcntl.flock(fd,fcntl.LOCK_EX)
st=os.fstat(fd); expected=(st.st_dev,st.st_ino)
# Deliberately unavailable receipt path exercises constructor failure too.
g=gm.Guardian(fd,str(path),expected,OUT/'nonexistent-parent'/'receipt',grace=.1,kill_wait=.1)
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
events=[]
assert g.finish(lambda:events.append('report'),lambda:events.append('idle')) is False
assert events==['report','idle'] and child.pid in g.statuses
gm.require_original_lock(fd,str(path),expected)
print('Guardian missing receipt + live owned child: child reaped, independent idle called, failed return, held lock PASS')

class Parked(Exception): pass
def park(message): raise Parked(message)
for reason in ('BUSY','UNKNOWN'):
    g=gm.Guardian(fd,str(path),expected,OUT/('guardian-'+reason+'.jsonl'),grace=.1,kill_wait=.1)
    g.park=park
    def bad_idle(): raise RuntimeError(reason)
    try: g.finish(lambda:None,bad_idle)
    except Parked as exc: assert reason in str(exc)
    else: raise AssertionError('unsafe release on '+reason)
    gm.require_original_lock(fd,str(path),expected)
    g.receipt.close()
print('Guardian busy/unknown independent idle: park path reached with original lock held PASS')
os.close(fd)
