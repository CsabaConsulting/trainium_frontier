import ast, hashlib, json, math, sys, types, importlib.util, os, fcntl
from pathlib import Path
import torch

ROOT = Path('/home/poojith-devan/p0-sprint-fresh-review-8kEym4')
P = ROOT/'research/v2/proposals'
OUT = Path(__file__).parent
torch.set_num_threads(1)
manifest = json.loads((ROOT/'MANIFEST.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p,h in manifest.items())
print('Manifest: all', len(manifest), 'file digests match', flush=True)

def module(path, name):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'prepare') and not (isinstance(n, ast.Import) and any(a.name == 'pyarrow.parquet' for a in n.names))]
    m = types.ModuleType(name); sys.modules[name] = m
    m.__dict__.update(print0=lambda *a: None, TOKENIZER_VOCAB_SIZE=256, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900, TRAIN_STARTUP_STEPS_EXCLUDED=10, TRAIN_TIME_BUDGET_SECONDS=1800)
    exec(compile(tree,str(path),'exec'),m.__dict__)
    return m

base = module(P/'20260926-sprint-p0-clean-recursive/base.py','review_base')
rec = module(P/'20260926-sprint-p0-clean-recursive/train.py','review_rec')
batch = module(P/'20260926-sprint-p0-clean-batch/train.py','review_batch')

# Independent elementary scalar reference, written from the schedule specification.
rec.RECURSIVE_SCHEDULE = True
cases = 0
for step in (0,1,299,300,1000):
    for p in (-0.1,0,0.05,0.2,0.35,0.9,1,1.1):
        q = max(0., min(1.,p))
        for kind, ratio in [('muon',0.95),('adamw',0.65)]:
            cooldown = (1.-q)/ratio
            expected = 1. if q < 1.-ratio else cooldown + (1.-cooldown)*0.05
            assert rec.lr_multiplier(step,p*1800,1800,kind) == expected
        expected_beta = 0.95 + max(0.,(q-(1.-0.95))/0.95)*(0.97-0.95)
        assert rec.muon_beta2(p) == expected_beta
        frac = min(step/300,1.)
        expected_mom = (1-frac)*0.85+frac*0.95
        if q > 1.-0.95:
            expected_mom = 0.95 + ((q-(1.-0.95))/0.95)**2*(0.79-0.95)
        assert rec.muon_momentum(step,p) == expected_mom
        cases += 1
print('Independent scalar reference exact:',cases,'cases', flush=True)

# Check the changed tensor beta2 input against a scalar plain-reference update.
torch.manual_seed(42)
one = torch.tensor(0.05,dtype=torch.float32)
second = torch.rand(2,7,1); vmean=torch.rand(2,7,1)
assert torch.equal(torch.lerp(second,vmean,one),torch.lerp(second,vmean,0.05))
print('Changed scalar-to-tensor beta2 lerp: exact', flush=True)

# Actual reduced model + optimizer, identical token batches and 3 steps.
rec.RECURSIVE_SCHEDULE = False
def run(m):
    torch.manual_seed(42)
    model=m.GPT(m.GPTConfig(sequence_len=32,vocab_size=256,n_layer=8,n_head=1,n_embd=128))
    model.init_weights(); optimizer=model.setup_optimizer()
    result=[]
    for step,p in enumerate((0.,0.5,0.98)):
        x=torch.randint(0,256,(1,32)); y=torch.randint(0,256,(1,32))
        optimizer.zero_grad(set_to_none=True)
        loss=model(x,y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        lrm=m.lr_multiplier(step,p*1800,1800)
        for group in optimizer.param_groups:
            group['lr']=group['initial_lr']*lrm
            if group['kind']=='muon':
                group['momentum']=m.muon_momentum(step,p)
                group['weight_decay']=m.WEIGHT_DECAY*(1.-p)
                group['beta2']=0.9
        optimizer.step()
        result.append((loss.detach().clone(),{k:v.detach().clone() for k,v in model.state_dict().items()}))
    return result
runs=[run(m) for m in (base,rec,batch)]
for other in runs[1:]:
    for (loss,state),(loss2,state2) in zip(runs[0],other):
        assert torch.equal(loss,loss2)
        assert all(torch.equal(v,state2[k]) for k,v in state.items())
print('Actual tiny model+optimizer 3-step base/recursive-OFF/batch-default: bit exact',flush=True)

# Local guardian test: reporting failure may return FAILED after independent idle.
gpath=P/'20260926-sprint-p0-clean-recursive/variants/recursive-trn21/lifetime_guard.py'
spec=importlib.util.spec_from_file_location('review_guard',gpath)
gm=importlib.util.module_from_spec(spec); spec.loader.exec_module(gm)
lock=OUT/'review.lock'; fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600); fcntl.flock(fd,fcntl.LOCK_EX)
st=os.fstat(fd); events=[]
g=gm.Guardian(fd,str(lock),(st.st_dev,st.st_ino),OUT/'guardian-receipt.jsonl',grace=.1,kill_wait=.1)
def reporting_failure():
    events.append('reporting'); raise OSError('simulated reporting failure')
def idle(): events.append('independent-idle')
assert g.finish(reporting_failure,idle) is False
assert events==['reporting','independent-idle']
gm.require_original_lock(fd,str(lock),(st.st_dev,st.st_ino))
print('Guardian reporting failure: independent idle called, failed result, original lock held through finish',flush=True)
os.close(fd)
print('PASS',flush=True)
