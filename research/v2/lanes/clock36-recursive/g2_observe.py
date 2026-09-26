"""Tiny CPU test-only observations/reference injection on unchanged G2 machinery."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

SUPPORT=Path('/home/poojith-devan/Desktop/Trainium')
PINS={'research/v2/lanes/initial-cache-refresh/g2_observe.py':'a1b9108c3aa3af22cd819e6c0553e6c5c2bb7f3887bc0d3c1b8b4e6d856db03d',
      'workflow/tools/preflight_dryrun.py':'6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8',
      'workflow/tools/stub_prepare/prepare.py':'b2ae4d6f7de54aa1b0f5a9cb2c2629c38c86c30178e6516a7b48b8ac004ca1a2',
      'prepare.py':'5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb'}
for p,h in PINS.items():
    if hashlib.sha256((SUPPORT/p).read_bytes()).hexdigest()!=h:
        raise RuntimeError('support mismatch: '+p)
spec=importlib.util.spec_from_file_location('cache_observer',SUPPORT/next(iter(PINS)))
observer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
g2=observer.g2
g2.__file__=__file__
original=observer.observe


def observe(mod):
    import torch
    mod.TRAIN_STARTUP_STEPS_EXCLUDED=5
    original(mod)
    reference=os.environ.get('CLOCK_RECURSIVE_REFERENCE')=='1'
    if reference:
        from test_static import ref_lr,ref_momentum,ref_beta2
        mod.lr_multiplier=lambda step,elapsed=0.,max_train_seconds=1.,kind='muon':ref_lr(step,elapsed,max_train_seconds,mod.WARMUP_STEPS,kind)
        mod.muon_momentum=ref_momentum
        mod.muon_beta2=ref_beta2
        mod.startup_adjusted_schedule_elapsed=lambda step,elapsed,budget_started: max(0.,elapsed+(budget_started-mod._PROCESS_STARTED)+30.-900.) if step<5 else elapsed
    records=[]
    calls=[]
    beta=[]
    lr=mod.lr_multiplier
    def capture_lr(step,*a,**kw):
        value=lr(step,*a,**kw)
        calls.append(dict(step=step,elapsed=kw.get('elapsed',0.),kind=kw.get('kind','muon'),result=value))
        return value
    mod.lr_multiplier=capture_lr
    math_fn=mod._muon_math
    def capture_math(*a,**kw):
        value=a[12]
        beta.append(dict(is_tensor=isinstance(value,torch.Tensor),value=float(value),dtype=str(value.dtype) if isinstance(value,torch.Tensor) else 'Python float'))
        return math_fn(*a,**kw)
    mod._muon_math=capture_math
    step_fn=mod.OwnedOptimizer.step
    def step(self,*a,**kw):
        keys=('kind','lr','initial_lr','momentum','weight_decay','beta2','betas')
        records.append([{k:g[k] for k in keys if k in g} for g in self.optimizer.param_groups])
        out=step_fn(self,*a,**kw)
        if len(records)==1:
            mod.time._t+=1000.
        return out
    mod.OwnedOptimizer.step=step
    cleanup_fn=mod.cleanup_runtime
    def cleanup():
        record=dict(reference_formulas_injected=reference,startup_steps_excluded=5,first_step_fake_delay_s=1000.,optimizer_groups=records,lr_calls=calls,beta2_routes=beta)
        (Path(os.environ['G2_WORKDIR'])/('recursive-observe-rank'+os.environ['RANK']+'.json')).write_text(json.dumps(record,sort_keys=True))
        return cleanup_fn()
    mod.cleanup_runtime=cleanup


observer.observe=observe
aggregate_fn=g2.aggregate
def aggregate(ns,train_py,prepare_dir,train_argv,ranks,*a):
    result=aggregate_fn(ns,train_py,prepare_dir,train_argv,ranks,*a)
    for r in ranks:
        if r.get('appended_argv'):
            path=Path(r['appended_argv'][-1]).parent/f"recursive-observe-rank{r['rank']}.json"
            r['recursive_observations']=json.loads(path.read_text()) if path.exists() else None
    result['observer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['support_pins']=PINS
    result['shims'].append('test-only five startup boundaries and 1000-second first-update fake-clock delay; optional independent schedule formulas only in reference arms')
    return result
g2.aggregate=aggregate
if __name__=='__main__':
    sys.exit(g2.main())
