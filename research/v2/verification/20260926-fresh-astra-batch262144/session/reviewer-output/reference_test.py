"""Bounded, independent CPU reference test. Does not import/run either train module.

AST extracts only reviewed statements/helpers. Toy model and SGD replace GPT/Muon;
four ranks are serial replicas and AVG is a deterministic local mock. No clock,
process group, compilation, dataset, accelerator, or full-size tensors are used.
Criteria are fixed here before execution: exact tensor bytes for default identity;
float64 reference atol=1e-12, rtol=1e-10; float32 atol=3e-6, rtol=3e-5.
"""
import os
THREAD_SETTINGS = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
os.environ.update(THREAD_SETTINGS)
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import platform
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True)
OUT = Path('/work')
RESULT = {'verdict': 'REFUSE', 'seed': 42, 'settings': {
    'requested_thread_environment': THREAD_SETTINGS, 'intraop': torch.get_num_threads(),
    'interop': torch.get_num_interop_threads(), 'python': platform.python_version(),
    'torch': torch.__version__, 'device': 'cpu', 'deterministic_algorithms': True},
    'bounds': {'width': 7, 'vocabulary': 11, 'context': 3, 'microbatch': 2,
               'world_replicas': 4, 'steps': 3, 'max_allocated_data_tokens_per_tensor': 240,
               'max_reference_tokens_per_update': 72, 'parameter_elements_per_model': 165},
    'criteria': {'default_identity': 'byte-for-byte, no tolerance',
                 'float64': {'atol': 1e-12, 'rtol': 1e-10}, 'float32': {'atol': 3e-6, 'rtol': 3e-5}},
    'checks': [], 'cases': [], 'failures': [], 'extractions': {}}

def check(name, condition, **details):
    item = dict(name=name, passed=bool(condition), **details)
    RESULT['checks'].append(item)
    print(json.dumps(item, sort_keys=True), flush=True)
    if not condition:
        raise AssertionError(name)

def module_code(nodes, filename):
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + copy.deepcopy(nodes), type_ignores=[])
    return compile(ast.fix_missing_locations(tree), filename, 'exec')

def target(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)

class Extracted:
    def __init__(self, name):
        self.name = name
        self.path = OUT / (name + '.py')
        self.source = self.path.read_text()
        self.tree = ast.parse(self.source)
        self.main = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        self.loop = next(n for n in self.main.body if isinstance(n, ast.While))
        self.g = dict(torch=torch, nn=nn, F=F, argparse=argparse, print0=lambda *a, **k: None)
        # Only literal top-level settings; never execute module imports/initializers.
        for n in self.tree.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                try:
                    value = ast.literal_eval(n.value)
                except (ValueError, TypeError):
                    continue
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        self.g[t.id] = value
        self.g.update(_HAS_NKI=False, RECURSIVE_SCHEDULE=True, WARMUP_STEPS=0,
                      BATCH_RAMP=False, LOSS_BYTE_WEIGHT=False, BYTE_LOSS_W=None)
        funcs = ['softcap_cross_entropy', 'sync_gradients', 'sync_gradients_flat',
                 'clip_and_sync_gradients', 'clip_grad_norm_flat_', 'unscale_grads_',
                 'batch_ramp_stage', 'batch_ramp_grad_accum', 'lr_multiplier', 'muon_momentum', 'muon_beta2']
        definitions = [next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == key) for key in funcs]
        exec(module_code(definitions, str(self.path)), self.g)
        self.fragments = {}
        mi = self.main.body
        start = next(i for i, n in enumerate(mi) if target(n, 'world_tokens_per_micro'))
        self.add('arithmetic', mi[start:start + 3])
        self.add('fetch_definition', [next(n for n in mi if isinstance(n, ast.FunctionDef) and n.name == 'fetch')])
        self.add('initial_fetch', [next(n for n in mi if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == '(x, y, loader_state)')])
        lp = self.loop.body
        a = next(i for i,n in enumerate(lp) if target(n, 'adam_step_now'))
        b = next(i for i,n in enumerate(lp) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'micro_index')
        self.add('accumulate', lp[a:b+1])
        self.add('unscale_and_clip', [n for n in lp if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id in ('unscale_grads_', 'clip_and_sync_gradients')])
        self.add('step', [n for n in lp if isinstance(n, ast.If) and ast.unparse(n.test) == 'not skip_step'])
        self.add('reported_loss', [n for n in lp if target(n, 'train_loss')])
        self.add('token_account', [n for n in lp if target(n, 'step_tokens') or (isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == 'total_train_tokens')])
        if name == 'candidate':
            start = next(i for i,n in enumerate(mi) if isinstance(n, ast.Expr) and '--total-batch-size' in ast.unparse(n))
            self.add('cli', mi[start:start+4])
        RESULT['extractions'][name] = {key: [[n.lineno, n.end_lineno] for n in ns] for key, ns in self.fragments.items()}
        RESULT['extractions'][name]['helpers'] = {n.name: [n.lineno, n.end_lineno] for n in definitions}
        self.codes = {key: module_code(ns, str(self.path)) for key, ns in self.fragments.items()}

    def add(self, key, nodes):
        assert nodes, key
        self.fragments[key] = nodes

    def parse(self, argv):
        self.g['TOTAL_BATCH_SIZE'] = 131072  # Fresh process semantics for every CLI case.
        if self.name == 'base':
            assert argv == []
            return self.g['TOTAL_BATCH_SIZE']
        parser = argparse.ArgumentParser(prog='extracted-total-batch')
        self.g['parser'] = parser
        previous = sys.argv
        sys.argv = ['extracted-total-batch'] + argv
        try:
            exec(self.codes['cli'], self.g)
        finally:
            sys.argv = previous
        return self.g['TOTAL_BATCH_SIZE']

    def arithmetic(self, total, device_batch=1, pack=32, context=1024, world=4):
        g = self.g
        g.update(TOTAL_BATCH_SIZE=total, DEVICE_BATCH_SIZE=device_batch, PACK_FACTOR=pack, SEQ_LEN=context, world_size=world)
        exec(self.codes['arithmetic'], g)
        return g['world_tokens_per_micro'], g['grad_accum_steps']

def bytes_of(t):
    return t.detach().contiguous().cpu().numpy().tobytes()

def tensor_record(t):
    return {'dtype': str(t.dtype), 'shape': list(t.shape), 'values': t.detach().cpu().tolist(),
            'sha256': hashlib.sha256(bytes_of(t)).hexdigest()}

class Tiny(nn.Module):
    def __init__(self, ce, dtype):
        super().__init__()
        self.embed = nn.Embedding(11, 7, dtype=dtype)
        self.head = nn.Linear(7, 11, dtype=dtype)
        self.ce = ce
        self.losses = []
    def logits(self, x):
        return self.head(torch.tanh(self.embed(x)))
    def forward(self, x, y):
        loss = self.ce(self.logits(x), y, 'mean')
        self.losses.append(loss.detach().clone())
        return loss

class SGDAdapter:
    """Bounded reference optimizer; explicitly not production Muon/Adam ownership."""
    def __init__(self, model):
        self.opt = torch.optim.SGD(model.parameters(), lr=0.015, momentum=0.8)
        self.calls = 0
        self.zero_calls = 0
    def zero_grad(self, set_to_none):
        assert set_to_none
        self.zero_calls += 1
        self.opt.zero_grad(set_to_none=set_to_none)
    def step(self, update_adamw):
        assert update_adamw
        self.calls += 1
        self.opt.step()

class MockAverage:
    ReduceOp = SimpleNamespace(AVG='AVG')
    def __init__(self, model, averages):
        self.index = {id(p.grad): i for i,p in enumerate(model.parameters())}
        self.averages = averages
        self.calls = 0
    def is_available(self): return True
    def is_initialized(self): return True
    def all_reduce(self, tensor, op):
        assert op == self.ReduceOp.AVG
        tensor.copy_(self.averages[self.index[id(tensor)]])
        self.calls += 1

def compare_tensors(actual, expected, dtype):
    criteria = RESULT['criteria'][str(dtype).split('.')[-1]]
    ae = [(a-b).abs() for a,b in zip(actual, expected)]
    ok = all(torch.allclose(a,b, **criteria) for a,b in zip(actual,expected))
    absolute = max(t.max().item() for t in ae)
    relative = max((e / b.abs().clamp_min(criteria['atol'])).max().item() for e,b in zip(ae,expected))
    return dict(passed=ok, max_abs=absolute, max_relative_floor_atol=relative, **criteria)

def run_case(ex, label, total, gas, dtype, loss_scale=1.0):
    # Actual intended arithmetic is checked separately without allocating its shape.
    # Tiny shape corresponds to 24 global tokens/micro and gas*24 per update.
    check(label + ': tiny arithmetic', ex.arithmetic(24*gas, 1, 2, 3, 4) == (24,gas))
    torch.manual_seed(42)
    initial = Tiny(ex.g['softcap_cross_entropy'], dtype)
    models = [copy.deepcopy(initial) for _ in range(4)]
    reference = copy.deepcopy(initial)
    opts = [SGDAdapter(m) for m in models]
    momenta = [torch.zeros_like(p) for p in reference.parameters()]
    gen = torch.Generator(device='cpu').manual_seed(42)
    # Include exactly one lookahead batch. Largest case gas=3: 240 tokens/array.
    xs = torch.randint(0,11,(4,3*gas+1,2,3),generator=gen)
    ys = torch.randint(0,11,(4,3*gas+1,2,3),generator=gen)
    fetch_counts = [0]*4
    envs = []
    for rank, (model,opt) in enumerate(zip(models,opts)):
        g = dict(ex.g)
        def loader(r=rank):
            for j in range(3*gas+1):
                fetch_counts[r] += 1
                yield xs[r,j].reshape(1,6), ys[r,j].reshape(1,6), {'epoch':1, 'index':j}
        g.update(model=model, optimizer=opt, train_loader=loader(), DEVICE_BATCH_SIZE=1,
                 PACK_FACTOR=2, SEQ_LEN=3, grad_accum_steps=gas, grad_accum_now=gas,
                 ADAM_EVERY_N=1, BATCH_RAMP=False, LOSS_SCALE=loss_scale,
                 grad_flat_buffer=None, debug_numerics=lambda *a, **k: None,
                 clip_main_params=list(model.parameters()), clip_ngram_params=[],
                 GRAD_CLIP=0.08, CLIP_AFTER_REDUCE=True, args=SimpleNamespace(clip_foreach=False),
                 adamw_skip_ids=frozenset(), skip_step=False, unscale_with=loss_scale,
                 total_train_tokens=0, world_tokens_per_micro=24)
        # Helpers keep their globals in ex.g, intentionally updated below for mock dist.
        exec(ex.codes['fetch_definition'], g)
        exec(ex.codes['initial_fetch'], g)
        envs.append(g)
    history, comparisons = [], []
    for step in range(3):
        local_grads = []
        for model,g in zip(models,envs):
            g['step'] = step
            exec(ex.codes['accumulate'], g)
            local_grads.append([p.grad.detach().clone()/loss_scale for p in model.parameters()])
        averaged = [torch.stack([local_grads[r][i] for r in range(4)]).mean(0) for i in range(len(local_grads[0]))]
        # Independent token-summed reference, not a second execution of the loss helper.
        reference.zero_grad(set_to_none=True)
        losses = []
        for r in range(4):
            for j in range(step*gas,(step+1)*gas):
                z = reference.logits(xs[r,j])
                capped = 15.0*torch.tanh(z/15.0)
                flat = capped.reshape(-1,11)
                true = flat.gather(1,ys[r,j].reshape(-1,1)).squeeze(1)
                losses.append((torch.logsumexp(flat,dim=1)-true).sum())
        ref_loss = torch.stack(losses).sum()/(4*gas*6)
        ref_loss.backward()
        ref_grad = [p.grad.detach().clone() for p in reference.parameters()]
        grad_error = compare_tensors(averaged, ref_grad, dtype)
        check(label + f': step{step} mean gradient', grad_error['passed'], **{k:v for k,v in grad_error.items() if k != 'passed'})
        micro_mean = torch.stack([v for m in models for v in m.losses[-gas:]]).mean()
        loss_error = compare_tensors([micro_mean], [ref_loss.detach()], dtype)
        check(label + f': step{step} mean loss', loss_error['passed'], **{k:v for k,v in loss_error.items() if k != 'passed'})
        # Explicit independent norm/update formula (different reduction ordering).
        norm = torch.cat([v.reshape(-1) for v in ref_grad]).norm()
        coef = (0.08/(norm+1e-6)).clamp(max=1.0)
        ref_clipped = [v*coef for v in ref_grad]
        with torch.no_grad():
            for p,momentum,grad in zip(reference.parameters(),momenta,ref_clipped):
                momentum.mul_(0.8).add_(grad)
                p.add_(momentum,alpha=-0.015)
        per_rank = []
        for rank,(model,g,opt) in enumerate(zip(models,envs,opts)):
            mock = MockAverage(model,averaged)
            ex.g['dist'] = mock
            exec(ex.codes['unscale_and_clip'],g)
            clipped = [p.grad.detach().clone() for p in model.parameters()]
            clip_error = compare_tensors(clipped,ref_clipped,dtype)
            exec(ex.codes['step'],g)
            exec(ex.codes['reported_loss'],g)
            exec(ex.codes['token_account'],g)
            weight_error = compare_tensors(list(model.parameters()),list(reference.parameters()),dtype)
            check(label + f': step{step} rank{rank} clip/update', clip_error['passed'] and weight_error['passed'] and mock.calls == 3,
                  clip_error=clip_error, weight_error=weight_error, avg_calls=mock.calls)
            per_rank.append({'losses':[tensor_record(v) for v in model.losses[-gas:]],
                             'local_grads':[tensor_record(v) for v in local_grads[rank]],
                             'clipped_grads':[tensor_record(v) for v in clipped],
                             'parameters':[tensor_record(p) for p in model.parameters()],
                             'momentum':[tensor_record(opt.opt.state[p]['momentum_buffer']) for p in model.parameters()],
                             'reported_last_loss':g['train_loss'], 'fetches':fetch_counts[rank],
                             'logical_tokens':g['total_train_tokens']})
        check(label + f': step{step} replicas exact', all(per_rank[r]['parameters'] == per_rank[0]['parameters'] for r in range(4)))
        history.append(per_rank)
        comparisons.append({'step':step,'loss':loss_error,'gradient':grad_error,
                            'reference_loss':ref_loss.item(),'gradient_norm':norm.item(),'clip_coefficient':coef.item()})
    check(label + ': fetch/optimizer counts', fetch_counts == [3*gas+1]*4 and all(o.calls==3 and o.zero_calls==3 for o in opts), fetches=fetch_counts)
    check(label + ': token accounting', all(g['total_train_tokens']==3*gas*24 for g in envs))
    check(label + ': clipping exercised', all(x['clip_coefficient'] < 1.0 for x in comparisons))
    result = dict(label=label,source=ex.name,intended_total=total,gas=gas,dtype=str(dtype),loss_scale=loss_scale,
                  fetches=fetch_counts,optimizer_calls=[o.calls for o in opts],comparisons=comparisons,history=history)
    RESULT['cases'].append(result)
    return result

def main():
    base, candidate = Extracted('base'), Extracted('candidate')
    # AST equivalence of the entire source after removing exactly the five added lines.
    cleaned = copy.deepcopy(candidate.tree)
    main_node = next(n for n in cleaned.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    added = {3433,3616,3618,3620}
    main_node.body = [n for n in main_node.body if n.lineno not in added]
    check('whole source AST unchanged except assigned CLI patch', ast.dump(base.tree,include_attributes=False)==ast.dump(cleaned,include_attributes=False))
    check('base default constant', base.parse([])==131072)
    for argv,want in [([],131072),(['--total-batch-size','131072'],131072),(['--total-batch-size','262144'],262144),(['--total-batch-size=393216'],393216)]:
        check('valid CLI '+repr(argv),candidate.parse(argv)==want)
    invalid = []
    for argv in [['--total-batch-size',v] for v in ('0','-1','1.5','262144.0','abc','nan','inf','')]+[['--total-batch-size']]:
        stream=io.StringIO()
        with contextlib.redirect_stderr(stream):
            try:
                candidate.parse(argv)
                code=0
            except SystemExit as exc:
                code=exc.code
        invalid.append(dict(argv=argv,exit=code,stderr=stream.getvalue()))
        check('invalid CLI '+repr(argv),code==2)
    RESULT['invalid_arguments']=invalid
    for total,gas in [(131072,1),(262144,2),(393216,3)]:
        check('production arithmetic '+str(total),candidate.arithmetic(total)==(131072,gas))
    for total in (1,65536,131071,131073,262145):
        check('positive accepted before divisibility '+str(total),candidate.parse(['--total-batch-size',str(total)])==total)
        try:
            candidate.arithmetic(total)
            error=None
        except ValueError as exc:
            error=str(exc)
        check('production divisibility rejection '+str(total),error is not None,error=error)
    for progress in (0.0,0.05,0.35,0.5,0.75,1.0):
        for gas in (1,2,3):
            check(f'no batch LR scale {progress}/{gas}',candidate.g['batch_ramp_grad_accum'](progress,gas)==(gas,1.0))
        for key in ('lr_multiplier','muon_momentum','muon_beta2'):
            if key=='lr_multiplier':
                for kind in ('muon','adamw'):
                    a=base.g[key](100,elapsed=1800*progress,max_train_seconds=1800,kind=kind)
                    b=candidate.g[key](100,elapsed=1800*progress,max_train_seconds=1800,kind=kind)
                    check(f'schedule exact {key}/{kind}/{progress}',a.hex()==b.hex(),value=b)
            else:
                args=(100,progress) if key=='muon_momentum' else (progress,)
                a,b=base.g[key](*args),candidate.g[key](*args)
                check(f'schedule exact {key}/{progress}',a.hex()==b.hex(),value=b)
    for dtype in (torch.float64,torch.float32):
        tag=str(dtype).split('.')[-1]
        b=run_case(base,'base_default_'+tag,base.parse([]),1,dtype)
        d=run_case(candidate,'candidate_default_'+tag,candidate.parse([]),1,dtype)
        e=run_case(candidate,'candidate_explicit_default_'+tag,candidate.parse(['--total-batch-size','131072']),1,dtype)
        check('exact default histories '+tag,b['history']==d['history']==e['history'])
        two=run_case(candidate,'candidate_262144_'+tag,candidate.parse(['--total-batch-size','262144']),2,dtype)
        check('factor two deliberately different '+tag,two['history'][-1][0]['parameters'] != d['history'][-1][0]['parameters'])
        scaled=run_case(candidate,'candidate_262144_scale8_'+tag,262144,2,dtype,loss_scale=8.0)
        check('power-of-two loss unscale exact '+tag,two['history']==scaled['history'])
    run_case(candidate,'candidate_393216_float64',393216,3,torch.float64)
    # Order-sensitive gradients: AVG([4,0],[0,1],[-2,0],[0,1])=[.5,.5].
    # Local clipping then average differs; ensure the mock/function sees correct order.
    m=nn.Linear(2,1,bias=False,dtype=torch.float64)
    m.weight.grad=torch.tensor([[4.0,0.0]],dtype=torch.float64)
    avg=[torch.tensor([[0.5,0.5]],dtype=torch.float64)]
    mock=MockAverage(m,avg)
    candidate.g['dist']=mock
    candidate.g['clip_and_sync_gradients'](m,list(m.parameters()),[],max_norm=0.25,clip_after_reduce=True,foreach=False)
    expected=avg[0]*(0.25/(avg[0].norm()+1e-6))
    check('order-sensitive AVG before clipping',bytes_of(m.weight.grad)==bytes_of(expected))
    RESULT['verdict']='PASS'

try:
    main()
except BaseException as exc:
    RESULT['failures'].append({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
    traceback.print_exc()
finally:
    (OUT/'reference_results.json').write_text(json.dumps(RESULT,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'verdict':RESULT['verdict'],'checks':len(RESULT['checks']),'cases':len(RESULT['cases']),'failures':RESULT['failures']},indent=2),flush=True)
sys.exit(0 if RESULT['verdict']=='PASS' else 1)
