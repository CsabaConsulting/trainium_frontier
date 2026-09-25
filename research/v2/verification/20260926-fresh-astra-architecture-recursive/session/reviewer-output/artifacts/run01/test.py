"""Reviewer-authored bounded CPU test. No full-module import or training model.

Run via run_tests.sh. Source functions/statements are extracted without edits.
Independent Decimal arithmetic supplies an analytical oracle; frozen sources
supply strict binary64 / tensor-byte equivalence controls.
"""
import argparse
import ast
import collections
import copy
import dataclasses
from decimal import Decimal, localcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import time
import traceback
import types
import weakref
import torch
from torch.utils._python_dispatch import TorchDispatchMode

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
START = time.perf_counter()
OUT = Path('/work/artifacts')
ROWS = []
EXTRACTIONS = []

def record(name, actual, expected, *, tolerance=0.0, context=None):
    if tolerance:
        ok = math.isfinite(actual) and abs(actual - expected) <= tolerance
    else:
        ok = actual == expected
        if isinstance(actual, float) and isinstance(expected, float):
            ok = struct.pack('!d', actual) == struct.pack('!d', expected)
    ROWS.append(dict(name=name, passed=ok, actual=actual, expected=expected,
                     absolute_tolerance=tolerance, relative_tolerance=0.0, context=context))

def compile_nodes(nodes, ns, filename):
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + copy.deepcopy(nodes), type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, filename, 'exec'), ns)

class Source:
    def __init__(self, name):
        self.name = name
        self.path = Path('/work/source') / (name + '.py')
        self.text = self.path.read_text()
        self.tree = ast.parse(self.text)
        self.ns = dict(torch=torch, math=math, argparse=argparse, Path=Path,
                       asdict=dataclasses.asdict, print0=lambda *a, **k: None)
        # Literal initial module settings only; no source import side effects.
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                try: value = ast.literal_eval(node.value)
                except (ValueError, TypeError): continue
                for target in node.targets:
                    if isinstance(target, ast.Name): self.ns[target.id] = value
        funcs = ['lr_multiplier', 'muon_momentum', 'muon_beta2', 'demon_beta1_scale',
                 '_muon_math', '_muon_lr_g', 'current_toggles', 'restore_eval_toggles',
                 'save_checkpoint', 'torch_load_weights']
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in funcs:
                compile_nodes([node], self.ns, str(self.path))
                self.note(node.name, node)
        main = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        # Extract the exact contiguous schedule/group-update statements from the loop.
        loop = next(n for n in ast.walk(main) if isinstance(n, ast.For) and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'lrm' for t in x.targets) for x in n.body))
        begin = next(i for i,n in enumerate(loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'lrm' for t in n.targets))
        end = next(i for i in range(begin, len(loop.body)) if isinstance(loop.body[i], ast.For) and ast.unparse(loop.body[i].target) == 'group')
        self.group_nodes = loop.body[begin:end+1]
        self.group_code = compile(ast.Module(body=copy.deepcopy(self.group_nodes), type_ignores=[]), str(self.path), 'exec')
        self.note('group_update', self.group_nodes[0], self.group_nodes[-1])
        klass = next(n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == 'MuonAdamW')
        methods = ['_device_scalar', '_muon_depth_dev', '_fill_depth_momentum', '_muon_depth_mom', '_muon_compile_scalars', '_muon_step']
        body = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        # Test-only shell replaces optimizer construction/state registration, not arithmetic.
        fixture = ast.ClassDef(name='Fixture', bases=[], keywords=[], body=copy.deepcopy(body), decorator_list=[])
        compile_nodes([fixture], self.ns, str(self.path))
        for node in body: self.note('MuonAdamW.' + node.name, node)
        self.ns.update(_MUON_COMPILED={}, _HAS_NKI=False, USE_NKI_MUON=False)
        # Execute the exact list extensions used to save/restore model toggles.
        for node in self.tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'ARCH_TOGGLE_NAMES':
                compile_nodes([node], self.ns, str(self.path))
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == 'ARCH_TOGGLE_NAMES':
                compile_nodes([node], self.ns, str(self.path))
        missing = [n for n in self.ns['ARCH_TOGGLE_NAMES'] if n not in self.ns]
        assert not missing, (name, missing)
        self.ns['_EVAL_TOGGLE_DEFAULTS'] = self.ns['current_toggles']()
        self.defaults = {k: copy.deepcopy(v) for k,v in self.ns.items() if k.isupper() and not k.startswith('_')}
        self.main = main

    def note(self, name, first, last=None):
        last = first if last is None else last
        segment = '\n'.join(self.text.splitlines()[first.lineno-1:last.end_lineno]) + '\n'
        EXTRACTIONS.append(dict(source=self.name, name=name, first_line=first.lineno,
                                last_line=last.end_lineno, sha256=hashlib.sha256(segment.encode()).hexdigest()))

    def reset(self, on=None, **settings):
        self.ns.update(copy.deepcopy(self.defaults))
        if on is not None: self.ns['RECURSIVE_SCHEDULE'] = on
        self.ns.update(settings)

    def groups(self, groups, step, elapsed, budget, batch_scale):
        local = dict(step=step, elapsed=elapsed, args=types.SimpleNamespace(max_train_seconds=budget),
                     batch_lr_scale=batch_scale, optimizer=types.SimpleNamespace(param_groups=copy.deepcopy(groups)))
        exec(self.group_code, self.ns, local)
        return local['optimizer'].param_groups


def D(v): return Decimal(str(v))

def oracle(step, elapsed, budget, warmup, kind, progress=None):
    """Independent high-precision piecewise equations, not source function calls.

    Float thresholds are intentional: the reference branches on 1.0 - 0.95,
    whose binary64 value is slightly greater than literal 0.05.
    """
    p_float = min(1.0, max(0.0, elapsed / max(1e-6, budget))) if progress is None else min(1.0, max(0.0, progress))
    with localcontext() as ctx:
        ctx.prec = 60
        p = Decimal.from_float(p_float)
        span = D('0.95') if kind == 'muon' else D('0.65')
        if step < warmup:
            lr = Decimal(step + 1) / Decimal(max(1, warmup))
        elif p_float < 1.0 - float(span):
            lr = Decimal(1)
        else:
            lr = D('0.05') + D('0.95') * (1 - p) / span
        f = min(Decimal(step) / 300, Decimal(1))
        mom = D('0.85') + D('0.10') * f
        if p_float > 1.0 - 0.95:
            mom = D('0.95') - D('0.16') * ((p - D('0.05')) / D('0.95')) ** 2
        b2 = D('0.95') + D('0.02') * max(Decimal(0), (p - D('0.05')) / D('0.95'))
        return tuple(float(x) for x in (lr, mom, b2))


class TensorMemory(TorchDispatchMode):
    """Track all tensor objects emitted by dispatcher, including temporary views.

    Counting each view's entire backing storage overcounts aliases, conservatively.
    Track before/after each operator; small allocator/workspace internals are not
    tensor objects and are not claimed as RSS. No fixture tensor predates this mode.
    """
    def __init__(self):
        super().__init__()
        self.refs = {}
        self.peak = 0
        self.max_single_storage = 0
        self.calls = 0
    def add(self, obj):
        if isinstance(obj, torch.Tensor):
            size = obj.untyped_storage().nbytes()
            self.max_single_storage = max(self.max_single_storage, size)
            self.refs[id(obj)] = (weakref.ref(obj), size)
        elif isinstance(obj, (list, tuple)):
            for x in obj: self.add(x)
        elif isinstance(obj, dict):
            for x in obj.values(): self.add(x)
    def sample(self):
        self.refs = {k:v for k,v in self.refs.items() if v[0]() is not None}
        total = sum(v[1] for v in self.refs.values())
        self.peak = max(self.peak, total)
        assert total < 64 * 1024**2, total
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.add(args); self.add(kwargs); self.sample()
        out = func(*args, **kwargs)
        self.add(out); self.sample(); self.calls += 1
        return out


def tensor_bytes(t):
    return t.detach().contiguous().view(torch.uint8).flatten().tolist()


def trajectory(src, on, compiled_body, compute_dtype, shape, transposed=False, beta2_final=0.0):
    src.reset(on, USE_COMPILE_MUON=compiled_body, COMPUTE_DTYPE=compute_dtype, MUON_BETA2_FINAL=beta2_final)
    src.ns['_MUON_COMPILED'] = {}
    fixture = src.ns['Fixture']()
    fixture._scalar_cache = {}
    fixture.state = collections.defaultdict(dict)
    p = torch.nn.Parameter(torch.arange(1, math.prod(shape)+1, dtype=torch.float32).reshape(shape) / 17)
    group = dict(kind='muon', params=[p], initial_lr=.015, ns_steps=5, transposed=transposed)
    records = []
    transport = []
    real_math = src.ns['_muon_math']
    def observe(*args):
        value = args[12]
        transport.append(dict(type='tensor' if isinstance(value, torch.Tensor) else 'float',
                              value=float(value), dtype=str(value.dtype) if isinstance(value, torch.Tensor) else 'binary64'))
        return real_math(*args)
    src.ns['_muon_math'] = observe
    try:
        for step, progress in [(0, 0.0), (1, 0.05), (20, 0.051), (299, 0.35), (300, 0.70), (777, 1.0)]:
            p.grad = torch.arange(1, p.numel()+1, dtype=torch.float32).reshape(shape) * ((step % 3) + 1) / 31 - .12
            group.update(lr=.015 * src.ns['lr_multiplier'](step, progress, 1.0),
                         momentum=src.ns['muon_momentum'](step, progress),
                         beta2=src.ns['muon_beta2'](progress), weight_decay=.1 * (1-progress))
            fixture._muon_step(group)
            state = fixture.state[p]
            records.append(dict(step=step, progress=progress, lr=group['lr'], momentum=group['momentum'], beta2=group['beta2'],
                parameter=p.detach().tolist(), parameter_bytes=tensor_bytes(p),
                momentum_bytes=tensor_bytes(state['momentum_buffer']), second_bytes=tensor_bytes(state['second_momentum_buffer']),
                finite=bool(torch.isfinite(p).all() and torch.isfinite(state['momentum_buffer']).all() and torch.isfinite(state['second_momentum_buffer']).all())))
    finally:
        src.ns['_muon_math'] = real_math
        src.ns['_MUON_COMPILED'] = {}
    return dict(trajectory=records, transport=transport)


def run():
    baseline, candidate, reference = [Source(x) for x in ('baseline', 'candidate', 'schedule_reference')]
    # Exact CLI action and main assignment, without parsing/importing model code.
    action = next(n for n in ast.walk(candidate.main) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and n.value.args and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value == '--recursive-schedule')
    assignment = next(n for n in candidate.main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RECURSIVE_SCHEDULE' for t in n.targets))
    for argv, expected in [([], False), (['--no-recursive-schedule'], False), (['--recursive-schedule'], True), (['--recursive-schedule', '--no-recursive-schedule'], False)]:
        parser = argparse.ArgumentParser()
        ns = dict(parser=parser, argparse=argparse)
        compile_nodes([action], ns, str(candidate.path))
        ns['args'] = parser.parse_args(argv)
        compile_nodes([assignment], ns, str(candidate.path))
        record('cli', ns['RECURSIVE_SCHEDULE'], expected, context=argv)

    threshold = 1.0 - .95
    adam_threshold = 1.0 - .65
    progresses = [-.2, 0., .01, math.nextafter(.05, -math.inf), .05, math.nextafter(.05, math.inf),
                  math.nextafter(threshold, -math.inf), threshold, math.nextafter(threshold, math.inf), .051,
                  math.nextafter(adam_threshold, -math.inf), adam_threshold, math.nextafter(adam_threshold, math.inf),
                  .5, .95, 1., 1.2]
    steps = [0, 1, 6, 7, 150, 299, 300, 777]
    for warmup in [0, 7]:
        for mode in [None, False, True]:
            candidate.reset(mode, WARMUP_STEPS=warmup)
            other = reference if mode else baseline
            other.reset(mode, WARMUP_STEPS=warmup)
            for step in steps:
                for progress in progresses:
                    ctx = dict(mode=mode, warmup=warmup, step=step, progress=progress)
                    for fn, args in [('muon_momentum', (step, progress)), ('muon_beta2', (progress,))]:
                        record('frozen_exact:' + fn, candidate.ns[fn](*args), other.ns[fn](*args), context=ctx)
                    if mode:
                        _, m, b = oracle(step, progress, 1., warmup, 'muon', progress=progress)
                        record('independent:momentum', candidate.ns['muon_momentum'](step, progress), m, tolerance=5e-16, context=ctx)
                        record('independent:beta2', candidate.ns['muon_beta2'](progress), b, tolerance=5e-16, context=ctx)
                    for kind in ['muon', 'adamw', 'rmsprop']:
                        actual = candidate.ns['lr_multiplier'](step, progress, 1., kind=kind)
                        expected = other.ns['lr_multiplier'](step, progress, 1., **({'kind':kind} if mode else {}))
                        record('frozen_exact:lr', actual, expected, context={**ctx, 'kind':kind})
                        if mode:
                            expected = oracle(step, progress, 1., warmup, kind)[0]
                            record('independent:lr', actual, expected, tolerance=5e-16, context={**ctx, 'kind':kind})
    # Denominator floor, retained warmup precedence, and OFF legacy knobs.
    for settings in [dict(WARMUP_STEPS=0, WARMDOWN_RATIO=0., LRM_QUANTUM=0), dict(WARMUP_STEPS=7, WARMDOWN_RATIO=.6, FINAL_LR_FRAC=.2, LRM_QUANTUM=16, MUON_MOMENTUM_WARMUP_STEPS=37, MUON_MOMENTUM_COOLDOWN_FRAC=.2, MUON_MOMENTUM_FINAL=.71, MUON_BETA2=.88, MUON_BETA2_FINAL=.99)]:
        for src in [baseline, candidate]: src.reset(False, **settings)
        for budget in [-1., 0., 1e-8, 1800.]:
            for elapsed in [-1., 0., 1e-7, 100., 1801.]:
                for step in [0, 8, 301]:
                    ctx = dict(settings=settings, budget=budget, elapsed=elapsed, step=step)
                    for fn, args in [('lr_multiplier',(step,elapsed,budget)), ('muon_momentum',(step, elapsed/max(1e-6,budget))), ('muon_beta2',(elapsed/max(1e-6,budget),))]:
                        record('off_legacy_exact:'+fn, candidate.ns[fn](*args), baseline.ns[fn](*args), context=ctx)
        for src in [candidate, reference]: src.reset(True, **settings)
        for budget in [-1., 0., 1e-8, 1800.]:
            for elapsed in [-1., 0., 1e-7, 100., 1801.]:
                actual = candidate.ns['lr_multiplier'](8, elapsed, budget)
                record('on_denominator_exact', actual, reference.ns['lr_multiplier'](8,elapsed,budget), context=dict(budget=budget, elapsed=elapsed,settings=settings))
                record('on_denominator_independent', actual, oracle(8, elapsed,budget,settings['WARMUP_STEPS'],'muon')[0],tolerance=5e-16)

    groups = [dict(kind=kind, initial_lr=lr, no_lr_decay=flat, betas=(.8,.96), initial_betas=(.8,.96))
              for kind, lr in [('muon',.015), ('adamw',.2), ('rmsprop',.3)] for flat in [False,True]]
    for mode in [None, False, True]:
        for warmup in [0,7]:
            candidate.reset(mode, WARMUP_STEPS=warmup)
            other = reference if mode else baseline
            other.reset(mode, WARMUP_STEPS=warmup)
            for step, progress in [(0,0.),(1,.9),(7,.051),(20,.4),(300,1.),(777,1.2)]:
                for scale in [1., math.sqrt(2.)]:
                    ctx = dict(mode=mode,warmup=warmup,step=step,progress=progress,batch_scale=scale)
                    actual = candidate.groups(groups,step,progress,1.,scale)
                    expected = other.groups(groups,step,progress,1.,scale)
                    record('group_update_frozen_exact',actual,expected,context=ctx)
                    if mode:
                        for g in actual:
                            multiplier = min(1., (step+1)/max(1,warmup)) if g['no_lr_decay'] else oracle(step,progress,1.,warmup,g['kind'])[0]
                            record('group_lr_independent',g['lr'],g['initial_lr']*scale*multiplier,tolerance=5e-16,context={**ctx,'group':g})

    # Negative controls are reviewer-written wrong formulas, never source edits.
    candidate.reset(True)
    wrong_controls = [
        ('wrong_floor',candidate.ns['lr_multiplier'](300,1.,1.), .10),
        ('wrong_lr_kind',candidate.ns['lr_multiplier'](300,.2,1.,kind='adamw'),candidate.ns['lr_multiplier'](300,.2,1.,kind='muon')),
        ('wrong_linear_momentum',candidate.ns['muon_momentum'](300,.525), .95 + .5*(.79-.95)),
        ('wrong_ramp_precedence',candidate.ns['muon_momentum'](1,.051),(1-1/300)*.85+(1/300)*.95),
        ('wrong_literal_threshold',candidate.ns['muon_momentum'](1,math.nextafter(.05,math.inf)),.95),
        ('wrong_early_beta2_ramp',candidate.ns['muon_beta2'](.025),.95+.025*(.97-.95)),
    ]
    for name, actual, wrong in wrong_controls:
        record('negative_control:'+name,abs(actual-wrong)>1e-12,True,context=dict(actual=actual,deliberately_wrong=wrong))
    actual_flat = candidate.groups(groups,300,1.,1.,1.)[1]['lr']
    record('negative_control:lost_flat_precedence',actual_flat != .015*.05,True,context=dict(actual=actual_flat,deliberately_wrong=.015*.05))

    memory = TensorMemory()
    with memory:
        for dtype in [torch.float32, torch.bfloat16]:
            for shape, transposed in [((3,2),False),((2,3),False),((2,2),False),((3,2),True)]:
                for compiled in [False,True]:
                    ctx = dict(dtype=str(dtype),shape=shape,transposed=transposed,compiled_body_cpu=compiled)
                    expected_off = trajectory(baseline,False,compiled,dtype,shape,transposed)
                    for mode in [None,False]:
                        actual = trajectory(candidate,mode,compiled,dtype,shape,transposed)
                        record('tiny_muon_off_bytes_exact',actual,expected_off,context={**ctx,'mode':mode})
                    expected_on = trajectory(reference,True,compiled,dtype,shape,transposed)
                    actual_on = trajectory(candidate,True,compiled,dtype,shape,transposed)
                    record('tiny_muon_on_reference_bytes_exact',actual_on,expected_on,context=ctx)
                    record('tiny_muon_finite',all(r['finite'] for r in actual_on['trajectory']),True,context=ctx)
                    record('negative_control:ON_is_not_OFF',actual_on['trajectory'][-1]['parameter_bytes'] != expected_off['trajectory'][-1]['parameter_bytes'],True,context=ctx)
                    if compiled:
                        for obs, state in zip(actual_on['transport'],actual_on['trajectory']):
                            expected = struct.unpack('f',struct.pack('f',1.0-state['beta2']))[0]
                            record('beta2_device_transport',obs,dict(type='tensor',value=expected,dtype='torch.float32'),context={**ctx,'progress':state['progress']})
                        record('beta2_default_transport_types',[x['type'] for x in expected_off['transport']],['float']*6,context=ctx)
                        frozen = struct.unpack('f',struct.pack('f',.1))[0]
                        record('negative_control:stale_beta2',actual_on['transport'][-1]['value'] != frozen,True,context=dict(actual=actual_on['transport'][-1],wrong=frozen))
        # Existing optional beta2 ramp remains staged even with new package OFF.
        record('legacy_ramp_tensor_path',trajectory(candidate,False,True,torch.float32,(3,2),beta2_final=.98),trajectory(baseline,False,True,torch.float32,(3,2),beta2_final=.98))

        # Actual checkpoint serializer with a six-element test-only state/config fixture.
        @dataclasses.dataclass
        class Config:
            sequence_len: int = 2
            vocab_size: int = 2
            n_embd: int = 2
        class CheckpointFixture:
            config = Config()
            def state_dict(self): return {'tiny.weight': torch.arange(6,dtype=torch.float32).reshape(3,2)}
        candidate.reset()
        for saved in [False,True]:
            candidate.ns['RECURSIVE_SCHEDULE'] = saved
            path = OUT / ('checkpoint_on.pt' if saved else 'checkpoint_off.pt')
            candidate.ns['save_checkpoint'](path,CheckpointFixture(),dict(test_only=True))
            payload = candidate.ns['torch_load_weights'](path,map_location='cpu')
            record('checkpoint_saved_toggle',payload['toggles']['RECURSIVE_SCHEDULE'],saved)
            candidate.ns['RECURSIVE_SCHEDULE'] = not saved
            candidate.ns['restore_eval_toggles'](payload)
            record('checkpoint_restored_toggle',candidate.ns['RECURSIVE_SCHEDULE'],saved)
            record('checkpoint_model_values',payload['model_state']['tiny.weight'].tolist(),[[0.,1.],[2.,3.],[4.,5.]])
            del payload
        for old in [{}, {'toggles':{}}]:
            candidate.ns['RECURSIVE_SCHEDULE'] = True
            candidate.ns['restore_eval_toggles'](old)
            record('old_checkpoint_resets_OFF',candidate.ns['RECURSIVE_SCHEDULE'],False,context=old)
        candidate.ns['restore_eval_toggles']({'toggles':{'RECURSIVE_SCHEDULE':True,'ARBITRARY_PAYLOAD_GLOBAL':17}})
        record('checkpoint_unknown_key_ignored','ARBITRARY_PAYLOAD_GLOBAL' in candidate.ns,False)
        try:
            candidate.ns['restore_eval_toggles']({'toggles':[]})
        except ValueError as exc:
            record('checkpoint_bad_toggle_type',str(exc),'checkpoint toggles must be a dictionary')
        else: record('checkpoint_bad_toggle_type',False,True)
    gc.collect()
    record('tensor_memory_bound',memory.peak < 64*1024**2,True)
    return dict(torch=torch.__version__,python=platform.python_version(),device='cpu',
                threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                schedule_dtype='Python binary64; independent Decimal precision 60',
                optimizer_dtypes=['FP32 parameters/state','FP32 or BF16 Newton-Schulz computation'],
                tracked_peak_tensor_storage_bytes_overcounting_views=memory.peak,
                max_single_tensor_storage_bytes=memory.max_single_storage,dispatch_calls=memory.calls,
                models_instantiated=0,largest_parameter_shape=[3,2],max_live_optimizer_fixtures=1)


environment = None
error = None
try:
    environment = run()
except Exception:
    error = traceback.format_exc()
    print(error)
finally:
    elapsed = time.perf_counter()-START
    OUT.joinpath('extractions.json').write_text(json.dumps(EXTRACTIONS,indent=2)+'\n')
    OUT.joinpath('exactness_cases.json').write_text(json.dumps(ROWS,indent=2,allow_nan=False)+'\n')
    summary = dict(status='PASS' if not error and all(r['passed'] for r in ROWS) else 'REFUSE',
                   checks=len(ROWS), failed_checks=sum(not r['passed'] for r in ROWS),
                   runtime_seconds=elapsed,environment=environment,error=error,
                   tolerance_policy='zero for frozen float/byte identity; absolute 5e-16, relative 0 for independent Decimal schedule/group oracle')
    OUT.joinpath('exactness_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    OUT.joinpath('exactness_failures.json').write_text(json.dumps([r for r in ROWS if not r['passed']],indent=2)+'\n')
    print(json.dumps(summary,indent=2))
raise SystemExit(summary['status'] != 'PASS')
