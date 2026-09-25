"""Independent bounded review test; never imports or runs the full training source.

The exact while-loop AST and selected CPU-capable definitions are extracted from
the verified dependency. A tiny, expressly synthetic model/data/runtime fixture
supplies its dependencies. Candidate LoopTrace and run_exact are unmodified.
This tests instrumentation neutrality, not native/distributed training identity.
"""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import runpy
import sys
import traceback
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT = Path('/work')
SOURCE = ROOT / 'recursive6.py'
WRAPPER = ROOT / 'profile_actual_loop.py'
RAW = SOURCE.read_text()
TREE = ast.parse(RAW)
spec = importlib.util.spec_from_file_location('reviewed_wrapper', WRAPPER)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
PLAN = wrapper.landmarks(RAW)
MAIN = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
LOOP = next(n for n in MAIN.body if isinstance(n, ast.While))


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone(v) for v in value)
    return copy.deepcopy(value)


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def summary(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest(value))
    if isinstance(value, dict):
        return {str(k): summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [summary(v) for v in value]
    return value


def exact(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a, b) and digest(a) == digest(b), path
        if a.is_floating_point():
            assert torch.isfinite(a).all(), path
        return a.numel()
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        return sum(exact(a[k], b[k], f'{path}.{k}') for k in a)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        return sum(exact(x, y, f'{path}[{i}]') for i, (x, y) in enumerate(zip(a, b)))
    assert a == b, (path, a, b)
    return 0


class Clock:
    """Separate fixture clock: trace's real monotonic calls cannot advance it."""
    def __init__(self):
        self.ticks = 0

    def monotonic(self):
        self.ticks += 1
        return self.ticks / 1000


class TinyModel(nn.Module):
    def __init__(self, fixture):
        super().__init__()
        self.fixture = fixture
        self.embedding = nn.Embedding(32, 16)
        self.matrix = nn.Linear(16, 16, bias=False)
        self.head = nn.Linear(16, 32, bias=False)

    def forward(self, x, y):
        self.fixture.calls.append(('forward', self.fixture.step))
        if self.fixture.fail_forward and self.fixture.step == 20:
            raise RuntimeError('INJECTED_FORWARD_FAILURE')
        x = self.embedding(x).to(self.fixture.dtype)
        x = F.linear(x, self.matrix.weight.to(self.fixture.dtype))
        x = F.scaled_dot_product_attention(x[:, None], x[:, None], x[:, None], is_causal=True)[:, 0]
        x = F.dropout(x, p=0.1, training=True)
        logits = F.linear(x, self.head.weight.to(self.fixture.dtype)).float()
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        self.fixture.losses.append(loss.detach().clone())
        return loss

    def refresh_weight_cache(self, dtype):
        self.fixture.calls.append(('cache', self.fixture.step))
        self.fixture.cache = self.matrix.weight.detach().to(dtype).clone()


NAMES = {
    'MuonAdamW', '_muon_math', '_muon_lr_g', 'unscale_grads_',
    'clip_and_sync_gradients', 'sync_gradients', 'sync_gradients_flat',
    'clip_grad_norm_flat_', 'lr_multiplier', 'batch_ramp_stage',
    'batch_ramp_grad_accum', 'muon_momentum', 'muon_beta2', 'demon_beta1_scale',
}


class Fixture:
    def __init__(self, dtype, steps=22, accumulation=2, fail_forward=False, startup_origin=0.0):
        torch.manual_seed(42)
        random.seed(42)
        self.dtype, self.fail_forward = dtype, fail_forward
        self.step, self.fetch_count = 0, 0
        self.calls, self.losses, self.snapshots = [], [], []
        self.clock = Clock()
        self.generator = torch.Generator().manual_seed(42)
        self.model = TinyModel(self)
        self.initial_parameters = clone(self.model.state_dict())
        namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math, '__name__': 'cpu_reference'}
        # Only literal assignments are evaluated; no dependency module imports.
        for node in TREE.body:
            if isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        namespace[target.id] = copy.deepcopy(value)
        namespace.update(COMPUTE_DTYPE=dtype, RECURSIVE_SCHEDULE=True,
                         USE_FOREACH_OPTIM=True, USE_TENSOR_LR_SCALARS=True,
                         USE_COMPILE_MUON=True, USE_NKI_MUON=False, _HAS_NKI=False,
                         USE_CAUTIOUS_WD=False, USE_MUON_LR_FP32=False,
                         _MUON_COMPILED={}, LOSS_SCALE=8.0, BATCH_RAMP=False,
                         LOSS_SCALE_DYNAMIC=False, NGRAM_VE_FREEZE_EPOCH=0,
                         USE_WEIGHT_EMA=False, ADAM_EVERY_N=1, DEMON_BETA1=False,
                         CLIP_AFTER_REDUCE=True, GRAD_CLIP=1.0,
                         TRAIN_STARTUP_STEPS_EXCLUDED=2,
                         TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900.0,
                         _PROCESS_STARTED=startup_origin,
                         time=self.clock, fixture=self,
                         print0=lambda *a, **k: None,
                         print=lambda *a, **k: None,
                         gc=SimpleNamespace(collect=lambda: None, freeze=lambda: None, disable=lambda: None))
        # A declared identity-collective fixture: no rank/device coverage claimed.
        collective_ns = {'fixture': self}
        exec(compile('def all_reduce(tensor, op=None):\n    fixture.calls.append(("all_reduce", fixture.step, str(op), tuple(tensor.shape)))\n',
                     '/work/cpu_fixture/distributed_c10d.py', 'exec'), collective_ns)
        namespace['dist'] = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True,
                                           ReduceOp=SimpleNamespace(AVG='AVG', MAX='MAX'),
                                           all_reduce=collective_ns['all_reduce'])
        namespace['synchronize'] = lambda device: self.calls.append(('sync', self.step))
        namespace['record_step_boundary'] = lambda step: self.calls.append(('boundary', step))
        namespace['debug_numerics'] = self.debug
        definitions = [copy.deepcopy(n) for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NAMES]
        assert {n.name for n in definitions} == NAMES
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), 'exec',
                     flags=__import__('__future__').annotations.compiler_flag), namespace)
        groups = [dict(kind='adamw', params=[self.model.embedding.weight, self.model.head.weight],
                       lr=0.002, initial_lr=0.002, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01),
                  dict(kind='muon', params=[self.model.matrix.weight], lr=0.015, initial_lr=0.015,
                       momentum=0.95, beta2=0.9, ns_steps=3, weight_decay=0.01)]
        self.optimizer = namespace['MuonAdamW'](groups)
        x, y, loader_state = self.fetch()
        self.initial = dict(model=self.model, orig_model=self.model, optimizer=self.optimizer,
                            x=x, y=y, loader_state=loader_state, fetch=self.fetch,
                            step=0, args=SimpleNamespace(num_steps=steps, max_train_seconds=1800.0, clip_foreach=True),
                            device=torch.device('cpu'), rank=0, world_size=1,
                            world_tokens_per_micro=16, grad_accum_steps=accumulation,
                            grad_accum_now=accumulation, grad_flat_buffer=None, adamw_skip_ids=frozenset(),
                            clip_main_params=list(self.model.parameters()), clip_ngram_params=[],
                            cached_linears=[self.model], budget_started=0.0, loop_started=0.0,
                            smooth_loss=0.0, total_train_tokens=0, recent_step_times=[],
                            startup_allowance_est=0.0, tables_frozen_logged=False)
        # Retain original source line numbers and filename in the unchanged loop.
        prelude = ast.parse('def main():\n    global LOSS_SCALE, loss_scale_clean, loss_scale_skips\n' +
                            ''.join(f'    {name} = fixture.initial[{name!r}]\n' for name in self.initial)).body[0]
        tail = ast.parse('return fixture.finish(locals())').body[0]
        ast.increment_lineno(tail, LOOP.end_lineno)
        prelude.body += [copy.deepcopy(LOOP), tail]
        ast.fix_missing_locations(prelude)
        assert ast.dump(prelude.body[-2], include_attributes=True) == ast.dump(LOOP, include_attributes=True)
        exec(compile(ast.Module(body=[prelude], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        self.namespace = namespace
        self.main = namespace['main']

    def fetch(self):
        self.calls.append(('fetch', self.step, self.fetch_count))
        self.fetch_count += 1
        return (torch.randint(32, (2, 8), generator=self.generator),
                torch.randint(32, (2, 8), generator=self.generator), {'epoch': 1})

    def debug(self, stage, step_number, loss, inspect_tensors=False):
        self.step = step_number
        # First forward at a new step precedes debug, so the boundary also advances step below.
        if stage == 'optimizer':
            self.snapshots.append(dict(parameters=clone(self.model.state_dict()),
                                       gradients=[clone(p.grad) for p in self.model.parameters()],
                                       optimizer=clone(self.optimizer.state_dict()),
                                       loss=clone(loss)))
        if stage == 'weight_cache':
            pass

    def finish(self, local):
        self.calls.append(('finish', local['step']))
        return dict(losses=self.losses, snapshots=self.snapshots,
                    parameters=clone(self.model.state_dict()), optimizer=clone(self.optimizer.state_dict()),
                    rng=torch.get_rng_state().clone(), python_rng=random.getstate(),
                    data_rng=self.generator.get_state().clone(), cache=clone(self.cache),
                    calls=self.calls, fetch_count=self.fetch_count,
                    steps=local['step'], total_tokens=local['total_train_tokens'],
                    startup_allowance=local['startup_allowance_est'], budget_started=local['budget_started'])


class CpuEngine:
    def __init__(self, name, fixture):
        self.name, self.fixture = name, fixture
        self.lifecycle, self.regions = [], []
        self.depth = 0
        self.prof = None
        self.running = False
        self.events = []

    def prepare(self, frame):
        self.lifecycle.append(['prepare', frame.f_locals['step'], len(self.fixture.snapshots)])

    def start(self, frame):
        self.lifecycle.append(['start', frame.f_locals['step'], len(self.fixture.snapshots)])
        self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True)
        self.prof.__enter__()
        self.running = True

    @contextlib.contextmanager
    def region(self, label):
        self.depth += 1
        self.regions.append(['enter', label, self.fixture.step])
        try:
            with torch.profiler.record_function('P/' + label):
                yield
        finally:
            self.regions.append(['exit', label, self.fixture.step])
            self.depth -= 1

    def close(self):
        if self.running:
            self.prof.__exit__(None, None, None)
            self.running = False
            self.prof.export_chrome_trace(str(ROOT / f'{self.name}.trace.json'))
            self.events = [dict(name=e.name, cpu_time_total=e.cpu_time_total,
                                input_shapes=e.input_shapes) for e in self.prof.events()]

    def stop(self, frame):
        self.lifecycle.append(['stop', frame.f_locals['step'], len(self.fixture.snapshots)])
        assert self.depth == 0
        self.close()

    def abort(self):
        self.lifecycle.append(['abort'])
        self.close()


def execute(fixture, name, capture):
    # Boundary callback is fixture bookkeeping, identical OFF/ON.
    def boundary(step):
        fixture.calls.append(('boundary', step))
        fixture.step = step + 1
    fixture.namespace['record_step_boundary'] = boundary
    engine = CpuEngine(name, fixture)
    trace = wrapper.LoopTrace(SOURCE, PLAN, engine, target=20)
    try:
        if capture:
            sys.settrace(trace)
        value = fixture.main()
    except BaseException:
        sys.settrace(None)
        trace.abort()
        raise
    finally:
        sys.settrace(None)
    return value, engine, trace


RESULTS = []


def test(name, fn):
    try:
        evidence = fn()
        result = dict(name=name, status='PASS', evidence=evidence)
    except BaseException as error:
        result = dict(name=name, status='FAIL', error=repr(error), traceback=traceback.format_exc())
    RESULTS.append(result)
    print(json.dumps(result, sort_keys=True), flush=True)


def binding_and_arguments():
    assert wrapper.sha(SOURCE) == '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'
    assert wrapper.sha(WRAPPER) == 'eaad6aff2e18a825eb8ddca91aed5efa72c91a385eb02cbdaee36b01c2644f7d'
    expected = {3965: 'existing_synchronize', 3977: 'forward', 3980: 'backward',
                3982: 'data_fetch', 4007: 'gradient_unscale', 4008: 'clip_and_reduce',
                4027: 'optimizer', 4039: 'weight_cache_refresh',
                4041: 'existing_synchronize', 4042: 'record_boundary'}
    assert PLAN == {'loop_line': 3953, 'statements': expected}, PLAN
    args = '--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015 --seed 42 --max-train-seconds 1800 --num-steps 22 --out-dir /work/NEVER_LAUNCHED --recursive-schedule'.split()
    assert wrapper.launch_arguments('recursive6', '/work/NEVER_LAUNCHED') == args
    wrapper.check_capture_arguments('recursive6', args)
    invalid = [args[:-1], args + ['--seed', '43'], args + ['--out-dir', '/work/other'],
               [x if x != '22' else '21' for x in args], args[2:] + args[:2]]
    for wrong in invalid:
        try:
            wrapper.check_capture_arguments('recursive6', wrong)
        except RuntimeError:
            pass
        else:
            raise AssertionError(wrong)
    # Check CLI default OFF via a recording call double, never launch exact native argv.
    with mock.patch.object(sys, 'argv', ['wrapper', '--train-py', str(SOURCE), '--', '--fixture-only']), \
         mock.patch.object(wrapper, 'run_exact') as run, mock.patch.object(wrapper, 'NativeEngine') as native:
        assert wrapper.main() == 0
        run.assert_called_once_with(SOURCE, ['--fixture-only'])
        native.assert_not_called()
    return dict(hashes_match=True, exact_landmarks=PLAN, rejected_argument_variants=len(invalid), default_off=True)


def parity(dtype):
    label = 'float32' if dtype == torch.float32 else 'bfloat16'
    off_fixture = Fixture(dtype)
    off, off_engine, off_trace = execute(off_fixture, label + '_off', False)
    on_fixture = Fixture(dtype)
    on, engine, trace = execute(on_fixture, label + '_on', True)
    elements = exact(off, on)
    assert on['steps'] == 22 and on['fetch_count'] == 45 and on['total_tokens'] == 704
    assert len(on['snapshots']) == 22 and len(on['losses']) == 44
    assert engine.lifecycle == [['prepare', 0, 0], ['start', 20, 20], ['stop', 21, 21]], engine.lifecycle
    assert not off_engine.lifecycle and not off_trace.prepared
    assert trace.finished and not trace.active and not trace.scopes and not trace.function_scopes
    assert [x['completed_step'] for x in trace.warm_steps] == list(range(21))
    assert engine.depth == 0 and all(r[2] in (20, 21) for r in engine.regions)
    assert sum(c[0] == 'sync' for c in on['calls']) == 44
    labels = [e['name'] for e in engine.events if e['name'].startswith('P/')]
    from collections import Counter
    counts = Counter(labels)
    required = {'P/forward': 2, 'P/backward': 2, 'P/data_fetch': 2,
                'P/existing_synchronize': 2, 'P/gradient_unscale': 1,
                'P/clip_and_reduce': 1, 'P/optimizer': 1, 'P/weight_cache_refresh': 1,
                'P/record_boundary': 1, 'P/optimizer_muon_group_inclusive': 1,
                'P/optimizer_muon_scalar_upload': 1, 'P/optimizer_muon_eager_math_if_executed': 1,
                'P/optimizer_adamw_group_inclusive': 1, 'P/all_reduce_host_call': 4}
    assert counts == required, counts
    assert any('embedding' in e['name'].lower() and 'backward' in e['name'].lower() for e in engine.events)
    assert any(not torch.equal(off_fixture.initial_parameters[k], off['parameters'][k]) for k in off['parameters'])
    # Save full raw tensors, including every update's gradients and optimizer state.
    torch.save({'off': off, 'on': on}, ROOT / f'{label}.raw.pt')
    (ROOT / f'{label}.raw.json').write_text(json.dumps(dict(off=summary(off), on=summary(on),
         engine=engine.lifecycle, regions=engine.regions, warm_boundaries=trace.warm_steps,
         profiler_events=engine.events), indent=2) + '\n')
    return dict(bitwise_equal=True, compared_tensor_elements=elements, steps=22,
                losses=44, snapshots=22, fetches=45, existing_sync_calls=44,
                lifecycle=engine.lifecycle, annotation_counts=counts,
                final_parameter_hashes={k: digest(v) for k, v in on['parameters'].items()})


def abort_and_early_exit():
    fixture = Fixture(torch.float32, steps=20)
    _, engine, trace = execute(fixture, 'early_exit', True)
    assert not trace.finished and not trace.active
    try:
        wrapper.require(trace.finished, 'SOURCE_EXITED_BEFORE_PROFILE_WINDOW_COMPLETED')
    except RuntimeError as error:
        assert str(error) == 'SOURCE_EXITED_BEFORE_PROFILE_WINDOW_COMPLETED'
    else:
        raise AssertionError('early exit accepted')
    fixture = Fixture(torch.float32, fail_forward=True)
    def boundary(step):
        fixture.calls.append(('boundary', step))
        fixture.step = step + 1
    fixture.namespace['record_step_boundary'] = boundary
    engine = CpuEngine('exception', fixture)
    trace = wrapper.LoopTrace(SOURCE, PLAN, engine, target=20)
    try:
        sys.settrace(trace)
        fixture.main()
    except RuntimeError as error:
        sys.settrace(None)
        assert str(error) == 'INJECTED_FORWARD_FAILURE'
        trace.abort()
    finally:
        sys.settrace(None)
    assert not trace.finished and not trace.active and not trace.scopes and not trace.function_scopes
    assert engine.depth == 0 and not engine.running
    (ROOT / 'exception.raw.json').write_text(json.dumps(dict(lifecycle=engine.lifecycle, regions=engine.regions), indent=2) + '\n')
    return dict(early_exit_refused=True, injected_failure_preserved=True,
                exception_scopes_balanced=True, exception_lifecycle=engine.lifecycle)


def runpy_contract():
    path = ROOT / 'runpy_fixture.py'
    path.write_text('import sys\nimport torch\nOBSERVED = (list(sys.argv), sys.path[0], __name__)\ntorch.manual_seed(42)\nVALUE = torch.randn(2, 8)\nif "--raise" in sys.argv:\n    raise RuntimeError("fixture failure")\n')
    prior_argv, prior_path, prior_trace = sys.argv, sys.path[:], sys.gettrace()
    a = wrapper.run_exact(path, ['--token', '42'])
    with mock.patch.object(sys, 'argv', [str(path), '--token', '42']):
        sys.path.insert(0, str(path.parent))
        try:
            b = runpy.run_path(str(path), run_name='__main__')
        finally:
            sys.path.pop(0)
    exact(a['VALUE'], b['VALUE'])
    assert a['OBSERVED'] == b['OBSERVED'] == ([str(path), '--token', '42'], str(ROOT), '__main__')
    try:
        wrapper.run_exact(path, ['--raise'], lambda *a: None)
    except RuntimeError as error:
        assert str(error) == 'fixture failure'
    else:
        raise AssertionError('exception swallowed')
    assert sys.argv is prior_argv and sys.path == prior_path and sys.gettrace() is prior_trace
    hook = lambda *a: None
    try:
        sys.settrace(hook)
        try:
            wrapper.run_exact(path, [])
        except RuntimeError as error:
            assert str(error) == 'EXISTING_TRACE_HOOK_REFUSED'
        else:
            raise AssertionError('existing trace accepted')
        assert sys.gettrace() is hook
    finally:
        sys.settrace(prior_trace)
    return dict(argv_path_main_semantics=True, tensor_exact=True, exception_restoration=True, existing_trace_refused=True)


def startup_excess():
    off, _, _ = execute(Fixture(torch.float32, startup_origin=-1000), 'startup_off', False)
    on, _, trace = execute(Fixture(torch.float32, startup_origin=-1000), 'startup_on', True)
    exact(off, on)
    assert on['startup_allowance'] == 900.0
    assert on['budget_started'] < -130 + 0.001 and trace.finished
    # The source uses elapsed wallclock in its schedule: illustrate that it is sensitive.
    ns = Fixture(torch.float32).namespace
    early, later = ns['lr_multiplier'](20, 0, 1800), ns['lr_multiplier'](20, 900, 1800)
    assert early != later
    return dict(cap_seconds=on['startup_allowance'], budget_started=on['budget_started'],
                wallclock_lr_at_0=early, wallclock_lr_at_900=later,
                note='Fixture defines imported prepare constants; this does not verify absent prepare.py or Operator accounting.')


if __name__ == '__main__':
    test('source_binding_exact_argv_landmarks_and_default_off', binding_and_arguments)
    test('float32_real_loop_cpu_parity', lambda: parity(torch.float32))
    test('bfloat16_real_loop_cpu_parity', lambda: parity(torch.bfloat16))
    test('early_exit_and_exception_cleanup', abort_and_early_exit)
    test('run_exact_runpy_contract', runpy_contract)
    test('startup_cap_excess_and_wallclock_sensitivity', startup_excess)
    payload = dict(status='PASS' if all(r['status'] == 'PASS' for r in RESULTS) else 'FAIL',
                   python=sys.version, torch=torch.__version__, threads=torch.get_num_threads(),
                   interop_threads=torch.get_num_interop_threads(), seed=42,
                   bounds=dict(width=16, context=8, vocabulary=32, microbatch=2, steps=22, accumulation=2),
                   scope='CPU fixture using unchanged extracted loop and optimizer definitions, identity collectives and deterministic fixture clock',
                   extracted_definitions=sorted(NAMES), results=RESULTS)
    (ROOT / 'own_test_results.json').write_text(json.dumps(payload, indent=2) + '\n')
    raise SystemExit(0 if payload['status'] == 'PASS' else 1)
