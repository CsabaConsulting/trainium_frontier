"""Bounded author checks of the exact batch/LR + Recursive composition.

One process, one CPU thread, 4 GiB address-space cap, synthetic tiny inputs.
This program prints its receipt; it never writes checkpoints or reads a dataset.
No native, distributed, full113M, independent-review or launch claim.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import resource
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
import ast
import copy
import difflib
import gc
import hashlib
import inspect
import json
import math
import platform
import random
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path
import torch

STARTED = time.monotonic()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = ROOT / 'research/v2/proposals/20260926-sprint-p0-clean-batch/train.py'
DONOR = ROOT / 'research/v2/proposals/20260926-sprint-p0-clean-recursive/train.py'
COMMON = PARENT.with_name('base.py')
CANDIDATE = HERE / 'train.py'
PINS = {
    PARENT: '8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee',
    DONOR: '50fa6d144f19758dcdf9980e03f2302453174b956f0fa1a3e0845c4c0b72e6fb',
    COMMON: 'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43',
    CANDIDATE: 'c2e27ccde59e619d74102429e340716dbec16a6ba538ea9beca9c714d0d33f87',
}
for path, expected in PINS.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
texts = {path: path.read_text() for path in PINS}
trees = {path: ast.parse(text) for path, text in texts.items()}


def named(tree, name):
    return next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


# Derive only the six inserted batch-port statements from its frozen parent.
common_main = named(trees[COMMON], 'main')
parent_main = named(trees[PARENT], 'main')
matcher = difflib.SequenceMatcher(a=list(map(dump, common_main.body)), b=list(map(dump, parent_main.body)), autojunk=False)
batch_nodes = []
for kind, a0, a1, b0, b1 in matcher.get_opcodes():
    if kind == 'equal':
        continue
    assert kind == 'insert', kind
    batch_nodes.extend(parent_main.body[b0:b1])
assert len(batch_nodes) == 6
batch_dumps = {dump(n) for n in batch_nodes}


def remove_batch_port(path):
    lines = texts[path].splitlines(keepends=True)
    found = [n for n in named(trees[path], 'main').body if dump(n) in batch_dumps]
    assert len(found) == 6
    skip = {i for n in found for i in range(n.lineno - 1, n.end_lineno)}
    return ''.join(line for i, line in enumerate(lines) if i not in skip)


assert remove_batch_port(PARENT) == texts[COMMON]
assert remove_batch_port(CANDIDATE) == texts[DONOR]
for name in ('startup_adjusted_schedule_elapsed', 'load_for_eval', 'save_checkpoint', 'GPT', 'OwnedOptimizer', '_muon_math'):
    assert ast.get_source_segment(texts[PARENT], named(trees[PARENT], name)) == ast.get_source_segment(texts[CANDIDATE], named(trees[CANDIDATE], name)), name


def load(path, name):
    # Remove only optional data imports. All scientific definitions execute
    # unchanged; main() is not called. Real prepare/evaluator/data never run.
    tree = copy.deepcopy(trees[path])
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.ImportFrom) and n.module == 'prepare'
        or isinstance(n, ast.Import) and any(a.name == 'pyarrow.parquet' for a in n.names)
    )]
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    mod.__dict__.update(print0=lambda *args, **kwargs: None,
                        TOKENIZER_VOCAB_SIZE=8192,
                        TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900,
                        TRAIN_STARTUP_STEPS_EXCLUDED=5,
                        TRAIN_TIME_BUDGET_SECONDS=1800)
    exec(compile(tree, str(path), 'exec'), mod.__dict__)
    main = named(trees[path], 'main')
    loop = next(n for n in main.body if isinstance(n, ast.While))
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'lrm')
    end = next(i for i, n in enumerate(loop.body[start:], start) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'group')
    mod._schedule_block = compile(ast.Module(body=copy.deepcopy(loop.body[start:end + 1]), type_ignores=[]), str(path) + ':actual-group-routing', 'exec')
    return mod


parent = load(PARENT, 'batch_stack_parent')
candidate = load(CANDIDATE, 'batch_stack_candidate')
donor = load(DONOR, 'batch_stack_donor')


def cli_prefix(mod, path, argv):
    node = copy.deepcopy(named(trees[path], 'main'))
    stop = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'device_type')
    node.name = '_author_cli_prefix'
    node.body = node.body[:stop] + [ast.Return(value=ast.Name(id='args', ctx=ast.Load()))]
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(code, str(path) + ':cli-prefix', 'exec'), mod.__dict__)
    old = sys.argv
    try:
        sys.argv = [str(path)] + argv
        return mod._author_cli_prefix()
    finally:
        sys.argv = old


defaults = cli_prefix(candidate, CANDIDATE, [])
assert (defaults.total_batch_size, defaults.matrix_lr, defaults.recursive_schedule) == (131072, 0.015, False)
proposal = ROOT / 'research/v2/proposals/20260926-batch-lr021-recursive-stack/EXPERIMENT_PROPOSAL.json'
target_argv = json.loads(proposal.read_text())['candidate']['argv']
target = cli_prefix(candidate, CANDIDATE, target_argv)
assert (candidate.TOTAL_BATCH_SIZE, candidate.MATRIX_LR, candidate.RECURSIVE_SCHEDULE) == (262144, 0.021, True)
assert (target.n_embd, target.seq_len, target.pack_factor, target.seed, target.max_train_seconds) == (1024, 1024, 32, 42, 1800)
assert candidate.TOTAL_BATCH_SIZE // (candidate.DEVICE_BATCH_SIZE * target.pack_factor * target.seq_len * 4) == 2
off = cli_prefix(candidate, CANDIDATE, target_argv + ['--no-recursive-schedule'])
assert off.recursive_schedule is False


def route(mod, groups, step, elapsed):
    ns = dict(mod.__dict__)
    ns.update(optimizer=types.SimpleNamespace(param_groups=groups), step=step,
              schedule_elapsed=elapsed, args=types.SimpleNamespace(max_train_seconds=1800.0))
    exec(mod._schedule_block, ns)
    return groups


# Reuse the donor's existing 40 boundary cases and include adjacent-float
# witnesses at both cooldown transitions. Compare exact float encodings.
elapsed_cases = [-1., 0., 90., 630., 900., 1799., 1800., 1801.]
for p in (1.0 - 0.95, 1.0 - 0.65):
    elapsed_cases += [math.nextafter(p, -math.inf) * 1800, p * 1800,
                      math.nextafter(p, math.inf) * 1800]
candidate.RECURSIVE_SCHEDULE = donor.RECURSIVE_SCHEDULE = True
scalar_cases = routing_cases = 0
for step in (0, 1, 299, 300, 2000):
    for elapsed in elapsed_cases:
        p = elapsed / 1800
        for kind in ('muon', 'adamw', 'other'):
            assert candidate.lr_multiplier(step, elapsed, 1800, kind).hex() == donor.lr_multiplier(step, elapsed, 1800, kind).hex()
        assert candidate.muon_momentum(step, p).hex() == donor.muon_momentum(step, p).hex()
        assert candidate.muon_beta2(p).hex() == donor.muon_beta2(p).hex()
        groups = [dict(kind=kind, initial_lr=lr, no_lr_decay=flat) for kind, lr in (('muon', .021), ('adamw', .021), ('adamw', .3), ('rmsprop', .2)) for flat in (False, True)]
        got = route(candidate, copy.deepcopy(groups), step, elapsed)
        expected = route(donor, copy.deepcopy(groups), step, elapsed)
        assert got == expected
        for group in got:
            if group['no_lr_decay']:
                assert group['lr'] == group['initial_lr']
        scalar_cases += 1
        routing_cases += len(groups)


def digest(value):
    h = hashlib.sha256()

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu().contiguous()
            h.update(str((str(t.dtype), tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=lambda k: str(k)):
                h.update(repr(key).encode())
                visit(obj[key])
        elif isinstance(obj, (tuple, list)):
            h.update(type(obj).__name__.encode())
            for item in obj:
                visit(item)
        else:
            h.update(repr(obj).encode())
    visit(value)
    return h.hexdigest()


def run(mod, total_batch, matrix_lr, accumulation, recursive):
    mod.TOTAL_BATCH_SIZE = total_batch
    mod.MATRIX_LR = matrix_lr
    if hasattr(mod, 'RECURSIVE_SCHEDULE'):
        mod.RECURSIVE_SCHEDULE = recursive
    mod.COMPUTE_DTYPE = torch.bfloat16
    mod._MUON_COMPILED.clear()
    random.seed(42)
    torch.manual_seed(42)
    config = mod.GPTConfig(sequence_len=32, vocab_size=256, n_layer=8, n_head=1, n_embd=128)
    model = mod.GPT(config)
    model.init_weights()
    optimizer = model.setup_optimizer()
    cached = [m for m in model.modules() if isinstance(m, mod.Linear)]
    named_params = {id(p): name for name, p in model.named_parameters()}
    groups = [dict(kind=g['kind'], initial_lr=g['initial_lr'], names=[named_params[id(p)] for p in g['params']]) for g in optimizer.param_groups]
    conv = [g for g in groups if any(name.endswith('local_conv.weight') for name in g['names'])]
    assert conv and all(g['kind'] == 'adamw' and g['initial_lr'] == matrix_lr for g in conv)
    assert all(g['initial_lr'] == matrix_lr for g in groups if g['kind'] == 'muon')
    # Observe the actual optimized CPU fallback call without changing arithmetic.
    original_math = mod._muon_math
    math_signature = inspect.signature(original_math)
    transport_types = []

    def observe(*args, **kwargs):
        beta = math_signature.bind(*args, **kwargs).arguments['one_beta2']
        transport_types.append('tensor' if isinstance(beta, torch.Tensor) else 'float')
        return original_math(*args, **kwargs)
    mod._muon_math = observe
    steps = []
    initial = digest(model.state_dict())
    for step, elapsed in enumerate((0., 900., 1764.)):
        if step == 0:
            for layer in cached:
                layer.refresh_weight_cache(mod.COMPUTE_DTYPE)
        optimizer.zero_grad(set_to_none=True)
        inputs, losses = [], []
        for micro in range(accumulation):
            x = torch.randint(0, 256, (1, 32))
            y = torch.randint(0, 256, (1, 32))
            inputs.append((x, y))
            loss = model(x, y)
            assert torch.isfinite(loss)
            losses.append(float(loss.detach()))
            (loss / accumulation * mod.LOSS_SCALE).backward()
        mod.clip_and_sync_gradients(model, list(model.parameters()), max_norm=mod.GRAD_CLIP)
        grads = digest({name: p.grad for name, p in model.named_parameters()})
        route(mod, optimizer.param_groups, step, elapsed)
        optimizer.step()
        for layer in cached:
            layer.refresh_weight_cache(mod.COMPUTE_DTYPE)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        steps.append(dict(losses=losses, input_sha256=digest(inputs), gradient_sha256=grads,
                          model_sha256=digest(model.state_dict()), optimizer_sha256=digest(optimizer.state_dict()),
                          rng_sha256=digest(torch.get_rng_state())))
    assert set(transport_types) == ({'tensor'} if recursive else {'float'})
    mod._muon_math = original_math
    mod._MUON_COMPILED.clear()
    result = dict(config=asdict(config), parameters=sum(p.numel() for p in model.parameters()),
                  total_batch_argument=total_batch, matrix_lr=matrix_lr,
                  reduced_accumulation=accumulation, initial_model_sha256=initial,
                  optimizer_groups=groups, steps=steps,
                  actual_beta2_transport=sorted(set(transport_types)),
                  actual_muon_math_calls=len(transport_types))
    del model, optimizer, cached
    gc.collect()
    return result


checks = []
for total_batch, lr, accumulation in ((131072, .015, 1), (262144, .021, 2)):
    a = run(parent, total_batch, lr, accumulation, False)
    b = run(candidate, total_batch, lr, accumulation, False)
    assert a == b, 'OFF3 mismatch'
    checks.append(dict(check='parent_vs_candidate_OFF3', result=b))
# The tiny ON witness also exercises the changed 1-beta2 tensor transport with
# the real optimizer, using the exact donor and requested LR/accumulation.
a = run(donor, 262144, .021, 2, True)
b = run(candidate, 262144, .021, 2, True)
assert a == b, 'donor ON3 mismatch'
checks.append(dict(check='donor_vs_candidate_ON3', result=b))

print(json.dumps(dict(
    passed=True, evidence_role='implementer_author_checks_not_independent_review',
    source_pins={str(p.relative_to(ROOT)): h for p, h in PINS.items()},
    test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    literal_composition='Removing only the six byte-exact batch-port statements reproduces exact50fa; removing them from exact8c6 reproduces exacte1e2.',
    full_cli_parse=True, default_flag_off=True, explicit_no_flag=True,
    schedule_boundary_cases=scalar_cases, group_routing_cases=routing_cases,
    checks=checks, atol=0, rtol=0, seed=42,
    python=platform.python_version(), torch=torch.__version__, threads=torch.get_num_threads(),
    address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    elapsed_seconds=time.monotonic() - STARTED,
    limits=dict(synthetic_data_only=True, model_layers=8, model_width=128,
                vocab_size=256, sequence_length=32, actual_device='cpu',
                global_batch_not_materialized=True, microbatches_per_step=[1, 2],
                native=False, distributed=False, owned_optimizer_runtime=False,
                full113M=False, checkpoint_io=False, bpB_or_speed_claim=False,
                independent_review=False, launch_authorization=False)
), indent=2))
