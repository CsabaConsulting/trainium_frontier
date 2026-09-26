"""Bounded warmdown parser/scalar/default-identity checks; no native claim."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import resource
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
import ast
import contextlib
import copy
import gc
import hashlib
import inspect
import io
import json
import math
import platform
import random
import sys
import time
import types
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import torch

STARTED = time.monotonic()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = ROOT / 'research/v2/proposals/20260926-sprint-p0-clean-batch/train.py'
CANDIDATE = HERE / 'train.py'
FIXTURE = ROOT / 'research/v2/lanes/sprint-batch-lr021-recursive-stack/author_cpu.py'
SPEC = ROOT / 'research/v2/proposals/20260926-batch-lr021-warmdown-flag/IMPLEMENTATION_SPEC.md'
PINS = {
    PARENT: '8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee',
    CANDIDATE: '4ec8b1b46562612cdcbfe3d7f66b0c8d25666bdbac3516b30bd9149c2dc08d2d',
    FIXTURE: '76339cfa66a8b96daa7fdb846e407b412de86fd636b6263539cd45a78a1bcbbf',
    SPEC: '61ea9e1df60b47f0c8097ad4f02dcb16c05ecc7dbb6aa60185e9fbf89330e8d2',
}
for path, expected in PINS.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
texts = {path: path.read_text() for path in (PARENT, CANDIDATE)}
trees = {path: ast.parse(text) for path, text in texts.items()}

# Reuse only these unchanged function bodies from the previously executed,
# hash-pinned bounded fixture. Do NOT import/run its Recursive-specific tests.
helper_names = {'named', 'dump', 'load', 'cli_prefix', 'route', 'digest', 'run'}
fixture_tree = ast.parse(FIXTURE.read_text())
helpers = [node for node in fixture_tree.body
           if isinstance(node, ast.FunctionDef) and node.name in helper_names]
assert {node.name for node in helpers} == helper_names
exec(compile(ast.Module(body=helpers, type_ignores=[]), str(FIXTURE), 'exec'))

# Prove all scientific code outside main byte-identical. Normalize precisely
# the four permitted main changes and prove the complete original AST remains.
parent_other = [node for node in trees[PARENT].body
                if not (isinstance(node, ast.FunctionDef) and node.name == 'main')]
candidate_other = [node for node in trees[CANDIDATE].body
                   if not (isinstance(node, ast.FunctionDef) and node.name == 'main')]
assert len(parent_other) == len(candidate_other)
for left, right in zip(parent_other, candidate_other):
    assert dump(left) == dump(right)
    assert ast.get_source_segment(texts[PARENT], left) == ast.get_source_segment(texts[CANDIDATE], right)
main = copy.deepcopy(named(trees[CANDIDATE], 'main'))
removed = []
for node in list(main.body):
    if isinstance(node, ast.Global) and 'WARMDOWN_RATIO' in node.names:
        assert node.names == ['TOTAL_BATCH_SIZE', 'MATRIX_LR', 'WARMDOWN_RATIO']
        node.names.remove('WARMDOWN_RATIO')
        removed.append('global')
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and node.value.args and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value == '--warmdown-ratio':
        main.body.remove(node)
        removed.append('parser')
    elif isinstance(node, ast.If) and any(isinstance(part, ast.Attribute) and part.attr == 'warmdown_ratio' for part in ast.walk(node.test)):
        main.body.remove(node)
        removed.append('validation')
    elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'WARMDOWN_RATIO' for target in node.targets):
        main.body.remove(node)
        removed.append('assignment')
assert sorted(removed) == ['assignment', 'global', 'parser', 'validation']
assert dump(main) == dump(named(trees[PARENT], 'main'))

parent = load(PARENT, 'warmdown_parent')
candidate = load(CANDIDATE, 'warmdown_candidate')
assert parent.WARMDOWN_RATIO == candidate.WARMDOWN_RATIO == .75
assert not hasattr(candidate, 'RECURSIVE_SCHEDULE') and not hasattr(candidate, 'K2_EMBEDDING')

def parse(argv, mod=candidate, path=CANDIDATE):
    # main normally executes once per process; reset its known scalar defaults
    # between independent parser cases instead of inheriting a previous case.
    mod.TOTAL_BATCH_SIZE, mod.MATRIX_LR, mod.WARMDOWN_RATIO = 131072, .015, .75
    return cli_prefix(mod, path, argv)

normal = ['--n-embd', '1024', '--no-ngram-ve', '--no-qk-shift',
          '--compile-sdpa-direct', '--nki-local-conv', '--bf16-norm-output',
          '--pack-factor', '32', '--clip-after-reduce', '--seq-len', '1024',
          '--no-eval-public', '--total-batch-size', '262144', '--matrix-lr', '.021']
accepted = []
for value in (None, '.75', '.60', '.85', '1'):
    argv = normal + ([] if value is None else ['--warmdown-ratio=' + value])
    args = parse(argv)
    expected = .75 if value is None else float(value)
    assert args.warmdown_ratio == candidate.WARMDOWN_RATIO == expected
    assert (candidate.TOTAL_BATCH_SIZE, candidate.MATRIX_LR) == (262144, .021)
    assert (args.seed, args.max_train_seconds, args.n_embd, args.seq_len, args.pack_factor) == (42, 1800, 1024, 1024, 32)
    assert candidate.TOTAL_BATCH_SIZE // (candidate.DEVICE_BATCH_SIZE * args.pack_factor * args.seq_len * 4) == 2
    accepted.append({'argument': value, 'value': expected})
defaults = parse([])
assert (defaults.total_batch_size, defaults.matrix_lr, defaults.warmdown_ratio) == (131072, .015, .75)
rejected = []
for value in ('0', '-0.1', '1.0000000000000002', 'nan', 'inf', '-inf'):
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            parse(normal + ['--warmdown-ratio=' + value])
    except SystemExit as exc:
        assert exc.code == 2
        assert '--warmdown-ratio must be finite and in (0, 1]' in stderr.getvalue()
        rejected.append({'argument': value, 'exit': exc.code, 'error': stderr.getvalue().splitlines()[-1]})
    else:
        raise AssertionError('Invalid ratio accepted: ' + value)
for argv in (['--total-batch-size=0'], ['--matrix-lr=nan'], ['--matrix-lr=-1']):
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            parse(argv)
    except SystemExit as exc:
        assert exc.code == 2 and 'positive total batch and finite positive matrix LR required' in stderr.getvalue()
    else:
        raise AssertionError('Existing invalid batch/LR accepted')

def reference(mod, step, elapsed, budget, ratio):
    if step < mod.WARMUP_STEPS:
        return float(Fraction(step + 1, max(1, mod.WARMUP_STEPS)))
    progress = min(1.0, elapsed / max(1e-6, budget))
    if progress < 1.0 - ratio:
        return 1.0
    # Independent exact-rational evaluation of the piecewise scalar contract;
    # compare with a small ULP allowance for the source's unchanged rounding.
    cooldown = (Fraction(1) - Fraction.from_float(progress)) / Fraction.from_float(ratio)
    return float(cooldown + (1 - cooldown) * Fraction.from_float(mod.FINAL_LR_FRAC))

scalar_cases = routing_cases = 0
largest_reference_error = 0.0
for ratio in (.60, .75, .85, 1.0):
    parent.WARMDOWN_RATIO = candidate.WARMDOWN_RATIO = ratio
    boundary = (1.0 - ratio) * 1800.0
    elapsed_cases = [-1.0, 0.0, math.nextafter(boundary, -math.inf), boundary,
                     math.nextafter(boundary, math.inf), 900.0, 1764.0, 1800.0, 1801.0]
    for warmup in (0, 300):
        parent.WARMUP_STEPS = candidate.WARMUP_STEPS = warmup
        for step in (0, 1, 299, 300, 2000):
            for elapsed in elapsed_cases:
                actual = candidate.lr_multiplier(step, elapsed, 1800.0)
                unchanged = parent.lr_multiplier(step, elapsed, 1800.0)
                assert actual.hex() == unchanged.hex(), 'Unchanged scalar arithmetic mismatch'
                expected = reference(candidate, step, elapsed, 1800.0, ratio)
                error = abs(actual - expected)
                assert error <= 4 * max(math.ulp(expected), math.ulp(actual)), (ratio, step, elapsed, actual, expected)
                largest_reference_error = max(largest_reference_error, error)
                groups = [dict(kind=kind, initial_lr=lr, no_lr_decay=flat)
                          for kind, lr in (('muon', .021), ('adamw', .021), ('adamw', .3), ('other', .2))
                          for flat in (False, True)]
                actual_groups = route(candidate, copy.deepcopy(groups), step, elapsed)
                assert actual_groups == route(parent, copy.deepcopy(groups), step, elapsed)
                candidate.WARMDOWN_RATIO = .75
                default_groups = route(candidate, copy.deepcopy(groups), step, elapsed)
                candidate.WARMDOWN_RATIO = ratio
                for group, original in zip(actual_groups, default_groups):
                    assert {k: v for k, v in group.items() if k != 'lr'} == {k: v for k, v in original.items() if k != 'lr'}
                    if group['no_lr_decay']:
                        assert group['lr'] == original['lr']
                scalar_cases += 1
                routing_cases += len(groups)

parent.WARMUP_STEPS = candidate.WARMUP_STEPS = 0
checks = []
for total_batch, lr, accumulation in ((131072, .015, 1), (262144, .021, 2)):
    parent.WARMDOWN_RATIO = .75
    expected = run(parent, total_batch, lr, accumulation, False)
    for explicit in (False, True):
        argv = ['--total-batch-size', str(total_batch), '--matrix-lr', str(lr)]
        if explicit:
            argv += ['--warmdown-ratio', '.75']
        parse(argv)
        actual = run(candidate, total_batch, lr, accumulation, False)
        assert actual == expected, 'Three-step default identity mismatch'
        checks.append({'check': 'explicit075_3step_identity' if explicit else 'omitted_flag_3step_identity',
                       'result': actual})

result = dict(
    passed=True, evidence_role='implementer_author_checks_not_independent_review',
    source_pins={str(path.relative_to(ROOT)): value for path, value in PINS.items()},
    test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    unchanged_other_top_level_nodes=len(parent_other), main_changes=removed,
    reused_fixture_functions=sorted(helper_names), parser_accepted=accepted, parser_rejected=rejected,
    existing_invalid_batch_lr_preserved=True, scalar_cases=scalar_cases,
    scalar_parent_hex_exact=True, rational_reference_tolerance_ulps=4,
    largest_rational_reference_abs_error=largest_reference_error,
    optimizer_group_routing_cases=routing_cases, checks=checks,
    atol=0, rtol=0, seed=42, python=platform.python_version(), torch=torch.__version__,
    threads=torch.get_num_threads(), address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    elapsed_seconds=time.monotonic() - STARTED,
    limits=dict(synthetic_data_only=True, layers=8, width=128, vocab=256, sequence_length=32,
                global_batch_not_materialized=True, accumulation=[1, 2], actual_device='cpu',
                native=False, compiler=False, distributed=False, full113M=False, checkpoint_io=False,
                evaluator=False, independent_review=False, launch_authorized=False))
(HERE / 'CPU_RECEIPT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
