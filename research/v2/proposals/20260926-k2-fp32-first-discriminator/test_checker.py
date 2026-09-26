#!/usr/bin/env python3
"""Implementer CPU regression tests; not independent review or native evidence."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
checks = 0


def require(ok, reason):
    global checks
    checks += 1
    if not ok:
        raise RuntimeError(reason)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m = load('discriminator', HERE / 'check_native.py')
ref = load('fixed_reference', HERE / 'reference.py')
for name, pin in [('consumer_nki.py', m.KERNEL_SHA), ('reference.py', m.REFERENCE_SHA),
                  ('original_check_native.py', '89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850')]:
    require(hashlib.sha256((HERE/name).read_bytes()).hexdigest() == pin, 'immutable ' + name)
def wrapper(path):
    return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == 'make_wrapper')
require(ast.dump(wrapper(HERE/'check_native.py')) == ast.dump(wrapper(HERE/'original_check_native.py')), 'exact custom-wrapper AST')
disabled = subprocess.run([sys.executable, '-B', str(HERE/'check_native.py')], text=True, capture_output=True, check=True)
require(json.loads(disabled.stdout) == dict(status='DISABLED', native_imported=False, native_executed=False), 'disabled default')
m.torch, m.F = torch, F
original_compile = torch.compile
injection = {'zero': False}
compile_calls = []


def compile_cpu(function, *, backend, fullgraph, dynamic):
    require((backend, fullgraph, dynamic) == ('neuron', True, False), 'unchanged compile options')
    compile_calls.append(function.__name__)
    if injection['zero'] and function.__name__ == 'enclosing':
        def deliberate_zero(weight, *args):
            output = function(weight, *args)
            return output.detach() + weight.sum() * 0
        return original_compile(deliberate_zero, backend='aot_eager', fullgraph=True, dynamic=False)
    return original_compile(function, backend='aot_eager', fullgraph=True, dynamic=False)


torch.compile = compile_cpu
ids, rows, metadata = ref.fixture('unique', 'fp32')
require(ids == list(range(256)) and metadata[3] == [1, 1], 'unique fixture')
upstream = torch.tensor(rows, dtype=torch.float32)
require(int(torch.count_nonzero(upstream)) == 32709, 'nonzero fixture')
require(m.tensor_record(upstream)['sha256'] == 'dee95c1da97debf6a901e00260bd60a74e7be1992eca64b832c4f6633cccc671', 'fixture hash')


def cpu_unique_kernel(gradient, counts, starts, positions, maximums):
    return gradient.float()


results = []
try:
    for zero in (False, True):
        injection['zero'] = zero
        torch.manual_seed(42)
        case = m.run_routes(cpu_unique_kernel, torch.device('cpu'), lambda: None, ids, rows, metadata, ref.dense)
        status, rc = m.classify(case)
        require([r['name'] for r in case['routes']] == list(m.ROUTES), 'all routes retained')
        require(all(r['inputs_unchanged'] for r in case['routes']), 'pre/post input identity')
        require(case['routes'][-1]['status'] == 'EXACT', 'ordinary control runs after injected failure')
        require(rc == int(zero), 'nonzero mismatch return')
        require('gradient' in case['routes'][2]['checks'], json.dumps(case['routes'][2]))
        bad = case['routes'][2]['checks']['gradient']
        require(bad['exact'] is not zero, 'compiled result classification')
        if zero:
            require(bad['actual']['nonzero'] == 0 and bad['max_abs'] == 8, 'deliberate zero negative control')
        results.append(dict(injected_zero_vjp=zero, status=status, returncode=rc, case=case))
    require(compile_calls == ['enclosing', 'ordinary', 'enclosing', 'ordinary'], 'custom before ordinary; no dtype/history replay')
finally:
    torch.compile = original_compile

# Error and mutation paths remain observable, including byte-only signed-zero change.
inputs = {key: torch.ones(1) for key in ('upstream', 'ids', 'weight', 'counts', 'starts', 'positions', 'maximums')}
def fail():
    raise RuntimeError('deliberate route exception')
err = m.observe_route('injected_exception', inputs, fail, {'gradient': torch.ones(1)}, lambda: None)
require(err['status'] == 'ERROR' and err['inputs_unchanged'] and 'deliberate route exception' in err['error'], 'exception receipt')
inputs['weight'].zero_()
def mutate():
    inputs['weight'].fill_(-0.0)
    return dict(gradient=torch.ones(1))
changed = m.observe_route('injected_mutation', inputs, mutate, {'gradient': torch.ones(1)}, lambda: None)
require(not changed['inputs_unchanged'] and changed['status'] == 'MISMATCH_OR_ERROR', 'signed zero mutation rejected')
require(not m.compare(None, torch.ones(1))['exact'], 'missing gradient rejected')
nonfinite = m.compare(torch.tensor([float('nan')]), torch.zeros(1))
require(not nonfinite['exact'] and nonfinite['max_abs'] is None, 'nonfinite rejected')
json.dumps(nonfinite, allow_nan=False)
print(json.dumps(dict(status='IMPLEMENTER_CPU_TESTS_PASS_NOT_INDEPENDENT_REVIEW', checks=checks,
    torch=torch.__version__, native_executed=False, backend_substitution='neuron -> aot_eager',
    kernel_substitution='unique-row identity CPU reference, not NKI',
    checker_sha256=hashlib.sha256((HERE/'check_native.py').read_bytes()).hexdigest(),
    wrapper_ast_identity=True, results=results, exception_negative=err, mutation_negative=changed,
    nonfinite_negative=nonfinite), indent=2, allow_nan=False))
