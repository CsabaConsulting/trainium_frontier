import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
ROOT = Path('/tmp/trainium-k2-fp32-independent-7QD8G2')
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

checks = []
def require(ok, name):
    checks.append(dict(name=name, passed=bool(ok)))
    if not ok:
        raise RuntimeError(name)

checker = load('check_native')
ref = load('reference')
checker.torch, checker.F = torch, F
torch.set_num_threads(1)
torch.manual_seed(42)
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file()}
require(hashes['consumer_nki.py'] == checker.KERNEL_SHA, 'kernel pin')
require(hashes['reference.py'] == checker.REFERENCE_SHA, 'reference pin')
require(hashes['original_check_native.py'] == '89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850', 'original checker pin')
def wrapper(path):
    return ast.dump(next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == 'make_wrapper'), include_attributes=False)
require(wrapper(ROOT / 'check_native.py') == wrapper(ROOT / 'original_check_native.py'), 'wrapper AST identity')

# Independently specified three-row permutation, checked against handwritten mapping.
tiny_ids = torch.tensor([2, 0, 1], dtype=torch.int64)
tiny_upstream = torch.tensor([[1., -2.], [3., 4.], [-5., 6.]])
tiny_weight = torch.zeros((3, 2), requires_grad=True)
F.embedding(tiny_ids, tiny_weight).backward(tiny_upstream)
tiny_expected = torch.tensor([[3., 4.], [-5., 6.], [1., -2.]])
require(checker.compare(tiny_weight.grad, tiny_expected)['exact'], 'own tiny handwritten gradient oracle')
ids, rows, metadata = ref.fixture('unique', 'fp32')
require(ids == list(range(256)), 'unique fixture identity ids')
require(checker.compare(torch.tensor(ref.dense(ids, rows)), torch.tensor(rows))['exact'], 'dense unique oracle independently equals upstream rows')

real_compile = torch.compile
cases = []
for inject_zero in (False, True):
    calls = []
    def cpu_compile(function, **kwargs):
        require(kwargs == dict(backend='neuron', fullgraph=True, dynamic=False), 'compile configuration preserved')
        calls.append(function.__name__)
        compiled = real_compile(function, backend='aot_eager', fullgraph=True, dynamic=False)
        if inject_zero and function.__name__ == 'enclosing':
            def corrupt(*args):
                output = compiled(*args)
                # Explicit test-only corruption at weight-gradient delivery.
                args[0].register_hook(lambda gradient: torch.zeros_like(gradient))
                return output
            return corrupt
        return compiled
    torch.compile = cpu_compile
    case = checker.run_routes(lambda gradient, *meta: gradient.clone(), 'cpu', lambda: None, ids, rows, metadata, ref.dense)
    status, rc = checker.classify(case)
    require(calls == ['enclosing', 'ordinary'], 'separate compiled functions and call order')
    require([r['name'] for r in case['routes']] == list(checker.ROUTES), 'all four routes retained')
    require(rc == int(inject_zero), 'failure maps to nonzero classification')
    require(case['routes'][-1]['status'] == 'EXACT', 'ordinary control executed and exact after custom result')
    if inject_zero:
        require(case['routes'][2]['status'] == 'MISMATCH', 'zero custom gradient detected')
        require(case['routes'][2]['checks']['gradient']['actual']['nonzero'] == 0, 'negative control actually zero')
    else:
        require(all(r['status'] == 'EXACT' for r in case['routes']), 'all routes exact CPU positive control')
    cases.append(dict(inject_zero=inject_zero, compile_calls=calls, status=status, returncode=rc, case=case))
torch.compile = real_compile

expected = torch.tensor([1., 2.])
comparisons = {}
for name, actual in [('missing', None), ('shape', expected.reshape(1,2)), ('dtype', expected.double()), ('nan', torch.tensor([float('nan'), 2.])), ('inf', torch.tensor([float('inf'), 2.])), ('bytes', torch.tensor([1., 3.]))]:
    comparisons[name] = checker.compare(actual, expected)
    require(not comparisons[name]['exact'], 'strict rejection ' + name)
require(not checker.compare(torch.tensor([-0.]), torch.tensor([0.]))['exact'], 'signed zero byte distinction')
inputs = {key: torch.ones(2) for key in ('upstream', 'ids', 'weight', 'counts', 'starts', 'positions', 'maximums')}
def mutate():
    inputs['positions'][0] = 2.
    return dict(gradient=expected.clone())
mutation = checker.observe_route('mutation', inputs, mutate, dict(gradient=expected), lambda: None)
require(not mutation['inputs_unchanged'] and mutation['status'] == 'MISMATCH_OR_ERROR', 'metadata mutation rejected with exact output')
def broken():
    raise RuntimeError('INDEPENDENT_INJECTED_ERROR')
error = checker.observe_route('exception', inputs, broken, dict(gradient=expected), lambda: None)
require(error['status'] == 'ERROR' and 'INDEPENDENT_INJECTED_ERROR' in error['error'] and len(error['inputs_after']) == 7, 'exception retained with post observations')
for record in (mutation, error):
    bad_case = dict(cases[0]['case'])
    bad_case['routes'] = [dict(r) for r in bad_case['routes']]
    record = dict(record, name=checker.ROUTES[0])
    bad_case['routes'][0] = record
    require(checker.classify(bad_case)[1] == 1, 'mutation or exception nonzero classification')
print(json.dumps(dict(status='PASS_BOUNDED_CPU_SOURCE_REVIEW_TESTS', optimized=not __debug__, python=sys.executable, torch=torch.__version__, native_executed=False, substitutions='neuron -> aot_eager; NKI -> unique identity clone; negative custom weight-gradient hook', hashes=hashes, checks=checks, cases=cases, comparisons=comparisons, mutation=mutation, error=error), indent=2, allow_nan=False))
