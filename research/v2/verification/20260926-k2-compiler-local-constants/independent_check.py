"""Fresh CPU-only verifier; reads only the five allowed packet inputs."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parent
EXPECTED = '662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00'
def sha(data):
    return hashlib.sha256(data).hexdigest()

manifest_bytes = (ROOT / 'MANIFEST.json').read_bytes()
assert sha(manifest_bytes) == '06f77b50cd45fb8f31716f1dd118353c4520f99a3beb32fa610d4500df530fd3'
manifest = json.loads(manifest_bytes)
assert manifest['source_sha256'] == EXPECTED
for name, record in manifest['files'].items():
    data = (ROOT / name).read_bytes()
    assert len(data) == record['bytes'] and sha(data) == record['sha256'], name
print('PASS: all five input identities match frozen manifest')

lines = (ROOT / 'candidate.diff').read_text().splitlines(keepends=True)
assert sum(line.startswith('diff --git ') for line in lines) == 1
assert sum(line.startswith('@@ ') for line in lines) == 1
start = next(i for i, line in enumerate(lines) if line.startswith('@@ ')) + 1
hunk = lines[start:]
assert all(line[0] in ' +-' for line in hunk)
old = ''.join(line[1:] for line in hunk if line[0] in ' -')
new = ''.join(line[1:] for line in hunk if line[0] in ' +')
assert len(old.splitlines()) == 272 and len(new.splitlines()) == 271
removed = [line[1:].strip() for line in hunk if line.startswith('-')]
added = [line[1:].strip() for line in hunk if line.startswith('+')]
assert [line for line in removed if line] == added
assert len(added) == 3 and len(removed) == 4

def factory(source):
    fragment = source[source.index('def _k2_build_kernel('):source.index('def _k2_prepare_metadata(')]
    return ast.parse(fragment).body[0]
before, after = factory(old), factory(new)
kernel_before = next(n for n in before.body if isinstance(n, ast.FunctionDef))
kernel_after = next(n for n in after.body if isinstance(n, ast.FunctionDef))
constants_before = [n for n in before.body if isinstance(n, ast.Assign)]
constants_after = kernel_after.body[:2]
dump = lambda n: ast.dump(n, include_attributes=False)
assert len(constants_before) == 2
assert [dump(n) for n in constants_before] == [dump(n) for n in constants_after]
assert [dump(n) for n in kernel_before.body] == [dump(n) for n in kernel_after.body[2:]]
normalized = copy.deepcopy(after)
nk = next(n for n in normalized.body if isinstance(n, ast.FunctionDef))
index = normalized.body.index(nk)
normalized.body[index:index] = nk.body[:2]
nk.body = nk.body[2:]
assert dump(normalized) == dump(before)
# Entire supplied source context, including unrelated functions, differs only
# by that textual move. The full file is not supplied and is not fabricated.
assert old.replace('    # Fixed production geometry. No flag, shape fallback, or CPU-ID route.\n    vocab, tokens, channels = 8192, 32768, 1024\n    partition, features, groups = 128, 512, 64\n\n', '') == new.replace('        # Fixed production geometry. No flag, shape fallback, or CPU-ID route.\n        vocab, tokens, channels = 8192, 32768, 1024\n        partition, features, groups = 128, 512, 64\n', '')
print('PASS: exact constants-only relocation; all original kernel operations and factory AST preserved')

names = ('vocab', 'tokens', 'channels', 'partition', 'features', 'groups')
def code_for(node):
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]))
    outer = next(c for c in compile(module, '<packet-factory>', 'exec').co_consts if isinstance(c, types.CodeType))
    inner = next(c for c in outer.co_consts if isinstance(c, types.CodeType) and c.co_name == 'resident_consumer')
    return inner
assert set(code_for(before).co_freevars) == set(names)
assert code_for(after).co_freevars == ()

# Execute an independent scope probe of the exact assignment nodes at their
# original scope. Only native operations/imports/decorator are replaced by a
# tuple return; this is a CPU name-binding proof, not kernel execution.
def execute_geometry(node):
    node = copy.deepcopy(node)
    inner = next(n for n in node.body if isinstance(n, ast.FunctionDef))
    inner.decorator_list = []
    inner.args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    inner.body = [n for n in inner.body[:2] if isinstance(n, ast.Assign)]
    inner.body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load()) for n in names], ctx=ast.Load())))
    node.body = [n for n in node.body if isinstance(n, (ast.Assign, ast.FunctionDef))]
    node.body.append(ast.Return(value=ast.Call(func=ast.Name(id=inner.name, ctx=ast.Load()), args=[], keywords=[])))
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {}
    exec(compile(module, '<independent-scope-probe>', 'exec'), namespace)
    return namespace[node.name]()
old_values, new_values = execute_geometry(before), execute_geometry(after)
assert old_values == new_values == (8192, 32768, 1024, 128, 512, 64)
vocab, tokens, channels, partition, features, groups = new_values
rows = [g * partition + lane for g in range(groups) for lane in range(partition)]
cols = [program * features + lane for program in range(2) for lane in range(features)]
assert rows == list(range(vocab)) and cols == list(range(channels))
assert tokens * channels == 33554432
print('PASS: executed before/after scope probes resolve identical six values', new_values)
print('PASS: compiled Python freevars change from exactly six geometry names to none')
print('PASS: independent production geometry enumeration covers all 8192 rows and 1024 columns exactly once')

cpu = json.loads((ROOT / 'cpu-result.json').read_text())
assert EXPECTED in cpu['hashes'].values() and cpu['native_compiled'] is False
assert set(cpu['by_index_dtype']) == {'torch.int32', 'torch.int64'}
for dtype, record in cpu['by_index_dtype'].items():
    finite = [r for r in record['operation'] if r['verdict'] == 'EXACT_FINITE']
    diagnostics = [r for r in record['operation'] if r['verdict'] == 'FAIL_NONFINITE_DIAGNOSTIC']
    assert len(finite) == 12 and len(diagnostics) == 4 and len(record['operation']) == 16
    for r in finite:
        assert r['actual_bits'] == r['expected_bits']
        assert all((v & 0x7f800000) != 0x7f800000 for row in r['actual_bits'] for v in row)
    m = record['model']
    assert m['status'] == 'OFF_3_STEPS_BIT_IDENTICAL'
    assert m['on_status'] == 'ON_3_STEPS_BIT_IDENTICAL_CPU_INTERPRETER'
    assert len(m['losses']) == len(m['on_losses']) == m['metadata_calls_for_three_on_steps'] == 3
    assert m['eval_ordinary'] is True
    print('PASS: supporting receipt consistency', dtype, '12 finite bitwise fixtures; 4 excluded diagnostics; OFF3/ON3')
assert len(cpu['input_contract']['rejected']) == 21
parser = json.loads((ROOT / 'parser-result.json').read_text())
assert parser['source_sha256'] == EXPECTED and parser['returncode'] == 0 and parser['stderr'] == ''
p = json.loads(parser['stdout'])
assert p['source_sha256'] == EXPECTED and p['status'] == 'PARSER_MLIR_PASS'
assert p['closure'] == {} and p['mlir_verified'] and p['exact_unmodified_factory_ast']
assert p['backend_executed'] is False and p['no_device_execution'] is True
assert p['target'] == 'trn2' and p['lnc'] == 2
assert p['nki_version'] == '0.5.0+28631259367.ga768afa6'
assert p['production_geometry'] == {
    'gradient': {'shape': [tokens, channels], 'dtype': 'bfloat16'},
    'counts': {'shape': [vocab, 1], 'dtype': 'int32'},
    'starts': {'shape': [vocab, 1], 'dtype': 'int32'},
    'positions': {'shape': [tokens, 1], 'dtype': 'int32'},
    'group_maximums': {'shape': [groups, 1], 'dtype': 'int32'},
}
print('PASS: supporting production parser receipt identity and scope consistency')
print('LIMIT: full-file candidate identity is attested by supplied receipts; full file was not an allowed input')
print('LIMIT: no independent numerical kernel execution or parser rerun; no backend/native/cold/timing proof')
print('VERDICT: PASS_FOR_NEW_SOURCE_NATIVE_CHECK')
