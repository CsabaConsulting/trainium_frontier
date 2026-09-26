"""Independent tiny CPU helper-routing test. Does not import candidate source."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(42)
root = Path(__file__).parent
diff = (root / 'SOURCE.diff').read_text()
added = '\n'.join(line[1:] for line in diff.splitlines() if line.startswith('+') and not line.startswith('+++'))
helper_text = added[:added.index('    return x, values') + len('    return x, values')]
ast.parse(helper_text)
calls = []
def metadata(idx, vocab):
    calls.append(('metadata', vocab))
    return (idx,)
def lookup(weight, idx):
    calls.append(('lookup', id(weight)))
    return F.embedding(idx, weight)
scope = {'torch': torch, '_K2_METADATA': metadata, '_K2_LOOKUP': lookup}
exec(compile(helper_text, 'SOURCE.diff:extracted-helper', 'exec'), scope)
helper = scope['_k2_eager_embeddings']

def model(attention, keys):
    def table():
        return SimpleNamespace(weight=torch.randn(7, 4, dtype=torch.float64, requires_grad=True), num_embeddings=7)
    return SimpleNamespace(transformer=SimpleNamespace(wte=table(), h=[SimpleNamespace(attn=object() if x else None) for x in attention]), value_embeds={str(k): table() for k in keys})

def tables(m):
    return {'main': m.transformer.wte.weight, **{k: v.weight for k, v in m.value_embeds.items()}}

cases = 0
microsteps = 0
tensor_checks = 0
for dtype in (torch.int32, torch.int64):
    for attention, keys, expected in [([True, False, True, True], [0, 1, 3], ['0', '3']), ([False, False], [0, 1], []), ([True, True], [], [])]:
        m = model(attention, keys)
        r = copy.deepcopy(m)
        for ids in ([[1, 1, 6], [0, 1, 6]], [[6, 0, 0], [6, 6, 1]]):
            idx = torch.tensor(ids, dtype=dtype)
            calls.clear()
            x, values = helper(m, idx)
            assert list(values) == expected
            assert calls == [('metadata', 7)] + [('lookup', id(tables(m)[k])) for k in ['main'] + expected]
            outputs = {'main': x, **values}
            # Explicit row stacking, independent of embedding/index-select operator.
            reference = {k: torch.stack([torch.stack([tables(r)[k][int(i)] for i in row]) for row in ids]) for k in ['main'] + expected}
            for k in outputs:
                assert torch.equal(outputs[k], reference[k]), k
                tensor_checks += 1
            coefficients = {k: torch.arange(v.numel(), dtype=v.dtype).reshape(v.shape) + j + 1 for j, (k, v) in enumerate(outputs.items())}
            sum((outputs[k] * coefficients[k]).sum() for k in outputs).backward()
            sum((reference[k] * coefficients[k]).sum() for k in reference).backward()
            for k, weight in tables(m).items():
                other = tables(r)[k]
                if k not in outputs:
                    assert weight.grad is None and other.grad is None
                else:
                    assert torch.equal(weight.grad, other.grad), k
                    tensor_checks += 1
            microsteps += 1
        cases += 1
graphs = json.loads((root / 'AOT_GRAPHS.json').read_text())
graph_summary = []
for graph in graphs:
    tree = ast.parse(graph['code'])
    operations = sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)})
    suspect = [op for op in operations if any(token in op.lower() for token in ('k2', 'nki', 'embedding', 'index_select', 'scatter', 'argsort', 'sort.'))]
    assert not suspect, suspect
    graph_summary.append({'kind': graph['kind'], 'declared_nodes': graph['nodes'], 'unique_call_targets': len(operations), 'lookup_related_targets': suspect})
print(json.dumps({'status': 'PASS', 'torch_version': torch.__version__, 'cases': cases, 'microsteps': microsteps, 'exact_tensor_checks': tensor_checks, 'aot_graphs_inspected': graph_summary, 'scope': 'helper routing and CPU surrogate gradients; not NKI math/native compatibility'}))
