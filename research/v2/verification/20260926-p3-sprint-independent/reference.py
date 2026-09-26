"""Independent bounded CPU checks; only source bundle input, no author tests/imports."""
import ast
import hashlib
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(42)
ROOT = Path('/home/poojith-devan/p3-sprint-review-ulRZiF')
OUT = Path(__file__).resolve().parent
candidate = ast.parse((ROOT / 'train.py').read_text())
base = ast.parse((ROOT / 'base.py').read_text())
names = {'reference_forward', '_fused_mlp_kernel', '_FusedMLP'}
names.update(n.name for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name.startswith('_p3_') and n.name != '_p3_component_run')
namespace = {'torch': torch, 'F': F}
exec(compile(ast.Module(body=[n for n in candidate.body if getattr(n, 'name', None) in names], type_ignores=[]), '<candidate-exact-ast>', 'exec'), namespace)

# Structural disabled-path proof: explicitly remove precisely the additions.
rest = ast.parse((ROOT / 'train.py').read_text())
rest.body = [n for n in rest.body if not (isinstance(n, ast.FunctionDef) and n.name.startswith('_p3_'))]
main = next(n for n in rest.body if getattr(n, 'name', None) == 'main')
removed = []
for node in list(main.body):
    option = isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'parser.add_argument' and node.value.args and isinstance(node.value.args[0], ast.Constant) and str(node.value.args[0].value).startswith('--p3-')
    dispatch = isinstance(node, ast.If) and ast.unparse(node.test) == 'args.p3_component_timing'
    if option or dispatch:
        removed.append(ast.unparse(node))
        main.body.remove(node)
assert len(removed) == 5
assert ast.dump(rest) == ast.dump(base)

def exact(x, y):
    assert x.shape == y.shape and x.dtype == y.dtype
    assert torch.equal(x.detach().contiguous().view(torch.uint8), y.detach().contiguous().view(torch.uint8))

# An independent scalar loop matrix reference, exact integer arithmetic in this
# deliberately small BF16 domain. Does not use the candidate's matmul formula.
def mm(a, b):
    a, b = a.tolist(), b.tolist()
    return torch.tensor([[sum(float(a[i][k]) * float(b[k][j]) for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))], dtype=torch.bfloat16)

x = torch.tensor([[1., -1., 2.], [-2., 1., 0.]], dtype=torch.bfloat16, requires_grad=True)
up = torch.tensor([[1., 0., -1.], [0., 1., 1.], [-1., -1., 0.], [1., 1., 1.]], dtype=torch.bfloat16, requires_grad=True)
down = torch.tensor([[1., -1., 0., 1.], [0., 1., -1., 0.], [-1., 0., 1., 1.]], dtype=torch.bfloat16, requires_grad=True)
dy = torch.tensor([[1., -1., 1.], [-1., 1., 0.]], dtype=torch.bfloat16)
pre = mm(x, up.T)
relu = torch.tensor([[max(0., v) for v in row] for row in pre.tolist()], dtype=torch.bfloat16)
activated = torch.tensor([[v * v for v in row] for row in relu.tolist()], dtype=torch.bfloat16)
yref = mm(activated, down.T)
dh = mm(dy, down)
dp = torch.tensor([[dh[i,j].item() * 2. * relu[i,j].item() for j in range(4)] for i in range(2)], dtype=torch.bfloat16)
dxref, duref, ddref = mm(dp, up), mm(dp.T, x), mm(dy.T, activated)
y = namespace['_p3_mlp'](x, up, down)
exact(y, yref)
for repetition in range(5):
    got = torch.autograd.grad(y, (x, up, down), dy, retain_graph=True)
    for value, reference in zip(got, (dxref, duref, ddref)):
        exact(value, reference)
af = namespace['_p3_activation_forward'](pre)
exact(af[0], activated)
exact(af[1], relu.T.contiguous())
ab = namespace['_p3_activation_backward'](af[1], dh.T.contiguous())
exact(ab[0], activated.T.contiguous())
exact(ab[1], dp.T.contiguous())
mf = namespace['_p3_matrix_forward'](x, up, down, activated)
exact(mf[0], pre)
exact(mf[1], yref)
mb = namespace['_p3_matrix_backward'](x, up, down, dy, ab[1], ab[0])
for value, reference in zip(mb, (dh.T, dxref, duref, ddref)):
    exact(value, reference)

# SDPA has an independent dense masked-softmax reference, including gradients.
q, k, v = [torch.randn(2, 5, 2, 4).transpose(1, 2).requires_grad_() for _ in range(3)]
grad = torch.randn(2, 5, 2, 4).transpose(1, 2)
sdpa = namespace['_p3_sdpa'](q, k, v)
scores = (q @ k.transpose(-1, -2)) / 2.
mask = torch.arange(5)[:, None] < torch.arange(5)[None, :]
reference = torch.softmax(scores.masked_fill(mask, float('-inf')), dim=-1) @ v
torch.testing.assert_close(sdpa, reference, rtol=2e-6, atol=2e-6)
ga = torch.autograd.grad(sdpa, (q, k, v), grad, retain_graph=True)
gb = torch.autograd.grad(reference, (q, k, v), grad)
for a, b in zip(ga, gb):
    torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-6)
sdpa_error = (sdpa-reference).abs().max().item()

# Execute the exact nested timing controller with identity compile and a
# deterministic block wrapper. This tests closure rebinding and saved graph
# replay; it deliberately claims nothing about the Neuron compiler.
runner = next(n for n in candidate.body if getattr(n, 'name', None) == '_p3_component_run')
timing = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == 'timing')
from types import SimpleNamespace
events, records, saved = [], [], []
def block(call, repetitions=1):
    for _ in range(repetitions):
        output = call()
    events.append(repetitions)
    return output, .001 * repetitions
def finite(output):
    for t in output if isinstance(output, (tuple, list)) else (output,):
        assert torch.isfinite(t).all()
proxy = SimpleNamespace(compile=lambda fn, **kwargs: fn, autograd=torch.autograd, ones_like=torch.ones_like)
env = dict(namespace, torch=proxy, block=block, finite=finite, args=SimpleNamespace(p3_blocks=4, p3_repeats=3), statistics=statistics, records=records, save=saved.append, rank=0, json=json)
exec(compile(ast.Module(body=[timing], type_ignores=[]), '<exact-timing-controller>', 'exec'), env)
env['timing']('independent_small_mlp', namespace['_p3_mlp'], (x, up, down), backward=True, upstream=dy)
assert events == [1, 1] + [3, 3] * 6
assert len(records[0]['forward_ms_per_call']) == len(records[0]['backward_ms_per_call']) == 4
assert saved == ['IN_PROGRESS']

result = {'verdict': 'CPU_REFERENCE_PASS', 'source_sha256': hashlib.sha256((ROOT/'train.py').read_bytes()).hexdigest(), 'base_sha256': hashlib.sha256((ROOT/'base.py').read_bytes()).hexdigest(), 'off_remaining_ast_identical': True, 'independent_scalar_loop_bf16_mlp_and_all_splits': 'BIT_EXACT', 'repeated_saved_graph_backward': 'FIVE_BIT_EXACT_REPETITIONS', 'manual_causal_sdpa_forward_and_gradients': 'PASS_TOLERANCE_3e-6', 'sdpa_forward_max_abs_error': sdpa_error, 'timing_controller_small_cpu': 'PASS_2_WARMUP_4_TIMED_3_REPETITIONS', 'native_executed': False, 'torch': torch.__version__}
(OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
