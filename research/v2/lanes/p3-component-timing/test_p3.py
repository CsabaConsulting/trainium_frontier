"""Small CPU boundary checks and real eight-layer default-OFF identity."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)


def exact(a, b):
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
        return
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                       b.detach().contiguous().reshape(-1).view(torch.uint8)), 'raw bit mismatch'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checks(base):
    sys.path.insert(0, str(base.parents[3]/'workflow/tools/stub_prepare'))
    arrow, parquet = types.ModuleType('pyarrow'), types.ModuleType('pyarrow.parquet')
    arrow.parquet = parquet
    sys.modules['pyarrow'], sys.modules['pyarrow.parquet'] = arrow, parquet
    a, b = load('p3_base', base), load('p3_candidate', ROOT/'train.py')
    # After removing only additions, the exact original module AST remains.
    original = ast.parse(base.read_text())
    tree = ast.parse((ROOT/'train.py').read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or not n.name.startswith('_p3_')]
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    def added(n):
        if isinstance(n, ast.If) and ast.unparse(n.test) == 'args.p3_component_timing':
            return True
        return (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and ast.unparse(n.value.func) == 'parser.add_argument'
                and n.value.args and isinstance(n.value.args[0], ast.Constant)
                and str(n.value.args[0].value).startswith('--p3-'))
    main.body = [n for n in main.body if not added(n)]
    assert ast.dump(tree) == ast.dump(original), 'non-diagnostic original code changed'
    torch.manual_seed(42)
    q, k, v = [(torch.randn(2, 8, 2, 4).bfloat16().requires_grad_()).transpose(1, 2)
               for _ in range(3)]
    dy = torch.randn(2, 8, 2, 4).bfloat16().transpose(1, 2)
    sdpa = b._p3_sdpa(q, k, v)
    plain = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=None)
    exact(sdpa, plain)
    exact(torch.autograd.grad(sdpa, (q, k, v), dy, retain_graph=True),
          torch.autograd.grad(plain, (q, k, v), dy, retain_graph=True))
    # Repeat VJP on the same saved graph, as the measured backward blocks do.
    exact(torch.autograd.grad(sdpa, (q, k, v), dy, retain_graph=True),
          torch.autograd.grad(plain, (q, k, v), dy, retain_graph=True))
    x = (torch.randn(16, 8)*.1).bfloat16().requires_grad_()
    up = (torch.randn(16, 8)*.1).bfloat16().requires_grad_()
    down = (torch.randn(8, 16)*.1).bfloat16().requires_grad_()
    grad = torch.randn(16, 8).bfloat16()
    y = b._p3_mlp(x, up, down)
    y_base = a._FusedMLP.apply(x, up, down)
    y_plain = F.linear(F.relu(F.linear(x, up)).square(), down)
    exact(y, y_plain)
    actual_grad = torch.autograd.grad(y, (x, up, down), grad, retain_graph=True)
    exact(actual_grad, torch.autograd.grad(y_base, (x, up, down), grad, retain_graph=True))
    pre = F.linear(x.detach(), up.detach())
    act, relu_t = b._p3_activation_forward(pre)
    exact((act, relu_t), (torch.relu(pre).square(), torch.relu(pre).T.contiguous()))
    dhidden = down.detach().T @ grad.T
    act_t, dpre_t = b._p3_activation_backward(relu_t, dhidden)
    exact((act_t, dpre_t), (relu_t.square(), dhidden * (2.0*relu_t)))
    exact(b._p3_matrix_forward(x.detach(), up.detach(), down.detach(), act), (pre, y.detach()))
    matrix_grad = b._p3_matrix_backward(x.detach(), up.detach(), down.detach(), grad, dpre_t, act_t)
    exact(matrix_grad[0], dhidden)
    exact(matrix_grad[1:], actual_grad)
    assert relu_t.is_contiguous() and act_t.is_contiguous() and dpre_t.is_contiguous()
    models = []
    for module in (a, b):
        torch.manual_seed(42)
        model = module.GPT(module.GPTConfig(sequence_len=8, vocab_size=64, n_layer=8,
                                           n_head=2, n_embd=64, mlp_ratio=2))
        model.init_weights()
        models.append(model)
    opts = [torch.optim.SGD(m.parameters(), lr=.01) for m in models]
    losses = []
    for step in range(3):
        torch.manual_seed(100+step)
        ids, targets = torch.randint(64, (2, 8)), torch.randint(64, (2, 8))
        ls = []
        for model, opt in zip(models, opts):
            opt.zero_grad(set_to_none=True)
            loss = model(ids, targets)
            loss.backward()
            ls.append(loss)
        exact(*ls)
        for pa, pb in zip(models[0].parameters(), models[1].parameters()):
            if pa.grad is None or pb.grad is None:
                assert pa.grad is pb.grad
            else:
                exact(pa.grad, pb.grad)
        for opt in opts:
            opt.step()
        for key, value in models[0].state_dict().items():
            exact(value, models[1].state_dict()[key])
        losses.append(float(ls[0].detach()))
    return {'status': 'PASS_IMPLEMENTER_CPU_ONLY', 'original_ast_unchanged': True,
            'sdpa_forward_backward': 'EXACT_PLAIN_TORCH',
            'mlp_forward': 'EXACT_PLAIN_TORCH',
            'mlp_backward': 'EXACT_BASE_AND_LITERAL_BOUNDARY_DECOMPOSITION',
            'activation_and_matrix_boundaries': 'BIT_EXACT',
            'off_three_steps': {'status': 'BIT_IDENTICAL', 'losses': losses,
                                'shape': '8layers/width64/context8/batch2', 'optimizer': 'SGD .01'},
            'imports': ['existing test-only prepare', 'unused pyarrow.parquet placeholder'],
            'native_executed': False, 'torch': torch.__version__,
            'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (base, ROOT/'train.py')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    result = checks(args.base)
    (ROOT/'cpu-result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
