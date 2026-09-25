"""Independent bounded CPU test of initial cache refresh, written for this review.

Extract definitions and actual loop snippets from both hash-verified source texts.
Never import either full training program. A direct F.linear oracle tests Linear;
the source's cache-disabled custom-MLP path controls its existing backward math.
No native backend, compiler, full GPT, data, evaluator, or gate harness is used.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
WORK = Path('/work')
EXPECTED = {
    'original': '93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5',
    'candidate': '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c',
}
CHECKS = 0
COMPARISONS = 0
RAW = {}

def require(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)

def raw_bytes(t):
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

def exact(a, b, label):
    global COMPARISONS
    COMPARISONS += 1
    require(a.shape == b.shape and a.dtype == b.dtype, label + ': metadata')
    require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), label + ': finite')
    require(raw_bytes(a) == raw_bytes(b), label + ': bitwise equality')

def code(nodes, filename):
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), filename, 'exec')

def load_source(name):
    path = WORK / (name + '_train.py')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == EXPECTED[name], name + ' source hash')
    tree = ast.parse(data, filename=str(path))
    names = {'Linear', 'reference_forward', 'make_autograd', 'make_forward'}
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    require(len(definitions) == len(names), name + ' extracted definitions')
    ns = {'torch': torch, 'nn': nn, 'F': F, 'USE_WEIGHT_CACHE': True}
    exec(code(definitions, str(path)), ns)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in main.body if isinstance(n, ast.While) and ast.unparse(n.test) == 'step < args.num_steps')
    initial = [n for n in loop.body if isinstance(n, ast.If) and 'args.initial_cache_refresh' in ast.unparse(n.test)]
    collection = [n for n in main.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'cached_linears']
    refresh = [n for n in loop.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'cached_linears']
    parser_node = [n for n in main.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                   and n.value.args and isinstance(n.value.args[0], ast.Constant)
                   and n.value.args[0].value == '--initial-cache-refresh']
    require(len(collection) == len(refresh) == 1, name + ' loop extraction')
    require(len(initial) == len(parser_node) == (name == 'candidate'), name + ' additions located')
    if name == 'candidate':
        parser_ns = {'parser': argparse.ArgumentParser(), 'argparse': argparse}
        exec(code(parser_node, str(path)), parser_ns)
        parser = parser_ns['parser']
        require(parser.parse_args([]).initial_cache_refresh is False, 'default is OFF')
        require(parser.parse_args(['--no-initial-cache-refresh']).initial_cache_refresh is False, 'explicit OFF')
        require(parser.parse_args(['--initial-cache-refresh']).initial_cache_refresh is True, 'explicit ON')
        ns['parser'] = parser
    ns['initial_code'] = code(initial, str(path))
    ns['collection_code'] = code(collection, str(path))
    ns['refresh_code'] = code(refresh, str(path))
    return ns

class TinyMLP(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.c_fc = linear(32, 64, bias=False)
        self.c_proj = linear(64, 32, bias=False)

    def forward(self, x, fc=None):
        h = self.c_fc(x) if fc is None else fc
        return self.c_proj(F.relu(h).square())

def snapshots(model, optimizer, x, y, loss):
    out = {'output': y.detach().clone(), 'loss': loss.detach().clone(), 'input_gradient': x.grad.detach().clone()}
    for name, p in model.named_parameters():
        require(p.grad is not None, name + ' has gradient')
        out['gradient/' + name] = p.grad.detach().clone()
        out['parameter/' + name] = p.detach().clone()
        for k, v in optimizer.state[p].items():
            if isinstance(v, torch.Tensor):
                out['optimizer/' + name + '/' + k] = v.detach().clone()
    return out

def run_arm(case, arm):
    torch.manual_seed(42)
    source = 'original' if arm == 'original' else 'candidate'
    ns = load_source(source)
    ns['USE_WEIGHT_CACHE'] = arm != 'uncached'
    linear = ns['Linear']
    model = TinyMLP(linear) if case['mlp'] else linear(16, 12, bias=False)
    if case.get('transpose'):
        model.weight = nn.Parameter(model.weight.detach().T.contiguous())
        model.weight_transposed = True
    if case['mlp']:
        forward = ns['make_forward']('cpu', TinyMLP.forward, ns['USE_WEIGHT_CACHE'], forward_backend='native')
        model.forward = MethodType(forward, model)
    model.train()
    modules = [m for m in model.modules() if isinstance(m, linear)]
    ns.update(model=model)
    exec(ns['collection_code'], ns)
    expected_cache_count = 0 if arm == 'uncached' else len(modules)
    require(len(ns['cached_linears']) == expected_cache_count, 'actual cache collection')
    call_counts = [0] * len(modules)
    for index, module in enumerate(modules):
        original_refresh = module.refresh_weight_cache
        def traced(self, dtype, index=index, original_refresh=original_refresh):
            call_counts[index] += 1
            return original_refresh(dtype)
        module.refresh_weight_cache = MethodType(traced, module)
    flag = arm == 'on'
    if source == 'candidate':
        argv = [] if arm == 'default' else ['--initial-cache-refresh' if flag else '--no-initial-cache-refresh']
        flag = ns['parser'].parse_args(argv).initial_cache_refresh
    ns.update(args=SimpleNamespace(initial_cache_refresh=flag), COMPUTE_DTYPE=case['cache_dtype'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=0.02, foreach=False, fused=False)
    identities = tuple(id(p) for p in model.parameters())
    keys = tuple(model.state_dict())
    width = 32 if case['mlp'] else 16
    out_width = 32 if case['mlp'] else 12
    records, cache_trace = [], []
    # Two independently backpropagated microbatches exercise reuse of differentiable casts.
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        ns['step'] = step
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        before_rng = torch.get_rng_state().clone()
        exec(ns['initial_code'], ns)
        exact(torch.get_rng_state(), before_rng, 'refresh RNG invariance')
        for name, p in model.named_parameters():
            exact(p, before[name], 'refresh master invariance/' + name)
        require(tuple(model.state_dict()) == keys, 'cache is not checkpoint state')
        require(tuple(id(p) for p in model.parameters()) == identities, 'parameter identities preserved')
        primed = arm == 'on' or (step > 0 and arm != 'uncached')
        for m in modules:
            require((m._cached_weight is not None) == primed, 'expected cache state before forward')
            if primed:
                exact(m._cached_weight, m.weight.to(case['cache_dtype']), 'fresh cache values')
                require(m._cached_weight.requires_grad, 'cache remains differentiable')
                if case['cache_dtype'] != m.weight.dtype:
                    require(m._cached_weight.grad_fn is not None, 'cast retains autograd edge')
        cache_trace.append([m._cached_weight is not None for m in modules])
        for micro in range(2):
            x = torch.randn(2, case['context'], width).to(case['input_dtype']).requires_grad_()
            target = torch.randn(2, case['context'], out_width)
            if arm == 'oracle':
                w = model.weight.to(x.dtype)
                y = x @ w if model.weight_transposed else F.linear(x, w)
            else:
                y = model(x)
            if case.get('custom'):
                require(type(y.grad_fn.next_functions[0][0]).__name__ == 'FusedMLPBackward', 'custom backward path reached')
            loss = (y.float() - target).square().mean() / 2
            loss.backward()
            records.append(snapshots(model, optimizer, x, y, loss))
        optimizer.step()
        records.append(snapshots(model, optimizer, x, y, loss))
        # Match the unchanged source: refresh after optimizer step, with grad enabled.
        exec(ns['refresh_code'], ns)
        for m in ns['cached_linears']:
            exact(m._cached_weight, m.weight.to(case['cache_dtype']), 'post-update cache')
    expected_calls = 0 if arm == 'uncached' else 3 + (arm == 'on')
    require(call_counts == [expected_calls] * len(modules), 'refresh counts')
    require(any(not torch.equal(p, before[name]) for name, p in model.named_parameters()), 'nontrivial parameter update')
    return records, {'cache_before_forward': cache_trace, 'refresh_calls_per_module': call_counts}

def main():
    cases = [
        dict(name='linear_bf16', mlp=False, context=7, input_dtype=torch.bfloat16, cache_dtype=torch.bfloat16),
        dict(name='linear_fp32_alias', mlp=False, context=7, input_dtype=torch.float32, cache_dtype=torch.float32),
        dict(name='linear_transposed_bf16', mlp=False, transpose=True, context=7, input_dtype=torch.bfloat16, cache_dtype=torch.bfloat16),
        dict(name='linear_dtype_fallback', mlp=False, context=7, input_dtype=torch.bfloat16, cache_dtype=torch.float32),
        dict(name='mlp_fallback_bf16', mlp=True, context=7, input_dtype=torch.bfloat16, cache_dtype=torch.bfloat16),
        dict(name='mlp_custom_bf16', mlp=True, custom=True, context=128, input_dtype=torch.bfloat16, cache_dtype=torch.bfloat16),
    ]
    summaries = []
    for case in cases:
        arms = ['original', 'default', 'off', 'on', 'uncached']
        if not case['mlp']:
            arms.append('oracle')
        collected = {}
        traces = {}
        for arm in arms:
            collected[arm], traces[arm] = run_arm(case, arm)
        baseline = collected['original']
        for arm in arms[1:]:
            require(len(baseline) == len(collected[arm]), 'record count')
            for index, (a, b) in enumerate(zip(baseline, collected[arm])):
                require(a.keys() == b.keys(), 'snapshot keys')
                for key in a:
                    exact(a[key], b[key], f'{case["name"]}/{arm}/{index}/{key}')
        RAW[case['name']] = collected
        summary = {'case': case['name'], 'status': 'PASS', 'arms': arms, 'steps': 3,
                   'microbatches_per_step': 2, 'microbatch': 2, 'context': case['context'],
                   'width': 32 if case['mlp'] else 16, 'max_hidden_width': 64 if case['mlp'] else 16,
                   'comparison': 'exact bytes, zero tolerance', 'traces': traces,
                   'step_losses_original': [float(baseline[i]['loss']) for i in (1, 4, 7)]}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    torch.save(RAW, WORK / 'reference_exactness.tensors.pt')
    results = {'status': 'PASS', 'python': platform.python_version(), 'torch': str(torch.__version__),
               'seed': 42, 'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
               'device': 'cpu', 'source_hashes': EXPECTED, 'checks': CHECKS,
               'tensor_bitwise_comparisons': COMPARISONS, 'cases': summaries,
               'optimizer_scope': 'tiny AdamW comparator only; supplied G2 covers the unchanged production optimizer',
               'no_full_model': True, 'no_compilation': True}
    (WORK / 'reference_exactness.results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({k: v for k, v in results.items() if k != 'cases'}, indent=2), flush=True)

if __name__ == '__main__':
    main()
