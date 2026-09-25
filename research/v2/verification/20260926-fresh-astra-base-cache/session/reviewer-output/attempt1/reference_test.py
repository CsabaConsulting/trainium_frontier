"""Independent, bounded CPU exactness reference; never imports either training module.

Extracts only named definitions and the CLI/refresh statements from authenticated
sources. Uses at most 256x8 activations, 64 master scalars per fixture, four steps,
and three sequential backward calls per step. No distributed/compiled/native run.
"""
import argparse
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import textwrap
import traceback
import types

import torch
from torch import nn
from torch.nn import functional as F

OUT = Path('/work')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(250925)
RESULT = {'status': 'RUNNING', 'python': sys.version, 'torch': torch.__version__,
          'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
          'device': 'cpu', 'seed': 250925, 'steps': 4, 'microbatches_per_step': 3,
          'comparisons': [], 'cases': [], 'diagnostics': {}}
RAW = {}

def check(ok, message):
    if not ok:
        raise AssertionError(message)

def bytes_of(t):
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

def summary(t):
    return dict(shape=list(t.shape), dtype=str(t.dtype),
                sha256=hashlib.sha256(bytes_of(t)).hexdigest())

def exact(a, b, label):
    if isinstance(a, torch.Tensor):
        ok = a.dtype == b.dtype and a.shape == b.shape and bytes_of(a) == bytes_of(b)
        RESULT['comparisons'].append(dict(label=label, exact=ok, a=summary(a), b=summary(b),
             max_abs=float((a.double()-b.double()).abs().max()) if a.numel() else 0.0))
        check(ok, label)
    elif isinstance(a, dict):
        check(a.keys() == b.keys(), label + ':keys')
        for k in a:
            exact(a[k], b[k], label + '/' + str(k))
    elif isinstance(a, (list, tuple)):
        check(len(a) == len(b), label + ':length')
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, label + '/' + str(i))
    else:
        check(a == b, label + ':value')

def snapshot(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: snapshot(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [snapshot(v) for v in x]
    return x

def reaches(cache, parameter):
    if cache is parameter:
        return True
    seen = set()
    queue = [cache.grad_fn]
    while queue:
        node = queue.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if getattr(node, 'variable', None) is parameter:
            return True
        queue.extend(n for n, _ in node.next_functions)
    return False

trees = {}
names = {'Linear', 'MLP', 'relu_squared', 'transpose_linear_weights',
         'reference_forward', 'make_autograd', 'make_forward', '_muon_math', '_muon_lr_g'}
envs = {}
for arm in ['base', 'candidate']:
    src = (OUT / f'{arm}.py').read_text()
    tree = ast.parse(src)
    trees[arm] = tree
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    selected += [n for n in tree.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'POLAR_EXPRESS_COEFFS' for t in n.targets)]
    check(len(selected) == len(names) + 1, 'all requested definitions extracted')
    extracted = '\n\n'.join(ast.get_source_segment(src, n) for n in selected) + '\n'
    (OUT / f'extracted_{arm}.py').write_text(extracted)
    env = dict(torch=torch, nn=nn, F=F, GPTConfig=types.SimpleNamespace,
               USE_WEIGHT_CACHE=True, USE_COALESCED_SWIGLU=False,
               USE_GATED_MLP=False, USE_NKI_RELU2=False, _HAS_NKI=False, RELU2_TAU=0.0)
    exec(compile(extracted, f'extracted_{arm}.py', 'exec'), env)
    envs[arm] = env

candidate_main = next(n for n in trees['candidate'].body if isinstance(n, ast.FunctionDef) and n.name == 'main')
cli = next(n for n in candidate_main.body if isinstance(n, ast.Expr) and
           isinstance(n.value, ast.Call) and n.value.args and
           isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value == '--initial-cache-refresh')
loop = next(n for n in candidate_main.body if isinstance(n, ast.While))
initial = next(n for n in loop.body if isinstance(n, ast.If) and 'initial_cache_refresh' in ast.unparse(n.test))
later = next(n for n in loop.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'cached_linears')
src = (OUT / 'candidate.py').read_text()
extracted_controls = '\n\n'.join(textwrap.dedent(ast.get_source_segment(src, n)) for n in [cli, initial, later])
# get_source_segment omits first-line indentation; unparse gives valid executable statements.
(OUT / 'extracted_controls.py').write_text('\n\n'.join(ast.unparse(n) for n in [cli, initial, later]) + '\n')
reduced = copy.deepcopy(trees['candidate'])
rm = next(n for n in reduced.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
rm.body = [n for n in rm.body if n.lineno != cli.lineno]
rl = next(n for n in rm.body if isinstance(n, ast.While))
rl.body = [n for n in rl.body if n.lineno != initial.lineno]
check(ast.dump(reduced, include_attributes=False) == ast.dump(trees['base'], include_attributes=False),
      'removing exactly two added AST statements reproduces complete base AST')
positions = {label: n.lineno for label, n in [('cli', cli), ('initial', initial), ('later', later)]}
timer = next(n for n in loop.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 't0' for t in n.targets))
micro = next(n for n in loop.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'micro_index')
optimizer_step = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'not skip_step')
check(timer.lineno < initial.lineno < micro.lineno < optimizer_step.lineno < later.lineno,
      'refresh and optimizer source order')
RESULT['source_checks'] = dict(all_other_ast_identical=True, source_lines=positions,
                               timed_initial_refresh=True, later_refresh_unchanged=True)
parser = argparse.ArgumentParser()
exec(compile(ast.Module(body=[cli], type_ignores=[]), '<extracted-cli>', 'exec'), {'parser': parser, 'argparse': argparse})
options = {'default': parser.parse_args([]), 'off': parser.parse_args(['--no-initial-cache-refresh']),
           'on': parser.parse_args(['--initial-cache-refresh'])}
check([options[a].initial_cache_refresh for a in ['default', 'off', 'on']] == [False, False, True], 'Boolean CLI')
initial_code = compile(ast.Module(body=[initial], type_ignores=[]), '<extracted-initial-refresh>', 'exec')
later_code = compile(ast.Module(body=[later], type_ignores=[]), '<extracted-later-refresh>', 'exec')

def make_fixture(env, family, transposed):
    if family == 'linear':
        model = env['Linear'](4, 3, bias=False)
    else:
        model = env['MLP'](types.SimpleNamespace(n_embd=4, mlp_ratio=2))
        if family == 'adapted_mlp':
            forward = env['make_forward']('cpu', type(model).forward, True, forward_backend='native')
            model.forward = types.MethodType(forward, model)
    if transposed:
        env['transpose_linear_weights'](model)
    return model

def oracle_forward(model, x, family, transposed):
    # Independent ordinary PyTorch reference, with a fresh differentiable cast every call.
    if family == 'linear':
        w = model.weight.to(x.dtype)
        return x @ w if transposed else F.linear(x, w)
    h = F.linear(x, model.c_fc.weight.to(x.dtype))
    return F.linear(torch.relu(h).square(), model.c_proj.weight.to(x.dtype))

def do_update(model, state, optimizer, step, env):
    if optimizer is not None:
        optimizer.param_groups[0]['lr'] = 0.025 / (step + 1)
        optimizer.step()
        return snapshot(optimizer.state_dict())
    # Actual extracted pure Muon update math on tiny weights, independently fed per arm.
    with torch.no_grad():
        for name, p in model.named_parameters():
            w, g = p.unsqueeze(0), p.grad.unsqueeze(0)
            tall = w.shape[-2] >= w.shape[-1]
            red_dim = -1 if tall else -2
            if name not in state:
                shape = list(w.shape)
                shape[red_dim] = 1
                state[name] = [torch.zeros_like(w), torch.zeros(shape)]
            mom, second = state[name]
            updated, mom, second = env['_muon_math'](
                w, g, mom, second, torch.tensor(.9), torch.tensor(.1),
                torch.tensor(.0125 / (step + 1)), torch.tensor(.0001),
                tall, 5, red_dim, w.size(red_dim), .1, True)
            p.copy_(updated.squeeze(0))
            state[name] = [mom, second]
    return snapshot(state)

def run_arm(arm, family, dtype, cache_enabled, transposed, update, initial_state, inputs, upstreams):
    env = envs['base' if arm == 'base' else 'candidate']
    env['USE_WEIGHT_CACHE'] = cache_enabled
    model = make_fixture(env, family, transposed)
    model.load_state_dict(initial_state, strict=True)
    check(sum(p.numel() for p in model.parameters()) <= 64, 'tiny parameter bound')
    params_before = tuple(id(p) for p in model.parameters())
    cached = [m for m in model.modules() if isinstance(m, env['Linear'])] if cache_enabled and arm != 'oracle' else []
    trace = []
    cache_trace = []
    observed_backward = []
    optim = torch.optim.AdamW(model.parameters(), lr=.025, betas=(.8, .9), eps=1e-8, weight_decay=.1, foreach=False) if update == 'adamw' else None
    state = {}
    for step in range(4):
        model.zero_grad(set_to_none=True)
        scope = dict(step=step, args=options.get(arm, options['off']),
                     cached_linears=cached, COMPUTE_DTYPE=dtype)
        if arm not in ['base', 'oracle']:
            exec(initial_code, scope)
        entries = []
        for m in cached:
            c = m._cached_weight
            if c is not None:
                check(c.dtype == dtype and reaches(c, m.weight), 'cache dtype/connectivity')
                exact(c, m.weight.to(dtype), f'{arm}/step{step}/cache_value')
            entries.append(None if c is None else dict(dtype=str(c.dtype), requires_grad=c.requires_grad,
                         grad_fn=type(c.grad_fn).__name__ if c.grad_fn else None,
                         connected=reaches(c, m.weight), master_dtype=str(m.weight.dtype)))
        if step == 0 and cached:
            check(all((e is not None) == (arm == 'on') for e in entries), 'first-cache arm selection')
        cache_trace.append(entries)
        cache_ids = [id(m._cached_weight) for m in cached]
        micros = []
        for micro in range(3):
            x = inputs[step][micro].detach().clone().requires_grad_()
            y = oracle_forward(model, x, family, transposed) if arm == 'oracle' else model(x)
            if family == 'adapted_mlp' and arm != 'oracle':
                todo, seen, nodes = [y.grad_fn], set(), []
                while todo:
                    n = todo.pop()
                    if n is None or n in seen:
                        continue
                    seen.add(n)
                    if type(n).__name__ == 'FusedMLPBackward':
                        nodes.append(n)
                    todo.extend(q for q, _ in n.next_functions)
                check(len(nodes) == 1, 'actual adapted MLP branch selected')
                nodes[0].register_hook(lambda *_, s=step, m=micro: observed_backward.append([s, m]))
            loss = (y.float() * upstreams[step][micro].float()).sum()
            loss.backward()
            check(all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()
                      for p in model.parameters()), 'finite connected FP32 master gradients')
            micros.append(dict(output=snapshot(y), loss=snapshot(loss), input_gradient=snapshot(x.grad),
                               accumulated_gradients={n: snapshot(p.grad) for n,p in model.named_parameters()}))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, foreach=False)
        clipped = {n: snapshot(p.grad) for n,p in model.named_parameters()}
        update_state = do_update(model, state, optim, step, env)
        trace.append(dict(micros=micros, gradient_norm=snapshot(norm), clipped_gradients=clipped,
                          parameters=snapshot(model.state_dict()), optimizer_state=update_state))
        exec(later_code, scope)
        for m, old_id in zip(cached, cache_ids):
            check(reaches(m._cached_weight, m.weight), 'post-update connectivity')
            exact(m._cached_weight, m.weight.to(dtype), f'{arm}/step{step}/fresh_after_update')
            if dtype == torch.bfloat16:
                check(id(m._cached_weight) != old_id, 'BF16 cache replaced after step')
        check(tuple(id(p) for p in model.parameters()) == params_before, 'optimizer still owns original masters')
        check(all(p.dtype == torch.float32 for p in model.parameters()), 'FP32 master dtype preserved')
        check(all('cache' not in k for k in model.state_dict()), 'no serialized caches')
    check(any(not torch.equal(initial_state[n], p) for n,p in model.state_dict().items()), 'nontrivial update')
    if family == 'adapted_mlp' and arm != 'oracle':
        check(observed_backward == [[s,m] for s in range(4) for m in range(3)], 'executed backward hook count')
    # Serialize only model state and strict-load fresh modules, as the real evaluation path does.
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    loaded = torch.load(stream, map_location='cpu', weights_only=True)
    fresh = make_fixture(env, family, transposed)
    fresh.load_state_dict(loaded, strict=True)
    fresh.eval().requires_grad_(False)
    check(all(m._cached_weight is None for m in fresh.modules() if isinstance(m, env['Linear'])), 'fresh load has no caches')
    model.eval()
    with torch.no_grad():
        evaluation = model(inputs[0][0])
        reloaded_evaluation = fresh(inputs[0][0])
    exact(evaluation, reloaded_evaluation, f'{arm}/fresh_load_eval')
    return dict(trace=trace, final_state=loaded, evaluation=evaluation, cache_trace=cache_trace,
                executed_custom_backward_hooks=observed_backward)

try:
    configs = []
    for update in ['adamw', 'muon']:
        for family in ['linear', 'native_mlp', 'adapted_mlp']:
            configs.append((family, torch.bfloat16, True, False, update))
    configs += [('linear', torch.bfloat16, True, True, 'adamw'),
                ('linear', torch.float32, True, False, 'adamw'),
                ('linear', torch.bfloat16, False, False, 'adamw')]
    for index, (family, dtype, caching, transposed, update) in enumerate(configs):
        key = f'{index}_{family}_{dtype}_{caching}_{transposed}_{update}'
        torch.manual_seed(250925 + index)
        initial_model = make_fixture(envs['candidate'], family, transposed)
        initial_state = snapshot(initial_model.state_dict())
        shape = (256, 4) if family == 'adapted_mlp' else (2, 3, 4)
        outshape = shape[:-1] + ((3,) if family == 'linear' else (4,))
        inputs = [[torch.randn(shape).to(dtype) for _ in range(3)] for _ in range(4)]
        upstreams = [[(torch.randn(outshape) / 32).to(dtype) for _ in range(3)] for _ in range(4)]
        data = dict(initial_state=initial_state, inputs=inputs, upstreams=upstreams, arms={})
        RAW[key] = data
        for arm in ['base', 'default', 'off', 'on', 'oracle']:
            data['arms'][arm] = run_arm(arm, family, dtype, caching, transposed, update, initial_state, inputs, upstreams)
        for arm in ['default', 'off', 'on', 'oracle']:
            for field in ['trace', 'final_state', 'evaluation']:
                exact(data['arms']['base'][field], data['arms'][arm][field], f'{key}/base_vs_{arm}/{field}')
        RESULT['cases'].append(dict(name=key, status='PASS', master_elements=sum(v.numel() for v in initial_state.values()),
                                   input_shape=list(shape), custom_backward_executions_per_arm=
                                   {a: len(v['executed_custom_backward_hooks']) for a,v in data['arms'].items()}))
        print('PASS', key, flush=True)

    # Sensitivity controls: detached and stale casts must be detected by these checks.
    env = envs['candidate']
    env['USE_WEIGHT_CACHE'] = True
    m = env['Linear'](4, 3, bias=False)
    m.refresh_weight_cache(torch.bfloat16)
    good = m._cached_weight
    m._cached_weight = good.detach()
    x = torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True)
    m(x).float().sum().backward()
    check(m.weight.grad is None and not reaches(m._cached_weight, m.weight), 'detached-cache negative control')
    RESULT['diagnostics']['detached_cache_detected'] = True
    # Hot-loading into an already cached instance is unsupported in the unchanged class.
    stale = {}
    for name in ['base', 'candidate']:
        e = envs[name]
        e['USE_WEIGHT_CACHE'] = True
        m = e['Linear'](4, 3, bias=False)
        with torch.no_grad():
            m.weight.fill_(1.0)
        m.refresh_weight_cache(torch.bfloat16)
        m.load_state_dict({'weight': torch.full((3,4), 2.0)})
        before = m(x.detach())
        wanted = F.linear(x.detach(), m.weight.to(torch.bfloat16))
        check(not torch.equal(before, wanted), 'stale-cache negative control')
        m.refresh_weight_cache(torch.bfloat16)
        after = m(x.detach())
        exact(after, wanted, name + '/explicit_refresh_after_hot_load')
        stale[name] = dict(stale_output=before.tolist(), fresh_output=after.tolist())
    RESULT['diagnostics']['hot_load_stale_cache_preexisting_in_both'] = stale
    RESULT['status'] = 'PASS'
except Exception:
    RESULT['status'] = 'FAIL'
    RESULT['traceback'] = traceback.format_exc()
    print(RESULT['traceback'], file=sys.stderr)
finally:
    RESULT['tensor_comparisons'] = len(RESULT['comparisons'])
    RESULT['nonexact_comparisons'] = sum(not c['exact'] for c in RESULT['comparisons'])
    torch.save(RAW, OUT / 'reference_raw.pt')
    (OUT / 'reference_results.json').write_text(json.dumps(RESULT, indent=2) + '\n')
    print(json.dumps({k: RESULT[k] for k in ['status', 'tensor_comparisons', 'nonexact_comparisons']}, indent=2))
sys.exit(0 if RESULT['status'] == 'PASS' else 1)
