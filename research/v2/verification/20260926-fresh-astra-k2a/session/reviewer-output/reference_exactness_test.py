"""Independent bounded CPU review. No full training module import or native execution.

Equality is byte equality, with zero tolerance. Candidate AST definitions and
embedding call-site statements execute unchanged. The native algorithm model
below is reviewer-written CPU emulation, NOT NKI execution or a compiler model.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'  # Set only; do not inspect process environments.
import argparse
import ast
import copy
import hashlib
import json
import platform
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(42)
random.seed(42)
ROOT = Path('/work')
RECORDS = []
ARTIFACT_TENSORS = {}
TREES = {name: ast.parse((ROOT / (name + '.py')).read_text()) for name in ('base', 'candidate')}
EXTRACTED = []

def definition(tree, name):
    return next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)

def execute(nodes, ns, source='candidate'):
    for node in nodes:
        EXTRACTED.append(dict(source=source + '.py', name=getattr(node, 'name', type(node).__name__),
                              line=getattr(node, 'lineno', None), end_line=getattr(node, 'end_lineno', None)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[])),
                 str(ROOT / (source + '.py')), 'exec'), ns)

NS = dict(torch=torch, nn=nn, F=F, _embedding_scatter_kernel=None)
execute([definition(TREES['candidate'], '_EmbeddingScatter'),
         definition(TREES['candidate'], '_install_embedding_scatter'),
         definition(TREES['candidate'], 'build_flat_grad_buffer')], NS)
Scatter = NS['_EmbeddingScatter']
NS['_install_embedding_scatter'](torch.device('cpu'))

def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def compare(name, actual, expected, **metadata):
    same_type = actual.shape == expected.shape and actual.dtype == expected.dtype
    exact = same_type and torch.equal(actual.detach().contiguous().view(torch.uint8),
                                      expected.detach().contiguous().view(torch.uint8))
    delta = (actual.detach().double() - expected.detach().double()).abs()
    record = dict(name=name, pass_exact=bool(exact), shape=list(actual.shape), dtype=str(actual.dtype),
                  max_abs_error=delta.max().item() if delta.numel() else 0.0,
                  actual_sha256=digest(actual), expected_sha256=digest(expected), **metadata)
    if not exact:
        indices = torch.nonzero(actual != expected)[:12]
        record['first_numeric_mismatches'] = [dict(index=ix.tolist(), actual=actual[tuple(ix)].item(),
                                                   expected=expected[tuple(ix)].item()) for ix in indices]
        ARTIFACT_TENSORS[name] = {'actual': actual.detach().clone(), 'expected': expected.detach().clone()}
    RECORDS.append(record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return exact

def check(name, condition, **metadata):
    record = dict(name=name, pass_exact=bool(condition), **metadata)
    RECORDS.append(record)
    print(json.dumps(record, sort_keys=True), flush=True)

def sequential_reference(ids, gradient, vocabulary, channels):
    # Intentionally independent of both index_add_ and embedding backward.
    out = torch.zeros((vocabulary, channels), dtype=torch.float32)
    for row, value in zip(ids.reshape(-1).tolist(), gradient.reshape(-1, channels).float()):
        out[row] = out[row] + value
    return out

def tile_coalescing_emulation(ids, gradient, vocabulary, method):
    """Model candidate.py:1930-1979 using CPU tensors, valid finite inputs only.

    Preserve 128-token tiles, two disjoint channel owners, last occurrence,
    unique private dummy rows and FP32 read/add/write. FP32 CPU matmul over
    promoted BF16 operands models algebra, not native reduction instructions.
    The 'sequential' variant isolates across-tile regrouping from CPU matmul.
    """
    ids, g = ids.reshape(-1), gradient.reshape(-1, gradient.shape[-1])
    tokens, channels = g.shape
    assert g.dtype == torch.bfloat16 and tokens % 128 == 0 and vocabulary % 128 == 0
    out = torch.zeros((vocabulary + 128, channels), dtype=torch.float32)
    width = channels // 2
    for c0 in (0, width):
        for start in range(0, tokens, 128):
            rows = ids[start:start + 128].long()
            equal = rows[:, None] == rows[None, :]
            later = torch.arange(128)[None, :] > torch.arange(128)[:, None]
            keep = ~(equal & later).any(dim=1)
            addresses = torch.where(keep, rows, vocabulary + torch.arange(128))
            assert addresses.unique().numel() == 128
            values = g[start:start + 128, c0:c0 + width].float()
            if method == 'matmul':
                grouped = equal.float().T @ values
            else:
                grouped = torch.zeros((128, width), dtype=torch.float32)
                for j in range(128):
                    grouped += equal[:, j:j + 1].float() * values[j]
            selected = grouped * keep[:, None]
            out[addresses, c0:c0 + width] = out[addresses, c0:c0 + width] + selected
    assert torch.count_nonzero(out[vocabulary:]) == 0
    return out[:vocabulary]

def component_cases():
    cases = []
    for kind in ('unique', 'duplicates_within_across_tiles', 'all_same', 'boundary_ids', 'zero', 'cancellation', 'empty', 'noncontiguous'):
        vocabulary, channels = (512, 8)
        ids = torch.arange(256).reshape(2, 128)
        if kind in ('duplicates_within_across_tiles', 'zero', 'cancellation', 'noncontiguous'):
            ids = (ids * 7) % 11
        if kind == 'all_same':
            ids.fill_(7)
            channels = 128
        if kind == 'boundary_ids':
            ids = torch.tensor([0, 255, 256, 511] * 64).reshape(2, 128)
        if kind == 'empty':
            ids = torch.empty((1, 0), dtype=torch.long)
        g = torch.randn((*ids.shape, channels))
        if kind == 'zero':
            g.zero_()
        if kind == 'cancellation':
            ids = torch.arange(128).remainder(11).repeat(2).reshape(2, 128)
            g[1] = -g[0]
        cases.append((kind, ids, g, vocabulary, channels))
    # These values are EXACTLY representable as BF16: no input quantization error.
    for kind, big, small in (('cross_tile_rounding_unit', 1.0, 2.0 ** -24),
                             ('cross_tile_rounding_large', 2.0 ** 24, 1.0)):
        ids = torch.full((2, 128), 7, dtype=torch.long)
        ids.flatten()[[0, 128, 129]] = 3
        g = torch.zeros((2, 128, 8))
        g.reshape(256, 8)[0] = big
        g.reshape(256, 8)[128] = small
        g.reshape(256, 8)[129] = -big
        cases.append((kind, ids, g, 256, 8))
    return cases

def test_components():
    for kind, raw_ids, raw_g, vocabulary, channels in component_cases():
        for dtype in (torch.bfloat16, torch.float32):
            for index_dtype in (torch.int32, torch.int64):
                label = f'component/{kind}/{dtype}/{index_dtype}'
                ids, g = raw_ids.to(index_dtype), raw_g.to(dtype)
                if kind == 'noncontiguous':
                    expanded = torch.zeros((*ids.shape[:-1], ids.shape[-1] * 2), dtype=index_dtype)
                    expanded[..., ::2] = ids
                    ids = expanded[..., ::2]
                    g2 = torch.zeros((*g.shape[:-1], channels * 2), dtype=dtype)
                    g2[..., ::2] = g
                    g = g2[..., ::2]
                    assert not ids.is_contiguous() and not g.is_contiguous()
                w0 = torch.randn(vocabulary, channels)
                dense, candidate = (w0.clone().requires_grad_() for _ in range(2))
                y_dense = F.embedding(ids, dense).to(dtype)
                y_candidate = Scatter.apply(candidate, ids, dtype)
                compare(label + '/forward', y_candidate, y_dense)
                # Reuse the graph twice; .grad must accumulate, not be replaced.
                for backward_index, upstream in enumerate((g, -g * 0.5)):
                    y_dense.backward(upstream, retain_graph=backward_index == 0)
                    y_candidate.backward(upstream, retain_graph=backward_index == 0)
                    one = sequential_reference(ids, upstream, vocabulary, channels)
                    expected = one if backward_index == 0 else expected + one
                    compare(label + f'/backward{backward_index}_dense', candidate.grad, dense.grad)
                    compare(label + f'/backward{backward_index}_sequential', candidate.grad, expected)
                    unused = torch.ones(vocabulary, dtype=torch.bool)
                    unused[ids.reshape(-1).long()] = False
                    check(label + f'/unused{backward_index}', torch.count_nonzero(candidate.grad[unused]) == 0)
                if dtype == torch.bfloat16 and index_dtype == torch.int64 and ids.numel():
                    dense_one = sequential_reference(ids, g, vocabulary, channels)
                    for method in ('matmul', 'sequential'):
                        actual = tile_coalescing_emulation(ids, g, vocabulary, method)
                        compare(f'emulation/{kind}/{method}', actual, dense_one,
                                execution='reviewer CPU emulation; no NKI', input_bf16_exact=bool(torch.equal(raw_g, g.float())))
                        if kind.startswith('cross_tile_rounding'):
                            ARTIFACT_TENSORS[f'inputs/{kind}'] = dict(ids=ids, gradient=g)
    # FP32 upstream supplied to BF16 output is quantized BEFORE FP32 row sums.
    ids = torch.zeros((1, 128), dtype=torch.int32)
    g = torch.full((1, 128, 8), 1.003, dtype=torch.float32)
    w = torch.zeros((32, 8), requires_grad=True)
    Scatter.apply(w, ids, torch.bfloat16).backward(g)
    compare('cast_backward/quantized_upstream', w.grad, sequential_reference(ids, g.bfloat16(), 32, 8))
    check('cast_backward/quantization_is_distinct_from_fp32_accumulation',
          not torch.equal(w.grad, sequential_reference(ids, g, 32, 8)))
    try:
        Scatter.apply(torch.zeros((32, 8), dtype=torch.bfloat16), ids, torch.bfloat16)
    except ValueError as exc:
        check('reject_non_fp32_master', str(exc) == 'K2 requires FP32 embedding masters', message=str(exc))
    else:
        check('reject_non_fp32_master', False)
    for bad in (-1, 32):
        try:
            Scatter.apply(torch.zeros((32, 8)), torch.tensor([[bad]]), torch.bfloat16)
        except (IndexError, RuntimeError) as exc:
            check(f'reject_invalid_index/{bad}', True, message=str(exc))
        else:
            check(f'reject_invalid_index/{bad}', False)

def call_site_namespace(source, mode, dtype):
    tree = TREES[source]
    ns = dict(torch=torch, nn=nn, F=F, argparse=argparse, _EmbeddingScatter=Scatter,
              COMPUTE_DTYPE=dtype, USE_TENSOR_LR_SCALARS=True, USE_CAUTIOUS_WD=False)
    if source == 'candidate':
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and
                          any(isinstance(t, ast.Name) and t.id == 'USE_EMBEDDING_SCATTER' for t in n.targets))
        execute([assignment], ns)
        main = definition(tree, 'main')
        parser_node = next(n for n in ast.walk(main) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                           and n.value.args and isinstance(n.value.args[0], ast.Constant)
                           and n.value.args[0].value == '--embedding-scatter')
        ns['parser'] = argparse.ArgumentParser()
        execute([parser_node], ns)
        args = [] if mode == 'default' else ['--no-embedding-scatter' if mode == 'off' else '--embedding-scatter']
        ns['args'] = ns['parser'].parse_args(args)
        flag_assign = next(n for n in main.body if isinstance(n, ast.Assign) and
                           any(isinstance(t, ast.Name) and t.id == 'USE_EMBEDDING_SCATTER' for t in n.targets))
        execute([flag_assign], ns)
        check(f'cli/{mode}/{dtype}', ns['USE_EMBEDDING_SCATTER'] == (mode == 'on'))
    forward = definition(definition(tree, 'GPT'), 'forward')
    xnode = next(n for n in forward.body if (isinstance(n, ast.If) and 'USE_EMBEDDING_SCATTER' in ast.unparse(n.test))
                 or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'x'))
    loop = next(n for n in forward.body if isinstance(n, ast.For))
    venode = loop.body[0]
    for name, node, result in (('token_lookup', xnode, 'x'), ('value_lookup', venode, 've')):
        wrapper = ast.parse(f'def {name}(self, idx, i=0):\n    pass\n').body[0]
        wrapper.body = [node, ast.Return(value=ast.Name(id=result, ctx=ast.Load()))]
        execute([wrapper], ns, source)
    # Exact unchanged production optimizer methods, with a tiny reviewer holder.
    opt = definition(tree, 'MuonAdamW')
    selected = [definition(opt, n) for n in ('_device_scalar', '_adamw_state', '_adamw_step_foreach')]
    for node in selected:
        execute([node], ns, source)
    holder = type('ReviewerAdamHolder', (), {n.name: ns[n.name] for n in selected})
    ns['holder'] = holder
    return ns

def test_three_steps_and_evaluation():
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(42)
        initial = [torch.randn(64, 8) for _ in range(3)]
        upstreams = [torch.randn(2, 16, 8).to(dtype) for _ in range(6)]
        ids = torch.arange(32, dtype=torch.int32).remainder(7).reshape(2, 16)
        baseline = None
        for source, mode in (('base', 'base'), ('candidate', 'default'), ('candidate', 'off'), ('candidate', 'on')):
            ns = call_site_namespace(source, mode, dtype)
            module = nn.Module()
            module.transformer = nn.ModuleDict({'wte': nn.Embedding(64, 8)})
            module.value_embeds = nn.ModuleDict({'0': nn.Embedding(64, 8), '1': nn.Embedding(64, 8)})
            params = list(module.parameters())
            with torch.no_grad():
                for p, init in zip(params, initial):
                    p.copy_(init)
            flat, _ = NS['build_flat_grad_buffer'](module)
            addresses = [p.grad.data_ptr() for p in params]
            opt = ns['holder']()
            opt.state, opt._scalar_cache = defaultdict(dict), {}
            group = dict(params=params, betas=(0.8, 0.96), lr=0.003, weight_decay=0.001, eps=1e-10)
            snapshots = {}
            for step in range(3):
                flat.zero_()
                for micro in range(2):
                    outputs = [ns['token_lookup'](module, ids)] + [ns['value_lookup'](module, ids, i) for i in (0, 1)]
                    check(f'absent_value/{mode}/{dtype}/{step}/{micro}', ns['value_lookup'](module, ids, 2) is None)
                    loss = sum((out.float() * upstreams[2 * step + micro].float() * (i + 1)).sum()
                               for i, out in enumerate(outputs))
                    loss.backward()
                    for i, out in enumerate(outputs):
                        snapshots[f'{step}/{micro}/forward/{i}'] = out.detach().clone()
                    snapshots[f'{step}/{micro}/grad_buffer'] = flat.clone()
                check(f'grad_alias/{mode}/{dtype}/{step}', [p.grad.data_ptr() for p in params] == addresses)
                opt._adamw_step_foreach(group)
                for i, p in enumerate(params):
                    snapshots[f'{step}/weight/{i}'] = p.detach().clone()
                    for key in ('exp_avg', 'exp_avg_sq'):
                        snapshots[f'{step}/{key}/{i}'] = opt.state[p][key].clone()
            if baseline is None:
                baseline = snapshots
            else:
                for key, value in snapshots.items():
                    compare(f'three_step/{mode}/{dtype}/{key}', value, baseline[key])
            # Evaluation branch MUST bypass the custom Function, even with flag ON.
            module.eval()
            class ForbiddenScatter:
                @staticmethod
                def apply(*args):
                    raise AssertionError('evaluation called embedding scatter')
            ns['_EmbeddingScatter'] = ForbiddenScatter
            compare(f'eval/{mode}/{dtype}/token', ns['token_lookup'](module, ids), module.transformer.wte(ids).to(dtype))
            compare(f'eval/{mode}/{dtype}/value', ns['value_lookup'](module, ids, 0), module.value_embeds['0'](ids).to(dtype))

def static_checks():
    for name in ('load_for_eval', 'save_checkpoint', 'current_toggles', 'restore_eval_toggles',
                 'lr_multiplier', 'build_flat_grad_buffer', 'clip_and_sync_gradients', 'OwnedOptimizer',
                 '_SubmissionEvalGPT', 'MuonAdamW'):
        check('unchanged_ast/' + name, ast.dump(definition(TREES['base'], name)) == ast.dump(definition(TREES['candidate'], name)))
    # Checkpoint flag omission is recorded as an observation, not silently added.
    names_node = next(n for n in TREES['candidate'].body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'ARCH_TOGGLE_NAMES' for t in n.targets))
    check('config/scatter_absent_from_eval_whitelist', 'USE_EMBEDDING_SCATTER' not in ast.literal_eval(names_node.value),
          interpretation='Training-backward-only flag omitted from saved toggles; eval bypasses it. No full load_for_eval executed.')

def main():
    metadata = dict(seed=42, python=platform.python_version(), torch=torch.__version__, device='cpu',
                    threads=dict(intra_op=torch.get_num_threads(), inter_op=torch.get_num_interop_threads(),
                                 configured_OMP=1, configured_MKL=1, configured_OpenBLAS=1, configured_NUMEXPR=1),
                    limits=dict(max_width=128, max_vocabulary=512, max_tokens=256, max_context=128, max_microbatch=2),
                    tolerance=dict(atol=0, rtol=0, criterion='same dtype/shape and identical tensor bytes'))
    print(json.dumps(dict(metadata=metadata)), flush=True)
    error = None
    try:
        static_checks()
        test_components()
        test_three_steps_and_evaluation()
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
    failures = [r for r in RECORDS if not r['pass_exact']]
    result = dict(metadata=metadata, extracted_code=EXTRACTED, checks=RECORDS, failures=failures,
                  harness_error=error, summary=dict(checks=len(RECORDS), failures=len(failures),
                  actual_cpu_path_failures=sum(not r['name'].startswith('emulation/') for r in failures)),
                  native_executed=False)
    (ROOT / 'reference_results.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    torch.save(ARTIFACT_TENSORS, ROOT / 'reference_tensors.pt')
    (ROOT / 'torch_build.txt').write_text(torch.__config__.show())
    print(json.dumps(dict(summary=result['summary'], harness_error=error)), flush=True)
    return 2 if error else (1 if failures else 0)

if __name__ == '__main__':
    sys.exit(main())
