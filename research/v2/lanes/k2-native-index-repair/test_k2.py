"""CPU operation exactness and reduced-shape eight-layer OFF identity.

The NKI body is interpreted, never claimed to be NKI simulation/compilation.
The actual torch device metadata producer and actual registered autograd are
executed. Native numerical semantics and speed remain to be measured.
"""
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
from cpu_instruction_model import interpreted_kernel, Tensor

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
INDEX_DTYPE = torch.int64

# Test the actual inlined helpers, never the standalone reference modules.
helper_tree = ast.parse((ROOT/'train.py').read_text())
helper_names = {'_k2_prepare_metadata', '_k2_build_registered_lookup'}
helper_tree = ast.Module(body=[n for n in helper_tree.body
                             if isinstance(n, ast.FunctionDef) and n.name in helper_names],
                         type_ignores=[])
helper_namespace = {'torch': torch, 'F': F}
exec(compile(helper_tree, str(ROOT/'train.py'), 'exec'), helper_namespace)
prepare_metadata = helper_namespace['_k2_prepare_metadata']
build_registered_lookup = helper_namespace['_k2_build_registered_lookup']


def bits(x):
    return x.detach().contiguous().reshape(-1).view(torch.uint8)


def exact(a, b):
    if not torch.equal(bits(a), bits(b)):
        raise AssertionError('bitwise mismatch')


def cpu_kernel(g, counts, starts, positions, maximums):
    features = g.shape[1]//2
    with interpreted_kernel(counts.shape[0], g.shape[0], g.shape[1],
                            counts.shape[0]//maximums.shape[0]) as (kernel, lang, isa):
        parts = []
        for owner in (0, 1):
            lang.pid = owner
            out = kernel(Tensor(g), *(Tensor(x) for x in (counts, starts, positions, maximums)))
            parts.append(out.data[:, owner*features:(owner+1)*features])
        return torch.cat(parts, dim=1)


lookup = build_registered_lookup(kernel=cpu_kernel, nki_op_factory=torch.library.custom_op)


def operation_tests():
    fixtures = []
    values = {
        'min_pos': [0x0001], 'min_neg': [0x8001],
        'normal_cancel_pos': [0x0081, 0x8080],
        'normal_cancel_neg': [0x8081, 0x0080],
        'large_order': [0x4b80, 0x3f80, 0xcb80],
        'small_order': [0x3f80, 0x3380, 0xbf80],
        'zero_sign': [0x8000, 0x0000],
        'cancel_then_subnormal': [0x4b80, 0xcb80, 0x0001],
        'overflow_pos': [0x7f7f, 0x7f7f], 'overflow_neg': [0xff7f, 0xff7f],
        'inf_cancel': [0x7f80, 0xff80], 'nan': [0x7fc1],
    }
    cases = []
    for name, raw in values.items():
        ids = torch.full((2, 8), 7, dtype=torch.int64)
        encoded = torch.zeros((16, 8), dtype=torch.int16)
        for j, x in enumerate(raw):
            ids.reshape(-1)[j*5] = 0
            encoded[j*5] = x if x < 32768 else x - 65536
        cases.append((name, ids, encoded.view(torch.bfloat16)))
    torch.manual_seed(42)
    for name, ids in [('balanced', torch.arange(16) % 8),
                      ('duplicates', torch.tensor([7, 0, 3, 0, 7, 3, 0, 7]*2)),
                      ('all_one_row', torch.full((16,), 7)),
                      ('empty_group', torch.tensor([4, 7]*8))]:
        cases.append((name, ids.reshape(2, 8), torch.randn(16, 8).bfloat16()))
    for name, ids, tangent in cases:
        w = torch.randn(8, 8, requires_grad=True)
        wr = w.detach().clone().requires_grad_()
        ids = ids.to(INDEX_DTYPE)
        original = ids.clone()
        meta = prepare_metadata(ids, 8, 4)
        actual = lookup(w, *meta)
        expected = F.embedding(original, wr).bfloat16()
        exact(actual, expected)
        # Reused loader storage must not affect saved routing or forward.
        ids.fill_(1)
        actual.backward(tangent.reshape(2, 8, 8))
        expected.backward(tangent.reshape(2, 8, 8))
        finite = bool(torch.isfinite(wr.grad).all())
        if finite:
            exact(w.grad, wr.grad)
        else:
            assert torch.equal(torch.isnan(w.grad), torch.isnan(wr.grad))
            assert torch.equal(torch.isinf(w.grad), torch.isinf(wr.grad))
            exact(w.grad[torch.isinf(w.grad)], wr.grad[torch.isinf(wr.grad)])
        fixtures.append({'name': name, 'verdict': 'EXACT_FINITE' if finite else 'FAIL_NONFINITE_DIAGNOSTIC',
                         'actual_bits': w.grad.view(torch.int32).tolist(),
                         'expected_bits': wr.grad.view(torch.int32).tolist()})
    # Full production metadata executes without production gradient allocation.
    for ids in (torch.arange(32768).reshape(32, 1024) % 8192,
                torch.full((32, 1024), 8191, dtype=torch.int64),
                torch.randint(8192, (32, 1024))):
        ids = ids.to(INDEX_DTYPE)
        snap, counts, starts, positions, maximums = prepare_metadata(ids)
        flat = ids.reshape(-1)
        order = sorted(range(flat.numel()), key=lambda i: int(flat[i]))
        exact(positions.reshape(-1).long(), torch.tensor(order))
        exact(counts.reshape(-1).long(), torch.bincount(flat, minlength=8192))
        assert int(counts.sum()) == 32768
        assert snap.data_ptr() != ids.data_ptr()
    # Two tables and two accumulation microsteps retain separate gradients.
    weights = [torch.randn(8, 8, requires_grad=True) for _ in range(2)]
    references = [w.detach().clone().requires_grad_() for w in weights]
    for step in range(2):
        ids = torch.randint(8, (2, 8), dtype=INDEX_DTYPE)
        meta = prepare_metadata(ids, 8, 4)
        for w, wr in zip(weights, references):
            g = torch.randn(2, 8, 8).bfloat16()
            lookup(w, *meta).backward(g)
            F.embedding(ids, wr).bfloat16().backward(g)
            exact(w.grad, wr.grad)
    return fixtures


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def off_identity(base):
    # Neither test calls data/scoring. Use the repository's existing explicit
    # test-only prepare import and a parquet import placeholder, since this
    # laptop lacks pyarrow. No implementation under test is changed.
    sys.path.insert(0, str(base.parents[3]/'workflow/tools/stub_prepare'))
    arrow = types.ModuleType('pyarrow')
    parquet = types.ModuleType('pyarrow.parquet')
    arrow.parquet = parquet
    sys.modules['pyarrow'], sys.modules['pyarrow.parquet'] = arrow, parquet
    a, b = load('k2_base', base), load('k2_candidate', ROOT/'train.py')
    assert b.K2_EMBEDDING is False
    models = []
    for module in (a, b):
        torch.manual_seed(42)
        model = module.GPT(module.GPTConfig(sequence_len=8, vocab_size=64, n_layer=8,
                                           n_head=2, n_embd=64, mlp_ratio=2))
        model.init_weights()
        models.append(model)
    opts = [torch.optim.SGD(m.parameters(), lr=.01) for m in models]
    losses = []
    metadata_calls = []
    def metadata(ids, vocab):
        metadata_calls.append(1)
        return prepare_metadata(ids, vocab, 4)
    for step in range(6):
        if step == 3:
            b.K2_EMBEDDING, b._K2_LOOKUP, b._K2_METADATA = True, lookup, metadata
        torch.manual_seed(100 + step)
        ids, targets = torch.randint(64, (2, 8), dtype=INDEX_DTYPE), torch.randint(64, (2, 8))
        current = []
        for model, opt in zip(models, opts):
            opt.zero_grad(set_to_none=True)
            loss = model(ids, targets)
            loss.backward()
            current.append(loss.detach())
        exact(*current)
        for pa, pb in zip(models[0].parameters(), models[1].parameters()):
            if pa.grad is None or pb.grad is None:
                assert pa.grad is pb.grad
            else:
                exact(pa.grad, pb.grad)
        for opt in opts:
            opt.step()
        for key, value in models[0].state_dict().items():
            exact(value, models[1].state_dict()[key])
        losses.append(float(current[0]))
    assert len(metadata_calls) == 3
    models[0].eval(), models[1].eval()
    with torch.no_grad():
        exact(models[0](ids), models[1](ids))
    assert len(metadata_calls) == 3
    return {'status': 'OFF_3_STEPS_BIT_IDENTICAL', 'losses': losses[:3],
            'on_status': 'ON_3_STEPS_BIT_IDENTICAL_CPU_INTERPRETER', 'on_losses': losses[3:],
            'metadata_calls_for_three_on_steps': len(metadata_calls), 'eval_ordinary': True,
            'geometry': '8 layers, width64, context8, batch2', 'optimizer': 'SGD .01',
            'scope': 'real model forward/backward/updates; not full training-loop or native proof',
            'import_shims': ['existing test-only prepare', 'unused pyarrow.parquet placeholder']}


def input_contract_tests():
    ids64 = torch.tensor([[0, 7, 0, 3], [7, 3, 0, 7]], dtype=torch.int64).t()
    assert not ids64.is_contiguous()
    meta64 = prepare_metadata(ids64, 8, 4)
    meta32 = prepare_metadata(ids64.to(torch.int32), 8, 4)
    exact(meta64[0], meta32[0].long())
    assert meta32[0].dtype == torch.int32 and meta64[0].dtype == torch.int64
    for left, right in zip(meta64[1:], meta32[1:]):
        exact(left, right)
        assert left.dtype == right.dtype == torch.int32

    rejected = []
    def refuses_metadata(name, ids, vocab=8, group=4):
        try:
            prepare_metadata(ids, vocab, group)
        except ValueError:
            rejected.append(name)
            return
        raise AssertionError('invalid metadata accepted: ' + name)

    for dtype in (torch.float32, torch.float64, torch.bfloat16,
                  torch.bool, torch.int16, torch.uint8):
        refuses_metadata(str(dtype), ids64.to(dtype))
    for dtype in (torch.int32, torch.int64):
        for shape in ((), (8,), (1, 2, 4)):
            refuses_metadata(str(dtype) + str(shape), torch.zeros(shape, dtype=dtype))
        refuses_metadata(str(dtype) + ' bad geometry', torch.zeros(2, 8, dtype=dtype), 7, 4)
        refuses_metadata(str(dtype) + ' position overflow',
                         torch.zeros(1, dtype=dtype).expand(4096, 4096))
        for invalid in (-1, 8):
            ids = torch.tensor([[0, invalid]], dtype=dtype)
            try:
                lookup(torch.randn(8, 8), *prepare_metadata(ids, 8, 4))
            except (IndexError, RuntimeError):
                rejected.append(str(dtype) + ' range ' + str(invalid))
            else:
                raise AssertionError('out-of-range gather accepted')
    # A value outside int32 cannot be silently narrowed into a valid row.
    large = torch.tensor([[2**32]], dtype=torch.int64)
    meta = prepare_metadata(large, 8, 4)
    assert int(meta[0][0, 0]) == 2**32
    try:
        lookup(torch.randn(8, 8), *meta)
    except (IndexError, RuntimeError):
        rejected.append('int64 no narrowing')
    else:
        raise AssertionError('large int64 ID accepted')

    frozen = ROOT.parent / 'k2-final-sprint' / 'train.py'
    tree = ast.parse(frozen.read_text())
    only = ast.Module(body=[n for n in tree.body
                           if isinstance(n, ast.FunctionDef) and n.name == '_k2_prepare_metadata'],
                      type_ignores=[])
    ns = {'torch': torch}
    exec(compile(only, str(frozen), 'exec'), ns)
    try:
        ns['_k2_prepare_metadata'](ids64.to(torch.int32), 8, 4)
    except ValueError as exc:
        assert str(exc) == 'K2 requires two-dimensional int64 token IDs'
    else:
        raise AssertionError('frozen failure did not reproduce')
    return {'int32_int64_metadata': 'BIT_IDENTICAL_VALUES_AND_ROUTING',
            'dtype_preserved': True, 'frozen_int32_failure_reproduced': True,
            'rejected': rejected, 'range_guard': 'unchanged F.embedding rejection'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    by_dtype = {}
    for INDEX_DTYPE in (torch.int32, torch.int64):
        torch.manual_seed(42)
        by_dtype[str(INDEX_DTYPE)] = {'operation': operation_tests(), 'model': off_identity(args.base)}
    result = {'by_index_dtype': by_dtype, 'input_contract': input_contract_tests(),
              'native_compiled': False, 'torch': torch.__version__,
              'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (ROOT/'train.py', ROOT/'test_k2.py',
                                   ROOT/'cpu_instruction_model.py',
                                   ROOT.parent/'k2-final-sprint'/'train.py', args.base)}}
    (ROOT/'cpu-result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'by_index_dtype': {
        dtype: {'operation_cases': len(value['operation']), 'model': value['model']}
        for dtype, value in result['by_index_dtype'].items()},
        'input_contract': result['input_contract'], 'native_compiled': False}))
