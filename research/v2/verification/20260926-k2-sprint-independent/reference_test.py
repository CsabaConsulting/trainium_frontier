"""Independent bounded CPU checks of the frozen candidate.

No native claims. The sequential reference below does not use metadata, sort,
scatter/reduction, or the author's reference implementation. The actual NKI
source is exercised through the supplied instruction interpreter, whose native
limitations remain explicit. Full-source imports use unused prepare/pyarrow
placeholders constructed here; no external repository inputs are read.
"""
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
torch.manual_seed(42)
torch.set_flush_denormal(False)
sys.path.insert(0, str(ROOT))
from cpu_instruction_model import Tensor, interpreted_kernel


def bits(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def exact(actual, expected, label):
    assert actual.dtype == expected.dtype, label + ': dtype'
    assert actual.shape == expected.shape, label + ': shape'
    assert torch.equal(bits(actual), bits(expected)), label + ': bits'


def sequential_reference(ids, tangent, vocab):
    """Plain-Torch chronological FP32 accumulation from actual BF16 tangents."""
    assert tangent.dtype == torch.bfloat16
    flat_ids = ids.reshape(-1)
    flat_g = tangent.reshape(flat_ids.numel(), -1)
    result = torch.zeros(vocab, flat_g.shape[1], dtype=torch.float32)
    for token_index in range(flat_ids.numel()):
        result[int(flat_ids[token_index])].add_(flat_g[token_index].float())
    return result


tree = ast.parse((ROOT / 'train.py').read_text())
helper_names = {'_k2_prepare_metadata', '_k2_build_registered_lookup'}
helper_module = ast.Module(body=[n for n in tree.body
                              if isinstance(n, ast.FunctionDef) and n.name in helper_names],
                          type_ignores=[])
namespace = {'torch': torch, 'F': F}
exec(compile(helper_module, str(ROOT / 'train.py'), 'exec'), namespace)
metadata = namespace['_k2_prepare_metadata']
build_lookup = namespace['_k2_build_registered_lookup']
kernel_calls = []


def interpreted_actual_kernel(g, counts, starts, positions, maximums):
    assert g.dtype == torch.bfloat16 and g.is_contiguous()
    group = counts.shape[0] // maximums.shape[0]
    halves = []
    with interpreted_kernel(counts.shape[0], g.shape[0], g.shape[1], group) as (kernel, lang, isa):
        for owner in range(2):
            lang.pid = owner
            output = kernel(Tensor(g), *(Tensor(x) for x in (counts, starts, positions, maximums)))
            half = slice(owner * (g.shape[1] // 2), (owner + 1) * (g.shape[1] // 2))
            assert torch.all(output.writes[:, half] == 1), 'unique owner writes'
            assert torch.all(output.writes[:, slice((1-owner)*(g.shape[1]//2), (2-owner)*(g.shape[1]//2))] == 0)
            halves.append(output.data[:, half])
    kernel_calls.append({'shape': list(g.shape), 'dtype': str(g.dtype)})
    return torch.cat(halves, dim=1)


lookup = build_lookup(kernel=interpreted_actual_kernel, nki_op_factory=torch.library.custom_op)


def metadata_check(ids, vocab, group):
    snapshot, counts, starts, positions, maxima = metadata(ids, vocab, group)
    flat = ids.reshape(-1)
    for row in range(vocab):
        chronological = torch.tensor([i for i in range(flat.numel()) if int(flat[i]) == row], dtype=torch.int64)
        count, start = int(counts[row]), int(starts[row])
        exact(positions[start:start+count].reshape(-1).long(), chronological, 'metadata token order')
        assert count == chronological.numel()
    for block in range(vocab // group):
        assert int(maxima[block]) == max(int(counts[r]) for r in range(block*group, (block+1)*group))
    assert snapshot.data_ptr() != ids.data_ptr()
    return snapshot, counts, starts, positions, maxima


def operation_checks():
    vocab, width = 12, 10
    # Noncontiguous IDs and tangent; interleaved duplicates span groups and
    # include an empty group. The token sequence is defined by logical reshape.
    ids = torch.tensor([[0, 9, 0, 9, 0, 9], [3, 11, 3, 11, 3, 11]], dtype=torch.int64).t()
    assert not ids.is_contiguous()
    raw = torch.zeros(ids.numel(), width, dtype=torch.int16)
    witnesses = [[0x4b80, 0x3f80, 0xcb80], [0x3f80, 0x3380, 0xbf80],
                 [0x0081, 0x8080, 0x8000], [0x8081, 0x0080, 0x0000],
                 [0x0001, 0x8000, 0x0000], [0x8001, 0x0000, 0x8000],
                 [0x8000, 0x8000, 0x0000], [0x4b80, 0xcb80, 0x0001],
                 [0x3f80, 0xbf80, 0x3380], [0x0000, 0x8000, 0x8000]]
    zero_tokens = [i for i, row in enumerate(ids.reshape(-1)) if int(row) == 0]
    for channel, witness in enumerate(witnesses):
        for token, encoded in zip(zero_tokens, witness):
            raw[token, channel] = encoded if encoded < 32768 else encoded - 65536
    g = raw.view(torch.bfloat16).reshape(*ids.shape, width).transpose(0, 1).contiguous().transpose(0, 1)
    assert not g.is_contiguous()
    cases = [('mixed_order_subnormal_and_zero', ids, g)]
    generator = torch.Generator().manual_seed(42)
    for index in range(24):
        random_ids = torch.randint(vocab, (3, 8), generator=generator)
        if index % 4 == 0:
            random_ids.fill_(index % vocab)
        random_g = (torch.randn(3, 8, width, generator=generator) * 2**(index % 16 - 8)).bfloat16()
        cases.append((f'random_{index}', random_ids, random_g))
    case_results = []
    for name, source_ids, tangent in cases:
        original_ids = source_ids.clone()
        meta = metadata_check(source_ids, vocab, 4)
        w = torch.randn(vocab, width, requires_grad=True)
        reference_w = w.detach().clone().requires_grad_()
        y = lookup(w, *meta)
        expected_y = F.embedding(original_ids, reference_w).bfloat16()
        exact(y, expected_y, name + ' forward')
        source_ids.fill_(1)
        y.backward(tangent)
        expected_y.backward(tangent)
        expected = sequential_reference(original_ids, tangent, vocab)
        exact(w.grad, expected, name + ' sequential backward')
        exact(reference_w.grad, expected, name + ' Torch backward')
        assert torch.isfinite(w.grad).all()
        case_results.append({'name': name, 'status': 'BIT_EXACT_FINITE'})
    return {'cases': case_results, 'first_witness_token_indices': zero_tokens,
            'coverage': ['noncontiguous IDs/tangents', 'duplicate order', 'both subnormal signs',
                         'cancellation', 'signed zeros', 'unused rows', 'all-one-row',
                         'source storage mutation before backward', 'one write per owner element']}


def delayed_lifetime_check():
    weights = [torch.randn(12, 10, requires_grad=True) for _ in range(2)]
    reference_weights = [w.detach().clone().requires_grad_() for w in weights]
    expected = [torch.zeros_like(w) for w in weights]
    storage = torch.zeros(3, 8, dtype=torch.int64)
    delayed = []
    reference_delayed = []
    # Both microsteps remain outstanding while the same loader storage changes.
    for step in range(2):
        storage.copy_((torch.arange(24).reshape(3, 8) * (step + 1) + step) % 12)
        old_ids = storage.clone()
        meta = metadata(storage, 12, 4)
        for table, (weight, ref_weight) in enumerate(zip(weights, reference_weights)):
            tangent = torch.randn(3, 8, 10).bfloat16()
            delayed.append((lookup(weight, *meta), tangent))
            reference_delayed.append((F.embedding(old_ids, ref_weight).bfloat16(), tangent))
            expected[table].add_(sequential_reference(old_ids, tangent, 12))
    storage.fill_(11)
    for y, g in delayed:
        y.backward(g)
    for y, g in reference_delayed:
        y.backward(g)
    for index, (weight, ref_weight) in enumerate(zip(weights, reference_weights)):
        exact(weight.grad, expected[index], 'delayed lifetime reference')
        exact(weight.grad, ref_weight.grad, 'delayed lifetime Torch')
    return {'status': 'BIT_EXACT', 'tables': 2, 'simultaneously_live_microsteps': 2,
            'backward_after_source_overwrite': True}


def tangent_boundary_check():
    class Capture:
        def __init__(self, fn):
            self.fn = fn
        def register_autograd(self, backward, setup_context):
            self.backward = backward
            self.setup = setup_context
    captured = build_lookup(kernel=lambda *args: (_ for _ in ()).throw(AssertionError('kernel should not run')),
                            nki_op_factory=lambda *args, **kwargs: Capture)
    try:
        captured.backward(types.SimpleNamespace(saved_tensors=()), torch.ones(2, 2, 10))
    except ValueError as exc:
        assert 'non-BF16' in str(exc)
        return {'status': 'FP32_CALLBACK_TANGENT_REJECTED', 'message': str(exc)}
    raise AssertionError('FP32 callback tangent was accepted')


def full_source_imports():
    # Model-only tests never call any of these excluded data/evaluator symbols.
    prepare = types.ModuleType('prepare')
    def forbidden(*args, **kwargs):
        raise AssertionError('Data/evaluator placeholder was called')
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'prepare':
            for alias in node.names:
                setattr(prepare, alias.name, forbidden)
    for name, value in {'TOKENIZER_VOCAB_SIZE': 8192, 'EVAL_TIMEOUT_SECONDS': 1,
                        'TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS': 1,
                        'TRAIN_STARTUP_STEPS_EXCLUDED': 5,
                        'TRAIN_TIME_BUDGET_SECONDS': 1800}.items():
        setattr(prepare, name, value)
    prepare.print0 = lambda *args, **kwargs: None
    arrow, parquet = types.ModuleType('pyarrow'), types.ModuleType('pyarrow.parquet')
    arrow.parquet = parquet
    sys.modules.update({'prepare': prepare, 'pyarrow': arrow, 'pyarrow.parquet': parquet})
    modules = []
    for name, path in [('independent_base', ROOT/'base.py'), ('independent_candidate', ROOT/'train.py')]:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def model_checks():
    base, candidate = full_source_imports()
    assert candidate.K2_EMBEDDING is False
    models = []
    for module in (base, candidate):
        torch.manual_seed(42)
        model = module.GPT(module.GPTConfig(sequence_len=8, vocab_size=64, n_layer=8,
                                           n_head=2, n_embd=64, mlp_ratio=2))
        model.init_weights()
        models.append(model)
    parameter_objects = [[id(p) for p in model.parameters()] for model in models]
    opts = [torch.optim.SGD(model.parameters(), lr=.01) for model in models]
    calls = []
    def model_metadata(ids, vocab):
        calls.append(1)
        return metadata(ids, vocab, 4)
    def forbidden(*args, **kwargs):
        raise AssertionError('OFF/evaluation used K2 helpers')
    candidate._K2_METADATA = candidate._K2_LOOKUP = forbidden
    losses = []
    for step in range(6):
        if step == 3:
            candidate.K2_EMBEDDING = True
            candidate._K2_METADATA, candidate._K2_LOOKUP = model_metadata, lookup
        torch.manual_seed(100 + step)
        ids, targets = torch.randint(64, (2, 8)), torch.randint(64, (2, 8))
        outputs = []
        for model, optimizer in zip(models, opts):
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, targets)
            loss.backward()
            outputs.append(loss)
        exact(*outputs, 'model loss')
        left, right = dict(models[0].named_parameters()), dict(models[1].named_parameters())
        assert left.keys() == right.keys()
        for name in left:
            if left[name].grad is None:
                assert right[name].grad is None
            else:
                exact(left[name].grad, right[name].grad, 'gradient ' + name)
        for optimizer in opts:
            optimizer.step()
        for name, value in models[0].state_dict().items():
            exact(value, models[1].state_dict()[name], 'state ' + name)
        losses.append(float(outputs[0].detach()))
    assert len(calls) == 3
    assert parameter_objects == [[id(p) for p in model.parameters()] for model in models]
    candidate._K2_METADATA = candidate._K2_LOOKUP = forbidden
    # Independently cover all ordinary no-grad/eval routes, not just both at once.
    for training, grad_enabled in [(False, True), (False, False), (True, False)]:
        for model in models:
            model.train(training)
        with torch.set_grad_enabled(grad_enabled):
            exact(models[0](ids), models[1](ids), 'ordinary eval/no-grad')
    return {'off': 'THREE_STEPS_BIT_EXACT', 'off_losses': losses[:3],
            'on': 'THREE_STEPS_BIT_EXACT_INTERPRETED_CPU', 'on_losses': losses[3:],
            'geometry': '8 layers, width64, context8, batch2', 'optimizer': 'SGD lr=.01',
            'on_metadata_calls': len(calls), 'eval_and_no_grad_routes': 'BIT_EXACT_K2_HELPERS_UNCALLED',
            'same_parameter_objects': True,
            'import_placeholders': 'review-local unused prepare symbols and pyarrow.parquet; no data/evaluator calls'}


def source_checks():
    base_lines = (ROOT/'base.py').read_text().splitlines(keepends=True)
    candidate_lines = (ROOT/'train.py').read_text().splitlines(keepends=True)
    actual = list(difflib.unified_diff(base_lines, candidate_lines))[2:]
    frozen = (ROOT/'SOURCE.diff').read_text().splitlines(keepends=True)[4:]
    # Git adds the containing function to each hunk header; line counts must agree.
    def normalized(lines):
        return [line[:line.rfind('@@')+2] + '\n' if line.startswith('@@') else line for line in lines]
    assert normalized(actual) == normalized(frozen), 'SOURCE.diff must describe exact pair'
    receipt = json.loads((ROOT/'cpu-result.json').read_text())
    assert len(receipt['operation']) == 16
    finite = [case for case in receipt['operation'] if case['verdict'] == 'EXACT_FINITE']
    assert len(finite) == 12
    assert all(case['actual_bits'] == case['expected_bits'] for case in finite)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
              [ROOT/name for name in ('base.py', 'train.py', 'SOURCE.diff', 'SPEC.md',
                                     'cpu-result.json', 'test_k2.py', 'cpu_instruction_model.py')]}
    assert hashes['base.py'] == 'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43'
    assert hashes['train.py'] == '63925519ac6c0aab1a1d598e96fef3f4413fbd083d9b6d177901a2e972060db5'
    return {'source_diff': 'EXACT', 'frozen_hashes': hashes,
            'author_receipt': {'finite_bit_exact': len(finite), 'nonfinite_diagnostics': 4,
                               'source_hashes_match': all(hashes[p] in receipt['hashes'].values() for p in ('base.py','train.py'))}}


if __name__ == '__main__':
    results = {'scope': 'SOURCE_CPU_ONLY_FOR_SHORT_NATIVE_TEST', 'source': source_checks(),
               'operations': operation_checks(), 'lifetime': delayed_lifetime_check(),
               'tangent_boundary': tangent_boundary_check(), 'model': model_checks(),
               'kernel_calls': len(kernel_calls), 'torch': torch.__version__, 'native_compiled': False,
               'reference_test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(results, indent=2))
