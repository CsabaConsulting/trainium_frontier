"""Tiny numerical design falsifier only; no candidate/native/data imports."""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
V, C = 5, 4


def raw(x):
    return {'values': x.float().tolist(),
            'bf16_hex': [[f'{int(v) & 65535:04x}' for v in row]
                         for row in x.contiguous().view(torch.int16).tolist()]}


def same(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def mismatch(a, b):
    mask = a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)
    return mask.nonzero().tolist()


def sequential(rows, grad, dtype):
    result = torch.zeros(V, C, dtype=dtype)
    for row, value in zip(rows.tolist(), grad):
        result[row] = result[row] + value.to(dtype)
    return result.to(torch.bfloat16)


def fixture(name, sequences):
    steps = []
    for values in sequences:
        # Every target-row occurrence is separated by a duplicate second row.
        rows = torch.tensor([r for _ in values for r in (1, 3)], dtype=torch.int64)
        columns = torch.tensor([1., -1., 2., .5], dtype=torch.bfloat16)
        grads = []
        for i, value in enumerate(values):
            grads.extend([value*columns, (.25 if i % 2 == 0 else -.25)*columns])
        steps.append((rows, torch.stack(grads).to(torch.bfloat16)))
    table_index = torch.zeros(V, C, dtype=torch.bfloat16, requires_grad=True)
    table_embed = torch.zeros_like(table_index, requires_grad=True)
    accum_bf16 = torch.zeros_like(table_index)
    accum_fp32_per_call = torch.zeros_like(table_index)
    reports, all_rows, all_grad = [], [], []
    for number, (rows, grad) in enumerate(steps):
        # Independent parameters make per-call and accumulated references explicit.
        index_single = torch.zeros_like(table_index, requires_grad=True)
        embed_single = torch.zeros_like(table_embed, requires_grad=True)
        index_single[rows].backward(grad)
        F.embedding(rows, embed_single).backward(grad)
        table_index[rows].backward(grad)
        F.embedding(rows, table_embed).backward(grad)
        bf16 = sequential(rows, grad, torch.bfloat16)
        fp32 = sequential(rows, grad, torch.float32)
        # Autograd accumulates a BF16 contribution for each separate backward.
        accum_bf16 = (accum_bf16 + bf16).to(torch.bfloat16)
        accum_fp32_per_call = (accum_fp32_per_call + fp32).to(torch.bfloat16)
        comparisons = {}
        for operator, actual in [('advanced_indexing', index_single.grad),
                                 ('embedding', embed_single.grad)]:
            comparisons[operator] = {
                'matches_bf16_each_add': same(actual, bf16),
                'matches_fp32_cast_once': same(actual, fp32),
                'mismatch_vs_bf16': mismatch(actual, bf16),
                'mismatch_vs_fp32': mismatch(actual, fp32)}
        reports.append({'microstep': number, 'rows': rows.tolist(), 'upstream': raw(grad),
                        'advanced_indexing_single': raw(index_single.grad),
                        'embedding_single': raw(embed_single.grad),
                        'sequential_bf16': raw(bf16), 'sequential_fp32_cast_once': raw(fp32),
                        'comparisons': comparisons,
                        'accumulated_indexing': raw(table_index.grad),
                        'accumulated_embedding': raw(table_embed.grad)})
        all_rows.append(rows)
        all_grad.append(grad)
    joined_rows, joined_grad = torch.cat(all_rows), torch.cat(all_grad)
    continuous_bf16 = sequential(joined_rows, joined_grad, torch.bfloat16)
    global_fp32 = sequential(joined_rows, joined_grad, torch.float32)
    refs = {'per_call_bf16_then_bf16_accumulation': accum_bf16,
            'per_call_fp32_cast_once_then_bf16_accumulation': accum_fp32_per_call,
            'continuous_bf16_ignoring_microsteps': continuous_bf16,
            'global_fp32_cast_once_ignoring_microsteps': global_fp32}
    return {'name': name, 'steps': reports,
            'final_references': {k: raw(v) for k, v in refs.items()},
            'final_matches': {operator: {key: same(actual, ref) for key, ref in refs.items()}
                              for operator, actual in [('advanced_indexing', table_index.grad),
                                                       ('embedding', table_embed.grad)]}}


if __name__ == '__main__':
    cases = [fixture('single_rounding_witness', [[1., 2**-8, -1.]]),
             fixture('reverse_order_witness', [[1., -1., 2**-8]]),
             fixture('two_microsteps_split_before_cancellation', [[1., 2**-8], [-1.]]),
             fixture('two_microsteps_split_before_small_add', [[1.], [2**-8, -1.]]),
             fixture('two_microsteps_repeated_witness', [[1., 2**-8, -1.], [1., 2**-8, -1.]])]
    result = {'purpose': 'CPU_NUMERICAL_DESIGN_FALSIFIER_NOT_CANDIDATE_GATE',
              'torch': torch.__version__, 'threads': torch.get_num_threads(),
              'device': 'cpu', 'table_shape': [V, C], 'largest_tensor_elements': 48,
              'native_executed': False, 'cases': cases,
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(__file__).with_name('result.json').write_text(json.dumps(result, indent=2)+'\n')
    for case in cases:
        print(json.dumps({'name': case['name'],
                          'first_call': case['steps'][0]['comparisons'],
                          'final_matches': case['final_matches']}))
