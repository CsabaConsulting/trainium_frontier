"""Bounded independent CPU reference check; consumes only supplied diff."""
import hashlib
from pathlib import Path
import torch

root = Path(__file__).resolve().parent
diff = (root / 'candidate.diff').read_text()
lines = diff.splitlines()
body = []
copying = False
for line in lines:
    if line == ' def _k2_prepare_metadata(ids, vocab=8192, group=128):':
        copying = True
    elif copying and line.startswith(' def _k2_build_registered_lookup'):
        break
    if copying and line[:1] in (' ', '+'):
        body.append(line[1:])
assert body and body[0].startswith('def _k2_prepare_metadata')
scope = {'torch': torch}
exec('\n'.join(body), scope)
metadata = scope['_k2_prepare_metadata']
torch.manual_seed(42)
vocab, channels = 8192, 7
cases = [
    [[8191, 0, 8191, 13], [0, 13, 13, 1]],
    [[0, 0, 0], [0, 0, 0]],
    [[8191, 8191], [8191, 8191]],
    [[3, 2, 1, 0], [8191, 8190, 8189, 8188]],
]
weight = torch.randn(vocab, channels, dtype=torch.float32).to(torch.bfloat16)
comparisons = 0
for case_no, values in enumerate(cases):
    flat = [x for row in values for x in row]
    by_row = [[] for _ in range(vocab)]
    for pos, row in enumerate(flat):
        by_row[row].append(pos)
    expected_positions = [pos for positions in by_row for pos in positions]
    expected_counts = [len(positions) for positions in by_row]
    expected_starts = []
    running = 0
    for count in expected_counts:
        expected_starts.append(running)
        running += count
    expected_max = [max(expected_counts[i:i+128]) for i in range(0, vocab, 128)]
    tangent = torch.randn(len(flat), channels).to(torch.bfloat16)
    expected_grad = torch.zeros(vocab, channels, dtype=torch.float32)
    for pos, row in enumerate(flat):
        expected_grad[row] = expected_grad[row] + tangent[pos].float()
    results = []
    for dtype in (torch.int32, torch.int64):
        ids = torch.tensor(values, dtype=dtype)
        result = metadata(ids)
        snapshot, counts, starts, positions, maxima = result
        assert snapshot.dtype == dtype and snapshot.data_ptr() != ids.data_ptr()
        for tensor, expected in zip(result[1:], (expected_counts, expected_starts, expected_positions, expected_max)):
            assert tensor.dtype == torch.int32
            assert tensor.flatten().tolist() == expected
            comparisons += 1
        actual_gather = torch.nn.functional.embedding(snapshot, weight).to(torch.bfloat16)
        expected_gather = torch.stack([weight[row] for row in flat]).reshape(*ids.shape, channels)
        assert torch.equal(actual_gather, expected_gather)
        actual_grad = torch.zeros_like(expected_grad)
        for row in range(vocab):
            start, count = int(starts[row, 0]), int(counts[row, 0])
            for offset in range(count):
                pos = int(positions[start + offset, 0])
                actual_grad[row] = actual_grad[row] + tangent[pos].float()
        assert torch.equal(actual_grad, expected_grad)
        assert torch.isfinite(actual_grad).all()
        frozen = tuple(x.clone() for x in result)
        ids.fill_(77)
        assert all(torch.equal(a, b) for a, b in zip(result, frozen))
        assert torch.equal(torch.nn.functional.embedding(snapshot, weight), expected_gather)
        results.append(result)
    assert all(torch.equal(a, b) for a, b in zip(results[0], results[1]))
    print(f'case={case_no} tokens={len(flat)} int32/int64 metadata, stable reference, gather, sequential FP32 gradient, snapshot isolation: PASS')

rejected = 0
for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.bool, torch.int8, torch.uint8, torch.int16):
    try:
        metadata(torch.zeros((2, 3), dtype=dtype))
    except ValueError as exc:
        assert str(exc) == 'K2 requires two-dimensional int32 or int64 token IDs'
        rejected += 1
    else:
        raise AssertionError(f'accepted invalid dtype {dtype}')
for dtype in (torch.int32, torch.int64):
    for shape in ((), (3,), (1, 2, 3)):
        try:
            metadata(torch.zeros(shape, dtype=dtype))
        except ValueError as exc:
            assert str(exc) == 'K2 requires two-dimensional int32 or int64 token IDs'
            rejected += 1
        else:
            raise AssertionError(f'accepted invalid dimension {shape}')
print(f'invalid dtype/dimension rejections={rejected}: PASS')
print(f'exact metadata/reference comparisons={comparisons}: PASS')
print('torch_version=' + torch.__version__)
print('candidate_diff_sha256=' + hashlib.sha256((root / 'candidate.diff').read_bytes()).hexdigest())
print('independent_check_sha256=' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
print('RESULT: PASS_CPU_BOUNDED_REFERENCE; native execution/lowering/timing not tested; no range-check claim')
