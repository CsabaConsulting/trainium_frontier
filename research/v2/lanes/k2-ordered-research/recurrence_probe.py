"""Tiny Researcher-owned CPU arithmetic probe, not a candidate or NKI emulator.

No train.py import, distributed job, device access, external data, or tolerance.
This tests the proposed ordering, not native API/compiler/rounding guarantees.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def digest(a):
    return hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def token_order(ids, grad, vocabulary):
    out = torch.zeros(vocabulary, grad.shape[-1], dtype=torch.float32)
    for row, value in zip(ids.reshape(-1).tolist(), grad.reshape(-1, grad.shape[-1]).float()):
        out[row] = out[row] + value
    return out


def ordered_waves(ids, grad, vocabulary):
    """Padded unique-address gather/add/store; never coalesce gradients.

    Within each 128-token tile, rank each occurrence among identical IDs.
    Wave k processes exactly occurrence k of every ID. Inter-row reordering
    is harmless; intra-row ordering is unchanged. Dummy rows are private.
    """
    ids = ids.reshape(-1).long()
    grad = grad.reshape(-1, grad.shape[-1]).float()
    out = torch.zeros(vocabulary + 128, grad.shape[-1], dtype=torch.float32)
    rounds = []
    observed_order = [[] for _ in range(vocabulary)]
    for start in range(0, ids.numel(), 128):
        rows = ids[start:start + 128]
        values = grad[start:start + 128]
        pos = torch.arange(rows.numel())
        prior = pos[None, :] < pos[:, None]
        ordinal = ((rows[:, None] == rows[None, :]) & prior).sum(dim=1)
        count = int(ordinal.max()) + 1
        rounds.append(count)
        for wave in range(count):
            active = ordinal == wave
            addresses = torch.where(active, rows, vocabulary + pos)
            require(addresses.unique().numel() == addresses.numel(), 'nonunique wave address')
            # Do not multiply by zero: selection keeps inactive NaNs out too.
            selected = torch.where(active[:, None], values, torch.zeros_like(values))
            out[addresses] = out[addresses] + selected
            for j in pos[active].tolist():
                observed_order[int(rows[j])].append(start + j)
        expected_prefix = token_order(ids[:start + rows.numel()], grad[:start + rows.numel()], vocabulary)
        require(exact(out[:vocabulary], expected_prefix), 'tile-prefix byte mismatch')
    for row, order in enumerate(observed_order):
        expected = [j for j, value in enumerate(ids.tolist()) if value == row]
        require(order == expected, 'per-row order changed')
    require(torch.count_nonzero(out[vocabulary:]).item() == 0, 'dummy row changed')
    return out[:vocabulary], rounds


def tile_sums(ids, grad, vocabulary):
    """Rejected arithmetic: even sequential tile sums cannot repair R1."""
    out = torch.zeros(vocabulary, grad.shape[-1], dtype=torch.float32)
    ids, grad = ids.reshape(-1), grad.reshape(-1, grad.shape[-1])
    for start in range(0, ids.numel(), 128):
        out += token_order(ids[start:start + 128], grad[start:start + 128], vocabulary)
    return out


def cases():
    for name, big, small in [('review_large', 2.0 ** 24, 1.0), ('review_unit', 1.0, 2.0 ** -24)]:
        ids = torch.full((2, 128), 7, dtype=torch.int64)
        ids.reshape(-1)[[0, 128, 129]] = 3
        grad = torch.zeros(2, 128, 8, dtype=torch.bfloat16)
        grad.reshape(-1, 8)[0] = big
        grad.reshape(-1, 8)[128] = small
        grad.reshape(-1, 8)[129] = -big
        yield name, ids, grad, 256
    generator = torch.Generator().manual_seed(42)
    for name, ids in [
        ('unique', torch.arange(256)),
        ('same_id', torch.full((256,), 7)),
        ('interleaved_three_tiles', (torch.arange(384) * 7) % 11),
        ('random_ids', torch.randint(0, 256, (384,), generator=generator)),
        ('signed_zero', torch.tensor([0, 1, 0, 1] * 64)),
        ('bf16_subnormal', torch.tensor([0, 1, 0, 1] * 64)),
    ]:
        grad = torch.randn(ids.numel(), 8, generator=generator).bfloat16()
        if name == 'signed_zero':
            grad.zero_()
            grad[::2] = -0.0
        if name == 'bf16_subnormal':
            grad.fill_(2.0 ** -133)
            grad[1::3] = -(2.0 ** -133)
        yield name, ids, grad, 512
    ids = torch.arange(256).remainder(13).reshape(2, 128).T
    grad = torch.randn(2, 128, 8, generator=generator).bfloat16().transpose(0, 1)
    yield 'noncontiguous', ids, grad, 256
    yield 'empty', torch.empty(0, dtype=torch.int64), torch.empty(0, 8, dtype=torch.bfloat16), 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    records = []
    for name, ids, grad, vocabulary in cases():
        manual = token_order(ids, grad, vocabulary)
        waves, rounds = ordered_waves(ids, grad, vocabulary)
        weight = torch.zeros(vocabulary, grad.shape[-1], requires_grad=True)
        F.embedding(ids, weight).bfloat16().backward(grad)
        require(exact(waves, manual), name + ': manual mismatch')
        require(exact(waves, weight.grad), name + ': actual CPU dense mismatch')
        row = {'case': name, 'shape': list(grad.shape), 'vocabulary': vocabulary,
               'manual_exact': True, 'dense_cpu_exact': True, 'tile_prefix_exact': True,
               'per_row_order_exact': True, 'wave_rounds_per_tile': rounds,
               'gradient_sha256': digest(waves)}
        if name.startswith('review_'):
            grouped = tile_sums(ids, grad, vocabulary)
            require(not exact(grouped, manual), 'expected rejected regrouping counterexample disappeared')
            row.update(dense_row3=manual[3, 0].item(), ordered_row3=waves[3, 0].item(),
                       rejected_grouped_row3=grouped[3, 0].item())
        records.append(row)
    result = {'scope': 'Researcher CPU arithmetic feasibility only; no candidate implementation',
              'native_executed': False, 'candidate_status': 'e4d93 REFUSE unchanged',
              'python': platform.python_version(), 'torch': torch.__version__, 'threads': 1,
              'seed': 42, 'tolerance': 'identical dtype/shape/bytes; zero tolerance',
              'max_tokens': 384, 'max_channels': 8, 'max_vocabulary': 512,
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cases': records, 'checks_pass': True}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
