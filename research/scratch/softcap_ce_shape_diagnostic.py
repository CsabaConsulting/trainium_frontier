#!/usr/bin/env python3
"""Localize CE exactness differences by loop trip count and input precision.

Every128-row block is identical, so a physical-core-dependent difference can
be distinguished from a whole-kernel shape effect. General FP32 and BF16-valued
FP32 inputs use the same protected training primitives. This is a diagnostic;
it does not waive the failed exact-agreement gates or promote any kernel.
"""
import gc
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
from softcap_ce_parallel_nki import build_kernels


def emit(kind, record):
    print(kind + ' ' + json.dumps(record), flush=True)


def difference(actual, reference):
    actual, reference = actual.double(), reference.double()
    absolute = (actual - reference).abs()
    row_error = absolute.reshape(actual.shape[0], -1).max(dim=1).values
    return dict(exact=bool(torch.equal(actual, reference)),
                finite=bool(torch.isfinite(actual).all()),
                max_abs=float(absolute.max()),
                relative_rms=float(absolute.square().mean().sqrt() /
                                   reference.square().mean().sqrt().clamp_min(1e-30)),
                changed_rows=int((row_error != 0).sum()),
                first_changed_rows=torch.nonzero(row_error != 0).flatten()[:16].tolist(),
                max_abs_by_128_rows=[float(block.max()) for block in row_error.split(128)],
                repeated_blocks_exact=all(torch.equal(block, actual[:128])
                                          for block in actual.split(128)))


def main():
    import torch_neuronx
    torch.set_num_threads(4)
    forward, backward = (torch_neuronx.wrap_nki(k) for k in build_kernels())
    variants = [('mask_one_core', 1, False), ('gather_one_core', 1, True),
                ('mask_two_cores', 2, False), ('gather_two_cores', 2, True)]
    generator = torch.Generator().manual_seed(9471)
    raw = torch.randn(128, 8192, generator=generator) * 6
    targets = torch.randint(8192, (128, 1), generator=generator)
    targets[:4, 0] = torch.tensor([0, 4095, 4096, 8191])
    gradients = torch.randn(128, 1, generator=generator) * .5
    emit('CE_SHAPE_CONTRACT', dict(repeated_block_rows=128,
         rows=[256, 512, 768, 1024], representations=['fp32', 'bf16_values_in_fp32'],
         diagnostic_only=True, precision_thresholds_changed=False,
         stages=['loss', 'lse', 'dz_using_native_lse', 'dz_using_own_lse']))
    with torch.no_grad():
        for case in ('random', 'uniform', 'confident_correct'):
            block = raw.clone()
            if case == 'uniform':
                block.fill_(.7)
            elif case == 'confident_correct':
                block.zero_()
                block.scatter_(1, targets, 500.0)
            for representation in ('fp32', 'bf16_values_in_fp32'):
                base = block if representation == 'fp32' else block.bfloat16().float()
                canonical = None
                for rows in (256, 512, 768, 1024):
                    repeats = rows // 128
                    z = base.repeat(repeats, 1).to('neuron')
                    labels = targets.float().repeat(repeats, 1).to('neuron')
                    g = gradients.repeat(repeats, 1).to('neuron')
                    native_loss, native_lse = T._softcap_ce_fwd(z, labels)
                    native_dz = T._softcap_ce_bwd(z, labels, native_lse, g)
                    native = tuple(a.cpu() for a in (native_loss, native_lse, native_dz))
                    if canonical is None:
                        canonical = tuple(a[:128].clone() for a in native)
                    emit('CE_SHAPE_NATIVE', dict(case=case, representation=representation,
                         rows=rows, compared_with_first_shape_repeated_block=[
                             difference(a, b.repeat(repeats, 1)) for a, b in zip(native, canonical)]))
                    for name, lanes, gather in variants:
                        loss, lse = forward[lanes](z, labels, lanes, gather)
                        own_dz = backward[lanes](z, labels, lse, g, lanes)
                        fixed_dz = backward[lanes](z, labels, native_lse, g, lanes)
                        values = tuple(a.cpu() for a in (loss, lse, fixed_dz, own_dz))
                        comparisons = [difference(a, b) for a, b in
                                       zip(values, (*native[:2], native[2], native[2]))]
                        emit('CE_SHAPE_VARIANT', dict(case=case,
                             representation=representation, rows=rows,
                             outer_loop_trip_count=rows // (128 * lanes),
                             arm=name, comparisons=comparisons))
                        repeated = forward[lanes](z, labels, lanes, gather)
                        assert all(torch.equal(a.cpu(), b) for a, b in zip(repeated, values[:2]))
                        assert all(r['finite'] for r in comparisons)
                    assert torch.equal(z.cpu(), base.repeat(repeats, 1))
                    del z, labels, g, native_loss, native_lse, native_dz, native, values
                    gc.collect()
    emit('CE_SHAPE_COMPLETE', dict(structural_checks_passed=True,
                                   promoted=False, hardware_core_difference_not_assumed=True))


if __name__ == '__main__':
    main()
