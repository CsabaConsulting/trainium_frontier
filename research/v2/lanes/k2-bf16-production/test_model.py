"""Implementer tests only. Tiny CPU tensors, no model/native/backend import."""
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import sys

import torch
from prototype import metadata_model, consume_model, production_cost, build_native


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def bits(tensor):
    return [f'{int(x) & 0xffffffff:08x}' for x in tensor.contiguous().view(torch.int32).flatten()]


def bf16(values):
    return torch.tensor([x if x < 32768 else x - 65536 for x in values], dtype=torch.int16).view(torch.bfloat16).reshape(-1, 1)


def expected_cpu(ids, gradient, vocab):
    out = torch.zeros((vocab, gradient.shape[1]), dtype=torch.float32)
    for token, row in enumerate(ids):
        out[row] += gradient[token].float()
    return out


def main():
    cases = {
        'minimum_positive': [0x0001], 'minimum_negative': [0x8001],
        'normal_cancel_positive': [0x0081, 0x8080],
        'normal_cancel_negative': [0x8081, 0x0080],
        'large_order': [0x4b80, 0x3f80, 0xcb80],
        'small_order': [0x3f80, 0x3380, 0xbf80],
        'large_cancel_then_subnormal': [0x4b80, 0xcb80, 0x0001],
        'zero_signs': [0x8000, 0x0000, 0x8000],
        'positive_overflow': [0x7f7f, 0x7f7f],
        'negative_overflow': [0xff7f, 0xff7f],
        'inf_cancel': [0x7f80, 0xff80], 'nan_payload': [0x7fc1],
    }
    records = []
    for name, values in cases.items():
        ids = [3] * len(values)
        gradient = bf16(values)
        meta = metadata_model(ids, 8, tile=1)
        actual = consume_model(gradient, meta)
        expected = expected_cpu(ids, gradient, 8)
        finite = bool(torch.isfinite(actual).all())
        actual_bits, expected_bits = bits(actual), bits(expected)
        if finite:
            require(actual_bits == expected_bits, 'FINITE_RAW_MISMATCH:' + name)
        else:
            # Record every bit, never turn NaN equality into a passing gate.
            require(torch.equal(torch.isnan(actual), torch.isnan(expected)), 'NAN_CLASS')
            require(torch.equal(torch.isinf(actual), torch.isinf(expected)), 'INF_CLASS')
            require(torch.equal(torch.signbit(actual[torch.isinf(actual)]),
                                torch.signbit(expected[torch.isinf(expected)])), 'INF_SIGN')
        records.append(dict(case=name, inputs_bf16=[f'{x:04x}' for x in values],
                            actual=actual_bits, expected=expected_bits,
                            verdict='EXACT_FINITE' if finite else 'FAIL_NONFINITE_DIAGNOSTIC'))
    require(records[2]['actual'][3] == '00010000', 'POSITIVE_SUBNORMAL')
    require(records[3]['actual'][3] == '80010000', 'NEGATIVE_SUBNORMAL')

    routing = []
    for ids in ([], [0, 7, 0, 7, 3, 0, 3, 7, 0], [3] * 257,
                list(range(8)) * 8, [7, 0, 7, 0, 7, 0, 1]):
        meta = metadata_model(ids, 8, tile=4)
        gradient = torch.arange(len(ids) * 2, dtype=torch.float32).reshape(-1, 2).to(torch.bfloat16)
        actual = consume_model(gradient, meta)
        require(bits(actual) == bits(expected_cpu(ids, gradient, 8)), 'ROUTING')
        routing.append(dataclasses.asdict(meta))

    loader_ids = [0, 1, 0]
    first = metadata_model(loader_ids, 8, tile=1)
    loader_ids[:] = [7, 6, 7]
    second = metadata_model(loader_ids, 8, tile=1)
    require(first.ids == (0, 1, 0) and second.ids == (7, 6, 7), 'SNAPSHOT_ALIAS')
    require(first.positions != second.positions, 'REPLAY_METADATA_STALE')
    gradient = bf16([0x3f80, 0x4000, 0x4040])
    require(bits(consume_model(gradient, first)) != bits(consume_model(gradient, second)), 'REPLAY_GRADIENT_STALE')
    # WTE and VE7 share the same immutable routing, but return separate dense grads.
    wte = consume_model(gradient, first)
    ve = consume_model(bf16([0x4080, 0x40a0, 0x40c0]), first)
    require(wte.data_ptr() != ve.data_ptr() and bits(wte) != bits(ve), 'TABLE_ALIAS')
    # Existing accumulation boundaries matter: concatenation would change 1 to 0.
    one = consume_model(bf16([0x4b80]), metadata_model([0], 8))
    two = consume_model(bf16([0xcb80, 0x3f80]), metadata_model([0, 0], 8))
    accumulated = one + two
    concatenated = consume_model(bf16([0x4b80, 0xcb80, 0x3f80]), metadata_model([0, 0, 0], 8))
    require(bits(accumulated)[0] == '3f800000' and bits(concatenated)[0] == '3f800000', 'MICROSTEP_BASE')
    # A stronger pair has lost unit in first call, recovered only if concatenated.
    a = consume_model(bf16([0x4b80]), metadata_model([0], 8))
    b = consume_model(bf16([0x3f80, 0xcb80]), metadata_model([0, 0], 8))
    require(bits(a + b)[0] != bits(consume_model(bf16([0x4b80, 0x3f80, 0xcb80]), metadata_model([0, 0, 0], 8)))[0], 'MICROSTEP_WITNESS')

    rejected = []
    meta = metadata_model([0, 1, 0], 8)
    for name, malformed in (
        ('wrong_positions', dataclasses.replace(meta, positions=(2, 0, 1))),
        ('wrong_owner', dataclasses.replace(meta, positions=(0, 1, 2))),
        ('wrong_count', dataclasses.replace(meta, counts=(3,) + meta.counts[1:])),
        ('wrong_maximum', dataclasses.replace(meta, maximums=(1, 0))),
        ('negative_start', dataclasses.replace(meta, starts=(-1,) + meta.starts[1:])),
    ):
        try:
            consume_model(gradient, malformed)
        except ValueError as exc:
            rejected.append(dict(case=name, error=str(exc)))
        else:
            raise RuntimeError('FALSE_ACCEPTANCE:' + name)
    for value in (-1, 8, 2**32, True):
        try:
            metadata_model([value], 8)
        except ValueError as exc:
            rejected.append(dict(case='invalid_id:' + str(value), error=str(exc)))
        else:
            raise RuntimeError('INVALID_ID_ACCEPTED')
    for enabled in (False, True):
        try:
            build_native(enabled=enabled)
        except RuntimeError as exc:
            rejected.append(dict(case='native_enabled:' + str(enabled), error=str(exc)))
        else:
            raise RuntimeError('NATIVE_MUST_BE_CLOSED')
    root = pathlib.Path(__file__).resolve().parent
    source = root.parents[3] / 'workflow/proposals/research-v2/train.py'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    require(source_hash == '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb', 'FROZEN_SOURCE_CHANGED')
    result = dict(status='IMPLEMENTER_TINY_MODEL_ONLY_NATIVE_INCOMPLETE', source_sha256=source_hash,
                  python=sys.version, torch=torch.__version__, scalar_cases=records,
                  routing=routing, rejections=rejected, snapshot_replay=True,
                  per_call_accumulation_witness=True, two_tables_independent=True,
                  cost=production_cost(), native_modules={name: importlib.util.find_spec(name) is not None
                     for name in ('nki', 'neuronxcc', 'torch_neuronx')},
                  max_input_tensor_elements=514)
    path = root / ('raw-optimized.json' if sys.flags.optimize else 'raw-normal.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], scalar_cases=len(records), routing_cases=len(routing),
                          rejections=len(rejected), receipt=str(path))))


if __name__ == '__main__':
    main()
