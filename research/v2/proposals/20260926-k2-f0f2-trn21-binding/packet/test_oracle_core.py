"""Implementer-only tiny checks. Not independent review or native evidence."""
import copy
import hashlib
import json
from pathlib import Path
import random
import struct
import tempfile
import numpy as np
from oracle import add32, recurrence, classify
from fixtures import token_id, tangent_bits, construct_metadata, validate_metadata
from artifacts import Blobs, compare_fp32, compare_record_bytes
from generate import OracleBudget


def require(condition, name):
    if not condition:
        raise RuntimeError(name)


def rejected(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise RuntimeError('NEGATIVE_NOT_REJECTED')


def main():
    rng = random.Random(42)
    pairs = [(0, 0x80000000), (0x80000000, 0x80000000),
             (0x00810000, 0x80800000), (0x80810000, 0x00800000),
             (0x3f800000, 0x33800000), (0x7f7f0000, 0x7f7f0000)]
    pairs += [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(20000)]
    # Independent scalar NumPy binary32 comparison, not the production oracle.
    for left, right in pairs:
        with np.errstate(all='ignore'):
            lhs = np.frombuffer(struct.pack('<I', left), dtype='<f4')[0]
            rhs = np.frombuffer(struct.pack('<I', right), dtype='<f4')[0]
            expected = struct.unpack('<I', np.float32(lhs + rhs).tobytes())[0]
        actual = add32(left, right)
        require(actual == expected or classify(actual) == classify(expected) == 'nan', 'ADD32')
    require(recurrence([0x4b80, 0x3f80, 0xcb80]) == 0, 'ORDER_A')
    require(recurrence([0x4b80, 0xcb80, 0x3f80]) == 0x3f800000, 'ORDER_B')
    require(recurrence([0x81, 0x8080]) == 0x10000, 'SUBNORMAL_POSITIVE')
    require(recurrence([0x8081, 0x80]) == 0x80010000, 'SUBNORMAL_NEGATIVE')
    require(recurrence([0x8000, 0x8000]) == 0, 'POSITIVE_ZERO_INITIALIZER')
    ids = [0, 3, 0, 1, 3, 0, 1, 2]
    metadata = construct_metadata(ids, 4, 2)
    require(validate_metadata(ids, metadata, 4, 2)[0] == [0, 2, 5], 'MEMBERSHIP')
    negatives = []
    for field, offset, value in ((0, 0, 4), (1, 1, 4), (2, 0, 2),
                                  (2, 0, 1), (2, 0, 8), (2, 0, -1), (3, 0, 99)):
        changed = copy.deepcopy(metadata)
        changed[field][offset] = value
        negatives.append(changed)
    swapped = copy.deepcopy(metadata)
    swapped[2][0], swapped[2][1] = swapped[2][1], swapped[2][0]
    negatives.append(swapped)
    for changed in negatives:
        rejected(lambda: validate_metadata(ids, changed, 4, 2))
    rejected(lambda: validate_metadata([1] + ids[1:], metadata, 4, 2))
    for bad in (-1, 4, True):
        rejected(lambda: construct_metadata([bad], 4, 2))
    # Scalar boundary recipe checks only; no full-size tensors generated.
    require(token_id('F1', 4095) == 8191 and token_id('F1', 4096) == 0, 'F1_ID')
    require(token_id('F2', 129) == 0 and token_id('F2', 130) == 127, 'F2_ID')
    require(token_id('F3', 32767) == 8191 and token_id('F4', 0) == 17, 'F34_ID')
    require(tangent_bits('F0', 32767, 1023) == 0x3bff, 'F0_CHANNEL')
    require(tangent_bits('F1', 4096, 0) == 0x3f00, 'F1_ROTATION')
    require(tangent_bits('F2', 1, 2) == 0x81, 'F2_SUBNORMAL')
    require(tangent_bits('F3', 3, 0) == 1, 'F3_SUBNORMAL')
    require(tangent_bits('F4', 24576, 0) == 1, 'F4_SUBNORMAL')
    require(tangent_bits('F5', 0, 0) == 0x7fc1, 'F5_PAYLOAD')
    # Keep tiny raw artifacts: no cleanup of experiment evidence.
    root = Path(tempfile.mkdtemp(prefix='k2-harness-core-', dir='/home/poojith-devan'))
    store = Blobs(root / 'blobs', ceiling=4096)
    def blob(words):
        return store.put([struct.pack('<' + 'I' * len(words), *words)])
    expected = blob([0, 0x3f800000, 0x10000, 0x80010000])
    result = compare_fp32(store.check(expected), store.check(expected), 4)
    require(result['verdict'] == 'EXACT_FINITE', 'EXACT')
    for words in ([0] * 4, [0, 0, 0x10000, 0x80010000],
                  [0, 0x3f800001, 0x10000, 0x80010000],
                  [0x80000000, 0x3f800000, 0x10000, 0x80010000]):
        require(compare_fp32(store.check(blob(words)), store.check(expected), 4)['verdict'] == 'FAIL_MISMATCH', 'COMPARE_NEGATIVE')
    nan_a, nan_b = blob([0x7fc10000]), blob([0x7fc00000])
    diagnostic = compare_fp32(store.check(nan_a), store.check(nan_b), 1, exceptional=True)
    require(diagnostic['verdict'] == 'FAIL_NONFINITE_DIAGNOSTIC' and diagnostic['nan_payload_differences'] == 1, 'NAN_DIAGNOSTIC')
    rejected(lambda: compare_record_bytes({}, store, expected))
    rejected(lambda: compare_fp32(store.check(expected), store.check(expected), 5))
    rejected(lambda: store.check(dict(sha256='0'*64, bytes=4)))
    budget = OracleBudget()
    budget.consume(1314816)
    rejected(lambda: budget.consume(1))
    rejected(lambda: Blobs(root / 'cap', 3).put([b'abcd']))
    print(json.dumps(dict(status='LOCAL_CORE_ONLY', seed=42, binary32_pair_checks=len(pairs),
                         metadata_negative_checks=12, largest_numeric_test_array_elements=4,
                         fullsize_generation_executed=False, native_executed=False,
                         raw_artifact_directory=str(root), dependencies=dict(numpy=np.__version__),
                         source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(Path(__file__).parent.glob('*.py'))}), sort_keys=True))


if __name__ == '__main__':
    main()
