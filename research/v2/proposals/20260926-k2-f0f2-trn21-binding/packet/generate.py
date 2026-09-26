"""Streaming synthetic recipe/oracle preparation. Not executed at full size locally."""
import time
import numpy as np
from fixtures import V, T, C, FIXTURES, token_id, tangent_bits, construct_metadata, validate_metadata
from oracle import recurrence


def gradient_chunks(fixture):
    classes = np.arange(C) % 8
    for first in range(0, T, 128):
        if fixture == 'F0':
            data = np.broadcast_to(np.arange(0x3800, 0x3800 + C, dtype='<u2'), (128, C))
        else:
            motif = np.asarray([[tangent_bits(fixture, t, c) for c in range(8)]
                                for t in range(first, first + 128)], dtype='<u2')
            data = motif[:, classes]
        yield data.astype('<u2', copy=False).tobytes()


class OracleBudget:
    """One instance must span all six fixtures; never reset per fixture."""
    def __init__(self):
        self.additions = 0

    def consume(self, count):
        self.additions += count
        if self.additions > 1314816:
            raise RuntimeError('ORACLE_ADDITION_BOUND')


def prepare_fixture(fixture, store, deadline, budget):
    if fixture not in FIXTURES:
        raise ValueError('FIXTURE')
    def check_time():
        if time.monotonic() > deadline:
            raise RuntimeError('ORACLE_PHASE_DEADLINE')
    ids = [token_id(fixture, t) for t in range(T)]
    metadata = construct_metadata(ids)
    rows = validate_metadata(ids, metadata)
    ids_blob = store.put([np.asarray(ids, dtype='<i4').tobytes()])
    meta_blobs = [store.put([np.asarray(values, dtype='<i4').tobytes()]) for values in metadata]
    def bounded_chunks():
        for chunk in gradient_chunks(fixture):
            check_time()
            yield chunk
    tangent_blob = store.put(bounded_chunks())
    tangent_path = store.check(tangent_blob)
    # Validate actual retained full bytes against the recipe, including every
    # channel's class membership. No class assumption is accepted on hash alone.
    with tangent_path.open('rb') as stream:
        for expected_chunk in gradient_chunks(fixture):
            check_time()
            if stream.read(len(expected_chunk)) != expected_chunk:
                raise ValueError('RAW_TIMELINE_CLASS_EXPANSION')
        if stream.read(1):
            raise ValueError('EXCESS_TANGENT_BYTES')
    raw = np.memmap(tangent_path, mode='r', dtype='<u2', shape=(T, C))
    cache, additions, traces = {}, 0, []
    def calculate(sequence):
        nonlocal additions
        if sequence not in cache:
            budget.consume(len(sequence))
            additions += len(sequence)
            cache[sequence] = recurrence(sequence)
        return cache[sequence]
    def expected_chunks():
        if fixture == 'F0':
            # All row memberships independently checked; retained gradient
            # validation proves each channel's entire timeline is constant.
            if any(len(row) != 4 for row in rows):
                raise ValueError('F0_WHOLE_TIMELINE')
            result = np.asarray([calculate(tuple(int(raw[t, c]) for t in rows[0]))
                                 for c in range(C)], dtype='<u4').tobytes()
            for row in range(V):
                check_time()
                yield result
        else:
            for row, membership in enumerate(rows):
                check_time()
                representative = [calculate(tuple(int(raw[t, c]) for t in membership)) for c in range(8)]
                if row in (0, 127, 128, 4095, 4096, 8191):
                    traces.append(dict(row=row, token_positions=membership,
                                       final_bits=[f'{word:08x}' for word in representative]))
                yield np.asarray([representative[c % 8] for c in range(C)], dtype='<u4').tobytes()
    expected_blob = store.put(expected_chunks())
    del raw
    return dict(fixture=fixture, ids=ids_blob, metadata=meta_blobs, tangent=tangent_blob,
                expected=expected_blob, whole_timeline_classes_validated=True,
                oracle_additions=additions, oracle_unique_timelines=len(cache), witnesses=traces)
