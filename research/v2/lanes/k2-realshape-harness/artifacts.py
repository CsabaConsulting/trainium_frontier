"""Lossless raw content-addressed storage with full retained-blob checks."""
import hashlib
import os
from pathlib import Path
import tempfile


class Blobs:
    def __init__(self, root, ceiling=2 * 1024**3):
        self.root, self.ceiling = Path(root), ceiling
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, chunks):
        fd, name = tempfile.mkstemp(prefix='pending-', dir=self.root)
        path = Path(name)
        digest, size = hashlib.sha256(), 0
        with os.fdopen(fd, 'wb') as stream:
            for chunk in chunks:
                size += len(chunk)
                stream.flush()
                # Include pending files in the hard on-disk cap.
                if sum(p.stat().st_size for p in self.root.iterdir() if p.is_file()) + len(chunk) > self.ceiling:
                    raise RuntimeError('ARTIFACT_CEILING')
                stream.write(chunk)
                digest.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        key = digest.hexdigest()
        target = self.root / key
        if target.exists():
            self.check(dict(sha256=key, bytes=size))
            # Only our just-created, complete duplicate scratch file is removed.
            path.unlink()
        else:
            path.rename(target)
        return dict(sha256=key, bytes=size, encoding='raw-little-endian', compressed=False)

    def check(self, reference):
        key = reference.get('sha256', '')
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('BLOB_KEY')
        path = self.root / key
        if not path.is_file() or path.stat().st_size != reference.get('bytes'):
            raise ValueError('MISSING_OR_WRONG_BLOB')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                digest.update(chunk)
        if digest.hexdigest() != key:
            raise ValueError('CORRUPT_BLOB')
        return path


def compare_fp32(actual_path, expected_path, elements, *, exceptional=False):
    """Every raw coordinate inspected; no digest-only or equal-NaN pass."""
    import numpy as np
    actual_path, expected_path = Path(actual_path), Path(expected_path)
    if actual_path.stat().st_size != elements * 4 or expected_path.stat().st_size != elements * 4:
        raise ValueError('OUTPUT_BYTES')
    mismatch, nonfinite, payload_differences, witnesses = 0, 0, 0, []
    offset = 0
    with actual_path.open('rb') as aa, expected_path.open('rb') as ee:
        while offset < elements:
            length = min(16384, elements - offset)
            a = np.frombuffer(aa.read(length * 4), dtype='<u4')
            e = np.frombuffer(ee.read(length * 4), dtype='<u4')
            am, em = a & 0x7fffffff, e & 0x7fffffff
            an, en = am > 0x7f800000, em > 0x7f800000
            ai, ei = am == 0x7f800000, em == 0x7f800000
            nf = (am >= 0x7f800000) | (em >= 0x7f800000)
            nonfinite += int(nf.sum())
            payload_differences += int((an & en & (a != e)).sum())
            wrong = (a != e) if not exceptional else ((~nf & (a != e)) | (an != en) | (ai != ei) | (ai & ei & (a != e)))
            mismatch += int(wrong.sum())
            for j in np.flatnonzero(wrong)[:max(0, 16 - len(witnesses))]:
                witnesses.append(dict(index=offset + int(j), actual=f'{int(a[j]):08x}', expected=f'{int(e[j]):08x}'))
            offset += length
    verdict = ('FAIL_MISMATCH' if mismatch else 'FAIL_NONFINITE_DIAGNOSTIC' if nonfinite else 'EXACT_FINITE')
    return dict(verdict=verdict, mismatches=mismatch, nonfinite_coordinates=nonfinite,
                nan_payload_differences=payload_differences, witnesses=witnesses, compared_elements=elements)


def compare_record_bytes(record, store, expected, *, exceptional=False):
    """Byte comparison only: does NOT authenticate a native invocation or pass a gate."""
    required = {'rank', 'route', 'fixture', 'dtype', 'shape', 'actual_blob'}
    if not isinstance(record, dict) or not required <= record.keys():
        raise ValueError('MISSING_RECORD_FIELD')
    if type(record['rank']) is not int or record['rank'] not in range(4) or record['dtype'] != 'float32' or record['shape'] != [8192, 1024]:
        raise ValueError('OUTPUT_IDENTITY_OR_DOMAIN')
    if record['route'] not in ('direct', 'ordinary_eager', 'ordinary_compiled') or record['fixture'] not in ('F0', 'F1', 'F2', 'F3', 'F4', 'F5'):
        raise ValueError('ROUTE_OR_FIXTURE')
    return compare_fp32(store.check(record['actual_blob']), store.check(expected), 8192 * 1024, exceptional=exceptional)
