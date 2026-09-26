"""Prepared only: two observed installed framework source files, no SDK imports."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import time

resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**2, 128 * 1024**2))
ROOT = Path('/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages/nki/framework')
FILES = ('kernel.py', 'compiled.py')
started = time.monotonic()
out = {'schema': 'two-observed-framework-source-read-v1',
       'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'status': 'UNKNOWN', 'files': [], 'read_bytes': 0,
       'no_sdk_imports': True, 'no_host_writes': True, 'no_device_or_cache_access': True}


def timeout(*_args):
    raise TimeoutError('TWO_SOURCE_READ_WALL_LIMIT')


def identity(s):
    return {'device': s.st_dev, 'inode': s.st_ino, 'size': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns, 'mode': s.st_mode}


signal.signal(signal.SIGALRM, timeout)
signal.alarm(14)
try:
    for name in FILES:
        p = ROOT / name
        for component in (p, *p.parents):
            if component.is_symlink():
                raise RuntimeError('SYMLINK_REFUSED:' + str(component))
        before = p.lstat()
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= 65536:
            raise RuntimeError('SIZE_OR_TYPE_REFUSED:' + name)
        expected_size = before.st_size
        if out['read_bytes'] + expected_size > 131072:
            raise RuntimeError('AGGREGATE_READ_BOUND')
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            if identity(os.fstat(fd)) != identity(before):
                raise RuntimeError('OPEN_RACE:' + name)
            raw = os.read(fd, expected_size + 1)
            after = os.fstat(fd)
            if identity(before) != identity(after) or identity(before) != identity(p.lstat()):
                raise RuntimeError('FILE_CHANGED:' + name)
        finally:
            os.close(fd)
        sha = hashlib.sha256(raw).hexdigest()
        if len(raw) != expected_size:
            raise RuntimeError('SHORT_OR_CHANGED_SOURCE_READ:' + name)
        out['read_bytes'] += len(raw)
        out['files'].append({'path': str(p), 'identity': identity(before),
                             'bytes': len(raw), 'sha256': sha, 'text': raw.decode('utf-8')})
    out['status'] = 'TWO_OBSERVED_SOURCES_RETRIEVED'
except Exception as e:
    out['status'] = 'REFUSED_OR_INCOMPLETE'
    out['error'] = type(e).__name__ + ': ' + str(e)
finally:
    signal.alarm(0)
    out['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out['elapsed_seconds'] = time.monotonic() - started
    encoded = json.dumps(out)
    if len(encoded.encode()) > 196608:
        encoded = json.dumps({'status': 'REFUSED_OR_INCOMPLETE', 'error': 'OUTPUT_192K_BOUND'})
    print(encoded)
