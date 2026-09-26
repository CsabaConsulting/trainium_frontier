"""Fixed F0/F2 packet constants, framing and evidence utilities; no native imports."""
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time

HERE = Path(__file__).resolve().parent
LABEL = 'fs-20260926-k2-realshape-f0f2-partial-r1'
CONSUMER_SHA = '9425390c96053a1b24d4addf8b61b1a87c4993f3551f08c7cdcefcf142d84dcd'
ROUTES = (('F0', 'direct'), ('F0', 'repeat'), ('F0', 'eager'), ('F0', 'compiled'),
          ('F2', 'direct'), ('F2', 'eager'), ('F2', 'compiled'))
MIB, GIB = 1024**2, 1024**3
UNKNOWN = {k: None for k in ('hbm_exact_peak_bytes', 'sbuf_allocated_peak_bytes',
    'sbuf_spill_bytes', 'compiler_separate_peak_rss_bytes', 'worker_separate_peak_rss_bytes',
    'complete_graph_count', 'hardware_owner_execution_trace')}
UNKNOWN['reason'] = 'No evidenced frozen-SDK binding; sampled guards and full-byte behavior are not precision traces.'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(MIB), b''):
            h.update(block)
    return h.hexdigest()


def source_pins():
    pins = json.loads((HERE / 'SOURCE_PINS.json').read_text())
    require({p.name for p in HERE.glob('*.py')} == set(pins), 'UNEXPECTED_OR_MISSING_PYTHON_SOURCE')
    for name, sha in pins.items():
        require(Path(name).name == name and not (HERE / name).is_symlink(), 'UNSAFE_SOURCE')
        require(digest(HERE / name) == sha, 'SOURCE_SHA:' + name)
    require(pins['consumer_nki.py'] == CONSUMER_SHA, 'CONSUMER_PIN')
    return pins


def identity(pid):
    text = Path('/proc/%d/stat' % pid).read_text().rsplit(') ', 1)[1].split()
    return dict(pid=pid, state=text[0], ppid=int(text[1]), pgid=int(text[2]),
                sid=int(text[3]), start=int(text[19]))


def send(sock, value):
    raw = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
    require(len(raw) <= MIB, 'MESSAGE_CAP')
    sock.sendall(struct.pack('!I', len(raw)) + raw)


def receive_bytes(sock, size):
    data = bytearray()
    while len(data) < size:
        block = sock.recv(min(MIB, size - len(data)))
        require(block, 'WORKER_CHANNEL_LOST')
        data.extend(block)
    return bytes(data)


def receive(sock):
    length, = struct.unpack('!I', receive_bytes(sock, 4))
    require(length <= MIB, 'MESSAGE_CAP')
    return json.loads(receive_bytes(sock, length))


def peer(sock):
    pid, uid, gid = struct.unpack('3i', sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    require(uid == os.getuid(), 'PEER_UID')
    return identity(pid)


def exact_files(left, right):
    require(Path(left).stat().st_size == Path(right).stat().st_size, 'BYTE_LENGTH')
    with Path(left).open('rb') as a, Path(right).open('rb') as b:
        while True:
            x, y = a.read(MIB), b.read(MIB)
            require(x == y, 'FULL_BYTE_MISMATCH')
            if not x:
                break


def remaining(deadline):
    value = deadline - time.monotonic()
    require(value > 0, 'DEADLINE')
    return value
