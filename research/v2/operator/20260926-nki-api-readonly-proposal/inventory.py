"""Disabled proposal: stdlib-only bounded read of six exact installed SDK files."""
import ast
import datetime
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time

resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**2, 128 * 1024**2))
ROOT = Path('/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx')
SELECTED = ('__init__.py', 'nki_hop.py', 'nki_kernel.py', 'nki_op.py',
            'nki.py', 'experimental/__init__.py')
TERMS = ('nki_op', 'wrap_nki')
MAX_FILE = 512 * 1024
MAX_READ = 2 * 1024**2
MAX_EXCERPT = 192 * 1024
MAX_OUTPUT = 256 * 1024
started = time.monotonic()
out = {'schema': 'stdlib-readonly-installed-nki-wrapper-inventory-v1',
       'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'status': 'UNKNOWN', 'files': [], 'read_bytes': 0,
       'no_sdk_imports': True, 'no_host_writes': True, 'no_device_access': True,
       'interpretation': 'Source/export presence only; no runtime API availability proof.'}


def deadline(*_args):
    raise TimeoutError('READONLY_INVENTORY_WALL_LIMIT')


def identity(s):
    return {'device': s.st_dev, 'inode': s.st_ino, 'bytes': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
            'mode': s.st_mode}


def no_links(p):
    for ancestor in (p, *p.parents):
        if ancestor.is_symlink():
            raise RuntimeError('SYMLINK_REFUSED:' + str(ancestor))


signal.signal(signal.SIGALRM, deadline)
signal.alarm(14)
excerpt_bytes = 0
try:
    no_links(ROOT)
    if not ROOT.is_dir():
        raise RuntimeError('SELECTED_PACKAGE_ROOT_MISSING')
    spec = importlib.machinery.PathFinder.find_spec('torch_neuronx', [str(ROOT.parent)])
    out['package_spec'] = None if spec is None else {
        'origin': spec.origin,
        'submodule_search_locations': list(spec.submodule_search_locations or ())}
    if spec is None or spec.origin != str(ROOT / '__init__.py'):
        raise RuntimeError('SELECTED_SOURCE_ORIGIN_NOT_CONFIRMED')
    for rel in SELECTED:
        p = ROOT / rel
        rec = {'path': str(p)}
        out['files'].append(rec)
        no_links(p)
        try:
            before = p.lstat()
        except FileNotFoundError:
            rec['status'] = 'ABSENT'; continue
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError('NONREGULAR_REFUSED:' + str(p))
        if not 0 <= before.st_size <= MAX_FILE or out['read_bytes'] + before.st_size > MAX_READ:
            raise RuntimeError('READ_BOUND_REFUSED:' + str(p))
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            if identity(os.fstat(fd)) != identity(before):
                raise RuntimeError('OPEN_RACE:' + str(p))
            chunks = []; size = 0
            while True:
                chunk = os.read(fd, min(65536, before.st_size - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                if size > before.st_size:
                    raise RuntimeError('FILE_GREW:' + str(p))
                chunks.append(chunk)
            after = os.fstat(fd)
            if identity(before) != identity(after) or identity(before) != identity(p.lstat()):
                raise RuntimeError('FILE_CHANGED:' + str(p))
        finally:
            os.close(fd)
        raw = b''.join(chunks)
        if len(raw) != before.st_size:
            raise RuntimeError('SHORT_READ:' + str(p))
        out['read_bytes'] += len(raw)
        text = raw.decode('utf-8')
        lines = text.splitlines()
        hits = [i for i, line in enumerate(lines) if any(term in line for term in TERMS)]
        selected_lines = set()
        for i in hits:
            selected_lines.update(range(max(0, i - 5), min(len(lines), i + 12)))
        excerpts = [{'line': i + 1, 'text': lines[i]} for i in sorted(selected_lines)]
        excerpt_bytes += sum(len(row['text'].encode('utf-8')) for row in excerpts)
        if excerpt_bytes > MAX_EXCERPT:
            raise RuntimeError('EXCERPT_BOUND_REFUSED')
        tree = ast.parse(text, filename=str(p))
        definitions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in TERMS:
                definition = ast.get_source_segment(text, node) or ''
                signature = node.name + '(' + ast.unparse(node.args) + ')'
                if node.returns is not None:
                    signature += ' -> ' + ast.unparse(node.returns)
                definitions.append({'name': node.name, 'line': node.lineno,
                                    'signature': signature,
                                    'signature_sha256': hashlib.sha256(signature.encode()).hexdigest(),
                                    'definition_sha256': hashlib.sha256(definition.encode()).hexdigest()})
        rec.update(status='READ_STABLE', identity=identity(before),
                   sha256=hashlib.sha256(raw).hexdigest(), excerpts=excerpts,
                   static_definitions=definitions)
    forbidden = [name for name in sys.modules if name == 'torch' or name.startswith('torch.')
                 or name == 'torch_neuronx' or name.startswith('torch_neuronx.')
                 or name == 'nki' or name.startswith('nki.')]
    if forbidden:
        raise RuntimeError('UNEXPECTED_SDK_IMPORT')
    out['status'] = 'READONLY_INVENTORY_COMPLETE'
except Exception as e:
    out['status'] = 'REFUSED_OR_INCOMPLETE'
    out['error'] = type(e).__name__ + ': ' + str(e)
finally:
    signal.alarm(0)
    out['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out['elapsed_seconds'] = time.monotonic() - started
    serialized = json.dumps(out)
    if len(serialized.encode()) > MAX_OUTPUT:
        serialized = json.dumps({'status': 'REFUSED_OR_INCOMPLETE',
                                 'error': 'OUTPUT_BOUND_REFUSED',
                                 'read_bytes': out['read_bytes']})
    print(serialized)
