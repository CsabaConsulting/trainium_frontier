"""One reviewed native component command through the established detached manual path."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path('/home/ubuntu/trainium_l1_fp8_native_20260925_trn21_r1')
BINDING_SHA = '552c9cee54c9e107c3a33c59fc937bc084ad928cfb560adf661c0fc7f52d6fa7'
SOURCE_SHA = '921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858'
QUEUE_ROOT = Path('/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1')
REVIEW_PATH = ROOT / 'cpu-independent-review.json'
REVIEW_SHA = sys.argv[1]

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)

require(sha(ROOT / 'NATIVE_BINDING-v2.json') == BINDING_SHA, 'BINDING_SHA')
BINDING = json.loads((ROOT / 'NATIVE_BINDING-v2.json').read_text())
PHASE = BINDING['phases'][0]
DIRECTORY = ROOT / 'runs' / PHASE['label']
DIRECTORY.mkdir()

def record(name, value):
    with (DIRECTORY / name).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def checks():
    require(socket.gethostname() == 'ip-172-31-34-85', 'HOSTNAME')
    require(sha(ROOT / 'train.py') == SOURCE_SHA, 'SOURCE_SHA')
    require(sha(ROOT / 'check_native.py') == '7ba967b06834071f33cf9158c1f2a76555e9850d0abc988421b4b870ce1e3bc0', 'CHECKER_SHA')
    require(sha(Path(BINDING['prepare_path'])) == BINDING['prepare_sha256'], 'PREPARE_SHA')
    require(sha(REVIEW_PATH) == REVIEW_SHA, 'CPU_INDEPENDENT_REVIEW_SHA')
    require(json.loads((QUEUE_ROOT / 'runs/PAUSE_NEXT').read_text())['owner'] == 'trn21_queue_operator', 'OWNED_PAUSE')
    prior_queue = json.loads((QUEUE_ROOT / 'bootstrap-manifest.json').read_text())
    for path in [*prior_queue['stop_paths'], str(QUEUE_ROOT / 'runs/STOP')]:
        require(not os.path.lexists(path), 'STOP:' + path)
    for base in (Path('/home/ubuntu'), Path(BINDING['prepare_path']).parent, Path('/home/ubuntu/trainium_freshstart_20260918'), QUEUE_ROOT, ROOT):
        for rel in ('STOP', 'research/agentic/STOP', 'research/v2/STOP'):
            require(not os.path.lexists(base / rel), 'STOP:' + str(base / rel))
    s = os.statvfs('/')
    free = s.f_bavail * s.f_frsize
    require(free >= (6.5 + 4) * 1024**3, 'DISK_FLOOR_AND_FOOTPRINT')
    active = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            names = [Path(v.decode(errors='replace')).name for v in p.read_bytes().split(b'\0') if v]
            if any(n in names for n in ('train.py', 'torchrun', 'host_driver.py', 'preflight_dryrun.py', 'host_queue-6ef7.py', 'check_native.py')) or ('prepare.py' in names and 'eval-public' in names):
                active.append(int(p.parent.name))
        except OSError:
            pass
    require(not active, 'OTHER_WORKLOAD:' + str(active))
    devices = list(map(str, Path('/dev').glob('neuron[0-9]*')))
    require(bool(devices), 'NO_NEURON_DEVICE')
    holders = subprocess.run(['/usr/bin/fuser', *devices], capture_output=True, text=True)
    require(holders.returncode == 1, 'NEURON_BUSY:' + holders.stdout)
    return {'utc': utc(), 'free_disk_bytes': free, 'active_workloads': active,
            'device_holders': [], 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'source_sha256': SOURCE_SHA, 'cpu_review_sha256': REVIEW_SHA}

try:
    record('preflight.json', checks())
    command = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=60s', str(PHASE['wall_timeout_seconds']) + 's', *PHASE['command']]
    started = time.monotonic()
    record('launch.json', {'utc': utc(), 'monotonic': started, 'label': PHASE['label'],
                           'command': command, 'environment': PHASE['environment'], 'cpu_review_sha256': REVIEW_SHA})
    with (DIRECTORY / 'component.log').open('x') as log:
        child = subprocess.Popen(command, cwd=DIRECTORY, env=PHASE['environment'], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        record('process.json', {'utc': utc(), 'monotonic': time.monotonic(), 'pid': child.pid})
        rc = child.wait()
    ended = time.monotonic()
    record('exit.json', {'utc': utc(), 'monotonic': ended, 'exit_code': rc, 'wall_seconds': ended - started})
    require(rc == 0, 'NATIVE_COMPONENT_EXIT_' + str(rc))
    ranks = [json.loads((ROOT / 'receipts/native-component' / ('rank' + str(r) + '.json')).read_text()) for r in range(4)]
    require(all(v['verdict'] == 'PASS_NATIVE_COMPONENT_NUMERICS_ONLY' and v['train_sha256'] == SOURCE_SHA and v['world_size'] == 4 and v['rank'] == i for i, v in enumerate(ranks)), 'COMPONENT_RECEIPTS')
    record('postflight.json', checks())
    record('terminal.json', {'utc': utc(), 'status': 'PASS_NATIVE_COMPONENT_ONLY', 'label': PHASE['label'],
                            'source_sha256': SOURCE_SHA, 'exit_code': rc, 'ranks': ranks})
except Exception as error:
    if not (DIRECTORY / 'terminal.json').exists():
        record('terminal.json', {'utc': utc(), 'status': 'FAILED', 'error': str(error), 'label': PHASE['label'], 'source_sha256': SOURCE_SHA})
    raise
