"""One bounded diagnostic through the established detached component manual path."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path('/home/ubuntu/trainium_l1_fp8_vjpdiag_20260925_trn21_r1')
BINDING_SHA = 'abd2c37456b5ea6faf6c73d9b6285ca2027f7e17ca4fe6918e73ec7ac086a2ad'
SOURCE_SHA = '921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858'
QUEUE_ROOT = Path('/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1')
REVIEW_PATH = ROOT / 'diagnostic-independent-review.md'
REVIEW_SHA = sys.argv[1]

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)

require(sha(ROOT / 'DIAGNOSTIC_BINDING.json') == BINDING_SHA, 'BINDING_SHA')
BINDING = json.loads((ROOT / 'DIAGNOSTIC_BINDING.json').read_text())
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
    require(sha(ROOT / 'check_native.py') == 'd8187ad2f3e68d204203a6252e16062e52521e8b93f119c7f0a9fbe82e4367fd', 'CHECKER_SHA')
    require(sha(Path(BINDING['prepare_path'])) == BINDING['prepare_sha256'], 'PREPARE_SHA')
    require(sha(REVIEW_PATH) == REVIEW_SHA, 'CPU_INDEPENDENT_REVIEW_SHA')
    require(json.loads((QUEUE_ROOT / 'runs/PAUSE_NEXT').read_text())['owner'] == 'trn22_queue_operator', 'OWNED_PAUSE')
    prior_queue = json.loads((QUEUE_ROOT / 'bootstrap-manifest.json').read_text())
    for path in [*prior_queue['stop_paths'], str(QUEUE_ROOT / 'runs/STOP')]:
        require(not os.path.lexists(path), 'STOP:' + path)
    for base in (Path('/home/ubuntu'), Path(BINDING['prepare_path']).parent, Path('/home/ubuntu/trainium_freshstart_20260918'), QUEUE_ROOT, ROOT, *[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
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
    with (DIRECTORY / 'diagnostic.log').open('x') as log:
        child = subprocess.Popen(command, cwd=DIRECTORY, env=PHASE['environment'], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        record('process.json', {'utc': utc(), 'monotonic': time.monotonic(), 'pid': child.pid})
        rc = child.wait()
    ended = time.monotonic()
    record('exit.json', {'utc': utc(), 'monotonic': ended, 'exit_code': rc, 'wall_seconds': ended - started})
    require(rc == 0, 'DIAGNOSTIC_EXIT_' + str(rc))
    ranks = [json.loads((ROOT / 'receipts/diagnostic' / ('diagnostic-rank' + str(r) + '.json')).read_text()) for r in range(4)]
    require(all(v['verdict'] == 'DIAGNOSTIC_ONLY_NOT_COMPONENT_PASS' and v['train_sha256'] == SOURCE_SHA and v['checker_sha256'] == BINDING['checker_sha256'] and v['label'] == PHASE['label'] and v['world_size'] == 4 and v['rank'] == i for i, v in enumerate(ranks)), 'DIAGNOSTIC_RECEIPTS')
    expected_cases = {'original_compiled_bound_apply', 'compiled_enclosing_wrapper',
                      'eager_apply_inspected', 'compiled_bound_apply_inspected_replay'}
    require(all(set(v['cases']) == expected_cases and
                all(case['status'] == 'OBSERVED' for case in v['cases'].values())
                for v in ranks), 'DIAGNOSTIC_CASE_INCOMPLETE')
    record('postflight.json', checks())
    record('terminal.json', {'utc': utc(), 'status': 'DIAGNOSTIC_COMPLETE_NOT_COMPONENT_PASS', 'label': PHASE['label'],
                            'source_sha256': SOURCE_SHA, 'exit_code': rc, 'ranks': ranks})
except Exception as error:
    if not (DIRECTORY / 'terminal.json').exists():
        record('terminal.json', {'utc': utc(), 'status': 'DIAGNOSTIC_INCOMPLETE', 'error': str(error), 'label': PHASE['label'], 'source_sha256': SOURCE_SHA})
    raise
