"""Operator-only detached execution of the approved synthetic CPU G2 profile."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/ubuntu/trainium_base_initial_cache_bcbd_cpu_20260926_trn22_r1/g2/fs-20260926-base-cache-default-trn22-g2-r1')
REQUEST = json.loads((ROOT / 'request.json').read_text())
PREP = Path(REQUEST['prepare_dir']) / 'prepare.py'
VENV = '/home/ubuntu/workspace/native_venv'

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def require(value, message):
    if not value:
        raise RuntimeError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def record(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def checks():
    require(socket.gethostname() == 'ip-172-31-32-253', 'HOSTNAME')
    require(sha(Path(REQUEST['command'][REQUEST['command'].index('--train-py') + 1])) == REQUEST['source_sha256'], 'SOURCE_SHA')
    require(sha(Path(REQUEST['command'][2])) == REQUEST['g2_tool_sha256'], 'G2_SHA')
    require(sha(PREP) == REQUEST['prepare_sha256'], 'PREPARE_SHA')
    bases = [Path('/home/ubuntu'), PREP.parent, ROOT, *[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]]
    for base in bases:
        for name in ('STOP', 'research/agentic/STOP', 'research/v2/STOP'):
            require(not os.path.lexists(base / name), 'STOP:' + str(base / name))
    s = os.statvfs('/')
    free = s.f_bavail * s.f_frsize
    require(free >= 6.5 * 1024**3, 'DISK_FLOOR')
    mem = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:'))) * 1024
    require(mem >= 64 * 1024**3, 'AVAILABLE_RAM_FLOOR')
    active = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if int(p.parent.name) == os.getpid():
                continue  # Current driver was authenticated by the inherited-lock guard.
            names = [Path(t.decode(errors='replace')).name for t in p.read_bytes().split(b'\0') if t]
            if any(x in names for x in ('train.py', 'torchrun', 'host_driver.py', 'submission_rehearsal.py',
                                        'preflight_dryrun.py')) or ('prepare.py' in names and 'eval-public' in names):
                active.append(int(p.parent.name))
        except (OSError, ValueError):
            pass
    require(not active, 'OTHER_WORKLOAD:' + str(active))
    devices = sorted(str(p) for p in Path('/dev').glob('neuron[0-9]*'))
    require(devices, 'NO_NEURON_DEVICE')
    holders = subprocess.run(['/usr/bin/fuser', *devices], capture_output=True, text=True)
    require(holders.returncode == 1, 'NEURON_NOT_IDLE:' + holders.stdout)
    return {'utc': utc(), 'free_disk_bytes': free, 'available_ram_bytes': mem,
            'other_workloads': active, 'neuron_holders': [], 'source_sha256': REQUEST['source_sha256']}

try:
    require(sha(ROOT / 'idle_guard.py') == 'adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737', 'IDLE_GUARD_SHA')
    spec = importlib.util.spec_from_file_location('bound_idle_guard', ROOT / 'idle_guard.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    lock_fd = guard.require_idle_under_inherited_lock(
        '/home/ubuntu/.cache/trainium_freshstart/device0.lock', ROOT / 'host_driver.py', record)
    record('preflight.json', checks())
    env = {'HOME': '/home/ubuntu', 'PATH': VENV + '/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin',
           'NKI_VENV_PATH': VENV, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'TMPDIR': str(ROOT / 'tmp'), 'NEURON_LOGICAL_NC_CONFIG': '2',
           'NEURON_CC_FLAGS': '--optlevel=1 --auto-cast matmult --auto-cast-type bf16',
           'NEURON_COMPETITION_R1_CACHE_DIR': '/home/ubuntu/.cache/neuron_competition_r1',
           'NEURON_COMPETITION_R1_SEED': '42', 'NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS': '1800',
           'TRN_EXP_LABEL': REQUEST['label']}
    command = REQUEST['command']
    record('launch.json', {'utc': utc(), 'label': REQUEST['label'], 'command': command,
                           'environment': env, 'source_sha256': REQUEST['source_sha256'], 'cpu_only': True})
    with (ROOT / 'g2.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
        record('process.json', {'utc': utc(), 'pid': child.pid})
        code = child.wait()
    result = json.loads((ROOT / 'g2.json').read_text()) if (ROOT / 'g2.json').exists() else None
    if code == 0 and result and result.get('verdict') == 'PASS':
        checkpoints = list((ROOT / 'tmp').glob('g2-*/out/final.pt'))
        require(len(checkpoints) == 1, 'EXACTLY_ONE_RETAINED_CHECKPOINT')
        checkpoint = checkpoints[0]
        with checkpoint.open('rb') as stream:
            checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        record('checkpoint-identity.json', {'path': str(checkpoint), 'bytes': checkpoint.stat().st_size,
                                          'sha256': checkpoint_sha, 'scope': 'file bytes only; no tensor load'})
    record('terminal.json', {'utc': utc(), 'exit_code': code, 'label': REQUEST['label'],
                             'cpu_only': True, 'verdict': result.get('verdict') if result else None,
                             'reason': result.get('reason') if result else 'MISSING_G2_RECEIPT',
                             'source_sha256': REQUEST['source_sha256']})
    sys.exit(code)
except Exception as error:
    if not (ROOT / 'terminal.json').exists():
        record('terminal.json', {'utc': utc(), 'exit_code': 1, 'cpu_only': True,
                                'label': REQUEST['label'], 'error': str(error),
                                'source_sha256': REQUEST['source_sha256']})
    raise
