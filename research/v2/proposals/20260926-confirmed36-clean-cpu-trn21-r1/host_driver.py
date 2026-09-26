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

PAIR_ROOT = Path('/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1')
ROOT = PAIR_ROOT
REQUEST = None
PREP = Path("/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2") / 'prepare.py'
PINS = {"reference/request.json":"38160400738af45274aad1c96f9a54e0c5ec1343ab9eb87f1279b46e4f5ba354","candidate/request.json":"64d0b756784eb8a8306d1fcd739c09c355d7611479f8da9f64bf2fb7ffc4eca3"}
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

def checks(authenticated_outer_pid):
    require(os.getppid() == authenticated_outer_pid, 'AUTHENTICATED_PARENT_CHANGED')
    require(socket.gethostname() == 'ip-172-31-34-85', 'HOSTNAME')
    require(sha(Path(REQUEST['command'][REQUEST['command'].index('--train-py') + 1])) == REQUEST['source_sha256'], 'SOURCE_SHA')
    require(sha(Path(REQUEST['command'][2])) == REQUEST['g2_tool_sha256'], 'G2_SHA')
    require(sha(PREP) == REQUEST['prepare_sha256'], 'PREPARE_SHA')
    bases = [Path('/home/ubuntu'), PREP.parent, ROOT, *[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]]
    for base in bases:
        for name in ('STOP', 'research/agentic/STOP', 'research/v2/STOP'):
            require(not os.path.lexists(base / name), 'STOP:' + str(base / name))
    s = os.statvfs('/')
    free = s.f_bavail * s.f_frsize
    require(free >= 10.5 * 1024**3, 'DISK_FLOOR')
    mem = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:'))) * 1024
    require(mem >= 64 * 1024**3, 'AVAILABLE_RAM_FLOOR')
    active = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if int(p.parent.name) in (os.getpid(), authenticated_outer_pid):
                continue  # Only self and the exact parent authenticated by the shared guard.
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
    require(sha(PAIR_ROOT / 'idle_guard.py') == 'adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737', 'IDLE_GUARD_SHA')
    require(sha(PAIR_ROOT / 'dispatch_driver.py') == 'c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a', 'DISPATCH_SHA')
    for relative, expected in PINS.items():
        require(sha(PAIR_ROOT / relative) == expected, 'PACKET_SHA:' + relative)
    spec = importlib.util.spec_from_file_location('bound_idle_guard', PAIR_ROOT / 'idle_guard.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    authenticated_outer_pid = os.getppid()
    lock_fd = guard.require_idle_under_inherited_lock(
        '/home/ubuntu/.cache/trainium_freshstart/device0.lock', PAIR_ROOT / 'host_driver.py', record)
    for name in ('reference', 'candidate'):
        ROOT = PAIR_ROOT / name
        REQUEST = json.loads((ROOT / 'request.json').read_text())
        lock_fd = guard.require_idle_under_inherited_lock(
            '/home/ubuntu/.cache/trainium_freshstart/device0.lock', PAIR_ROOT / 'host_driver.py', record)
        record('preflight.json', checks(authenticated_outer_pid))
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
        record('terminal.json', {'utc': utc(), 'exit_code': code, 'label': REQUEST['label'],
                                 'cpu_only': True, 'verdict': result.get('verdict') if result else None,
                                 'reason': result.get('reason') if result else 'MISSING_G2_RECEIPT',
                                 'source_sha256': REQUEST['source_sha256']})
        require(code == 0 and result and result.get('verdict') == 'PASS', 'G2_ARM_FAILED:' + name)
        checkpoints = list((ROOT / 'tmp').glob('g2-*/out/final.pt'))
        require(len(checkpoints) == 1, 'EXACTLY_ONE_RETAINED_CHECKPOINT')
        checkpoint = checkpoints[0]
        with checkpoint.open('rb') as stream:
            checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        record('checkpoint-identity.json', {'path': str(checkpoint), 'bytes': checkpoint.stat().st_size,
                                          'sha256': checkpoint_sha, 'scope': 'file bytes only; ordinary-loader comparison requires a separately assigned later check'})
    ROOT = PAIR_ROOT
    # Same original flock remains owned throughout both arms and these bounded comparisons.
    env['PYTHONPATH'] = str(PREP.parent)
    comparisons = [
        ('standard-compare', [VENV + '/bin/python', '-B', str(PAIR_ROOT / 'preflight_dryrun.py'),
                             '--compare', str(PAIR_ROOT / 'reference/g2.json'), str(PAIR_ROOT / 'candidate/g2.json')]),
    ]
    for phase, command in comparisons:
        lock_fd = guard.require_idle_under_inherited_lock(
            '/home/ubuntu/.cache/trainium_freshstart/device0.lock', PAIR_ROOT / 'host_driver.py',
            lambda filename, value: record(phase + '-' + filename, value))
        record(phase + '-preflight.json', checks(authenticated_outer_pid))
        record(phase + '-launch.json', {'utc': utc(), 'command': command, 'environment': env, 'cpu_only': True})
        with (ROOT / (phase + '.json')).open('x') as out, (ROOT / (phase + '.stderr')).open('x') as err:
            code = subprocess.call(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=err, pass_fds=(lock_fd,))
        record(phase + '-exit.json', {'utc': utc(), 'exit_code': code})
        require(code == 0, 'COMPARISON_FAILED:' + phase)
        observed = json.loads((ROOT / (phase + '.json')).read_text())
        if phase == 'standard-compare':
            require(observed['identical'] is True and observed['n_params_compared'] == 65, 'STANDARD_IDENTITY')
    record('terminal.json', {'utc': utc(), 'exit_code': 0, 'status': 'CPU_PAIR_COMPLETE_PENDING_INDEPENDENT_REVIEW',
                             'cpu_only': True, 'native_launch_authorized': False})
except Exception as error:
    ROOT = PAIR_ROOT
    if not (ROOT / 'terminal.json').exists():
        record('terminal.json', {'utc': utc(), 'exit_code': 1, 'cpu_only': True, 'error': str(error),
                                'native_launch_authorized': False})
    raise
