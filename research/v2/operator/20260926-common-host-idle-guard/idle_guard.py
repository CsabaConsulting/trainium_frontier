"""Fail-closed idle evidence inside the existing inherited lifetime flock.

No locking path, source, evaluator, cache or device state is changed here.
The outer util-linux flock owns the lock until the synchronous driver exits.
"""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

PS = ['/usr/bin/ps', '-eo', 'pid=,ppid=,comm=,args=']
NEURON_LS = ['/opt/aws/neuron/bin/neuron-ls', '--json-output']
CHECK_ENV = {'PATH': '/usr/bin:/bin:/opt/aws/neuron/bin', 'LC_ALL': 'C'}
JOB_NAMES = {
    'torchrun', 'train.py', 'profile_actual_loop.py', 'host_driver.py',
    'host_queue.py', 'host_queue-6ef7.py', 'cpu_driver.py',
    'preflight_dryrun.py', 'submission_rehearsal.py', 'neuronx-cc',
    'walrus_driver', 'neuron_parallel_compile', 'torch.distributed.run',
}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def inherited_lock(lock_path, driver_path):
    """Authenticate the exact outer command and its already-locked open FD."""
    lock = Path(lock_path)
    st = lock.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
        raise RuntimeError('LOCK_PATH_NOT_OWNED_REGULAR_FILE')
    parent = os.getppid()
    expected = ['/usr/bin/flock', '-n', str(lock), '/usr/bin/python3',
                '-I', '-B', str(driver_path)]
    actual = Path(f'/proc/{parent}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    if actual != expected:
        raise RuntimeError('OUTER_FLOCK_COMMAND_MISMATCH')
    matches = []
    for entry in Path('/proc/self/fd').iterdir():
        fd = int(entry.name)
        try:
            fst = os.fstat(fd)
        except OSError:
            continue  # The enumeration directory descriptor has already closed.
        if (fst.st_dev, fst.st_ino) == (st.st_dev, st.st_ino):
            matches.append(fd)
    if len(matches) != 1:
        raise RuntimeError('EXACT_INHERITED_LOCK_FD_NOT_FOUND')
    fd = matches[0]
    info = Path(f'/proc/self/fdinfo/{fd}').read_text()
    if not any(line.startswith('lock:') and 'FLOCK' in line and 'WRITE' in line
               for line in info.splitlines()):
        raise RuntimeError('INHERITED_FD_NOT_ALREADY_EXCLUSIVELY_LOCKED')
    # Same open-file description, hence no release/reacquire window.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return {'path': str(lock), 'fd': fd, 'device': st.st_dev, 'inode': st.st_ino,
            'parent_pid': parent, 'driver_pid': os.getpid(),
            'verified_existing_exclusive_lock': True,
            'lifetime': 'Outer flock remains through driver and all synchronous phases.'}


def check_idle(run, own_pid, outer_pid, evidence):
    ps = run(PS, capture_output=True, text=True, timeout=15, env=CHECK_ENV)
    evidence['ps'] = {'command': PS, 'returncode': ps.returncode,
                      'stdout_sha256': digest(ps.stdout),
                      'stderr_sha256': digest(ps.stderr)}
    if ps.returncode != 0 or ps.stderr:
        raise RuntimeError('PS_COMMAND_FAILED')
    rows = ps.stdout.splitlines()
    if not rows:
        raise RuntimeError('PS_EMPTY_OUTPUT')
    busy, seen = [], set()
    for line in rows:
        fields = line.split(None, 3)
        if len(fields) != 4 or not fields[0].isdigit() or not fields[1].isdigit():
            raise RuntimeError('PS_PARSE_FAILED')
        pid, ppid, comm, command = int(fields[0]), int(fields[1]), fields[2], fields[3]
        if pid in seen or pid <= 0:
            raise RuntimeError('PS_DUPLICATE_OR_INVALID_PID')
        seen.add(pid)
        if pid in (own_pid, outer_pid):
            continue  # Only the authenticated current driver and its exact flock parent.
        names = {Path(part).name for part in command.split()}
        matched = sorted(names & JOB_NAMES)
        if 'prepare.py' in names and 'eval-public' in command.split():
            matched.append('prepare.py eval-public')
        if comm.startswith(('neuron', 'walrus', 'torchrun')):
            matched.append('native_process_comm:' + comm)
        if matched:
            busy.append({'pid': pid, 'ppid': ppid, 'comm': comm,
                         'matched': matched, 'command_sha256': digest(command)})
    evidence['ps'].update({'row_count': len(rows), 'busy': busy,
                           'excluded_pids': [own_pid, outer_pid]})
    if not {own_pid, outer_pid}.issubset(seen):
        raise RuntimeError('PS_OWN_LOCK_PROCESSES_MISSING')
    # Run both requested checks even when ps reports a busy machine.
    nl = run(NEURON_LS, capture_output=True, text=True, timeout=15, env=CHECK_ENV)
    evidence['neuron_ls'] = {'command': NEURON_LS, 'returncode': nl.returncode,
                             'stdout_sha256': digest(nl.stdout),
                             'stderr_sha256': digest(nl.stderr)}
    if nl.returncode != 0 or nl.stderr:
        raise RuntimeError('NEURON_LS_COMMAND_FAILED')
    devices = json.loads(nl.stdout)
    # Schema observed on both exact trn2.3xlarge hosts: one device0, four LNC2 cores.
    if not isinstance(devices, list) or len(devices) != 1:
        raise RuntimeError('NEURON_LS_DEVICE_SCHEMA_UNKNOWN')
    device = devices[0]
    if (not isinstance(device, dict) or device.get('neuron_device') != 0
            or device.get('instance_type') != 'trn2.3xlarge'
            or device.get('nc_count') != 4
            or device.get('logical_neuroncore_config') != 2
            or device.get('neuroncore_ids') != [0, 1, 2, 3]
            or not isinstance(device.get('neuron_processes'), list)):
        raise RuntimeError('NEURON_LS_DEVICE_SCHEMA_UNKNOWN')
    processes = device['neuron_processes']
    pids = []
    for process in processes:
        if not isinstance(process, dict) or type(process.get('pid')) is not int or process['pid'] <= 0:
            raise RuntimeError('NEURON_LS_PROCESS_SCHEMA_UNKNOWN')
        pids.append(process['pid'])
    evidence['neuron_ls'].update({'device': 0, 'lnc': 2, 'cores': [0, 1, 2, 3],
                                  'process_pids': pids})
    if busy:
        raise RuntimeError('PS_NATIVE_JOB_BUSY')
    if pids:
        raise RuntimeError('NEURON_DEVICE_BUSY')


def require_idle_under_inherited_lock(lock_path, driver_path, record):
    evidence = {'schema': 'same-host-lifetime-lock-idle-v1', 'started_utc': utc(),
                'status': 'REFUSED', 'lock_path': str(lock_path)}
    try:
        evidence['lock'] = inherited_lock(lock_path, driver_path)
        check_idle(subprocess.run, os.getpid(), os.getppid(), evidence)
        evidence['status'] = 'PASS'
    except Exception as exc:
        evidence['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        evidence['ended_utc'] = utc()
        # No complete ps argv dump or process environment is persisted.
        record('under-lock-idle.json', evidence)
