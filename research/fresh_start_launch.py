#!/usr/bin/env python3
"""Launch one bounded, manifested job on an idle campaign host.

The JSON request supplies host (trn21/trn22), remote checkout, expected_commit,
label, purpose, command (argv, beginning with timeout), and explicit environment.
Set runner_owns_manifest for run_experiment.py: its log and launch manifest must
not be pre-created. This helper never stops an existing process or edits source.
Deployment and selection of an experiment remain separate decisions.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from fresh_start_sync import HOSTS, ROOT, SSH


REMOTE_SCRIPT = r'''
import datetime, fcntl, hashlib, json, os, pathlib, re, shutil, socket, stat, subprocess, sys
# This interpreter is independent of the local launcher and the later
# training child's env -i. Refuse optimization before checkout, lock, receipt,
# manifest or guard side effects; assert statements elsewhere cannot save us.
if sys.flags.optimize != 0 or 'PYTHONOPTIMIZE' in os.environ:
    raise RuntimeError('optimized Python is forbidden in the remote launcher')
m = json.load(sys.stdin)
if not isinstance(m, dict) or m.get('dispatch_authorized') is not True:
    raise RuntimeError('one-label remote dispatch is not authorized')
label = m.get('label')
if not isinstance(label, str) or re.fullmatch(r'fs-[A-Za-z0-9_.-]+', label) is None:
    raise RuntimeError('invalid remote science label')
if (not isinstance(m.get('expected_commit'), str)
        or re.fullmatch(r'[0-9a-f]{40}', m['expected_commit']) is None):
    raise RuntimeError('invalid remote expected commit')
if (not isinstance(m.get('remote'), str)
        or not pathlib.Path(m['remote']).is_absolute()
        or '..' in pathlib.Path(m['remote']).parts):
    raise RuntimeError('invalid remote checkout path')
if m.get('host') not in ('trn21', 'trn22') or m.get('expected_hostname') != socket.gethostname():
    raise RuntimeError('remote host identity mismatch')
config = m.get('monitor_interlock')
if not isinstance(config, dict):
    raise RuntimeError('paired remote interlock is required')
interlock_paths = {}
for key in ('pair_binding_path', 'interlock_path', 'no_spawn_path'):
    value = config.get(key)
    if (not isinstance(value, str) or not pathlib.Path(value).is_absolute()
            or '..' in pathlib.Path(value).parts):
        raise RuntimeError('invalid remote interlock path: ' + key)
    interlock_paths[key] = pathlib.Path(value)
if len({path.parent for path in interlock_paths.values()}) != 1:
    raise RuntimeError('remote interlock paths have different parents')
if len(set(interlock_paths.values())) != 3:
    raise RuntimeError('remote interlock paths are not distinct')
command = m.get('command')
if (not isinstance(command, list) or not command
        or not all(isinstance(arg, str) for arg in command)
        or command[0] != 'timeout'
        or any(arg.startswith('PYTHONOPTIMIZE=') for arg in command)):
    raise RuntimeError('invalid bounded remote command')
environment = m.get('environment')
if (not isinstance(environment, dict)
        or not all(isinstance(key, str) and isinstance(value, str)
                   for key, value in environment.items())
        or 'PYTHONOPTIMIZE' in environment
        or environment.get('NEURON_LOGICAL_NC_CONFIG') not in ('1', '2')
        or '--optlevel=1' not in environment.get('NEURON_CC_FLAGS', '')):
    raise RuntimeError('unsafe remote child environment')
if (type(m.get('min_free_bytes')) is not int
        or m['min_free_bytes'] != 13 * 1024**3 // 2
        or type(m.get('pre_launch_required_free_bytes')) is not int
        or m['pre_launch_required_free_bytes'] < 10 * 1024**3):
    raise RuntimeError('remote disk threshold mismatch')
if type(m.get('runner_owns_manifest', False)) is not bool:
    raise RuntimeError('invalid manifest ownership policy')
stage = 'before_checkout'
spawn_call_started = False

def write_no_spawn_before_popen(stage, error):
    # This source is sent by the pinned launcher and works even if the remote
    # checkout is absent or its imports fail. It matches the paired monitor's
    # exclusive, private, fsynced no-spawn receipt contract.
    config = m['monitor_interlock']
    path = pathlib.Path(config['no_spawn_path'])
    if not path.is_absolute():
        raise RuntimeError('no-spawn receipt path is not absolute')
    parent = path.parent.stat(follow_symlinks=False)
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid()
            or parent.st_mode & 0o022):
        raise RuntimeError('no-spawn receipt directory is not privately owned')
    payload = {
        'schema': 'paired-monitor-no-spawn-v1', 'status': 'NO_GUARD_SPAWNED',
        'label': m['label'], 'stage': stage,
        'expected_commit': m.get('expected_commit'),
        'pair_binding_path': config['pair_binding_path'],
        'error_type': type(error).__name__, 'error': str(error)[:240],
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        data = (json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n').encode()
        while data:
            data = data[os.write(fd, data):]
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

try:
    r = pathlib.Path(m['remote'])
    if not r.is_dir() or r.is_symlink():
        raise RuntimeError('checkout missing or symlinked')
    stage = 'checkout_import'
    sys.path.insert(0, str(r / 'research'))
    from fresh_start_host import HOST_DIRECTORY, HOST_LOCK, MIN_OPERATIONAL_FREE_BYTES, preflight, sufficient_free_space
    stage = 'host_lock'
    HOST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    host_lock = HOST_LOCK.open('a')
    fcntl.flock(host_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Prior-run cleanup is inventory-only. A stale process causes HOLD; it is
    # never an authorization to kill by marker or numeric PID.
    stage = 'early_disk'
    free = shutil.disk_usage(r).free
    if m.get('min_free_bytes') != MIN_OPERATIONAL_FREE_BYTES:
        raise RuntimeError('request operational floor mismatch')
    if not sufficient_free_space(free):
        raise RuntimeError('less than 6.5 GiB operational free space')
    stage = 'source_identity'
    source_paths = ['train.py', 'prepare.py', 'research/scratch', 'research/run_experiment.py',
                    'research/fresh_start_host.py', 'research/fresh_start_guard.py',
                    'research/fresh_start_launch.py', 'research/fresh_start_backup.py',
                    'research/fresh_start_monitor_interlock.py', 'research/fresh_start_sync.py',
                    'research/v2/disk-monitor/root_free_guard_monitor.py',
                    'research/v2/disk-monitor/paired_root_free_guard_monitor.py']
    fingerprints = m.get('fingerprint_paths', [])
    if (not isinstance(fingerprints, list)
            or any(not isinstance(path, str) or path not in source_paths
                   for path in fingerprints)):
        raise RuntimeError('invalid source fingerprint paths')
    dirty = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *source_paths], cwd=r)
    new = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', *source_paths], cwd=r)
    if dirty.strip() or new.strip():
        raise RuntimeError('experiment source is not committed')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=r, text=True).strip()
    if commit != m['expected_commit']:
        raise RuntimeError('unexpected source commit: ' + commit)
    stage = 'label_identity'
    fdir = r / 'research/run-logs'
    manifest = fdir / (m['label'] + '.launch.json')
    log = fdir / (m['label'] + '.log')
    if os.path.lexists(manifest) or os.path.lexists(log):
        raise RuntimeError('label already exists')
    if m.get('runner_owns_manifest'):
        log = fdir / (m['label'] + '.driver.log')
    if os.path.lexists(log):
        raise RuntimeError('driver label already exists')
    stage = 'interlock_import'
    from fresh_start_monitor_interlock import (require_live_pair, write_interlock_pass,
                                               write_no_spawn)
except BaseException as error:
    try:
        write_no_spawn_before_popen(stage, error)
    except BaseException as receipt_error:
        print('NO_SPAWN_RECEIPT_FAILURE ' + repr(receipt_error), file=sys.stderr)
    raise
spawned = False
stage = 'before_preflight'
try:
    # Preflight can take time; check before and immediately after it. A dying
    # sidecar cannot leave a future guard unmonitored merely because the first
    # check passed.
    require_live_pair(m)
    stage = 'preflight'
    pre = preflight(r, m['label'])
    stage = 'after_preflight'
    require_live_pair(m)
    paths = ['train.py', 'prepare.py', 'research/run_experiment.py',
             'research/fresh_start_host.py', 'research/fresh_start_guard.py',
             'research/fresh_start_backup.py',
             'research/fresh_start_monitor_interlock.py'] + m.get('fingerprint_paths', [])
    m.update(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             hostname=socket.gethostname(), git_commit=commit, source_dirty=False,
             preflight_free_bytes=free,
             source_sha256={p: hashlib.sha256((r / p).read_bytes()).hexdigest()
                            for p in dict.fromkeys(paths)})
    env = os.environ.copy()
    for key in ('NEURON_RT_VISIBLE_CORES', 'NEURON_RT_NUM_CORES'):
        env.pop(key, None)
    env.update(m['environment'])
    if not m.get('runner_owns_manifest'):
        with manifest.open('x') as f:
            json.dump(m, f, indent=2)
    stage = 'before_job'
    snapshot = require_live_pair(m)
    write_interlock_pass(m, snapshot)
    job = fdir / (m['label'] + '.job.json')
    with job.open('x') as f:
        json.dump(m, f, indent=2)
    # This is the last pre-spawn instruction with nontrivial latency. On a
    # single-sidecar failure after this point, the survivor awaits the exact
    # guard/process receipt and TERM-signals only the authenticated guard pidfd.
    stage = 'before_guard_spawn'
    require_live_pair(m)
    # A signal can arrive after Popen has spawned the guard but before Python
    # assigns its return value. Never claim NO_GUARD_SPAWNED once this call
    # begins; ambiguity is a manual HOLD and the surviving monitor stays up.
    with log.open('x') as f:
        spawn_call_started = True
        p = subprocess.Popen([sys.executable, str(r / 'research/fresh_start_guard.py'),
                              str(job), '--lock-fd', str(host_lock.fileno())],
                             cwd=r, env=env, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True, pass_fds=(host_lock.fileno(),))
    spawned = True
    print(json.dumps(dict(label=m['label'], pid=p.pid, commit=commit,
                         hostname=m['hostname'], log=str(log), started_at=m['started_at'],
                         protocol_version=2, preflight_passed=pre['passed'])))
except BaseException as error:
    if not spawned and not spawn_call_started:
        try:
            write_no_spawn(m, stage, error)
        except BaseException as receipt_error:
            print('NO_SPAWN_RECEIPT_FAILURE ' + repr(receipt_error), file=sys.stderr)
    elif not spawned:
        print('GUARD_SPAWN_STATUS_UNKNOWN_HOLD ' + repr(error), file=sys.stderr)
    raise
'''


def require_unoptimized_python():
    """Refuse before local receipt or SSH; remote has an independent gate."""
    if sys.flags.optimize != 0 or "PYTHONOPTIMIZE" in os.environ:
        raise RuntimeError("optimized Python is forbidden in the local launcher")


def validate_request(request):
    """Reject an unreviewed or malformed request before local side effects."""
    if not isinstance(request, dict) or request.get("dispatch_authorized") is not True:
        raise RuntimeError("one-label dispatch is not authorized")
    host = request.get("host")
    if not isinstance(host, str) or host not in HOSTS:
        raise RuntimeError("unknown dispatch host")
    if (not isinstance(request.get("expected_hostname"), str)
            or not request["expected_hostname"]):
        raise RuntimeError("missing expected host identity")
    label = request.get("label")
    if not isinstance(label, str) or re.fullmatch(r"fs-[A-Za-z0-9_.-]+", label) is None:
        raise RuntimeError("invalid science label")
    expected_commit = request.get("expected_commit")
    if (not isinstance(expected_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None):
        raise RuntimeError("invalid expected commit")
    remote_value = request.get("remote")
    if not isinstance(remote_value, str):
        raise RuntimeError("invalid checkout path")
    remote = Path(remote_value)
    if (remote.parent != Path("/home/ubuntu") or '..' in remote.parts
            or not remote.name.startswith("trainium_freshstart_")):
        raise RuntimeError("invalid checkout path")
    config = request.get("monitor_interlock")
    if not isinstance(config, dict):
        raise RuntimeError("paired interlock is required")
    paths = {}
    for key in ("pair_binding_path", "interlock_path", "no_spawn_path"):
        value = config.get(key)
        if (not isinstance(value, str) or not Path(value).is_absolute()
                or '..' in Path(value).parts):
            raise RuntimeError("invalid interlock path: " + key)
        paths[key] = Path(value)
    if len({path.parent for path in paths.values()}) != 1:
        raise RuntimeError("interlock paths have different parents")
    if len(set(paths.values())) != 3:
        raise RuntimeError("interlock paths are not distinct")
    command = request.get("command")
    if (not isinstance(command, list) or not command
            or not all(isinstance(arg, str) for arg in command)
            or command[0] != "timeout"
            or any(arg.startswith("PYTHONOPTIMIZE=") for arg in command)):
        raise RuntimeError("a bounded command without Python optimization is required")
    environment = request.get("environment")
    if (not isinstance(environment, dict)
            or not all(isinstance(key, str) and isinstance(value, str)
                       for key, value in environment.items())
            or "PYTHONOPTIMIZE" in environment
            or environment.get("NEURON_LOGICAL_NC_CONFIG") not in ("1", "2")
            or "--optlevel=1" not in environment.get("NEURON_CC_FLAGS", "")):
        raise RuntimeError("unsafe child environment")
    from fresh_start_host import MIN_OPERATIONAL_FREE_BYTES
    if (type(request.get("min_free_bytes")) is not int
            or request["min_free_bytes"] != MIN_OPERATIONAL_FREE_BYTES
            or type(request.get("pre_launch_required_free_bytes")) is not int
            or request["pre_launch_required_free_bytes"] < 10 * 1024**3):
        raise RuntimeError("disk threshold mismatch")
    if type(request.get("runner_owns_manifest", False)) is not bool:
        raise RuntimeError("invalid manifest ownership policy")
    fingerprints = request.get("fingerprint_paths", [])
    if (not isinstance(fingerprints, list)
            or any(not isinstance(path, str) or Path(path).is_absolute()
                   or any(part.startswith(".") for part in Path(path).parts)
                   for path in fingerprints)):
        raise RuntimeError("invalid source fingerprint paths")


def main():
    require_unoptimized_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    validate_request(request)
    receipt = ROOT / "research/run-logs" / (request["label"] + ".dispatch.json")
    with receipt.open("x") as output:
        # Preserve the intended action locally even if the remote host disappears.
        payload = dict(request=request, requested_at=datetime.now(timezone.utc).isoformat())
        json.dump(payload, output, indent=2)
        output.flush()
    completed = subprocess.run(
        [*SSH, HOSTS[request["host"]], "python3 -c " + shlex.quote(REMOTE_SCRIPT)],
        input=json.dumps(request), text=True, capture_output=True, timeout=90)
    payload.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    receipt.write_text(json.dumps(payload, indent=2) + "\n")
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    completed.check_returncode()


if __name__ == "__main__":
    main()
