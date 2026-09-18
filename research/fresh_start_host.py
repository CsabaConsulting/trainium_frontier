#!/usr/bin/env python3
"""Host-wide experiment ownership, bounded health capture and owned cleanup.

No broad process killing, driver changes, cache deletion or artifact deletion.
The launcher holds the same lock through setup, workload, cleanup and postflight.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time

PROTOCOL_VERSION = 2
HOST_DIRECTORY = Path('/home/ubuntu/.cache/trainium_freshstart')
HOST_LOCK = HOST_DIRECTORY / 'device0.lock'
HOST_STATE = HOST_DIRECTORY / 'device0-last-postflight.json'
MARKER = 'TRAINIUM_FRESH_START_RUN_ID'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with temp.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    if replace:
        temp.replace(path)
    else:
        try:
            os.link(temp, path)
        finally:
            temp.unlink()


def command(argv, timeout=6):
    start = time.monotonic()
    try:
        p = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        result = dict(argv=argv, returncode=p.returncode, stdout=p.stdout[-65536:],
                      stderr=p.stderr[-8192:], timed_out=False)
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        result = dict(argv=argv, returncode=None, stdout=decode(error.stdout)[-65536:],
                      stderr=decode(error.stderr)[-8192:], timed_out=True)
    except OSError as error:
        result = dict(argv=argv, returncode=None, stdout='', stderr=str(error), timed_out=False)
    result['seconds'] = time.monotonic() - start
    return result


def process_record(pid):
    root = Path('/proc') / str(pid)
    if root.stat().st_uid != os.getuid():
        return None
    fields = (root / 'stat').read_text().rpartition(')')[2].split()
    argv = [x.decode(errors='replace') for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
    try:
        cwd = os.readlink(root / 'cwd')
    except OSError:
        cwd = None
    return dict(pid=int(pid), ppid=int(fields[1]), process_group=int(fields[2]),
                session=int(fields[3]), state=fields[0], start_ticks=int(fields[19]),
                rss_bytes=int(fields[21]) * os.sysconf('SC_PAGE_SIZE'),
                argv=argv[:24], cwd=cwd)


def owned_processes(label):
    needle = (MARKER + '=' + label).encode()
    result = []
    for root in Path('/proc').iterdir():
        if not root.name.isdigit() or int(root.name) == os.getpid():
            continue
        try:
            if root.stat().st_uid != os.getuid():
                continue
            if needle not in (root / 'environ').read_bytes().split(b'\0'):
                continue
            record = process_record(int(root.name))
            if record and record['state'] != 'Z':
                result.append(record)
        except (OSError, ValueError, IndexError):
            continue
    return result


def cleanup_owned(label):
    """Only a matching UID, unique marker and unchanged process birth qualify."""
    actions = []
    for sig, grace in ((signal.SIGTERM, 8), (signal.SIGKILL, 3)):
        pending = owned_processes(label)
        if not pending:
            break
        for old in pending:
            try:
                current = process_record(old['pid'])
                if current is None or current['start_ticks'] != old['start_ticks']:
                    continue
                if not any(p['pid'] == old['pid'] for p in owned_processes(label)):
                    continue
                os.kill(old['pid'], sig)
                actions.append(dict(pid=old['pid'], start_ticks=old['start_ticks'],
                                    signal=sig.name, argv=old['argv']))
            except (OSError, ProcessLookupError):
                continue
        deadline = time.monotonic() + grace
        while owned_processes(label) and time.monotonic() < deadline:
            time.sleep(0.25)
    return dict(actions=actions, remaining=owned_processes(label),
                artifacts_deleted=[], caches_deleted=[], driver_modified=False)


def snapshot(remote, since=None):
    started = utc_now()
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        parts = value.split()
        memory[key] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == 'kB' else 1)
    holders = command(['fuser', '/dev/neuron0'])
    listing = command(['/opt/aws/neuron/bin/neuron-ls', '--json-output'])
    # A bounded monitor sample also records temperatures/power/runtime counters
    # when this SDK exposes them. timeout's124 is expected for a streaming tool.
    monitor = command(['timeout', '--kill-after=1s', '3s', '/opt/aws/neuron/bin/neuron-monitor'], 5)
    kernel_command = ['journalctl', '-k', '--no-pager', '-n', '200', '-o', 'short-iso']
    if since:
        kernel_command += ['--since', since]
    kernel = command(kernel_command)
    if kernel['returncode'] != 0 or 'permission' in kernel['stderr'].lower():
        kernel = command(['sudo', '-n', *kernel_command])
    error_pattern = re.compile(r'neuron.*(?:fatal|timeout|hang|reset|uncorrect|error)|'
                               r'out of memory|oom-kill|killed process', re.I)
    kernel['relevant_error_lines'] = [line for line in kernel['stdout'].splitlines()
                                      if error_pattern.search(line)]
    relevant_processes = []
    for root in Path('/proc').iterdir():
        if not root.name.isdigit():
            continue
        try:
            record = process_record(int(root.name))
            if record and any(any(term in arg for term in ('torchrun', 'neuronx-cc', 'phase_timer.py', 'train.py'))
                              for arg in record['argv'][:3]):
                relevant_processes.append(record)
        except (OSError, ValueError, IndexError):
            continue
    disk = shutil.disk_usage(remote)
    vfs = os.statvfs(remote)
    caches = {}
    for path in ('/tmp/neuron_cache', '/tmp/neff_cache'):
        if Path(path).exists():
            caches[path] = command(['du', '-sk', path], 5)
    sdk = command(['/home/ubuntu/workspace/native_venv/bin/python', '-c',
                   'import importlib.metadata as m,json; print(json.dumps({p:m.version(p) '
                   'for p in ["torch","torch-neuronx","neuronx-cc","nki"]}))'])
    result = dict(protocol_version=PROTOCOL_VERSION, started_at=started, completed_at=utc_now(),
                  hostname=socket.gethostname(), kernel_release=os.uname().release,
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  uptime_seconds=float(Path('/proc/uptime').read_text().split()[0]), sdk=sdk,
                  memory_bytes={key:memory.get(key) for key in ('MemTotal','MemFree','MemAvailable','Cached','SwapTotal','SwapFree')},
                  disk=dict(total_bytes=disk.total, used_bytes=disk.used, free_bytes=disk.free,
                            free_inodes=vfs.f_favail), device_holders=holders,
                  neuron_ls=listing, neuron_monitor=monitor, kernel_log=kernel,
                  relevant_processes=relevant_processes, cache_sizes=caches,
                  cache_policy='reuse shared immutable compilation caches; no purge or OS cache drop')
    try:
        devices = json.loads(listing['stdout'])
        result['neuron_devices'] = devices
        result['neuron_process_count'] = sum(len(d.get('neuron_processes', [])) for d in devices)
    except (ValueError, TypeError):
        result['neuron_devices'] = None
        result['neuron_process_count'] = None
    return result


def health_failures(record, exclude_pids=()):
    failures = []
    if record['device_holders']['returncode'] != 1:
        failures.append('device has holders, or fuser failed')
    if record['neuron_ls']['returncode'] != 0 or record['neuron_process_count'] != 0:
        failures.append('neuron-ls failed or reports a live Neuron process')
    if record['memory_bytes']['MemAvailable'] < 16 * 1024**3:
        failures.append('less than16GiB host memory available')
    if record['disk']['free_bytes'] < 8 * 1024**3:
        failures.append('less than8GiB disk space available')
    if record['disk']['free_inodes'] < 10000:
        failures.append('less than10000 free filesystem inodes')
    pending = [p for p in record['relevant_processes']
               if p['pid'] not in exclude_pids and p['state'] != 'Z'
               and 'fresh_start_' not in Path(p['argv'][0]).name]
    if pending:
        failures.append('training/compiler processes remain: ' + ','.join(str(p['pid']) for p in pending))
    return failures


def preflight(remote, label):
    previous = json.loads(HOST_STATE.read_text()) if HOST_STATE.exists() else None
    pre_cleanup = cleanup_owned(previous['label']) if previous else dict(actions=[], remaining=[])
    record = snapshot(remote, previous.get('completed_at') if previous else None)
    record.update(label=label, remote=str(remote), stage='preflight', previous_postflight=previous,
                  cleanup=pre_cleanup)
    record['failures'] = health_failures(record, (os.getpid(), os.getppid()))
    record['passed'] = not record['failures']
    write_json(Path(remote) / 'research/run-logs' / (label + '.preflight.json'), record)
    if record['failures']:
        raise RuntimeError('preflight failed: ' + '; '.join(record['failures']))
    return record


def main():
    """Record a postflight for a job launched before automatic guards existed."""
    import argparse
    import fcntl
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--adopt-completed', action='store_true', required=True)
    ap.add_argument('--remote', type=Path, required=True)
    ap.add_argument('--label', required=True)
    args = ap.parse_args()
    assert re.fullmatch(r'fs-[A-Za-z0-9_.-]+', args.label)
    manifest = json.loads((args.remote / 'research/run-logs' / (args.label + '.launch.json')).read_text())
    HOST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with HOST_LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        post = snapshot(args.remote, manifest.get('started_at'))
        failures = health_failures(post, (os.getpid(), os.getppid()))
        record = dict(protocol_version=2, stage='postflight', label=args.label,
                      remote=str(args.remote), health=post, failures=failures,
                      host_clean=not failures, completed_at=utc_now(),
                      source_commit=manifest.get('git_commit', manifest.get('expected_commit')),
                      legacy_job_adopted_after_completion=True,
                      automatic_preflight_not_retroactively_claimed=True,
                      cleanup=dict(actions=[], artifacts_deleted=[], caches_deleted=[],
                                   reason='old job has no inherited ownership marker; read-only check'),
                      local_preservation_pending=True)
        path = args.remote / 'research/run-logs' / (args.label + '.postflight.json')
        write_json(path, record)
        write_json(HOST_STATE, dict(label=args.label, remote=str(args.remote), postflight=str(path),
                                   host_clean=not failures, completed_at=record['completed_at']), replace=True)
        print(json.dumps(dict(label=args.label, host_clean=not failures, postflight=str(path))))
        if failures:
            raise SystemExit('; '.join(failures))


if __name__ == '__main__':
    main()
