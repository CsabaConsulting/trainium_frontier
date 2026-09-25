"""Bounded helpers for this one fixed offline profile decoder; no SDK imports."""
import hashlib
import ctypes
import os
from pathlib import Path
import resource
import selectors
import signal
import stat
import subprocess
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def no_symlink(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'UNSAFE_PATH')
    for part in (path, *path.parents):
        require(not stat.S_ISLNK(part.lstat().st_mode), 'SYMLINK:' + str(part))
    require(path.resolve(strict=True) == path, 'RESOLVED_PATH_CHANGED')
    return path


def signature(path):
    s = no_symlink(path).lstat()
    require(stat.S_ISREG(s.st_mode), 'NOT_REGULAR:' + str(path))
    return {k: getattr(s, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_nlink')}


def check_deadline(deadline):
    require(time.monotonic() < deadline, 'OVERALL_WORK_DEADLINE')


def stream_hash(path, deadline, max_bytes):
    before = signature(path)
    require(before['st_size'] <= max_bytes, 'FILE_SIZE_LIMIT')
    h = hashlib.sha256()
    total = 0
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        opened = os.fstat(stream.fileno())
        require((opened.st_dev, opened.st_ino) == (before['st_dev'], before['st_ino']), 'OPEN_IDENTITY_CHANGED')
        while True:
            check_deadline(deadline)
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            require(total <= max_bytes, 'FILE_GREW')
            h.update(chunk)
    require(signature(path) == before and total == before['st_size'], 'FILE_CHANGED_DURING_HASH')
    return dict(before, sha256=h.hexdigest())


def exact_layout(root, entries):
    root = no_symlink(root)
    expected = {}
    for entry in entries:
        relative = Path(entry['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'UNSAFE_RELATIVE_PATH')
        for parent in (relative.parent, *relative.parents):
            if parent == Path('.'):
                continue
            expected.setdefault(parent, set())
        expected.setdefault(Path('.'), set())
        for parent, child in zip((Path('.'), *list(reversed(relative.parents[:-1]))), relative.parts):
            expected.setdefault(parent, set()).add(child)
        s = signature(root / relative)
        require(s['st_size'] == entry['bytes'], 'INPUT_SIZE_CHANGED:' + entry['path'])
    # Walk only explicitly named ancestor directories, not another tree.
    for relative, children in expected.items():
        directory = no_symlink(root / relative)
        actual = set()
        with os.scandir(directory) as scan:
            for item in scan:
                require(not item.is_symlink(), 'SYMLINK_ENTRY')
                actual.add(item.name)
                require(len(actual) <= len(children), 'UNEXPECTED_INPUT_ENTRY')
        require(actual == children, 'INPUT_LAYOUT_MISMATCH:' + str(relative))


def copy_inputs(source, destination, entries, deadline):
    exact_layout(source, entries)
    require(not os.path.lexists(destination), 'INPUT_DESTINATION_EXISTS')
    destination.mkdir(parents=True, exist_ok=False)
    receipts = []
    for entry in entries:
        check_deadline(deadline)
        src, dst = source / entry['path'], destination / entry['path']
        before = signature(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        count = 0
        with os.fdopen(os.open(src, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as inp, dst.open('xb') as out:
            opened = os.fstat(inp.fileno())
            require((opened.st_dev, opened.st_ino) == (before['st_dev'], before['st_ino']), 'OPEN_IDENTITY_CHANGED')
            while True:
                check_deadline(deadline)
                chunk = inp.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                require(count <= entry['bytes'], 'INPUT_GREW')
                digest.update(chunk)
                out.write(chunk)
        require(count == entry['bytes'] and signature(src) == before, 'SOURCE_CHANGED_DURING_COPY')
        dst.chmod(0o444)
        after = stream_hash(dst, deadline, entry['bytes'])
        require(after['sha256'] == digest.hexdigest(), 'COPY_HASH_MISMATCH')
        require(after['st_nlink'] == 1 and (after['st_dev'], after['st_ino']) != (before['st_dev'], before['st_ino']), 'NOT_INDEPENDENT_COPY')
        receipts.append(dict(path=entry['path'], source=dict(before, sha256=digest.hexdigest()), copy=after))
    directories = {destination}
    for entry in entries:
        path = destination / entry['path']
        directories.update(p for p in path.parents if p == destination or destination in p.parents)
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        directory.chmod(0o555)
    exact_layout(destination, entries)
    return receipts


def post_inputs(source, destination, receipts, deadline):
    for row in receipts:
        for kind, root in (('source', source), ('copy', destination)):
            actual = stream_hash(root / row['path'], deadline, row[kind]['st_size'])
            require(actual == row[kind], 'INPUT_CHANGED_AFTER_DECODE:' + kind + ':' + row['path'])


def scratch_bytes(root, excluded, entry_cap):
    total, seen = 0, 0
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as scan:
            for item in scan:
                seen += 1
                require(seen <= entry_cap, 'SCRATCH_ENTRY_CAP')
                require(not item.is_symlink(), 'SCRATCH_SYMLINK')
                path = Path(item.path)
                if item.is_dir(follow_symlinks=False):
                    pending.append(path)
                else:
                    s = item.stat(follow_symlinks=False)
                    require(stat.S_ISREG(s.st_mode), 'SCRATCH_SPECIAL_FILE')
                    if path in excluded and isinstance(excluded, dict):
                        require(s.st_size == excluded[path] and s.st_mode & 0o222 == 0 and s.st_nlink == 1,
                                'READONLY_INPUT_IDENTITY_CHANGED')
                    if path not in excluded:
                        total += s.st_size
    return total


def proc_rows():
    rows = {}
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        require(len(rows) < 65536, 'PROCESS_CENSUS_CAP')
        try:
            parts = (item / 'stat').read_text().rsplit(') ', 1)[1].split()
            rows[int(item.name)] = dict(pid=int(item.name), state=parts[0], ppid=int(parts[1]),
                                       pgid=int(parts[2]), sid=int(parts[3]), start=int(parts[19]))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return rows


def owned_descendants(rows, leader, known, supervisor=None):
    owned = {pid: row for pid, row in rows.items()
             if pid in known and row['start'] == known[pid] and row['state'] != 'Z'}
    parents = set(owned)
    if supervisor is not None:
        # This fixed driver has only one active child. Subreaping retains fast double-forks.
        parents.add(supervisor)
    changed = True
    while changed:
        changed = False
        for pid, row in rows.items():
            if row['ppid'] in parents and pid not in parents and row['state'] != 'Z':
                owned[pid] = row
                parents.add(pid)
                changed = True
    return owned


def stop_owned(child, known, grace):
    events = []
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 1.0)):
        rows = proc_rows()
        owned = owned_descendants(rows, child.pid, known, os.getpid())
        known.update({pid: row['start'] for pid, row in owned.items()})
        # Only the session/group created by this exact Popen, plus authenticated descendants.
        leader = rows.get(child.pid)
        if leader and child.pid in known and leader['start'] == known[child.pid]:
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
        for pid, row in owned.items():
            if pid == child.pid:
                continue
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        events.append(dict(signal=int(sig), identities=list(owned.values())))
        until = time.monotonic() + wait
        while time.monotonic() < until:
            child.poll()
            for pid in known:
                if pid != child.pid:
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
            if child.returncode is not None and not owned_descendants(proc_rows(), child.pid, known, os.getpid()):
                break
            time.sleep(0.02)
    child.poll()
    require(child.returncode is not None and not owned_descendants(proc_rows(), child.pid, known, os.getpid()), 'OWNED_PROCESSES_NOT_QUIESCENT')
    return events


def run_bounded(command, env, root, excluded, lock_fd, limits, deadline, stdout_path, stderr_path):
    """One fixed-command caller; argv flexibility exists only for tiny stdlib fixtures."""
    cpu = min(os.sched_getaffinity(0))
    # Linux self-process setting only; no SDK/device/service changes. Unsupported => no spawn.
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(36, 1, 0, 0, 0) == 0, 'CHILD_SUBREAPER_UNAVAILABLE')
    def child_limits():
        os.sched_setaffinity(0, {cpu})
        resource.setrlimit(resource.RLIMIT_AS, (limits['address_space'],) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (limits['cpu_seconds'],) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits['file_size'],) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    started = time.monotonic()
    prior_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = dict(status='INCOMPLETE', command=command, selected_cpu=cpu,
                  retained_bytes=0, observed_pipe_bytes=0, scratch_peak_bytes=0,
                  scratch_sample_interval_seconds=limits['interval'], child_subreaper=True,
                  descendant_limitations='Single-child Linux subreaper plus PID/start-time census; no cgroup memory claim. Shared postflight is still mandatory.')
    child, known = None, {}
    selector = selectors.DefaultSelector()
    try:
        with stdout_path.open('xb') as out, stderr_path.open('xb') as err:
            child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                                     pass_fds=(lock_fd,), preexec_fn=child_limits)
            if limits.get('parent_final_cpu_hard'):
                resource.setrlimit(resource.RLIMIT_CPU, (55, limits['parent_final_cpu_hard']))
            rows = proc_rows()
            if child.pid in rows:
                known[child.pid] = rows[child.pid]['start']
            selector.register(child.stdout, selectors.EVENT_READ, out)
            selector.register(child.stderr, selectors.EVENT_READ, err)
            while selector.get_map() or child.poll() is None:
                now = time.monotonic()
                require(now < deadline, 'OVERALL_WORK_DEADLINE')
                require(now - started < limits['wall_seconds'], 'DECODER_WALL_LIMIT')
                rows = proc_rows()
                owned = owned_descendants(rows, child.pid, known, os.getpid())
                known.update({pid: row['start'] for pid, row in owned.items()})
                require(not (set(owned) - {child.pid}), 'UNEXPECTED_DECODER_DESCENDANT')
                size = scratch_bytes(root, excluded, limits['entry_cap'])
                result['scratch_peak_bytes'] = max(result['scratch_peak_bytes'], size)
                require(size <= limits['scratch_bytes'], 'GENERATED_SCRATCH_CAP')
                for key, _ in selector.select(limits['interval']):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    result['observed_pipe_bytes'] += len(chunk)
                    keep = max(0, limits['retained_bytes'] - result['retained_bytes'])
                    key.data.write(chunk[:keep])
                    key.data.flush()
                    result['retained_bytes'] += min(len(chunk), keep)
                    require(result['observed_pipe_bytes'] < limits['retained_bytes'], 'COMBINED_OUTPUT_CAP')
            size = scratch_bytes(root, excluded, limits['entry_cap'])
            result['scratch_peak_bytes'] = max(result['scratch_peak_bytes'], size)
            require(size <= limits['scratch_bytes'], 'GENERATED_SCRATCH_CAP')
            result['returncode'] = child.wait(timeout=1)
            require(result['returncode'] == 0, 'DECODER_NONZERO_EXIT')
            result['status'] = 'PROCESS_EXIT_ZERO_COVERAGE_UNKNOWN'
    except BaseException as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        result['status'] = 'REFUSED_OR_INCOMPLETE'
    finally:
        selector.close()
        if child is not None:
            try:
                result['termination'] = stop_owned(child, known, limits['grace'])
            finally:
                child.stdout.close()
                child.stderr.close()
            result['returncode'] = child.returncode
            result['observed_process_identities'] = known
        result['wall_seconds'] = time.monotonic() - started
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result['children_cumulative_cpu_seconds'] = usage.ru_utime + usage.ru_stime
        result['child_cpu_seconds'] = usage.ru_utime + usage.ru_stime - prior_usage.ru_utime - prior_usage.ru_stime
        result['children_max_rss_kib'] = usage.ru_maxrss
        result['scratch_peak_overshoot_bytes'] = max(0, result['scratch_peak_bytes'] - limits['scratch_bytes'])
    return result
