"""Fixed-job Linux supervision adapted from decoder_support, not a launcher API.

No RLIMIT_AS/CPU/affinity or no-descendant donor restriction is copied.
pidfds demonstrate observed exits, including descendants reaped by their parents.
"""
import ctypes
import os
from pathlib import Path
import select
import signal
import stat
import time
from common import GIB, MIB, identity, require


def subreaper():
    require(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0, 'SUBREAPER_UNAVAILABLE')


def census():
    rows = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            require(len(rows) < 65536, 'PROCESS_CENSUS_CAP')
            try:
                row = identity(int(entry.name))
                rows[row['pid']] = row
            except (FileNotFoundError, ProcessLookupError):
                continue
    return rows


def tree_bytes(root, cap=65536):
    total, entries, pending = 0, 0, [Path(root)]
    while pending:
        for item in os.scandir(pending.pop()):
            entries += 1
            require(entries <= cap, 'ENTRY_CAP')
            s = item.stat(follow_symlinks=False)
            require(not stat.S_ISLNK(s.st_mode), 'SCRATCH_SYMLINK')
            if stat.S_ISDIR(s.st_mode):
                pending.append(Path(item.path))
            else:
                require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'SPECIAL_OR_HARDLINK')
                total += s.st_size
    return total, entries


def memavailable():
    rows = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(rows['MemAvailable'].split()[0]) * 1024


def free_bytes(path):
    fs = os.statvfs(path)
    return fs.f_bavail * fs.f_frsize


def writable_paths(rows, root):
    """Fail on observed regular-file writes escaping this job's accounted trees.

    This is sampled descriptor evidence, not a syscall audit. Closed descriptors
    between samples are not claimed observable. Native device/pipe/socket handles
    are not regular disk files. Supervisor inherited stdio is Operator-owned.
    """
    observed = []
    allowed = tuple(str(root/name)+'/' for name in ('artifacts','cache','tmp','logs'))
    for row in rows:
        directory = Path('/proc/%d/fd' % row['pid'])
        for count, fd in enumerate(directory.iterdir()):
            require(count < 65536, 'FD_CENSUS_CAP')
            if row['pid'] == os.getpid() and fd.name in ('0','1','2'):
                continue
            try:
                details = dict(line.split(':',1) for line in Path('/proc/%d/fdinfo/%s' % (row['pid'],fd.name)).read_text().splitlines())
                flags = int(details['flags'].strip(),8)
                if flags & os.O_ACCMODE == os.O_RDONLY:
                    continue
                if not stat.S_ISREG(fd.stat().st_mode):
                    continue
                target = os.readlink(fd)
                require(target.startswith(allowed) and not target.endswith(' (deleted)'), 'UNACCOUNTED_WRITABLE_PATH')
                observed.append(dict(pid=row['pid'],start=row['start'],fd=int(fd.name),path=target))
            except FileNotFoundError:
                continue  # Descriptor closed during census; not a zero-RSS inference.
    return observed


class Processes:
    def __init__(self):
        self.root = os.getpid()
        self.known, self.exits = {}, []

    def observe(self):
        rows = census()
        owned = {self.root}
        owned.update(pid for pid, (start, fd) in self.known.items()
                     if pid in rows and rows[pid]['start'] == start)
        changed = True
        while changed:
            more = {pid for pid, row in rows.items() if row['ppid'] in owned}
            changed = bool(more - owned)
            owned |= more
        for pid in owned - {self.root}:
            row = rows[pid]
            require(pid not in self.known or self.known[pid][0] == row['start'], 'OWNED_PID_REUSED')
            if pid not in self.known:
                # If it disappears before pidfd_open, exit cannot be demonstrated:
                # fail closed, rather than assuming zero RSS.
                fd = os.pidfd_open(pid)
                require(identity(pid)['start'] == row['start'], 'PID_IDENTITY_RACE')
                self.known[pid] = row['start'], fd
        active, total = [], 0
        for pid in sorted(owned):
            row = rows[pid]
            if row['state'] == 'Z':
                continue
            try:
                status = dict(line.split(':', 1) for line in Path('/proc/%d/status' % pid).read_text().splitlines())
                rss = int(status['VmRSS'].split()[0]) * 1024
                require(identity(pid)['start'] == row['start'], 'PID_REUSED_DURING_RSS')
            except (FileNotFoundError, KeyError, ProcessLookupError):
                require(pid in self.known and select.select([self.known[pid][1]], [], [], 0)[0], 'AMBIGUOUS_LIVE_RSS')
                continue
            total += rss
            active.append(dict(row, rss_bytes=rss))
        for pid, (start, fd) in self.known.items():
            if pid not in rows or rows[pid]['start'] != start or rows[pid]['state'] == 'Z':
                require(select.select([fd], [], [], 0)[0], 'DISAPPEARED_WITHOUT_EXIT')
                key = (pid, start)
                if key not in self.exits:
                    self.exits.append(key)
        return active, total

    def signal_owned(self, sig):
        # Signal by pidfd, not a potentially reused PID/group number.
        self.observe()
        for pid, (start, fd) in self.known.items():
            if not select.select([fd], [], [], 0)[0]:
                signal.pidfd_send_signal(fd, sig)

    def reap(self):
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                break

    def alive(self):
        self.observe()
        return [dict(pid=p, start=s) for p, (s, fd) in self.known.items()
                if not select.select([fd], [], [], 0)[0]]

    def quiescent(self):
        """Final kernel child proof, including helpers created after earlier stop."""
        require(not self.alive(), 'OWNED_CHILDREN_STILL_LIVE')
        self.reap()
        require(not self.alive(), 'CHILD_APPEARED_DURING_REAP')
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
                require(pid != 0, 'KERNEL_CHILDREN_REMAIN')
            except ChildProcessError:
                return dict(kernel_echild=True, at=time.monotonic(), demonstrated_exits=self.exits.copy())

    def stop(self, term_deadline, kill_deadline):
        events = []
        for sig, deadline in ((signal.SIGTERM, term_deadline), (signal.SIGKILL, kill_deadline)):
            self.signal_owned(sig)
            events.append(dict(signal=int(sig), at=time.monotonic(), identities=list(self.known)))
            while time.monotonic() < deadline:
                self.reap()
                if not self.alive():
                    self.reap()
                    return dict(events=events, survivors=[])
                time.sleep(0.02)
        return dict(events=events, survivors=self.alive())


def guards(sample, previous=None):
    if previous is not None:
        require(0 <= sample['at'] - previous <= 2, 'MISSING_SAMPLE_OR_GAP')
    require(sample['memavailable'] >= 4 * GIB, 'MEMAVAILABLE_FLOOR')
    require(sample['rss'] <= 20 * GIB, 'SUMMED_RSS_CAP')
    require(min(sample['free']) >= 6.5 * GIB, 'DISK_FLOOR')
    require(sample['cache'] <= 6 * GIB, 'CACHE_TMP_CAP')
    require(sample['artifacts'] <= 2 * GIB, 'ARTIFACT_CAP')
    require(sample['logs'] <= 128 * MIB, 'LOG_METRIC_CAP')


def sample(root, processes):
    started = time.monotonic()
    rows, rss = processes.observe()
    paths = writable_paths(rows, root)
    sizes, counts = {}, {}
    for name in ('cache', 'tmp', 'artifacts', 'logs'):
        sizes[name], counts[name] = tree_bytes(root / name)
    require(sum(counts.values()) <= 65536, 'COMBINED_ENTRY_CAP')
    return dict(at=started, duration=time.monotonic() - started,
                memavailable=memavailable(), rss=rss, processes=rows,
                observed_writable_paths=paths,
                demonstrated_exits=processes.exits.copy(),
                free=[free_bytes(root / name) for name in sizes],
                cache=sizes['cache'] + sizes['tmp'], artifacts=sizes['artifacts'], logs=sizes['logs'])
