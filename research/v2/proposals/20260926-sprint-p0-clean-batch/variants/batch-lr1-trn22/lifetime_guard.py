"""One-job Linux descendant guardian; no dispatch, queue, science or device writes.

The existing driver remains the authenticated flock child. Becoming a subreaper
before spawn makes orphaned descendants (including setsid/double-fork children)
our children. ECHILD, not an empty process-name scan, proves lineage quiescence.
Only pidfds opened for verified children are signalled. Unknown state parks with
the original inherited lock open. SIGKILL of the guardian/host loss cannot be
caught by userspace; no such action is part of this fixed-job protocol.
"""
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import stat
import subprocess
import time


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]), 'pgid': int(fields[2]),
            'sid': int(fields[3]), 'start_ticks': int(fields[19])}


def require_original_lock(fd, path, expected):
    current, opened = Path(path).lstat(), os.fstat(fd)
    for value in (current, opened):
        if (not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid()
                or (value.st_dev, value.st_ino) != tuple(expected)):
            raise RuntimeError('ORIGINAL_LOCK_IDENTITY_CHANGED')
    lines = Path(f'/proc/self/fdinfo/{fd}').read_text().splitlines()
    if not any(line.startswith('lock:') and 'FLOCK' in line and 'WRITE' in line
               for line in lines):
        raise RuntimeError('ORIGINAL_LOCK_NOT_HELD')


class Guardian:
    def __init__(self, lock_fd, lock_path, expected, receipt, grace=60.0, kill_wait=10.0):
        require_original_lock(lock_fd, lock_path, expected)
        self.fd, self.lock_path, self.expected = lock_fd, lock_path, expected
        self.receipt = None
        self.owned, self.statuses = {}, {}
        self.signal_received = None
        self.audit_error = None
        try:
            self.receipt = open(receipt, 'x', buffering=1)
        except BaseException as exc:
            self.audit_error = repr(exc)
        self.job_started = False
        self.grace, self.kill_wait = grace, kill_wait
        libc = ctypes.CDLL(None, use_errno=True)
        enabled = ctypes.c_int()
        if (libc.prctl(36, 1, 0, 0, 0) != 0
                or libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0
                or enabled.value != 1):
            raise RuntimeError('SUBREAPER_UNAVAILABLE')
        # Require pidfds before any job can start; there is no numeric-PID fallback.
        probe = os.pidfd_open(os.getpid())
        try:
            signal.pidfd_send_signal(probe, 0)
        finally:
            os.close(probe)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
            signal.signal(sig, self._signal)
        self.event('guardian_started', identity=identity(os.getpid()),
                   lock_device=expected[0], lock_inode=expected[1], subreaper=True)

    def event(self, event, **fields):
        try:
            require_original_lock(self.fd, self.lock_path, self.expected)
            self.receipt.write(json.dumps(dict(event=event, monotonic=time.monotonic(),
                                              lock_held=True, **fields), sort_keys=True) + '\n')
            self.receipt.flush()
        except BaseException as exc:
            # Receipt failures must NEVER interrupt descendant cleanup/reaping.
            self.audit_error = self.audit_error or repr(exc)

    def _signal(self, sig, _frame):
        if self.signal_received is None:
            self.signal_received = sig

    def check_cancelled(self):
        if self.audit_error is not None:
            raise RuntimeError('LIFETIME_AUDIT_FAILED: ' + self.audit_error)
        if self.signal_received is not None:
            raise InterruptedError('GUARDIAN_SIGNAL_' + str(self.signal_received))

    def discover(self):
        """Follow actual kernel child links; names, labels and groups never own a PID."""
        for key, (pidfd, row) in list(self.owned.items()):
            if select.select([pidfd], [], [], 0)[0]:
                self.event('owned_exited', identity=row)
                os.close(pidfd)
                del self.owned[key]  # Compile subprocess turnover must not exhaust FDs.
        pending = [(os.getpid(), None)]
        seen = set()
        while pending:
            parent, parent_start = pending.pop()
            if parent in seen:
                continue
            seen.add(parent)
            parent_fd = None if parent_start is None else self.owned[(parent, parent_start)][0]
            if parent_fd is not None and select.select([parent_fd], [], [], 0)[0]:
                continue
            try:
                before = identity(parent)
                if parent_start is not None and before['start_ticks'] != parent_start:
                    continue
                children = []
                for task in Path(f'/proc/{parent}/task').iterdir():
                    try:
                        children += (task/'children').read_text().split()
                    except FileNotFoundError:
                        continue
                if identity(parent)['start_ticks'] != before['start_ticks']:
                    continue
            except (FileNotFoundError, ProcessLookupError):
                continue  # Any orphans become our children, independently of session/group.
            for word in children:
                pid = int(word)
                pidfd = None
                try:
                    row = identity(pid)
                    if row['ppid'] != parent or row['state'] == 'Z':
                        continue
                    key = (pid, row['start_ticks'])
                    if key not in self.owned:
                        pidfd = os.pidfd_open(pid)
                        after = identity(pid)
                        if (after['start_ticks'] != row['start_ticks'] or after['ppid'] != parent
                                or (parent_fd is not None and select.select([parent_fd], [], [], 0)[0])):
                            os.close(pidfd)
                            pidfd = None
                            continue
                        self.owned[key] = (pidfd, row)
                        pidfd = None
                        self.event('owned_identity', identity=row)
                    pending.append(key)
                except (FileNotFoundError, ProcessLookupError):
                    if pidfd is not None:
                        os.close(pidfd)

    def reap(self):
        """Return True only on kernel ECHILD (no live OR zombie direct child)."""
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return True
            if pid == 0:
                return False
            code = os.waitstatus_to_exitcode(status)
            self.statuses[pid] = code
            self.event('reaped', pid=pid, wait_status=status, returncode=code)

    def signal_owned(self, sig):
        self.discover()
        for pidfd, row in self.owned.values():
            if select.select([pidfd], [], [], 0)[0]:
                continue  # pidfd still refers to the old process, even after PID reuse.
            try:
                signal.pidfd_send_signal(pidfd, sig)
                self.event('signal_owned', signal=sig, identity=row)
            except ProcessLookupError:
                pass

    def quiesce(self, reason):
        self.discover()
        if self.reap():
            self.event('lineage_quiescent', reason=reason, proof='waitpid_ECHILD')
            return
        self.event('cleanup_started', reason=reason)
        for sig, seconds in ((signal.SIGTERM, self.grace), (signal.SIGKILL, self.kill_wait)):
            deadline = time.monotonic() + seconds
            while True:
                self.signal_owned(sig)
                if self.reap():
                    self.event('lineage_quiescent', reason=reason, proof='waitpid_ECHILD')
                    return
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        self.park('OWNED_DESCENDANTS_NOT_QUIESCENT: ' + reason)

    def park(self, error):
        """Terminal failure; retain lock forever, never launch a successor or retry."""
        try:
            self.event('FAILED_LOCK_RETAINED', error=str(error))
        except BaseException:
            pass
        try:
            os.write(2, ('FAILED_LOCK_RETAINED: ' + str(error) + '\n').encode())
        except BaseException:
            pass
        while True:
            try:
                time.sleep(1)
            except BaseException:
                pass

    def run(self, command, *, cwd, env, out):
        self.check_cancelled()
        require_original_lock(self.fd, self.lock_path, self.expected)
        try:
            self.discover()
            if not self.reap():
                raise RuntimeError('PREFLIGHT_HELPER_DESCENDANTS_REMAIN')
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=subprocess.STDOUT, pass_fds=(self.fd,))
            self.job_started = True
            self.statuses.pop(process.pid, None)  # A prior phase's reaped PID can be reused.
            self.event('phase_spawned', pid=process.pid, identity=identity(process.pid), command=command)
            while True:
                self.discover()
                empty = self.reap()
                if self.audit_error is not None:
                    self.quiesce('LIFETIME_AUDIT_FAILED: ' + self.audit_error)
                    raise RuntimeError('LIFETIME_AUDIT_FAILED: ' + self.audit_error)
                result = self.statuses.get(process.pid)
                if result is not None:
                    process.returncode = result
                    self.event('phase_exit', pid=process.pid, returncode=result)
                    if not empty:
                        self.quiesce('PHASE_EXIT_WITH_DESCENDANTS_' + str(result))
                        if result == 0:
                            result = 125
                    if self.signal_received is not None and result == 0:
                        result = 128 + self.signal_received
                    return result
                if self.signal_received is not None:
                    self.event('guardian_signal', signal=self.signal_received)
                    self.quiesce('GUARDIAN_SIGNAL_' + str(self.signal_received))
                    process.returncode = self.statuses.get(process.pid)
                    return 128 + self.signal_received
                if empty:
                    raise RuntimeError('PHASE_LEADER_STATUS_MISSING')
                time.sleep(0.05)
        except BaseException as exc:
            try:
                self.quiesce(type(exc).__name__ + ': ' + str(exc))
            except BaseException as cleanup:
                self.park('PRIMARY_ERROR: ' + repr(exc) + '; CLEANUP_UNKNOWN: ' + repr(cleanup))
            raise

    def finish(self, postflight, independent_idle):
        """Reporting failure is a failed result, not authority to retain idle hardware.

        Release still requires kernel lineage proof, separate fresh machine-idle
        checks and the original held inode. Return False on any reporting failure.
        The outer flock releases only when this synchronous driver exits.
        """
        reporting_error = None
        try:
            self.quiesce('DRIVER_FINALLY')
            try:
                postflight()
            except BaseException as exc:
                reporting_error = repr(exc)
            # A reporting callback can have created helpers before failing.
            self.quiesce('AFTER_REPORTING_POSTFLIGHT')
            require_original_lock(self.fd, self.lock_path, self.expected)
            independent_idle()
            # Idle helpers themselves may not leave descendants.
            self.discover()
            if not self.reap():
                raise RuntimeError('POSTFLIGHT_CHILD_REMAINS')
            require_original_lock(self.fd, self.lock_path, self.expected)
            self.event('release_ready', proof='waitpid_ECHILD_AND_INDEPENDENT_IDLE_PASS',
                       signal_received=self.signal_received, reporting_error=reporting_error,
                       audit_error=self.audit_error)
            require_original_lock(self.fd, self.lock_path, self.expected)
        except BaseException as exc:
            try:
                self.quiesce('POSTFLIGHT_FAILURE: ' + repr(exc))
            except BaseException:
                pass
            self.park('POSTFLIGHT_OR_QUIESCENCE_FAILED: ' + repr(exc))
        for pidfd, _ in self.owned.values():
            os.close(pidfd)
        try:
            self.receipt.close()
        except BaseException as exc:
            reporting_error = reporting_error or repr(exc)
        return reporting_error is None and self.audit_error is None
