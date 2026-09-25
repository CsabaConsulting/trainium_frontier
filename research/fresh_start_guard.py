#!/usr/bin/env python3
"""Run one manifested command and record scoped, signal-free cleanup inventory."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from fresh_start_host import (HOST_STATE, MARKER, cleanup_owned, health_failures,
                              process_record, snapshot, utc_now, write_json)


def contain_failed_child(process, child_pidfd):
    """Boundedly stop the exact unreaped Popen child after a guard exception.

    If pidfd acquisition itself failed, Popen still owns the unreaped direct
    child. Its PID cannot be recycled until this single-threaded guard reaps
    it; Popen.terminate/kill are limited to that child, never a process group.
    This exceptional fallback is not used for descendants or prior runs.
    """
    result = dict(signals=[], signal_errors=[], quiescent=True,
                  fallback='unreaped_popen_direct_child' if child_pidfd is None else None)
    if process is None:
        return result
    try:
        if process.poll() is not None:
            return result
    except Exception as exc:
        result['signal_errors'].append('child liveness: ' + repr(exc))
        result['quiescent'] = False
        return result

    def send(sig):
        try:
            if child_pidfd is not None:
                signal.pidfd_send_signal(child_pidfd, sig, None, 0)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
            result['signals'].append(sig.name)
        except ProcessLookupError:
            pass
        except OSError as exc:
            result['signal_errors'].append(repr(exc))

    send(signal.SIGTERM)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            result['quiescent'] = False
        except Exception as exc:
            result['signal_errors'].append('child KILL wait: ' + repr(exc))
            result['quiescent'] = False
    except Exception as exc:
        result['signal_errors'].append('child TERM wait: ' + repr(exc))
        result['quiescent'] = False
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('request', type=Path)
    ap.add_argument('--lock-fd', type=int, required=True)
    args = ap.parse_args()
    request = json.loads(args.request.read_text())
    label, remote = request['label'], Path(request['remote'])
    start = utc_now()
    env = os.environ.copy()
    env[MARKER] = label
    process = None
    child_pidfd = None
    child_start_ticks = None
    interrupted = []
    interrupted_signal_errors = []
    exception_containment = None

    def term_bound_child():
        # The Popen child cannot be PID-reused before this guard reaps it.
        # pidfd_open is called immediately after spawn; later TERM uses only
        # that descriptor, never a numeric PID or an unscoped process group.
        if child_pidfd is None:
            interrupted_signal_errors.append('child pidfd unavailable')
            return
        try:
            signal.pidfd_send_signal(child_pidfd, signal.SIGTERM, None, 0)
        except ProcessLookupError:
            pass
        except OSError as exc:
            interrupted_signal_errors.append(repr(exc))

    def handle_signal(sig, _frame):
        interrupted.append(sig)
        if process is not None:
            term_bound_child()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    returncode = None
    error = None
    try:
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise RuntimeError('pidfd support is required before guard spawn')
        process = subprocess.Popen(request['command'], cwd=remote, env=env,
                                   start_new_session=True, pass_fds=(args.lock_fd,))
        child_pidfd = os.pidfd_open(process.pid, 0)
        child_record = process_record(process.pid)
        if (child_record is None or child_record['session'] != process.pid
                or child_record['process_group'] != process.pid
                or type(child_record['start_ticks']) is not int):
            raise RuntimeError('direct child birth/session identity unavailable')
        child_start_ticks = child_record['start_ticks']
        if interrupted:
            term_bound_child()
        identity = dict(label=label, guard_pid=os.getpid(), child_pid=process.pid,
                        child_session=process.pid, child_start_ticks=child_start_ticks,
                        started_at=start, command=request['command'])
        write_json(remote / 'research/run-logs' / (label + '.process.json'), identity)
        returncode = process.wait()
    except BaseException as exc:
        error = repr(exc)
        try:
            exception_containment = contain_failed_child(process, child_pidfd)
            if process is not None:
                returncode = process.returncode
        except BaseException as containment_error:
            exception_containment = dict(signals=[], signal_errors=[repr(containment_error)],
                                         quiescent=False, fallback=None)
    finally:
        # A public marker and a stale session number are inventory hints only.
        # The bound external monitor owns emergency escalation while live.
        time.sleep(2)
        cleanup = cleanup_owned(label, process.pid if process is not None else None,
                                child_start_ticks)
        try:
            post = snapshot(remote, start)
            failures = health_failures(post, (os.getpid(), os.getppid()))
        except Exception as exc:
            post = dict(snapshot_error=repr(exc))
            failures = ['postflight snapshot failed']
        if cleanup['remaining']:
            failures.append('run-marker, child-session, or post-child process remains; cleanup is HOLD-only')
        if error is not None or interrupted_signal_errors:
            failures.append('guard error or exact-child TERM forwarding failed')
        if exception_containment is not None and not exception_containment['quiescent']:
            failures.append('unreceipted direct child did not quiesce after bounded stop')
        record = dict(protocol_version=2, label=label, remote=str(remote), stage='postflight',
                      source_commit=request['expected_commit'], command_returncode=returncode,
                      child_start_ticks=child_start_ticks,
                      guard_error=error, interrupted_signals=interrupted,
                      interrupted_signal_errors=interrupted_signal_errors,
                      exception_containment=exception_containment, cleanup=cleanup,
                      health=post, failures=failures, host_clean=not failures,
                      completed_at=utc_now(), artifacts_preserved_on_remote=True,
                      local_preservation_pending=True)
        path = remote / 'research/run-logs' / (label + '.postflight.json')
        write_json(path, record)
        write_json(HOST_STATE, dict(label=label, remote=str(remote), postflight=str(path),
                                   child_session=process.pid if process is not None else None,
                                   child_start_ticks=child_start_ticks,
                                   host_clean=record['host_clean'], completed_at=record['completed_at']), replace=True)
        print('EXPERIMENT_POSTFLIGHT ' + json.dumps(dict(label=label, returncode=returncode,
              host_clean=record['host_clean'], cleanup_actions=len(cleanup['actions']),
              postflight=str(path))), flush=True)
        if child_pidfd is not None:
            os.close(child_pidfd)
            child_pidfd = None
        os.close(args.lock_fd)
    return (returncode if returncode is not None and returncode >= 0 else 1) or (1 if failures else 0)


if __name__ == '__main__':
    sys.exit(main())
