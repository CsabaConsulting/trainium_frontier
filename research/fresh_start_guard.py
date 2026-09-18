#!/usr/bin/env python3
"""Run one manifested command and always attempt scoped cleanup/postflight."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from fresh_start_host import (HOST_STATE, MARKER, cleanup_owned, health_failures,
                              snapshot, utc_now, write_json)


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
    interrupted = []

    def handle_signal(sig, _frame):
        interrupted.append(sig)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    returncode = None
    error = None
    try:
        process = subprocess.Popen(request['command'], cwd=remote, env=env,
                                   start_new_session=True, pass_fds=(args.lock_fd,))
        identity = dict(label=label, guard_pid=os.getpid(), child_pid=process.pid,
                        child_session=process.pid, started_at=start, command=request['command'])
        write_json(remote / 'research/run-logs' / (label + '.process.json'), identity)
        returncode = process.wait()
    except BaseException as exc:
        error = repr(exc)
    finally:
        # The unique inherited marker covers children which started their own
        # session or were reparented. Unknown process holders are never killed.
        cleanup = cleanup_owned(label)
        time.sleep(2)
        try:
            post = snapshot(remote, start)
            failures = health_failures(post, (os.getpid(), os.getppid()))
        except Exception as exc:
            post = dict(snapshot_error=repr(exc))
            failures = ['postflight snapshot failed']
        if cleanup['remaining']:
            failures.append('owned descendants remain after bounded cleanup')
        record = dict(protocol_version=2, label=label, remote=str(remote), stage='postflight',
                      source_commit=request['expected_commit'], command_returncode=returncode,
                      guard_error=error, interrupted_signals=interrupted, cleanup=cleanup,
                      health=post, failures=failures, host_clean=not failures,
                      completed_at=utc_now(), artifacts_preserved_on_remote=True,
                      local_preservation_pending=True)
        path = remote / 'research/run-logs' / (label + '.postflight.json')
        write_json(path, record)
        write_json(HOST_STATE, dict(label=label, remote=str(remote), postflight=str(path),
                                   host_clean=record['host_clean'], completed_at=record['completed_at']), replace=True)
        print('EXPERIMENT_POSTFLIGHT ' + json.dumps(dict(label=label, returncode=returncode,
              host_clean=record['host_clean'], cleanup_actions=len(cleanup['actions']),
              postflight=str(path))), flush=True)
        os.close(args.lock_fd)
    return (returncode if returncode is not None and returncode >= 0 else 1) or (1 if failures else 0)


if __name__ == '__main__':
    sys.exit(main())
