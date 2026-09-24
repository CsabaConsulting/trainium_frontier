#!/usr/bin/env python3
"""Private-session lifecycle primitives inherited from verified core f58ab4a0."""

import ctypes
import os
from pathlib import Path
import signal
import subprocess
import time

import artifact_io as aio

PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37


def enable_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise SystemExit(f"cannot enable child subreaper: errno={ctypes.get_errno()}")
    observed = ctypes.c_int()
    if libc.prctl(PR_GET_CHILD_SUBREAPER, ctypes.byref(observed), 0, 0, 0) != 0 or observed.value != 1:
        raise SystemExit("child subreaper verification failed")


def reap_group(pgid):
    reaped = []
    while True:
        try:
            pid, _status = os.waitpid(-pgid, os.WNOHANG)
        except ChildProcessError:
            break
        if pid <= 0:
            break
        reaped.append(pid)
    return reaped


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise SystemExit(f"cannot inspect process group {pgid}") from error


def wait_group_gone(pgid, seconds, reaped):
    deadline = time.monotonic() + seconds
    while True:
        reaped.extend(reap_group(pgid))
        if not group_exists(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def quiesce_group(process, term_grace=15.0, kill_grace=10.0):
    pgid = process.pid
    leader_returncode = process.poll()
    reaped = []
    cleanup_required = group_exists(pgid)
    term_sent = False
    kill_sent = False
    if cleanup_required:
        try:
            os.killpg(pgid, signal.SIGTERM)
            term_sent = True
        except ProcessLookupError:
            pass
        if leader_returncode is None:
            try:
                leader_returncode = process.wait(timeout=term_grace)
            except subprocess.TimeoutExpired:
                pass
        gone = wait_group_gone(pgid, term_grace, reaped)
        if not gone:
            try:
                os.killpg(pgid, signal.SIGKILL)
                kill_sent = True
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    leader_returncode = process.wait(timeout=kill_grace)
                except subprocess.TimeoutExpired:
                    pass
            gone = wait_group_gone(pgid, kill_grace, reaped)
        if not gone:
            raise SystemExit(f"process group {pgid} survived TERM/KILL escalation")
    if process.poll() is None:
        try:
            leader_returncode = process.wait(timeout=0.2)
        except subprocess.TimeoutExpired as error:
            raise SystemExit("leader remained after process-group quiescence") from error
    else:
        leader_returncode = process.returncode
    reaped.extend(reap_group(pgid))
    if group_exists(pgid):
        raise SystemExit(f"process group {pgid} reappeared after quiescence")
    return {
        "pgid": pgid, "leader_returncode": leader_returncode,
        "cleanup_required": cleanup_required, "term_sent": term_sent,
        "kill_sent": kill_sent, "quiescent": True,
        "reaped_descendants": sorted(set(reaped)),
        "term_grace_seconds": float(term_grace), "kill_grace_seconds": float(kill_grace),
    }


def effective_returncode(process, lifecycle):
    value = process.returncode
    if lifecycle["cleanup_required"] and value == 0:
        return 125
    return value


def run_capture(argv, cwd, env, timeout, term_grace=15.0, kill_grace=10.0):
    enable_subreaper()
    started = time.monotonic()
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        lifecycle = quiesce_group(process, term_grace, kill_grace)
        output, _ = process.communicate()
    else:
        lifecycle = quiesce_group(process, term_grace, kill_grace)
    return effective_returncode(process, lifecycle), output, time.monotonic() - started, timed_out, lifecycle


def run_training(argv, sandbox, env, timeout, term_grace=15.0, kill_grace=10.0):
    enable_subreaper()
    sandbox = Path(sandbox)
    raw_log = sandbox / "training.raw.log"
    step_log = sandbox / "steps.jsonl"
    started = time.monotonic()
    observed = []
    with raw_log.open("xb") as output:
        process = subprocess.Popen(argv, cwd=sandbox, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout:
                timed_out = True
                break
            try:
                steps = step_log.read_text().splitlines()
            except FileNotFoundError:
                steps = []
            while len(observed) < len(steps):
                observed.append({"step": int(steps[len(observed)]), "observed_seconds": elapsed})
            time.sleep(0.1)
        lifecycle = quiesce_group(process, term_grace, kill_grace)
        output.flush()
        os.fsync(output.fileno())
    return (effective_returncode(process, lifecycle), aio.regular_bytes(raw_log),
            time.monotonic() - started, timed_out, observed, lifecycle)
