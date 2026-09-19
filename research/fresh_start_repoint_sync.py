#!/usr/bin/env python3
"""Move one exact owned local log watcher between immutable remote worktrees.

This never signals a remote experiment, a compiler, or another user's process.
Only the supplied old watcher argv, local UID, repository cwd and start ticks
authorize retirement. A pidfd prevents PID reuse from redirecting the signal.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys

from fresh_start_sync import ROOT


def arguments(host, wave, rounds, interval):
    return ['research/fresh_start_sync.py', '--host', host, '--remote',
            f'/home/ubuntu/trainium_freshstart_20260918_wave{wave}',
            '--prefix', f'fs-w{wave}-', '--rounds', str(rounds), '--interval', str(interval)]


def inspect(path, expected):
    argv = [part.decode() for part in (path / 'cmdline').read_bytes().split(b'\0') if part]
    if (argv[1:] != expected or path.stat().st_uid != os.getuid()
            or (path / 'cwd').resolve() != ROOT):
        return None
    ticks = (path / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    return dict(pid=int(path.name), start_ticks=ticks, argv=argv)


def matches(expected):
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            record = inspect(path, expected)
            if record is not None:
                found.append(record)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('trn21', 'trn22'), required=True)
    parser.add_argument('--from-wave', type=int, required=True)
    parser.add_argument('--to-wave', type=int, required=True)
    parser.add_argument('--rounds', type=int, default=90)
    parser.add_argument('--interval', type=int, default=45)
    args = parser.parse_args()
    if args.from_wave == args.to_wave or min(args.from_wave, args.to_wave, args.rounds, args.interval) < 1:
        parser.error('Use distinct positive waves and positive watcher limits')
    old = arguments(args.host, args.from_wave, args.rounds, args.interval)
    new = arguments(args.host, args.to_wave, args.rounds, args.interval)
    already = matches(new)
    retired = []
    for record in matches(old):
        try:
            fd = os.pidfd_open(record['pid'])
            try:
                assert inspect(Path('/proc') / str(record['pid']), old) == record
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                retired.append(record)
            finally:
                os.close(fd)
        except (FileNotFoundError, ProcessLookupError):
            continue
    now = datetime.now(timezone.utc)
    receipt = dict(at=now.isoformat(), host=args.host, old_argv=old, new_argv=new,
                   retired=retired, already_running=already, remote_processes_touched=False)
    destination = ROOT / 'research/run-logs' / (
        'fs-sync-repoint-' + args.host + '-' + now.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    destination.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    if already:
        return
    os.chdir(ROOT)
    os.execv(sys.executable, [sys.executable, *new])


if __name__ == '__main__':
    main()
