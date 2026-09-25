#!/usr/bin/env python3
"""Persist exit statuses and run both stdlib scripts with CPU/memory/time limits."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))


def main():
    receipts = []
    for script, stdout, stderr in [
        ('reconstruct_packet.py', 'reconstruction.stdout.txt', 'reconstruction.stderr.txt'),
        ('cpu_exactness_test.py', 'cpu_test.stdout.jsonl', 'cpu_test.stderr.txt'),
    ]:
        command = [sys.executable, str(WORK / script)]
        started = time.monotonic()
        with (WORK / stdout).open('wb') as out, (WORK / stderr).open('wb') as err:
            try:
                completed = subprocess.run(command, cwd=WORK, stdout=out, stderr=err,
                                           timeout=30, preexec_fn=limits, check=False)
                exit_code = completed.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                exit_code = None
                timed_out = True
        receipt = dict(command=command, exit_code=exit_code, timed_out=timed_out,
                       elapsed_seconds=time.monotonic() - started, stdout=stdout, stderr=stderr,
                       limits=dict(cpu_seconds=15, address_space_bytes=256*1024*1024,
                                   wall_seconds=30))
        receipts.append(receipt)
        (WORK / 'run_receipts.json').write_text(json.dumps(receipts, indent=2, allow_nan=False) + '\n')
        print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
        if timed_out or exit_code != 0:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
