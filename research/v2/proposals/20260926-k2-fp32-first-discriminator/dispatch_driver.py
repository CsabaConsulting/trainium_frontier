"""Small detached receipt wrapper for the unchanged exact outer flock command."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

LOCK = '/home/ubuntu/.cache/trainium_freshstart/device0.lock'


def run(root):
    command = ['/usr/bin/flock', '-n', LOCK, '/usr/bin/python3', '-I', '-B',
               str(root / 'host_driver.py')]
    result = {'command': command, 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'status': 'OUTER_COMMAND_FAILED', 'returncode': None}
    try:
        with (root / 'outer-flock.log').open('x') as out:
            result['returncode'] = subprocess.call(command, stdin=subprocess.DEVNULL,
                                                   stdout=out, stderr=subprocess.STDOUT)
        if result['returncode'] == 0:
            result['status'] = 'OUTER_COMMAND_EXITED'
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        result['ended_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        result['under_lock_receipt_exists'] = (root / 'under-lock-idle.json').is_file()
        result['driver_terminal_exists'] = (root / 'terminal.json').is_file()
        with (root / 'outer-flock-exit.json').open('x') as out:
            json.dump(result, out, indent=2, sort_keys=True); out.write('\n')
    return result['returncode']


if __name__ == '__main__':
    sys.exit(run(Path(__file__).resolve().parent))
