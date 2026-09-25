"""Run an explicitly supplied local command and retain its raw streams and exit."""
import datetime
import json
import subprocess
import sys
from pathlib import Path

label, *command = sys.argv[1:]
out = Path('/work')
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
with (out / (label + '.stdout.txt')).open('wb') as stdout, (out / (label + '.stderr.txt')).open('wb') as stderr:
    p = subprocess.run(command, cwd=out, stdout=stdout, stderr=stderr)
record = dict(command=command, cwd=str(out), started_utc=started,
              ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), exit_code=p.returncode,
              stdout=label + '.stdout.txt', stderr=label + '.stderr.txt')
(out / (label + '.command.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
sys.exit(p.returncode)
