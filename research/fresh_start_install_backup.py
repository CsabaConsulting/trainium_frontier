#!/usr/bin/env python3
"""Install/update only this campaign's cron entry, preserving other user jobs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from fresh_start_backup import ROOT, atomic_json


def main():
    directory = ROOT / 'out/fresh-start-backup'
    receipt = json.loads((directory / 'latest.json').read_text())
    assert receipt['pushed'] and receipt['remote_retrieval_complete'], \
        'An actual successful unattended backup must precede installation'
    wrapper = ROOT / 'research/fresh_start_backup_cron.sh'
    subprocess.run(['sh', '-n', str(wrapper)], check=True)
    current = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    assert current.returncode == 0 or (current.returncode == 1 and 'no crontab' in current.stderr)
    before = current.stdout if current.returncode == 0 else ''
    marker = '# trainium-frontier-research-backup'
    entry = f'*/10 * * * * /bin/sh {wrapper} {marker}'
    lines = [s for s in before.splitlines() if not s.endswith(marker)]
    after = '\n'.join([*lines, entry]) + '\n'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    old = directory / f'crontab-before-{stamp}.txt'
    new = directory / f'crontab-installed-{stamp}.txt'
    old.write_text(before)
    new.write_text(after)
    subprocess.run(['crontab', str(new)], check=True)
    actual = subprocess.check_output(['crontab', '-l'], text=True)
    assert actual == after, 'Installed crontab differs from reviewed entry'
    result = dict(installed_at=stamp,entry=entry,
                  previous_crontab=str(old.relative_to(ROOT)),
                  previous_sha256=hashlib.sha256(before.encode()).hexdigest(),
                  installed_sha256=hashlib.sha256(actual.encode()).hexdigest(),
                  preceding_verified_backup=receipt,
                  independent_of_gui=True,existing_jobs_preserved=True)
    output = ROOT / 'research/run-logs' / f'fs-backup-cron-install-{stamp}.json'
    atomic_json(output,result)
    print(json.dumps(dict(installed=True,entry=entry,receipt=str(output))))


if __name__ == '__main__':
    main()
