#!/usr/bin/env python3
"""Install/update only this campaign's cron entry, preserving other user jobs."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from fresh_start_backup import ROOT, atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-artifacts',action='store_true')
    args=parser.parse_args()
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
    entries=[entry]
    if args.include_artifacts:
        artifacts=ROOT/'research/fresh_start_artifact_backup_cron.sh'
        subprocess.run(['sh','-n',str(artifacts)],check=True)
        verified=list((ROOT/'research/run-logs').glob('fs-artifact-backup-*.json'))
        assert any(json.loads(p.read_text()).get('all_github_digests_match') for p in verified), \
            'Verify an actual artifact upload before scheduling it'
        artifact_marker='# trainium-frontier-artifact-backup'
        lines=[s for s in lines if not s.endswith(artifact_marker)]
        entries.append(f'5,35 * * * * /bin/sh {artifacts} {artifact_marker}')
    after = '\n'.join([*lines, *entries]) + '\n'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    old = directory / f'crontab-before-{stamp}.txt'
    new = directory / f'crontab-installed-{stamp}.txt'
    old.write_text(before)
    new.write_text(after)
    # This host's crontab truncates a long input pathname. Standard input also
    # avoids depending on the workspace path length; retain the exact file above.
    subprocess.run(['crontab', '-'], input=after, text=True, check=True)
    actual = subprocess.check_output(['crontab', '-l'], text=True)
    assert actual == after, 'Installed crontab differs from reviewed entry'
    result = dict(installed_at=stamp,entries=entries,
                  previous_crontab=str(old.relative_to(ROOT)),
                  previous_sha256=hashlib.sha256(before.encode()).hexdigest(),
                  installed_sha256=hashlib.sha256(actual.encode()).hexdigest(),
                  preceding_verified_backup=receipt,
                  independent_of_gui=True,existing_jobs_preserved=True)
    output = ROOT / 'research/run-logs' / f'fs-backup-cron-install-{stamp}.json'
    atomic_json(output,result)
    print(json.dumps(dict(installed=True,entries=entries,receipt=str(output))))


if __name__ == '__main__':
    main()
