#!/usr/bin/env python3
"""Start one finite sequence only after its exact plan and requests are committed."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from fresh_start_sequence import read_requests
from fresh_start_sync import ROOT


def require_committed_files(root, commit, paths):
    for path in paths:
        relative = path.resolve().relative_to(root.resolve())
        result = subprocess.run(['git', 'show', f'{commit}:{relative.as_posix()}'],
                                cwd=root, capture_output=True)
        if result.returncode or result.stdout != path.read_bytes():
            raise ValueError(f'Commit the exact sequence file before dispatch: {relative}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    label = args.label
    assert label.startswith('fs-') and '/' not in label
    logs = ROOT / 'research/run-logs'
    plan = logs / (label + '.plan.json')
    payload = json.loads(plan.read_text())
    requests = read_requests(payload)
    assert payload['label'] == label
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require_committed_files(ROOT, source, [plan, *[ROOT / step['request'] for step in payload['steps']]])
    log = logs / (label + '.coordinator.log')
    receipt = logs / (label + '.coordinator-launch.json')
    assert not receipt.exists() and not log.exists() and not (logs / (label + '.sequence.json')).exists()
    command = [sys.executable, '-u', str(ROOT / 'research/fresh_start_sequence.py'), str(plan)]
    with log.open('x') as output:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(label=label, pid=process.pid, started_at=datetime.now(timezone.utc).isoformat(),
                  source_commit=source, kernel_source_commit=requests[0]['expected_commit'],
                  plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
                  starter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  plan_and_requests_match_committed_blobs=True,
                  command=command, log=str(log.relative_to(ROOT)),
                  max_seconds=payload['max_seconds'], cron_changed=False)
    with receipt.open('x') as output:
        output.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
