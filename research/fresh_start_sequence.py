#!/usr/bin/env python3
"""Run a fixed, bounded list of already prepared experiment requests.

Every experiment still uses fresh_start_launch and its host lock, preflight,
bounded remote guard and postflight. This coordinator selects no new experiments,
never retries an ambiguous launch, and stops on any failed completion gate.
It does not change cron, merge branches, reboot a host, or kill device processes.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from fresh_start_sync import HOSTS, ROOT, sync_once


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def records(raw, marker):
    decoder = json.JSONDecoder()
    return [decoder.raw_decode(raw[m.end():])[0]
            for m in re.finditer(re.escape(marker) + ' ', raw)]


def require_completed(post, raw, request, requirements):
    if (post.get('source_commit') != request['expected_commit']
            or post.get('label') != request['label']
            or post.get('command_returncode') != 0
            or post.get('host_clean') is not True
            or post.get('guard_error') or post.get('failures')
            or post.get('interrupted_signals')):
        raise ValueError('Failed or mismatched experiment postflight')
    for rule in requirements:
        found = records(raw, rule['marker'])
        if len(found) != rule['count']:
            raise ValueError('Unexpected result count: ' + rule['marker'])
        for result in found:
            for field, expected in rule.get('fields', {}).items():
                if field not in result or result[field] != expected:
                    raise ValueError('Failed result field: ' + rule['marker'] + '.' + field)
            for field, length in rule.get('lengths', {}).items():
                if not isinstance(result.get(field), list) or len(result[field]) != length:
                    raise ValueError('Unexpected result length: ' + field)
            if rule.get('phase_timing'):
                if not math.isfinite(result['last_loss']):
                    raise ValueError('Nonfinite diagnostic loss')
                for sample in result['steps']:
                    if not all(math.isfinite(v) for v in sample.values()):
                        raise ValueError('Nonfinite phase timing')
                    if sample['step_total_ms'] <= 0:
                        raise ValueError('Nonpositive step timing')
        if 'unique_field' in rule:
            observed = [result[rule['unique_field']] for result in found]
            if sorted(observed) != sorted(rule['values']):
                raise ValueError('Missing or duplicate result identities')


def read_requests(plan):
    if not re.fullmatch(r'fs-[A-Za-z0-9_.-]+', plan['label']):
        raise ValueError('Invalid sequence label')
    if not 1 <= len(plan['steps']) <= 4 or not 60 <= plan['max_seconds'] <= 12000:
        raise ValueError('Sequence must have one to four steps and a bounded deadline')
    requests = []
    for step in plan['steps']:
        path = (ROOT / step['request']).resolve()
        if path.parent != ROOT / 'research/run-logs' or digest(path) != step['sha256']:
            raise ValueError('Request path or digest changed')
        request = json.loads(path.read_text())
        if path.name != request['label'] + '.request.json' or not step['requirements']:
            raise ValueError('Each request needs an exact label and completion gates')
        requests.append(request)
    identities = {(r['host'], r['remote'], r['expected_commit']) for r in requests}
    if len(identities) != 1 or requests[0]['host'] not in HOSTS:
        raise ValueError('A sequence must use one host, checkout and source commit')
    if len({r['label'] for r in requests}) != len(requests):
        raise ValueError('Duplicate experiment labels')
    return requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan_hash = digest(plan_path)
    plan = json.loads(plan_path.read_text())
    requests = read_requests(plan)
    folder = ROOT / 'out/fresh-start-sequences'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / (plan['label'] + '.lock')).open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    helpers = [Path(__file__).resolve(), ROOT / 'research/fresh_start_launch.py',
               ROOT / 'research/fresh_start_sync.py']
    hashes = {str(p): digest(p) for p in helpers}
    state_path = ROOT / 'research/run-logs' / (plan['label'] + '.sequence.json')
    state = dict(label=plan['label'], started_at=now(), pid=os.getpid(), status='running',
                 plan_sha256=plan_hash, helper_sha256=hashes, completed=[],
                 source_commit=requests[0]['expected_commit'], host=requests[0]['host'],
                 deadline_seconds=plan['max_seconds'])
    deadline = time.monotonic() + plan['max_seconds']
    try:
        for step, request in zip(plan['steps'], requests):
            state.update(current_label=request['label'], updated_at=now())
            save(state_path, state)
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Sequence deadline; any launched job retains its own remote bound')
                if digest(plan_path) != plan_hash or any(digest(Path(p)) != h for p, h in hashes.items()):
                    raise ValueError('Plan or coordinator helper changed while running')
                read_requests(plan)  # Recheck every prepared request before an action.
                receipt_path = ROOT / 'research/run-logs' / (request['label'] + '.dispatch.json')
                if not receipt_path.exists():
                    subprocess.run([sys.executable, str(ROOT / 'research/fresh_start_launch.py'),
                                    str(ROOT / step['request'])], cwd=ROOT, check=True, timeout=105)
                receipt = json.loads(receipt_path.read_text())
                if receipt.get('request') != request or receipt.get('returncode') != 0:
                    raise ValueError('Failed, incomplete or mismatched dispatch; no automatic retry')
                fresh = sync_once(remote=request['remote'], prefix=request['label'],
                                  hosts={request['host']: HOSTS[request['host']]})
                post_path = ROOT / 'research/run-logs' / (request['label'] + '.postflight.json')
                log_path = ROOT / 'research/run-logs' / (request['label'] + '.log')
                if fresh and post_path.exists():
                    post = json.loads(post_path.read_text())
                    require_completed(post, log_path.read_text(), request, step['requirements'])
                    state['completed'].append(dict(label=request['label'],
                        completed_at=post['completed_at'], postflight_sha256=digest(post_path),
                        log_sha256=digest(log_path)))
                    state['updated_at'] = now()
                    save(state_path, state)
                    break
                time.sleep(min(45, max(0, deadline - time.monotonic())))
        state.update(status='complete', completed_at=now(), current_label=None)
    except Exception as error:
        state.update(status='stopped', stopped_at=now(), error=str(error))
        save(state_path, state)
        raise
    save(state_path, state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
