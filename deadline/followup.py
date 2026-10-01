#!/usr/bin/env python3
"""Launch manifested scoring only after parent postflight; preserve checkpoints.

Runs locally and survives a GUI restart. It never signals a remote process.
The common remote device lock remains the final authority for every launch.
"""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from preserve import HOSTS, ROOT, SSH

PARENT = '/home/ubuntu/trainium_freshstart_deadline_20261001'
PARENTS = [('trn21', 'middle'), ('trn24', 'rmsfp32'), ('trn25', 'mimetic')]
REMOTE_CHECK = r'''
import json, pathlib, sys
r=pathlib.Path(sys.argv[1]); label=sys.argv[2]
p=r/'research/run-logs'/(label+'.postflight.json')
if not p.exists():
 print(json.dumps({'complete':False})); sys.exit(0)
post=json.loads(p.read_text())
log=(r/'research/run-logs'/(label+'.log')).read_text(errors='replace')
markers=[line.split('DEADLINE_TRAIN_COMPLETE ',1)[1] for line in log.splitlines()
         if line.startswith('DEADLINE_TRAIN_COMPLETE ')]
train=(r/'out'/label/'train.log').read_text(errors='replace')
print(json.dumps({'complete':True, 'postflight':post,
                  'training':json.loads(markers[-1]) if markers else None,
                  'within_budget':'over_budget:       False' in train}))
'''


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def emit(host, **data):
    print(json.dumps(dict(timestamp=stamp(), host=host, **data)), flush=True)


def checkpoint_backup(host, variant, train_label, record):
    target = ROOT / 'out/checkpoints' / variant
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / 'final.pt'
    p = subprocess.run(['rsync', '-az', '--partial', '--timeout=40',
                        '-e', shlex.join(SSH),
                        HOSTS[host]+':'+record['checkpoint'], str(checkpoint)],
                       text=True, capture_output=True, timeout=360)
    p.check_returncode()
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert sha == record['checkpoint_sha256'], (sha, record)
    assert checkpoint.stat().st_size == record['checkpoint_bytes']
    receipt = dict(timestamp=stamp(), host=host, label=train_label,
                   remote_checkpoint=record['checkpoint'],
                   local_checkpoint=str(checkpoint), sha256=sha,
                   bytes=checkpoint.stat().st_size, verified=True,
                   remote_artifacts_deleted=False)
    (ROOT / 'research/run-logs' / (train_label+'.checkpoint-backup.json')).write_text(
        json.dumps(receipt, indent=2)+'\n')
    emit(host, checkpoint_backup=receipt)


def follow(item):
    host, variant = item
    train_label = 'fs-20261001-deadline-'+variant+'-r1'
    match_label = 'fs-20261001-deadline-'+variant+'-match-r1'
    dispatch = ROOT / 'research/run-logs' / (match_label+'.dispatch.json')
    stop = datetime.datetime.fromisoformat('2026-10-01T06:35:00+00:00')
    while datetime.datetime.now(datetime.timezone.utc) < stop:
        command = 'python3 -c '+shlex.quote(REMOTE_CHECK)+' '+shlex.quote(PARENT)+' '+shlex.quote(train_label)
        p = subprocess.run([*SSH, HOSTS[host], command], capture_output=True,
                           text=True, timeout=35)
        if p.returncode:
            emit(host, read_error=p.stderr[-1500:])
            time.sleep(30)
            continue
        record = json.loads(p.stdout)
        if not record['complete']:
            emit(host, waiting_for_parent_postflight=True)
            time.sleep(30)
            continue
        post = record['postflight']
        assert post['host_clean'] and not post['failures'], post
        assert record['training'] and record['within_budget'], record
        (ROOT / 'research/run-logs' / (match_label+'.parent-release.json')).write_text(
            json.dumps(dict(timestamp=stamp(), parent=record), indent=2)+'\n')
        if not dispatch.exists():
            request = ROOT / 'deadline/requests' / ('request-match-'+host+'.json')
            launch = subprocess.run([sys.executable,
                                     str(ROOT/'research/fresh_start_launch.py'), str(request)],
                                    text=True, capture_output=True, timeout=120)
            emit(host, scoring_launch_returncode=launch.returncode,
                 stdout=launch.stdout, stderr=launch.stderr)
            # An ambiguous launch is never automatically retried with the same
            # label: its preserved dispatch/remote job must be inspected first.
            launch.check_returncode()
        else:
            emit(host, existing_dispatch_preserved=True)
        checkpoint_backup(host, variant, train_label, record['training'])
        return
    raise TimeoutError('Parent did not release host before scoring cutoff: '+host)


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(follow, item) for item in PARENTS]
        failures = []
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as error:
                failures.append(repr(error))
                emit('controller', error=repr(error))
    if failures:
        raise RuntimeError(failures)


if __name__ == '__main__':
    main()
