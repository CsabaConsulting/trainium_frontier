#!/usr/bin/env python3
"""Bounded local scheduler: never overlap a matched eval and a new training job."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from preserve import ROOT, SSH, HOSTS

REMOTE_CHECK = r'''
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);label=sys.argv[2]
p=r/'research/run-logs'/(label+'.postflight.json')
if not p.exists():
 print(json.dumps({'complete':False}));sys.exit(0)
score=r/'research/run-logs'/(label+'.matched-eval.json')
print(json.dumps({'complete':True,'postflight':json.loads(p.read_text()),
                  'score':json.loads(score.read_text()) if score.exists() else None}))
'''


def run(item):
    host, parent = item
    label = 'fs-20261001-deadline-'+parent+'-match-r1'
    request_path = ROOT/'deadline/requests'/('request-more-'+host+'.json')
    request = json.loads(request_path.read_text())
    stop = datetime.datetime.fromisoformat('2026-10-01T06:02:00+00:00')
    remote = '/home/ubuntu/trainium_freshstart_deadline_scoring_20261001'
    while datetime.datetime.now(datetime.timezone.utc) < stop:
        command = 'python3 -c '+shlex.quote(REMOTE_CHECK)+' '+shlex.quote(remote)+' '+shlex.quote(label)
        p = subprocess.run([*SSH, HOSTS[host], command], text=True,
                           capture_output=True, timeout=35)
        if p.returncode:
            print(json.dumps(dict(host=host, read_error=p.stderr[-1000:])), flush=True)
            time.sleep(25); continue
        record = json.loads(p.stdout)
        if not record['complete']:
            print(json.dumps(dict(host=host, waiting_for_matched_eval=True)), flush=True)
            time.sleep(25); continue
        post = record['postflight']
        assert post['host_clean'] and not post['failures'], post
        assert post['command_returncode'] == 0 and record['score'], record
        (ROOT/'research/run-logs'/(request['label']+'.parent-release.json')).write_text(
            json.dumps(record, indent=2)+'\n')
        dispatch = ROOT/'research/run-logs'/(request['label']+'.dispatch.json')
        if dispatch.exists():
            print(json.dumps(dict(host=host, existing_dispatch_preserved=True)), flush=True)
            return
        p = subprocess.run([sys.executable, str(ROOT/'research/fresh_start_launch.py'),
                            str(request_path)], text=True, capture_output=True, timeout=120)
        print(json.dumps(dict(host=host, returncode=p.returncode,
                             stdout=p.stdout, stderr=p.stderr)), flush=True)
        p.check_returncode()
        return
    receipt = dict(host=host, skipped=True,
        reason='Matched validation did not release device before 06:02 UTC; preserve submission time.')
    (ROOT/'research/run-logs'/(request['label']+'.deadline-skip.json')).write_text(
        json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run, x) for x in
            [('trn21', 'middle'), ('trn24', 'rmsfp32')]]):
            f.result()
