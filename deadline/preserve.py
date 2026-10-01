#!/usr/bin/env python3
"""Bounded read-only remote mirroring; commit only this wave's evidence."""
import argparse, concurrent.futures, datetime, fcntl, hashlib, json, os
from pathlib import Path
import shlex, subprocess, time

ROOT = Path(__file__).resolve().parents[1]
SSH = ['ssh', '-i', '/home/csaba/.ssh/trn2.pem', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=10', '-o', 'UpdateHostKeys=no']
HOSTS = {
    'trn21': 'ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com',
    'trn22': 'ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com',
    'trn23': 'ubuntu@ec2-16-26-234-102.ap-southeast-4.compute.amazonaws.com',
    'trn24': 'ubuntu@ec2-16-50-99-66.ap-southeast-4.compute.amazonaws.com',
    'trn25': 'ubuntu@ec2-16-50-101-219.ap-southeast-4.compute.amazonaws.com',
}


def sync(request_path):
    m = json.loads(request_path.read_text())
    host, remote, label = HOSTS[m['host']], m['remote'], m['label']
    dest = ROOT / 'research/run-logs'
    out = ROOT / 'out' / label
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for src, target, options in [
        (remote + '/research/run-logs/', str(dest) + '/',
         ['--include=' + label + '*', '--exclude=*']),
        (remote + '/out/' + label + '/', str(out) + '/',
         ['--include=train.log', '--include=eval.log', '--exclude=*']),
    ]:
        p = subprocess.run(['rsync', '-az', '--timeout=18', '-e', shlex.join(SSH),
                            *options, host + ':' + src, target],
                           text=True, capture_output=True, timeout=24)
        results.append(dict(returncode=p.returncode, stderr=p.stderr[-2000:]))
    # Keep raw training/evaluation text in version control too. Checkpoints and
    # compiler caches stay under ignored out/, and are retrieved separately.
    for name in ['train.log', 'eval.log']:
        path = out / name
        if path.exists():
            (dest / (label + '.' + name)).write_bytes(path.read_bytes())
    post = dest / (label + '.postflight.json')
    return dict(host=m['host'], label=label, mirror=results,
                postflight=json.loads(post.read_text()) if post.exists() else None)


def git_backup():
    # Never commit another editor's staged files; use a separate worktree lock.
    lock_path = ROOT / 'out/evidence-backup.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode:
            return 'skipped: index already has staged work'
        subprocess.run(['git', 'add', '--', 'deadline/requests',
                        'deadline/live-status.json', 'research/run-logs'], cwd=ROOT, check=True)
        if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode == 0:
            return 'no new evidence'
        subprocess.run(['git', 'commit', '-m', 'evidence: preserve live deadline experiment logs'],
                       cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p = subprocess.run(['git', 'push', 'origin', 'HEAD:research/deadline-hailmary-20260930'],
                           cwd=ROOT, text=True, capture_output=True, timeout=35)
        return dict(push_returncode=p.returncode, push_output=p.stdout + p.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--until', default='2026-10-01T06:50:00+00:00')
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    stop = datetime.datetime.fromisoformat(args.until)
    round_number = 0
    while datetime.datetime.now(datetime.timezone.utc) < stop:
        requests = sorted((ROOT / 'deadline/requests').glob('request-*.json'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(sync, path) for path in requests]
            records = []
            for f in futures:
                try: records.append(f.result())
                except Exception as e: records.append({'mirror_error': repr(e)})
        status = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      remote_jobs=records, checkpoint_retrieval_separate=True)
        path = ROOT / 'deadline/live-status.json'
        temp = path.with_suffix('.partial')
        temp.write_text(json.dumps(status, indent=2) + '\n'); temp.replace(path)
        if round_number % 2 == 0 or args.once:
            try: status['git_backup'] = git_backup()
            except Exception as e: status['git_backup_error'] = repr(e)
        print(json.dumps(status), flush=True)
        if args.once: break
        round_number += 1
        time.sleep(45)


if __name__ == '__main__':
    main()
