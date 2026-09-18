#!/usr/bin/env python3
"""Crash-independent research backup; never changes the working branch/index.

Read manifested EC2 jobs, retrieve their evidence into ignored staging, create a
WIP recovery commit with an alternate Git index, then fast-forward-push both the
committed research branch and the recovery branch. No checkout, reset, merge,
force push, process termination, or deletion of remote files is performed.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from fresh_start_sync import HOSTS, ROOT, SSH


TEXT_SUFFIXES = {'.py', '.md', '.rst', '.txt', '.json', '.jsonl', '.log', '.csv',
                 '.tsv', '.sh', '.toml', '.yaml', '.yml', '.patch', '.diff'}
MAX_BLOB = 50 * 1024**2


class GitBusy(RuntimeError):
    """Skip this cycle while the user is changing repository history/staging."""


def git_path(repo, name):
    path = Path(git(repo, 'rev-parse', '--git-path', name).decode().strip())
    return path if path.is_absolute() else repo / path


def require_idle_git(repo, branch):
    markers = ['index.lock', 'HEAD.lock', 'packed-refs.lock', 'MERGE_HEAD',
               'rebase-merge', 'rebase-apply', 'CHERRY_PICK_HEAD', 'REVERT_HEAD',
               'sequencer', 'BISECT_LOG', 'refs/heads/' + branch + '.lock']
    busy = [name for name in markers if git_path(repo, name).exists()]
    if busy:
        raise GitBusy('Repository operation in progress: ' + ', '.join(busy))


def index_identity(repo):
    path = git_path(repo, 'index')
    if not path.exists():
        return None
    before = path.stat()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    fields = lambda s: (s.st_ino, s.st_size, s.st_mtime_ns)
    if fields(before) != fields(after):
        raise GitBusy('Real Git index changed during inspection')
    return (*fields(after), sha)


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def desktop_keyring_environment():
    """Let cron use the existing user's keyring; never copy tokens to disk.

    This Devuan desktop uses a session bus under /tmp rather than the usual
    /run/user/<uid>/bus. Read only the bus address from a same-UID keyring or
    session manager process if cron did not inherit it. A locked/unavailable
    keyring remains an authentication failure, retried at the next interval.
    """
    if os.environ.get('DBUS_SESSION_BUS_ADDRESS'):
        return
    uid = os.getuid()
    standard = Path(f'/run/user/{uid}/bus')
    if standard.exists():
        os.environ['DBUS_SESSION_BUS_ADDRESS'] = 'unix:path=' + str(standard)
        return
    names = {'gnome-keyring-d', 'gnome-session-b', 'xfce4-session', 'cinnamon-sessio',
             'mate-session', 'lxqt-session'}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            if directory.stat().st_uid != uid or (directory / 'comm').read_text().strip() not in names:
                continue
            values = (directory / 'environ').read_bytes().split(b'\0')
            for value in values:
                if value.startswith(b'DBUS_SESSION_BUS_ADDRESS=unix:'):
                    os.environ['DBUS_SESSION_BUS_ADDRESS'] = value.split(b'=', 1)[1].decode()
                    return
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
        temporary = f.name
    os.replace(temporary, path)


def command(argv, *, cwd=ROOT, env=None, input=None, timeout=120):
    result = subprocess.run(argv, cwd=cwd, env=env, input=input,
                            capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{shlex.join(argv)}: exit{result.returncode}: '
                           + result.stderr.decode(errors='replace')[-4000:])
    return result.stdout


def git(repo, *args, **kw):
    return command(['git', *args], cwd=repo, **kw)


REMOTE_DISCOVERY = r'''
import datetime,json,pathlib,subprocess,sys
jobs=json.load(sys.stdin)
home=pathlib.Path('/home/ubuntu')
files=set(); runs=[]; trees={}; errors=[]
for job in jobs:
    root=pathlib.Path(job['remote']); label=job['label']
    assert root.parent==home and root.name.startswith('trainium_freshstart_20260918')
    assert label.startswith('fs-') and all(c.isalnum() or c in '-_.' for c in label)
    if not root.is_dir():
        errors.append(dict(remote=str(root),label=label,error='checkout missing')); continue
    if str(root) not in trees:
        head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        scope=['train.py','prepare.py','research/scratch','research/fresh_start_guard.py',
               'research/fresh_start_host.py','research/fresh_start_launch.py',
               'research/run_experiment.py']
        changed=subprocess.check_output(['git','-C',str(root),'diff','--name-only',
                                         'HEAD','--',*scope],text=True).splitlines()
        untracked=subprocess.check_output(['git','-C',str(root),'ls-files','--others',
                           '--exclude-standard','--',*scope],text=True).splitlines()
        trees[str(root)]=dict(commit=head,source_changed=changed,source_untracked=untracked)
        for relative in changed+untracked+['research/experiment-log.jsonl']:
            p=root/relative
            if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()):
                files.add(str(p.relative_to(home)))
    folder=root/'research/run-logs'
    for p in folder.glob(label+'.*'):
        if p.is_file() and not p.is_symlink() and p.stat().st_size <= 50*1024**2:
            files.add(str(p.relative_to(home)))
    post=folder/(label+'.postflight.json')
    state=dict(label=label,remote=str(root),source_commit=trees[str(root)]['commit'])
    if post.exists():
        try:
            j=json.loads(post.read_text())
            state.update(state='completed',returncode=j.get('command_returncode'),
                         host_clean=j.get('host_clean'),completed_at=j.get('completed_at'))
        except (ValueError,OSError) as e: state.update(state='unreadable_postflight',error=str(e))
    else: state['state']='no_postflight; inspect before relaunching'
    runs.append(state)
print(json.dumps(dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      files=sorted(files),runs=runs,worktrees=trees,errors=errors)))
'''


def dispatched_jobs(repo):
    jobs = {h: {} for h in HOSTS}
    for path in sorted((repo / 'research/run-logs').glob('fs-*.dispatch.json')):
        try:
            request = json.loads(path.read_text())['request']
        except (ValueError, KeyError):
            continue  # A just-created dispatch may still be being written.
        host, remote, label = (request.get(k) for k in ('host', 'remote', 'label'))
        if host not in HOSTS or not isinstance(remote, str) or not isinstance(label, str):
            continue
        if (Path(remote).parent != Path('/home/ubuntu') or
                not Path(remote).name.startswith('trainium_freshstart_20260918') or
                not re.fullmatch(r'fs-[A-Za-z0-9_.-]+', label)):
            raise ValueError(f'Invalid dispatch scope in {path}')
        jobs[host][(remote, label)] = dict(remote=remote, label=label)
    return {host: list(rows.values()) for host, rows in jobs.items()}


def retrieve(repo, directory):
    status = {}
    for host, jobs in dispatched_jobs(repo).items():
        if not jobs:
            continue
        try:
            raw = command([*SSH, HOSTS[host], 'python3 -c ' + shlex.quote(REMOTE_DISCOVERY)],
                          input=json.dumps(jobs).encode(), timeout=75)
            discovery = json.loads(raw)
            files = discovery['files']
            for name in files:
                p = Path(name)
                if p.is_absolute() or '..' in p.parts or not p.parts[0].startswith('trainium_freshstart_20260918'):
                    raise ValueError('Unsafe discovered relative path')
            destination = directory / 'hosts' / host
            destination.mkdir(parents=True, exist_ok=True)
            listing = directory / (host + '-files.nul')
            listing.write_bytes(b''.join(s.encode() + b'\0' for s in files))
            command(['rsync', '-az', '--checksum', '--relative', '--delay-updates',
                     '--timeout=30', '--bwlimit=2048', '--from0', '--files-from=' + str(listing),
                     '-e', shlex.join(SSH), HOSTS[host] + ':/home/ubuntu/',
                     str(destination) + '/'], timeout=150)
            atomic_json(destination / 'discovery.json', discovery)
            status[host] = dict(ok=True, files=len(files), jobs=len(jobs),
                                errors=discovery['errors'], checked_at=discovery['checked_at'])
        except Exception as error:
            # A dead host must not prevent pushing the available local evidence.
            status[host] = dict(ok=False, error=str(error), checked_at=now())
    return status


def index_blob(repo, env, path, data, mode='100644'):
    if len(data) > MAX_BLOB:
        raise ValueError(f'Artifact exceeds Git backup limit: {path}')
    digest = git(repo, 'hash-object', '-w', '--stdin', input=data).strip()
    git(repo, 'update-index', '-z', '--index-info', env=env,
        input=mode.encode() + b' ' + digest + b'\t' + path.encode() + b'\0')


def create_snapshot(repo, directory, branch, backup_branch, remote_status):
    require_idle_git(repo, branch)
    original_index = index_identity(repo)
    ref = 'refs/heads/' + backup_branch
    if git(repo, 'branch', '--show-current').decode().strip() != branch:
        raise ValueError('Active branch changed; refusing to back up the wrong campaign')
    head = git(repo, 'rev-parse', 'HEAD').decode().strip()
    prior = subprocess.run(['git', 'rev-parse', '--verify', ref], cwd=repo,
                           capture_output=True, text=True)
    previous = prior.stdout.strip() if prior.returncode == 0 else None
    skipped = []
    with tempfile.TemporaryDirectory(prefix='index-', dir=directory) as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'),
                   GIT_LITERAL_PATHSPECS='1')
        git(repo, 'read-tree', head, env=env)
        tracked = git(repo, 'ls-files', '-z', env=env).split(b'\0')
        for raw in tracked:
            if not raw:
                continue
            path = repo / os.fsdecode(raw)
            if path.is_file() and not path.is_symlink() and path.stat().st_size > MAX_BLOB:
                raise ValueError(f'Tracked file exceeds Git backup limit: {path}')
        git(repo, 'add', '-u', '--', '.', env=env)
        # Compare against the temporary HEAD index, so newly staged files in the
        # real index are captured too. Preserve the exact staged patch separately.
        new = git(repo, 'ls-files', '--others', '--exclude-standard', '-z', env=env).split(b'\0')
        real_tracked = set(git(repo, 'ls-files', '-z').split(b'\0'))
        selected = []
        for raw in new:
            if not raw:
                continue
            name = os.fsdecode(raw)
            path = repo / name
            if (raw in real_tracked and path.is_file() and path.stat().st_size <= MAX_BLOB):
                selected.append(raw)
            elif (path.is_file() and not path.is_symlink() and path.suffix in TEXT_SUFFIXES
                    and path.stat().st_size <= MAX_BLOB
                    and (len(Path(name).parts) == 1 or name.startswith(
                        ('research/', 'articles/', '.memsearch/memory/')))):
                selected.append(raw)
            else:
                skipped.append(name)
        if selected:
            git(repo, 'add', '--pathspec-from-file=-', '--pathspec-file-nul', env=env,
                input=b'\0'.join(selected) + b'\0')
        staged = git(repo, 'diff', '--cached', '--binary', '--full-index', head, '--')
        index_blob(repo, env, 'research/continuity/staged.patch', staged)
        # Remote evidence is kept under a distinct path in the recovery commit;
        # stale remote files cannot replace canonical source, logs or the registry.
        remote_files = 0
        for path in sorted((directory / 'hosts').rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            relative = 'research/continuity/hosts/' + str(path.relative_to(directory / 'hosts'))
            index_blob(repo, env, relative, path.read_bytes())
            remote_files += 1
        manifest = dict(created_at=now(),source_branch=branch,source_commit=head,
                        recovery_branch=backup_branch,previous_snapshot=previous,
                        kind='WIP recovery snapshot; not a validation or promotion claim',
                        remote_status=remote_status,remote_files=remote_files,
                        excluded_new_files=skipped,
                        ignored_artifacts='Checkpoints and raw profiles require artifact backup.',
                        restore='Read research/RESUME.md; recover into a fresh checkout first.')
        last = directory / 'latest.json'
        if last.exists():
            manifest['previous_backup_receipt'] = json.loads(last.read_text())
        index_blob(repo, env, 'research/continuity/latest.json',
                   (json.dumps(manifest, indent=2) + '\n').encode())
        tree = git(repo, 'write-tree', env=env).decode().strip()
        require_idle_git(repo, branch)
        if index_identity(repo) != original_index:
            raise GitBusy('Real Git index changed during snapshot; retry next cycle')
        if git(repo, 'rev-parse', 'HEAD').decode().strip() != head:
            raise GitBusy('HEAD changed during backup; retry next cycle')
        parents = ['-p', previous] if previous else []
        if head != previous:
            parents += ['-p', head]
        snapshot = git(repo, 'commit-tree', tree, *parents,
                       input=f'Recovery snapshot {now()} from {head[:12]} (WIP)\n'.encode()).decode().strip()
        require_idle_git(repo, branch)
        git(repo, 'update-ref', ref, snapshot, previous or '0' * 40)
    return dict(source_commit=head,snapshot_commit=snapshot,source_branch=branch,
                snapshot_branch=backup_branch,excluded_new_files=skipped,
                remote_files=remote_files)


def push_snapshot(repo, remote, result):
    require_idle_git(repo, result['source_branch'])
    expected = {'refs/heads/' + result['source_branch']: result['source_commit'],
                'refs/heads/' + result['snapshot_branch']: result['snapshot_commit']}
    # --atomic prevents a half-updated pair; normal fast-forward checks protect
    # both branches. A conflicting external update fails instead of overwriting.
    git(repo, 'push', '--atomic', remote, *(sha + ':' + ref for ref, sha in expected.items()),
        timeout=150, env=dict(os.environ, GIT_TERMINAL_PROMPT='0', GH_PROMPT_DISABLED='1'))
    rows = git(repo, 'ls-remote', '--heads', remote, *expected).decode().splitlines()
    actual = {line.split()[1]: line.split()[0] for line in rows}
    if actual != expected:
        raise RuntimeError(f'Remote verification differs: {actual}')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'research/continuity.json')
    parser.add_argument('--skip-remote', action='store_true')
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args()
    desktop_keyring_environment()
    config = json.loads(args.config.read_text())
    directory = ROOT / 'out/fresh-start-backup'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'backup.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Backup already running; skipped', flush=True)
            return
        receipt = dict(started_at=now(),remote=config['remote'],pushed=False)
        try:
            require_idle_git(ROOT, config['source_branch'])
            url = git(ROOT, 'remote', 'get-url', '--push', config['remote']).decode().strip()
            if url != config['expected_remote_url']:
                raise ValueError('Push remote changed; refusing an unintended destination')
            status = {} if args.skip_remote else retrieve(ROOT, directory)
            result = create_snapshot(ROOT, directory, config['source_branch'],
                                     config['recovery_branch'], status)
            receipt.update(result,remote_status=status)
            if not args.local_only:
                receipt['verified_remote_refs'] = push_snapshot(ROOT, config['remote'], result)
                receipt['pushed'] = True
            receipt['remote_retrieval_complete'] = all(s['ok'] and not s.get('errors') for s in status.values())
        except GitBusy as error:
            receipt.update(skipped=True,skipped_reason=str(error))
        except Exception as error:
            receipt['error'] = str(error)
        receipt['completed_at'] = now()
        name = receipt['started_at'].replace(':', '').replace('+', '_') + '.json'
        atomic_json(directory / 'receipts' / name, receipt)
        atomic_json(directory / 'latest.json', receipt)
        if receipt['pushed']:
            atomic_json(directory / 'last-success.json', receipt)
        print(json.dumps(receipt), flush=True)
        if receipt.get('skipped'):
            return
        if 'error' in receipt or not receipt.get('remote_retrieval_complete', False):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
