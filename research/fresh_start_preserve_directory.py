#!/usr/bin/env python3
"""Retrieve a completed profile directory and independently verify every file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

from fresh_start_backup import atomic_json, command, now
from fresh_start_sync import HOSTS, ROOT, SSH


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host',choices=tuple(HOSTS),required=True)
    ap.add_argument('--remote',type=Path,required=True)
    ap.add_argument('--label',required=True)
    ap.add_argument('--commit',required=True)
    ap.add_argument('--relative',type=Path,required=True)
    ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--recovery-receipt',type=Path,
                    help='Explicit clean maintenance receipt for a successful command whose only postflight failure was disk reserve; original failure is retained.')
    args=ap.parse_args()
    assert args.remote.parent==Path('/home/ubuntu') and args.remote.name.startswith('trainium_freshstart_')
    assert re.fullmatch(r'fs-[A-Za-z0-9_.-]+',args.label)
    assert re.fullmatch(r'[0-9a-f]{40}',args.commit)
    assert not args.relative.is_absolute() and '..' not in args.relative.parts and args.relative.parts[0]=='out'
    destination=(ROOT/args.destination).resolve()
    assert destination.is_relative_to((ROOT/'out/fresh-start-profiles').resolve())
    recovery_name=None
    recovery_digest=None
    if args.recovery_receipt is not None:
        recovery_path=(ROOT/args.recovery_receipt).resolve()
        assert recovery_path.parent==(ROOT/'research/run-logs').resolve()
        recovery=json.loads(recovery_path.read_text())
        assert recovery['host']==args.host and recovery['expected_previous_label']==args.label
        assert recovery['host_clean'] and recovery['failures']==[] and recovery['host_lock_held']
        recovery_name=str(recovery_path.relative_to(ROOT))
        recovery_digest=hashlib.sha256(json.dumps(recovery,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    script='\n'.join([
        'import pathlib,subprocess,json,hashlib',
        f'root=pathlib.Path({str(args.remote)!r})',
        f'source=root/{str(args.relative)!r}',
        f"post=json.loads((root/'research/run-logs'/{(args.label+'.postflight.json')!r}).read_text())",
        "assert post['command_returncode']==0",
        f'recovery_name={recovery_name!r}',
        'recovery_identity=None',
        "if not post['host_clean']:",
        " assert post['failures']==['less than8GiB disk space available']",
        ' assert recovery_name is not None',
        ' recovery=json.loads((root/recovery_name).read_text())',
        " digest=hashlib.sha256(json.dumps(recovery,sort_keys=True,separators=(',',':')).encode()).hexdigest()",
        f' assert digest=={recovery_digest!r}',
        f" assert recovery['host']=={args.host!r} and recovery['expected_previous_label']=={args.label!r}",
        " assert recovery['host_clean'] and recovery['failures']==[] and recovery['host_lock_held']",
        " assert recovery['completed_at']>=post['completed_at']",
        " recovery_identity=dict(receipt=recovery_name,canonical_sha256=digest,label=recovery['label'],completed_at=recovery['completed_at'])",
        "head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()",
        f"assert head=={args.commit!r}==post['source_commit']",
        'files=[]',
        "for path in sorted(source.rglob('*')):",
        ' assert not path.is_symlink()',
        ' if not path.is_file(): continue',
        ' before=path.stat()',
        " with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()",
        ' after=path.stat()',
        ' assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)',
        " files.append(dict(path=str(path.relative_to(source)),bytes=after.st_size,sha256=digest))",
        'assert files',
        "print(json.dumps(dict(source=str(source),files=files,commit=head,original_postflight_host_clean=post['host_clean'],original_postflight_failures=post['failures'],postflight_recovery=recovery_identity)))"])
    remote=json.loads(command([*SSH,HOSTS[args.host],'python3 -c '+shlex.quote(script)],timeout=120))
    partial=destination.with_name(destination.name+'.partial')
    if not destination.exists():
        partial.mkdir(parents=True,exist_ok=True)
        command(['rsync','-a','--partial','--timeout=45','--bwlimit=8192','-e',shlex.join(SSH),
                 HOSTS[args.host]+':'+remote['source']+'/',str(partial)+'/'],timeout=900)
        checking=partial
    else:
        checking=destination
    expected={f['path'] for f in remote['files']}
    assert {str(p.relative_to(checking)) for p in checking.rglob('*') if p.is_file()}==expected
    for row in remote['files']:
        path=checking/row['path']
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], row['path']
    if checking==partial:
        assert not destination.exists()
        os.rename(partial,destination)
    result=dict(label=args.label,host=args.host,remote=remote['source'],source_commit=args.commit,
                local=str(destination.relative_to(ROOT)),verified_at_utc=now(),
                total_bytes=sum(r['bytes'] for r in remote['files']),files=remote['files'],
                all_remote_local_hashes_match=True,
                original_postflight_host_clean=remote['original_postflight_host_clean'],
                original_postflight_failures=remote['original_postflight_failures'],
                postflight_recovery=remote['postflight_recovery'])
    receipt=ROOT/'research/run-logs'/(args.label+'.preservation.json')
    if receipt.exists():
        old=json.loads(receipt.read_text())
        assert old['files']==result['files'] and old['source_commit']==result['source_commit']
    else:
        atomic_json(receipt,result)
    print(json.dumps(dict(label=args.label,files=len(expected),bytes=result['total_bytes'],
                         matched=True,local=result['local'])),flush=True)


if __name__=='__main__':
    main()
