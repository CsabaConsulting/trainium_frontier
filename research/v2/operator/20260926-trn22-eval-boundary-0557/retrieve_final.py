"""One scoped final read: complete bounded noncache artifacts and actual lineage census."""
import ast
import datetime
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
DEST = HERE / 'completed'
assert not DEST.exists(), 'LOCAL_ARCHIVE_EXISTS'
template = HERE.parent / '20260926-trn22-confirmed36-clean-cold-five-r2/retrieve_final.py'
raw = template.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'd71d9b710b33a5e8f9051d079e3e8a7d13b83676f3de16d32be02ae5cfb1c482'
tree = ast.parse(raw)
body = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'body' for t in n.targets))
body = body.replace('fs-20260926-confirmed36-clean-cold-trn22-r2', 'fs-20260926-confirmed36-clean-coldfull-trn22-r2')
needle = "identities={};checkpoints={};known=set(P['known_prior_identities']);files={};total=0"
assert body.count(needle) == 1
inventory = r'''
# Enumerate only this already-terminal isolated run, never enter caches or follow links.
# Hidden files are included; checkpoint bytes remain on the host and are hashed below.
inventory=[];skipped=[];names=[];pending=[root];enum_start=time.monotonic()
while pending:
 d=pending.pop()
 for p in sorted(d.iterdir()):
  assert time.monotonic()-enum_start<=10 and len(inventory)<512,'INVENTORY_BOUND'
  s=p.lstat();rel=str(p.relative_to(root))
  rec={'path':rel,'bytes':s.st_size,'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
  inventory.append(rec)
  if stat.S_ISLNK(s.st_mode):skipped.append({'path':rel,'reason':'SYMLINK_NOT_FOLLOWED'});continue
  if stat.S_ISDIR(s.st_mode):
   if p.name in ('caches','__pycache__','.git'):skipped.append({'path':rel,'reason':'CACHE_OR_GIT_SUBTREE_NOT_ENTERED'})
   else:pending.append(p)
  elif stat.S_ISREG(s.st_mode):
   if p.suffix in ('.pt','.pth','.neff','.ntff','.pyc','.bin'):skipped.append({'path':rel,'reason':'BINARY_NOT_TRANSFERRED'})
   else:names.append(rel)
  else:skipped.append({'path':rel,'reason':'NOT_REGULAR'})
'''
body = body.replace(needle, inventory + '\n' + needle)
old = " raw=p.read_bytes();total+=len(raw);assert total<=16*1024*1024,'RECEIPT_TOTAL_BOUND'"
new = " fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)\n with os.fdopen(fd,'rb') as f:\n  a=os.fstat(f.fileno());raw=f.read(2*1024*1024+1);z=os.fstat(f.fileno())\n assert len(raw)==s.st_size and (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'RECEIPT_CHANGED'\n total+=len(raw);assert total<=16*1024*1024,'RECEIPT_TOTAL_BOUND'"
assert body.count(old) == 1
body = body.replace(old, new)
body = body.replace("'files':files,'raw_bytes':total", "'files':files,'artifact_inventory':inventory,'skipped_paths':skipped,'raw_bytes':total")
body = body.replace('current r2 live/boundary receipts', 'current full-r2 live/boundary receipts and every actual identity in the complete guardian lifetime log')
folder = HERE.parent / '20260926-trn22-confirmed36-clean-cold-full-r2'
paths = [folder/'trn22.json',folder/'progress/boundary-0509.json',HERE.parent/'20260926-dual-status-0523/trn22.json',HERE.parent/'20260926-trn22-status-0537/trn22.json',HERE.parent/'20260926-trn22-training-handoff-0543/trn22.json',HERE.parent/'20260926-trn22-eval-boundary-0550/trn22.json',HERE/'trn22.json']
identities = {}
def add_identity(p):
    p = dict(p)
    if 'start_ticks' in p:
        p['starttime'] = p['start_ticks']
    pid = int(p['pid'])
    assert p.get('starttime') is not None
    if pid in identities:
        assert identities[pid]['starttime'] == p['starttime'], 'OBSERVED_PID_REUSE'
    else:
        identities[pid] = p
for path in paths:
    remote = json.loads(path.read_bytes())['remote']
    for p in remote['processes']:
        if p.get('exists'):
            add_identity(p)
    log = remote['files'].get('lifetime-events.jsonl')
    if log:
        assert log['capture'] == 'complete_at_observation'
        for line in log['text'].splitlines():
            event = json.loads(line)
            if 'identity' in event:
                add_identity(event['identity'])
assert {3281907,3281908,3281909,3281925,3281926,3281940,3281955,3281956,3281957,3281958,3307829,3307830,3307844} <= set(identities)
payload = {'known_prior_identities': identities}
spec = importlib.util.spec_from_file_location('transport', REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
wire = 'P='+repr(payload)+'\n'+body
assert not m.SENSITIVE.search(wire.encode())
argv, *_ = m.load_transport('trn22'); argv[-1] = 'exec python3 -I -B -'
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
r = subprocess.run(argv, input=wire, text=True, capture_output=True, timeout=60)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode())
DEST.mkdir(exist_ok=False)
meta = {'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stderr':r.stderr,'remote_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'template_sha256':hashlib.sha256(raw).hexdigest(),'prior_receipts':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
with (DEST/'retrieval.json').open('x') as f:json.dump(meta,f,indent=2);f.write('\n')
if r.returncode:
    with (DEST/'failed-output.txt').open('x') as f:f.write(r.stdout)
    print(json.dumps(meta));raise SystemExit(r.returncode)
o = json.loads(r.stdout)
for name, value in [('results-bundle.json',o),('explicit-process-postflight.json',o['postflight']),('retained-input-identities.json',o['retained_input_identities'])]:
    with (DEST/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
for name, value in o['files'].items():
    b=value['text'].encode();assert len(b)==value['bytes'] and hashlib.sha256(b).hexdigest()==value['sha256'],name
post=o['postflight']
print(json.dumps({'utc':o['utc'],'file_count':len(o['files']),'raw_bytes':o['raw_bytes'],'checkpoint_after_terminal':o['checkpoint_after_terminal'],'all_observed_pids_absent':post['all_observed_pids_absent'],'observed_pid_count':len(post['observed_pids']),'related_rows':post['related_rows'],'busy_job_rows':post['busy_job_rows'],'census_count':len(post['complete_process_census']),'lock_competitor_rc':post['lock_competitor_rc'],'device_idle':post['neuron_device_idle'],'fuser_rc':post['fuser_rc'],'disk_free':o['free_bytes'],'skipped_paths':o['skipped_paths'],'bundle_sha256':hashlib.sha256((DEST/'results-bundle.json').read_bytes()).hexdigest()},indent=2))
