"""One authorized >=03:33 read-only completion check; no host mutation."""
import ast,datetime,hashlib,importlib.util,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
OWN=HERE.parents[3]
REPO=Path('/home/poojith-devan/Desktop/Trainium')
assert datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,9,26,3,33,tzinfo=datetime.timezone.utc),'NOT_YET_ASSIGNED_BOUNDARY'
assert not (HERE/'trn21.json').exists(),'LOCAL_RECEIPT_ALREADY_EXISTS'
paths=[
 'research/v2/operator/20260926-dual-live-clock-child-0249/trn21.json',
 'research/v2/operator/20260926-dual-full-progress-0306/trn21.json',
 'research/v2/operator/20260926-dual-full-progress-0318/trn21.json',
 'research/v2/operator/20260926-trn21-lr-train-eval-boundary-0326/trn21.json',
]
known={3152603,3152604,3152605,3152621,3152622,3152637,3152652,3152653,3152654,3152655,3153251,3153259,3153290,3175131,3175132,3175146}
identities={};prior_receipts={}
for name in paths:
    p=OWN/name
    raw=p.read_bytes();o=json.loads(raw);o=o.get('remote',o)
    prior_receipts[name]=hashlib.sha256(raw).hexdigest()
    for item in o['processes']:
        known.add(item['pid'])
        if item.get('exists'):
            identities[item['pid']]={k:item[k] for k in ('pid','starttime','ppid','pgid','sid','argv') if k in item}
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'],'LOCAL_STOP'
source=(HERE/'remote_snapshot.py').read_bytes();ast.parse(source)
assert not m.SENSITIVE.search(source),'SOURCE_SENSITIVE_MARKER'
payload={'known_pids':sorted(known),'known_identity_records':list(identities.values())}
argv,*_=m.load_transport('trn21');argv[-1]='exec python3 -I -B -'
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(argv,input=('P='+repr(payload)+'\n').encode()+source,capture_output=True,timeout=60)
assert not m.SENSITIVE.search(r.stdout+r.stderr),'OUTPUT_SENSITIVE_MARKER'
receipt={'started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rc':r.returncode,'remote_source_sha256':hashlib.sha256(source).hexdigest(),'prior_receipt_sha256':prior_receipts,'local_before':state,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
with (HERE/'trn21.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
if r.returncode:raise RuntimeError('READONLY_CHECK_FAILED:'+r.stderr.decode())
o=receipt['remote']
if o['terminal_seen']:
    dest=OWN/'research/v2/operator/20260926-trn21-clock36-lr012/completed'
    dest.mkdir(exist_ok=False)
    with (dest/'results-bundle.json').open('x') as f:json.dump(o,f,indent=2);f.write('\n')
    with (dest/'explicit-process-postflight.json').open('x') as f:json.dump(o['postflight'],f,indent=2);f.write('\n')
print(json.dumps({'utc':o['utc'],'terminal_seen':o['terminal_seen'],'receipt_sha256':hashlib.sha256((HERE/'trn21.json').read_bytes()).hexdigest(),'terminal':o['files'].get('terminal.json',{}).get('text'),'last_eval_lines':o['files'].get('fresh-public20m.log',{}).get('text','').splitlines()[-12:],'postflight':o['postflight'],'checkpoint_after_terminal':o['checkpoint_after_terminal'],'live_pids':[p['pid'] for p in o['processes'] if p['exists']],'disk_free':o['free_bytes']},indent=2))
