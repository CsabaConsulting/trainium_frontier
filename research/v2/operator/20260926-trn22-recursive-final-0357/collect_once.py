"""One authorized >=03:57 read-only completion check; no host mutation."""
import ast,datetime,hashlib,importlib.util,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
OWN=HERE.parents[3]
REPO=Path('/home/poojith-devan/Desktop/Trainium')
assert datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,9,26,3,57,tzinfo=datetime.timezone.utc),'NOT_YET_ASSIGNED_BOUNDARY'
assert not (HERE/'trn22.json').exists(),'LOCAL_RECEIPT_ALREADY_EXISTS'
paths=[
 'research/v2/operator/20260926-trn22-clock36-recursive-full/launch/first-four-rank-evidence.json',
 'research/v2/operator/20260926-dual-full-progress-0306/trn22.json',
 'research/v2/operator/20260926-dual-full-progress-0318/trn22.json',
 'research/v2/operator/20260926-trn22-recursive-boundary-0347/trn22.json',
]
known={3252200,3252201,3252202,3252218,3252219,3252233,3252248,3252249,3252250,3252251,3252754,3252756,3252771,3252772,3252773,3252774,3274486,3274487,3274501}
identities={};prior_receipts={}
for name in paths:
    p=OWN/name
    raw=p.read_bytes();o=json.loads(raw);o=o.get('remote',o)
    prior_receipts[name]=hashlib.sha256(raw).hexdigest()
    for item in o['processes']:
        known.add(item['pid'])
        if item.get('exists', 'starttime_ticks' in item or 'starttime' in item):
            identities[item['pid']]={k:item[k] for k in ('pid','starttime','starttime_ticks','ppid','pgid','sid','argv','args') if k in item}
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'],'LOCAL_STOP'
source=(HERE/'remote_snapshot.py').read_bytes();ast.parse(source)
assert not m.SENSITIVE.search(source),'SOURCE_SENSITIVE_MARKER'
payload={'known_pids':sorted(known),'known_identity_records':list(identities.values())}
argv,*_=m.load_transport('trn22');argv[-1]='exec python3 -I -B -'
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(argv,input=('P='+repr(payload)+'\n').encode()+source,capture_output=True,timeout=60)
assert not m.SENSITIVE.search(r.stdout+r.stderr),'OUTPUT_SENSITIVE_MARKER'
receipt={'started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rc':r.returncode,'remote_source_sha256':hashlib.sha256(source).hexdigest(),'prior_receipt_sha256':prior_receipts,'local_before':state,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
with (HERE/'trn22.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
if r.returncode:raise RuntimeError('READONLY_CHECK_FAILED:'+r.stderr.decode())
o=receipt['remote']
if o['terminal_seen']:
    dest=OWN/'research/v2/operator/20260926-trn22-clock36-recursive-full/completed'
    dest.mkdir(exist_ok=False)
    with (dest/'results-bundle.json').open('x') as f:json.dump(o,f,indent=2);f.write('\n')
    with (dest/'explicit-process-postflight.json').open('x') as f:json.dump(o['postflight'],f,indent=2);f.write('\n')
print(json.dumps({'utc':o['utc'],'terminal_seen':o['terminal_seen'],'receipt_sha256':hashlib.sha256((HERE/'trn22.json').read_bytes()).hexdigest(),'terminal':o['files'].get('terminal.json',{}).get('text'),'last_eval_lines':o['files'].get('fresh-public20m.log',{}).get('text','').splitlines()[-12:],'postflight':o['postflight'],'checkpoint_after_terminal':o['checkpoint_after_terminal'],'live_pids':[p['pid'] for p in o['processes'] if p['exists']],'disk_free':o['free_bytes']},indent=2))
