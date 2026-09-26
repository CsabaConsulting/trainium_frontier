"""Read-only P0 terminal retrieval using the already-executed postflight body."""
import ast,datetime,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
host=sys.argv[1];assert host in ('trn21','trn22')
dest=HERE/(host+'-p0-completed');assert not dest.exists()
old=HERE.parent/'20260926-trn22-confirmed36-clean-cold-five-r2/retrieve_final.py'
raw=old.read_bytes();assert hashlib.sha256(raw).hexdigest()=='d71d9b710b33a5e8f9051d079e3e8a7d13b83676f3de16d32be02ae5cfb1c482'
tree=ast.parse(raw);body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
packet=json.loads((HERE.parent/'20260926-final-sprint-p0'/(host+'-bound-packet.json')).read_text());b=packet['binding']
oldroot='/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-cold-trn22-r2-packet'
assert body.count(oldroot)==1;body=body.replace(oldroot,b['root'])
body=body.replace('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py',b['prepare_path']).replace('(66305,832589)',repr(tuple(b['lock_identity'])))
needle="identities={};checkpoints={};known=set(P['known_prior_identities']);files={};total=0"
extra=['operator-preflight.json','NATIVE_BINDING.json','RELEASE.json','lifetime_guard.py','lifetime-events.jsonl','training-preflight-fuser.json','fresh-public20m-launch.json','fresh-public20m.log','eval/rehearsal.json','eval/stdout.log','eval/stderr.log','eval-exit-code.txt','checkpoint.json','checkpoint-after-postflight.json','eval-under-lock-idle.json','eval-preflight-fuser.json','postflight-under-lock-idle.json','postflight-fuser.json','independent-release-idle.json']
# Include immediate small hidden/partial receipts, never descend into cache or
# checkpoint directories. The inherited per-file/total bounds remain intact.
more="\nfor folder in (root,root/'train',root/'eval',root/'metrics-training'):\n if folder.is_dir():\n  for p in folder.iterdir():\n   if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt','.csv'):\n    names.append(str(p.relative_to(root)))\nassert len(set(names))<=120,'RECEIPT_COUNT_BOUND'\n"
assert body.count(needle)==1;body=body.replace(needle,'names += '+repr(extra)+more+needle)
body=body.replace('current r2 live/boundary receipts','P0 launch/startup/milestone/handoff/result receipts')
paths=[];identity_map={}
for dirname in ('20260926-final-sprint-p0','20260926-final-sprint-p0-startup','20260926-final-sprint-p0-milestone','20260926-final-sprint-p0-train-eval','20260926-final-sprint-p0-handoff','20260926-final-sprint-p0-result-check'):
 p=HERE.parent/dirname/(host+'-live.json')
 if not p.is_file():continue
 paths.append(p);remote=json.loads(p.read_text())['remote']
 for proc in remote['processes']:
  ident=proc['identity_before'];pid=ident['pid'];v={'pid':pid,'starttime':ident['start_ticks'],'ppid':ident['ppid'],'pgid':ident['pgid'],'sid':ident['sid'],'exists':True,'argv':proc['argv']}
  if pid in identity_map:assert identity_map[pid]['starttime']==v['starttime'],'OBSERVED_PID_REUSE'
  identity_map[pid]=v
assert len(paths)>=5 and len(identity_map)>=13
payload={'known_prior_identities':identity_map};wire=('P='+repr(payload)+'\n'+body).encode()
spec=importlib.util.spec_from_file_location('connector',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert not m.SENSITIVE.search(wire);ast.parse(body)
if sys.argv[2:]==['--local-check']:
 print(json.dumps({'host':host,'status':'LOCAL_CHECK_PASS','observed_identities':len(identity_map),'payload_sha256':hashlib.sha256(wire).hexdigest()}));raise SystemExit
assert sys.argv[2:]==['--execute']
argv,*_=m.load_transport(host);argv[-1]='exec /usr/bin/python3 -I -B -S -'
r=subprocess.run(argv,input=wire,capture_output=True,timeout=60);assert not m.SENSITIVE.search(r.stdout+r.stderr)
if r.returncode:
 err={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rc':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode()}
 with (HERE/(host+'-retrieval-error.json')).open('x') as f:json.dump(err,f,indent=2)
 print(json.dumps(err));raise SystemExit(r.returncode)
o=json.loads(r.stdout);o['local_retrieval']={'prior_receipts':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],'body_sha256':hashlib.sha256(body.encode()).hexdigest()}
dest.mkdir()
for n,v in [('results-bundle.json',o),('explicit-process-postflight.json',o['postflight']),('retained-input-identities.json',o['retained_input_identities'])]:
 with (dest/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
post=o['postflight'];terminal=json.loads(o['files']['terminal.json']['text'])
print(json.dumps({'host':host,'utc':o['utc'],'terminal':terminal,'postflight':{k:post[k] for k in ('all_observed_pids_absent','related_rows','busy_job_rows','neuron_device_idle','fuser_rc','lock_competitor_rc','lock_device','lock_inode')},'observed_identities':len(identity_map),'checkpoint':o['checkpoint_after_terminal'],'free_bytes':o['free_bytes'],'bundle_sha256':hashlib.sha256((dest/'results-bundle.json').read_bytes()).hexdigest()}))
