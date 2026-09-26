"""One protected-job handoff observation and one exact read-only path check."""
import datetime,hashlib,importlib.util,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=Path('/home/poojith-devan/Desktop/Trainium')
assert not (HERE/'trn22.json').exists(),'LOCAL_RECEIPT_EXISTS'
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
body=(HERE.parent/'20260926-trn22-confirmed36-clean-cold-five-r2/remote_snapshot.py').read_text()
folder=HERE.parent/'20260926-trn22-confirmed36-clean-cold-full-r2';known=set();pins=[]
for path in [folder/'trn22.json',folder/'progress/boundary-0509.json',HERE.parent/'20260926-dual-status-0523/trn22.json',HERE.parent/'20260926-trn22-status-0537/trn22.json']:
 raw=path.read_bytes();pins.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()});o=json.loads(raw)['remote']
 known.update(p['pid'] for p in o['processes'] if p.get('exists'))
 receipt=o['files'].get('lifetime-events.jsonl')
 if receipt and receipt['capture']=='complete_at_observation':
  for line in receipt['text'].splitlines():
   e=json.loads(line)
   if 'identity' in e:known.add(e['identity']['pid'])
assert {3281907,3281908,3281909,3281925,3281926,3281940,3281955,3281956,3281957,3281958}<=known
body=body.replace('fs-20260926-confirmed36-clean-cold-trn22-r2','fs-20260926-confirmed36-clean-coldfull-trn22-r2').replace('known = {3276148}','known = '+repr(known))
body=body.replace("'dispatch.log']","'dispatch.log','lifetime-events.jsonl','training-preflight-fuser.json','eval-preflight-fuser.json','postflight-under-lock-idle.json','postflight-fuser.json','checkpoint-after-postflight.json']")
# The one additional requested path is lstat-only if missing, symlinked or non-regular.
needle='        st=p.lstat()'
# The optional-path body uses s rather than st; leave normal artifact reads unchanged.
needle="        p=Path(path);s=p.lstat()\n"
replacement=needle+"        if stat.S_ISLNK(s.st_mode):\n            optional[path]={'exists':True,'symlink':True,'hash':None};continue\n"
assert body.count(needle)==1;body=body.replace(needle,replacement)
body=body.replace("optional[path]={'device':s.st_dev", "optional[path]={'exists':True,'symlink':False,'device':s.st_dev")
target='/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/prepare.py'
wire=('P='+repr({'completed_parent_paths':[target],'completed_parent_cache_paths':[]})+'\n'+body).encode()
assert not m.SENSITIVE.search(wire)
argv,*_=m.load_transport('trn22');argv[-1]='exec python3 -I -B -'
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(argv,input=wire,capture_output=True,timeout=60);assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt={'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'remote_source_sha256':hashlib.sha256(wire).hexdigest(),'prior_pins':pins,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
p=HERE/'trn22.json'
with p.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
if r.returncode:print(json.dumps(receipt));raise SystemExit(r.returncode)
o=receipt['remote'];files=o['files']
for name,value in files.items():
 raw=value['text'].encode();assert len(raw)==value['captured_bytes'] and hashlib.sha256(raw).hexdigest()==value['sha256_captured'],'CAPTURE_IDENTITY:'+name
def parsed(name):return json.loads(files[name]['text']) if name in files and files[name]['capture']=='complete_at_observation' else None
training=parsed('train/rehearsal.json')
if training:training={k:v for k,v in training.items() if k!='observations'}
print(json.dumps({'utc':o['utc'],'live_processes':[{'pid':x['pid'],'ppid':x['ppid'],'pgid':x['pgid'],'sid':x['sid'],'starttime':x['starttime'],'argv':x['argv']} for x in o['processes'] if x.get('exists')],'lock':o['lock'],'free_bytes':o['free_bytes'],'training_rehearsal':training,'training_exit':files.get('run-exit-code.txt'),'checkpoint_record':parsed('checkpoint.json'),'training_tail':files.get('training.log',{}).get('text','').splitlines()[-12:],'terminal':parsed('terminal.json'),'eval_launch':parsed('fresh-public20m-launch.json'),'eval_tail':files.get('fresh-public20m.log',{}).get('text','').splitlines()[-8:],'additional_exact_path':o['completed_parent_metadata_only'],'captured_files_rehashed':len(files),'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
