"""One assigned bounded read-only observation; preserve the protected full replay."""
import datetime, hashlib, importlib.util, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
assert not (HERE/'trn22.json').exists(),'LOCAL_RECEIPT_EXISTS'
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
body=(HERE.parent/'20260926-trn22-confirmed36-clean-cold-five-r2/remote_snapshot.py').read_text()
folder=HERE.parent/'20260926-trn22-confirmed36-clean-cold-full-r2'
known=set();pins=[]
for path in [folder/'trn22.json',folder/'progress/boundary-0509.json',HERE.parent/'20260926-dual-status-0523/trn22.json']:
 raw=path.read_bytes();pins.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()});o=json.loads(raw)['remote']
 known.update(p['pid'] for p in o['processes'] if p.get('exists'))
 receipt=o['files'].get('lifetime-events.jsonl')
 if receipt and receipt['capture']=='complete_at_observation':
  for line in receipt['text'].splitlines():
   e=json.loads(line)
   if 'identity' in e:known.add(e['identity']['pid'])
assert {3281907,3281908,3281909,3281925,3281926,3281940,3281955,3281956,3281957,3281958}<=known
body=body.replace('fs-20260926-confirmed36-clean-cold-trn22-r2','fs-20260926-confirmed36-clean-coldfull-trn22-r2').replace('known = {3276148}','known = '+repr(known))
body=body.replace("'dispatch.log']","'dispatch.log','lifetime-events.jsonl','training-preflight-fuser.json','postflight-under-lock-idle.json','postflight-fuser.json','checkpoint-after-postflight.json']")
wire=('P='+repr({'completed_parent_paths':[],'completed_parent_cache_paths':[]})+'\n'+body).encode()
assert not m.SENSITIVE.search(wire)
argv,*_=m.load_transport('trn22');argv[-1]='exec python3 -I -B -'
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(argv,input=wire,capture_output=True,timeout=60)
assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt={'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'remote_source_sha256':hashlib.sha256(wire).hexdigest(),'prior_pins':pins,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
p=HERE/'trn22.json'
with p.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
if r.returncode:print(json.dumps(receipt));raise SystemExit(r.returncode)
o=receipt['remote'];rows=o['files'].get('training.log',{}).get('text','').splitlines()
print(json.dumps({'utc':o['utc'],'live_pids':[x['pid'] for x in o['processes'] if x.get('exists')],'workers':[{'pid':x['pid'],'starttime':x['starttime'],'rank':x['environment_whitelist']['RANK']} for x in o['processes'] if x.get('environment_whitelist',{}).get('RANK') is not None],'lock':o['lock'],'free_bytes':o['free_bytes'],'training_tail':rows[-8:],'training_errors':[x for x in rows if any(k in x for k in ('Traceback','RuntimeError','AssertionError','FATAL'))],'training_rehearsal_present':'train/rehearsal.json' in o['files'],'terminal':o['files'].get('terminal.json'),'eval_tail':o['files'].get('fresh-public20m.log',{}).get('text','').splitlines()[-8:],'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
