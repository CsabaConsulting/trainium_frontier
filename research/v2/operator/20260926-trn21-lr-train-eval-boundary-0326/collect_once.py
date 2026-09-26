"""One scoped read-only host observation; local receipt writes only."""
import datetime,hashlib,importlib.util,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
assert not (HERE/'trn21.json').exists(),'LOCAL_RECEIPT_ALREADY_EXISTS'
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'],'LOCAL_STOP'
parentrel='research/v2/proposals/20260926-startup-adjusted-full-trn21-replication/NATIVE_BINDING.disabled.json'
parentraw=subprocess.check_output(['git','show','d0a2b1e79621384a656266688090c16da1b1bb2c:'+parentrel],cwd=REPO)
parent=json.loads(parentraw)
cachekeys=('TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR')
payload={'completed_parent_paths':[parent['root']+'/train/train.py',parent['root']+'/train/prepare.py'],'completed_parent_cache_paths':[parent['training_environment'][k] for k in cachekeys]}
source=(HERE/'remote_snapshot.py').read_bytes()
assert not m.SENSITIVE.search(source),'SOURCE_SENSITIVE_MARKER'
argv,*_=m.load_transport('trn21');argv[-1]='exec python3 -I -B -'
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(argv,input=('P='+repr(payload)+'\n').encode()+source,capture_output=True,timeout=60)
assert not m.SENSITIVE.search(r.stdout+r.stderr),'OUTPUT_SENSITIVE_MARKER'
receipt={'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rc':r.returncode,'remote_source_sha256':hashlib.sha256(source).hexdigest(),'parent_binding_sha256':hashlib.sha256(parentraw).hexdigest(),'local_before':state,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
with (HERE/'trn21.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
if r.returncode:raise RuntimeError('READONLY_SNAPSHOT_FAILED:'+r.stderr.decode())
o=receipt['remote']
summary={'utc':o['utc'],'receipt':str(HERE/'trn21.json'),'receipt_sha256':hashlib.sha256((HERE/'trn21.json').read_bytes()).hexdigest(),'live_pids':[p['pid'] for p in o['processes'] if p['exists']],'lock':o['lock'],'free_bytes':o['free_bytes'],'files':list(o['files']),'last_training_lines':o['files'].get('training.log',{}).get('text','').splitlines()[-8:],'last_eval_lines':o['files'].get('fresh-public20m.log',{}).get('text','').splitlines()[-10:]}
print(json.dumps(summary,indent=2))
