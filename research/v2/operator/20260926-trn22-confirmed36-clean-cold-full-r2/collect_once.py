"""Capture the released full-cold job's actual first processes without changes."""
import datetime, hashlib, importlib.util, json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
assert not (HERE/'trn22.json').exists(), 'LOCAL_RECEIPT_ALREADY_EXISTS'
spec = importlib.util.spec_from_file_location('transport', REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
source_path = HERE.parent/'20260926-trn22-confirmed36-clean-cold-five-r2/remote_snapshot.py'
source = source_path.read_text()
old = 'fs-20260926-confirmed36-clean-cold-trn22-r2'
new = 'fs-20260926-confirmed36-clean-coldfull-trn22-r2'
assert source.count(old) == 2
source = source.replace(old, new).replace('known = {3276148}', 'known = {3281907}')
source = source.replace("'dispatch.log']", "'dispatch.log','lifetime-events.jsonl','training-preflight-fuser.json','postflight-under-lock-idle.json','postflight-fuser.json','checkpoint-after-postflight.json']")
root = '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/'+new+'-packet'
payload = {'completed_parent_paths': [root+'/package/train.py', root+'/train/train.py', root+'/lifetime_guard.py'], 'completed_parent_cache_paths': []}
wire = ('P='+repr(payload)+'\n'+source).encode()
assert not m.SENSITIVE.search(wire)
argv, *_ = m.load_transport('trn22'); argv[-1] = 'exec python3 -I -B -'
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
r = subprocess.run(argv, input=wire, capture_output=True, timeout=60)
assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt = {'started_utc': started, 'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'rc': r.returncode, 'remote_source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'mechanical_changes': ['Exact full label/root and actual dispatch PID', 'Existing guardian and device-preflight receipt names', 'Hash only exact package/train source and guardian bytes'], 'stderr': r.stderr.decode(), 'remote': json.loads(r.stdout) if r.returncode == 0 else None, 'stdout_if_error': r.stdout.decode() if r.returncode else None}
with (HERE/'trn22.json').open('x') as f: json.dump(receipt, f, indent=2); f.write('\n')
if r.returncode: raise RuntimeError('READONLY_SNAPSHOT_FAILED: '+r.stderr.decode())
o = receipt['remote']
print(json.dumps({'utc': o['utc'], 'live_pids': [p['pid'] for p in o['processes'] if p['exists']], 'workers': [{'pid': p['pid'], 'ppid': p['ppid'], 'pgid': p['pgid'], 'sid': p['sid'], 'starttime': p['starttime'], 'argv': p['argv'], 'environment': p['environment_whitelist']} for p in o['processes'] if p.get('environment_whitelist', {}).get('RANK') is not None], 'lock': o['lock'], 'free_bytes': o['free_bytes'], 'guardian': o['files'].get('lifetime-events.jsonl'), 'terminal': o['files'].get('terminal.json'), 'training_tail': o['files'].get('training.log', {}).get('text', '').splitlines()[-8:], 'receipt_sha256': hashlib.sha256((HERE/'trn22.json').read_bytes()).hexdigest()}, indent=2))
