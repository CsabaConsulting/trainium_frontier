"""One no-earlier-than-05:09 UTC read-only observation of the protected full run."""
import datetime, hashlib, importlib.util, json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
earliest = datetime.datetime(2026,9,26,5,9,tzinfo=datetime.timezone.utc)
assert datetime.datetime.now(datetime.timezone.utc) >= earliest, 'DO_NOT_OBSERVE_BEFORE_0509_UTC'
dest = HERE/'progress'/'boundary-0509.json'
assert not dest.exists(), 'LOCAL_RECEIPT_ALREADY_EXISTS'
spec = importlib.util.spec_from_file_location('transport', REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
source_path = HERE.parent/'20260926-trn22-confirmed36-clean-cold-five-r2/remote_snapshot.py'
source = source_path.read_text()
old = 'fs-20260926-confirmed36-clean-cold-trn22-r2'
new = 'fs-20260926-confirmed36-clean-coldfull-trn22-r2'
assert source.count(old) == 2
prior = json.loads((HERE/'trn22.json').read_bytes())['remote']
known = {p['pid'] for p in prior['processes'] if p.get('exists')}
events = [json.loads(line) for line in prior['files']['lifetime-events.jsonl']['text'].splitlines()]
known.update(e['identity']['pid'] for e in events if 'identity' in e)
assert {3281907,3281908,3281909,3281925,3281926,3281940,3281955,3281956,3281957,3281958} <= known
source = source.replace(old, new).replace('known = {3276148}', 'known = '+repr(known))
source = source.replace("'dispatch.log']", "'dispatch.log','lifetime-events.jsonl','training-preflight-fuser.json','postflight-under-lock-idle.json','postflight-fuser.json','checkpoint-after-postflight.json']")
wire = ('P='+repr({'completed_parent_paths': [], 'completed_parent_cache_paths': []})+'\n'+source).encode()
assert not m.SENSITIVE.search(wire)
argv, *_ = m.load_transport('trn22'); argv[-1] = 'exec python3 -I -B -'
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
r = subprocess.run(argv, input=wire, capture_output=True, timeout=60)
assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt = {'started_utc': started, 'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'rc': r.returncode, 'remote_source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'prior_snapshot_sha256': hashlib.sha256((HERE/'trn22.json').read_bytes()).hexdigest(), 'known_identity_count_including_guardian': len(known), 'stderr': r.stderr.decode(), 'remote': json.loads(r.stdout) if r.returncode == 0 else None, 'stdout_if_error': r.stdout.decode() if r.returncode else None}
dest.parent.mkdir(exist_ok=True)
with dest.open('x') as f: json.dump(receipt, f, indent=2); f.write('\n')
if r.returncode: raise RuntimeError('READONLY_SNAPSHOT_FAILED: '+r.stderr.decode())
o = receipt['remote']
log = o['files'].get('training.log', {}).get('text', '')
rows = log.splitlines()
print(json.dumps({'utc': o['utc'], 'live_pids': [p['pid'] for p in o['processes'] if p['exists']], 'workers': [{'pid': p['pid'], 'ppid': p['ppid'], 'pgid': p['pgid'], 'sid': p['sid'], 'starttime': p['starttime'], 'rank': p['environment_whitelist']['RANK']} for p in o['processes'] if p.get('environment_whitelist', {}).get('RANK') is not None], 'lock': o['lock'], 'free_bytes': o['free_bytes'], 'terminal': o['files'].get('terminal.json'), 'rehearsal_present': 'train/rehearsal.json' in o['files'], 'steps_receipt': o['files'].get('train/steps.jsonl'), 'first_step_lines': [x for x in rows if 'step' in x.lower() and 'loss' in x.lower()][:6], 'training_tail': rows[-10:], 'receipt_sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}, indent=2))
