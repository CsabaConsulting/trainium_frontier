"""One read-only boundary observation using the prior scoped snapshot body."""
import datetime, hashlib, importlib.util, json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
assert not (HERE/'trn21.json').exists(), 'LOCAL_RECEIPT_ALREADY_EXISTS'
spec = importlib.util.spec_from_file_location('transport', REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
source_path = HERE.parent/'20260926-trn21-anchored-boundary-0439/remote_snapshot.py'
source = source_path.read_text()
old = 'known = {3177618,3177619,3177620,3177651,3177652,3177666,3177681,3177682,3177683,3177684}'
new = old[:-1] + ',3199120,3199121,3199135}'
assert source.count(old) == 1
source = source.replace(old, new)
payload = {'completed_parent_paths': [], 'completed_parent_cache_paths': []}
wire = ('P='+repr(payload)+'\n'+source).encode()
assert not m.SENSITIVE.search(wire)
argv, *_ = m.load_transport('trn21'); argv[-1] = 'exec python3 -I -B -'
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
r = subprocess.run(argv, input=wire, capture_output=True, timeout=60)
assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt = {'started_utc': started, 'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'rc': r.returncode, 'remote_source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'mechanical_change': 'Add three previously observed eval process identities to the prior snapshot floor.', 'stderr': r.stderr.decode(), 'remote': json.loads(r.stdout) if r.returncode == 0 else None, 'stdout_if_error': r.stdout.decode() if r.returncode else None}
with (HERE/'trn21.json').open('x') as f: json.dump(receipt, f, indent=2); f.write('\n')
if r.returncode: raise RuntimeError('READONLY_SNAPSHOT_FAILED: '+r.stderr.decode())
o = receipt['remote']
print(json.dumps({'utc': o['utc'], 'live_pids': [p['pid'] for p in o['processes'] if p['exists']], 'lock': o['lock'], 'free_bytes': o['free_bytes'], 'terminal': o['files'].get('terminal.json'), 'eval_tail': o['files'].get('fresh-public20m.log', {}).get('text', '').splitlines()[-12:], 'receipt_sha256': hashlib.sha256((HERE/'trn21.json').read_bytes()).hexdigest()}, indent=2))
