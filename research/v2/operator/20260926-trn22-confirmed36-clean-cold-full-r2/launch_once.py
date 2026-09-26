"""Record the single released, exact full-cold invocation; no automatic retry."""
import datetime, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
dest = HERE/'launch'; dest.mkdir(exist_ok=False)
argv = [sys.executable, '-B', str(HERE/'stage-dispatch.py'), 'aa43bd8f0e8b558cad40828dcf2672489f107465', '5c8ccf4974a5667326dda26af3ef346e656acfd1', 'research/v2/verification/20260926-confirmed36-clean-fullcold-trn22-assignment.md']
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
r = subprocess.run(argv, capture_output=True, text=True, timeout=90)
receipt = {'started_utc': started, 'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': argv, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr, 'no_retry': True}
with (dest/'stage-transport.json').open('x') as f: json.dump(receipt, f, indent=2); f.write('\n')
if r.returncode: print(json.dumps(receipt)); raise SystemExit(r.returncode)
result = json.loads(r.stdout)
if result['rc']: print(json.dumps(result)); raise SystemExit(result['rc'])
remote = json.loads(result['stdout'])
print(json.dumps({'launch': remote['launch'], 'free_bytes': remote['preflight']['free_bytes'], 'twelve_new_cache_paths_absent': remote['preflight']['twelve_new_cache_paths_absent'], 'lock_device': remote['preflight']['lock_device'], 'lock_inode': remote['preflight']['lock_inode'], 'job_lines': remote['preflight']['job_lines'], 'neuron_ls': remote['preflight']['neuron_ls']}, indent=2))
