"""Authenticate this immutable read-only snapshot and scoped occupancy arithmetic."""
import datetime, hashlib, json
from pathlib import Path

here = Path(__file__).resolve().parent
initial = here.parent / '20260926-final-sprint-p0'
rows = []
expected = {'trn21': (3203817, [3203864, 3203865, 3203866, 3203867], 596823),
            'trn22': (3315643, [3315691, 3315692, 3315693, 3315694], 832589)}
def require(value, error):
    if not value: raise RuntimeError(error)
def sha(raw): return hashlib.sha256(raw).hexdigest()
for host, (guardian, worker_pids, inode) in expected.items():
    raw = (here / (host + '-live.json')).read_bytes()
    receipt = json.loads(raw)
    require(receipt['returncode'] == 0 and not receipt['stderr'], 'READ_FAILED')
    r = receipt['remote']
    original = json.loads((initial / (host + '-live.json')).read_text())['remote']
    b = json.loads((initial / (host + '-bound-packet.json')).read_text())['binding']
    before = {p['identity_before']['pid']: p for p in original['processes']}
    now = {p['identity_before']['pid']: p for p in r['processes']}
    require(r['label'] == b['label'], 'LABEL')
    require(r['original_lock'] == {'device': 66305, 'inode': inode, 'held': True}, 'LOCK')
    require(r['actual_train_source']['sha256'] == b['source_sha256'], 'SOURCE')
    for pid in [guardian] + worker_pids:
        require(pid in now and pid in before, 'PID_MISSING')
        require(now[pid]['identity_before']['start_ticks'] == before[pid]['identity_before']['start_ticks'], 'PID_REUSED')
        require(now[pid]['identity_before']['start_ticks'] == now[pid]['identity_after']['start_ticks'], 'IDENTITY_RACE')
        require(now[pid]['identity_after']['state'] != 'Z', 'ZOMBIE')
    require(now[guardian]['original_lock_fds'], 'GUARDIAN_DROPPED_LOCK')
    cache_keys = ('NEURON_COMPILE_CACHE_URL', 'NEURON_COMPILE_CACHE_DIR', 'TORCH_NEURONX_NEFF_CACHE_DIR', 'TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR', 'TORCH_NEURONX_HLO_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_NEURONX_DEBUG_DIR')
    for rank, pid in enumerate(worker_pids):
        env = now[pid]['environment_whitelist']
        require(env['RANK'] == str(rank) and env['LOCAL_RANK'] == str(rank) and env['WORLD_SIZE'] == '4' and env['NEURON_LOGICAL_NC_CONFIG'] == '2', 'RANK_PROFILE')
        for key in cache_keys: require(env[key] == b['train_environment'][key], 'CACHE_ENV')
    for name, entry in r['files'].items():
        if 'text' in entry: require(sha(entry['text'].encode()) == entry['sha256'], 'CAPTURE_HASH:' + name)
    require(r['files']['terminal.json'].get('absent') is True, 'UNEXPECTED_TERMINAL')
    launch = json.loads(r['files']['training-launch.json']['text'])
    log = r['files']['training.log']['text']
    fatal = [line for line in log.splitlines() if any(term in line for term in ('Traceback (most recent call last)', 'ChildFailedError', 'RuntimeError:', 'TRAINING_ERROR', 'AssertionError:'))]
    step_lines = [line for line in log.splitlines() if line.startswith('step ')]
    compiler = [p['identity_after'] if 'identity_after' in p else p['identity_before'] for p in r['processes'] if any('neuronx-cc' in arg or 'walrus_driver' in arg for arg in p['argv'])]
    rows.append({'host': host, 'label': r['label'], 'observed_utc': r['utc'], 'training_phase_start_utc': launch['utc'], 'guardian_pid': guardian, 'worker_pids': worker_pids, 'original_lock_held': True, 'free_bytes': r['free_bytes'], 'phase': 'COLD_COMPILATION_FIRST_STEP_NOT_YET_REPORTED', 'latest_step_lines': step_lines, 'fatal_lines_observed': fatal, 'compiler_identities': compiler, 'receipt_sha256': sha(raw), 'actual_uncapped_fifth_boundary': None, 'quality_result': None})
window_start = datetime.datetime.fromisoformat('2026-09-26T07:05:49+00:00')
window_end = min(datetime.datetime.fromisoformat(row['observed_utc']) for row in rows)
duration = (window_end - window_start).total_seconds()
for row in rows:
    start = datetime.datetime.fromisoformat(row['training_phase_start_utc'])
    occupied = (window_end - max(window_start, start)).total_seconds()
    row['occupied_seconds_in_window'] = occupied
    row['occupied_percent_in_window'] = occupied / duration * 100
result = {'verdict': 'LIVE_IDENTITIES_SOURCE_RANK_CACHE_LOCK_CAPTURE_HASHES_AUTHENTICATED', 'hosts': rows,
          'occupancy_window': {'start_utc': window_start.isoformat(), 'end_utc': window_end.isoformat(), 'duration_seconds': duration, 'combined_percent': sum(row['occupied_seconds_in_window'] for row in rows)/(2*duration)*100, 'definition': 'Existing detached training phase occupies the host, including startup/compilation; not Neuron hardware utilization or useful-token throughput.', 'whole_day_utilization': None},
          'remote_actions': 'One bounded read-only snapshot per host; no launch, stop, stage, cleanup, source or cache change.'}
out = here / 'AUTHENTICATED.json'
with out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
print(json.dumps(result, indent=2))
