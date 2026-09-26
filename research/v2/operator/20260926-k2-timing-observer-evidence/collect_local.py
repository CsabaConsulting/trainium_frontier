"""Read already-existing local evidence only; never imports or executes harness."""
import ast, hashlib, json
from pathlib import Path
here = Path(__file__).resolve().parent
source_path = Path('/home/poojith-devan/clean-fullcold-review-DrcoWi/submission_rehearsal.py')
raw = source_path.read_bytes()
expected = '7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b'
if hashlib.sha256(raw).hexdigest() != expected: raise RuntimeError('HARNESS_PIN_MISMATCH')
text = raw.decode()
required = ('while len(observed) < len(steps):', 'observed_seconds=time.monotonic() - started', 'time.sleep(.05)', 'checkpoint_sha256=sha(checkpoint), observations=observed)', '(sandbox / "rehearsal.json").write_text')
for fragment in required:
    if fragment not in text: raise RuntimeError('EXPECTED_OBSERVER_SOURCE_ABSENT')
ast.parse(text)
rows = []
for relative in ('20260926-trn21-batch262144-timing/completed/train/rehearsal.json', '20260926-trn22-arch3e79-full/completed-training/train/rehearsal.json'):
    path = here.parent / relative
    value = path.read_bytes(); report = json.loads(value); observations = report['observations']
    if report['returncode'] != 0 or [row['step'] for row in observations] != list(range(report['step_boundaries'])): raise RuntimeError('HISTORICAL_OBSERVER_SEQUENCE')
    rows.append({'path': str(path), 'sha256': hashlib.sha256(value).hexdigest(), 'returncode': report['returncode'], 'boundary_count': report['step_boundaries'], 'observation_count': len(observations), 'step9': observations[9], 'step44': observations[44], 'step44_minus_step9_seconds': observations[44]['observed_seconds'] - observations[9]['observed_seconds'], 'last_observation': observations[-1]})
evidence = {'scope': 'LOCAL_READ_ONLY_EXISTING_HARNESS_AND_TERMINAL_RECEIPTS', 'harness': {'path': str(source_path), 'sha256': expected, 'bytes': len(raw), 'source_text': text}, 'historical_receipts': rows, 'result': 'ALL_OBSERVED_BOUNDARY_TIMES_PERSIST_ON_SUCCESSFUL_TRAIN_TERMINAL', 'poll_seconds': .05, 'live_steps_file_has_indices_only': True, 'k2_metric_supported_unchanged': '35 * 131072 / (observations[44].observed_seconds - observations[9].observed_seconds)', 'new_observer_needed': False, 'harness_modified': False, 'host_calls': 0}
with (here / 'EVIDENCE.json').open('x') as f: json.dump(evidence, f, indent=2); f.write('\n')
print(json.dumps({'harness_sha256': expected, 'historical_receipts': rows, 'new_observer_needed': False, 'host_calls': 0}))
