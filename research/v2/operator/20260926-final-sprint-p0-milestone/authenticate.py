"""Authenticate existing live receipts, then summarize actual printed progress."""
import datetime, hashlib, json, re
from pathlib import Path

here = Path(__file__).resolve().parent
parent = here.parent / '20260926-final-sprint-p0-startup/authenticate.py'
# The previous authentication verifies original identities, source, rank/env,
# held lock and captured hashes. Only its archive location and observed phase
# description change; no new scientific qualification is performed.
text = parent.read_text().replace("'COLD_COMPILATION_FIRST_STEP_NOT_YET_REPORTED'", "'TRAINING_AFTER_COLD_STARTUP'")
exec(compile(text, str(parent), 'exec'), {'__file__': str(here / 'authenticate.py'), '__name__': '__main__'})
rows = []
pattern = re.compile(r'^step (\d+) \| loss ([\d.]+) \| lrm ([\d.]+) \| dt ([\d.]+)s \| tok/s ([\d,]+).*remaining_time (\d+)s', re.M)
for host in ('trn21', 'trn22'):
    receipt = json.loads((here / (host + '-live.json')).read_text())
    r = receipt['remote']
    matches = list(pattern.finditer(r['files']['training.log']['text']))
    if not matches: raise RuntimeError('NO_ACTUAL_STEP_PROGRESS')
    first = [m for m in matches if int(m[1]) < 5]
    if [int(m[1]) for m in first] != list(range(5)): raise RuntimeError('FIRST_FIVE_INCOMPLETE')
    step = matches[-1]
    boundary_file = r['files']['existing-observer-artifacts']['files']['steps.jsonl']
    if hashlib.sha256(boundary_file['text'].encode()).hexdigest() != boundary_file['sha256']: raise RuntimeError('BOUNDARY_HASH')
    boundaries = [int(line) for line in boundary_file['text'].splitlines()]
    if boundaries != list(range(len(boundaries))): raise RuntimeError('BOUNDARY_SEQUENCE')
    observed = datetime.datetime.fromisoformat(r['utc'])
    eta = observed + datetime.timedelta(seconds=int(step[6]))
    rows.append({'host': host, 'observed_utc': r['utc'], 'latest_printed_step': int(step[1]), 'latest_loss': float(step[2]), 'latest_tokens_per_second': int(step[5].replace(',', '')), 'source_remaining_seconds': int(step[6]), 'actual_step_indices_through': boundaries[-1], 'first_five_printed_dt_seconds': [float(m[4]) for m in first], 'first_five_printed_dt_sum_seconds': sum(float(m[4]) for m in first), 'exact_harness_uncapped_fifth_boundary': None, 'timing_caveat': 'Printed step durations exclude other process/startup overhead. Existing observer step file has indices only; exact launch-to-fifth time is preserved in harness memory until terminal rehearsal.json.', 'source_remaining_at_step4_seconds': int(first[-1][6]), 'source_based_train_eta_utc': eta.isoformat(), 'eta_is_estimate_not_deadline': True, 'free_bytes': r['free_bytes']})
history = []
for folder in ('20260926-trn22-eval-boundary-0557', '20260926-trn21-anchored-boundary-0450'):
    path = here.parent / folder / 'completed/results-bundle.json'
    bundle = json.loads(path.read_text())
    raw = bundle['files']['eval/rehearsal.json']['text'].encode()
    if hashlib.sha256(raw).hexdigest() != bundle['files']['eval/rehearsal.json']['sha256']: raise RuntimeError('HISTORICAL_EVAL_PIN')
    report = json.loads(raw)
    history.append({'local_bundle': str(path), 'eval_rehearsal_sha256': hashlib.sha256(raw).hexdigest(), 'eval_returncode': report['returncode'], 'wall_seconds': report['wall_seconds']})
result = {'hosts': rows, 'eta': {'training_utc_range': ['2026-09-26T07:49:00+00:00', '2026-09-26T07:51:00+00:00'], 'training_ist_range': ['13:19', '13:21'], 'fresh20m_utc_range_if_comparable_eval_duration': ['2026-09-26T08:00:00+00:00', '2026-09-26T08:06:00+00:00'], 'fresh20m_ist_range_if_comparable_eval_duration': ['13:30', '13:36'], 'basis': 'Current source remaining clocks plus prior 8-layer fresh20M eval walls of about11.7 and12.2 minutes; allowance broadened to11–15 minutes, not a guarantee.', 'historical_eval_evidence': history}, 'no_native_successor': True, 'no_quality_result_yet': True}
with (here / 'MILESTONE.json').open('x') as f: json.dump(result, f, indent=2); f.write('\n')
print(json.dumps(result, indent=2))
