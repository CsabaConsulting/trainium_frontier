"""Authenticate retrieved current full-run receipts and emit a ledger candidate.

No host access, ledger mutation, adoption or changed decision thresholds.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys

HERE = Path(__file__).resolve().parent
host = sys.argv[1]
assert not sys.flags.optimize
repo = HERE.parents[3]
manual_key = None
if host not in ('trn21','trn22'):
    manual_key = host
    manifest = json.loads((HERE/'manual-jobs/manifest.json').read_bytes())
    entry = manifest[manual_key]
    host = entry['host']
    packet_raw = (HERE/'manual-jobs'/entry['packet']).read_bytes()
    assert hashlib.sha256(packet_raw).hexdigest() == entry['packet_sha256']
    expected_binding = json.loads(packet_raw)['binding']
    path = HERE/'manual-jobs'/manual_key/'results-bundle.json'
else:
    path = HERE / (host + '-current-completed/results-bundle.json')
raw = path.read_bytes()
bundle = json.loads(raw)
files = bundle['files']
for name, item in files.items():
    data = item['text'].encode()
    assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], name


def obj(name): return json.loads(files[name]['text'])


b = obj('NATIVE_BINDING.json')
proposal = obj('EXPERIMENT_PROPOSAL.json')
arm_id = b.get('arm_id') or proposal['arm_id']
assert not b.get('arm_id') or b['arm_id'] == proposal['arm_id']
assert b['host'] == host
if manual_key:
    for field in ('label','host','root','source_sha256','launch_command','prepare_sha256','harness_sha256','lock_identity','base_sha256'):
        assert b[field] == expected_binding[field], 'WRONG_STAGED_JOB:'+field
assert files['package/train.py']['sha256'] == b['source_sha256']
assert files['package/launch-command.txt']['text'].strip() == b['launch_command']
assert obj('operator-preflight.json')['source_sha256'] == b['source_sha256']
assert obj('terminal.json')['status'] == 'COMPLETE_PENDING_INDEPENDENT_REVIEW'
assert obj('terminal.json')['eval_exit_code'] == 0
tr, ev = obj('train/rehearsal.json'), obj('eval/rehearsal.json')
assert tr['returncode'] == ev['returncode'] == 0
assert tr['charged_through_process_exit_within_budget'] is True
assert 0 < tr['charged_through_process_exit'] <= 1800
assert b['train_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'] == '1800'
assert b['train_environment']['NEURON_COMPETITION_R1_SEED'] == '42'
assert b['train_environment']['NEURON_LOGICAL_NC_CONFIG'] == '2'
assert tr['step_boundaries'] == len(tr['observations']) > 5
startup = tr['observations'][4]['observed_seconds']
assert abs(min(startup, 900) - tr['startup_allowance_observed']) < 1e-8
assert abs(tr['wall_seconds'] - tr['startup_allowance_observed'] - tr['charged_through_process_exit']) < 1e-8
cp = bundle['checkpoint_after_terminal']
assert cp['stable'] and cp['bytes'] == tr['checkpoint_bytes'] and cp['sha256'] == tr['checkpoint_sha256']
assert obj('checkpoint.json')['sha256'] == obj('checkpoint-after-postflight.json')['sha256'] == cp['sha256']
for name, item in bundle['retained_input_identities'].items():
    assert item['stable']
    expected = b['source_sha256'] if name.endswith('/package/train.py') else b['prepare_sha256'] if name.endswith('/prepare.py') else b['harness_sha256']
    assert item['sha256'] == expected
post = bundle['postflight']
assert post['all_observed_pids_absent'] and not post['busy_job_rows'] and not post['related_rows']
assert post['neuron_device_idle'] and post['lock_competitor_rc'] == 0 and post['fuser_rc'] == 1
assert [post['lock_device'], post['lock_inode']] == b['lock_identity']
log = files['fresh-public20m.log']['text']
commands = [json.loads(line.split(' ', 1)[1]) for line in log.splitlines() if line.startswith('REHEARSAL_COMMAND ')]
assert len(commands) == 1
argv_eval = commands[0]
for flag, value in (('--eval-tokens', '20971520'), ('--seq-len', '1024'), ('--batch-size', '1')):
    assert argv_eval[argv_eval.index(flag) + 1] == value
assert 'check passed' in log
values = []
for match in re.finditer(r'\{', log):
    try: value, _ = json.JSONDecoder().raw_decode(log[match.start():])
    except json.JSONDecodeError: continue
    if isinstance(value, dict) and 'val_bpb' in value:
        values.append(value)
assert len(values) == 1
score = values[0]
assert score['num_tokens'] == 20971520 and score['eval_over_timeout'] is False
assert math.isfinite(score['val_bpb']) and abs(score['total_nats'] / (math.log(2) * score['num_bytes']) - score['val_bpb']) < 1e-12
cold = bool(b['fresh_cache_directories'])
if cold:
    created = obj('operator-preflight.json')['fresh_cache_absent_then_created_empty']
    assert len(created) == 6 and all(x['created_empty'] for x in created)
    assert {x['path'] for x in created} == set(b['fresh_cache_directories'])
control = .9869854734569898 if host == 'trn21' else .9867730678453942
tokens = shlex.split(b['launch_command'])
row = dict(schema='trn-run-record-v2', label=b['label'], arm_id=arm_id, kind=b['kind'], run_purpose='quality', host=host,
           timestamp_utc=obj('terminal.json')['utc'], base_sha256=b['base_sha256'], train_py_sha256=b['source_sha256'],
           status='evaluated', git_dirty=False, git_dirty_scope='Authenticated exact staged scientific source; not whole-worktree cleanliness',
           argv=tokens[1:], launch_command=b['launch_command'], env=b['train_environment'], training_budget_seconds=1800,
           over_budget=False, steps=tr['step_boundaries'], charged_through_exit_seconds=tr['charged_through_process_exit'],
           startup_measured_seconds=startup, startup_allowance_seconds=tr['startup_allowance_observed'], startup_excess_seconds=max(0, startup-900),
           cold_startup_seconds=startup if cold else None, cold_startup_proven=cold, causality_passed=None,
           eval_tokens=20971520, eval_seq_len=1024, eval_batch_size=1, public_20m=score['val_bpb'],
           eval_num_bytes=score['num_bytes'], eval_total_nats=score['total_nats'], checkpoint_sha256=cp['sha256'], checkpoint_bytes=cp['bytes'],
           raw_archive=str(path.relative_to(repo)), raw_archive_sha256=hashlib.sha256(raw).hexdigest(), submission_eligible=False,
           same_host_comparator_public_20m=control, scientific_delta=score['val_bpb']-control,
           notes='Actual fixed1800/fresh20M result. Existing comparator ledger roles retained; raw scientific delta is not a fabricated formal-control adoption.')
if manual_key:
    row['cold_startup_evidence'] = str(path.relative_to(repo)) + '#operator-preflight.json/fresh_cache_absent_then_created_empty;train/rehearsal.json/observations[4]' if cold else None
    row['cold_startup_scope'] = 'Actual private-cache empty-before-start proof' if cold else 'Warm research run; not submission-cold evidence'
    row['manual_job_key'] = manual_key
    out = HERE/'manual-jobs'/manual_key/'ledger-candidate.json'
else:
    out = HERE / (host + '-current-ledger-candidate.json')
encoded = (json.dumps(row, indent=2) + '\n').encode()
if out.exists(): assert out.read_bytes() == encoded
else:
    with out.open('xb') as stream: stream.write(encoded)
print(json.dumps({'host': host, 'public_20m': score['val_bpb'], 'delta': row['scientific_delta'], 'steps': row['steps'],
                  'startup': startup, 'charged': row['charged_through_exit_seconds'], 'cold': cold, 'ledger_candidate': str(out.relative_to(repo)), 'files_verified': len(files)}))
