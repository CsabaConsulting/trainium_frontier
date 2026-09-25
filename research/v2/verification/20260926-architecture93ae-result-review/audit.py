"""Offline raw-result authentication and unchanged-selector query; no ledger edits."""
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
NATIVE = ROOT / 'research/v2/operator/20260925-trn22-l2-93ae-cpu-r1/native'
FINAL = NATIVE / 'final-evaluation'
SOURCE = '93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5'
BASE = 'c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586'
PREP = '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb'
SELECTOR = 'fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac'
LEDGER = '6cf6b8850ca50be18d649daf9e37e6adc44e2d7d0cf32503532c1228bbeed9b2'
LABEL = 'fs-20260925-l2-unet8x1024-93ae-trn22-full-r1'
checks = []

def require(ok, name):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def j(path):
    return json.loads(path.read_text())

retrieval = j(FINAL / 'retrieval.json')
for name, meta in retrieval['files'].items():
    path = FINAL / name
    require(path.stat().st_size == meta['bytes'] and sha(path) == meta['sha256'], 'final hash:' + name)
launch = j(FINAL / 'training-launch.json')
evaluation_launch = j(FINAL / 'fresh-public20m-launch.json')
job = j(NATIVE / 'native-job.ready.json')
terminal = j(FINAL / 'terminal.json')
preflight = j(FINAL / 'preflight.json')
checkpoint = j(FINAL / 'checkpoint.json')
source = subprocess.check_output(['git', 'show', '1e21c3d797eed764f108c41546b270065ad5650c:workflow/proposals/research-v2/train.py'], cwd=ROOT)
require(hashlib.sha256(source).hexdigest() == SOURCE, 'exact committed source')
require(sha(ROOT / 'prepare.py') == PREP, 'unchanged prepare')
require(launch['train_sha256'] == evaluation_launch['train_sha256'] == preflight['source_sha256'] == retrieval['source_sha256'] == SOURCE, 'all source bindings')
require(launch['prepare_sha256'] == evaluation_launch['prepare_sha256'] == preflight['prepare_sha256'] == PREP, 'all prepare bindings')
require(job['base_sha256'] == BASE and job['source_sha256'] == SOURCE and job['kind'] == 'architecture_arm', 'architecture old-base attribution')
require(job['label'] == launch['label'] == terminal['label'] == evaluation_launch['label'] == LABEL, 'unique label')
require(launch['command'] == job['command'] and launch['command'][4:] == job['argv'], 'exact training argv')
require(launch['environment'] == job['environment'], 'exact archived training environment')
require(preflight['idle'] and preflight['device_clear'] and preflight['common_lock_available'], 'archived idle preflight')
require(not preflight['cold_start_proven'], 'warm not cold')
require(terminal['status'] == 'COMPLETE' and terminal['eval_exit_code'] == 0 and int((FINAL / 'run-exit-code.txt').read_text()) == 0 and int((FINAL / 'eval-exit-code.txt').read_text()) == 0, 'both exits zero and complete')
require(not retrieval['relevant_processes'] and all(v == '' for v in retrieval['device_holders'].values()), 'retrieved quiescence')
for key, value in [('NEURON_LOGICAL_NC_CONFIG', '2'), ('NEURON_COMPETITION_R1_SEED', '42'), ('NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS', '1800')]:
    require(launch['environment'][key] == value, 'env:' + key)

train = (FINAL / 'training.log').read_text()
require((FINAL / 'training.log').read_bytes() == (NATIVE / 'completed-training/training.log').read_bytes(), 'training unchanged from exit review')
require((FINAL / 'training-launch.json').read_bytes() == (NATIVE / 'steady-progress-1814/training-launch.json').read_bytes(), 'launch unchanged from timing review')
metrics = {}
for key, value in [('num_steps', 2246), ('training_seconds', 1784.6), ('charged_seconds_est', 1785.0), ('startup_allowance_est', 900.0)]:
    metrics[key] = float(re.search(r'^' + key + r':\s*(\S+)', train, re.M).group(1))
    require(metrics[key] == value, 'source metric:' + key)
require(re.search(r'^over_budget:\s*False$', train, re.M) is not None, 'source over_budget false')
losses = [float(x) for x in re.findall(r'^step \d+ \| loss (\S+)', train, re.M)]
require(bool(losses) and all(math.isfinite(x) for x in losses) and 'Traceback' not in train, 'finite logged training losses')
first = re.findall(r'^step (\d+) .*?dt ([0-9.]+)s', train, re.M)[:5]
require([int(x[0]) for x in first] == list(range(5)), 'first five step intervals')
startup_lower = sum(float(x[1]) for x in first) - .005 * 5
require(abs(startup_lower - 1544.745) < 1e-9 and startup_lower > 900, 'startup cap saturation proven')
def boundary(text, name):
    return dt.datetime.fromisoformat(re.search('^' + name + '=(.+)$', text, re.M).group(1))
start = boundary(train, 'TRAINING_START_UTC')
end = boundary(train, 'TRAINING_END_UTC')
wall = (end - start).total_seconds()
charged_upper = wall - 900
require(abs(wall - 2663.505610) < 1e-9 and charged_upper <= 1800, 'UTC through-exit enclosure budget')

eval_text = (FINAL / 'fresh-public20m.log').read_text()
evaluation, _ = json.JSONDecoder().raw_decode(eval_text[eval_text.index('{\n'):])
require(evaluation['num_tokens'] == 20971520 and evaluation['num_bytes'] == 83462186 and evaluation['total_nats'] == 58854764.0, 'complete raw public20M totals')
bpb = evaluation['total_nats'] / (evaluation['num_bytes'] * math.log(2))
require(math.isfinite(bpb) and bpb == evaluation['val_bpb'] == 1.017340669168314, 'independent BPB arithmetic')
require(not evaluation['eval_over_timeout'] and evaluation['eval_seconds'] < evaluation['eval_timeout'] == 1200, 'evaluation under timeout')
ecommand = evaluation_launch['command']
for flag, value in [('--seq-len', '1024'), ('--eval-tokens', '20971520'), ('--batch-size', '1'), ('--device', 'neuron'), ('--timeout-seconds', '1200')]:
    require(ecommand[ecommand.index(flag) + 1] == value, 'eval argv:' + flag)
require('eval-public' in ecommand and ecommand[ecommand.index('--train-py') + 1] == job['source']['path'], 'fresh evaluator exact source')
retained = retrieval['checkpoint_retained_not_downloaded']
require(ecommand[ecommand.index('--checkpoint') + 1] == retained['path'] == job['working_directory'] + '/out/final.pt', 'fresh evaluator trained checkpoint')
require(all(checkpoint[k] == retained[k] for k in ('bytes', 'sha256')) and checkpoint['bytes'] == 453048594 and checkpoint['sha256'] == 'fb41241f812b0a357a36c18e0cbfbf81adc1df26ababf86d90fa40f51ded7c6b', 'retained checkpoint freshly rehashed')
eval_start = boundary(eval_text, 'FRESH-PUBLIC20M_START_UTC')
eval_end = boundary(eval_text, 'FRESH-PUBLIC20M_END_UTC')
require(start < end < eval_start < eval_end < dt.datetime.fromisoformat(terminal['utc']), 'training then separate fresh evaluation')

selector_path = ROOT / 'workflow/tools/decide.py'
ledger_path = ROOT / 'research/v2/ledger.jsonl'
require(sha(selector_path) == SELECTOR and sha(ledger_path) == LEDGER, 'frozen unchanged selector and full ledger')
spec = importlib.util.spec_from_file_location('decide_unchanged', selector_path)
decide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decide)
rows = decide.load(str(ledger_path))
require(len(rows) == 11 and all(r['label'] != LABEL for r in rows), 'all current rows without candidate')
controls = decide.controls_for(rows, 'trn22', BASE)
require(len(controls) == 1 and controls[0]['label'] == job['same_host_control']['run_label'], 'correct same-host c5c control')
control = controls[0]
require(control['public_20m'] == .991513958330512 and control['steps'] == 3633, 'control value and steps')
require(sha(ROOT / control['eval_log']) == control['eval_log_sha256'], 'control raw evaluation hash')
control_text = (ROOT / control['eval_log']).read_text()
control_eval, _ = json.JSONDecoder().raw_decode(control_text[control_text.index('{\n'):])
require(control_eval['total_nats'] / (control_eval['num_bytes'] * math.log(2)) == control['public_20m'], 'control arithmetic')
candidate = {
    'label': LABEL, 'arm_id': job['arm_id'], 'kind': job['kind'], 'host': 'trn22',
    'timestamp_utc': terminal['utc'], 'base_sha256': BASE, 'train_py_sha256': SOURCE,
    'status': 'evaluated', 'git_dirty': False,
    'git_dirty_scope': 'Exact committed science bytes; no globally clean host claim',
    'argv': job['argv'], 'env': launch['environment'], 'training_budget_seconds': 1800,
    'over_budget': False, 'steps': int(metrics['num_steps']),
    'training_seconds': metrics['training_seconds'], 'charged_seconds_est': metrics['charged_seconds_est'],
    'charged_through_exit_seconds': charged_upper,
    'charged_through_exit_evidence_scope': 'Conservative same-driver UTC enclosure minus proven saturated 900-second cap; not independent monotonic measurement',
    'process_through_exit_wall_seconds': wall, 'startup_measured_seconds': None,
    'startup_measured_lower_bound_seconds': startup_lower, 'startup_allowance_est': 900,
    'cold_startup_proven': False, 'causality_passed': None,
    'eval_tokens': evaluation['num_tokens'], 'eval_seq_len': 1024, 'public_20m': bpb,
    'checkpoint_sha256': checkpoint['sha256'],
    'timing_diagnostics': {'step_time_spikes': [{'step': int(s), 'dt_seconds': float(v)} for s, v in first[:2]]}}
require(decide.invalid_reason(candidate) is None, 'selector valid reviewed research scope')
verdict = decide.decide_arm(rows + [candidate], job['arm_id'])
delta = bpb - control['public_20m']
require(verdict['verdict'] == 'DISCARD' and verdict['per_host'][0]['delta'] == round(delta, 7), 'unchanged selector discard')
require(verdict['per_host'][0]['controls'] == [control['label']], 'selector correct control')
require(sha(selector_path) == SELECTOR and sha(ledger_path) == LEDGER, 'ledger and selector untouched')
print(json.dumps({'audit_status': 'PASS_RESULT_AUTHENTICATION_QUALITY_DISCARD', 'check_count': len(checks), 'checks': checks, 'source_sha256': SOURCE, 'selector_sha256': SELECTOR, 'ledger_sha256': LEDGER, 'historical_rows': len(rows), 'candidate_query_only': candidate, 'control_bpb': control['public_20m'], 'delta_unrounded': delta, 'selector_result': verdict, 'evidence_sha256': {name: meta['sha256'] for name, meta in retrieval['files'].items()}}, indent=2))
