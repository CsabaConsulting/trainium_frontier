"""Independent Supervisor authentication of the Operator's seed-only run receipts."""
import hashlib
import json
import math
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / 'research/v2/operator/20260926-trn21-recursive-seed43-guarded'
SOURCE = '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'
PREPARE = '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb'
CHECKPOINT = 'ebe67c94d93d4fa7c1bab915bb472be7e6877ecf1610dc01799fe7ae7f0ee5fa'
LABEL = 'fs-20260926-recursive6ef7-seed43-trn21-r1-full'
checks = []


def require(ok, name):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)


def read(path):
    return json.loads(path.read_text())


for phase in ('completed-training', 'completed-eval'):
    directory = ROOT / phase
    manifest = read(directory / 'retrieval-manifest.json')
    if 'metadata' in manifest:
        manifest = manifest['metadata']
    for name, expected in manifest['files'].items():
        raw = (directory / name).read_bytes()
        require(len(raw) == expected['bytes'], f'{phase}/{name} length')
        require(hashlib.sha256(raw).hexdigest() == expected['sha256'], f'{phase}/{name} sha256')

train_dir, eval_dir = ROOT / 'completed-training', ROOT / 'completed-eval'
train, evaluated = read(train_dir / 'train/rehearsal.json'), read(eval_dir / 'eval/rehearsal.json')
train_log, eval_log = (train_dir / 'training.log').read_text(), (eval_dir / 'fresh-public20m.log').read_text()
require(train['returncode'] == evaluated['returncode'] == 0, 'both process exits zero')
require(train['mode'] == 'train' and evaluated['mode'] == 'eval', 'separate process modes')
require(train['step_boundaries'] == 3646, 'all3646 step boundaries')
require([r['step'] for r in train['observations']] == list(range(3646)), 'complete ordered observations')
startup = train['observations'][4]['observed_seconds']
require(startup == train['startup_allowance_observed'], 'uncapped fifth startup')
charge = train['wall_seconds'] - min(startup, 900.0)
require(math.isclose(charge, train['charged_through_process_exit'], abs_tol=1e-9, rel_tol=0), 'independent official charge')
require(0 < charge <= 1800 and re.search(r'^over_budget:\s+False$', train_log, re.M), 'actual and source budget pass')
require(not train['fresh_empty_sdk_caches'], 'not relabelled cold')
require(train['checkpoint_sha256'] == CHECKPOINT and train['checkpoint_bytes'] == 469834913, 'checkpoint receipt identity')
require(read(train_dir / 'checkpoint.json') == read(eval_dir / 'checkpoint.json'), 'same retained pre-eval identity receipt')
require(re.search(r'^num_steps:\s+3646$', train_log, re.M), 'source step summary agrees')
for log in (train_log, eval_log):
    isolated = json.loads(next(line.split(' ', 1)[1] for line in log.splitlines() if line.startswith('ISOLATED_SUBMISSION ')))
    require(isolated['train_sha256'] == SOURCE and isolated['prepare_sha256'] == PREPARE, 'isolated source and prepare identity')
    require('--seed 43' in isolated['launch_command'] and '--recursive-schedule' in isolated['launch_command'], 'seed43 schedule config')
command = json.loads(next(line.split(' ', 1)[1] for line in eval_log.splitlines() if line.startswith('REHEARSAL_COMMAND ')))
for key, value in (('--batch-size','1'),('--seq-len','1024'),('--eval-tokens','20971520'),('--timeout-seconds','1200')):
    require(command[command.index(key)+1] == value, key)
require(command[2:6] == ['eval-public', '--train-py', 'train.py', '--checkpoint'], 'unmodified scorer route')
score = json.loads(re.search(r'^\{\n  "eval_over_timeout".*?^\}', eval_log, re.M | re.S).group())
require(score['num_tokens'] == 20971520 and score['num_bytes'] == 83462186, 'public20M same bytes and tokens')
require(score['total_nats'] == 57400540.0, 'reported nats')
computed = score['total_nats'] / (math.log(2) * score['num_bytes'])
require(math.isclose(computed, score['val_bpb'], abs_tol=1e-15, rel_tol=0), 'recomputed byte-weighted BPB')
require(not score['eval_over_timeout'] and score['eval_seconds'] < 1200, 'eval completed within timeout')
require(read(eval_dir / 'terminal.json')['eval_exit_code'] == 0 and read(eval_dir / 'outer-flock-exit.json')['returncode'] == 0, 'driver and outer lock terminal zero')
guard = read(eval_dir / 'eval-under-lock-idle.json')
require(guard['status'] == 'PASS' and guard['lock']['verified_existing_exclusive_lock'] is True, 'eval inherited original lifetime lock')
require(guard['lock']['inode'] == 596823 and not guard['ps']['busy'] and not guard['neuron_ls']['process_pids'], 'same inode and fresh idle checks')
ledger = [json.loads(line) for line in (REPO / 'research/v2/ledger.jsonl').read_text().splitlines() if line.strip()]
control = next(row for row in ledger if row['label'] == 'fs-20260925-l3-base-repeat-trn21-r2-full')
require(control['train_py_sha256'] == SOURCE and control['host'] == 'trn21', 'matched source and host control')
require(control['public_20m'] == 0.9899528600562257, 'matched seed42 control score')
print(json.dumps({'status':'SEED43_RESULT_AUTHENTICATED_VARIANCE_ONLY','checks':len(checks),
 'label':LABEL,'source_sha256':SOURCE,'seed':43,'steps':3646,'public20m':score['val_bpb'],
 'matched_seed42_public20m':control['public_20m'],'delta':score['val_bpb']-control['public_20m'],
 'startup_seconds':startup,'charged_through_exit_seconds':charge,'over_budget':False,
 'eval_tokens':20971520,'eval_seq_len':1024,'eval_seconds':score['eval_seconds'],
 'checkpoint_sha256':CHECKPOINT,'checkpoint_evidence':'pre-eval receipt reauthenticated; no post-eval byte rehash',
 'cold_full':False,'causality_passed':None,'adoption':False,'submission_eligible':False,
 'limits':'One paired seed comparison is not a variance estimate or confidence interval; no seed selection.'}, indent=2))
