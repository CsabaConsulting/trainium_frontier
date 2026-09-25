#!/usr/bin/env python3
"""Read-only authentication of the genuine five-step cold architecture test."""
import hashlib
import json
import math
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT/'research/v2/operator/20260926-ct-architecture-startup-only'
checks = []
def check(value, name):
    if not value:
        raise RuntimeError(name)
    checks.append(name)
def read(relative):
    return json.loads((RAW/relative).read_text())
r = read('completed/train/rehearsal.json')
t = read('completed/terminal.json')
q = read('completed/retrieval-and-quiescence.json')
p = read('launched/four-worker-effective-cold-cache.json')
g = read('launched/under-lock-idle.json')
check(hashlib.sha256((RAW/'completed/train/rehearsal.json').read_bytes()).hexdigest() ==
      'f49d1f7f912e4d0e269a2c9cddcd2bd472485bcdf59c60002c613b1eae752bf7', 'retrieved receipt hash')
check(r['returncode'] == t['returncode'] == 0 and (RAW/'completed/run-exit-code.txt').read_text().strip() == '0', 'successful terminal chain')
check(r['mode'] == 'train' and r['step_boundaries'] == 5, 'startup-only five-step scope')
obs = r['observations']
check([x['step'] for x in obs] == list(range(5)), 'five sequential observed boundaries')
times = [x['observed_seconds'] for x in obs]
check(all(math.isfinite(x) and x > 0 for x in times) and times == sorted(times), 'finite monotonic times')
check(times[-1] == 844.35300997796 and times[-1] <= 900, 'uncapped fifth boundary passes 900 gate')
check(times[-1] <= 870, 'cold architecture target870 passes')
check(r['startup_allowance_observed'] == times[-1], 'allowance equals uncapped boundary because below cap')
check(math.isclose(r['wall_seconds']-times[-1], r['charged_through_process_exit'], abs_tol=1e-9), 'measured charged arithmetic')
check(r['charged_through_process_exit_within_budget'] is True, 'diagnostic budget receipt')
check(r['checkpoint_bytes'] == 453048594 and len(r['checkpoint_sha256']) == 64, 'retained diagnostic checkpoint record')
check(p['sandbox_hashes']['train/train.py'] == '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c', 'effective source3e79')
check(p['sandbox_hashes']['train/prepare.py'] == '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb', 'unchanged evaluator')
workers = [x for x in p['processes'] if 'RANK' in x.get('selected_environment',{})]
check(len(workers) == 4, 'four actual workers')
check({x['selected_environment']['RANK'] for x in workers} == {'0','1','2','3'}, 'four distinct ranks')
check(len(r['fresh_empty_sdk_caches']) == 6, 'six fresh cache paths')
for worker in workers:
    env = worker['selected_environment']
    check(env['WORLD_SIZE'] == '4' and env['NEURON_LOGICAL_NC_CONFIG'] == '2', 'effective full-chip rank'+env['RANK'])
    check(all(env[k] == v for k,v in r['fresh_empty_sdk_caches'].items()), 'effective cold cache paths rank'+env['RANK'])
    check(env['NEURON_COMPILE_CACHE_URL'] == env['NEURON_COMPILE_CACHE_DIR'], 'effective compiler URL rank'+env['RANK'])
    argv = worker['argv']
    check(argv[argv.index('--num-steps')+1] == '5' and '--initial-cache-refresh' in argv, 'actual cold-tested flags rank'+env['RANK'])
    check(argv[argv.index('--max-train-seconds')+1] == '1800' and argv[argv.index('--seed')+1] == '42', 'actual accounting and seed rank'+env['RANK'])
check(g['status'] == 'PASS' and g['lock']['verified_existing_exclusive_lock'] is True, 'shared lock authenticated')
check(g['ps']['returncode'] == g['neuron_ls']['returncode'] == 0 and not g['ps']['busy'] and not g['neuron_ls']['process_pids'], 'in-lock ps plus Neuron idle')
check(q['ps_rc'] == q['neuron_ls_rc'] == q['lock_competitor_rc'] == 0 and not q['job_lines'], 'post-terminal process/lock quiescence')
check(all(not x['neuron_processes'] for x in json.loads(q['neuron_ls'])), 'post-terminal device quiescence')
log = (RAW/'completed/training.log').read_text()
check(hashlib.sha256(log.encode()).hexdigest() == q['logs']['training.log']['sha256'], 'raw training log hash')
rows = re.findall(r'^step (\d+) \| loss (\S+) \| lrm (\S+) \| dt (\S+)s \| tok/s ([\d,]+)',log,re.M)
check(len(rows) == 5 and all(math.isfinite(float(row[1])) for row in rows), 'five finite logged losses')
print(json.dumps({'status':'COLD_STARTUP_GATE_PASS','checks':checks,'count':len(checks),
      'observed_uncapped_startup_seconds':times[-1], 'source_startup_estimate_seconds':842.2,
      'charged_through_process_exit_seconds':r['charged_through_process_exit'],
      'last_three_printed_tokens_per_second':[int(x[-1].replace(',','')) for x in rows[-3:]],
      'throughput_scope':'Three observations only, not a robust speed benchmark',
      'quality_evidence':False,'full_launch_automatically_authorized':False},indent=2))
