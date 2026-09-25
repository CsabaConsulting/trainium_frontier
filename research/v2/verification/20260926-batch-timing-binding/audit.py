#!/usr/bin/env python3
"""Offline binding/analyzer audit. No execution of host drivers or candidate model."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[4]
PACK = ROOT / 'research/v2/operator/20260926-trn21-batch262144-timing'
checks = []
def check(value, name):
    if not value:
        raise RuntimeError(name)
    checks.append(name)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    return json.loads(path.read_text())

binding = read(PACK / 'BATCH_BINDING.disabled.json')
proposal = read(PACK / 'EXPERIMENT_PROPOSAL.json')
provenance = read(PACK / 'package/provenance.json')
check(binding['launch_authorized'] is False and binding['stage_authorized'] is False, 'disabled preparation')
check(binding['science_argv'] == binding['training_command'][4:], 'science argv binding')
check(binding['training_command'] == proposal['argv'], 'proposal argv')
check(shlex.split((PACK/'package/launch-command.txt').read_text()) == proposal['argv'], 'package command')
check(shlex.split(provenance['launch_command']) == proposal['argv'], 'provenance command')
source = subprocess.check_output(['git', 'show', binding['source_commit']+':'+binding['source_git_path']], cwd=ROOT)
check(hashlib.sha256(source).hexdigest() == binding['source_sha256'], 'committed source bytes')
check(sha(PACK/'host_driver.py') == binding['driver_sha256'] == proposal['driver_sha256'], 'driver pin')
for filename, field in [('idle_guard.py', 'idle_guard_sha256'), ('dispatch_driver.py', 'dispatch_driver_sha256')]:
    path = ROOT / 'research/v2/operator/20260926-common-host-idle-guard' / filename
    check(sha(path) == binding[field] == proposal[field], filename+' unchanged reviewed bytes')
assignments = {}
driver = ast.parse((PACK/'host_driver.py').read_text())
for node in driver.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        try:
            assignments[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
check(assignments['COMMAND'] == binding['command'], 'literal driver command')
check(assignments['ENV'] == binding['environment'] == proposal['environment'], 'literal driver environment')
check(assignments['SOURCE_SHA'] == binding['source_sha256'], 'literal source pin')
check(assignments['PREP_SHA'] == binding['prepare_sha256'], 'literal evaluator pin')
check(assignments['HARNESS_SHA'] == binding['harness_sha256'], 'literal harness pin')
check(binding['outer_command'][:3] == ['/usr/bin/flock', '-n', binding['device_lock']], 'common outer flock')
check(binding['device_lock'] == '/home/ubuntu/.cache/trainium_freshstart/device0.lock', 'one shared inode path')
check(binding['outer_command'][3:6] == ['/usr/bin/python3', '-I', '-B'], 'isolated unoptimized driver interpreter')
calls = [n for n in ast.walk(driver) if isinstance(n, ast.Call)]
child_calls = [n for n in calls if ast.unparse(n.func) == 'subprocess.call']
check(len(child_calls) == 1, 'one synchronous child launch')
check(any(k.arg == 'pass_fds' and ast.unparse(k.value) == '(lock_fd,)' for k in child_calls[0].keywords), 'inherited lock survives child')
guard_calls = [n for n in calls if ast.unparse(n.func) == 'guard.require_idle_under_inherited_lock']
check(len(guard_calls) == 1 and guard_calls[0].lineno < child_calls[0].lineno, 'in-lock idle check precedes spawn')
check(binding['no_eval'] is True and binding['no_automatic_full'] is True, 'no eval or automatic full')

cpu_root = ROOT / 'research/v2/operator/20260925-trn22-batch262144-g2-r1/remaining-results'
for relative, field in [('candidate-262144/g2.json','candidate_on_sha256'),
                        ('candidate-default/g2.json','candidate_default_sha256'),
                        ('default-lockstep.json','default_identity_sha256')]:
    check(sha(cpu_root/relative) == proposal['full_parameter_cpu'][field], 'raw CPU receipt '+relative)
cpu_on = read(cpu_root/'candidate-262144/g2.json')
cpu_argv = cpu_on['argv'][:]
check(cpu_on['train_py_sha256'] == binding['source_sha256'] and cpu_on['torchrun_rc'] == 0, 'CPU ON source and exit')
check(cpu_on['nproc'] == 4 and cpu_on['steps'] == 3, 'CPU four-rank three-step scope')
# Production arguments are the same except the prospective diagnostic step cap.
expected_argv = binding['science_argv'][:]
expected_argv[expected_argv.index('--num-steps')+1] = '1000000'
check(expected_argv == cpu_argv, 'CPU ON approved argv versus diagnostic step cap')
identity = read(cpu_root/'default-lockstep.json')
check(identity['identical'] is True and identity['n_params_compared'] == 62, 'default identity receipt')

spec = importlib.util.spec_from_file_location('batch_timing_analyzer', PACK/'analyze_timing.py')
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)
control_log_path = ROOT / proposal['control']['log_path']
check(sha(control_log_path) == proposal['control']['log_sha256'], 'matched control log pin')
control_rows = {int(m[1]): (float(m[3]), int(m[5].replace(',', ''))) for m in analyzer.ROW.finditer(control_log_path.read_text())}
window = [control_rows[s] for s in analyzer.WINDOW]
check(all(lrm == 1 for lrm, _ in window), 'same control LR window')
check(statistics.median(speed for _, speed in window) == analyzer.CONTROL_MEDIAN == 268831, 'computed matched control median')
check(analyzer.THRESHOLD == proposal['prospective_screen']['threshold_tok_s'] == 282272.55, 'fixed five percent threshold')
log = 'device=neuron dtype=torch.bfloat16 world_size=4\nmicro_tokens=131,072 grad_accum_steps=2 pack_factor=32\n'
log += '\n'.join(f'step {s:05} | loss 5.0000 | lrm 1.000 | dt 0.80s | tok/s 300,000' for s in analyzer.WINDOW)
receipt = {'returncode':0, 'mode':'train', 'charged_through_process_exit_within_budget':True,
           'checkpoint_bytes':123, 'observations':[{'step':s,'observed_seconds':800+s} for s in range(105)]}
check(analyzer.analyze(log, receipt)['status'] == 'SPEED_SCREEN_PASS_PENDING_REVIEW', 'own pass fixture')
check(analyzer.analyze(log.replace('300,000','270,000'), receipt)['status'] == 'SPEED_SCREEN_BELOW_THRESHOLD', 'own below-threshold fixture')
slow = copy.deepcopy(receipt)
slow['observations'] = [{'step':s,'observed_seconds':800+2*s} for s in range(105)]
slow_result = analyzer.analyze(log, slow)
check(slow_result['status'] == 'SPEED_SCREEN_PASS_PENDING_REVIEW' and len(slow_result['sanity_flags']) == 1, 'different estimator flags but does not veto')
for bad_key, value in [('returncode',1), ('checkpoint_bytes',0), ('charged_through_process_exit_within_budget',False)]:
    bad = copy.deepcopy(receipt)
    bad[bad_key] = value
    try:
        analyzer.analyze(log, bad)
    except ValueError:
        check(True, 'own rejection '+bad_key)
    else:
        raise RuntimeError('failed to reject '+bad_key)
print(json.dumps({'status':'PASS_BINDING_ONLY_SOURCE_REVIEW_PENDING','checks':checks,
                  'count':len(checks),'host_execution':False,'model_execution':False}, indent=2))
