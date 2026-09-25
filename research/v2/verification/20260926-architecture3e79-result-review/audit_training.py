#!/usr/bin/env python3
"""Authenticate architecture training/guard handoff only; no tensors or hosts."""
import datetime
import hashlib
import json
import math
import pathlib
import re
import shlex
import subprocess

REPO = pathlib.Path('/home/poojith-devan/Desktop/Trainium')
ROOT = 'research/v2/operator/20260926-trn22-arch3e79-full'
REF = '8d9ee4b03f0042fc633a96b90637887e0d864490'
BASE = 'c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586'
SELECTOR_SHA = 'fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac'
checks = []


def check(name, condition):
    if not condition:
        raise RuntimeError(name)
    checks.append(name)


def git(path, ref=REF):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=REPO)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record(name):
    return json.loads(git(ROOT + '/completed-training/' + name))


def prefixed(log, prefix):
    values = [json.loads(line[len(prefix):]) for line in log.splitlines() if line.startswith(prefix)]
    check('one ' + prefix, len(values) == 1)
    return values[0]


def read_eval(log):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', log):
        try: value, _ = decoder.raw_decode(log[match.start():])
        except ValueError: continue
        if isinstance(value, dict) and 'total_nats' in value and 'val_bpb' in value:
            return value
    raise RuntimeError('evaluation metrics not found')


b = json.loads(git(ROOT + '/NATIVE_BINDING.disabled.json', '83db5fe3'))
retrieval = record('retrieval.json')
file_hashes = {}
for name, spec in retrieval['metadata']['files'].items():
    raw = git(ROOT + '/completed-training/' + name)
    check('retrieved ' + name, len(raw) == spec['bytes'] and sha(raw) == spec['sha256'])
    file_hashes[name] = spec['sha256']
    check('canonical import ' + name, raw == git(ROOT + '/completed-training/' + name, '55da2532'))
check('eight artifacts', len(file_hashes) == 8)
check('source pin', sha(git(b['source_git_path'], b['source_commit'])) == b['source_sha256'])
check('prepare pin', sha((REPO / 'prepare.py').read_bytes()) == b['prepare_sha256'])
check('harness pin', sha(git('research/scratch/submission_rehearsal.py', '00f2ca7f')) == b['harness_sha256'])
check('driver pin', sha(git(ROOT + '/host_driver.py', '83db5fe3')) == b['driver_sha256'])
for name, key in (('idle_guard.py', 'idle_guard_sha256'), ('dispatch_driver.py', 'dispatch_driver_sha256')):
    check('support ' + name, sha((REPO / 'research/v2/operator/20260926-common-host-idle-guard' / name).read_bytes()) == b[key])
train_launch = record('training-launch.json')
eval_launch = record('fresh-public20m-launch.json')
for launch, command, environment in ((train_launch, 'command', 'training_environment'), (eval_launch, 'evaluation_command', 'evaluation_environment')):
    check('exact launch ' + command, launch['command'] == b[command] and launch['environment'] == b[environment])
    check('identities ' + command, all(launch[k] == b[k] for k in ('label', 'source_sha256', 'prepare_sha256', 'harness_sha256')))
log = git(ROOT + '/completed-training/training.log').decode()
isolated = prefixed(log, 'ISOLATED_SUBMISSION ')
command = prefixed(log, 'REHEARSAL_COMMAND ')
summary = prefixed(log, 'SUBMISSION_REHEARSAL ')
r = record('train/rehearsal.json')
check('full rehearsal summary agrees', summary == {k: v for k, v in r.items() if k != 'observations'})
check('isolated source and prepare', isolated['train_sha256'] == b['source_sha256'] and isolated['prepare_sha256'] == b['prepare_sha256'])
check('exact effective scientific argv', command == b['training_command'] == shlex.split(isolated['launch_command']))
check('old schedule not adopted Recursive', '--recursive-schedule' not in command)
check('max1800 seed42 four ranks', command[command.index('--max-train-seconds') + 1] == '1800' and command[command.index('--seed') + 1] == '42' and '--nproc_per_node=4' in command)
check('warm reuse explicitly not cold', r['fresh_empty_sdk_caches'] == isolated['fresh_empty_sdk_caches'] == {} and '--fresh-caches' not in b['command'])
check('same cache source expected', b['training_environment']['NEURON_COMPILE_CACHE_URL'] == b['training_environment']['NEURON_COMPILE_CACHE_DIR'] and 'ct-arch3e79-cold' in b['training_environment']['NEURON_COMPILE_CACHE_DIR'])
steps = [int(x) for x in git(ROOT + '/completed-training/train/steps.jsonl').splitlines()]
observed = r['observations']
check('3629 exact consecutive boundaries', steps == list(range(3629)) == [x['step'] for x in observed] and r['step_boundaries'] == 3629)
check('monotonic observed boundaries', all(0 <= a['observed_seconds'] <= z['observed_seconds'] for a, z in zip(observed, observed[1:])))
fifth = observed[4]['observed_seconds']
allowance = min(fifth, 900.0)
charged = r['wall_seconds'] - allowance
check('uncapped fifth distinct and allowance correct', fifth == 814.6809981259285 and r['startup_allowance_observed'] == allowance)
check('strict through-exit budget', charged == r['charged_through_process_exit'] and charged <= 1800 and r['charged_through_process_exit_within_budget'] is True)
check('exit success', r['returncode'] == 0 and git(ROOT + '/completed-training/run-exit-code.txt').strip() == b'0')
checkpoint = record('checkpoint.json')
check('checkpoint identity', checkpoint['sha256'] == r['checkpoint_sha256'] == '068749c2559f7c68f6acc76d558caff9d3806d85273fa8562483d87bbad50d25' and checkpoint['bytes'] == r['checkpoint_bytes'] == 453048594)
source_fields = {}
for field, cast in (('training_seconds', float), ('startup_allowance_est', float), ('charged_seconds_est', float), ('num_steps', int), ('world_size', int), ('lnc', int), ('seq_len', int)):
    match = re.search(r'^' + field + r':\s*(\S+)', log, re.M)
    check('source field ' + field, match is not None)
    source_fields[field] = cast(match.group(1))
check('source config/steps', all(source_fields[k] == v for k, v in {'num_steps': 3629, 'world_size': 4, 'lnc': 2, 'seq_len': 1024}.items()))
check('source budget separately valid', source_fields['charged_seconds_est'] <= 1800 and re.search(r'^over_budget:\s*False$', log, re.M))
loss_rows = re.findall(r'^step\s+(\d+)\s+\| loss\s+(\S+)\s+\|.*?dt\s+(\S+)s', log, re.M)
check('all printed losses/timings finite', len(loss_rows) > 360 and all(math.isfinite(float(loss)) and math.isfinite(float(dt)) for _, loss, dt in loss_rows))
start = re.search(r'^TRAINING_START_UTC=(.*)$', log, re.M).group(1)
end = re.search(r'^TRAINING_END_UTC=(.*)$', log, re.M).group(1)
parse = datetime.datetime.fromisoformat
check('UTC enclosure contains harness wall', (parse(end) - parse(start)).total_seconds() >= r['wall_seconds'])
train_guard = json.loads((REPO / ROOT / 'launch/under-lock-idle.json').read_text())
eval_guard = record('eval-under-lock-idle.json')
check('same lock inode/fd/owner', train_guard['lock'] == eval_guard['lock'] and train_guard['lock']['path'] == b['device_lock'])
for name, guard in (('training', train_guard), ('eval', eval_guard)):
    check('fresh idle checks ' + name, guard['lock']['verified_existing_exclusive_lock'] and guard['ps']['returncode'] == guard['neuron_ls']['returncode'] == 0 and not guard['ps']['busy'] and not guard['neuron_ls']['process_pids'])
check('handoff chronology', parse(end) < parse(checkpoint['utc']) < parse(eval_guard['started_utc']) <= parse(eval_guard['ended_utc']) < parse(eval_launch['utc']))
check('bound fresh20M requested', eval_launch['command'][-2:] == ['--eval-tokens', '20971520'] and eval_launch['command'][eval_launch['command'].index('--checkpoint') + 1] == b['sandbox'] + '/out/final.pt' and b['evaluation']['seq_len'] == 1024)
live = json.loads((REPO / ROOT / 'launch/live-processes.json').read_text())
workers = [x for x in live['processes'] if x.get('rank') is not None]
check('four live ranks bound', sorted(int(x['rank']) for x in workers) == [0, 1, 2, 3] and all(int(x['world_size']) == 4 for x in workers))

ledger_bytes = (REPO / 'research/v2/ledger.jsonl').read_bytes()
ledger = [json.loads(x) for x in ledger_bytes.splitlines() if x.strip()]
control = next(x for x in ledger if x['label'] == 'fs-20260925-v2-c5c-a0-control-trn22-r10')
check('matched control same host old c5c base', control['host'] == 'trn22' and control['base_sha256'] == control['train_py_sha256'] == BASE and control['public_20m'] == b['same_host_control']['public20m_bpb'])
control_log_bytes = (REPO / control['eval_log']).read_bytes()
check('control raw eval hash', sha(control_log_bytes) == control['eval_log_sha256'])
c = read_eval(control_log_bytes.decode())
check('control fresh20M arithmetic', c['num_tokens'] == 20971520 and not c['eval_over_timeout'] and c['val_bpb'] == control['public_20m'] == c['total_nats'] / (c['num_bytes'] * math.log(2)))
check('selector unchanged', sha((REPO / 'workflow/tools/decide.py').read_bytes()) == SELECTOR_SHA)
check('no ledger mutation', (REPO / 'research/v2/ledger.jsonl').read_bytes() == ledger_bytes)
print(json.dumps({'status': 'PASS_TRAINING_AUTHENTICATION_ONLY', 'quality_verdict': 'PENDING_FRESH20M',
                  'checks': len(checks), 'check_names': checks, 'raw_commit': REF, 'label': b['label'],
                  'source_sha256': b['source_sha256'], 'steps': len(steps), 'uncapped_fifth_seconds': fifth,
                  'process_wall_seconds': r['wall_seconds'], 'startup_allowance_seconds': allowance,
                  'charged_through_exit_seconds': charged, 'headroom_seconds': 1800 - charged,
                  'post_last_boundary_through_exit_seconds': r['wall_seconds'] - observed[-1]['observed_seconds'],
                  'source_estimates_separate': source_fields, 'checkpoint': checkpoint,
                  'cache_state': 'warm reused exact-source cold-test roots; not a cold full replay',
                  'training_start_utc': start, 'training_end_utc': end, 'fresh20M_launch_utc': eval_launch['utc'],
                  'lock_device_inode': [eval_guard['lock']['device'], eval_guard['lock']['inode']],
                  'control_label': control['label'], 'control_bpb': control['public_20m'],
                  'control_steps': control['steps'], 'ledger_sha256': sha(ledger_bytes),
                  'selector_sha256': SELECTOR_SHA, 'retrieved_artifact_sha256': file_hashes,
                  'checkpoint_downloaded': False, 'native_causality_observed': None}, indent=2))
