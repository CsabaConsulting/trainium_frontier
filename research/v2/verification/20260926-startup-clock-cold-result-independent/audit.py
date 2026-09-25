"""Own read-only authentication of frozen startup-clock cold-five evidence."""
import csv
import hashlib
import io
import json
import math
import re
import shlex
import subprocess
from datetime import datetime

REPO = '/home/poojith-devan/Desktop/Trainium'
LIVE = '4fa3355b'
RAW = '15261d14'
PACKET = 'f35ac4947cfcb9a06a5833d00afe098ed79f07f9'
PREFIX = 'research/v2/operator/20260926-trn22-startup-clock-cold-five/'
PROPOSAL = 'research/v2/proposals/20260926-startup-adjusted-cold-five-trn22/'
checks = []


def check(label, condition):
    if not condition:
        raise RuntimeError(label)
    checks.append(label)


def blob(rev, path):
    return subprocess.check_output(['git', 'show', rev + ':' + path], cwd=REPO)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_bundle(bundle, count, size, label):
    check(label + ' count', bundle['file_count'] == len(bundle['files']) == count)
    for name, entry in bundle['files'].items():
        data = entry['text'].encode('utf-8')
        check(label + ' file ' + name, len(data) == entry['bytes'] and sha(data) == entry['sha256'])
    check(label + ' bytes', sum(x['bytes'] for x in bundle['files'].values()) == bundle['raw_bytes'] == size)


binding_bytes = blob(PACKET, PROPOSAL + 'CT_BINDING.disabled.json')
b = json.loads(binding_bytes)
live_bytes = blob(LIVE, PREFIX + 'launch/first-four-rank-evidence.json')
live = json.loads(live_bytes)
verify_bundle(live, 11, 43805, 'launch')
lj = lambda name: json.loads(live['files'][name]['text'])
check('exact binding', sha(binding_bytes) == 'ddd5cfea65d4795c3d14e871f934fd07162412e56267253b94cdc27acbe90c31' and lj('CT_BINDING.disabled.json') == b)
for name in ('CT_BINDING.disabled.json', 'MANIFEST.sha256.json', 'EXPERIMENT_PROPOSAL.json'):
    check('frozen packet ' + name, live['files'][name]['text'].encode() == blob(PACKET, PROPOSAL + name))
source = blob(b['source_commit'], b['source_git_path'])
check('frozen source', sha(source) == b['source_sha256'] == '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb')
launch, pre, guard = lj('training-launch.json'), lj('preflight.json'), lj('under-lock-idle.json')
check('actual command and clean environment', launch['command'] == b['command'] and launch['environment'] == b['training_environment'])
check('actual source and support', all(launch[k] == b[k] == pre[k] for k in ('source_sha256', 'prepare_sha256', 'harness_sha256')))
check('source stage bytes', pre['staged_hashes']['package/train.py'] == b['source_sha256'])
check('driver stage bytes', pre['staged_hashes']['host_driver.py'] == b['driver_sha256'] == sha(blob(PACKET, PROPOSAL + 'host_driver.py')))
for name, digest in b['package_files'].items():
    check('package stage ' + name, pre['staged_hashes'][name] == digest == sha(blob(PACKET, PROPOSAL + name)))
for name, key in (('idle_guard.py', 'idle_guard_sha256'), ('dispatch_driver.py', 'dispatch_driver_sha256')):
    check('shared support stage ' + name, pre['staged_hashes'][name] == b[key] == sha(blob(PACKET, 'research/v2/operator/20260926-common-host-idle-guard/' + name)))
check('hostname and root', pre['hostname'] == b['expected_hostname'] == 'ip-172-31-32-253' and live['root'] == b['root'])
check('MAX1800, steps5, one new flag', b['training_command'] == b['base_training_command'] + ['--startup-adjusted-schedule'] and b['training_command'][b['training_command'].index('--num-steps') + 1] == '5' and b['training_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'] == '1800')
cache_paths = {k: v for k, v in b['harness_injected_environment'].items() if 'CACHE_DIR' in k or k == 'TORCH_NEURONX_DEBUG_DIR'}
check('six absent private roots before stage', len(cache_paths) == 6 and len(set(cache_paths.values())) == 6 and pre['cold_cache_dirs_absent_before_stage'] == cache_paths)
check('fresh harness flag', '--fresh-caches' in launch['command'])
workers = live['workers']
check('four unique live ranks', len(workers) == 4 and {w['environment']['RANK'] for w in workers} == {'0', '1', '2', '3'})
for worker in workers:
    env = worker['environment']
    rank = env['RANK']
    check('worker argv ' + rank, shlex.split(worker['argv'])[2:] == b['training_command'][3:])
    check('worker cwd ' + rank, worker['cwd'] == b['sandbox'])
    check('worker mapping ' + rank, env['LOCAL_RANK'] == rank and env['WORLD_SIZE'] == '4' and env['NEURON_LOGICAL_NC_CONFIG'] == '2')
    check('worker six private caches ' + rank, all(env[k] == v for k, v in cache_paths.items()))
    check('worker compiler URL and flags ' + rank, env['NEURON_COMPILE_CACHE_URL'] == env['NEURON_COMPILE_CACHE_DIR'] and env['NEURON_CC_FLAGS'] == b['training_environment']['NEURON_CC_FLAGS'])
live_pids = {int(row.split()[0]) for row in live['job_lines']}
check('all workers in actual process tree', {w['pid'] for w in workers} <= live_pids and len(live_pids) == 10)
check('fresh guard before launch', datetime.fromisoformat(guard['ended_utc']) < datetime.fromisoformat(launch['utc']))
check('guard same original exclusive lock', guard['lock']['verified_existing_exclusive_lock'] and guard['lock']['path'] == b['device_lock'] and guard['lock']['inode'] == pre['lock_inode'] == live['lock_inode'] == 832589 and guard['lock']['device'] == pre['lock_device'] == live['lock_device'] == 66305)
check('guard both checks clean', guard['ps']['returncode'] == guard['neuron_ls']['returncode'] == 0 and guard['ps']['busy'] == guard['neuron_ls']['process_pids'] == [] and guard['ps']['stderr'] == guard['neuron_ls']['stderr'] == '')
check('lock held at live snapshot', live['lock_competitor_rc'] == 1)
check('old queues absent', all(not p['exists'] for p in live['old_queue_pids']))
check('previous K2 ended and descendants absent', pre['previous_k2_outer']['returncode'] == 1 and pre['previous_lineage_rows'] == [] and datetime.fromisoformat(pre['previous_k2_outer']['ended_utc']) < datetime.fromisoformat(launch['utc']))

bundle_bytes = blob(RAW, PREFIX + 'completed/results-bundle.json')
bundle = json.loads(bundle_bytes)
verify_bundle(bundle, 20, 59544, 'terminal')
files = bundle['files']
j = lambda name: json.loads(files[name]['text'])
for name in live['files']:
    if name != 'training.log':
        check('closed immutable launch artifact ' + name, files[name] == live['files'][name])
check('training log retains launch prefix', files['training.log']['text'].startswith(live['files']['training.log']['text']))
check('terminal root', bundle['root'] == b['root'])
log = files['training.log']['text']
lines = log.splitlines()
iso = json.loads(next(x.removeprefix('ISOLATED_SUBMISSION ') for x in lines if x.startswith('ISOLATED_SUBMISSION ')))
check('actual sandbox source/prepare', iso['train_sha256'] == b['source_sha256'] and iso['prepare_sha256'] == b['prepare_sha256'] and iso['sandbox'] == b['sandbox'])
check('actual package launch parsed', shlex.split(iso['launch_command']) == b['training_command'])
actual_argv = json.loads(next(x.removeprefix('REHEARSAL_COMMAND ') for x in lines if x.startswith('REHEARSAL_COMMAND ')))
check('actual torchrun argv', actual_argv == b['training_command'])
check('official prepare check completed', '\ncheck passed\n' in log)
step_rows = re.findall(r'^step (\d+) \| loss (\S+) \| lrm (\S+) \| dt (\S+)s \|', log, re.M)
check('five finite printed steps', [int(x[0]) for x in step_rows] == list(range(5)) and all(math.isfinite(float(v)) for row in step_rows for v in row[1:]))
losses = [float(x[1]) for x in step_rows]
check('reported first-five losses', losses == [9.0101, 8.7779, 8.4047, 8.0711, 7.7681])
check('printed first-five LR multipliers', all(float(x[2]) == 1.0 for x in step_rows))
check('official step log', [json.loads(x) for x in files['train/steps.jsonl']['text'].splitlines()] == list(range(5)))
r = j('train/rehearsal.json')
observations = r['observations']
check('five ordered observed boundaries', r['step_boundaries'] == 5 and [x['step'] for x in observations] == list(range(5)) and all(math.isfinite(x['observed_seconds']) for x in observations) and all(a['observed_seconds'] < c['observed_seconds'] for a, c in zip(observations, observations[1:])))
fifth = observations[4]['observed_seconds']
wall = r['wall_seconds']
charge = wall - min(fifth, 900.0)
check('actual uncapped fifth', fifth == 845.6273377500474)
check('cold hard ceiling and target', fifth <= b['cold_decision']['maximum_seconds'] == 900 and fifth <= b['target_startup_seconds'] == 870)
check('actual process wall', wall == 850.0966789470986 and wall >= fifth)
check('capped allowance, not capped cold criterion', r['startup_allowance_observed'] == min(fifth, 900.0))
check('actual charged through process exit', r['charged_through_process_exit'] == charge == 4.469341197051108 and charge <= 1800 and r['charged_through_process_exit_within_budget'] is True)
check('all six caches consistently fresh and assigned', r['fresh_empty_sdk_caches'] == iso['fresh_empty_sdk_caches'] == cache_paths)
summary = json.loads(next(x.removeprefix('SUBMISSION_REHEARSAL ') for x in lines if x.startswith('SUBMISSION_REHEARSAL ')))
check('raw log summary agrees with full observer receipt', summary == {k: v for k, v in r.items() if k != 'observations'})
check('retained checkpoint identity', r['checkpoint_sha256'] == '053211cd99b1d6b0494a1027d157dfc85784b183d5db8f5442fd6fde8f2531bb' and r['checkpoint_bytes'] == 453048594 and bundle['no_checkpoint_copy'] is True)
terminal, outer = j('terminal.json'), j('outer-flock-exit.json')
check('all aggregate exits zero', r['returncode'] == terminal['returncode'] == outer['returncode'] == int(files['run-exit-code.txt']['text']) == 0)
check('five-step train only terminal', r['mode'] == 'train' and terminal['status'] == 'CT_HARNESS_EXITED' and terminal['label'] == b['label'] and terminal['no_eval'] and terminal['no_automatic_full'])
check('same outer command', outer['command'] == b['outer_command'])
owners = [json.loads(x.removeprefix('SUBMISSION_OPTIMIZER_INTEGRATION ')) for x in lines if x.startswith('SUBMISSION_OPTIMIZER_INTEGRATION ')]
check('four training rank integrations', len(owners) == 4 and {x['rank'] for x in owners} == set(range(4)))
check('real full shape and actual backend', 'num_params=113,256,724' in log and 'device=neuron dtype=torch.bfloat16 world_size=4' in log and 'micro_tokens=131,072 grad_accum_steps=1 pack_factor=32' in log)
check('source-estimate budget reported separately', 'over_budget:       False' in log and 'startup_allowance_est: 843.4' in log and 'charged_seconds_est:   0.4' in log)
graph_rows = {}
for rank in range(4):
    rows = list(csv.DictReader(io.StringIO(files[f'metrics-training/metric_table_graph_stats_rank{rank}.csv']['text'])))
    check('nonempty graph evidence rank ' + str(rank), bool(rows) and any('_forward_' in row['graph_name'] for row in rows) and any('_backward_' in row['graph_name'] for row in rows))
    graph_rows[str(rank)] = len(rows)
post_bytes = blob(RAW, PREFIX + 'completed/explicit-process-postflight.json')
post = json.loads(post_bytes)
post_pids = set(post['observed_pids'])
check('sixteen explicit postflight PIDs, no duplicates', len(post_pids) == len(post['observed_pids']) == len(post['pid_rows']) == 16 and {x['pid'] for x in post['pid_rows']} == post_pids)
check('every independently observed launch PID included', live_pids <= post_pids)
check('all declared observed processes absent', all(x['exists'] is False for x in post['pid_rows']) and post['all_observed_pids_absent'] is True)
check('no PPID PGID SID or compiler descendants', post['related_rows'] == [] and post['ps_rc'] == 0 and post['ps_stderr'] == '')
check('postflight device idle', post['neuron_device_idle'] is True and post['neuron_ls_stderr'] == '' and len(post['neuron_ls']) == 1 and post['neuron_ls'][0]['neuron_processes'] == [])
check('postflight lock free without manual kill', post['lock_competitor_rc'] == 0 and post['no_kills_or_changes'] is True)
check('retrieval also quiescent', bundle['job_lines'] == bundle['workers'] == [] and bundle['neuron_ls_rc'] == bundle['lock_competitor_rc'] == 0 and bundle['neuron_ls_stderr'] == '' and all(not d['neuron_processes'] for d in json.loads(bundle['neuron_ls'])))
check('original lock identity at retrieval', bundle['lock_inode'] == live['lock_inode'] == 832589 and bundle['lock_device'] == live['lock_device'] == 66305)
check('old queues remain absent', all(not p['exists'] for p in bundle['old_queue_pids']))
check('postflight after completed outer exit', datetime.fromisoformat(post['utc']) > datetime.fromisoformat(outer['ended_utc']) > datetime.fromisoformat(terminal['utc']) > datetime.fromisoformat(launch['utc']))
check('no evaluation artifacts or result', not any('eval' in name for name in files) and 'val_bpb' not in log)

result = {
    'status': 'PASS_COLD_FIVE_STARTUP_DIAGNOSTIC_ONLY',
    'checks': len(checks), 'checks_passed': checks,
    'launch_commit': LIVE, 'binding_commit': PACKET,
    'raw_commit': RAW, 'raw_files': 20, 'raw_bytes': 59544,
    'raw_bundle_sha256': sha(bundle_bytes),
    'rehearsal_sha256': files['train/rehearsal.json']['sha256'],
    'postflight_sha256': sha(post_bytes),
    'launch_bundle_sha256': sha(live_bytes), 'source_sha256': b['source_sha256'],
    'uncapped_fifth_seconds': fifth, 'torchrun_process_wall_seconds': wall,
    'startup_allowance_seconds': min(fifth, 900.0), 'charged_through_exit_seconds': charge,
    'checkpoint_sha256': r['checkpoint_sha256'], 'checkpoint_bytes': r['checkpoint_bytes'],
    'printed_losses': losses, 'printed_lr_multipliers': [float(x[2]) for x in step_rows],
    'graph_row_counts': graph_rows, 'cache_hit_inference': False,
    'explicit_postflight_pids': 16, 'independently_matched_launch_pids': len(live_pids),
    'cold_result_accepted': True, 'launch_authorized': False,
    'steady_throughput_proven': False, 'quality_or_control_claim': False,
    'no_hosts': True, 'no_tensors': True, 'no_source_or_ledger_edits': True,
}
print(json.dumps(result, indent=2))
