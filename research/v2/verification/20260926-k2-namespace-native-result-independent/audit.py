#!/usr/bin/env python3
"""Read-only, stdlib-only authentication of frozen K2 r3 failure evidence."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--repo', default='/home/poojith-devan/Desktop/Trainium')
args = parser.parse_args()
RAW = '7c40e4f5'
LIVE = '95666f13'
SOURCE = 'd0f0d7fbdbe07ddee39680ee1a2b79fdfac388ca'
BINDING = '0abd35a071508679e4d184ca2aeb3ab1295cba3f'
PREFIX = 'research/v2/operator/20260926-trn22-k2-namespace-native/'
PACKET = 'research/v2/proposals/20260926-k2-name-binding-native/'
MODULE = 'research/v2/lanes/k2-resident-consumer/'
GUARD = 'research/v2/operator/20260926-common-host-idle-guard/'
ERROR = ('[INTERNAL_ERROR] [NCC_IXGM002] Expected function sg0000 in subgraph 0 '
         'to have 9 basic blocks, but on core 1 it has 1 basic blocks')
checks = []


def check(condition, label):
    if not condition:
        raise RuntimeError(label)
    checks.append(label)


def git(rev, path):
    return subprocess.check_output(['git', 'show', f'{rev}:{path}'], cwd=args.repo)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get(rev, path):
    return json.loads(git(rev, path))


def verify_bundle(bundle, count, size, name):
    check(bundle['file_count'] == len(bundle['files']) == count, name + ': count')
    total = 0
    for path, entry in bundle['files'].items():
        data = entry['text'].encode('utf-8')
        check(len(data) == entry['bytes'], name + ': size ' + path)
        check(sha(data) == entry['sha256'], name + ': sha ' + path)
        total += len(data)
    check(total == bundle['raw_bytes'] == size, name + ': total bytes')


def idle(devices):
    return (len(devices) == 1 and devices[0]['neuron_device'] == 0
            and devices[0]['logical_neuroncore_config'] == 2
            and devices[0]['neuroncore_ids'] == [0, 1, 2, 3]
            and devices[0]['neuron_processes'] == [])


raw_path = PREFIX + 'completed/results-bundle.json'
live_path = PREFIX + 'launch/first-four-rank-evidence.json'
post_path = PREFIX + 'completed/postflight-explicit-pids.json'
raw, live, post = get(RAW, raw_path), get(LIVE, live_path), get(RAW, post_path)
verify_bundle(raw, 22, 70941, 'terminal bundle')
verify_bundle(live, 16, 49775, 'live bundle')
files = raw['files']
j = lambda path: json.loads(files[path]['text'])
binding = get(BINDING, PACKET + 'NATIVE_BINDING.disabled.json')
check(binding == j('NATIVE_BINDING.disabled.json'), 'remote binding exactly frozen')
check(binding['source_module_commit'] == SOURCE, 'exact source commit')
check(binding['host'] == 'trn22', 'bound host')
check(binding['root'] == raw['root'] == live['root'], 'all roots identical')
pins = binding['support_sha256']
source_files = {'consumer_nki.py', 'check_native.py', 'reference.py'}
for name, expected in pins.items():
    path = MODULE + name if name in source_files else (
        GUARD + name if name in ('idle_guard.py', 'dispatch_driver.py') else PACKET + name)
    rev = SOURCE if name in source_files else BINDING
    check(sha(git(rev, path)) == expected, 'source/support exact ' + name)
for name in ('EXPERIMENT_PROPOSAL.json', 'NATIVE_BINDING.disabled.json', 'MANIFEST.sha256.json'):
    check(git(BINDING, PACKET + name) == files[name]['text'].encode(), 'frozen bytes ' + name)
check(sha(git(BINDING, PACKET + 'host_driver.py')) == binding['driver_sha256'], 'driver pin')
pre = j('preflight.json')
for name, expected in pins.items():
    check(pre['staged_hashes'][name] == expected, 'staged pin ' + name)
check(pre['staged_hashes']['host_driver.py'] == binding['driver_sha256'], 'staged driver')
for name in ('NATIVE_BINDING.disabled.json', 'EXPERIMENT_PROPOSAL.json', 'MANIFEST.sha256.json', 'SUPERVISOR_ASSIGNMENT.md'):
    check(pre['staged_hashes'][name] == files[name]['sha256'], 'staged artifact ' + name)
launch = j('diagnostic-launch.json')
check(launch['command'] == binding['command'], 'exact native command')
check(launch['environment'] == binding['environment'], 'exact environment')
check(launch['support_sha256'] == pins, 'launch pins')
check(launch['no_eval'] and not launch['quality_evidence'], 'diagnostic only')
check(binding['command'][:5] == ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=60s', '600s'], '600+60 timeout')
check(binding['retries'] == 0 and not any(binding[k] for k in ('automatic_full', 'automatic_eval', 'automatic_next_job')), 'no automatic continuation')
live_pids = {int(row.split()[0]) for row in live['job_lines']}
check(len(live_pids) == 9, 'nine live process identities')
worker_pids = set()
for rank in range(4):
    env = j(f'rank-environment-r{rank}.json')
    check(env['rank'] == env['local_rank'] == rank and env['world_size'] == 4, f'rank {rank}: mapping')
    check(env['environment'] == binding['environment'], f'rank {rank}: environment')
    check(env['checker_argv'] == binding['checker_args'], f'rank {rank}: argv')
    check(env['checker_sha256'] == pins['check_native.py'], f'rank {rank}: checker pin')
    check(env['pid'] in live_pids, f'rank {rank}: live PID')
    worker_pids.add(env['pid'])
live_device = json.loads(live['neuron_ls'])[0]
check({p['pid'] for p in live_device['neuron_processes']} == worker_pids, 'four workers observed on device')
guard = j('under-lock-idle.json')
check(guard['lock']['path'] == '/home/ubuntu/.cache/trainium_freshstart/device0.lock', 'same lock path')
check(guard['lock']['verified_existing_exclusive_lock'], 'inherited exclusive lock authenticated')
check(guard['ps']['returncode'] == 0 and not guard['ps']['busy'] and not guard['ps']['stderr'], 'in-lock ps clean')
check(guard['neuron_ls']['returncode'] == 0 and not guard['neuron_ls']['process_pids'] and not guard['neuron_ls']['stderr'], 'in-lock neuron-ls clean')
check(guard['lock']['inode'] == pre['lock_inode'] == live['lock_inode'] == raw['lock_inode'] == 832589, 'original inode continuous at snapshots')
check(guard['lock']['device'] == pre['lock_device'] == live['lock_device'] == raw['lock_device'] == 66305, 'lock filesystem device')
check(live['lock_competitor_rc'] == 1 and raw['lock_competitor_rc'] == 0, 'held while live, free after exit')
cache = j('fresh-caches.json')
check(set(cache['directories']) == set(binding['cache_keys']) and len(cache['directories']) == 6, 'six private cache directories')
for key, entry in cache['directories'].items():
    check(entry['initially_empty'] and entry['path'] == binding['environment'][key], 'fresh path ' + key)
check(cache['compile_cache_url'] == binding['environment']['NEURON_COMPILE_CACHE_URL'] == binding['environment']['NEURON_COMPILE_CACHE_DIR'], 'compiler URL equals DIR')
terminal, child, outer = j('terminal.json'), j('child-exit.json'), j('outer-flock-exit.json')
check(terminal['status'] == 'DIAGNOSTIC_CHILD_FAILED' and not terminal['complete'], 'terminal failed, incomplete')
check(terminal['returncode'] == child['returncode'] == outer['returncode'] == 1, 'all enclosing exits 1')
check(outer['command'] == binding['outer_command'], 'same outer flock command')
check(terminal['no_automatic_next_job'] and terminal['no_eval'], 'no next/eval')
wall = child['timeout_wrapped_child_wall_seconds']
check(wall == 25.8187616559444 and wall < 600, 'observed child wall, not timeout')
utc_delta = (datetime.fromisoformat(child['utc']) - datetime.fromisoformat(launch['utc'])).total_seconds()
check(abs(utc_delta - wall) < .001, 'UTC interval agrees within 1 ms with monotonic duration')
rank0 = j('receipts/rank0.json')
check(rank0['status'] == 'REFUSE_OR_UNSUPPORTED', 'rank0 refusal')
check(j('rank-wrapper-exit-r0.json')['returncode'] == 1, 'rank0 wrapper failed')
check(rank0['kernel_sha256'] == pins['consumer_nki.py'] and rank0['checker_sha256'] == pins['check_native.py'] and rank0['reference_sha256'] == pins['reference.py'], 'rank0 three source pins')
check(rank0['argv'] == binding['checker_args'], 'rank0 exact argv')
check(rank0['shape'] == {'V': 256, 'C': 128, 'T': 256} and rank0['seed'] == 42, 'tiny shape and seed')
check(rank0['tolerance'] == {'atol': 0, 'rtol': 0, 'criterion': 'identical bytes'}, 'no tolerance waiver')
check(len(rank0['cases']) == 1, 'only first case recorded')
case = rank0['cases'][0]
check(case['name'] == 'cross_tile_large' and case['dtype'] == 'torch.bfloat16' and case['status'] == 'STARTED', 'first BF16 case incomplete')
check([c['name'] for c in case['checks']] == ['upstream_roundtrip'], 'no consumer comparison')
up = case['checks'][0]
check(up['exact'] and up['finite'] and up['max_abs'] == 0 and up['actual_sha256'] == up['expected_sha256'], 'only successful upstream transport check')
check(rank0['timings'] == [] and 'direct_first_call_seconds' not in case, 'no consumer timing')
check(not rank0['source_adoption'] and not rank0['accepted_speedup'] and rank0['whole_step_speedup'] == 'UNMEASURED' and rank0['actual_model_autograd'] == 'NOT_TESTED', 'no source/speed/model claims')
log = files['diagnostic.log']['text']
check(ERROR in rank0['error'] and ERROR in log, 'exact first compiler failure in two raw records')
check('line 134' in rank0['error'] and 'sync()' in rank0['error'], 'asynchronous error detected at synchronization')
checker_lines = git(SOURCE, MODULE + 'check_native.py').decode().splitlines()
check(checker_lines[132].strip() == 'actual = kernel(device_upstream, *device_meta)' and checker_lines[133].strip() == 'sync()', 'pinned source call and detection lines')
check('failed to resolve name' not in rank0['error'], 'different failure from original namespace error')
check(sorted(n for n in files if n.startswith('receipts/')) == ['receipts/rank0.json'], 'three missing terminal receipts preserved')
check('completion-check.json' not in files, 'no fabricated completion validation')
for rank in (1, 2, 3):
    pid = j(f'rank-environment-r{rank}.json')['pid']
    check(f'rank-wrapper-exit-r{rank}.json' not in files, f'rank {rank}: wrapper exit missing')
    check(f'Sending process {pid} closing signal SIGTERM' in log and f'exitcode  : -15 (pid: {pid})' in log, f'rank {rank}: actual SIGTERM')
compiler_path = 'caches/debug/neuronx-cc/493cb6c5fdd9ec22c0a638cabf5f8a22/log-neuron-cc.txt'
compiler = files[compiler_path]['text']
check(files[compiler_path]['bytes'] == 6461 and 'NeuronX Compiler version 2.26.6360.0+6f180f47' in compiler, 'compiler log version')
check('--target trn2 --lnc 2' in compiler and 'Start tensorization' in compiler, 'native compiler invoked')
check('NCC_IXGM002' not in compiler, 'compiler log does not itself contain fatal message')
check(len(raw['bounded_debug_inventory']) == 2 and not any(n.endswith('module.mlir') for n in files), 'MLIR inventory only, no contents claimed')
check(set(post['observed_pids']) == live_pids == {r['pid'] for r in post['pid_rows']}, 'all nine observed PIDs covered')
check(all(not r['exists'] for r in post['pid_rows']) and post['all_observed_pids_absent'], 'all observed PIDs absent')
check(post['ps_rc'] == 0 and post['ps_stderr'] == '' and post['related_rows'] == [], 'PPID/PGID/SID and compiler query empty')
check(idle(post['neuron_ls']) and post['neuron_device_idle'] and not post['neuron_ls_stderr'], 'postflight device idle')
check(post['lock_competitor_rc'] == 0 and post['no_kills_or_changes'], 'postflight lock free, no manual kill')
check(datetime.fromisoformat(post['utc']) > datetime.fromisoformat(outer['ended_utc']), 'postflight after outer exit')
check(raw['job_lines'] == [] and raw['neuron_ls_rc'] == 0 and idle(json.loads(raw['neuron_ls'])), 'later retrieved snapshot also quiescent')
check(all(not p['exists'] for p in raw['old_queue_pids']), 'old queue processes absent')

print(json.dumps({
    'audit_status': 'PASS_RAW_AUTHENTICATION_ONLY',
    'experiment_status': 'PARKED_COMPILER_FAILURE',
    'checks': len(checks),
    'raw_commit': RAW, 'launch_commit': LIVE, 'source_commit': SOURCE, 'binding_commit': BINDING,
    'artifact_sha256': {raw_path: sha(git(RAW, raw_path)), live_path: sha(git(LIVE, live_path)), post_path: sha(git(RAW, post_path))},
    'support_sha256': pins,
    'native_child_wall_seconds': wall,
    'utc_child_interval_seconds': utc_delta,
    'error': ERROR,
    'rank0_completed_checks': ['upstream_roundtrip'],
    'missing_checker_receipt_ranks': [1, 2, 3],
    'native_consumer_exactness': 'NOT_REACHED',
    'consumer_timings': 'NOT_REACHED',
    'whole_step_speedup': 'UNMEASURED', 'bpb': None,
    'further_retry_authorized': False,
    'tests_executed': checks,
}, indent=2, sort_keys=True))
