#!/usr/bin/env python3
"""Independent read-only authentication of the four actual full-model CPU runs."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

REPO = Path('/home/poojith-devan/Desktop/Trainium')
RAW = '9995964bca3d40ce0887e749d49a6f41a4444e9d'
ROOT = 'research/v2/operator/20260926-trn21-architecture-recursive-cpu'
PACKET = 'research/v2/proposals/20260926-architecture-recursive-exact-cpu-trn21'
checks = []

def check(ok, name):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def git(path, ref=RAW):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=REPO)

raw = git(ROOT + '/results-bundle.json')
bundle = json.loads(raw)
check(raw == (REPO / ROOT / 'results-bundle.json').read_bytes(), 'canonical raw bundle equals frozen operator commit')
files = bundle['files']
check(bundle['file_count'] == len(files) == 91 and bundle['no_checkpoint_copy'], '91 exact raw small files, no checkpoint copy')
for name, record in files.items():
    value = record['text'].encode()
    check(sha(value) == record['sha256'] and len(value) == record['bytes'], 'own raw hash ' + name)

def read(name):
    return json.loads(files[name]['text'])

plan = read('CPU_PLAN.disabled.json')
check(files['CPU_PLAN.disabled.json']['text'].encode() == (REPO / PACKET / 'CPU_PLAN.disabled.json').read_bytes(), 'exact reviewed plan bytes')
check(bundle['root'] == plan['stage_root'] and plan['proposed_host'] == 'trn21', 'exact host and stage root')
for key, pin in plan['sources'].items():
    if 'commit' in pin:
        check(sha(git(pin['git_path'], pin['commit'])) == pin['sha256'], 'immutable source ' + key)
check(plan['full_model']['parameters'] == 113256724 and plan['full_model']['parameter_tensors'] == 65, 'full model not reduced width')

def idle(name):
    item = read(name)
    check(item['status'] == 'PASS' and item['lock']['verified_existing_exclusive_lock'] is True
          and [item['lock']['device'], item['lock']['inode']] == [66305, 596823]
          and item['ps']['returncode'] == item['neuron_ls']['returncode'] == 0
          and not item['ps']['busy'] and not item['neuron_ls']['process_pids'], 'fresh original-lock guard ' + name)

idle('under-lock-idle.json')
arms = {}
prior_end = read('outer-flock-exit.json')['started_utc']
for probe in plan['probes']:
    name = probe['name']
    launch, terminal = read(name + '/launch.json'), read(name + '/terminal.json')
    preflight, report = read(name + '/preflight.json'), read(name + '/g2.json')
    source = plan['sources'][name]['sha256']
    check(launch['command'] == probe['command'] and launch['label'] == probe['label']
          and launch['source_sha256'] == source and launch['cpu_only'], name + ' exact argv/source/label')
    check(files[name + '/request.json']['sha256'] == probe['request_sha256'], name + ' exact request')
    check(preflight['available_ram_bytes'] >= 64 * 2**30 and preflight['free_disk_bytes'] >= 6.5 * 2**30
          and not preflight['neuron_holders'] and not preflight['other_workloads']
          and preflight['source_sha256'] == source, name + ' fresh machine/resources/source')
    idle(name + '/under-lock-idle.json')
    check(prior_end < launch['utc'] < terminal['utc'] and terminal['exit_code'] == 0
          and terminal['verdict'] == 'PASS' and terminal['source_sha256'] == source, name + ' serial chronology/zero exit')
    prior_end = terminal['utc']
    check(report['verdict'] == 'PASS' and report['torchrun_rc'] == 0 and not report['timed_out']
          and report['train_py_sha256'] == source and report['nproc'] == 4 and report['steps'] == 3
          and report['micro_batch'] == [2, 256] and report['prepare']['stub'] is False
          and report['prepare']['sha256'] == plan['sources']['prepare']['sha256'], name + ' actual full-model G2 with disclosed microbatch')
    check(report['plan_sha256'] == plan['expected_plan_sha256'] and len(report['ranks']) == 4, name + ' exact ownership plan')
    check(len(report['shims']) == 5 and any('fake clock' in x for x in report['shims'])
          and any('bfloat16' in x for x in report['shims']), name + ' all CPU shims disclosed')
    ranks = {}
    for r in range(4):
        paths = [p for p in files if p.startswith(name + '/tmp/') and p.endswith('/rank%d.json' % r)]
        check(len(paths) == 1, name + ' unique raw rank ' + str(r))
        rank = read(paths[0])
        check(rank == report['ranks'][r] and rank['rank'] == r, name + ' independent raw/report rank identity ' + str(r))
        check(rank['status'] == 'PASS' and rank['n_params'] == rank['n_changed'] == 65
              and not rank['unchanged_params'] and not rank['cross_rank_mismatches']
              and not rank['tables'] and rank['plan_sha256'] == plan['expected_plan_sha256'], name + ' all parameters updated/synchronized ' + str(r))
        check(len(rank['losses']) == 3 and all(math.isfinite(v) for v in rank['losses']), name + ' three finite raw losses ' + str(r))
        check(sum(rank['plan_summary']['elements_by_owner']) == 113256724, name + ' actual full parameter count ' + str(r))
        ranks[r] = rank
    check(ranks[0]['checkpoint_check']['ok'] and ranks[0]['checkpoint_check']['deterministic']
          and ranks[0]['checkpoint_check']['logits_shape'] == [1, 256, 8192]
          and len(ranks[0]['final_digests']) == 65, name + ' strict saved-model reload evidence')
    logs = [p for p in files if p.startswith(name + '/tmp/') and p.endswith('/torchrun.log')]
    check(len(logs) == 1 and files[logs[0]]['bytes'] > 0, name + ' complete raw torchrun log retained')
    arms[name] = ranks

for name in ('default', 'off'):
    for rank in range(4):
        check(arms[name][rank]['losses'] == arms['reference'][rank]['losses']
              and arms[name][rank]['plan_summary'] == arms['reference'][rank]['plan_summary'], name + ' exact losses/ownership ' + str(rank))
    check(arms[name][0]['final_digests'] == arms['reference'][0]['final_digests'], name + ' exact65 updated parameter digests')
check(arms['on'][0]['final_digests'] != arms['reference'][0]['final_digests'], 'ON is scientifically different as declared')
for phase in ('standard-compare-default', 'standard-compare-off', 'strict-receipts', 'checkpoint-compare'):
    idle(phase + '-under-lock-idle.json')
    check(read(phase + '-exit.json')['exit_code'] == 0, phase + ' successful separate process')
strict = read('strict-receipts.json')
compare = read('checkpoint-compare.json')
check(strict['all_parameter_digests'] == 65 and strict['all_rank_losses'] == 12 and strict['native_correctness'] is False, 'strict raw equality scope')
check(compare['states'] == 65 and compare['parameters'] == 113256724
      and compare['logits_shape'] == [1, 256, 8192] and compare['native_correctness'] is False
      and compare['quality_evidence'] is False, 'actual checkpoint/logit comparison scope')
for row in compare['records']:
    name = row['name']
    check(row['checkpoint_sha256'] == read(name + '/checkpoint-identity.json')['sha256']
          and row['source_sha256'] == plan['sources'][name]['sha256'], 'retained checkpoint identity ' + name)
terminal, outer = read('terminal.json'), read('outer-flock-exit.json')
check(terminal['exit_code'] == outer['returncode'] == 0 and terminal['native_launch_authorized'] is False, 'successful terminal, no inherited native authority')
check(outer['command'][:3] == ['/usr/bin/flock', '-n', plan['device_lock']] and outer['ended_utc'] < bundle['utc'], 'same outer lock and final observation')
check(bundle['lock_competitor_rc'] == bundle['neuron_ls_rc'] == 0 and not bundle['job_lines']
      and not bundle['old_queue_pid_exists'] and [bundle['lock_device'], bundle['lock_inode']] == [66305,596823]
      and all(not d['neuron_processes'] for d in json.loads(bundle['neuron_ls'])), 'actual post-sequence quiescence/original lock free')
print(json.dumps(dict(status='PASS_FULL_PARAMETER_CPU_RESULT_ONLY', checks=len(checks), check_names=checks,
    raw_commit=RAW, bundle_sha256=sha(raw), source_reference=plan['sources']['reference']['sha256'],
    source_candidate=plan['sources']['on']['sha256'], parameters=113256724, tensors=65,
    arms=list(arms), ranks_each=4, updates_each=3, micro_batch=[2,256],
    started_utc=outer['started_utc'], ended_utc=outer['ended_utc'], native_correctness=False,
    quality_evidence=False, no_automatic_native_release=True), indent=2))
