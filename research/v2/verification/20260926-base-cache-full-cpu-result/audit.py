"""Independent frozen-artifact audit; stdlib only, no model import or host access."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/poojith-devan/Desktop/Trainium')
REF = '2718bc5c2de31a73ef01ffaad753eb4da0b9fe45'
PREFIX = 'research/v2/operator/20260926-trn22-base-cache-cpu/results-complete/'
PACKET = 'research/v2/proposals/20260926-base-cache-exact-cpu/'
PACKET_REF = '2cf504d9b763e98e7130447d52fc93799e26db50'
rows = []

def check(name, value):
    if not value:
        raise RuntimeError('FAILED: ' + name)
    rows.append(name)

def raw(path, ref=REF):
    return subprocess.check_output(['git', '-C', str(REPO), 'show', ref + ':' + path])

def get(name):
    return json.loads(raw(PREFIX + name))

def sha(data):
    return hashlib.sha256(data).hexdigest()

manifest = get('RETRIEVAL_MANIFEST.json')
check('88 complete retrieved host artifacts', len(manifest['files']) == 88)
for name, item in manifest['files'].items():
    data = raw(PREFIX + name)
    check('artifact bytes/hash ' + name, len(data) == item['bytes'] and sha(data) == item['sha256'])
plan = get('stage/CPU_PLAN.disabled.json')
check('reviewed exact CPU plan', raw(PREFIX + 'stage/CPU_PLAN.disabled.json') == raw(PACKET + 'CPU_PLAN.disabled.json', PACKET_REF))
check('plan pin', sha(raw(PREFIX + 'stage/CPU_PLAN.disabled.json')) == 'b0c2a4abbce771a344e886d8a6da0723d42314997db245bdccb6dcd5b39202bc')
check('synthetic qualification not production equivalent', plan['diagnostic_tokens_all_ranks_per_update'] == 2048 and plan['source_global_tokens_per_update'] == 131072 and plan['production_update_equivalent'] is False)
staging = get('stage/staging-receipt.json')
for key in ('reference', 'candidate', 'g2'):
    source = plan['sources'][key]
    check('git source pin ' + key, sha(raw(source['git_path'], source['commit'])) == source['sha256'])
    rel = source['stage_path'].removeprefix(plan['stage_root'] + '/')
    check('staged source/tool ' + key, staging['hashes'][rel] == source['sha256'])
check('unchanged real prepare', sha((REPO / 'prepare.py').read_bytes()) == plan['sources']['prepare']['sha256'])
expected_order = ['reference-6ef7', 'candidate-default', 'candidate-off', 'candidate-on']
check('four prescribed arms', plan['ordering'] == expected_order)
reference = None
checkpoint_identity = None
previous_exit = None
arms = []
for index, probe in enumerate(plan['probes']):
    name = probe['name']
    def arm(file):
        return get(name + '/' + file)
    request, launch, terminal, outer = [arm(f + '.json') for f in ('request', 'launch', 'terminal', 'outer-flock-exit')]
    report, guard, preflight, postflight = [arm(f + '.json') for f in ('g2', 'under-lock-idle', 'preflight', 'operator-postflight')]
    source = plan['sources']['reference' if index == 0 else 'candidate']['sha256']
    for file, field in [('request.json', 'request_sha256'), ('host_driver.py', 'driver_sha256')]:
        data = raw(PREFIX + name + '/' + file)
        check(name + ' reviewed ' + file, sha(data) == probe[field] and data == raw(PACKET + name + '/' + file, PACKET_REF))
    check(name + ' source binding', report['train_py_sha256'] == request['source_sha256'] == launch['source_sha256'] == terminal['source_sha256'] == source)
    extra = [] if index < 2 else ['--no-initial-cache-refresh' if index == 2 else '--initial-cache-refresh']
    check(name + ' exact argv', report['argv'] == request['argv'] == plan['common_train_argv'] + extra)
    check(name + ' exact G2 command', launch['command'] == request['command'] == probe['command'])
    check(name + ' real prepare hash', report['prepare']['sha256'] == plan['sources']['prepare']['sha256'] and report['prepare']['stub'] is False)
    check(name + ' full parameter/synthetic token shape', report['nproc'] == 4 and report['steps'] == 3 and report['micro_batch'] == [2, 256])
    check(name + ' CPU/no compile/3 updates forced', report['appended_argv'][:5] == ['--device-type', 'cpu', '--no-compile', '--num-steps', '3'])
    check(name + ' explicit successful exits', report['torchrun_rc'] == terminal['exit_code'] == outer['returncode'] == 0 and report['timed_out'] is False and terminal['cpu_only'] is True)
    check(name + ' standard G2 pass', report['verdict'] == terminal['verdict'] == 'PASS')
    check(name + ' prescribed live flock', outer['command'] == probe['expected_outer_command'] and guard['lock']['path'] == plan['device_lock'] and guard['lock']['verified_existing_exclusive_lock'] is True and (guard['lock']['device'], guard['lock']['inode']) == (66305, 832589))
    check(name + ' both fresh idle checks', guard['status'] == 'PASS' and guard['ps']['returncode'] == guard['neuron_ls']['returncode'] == 0 and guard['ps']['busy'] == [] and guard['neuron_ls']['process_pids'] == [])
    check(name + ' under-lock chronology', outer['started_utc'] <= guard['started_utc'] <= guard['ended_utc'] <= launch['utc'] <= terminal['utc'] <= outer['ended_utc'])
    check(name + ' serialized arms', previous_exit is None or previous_exit < outer['started_utc'])
    previous_exit = outer['ended_utc']
    check(name + ' resource floors', preflight['available_ram_bytes'] >= 64 * 1024**3 and preflight['free_disk_bytes'] >= 6.5 * 1024**3)
    check(name + ' one thread CPU env', all(launch['environment'][k] == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')))
    check(name + ' quiescent postflight', postflight['descendants'] == [] and all(not d['neuron_processes'] for d in postflight['neuron_ls']))
    ranks = sorted(report['ranks'], key=lambda r: r['rank'])
    check(name + ' all unique ranks', [r['rank'] for r in ranks] == list(range(4)))
    for rank in ranks:
        r = rank['rank']
        check(name + ' raw rank equality ' + str(r), rank == arm('rank' + str(r) + '.json'))
        check(name + ' complete updated params ' + str(r), rank['status'] == 'PASS' and rank['n_params'] == rank['n_changed'] == 62 and rank['cross_rank_mismatches'] == [] and rank['unchanged_params'] == [])
        check(name + ' full117M ownership ' + str(r), sum(rank['plan_summary']['elements_by_owner']) == 117453580 and rank['plan_sha256'] == report['plan_sha256'] == plan['expected_plan_sha256'])
        check(name + ' 3 finite losses ' + str(r), len(rank['losses']) == 3 and all(math.isfinite(v) for v in rank['losses']))
        check(name + ' CPU scope disclosed ' + str(r), rank['torch'] == '2.11.0+cpu' and any('deterministic fake clock' in s for s in rank['shims']) and any('synthetic dataloader' in s for s in rank['shims']))
        check(name + ' no RMSProp branch tested ' + str(r), rank['rmsprop_trace']['calls'] == 0 and rank['rmsprop_trace']['checks'] == [] and rank['rmsprop_trace']['optimizer_steps'] == 3)
    ck = ranks[0]['checkpoint_check']
    check(name + ' checkpoint reload', ck == {'deterministic': True, 'logits_shape': [1, 256, 8192], 'ok': True, 'path_exists': True})
    digests = ranks[0]['final_digests']
    check(name + ' all62 final digests', len(digests) == 62 and all(len(x) == 16 for x in digests.values()))
    values = ([r['losses'] for r in ranks], digests, report['plan_sha256'], ranks[0]['plan_summary'])
    if reference is None:
        reference = values
    check(name + ' own exact loss/digest/ownership comparison', values == reference)
    checkpoint = arm('checkpoint-identity.json')
    identity = (checkpoint['sha256'], checkpoint['bytes'])
    if checkpoint_identity is None:
        checkpoint_identity = identity
    check(name + ' whole checkpoint SHA/size equality', identity == checkpoint_identity and identity[1] == 469834849 and len(identity[0]) == 64)
    if index:
        comparison = arm('comparison-vs-reference.json')
        observed = json.loads(comparison['stdout'])
        check(name + ' unchanged original comparator', comparison['returncode'] == 0 and comparison['stderr'] == '' and comparison['command'][2] == plan['sources']['g2']['stage_path'] and comparison['command'][3] == '--compare')
        check(name + ' comparator raw exact result', observed['identical'] is True and observed['plan_equal'] is True and observed['losses_identical'] is True and observed['loss_differences'] == {} and observed['state_keys_only_in_one'] == [] and observed['param_digest_differences'] == [] and observed['n_params_compared'] == 62 and observed['a_sha256'] == plan['sources']['reference']['sha256'] and observed['b_sha256'] == source)
    arms.append({'name': name, 'label': terminal['label'], 'g2_sha256': sha(raw(PREFIX + name + '/g2.json')), 'start_utc': outer['started_utc'], 'exit_utc': outer['ended_utc'], 'elapsed_seconds': report['elapsed_s']})

print(json.dumps({'verdict': 'PASS_FULL_PARAMETER_STANDARD_G2_IDENTITY_ONLY', 'checks': len(rows), 'tests': rows,
 'raw_commit': REF, 'retrieval_manifest_sha256': sha(raw(PREFIX + 'RETRIEVAL_MANIFEST.json')),
 'candidate_sha256': plan['sources']['candidate']['sha256'], 'reference_sha256': plan['sources']['reference']['sha256'],
 'artifacts_authenticated': 88, 'raw_rank_reports': 16, 'arms': arms, 'losses_by_rank': reference[0],
 'equal_final_parameter_digests': len(reference[1]), 'plan_sha256': reference[2],
 'checkpoint_sha256': checkpoint_identity[0], 'checkpoint_bytes': checkpoint_identity[1],
 'synthetic_tokens_per_update': 2048, 'production_update_tokens': 131072,
 'native_compiler_correctness': 'NOT_TESTED', 'cold_startup': 'NOT_TESTED', 'quality_or_speedup': 'NOT_TESTED',
 'host_or_tensor_execution_by_auditor': False, 'native_launch_authority': False}, indent=2))
