"""Read-only Supervisor authentication of actual clean r2 cold-five evidence."""
import csv
import hashlib
import io
import json
import math
import pathlib
import re
import subprocess
from datetime import datetime

REPO = pathlib.Path('/home/poojith-devan/Desktop/Trainium')
PREFIX = 'research/v2/operator/20260926-trn22-confirmed36-clean-cold-five-r2/'
RAW = 'aa43bd8f'
PACKET = 'c1384be93c2d314b49dd574f4b936a732f0b1a26'
PROPOSAL = 'research/v2/proposals/20260926-confirmed36-clean-cold-five-trn22-r2/'
checks = []

def check(name, condition):
    if not condition:
        raise RuntimeError(name)
    checks.append(name)

def blob(rev, path):
    return subprocess.check_output(['git', 'show', rev + ':' + path], cwd=REPO)

def sha(value):
    return hashlib.sha256(value).hexdigest()

def read(rev, path):
    return json.loads(blob(rev, path))

raw = blob(RAW, PREFIX + 'completed/results-bundle.json')
bundle = json.loads(raw)
post_raw = blob(RAW, PREFIX + 'completed/explicit-process-postflight.json')
post = json.loads(post_raw)
check('frozen raw bundle', sha(raw) == 'c03997f87f4bb95f539b0e4ac10bfeff012e1ac0dc12b3ead27b3dcc25aa9c87')
check('frozen explicit postflight', sha(post_raw) == '2dcb3a9f2513a17842c2f48dc860b97fc957563b2a0e5c44c769445e1be053cd')
files = bundle['files']
check('26 complete files', len(files) == 26)
for name, entry in files.items():
    data = entry['text'].encode()
    check('closed file ' + name, len(data) == entry['bytes'] and sha(data) == entry['sha256'])
check('aggregate bytes', sum(e['bytes'] for e in files.values()) == bundle['raw_bytes'] == 156573)
j = lambda name: json.loads(files[name]['text'])
b = j('CT_BINDING.disabled.json')
check('exact binding', files['CT_BINDING.disabled.json']['text'].encode() == blob(PACKET, PROPOSAL + 'CT_BINDING.disabled.json'))
for name in ('host_driver.py', 'EXPERIMENT_PROPOSAL.json', 'MANIFEST.sha256.json', *b['package_files']):
    check('exact packet ' + name, files[name]['text'].encode() == blob(PACKET, PROPOSAL + name))
source = blob(b['source_commit'], b['source_git_path'])
check('clean scientific bytes', files['package/train.py']['text'].encode() == source and sha(source) == b['source_sha256'] == 'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43')
for name, key in [('idle_guard.py', 'idle_guard_sha256'), ('dispatch_driver.py', 'dispatch_driver_sha256'), ('host_driver.py', 'driver_sha256')]:
    check('support ' + name, files[name]['sha256'] == b[key])
pre, launch, guard = j('preflight.json'), j('training-launch.json'), j('under-lock-idle.json')
check('actual command and environment', launch['command'] == b['command'] and launch['environment'] == b['training_environment'])
check('exact source/prepare/harness', all(launch[k] == b[k] for k in ('source_sha256', 'prepare_sha256', 'harness_sha256')))
check('new contained root', bundle['root'] == b['root'] and pathlib.PurePosixPath(b['sandbox']).is_relative_to(pathlib.PurePosixPath(b['harness_path']).parents[2] / 'out'))
check('actual staging hashes', all(files[name]['sha256'] == digest for name, digest in pre['staged_hashes'].items()))
check('preflight headroom', pre['free_bytes'] >= int(13.5 * 2**30) and pre['staging_checks_under_original_lock'])
check('fresh idle checks', guard['status'] == 'PASS' and guard['ps']['returncode'] == guard['neuron_ls']['returncode'] == 0 and guard['ps']['busy'] == guard['neuron_ls']['process_pids'] == [])
check('original lifetime lock observed', guard['lock']['verified_existing_exclusive_lock'] and (guard['lock']['device'], guard['lock']['inode']) == (66305, 832589))
check('idle before launch', datetime.fromisoformat(guard['ended_utc']) < datetime.fromisoformat(launch['utc']))
log = files['training.log']['text']
iso = json.loads(next(l[len('ISOLATED_SUBMISSION '):] for l in log.splitlines() if l.startswith('ISOLATED_SUBMISSION ')))
argv = json.loads(next(l[len('REHEARSAL_COMMAND '):] for l in log.splitlines() if l.startswith('REHEARSAL_COMMAND ')))
check('five-step-only diagnostic override', argv == b['training_command'] == b['base_training_command'] + ['--num-steps', '5'])
check('actual source sandbox', iso['train_sha256'] == b['source_sha256'] and iso['prepare_sha256'] == b['prepare_sha256'] and iso['sandbox'] == b['sandbox'])
cache = iso['fresh_empty_sdk_caches']
check('six initially absent private caches', len(cache) == len(set(cache.values())) == 6 and set(pre['six_caches_absent_before_harness']) == set(cache.values()) and all(pathlib.PurePosixPath(v).is_relative_to(b['sandbox']) for v in cache.values()))
live = read('65b377e179c3330e9542348fa35be30ce5a19ece', PREFIX + 'trn22.json')['remote']
boundary = read('072e5641', PREFIX + 'progress/boundary-0428.json')['remote']
union = {}
for snap in (live, boundary):
    for name, entry in snap['files'].items():
        data = entry['text'].encode()
        check('captured bytes ' + snap['utc'] + ' ' + name, len(data) == entry['captured_bytes'] and sha(data) == entry['sha256_captured'])
    for row in snap['processes']:
        if row.get('exists'):
            check('stable observed identity ' + str(row['pid']), row['identity_stable'])
            if row['pid'] in union:
                check('same continuing identity ' + str(row['pid']), all(row[k] == union[row['pid']][k] for k in ('pid', 'ppid', 'pgid', 'sid', 'starttime')))
            union[row['pid']] = row
    check('original held lock at snapshot ' + snap['utc'], snap['lock']['held'] and (snap['lock']['device'], snap['lock']['inode']) == (66305, 832589))
workers = [p for p in live['processes'] if 'RANK' in p.get('environment_whitelist', {})]
check('four actual native ranks', len(workers) == 4 and {p['environment_whitelist']['RANK'] for p in workers} == {'0', '1', '2', '3'})
for p in workers:
    env = p['environment_whitelist']
    check('worker route ' + env['RANK'], p['argv'][2:] == argv[3:] and p['cwd'] == b['sandbox'] and env['WORLD_SIZE'] == '4' and env['LOCAL_RANK'] == env['RANK'] and env['NEURON_LOGICAL_NC_CONFIG'] == '2')
    check('worker actual cold cache env ' + env['RANK'], all(env[k] == v for k, v in cache.items()) and env['NEURON_COMPILE_CACHE_URL'] == env['NEURON_COMPILE_CACHE_DIR'])
    check('worker unchanged budget/seed/flags ' + env['RANK'], all(env[k] == b['training_environment'][k] for k in ('NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS', 'NEURON_COMPETITION_R1_SEED', 'NEURON_CC_FLAGS')))
r = j('train/rehearsal.json')
obs = r['observations']
check('five real ordered boundaries', [o['step'] for o in obs] == list(range(5)) and r['step_boundaries'] == 5 and all(math.isfinite(o['observed_seconds']) for o in obs) and all(a['observed_seconds'] < z['observed_seconds'] for a, z in zip(obs, obs[1:])))
fifth = obs[-1]['observed_seconds']
check('uncapped cold qualification', fifth == 859.1679093550192 and fifth <= b['cold_decision']['target_seconds'] == 870 and fifth <= b['cold_decision']['maximum_seconds'] == 900)
check('official charged arithmetic', r['wall_seconds'] == 863.638870780007 and r['startup_allowance_observed'] == min(fifth, 900) and r['charged_through_process_exit'] == r['wall_seconds'] - min(fifth, 900) == 4.470961424987763 and r['charged_through_process_exit_within_budget'] is True)
check('actual cold assignment retained', r['fresh_empty_sdk_caches'] == cache)
steps = re.findall(r'^step (\d+) \| loss (\S+) \| lrm (\S+) \| dt (\S+)s \|', log, re.M)
check('five finite losses', [int(s[0]) for s in steps] == list(range(5)) and all(math.isfinite(float(v)) for s in steps for v in s[1:]))
check('loss matches confirmed research start', [float(s[1]) for s in steps] == [9.0101, 8.7779, 8.4047, 8.0711, 7.7681])
check('official step markers', [json.loads(l) for l in files['train/steps.jsonl']['text'].splitlines()] == list(range(5)))
check('full native shape', 'num_params=113,256,724' in log and 'device=neuron dtype=torch.bfloat16 world_size=4' in log and 'micro_tokens=131,072 grad_accum_steps=1 pack_factor=32' in log and 'over_budget:       False' in log)
summary = json.loads(next(l[len('SUBMISSION_REHEARSAL '):] for l in log.splitlines() if l.startswith('SUBMISSION_REHEARSAL ')))
check('independent log receipt agrees', summary == {k: v for k, v in r.items() if k != 'observations'})
terminal, outer = j('terminal.json'), j('outer-flock-exit.json')
check('aggregate completion all zero', r['returncode'] == terminal['returncode'] == outer['returncode'] == int(files['run-exit-code.txt']['text']) == 0)
check('no eval or automatic full', terminal['no_eval'] and terminal['no_automatic_full'] and outer['command'] == b['outer_command'])
ckpt = bundle['checkpoint_after_terminal']
check('checkpoint post-terminal full rehash', ckpt['stable'] and ckpt['sha256'] == r['checkpoint_sha256'] == '8e867c7a541f799e96502001e91ea06388e46fe0c2159a7c07c684d947b9dbde' and ckpt['bytes'] == r['checkpoint_bytes'] == 453044626)
for path, identity in bundle['retained_input_identities'].items():
    expected = b['prepare_sha256'] if path == b['prepare_path'] else b['harness_sha256'] if path == b['harness_path'] else b['source_sha256']
    check('retained immutable input ' + path, identity['stable'] and identity['sha256'] == expected)
check('all actual observed identities included', set(union) == set(post['observed_pids']) == {int(x) for x in post['known_prior_identities']} and len(union) == 29)
for pid, row in union.items():
    check('postflight original identity ' + str(pid), all(row[k] == post['known_prior_identities'][str(pid)][k] for k in ('pid', 'ppid', 'pgid', 'sid', 'starttime')))
check('explicit observed PID absence', len(post['pid_rows']) == 29 and {p['pid'] for p in post['pid_rows']} == set(union) and all(p['exists'] is False for p in post['pid_rows']))
census = post['complete_process_census']
check('complete unique census', len(census) == len({p['pid'] for p in census}) == 255 and post['ps_rc'] == 0)
check('independent ancestry absence', not [p for p in census if any(p[k] in union for k in ('pid', 'ppid', 'pgid', 'sid'))])
check('postflight no declared job/compiler', post['related_rows'] == post['busy_job_rows'] == [] and post['all_observed_pids_absent'])
check('postflight device and original lock idle', post['neuron_ls_rc'] == 0 and post['neuron_ls_stderr'] == '' and all(d['neuron_processes'] == [] for d in post['neuron_ls']) and post['fuser_rc'] == 1 and post['fuser_stdout'] == post['fuser_stderr'] == '' and post['lock_competitor_rc'] == 0 and (post['lock_device'], post['lock_inode']) == (66305, 832589))
check('postflight after actual complete exit', datetime.fromisoformat(post['utc']) > datetime.fromisoformat(outer['ended_utc']) > datetime.fromisoformat(terminal['utc']) > datetime.fromisoformat(launch['utc']))
graph_counts = {}
for rank in range(4):
    rows = list(csv.DictReader(io.StringIO(files[f'metrics-training/metric_table_graph_stats_rank{rank}.csv']['text'])))
    check('actual graph records rank' + str(rank), any('_forward_' in row['graph_name'] for row in rows) and any('_backward_' in row['graph_name'] for row in rows))
    graph_counts[str(rank)] = len(rows)
print(json.dumps({'status': 'ACCEPT_ACTUAL_COLD_FIVE_ONLY', 'checks': len(checks), 'checks_passed': checks, 'source_sha256': b['source_sha256'], 'raw_commit': RAW, 'uncapped_fifth_seconds': fifth, 'wall_seconds': r['wall_seconds'], 'charged_through_exit': r['charged_through_process_exit'], 'checkpoint': ckpt, 'graph_row_counts': graph_counts, 'observed_identities_postflight': len(union), 'submission_ready': False, 'full_launch_authorized': False, 'native_batch_consistency_proven': False, 'launcher_timeout_safety_approved': False, 'quality_claim': False, 'steady_throughput_proven': False, 'cache_hit_claim': False, 'no_hosts_or_tensor_execution': True}, indent=2))
