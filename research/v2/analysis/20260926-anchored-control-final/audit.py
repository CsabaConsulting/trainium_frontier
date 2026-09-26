"""Read-only independent Supervisor archive audit; no native or model execution."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / 'research/v2/operator/20260926-trn21-anchored-boundary-0450/completed'
SOURCE = '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'
COUNT = 0

def check(value, message):
    global COUNT
    if not value:
        raise RuntimeError(message)
    COUNT += 1

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

raw = (ARCHIVE / 'results-bundle.json').read_bytes()
check(digest(raw) == '8d48db60dea8dee0d591cd2a51c7649ae617d4594334ee89f2dd2bd39e4766ea', 'frozen bundle')
b = json.loads(raw)
files = b['files']
check(len(files) == 39, 'complete artifact count')
for name, record in files.items():
    value = record['text'].encode()
    check(len(value) == record['bytes'] and digest(value) == record['sha256'], name)

def parsed(name):
    return json.loads(files[name]['text'])

binding = parsed('ANCHORED_BINDING.disabled.json')
proposal = parsed('EXPERIMENT_PROPOSAL.json')
check(binding['source_sha256'] == binding['base_sha256'] == SOURCE, 'source/base')
check(files['ANCHORED_BINDING.disabled.json']['sha256'] == 'e71a92a55043871b7dd5660c191fad4135d20384e6fa1be3b6edc91398d83e10', 'approved binding')
check(files['host_driver.py']['sha256'] == '706b7945a7316fe610c3abef1efedaf8a95d0203eba1516fb93ac1d4a6aa9644', 'approved driver')
scientific = subprocess.check_output(['git', 'show', binding['source_commit'] + ':' + binding['source_git_path']], cwd=ROOT)
check(digest(scientific) == SOURCE == files['package/train.py']['sha256'], 'scientific bytes')
for name, envkey in [('training-launch.json', 'training_environment'), ('fresh-public20m-launch.json', 'evaluation_environment')]:
    launch = parsed(name)
    check(launch['source_sha256'] == SOURCE and launch['prepare_sha256'] == binding['prepare_sha256'] and launch['harness_sha256'] == binding['harness_sha256'], name + ' pins')
    check(launch['environment'] == binding[envkey], name + ' environment')
for phase in ('training', 'eval'):
    copies = parsed(phase + '-copies.json')
    check(copies['train']['sha256'] == SOURCE and copies['prepare']['sha256'] == binding['prepare_sha256'], phase + ' actual copies')
anchors = [parsed(n)['files'] for n in ('anchor-pre-training.json', 'anchor-post-training.json', 'anchor-pre-eval.json', 'anchor-post-eval.json', 'anchor-final.json')]
for item in anchors:
    for name in ('executed_train', 'expected_imported_prepare'):
        check(item[name] == anchors[0][name], name + ' observed anchor stable')
    check(item['executed_train']['sha256'] == SOURCE, 'executed anchor hash')

train = parsed('train/rehearsal.json')
observations = train['observations']
check(train['returncode'] == 0 and train['step_boundaries'] == len(observations) == 3631, 'training completed')
check([r['step'] for r in observations] == list(range(3631)), 'all sequential boundaries')
times = [r['observed_seconds'] for r in observations]
check(all(math.isfinite(t) and t >= 0 for t in times) and times == sorted(times), 'finite monotone boundaries')
check([int(x) for x in files['train/steps.jsonl']['text'].splitlines()] == list(range(3631)), 'step marker file')
charge = train['wall_seconds'] - min(times[4], 900)
check(abs(charge - train['charged_through_process_exit']) < 1e-9 and 0 < charge <= 1800, 'official through-exit charge')
check(train['charged_through_process_exit_within_budget'] is True, 'budget receipt')
check(re.search(r'over_budget:\s+False', files['training.log']['text']) is not None, 'source over-budget false')
check(parsed('training-launch.json')['environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'] == '1800', 'explicit budget')
check(parsed('training-launch.json')['environment']['NEURON_COMPETITION_R1_SEED'] == '42', 'seed42')

evlog = files['fresh-public20m.log']['text']
scores = re.findall(r'\{\s*"eval_over_timeout".*?\n\}', evlog, re.S)
check(len(scores) == 1, 'single scoring result')
ev = json.loads(scores[0])
check(ev['eval_over_timeout'] is False and ev['num_tokens'] == 20971520 and ev['num_bytes'] == 83462186, 'full public20M')
bpb = ev['total_nats'] / math.log(2) / ev['num_bytes']
check(abs(bpb - ev['val_bpb']) < 1e-14 and math.isfinite(bpb), 'independent BPB arithmetic')
evalcmd = json.loads(next(x.removeprefix('REHEARSAL_COMMAND ') for x in evlog.splitlines() if x.startswith('REHEARSAL_COMMAND ')))
for flag, expected in [('--device', 'neuron'), ('--batch-size', '1'), ('--seq-len', '1024'), ('--eval-tokens', '20971520'), ('--timeout-seconds', '1200')]:
    check(evalcmd[evalcmd.index(flag) + 1] == expected, 'scorer ' + flag)
check(evalcmd[1:3] == ['prepare.py', 'eval-public'] and parsed('eval/rehearsal.json')['returncode'] == 0, 'ordinary fresh evaluator')
check(files['run-exit-code.txt']['text'].strip() == files['eval-exit-code.txt']['text'].strip() == '0', 'phase exits')
terminal = parsed('terminal.json')
check(terminal['error'] is None and terminal['final_errors'] == [] and parsed('outer-flock-exit.json')['returncode'] == 0, 'terminal exits')
check(terminal['native_cache_hit'] == terminal['path_only_causality'] == 'UNPROVEN' and terminal['actual_worker_import_origin_observed'] is False, 'retain causality limitations')
checkpoint = b['checkpoint_after_terminal']
check(checkpoint['stable'] and checkpoint['sha256'] == train['checkpoint_sha256'] == parsed('checkpoint.json')['sha256'], 'checkpoint hash stable')
check(checkpoint['bytes'] == train['checkpoint_bytes'] == 453048594, 'checkpoint bytes')

post = json.loads((ARCHIVE / 'explicit-process-postflight.json').read_bytes())
ids = set(post['observed_pids'])
check(ids == {3177618,3177619,3177620,3177651,3177652,3177666,3177681,3177682,3177683,3177684,3199120,3199121,3199135}, 'actual13 IDs')
check(all(not row['exists'] for row in post['pid_rows']) and post['all_observed_pids_absent'], 'all observed gone')
check(len(post['complete_process_census']) == 254 and not any(any(row[k] in ids for k in ('pid','ppid','pgid','sid')) for row in post['complete_process_census']), 'full ancestry census')
check(post['related_rows'] == post['busy_job_rows'] == [] and post['ps_rc'] == 0, 'no related jobs')
check(post['neuron_ls_rc'] == 0 and post['neuron_device_idle'] and all(not d['neuron_processes'] for d in post['neuron_ls']), 'device idle')
check(post['fuser_rc'] == 1 and not post['fuser_stdout'].strip() and not post['fuser_stderr'].strip(), 'no device holders')
check(post['lock_competitor_rc'] == 0 and (post['lock_device'],post['lock_inode']) == (66305,596823), 'original lock free')
print(json.dumps({'verdict':'ACCEPT_PROSPECTIVE_WARM_CONTROL_ONLY','checks_passed':COUNT,'public20m':bpb,'steps':3631,'uncapped_fifth_seconds':times[4],'charged_through_exit_seconds':charge,'checkpoint_sha256':checkpoint['sha256'],'cache_hit_proven':False,'path_only_causality_proven':False,'submission_qualified':False}, indent=2))
