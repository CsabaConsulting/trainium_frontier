"""Authenticate the completed clean cold replay; no native parity/audit waiver."""
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'research/v2/operator/20260926-trn22-eval-boundary-0557/completed'
count = 0


def check(ok, label):
    global count
    if not ok:
        raise RuntimeError(label)
    count += 1


def pinned(name, digest):
    raw = (BASE / name).read_bytes()
    check(hashlib.sha256(raw).hexdigest() == digest, name)
    return json.loads(raw)


r = pinned('results-bundle.json', '20e1e5f920dff44a4d0a047790e02397ed916945e975a6822d6385efe2f9151d')
p = pinned('explicit-process-postflight.json', 'ebbbf6ce4732cc01ed6ead64787239d1208d5da3fa8db120f8ae82d0576da418')
files = r['files']
check(len(files) == 43 and len(r['artifact_inventory']) == 52, '43 full artifacts plus8 directories and retained checkpoint')
regular = {x['path'] for x in r['artifact_inventory'] if stat.S_ISREG(x['mode'])}
check(regular == set(files) | {'train/out/final.pt'}, 'all non-checkpoint regular artifacts captured')
check(all(stat.S_ISREG(x['mode']) or stat.S_ISDIR(x['mode']) for x in r['artifact_inventory']), 'no omitted links or special files')
check({x['path'] for x in r['skipped_paths']} == {'train/out/final.pt', 'train/caches', 'eval/caches'}, 'only checkpoint and two caches excluded')
for name, f in files.items():
    raw = f['text'].encode()
    check(len(raw) == f['bytes'] and hashlib.sha256(raw).hexdigest() == f['sha256'], name)


def get(name):
    return json.loads(files[name]['text'])


train = get('train/rehearsal.json')
evaluation = get('eval/rehearsal.json')
check(train['returncode'] == evaluation['returncode'] == 0, 'both phases exit0')
check(files['run-exit-code.txt']['text'].strip() == files['eval-exit-code.txt']['text'].strip() == '0', 'phase exits')
steps = [int(x) for x in files['train/steps.jsonl']['text'].splitlines()]
check(steps == list(range(3628)), '3628 sequential completed steps')
check([x['step'] for x in train['observations']] == steps, 'timed boundaries match')
times = [x['observed_seconds'] for x in train['observations']]
check(all(math.isfinite(x) and x >= 0 for x in times) and all(a <= b for a, b in zip(times, times[1:])), 'finite monotonic times')
startup = times[4]
charge = train['wall_seconds'] - min(startup, 900.)
check(startup == train['startup_allowance_observed'] == 859.4735238229623, 'actual fifth')
check(startup < 870 and charge <= 1800 and abs(charge - train['charged_through_process_exit']) < 1e-9, 'budget arithmetic')
check(train['charged_through_process_exit_within_budget'] is True and train['wall_seconds'] > times[-1], 'checkpoint/exit included')
log = files['fresh-public20m.log']['text']
match = re.search(r'(?m)^\{\n  "eval_over_timeout".*?\n\}', log, re.S)
check(match is not None, 'scorer result exists')
score = json.loads(match.group())
check(score['eval_over_timeout'] is False and score['eval_seconds'] <= score['eval_timeout'] == 1200, 'eval timeout pass')
check(score['num_tokens'] == 20971520 and score['num_bytes'] == 83462186, 'public20M totals')
check(score['val_bpb'] == 0.9883890652263234 and abs(score['val_bpb'] - score['total_nats'] / (math.log(2) * score['num_bytes'])) < 1e-12, 'independent BPB arithmetic')
argv = json.loads(next(x.removeprefix('REHEARSAL_COMMAND ') for x in log.splitlines() if x.startswith('REHEARSAL_COMMAND ')))
check(argv[1:3] == ['prepare.py', 'eval-public'], 'ordinary fresh scoring')
for flag, value in [('--batch-size', '1'), ('--seq-len', '1024'), ('--eval-tokens', '20971520'), ('--device', 'neuron')]:
    check(argv[argv.index(flag) + 1] == value, flag)
source = 'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43'
for name in ('package/train.py', 'train/train.py', 'eval/train.py'):
    check(files[name]['sha256'] == source, 'same exact candidate ' + name)
for name in ('train/prepare.py', 'eval/prepare.py'):
    check(files[name]['sha256'] == '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb', 'unchanged scorer ' + name)
check(files['package/launch-command.txt']['sha256'] == 'b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd', 'unchanged approved command')
checkpoint = r['checkpoint_after_terminal']
check(checkpoint['stable'] and checkpoint['bytes'] == 453044626 and checkpoint['sha256'] == get('checkpoint.json')['sha256'] == get('checkpoint-after-postflight.json')['sha256'] == '4d024cce5120189010d0017fcbf6aaac22a064506eaeb8ca98b453972997d2f2', 'checkpoint retained unchanged')
check(get('terminal.json')['eval_exit_code'] == 0 and get('terminal.json')['submission_authorized'] is False, 'terminal not submission permission')
check(p['all_observed_pids_absent'] and len(p['pid_rows']) == len(p['observed_pids']) == 702, 'all observed identities covered')
check({x['pid'] for x in p['pid_rows']} == set(p['observed_pids']) and all(x['exists'] is False for x in p['pid_rows']), 'every observed PID absent')
check(p['busy_job_rows'] == p['related_rows'] == [] and p['ps_rc'] == 0 and len(p['complete_process_census']) == 252, 'full process census clean')
check(p['neuron_ls_rc'] == 0 and p['neuron_device_idle'] and p['fuser_rc'] == 1 and p['fuser_stdout'] == p['fuser_stderr'] == '', 'device idle')
check((p['lock_device'], p['lock_inode'], p['lock_competitor_rc']) == (66305, 832589, 0), 'original lock free')
check(get('postflight-under-lock-idle.json')['status'] == 'PASS', 'guardian final idle check')
check('waitpid_ECHILD_AND_POSTFLIGHT_PASS' in files['lifetime-events.jsonl']['text'], 'guardian descendants reaped before release')
print(json.dumps({'verdict': 'ACCEPT_CLEAN_COLD_REPLAY_PUBLIC20M_ONLY', 'checks_passed': count, 'optimized': bool(sys.flags.optimize), 'public20m': score['val_bpb'], 'steps': len(steps), 'cold_startup_seconds': startup, 'charged_seconds': charge, 'over_budget': False, 'checkpoint_sha256': checkpoint['sha256'], 'native_consistency_pending': True, 'human_claude_audit_pending': True, 'submission_ready': False}, sort_keys=True))
