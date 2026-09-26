"""Read-only authentication of the actual full-run launch, not a quality verdict."""
import hashlib
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

R = Path('/home/poojith-devan/Desktop/Trainium')
Q = R / 'research/v2/proposals/20260926-clock36-recursive-full-trn22'
P = R / 'research/v2/operator/20260926-trn22-clock36-recursive-full/launch/first-four-rank-evidence.json'
checks = []
def need(condition, name):
    if not condition:
        raise RuntimeError('LAUNCH_REFUSE:' + name)
    checks.append(name)
def sha(data):
    return hashlib.sha256(data).hexdigest()

raw = json.loads(P.read_text())
b = json.loads((Q / 'NATIVE_BINDING.disabled.json').read_text())
m = json.loads((Q / 'MANIFEST.sha256.json').read_text())
need(sha(P.read_bytes()) == '6bb597aba9190d8bd265009b5cae7b6a2aef174e5560a524430e5fad2ca16898', 'raw-pin')
for name, item in raw['files'].items():
    data = item['text'].encode()
    need(len(data) == item['bytes'] and sha(data) == item['sha256'], 'raw-file:' + name)
get = lambda name: json.loads(raw['files'][name]['text'])
pre, guard, launch, dispatch = [get(n) for n in ('preflight.json', 'under-lock-idle.json', 'training-launch.json', 'operator-launch.json')]
for name, digest in m['files'].items():
    need(sha((Q / name).read_bytes()) == digest, 'frozen:' + name)
    if name in pre['staged_hashes']:
        need(pre['staged_hashes'][name] == digest, 'staged:' + name)
source = subprocess.check_output(['git', 'show', b['source_commit'] + ':' + b['source_git_path']], cwd=R)
need(sha(source) == b['source_sha256'] == pre['staged_hashes']['package/train.py'], 'source')
need(pre['staged_hashes']['idle_guard.py'] == m['guard_sha256'] and pre['staged_hashes']['dispatch_driver.py'] == m['dispatch_sha256'], 'guard-dispatch')
need(launch['command'] == b['command'] and launch['environment'] == b['training_environment'], 'command-environment')
for key in ('source_sha256', 'prepare_sha256', 'harness_sha256'):
    need(launch[key] == pre[key] == b[key], key)
need(raw['label'] == launch['label'] == dispatch['label'] == b['label'], 'label')
need(guard['status'] == 'PASS' and guard['lock']['verified_existing_exclusive_lock'], 'guard-pass')
need(guard['lock']['device'] == pre['lock_device'] == raw['lock']['device'] == 66305 and guard['lock']['inode'] == pre['lock_inode'] == raw['lock']['inode'] == 832589 and raw['lock']['held'], 'original-lock')
need(guard['ps']['returncode'] == guard['neuron_ls']['returncode'] == 0 and guard['ps']['busy'] == guard['neuron_ls']['process_pids'] == [], 'fresh-idle')
need(datetime.fromisoformat(guard['ended_utc']) < datetime.fromisoformat(launch['utc']) < datetime.fromisoformat(raw['utc']), 'guard-order')
need(pre['free_bytes'] >= 10.5 * 1024**3 and pre['fuser_rc'] == 1 and pre['fuser_stdout'] == pre['fuser_stderr'] == '', 'disk-device')
need(pre['prior_diagnostic_lineage_rows'] == [] and pre['previous_diagnostic_terminal']['returncode'] == pre['previous_diagnostic_outer']['returncode'] == 0, 'prior-terminal')
need(all(not x['exists'] for x in raw['retired_queue_pids']), 'retired-queues')
rows = {x['pid']: x for x in raw['processes']}
need(set(rows) == {3252200,3252201,3252202,3252218,3252219,3252233,3252248,3252249,3252250,3252251}, 'process-set')
need(all(x['identity_stable'] and x['starttime_ticks'] > 0 for x in rows.values()), 'stable-identities')
for child, parent in ((3252201,3252200),(3252202,3252201),(3252218,3252202),(3252219,3252218),(3252233,3252219)):
    need(rows[child]['ppid'] == parent, 'lineage:' + str(child))
ranks = set()
for pid in (3252248,3252249,3252250,3252251):
    w = rows[pid]
    e = w['environment_whitelist']
    ranks.add(e['RANK'])
    need(w['ppid'] == 3252233 and w['pgid'] == w['sid'] == pid and w['starttime_ticks'] == 100573985, 'worker-identity:' + str(pid))
    need(e['RANK'] == e['LOCAL_RANK'] and e['WORLD_SIZE'] == '4', 'rank:' + str(pid))
    need(w['cwd'] == b['sandbox'] and shlex.split(w['argv'])[2:] == b['training_command'][3:] and w['actual_train_sha256'] == b['source_sha256'], 'worker-source-argv:' + str(pid))
    need(all(e[k] == v for k, v in b['training_environment'].items() if k in e), 'worker-environment:' + str(pid))
need(ranks == {'0','1','2','3'}, 'four-ranks')
need('--max-train-seconds' in b['training_command'] and b['training_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'] == '1800', 'budget-request')
need('Traceback (most recent call last)' not in raw['files']['training.log']['text'], 'no-startup-traceback')
print(json.dumps({'status':'PASS_ACTUAL_LAUNCH_ONLY','checks':len(checks),'raw_sha256':sha(P.read_bytes()),'dispatch_utc':dispatch['utc'],'training_phase_utc':launch['utc'],'observed_utc':raw['utc'],'source_sha256':b['source_sha256'],'four_workers':True,'original_lifetime_lock':True,'cache_hit_proven':False,'full_startup_seconds':None,'quality_bpb':None,'new_launch_authorized':False}, indent=2))
