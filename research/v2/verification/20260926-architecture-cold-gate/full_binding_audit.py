#!/usr/bin/env python3
"""Mechanical audit only; never executes the remote driver."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[4]
P = ROOT/'research/v2/operator/20260926-trn22-arch3e79-full'
COLD = ROOT/'research/v2/operator/20260926-ct-architecture-startup-only'
checks = []
def check(ok, name):
    if not ok: raise RuntimeError(name)
    checks.append(name)
def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
b = read(P/'NATIVE_BINDING.disabled.json')
p = read(P/'EXPERIMENT_PROPOSAL.json')
check(sha(P/'NATIVE_BINDING.disabled.json') == 'bb24dee24176414b014997c039b3465ed4a309bbef048ba952eaf414fcc30823', 'binding SHA')
check(sha(P/'EXPERIMENT_PROPOSAL.json') == '6232be990ec3f4c734e0f683d877b1d320b0c9dd5e237d70b15a7cbeec053209', 'proposal SHA')
check(sha(P/'host_driver.py') == b['driver_sha256'] == p['driver_sha256'] == 'c8957e5dbd67bbf017510d6110bcd0e0ab09d2183db098d18654a54f5498e7e0', 'driver SHA')
source = subprocess.check_output(['git','show',b['source_commit']+':'+b['source_git_path']],cwd=ROOT)
check(hashlib.sha256(source).hexdigest() == b['source_sha256'], 'exact committed science')
worker = next(x for x in read(COLD/'launched/four-worker-effective-cold-cache.json')['processes'] if x.get('selected_environment',{}).get('RANK') == '0')
cold_argv = worker['argv'][worker['argv'].index('train.py')+1:]
cold_argv[cold_argv.index('--num-steps')+1] = '1000000'
check(cold_argv == b['training_command'][4:] == p['argv'][4:], 'only cold-to-full science change is step cap')
check(b['training_command'][:4] == ['/home/ubuntu/workspace/native_venv/bin/torchrun','--standalone','--nproc_per_node=4','train.py'], 'full chip launch')
check('--recursive-schedule' not in cold_argv, 'isolated architecture not untested stack')
raw = read(COLD/'completed/train/rehearsal.json')
check(raw['observations'][4]['observed_seconds'] == p['cold_gate']['observed_uncapped_fifth_seconds'] <= 900, 'actual cold gate qualification')
check(sha(COLD/'completed/train/rehearsal.json') == p['cold_gate']['receipt_sha256'], 'cold receipt pin')
for key,value in raw['fresh_empty_sdk_caches'].items():
    check(b['training_environment'][key] == value, 'reuse proven training cache '+key)
check(b['training_environment']['NEURON_COMPILE_CACHE_URL'] == b['training_environment']['NEURON_COMPILE_CACHE_DIR'], 'compiler cache URL binding')
tree = ast.parse((P/'host_driver.py').read_text())
constants = {}
for node in tree.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        try: constants[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError,TypeError): pass
for key,field in [('TRAIN_COMMAND','command'),('EVAL_COMMAND','evaluation_command'),('TRAIN_ENV','training_environment'),('EVAL_ENV','evaluation_environment'),('SOURCE_SHA','source_sha256'),('PREP_SHA','prepare_sha256'),('HARNESS_SHA','harness_sha256')]:
    check(constants[key] == b[field], 'driver literal '+key)
check(constants['TRAIN_COMMAND'][4]=='2730s' and constants['EVAL_COMMAND'][4]=='1800s', 'bounded watchdogs separate from source budget')
check(constants['TRAIN_ENV']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS']=='1800', 'source budget1800')
check(constants['EVAL_COMMAND'][-2:]==['--eval-tokens','20971520'], 'fresh public20M')
check(constants['EVAL_COMMAND'][constants['EVAL_COMMAND'].index('--checkpoint')+1]==b['root']+'/train/out/final.pt', 'fresh trained checkpoint')
guard_root = ROOT/'research/v2/operator/20260926-common-host-idle-guard'
for name,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
    check(sha(guard_root/name)==b[key]==p[key], 'reviewed '+name)
check(b['outer_command'][:6]==['/usr/bin/flock','-n','/home/ubuntu/.cache/trainium_freshstart/device0.lock','/usr/bin/python3','-I','-B'], 'same isolated lifetime flock')
calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)]
guards = [n for n in calls if ast.unparse(n.func)=='guard.require_idle_under_inherited_lock']
check(len(guards)==2, 'idle guard before both phases')
child = [n for n in calls if ast.unparse(n.func)=='subprocess.call']
check(len(child)==1 and any(k.arg=='pass_fds' and ast.unparse(k.value)=='(lock_fd,)' for k in child[0].keywords), 'both phases share synchronous inherited lock')
check(any(isinstance(n,ast.Assert) and ast.unparse(n.test)=='eval_fd == lock_fd' for n in ast.walk(tree)), 'same descriptor asserted between phases')
harness = ROOT/'runs/Trainium-current-base-dual-control/research/scratch/submission_rehearsal.py'
check(sha(harness)==b['harness_sha256'], 'actual unchanged harness read locally')
text = harness.read_text()
check('"--batch-size", "1", "--seq-len", "1024", "--eval-tokens", str(args.eval_tokens)' in text, 'harness actual eval settings')
check('allowance = min(observed[4]["observed_seconds"], 900.0)' in text, 'capped actual observer allowance')
check('result["charged_through_process_exit"] <= 1800' in text, 'process exit budget enforced')
check(b['same_host_control']['label']=='fs-20260925-v2-c5c-a0-control-trn22-r10' and b['same_host_control']['public20m_bpb']==0.991513958330512, 'correct pre-schedule same-host comparator')
print(json.dumps({'status':'PASS_MECHANICAL_FULL_BINDING','count':len(checks),'checks':checks,'host_execution':False},indent=2))
