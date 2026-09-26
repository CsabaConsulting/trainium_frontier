"""Independent stdlib authentication of unchanged control in guarded full runner."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PACK = REPO / 'research/v2/proposals/20260926-recursive6ef7-control-trn22-r2'
checks = []
def check(name, value):
    if not value: raise RuntimeError(name)
    checks.append(name)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def git(ref, path): return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)
def constants(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            v = node.value
            if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == 'Path': v=v.args[0]
            result[node.targets[0].id] = ast.literal_eval(v)
    return result
def body(tree):
    return ast.dump(ast.Module(body=[x for x in tree.body if not isinstance(x,ast.Assign) and not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))],type_ignores=[]))

b=json.loads((PACK/'NATIVE_BINDING.disabled.json').read_text())
p=json.loads((PACK/'EXPERIMENT_PROPOSAL.json').read_text())
m=json.loads((PACK/'MANIFEST.sha256.json').read_text())
for name,pin in m['files'].items(): check('manifest '+name,sha((PACK/name).read_bytes())==pin)
check('exact binding','dab176a8424bc750e7352877cf0c2703d855fd36b1e96b3af0bdd935446be77a'==sha((PACK/'NATIVE_BINDING.disabled.json').read_bytes()))
check('exact manifest','0f0ca8a9bd5a73fc348d23e4ae3d7ef22f7afc212e03b2b5099b4149701bc1d8'==sha((PACK/'MANIFEST.sha256.json').read_bytes()))
source=git(b['source_commit'],b['source_git_path'])
check('original adopted bytes',sha(source)==b['source_sha256']==b['base_sha256']=='6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3')
oldraw=git(b['source_provenance']['raw_launch_commit'],b['source_provenance']['raw_launch'])
check('actual prior launch pin',sha(oldraw)==b['source_provenance']['raw_launch_sha256'])
old=json.loads(oldraw)
check('exact genuine control science',old['command']==b['training_command']==p['argv']==shlex.split((PACK/'package/launch-command.txt').read_text()))
check('prospective formal role only',b['kind']==b['planned_record']['kind']=='control' and b['planned_record']['status']=='PENDING_NOT_RUN' and b['existing_trn22_schedule_arm']['kind']=='schedule_arm')
check('fixed science flags',b['training_command'][b['training_command'].index('--matrix-lr')+1]=='0.015' and b['training_command'][b['training_command'].index('--max-train-seconds')+1]=='1800' and '--recursive-schedule' in b['training_command'])
check('no experiment additions',not any(x in b['training_command'] for x in ['--initial-cache-refresh','--unet-8x1024','--startup-adjusted-schedule','--fp8-proj']))
for k in ['NEURON_CC_FLAGS','NEURON_LOGICAL_NC_CONFIG','NEURON_COMPETITION_R1_SEED','NKI_VENV_PATH','PATH','PYTHONDONTWRITEBYTECODE','PYTHONUNBUFFERED','NEURON_COMPETITION_R1_CACHE_DIR']:
    check('science environment '+k,old['environment'][k]==b['training_environment'][k])
old_driver=git(b['derivation']['parent_commit'],b['derivation']['parent_driver_path'])
check('reviewed parent exact',sha(old_driver)==b['derivation']['parent_driver_sha256'])
new_tree,old_tree=ast.parse((PACK/'host_driver.py').read_text()),ast.parse(old_driver)
check('all functional AST unchanged',body(new_tree)==body(old_tree))
newc,oldc=constants(new_tree),constants(old_tree)
allowed={'ROOT','SOURCE_SHA','LABEL','TRAIN_COMMAND','EVAL_COMMAND','TRAIN_ENV','EVAL_ENV'}
check('no extra changed constants',{k for k in newc if newc[k]!=oldc[k]}==allowed)
for k in sorted(allowed):
    expected=json.dumps(oldc[k]).replace('fs-20260926-arch36a8-startup-clock-trn22-r1-full',b['label']).replace('36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb',b['source_sha256'])
    check('only label and source mapping '+k,newc[k]==json.loads(expected))
for name,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
    check('unchanged checked '+name,sha((REPO/'research/v2/operator/20260926-common-host-idle-guard'/name).read_bytes())==b[key])
for key in ['stage_authorized','launch_authorized','source_adoption','submission_authorized','host_reserved']: check('disabled '+key,b[key] is False and p[key] is False)
check('original host lock',b['hostname']=='ip-172-31-32-253' and b['lock_identity']['expected_inode']==832589 and b['lock_identity']['expected_device']==66305)
check('normal isolated driver preserves assertions',b['outer_command']==['/usr/bin/flock','-n',b['device_lock'],'/usr/bin/python3','-I','-B',b['driver_path']])
check('no assertion suppression env',all('PYTHONOPTIMIZE' not in b[k] for k in ['training_environment','evaluation_environment']))
check('actual budget and watchdog',b['budgets']['source_seconds']==1800 and b['budgets']['startup_cap_seconds']==900 and b['command'][4]=='2730s' and b['evaluation_command'][4]=='1800s' and all('--kill-after=60s' in b[k] for k in ['command','evaluation_command']))
check('attached fresh20M',b['evaluation']=={'fresh_process':True,'tokens':20971520,'seq_len':1024,'batch_size':1,'device':'neuron','prepare_unchanged':True,'same_lifetime_lock':True,'new_guard_before_eval':True})
for env in ['training_environment','evaluation_environment']:
    check('cache URL matches '+env,b[env]['NEURON_COMPILE_CACHE_URL']==b[env]['NEURON_COMPILE_CACHE_DIR'])
check('fresh reuse disclosed',not any('--fresh-caches' in b[k] for k in ['command','evaluation_command']) and 'not a true-cold' in b['cache_policy'].lower())
check('source/protected harness pins',sha((REPO/'prepare.py').read_bytes())==b['prepare_sha256'] and sha(git('00f2ca7f79696e65ce52458f074dd13b871da90c','research/scratch/submission_rehearsal.py'))==b['harness_sha256'])
result=dict(status='PASS_UNCHANGED_CONTROL_MECHANICS_ONLY',checks=checks,check_count=len(checks),source_sha256=b['source_sha256'],packet='9889c4e07de37fd4c0c84c2efc4a014e9ff82b26',host='trn22',no_host_action=True,launch_authorized=False,submission_authorized=False)
print(json.dumps(result,indent=2))
if len(sys.argv)==3 and sys.argv[1]=='--output': Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
