"""Stdlib/Git mechanical pin checks only; never imports scientific code or drivers."""
import ast, hashlib, json, shlex, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
checks=[]
def need(ok,name):
    if not ok: raise RuntimeError(name)
    checks.append(name)
def sha(x):return hashlib.sha256(x).hexdigest()
def blob(c,p):return subprocess.check_output(['git','show',c+':'+p],cwd=REPO)
b=json.loads((HERE/'CT_BINDING.disabled.json').read_bytes())
q=(HERE/'EXPERIMENT_PROPOSAL.json').read_bytes();p=json.loads(q)
d=(HERE/'host_driver.py').read_bytes();tree=ast.parse(d)
need(q==blob(b['proposal_first_commit'],str((HERE/'EXPERIMENT_PROPOSAL.json').relative_to(REPO))), 'exact Supervisor proposal')
need(sha(q)==b['proposal_sha256'], 'proposal SHA')
need(sha(blob(p['source_commit'],p['source_git_path']))==p['source_sha256']==b['source_sha256'], 'unchanged clean source blob')
drv=b['driver_derivation'];parent=blob(drv['commit'],drv['path'])
need(sha(parent)==drv['sha256'], 'proven driver parent SHA')
expected=parent.decode()
for old,new in drv['replacements']:
    need(old in expected,'mechanical literal exists '+old)
    expected=expected.replace(old,new)
need(expected.encode()==d, 'complete driver literal-only transform')
need(sha(d)==b['driver_sha256'], 'new exact driver SHA')
constants={}
for n in tree.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
        v=n.value.args[0] if isinstance(n.value,ast.Call) else n.value
        constants[n.targets[0].id]=ast.literal_eval(v)
for field,key in [('ROOT','root'),('LABEL','label'),('SOURCE_SHA','source_sha256'),('PREP','prepare_path'),('PREP_SHA','prepare_sha256'),('HARNESS','harness_path'),('HARNESS_SHA','harness_sha256'),('COMMAND','command'),('ENV','training_environment')]:
    need(constants[field]==b[key], 'driver binding '+field)
sc=blob(p['source_commit'],'workflow/candidates/confirmed-clock36-116188/launch-command.txt')
need(sha(sc)==p['submission_command_sha256']==b['submission_command_sha256'], 'exact proven submission command pin')
argv=shlex.split(sc.decode())
need(argv[0]=='NEURON_LOGICAL_NC_CONFIG=2' and argv[1]=='torchrun', 'submission launch environment')
base=[p['argv'][0]]+argv[2:]
need(b['base_training_command']==base, 'absolute binary only normalized submission argv')
need(b['training_command']==p['argv']==base+['--num-steps','5'], 'sole diagnostic override five steps')
for flag in ['--recursive-schedule','--startup-adjusted-schedule','--initial-cache-refresh','--unet-8x1024','--attention-layers','--matrix-lr','--max-train-seconds','--seed']:
    need(flag not in b['training_command'], 'no extra research flag '+flag)
need(shlex.split((HERE/'package/launch-command.txt').read_text())==p['argv'], 'package exact argv')
for f,h in b['package_files'].items():need(sha((HERE/f).read_bytes())==h, 'package pin '+f)
prov=json.loads((HERE/'package/provenance.json').read_bytes())
need(prov['train_sha256']==p['source_sha256'] and prov['source_commit']==p['source_commit'], 'source provenance')
need(b['host']=='trn21' and b['expected_hostname']=='ip-172-31-34-85', 'exact host')
need(b['root']==p['root'] and b['sandbox']==p['root']+'/train', 'private unique root')
need(b['original_lock']['device']==66305 and b['original_lock']['inode']==596823, 'original lock inode')
need(b['device_lock']==b['original_lock']['path']=='/home/ubuntu/.cache/trainium_freshstart/device0.lock', 'single common lock')
need(b['command'][:5]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','1800s'], 'bounded old timeout')
need(b['command'][-3:]==['--mode','train','--fresh-caches'], 'fresh-cache train-only harness')
need(sha(blob(b['harness_source_commit'],b['harness_source_git_path']))==b['harness_sha256']==p['harness_sha256'], 'unchanged real harness')
for field,path in [('idle_guard_sha256',b['idle_guard_source']),('dispatch_driver_sha256','research/v2/operator/20260926-common-host-idle-guard/dispatch_driver.py')]:
    need(sha((REPO/path).read_bytes())==b[field]==p[field], 'unchanged shared '+field)
need(b['prepare_sha256']==p['prepare_sha256'], 'prepare pin not modified')
need(b['training_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS']=='1800' and b['training_environment']['NEURON_COMPETITION_R1_SEED']=='42' and b['training_environment']['NEURON_LOGICAL_NC_CONFIG']=='2', 'unchanged budget seed LNC environment')
keys=['TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR']
need(set(b['harness_injected_environment'][k] for k in keys)=={b['sandbox']+'/caches/'+n for n in ['neff','local','hlo','inductor','compiler','debug']}, 'six new private caches')
need(b['training_environment']['NEURON_COMPILE_CACHE_URL']==b['harness_injected_environment']['NEURON_COMPILE_CACHE_DIR'], 'cache URL equals private DIR')
for key in keys:need(key not in b['training_environment'], 'harness owns fresh cache '+key)
need(b['cold_decision']['maximum_seconds']==900 and b['cold_decision']['target_seconds']==870, 'hard startup cap unchanged')
for k in ['stage_authorized','launch_authorized','native_launch_authorized','source_adoption','full_run_automatic','host_reserved']:
    need(b[k] is False, 'disabled '+k)
need(b['no_eval'] and b['no_automatic_full'] and b['cold_decision']['automatic_full'] is False, 'no eval or follow-on')
need(b['source_gate']==p['gates'] and p['gates']['fullsize_cpu'].startswith('PENDING') and p['gates']['ordinary_loader_trained_checkpoint'].startswith('PENDING'), 'actual fullCPU and ordinary loader gates still pending')
need(b['resource_floors']['operator_preflight_disk_bytes']==int(10.5*1024**3) and b['resource_floors']['unchanged_inner_driver_disk_bytes']==int(6.5*1024**3), 'higher Operator floor plus headroom not waived')
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
guards=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='require_idle_under_inherited_lock']
launches=[n for n in calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess' and n.func.attr=='call']
need(len(guards)==len(launches)==1 and guards[0].lineno<launches[0].lineno, 'one guarded child only')
fd=next(k.value for k in launches[0].keywords if k.arg=='pass_fds')
need(ast.dump(fd)=="Tuple(elts=[Name(id='lock_fd', ctx=Load())], ctx=Load())", 'lock fd inherited by child')
need(b['outer_command'][:6]==['/usr/bin/flock','-n',b['device_lock'],'/usr/bin/python3','-I','-B'], 'normal-Python lifetime flock')
need(b['detached_command']==['/usr/bin/python3','-I','-B',b['root']+'/dispatch_driver.py'], 'unchanged sole dispatcher')
for key in ['command','training_command','outer_command','detached_command']:
    need(sha(json.dumps(b[key],separators=(',',':'),ensure_ascii=True).encode())==b[key+'_sha256'], 'command pin '+key)
need('PPID/PGID/SID' in b['postflight'] and 'HOLD' in b['postflight'], 'explicit every-outcome postflight')
print(json.dumps({'status':'LOCAL_MECHANICS_PASS_NOT_INDEPENDENT_REVIEW','checks':len(checks),'names':checks,'host_calls':False,'tensors':False,'source_execution':False,'stage_authorized':False,'launch_authorized':False},indent=2))
