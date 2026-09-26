"""Read-only pinned-artifact/clock analysis; no host, SDK, tensors or approval."""
import ast,hashlib,json,shlex,subprocess
from pathlib import Path
ROOT=Path('/home/poojith-devan/Desktop/Trainium')
PIN='42192fd0'
def blob(path,rev=PIN):return subprocess.check_output(['git','show',rev+':'+path],cwd=ROOT)
def sha(x):return hashlib.sha256(x).hexdigest()
def need(v,label):
    if not v:raise RuntimeError(label)
rawpath='research/v2/operator/20260926-trn21-lr-train-eval-boundary-0326/trn21.json'
raw=blob(rawpath);need(raw==blob(rawpath,'7d58fb48'),'OPERATOR_CANONICAL_BYTES')
need(sha(raw)=='fc48525fee3d026bd2fdda02daeb5ca8d1ef78369d321158711b978b30d67ccf','RAW_PIN')
remote=json.loads(raw)['remote'];anchor=remote['completed_parent_metadata_only'];caches=remote['completed_parent_cache_exists_only']
need(len(anchor)==2 and len(caches)==6 and all(caches.values()),'ANCHOR_COUNTS')
train=next(k for k in anchor if k.endswith('/train.py'));prep=train.rsplit('/',1)[0]+'/prepare.py'
need(anchor[train]['mode']==0o100444 and anchor[prep]['mode']==0o100600,'MODES')
need(all(v['stable_during_read'] for v in anchor.values()),'OBSERVATION_STABLE')
parent='research/v2/proposals/20260926-startup-adjusted-full-trn21-replication/'
binding=json.loads(blob(parent+'NATIVE_BINDING.disabled.json'))
old=shlex.split(blob(parent+'package/launch-command.txt').decode());new=old.copy();new[3]=train
need(old==binding['training_command'],'PARENT_ARGV')
need([i for i,(a,b) in enumerate(zip(old,new)) if a!=b]==[3],'ONLY_SCRIPT_PATH_DIFF')
need(old[old.index('--matrix-lr')+1]=='0.015' and old[old.index('--num-steps')+1]=='1000000','FULL_CONTROL')
source=blob('workflow/proposals/research-v2/train.py','7419927c7417b5666a984ec9c2b849feadabaaa9')
need(sha(source)==anchor[train]['sha256'],'SOURCE')
nodes=ast.parse(source).body
fn=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='startup_adjusted_schedule_elapsed')
ns=dict(TRAIN_STARTUP_STEPS_EXCLUDED=5,TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900.,STARTUP_LAUNCH_MARGIN_SECONDS=30.,_PROCESS_STARTED=100.)
exec(compile(ast.Module([fn],type_ignores=[]),'pinned-clock','exec'),ns)
cases=[]
for step in (0,4,5,6):
    for age in (120.,180.,845.,870.,871.,900.,1000.):
        elapsed=10.;start=100.+age-elapsed
        got=ns[fn.name](step,elapsed,start)
        expected=elapsed if step>=5 else max(0.,age+30.-900.)
        need(got==expected,'CLOCK')
        cases.append(dict(step=step,process_age=age,schedule_elapsed=got))
helper=blob('research/v2/proposals/20260926-stablepath-warm-observation-trn22/identity_checks.py')
need(b"stable/'prepare.py', plan['prepare_sha256'], True" in helper,'W1_READONLY_PREPARE')
harness=blob('research/scratch/submission_rehearsal.py','00f2ca7f79696e65ce52458f074dd13b871da90c')
need(sha(harness)=='7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b','HARNESS')
print(json.dumps(dict(status='DESIGN_EVIDENCE_ONLY_NOT_GATE_PASS',raw_sha256=sha(raw),anchor=anchor,cache_exists_only=caches,training_argv=new,original_argv_sha256=sha(blob(parent+'package/launch-command.txt')),changed_argument_indices=[3],actual_pinned_clock_cases=cases,unchanged_harness_sha256=sha(harness),w1_inputs_not_applicable_to_mode0600=True,source_gate_inherited_not_reapproved=True,no_host=True),indent=2))
