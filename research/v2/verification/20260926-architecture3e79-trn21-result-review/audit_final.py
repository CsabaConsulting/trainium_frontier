#!/usr/bin/env python3
"""Authenticate failed architecture confirmation; query unchanged selector only."""
import contextlib
import copy
import datetime
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import runpy
import shlex
import subprocess

REPO = Path('/home/poojith-devan/Desktop/Trainium')
HERE = Path(__file__).resolve().parent
ROOT = 'research/v2/operator/20260926-trn21-arch3e79-replicate'
RAW = '03a5fbf3951ad81c43a9b83e23a24c20d4e99f2c'
ARM = 'L2-unet8x1024-3e79-primed'
SELECTOR = 'fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac'
checks = []
def check(ok, name):
    if not ok: raise RuntimeError(name)
    checks.append(name)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def git(path, ref=RAW):
    return subprocess.check_output(['git','show',ref+':'+path], cwd=REPO)
def raw(name): return git(ROOT+'/completed-eval/'+name)
def read(name): return json.loads(raw(name))
def tagged(log, prefix):
    values = [json.loads(l[len(prefix):]) for l in log.splitlines() if l.startswith(prefix)]
    check(len(values)==1, 'one '+prefix)
    return values[0]
def parse_eval(log):
    for m in re.finditer(r'\{',log):
        try: v,_=json.JSONDecoder().raw_decode(log[m.start():])
        except ValueError: continue
        if isinstance(v,dict) and 'total_nats' in v and 'val_bpb' in v: return v
    raise RuntimeError('missing raw eval totals')

capture=io.StringIO()
with contextlib.redirect_stdout(capture): runpy.run_path(str(HERE/'audit_training.py'),run_name='__main__')
training=json.loads(capture.getvalue())
check(training['checks']==73 and training['status']=='PASS_TRAINING_AUTHENTICATION_ONLY', 'rerun73 training/source/guard/accounting checks')
binding=json.loads(git(ROOT+'/NATIVE_BINDING.disabled.json','af53137f'))
retrieval=read('retrieval.json')
for name, expected in retrieval['metadata']['files'].items():
    data=raw(name)
    check(sha(data)==expected['sha256'] and len(data)==expected['bytes'], 'independent final hash '+name)
    check(data==(REPO/ROOT/'completed-eval'/name).read_bytes(), 'canonical identity '+name)
check(len(retrieval['retrieved'])==len(retrieval['metadata']['files'])==10, 'complete ten-file transport')
check(all(x['returncode']==0 and x['hash_verified'] is True and x['sha256']==sha(raw(x['path'])) for x in retrieval['retrieved']), 'transport receipts agree')
for name in ('checkpoint.json','eval-under-lock-idle.json','fresh-public20m-launch.json'):
    check(raw(name)==git(ROOT+'/completed-training/'+name,'f4726fd3'), 'unchanged handoff '+name)
log=raw('fresh-public20m.log').decode()
isolation=tagged(log,'ISOLATED_SUBMISSION ')
command=tagged(log,'REHEARSAL_COMMAND ')
check(isolation['mode']=='eval' and isolation['sandbox']==binding['eval_sandbox']
      and isolation['files']==['prepare.py','train.py'] and isolation['train_sha256']==binding['source_sha256']
      and isolation['prepare_sha256']==binding['prepare_sha256'], 'exact minimal fresh-process eval sandbox')
check(shlex.split(isolation['launch_command'])==binding['training_command'], 'same trained science argv')
expected=[binding['evaluation_command'][5],'prepare.py','eval-public','--train-py','train.py','--checkpoint',binding['sandbox']+'/out/final.pt','--device','neuron','--batch-size','1','--seq-len','1024','--eval-tokens','20971520','--timeout-seconds','1200']
check(command==expected, 'exact fresh public20M context1024 batch1 argv')
evaluation=parse_eval(log)
check(evaluation['num_tokens']==20971520 and evaluation['num_bytes']==83462186 and evaluation['total_nats']==57454092.0, 'complete raw20M totals')
bpb=evaluation['total_nats']/(evaluation['num_bytes']*math.log(2))
check(math.isfinite(bpb) and bpb==evaluation['val_bpb']==0.9931291951444726, 'own BPB arithmetic')
check(evaluation['eval_over_timeout'] is False and evaluation['eval_seconds']<evaluation['eval_timeout']==1200, 'eval timeout respected')
receipt=read('eval/rehearsal.json')
check(tagged(log,'SUBMISSION_REHEARSAL ')==receipt and receipt['returncode']==0, 'fresh observer exit0')
terminal,outer=read('terminal.json'),read('outer-flock-exit.json')
check(terminal['label']==binding['label'] and terminal['eval_exit_code']==outer['returncode']==0 and raw('eval-exit-code.txt').strip()==b'0', 'complete zero exits')
check(terminal['no_automatic_next_job'] is True and terminal['submission_authorized'] is False, 'no auto-adoption/upload')
check(outer['command'][:3]==['/usr/bin/flock','-n',binding['device_lock']] and outer['command'][-1]==binding['root']+'/host_driver.py', 'original lifetime flock')
parse=datetime.datetime.fromisoformat
start=parse(re.search(r'^FRESH-PUBLIC20M_START_UTC=(.+)$',log,re.M).group(1))
end=parse(re.search(r'^FRESH-PUBLIC20M_END_UTC=(.+)$',log,re.M).group(1))
check(parse(training['training_end_utc'])<parse(training['fresh20M_launch_utc'])<=start<end<=parse(terminal['utc'])<parse(outer['ended_utc']), 'train/eval/outer chronology')
check(evaluation['eval_seconds']<receipt['wall_seconds']<=(end-start).total_seconds(), 'eval and complete-process durations distinguished')
snapshot=read('terminal-host-snapshot.json'); host=snapshot['data']
check(snapshot['rc']==host['ps_rc']==host['neuron_ls_rc']==host['lock_competitor_rc']==0 and not host['job_lines']
      and not host['process_details'] and not host['compiler_capture'] and not host['old_queue_command_lines']
      and not any(p['exists'] for p in host['old_queue_pid_checks']), 'actual post-run job/oldqueue absence and free lock')
check(all(not d['neuron_processes'] for d in json.loads(host['neuron_ls'])) and [host['lock_device'],host['lock_inode']]==[66305,596823], 'device clear and original inode')
check(parse(outer['ended_utc'])<parse(host['utc']) and host['logs']['fresh-public20m.log']['sha256']==sha(raw('fresh-public20m.log')), 'second observation agrees after exit')
checkpoint=read('post-eval-checkpoint.json')
check(checkpoint['post_eval_rehash'] is True and checkpoint['bytes']==training['checkpoint']['bytes']==453048594
      and checkpoint['sha256']==training['checkpoint']['sha256'] and parse(checkpoint['utc'])>parse(outer['ended_utc'])
      and checkpoint['checkpoint_path']==binding['sandbox']+'/out/final.pt', 'actual post-eval retained checkpoint rehash')

selector_path=REPO/'workflow/tools/decide.py'; ledger_path=REPO/'research/v2/ledger.jsonl'
selector_bytes=selector_path.read_bytes(); ledger_bytes=ledger_path.read_bytes()
check(sha(selector_bytes)==SELECTOR, 'selector unmodified')
spec=importlib.util.spec_from_file_location('unchanged_selector',selector_path); decide=importlib.util.module_from_spec(spec); spec.loader.exec_module(decide)
rows=decide.load(str(ledger_path))
old=[r for r in rows if r.get('label')=='fs-20260926-arch3e79-primed-trn22-r2-full']
check(len(old)==1 and old[0]['public_20m']==.9863403052402583, 'original immutable architecture row preserved')
controls=decide.controls_for(rows,'trn21',old[0]['base_sha256'])
check(len(controls)==1 and controls[0]['label']==binding['same_host_control']['label'], 'actual same-host c5c control')
control=controls[0]; control_raw=(REPO/control['eval_log']).read_bytes(); ce=parse_eval(control_raw.decode())
check(sha(control_raw)==control['eval_log_sha256'] and ce['total_nats']/(ce['num_bytes']*math.log(2))==control['public_20m']==.9914739939934344, 'control raw arithmetic/hash')
delta=bpb-control['public_20m']
candidate=copy.deepcopy(old[0])
for k in ('verdict','selector_verdict','control_public_20m','delta_public_20m','independent_review','supervisor_decision'):
    candidate.pop(k,None)
candidate.update(label=binding['label'],host='trn21',timestamp_utc=terminal['utc'],source_commit=binding['source_commit'],
    argv=binding['training_command'][4:],env=binding['training_environment'],steps=training['steps'],
    training_seconds=training['source_estimates_separate']['training_seconds'],charged_seconds_est=training['source_estimates_separate']['charged_seconds_est'],
    charged_through_exit_seconds=training['charged_through_exit_seconds'],process_through_exit_wall_seconds=training['process_wall_seconds'],
    startup_measured_seconds=training['uncapped_fifth_seconds'],startup_allowance_seconds=training['startup_allowance_seconds'],
    startup_allowance_est=training['source_estimates_separate']['startup_allowance_est'],
    cold_scope='Existing same-host research cache roots; not a cold full replay.',public_20m=bpb,eval_seconds=evaluation['eval_seconds'],
    checkpoint_sha256=checkpoint['sha256'],checkpoint_bytes=checkpoint['bytes'],checkpoint_path=checkpoint['checkpoint_path'],
    checkpoint_evidence_scope='Retained on host and rehashed after eval at '+checkpoint['utc'],
    eval_log=ROOT+'/completed-eval/fresh-public20m.log',eval_log_sha256=sha(raw('fresh-public20m.log')),
    training_log=ROOT+'/completed-training/training.log',training_log_sha256=training['retrieved_artifact_sha256']['training.log'],
    training_archive_commit=training['raw_commit'],final_archive_commit=RAW,submission_eligible=False)
check(decide.invalid_reason(candidate) is None, 'valid fixed-budget research result')
existing=[r for r in rows if r['label']==candidate['label']]
check(len(existing)<=1 and (not existing or existing[0]['public_20m']==bpb), 'no conflicting or duplicate observation')
query=[r for r in rows if r['label']!=candidate['label']]+[candidate]
verdict=decide.decide_arm(query,ARM)
check(verdict['verdict']=='DISCARD', 'unchanged two-host selector DISCARD')
check(delta>0 and old[0]['delta_public_20m']<0, 'mixed-sign confirmation failure')
check(ledger_path.read_bytes()==ledger_bytes and selector_path.read_bytes()==selector_bytes, 'no ledger/selector mutation')
print(json.dumps(dict(status='PASS_RAW_RESULT_AUTHENTICATION_QUALITY_NOT_CONFIRMED',training_checks=73,final_checks=len(checks),
    check_names=checks,evaluation=evaluation,candidate_query_only=candidate,selector_result=verdict,
    delta_vs_same_host_control=delta,control_steps=control['steps'],candidate_steps=training['steps'],
    original_public_20m=old[0]['public_20m'],pair_mean=(bpb+old[0]['public_20m'])/2,
    pair_mean_delta=(delta+old[0]['delta_public_20m'])/2,public_spread=bpb-old[0]['public_20m'],
    startup_lr_confounder_proven_cause_of_spread=False,adoption=False,submission_eligible=False,
    selector_sha256=SELECTOR,raw_commit=RAW),indent=2))
