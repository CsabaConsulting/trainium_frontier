"""Own stdlib raw-result/control/selector cross-check; does not execute supplied audit."""
import argparse, hashlib, importlib.util, json, math, re, subprocess
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);args=ap.parse_args()
R=args.repo; checks=[]; hashes={}
def ck(name,ok):
    if not ok: raise RuntimeError(name)
    checks.append(name)
def read(path):
    b=(R/path).read_bytes(); hashes[path]=hashlib.sha256(b).hexdigest(); return b
def obj(path): return json.loads(read(path))
def git(ref,path): return subprocess.check_output(['git','show',ref+':'+path],cwd=R)
def metric(text):
    matches=[]
    for m in re.finditer(r'\{',text):
        try: x,_=json.JSONDecoder().raw_decode(text[m.start():])
        except ValueError: continue
        if isinstance(x,dict) and {'num_bytes','total_nats','val_bpb'} <= x.keys(): matches.append(x)
    ck('one complete metric',len(matches)==1)
    x=matches[0]; value=x['total_nats']/(x['num_bytes']*math.log(2))
    ck('independent BPB arithmetic',value==x['val_bpb'] and math.isfinite(value))
    ck('actual20971520 bytes83462186',x['num_tokens']==20971520 and x['num_bytes']==83462186)
    ck('finite timed eval',x['eval_over_timeout'] is False and 0<x['eval_seconds']<x['eval_timeout']==1200)
    return x
C='research/v2/operator/20260925-trn21-l3-base-repeat-r2/'
A='research/v2/operator/20260926-trn21-architecture-recursive-full/completed/'
V='research/v2/verification/20260926-architecture-recursive-full-result/'
before={p:read(p) for p in ['research/v2/ledger.jsonl','workflow/tools/decide.py']}
control_obs=obj(C+'completed-eval/observation.json')
for name,item in control_obs['files'].items():
    path=C+'completed-eval/'+name;b=read(path)
    ck('control retrieval '+name,len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'])
    ck('control immutable Git '+name,b==git('e78c9e7a68639b14569d60048044682b4e9710fc',path))
control=metric(read(C+'completed-eval/fresh-public20m.log').decode())
ct=obj(C+'completed-eval/terminal.json');cl=obj(C+'completed-training/training-launch.json')
cp=obj(C+'launched/preflight.json');ce=obj(C+'completed-eval/fresh-public20m-launch.json')
prop=obj(C+'EXPERIMENT_PROPOSAL.priority.json')
ck('prospective same-config control',prop['mechanism_class']=='same_config_control_repeat' and ct['kind']=='control')
ck('prospective proposal pinned in raw terminal',hashes[C+'EXPERIMENT_PROPOSAL.priority.json']==ct['proposal_sha256'])
ck('source and prepare exact',ct['train_py_sha256']==cp['train_sha256']=='6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3' and ct['prepare_sha256']==cp['prepare_sha256']=='5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb')
ck('Git source control',hashlib.sha256(git('a4aa4351789115a2d0aa947d9ca00ff689f9bd0b','workflow/proposals/research-v2/train.py')).hexdigest()==ct['train_py_sha256'])
ck('control argv seed and budget',cl['command'][4:]==ct['argv'] and ct['argv'][ct['argv'].index('--seed')+1]=='42' and ct['argv'][ct['argv'].index('--max-train-seconds')+1]=='1800')
ck('control budget summary',ct['metrics']['over_budget'] is False and ct['metrics']['charged_seconds_est']==1785.1<1800)
trlog=read(C+'completed-training/training.log').decode()
ck('raw control budget',bool(re.search(r'over_budget:\s+False',trlog)) and bool(re.search(r'charged_seconds_est:\s+1785.1',trlog)))
ck('control eval exact scope',ce['command'][-10:]==['--device','neuron','--batch-size','1','--seq-len','1024','--eval-tokens','20971520','--timeout-seconds','1200'])
train_exit=obj(C+'completed-eval/training-exit.json');eval_exit=obj(C+'completed-eval/fresh-public20m-exit.json')
ck('fresh control eval after train',train_exit['exit_code']==eval_exit['exit_code']==0 and train_exit['utc']<ce['utc']<eval_exit['utc'])
ck('control metric matches terminal',ct['evaluation']==control and ct['public_20m']==control['val_bpb']==0.9899528600562257)
bundle_raw=read(A+'results-bundle.json')
ck('candidate immutable raw',bundle_raw==git('f7db082124074a070606c93d7d729d2bdcc69822',A+'results-bundle.json'))
bundle=json.loads(bundle_raw);files=bundle['files']
for name,item in files.items():
    b=item['text'].encode();ck('candidate artifact '+name,len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'])
    hashes[A+'bundle::'+name]=hashlib.sha256(b).hexdigest()
j=lambda name:json.loads(files[name]['text'])
binding=j('NATIVE_BINDING.disabled.json');candidate=metric(files['fresh-public20m.log']['text'])
ck('candidate exact score',candidate['val_bpb']==0.9896621160468637 and candidate['total_nats']==57253516)
ck('candidate source Git',hashlib.sha256(git(binding['source_commit'],binding['source_git_path'])).hexdigest()==binding['source_sha256']=='75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f')
tags=lambda prefix:[json.loads(l[len(prefix):]) for l in files['fresh-public20m.log']['text'].splitlines() if l.startswith(prefix)]
ck('candidate real fresh B1 eval',len(tags('REHEARSAL_COMMAND '))==1 and tags('REHEARSAL_COMMAND ')[0][-10:]==ce['command'][-10:])
ck('candidate isolated source unchanged',tags('ISOLATED_SUBMISSION ')[0]['train_sha256']==binding['source_sha256'] and tags('ISOLATED_SUBMISSION ')[0]['prepare_sha256']==ct['prepare_sha256'])
train=j('train/rehearsal.json');ev=j('eval/rehearsal.json')
ck('candidate exact through-exit budget',train['returncode']==0 and train['charged_through_process_exit_within_budget'] is True and 0<=train['charged_through_process_exit']<=1800)
ck('candidate eval all exits zero',ev['returncode']==j('terminal.json')['eval_exit_code']==j('outer-flock-exit.json')['returncode']==0 and files['eval-exit-code.txt']['text'].strip()=='0')
postck=obj(A+'post-eval-checkpoint.json')
ck('retained candidate checkpoint equality',postck['unchanged_after_eval'] is True and postck['sha256']==train['checkpoint_sha256']==postck['expected_training_sha256'])
rows=[json.loads(l) for l in before['research/v2/ledger.jsonl'].splitlines() if l.strip()]
r2=next(r for r in rows if r['label']==ct['label'])
ck('ledger genuine control same host/base',r2['kind']=='control' and r2['host']=='trn21' and r2['base_sha256']==r2['train_py_sha256']==ct['train_py_sha256'] and r2['public_20m']==control['val_bpb'])
spec=importlib.util.spec_from_file_location('decide_audit',R/'workflow/tools/decide.py');sel=importlib.util.module_from_spec(spec);spec.loader.exec_module(sel)
ck('selector pinned',hashlib.sha256(before['workflow/tools/decide.py']).hexdigest()=='fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac')
ck('control valid',sel.invalid_reason(r2) is None)
own=dict(label=binding['label'],arm_id='L2-architecture-recursive-stack-75b944',kind='architecture_schedule_stack_arm',host='trn21',base_sha256=ct['train_py_sha256'],timestamp_utc=j('terminal.json')['utc'],status='evaluated',over_budget=False,training_budget_seconds=1800,charged_through_exit_seconds=train['charged_through_process_exit'],argv=binding['training_command'][4:],env=binding['training_environment'],causality_passed=None,git_dirty=False,eval_tokens=20971520,eval_seq_len=1024,public_20m=candidate['val_bpb'],run_purpose='quality')
ctrls=sel.controls_for(rows,'trn21',own['base_sha256'])
own.update(steps=train['step_boundaries'], startup_measured_seconds=train['observations'][4]['observed_seconds'], startup_allowance_seconds=train['startup_allowance_observed'], process_through_exit_wall_seconds=train['wall_seconds'])
ck('only genuine r2 formal control',len(ctrls)==1 and ctrls[0]['label']==r2['label'])
delta=candidate['val_bpb']-control['val_bpb']
ck('manual selector branch',-.0006<delta<.0005 and own['kind'] not in {'table_arm','data_order_arm'} and delta>-.010 and candidate['val_bpb']>=.975)
verdict=sel.decide_arm([r for r in rows if r['label']!=own['label']]+[own],own['arm_id'])
ck('own row selector DISCARD',sel.invalid_reason(own) is None and verdict['verdict']=='DISCARD' and verdict['why']=='ambiguous band, low-prior kind' and verdict['per_host'][0]['controls']==[r2['label']])
generated=obj(V+'final-audit.json')
ck('generated results match own totals',generated['public20m']==candidate['val_bpb'] and generated['delta_vs_formal_control']==delta and generated['selector_result']['verdict']==verdict['verdict'])
read(V+'audit_final.py');read(V+'final-audit-optimized.json');read(V+'ledger-row.json')
for path,b in before.items(): ck('unchanged '+path,(R/path).read_bytes()==b)
print(json.dumps(dict(status='RAW_ARITHMETIC_AND_FORMAL_CONTROL_DISCARD_CONFIRMED',checks=len(checks),check_names=checks,hashes=hashes,candidate=candidate,control=control,delta=delta,selector=verdict,limitations=['This auditor previously implemented75b9; this is an independent calculation from retrieved bytes, not a fresh independent source review.','No host/checkpoint data access, native evaluation, new causality or cold-submission proof.','Control charged-through-exit/startup remain unknown; source charged estimate1785.1 and authenticated1800 declaration accepted by unchanged selector.','Different historical evaluation env/PYTHONPATH contexts are disclosed; this audit establishes prescribed selector eligibility, not deterministic or perfectly matched host trajectory.']),indent=2))
