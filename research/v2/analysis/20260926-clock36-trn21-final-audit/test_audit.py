"""Bounded readiness tests; prior trn22 result is a parser fixture, not trn21 evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import shlex

spec=importlib.util.spec_from_file_location('replica_audit',Path(__file__).with_name('audit.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
cases=[]
def reject(name,fn):
    try:fn()
    except (ValueError,KeyError,TypeError,IndexError):cases.append(dict(name=name,outcome='EXPECTED_REFUSAL'))
    else:raise RuntimeError('NEGATIVE_ACCEPTED:'+name)
p='research/v2/operator/20260926-trn22-startup-clock-full/completed/'
raw=a.git('110054fe',p+'results-bundle.json')
bundle,f=a.decode(raw,'2557fbc01b642425201eb077fe84a1967ac4b518b52a9f9bf6ac1c0f8be2cb99')
b=a.parse(f['NATIVE_BINDING.disabled.json']);post=a.parse(a.git('110054fe',p+'explicit-process-postflight.json'));ck=a.parse(a.git('110054fe',p+'post-eval-checkpoint-hash.json'))
def call(files=f,po=post,ch=ck,bi=b):return a.closed_result(bundle,files,bi,po,ch,832589,set(post['observed_pids']))
facts=call();a.require(facts['evaluation']['val_bpb']==0.9883821509811542,'prior trn22 positive parser fixture only')
cases.append(dict(name='closed archived trn22 full36 parser fixture',outcome='FIXTURE_ONLY_PASS'))
reject('wrong outer hash',lambda:a.decode(raw,'0'*64))
bad=a.parse(raw);bad['files']['terminal.json']['bytes']+=1;enc=json.dumps(bad).encode()
reject('damaged inner artifact size',lambda:a.decode(enc,a.sha(enc)))
reject('nonfinite JSON',lambda:a.parse('{"n":NaN}'))
reject('duplicate JSON keys',lambda:a.parse('{"n":1,"n":2}'))
reject('over-budget through exit',lambda:a.timing(dict(a.parse(f['train/rehearsal.json']),charged_through_process_exit=1801)))
bad=copy.deepcopy(ck);bad['sha256']='0'*64;reject('posteval checkpoint changed',lambda:call(ch=bad))
bad=copy.deepcopy(post);bad['pid_rows'][0]['exists']=True;reject('known process survives',lambda:call(po=bad))
bad=copy.deepcopy(post);bad['all_process_identity_rows'][0]['sid']=post['observed_pids'][0];reject('related SID survives',lambda:call(po=bad))
bad=dict(f);bad.pop('metrics-training/metric_table_graph_stats_rank3.csv');reject('fourth graph receipt absent',lambda:call(files=bad))
reject('missing complete eval totals',lambda:a.evaluation('still running'))
reject('wrong eval token count',lambda:a.evaluation(f['fresh-public20m.log'].decode().replace('20971520','20971519')))
binding,prefixes,rows=a.pins()
reject('prior trn22 result cannot become trn21 replica',lambda:call(bi=binding))
for key,value in [('host','trn22'),('source_sha256','0'*64),('label','unrelated')]:
    bad=copy.deepcopy(binding);bad[key]=value;reject('wrong '+key,lambda bad=bad:a.intended(bad))
bad=copy.deepcopy(binding);bad['training_command'].append('--recursive-schedule');reject('extra Recursive factor',lambda:a.intended(bad))
# Explicitly synthetic live-eval fixture: these IDs are NEVER inserted into the run auditor.
expected=[binding['evaluation_command'][5],'prepare.py','eval-public','--train-py','train.py','--checkpoint',binding['sandbox']+'/out/final.pt','--device','neuron','--batch-size','1','--seq-len','1024','--eval-tokens','20971520','--timeout-seconds','1200']
cmds=[binding['evaluation_command'],binding['evaluation_command'][5:],expected]
obs=dict(root=binding['root'],utc='2026-09-26T02:00:00+00:00',job_lines=[f'{pid} {parent} 90000 90000 1 python '+shlex.join(cmd) for pid,parent,cmd in zip([90001,90002,90003],[90000,90001,90002],cmds)],lock_inode=596823,lock_device=66305,lock_competitor_rc=1,neuron_ls_rc=0,neuron_ls_stderr='',neuron_ls=json.dumps([{'neuron_processes':[{'pid':90003}]}]))
of={'NATIVE_BINDING.disabled.json':json.dumps(binding).encode(),'fresh-public20m-launch.json':json.dumps(dict(command=binding['evaluation_command'],source_sha256=a.SOURCE,environment=binding['evaluation_environment'],utc='2026-09-26T01:59:00+00:00')).encode()}
a.require(a.eval_identities(obs,of,binding)=={90001,90002,90003},'synthetic eval PID extraction positive; no actual identities inferred')
cases.append(dict(name='synthetic timeout/harness/evaluator identity chain',outcome='SYNTHETIC_FIXTURE_ONLY_PASS'))
bad=copy.deepcopy(obs);bad['job_lines'].pop();reject('actual evaluator identity missing',lambda:a.eval_identities(bad,of,binding))
bad=copy.deepcopy(obs);bad['job_lines'][2]=bad['job_lines'][2].replace('90003 90002','90003 90001');reject('broken actual eval lineage',lambda:a.eval_identities(bad,of,binding))
bad=copy.deepcopy(obs);bad['neuron_ls']='[{"neuron_processes":[]}]';reject('evaluator not observed on device',lambda:a.eval_identities(bad,of,binding))
bad=copy.deepcopy(obs);bad['lock_competitor_rc']=0;reject('lifetime lock missing during eval',lambda:a.eval_identities(bad,of,binding))
bad=copy.deepcopy(obs);bad['root']='different';reject('foreign eval observation',lambda:a.eval_identities(bad,of,binding))
for pid in sorted(a.KNOWN_COMPILER_PIDS|{90003}):
    required=a.KNOWN_PIDS|{90001,90002,90003}
    reject('missing known/actual final PID '+str(pid),lambda pid=pid:a.known_postflight({'observed_pids':sorted(required-{pid})},required))
reject('walrus survives census',lambda:a.known_postflight({'observed_pids':sorted(a.KNOWN_PIDS),'all_process_identity_rows':[{'comm':'walrus_driver'}]},a.KNOWN_PIDS))
pending=a.main([]);a.require(pending['status']=='ONLY_PENDING_REAL_RAW' and pending['public_20m'] is None and pending['evaluation_pids'] is None and not pending['ledger_written'],'no args remains pending')
reject('final pins without actual evaluation observation',lambda:a.main(['--raw-commit','f'*40,'--bundle-sha','a'*64]))
print(json.dumps(dict(status='AUDITOR_READINESS_ONLY',actual_replica_result=None,fixture_cases=cases,checks=len(a.checks),known_process_ids=sorted(a.KNOWN_PIDS),ledger_written=False),indent=2,allow_nan=False))
