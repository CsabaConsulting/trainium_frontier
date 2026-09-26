"""Fixture parsing only: never promotes full36 evidence into a bcbd result."""
import copy, importlib.util, json
from pathlib import Path
spec=importlib.util.spec_from_file_location('prepared_audit',Path(__file__).with_name('audit.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
cases=[]
def reject(name,fn):
    try:fn()
    except (ValueError,KeyError,TypeError,IndexError):cases.append(dict(name=name,outcome='EXPECTED_REFUSAL'))
    else:raise RuntimeError('NEGATIVE_ACCEPTED:'+name)
raw=a.git('110054fe','research/v2/operator/20260926-trn22-startup-clock-full/completed/results-bundle.json')
bundle,f=a.decode(raw,'2557fbc01b642425201eb077fe84a1967ac4b518b52a9f9bf6ac1c0f8be2cb99')
p='research/v2/operator/20260926-trn22-startup-clock-full/completed/'
post=a.parse(a.git('110054fe',p+'explicit-process-postflight.json'))
ck=a.parse(a.git('110054fe',p+'post-eval-checkpoint-hash.json'))
b=a.parse(f['NATIVE_BINDING.disabled.json'])
def call(files=None,post_override=None,ck_override=None,b_override=None):
    return a.closed_result(bundle,f if files is None else files,b if b_override is None else b_override,post if post_override is None else post_override,ck if ck_override is None else ck_override,832589,set(post['observed_pids']))
facts=call()
a.require(facts['evaluation']['val_bpb']==0.9883821509811542 and facts['steps']==3630,'archived full36 positive parser fixture')
cases.append(dict(name='archived full36 parser fixture ONLY',outcome='FIXTURE_ACCEPTED_NOT_BCBD_EVIDENCE'))
reject('wrong outer SHA',lambda:a.decode(raw,'0'*64))
bad=a.parse(raw);bad['files']['terminal.json']['bytes']+=1;encoded=json.dumps(bad).encode()
reject('inner bytes corruption',lambda:a.decode(encoded,a.sha(encoded)))
def changed(name,fn):
    files=dict(f);v=a.parse(files[name]);fn(v);files[name]=json.dumps(v).encode();return files
reject('terminal incomplete',lambda:call(changed('terminal.json',lambda x:x.update(status='TRAINING_ONLY'))))
reject('nonzero eval exit',lambda:call(changed('eval/rehearsal.json',lambda x:x.update(returncode=1))))
reject('over-budget flag contradiction',lambda:a.timing(dict(a.parse(f['train/rehearsal.json']),charged_through_process_exit=1801)))
h=a.parse(f['train/rehearsal.json']);h['observations'][4]['observed_seconds']=float('nan')
reject('nonfinite boundary',lambda:a.timing(h))
missing=dict(f);missing.pop('metrics-training/metric_table_graph_stats_rank3.csv')
reject('missing fourth graph CSV',lambda:call(missing))
bad=copy.deepcopy(ck);bad['sha256']='0'*64
reject('changed post-eval checkpoint',lambda:call(ck_override=bad))
bad=copy.deepcopy(post);bad['pid_rows'][0]['exists']=True
reject('surviving PID',lambda:call(post_override=bad))
bad=copy.deepcopy(post);bad['all_process_identity_rows'][0]['sid']=post['observed_pids'][0]
reject('surviving related SID',lambda:call(post_override=bad))
bad=copy.deepcopy(b);bad['source_sha256']='0'*64
reject('wrong source binding',lambda:call(b_override=bad))
el=f['fresh-public20m.log'].decode()
reject('missing final totals',lambda:a.evaluation('evaluation running'))
reject('wrong token count',lambda:a.evaluation(el.replace('20971520','20971519')))
reject('BPB disagrees with totals',lambda:a.evaluation(el.replace('0.9883821509811542','0.1')))
reject('nonfinite nats',lambda:a.evaluation(el.replace('57179468.0','NaN')))
reject('eval over timeout',lambda:a.evaluation(el.replace('"eval_over_timeout": false','"eval_over_timeout": true')))
actual_binding,control,ledger,selector=a.pins()
reject('full36 cannot substitute for bcbd final',lambda:call(b_override=actual_binding))
ns={'__name__':'readonly_fixture_selector'};exec(compile(selector,'immutable-decide.py','exec'),ns)
a.require(ns['invalid_reason'](control) is None,'genuine pinned control accepted by unchanged selector')
mutant=copy.deepcopy(control);mutant['training_budget_seconds']=1799
a.require(ns['invalid_reason'](mutant) is not None,'unchanged selector budget negative')
mutant=copy.deepcopy(control);mutant['charged_through_exit_seconds']=1801
a.require(ns['invalid_reason'](mutant) is not None,'unchanged selector contradictory budget negative')
print(json.dumps(dict(status='ONLY_PENDING_REAL_RAW',candidate_result=None,selector_candidate_verdict=None,fixture_only='exact closed full36 archive110054fe',fixture_cases=cases,checks=len(a.checks),nonfixture_control_bpb=control['public_20m'],authoritative_writes=False),indent=2,allow_nan=False))
