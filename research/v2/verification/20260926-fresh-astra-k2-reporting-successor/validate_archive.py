"""Authenticate final fresh-session artifacts; never decide source correctness."""
import hashlib
import json
import argparse
from pathlib import Path

HERE=Path(__file__).resolve().parent
checks=[]
def check(name,ok):
    if not ok:raise RuntimeError(name)
    checks.append(name)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads((HERE/path).read_text())
archive=read('ARCHIVE_MANIFEST.json')
for row in archive['files']:
    path=HERE/row['path']
    check('archive '+row['path'],not path.is_symlink() and path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'])
check('archive count/bytes',len(archive['files'])==archive['count'] and sum(r['bytes'] for r in archive['files'])==archive['total_bytes'])
prepared=read('PREPARED.json');inputs=read('packet/INPUT_MANIFEST.json')
check('frozen input manifest',sha(HERE/'packet/INPUT_MANIFEST.json')==prepared['input_manifest_sha256'])
for row in inputs['included']:
    path=HERE/'packet'/row['path'];check('input '+row['path'],sha(path)==row['sha256'] and path.stat().st_size==row['bytes'])
check('input inventory only',set(str(p.relative_to(HERE/'packet')) for p in (HERE/'packet').rglob('*') if p.is_file())=={'INPUT_MANIFEST.json'}|{r['path'] for r in inputs['included']})
check('no input history/branch/conclusion',inputs['prior_history'] is inputs['shared_branch'] is inputs['scalar_inputs_projection']['prior_conclusions_included'] is False)
def reconstruct(name):
    old=[];new=[]
    for line in (HERE/'packet'/name).read_text().splitlines(True):
        if line.startswith(('--- ','+++ ','@@')):continue
        if line.startswith((' ','-')):old.append(line[1:])
        if line.startswith((' ','+')):new.append(line[1:])
    return ''.join(old).encode(),''.join(new).encode()
old,new=reconstruct('checker.diff')
check('baseline complete checker',hashlib.sha256(old).hexdigest()==inputs['baseline_checker_sha256'])
check('candidate complete checker',hashlib.sha256(new).hexdigest()==inputs['candidate_source_sha256']['check_native.py'])
for name in ('consumer_nki.py','reference.py'):
    old,new=reconstruct(name+'.diff');check('complete '+name,old==b'' and hashlib.sha256(new).hexdigest()==inputs['candidate_source_sha256'][name])
launch=read('session/launch.json');terminal=read('session/terminal.json');preflight=read('session/preflight.json')
check('isolation preflight',preflight['exit']==0)
check('launch/terminal input binding',launch['manifest_sha256']==terminal['manifest_sha256']==prepared['input_manifest_sha256'])
check('bounded completed session',terminal['exit_code']==0 and terminal['elapsed_seconds']<terminal['wall_cap_seconds']==1200 and not terminal.get('wall_cap_triggered'))
check('model/history/repo',launch['model_requested']=='gpt-6-astra' and launch['prior_history'] is launch['repository_mounted'] is False)
cmd=launch['command'];check('new ephemeral CLI',all(x in cmd for x in ('--ephemeral','--ignore-user-config','--ignore-rules','--skip-git-repo-check','--unshare-all')) and 'resume' not in cmd and 'fork' not in cmd and cmd[cmd.index('-m')+1]=='gpt-6-astra')
events=[json.loads(s) for s in (HERE/'session/events.jsonl').read_text().splitlines() if s.strip()]
threads=[e['thread_id'] for e in events if e.get('type')=='thread.started'];turns=[e for e in events if e.get('type')=='turn.completed']
check('one new completed turn',len(threads)==len(turns)==1 and sum(e.get('type')=='turn.started' for e in events)==1 and not any(e.get('type')=='turn.failed' for e in events))
decision=read('session/reviewer-output/decision.json')
check('literal reviewer disposition',decision['decision'] in ('PASS','REFUSE'))
check('complete report retained',(HERE/'session/reviewer-output/REPORT.md').is_file())
output=HERE/'session/reviewer-output'
own=json.loads((output/'artifact-manifest.json').read_text())
for row in own['entries']:
    path=output/row['path']
    check('reviewer manifest '+row['path'],path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'])
check('reviewer inventory plus CLI final response',set(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())=={r['path'] for r in own['entries']}|{'artifact-manifest.json','final-response.md'})
check('exact source scope',decision['candidate_source_sha256']==inputs['candidate_source_sha256'] and decision['baseline_checker_sha256']==inputs['baseline_checker_sha256'])
check('no launch or broader approval',all(decision[k] is False for k in ('source_modified','adoption','submission_authorized','launch_authorized','hardware_authorized')) and decision['atol']==decision['rtol']==0 and decision['native_legality']==decision['native_numerical_correctness']=='UNPROVEN')
for attempt,key in [('attempt-001','normal'),('attempt-002-optimized','optimized')]:
    raw=output/'raw'/attempt
    summary=json.loads((raw/'summary.json').read_text())
    check(attempt+' recorded evidence',summary==decision['evidence'][key] and summary['status']=='PASS' and summary['recurrence_cases']==32 and summary['integer_addition_pairs']==4217 and summary['comparison_cases']==17 and summary['reporting_cases']==24)
    check(attempt+' CPU bounds',summary['native_imported'] is summary['native_executed'] is summary['launch_authorized'] is False and summary['intraop_threads']==summary['interop_threads']==1 and summary['peak_live_tensor_storage_bytes']<64*1024*1024)
    execution=json.loads((output/'raw'/(attempt+'.execution.json')).read_text())
    check(attempt+' actual execution',execution['exit_code']==0 and execution['sequential'] is True and execution['script_sha256']==sha(output/'raw'/(attempt+'.script.py')))
    cases=sorted((raw/'recurrence').glob('*/result.json'))
    check(attempt+' complete recurrence cases',len(cases)==32)
    negatives=0
    for case in cases:
        record=json.loads(case.read_text());base=case.parent
        expected=(base/'expected.f32le').read_bytes()
        check(attempt+' recurrence bytes '+base.name,len(expected)==256*128*4 and expected==(base/'dense_actual.f32le').read_bytes()==(base/'consumer_actual.f32le').read_bytes() and record['max_abs']==record['atol']==record['rtol']==0 and record['byte_exact'] is True)
        for negative in base.glob('negative_*.f32le'):
            check(attempt+' negative '+base.name+'/'+negative.name,negative.read_bytes()!=expected)
            negatives+=1
    check(attempt+' retained regroup/reverse controls',negatives==6)
    for name,finite,metric,description in [('positive_inf',False,None,'inf'),('negative_inf',False,None,'inf'),('quiet_nan',False,None,'nan'),('signed_zero_mismatch',True,0,None)]:
        result=json.loads((raw/'comparison'/(name+'.json')).read_text())
        actual=result['actual']
        check(attempt+' strict scalar '+name,actual==result['expected'] and actual['exact'] is False and actual['finite'] is finite and actual['max_abs']==metric and actual.get('max_abs_nonfinite')==description and json.loads(result['serialized'],parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))==actual)
boundaries=json.loads((output/'raw/attempt-003-boundaries/summary.json').read_text())
check('own boundary evidence',boundaries==decision['evidence']['boundaries'] and boundaries['bf16_cases']==2059 and boundaries['setup_scenarios']==14 and boundaries['native_imported'] is boundaries['native_executed'] is False)
check('original setup limitation retained',boundaries['known_unchanged_no_cleanup_after_init']==['device construction failure','missing synchronize API'])
result=json.dumps({'archive_integrity':'PASS','checks':len(checks),'files':archive['count'],'bytes':archive['total_bytes'],'thread_id':threads[0],'model_requested':launch['model_requested'],'started_utc':launch['started_utc'],'completed_utc':terminal['completed_utc'],'elapsed_seconds':terminal['elapsed_seconds'],'usage':turns[0]['usage'],'manifest_sha256':prepared['input_manifest_sha256'],'actual_review_decision':decision['decision'],'source_correctness_not_decided_by_this_validator':True},indent=2)+'\n'
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
if args.output:args.output.write_text(result)
print(result,end='')
