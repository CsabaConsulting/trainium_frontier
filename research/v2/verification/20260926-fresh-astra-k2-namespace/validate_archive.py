"""Authenticate fresh reviewer artifacts/execution; no independent source verdict."""
import argparse
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
checks=[]
def check(name,value):
    if not value:raise RuntimeError(name)
    checks.append(name)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads((HERE/path).read_text())
archive=read('ARCHIVE_MANIFEST.json')
for row in archive['files']:
    path=HERE/row['path'];check('archive '+row['path'],path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'])
check('archive complete count',archive['terminal'] is True and archive['count']==len(archive['files'])==80 and archive['total_bytes']==sum(x['bytes'] for x in archive['files']))
prepared=read('PREPARED.json');inputs=read('packet/INPUT_MANIFEST.json')
check('exact input manifest',sha(HERE/'packet/INPUT_MANIFEST.json')==prepared['input_manifest_sha256'])
check('input-only no history',len(inputs['included'])==7 and inputs['prior_history'] is inputs['shared_branch'] is False)
for row in inputs['included']:
    path=HERE/'packet'/row['path'];check('input '+row['path'],sha(path)==row['sha256'] and path.stat().st_size==row['bytes'])
check('input inventory',set(str(x.relative_to(HERE/'packet')) for x in (HERE/'packet').rglob('*') if x.is_file())=={x['path'] for x in inputs['included']}|{'INPUT_MANIFEST.json'})
out=HERE/'session/reviewer-output';decision=json.loads((out/'decision.json').read_text());results=json.loads((out/'results.json').read_text())
own=json.loads((out/'output-hashes.json').read_text())
for row in own['files']:
    path=out/row['path'];check('reviewer hash '+row['path'],path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'])
check('reviewer inventory with final CLI response',set(str(x.relative_to(out)) for x in out.rglob('*') if x.is_file())=={x['path'] for x in own['files']}|{'output-hashes.json','final-response.md'})
for which in ('baseline','candidate'):
    check('exact reconstructed '+which,sha(out/(which+'.py'))==inputs['source_sha256'][which]==decision[which+'_sha256'])
launch=read('session/launch.json');terminal=read('session/terminal.json');preflight=read('session/preflight.json')
check('real isolation preflight',preflight['exit']==0)
check('one bounded terminal',terminal['exit_code']==0 and not terminal.get('wall_cap_triggered') and terminal['elapsed_seconds']<terminal['wall_cap_seconds']==1200)
check('same input launch/terminal',launch['manifest_sha256']==terminal['manifest_sha256']==prepared['input_manifest_sha256'])
command=launch['command'];check('fresh Astra no fork resume',launch['model_requested']==command[command.index('-m')+1]=='gpt-6-astra' and all(x in command for x in ('--ephemeral','--ignore-user-config','--ignore-rules','--unshare-all')) and 'fork' not in command and 'resume' not in command and launch['prior_history'] is launch['repository_mounted'] is False)
events=[json.loads(s) for s in (HERE/'session/events.jsonl').read_text().splitlines() if s]
threads=[e['thread_id'] for e in events if e.get('type')=='thread.started'];turns=[e for e in events if e.get('type')=='turn.completed']
check('one completed turn',len(threads)==len(turns)==1 and sum(e.get('type')=='turn.started' for e in events)==1 and not any(e.get('type')=='turn.failed' for e in events))
executions=[e['item'] for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='command_execution']
test_runs=[e for e in executions if e['command']=="/usr/bin/bash -lc 'bash /work/run_review.sh'"]
check('actual single recorded own test run',len(test_runs)==1 and test_runs[0]['exit_code']==0 and (out/'test.exitcode.txt').read_text().strip()=='0' and not (out/'test.stderr.txt').read_text())
check('literal verdict and no larger authority',decision['verdict']=='PASS' and decision['default_verdict']==decision['broader_use_verdict']=='REFUSE' and all(decision[k] is False for k in ('device_launch_authorized','adoption_authorized','submission_authorized','checker_reference_bodies_inspected')))
check('no native claim',decision['native_api_runtime_validation']==decision['native_compilation']==decision['native_exactness']=='UNEXECUTED' and decision['performance']=='UNMEASURED' and decision['quality']=='UNTESTED')
check('old refusal retained',decision['supplied_evidence']['original_failure_waived'] is False and decision['supplied_evidence']['successor_native_receipts']==0)
evidence=decision['independently_executed_evidence']
check('own test counts',results['status']=='PASS' and len(results['tests'])==sum(x['pass'] for x in results['tests'])==evidence['checks_total']==evidence['checks_passed']==91 and len(results['expected_errors'])==evidence['expected_exceptions']==14 and not results['unexpected_errors'] and evidence['attempts']==1)
check('resource scope',results['resources']==decision['resources'] and results['resources']['threads']==1 and results['resources']['max_logical_tensor_bytes']==131072 and results['resources']['max_rss_kib_linux']==31188 and results['native_executed'] is results['model_is_installed_frontend'] is False)
check('exactness zero tolerances',decision['contract']['acceptance']=='identical bytes, atol=rtol=0' and len(results['cases'])==evidence['fixtures']==7 and evidence['exact_byte_comparisons']==14)
for case in results['cases']:
    name=case['name'];expected=(out/'exactness'/(name+'.reference.f32le')).read_bytes()
    check('oracle bytes '+name,len(expected)==131072 and hashlib.sha256(expected).hexdigest()==case['reference_sha256'])
    for which in ('baseline','candidate'):
        raw=(out/'exactness'/(name+'.'+which+'.f32le')).read_bytes();row=case['outputs'][which]
        check(which+' actual byte comparison '+name,raw==expected and row['exact'] is True and row['bytes']==len(raw) and row['sha256']==hashlib.sha256(raw).hexdigest())
namespace=json.loads((out/'namespace-evidence.json').read_text())
check('namespace observations retained',set(namespace['baseline']['freevars'])=={'nl','nisa'} and namespace['baseline']['global_aliases']==[] and namespace['candidate']['freevars']==[] and set(namespace['candidate']['global_aliases'])=={'nl','nisa'} and namespace['candidate']['alias_resolution_after_rebind']=='new global')
negative=json.loads((out/'negative-controls.json').read_text())
check('arithmetic counterexamples retained',negative['ordered_hex']=='00000000' and negative['reordered_hex']==negative['widened_hex']=='3f800000')
check('strict byte counterexamples retained',negative['bytes_expected']!=negative['bytes_wrong_sign'] and negative['bytes_expected']!=negative['bytes_wrong_ulp'])
for rank in range(4):check('verbatim original rank '+str(rank),(out/'supplied'/f'rank{rank}.json').read_bytes()==(HERE/'packet/original-native/receipts'/f'rank{rank}.json').read_bytes())
check('verbatim terminal',(out/'supplied/terminal.json').read_bytes()==(HERE/'packet/original-native/terminal.json').read_bytes())
result={'archive_integrity':'PASS','checks':len(checks),'files':archive['count'],'bytes':archive['total_bytes'],'thread_id':threads[0],'model_requested':launch['model_requested'],'started_utc':launch['started_utc'],'completed_utc':terminal['completed_utc'],'elapsed_seconds':terminal['elapsed_seconds'],'usage':turns[0]['usage'],'actual_reviewer_verdict':decision['verdict'],'actual_scope':decision['scope'],'source_judgment_not_decided_by_this_validator':True,'input_manifest_sha256':prepared['input_manifest_sha256'],'report_sha256':sha(out/'REPORT.md'),'decision_sha256':sha(out/'decision.json')}
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args();encoded=json.dumps(result,indent=2)+'\n'
if args.output:args.output.write_text(encoded)
print(encoded,end='')
