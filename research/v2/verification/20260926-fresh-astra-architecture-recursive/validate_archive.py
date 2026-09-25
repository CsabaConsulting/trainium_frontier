"""Authenticate preserved fresh session and recorded tests; not a source re-review."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
checks=[]
def require(ok,name):
    if not ok:raise RuntimeError(name)
    checks.append(name)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads((HERE/path).read_text())
archive=read('ARCHIVE_MANIFEST.json')
for row in archive['files']:
    path=HERE/row['path']
    require(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'archive '+row['path'])
require(len(archive['files'])==archive['count'] and sum(r['bytes'] for r in archive['files'])==archive['total_bytes'],'archive totals')
prepared=read('PREPARED.json');inputs=read('packet/INPUT_MANIFEST.json')
require(sha(HERE/'packet/INPUT_MANIFEST.json')==prepared['input_manifest_sha256'],'exact input manifest')
for row in inputs['included']:
    path=HERE/'packet'/row['path']
    require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'input '+row['path'])
require({str(p.relative_to(HERE/'packet')) for p in (HERE/'packet').rglob('*') if p.is_file()}=={'INPUT_MANIFEST.json'}|{r['path'] for r in inputs['included']},'input allowlist only')
require(inputs['prior_history'] is inputs['shared_branch'] is False,'no shared history/branch')
lines=(HERE/'packet/baseline-to-candidate.diff').read_text().splitlines(keepends=True)
old=[];new=[]
for line in lines:
    if line.startswith(('--- ','+++ ','@@')):continue
    if line.startswith((' ','-')):old.append(line[1:])
    if line.startswith((' ','+')):new.append(line[1:])
reference=''.join(s[1:] for s in (HERE/'packet/schedule-reference.diff').read_text().splitlines(keepends=True) if s.startswith('+') and not s.startswith('+++ '))
for key,text in [('baseline',''.join(old)),('candidate',''.join(new)),('schedule_reference',reference)]:
    require(hashlib.sha256(text.encode()).hexdigest()==inputs['source_sha256'][key],'reconstructed '+key)
    require(sha(HERE/'session/reviewer-output/source'/f'{key}.py')==inputs['source_sha256'][key],'reviewer source '+key)
launch=read('session/launch.json');terminal=read('session/terminal.json');preflight=read('session/preflight.json')
require(preflight['exit']==0,'isolation preflight')
require(launch['manifest_sha256']==terminal['manifest_sha256']==prepared['input_manifest_sha256'],'input bound to run')
require(terminal['exit_code']==0 and terminal['elapsed_seconds']<terminal['wall_cap_seconds']==1200,'terminal bounded exit0')
require(launch['prior_history'] is launch['repository_mounted'] is False and launch['model_requested']=='gpt-6-astra','fresh requested model/isolation')
command=launch['command']
require(all(f in command for f in ('--ephemeral','--ignore-user-config','--ignore-rules','--skip-git-repo-check','--unshare-all')) and 'resume' not in command and 'fork' not in command,'fresh CLI flags')
require(command[command.index('-m')+1]=='gpt-6-astra','actual argv model')
events=[json.loads(line) for line in (HERE/'session/events.jsonl').read_text().splitlines() if line.strip()]
threads=[e['thread_id'] for e in events if e.get('type')=='thread.started']
turns=[e for e in events if e.get('type')=='turn.completed']
require(len(threads)==len(turns)==1 and len([e for e in events if e.get('type')=='turn.started'])==1 and not any(e.get('type')=='turn.failed' for e in events),'single completed new turn')
output=HERE/'session/reviewer-output';own=read('session/reviewer-output/artifact_manifest.json')
for row in own:
    path=output/row['path']
    require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'reviewer manifest '+row['path'])
decision=read('session/reviewer-output/decision.json')
require(decision['decision']=='PASS' and decision['source_hashes']==inputs['source_sha256'] and decision['source_commits']['candidate_commit']==inputs['candidate_commit'],'literal scoped verdict/source')
require(decision['authorization']['next_full_size_cpu_qualification_suitability'] is True and all(decision['authorization'][k] is False for k in ('full_size_gate_complete','host','native','cold','quality','adoption','submission')),'authorization not widened')
cases=read('session/reviewer-output/artifacts/exactness_cases.json');summary=read('session/reviewer-output/artifacts/exactness_summary.json')
require(len(cases)==summary['checks']==6263 and all(r['passed'] is True for r in cases) and summary['failed_checks']==0 and summary['status']=='PASS','6263 recorded numerical tests')
require(len([r for r in cases if r['name'].startswith('negative_control:')])==31,'31 meaningful negative controls')
require(read('session/reviewer-output/artifacts/exactness_failures.json')==[],'successful run no omitted numeric failures')
require(summary['environment']['models_instantiated']==0 and summary['environment']['threads']==summary['environment']['interop_threads']==1 and summary['environment']['tracked_peak_tensor_storage_bytes_overcounting_views']==522,'tiny sequential CPU resource evidence')
failed=read('session/reviewer-output/artifacts/run01/exactness_summary.json')
require(failed['status']=='REFUSE' and failed['checks']==0 and 'StopIteration' in failed['error'],'initial extractor failure retained')
require(sha(output/'artifacts/run01/test.py')!=sha(output/'artifacts/run02/test.py'),'corrected reviewer test version retained')
for name,count in [('receipt_audit',116),('source_audit',12)]:
    records=read('session/reviewer-output/artifacts/'+name+'.json')['checks']
    require(len(records)==count and all(x['passed'] is True for x in records),name+' recorded tests')
print(json.dumps({'archive_integrity':'PASS','checks':len(checks),'files':archive['count'],'bytes':archive['total_bytes'],'thread_id':threads[0],'started_utc':launch['started_utc'],'completed_utc':terminal['completed_utc'],'elapsed_seconds':terminal['elapsed_seconds'],'model_requested':launch['model_requested'],'usage':turns[0]['usage'],'input_manifest_sha256':prepared['input_manifest_sha256'],'actual_review_decision':decision['decision'],'review_scope':decision['scope'],'passing_numerical_records':len(cases),'negative_controls':31,'reviewer_harness_initial_failure_preserved':True,'source_correctness_not_decided_by_archive_validator':True},indent=2))
