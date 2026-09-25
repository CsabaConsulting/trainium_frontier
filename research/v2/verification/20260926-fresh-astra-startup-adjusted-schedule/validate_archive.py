"""Authenticate a completed fresh review, not decide source correctness."""
import argparse
import hashlib
import json
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
check('archive count/bytes',len(archive['files'])==archive['count'] and sum(r['bytes'] for r in archive['files'])==archive['total_bytes'] and archive['terminal'] is True)
launch_archive=read('LAUNCH_ARCHIVE_MANIFEST.json')
for row in launch_archive['files']:
    path=HERE/row['path'];check('launch snapshot unchanged '+row['path'],sha(path)==row['sha256'] and path.stat().st_size==row['bytes'])
prepared=read('PREPARED.json');inputs=read('packet/INPUT_MANIFEST.json')
check('frozen input manifest',sha(HERE/'packet/INPUT_MANIFEST.json')==prepared['input_manifest_sha256'])
for row in inputs['included']:
    path=HERE/'packet'/row['path'];check('input '+row['path'],sha(path)==row['sha256'] and path.stat().st_size==row['bytes'])
check('input inventory only',set(str(p.relative_to(HERE/'packet')) for p in (HERE/'packet').rglob('*') if p.is_file())=={'INPUT_MANIFEST.json'}|{r['path'] for r in inputs['included']})
check('no input history/branch',inputs['prior_history'] is inputs['shared_branch'] is False)
old=[];new=[]
for line in (HERE/'packet/baseline-to-candidate.diff').read_text().splitlines(True):
    if line.startswith(('--- ','+++ ','@@')):continue
    if line.startswith((' ','-')):old.append(line[1:])
    if line.startswith((' ','+')):new.append(line[1:])
check('complete baseline',hashlib.sha256(''.join(old).encode()).hexdigest()==inputs['source_sha256']['baseline'])
check('complete candidate',hashlib.sha256(''.join(new).encode()).hexdigest()==inputs['source_sha256']['candidate'])
launch=read('session/launch.json');terminal=read('session/terminal.json');preflight=read('session/preflight.json')
check('isolation preflight',preflight['exit']==0)
check('launch/terminal input binding',launch['manifest_sha256']==terminal['manifest_sha256']==prepared['input_manifest_sha256'])
check('bounded completed session',terminal['exit_code']==0 and terminal['elapsed_seconds']<terminal['wall_cap_seconds']==1200 and not terminal.get('wall_cap_triggered'))
check('model/history/repo',launch['model_requested']=='gpt-6-astra' and launch['prior_history'] is launch['repository_mounted'] is False)
cmd=launch['command'];check('new ephemeral CLI',all(x in cmd for x in ('--ephemeral','--ignore-user-config','--ignore-rules','--skip-git-repo-check','--unshare-all')) and 'resume' not in cmd and 'fork' not in cmd and cmd[cmd.index('-m')+1]=='gpt-6-astra')
events=[json.loads(s) for s in (HERE/'session/events.jsonl').read_text().splitlines() if s.strip()]
threads=[e['thread_id'] for e in events if e.get('type')=='thread.started'];turns=[e for e in events if e.get('type')=='turn.completed']
check('one new completed turn',len(threads)==len(turns)==1 and sum(e.get('type')=='turn.started' for e in events)==1 and not any(e.get('type')=='turn.failed' for e in events))
output=HERE/'session/reviewer-output'
decision=read('session/reviewer-output/decision.json')
check('complete report retained',(output/'REPORT.md').is_file())

check('literal reviewer decision and narrow scope',decision['decision']=='PASS' and decision['scope']=='Source review and independently executed bounded reduced-CPU exactness evidence establish suitability only for the next separately assigned full-size CPU qualification.')
check('exact reviewed sources',decision['baseline']['train_py_sha256']==inputs['source_sha256']['baseline'] and decision['candidate']['train_py_sha256']==inputs['source_sha256']['candidate'] and decision['input_manifest_sha256']==prepared['input_manifest_sha256'])
check('no broader approval',set(decision['not_authorized'])=={'native compilation','hardware execution','cold training','quality or BPB benefit claims','adoption','submission','full-size CPU execution within this task'} and decision['independent_source_review']['source_files_edited'] is False and decision['independent_numerical_evidence']['model_executed'] is False)
own=json.loads((output/'artifact_manifest.json').read_text())
for row in own:
    path=output/row['path'];check('reviewer artifact '+row['path'],not path.is_symlink() and path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'])
check('complete reviewer inventory',set(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())=={r['path'] for r in own}|{'artifact_manifest.json','final-response.md','finalize_review.stdout','finalize_review.stderr'})
check('explicit finalization output exclusions',"['artifact_manifest.json','finalize_review.stdout','finalize_review.stderr']" in (output/'finalize_review.py').read_text() and (output/'finalize_review.stderr').read_bytes()==b'' and json.loads((output/'finalize_review.stdout').read_text())['artifact_files']==len(own))
for row in decision['evidence_artifacts']:
    check('decision artifact '+row['path'],row['path'].startswith('/work/') and sha(output/row['path'].removeprefix('/work/'))==row['sha256'])
command_events=[e['item'] for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='command_execution']
check('observed both tiny numerical executions',all(any('--attempt '+attempt in e.get('command','') and 'reference_test.py' in e.get('command','') and e.get('status')=='completed' for e in command_events) for attempt in ('attempt1','attempt2')))
import struct
def fp32_bytes(values):return b''.join(struct.pack('<f',value) for value in values)
for attempt in ('attempt1','attempt2'):
    raw=output/attempt;summary=json.loads((raw/'summary.json').read_text())
    check(attempt+' positive and negative counts',summary['status']=='PASS' and summary['counts']=={'check':996,'negative_control':7} and summary['failures']==0 and summary['threads']==summary['interop_threads']==1)
    rows=[json.loads(line) for line in (raw/'results.jsonl').read_text().splitlines()]
    check(attempt+' all records passed',len(rows)==1003 and all(r['passed'] is True for r in rows) and json.loads((raw/'failures.json').read_text())==[])
    negatives=[r for r in rows if r['kind']=='negative_control']
    check(attempt+' retained expected failures',len(negatives)==7 and all(r['expected']!=r['wrong_actual'] and 'AssertionError' in r['expected_failure'] for r in negatives))
    memory=next(r for r in rows if r['name']=='tensor_memory_bound')
    check(attempt+' actual tiny bound',memory['tracked_peak_storage_bytes']=={'attempt1':476,'attempt2':372}[attempt] and memory['max_tensor_bytes']==16 and memory['limit_bytes']==64*1024*1024)
    tensor=json.loads((raw/'tensor_results.json').read_text());check(attempt+' four complete numerical modes',set(tensor)=={'baseline','default','off','on'} and all(len(v)==8 for v in tensor.values()))
    count=0
    for mode,steps in tensor.items():
        for row in steps:
            check(attempt+'/'+mode+'/'+str(row['step'])+' exact optimizer LR',struct.pack('<f',row['lr_expected'])==struct.pack('<f',row['lr_actual']))
            for p in row['parameters']:
                label=attempt+'/'+mode+'/'+str(row['step'])+'/'+str(p['index']);count+=1
                check(label+' actual/reference parameter bytes',bytes.fromhex(p['expected_bytes'])==bytes.fromhex(p['actual_bytes'])==fp32_bytes(p['expected_parameter'])==fp32_bytes(p['actual_parameter']))
                check(label+' delta/state',p['expected_delta']==p['actual_delta'] and p['expected_state']['step']==p['actual_state']['step'] and all(fp32_bytes(p['expected_state'][k])==fp32_bytes(p['actual_state'][k]) for k in ('m','v')))
                if p['index']==2:check(label+' permanently missing gradient',p['gradient_missing'] is True and p['state_present'] is False and p['expected_state']['step']==0)
    check(attempt+' 96 parameter/state comparisons',count==96)
    for mode in ('default','off'):
        check(attempt+' baseline identity '+mode,[[r['parameters'],r['gradients'],r['lr_actual']] for r in tensor['baseline']]==[[r['parameters'],r['gradients'],r['lr_actual']] for r in tensor[mode]])
    check(attempt+' intentional ON difference',tensor['baseline'][-1]['parameters']!=tensor['on'][-1]['parameters'])
    for row in json.loads((raw/'extractions.json').read_text()):
        if 'artifact' in row:
            check(attempt+' exact extracted artifact '+row['artifact'],sha(raw/row['artifact'])==row['sha256'])
        else:
            import ast
            import textwrap
            check(attempt+' callsite source metadata',row['label']=='callsite' and row['role'] in ('baseline','candidate'))
            lines=(output/(row['role']+'_train.py')).read_text().splitlines(True)
            selected=''.join(''.join(lines[first-1:last]) for first,last in row['original_lines'])
            actual=(raw/(row['role']+'_callsite.py')).read_text()
            check(attempt+' exact callsite AST '+row['role'],ast.dump(ast.parse(textwrap.dedent(selected)))==ast.dump(ast.parse(actual)))
receipt=read('session/reviewer-output/receipt_audit_attempt2/summary.json')
check('literal corrected receipt audit',receipt['status']=='PASS' and receipt['checks']==2671 and receipt['failed_checks']==0 and receipt['errors']==[])
failed=read('session/reviewer-output/receipt_audit/summary.json')
check('prior reviewer schema failure preserved',failed['status']=='REFUSE' and bool(failed['errors']))
check('probe and CPU warnings retained',"KeyError" in (output/'diagnostics/receipt_probe_failure.txt').read_text() and all('failed to parse processor information from /proc/cpuinfo' in (output/(a+'.stderr')).read_text() for a in ('attempt1','attempt2')))
check('original failed observer remains failed',read('packet/raw/failed-attempt1-base.json')['verdict']=='FAIL')
result={'archive_integrity':'PASS','checks':len(checks),'files':archive['count'],'bytes':archive['total_bytes'],'thread_id':threads[0],'model_requested':launch['model_requested'],'started_utc':launch['started_utc'],'completed_utc':terminal['completed_utc'],'elapsed_seconds':terminal['elapsed_seconds'],'usage':turns[0]['usage'],'manifest_sha256':prepared['input_manifest_sha256'],'actual_review_decision':decision['decision'],'scope':decision['scope'],'source_correctness_not_decided_by_this_validator':True}
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
text=json.dumps(result,indent=2)+'\n'
if args.output:args.output.write_text(text)
print(text,end='')
