"""Independent mechanical audit: immutable Git reads and in-memory stdlib mocks only."""
import ast, base64, copy, hashlib, json, shlex, subprocess, sys
from pathlib import PurePosixPath
from types import SimpleNamespace as NS

C='f4fe5d8e557f3d0646d1a2d6a47858768e29954d'
P='research/v2/proposals/20260926-startup-adjusted-full-trn21-replication/'
checks=[]
def ok(value,name):
    if not value: raise ValueError(name)
    checks.append(name)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def git(commit,path):return subprocess.check_output(['git','show',commit+':'+path])
def doc(name):return json.loads(git(C,P+name))
def bundle(commit,path,pin):
    raw=git(commit,path);ok(sha(raw)==pin,'outer archive '+commit)
    obj=json.loads(raw);out={}
    for name,item in obj['files'].items():
        raw=item['text'].encode() if 'text' in item else base64.b64decode(item['base64'],validate=True)
        ok(sha(raw)==item['sha256'] and len(raw)==item['bytes'],'inner archive '+name)
        out[name]=raw
    ok(len(out)==obj['file_count'],'archive count '+commit)
    return out
def constants(tree):
    ns={'Path':PurePosixPath}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.Assign)],type_ignores=[]),'constants','exec',optimize=0),ns)
    return ns
def mock_case(tree,c,scenario):
    events=[];records={};calls=[];guard_calls=[];bad_hash={'source':'SOURCE_SHA','prepare':'PREP_SHA','harness':'HARNESS_SHA'}
    class Path(PurePosixPath):
        def glob(self,pattern):return []
        def is_dir(self):return scenario!='cache_missing'
        def is_file(self):return scenario!='checkpoint_missing'
        def read_text(self):
            return json.dumps(dict(returncode=1 if scenario=='rehearsal_error' else 0,charged_through_process_exit_within_budget=scenario!='budget_false',checkpoint_sha256='a'*64,checkpoint_bytes=453048594))
    def digest(path):
        text=str(path)
        pins={str(c['ROOT']/'package/train.py'):c['SOURCE_SHA'],str(c['PREP']):c['PREP_SHA'],str(c['HARNESS']):c['HARNESS_SHA'],str(c['ROOT']/'idle_guard.py'):'adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737'}
        if scenario in bad_hash and pins.get(text)==c[bad_hash[scenario]]:return '0'*64
        if scenario=='guard_hash' and text.endswith('/idle_guard.py'):return '0'*64
        return pins[text]
    def exists(path):
        text=str(path)
        return (scenario=='stop' and text=='/home/ubuntu/STOP') or (scenario=='train_exists' and text==str(c['ROOT']/'train')) or (scenario=='eval_exists' and text==str(c['ROOT']/'eval'))
    def guard(*args):
        guard_calls.append(args);events.append('guard')
        if scenario=='first_idle_refuse' or (scenario=='second_idle_refuse' and len(guard_calls)==2):raise RuntimeError('IDLE_REFUSED')
        return 8 if scenario=='changed_fd' and len(guard_calls)==2 else 7
    def phase(name,cmd,env,fd):
        calls.append((name,cmd,env,fd));events.append(name)
        if name=='training' and scenario=='train_exit':return 124
        return 2 if name=='fresh-public20m' and scenario=='eval_exit' else 0
    guardobj=NS(require_idle_under_inherited_lock=guard)
    spec=NS(loader=NS(exec_module=lambda m:None))
    ns=dict(c);ns.update(Path=Path,ROOT=Path(c['ROOT']),PREP=Path(c['PREP']),HARNESS=Path(c['HARNESS']),
        os=NS(path=NS(lexists=exists),statvfs=lambda p:NS(f_bavail=1 if scenario=='disk_low' else 20*1024**3,f_frsize=1)),
        sha=digest,json=json,utc=lambda:'FIXED',record=lambda name,obj:records.__setitem__(name,obj),phase=phase,
        importlib=NS(util=NS(spec_from_file_location=lambda *a:spec,module_from_spec=lambda s:guardobj)),sys=sys)
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight']
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'exact-preflight','exec',optimize=0),ns)
    block=[n for n in tree.body if isinstance(n,ast.Try)]
    error=None;exit_code=None
    try:exec(compile(ast.Module(body=block,type_ignores=[]),'exact-driver-try','exec',optimize=0),ns)
    except SystemExit as e:exit_code=e.code
    except Exception as e:error=type(e).__name__+': '+str(e)
    if scenario=='success':
        ok(events==['guard','training','guard','fresh-public20m'],'one train then fresh eval with two guards')
        ok(exit_code==0 and records['terminal.json']['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW','success remains pending review')
        ok(all(x[3]==7 for x in calls),'same inherited FD across both phases')
    elif scenario=='eval_exit':
        ok(exit_code==2 and records['terminal.json']['status']=='EVAL_FAILED','eval failure propagated')
    else:
        ok(error is not None and records['terminal.json']['status']=='FULL_DRIVER_FAILED','fail closed '+scenario)
        ok(not any(x[0]=='fresh-public20m' for x in calls),'no eval after failure '+scenario)
        if scenario not in ['train_exit','rehearsal_error','budget_false','checkpoint_missing','second_idle_refuse','changed_fd']:
            ok(not calls,'no training after preflight failure '+scenario)
    ok(records['terminal.json']['no_automatic_next_job'] is True,'no auto next '+scenario)
    return dict(scenario=scenario,events=events,error=error,exit_code=exit_code,terminal=records['terminal.json'])
def main():
    b=doc('NATIVE_BINDING.disabled.json');proposal=doc('EXPERIMENT_PROPOSAL.json');manifest=doc('MANIFEST.sha256.json')
    for name,pin in manifest['files'].items():ok(sha(git(C,P+name))==pin,'manifest '+name)
    ok(git(C,P+'EXPERIMENT_PROPOSAL.json')==git('fbfd1927',P+'EXPERIMENT_PROPOSAL.json'),'proposal-first exact bytes')
    raw=git(C,P+'host_driver.py');old=git(b['derivation']['parent_commit'],b['derivation']['parent_driver_path'])
    tree=ast.parse(raw);oldtree=ast.parse(old);c=constants(tree);oc=constants(oldtree)
    expected=old.decode().replace(oc['LABEL'],c['LABEL']).replace(oc['SOURCE_SHA'],c['SOURCE_SHA']).replace(ast.get_docstring(oldtree),ast.get_docstring(tree))
    ok(expected.encode()==raw,'whole driver byte identity except exact label/source/docstring substitutions')
    ok(ast.dump(ast.Module(body=tree.body[1:],type_ignores=[])).replace(c['LABEL'],oc['LABEL']).replace(c['SOURCE_SHA'],oc['SOURCE_SHA'])==ast.dump(ast.Module(body=oldtree.body[1:],type_ignores=[])),'whole functional AST identity')
    ok(sha(raw)==b['driver_sha256'] and sha(old)==b['derivation']['parent_driver_sha256'],'exact old/new driver pins')
    for key,value in [('ROOT',b['root']),('PREP',b['prepare_path']),('HARNESS',b['harness_path'])]:ok(str(c[key])==value,'constant '+key)
    for key,field in [('TRAIN_COMMAND','command'),('EVAL_COMMAND','evaluation_command'),('TRAIN_ENV','training_environment'),('EVAL_ENV','evaluation_environment')]:ok(c[key]==b[field],'constant '+key)
    for field in ['stage_authorized','launch_authorized','host_reserved','source_adoption','submission_authorized','operator_assigned']:ok(b[field] is False,'disabled '+field)
    ok(b['host']=='trn21' and b['expected_hostname']=='ip-172-31-34-85' and b['original_lock']['inode']==596823 and b['original_lock']['device']==66305,'bound original trn21 identity')
    ok(b['outer_command']==['/usr/bin/flock','-n','/home/ubuntu/.cache/trainium_freshstart/device0.lock','/usr/bin/python3','-I','-B',b['driver_path']],'normal nonoptimized original outer flock')
    ok(b['dispatch_command']==['/usr/bin/python3','-I','-B',b['root']+'/dispatch_driver.py'],'normal detached dispatcher')
    source=git(b['source_commit'],b['source_git_path']);ok(sha(source)==b['source_sha256']==c['SOURCE_SHA'],'source exact Git bytes, no source review claimed')
    prov=doc('package/provenance.json');argv=shlex.split(git(C,P+'package/launch-command.txt').decode())
    ok(argv==b['training_command']==proposal['argv']==shlex.split(prov['launch_command']),'all proposed argv representations')
    ok(prov['train_sha256']==b['source_sha256'] and '--recursive-schedule' not in argv and argv.count('--startup-adjusted-schedule')==1,'one clock factor no Recursive')
    full=bundle('110054fe','research/v2/operator/20260926-trn22-startup-clock-full/completed/results-bundle.json','2557fbc01b642425201eb077fe84a1967ac4b518b52a9f9bf6ac1c0f8be2cb99')
    fb=json.loads(full['NATIVE_BINDING.disabled.json'])
    ok(argv==fb['training_command'] and b['source_sha256']==fb['source_sha256'],'exact completed trn22 scientific argv/source')
    import re
    command=re.findall(r'^REHEARSAL_COMMAND (.+)$',full['training.log'].decode(),re.M)
    ok(len(command)==1 and json.loads(command[0])==argv,'actual executed trn22 rehearsal argv')
    contrast={}
    for field,launchname in [('training_environment','training-launch.json'),('evaluation_environment','fresh-public20m-launch.json')]:
        before=json.loads(full[launchname])['environment'];after=b[field]
        ok(before==fb[field],'actual archived environment '+field)
        contrast[field]=[dict(key=k,trn22_completed=before.get(k),trn21_proposed=after.get(k)) for k in sorted(set(before)|set(after)) if before.get(k)!=after.get(k)]
        allowed={'NEURON_COMPILE_CACHE_DIR','NEURON_COMPILE_CACHE_URL','TORCHINDUCTOR_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_METRICS_DIR','TRN_EXP_LABEL'}
        ok(all(x['key'] in allowed for x in contrast[field]),'only disclosed per-host paths/labels '+field)
    ok(contrast==doc('ENVIRONMENT_CONTRAST.json')['differences'],'independently regenerated complete environment contrast')
    for key in ['TRAIN_ENV','EVAL_ENV']:
        a={k:v for k,v in c[key].items() if k not in ['TRN_EXP_LABEL','TORCH_NEURONX_METRICS_DIR']}
        z={k:v for k,v in oc[key].items() if k not in ['TRN_EXP_LABEL','TORCH_NEURONX_METRICS_DIR']}
        ok(a==z,'preserve all predecessor per-host caches and science '+key)
    ok(c['TRAIN_ENV']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS']=='1800' and argv[argv.index('--max-train-seconds')+1]=='1800','explicit 1800 both env/argv')
    ok(c['TRAIN_COMMAND'][4]=='2730s' and c['EVAL_COMMAND'][4]=='1800s' and all('--kill-after=60s' in x for x in [c['TRAIN_COMMAND'],c['EVAL_COMMAND']]),'separate outer watchdogs')
    ok(c['EVAL_COMMAND'][-2:]==['--eval-tokens','20971520'] and c['EVAL_COMMAND'][c['EVAL_COMMAND'].index('--checkpoint')+1]==b['sandbox']+'/out/final.pt','same checkpoint and attached fresh20M')
    harness=git('00f2ca7f','research/scratch/submission_rehearsal.py');ok(sha(harness)==b['harness_sha256'],'unchanged actual harness')
    ok('"--batch-size", "1", "--seq-len", "1024"' in harness.decode() and '"eval-public"' in harness.decode(),'actual harness B1/context1024 evaluator path')
    for path,pin in [('prepare.py',b['prepare_sha256']),('research/v2/operator/20260926-common-host-idle-guard/idle_guard.py',b['idle_guard_sha256']),('research/v2/operator/20260926-common-host-idle-guard/dispatch_driver.py',b['dispatch_driver_sha256'])]:ok(sha(git(C,path))==pin,'unchanged pinned support '+path)
    cg=b['cold_gate'];cold=bundle(cg['raw_commit'],cg['archive'],cg['raw_bundle_sha256'])
    ok(sha(cold[cg['receipt_key']])==cg['receipt_sha256'],'exact cold receipt')
    h=json.loads(cold[cg['receipt_key']]);fifth=h['observations'][4]['observed_seconds']
    ok(fifth==845.6273377500474==cg['observed_uncapped_fifth_seconds'] and fifth<=870 and h['returncode']==0 and h['step_boundaries']==5,'prior exact cold-five receipt')
    ok(cg['host']=='trn22' and b['cold_startup_proven'] is False and 'NOT trn21 cold' in b['cold_scope'],'no false trn21 cold claim')
    sg=b['source_gate'];fresh=git(sg['fresh_review_commit'],sg['fresh_review'])
    ok(fresh.startswith(b'**PASS') and b['source_sha256'].encode() in fresh,'separate accepted fresh source report')
    cpu=json.loads(git(sg['full_parameter_cpu_review_commit'],sg['full_parameter_cpu_review']))
    ok(cpu['status']=='PASS_FULL_PARAMETER_CPU_RESULT_ONLY' and cpu['source_candidate']==b['source_sha256'] and cpu['parameters']==113256724 and cpu['ranks_each']==4 and cpu['updates_each']==3 and cpu['micro_batch']==[2,256] and cpu['native_correctness'] is False,'accepted full CPU scope only')
    cpufiles=bundle(sg['full_parameter_cpu_raw_commit'],sg['full_parameter_cpu_raw'],sg['full_parameter_cpu_raw_sha256'])
    control=b['decision_comparator'];lines=git(control['ledger_commit'],control['ledger_path']).splitlines(keepends=True)
    rows=[line for line in lines if json.loads(line).get('label')==control['label']]
    ok(len(rows)==1 and sha(rows[0])==control['raw_row_sha256'],'exact genuine control ledger row bytes')
    row=json.loads(rows[0]);ok(row['kind']=='control' and row['host']=='trn21' and row['public_20m']==0.9899528600562257 and row['train_py_sha256']==control['source_sha256'],'genuine control role/host/score')
    ok(b['no_auto_next_job'] is True and b['prerequisites']['no_automatic_release'] is True and 'bcbd' in b['prerequisites']['current_job'],'protected predecessor and separate release')
    phases=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='phase']
    ok('pass_fds=(lock_fd,)' in ast.unparse(phases[0]),'real child inherits lifetime lock FD')
    scenarios=['success','stop','disk_low','source','prepare','harness','cache_missing','train_exists','eval_exists','guard_hash','first_idle_refuse','train_exit','rehearsal_error','budget_false','checkpoint_missing','second_idle_refuse','changed_fd','eval_exit']
    mocks=[mock_case(tree,c,s) for s in scenarios]
    return dict(status='MECHANICAL_PASS_ONLY',packet_commit=C,proposal_commit=b['proposal_first_commit'],checks=len(checks),check_names=checks,mock_scenarios=mocks,driver_sha256=sha(raw),binding_sha256=sha(git(C,P+'NATIVE_BINDING.disabled.json')),manifest_sha256=sha(git(C,P+'MANIFEST.sha256.json')),source_sha256=b['source_sha256'],fresh_report_sha256=sha(fresh),full_cpu_authenticated_files=len(cpufiles),cold_fifth_seconds=fifth,cold_host='trn22',trn21_cold_proven=False,actual_environment_contrast=contrast,control_row_sha256=sha(rows[0]),source_review_performed=False,native_execution=False,launch_authorized=False,limitations=['Historical source implementer; independent of this packet only. Separate fresh source and CPU dispositions authenticated, not replaced.','Host driver MUST use normal Python; exact extracted guards compiled optimize=0 even during -O test harness.','Hostname/original inode/unique staging/breaker and every-outcome explicit descendant postflight are Operator obligations, not newly automated driver guarantees.','Prior trn22 cold is not trn21 cold; warm cache paths do not prove a hit.'])
if __name__=='__main__':
    try:result=main()
    except Exception as e:
        print(json.dumps(dict(status='REFUSE_OR_INCOMPLETE',error=type(e).__name__+': '+str(e),checks_completed=len(checks))));raise SystemExit(2)
    print(json.dumps(result,indent=2,allow_nan=False))
