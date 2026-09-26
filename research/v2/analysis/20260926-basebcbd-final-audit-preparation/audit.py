"""One run, archive-only result parser. No args means ONLY_PENDING_REAL_RAW."""
import argparse, base64, csv, datetime, hashlib, io, json, math, re, shlex, subprocess
P='research/v2/proposals/20260926-base-cache-full-trn21/'
RAW_ROOT='research/v2/operator/20260926-trn21-basebcbd-full/completed/'
LABEL='fs-20260926-basebcbd-initial-cache-trn21-r1-full'
SOURCE='bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522'
BASE='6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'
CONTROL='fs-20260925-l3-base-repeat-trn21-r2-full'
CONTROL_ROW_SHA='0f6b507896a008adca065aabbd8cfaf60f1720add3f6ee1632272601a93d5cc7'
SELECTOR_SHA='fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac'
KNOWN_TRAIN_PIDS={3102316,3102317,3102318,3102334,3102335,3102353,3102368,3102369,3102370,3102371}
checks=[]
def require(ok,name):
    if not ok:raise ValueError(name)
    checks.append(name)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def git(ref,path):return subprocess.check_output(['git','show',ref+':'+path])
def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('DUPLICATE_JSON_KEY:'+k)
        out[k]=v
    return out
def parse(raw):return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON:'+x)))
def finite(x):return type(x) in (int,float) and math.isfinite(x) and x>=0
def decode(raw,pin):
    require(len(raw)<=16*1024**2 and sha(raw)==pin,'bounded exact outer bundle')
    bundle=parse(raw);files={}
    require(type(bundle.get('files')) is dict and len(bundle['files'])<=256,'bounded archive entries')
    for name,item in bundle['files'].items():
        require(('text' in item)!=('base64' in item),'exclusive artifact encoding '+name)
        value=item['text'].encode() if 'text' in item else base64.b64decode(item['base64'],validate=True)
        require(sha(value)==item['sha256'] and len(value)==item['bytes'],'artifact hash/size '+name);files[name]=value
    require(sum(map(len,files.values()))==bundle['raw_bytes'] and bundle.get('file_count',len(files))==len(files),'archive totals')
    return bundle,files
def tagged(log,prefix):
    values=[parse(x[len(prefix):]) for x in log.splitlines() if x.startswith(prefix)]
    require(len(values)==1,'one '+prefix);return values[0]
def evaluation(log):
    values=[]
    for match in re.finditer(r'\{',log):
        try:value,_=json.JSONDecoder().raw_decode(log[match.start():])
        except ValueError:continue
        if isinstance(value,dict) and 'total_nats' in value and 'val_bpb' in value:values.append(value)
    require(len(values)==1,'one final complete eval totals')
    e=values[0]
    require(type(e['num_tokens']) is int and e['num_tokens']==20971520 and type(e['num_bytes']) is int and e['num_bytes']==83462186,'exact public20M token/byte totals')
    require(finite(e['total_nats']) and e['total_nats']>0 and finite(e['val_bpb']),'finite positive eval totals')
    own=e['total_nats']/(e['num_bytes']*math.log(2))
    require(own==e['val_bpb'],'independently recomputed BPB')
    require(e['eval_over_timeout'] is False and finite(e['eval_seconds']) and e['eval_seconds']<e['eval_timeout']==1200,'eval timeout gate')
    return e
def timing(h):
    require(h['returncode']==0 and h['mode']=='train','training return/mode')
    observations=h['observations'];n=h['step_boundaries']
    require(type(n) is int and n>=5 and len(observations)==n and [r['step'] for r in observations]==list(range(n)),'all consecutive observed boundaries')
    times=[r['observed_seconds'] for r in observations]
    require(all(finite(x) for x in times) and all(a<=b for a,b in zip(times,times[1:])),'finite monotonic observations')
    fifth=times[4];allowance=min(fifth,900.0);wall=h['wall_seconds'];charge=h['charged_through_process_exit']
    require(finite(wall) and finite(charge) and wall>=times[-1] and allowance==h['startup_allowance_observed'],'measured wall/capped allowance')
    require(wall-allowance==charge and charge<=1800 and h['charged_through_process_exit_within_budget'] is True,'official charge through exit including excess')
    return dict(steps=n,fifth_seconds=fifth,wall_seconds=wall,charged_through_exit_seconds=charge,allowance_seconds=allowance)
def closed_result(bundle,f,b,post,ck,lock_inode,known_pids):
    # Same established final36 schema; only main() supplies immutable bcbd binding.
    rec=lambda name:parse(f[name]);tr=f['training.log'].decode();el=f['fresh-public20m.log'].decode()
    require(bundle['root']==b['root'] and rec('NATIVE_BINDING.disabled.json')==b,'exact run/root/binding')
    for name,cmd,env in [('training-launch.json','command','training_environment'),('fresh-public20m-launch.json','evaluation_command','evaluation_environment')]:
        launch=rec(name);require(launch['command']==b[cmd] and launch['environment']==b[env] and all(launch[k]==b[k] for k in ['label','source_sha256','prepare_sha256','harness_sha256']),'bound launch '+name)
    for log,mode,sandbox in [(tr,'train',b['sandbox']),(el,'eval',b['eval_sandbox'])]:
        iso=tagged(log,'ISOLATED_SUBMISSION ')
        require(iso['mode']==mode and iso['sandbox']==sandbox and iso['files']==['prepare.py','train.py'] and iso['train_sha256']==b['source_sha256'] and iso['prepare_sha256']==b['prepare_sha256'],'isolated source/prepare '+mode)
        require(shlex.split(iso['launch_command'])==b['training_command'],'isolated science command '+mode)
        require(iso['fresh_empty_sdk_caches']=={},'research caches, not fresh cold '+mode)
    require(tagged(tr,'REHEARSAL_COMMAND ')==b['training_command'],'actual training argv')
    h=rec('train/rehearsal.json');facts=timing(h)
    require(tagged(tr,'SUBMISSION_REHEARSAL ')=={k:v for k,v in h.items() if k!='observations'},'train summary exact')
    require([int(x) for x in f['train/steps.jsonl'].splitlines()]==list(range(facts['steps'])),'step log complete')
    require(f['run-exit-code.txt'].strip()==b'0' and re.search(r'^over_budget:\s*False$',tr,re.M),'source exit and over_budget False')
    for name,expected in [('num_steps',facts['steps']),('world_size',4),('lnc',2),('seq_len',1024)]:
        m=re.search(r'^'+name+r':\s*(\d+)$',tr,re.M);require(m is not None and int(m[1])==expected,'source summary '+name)
    loss_strings=re.findall(r'^step \d+ \| loss (\S+) \|',tr,re.M)
    require(bool(loss_strings) and all(finite(float(x)) for x in loss_strings),'all printed losses finite')
    checkpoint=rec('checkpoint.json')
    require(checkpoint['sha256']==h['checkpoint_sha256'] and bool(re.fullmatch('[0-9a-f]{64}',checkpoint['sha256'])) and checkpoint['bytes']==h['checkpoint_bytes']>0,'retained checkpoint')
    expected=[b['evaluation_command'][5],'prepare.py','eval-public','--train-py','train.py','--checkpoint',b['sandbox']+'/out/final.pt','--device','neuron','--batch-size','1','--seq-len','1024','--eval-tokens','20971520','--timeout-seconds','1200']
    require(tagged(el,'REHEARSAL_COMMAND ')==expected,'actual fresh20M B1/context1024 argv')
    ev=evaluation(el);eh=rec('eval/rehearsal.json')
    require(eh==tagged(el,'SUBMISSION_REHEARSAL ') and eh['returncode']==0 and eh['mode']=='eval' and f['eval-exit-code.txt'].strip()==b'0','fresh eval complete exit0')
    require(finite(eh['wall_seconds']) and ev['eval_seconds']<eh['wall_seconds']<=1800,'eval internal versus observer wall')
    terminal,outer=rec('terminal.json'),rec('outer-flock-exit.json')
    require(terminal['label']==b['label'] and terminal['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and terminal['eval_exit_code']==outer['returncode']==0 and terminal['no_automatic_next_job'] is True and terminal['submission_authorized'] is False and outer['command']==b['outer_command'],'final terminal/outer success, no adoption')
    g1,g2=rec('under-lock-idle.json'),rec('eval-under-lock-idle.json')
    require(g1['lock']==g2['lock'] and g1['lock']['device']==66305 and g1['lock']['inode']==lock_inode,'same original lifetime lock')
    for g in (g1,g2):
        require(g['status']=='PASS' and g['lock']['verified_existing_exclusive_lock'] is True and g['ps']['returncode']==g['neuron_ls']['returncode']==0 and not g['ps']['busy'] and not g['neuron_ls']['process_pids'],'fresh in-lock ps/neuron-ls')
    pids=set(post['observed_pids']);identities=post['all_process_identity_rows']
    require(known_pids<=pids and len(pids)==len(post['observed_pids'])==len(post['pid_rows']) and pids=={r['pid'] for r in post['pid_rows']} and all(r['exists'] is False for r in post['pid_rows']),'all known and final observed PIDs absent')
    require(identities and len({r['pid'] for r in identities})==len(identities) and not any(r[k] in pids for r in identities for k in ['pid','ppid','pgid','sid']),'unfiltered process census no related identities')
    require(not any(re.search('neuronx?-cc|neuron-cc|neuronx-compile',r['comm']) for r in identities),'no compiler-named process in final census')
    require(post['all_observed_pids_absent'] is True and not post['related_rows'] and post['ps_rc']==post['lock_competitor_rc']==0 and not post['ps_stderr'] and not post['neuron_ls_stderr'] and post['neuron_device_idle'] is True and all(not d['neuron_processes'] for d in post['neuron_ls']),'explicit final process/device quiescence')
    require(post['lock_device']==66305 and post['lock_inode']==lock_inode,'original postflight lock')
    require(ck['status']=='PASS' and ck['unchanged_after_eval'] is True and ck['before']==ck['after'] and ck['sha256']==ck['expected_training_sha256']==checkpoint['sha256'] and ck['bytes_hashed']==ck['before']['size']==checkpoint['bytes'] and ck['path']==b['sandbox']+'/out/final.pt','actual post-eval checkpoint hash/stat')
    require(ck['lock']['device']==66305 and ck['lock']['inode']==lock_inode and ck['lock']['exclusive_held_through_hash'] is True and ck['idle_checks']['ps']['returncode']==ck['idle_checks']['neuron_ls']['returncode']==0 and not ck['idle_checks']['ps']['busy'] and not ck['idle_checks']['neuron_ls']['process_pids'],'checkpoint hash under original lock and idle')
    inventory={}
    for rank in range(4):
        rows=list(csv.DictReader(io.StringIO(f[f'metrics-training/metric_table_graph_stats_rank{rank}.csv'].decode())))
        require(rows and all({'graph_id','graph_name','graph_node_count','torch_neuronx_compile_us'}<=set(r) for r in rows),'closed graph CSV rank '+str(rank))
        require(all(finite(float(r['torch_neuronx_compile_us'])) and int(r['graph_node_count'])>=0 for r in rows),'finite graph inventory rank '+str(rank))
        inventory[str(rank)]=len(rows)
    dt=datetime.datetime.fromisoformat
    t_end=re.search(r'^TRAINING_END_UTC=(.+)$',tr,re.M).group(1)
    e_start=re.search(r'^FRESH-PUBLIC20M_START_UTC=(.+)$',el,re.M).group(1)
    e_end=re.search(r'^FRESH-PUBLIC20M_END_UTC=(.+)$',el,re.M).group(1)
    require(dt(t_end)<dt(checkpoint['utc'])<dt(g2['started_utc'])<=dt(g2['ended_utc'])<dt(rec('fresh-public20m-launch.json')['utc'])<=dt(e_start)<dt(e_end)<=dt(terminal['utc'])<dt(outer['ended_utc'])<dt(post['utc'])<=dt(ck['started_utc'])<dt(ck['ended_utc']),'ordered train/checkpoint/fresh eval/postflight/hash')
    return dict(**facts,evaluation=ev,checkpoint=checkpoint,terminal_utc=terminal['utc'],graph_csv_rows=inventory,observed_postflight_pids=len(pids),process_census_rows=len(identities),cache_hits='UNPROVEN',cold_full_proven=False)
def pins():
    braw=git('7da2559f',P+'NATIVE_BINDING.disabled.json');require(sha(braw)=='bc492cf2fcf863a615835008dd7598cf03b58fa2f5c157661a2e5022434d6909','bcbd frozen binding')
    b=parse(braw);require(b['label']==LABEL and b['host']=='trn21' and b['source_sha256']==SOURCE,'exact intended run')
    require(sha(git('40b7d400','research/v2/verification/20260926-basebcbd-full-trn21-assignment.md'))=='57c59917b8a41f68cbe775fdae461d9d7582fe0d3e52b299057d0f2c4184c33c','exact release')
    live,_=decode(git('7aa13bd021e36f7225ac05abb13ba68188be9def','research/v2/operator/20260926-trn21-basebcbd-full/launch/first-native-evidence.json'),'eefd25b143a4a266fb23bca84158bfe73386b633f4becdb13644902bba1ff669')
    require(live['root']==b['root'] and {int(x.split()[0]) for x in live['job_lines']}==KNOWN_TRAIN_PIDS,'known training PIDs authenticated from first live archive')
    require(sha(git(b['source_commit'],b['source_git_path']))==SOURCE,'source Git pin')
    for name,field in [('host_driver.py','driver_sha256')]:require(sha(git('7da2559f',P+name))==b[field],'driver pin')
    require(sha(git('7da2559f','prepare.py'))==b['prepare_sha256'] and sha(git('00f2ca7f','research/scratch/submission_rehearsal.py'))==b['harness_sha256'],'unchanged prepare/harness')
    for name,field in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:require(sha(git('7da2559f','research/v2/operator/20260926-common-host-idle-guard/'+name))==b[field],'shared support '+name)
    ledger=git('1c879c2b','research/v2/ledger.jsonl');lines=[x for x in ledger.splitlines(keepends=True) if parse(x).get('label')==CONTROL]
    require(len(lines)==1 and sha(lines[0])==CONTROL_ROW_SHA,'genuine unchanged control row');control=parse(lines[0])
    require(control['kind']=='control' and control['host']=='trn21' and control['train_py_sha256']==BASE and control['public_20m']==0.9899528600562257,'control role/source/score')
    cp='research/v2/operator/20260925-trn21-l3-base-repeat-r2/completed-eval/'
    clog=git('1c879c2b',cp+'fresh-public20m.log')
    require(sha(clog)=='37cf2f4d0d9ed52ebe6baa0234da6114167abbb12566d3f287893a3f378f70df','actual control raw eval log')
    require(evaluation(clog.decode())['val_bpb']==control['public_20m'],'control BPB independently recalculated')
    selector=git('1c879c2b','workflow/tools/decide.py');require(sha(selector)==SELECTOR_SHA,'unchanged selector code')
    return b,control,ledger,selector
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw-commit');ap.add_argument('--bundle-sha');ap.add_argument('--decide',action='store_true');a=ap.parse_args()
    b,control,ledger,selector=pins()
    if not a.raw_commit and not a.bundle_sha and not a.decide:
        return dict(status='ONLY_PENDING_REAL_RAW',label=LABEL,source_sha256=SOURCE,control_label=CONTROL,control_bpb=control['public_20m'],candidate_bpb=None,selector_verdict=None,required=['immutable final raw commit','exact final bundle SHA','completed train+fresh20M and all4 graph CSVs','explicit-process-postflight.json with all observed compiler/eval IDs','post-eval-checkpoint-hash.json'],checks=len(checks),ledger_written=False)
    require(bool(re.fullmatch('[0-9a-f]{40}',a.raw_commit or '')) and bool(re.fullmatch('[0-9a-f]{64}',a.bundle_sha or '')),'explicit immutable raw commit and bundle SHA required')
    bundle,f=decode(git(a.raw_commit,RAW_ROOT+'results-bundle.json'),a.bundle_sha)
    require(f['NATIVE_BINDING.disabled.json']==git('7da2559f',P+'NATIVE_BINDING.disabled.json') and f['MANIFEST.sha256.json']==git('7da2559f',P+'MANIFEST.sha256.json'),'actual frozen binding/manifest bytes')
    manifest=parse(f['MANIFEST.sha256.json'])
    for name,pin in manifest['files'].items():require(sha(git('7da2559f',P+name))==pin,'packet manifest '+name)
    for name,key in [('training_environment','NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS')]:require(b[name][key]=='1800','explicit fixed budget')
    require('--initial-cache-refresh' in b['training_command'] and '--recursive-schedule' in b['training_command'] and '--startup-adjusted-schedule' not in b['training_command'] and '--unet-8x1024' not in b['training_command'],'only selected base initial-cache factor')
    pr=git(a.raw_commit,RAW_ROOT+'explicit-process-postflight.json');cr=git(a.raw_commit,RAW_ROOT+'post-eval-checkpoint-hash.json')
    facts=closed_result(bundle,f,b,parse(pr),parse(cr),596823,KNOWN_TRAIN_PIDS)
    row=dict(schema='trn-run-record-v2',label=LABEL,arm_id='base-initial-cache-bcbd',kind='initial_cache_arm',run_purpose='quality',host='trn21',timestamp_utc=facts['terminal_utc'],base_sha256=BASE,train_py_sha256=SOURCE,source_commit=b['source_commit'],status='evaluated',git_dirty=False,git_dirty_scope='Authenticated exact scientific bytes, not a globally clean remote checkout',argv=b['training_command'][4:],env=b['training_environment'],training_budget_seconds=1800,over_budget=False,steps=facts['steps'],charged_through_exit_seconds=facts['charged_through_exit_seconds'],startup_measured_seconds=facts['fifth_seconds'],cold_startup_proven=False,causality_passed=None,eval_tokens=20971520,eval_seq_len=1024,public_20m=facts['evaluation']['val_bpb'],checkpoint_sha256=facts['checkpoint']['sha256'],submission_eligible=False)
    result=dict(status='AUTHENTICATED_RESULT_NOT_ADOPTED',facts=facts,comparison_control=control,delta_vs_genuine_control=row['public_20m']-control['public_20m'],secondary_historical_schedule_arm_bpb=0.9900560897366009,historical_rows_relabelled=False,ledger_candidate_only=row,ledger_written=False,source_adopted=False,submission_authorized=False,raw_commit=a.raw_commit,bundle_sha256=a.bundle_sha,postflight_sha256=sha(pr),posteval_checkpoint_receipt_sha256=sha(cr),checks=len(checks))
    if a.decide:
        ns={'__name__':'readonly_exact_selector','__file__':'workflow/tools/decide.py'};exec(compile(selector,'immutable-decide.py','exec'),ns)
        require(ns['invalid_reason'](row) is None,'valid authenticated quality candidate')
        rows=[parse(x) for x in ledger.splitlines() if x.strip()]
        controls=ns['controls_for'](rows,'trn21',BASE)
        require([x['label'] for x in controls]==[CONTROL],'only pinned genuine control used')
        result['separate_unchanged_selector']=ns['decide_arm']([r for r in rows if r['label']!=LABEL]+[row],row['arm_id'])
        result['selector_sha256']=SELECTOR_SHA;result['selector_ledger_snapshot']='1c879c2b'
    return result
if __name__=='__main__':
    try:out=main()
    except Exception as e:
        print(json.dumps(dict(status='HOLD_MISSING_OR_INVALID_FINAL_EVIDENCE',error=type(e).__name__+': '+str(e),checks=len(checks),ledger_written=False),allow_nan=False));raise SystemExit(2)
    print(json.dumps(out,indent=2,allow_nan=False))
