import hashlib,json,math,re
from pathlib import Path
from datetime import datetime
P=Path('/home/poojith-devan/checkpoint-review-packet-Abknyi/actual-full-g2')
pins={'results-bundle.json':'198f3307d82a71bc22664bd8638915f1b90b3f9d24fd8a953c280edfbaa3550d','explicit-process-postflight.json':'2aed3ac7d384477c03caa313a9a9e6e778aab7785a4c184846d4dcb59a9b407f','retained-input-identities.json':'9b54804666da69af0cf1771424c7024587fa9b337c5e1499288c0a11f45fb23c'}
checks=[]
def check(name,ok):
    if not ok:raise RuntimeError(name)
    checks.append(name)
for n,sha in pins.items():check('pin:'+n,hashlib.sha256((P/n).read_bytes()).hexdigest()==sha)
b=json.loads((P/'results-bundle.json').read_text());post=json.loads((P/'explicit-process-postflight.json').read_text());ids=json.loads((P/'retained-input-identities.json').read_text())
root='/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1'
sources={'reference':'36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb','candidate':'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43'}
requests={'reference':'38160400738af45274aad1c96f9a54e0c5ec1343ab9eb87f1279b46e4f5ba354','candidate':'64d0b756784eb8a8306d1fcd739c09c355d7611479f8da9f64bf2fb7ffc4eca3'}
prepare='5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb'
def text(n):
    f=b['files'][n];raw=f['text'].encode();check('raw_identity:'+n,len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256']);return f['text']
def read(n):return json.loads(text(n))
check('root',b['root']==root)
check('17_identities_match',len(ids)==17 and ids==b['retained_input_identities'])
for path,v in ids.items():
    check('stable:'+path,v['stable'] is True and v['bytes']>0 and bool(re.fullmatch('[0-9a-f]{64}',v['sha256'])))
    if path.startswith(root+'/') and path[len(root)+1:] in b['files']:
        f=b['files'][path[len(root)+1:]];check('retained:'+path,v['sha256']==f['sha256'] and v['bytes']==f['bytes'])
    if path.endswith('/prepare.py'):check('prepare_pin',v['sha256']==prepare)
    if path.endswith('/preflight_dryrun.py'):check('g2_pin',v['sha256']=='6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8')
g={};ranks={};process_pids=[]
for arm,source in sources.items():
    req=read(arm+'/request.json'); launch=read(arm+'/launch.json');term=read(arm+'/terminal.json');proc=read(arm+'/process.json')
    process_pids.append(proc['pid']);g[arm]=read(arm+'/g2.json');gg=g[arm]
    check(arm+'_request_pin',b['files'][arm+'/request.json']['sha256']==requests[arm])
    check(arm+'_source',req['source_sha256']==launch['source_sha256']==term['source_sha256']==gg['train_py_sha256']==ids[root+'/'+arm+'/train.py']['sha256']==source)
    check(arm+'_argv',gg['argv']==req['argv'] and launch['command']==req['command'])
    check(arm+'_full113M_request',req['parameters']==113256724 and req['parameter_tensors']==65 and req['shape']=='8x1024')
    check(arm+'_exit',term['exit_code']==0 and term['verdict']=='PASS' and gg['torchrun_rc']==0 and gg['verdict']=='PASS' and gg['timed_out'] is False)
    check(arm+'_g2_profile',gg['steps']==3 and gg['micro_batch']==[2,256] and gg['nproc']==4 and gg['prepare']['sha256']==prepare and gg['prepare']['stub'] is False)
    check(arm+'_thread_seed_env',all(launch['environment'][k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')) and launch['environment']['NEURON_COMPETITION_R1_SEED']=='42')
    checkpoint=read(arm+'/checkpoint-identity.json');record=b['checkpoint_records'][arm]
    check(arm+'_checkpoint_identity',checkpoint==record['record'] and record['observed']==ids[checkpoint['path']] and all(checkpoint[k]==record['observed'][k] for k in ('sha256','bytes')))
    check(arm+'_private_path',bool(re.fullmatch(re.escape(root+'/'+arm)+r'/tmp/g2-[A-Za-z0-9_-]+/out/final.pt',checkpoint['path'])))
    ranks[arm]={r['rank']:r for r in gg['ranks']};check(arm+'_four_ranks',len(gg['ranks'])==4 and set(ranks[arm])==set(range(4)))
    private=checkpoint['path'].removeprefix(root+'/').removesuffix('/out/final.pt')
    for r,rr in ranks[arm].items():
        raw=read(private+'/rank'+str(r)+'.json');check(arm+'_raw_rank_'+str(r),raw==rr)
        check(arm+'_rank_status_'+str(r),rr['status']=='PASS' and rr['n_params']==rr['n_changed']==65 and rr['unchanged_params']==[] and rr['cross_rank_mismatches']==[])
        check(arm+'_rank_losses_'+str(r),len(rr['losses'])==3 and all(math.isfinite(v) for v in rr['losses']))
        check(arm+'_plan_'+str(r),rr['plan_sha256']==gg['plan_sha256'] and sum(rr['plan_summary']['elements_by_owner'])==113256724)
    r0=ranks[arm][0];check(arm+'_checkpoint_check',r0['checkpoint_check']['ok'] and r0['checkpoint_check']['deterministic'] and r0['checkpoint_check']['path_exists'] and r0['checkpoint_check']['logits_shape']==[1,256,8192])
    check(arm+'_65_digests',len(r0['final_digests'])==65 and all(re.fullmatch('[0-9a-f]{16}',v) for v in r0['final_digests'].values()))
    log=text(private+'/torchrun.log');check(arm+'_full113M_log','num_params=113,256,724' in log)
d0=ranks['reference'][0]['final_digests'];d1=ranks['candidate'][0]['final_digests']
diff={k:(d0.get(k),d1.get(k)) for k in d0.keys()|d1.keys() if d0.get(k)!=d1.get(k)}
loss_delta={str(r):[a-c for a,c in zip(ranks['reference'][r]['losses'],ranks['candidate'][r]['losses'])] for r in range(4)}
check('own_65_digest_comparison',not diff and len(d0)==len(d1)==65)
check('own_12_loss_comparison',all(v==0 for ds in loss_delta.values() for v in ds))
check('own_plan_comparison',g['reference']['plan_sha256']==g['candidate']['plan_sha256'])
comp=read('standard-compare.json');check('standard_compare',comp['identical'] is True and comp['plan_equal'] is True and comp['losses_identical'] is True and comp['n_params_compared']==65 and comp['param_digest_differences']==[] and comp['state_keys_only_in_one']==[] and comp['loss_differences']=={} and comp['a_sha256']==sources['reference'] and comp['b_sha256']==sources['candidate'])
check('comparison_exit',read('standard-compare-exit.json')['exit_code']==0)
check('pair_terminal',read('terminal.json')['exit_code']==0)
outer=read('outer-flock-exit.json');operator=read('operator-launch.json');check('outer_exit',outer['returncode']==0)
census=post['complete_process_census'];observed=set(post['observed_pids']);check('census_253_unique',len(census)==253 and len({r['pid'] for r in census})==253)
check('recorded_pids_in_observed',set(process_pids+[operator['dispatch_pid']])<=observed)
related=[r for r in census if any(r[k] in observed for k in ('pid','ppid','pgid','sid'))]
busy=[r for r in census if any(k in r['comm'].lower() for k in ('torchrun','preflight_dry','host_driver','compare_g2'))]
check('own_no_observed_sessions_descendants',related==[])
check('own_no_busy_named_processes',busy==[])
check('pid_absence_rows',{r['pid'] for r in post['pid_rows']}==observed and all(r['exists'] is False for r in post['pid_rows']))
check('idle_device',post['ps_rc']==post['neuron_ls_rc']==0 and all(not d['neuron_processes'] for d in post['neuron_ls']) and post['fuser_rc']==1 and post['fuser_stdout']==post['fuser_stderr']=='')
check('original_lock_available',post['lock_device']==66305 and post['lock_inode']==596823 and post['lock_competitor_rc']==0)
elapsed=(datetime.fromisoformat(outer['ended_utc'])-datetime.fromisoformat(outer['started_utc'])).total_seconds()
print(json.dumps({'verdict':'ACCEPT_RETRIEVED_STANDARD_CPU_G2_ONLY','checks':checks,'count':len(checks),'receipt_sha256':pins,'own_digest_differences':diff,'own_loss_deltas':loss_delta,'rank_losses':{str(r):ranks['reference'][r]['losses'] for r in range(4)},'parameter_elements_owner_sum':113256724,'retained_input_count':len(ids),'checkpoint_identities':{a:b['checkpoint_records'][a]['record'] for a in sources},'outer_elapsed_seconds':elapsed,'final_census_rows':len(census),'own_related_rows':related,'own_busy_rows':busy,'worker_pid_limit':'Exact short-lived torchrun/worker PIDs were not captured; final census and recorded dispatcher/G2 parent IDs only. No live worker identity claim.','checkpoint_limit':'Host SHA/bytes identity receipts corroborated, checkpoint files not locally read or loaded; ordinary unpatched helper has not run.','standard_g2_shims':g['reference']['shims'],'native_or_bpb_approved':False,'launch_authorized':False},indent=2))
