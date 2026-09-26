import hashlib,json,math,struct
from pathlib import Path
base=Path(__file__).resolve().parent.parent
raw=Path('/home/poojith-devan/Desktop/Trainium/research/v2/operator/20260926-trn22-clock36-checkpoint-mask-r3/completed')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((raw/'results-bundle.json').read_text());post=json.loads((raw/'explicit-process-postflight.json').read_text())
old=json.loads((base/'results-bundle.json').read_text());plan=json.loads((base/'packet/CPU_PLAN.disabled.json').read_text());pins=json.loads((base/'packet/INPUTS.json').read_text())
checks=[]
def need(value,name):
    if not value:raise RuntimeError('REFUSE:'+name)
    checks.append(name)
def entry(bundle,name):return json.loads(bundle['files'][name]['text'])
for label,bundle in [('r3',b),('r2',old)]:
    for name,item in bundle['files'].items():
        data=item['text'].encode()
        need(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],label+':raw:'+name)
need(sha(base/'results-bundle.json')=='d2298e9cafa63d98e9aa54c8b324e42a5ac345f20f4144b811112900ba2b2a08','original_bundle_pin')
need(entry(b,'inputs-before.json')==entry(b,'inputs-after.json')==pins==entry(b,'INPUTS.json'),'all16_inputs_exact_unchanged')
staged=entry(b,'operator-preflight.json')['staged_hashes']
for name,path in [('compare_checkpoints.py',base/'new_helper.py'),('host_driver.py',base/'packet/host_driver.py'),('CPU_PLAN.disabled.json',base/'packet/CPU_PLAN.disabled.json'),('INPUTS.json',base/'packet/INPUTS.json'),('idle_guard.py',base/'packet/idle_guard.py'),('dispatch_driver.py',base/'packet/dispatch_driver.py')]:need(staged[name]==sha(path),'staged:'+name)
need(entry(b,'launch.json')['command']==entry(b,'process.json')['command']==plan['command'],'exact_sole_child')
need(entry(b,'outer-flock-exit.json')['command']==plan['outer_command'],'outer_command')
need(entry(b,'outer-flock-exit.json')['returncode']==entry(b,'checkpoint-exit.json')['exit_code']==entry(b,'terminal.json')['exit_code']==0,'all_exit_zero')
for name in ('under-lock-idle.json','launch-under-lock-idle.json','post-under-lock-idle.json'):
    r=entry(b,name)
    need(r['status']=='PASS' and r['lock']['device']==66305 and r['lock']['inode']==832589 and r['lock']['verified_existing_exclusive_lock'] is True,'lock:'+name)
    need(r['ps']['returncode']==r['neuron_ls']['returncode']==0 and r['ps']['busy']==r['neuron_ls']['process_pids']==[],'idle:'+name)
res=entry(b,'checkpoint-result.json')
need(res['status']=='FULL_PARAMETER_CPU_CHECKPOINT_AND_LOGIT_IDENTITY' and res['states']==65 and res['parameters']==113256724,'result_status_full')
need(res['original_r2_qualification']=='FAILED_PRESERVED' and res['checkpoint_only_retry'] is True and all(res[k] is False for k in ('native_correctness','quality_evidence','on_off_equivalence_claimed')),'scope')
need([r['name'] for r in res['records']]==['reference','default','off','on'],'four_arms')
# Independently stream exact little-endian FP32 reference bytes, one row at a time.
mask_hash={}
for window in (8192,4096):
    d=hashlib.sha256();z=struct.pack('<f',0.);inf=struct.pack('<f',-float('inf'))
    for i in range(8192):
        left=max(0,i-window+1);visible=i-left+1
        d.update(inf*left+z*visible+inf*(8191-i))
    mask_hash[window]=d.hexdigest()
states=[]
for arm in res['records']:
    name=arm['name'];q=arm['trained_model_qualification'];state=q['named_parameters_and_all_buffers']
    ident=entry(old,name+'/checkpoint-identity.json');request=entry(old,name+'/request.json')
    need(arm['checkpoint_sha256']==ident['sha256'] and arm['source_sha256']==request['source_sha256'],'source_checkpoint:'+name)
    need(q['route']=='ordinary_unpatched_cpu_load_for_eval' and q['repeated_bytes_equal'] is True and q['model_state_equal_after_each_call'] is True and q['organizer_causality_check_called'] is False,'purity_state:'+name)
    need(len(state)==q['state_tensor_count']==69 and sum(k.startswith('parameter:') for k in state)==65,'all_state:'+name)
    need(sum(math.prod(v['shape']) for k,v in state.items() if k.startswith('parameter:'))==113256724,'parameter_elements:'+name)
    need(q['mask_configuration']=={'sequence_len':8192,'full_window':8192,'short_window':4096,'saved_sequence_len':1024,'eval_min_sequence_len':8192,'restored_short_window_frac':.5},'restored_config:'+name)
    need(len(q['outputs'])==3 and q['outputs'][0]==q['outputs'][1] and all(o['shape']==[1,256,8192] and o['dtype']=='torch.float32' and o['requires_grad'] is False for o in q['outputs']),'output_repeat:'+name)
    c=q['causality'];need(c['max_abs_diff']==0. and c['tolerance']==.01 and c['passed'] is True and c['synthetic_only'] is True and c['cut']==128 and c['prefix_stop']==127,'causality:'+name)
    for key,w in [('mask_full',8192),('mask_short',4096)]:
        m=state['buffer:'+key];count=sum(min(i+1,w) for i in range(8192))
        need(m['sha256']==mask_hash[w] and m['window']==w and m['zero_count']==count and m['negative_infinity_count']==8192**2-count and m['shape']==[1,1,8192,8192] and m['dtype']=='torch.float32' and m['requires_grad'] is False and m['mask_semantics_validated'] is True and m['nan_count']==m['positive_infinity_count']==0,'independent_mask_bytes:'+name+':'+key)
    g=entry(old,name+'/g2.json')
    need(g['verdict']=='PASS' and g['torchrun_rc']==0 and g['nproc']==4 and g['steps']==3 and len(g['ranks'])==4,'original_g2:'+name)
    need(all(r['status']=='PASS' and r['n_params']==r['n_changed']==65 and not r['cross_rank_mismatches'] and all(math.isfinite(v) for v in r['losses']) for r in g['ranks']),'original_g2_ranks:'+name)
    if name!='on':states.append(state)
need(states[0]==states[1]==states[2],'three_arm_state_records_equal')
need(entry(old,'checkpoint-compare-exit.json')['exit_code']==1 and 'RuntimeError: NONFINITE_TENSOR' in old['files']['checkpoint-compare.stderr']['text'],'r2_failure_preserved')
pids=post['observed_pids'];need(len(pids)==len(set(pids))==5,'five_processes')
need(post['all_observed_pids_absent'] is True and {r['pid'] for r in post['pid_rows']}==set(pids) and all(r['exists'] is False for r in post['pid_rows']),'pid_absence')
identities=post['observed_starttime_records'];need(set(map(int,identities))==set(pids) and all(r['starttime_ticks']>0 for r in identities.values()),'five_starttimes')
need(pids[0]==entry(b,'operator-launch.json')['dispatch_pid'] and pids[2]==entry(b,'process.json')['parent_pid'] and pids[3]==entry(b,'process.json')['pid'],'process_links')
for prev,pid in zip(pids,pids[1:]):need(identities[str(pid)]['ppid']==prev,'ancestry:'+str(pid))
rows=post['all_process_identity_rows']
need(not any(r['pid'] in pids or r['ppid'] in pids or r['pgid']==pids[0] or r['sid']==pids[0] for r in rows),'full_census_no_descendants')
need(not any(r['comm'].startswith(('neuron','walrus','torchrun')) for r in rows) and post['related_rows']==[],'compiler_native_census')
need(post['ps_rc']==0 and post['ps_stderr']=='' and post['neuron_device_idle'] is True and len(post['neuron_ls'])==1 and post['neuron_ls'][0]['neuron_processes']==[] and post['neuron_ls_stderr']=='','device_idle')
need(post['fuser_rc']==1 and post['fuser_stdout']==post['fuser_stderr']=='','device_no_fuser_holders')
need(post['lock_device']==66305 and post['lock_inode']==832589 and post['lock_competitor_rc']==0,'original_lock_free')
print(json.dumps({'verdict':'SCOPED ACCEPT','checks':checks,'count':len(checks),'raw_bundle_sha256':sha(raw/'results-bundle.json'),'raw_postflight_sha256':sha(raw/'explicit-process-postflight.json'),'test_sha256':sha(Path(__file__)),'original_r2':'FAILED_PRESERVED','independent_stream_mask_hashes':mask_hash,'scope':'CPU retained-checkpoint diagnostic only; original CPU shim G2 remains separate; no native, BPB, ON-OFF equivalence, submission, adoption or launch authority'},indent=2))
