import hashlib,json,math,struct
from pathlib import Path
import torch
torch.set_num_threads(1)
P=Path('/home/poojith-devan/checkpoint-review-packet-Abknyi/actual-helper')
pins={'results-bundle.json':'c6243eac08765729e39976dd6294ec847753c372f185159997e621bd1f72f9d7','explicit-process-postflight.json':'a4b28ec8656563a659b432756bc63267f6ef6a0c71f1febcb9c7a0483992d27a'}
checks=[]
def ck(n,v):
    if not v:raise RuntimeError(n)
    checks.append(n)
for n,s in pins.items():ck('pin:'+n,hashlib.sha256((P/n).read_bytes()).hexdigest()==s)
b=json.loads((P/'results-bundle.json').read_text());post=json.loads((P/'explicit-process-postflight.json').read_text())
def txt(n):
    f=b['files'][n];v=f['text'].encode();ck('raw:'+n,len(v)==f['bytes'] and hashlib.sha256(v).hexdigest()==f['sha256']);return f['text']
def read(n):return json.loads(txt(n))
result=read('checkpoint-result.json');inputs=read('INPUTS.json');before=read('inputs-before.json');after=read('inputs-after.json');pre=read('operator-preflight.json');launch=read('launch.json');proc=read('process.json')
old=json.loads((P.parent/'actual-full-g2/retained-input-identities.json').read_text());oldpins={k:{'bytes':v['bytes'],'sha256':v['sha256']} for k,v in old.items()}
ck('exact17_frozen_before_after_prior',len(before)==17 and before==after==inputs['files']==oldpins)
ck('retrieved17',len(b['retained_input_identities'])==17 and {k:{'bytes':v['bytes'],'sha256':v['sha256']} for k,v in b['retained_input_identities'].items()}==oldpins)
ck('manifest_pin',result['input_manifest']['sha256']==b['files']['INPUTS.json']['sha256']==pre['staged_hashes']['INPUTS.json'] and result['input_manifest']['bytes']==b['files']['INPUTS.json']['bytes'])
ck('helper_pin',pre['staged_hashes']['compare_checkpoints.py']=='65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8')
ck('exact_command',proc['command']==launch['command'] and launch['command'][-1]==result['input_manifest']['sha256'] and launch['command'][1:5]==['--foreground','--signal=TERM','--kill-after=60s','600s'])
ck('resource_preflight',pre['available_ram_bytes']>=64*1024**3 and pre['free_bytes']>=10.5*1024**3+pre['projected_footprint_bytes'] and pre['lock_device']==66305 and pre['lock_inode']==596823)
ck('normal_exit',read('terminal.json')['exit_code']==read('checkpoint-exit.json')['exit_code']==read('outer-flock-exit.json')['returncode']==0)
stderr=txt('checkpoint-result.stderr')
ck('stderr_registration_warning_only','Traceback' not in stderr and 'dispatch key: PrivateUse1' in stderr and 'operator: aten::gather.out' in stderr)
ck('scope',result['status']=='CPU_CHECKPOINT_SYNTHETIC_QUALIFICATION_COMPLETE' and result['inputs_unchanged_after'] is True and result['threads']==1 and not result['native_correctness'] and not result['quality_or_submission_claim'])
ck('g2_pair',result['pair']['reference']==result['pair']['candidate'])
shapes=[(1,256),(2,256),(3,256),(1,257),(2,257),(3,257)]
params={};observations={};masks={}
def tr(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest(),'requires_grad':False}
# An independent row-byte mask formula, no model or 8192-square allocation.
for name,window in [('mask_full',8192),('mask_short',4096)]:
    digest=hashlib.sha256();zero=struct.pack('<f',0.);hidden=struct.pack('<f',float('-inf'));count=0
    for row in range(8192):
        start=max(0,row-window+1);visible=row-start+1;count+=visible
        digest.update(hidden*start+zero*visible+hidden*(8192-row-1))
    masks[name]={'sha256':digest.hexdigest(),'zero_count':count,'negative_infinity_count':8192**2-count}
for arm,a in result['arms'].items():
    ck(arm+'_source_pin',a['source_sha256']==before[inputs['root']+'/'+arm+'/train.py']['sha256'])
    ck(arm+'_checkpoint',a['checkpoint']==inputs['arms'][arm])
    ck(arm+'_ordinary',a['dtype_override'] is False and a['loader_class']==('GPT' if arm=='candidate' else '_SubmissionEvalGPT'))
    ck(arm+'_six_shapes',[tuple(s['shape']) for s in a['shapes']]==shapes)
    baseline=a['shapes'][0]['state_before'];params[arm]={k:v for k,v in baseline.items() if k.startswith('parameter:')}
    ck(arm+'_65_full113M',len(params[arm])==65 and sum(math.prod(v['shape']) for v in params[arm].values())==113256724)
    ck(arm+'_frozen_fp32',all(v['dtype']=='torch.float32' and v['requires_grad'] is False for v in params[arm].values()))
    ck(arm+'_g2_parameter_prefixes',all(v['sha256'][:16]==result['pair'][arm]['digests'][k.removeprefix('parameter:')] for k,v in params[arm].items()))
    ck(arm+'_buffers',set(k for k in baseline if k.startswith('buffer:'))==({'buffer:cos','buffer:sin'} if arm=='candidate' else {'buffer:cos','buffer:sin','buffer:mask_full','buffer:mask_short'}))
    if arm=='reference':
        for name,calc in masks.items():
            actual=baseline['buffer:'+name]
            ck('independent_full_mask_'+name,all(actual[k]==v for k,v in calc.items()) and actual['nan_count']==actual['positive_infinity_count']==0 and actual['shape']==[1,1,8192,8192])
    for s in a['shapes']:
        key=tuple(s['shape']);batch,length=key;tag=arm+str(key)
        x=torch.randint(1,8192,(batch,length),generator=torch.Generator().manual_seed(7));cut=length//2;suffix=x.clone();suffix[:,cut:]=x[:,cut:].flip(1)
        expected={'original':x,'repeat':x,'reversed_suffix':suffix};expected.update({'row'+str(i):x[i:i+1] for i in range(batch)})
        calls={c['call']:c for c in s['calls']}
        ck(tag+'_calls',len(s['calls'])==batch+3 and set(calls)==set(expected))
        ck(tag+'_own_seed7_input',s['input']==tr(x) and all(calls[k]['input']==tr(v) for k,v in expected.items()))
        ck(tag+'_nondegenerate',all(not torch.equal(x[i,cut:],suffix[i,cut:]) for i in range(batch)))
        ck(tag+'_output_shapes',all(c['output']['shape']==[expected[k].shape[0],length,8192] and c['output']['dtype']=='torch.float32' for k,c in calls.items()))
        original=calls['original']['output']
        ck(tag+'_own_repeat_hash',calls['repeat']['output']==original)
        for comparison in ('repeat','rowwise'):
            c=s[comparison];ck(tag+'_'+comparison,c['actual']==c['expected']==original and c['max_abs_diff']==c['relative_rms']==0 and c['exact_bytes'] is True)
        c=s['causality'];ck(tag+'_prefix_zero',c['max_abs_diff']==0 and c['prefix_exact'] is True and c['tolerance']==.01 and c['cut']==cut and c['prefix_stop']==cut-1)
        ck(tag+'_unchanged_all_state',s['state_before']==baseline and s['state_unchanged_after_every_forward'] is True and s['ordinary_loader'] is True)
        if arm=='reference':observations[key]=s
        else:
            ref=observations[key];ck(tag+'_own_all_call_cross_hashes',s['calls']==ref['calls'])
            c=s['cross_source'];ck(tag+'_cross_source',c['actual']==c['expected']==original and c['max_abs_diff']==c['relative_rms']==0 and c['exact_bytes'] is True)
ck('own65_full_parameter_records_equal',params['reference']==params['candidate'])
ck('rotary_buffers_equal',all(result['arms']['reference']['shapes'][0]['state_before']['buffer:'+n]==result['arms']['candidate']['shapes'][0]['state_before']['buffer:'+n] for n in ('cos','sin')))
rows=post['complete_process_census'];pids=set(post['observed_pids']);related=[r for r in rows if any(r[k] in pids for k in ('pid','ppid','pgid','sid'))]
ck('254_unique_census',len(rows)==len({r['pid'] for r in rows})==254)
ck('five_absent',len(pids)==5 and not related and {r['pid'] for r in post['pid_rows']}==pids and all(r['exists'] is False for r in post['pid_rows']))
ck('known_parent_ids',{proc['pid'],proc['parent_pid'],read('operator-launch.json')['dispatch_pid']}<=pids)
ck('idle_device_lock',post['ps_rc']==post['neuron_ls_rc']==0 and all(not n['neuron_processes'] for n in post['neuron_ls']) and post['fuser_rc']==1 and post['fuser_stdout']==post['fuser_stderr']=='' and post['lock_competitor_rc']==0 and post['lock_device']==66305 and post['lock_inode']==596823)
livepins={'observation-first.json':'89f5df0932a683ca053c3a87ddf187ec7a3215c482779b8da22abda127b9ab73','observation-terminalcheck.json':'d6be8a3826110e1fd7ab58d84e06c74e1e3ae1ccd49a616b07f9b1ce5d8fb865','observation-boundary0400.json':'740856d7c3a1f7711058d48b7011daf0d8e5a5db98f927d29fdf909669e407d2'}
live=[]
for name,pin in livepins.items():
    ck('live_pin:'+name,hashlib.sha256((P/name).read_bytes()).hexdigest()==pin)
    v=json.loads((P/name).read_text());ck('live_transport:'+name,v['returncode']==0 and v['stderr']=='')
    if name!='observation-boundary0400.json':
        rr=v['remote']['processes'];ck('five_live:'+name,len(rr)==5 and {r['pid'] for r in rr}==pids)
        live.append({r['pid']:{k:r[k] for k in ('pid','ppid','pgid','sid','starttime','argv','cwd')} for r in rr})
    else:ck('boundary_absent',v['remote']['processes']==[])
ck('live_identities_stable',live[0]==live[1])
chain=[3177118,3177119,3177120,3177150,3177151]
ck('live_chain',all(live[0][child]['ppid']==parent for parent,child in zip(chain,chain[1:])))
ck('live_session',all(r['pgid']==r['sid']==3177118 for r in live[0].values()))
ck('live_exact_timeout_argv',live[0][3177150]['argv']==' '.join(launch['command']))
ck('live_exact_helper_argv',live[0][3177151]['argv']==' '.join(launch['command'][5:]))
for name in ('under-lock-idle.json','launch-under-lock-idle.json','post-under-lock-idle.json'):
    v=read(name);ck('lifetime_lock:'+name,v['status']=='PASS' and v['lock']['device']==66305 and v['lock']['inode']==596823 and v['lock']['driver_pid']==3177120 and v['lock']['verified_existing_exclusive_lock'] is True)
print(json.dumps({'status':'ACCEPT_ACTUAL_CPU_CHECKPOINT_SYNTHETIC_ONLY','count':len(checks),'checks':checks,'input_pins':pins,'live_pins':livepins,'independent_mask_records':masks,'parameter_count':113256724,'tensor_count':65,'cross_source_call_comparisons':sum(b+3 for b,t in shapes),'shapes':shapes,'causal_max_abs':0,'exact_input_files_pre_post':17,'postflight_rows':254,'related_rows':related,'five_live_identities':live[0],'limitation':'Raw logits/checkpoint tensors and per-forward post-state snapshots are not in receipt; host-executed reviewed helper enforces these, own review compares complete recorded hashes and arithmetic. CPU synthetic only, no launch/native/BPB/adoption approval.'},indent=2))
