"""Independent Supervisor authentication of actual child cold receipts; read-only."""
import csv, hashlib, io, json, math, re, shlex, subprocess
from pathlib import Path
from datetime import datetime
R=Path('/home/poojith-devan/Desktop/Trainium')
P=R/'research/v2/operator/20260926-trn22-clock36-recursive-cold-five'
Q=R/'research/v2/proposals/20260926-clock36-recursive-cold-five-trn22'
checks=[]
def need(ok,name):
    if not ok: raise RuntimeError('COLD_RESULT_REFUSE:'+name)
    checks.append(name)
def sha(data): return hashlib.sha256(data).hexdigest()
def read(p): return json.loads(p.read_text())
def auth(bundle,label):
    for name,f in bundle['files'].items():
        data=f['text'].encode()
        need(len(data)==f['bytes'] and sha(data)==f['sha256'],label+':'+name)
    if 'raw_bytes' in bundle: need(sum(f['bytes'] for f in bundle['files'].values())==bundle['raw_bytes'],label+':total')
b=read(Q/'CT_BINDING.disabled.json')
rawpath=P/'completed/results-bundle.json'; raw=read(rawpath)
postpath=P/'completed/explicit-process-postflight.json'; post=read(postpath)
need(sha(rawpath.read_bytes())=='46cf8f5896ceeaf0d5110127fbd7018056f6fcc8060a3a496b67ffe6eb6a8224','raw-pin')
need(sha(postpath.read_bytes())=='3790753cc33d3b08428dfa719adb920008d4868171c8ba19795cbefa2a0a6bcb','post-pin')
need(sha((Q/'CT_BINDING.disabled.json').read_bytes())=='db70a499f6179659c38acfc221f484199b8011c4afc4c2c59b98d3a26fabd21b','binding-pin')
auth(raw,'terminal');need(len(raw['files'])==20 and raw['raw_bytes']==60797,'raw-count-size')
get=lambda name:json.loads(raw['files'][name]['text'])
launch=get('training-launch.json'); pre=get('preflight.json'); guard=get('under-lock-idle.json')
for name in ('CT_BINDING.disabled.json','EXPERIMENT_PROPOSAL.json','MANIFEST.sha256.json'):
    need(raw['files'][name]['text'].encode()==(Q/name).read_bytes(),'frozen-'+name)
source=subprocess.check_output(['git','show',b['source_commit']+':'+b['source_git_path']],cwd=R)
need(sha(source)==b['source_sha256']=='4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea','source')
need(launch['command']==b['command'] and launch['environment']==b['training_environment'],'launch-exact')
for key in ('source_sha256','prepare_sha256','harness_sha256'):
    need(launch[key]==pre[key]==b[key],key)
need(pre['staged_hashes']['package/train.py']==b['source_sha256'],'staged-source')
need(pre['staged_hashes']['host_driver.py']==b['driver_sha256']==sha((Q/'host_driver.py').read_bytes()),'staged-driver')
need(guard['status']=='PASS' and guard['lock']['verified_existing_exclusive_lock'],'guard-status')
need(guard['lock']['device']==pre['lock_device']==66305 and guard['lock']['inode']==pre['lock_inode']==832589,'original-lock-pre')
need(guard['ps']['returncode']==guard['neuron_ls']['returncode']==0 and guard['ps']['busy']==guard['neuron_ls']['process_pids']==[],'idle-before')
need(datetime.fromisoformat(guard['ended_utc'])<datetime.fromisoformat(launch['utc']),'guard-before-launch')
need('--fresh-caches' in launch['command'],'fresh-caches-request')
cache={k:v for k,v in b['harness_injected_environment'].items() if 'CACHE_DIR' in k or k=='TORCH_NEURONX_DEBUG_DIR'}
need(len(cache)==len(set(cache.values()))==6 and set(pre['six_caches_absent_before_harness'])==set(cache.values()),'six-absent')
snapshots=[read(P/'launch/first-four-rank-evidence.json'),read(R/'research/v2/operator/20260926-dual-live-clock-child-0249/trn22.json'),read(R/'research/v2/operator/20260926-dual-live-clock-child-0254/trn22.json'),read(P/'progress/boundary-0255.json')]
observed={}
for i,s in enumerate(snapshots):
    auth(s,'snapshot'+str(i))
    observations=s.get('observations',[s])
    for observation in observations:
        for row in observation.get('processes',[]):
            need(row['starttime_ticks']>0,'identity-'+str(i)+'-'+str(row['pid']))
            if row['pid'] in observed:need(observed[row['pid']]['starttime_ticks']==row['starttime_ticks'],'stable-'+str(row['pid']))
            observed[row['pid']]=row
    need(s['lock_device']==66305 and s['lock_inode']==832589,'snapshot-original-lock'+str(i))
workers=[x for x in snapshots[1]['processes'] if x['ppid']==3246509]
need(len(workers)==4,'four-workers')
for w in workers:
    env=w['whitelisted_environment'];rank=env['RANK']
    need(env['LOCAL_RANK']==rank and env['WORLD_SIZE']=='4' and env['NEURON_LOGICAL_NC_CONFIG']=='2','rank-'+rank)
    need(w['cwd']==b['sandbox'] and shlex.split(w['argv'])[2:]==b['training_command'][3:],'argv-'+rank)
    need(all(env[k]==v for k,v in cache.items()) and env['NEURON_COMPILE_CACHE_URL']==env['NEURON_COMPILE_CACHE_DIR'],'cache-'+rank)
need({w['whitelisted_environment']['RANK'] for w in workers}=={'0','1','2','3'},'rank-set')
log=raw['files']['training.log']['text']
need('Traceback (most recent call last)' not in log and '\ncheck passed\n' in log,'clean-log')
iso=json.loads(next(line[len('ISOLATED_SUBMISSION '):] for line in log.splitlines() if line.startswith('ISOLATED_SUBMISSION ')))
need(iso['train_sha256']==b['source_sha256'] and iso['prepare_sha256']==b['prepare_sha256'],'isolated-bytes')
need(shlex.split(iso['launch_command'])==b['training_command'],'isolated-argv')
steps=re.findall(r'^step (\d+) \| loss (\S+) \| lrm (\S+) \| dt (\S+)s \|',log,re.M)
need([int(x[0]) for x in steps]==list(range(5)) and all(math.isfinite(float(v)) for row in steps for v in row[1:]),'five-finite')
need([json.loads(x) for x in raw['files']['train/steps.jsonl']['text'].splitlines()]==list(range(5)),'official-boundaries')
r=get('train/rehearsal.json');obs=r['observations'];fifth=obs[4]['observed_seconds']
need(r['step_boundaries']==5 and [x['step'] for x in obs]==list(range(5)),'observer-count')
need(all(math.isfinite(x['observed_seconds']) for x in obs) and all(x['observed_seconds']<y['observed_seconds'] for x,y in zip(obs,obs[1:])),'ordered-observer')
need(fifth==845.8894737660885 and fifth<=b['cold_decision']['maximum_seconds']==900 and fifth<=870,'actual-cold-ceiling')
need(r['startup_allowance_observed']==min(fifth,900) and r['charged_through_process_exit']==r['wall_seconds']-min(fifth,900)<=1800 and r['charged_through_process_exit_within_budget'] is True,'actual-charge')
need(r['fresh_empty_sdk_caches']==iso['fresh_empty_sdk_caches']==cache,'actual-fresh-cache')
term=get('terminal.json');outer=get('outer-flock-exit.json')
need(r['returncode']==term['returncode']==outer['returncode']==int(raw['files']['run-exit-code.txt']['text'])==0,'all-exits')
need(term['label']==b['label'] and term['no_eval'] and term['no_automatic_full'],'terminal-scope')
need(outer['command']==b['outer_command'],'outer-argv')
ck=raw['checkpoint_after_terminal']
need(ck['sha256']==r['checkpoint_sha256']=='1ef38b27e7923b17b3e0d37a375fcbd5c02be2e7004b1aa7b2d43be20066af07' and ck['bytes']==r['checkpoint_bytes']==453048658 and ck['unchanged_during_hash'],'checkpoint-afterhash')
owners=[json.loads(x[len('SUBMISSION_OPTIMIZER_INTEGRATION '):]) for x in log.splitlines() if x.startswith('SUBMISSION_OPTIMIZER_INTEGRATION ')]
need(len(owners)==4 and {x['rank'] for x in owners}==set(range(4)),'rank-integrations')
graph_counts={}
for rank in range(4):
    rows=list(csv.DictReader(io.StringIO(raw['files'][f'metrics-training/metric_table_graph_stats_rank{rank}.csv']['text'])))
    need(bool(rows) and any('_forward_' in x['graph_name'] for x in rows) and any('_backward_' in x['graph_name'] for x in rows),'closed-graphs-'+str(rank))
    graph_counts[str(rank)]=len(rows)
need(set(observed)==set(post['observed_pids'])=={x['pid'] for x in post['pid_rows']} and len(observed)==29,'all-observed-union')
for pid,row in observed.items():need(post['observed_starttime_records'][str(pid)]['starttime_ticks']==row['starttime_ticks'],'post-identity-'+str(pid))
need(post['all_observed_pids_absent'] and all(x['exists'] is False for x in post['pid_rows']),'all-gone')
need(not any(x['pid'] in observed or x['ppid'] in observed or x['pgid'] in observed or x['sid'] in observed for x in post['all_process_identity_rows']),'independent-descendant-census')
need(post['related_rows']==[] and post['ps_rc']==0 and post['ps_stderr']=='','post-process-clean')
need(post['neuron_device_idle'] and all(not x['neuron_processes'] for x in post['neuron_ls']) and post['neuron_ls_stderr']=='','post-device')
need(post['fuser_rc']==1 and post['fuser_stdout']==post['fuser_stderr']=='','post-fuser')
need(post['lock_device']==raw['lock_device']==66305 and post['lock_inode']==raw['lock_inode']==832589 and post['lock_competitor_rc']==raw['lock_competitor_rc']==0,'post-original-free-lock')
need(post['no_kills_or_changes'] and all(not x['exists'] for x in post['old_queue_pids']),'retired-queues')
need(datetime.fromisoformat(post['utc'])>datetime.fromisoformat(outer['ended_utc'])>datetime.fromisoformat(term['utc']),'post-order')
need(raw['processes']==[] and raw['ps_rc']==raw['neuron_ls_rc']==0,'retrieval-quiescent')
print(json.dumps({'status':'PASS_COLD_FIVE_ONLY','checks':len(checks),'raw_commit':'587430bd4e2c0ec78a6a50cf516aa37cbaff1cf3','raw_sha256':sha(rawpath.read_bytes()),'post_sha256':sha(postpath.read_bytes()),'source_sha256':b['source_sha256'],'uncapped_fifth_seconds':fifth,'wall_seconds':r['wall_seconds'],'charged_through_exit_seconds':r['charged_through_process_exit'],'losses':[float(x[1]) for x in steps],'observed_pids':len(observed),'postflight_utc':post['utc'],'graph_rows':graph_counts,'no_quality_or_steady_speed_claim':True,'launch_authorized':False},indent=2))
