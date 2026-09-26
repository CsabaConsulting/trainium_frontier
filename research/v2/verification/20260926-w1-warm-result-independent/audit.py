"""Independent, local Git-archive-only W1 authentication. Never launches work."""
import base64, copy, csv, datetime, hashlib, io, json, math, re, shlex, stat, subprocess, sys

WPATH = 'research/v2/operator/20260926-trn22-stablepath-warm/'
SPATH = 'research/v2/operator/20260926-trn22-stable-source-pair/completed/'
PACKET = 'research/v2/proposals/20260926-stablepath-warm-observation-trn22/'
WSHA = 'e50418bffad38eaabf4d2b38be6c7d2e4d511bbe82b5591a5feec77d02436b5e'
SSHA = '6424cfcfbb311a49a6e39080a88afa6d05b027ce951a09a506b2547e49742b0c'
SOURCE = '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c'
checks = []
def check(ok, name):
    if not ok: raise ValueError(name)
    checks.append(name)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def git(commit, path): return subprocess.check_output(['git','show',commit+':'+path])
def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('DUPLICATE_JSON_KEY:'+key)
        result[key] = value
    return result
def parse(raw):
    return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('NONFINITE_JSON:'+x)))
def bundle(raw, pin):
    check(sha(raw)==pin, 'bundle SHA '+pin)
    obj=parse(raw); files={}
    for name,item in obj['files'].items():
        check(('text' in item)!=('base64' in item), 'exclusive encoding '+name)
        content=item['text'].encode() if 'text' in item else base64.b64decode(item['base64'],validate=True)
        check(len(content)==item['bytes'] and sha(content)==item['sha256'], 'inner bytes/hash '+name)
        files[name]=content
    check(len(files)==obj['file_count'] and sum(map(len,files.values()))==obj['raw_bytes'],'bundle counts')
    return obj,files
def identities(a,b):
    check(a==b, 'all exact integer-nanosecond identities unchanged')
    for key,item in a.items():
        check(type(item['st_mtime_ns']) is int and type(item['st_ctime_ns']) is int, 'integer ns '+key)
        check(not stat.S_ISLNK(item['st_mode']), 'not symlink '+key)
        check(stat.S_ISDIR(item['st_mode']) if key=='stable_directory' else stat.S_ISREG(item['st_mode']), 'file type '+key)
    for key in ['executed_train','expected_imported_prepare','package_train','launch','provenance','stable_directory']:
        check(a[key]['st_mode'] & 0o222 == 0, 'read-only mode '+key)
def timing(h):
    check(h['returncode']==0 and h['mode']=='train' and h['step_boundaries']==5, 'five-step successful harness')
    obs=h['observations'];check([x['step'] for x in obs]==list(range(5)), 'five ordered boundaries')
    values=[x['observed_seconds'] for x in obs]
    check(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in values), 'finite boundaries')
    check(all(x<y for x,y in zip(values,values[1:])), 'increasing boundaries')
    fifth=values[-1];allowance=min(fifth,900)
    check(h['startup_allowance_observed']==allowance,'official capped allowance')
    wall=h['wall_seconds'];charge=h['charged_through_process_exit']
    check(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in [wall,charge]),'finite wall/charge')
    check(math.isclose(wall-allowance,charge,rel_tol=0,abs_tol=1e-9),'through-exit accounting formula')
    check(charge<=1800 and h['charged_through_process_exit_within_budget'] is True,'official charged budget')
    check(h['checkpoint_bytes']==453048594 and bool(re.fullmatch('[0-9a-f]{64}',h['checkpoint_sha256'])),'retained checkpoint receipt')
    return fifth
def disposition(fifth,target=120):
    check(type(fifth) in (int,float) and math.isfinite(fifth) and fifth>=0,'valid target input')
    return 'WARM_TARGET_MET' if fifth<=target else 'WARM_TARGET_MISSED'
def losses(raw):
    rows=[dict(step=int(m[0]),loss=float(m[1]),lrm=float(m[2]),step_seconds=float(m[3])) for m in re.findall(r'^step (\d+) \| loss ([0-9.]+) \| lrm ([0-9.]+) \| dt ([0-9.]+)s',raw.decode(),re.M)]
    check([r['step'] for r in rows]==list(range(5)), 'exactly five printed losses')
    check(all(math.isfinite(r[k]) for r in rows for k in ['loss','lrm','step_seconds']),'finite logged losses')
    return rows
def postflight(post,live):
    observed={int(row.split()[0]) for row in live['job_lines']}
    check(len(observed)==10 and observed==set(post['observed_pids']), 'all ten live PIDs explicitly tracked')
    check({r['pid'] for r in post['pid_rows']}==observed and all(r['exists'] is False for r in post['pid_rows']), 'ten PID absences')
    rows=post['all_process_identity_rows']
    check(len(rows)==257 and len({r['pid'] for r in rows})==257,'257 unfiltered unique process identities')
    check(not any(r[k] in observed for r in rows for k in ['pid','ppid','pgid','sid']),'no observed PID/PPID/PGID/SID descendants')
    check(not any(re.search(r'neuronx?-cc|neuron-cc|neuronx-compile',r['comm']) for r in rows),'no compiler-named comm in census')
    check(post['related_rows']==[] and post['all_observed_pids_absent'] is True,'postflight related/absence agreement')
    check(post['ps_rc']==0 and not post['ps_stderr'] and not post['neuron_ls_stderr'],'postflight successful tools')
    dev=post['neuron_ls'];check(len(dev)==1 and dev[0]['neuron_device']==0 and dev[0]['nc_count']==4 and dev[0]['logical_neuroncore_config']==2 and not dev[0]['neuron_processes'] and post['neuron_device_idle'] is True,'explicit full-chip idle')
    check(post['lock_inode']==832589 and post['lock_device']==66305 and post['lock_competitor_rc']==0 and post['no_kills_or_changes'] is True,'original lock free without changes')
    return sorted(observed)
def refused(fn,name):
    try: fn()
    except (ValueError,KeyError,TypeError): checks.append('negative refused: '+name)
    else: raise ValueError('NEGATIVE_ACCEPTED:'+name)
def audit():
    wb,w=bundle(git('81ca7dbe',WPATH+'completed/results-bundle.json'),WSHA)
    sb,s=bundle(git('190ccf4a',SPATH+'results-bundle.json'),SSHA)
    live_raw=git('81ca7dbe',WPATH+'launch/first-native-evidence.json')
    live,lf=bundle(live_raw,'ad7e7fe14afd4243df419336288434f9e88c3bbdbd5c11b7c59ccb71b3d9d54f')
    post_raw=git('81ca7dbe',WPATH+'completed/explicit-process-postflight.json')
    check(sha(post_raw)=='c0c5c04cdc41ff57a6a17710079891a3de3ed07b71fdd73e51f07cf4413ca715','postflight exact SHA')
    post=parse(post_raw)
    record=lambda name: parse(w[name]);prior=lambda name:parse(s[name])
    binding=record('WARM_BINDING.disabled.json');proposal=record('EXPERIMENT_PROPOSAL.json');manifest=record('MANIFEST.sha256.json')
    for name in ['WARM_BINDING.disabled.json','EXPERIMENT_PROPOSAL.json','MANIFEST.sha256.json','S1-pre-source-identity.json']:
        check(w[name]==git('52c31b73',PACKET+name),'frozen packet equality '+name)
    first_proposal=parse(git('ae77965c',PACKET+'EXPERIMENT_PROPOSAL.json'))
    check({k for k in proposal if proposal[k]!=first_proposal.get(k)}=={'prepared_utc'},'proposal-first sole timestamp correction; scientific target unchanged')
    for name,pin in manifest['files'].items():check(sha(git('52c31b73',PACKET+name))==pin,'all packet manifest bytes '+name)
    staged=record('preflight.json')['staged_hashes']
    for name,pin in staged.items():
        if name in manifest['files']:check(pin==manifest['files'][name],'actual staged manifest pin '+name)
    for name,pin in [('w1/package/train.py',SOURCE),('idle_guard.py',binding['idle_guard_sha256']),('dispatch_driver.py',binding['dispatch_driver_sha256'])]:check(staged[name]==pin,'actual staged external pin '+name)
    check(staged['MANIFEST.sha256.json']==sha(w['MANIFEST.sha256.json']),'actual staged manifest itself')
    check(sha(git(binding['source_commit'],binding['source_git_path']))==SOURCE==binding['source_sha256'],'exact source Git blob')
    check(sha(git('52c31b73','prepare.py'))==binding['prepare_sha256'],'protected prepare blob')
    commit,path=binding['harness_git_object'].split(':',1)
    check(sha(git(commit,path))==binding['harness_sha256'],'harness exact Git blob')
    pre=record('W1-pre-source-identity.json');identities(pre,record('W1-post-source-identity.json'))
    check(w['S1-pre-source-identity.json']==s['S1-pre-source-identity.json'],'actual original S1 identity anchor')
    old=prior('S1-pre-source-identity.json')
    for key in ['executed_train','expected_imported_prepare','harness','protected_prepare','stable_directory']:
        check(pre[key]==old[key]==prior('S1-post-source-identity.json')[key],'S1/W1 lifetime identity '+key)
    for key,pin in [('executed_train',SOURCE),('package_train',SOURCE),('expected_imported_prepare',binding['prepare_sha256']),('protected_prepare',binding['prepare_sha256']),('harness',binding['harness_sha256'])]:check(pre[key]['sha256']==pin,'identity content pin '+key)
    for key,name in [('launch','w1/package/launch-command.txt'),('provenance','w1/package/provenance.json')]:check(pre[key]['sha256']==manifest['files'][name],'package content pin '+key)
    launch=record('W1-launch.json');sl=prior('S1-launch.json');phase=binding['phases'][0]
    check(len(binding['phases'])==1 and phase['name']=='W1','one new W1 phase')
    for key in ['training_command','command','environment']:check(launch[key]==phase[key],'bound launch '+key)
    check(launch['training_command']==sl['training_command']==proposal['candidate']['training_command'],'S1/W1 identical full science argv')
    check(shlex.split(git('52c31b73',PACKET+'w1/package/launch-command.txt').decode())==launch['training_command'],'actual launch file argv')
    check(launch['training_command'][3]==pre['executed_train']['path'] and launch['expected_prepare_path']==pre['expected_imported_prepare']['path'],'actual argv and adjacent expected import path')
    check({k for k in launch['environment'] if launch['environment'][k]!=sl['environment'].get(k)}=={'TRN_EXP_LABEL','TORCH_NEURONX_METRICS_DIR'},'only recorded environment nuisance differences')
    caches=binding['cache_environment'];check(len(caches)==6 and len(set(caches.values()))==6,'six distinct cache paths')
    for key,path in caches.items():check(launch['environment'][key]==sl['environment'][key]==path,'retained cache path '+key)
    check(launch['environment']['NEURON_COMPILE_CACHE_URL']==caches['NEURON_COMPILE_CACHE_DIR'],'compile URL equals directory')
    check('--fresh-caches' not in launch['command'] and '--fresh-caches' in sl['command'],'warm versus cold harness mode explicit')
    wh=record('w1/train/rehearsal.json');sh=prior('s1/train/rehearsal.json');wf=timing(wh);sf=timing(sh)
    check(sh['fresh_empty_sdk_caches']==caches and wh['fresh_empty_sdk_caches']=={},'cold receipt and no new W1 fresh-cache claim')
    obs=record('W1-mechanical-observation.json');check(obs['checkpoint']['sha256']==wh['checkpoint_sha256'] and obs['checkpoint']['st_size']==wh['checkpoint_bytes'],'checkpoint independent receipt agreement')
    check(stat.S_ISREG(obs['checkpoint']['st_mode']) and obs['checkpoint']['path']==phase['sandbox']+'/out/final.pt','checkpoint file type/path')
    wr=losses(w['w1/training.log']);sr=losses(s['s1/training.log']);check(wr==obs['first_five_logged'],'own log parser matches five recorded rows')
    check('over_budget:       False' in w['w1/training.log'].decode(),'source budget False recorded')
    target=binding['useful_warm_fifth_target_seconds'];check(target==120 and obs['useful_warm_target_seconds']==120,'original exact 120-second target')
    verdict=disposition(wf,target);check(verdict=='WARM_TARGET_MISSED'==obs['target_disposition']==record('terminal.json')['target_disposition'],'unrounded target miss agreement')
    check(record('terminal.json')['fifth_boundary_seconds']==wf and obs['fifth_boundary_seconds']==wf,'fifth receipt agreement')
    check(record('W1-exit.json')['returncode']==0 and record('outer-flock-exit.json')['returncode']==0,'child and outer exit zero')
    for name in ['under-lock-idle.json','W1-pre-under-lock-idle.json','W1-post-under-lock-idle.json']:
        idle=record(name);check(idle['status']=='PASS' and idle['lock']['verified_existing_exclusive_lock'] is True and idle['lock']['inode']==832589 and idle['lock']['device']==66305,'authenticated lifetime lock '+name)
        check(idle['ps']['returncode']==0 and idle['ps']['busy']==[] and idle['neuron_ls']['returncode']==0 and idle['neuron_ls']['process_pids']==[],'ps and neuron-ls idle '+name)
    check(record('W1-resources.json')['free_disk_bytes']>=binding['minimum_free_disk_gib']*1024**3,'disk floor')
    check(sorted(x['environment']['RANK'] for x in live['workers'])==['0','1','2','3'],'all four live ranks')
    for worker in live['workers']:
        check(worker['cwd']==phase['sandbox'] and shlex.split(worker['argv'])[2:]==launch['training_command'][3:],'each live worker argv/cwd')
        check(worker['environment']['WORLD_SIZE']=='4' and worker['environment']['NEURON_LOGICAL_NC_CONFIG']=='2','worker world/LNC')
        for key,path in caches.items():check(worker['environment'][key]==path,'worker cache path '+key)
    pids=postflight(post,live)
    times=[launch['utc'],record('W1-exit.json')['utc'],record('terminal.json')['utc'],post['utc']]
    check(all(datetime.datetime.fromisoformat(x)<datetime.datetime.fromisoformat(y) for x,y in zip(times,times[1:])),'launch/exit/terminal/explicit postflight order')
    check(prior('terminal.json')['status']=='PAIR_REFUSED_OR_INCOMPLETE' and prior('terminal.json')['error']=='MATCHED_PRINCIPAL_COMPILER_BYTES_MISSING' and not any(x.startswith('S2-') or x.startswith('s2/') for x in s),'original failure and no S2 preserved')
    graph_rows=[]
    for rank in range(4):
        rows=list(csv.DictReader(io.StringIO(w[f'w1/metrics-training/metric_table_graph_stats_rank{rank}.csv'].decode())))
        oldrows=list(csv.DictReader(io.StringIO(s[f's1/compiler-evidence/metric_table_graph_stats_rank{rank}.csv'].decode())))
        for name,nodes in [('model_default_0_forward_1',666),('model_default_0_backward_3',1169)]:
            new=[x for x in rows if x['graph_name']==name];oldr=[x for x in oldrows if x['graph_name']==name]
            check(len(new)==len(oldr)==1 and new[0]['graph_id']==oldr[0]['graph_id'] and int(new[0]['graph_node_count'])==nodes,'principal graph name/key/node identity rank '+str(rank)+' '+name)
            graph_rows.append(dict(rank=rank,graph_name=name,graph_id=new[0]['graph_id'],nodes=nodes,recorded_compile_us=float(new[0]['torch_neuronx_compile_us'])))
    evidence=record('W1-compiler-evidence.json')
    check(evidence['capture_status']=='PRINCIPAL_BYTES_MISSING' and evidence['authenticated_raw_inputs']=={} and evidence['exact_input_bytes_complete'] is False,'missing principal inputs explicit')
    check(record('terminal.json')['actual_import_origin_observed'] is False and evidence['zero_compile_us_is_hit_proof'] is False,'unobserved import and no zero-counter proof')
    bad=copy.deepcopy(pre);bad['executed_train']['st_mtime_ns']+=1;refused(lambda:identities(pre,bad),'one-nanosecond mutation')
    bad=copy.deepcopy(wh);bad['charged_through_process_exit']=1801;refused(lambda:timing(bad),'overbudget contradicts flag')
    bad=copy.deepcopy(wh);bad['observations'][-1]['observed_seconds']=float('nan');refused(lambda:timing(bad),'nonfinite boundary')
    bad=copy.deepcopy(post);bad['all_process_identity_rows'][0]['ppid']=pids[-1];refused(lambda:postflight(bad,live),'live descendant despite claimed idle')
    refused(lambda:bundle(git('81ca7dbe',WPATH+'completed/results-bundle.json'), '0'*64),'bundle pin mutation')
    bad=parse(git('81ca7dbe',WPATH+'completed/results-bundle.json'));bad['files']['terminal.json']['text']+=' ';raw=json.dumps(bad).encode();refused(lambda:bundle(raw,sha(raw)),'inner artifact mismatch')
    check(disposition(120)=='WARM_TARGET_MET' and disposition(120.000000001)=='WARM_TARGET_MISSED','strict boundary without rounding')
    refused(lambda:disposition(float('inf')),'infinite timing')
    return dict(status='AUTHENTICATED_DIAGNOSTIC_ONLY',target_disposition=verdict,target_seconds=target,target_miss_seconds=wf-target,w1_fifth_seconds=wf,s1_fifth_seconds=sf,observed_reduction_seconds=sf-wf,observed_reduction_fraction=(sf-wf)/sf,s1_over_w1_ratio=sf/wf,source_sha256=SOURCE,bundle_sha256=WSHA,prior_bundle_sha256=SSHA,authenticated_bundle_file_counts={'W1':len(w),'S1':len(s),'live_W1':len(lf)},artifact_sha256={k:sha(v) for k,v in w.items()},identity_records=pre,science_argv=launch['training_command'],retained_cache_environment=caches,w1_harness=wh,s1_harness=sh,w1_printed_rows=wr,s1_printed_rows=sr,printed_losses_match_only_first_two=True,checkpoint_byte_identity=False,arithmetic_equivalence='NOT_ESTABLISHED: observed printed LR/loss differences and different checkpoint hashes',principal_graph_rows=graph_rows,principal_input_bytes='MISSING',actual_import_origin='UNOBSERVED',cache_hit='UNPROVEN',native_neff_reuse='UNPROVEN',path_only_causality='UNPROVEN',postflight=dict(utc=post['utc'],tracked_pids=pids,all_absent=True,unfiltered_process_rows=257,related_identity_rows=0,device_idle=True,lock_inode=832589,lock_device=66305,lock_free=True,unobserved_between_snapshot_compiler_activity='UNKNOWN'),original_pair='REFUSED; S2 never launched',quality_verdict='NOT_APPLICABLE',launch_or_adoption_authority=False,checks=len(checks),negative_tests=8,optimized=not __debug__)
if __name__=='__main__':
    try: result=audit()
    except Exception as exc:
        print(json.dumps(dict(status='REFUSED_OR_INCOMPLETE',error=type(exc).__name__+': '+str(exc),checks_completed=len(checks)),allow_nan=False));raise SystemExit(2)
    print(json.dumps(result,indent=2,allow_nan=False))
