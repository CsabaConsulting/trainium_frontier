"""Archive-only independent result authentication, using stdlib bytes not tensors."""
import base64, copy, csv, datetime, hashlib, io, json, math, random, re, struct, subprocess
RAW='015bacec8fdbe47f1b63f01b160afdc264d780f9'
ROOT='research/v2/operator/20260926-trn22-k2-fp32-first/'
PACKET='research/v2/proposals/20260926-k2-fp32-first-discriminator/'
BUNDLE_SHA='db4fd0b148fd29930bc4925ba898a8f80108d1bd84c6ad3efc33e8694e15e2c6'
ROUTES=['direct_consumer','eager_custom','compiled_custom','compiled_ordinary_embedding']
checks=[]
def ok(value,name):
    if not value:raise ValueError(name)
    checks.append(name)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def git(c,p):return subprocess.check_output(['git','show',c+':'+p])
def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('DUPLICATE_JSON_KEY:'+k)
        out[k]=v
    return out
def parse(raw):return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON:'+x)))
def archive(raw,pin):
    ok(sha(raw)==pin,'archive outer SHA')
    b=parse(raw);files={}
    for n,v in b['files'].items():
        ok(('text' in v)!=('base64' in v),'exclusive encoding '+n)
        r=v['text'].encode() if 'text' in v else base64.b64decode(v['base64'],validate=True)
        ok(sha(r)==v['sha256'] and len(r)==v['bytes'],'artifact hash and bytes '+n);files[n]=r
    ok(sum(map(len,files.values()))==b['raw_bytes'],'decoded byte count')
    return b,files
def rec(values,shape,dtype,fmt):
    raw=struct.pack('<'+fmt*len(values),*values)
    return dict(shape=shape,dtype=dtype,finite=all(math.isfinite(x) for x in values),nonzero=sum(x!=0 for x in values),sha256=sha(raw))
def oracle():
    # Independently specified unique IDs: one occurrence per row. Each gradient
    # is exactly a dyadic multiple of1/32, so initial+gradient is exact binary32.
    rng=random.Random(42)
    values=[rng.randrange(-256,257)/32 for _ in range(256*128)]
    gradient=rec(values,[256,128],'torch.float32','f')
    zero=rec([0.0]*(256*128),[256,128],'torch.float32','f')
    index=rec(list(range(256)),[256],'torch.int32','i')
    start=rec(list(range(256)),[256,1],'torch.int32','i')
    return dict(gradient=gradient,zero=zero,upstream=gradient,ids=index,counts=rec([1]*256,[256,1],'torch.int32','i'),starts=start,positions=start,maximums=rec([1,1],[2,1],'torch.int32','i'),max_abs_vs_zero=max(map(abs,values)))
def compare_record(got,actual,expected,error,exact):
    ok(set(got)=={'exact','actual','expected','max_abs'},'complete finite comparison schema')
    ok(got['exact'] is exact and got['actual']==actual and got['expected']==expected and type(got['max_abs']) in (int,float) and math.isfinite(got['max_abs']) and got['max_abs']==error,'independent exact-byte comparison and error')
def rank_result(r,rank,b,o):
    ok(r['rank']==rank and r['world_size']==4 and r['shape']==dict(V=256,T=256,C=128) and r['seed']==42,'exact rank/world/shape/seed')
    ok(r['label']==b['label'] and r['argv']==b['checker_args'],'bound rank label/argv')
    for key,name in [('checker_sha256','check_native.py'),('kernel_sha256','consumer_nki.py'),('reference_sha256','reference.py')]:ok(r[key]==b['support_sha256'][name],'rank source pin '+key)
    ok(r['torch']=='2.11.0+cpu' and r['nki_version']=='0.5.0+28631259367.ga768afa6','recorded runtime versions')
    ok(r['route_order']==ROUTES and r['first_custom_compiled_fixture']=='unique/fp32' and r['timings']==[],'first custom compiled fixture and no timing claim')
    ok(r['tolerance']==dict(atol=0,rtol=0,criterion='identical bytes'),'unchanged strict tolerance')
    ok(r['original_twenty_case_gate']=='FAILED_EXACTNESS_PARKED_UNCHANGED','original failure retained')
    for key in ['native_qualification','source_adoption','accepted_speedup']:ok(r[key] is False,'no qualification '+key)
    ok(r['no_training'] is True and r['no_eval'] is True and r['whole_step_speedup']=='UNMEASURED' and r['public_20m'] is None,'no training/quality/speed')
    case=r['case'];ok(case['case']=='unique' and case['dtype']=='torch.float32' and case['metadata_group_bounds']==[1,1],'unique FP32 exact group bounds')
    compare_record(case['dense_cpu_oracle'],o['gradient'],o['gradient'],0,True)
    ok([x['name'] for x in case['routes']]==ROUTES,'all four routes in required order')
    weight=case['routes'][0]['inputs_before']['weight']
    ok(weight['shape']==[256,128] and weight['dtype']=='torch.float32' and weight['finite'] is True and weight['nonzero']==32768 and weight['sha256']=='2d20b77660fde17ac30bb287198be94fc36668f27ef670bdfe44234ea211f4cb','recorded common seed weight bytes')
    for route in case['routes']:
        name=route['name'];bad=name=='compiled_custom'
        ok(route['status']==('MISMATCH' if bad else 'EXACT') and 'error' not in route,'observed route outcome '+name)
        expected_inputs={k:o[k] for k in ['upstream','ids','counts','starts','positions','maximums']};expected_inputs['weight']=weight
        ok(route['inputs_before']==expected_inputs==route['inputs_after'] and route['inputs_unchanged'] is True,'all seven pre/post input records '+name)
        wanted={'gradient'} if name=='direct_consumer' else {'gradient','output'}
        ok(set(route['checks'])==wanted,'all expected output comparisons '+name)
        compare_record(route['checks']['gradient'],o['zero'] if bad else o['gradient'],o['gradient'],o['max_abs_vs_zero'] if bad else 0,not bad)
        if name!='direct_consumer':compare_record(route['checks']['output'],weight,weight,0,True)
    ok(r['status']=='FP32_FIRST_DIAGNOSTIC_MISMATCH_OR_ERROR' and r['returncode']==1,'rank mismatch is not PASS')
    return dict(rank=rank,route_statuses={x['name']:x['status'] for x in case['routes']},compiled_gradient_sha256=o['zero']['sha256'],receipt_returncode=1)
def postflight(post,live):
    pids={int(x.split()[0]) for x in live['job_lines']}
    ok(len(pids)==9 and pids==set(post['observed_pids']),'nine actually observed PIDs')
    timeout=[x for x in live['job_lines'] if re.search(r'\btimeout\s+/usr/bin/timeout',x)]
    ok(len(timeout)==1 and int(timeout[0].split()[0])==3217195,'actual observed timeout PID3217195')
    ok('Timeout and short-lived compiler PIDs were not observed' in post['scope'],'stale timeout-unobserved prose identified, not adopted')
    ok({x['pid'] for x in post['pid_rows']}==pids and all(x['exists'] is False for x in post['pid_rows']),'all nine explicit absences')
    rows=post['all_process_identity_rows'];ok(len(rows)==256 and len({x['pid'] for x in rows})==256,'256 unfiltered identity rows')
    ok(not any(r[k] in pids for r in rows for k in ['pid','ppid','pgid','sid']),'no surviving observed related process')
    ok(not any(re.search('neuronx?-cc|neuron-cc|neuronx-compile',r['comm']) for r in rows),'no compiler-named census process')
    ok(post['all_observed_pids_absent'] is True and post['related_rows']==[] and post['ps_rc']==0 and post['ps_stderr']=='','explicit ps agreement')
    d=post['neuron_ls'];ok(len(d)==1 and d[0]['neuron_device']==0 and d[0]['nc_count']==4 and d[0]['logical_neuroncore_config']==2 and not d[0]['neuron_processes'] and post['neuron_device_idle'] is True and not post['neuron_ls_stderr'],'device fullchip idle')
    ok(post['lock_identity']==dict(device=66305,inode=832589) and post['lock_competitor_rc']==0 and post['no_kills_or_changes'] is True,'original lock free/no changes')
    return sorted(pids)
def refused(call,name):
    try:call()
    except (ValueError,KeyError,TypeError):checks.append('negative refused '+name)
    else:raise ValueError('NEGATIVE_ACCEPTED:'+name)
def main():
    raw=git(RAW,ROOT+'completed/results-bundle.json');bundle,f=archive(raw,BUNDLE_SHA)
    ok(len(f)==31 and bundle['raw_bytes']==213475,'all31 final artifacts')
    record=lambda name:parse(f[name])
    b=record('NATIVE_BINDING.disabled.json');manifest=record('MANIFEST.sha256.json')
    for name in ['NATIVE_BINDING.disabled.json','EXPERIMENT_PROPOSAL.json','MANIFEST.sha256.json']:ok(f[name]==git('fa1a3652',PACKET+name),'frozen packet bytes '+name)
    launch=record('diagnostic-launch.json');pre=record('preflight.json')
    ok(launch['command']==b['command'] and launch['environment']==b['environment'] and launch['support_sha256']==b['support_sha256'],'exact launch/support/env')
    for name in manifest['runtime_files']:
        pin=manifest['files'][name];ok(sha(git('fa1a3652',PACKET+name))==pin==pre['staged_hashes'][name],'actual staged/runtime source '+name)
    ok(sha(git(b['diagnostic_checker_commit'],b['diagnostic_checker_path']))==b['support_sha256']['check_native.py'],'original diagnostic checker blob')
    for name in ['consumer_nki.py','reference.py']:ok(sha(git(b['source_module_commit'],b['source_module_path']+'/'+name))==b['support_sha256'][name],'unchanged original module '+name)
    ok(b['command'][:5]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','600s'] and b['timeout_seconds']==600 and b['term_grace_seconds']==60,'exact watchdog600+60')
    child=record('child-exit.json');terminal=record('terminal.json');outer=record('outer-flock-exit.json')
    ok(child['returncode']==terminal['returncode']==outer['returncode']==1 and terminal['status']=='DIAGNOSTIC_CHILD_FAILED' and terminal['complete'] is False,'actual child/terminal/outer failed consistently')
    ok(child['timeout_wrapped_child_wall_seconds']==28.54430987604428 and child['timeout_wrapped_child_wall_seconds']<600 and child['official_training_budget']=='NOT_A_QUALITY_RUN','bounded failure not watchdog timeout or quality run')
    ok(terminal['no_automatic_next_job'] is True and terminal['no_eval'] is True,'no automatic next or eval')
    idle=record('under-lock-idle.json')
    ok(idle['status']=='PASS' and idle['lock']['verified_existing_exclusive_lock'] is True and idle['lock']['inode']==832589 and idle['lock']['device']==66305,'original inherited lifetime lock')
    ok(idle['ps']['returncode']==idle['neuron_ls']['returncode']==0 and idle['ps']['busy']==[] and idle['neuron_ls']['process_pids']==[],'fresh in-lock ps/neuron idle')
    ok(pre['free_bytes']>=10.5*1024**3 and pre['lock_inode']==832589 and pre['lock_device']==66305,'disk/original identity preflight')
    caches=record('fresh-caches.json');ok(set(caches['directories'])==set(b['cache_keys']) and len(caches['directories'])==6,'six fresh caches')
    for k,v in caches['directories'].items():ok(v['initially_empty'] is True and v['path']==b['environment'][k] and v['device']==66305,'initially empty bound cache '+k)
    ok(caches['compile_cache_url']==b['environment']['NEURON_COMPILE_CACHE_URL']==b['environment']['NEURON_COMPILE_CACHE_DIR'],'cache URL=DIR')
    o=oracle();ok(o['gradient']['sha256']=='dee95c1da97debf6a901e00260bd60a74e7be1992eca64b832c4f6633cccc671' and o['gradient']['nonzero']==32709 and o['zero']['sha256']=='fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471' and o['max_abs_vs_zero']==8.0,'own independent scalar gradient/zero reconstruction')
    ranks=[];exits={}
    log=f['diagnostic.log'].decode()
    for rank in range(4):
        r=record(f'receipts/rank{rank}.json');ranks.append(rank_result(r,rank,b,o))
        env=record(f'rank-environment-r{rank}.json');exit=record(f'rank-wrapper-exit-r{rank}.json')
        ok(env['rank']==env['local_rank']==rank and env['world_size']==4 and env['pid']==3217200+rank and env['environment']==b['environment'] and env['checker_argv']==b['checker_args'],'allrank preimport env and argv')
        ok(exit['rank']==rank and exit['returncode']==1 and exit['error'] is None,'completed wrapper rc1 receipt')
        ok(datetime.datetime.fromisoformat(exit['utc'])<datetime.datetime.fromisoformat(child['utc']),'rank receipt before child exit')
        process_exit=-15 if rank in (1,2) else 1
        match=re.search(r'rank\s+: '+str(rank)+r' \(local_rank: '+str(rank)+r'\)\s+exitcode\s+: (-?\d+) \(pid: (\d+)\)',log)
        ok(match is not None and int(match[1])==process_exit and int(match[2])==env['pid'],'actual torchrun process exit '+str(rank))
        if rank in (1,2):ok('Sending process '+str(env['pid'])+' closing signal SIGTERM' in log,'torchrun SIGTERM after recorded return '+str(rank))
        exits[str(rank)]=dict(wrapper_receipt_rc=1,torchrun_process_exit=process_exit,pid=env['pid'])
        rows=list(csv.DictReader(io.StringIO(f[f'metrics/metric_table_graph_stats_rank{rank}.csv'].decode())))
        ok(bool(rows) and all('graph_id' in row for row in rows),'closed graph table retained rank '+str(rank))
    live,livefiles=archive(git(RAW,ROOT+'launch/first-native-evidence.json'),'69c09d36e50ccc9832707aae144c9956b660fa37be4e7945e4c120350ba8c096')
    postraw=git(RAW,ROOT+'completed/explicit-postflight.json');ok(sha(postraw)=='68731537bf363cd6a4639a8270253fa437a8dd7de5092f4be7cbf3ed3fe69049','postflight pinned')
    post=parse(postraw);pids=postflight(post,live)
    times=[caches['utc'],idle['started_utc'],idle['ended_utc'],launch['utc'],child['utc'],terminal['utc'],outer['ended_utc'],post['utc']]
    ok(all(datetime.datetime.fromisoformat(a)<datetime.datetime.fromisoformat(z) for a,z in zip(times,times[1:])),'cache/guard/launch/failure/postflight chronology')
    bad=copy.deepcopy(record('receipts/rank0.json'));bad['case']['routes'][2]['checks']['gradient']['exact']=True;refused(lambda:rank_result(bad,0,b,o),'zero gradient falsely exact')
    bad=copy.deepcopy(record('receipts/rank0.json'));bad['case']['routes'][0]['inputs_after']['upstream']['sha256']='0'*64;refused(lambda:rank_result(bad,0,b,o),'changed input despite true boolean')
    bad=copy.deepcopy(record('receipts/rank0.json'));bad['case']['routes'].pop();refused(lambda:rank_result(bad,0,b,o),'missing fourth route')
    bad=copy.deepcopy(record('receipts/rank0.json'));bad['case']['routes'][2]['checks']['gradient']['max_abs']=float('nan');refused(lambda:rank_result(bad,0,b,o),'nonfinite metric')
    bad=copy.deepcopy(post);bad['all_process_identity_rows'][0]['pgid']=pids[0];refused(lambda:postflight(bad,live),'surviving related process')
    refused(lambda:archive(raw,'0'*64),'outer hash mismatch')
    bad=parse(raw);bad['files']['terminal.json']['bytes']+=1;encoded=json.dumps(bad).encode();refused(lambda:archive(encoded,sha(encoded)),'inner bytes mismatch')
    return dict(status='AUTHENTICATED_FAILED_DIAGNOSTIC',native_qualification=False,quality_or_speed_claim=False,source_adoption=False,original_twenty_case_gate='FAILED_EXACTNESS_PARKED_UNCHANGED',raw_commit=RAW,bundle_sha256=BUNDLE_SHA,artifact_count=len(f),decoded_bytes=bundle['raw_bytes'],artifact_hashes={k:sha(v) for k,v in f.items()},source_pins=b['support_sha256'],independent_oracle=o,ranks=ranks,actual_process_exits=exits,child_wall_seconds=child['timeout_wrapped_child_wall_seconds'],terminal=terminal,postflight=dict(utc=post['utc'],observed_pids=pids,unfiltered_rows=256,all_absent=True,lock=dict(device=66305,inode=832589),device_idle=True,stale_timeout_prose_corrected_by_live_pid3217195=True),scope='Host pre/post recorded hashes; no compiled VJP internal values or precise compiler cause proved.',checks=len(checks),negative_controls=7,launch_or_retry_authority=False)
if __name__=='__main__':
    try:result=main()
    except Exception as e:
        print(json.dumps(dict(status='REFUSED_INCOMPLETE_AUDIT',error=type(e).__name__+': '+str(e),checks_completed=len(checks)),allow_nan=False));raise SystemExit(2)
    print(json.dumps(result,indent=2,allow_nan=False))
