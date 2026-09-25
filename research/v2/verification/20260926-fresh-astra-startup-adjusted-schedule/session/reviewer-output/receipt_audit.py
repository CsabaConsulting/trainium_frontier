"""Audit supplied JSON only. Does not execute or authenticate the absent observer."""
import collections, hashlib, json, math, pathlib, struct, traceback
P=pathlib.Path('/packet');W=pathlib.Path('/work');out=W/'receipt_audit_attempt2';out.mkdir(exist_ok=False)
checks=[];summary={};errors=[]
def chk(name,ok,**data):checks.append(dict(name=name,passed=bool(ok),**data))
def sha_obj(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def float_lr(seconds):
 p=min(seconds/1800.,1.)
 return 1. if p<.25 else 1.-.95*(p-.25)/.75
def reference_timeline(delay,steps,rank,on,startup):
 # Independent replay of source sampling points with the receipt's disclosed
 # fake time rule. No hidden observer code or original test is executed.
 t=0.
 def tick():
  nonlocal t
  t+=.001;return t
 process=tick();loop=tick();origin=loop;records=[];reset=[]
 for step in range(steps):
  start=tick();elapsed=start-origin;t0=tick()
  sched=elapsed if not on or step>=startup else max(0.,start-process-870.)
  if step==0:t+=delay # test shim states delay AFTER first optimizer update
  dt=tick()-t0
  if step+1==startup:
   now=tick();origin=min(now,process+870.);reset.append(dict(step=step,now=now,origin=origin,charged_excess=max(0.,now-process-870.)))
  elapsed_after=tick()-origin
  if rank==0 and (step<5 or step%10==0):remaining=1800.-(tick()-origin)
  records.append(dict(step=step,step_start=start,elapsed=elapsed,schedule_seconds=sched,lr_multiplier=float_lr(sched),wd=.2*(1.-min(1.,sched/1800.)),momentum=.85+.1*min(step/300.,1.),dt=dt,elapsed_after=elapsed_after))
 before_save=tick()-origin;after_save=tick()-origin
 return dict(process_origin=process,loop_origin=loop,records=records,resets=reset,before_save=before_save,after_save=after_save)
try:
 hashes=json.loads((P/'INPUT_MANIFEST.json').read_text())['source_sha256']
 for folder,startup in [('',5),('attempt2-stub2/',2)]:
  scope='final_five_boundary' if not folder else 'prior_stub_two_boundary'
  commands=json.loads((P/'raw'/folder/'tiny-commands.json').read_text());cmdmap={x['name']:x for x in commands};docs={}
  scope_summary={}
  for name in ['base','default','off','on779','on790','on1000']:
   d=json.loads((P/'raw'/folder/f'tiny-{name}.json').read_text());docs[name]=d
   role='baseline' if name=='base' else 'candidate';on=name.startswith('on');expected_steps=6 if on else 3
   chk('metadata',d['train_py_sha256']==hashes[role] and d['nproc']==4 and d['steps']==expected_steps and d['micro_batch']==[2,128] and not d['timed_out'],scope=scope,name_run=name)
   cmd=cmdmap[name];argv=cmd['argv'];trainargv=argv[argv.index('--')+1:]
   chk('command_train_argv_matches',trainargv==d['argv'],scope=scope,name_run=name,expected=trainargv,actual=d['argv'])
   chk('rank_set',sorted(r['rank'] for r in d['ranks'])==[0,1,2,3],scope=scope,name_run=name)
   run=dict(source_sha256=d['train_py_sha256'],receipt_sha256=hashlib.sha256((P/'raw'/folder/f'tiny-{name}.json').read_bytes()).hexdigest(),status_label=d['verdict'],reason_label=d['reason'],torchrun_rc=d['torchrun_rc'],observation_driver_sha256=d['observation_driver_sha256'],schedule_observer_sha256=d['schedule_observer_sha256'],support_pins=d['support_pins'],shims=d['shims'],command=cmd,appended_argv=d['appended_argv'],ranks=[])
   for r in d['ranks']:
    rank=r['rank'];c=r['cache_observations'];s=r['schedule_observations'];steps=c['steps'];final=steps[-1]['parameters_after_update']
    chk('rank_completion',r['status']=='PASS' and not r.get('error') and not r['cross_rank_mismatches'] and r['n_params']==65 and r['n_changed']==65 and len(steps)==expected_steps and len(r['losses'])==expected_steps,scope=scope,name_run=name,rank=rank)
    if startup==5:chk('actual_five_boundary_override_disclosed',s.get('startup_steps_excluded')==5,scope=scope,name_run=name,rank=rank)
    chk('first_step_delay_matches_command',s['first_step_fake_delay_s']==cmd['first_step_fake_delay_s'],scope=scope,name_run=name,rank=rank)
    timeline=reference_timeline(cmd['first_step_fake_delay_s'],expected_steps,rank,on,startup)
    missing=[];sample_results=[]
    for i,(step,obs,expect) in enumerate(zip(steps,s['optimizer_groups'],timeline['records'])):
     mb=[key for key,value in step['gradients_before_clip'].items() if value is None];ma=[key for key,value in step['gradients_after_clip'].items() if value is None];missing.append(dict(step=i,before=mb,after=ma))
     chk('gradients_and_parameters_present',not mb and not ma and len(step['gradients_before_clip'])==65 and len(step['gradients_after_clip'])==65 and len(step['parameters_after_update'])==65,scope=scope,name_run=name,rank=rank,step=i)
     gr=[]
     for j,g in enumerate(obs):
      exp_lr=g['initial_lr']*expect['lr_multiplier'];ok=math.isclose(g['lr'],exp_lr,rel_tol=0.,abs_tol=1e-12)
      if g['kind']=='muon':ok=ok and abs(g['weight_decay']-expect['wd'])<1e-12 and abs(g['momentum']-expect['momentum'])<1e-14 and g['beta2']==.9
      chk('schedule_matches_independent_sampling_reference',ok,scope=scope,name_run=name,rank=rank,step=i,group=j,expected_lr=exp_lr,expected_wd=expect['wd'] if g['kind']=='muon' else g['weight_decay'],actual=g)
      gr.append(g)
     sample_results.append(dict(expected=expect,actual=gr))
    # After update, synchronization yields replicated parameters, but optimizer states
    # are owner-sharded and losses/pre-reduce gradients are rank-specific.
    chk('replicated_final_parameter_digests',final==d['ranks'][0]['cache_observations']['steps'][-1]['parameters_after_update'],scope=scope,name_run=name,rank=rank)
    for i,step in enumerate(steps):
     chk('replicated_after_update_digests',step['parameters_after_update']==d['ranks'][0]['cache_observations']['steps'][i]['parameters_after_update'],scope=scope,name_run=name,rank=rank,step=i)
    rr=dict(rank=rank,losses=r['losses'],missing_gradients=missing,forward_shapes=[f['input_shape'] for f in c['forwards']],forward_custom_nodes=[f['custom_mlp_nodes'] for f in c['forwards']],custom_backward_calls=c['mlp_custom_backward_calls'],cache_refresh_count=len(c['cache_refreshes']),first_forward_cache_connected=all(v['connected_first_forward'] for v in c['forwards'][0]['caches'].values()),schedule_reference=timeline,schedule_expected_actual=sample_results,losses_digest=sha_obj(r['losses']),final_digests_digest=sha_obj(final),step_digests=[{k:sha_obj(v) for k,v in step.items()} for step in steps])
    if rank==0:
     cp=c['checkpoint'];rr.update(checkpoint_file_sha256=c['checkpoint_file_sha256'],checkpoint_model_state_digest=sha_obj(cp['model_state']),checkpoint_meta=cp['meta'],checkpoint_model_config=cp['model_config'],checkpoint_check=r['checkpoint_check'],model_numel=sum(math.prod(v['shape']) for v in cp['model_state'].values()))
     chk('checkpoint_time_matches_sampling_reference',abs(cp['meta']['budget_elapsed_before_save']-timeline['before_save'])<1e-10,scope=scope,name_run=name,expected=timeline['before_save'],actual=cp['meta']['budget_elapsed_before_save'])
     chk('checkpoint_model_digest_matches_update',cp['model_state']==steps[-1]['parameters_after_update'],scope=scope,name_run=name)
     chk('checkpoint_check_claim',r['checkpoint_check']['ok'] and r['checkpoint_check']['deterministic'] and r['checkpoint_check']['logits_shape']==[1,128,8192],scope=scope,name_run=name)
    run['ranks'].append(rr)
   scope_summary[name]=run
  comparisons=[]
  for ref,other in [('base','default'),('base','off'),('on779','on790')]:
   for rank in range(4):
    a=docs[ref]['ranks'][rank];b=docs[other]['ranks'][rank]
    for key in (['losses','cache_observations','final_digests'] if rank==0 else ['losses','cache_observations']):
     equal=a[key]==b[key];chk('cross_run_same_rank_exact_JSON',equal,scope=scope,pair=[ref,other],rank=rank,field=key)
     comparisons.append(dict(pair=[ref,other],rank=rank,field=key,equal=equal,digest_a=sha_obj(a[key]),digest_b=sha_obj(b[key])))
  scope_summary['comparisons']=comparisons;summary[scope]=scope_summary
 failed=json.loads((P/'raw/failed-attempt1-base.json').read_text())
 summary['supplied_failed_attempt']=dict(verdict=failed['verdict'],reason=failed['reason'],torchrun_rc=failed['torchrun_rc'],source_sha256=failed['train_py_sha256'],schedule_observer_sha256=failed['schedule_observer_sha256'],ranks=[{k:r.get(k) for k in ['rank','status','error','traceback_tail','schedule_observations','cache_observations']} for r in failed['ranks']])
 chk('supplied_failure_preserved',failed['verdict']=='FAIL' and all(r['status']=='FAIL' for r in failed['ranks']))
except Exception:
 errors.append(traceback.format_exc())
finally:
 bad=[r for r in checks if not r['passed']]
 result=dict(status='PASS' if not bad and not errors else 'REFUSE',checks=len(checks),failed_checks=len(bad),errors=errors,note='Receipt consistency audit, not hidden-harness authentication or reviewer-executed training')
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'details.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n');(out/'failures.json').write_text(json.dumps(bad,indent=2)+'\n');print(json.dumps(result,indent=2))
 if bad:print(json.dumps(bad[:3],indent=2))
 if errors:print('\n'.join(errors))
 raise SystemExit(0 if result['status']=='PASS' else 1)
