import datetime, hashlib, json, pathlib
W=pathlib.Path('/work');P=pathlib.Path('/packet')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((P/'INPUT_MANIFEST.json').read_text())
num=json.loads((W/'attempt2/summary.json').read_text());audit=json.loads((W/'receipt_audit_attempt2/summary.json').read_text());src=json.loads((W/'source_audit.json').read_text())
assert num['status']==audit['status']==src['status']=='PASS'
assert num['failures']==audit['failed_checks']==0 and not audit['errors']
for ent in manifest['included']:
 assert sha(P/ent['path'])==sha(W/'packet_copy'/ent['path'])==ent['sha256']
for role in ['baseline','candidate']:assert sha(W/f'{role}_train.py')==manifest['source_sha256'][role]
p=W/'REPORT.md';p.write_text(p.read_text().replace('candidate_train.py:4082','candidate_train.py:4083'))
rows=[json.loads(s) for s in (W/'attempt2/results.jsonl').read_text().splitlines()]
mem=next(r for r in rows if r['name']=='tensor_memory_bound')
assert all(r['passed'] for r in rows)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
evidence_paths=['REPORT.md','COMMANDS.md','input_integrity.json','source_audit.json','reference_test.py','reference_test_v1.py','attempt1/summary.json','attempt1/results.jsonl','attempt1.stderr','attempt2/summary.json','attempt2/results.jsonl','attempt2/tensor_inputs.json','attempt2/tensor_results.json','attempt2/failures.json','attempt2/extractions.json','attempt2.stderr','receipt_audit.py','receipt_audit_v1.py','receipt_audit/summary.json','receipt_audit_attempt2/summary.json','receipt_audit_attempt2/checks.json','receipt_audit_attempt2/details.json','diagnostics/receipt_probe_failed.py','diagnostics/receipt_probe_failure.txt']
decision={
 'decision':'PASS','scope':'Source review and independently executed bounded reduced-CPU exactness evidence establish suitability only for the next separately assigned full-size CPU qualification.',
 'not_authorized':['native compilation','hardware execution','cold training','quality or BPB benefit claims','adoption','submission','full-size CPU execution within this task'],
 'baseline':{'commit':manifest['baseline_commit'],'train_py_sha256':manifest['source_sha256']['baseline']},
 'candidate':{'commit':manifest['candidate_commit'],'train_py_sha256':manifest['source_sha256']['candidate']},
 'input_manifest_sha256':sha(P/'INPUT_MANIFEST.json'),'input_files':manifest['included'],
 'independent_source_review':{'complete_reconstruction_hashes_match':True,'complete_AST_identity_after_exact_delta_removal':True,'unchanged_protected_definitions':21,'unchanged_time_call_count':14,'source_files_edited':False,'findings':['BooleanOptionalAction is default False and explicit OFF bypasses helper','Before step 5, process age is reconstructed from existing elapsed sample and origins, then margin/cap applied','Positive startup excess is retained','LR and WD share adjusted seconds; downstream schedule/optimizer formulas unchanged','Fifth-boundary reset and predictor clearing unchanged','Budget/stop/log/checkpoint/through-exit operations unchanged','Model/inference/load/checkpoint format unchanged']},
 'independent_numerical_evidence':{'attempt':'attempt2','status':'PASS','positive_checks':996,'expected_failure_negative_controls':7,'unexpected_failures':0,'schedule_grid_cases':600,'default_off_baseline_identity_cases':200,'boundary_cases':36,'step5_continuation_cases':12,'tensor_update_state_exact_cases':96,'dtype':'CPU IEEE float32; binary64 schedule scalars','tensor_comparison':'exact bytes for parameters, applied deltas, first/second moments; exact step integers and missing-gradient behavior','scalar_tolerance':'exact dyadic binary64 clock outputs; independently rearranged LR at most 4e-16 absolute; optimizer LR must match float32 bytes','fixture_parameter_sizes':[4,2,1],'fixture_updates_per_run':8,'fixture_modes':['baseline','candidate default','candidate explicit OFF','candidate ON'],'fixture_betas':[0.5,0.75],'fixture_eps':0.125,'fixture_weight_decay':0.125,'fixture_initial_lr':0.02,'missing_gradient_status':'second parameter None at steps 0,2,5; third always None and no state','peak_tracked_live_storage_bytes':mem['tracked_peak_storage_bytes'],'max_tensor_bytes':mem['max_tensor_bytes'],'tensor_limit_bytes':67108864,'threads':1,'interop_threads':1,'python':num['python'],'torch':num['torch'],'extraction_and_stubs':num['stubs'],'model_executed':False},
 'receipt_interpretation':{'status':'PASS for internal consistency only','checks':2671,'final_receipts':6,'final_ranks_each':4,'final_boundary_contract':5,'earlier_stub_receipts':6,'earlier_boundary_contract':2,'earlier_transition_not_qualifying':True,'source_hashes_and_argv_checked':True,'same_rank_baseline_default_off_records_identical':True,'all_rank_after_update_digests_match':True,'missing_recorded_gradients':False,'reduced_parameter_count':4523316,'reduced_parameter_tensor_count':65,'observed_forward_shape':[2,128],'configured_depth':8,'configured_width':128,'configured_context':1024,'configured_vocab':8192,'hidden_harness_executed':False,'hidden_harness_authenticated':False,'checkpoint_files_available':False,'checkpoint_byte_identity_claim':'matching supplied file SHA256 values only, separate from compared tensor digest records','sampling_semantics':'1ms fake time per call, delay after first optimizer update, schedule sampled at step start; rank0 logging adds samples','limitations':['BF16 CPU selection with FP32 master parameters','NKI CE no-op','synthetic data','sliced forwards and nominal token accounting','fake clock and artificial delays','observer implementations absent','all supplied ON LR multipliers on flat portion; independent high-age tests exercise cooldown']},
 'preserved_failures_and_limits':[
 {'artifact':'packet_copy/raw/failed-attempt1-base.json','failure':"All four observers fail: AttributeError: 'OwnedOptimizer' object has no attribute 'param_groups'; aggregate process return code 0 does not override worker failures",'treatment':'not successful training evidence'},
 {'artifact':'diagnostics/receipt_probe_failure.txt','failure':"Reviewer schema probe KeyError: 'checkpoint_file_sha256' on a nonzero rank",'treatment':'preserved; rank0-only schema corrected'},
 {'artifact':'receipt_audit/summary.json','failure':"First reviewer audit REFUSE due to KeyError: 'final_digests' on a nonzero rank",'treatment':'preserved; corrected audit uses all-rank parameter-after-update records'},
 {'artifact':'attempt2.stderr','failure':'CPU-information warning: failed to parse processor information from /proc/cpuinfo','treatment':'preserved in both numerical attempts; no resulting numerical test failure'},
 {'artifact':'attempt2/results.jsonl','failure':'Seven deliberately wrong controls fail their expected assertions','treatment':'expected failures required to validate sensitivity; the tensor_one_bit_mutation label actually replaces a byte and is claimed only as byte corruption'}
 ],
 'remaining_gates':['Separately assigned full-size CPU qualification at the stated 113256724-parameter/65-tensor 8x1024 target','Separately authorized native/compiler/cold-start and real hardware execution','Real budget, stop, checkpoint and through-exit qualification','Scientific quality/BPB evaluation before any adoption or submission','Any batch-ramp/EMA/dynamic-loss-scaling combination outside selected OFF profile'],
 'scientific_interpretation':'ON intentionally changes optimizer scheduling; no ON/OFF equivalence, throughput gain, BPB benefit, or causal explanation of earlier score variation is established.',
 'failed_requirements_within_pass_scope':[],
 'started_no_later_than_utc':'2026-09-25T22:36:44+00:00','decision_written_utc':now,'outer_session_limit_seconds':1200,
 'evidence_artifacts':[{ 'path':str(W/x),'sha256':sha(W/x)} for x in evidence_paths]
}
(W/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
checks={'decision_is_literal_PASS':decision['decision']=='PASS','report_starts_with_PASS':(W/'REPORT.md').read_text().startswith('**PASS'),'all_17_inputs_revalidated':len(manifest['included'])==17,'numerical_checks_pass':all(r['passed'] for r in rows),'receipt_audit_pass':audit['status']=='PASS','source_hashes_pass':True,'within_20_minutes_of_first_time_observation':(datetime.datetime.fromisoformat(now)-datetime.datetime.fromisoformat(decision['started_no_later_than_utc'])).total_seconds()<1200,'finished_utc':now}
assert all(v for k,v in checks.items() if k!='finished_utc')
(W/'final_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
files=[]
for f in sorted(W.rglob('*')):
 if f.is_file() and f.name not in ['artifact_manifest.json','finalize_review.stdout','finalize_review.stderr']:
  files.append(dict(path=str(f.relative_to(W)),bytes=f.stat().st_size,sha256=sha(f)))
(W/'artifact_manifest.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps(dict(decision='PASS',verification=checks,artifact_files=len(files),total_artifact_bytes=sum(f['bytes'] for f in files)),indent=2))
