import ast, hashlib, json, pathlib
W=pathlib.Path('/work')
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
texts={r:(W/f'{r}_train.py').read_text() for r in ['baseline','candidate']}
trees={r:ast.parse(t) for r,t in texts.items()}
names=['GPTConfig','GPT','build_config','MuonAdamW','OwnedOptimizer','_submission_optimizer_install','_install_submission_training','_SubmissionEvalGPT','load_for_eval','save_checkpoint','current_toggles','restore_eval_toggles','lr_multiplier','muon_momentum','muon_beta2','demon_beta1_scale','batch_ramp_grad_accum','weight_ema_state','loss_scale_transition','_muon_math','_muon_lr_g']
result=[]
for name in names:
 nodes={r:next(n for n in tr.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name) for r,tr in trees.items()}
 sources={r:ast.get_source_segment(texts[r],n) for r,n in nodes.items()}
 result.append(dict(name=name,byte_identical=sources['baseline']==sources['candidate'],sha256={r:digest(s) for r,s in sources.items()},lines={r:[n.lineno,n.end_lineno] for r,n in nodes.items()}))
clocks={r:[ast.unparse(n) for n in ast.walk(tr) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='time'] for r,tr in trees.items()}
result.append(dict(name='all_time_calls_in_complete_file',identical=clocks['baseline']==clocks['candidate'],calls=clocks))
changed={'added':[],'removed':[]}
for l in (W/'packet_copy/baseline-to-candidate.diff').read_text().splitlines():
 if l.startswith('+') and not l.startswith('+++'):changed['added'].append(l[1:])
 if l.startswith('-') and not l.startswith('---'):changed['removed'].append(l[1:])
assert all(x.get('byte_identical',x.get('identical',False)) for x in result)
(W/'source_audit.json').write_text(json.dumps(dict(status='PASS',protected_definitions=result,exact_changed_lines=changed),indent=2)+'\n')
print(json.dumps(dict(status='PASS',protected_definitions=len(names),unchanged_time_call_count=len(clocks['baseline']),added_lines=len(changed['added']),removed_lines=len(changed['removed']))))
