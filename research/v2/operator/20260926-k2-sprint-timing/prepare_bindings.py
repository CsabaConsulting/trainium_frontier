"""Mechanical, local-only OFF/ON45 preparation; no native calls or launch."""
import ast, base64, hashlib, json, shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
SOURCE=REPO/'research/v2/lanes/k2-final-sprint/train.py'
PROPOSAL=REPO/'research/v2/proposals/20260926-k2-final-sprint-timing/EXPERIMENT_PROPOSAL.json'
ASSIGNMENT=REPO/'research/v2/assignments/20260926-k2-sprint-timing.md'
REVIEW=REPO/'research/v2/verification/20260926-k2-sprint-independent/REVIEW.md'
def sha(x):return hashlib.sha256(x).hexdigest()
source=SOURCE.read_bytes();assert sha(source)=='63925519ac6c0aab1a1d598e96fef3f4413fbd083d9b6d177901a2e972060db5'
proposal=json.loads(PROPOSAL.read_bytes());assert proposal['candidate']['source_sha256']==sha(source)
assert sha(REVIEW.read_bytes())=='2ddb87ac3c38897abf306e323d06c747b0db41423e2f6126ede94e98cadcf502'
assert 'PASS for the short native test only' in REVIEW.read_text()
rows=[]
for host in ('trn21','trn22'):
 original=json.loads((HERE.parent/'20260926-final-sprint-p0'/(host+'-bound-packet.json')).read_text())
 ob=original['binding'];oldfiles={n:base64.b64decode(r['b64']) for n,r in original['files'].items()}
 for name,raw in oldfiles.items():assert sha(raw)==original['files'][name]['sha256']
 for mode in ('off','on'):
  label='fs-20260926-k2-sprint-'+mode+'-'+host+'-r1'
  root=str(Path(ob['root']).parent/(label+'-packet'))
  command=proposal['candidate']['launch_command']
  if mode=='off':command=command.replace('--k2-embedding','--no-k2-embedding')
  tokens=shlex.split(command);assert tokens[tokens.index('--num-steps')+1]=='45'
  prov={'label':label,'train_sha256':sha(source),'launch_command':command,'base_sha256':'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43','scientific_causal_change':'K2 embedding backward flag only','submission_authorized':False,'run_purpose':'45-step-whole-window-timing-not-quality'}
  provbytes=(json.dumps(prov,indent=2)+'\n').encode();cmdbytes=(command+'\n').encode()
  text=oldfiles['host_driver.py'].decode().replace(ob['root'],root).replace(ob['label'],label).replace(ob['source_sha256'],sha(source))
  text=text.replace(sha(oldfiles['package/launch-command.txt']),sha(cmdbytes)).replace(sha(oldfiles['package/provenance.json']),sha(provbytes))
  tree=ast.parse(text)
  node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAIN_ENV' for t in n.targets))
  env=ast.literal_eval(node.value)
  keys={'NEURON_COMPILE_CACHE_URL':'compiler','NEURON_COMPILE_CACHE_DIR':'compiler','TORCH_NEURONX_NEFF_CACHE_DIR':'neff','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR':'local','TORCH_NEURONX_HLO_CACHE_DIR':'hlo','TORCHINDUCTOR_CACHE_DIR':'inductor','TORCH_NEURONX_DEBUG_DIR':'debug'}
  for key,leaf in keys.items():env[key]=(root+'/cold-training-caches/'+leaf if mode=='on' else ob['train_environment'][key])
  lines=text.splitlines(keepends=True);assert node.lineno==node.end_lineno
  lines[node.lineno-1]='TRAIN_ENV = '+repr(env)+'\n';text=''.join(lines)
  # The training invocation, exact pinned guardian and final ownership/idle
  # handling remain unchanged. Replace only the attached eval phase with a
  # raw timing-complete terminal, as explicitly required by the proposal.
  begin=text.index("    eval_fd=guard.require_idle_under_inherited_lock(")
  end=text.index('except BaseException as exc:',begin)
  removed=text[begin:end]
  assert "rc=phase('fresh-public20m',EVAL_COMMAND,EVAL_ENV,guardian)" in removed
  terminal="    record('terminal.json',{'utc':utc(),'label':LABEL,'status':'TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS','train_exit_code':rc,'no_eval':True,'no_automatic_full':True,'submission_authorized':False})\n"
  text=text[:begin]+terminal+text[end:]
  ast.parse(text)
  assert "phase('fresh-public20m'" not in text
  # All helper functions and safety/error/finally code remain identical after
  # constant rebinding; no throughput estimator is installed on the host.
  files={n:oldfiles[n] for n in ('dispatch_driver.py','idle_guard.py','lifetime_guard.py')}
  files.update({'host_driver.py':text.encode(),'package/train.py':source,'package/launch-command.txt':cmdbytes,'package/provenance.json':provbytes,'EXPERIMENT_PROPOSAL.json':PROPOSAL.read_bytes()})
  binding={k:v for k,v in ob.items() if k not in ('eval_command','eval_environment','fresh_cache_directories','original_reviewed_driver_sha256','cold_bound_driver_sha256','resource_reserve')}
  binding.update(schema='conditional-k2-sprint45-v1',host=host,label=label,root=root,source_sha256=sha(source),train_environment=env,stage_authorized=False,launch_authorized=False,release_file_created=False,submission_authorized=False,source_adoption=False,run_purpose='whole-step-timing-only',no_eval=True,steps=45,scientific_patch='K2_embedding_backward',cold_mode='ON_FRESH_EMPTY_SIX_PRIVATE_CACHES' if mode=='on' else 'OFF_REUSES_SAME_HOST_P0_RESEARCH_NAMESPACE_HIT_UNPROVEN',protected_cache_owner_label=None if mode=='on' else ob['label'],fresh_cache_directories=sorted(set(env[k] for k in keys)) if mode=='on' else [],independent_review_sha256=sha(REVIEW.read_bytes()),operator_assignment_sha256=sha(ASSIGNMENT.read_bytes()),driver_sha256=sha(text.encode()),launch_command=command)
  binding['train_command']=[x.replace(ob['root'],root) for x in ob['train_command']]
  binding['outer_command']=[x.replace(ob['root'],root) for x in ob['outer_command']]
  binding['resource_reserve']={'runtime_floor_bytes':int(10.5*1024**3),'projected_new_footprint_bytes':3*1024**3,'initial_floor_bytes':int(13.5*1024**3),'fresh_host_resources_pending':True}
  binding['timing']=proposal['timing']
  binding['native_gate_conditions']=['Exactly one selected host for both OFF and ON; same source/runtime/seed42/science except K2 flag.','Current P0 train AND attached fresh20M terminal/retrieved and explicit descendant/compiler/device quiescence before staging.','First-source review PASS authenticated; original lifetime lock, fresh ps/neuron-ls/fuser/STOP/hash/resource checks.','OFF naturally completes and raw45 boundary/exit/accounting captured before ON; failure is not a rate sample.','ON six directories absent then created empty; actual rank environments must match.','No full quality launch from this driver; paired45 finite completion/accounting and whole-window ON/OFF>=1.03 required separately.']
  files['NATIVE_BINDING.json']=(json.dumps(binding,indent=2)+'\n').encode()
  out=HERE/(host+'-'+mode+'-prepared.json');assert not out.exists()
  packet={'binding':binding,'files':{n:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for n,v in files.items()},'mechanical_derivation':{'parent_driver_sha256':sha(oldfiles['host_driver.py']),'changes':['literal root/label/source/package-hash binding','TRAIN_ENV cache paths: OFF retained P0 namespace; ON unique empty namespace','attached eval block omitted; timing-only terminal replaces it'],'shared_guard_unchanged':True,'scientific_source_modified':False},'local_only':True,'host_calls':0,'native_started':False}
  with out.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
  rows.append({'host':host,'mode':mode,'label':label,'packet':out.name,'packet_sha256':sha(out.read_bytes()),'driver_sha256':sha(text.encode()),'source_sha256':sha(source),'state':'PREPARED_CONDITIONAL_NOT_STAGED'})
for host in ('trn21','trn22'):
 off=json.loads((HERE/(host+'-off-prepared.json')).read_text())['binding']['launch_command']
 on=json.loads((HERE/(host+'-on-prepared.json')).read_text())['binding']['launch_command']
 assert off.replace('--no-k2-embedding','--k2-embedding')==on
out=HERE/'PREPARED.json';assert not out.exists()
with out.open('x') as f:json.dump({'proposal_sha256':sha(PROPOSAL.read_bytes()),'assignment_sha256':sha(ASSIGNMENT.read_bytes()),'independent_review_sha256':sha(REVIEW.read_bytes()),'one_pair_on_first_qualified_free_host_only':True,'actual_host_selected':None,'packets':rows,'native_started':False,'no_host_calls':True},f,indent=2);f.write('\n')
print(json.dumps({'status':'FOUR_LOCAL_ALTERNATIVES_FOR_ONE_SAME_HOST_PAIR','packets':rows,'no_eval_no_launch':True}))
