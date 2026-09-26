"""Plain exact-byte successor staging using the executed P0 remote body."""
import ast,base64,datetime,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=Path('/home/poojith-devan/Desktop/Trainium')
choice=sys.argv[1]
choices={
 'trn21-batch-lr14':('20260926-final-sprint-p0/batch-lr14-trn21-successor-prepared.json','b696405b08f57f541fad3090d4b90951bad77df605f902363233cefd87a5ba6e'),
 'trn22-off':('20260926-k2-sprint-timing/trn22-off-prepared.json','bf70a65e7ed3005d80f7556c5203d9f9bcf8190b21dbf0179928624bddd177e0'),
 'trn22-on':('20260926-k2-sprint-timing/trn22-on-prepared.json','85d55cd4d6877a6fd0ff0f9a89b7a3d7ed5392803b465244b77952565fc7799d'),
 'trn22-e1e2-control':('20260926-e1e2-prospective-controls/trn22-prepared.json','cf88de2158cff9ed7976630005147adbfd995baef433e977d7889933066fcaf6')}
rel,expected=choices[choice];raw=(HERE.parent/rel).read_bytes()
def sha(x):return hashlib.sha256(x).hexdigest()
assert sha(raw)==expected,'PREPARED_PACKET_DRIFT'
packet=json.loads(raw);b=packet['binding'];host=b['host'];files={n:base64.b64decode(v['b64']) for n,v in packet['files'].items()}
for n,v in files.items():assert sha(v)==packet['files'][n]['sha256'],'FILE_DRIFT:'+n
assert sha(files['package/train.py'])==b['source_sha256']
assert sha(files['lifetime_guard.py'])=='5df6897deeaea3435a963f79b912ce49f320d3a65624f60a89762249c67ecb6e'
assert sha(files['idle_guard.py'])=='adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737'
assignment=REPO/'research/v2/assignments/20260926-sprint-next-dispatch.md';auth=assignment.read_bytes();assert b'First qualified free host' in auth and b'LR.021' in auth
if choice in ('trn22-off','trn22-on'):
 review=(REPO/'research/v2/verification/20260926-k2-sprint-independent/REVIEW.md').read_bytes();assert sha(review)=='2ddb87ac3c38897abf306e323d06c747b0db41423e2f6126ede94e98cadcf502'
else:review=(REPO/'research/v2/verification/20260926-p0-sprint-fresh-independent/REVIEW.md').read_bytes();assert b'PASS within bounded source/CPU scope' in review
if choice=='trn22-e1e2-control':
 scored=(REPO/'workflow/SCORED_BASE_2026-09-26.md').read_bytes()
 assert b['source_sha256']=='e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43' and b'e1e2' in scored
 assert sha(files['package/launch-command.txt'])=='b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd'
 b['source_qualification']='UNCHANGED_AUDITED_SCORED_E1E2_NO_NEW_SOURCE_REVIEW'
 b['scored_source_receipt_sha256']=sha(scored)
 b['parent_control_plane_review_sha256']=b.pop('independent_review_sha256',sha(review))
b.update(stage_authorized=True,launch_authorized=True,submission_authorized=False,operator_assignment_sha256=sha(auth),fresh_cache_directories=b.get('fresh_cache_directories',[]))
b['resource_reserve']={'runtime_floor_bytes':11274289152,'projected_new_footprint_bytes':3221225472,'initial_floor_bytes':14495514624,'basis':'Unchanged prepared per-job13.5GiB initial/10.5GiB runtime floor; pair admission additional and explicit.'}
if choice=='trn22-off':
 admission=json.loads((HERE/'k2-pair-admission.json').read_text())
 assert admission['host']=='trn22' and admission['on_initial_floor_bytes']==14495514624 and admission['off_growth_reserve_bytes']>0
 b['whole_pair_admission']=admission;b['resource_reserve']['initial_floor_bytes']=max(14495514624,admission['on_initial_floor_bytes']+admission['off_growth_reserve_bytes'])
files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\n').encode()
files['RELEASE.json']=(json.dumps({'label':b['label'],'launch_authorized':True,'submission_authorized':False,'assignment_path':str(assignment.relative_to(REPO)),'assignment_sha256':sha(auth),'independent_review_sha256':sha(review),'source_sha256':b['source_sha256'],'guardian_sha256':sha(files['lifetime_guard.py']),'no_automatic_next_job':True},indent=2)+'\n').encode()
if choice=='trn22-e1e2-control':
 release=json.loads(files['RELEASE.json']);release['parent_control_plane_review_sha256']=release.pop('independent_review_sha256');release['scored_source_receipt_sha256']=sha(scored);release['source_qualification']=b['source_qualification'];files['RELEASE.json']=(json.dumps(release,indent=2)+'\n').encode()
template=HERE.parent/'20260926-final-sprint-p0/stage_dispatch.py';template_raw=template.read_bytes();assert sha(template_raw)=='f80abd337b0b18a10e2a59a52144fd1d634e2f5a2421d4da2d1a70632a12d5c3'
tree=ast.parse(template_raw);body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
old="for key in ('NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'):assert Path(b['eval_environment'][key]).is_dir(),'EVAL_CACHE_ABSENT:'+key"
new="phase_envs=([b['train_environment']] if not b['fresh_cache_directories'] else [])+([b['eval_environment']] if 'eval_environment' in b else [])\nfor env in phase_envs:\n for key in ('NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'):assert Path(env[key]).is_dir(),'EXISTING_PHASE_CACHE_ABSENT:'+key"
assert body.count(old)==1;body=body.replace(old,new).replace("(root/'cold-training-caches').mkdir(mode=0o700)","if b['fresh_cache_directories']:(root/'cold-training-caches').mkdir(mode=0o700)")
body=body.replace("'eval_cache_mode':'EXISTING_SHARED_RESEARCH_CACHE_PRESERVED'","'eval_cache_mode':'NO_EVAL_TIMING_ONLY' if b.get('no_eval') else 'EXISTING_SHARED_RESEARCH_CACHE_PRESERVED'")
spec=importlib.util.spec_from_file_location('connector',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'] and not any(x is True for x in state['breaker_tripped'].values())
payload={'binding':b,'controller':state,'files':{n:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for n,v in files.items()},'old_queue_pids':[2983496] if host=='trn21' else [3050509,3053905]}
wire=('P='+repr(payload)+'\n'+body).encode();ast.parse(body);assert not m.SENSITIVE.search(wire)
if sys.argv[2:]==['--local-check']:
 print(json.dumps({'status':'LOCAL_CHECK_PASS','choice':choice,'source_sha256':b['source_sha256'],'driver_sha256':sha(files['host_driver.py']),'initial_floor_bytes':b['resource_reserve']['initial_floor_bytes'],'payload_sha256':sha(wire)}));raise SystemExit
assert sys.argv[2:]==['--execute']
predecessor=HERE/('trn22-off-completed/results-bundle.json' if choice=='trn22-on' else host+'-p0-completed/results-bundle.json')
prior=json.loads(predecessor.read_text());p=prior['postflight'];assert p['all_observed_pids_absent'] and not p['related_rows'] and not p['busy_job_rows'] and p['neuron_device_idle'] and p['lock_competitor_rc']==0 and p['fuser_rc']==1
terminal=json.loads(prior['files']['terminal.json']['text'])
if choice=='trn22-on':assert terminal['status']=='TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS' and terminal['train_exit_code']==0
else:assert terminal['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and terminal['eval_exit_code']==0
attempt=HERE/(choice+'-attempt.json');out=HERE/(choice+'-dispatch.json');bound=HERE/(choice+'-bound-packet.json');assert not any(p.exists() for p in (attempt,out,bound))
for p,v in ((attempt,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'payload_sha256':sha(wire),'prior_sha256':sha(predecessor.read_bytes())}),(bound,{'binding':b,'files':payload['files']})):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
argv,*_=m.load_transport(host);argv[-1]='exec /usr/bin/python3 -I -B -S -'
r=subprocess.run(argv,input=wire,capture_output=True,timeout=60);assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt={'returncode':r.returncode,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None,'payload_sha256':sha(wire)}
with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'choice':choice,'returncode':r.returncode,'launch':receipt['remote']['launch'] if receipt['remote'] else None,'error':receipt['stderr'],'receipt_sha256':sha(out.read_bytes())}));raise SystemExit(r.returncode)
