"""Exact-byte P3 fallback staging via the proven lifetime-guarded transport."""
import ast,base64,datetime,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
def sha(raw):return hashlib.sha256(raw).hexdigest()
local=sys.argv[1:]==['--local-check']
assert local or (len(sys.argv)==3 and sys.argv[1]=='--execute')
raw=(HERE/'trn22-prepared.json').read_bytes()
assert sha(raw)=='da7a290e243239c432786ed64fc435e97f4a9217da034d9d75e2a3f7896fbc82'
packet=json.loads(raw);b=packet['binding'];files={n:base64.b64decode(v['b64']) for n,v in packet['files'].items()}
for n,value in files.items():assert sha(value)==packet['files'][n]['sha256'],n
assert sha(files['package/train.py'])==b['source_sha256']=='3637166b43a0b996e7eb7c71bb3b5d56e7f8119241b0278107a72787044dcc99'
assert sha(files['host_driver.py'])==b['driver_sha256']=='4c81569ada310d451bd8bbdba100cee5dda71f56048fef447218fb237449f58b'
proposal_path='research/v2/proposals/20260926-p3-component-timing/EXPERIMENT_PROPOSAL.json'
proposal=(REPO/proposal_path).read_bytes();assert sha(proposal)=='663e9848d56701e380a0a2ae4cfa61245c951e6800595aabd7ce28d7ee60a5ad'
review=(REPO/'research/v2/verification/20260926-p3-sprint-independent/REVIEW.md').read_bytes();assert sha(review)==b['independent_review_sha256']=='826268d46a4dc71ba893d4b382e5b62e56b78a7f5f095d9af53e1685320bb6da'
assignment_path='research/v2/assignments/20260926-sprint-next-dispatch.md' if local else sys.argv[2]
assignment=(REPO/assignment_path).read_bytes()
if local:assert b'P3 direct diagnostic' in assignment
else:assert sha(assignment)=='b474bd86479da5ef12e8812d8ad2f3a296c3bd825a093582a2fe439d2673b0c4' and b'P3 reviewed diagnostic' in assignment and b'3637166b' in assignment
b.update(stage_authorized=True,launch_authorized=True,release_file_created=True,full_proposal_present=True,submission_authorized=False,completed_proposal_path=proposal_path,completed_proposal_sha256=sha(proposal),operator_assignment_path=assignment_path,operator_assignment_sha256=sha(assignment))
# Retain the known unchanged harness as provenance only; this direct diagnostic
# never invokes it. These paths permit unchanged staging identity checks.
b['prepare_path']=b['prepare_source_path'];b['harness_path']=str(Path(b['prepare_path']).parent/'research/scratch/submission_rehearsal.py');b['harness_sha256']='7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b'
files['EXPERIMENT_PROPOSAL.json']=proposal
files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\n').encode()
files['RELEASE.json']=(json.dumps({'label':b['label'],'launch_authorized':True,'submission_authorized':False,'assignment_path':assignment_path,'assignment_sha256':sha(assignment),'source_sha256':b['source_sha256'],'independent_review_sha256':sha(review),'no_automatic_next_job':True},indent=2)+'\n').encode()
original=HERE.parent/'20260926-final-sprint-p0/stage_dispatch.py';raw=original.read_bytes();assert sha(raw)=='f80abd337b0b18a10e2a59a52144fd1d634e2f5a2421d4da2d1a70632a12d5c3'
tree=ast.parse(raw);body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
old="for key in ('NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'):assert Path(b['eval_environment'][key]).is_dir(),'EVAL_CACHE_ABSENT:'+key"
assert body.count(old)==1;body=body.replace(old,"assert b['no_eval'] and b['no_checkpoint'] and b['no_harness'],'DIRECT_DIAGNOSTIC_ONLY'")
assert body.count("(root/'cold-training-caches').mkdir(mode=0o700)")==1
body=body.replace("(root/'cold-training-caches').mkdir(mode=0o700)","(root/'cold-diagnostic-caches').mkdir(mode=0o700)")
body=body.replace("'eval_cache_mode':'EXISTING_SHARED_RESEARCH_CACHE_PRESERVED'","'eval_cache_mode':'NO_EVAL_DIRECT_COMPONENT_DIAGNOSTIC'")
spec=importlib.util.spec_from_file_location('connector',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'] and not any(v is True for v in state['breaker_tripped'].values())
payload={'binding':b,'controller':state,'files':{n:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for n,v in files.items()},'old_queue_pids':[3050509,3053905]}
wire=('P='+repr(payload)+'\n'+body).encode();ast.parse(body);assert not m.SENSITIVE.search(wire)
if local:
 print(json.dumps({'status':'LOCAL_CHECK_PASS','label':b['label'],'source_sha256':b['source_sha256'],'driver_sha256':sha(files['host_driver.py']),'completed_proposal_sha256':sha(proposal),'diagnostic_command':b['diagnostic_command'],'resource_reserve':b['resource_reserve'],'payload_sha256':sha(wire),'no_host_action':True}));raise SystemExit
prior=HERE.parent/'20260926-final-sprint-successors/trn22-on-parentcontrol-completed/results-bundle.json';r=json.loads(prior.read_bytes());p=r['postflight']
assert p['all_observed_pids_absent'] and not p['related_rows'] and not p['busy_job_rows'] and p['neuron_device_idle'] and p['lock_competitor_rc']==0 and p['fuser_rc']==1 and p['lock_inode']==832589
assert json.loads(r['files']['terminal.json']['text'])['error']=='TRAIN_EXIT_1'
attempt=HERE/'trn22-attempt.json';out=HERE/'trn22-dispatch.json';bound=HERE/'trn22-bound-packet.json';assert not any(p.exists() for p in (attempt,out,bound))
for p,v in ((attempt,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'payload_sha256':sha(wire),'prior_sha256':sha(prior.read_bytes())}),(bound,{'binding':b,'files':payload['files']})):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
argv,*_=m.load_transport('trn22');argv[-1]='exec /usr/bin/python3 -I -B -S -'
rr=subprocess.run(argv,input=wire,capture_output=True,timeout=60);assert not m.SENSITIVE.search(rr.stdout+rr.stderr)
receipt={'returncode':rr.returncode,'stderr':rr.stderr.decode(),'remote':json.loads(rr.stdout) if rr.returncode==0 else None,'stdout_if_error':rr.stdout.decode() if rr.returncode else None,'payload_sha256':sha(wire)}
with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'returncode':rr.returncode,'launch':receipt['remote']['launch'] if receipt['remote'] else None,'error':receipt['stderr'],'receipt_sha256':sha(out.read_bytes())}));raise SystemExit(rr.returncode)
