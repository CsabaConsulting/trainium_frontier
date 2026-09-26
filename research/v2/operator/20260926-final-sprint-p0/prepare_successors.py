"""Local mechanical preparation only; no connection, stage, RELEASE or launch."""
import ast, base64, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
FROZEN=Path('/home/poojith-devan/p0-sprint-fresh-review-8kEym4/MANIFEST.json')
raw=FROZEN.read_bytes();assert hashlib.sha256(raw).hexdigest()=='c28e772438afd2d8c46be423700dc6d8cdeea085e3465c901558c762cce356a3'
manifest=json.loads(raw)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):
 raw=path.read_bytes();assert sha(raw)==manifest[str(path.relative_to(REPO))];return raw
queue=[]
for host,variant in [('trn21','batch-lr14-trn21'),('trn21','batch-lr1-trn21'),('trn22','recursive-trn22'),('trn22','batch-lr14-trn22')]:
 group='20260926-sprint-p0-clean-recursive' if variant.startswith('recursive') else '20260926-sprint-p0-clean-batch'
 base=REPO/'research/v2/proposals'/group/'variants'/variant
 binding=json.loads(read(base/'NATIVE_BINDING.disabled.json'))
 current=json.loads((HERE/(host+'-bound-packet.json')).read_text())['binding']
 files={name:read(base/name) for name in ('host_driver.py','dispatch_driver.py','idle_guard.py','lifetime_guard.py','package/train.py','package/launch-command.txt','package/provenance.json')}
 env=dict(binding['train_environment'])
 keys=('NEURON_COMPILE_CACHE_URL','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR')
 for key in keys:env[key]=current['train_environment'][key]
 original=files['host_driver.py'];text=original.decode();tree=ast.parse(text)
 n=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAIN_ENV' for t in n.targets))
 assert n.lineno==n.end_lineno and ast.literal_eval(n.value)==binding['train_environment']
 lines=text.splitlines(keepends=True);old=lines[n.lineno-1];lines[n.lineno-1]='TRAIN_ENV = '+repr(env)+'\n';bound=''.join(lines)
 restored=bound.splitlines(keepends=True);restored[n.lineno-1]=old;assert ''.join(restored).encode()==original
 ast.parse(bound);files['host_driver.py']=bound.encode()
 assert sha(files['package/train.py'])==binding['source_sha256']
 assert files['package/launch-command.txt'].decode().strip()==json.loads(files['package/provenance.json'])['launch_command']
 files['EXPERIMENT_PROPOSAL.json']=read(base.parents[1]/'EXPERIMENT_PROPOSAL.json')
 binding.update(train_environment=env,original_reviewed_driver_sha256=sha(original),bound_driver_sha256=sha(files['host_driver.py']),cache_mode='REUSE_EXISTING_SAME_HOST_P0_RESEARCH_NAMESPACE_HIT_UNPROVEN',cache_owner_label=current['label'],cache_owner_root=current['root'],protected_cache_dependency=True,stage_authorized=False,launch_authorized=False,release_file_created=False,first_source_native_timing_pending=True)
 binding['dispatch_conditions']=['Predecessor training AND attached fresh20M naturally terminal; retrieve receipts and prove actual-owned descendant/compiler/device quiescence.','No higher-priority qualified K2 timing assignment.','Relevant first-source native timing/correctness observed; no unresolved actual failure or STOP/breaker.','Fresh original-lock hostname/ps/neuron-ls/fuser/source/argv/disk/RAM checks; no second launcher.','Science/source/command bytes unchanged; append exact attempt and actual first launch, then full1800+fresh20M; no inherited success claim.']
 binding['resource_reserve']={'runtime_floor_bytes':int(10.5*1024**3),'prospective_new_footprint_bytes':3*1024**3,'fresh_resources_not_checked':True}
 files['NATIVE_BINDING.json']=(json.dumps(binding,indent=2)+'\n').encode()
 out=HERE/(variant+'-successor-prepared.json');assert not out.exists()
 packet={'binding':binding,'files':{k:{'b64':base64.b64encode(v).decode(),'sha256':sha(v)} for k,v in files.items()},'review_manifest_sha256':sha(raw),'local_only':True,'host_calls':0,'ready_for_launch':False,'reason_not_ready':'Predecessor and first-source native timing still active; exact resource/idle checks necessarily future.'}
 with out.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
 queue.append({'host':host,'order_after_current':1+sum(1 for item in queue if item['host']==host),'variant':variant,'label':binding['label'],'source_sha256':binding['source_sha256'],'driver_sha256':sha(files['host_driver.py']),'prepared_packet':str(out.relative_to(HERE)),'packet_sha256':sha(out.read_bytes()),'status':'PREPARED_CONDITIONAL_NOT_STAGED','cache_dependency_label':current['label'],'k2_preemption_allowed':True})
out=HERE/'successors-prepared.json';assert not out.exists()
with out.open('x') as f:json.dump({'schema':'p0-conditional-local-preparation-v1','host_calls':0,'new_launcher_installed':False,'jobs':queue},f,indent=2);f.write('\n')
print(json.dumps({'status':'FOUR_MECHANICAL_SUCCESSORS_PREPARED_LOCALLY','jobs':queue,'no_stage_no_launch_no_release':True}))
