"""Plain copy/hash dispatch of two independently reviewed P0 variants.

Only TRAIN_ENV cache path literals are prospectively rebound for a genuine
cold first run. Scientific source, science argv, accounting and guard logic
are byte-identical to the single-reviewed variants.
"""
import ast, base64, datetime, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
REVIEW=Path('/home/poojith-devan/Documents/ChatGPT/Trainium/research/v2/verification/20260926-p0-sprint-fresh-independent/REVIEW.md')
ASSIGNMENT=REPO/'research/v2/assignments/20260926-final-sprint-p0.md'
MANIFEST=Path('/home/poojith-devan/p0-sprint-fresh-review-8kEym4/MANIFEST.json')
manifest_raw=MANIFEST.read_bytes();manifest=json.loads(manifest_raw)
assert hashlib.sha256(manifest_raw).hexdigest()=='c28e772438afd2d8c46be423700dc6d8cdeea085e3465c901558c762cce356a3'
review=REVIEW.read_bytes();assignment=ASSIGNMENT.read_bytes()
assert b'PASS within bounded source/CPU scope' in review and b'trn21 first' in assignment and b'trn22 first' in assignment
host=sys.argv[1];assert host in ('trn21','trn22')
variant={'trn21':'20260926-sprint-p0-clean-recursive/variants/recursive-trn21','trn22':'20260926-sprint-p0-clean-batch/variants/batch-lr1-trn22'}[host]
base=REPO/'research/v2/proposals'/variant
def sha(raw):return hashlib.sha256(raw).hexdigest()
def checked(path):
 raw=path.read_bytes();assert sha(raw)==manifest[str(path.relative_to(REPO))],'FROZEN_INPUT_DRIFT:'+str(path);return raw
names=['host_driver.py','dispatch_driver.py','idle_guard.py','lifetime_guard.py','package/train.py','package/launch-command.txt','package/provenance.json']
files={name:checked(base/name) for name in names}
b=json.loads(checked(base/'NATIVE_BINDING.disabled.json'))
assert b['host']==host and b['lock_identity']==[66305,596823 if host=='trn21' else 832589]
assert sha(files['package/train.py'])==b['source_sha256'] and sha(files['lifetime_guard.py'])=='5df6897deeaea3435a963f79b912ce49f320d3a65624f60a89762249c67ecb6e'
env=dict(b['train_environment'])
keys={'NEURON_COMPILE_CACHE_URL':'compiler','NEURON_COMPILE_CACHE_DIR':'compiler','TORCH_NEURONX_NEFF_CACHE_DIR':'neff','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR':'local','TORCH_NEURONX_HLO_CACHE_DIR':'hlo','TORCHINDUCTOR_CACHE_DIR':'inductor','TORCH_NEURONX_DEBUG_DIR':'debug'}
for key,leaf in keys.items():env[key]=b['root']+'/cold-training-caches/'+leaf
source=files['host_driver.py'].decode();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAIN_ENV' for t in n.targets))
assert ast.literal_eval(node.value)==b['train_environment'] and node.lineno==node.end_lineno
lines=source.splitlines(keepends=True)
lines[node.lineno-1]='TRAIN_ENV = '+repr(env)+'\n'
bound=''.join(lines)
restored=bound.splitlines(keepends=True);restored[node.lineno-1]=source.splitlines(keepends=True)[node.lineno-1]
assert ''.join(restored)==source,'NOT_ENV_ONLY_REBIND'
ast.parse(bound);files['host_driver.py']=bound.encode()
b.update(train_environment=env,stage_authorized=True,launch_authorized=True,host_reserved=False,source_adoption=False,submission_authorized=False,independent_review_sha256=sha(review),operator_assignment_sha256=sha(assignment),fresh_private_training_cache=True)
b['original_reviewed_driver_sha256']=sha(source.encode());b['cold_bound_driver_sha256']=sha(files['host_driver.py'])
b['resource_reserve']={'runtime_floor_bytes':int(10.5*1024**3),'projected_new_footprint_bytes':3*1024**3,'initial_floor_bytes':int(13.5*1024**3),'basis':'Same full113M cold-run staging reserve used for completed clean e1e2 replay;25GB maintenance target is not a launch gate.'}
b['fresh_cache_directories']=sorted(set(env[k] for k in keys))
files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\n').encode()
proposal=base.parents[1]/'EXPERIMENT_PROPOSAL.json'
files['EXPERIMENT_PROPOSAL.json']=checked(proposal)
files['RELEASE.json']=(json.dumps({'label':b['label'],'launch_authorized':True,'submission_authorized':False,'assignment_path':str(ASSIGNMENT.relative_to(REPO)),'assignment_sha256':sha(assignment),'independent_review_sha256':sha(review),'source_sha256':b['source_sha256'],'guardian_sha256':sha(files['lifetime_guard.py']),'no_automatic_next_job':True},indent=2)+'\n').encode()
spec=importlib.util.spec_from_file_location('strict_connector',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'] and not any(v is True for v in state['breaker_tripped'].values())
payload={'binding':b,'controller':state,'files':{name:{'b64':base64.b64encode(raw).decode(),'sha256':sha(raw)} for name,raw in files.items()},'old_queue_pids':[2983496] if host=='trn21' else [3050509,3053905]}
body=r'''
import base64,datetime,fcntl,hashlib,json,os,socket,stat,subprocess,time
from pathlib import Path
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
b=P['binding'];root=Path(b['root']);prep=Path(b['prepare_path']);started=utc()
assert socket.gethostname()==b['hostname'] and os.getuid()==1000,'WRONG_HOST_UID'
age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(P['controller']['utc'])).total_seconds();assert 0<=age<=60,'CONTROLLER_RECEIPT_STALE'
lockfd=os.open(b['lock_path'],os.O_RDONLY|os.O_NOFOLLOW);lock=os.fstat(lockfd)
assert [lock.st_dev,lock.st_ino]==b['lock_identity'],'ORIGINAL_LOCK_IDENTITY'
fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
for base in (Path('/home/ubuntu'),prep.parent,root,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
 for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):assert not os.path.lexists(base/rel),'STOP:'+str(base/rel)
assert not os.path.lexists(root),'UNIQUE_ROOT_ALREADY_EXISTS'
for parent in reversed(root.parents):assert not stat.S_ISLNK(parent.lstat().st_mode),'SYMLINK_PARENT:'+str(parent)
assert root.parent.is_dir(),'MISSING_EXISTING_OUT_PARENT'
for pid in P['old_queue_pids']:assert not Path('/proc/'+str(pid)).exists(),'OLD_QUEUE_STILL_PRESENT'
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=10)
assert ps.returncode==0,ps.stderr
terms=('torchrun','train.py','host_driver.py','host_queue.py','submission_rehearsal.py','preflight_dryrun.py','compare_checkpoints.py','neuronx-cc','walrus_driver','run_job.py','contained.py','prepare.py eval-public')
jobs=[line for line in ps.stdout.splitlines() if any(k in line for k in terms)];assert not jobs,'BUSY_PS:'+repr(jobs)
nls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=15);assert nls.returncode==0,nls.stderr
devices=json.loads(nls.stdout);assert len(devices)==1 and devices[0].get('neuron_processes')==[] and devices[0].get('logical_neuroncore_config')==2 and devices[0].get('nc_count')==4,'DEVICE_BUSY_OR_UNKNOWN'
fu=subprocess.run(['/usr/bin/fuser','/dev/neuron0'],capture_output=True,text=True,timeout=10);assert fu.returncode==1 and not fu.stdout.strip() and not fu.stderr.strip(),'DEVICE_HOLDER_OR_FUSER_ERROR'
v=os.statvfs('/');free=v.f_bavail*v.f_frsize
assert free>=b['resource_reserve']['initial_floor_bytes'],'INSUFFICIENT_COLD_RESERVE:'+str(free)
mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'));assert mem>=64*1024**3,'MEMORY_HEADROOM'
assert digest(prep)==b['prepare_sha256'] and digest(Path(b['harness_path']))==b['harness_sha256'],'PREPARE_OR_HARNESS_DRIFT'
project=Path(b['harness_path']).resolve().parents[2];assert project==prep.parent.resolve() and root.resolve().is_relative_to(project/'out'),'HARNESS_CONTAINMENT'
for path in b['fresh_cache_directories']:assert not os.path.lexists(path),'COLD_CACHE_PREEXISTS'
for key in ('NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'):assert Path(b['eval_environment'][key]).is_dir(),'EVAL_CACHE_ABSENT:'+key
for path in ('/usr/bin/python3','/usr/bin/flock','/usr/bin/timeout','/home/ubuntu/workspace/native_venv/bin/python','/home/ubuntu/workspace/native_venv/bin/torchrun'):assert os.access(path,os.X_OK),'EXECUTABLE_UNAVAILABLE:'+path
root.mkdir(mode=0o700);(root/'package').mkdir(mode=0o700);(root/'metrics-training').mkdir(mode=0o700)
for name,item in P['files'].items():
 path=root/name;raw=base64.b64decode(item['b64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 with path.open('xb') as f:f.write(raw)
 os.chmod(path,0o444);assert digest(path)==item['sha256'],'STAGED_HASH_MISMATCH:'+name
(root/'cold-training-caches').mkdir(mode=0o700)
caches=[]
for path in b['fresh_cache_directories']:
 p=Path(path);p.mkdir(mode=0o700);assert not list(p.iterdir());s=p.stat();caches.append({'path':path,'device':s.st_dev,'inode':s.st_ino,'created_empty':True})
receipt={'started_utc':started,'utc':utc(),'host':b['host'],'label':b['label'],'original_lock':{'device':lock.st_dev,'inode':lock.st_ino,'held_during_preflight':True},'free_bytes':free,'mem_available_bytes':mem,'resource_reserve':b['resource_reserve'],'source_sha256':b['source_sha256'],'prepare_sha256':b['prepare_sha256'],'harness_sha256':b['harness_sha256'],'files':{n:r['sha256'] for n,r in P['files'].items()},'fresh_cache_absent_then_created_empty':caches,'ps_job_rows':jobs,'ps_sha256':hashlib.sha256(ps.stdout.encode()).hexdigest(),'neuron_ls':devices,'fuser':{'returncode':fu.returncode,'stdout':fu.stdout,'stderr':fu.stderr},'controller':P['controller'],'old_queue_pids_absent':P['old_queue_pids'],'eval_cache_mode':'EXISTING_SHARED_RESEARCH_CACHE_PRESERVED'}
with (root/'operator-preflight.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
fcntl.flock(lockfd,fcntl.LOCK_UN);os.close(lockfd)
# The reviewed dispatch immediately reacquires and authenticates this original
# inode, checks ps/neuron-ls again INSIDE it, then holds it through both phases.
with (root/'dispatch.log').open('xb') as out:
 child=subprocess.Popen(['/usr/bin/python3','-I','-B',str(root/'dispatch_driver.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
launch={'utc':utc(),'host':b['host'],'label':b['label'],'root':str(root),'dispatch_pid':child.pid,'status':'DETACHED_DISPATCH_STARTED','source_sha256':b['source_sha256'],'automatic_next_job':False}
with (root/'operator-launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n')
print(json.dumps({'preflight':receipt,'launch':launch}))
'''
ast.parse(body);wire=('P='+repr(payload)+'\n'+body).encode();assert not m.SENSITIVE.search(wire)
if sys.argv[2:]==['--local-check']:
 print(json.dumps({'status':'LOCAL_CHECK_PASS','host':host,'label':b['label'],'root':b['root'],'source_sha256':b['source_sha256'],'driver_sha256':sha(files['host_driver.py']),'only_reviewed_train_env_path_rebind':True,'payload_sha256':sha(wire),'required_free_bytes':b['resource_reserve']['initial_floor_bytes']}));raise SystemExit(0)
assert sys.argv[2:]==['--execute']
out=HERE/(host+'-dispatch.json');attempt=HERE/(host+'-attempt.json');bound_path=HERE/(host+'-bound-packet.json')
assert not out.exists() and not attempt.exists() and not bound_path.exists(),'LOCAL_ATTEMPT_ALREADY_EXISTS'
for path,value in ((bound_path,{'binding':b,'files':payload['files'],'review_sha256':sha(review),'assignment_sha256':sha(assignment)}),(attempt,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':host,'payload_sha256':sha(wire),'source_sha256':b['source_sha256']})):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
argv,*_=m.load_transport(host);argv[-1]='exec /usr/bin/python3 -I -B -S -'
p=subprocess.run(argv,input=wire,capture_output=True,timeout=60)
assert not m.SENSITIVE.search(p.stdout+p.stderr)
receipt={'returncode':p.returncode,'stderr':p.stderr.decode(),'remote':json.loads(p.stdout) if p.returncode==0 else None,'stdout_if_error':p.stdout.decode() if p.returncode else None,'payload_sha256':sha(wire)}
with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'host':host,'returncode':p.returncode,'launch':receipt['remote']['launch'] if receipt['remote'] else None,'error':receipt['stderr'],'receipt_sha256':sha(out.read_bytes())}));raise SystemExit(p.returncode)
