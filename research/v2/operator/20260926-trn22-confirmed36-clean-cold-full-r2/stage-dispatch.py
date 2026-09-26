"""Prepared full-cold transport; actual accepted r2 cold archive and scoped release required."""
import base64,hashlib,importlib.util,json,math,re,shlex,subprocess,sys
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('NORMAL_PYTHON_REQUIRED')
repo=Path('/home/poojith-devan/Desktop/Trainium');own=Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
rel='research/v2/proposals/20260926-confirmed36-clean-cold-full-trn22-r2';d=own/rel
frozen='d74ba2f8d367c61a00f172f9cb7ca0e1145140ca'
pins={'NATIVE_BINDING.disabled.json':'65ad17ce3a00e3de228d5fb7296f09a6bf5a8578c2b36ef5dccf57859f4c60bb','EXPERIMENT_PROPOSAL.json':'475e62406dc419dea2e218e1a5a631b80a55c8571efb482ea0690d4d913d0069','host_driver.py':'ee5bc9f7a984c77b2c489eb5f361be37949de29f7b48637c58eb3a384c27dd90','MANIFEST.sha256.json':'92c37438fc3eed5519e28a583f4a45207fd7199925b9f7b037edfb5d59a4b4ce'}
for n,h in pins.items():
 raw=(d/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==h,n
 assert raw==subprocess.check_output(['git','show',frozen+':'+rel+'/'+n],cwd=own),'FROZEN_PACKET_CHANGED'
manifest_raw=(d/'MANIFEST.sha256.json').read_bytes()
for n,h in json.loads(manifest_raw)['files'].items():
 raw=(d/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==h,n
 assert raw==subprocess.check_output(['git','show',frozen+':'+rel+'/'+n],cwd=own),n
b=json.loads((d/'NATIVE_BINDING.disabled.json').read_bytes())
assert b['host']=='trn22' and b['label']=='fs-20260926-confirmed36-clean-coldfull-trn22-r2'
source=subprocess.check_output(['git','show',b['source_commit']+':'+b['source_git_path']],cwd=own);assert hashlib.sha256(source).hexdigest()==b['source_sha256']
files={n:(d/n).read_bytes() for n in ('NATIVE_BINDING.disabled.json','EXPERIMENT_PROPOSAL.json','host_driver.py','MANIFEST.sha256.json','package/launch-command.txt','package/provenance.json')}
files['package/train.py']=source
assert hashlib.sha256(files['package/launch-command.txt']).hexdigest()==b['package_launch_sha256']
assert hashlib.sha256(files['package/provenance.json']).hexdigest()==b['package_provenance_sha256']
assert files['package/launch-command.txt'].decode().strip()==b['launch_command']
assert shlex.split(b['launch_command'])[1:]==b['training_command']
for n,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
 raw=(own/'research/v2/operator/20260926-common-host-idle-guard'/n).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==b[key];files[n]=raw
if sys.argv[1:]==['--local-check']:
 print(json.dumps({'status':'LOCAL_EXACT_FULL_COLD_TRANSPORT_READY_ONLY','files':len(files),'host_calls':False,'launch_authorized':False,'future_actual_cold_r2_archive_required':True}));sys.exit(0)
assert len(sys.argv)==4,'EXACT_COLD_R2_ARCHIVE_COMMIT_RELEASE_COMMIT_AND_PATH_REQUIRED'
prior_commit,release_commit,release_path=sys.argv[1:]
assert all(re.fullmatch(r'[0-9a-f]{40}',v) for v in (prior_commit,release_commit))
assert release_path.startswith('research/v2/verification/') and release_path.endswith('.md') and '..' not in Path(release_path).parts
assignment=(repo/release_path).read_bytes()
assert assignment==subprocess.check_output(['git','show',release_commit+':'+release_path],cwd=repo),'RELEASE_CHANGED'
for value in (b['label'],b['source_sha256'],pins['NATIVE_BINDING.disabled.json'],pins['host_driver.py'],prior_commit):
 assert value in assignment.decode(),'RELEASE_PIN_MISMATCH'
files['SUPERVISOR_ASSIGNMENT.md']=assignment
cold_root='/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-cold-trn22-r2-packet'
archive=own/'research/v2/operator/20260926-trn22-confirmed36-clean-cold-five-r2/completed'
raws={}
for n in ('results-bundle.json','explicit-process-postflight.json'):
 p=archive/n;value=p.read_bytes()
 assert value==subprocess.check_output(['git','show',prior_commit+':'+str(p.relative_to(own))],cwd=own),'ACTUAL_COLD_R2_ARCHIVE_CHANGED'
 raws[n]=value
raw=json.loads(raws['results-bundle.json']);prior=json.loads(raws['explicit-process-postflight.json'])
assert raw['root']==cold_root,'WRONG_COLD_PREDECESSOR'
assert prior['all_observed_pids_absent'] and prior['neuron_device_idle'] and prior['related_rows']==[] and prior['busy_job_rows']==[] and prior['lock_competitor_rc']==0,'COLD_R2_NOT_QUIESCENT'
assert prior['fuser_rc']==1 and not prior['fuser_stdout'].strip()
assert (prior['lock_device'],prior['lock_inode'])==(66305,832589),'ORIGINAL_LOCK_CHANGED'
known=set(prior['observed_pids'])
assert {3276148,3276149,3276150,3276165,3276166,3276180,3276195,3276196,3276197,3276198}<=known,'MISSING_ACTUAL_R2_IDS'
assert all(not p['exists'] for p in prior['pid_rows'])
assert not any(any(row[k] in known for k in ('pid','ppid','pgid','sid')) for row in prior['complete_process_census']),'SURVIVING_R2_LINEAGE'
for n,r in raw['files'].items():
 value=r['text'].encode();assert len(value)==r['bytes'] and hashlib.sha256(value).hexdigest()==r['sha256'],'RAW_FILE_IDENTITY:'+n
def parsed(n):return json.loads(raw['files'][n]['text'])
terminal=parsed('terminal.json');outer=parsed('outer-flock-exit.json');rehearsal=parsed('train/rehearsal.json')
assert terminal['label']=='fs-20260926-confirmed36-clean-cold-trn22-r2' and terminal['status']=='CT_HARNESS_EXITED' and terminal['returncode']==0
assert outer['returncode']==0 and raw['files']['run-exit-code.txt']['text'].strip()=='0'
assert rehearsal['returncode']==0 and rehearsal['mode']=='train' and rehearsal['step_boundaries']==5
observed=rehearsal['observations'];assert len(observed)==5 and [x['step'] for x in observed]==list(range(5))
times=[x['observed_seconds'] for x in observed];assert all(math.isfinite(x) and x>=0 for x in times) and times==sorted(times)
fifth=times[4];assert fifth<=900,'ACTUAL_COLD_FIFTH_OVER_900'
assert rehearsal['charged_through_process_exit_within_budget'] is True and math.isfinite(rehearsal['wall_seconds'])
assert rehearsal['wall_seconds']-min(fifth,900)<=1800,'ACTUAL_COLD_PROCESS_BUDGET'
checkpoint=raw['checkpoint_after_terminal']
assert checkpoint['stable'] and checkpoint['sha256']==rehearsal['checkpoint_sha256'] and checkpoint['bytes']==rehearsal['checkpoint_bytes'],'COLD_CHECKPOINT_BINDING'
trainlaunch=parsed('training-launch.json');assert trainlaunch['source_sha256']==b['source_sha256'] and trainlaunch['prepare_sha256']==b['prepare_sha256'] and trainlaunch['harness_sha256']==b['harness_sha256']
live_path=own/'research/v2/operator/20260926-trn22-confirmed36-clean-cold-five-r2/trn22.json'
live_raw=live_path.read_bytes();assert hashlib.sha256(live_raw).hexdigest()=='949e360b82efab76721559b79c7b13aa3f8a93decb35c8765be865c8236ae214'
assert live_raw==subprocess.check_output(['git','show','65b377e179c3330e9542348fa35be30ce5a19ece:'+str(live_path.relative_to(own))],cwd=own)
live=json.loads(live_raw)['remote'];workers=[p for p in live['processes'] if p.get('environment_whitelist',{}).get('RANK') is not None]
assert len(workers)==4 and {p['environment_whitelist']['RANK'] for p in workers}=={'0','1','2','3'}
cache_dirs=dict(zip(('TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'),[cold_root+'/train/caches/'+v for v in ('neff','local','hlo','inductor','compiler','debug')]))
assert rehearsal['fresh_empty_sdk_caches']==cache_dirs,'ACTUAL_FRESH_CACHE_RECEIPT'
for w in workers:
 env=w['environment_whitelist']
 assert w['pid'] in known and w['identity_stable'] and env['WORLD_SIZE']=='4' and env['NEURON_LOGICAL_NC_CONFIG']=='2'
 assert env['RANK']==env['LOCAL_RANK'] and all(env[k]==v for k,v in cache_dirs.items())
 assert env['NEURON_COMPILE_CACHE_URL']==cache_dirs['NEURON_COMPILE_CACHE_DIR']
 assert w['argv'][2:]==b['training_command'][3:]+['--num-steps','5'],'ACTUAL_COLD_SCIENCE_ARGV'
# Four observed ranks plus successful torchrun/harness completion is the aggregate
# rank-exit proof; no unobserved individual wait-status timestamps are invented.
spec=importlib.util.spec_from_file_location('transport',repo/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'] and not any(state['breaker_tripped'].values()),state
assert state['host_launch_allowed']['trn22'] is True,'HOST_NOT_RELEASED'
hs=json.loads((repo/'research/v2/state.json').read_bytes())['hosts']['trn22']
assert hs['label']==b['label'] and hs['active_assignment']==release_path,'STATE_SCOPE_MISMATCH'
payload={'binding':b,'prior_root':cold_root,'prior_observed_pids':prior['observed_pids'],'prior_postflight_sha256':hashlib.sha256(raws['explicit-process-postflight.json']).hexdigest(),'prior_terminal_sha256':raw['files']['terminal.json']['sha256'],'prior_outer_sha256':raw['files']['outer-flock-exit.json']['sha256'],'prior_rehearsal_sha256':raw['files']['train/rehearsal.json']['sha256'],'prior_checkpoint_sha256':checkpoint['sha256'],'actual_cold_fifth_seconds':fifth,'files':{n:{'b64':base64.b64encode(x).decode(),'sha256':hashlib.sha256(x).hexdigest()} for n,x in files.items()},'local_state':state}

body=r"""
import base64,datetime,fcntl,hashlib,json,os,socket,subprocess
from pathlib import Path
b=P['binding'];root=Path(b['root']);prep=Path(b['prepare_path'])
assert socket.gethostname()=='ip-172-31-32-253' and os.getuid()==1000,'HOST_IDENTITY'
assert not os.path.lexists(root),'LABEL_ALREADY_CONSUMED'
for pid in (3050509,3053905):assert not Path('/proc/'+str(pid)).exists(),'OLD_QUEUE_PID_LIVE:'+str(pid)
previous=Path(P['prior_root'])
for name,key in [('terminal.json','prior_terminal_sha256'),('outer-flock-exit.json','prior_outer_sha256'),('train/rehearsal.json','prior_rehearsal_sha256')]:assert hashlib.sha256((previous/name).read_bytes()).hexdigest()==P[key],'PRIOR_COLD_RECEIPT_CHANGED:'+name
prevterm=json.loads((previous/'terminal.json').read_text());assert prevterm.get('status')=='CT_HARNESS_EXITED' and prevterm.get('returncode')==0,'PRIOR_DIAGNOSTIC_TERMINAL_MISMATCH'
prevouter=json.loads((previous/'outer-flock-exit.json').read_text());assert prevouter.get('returncode')==0,'PRIOR_DIAGNOSTIC_OUTER_NOT_EXITED'
for pid in P['prior_observed_pids']:assert not Path('/proc/'+str(pid)).exists(),'PRIOR_PRIOR_DIAGNOSTIC_PID_LIVE:'+str(pid)
lockfd=os.open('/home/ubuntu/.cache/trainium_freshstart/device0.lock',os.O_RDWR|os.O_NOFOLLOW)
lk=os.fstat(lockfd);assert (lk.st_dev,lk.st_ino)==(66305,832589),'LOCK_IDENTITY_CHANGED'
fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
for base in (Path('/home/ubuntu'),prep.parent,root,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
 for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):assert not os.path.lexists(base/rel),'STOP:'+str(base/rel)
v=os.statvfs('/');free=v.f_bavail*v.f_frsize;projected=3*1024**3
assert free>=10.5*1024**3+projected,'DISK_HEADROOM_PLUS_NEW_FOOTPRINT'
for p,k in [(prep,'prepare_sha256'),(Path(b['harness_path']),'harness_sha256')]:assert hashlib.sha256(p.read_bytes()).hexdigest()==b[k],k
pause=Path('/home/ubuntu/trainium_finalpush_queue_20260925_trn22_wait_r1/runs/PAUSE_NEXT');assert pause.is_file(),'OLD_QUEUE_PAUSE_MISSING'
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=10);assert ps.returncode==0,ps.stderr
prior_ids=set(P['prior_observed_pids'])
prior_lineage=[]
for line in ps.stdout.splitlines()[1:]:
 row=line.split(None,5)
 if len(row)==6 and any(int(x) in prior_ids for x in row[:4]):prior_lineage.append(line)
assert not prior_lineage,'PRIOR_PRIOR_DIAGNOSTIC_LINEAGE_REMAINS:'+str(prior_lineage)
jobs=[l for l in ps.stdout.splitlines() if any(k in l for k in ('torchrun','train.py','host_queue.py','host_driver.py','submission_rehearsal.py','neuronx-cc','walrus_driver','profile_actual_loop.py','preflight_dryrun.py','prepare.py eval-public','run_job.py','contained.py','rank_entry.py','check_native.py','compare_checkpoints.py'))];assert not jobs,'BUSY_PS:'+str(jobs)
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=20);assert ls.returncode==0,ls.stderr
devices=json.loads(ls.stdout);assert len(devices)==1 and devices[0]['neuron_processes']==[],'BUSY_DEVICE'
harness_root=Path(b['harness_path']).resolve().parents[2]
assert harness_root==prep.parent.resolve(),'HARNESS_PROJECT_MISMATCH'
cold_paths=[]
for phase,key in [('train','sandbox'),('eval','eval_sandbox')]:
 sandbox=Path(b[key])
 assert sandbox==root/phase and sandbox.resolve().is_relative_to(harness_root/'out'),'ACTUAL_HARNESS_CONTAINMENT:'+phase
 assert not os.path.lexists(sandbox),'PHASE_SANDBOX_CONSUMED:'+phase
 env=b['training_environment' if phase=='train' else 'evaluation_environment']
 expected=b['fresh_cache_contract']['expected_training_dirs' if phase=='train' else 'expected_eval_dirs']
 assert len(expected)==6 and env['NEURON_COMPILE_CACHE_URL']==str(sandbox/'caches/compiler'),'PRIVATE_CACHE_BINDING'
 for path in expected.values():
  assert Path(path).is_relative_to(sandbox/'caches') and not os.path.lexists(path),'CACHE_NOT_FRESH:'+path
  cold_paths.append(path)
assert len(set(cold_paths))==12,'TWELVE_PHASE_PRIVATE_CACHES'
with (previous/'train/out/final.pt').open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==P['prior_checkpoint_sha256'],'PRIOR_CHECKPOINT_CHANGED'
nodes=sorted(str(p) for p in Path('/dev').glob('neuron[0-9]*'));assert nodes
holders=subprocess.run(['/usr/bin/fuser',*nodes],capture_output=True,text=True,timeout=10);assert holders.returncode==1 and not holders.stdout.strip(),'DEVICE_HOLDERS'
assert Path('/home/ubuntu/workspace/native_venv/bin/torchrun').is_file()
root.mkdir(mode=0o700)
(root/'package').mkdir(mode=0o700)
(root/'metrics-training').mkdir(mode=0o700)
(root/'metrics-eval').mkdir(mode=0o700)
for name,rec in P['files'].items():
 dst=root/name;assert not dst.is_symlink() and not dst.exists()
 data=base64.b64decode(rec['b64']);assert hashlib.sha256(data).hexdigest()==rec['sha256']
 with dst.open('xb') as f:f.write(data)
 os.chmod(dst,0o444);assert hashlib.sha256(dst.read_bytes()).hexdigest()==rec['sha256']
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['label'],'host':'trn22','free_bytes':free,'projected_footprint_bytes':projected,'staging_checks_under_original_lock':True,'fuser_rc':holders.returncode,'fuser_stdout':holders.stdout,'fuser_stderr':holders.stderr,'prepare_sha256':b['prepare_sha256'],'harness_sha256':b['harness_sha256'],'staged_hashes':{n:r['sha256'] for n,r in P['files'].items()},'ps_rc':ps.returncode,'job_lines':jobs,'neuron_ls':devices,'pause_sha256':hashlib.sha256(pause.read_bytes()).hexdigest(),'source_sha256':b['source_sha256'],'local_state':P['local_state'],'actual_cold_fifth_seconds':P['actual_cold_fifth_seconds'],'actual_harness_root':str(harness_root),'twelve_new_cache_paths_absent':cold_paths,'prior_diagnostic_lineage_rows':prior_lineage,'previous_diagnostic_terminal':prevterm,'previous_diagnostic_outer':prevouter,'old_queue_pids_absent':[3050509,3053905],'lock_inode':lk.st_ino,'lock_device':lk.st_dev,'launch_guard':'Fresh actual common lifetime flock and in-lock ps/neuron-ls are enforced by exact driver; this staging check does not replace that guard.'}
(root/'preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
fcntl.flock(lockfd,fcntl.LOCK_UN);os.close(lockfd)
with (root/'dispatch.log').open('xb') as out:
 process=subprocess.Popen(['/usr/bin/python3','-I','-B',str(root/'dispatch_driver.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
launch={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['label'],'dispatch_pid':process.pid,'root':str(root),'status':'DETACHED_DISPATCH_STARTED_UNDER_LOCK_DRIVER_PENDING','no_automatic_next_job':True}
(root/'operator-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps({'preflight':receipt,'launch':launch}))
"""
argv,*_=m.load_transport('trn22');argv[-1]='exec python3 -I -B -'
wire='P='+repr(payload)+'\n'+body
assert not m.SENSITIVE.search(wire.encode()),'PAYLOAD_SENSITIVE_MARKER'
r=subprocess.run(argv,input=wire,text=True,capture_output=True,timeout=60)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode()),'OUTPUT_SENSITIVE_MARKER'
print(json.dumps({'host':'trn22','rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
