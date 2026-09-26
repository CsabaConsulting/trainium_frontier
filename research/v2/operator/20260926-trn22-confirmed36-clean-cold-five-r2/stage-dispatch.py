"""Prepared one-use clean cold-five transport; no authority without exact release."""
import base64,hashlib,importlib.util,json,re,shlex,subprocess,sys
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('NORMAL_PYTHON_REQUIRED')
repo=Path('/home/poojith-devan/Desktop/Trainium');own=Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
rel='research/v2/proposals/20260926-confirmed36-clean-cold-five-trn22-r2';d=own/rel
frozen='c1384be93c2d314b49dd574f4b936a732f0b1a26'
pins={"CT_BINDING.disabled.json":"932d545020b822acd00deb3b5c93b68a421187a6ec424e6853428180ac541261","EXPERIMENT_PROPOSAL.json":"2a0c3c46430515a04bbd238743b81a17669646fd9f5f3b649ef7e34288f00d06","host_driver.py":"3d378afe134eeb46e813b013aaaf55d42743b25f6175e80cf3087975aecc269d","MANIFEST.sha256.json":"0ac465c5ea16ebe5aac0f79225db098a4201d6cc0fb3118ca9e650767ad48884"}
for n,h in pins.items():
 raw=(d/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==h,n
 assert raw==subprocess.check_output(['git','show',frozen+':'+rel+'/'+n],cwd=own),n
manifest_raw=(d/'MANIFEST.sha256.json').read_bytes()
for n,h in json.loads(manifest_raw)['files'].items():
 raw=(d/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==h,n
 assert raw==subprocess.check_output(['git','show',frozen+':'+rel+'/'+n],cwd=own),n
b=json.loads((d/'CT_BINDING.disabled.json').read_text())
assert b['host']=='trn22' and b['root']=="/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-cold-trn22-r2-packet"
assert b['training_command']==b['base_training_command']+['--num-steps','5']
source=subprocess.check_output(['git','show',b['source_commit']+':'+b['source_git_path']],cwd=own)
assert hashlib.sha256(source).hexdigest()==b['source_sha256']
files={n:(d/n).read_bytes() for n in ('CT_BINDING.disabled.json','EXPERIMENT_PROPOSAL.json','host_driver.py','MANIFEST.sha256.json','package/launch-command.txt','package/provenance.json')}
files['package/train.py']=source
assert files['package/launch-command.txt'].decode().strip()==shlex.join(b['training_command'])
for n,h in b['package_files'].items():assert hashlib.sha256(files[n]).hexdigest()==h
g=own/'research/v2/operator/20260926-common-host-idle-guard'
for n,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
 raw=(g/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==b[key];files[n]=raw
if sys.argv[1:]==['--local-check']:
 print(json.dumps({'status':'LOCAL_EXACT_COLD_PACKET_READY_ONLY','files':len(files),'host_calls':False,'launch_authorized':False,'floor_plus_footprint_bytes':int(13.5*1024**3)}));sys.exit(0)
assert len(sys.argv)==4,'EXACT_RELEASE_COMMIT_PATH_AND_PREDECESSOR_COMMIT_REQUIRED'
release_commit,release_path,prior_commit=sys.argv[1:]
assert all(re.fullmatch(r'[0-9a-f]{40}',v) for v in (release_commit,prior_commit))
assert release_path.startswith('research/v2/verification/') and release_path.endswith('.md') and '..' not in Path(release_path).parts
assignment=(repo/release_path).read_bytes()
assert assignment==subprocess.check_output(['git','show',release_commit+':'+release_path],cwd=repo)
for x in [b['label'],b['source_sha256'],pins['host_driver.py'],pins['CT_BINDING.disabled.json']]:assert x in assignment.decode(),'RELEASE_PIN_MISMATCH'
files['SUPERVISOR_ASSIGNMENT.md']=assignment
archive=own/'research/v2/operator/20260926-trn21-confirmed36-clean-checkpoint/completed'
for name in ('results-bundle.json','explicit-process-postflight.json'):
 p=archive/name;assert p.read_bytes()==subprocess.check_output(['git','show',prior_commit+':'+str(p.relative_to(own))],cwd=own),'PREDECESSOR_RAW_CHANGED'
prior=json.loads((archive/'explicit-process-postflight.json').read_bytes());raw=json.loads((archive/'results-bundle.json').read_bytes())
assert prior['all_observed_pids_absent'] and prior['neuron_device_idle'] and prior['related_rows']==[] and prior['lock_competitor_rc']==0
assert (prior['lock_device'],prior['lock_inode'])==(66305,596823)
assert {3177118,3177119,3177120,3177150,3177151}<=set(prior['observed_pids'])
assert not any(any(row[key] in prior['observed_pids'] for key in ('pid','ppid','pgid','sid')) for row in prior['complete_process_census'])
terminal_raw=raw['files']['terminal.json']['text'].encode();outer_raw=raw['files']['outer-flock-exit.json']['text'].encode()
assert json.loads(terminal_raw)['status']=='CPU_CHECKPOINT_ONLY_COMPLETE_PENDING_INDEPENDENT_REVIEW' and json.loads(terminal_raw)['exit_code']==0
assert json.loads(outer_raw)['returncode']==0
assert json.loads(raw['files']['checkpoint-result.json']['text'])['status']=='CPU_CHECKPOINT_SYNTHETIC_QUALIFICATION_COMPLETE'
# The host predecessor is the completed trn22 full run; helper proof is cross-host.
helper_prior=prior
full_archive=own/'research/v2/operator/20260926-trn22-clock36-recursive-full/completed'
full_commit='1491c1c2a18a2d86da8783cf632fc2290bfeafb1'
for name in ('results-bundle.json','explicit-process-postflight.json'):
 p=full_archive/name
 assert p.read_bytes()==subprocess.check_output(['git','show',full_commit+':'+str(p.relative_to(own))],cwd=own),'FULL_PREDECESSOR_CHANGED'
prior=json.loads((full_archive/'explicit-process-postflight.json').read_bytes())
fullraw=json.loads((full_archive/'results-bundle.json').read_bytes())
assert prior['all_observed_pids_absent'] and prior['related_rows']==[] and prior['neuron_device_idle'] and prior['lock_competitor_rc']==0
assert (prior['lock_device'],prior['lock_inode'])==(66305,832589)
assert len(prior['observed_pids'])==19 and len(prior['known_prior_identities'])==19
assert all(not row['exists'] for row in prior['pid_rows'])
terminal_raw=fullraw['files']['terminal.json']['text'].encode();outer_raw=fullraw['files']['outer-flock-exit.json']['text'].encode()
assert json.loads(terminal_raw)['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and json.loads(terminal_raw)['eval_exit_code']==0
assert json.loads(outer_raw)['returncode']==0
assert fullraw['checkpoint_after_terminal']['stable'] and fullraw['checkpoint_after_terminal']['sha256']=='1e1bf15aa6c2b95578f2d6317b8a400821cb4f66bea13312319e385236902b9e'

failed_archive=own/'research/v2/operator/20260926-trn22-confirmed36-clean-cold-five/completed'
failed_commit='56b5220beb673f956c7ad8b270f018050505cc8c'
for n in ('results-bundle.json','explicit-process-postflight.json'):
 p=failed_archive/n
 assert p.read_bytes()==subprocess.check_output(['git','show',failed_commit+':'+str(p.relative_to(own))],cwd=own),'FAILED_R1_RAW_CHANGED'
failed_post=json.loads((failed_archive/'explicit-process-postflight.json').read_bytes())
failed_raw=json.loads((failed_archive/'results-bundle.json').read_bytes())
assert failed_post['all_observed_pids_absent'] and failed_post['related_rows']==[] and failed_post['busy_job_rows']==[] and failed_post['neuron_device_idle'] and failed_post['lock_competitor_rc']==0
assert failed_post['observed_pids']==[3275899,3275900,3275901] and (failed_post['lock_device'],failed_post['lock_inode'])==(66305,832589)
assert not any(any(row[key] in failed_post['observed_pids'] for key in ('pid','ppid','pgid','sid')) for row in failed_post['complete_process_census'])
failed_term=failed_raw['files']['terminal.json']['text'].encode()
failed_outer=failed_raw['files']['outer-flock-exit.json']['text'].encode()
assert json.loads(failed_term)['status']=='CT_HARNESS_FAILED' and json.loads(failed_term)['returncode']==1
assert json.loads(failed_outer)['returncode']==1

spec=importlib.util.spec_from_file_location('transport',repo/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'] and not any(state['breaker_tripped'].values()),state
assert state['host_launch_allowed']['trn22'] is True
host_state=json.loads((repo/'research/v2/state.json').read_text())['hosts']['trn22']
assert host_state['label']==b['label'] and host_state['active_assignment']==release_path,'STATE_SCOPE_MISMATCH'
payload={'binding':b,'prior_observed_pids':prior['observed_pids'],'prior_postflight_sha256':hashlib.sha256((full_archive/'explicit-process-postflight.json').read_bytes()).hexdigest(),'helper_postflight_sha256':hashlib.sha256((archive/'explicit-process-postflight.json').read_bytes()).hexdigest(),'previous_terminal_sha256':hashlib.sha256(terminal_raw).hexdigest(),'previous_outer_sha256':hashlib.sha256(outer_raw).hexdigest(),'files':{n:{'b64':base64.b64encode(x).decode(),'sha256':hashlib.sha256(x).hexdigest()} for n,x in files.items()},'local_state':state,'failed_r1_pids':failed_post['observed_pids'],'failed_r1_terminal_sha256':hashlib.sha256(failed_term).hexdigest(),'failed_r1_outer_sha256':hashlib.sha256(failed_outer).hexdigest()}

body=r"""
import base64,datetime,fcntl,hashlib,json,os,socket,subprocess
from pathlib import Path
b=P['binding'];root=Path(b['root']);prep=Path(b['prepare_path'])
assert socket.gethostname()=='ip-172-31-32-253' and os.getuid()==1000,'HOST_IDENTITY'
assert not os.path.lexists(root),'LABEL_ALREADY_CONSUMED'

harness_root=Path(b['harness_path']).resolve().parents[2]
resolved_sandbox=Path(b['sandbox']).resolve()
assert b['sandbox']==str(root/'train') and resolved_sandbox.is_relative_to(harness_root/'out'),'ACTUAL_HARNESS_CONTAINMENT'
assert harness_root==prep.parent.resolve(),'HARNESS_PROJECT_MISMATCH'
failed=Path('/home/ubuntu/trainium_confirmed36_clean_cold_20260926_trn22_r1')
assert hashlib.sha256((failed/'terminal.json').read_bytes()).hexdigest()==P['failed_r1_terminal_sha256'],'FAILED_R1_TERMINAL_CHANGED'
assert hashlib.sha256((failed/'outer-flock-exit.json').read_bytes()).hexdigest()==P['failed_r1_outer_sha256'],'FAILED_R1_OUTER_CHANGED'
for pid in P['failed_r1_pids']:assert not Path('/proc/'+str(pid)).exists(),'FAILED_R1_PID_LIVE:'+str(pid)

for pid in (3050509,3053905):assert not Path('/proc/'+str(pid)).exists(),'OLD_QUEUE_PID_LIVE:'+str(pid)
previous=Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clock36-recursive-trn22-r1-full-packet')
assert hashlib.sha256((previous/'terminal.json').read_bytes()).hexdigest()==P['previous_terminal_sha256'],'PRIOR_TERMINAL_CHANGED'
assert hashlib.sha256((previous/'outer-flock-exit.json').read_bytes()).hexdigest()==P['previous_outer_sha256'],'PRIOR_OUTER_CHANGED'
prevterm=json.loads((previous/'terminal.json').read_text());assert prevterm.get('status')=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and prevterm.get('eval_exit_code')==0,'PRIOR_DIAGNOSTIC_TERMINAL_MISMATCH'
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
prior_ids=set(P['prior_observed_pids'])|set(P['failed_r1_pids'])
prior_lineage=[]
for line in ps.stdout.splitlines()[1:]:
 row=line.split(None,5)
 if len(row)==6 and any(int(x) in prior_ids for x in row[:4]):prior_lineage.append(line)
assert not prior_lineage,'PRIOR_PRIOR_DIAGNOSTIC_LINEAGE_REMAINS:'+str(prior_lineage)
jobs=[l for l in ps.stdout.splitlines() if any(k in l for k in ('torchrun','train.py','host_queue.py','host_driver.py','submission_rehearsal.py','neuronx-cc','walrus_driver','profile_actual_loop.py','preflight_dryrun.py','prepare.py eval-public','run_job.py','contained.py','rank_entry.py','check_native.py','compare_checkpoints.py'))];assert not jobs,'BUSY_PS:'+str(jobs)
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=20);assert ls.returncode==0,ls.stderr
devices=json.loads(ls.stdout);assert len(devices)==1 and devices[0]['neuron_processes']==[],'BUSY_DEVICE'
cache_keys=('TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR')
cache_paths=[b['harness_injected_environment'][k] for k in cache_keys]
assert len(set(cache_paths))==6
assert b['training_environment']['NEURON_COMPILE_CACHE_URL']==b['harness_injected_environment']['NEURON_COMPILE_CACHE_DIR']
for path in cache_paths:assert Path(path).is_relative_to(root) and not os.path.lexists(path),'CACHE_NOT_FRESH:'+path
nodes=sorted(str(p) for p in Path('/dev').glob('neuron[0-9]*'));assert nodes
holders=subprocess.run(['/usr/bin/fuser',*nodes],capture_output=True,text=True,timeout=10);assert holders.returncode==1 and not holders.stdout.strip(),'DEVICE_HOLDERS'
assert Path('/home/ubuntu/workspace/native_venv/bin/torchrun').is_file()
root.mkdir(mode=0o700)
(root/'package').mkdir(mode=0o700)
(root/'metrics-training').mkdir(mode=0o700)
for name,rec in P['files'].items():
 dst=root/name;assert not dst.is_symlink() and not dst.exists()
 data=base64.b64decode(rec['b64']);assert hashlib.sha256(data).hexdigest()==rec['sha256']
 with dst.open('xb') as f:f.write(data)
 os.chmod(dst,0o444);assert hashlib.sha256(dst.read_bytes()).hexdigest()==rec['sha256']
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['label'],'host':'trn22','free_bytes':free,'six_caches_absent_before_harness':cache_paths,'projected_footprint_bytes':projected,'staging_checks_under_original_lock':True,'fuser_rc':holders.returncode,'fuser_stdout':holders.stdout,'fuser_stderr':holders.stderr,'prepare_sha256':b['prepare_sha256'],'harness_sha256':b['harness_sha256'],'staged_hashes':{n:r['sha256'] for n,r in P['files'].items()},'ps_rc':ps.returncode,'job_lines':jobs,'neuron_ls':devices,'pause_sha256':hashlib.sha256(pause.read_bytes()).hexdigest(),'source_sha256':b['source_sha256'],'local_state':P['local_state'],'actual_harness_containment':{'harness_resolved_root':str(harness_root),'resolved_sandbox':str(resolved_sandbox),'pass':True},'failed_r1_pids_verified_absent':P['failed_r1_pids'],'failed_r1_terminal_sha256':P['failed_r1_terminal_sha256'],'prior_diagnostic_lineage_rows':prior_lineage,'previous_diagnostic_terminal':prevterm,'previous_diagnostic_outer':prevouter,'old_queue_pids_absent':[3050509,3053905],'lock_inode':lk.st_ino,'lock_device':lk.st_dev,'launch_guard':'Fresh actual common lifetime flock and in-lock ps/neuron-ls are enforced by exact driver; this staging check does not replace that guard.'}
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
r=subprocess.run(argv,input=wire,text=True,capture_output=True,timeout=45)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode()),'OUTPUT_SENSITIVE_MARKER'
print(json.dumps({'host':'trn22','rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
