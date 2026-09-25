import base64,hashlib,importlib.util,json,subprocess
from pathlib import Path
repo=Path('/home/poojith-devan/Desktop/Trainium');own=Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
d=repo/'research/v2/proposals/20260926-p-offline-decoder-trn21'
bp=d/'OFFLINE_BINDING.disabled.json';manifest=d/'MANIFEST.sha256.json'
assert hashlib.sha256(bp.read_bytes()).hexdigest()=='f5ddd27cfa7e43ee6238ea91f5f77b8495d5d148db00db4e3aa02235f618b299'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='c12bf98ba7baebe28bab72f09041e4e640d871f27f01fe941c01e65f8ddb2219'
for n,h in json.loads(manifest.read_text())['files'].items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==h,n
b=json.loads(bp.read_text())
assignment=repo/'research/v2/verification/20260926-p-offline-decoder-trn21-assignment.md'
assert assignment.read_bytes()==subprocess.check_output(['git','show','27768550:research/v2/verification/20260926-p-offline-decoder-trn21-assignment.md'],cwd=repo)
for x in (b['label'],b['runtime_sha256']['host_driver.py'],b['runtime_sha256']['decoder_support.py']):assert x in assignment.read_text()
files={n:(d/n).read_bytes() for n in ('host_driver.py','decoder_support.py','EXPERIMENT_PROPOSAL.json','OFFLINE_BINDING.disabled.json','MANIFEST.sha256.json')}
g=own/'research/v2/operator/20260926-common-host-idle-guard'
for n in ('idle_guard.py','dispatch_driver.py'):files[n]=(g/n).read_bytes()
for n,h in b['runtime_sha256'].items():assert hashlib.sha256(files[n]).hexdigest()==h,n
files['SUPERVISOR_ASSIGNMENT.md']=assignment.read_bytes()
postpath=own/'research/v2/operator/20260926-trn21-architecture-recursive-full/completed/explicit-process-postflight.json'
post=json.loads(postpath.read_text())
assert post['all_observed_pids_absent'] and not post['related_rows'] and post['neuron_device_idle'] and post['lock_competitor_rc']==0
spec=importlib.util.spec_from_file_location('transport',repo/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'] and not any(state['breaker_tripped'].values()),state
assert state['host_launch_allowed']['trn21'] is True
P={'binding':b,'files':{n:{'b64':base64.b64encode(x).decode(),'sha256':hashlib.sha256(x).hexdigest()} for n,x in files.items()},'local_state':state,'prior_postflight':post}
body=r"""
import base64,datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
b=P['binding'];root=Path(b['root']);project=Path('/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2')
assert socket.gethostname()==b['hostname']=='ip-172-31-34-85' and os.getuid()==1000,'HOST_IDENTITY'
assert not os.path.lexists(root),'LABEL_ALREADY_CONSUMED'
previous=project/'out/fs-20260926-arch75b9-recursive-trn21-r1-full-packet'
prevterm=json.loads((previous/'terminal.json').read_text())
prevouter=json.loads((previous/'outer-flock-exit.json').read_text())
assert prevterm.get('status')=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and prevterm.get('eval_exit_code')==0,'PRIOR_FULL_NOT_SUCCESSFULLY_TERMINAL'
assert prevouter.get('returncode')==0,'PRIOR_FULL_DRIVER_NOT_EXITED'
prior=set(P['prior_postflight']['observed_pids'])
for pid in prior|{2983496}:assert not Path('/proc/'+str(pid)).exists(),'PRIOR_PID_LIVE:'+str(pid)
lk=Path('/home/ubuntu/.cache/trainium_freshstart/device0.lock').stat();assert (lk.st_dev,lk.st_ino)==(66305,596823),'LOCK_IDENTITY'
for base in (Path('/home/ubuntu'),project,root,Path(b['source_root']).parents[2],*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
 for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):assert not os.path.lexists(base/rel),'STOP:'+str(base/rel)
v=os.statvfs('/');free=v.f_bavail*v.f_frsize;assert free>=12*1024**3,'DISK_BELOW_12_GIB'
mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());available=int(mem['MemAvailable'].split()[0])*1024;assert available>=24*1024**3,'RAM_BELOW_24_GIB'
pause=Path('/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1/runs/PAUSE_NEXT');assert pause.is_file(),'OLD_QUEUE_PAUSE_MISSING'
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=10);assert ps.returncode==0 and not ps.stderr,'PS_FAILED:'+ps.stderr
related=[]
for line in ps.stdout.splitlines()[1:]:
 row=line.split(None,5);assert len(row)==6
 if any(int(x) in prior for x in row[:4]) or any(k in line for k in ('torchrun','train.py','host_queue.py','host_driver.py','submission_rehearsal.py','neuronx-cc','walrus_driver','profile_actual_loop.py','preflight_dryrun.py','prepare.py eval-public','run_job.py','contained.py')):related.append(line)
assert not related,'BUSY_PS:'+str(related)
nls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=15);assert nls.returncode==0 and not nls.stderr,'NLS_FAILED:'+nls.stderr
devices=json.loads(nls.stdout);assert len(devices)==1 and devices[0]['neuron_processes']==[],'BUSY_DEVICE'
assert Path(b['source_root']).is_dir() and not Path(b['source_root']).is_symlink(),'PROFILE_ROOT_MISSING_OR_SYMLINK'
root.mkdir(mode=0o700)
for name,r in P['files'].items():
 dst=root/name;assert not os.path.lexists(dst)
 raw=base64.b64decode(r['b64']);assert hashlib.sha256(raw).hexdigest()==r['sha256']
 with dst.open('xb') as f:f.write(raw)
 os.chmod(dst,0o600);assert hashlib.sha256(dst.read_bytes()).hexdigest()==r['sha256']
preflight={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['label'],'host':'trn21','free_bytes':free,'available_memory_bytes':available,'staged_hashes':{n:r['sha256'] for n,r in P['files'].items()},'prior_full_terminal':prevterm,'prior_full_outer':prevouter,'prior_postflight':P['prior_postflight'],'local_state':P['local_state'],'ps_rc':ps.returncode,'related_rows':related,'neuron_ls':devices,'old_queue_pids_absent':[2983496],'pause_sha256':hashlib.sha256(pause.read_bytes()).hexdigest(),'lock_device':lk.st_dev,'lock_inode':lk.st_ino,'profile_inputs_not_read_or_copied_during_staging':True,'same_guard_in_lock_checks_required':True}
(root/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
with (root/'dispatch.log').open('xb') as log:
 p=subprocess.Popen(b['dispatch_command'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
launch={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['label'],'dispatch_pid':p.pid,'root':str(root),'status':'DETACHED_DISPATCH_STARTED_UNDER_LOCK_DRIVER_PENDING','no_automatic_retry':True,'no_automatic_next_job':True}
(root/'operator-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps({'preflight':preflight,'launch':launch}))
"""
argv,*_=m.load_transport('trn21');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input='P='+repr(P)+'\n'+body,text=True,capture_output=True,timeout=45)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode())
print(json.dumps({'host':'trn21','rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
