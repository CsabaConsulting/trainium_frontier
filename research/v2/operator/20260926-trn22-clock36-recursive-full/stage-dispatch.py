import base64,hashlib,importlib.util,json,re,shlex,subprocess,sys
from pathlib import Path
repo=Path('/home/poojith-devan/Desktop/Trainium');own=Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
d=repo/'research/v2/proposals/20260926-clock36-recursive-full-trn22'
bp=d/'NATIVE_BINDING.disabled.json';qp=d/'EXPERIMENT_PROPOSAL.json';dp=d/'host_driver.py'
pins={'binding':'f89c00a2adeaf435cf055f25470246ca66bf11bd5aa28ef4c33f4af528be3d06','proposal':'3c279fb7c558760fb0aab80761f6d11a9cca6fd9418c6f7665f40587c584e617','driver':'0c2230da6aeee5aa55c5c19aeb0a0925e3a815d61554c0cc2781b7def10ed12a'}
for k,p in [('binding',bp),('proposal',qp),('driver',dp)]:assert hashlib.sha256(p.read_bytes()).hexdigest()==pins[k]
manifest_raw=(d/'MANIFEST.sha256.json').read_bytes();assert hashlib.sha256(manifest_raw).hexdigest()=='6d0a3aafc39ed02563c70b3c94f92a63a821d87ac912a2463fd442e249d323ce'
for n,h in json.loads(manifest_raw)['files'].items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==h

frozen='0090fd37ed605461f9e15414cf8f3bfc4d43ad66'
for p in (bp,qp,dp,d/'MANIFEST.sha256.json',d/'package/launch-command.txt',d/'package/provenance.json'):
 assert p.read_bytes()==subprocess.check_output(['git','show',frozen+':'+str(p.relative_to(repo))],cwd=repo),'FROZEN_PACKET_CHANGED'
prior_local=own/'research/v2/operator/20260926-trn22-clock36-recursive-cold-five/completed/explicit-process-postflight.json'
assert len(sys.argv)==4 and re.fullmatch(r'[0-9a-f]{40}',sys.argv[1]) and re.fullmatch(r'[0-9a-f]{40}',sys.argv[2]),'REQUIRES_COMMITTED_FINAL_POSTFLIGHT_AND_SCOPED_RELEASE'
assert sys.argv[3].startswith('research/v2/verification/') and '..' not in Path(sys.argv[3]).parts,'RELEASE_PATH_SCOPE'
prior_commit=sys.argv[1]
prior_raw=prior_local.read_bytes()
assert prior_raw==subprocess.check_output(['git','show',prior_commit+':'+str(prior_local.relative_to(own))],cwd=own),'PRIOR_DIAGNOSTIC_POSTFLIGHT_CHANGED'
prior=json.loads(prior_raw)
assert prior['all_observed_pids_absent'] and prior['neuron_device_idle'] and prior['related_rows']==[] and prior['lock_competitor_rc']==0,'PRIOR_DIAGNOSTIC_POSTFLIGHT_NOT_QUIESCENT'
assert prior['lock_inode']==832589 and prior['lock_device']==66305,'PRIOR_DIAGNOSTIC_LOCK_IDENTITY'

assert set(prior['observed_pids'])=={3246477,3246478,3246479,3246494,3246495,3246509,3246524,3246525,3246526,3246527,3247539,3247547,3247725,3250697,3250733,3250738,3250746,3250812,3250813,3250814,3250815,3250816,3250817,3250818,3250819,3250820,3250821,3250822,3250825},'MISSING_PRIOR_IDENTITIES'
assert hashlib.sha256(prior_raw).hexdigest()=='3790753cc33d3b08428dfa719adb920008d4868171c8ba19795cbefa2a0a6bcb','PRIOR_POSTFLIGHT_SHA'

b=json.loads(bp.read_text())
assignment=repo/sys.argv[3]
assert assignment.read_bytes()==subprocess.check_output(['git','show',sys.argv[2]+':'+sys.argv[3]],cwd=repo)
for x in [b['label'],b['source_sha256'],pins['driver'],pins['binding']]:assert x in assignment.read_text()
source=subprocess.check_output(['git','show',b['source_commit']+':'+b['source_git_path']],cwd=own);assert hashlib.sha256(source).hexdigest()==b['source_sha256']
launch=shlex.join(b['training_command'])
files={'package/train.py':source,'package/launch-command.txt':(d/'package/launch-command.txt').read_bytes(),'package/provenance.json':(d/'package/provenance.json').read_bytes(),'EXPERIMENT_PROPOSAL.json':qp.read_bytes(),'NATIVE_BINDING.disabled.json':bp.read_bytes(),'host_driver.py':dp.read_bytes(),'SUPERVISOR_ASSIGNMENT.md':assignment.read_bytes(),'MANIFEST.sha256.json':manifest_raw}
for n,h in b['package_support_sha256'].items():assert hashlib.sha256(files['package/'+n]).hexdigest()==h
assert files['package/launch-command.txt'].decode().strip()==launch
g=own/'research/v2/operator/20260926-common-host-idle-guard'
for n,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
 raw=(g/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==b[key];files[n]=raw
spec=importlib.util.spec_from_file_location('transport',repo/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'] and not any(state['breaker_tripped'].values()),state
assert state['host_launch_allowed']['trn22'] is True
full_state=json.loads((repo/'research/v2/state.json').read_text())
host_state=full_state['hosts']['trn22']
assert host_state['label']==b['label'],'STATE_SCOPE_LABEL_MISMATCH'
assert assignment.name in host_state['active_assignment'],'STATE_ASSIGNMENT_MISMATCH'
payload={'binding':b,'prior_observed_pids':prior['observed_pids'],'prior_postflight_sha256':hashlib.sha256(prior_raw).hexdigest(),'files':{n:{'b64':base64.b64encode(x).decode(),'sha256':hashlib.sha256(x).hexdigest()} for n,x in files.items()},'local_state':state}
body=r"""
import base64,datetime,fcntl,hashlib,json,os,socket,subprocess
from pathlib import Path
b=P['binding'];root=Path(b['root']);prep=Path(b['prepare_path'])
assert socket.gethostname()=='ip-172-31-32-253' and os.getuid()==1000,'HOST_IDENTITY'
assert not os.path.lexists(root),'LABEL_ALREADY_CONSUMED'
for pid in (3050509,3053905):assert not Path('/proc/'+str(pid)).exists(),'OLD_QUEUE_PID_LIVE:'+str(pid)
previous=Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-clock36-recursive-cold-trn22-r1-packet')
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
for env in (b['training_environment'],b['evaluation_environment']):
 for key in ('TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'):assert Path(env[key]).is_dir(),'CACHE_MISSING:'+key
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
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['label'],'host':'trn22','free_bytes':free,'projected_footprint_bytes':projected,'staging_checks_under_original_lock':True,'fuser_rc':holders.returncode,'fuser_stdout':holders.stdout,'fuser_stderr':holders.stderr,'prepare_sha256':b['prepare_sha256'],'harness_sha256':b['harness_sha256'],'staged_hashes':{n:r['sha256'] for n,r in P['files'].items()},'ps_rc':ps.returncode,'job_lines':jobs,'neuron_ls':devices,'pause_sha256':hashlib.sha256(pause.read_bytes()).hexdigest(),'source_sha256':b['source_sha256'],'local_state':P['local_state'],'prior_diagnostic_lineage_rows':prior_lineage,'previous_diagnostic_terminal':prevterm,'previous_diagnostic_outer':prevouter,'old_queue_pids_absent':[3050509,3053905],'lock_inode':lk.st_ino,'lock_device':lk.st_dev,'launch_guard':'Fresh actual common lifetime flock and in-lock ps/neuron-ls are enforced by exact driver; this staging check does not replace that guard.'}
(root/'preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
fcntl.flock(lockfd,fcntl.LOCK_UN);os.close(lockfd)
with (root/'dispatch.log').open('xb') as out:
 process=subprocess.Popen(['/usr/bin/python3','-I','-B',str(root/'dispatch_driver.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
launch={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['label'],'dispatch_pid':process.pid,'root':str(root),'status':'DETACHED_DISPATCH_STARTED_UNDER_LOCK_DRIVER_PENDING','no_automatic_next_job':True}
(root/'operator-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps({'preflight':receipt,'launch':launch}))
"""
argv,*_=m.load_transport('trn22');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input='P='+repr(payload)+'\n'+body,text=True,capture_output=True,timeout=45)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode())
print(json.dumps({'host':'trn22','rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
