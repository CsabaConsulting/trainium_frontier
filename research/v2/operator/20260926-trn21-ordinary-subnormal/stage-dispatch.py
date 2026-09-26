import base64,hashlib,importlib.util,json,subprocess
from pathlib import Path
repo=Path('/home/poojith-devan/Desktop/Trainium');own=Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
d=repo/'research/v2/proposals/20260926-ordinary-embedding-subnormal-control'
bp=d/'NATIVE_BINDING.disabled.json';dp=d/'host_driver.py'
pins={'NATIVE_BINDING.disabled.json':'352a37f4b1ad96afe9c0148a29d7f16a58a677c8a1b151a1a416f303acf96a9d','host_driver.py':'7e24fc86bebcde3691b22f94fb6288c12fe009b2ea2fe59f2208af6cc760a463'}
for n,pin in pins.items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==pin,n
b=json.loads(bp.read_text())
ap='research/v2/verification/20260926-ordinary-subnormal-trn21-assignment.md';assignment=repo/ap
assert assignment.read_bytes()==subprocess.check_output(['git','show','f9206857bb636e939d6d85e748884304d33a3c07:'+ap],cwd=repo)
for pin in pins.values():assert pin in assignment.read_text()
archive=own/'research/v2/operator/20260926-trn21-startup-clock-replication/completed'
raw=json.loads((archive/'results-bundle.json').read_text())
assert 'terminal.json' in raw['files'] and 'outer-flock-exit.json' in raw['files'],'CONTROL_NOT_ARCHIVED'
assert raw['job_lines']==[] and raw['lock_competitor_rc']==0,'CONTROL_NOT_QUIESCENT'
for name in ('results-bundle.json','postflight-explicit-pids.json','post-eval-checkpoint-hash.json'):
 p=archive/name
 assert p.read_bytes()==subprocess.check_output(['git','show','cc1fa2259c0401a7611d8591e7fe7fe43c1f5ed6:'+str(p.relative_to(own))],cwd=own),'CONTROL_ARCHIVE_CHANGED:'+name
hashed=json.loads((archive/'post-eval-checkpoint-hash.json').read_text())
assert hashed['status']=='PASS' and hashed['unchanged_after_eval'] and hashed['sha256']=='19458cc8243de923d57b2bfb7a15ed4d775e78f419bef9ec28c518f411a42607'
post=json.loads((archive/'postflight-explicit-pids.json').read_text())
assert post['all_observed_pids_absent'] and not post['related_rows'] and post['neuron_device_idle'] and post['lock_competitor_rc']==0
oldpids=post['observed_pids'];assert set([3127536,3127537,3127538,3127554,3127555,3127569,3127584,3127585,3127586,3127587,3128094,3128096,3128141,3128697,3128700,3128703,3128704,3128706,3128707,3150111,3150112,3150126])==set(oldpids)

files={'NATIVE_BINDING.disabled.json':bp.read_bytes(),'host_driver.py':dp.read_bytes(),'EXPERIMENT_PROPOSAL.json':(d/'EXPERIMENT_PROPOSAL.json').read_bytes(),'README.md':(d/'README.md').read_bytes(),'SUPERVISOR_ASSIGNMENT.md':assignment.read_bytes()}
frozen='500da8458ca410a1c6f246999991742938717c37'
for name in b['support_sha256']:
 val=subprocess.check_output(['git','show',frozen+':'+str(d.relative_to(repo))+'/'+name],cwd=repo)
 assert hashlib.sha256(val).hexdigest()==b['support_sha256'][name];files[name]=val
for name,val in files.items():
 if name!='SUPERVISOR_ASSIGNMENT.md':
  assert val==subprocess.check_output(['git','show',frozen+':'+str(d.relative_to(repo))+'/'+name],cwd=repo),'FROZEN_PACKET_CHANGED:'+name
spec=importlib.util.spec_from_file_location('transport',repo/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'] and not any(state['breaker_tripped'].values()),state
assert state['host_launch_allowed']['trn21'] is True
hs=json.loads((repo/'research/v2/state.json').read_text())['hosts']['trn21']
assert hs['label']==b['label'] and hs['active_assignment']==ap,'STATE_SCOPE_MISMATCH'
payload={'binding':b,'files':{n:{'b64':base64.b64encode(val).decode(),'sha256':hashlib.sha256(val).hexdigest()} for n,val in files.items()},'local_state':state,'previous_pids':oldpids}
body=r"""
import base64,datetime,fcntl,hashlib,json,os,socket,subprocess
from pathlib import Path
b=P['binding'];root=Path(b['root']);project=root.parent.parent
assert socket.gethostname()==b['expected_hostname'] and os.getuid()==1000,'HOST_IDENTITY'
assert not os.path.lexists(root),'LABEL_ALREADY_CONSUMED'
previous=Path('/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-arch36a8-startup-clock-trn21-r2-full-packet')
prior_terminal=json.loads((previous/'terminal.json').read_text());prior_outer=json.loads((previous/'outer-flock-exit.json').read_text())
assert prior_terminal['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and prior_terminal['eval_exit_code']==0 and prior_outer['returncode']==0,'CONTROL_NOT_TERMINAL'
for pid in (*P['previous_pids'],2983496):
 assert not Path('/proc/'+str(pid)).exists(),'PRIOR_PID_LIVE:'+str(pid)
lk=Path('/home/ubuntu/.cache/trainium_freshstart/device0.lock').stat();assert lk.st_ino==596823 and lk.st_dev==66305,'LOCK_IDENTITY_CHANGED'
lockfd=os.open('/home/ubuntu/.cache/trainium_freshstart/device0.lock',os.O_RDWR|os.O_NOFOLLOW)
assert (os.fstat(lockfd).st_dev,os.fstat(lockfd).st_ino)==(66305,596823)
fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
for base in (Path('/home/ubuntu'),project,root,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
 for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):assert not os.path.lexists(base/rel),'STOP:'+str(base/rel)
v=os.statvfs('/');free=v.f_bavail*v.f_frsize;assert free>=10.5*1024**3,'DISK_HEADROOM'
mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
assert mem>=8*1024**3,'AVAILABLE_RAM_FLOOR'
pause=Path('/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1/runs/PAUSE_NEXT');assert pause.is_file(),'OLD_QUEUE_PAUSE_MISSING'
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=10);assert ps.returncode==0,ps.stderr
jobs=[l for l in ps.stdout.splitlines() if any(k in l for k in ('neuron-explorer','torchrun','train.py','host_queue.py','host_driver.py','submission_rehearsal.py','neuronx-cc','walrus_driver','profile_actual_loop.py','preflight_dryrun.py','prepare.py eval-public','run_job.py','contained.py','probe.py','rank_entry.py','check_native.py'))];assert not jobs,'BUSY_PS:'+str(jobs)
rows=[l.split(None,5) for l in ps.stdout.splitlines()[1:]]
assert not any(int(r[1]) in P['previous_pids'] or int(r[2]) in P['previous_pids'] or int(r[3]) in P['previous_pids'] for r in rows),'CONTROL_DESCENDANT_PGID_SID_LIVE'
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=20);assert ls.returncode==0,ls.stderr
devices=json.loads(ls.stdout);assert len(devices)==1 and devices[0]['neuron_processes']==[],'BUSY_DEVICE'
assert Path('/home/ubuntu/workspace/native_venv/bin/torchrun').is_file()
root.mkdir(mode=0o700)
for name,rec in P['files'].items():
 dst=root/name;assert not dst.is_symlink() and not dst.exists()
 val=base64.b64decode(rec['b64']);assert hashlib.sha256(val).hexdigest()==rec['sha256']
 with dst.open('xb') as f:f.write(val)
 os.chmod(dst,0o444);assert hashlib.sha256(dst.read_bytes()).hexdigest()==rec['sha256']
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['label'],'host':'trn21','free_bytes':free,'available_ram_bytes':mem,'staged_hashes':{n:r['sha256'] for n,r in P['files'].items()},'ps_rc':ps.returncode,'job_lines':jobs,'neuron_ls':devices,'pause_sha256':hashlib.sha256(pause.read_bytes()).hexdigest(),'previous_pids_verified_absent':P['previous_pids'],'previous_pgid_sid_descendants_absent':True,'local_state':P['local_state'],'lock_inode':lk.st_ino,'lock_device':lk.st_dev,'launch_guard':'Same original lifetime flock, fresh in-lock ps/neuron-ls enforced by exact pinned driver after preliminary staging check.','scope':'One600+60s ordinary-only4rank diagnostic; no K2/train.py/timing/full/eval/autonext'}
with (root/'preflight.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
fcntl.flock(lockfd,fcntl.LOCK_UN);os.close(lockfd)
with (root/'dispatch.log').open('xb') as out:
 process=subprocess.Popen(b['dispatch_command'],cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
launch={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['label'],'dispatch_pid':process.pid,'root':str(root),'status':'DETACHED_DISPATCH_STARTED_UNDER_LOCK_DRIVER_PENDING','no_automatic_next_job':True}
with (root/'operator-launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n')
print(json.dumps({'preflight':receipt,'launch':launch}))
"""
argv,*_=m.load_transport('trn21');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input='P='+repr(payload)+'\n'+body,text=True,capture_output=True,timeout=45)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode())
print(json.dumps({'host':'trn21','rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))

