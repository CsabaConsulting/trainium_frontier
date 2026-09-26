import base64,hashlib,importlib.util,json,subprocess
from pathlib import Path
repo=Path('/home/poojith-devan/Desktop/Trainium');own=Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
d=repo/'research/v2/proposals/20260926-k2-registered-native-trn22'
bp=d/'NATIVE_BINDING.disabled.json';dp=d/'host_driver.py'
pins={'NATIVE_BINDING.disabled.json':'af1b84ee39ea516eee1189c72ed6e871638bff7c5fc7f398cce2cacf1bc1b8a9','host_driver.py':'09e5f9053d3c326bb6b8bf00b3959276317a783679e5e6d10c92d750e16404a4','MANIFEST.sha256.json':'fcb2c40adbe5a33a93b4ded6edb544afdff9a42c1f17f86a84d7519a346fd6d2'}
for n,pin in pins.items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==pin,n
b=json.loads(bp.read_text());manifest=json.loads((d/'MANIFEST.sha256.json').read_text())
ap='research/v2/verification/20260926-k2-registered-trn22-assignment.md';assignment=repo/ap
assert assignment.read_bytes()==subprocess.check_output(['git','show','ddbe5dcc609a8fd985d235b74269069b845c7326:'+ap],cwd=repo)
for pin in pins.values():assert pin in assignment.read_text()
archive=own/'research/v2/operator/20260926-trn22-recursive6ef7-control/completed'
raw=json.loads((archive/'results-bundle.json').read_text())
assert 'terminal.json' in raw['files'] and 'outer-flock-exit.json' in raw['files'],'CONTROL_NOT_ARCHIVED'
assert raw['job_lines']==[] and raw['lock_competitor_rc']==0,'CONTROL_NOT_QUIESCENT'
for name in ('results-bundle.json','explicit-process-postflight.json','post-eval-checkpoint-hash.json'):
 p=archive/name
 assert p.read_bytes()==subprocess.check_output(['git','show','0e518f9b5c074d89df8019fa2a6231083bbbefd3:'+str(p.relative_to(own))],cwd=own),'CONTROL_ARCHIVE_CHANGED:'+name
hashed=json.loads((archive/'post-eval-checkpoint-hash.json').read_text())
assert hashed['status']=='PASS' and hashed['unchanged_after_eval'] and hashed['sha256']=='786d37c6f0f46a555cb81b52979cd85c33f1b7efe1e5641a976ab1ef8d96de2e'
post=json.loads((archive/'explicit-process-postflight.json').read_text())
assert post['all_observed_pids_absent'] and not post['related_rows'] and post['neuron_device_idle'] and post['lock_competitor_rc']==0
oldpids=post['observed_pids'];assert set([3218053,3218054,3218055,3218072,3218073,3218087,3218102,3218103,3218104,3218105,3218647,3218649,3218671,3242332,3242333,3242348])==set(oldpids)

files={'NATIVE_BINDING.disabled.json':bp.read_bytes(),'host_driver.py':dp.read_bytes(),'EXPERIMENT_PROPOSAL.json':(d/'EXPERIMENT_PROPOSAL.json').read_bytes(),'MANIFEST.sha256.json':(d/'MANIFEST.sha256.json').read_bytes(),'bound-command.txt':(d/'bound-command.txt').read_bytes(),'SUPERVISOR_ASSIGNMENT.md':assignment.read_bytes()}
frozen='0009d9de8e37317952c7ff4753b74f2607f57ed4'
for name in b['support_sha256']:
 val=subprocess.check_output(['git','show',frozen+':'+str(d.relative_to(repo))+'/'+name],cwd=repo)
 assert hashlib.sha256(val).hexdigest()==b['support_sha256'][name];files[name]=val
for name,val in files.items():
 if name!='SUPERVISOR_ASSIGNMENT.md':
  assert val==subprocess.check_output(['git','show',frozen+':'+str(d.relative_to(repo))+'/'+name],cwd=repo),'FROZEN_PACKET_CHANGED:'+name
assert isinstance(manifest['files'],list) and len(manifest['files'])==28,'MANIFEST_SCHEMA'
for rec in manifest['files']:
 local=repo/rec['path'];frozen_raw=subprocess.check_output(['git','show',frozen+':'+rec['path']],cwd=repo)
 assert local.read_bytes()==frozen_raw and hashlib.sha256(frozen_raw).hexdigest()==rec['sha256'],'MANIFEST_PIN:'+rec['path']
manifest_map={Path(rec['path']).name:rec['sha256'] for rec in manifest['files']}
for name,val in files.items():
 if name not in ('SUPERVISOR_ASSIGNMENT.md','MANIFEST.sha256.json'):
  assert hashlib.sha256(val).hexdigest()==manifest_map[name],'STAGED_MANIFEST_PIN:'+name
spec=importlib.util.spec_from_file_location('transport',repo/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'] and not any(state['breaker_tripped'].values()),state
assert state['host_launch_allowed']['trn22'] is True
hs=json.loads((repo/'research/v2/state.json').read_text())['hosts']['trn22']
assert hs['label']==b['label'] and hs['active_assignment']==ap,'STATE_SCOPE_MISMATCH'
payload={'binding':b,'files':{n:{'b64':base64.b64encode(val).decode(),'sha256':hashlib.sha256(val).hexdigest()} for n,val in files.items()},'local_state':state,'previous_pids':oldpids}
body=r"""
import base64,datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
b=P['binding'];root=Path(b['root']);project=root.parent.parent
assert socket.gethostname()==b['expected_hostname'] and os.getuid()==1000,'HOST_IDENTITY'
assert not os.path.lexists(root),'LABEL_ALREADY_CONSUMED'
previous=Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-recursive6ef7-control-trn22-r2-packet')
prior_terminal=json.loads((previous/'terminal.json').read_text());prior_outer=json.loads((previous/'outer-flock-exit.json').read_text())
assert prior_terminal['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and prior_terminal['eval_exit_code']==0 and prior_outer['returncode']==0,'CONTROL_NOT_TERMINAL'
for pid in (*P['previous_pids'],3050509,3053905):
 assert not Path('/proc/'+str(pid)).exists(),'PRIOR_PID_LIVE:'+str(pid)
lk=Path('/home/ubuntu/.cache/trainium_freshstart/device0.lock').stat();assert lk.st_ino==832589 and lk.st_dev==66305,'LOCK_IDENTITY_CHANGED'
for base in (Path('/home/ubuntu'),project,root,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
 for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):assert not os.path.lexists(base/rel),'STOP:'+str(base/rel)
v=os.statvfs('/');free=v.f_bavail*v.f_frsize;assert free>=10.5*1024**3,'DISK_HEADROOM'
pause=Path('/home/ubuntu/trainium_finalpush_queue_20260925_trn22_wait_r1/runs/PAUSE_NEXT');assert pause.is_file(),'OLD_QUEUE_PAUSE_MISSING'
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
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['label'],'host':'trn22','free_bytes':free,'staged_hashes':{n:r['sha256'] for n,r in P['files'].items()},'ps_rc':ps.returncode,'job_lines':jobs,'neuron_ls':devices,'pause_sha256':hashlib.sha256(pause.read_bytes()).hexdigest(),'previous_pids_verified_absent':P['previous_pids'],'previous_pgid_sid_descendants_absent':True,'local_state':P['local_state'],'lock_inode':lk.st_ino,'lock_device':lk.st_dev,'launch_guard':'Same original lifetime flock, fresh in-lock ps/neuron-ls enforced by exact pinned driver after preliminary staging check.','scope':'One600+60s tiny4rank native diagnostic only; no train.py/full/eval/autonext'}
with (root/'preflight.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
with (root/'dispatch.log').open('xb') as out:
 process=subprocess.Popen(b['dispatch_command'],cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
launch={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['label'],'dispatch_pid':process.pid,'root':str(root),'status':'DETACHED_DISPATCH_STARTED_UNDER_LOCK_DRIVER_PENDING','no_automatic_next_job':True}
with (root/'operator-launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n')
print(json.dumps({'preflight':receipt,'launch':launch}))
"""
argv,*_=m.load_transport('trn22');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input='P='+repr(payload)+'\n'+body,text=True,capture_output=True,timeout=45)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode())
print(json.dumps({'host':'trn22','rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))

