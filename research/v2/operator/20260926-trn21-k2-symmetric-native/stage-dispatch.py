import base64,hashlib,importlib.util,json,subprocess
from pathlib import Path
repo=Path('/home/poojith-devan/Desktop/Trainium');own=Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
d=repo/'research/v2/proposals/20260926-k2-symmetric-native-trn21'
bp=d/'NATIVE_BINDING.disabled.json';dp=d/'host_driver.py'
pins={'NATIVE_BINDING.disabled.json':'926c49136160f8056776ed4678aa31d00714682c114afe0204aa0790d961eae6','host_driver.py':'ae4d1a328c868fdbfffe33d084e1525684821e1d1577c30e01afc19b982706c9','MANIFEST.sha256.json':'b85d9c91907d368a7884dbeee3c4f437e7b0f96081b57b005ff7f689a7462132'}
for n,pin in pins.items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==pin,n
b=json.loads(bp.read_text());manifest=json.loads((d/'MANIFEST.sha256.json').read_text())
ap='research/v2/verification/20260926-k2-symmetric-native-trn21-assignment.md';assignment=repo/ap
assert assignment.read_bytes()==subprocess.check_output(['git','show','6fd5f333:'+ap],cwd=repo)
for pin in pins.values():assert pin in assignment.read_text()
archive=own/'research/v2/operator/20260926-trn21-p-offline-decoder-r2/completed'
raw=json.loads((archive/'results-bundle.json').read_text())
assert 'terminal.json' in raw['files'] and 'outer-flock-exit.json' in raw['files'],'P2_NOT_ARCHIVED'
assert raw['job_lines']==[] and raw['lock_competitor_rc']==0,'P2_NOT_QUIESCENT'
post=json.loads((archive/'explicit-process-postflight.json').read_text())
assert post['all_observed_pids_absent'] and not post['related_rows'] and post['neuron_device_idle'] and post['lock_competitor_rc']==0
oldpids=post['observed_pids'];assert 3101006 in oldpids

files={'NATIVE_BINDING.disabled.json':bp.read_bytes(),'host_driver.py':dp.read_bytes(),'EXPERIMENT_PROPOSAL.json':(d/'EXPERIMENT_PROPOSAL.json').read_bytes(),'MANIFEST.sha256.json':(d/'MANIFEST.sha256.json').read_bytes(),'bound-command.txt':(d/'bound-command.txt').read_bytes(),'SUPERVISOR_ASSIGNMENT.md':assignment.read_bytes()}
for name in ('rank_entry.py','validate_receipts.py'):
 val=(d/name).read_bytes();assert hashlib.sha256(val).hexdigest()==b['support_sha256'][name];files[name]=val
for name in ('consumer_nki.py','check_native.py','reference.py'):
 val=subprocess.check_output(['git','show',b['source_module_commit']+':'+b['source_module_path']+'/'+name],cwd=repo)
 assert hashlib.sha256(val).hexdigest()==b['support_sha256'][name];files[name]=val
guard=repo/'research/v2/operator/20260926-common-host-idle-guard'
for name in ('idle_guard.py','dispatch_driver.py'):
 val=(guard/name).read_bytes();assert hashlib.sha256(val).hexdigest()==b['support_sha256'][name];files[name]=val
for name,val in files.items():
 if name in manifest['files']:assert hashlib.sha256(val).hexdigest()==manifest['files'][name]
spec=importlib.util.spec_from_file_location('transport',repo/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'] and not any(state['breaker_tripped'].values()),state
assert state['host_launch_allowed']['trn21'] is True
payload={'binding':b,'files':{n:{'b64':base64.b64encode(val).decode(),'sha256':hashlib.sha256(val).hexdigest()} for n,val in files.items()},'local_state':state,'previous_pids':oldpids}
body=r"""
import base64,datetime,hashlib,json,os,socket,subprocess
from pathlib import Path
b=P['binding'];root=Path(b['root']);project=root.parent.parent
assert socket.gethostname()==b['expected_hostname'] and os.getuid()==1000,'HOST_IDENTITY'
assert not os.path.lexists(root),'LABEL_ALREADY_CONSUMED'
previous=Path('/home/ubuntu/trainium_p_offline_summary_20260926_trn21_r2')
assert (previous/'terminal.json').is_file() and (previous/'outer-flock-exit.json').is_file(),'P2_NOT_TERMINAL'
for pid in (*P['previous_pids'],2983496):
 assert not Path('/proc/'+str(pid)).exists(),'PRIOR_PID_LIVE:'+str(pid)
lk=Path('/home/ubuntu/.cache/trainium_freshstart/device0.lock').stat();assert lk.st_ino==596823 and lk.st_dev==66305,'LOCK_IDENTITY_CHANGED'
for base in (Path('/home/ubuntu'),project,root,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
 for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):assert not os.path.lexists(base/rel),'STOP:'+str(base/rel)
v=os.statvfs('/');free=v.f_bavail*v.f_frsize;assert free>=10.5*1024**3,'DISK_HEADROOM'
pause=Path('/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1/runs/PAUSE_NEXT');assert pause.is_file(),'OLD_QUEUE_PAUSE_MISSING'
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=10);assert ps.returncode==0,ps.stderr
jobs=[l for l in ps.stdout.splitlines() if any(k in l for k in ('neuron-explorer','torchrun','train.py','host_queue.py','host_driver.py','submission_rehearsal.py','neuronx-cc','walrus_driver','profile_actual_loop.py','preflight_dryrun.py','prepare.py eval-public','run_job.py','contained.py','probe.py','rank_entry.py','check_native.py'))];assert not jobs,'BUSY_PS:'+str(jobs)
rows=[l.split(None,5) for l in ps.stdout.splitlines()[1:]]
assert not any(int(r[1]) in P['previous_pids'] or int(r[2]) in P['previous_pids'] or int(r[3]) in P['previous_pids'] for r in rows),'P2_DESCENDANT_PGID_SID_LIVE'
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=20);assert ls.returncode==0,ls.stderr
devices=json.loads(ls.stdout);assert len(devices)==1 and devices[0]['neuron_processes']==[],'BUSY_DEVICE'
assert Path('/home/ubuntu/workspace/native_venv/bin/torchrun').is_file()
root.mkdir(mode=0o700)
for name,rec in P['files'].items():
 dst=root/name;assert not dst.is_symlink() and not dst.exists()
 val=base64.b64decode(rec['b64']);assert hashlib.sha256(val).hexdigest()==rec['sha256']
 with dst.open('xb') as f:f.write(val)
 os.chmod(dst,0o600);assert hashlib.sha256(dst.read_bytes()).hexdigest()==rec['sha256']
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['label'],'host':'trn21','free_bytes':free,'staged_hashes':{n:r['sha256'] for n,r in P['files'].items()},'ps_rc':ps.returncode,'job_lines':jobs,'neuron_ls':devices,'pause_sha256':hashlib.sha256(pause.read_bytes()).hexdigest(),'previous_pids_verified_absent':P['previous_pids'],'previous_pgid_sid_descendants_absent':True,'local_state':P['local_state'],'lock_inode':lk.st_ino,'lock_device':lk.st_dev,'launch_guard':'Same original lifetime flock, fresh in-lock ps/neuron-ls enforced by exact pinned driver after preliminary staging check.','scope':'One600+60s tiny4rank native diagnostic only; no train.py/full/eval/autonext'}
with (root/'preflight.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
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
