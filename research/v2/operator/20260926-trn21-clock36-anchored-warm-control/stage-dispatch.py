"""Prepared one-use anchored full control transport, exact bytes, separate release."""
import base64,hashlib,importlib.util,json,re,shlex,subprocess,sys
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('NORMAL_PYTHON_REQUIRED')
repo=Path('/home/poojith-devan/Desktop/Trainium');own=Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
rel='research/v2/proposals/20260926-clock36-anchored-warm-control'
d=repo/rel;frozen='3170c0bf590ba2bc19fc7643084e6495c2220c97'
manifest_raw=(d/'PACKET_MANIFEST.json').read_bytes()
assert manifest_raw==subprocess.check_output(['git','show',frozen+':'+rel+'/PACKET_MANIFEST.json'],cwd=repo),'MANIFEST_CHANGED'
manifest=json.loads(manifest_raw);files={}
selected={'ANCHORED_BINDING.disabled.json','EXPERIMENT_PROPOSAL.json','audit-normal.json','identity_checks.py','host_driver.py','package/launch-command.txt','package/provenance.json'}
for name,pin in manifest['files'].items():
 raw=(d/name).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==pin,name
 assert raw==subprocess.check_output(['git','show',frozen+':'+rel+'/'+name],cwd=repo),name
 if name in selected:files[name]=raw
plan=json.loads(files['ANCHORED_BINDING.disabled.json'])
assert hashlib.sha256(files['ANCHORED_BINDING.disabled.json']).hexdigest()=='e71a92a55043871b7dd5660c191fad4135d20384e6fa1be3b6edc91398d83e10'
assert hashlib.sha256(files['host_driver.py']).hexdigest()=='706b7945a7316fe610c3abef1efedaf8a95d0203eba1516fb93ac1d4a6aa9644'
assert plan['label']=='fs-20260926-clock36-anchored-warm-control-trn21-r1'
assert plan['original_lock']=={'device':66305,'inode':596823}
assert [i for i,(a,b) in enumerate(zip(plan['parent_training_command'],plan['training_command'])) if a!=b]==[3]
assert plan['training_command'][3]==plan['executed_source']
assert shlex.split(files['package/launch-command.txt'].decode())==plan['training_command']
for key,pin in plan['package_pins'].items():assert hashlib.sha256(files['package/'+key]).hexdigest()==pin
source=subprocess.check_output(['git','show',plan['source_commit']+':'+plan['source_git_path']],cwd=repo)
assert hashlib.sha256(source).hexdigest()==plan['source_sha256']
files['package/train.py']=source;files['PACKET_MANIFEST.json']=manifest_raw
for name in ('idle_guard.py','dispatch_driver.py'):
 val=(repo/'research/v2/operator/20260926-common-host-idle-guard'/name).read_bytes()
 assert hashlib.sha256(val).hexdigest()==plan['support'][name]
 files[name]=val
for name,pin in plan['support'].items():assert hashlib.sha256(files[name]).hexdigest()==pin
raw_anchor=(repo/plan['anchor_raw_path']).read_bytes()
assert hashlib.sha256(raw_anchor).hexdigest()==plan['anchor_raw_sha256']
# Preserve all nanoseconds as Python int; no JavaScript numeric projection.
assert all(type(a[k]) is int and a[k]>2**53 for a in plan['anchor_observation'].values() for k in ('mtime_ns','ctime_ns'))
if sys.argv[1:]==['--local-check']:
 print(json.dumps({'status':'LOCAL_EXACT_ANCHORED_TRANSPORT_READY_ONLY','files':len(files),'integer_preserving_binding':True,'host_calls':False,'launch_authorized':False,'floor_plus_footprint_bytes':int(12.5*1024**3)}));sys.exit(0)
assert len(sys.argv)==3,'EXACT_RELEASE_COMMIT_AND_PATH_REQUIRED'
release_commit,release_path=sys.argv[1:]
assert re.fullmatch(r'[0-9a-f]{40}',release_commit)
assert release_path.startswith('research/v2/verification/') and release_path.endswith('.md') and '..' not in Path(release_path).parts
assignment=(repo/release_path).read_bytes()
assert assignment==subprocess.check_output(['git','show',release_commit+':'+release_path],cwd=repo)
for x in [plan['label'],manifest['files']['ANCHORED_BINDING.disabled.json'],manifest['files']['host_driver.py'],plan['source_sha256']]:assert x in assignment.decode(),'RELEASE_PIN_MISMATCH'
files['SUPERVISOR_ASSIGNMENT.md']=assignment
archive=own/'research/v2/operator/20260926-trn21-confirmed36-clean-checkpoint/completed'
prior_commit='2e7edfc815bbbd707348b5af8c40395b9fc25260'
for name in ('results-bundle.json','explicit-process-postflight.json'):
 p=archive/name;assert p.read_bytes()==subprocess.check_output(['git','show',prior_commit+':'+str(p.relative_to(own))],cwd=own),'PREDECESSOR_RAW_CHANGED'
prior=json.loads((archive/'explicit-process-postflight.json').read_bytes());raw=json.loads((archive/'results-bundle.json').read_bytes())
assert prior['all_observed_pids_absent'] and prior['neuron_device_idle'] and prior['related_rows']==[] and prior['lock_competitor_rc']==0
assert (prior['lock_device'],prior['lock_inode'])==(66305,596823)
assert set(prior['observed_pids'])=={3177118,3177119,3177120,3177150,3177151}
assert not any(any(row[key] in prior['observed_pids'] for key in ('pid','ppid','pgid','sid')) for row in prior['complete_process_census'])
terminal_raw=raw['files']['terminal.json']['text'].encode();outer_raw=raw['files']['outer-flock-exit.json']['text'].encode()
assert json.loads(terminal_raw)['status']=='CPU_CHECKPOINT_ONLY_COMPLETE_PENDING_INDEPENDENT_REVIEW' and json.loads(terminal_raw)['exit_code']==0
assert json.loads(outer_raw)['returncode']==0
spec=importlib.util.spec_from_file_location('transport',repo/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state();assert not state['local_stops_present'] and not any(state['breaker_tripped'].values()),state
assert state['host_launch_allowed']['trn21'] is True,'HOST_NOT_RELEASED'
hs=json.loads((repo/'research/v2/state.json').read_bytes())['hosts']['trn21']
assert hs['active_assignment']==release_path and hs['label']==plan['label'],'STATE_SCOPE_MISMATCH'
payload={'plan':plan,'files':{n:{'b64':base64.b64encode(v).decode(),'sha256':hashlib.sha256(v).hexdigest()} for n,v in files.items()},'local_state':state,'previous_pids':prior['observed_pids'],'assignment_path':release_path,'previous_root':'/home/ubuntu/trainium_confirmed36_clean_checkpoint_20260926_trn21_r1','previous_terminal_sha256':hashlib.sha256(terminal_raw).hexdigest(),'previous_outer_sha256':hashlib.sha256(outer_raw).hexdigest()}

body=r'''
import base64,datetime,fcntl,hashlib,importlib.util,json,os,socket,subprocess
from pathlib import Path
plan=P['plan'];root=Path(plan['root']);project=Path(plan['prepare_path']).parent
assert socket.gethostname()==plan['expected_hostname'] and os.getuid()==1000,'HOST_IDENTITY'
assert not os.path.lexists(root),'LABEL_ALREADY_CONSUMED'
previous=Path(P['previous_root'])
assert (previous/'terminal.json').is_file() and (previous/'outer-flock-exit.json').is_file(),'PRIOR_NOT_TERMINAL'
assert hashlib.sha256((previous/'terminal.json').read_bytes()).hexdigest()==P['previous_terminal_sha256'],'PRIOR_TERMINAL_CHANGED'
assert hashlib.sha256((previous/'outer-flock-exit.json').read_bytes()).hexdigest()==P['previous_outer_sha256'],'PRIOR_OUTER_CHANGED'
assert json.loads((previous/'terminal.json').read_text())['status']=='CPU_CHECKPOINT_ONLY_COMPLETE_PENDING_INDEPENDENT_REVIEW','PRIOR_STATUS_CHANGED'
assert json.loads((previous/'outer-flock-exit.json').read_text())['returncode']==0,'PRIOR_OUTER_FAILED'
for pid in (*P['previous_pids'],2983496):
    assert not Path('/proc/'+str(pid)).exists(),'PRIOR_PID_LIVE:'+str(pid)
lock=Path('/home/ubuntu/.cache/trainium_freshstart/device0.lock')
lockfd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
lk=os.fstat(lockfd);assert (lk.st_dev,lk.st_ino)==(66305,596823),'LOCK_IDENTITY_CHANGED'
fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
for base in (Path('/home/ubuntu'),project,root,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
    for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):
        assert not os.path.lexists(base/rel),'STOP:'+str(base/rel)
v=os.statvfs('/');free=v.f_bavail*v.f_frsize
# Retained checkpoints stay in place; reserve new logs/support/scratch only.
projected=2*1024**3
assert free>=10.5*1024**3+projected,'DISK_FLOOR_PLUS_PROJECTED_FOOTPRINT'
mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
assert mem>=64*1024**3,'AVAILABLE_RAM_FLOOR'
assert hashlib.sha256((project/'prepare.py').read_bytes()).hexdigest()==plan['prepare_sha256'],'PREPARE_SHA'
pause=Path('/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1/runs/PAUSE_NEXT');assert pause.is_file(),'OLD_QUEUE_PAUSE_MISSING'
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=10)
assert ps.returncode==0,ps.stderr
jobs=[l for l in ps.stdout.splitlines() if any(k in l for k in ('neuron-explorer','torchrun','train.py','host_queue.py','host_driver.py','submission_rehearsal.py','neuronx-cc','walrus_driver','preflight_dryrun.py','prepare.py eval-public','run_job.py','contained.py','rank_entry.py','check_native.py','compare_checkpoints.py'))]
assert jobs==[],'BUSY_PS:'+str(jobs)
rows=[l.split(None,5) for l in ps.stdout.splitlines()[1:]]
assert not any(any(int(r[i]) in P['previous_pids'] for i in (1,2,3)) for r in rows),'PRIOR_DESCENDANT_PGID_SID_LIVE'
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=20)
assert ls.returncode==0,ls.stderr
devices=json.loads(ls.stdout);assert len(devices)==1 and devices[0]['neuron_processes']==[],'BUSY_DEVICE'
nodes=sorted(str(p) for p in Path('/dev').glob('neuron[0-9]*'));assert nodes,'NO_NEURON_DEVICE'
holders=subprocess.run(['/usr/bin/fuser',*nodes],capture_output=True,text=True,timeout=10)
assert holders.returncode==1 and not holders.stdout.strip(),'DEVICE_HOLDERS:'+holders.stdout
assert Path('/home/ubuntu/workspace/native_venv/bin/python').is_file()
# Exact readonly anchor stat/hash; unchanged driver enforces full identity/import contract.
anchor_observed={}
for name,pin in plan['anchor_observation'].items():
    path=Path(name);assert path.is_file() and not path.is_symlink(),'ANCHOR_FILE:'+name
    before=path.stat()
    fields={'device':before.st_dev,'inode':before.st_ino,'mode':before.st_mode,'uid':before.st_uid,'size':before.st_size,'mtime_ns':before.st_mtime_ns,'ctime_ns':before.st_ctime_ns}
    assert all(fields[key]==pin[key] for key in fields),'ANCHOR_STAT:'+name
    data=path.read_bytes();after=path.stat()
    assert all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_mode','st_uid','st_size','st_mtime_ns','st_ctime_ns')),'ANCHOR_CHANGED:'+name
    assert hashlib.sha256(data).hexdigest()==pin['sha256'],'ANCHOR_HASH:'+name
    anchor_observed[name]=fields|{'sha256':pin['sha256']}
assert hashlib.sha256(Path(plan['harness_path']).read_bytes()).hexdigest()==plan['harness_sha256'],'HARNESS_SHA'
for phase in ('training_environment','evaluation_environment'):
    for key in ('TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'):
        assert Path(plan[phase][key]).is_dir(),'RETAINED_CACHE_MISSING:'+phase+':'+key
root.mkdir(mode=0o700)
for name,rec in P['files'].items():
    dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True)
    assert not dst.is_symlink() and not dst.exists()
    val=base64.b64decode(rec['b64']);assert hashlib.sha256(val).hexdigest()==rec['sha256']
    with dst.open('xb') as f:f.write(val)
    os.chmod(dst,0o444);assert hashlib.sha256(dst.read_bytes()).hexdigest()==rec['sha256']

(root/'metrics-training').mkdir(mode=0o700)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'trn21','root':str(root),
    'free_bytes':free,'projected_footprint_bytes':projected,'minimum_free_plus_footprint_bytes':10.5*1024**3+projected,
    'available_ram_bytes':mem,'staged_hashes':{n:r['sha256'] for n,r in P['files'].items()},
    'ps_rc':ps.returncode,'job_lines':jobs,'neuron_ls':devices,'fuser_rc':holders.returncode,'fuser_stdout':holders.stdout,'fuser_stderr':holders.stderr,'exact_anchor_stat_hash':anchor_observed,'previous_pids_verified_absent':P['previous_pids'],
    'previous_pgid_sid_descendants_absent':True,'local_state':P['local_state'],'lock_inode':lk.st_ino,'lock_device':lk.st_dev,
    'resource_checks_under_original_exclusive_lock':True,'assignment':P['assignment_path'],
    'scope':'One separately released anchored exact36 full1800 and attached fresh20M only; no source changes, cache-hit claim, repair or automatic successor',
    'launch_guard':'Pinned reviewed composition reacquires original lifetime lock, checks ps/neuron-ls before both phases and authenticates all anchors at every boundary; no old file writes.'}
with (root/'operator-preflight.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
fcntl.flock(lockfd,fcntl.LOCK_UN);os.close(lockfd)
with (root/'dispatch.log').open('xb') as out:
    child=subprocess.Popen(plan['dispatch_command'],cwd=root,stdin=subprocess.DEVNULL,
        stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
launch={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'dispatch_pid':child.pid,
    'status':'DETACHED_DISPATCH_STARTED_UNDER_LOCK_DRIVER_PENDING','no_automatic_next_job':True}
with (root/'operator-launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n')
print(json.dumps({'preflight':receipt,'launch':launch}))
'''

argv,*_=m.load_transport('trn21');argv[-1]='exec python3 -I -B -'
wire='P='+repr(payload)+'\n'+body
assert not m.SENSITIVE.search(wire.encode()),'PAYLOAD_SENSITIVE_MARKER'
r=subprocess.run(argv,input=wire,text=True,capture_output=True,timeout=45)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode()),'OUTPUT_SENSITIVE_MARKER'
print(json.dumps({'host':'trn21','rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
