"""One-use exact checkpoint transport; preparation is not launch authorization."""
import base64, hashlib, importlib.util, json, re, subprocess, sys
from pathlib import Path

if sys.flags.optimize:
    raise RuntimeError('NORMAL_PYTHON_REQUIRED')
repo=Path('/home/poojith-devan/Desktop/Trainium')
own=Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
packet_rel='research/v2/proposals/20260926-confirmed36-clean-checkpoint-cpu'
packet=own/packet_rel
frozen='865f5bb82873aaff785977049381a780e6380b23'
pins={'INPUTS.json':'0cc0569936b0b1653bcc946edcdedd1bf91d3f671fabbc205a0a7b883bf9a15e',
      'host_driver.py':'9badeccbd2bb33deeb53749ac3a18afc322fba60ec38e5c100457dbc019fc3b5',
      'OPERATOR_BINDING.disabled.json':'87279109942cbac4453d110fa102879e7d3189692f78c2013ed0cf81632699ea'}
files={}
for name,pin in pins.items():
    val=(packet/name).read_bytes()
    assert hashlib.sha256(val).hexdigest()==pin,name
    assert val==subprocess.check_output(['git','show',frozen+':'+packet_rel+'/'+name],cwd=own),name
    files[name]=val
manifest=json.loads((packet/'ACTUAL_MANIFEST.json').read_bytes())
for name,pin in manifest['files'].items():
    val=(packet/name).read_bytes()
    assert val==subprocess.check_output(['git','show',frozen+':'+packet_rel+'/'+name],cwd=own) and hashlib.sha256(val).hexdigest()==pin,name
plan=json.loads(files['OPERATOR_BINDING.disabled.json']);inputs=json.loads(files['INPUTS.json'])
assert plan['root']=='/home/ubuntu/trainium_confirmed36_clean_checkpoint_20260926_trn21_r1'
assert plan['label']=='fs-20260926-confirmed36-clean-checkpoint-trn21-r1'
assert plan['driver_sha256']==pins['host_driver.py'] and plan['actual_inputs_sha256']==pins['INPUTS.json']
assert inputs['status']=='FROZEN_REAL_INPUTS' and len(inputs['files'])==17
assert inputs['root']==plan['old_readonly_root']
helper=subprocess.check_output(['git','show',plan['helper_commit']+':'+plan['helper_git_path']],cwd=repo)
assert hashlib.sha256(helper).hexdigest()==plan['helper_sha256']
files['compare_checkpoints.py']=helper
proposal=(repo/packet_rel/'EXPERIMENT_PROPOSAL.json').read_bytes()
assert hashlib.sha256(proposal).hexdigest()==plan['proposal_sha256']
files['EXPERIMENT_PROPOSAL.json']=proposal
for name,pin in (('idle_guard.py',plan['idle_guard_sha256']),('dispatch_driver.py',plan['dispatch_driver_sha256'])):
    val=(repo/'research/v2/operator/20260926-common-host-idle-guard'/name).read_bytes()
    assert hashlib.sha256(val).hexdigest()==pin,name
    files[name]=val
archive=own/'research/v2/operator/20260926-trn21-confirmed36-clean-cpu/completed'
previous_commit='96392d1dacf2b7537098528123333b2beefbd81f'
for name in ('results-bundle.json','explicit-process-postflight.json'):
    p=archive/name
    assert p.read_bytes()==subprocess.check_output(['git','show',previous_commit+':'+str(p.relative_to(own))],cwd=own),name
post=json.loads((archive/'explicit-process-postflight.json').read_bytes())
raw=json.loads((archive/'results-bundle.json').read_bytes())
assert post['all_observed_pids_absent'] and post['related_rows']==[] and post['neuron_device_idle']
assert post['lock_competitor_rc']==0 and (post['lock_device'],post['lock_inode'])==(66305,596823)
oldpids=post['observed_pids'];assert set(oldpids)=={3176685,3176686,3176687,3176720,3176776}
assert all(not row['exists'] for row in post['pid_rows'])
assert not any(any(row[key] in oldpids for key in ('pid','ppid','pgid','sid')) for row in post['complete_process_census'])
terminal_raw=raw['files']['terminal.json']['text'].encode()
outer_raw=raw['files']['outer-flock-exit.json']['text'].encode()
assert json.loads(terminal_raw)['status']=='CPU_PAIR_COMPLETE_PENDING_INDEPENDENT_REVIEW'
assert json.loads(outer_raw)['returncode']==0
if sys.argv[1:]==['--local-check']:
    print(json.dumps({'status':'LOCAL_EXACT_INPUTS_AND_TRANSPORT_READY_NOT_AUTHORIZED','frozen':frozen,'files':len(files),'retained_inputs':17,'host_calls':False,'stage_authorized':False,'launch_authorized':False}))
    sys.exit(0)
assert len(sys.argv)==3,'EXACT_RELEASE_COMMIT_AND_PATH_REQUIRED'
release_commit,release_path=sys.argv[1:]
assert re.fullmatch(r'[0-9a-f]{40}',release_commit)
assert release_path.startswith('research/v2/verification/') and release_path.endswith('.md') and '..' not in Path(release_path).parts
assignment=(repo/release_path).read_bytes()
assert assignment==subprocess.check_output(['git','show',release_commit+':'+release_path],cwd=repo)
for value in (plan['label'],pins['INPUTS.json'],pins['host_driver.py'],plan['helper_sha256']):
    assert value in assignment.decode(),'RELEASE_NOT_EXACT_PACKET'
files['SUPERVISOR_ASSIGNMENT.md']=assignment
spec=importlib.util.spec_from_file_location('transport',repo/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state=m.local_state()
assert not state['local_stops_present'] and not any(state['breaker_tripped'].values()),state
assert state['host_launch_allowed']['trn21'] is True,'HOST_NOT_RELEASED'
hs=json.loads((repo/'research/v2/state.json').read_bytes())['hosts']['trn21']
assert hs['active_assignment']==release_path and hs['label']==plan['label'],'STATE_SCOPE_MISMATCH'
payload={'plan':plan,'files':{n:{'b64':base64.b64encode(v).decode(),'sha256':hashlib.sha256(v).hexdigest()} for n,v in files.items()},
         'local_state':state,'previous_pids':oldpids,'assignment_path':release_path,'input_pins':inputs['files'],
         'previous_terminal_sha256':hashlib.sha256(terminal_raw).hexdigest(),'previous_outer_sha256':hashlib.sha256(outer_raw).hexdigest()}

body = r'''
import base64,datetime,fcntl,hashlib,importlib.util,json,os,socket,subprocess
from pathlib import Path
plan=P['plan'];root=Path(plan['root']);project=Path(plan['prepare_path']).parent
assert socket.gethostname()==plan['expected_hostname'] and os.getuid()==1000,'HOST_IDENTITY'
assert not os.path.lexists(root),'LABEL_ALREADY_CONSUMED'
previous=Path(plan['old_readonly_root'])
assert (previous/'terminal.json').is_file() and (previous/'outer-flock-exit.json').is_file(),'PRIOR_NOT_TERMINAL'
assert hashlib.sha256((previous/'terminal.json').read_bytes()).hexdigest()==P['previous_terminal_sha256'],'PRIOR_TERMINAL_CHANGED'
assert hashlib.sha256((previous/'outer-flock-exit.json').read_bytes()).hexdigest()==P['previous_outer_sha256'],'PRIOR_OUTER_CHANGED'
assert json.loads((previous/'terminal.json').read_text())['status']=='CPU_PAIR_COMPLETE_PENDING_INDEPENDENT_REVIEW','PRIOR_STATUS_CHANGED'
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
projected=1024**3
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
for name,pin in P['input_pins'].items():
    p=Path(name);assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'],'RETAINED_INPUT_IDENTITY:'+name
    with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==pin['sha256'],'RETAINED_INPUT_SHA:'+name
root.mkdir(mode=0o700)
for name,rec in P['files'].items():
    dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True)
    assert not dst.is_symlink() and not dst.exists()
    val=base64.b64decode(rec['b64']);assert hashlib.sha256(val).hexdigest()==rec['sha256']
    with dst.open('xb') as f:f.write(val)
    os.chmod(dst,0o444);assert hashlib.sha256(dst.read_bytes()).hexdigest()==rec['sha256']

receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'trn21','root':str(root),
    'free_bytes':free,'projected_footprint_bytes':projected,'minimum_free_plus_footprint_bytes':10.5*1024**3+projected,
    'available_ram_bytes':mem,'staged_hashes':{n:r['sha256'] for n,r in P['files'].items()},
    'ps_rc':ps.returncode,'job_lines':jobs,'neuron_ls':devices,'fuser_rc':holders.returncode,'fuser_stdout':holders.stdout,'fuser_stderr':holders.stderr,'retained_input_pins_verified':len(P['input_pins']),'previous_pids_verified_absent':P['previous_pids'],
    'previous_pgid_sid_descendants_absent':True,'local_state':P['local_state'],'lock_inode':lk.st_ino,'lock_device':lk.st_dev,
    'resource_checks_under_original_exclusive_lock':True,'assignment':P['assignment_path'],
    'scope':'One retained-checkpoint CPU helper only; no G2 rerun/native/full/eval/autonext',
    'launch_guard':'Pinned driver reacquires same original lifetime lock and rechecks ps/neuron-ls before the sole child; retained inputs hashed before and after.'}
with (root/'operator-preflight.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
fcntl.flock(lockfd,fcntl.LOCK_UN);os.close(lockfd)
with (root/'dispatch.log').open('xb') as out:
    child=subprocess.Popen(plan['detached_command_template'],cwd=root,stdin=subprocess.DEVNULL,
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

