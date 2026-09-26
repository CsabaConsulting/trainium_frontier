"""Exact-byte one-use standard CPU pair transport. LOCAL-ONLY until separate release."""
import base64, hashlib, importlib.util, json, re, subprocess, sys
from pathlib import Path

if sys.flags.optimize:
    raise RuntimeError('NORMAL_PYTHON_REQUIRED')
repo = Path('/home/poojith-devan/Desktop/Trainium')
own = Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
packet_rel = 'research/v2/proposals/20260926-confirmed36-clean-cpu-trn21-r1'
packet = repo / packet_rel
frozen = '8cc56def09227556e08322254c85542439ea9b22'
pins = {
    'CPU_PLAN.disabled.json': 'e873dde0bc30de2cf358ce292c05d0b7d7269380dcb6cb15048b96e5b5c7817f',
    'host_driver.py': 'ed588e78272305a28a0ded69124fb5865a6794a62344f3d74707ed557b132344',
    'MANIFEST.sha256.json': '8dd9e4aec2d62cb40f04d7ec27bf3609c3ab28dfd94ce720a913a22b04612dc0',
    'EXPERIMENT_PROPOSAL.json': '46b8e3fc855fd66e3f51e30873e69c8e02622a1e78497b0a0cb8c4755e57d51a',
}
for name, pin in pins.items():
    raw = (packet/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin, name
    assert raw == subprocess.check_output(['git','show',frozen+':'+packet_rel+'/'+name],cwd=repo)
plan = json.loads((packet/'CPU_PLAN.disabled.json').read_bytes())
manifest = json.loads((packet/'MANIFEST.sha256.json').read_bytes())
assert plan['stage_root'] == '/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1'
assert plan['label'] == 'fs-20260926-confirmed36-clean-cpu-trn21-r1'
assert plan['driver_sha256'] == pins['host_driver.py']
assert plan['ordering'] == ['reference','candidate','standard-compare']
assert len(manifest['files']) == 10
files = {}
for name,pin in manifest['files'].items():
    raw = subprocess.check_output(['git','show',frozen+':'+packet_rel+'/'+name],cwd=repo)
    assert (packet/name).read_bytes() == raw and hashlib.sha256(raw).hexdigest() == pin,name
    if name in ('CPU_PLAN.disabled.json','EXPERIMENT_PROPOSAL.json','README.md','host_driver.py') or name.endswith('/request.json'):
        files[name] = raw
files['MANIFEST.sha256.json'] = (packet/'MANIFEST.sha256.json').read_bytes()
for role in ('reference','candidate','g2'):
    spec = plan['sources'][role]
    raw = subprocess.check_output(['git','show',spec['commit']+':'+spec['git_path']],cwd=repo)
    assert hashlib.sha256(raw).hexdigest() == spec['sha256'],role
    files[str(Path(spec['stage_path']).relative_to(plan['stage_root']))] = raw
for role,name in (('idle_guard','idle_guard.py'),('dispatch','dispatch_driver.py')):
    spec = plan['sources'][role];raw = (repo/spec['git_path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == spec['sha256'],role
    files[name] = raw
mechanics_path = 'research/v2/verification/20260926-confirmed36-clean-cpu-packet-root.md'
mechanics = (repo/mechanics_path).read_bytes()
assert mechanics == subprocess.check_output(['git','show','461f43511d518907934c6ea788a96858e326b2e0:'+mechanics_path],cwd=repo)
assert hashlib.sha256(mechanics).hexdigest() == '928511cb18d381abf899f8a816674c5bf8ca5d79e31fd53084ee872a93d2e910'
if sys.argv[1:] == ['--local-check']:
    print(json.dumps({'status':'EXACT_LOCAL_PACKET_AUTHENTICATED_ONLY','frozen':frozen,'files_to_stage':len(files),'plan_sha256':pins['CPU_PLAN.disabled.json'],'host_calls':False,'stage_authorized':False,'launch_authorized':False}))
    sys.exit(0)

assert len(sys.argv) == 4,'EXACT_RELEASE_COMMIT_PATH_AND_PREDECESSOR_FINAL_COMMIT_REQUIRED'
release_commit,release_path,previous_commit = sys.argv[1:]
assert re.fullmatch(r'[0-9a-f]{40}',release_commit) and re.fullmatch(r'[0-9a-f]{40}',previous_commit),'EXACT_COMMIT_FORMAT'
assert release_path.startswith('research/v2/verification/') and release_path.endswith('.md') and '..' not in Path(release_path).parts
assignment = (repo/release_path).read_bytes()
assert assignment == subprocess.check_output(['git','show',release_commit+':'+release_path],cwd=repo)
for pin in (plan['label'],pins['CPU_PLAN.disabled.json'],pins['host_driver.py']):
    assert pin in assignment.decode(),'RELEASE_NOT_EXACT_PACKET'
files['SUPERVISOR_ASSIGNMENT.md'] = assignment
archive = own/'research/v2/operator/20260926-trn21-clock36-lr012/completed'
for name in ('results-bundle.json','explicit-process-postflight.json'):
    path = archive/name
    assert path.read_bytes() == subprocess.check_output(['git','show',previous_commit+':'+str(path.relative_to(own))],cwd=own),'PREDECESSOR_RAW_NOT_COMMITTED:'+name
post = json.loads((archive/'explicit-process-postflight.json').read_bytes())
raw = json.loads((archive/'results-bundle.json').read_bytes())
assert post['all_observed_pids_absent'] and post['related_rows'] == [] and post['neuron_device_idle']
assert post['lock_competitor_rc'] == 0 and (post['lock_device'],post['lock_inode']) == (66305,596823)
oldpids = post['observed_pids']
known = {3152603,3152604,3152605,3152621,3152622,3152637,3152652,3152653,3152654,3152655,3153251,3153259,3153290}
assert known <= set(oldpids),'KNOWN_PREDECESSOR_IDENTITIES_MISSING'
assert all(not row['exists'] for row in post['pid_rows']),'PRIOR_PID_STILL_EXISTS'
assert set(row['pid'] for row in post['pid_rows']) == set(oldpids),'POSTFLIGHT_PID_COVERAGE'
terminal_raw = raw['files']['terminal.json']['text'].encode()
outer_raw = raw['files']['outer-flock-exit.json']['text'].encode()
terminal = json.loads(terminal_raw);outer = json.loads(outer_raw)
assert terminal['label'] == 'fs-20260926-clock36-lr012-trn21-r1'
assert terminal['status'] == 'COMPLETE_PENDING_INDEPENDENT_REVIEW' and terminal['eval_exit_code'] == 0
assert outer['returncode'] == 0,'PRIOR_OUTER_NOT_SUCCESSFUL'
for name in ('training.log','train/rehearsal.json','fresh-public20m.log','eval/rehearsal.json','checkpoint.json','run-exit-code.txt','eval-exit-code.txt'):
    assert name in raw['files'],'PREDECESSOR_ARTIFACT_MISSING:'+name
train = json.loads(raw['files']['train/rehearsal.json']['text'])
assert train['returncode'] == 0 and train['charged_through_process_exit_within_budget'] is True
assert raw['files']['run-exit-code.txt']['text'].strip() == raw['files']['eval-exit-code.txt']['text'].strip() == '0'
spec = importlib.util.spec_from_file_location('transport',repo/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state = m.local_state()
assert not state['local_stops_present'] and not any(state['breaker_tripped'].values()),state
assert state['host_launch_allowed']['trn21'] is True,'HOST_NOT_RELEASED'
hs = json.loads((repo/'research/v2/state.json').read_bytes())['hosts']['trn21']
assert hs['active_assignment'] == release_path and hs['label'] == plan['label'],'STATE_SCOPE_MISMATCH'
payload = {'plan':plan,'files':{n:{'b64':base64.b64encode(v).decode(),'sha256':hashlib.sha256(v).hexdigest()} for n,v in files.items()},'local_state':state,'previous_pids':oldpids,'assignment_path':release_path,'previous_terminal_sha256':hashlib.sha256(terminal_raw).hexdigest(),'previous_outer_sha256':hashlib.sha256(outer_raw).hexdigest()}

body = "\nimport base64,datetime,fcntl,hashlib,importlib.util,json,os,socket,subprocess\nfrom pathlib import Path\nplan=P['plan'];root=Path(plan['stage_root']);project=Path(plan['prepare_dir'])\nassert socket.gethostname()==plan['expected_hostname'] and os.getuid()==1000,'HOST_IDENTITY'\nassert not os.path.lexists(root),'LABEL_ALREADY_CONSUMED'\nprevious=project/'out/fs-20260926-clock36-lr012-trn21-r1-packet'\nassert (previous/'terminal.json').is_file() and (previous/'outer-flock-exit.json').is_file(),'PRIOR_NOT_TERMINAL'\nassert hashlib.sha256((previous/'terminal.json').read_bytes()).hexdigest()==P['previous_terminal_sha256'],'PRIOR_TERMINAL_CHANGED'\nassert hashlib.sha256((previous/'outer-flock-exit.json').read_bytes()).hexdigest()==P['previous_outer_sha256'],'PRIOR_OUTER_CHANGED'\nassert json.loads((previous/'terminal.json').read_text())['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW','PRIOR_STATUS_CHANGED'\nassert json.loads((previous/'terminal.json').read_text())['eval_exit_code']==0 and json.loads((previous/'outer-flock-exit.json').read_text())['returncode']==0,'PRIOR_NOT_SUCCESSFUL'\nfor pid in (*P['previous_pids'],2983496):\n    assert not Path('/proc/'+str(pid)).exists(),'PRIOR_PID_LIVE:'+str(pid)\nlock=Path('/home/ubuntu/.cache/trainium_freshstart/device0.lock')\nlockfd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)\nlk=os.fstat(lockfd);assert (lk.st_dev,lk.st_ino)==(66305,596823),'LOCK_IDENTITY_CHANGED'\nfcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)\nfor base in (Path('/home/ubuntu'),project,root,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):\n    for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):\n        assert not os.path.lexists(base/rel),'STOP:'+str(base/rel)\nv=os.statvfs('/');free=v.f_bavail*v.f_frsize\n# Two retained ~453MB checkpoints plus exact approved2GiB scratch/log footprint.\nprojected=2*1024**3\nassert free>=10.5*1024**3+projected,'DISK_FLOOR_PLUS_PROJECTED_FOOTPRINT'\nmem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024\nassert mem>=64*1024**3,'AVAILABLE_RAM_FLOOR'\nassert hashlib.sha256((project/'prepare.py').read_bytes()).hexdigest()==plan['sources']['prepare']['sha256'],'PREPARE_SHA'\npause=Path('/home/ubuntu/trainium_final_push_conditional_20260925_trn21_r1/runs/PAUSE_NEXT');assert pause.is_file(),'OLD_QUEUE_PAUSE_MISSING'\nps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=10)\nassert ps.returncode==0,ps.stderr\njobs=[l for l in ps.stdout.splitlines() if any(k in l for k in ('neuron-explorer','torchrun','train.py','host_queue.py','host_driver.py','submission_rehearsal.py','neuronx-cc','walrus_driver','preflight_dryrun.py','prepare.py eval-public','run_job.py','contained.py','rank_entry.py','check_native.py','compare_checkpoints.py'))]\nassert jobs==[],'BUSY_PS:'+str(jobs)\nrows=[l.split(None,5) for l in ps.stdout.splitlines()[1:]]\nassert not any(any(int(r[i]) in P['previous_pids'] for i in (1,2,3)) for r in rows),'PRIOR_DESCENDANT_PGID_SID_LIVE'\nls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=20)\nassert ls.returncode==0,ls.stderr\ndevices=json.loads(ls.stdout);assert len(devices)==1 and devices[0]['neuron_processes']==[],'BUSY_DEVICE'\nnodes=sorted(str(p) for p in Path('/dev').glob('neuron[0-9]*'));assert nodes,'NO_NEURON_DEVICE'\nholders=subprocess.run(['/usr/bin/fuser',*nodes],capture_output=True,text=True,timeout=10)\nassert holders.returncode==1 and not holders.stdout.strip(),'DEVICE_HOLDERS:'+holders.stdout\nassert Path('/home/ubuntu/workspace/native_venv/bin/python').is_file()\nroot.mkdir(mode=0o700)\nfor name,rec in P['files'].items():\n    dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True)\n    assert not dst.is_symlink() and not dst.exists()\n    val=base64.b64decode(rec['b64']);assert hashlib.sha256(val).hexdigest()==rec['sha256']\n    with dst.open('xb') as f:f.write(val)\n    os.chmod(dst,0o444);assert hashlib.sha256(dst.read_bytes()).hexdigest()==rec['sha256']\nfor role in ('reference','candidate'):(root/role/'tmp').mkdir(mode=0o700)\nreceipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'trn21','root':str(root),\n    'free_bytes':free,'projected_footprint_bytes':projected,'minimum_free_plus_footprint_bytes':10.5*1024**3+projected,\n    'available_ram_bytes':mem,'staged_hashes':{n:r['sha256'] for n,r in P['files'].items()},\n    'ps_rc':ps.returncode,'job_lines':jobs,'neuron_ls':devices,'fuser_rc':holders.returncode,'fuser_stdout':holders.stdout,'fuser_stderr':holders.stderr,'previous_pids_verified_absent':P['previous_pids'],\n    'previous_pgid_sid_descendants_absent':True,'local_state':P['local_state'],'lock_inode':lk.st_ino,'lock_device':lk.st_dev,\n    'resource_checks_under_original_exclusive_lock':True,'assignment':P['assignment_path'],\n    'scope':'Two serial full-size standard synthetic CPU G2 arms plus unchanged standard comparison only; no ordinary-loader/cold/native/full/eval/autonext',\n    'launch_guard':'Pinned driver reacquires same original lifetime lock and rechecks ps/neuron-ls before every child.'}\nwith (root/'preflight.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\\n')\nfcntl.flock(lockfd,fcntl.LOCK_UN);os.close(lockfd)\nwith (root/'dispatch.log').open('xb') as out:\n    child=subprocess.Popen(plan['detached_dispatch_command'],cwd=root,stdin=subprocess.DEVNULL,\n        stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)\nlaunch={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'dispatch_pid':child.pid,\n    'label':plan['label'],'status':'DETACHED_DISPATCH_STARTED_UNDER_LOCK_DRIVER_PENDING','no_automatic_next_job':True}\nwith (root/'operator-launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\\n')\nprint(json.dumps({'preflight':receipt,'launch':launch}))\n"

argv,*_=m.load_transport('trn21');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input='P='+repr(payload)+'\n'+body,text=True,capture_output=True,timeout=45)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode())
print(json.dumps({'host':'trn21','rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
