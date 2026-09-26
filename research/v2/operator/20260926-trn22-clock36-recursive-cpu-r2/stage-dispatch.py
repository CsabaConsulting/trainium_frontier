"""Exact-byte transport for one separately released CPU packet; no science edits."""
import base64, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path

repo = Path('/home/poojith-devan/Desktop/Trainium')
own = Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
packet_rel = 'research/v2/proposals/20260926-clock36-recursive-exact-cpu-r2-trn22'
packet = repo / packet_rel
frozen = 'e3e5f149f9cbff58cdbf91186a104e727b5696e8'
assert len(sys.argv) == 3, 'EXACT_RELEASE_COMMIT_AND_PATH_REQUIRED'
release_commit, release_path = sys.argv[1:]
assert release_path.startswith('research/v2/verification/') and release_path.endswith('.md')
assignment = (repo / release_path).read_bytes()
assert assignment == subprocess.check_output(['git', 'show', release_commit + ':' + release_path], cwd=repo)
pins = {
    'CPU_PLAN.disabled.json': '1c23d1a8cd28ec7ca2fa3ffea6502045090c03d1d87a0c037944f0e9150d79db',
    'host_driver.py': '441a9833b3fe9c81dcbef0d2fff88333384b662660e2fc225edd6bcf571c8e5a',
    'MANIFEST.json': '3a0253db33b71bb126a65be0b39076f7b1f22fd7bf69f07b8a18ab68ceef9e13',
}
for name, pin in pins.items():
    assert hashlib.sha256((packet / name).read_bytes()).hexdigest() == pin, name
plan = json.loads((packet / 'CPU_PLAN.disabled.json').read_bytes())
manifest = json.loads((packet / 'MANIFEST.json').read_bytes())
assert plan['stage_root'] == '/home/ubuntu/trainium_clock36_recursive_cpu_20260926_trn22_r2'
assert plan['driver_sha256'] == pins['host_driver.py']
assert plan['source_candidate_sha256'] == '4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea'
assert len(manifest['files']) == 24
files = {}
for rec in manifest['files']:
    name = rec['path']
    val = subprocess.check_output(['git', 'show', frozen + ':' + packet_rel + '/' + name], cwd=repo)
    assert (packet / name).read_bytes() == val and hashlib.sha256(val).hexdigest() == rec['sha256'], name
    if name in ('CPU_PLAN.disabled.json', 'EXPERIMENT_PROPOSAL.json', 'README.md', 'host_driver.py',
                'validate_receipts.py', 'compare_checkpoints.py') or name.endswith('/request.json'):
        files[name] = val
files['MANIFEST.json'] = (packet / 'MANIFEST.json').read_bytes()
files['SUPERVISOR_ASSIGNMENT.md'] = assignment
for role in ('reference', 'default', 'off', 'on', 'g2'):
    spec = plan['sources'][role]
    val = subprocess.check_output(['git', 'show', spec['commit'] + ':' + spec['git_path']], cwd=repo)
    assert hashlib.sha256(val).hexdigest() == spec['sha256'], role
    relative = str(Path(spec['stage_path']).relative_to(plan['stage_root']))
    files[relative] = val
for role, name in (('idle_guard', 'idle_guard.py'), ('dispatch', 'dispatch_driver.py')):
    spec = plan['sources'][role]
    val = (repo / spec['git_path']).read_bytes()
    assert hashlib.sha256(val).hexdigest() == spec['sha256'], role
    files[name] = val

archive = own / 'research/v2/operator/20260926-trn22-k2-registered/completed'
for name in ('results-bundle.json', 'postflight-explicit-pids.json'):
    p = archive / name
    assert p.read_bytes() == subprocess.check_output(['git', 'show',
        '30630c8441249f7db894a80c19200f67003bd18d:' + str(p.relative_to(own))], cwd=own), name
post = json.loads((archive / 'postflight-explicit-pids.json').read_bytes())
assert post['all_observed_pids_absent'] and post['related_rows'] == [] and post['neuron_device_idle']
assert post['lock_competitor_rc'] == 0 and (post['lock_device'], post['lock_inode']) == (66305, 832589)
oldpids = post['observed_pids']
assert set(oldpids) == {3244018,3244019,3244020,3244034,3244035,3244039,3244040,3244041,3244042,3244718,3244724}
spec = importlib.util.spec_from_file_location('transport', repo / 'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
state = m.local_state()
assert not state['local_stops_present'] and not any(state['breaker_tripped'].values()), state
assert state['host_launch_allowed']['trn22'] is True, 'HOST_NOT_RELEASED'
hs = json.loads((repo / 'research/v2/state.json').read_bytes())['hosts']['trn22']
assert hs['active_assignment'] == release_path, 'STATE_SCOPE_MISMATCH'
assert hs['label'] == 'fs-20260926-clock36-recursive-trn22-g2-r2', 'STATE_LABEL_MISMATCH'
payload = {'plan': plan, 'files': {n: {'b64': base64.b64encode(v).decode(),
            'sha256': hashlib.sha256(v).hexdigest()} for n,v in files.items()},
           'local_state': state, 'previous_pids': oldpids, 'assignment_path': release_path}
body = r'''
import base64,datetime,fcntl,hashlib,importlib.util,json,os,socket,subprocess
from pathlib import Path
plan=P['plan'];root=Path(plan['stage_root']);project=Path(plan['prepare_dir'])
assert socket.gethostname()==plan['expected_hostname'] and os.getuid()==1000,'HOST_IDENTITY'
assert not os.path.lexists(root),'LABEL_ALREADY_CONSUMED'
previous=project/'out/fs-20260926-k2-registered-forward-trn22-r1-packet'
assert (previous/'terminal.json').is_file() and (previous/'outer-flock-exit.json').is_file(),'PRIOR_NOT_TERMINAL'
assert json.loads((previous/'terminal.json').read_text())['status']=='DIAGNOSTIC_CHILD_FAILED','PRIOR_STATUS_CHANGED'
for pid in (*P['previous_pids'],3050509,3053905):
    assert not Path('/proc/'+str(pid)).exists(),'PRIOR_PID_LIVE:'+str(pid)
lock=Path('/home/ubuntu/.cache/trainium_freshstart/device0.lock')
lockfd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
lk=os.fstat(lockfd);assert (lk.st_dev,lk.st_ino)==(66305,832589),'LOCK_IDENTITY_CHANGED'
fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
for base in (Path('/home/ubuntu'),project,root,*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
    for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):
        assert not os.path.lexists(base/rel),'STOP:'+str(base/rel)
v=os.statvfs('/');free=v.f_bavail*v.f_frsize
# Four retained ~453MB checkpoints plus a conservative log/support/scratch allowance.
projected=4*512*1024**2+1024**3
assert free>=10.5*1024**3+projected,'DISK_FLOOR_PLUS_PROJECTED_FOOTPRINT'
mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
assert mem>=64*1024**3,'AVAILABLE_RAM_FLOOR'
assert hashlib.sha256((project/'prepare.py').read_bytes()).hexdigest()==plan['sources']['prepare']['sha256'],'PREPARE_SHA'
pause=Path('/home/ubuntu/trainium_finalpush_queue_20260925_trn22_wait_r1/runs/PAUSE_NEXT');assert pause.is_file(),'OLD_QUEUE_PAUSE_MISSING'
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=10)
assert ps.returncode==0,ps.stderr
jobs=[l for l in ps.stdout.splitlines() if any(k in l for k in ('neuron-explorer','torchrun','train.py','host_queue.py','host_driver.py','submission_rehearsal.py','neuronx-cc','walrus_driver','preflight_dryrun.py','prepare.py eval-public','run_job.py','contained.py','rank_entry.py','check_native.py','compare_checkpoints.py'))]
assert jobs==[],'BUSY_PS:'+str(jobs)
rows=[l.split(None,5) for l in ps.stdout.splitlines()[1:]]
assert not any(any(int(r[i]) in P['previous_pids'] for i in (1,2,3)) for r in rows),'PRIOR_DESCENDANT_PGID_SID_LIVE'
ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=20)
assert ls.returncode==0,ls.stderr
devices=json.loads(ls.stdout);assert len(devices)==1 and devices[0]['neuron_processes']==[],'BUSY_DEVICE'
assert Path('/home/ubuntu/workspace/native_venv/bin/python').is_file()
root.mkdir(mode=0o700)
for name,rec in P['files'].items():
    dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True)
    assert not dst.is_symlink() and not dst.exists()
    val=base64.b64decode(rec['b64']);assert hashlib.sha256(val).hexdigest()==rec['sha256']
    with dst.open('xb') as f:f.write(val)
    os.chmod(dst,0o444);assert hashlib.sha256(dst.read_bytes()).hexdigest()==rec['sha256']
for role in ('reference','default','off','on'):(root/role/'tmp').mkdir(mode=0o700)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'trn22','root':str(root),
    'free_bytes':free,'projected_footprint_bytes':projected,'minimum_free_plus_footprint_bytes':10.5*1024**3+projected,
    'available_ram_bytes':mem,'staged_hashes':{n:r['sha256'] for n,r in P['files'].items()},
    'ps_rc':ps.returncode,'job_lines':jobs,'neuron_ls':devices,'previous_pids_verified_absent':P['previous_pids'],
    'previous_pgid_sid_descendants_absent':True,'local_state':P['local_state'],'lock_inode':lk.st_ino,'lock_device':lk.st_dev,
    'resource_checks_under_original_exclusive_lock':True,'assignment':P['assignment_path'],
    'scope':'Four serial full-size synthetic CPU arms and checkpoint tests only; no cold/native/full/eval/autonext',
    'launch_guard':'Pinned driver reacquires same original lifetime lock and rechecks ps/neuron-ls before every child.'}
with (root/'preflight.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
fcntl.flock(lockfd,fcntl.LOCK_UN);os.close(lockfd)
with (root/'dispatch.log').open('xb') as out:
    child=subprocess.Popen(plan['detached_dispatch_command'],cwd=root,stdin=subprocess.DEVNULL,
        stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
launch={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'dispatch_pid':child.pid,
    'status':'DETACHED_DISPATCH_STARTED_UNDER_LOCK_DRIVER_PENDING','no_automatic_next_job':True}
with (root/'operator-launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n')
print(json.dumps({'preflight':receipt,'launch':launch}))
'''
argv,*_=m.load_transport('trn22');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input='P='+repr(payload)+'\n'+body,text=True,capture_output=True,timeout=45)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode())
print(json.dumps({'host':'trn22','rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
