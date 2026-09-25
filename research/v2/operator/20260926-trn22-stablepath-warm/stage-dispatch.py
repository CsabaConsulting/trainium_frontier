"""Exact assigned W1 byte staging; only after full36 plus eval archive/postflight.

Prepared locally. Execution consumes the unique remote label. Never retry it.
No writes to the old stable source, retained caches, evaluator or prior outputs.
"""
import base64
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

REPO = Path('/home/poojith-devan/Desktop/Trainium')
OWN = Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
PACKET = '52c31b73bbd5d76de35f32f55a66b6125aa8effb'
DIR = 'research/v2/proposals/20260926-stablepath-warm-observation-trn22/'
ASSIGNMENT = 'research/v2/verification/20260926-stablepath-warm-trn22-assignment.md'

def blob(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=REPO)

def sha(data):
    return hashlib.sha256(data).hexdigest()

binding_raw = blob(PACKET, DIR + 'WARM_BINDING.disabled.json')
assert sha(binding_raw) == 'efbaf85470bd1ac51c0e791fe6bc6794d03510551205c62bf7ab1cc338e08796'
binding = json.loads(binding_raw)
manifest_raw = blob(PACKET, DIR + 'MANIFEST.sha256.json')
assert sha(manifest_raw) == 'c0a19be136e1c7c1d83451290cd4fdbba92d04438d1b0e963cddfd7177dbe331'
manifest = json.loads(manifest_raw)
for name, digest in manifest['files'].items():
    assert sha(blob(PACKET, DIR + name)) == digest, name
files = {name: blob(PACKET, DIR + name) for name in (
    'host_driver.py', 'identity_checks.py', 'S1-pre-source-identity.json',
    'EXPERIMENT_PROPOSAL.json', 'WARM_BINDING.disabled.json', 'MANIFEST.sha256.json',
    'w1/package/launch-command.txt', 'w1/package/provenance.json')}
files['w1/package/train.py'] = blob(binding['source_commit'], binding['source_git_path'])
assert sha(files['w1/package/train.py']) == binding['source_sha256']
files['SUPERVISOR_ASSIGNMENT.md'] = blob('b7f56465', ASSIGNMENT)
assert (REPO / ASSIGNMENT).read_bytes() == files['SUPERVISOR_ASSIGNMENT.md']
for name, key in [('idle_guard.py', 'idle_guard_sha256'), ('dispatch_driver.py', 'dispatch_driver_sha256')]:
    raw = (OWN / 'research/v2/operator/20260926-common-host-idle-guard' / name).read_bytes()
    assert sha(raw) == binding[key], name
    files[name] = raw
prior_path = OWN / 'research/v2/operator/20260926-trn22-startup-clock-full/completed/explicit-process-postflight.json'
prior = json.loads(prior_path.read_text())
assert prior['all_observed_pids_absent'] and not prior['related_rows']
assert prior['neuron_device_idle'] and prior['lock_competitor_rc'] == 0
spec = importlib.util.spec_from_file_location('transport', REPO / 'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
state = transport.local_state()
assert not state['local_stops_present'] and not any(state['breaker_tripped'].values())
assert state['host_launch_allowed']['trn22'] is True
payload = {'binding': binding, 'files': {name: {'b64': base64.b64encode(raw).decode(), 'sha256': sha(raw)} for name, raw in files.items()}, 'local_state': state, 'prior_postflight': prior}
remote = r'''
import base64, datetime, hashlib, json, os, socket, stat, subprocess
from pathlib import Path
b=P['binding']; root=Path(b['root']); project=Path(b['prepare_path']).parent
assert socket.gethostname()==b['expected_hostname']=='ip-172-31-32-253' and os.getuid()==1000,'HOST_IDENTITY'
assert not os.path.lexists(root),'LABEL_ALREADY_CONSUMED'
previous=project/'out/fs-20260926-arch36a8-startup-clock-trn22-r1-full-packet'
terminal=json.loads((previous/'terminal.json').read_text())
outer=json.loads((previous/'outer-flock-exit.json').read_text())
assert terminal['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and terminal['eval_exit_code']==0,'FULL36_NOT_TERMINAL'
assert outer['returncode']==0,'FULL36_DRIVER_NOT_EXITED'
prior=set(P['prior_postflight']['observed_pids'])
for pid in prior|{3050509,3053905}: assert not Path('/proc/'+str(pid)).exists(),'PRIOR_PID_LIVE:'+str(pid)
lock=Path(b['device_lock']).stat(); assert (lock.st_dev,lock.st_ino)==(66305,832589),'LOCK_IDENTITY'
for base in (Path('/home/ubuntu'),project,root,Path(b['stable_directory']),*[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
 for rel in ('STOP','research/agentic/STOP','research/v2/STOP'): assert not os.path.lexists(base/rel),'STOP:'+str(base/rel)
v=os.statvfs('/'); free=v.f_bavail*v.f_frsize; assert free>=6.5*1024**3,'DISK_BELOW_6_5_GIB'
pause=Path('/home/ubuntu/trainium_finalpush_queue_20260925_trn22_wait_r1/runs/PAUSE_NEXT'); assert pause.is_file(),'OLD_QUEUE_PAUSE_MISSING'
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=10)
assert ps.returncode==0 and not ps.stderr,'PS_FAILED:'+ps.stderr
related=[]
for line in ps.stdout.splitlines()[1:]:
 row=line.split(None,5); assert len(row)==6
 if any(int(x) in prior for x in row[:4]) or any(k in line for k in ('torchrun','train.py','host_queue.py','host_driver.py','submission_rehearsal.py','neuronx-cc','walrus_driver','profile_actual_loop.py','preflight_dryrun.py','prepare.py eval-public','run_job.py','contained.py')): related.append(line)
assert not related,'BUSY_PS:'+str(related)
nls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=15)
assert nls.returncode==0 and not nls.stderr,'NLS_FAILED:'+nls.stderr
devices=json.loads(nls.stdout); assert len(devices)==1 and devices[0]['neuron_processes']==[],'BUSY_DEVICE'
anchor=json.loads(base64.b64decode(P['files']['S1-pre-source-identity.json']['b64']))
fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode','st_uid','st_gid','st_nlink')
for key in ('executed_train','expected_imported_prepare','protected_prepare','harness'):
 a=anchor[key]; p=Path(a['path'])
 for q in (p,*p.parents): assert not q.is_symlink(),'SYMLINK:'+str(q)
 before=p.stat(); assert stat.S_ISREG(before.st_mode) and before.st_nlink==1
 with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 after=p.stat(); assert digest==a['sha256'] and all(getattr(before,k)==getattr(after,k)==a[k] for k in fields),'OLD_IDENTITY_CHANGED:'+key
stable=Path(b['stable_directory']); st=stable.stat()
assert all(getattr(st,k)==v for k,v in anchor['stable_directory'].items()),'OLD_STABLE_DIRECTORY_CHANGED'
assert sorted(p.name for p in stable.iterdir())==['prepare.py','train.py'],'OLD_STABLE_CONTENTS'
for value in b['cache_environment'].values():
 p=Path(value)
 for q in (p,*p.parents): assert not q.is_symlink(),'CACHE_SYMLINK:'+str(q)
 assert p.is_dir(),'RETAINED_CACHE_MISSING'
root.mkdir(mode=0o700)
for name,entry in P['files'].items():
 p=root/name; p.parent.mkdir(parents=True,exist_ok=True); assert not os.path.lexists(p)
 raw=base64.b64decode(entry['b64']); assert hashlib.sha256(raw).hexdigest()==entry['sha256']
 with p.open('xb') as stream: stream.write(raw)
 os.chmod(p,0o444 if name.startswith('w1/package/') else 0o600)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
Path(b['phases'][0]['environment']['TORCH_NEURONX_METRICS_DIR']).mkdir(mode=0o700)
assert not os.path.lexists(b['phases'][0]['sandbox'])
preflight={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['phases'][0]['label'],'host':'trn22','free_bytes':free,'staged_hashes':{n:e['sha256'] for n,e in P['files'].items()},'local_state':P['local_state'],'prior_full_terminal':terminal,'prior_full_outer':outer,'prior_postflight':P['prior_postflight'],'related_rows':related,'neuron_ls':devices,'old_queue_pids_absent':[3050509,3053905],'pause_sha256':hashlib.sha256(pause.read_bytes()).hexdigest(),'lock_device':lock.st_dev,'lock_inode':lock.st_ino,'old_source_and_caches_not_changed':True,'same_guard_in_lock_checks_required':True}
(root/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
with (root/'dispatch.log').open('xb') as log:
 process=subprocess.Popen(b['detached_dispatch_command'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
launch={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':b['phases'][0]['label'],'dispatch_pid':process.pid,'root':str(root),'status':'DETACHED_DISPATCH_STARTED_UNDER_LOCK_DRIVER_PENDING','no_automatic_retry':True,'no_automatic_next_job':True}
(root/'operator-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps({'preflight':preflight,'launch':launch}))
'''
argv, *_ = transport.load_transport('trn22')
argv[-1] = 'exec python3 -I -B -'
result = subprocess.run(argv, input='P='+repr(payload)+'\n'+remote, text=True, capture_output=True, timeout=45)
assert not transport.SENSITIVE.search((result.stdout+result.stderr).encode())
print(json.dumps({'host':'trn22','rc':result.returncode,'stdout':result.stdout,'stderr':result.stderr}))
