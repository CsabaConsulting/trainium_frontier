"""Read-only executable/input binding observation; no imports, staging or launch."""
import datetime
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
PACKET=REPO/'research/v2/proposals/20260926-confirmed36-clean-native-trn21-packet-r1/packet'
assert not (HERE/'runtime-inspection.json').exists(),'ONE_USE_RECEIPT_EXISTS'
binding_raw=(PACKET/'NATIVE_BINDING.disabled.json').read_bytes()
assert hashlib.sha256(binding_raw).hexdigest()=='c29fe99533c21cae045a4cd7b5116f40808feda65bdab14e5ed38e1ab0b11760'
binding=json.loads(binding_raw)
input_raw=(PACKET/'INPUTS.json').read_bytes()
assert hashlib.sha256(input_raw).hexdigest()=='0cc0569936b0b1653bcc946edcdedd1bf91d3f671fabbc205a0a7b883bf9a15e'
inputs=json.loads(input_raw)['files'];assert len(inputs)==17 and len(binding['runtime_pins'])==8
support={}
for rel in ('proposals/20260926-confirmed36-clean-native-consistency/probe.py','proposals/20260926-confirmed36-clean-checkpoint-cpu/compare_checkpoints.py'):
    b=(PACKET/rel).read_bytes();digest=hashlib.sha256(b).hexdigest();assert digest==binding['support_pins'][rel]
    support[rel]={'sha256':digest,'bytes':len(b),'local_file':str(PACKET/rel),'proposed_remote_path':binding['root']+'/'+rel,'remote_staging_authorized':False}
state_raw=(REPO/'research/v2/state.json').read_bytes();state=json.loads(state_raw)
controller={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state_sha256':hashlib.sha256(state_raw).hexdigest(),'host_launch_allowed':state['hosts']['trn21']['launch_allowed'],'host_status':state['hosts']['trn21']['status'],'global_correctness_breaker':state['breakers']['global_correctness']['tripped'],'trn21_breaker':state['breakers']['trn21']['tripped'],'local_stops_present':[str(REPO/rel) for rel in ('STOP','research/agentic/STOP','research/v2/STOP') if (REPO/rel).exists() or (REPO/rel).is_symlink()],'observation_only_not_a_release':True}
payload={'root':binding['root'],'label':binding['label'],'hostname':binding['expected_hostname'],'project':binding['environment']['PYTHONPATH'],'runtime_paths':list(binding['runtime_pins']),'inputs':inputs,'proposed_helpers':support,'resources':binding['resources']}
body=r'''
import datetime,fcntl,hashlib,json,os,socket,stat,subprocess,time,traceback
from pathlib import Path
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def metadata(s):return {'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,'gid':s.st_gid,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
started=time.monotonic();deadline=started+45;read_total=0;fd=None;held=False
out={'started_utc':utc(),'host':'trn21','hostname':socket.gethostname(),'label':P['label'],'runtime_pins':{},'runtime_identity_evidence':{},'retained_inputs':{},'no_host_files_changed':True,'native_imports':False,'helper_invoked':False,'staging_or_launch':False}
def file_record(path):
 global read_total
 p=Path(path)
 if not p.is_absolute() or '..' in p.parts:raise RuntimeError('NONEXACT_PATH')
 if any(q.is_symlink() for q in (p,*p.parents)):raise RuntimeError('SYMLINK_INPUT:'+str(p))
 before=p.lstat()
 if not stat.S_ISREG(before.st_mode) or before.st_size>1024**3:raise RuntimeError('FILE_TYPE_OR_BOUND')
 f=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  a=os.fstat(f);digest=hashlib.sha256();count=0
  while True:
   if time.monotonic()>deadline:raise RuntimeError('TIME_BOUND')
   b=os.read(f,65536)
   if not b:break
   count+=len(b);read_total+=len(b)
   if read_total>2*1024**3:raise RuntimeError('TOTAL_READ_BOUND')
   digest.update(b)
  z=os.fstat(f)
 finally:os.close(f)
 after=p.lstat()
 if not metadata(before)==metadata(a)==metadata(z)==metadata(after) or count!=before.st_size:raise RuntimeError('FILE_DRIFT:'+str(p))
 return {'sha256':digest.hexdigest(),'bytes':count},metadata(before)
try:
 if socket.gethostname()!=P['hostname'] or os.getuid()!=1000:raise RuntimeError('WRONG_HOST_OR_UID')
 lock='/home/ubuntu/.cache/trainium_freshstart/device0.lock'
 fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);lk=os.fstat(fd)
 if (lk.st_dev,lk.st_ino)!=(66305,596823) or not stat.S_ISREG(lk.st_mode):raise RuntimeError('ORIGINAL_LOCK_MISMATCH')
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);held=True
 out['original_lock']={'device':lk.st_dev,'inode':lk.st_ino,'exclusive_held_during_observation':True}
 ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=8)
 if ps.returncode:raise RuntimeError('PS_FAILED:'+ps.stderr)
 busy=[l for l in ps.stdout.splitlines()[1:] if any(k in l for k in ('torchrun','train.py','host_driver.py','host_queue.py','preflight_dryrun.py','compare_checkpoints.py','submission_rehearsal.py','neuronx-cc','walrus_driver','prepare.py eval-public'))]
 out['ps']={'returncode':ps.returncode,'row_count':len(ps.stdout.splitlines())-1,'busy':busy,'stdout_sha256':hashlib.sha256(ps.stdout.encode()).hexdigest(),'stderr':ps.stderr}
 if busy:raise RuntimeError('BUSY_HOST')
 ls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=12)
 if ls.returncode:raise RuntimeError('NEURON_LS_FAILED:'+ls.stderr)
 devices=json.loads(ls.stdout);out['neuron_ls']={'returncode':ls.returncode,'devices':devices,'stderr':ls.stderr}
 if not (isinstance(devices,list) and len(devices)==1 and devices[0].get('neuron_device')==0 and devices[0].get('neuron_processes')==[]):raise RuntimeError('DEVICE_BUSY_OR_UNKNOWN')
 fu=subprocess.run(['/usr/bin/fuser','/dev/neuron0'],capture_output=True,text=True,timeout=8)
 out['fuser']={'returncode':fu.returncode,'stdout':fu.stdout,'stderr':fu.stderr}
 if fu.returncode!=1 or fu.stdout.strip() or fu.stderr.strip():raise RuntimeError('DEVICE_HOLDER_OR_FUSER_ERROR')
 project=Path(P['project']);root=Path(P['root'])
 out['proposed_root']={'path':str(root),'lexists':os.path.lexists(root),'parent_identity':metadata(root.parent.stat())}
 out['proposed_helpers']={rel:{'path':v['proposed_remote_path'],'lexists':os.path.lexists(v['proposed_remote_path']),'expected_local_sha256':v['sha256'],'not_staged':not os.path.lexists(v['proposed_remote_path'])} for rel,v in P['proposed_helpers'].items()}
 if os.path.lexists(root):raise RuntimeError('PROPOSED_ROOT_EXISTS')
 if project.is_symlink() or not project.is_dir():raise RuntimeError('PROJECT_TYPE')
 git=[]
 for args in (['rev-parse','HEAD'],['status','--porcelain=v1','--untracked-files=normal']):
  r=subprocess.run(['/usr/bin/git','--no-optional-locks','-c','core.fsmonitor=false','-C',str(project),*args],capture_output=True,text=True,timeout=8)
  if r.returncode:raise RuntimeError('GIT_READ_FAILED:'+r.stderr)
  git.append({'args':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 out['project']={'path':str(project),'identity':metadata(project.stat()),'git':git}
 bases={Path('/home/ubuntu'),project,root}
 for p in Path('/home/ubuntu').glob('trainium*'):
  if p.is_dir():bases.add(p)
 if len(bases)>512:raise RuntimeError('STOP_BASE_BOUND')
 stop_paths=[str(p/rel) for p in sorted(bases) for rel in ('STOP','research/agentic/STOP','research/v2/STOP')]
 out['stop_check']={'paths':stop_paths,'present':[p for p in stop_paths if os.path.lexists(p)]}
 if out['stop_check']['present']:raise RuntimeError('REMOTE_STOP_PRESENT')
 for path in P['runtime_paths']:
  lexical=Path(path);before=lexical.lstat();resolved=lexical.resolve(strict=True)
  link=os.readlink(lexical) if lexical.is_symlink() else None
  pin,ident=file_record(resolved)
  if metadata(before)!=metadata(lexical.lstat()) or lexical.resolve(strict=True)!=resolved:raise RuntimeError('RUNTIME_LINK_DRIFT')
  out['runtime_pins'][path]={'resolved_path':str(resolved),'link_target':link,**pin}
  out['runtime_identity_evidence'][path]={'lexical_identity':metadata(before),'resolved_identity':ident,'stable_during_read':True}
 for path,pin in P['inputs'].items():
  actual,ident=file_record(path)
  out['retained_inputs'][path]={'pin':actual,'identity':ident,'matches_expected':actual==pin}
  if actual!=pin:raise RuntimeError('RETAINED_INPUT_MISMATCH:'+path)
 v=os.statvfs(root.parent)
 mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
 disk=v.f_bavail*v.f_frsize;available=int(mem['MemAvailable'].split()[0])*1024
 out['resources']={'free_disk_bytes':disk,'available_ram_bytes':available,'total_ram_bytes':int(mem['MemTotal'].split()[0])*1024,'disk_floor_bytes':P['resources']['initial_disk_bytes'],'ram_floor_bytes':P['resources']['initial_ram_bytes'],'disk_floor_met':disk>=P['resources']['initial_disk_bytes'],'ram_floor_met':available>=P['resources']['initial_ram_bytes'],'disk_headroom_bytes':disk-P['resources']['initial_disk_bytes']}
 out['native_package_version_inspection']='NOT_PERFORMED: only eight explicitly assigned executable identities; no SDK imports.'
 out['status']='READONLY_IDENTITIES_CAPTURED_NOT_LAUNCH_AUTHORITY'
except BaseException as exc:
 out['status']='REFUSED_OR_INCOMPLETE';out['error_type']=type(exc).__name__;out['error']=str(exc);out['traceback']=traceback.format_exc()
finally:
 out['ended_utc']=utc();out['read_bytes']=read_total;out['elapsed_seconds']=time.monotonic()-started
 if held:fcntl.flock(fd,fcntl.LOCK_UN)
 if fd is not None:os.close(fd)
 print(json.dumps(out),flush=True)
raise SystemExit(0 if out['status']=='READONLY_IDENTITIES_CAPTURED_NOT_LAUNCH_AUTHORITY' else 2)
'''
wire=('P='+repr(payload)+'\n'+body).encode()
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert not m.SENSITIVE.search(wire)
argv,*_=m.load_transport('trn21');argv[-1]='exec python3 -I -B -'
r=subprocess.run(argv,input=wire,capture_output=True,timeout=60)
assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'payload_sha256':hashlib.sha256(wire).hexdigest(),'binding_path':str(PACKET/'NATIVE_BINDING.disabled.json'),'binding_sha256':hashlib.sha256(binding_raw).hexdigest(),'inputs_sha256':hashlib.sha256(input_raw).hexdigest(),'local_support':support,'controller_observation':controller,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.stdout else None}
with (HERE/'runtime-inspection.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
o=receipt['remote']
print(json.dumps({'returncode':r.returncode,'status':o.get('status'),'utc':o.get('ended_utc'),'runtime_pins':o.get('runtime_pins'),'retained_input_count':len(o.get('retained_inputs',{})),'resources':o.get('resources'),'root':o.get('proposed_root'),'stops_present':o.get('stop_check',{}).get('present'),'controller':controller,'error':o.get('error'),'receipt_sha256':hashlib.sha256((HERE/'runtime-inspection.json').read_bytes()).hexdigest()},indent=2))
raise SystemExit(r.returncode)
