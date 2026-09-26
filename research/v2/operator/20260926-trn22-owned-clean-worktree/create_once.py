"""Exact one-use owned detached checkout; no packet stage or native launch."""
import datetime,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;C=Path('/home/poojith-devan/Desktop/Trainium')
ASSIGN='research/v2/verification/20260926-k2-trn22-clean-worktree-assignment.md'
PIN='440e1c649138eba18006a9082f4aaa43d2b11c9e149c45423c1923fab47e7865'
def sha(raw):return hashlib.sha256(raw).hexdigest()
if sha((C/ASSIGN).read_bytes())!=PIN:raise RuntimeError('ASSIGNMENT_PIN')
if subprocess.run(['git','show','4332cf4b:'+ASSIGN],cwd=C,capture_output=True,check=True).stdout!=(C/ASSIGN).read_bytes():raise RuntimeError('COMMITTED_ASSIGNMENT')
connector=C/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
if sha(connector.read_bytes())!='7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899':raise RuntimeError('CONNECTOR_PIN')
spec=importlib.util.spec_from_file_location('connector',connector);con=importlib.util.module_from_spec(spec);spec.loader.exec_module(con)
obs=con.local_state();state_raw=(C/'research/v2/state.json').read_bytes();state=json.loads(state_raw)
if obs['local_stops_present'] or obs['breaker_tripped']['global_correctness'] is not False or obs['breaker_tripped']['trn22'] is not False:raise RuntimeError('LOCAL_STOP_BREAKER')
controller={'utc':con.utc(),'state_sha256':sha(state_raw),'host_launch_allowed':state['hosts']['trn22']['launch_allowed'],'breakers':{'global_correctness':False,'trn22':False},'local_stops_present':[],'assignment_sha256':PIN,'preparation_only':True}
body=r'''
import datetime,fcntl,hashlib,json,os,socket,stat,subprocess,time,traceback
from pathlib import Path
PROJECT=Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1')
TARGET=Path('/home/ubuntu/trainium_k2_f0f2_trn22_evidence_20260926_r1')
COMMIT='65155c668efaa215a2136f67617b44ed32d4c9f6';LOCK='/home/ubuntu/.cache/trainium_freshstart/device0.lock'
FLOOR=16*1024**3;PRE=FLOOR+95301632+8*1024**2;RAM=24*1024**3
started=time.monotonic();fd=None;held=False;created=False
out={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'trn22','assignment_sha256':'440e1c649138eba18006a9082f4aaa43d2b11c9e149c45423c1923fab47e7865','commands':[],'creation_attempted':False,'packet_staged':False,'native_launched':False,'cleanup_performed':False}
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def meta(p):
 s=Path(p).lstat();return {'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def command(name,argv,timeout=15):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
 require(len(r.stdout.encode())+len(r.stderr.encode())<=2*1024**2,'COMMAND_OUTPUT_BOUND:'+name)
 row={'name':name,'argv':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};out['commands'].append(row);return r
GIT=['/usr/bin/git','--no-optional-locks','-c','core.fsmonitor=false','-c','core.hooksPath=/dev/null','-c','core.attributesFile=/dev/null','-c','core.autocrlf=false']
def git(name,path,args,timeout=15):return command(name,[*GIT,'-C',str(path),*args],timeout)
def resources():
 v=os.statvfs(TARGET.parent);mem=dict(l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines());return {'free_disk_bytes':v.f_bavail*v.f_frsize,'available_ram_bytes':int(mem['MemAvailable'].split()[0])*1024,'block_size':v.f_frsize}
def stops():
 bases={Path('/home/ubuntu'),PROJECT,TARGET}
 bases.update(p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir());require(len(bases)<512,'STOP_BOUND')
 present=[str(p/rel) for p in bases for rel in ('STOP','research/agentic/STOP','research/v2/STOP') if os.path.lexists(p/rel)]
 require(not present,'STOP:'+repr(present));return present
def idle(phase):
 p=command(phase+'-ps',['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args']);require(p.returncode==0,'PS_FAILED')
 busy=[l for l in p.stdout.splitlines()[1:] if any(x in l for x in ('torchrun','train.py','host_driver.py','host_queue.py','preflight_dryrun.py','compare_checkpoints.py','submission_rehearsal.py','neuronx-cc','walrus_driver','prepare.py eval-public'))]
 # Preserve matching workload argv only, not unrelated command arguments.
 out['commands'][-1]['stdout_sha256']=hashlib.sha256(p.stdout.encode()).hexdigest();out['commands'][-1]['stdout']='[non-workload argv omitted]';out[phase+'_job_rows']=busy;require(not busy,'BUSY_HOST')
 n=command(phase+'-neuron_ls',['/opt/aws/neuron/bin/neuron-ls','--json-output']);require(n.returncode==0,'NEURON_LS_FAILED');d=json.loads(n.stdout);require(isinstance(d,list) and len(d)==1 and d[0].get('neuron_processes')==[] and d[0].get('neuron_device')==0,'DEVICE_BUSY_OR_UNKNOWN')
 f=command(phase+'-fuser',['/usr/bin/fuser','/dev/neuron0']);require(f.returncode==1 and not f.stdout.strip() and not f.stderr.strip(),'DEVICE_HOLDER')
def original(phase):
 h=git(phase+'-head',PROJECT,['rev-parse','HEAD']);require(h.returncode==0 and h.stdout.strip()==COMMIT,'ORIGINAL_HEAD_DRIFT')
 s=git(phase+'-status',PROJECT,['status','--porcelain=v1','--untracked-files=all']);expected={f'?? research/run-logs/fs-20260924-v2-35c9-lr015-control-trn22-r1.{e}' for e in ('job.json','launch.json','log','postflight.json','preflight.json','process.json')};require(s.returncode==0 and set(s.stdout.splitlines())==expected,'ORIGINAL_SIX_LOG_STATUS_DRIFT');return {'head':h.stdout.strip(),'status':s.stdout}
try:
 require(socket.gethostname()=='ip-172-31-32-253' and os.getuid()==1000,'HOST_UID')
 fd=os.open(LOCK,os.O_RDONLY|os.O_NOFOLLOW);lk=os.fstat(fd);require(stat.S_ISREG(lk.st_mode) and (lk.st_dev,lk.st_ino)==(66305,832589),'ORIGINAL_LOCK')
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);held=True
 out['lock']={'device':lk.st_dev,'inode':lk.st_ino,'held_across_creation_and_postflight':True}
 idle('before');out['stops_before']=stops();require(not os.path.lexists(TARGET),'TARGET_EXISTS')
 out['target_parent_before']=meta(TARGET.parent)
 for parent in TARGET.parents:require(stat.S_ISDIR(parent.lstat().st_mode) and not parent.is_symlink(),'PARENT_SYMLINK_OR_TYPE')
 filters=git('configured-filter-names-only',PROJECT,['config','--name-only','--get-regexp',r'^filter\.']);require(filters.returncode in (0,1) and not filters.stdout.strip(),'CONFIGURED_CHECKOUT_FILTER')
 for key in ('core.sparseCheckout','core.sparseCheckoutCone'):
  sparse=git('config-'+key,PROJECT,['config','--bool','--get',key]);require((sparse.returncode==1 and not sparse.stdout.strip()) or (sparse.returncode==0 and sparse.stdout.strip()=='false'),'SPARSE_CHECKOUT_CONFIG')
 info=git('info-attributes-path',PROJECT,['rev-parse','--git-path','info/attributes']);require(info.returncode==0,'INFO_ATTRIBUTES_PATH');ip=Path(info.stdout.strip());ip=ip if ip.is_absolute() else PROJECT/ip
 require(not os.path.lexists(ip) or (not ip.is_symlink() and ip.is_file() and ip.stat().st_size==0),'LOCAL_INFO_ATTRIBUTES_PRESENT')
 out['original_before']=original('before')
 tree=git('fixed-tree-names',PROJECT,['ls-tree','-r','--name-only',COMMIT]);require(tree.returncode==0,'TREE_READ')
 names=tree.stdout.splitlines();require(len(names)==3798 and all(Path(n).name not in ('.gitattributes','.gitmodules') for n in names),'TREE_ATTRIBUTES_OR_COUNT')
 out['commands'][-1]['stdout_sha256']=hashlib.sha256(tree.stdout.encode()).hexdigest();out['commands'][-1]['stdout']='[3798 fixed tree names checked; raw listing preserved in prior read-only receipt]'
 out['resources_before']=resources();require(out['resources_before']['free_disk_bytes']>=PRE,'DISK_BEFORE_CREATION');require(out['resources_before']['available_ram_bytes']>=RAM,'RAM_BEFORE_CREATION')
 out['required_precreation_disk_bytes']=PRE;out['required_postcreation_disk_bytes']=FLOOR
 out['creation_attempted']=True;out['creation_started_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 result=git('create-one-detached-worktree',PROJECT,['worktree','add','--detach',str(TARGET),COMMIT],45)
 out['creation_ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();created=os.path.lexists(TARGET);out['created_path_exists']=created
 require(result.returncode==0,'WORKTREE_CREATION_FAILED:'+result.stderr)
 head=git('new-head',TARGET,['rev-parse','HEAD']);status=git('new-status',TARGET,['status','--porcelain=v1','--untracked-files=all']);symbolic=git('new-detached-state',TARGET,['symbolic-ref','-q','HEAD'])
 require(head.returncode==0 and head.stdout.strip()==COMMIT and status.returncode==0 and status.stdout=='' and symbolic.returncode==1,'NEW_CHECKOUT_NOT_EXACT_CLEAN_DETACHED')
 out['new_checkout']={'path':str(TARGET),'identity':meta(TARGET),'head':head.stdout.strip(),'status':status.stdout,'detached':True}
 entries=0;logical=0;allocated=0;files=0;dirs=0
 for directory,subdirs,filenames in os.walk(TARGET,followlinks=False):
  for p in [Path(directory),*(Path(directory)/n for n in filenames)]:
   require(not p.is_symlink(),'NEW_CHECKOUT_SYMLINK');s=p.lstat();entries+=1;require(entries<=5000,'FOOTPRINT_BOUND');allocated+=s.st_blocks*512
   if stat.S_ISREG(s.st_mode):files+=1;logical+=s.st_size
   elif stat.S_ISDIR(s.st_mode):dirs+=1
   else:raise RuntimeError('NEW_CHECKOUT_SPECIAL_FILE')
 out['new_checkout_footprint']={'regular_files':files,'directories':dirs,'regular_logical_bytes':logical,'allocated_bytes_files_and_directories':allocated,'git_administration_outside_target_not_included':True}
 out['original_after']=original('after');require(out['original_after']==out['original_before'],'ORIGINAL_CONTENT_STATUS_CHANGED')
 registry=git('worktree-registry-after',PROJECT,['worktree','list','--porcelain']);require(registry.returncode==0,'REGISTRY_POSTFLIGHT')
 out['resources_after']=resources();require(out['resources_after']['free_disk_bytes']>=FLOOR,'DISK_AFTER_CREATION');require(out['resources_after']['available_ram_bytes']>=RAM,'RAM_AFTER_CREATION')
 out['stops_after']=stops();idle('after');again=os.fstat(fd);require((again.st_dev,again.st_ino)==(66305,832589) and (Path(LOCK).stat().st_dev,Path(LOCK).stat().st_ino)==(66305,832589),'LOCK_POSTFLIGHT')
 out['status']='OWNED_CLEAN_DETACHED_SOURCE_PREPARATION_COMPLETE_NO_LAUNCH'
except BaseException as exc:
 out['status']='FAILED_OR_UNRESOLVED_PRESERVE_ALL';out['error']=type(exc).__name__+': '+str(exc);out['traceback']=traceback.format_exc();out['created_path_exists']=os.path.lexists(TARGET)
 try:out['resources_after_failure']=resources()
 except Exception as failure:out['resource_error']=repr(failure)
finally:
 out['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();out['elapsed_seconds']=time.monotonic()-started
 if held:fcntl.flock(fd,fcntl.LOCK_UN)
 if fd is not None:os.close(fd)
 print(json.dumps(out),flush=True)
raise SystemExit(0 if out['status']=='OWNED_CLEAN_DETACHED_SOURCE_PREPARATION_COMPLETE_NO_LAUNCH' else 2)
'''.encode()
if con.SENSITIVE.search(body):raise RuntimeError('SENSITIVE_PAYLOAD')
argv,*_=con.load_transport('trn22');argv[-1]='exec /usr/bin/python3 -I -B -S -'
started=con.utc();r=subprocess.run(argv,input=body,capture_output=True,timeout=90)
if con.SENSITIVE.search(r.stdout+r.stderr):raise RuntimeError('SENSITIVE_OUTPUT')
receipt={'started_utc':started,'ended_utc':con.utc(),'assignment_sha256':PIN,'controller':controller,'source_sha256':sha(body),'returncode':r.returncode,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.stdout else None}
dest=HERE/'trn22.json'
with dest.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
o=receipt['remote'] or {}
print(json.dumps({'utc':o.get('ended_utc'),'status':o.get('status'),'creation_attempted':o.get('creation_attempted'),'new_checkout':o.get('new_checkout'),'footprint':o.get('new_checkout_footprint'),'resources_before':o.get('resources_before'),'resources_after':o.get('resources_after'),'error':o.get('error'),'receipt_sha256':sha(dest.read_bytes())},indent=2))
raise SystemExit(r.returncode)
