"""Explicit observed-lineage/device/lock status; never signals a process."""
import importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('local_one_use',HERE/'execute_once.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
con=m.load('connector',m.C/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py','7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899')
known={}
initial=json.loads((HERE/'first-observation.json').read_bytes())['remote']
for row in initial['processes']:
 if row.get('exists'):known[str(row['pid'])]=row['starttime']
events=HERE/'retrieved/files/lifetime-events.jsonl'
for line in events.read_text().splitlines():
 row=json.loads(line).get('identity')
 if row:known[str(row['pid'])]=row['start_ticks']
body=('KNOWN='+repr(known)+'\n').encode()+r'''
import datetime,fcntl,json,os,socket,stat,subprocess
from pathlib import Path
if socket.gethostname()!='ip-172-31-34-85':raise RuntimeError('HOST')
census=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  raw=(p/'stat').read_text();fields=raw.rsplit(')',1)[1].split()
  census.append({'pid':int(p.name),'comm':raw.split('(',1)[1].rsplit(')',1)[0],'ppid':int(fields[1]),'pgid':int(fields[2]),'sid':int(fields[3]),'starttime':int(fields[19]),'state':fields[0]})
 except (OSError,ValueError):pass
rows={str(r['pid']):r for r in census}
observed={pid:{'expected_starttime':tick,'exists':pid in rows,'same_identity':pid in rows and rows[pid]['starttime']==tick,'actual':rows.get(pid)} for pid,tick in KNOWN.items()}
children={}
for tid in Path('/proc/3202398/task').iterdir():
 try:children[tid.name]=(tid/'children').read_text()
 except OSError as e:children[tid.name]={'error':str(e)}
checks={}
for name,argv in [('ps',['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm']),('neuron_ls',['/opt/aws/neuron/bin/neuron-ls','--json-output']),('fuser',['/usr/bin/fuser','/dev/neuron0'])]:
 r=subprocess.run(argv,capture_output=True,text=True,timeout=20);checks[name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
lock='/home/ubuntu/.cache/trainium_freshstart/device0.lock';fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);s=os.fstat(fd)
if (s.st_dev,s.st_ino)!=(66305,596823):raise RuntimeError('LOCK_IDENTITY')
free=False
try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
except BlockingIOError:pass
os.close(fd)
root=Path('/home/ubuntu/trainium_confirmed36_clean_native_consistency_20260926_trn21_r1')
target=root/'caches/debug/neuronx-cc/fe9ba2da9e05617e76c02c31e3816f51/neuronxcc-wxrrwtkv/nc00/sg00/default_control_fast_table.bin'
try:
 t=target.lstat();bad={'path':str(target),'device':t.st_dev,'inode':t.st_ino,'mode':t.st_mode,'uid':t.st_uid,'bytes':t.st_size,'regular':stat.S_ISREG(t.st_mode),'symlink':stat.S_ISLNK(t.st_mode),'readlink':os.readlink(target) if stat.S_ISLNK(t.st_mode) else None,'target_contents_read':False}
except OSError as e:bad={'path':str(target),'error':str(e)}
v=os.statvfs('/')
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observed_identities':observed,'census':census,'guardian_thread_children':children,'checks':checks,'lock':{'device':s.st_dev,'inode':s.st_ino,'free':free},'same_session_rows':[r for r in census if r['sid']==3202396 or r['pgid']==3202396],'failing_cache_entry_metadata_only':bad,'free_disk_bytes':v.f_bavail*v.f_frsize,'no_signals':True,'no_host_writes':True,'host_reusable':False,'retired_queue_pid_exists':Path('/proc/2983496').exists()}))
'''.encode()
m.require(con.SENSITIVE.search(body) is None,'BODY_SENSITIVE')
argv,*_=con.load_transport('trn21');argv[-1]='exec /usr/bin/python3 -I -B -'
r=subprocess.run(argv,input=body,capture_output=True,timeout=45)
m.require(con.SENSITIVE.search(r.stdout+r.stderr) is None,'OUTPUT_SENSITIVE')
out={'utc':con.utc(),'source_sha256':m.sha(body),'returncode':r.returncode,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
m.save('explicit-postflight.json',out)
o=out['remote'] or {}
print(json.dumps({'utc':o.get('utc'),'returncode':r.returncode,'known_count':len(known),'same_live':[v['actual'] for v in o.get('observed_identities',{}).values() if v['same_identity']],'guardian_thread_children':o.get('guardian_thread_children'),'lock':o.get('lock'),'failing_cache_entry_metadata_only':o.get('failing_cache_entry_metadata_only'),'neuron_ls':o.get('checks',{}).get('neuron_ls'),'stderr':out['stderr']},indent=2))
