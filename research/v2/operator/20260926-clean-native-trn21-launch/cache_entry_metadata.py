"""One exact cache symlink and literal SDK-target lstat; no bytes or recursion."""
import importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('local_one_use',HERE/'execute_once.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
con=m.load('connector',m.C/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py','7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899')
wire=r'''
import datetime,json,os,socket,stat
from pathlib import Path
if socket.gethostname()!='ip-172-31-34-85':raise RuntimeError('HOST')
link=Path('/home/ubuntu/trainium_confirmed36_clean_native_consistency_20260926_trn21_r1/caches/debug/neuronx-cc/fe9ba2da9e05617e76c02c31e3816f51/neuronxcc-wxrrwtkv/nc00/sg00/default_control_fast_table.bin')
target=Path('/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages/neuronxcc/dve/dve_bin_gen3/default_control_fast_table.bin')
def record(p):
 s=p.lstat()
 return {'path':str(p),'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,'gid':s.st_gid,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'regular':stat.S_ISREG(s.st_mode),'symlink':stat.S_ISLNK(s.st_mode)}
before=record(link)
if not before['symlink'] or os.readlink(link)!=str(target):raise RuntimeError('EXACT_LINK_DRIFT')
to=record(target);after=record(link)
if before!=after:raise RuntimeError('LINK_METADATA_DRIFT')
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'link':before,'literal_target':to,'link_unchanged':True,'target_contents_read':False,'no_recursive_resolution':True,'no_host_writes':True}))
'''.encode()
argv,*_=con.load_transport('trn21');argv[-1]='exec /usr/bin/python3 -I -B -'
r=subprocess.run(argv,input=wire,capture_output=True,timeout=20)
m.require(con.SENSITIVE.search(r.stdout+r.stderr) is None,'OUTPUT_SENSITIVE')
result={'utc':con.utc(),'returncode':r.returncode,'source_sha256':m.sha(wire),'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if not r.returncode else None}
m.save('cache-entry-metadata.json',result);print(json.dumps(result,indent=2))
