"""Fresh read-only exact guardian lineage/device evidence; no signals."""
import ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20260926-clean-native-trn21-launch'
old=(PRIOR/'postflight_once.py').read_bytes()
if hashlib.sha256(old).hexdigest()!='267e21e5c5fd378cc3938a109ed2f67ae26ee3de009fe0a60a58e56552d0655d':raise RuntimeError('PRIOR_POSTFLIGHT_PIN')
body=next(n.value for n in ast.walk(ast.parse(old)) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('\nimport datetime,fcntl,json,os,socket,stat,subprocess'))
known={}
initial=json.loads((PRIOR/'first-observation.json').read_bytes())['remote']
for row in initial['processes']:
 if row.get('exists'):known[str(row['pid'])]=row['starttime']
for line in (PRIOR/'retrieved/files/lifetime-events.jsonl').read_text().splitlines():
 row=json.loads(line).get('identity')
 if row:known[str(row['pid'])]=row['start_ticks']
if len(known)!=43 or {k:known[k] for k in ('3202396','3202397','3202398')}!={'3202396':94436551,'3202397':94436553,'3202398':94436554}:raise RuntimeError('EXACT_PRIOR_IDENTITY_SET')
start=body.index("root=Path('/home/ubuntu/trainium_confirmed36")
end=body.index("v=os.statvfs('/')",start)
body=body[:start]+body[end:]
body=body.replace("'failing_cache_entry_metadata_only':bad,",'')
body=body.replace("for tid in Path('/proc/3202398/task').iterdir():","for tid in (Path('/proc/3202398/task').iterdir() if Path('/proc/3202398/task').exists() else []):")
anchor="v=os.statvfs('/')"
extra=r'''
workload=[];unreadable=[]
for row in census:
 try:
  cmd=Path('/proc',str(row['pid']),'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if any(k in cmd for k in ('torchrun','train.py','host_driver.py','host_queue.py','preflight_dryrun.py','compare_checkpoints.py','submission_rehearsal.py','neuronx-cc','walrus_driver','prepare.py eval-public','probe.py')):workload.append({**row,'cmdline':cmd})
 except FileNotFoundError:pass
 except OSError as e:unreadable.append({'pid':row['pid'],'error':str(e)})
lock_fds=[]
if observed['3202398']['same_identity']:
 for p in Path('/proc/3202398/fd').iterdir():
  try:
   target=os.readlink(p)
   if target!=lock:continue
   st=os.stat(p);info=Path('/proc/3202398/fdinfo',p.name).read_text()
   if len(info)>4096:raise RuntimeError('FDINFO_BOUND')
   lock_fds.append({'fd':int(p.name),'target':target,'device':st.st_dev,'inode':st.st_ino,'fdinfo':info})
  except FileNotFoundError:pass
'''
body=body.replace(anchor,extra+'\n'+anchor)
body=body.replace("'guardian_thread_children':children,","'guardian_thread_children':children,'guardian_original_lock_fds':lock_fds,'workload_rows':workload,'unreadable_cmdlines':unreadable,")
ast.parse(body);wire=('KNOWN='+repr(known)+'\n'+body).encode()
C=Path('/home/poojith-devan/Desktop/Trainium');connector=C/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
if hashlib.sha256(connector.read_bytes()).hexdigest()!='7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899':raise RuntimeError('CONNECTOR_PIN')
spec=importlib.util.spec_from_file_location('connector',connector);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if m.SENSITIVE.search(wire):raise RuntimeError('PAYLOAD_SENSITIVE')
argv,*_=m.load_transport('trn21');argv[-1]='exec /usr/bin/python3 -I -B -S -'
started=m.utc();r=subprocess.run(argv,input=wire,capture_output=True,timeout=45)
if m.SENSITIVE.search(r.stdout+r.stderr):raise RuntimeError('OUTPUT_SENSITIVE')
result={'started_utc':started,'ended_utc':m.utc(),'returncode':r.returncode,'source_sha256':hashlib.sha256(wire).hexdigest(),'prior_postflight_sha256':'d60637b60143398cb757332eed3bf7a74a0a9cf4cfa966e9f23a314cb6c70584','stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.stdout else None}
dest=HERE/'trn21.json'
with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
o=result['remote'] or {}
print(json.dumps({'utc':o.get('utc'),'returncode':r.returncode,'same_live':[v['actual'] for v in o.get('observed_identities',{}).values() if v['same_identity']],'census_rows':len(o.get('census',[])),'guardian_children':o.get('guardian_thread_children'),'lock':o.get('lock'),'lock_fds':o.get('guardian_original_lock_fds'),'workload_rows':o.get('workload_rows'),'unreadable_cmdlines':o.get('unreadable_cmdlines'),'neuron_ls':o.get('checks',{}).get('neuron_ls'),'fuser':o.get('checks',{}).get('fuser'),'free_disk_bytes':o.get('free_disk_bytes'),'receipt_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
raise SystemExit(r.returncode)
