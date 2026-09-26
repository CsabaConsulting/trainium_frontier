"""Bounded read-only Git tree sizing under the original trn22 idle lock."""
import ast,base64,datetime,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
C=Path('/home/poojith-devan/Desktop/Trainium')
PRIOR=HERE.parent/'20260926-trn21-clean-native-runtime-readonly/inspect_once.py'
raw=PRIOR.read_bytes()
if hashlib.sha256(raw).hexdigest()!='72e5bb6744c41e16199d533a59495c884e452b855c83222968f9615becad3a45':raise RuntimeError('PRIOR_COLLECTOR_PIN')
body=next(ast.literal_eval(n.value) for n in ast.parse(raw).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
body=body.replace("'host':'trn21'","'host':'trn22'").replace('(66305,596823)','(66305,832589)')
needle=" out['native_package_version_inspection']='NOT_PERFORMED: only eight explicitly assigned executable identities; no SDK imports.'"
extra=r'''
 import base64
 expected='65155c668efaa215a2136f67617b44ed32d4c9f6'
 if git[0]['stdout'].strip()!=expected:raise RuntimeError('ORIGINAL_HEAD_DRIFT')
 prior_logs={f'?? research/run-logs/fs-20260924-v2-35c9-lr015-control-trn22-r1.{suffix}' for suffix in ('job.json','launch.json','log','postflight.json','preflight.json','process.json')}
 actual=set(git[1]['stdout'].splitlines())
 out['original_status_matches_prior_six_untracked']=actual==prior_logs
 if actual!=prior_logs:raise RuntimeError('ORIGINAL_STATUS_DRIFT')
 cmd=['/usr/bin/git','--no-optional-locks','-c','core.fsmonitor=false','-C',str(project),'ls-tree','-r','-l','-z',expected]
 tree=subprocess.run(cmd,capture_output=True,timeout=12)
 if tree.returncode:raise RuntimeError('TREE_READ_FAILED:'+tree.stderr.decode(errors='replace'))
 if len(tree.stdout)>8*1024**2 or len(tree.stderr)>65536:raise RuntimeError('TREE_OUTPUT_BOUND')
 entries=[];total=0;regular=0;symlink=0;submodule=0;allocated=0;dirs=set();other=[];symlink_bytes=0
 for row in tree.stdout.split(b'\0'):
  if not row:continue
  left,name=row.split(b'\t',1);mode,kind,oid,size=left.split()
  entries.append((mode,kind,oid,size,name))
  parts=name.split(b'/')
  for i in range(1,len(parts)):dirs.add(b'/'.join(parts[:i]))
  if mode in (b'100644',b'100755') and kind==b'blob':
   n=int(size);total+=n;regular+=1;allocated+=((n+4095)//4096)*4096
  elif mode==b'120000' and kind==b'blob':symlink+=1;symlink_bytes+=int(size)
  elif mode==b'160000' and kind==b'commit':submodule+=1
  else:other.append({'mode':mode.decode(),'kind':kind.decode(),'size':size.decode()})
 if len(entries)>100000:raise RuntimeError('TREE_ENTRY_BOUND')
 out['tracked_tree']={'command':cmd,'returncode':tree.returncode,'raw_sha256':hashlib.sha256(tree.stdout).hexdigest(),'raw_bytes':len(tree.stdout),'raw_listing_base64':base64.b64encode(tree.stdout).decode(),'regular_blob_count':regular,'regular_blob_logical_bytes':total,'symlink_count':symlink,'symlink_logical_bytes':symlink_bytes,'submodule_count':submodule,'other_entry_types':other,'all_entry_count':len(entries),'directory_path_count':len(dirs),'regular_blob_4k_allocation_estimate_bytes':allocated,'not_included_in_estimate':'Directory metadata, .git worktree administrative files, index, filesystem overhead, concurrent disk changes and any diagnostic output.'}
 wr=subprocess.run(['/usr/bin/git','--no-optional-locks','-c','core.fsmonitor=false','-C',str(project),'worktree','list','--porcelain'],capture_output=True,text=True,timeout=8)
 if len(wr.stdout.encode())+len(wr.stderr.encode())>65536:raise RuntimeError('WORKTREE_OUTPUT_BOUND')
 out['worktree_registry']={'returncode':wr.returncode,'stdout':wr.stdout,'stderr':wr.stderr}
 if wr.returncode:raise RuntimeError('WORKTREE_REGISTRY_READ_FAILED')
 ancestors=[]
 for p in reversed(root.parents):
  s=p.lstat();ancestors.append({'path':str(p),'identity':metadata(s),'symlink':stat.S_ISLNK(s.st_mode),'directory':stat.S_ISDIR(s.st_mode)})
  if stat.S_ISLNK(s.st_mode) or not stat.S_ISDIR(s.st_mode):raise RuntimeError('PROSPECTIVE_PARENT_NOT_PLAIN')
 out['proposed_root']['ancestors']=ancestors
 out['proposed_root']['resolved_parent']=str(root.parent.resolve(strict=True))
 out['proposed_root']['resolved_lexical_target']=str(root.parent.resolve(strict=True)/root.name)
 fs=subprocess.run(['/usr/bin/findmnt','-n','-o','FSTYPE,SOURCE,MAJ:MIN','-T',str(root.parent)],capture_output=True,text=True,timeout=8)
 if len(fs.stdout.encode())+len(fs.stderr.encode())>4096:raise RuntimeError('FILESYSTEM_OUTPUT_BOUND')
 out['filesystem']={'returncode':fs.returncode,'stdout':fs.stdout,'stderr':fs.stderr,'parent_device':root.parent.stat().st_dev}
 out['capacity_estimate']={'free_bytes_at_observation':disk,'k2_floor_bytes':16*1024**3,'headroom_before_checkout_bytes':disk-16*1024**3,'logical_bytes_after_checkout':disk-total,'regular_blob_4k_estimated_free_after_checkout':disk-allocated,'regular_blob_4k_estimated_margin_above_floor':disk-allocated-16*1024**3,'checkout_feasibility_not_authorization':True,'full_footprint_unproven':True}
 out['sdk_static_mapping']='NOT_REREAD: prior unchanged-source static mapping facts 3605794a already bind known installed SDK sources; no new SDK/package fact needed for tracked-tree sizing.'
 out['source_or_data_executed_or_read']=False
 out['git_project_modified']=False
 out['worktree_created']=False
 out['retired_queue_pids_exist']={str(pid):Path('/proc',str(pid)).exists() for pid in (3050509,3053905)}
'''
if body.count(needle)!=1:raise RuntimeError('TEMPLATE_ANCHOR')
body=body.replace(needle,extra);ast.parse(body)
payload={'hostname':'ip-172-31-32-253','project':'/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1','root':'/home/ubuntu/trainium_k2_f0f2_trn22_evidence_20260926_r1','label':'READONLY_K2_CLEAN_CHECKOUT_SIZE_FACTS','runtime_paths':[],'inputs':{},'proposed_helpers':{},'resources':{'initial_disk_bytes':16*1024**3,'initial_ram_bytes':24*1024**3}}
wire=('P='+repr(payload)+'\n'+body).encode()
connector=C/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
if hashlib.sha256(connector.read_bytes()).hexdigest()!='7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899':raise RuntimeError('CONNECTOR_PIN')
spec=importlib.util.spec_from_file_location('connector',connector);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
state_raw=(C/'research/v2/state.json').read_bytes();state=json.loads(state_raw);obs=m.local_state()
if obs['local_stops_present'] or obs['breaker_tripped']['global_correctness'] is not False or obs['breaker_tripped']['trn22'] is not False:raise RuntimeError('LOCAL_STOP_BREAKER')
controller={'utc':m.utc(),'state_sha256':hashlib.sha256(state_raw).hexdigest(),'host_launch_allowed':state['hosts']['trn22']['launch_allowed'],'global_correctness_breaker':obs['breaker_tripped']['global_correctness'],'trn22_breaker':obs['breaker_tripped']['trn22'],'local_stops_present':obs['local_stops_present'],'read_only_not_launch_authority':True}
if m.SENSITIVE.search(wire):raise RuntimeError('SENSITIVE_PAYLOAD')
argv,*_=m.load_transport('trn22');argv[-1]='exec /usr/bin/python3 -I -B -S -'
result=subprocess.run(argv,input=wire,capture_output=True,timeout=60)
if m.SENSITIVE.search(result.stdout+result.stderr):raise RuntimeError('SENSITIVE_OUTPUT')
receipt={'utc':m.utc(),'returncode':result.returncode,'payload_sha256':hashlib.sha256(wire).hexdigest(),'controller':controller,'remote':json.loads(result.stdout) if result.stdout else None,'stderr':result.stderr.decode()}
dest=HERE/'trn22.json'
with dest.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
o=receipt['remote'] or {};summary={k:v for k,v in o.get('tracked_tree',{}).items() if k not in ('raw_listing_base64','command')}
print(json.dumps({'utc':o.get('ended_utc'),'status':o.get('status'),'tree':summary,'capacity':o.get('capacity_estimate'),'root_absent':not o.get('proposed_root',{}).get('lexists',True),'original_status_matches_prior_six_untracked':o.get('original_status_matches_prior_six_untracked'),'resources':o.get('resources'),'filesystem':o.get('filesystem'),'worktree_registry':o.get('worktree_registry'),'lock':o.get('original_lock'),'error':o.get('error'),'receipt_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
raise SystemExit(result.returncode)
