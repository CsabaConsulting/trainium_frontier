"""Per-host one-use, stdlib-only metadata and exact read-only identity inspection."""
import ast
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
assert len(sys.argv)==2 and sys.argv[1] in ('trn21','trn22')
host=sys.argv[1];dest=HERE/(host+'.json');assert not dest.exists(),'ONE_USE_RECEIPT_EXISTS'
template=HERE.parent/'20260926-trn21-clean-native-runtime-readonly/inspect_once.py'
raw=template.read_bytes();assert hashlib.sha256(raw).hexdigest()=='72e5bb6744c41e16199d533a59495c884e452b855c83222968f9615becad3a45'
body=next(ast.literal_eval(n.value) for n in ast.parse(raw).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
packet=REPO/'research/v2/proposals/20260926-confirmed36-clean-native-trn21-packet-r1/packet'
binding=json.loads((packet/'NATIVE_BINDING.disabled.json').read_bytes())
project21='/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2'
project22='/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1'
anchor=project22+'/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet/train/train.py'
payload={'hostname':'ip-172-31-34-85' if host=='trn21' else 'ip-172-31-32-253','project':project21 if host=='trn21' else project22,'root':binding['root'] if host=='trn21' else '/home/ubuntu/k2-f0f2-trn22-evidence-r1-packet','label':'READONLY_NATIVE_PACKAGE_INSPECTION' if host=='trn21' else 'fs-20260926-k2-realshape-f0f2-trn22-evidence-r1','runtime_paths':list(binding['runtime_pins']),'inputs':{} if host=='trn21' else {anchor:{'bytes':87986,'sha256':'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43'}},'proposed_helpers':{},'resources':binding['resources'] if host=='trn21' else {'initial_disk_bytes':16*1024**3,'initial_ram_bytes':24*1024**3},'extra_absent_paths':[] if host=='trn21' else ['/home/ubuntu/k2-f0f2-trn22-evidence-r1-output']}
if host=='trn22':
    body=body.replace("'host':'trn21'","'host':'trn22'").replace('(66305,596823)','(66305,832589)')
needle=" out['native_package_version_inspection']='NOT_PERFORMED: only eight explicitly assigned executable identities; no SDK imports.'"
assert body.count(needle)==1
metadata_body=r'''
 import importlib.metadata as distribution_metadata
 import importlib.machinery as machinery
 import re,sys
 site=Path('/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages')
 if site.is_symlink() or not site.is_dir():raise RuntimeError('SITE_PACKAGES_ROOT')
 out['interpreter']={'invoked_path':sys.executable,'resolved_path':str(Path(sys.executable).resolve(strict=True)),'version':sys.version,'no_site':sys.flags.no_site,'isolated':sys.flags.isolated,'no_bytecode':sys.dont_write_bytecode,'sdk_imports_forbidden':True}
 if not sys.flags.no_site or not sys.flags.isolated or not sys.dont_write_bytecode:raise RuntimeError('INSPECTION_INTERPRETER_FLAGS')
 distdirs=sorted(site.glob('*.dist-info'))
 if len(distdirs)>512:raise RuntimeError('DIST_INFO_ENTRY_BOUND')
 selected=[d for d in distdirs if d.name.lower().startswith('torch-') or 'neuron' in d.name.lower() or d.name.lower().startswith('nki-')]
 if len(selected)>32:raise RuntimeError('SELECTED_PACKAGE_BOUND')
 records=[]
 mapping={'torch':['torch'],'torch-neuronx':['torch_neuronx'],'neuronx-cc':['neuronxcc'],'neuronx-distributed':['neuronx_distributed'],'nki':['nki']}
 for d in selected:
  if time.monotonic()>deadline:raise RuntimeError('METADATA_TIME_BOUND')
  if d.is_symlink() or not d.is_dir():raise RuntimeError('DIST_INFO_TYPE')
  mp=d/'METADATA'
  if mp.stat().st_size>1024**2:raise RuntimeError('METADATA_SIZE_BOUND')
  mpin,mident=file_record(mp)
  dist=distribution_metadata.Distribution.at(str(d));name=dist.metadata['Name'];version=dist.metadata['Version']
  norm=re.sub('[-_.]+','-',name).lower()
  top=d/'top_level.txt';top_pin=None;modules=mapping.get(norm,[])
  if top.exists():
   if top.stat().st_size>65536:raise RuntimeError('TOPLEVEL_SIZE_BOUND')
   top_pin,_=file_record(top)
   modules=sorted(set(modules+[line.strip() for line in top.read_text().splitlines() if re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',line.strip())]))
  if len(modules)>64:raise RuntimeError('MODULE_COUNT_BOUND')
  origins=[]
  for mod in modules:
   # PathFinder reads only the explicit directory; it does not execute the package.
   spec=machinery.PathFinder.find_spec(mod,[str(site)])
   record={'module':mod,'found':spec is not None,'scope':'Explicit-site filesystem resolution without import, not executed runtime attestation'}
   if spec is not None:
    record.update(origin=spec.origin,package_search_locations=list(spec.submodule_search_locations or []))
    if spec.origin and spec.origin not in ('built-in','frozen'):
     origin=Path(spec.origin);record['origin_identity']=metadata(origin.stat())
     if origin.suffix=='.py' and origin.stat().st_size<=1024**2:record['origin_pin'],record['origin_identity']=file_record(origin)
   origins.append(record)
  after_pin,_=file_record(mp)
  if after_pin!=mpin:raise RuntimeError('METADATA_DRIFT')
  records.append({'name':name,'version':version,'dist_info_root':str(d),'package_root':str(site),'metadata_path':str(mp),'metadata_pin':mpin,'metadata_identity':mident,'top_level_pin':top_pin,'filesystem_import_origins':origins})
 native_loaded=[n for n in sys.modules if n.split('.')[0] in ('torch','torch_neuronx','neuronxcc','neuronx_distributed','nki')]
 if native_loaded:raise RuntimeError('UNEXPECTED_NATIVE_MODULE_IMPORT:'+repr(native_loaded))
 installed={re.sub('[-_.]+','-',x['name']).lower() for x in records}
 out['native_package_version_inspection']={'site_packages':str(site),'selected_packages':records,'requested_distributions_not_observed':[n for n in ('torch','torch-neuronx','neuronx-cc','neuronx-distributed','nki') if n not in installed],'dist_info_entries_considered':len(distdirs),'native_modules_loaded':native_loaded,'package_code_executed':False,'metadata_is_hardware_attestation':False}
 out['extra_absent_paths']={p:not os.path.lexists(p) for p in P['extra_absent_paths']}
 if not all(out['extra_absent_paths'].values()):raise RuntimeError('PROPOSED_OUTPUT_EXISTS')
'''
body=body.replace(needle,metadata_body)
ast.parse(body)
state_raw=(REPO/'research/v2/state.json').read_bytes();s=json.loads(state_raw)
controller={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state_sha256':hashlib.sha256(state_raw).hexdigest(),'host_launch_allowed':s['hosts'][host]['launch_allowed'],'host_status':s['hosts'][host]['status'],'global_correctness_breaker':s['breakers']['global_correctness']['tripped'],'host_breaker':s['breakers'][host]['tripped'],'local_stops_present':[str(REPO/rel) for rel in ('STOP','research/agentic/STOP','research/v2/STOP') if (REPO/rel).exists() or (REPO/rel).is_symlink()],'not_a_release':True}
wire=('P='+repr(payload)+'\n'+body).encode()
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);assert not m.SENSITIVE.search(wire)
argv,*_=m.load_transport(host);argv[-1]='exec /home/ubuntu/workspace/native_venv/bin/python -I -B -S -'
r=subprocess.run(argv,input=wire,capture_output=True,timeout=60)
assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':host,'returncode':r.returncode,'remote_body_sha256':hashlib.sha256(wire).hexdigest(),'template_sha256':hashlib.sha256(raw).hexdigest(),'selected_k2_proposal_commit':'d6ded1f925bf6b929407ff36cd365c8d50655ea7','controller_observation':controller,'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.stdout else None}
with dest.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
o=receipt['remote'] or {}
print(json.dumps({'host':host,'returncode':r.returncode,'status':o.get('status'),'utc':o.get('ended_utc'),'packages':[{'name':v['name'],'version':v['version']} for v in o.get('native_package_version_inspection',{}).get('selected_packages',[])],'runtime_count':len(o.get('runtime_pins',{})),'anchor':o.get('retained_inputs'),'resources':o.get('resources'),'root':o.get('proposed_root'),'extra_absent_paths':o.get('extra_absent_paths'),'error':o.get('error'),'receipt_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
raise SystemExit(r.returncode)
