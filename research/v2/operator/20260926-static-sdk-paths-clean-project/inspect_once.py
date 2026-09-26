"""One static SDK path/worktree inspection. No SDK imports or repository changes."""
import ast,datetime,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=Path('/home/poojith-devan/Desktop/Trainium')
assert len(sys.argv)==2 and sys.argv[1] in ('trn21','trn22')
host=sys.argv[1];dest=HERE/(host+'.json');assert not dest.exists(),'ONE_USE_RECEIPT_EXISTS'
template=HERE.parent/'20260926-trn21-clean-native-runtime-readonly/inspect_once.py'
raw=template.read_bytes();assert hashlib.sha256(raw).hexdigest()=='72e5bb6744c41e16199d533a59495c884e452b855c83222968f9615becad3a45'
body=next(ast.literal_eval(n.value) for n in ast.parse(raw).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
prior_path=HERE.parent/'20260926-native-metadata-and-trn22-k2-facts'/(host+'.json');prior=json.loads(prior_path.read_bytes())
packages=[v for v in prior['remote']['native_package_version_inspection']['selected_packages'] if v['name'] in ('torch-neuronx','nki')]
project=prior['remote']['project']['path']
payload={'hostname':prior['remote']['hostname'],'project':project,'root':prior['remote']['proposed_root']['path'],'label':'READONLY_STATIC_IMPORT_MAPPING_AND_EXISTING_PROJECTS','runtime_paths':[],'inputs':{},'proposed_helpers':{},'resources':{'initial_disk_bytes':0,'initial_ram_bytes':0},'packages':packages,'check_projects':host=='trn22'}
if host=='trn22':body=body.replace("'host':'trn21'","'host':'trn22'").replace('(66305,596823)','(66305,832589)')
needle=" out['native_package_version_inspection']='NOT_PERFORMED: only eight explicitly assigned executable identities; no SDK imports.'"
assert body.count(needle)==1
extra=r'''
 import ast,csv,io,importlib.machinery as machinery,re,sys
 site=Path('/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages')
 roots=[str(site)];pth_candidates=set();record_rows=[];source_records={};finder_mappings={};unresolved=[]
 def smalltext(path,limit):
  p=Path(path)
  if p.stat().st_size>limit:raise RuntimeError('SMALL_METADATA_BOUND:'+str(p))
  pin,ident=file_record(p);text=p.read_text()
  if hashlib.sha256(text.encode()).hexdigest()!=pin['sha256']:raise RuntimeError('TEXT_DRIFT_OR_ENCODING')
  source_records[str(p)]={'pin':pin,'identity':ident,'text':text}
  return text
 def allowed_module_path(p):
  resolved=Path(p).resolve(strict=True)
  if not (str(resolved).startswith('/home/ubuntu/workspace/') or str(resolved).startswith('/opt/aws/neuron/')):raise RuntimeError('MODULE_MAPPING_OUT_OF_SCOPE:'+str(resolved))
  return resolved
 for package in P['packages']:
  d=Path(package['dist_info_root']);mpin,_=file_record(d/'METADATA')
  if mpin!=package['metadata_pin']:raise RuntimeError('PRIOR_METADATA_DRIFT')
  record=d/'RECORD'
  if not record.is_file():unresolved.append({'package':package['name'],'reason':'RECORD_MISSING'});continue
  text=smalltext(record,2*1024**2);rows=list(csv.reader(io.StringIO(text)))
  if len(rows)>32768:raise RuntimeError('RECORD_ROWS_BOUND')
  roots_in_record=sorted({row[0].split('/')[0] for row in rows if row})
  relevant=[]
  for row in rows:
   if not row:continue
   rel=row[0]
   if rel.endswith('.pth'):
    pth_candidates.add(site/rel)
   if rel.endswith('/__init__.py') and (rel.count('/')<=2) and any(x in rel.lower() for x in ('torch','neuron','nki')):relevant.append(row)
  record_rows.append({'package':package['name'],'record_path':str(record),'row_count':len(rows),'top_entries':roots_in_record,'selected_root_init_rows':relevant})
 for p in site.glob('*.pth'):
  if any(x in p.name.lower() for x in ('neuron','torch','nki')):pth_candidates.add(p)
 if len(pth_candidates)>32:raise RuntimeError('PTH_COUNT_BOUND')
 pth_records=[]
 for p in sorted(pth_candidates):
  text=smalltext(p,65536);rec={'path':str(p),'lines':[]}
  for line in text.splitlines():
   line=line.strip()
   if not line or line.startswith('#'):continue
   if not line.startswith(('import ','import\t')):
    q=Path(line);q=q if q.is_absolute() else site/q
    row={'kind':'path','value':line,'exists':q.exists()}
    if q.exists():resolved=allowed_module_path(q);row['resolved_path']=str(resolved);roots.append(str(resolved))
    rec['lines'].append(row)
   else:
    rec['lines'].append({'kind':'import_NOT_EXECUTED','text':line})
    try:tree=ast.parse(line)
    except SyntaxError:unresolved.append({'path':str(p),'reason':'NONPARSEABLE_PTH_IMPORT'});continue
    names=[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    for name in names:
     if not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',name) or not any(x in name.lower() for x in ('torch','neuron','nki')):continue
     finder=site/(name+'.py')
     if not finder.is_file():unresolved.append({'path':str(finder),'reason':'STATIC_FINDER_FILE_MISSING'});continue
     ftext=smalltext(finder,128*1024);ftree=ast.parse(ftext)
     for node in ftree.body:
      target=node.targets[0] if isinstance(node,ast.Assign) and len(node.targets)==1 else node.target if isinstance(node,ast.AnnAssign) else None
      if isinstance(target,ast.Name) and target.id in ('MAPPING','NAMESPACES'):
       try:value=ast.literal_eval(node.value)
       except (ValueError,TypeError):unresolved.append({'path':str(finder),'reason':'NONLITERAL_'+target.id});continue
       finder_mappings[str(finder)+':'+target.id]=value
  pth_records.append(rec)
 origins=[]
 for mod in ('torch_neuronx','torch_mlir','nki'):
  seen=[]
  for search in dict.fromkeys(roots):
   spec=machinery.PathFinder.find_spec(mod,[search])
   if spec is not None:seen.append({'method':'EXPLICIT_FILESYSTEM_SEARCH_NOT_IMPORT','search_root':search,'origin':spec.origin,'locations':list(spec.submodule_search_locations or [])})
  for key,mapping in finder_mappings.items():
   if isinstance(mapping,dict) and mod in mapping:
    values=mapping[mod] if isinstance(mapping[mod],list) else [mapping[mod]]
    for value in values:
     if isinstance(value,str):
      p=allowed_module_path(value);origin=p/'__init__.py' if p.is_dir() else p
      seen.append({'method':'STATIC_LITERAL_EDITABLE_MAPPING_NOT_EXECUTED','mapping':key,'origin':str(origin),'locations':[str(p)] if p.is_dir() else []})
  for row in seen:
   if row['origin'] and row['origin'] not in ('built-in','frozen'):
    p=allowed_module_path(row['origin']);pin,ident=file_record(p);row['origin_pin']=pin;row['origin_identity']=ident
  origins.append({'module':mod,'candidates':seen,'resolved_statically':bool(seen),'actual_runtime_import_executed':False})
 out['static_sdk_mapping']={'record_rows':record_rows,'pth':pth_records,'literal_finder_mappings':finder_mappings,'origin_candidates':origins,'selected_metadata_files':source_records,'unresolved':unresolved,'sdk_modules_loaded':[m for m in sys.modules if m.split('.')[0] in ('torch','torch_neuronx','torch_mlir','nki','neuronxcc')],'no_pth_or_finder_executed':True}
 if out['static_sdk_mapping']['sdk_modules_loaded']:raise RuntimeError('UNEXPECTED_SDK_IMPORT')
 if P['check_projects']:
  def gitread(path,args):
   r=subprocess.run(['/usr/bin/git','--no-optional-locks','-c','core.fsmonitor=false','-C',str(path),*args],capture_output=True,text=True,timeout=8)
   if len(r.stdout.encode())+len(r.stderr.encode())>65536:raise RuntimeError('GIT_OUTPUT_BOUND')
   return {'args':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
  worktrees=gitread(project,['worktree','list','--porcelain']);out['original_worktree_list']=worktrees
  paths=[Path('/home/ubuntu/trainium_frontier'),Path('/home/ubuntu/trainium_freshstart_20260918')]
  if worktrees['returncode']==0:
   listed=[Path(l[9:]) for l in worktrees['stdout'].splitlines() if l.startswith('worktree ')]
   choice=next((p for p in listed if p not in paths and p!=project),None)
   if choice is None:choice=next((p for p in listed if p not in paths),None)
   if choice is not None:paths.append(choice)
  observations=[]
  for p in paths:
   row={'path':str(p),'exists':os.path.lexists(p),'clean':False}
   if p.exists() and not p.is_symlink() and p.is_dir():
    row['identity']=metadata(p.stat());row['head']=gitread(p,['rev-parse','HEAD']);row['top_level']=gitread(p,['rev-parse','--show-toplevel']);row['status']=gitread(p,['status','--porcelain=v1','--untracked-files=normal'])
    row['clean']=all(row[k]['returncode']==0 for k in ('head','top_level','status')) and row['status']['stdout']=='' and row['top_level']['stdout'].strip()==str(p.resolve())
   elif row['exists']:row['refused']='NOT_PLAIN_DIRECTORY'
   observations.append(row)
  out['at_most_three_existing_projects']=observations
  out['clean_existing_project_paths']=[row['path'] for row in observations if row['clean']]
'''
body=body.replace(needle,extra);ast.parse(body)
wire=('P='+repr(payload)+'\n'+body).encode()
spec=importlib.util.spec_from_file_location('transport',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);assert not m.SENSITIVE.search(wire)
argv,*_=m.load_transport(host);argv[-1]='exec /home/ubuntu/workspace/native_venv/bin/python -I -B -S -'
r=subprocess.run(argv,input=wire,capture_output=True,timeout=60)
assert not m.SENSITIVE.search(r.stdout+r.stderr)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':host,'returncode':r.returncode,'remote_body_sha256':hashlib.sha256(wire).hexdigest(),'prior_receipt_sha256':hashlib.sha256(prior_path.read_bytes()).hexdigest(),'stderr':r.stderr.decode(),'remote':json.loads(r.stdout) if r.stdout else None}
with dest.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
o=receipt['remote'] or {}
print(json.dumps({'host':host,'utc':o.get('ended_utc'),'status':o.get('status'),'mapping':o.get('static_sdk_mapping',{}).get('origin_candidates'),'clean_projects':o.get('clean_existing_project_paths'),'project_observations':o.get('at_most_three_existing_projects'),'error':o.get('error'),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
raise SystemExit(r.returncode)
