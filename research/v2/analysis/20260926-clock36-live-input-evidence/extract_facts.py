"""Read only local immutable archives; no MLIR/SDK import, rewrite or host access."""
import base64,csv,hashlib,io,json,re,subprocess
from pathlib import Path
R=Path('/home/poojith-devan/Desktop/Trainium')
paths={};checks=[]
def check(name,ok):
    if not ok:raise RuntimeError(name)
    checks.append(name)
def get(path):
    raw=(R/path).read_bytes();paths[path]=hashlib.sha256(raw).hexdigest();return raw
def blob(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=R)
def unpack(f):
    raw=f['text'].encode() if 'text' in f else base64.b64decode(f['base64'],validate=True)
    check('artifact digest',len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256']);return raw
prefix='research/v2/operator/20260926-trn22-clock-full-live-compiler/'
observations=[]
for i,ref in [(1,'bdde21c9'),(2,'9b525f5f')]:
    path=prefix+f'observation-{i}.json';raw=get(path);check('immutable observation',raw==blob(ref,path))
    o=json.loads(raw);check('one observed directory',o['status']=='OBSERVED' and len(o['directories'])==1)
    d=o['directories'][0];fs={f['name']:f for f in d['files']};module=base64.b64decode(fs['module.mlir']['base64'],validate=True);log=base64.b64decode(fs['log-neuron-cc.txt']['base64'],validate=True)
    for name,content in [('module.mlir',module),('log-neuron-cc.txt',log)]:
        f=fs[name];check('stable exact capture',f['before']==f['after']==f['path_after'] and f['status']=='READ_STABLE' and len(content)==f['bytes_retained'] and hashlib.sha256(content).hexdigest()==f['sha256'])
    strings=[x.decode('ascii') for x in re.findall(rb'[ -~]{7,}',module)]
    metadata=[]
    for x in strings:
        if x.startswith('eyJ'):
            try:m=json.loads(base64.b64decode(x,validate=True))
            except Exception:continue
            if isinstance(m,dict) and 'func_name' in m:metadata.append(m)
    lines=log.decode(errors='strict').splitlines()
    selected=[{'line':j+1,'text':l} for j,l in enumerate(lines) if any(s in l for s in ['[root]: /','Compiler version','Pipeline:','Pre-Partition Histogram','IR signature:','Tensorizer options:','needsModular_PreSplit'])]
    selected=[r for r in selected if 'Tensorizer options:' not in r['text']]+[next(r for r in selected if 'Tensorizer options:' in r['text'])]
    observations.append(dict(receipt=path,commit=ref,utc=o['started_utc'],directory=d['directory'],compiler_processes=d['compiler_processes'],module_sha256=fs['module.mlir']['sha256'],module_bytes=len(module),magic_hex=module[:16].hex(),source_paths=[s for s in strings if s.endswith('/train.py')],kernel_metadata=metadata,log_prefix_sha256=fs['log-neuron-cc.txt']['sha256'],log_full_size_at_read=fs['log-neuron-cc.txt']['before']['size'],log_bytes_retained=len(log),log_selected_lines=selected,explicit_persistent_messages=[l for l in lines if any(s in l for s in ['Persistent cache hit:','Persistent cache miss:','execution_cache_key=','Failed to read cache file','Skipping persistent cache'])]))
coldpath='research/v2/operator/20260926-trn22-startup-clock-cold-five/completed/results-bundle.json'
coldraw=get(coldpath);check('cold immutable',coldraw==blob('e08afdec',coldpath));cold=json.loads(coldraw)
for f in cold['files'].values():unpack(f)
coldreport=json.loads(unpack(cold['files']['train/rehearsal.json']))
coldrows={}
for name,f in cold['files'].items():
    if name.endswith('.csv'):coldrows[name]=list(csv.DictReader(io.StringIO(unpack(f).decode())))
snapshotpath='research/v2/operator/20260926-trn22-startup-clock-full/early-training-compact-snapshot.json'
sraw=get(snapshotpath);check('snapshot immutable',sraw==blob('59bdcace',snapshotpath));snapshot=json.loads(sraw)
env=snapshot['parsed']['training-launch.json']['environment']
check('all six retained cache roots',all(env[k]==v for k,v in coldreport['fresh_empty_sdk_caches'].items()))
check('all four workers effective roots',len(snapshot['workers'])==4 and all(all(w['environment'][k]==v for k,v in coldreport['fresh_empty_sdk_caches'].items()) for w in snapshot['workers']))
source=blob('7419927c','workflow/proposals/research-v2/train.py')
check('exact36source',hashlib.sha256(source).hexdigest()=='36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb')
coldids={r['graph_id'].split(':')[0] for rows in coldrows.values() for r in rows}
for o in observations:o['module_hash_present_in_any_cold_graph_key_prefix']=o['module_sha256'] in coldids
prior=[
'research/v2/operator/20260926-architecture-cache-readonly-2148/EVIDENCE.md',
'research/v2/operator/20260925-trn21-live-cache-diagnosis/compile-key-and-bounded-artifact-headers.txt',
'research/v2/operator/20260925-trn21-live-cache-diagnosis/cache-key-composition.txt',
'research/v2/lanes/table-compile/CACHE_SOURCE_ADDENDUM_2026-09-25.txt',
'research/v2/analysis/20260926-stable-s1-capture-miss/DIAGNOSIS.md']
for p in prior:get(p)
out=dict(scope='READ_ONLY_ARCHIVE_FACTS_NO_PRINCIPAL_GRAPH_OR_CACHE_HIT_VERDICT',checks=len(checks),artifact_sha256=paths,source_git='7419927c:workflow/proposals/research-v2/train.py',source_sha256=hashlib.sha256(source).hexdigest(),observations=observations,cold_fifth_seconds=coldreport['observations'][4]['observed_seconds'],cold_rows=coldrows,full_environment=env,full_printed_first_steps=[l for l in snapshot['logs']['training.log'].splitlines() if re.match(r'^step 0000[0-4] ',l)],full_muon_specialization_prints=[l for l in snapshot['logs']['training.log'].splitlines() if l.startswith('muon step:')],closed_full_graph_stats_available_at_audit=False)
print(json.dumps(out,indent=2))
