"""Bounded local archive-only join. No SDK, host access, cache mutation or hit inference."""
import argparse, base64, csv, hashlib, io, json, math, re
from pathlib import Path
MAX_BYTES=16*1024*1024
SOURCE='36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'
LABEL='fs-20260926-arch36a8-startup-clock-trn22-r1-full'
ROOT='/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/'+LABEL+'-packet'
CAPTURES=[
 dict(id='A',commit='bdde21c9',observation_sha256='2028695a0c2873a6fb5228ed7108b562604f491b13c82aebad4740680ed63e43',module_sha256='5a0c39fda4d4962afd222db766c1b2eda6ad6ed7d0b0fd1889afc42edc9a4e3c',module_bytes=17463,directory_name='5818832ea9b1141bb19fe350b46c050b',compiler_ir_signature='61720c5b1acb495628a6deea84ac5ec511d36e7584a6c33990d64f477e72b92a'),
 dict(id='B',commit='9b525f5f',observation_sha256='070aefbb08133233c56ff64830d7a84f69911027f887be4b8545e63764dbc12d',module_sha256='f21636e78487960b13e9682fbfd8cc987854ffc2da4d7df7ad314474ec467848',module_bytes=20784,directory_name='9cb2b39af637430f69a725b239ac9490',compiler_ir_signature='e59fc5a218c4d2215eb92907b3d17f10a3ee5a2f69e0ff443fb746a1efd0092a')]
class InvalidArchive(ValueError):pass
def sha(b):return hashlib.sha256(b).hexdigest()
def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise InvalidArchive('DUPLICATE_JSON_KEY:'+k)
        out[k]=v
    return out
def parse(raw):return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(InvalidArchive('NONFINITE_JSON:'+x)))
def finite(v):return not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and v>=0
def analyze(raw,expected_sha):
    if len(raw)>MAX_BYTES:raise InvalidArchive('BUNDLE_OVER_16_MIB')
    if sha(raw)!=expected_sha:raise InvalidArchive('BUNDLE_SHA_MISMATCH')
    bundle=parse(raw)
    if not isinstance(bundle,dict):raise InvalidArchive('BUNDLE_OBJECT_REQUIRED')
    entries=bundle.get('files')
    if not isinstance(entries,dict) or len(entries)>256:raise InvalidArchive('FILES_SCHEMA_OR_COUNT')
    files={};pins={};total=0
    for name,item in entries.items():
        if not isinstance(item,dict) or ('text' in item)==('base64' in item):raise InvalidArchive('ENCODING:'+name)
        b=item['text'].encode('utf-8') if 'text' in item else base64.b64decode(item['base64'],validate=True)
        total+=len(b)
        if total>MAX_BYTES:raise InvalidArchive('DECODED_OVER_16_MIB')
        if item.get('bytes')!=len(b) or item.get('sha256')!=sha(b):raise InvalidArchive('ARTIFACT_HASH_OR_SIZE:'+name)
        files[name]=b;pins[name]=sha(b)
    if 'file_count' in bundle and bundle['file_count']!=len(files):raise InvalidArchive('FILE_COUNT_MISMATCH')
    if 'raw_bytes' in bundle and bundle['raw_bytes']!=total:raise InvalidArchive('RAW_BYTES_MISMATCH')
    def record(name):return parse(files[name]) if name in files else {}
    terminal=record('terminal.json');outer=record('outer-flock-exit.json');binding=record('NATIVE_BINDING.disabled.json');h=record('train/rehearsal.json')
    unknown=[]
    if not (bundle.get('root')==ROOT and binding.get('label')==LABEL and binding.get('source_sha256')==SOURCE and binding.get('host')=='trn22' and binding.get('sandbox')==ROOT+'/train'):unknown.append('NOT_EXACT_FULL36_BINDING')
    if not (terminal.get('label')==LABEL and terminal.get('status')=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and terminal.get('eval_exit_code')==0 and outer.get('returncode')==0 and h.get('returncode')==0 and h.get('mode')=='train'):unknown.append('FINAL_SUCCESSFUL_FULL_BUNDLE_UNPROVEN')
    timing={k:h.get(k) for k in ['returncode','mode','wall_seconds','step_boundaries','startup_allowance_observed','charged_through_process_exit','charged_through_process_exit_within_budget','checkpoint_sha256','checkpoint_bytes']}
    obs=h.get('observations',[])
    fifth=[x.get('observed_seconds') for x in obs if isinstance(x,dict) and x.get('step')==4] if isinstance(obs,list) else []
    timing['uncapped_fifth_seconds']=fifth[0] if len(fifth)==1 else None
    if not all(finite(timing[k]) for k in ['wall_seconds','startup_allowance_observed','charged_through_process_exit','uncapped_fifth_seconds']):unknown.append('FINAL_HARNESS_TIMING_INCOMPLETE_OR_INVALID')
    rows=[];ranks=[]
    for name,b in files.items():
        m=re.fullmatch(r'metrics-training/metric_table_graph_stats_rank([0-3])\.csv',name)
        if not m:continue
        rank=int(m[1]);ranks.append(rank)
        reader=csv.DictReader(io.StringIO(b.decode('utf-8')))
        required={'graph_id','graph_name','graph_node_count','torch_neuronx_compile_us'}
        if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)) or not required<=set(reader.fieldnames):raise InvalidArchive('CSV_HEADERS:'+name)
        for index,row in enumerate(reader):
            if index>=1000 or None in row:raise InvalidArchive('CSV_ROW_LIMIT_OR_SCHEMA:'+name)
            if any(not isinstance(row.get(k),str) or not row[k] for k in required):raise InvalidArchive('CSV_MISSING_VALUE:'+name)
            key=row['graph_id'];prefix=key.split(':',1)[0]
            if not re.fullmatch('[0-9a-f]{64}',prefix):raise InvalidArchive('GRAPH_ID_PREFIX:'+name)
            try:us=float(row['torch_neuronx_compile_us']);nodes=int(row['graph_node_count'])
            except (ValueError,TypeError):raise InvalidArchive('CSV_NUMERIC:'+name)
            if not math.isfinite(us) or us<0 or nodes<0:raise InvalidArchive('CSV_NUMERIC_RANGE:'+name)
            rows.append(dict(rank=rank,csv=name,row=index+2,graph_id=key,graph_key_prefix=prefix,key_suffix=key.split(':',1)[1] if ':' in key else None,graph_name=row['graph_name'],model_name=row.get('model_name'),graph_node_count=nodes,timestamp=row.get('timestamp'),compile_us=us,positive_compile_seconds=us/1e6 if us>0 else None,zero_compile_interpretation='UNKNOWN_NOT_CACHE_HIT' if us==0 else None))
    if sorted(ranks)!=[0,1,2,3]:unknown.append('FOUR_CLOSED_RANK_TABLES_MISSING')
    joined=[]
    for cap in CAPTURES:
        matches=[r for r in rows if r['graph_key_prefix']==cap['module_sha256']]
        complete=sorted(r['rank'] for r in matches)==[0,1,2,3] and len({(r['graph_name'],r['graph_node_count']) for r in matches})==1
        if not complete:unknown.append('CAPTURE_'+cap['id']+'_MISSING_PARTIAL_OR_NONUNIQUE_JOIN')
        joined.append(dict(capture=cap,matches=matches,all_four_ranks_one_consistent_row=complete))
    return dict(status='MATCHED_CLOSED_FULL36_INPUTS_ONLY' if not unknown else 'UNKNOWN',unknown_reasons=unknown,bundle_sha256=sha(raw),authenticated_files=len(files),artifact_sha256=pins,all_rank_rows=rows,capture_joins=joined,original_final_harness_timing=timing,cache_hit_status='UNPROVEN',path_causality='UNPROVEN',compile_times_are='Per-row positive recorded compile time only; not summed across ranks or equated to elapsed startup.',identity_note='Only module SHA joins graph_id prefix. Compiler IR signature and directory name are separately retained, never substituted.',quality_verdict='NOT_PERFORMED')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--expected-sha256',required=True);a=ap.parse_args()
    try:
        if a.bundle.is_symlink() or not a.bundle.is_file() or a.bundle.stat().st_size>MAX_BYTES:raise InvalidArchive('LOCAL_BUNDLE_PATH_OR_SIZE')
        with a.bundle.open('rb') as f:raw=f.read(MAX_BYTES+1)
        out=analyze(raw,a.expected_sha256)
    except (InvalidArchive,ValueError,TypeError,KeyError,UnicodeError,OSError) as e:
        print(json.dumps(dict(status='REFUSED_INVALID_ARCHIVE',error=type(e).__name__+': '+str(e),cache_hit_status='UNPROVEN',path_causality='UNPROVEN')));return 2
    print(json.dumps(out,indent=2,allow_nan=False));return 0 if out['status']=='MATCHED_CLOSED_FULL36_INPUTS_ONLY' else 3
if __name__=='__main__':raise SystemExit(main())
