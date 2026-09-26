"""Read-only stdlib analysis of immutable local Git receipts; no host/SDK/model."""
import csv,hashlib,io,json,subprocess
from collections import Counter
from pathlib import Path
ROOT=Path('/home/poojith-devan/Desktop/Trainium')
PIN='be05918f'
PATH='research/v2/operator/20260926-trn21-lr-final-boundary-0333/trn21.json'
PARENT='research/v2/operator/20260926-trn21-startup-clock-replication/completed/results-bundle.json'
FIELDS=['graph_id','graph_name','model_name','has_collectives','graph_node_count','timestamp','torch_neuronx_fx_passes_us','torch_neuronx_lower_us','torch_neuronx_compile_us']
def get(p,rev=PIN):return subprocess.check_output(['git','show',rev+':'+p],cwd=ROOT)
def sha(x):return hashlib.sha256(x).hexdigest()
def need(v,m):
    if not v:raise RuntimeError(m)
raw=get(PATH);need(raw==get(PATH,'08e1ecbd'),'CANONICAL_OPERATOR_IDENTICAL')
need(sha(raw)=='c7cdbdbea99fdb734450db1da3cfb56eaddaea3059e44a5f3b853fd57055227a','RAW_SHA')
current=json.loads(raw)['remote'];parentraw=get(PARENT);parent=json.loads(parentraw)
def authenticate(entry):
    raw=entry['text'].encode()
    if 'sha256_captured' in entry:
        need(entry['capture']=='complete_at_observation','COMPLETE_CAPTURE')
        need(len(raw)==entry['captured_bytes']==entry['bytes_at_stat'],'CAPTURE_SIZE')
        need(sha(raw)==entry['sha256_captured'],'CAPTURE_SHA')
    else:
        need(len(raw)==entry['bytes'],'ARCHIVE_SIZE');need(sha(raw)==entry['sha256'],'ARCHIVE_SHA')
    return raw
def analyze(files):
    result=[]
    for rank in range(4):
        name=f'metrics-training/metric_table_graph_stats_rank{rank}.csv'
        raw=authenticate(files[name]);reader=csv.DictReader(io.StringIO(raw.decode(),newline=''))
        need(reader.fieldnames==FIELDS,'CSV_SCHEMA')
        rows=[]
        for r in reader:
            for k in ['graph_node_count',*FIELDS[6:]]:
                r[k]=int(r[k]);need(r[k]>=0,'NONNEGATIVE_INTEGER')
            r['reported_compile_seconds']=r['torch_neuronx_compile_us']/1e6
            rows.append(r)
        counts=Counter(r['graph_id'] for r in rows)
        result.append(dict(rank=rank,file=name,sha256=sha(raw),bytes=len(raw),graph_rows=len(rows),unique_ids=len(counts),repeated_ids={k:v for k,v in counts.items() if v>1},reported_compile_seconds=sum(r['torch_neuronx_compile_us'] for r in rows)/1e6,reported_fx_seconds=sum(r['torch_neuronx_fx_passes_us'] for r in rows)/1e6,reported_lower_seconds=sum(r['torch_neuronx_lower_us'] for r in rows)/1e6,rows=rows))
    return result
now=analyze(current['files']);old=analyze(parent['files']);comp=[]
for a,b in zip(now,old):
    ar={r['graph_name']:r for r in a['rows']};br={r['graph_name']:r for r in b['rows']}
    comp.append(dict(rank=a['rank'],names_equal=ar.keys()==br.keys(),by_name=[dict(name=n,nodes_now=ar[n]['graph_node_count'],nodes_parent=br[n]['graph_node_count'],exact_id_equal=ar[n]['graph_id']==br[n]['graph_id'],id_prefix_equal=ar[n]['graph_id'].split(':')[0]==br[n]['graph_id'].split(':')[0],id_suffix_equal=ar[n]['graph_id'].split(':')[1]==br[n]['graph_id'].split(':')[1],compile_seconds_now=ar[n]['reported_compile_seconds'],compile_seconds_parent=br[n]['reported_compile_seconds']) for n in sorted(ar.keys() & br.keys())]))
def facts(files):
    launch=json.loads(authenticate(files['training-launch.json']));rehearsal=json.loads(authenticate(files['train/rehearsal.json']))
    result=dict(source_sha256=launch.get('source_sha256'),environment=launch['environment'],fifth_seconds=rehearsal['observations'][4]['observed_seconds'],wall_seconds=rehearsal['wall_seconds'],charged_seconds=rehearsal['charged_through_process_exit'])
    if 'package/launch-command.txt' in files:result['science_command']=authenticate(files['package/launch-command.txt']).decode().strip()
    else:result['science_command']=json.loads(authenticate(files['NATIVE_BINDING.disabled.json']))['training_command']
    return result
cross={}
for label,ranks in [('current',now),('parent',old)]:
    counts=Counter(r['graph_id'] for rank in ranks for r in rank['rows'])
    cross[label]=dict(rows=sum(counts.values()),unique_ids=len(counts),id_multiplicities=sorted(counts.values()),summed_rank_compile_seconds=sum(r['reported_compile_seconds'] for r in ranks),sum_is_not_elapsed_wall_time=True)
output=dict(status='AUTHENTICATED_GRAPH_ACCOUNTING_ONLY',current_raw_sha256=sha(raw),parent_archive_sha256=sha(parentraw),schema=FIELDS,current=now,parent=old,cross_rank=cross,comparison=comp,current_facts=facts(current['files']),parent_facts=facts(parent['files']),source_scope='Both training receipts use exact36 bytes; LR012 current versus LR015 parent. Training paths differ. Not a one-factor cache experiment.',cache_hit_evidence='UNKNOWN: schema has no hit/miss/reuse field; compile_us=0 is not proof',runtime_repeat_count='UNKNOWN: CSV rows are graph records, not execution counts',live_eval_in_observation=not current['terminal_seen'],no_host_or_tensor_work=True)
print(json.dumps(output,indent=2,sort_keys=True))
