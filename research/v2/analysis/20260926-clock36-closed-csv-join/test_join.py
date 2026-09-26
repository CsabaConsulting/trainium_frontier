"""Tiny stdlib fixtures only; synthetic graph names are not native evidence."""
import base64,copy,hashlib,importlib.util,json,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('join',Path(__file__).with_name('join_closed_csv.py'));j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def fixture():
    files={}
    def add(name,value):
        b=(json.dumps(value) if not isinstance(value,str) else value).encode();files[name]=dict(bytes=len(b),sha256=j.sha(b),text=b.decode())
    add('NATIVE_BINDING.disabled.json',dict(label=j.LABEL,source_sha256=j.SOURCE,host='trn22',sandbox=j.ROOT+'/train'))
    add('terminal.json',dict(label=j.LABEL,status='COMPLETE_PENDING_INDEPENDENT_REVIEW',eval_exit_code=0))
    add('outer-flock-exit.json',dict(returncode=0))
    add('train/rehearsal.json',dict(returncode=0,mode='train',wall_seconds=1800,step_boundaries=9,startup_allowance_observed=800,charged_through_process_exit=1000,charged_through_process_exit_within_budget=True,observations=[dict(step=4,observed_seconds=800)]))
    for rank in range(4):
        text='graph_id,graph_name,graph_node_count,torch_neuronx_compile_us\n'
        for i,c in enumerate(j.CAPTURES):text+=c['module_sha256']+':'+('c'*64)+',synthetic_'+str(i)+','+str(10+i)+','+str((i+1)*1000000 if rank==0 else 0)+'\n'
        add(f'metrics-training/metric_table_graph_stats_rank{rank}.csv',text)
    return dict(root=j.ROOT,files=files,file_count=len(files),raw_bytes=sum(f['bytes'] for f in files.values()))
def encode(x):return json.dumps(x).encode()
def run(x):
    b=encode(x);return j.analyze(b,j.sha(b))
def replace(x,name,text):
    f=x['files'][name];f.update(text=text,bytes=len(text.encode()),sha256=j.sha(text.encode()));x['raw_bytes']=sum(f['bytes'] for f in x['files'].values())
class TestJoin(unittest.TestCase):
    def test_exact_positive_zero(self):
        o=run(fixture());self.assertEqual(o['status'],'MATCHED_CLOSED_FULL36_INPUTS_ONLY');self.assertEqual(len(o['all_rank_rows']),8);self.assertEqual(o['capture_joins'][0]['matches'][0]['positive_compile_seconds'],1);self.assertIsNone(o['capture_joins'][0]['matches'][1]['positive_compile_seconds']);self.assertEqual(o['cache_hit_status'],'UNPROVEN')
    def test_hash_mismatch(self):
        x=fixture();x['files']['terminal.json']['text']+=' ';self.assertRaises(j.InvalidArchive,run,x)
    def test_bundle_hash(self):self.assertRaises(j.InvalidArchive,j.analyze,encode(fixture()),'0'*64)
    def test_missing_rank(self):
        x=fixture();del x['files']['metrics-training/metric_table_graph_stats_rank3.csv'];x.pop('file_count');x.pop('raw_bytes');self.assertEqual(run(x)['status'],'UNKNOWN')
    def test_not_final(self):
        x=fixture();replace(x,'terminal.json',json.dumps(dict(label=j.LABEL,status='ACTIVE')));self.assertIn('FINAL_SUCCESSFUL_FULL_BUNDLE_UNPROVEN',run(x)['unknown_reasons'])
    def test_signature_is_not_key(self):
        x=fixture()
        for rank in range(4):
            n=f'metrics-training/metric_table_graph_stats_rank{rank}.csv';replace(x,n,x['files'][n]['text'].replace(j.CAPTURES[0]['module_sha256'],j.CAPTURES[0]['compiler_ir_signature']))
        o=run(x);self.assertEqual(o['status'],'UNKNOWN');self.assertEqual(o['capture_joins'][0]['matches'],[])
    def test_nonfinite_compile(self):
        x=fixture();n='metrics-training/metric_table_graph_stats_rank0.csv';replace(x,n,x['files'][n]['text'].replace(',1000000',',nan'));self.assertRaises(j.InvalidArchive,run,x)
    def test_base64(self):
        x=fixture()
        for f in x['files'].values():f['base64']=base64.b64encode(f.pop('text').encode()).decode()
        self.assertEqual(run(x)['status'],'MATCHED_CLOSED_FULL36_INPUTS_ONLY')
    def test_missing_value_and_missing_harness(self):
        x=fixture();n='metrics-training/metric_table_graph_stats_rank0.csv';replace(x,n,'graph_id,graph_name,graph_node_count,torch_neuronx_compile_us\n'+j.CAPTURES[0]['module_sha256']+',x\n');self.assertRaises(j.InvalidArchive,run,x)
        x=fixture();del x['files']['train/rehearsal.json'];x.pop('file_count');x.pop('raw_bytes');self.assertEqual(run(x)['status'],'UNKNOWN')
    def test_duplicate_json_and_wrong_counts(self):
        self.assertRaises(j.InvalidArchive,j.parse,b'{"files":{},"files":{}}');x=fixture();x['file_count']+=1;self.assertRaises(j.InvalidArchive,run,x)
    def test_conflicting_same_rank(self):
        x=fixture();n='metrics-training/metric_table_graph_stats_rank0.csv';txt=x['files'][n]['text'];replace(x,n,txt+txt.splitlines()[1]+'\n');self.assertEqual(run(x)['status'],'UNKNOWN')
if __name__=='__main__':unittest.main(verbosity=2)
