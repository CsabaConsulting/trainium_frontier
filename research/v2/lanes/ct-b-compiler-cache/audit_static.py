#!/usr/bin/env python3
"""Read existing raw CT evidence; no torch import, tensors, host access or compiler."""
import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import types

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RAW = ROOT / 'research/v2/operator/20260925-trn22-l2-93ae-cpu-r1'
SOURCE_COMMIT = '1e21c3d797eed764f108c41546b270065ad5650c'
checks = []
def check(name, value):
    if not value:
        raise RuntimeError(name)
    checks.append(name)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

source = subprocess.check_output(['git','show',SOURCE_COMMIT+':workflow/proposals/research-v2/train.py'],cwd=ROOT)
check('exact original architecture source',hashlib.sha256(source).hexdigest() == '93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5')
tree = ast.parse(source)
fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'make_plan')
ns = {'torch': types.SimpleNamespace(float32='fp32')}
exec(compile(ast.Module(body=[fn], type_ignores=[]),'<exact make_plan shape-only>','exec'),ns)
class ParameterShape:
    dtype = 'fp32'
    def __init__(self, shape): self.shape = tuple(shape)
    def numel(self): return math.prod(self.shape)
    def is_contiguous(self): return True

# Exact declared architecture has attention on 0,2,4,6,7, MLP on all eight,
# one last-layer value embedding/8x32 gate, 20 scalar parameters and five
# 1024x1x2 trained convolutions. Preserve setup_optimizer's group order.
groups = []
def group(kind, shape, count):
    groups.append({'kind':kind, 'params':[ParameterShape(shape) for _ in range(count)]})
group('adamw',(8192,1024),1) # wte
group('adamw',(8192,1024),1) # head
group('adamw',(1,),20)
group('adamw',(8192,1024),1) # value table
group('adamw',(1024,1,2),5)
for shape,count in [((8,32),1),((1024,1024),20),((1024,4096),8),((4096,1024),8)]:
    group('muon',shape,count)
entries,params,summary = ns['make_plan'](groups,4)
raw_rank = json.loads((RAW/'candidate-on/results/rank0.json').read_text())
check('shape-only reconstruction has all 65 actual parameters',len(params)==raw_rank['n_params']==65)
check('shape-only exact source ownership matches raw G2 summary',summary==raw_rank['plan_summary'])
owned=[]
for rank in range(4):
    shapes=[]
    for gi,g in enumerate(groups):
        if g['kind']!='muon': continue
        n=sum(e['owner']==rank and e['group']==gi for e in entries)
        if n:
            shapes.append({'matrix_shape':list(g['params'][0].shape),'owned_count':n,
                           'compiled_stack_count':2 if n==1 else n})
    owned.append({'rank':rank,'muon_stacks':shapes})

rows=[]; inputs={}
for rank in range(4):
    p=RAW/f'native/completed-training/metrics-training/metric_table_graph_stats_rank{rank}.csv'
    inputs[str(p.relative_to(ROOT))]=digest(p)
    with p.open() as f:
        rankrows=list(csv.DictReader(f))
    for row in rankrows:
        row['rank']=rank
        row['reported_compile_s']=int(row['torch_neuronx_compile_us'])/1e6
    check(f'rank{rank} model graphs have no in-graph collectives',all(r['has_collectives']=='False' for r in rankrows))
    rows+=rankrows
byid=collections.defaultdict(list)
for row in rows: byid[row['graph_id']].append(row)
model=[]; small=[]
for gid,rs in sorted(byid.items()):
    item={'graph_id':gid,'ranks':[r['rank'] for r in rs],
          'reported_compile_s_by_rank':{str(r['rank']):r['reported_compile_s'] for r in rs},
          'node_count':int(rs[0]['graph_node_count']),
          'names':[r['graph_name'] for r in rs]}
    if any('_forward_' in r['graph_name'] or '_backward_' in r['graph_name'] for r in rs):
        check('model graph all ranks '+gid[:8],item['ranks']==[0,1,2,3])
        check('one nonzero reported compiler timer '+gid[:8],sum(r['reported_compile_s']>0 for r in rs)==1)
        model.append(item)
    else: small.append(item)
check('four distinct model graph IDs',len(model)==4)
check('eight small inference graph IDs / thirteen occurrences',len(small)==8 and sum(len(r['ranks']) for r in small)==13)
capture=RAW/'native/live-compiler-input/capture.json'
cap=json.loads(capture.read_text())
mlir=RAW/'native/live-compiler-input/module.mlir'
check('archived input bytes match live capture',digest(mlir)==cap['input_sha256'])
check('captured compiler input identifies first model forward',cap['input_sha256']+':cd47b8171ac56a23cc896e6ceb62fa0bfd9e8cf0bf9b08bd2a965094c3965433' in byid)
argv=cap['live_compiler_processes'][0]['argv']
check('model-type transformer is already requested',argv[argv.index('--model-type')+1]=='transformer')
check('optlevel1 and preceding O2 both recorded','--optlevel=1' in argv and '-O2' in argv and argv.index('-O2')<argv.index('--optlevel=1'))
for p in [capture,mlir,RAW/'native/live-compiler-input/compiler-log-prefix.txt',RAW/'candidate-on/results/rank0.json']:
    inputs[str(p.relative_to(ROOT))]=digest(p)
result={'status':'STATIC_EVIDENCE_ONLY_NO_NATIVE_GAIN', 'checks':checks,
        'source_commit':SOURCE_COMMIT,'source_sha256':hashlib.sha256(source).hexdigest(),
        'inputs_sha256':inputs,'model_graphs':model,'small_graphs':small,
        'shape_only_ownership':owned,'ownership_summary':summary,
        'reported_unique_model_compile_seconds':sum(sum(g['reported_compile_s_by_rank'].values()) for g in model),
        'reported_unique_small_compile_seconds':sum(sum(g['reported_compile_s_by_rank'].values()) for g in small),
        'limits':['not actual cold startup','zero metric alone is not persistent cache proof',
                  'small graph key to exact Muon stack mapping not present in raw IR',
                  'no torch import or tensor allocation','no compiler execution or host calls']}
(OUT/'STATIC_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':result['status'],'checks':len(checks),
                  'owned_stacks':owned,'model_compile_s':result['reported_unique_model_compile_seconds'],
                  'small_compile_s':result['reported_unique_small_compile_seconds']},sort_keys=True))
