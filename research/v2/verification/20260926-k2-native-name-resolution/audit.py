"""Read-only stdlib authentication and Python scope inspection; no native import."""
import argparse
import ast
import dis
import hashlib
import json
from pathlib import Path
import subprocess
import types

REPO=Path(__file__).resolve().parents[4]
RAW='94fb52c4'
RAWPATH='research/v2/operator/20260926-trn22-k2-reporting-native/completed/results-bundle.json'
SOURCE='18b9365aabb63fa85e1720fdefc11177ca8e7853'
PREFIX='research/v2/lanes/k2-resident-consumer/'
checks=[]
def check(name,value):
    if not value:raise RuntimeError(name)
    checks.append(name)
def blob(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)
def sha(raw):return hashlib.sha256(raw).hexdigest()
raw=blob(RAW,RAWPATH);bundle=json.loads(raw)
check('frozen raw bundle',sha(raw)=='129b9e69352a63d49922dc386c849836ad829dcbb85ef3e33b50b28b64cf9e04')
for name,item in bundle['files'].items():
    content=item['text'].encode()
    check('retrieved artifact '+name,len(content)==item['bytes'] and sha(content)==item['sha256'])
check('complete retrieval totals',len(bundle['files'])==bundle['file_count']==27 and sum(x['bytes'] for x in bundle['files'].values())==bundle['raw_bytes']==113871)
def record(name):return json.loads(bundle['files'][name]['text'])
kernel=blob(SOURCE,PREFIX+'consumer_nki.py');checker=blob(SOURCE,PREFIX+'check_native.py');reference=blob(SOURCE,PREFIX+'reference.py')
pins={'kernel_sha256':sha(kernel),'checker_sha256':sha(checker),'reference_sha256':sha(reference)}
check('exact kernel',pins['kernel_sha256']=='57b5b118e86eb9d78e26a011b276904aed9065868b8a12f39169e1257c3a1843')
check('exact checker',pins['checker_sha256']=='89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850')
check('kernel unchanged from original',kernel==blob('077481d0476cebfd79ddc751f8f63e61c1fbf82f',PREFIX+'consumer_nki.py'))
ranks=[]
for rank in range(4):
    r=record(f'receipts/rank{rank}.json');env=record(f'rank-environment-r{rank}.json')
    check('rank source/runtime '+str(rank),r['rank']==env['rank']==rank and r['world_size']==4 and r['nki_version']=='0.5.0+28631259367.ga768afa6' and r['torch']=='2.11.0+cpu' and all(r[k]==v for k,v in pins.items()))
    check('rank frontend refusal '+str(rank),r['status']=='REFUSE_OR_UNSUPPORTED' and "failed to resolve name 'nl.bfloat16'" in r['error'] and 'line 133, in main' in r['error'] and 'nki/compiler/frontend.py(300)' in r['error'])
    check('rank no consumer outcome '+str(rank),len(r['cases'])==1 and r['cases'][0]['name']=='cross_tile_large' and r['cases'][0]['dtype']=='torch.bfloat16' and r['cases'][0]['status']=='STARTED' and r['timings']==[])
    upstream=r['cases'][0]['checks']
    check('rank upstream only '+str(rank),len(upstream)==1 and upstream[0]['name']=='upstream_roundtrip' and upstream[0]['exact'] is upstream[0]['finite'] is True and upstream[0]['max_abs']==0 and upstream[0]['actual_sha256']==upstream[0]['expected_sha256'])
    check('rank no acceptance '+str(rank),r['accepted_speedup'] is r['source_adoption'] is r['launch_authorized'] is False and r['tolerance']=={'atol':0,'rtol':0,'criterion':'identical bytes'})
    ranks.append({'rank':rank,'status':r['status'],'nki_version':r['nki_version'],'case':r['cases'][0]['name'],'checker_call_line':133,'completed_checks':[x['name'] for x in upstream],'timings':r['timings']})
check('terminal failed',record('terminal.json')['status']=='DIAGNOSTIC_CHILD_FAILED' and record('terminal.json')['returncode']==1 and record('terminal.json')['complete'] is False)
ktree=ast.parse(kernel)
factory=next(n for n in ktree.body if isinstance(n,ast.FunctionDef) and n.name=='build_kernel')
inner=next(n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name=='resident_consumer')
bf16=[n for n in ast.walk(inner) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='nl' and n.attr=='bfloat16']
check('precise kernel dtype source site',len(bf16)==1 and bf16[0].lineno==22)
check('no global namespace declaration',not any(isinstance(n,ast.Global) for n in factory.body) and not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ktree.body))
# Inspect the unoptimized source assertion even when this audit runs with -O.
module_code=compile(kernel,'frozen-consumer.py','exec',optimize=0)
factory_code=next(c for c in module_code.co_consts if isinstance(c,types.CodeType) and c.co_name=='build_kernel')
inner_code=next(c for c in factory_code.co_consts if isinstance(c,types.CodeType) and c.co_name=='resident_consumer')
check('both API aliases are closure cells',set(inner_code.co_freevars)=={'nl','nisa'} and 'nl' in factory_code.co_cellvars and 'nisa' in factory_code.co_cellvars)
ops=[{'op':i.opname,'arg':i.argval,'line':i.positions.lineno} for i in dis.get_instructions(inner_code) if i.argval in ('nl','nisa','bfloat16')]
check('nl resolves through closure in ordinary Python',any(x['op']=='LOAD_DEREF' and x['arg']=='nl' and x['line']==22 for x in ops) and not any(x['op']=='LOAD_GLOBAL' and x['arg']=='nl' for x in ops))
baseline=blob('b4d83b5db66971f3c348837f04f53cba65aa6d33','workflow/proposals/research-v2/train.py')
btree=ast.parse(baseline);bfactory=next(n for n in btree.body if isinstance(n,ast.FunctionDef) and n.name=='build_kernel')
check('existing model factory explicitly binds API globals',any(isinstance(n,ast.Global) and set(n.names)=={'nl','nisa'} for n in bfactory.body))
check('checker133 is dispatch not dtype definition',checker.decode().splitlines()[132].strip()=='actual = kernel(device_upstream, *device_meta)')
result={'status':'NATIVE_FRONTEND_FAILURE_CONFIRMED_NO_NUMERICAL_RESULT','checks':len(checks),'raw_commit':RAW,'raw_bundle_sha256':sha(raw),'source_commit':SOURCE,'pins':pins,'ranks':ranks,'source_scope':{'kernel_dtype_line':22,'checker_dispatch_line':133,'freevars':list(inner_code.co_freevars),'api_namespace_loads':ops,'source_scope_difference_confirmed':True},'inference':'Factory-local API aliases are closure-only, unlike the existing model factory globals; this is the strongest local explanation for the observed frontend name-resolution error. Exact installed frontend namespace implementation is not archived, so closure non-support is not independently proved here.','installed_dtype_inventory_available':False,'native_imported_or_retried':False,'tensors':False,'hosts':False,'candidate_edited':False,'native_correctness':'UNPROVEN','speedup':'UNMEASURED','checks_passed':checks}
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('source_scope','checks_passed')},indent=2))
