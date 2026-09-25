"""Own 1x1 CPU nonfinite-receipt reproduction; no native imports or model."""
import ast
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

REPO=Path('/home/poojith-devan/Desktop/Trainium')
OLD='077481d0476cebfd79ddc751f8f63e61c1fbf82f'
NEW='18b9365aabb63fa85e1720fdefc11177ca8e7853'
PREFIX='research/v2/lanes/k2-resident-consumer/'
checks=[]; cases=[]; finalizers=[]
def check(name,value):
    if not value:raise RuntimeError('FAIL: '+name)
    checks.append(name)
def blob(name,ref):return subprocess.check_output(['git','-C',str(REPO),'show',ref+':'+PREFIX+name])
def sha(data):return hashlib.sha256(data).hexdigest()
old,new=blob('check_native.py',OLD),blob('check_native.py',NEW)
check('exact repaired checker',sha(new)=='89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850')
trees=[ast.parse(x) for x in (old,new)]
main=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main')
nested=lambda t,name:next(n for n in main(t).body if isinstance(n,ast.FunctionDef) and n.name==name)
outer=lambda t:next(n for n in main(t).body if isinstance(n,ast.Try) and n.finalbody)
normalized=copy.deepcopy(trees[1])
normalized.body=[n for n in normalized.body if not (isinstance(n,ast.Import) and len(n.names)==1 and n.names[0].name=='math')]
nested(normalized,'compare').body=copy.deepcopy(nested(trees[0],'compare').body)
outer(normalized).finalbody=copy.deepcopy(outer(trees[0]).finalbody)
check('whole AST only compare reporting and finalizer',ast.dump(normalized)==ast.dump(trees[0]))
for name in ('consumer_nki.py','reference.py','test_reference.py'):
    check('unchanged '+name,blob(name,OLD)==blob(name,NEW))
for tree in trees:
    constants=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant)]
    check('byte criterion unchanged '+str(len(checks)),'identical bytes' in constants)

import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
helpers=[]
for tree in trees:
    ns={'torch':torch,'hashlib':hashlib,'math':math}
    exec(compile(ast.Module(body=[nested(tree,'tensor_bytes'),nested(tree,'compare')],type_ignores=[]),'exact-comparison-helpers','exec'),ns)
    helpers.append(ns)
for label,actual,expected in [('positive infinity',float('inf'),0.),('negative infinity',-float('inf'),0.),('quiet NaN',float('nan'),0.),('finite exact',1.,1.),('finite mismatch',1.,2.),('signed zero',-0.,0.),('tiny finite mismatch',2**-20,0.)]:
    a=torch.tensor([[actual]],dtype=torch.float32);e=torch.tensor([[expected]],dtype=torch.float32)
    records=[h['compare'](a,e,label) for h in helpers]
    failures=[]
    for rec in records:
        try:json.dumps(rec,allow_nan=False);failures.append(None)
        except ValueError as exc:failures.append(str(exc))
    nonfinite=not math.isfinite(actual)
    if nonfinite:
        check(label+' original strict JSON fails',failures[0] is not None)
        check(label+' successor refuses with JSON-safe metrics',failures[1] is None and records[1]['exact'] is False and records[1]['finite'] is False and records[1]['max_abs'] is None and records[1]['max_abs_nonfinite'] in ('nan','inf'))
        old_copy=dict(records[0]);new_copy=dict(records[1]);old_copy.pop('max_abs');new_copy.pop('max_abs');new_copy.pop('max_abs_nonfinite')
        check(label+' all nonmetric fields preserved',old_copy==new_copy)
    else:
        check(label+' records unchanged',records[0]==records[1] and failures==[None,None])
        if label=='signed zero':check('signed zero is byte mismatch despite zero absolute error',records[1]['exact'] is False and records[1]['max_abs']==0.)
    def safe(rec):return {k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in rec.items()}
    cases.append({'label':label,'actual_bits':a.view(torch.int32).item() & 0xffffffff,'expected_bits':e.view(torch.int32).item() & 0xffffffff,'original_record_sanitized_for_archive':safe(records[0]),'repaired_record':records[1],'original_serialization_error':failures[0],'repaired_serialization_error':failures[1]})

# Execute both exact finalizers on real temporary output files. Faults are local
# serialization/exclusive-file/stdout failures, not simulated native success.
for version,tree in [('original',trees[0]),('repaired',trees[1])]:
    scenarios=['nonfinite','finite','invalid other metric','existing file','broken stdout','write error']
    for scenario in scenarios:
        with tempfile.TemporaryDirectory(prefix='k2-report-review-') as temp:
            directory=Path(temp);path=directory/'rank0.json';cleanup=[];printed=[]
            record=copy.deepcopy(cases[4 if scenario=='finite' else 0]['repaired_record'])
            if version=='original' and scenario=='nonfinite':record['max_abs']=float('inf');record.pop('max_abs_nonfinite')
            receipt={'status':'REFUSE_OR_UNSUPPORTED','cases':[{'status':'FAIL_EXACTNESS','checks':[record]}]}
            if scenario=='invalid other metric':receipt['unrelated_bad_metric']=float('nan')
            if scenario=='existing file':path.write_text('preserved\n')
            def output(value,**kw):
                if scenario=='broken stdout':raise BrokenPipeError('injected stdout failure')
                printed.append(value)
            class BrokenWriter:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def write(self,_):raise OSError('injected disk write failure')
            class BrokenPath:
                def is_dir(self):return True
                def __truediv__(self,_):return self
                def open(self,*args):return BrokenWriter()
            ns={'receipt':receipt,'time':SimpleNamespace(monotonic=lambda:2.),'started':1.,'args':SimpleNamespace(out_dir=BrokenPath() if scenario=='write error' else directory),'rank':0,'json':json,'print':output,'dist':SimpleNamespace(destroy_process_group=lambda:cleanup.append(True))}
            error=None
            try:exec(compile(ast.Module(body=outer(tree).finalbody,type_ignores=[]),'exact-finalizer','exec'),ns)
            except Exception as exc:error=type(exc).__name__+': '+str(exc)
            contents=path.read_text() if path.exists() else None
            valid=False
            if contents is not None:
                try:valid=json.loads(contents)['status']=='REFUSE_OR_UNSUPPORTED'
                except (ValueError,TypeError,KeyError):pass
            if version=='repaired':
                check('repaired cleanup '+scenario,cleanup==[True])
                if scenario in ('nonfinite','finite'):check('complete refusal '+scenario,error is None and valid and len(printed)==1)
                if scenario=='invalid other metric':check('encode before file creation',error.startswith('ValueError:') and contents is None)
                if scenario=='existing file':check('existing file preserved',error.startswith('FileExistsError:') and contents=='preserved\n')
                if scenario=='broken stdout':check('stdout error does not erase valid file',error.startswith('BrokenPipeError:') and valid)
                if scenario=='write error':check('write error stays failure',error.startswith('OSError:'))
            elif scenario=='nonfinite':check('original malformed file and skipped cleanup reproduced',error.startswith('ValueError:') and contents is not None and not valid and not cleanup)
            finalizers.append({'version':version,'scenario':scenario,'error':error,'cleanup_calls':len(cleanup),'valid_refusal_file':valid,'file_contents':contents,'stdout_count':len(printed)})
print(json.dumps({'verdict':'PASS_REPORTING_REPAIR_ONLY','checks':len(checks),'tests':checks,'original_commit':OLD,'repaired_commit':NEW,'checker_sha256':sha(new),'torch':torch.__version__,'threads':torch.get_num_threads(),'largest_tensor_shape':[1,1],'cases':cases,'finalizers':finalizers,'native_executed':False,'model_instantiated':False,'original_source_REFUSE_unchanged':True,'native_binding_stale_disabled':True,'native_launch_authorized':False},indent=2,allow_nan=False))
