import os, sys, json, hashlib, struct, random, copy, importlib.util, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'repair-review-output'
def digest(b): return hashlib.sha256(b).hexdigest()
diff=(ROOT/'corrected-source.diff').read_text()
sources={}
current=None
for line in diff.splitlines(keepends=True):
    if line.startswith('+++ b/'):
        current=line.strip().rsplit('/',1)[1]; sources[current]=[]
    elif current and line.startswith('+') and not line.startswith('+++'):
        sources[current].append(line[1:])
source=''.join(sources['ordinary_subnormal_probe.py'])
path=OUT/'reviewed_probe.py'
path.write_text(source)
spec=importlib.util.spec_from_file_location('reviewed_probe',path)
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
import torch
import torch.nn.functional as F
torch.set_num_threads(1); torch.set_num_interop_threads(1)
assert torch.get_num_threads()==1
report={'source_sha256':digest(source.encode()),'torch':torch.__version__,'native':False,'fixtures':{},'negative_tests':{}}
# Independent encoding uses exact Python binary64 ldexp-equivalent powers and
# struct's IEEE binary32 conversion. All coefficients have <=9 significant bits.
# No candidate encoder, fixture, or device reduction computes our expected bytes.
for case in ['sentinel','subnormal','normal','negated']:
    ids=list(range(256)) if case=='sentinel' else [i%11 for i in range(256)]
    r=random.Random(42)
    coeff=[r.randint(-256,256) for _ in range(32768)] if case=='sentinel' else [(-1 if (t+c)%3==0 else 1)*(-1 if case=='negated' else 1) for t in range(256) for c in range(128)]
    exponent={'sentinel':-5,'normal':-126}.get(case,-149)
    sums=[sum(coeff[t*128+c] for t in range(256) if ids[t]==row) for row in range(256) for c in range(128)]
    own={'ids':struct.pack('<256i',*ids),'tangent':b''.join(struct.pack('<f',n*2.0**exponent) for n in coeff),'gradient':b''.join(struct.pack('<f',n*2.0**exponent) for n in sums)}
    assert own==p.fixture(case),case
    for key,raw in own.items(): (OUT/(case+'-'+key+'.bin')).write_bytes(raw)
    report['fixtures'][case]={k:digest(v) for k,v in own.items()}
    if case=='subnormal':
        assert {n:sums.count(n) for n in set(sums)}=={0:31360,7:683,8:384,9:341}
        assert report['fixtures'][case]==p.PINS
    # Independent CPU autograd equality, including untouched positive zero.
    w=torch.arange(32768,dtype=torch.float32).reshape(256,128).requires_grad_()
    ix=torch.tensor(ids,dtype=torch.int32)
    tangent=torch.frombuffer(bytearray(own['tangent']),dtype=torch.float32).reshape(256,128)
    F.embedding(ix,w).backward(tangent)
    assert w.grad.contiguous().view(torch.uint8).numpy().tobytes()==own['gradient']
torch.manual_seed(42)
weight=torch.randn(256,128)
receipt=dict(label=p.LABEL,rank=0,world_size=4,error=None,cleanup='DESTROYED',original_K2_gate='FAILED_PRESERVED',compiled_callable_count=1,returncode=0,routes=[],blobs={})
receipt['weight']=p.tensor_record(weight,lambda:None,receipt['blobs'])
for case in p.CASES:
    for route in p.ROUTES:
        receipt['routes'].append(p.run_route(lambda w,i:F.embedding(i,w).to(torch.float32),route,case,weight,p.fixture(case),torch.device('cpu'),lambda:None,receipt['blobs']))
assert p.validate_receipt(receipt,0)
(OUT/'baseline-cpu-receipt.json').write_text(json.dumps(receipt,sort_keys=True,allow_nan=False))
def mutated_gradient(r,bits):
    raw=bytearray(__import__('base64').b64decode(r['blobs'][r['routes'][2]['gradient']['actual']['sha256']]))
    raw[-4:]=struct.pack('<I',bits)
    actual=p.blob(bytes(raw),'torch.float32',(256,128),r['blobs'])
    r['routes'][2]['gradient']=p.comparison(actual,r['routes'][2]['expected_gradient'],r['blobs'])
    r['routes'][2]['gradient']['exact']=True
tests={
 'wrong_input':lambda r:r['routes'][0]['before_backward'].update(tangent=r['routes'][2]['before_backward']['tangent']),
 'wrong_route':lambda r:r['routes'][0].update(route='ordinary_compiled'),
 'wrong_rank':lambda r:r.update(rank=1),
 'boolean_rank':lambda r:r.update(rank=False),
 'missing_raw_bytes':lambda r:r['blobs'].pop(r['routes'][0]['expected_inputs']['ids']['sha256']),
 'missing_gradient':lambda r:r['routes'][2]['gradient'].update(actual=None),
 'signed_zero':lambda r:mutated_gradient(r,0x80000000),
 'nan':lambda r:mutated_gradient(r,0x7fc00001),
 'positive_infinity':lambda r:mutated_gradient(r,0x7f800000),
 'negative_infinity':lambda r:mutated_gradient(r,0xff800000),
 'stale_positive':lambda r:r['routes'][6].update(gradient=r['routes'][2]['gradient']),
 'zero_gradient':lambda r:r['routes'][2]['gradient'].update(actual=p.blob(bytes(131072),'torch.float32',(256,128),r['blobs'])),
 'wrong_case':lambda r:r['routes'][2].update(case='negated'),
 'missing_route':lambda r:r['routes'].pop(),
 'boolean_metric':lambda r:r['routes'][0]['gradient'].update(max_abs=False),
 'cleanup_error':lambda r:r.update(cleanup='ERROR'),
}
for name,mutate in tests.items():
    changed=copy.deepcopy(receipt); mutate(changed)
    try: p.validate_receipt(changed,0)
    except Exception as e: report['negative_tests'][name]={'rejected':True,'error':type(e).__name__+': '+str(e)}
    else: report['negative_tests'][name]={'rejected':False}
assert all(x['rejected'] for x in report['negative_tests'].values())
for bits in [0x7fc00001,0x7f800000,0xff800000]:
    blobs={}; actual=p.blob(struct.pack('<I',bits),'torch.float32',(1,),blobs); expected=p.blob(bytes(4),'torch.float32',(1,),blobs)
    result=p.comparison(actual,expected,blobs)
    assert result['exact'] is False and result['max_abs']=='NONFINITE'
    json.dumps(result,allow_nan=False)
disabled=subprocess.run([sys.executable,'-I','-B',str(path)],text=True,capture_output=True)
report['disabled']={'returncode':disabled.returncode,'stdout':disabled.stdout,'stderr':disabled.stderr}
assert disabled.returncode==0 and json.loads(disabled.stdout)=={'status':'DISABLED','native_imported':False}
report['status']='OWN_CPU_PASS'
(OUT/'results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))
