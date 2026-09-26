"""Research evidence only: tiny cast-domain arithmetic, no candidate or SDK import."""
import ast,base64,hashlib,json,math,struct,subprocess,sys
from pathlib import Path
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT=Path('/home/poojith-devan/Desktop/Trainium')
PIN='e018498e4bdd91ce5c93a184c3e7602a42228311'
def need(value,label):
    if not value:raise RuntimeError(label)
def blob(path,rev=PIN):return subprocess.check_output(['git','show',rev+':'+path],cwd=ROOT)
def sha(x):return hashlib.sha256(x).hexdigest()
def f32(x):
    try:return struct.unpack('<f',struct.pack('<f',x))[0]
    except OverflowError:return math.copysign(math.inf,x)
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def hx(x):return f'{bits(x):08x}'
q=2.**-133
finite_count=0
for u in range(65536):
    x=struct.unpack('<f',struct.pack('<I',u<<16))[0]
    if math.isfinite(x):
        need((x/q).is_integer(),'BF16_LATTICE');finite_count+=1
source=blob('workflow/proposals/research-v2/train.py','7419927c7417b5666a984ec9c2b849feadabaaa9')
need(sha(source)=='36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb','SOURCE')
tree=ast.parse(source)
gpt=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPT')
forward=next(n for n in gpt.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
calls=[ast.unparse(n) for n in ast.walk(forward) if isinstance(n,ast.Call)]
for required in ['self.transformer.wte(idx).to(COMPUTE_DTYPE)','self.value_embeds[str(i)](idx).to(COMPUTE_DTYPE)']:need(required in calls,required)
inputs={
 'min_positive':[q], 'min_negative':[-q], 'cancel_to_min':[2**-126+q,-2**-126],
 'cancel_to_negative_min':[-2**-126-q,2**-126], 'alternating_min':[q,-q,q],
 'cross_tile_rounding':[2**24,1.,-2**24], 'small_rounding':[1.,2**-24,-1.],
 'signed_zero':[-0.,0.,-0.], 'max_cancel':[float.fromhex('0x1.fep127'),-float.fromhex('0x1.fep127'),q],
 'overflow':[float.fromhex('0x1.fep127')]*2,
 'positive_infinity':[math.inf], 'negative_infinity':[-math.inf],
 'opposite_infinities':[math.inf,-math.inf], 'nan':[math.nan],
 'rounded_fp32_min':[2.**-149], 'rounded_normal_cancel':[2**-126+2**-149,-2**-126],
}
rows=[]
for name,values in inputs.items():
    upstream=torch.tensor(values,dtype=torch.bfloat16).reshape(-1,1)
    weight=torch.ones((3,1),dtype=torch.float32,requires_grad=True)
    ids=torch.ones(len(values),dtype=torch.long)
    raw=F.embedding(ids,weight); out=raw.to(torch.bfloat16)
    seen={}
    raw.register_hook(lambda g:seen.update(dtype=str(g.dtype),bits=[hx(v) for v in g.flatten().tolist()]))
    out.backward(upstream)
    promoted=upstream.float().flatten().tolist();acc=0.;history=[]
    for v in promoted:
        acc=f32(acc+v);history.append(hx(acc))
        if math.isfinite(acc):need((acc/q).is_integer(),'RECURRENCE_LATTICE')
    actual=weight.grad[1,0].item()
    if math.isnan(acc):need(math.isnan(actual),'NAN_PRESERVED')
    else:need(bits(actual)==bits(acc),'ORDERED_VALUE')
    need(seen['dtype']=='torch.float32' and seen['bits']==[hx(v) for v in promoted],'WIDENING')
    need(bits(weight.grad[0,0].item())==bits(weight.grad[2,0].item())==0,'UNTOUCHED')
    rows.append(dict(name=name,input_fp32_bits=[hx(v) for v in values],promoted_bits=[hx(v) for v in promoted],intermediates=history,actual=hx(actual),training_acceptance='FAIL_NONFINITE' if not math.isfinite(actual) else 'FINITE_EXACT'))
archive_bytes=blob('research/v2/operator/20260926-trn22-k2-registered/completed/results-bundle.json')
archive=json.loads(archive_bytes);native=[]
for rank in range(4):
    ent=archive['files'][f'receipts/rank{rank}.json'];data=base64.b64decode(ent['base64'])
    need(len(data)==ent['bytes'] and sha(data)==ent['sha256'],'RECEIPT_HASH')
    receipt=json.loads(data)
    need(receipt['status']=='REGISTERED_FULL_CASES_MISMATCH_OR_ERROR','FAILED_AGGREGATE')
    selected=[]
    for case in receipt['cases']:
        if case['name'] in ('bf16_subnormal','fp32_subnormal'):
            selected.append(dict(name=case['name'],dtype=case['dtype'],status=case['status'],input_nonzero=case['checks']['upstream_roundtrip']['actual']['nonzero'],routes=[dict(name=r['name'],status=r['status'],nonzero=r.get('checks',{}).get('gradient',{}).get('actual',{}).get('nonzero')) for r in case['routes']]))
    native.append(dict(rank=rank,sha256=sha(data),original_status=receipt['status'],cases=selected))
paths=['research/v2/lanes/k2-symmetric-consumer/consumer_nki.py','research/v2/lanes/k2-registered-forward-autograd/adapter.py','research/v2/analysis/20260926-k2-cast-boundary/audit.py','research/v2/analysis/20260926-k2-cast-boundary/result.json']
print(json.dumps(dict(status='RESEARCH_ONLY_NOT_GATE_PASS',interpreter=sys.executable,torch=torch.__version__,pin=PIN,source_sha256=sha(source),inputs={p:sha(blob(p)) for p in paths},finite_bf16_encodings_checked=finite_count,cpu_cases=rows,native_archive_sha256=sha(archive_bytes),native=native,largest_tensor_elements=max(len(v) for v in inputs.values()),no_native_launch=True),indent=2,allow_nan=False))
