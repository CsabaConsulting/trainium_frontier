import hashlib, json, pathlib, subprocess, sys
import torch

ROOT = pathlib.Path(__file__).parent
REPO = '/home/poojith-devan/Desktop/Trainium'
def gitbody(rev, path):
    return subprocess.check_output(['git','-C',REPO,'show',rev+':'+path])
source = gitbody('7419927c7417b5666a984ec9c2b849feadabaaa9','workflow/proposals/research-v2/train.py')
source_hash = hashlib.sha256(source).hexdigest()
if source_hash != '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb':
    raise RuntimeError('source hash mismatch')

# Own integer IEEE-754 binary32 round-to-nearest-even reference.
# Finite values represented in exact integer units of 2**-149, never FP64.
def units(bits):
    exp = (bits >> 23) & 255
    mant = bits & 0x7fffff
    if exp == 255: raise ValueError('nonfinite')
    n = mant if exp == 0 else ((1<<23)|mant) << (exp-1)
    return -n if bits>>31 else n
def encode(n):
    sign = 0x80000000 if n < 0 else 0
    n = abs(n)
    if not n: return 0
    shift = max(0,n.bit_length()-24)
    q, rem = divmod(n, 1<<shift)
    if shift and (rem > (1<<(shift-1)) or (rem == (1<<(shift-1)) and q&1)):
        q += 1
    if q == 1<<24:
        q >>= 1; shift += 1
    if q < 1<<23: return sign|q
    exp = shift+1
    if exp >= 255: return sign|0x7f800000
    return sign|(exp<<23)|(q-(1<<23))
def add(a,b):
    if a&0x7f800000 == 0x7f800000 or b&0x7f800000 == 0x7f800000:
        if (a&0x7fffffff)>0x7f800000 or (b&0x7fffffff)>0x7f800000: return 0x7fc00000
        if (a&0x7fffffff)==0x7f800000 and (b&0x7fffffff)==0x7f800000 and (a^b)>>31: return 0x7fc00000
        return a if a&0x7fffffff==0x7f800000 else b
    n=units(a)+units(b)
    return 0x80000000 if n==0 and a==b==0x80000000 else encode(n)
def tensor(bits):
    return torch.tensor(bits if bits<1<<31 else bits-(1<<32),dtype=torch.int32).view(torch.float32)
def raw(t): return t.view(torch.int32).item() & 0xffffffff
def recurrence(bf):
    a=0; trace=[]; actual=torch.zeros((),dtype=torch.float32)
    for b in bf:
        wide=b<<16
        tangent=tensor(wide).to(torch.bfloat16).to(torch.float32)
        if wide&0x7fffffff<=0x7f800000 and raw(tangent)!=wide: raise RuntimeError('widen changed raw bits')
        a=add(a,wide); actual=actual+tangent
        observed=raw(actual)
        nan=lambda x: (x&0x7fffffff)>0x7f800000
        if not (nan(a) and nan(observed)) and a!=observed: raise RuntimeError('oracle mismatch')
        if a&0x7f800000!=0x7f800000 and units(a)%65536: raise RuntimeError('lattice violation')
        trace.append({'reference':f'{a:08x}','cpu':f'{observed:08x}','cpu_widened_tangent':f'{raw(tangent):08x}'})
    return trace
cases={
 'positive_min':[0x0001], 'negative_min':[0x8001],
 'positive_cancellation':[0x0081,0x8080], 'negative_cancellation':[0x8081,0x0080],
 'cross_tile_large':[0x4b80,0x3f80,0xcb80],
 'cross_tile_small':[0x3f80,0x3380,0xbf80],
 'large_then_cancel_then_subnormal':[0x4b80,0xcb80,0x0001],
 'negative_zero_input':[0x8000], 'both_zero_signs':[0x8000,0x0000],
 'positive_overflow':[0x7f7f,0x7f7f], 'negative_overflow':[0xff7f,0xff7f],
 'overflow_then_opposite_finite':[0x7f7f,0x7f7f,0xff7f],
 'infinity_cancellation':[0x7f80,0xff80], 'nan_input':[0x7fc1]
}
results={name:{'inputs_bf16':[f'{b:04x}' for b in bf],'trace':recurrence(bf)} for name,bf in cases.items()}
grouped={}
for name in ('cross_tile_large','cross_tile_small'):
    x,y,z=[b<<16 for b in cases[name]]
    grouped[name]={'ordered':f'{add(add(x,y),z):08x}','reassociated':f'{add(x,add(y,z)):08x}'}
    if grouped[name]['ordered']==grouped[name]['reassociated']: raise RuntimeError('lost witness')
finite_count=0
for b in range(65536):
    if b&0x7f80==0x7f80: continue
    n=units(b<<16)
    if n%65536: raise RuntimeError('finite BF16 off lattice')
    if n and encode(n)!=b<<16: raise RuntimeError('roundtrip mismatch')
    finite_count+=1
if add(0x80000000,0x80000000)!=0x80000000: raise RuntimeError('negative zero addition')
result={'verdict':'CPU_SPEC_MATH_SUPPORTED_NOT_IMPLEMENTATION_OR_NATIVE_APPROVAL',
 'source_sha256':source_hash,'python':sys.version,'torch':torch.__version__,
 'finite_BF16_integer_encodings_checked':finite_count,'max_tensor_elements':1,
 'cases':results,'reassociation_witnesses':grouped,
 'zero_note':'RN-even recurrence starts positive zero; negative zero tangent produces positive zero. Negative-zero plus negative-zero yields negative-zero only with negative initial state, outside this recurrence.',
 'nonfinite_policy':'All nonfinite outputs are diagnostic failures, even when classification agrees. NaN payloads recorded, not declared identical.'}
encoded=json.dumps(result,indent=2)
(ROOT / ('receipt-optimized.json' if sys.flags.optimize else 'receipt-normal.json')).write_text(encoded+'\n')
print(encoded)
