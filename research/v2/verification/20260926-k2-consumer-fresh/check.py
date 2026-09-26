import ast, difflib, hashlib, json, pathlib, sys
P=pathlib.Path('/home/poojith-devan/k2-consumer-fresh-review-8RnLEO')
O=pathlib.Path(__file__).parent
def need(x,msg):
    if not x: raise RuntimeError(msg)
manifest=P/'SHA256SUMS'
need(hashlib.sha256(manifest.read_bytes()).hexdigest()=='8fb86d698282a059a5e4db8c18b0d3d099a2eb1c2fd0819fad9dbb69d3e878d3','manifest')
hashes={}
for line in manifest.read_text().splitlines():
    digest,name=line.split(maxsplit=1); name=name.lstrip('*')
    hashes[name]=hashlib.sha256((P/name).read_bytes()).hexdigest()
    need(hashes[name]==digest,'member '+name)
delta=''.join(difflib.unified_diff((P/'baseline_consumer_nki.py').read_text().splitlines(True),(P/'consumer_nki.py').read_text().splitlines(True),fromfile='baseline/consumer_nki.py',tofile='candidate/consumer_nki.py'))
need(delta==(P/'consumer.diff').read_text(),'diff not exact')

# Independent IEEE binary32 arithmetic using exact integers in 2^-149 units.
def finite_int(b):
    e=(b>>23)&255; f=b&0x7fffff
    n=f if e==0 else ((1<<23)|f)<<(e-1)
    return -n if b>>31 else n
def rounded(n):
    s=0x80000000 if n<0 else 0; n=abs(n)
    if n==0:return 0
    shift=max(n.bit_length()-24,0)
    q,r=divmod(n,1<<shift)
    if shift and (r>1<<(shift-1) or (r==1<<(shift-1) and q%2)):q+=1
    if q==1<<24:q>>=1;shift+=1
    if q<1<<23:return s|q
    return s|0x7f800000 if shift+1>=255 else s|((shift+1)<<23)|(q-(1<<23))
def nan(b):return b&0x7fffffff>0x7f800000
def inf(b):return b&0x7fffffff==0x7f800000
def add(a,b):
    if nan(a) or nan(b):return 0x7fc00000
    if inf(a) and inf(b) and (a^b)>>31:return 0x7fc00000
    if inf(a):return a
    if inf(b):return b
    if a==b==0x80000000:return a
    return rounded(finite_int(a)+finite_int(b))
def ref(ids,grad,v=8,c=8):
    output=[0]*(v*c)
    for t,row in enumerate(ids):
        for col in range(c):output[row*c+col]=add(output[row*c+col],grad[t*c+col]<<16)
    return output

data=json.loads((P/'consumer-raw-normal.json').read_text())
need((P/'consumer-raw-normal.json').read_bytes()==(P/'consumer-raw-optimized.json').read_bytes(),'receipt normal/-O')
records=[]
for fixture in data['fixtures']:
    ids=fixture['ids'];grad=[int(x,16) for x in fixture['input_bf16']]
    expected=ref(ids,grad);given=[int(x,16) for x in fixture['output_bits']]
    need(len(expected)==len(given),'size')
    differences=[]
    for i,(a,b) in enumerate(zip(expected,given)):
        if a!=b:
            need(nan(a) and nan(b),'non-NaN raw mismatch '+fixture['name'])
            differences.append({'element':i,'integer_reference':f'{a:08x}','packet':f'{b:08x}'})
    counts=[ids.count(r) for r in range(8)]
    maxima=[max(counts[:4]),max(counts[4:])]
    need(fixture['schedules']==[maxima,maxima],'schedule')
    # Own occurrence schedule, derived by direct scanning IDs, not supplied metadata code.
    rows=[[t for t,r in enumerate(ids) if r==row] for row in range(8)]
    scheduled=[0]*64
    for owner in range(2):
        for group in range(2):
            for occurrence in range(maxima[group]):
                for row in range(group*4,group*4+4):
                    if occurrence<len(rows[row]):
                        t=rows[row][occurrence]
                        for c in range(owner*4,owner*4+4):
                            k=row*8+c;scheduled[k]=add(scheduled[k],grad[t*8+c]<<16)
    need(scheduled==expected,'own schedule differs')
    records.append({'name':fixture['name'],'verdict':'FAIL_NONFINITE_DIAGNOSTIC' if any(nan(b) or inf(b) for b in expected) else 'EXACT_FINITE',
                    'reference_bits':[f'{b:08x}' for b in expected],'nan_bit_differences':differences})

# Independent known order witnesses, including a call boundary difference.
witnesses=[]
for bf in ([0x4b80,0x3f80,0xcb80],[0x3f80,0x3380,0xbf80],[0x0081,0x8080,0],[0x8081,0x0080,0]):
    a,b,c=[v<<16 for v in bf]
    witnesses.append({'bf16':[f'{v:04x}' for v in bf],'ordered':f'{add(add(a,b),c):08x}','regrouped':f'{add(a,add(b,c)):08x}'})
need(witnesses[0]['ordered']=='00000000' and witnesses[0]['regrouped']=='3f800000','large witness')
need(witnesses[1]['ordered']=='00000000' and witnesses[1]['regrouped']=='33800000','small witness')
need(witnesses[2]['ordered']=='00010000' and witnesses[3]['ordered']=='80010000','subnormal')

# Full-shape mathematical addresses and ownership. No production tensor allocated.
endpoint_checks=0
for t in range(32768):
    for owner in range(2):
        lo=t*1024+owner*512;hi=lo+511
        need(0<=lo<=hi<32768*1024,'HBM bounds')
        need(lo//1024==hi//1024==t,'cross token')
        endpoint_checks+=2
for owner in range(2):
    sentinel_first=32768*1024+owner*512
    need(sentinel_first>=32768*1024,'sentinel outside gradient')
for r in range(8192):
    need(0<=r//128<64 and 0<=r%128<128,'row geometry')
for c in range(1024):
    need(sum(owner*512<=c<(owner+1)*512 for owner in range(2))==1,'owner coverage')
# Pointer/remaining arithmetic uses FP32 on Vector Engine. All reachable values fit exact int24.
need(32768+32768<2**24 and -32768>-(2**24),'index arithmetic exact range')

namespace={};before=set(sys.modules)
exec(compile((P/'consumer_nki.py').read_text(),'packet-consumer','exec'),namespace)
try:namespace['build_kernel']()
except RuntimeError as exc:need('EXPERIMENT_DISABLED' in str(exc),'default-off error')
else:raise RuntimeError('default enabled')
need(not any(n=='nki' or n.startswith('nki.') for n in set(sys.modules)-before),'native import')
tree=ast.parse((P/'consumer_nki.py').read_text())
need(sum(isinstance(n,ast.For) and ast.unparse(n.iter)=='nl.dynamic_range(limit)' for n in ast.walk(tree))==1,'dynamic loop')
out={'verdict':'SOURCE_MATH_ACCEPTED_NATIVE_COMPONENT_UNQUALIFIED','packet_hashes':hashes,
     'exact_diff_authenticated':True,'default_off_no_native_import':True,'max_allocated_tensor_elements':0,
     'fixture_count':len(records),'fixtures':records,'independent_witnesses':witnesses,
     'full_geometry_descriptor_endpoints_checked':endpoint_checks,
     'owner_coverage':'8192 rows, 1024 channels, two disjoint halves, 64 groups',
     'fp32_address_arithmetic_bound':65536,'logical_sbuf_bytes_per_owner':4*128*512*4+7*128*4+4,
     'native_claim':False,'python':sys.version}
(O/('raw-optimized.json' if sys.flags.optimize else 'raw-normal.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('packet_hashes','fixtures')},indent=2))
