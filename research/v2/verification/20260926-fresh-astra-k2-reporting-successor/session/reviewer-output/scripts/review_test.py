#!/usr/bin/env python3
"""Independent, sequential CPU evidence; never imports or simulates an NKI API.
Candidate functions and reporting statements are AST-extracted without edits.
Oracle addition uses exact integers in units of 2**-149 and ties-to-even rounding.
"""
import argparse, ast, builtins, contextlib, copy, gc, hashlib, importlib.util
import io, json, math, os, pathlib, random, struct, sys, time, traceback, types, weakref

p = argparse.ArgumentParser()
p.add_argument('--output', type=pathlib.Path, required=True)
args = p.parse_args()
ROOT = args.output
ROOT.mkdir(parents=True, exist_ok=False)
START = time.monotonic()
RESULTS = []

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')

def check(ok, msg):
    if not ok:
        raise AssertionError(msg)

def strict_load(s):
    def bad(x):
        raise ValueError('nonstandard JSON constant: '+x)
    return json.loads(s, parse_constant=bad)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]

def value(b):
    return struct.unpack('<f', struct.pack('<I', b))[0]

def pack_bits(rows):
    return b''.join(struct.pack('<I', x) for row in rows for x in row)

def float_rows_bytes(rows):
    return b''.join(struct.pack('<f', x) for row in rows for x in row)

def finite_units(b):
    exp, mant = (b >> 23) & 255, b & 0x7fffff
    check(exp != 255, 'nonfinite oracle input')
    magnitude = mant if exp == 0 else ((1 << 23) + mant) << (exp - 1)
    return -magnitude if b >> 31 else magnitude

def add32(a, b):
    total = finite_units(a) + finite_units(b)
    if total == 0:
        return (a & b) & 0x80000000  # RNE exact cancellation; -0 + -0 remains -0.
    sign = 0x80000000 if total < 0 else 0
    n = abs(total)
    if n < (1 << 23):
        return sign | n
    shift = max(0, n.bit_length() - 24)
    q, remainder = divmod(n, 1 << shift)
    if shift and (remainder > (1 << (shift-1)) or (remainder == (1 << (shift-1)) and q & 1)):
        q += 1
    if q == 1 << 24:
        q >>= 1
        shift += 1
    exponent = shift + 1
    if exponent >= 255:
        return sign | 0x7f800000
    return sign | (exponent << 23) | (q - (1 << 23))

def cast_bf16(b):
    # Independent quotient/remainder RNE, preserve signed zero.
    hi, lo = divmod(b, 65536)
    if lo > 32768 or (lo == 32768 and hi & 1):
        hi += 1
    return (hi & 65535) << 16

def oracle(ids, rows):
    result = [[0] * 128 for _ in range(256)]
    for row, grad in zip(ids, rows):
        for c, g in enumerate(grad):
            result[row][c] = add32(result[row][c], g)
    return result

def routes(ids):
    counts = [sum(r == row for r in ids) for row in range(256)]
    starts = [sum(counts[:row]) for row in range(256)]
    positions = sorted(range(len(ids)), key=lambda token: (ids[token], token))
    positions += [256] * (256 - len(ids))
    return counts, starts, positions, [max(counts[:128]), max(counts[128:])]

def load_source(name):
    path = pathlib.Path('/work/source')/name
    ns = {'__name__': 'review_'+name.replace('.','_'), '__file__': str(path)}
    exec(compile(path.read_text(), str(path), 'exec'), ns)
    return types.SimpleNamespace(**ns)

# Guards execute before importing Torch into this test process.
def disabled_guards():
    old_import = builtins.__import__
    denied = []
    def guarded(name, *a, **k):
        if name.split('.')[0] in ('torch','nki','torch_neuronx'):
            denied.append(name)
            raise RuntimeError('TEST_FORBIDS_NATIVE_OR_TORCH_IMPORT:'+name)
        return old_import(name, *a, **k)
    saved_argv = sys.argv
    out, err = io.StringIO(), io.StringIO()
    try:
        builtins.__import__ = guarded
        checker = load_source('check_native.py')
        sys.argv = ['check_native.py']
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = checker.main()
        check(code == 0 and strict_load(out.getvalue()) == dict(status='DISABLED',native_imported=False,native_executed=False), 'default checker')
        kernel = load_source('consumer_nki.py')
        try:
            kernel.build_kernel()
        except RuntimeError as e:
            message = str(e)
        else:
            raise AssertionError('kernel default enabled')
        check(message.startswith('EXPERIMENT_DISABLED') and not denied, 'disabled import boundary')
        dump(ROOT/'disabled.json', dict(checker_stdout=out.getvalue(), checker_stderr=err.getvalue(), checker_exit=code, kernel_error=message, denied_import_attempts=denied))
    finally:
        sys.argv = saved_argv
        builtins.__import__ = old_import

class Audit:
    def __init__(self):
        self.live = {}
        self.peak_bytes = 0
        self.largest_storage = 0
        self.operations = 0
    def see(self, obj):
        if isinstance(obj, torch.Tensor):
            check(obj.device.type == 'cpu', 'CPU only')
            self.live[id(obj)] = weakref.ref(obj)
        elif isinstance(obj, (list,tuple)):
            for x in obj: self.see(x)
        elif isinstance(obj, dict):
            for x in obj.values(): self.see(x)
    def sample(self):
        stores = {}
        for key, ref in list(self.live.items()):
            t = ref()
            if t is None:
                del self.live[key]
            else:
                s = t.untyped_storage()
                stores[s._cdata] = s.nbytes()
        total = sum(stores.values())
        self.peak_bytes = max(self.peak_bytes, total)
        self.largest_storage = max([self.largest_storage]+list(stores.values()))
        check(total < 8*1024*1024, 'stricter 8MiB live tensor budget exceeded')


def arithmetic_selftest():
    r = random.Random(618209)
    special = [0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800000,0x33800000,0x4b800000,0x7f7fffff,0xff7fffff]
    pairs = [(a,b) for a in special for b in special]
    for _ in range(4096):
        a = r.getrandbits(32)
        b = r.getrandbits(32)
        if (a >> 23) & 255 == 255: a ^= 1 << 23
        if (b >> 23) & 255 == 255: b ^= 1 << 23
        pairs.append((a,b))
    expected = [add32(a,b) for a,b in pairs]
    a = torch.tensor([value(x) for x,y in pairs], dtype=torch.float32)
    b = torch.tensor([value(y) for x,y in pairs], dtype=torch.float32)
    actual = (a+b).view(torch.int32).tolist()
    actual = [x & 0xffffffff for x in actual]
    directory = ROOT/'oracle_selftest'
    directory.mkdir()
    (directory/'inputs.u32le').write_bytes(b''.join(struct.pack('<II', a,b) for a,b in pairs))
    (directory/'expected.u32le').write_bytes(struct.pack('<'+'I'*len(expected), *expected))
    (directory/'actual.u32le').write_bytes(struct.pack('<'+'I'*len(actual), *actual))
    dump(directory/'result.json',dict(pairs=len(pairs),exact=actual==expected,atol=0,rtol=0,seed=618209))
    check(actual == expected, 'integer oracle disagrees with real CPU FP32 addition')


def independent_cases():
    # Fixed physical shape, independently chosen routes and channel values.
    for dtype in ('fp32','bf16'):
        ids = [128]*150 + [127]*73 + [0]*25 + [255]*7 + [129]
        random.Random(831).shuffle(ids)
        pool = [0,0x80000000,0x3f800000,0xbf800000,0x3eaaaaab,0x33800000,1,0x80000001,0x00800000,0x80800000]
        r = random.Random(922)
        rows = [[r.choice(pool) for _ in range(128)] for _ in ids]
        if dtype == 'bf16': rows = [[cast_bf16(b) for b in row] for row in rows]
        yield 'own_unequal_both_groups',dtype,ids,rows
        ids = [127,128,255,0,127,128,255,0,127,128,255,0] + [42]*244
        rows = [[0]*128 for _ in ids]
        for base in range(4):
            for token,b in zip((base,base+4,base+8),(0x4b800000,0x3f800000,0xcb800000)):
                rows[token] = [b]*128
        yield 'own_order_sensitive_boundaries',dtype,ids,rows
        ids = [0]*256
        rows = [[0x80000000 if (t+c)%2 else 0 for c in range(128)] for t in range(256)]
        yield 'own_signed_zeros',dtype,ids,rows
        yield 'own_empty',dtype,[],[]
    r = random.Random(91231)
    for i in range(4):
        ids = [r.choice([0,1,126,127,128,129,254,255]) for _ in range(256)]
        rows = []
        for t in ids:
            row = []
            for c in range(128):
                # Broad but bounded exponents, with cancellation and rounding gaps.
                sign = r.getrandbits(1) << 31
                exp = r.randrange(1,161)
                row.append(sign | (exp << 23) | r.getrandbits(23))
            rows.append(row)
        yield 'own_seeded_'+str(i),'fp32',ids,rows


def run_reference_case(name,dtype,ids,row_bits,origin):
    path = ROOT/'recurrence'/(origin+'_'+name+'_'+dtype)
    path.mkdir(parents=True)
    rows = [[value(b) for b in row] for row in row_bits]
    meta = routes(ids)
    dump(path/'input.json', dict(name=name,dtype=dtype,origin=origin,ids=ids,metadata=meta,physical_shape=[256,128],logical_length=len(ids),input_encoding='little-endian FP32 representation after source dtype conversion',seed_scope='own seeds in script; supplied fixtures seed 42'))
    (path/'input.f32le').write_bytes(pack_bits(row_bits))
    expected_bits = oracle(ids,row_bits)
    expected = pack_bits(expected_bits)
    (path/'expected.f32le').write_bytes(expected)
    supplied_meta = REF.metadata(ids)
    check(supplied_meta == meta, name+' independent routing mismatch')
    actual,trace = REF.consumer(ids,rows,meta)
    dense = REF.dense(ids,rows)
    actual_bytes,dense_bytes = float_rows_bytes(actual),float_rows_bytes(dense)
    (path/'consumer_actual.f32le').write_bytes(actual_bytes)
    (path/'dense_actual.f32le').write_bytes(dense_bytes)
    dump(path/'trace.json',trace)
    for row in range(256):
        visits = [t for r,t in trace['visits'] if r == row]
        check(visits == [t for t,r in enumerate(ids) if r == row], name+' occurrence order')
    check(trace['active_gathers'] == len(ids),name+' active gathers')
    check(trace['inactive_gathers_skipped'] == 128*sum(meta[3])-len(ids),name+' inactive skips')
    check(trace['output_rows_stored'] == 256 and trace['accumulator_hbm_reads'] == 0,name+' state trace')
    check(trace['group_rounds'] == meta[3],name+' group bounds')
    max_error = max(abs(value(e)-a) for row,got in zip(expected_bits,actual) for e,a in zip(row,got))
    record = dict(name=name,dtype=dtype,origin=origin,byte_exact=expected==actual_bytes==dense_bytes,sha256=sha(actual_bytes),expected_sha256=sha(expected),max_abs=max_error,atol=0,rtol=0,criterion='identical shape, dtype, bytes; signed zero included',group_rounds=meta[3],active_gathers=trace['active_gathers'],inactive_gathers_skipped=trace['inactive_gathers_skipped'],row3=actual[3][0],rejected_grouped_row3=None)
    if name.startswith('cross_tile_'):
        partials = [oracle(ids[:128],row_bits[:128]), oracle(ids[128:],row_bits[128:])]
        wrong = [[add32(a,b) for a,b in zip(row1,row2)] for row1,row2 in zip(*partials)]
        wrong_raw=pack_bits(wrong)
        (path/'negative_grouped.f32le').write_bytes(wrong_raw)
        check(wrong_raw != expected, 'grouped negative control must differ')
        check(wrong_raw == float_rows_bytes(REF.grouped_wrong(ids,rows)), 'grouped control agreement')
        record['rejected_grouped_row3'] = value(wrong[3][0])
        record['negative_grouping_rejected'] = True
    if name == 'own_order_sensitive_boundaries':
        reverse = oracle(list(reversed(ids)),list(reversed(row_bits)))
        reverse_bytes=pack_bits(reverse)
        (path/'negative_reversed.f32le').write_bytes(reverse_bytes)
        check(reverse_bytes != expected,'reversal negative control')
        record['negative_reversal_rejected'] = True
        record['row127_expected_bits'] = f'{expected_bits[127][0]:08x}'
        record['row127_reversed_bits'] = f'{reverse[127][0]:08x}'
    dump(path/'result.json',record)
    check(record['byte_exact'] and max_error == 0,name+' recurrence exactness')
    return record


def metadata_negatives():
    ids = [127,128,127,255,128]
    original = routes(ids)
    mutations = []
    for label,where,index,new in [('negative_count',0,127,-1),('count_bool',0,127,True),('wrong_offset',1,128,0),('wrong_route',2,0,1),('duplicate_position',2,1,0),('bad_padding',2,255,0),('short_group_loop',3,0,1),('long_group_loop',3,1,3)]:
        m=copy.deepcopy(original); m[where][index]=new; mutations.append((label,m))
    m=copy.deepcopy(original); m[2][0],m[2][1]=m[2][1],m[2][0]; mutations.append(('unstable_order',m))
    m=copy.deepcopy(original); m[2].pop(); mutations.append(('wrong_shape',m))
    result=[]
    for name,m in mutations:
        try: REF.validate_metadata(ids,m)
        except Exception as e: error=type(e).__name__+': '+str(e)
        else: error=None
        result.append(dict(name=name,ids=ids,metadata=m,expected='ValueError',actual_error=error))
        check(error and error.startswith('ValueError:'),name+' not rejected')
    for name,bad_ids in [('negative_id',[-1]),('high_id',[256]),('bool_id',[True]),('float_id',[1.0]),('too_many',[0]*257)]:
        try: REF.metadata(bad_ids)
        except Exception as e: error=type(e).__name__+': '+str(e)
        else: error=None
        result.append(dict(name=name,ids=bad_ids,expected='ValueError',actual_error=error))
        check(error and error.startswith('ValueError:'),name+' not rejected')
    dump(ROOT/'metadata_negative_controls.json',result)


def historical_audit(records):
    lookup={(x['name'],x['dtype']):x for x in records if x['origin']=='supplied'}
    result=[]
    for name in ('checks-normal.json','checks-optimized.json'):
        old = strict_load((pathlib.Path('/packet/raw')/name).read_text())
        check(len(old['cases']) == len(lookup) == 20,'receipt case count')
        seen=set()
        for entry in old['cases']:
            key=(entry['case'],entry['dtype']); check(key not in seen,'receipt duplicate');seen.add(key)
            new=lookup[key]
            for field in ('byte_exact','sha256','group_rounds','active_gathers','inactive_gathers_skipped','row3','rejected_grouped_row3'):
                check(entry[field] == new[field],name+' field '+field+' '+str(key))
        check(old['exit_code']==0 and old['native_executed'] is False and old['torch_imported'] is False and old['launch_authorized'] is False,'receipt scope fields')
        result.append(dict(file=name,cases=20,verified_arithmetic_fields=['byte_exact','sha256','group_rounds','active_gathers','inactive_gathers_skipped','row3','rejected_grouped_row3'],unverified_historical_claims=['actual command execution','recorded timestamp','peak_traced_bytes','production_files_changed','historical disabled-import test'],status_label_used_as_proof=False))
    dump(ROOT/'historical_audit.json',result)


def extract_checker():
    source=pathlib.Path('/work/source/check_native.py').read_text()
    tree=ast.parse(source)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    defs=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in ('compare','tensor_bytes')]
    ns=dict(torch=torch,hashlib=hashlib,math=math)
    exec(compile(ast.Module(body=defs,type_ignores=[]),'/work/source/check_native.py','exec'),ns)
    outer=next(n for n in main.body if isinstance(n,ast.Try) and n.finalbody)
    dtype_loop=next(n for n in outer.body if isinstance(n,ast.For))
    case_loop=next(n for n in dtype_loop.body if isinstance(n,ast.For))
    tail=case_loop.body[-3:]
    check(tail[0].lineno==158 and tail[-1].lineno==160,'case tail extraction coordinates')
    # This harness runs real case reporting + handler/finalizer + return code.
    # It supplies a completed case and a test cleanup sink, never a native group.
    harness_try=ast.Try(body=tail+[outer.body[-1]],handlers=outer.handlers,orelse=outer.orelse,finalbody=outer.finalbody)
    end=main.body[main.body.index(outer)+1:]
    init=ast.parse('error = None').body
    fn=ast.FunctionDef(name='report_case',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=init+[harness_try]+end,decorator_list=[])
    code=compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'extracted_reporting_harness','exec')
    dump(ROOT/'extraction.json',dict(candidate_sha256=sha(source.encode()),comparison_lines=[76,91],case_reporting_lines=[158,161],handler_lines=[179,182],finalizer_lines=[183,196],return_lines=[197,200],case_body_substituted='only completed CPU comparison records; native imports, setup, calls, autograd and distributed operations omitted',dist_stub='destroy_process_group counts calls; optional explicit test failure',source_statements_changed=False))
    return ns['compare'],code


def scalar_reference(actual,expected):
    # Metric evaluated in Python float64, independent of tensor operations.
    delta=abs(value(actual)-value(expected))
    result=dict(exact=actual==expected,finite=math.isfinite(value(actual)),nonzero=int(value(actual)!=0.0),max_abs=delta if math.isfinite(delta) else None)
    if not math.isfinite(delta): result['max_abs_nonfinite']=repr(delta)
    return result


def compare_tests(compare):
    cases = strict_load(pathlib.Path('/packet/raw/scalar-inputs.json').read_text())['cases']
    cases += [dict(name='finite_ulp',actual_hex='3f800001',expected_hex='3f800000'),dict(name='negative_zero_exact',actual_hex='80000000',expected_hex='80000000'),dict(name='reverse_signed_zero',actual_hex='00000000',expected_hex='80000000'),dict(name='finite_opposite_max',actual_hex='7f7fffff',expected_hex='ff7fffff'),dict(name='smallest_subnormal_mismatch',actual_hex='00000001',expected_hex='00000000')]
    result=[]
    for c in cases:
        a,e=int(c['actual_hex'],16),int(c['expected_hex'],16)
        actual=torch.tensor([[value(a)]],dtype=torch.float32)
        expected=torch.tensor([[value(e)]],dtype=torch.float32)
        got=compare(actual,expected,c['name'])
        want=scalar_reference(a,e)
        want.update(name=c['name'],actual_dtype='torch.float32',shape=[1,1],actual_sha256=sha(struct.pack('<I',a)),expected_sha256=sha(struct.pack('<I',e)))
        encoded=json.dumps(got,allow_nan=False)
        result.append(dict(input=c,expected=want,actual=got,serialized=encoded,atol=0,rtol=0))
        dump(ROOT/'comparison'/(c['name']+'.json'),result[-1])
        check(got == want and strict_load(encoded) == got,c['name']+' checker comparison')
    extras=[]
    def extra(name,a,e,want_exact,want_max):
        got=compare(a,e,name)
        record=dict(name=name,actual_shape=list(a.shape),expected_shape=list(e.shape),actual_dtype=str(a.dtype),expected_dtype=str(e.dtype),actual_hex=a.contiguous().view(torch.uint8).numpy().tobytes().hex(),expected_hex=e.contiguous().view(torch.uint8).numpy().tobytes().hex(),expected_exact=want_exact,expected_max_abs=want_max,actual=got,atol=0,rtol=0)
        extras.append(record)
        check(got['exact'] is want_exact,name+' exact')
        if want_max == 'ABSENT': check('max_abs' not in got,name+' shape no metric')
        else: check(got['max_abs'] == want_max,name+' metric')
        strict_load(json.dumps(got,allow_nan=False))
    extra('shape_mismatch_same_bytes',torch.zeros((1,2)),torch.zeros((2,1)),False,'ABSENT')
    extra('dtype_mismatch_same_numeric',torch.ones((1,1),dtype=torch.float64),torch.ones((1,1)),False,0.0)
    extra('bf16_exact',torch.tensor([[1.5]],dtype=torch.bfloat16),torch.tensor([[1.5]],dtype=torch.bfloat16),True,0.0)
    extra('bf16_vs_fp32',torch.tensor([[1.5]],dtype=torch.bfloat16),torch.tensor([[1.5]]),False,0.0)
    extra('empty',torch.empty((0,128)),torch.empty((0,128)),True,0.0)
    extra('noncontiguous_exact',torch.tensor([[1.,2.],[3.,4.]]).t(),torch.tensor([[1.,3.],[2.,4.]]),True,0.0)
    dump(ROOT/'comparison'/'additional.json',extras)
    return result


class CleanupSink:
    def __init__(self,fail=False): self.calls=0; self.fail=fail
    def destroy_process_group(self):
        self.calls+=1
        if self.fail: raise RuntimeError('INJECTED_DESTROY_FAILURE')

class FailingStdout(io.StringIO):
    def write(self,s): raise BrokenPipeError('INJECTED_STDOUT_FAILURE')

class BadDirectory:
    def __init__(self,kind): self.kind=kind
    def is_dir(self):
        if self.kind=='is_dir': raise OSError('INJECTED_IS_DIR_FAILURE')
        return True
    def __truediv__(self,name): return self
    def open(self,mode):
        check(mode=='x','exclusive mode retained')
        if self.kind=='open': raise PermissionError('INJECTED_OPEN_FAILURE')
        return self
    def __enter__(self): return self
    def __exit__(self,*a): return False
    def write(self,s): raise OSError('INJECTED_WRITE_FAILURE')


def reporting_tests(code,comparison_records):
    rows=[]
    def run(name,checks,mode='normal',rank=0):
        d=ROOT/'reporting'/name;d.mkdir(parents=True)
        destination=d/'receipt';destination.mkdir()
        old=b'PREEXISTING DO NOT OVERWRITE\n'
        if mode=='exclusive': (destination/f'rank{rank}.json').write_bytes(old)
        if mode=='absent': destination=d/'absent'
        case=dict(name=name,dtype='torch.float32',checks=copy.deepcopy(checks),status='STARTED')
        receipt=dict(scope='EXTRACTED_CPU_REPORTING_ONLY',cases=[case],tolerance=dict(atol=0,rtol=0,criterion='identical bytes'),launch_authorized=False)
        if mode=='invalid_json': receipt['injected_nonfinite']=float('nan')
        if mode=='invalid_type': receipt['injected_type']=object()
        dist=CleanupSink(mode=='destroy')
        stdout=FailingStdout() if mode=='stdout' else io.StringIO()
        stderr=io.StringIO()
        target=BadDirectory(mode) if mode in ('is_dir','open','write') else destination
        ns=dict(json=json,time=time,traceback=traceback,sys=sys,args=types.SimpleNamespace(out_dir=target),rank=rank,receipt=receipt,case=case,name=name,dtype_name='fp32',dist=dist,started=time.monotonic())
        exec(code,ns)
        returned=None; exception=None; error_trace=None
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            try: returned=ns['report_case']()
            except Exception as e:
                exception=type(e).__name__+': '+str(e);error_trace=traceback.format_exc()
        (d/'stdout.txt').write_text(stdout.getvalue())
        (d/'stderr.txt').write_text(stderr.getvalue())
        if error_trace: (d/'exception.txt').write_text(error_trace)
        out_file=destination/f'rank{rank}.json'
        record=dict(name=name,mode=mode,rank=rank,input_checks=checks,expected_cleanup_calls=1,actual_cleanup_calls=dist.calls,returned=returned,exception=exception,receipt_file_exists=out_file.exists(),receipt_status=receipt.get('status'),case_status=case['status'])
        dump(d/'result.json',record)
        rows.append(record)
        check(dist.calls==1,name+' cleanup once')
        if mode=='normal' or mode=='absent':
            exact=all(x['exact'] for x in checks)
            check(exception is None and returned==(0 if exact else 1),name+' return')
            lines=stdout.getvalue().splitlines()
            check(len(lines)==2,name+' two JSON stdout receipts')
            parsed_case,parsed_receipt=map(strict_load,lines)
            check(parsed_case['status']==('EXACT' if exact else 'FAIL_EXACTNESS'),name+' case status')
            check(parsed_receipt['status']==('TINY_CONSUMER_EXACT_ONLY' if exact else 'REFUSE_OR_UNSUPPORTED'),name+' final status')
            if not exact: check('EXACTNESS_FALSIFIED' in parsed_receipt['error'] and 'EXACTNESS_FALSIFIED' in stderr.getvalue(),name+' legible error')
            if mode=='normal': check(strict_load(out_file.read_text())==parsed_receipt,name+' file vs stdout')
            else: check(not out_file.exists(),name+' missing dir not fabricated')
        elif mode=='exclusive':
            check(exception.startswith('FileExistsError:') and out_file.read_bytes()==old,name+' exclusive preservation')
        elif mode in ('invalid_json','invalid_type'):
            check(exception is not None and not out_file.exists(),name+' preencoding no partial JSON')
        elif mode=='stdout':
            # The first case print also fails, but the error handler records it;
            # final stdout failure happens after a valid receipt file is written.
            check(exception.startswith('BrokenPipeError:') and strict_load(out_file.read_text())['status']=='REFUSE_OR_UNSUPPORTED',name+' stdout cleanup/file')
        elif mode=='destroy': check(exception.startswith('RuntimeError: INJECTED_DESTROY_FAILURE'),name+' destroy exception')
        else: check(exception is not None,name+' injected I/O error')
    for entry in comparison_records:
        run(entry['input']['name'],[entry['actual']])
    run('missing_gradient',[dict(name='backward',exact=False,missing_gradient=True)])
    finite=[comparison_records[1]['actual']]
    for mode in ('exclusive','absent','invalid_json','invalid_type','is_dir','open','write','stdout','destroy'):
        run('fault_'+mode,finite,mode=mode)
    # Finalization writes this rank independently; no actual process group exists.
    for rank in (1,2,3): run('rank_'+str(rank),finite,rank=rank)
    dump(ROOT/'reporting_summary.json',rows)


def main():
    global torch, REF
    disabled_guards()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from torch.utils._python_dispatch import TorchDispatchMode
    audit=Audit()
    class Track(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            audit.see(args);audit.see(kwargs or {});audit.sample()
            out=func(*args,**(kwargs or {}))
            audit.see(out);audit.sample();audit.operations+=1
            return out
    REF=load_source('reference.py')
    with Track():
        arithmetic_selftest()
        records=[]
        for name in REF.CASE_NAMES:
            for dtype in ('bf16','fp32'):
                ids,rows,meta=REF.fixture(name,dtype)
                records.append(run_reference_case(name,dtype,ids,[[bits(x) for x in row] for row in rows],'supplied'))
        for name,dtype,ids,row_bits in independent_cases():
            records.append(run_reference_case(name,dtype,ids,row_bits,'independent'))
        dump(ROOT/'recurrence_summary.json',records)
        metadata_negatives()
        historical_audit(records)
        compare,code=extract_checker()
        comparison_records=compare_tests(compare)
        reporting_tests(code,comparison_records)
    gc.collect();audit.sample()
    check(not any(n.startswith(('nki','torch_neuron')) for n in sys.modules),'native module unexpectedly imported')
    summary=dict(status='PASS',scope='independent CPU arithmetic and extracted checker reporting only',recurrence_cases=len(records),integer_addition_pairs=4217,comparison_cases=len(comparison_records)+6,metadata_rejection_cases=15,reporting_cases=len(strict_load((ROOT/'reporting_summary.json').read_text())),torch=torch.__version__,python=sys.version,optimized=bool(sys.flags.optimize),elapsed_seconds=time.monotonic()-START,intraop_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),peak_live_tensor_storage_bytes=audit.peak_bytes,largest_tensor_storage_bytes=audit.largest_storage,tracked_tensor_operations=audit.operations,storage_limit_bytes=64*1024*1024,enforced_stricter_storage_limit_bytes=8*1024*1024,storage_tracking='dispatch input/output tensors, weak references, storage identity deduplication; excludes Python objects and library RSS',native_imported=False,native_executed=False,launch_authorized=False,atol=0,rtol=0)
    dump(ROOT/'summary.json',summary)
    print(json.dumps(summary,allow_nan=False))

try:
    main()
except BaseException as e:
    dump(ROOT/'FAILED.json',dict(status='REFUSE',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START))
    raise
