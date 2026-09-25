#!/usr/bin/env python3
"""Targeted unresolved coverage: BF16 conversion and full main setup boundaries.
Native imports, Torch setup API and distributed calls are explicit test stubs.
No NKI API is simulated and no kernel call can succeed.
"""
import argparse, builtins, contextlib, hashlib, importlib.metadata, io, json, math
import os, pathlib, random, struct, sys, time, traceback, types
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True)
ROOT=p.parse_args().output;ROOT.mkdir(parents=True,exist_ok=False)
def dump(path,obj):
    with path.open('x') as f: json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def check(ok,text):
    if not ok: raise AssertionError(text)
def source(name):
    f=pathlib.Path('/work/source')/name
    ns=dict(__name__='bounded_test_source',__file__=str(f))
    exec(compile(f.read_text(),str(f),'exec'),ns)
    return ns

def bf16_check():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ref=source('reference.py')
    r=random.Random(762190)
    values=[0,0x80000000,1,0x80000001,0x00008000,0x00008001,0x00018000,0x3f808000,0x3f818000,0x7f7fffff,0xff7fffff]
    for _ in range(2048):
        b=r.getrandbits(32)
        if ((b>>23)&255)==255: b^=1<<23
        values.append(b)
    expected=[]
    for b in values:
        q,rem=divmod(b,65536)
        expected.append(((q+int(rem>32768 or (rem==32768 and q%2==1)))&65535)<<16)
    floats=[struct.unpack('<f',struct.pack('<I',b))[0] for b in values]
    cpu_in=torch.tensor(floats,dtype=torch.float32)
    cpu_bf16=cpu_in.to(torch.bfloat16)
    cpu_out=cpu_bf16.to(torch.float32)
    actual=[x&0xffffffff for x in cpu_out.view(torch.int32).tolist()]
    reference=[struct.unpack('<I',struct.pack('<f',ref['bf16'](x)))[0] for x in floats]
    for name,data in [('inputs',values),('expected',expected),('cpu_actual',actual),('reference_actual',reference)]:
        (ROOT/(name+'.u32le')).write_bytes(struct.pack('<'+'I'*len(data),*data))
    stores={t.untyped_storage()._cdata:t.untyped_storage().nbytes() for t in (cpu_in,cpu_bf16,cpu_out)}
    live=sum(stores.values())
    result=dict(cases=len(values),seed=762190,exact=expected==actual==reference,atol=0,rtol=0,peak_live_tensor_storage_bytes=live,largest_storage_bytes=max(stores.values()),tracking='three explicitly retained tensors; int32 view aliases output storage',intraop_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),native=False)
    dump(ROOT/'bf16_result.json',result)
    check(expected==actual==reference,'BF16 conversion mismatch')
    check(live<64*1024*1024,'tensor bound')
    return result


def setup_checks():
    checker=source('check_native.py')
    hashes={name:hashlib.sha256((pathlib.Path('/work/source')/name).read_bytes()).hexdigest() for name in ('consumer_nki.py','reference.py')}
    old_import=builtins.__import__;old_argv=sys.argv;old_version=importlib.metadata.version
    environment={k:os.environ.get(k) for k in ('WORLD_SIZE','NEURON_LOGICAL_NC_CONFIG','RANK')}
    records=[]
    for scenario,expected_init,expected_destroy,expected_return,expected_error in [
        ('missing_arguments',0,0,None,'SystemExit'),
        ('support_hash',0,0,None,'RuntimeError'),
        ('world_binding',0,0,None,'RuntimeError'),
        ('native_import',0,0,None,'ImportError'),
        ('rank_parse',0,0,None,'ValueError'),
        ('init',1,0,None,'RuntimeError'),
        ('device',1,0,None,'RuntimeError'),
        ('sync_api',1,0,None,'RuntimeError'),
        ('existing_directory',1,1,1,None),
        ('existing_rank_file',1,1,None,'FileExistsError'),
        ('barrier',1,1,1,None),
        ('unsupported_build',1,1,1,None),
        ('rank1_missing_directory',1,1,1,None),
        ('destroy',1,1,None,'RuntimeError'),
    ]:
        d=ROOT/scenario;d.mkdir();out_dir=d/'destination';events=[];import_events=[]
        if scenario in ('existing_directory','existing_rank_file'):
            out_dir.mkdir()
        if scenario=='existing_rank_file': (out_dir/'rank0.json').write_text('PREEXISTING\n')
        dist=types.ModuleType('torch.distributed')
        def init(backend):
            events.append('init:'+backend)
            if scenario=='init': raise RuntimeError('INJECTED_INIT_FAILURE')
        def barrier():
            events.append('barrier')
            if scenario=='barrier': raise RuntimeError('INJECTED_BARRIER_FAILURE')
        def destroy():
            events.append('destroy')
            if scenario=='destroy': raise RuntimeError('INJECTED_DESTROY_FAILURE')
        dist.init_process_group=init;dist.barrier=barrier;dist.destroy_process_group=destroy
        fake=types.ModuleType('torch');fake.__version__='TEST_SETUP_STUB'
        fake.distributed=dist
        fake.nn=types.SimpleNamespace(functional=types.ModuleType('torch.nn.functional'))
        fake.set_num_threads=lambda n:events.append('set_num_threads:'+str(n))
        fake.set_num_interop_threads=lambda n:events.append('set_num_interop_threads:'+str(n))
        fake.manual_seed=lambda n:events.append('seed:'+str(n))
        def device(name):
            events.append('device:'+name)
            if scenario=='device': raise RuntimeError('INJECTED_DEVICE_FAILURE')
            return 'TEST_ONLY_DEVICE'
        fake.device=device
        fake.neuron=types.SimpleNamespace() if scenario=='sync_api' else types.SimpleNamespace(synchronize=lambda:events.append('sync'))
        consumer=types.ModuleType('consumer_nki')
        def build_kernel(*,enabled=False):
            events.append('build:'+str(enabled))
            raise RuntimeError('INJECTED_UNSUPPORTED_NKI_API')
        consumer.build_kernel=build_kernel
        reference=types.ModuleType('reference');reference.__dict__.update(source('reference.py'))
        def imports(name,*a,**kw):
            if name.split('.')[0] in ('torch','torch_neuronx','consumer_nki','reference','nki'):
                import_events.append(name)
                if name.startswith('nki'): raise AssertionError('no NKI import allowed')
                if name=='torch_neuronx':
                    if scenario=='native_import': raise ImportError('INJECTED_NATIVE_IMPORT_FAILURE')
                    return types.ModuleType('torch_neuronx')
                if name in ('torch','torch.distributed','torch.nn.functional'): return fake
                if name=='consumer_nki': return consumer
                if name=='reference': return reference
                raise AssertionError('unexpected native import '+name)
            return old_import(name,*a,**kw)
        os.environ.update(WORLD_SIZE='3' if scenario=='world_binding' else '4',NEURON_LOGICAL_NC_CONFIG='2',RANK='bad' if scenario=='rank_parse' else ('1' if scenario=='rank1_missing_directory' else '0'))
        sys.argv=['check_native.py','--enable-experimental','--label','bounded-setup-test','--out-dir',str(out_dir),'--expected-kernel-sha256','bad' if scenario=='support_hash' else hashes['consumer_nki.py'],'--expected-reference-sha256',hashes['reference.py']]
        if scenario=='missing_arguments': sys.argv=['check_native.py','--enable-experimental']
        arguments=sys.argv[:]
        importlib.metadata.version=lambda name:'TEST_STUB_NO_INSTALLED_API_INVENTORY'
        builtins.__import__=imports
        stdout,stderr=io.StringIO(),io.StringIO();ret=None;error=None;trace=None
        try:
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                try: ret=checker['main']()
                except BaseException as e:
                    error=type(e).__name__;trace=traceback.format_exc()
        finally:
            builtins.__import__=old_import;importlib.metadata.version=old_version
            sys.argv=old_argv
            for k,v in environment.items():
                if v is None: os.environ.pop(k,None)
                else: os.environ[k]=v
        (d/'stdout.txt').write_text(stdout.getvalue());(d/'stderr.txt').write_text(stderr.getvalue())
        if trace: (d/'exception.txt').write_text(trace)
        record=dict(scenario=scenario,argv=arguments,events=events,stubbed_imports=import_events,returned=ret,exception_type=error,init_calls=sum(x.startswith('init:') for x in events),destroy_calls=events.count('destroy'),expected=dict(init_calls=expected_init,destroy_calls=expected_destroy,returned=expected_return,exception_type=expected_error),native_executed=False)
        dump(d/'result.json',record);records.append(record)
        check(record['init_calls']==expected_init and record['destroy_calls']==expected_destroy and ret==expected_return and error==expected_error,scenario+' boundary mismatch')
        if expected_return==1:
            receipt=json.loads(stdout.getvalue())
            check(receipt['status']=='REFUSE_OR_UNSUPPORTED' and 'error' in receipt,scenario+' failure legibility')
        if scenario=='existing_rank_file': check((out_dir/'rank0.json').read_text()=='PREEXISTING\n','exclusive file overwritten')
        if expected_destroy==0: check(not (out_dir/'rank0.json').exists(),scenario+' no preguard receipt')
    dump(ROOT/'setup_summary.json',records)
    return records

try:
    bf16=bf16_check();setup=setup_checks()
    result=dict(status='PASS',scope='CPU BF16 conversion and full checker main with disclosed setup stubs',bf16_cases=bf16['cases'],setup_scenarios=len(setup),peak_live_tensor_storage_bytes=bf16['peak_live_tensor_storage_bytes'],known_unchanged_no_cleanup_after_init=['device construction failure','missing synchronize API'],native_imported=False,native_executed=False,launch_authorized=False)
    dump(ROOT/'summary.json',result);print(json.dumps(result))
except BaseException as e:
    dump(ROOT/'FAILED.json',dict(status='REFUSE',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc()))
    raise
