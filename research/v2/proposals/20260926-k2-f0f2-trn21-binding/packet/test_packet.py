"""Tiny implementer fault/association checks, not native or independent review."""
import copy
import json
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import sys
import tempfile
import time
import common
from common import *
from monitor import Processes, guards, subreaper, tree_bytes, sample
from evidence import Writer, Invocation, blob_from_worker
from artifacts import compare_fp32
from fixtures import construct_metadata, validate_metadata
from worker import TensorLedger, capture
from host_driver import bounded_run


def reject(fn):
    try:
        fn()
    except (RuntimeError, ValueError, FileNotFoundError):
        return
    raise RuntimeError('NEGATIVE_ACCEPTED')


def main():
    root = Path(tempfile.mkdtemp(prefix='k2-f0f2-packet-test-', dir='/home/poojith-devan'))
    writer = Writer(root/'artifacts')
    checks = []
    def check(name, fn):
        fn()
        checks.append(name)
    def packed(words):
        return writer.put([struct.pack('<'+'I'*len(words),*words)])
    expected = packed([0,0x3f800000,0x10000,0x80010000,0x40000000,0,0,0])
    for name, words in (
        ('zero',[0]*8), ('stale',[0,0x40000000,0x10000,0x80010000,0x40000000,0,0,0]),
        ('missing_owner',[0,0x3f800000,0x10000,0x80010000,0,0,0,0]),
        ('one_bit',[0,0x3f800001,0x10000,0x80010000,0x40000000,0,0,0]),
        ('signed_zero',[0x80000000,0x3f800000,0x10000,0x80010000,0x40000000,0,0,0]),
        ('nonfinite',[0,0x7f800000,0x10000,0x80010000,0x40000000,0,0,0])):
        actual = packed(words)
        check(name, lambda a=actual: require(compare_fp32(writer.check(a),writer.check(expected),8)['verdict'] != 'EXACT_FINITE', name))
    check('missing_blob',lambda:reject(lambda:writer.check(dict(sha256='0'*64,bytes=32))))
    check('wrong_shape',lambda:reject(lambda:compare_fp32(writer.check(expected),writer.check(expected),9)))
    inv = Invocation(0,0,'challenge',dict(pid=os.getpid(),start=identity(os.getpid())['start']),{'ids'},'source')
    inv.output = expected
    check('fabricated_reference_as_actual',lambda:reject(inv.complete))
    check('missing_route',lambda:reject(Invocation(0,0,'n',{},set(),'s').complete))
    check('no_pretransport',lambda:reject(lambda:inv.call_start(dict(event='call_start',nonce='challenge',ordinal=0))))
    inv.pre = {'ids'}
    inv.call_start(dict(event='call_start',nonce='challenge',ordinal=0))
    bad = dict(event='call_end',nonce='challenge',ordinal=0,output_device='neuron',output_dtype='torch.bfloat16',
               output_shape=[8192,1024],tangent_dtype='torch.bfloat16',elapsed=0.1)
    check('wrong_output_dtype',lambda:reject(lambda:inv.call_end(bad)))
    check('replayed_nonce',lambda:reject(lambda:inv.call_end(dict(bad,nonce='stale'))))
    class FakeCPU:
        class device:
            type = 'cpu'
    check('bypass_native_cpu_object',lambda:reject(lambda:capture(None,{},'output','gradient',FakeCPU(),TensorLedger())))
    # Header pretending to associate an already-existing blob is not a byte stream.
    a,b = socket.socketpair()
    send(a,dict(actual_blob=expected,event='blob'))
    check('fabricated_blob_association',lambda:reject(lambda:blob_from_worker(b,writer,inv,'output','gradient',expected,'torch.float32',[8],fp32=True)))
    a.close(); b.close()
    # Source-modified native bypass cannot reuse the externally frozen pins.
    fake_root=root/'fake-source'; fake_root.mkdir()
    fake_source=fake_root/'worker.py'; fake_source.write_bytes(b'fake native output\n')
    (fake_root/'SOURCE_PINS.json').write_text(json.dumps({'worker.py':'0'*64}))
    old_here=common.HERE
    try:
        common.HERE=fake_root
        check('source_modified_bypass',lambda:reject(common.source_pins))
    finally:
        common.HERE=old_here
    ids = [0,3,0,1,3,0,1,2]
    meta = construct_metadata(ids,4,2)
    variants = []
    for field,index,value in ((0,0,4),(1,1,4),(2,0,1),(2,0,2),(2,0,8),(2,0,-1),(3,0,9)):
        value_meta=copy.deepcopy(meta); value_meta[field][index]=value; variants.append(value_meta)
    value_meta=copy.deepcopy(meta); value_meta[2][0],value_meta[2][1]=value_meta[2][1],value_meta[2][0]; variants.append(value_meta)
    for i,value_meta in enumerate(variants):
        check('metadata_corrupt_'+str(i),lambda m=value_meta:reject(lambda:validate_metadata(ids,m,4,2)))
    check('stale_metadata',lambda:reject(lambda:validate_metadata([1]+ids[1:],meta,4,2)))
    check('missing_position',lambda:reject(lambda:validate_metadata(ids,(meta[0],meta[1],meta[2][:-1],meta[3]),4,2)))
    for bad_id in (-1,4):
        check('invalid_id_'+str(bad_id),lambda v=bad_id:reject(lambda:construct_metadata([v],4,2)))
    base=dict(at=1.0,memavailable=24*GIB,rss=MIB,free=[16*GIB],cache=0,artifacts=0,logs=0)
    for key,value in (('at',3.1),('memavailable',4*GIB-1),('rss',20*GIB+1),('free',[6*GIB]),
                      ('cache',6*GIB+1),('artifacts',2*GIB+1),('logs',128*MIB+1)):
        check('monitor_'+key,lambda k=key,v=value:reject(lambda:guards(dict(base,**{k:v}),1.0)))
    check('write_reservation_cap',lambda:reject(lambda:writer.reserve(2*GIB+1)))
    ledger=TensorLedger(); ledger.reserve('a',448*MIB)
    check('tensor_cap',lambda:reject(lambda:ledger.reserve('b',1)))
    unsafe=root/'unsafe'; unsafe.mkdir(); (unsafe/'link').symlink_to(root/'artifacts')
    check('scratch_symlink',lambda:reject(lambda:tree_bytes(unsafe)))
    # Actual tiny Linux child failures; only newly created owned processes killed.
    subreaper()
    child_results=[]
    for mode in ('stuck','double_fork','loss'):
        tracker=Processes()
        child=subprocess.Popen([sys.executable,str(HERE/'fault_child.py'),mode],stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,start_new_session=True)
        try:
            if mode != 'loss':
                require(select.select([child.stdout],[],[],2)[0], 'FAULT_CHILD_NOT_READY')
                require(child.stdout.readline()==b'ready\n','FAULT_READY')
            tracker.observe()
            if mode=='loss':
                require(child.wait(timeout=2)==7,'LOSS_NOT_OBSERVED')
                tracker.observe()
            now=time.monotonic()
            result=tracker.stop(now+0.08,now+1.0)
            require(not result['survivors'],'FAULT_SURVIVOR')
            child_results.append(dict(mode=mode,known={p:s for p,(s,fd) in tracker.known.items()},
                                      demonstrated_exits=tracker.exits,termination=result))
            checks.append('owned_'+mode)
        finally:
            child.stdout.close(); child.stderr.close()
            if child.poll() is None:
                child.kill(); child.wait(timeout=1)
    # Disabled default must fail before importing SDK or creating host output.
    disabled=subprocess.run([sys.executable,str(HERE/'host_driver.py')],capture_output=True,text=True,timeout=3)
    require(disabled.returncode != 0 and 'DISABLED_NO_SEPARATE_HOST_RELEASE' in disabled.stderr,'DEFAULT_ENABLEMENT')
    checks.append('default_disabled')
    check('bounded_pipe_output_cap',lambda:reject(lambda:bounded_run([sys.executable,str(HERE/'fault_child.py'),'output'],timeout=3)))
    for directory in ('cache','tmp','logs'):
        (root/directory).mkdir()
    live_sample=sample(root,Processes())
    require(live_sample['rss'] > 0 and any(p['pid']==os.getpid() for p in live_sample['processes']), 'ACTUAL_RSS_SAMPLE')
    checks.append('actual_pid_start_rss_sample')
    result=dict(status='TINY_IMPLEMENTER_CHECKS_ONLY', checks=checks, process_faults=child_results,
                artifacts=str(root), resource_sample=live_sample,
                fullsize_or_native_executed=False, independent_review=False,
                source_hashes={p.name:digest(p) for p in sorted(HERE.glob('*.py'))})
    writer.record('tiny-results.json',result)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
