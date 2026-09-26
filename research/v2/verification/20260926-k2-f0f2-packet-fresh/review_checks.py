"""Fresh reviewer CPU-only probes. Never loads SDK or full-size tensors."""
import contextlib
import ctypes
import fcntl
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from unittest.mock import patch

PACKET = Path('/home/poojith-devan/k2-f0f2-fresh-review-PC6TPV')
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(PACKET))
import common, monitor, host_driver, oracle, fixtures, artifacts, evidence, coordinator

def check(value, message):
    if not value:
        raise RuntimeError(message)

def rational(word):
    sign = -1 if word >> 31 else 1
    exp = (word >> 23) & 255
    mant = word & 0x7fffff
    check(exp != 255, 'finite reference inputs')
    return sign * Fraction(mant if exp == 0 else (1 << 23) + mant) * Fraction(2) ** (-149 if exp == 0 else exp - 150)

def independent_add(a, b):
    """Exact rational sum; binary search ordered finite float lattice, nearest-even.

    Deliberately differs from implementation bit-length/shift rounding algorithm.
    """
    value = rational(a) + rational(b)
    if value == 0:
        return 0x80000000 if a == b == 0x80000000 else 0
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    # Overflow threshold is midpoint from max finite to hypothetical 2**128.
    if value >= (rational(0x7f7fffff) + Fraction(2)**128) / 2:
        return sign | 0x7f800000
    low, high = 0, 0x7f7fffff
    while low < high:
        mid = (low + high + 1) // 2
        if rational(mid) <= value:
            low = mid
        else:
            high = mid - 1
    if low == 0x7f7fffff:
        return sign | low
    dl, dh = value - rational(low), rational(low + 1) - value
    result = low if dl < dh or (dl == dh and low % 2 == 0) else low + 1
    return sign | result

def exact_math():
    rng = random.Random(42)
    edges = [0, 0x80000000, 1, 0x80000001, 0x7fffff, 0x800000, 0x00810000,
             0x80800000, 0x3f800000, 0x33800000, 0x4b800000, 0xcb800000,
             0x7f7fffff, 0xff7fffff]
    pairs = [(a,b) for a in edges for b in edges]
    for _ in range(1200):
        a,b = rng.getrandbits(32),rng.getrandbits(32)
        if ((a >> 23)&255) != 255 and ((b >> 23)&255) != 255:
            pairs.append((a,b))
    for a,b in pairs:
        check(oracle.add32(a,b) == independent_add(a,b), 'independent RN32 mismatch')
    f0 = []
    for c in range(1024):
        s = 0
        for _ in range(4): s = independent_add(s,(0x3800+c)<<16)
        check(s == oracle.recurrence([0x3800+c]*4), 'F0 recurrence')
        f0.append(s)
    # Scalar full token-ID scan only: never allocates a full geometry tensor.
    overrides = {0:0,128:0,129:0,1:127,130:127,2:128,131:128,3:8191,132:8191,4:4095,133:4095}
    expected_values = [{0:0x4b80,128:0x3f80,129:0xcb80},{0:0x3f80,128:0x3380,129:0xbf80},
                       {1:0x81,130:0x8080},{2:0x8081,131:0x80},{3:1},{3:0x8001},
                       {4:0x8000,133:0},{4:0x8000,133:0x8000}]
    rows = {r:[] for r in (0,127,128,4095,4096,8191)}
    for t in range(32768):
        r = overrides.get(t,t%8192)
        check(fixtures.token_id('F2',t) == r, 'F2 ID recipe')
        if r in rows: rows[r].append(t)
    outputs = {}
    for r,ts in rows.items():
        result=[]
        for c in range(8):
            timeline=[expected_values[c].get(t,0) for t in ts]
            check(timeline == [fixtures.tangent_bits('F2',t,c) for t in ts], 'F2 tangent recipe')
            s=0
            for x in timeline: s=independent_add(s,x<<16)
            check(s==oracle.recurrence(timeline), 'F2 recurrence')
            result.append('%08x'%s)
        outputs[r]=result
    return {'reference':'exact Fraction plus ordered float-lattice binary search', 'pairs':len(pairs),
            'f0_channels':len(f0),'f0_word_hash':hashlib.sha256(struct.pack('<1024I',*f0)).hexdigest(),
            'f2_boundary_rows':outputs,'fullsize_tensor_allocated':False}

def diff_check():
    text=(PACKET/'implementation.diff').read_text()
    verified=[]
    for part in text.split('diff --git ')[1:]:
        lines=part.splitlines(keepends=True)
        name=Path(lines[0].split(' b/',1)[1].strip()).name
        additions=''.join(line[1:] for line in lines if line.startswith('+') and not line.startswith('+++'))
        check(additions.encode()==(PACKET/name).read_bytes(),'diff/snapshot '+name)
        verified.append(name)
    common.source_pins()
    return {'all_added_files_match_snapshots':verified, 'sha256sums_sha256':common.digest(PACKET/'SHA256SUMS')}

def output_attacks():
    root=OUT/'byte-probes'; writer=evidence.Writer(root)
    words=[0,0x3f800000,0x10000,0x80010000,0x40000000,0,0,0]
    ref=writer.put([struct.pack('<8I',*words)])
    results={}
    for name,change in [('signed_zero',(0,0x80000000)),('last_word_one_bit',(7,1)),
                        ('missing_second_owner',(4,0)),('nonfinite',(2,0x7f800000))]:
        altered=words.copy(); altered[change[0]]=change[1]
        actual=writer.put([struct.pack('<8I',*altered)])
        result=artifacts.compare_fp32(writer.check(actual),writer.check(ref),8)
        check(result['verdict']!='EXACT_FINITE',name)
        results[name]=result
    nan=writer.put([struct.pack('<I',0x7fc00001)])
    results['equal_nan']=artifacts.compare_fp32(writer.check(nan),writer.check(nan),1)
    check(results['equal_nan']['verdict']!='EXACT_FINITE','equal NaN')
    # Exercise real streamed association with a correct staged challenge but corrupted retained input.
    for role in ('input_before','input_after','retained'):
        inv=evidence.Invocation(0,1,'independent-nonce',{'pid':os.getpid(),'start':common.identity(os.getpid())['start']},{'tangent'},'test-source')
        a,b=socket.socketpair(); a.settimeout(2); b.settimeout(2)
        errors=[]
        def sender():
            try:
                common.send(a,dict(event='blob',role=role,name='tangent',nonce=inv.nonce,ordinal=1,
                            bytes=ref['bytes'],dtype='torch.float32',shape=[8],device='neuron'))
                common.receive(a)
                a.sendall(struct.pack('<8I',*(words[:-1]+[1])))
            except Exception as e: errors.append(str(e))
        thread=threading.Thread(target=sender);thread.start()
        try:
            evidence.blob_from_worker(b,writer,inv,role,'tangent',ref,'torch.float32',[8],fp32=role=='retained')
            raise RuntimeError('corruption accepted')
        except RuntimeError as e:
            check(str(e) in ('FULL_BYTE_MISMATCH','OUTPUT_NOT_EXACT_FINITE'),str(e))
            results[role+'_corruption']=str(e)
        finally:
            thread.join(3);a.close();b.close()
        check(not thread.is_alive() and not errors,'sender failure')
    return results

def orphan_mode():
    # Runs in its own process without subreaper: mirrors driver preflight ordering.
    code="import os,signal,time; p=os.fork(); signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpid(),flush=True) if p==0 else None; time.sleep(30)"
    error=None
    try: host_driver.bounded_run([sys.executable,'-c',code],timeout=0.2)
    except Exception as e: error=type(e).__name__+': '+str(e)
    print(json.dumps({'bounded_run_error':error,'probe_pid':os.getpid()}),flush=True)

def orphan_test():
    # Reviewer becomes subreaper solely to contain/reap probe's orphan safely.
    monitor.subreaper()
    tracker=monitor.Processes()
    p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'orphan'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout,stderr=p.communicate(timeout=4)
    rows,_=tracker.observe()
    survivors=[r for r in rows if r['pid']!=os.getpid()]
    check(survivors,'expected orphan absent')
    adopted=[r for r in survivors if r['ppid']==os.getpid()]
    check(adopted,'reviewer did not adopt escaped preflight descendant')
    now=time.monotonic(); cleanup=tracker.stop(now+0.1,now+2)
    check(not cleanup['survivors'],'reviewer cleanup')
    return {'child_probe_stdout':stdout.decode(),'stderr':stderr.decode(),
            'escaped_preflight_descendants':adopted,'reviewer_cleanup':cleanup,
            'fault':'bounded_run timeout leaves grandchild alive; real driver subreaper starts after preflight'}

def terminal_mode():
    root=OUT/'terminal-job'
    lockpath=OUT/'reviewer-only.lock'
    lock=open(lockpath,'a+b');fcntl.flock(lock,fcntl.LOCK_EX)
    h=host_driver
    h.BINDING={'output_root':str(root),'label':common.LABEL,'hostname':socket.gethostname(),
               'independent_review_sha256':'mock-review','supervisor_release_sha256':'mock-release',
               'stop_roots':[],'scientific_source':str(OUT/'mock-science'),
               'project_root':str(OUT),'project_commit':'mock-commit'}
    h.source_pins=lambda:{'test':'mock'}
    h.digest=lambda path:'36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'
    h.idle_guard.inherited_lock=lambda *a:{'fd':lock.fileno()}
    h.free_bytes=lambda p:32*common.GIB
    h.memavailable=lambda:32*common.GIB
    h.bounded_run=lambda cmd,**kw:subprocess.CompletedProcess(cmd,0,'' if 'status' in cmd else 'mock-commit','')
    def fail_idle(*args): raise RuntimeError('synthetic_failed_postflight')
    h.idle_guard.check_idle=fail_idle
    original_open=Path.open
    def fail_terminal(path,*args,**kwargs):
        if path.name=='terminal.json': raise OSError(28,'synthetic disk full during terminal write')
        return original_open(path,*args,**kwargs)
    try:
        with patch.object(Path,'open',fail_terminal): h.main()
    except Exception as e:
        print(json.dumps({'main_escaped':type(e).__name__+': '+str(e),'original_lock_fd':lock.fileno()}),flush=True)
    # Process exits, releasing reviewer-only lock. Real outer flock also exits when driver does.

def terminal_test():
    p=subprocess.run([sys.executable,'-B',str(Path(__file__)),'terminal'],capture_output=True,text=True,timeout=3)
    check('main_escaped' in p.stdout,'terminal-write reproduction incomplete: '+p.stderr)
    with (OUT/'reviewer-only.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return {'stdout':p.stdout,'stderr':p.stderr,'exit_code':p.returncode,
            'lock_acquirable_after_failed_containment':True,'protection_hold_entered':False}

def postflight_mode():
    monitor.subreaper(); tracker=monitor.Processes()
    now=time.monotonic(); prior=tracker.stop(now+0.1,now+0.2)
    # Postflight helper forks and its direct parent exits successfully with closed child stdio.
    code="import os,time; p=os.fork(); (os.close(1),os.close(2),time.sleep(30)) if p==0 else None"
    result=host_driver.bounded_run([sys.executable,'-c',code],timeout=1)
    rows,_=tracker.observe(); live=[r for r in rows if r['pid']!=os.getpid()]
    now=time.monotonic(); cleanup=tracker.stop(now+0.1,now+1)
    check(live and not cleanup['survivors'],'postflight repro cleanup')
    return {'stop_before_postflight':prior,'helper_returncode':result.returncode,
            'live_descendant_after_successful_postflight_helper':live,'reviewer_cleanup':cleanup}

def guard_tests():
    base=dict(at=1.,memavailable=24*common.GIB,rss=common.MIB,free=[16*common.GIB],cache=0,artifacts=0,logs=0)
    rejected={}
    for key,value in [('at',3.01),('rss',20*common.GIB+1),('cache',6*common.GIB+1),
                      ('artifacts',2*common.GIB+1),('logs',128*common.MIB+1),
                      ('memavailable',4*common.GIB-1),('free',[6*common.GIB])]:
        try: monitor.guards(dict(base,**{key:value}),1.)
        except RuntimeError as e: rejected[key]=str(e)
    check(len(rejected)==7,'guard not closed')
    return {'threshold_rejections':rejected,'unknown_precision':common.UNKNOWN}

def rank_child(root):
    clients=[]
    for rank in range(4):
        s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect(str(Path(root)/'control.sock'))
        own=common.identity(os.getpid())
        common.send(s,dict(rank=rank,pid=own['pid'],start=own['start'],world=4,pins=common.source_pins()))
        clients.append(s)
    for s in clients:
        while s.recv(1024): pass

def rank_test():
    root=OUT/'rank-probe';root.mkdir();(root/'artifacts').mkdir()
    fd=os.open(OUT/'rank-mock.lock',os.O_CREAT|os.O_RDWR,0o600)
    created=[];seen=[]; real_popen=subprocess.Popen
    def fake_popen(*args,**kw):
        p=real_popen([sys.executable,'-B',str(Path(__file__)),'ranks',str(root)],stdin=subprocess.DEVNULL,
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        created.append(p);return p
    def fake_route(sock,writer,rank,peer,*args):
        seen.append({'rank':rank,'peer':peer});raise RuntimeError('REVIEW_STOP_AFTER_PEER_ACCEPTANCE')
    with patch.object(coordinator,'canonical',lambda *a:{'F0':{},'F2':{}}),patch.object(coordinator,'route',fake_route), \
         patch.object(coordinator.subprocess,'Popen',fake_popen),patch.dict(os.environ,{'K2_LOCK_FD':str(fd)}), \
         patch.object(sys,'argv',['coordinator.py',str(root),str(time.monotonic())]):
        status=coordinator.main()
    for p in created:
        p.communicate(timeout=3)
    os.close(fd)
    # FIRST_EXCEPTION may stop before all mocked routes execute, but all four hello receipts were written.
    hellos=[json.loads((root/'artifacts'/('rank%d-environment.json'%r)).read_text()) for r in range(4)]
    check(len({h['pid'] for h in hellos})==1,'same peer mock not exercised')
    return {'status':status,'accepted_hello_ranks':[h['rank'] for h in hellos],
            'accepted_hello_pids':[h['pid'] for h in hellos],'scheduled_routes':seen,
            'scope':'real Unix peer/ancestry acceptance; generation and torchrun launcher mocked; no native execution'}

def main():
    result={'independent':True,'native_executed':False,'fullsize_allocations':False,
            'diff':diff_check(),'math':exact_math(),'outputs':output_attacks(),
            'guards':guard_tests(),'preflight_orphan':orphan_test(),'terminal_hold':terminal_test(),
            'postflight_orphan':postflight_mode(),'rank_association':rank_test()}
    (OUT/'raw-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':
    if len(sys.argv)>1:
        if sys.argv[1]=='orphan': orphan_mode()
        elif sys.argv[1]=='terminal': terminal_mode()
        elif sys.argv[1]=='ranks': rank_child(sys.argv[2])
    else: main()
