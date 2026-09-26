import os, sys, time, json, socket, subprocess, signal, fcntl, errno
from pathlib import Path
from fractions import Fraction
P=Path('/home/poojith-devan/k2-f0f2-repair-review-EHvK9J')
R=Path(__file__).parent
sys.path.insert(0,str(P))
import host_driver as h, monitor as m, coordinator as c, common, oracle, fixtures

def check(v,msg):
    if not v: raise RuntimeError(msg)

def driver(mode):
    root=R/('job-'+mode)
    fd=os.open(R/('lock-'+mode),os.O_RDWR|os.O_CREAT,0o600)
    fcntl.flock(fd,fcntl.LOCK_EX)
    h.BINDING=dict(output_root=str(root),label=h.LABEL,hostname=socket.gethostname(),independent_review_sha256='local-test',supervisor_release_sha256='local-test',stop_roots=[],scientific_source=str(P/'oracle.py'),project_root=str(R),project_commit='test')
    h.source_pins=lambda: {}
    h.idle_guard.inherited_lock=lambda *a: {'fd':fd}
    h.digest=lambda p:'36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'
    h.free_bytes=lambda p:32*h.GIB
    h.memavailable=lambda:32*h.GIB
    h.bounded_run=lambda argv,**kw:subprocess.CompletedProcess(argv,0,'test\n' if 'rev-parse' in argv else '', '')
    calls=0
    def idle(*args):
        nonlocal calls
        calls+=1
        if calls==1: raise RuntimeError('independent-preflight-failure')
        if mode=='late':
            pid=os.fork()
            if pid==0:
                os.setsid()
                signal.signal(signal.SIGTERM,signal.SIG_IGN)
                Path(R/'late-pid').write_text(str(os.getpid()))
                time.sleep(30)
                os._exit(0)
            time.sleep(.05)
    h.idle_guard.check_idle=idle
    original=Path.open
    def opening(path,*a,**kw):
        if path.name=='terminal.json' and mode=='enospc': raise OSError(errno.ENOSPC,'independent injection')
        return original(path,*a,**kw)
    Path.open=opening
    if mode=='exists':
        old=idle
        def idle_existing(*args):
            if (root/'logs').exists(): (root/'logs'/'terminal.json').write_text('existing evidence')
            return old(*args)
        h.idle_guard.check_idle=idle_existing
    print('ENTER',flush=True)
    result=h.main()
    print(json.dumps({'result':result,'calls':calls}),flush=True)

def peer_child():
    s=socket.socket(socket.AF_UNIX)
    s.connect(sys.argv[1])
    common.send(s,{'pid':os.getpid()})
    s.recv(1)

def exact_value(w):
    e=(w>>23)&255; f=w&0x7fffff
    v=Fraction(f if e==0 else f+2**23)*Fraction(2)**(-149 if e==0 else e-150)
    return -v if w>>31 else v

def ref(a,b):
    value=exact_value(a)+exact_value(b)
    if not value:return 0x80000000 if a==b==0x80000000 else 0
    sign=0x80000000 if value<0 else 0; value=abs(value)
    lo,hi=0,0x7f7fffff
    while lo<hi:
        mid=(lo+hi+1)//2
        if exact_value(mid)<=value:lo=mid
        else:hi=mid-1
    if lo==0x7f7fffff:
        return sign|(0x7f800000 if value>=exact_value(lo)+Fraction(2)**103 else lo)
    dl=value-exact_value(lo); du=exact_value(lo+1)-value
    return sign|(lo if dl<du or (dl==du and lo%2==0) else lo+1)

def run():
    result={'scope':'independent CPU repair review; no native execution','pins':common.source_pins()}
    check(h.BINDING is None,'binding enabled')
    for mode in ('enospc','exists','late'):
        proc=subprocess.Popen([sys.executable,'-B',__file__,'driver',mode],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            check(proc.stdout.readline().strip()=='ENTER','child failed before main')
            time.sleep(.7)
            fd=os.open(R/('lock-'+mode),os.O_RDWR)
            def locked():
                try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
                except BlockingIOError:return True
            if mode!='late':
                check(proc.poll() is None and locked(),'failed persistence released lock')
                proc.send_signal(signal.SIGTERM);time.sleep(.2)
                check(proc.poll() is None and locked(),'TERM bypassed hold')
                result[mode]={'held_before_and_after_TERM':True}
            else:
                out,err=proc.communicate(timeout=3)
                check(proc.returncode==0,err)
                terminal=json.loads((R/'job-late/logs/terminal.json').read_text())
                check(terminal['postflight']['final_quiescence']['kernel_echild'],'missing ECHILD')
                pid=int((R/'late-pid').read_text());check(not Path('/proc/%d'%pid).exists(),'late survivor')
                result[mode]={'stdout':out,'terminal':terminal}
            os.close(fd)
        finally:
            if proc.poll() is None:proc.kill()
            out,err=proc.communicate(timeout=3)
            result.setdefault(mode,{})['stderr']=err
    tracker=m.Processes();m.subreaper()
    helper_results=[]
    for detach in (False,True):
        code="import os,signal,time; p=os.fork(); signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        code+=("(os.close(1),os.close(2),os.setsid(),time.sleep(30)) if p==0 else os._exit(0)" if detach else 'time.sleep(30)')
        events=[];start=time.monotonic()
        try:h.bounded_run([sys.executable,'-c',code],timeout=1.2,tracker=tracker,events=events);raise RuntimeError('accepted orphan')
        except RuntimeError as exc:check(str(exc)!='accepted orphan','accepted orphan');error=str(exc)
        proof=tracker.quiescent();check(time.monotonic()-start<1.5,'timeout overrun')
        helper_results.append(dict(detach=detach,error=error,events=events,proof=proof))
    result['helpers']=helper_results
    address=str(R/'peer.sock');server=socket.socket(socket.AF_UNIX);server.bind(address);server.listen(4)
    children=[];sockets=[];seen=set();authenticated=[];negatives=[]
    try:
        for rank in range(4):
            env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='4',NEURON_LOGICAL_NC_CONFIG='2',INDEPENDENT_PEER='1')
            children.append(subprocess.Popen([sys.executable,__file__,address],env=env))
            s,_=server.accept();sockets.append(s);common.receive(s);info=common.peer(s)
            hello=dict(rank=rank,world=4,pid=info['pid'],start=info['start'],pins={})
            authenticated.append(c.authenticate_worker(info,hello,common.identity(os.getpid()),seen,{},Path(address),Path(__file__),Path(sys.executable)))
            for label,changed,used,launcher in [('duplicate',hello,seen,common.identity(os.getpid())),('false_rank',dict(hello,rank=(rank+1)%4),set(),common.identity(os.getpid())),('launcher',hello,set(),info)]:
                try:c.authenticate_worker(info,changed,launcher,used,{},Path(address),Path(__file__),Path(sys.executable));raise RuntimeError('BAD_ACCEPT')
                except RuntimeError as exc:check(str(exc)!='BAD_ACCEPT',label);negatives.append([label,str(exc)])
    finally:
        for s in sockets:s.close()
        for child in children:child.wait(timeout=3)
        server.close()
    result['peers']={'authenticated':authenticated,'negatives':negatives}
    count=0
    for ch in range(1024):
        a=0
        for _ in range(4):a=ref(a,(0x3800+ch)<<16)
        check(a==oracle.recurrence([0x3800+ch]*4),'F0 numerical');count+=1
    witnesses=[]
    for row in (0,127,128,4095,4096,8191):
        seq=[t for t in range(32768) if fixtures.token_id('F2',t)==row]
        for ch in range(8):
            words=[fixtures.tangent_bits('F2',t,ch) for t in seq];a=0
            for w in words:a=ref(a,w<<16)
            check(a==oracle.recurrence(words),'F2 numerical');count+=1;witnesses.append([row,ch,hex(a)])
    result['numerical']={'reference':'Fraction exact sum, binary search finite FP32 lattice, nearest-even','recurrences':count,'F2':witnesses}
    (R/'raw-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'recurrences':count,'raw':str(R/'raw-results.json')}))

if __name__=='__main__':
    if os.environ.get('INDEPENDENT_PEER'):peer_child()
    elif len(sys.argv)>1 and sys.argv[1]=='driver':driver(sys.argv[2])
    else:run()
