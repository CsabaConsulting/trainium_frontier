"""Implementer regressions for fresh REFUSE; real local processes, no native SDK."""
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
from common import *
from monitor import Processes, subreaper
from coordinator import authenticate_worker
import host_driver as h
import coordinator


def spoof_launcher(root):
    clients=[]
    try:
        for rank in range(4):
            sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(2)
            sock.connect(str(root/'control.sock'));clients.append(sock)
            own=identity(os.getpid())
            send(sock,dict(rank=rank,world=4,pid=own['pid'],start=own['start'],pins={'fixture':'pins'}))
        clients[0].recv(1)
    finally:
        for sock in clients:sock.close()


def main_fault(mode, root):
    lock=(root/'test-only.lock').open('a+b')
    fcntl.flock(lock,fcntl.LOCK_EX)
    out=root/'job'
    h.BINDING=dict(output_root=str(out),label=LABEL,hostname=socket.gethostname(),
        independent_review_sha256='local-mock',supervisor_release_sha256='local-mock',stop_roots=[],
        scientific_source=str(root/'mock-science'),project_root=str(root),project_commit='mock')
    h.BINDING.update(launch_authorized=True,stage_authorized=True,packet_root=str(HERE),
                     expected_lock_device=os.fstat(lock.fileno()).st_dev,
                     expected_lock_inode=os.fstat(lock.fileno()).st_ino)
    h.source_pins=lambda:{'fixture':'pins'}
    h.digest=lambda p:'36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'
    h.idle_guard.inherited_lock=lambda *a:dict(fd=lock.fileno(),device=os.fstat(lock.fileno()).st_dev,
                                             inode=os.fstat(lock.fileno()).st_ino)
    h.free_bytes=lambda p:32*GIB
    h.memavailable=lambda:32*GIB
    h.bounded_run=lambda cmd,**kw:subprocess.CompletedProcess(cmd,0,'' if 'status' in cmd else 'mock','')
    calls=[]
    real_popen=subprocess.Popen
    spawned=[]
    def idle(*args):
        calls.append(time.monotonic())
        if mode=='terminal_enospc':
            raise RuntimeError('MOCK_IDLE_FAILURE')
        if mode=='postflight_descendant' and len(calls)==2:
            # Deliberately create a child AFTER the first termination census,
            # beyond helper's own cleanup, to exercise final main recensus.
            p=real_popen([sys.executable,str(HERE/'fault_child.py'),'double_fork'],
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            require(select.select([p.stdout],[],[],2)[0],'POST_HELPER_READY')
            require(p.stdout.readline()==b'ready\n','POST_HELPER_READY_BYTES')
            p.wait(timeout=2); p.stdout.close(); p.stderr.close(); spawned.append(p.pid)
    h.idle_guard.check_idle=idle
    def local_child(command,**kw):
        return real_popen([sys.executable,str(HERE/'fault_child.py'),'loss'],**kw)
    h.subprocess.Popen=local_child
    h.sample=lambda root,tracker:dict(at=time.monotonic(),duration=0,memavailable=32*GIB,
        rss=tracker.observe()[1],free=[32*GIB],cache=0,artifacts=0,logs=0)
    original_open=Path.open
    class BadClose:
        def __init__(self,stream):self.stream=stream
        def __getattr__(self,name):return getattr(self.stream,name)
        def close(self):
            self.stream.close()
            raise OSError('MOCK_MONITOR_CLOSE_FAILURE')
    def fault_open(path,*args,**kwargs):
        if mode=='terminal_enospc' and path.name=='terminal.json':
            raise OSError(28,'MOCK_ENOSPC_TERMINAL')
        stream=original_open(path,*args,**kwargs)
        return BadClose(stream) if mode=='monitor_close' and path.name=='resource-samples.jsonl' else stream
    real_require=h.require
    def fault_require(ok,message):
        if mode=='log_cap' and message=='DRIVER_LOG_CAP':
            raise RuntimeError('MOCK_DRIVER_LOG_CAP')
        return real_require(ok,message)
    h.require=fault_require
    real_hold=h.protection_hold
    def marked_hold():
        print('HOLD_ENTERED',flush=True)
        real_hold()
    h.protection_hold=marked_hold
    with patch.object(Path,'open',fault_open):
        result=h.main()
    print(json.dumps(dict(main_result=result,idle_calls=len(calls),spawned=spawned)),flush=True)


def rejected(fn, expected):
    try:fn()
    except RuntimeError as exc:
        require(str(exc)==expected,'WRONG_REJECTION:'+str(exc))
        return str(exc)
    raise RuntimeError('EXPECTED_REJECTION:'+expected)


def regressions():
    root=Path(tempfile.mkdtemp(prefix='k2-repair-regressions-',dir='/home/poojith-devan'))
    subreaper()
    result=dict(scope='LOCAL_IMPLEMENTER_REGRESSIONS_ONLY',root=str(root),native=False,independent_review=False)
    holds=[]
    for mode in ('terminal_enospc','monitor_close','log_cap'):
        job=root/mode;job.mkdir()
        p=subprocess.Popen([sys.executable,str(HERE/'test_repairs.py'),mode,str(job)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        try:
            require(select.select([p.stdout],[],[],5)[0],'HOLD_NOT_OBSERVED:'+mode)
            line=p.stdout.readline().decode().strip()
            require(line=='HOLD_ENTERED','MAIN_ESCAPED:'+mode+':'+line)
            with (job/'test-only.lock').open('a+b') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:raise RuntimeError('LOCK_RELEASED:'+mode)
            p.send_signal(signal.SIGTERM)
            time.sleep(0.05)
            require(p.poll() is None,'SIGNAL_ESCAPED_HOLD')
            holds.append(dict(mode=mode,hold_observed=True,lock_held=True,term_did_not_release=True))
        finally:
            p.kill(); p.wait(timeout=2);p.stdout.close();p.stderr.close()
    result['real_lock_holds']=holds
    # Same timeout/orphan reproductions as refusal, now through actual helper.
    tracker=Processes(); events=[]
    forks=("import os,signal,time; p=os.fork(); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)",
           "import os,time; p=os.fork(); (os.close(1),os.close(2),time.sleep(30)) if p==0 else None")
    helper_results=[]
    for code in forks:
        try:h.bounded_run([sys.executable,'-c',code],timeout=1,tracker=tracker,events=events)
        except RuntimeError as exc:error=str(exc)
        else:raise RuntimeError('ORPHAN_HELPER_ACCEPTED')
        proof=tracker.quiescent()
        helper_results.append(dict(error=error,proof=proof))
    result['helpers']=dict(results=helper_results,events=events)
    post=root/'postflight_descendant';post.mkdir()
    completed=subprocess.run([sys.executable,str(HERE/'test_repairs.py'),'postflight_descendant',str(post)],
        capture_output=True,text=True,timeout=5)
    require(completed.returncode==0,'POSTFLIGHT_PROBE:'+completed.stderr)
    terminal=json.loads((post/'job/logs/terminal.json').read_text())
    require(terminal['postflight']['final_quiescence']['kernel_echild'] and
            not terminal['postflight']['final_termination']['survivors'],'FINAL_REAP_MISSING')
    tracker.quiescent()
    result['postflight_final_recensus']=dict(stdout=completed.stdout,terminal=terminal)
    # Reproduce prior actual-main admission test: launcher is one real PID
    # claiming four ranks. It must be rejected before any route is scheduled.
    admission=root/'admission';admission.mkdir();(admission/'artifacts').mkdir()
    launched=[];scheduled=[];real_popen=subprocess.Popen
    def fake_launcher(*args,**kwargs):
        child=real_popen([sys.executable,str(HERE/'test_repairs.py'),'spoof_launcher',str(admission)],
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        launched.append(child);return child
    def forbidden_route(*args):
        scheduled.append(True)
        raise RuntimeError('ROUTE_MUST_NOT_BE_SCHEDULED')
    lock_fd=os.open(root/'rank-only.lock',os.O_CREAT|os.O_RDWR,0o600)
    with patch.object(coordinator,'canonical',lambda *a:{'F0':{},'F2':{}}), \
         patch.object(coordinator,'source_pins',lambda:{'fixture':'pins'}), \
         patch.object(coordinator,'route',forbidden_route), \
         patch.object(coordinator.subprocess,'Popen',fake_launcher), \
         patch.dict(os.environ,{'K2_LOCK_FD':str(lock_fd)}), \
         patch.object(sys,'argv',['coordinator.py',str(admission),str(time.monotonic())]):
        status=coordinator.main()
    os.close(lock_fd)
    for child in launched:child.communicate(timeout=3)
    receipt=json.loads((admission/'artifacts/coordinator-terminal.json').read_text())
    require(status==1 and 'LAUNCHER_IS_NOT_WORKER' in receipt.get('error','') and
            not scheduled and not list((admission/'artifacts').glob('rank*-environment.json')), 'COORDINATOR_ADMISSION_BYPASS')
    result['actual_coordinator_refuses_one_pid_four_ranks']=receipt
    # Actual SO_PEERCRED plus real env/argv/executable, four distinct processes.
    peer_root=root/'peers';peer_root.mkdir()
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    listener.bind(str(peer_root/'peers.sock'));listener.listen(4);listener.settimeout(3)
    children=[];clients=[];seen=set();peers=[];negatives={}
    launcher=identity(os.getpid())
    try:
        for rank in range(4):
            env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='4',NEURON_LOGICAL_NC_CONFIG='2')
            children.append(subprocess.Popen([sys.executable,str(HERE/'peer_fixture.py'),str(peer_root)],env=env))
        for _ in range(4):
            sock,_=listener.accept();sock.settimeout(3);clients.append(sock)
            info,hello=peer(sock),receive(sock)
            arguments=(info,hello,launcher,seen,{'fixture':'pins'},peer_root,HERE/'peer_fixture.py',Path(sys.executable))
            proof=authenticate_worker(*arguments);peers.append(proof)
            negatives['duplicate_pid']=rejected(lambda:authenticate_worker(*arguments),'DUPLICATE_WORKER_PROCESS')
            spoof=dict(hello,rank=(hello['rank']+1)%4)
            negatives['false_rank_env']=rejected(lambda:authenticate_worker(info,spoof,launcher,set(),{'fixture':'pins'},peer_root,HERE/'peer_fixture.py',Path(sys.executable)),'ACTUAL_WORKER_ENVIRONMENT')
            negatives['launcher_as_rank']=rejected(lambda:authenticate_worker(launcher,hello,launcher,set(),{},peer_root,HERE/'peer_fixture.py',Path(sys.executable)),'LAUNCHER_IS_NOT_WORKER')
        require(len(seen)==4,'NOT_FOUR_ACTUAL_PEERS')
        result['worker_peers']=dict(authenticated=peers,negatives=negatives)
    finally:
        for sock in clients:
            send(sock,{'finish':True});sock.close()
        listener.close()
        for child in children:
            child.wait(timeout=3)
    tracker.quiescent()
    result['source_hashes']={p.name:digest(p) for p in sorted(HERE.glob('*.py'))}
    (root/'raw-regressions.json').write_text(json.dumps(result,sort_keys=True)+'\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='spoof_launcher':spoof_launcher(Path(sys.argv[2]))
    elif len(sys.argv)>1:main_fault(sys.argv[1],Path(sys.argv[2]))
    else:regressions()
