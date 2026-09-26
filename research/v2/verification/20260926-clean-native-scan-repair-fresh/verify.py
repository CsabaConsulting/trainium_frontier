import os, sys, json, hashlib, importlib.util, fcntl, signal, subprocess, time
from pathlib import Path
from unittest.mock import patch

BASE=Path('/home/poojith-devan/clean36-host-scan-repair-fresh-y3R9n0')
OUT=Path(__file__).parent
def check(ok, message):
    if not ok: raise RuntimeError(message)
def load(name):
    spec=importlib.util.spec_from_file_location(name,BASE/'packet'/f'{name}.py')
    obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def busy(lock):
    with lock.open('r') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return True
        return False
def events(root):
    p=root/'events.jsonl'
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
def waitfor(fn, seconds=10):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if fn(): return
        time.sleep(.03)
    raise RuntimeError('WAIT_TIMEOUT')

def worker(root,where,kind):
    d=load('host_driver'); g=load('lifetime_guard'); d.ROOT=root
    d.resources=lambda:{'free_disk_bytes':10**12,'available_ram_bytes':10**12}
    lock=root/'lock'; fd=os.open(lock,os.O_RDWR); fcntl.flock(fd,fcntl.LOCK_EX)
    s=os.fstat(fd); guardian=g.Guardian(fd,str(lock),(s.st_dev,s.st_ino),root/'events.jsonl',grace=.1,kill_wait=2)
    child="import os,signal,time; from pathlib import Path; os.setsid(); signal.signal(signal.SIGTERM,signal.SIG_IGN); Path('leaf').write_text(str(os.getpid())); time.sleep(20)"
    d.COMMAND=[sys.executable,'-B','-c',child]; d.ENV=dict(os.environ)
    real=os.scandir
    def fail(path):
        if (root/'trigger').exists() and Path(path)==(root if where=='root' else root/'receipts'):
            raise (PermissionError if kind=='permission' else OSError)(13 if kind=='permission' else 5,'reviewer scan fault',str(path))
        return real(path)
    with patch('os.scandir',fail):
        try:
            d.run_child(guardian)
            (root/'unexpected-success').write_text('FAIL')
        except BaseException as exc:
            (root/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc)}))
        finally:
            guardian.finish(lambda:d.sizes(root))

def main():
    d=load('host_driver'); mode='optimized' if sys.flags.optimize else 'normal'
    run=OUT/mode; run.mkdir()
    check(sha(BASE/'SHA256SUMS')=='d6e6e80997c298cfd69f3e5cadde6bb308dd4d5d2f66da3418b92b76ee824e55','input manifest')
    for line in (BASE/'SHA256SUMS').read_text().splitlines():
        h,n=line.split(maxsplit=1); check(sha(BASE/n)==h,'frozen '+n)
    b=json.loads((BASE/'packet/NATIVE_BINDING.disabled.json').read_text())
    for a,key in [('CHILD_ARGV','child_argv'),('COMMAND','command'),('OUTER','outer_command'),('ENV','environment'),('PINS','support_pins')]:
        check(getattr(d,a)==b[key],'binding '+a)
    for key,pin in json.loads((BASE/'packet/MANIFEST.json').read_text()).items():
        p=BASE/'packet'/key; check(sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'packet '+key)
    src=(BASE/'packet/host_driver.py').read_text()
    old=src.replace('    def scan_error(error):\n        raise error\n','').replace('followlinks=False,onerror=scan_error','followlinks=False')
    check(hashlib.sha256(old.encode()).hexdigest()=='132fa83ae654a3d7921ef8b907f759c6c69d2d3be11035170b56d1d98ff9c54c','exact reverse repair proves unchanged old code')
    for a,k in [('INITIAL_DISK','initial_disk_bytes'),('RUNTIME_DISK','runtime_disk_bytes'),('RAM','initial_ram_bytes'),('ARTIFACT_CAP','artifact_cap_bytes'),('RANK_CAP','rank_receipt_cap_bytes'),('CHILD_SECONDS','child_seconds'),('TERM_SECONDS','term_seconds')]: check(getattr(d,a)==b['resources'][k],a)
    check(d.LOCK_ID==(66305,596823),'lock')
    check(all(b[k] is False for k in ['stage_authorized','launch_authorized','native_launch_authorized','submission_authorized']),'disabled')
    check(all(v is None for v in b['runtime_pins'].values()),'pending runtime')
    check(not (BASE/'packet/RELEASE.json').exists(),'no release')
    fixture=run/'bytes'; fixture.mkdir()
    # Independent exact byte/count oracle, from payload lengths and explicit labels,
    # without filesystem traversal or stat arithmetic from implementation.
    data={'receipts/.rank0.interrupted.tmp':b'abcdefghijklmnopq','receipts/rank2.partial.json':bytes(range(31)),'.hidden':b'xyz','caches/neff/a':bytes(range(101)),'metrics/rank3.log':b'12'}
    oracle={'artifact_bytes':0,'cache_bytes':0,'rank_receipt_bytes':{str(i):0 for i in range(4)}}
    for name,payload in data.items():
        p=fixture/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(payload)
        oracle['cache_bytes' if name.startswith('caches/') else 'artifact_bytes']+=len(payload)
    oracle['rank_receipt_bytes']['0']=17; oracle['rank_receipt_bytes']['2']=31
    check(d.sizes(fixture)==oracle,'independent bytes oracle')
    real=os.scandir; counts=0
    for where in [fixture,fixture/'receipts']:
        for cls in [PermissionError,OSError]:
            error=cls(13 if cls is PermissionError else 5,'reviewer exact failure',str(where))
            def fail(path):
                if Path(path)==where: raise error
                return real(path)
            with patch('os.scandir',fail):
                try: d.sizes(fixture)
                except OSError as got: check(got is error,'original exception identity')
                else: raise RuntimeError('scan accepted partial count')
            counts+=1
    for where in [fixture,fixture/'receipts']:
        where.chmod(0)
        try:
            try: d.sizes(fixture)
            except PermissionError: pass
            else: raise RuntimeError('actual permission scan accepted')
        finally: where.chmod(0o700)
    for where in [fixture/'missing',fixture/'.hidden']:
        try: d.sizes(where)
        except OSError: pass
        else: raise RuntimeError('missing/non-directory accepted')
    scenarios=[]
    for where in ['root','nested']:
        for kind in ['permission','io']:
            root=run/(where+'-'+kind); root.mkdir(); (root/'receipts').mkdir(); (root/'lock').touch()
            log=(root/'worker.log').open('w')
            p=subprocess.Popen([sys.executable,*(['-O'] if sys.flags.optimize else []),'-B',__file__,'worker',str(root),where,kind],stdout=log,stderr=subprocess.STDOUT)
            waitfor(lambda:(root/'leaf').exists()); leaf=int((root/'leaf').read_text())
            st=(root/'lock').stat(); check(busy(root/'lock'),'lock excluded during live child')
            (root/'trigger').touch()
            waitfor(lambda:any(e['event']=='FAILED_LOCK_RETAINED' for e in events(root)))
            ev=events(root)
            check(p.poll() is None and busy(root/'lock'),'park retains actual lock')
            check((st.st_dev,st.st_ino)==((root/'lock').stat().st_dev,(root/'lock').stat().st_ino),'same lock inode')
            check(not (root/'unexpected-success').exists(),'no zero success')
            check(any(e['event']=='signal_owned' and e['signal']==9 for e in ev),'owned pidfd kill')
            check(any(e['event']=='lineage_quiescent' and e['proof']=='waitpid_ECHILD' for e in ev),'ECHILD')
            check(not any(e['event']=='release_ready' for e in ev),'no release')
            check(not Path(f'/proc/{leaf}').exists(),'leaf fully reaped')
            check(not Path(f'/proc/{p.pid}/task/{p.pid}/children').read_text().strip(),'no guardian children')
            check(all((e.get('lock_device',st.st_dev),e.get('lock_inode',st.st_ino))==(st.st_dev,st.st_ino) for e in ev),'original lock observations')
            failure=json.loads((root/'failure.json').read_text()); check(failure['type']==('PermissionError' if kind=='permission' else 'OSError'),'terminal scan failure')
            # Dispose only this quiescent local reviewer process after independent
            # child-absence checks. No artifact deletion or host cleanup.
            p.kill(); p.wait(timeout=5); log.close()
            scenarios.append({'where':where,'kind':kind,'leaf':leaf,'guardian':p.pid,'lock_device':st.st_dev,'lock_inode':st.st_ino,'failure':failure,'status':'PASS'})
    result={'verdict':'ACCEPT_NARROW_SCAN_REPAIR','mode':mode,'injected_scan_cases':counts,'actual_permission_cases':2,'invalid_root_cases':2,'independent_byte_oracle':oracle,'process_scenarios':scenarios}
    (run/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker': worker(Path(sys.argv[2]),sys.argv[3],sys.argv[4])
    else: main()
