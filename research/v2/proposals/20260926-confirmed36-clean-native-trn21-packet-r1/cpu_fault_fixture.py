"""Author-only local process fixture; never loads native or scientific modules."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).absolute().parent


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def ident():
    fields = Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':os.getpid(),'start_ticks':int(fields[19])}


def main():
    root,mode = Path(sys.argv[1]),sys.argv[2]
    if mode=='leaf':
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        (root/('leaf-'+str(os.getpid())+'.json')).write_text(json.dumps(ident()))
        time.sleep(10)
        return 0
    if mode.startswith('child-'):
        for _ in range(2):
            subprocess.Popen([sys.executable,'-B',__file__,str(root),'leaf'],close_fds=True,start_new_session=True)
        while len(list(root.glob('leaf-*.json')))<2: time.sleep(.01)
        (root/'children-ready').touch()
        if mode=='child-helperfail': return 7
        if mode=='child-rankcap':
            with (root/'receipts/.rank0.interrupted.tmp').open('wb') as stream:
                stream.truncate(16*1024**2+1)
        time.sleep(10)
        return 0
    driver = load('cpu_packet_driver',HERE/'packet/host_driver.py')
    lifetime = load('cpu_unchanged_lifetime',HERE/'packet/lifetime_guard.py')
    fd = os.open(root/'device0.lock',os.O_RDWR)
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity = os.fstat(fd)
    guardian = lifetime.Guardian(fd,root/'device0.lock',(identity.st_dev,identity.st_ino),root/'events.jsonl',grace=.1,kill_wait=.3)
    (root/'guardian.json').write_text(json.dumps(ident()))
    driver.ROOT = root
    driver.COMMAND = [sys.executable,'-B',__file__,str(root),'child-'+mode]
    driver.ENV = {'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'}
    driver.CHILD_SECONDS = .35 if mode=='timeout' else 3
    def observation():
        if mode=='auditfail' and (root/'children-ready').exists():
            guardian.receipt.close()
            guardian.event('injected_closed_audit_stream')
        return {'free_disk_bytes':driver.RUNTIME_DISK-1 if mode=='disk' and (root/'children-ready').exists() else driver.INITIAL_DISK,
                'available_ram_bytes':driver.RAM}
    driver.resources = observation
    if mode in ('scanroot','scannested'):
        scan = os.scandir
        target = root if mode=='scanroot' else root/'receipts'
        def denied_scan(path):
            if Path(path)==target and (root/'children-ready').exists():
                raise PermissionError(13,'injected scan '+mode,str(target))
            return scan(path)
        driver.os.scandir = denied_scan
    def postflight():
        if mode in ('scanroot','scannested'):
            driver.artifact_gate(driver.sizes(root))
        (root/'postflight.json').write_text(json.dumps({'cpu_only':True}))
    try:
        rc = driver.run_child(guardian)
        (root/'result.json').write_text(json.dumps({'rc':rc}))
    except BaseException as error:
        (root/'result.json').write_text(json.dumps({'error':str(error)}))
    finally:
        guardian.finish(postflight)
    return 0


if __name__=='__main__': raise SystemExit(main())
