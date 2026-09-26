"""One user-requested read-only storage identity check per host; no mutation."""
import concurrent.futures
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
connector=REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
assert hashlib.sha256(connector.read_bytes()).hexdigest()=='7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899'
spec=importlib.util.spec_from_file_location('connector',connector)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
BODY=r'''
import datetime,json,socket,subprocess,time
start=time.monotonic()
expected={'trn21':'ip-172-31-34-85','trn22':'ip-172-31-32-253'}
assert socket.gethostname()==expected[HOST],'HOSTNAME'
commands=[['/usr/bin/findmnt','-J','-b','-o','SOURCE,FSTYPE,SIZE,USED,AVAIL,TARGET','/'],['/usr/bin/lsblk','-b','-J','-o','NAME,SIZE,FSTYPE,MOUNTPOINTS,SERIAL'],['/usr/bin/df','-hT','/']]
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':HOST,'hostname':socket.gethostname(),'read_only':True,'job_processes_not_polled':True,'commands':[]}
for cmd in commands:
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=6,env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C'})
        assert len(r.stdout.encode())+len(r.stderr.encode())<=32768,'OUTPUT_BOUND'
        out['commands'].append({'argv':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    except subprocess.TimeoutExpired as e:
        out['commands'].append({'argv':cmd,'returncode':'READ_ONLY_COMMAND_TIMEOUT','stdout':(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else (e.stdout or ''),'stderr':(e.stderr or b'').decode() if isinstance(e.stderr,bytes) else (e.stderr or '')})
out['elapsed_seconds']=time.monotonic()-start
print(json.dumps(out))
'''
def one(host):
    out=HERE/(host+'-storage-identity-0834.json');assert not out.exists()
    wire=('HOST='+repr(host)+'\n'+BODY).encode();assert not m.SENSITIVE.search(wire)
    argv,*_=m.load_transport(host);argv[-1]='exec /usr/bin/python3 -I -B -S -'
    r=subprocess.run(argv,input=wire,capture_output=True,timeout=30)
    assert not m.SENSITIVE.search(r.stdout+r.stderr)
    receipt={'returncode':r.returncode,'stderr':r.stderr.decode(),'payload_sha256':hashlib.sha256(wire).hexdigest(),'remote':json.loads(r.stdout) if r.returncode==0 else None,'stdout_if_error':r.stdout.decode() if r.returncode else None}
    with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({'host':host,'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),**receipt}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(one,('trn21','trn22')))
