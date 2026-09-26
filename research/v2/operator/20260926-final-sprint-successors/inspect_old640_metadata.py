"""One exact two-path metadata-only inspection; no file contents or host writes."""
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

REPO = Path('/home/poojith-devan/Desktop/Trainium')
HERE = Path(__file__).resolve().parent
ASSIGNMENT = REPO / 'research/v2/assignments/20260926-old640-discarded-checkpoints.md'
CONNECTOR = REPO / 'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
CONNECTOR_SHA = '7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899'
OUT = HERE / 'trn22-old640-checkpoint-metadata.json'
TARGETS = [
    {
        'path': '/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260922-ngram-rms-fp32-only-default-shape--al-20260922-ngram-rms-fp32-only-default-shape-smoke-r1/out/al-20260922-ngram-rms-fp32-only-default-shape-smoke-r1/final.pt',
        'expected_sha256': '053d5c2e5974216514a4d6c514c19838cf8c0c9ef71447c791aad7031a27140c',
        'expected_size': 1297950407,
    },
    {
        'path': '/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260922-nki-local-conv-only-default-shape--al-20260922-nki-local-conv-only-default-shape-smoke-r1/out/al-20260922-nki-local-conv-only-default-shape-smoke-r1/final.pt',
        'expected_sha256': 'e0f03915dc9419b82e82358c5230eb73d2da3949eeef98cb367824940d821899',
        'expected_size': 1297950407,
    },
]
BODY = r'''
import datetime,json,os,socket,stat,time
started=time.monotonic()
def meta(s):
    return {'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,'gid':s.st_gid,'size':s.st_size,'allocated_bytes':s.st_blocks*512,'nlink':s.st_nlink,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'regular':stat.S_ISREG(s.st_mode),'directory':stat.S_ISDIR(s.st_mode),'symlink':stat.S_ISLNK(s.st_mode)}
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'read_only':True,'metadata_only':True,'target_contents_read':False,'host_files_written':False,'targets':[]}
if out['hostname']!='ip-172-31-32-253':
    raise RuntimeError('UNEXPECTED_HOST')
for target in TARGETS:
    entry=dict(target);entry['parents']=[]
    fd=None
    try:
        parts=target['path'].split('/')[1:]
        fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        walked=''
        for component in parts[:-1]:
            if time.monotonic()-started>10:raise TimeoutError('METADATA_TIME_BOUND')
            child=os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=child;walked+='/'+component
            entry['parents'].append({'path':walked,**meta(os.fstat(fd))})
        first=os.stat(parts[-1],dir_fd=fd,follow_symlinks=False)
        second=os.stat(parts[-1],dir_fd=fd,follow_symlinks=False)
        entry.update({'exists':True,'before':meta(first),'after':meta(second),'stable':meta(first)==meta(second),'size_matches':first.st_size==target['expected_size'],'hash_checked':False})
    except FileNotFoundError as e:
        entry.update({'exists':False,'hash_checked':False,'error':str(e)})
    except Exception as e:
        entry.update({'status':'REFUSE_METADATA','hash_checked':False,'error_type':type(e).__name__,'error':str(e)})
    finally:
        if fd is not None:os.close(fd)
    out['targets'].append(entry)
v=os.statvfs('/home/ubuntu/workspace')
out['free_bytes']=v.f_bavail*v.f_frsize
out['elapsed_seconds']=time.monotonic()-started
print(json.dumps(out))
'''

if OUT.exists():
    raise RuntimeError('EXISTING_RECEIPT_NO_REPEAT')
assignment = ASSIGNMENT.read_bytes()
for target in TARGETS:
    for value in (target['path'], target['expected_sha256'], str(target['expected_size'])):
        if value.encode() not in assignment:
            raise RuntimeError('ASSIGNMENT_BINDING_MISMATCH')
if hashlib.sha256(CONNECTOR.read_bytes()).hexdigest() != CONNECTOR_SHA:
    raise RuntimeError('CONNECTOR_HASH_MISMATCH')
spec = importlib.util.spec_from_file_location('connector', CONNECTOR)
connector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(connector)
wire = ('TARGETS=' + repr(TARGETS) + '\n' + BODY).encode()
if connector.SENSITIVE.search(wire):
    raise RuntimeError('SENSITIVE_PAYLOAD_REFUSED')
argv, *_ = connector.load_transport('trn22')
argv[-1] = 'exec /usr/bin/python3 -I -B -S -'
try:
    result = subprocess.run(argv, input=wire, capture_output=True, timeout=20)
    stdout, stderr, rc = result.stdout, result.stderr, result.returncode
except subprocess.TimeoutExpired as e:
    stdout, stderr, rc = e.stdout or b'', e.stderr or b'', 'OBSERVATION_TIMEOUT_NOT_JOB_TERMINAL'
if connector.SENSITIVE.search(stdout + stderr):
    raise RuntimeError('SENSITIVE_RESPONSE_REFUSED')
receipt = {'assignment_sha256': hashlib.sha256(assignment).hexdigest(), 'connector_sha256': CONNECTOR_SHA,
           'payload_sha256': hashlib.sha256(wire).hexdigest(), 'returncode': rc,
           'stderr': stderr.decode(), 'remote': json.loads(stdout) if rc == 0 else None,
           'stdout_if_error': stdout.decode() if rc != 0 else None}
with OUT.open('x') as f:
    json.dump(receipt, f, indent=2)
    f.write('\n')
remote = receipt['remote'] or {}
print(json.dumps({'returncode': rc, 'utc': remote.get('utc'), 'free_bytes': remote.get('free_bytes'),
                  'targets': [{k: t.get(k) for k in ('path','exists','before','stable','size_matches','error','hash_checked')} for t in remote.get('targets',[])],
                  'receipt_sha256': hashlib.sha256(OUT.read_bytes()).hexdigest()}))
