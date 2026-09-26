"""One read-only metadata-only refresh of exactly the five selected public train shards."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
assert len(sys.argv) == 2 and sys.argv[1] in ('trn21', 'trn22')
host = sys.argv[1]
receipt = json.loads((HERE / (host + '-full-recovery1105.json')).read_bytes())
assert receipt['returncode'] == 0 and receipt['remote']['host'] == host
observed = receipt['remote']
launch = json.loads(observed['files']['training-launch.json']['text'])
expected_cache = launch['environment']['NEURON_COMPETITION_R1_CACHE_DIR']
assert expected_cache == '/home/ubuntu/.cache/neuron_competition_r1'
workers = [p['identity_before'] for p in observed['processes'] if p.get('environment_whitelist', {}).get('RANK') is not None]
assert len(workers) == 4
packet_name = {'trn21': 'trn21-batch-recursive-stack', 'trn22': 'trn22-batch-lr02625'}[host]
b = json.loads((HERE.parent / '20260926-final-sprint-successors' / (packet_name + '-bound-packet.json')).read_bytes())['binding']
body = r'''
import datetime,json,os,socket,stat
from pathlib import Path
assert socket.gethostname()==P['hostname']
def ident(pid):
 raw=Path('/proc/'+str(pid)+'/stat').read_text();parts=raw[raw.rfind(')')+2:].split()
 return {'pid':pid,'start_ticks':int(parts[19])}
actual=[]
for expected in P['workers']:
 before=ident(expected['pid']);assert before['start_ticks']==expected['start_ticks'],'WORKER_IDENTITY_CHANGED'
 cache_values=[]
 for item in Path('/proc/'+str(expected['pid'])+'/environ').read_bytes().split(b'\0'):
  key,sep,value=item.partition(b'=')
  if sep and key==b'NEURON_COMPETITION_R1_CACHE_DIR':cache_values.append(value.decode())
 assert cache_values==[P['expected_cache']],'EFFECTIVE_DATA_CACHE_BINDING'
 assert ident(expected['pid'])==before,'WORKER_IDENTITY_RACE'
 actual.append(dict(before,effective_cache_dir=cache_values[0]))
data=Path(P['expected_cache'])/'data';rows=[]
for index in range(5):
 name=f'shard_{index:05d}.parquet';path=data/name
 try:
  s=os.lstat(path)
  rows.append({'name':name,'path':str(path),'exists':True,'regular_file':stat.S_ISREG(s.st_mode),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'device':s.st_dev,'inode':s.st_ino})
 except FileNotFoundError:rows.append({'name':name,'path':str(path),'exists':False})
print(json.dumps({'schema':'five-train-shards-stat-only-v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':P['host'],'label':P['label'],'effective_data_dir':str(data),'workers':actual,'files':rows,'expected_selected_count':5,'present_regular_count':sum(x.get('exists',False) and x.get('regular_file',False) for x in rows),'total_present_bytes':sum(x.get('bytes',0) for x in rows),'shard_contents_read':False,'shard_hashing_performed':False,'validation_files_accessed':False,'setup_or_download':False,'read_only':True,'prepare_sha256_from_authenticated_training_launch':P['prepare_sha256'],'prepare_sha256_scope':'Previously captured training-launch identity, not freshly hashed by this metadata-only call'}))
'''
payload = {'host': host, 'hostname': b['hostname'], 'label': b['label'], 'workers': workers,
           'expected_cache': expected_cache, 'prepare_sha256': launch['prepare_sha256']}
assert payload['prepare_sha256'] == b['prepare_sha256'] == '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb'
connector = REPO / 'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py'
assert hashlib.sha256(connector.read_bytes()).hexdigest() == '7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899'
spec = importlib.util.spec_from_file_location('fixed_connector', connector)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
wire = ('P=' + repr(payload) + '\n' + body).encode()
assert not m.SENSITIVE.search(wire)
out = HERE / (host + '-five-train-shards-stat.json')
assert not out.exists()
argv, *_ = m.load_transport(host)
argv[-1] = 'exec /usr/bin/python3 -I -B -S -'
r = subprocess.run(argv, input=wire, capture_output=True, timeout=20)
assert not m.SENSITIVE.search(r.stdout + r.stderr)
result = {'returncode': r.returncode, 'stderr': r.stderr.decode(),
          'payload_sha256': hashlib.sha256(wire).hexdigest(),
          'remote': json.loads(r.stdout) if r.returncode == 0 else None,
          'stdout_if_error': r.stdout.decode() if r.returncode else None}
with out.open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
remote = result['remote'] or {}
print(json.dumps({'host': host, 'returncode': r.returncode, 'error': result['stderr'],
                  'utc': remote.get('utc'), 'present_regular_count': remote.get('present_regular_count'),
                  'total_present_bytes': remote.get('total_present_bytes'),
                  'effective_data_dir': remote.get('effective_data_dir'),
                  'receipt_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))
