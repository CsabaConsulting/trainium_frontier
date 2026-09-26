"""Explicit manual fallback: reuse pinned remote staging and terminal collection.

Never schedules, retries, changes train.py, kills jobs, or releases another lock.
"""
import ast
import base64
import datetime as dt
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
JOBS = HERE / 'manual-jobs'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_body(path, digest):
    raw = path.read_bytes()
    assert sha(raw) == digest, 'TEMPLATE_DRIFT'
    return next(ast.literal_eval(n.value) for n in ast.parse(raw).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'body' for t in n.targets))


def checked_packet(key):
    manifest = json.loads((JOBS / 'manifest.json').read_bytes())
    entry = manifest[key]
    raw = (JOBS / entry['packet']).read_bytes()
    assert sha(raw) == entry['packet_sha256']
    p = json.loads(raw)
    for name, item in p['files'].items():
        assert sha(base64.b64decode(item['b64'], validate=True)) == item['sha256'], name
    assert p['binding']['label'] == entry['label']
    assert p['binding']['source_sha256'] == entry['source_sha256']
    assert entry['source_sha256'] in (
        '4ec8b1b46562612cdcbfe3d7f66b0c8d25666bdbac3516b30bd9149c2dc08d2d',
        '4b0641db9f009af1601ff48e6944a0c6d851164f42011f026d06ed20761c3020')
    assert p['files']['package/train.py']['sha256'] == entry['source_sha256']
    provenance = json.loads(base64.b64decode(p['files']['package/provenance.json']['b64'], validate=True))
    assert provenance['train_sha256'] == entry['source_sha256'], 'PROVENANCE_SOURCE_MISMATCH'
    assert provenance['label'] == entry['label'], 'PROVENANCE_LABEL_MISMATCH'
    assert provenance['launch_command'] == entry['command'] == p['binding']['launch_command'], 'PROVENANCE_COMMAND_MISMATCH'
    assert p['binding']['budget']['source_seconds'] == 1800
    return entry, p


def connector():
    spec = importlib.util.spec_from_file_location('strict_connector', REPO / 'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remote(host, payload, body, archive):
    module = connector()
    wire = ('P=' + repr(payload) + '\n' + body).encode()
    ast.parse(body)
    assert not module.SENSITIVE.search(wire)
    argv, *_ = module.load_transport(host)
    argv[-1] = 'exec /usr/bin/python3 -I -B -S -'
    result = subprocess.run(argv, input=wire, capture_output=True, timeout=60)
    assert not module.SENSITIVE.search(result.stdout + result.stderr), 'SENSITIVE_OUTPUT_SUPPRESSED'
    receipt = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'returncode': result.returncode,
               'stderr': result.stderr.decode(), 'stdout': result.stdout.decode(), 'payload_sha256': sha(wire)}
    with archive.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    if result.returncode:
        raise RuntimeError(result.stderr.decode() or result.stdout.decode())
    return json.loads(result.stdout)


def stage(key, execute=False, previous=None):
    entry, packet = checked_packet(key)
    queue = json.loads((REPO / 'research/v2/queue.json').read_bytes())['current_low_usage_queue']
    assert key in queue[entry['host']] or key == queue.get('running', {}).get(entry['host']), 'JOB_REMOVED_FROM_CURRENT_QUEUE'
    if entry['order'] == 1:
        # Keep the already released first jobs byte-for-byte on their existing path.
        return subprocess.call([sys.executable, str(HERE / 'stage_warmdown.py'), key,
                                '--execute' if execute else '--local-check'])
    b = packet['binding']
    body = load_body(HERE.parent / '20260926-final-sprint-p0/stage_dispatch.py',
                    'f80abd337b0b18a10e2a59a52144fd1d634e2f5a2421d4da2d1a70632a12d5c3')
    old = "for key in ('NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'):assert Path(b['eval_environment'][key]).is_dir(),'EVAL_CACHE_ABSENT:'+key"
    new = "phase_envs=([b['train_environment']] if not b['fresh_cache_directories'] else [])+[b['eval_environment']]\nfor env in phase_envs:\n for key in ('NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR'):assert Path(env[key]).is_dir(),'EXISTING_PHASE_CACHE_ABSENT:'+key"
    assert body.count(old) == 1
    body = body.replace(old, new).replace("(root/'cold-training-caches').mkdir(mode=0o700)", "if b['fresh_cache_directories']:(root/'cold-training-caches').mkdir(mode=0o700)")
    state = connector().local_state()
    assert not state['local_stops_present'] and not any(v is True for v in state['breaker_tripped'].values())
    payload = {'binding': b, 'files': packet['files'], 'controller': state,
               'old_queue_pids': [2983496] if b['host'] == 'trn21' else [3050509, 3053905]}
    ast.parse(body)
    if not execute:
        print(json.dumps({'job': key, 'status': 'LOCAL_CHECK_PASS_NO_HOST_CALLS', 'source_sha256': b['source_sha256'],
                          'remote_lifetime_guard_unchanged': True, 'cold_cache': bool(b['fresh_cache_directories'])}))
        return 0
    assert previous, 'RETRIEVED_PREDECESSOR_REQUIRED'
    prior_raw = Path(previous).read_bytes()
    prior = json.loads(prior_raw)
    expected_key = (b['host'] + ('-warmdown060' if b['host'] == 'trn21' else '-warmdown085')) if entry['order'] == 2 else (b['host'] + ('-warmdown085' if b['host'] == 'trn21' else '-warmdown060'))
    if entry['order'] == 4:
        expected_key = b['host'] + '-lr0168'
    expected_key = queue.get('predecessors', {}).get(key, entry.get('predecessor', expected_key))
    prior_entry, prior_packet = checked_packet(expected_key)
    assert prior['root'] == prior_packet['binding']['root'], 'WRONG_PREDECESSOR'
    for name, item in prior['files'].items():
        assert sha(item['text'].encode()) == item['sha256'], 'PRIOR_FILE_DRIFT:' + name
    term = json.loads(prior['files']['terminal.json']['text'])
    assert term['label'] == prior_entry['label'] and term['status'] == 'COMPLETE_PENDING_INDEPENDENT_REVIEW' and term['eval_exit_code'] == 0
    post = prior['postflight']
    assert post['all_observed_pids_absent'] and not post['related_rows'] and not post['busy_job_rows']
    assert post['neuron_device_idle'] and post['fuser_rc'] == 1 and post['lock_competitor_rc'] == 0
    assert [post['lock_device'], post['lock_inode']] == b['lock_identity']
    folder = JOBS / key
    attempt = folder / 'attempt.json'
    assert not attempt.exists(), 'CONSUMED_LABEL_CHECK_STATUS_DO_NOT_RETRY'
    with attempt.open('x') as stream:
        json.dump({'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'prior_sha256': sha(prior_raw)}, stream)
    response = remote(b['host'], payload, body, folder / 'dispatch.json')
    print(json.dumps(response['launch']))
    return 0


def collect(key):
    entry, packet = checked_packet(key)
    b = packet['binding']
    body = load_body(HERE / 'proven_terminal_body.py', 'd71d9b710b33a5e8f9051d079e3e8a7d13b83676f3de16d32be02ae5cfb1c482')
    body = body.replace('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-cold-trn22-r2-packet', b['root'])
    body = body.replace('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py', b['prepare_path']).replace('(66305,832589)', repr(tuple(b['lock_identity'])))
    needle = "identities={};checkpoints={};known=set(P['known_prior_identities']);files={};total=0"
    extra = ['lifetime_guard.py', 'lifetime-events.jsonl', 'operator-preflight.json', 'NATIVE_BINDING.json', 'RELEASE.json', 'CPU_RECEIPT.json', 'EXACT_SOURCE_REVIEW.md',
             'fresh-public20m-launch.json', 'fresh-public20m.log', 'eval/rehearsal.json', 'eval/stdout.log', 'eval/stderr.log', 'eval-exit-code.txt',
             'checkpoint.json', 'checkpoint-after-postflight.json', 'postflight-under-lock-idle.json', 'postflight-fuser.json', 'independent-release-idle.json']
    prefix = "names += " + repr(extra) + "\nevents=root/'lifetime-events.jsonl'\nassert events.stat().st_size<=2*1024*1024\nfor line in events.read_text().splitlines():\n event=json.loads(line); i=event.get('identity')\n if i:P['known_prior_identities'][i['pid']]=i\nassert P['known_prior_identities'],'MISSING_GUARDIAN_IDENTITIES'\n"
    assert body.count(needle) == 1
    body = body.replace(needle, prefix + needle)
    folder = JOBS / key
    archive = folder / ('collect-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json')
    result = remote(b['host'], {'known_prior_identities': {}}, body, archive)
    destination = folder / 'results-bundle.json'
    assert not destination.exists(), 'ALREADY_RETRIEVED'
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'job': key, 'bundle': str(destination), 'terminal': json.loads(result['files']['terminal.json']['text']),
                      'idle': result['postflight']['neuron_device_idle'], 'lock_free': result['postflight']['lock_competitor_rc'] == 0}))


def status(host):
    assert host in ('trn21', 'trn22')
    initial = HERE.parent / '20260926-final-sprint-successors' / ({'trn21': 'trn21-batch-recursive-stack', 'trn22': 'trn22-batch-lr02625'}[host] + '-bound-packet.json')
    bindings = [json.loads(initial.read_bytes())['binding']]
    manifest = json.loads((JOBS / 'manifest.json').read_bytes())
    bindings += [checked_packet(k)[1]['binding'] for k, v in manifest.items() if v['host'] == host]
    body = r'''
import datetime,fcntl,json,os,subprocess
from pathlib import Path
b=P['bindings'][0]
ps=subprocess.run(['/usr/bin/ps','-eo','pid,ppid,pgid,sid,comm,args'],capture_output=True,text=True,timeout=10)
assert ps.returncode==0,ps.stderr
terms=('torchrun','train.py','host_driver.py','host_queue.py','submission_rehearsal.py','neuronx-cc','walrus_driver','prepare.py eval-public','run_job.py')
jobs=[line for line in ps.stdout.splitlines() if any(k in line for k in terms)]
nls=subprocess.run(['/opt/aws/neuron/bin/neuron-ls','--json-output'],capture_output=True,text=True,timeout=15)
assert nls.returncode==0,nls.stderr
fd=os.open(b['lock_path'],os.O_RDONLY|os.O_NOFOLLOW);s=os.fstat(fd)
assert [s.st_dev,s.st_ino]==b['lock_identity'],'ORIGINAL_LOCK_CHANGED'
held=True
try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);held=False;fcntl.flock(fd,fcntl.LOCK_UN)
except BlockingIOError:pass
os.close(fd)
roots=[]
for binding in P['bindings']:
 root=Path(binding['root'])
 if not root.is_dir():continue
 files={}
 for name in ('training.log','fresh-public20m.log','terminal.json','outer-flock-exit.json','train/rehearsal.json','eval/rehearsal.json'):
  path=root/name
  if not path.is_file():continue
  with path.open('rb') as f:
   size=os.fstat(f.fileno()).st_size
   if size>8192:f.seek(-8192,2)
   files[name]={'tail_only':size>8192,'text':f.read(8192).decode(errors='replace')}
 roots.append({'label':binding['label'],'root':str(root),'files':files})
v=os.statvfs('/')
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':b['host'],'ps_job_rows':jobs,'neuron_ls':json.loads(nls.stdout),'original_lock_held':held,'roots':roots,'free_bytes':v.f_bavail*v.f_frsize,'read_only':True}))
'''
    archive = HERE / ('manual-status-' + host + '-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json')
    result = remote(host, {'bindings': bindings}, body, archive)
    print(json.dumps(result))


if __name__ == '__main__':
    if sys.flags.optimize:
        raise SystemExit('DO_NOT_DISABLE_SAFETY_ASSERTIONS')
    action, key, *rest = sys.argv[1:]
    if action == 'status':
        assert not rest
        status(key)
        raise SystemExit(0)
    if action == 'check':
        assert not rest
        raise SystemExit(stage(key))
    if action == 'launch':
        assert len(rest) <= 1
        raise SystemExit(stage(key, True, rest[0] if rest else None))
    if action == 'collect':
        assert not rest
        collect(key)
    else:
        raise SystemExit('Use check JOB, launch JOB PREVIOUS_BUNDLE, or collect JOB')
