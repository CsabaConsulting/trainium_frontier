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
        '4b0641db9f009af1601ff48e6944a0c6d851164f42011f026d06ed20761c3020',
        'cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d',
        '949f262c092c141f715bdd71532c42e0bcf502d926a1ee20bc885a2255062daa')
    if entry['source_sha256'].startswith('949f262c'):
        receipt_item = p['files']['INDEPENDENT_REVIEW_RECEIPT.json']
        receipt = json.loads(base64.b64decode(receipt_item['b64'], validate=True))
        assert receipt['source_sha256'] == entry['source_sha256']
        if p['binding'].get('human_timing_only_authority'):
            assert receipt['status'] == 'DEFERRED_BY_HUMAN_RESEARCH_TIMING_ONLY' and receipt['human_timing_only_authority'] is True
            assert p['binding']['no_eval'] is True and p['binding']['steps'] == 201 and p['binding']['submission_authorized'] is False
            assert p['binding']['review_required_before_submission'] is True and p['binding']['independent_review_sha256'] is None
        else:
            assert receipt['status'] == 'PASS_NATIVE_TIMING_SCOPE', 'K2_REVIEW_NOT_PASSED'
            assert receipt['fresh_context'] is True and receipt['reviewer_id'] != '/root', 'K2_REVIEW_NOT_INDEPENDENT'
            assert receipt['independent_cpu_test_exit_code'] == 0, 'K2_INDEPENDENT_CPU_TEST_FAILED'
            assert p['binding']['independent_review_sha256'] == p['files']['EXACT_SOURCE_REVIEW.md']['sha256'], 'K2_REVIEW_BINDING_DRIFT'
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


def timing_predecessor_outcome(key, expected_key, prior, prior_packet):
    """Do not confuse a failed experiment with unsafe ownership of its host.

    Only the explicitly queued K2 -> clean cold fallback may follow an archived
    training-process failure. All original postflight and fresh launch checks
    still execute below. This is not a retry, success verdict or K2 promotion.
    """
    files = prior['files']
    terminal = json.loads(files['terminal.json']['text'])
    if terminal['status'] == 'TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS':
        assert terminal['train_exit_code'] == 0, 'TIMING_PREDECESSOR_FAILED'
        assert json.loads(files['train/rehearsal.json']['text'])['charged_through_process_exit_within_budget'] is True
        return 'TIMING_COMPLETED'
    assert key == 'trn21-finalcold' and expected_key == 'trn21-k2rescue', 'FAILED_PREDECESSOR_NOT_APPROVED_FOR_THIS_JOB'
    binding = prior_packet['binding']
    assert binding['source_sha256'] == '949f262c092c141f715bdd71532c42e0bcf502d926a1ee20bc885a2255062daa'
    assert binding['host'] == 'trn21' and binding['no_eval'] is True and binding['steps'] == 201
    assert terminal['status'] == 'FULL_DRIVER_FAILED', 'UNRECOGNIZED_TIMING_TERMINAL'
    code = int(files['run-exit-code.txt']['text'].strip())
    assert code != 0 and terminal.get('error') == 'TRAIN_EXIT_' + str(code), 'UNPROVEN_TRAINING_PROCESS_FAILURE'
    assert files['package/train.py']['sha256'] == binding['source_sha256'], 'FAILED_PREDECESSOR_SOURCE_DRIFT'
    actual = json.loads(files['NATIVE_BINDING.json']['text'])
    for field in ('label', 'root', 'host', 'source_sha256', 'launch_command', 'lock_identity'):
        assert actual[field] == binding[field], 'FAILED_PREDECESSOR_BINDING_DRIFT:' + field
    assert 'outer-flock-exit.json' in files, 'FAILED_PREDECESSOR_GUARDIAN_NOT_EXITED'
    return 'FAILED_K2_ARCHIVED_COLD_FALLBACK_ONLY'


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
    if entry.get('requires_cold_bundle'):
        cold_path = REPO / entry['requires_cold_bundle']
        cold_raw = cold_path.read_bytes()
        cold_bundle = json.loads(cold_raw)
        for name, item in cold_bundle['files'].items():
            assert sha(item['text'].encode()) == item['sha256'], 'COLD_FILE_DRIFT:'+name
        cold_binding = json.loads(cold_bundle['files']['NATIVE_BINDING.json']['text'])
        cold_terminal = json.loads(cold_bundle['files']['terminal.json']['text'])
        assert cold_binding['source_sha256'] == b['source_sha256'] and cold_binding['fresh_cache_directories'], 'WRONG_FINAL_COLD_SOURCE'
        assert cold_terminal['status'] == 'COMPLETE_PENDING_INDEPENDENT_REVIEW' and cold_terminal['eval_exit_code'] == 0, 'COLD_RUN_NOT_COMPLETE'
        assert json.loads(cold_bundle['files']['train/rehearsal.json']['text'])['charged_through_process_exit_within_budget'] is True
        cold_row = json.loads((cold_path.parent / 'ledger-candidate.json').read_bytes())
        assert cold_row['raw_archive_sha256'] == sha(cold_raw) and cold_row['train_py_sha256'] == b['source_sha256'], 'COLD_AUTHENTICATION_STALE'
        assert cold_row['cold_startup_proven'] is True and cold_row['over_budget'] is False, 'COLD_NOT_QUALIFIED'
        assert cold_row['eval_tokens'] == 20971520 and cold_row['eval_seq_len'] == 1024, 'COLD_EVAL_WRONG_SCOPE'
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
    assert term['label'] == prior_entry['label']
    prior_outcome = 'FULL_EVAL_COMPLETED'
    if prior_packet['binding'].get('no_eval'):
        prior_outcome = timing_predecessor_outcome(key, expected_key, prior, prior_packet)
    else:
        assert term['status'] == 'COMPLETE_PENDING_INDEPENDENT_REVIEW' and term['eval_exit_code'] == 0
    post = prior['postflight']
    assert post['all_observed_pids_absent'] and not post['related_rows'] and not post['busy_job_rows']
    assert post['neuron_device_idle'] and post['fuser_rc'] == 1 and post['lock_competitor_rc'] == 0
    assert [post['lock_device'], post['lock_inode']] == b['lock_identity']
    folder = JOBS / key
    attempt = folder / 'attempt.json'
    assert not attempt.exists(), 'CONSUMED_LABEL_CHECK_STATUS_DO_NOT_RETRY'
    with attempt.open('x') as stream:
        json.dump({'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'prior_sha256': sha(prior_raw),
                   'prior_outcome': prior_outcome}, stream)
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
