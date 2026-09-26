"""Scoped read-only final archive; reuse the previously executed census body."""
import ast, datetime, hashlib, importlib.util, json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
dest = HERE/'completed'
assert not dest.exists(), 'LOCAL_ARCHIVE_ALREADY_EXISTS'
template = HERE.parent/'20260926-trn22-confirmed36-clean-cold-five-r2/retrieve_final.py'
raw = template.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'd71d9b710b33a5e8f9051d079e3e8a7d13b83676f3de16d32be02ae5cfb1c482'
tree = ast.parse(raw)
body = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'body' for t in n.targets))
old_root = '/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-cold-trn22-r2-packet'
new_root = '/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-clock36-anchored-warm-control-trn21-r1-packet'
assert body.count(old_root) == 1
body = body.replace(old_root, new_root)
body = body.replace('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py', '/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/prepare.py')
body = body.replace('(66305,832589)', '(66305,596823)')
body = body.replace('CT_BINDING.disabled.json', 'ANCHORED_BINDING.disabled.json')
extra = ['fresh-public20m-launch.json','fresh-public20m.log','eval/rehearsal.json','eval/stdout.log','eval/stderr.log','eval-exit-code.txt','checkpoint.json','training-accounting.json','closed-graph-inventory.json','anchor-pre-training.json','anchor-post-training.json','anchor-pre-eval.json','anchor-post-eval.json','anchor-final.json','training-copies.json','eval-copies.json','eval-under-lock-idle.json']
needle = 'identities={};checkpoints={};known=set(P[\'known_prior_identities\']);files={};total=0'
assert body.count(needle) == 1
body = body.replace(needle, 'names += '+repr(extra)+'\n'+needle)
anchor = '/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-arch36a8-startup-clock-trn21-r2-full-packet/train/train.py'
body = body.replace("for name in [root/'package/train.py',prep,prep.parent/'research/scratch/submission_rehearsal.py']:", "for name in [root/'package/train.py',Path("+repr(anchor)+"),prep,prep.parent/'research/scratch/submission_rehearsal.py']:")
body = body.replace('current r2 live/boundary receipts', 'anchored-control live/boundary receipts')
identity_map = {}
paths = [HERE.parent/'20260926-trn21-clock36-anchored-warm-control/trn21.json', HERE.parent/'20260926-trn21-anchored-boundary-0410/trn21.json', HERE.parent/'20260926-trn21-anchored-status-0428/trn21.json', HERE.parent/'20260926-trn21-anchored-boundary-0439/trn21.json', HERE/'trn21.json']
for path in paths:
    receipt = json.loads(path.read_bytes()); remote = receipt.get('remote', receipt)
    for proc in remote.get('processes', []):
        if proc.get('exists') and proc.get('starttime') is not None:
            key = int(proc['pid'])
            if key in identity_map: assert identity_map[key]['starttime'] == proc['starttime'], 'PID_REUSE_IN_OBSERVATIONS'
            identity_map[key] = proc
assert {3177618,3177619,3177620,3177651,3177652,3177666,3177681,3177682,3177683,3177684,3199120,3199121,3199135} <= set(identity_map)
payload = {'known_prior_identities': identity_map}
spec = importlib.util.spec_from_file_location('transport', REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
wire = 'P='+repr(payload)+'\n'+body
assert not m.SENSITIVE.search(wire.encode())
argv, *_ = m.load_transport('trn21'); argv[-1] = 'exec python3 -I -B -'
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
r = subprocess.run(argv, input=wire, text=True, capture_output=True, timeout=60)
assert not m.SENSITIVE.search((r.stdout+r.stderr).encode())
if r.returncode:
    print(json.dumps({'rc': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr})); raise SystemExit(r.returncode)
o = json.loads(r.stdout)
o['local_retrieval'] = {'started_utc': started, 'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'template_sha256': hashlib.sha256(raw).hexdigest(), 'remote_body_sha256': hashlib.sha256(body.encode()).hexdigest(), 'prior_receipts': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
dest.mkdir(exist_ok=False)
for name, value in [('results-bundle.json', o), ('explicit-process-postflight.json', o['postflight']), ('retained-input-identities.json', o['retained_input_identities'])]:
    with (dest/name).open('x') as f: json.dump(value, f, indent=2); f.write('\n')
print(json.dumps({'utc': o['utc'], 'file_count': len(o['files']), 'raw_bytes': o['raw_bytes'], 'checkpoint_after_terminal': o['checkpoint_after_terminal'], 'all_observed_pids_absent': o['postflight']['all_observed_pids_absent'], 'observed_pid_count': len(o['postflight']['observed_pids']), 'related_rows': o['postflight']['related_rows'], 'busy_job_rows': o['postflight']['busy_job_rows'], 'census_count': len(o['postflight']['complete_process_census']), 'lock_competitor_rc': o['postflight']['lock_competitor_rc'], 'device_idle': o['postflight']['neuron_device_idle'], 'fuser_rc': o['postflight']['fuser_rc'], 'disk_free': o['free_bytes'], 'bundle_sha256': hashlib.sha256((dest/'results-bundle.json').read_bytes()).hexdigest()}, indent=2))
