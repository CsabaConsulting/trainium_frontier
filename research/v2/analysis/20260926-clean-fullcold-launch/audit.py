"""Authenticate archived live launch only; not a completed cold or quality pass."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / 'research/v2/operator/20260926-trn22-confirmed36-clean-cold-full-r2/trn22.json'
COUNT = 0

def check(value, message):
    global COUNT
    if not value:
        raise RuntimeError(message)
    COUNT += 1

raw = P.read_bytes()
check(hashlib.sha256(raw).hexdigest() == '3b399717e2896f2880592b41acc6d557c4cee061db1e947f406f5c2d0e6f1897', 'frozen snapshot')
x = json.loads(raw)
check(x['rc'] == 0 and not x['stderr'], 'collector success')
r = x['remote']
files = r['files']
check(len(files) == 7, 'seven captured files')
for name, record in files.items():
    value = record['text'].encode()
    check(record['capture'] == 'complete_at_observation' and len(value) == record['captured_bytes'] == record['bytes_at_stat'] and hashlib.sha256(value).hexdigest() == record['sha256_captured'], name)
source = 'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43'
guardian = 'e461ec5c76c784fd51471c9e8c268cdde42fe1def68b31bfc9524480f980c312'
check(len(r['completed_parent_metadata_only']) == 3, 'actual source metadata count')
for name, metadata in r['completed_parent_metadata_only'].items():
    check(metadata['stable_during_read'] and metadata['sha256'] == (guardian if name.endswith('lifetime_guard.py') else source), name)
launch = json.loads(files['training-launch.json']['text'])
check(launch['source_sha256'] == source and launch['harness_sha256'] == '7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b' and launch['prepare_sha256'] == '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb', 'protected launch pins')
env = launch['environment']
check(env['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'] == '1800' and env['NEURON_COMPETITION_R1_SEED'] == '42' and env['NEURON_LOGICAL_NC_CONFIG'] == '2' and '--fresh-caches' in launch['command'], 'budget seed shape cold request')
check((r['lock']['device'], r['lock']['inode'], r['lock']['held']) == (66305,832589,True), 'original held lock')
idle = json.loads(files['under-lock-idle.json']['text'])
check(idle['status'] == 'PASS' and not idle['ps']['busy'] and not idle['neuron_ls']['process_pids'] and idle['lock']['verified_existing_exclusive_lock'], 'locked preflight idle')
ps = {p['pid']:p for p in r['processes']}
parents = {3281908:3281907,3281909:3281908,3281925:3281909,3281926:3281925,3281940:3281926}
for pid, parent in parents.items():
    check(ps[pid]['exists'] and ps[pid]['identity_stable'] and ps[pid]['ppid'] == parent, 'actual parent ' + str(pid))
expected = ['--n-embd','1024','--no-ngram-ve','--no-qk-shift','--compile-sdpa-direct','--nki-local-conv','--bf16-norm-output','--pack-factor','32','--clip-after-reduce','--seq-len','1024','--no-eval-public']
cache_fields = {'NEURON_COMPILE_CACHE_URL':'compiler','TORCH_NEURONX_NEFF_CACHE_DIR':'neff','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR':'local','TORCH_NEURONX_HLO_CACHE_DIR':'hlo','TORCHINDUCTOR_CACHE_DIR':'inductor','NEURON_COMPILE_CACHE_DIR':'compiler'}
for rank, pid in enumerate(range(3281955,3281959)):
    p = ps[pid]
    e = p['environment_whitelist']
    check(p['exists'] and p['identity_stable'] and p['ppid'] == 3281940 and p['pgid'] == p['sid'] == pid, 'worker identity ' + str(rank))
    check(e['RANK'] == e['LOCAL_RANK'] == str(rank) and e['WORLD_SIZE'] == '4', 'worker rank ' + str(rank))
    check(p['argv'][2] == 'train.py' and p['argv'][3:] == expected, 'worker argv ' + str(rank))
    check(all(e[k] == env[k] for k in ('NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS','NEURON_COMPETITION_R1_SEED','NEURON_LOGICAL_NC_CONFIG')), 'worker limits ' + str(rank))
    check(all(e[k] == r['root'] + '/train/caches/' + suffix for k,suffix in cache_fields.items()), 'actual private caches ' + str(rank))
events = [json.loads(line) for line in files['lifetime-events.jsonl']['text'].splitlines()]
check(events[0]['event'] == 'guardian_started' and events[0]['subreaper'] is True and events[0]['lock_held'] is True and events[0]['identity']['pid'] == 3281909, 'actual guardian')
check(any(e['event'] == 'phase_spawned' and e['pid'] == 3281925 for e in events), 'actual owned training phase')
check(r['free_bytes'] >= 13.5 * 1024**3, 'initial live disk reserve')
print(json.dumps({'verdict':'ACCEPT_LIVE_LAUNCH_ONLY','checks_passed':COUNT,'observed_utc':r['utc'],'workers':[3281955,3281956,3281957,3281958],'source_sha256':source,'startup_pass':False,'quality_pass':False,'submission_ready':False}, indent=2))
