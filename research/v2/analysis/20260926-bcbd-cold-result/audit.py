"""Authenticate closed cold-five receipts; no native execution or quality verdict."""
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / 'research/v2/operator/20260926-trn22-basebcbd-cold-five'
DIR = ROOT / 'completed'
checks = []

def require(ok, name):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)

def read(path):
    return json.loads(path.read_text())

manifest = read(DIR / 'retrieval-manifest.json')['metadata']
for name, expected in manifest['files'].items():
    raw = (DIR / name).read_bytes()
    require(len(raw) == expected['bytes'], name + ' length')
    require(hashlib.sha256(raw).hexdigest() == expected['sha256'], name + ' sha256')
result = read(DIR / 'train/rehearsal.json')
require(result['mode'] == 'train' and result['returncode'] == 0, 'native diagnostic exit0')
require(result['step_boundaries'] == 5, 'five boundaries only')
require([r['step'] for r in result['observations']] == [0,1,2,3,4], 'ordered exact observation scope')
startup = result['observations'][4]['observed_seconds']
require(startup == result['startup_allowance_observed'], 'uncapped fifth-boundary observation')
require(startup <= 600, 'base600 target met')
charge = result['wall_seconds'] - min(startup, 900)
require(math.isclose(charge, result['charged_through_process_exit'], abs_tol=1e-9, rel_tol=0), 'through-exit official charge recomputed')
require(0 <= charge < 1800 and result['charged_through_process_exit_within_budget'] is True, 'charge within budget')
require(len(result['fresh_empty_sdk_caches']) == 6, 'all six fresh private roots recorded by pinned harness')
require(len(set(result['fresh_empty_sdk_caches'].values())) == 6, 'six distinct roots')
require(all('/fs-20260926-ct-basebcbd-cold-trn22-r1-packet/train/caches/' in p for p in result['fresh_empty_sdk_caches'].values()), 'roots bound to unique label')
log = (DIR / 'training.log').read_text()
isolated = json.loads(next(l.split(' ',1)[1] for l in log.splitlines() if l.startswith('ISOLATED_SUBMISSION ')))
require(isolated['train_sha256'] == 'bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522', 'exact bcbd source')
require(isolated['prepare_sha256'] == '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb', 'unchanged organizer prepare')
require(isolated['fresh_empty_sdk_caches'] == result['fresh_empty_sdk_caches'], 'launch and final coldness observations agree')
for flag in ('--num-steps 5','--max-train-seconds 1800','--initial-cache-refresh','--recursive-schedule'):
    require(flag in isolated['launch_command'], 'command '+flag)
require(result['checkpoint_bytes'] == 469834849 and result['checkpoint_sha256'] == '721d272268dec5d5472c0b0246e10626ad49da7a29686445cd754caa6efd7f63', 'retained diagnostic checkpoint identity')
guard = read(DIR / 'under-lock-idle.json')
require(guard['status'] == 'PASS' and guard['lock']['verified_existing_exclusive_lock'] is True, 'inherited lifetime lock verified')
require(guard['lock']['inode'] == 832589 and not guard['ps']['busy'] and not guard['neuron_ls']['process_pids'], 'same original inode and both idle checks')
require(read(DIR / 'outer-flock-exit.json')['returncode'] == 0, 'outer process terminal zero')
require((DIR / 'run-exit-code.txt').read_text().strip() == '0', 'native child terminal zero')
print(json.dumps({'status':'COLD_FIVE_BASE_STARTUP_TARGET_PASS_ONLY','checks':len(checks),
 'label':'fs-20260926-ct-basebcbd-cold-trn22-r1','startup_seconds':startup,
 'historical_6ef7_cold_seconds':855.0719942810247,
 'startup_seconds_saved':855.0719942810247-startup,
 'wall_seconds':result['wall_seconds'],'charged_through_exit_seconds':charge,'steps':5,
 'cold_proof':'Pinned rehearsal six absent/empty private caches plus separately archived four-worker environments',
 'steady_throughput_proven':False,'public_bpb':None,'adopted':False,'submission_eligible':False,
 'limits':'Only startup diagnostic; no fixed-budget quality run, no claim of candidate-checkpoint native numerical equivalence.'}, indent=2))
