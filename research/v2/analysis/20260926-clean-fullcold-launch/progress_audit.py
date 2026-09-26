"""Read-only audit of first-five progress, not final startup/budget/quality."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / 'research/v2/operator/20260926-trn22-confirmed36-clean-cold-full-r2'
old = json.loads((P / 'trn22.json').read_text())['remote']
raw = (P / 'progress/boundary-0509.json').read_bytes()
COUNT = 0

def check(value, name):
    global COUNT
    if not value:
        raise RuntimeError(name)
    COUNT += 1

check(hashlib.sha256(raw).hexdigest() == 'd6484bba43717e32726cecf80f566a08c5466925e4d2408c1b1f8265208e9194', 'snapshot hash')
x = json.loads(raw)
check(x['rc'] == 0 and not x['stderr'], 'collector')
r = x['remote']
check(len(r['files']) == 8, 'eight complete captures')
for n,f in r['files'].items():
    b = f['text'].encode()
    check(f['capture'] == 'complete_at_observation' and len(b) == f['captured_bytes'] == f['bytes_at_stat'] and hashlib.sha256(b).hexdigest() == f['sha256_captured'], n)
check(r['completed_parent_metadata_only'] == {}, 'boundary intentionally has no repeated source hash observation')
check(x['prior_snapshot_sha256'] == '3b399717e2896f2880592b41acc6d557c4cee061db1e947f406f5c2d0e6f1897', 'authenticated launch evidence reference')
check((r['lock']['device'],r['lock']['inode'],r['lock']['held']) == (66305,832589,True), 'original held lock')
prior = {p['pid']:p for p in old['processes']}
ps = {p['pid']:p for p in r['processes']}
for pid in (3281907,3281908,3281909,3281925,3281926,3281940,3281955,3281956,3281957,3281958):
    p = ps[pid]
    check(p['exists'] and p['identity_stable'], 'live ' + str(pid))
    check(all(p[k] == prior[pid][k] for k in ('starttime','ppid','pgid','sid','argv','cwd','environment_whitelist')), 'unchanged identity/env ' + str(pid))
markers = [int(s) for s in r['files']['train/steps.jsonl']['text'].splitlines()]
check(markers == list(range(39)), '39 sequential completed step markers')
matches = re.findall(r'^step (\d+) \| loss ([0-9.]+).*?tok/s ([0-9,]+)',r['files']['training.log']['text'],re.M)
check([int(m[0]) for m in matches[:5]] == list(range(5)), 'first five log steps')
check([float(m[1]) for m in matches[:5]] == [9.0101,8.7779,8.4047,8.0711,7.7681], 'same finite first five losses as cold-five')
check(all(math.isfinite(float(m[1])) for m in matches), 'finite printed losses')
launch = json.loads(r['files']['training-launch.json']['text'])
upper = (datetime.fromisoformat(r['utc']) - datetime.fromisoformat(launch['utc'])).total_seconds()
check(0 < upper < 900, 'phase-to-observation UTC upper bound below cap')
check(r['free_bytes'] > 13.5*1024**3, 'disk reserve')
print(json.dumps({'verdict':'ACCEPT_LIVE_FIRST_FIVE_ONLY','checks_passed':COUNT,'observed_utc':r['utc'],'completed_markers':len(markers),'latest_printed_step':int(matches[-1][0]),'latest_printed_toks_per_second':int(matches[-1][2].replace(',','')),'phase_to_observation_seconds_upper_bound':upper,'source_rehashed_at_this_boundary':False,'exact_uncapped_fifth_seconds':None,'charged_through_exit_seconds':None,'public20m':None,'submission_ready':False},indent=2))
