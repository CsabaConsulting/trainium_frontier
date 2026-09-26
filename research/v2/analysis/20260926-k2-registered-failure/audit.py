"""Authenticate one failed diagnostic; never waive its gate or launch work."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COMMIT = '30630c8441249f7db894a80c19200f67003bd18d'
PREFIX = 'research/v2/operator/20260926-trn22-k2-registered/completed/'
PIN = 'a210263a3f0cd67f4896cb76c55ae67198331053220432800639b4483b1d2563'
LABEL = 'fs-20260926-k2-registered-forward-trn22-r1'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def git(path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', COMMIT + ':' + PREFIX + path])

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

raw = git('results-bundle.json')
require(sha(raw) == PIN, 'bundle pin')
bundle = json.loads(raw)
files = {}
for name, record in bundle['files'].items():
    data = base64.b64decode(record['base64'], validate=True) if 'base64' in record else record['text'].encode()
    require(len(data) == record['bytes'] and sha(data) == record['sha256'], 'artifact integrity: ' + name)
    files[name] = data
require(len(files) == 31 and sum(map(len, files.values())) == 7682695, 'complete frozen bundle')
terminal = json.loads(files['terminal.json'])
child = json.loads(files['child-exit.json'])
require(terminal['label'] == LABEL and terminal['status'] == 'DIAGNOSTIC_CHILD_FAILED', 'failed terminal identity')
require(terminal['returncode'] == child['returncode'] == 1 and terminal['complete'] is False, 'failure retained')
require(terminal['no_automatic_next_job'] and terminal['no_eval'], 'bounded diagnostic scope')
require(b'ChildFailedError' in files['diagnostic.log'], 'actual error traceback')
failures = []
candidate_cases = 0
for rank in range(4):
    receipt = json.loads(files[f'receipts/rank{rank}.json'])
    require(receipt['rank'] == rank and receipt['world_size'] == 4 and receipt['label'] == LABEL, 'rank identity')
    require(receipt['status'] == 'REGISTERED_FULL_CASES_MISMATCH_OR_ERROR' and receipt['returncode'] == 1, 'rank failure')
    require(receipt['native_qualification'] is False and receipt['source_adoption'] is False, 'no qualification')
    require(receipt['checker_sha256'] == '0946aee2add168c1fce7bc1f912d1836b0c484f8c6cc401a10d765e9cb8e7663', 'checker pin')
    require(receipt['adapter_sha256'] == 'd761a7274e50c21d5930bfa8b984da710ed257be73381fc17ea5f062c03f190a', 'adapter pin')
    require(receipt['kernel_sha256'] == '2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae', 'kernel pin')
    require(receipt['reference_sha256'] == '14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f', 'reference pin')
    require(len(receipt['cases']) == 20 and len(receipt['tangent_replays']) == 2, 'complete case counts')
    bad = []
    for case in receipt['cases'] + receipt['tangent_replays']:
        routes = {r['name']: r for r in case['routes']}
        names = ('direct_consumer', 'direct_repeated_call') if case['name'] == 'empty' else ('direct_consumer', 'direct_repeated_call', 'registered_eager', 'registered_compiled')
        expected_routes = set(names) if case['name'] == 'empty' else set(names) | {'compiled_ordinary_embedding'}
        require(set(routes) == expected_routes and len(case['routes']) == len(routes), 'case-specific distinct required routes')
        for name in names:
            route = routes[name]
            require(route['status'] == 'EXACT' and route['inputs_unchanged'], 'candidate route not exact')
            require(all(c['exact'] for c in route['checks'].values()), 'candidate exactness checks')
            candidate_cases += 1
        control = routes.get('compiled_ordinary_embedding')
        if control is not None and control['status'] != 'EXACT':
            bad.append((case, control))
    require(len(bad) == 1, 'one failed required control per rank')
    case, route = bad[0]
    require(case['name'] == 'fp32_subnormal' and case['dtype'] == 'torch.float32', 'specific failed case')
    gradient = route['checks']['gradient']
    require(route['inputs_unchanged'] and route['checks']['output']['exact'], 'control inputs/output')
    require(gradient['actual']['finite'] and gradient['expected']['finite'], 'finite control gradients')
    require(gradient['actual']['nonzero'] == 0 and gradient['expected']['nonzero'] == 1408, 'control denormal discrepancy')
    require(gradient['max_abs'] == 1.2611686178923354e-44, 'exact observed error')
    failures.append(dict(rank=rank, case=case['name'], route=route['name'], **gradient))
post_raw = git('postflight-explicit-pids.json')
post = json.loads(post_raw)
known = {3244018,3244019,3244020,3244034,3244035,3244039,3244040,3244041,3244042,3244718,3244724}
require(set(post['observed_pids']) == known and post['all_observed_pids_absent'], 'all11 observed identities')
require({x['pid'] for x in post['pid_rows']} == known and all(not x['exists'] for x in post['pid_rows']), 'explicit PID absence')
require(not (known & {x['pid'] for x in post['all_process_identity_rows']}), 'complete process census absence')
require(post['ps_rc'] == 0 and not post['ps_stderr'] and not post['related_rows'], 'successful quiet census')
require(post['neuron_device_idle'] and not post['neuron_ls_stderr'], 'idle device observation')
require(post['lock_device'] == 66305 and post['lock_inode'] == 832589 and post['lock_competitor_rc'] == 0, 'original lock free')
print(json.dumps(dict(status='AUTHENTICATED_FAILED_DIAGNOSTIC_NOT_QUALIFIED', raw_commit=COMMIT,
    bundle_sha256=PIN, files=len(files), bytes=sum(map(len, files.values())), label=LABEL,
    terminal=terminal, child=child, candidate_route_case_observations_exact=candidate_cases,
    required_control_failures=failures, postflight_utc=post['utc'], postflight_sha256=sha(post_raw),
    native_qualification=False, automatic_retry=False, source_adopted=False,
    speedup=None, public_20m=None, old_failures_preserved=True), indent=2, allow_nan=False))
