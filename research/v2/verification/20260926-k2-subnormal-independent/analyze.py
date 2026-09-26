"""Independent frozen raw receipt analysis; stdlib CPU only, no host access."""
import base64
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

REPO = '/home/poojith-devan/Desktop/Trainium'
COMMIT = '30630c8441249f7db894a80c19200f67003bd18d'
PREFIX = 'research/v2/operator/20260926-trn22-k2-registered/completed/'
PACKET = Path('/tmp/trainium-k2-validator-repair-review-b3ZFC3/research/v2/proposals/20260926-k2-registered-native-trn22')

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields), sort_keys=True, allow_nan=False), flush=True)

def frozen(name):
    return subprocess.check_output(['git', 'cat-file', 'blob', COMMIT + ':' + PREFIX + name], cwd=REPO)

bundle_raw = frozen('results-bundle.json')
require(sha(bundle_raw) == 'a210263a3f0cd67f4896cb76c55ae67198331053220432800639b4483b1d2563', 'bundle SHA')
bundle = json.loads(bundle_raw)
decoded = {}
for name, entry in bundle['files'].items():
    raw = base64.b64decode(entry['base64'], validate=True)
    require(len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], 'raw file identity ' + name)
    decoded[name] = raw
emit('bundle_authenticated', commit=COMMIT, sha256=sha(bundle_raw), files=len(decoded),
     bytes=sum(map(len, decoded.values())), file_hashes={n: sha(r) for n, r in decoded.items()})

# Only scientific packet and raw observations are interpreted. Other bundled material is hashed, not read as prose.
spec = importlib.util.spec_from_file_location('frozen_validator', PACKET / 'validate_receipts.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
require(sha((PACKET / 'validate_receipts.py').read_bytes()) == 'b6b73db9c5da424a9b8d3cb38d1666cea2dc08dae82acec41e3c6467d370afb5', 'validator identity')
pins = {'check_native.py': '0946aee2add168c1fce7bc1f912d1836b0c484f8c6cc401a10d765e9cb8e7663',
        'adapter.py': 'd761a7274e50c21d5930bfa8b984da710ed257be73381fc17ea5f062c03f190a',
        'consumer_nki.py': '2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae',
        'reference.py': '14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f'}
for name, digest in pins.items():
    require(sha((PACKET / name).read_bytes()) == digest, 'source ' + name)

# Independent integer-unit oracle: binary32 least positive subnormal is one 2^-149 unit.
# Reconstruct the fixture directly from its neutral definition, without calling dense/fixture/raw.
units = [0] * 32768
up_bits = []
for token in range(256):
    for channel in range(128):
        sign = 1 if (token + channel) % 3 else -1
        units[(token % 11) * 128 + channel] += sign
        up_bits.append(1 if sign > 0 else 0x80000001)
def unit_bits(value):
    return value if value >= 0 else 0x80000000 | -value
expected_raw = struct.pack('<32768I', *(unit_bits(x) for x in units))
up_raw = struct.pack('<32768I', *up_bits)
expected = dict(shape=[256, 128], dtype='torch.float32', finite=True,
                nonzero=sum(x != 0 for x in units), sha256=sha(expected_raw))
require(expected['nonzero'] == 1408, 'own nonzero')
max_error = max(map(abs, units)) * 2.0 ** -149
require(max_error == 1.2611686178923354e-44, 'own zero-error bound')
require(sha(expected_raw) == '402743e587f15111ef1d3b927df591942db4bd96038726e4edc134a622e44085', 'own exact gradient SHA')
emit('own_integer_subnormal_reference', expected=expected, coefficient_counts=dict(Counter(units)),
     max_error_if_zero=max_error, upstream_sha256=sha(up_raw), zero_sha256=sha(bytes(131072)),
     numeric_elements=32768, imports_torch=False)

all_failures = []
for rank in range(4):
    key = next(n for n in decoded if n.endswith('receipts/rank%d.json' % rank))
    receipt = json.loads(decoded[key])
    require(receipt['rank'] == rank and receipt['world_size'] == 4, 'rank identity')
    for field, name in [('checker_sha256', 'check_native.py'), ('adapter_sha256', 'adapter.py'),
                        ('kernel_sha256', 'consumer_nki.py'), ('reference_sha256', 'reference.py')]:
        require(receipt[field] == pins[name], 'receipt scientific pin')
    require({k: receipt[k] for k in validator.RUNTIME} == validator.RUNTIME, 'native runtime')
    require(receipt['status'] == 'REGISTERED_FULL_CASES_MISMATCH_OR_ERROR' and receipt['returncode'] == 1, 'failed gate preserved')
    require(len(receipt['cases']) == 20 and len(receipt['tangent_replays']) == 2, 'full coverage')
    require([(c['name'], c['dtype']) for c in receipt['cases']] ==
            [(n, 'torch.' + d) for d in ('bfloat16', 'float32') for n in validator.reference.CASE_NAMES], 'case ordering')
    failures, old_statuses, candidate_checks = [], Counter(), 0
    all_cases = [receipt['sentinel']] + receipt['cases'] + receipt['tangent_replays']
    for case in all_cases:
        dtype = 'bf16' if case['dtype'] == 'torch.bfloat16' else 'fp32'
        contract = validator.fixture_contract(case['name'], dtype, case['changed_tangent'])
        for route in case['routes']:
            require(route['inputs_unchanged'] is True, 'mutation')
            validator.inputs(route, contract, dtype)
            for checkname, check in route['checks'].items():
                if check['exact'] is not True:
                    failures.append(dict(case=case['name'], dtype=case['dtype'], route=route['name'], check=checkname,
                                         actual=check.get('actual'), expected=check.get('expected'), max_abs=check.get('max_abs')))
                elif route['name'] != 'compiled_ordinary_embedding':
                    validator.exact(check, [256, 128], 'torch.float32' if checkname == 'gradient' else case['dtype'],
                                    contract['gradient'] if checkname == 'gradient' else None)
                    candidate_checks += 1
            if route['name'] != 'compiled_ordinary_embedding':
                require(route['status'] == 'EXACT', 'candidate status')
        for route in case['original_routes']:
            old_statuses[route['name'] + ':' + route['status']] += 1
        if case['name'] == 'fp32_subnormal' and dtype == 'fp32':
            require(case['checks']['upstream_roundtrip']['actual']['sha256'] == sha(up_raw), 'upstream exact bits')
            require(case['checks']['upstream_roundtrip']['exact'] is True, 'upstream roundtrip exact')
            validator.exact(case['checks']['dense_CPU_oracle'], [256, 128], 'torch.float32', expected)
            for route in case['routes'][:4]:
                require(route['checks']['gradient']['actual'] == expected, 'candidate actual subnormal exact')
            ordinary = case['routes'][4]['checks']['gradient']
            require(ordinary['actual']['sha256'] == sha(bytes(131072)) and ordinary['actual']['nonzero'] == 0,
                    'ordinary actual zero')
            require(ordinary['expected'] == expected and ordinary['max_abs'] == max_error, 'ordinary own oracle')
    require(len(failures) == 1 and failures[0]['case'] == 'fp32_subnormal'
            and failures[0]['route'] == 'compiled_ordinary_embedding' and failures[0]['check'] == 'gradient', 'isolated mismatch')
    all_failures.extend(failures)
    emit('rank_observations', rank=rank, receipt_sha256=sha(decoded[key]), cases=20, sentinel=1, tangent_replays=2,
         candidate_exact_comparisons=candidate_checks, required_mismatches=failures, original_statuses=dict(old_statuses))

post_raw = frozen('postflight-explicit-pids.json')
post = json.loads(post_raw)
require(len(post['observed_pids']) == 11 and all(r['exists'] is False for r in post['pid_rows']), 'observed PIDs absent')
require(post['all_observed_pids_absent'] is True and post['related_rows'] == [] and post['ps_rc'] == 0, 'postflight process evidence')
require(post['neuron_device_idle'] is True and all(d['neuron_processes'] == [] for d in post['neuron_ls']), 'device idle')
require(post['lock_competitor_rc'] == 0 and post['lock_device'] == 66305 and post['lock_inode'] == 832589, 'original lock released')
emit('postflight', sha256=sha(post_raw), utc=post['utc'], observed_pids_absent=11,
     descendants=[], device_idle=True, original_lock_free=True, lock_device=66305, lock_inode=832589)
emit('terminal', record=json.loads(decoded['terminal.json']))
require(not any(n == 'torch' or n.startswith(('torch.', 'nki', 'torch_neuronx')) for n in sys.modules), 'runtime import')
emit('summary', gate='FAILED_PRESERVED', required_control_failures=len(all_failures),
     observed_candidate_failures=0, conclusion='control-route-specific subnormal behavior; precise mechanism not proven',
     native_executed_by_reviewer=False, automatic_retry=False)
