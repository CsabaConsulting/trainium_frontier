#!/usr/bin/env python3
"""Read-only, stdlib authentication of the frozen failed K2 experiment."""
import base64
import csv
import hashlib
import io
import json
import random
import struct
import subprocess
from pathlib import Path

REPO = '/home/poojith-devan/Desktop/Trainium'
RAW = 'c087cbb7'
SOURCE = '9ce2e6a037212aaa76bd85b9b4ab7d311a481757'
PACKET = '710ea6772d5525d97d990ecdf7f835f7f65d8380'
ROOT = 'research/v2/operator/20260926-trn21-k2-symmetric-native/'
LANE = 'research/v2/lanes/k2-symmetric-consumer/'
PROPOSAL = 'research/v2/proposals/20260926-k2-symmetric-native-trn21/'
LABEL = 'fs-20260926-k2-symmetric-consumer-trn21-r1'
PINS = {
    'consumer_nki.py': '2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae',
    'check_native.py': '89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850',
    'reference.py': '14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f',
}
checks = 0


def require(ok, why):
    global checks
    checks += 1
    if not ok:
        raise RuntimeError(why)


def git(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=REPO)


def sha(data):
    return hashlib.sha256(data).hexdigest()


bundle_bytes = git(RAW, ROOT + 'completed/results-bundle.json')
bundle = json.loads(bundle_bytes)
files = {}
for name, record in bundle['files'].items():
    data = base64.b64decode(record['base64'], validate=True)
    require(len(data) == record['bytes'], 'size ' + name)
    require(sha(data) == record['sha256'], 'hash ' + name)
    files[name] = data
require(len(files) == 31 and sum(map(len, files.values())) == bundle['raw_bytes'] == 556014, 'archive size')
require(git(RAW, ROOT + 'launch/first-native-evidence.json') == bundle_bytes, 'launch snapshot equality')


def raw(name):
    return json.loads(files[name])


launch = raw('diagnostic-launch.json')
pre = raw('preflight.json')
for name, pin in PINS.items():
    require(sha(git(SOURCE, LANE + name)) == pin, 'frozen source ' + name)
    require(launch['support_sha256'][name] == pre['staged_hashes'][name] == pin, 'actual pin ' + name)
for name in ('host_driver.py', 'rank_entry.py', 'validate_receipts.py', 'NATIVE_BINDING.disabled.json',
             'EXPERIMENT_PROPOSAL.json', 'MANIFEST.sha256.json', 'bound-command.txt'):
    require(sha(git(PACKET, PROPOSAL + name)) == pre['staged_hashes'][name], 'packet pin ' + name)
command = launch['command']
require(command[:5] == ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=60s', '600s'], 'watchdog')
require(command[6:8] == ['--standalone', '--nproc_per_node=4'], 'four ranks')
require(launch['label'] == pre['label'] == LABEL and pre['host'] == 'trn21', 'identity')
require(launch['no_eval'] and not launch['quality_evidence'], 'diagnostic only')
env = launch['environment']
require(env['NEURON_LOGICAL_NC_CONFIG'] == '2' and env['NEURON_COMPETITION_R1_SEED'] == '42', 'LNC/seed')
require(env['NEURON_CC_FLAGS'] == '--optlevel=1 --auto-cast matmult --auto-cast-type bf16', 'compiler flags')
fresh = raw('fresh-caches.json')
require(len(fresh['directories']) == 6, 'six cache directories')
for key, record in fresh['directories'].items():
    require(record['initially_empty'] and record['path'] == env[key] and LABEL in record['path'], 'fresh env ' + key)
require(env['NEURON_COMPILE_CACHE_URL'] == env['NEURON_COMPILE_CACHE_DIR'] == fresh['compile_cache_url'], 'cache URL')
idle = raw('under-lock-idle.json')
require(idle['lock']['verified_existing_exclusive_lock'], 'exclusive inherited lock')
require((idle['lock']['device'], idle['lock']['inode']) == (66305, 596823), 'original lock')
require(idle['ps']['returncode'] == 0 and idle['ps']['busy'] == [], 'fresh ps')
require(idle['neuron_ls']['returncode'] == 0 and idle['neuron_ls']['process_pids'] == [], 'fresh neuron-ls')

# Independently reconstruct the unique fixture, without executing candidate reference.py.
# Every vocabulary row occurs once: its gradient must equal that token's upstream.
rng = random.Random(42)
values = [rng.randint(-256, 256) / 32 for _ in range(256 * 128)]
expected_bytes = struct.pack('<32768f', *values)
expected_hash = sha(expected_bytes)
zero_hash = sha(bytes(256 * 128 * 4))
require(expected_hash == 'dee95c1da97debf6a901e00260bd60a74e7be1992eca64b832c4f6633cccc671', 'own fixture hash')
require(sum(x != 0 for x in values) == 32709 and max(map(abs, values)) == 8, 'own nonzero/max')
require(zero_hash == 'fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471', 'positive-zero hash')
require(expected_bytes != bytes(len(expected_bytes)), 'negative control zero rejected')
relative_l2 = (sum(x*x for x in values) / sum(x*x for x in values)) ** .5
require(relative_l2 == 1.0, 'zero versus reference relative L2')
# Binary32 nearest-even cancellation in the preceding two cases yields all zeros.
def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]
require(f32(f32(2**24 + 1) - 2**24) == 0, 'large cancellation control')
require(f32(f32(1 + 2**-24) - 1) == 0, 'unit cancellation control')

names = ['cross_tile_large', 'cross_tile_unit', 'unique', 'all_same', 'counts_253_1_1_1',
         'boundary_rows', 'signed_zero', 'bf16_subnormal', 'fp32_subnormal', 'empty']
rank_summary = []
graph_summary = []
for rank in range(4):
    receipt = raw('receipts/rank%d.json' % rank)
    renv = raw('rank-environment-r%d.json' % rank)
    require(receipt['rank'] == renv['rank'] == renv['local_rank'] == rank, 'rank identity')
    require(receipt['world_size'] == renv['world_size'] == 4, 'world size')
    require(renv['environment'] == env and renv['checker_argv'] == receipt['argv'] == command[9:], 'rank argv/env')
    require(receipt['label'] == LABEL and receipt['seed'] == 42, 'rank label seed')
    require(receipt['kernel_sha256'] == PINS['consumer_nki.py'] and receipt['checker_sha256'] == PINS['check_native.py']
            and receipt['reference_sha256'] == PINS['reference.py'], 'rank source pins')
    require(receipt['torch'] == '2.11.0+cpu' and receipt['nki_version'] == '0.5.0+28631259367.ga768afa6', 'installed versions')
    require(receipt['tolerance'] == dict(atol=0, rtol=0, criterion='identical bytes'), 'strict gate')
    cases = receipt['cases']
    require(len(cases) == 13 and [c['name'] for c in cases] == names + names[:3], '13 of 20 exact ordering')
    require([c['dtype'] for c in cases] == ['torch.bfloat16']*10 + ['torch.float32']*3, 'dtype ordering')
    require(all(c['status'] == 'EXACT' and all(x['exact'] for x in c['checks']) for c in cases[:-1]), 'first 12 case records')
    require(cases[-1]['status'] == 'FAIL_EXACTNESS', 'actual failed gate')
    failed = cases[-1]
    require(failed['metadata_group_bounds'] == [1, 1], 'unique metadata bounds')
    by_name = {x['name']: x for x in failed['checks']}
    require(len(by_name) == 8, 'all failing-case measurements present')
    for name in ('upstream_roundtrip', 'direct_consumer', 'direct_repeated_call', 'dense_CPU_oracle', 'eager_autograd_backward'):
        m = by_name[name]
        require(m['exact'] and m['finite'] and m['nonzero'] == 32709, 'nonzero exact ' + name)
        require(m['actual_sha256'] == m['expected_sha256'] == expected_hash and m['max_abs'] == 0, 'correct bytes ' + name)
    bad = by_name['compiled_enclosing_autograd_backward']
    require(not bad['exact'] and bad['finite'] and bad['nonzero'] == 0 and bad['max_abs'] == 8, 'zero gradient failure')
    require(bad['actual_sha256'] == zero_hash and bad['expected_sha256'] == expected_hash, 'zero gradient hashes')
    for mode in ('eager_autograd_forward', 'compiled_enclosing_autograd_forward'):
        require(by_name[mode]['exact'] and by_name[mode]['nonzero'] == 32768, 'forward passes ' + mode)
    bf = {x['name']: x for x in cases[2]['checks']}['compiled_enclosing_autograd_backward']
    require(bf['exact'] and bf['actual_sha256'] == expected_hash and bf['nonzero'] == 32709, 'BF16 same expected gradient passes')
    for case in cases[10:12]:
        backward = {x['name']: x for x in case['checks']}['compiled_enclosing_autograd_backward']
        require(backward['actual_sha256'] == backward['expected_sha256'] == zero_hash, 'earlier FP32 zero expectations')
    require(receipt['status'] == 'REFUSE_OR_UNSUPPORTED' and 'EXACTNESS_FALSIFIED: unique/fp32' in receipt['error'], 'failure terminal')
    require(not receipt['source_adoption'] and not receipt['accepted_speedup'] and receipt['whole_step_speedup'] == 'UNMEASURED', 'scope')
    require(len(receipt['timings']) == 2 and all(t['dtype'] == 'torch.bfloat16' and len(t['seconds']) == 3 for t in receipt['timings']), 'partial component timings only')
    require(raw('rank-wrapper-exit-r%d.json' % rank)['returncode'] == 1, 'wrapper failed')
    rows = list(csv.DictReader(io.StringIO(files['metrics/metric_table_graph_stats_rank%d.csv' % rank].decode())))
    require(len(rows) == 4 and [x['graph_node_count'] for x in rows] == ['9', '7', '8', '8'], 'four graph summaries')
    graph_summary.append({'rank': rank, 'graphs': [{k: row[k] for k in ('graph_id', 'graph_name', 'graph_node_count', 'torch_neuronx_compile_us')} for row in rows]})
    rank_summary.append({'rank': rank, 'receipt_sha256': sha(files['receipts/rank%d.json' % rank]),
                         'cases_recorded': 13, 'expected_cases': 20, 'failed_case': 'unique/fp32',
                         'failed_check': 'compiled_enclosing_autograd_backward', 'zero_sha256': zero_hash,
                         'expected_sha256': expected_hash, 'max_abs_error': 8.0, 'relative_l2_error_derived': relative_l2})

terminal, child = raw('terminal.json'), raw('child-exit.json')
require(terminal['status'] == 'DIAGNOSTIC_CHILD_FAILED' and terminal['returncode'] == child['returncode'] == 1, 'outer failure')
require(terminal['no_automatic_next_job'] and terminal['no_eval'] and not terminal['complete'], 'no successor')
require(child['timeout_wrapped_child_wall_seconds'] == 30.136715991073288 < 600, 'bounded child')
post_bytes = git(RAW, ROOT + 'completed/postflight-explicit-pids.json')
post = json.loads(post_bytes)
observed = {3101403, 3101404, 3101405, 3101422, 3101427, 3101428, 3101429, 3101430}
require(set(post['observed_pids']) == observed and all(not x['exists'] for x in post['pid_rows']), 'eight observed identities gone')
require(post['ps_rc'] == 0 and post['ps_stderr'] == '' and post['related_rows'] == [], 'external process postflight')
require(len(post['all_process_identity_rows']) == 261, 'full census size')
require(not any(x[k] in observed for x in post['all_process_identity_rows'] for k in ('pid', 'ppid', 'pgid', 'sid')), 'recorded PID lineage absent')
require(all(not d['neuron_processes'] for d in post['neuron_ls']) and post['neuron_device_idle'], 'device idle')
require(post['lock_identity'] == dict(device=66305, inode=596823) and post['lock_competitor_rc'] == 0, 'original lock free')
require(post['no_kills_or_changes'], 'read-only postflight')

print(json.dumps(dict(status='AUTHENTICATED_FAILURE_NOT_NATIVE_PASS', checks=checks,
    raw_commit=RAW, source_commit=SOURCE, packet_commit=PACKET, source_pins=PINS,
    bundle_sha256=sha(bundle_bytes), postflight_sha256=sha(post_bytes), files=31, bytes=556014,
    source_arithmetic_imported=False, torch_or_nki_imported=False,
    independent_unique_reference=dict(elements=len(values), output_bytes=len(expected_bytes), nonzero=32709,
        max_abs=8, expected_sha256=expected_hash, actual_zero_sha256=zero_hash, relative_l2_error=relative_l2),
    ranks=rank_summary, graphs=graph_summary, observed_pid_count=8, census_rows=261,
    limitations=['Only hashes/scalars, not full native tensors or compiled graph bodies, were archived.',
      'Input/saved-metadata values at the compiled backward boundary were not observed.',
      'Timeout and short-lived compiler PIDs were not observed; no invented identities.',
      'No native experiment, source change, retry, speed claim or approval performed by this audit.']), indent=2))
