"""Read-only coarse K1 attribution from already-retrieved P runtime traces."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[4]
SOURCE_COMMIT = 'a4aa4351789115a2d0aa947d9ca00ff689f9bd0b'
SOURCE_SHA = '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'
MANIFEST = REPO / 'research/v2/operator/20260926-trn21-profile-recursive6ef7/completed/runtime-trace-retrieval.json'
MANIFEST_SHA = '0ba2cc056c48be8d2392f8ff002e199120b19f1c520ce372ef6b971e458b3694'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def one(items, description):
    require(len(items) == 1, description)
    return items[0]


source = subprocess.check_output(['git', '-C', str(REPO), 'show', SOURCE_COMMIT + ':workflow/proposals/research-v2/train.py'])
require(sha(source) == SOURCE_SHA, 'Exact adopted source')
tree = ast.parse(source)
cls = one([n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OwnedOptimizer'], 'OwnedOptimizer identity')
method = one([n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step'], 'OwnedOptimizer.step identity')
tail = [ast.unparse(n) for n in method.body[-4:]]
require(tail == [
    'if self.owned:\n    torch._foreach_copy_(self.send_views, self.owned)',
    'dist.all_gather_into_tensor(self.received, self.send)',
    'torch._foreach_copy_(self.parameters, self.receive_views)',
    'self.calls += 1',
], 'Exact pack/gather/final-writeback tail, not another optimizer path')
raw_manifest = MANIFEST.read_bytes()
require(sha(raw_manifest) == MANIFEST_SHA, 'Retrieval manifest identity')
manifest = json.loads(raw_manifest)
rows = []
for pin in sorted(manifest['retrieved'], key=lambda p: p['rank']):
    raw = Path(pin['local_path']).read_bytes()
    require(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], pin['rank'] + ' trace identity')
    events = json.loads(raw)['traceEvents']
    del raw
    indexed = list(enumerate(events))
    phase_index, phase = one([(i, e) for i, e in indexed if e.get('cat') == 'user_annotation' and e.get('name') == 'P/optimizer'], 'One captured optimizer phase')
    cpu = [(i, e) for i, e in indexed if e.get('cat') == 'cpu_op' and e.get('tid') == phase['tid']
           and phase['ts'] <= e.get('ts', 0) and e.get('ts', 0) + e.get('dur', 0) <= phase['ts'] + phase['dur']]
    copies = [(i, e) for i, e in cpu if e.get('name') == 'aten::_foreach_copy_']
    require(len(copies) >= 2, 'Pack and final copies exist')
    gather = one([(i, e) for i, e in cpu if e.get('name') == 'c10d::_allgather_base_'], 'One owner gather')
    selected = [('send_pack', copies[-2]), ('gather', gather), ('final_writeback', copies[-1])]
    require(copies[-2][1]['ts'] < gather[1]['ts'] < copies[-1][1]['ts'], 'Source-consistent final host call order')
    record = {'rank': int(pin['rank'][4:]), 'trace_sha256': pin['sha256'], 'optimizer_annotation_index': phase_index, 'operations': {}}
    for label, (cpu_index, event) in selected:
        external = event['args']['External id']
        expected_name = 'allgather' if label == 'gather' else 'aten::_foreach_copy_'
        index, driver = one([(i, e) for i, e in indexed if e.get('cat') == 'privateuse1_driver'
                            and e.get('name') == expected_name and e.get('args', {}).get('External id') == external], 'Unique correlated ' + label)
        require(driver['args']['xu_type'] == 'Compute', 'Runtime Compute label')
        record['operations'][label] = {
            'cpu_index': cpu_index, 'driver_index': index, 'external_id': external,
            'host_ms': event['dur'] / 1000, 'runtime_device_labeled_ms': driver['dur'] / 1000,
            'driver_start_us': driver['ts'], 'driver_end_us': driver['ts'] + driver['dur'],
            'persistent_cache_key': driver['args']['persistent_cache_key'],
            'neff_cache_key_sha256': sha(driver['args']['neff_cache_key'].encode()),
            'nrt_sequence_id': driver['args']['nrt_sequence_id'],
        }
    rows.append(record)
require([r['rank'] for r in rows] == [0, 1, 2, 3], 'All four ranks')
require(len({r['operations']['final_writeback']['neff_cache_key_sha256'] for r in rows}) == 1, 'Same final full-list copy graph on all ranks')
print(json.dumps({
    'source_sha256': SOURCE_SHA, 'manifest_sha256': MANIFEST_SHA, 'profiled_step': 20,
    'role': 'Supervisor/Analyst; local existing-trace inspection, not fresh source review',
    'interpretation': 'Coarse runtime device-labeled intervals, not decoded exclusive engine time or a matched throughput estimate',
    'per_rank': rows,
    'candidate_status': 'REMOTE_ONLY_WRITEBACK_NOT_SELECTED_FOR_IMPLEMENTATION_OR_NATIVE_SLOT',
    'causal_speedup': None, 'native_launch_authorized': False,
    'limits': ['Single profiled step with capture skew', 'Do not sum nested CPU/runtime/driver layers',
               'Full-copy duration is not a measured speedup from omitting only owner rows',
               'No hardware-instruction/embedding/dtype/MFU attribution', 'No production edit or host action'],
}, indent=2))
