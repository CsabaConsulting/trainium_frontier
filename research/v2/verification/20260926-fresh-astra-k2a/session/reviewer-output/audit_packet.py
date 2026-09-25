"""Reviewer-authored, standard-library-only packet reconstruction and receipt audit."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/work')
PACKET = Path('/packet')
def sha(data):
    return hashlib.sha256(data).hexdigest()
def write(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')

manifest = json.loads((PACKET / 'INPUT_MANIFEST.json').read_text())
expected = {e['path']: e for e in manifest['files']}
hashes = []
for path in sorted(PACKET.rglob('*')):
    if path.is_file():
        data = path.read_bytes()
        rel = str(path.relative_to(PACKET))
        record = dict(path=str(path), bytes=len(data), sha256=sha(data))
        if rel in expected:
            record['matches_supplied_manifest'] = (len(data) == expected[rel]['bytes'] and sha(data) == expected[rel]['sha256'])
            assert record['matches_supplied_manifest'], record
        hashes.append(record)

lines = (PACKET / 'candidate.diff').read_bytes().splitlines(keepends=True)
assert lines[4] == b'@@ -1,4890 +1,5014 @@\n'
assert len([x for x in lines if x.startswith(b'@@')]) == 1
base, candidate = [], []
for line in lines[5:]:
    assert line[:1] in (b' ', b'+', b'-'), repr(line)
    if line[:1] in (b' ', b'-'):
        base.append(line[1:])
    if line[:1] in (b' ', b'+'):
        candidate.append(line[1:])
assert len(base) == 4890 and len(candidate) == 5014
sources = {}
for name, data, want in (
    ('base.py', b''.join(base), '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'),
    ('candidate.py', b''.join(candidate), 'e4d93babb4cec8ce53ee1ef9f9d4cc4238c40b93dff0245b66fbcf6bae158b03'),
):
    assert sha(data) == want, (name, sha(data))
    (ROOT / name).write_bytes(data)
    ast.parse(data, filename=name)
    sources[name] = dict(sha256=sha(data), bytes=len(data), lines=len(data.splitlines()), matches_lane_spec=True)
write('input_hashes.json', dict(inputs=hashes, reconstructed_sources=sources, ancestry='supplied metadata only; not independently verified'))
(ROOT / 'changed_context.diff').write_text(''.join(difflib.unified_diff(
    b''.join(base).decode().splitlines(True), b''.join(candidate).decode().splitlines(True),
    fromfile='base.py', tofile='candidate.py', n=7)))

g2 = {p.stem: json.loads(p.read_text()) for p in sorted((PACKET / 'raw/g2').glob('*.json'))}
audit = dict(g2={}, components={})
for name, record in g2.items():
    audit['g2'][name] = {k: v for k, v in record.items() if k not in ('torchrun_log_tail', 'ranks')}
    audit['g2'][name]['ranks'] = [{k: v for k, v in r.items() if k not in ('shims', 'appended_argv')} for r in record['ranks']]
comparison_keys = ['losses', 'final_digests', 'plan_sha256', 'plan_summary', 'n_params', 'n_changed', 'cross_rank_mismatches']
audit['g2_comparisons_to_base'] = {}
for name, record in g2.items():
    audit['g2_comparisons_to_base'][name] = {
        str(r['rank']): {k: dict(equal=r.get(k) == b.get(k), present_in_both=k in r and k in b)
                        for k in comparison_keys}
        for r, b in zip(record['ranks'], g2['tiny-base']['ranks'])
    }
for p in sorted((PACKET / 'raw/component').glob('*.json')):
    record = json.loads(p.read_text())
    out = {k: v for k, v in record.items() if k != 'cases'}
    out['cases'] = []
    for case in record['cases']:
        out['cases'].append(dict(kind=case['kind'], algebra=case['algebra'], paths=case['paths']))
    audit['components'][p.stem] = out
write('receipt_audit.json', audit)
print(json.dumps(dict(reconstructed_sources=sources, input_files=len(hashes),
                     manifest_verified=all(x.get('matches_supplied_manifest', True) for x in hashes)), indent=2))
for name, ranks in audit['g2_comparisons_to_base'].items():
    print('G2', name, json.dumps(ranks, sort_keys=True))
for name, record in audit['components'].items():
    print('COMPONENT', name, json.dumps({k: v for k, v in record.items() if k != 'cases'}, sort_keys=True))
    for case in record['cases']:
        print('CASE', case['kind'], 'algebra_error', case['algebra']['max_abs_error'],
              'paths', [(k, v['max_abs_error'], v['max_allowed']) for k, v in case['paths'].items()])
