"""Independent packet reconstruction and receipt audit; stdlib only."""
import difflib
import hashlib
import json
import re
from pathlib import Path

ROOT = Path('/packet')
OUT = Path('/work')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

manifest = json.loads((ROOT / 'INPUT_MANIFEST.json').read_text())
input_hashes = []
for path in sorted(ROOT.rglob('*')):
    if path.is_file():
        data = path.read_bytes()
        input_hashes.append(dict(path=str(path.relative_to(ROOT)), bytes=len(data), sha256=sha(data)))
write('input_hashes.json', input_hashes)
actual = {x['path']: x for x in input_hashes}
manifest_checks = {x['path']: actual[x['path']] == x for x in manifest['files']}
assert all(manifest_checks.values()), manifest_checks

lines = (ROOT / 'candidate.diff').read_bytes().splitlines(keepends=True)
assert sum(x.startswith(b'@@ ') for x in lines) == 1
hunk_index = next(i for i, x in enumerate(lines) if x.startswith(b'@@ '))
assert lines[hunk_index].strip() == b'@@ -1,4890 +1,4895 @@'
base, candidate = [], []
for line in lines[hunk_index + 1:]:
    assert line[:1] in (b' ', b'+', b'-'), repr(line)
    if line[:1] in (b' ', b'-'):
        base.append(line[1:])
    if line[:1] in (b' ', b'+'):
        candidate.append(line[1:])
assert len(base) == 4890 and len(candidate) == 4895
source_hashes = {}
for name, source, expected in [
    ('base', base, '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'),
    ('candidate', candidate, 'a489f7f9e935e4fe9ecd2ad9df5f0f327afcbe9475b71e7fbe0f4e71a48925d9'),
]:
    data = b''.join(source)
    source_hashes[name] = dict(bytes=len(data), lines=len(source), sha256=sha(data), matches=sha(data) == expected)
    assert source_hashes[name]['matches']
    (OUT / (name + '.py')).write_bytes(data)
(OUT / 'changes.diff').write_text(''.join(difflib.unified_diff(
    b''.join(base).decode().splitlines(True), b''.join(candidate).decode().splitlines(True),
    fromfile='base.py', tofile='candidate.py', n=4)))
write('source_hashes.json', source_hashes)

arms = {}
for name in ('reference', 'default', 'batch262144'):
    folder = ROOT / 'raw' / name
    g2 = json.loads((folder / 'g2.json').read_text())
    ranks = [json.loads((folder / f'rank{r}.json').read_text()) for r in range(4)]
    log = (folder / 'torchrun.log').read_text()
    fields = ['argv', 'appended_argv', 'steps', 'nproc', 'micro_batch', 'shims', 'python', 'torch',
              'train_py_sha256', 'prepare', 'elapsed_s', 'torchrun_rc', 'timed_out', 'plan_sha256']
    summary = {key: g2[key] for key in fields}
    summary.update(
        embedded_ranks_equal_files=[g2['ranks'][r] == ranks[r] for r in range(4)],
        losses=[r['losses'] for r in ranks],
        rank_fields=[sorted(r) for r in ranks],
        plan_summary=ranks[0]['plan_summary'],
        plan_all_equal=all(r['plan_sha256'] == g2['plan_sha256'] and r['plan_summary'] == ranks[0]['plan_summary'] for r in ranks),
        parameter_count_sum=sum(ranks[0]['plan_summary']['elements_by_owner']),
        n_params=[r['n_params'] for r in ranks], n_changed=[r['n_changed'] for r in ranks],
        unchanged_params=[r['unchanged_params'] for r in ranks],
        cross_rank_mismatches=[r['cross_rank_mismatches'] for r in ranks],
        checkpoint_check=ranks[0]['checkpoint_check'],
        final_digests=ranks[0]['final_digests'],
        tables=[r['tables'] for r in ranks],
        log_evidence=[line for line in log.splitlines() if re.search(r'(num_params=|micro_tokens=|step 0|training_seconds:|compiled:|batch_ramp:|total_tokens_M:|world_size:|lnc:|seq_len:|SUBMISSION_OPTIMIZER_INTEGRATION|Traceback|ERROR)', line)],
        recorded_log_tail_is_actual_tail=log.splitlines()[-len(g2['torchrun_log_tail']):] == g2['torchrun_log_tail'],
    )
    arms[name] = summary

a, b, c = [arms[k] for k in ('reference', 'default', 'batch262144')]
checks = {
    'manifest_matches': manifest_checks,
    'source_hashes': source_hashes,
    'default_loss_exact_numeric_equality_all_ranks': a['losses'] == b['losses'],
    'default_loss_float_hex_equality_all_ranks': [[float(x).hex() for x in row] for row in a['losses']] == [[float(x).hex() for x in row] for row in b['losses']],
    'default_rank0_digest_map_exact_equality': a['final_digests'] == b['final_digests'],
    'default_rank0_digests_compared': len(a['final_digests']),
    'default_checkpoint_fields_equal': a['checkpoint_check'] == b['checkpoint_check'],
    'all_plan_hashes_and_summaries_equal': a['plan_sha256'] == b['plan_sha256'] == c['plan_sha256'] and a['plan_summary'] == b['plan_summary'] == c['plan_summary'],
    'changed_arm_rank0_digests_differ': [k for k,v in b['final_digests'].items() if c['final_digests'].get(k) != v],
    'changed_arm_first_micro_loss_same_each_rank': [b['losses'][r][0] == c['losses'][r][0] for r in range(4)],
    'identity_receipt': json.loads((ROOT / 'raw/identity/default-lockstep.json').read_text()),
    'identity_exit_receipt': json.loads((ROOT / 'raw/identity/default-lockstep-exit.json').read_text()),
    'arms': arms,
}
write('receipt_audit.json', checks)
print(json.dumps({k:v for k,v in checks.items() if k not in ('arms', 'identity_receipt', 'identity_exit_receipt', 'manifest_matches')}, indent=2))
for name, arm in arms.items():
    print(name, json.dumps({k:v for k,v in arm.items() if k not in ('final_digests', 'rank_fields', 'argv', 'appended_argv', 'prepare')}, indent=2))
