"""Inspect raw supplied fields without inheriting their PASS conclusions."""
from collections import Counter
import hashlib
import json
from pathlib import Path

packet = Path('/packet')
receipts = {p.name: json.loads(p.read_text()) for p in sorted((packet / 'raw/g2').glob('*.json'))}
audit = {'receipts': {}, 'off_on_comparison': []}
for name, value in receipts.items():
    audit['receipts'][name] = {
        'claimed_verdict': value['verdict'], 'reason': value['reason'],
        'torchrun_rc': value['torchrun_rc'], 'timed_out': value['timed_out'],
        'source_sha256': value['train_py_sha256'],
        'argv': value['argv'], 'appended_argv': value['appended_argv'],
        'prepare': value['prepare'], 'shims': value['shims'],
        'steps': value['steps'], 'micro_batch': value['micro_batch'],
        'python': value['python'], 'torch': value['torch'],
        'ranks': [{
            'rank': r['rank'], 'claimed_status': r['status'],
            'error': r.get('error'), 'losses': r.get('losses'),
            'n_changed': r.get('n_changed'), 'n_params': r.get('n_params'),
            'cross_rank_mismatches': r.get('cross_rank_mismatches'),
            'checkpoint_check': r.get('checkpoint_check'),
            'final_digests_count': len(r.get('final_digests', {})),
            'final_digest_lengths': sorted({len(d) for d in r.get('final_digests', {}).values()}),
            'plan_sha256': r.get('plan_sha256'),
            'profile': None if r.get('p_cpu_profile') is None else {
                **{k: v for k, v in r['p_cpu_profile'].items() if k != 'events'},
                'event_counts': dict(Counter(r['p_cpu_profile']['events']))}
        } for r in value['ranks']]
    }
off = receipts['tiny-off-r3.json']
on = receipts['tiny-on.json']
assert [r['rank'] for r in off['ranks']] == [0, 1, 2, 3]
assert [r['rank'] for r in on['ranks']] == [0, 1, 2, 3]
for a, b in zip(off['ranks'], on['ranks']):
    audit['off_on_comparison'].append({
        'rank': a['rank'], 'losses_exact': a['losses'] == b['losses'],
        'final_digests_present': 'final_digests' in a and 'final_digests' in b,
        'final_digests_equal_when_present': a['final_digests'] == b['final_digests'] if 'final_digests' in a and 'final_digests' in b else None,
        'plan_sha256_equal': a['plan_sha256'] == b['plan_sha256'],
        'plan_summary_equal': a['plan_summary'] == b['plan_summary'],
    })
capability = json.loads((packet / 'raw/sdk/capability.json').read_text())
sdk_sha = hashlib.sha256((packet / 'raw/sdk/profiling.py.txt').read_bytes()).hexdigest()
audit['sdk'] = {
    'source_sha256': sdk_sha,
    'matches_candidate_pin': sdk_sha == 'ac8a9a43e5d8326cf532dfd2916cf6b7d7833d3a1d67e8ebe0e075aa1dfea3d6',
    'metadata_scope': capability['scope'], 'commands': capability['commands'],
    'missing_referenced_files': [c['file'] for c in capability['commands'] if not (packet / 'raw/sdk' / c['file']).exists()] +
        [v['local_file'] for v in capability['files'].values() if not (packet / 'raw/sdk' / v['local_file']).exists()],
    'native_execution_evidence': False,
}
Path('/work/receipt_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({'off_on_comparison': audit['off_on_comparison'], 'sdk': audit['sdk']}, indent=2))
