"""Compare raw packet records, without trusting their verdicts or counter names."""
import collections
import json
from pathlib import Path

OUT = Path('/work')
arms = {a: json.loads(Path(f'/packet/raw/tiny-{a}.json').read_text())
        for a in ('base', 'default', 'off', 'on')}

def differences(a, b, path=''):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        result = [f'{path}/{k}:missing' for k in a.keys() ^ b.keys()]
        for k in a.keys() & b.keys():
            result.extend(differences(a[k], b[k], f'{path}/{k}'))
        return sorted(result)
    if isinstance(a, list):
        if len(a) != len(b):
            return [path + ':length']
        return [p for i, (x, y) in enumerate(zip(a, b))
                for p in differences(x, y, f'{path}/{i}')]
    return [] if a == b else [path]

result = {'status': 'PASS', 'scope': 'supplied reduced CPU observations only',
          'comparisons': [], 'arms': {}, 'observer_caveat':
          'Observer implementation is not supplied. custom_mlp_nodes and '
          'mlp_custom_backward_calls are retained as reported fields, not independently '
          'verified counts of executed backward calls. Later connected_first_forward '
          'is null, not evidence of a later disconnection or later connectivity.'}
semantic_fields = ['losses', 'final_digests', 'n_changed', 'n_params',
                   'cross_rank_mismatches', 'plan_sha256', 'plan_summary',
                   'checkpoint_check', 'unchanged_params', 'unchanged_tables', 'tables']
for arm, data in arms.items():
    assert data['nproc'] == 4 and data['steps'] == 3 and data['micro_batch'] == [2, 128]
    ranks = sorted(data['ranks'], key=lambda r: r['rank'])
    assert [r['rank'] for r in ranks] == [0, 1, 2, 3]
    summaries = []
    for r in ranks:
        c = r['cache_observations']
        assert len(c['forwards']) == len(c['steps']) == 3
        assert all(f['input_shape'] == [2, 128] for f in c['forwards'])
        refresh_histogram = collections.Counter(
            (v['completed_steps'], v['forward_calls'], v['grad_enabled'])
            for v in c['cache_refreshes'])
        caches = c['forwards'][0]['caches']
        assert len(caches) == 40
        if arm == 'on':
            assert all(v['value']['dtype'] == 'torch.bfloat16' and
                       v['master_dtype'] == 'torch.float32' and v['requires_grad'] and
                       v['grad_fn'] == 'ToCopyBackward0' and
                       v['connected_first_forward'] is True for v in caches.values())
        else:
            assert all(v is None for v in caches.values())
        summaries.append({
            'rank': r['rank'], 'losses': r['losses'], 'n_params': r['n_params'],
            'n_changed': r['n_changed'],
            'checkpoint_file_sha256_reported': c.get('checkpoint_file_sha256'),
            'first_forward_cache_entries': len(caches),
            'first_forward_present_caches': sum(v is not None for v in caches.values()),
            'refresh_record_histogram': [dict(completed_steps=k[0], forward_calls=k[1],
                                             grad_enabled=k[2], records=v)
                                         for k, v in sorted(refresh_histogram.items())],
            'custom_mlp_nodes_reported': [f['custom_mlp_nodes'] for f in c['forwards']],
            'mlp_custom_backward_calls_reported': c['mlp_custom_backward_calls'],
            'step_fields': list(c['steps'][0]),
            'step_tensor_counts': [{k: len(v) for k, v in s.items()} for s in c['steps']],
            'checkpoint_check_reported': r.get('checkpoint_check'),
        })
        if arm == 'base':
            continue
        base = next(x for x in arms['base']['ranks'] if x['rank'] == r['rank'])
        bc = base['cache_observations']
        mismatch = []
        for field in semantic_fields:
            mismatch.extend(differences(base.get(field), r.get(field), field))
        for field in ['steps', 'checkpoint', 'checkpoint_file_sha256', 'scope']:
            mismatch.extend(differences(bc.get(field), c.get(field), 'cache_observations/' + field))
        if arm != 'on':
            mismatch.extend(differences(bc, c, 'cache_observations'))
        else:
            for i in range(3):
                for field in ['input_shape', 'loss']:
                    mismatch.extend(differences(bc['forwards'][i][field], c['forwards'][i][field],
                                                f'forwards/{i}/{field}'))
                if i > 0:
                    mismatch.extend(differences(bc['forwards'][i]['caches'], c['forwards'][i]['caches'],
                                                f'forwards/{i}/caches'))
            extra = c['cache_refreshes'][:40]
            assert all(v == dict(completed_steps=0, forward_calls=0, grad_enabled=True) for v in extra)
            mismatch.extend(differences(bc['cache_refreshes'], c['cache_refreshes'][40:], 'later_refresh_records'))
        result['comparisons'].append(dict(arm=arm, rank=r['rank'], mismatches=mismatch))
        if mismatch:
            result['status'] = 'FAIL'
    result['arms'][arm] = dict(
        source_sha256=data['train_py_sha256'], argv=data['argv'],
        appended_argv=data['appended_argv'], shims=data['shims'],
        prepare=data['prepare'], observer_sha256=data['observation_driver_sha256'],
        ranks=summaries)

(OUT / 'receipt_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'paired_rank_comparisons': len(result['comparisons']),
                  'mismatches': [c for c in result['comparisons'] if c['mismatches']]}, indent=2))
assert result['status'] == 'PASS'
