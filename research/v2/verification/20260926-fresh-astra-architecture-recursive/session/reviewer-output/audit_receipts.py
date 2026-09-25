"""Independent field-level audit of the four supplied receipts (not a rerun)."""
import hashlib
import json
import math
from pathlib import Path

root = Path('/packet/raw')
receipts = {n: json.loads((root / f'tiny-g2-{n}.json').read_text()) for n in ('reference', 'default', 'off', 'on')}
hashes = json.loads(Path('/packet/INPUT_MANIFEST.json').read_text())['source_sha256']
checks = []
def check(name, value, detail=None):
    checks.append(dict(name=name, passed=bool(value), detail=detail))

summary = {}
for name, r in receipts.items():
    ranks = r['ranks']
    check(name + ':source_hash', r['train_py_sha256'] == hashes['baseline' if name == 'reference' else 'candidate'])
    check(name + ':run_bounds', r['nproc'] == 4 and r['steps'] == 3 and r['micro_batch'] == [1, 16])
    check(name + ':exit', r['torchrun_rc'] == 0 and r['timed_out'] is False)
    check(name + ':ranks', sorted(x['rank'] for x in ranks) == [0, 1, 2, 3])
    for rank in ranks:
        prefix = f'{name}:rank{rank["rank"]}'
        check(prefix + ':loss_count_finite', len(rank['losses']) == 3 and all(math.isfinite(x) for x in rank['losses']))
        check(prefix + ':all_changed', rank['n_params'] == rank['n_changed'] == 65 and rank['unchanged_params'] == [])
        check(prefix + ':rank_claim', rank['cross_rank_mismatches'] == [])
        check(prefix + ':plan', rank['plan_sha256'] == r['plan_sha256'])
        check(prefix + ':plan_elements', sum(rank['plan_summary']['elements_by_owner']) == 4523316)
    digests = ranks[0]['final_digests']
    check(name + ':digest_format', len(digests) == 65 and all(len(v) == 16 and all(c in '0123456789abcdef' for c in v) for v in digests.values()))
    reload = ranks[0]['checkpoint_check']
    check(name + ':reload_claim', reload['ok'] and reload['deterministic'] and reload['path_exists'] and reload['logits_shape'] == [1, 16, 8192])
    argv = r['argv']
    check(name + ':flag', ('--recursive-schedule' in argv) == (name == 'on') and ('--no-recursive-schedule' in argv) == (name == 'off'))
    summary[name] = dict(source_sha256=r['train_py_sha256'], recorded_verdict=r['verdict'], plan_sha256=r['plan_sha256'],
        argv=argv, appended_argv=r['appended_argv'], losses={str(x['rank']): x['losses'] for x in ranks},
        digest_count=len(digests), plan_summary=ranks[0]['plan_summary'], checkpoint_check=reload,
        shims=r['shims'], tensor_finiteness_fields=[k for k in ranks[0] if 'finite' in k],
        ranks_with_final_digests=[x['rank'] for x in ranks if 'final_digests' in x],
        recorded_rank_statuses=[x['status'] for x in ranks])

baseline = receipts['reference']
comparisons = {}
for name in ('default', 'off', 'on'):
    r = receipts[name]
    a, b = baseline['ranks'][0]['final_digests'], r['ranks'][0]['final_digests']
    changed = [k for k in a if a[k] != b.get(k)]
    cmp = dict(losses_equal_by_rank=[a['losses'] == b['losses'] for a,b in zip(baseline['ranks'], r['ranks'])],
        parameter_digest_keys_equal=set(a) == set(b), changed_parameter_digests=changed,
        plan_hash_equal=baseline['plan_sha256'] == r['plan_sha256'],
        plans_equal=all(a['plan_summary'] == b['plan_summary'] for a,b in zip(baseline['ranks'], r['ranks'])))
    comparisons[name] = cmp
    check(name + ':plan_baseline_identity', cmp['plan_hash_equal'] and cmp['plans_equal'])
    if name != 'on':
        check(name + ':loss_baseline_identity', all(cmp['losses_equal_by_rank']))
        check(name + ':digest_baseline_identity', cmp['parameter_digest_keys_equal'] and not changed)
    else:
        check('on:expected_nonidentity', bool(changed))

out = dict(checks=checks, summaries=summary, comparisons_with_baseline=comparisons,
    limitations=['Receipt bytes and source fields authenticated to manifest, not cryptographic provenance of execution.',
    'Plan digests cannot be recomputed without full plans; compare recorded hashes and summaries only.',
    'Only rank 0 has parameter digests; cross-rank equality is a recorded empty-mismatch claim.',
    'Loss finiteness independently checked. No raw tensor values or explicit tensor-finiteness records supplied.',
    'Reload success/determinism is recorded; no checkpoint/logit bytes supplied for independent replay.',
    '16-hex-character parameter digests are equality evidence, not bytewise checkpoint verification.',
    'Recorded token counters and compile-body log wording do not override disclosed microbatch/fake-clock/CPU shims.'])
Path('/work/artifacts/receipt_audit.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(dict(checks=len(checks), failures=[x for x in checks if not x['passed']], comparisons=comparisons), indent=2))
raise SystemExit(any(not x['passed'] for x in checks))
