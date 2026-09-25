"""Verify delivered artifacts and write a content manifest; no repeat qualification."""
import hashlib
import json
from pathlib import Path
import torch

root = Path('/work')
auth = json.loads((root / 'authentication.json').read_text())
for s in auth['sources']:
    assert hashlib.sha256((root / s['path']).read_bytes()).hexdigest() == s['sha256']
results = json.loads((root / 'reference_results.json').read_text())
audit = json.loads((root / 'receipt_audit.json').read_text())
assert results['status'] == audit['status'] == 'PASS'
assert results['tensor_comparisons'] == 3183 and results['nonexact_comparisons'] == 0
raw = torch.load(root / 'reference_raw.pt', map_location='cpu', weights_only=True)
assert len(raw) == len(results['cases']) == 8
assert all(set(c['arms']) == {'base', 'default', 'off', 'on', 'oracle'} for c in raw.values())
decision = dict(
    decision='PASS_LIMITED_CPU_QUALIFICATION_SUITABILITY',
    candidate_sha256=auth['sources'][1]['sha256'],
    scope='Suitability for a later exact full-size CPU qualification in the assigned BF16 compute / FP32 master lane only',
    reference_status='PASS', reference_cases=8, tensor_comparisons=3183,
    nonexact_comparisons=0, receipt_paired_rank_comparisons=12,
    full117M_standard_G2='NOT_RUN', fresh_session_policy='NOT_SEPARATELY_CERTIFIED',
    native_authority=False, native_or_scientific_adoption='REFUSE',
    limitations=['pre-existing hot-load stale-cache behavior',
                 'pre-existing FP32 same-dtype Parameter alias outside assigned BF16 compute',
                 'raw observer counts do not independently prove execution'],
    report='/work/report.md')
(root / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
manifest = []
for p in sorted(root.rglob('*')):
    if p.is_file() and p.name not in {'artifact_manifest.json', 'finalize.stdout.txt', 'finalize.stderr.txt'}:
        data = p.read_bytes()
        manifest.append(dict(path=str(p.relative_to(root)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
(root / 'artifact_manifest.json').write_text(json.dumps(dict(
    artifacts=manifest, excluded_mutable_outputs=['finalize.stdout.txt', 'finalize.stderr.txt'],
    note='Manifest excludes itself and the open output streams of the finalizer.'), indent=2) + '\n')
print(json.dumps(dict(status='PASS', exit_code=0, raw_cases_loaded=len(raw),
                     artifacts_hashed=len(manifest), candidate_hash_rechecked=True), indent=2))
