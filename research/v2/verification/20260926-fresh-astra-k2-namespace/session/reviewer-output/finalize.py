"""Finalize evidence without repeating or broadening the tests."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path('/work')
results = json.loads((root / 'results.json').read_text())
assert results['status'] == 'PASS'
assert len(results['tests']) == 91 and all(item['pass'] for item in results['tests'])
assert len(results['cases']) == 7 and not results['unexpected_errors']
assert (root / 'test.exitcode.txt').read_text() == '0\n'
assert (root / 'test.stderr.txt').read_bytes() == b''
assert (root / 'candidate.py').read_bytes().replace(b'    global nl, nisa\n', b'', 1) == (root / 'baseline.py').read_bytes()
decision = {
    'verdict': 'PASS',
    'default_verdict': 'REFUSE',
    'scope': 'Source suitability only for a separately bound, disabled tiny diagnostic of the single global nl, nisa declaration',
    'contract': {'V': 256, 'C': 128, 'T': 256, 'upstream_dtypes': ['BF16', 'FP32'],
                 'metadata': 'synthetic, stable token order', 'sum': 'token-ordered FP32',
                 'acceptance': 'identical bytes, atol=rtol=0'},
    'candidate_sha256': hashlib.sha256((root / 'candidate.py').read_bytes()).hexdigest(),
    'baseline_sha256': hashlib.sha256((root / 'baseline.py').read_bytes()).hexdigest(),
    'checker_sha256_identity_only': '89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850',
    'reference_sha256_identity_only': '14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f',
    'checker_reference_bodies_inspected': False,
    'supplied_evidence': {'input_manifest_verified': True, 'rank_receipts': 4,
                          'original_native_status': 'REFUSE_OR_UNSUPPORTED',
                          'original_native_error': "failed to resolve name 'nl.bfloat16'",
                          'terminal_status': 'DIAGNOSTIC_CHILD_FAILED',
                          'original_failure_waived': False, 'successor_native_receipts': 0},
    'independently_executed_evidence': {
        'checks_passed': 91, 'checks_total': 91, 'fixtures': 7, 'exact_byte_comparisons': 14,
        'expected_exceptions': 14, 'unexpected_errors': 0, 'attempts': 1, 'exit_code': 0,
        'entire_kernel_bytes_preserved': True, 'whole_module_minus_declaration_bytes_preserved': True,
        'default_disabled_and_lazy_imports_verified': True,
        'python_global_namespace_effect_and_rebinding_verified': True,
        'cpu_model_vs_integer_binary32_oracle': 'PASS',
        'installed_frontend_emulated_or_proven': False,
        'actual_native_arithmetic_proven': False,
        'results': 'results.json', 'source': 'reference_test.py', 'stdout': 'test.stdout.txt',
        'stderr': 'test.stderr.txt', 'command': 'run-command.txt'},
    'resources': results['resources'],
    'native_compilation': 'UNEXECUTED',
    'native_exactness': 'UNEXECUTED',
    'native_api_runtime_validation': 'UNEXECUTED',
    'performance': 'UNMEASURED',
    'quality': 'UNTESTED',
    'production_integration': 'NOT_ESTABLISHED; stated absent in lane specification',
    'adoption_authorized': False,
    'submission_authorized': False,
    'device_launch_authorized': False,
    'broader_use_verdict': 'REFUSE',
    'unknowns': [
        'Installed NKI frontend global-name lookup and decorator capture',
        'Actual API symbols, signatures, keyword and dtype support',
        'Access-pattern, vector-offset, indirect-dimension and DMA skip semantics',
        'BF16 transfer/conversion and FP32 rounding, signed zero and subnormal behavior',
        'Dynamic registers/loops, allocation, buffer/layout, engine and scheduling legality',
        'Synchronization, aliasing, native compiler transformations and dispatch',
        'NaN, Inf and overflow behavior; generality outside bounded finite fixtures',
        'Complete checker/reference bodies and case execution',
        'Native multi-rank exact bytes, performance, quality, training and production integration',
        'Concurrent global rebinding and behavior with changed imported module identities'],
    'artifacts_manifest': 'output-hashes.json',
    'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
(root / 'decision.json').write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n')
entries = []
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name != 'output-hashes.json':
        raw = path.read_bytes()
        entries.append({'path': str(path.relative_to(root)), 'bytes': len(raw),
                        'sha256': hashlib.sha256(raw).hexdigest()})
manifest = {'algorithm': 'sha256', 'root': '/work',
            'excludes': ['output-hashes.json (self-reference)'], 'files': entries}
(root / 'output-hashes.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
for entry in entries:
    raw = (root / entry['path']).read_bytes()
    assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
print(json.dumps({'verdict': decision['verdict'], 'scope': decision['scope'],
                  'manifest_entries_verified': len(entries),
                  'completed_utc': decision['completed_utc']}, sort_keys=True))
