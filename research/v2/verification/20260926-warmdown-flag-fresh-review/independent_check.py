"""Independent stdlib review; reads only the four supplied review inputs."""
import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import textwrap

ROOT = Path('/home/poojith-devan/.cache/trainium-warmdown-review.koSDnv')
NAMES = ('SOURCE.diff', 'SPEC.md', 'CPU_RECEIPT.json', 'IMPLEMENTATION_SPEC.tested.md')
data = {name: (ROOT / name).read_bytes() for name in NAMES}
hashes = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
diff = data['SOURCE.diff'].decode()
spec = data['SPEC.md'].decode()
old_spec = data['IMPLEMENTATION_SPEC.tested.md'].decode()
assert old_spec.count('source-size') == 1
assert old_spec.replace('source-size', 'model-size') == spec
receipt = json.loads(data['CPU_RECEIPT.json'])
assert receipt['source_pins']['research/v2/proposals/20260926-batch-lr021-warmdown-flag/IMPLEMENTATION_SPEC.md'] == hashes['IMPLEMENTATION_SPEC.tested.md']
candidate_hash = receipt['source_pins']['research/v2/lanes/sprint-batch-lr021-warmdown-flag/train.py']
assert candidate_hash == '4ec8b1b46562612cdcbfe3d7f66b0c8d25666bdbac3516b30bd9149c2dc08d2d'
added = [line[1:] for line in diff.splitlines() if line.startswith('+') and not line.startswith('+++')]
removed = [line[1:] for line in diff.splitlines() if line.startswith('-') and not line.startswith('---')]
assert removed == ['    global TOTAL_BATCH_SIZE, MATRIX_LR']
assert added == [
    '    global TOTAL_BATCH_SIZE, MATRIX_LR, WARMDOWN_RATIO',
    "    parser.add_argument('--warmdown-ratio', type=float, default=WARMDOWN_RATIO)",
    '    if not math.isfinite(args.warmdown_ratio) or not 0.0 < args.warmdown_ratio <= 1.0:',
    "        parser.error('--warmdown-ratio must be finite and in (0, 1]')",
    '    WARMDOWN_RATIO = args.warmdown_ratio',
]
# Execute the actual supplied parser/validation/assignment fragment, with a
# fresh namespace each time to model independent process launches.
new_lines = [line[1:] for line in diff.splitlines() if line.startswith((' ', '+')) and not line.startswith('+++')]
start = new_lines.index("    parser.add_argument('--total-batch-size', type=int, default=TOTAL_BATCH_SIZE)")
end = new_lines.index('    WARMDOWN_RATIO = args.warmdown_ratio') + 1
fragment = textwrap.dedent('\n'.join(new_lines[start:end]))
def candidate(argv):
    ns = dict(TOTAL_BATCH_SIZE=262144, MATRIX_LR=0.021, WARMDOWN_RATIO=0.75, math=math)
    parser = argparse.ArgumentParser(prog='isolated-cli-check')
    # Adapt argv plumbing only; parser statements themselves are unmodified.
    original = parser.parse_args
    parser.parse_args = lambda: original(argv)
    ns['parser'] = parser
    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            exec(compile(fragment, 'supplied-diff-fragment', 'exec'), ns)
    except SystemExit as exc:
        return ('error', exc.code, errors.getvalue().splitlines()[-1])
    return ('ok', ns['TOTAL_BATCH_SIZE'], ns['MATRIX_LR'].hex(), ns['WARMDOWN_RATIO'].hex())

# Own independent mathematical domain predicate, separate from the diff.
def reference(raw):
    value = 0.75 if raw is None else float(raw)
    return value if value > 0 and value <= 1 and value not in (math.inf, -math.inf) else None

cases = [None, '.75', '.60', '.85', '1', '0', '-0.0', '-0.1',
         '1.0000000000000002', 'nan', 'NaN', 'inf', '-inf', '1e999', '-1e999',
         '5e-324', repr(math.nextafter(1.0, 0.0)), '1e-999']
results = []
for raw in cases:
    argv = [] if raw is None else ['--warmdown-ratio=' + raw]
    got = candidate(argv)
    expected = reference(raw)
    if expected is None:
        assert got == ('error', 2, 'isolated-cli-check: error: --warmdown-ratio must be finite and in (0, 1]'), (raw, got)
    else:
        assert got == ('ok', 262144, float(.021).hex(), expected.hex()), (raw, got)
    results.append({'argument': raw, 'result': list(got)})
assert candidate([]) == candidate(['--warmdown-ratio=.75'])
assert candidate(['--total-batch-size=262144', '--matrix-lr=.021', '--warmdown-ratio=.60']) == ('ok', 262144, float(.021).hex(), float(.60).hex())
invalid_existing = ['--total-batch-size=0', '--total-batch-size=-1', '--matrix-lr=0', '--matrix-lr=-1', '--matrix-lr=nan', '--matrix-lr=inf', '--matrix-lr=-inf']
for argument in invalid_existing:
    for warmdown in ([], ['--warmdown-ratio=nan']):
        assert candidate([argument] + warmdown) == ('error', 2, 'isolated-cli-check: error: positive total batch and finite positive matrix LR required')
checks = receipt['checks']
assert len(checks) == 4
assert checks[0]['result'] == checks[1]['result']
assert checks[2]['result'] == checks[3]['result']
assert all(len(row['result']['steps']) == 3 and row['result']['parameters'] == 1475892 for row in checks)
print(json.dumps({
    'verdict': 'PASS', 'scope': 'supplied diff and independent reconstructed CLI fragment only',
    'input_sha256': hashes, 'receipt_candidate_source_sha256_not_recomputed': candidate_hash,
    'spec_wording_only_change': True, 'exact_allowed_diff': True,
    'independent_parser_cases': results, 'existing_invalid_checks': 14,
    'default_explicit075_hex_exact': True,
    'receipt_default_explicit_pairs_equal': True,
    'model_or_native_execution': False, 'python': sys.version.split()[0],
}, indent=2))
