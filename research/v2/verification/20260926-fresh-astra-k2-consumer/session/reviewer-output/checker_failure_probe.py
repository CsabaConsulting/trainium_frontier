#!/usr/bin/env python3
"""CPU-only probe of unmodified checker helpers and strict JSON serialization.
Extracts two Python function definitions verbatim; never enters main(), imports
NKI, compiles a kernel, mocks native success, or changes reconstructed sources.
"""
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys
import traceback
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT = Path('/work')
source_path = ROOT / 'reconstructed/research/v2/lanes/k2-resident-consumer/check_native.py'
source = source_path.read_text()
module = ast.parse(source)
main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
namespace = {'torch': torch, 'hashlib': hashlib}
extracted = []
for name in ('tensor_bytes', 'compare'):
    node = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == name)
    text = ast.get_source_segment(source, node)
    # ast source includes the original nested body indentation, which is valid
    # when its def line becomes top level. This evaluates ordinary Python only.
    exec(text, namespace)
    extracted.append({'name': name, 'line': node.lineno, 'source_sha256': hashlib.sha256(text.encode()).hexdigest()})

outdir = ROOT / 'checker_failure_artifacts'
outdir.mkdir(exist_ok=False)
report = {'executed_scope': 'verbatim extracted tensor_bytes/compare on CPU plus original JSON serialization calls',
          'native_main_executed': False, 'native_executed': False, 'extracted_functions': extracted, 'cases': []}
for name, word in [('finite_exact', 0), ('finite_mismatch', 0x3f800000),
                   ('signed_zero_mismatch', 0x80000000), ('positive_inf', 0x7f800000),
                   ('negative_inf', 0xff800000), ('quiet_nan', 0x7fc00000)]:
    actual = torch.tensor([[struct.unpack('<f', struct.pack('<I', word))[0]]], dtype=torch.float32)
    expected = torch.zeros((1, 1), dtype=torch.float32)
    (outdir / (name + '.actual.fp32le.bin')).write_bytes(actual.numpy().tobytes())
    (outdir / (name + '.expected.fp32le.bin')).write_bytes(expected.numpy().tobytes())
    comparison = namespace['compare'](actual, expected, 'direct_consumer')
    case = {'name': name, 'checks': [comparison], 'status': 'EXACT' if comparison['exact'] else 'FAIL_EXACTNESS'}
    receipt = {'scope': 'CPU_ONLY_SERIALIZATION_REPRODUCTION', 'cases': [case]}
    errors = []
    try:
        # Exact serialization at candidate line 155, prior to explicit refusal.
        json.dumps(case, allow_nan=False)
        case_serializable = True
    except Exception:
        case_serializable = False
        error = traceback.format_exc()
        errors.append(error)
        print(error, file=sys.stderr)
        # Candidate lines 175-178 keep the offending case in the receipt.
        receipt['status'] = 'REFUSE_OR_UNSUPPORTED'
        receipt['error'] = error
    file_path = outdir / (name + '.rank0.json')
    try:
        # Exact writer behavior at candidate lines 184-186 (including partial file).
        with file_path.open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
        receipt_serializable = True
    except Exception:
        receipt_serializable = False
        error = traceback.format_exc()
        errors.append(error)
        print(error, file=sys.stderr)
    try:
        json.loads(file_path.read_text())
        valid_json = True
    except json.JSONDecodeError:
        valid_json = False
    observed = {'name': name, 'actual_hex': f'{word:08x}', 'expected_hex': '00000000',
                'exact': comparison['exact'], 'finite': comparison['finite'],
                'max_abs_repr': repr(comparison['max_abs']), 'case_serializable': case_serializable,
                'receipt_serializable': receipt_serializable, 'receipt_valid_json': valid_json,
                'partial_receipt_bytes': file_path.stat().st_size, 'errors': errors}
    report['cases'].append(observed)
    print(json.dumps(observed), flush=True)
report['defect_reproduced'] = all(not c['exact'] and not c['finite'] and not c['case_serializable']
    and not c['receipt_serializable'] and not c['receipt_valid_json'] for c in report['cases'][-3:])
report['finite_controls_work'] = all(c['receipt_valid_json'] for c in report['cases'][:3])
report['classification'] = 'Checker failure-reporting defect; not native arithmetic falsification or a native API availability claim'
(ROOT / 'checker_failure_results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({'defect_reproduced': report['defect_reproduced'], 'finite_controls_work': report['finite_controls_work']}))
raise SystemExit(0 if report['defect_reproduced'] and report['finite_controls_work'] else 1)
