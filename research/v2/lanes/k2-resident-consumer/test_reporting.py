"""Bounded reporting regression; default stdlib scalar emulation, never native.

Optional K2_REPORT_TEST_REAL_TORCH=1 tests the same helper using <=1-element
CPU tensors, only after the shared tensor slot is explicitly available.
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PATH = 'research/v2/lanes/k2-resident-consumer/check_native.py'
OLD = subprocess.check_output(['git', 'show', '077481d0:' + PATH], cwd=REPO).decode()
NEW = (HERE / 'check_native.py').read_text()


class ScalarTensor:
    """Only the extracted reporting helper's scalar operations; no NKI emulation."""
    dtype = 'torch.float32'
    shape = (1, 1)

    def __init__(self, value): self.value = value
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def view(self, dtype): return self
    def numpy(self): return self
    def tobytes(self): return struct.pack('<f', self.value)
    def numel(self): return 1
    def double(self): return self
    def __sub__(self, other): return ScalarTensor(self.value - other.value)
    def abs(self): return ScalarTensor(abs(self.value))
    def max(self): return self.value


class ScalarTorch:
    uint8 = 'torch.uint8'

    @staticmethod
    def isfinite(value):
        return SimpleNamespace(all=lambda: math.isfinite(value.value))

    @staticmethod
    def count_nonzero(value): return int(value.value != 0)


REAL_TORCH = os.environ.get('K2_REPORT_TEST_REAL_TORCH') == '1'
if REAL_TORCH:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tensor = lambda value: torch.tensor([[value]], dtype=torch.float32)
else:
    torch = ScalarTorch
    tensor = ScalarTensor


def main_node(source):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')


def comparison(source):
    scope = {'torch': torch, 'hashlib': hashlib, 'math': math}
    for name in ('tensor_bytes', 'compare'):
        node = next(n for n in main_node(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<exact-helper>', 'exec'), scope)
    return scope['compare']


class Reporting(unittest.TestCase):
    def test_original_defect_retained_new_nonfinite_receipts_valid(self):
        old, new = comparison(OLD), comparison(NEW)
        for value in (float('inf'), -float('inf'), float('nan')):
            old_record = old(tensor(value), tensor(0.0), 'direct_consumer')
            with self.assertRaises(ValueError): json.dumps(old_record, allow_nan=False)
            record = new(tensor(value), tensor(0.0), 'direct_consumer')
            self.assertFalse(record['exact'])
            self.assertFalse(record['finite'])
            self.assertIsNone(record['max_abs'])
            self.assertIn(record['max_abs_nonfinite'], ('inf', 'nan'))
            case = {'checks': [record], 'status': 'FAIL_EXACTNESS'}
            receipt = {'cases': [case], 'status': 'REFUSE_OR_UNSUPPORTED'}
            self.assertEqual(json.loads(json.dumps(receipt, allow_nan=False)), receipt)

    def test_finite_and_signed_zero_records_unchanged(self):
        old, new = comparison(OLD), comparison(NEW)
        for value in (0.0, -0.0, 1.0, -1.0):
            before = old(tensor(value), tensor(0.0), 'control')
            after = new(tensor(value), tensor(0.0), 'control')
            self.assertEqual(before, after)
            self.assertEqual(after['exact'], struct.pack('<f', value) == struct.pack('<f', 0.0))
            json.dumps(after, allow_nan=False)

    def test_finalization_complete_refusal_and_cleanup_on_reporting_errors(self):
        body = next(n for n in main_node(NEW).body if isinstance(n, ast.Try) and n.finalbody).finalbody
        code = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), '<exact-finalizer>', 'exec')
        for fault in ('none', 'invalid_metric', 'existing_file', 'stdout_failure'):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                file = out / 'rank0.json'
                record = comparison(NEW)(tensor(float('nan')), tensor(0.0), 'direct_consumer')
                receipt = {'status': 'REFUSE_OR_UNSUPPORTED', 'cases': [{'status': 'FAIL_EXACTNESS', 'checks': [record]}], 'error': 'EXACTNESS_FALSIFIED'}
                if fault == 'invalid_metric': receipt['injected_bad_metric'] = float('nan')
                if fault == 'existing_file': file.write_text('protected previous receipt')
                cleaned = []
                scope = dict(receipt=receipt, time=SimpleNamespace(monotonic=lambda: 2.0), started=1.0,
                             args=SimpleNamespace(out_dir=out), rank=0, json=json,
                             dist=SimpleNamespace(destroy_process_group=lambda: cleaned.append(True)))
                if fault == 'stdout_failure':
                    def failed_print(*args, **kwargs): raise OSError('injected stdout failure')
                    scope['print'] = failed_print
                with contextlib.redirect_stdout(io.StringIO()):
                    if fault == 'none': exec(code, scope)
                    else:
                        with self.assertRaises((ValueError, FileExistsError, OSError)): exec(code, scope)
                self.assertEqual(cleaned, [True])
                if fault == 'invalid_metric': self.assertFalse(file.exists())
                elif fault == 'existing_file': self.assertEqual(file.read_text(), 'protected previous receipt')
                else: self.assertEqual(json.loads(file.read_text())['status'], 'REFUSE_OR_UNSUPPORTED')

    def test_only_reporting_scope_and_immutable_kernel_reference(self):
        before, after = ast.parse(OLD), ast.parse(NEW)
        after.body = [n for n in after.body if not (isinstance(n, ast.Import) and [a.name for a in n.names] == ['math'])]
        bm = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        am = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        original_compare = next(n for n in bm.body if isinstance(n, ast.FunctionDef) and n.name == 'compare')
        am.body = [copy.deepcopy(original_compare) if isinstance(n, ast.FunctionDef) and n.name == 'compare' else n for n in am.body]
        next(n for n in am.body if isinstance(n, ast.Try) and n.finalbody).finalbody = copy.deepcopy(next(n for n in bm.body if isinstance(n, ast.Try) and n.finalbody).finalbody)
        self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False))
        for file in ('consumer_nki.py', 'reference.py', 'test_reference.py'):
            old = subprocess.check_output(['git', 'show', '077481d0:research/v2/lanes/k2-resident-consumer/' + file], cwd=REPO)
            self.assertEqual(old, (HERE/file).read_bytes())

    def test_default_disabled_without_native_import(self):
        result = subprocess.run(['/usr/bin/python3', '-I', '-B', str(HERE/'check_native.py')], capture_output=True, text=True, timeout=10, check=True)
        self.assertEqual(json.loads(result.stdout), {'status': 'DISABLED', 'native_imported': False, 'native_executed': False})


if __name__ == '__main__':
    unittest.main(verbosity=2)
