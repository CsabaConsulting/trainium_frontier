"""Bounded CPU self-check of diagnostic plumbing, NOT a native numerical gate.

Uses exact candidate CPU formulas and explicitly synthetic CPU kernel adapters;
never imports torch_neuronx/NKI, reads data, or creates a model/optimizer.
"""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import types
import unittest

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[4]
TRAIN = ROOT / 'workflow/proposals/research-v2/train.py'
CHECKER = Path(__file__).with_name('check_native.py')


def selected_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class CPUKernel:
    def __init__(self, function):
        self.function = function

    def __getitem__(self, lanes):
        if lanes != 2:
            raise ValueError('diagnostic only uses two lanes')
        return self.function


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ns = selected_definitions(CHECKER,
            {'metric', 'tensor_summary', 'diagnostic_comparison', 'inspect_saved_tensors', 'first_case_diagnostic'},
            dict(torch=torch, json=json))
        cls.source = selected_definitions(TRAIN, {'_fp8_row_reference', '_FP8Projection'}, dict(torch=torch, F=F))

    def test_exact_source_and_original_gate_preserved(self):
        self.assertEqual(hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
            '921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858')
        text = CHECKER.read_text()
        self.assertIn("if any(v['relative_rms'] > 0.01", text)
        self.assertIn("verdict='PASS_NATIVE_COMPONENT_NUMERICS_ONLY'", text)

    def test_summary_and_missing_nonfinite_gradients(self):
        compare = self.ns['diagnostic_comparison']
        reference = torch.tensor([3., 4.])
        zeros = compare(torch.zeros(2), reference)
        self.assertEqual(zeros['relative_rms'], 1.)
        self.assertEqual(zeros['nonzero'], 0)
        self.assertEqual(zeros['max_abs'], 0.)
        self.assertEqual(zeros['error_max_abs'], 4.)
        self.assertFalse(zeros['within_original_1_percent_rms'])
        self.assertFalse(compare(None, reference)['within_original_1_percent_rms'])
        nan = compare(torch.tensor([float('nan'), 0]), reference)
        self.assertFalse(nan['finite'])
        self.assertFalse(nan['within_original_1_percent_rms'])
        json.dumps(nan, allow_nan=False)

    def test_fixed_case_with_explicit_cpu_kernel_adapters(self):
        ref = self.source['_fp8_row_reference']

        def prepare(weight, lanes):
            q, s, effective = ref(weight)
            packed = q.to(torch.float8_e4m3fn).view(torch.uint8).reshape(256, 2, 128).permute(2, 1, 0).contiguous()
            return packed, s[:, 0].unsqueeze(0).expand(128, 256).contiguous(), effective

        def project(x, packed, scale, lanes):
            qx, sx, ex = ref(x)
            qw = packed.permute(2, 1, 0).contiguous().reshape(256, 256).view(torch.float8_e4m3fn).float()
            return ((qx @ qw.T) * scale[0].unsqueeze(0) * sx).bfloat16(), ex

        mod = types.SimpleNamespace(_fp8_row_reference=ref,
            _FP8Projection=self.source['_FP8Projection'],
            _FP8_PROJ_KERNELS=(CPUKernel(prepare), CPUKernel(project)),
            synchronize=lambda device: None)
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.ns['first_case_diagnostic'](mod, torch.device('cpu'),
                compiler=lambda fn: torch.compile(fn, backend='aot_eager', fullgraph=True, dynamic=False))
        self.assertEqual(result['verdict'], 'DIAGNOSTIC_ONLY_NOT_COMPONENT_PASS')
        self.assertFalse(result['native_kernel_executed'])
        self.assertFalse(result['full_component_gate_passed'])
        self.assertFalse(result['whole_step_measured'])
        self.assertEqual(result['shape'], [256, 256, 256])
        self.assertEqual(list(result['cases']), ['original_compiled_bound_apply', 'compiled_enclosing_wrapper',
            'eager_apply_inspected', 'compiled_bound_apply_inspected_replay'])
        for name, case in result['cases'].items():
            with self.subTest(case=name):
                self.assertEqual(case['status'], 'OBSERVED')
                for field in ('forward', 'dx', 'dw', 'upstream_roundtrip'):
                    self.assertEqual(case[field]['relative_rms'], 0.)
                self.assertEqual(case['dx']['nonzero'], 65536)
                self.assertEqual(case['dw']['nonzero'], 65536)
                self.assertEqual(case['dx']['max_abs'], 4.875)
                self.assertEqual(case['dw']['max_abs'], 326.)
                if case['pre_backward_saved_tensor_inspection']:
                    saved = case['saved_context']
                    self.assertTrue(saved['tensors'])
                    self.assertFalse(saved['inspection_errors'])
                else:
                    self.assertNotIn('saved_context', case)
        for name in ('manual_native_vjp', 'ordinary_bf16_control'):
            self.assertEqual(result[name]['dx']['relative_rms'], 0.)
            self.assertEqual(result[name]['dw']['relative_rms'], 0.)
        json.dumps(result, allow_nan=False)
        # Deliberate CPU-only VJP fault proves the diagnostic distinguishes
        # actual zero gradients from error magnitudes and never grants PASS.
        original = mod._FP8Projection

        class ZeroGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w):
                return original.forward(ctx, x, w)

            @staticmethod
            def backward(ctx, gradient):
                dx, dw = original.backward(ctx, gradient)
                return torch.zeros_like(dx), torch.zeros_like(dw)

        mod._FP8Projection = ZeroGradient
        with contextlib.redirect_stdout(io.StringIO()):
            failed = self.ns['first_case_diagnostic'](mod, torch.device('cpu'), compiler=lambda fn: fn)
        case = failed['cases']['original_compiled_bound_apply']
        self.assertEqual(case['dx']['relative_rms'], 1.)
        self.assertEqual(case['dw']['relative_rms'], 1.)
        self.assertEqual(case['dx']['max_abs'], 0.)
        self.assertEqual(case['dw']['max_abs'], 0.)
        self.assertEqual(case['dx']['error_max_abs'], 4.875)
        self.assertEqual(case['dw']['error_max_abs'], 326.)
        self.assertEqual(failed['manual_native_vjp']['dx']['relative_rms'], 0.)
        self.assertFalse(failed['full_component_gate_passed'])
        self.assertEqual(failed['verdict'], 'DIAGNOSTIC_ONLY_NOT_COMPONENT_PASS')


if __name__ == '__main__':
    unittest.main()
