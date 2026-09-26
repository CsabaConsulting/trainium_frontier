#!/usr/bin/env python3
"""Own stdlib-only scope/default tests; no independent or native disposition."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cpu_diagnostic_static', HERE / 'check_cpu.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class ScopeTests(unittest.TestCase):
    def test_protected_files_and_registration_scope(self):
        self.assertEqual(len(check.statics()), 7)

    def test_default_adapter_no_runtime(self):
        result = subprocess.run([sys.executable, '-I', '-B', str(HERE / 'adapter.py')],
                                capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(json.loads(result.stdout), dict(status='EXPERIMENT_DISABLED',
                         launch_authorized=False, torch_imported=False, nki_imported=False))

    def test_default_checker_no_runtime(self):
        result = subprocess.run([sys.executable, '-I', '-B', str(HERE / 'check_cpu.py')],
                                capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(json.loads(result.stdout), dict(status='EXPERIMENT_DISABLED',
                         launch_authorized=False, torch_imported=False, nki_imported=False))

    def test_factory_default_no_runtime(self):
        module = check.load_adapter()
        with self.assertRaisesRegex(RuntimeError, 'EXPERIMENT_DISABLED'):
            module.build_registered_forward(None, None)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('nki', sys.modules)

    def test_exact_forward_and_derivative_expression(self):
        tree = ast.parse((HERE / 'adapter.py').read_text())
        methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertEqual(len(methods['forward'].args.args), 6)
        self.assertEqual([ast.unparse(n) for n in methods['forward'].body],
                         ['return F.embedding(ids, weight).to(output_dtype)'])
        self.assertEqual([ast.unparse(n) for n in methods['backward'].body], [
            'counts, starts, positions, maximums = ctx.saved_tensors',
            'result = kernel(gradient.contiguous(), counts, starts, positions, maximums)',
            'return (result, None, None, None, None, None)'])
        self.assertEqual([ast.unparse(n) for n in methods['setup_context'].body], [
            '_, _, counts, starts, positions, maximums = inputs',
            'ctx.save_for_backward(counts, starts, positions, maximums)'])

    def test_no_production_or_host_calls(self):
        tree = ast.parse((HERE / 'adapter.py').read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {n.module if isinstance(n, ast.ImportFrom) else x.name
                 for n in imports for x in (n.names if isinstance(n, ast.Import) else [None])}
        self.assertEqual(names, {'torch', 'torch.nn.functional', 'torch_neuronx', 'json', 'sys'})
        self.assertFalse(any(isinstance(n, ast.Global) for n in ast.walk(tree)))

    def test_only_explicit_cpu_opt_in(self):
        code = (HERE / 'check_cpu.py').read_text()
        self.assertIn("parser.add_argument('--enable-cpu-test', action='store_true')", code)
        self.assertNotIn('import torch_neuronx', code)
        self.assertNotIn('import nki\n', code)


if __name__ == '__main__':
    unittest.main(verbosity=2)
