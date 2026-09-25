"""Run the reviewed timer suite unchanged, plus exact L2 port identity checks."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import test_startup_predictor as reviewed

ROOT = Path(__file__).resolve().parents[4]
BASE_COMMIT = '99261e750ec8712f6ccb4a2d961d6134d3f87b12'
BASE_SHA = '52e234e90f33fc2ccfcc22edb7afe101be25089020a58c352afda4470ac5a7c1'
SOURCE = ROOT / 'workflow/proposals/research-v2/train.py'


def from_git(ref, path='workflow/proposals/research-v2/train.py'):
    return subprocess.run(['git', 'show', f'{ref}:{path}'], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout


class L2PortTests(unittest.TestCase):
    def test_only_two_reviewed_edits_restore_exact_original_bytes_and_ast(self):
        old = from_git(BASE_COMMIT)
        fixed = SOURCE.read_text()
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(), BASE_SHA)
        clear = ('            # Compile-heavy startup samples must not predict steady-state step cost.\n'
                 '            recent_step_times.clear()\n')
        new_max = 'max(recent_step_times[-20:], default=0.0)'
        self.assertEqual(fixed.count(clear), 1)
        self.assertEqual(fixed.count(new_max), 1)
        reverted = fixed.replace(clear, '', 1).replace(new_max, 'max(recent_step_times[-20:])', 1)
        self.assertEqual(reverted, old)
        self.assertEqual(ast.dump(ast.parse(reverted)), ast.dump(ast.parse(old)))

    def test_timer_ast_equals_reviewed_a4aa4351(self):
        fixed = reviewed.ActualTimer(SOURCE.read_text())
        reference = reviewed.ActualTimer(from_git('a4aa4351789115a2d0aa947d9ca00ff689f9bd0b'))
        self.assertEqual(ast.dump(ast.Module(body=fixed.boundary_nodes, type_ignores=[])),
                         ast.dump(ast.Module(body=reference.boundary_nodes, type_ignores=[])))

    def test_l2_old_regression_fixed_and_charging_identical(self):
        old = reviewed.ActualTimer(from_git(BASE_COMMIT))
        fixed = reviewed.ActualTimer(SOURCE.read_text())
        for duration in reviewed.A2_DURATIONS:
            old_stop, fixed_stop = old.advance(duration), fixed.advance(duration)
        self.assertTrue(old_stop)
        self.assertFalse(fixed_stop)
        for name in ('observed_startup', 'startup_allowance_est', 'excess',
                     'budget_started', 'elapsed_after_step'):
            self.assertEqual(old.env[name], fixed.env[name])
        self.assertAlmostEqual(fixed.env['excess'], 1092.09)
        self.assertEqual(fixed.env['recent_step_times'], [])
        self.assertFalse(fixed.advance(0.62))


if __name__ == '__main__':
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(reviewed.StartupPredictorTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(L2PortTests),
    ])
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    receipt = {
        'verdict': 'PASS_SELF_CHECK_ONLY' if result.wasSuccessful() else 'FAIL',
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'base_commit': BASE_COMMIT, 'base_sha256': BASE_SHA,
        'candidate_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'reviewed_timer_commit': 'a4aa4351789115a2d0aa947d9ca00ff689f9bd0b',
        'reviewed_test_sha256': hashlib.sha256(Path(reviewed.__file__).read_bytes()).hexdigest(),
        'prepare_sha256': hashlib.sha256((ROOT / 'prepare.py').read_bytes()).hexdigest(),
        'scope': 'stdlib AST execution only; exact two-edit inverse restores original L2 bytes and AST; no model import, tensors, training or hosts',
        'independent_review': False,
    }
    Path(__file__).with_name('timer-port-self-check.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
