"""Stdlib proof of literal-only successor; no process or tensor execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
P = json.loads((HERE / 'NATIVE_BINDING.disabled.json').read_text())
PARENT = 'df48361a619baa65fe9864cd815a1cd864d7a789'
OLD = 'research/v2/proposals/20260926-k2-reporting-native-binding/'


def blob(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=REPO)


Q = json.loads(blob(PARENT, OLD + 'NATIVE_BINDING.disabled.json'))


def mapped(raw):
    for old, new in [(Q['label'], P['label']),
                     (Q['support_sha256']['consumer_nki.py'], P['support_sha256']['consumer_nki.py']),
                     (Q['support_sha256']['rank_entry.py'], P['support_sha256']['rank_entry.py'])]:
        raw = raw.replace(old, new)
    return raw


class Mechanical(unittest.TestCase):
    def test_parent_frozen(self):
        names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', PARENT, '--', OLD], cwd=REPO).decode().splitlines()
        self.assertTrue(names)
        for name in names:
            self.assertEqual(blob(PARENT, name), (REPO / name).read_bytes())

    def test_driver_entry_command_only_constants(self):
        for name in ('host_driver.py', 'rank_entry.py', 'bound-command.txt'):
            self.assertEqual(mapped(blob(PARENT, OLD + name).decode()), (HERE / name).read_text(), name)
        self.assertEqual(blob(PARENT, OLD + 'validate_receipts.py'), (HERE / 'validate_receipts.py').read_bytes())

    def test_all_execution_settings_preserved(self):
        for key in ('command', 'checker_args', 'environment', 'dispatch_command', 'outer_command'):
            self.assertEqual(json.loads(mapped(json.dumps(Q[key]))), P[key], key)
        for key in ('expected_hostname', 'cache_keys', 'timeout_seconds', 'term_grace_seconds',
                    'pending_gates', 'completion_rule', 'scientific_scope', 'entry_wrapper'):
            self.assertEqual(P[key], Q[key], key)

    def test_frozen_source_scope_only(self):
        self.assertEqual(P['source_module_commit'], 'd0f0d7fbdbe07ddee39680ee1a2b79fdfac388ca')
        prefix = P['source_module_path'] + '/'
        for name in ('consumer_nki.py', 'check_native.py', 'reference.py'):
            actual = blob(P['source_module_commit'], prefix + name)
            self.assertEqual(hashlib.sha256(actual).hexdigest(), P['support_sha256'][name])
            prior = blob(Q['source_module_commit'], prefix + name)
            if name == 'consumer_nki.py':
                prior = prior.replace(b'def build_kernel(*, enabled=False):\n',
                                      b'def build_kernel(*, enabled=False):\n    global nl, nisa\n', 1)
            self.assertEqual(actual, prior)

    def test_disabled_retry_and_explicit_postflight(self):
        self.assertTrue(P['postflight']['required'])
        for word in ('PIDs', 'PPID/PGID/SID', 'compiler', 'neuron-ls', 'name filters', 'parent lock'):
            self.assertIn(word, P['postflight']['method'])
        self.assertFalse(P['retry_classification']['authorization'])
        self.assertEqual(P['retry_classification']['automatic_retries'], 0)
        self.assertEqual(P['postflight']['artifact'], 'postflight-explicit-pids.json')
        for key in ('stage_authorized', 'launch_authorized', 'source_adoption', 'operator_assigned',
                    'host_reserved', 'automatic_full', 'automatic_eval', 'automatic_next_job'):
            self.assertFalse(P[key])


if __name__ == '__main__':
    unittest.main(verbosity=2)
