"""Synthetic local-only terminal tests; never invoke a remote or a launch."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('manual_handoff', HERE/'manual_job.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
_, packet = m.checked_packet('trn21-k2rescue')
b = packet['binding']


def member(x): return {'text': json.dumps(x)}


class FailedTimingHandoff(unittest.TestCase):
    def fixture(self):
        return {'files': {
            'terminal.json': member({'status': 'FULL_DRIVER_FAILED', 'error': 'TRAIN_EXIT_1'}),
            'run-exit-code.txt': {'text': '1\n'},
            'package/train.py': {'sha256': b['source_sha256']},
            'NATIVE_BINDING.json': member(b),
            'outer-flock-exit.json': member({'exit_code': 1}),
        }}

    def call(self, prior, key='trn21-finalcold', previous='trn21-k2rescue'):
        return m.timing_predecessor_outcome(key, previous, prior, packet)

    def test_exact_failed_timing_allows_only_cold_fallback(self):
        self.assertEqual(self.call(self.fixture()), 'FAILED_K2_ARCHIVED_COLD_FALLBACK_ONLY')

    def test_completed_timing_unchanged(self):
        p = self.fixture()
        p['files']['terminal.json'] = member({'status': 'TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS', 'train_exit_code': 0})
        p['files']['train/rehearsal.json'] = member({'charged_through_process_exit_within_budget': True})
        self.assertEqual(self.call(p), 'TIMING_COMPLETED')
        p['files']['train/rehearsal.json'] = member({'charged_through_process_exit_within_budget': False})
        with self.assertRaises(AssertionError): self.call(p)

    def test_wrong_destination_or_predecessor(self):
        for key, previous in [('trn22-finalseed43','trn21-k2rescue'), ('trn21-finalcold','old-k2')]:
            with self.assertRaises(AssertionError): self.call(self.fixture(), key, previous)

    def test_bad_failure_receipts(self):
        cases = [
            ('terminal.json', member({'status': 'RUNNING', 'error': 'TRAIN_EXIT_1'})),
            ('terminal.json', member({'status': 'FULL_DRIVER_FAILED', 'error': 'SOURCE_SHA'})),
            ('run-exit-code.txt', {'text': '0\n'}),
            ('run-exit-code.txt', {'text': '2\n'}),
            ('package/train.py', {'sha256': 'wrong'}),
            ('NATIVE_BINDING.json', member({**b, 'host': 'trn22'})),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                p = self.fixture(); p['files'][name] = value
                with self.assertRaises(AssertionError): self.call(p)
        p = self.fixture(); del p['files']['outer-flock-exit.json']
        with self.assertRaises(AssertionError): self.call(p)


if __name__ == '__main__': unittest.main()
