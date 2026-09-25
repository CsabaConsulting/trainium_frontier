"""Local fake-command checks plus real util-linux lifetime-lock inheritance."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('idle_guard', HERE / 'idle_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def device(processes=None):
    return [{'instance_type': 'trn2.3xlarge', 'neuron_device': 0, 'nc_count': 4,
             'logical_neuroncore_config': 2, 'neuroncore_ids': [0, 1, 2, 3],
             'neuron_processes': [] if processes is None else processes}]


class IdleGuardTests(unittest.TestCase):
    def run_case(self, extra='', devices=None, ps_rc=0, nl_rc=0, malformed_ps=False):
        calls = []
        def fake(command, **kwargs):
            calls.append(command)
            if command == guard.PS:
                out = 'unparseable' if malformed_ps else '10 9 python3 /current/host_driver.py\n9 1 flock /usr/bin/flock current\n' + extra
                return subprocess.CompletedProcess(command, ps_rc, out, '')
            self.assertEqual(command, guard.NEURON_LS)
            return subprocess.CompletedProcess(command, nl_rc, json.dumps(device() if devices is None else devices), '')
        evidence = {}
        guard.check_idle(fake, 10, 9, evidence)
        return evidence, calls

    def test_clean(self):
        evidence, calls = self.run_case()
        self.assertEqual(calls, [guard.PS, guard.NEURON_LS])
        self.assertEqual(evidence['neuron_ls']['process_pids'], [])

    def test_busy_torchrun(self):
        with self.assertRaisesRegex(RuntimeError, 'PS_NATIVE_JOB_BUSY'):
            self.run_case('25 1 python3 /venv/bin/torchrun --nproc_per_node=4 train.py\n')

    def test_busy_compiler(self):
        with self.assertRaisesRegex(RuntimeError, 'PS_NATIVE_JOB_BUSY'):
            self.run_case('25 1 neuronx-cc /opt/bin/neuronx-cc compile\n')

    def test_busy_eval(self):
        with self.assertRaisesRegex(RuntimeError, 'PS_NATIVE_JOB_BUSY'):
            self.run_case('25 1 python3 /repo/prepare.py eval-public --checkpoint checkpoint\n')

    def test_busy_device(self):
        with self.assertRaisesRegex(RuntimeError, 'NEURON_DEVICE_BUSY'):
            self.run_case(devices=device([{'pid': 123}]))

    def test_ps_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'PS_COMMAND_FAILED'):
            self.run_case(ps_rc=1)

    def test_device_command_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'NEURON_LS_COMMAND_FAILED'):
            self.run_case(nl_rc=1)

    def test_ps_parse_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'PS_PARSE_FAILED'):
            self.run_case(malformed_ps=True)

    def test_unknown_device_schema(self):
        with self.assertRaisesRegex(RuntimeError, 'NEURON_LS_DEVICE_SCHEMA_UNKNOWN'):
            self.run_case(devices=[{'neuron_processes': []}])

    def test_refused_lock_receipt(self):
        records = []
        with patch.object(guard, 'inherited_lock', side_effect=RuntimeError('LOCK_ERROR')):
            with self.assertRaisesRegex(RuntimeError, 'LOCK_ERROR'):
                guard.require_idle_under_inherited_lock('/unused', '/unused', lambda name, value: records.append((name, value)))
        self.assertEqual(records[0][1]['status'], 'REFUSED')
        self.assertIn('LOCK_ERROR', records[0][1]['error'])

    def test_real_inherited_flock(self):
        # The exact production command shape is used; this script acts as its driver.
        with tempfile.TemporaryDirectory(prefix='trainium-idle-guard-test-') as tmp:
            lock = str(Path(tmp) / 'device0.lock')
            env = dict(os.environ, TRN_LOCAL_GUARD_PROBE=lock)
            result = subprocess.run(['/usr/bin/flock', '-n', lock, '/usr/bin/python3', '-I', '-B', str(Path(__file__).resolve())], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('INHERITED_EXCLUSIVE_LOCK_AND_CONTENTION_PASS', result.stdout)


if __name__ == '__main__':
    probe = os.environ.get('TRN_LOCAL_GUARD_PROBE')
    if probe:
        evidence = guard.inherited_lock(probe, Path(__file__).resolve())
        contender = subprocess.run(['/usr/bin/flock', '-n', probe, '/usr/bin/true'], capture_output=True)
        if contender.returncode != 1 or not evidence['verified_existing_exclusive_lock']:
            raise RuntimeError('CONTENDING_LOCK_NOT_REFUSED')
        print('INHERITED_EXCLUSIVE_LOCK_AND_CONTENTION_PASS')
    else:
        unittest.main(verbosity=2)
