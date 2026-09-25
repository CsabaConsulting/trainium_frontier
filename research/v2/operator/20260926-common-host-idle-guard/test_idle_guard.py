"""Local fake-command checks plus real util-linux lifetime-lock inheritance."""
import importlib.util
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
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

    def test_actual_stderr_preserved(self):
        evidence = {}
        def fake(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, '', 'ps: permission denied\n')
        with self.assertRaisesRegex(RuntimeError, 'PS_COMMAND_FAILED'):
            guard.check_idle(fake, 10, 9, evidence)
        self.assertEqual(evidence['ps']['stderr'], 'ps: permission denied\n')

    def test_timeout_evidence_preserved(self):
        evidence = {}
        def fake(command, **kwargs):
            raise subprocess.TimeoutExpired(command, 15, stderr=b'probe did not respond\n')
        with self.assertRaisesRegex(RuntimeError, 'PS_COMMAND_TIMEOUT'):
            guard.check_idle(fake, 10, 9, evidence)
        self.assertTrue(evidence['ps']['timed_out'])
        self.assertEqual(evidence['ps']['stderr'], 'probe did not respond\n')

    def test_outer_contention_is_terminal_receipt(self):
        spec = importlib.util.spec_from_file_location('dispatch', HERE / 'dispatch_driver.py')
        dispatch = importlib.util.module_from_spec(spec); spec.loader.exec_module(dispatch)
        with tempfile.TemporaryDirectory(prefix='trainium-outer-refusal-test-') as tmp:
            root = Path(tmp); lock = root / 'device0.lock'
            with lock.open('w') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(dispatch, 'LOCK', str(lock)):
                    self.assertEqual(dispatch.run(root), 1)
            receipt = json.loads((root / 'outer-flock-exit.json').read_text())
            self.assertEqual(receipt['status'], 'OUTER_COMMAND_FAILED')
            self.assertFalse(receipt['under_lock_receipt_exists'])
            self.assertFalse(receipt['driver_terminal_exists'])

    def test_real_inherited_flock(self):
        # The exact production command shape is used; this script acts as its driver.
        with tempfile.TemporaryDirectory(prefix='trainium-idle-guard-test-') as tmp:
            lock = str(Path(tmp) / 'device0.lock')
            env = dict(os.environ, TRN_LOCAL_GUARD_PROBE=lock)
            result = subprocess.run(['/usr/bin/flock', '-n', lock, '/usr/bin/python3', '-I', '-B', str(Path(__file__).resolve())], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('INHERITED_EXCLUSIVE_LOCK_AND_CONTENTION_PASS', result.stdout)

    def test_lock_survives_driver_death_until_child_exit(self):
        with tempfile.TemporaryDirectory(prefix='trainium-driver-death-test-') as tmp:
            lock = str(Path(tmp) / 'device0.lock')
            env = dict(os.environ, TRN_LOCAL_GUARD_PROBE=lock, TRN_LOCAL_GUARD_DEATH='1')
            proc = subprocess.Popen(['/usr/bin/flock', '-n', lock, '/usr/bin/python3', '-I', '-B', str(Path(__file__).resolve())], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            ready = json.loads(proc.stdout.readline())
            os.kill(ready['driver_pid'], 15)
            proc.wait(timeout=5)
            refused = subprocess.run(['/usr/bin/flock', '-n', lock, '/usr/bin/true'], capture_output=True)
            self.assertEqual(refused.returncode, 1, 'Child must retain same lifetime lock after driver death')
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                released = subprocess.run(['/usr/bin/flock', '-n', lock, '/usr/bin/true'], capture_output=True)
                if released.returncode == 0:
                    break
                time.sleep(.05)
            self.assertEqual(released.returncode, 0, 'Lock must release after harmless child exit')
            proc.stdout.close(); proc.stderr.close()


if __name__ == '__main__':
    probe = os.environ.get('TRN_LOCAL_GUARD_PROBE')
    if probe:
        evidence = guard.inherited_lock(probe, Path(__file__).resolve())
        if os.environ.get('TRN_LOCAL_GUARD_DEATH'):
            child = subprocess.Popen(['/usr/bin/timeout', '4s', '/usr/bin/python3', '-I', '-B', '-c', 'import time;time.sleep(1.5)'], pass_fds=(evidence['fd'],), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(json.dumps({'driver_pid': os.getpid(), 'child_pid': child.pid}), flush=True)
            child.wait()
            sys.exit(0)
        contender = subprocess.run(['/usr/bin/flock', '-n', probe, '/usr/bin/true'], capture_output=True)
        if contender.returncode != 1 or not evidence['verified_existing_exclusive_lock']:
            raise RuntimeError('CONTENDING_LOCK_NOT_REFUSED')
        print('INHERITED_EXCLUSIVE_LOCK_AND_CONTENTION_PASS')
    else:
        unittest.main(verbosity=2)
