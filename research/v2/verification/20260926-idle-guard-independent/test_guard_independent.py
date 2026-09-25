"""Independent stdlib-only adversarial tests; no host or model execution."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve()
GUARD = HERE.parents[2] / 'operator/20260926-common-host-idle-guard/idle_guard.py'
spec = importlib.util.spec_from_file_location('candidate_guard', GUARD)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def observed_device():
    return [{'instance_type': 'trn2.3xlarge', 'neuron_device': 0, 'nc_count': 4,
             'logical_neuroncore_config': 2, 'neuroncore_ids': [0, 1, 2, 3],
             'neuron_processes': []}]


def check_fixture(extra_ps='', device_json=None, ps_stderr='', device_stderr=''):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv == guard.PS:
            return subprocess.CompletedProcess(argv, 0,
                '120 110 python3 /packet/host_driver.py\n'
                '110 1 flock /usr/bin/flock current\n' + extra_ps, ps_stderr)
        if argv != guard.NEURON_LS:
            raise AssertionError('Unexpected command')
        return subprocess.CompletedProcess(argv, 0,
            json.dumps(observed_device()) if device_json is None else device_json,
            device_stderr)
    evidence = {}
    guard.check_idle(run, 120, 110, evidence)
    return evidence, calls


class IndependentTests(unittest.TestCase):
    def test_clean_and_both_commands(self):
        evidence, calls = check_fixture()
        self.assertEqual(calls, [guard.PS, guard.NEURON_LS])
        self.assertEqual(evidence['ps']['excluded_pids'], [120, 110])

    def test_other_same_named_driver_not_excluded(self):
        with self.assertRaisesRegex(RuntimeError, 'PS_NATIVE_JOB_BUSY'):
            check_fixture('121 1 python3 /other/host_driver.py\n')

    def test_torchrun_busy(self):
        with self.assertRaisesRegex(RuntimeError, 'PS_NATIVE_JOB_BUSY'):
            check_fixture('400 1 python3 /venv/bin/torchrun --standalone train.py\n')

    def test_malformed_json_refused(self):
        with self.assertRaises(json.JSONDecodeError):
            check_fixture(device_json='[{broken-json]')

    def test_unknown_device_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'NEURON_LS_DEVICE_SCHEMA_UNKNOWN'):
            check_fixture(device_json='[]')

    def test_error_stderr_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'PS_COMMAND_FAILED'):
            check_fixture(ps_stderr='permission denied')
        with self.assertRaisesRegex(RuntimeError, 'NEURON_LS_COMMAND_FAILED'):
            check_fixture(device_stderr='runtime failure')

    def test_process_pid_boolean_refused(self):
        device = observed_device()
        device[0]['neuron_processes'] = [{'pid': True}]
        with self.assertRaisesRegex(RuntimeError, 'NEURON_LS_PROCESS_SCHEMA_UNKNOWN'):
            check_fixture(device_json=json.dumps(device))

    def test_direct_invocation_refused(self):
        with tempfile.TemporaryDirectory(prefix='trn-guard-direct-') as tmp:
            lock = Path(tmp) / 'device0.lock'
            lock.touch()
            result = subprocess.run(['/usr/bin/python3', '-I', '-B', str(HERE)],
                env={'PATH': '/usr/bin:/bin', 'INDEPENDENT_PROBE_LOCK': str(lock),
                     'INDEPENDENT_PROBE_MODE': 'authenticate'},
                capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('OUTER_FLOCK_COMMAND_MISMATCH', result.stderr)

    def test_real_lifetime_lock_and_contender(self):
        with tempfile.TemporaryDirectory(prefix='trn-guard-inherited-') as tmp:
            lock = Path(tmp) / 'device0.lock'
            result = subprocess.run(['/usr/bin/flock', '-n', str(lock), '/usr/bin/python3', '-I', '-B', str(HERE)],
                env={'PATH': '/usr/bin:/bin', 'INDEPENDENT_PROBE_LOCK': str(lock),
                     'INDEPENDENT_PROBE_MODE': 'authenticate'},
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['contender_rc'], 1)

    def test_driver_death_does_not_protect_child(self):
        """Reproduce the production subprocess.call FD boundary with a harmless child."""
        with tempfile.TemporaryDirectory(prefix='trn-guard-orphan-') as tmp:
            lock, pidfile = Path(tmp) / 'device0.lock', Path(tmp) / 'child.pid'
            outer = subprocess.Popen(['/usr/bin/flock', '-n', str(lock), '/usr/bin/python3', '-I', '-B', str(HERE)],
                env={'PATH': '/usr/bin:/bin', 'INDEPENDENT_PROBE_LOCK': str(lock),
                     'INDEPENDENT_PROBE_MODE': 'wait-child', 'INDEPENDENT_CHILD_PIDFILE': str(pidfile)},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            child = None
            try:
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(pidfile.exists(), 'Fixture child did not start')
                child = int(pidfile.read_text())
                first = subprocess.run(['/usr/bin/flock', '-n', str(lock), '/usr/bin/true']).returncode
                self.assertEqual(first, 1, 'Lock not held while driver waits')
                driver = int(outer.stdout.readline())
                os.kill(driver, signal.SIGKILL)  # Only the known fixture driver.
                outer.wait(timeout=5)
                os.kill(child, 0)
                second = subprocess.run(['/usr/bin/flock', '-n', str(lock), '/usr/bin/true']).returncode
                self.assertEqual(second, 0, 'Expected production FD boundary changed')
                print(json.dumps({'finding': 'LOCK_RELEASED_WITH_LIVE_CHILD',
                                  'before_driver_death_contender_rc': first,
                                  'after_driver_death_contender_rc': second,
                                  'child_alive': True}), flush=True)
            finally:
                if child is not None:
                    try: os.kill(child, signal.SIGTERM)
                    except ProcessLookupError: pass
                if outer.poll() is None:
                    outer.terminate()
                    outer.wait(timeout=5)
                outer.communicate(timeout=5)


if __name__ == '__main__':
    lock = os.environ.get('INDEPENDENT_PROBE_LOCK')
    if lock:
        evidence = guard.inherited_lock(lock, HERE)
        if os.environ['INDEPENDENT_PROBE_MODE'] == 'authenticate':
            rc = subprocess.run(['/usr/bin/flock', '-n', lock, '/usr/bin/true']).returncode
            print(json.dumps({'inherited': evidence['verified_existing_exclusive_lock'], 'contender_rc': rc}))
        else:
            print(os.getpid(), flush=True)
            # This is the drivers' same synchronous subprocess.call and default close_fds.
            code = 'import os,time; from pathlib import Path; Path(os.environ["INDEPENDENT_CHILD_PIDFILE"]).write_text(str(os.getpid())); time.sleep(30)'
            subprocess.call(['/usr/bin/python3', '-I', '-B', '-c', code],
                env={'INDEPENDENT_CHILD_PIDFILE': os.environ['INDEPENDENT_CHILD_PIDFILE']},
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        unittest.main(verbosity=2)
