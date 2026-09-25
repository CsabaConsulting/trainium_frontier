"""Independent stdlib-only adversarial tests; no host or model execution."""
import ast
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
from unittest.mock import patch

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
    def test_guard_returns_authenticated_fd(self):
        records, order = [], []
        def auth(*args):
            order.append('lock')
            return {'fd': 42}
        def idle(*args):
            order.append('idle')
        with patch.object(guard, 'inherited_lock', auth), patch.object(guard, 'check_idle', idle):
            fd = guard.require_idle_under_inherited_lock('/fixture', '/fixture-driver',
                lambda name, evidence: records.append((name, evidence)))
        self.assertEqual(fd, 42)
        self.assertEqual(order, ['lock', 'idle'])
        self.assertEqual(records[0][1]['status'], 'PASS')

    def test_dispatch_contention_is_recorded(self):
        dispatch_path = GUARD.parent / 'dispatch_driver.py'
        ds = importlib.util.spec_from_file_location('candidate_dispatch', dispatch_path)
        dispatch = importlib.util.module_from_spec(ds)
        ds.loader.exec_module(dispatch)
        with tempfile.TemporaryDirectory(prefix='trn-guard-dispatch-') as tmp:
            root, calls = Path(tmp), []
            def contend(argv, **kwargs):
                calls.append(argv)
                return 1
            with patch.object(dispatch.subprocess, 'call', contend):
                self.assertEqual(dispatch.run(root), 1)
            record = json.loads((root / 'outer-flock-exit.json').read_text())
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][:3], ['/usr/bin/flock', '-n', '/home/ubuntu/.cache/trainium_freshstart/device0.lock'])
            self.assertEqual(record['status'], 'OUTER_COMMAND_FAILED')
            self.assertFalse(record['under_lock_receipt_exists'])
            self.assertFalse(record['driver_terminal_exists'])

    def test_timeout_keeps_actual_error(self):
        evidence = {}
        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(guard.PS, 15, stderr=b'fixture timeout detail')
        with self.assertRaisesRegex(RuntimeError, 'PS_COMMAND_TIMEOUT'):
            guard.check_idle(timeout, 120, 110, evidence)
        self.assertEqual(evidence['ps']['stderr'], 'fixture timeout detail')
        self.assertTrue(evidence['ps']['timed_out'])

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

    def test_driver_death_protects_child(self):
        for binding in ('ct', 'p'):
            with self.subTest(binding=binding):
                self.exercise_driver_death(binding)

    def exercise_driver_death(self, binding):
        """Run the unchanged candidate call AST with harmless fixture arguments."""
        with tempfile.TemporaryDirectory(prefix='trn-guard-orphan-') as tmp:
            lock, pidfile = Path(tmp) / 'device0.lock', Path(tmp) / 'child.pid'
            outer = subprocess.Popen(['/usr/bin/flock', '-n', str(lock), '/usr/bin/python3', '-I', '-B', str(HERE)],
                env={'PATH': '/usr/bin:/bin', 'INDEPENDENT_PROBE_LOCK': str(lock),
                     'INDEPENDENT_PROBE_MODE': 'wait-child', 'INDEPENDENT_CHILD_PIDFILE': str(pidfile),
                     'INDEPENDENT_DRIVER_BINDING': binding},
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
                self.assertEqual(second, 1, 'Lock released while child still lives')
                os.kill(child, signal.SIGTERM)
                child = None
                deadline = time.monotonic() + 5
                after_child = 1
                while after_child != 0 and time.monotonic() < deadline:
                    after_child = subprocess.run(['/usr/bin/flock', '-n', str(lock), '/usr/bin/true']).returncode
                    if after_child: time.sleep(.01)
                self.assertEqual(after_child, 0, 'Lock retained after fixture job exit')
                print(json.dumps({'finding': 'LOCK_RETAINED_WITH_LIVE_CHILD', 'binding': binding,
                                  'before_driver_death_contender_rc': first,
                                  'after_driver_death_contender_rc': second,
                                  'child_alive_at_second_contender': True,
                                  'after_child_exit_contender_rc': after_child}), flush=True)
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
            # Execute the reviewed driver's actual call expression unchanged.
            code = 'import os,time; from pathlib import Path; Path(os.environ["INDEPENDENT_CHILD_PIDFILE"]).write_text(str(os.getpid())); time.sleep(30)'
            name = ('20260926-ct-architecture-startup-only' if os.environ['INDEPENDENT_DRIVER_BINDING'] == 'ct'
                    else '20260926-trn21-profile-recursive6ef7')
            driver_path = GUARD.parent.parent / name / 'host_driver.py'
            tree = ast.parse(driver_path.read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                     and node.func.value.id == 'subprocess' and node.func.attr == 'call']
            if len(calls) != 1: raise RuntimeError('Expected exact one job call')
            with open(os.devnull, 'w') as out:
                namespace = {'subprocess': subprocess,
                    'COMMAND': ['/usr/bin/timeout', '--foreground', '30s', '/usr/bin/python3', '-I', '-B', '-c', code],
                    'ENV': {'INDEPENDENT_CHILD_PIDFILE': os.environ['INDEPENDENT_CHILD_PIDFILE']},
                    'ROOT': Path(os.environ['INDEPENDENT_CHILD_PIDFILE']).parent,
                    'out': out, 'lock_fd': evidence['fd']}
                eval(compile(ast.Expression(calls[0]), str(driver_path), 'eval'), namespace)
    else:
        unittest.main(verbosity=2)
