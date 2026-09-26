"""CPU-only refusal tests for the unchanged remote staging body's launch guards."""
import ast
import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
template = HERE.parent / '20260926-final-sprint-p0/stage_dispatch.py'
body = next(ast.literal_eval(n.value) for n in ast.parse(template.read_bytes()).body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'body' for t in n.targets))


class GuardRefusal(unittest.TestCase):
    def run_case(self, mode):
        cache = Path('/home/poojith-devan/.cache')
        with tempfile.TemporaryDirectory(prefix='trainium-manual-guard-', dir=cache) as folder:
            base = Path(folder)
            lock = base / 'device0.lock'
            lock.touch()
            s = lock.stat()
            root = base / 'unused-job'
            binding = {'root': str(root), 'prepare_path': str(base / 'prepare.py'), 'hostname': 'cpu-test',
                       'lock_path': str(lock), 'lock_identity': [s.st_dev, s.st_ino],
                       'resource_reserve': {'initial_floor_bytes': 15032385536}}
            payload = {'binding': binding, 'controller': {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, 'old_queue_pids': []}
            calls = []

            def run(argv, **kwargs):
                calls.append(argv)
                if argv[0] == '/usr/bin/ps':
                    return subprocess.CompletedProcess(argv, 0, '123 1 1 1 python torchrun train.py' if mode == 'ps_busy' else 'PID PPID PGID SID COMMAND ARGS\n', '')
                if 'neuron-ls' in argv[0]:
                    if mode == 'neuron_error':
                        return subprocess.CompletedProcess(argv, 1, '', 'test device query failure')
                    if mode == 'neuron_busy':
                        return subprocess.CompletedProcess(argv, 0, '[{"neuron_processes":[{"pid":123}],"logical_neuroncore_config":2,"nc_count":4}]', '')
                    return subprocess.CompletedProcess(argv, 0, '[{"neuron_processes":[],"logical_neuroncore_config":2,"nc_count":4}]', '')
                if argv[0] == '/usr/bin/fuser':
                    return subprocess.CompletedProcess(argv, 0 if mode == 'fuser_busy' else 1, '123' if mode == 'fuser_busy' else '', '')
                raise AssertionError('UNEXPECTED_COMMAND:' + repr(argv))

            held = os.open(lock, os.O_RDONLY)
            if mode == 'lock_busy':
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            captured = []
            real_open = os.open

            def open_track(*args, **kwargs):
                fd = real_open(*args, **kwargs)
                captured.append(fd)
                return fd

            try:
                with patch('socket.gethostname', return_value='cpu-test'), patch('os.getuid', return_value=1000), \
                     patch('subprocess.run', side_effect=run), patch('os.statvfs', return_value=types.SimpleNamespace(f_bavail=0, f_frsize=4096)), \
                     patch('os.open', side_effect=open_track), patch('subprocess.Popen', side_effect=AssertionError('MUST_NOT_LAUNCH')):
                    with self.assertRaises((AssertionError, BlockingIOError)):
                        exec(compile(body, '<unchanged-staging-body>', 'exec'), {'P': payload})
                self.assertFalse(root.exists())
                if mode == 'lock_busy':
                    self.assertEqual(calls, [])
                if mode == 'ps_busy':
                    self.assertEqual(len(calls), 1)
            finally:
                for fd in captured:
                    os.close(fd)
                os.close(held)

    def test_lock_busy(self): self.run_case('lock_busy')
    def test_ps_busy(self): self.run_case('ps_busy')
    def test_neuron_busy(self): self.run_case('neuron_busy')
    def test_neuron_query_failure(self): self.run_case('neuron_error')
    def test_fuser_busy(self): self.run_case('fuser_busy')
    def test_low_disk(self): self.run_case('low_disk')


if __name__ == '__main__':
    unittest.main()
