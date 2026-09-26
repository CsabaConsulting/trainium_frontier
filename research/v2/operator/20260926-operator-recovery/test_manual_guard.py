"""CPU-only refusal tests for the unchanged remote staging body's launch guards."""
import ast
import base64
import copy
import datetime
import fcntl
import importlib.util
import json
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
    def test_final_seed_parallel_cold_still_requires_own_predecessor(self):
        spec = importlib.util.spec_from_file_location('manual_final_seed_under_test', HERE / 'manual_job.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fake = types.SimpleNamespace(local_state=lambda: {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                                          'local_stops_present': [], 'breaker_tripped': {}})
        entry, packet = module.checked_packet('trn22-finalseed43')
        _, cold = module.checked_packet('trn21-finalcold')
        self.assertTrue(entry['parallel_cold_replay_allowed'])
        self.assertTrue(entry['cold_replay_required_for_submission'])
        self.assertNotIn('requires_cold_bundle',entry)
        self.assertEqual(packet['binding']['source_sha256'],cold['binding']['source_sha256'])
        self.assertEqual(packet['binding']['launch_command'].removesuffix(' --seed 43'),cold['binding']['launch_command'])
        self.assertEqual(packet['binding']['train_environment']['NEURON_COMPETITION_R1_SEED'],'43')
        with patch.object(module, 'connector', return_value=fake), \
             patch.object(module, 'remote', side_effect=AssertionError('MUST_NOT_DISPATCH')):
            with self.assertRaisesRegex(FileNotFoundError, 'test-unused-predecessor'):
                module.stage('trn22-finalseed43', execute=True, previous='test-unused-predecessor')
        self.assertFalse((HERE / 'manual-jobs/trn22-finalseed43/attempt.json').exists())

    def test_rehashed_wrong_provenance_refused(self):
        spec = importlib.util.spec_from_file_location('manual_job_under_test', HERE / 'manual_job.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest_path = HERE / 'manual-jobs/manifest.json'
        original_manifest = json.loads(manifest_path.read_bytes())
        key = 'trn21-muon150'
        packet_path = HERE / 'manual-jobs' / original_manifest[key]['packet']
        original_packet = json.loads(packet_path.read_bytes())
        real_read = Path.read_bytes
        for field, error in [('train_sha256', 'PROVENANCE_SOURCE_MISMATCH'),
                             ('label', 'PROVENANCE_LABEL_MISMATCH'),
                             ('launch_command', 'PROVENANCE_COMMAND_MISMATCH')]:
            with self.subTest(field=field):
                manifest, packet = copy.deepcopy(original_manifest), copy.deepcopy(original_packet)
                item = packet['files']['package/provenance.json']
                provenance = json.loads(base64.b64decode(item['b64']))
                provenance[field] = 'wrong-test-value'
                data = json.dumps(provenance).encode()
                item.update(b64=base64.b64encode(data).decode(), sha256=module.sha(data))
                packet_raw = json.dumps(packet).encode()
                manifest[key]['packet_sha256'] = module.sha(packet_raw)

                def read(path):
                    if path == packet_path: return packet_raw
                    if path == manifest_path: return json.dumps(manifest).encode()
                    return real_read(path)

                with patch.object(Path, 'read_bytes', read), patch.object(module, 'remote', side_effect=AssertionError('NO_HOST_CALL')):
                    with self.assertRaisesRegex(AssertionError, error):
                        module.checked_packet(key)

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
