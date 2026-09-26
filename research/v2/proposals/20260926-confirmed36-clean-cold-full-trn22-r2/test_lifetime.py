"""Author CPU fault receipts, not independent acceptance or launch authority."""
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
OUTPUT = Path(os.environ['FULLCOLD_CPU_RECEIPT_ROOT']) if 'FULLCOLD_CPU_RECEIPT_ROOT' in os.environ else Path(tempfile.mkdtemp(prefix='fullcold-author-cpu-'))
OUTPUT = OUTPUT.resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # Reap the deliberately orphaned OLD-fail fixture.


def live(row):
    try:
        fields = Path(f"/proc/{row['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and int(fields[19]) == row['start_ticks']
    except (FileNotFoundError, ProcessLookupError):
        return False


def competitor(lock):
    fd = os.open(lock, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 1
        fcntl.flock(fd, fcntl.LOCK_UN)
        return 0
    finally:
        os.close(fd)


class LifetimeTests(unittest.TestCase):
    def run_case(self, mode, expected, parked=False):
        directory = OUTPUT/mode
        directory.mkdir()
        (directory/'case.txt').write_text(mode)
        lock = directory/'device0.lock'
        lock.touch()
        with (directory/'outer.log').open('x') as out:
            process = subprocess.Popen(['/usr/bin/flock', '-n', str(lock), '/usr/bin/python3',
                                        '-I', '-B', str(HERE/'cpu_lifetime_fixture.py')],
                                       cwd=directory, stdout=out, stderr=subprocess.STDOUT)
            sent = False
            probes = []
            boundaries = {}
            deadline = time.monotonic() + 8
            while process.poll() is None and time.monotonic() < deadline:
                rows = [json.loads(p.read_text()) for p in directory.glob('leaf-*.json')]
                alive = [r for r in rows if live(r)]
                if alive:
                    rc = competitor(lock)
                    probes.append({'live_pids':[r['pid'] for r in alive], 'competitor_rc':rc})
                    self.assertEqual(rc, 1, 'lock released with a live owned descendant')
                if mode in ('signal','interrupt') and alive and not sent:
                    guardian = json.loads((directory/'guardian.json').read_text())
                    os.kill(guardian['pid'], signal.SIGTERM if mode=='signal' else signal.SIGINT)
                    sent = True
                for marker in ('between-phases','postflight-entered'):
                    if (directory/marker).exists() and marker not in boundaries:
                        boundaries[marker] = competitor(lock)
                        self.assertEqual(boundaries[marker],1,'lock covers phase gap and postflight')
                events_path = directory/'events.jsonl'
                events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
                held_failure = any(e['event'] == 'FAILED_LOCK_RETAINED' for e in events)
                if mode == 'audit-fail':
                    held_failure = 'FAILED_LOCK_RETAINED' in (directory/'outer.log').read_text()
                if parked and held_failure:
                    self.assertEqual(competitor(lock), 1)
                    self.assertFalse(alive, 'fixture must be quiescent before test-only guardian disposal')
                    guardian = json.loads((directory/'guardian.json').read_text())
                    # Test-only disposal of our fail-locked CPU guardian AFTER all children exit.
                    os.kill(guardian['pid'], signal.SIGKILL)
                    break
                time.sleep(0.01)
            try:
                result = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.fail('bounded CPU fixture failed to finish: ' + str(directory))
        self.assertEqual(result, expected, (directory/'outer.log').read_text())
        rows = [json.loads(p.read_text()) for p in directory.glob('leaf-*.json')]
        self.assertFalse([r for r in rows if live(r)])
        self.assertEqual(competitor(lock), 0)
        self.assertTrue(all(r['lock_fds'] == [] for r in rows), 'model actual close_fds harness')
        events_path = directory/'events.jsonl'
        events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        if mode != 'wrong-inode':
            if mode != 'preflight-refuse':
                self.assertTrue(probes, 'test observed live descendants under lock')
            self.assertTrue(all(e['lock_held'] for e in events))
            if parked and mode != 'audit-fail':
                self.assertEqual(events[-1]['event'], 'FAILED_LOCK_RETAINED')
            elif not parked:
                self.assertEqual(events[-1]['event'], 'release_ready')
                self.assertTrue(any(e['event']=='lineage_quiescent' for e in events))
        if mode in ('timeout','harness-crash','signal','interrupt','driver-exception'):
            self.assertFalse((directory/'eval-started').exists())
            self.assertTrue(any(e['event']=='signal_owned' for e in events))
            self.assertGreaterEqual(len(rows), 2, 'child starts its own session and grandchild')
            observed = {e['identity']['pid'] for e in events if e['event']=='owned_identity'}
            self.assertTrue({r['pid'] for r in rows} <= observed)
        if mode in ('train-fail','wrong-inode','preflight-refuse'):
            self.assertFalse((directory/'eval-started').exists())
        if mode == 'eval-fail':
            self.assertTrue((directory/'eval-started').exists())
        receipt = {'case':mode, 'outer_rc':result, 'expected_rc':expected,
                   'competitors_during_live_descendants':probes, 'leaf_identities':rows,
                   'all_owned_exited':True, 'lock_available_after_terminal':True,
                   'phase_gap_and_postflight_competitors':boundaries,
                   'park_disposal_only_after_quiescence':parked, 'author_test_only':True}
        (directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'case':mode,'outer_rc':result,'live_competitor_checks':len(probes),
                          'boundary_competitors':boundaries,'parked':parked}),flush=True)

    def test_00_original_timeout_defect(self):
        directory = OUTPUT/'old-timeout'
        directory.mkdir()
        (directory/'case.txt').write_text('old-timeout')
        lock = directory/'device0.lock'
        lock.touch()
        process = subprocess.Popen(['/usr/bin/flock','-n',str(lock),'/usr/bin/python3','-I','-B',str(HERE/'cpu_lifetime_fixture.py')],cwd=directory)
        self.assertEqual(process.wait(timeout=3),124)
        rows = [json.loads(p.read_text()) for p in directory.glob('leaf-*.json')]
        self.assertTrue(any(live(r) for r in rows))
        self.assertTrue(all(not r['lock_fds'] for r in rows))
        self.assertEqual(competitor(lock),0)
        receipt = {'old_defect_reproduced':True,'outer_rc':124,'child_alive_after_outer_exit':True,
                   'child_has_original_lock_fd':False,'competitor_rc':0}
        deadline = time.monotonic()+3
        while any(live(r) for r in rows) and time.monotonic()<deadline:
            time.sleep(0.02)
        self.assertFalse(any(live(r) for r in rows))
        while True:
            try:
                pid, _ = os.waitpid(-1,os.WNOHANG)
                if not pid:
                    break
            except ChildProcessError:
                break
        (directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)

    def test_normal_train_then_eval(self): self.run_case('normal',0)
    def test_timeout_new_session_grandchild(self):
        unrelated=subprocess.Popen(['/usr/bin/python3','-c','import time; time.sleep(2)'],start_new_session=True)
        try:
            self.run_case('timeout',124)
            self.assertIsNone(unrelated.poll(),'must never signal an unrelated process')
        finally:
            self.assertEqual(unrelated.wait(timeout=3),0)
    def test_harness_crash_new_session_grandchild(self): self.run_case('harness-crash',137)
    def test_signal_new_session_grandchild(self): self.run_case('signal',143)
    def test_interrupt_new_session_grandchild(self): self.run_case('interrupt',130)
    def test_initial_preflight_refusal_releases_without_job(self): self.run_case('preflight-refuse',1)
    def test_driver_exception(self): self.run_case('driver-exception',1)
    def test_train_failure(self): self.run_case('train-fail',7)
    def test_eval_failure(self): self.run_case('eval-fail',9)
    def test_postflight_failure_retains_lock(self): self.run_case('postflight-fail',137,parked=True)
    def test_unknown_quiescence_retains_lock(self): self.run_case('quiescence-unknown',137,parked=True)
    def test_failed_audit_write_retains_lock_after_cleanup(self): self.run_case('audit-fail',137,parked=True)
    def test_wrong_original_inode_refuses(self): self.run_case('wrong-inode',1)


if __name__ == '__main__':
    print('AUTHOR_CPU_ARTIFACTS='+str(OUTPUT),flush=True)
    suite=unittest.main(verbosity=2,exit=False)
    evidence={'author_test_only':True,'optimized_test_program':bool(sys.flags.optimize),
              'fixture_driver_optimized':False,'successful':suite.result.wasSuccessful(),
              'tests_run':suite.result.testsRun,'artifact_directory':str(OUTPUT),
              'input_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                              for name in ('lifetime_guard.py','cpu_lifetime_fixture.py','test_lifetime.py')},
              'artifacts':{str(p.relative_to(OUTPUT)):p.read_text() for p in sorted(OUTPUT.rglob('*'))
                           if p.is_file() and p.name!='device0.lock'}}
    (OUTPUT.parent/(OUTPUT.name+'-evidence.json')).write_text(json.dumps(evidence,indent=2)+'\n')
    sys.exit(0 if suite.result.wasSuccessful() else 1)
