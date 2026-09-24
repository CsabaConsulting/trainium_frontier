"""Local fake-guard contract checks; no Trainium process or source is used."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


OBSERVER = Path(__file__).resolve().parents[1] / "a0_boundary_observer.py"
LABEL = "fs-fixture-a0-r1"
GUARD = r'''
import json, os, pathlib, subprocess, sys, time
from datetime import datetime, timezone
root, label, mode = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
logs = root / 'research' / 'run-logs'
step = logs / (label + '.steps.jsonl')
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)'], start_new_session=True)
def utc(): return datetime.now(timezone.utc).isoformat()
def put(path, value): path.write_text(json.dumps(value))
def boundary(value):
    with step.open('a') as fh:
        fh.write(str(value) + '\n')
        fh.flush()
        os.fsync(fh.fileno())
try:
    put(logs / (label + '.process.json'), dict(label=label, guard_pid=os.getpid(),
        child_pid=child.pid, child_session=child.pid, started_at=utc(),
        command=['env', 'NEURON_COMPETITION_R1_STEP_LOG=' + str(step), 'torchrun']))
    if mode == 'late_first':
        boundary(0)
        (root / 'first-written').touch()
        time.sleep(.3)
        values = [1, 2, 3, 4]
    else:
        time.sleep(.2)
        values = [0, 1, 2, 3, 4] if mode != 'duplicate' else [0, 1, 1, 3, 4]
    if mode == 'burst':
        with step.open('a') as fh:
            fh.write('0\n1\n')
            fh.flush()
            os.fsync(fh.fileno())
        values = [2, 3, 4]
    if mode == 'truncate':
        boundary(0)
        time.sleep(.09)
        boundary(1)
        time.sleep(.09)
        step.write_text('0\n')
        values = [2, 3, 4]
    for value in values:
        boundary(value)
        time.sleep(.09)
    if mode == 'partial_terminal':
        with step.open('a') as fh:
            fh.write('5')
            fh.flush()
            os.fsync(fh.fileno())
    child.terminate()
    child.wait(timeout=2)
    time.sleep(.15)
    put(logs / (label + '.postflight.json'), dict(protocol_version=2, label=label,
        remote=str(root), stage='postflight', command_returncode=0, guard_error=None,
        host_clean=True, failures=[], cleanup={'remaining': []}, completed_at=utc()))
finally:
    if child.poll() is None:
        child.terminate()
        child.wait(timeout=2)
'''


class BoundaryObserverFixture(unittest.TestCase):
    def run_case(self, mode, wrong_guard=False):
        with tempfile.TemporaryDirectory(prefix="a0-observer-fixture-") as temp:
            root = Path(temp)
            logs = root / "research" / "run-logs"
            logs.mkdir(parents=True)
            guard = root / "research" / "fresh_start_guard.py"
            guard.write_text(GUARD)
            running = subprocess.Popen([sys.executable, str(guard), str(root), LABEL, mode])
            try:
                if mode == "late_first":
                    deadline = time.monotonic() + 3
                    while not (root / "first-written").exists() and time.monotonic() < deadline:
                        time.sleep(.005)
                    self.assertTrue((root / "first-written").exists())
                argv = [sys.executable, str(OBSERVER), "--label", LABEL,
                        "--process-json", str(logs / (LABEL + ".process.json")),
                        "--boundary-file", str(logs / (LABEL + ".steps.jsonl")),
                        "--postflight-json", str(logs / (LABEL + ".postflight.json")),
                        "--guard-script", str(OBSERVER if wrong_guard else guard),
                        "--poll-seconds", "0.005", "--timeout-seconds", "4"]
                completed = subprocess.run(argv, text=True, capture_output=True, timeout=6)
                running.wait(timeout=3)
                return completed.returncode, json.loads(completed.stdout)
            finally:
                if running.poll() is None:
                    running.terminate()
                    running.wait(timeout=3)

    def test_complete_sequential_fixture(self):
        code, receipt = self.run_case("normal")
        self.assertEqual(code, 0)
        self.assertTrue(receipt["scored_timing_proof"])
        self.assertEqual([x["value"] for x in receipt["boundaries"]], list(range(5)))
        self.assertGreater(receipt["timing"]["launch_to_boundary5_seconds"], 0)
        self.assertEqual(receipt["invalid_reasons"], [])

    def test_first_line_preexists_observer(self):
        code, receipt = self.run_case("late_first")
        self.assertEqual(code, 2)
        self.assertFalse(receipt["scored_timing_proof"])
        self.assertIn("boundary_preexisted_observer", receipt["invalid_reasons"])

    def test_duplicate_is_non_scored(self):
        code, receipt = self.run_case("duplicate")
        self.assertEqual(code, 2)
        self.assertIn("boundary_gap_or_duplicate", receipt["invalid_reasons"])

    def test_wrong_live_guard_is_non_scored(self):
        code, receipt = self.run_case("normal", wrong_guard=True)
        self.assertEqual(code, 2)
        self.assertTrue(any("live guard script mismatch" in x for x in receipt["invalid_reasons"]))

    def test_multiple_boundaries_per_poll_is_non_scored(self):
        code, receipt = self.run_case("burst")
        self.assertEqual(code, 2)
        self.assertIn("observation_gap_multiple_boundaries_in_one_poll", receipt["invalid_reasons"])

    def test_truncation_is_non_scored(self):
        code, receipt = self.run_case("truncate")
        self.assertEqual(code, 2)
        self.assertIn("boundary_file_truncated_or_rewritten", receipt["invalid_reasons"])

    def test_terminal_partial_line_is_non_scored(self):
        code, receipt = self.run_case("partial_terminal")
        self.assertEqual(code, 2)
        self.assertIn("boundary_file_has_terminal_partial_line", receipt["invalid_reasons"])


if __name__ == "__main__":
    unittest.main()
