#!/usr/bin/env python3
"""Local fake-file tests for a0_boundary_observer.py; no host or Neuron access."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


HERE = Path(__file__).resolve().parent
OBSERVER = HERE / "a0_boundary_observer.py"
LABEL = "fs-test-a0-boundary-observer-r1"


def durable_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("x") as handle:
        json.dump(value, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def append_boundary(path: Path, value: int) -> None:
    with path.open("ab") as handle:
        handle.write(f"{value}\n".encode())
        handle.flush()
        os.fsync(handle.fileno())


class ObserverTests(unittest.TestCase):
    def run_case(self, actions, timeout=3.0):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            process_path = root / "process.json"
            boundary_path = root / "steps.log"
            postflight_path = root / "postflight.json"
            cmd = [
                sys.executable, str(OBSERVER), "--label", LABEL,
                "--process-json", str(process_path), "--boundary-file", str(boundary_path),
                "--postflight-json", str(postflight_path), "--poll-seconds", "0.005",
                "--timeout-seconds", str(timeout),
            ]
            observed = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
            try:
                time.sleep(0.03)
                durable_json(process_path, {
                    "label": LABEL,
                    "guard_pid": os.getpid(),
                    "child_pid": child.pid,
                    "child_session": child.pid,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "command": ["fake-direct-a0"],
                })
                time.sleep(0.03)
                actions(boundary_path)
                child.terminate()
                child.wait(timeout=2)
                time.sleep(0.03)
                durable_json(postflight_path, {
                    "protocol_version": 2,
                    "label": LABEL,
                    "stage": "postflight",
                    "command_returncode": 0,
                    "guard_error": None,
                    "cleanup": {"remaining": []},
                    "failures": [],
                    "host_clean": True,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
                stdout, stderr = observed.communicate(timeout=3)
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=2)
                if observed.poll() is None:
                    observed.terminate()
                    observed.wait(timeout=2)
            self.assertEqual(stderr, "")
            return observed.returncode, json.loads(stdout)

    @staticmethod
    def spaced(values):
        def apply(path):
            for value in values:
                append_boundary(path, value)
                time.sleep(0.03)
        return apply

    def test_valid_five_boundaries(self):
        rc, result = self.run_case(self.spaced(range(5)))
        self.assertEqual(rc, 0)
        self.assertEqual(result["status"], "COMPLETE_SCORED_TIMING_PROOF")
        self.assertTrue(result["scored_timing_proof"])
        self.assertEqual([x["value"] for x in result["boundaries"]], list(range(5)))
        self.assertEqual(result["timing"]["boundary5_value"], 4)
        self.assertGreaterEqual(result["timing"]["launch_to_boundary5_seconds"], 0)
        self.assertGreaterEqual(result["timing"]["boundary5_to_child_terminal_observed_seconds"], 0)
        self.assertGreaterEqual(result["timing"]["boundary5_to_postflight_observed_seconds"], 0)

    def test_multi_boundary_poll_is_non_scored(self):
        def apply(path):
            with path.open("ab") as handle:
                handle.write(b"0\n1\n")
                handle.flush()
                os.fsync(handle.fileno())
            time.sleep(0.03)
            for value in range(2, 5):
                append_boundary(path, value)
                time.sleep(0.03)
        rc, result = self.run_case(apply)
        self.assertEqual(rc, 2)
        self.assertEqual(result["status"], "COMPLETE_NON_SCORED")
        self.assertIn("observation_gap_multiple_boundaries_in_one_poll", result["invalid_reasons"])

    def test_duplicate_is_non_scored(self):
        rc, result = self.run_case(self.spaced([0, 1, 1, 2, 3, 4]))
        self.assertEqual(rc, 2)
        self.assertIn("boundary_gap_or_duplicate", result["invalid_reasons"])

    def test_truncation_is_non_scored(self):
        def apply(path):
            append_boundary(path, 0)
            time.sleep(0.03)
            append_boundary(path, 1)
            time.sleep(0.03)
            with path.open("wb") as handle:
                handle.write(b"0\n")
                handle.flush()
                os.fsync(handle.fileno())
            time.sleep(0.03)
            for value in range(1, 5):
                append_boundary(path, value)
                time.sleep(0.03)
        rc, result = self.run_case(apply)
        self.assertEqual(rc, 2)
        self.assertIn("boundary_file_truncated_or_rewritten", result["invalid_reasons"])

    def test_timeout_without_postflight_is_non_scored(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
            process_path = root / "process.json"
            durable_json(process_path, {
                "label": LABEL, "guard_pid": os.getpid(), "child_pid": child.pid,
                "child_session": child.pid, "started_at": datetime.now(timezone.utc).isoformat(),
                "command": ["fake-direct-a0"],
            })
            try:
                done = subprocess.run([
                    sys.executable, str(OBSERVER), "--label", LABEL,
                    "--process-json", str(process_path),
                    "--boundary-file", str(root / "steps.log"),
                    "--postflight-json", str(root / "postflight.json"),
                    "--poll-seconds", "0.005", "--timeout-seconds", "0.1",
                ], text=True, capture_output=True, timeout=2)
            finally:
                child.terminate()
                child.wait(timeout=2)
            result = json.loads(done.stdout)
            self.assertEqual(done.returncode, 2)
            self.assertEqual(done.stderr, "")
            self.assertEqual(result["status"], "TIMEOUT_NON_SCORED")
            self.assertIn("observer_timeout_before_postflight", result["invalid_reasons"])


if __name__ == "__main__":
    unittest.main()
