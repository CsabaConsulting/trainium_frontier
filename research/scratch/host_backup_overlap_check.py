#!/usr/bin/env python3
"""Regression checks for the observed backup/preflight process collision."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fresh_start_backup import REMOTE_DISCOVERY
from fresh_start_host import health_failures, known_read_only_backup


class BackupOverlap(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.receipt = json.loads((root / "research/run-logs" /
            "fs-w51-branch-norm-rsqrt-gate-r1.preflight.json").read_text())
        self.backup = next(p for p in self.receipt["relevant_processes"] if p["pid"] == 1406049)

    def record_with(self, argv):
        record = copy.deepcopy(self.receipt)
        process = dict(self.backup, argv=argv)
        record["relevant_processes"] = [process]
        return record

    def test_actual_failed_preflight_was_read_only_inventory(self):
        self.assertTrue(known_read_only_backup(self.backup["argv"]))
        record = self.record_with(self.backup["argv"])
        self.assertEqual(health_failures(record), [])

    def test_changed_inline_source_is_not_exempt(self):
        argv = ["python3", "-c", REMOTE_DISCOVERY + "\nimport train\n"]
        self.assertFalse(known_read_only_backup(argv))
        self.assertTrue(any("processes remain" in f for f in health_failures(self.record_with(argv))))

    def test_training_and_compiler_commands_remain_blocked(self):
        for argv in (["python3", "train.py"], ["torchrun", "train.py"],
                     ["neuronx-cc", "compile", "model"],
                     ["python3", "research/scratch/phase_timer.py"]):
            with self.subTest(argv=argv):
                self.assertFalse(known_read_only_backup(argv))
                self.assertTrue(any("processes remain" in f for f in health_failures(self.record_with(argv))))

    def test_name_or_marker_alone_cannot_bypass_check(self):
        for argv in (["python3", "-c", "# read_only_backup\nimport train"],
                     ["python3", "-c", REMOTE_DISCOVERY, "train.py"],
                     ["unrecognized-program", "-c", REMOTE_DISCOVERY]):
            with self.subTest(argv=argv):
                self.assertFalse(known_read_only_backup(argv))

    def test_backup_classification_never_masks_device_holder(self):
        record = self.record_with(self.backup["argv"])
        record["device_holders"]["returncode"] = 0
        self.assertIn("device has holders, or fuser failed", health_failures(record))
        record["neuron_process_count"] = 1
        self.assertIn("neuron-ls failed or reports a live Neuron process", health_failures(record))


if __name__ == "__main__":
    unittest.main()
