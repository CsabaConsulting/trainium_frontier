#!/usr/bin/env python3
"""Local negative proof for an early launcher failure; no host is contacted."""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fresh_start_launch as launcher


class EarlyLauncherReceiptIndependentTest(unittest.TestCase):
    def test_missing_checkout_has_durable_no_spawn_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            no_spawn = root / "no-spawn.json"
            request = {
                "remote": str(root / "missing-checkout"),
                "label": "fs-synthetic-early-launcher-failure",
                "expected_commit": "0" * 40,
                "monitor_interlock": {
                    "pair_binding_path": str(root / "pair-binding.json"),
                    "no_spawn_path": str(no_spawn),
                },
            }
            with (patch.object(sys, "stdin", io.StringIO(json.dumps(request))),
                  patch.object(subprocess, "Popen",
                               side_effect=AssertionError("guard spawn reached"))):
                with self.assertRaisesRegex(AssertionError, "checkout missing"):
                    exec(compile(launcher.REMOTE_SCRIPT, "<local-remote-script>", "exec"),
                         {"__name__": "__main__"})
            self.assertTrue(no_spawn.exists(),
                            "a pre-try launcher refusal has no durable no-spawn proof")


if __name__ == "__main__":
    unittest.main()
