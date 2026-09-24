#!/usr/bin/env python3
"""Static contract tests for the non-launching trn22 CPU G2 operator job."""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "research/v2/run_g2_baseline_s0_trn22_r1.sh"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class G2OperatorJobContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text()

    def test_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_exact_protected_hashes(self) -> None:
        expected = {
            "prepare.py": "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb",
            "workflow/tools/contained.py": "0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d",
            "workflow/tools/preflight_dryrun.py": "6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8",
            "04-train-research-35c9.py": "35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2",
            "submissions/research_v2/train.py": "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586",
            "05-train-submission-clean-116188.py": "116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d",
            "submissions/candidates/S0/train.py": "ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb",
        }
        for rel, expected_sha in expected.items():
            self.assertEqual(sha256(ROOT / rel), expected_sha, rel)
            self.assertIn(expected_sha, self.text)
    def test_exact_four_labels_and_one_probe_each(self) -> None:
        labels = [
            "fs-20260925-g2-a0-35c9-trn22-r1",
            "fs-20260925-g2-a0-research-v2-trn22-r1",
            "fs-20260925-g2-s0-116188-trn22-r1",
            "fs-20260925-g2-s0-bare-trn22-r1",
        ]
        for label in labels:
            self.assertEqual(self.text.count(f'="{label}"'), 1)
        self.assertEqual(self.text.count("\nrun_probe \"$LABEL_"), 4)

    def test_parent_holds_canonical_lock_through_quiescence(self) -> None:
        lock = 'HOST_LOCK="/home/ubuntu/.cache/trainium_freshstart/device0.lock"'
        self.assertIn(lock, self.text)
        self.assertIn('[[ -f "$HOST_LOCK" && ! -L "$HOST_LOCK" ]]', self.text)
        acquire = self.text.index('/usr/bin/flock --exclusive --nonblock 9')
        first_probe = self.text.index('run_probe "$LABEL_35C9"')
        last_probe = self.text.index('run_probe "$LABEL_S0"')
        receipt = self.text.index('readonly BATCH_RECEIPT=')
        release = self.text.index('trap - EXIT INT TERM HUP', receipt)
        self.assertLess(acquire, first_probe)
        self.assertLess(first_probe, last_probe)
        self.assertLess(last_probe, receipt)
        self.assertLess(receipt, release)
        self.assertIn("stop_and_quiesce_active", self.text)
        self.assertIn("tagged descendants survived the leader", self.text)

    def test_exact_science_argv_and_three_comparisons(self) -> None:
        dense = """--n-embd 1024
  --no-ngram-ve
  --no-qk-shift
  --compile-sdpa-direct
  --nki-local-conv
  --bf16-norm-output
  --pack-factor 32
  --clip-after-reduce
  --seq-len 1024
  --no-eval-public"""
        self.assertEqual(self.text.count(dense), 2)
        self.assertEqual(self.text.count("--matrix-lr 0.015"), 1)
        self.assertIn('run_probe "$LABEL_S0" "$TRAIN_S0" "$SHA_S0"\n', self.text)
        self.assertEqual(self.text.count("compare_receipts \""), 3)
        self.assertIn('"35c9-vs-research-v2-default-off"', self.text)
        self.assertIn('"116188-vs-s0"', self.text)
        self.assertIn('"35c9-dense-flags-vs-116188"', self.text)

    def test_cpu_synthetic_only_and_no_destructive_or_remote_actions(self) -> None:
        self.assertIn("--steps 3 --nproc 4 --timeout 3600", self.text)
        self.assertIn("--out \"$receipt\" --keep", self.text)
        for forbidden in (
            "--real-data",
            "torchrun ",
            " prepare.py eval-public",
            "ssh ",
            "scp ",
            "rm ",
            "rmdir ",
            "git clean",
            "git reset",
            "NEURON_LOGICAL_NC_CONFIG=2",
        ):
            self.assertNotIn(forbidden, self.text)

    def test_unique_absent_paths_and_fail_closed_order(self) -> None:
        self.assertIn('[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]', self.text)
        self.assertIn('[[ ! -e "$dir" && ! -L "$dir" ]]', self.text)
        self.assertIn('[[ ! -e "$receipt" && ! -L "$receipt" ]]', self.text)
        self.assertIn('[[ ! -e "$tmpdir" && ! -L "$tmpdir" ]]', self.text)
        calls = [
            self.text.index('run_probe "$LABEL_35C9"'),
            self.text.index('run_probe "$LABEL_RESEARCH_V2"'),
            self.text.index('run_probe "$LABEL_116188"'),
            self.text.index('run_probe "$LABEL_S0"'),
        ]
        self.assertEqual(calls, sorted(calls))
        self.assertIn("set -Eeuo pipefail", self.text)


if __name__ == "__main__":
    unittest.main()
