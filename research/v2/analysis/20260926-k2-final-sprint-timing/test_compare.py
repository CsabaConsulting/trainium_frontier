"""Synthetic arithmetic/validation checks only; no candidate or native review."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import compare


def receipt(duration=10, count=45, startup=100):
    times = [startup + i * duration for i in range(count)]
    allowance = min(times[4], 900)
    wall = times[-1] + 10
    return {"mode": "train", "returncode": 0, "step_boundaries": count,
            "observations": [{"step": i, "observed_seconds": t} for i, t in enumerate(times)],
            "wall_seconds": wall, "startup_allowance_observed": allowance,
            "charged_through_process_exit": wall - allowance,
            "charged_through_process_exit_within_budget": True,
            "checkpoint_bytes": 123, "checkpoint_sha256": "a" * 64,
            "fresh_empty_sdk_caches": {"unproved": "unused"}}


class CompareTests(unittest.TestCase):
    def test_repaired_source_context_does_not_change_arithmetic(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ("control.json", "candidate.json")]
            for path, data in zip(paths, (receipt(12), receipt())):
                path.write_text(json.dumps(data))
            old = compare.compare(*paths)
            source = "a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71"
            repaired = compare.compare(*paths, source)
            self.assertEqual(repaired.pop("context_only_expected_candidate_source_sha256"), source)
            old.pop("context_only_expected_candidate_source_sha256")
            self.assertEqual(old, repaired)
            self.assertFalse(repaired["source_identity_verified_by_this_helper"])
            with self.assertRaises(ValueError):
                compare.compare(*paths, "unverified-short-hash")

    def run_pair(self, control=None, candidate=None):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ("control.json", "candidate.json")]
            for path, data in zip(paths, (control if control is not None else receipt(12),
                                         candidate if candidate is not None else receipt())):
                path.write_text(json.dumps(data))
            result = compare.compare(*paths)
            json.dumps(result, allow_nan=False)
            self.assertEqual(result["control"]["raw_input_sha256"],
                             hashlib.sha256(paths[0].read_bytes()).hexdigest())
            return result

    def test_normal_and_long_parent(self):
        result = self.run_pair(receipt(12, 105))
        self.assertTrue(result["receipts_valid"])
        self.assertTrue(result["timing_threshold_pass"])
        self.assertEqual(result["candidate_to_control_rate_ratio"], "1.2")
        self.assertEqual(result["candidate"]["first_five_uncapped_startup_seconds"], "140")
        self.assertEqual(result["native_correctness"], "NOT_ESTABLISHED")
        self.assertTrue(result["cold_cache_proof"].startswith("NOT_ESTABLISHED"))

    def test_exact_three_percent_and_just_below(self):
        # Modify only the final interval endpoint to avoid binary-float fixture noise.
        candidate = receipt()
        control = receipt()
        control["observations"][44]["observed_seconds"] += 10.5
        control["wall_seconds"] += 10.5
        control["charged_through_process_exit"] += 10.5
        exact = self.run_pair(control, candidate)
        self.assertEqual(exact["candidate_to_control_rate_ratio"], "1.03")
        self.assertTrue(exact["timing_threshold_pass"])
        control["observations"][44]["observed_seconds"] -= 0.000001
        self.assertFalse(self.run_pair(control, candidate)["timing_threshold_pass"])
        self.assertFalse(self.run_pair(receipt(), receipt())["timing_threshold_pass"])

    def test_startup_cap(self):
        result = self.run_pair(receipt(startup=1000), receipt(startup=1000))
        self.assertTrue(result["receipts_valid"])
        self.assertEqual(result["candidate"]["first_five_uncapped_startup_seconds"], "1040")
        self.assertEqual(result["candidate"]["capped_startup_allowance_seconds"], "900")
        self.assertEqual(result["candidate"]["startup_excess_seconds"], "140")

    def test_invalid_receipts(self):
        mutations = [
            lambda d: d["observations"].pop(),
            lambda d: d["observations"][12].update(step=11),
            lambda d: d["observations"][12].update(observed_seconds=float("nan")),
            lambda d: d["observations"][12].update(observed_seconds=float("inf")),
            lambda d: d["observations"][12].update(observed_seconds=0),
            lambda d: d.update(charged_through_process_exit_within_budget=False),
            lambda d: d.update(mode="eval"),
            lambda d: d.update(returncode=1),
            lambda d: d.update(step_boundaries=44),
            lambda d: d.update(checkpoint_bytes=0),
            lambda d: d.update(checkpoint_bytes=float("nan")),
            lambda d: d.update(checkpoint_sha256="missing"),
            lambda d: d.update(charged_through_process_exit=1801),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                candidate = copy.deepcopy(receipt())
                mutate(candidate)
                result = self.run_pair(candidate=candidate)
                self.assertFalse(result["receipts_valid"])
                self.assertIsNone(result["timing_threshold_pass"])

    def test_ties_and_zero_selected_interval(self):
        tied = receipt()
        tied["observations"][20]["observed_seconds"] = tied["observations"][19]["observed_seconds"]
        result = self.run_pair(candidate=tied)
        self.assertTrue(result["receipts_valid"])
        self.assertTrue(result["timing_diagnostic_review_required"])
        self.assertTrue(result["candidate"]["diagnostics"])
        for item in tied["observations"][9:]:
            item["observed_seconds"] = tied["observations"][9]["observed_seconds"]
        self.assertIsNone(self.run_pair(candidate=tied)["timing_threshold_pass"])

    def test_missing_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            report, interval = compare.inspect(path)
            self.assertFalse(report["receipt_valid"])
            self.assertIsNone(interval)
            path.write_text('{"mode":"train","mode":"eval"}')
            report, interval = compare.inspect(path)
            self.assertFalse(report["receipt_valid"])
            self.assertIn("duplicate JSON key", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
