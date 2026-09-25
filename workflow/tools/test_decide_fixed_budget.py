"""General fixed-budget selector regression tests; stdlib only, no host access."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

_spec = importlib.util.spec_from_file_location("fixed_budget_decide", Path(__file__).with_name("decide.py"))
decide = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(decide)


def row(label, host="trn21", *, kind="control", bpb=0.9915, steps=3650, **extra):
    result = dict(label=label, host=host, kind=kind, arm_id="A1", base_sha256="c5c",
                  timestamp_utc="2026-09-25T16:00:00Z", status="evaluated",
                  training_budget_seconds=1800, over_budget=False, causality_passed=True,
                  git_dirty=False, eval_tokens=20971520, eval_seq_len=1024,
                  public_20m=bpb, steps=steps, training_seconds=1784.8)
    result.update(extra)
    return result


class FixedBudgetTests(unittest.TestCase):
    def result(self, **extra):
        candidate = row("arm", kind="flag_arm", bpb=.9900, steps=3000)
        candidate.update(extra)
        return decide.decide_arm([row("control"), candidate], "A1")

    def test_rules_version_and_numeric_thresholds(self):
        self.assertEqual(decide.RULES_VERSION, "v2-2026-09-25-general-fixed-budget")
        self.assertEqual((decide.NOMINATE, decide.DISCARD_SINGLE, decide.ADOPT_MEAN,
                          decide.ADOPT_DISAGREE, decide.DISCARD_MEAN, decide.HOLD_DELTA,
                          decide.HOLD_ABS, decide.DRIFT),
                         (-.0006, .0005, -.0010, .0015, -.0003, -.010, .975, .0010))
        self.assertFalse(hasattr(decide, "STEP_WINDOW"))
        self.assertFalse(hasattr(decide, "step_window_exempt"))

    def test_all_arms_ignore_step_window_and_exemption_metadata(self):
        for kind in ("architecture_arm", "throughput_arm", "schedule_arm", "table_arm", "flag_arm", "data_order_arm"):
            for steps in (0, 500, 3000, 6000):
                for exemption in (None, False, True, "true"):
                    with self.subTest(kind=kind, steps=steps, exemption=exemption):
                        result = self.result(kind=kind, steps=steps, throughput_exempt=exemption)
                        self.assertEqual(result["verdict"], "NOMINATE")
                        self.assertNotIn("step_window_exempt_hosts", result)

    def test_sixty_percent_alarm_uses_unrounded_ratio_and_does_not_gate(self):
        for steps, alarm in ((2190, False), (2189.999, True), (500, True)):
            with self.subTest(steps=steps):
                result = self.result(steps=steps)
                self.assertEqual(result["verdict"], "NOMINATE")
                codes = result["run_diagnostics"]["arm"]["diagnostic_alarms"]
                self.assertEqual("steps_below_60_percent_of_same_host_control" in codes, alarm)

    def test_missing_step_metadata_reports_without_gating(self):
        for steps in (None, "unknown", float("nan"), -1):
            with self.subTest(steps=steps):
                result = self.result(steps=steps)
                self.assertEqual(result["verdict"], "NOMINATE")
                self.assertIsNone(result["per_host"][0]["step_ratio"])
                self.assertIn("step_count_missing_or_invalid", result["run_diagnostics"]["arm"]["diagnostic_alarms"])
        result = decide.decide_arm([row("control", steps=None), row("arm", kind="flag_arm", bpb=.99)], "A1")
        self.assertEqual(result["verdict"], "NOMINATE")
        self.assertIsNone(result["per_host"][0]["step_ratio"])

    def test_recorded_spikes_and_stalls_alarm_only(self):
        for extra in ({"timing_diagnostics": {"step_time_spikes": [{"step": 80}], "stalls": [{"seconds": 10}]}},
                      {"step_time_spike_detected": True, "stall_detected": True}):
            result = self.result(**extra)
            self.assertEqual(result["verdict"], "NOMINATE")
            codes = result["run_diagnostics"]["arm"]["diagnostic_alarms"]
            self.assertIn("recorded_step_time_spikes", codes)
            self.assertIn("recorded_stalls", codes)

    def test_explicit_budget_sources_and_reserve(self):
        for declarations in ({"training_budget_seconds": 1800},
                             {"env": {"NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS": "1800"}},
                             {"argv": ["--max-train-seconds", "1800"]},
                             {"argv": ["--max-train-seconds=1800"]},
                             {"training_budget_seconds": 1800., "env": {"NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS": "1800"}, "argv": ["--max-train-seconds", "1800"]}):
            with self.subTest(declarations=declarations):
                r = row("arm", kind="flag_arm", bpb=.99, training_seconds=1780.1)
                del r["training_budget_seconds"]
                r.update(declarations)
                self.assertIsNone(decide.invalid_reason(r))
        # Actual optimizer duration is not required to equal the budget.
        self.assertEqual(self.result(training_seconds=1780.1)["verdict"], "NOMINATE")

    def test_missing_non1800_and_malformed_budgets_rejected(self):
        for value in (None, True, 120, 1799, 1801, "", "oops", "nan", "inf", float("nan"), [], {}):
            with self.subTest(value=value):
                self.assertIsNotNone(decide.invalid_reason(row("arm", training_budget_seconds=value)))
        r = row("arm"); del r["training_budget_seconds"]
        self.assertIn("missing explicit", decide.invalid_reason(r))
        r["argv"] = ["--max-train-seconds"]
        self.assertIn("missing --max", decide.invalid_reason(r))
        r["argv"] = ["--max-train-seconds="]
        self.assertIn("invalid training budget", decide.invalid_reason(r))

    def test_conflicting_declarations_fail_even_if_cli_would_override(self):
        for extra in ({"env": {"NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS": "120"}},
                      {"argv": ["--max-train-seconds", "120"]},
                      {"argv": ["--max-train-seconds=1800", "--max-train-seconds=120"]}):
            with self.subTest(extra=extra):
                self.assertIn("conflicting", decide.invalid_reason(row("arm", **extra)))

    def test_wrong_budget_control_not_pooled(self):
        result = decide.decide_arm([row("control", training_budget_seconds=120), row("arm", kind="flag_arm", bpb=.99)], "A1")
        self.assertEqual(result["verdict"], "INSUFFICIENT")
        self.assertIn("no valid same-host control", result["why"])

    def test_explicit_charged_time_cannot_contradict_over_budget_false(self):
        for field in ("charged_through_exit_seconds", "harness_charged_through_exit_s", "charged_seconds_est"):
            for value in (1800.0001, -1, float("nan"), float("inf"), "1800", True, [], {}):
                with self.subTest(field=field, value=value):
                    result = self.result(**{field: value})
                    self.assertEqual(result["verdict"], "INSUFFICIENT")
                    self.assertIn("charged-time evidence", result["invalid_runs"]["arm"])
            for value in (None, 0, 1784.8, 1800):
                with self.subTest(field=field, value=value):
                    self.assertEqual(self.result(**{field: value})["verdict"], "NOMINATE")
        # Neither a null preferred field nor a lower measured value hides another contradiction.
        self.assertEqual(self.result(charged_through_exit_seconds=None,
                                     harness_charged_through_exit_s=1801)["verdict"], "INSUFFICIENT")
        self.assertEqual(self.result(charged_through_exit_seconds=1790,
                                     charged_seconds_est=1801)["verdict"], "INSUFFICIENT")

    def test_correctness_and_incomplete_status_still_invalid(self):
        for change in ({"over_budget": True}, {"over_budget": None}, {"causality_passed": False},
                       {"eval_tokens": 2097152}, {"eval_seq_len": 2048}, {"git_dirty": True},
                       {"public_20m": float("nan")}, {"status": "failed"},
                       {"status": "startup_truncated"}, {"status": "human_aborted_not_quality"}):
            with self.subTest(change=change):
                self.assertEqual(self.result(**change)["verdict"], "INSUFFICIENT")

    def test_smoke_or_timing_cannot_be_quality_with_1800_declaration(self):
        for kind in ("diagnostic", "smoke", "timing", "compile_smoke", "native_smoke"):
            self.assertEqual(self.result(kind=kind)["verdict"], "INSUFFICIENT")
        for purpose in ("timing", "smoke", "compile", None):
            self.assertEqual(self.result(run_purpose=purpose)["verdict"], "INSUFFICIENT")
        self.assertEqual(self.result(run_purpose="quality")["verdict"], "NOMINATE")

    def test_high_prior_big_win_and_two_host_rules_unchanged(self):
        self.assertEqual(self.result(public_20m=.9913, kind="table_arm")["verdict"], "REPLICATE")
        self.assertEqual(self.result(public_20m=.9913, kind="flag_arm")["verdict"], "DISCARD")
        self.assertEqual(self.result(public_20m=.9925)["verdict"], "DISCARD")
        self.assertEqual(self.result(public_20m=.9800, steps=10)["verdict"], "HOLD")
        self.assertEqual(self.result(public_20m=.974)["verdict"], "HOLD")
        rows = [row("c21"), row("c22", host="trn22"),
                row("a21", kind="architecture_arm", bpb=.9903, steps=100),
                row("a22", host="trn22", kind="architecture_arm", bpb=.9903, steps=6000)]
        self.assertEqual(decide.decide_arm(rows, "A1")["verdict"], "ADOPT_PENDING_HUMAN_ACK")
        rows[-1]["public_20m"] = .9913; rows[-2]["public_20m"] = .9885
        self.assertEqual(decide.decide_arm(rows, "A1")["verdict"], "REPLICATE")
        rows[-1]["public_20m"] = .9914; rows[-2]["public_20m"] = .9914
        self.assertEqual(decide.decide_arm(rows, "A1")["verdict"], "DISCARD")

    def test_startup_unknown_not_zero_and_cold_gate_not_bpb_gate(self):
        result = self.result(startup_s=900, startup_allowance_s=900, startup_allowance_est=817.9)
        diagnostic = result["run_diagnostics"]["arm"]
        self.assertEqual(result["verdict"], "NOMINATE")
        self.assertIsNone(diagnostic["startup_measured_seconds"])
        self.assertEqual(diagnostic["legacy_startup_s_untyped"], 900)
        self.assertEqual(diagnostic["startup_allowance_source_estimate_seconds"], 817.9)
        self.assertEqual(diagnostic["startup_allowance_seconds"], 900)
        self.assertEqual(diagnostic["cold_startup_submission_gate"], "HOLD_MISSING_PROVEN_COLD_STARTUP")
        result = self.result(startup_measured_seconds=1000, startup_allowance_seconds=900,
                             charged_seconds_est=1784.9, cold_startup_seconds=1000,
                             cold_startup_proven=True, cold_startup_evidence="reviewed-receipt.json")
        diagnostic = result["run_diagnostics"]["arm"]
        self.assertEqual(result["verdict"], "NOMINATE")
        self.assertIn("startup_above_870_second_planning_warning", diagnostic["diagnostic_alarms"])
        self.assertEqual(diagnostic["cold_startup_submission_gate"], "EVIDENCE_PRESENT_NOT_SUBMISSION_APPROVAL")
        self.assertEqual(diagnostic["charged_source_estimate_seconds"], 1784.9)
        # A warm-directory claim or untyped legacy value is not proven cold.
        result = self.result(env={"cache_state": "cold"}, cold_startup_seconds=100)
        self.assertIn("cold_startup_unproven_submission_gate", result["run_diagnostics"]["arm"]["diagnostic_alarms"])

    def test_control_drift_threshold_unchanged(self):
        rows = [row("c1"), row("c2", bpb=.993, timestamp_utc="2026-09-25T17:00:00Z")]
        result = decide.drift(rows)
        self.assertEqual(len(result["drift_alarms"]), 1)
        self.assertEqual(result["drift_alarms"][0]["diff"], .0015)

    def test_current_ledger_a0_a1_a2_and_drift_regression_without_rewriting(self):
        path = Path(__file__).resolve().parents[2] / "research/v2/ledger.jsonl"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = decide.load(str(path))
        controls = [r for r in rows if r.get("kind") == "control" and r.get("status") == "evaluated"]
        self.assertEqual(len(controls), 2)
        for control in controls:
            self.assertNotIn("training_budget_seconds", control)
            self.assertEqual(control["env"]["NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS"], "1800")
            self.assertIsNone(decide.invalid_reason(control))
        self.assertEqual(decide.drift(rows)["drift_alarms"], [])
        a1, a2 = decide.decide_arm(rows, "A1"), decide.decide_arm(rows, "A2")
        self.assertEqual(a1["verdict"], "DISCARD")
        self.assertEqual(a1["per_host"][0]["delta"], .0071572)
        self.assertEqual(a2["verdict"], "INSUFFICIENT")
        self.assertEqual(set(a2["invalid_runs"].values()), {"status=startup_truncated", "status=human_aborted_not_quality"})
        self.assertTrue(all("steps_below_60_percent_of_same_host_control" in d["diagnostic_alarms"] for d in a2["run_diagnostics"].values()))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
