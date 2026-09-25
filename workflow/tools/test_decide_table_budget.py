"""Narrow regression tests for the human-authorized table step-window waiver."""
import importlib.util
from pathlib import Path
import unittest

_spec = importlib.util.spec_from_file_location("table_budget_decide", Path(__file__).with_name("decide.py"))
decide = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(decide)


def row(label, host="trn21", *, kind="control", bpb=0.9915, steps=3650, **extra):
    return dict(label=label, host=host, kind=kind, arm_id="A1", base_sha256="c5c",
                timestamp_utc="2026-09-25T16:00:00Z", status="evaluated",
                over_budget=False, causality_passed=True, git_dirty=False,
                eval_tokens=20971520, eval_seq_len=1024, public_20m=bpb,
                steps=steps, **extra)


class TableBudgetTests(unittest.TestCase):
    def verdict(self, **overrides):
        candidate = row("a1", kind="table_arm", bpb=0.988, steps=3000,
                        throughput_exempt=True, training_budget_seconds=1800)
        candidate.update(overrides)
        return decide.decide_arm([row("control"), candidate], "A1")

    def test_authorized_table_cost_does_not_confound(self):
        self.assertEqual(self.verdict()["verdict"], "NOMINATE")

    def test_exemption_is_explicit_and_table_only(self):
        for change in ({"throughput_exempt": False}, {"throughput_exempt": "true"},
                       {"kind": "flag_arm"}, {"training_budget_seconds": 120}):
            with self.subTest(change=change):
                self.assertEqual(self.verdict(**change)["verdict"], "CONFOUNDED")

    def test_existing_throughput_arm_remains_exempt(self):
        self.assertEqual(self.verdict(kind="throughput_arm")["verdict"], "NOMINATE")

    def test_budget_and_correctness_failures_still_invalid(self):
        for change in ({"over_budget": True}, {"causality_passed": False},
                       {"eval_tokens": 2097152}, {"git_dirty": True}):
            with self.subTest(change=change):
                self.assertEqual(self.verdict(**change)["verdict"], "INSUFFICIENT")

    def test_high_prior_and_big_win_rules_unchanged(self):
        self.assertEqual(self.verdict(public_20m=0.9913)["verdict"], "REPLICATE")
        self.assertEqual(self.verdict(public_20m=0.974)["verdict"], "HOLD")

    def test_two_host_adoption_and_per_host_exemption(self):
        rows = [row("c21"), row("c22", host="trn22")]
        for host in ("trn21", "trn22"):
            rows.append(row("a-"+host, host=host, kind="table_arm", bpb=0.988,
                            steps=3000, throughput_exempt=True, training_budget_seconds=1800))
        self.assertEqual(decide.decide_arm(rows, "A1")["verdict"], "ADOPT_PENDING_HUMAN_ACK")
        rows[-1]["throughput_exempt"] = False
        self.assertEqual(decide.decide_arm(rows, "A1")["verdict"], "CONFOUNDED")


if __name__ == "__main__":
    unittest.main()
