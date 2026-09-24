"""Local fail-closed checks for the final-week control-plane acceleration patch."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoloop import config as C  # noqa: E402
from autoloop.controller import Controller, startup_cache_diagnostics  # noqa: E402
from autoloop.remote import Inventory  # noqa: E402
from autoloop.selection import (  # noqa: E402
    PairedEvaluation,
    SelectionPolicyError,
    assess_confirmation,
    assess_screen,
)
from autoloop.state import Store  # noqa: E402


BASE = "a" * 64
CANDIDATE = "b" * 64
T0 = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def enabled_policy() -> C.EvaluationPolicy:
    return C.EvaluationPolicy(
        enabled=True,
        current_base_sha256=BASE,
        current_control_labels_by_host={
            "trn21": "current-control-trn21",
            "trn22": "current-control-trn22",
        },
        active_sweep_ids=["lr-local-sweep-r1"],
    )


def paired(
    *, population="screen2m", tokens=C.SCREEN_EVAL_TOKENS, candidate="0.9995",
    control="1.0000", label="candidate-r1", candidate_host="trn21",
    control_host="trn21", sweep=None,
) -> PairedEvaluation:
    num_bytes = C.SCREEN_EVAL_BYTES if population == "screen2m" else C.CONFIRMATION_EVAL_BYTES
    return PairedEvaluation.from_dict({
        "schema_version": 1,
        "population": population,
        "context_length": 1024,
        "num_tokens": tokens,
        "num_bytes": num_bytes,
        "candidate_label": label,
        "candidate_sha256": CANDIDATE,
        "candidate_host": candidate_host,
        "candidate_bpb": candidate,
        "control_label": "current-control-" + control_host,
        "control_sha256": BASE,
        "control_host": control_host,
        "control_bpb": control,
        "active_sweep_id": sweep,
    })


class SelectionPolicyTests(unittest.TestCase):
    def test_policy_is_inactive_until_separate_canonical_rebase(self):
        cfg = C.load_config(Path(__file__).resolve().parents[1] / "autoloop.json")
        self.assertFalse(cfg.evaluation_policy.enabled)
        self.assertEqual(cfg.evaluation_policy.current_base_sha256, "")
        with self.assertRaises(SelectionPolicyError):
            assess_screen(paired(), cfg.evaluation_policy)

    def test_screen_boundaries_and_middle_band_never_adopt(self):
        policy = enabled_policy()
        self.assertEqual(assess_screen(paired(candidate="0.9995"), policy).decision,
                         "NOMINATE_CONFIRMATION")
        self.assertEqual(assess_screen(paired(candidate="1.0005"), policy).decision, "DISCARD")
        self.assertEqual(assess_screen(paired(candidate="0.9998"), policy).decision, "DISCARD")
        swept = assess_screen(paired(candidate="0.9998", sweep="lr-local-sweep-r1"), policy)
        self.assertEqual(swept.decision, "NOMINATE_CONFIRMATION")
        self.assertNotEqual(swept.decision, "ADOPT")
        forged = assess_screen(paired(candidate="0.9998", sweep="unapproved-sweep"), policy)
        self.assertEqual(forged.decision, "DISCARD")

    def test_adoption_requires_two_negative_full20m_runs_at_decimal_boundary(self):
        policy = enabled_policy()
        records = [
            paired(population="full20m", tokens=C.CONFIRMATION_EVAL_TOKENS,
                   candidate="0.9991", label="candidate-full-r1"),
            paired(population="full20m", tokens=C.CONFIRMATION_EVAL_TOKENS,
                   candidate="0.9993", label="candidate-full-r2"),
        ]
        decision = assess_confirmation(records, policy)
        self.assertEqual(decision.decision, "ADOPT")
        self.assertEqual(decision.mean_delta, "-0.0008")
        with self.assertRaises(SelectionPolicyError):
            assess_confirmation(records[:1], policy)

    def test_confirmation_rejects_nonnegative_wrong_population_and_cross_host(self):
        policy = enabled_policy()
        nonnegative = [
            paired(population="full20m", tokens=C.CONFIRMATION_EVAL_TOKENS,
                   candidate="1.0000", label="candidate-full-r1"),
            paired(population="full20m", tokens=C.CONFIRMATION_EVAL_TOKENS,
                   candidate="0.9980", label="candidate-full-r2"),
        ]
        self.assertEqual(assess_confirmation(nonnegative, policy).decision, "DISCARD")
        with self.assertRaises(SelectionPolicyError):
            assess_screen(paired(tokens=123), policy)
        cross_host = paired(candidate_host="trn21", control_host="trn22")
        with self.assertRaises(SelectionPolicyError):
            assess_screen(cross_host, policy)

    def test_schema_rejects_extra_fields_and_policy_tamper(self):
        data = paired().to_dict()
        data["verdict"] = "ADOPT"
        with self.assertRaises(SelectionPolicyError):
            PairedEvaluation.from_dict(data)
        tampered = enabled_policy()
        tampered.screen_nominate_delta = "-0.0001"
        with self.assertRaises(SelectionPolicyError):
            assess_screen(paired(), tampered)


class DiagnosticsAndUtilizationTests(unittest.TestCase):
    def test_startup_and_cache_diagnostics_use_existing_fields_only(self):
        env = {}
        for index, key in enumerate(C.CACHE_ENV_KEYS):
            env[key] = f"/tmp/cache/train/{index}"
            env[f"AUTOLOOP_EVAL_{key}"] = f"/tmp/cache/eval/{index}"
        manifest = SimpleNamespace(env=env, execution_mode="exact-dense1024-cold-eval")
        diag = startup_cache_diagnostics(manifest, {
            "startup_allowance_est": 846.874,
            "wall_elapsed_seconds": 2640.0,
            "training_seconds": 1785.0,
            "compiled": True,
        })
        self.assertEqual(diag["startup_allowance_cap_seconds"], 900)
        self.assertEqual(diag["startup_allowance_status"], "within_cap")
        self.assertTrue(diag["cache"]["pairwise_unique"])
        self.assertEqual(diag["cache"]["hit_miss_observation"],
                         "unavailable in existing receipts")

    def test_each_cycle_journals_idle_and_reports_over_ten_minutes(self):
        class IdleOperator:
            def inventory(self, box):
                return Inventory(
                    box, C.PINNED_HOSTS[box], "host21", [], "", 6.85, True, True,
                    "2026-09-24T12:00:00Z", ["free disk 6.85 GiB < required 7.40 GiB"],
                )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent = root / "research" / "agentic"
            stop = agent / "STOP"
            stop.parent.mkdir(parents=True)
            store = Store(root=agent, state_path=agent / "runtime" / "state.json",
                          experiments_dir=agent / "experiments")
            clock = [T0]
            cfg = C.LoopConfig(
                researcher_model_id="x.anthropic.y", verifier_model_id="x.openai.y",
                boxes=["trn21"],
            )
            ctl = Controller(
                cfg=cfg, store=store, operator=IdleOperator(), model=None,
                archiver=object(), local_git=object(), stop_file=stop, now=lambda: clock[0],
            )
            ctl._reconcile = lambda _report: set()
            ctl._advance_box = lambda _box, _report: False
            first = ctl.cycle()
            self.assertFalse(any("idle >10min" in action for action in first.actions))
            clock[0] += timedelta(seconds=601)
            second = ctl.cycle()
            self.assertTrue(any("idle >10min" in action for action in second.actions))
            rows = [json.loads(line) for line in store.utilization_journal_path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["inventory_status"], "idle")
            self.assertIn("launch-blocked", rows[-1]["reason"])
            self.assertEqual(rows[-1]["idle_seconds"], 601.0)

    def test_busy_resets_idle_streak_and_unknown_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = Store(root=root, state_path=root / "state.json", experiments_dir=root / "experiments")
            store.record_host_utilization(
                box="trn21", host="h", status="idle", reason="idle",
                observed_at="2026-09-24T12:00:00Z",
            )
            store.record_host_utilization(
                box="trn21", host="h", status="busy", reason="process",
                observed_at="2026-09-24T12:05:00Z",
            )
            elapsed = store.record_host_utilization(
                box="trn21", host="h", status="idle", reason="idle",
                observed_at="2026-09-24T12:20:00Z",
            )
            self.assertEqual(elapsed, 0.0)
            store.record_host_utilization(
                box="trn21", host="h", status="unknown", reason="inventory failed",
                observed_at="2026-09-24T12:21:00Z",
            )
            row = json.loads(store.utilization_journal_path.read_text().splitlines()[-1])
            self.assertEqual(row["inventory_status"], "unknown")
            self.assertIsNone(row["idle_seconds"])


class FrozenSafetyBoundaryTests(unittest.TestCase):
    def test_safe_final_week_disk_gate_and_active_queue_base(self):
        raw = json.loads((Path(__file__).resolve().parents[1] / "autoloop.json").read_text())
        # The 2026-09-24 human-approved final-week override lowered only the operational gate.
        # It remains 1.5 GiB above the immutable protocol validator exercised below.
        self.assertEqual(raw["caps"]["min_free_gib"], 6.5)
        self.assertEqual(C.AUTHORITATIVE_TRAIN_SHA256,
                         "9173f62617789b058083c3a4722a906e2b165f4986716281f1915db3fb9fa344")
        errors = C.LoopConfig(
            researcher_model_id="x.anthropic.y", verifier_model_id="x.openai.y",
            caps=C.Caps(min_free_gib=4.99),
        ).validate()
        self.assertTrue(any("min_free_gib may not be lowered" in error for error in errors))

    def test_addendum_records_new_base_without_repinning_controller(self):
        text = C.FINAL_WEEK_RESEARCH_ADDENDUM.read_text()
        self.assertIn("116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d", text)
        self.assertIn("0.9918353324459732", text)
        self.assertIn("old width-640 base", text)
        self.assertIn("Audit source wiring first", text)


if __name__ == "__main__":
    unittest.main()
