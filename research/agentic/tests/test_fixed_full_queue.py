"""Fail-closed local tests for the human-authorized dense1024 full-only queue."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoloop import config as C
from autoloop.controller import Controller, CycleReport
from autoloop.manifest import ManifestError, assert_manifest_legal, build_manifest
from autoloop.queue_plan import FixedQueuePlan, QueuePlanError
from autoloop.remote import Inventory, LocalGit, guard_remote_argv
from autoloop.state import Store
from autoloop import remote_cache_cleanup as RCC

REPO_ROOT = Path(__file__).resolve().parents[3]


class NoRemote:
    def __getattr__(self, name):
        raise AssertionError(f"unexpected remote operation: {name}")


class NoModel:
    def complete(self, **_kwargs):
        raise AssertionError("fixed queue must never call a model/proposer")


class NoArchive:
    pass


def fixed_cfg() -> C.LoopConfig:
    return C.LoopConfig(
        researcher_model_id="a.anthropic.x", verifier_model_id="a.openai.y",
        plan_mode="fixed_full_only", boxes=["trn22", "trn21"],
        neuron_cc_flags=C.EXACT_DENSE1024_CC_FLAGS,
    )


class FixedQueueTests(unittest.TestCase):
    def setUp(self):
        self.root_patch = mock.patch.object(C, "ROOT", REPO_ROOT)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.plan = FixedQueuePlan.load(C.QUEUE_PLAN_FILE, REPO_ROOT)

    def test_plan_pins_order_hosts_duration_and_batch(self):
        self.assertEqual(len(self.plan.entries), 7)
        self.assertEqual([e.box for e in self.plan.entries[:2]], ["trn22", "trn21"])
        self.assertEqual(self.plan.entries[5].constant_change["to"], 262144)
        self.assertEqual(self.plan.entries[5].constant_change["legal_multiple"], 131072)
        self.assertEqual(
            self.plan.entries[6].constant_change,
            {"name": "MAX_TRAIN_SECONDS", "from": 1800,
             "requested_target": "approximately 1080-second one-epoch diagnostic", "to": 1080},
        )
        self.assertNotIn("NUM_STEPS", json.dumps(self.plan.raw["queue"][6]))

    def test_plan_tamper_order_eval_argv_and_cleanup_fail(self):
        mutations = []
        for mutate in (
            lambda d: d["queue"].reverse(),
            lambda d: d["evaluation"].update(expected_num_bytes=1),
            lambda d: d["exact_train_argv"].append("--num-steps"),
            lambda d: d["compile_cache_cleanup"].update(only_after_archive_commit_verified_on_configured_remote=False),
            lambda d: d["queue"][1].update(commit="0" * 40),
        ):
            data = copy.deepcopy(self.plan.raw)
            mutate(data)
            mutations.append(data)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "plan.json"
            for data in mutations:
                path.write_text(json.dumps(data))
                with self.assertRaises(QueuePlanError):
                    FixedQueuePlan.load(path, C.ROOT)

    def test_schema2_migration_parks_r3_and_preserves_all_other_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path = root / "state.json"
            original = {
                "schema_version": 2, "created_at": "old", "controller_id": "old-controller",
                "proposals": {"20260923-local-conv-pre-attn-serial-only-r3": {
                    "stage": "verified", "tiers": {"full": {"label": "old-label"}},
                    "verdict": None, "box": "trn21",
                    "candidate_commit": "79b0cecb2f655441f4e0898f6140308eea6072f0",
                    "updated_at": "old",
                }},
                "usage": {"2026-09-23": {"runs": 37}}, "box_failures": {"trn21": 2},
                "runs_total": 37, "breaker": {"reason": "disk", "detail": "preserve me"},
                "cycles": 287, "last_cycle": {"actions": ["history"]},
            }
            state_path.write_text(json.dumps(original))
            store = Store(root=root, state_path=state_path, experiments_dir=root / "experiments")
            store.activate_fixed_plan(
                plan_id=self.plan.raw["plan_id"], plan_sha256=self.plan.sha256,
                parked_proposals=self.plan.raw["parked_proposals"],
            )
            after = store.state
            rec = after["proposals"]["20260923-local-conv-pre-attn-serial-only-r3"]
            self.assertEqual((rec["stage"], rec["parked_from"], rec["launch_authorized"]),
                             ("parked", "verified", False))
            for key in ("usage", "box_failures", "runs_total", "breaker", "cycles", "last_cycle"):
                self.assertEqual(after[key], original[key])

    def test_breaker_needs_exact_operator_cleanup_receipt_and_retains_previous(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = Store(root=root, state_path=root / "state.json", experiments_dir=root / "experiments")
            store.open_breaker("disk gate", box="trn21")
            with self.assertRaises(RuntimeError):
                store.clear_breaker_for_fixed_plan(
                    receipt_commit="0" * 40, receipt_path=C.BREAKER_CLEANUP_RECEIPT_PATH,
                    receipt_sha256=C.BREAKER_CLEANUP_RECEIPT_SHA256, cleared_by_role="operator",
                )
            self.assertIsNotNone(store.breaker())
            store.clear_breaker_for_fixed_plan(
                receipt_commit=C.BREAKER_CLEANUP_RECEIPT_COMMIT,
                receipt_path=C.BREAKER_CLEANUP_RECEIPT_PATH,
                receipt_sha256=C.BREAKER_CLEANUP_RECEIPT_SHA256,
                cleared_by_role="operator",
            )
            self.assertIsNone(store.breaker())
            self.assertEqual(store.state["fixed_breaker_clearance"]["previous_breaker"]["reason"], "disk gate")

    def test_controller_parks_then_waits_without_remote_model_or_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path = root / "state.json"
            state = Store(root=root, state_path=state_path, experiments_dir=root / "experiments")
            state.proposals()["20260923-local-conv-pre-attn-serial-only-r3"] = {
                "stage": "verified", "tiers": {}, "verdict": None, "box": "trn21",
                "candidate_commit": "79b0cecb2f655441f4e0898f6140308eea6072f0",
                "updated_at": "old",
            }
            state.save()
            ctl = Controller(
                cfg=fixed_cfg(), store=state, operator=NoRemote(), model=NoModel(),
                archiver=NoArchive(), local_git=LocalGit(C.ROOT), stop_file=root / "STOP",
                now=lambda: datetime(2026, 9, 23, tzinfo=timezone.utc),
            )
            with mock.patch.object(ctl, "_propose", side_effect=AssertionError("must not propose")):
                report = ctl.cycle()
            self.assertTrue(any("breaker-clearance receipt" in x for x in report.actions))
            self.assertEqual(state.proposals()["20260923-local-conv-pre-attn-serial-only-r3"]["stage"], "parked")
            self.assertNotIn("smoke", json.dumps(state.state))

    def test_exact_manifest_has_no_train_overrides_and_pins_cold_eval(self):
        entry = self.plan.entries[1]
        manifest = build_manifest(
            proposal_id=entry.proposal_id, tier="full", box=entry.box,
            remote_run_dir=f"{C.REMOTE_RUNS_DIR}/{entry.commit[:8]}--x--label",
            label="al-20260923-fixed-test-full-r1", train_flags=[], cfg=fixed_cfg(), frozen_at="t",
            control_label="al-control-full-r1", abort_conditions=[], description_note="fixed",
            candidate_commit=entry.commit, execution_mode="exact-dense1024-cold-eval",
            comparison_role="candidate", expected_train_sha256=entry.train_sha256,
        )
        self.assertNotIn("--num-steps", manifest.argv)
        self.assertNotIn("--max-train-seconds", manifest.argv)
        self.assertNotIn("--out-dir", manifest.argv)
        self.assertEqual(manifest.exact_train_argv, list(C.EXACT_DENSE1024_TRAIN_ARGV))
        self.assertEqual((manifest.eval_context, manifest.eval_tokens, manifest.eval_bytes),
                         (1024, 20971520, 83462186))
        caches = [manifest.env[k] for k in C.CACHE_ENV_KEYS]
        caches += [manifest.env[f"AUTOLOOP_EVAL_{k}"] for k in C.CACHE_ENV_KEYS]
        self.assertEqual(len(caches), len(set(caches)))
        broken = copy.deepcopy(manifest)
        broken.eval_tokens = 0
        with self.assertRaises(ManifestError):
            assert_manifest_legal(broken)

    def test_commit_parent_changed_paths_and_protected_hashes_are_enforced(self):
        ctl = object.__new__(Controller)
        ctl.git = LocalGit(C.ROOT)
        ctl._validate_fixed_entry(self.plan.entries[1])
        with mock.patch.object(ctl.git, "parent_of", return_value="0" * 40):
            with self.assertRaises(QueuePlanError):
                ctl._validate_fixed_entry(self.plan.entries[1])
        with mock.patch.object(ctl.git, "changed_paths", return_value=["train.py", "prepare.py"]):
            with self.assertRaises(QueuePlanError):
                ctl._validate_fixed_entry(self.plan.entries[1])
        original = ctl.git.file_at_commit
        with mock.patch.object(
            ctl.git, "file_at_commit",
            side_effect=lambda commit, path: b"evil" if path == "prepare.py" else original(commit, path),
        ):
            with self.assertRaises(QueuePlanError):
                ctl._validate_fixed_entry(self.plan.entries[1])

    def test_fixed_result_rejects_nonfinite_wrong_counts_and_missing_steps(self):
        entry = self.plan.entries[0]
        rec = {"candidate_commit": entry.commit}
        manifest = {"expected_train_sha256": entry.train_sha256}
        row = {
            "status": "keep", "over_budget": False, "public_val_bpb": 0.99,
            "num_steps": 3600, "_dirt_explained": True,
            "eval_context": 1024, "eval_tokens_requested": 20971520,
            "eval_num_tokens": 20971520, "eval_num_bytes": 83462186,
            "eval_counts_match": True, "eval_over_timeout": False,
            "selection_metric": "public_val_bpb",
            "exact_train_argv": list(C.EXACT_DENSE1024_TRAIN_ARGV),
            "git_commit": entry.commit, "source_dirty": False,
            "source_sha256": {"train.py": entry.train_sha256},
            "external_eval": {"fresh_process": True, "returncode": 0,
                              "seq_len": 1024, "eval_tokens": 20971520},
        }
        result = {"returncode": 0, "ledger_row": row,
                  "collected": {"checkpoint_present": True, "checkpoint_sha256": "a" * 64}}
        self.assertEqual(Controller._fixed_correctness_failures(rec, result, manifest), [])
        for key, value in (("public_val_bpb", math.nan), ("eval_num_tokens", 1),
                           ("eval_num_bytes", 1), ("num_steps", 0)):
            bad = copy.deepcopy(result)
            bad["ledger_row"][key] = value
            self.assertTrue(Controller._fixed_correctness_failures(rec, bad, manifest), key)
    def test_paired_preflight_partial_failure_then_restart_launches_full_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            experiments = root / "experiments"
            for entry in self.plan.entries[:2]:
                dst = experiments / entry.proposal_id
                dst.mkdir(parents=True)
                shutil.copy(C.ROOT / entry.proposal_path, dst / "proposal.json")
            store = Store(root=root, state_path=root / "state.json", experiments_dir=experiments)
            store.proposals()["20260923-local-conv-pre-attn-serial-only-r3"] = {
                "stage": "verified", "tiers": {}, "verdict": None, "box": "trn21",
                "candidate_commit": "79b0cecb2f655441f4e0898f6140308eea6072f0", "updated_at": "old",
            }
            store.state["fixed_breaker_clearance"] = {
                "receipt_commit": C.BREAKER_CLEANUP_RECEIPT_COMMIT,
                "receipt_path": C.BREAKER_CLEANUP_RECEIPT_PATH,
                "receipt_sha256": C.BREAKER_CLEANUP_RECEIPT_SHA256,
                "cleared_by_role": "operator",
            }
            store.save()
            op = mock.Mock()
            dirty = Inventory("trn21", C.PINNED_HOSTS["trn21"], "h21", [], "", 11, True, True, "t", ["busy"])
            clean22 = Inventory("trn22", C.PINNED_HOSTS["trn22"], "h22", [], "", 11, True, True, "t", [])
            clean21 = Inventory("trn21", C.PINNED_HOSTS["trn21"], "h21", [], "", 11, True, True, "t", [])
            op.inventory.side_effect = [clean22, dirty]
            ctl = Controller(
                cfg=fixed_cfg(), store=store, operator=op, model=NoModel(), archiver=NoArchive(),
                local_git=LocalGit(C.ROOT), stop_file=root / "STOP",
                now=lambda: datetime(2026, 9, 23, tzinfo=timezone.utc),
            )
            launches = []
            ctl.fixed_plan = mock.Mock(wraps=self.plan)
            ctl.fixed_plan.entries = self.plan.entries
            ctl.fixed_plan.raw = self.plan.raw
            ctl.fixed_plan.sha256 = self.plan.sha256
            ctl.fixed_plan.all_verified.return_value = True
            with mock.patch.object(ctl, "_stage_and_launch", side_effect=lambda *a, **k: launches.append((a, k)) or True), \
                    mock.patch.object(ctl, "_reconcile", return_value=set()), \
                    mock.patch.object(ctl, "_propose", side_effect=AssertionError("must not propose")):
                ctl._cycle_fixed(CycleReport("t"))
                self.assertEqual(launches, [])
                op.inventory.side_effect = [clean22, clean21]
                ctl._cycle_fixed(CycleReport("t2"))
            self.assertEqual(len(launches), 2)
            self.assertTrue(all(call[0][1] == "full" for call in launches))
            self.assertEqual([call[0][2] for call in launches], ["trn22", "trn21"])

    def test_cache_cleanup_guard_accepts_only_exact_helper_shape(self):
        base = f"{C.REMOTE_AUTOLOOP_DIR}/cache-cleanup/al-fixed"
        guard_remote_argv([
            C.REMOTE_PYTHON, f"{base}/remote_cache_cleanup.py", f"{base}/cleanup.json"
        ])
        with self.assertRaises(Exception):
            guard_remote_argv([C.REMOTE_PYTHON, f"{base}/evil.py", f"{base}/cleanup.json"])

    def test_cache_cleanup_removes_only_exact_tree_and_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "caches"
            label = "al-20260923-cache-test"
            root = base / label
            paths = []
            for group in ("train", "eval"):
                for key in RCC.KEYS:
                    path = root / group / key.lower()
                    path.mkdir(parents=True)
                    (path / "compiled.neff").write_bytes(b"x")
                    paths.append(str(path))
            spec = Path(td) / "cleanup.json"
            receipt = Path(td) / "receipt.json"
            spec.write_text(json.dumps({"label": label, "cache_root": str(root),
                                        "cache_paths": paths, "receipt_path": str(receipt)}))
            with mock.patch.object(RCC, "BASE", str(base)), mock.patch.object(sys, "argv", ["x", str(spec)]):
                RCC.main()
            self.assertFalse(root.exists())
            self.assertTrue(json.loads(receipt.read_text())["absent_after"])

            bad_root = base / "bad"
            bad_root.mkdir(parents=True)
            (bad_root / "train").symlink_to(Path(td), target_is_directory=True)
            bad_paths = [str(bad_root / group / key.lower()) for group in ("train", "eval") for key in RCC.KEYS]
            bad_spec = Path(td) / "bad-cleanup.json"
            bad_spec.write_text(json.dumps({"label": "bad", "cache_root": str(bad_root),
                                            "cache_paths": bad_paths,
                                            "receipt_path": str(Path(td) / "bad-receipt.json")}))
            with mock.patch.object(RCC, "BASE", str(base)), mock.patch.object(sys, "argv", ["x", str(bad_spec)]):
                with self.assertRaises(SystemExit):
                    RCC.main()

    def test_archive_push_precedes_checkpoint_and_cache_prune_and_receipts_close(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent = root / "research" / "agentic"
            experiments = agent / "experiments"
            pid = self.plan.entries[0].proposal_id
            pdir = experiments / pid
            artifacts = pdir / "artifacts" / "full"
            artifacts.mkdir(parents=True)
            proposal = {"proposal_id": pid, "hypothesis": "control", "single_causal_change": "none",
                        "kind": "control"}
            (pdir / "proposal.json").write_text(json.dumps(proposal))
            (root / "verification.json").write_text("{}")
            entry = self.plan.entries[0]
            manifest = build_manifest(
                proposal_id=pid, tier="full", box="trn22",
                remote_run_dir=f"{C.REMOTE_RUNS_DIR}/{entry.commit[:8]}--x--al-20260923-archive-full-r1",
                label="al-20260923-archive-full-r1", train_flags=[], cfg=fixed_cfg(), frozen_at="t",
                control_label=None, abort_conditions=[], description_note="control",
                candidate_commit=entry.commit, execution_mode="exact-dense1024-cold-eval",
                comparison_role="control", expected_train_sha256=entry.train_sha256,
            )
            for name, data in {
                "full-manifest.json": manifest.to_dict(), "full-staged.json": {},
                "full-launch.json": {},
                "full-result.json": {"label": manifest.label,
                                     "collected": {"checkpoint_sha256": "a" * 64}},
                "full-verdict.json": {"decision": "KEEP", "metrics": {"candidate_bpb": 0.99}},
            }.items():
                (pdir / name).write_text(json.dumps(data))
            (pdir / "full-verdict.md").write_text("# verdict\n")
            for name in ("experiment-log.jsonl", f"{manifest.label}.log",
                         f"{manifest.label}.eval.log", f"{manifest.label}.launch.json",
                         "runner.log", "done.json", "launch.json"):
                (artifacts / name).write_text("evidence\n")
            store = Store(root=agent, state_path=agent / "state.json", experiments_dir=experiments)
            store.proposals()[pid] = {"stage": "analyzed:full", "tiers": {}, "verdict": "KEEP",
                                      "box": "trn22", "candidate_commit": entry.commit,
                                      "comparison_role": "control", "fixed_queue": True}
            store.save()
            events = []

            class Archiver:
                def commit(self, _paths, message):
                    events.append(("commit", message))
                    return f"c{len(events)}"

                def verify_commit(self, commit, paths):
                    self_paths = [Path(p) for p in paths]
                    if not all(path.is_file() for path in self_paths):
                        raise AssertionError("verification path missing")
                    events.append(("verify", commit))
                    return {"commit": commit, "local_verified": True, "remote_verified": True}

            class Operator:
                def prune_checkpoint(self, _manifest, expected):
                    events.append(("checkpoint_prune", expected))
                    return {"sha256": expected}

                def cleanup_compile_caches(self, _manifest, local_receipt):
                    events.append(("cache_cleanup", _manifest.label))
                    local_receipt.write_text(json.dumps({"absent_after": True}))
                    return {"absent_after": True}

            ctl = object.__new__(Controller)
            ctl.store, ctl.archiver, ctl.op = store, Archiver(), Operator()
            ctl.fixed_plan = SimpleNamespace(entries=(SimpleNamespace(
                proposal_id=pid, verification_path="verification.json"),))
            ctl.now_iso = lambda: "2026-09-23T00:00:00Z"
            decision_log = agent / "DECISION_LOG.md"
            decision_log.parent.mkdir(parents=True, exist_ok=True)
            decision_log.write_text("# decisions\n")
            with mock.patch.object(C, "ROOT", root), mock.patch.object(C, "DECISION_LOG", decision_log):
                ctl._archive_fixed(pid, "full", proposal,
                                   {"decision": "KEEP", "metrics": {"candidate_bpb": 0.99}},
                                   CycleReport("t"))
            order = [kind for kind, _value in events]
            self.assertLess(order.index("verify"), order.index("checkpoint_prune"))
            self.assertLess(order.index("checkpoint_prune"), order.index("cache_cleanup"))
            self.assertGreaterEqual(order.count("verify"), 4)
            self.assertEqual(store.proposals()[pid]["stage"], "closed")
            self.assertTrue((pdir / "full-checkpoint-pruned.json").is_file())
            self.assertTrue((pdir / "full-cache-cleanup.json").is_file())


if __name__ == "__main__":
    unittest.main()
