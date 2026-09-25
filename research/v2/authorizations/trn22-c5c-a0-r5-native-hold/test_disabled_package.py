"""Offline, read-only invariants for the disabled trn22 A0 r5 package."""

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parent
SOURCE = "446b657695f086f9b484fa601c9a3b61c2495912"
LABEL = "fs-20260925-v2-c5c-a0-control-trn22-r5"
OLD_LABEL = "fs-20260925-v2-c5c-a0-control-trn22-r4"
CHECKOUT = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r4"
MONITOR = "/home/ubuntu/trainium_a0_disk_monitor_20260925_r5"
SOURCE_MONITORS = "research/v2/disk-monitor/"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def frozen_file(path):
    return subprocess.check_output(["git", "show", f"{SOURCE}:{path}"], cwd=ROOT)


class DisabledPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = read_json(PACKAGE / "GATE_CONTRACT.json")
        cls.request = read_json(PACKAGE / "NATIVE_REQUEST.DISABLED.json")
        cls.base = read_json(PACKAGE / "BASE_BINDING.STATIC_ONLY.json")
        cls.pair = read_json(PACKAGE / "PAIR_BINDING.STATIC_ONLY.json")
        cls.eval = read_json(PACKAGE / "PUBLIC20M_EVAL.DISABLED.json")

    def test_source_bytes_are_exact_frozen_blobs(self):
        c, r, b = self.contract, self.request, self.base
        self.assertEqual(c["frozen_source_commit"], SOURCE)
        self.assertEqual(r["expected_commit"], SOURCE)
        self.assertEqual(b["expected_commit"], SOURCE)
        self.assertEqual(c["source_sha256"], r["expected_sha256"])
        self.assertEqual(c["source_sha256"], b["expected_sha256"])
        self.assertEqual(set(r["fingerprint_paths"]), set(c["source_sha256"]))
        for path, expected in c["source_sha256"].items():
            with self.subTest(path=path):
                self.assertEqual(sha256(frozen_file(path)), expected)

    def test_draft_hash_chain_and_runbook_pins(self):
        c, b, p = self.contract, self.base, self.pair
        runbook = (PACKAGE / "RUNBOOK.md").read_text()
        for key in ("request", "base_binding", "pair_binding", "public20m_eval_plan"):
            with self.subTest(key=key):
                artifact = c[key]
                path = ROOT / artifact["path"]
                self.assertEqual(path.parent, PACKAGE)
                self.assertEqual(sha256(path.read_bytes()), artifact["sha256"])
                self.assertIn(artifact["sha256"], runbook)
        self.assertEqual(b["request_sha256"], c["request"]["sha256"])
        self.assertEqual(p["request_sha256"], c["request"]["sha256"])
        self.assertEqual(p["base_binding_sha256"], c["base_binding"]["sha256"])
        self.assertEqual(
            b["monitor_script_sha256"],
            c["source_sha256"][SOURCE_MONITORS + "root_free_guard_monitor.py"],
        )
        self.assertEqual(
            p["wrapper_script_sha256"],
            c["source_sha256"][SOURCE_MONITORS + "paired_root_free_guard_monitor.py"],
        )

    def test_no_host_or_dispatch_or_evaluation_authority(self):
        c, r, b, p, e = (
            self.contract, self.request, self.base, self.pair, self.eval
        )
        self.assertFalse(c["host_action_authorized"])
        self.assertFalse(c["native_dispatch_authorized"])
        self.assertFalse(c["fresh_public20m_eval_authorized"])
        self.assertFalse(r["dispatch_authorized"])
        self.assertFalse(e["evaluation_authorized"])
        self.assertEqual(b["package_state"], "STATIC_ONLY")
        self.assertEqual(p["package_state"], "STATIC_ONLY")
        self.assertTrue(all(value is False for value in c["required_gates"].values()))
        for key in (
            "source_transfer_receipt_sha256",
            "source_checkout_result_review_sha256",
            "r5_proposal_ratification_sha256",
            "r5_final_request_review_sha256",
            "r5_one_shot_native_assignment_sha256",
            "r5_operator_terminal_receipt_sha256",
            "r5_public20m_eval_receipt_sha256",
        ):
            self.assertIsNone(c[key])
        self.assertIsNone(e["checkpoint_sha256"])
        self.assertFalse(list(PACKAGE.glob("*authorization*.json")))
        self.assertTrue(r["state"].startswith("DISABLED_"))
        self.assertTrue(e["state"].startswith("DISABLED_"))

    def test_r5_uses_r4_checkout_with_new_science_and_monitor_paths(self):
        c, r, b, p, e = (
            self.contract, self.request, self.base, self.pair, self.eval
        )
        self.assertEqual(c["science_label"], LABEL)
        self.assertEqual((r["label"], b["label"], p["label"], e["training_label"]),
                         (LABEL,) * 4)
        self.assertEqual((r["remote"], b["remote"], p["remote"], e["source_checkout"]),
                         (CHECKOUT,) * 4)
        self.assertEqual(c["future_accepted_source_checkout"], CHECKOUT)
        self.assertEqual(c["new_external_monitor_target"], MONITOR)
        self.assertEqual((r["expected_hostname"], b["expected_hostname"],
                          p["expected_hostname"]), ("ip-172-31-32-253",) * 3)
        self.assertEqual((b["expected_uid"], p["expected_uid"]), (1000, 1000))
        self.assertNotIn(OLD_LABEL, json.dumps((r, b, p, e)))
        self.assertEqual(r["monitor_interlock"]["pair_binding_path"],
                         p["base_binding_path"].replace(".base-binding.json", ".pair-binding.json"))
        self.assertEqual(r["monitor_interlock"]["interlock_path"], p["interlock_path"])
        self.assertEqual(r["monitor_interlock"]["no_spawn_path"], p["no_spawn_path"])
        for path in (
            b["request_path"], b["events_path"], b["ready_path"],
            p["authorization_path"], p["base_monitor_path"], p["wrapper_path"],
            p["interlock_path"], p["no_spawn_path"],
            *(entry[name] for entry in p["roles"].values()
              for name in ("events_path", "ready_path")),
        ):
            self.assertTrue(path.startswith(MONITOR + "/"), path)

    def test_training_recipe_is_r4_science_with_only_r5_label_paths(self):
        r = self.request
        old = read_json(ROOT / SOURCE_MONITORS / "r4-hold" / (OLD_LABEL + ".request.DISABLED.json"))
        self.assertEqual(r["command"], [x.replace(OLD_LABEL, LABEL) for x in old["command"]])
        self.assertEqual(r["environment"], old["environment"])
        self.assertEqual(r["warm_cache_policy"], old["warm_cache_policy"])
        self.assertEqual(r["external_boundary_observer"]["command"],
                         [x.replace(OLD_LABEL, LABEL)
                          for x in old["external_boundary_observer"]["command"]])
        self.assertEqual(r["external_boundary_observer"]["script_sha256"],
                         old["external_boundary_observer"]["script_sha256"])
        self.assertEqual(r["command"][:7],
                         ["timeout", "--foreground", "--signal=TERM",
                          "--kill-after=60s", "3600s", "env", "-i"])
        self.assertEqual(r["command"][-2:], ["--matrix-lr", "0.015"])
        self.assertEqual(r["command"][r["command"].index("train.py") + 1:],
                         read_json(ROOT / "research/v2/authorizations/20260925-trn22-c5c-a0-r2-proposal-ratification.json")["train_argv"])
        queue = read_json(ROOT / "research/v2/queue.json")
        self.assertEqual(shlex.split(queue["base_command"])[-len(r["command"][r["command"].index("train.py") + 1:]):],
                         r["command"][r["command"].index("train.py") + 1:])
        self.assertEqual((r["environment"]["NEURON_LOGICAL_NC_CONFIG"],
                          r["environment"]["NEURON_COMPETITION_R1_SEED"],
                          r["environment"]["NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS"]),
                         ("2", "42", "1800"))
        self.assertIn("--optlevel=1", r["environment"]["NEURON_CC_FLAGS"])
        env_end = r["command"].index("/home/ubuntu/workspace/native_venv/bin/torchrun")
        clean_env = dict(part.split("=", 1) for part in r["command"][7:env_end])
        for key, value in r["environment"].items():
            self.assertEqual(clean_env[key], value)
        self.assertEqual(r["command"][env_end:env_end + 3],
                         ["/home/ubuntu/workspace/native_venv/bin/torchrun",
                          "--standalone", "--nproc_per_node=4"])

    def test_disk_thresholds_and_live_monitor_shape(self):
        c, r, b, p = self.contract, self.request, self.base, self.pair
        t = c["safety_thresholds_bytes"]
        self.assertEqual((r["min_free_bytes"], b["floor_free_bytes"],
                          t["immutable_operational_floor"]), (6979321856,) * 3)
        self.assertEqual((r["pre_launch_required_free_bytes"], b["prelaunch_free_bytes"],
                          t["minimum_prelaunch_root_free"]), (10737418240,) * 3)
        self.assertEqual((b["stop_free_bytes"], p["stop_free_bytes"],
                          t["paired_monitor_stop"]), (9126805504,) * 3)
        self.assertEqual(b["stop_free_bytes"] - b["floor_free_bytes"], 2 * 1024**3)
        self.assertLessEqual(b["poll_seconds"], t["monitor_poll_max_seconds"])
        self.assertTrue(r["external_boundary_observer"]["must_start_before_guard"])
        self.assertEqual(r["external_boundary_observer"]["command"][3], LABEL)

    def test_postrun_public20m_plan_matches_calibrated_g1(self):
        e, r = self.eval, self.request
        g1 = read_json(ROOT / "research/v2/g1.json")
        calibration = read_json(ROOT / "research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn22-r1/dispatch/dispatch.json")["request"]
        expected = [x.replace("prepare.py", CHECKOUT + "/prepare.py")
                    .replace("<absolute-train.py>", CHECKOUT + "/train.py")
                    .replace("<absolute-final.pt>", e["checkpoint_path"])
                    for x in g1["evaluation"]["command_template"]]
        self.assertEqual(e["command"][:5],
                         ["timeout", "--foreground", "--signal=TERM",
                          "--kill-after=60s", "1800s"])
        self.assertEqual(e["command"][5:], expected)
        expected_env = dict(calibration["environment"])
        expected_env["PYTHONPATH"] = CHECKOUT
        self.assertEqual(e["environment"], expected_env)
        self.assertTrue(e["fresh_process"])
        self.assertEqual((e["context_tokens"], e["eval_tokens"], e["expected_num_bytes"]),
                         (1024, 20971520, g1["calibration"]["trn22"]["known_num_bytes"]))
        self.assertEqual(e["train_sha256"], r["expected_sha256"]["train.py"])
        self.assertEqual(e["prepare_sha256"], r["expected_sha256"]["prepare.py"])
        self.assertEqual(e["expected_source_commit"], SOURCE)
        self.assertEqual(e["cwd"], CHECKOUT)

    def test_prior_receipts_are_exact_and_r2_ratification_cannot_be_reused(self):
        ev = self.contract["upstream_evidence_not_dispatch_authority"]
        pinned = {
            "r4_source_stage_contract": "research/v2/authorizations/trn22-c5c-a0-r4-source-stage/GATE_CONTRACT.json",
            "r4_source_safety_review": "research/v2/disk-monitor/R4_446B6576_INDEPENDENT_VERIFIER_RECEIPT.md",
            "r4_disabled_package_review": "research/v2/disk-monitor/r4-hold/INDEPENDENT_VERIFIER_RECEIPT.md",
            "g2_carryforward_disposition": "research/v2/disk-monitor/r4-hold/G2_R3_TO_R4_SOURCE_IDENTITY_DISPOSITION.md",
        }
        for stem, path in pinned.items():
            with self.subTest(stem=stem):
                self.assertEqual(ev[stem + "_path"], path)
                self.assertEqual(sha256((ROOT / path).read_bytes()), ev[stem + "_sha256"])
        paths = {
            "g2_r3_operator_receipt_sha256": "research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn22-r3/operator-receipt.json",
            "g2_r3_independent_numeric_review_sha256": "research/agentic/verification/20260925-trn22-c5c-a0-r2-g2-r3-numeric-review.md",
            "trn22_g1_operator_receipt_sha256": "research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn22-r1/operator-receipt.json",
            "trn22_g1_independent_review_sha256": "research/agentic/verification/20260925-g1-trn22-calibration-review.md",
            "r2_proposal_sha256": "research/submission-validation/20260925-v2-c5c-a0-control-trn22-r2/EXPERIMENT_PROPOSAL.json",
            "r2_one_label_ratification_sha256": "research/v2/authorizations/20260925-trn22-c5c-a0-r2-proposal-ratification.json",
            "queue_snapshot_sha256": "research/v2/queue.json",
        }
        for key, path in paths.items():
            with self.subTest(key=key):
                self.assertEqual(sha256((ROOT / path).read_bytes()), ev[key])
        stage = read_json(ROOT / pinned["r4_source_stage_contract"])
        self.assertEqual(stage["local_exact_source_artifact"]["sha256"],
                         ev["r4_capsule_sha256"])
        r2 = read_json(ROOT / paths["r2_one_label_ratification_sha256"])
        self.assertNotEqual(r2["one_attempt_label"], LABEL)
        self.assertFalse(ev["r2_ratification_covers_r5_label"])
        a0 = next(item for item in read_json(ROOT / "research/v2/queue.json")["queue"]
                  if item["id"] == ev["queue_a0_id"])
        self.assertEqual(a0["kind"], "control")

    def test_frozen_launcher_rejects_disabled_request_before_host_contact(self):
        source = frozen_file("research/fresh_start_launch.py").decode()
        refusal = source.index("assert request.get('dispatch_authorized') is True")
        receipt = source.index('receipt = ROOT / "research/run-logs"')
        ssh = source.index("completed = subprocess.run(")
        self.assertLess(refusal, receipt)
        self.assertLess(receipt, ssh)
        self.assertFalse(self.request["dispatch_authorized"])


if __name__ == "__main__":
    unittest.main()
