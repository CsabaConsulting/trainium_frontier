"""Offline checks for a disabled trn22 r10 A0 draft; never contacts a host."""

import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parent
COMMIT = "22bfbab5a77d6feb670d2d737504f8b70b08ddc1"
LABEL = "fs-20260925-v2-c5c-a0-control-trn22-r10"
CHECKOUT = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r10_stage_r1"
MONITOR = "/home/ubuntu/trainium_a0_disk_monitor_20260925_r10"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def frozen(path):
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


def isolated_function(source, name):
    module = ast.parse(source)
    function = next(
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<frozen>", "exec"), namespace)
    return namespace[name]


class DisabledR10NativePackage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = read_json(PACKAGE / "GATE_CONTRACT.json")
        cls.request = read_json(PACKAGE / "NATIVE_REQUEST.DISABLED.json")
        cls.base = read_json(PACKAGE / "BASE_BINDING.STATIC_ONLY.json")
        cls.pair = read_json(PACKAGE / "PAIR_BINDING.STATIC_ONLY.json")
        cls.eval = read_json(PACKAGE / "PUBLIC20M_EVAL.DISABLED.json")

    def test_frozen_source_and_review_bytes(self):
        c, r, b = self.contract, self.request, self.base
        self.assertEqual((c["frozen_source_commit"], r["expected_commit"],
                          b["expected_commit"]), (COMMIT,) * 3)
        self.assertEqual(c["source_sha256"], r["expected_sha256"])
        self.assertEqual(c["source_sha256"], b["expected_sha256"])
        self.assertEqual(set(r["fingerprint_paths"]), set(c["source_sha256"]))
        self.assertEqual(len(c["source_sha256"]), 11)
        for path, expected in c["source_sha256"].items():
            with self.subTest(path=path):
                self.assertEqual(digest(frozen(path)), expected)
        ev = c["upstream_evidence_not_dispatch_authority"]
        for key, value in ev.items():
            if key.endswith("_path"):
                stem = key[:-5]
                self.assertEqual(digest((ROOT / value).read_bytes()),
                                 ev[stem + "_sha256"])
        self.assertFalse(ev["old_r2_ratification_covers_r10_label"])
        self.assertFalse(ev["current_stage_draft_is_accepted_result"])

    def test_exact_draft_chain(self):
        c, b, p = self.contract, self.base, self.pair
        runbook = (PACKAGE / "RUNBOOK.md").read_text()
        for key in ("request", "base_binding", "pair_binding", "public20m_eval_plan"):
            item = c[key]
            path = ROOT / item["path"]
            self.assertEqual(path.parent, PACKAGE)
            self.assertEqual(digest(path.read_bytes()), item["sha256"])
            self.assertIn(item["sha256"], runbook)
        self.assertEqual(b["request_sha256"], c["request"]["sha256"])
        self.assertEqual(p["request_sha256"], c["request"]["sha256"])
        self.assertEqual(p["base_binding_sha256"], c["base_binding"]["sha256"])
        self.assertEqual(b["monitor_script_sha256"],
                         c["source_sha256"]["research/v2/disk-monitor/root_free_guard_monitor.py"])
        self.assertEqual(p["wrapper_script_sha256"],
                         c["source_sha256"]["research/v2/disk-monitor/paired_root_free_guard_monitor.py"])

    def test_all_authority_remains_false(self):
        c, r, b, p, e = self.contract, self.request, self.base, self.pair, self.eval
        self.assertFalse(c["host_action_authorized"])
        self.assertFalse(c["native_dispatch_authorized"])
        self.assertFalse(c["fresh_public20m_eval_authorized"])
        self.assertFalse(r["dispatch_authorized"])
        self.assertFalse(e["evaluation_authorized"])
        self.assertEqual((b["package_state"], p["package_state"]),
                         ("STATIC_ONLY", "STATIC_ONLY"))
        self.assertTrue(all(value is False for value in c["required_gates"].values()))
        for key, value in c.items():
            if key.endswith("_sha256") and key not in ("source_sha256",):
                self.assertIsNone(value)
        self.assertIsNone(e["checkpoint_sha256"])
        self.assertFalse(list(PACKAGE.glob("*authorization*.json")))
        self.assertTrue(r["state"].startswith("DISABLED_"))
        self.assertTrue(e["state"].startswith("DISABLED_"))
        # Frozen launcher refuses before any local dispatch receipt/SSH path.
        source = frozen("research/fresh_start_launch.py").decode()
        self.assertLess(source.index('request.get("dispatch_authorized") is not True'),
                        source.index('receipt = ROOT / "research/run-logs"'))
        validate = isolated_function(source, "validate_request")
        with self.assertRaisesRegex(RuntimeError, "not authorized"):
            validate(r)

    def test_r10_label_checkout_host_and_monitor_binding(self):
        c, r, b, p, e = self.contract, self.request, self.base, self.pair, self.eval
        self.assertEqual((c["science_label"], r["label"], b["label"],
                          p["label"], e["training_label"]), (LABEL,) * 5)
        self.assertEqual((c["future_source_checkout"], r["remote"],
                          b["remote"], p["remote"], e["source_checkout"]),
                         (CHECKOUT,) * 5)
        self.assertEqual(c["future_monitor_target"], MONITOR)
        self.assertEqual((r["host"], b["expected_hostname"],
                          p["expected_hostname"]), ("trn22", "ip-172-31-32-253",
                                                    "ip-172-31-32-253"))
        self.assertEqual((b["expected_uid"], p["expected_uid"]), (1000, 1000))
        self.assertEqual(r["monitor_interlock"]["pair_binding_path"],
                         p["base_binding_path"].replace(".base-binding.json",
                                                        ".pair-binding.json"))
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

    def test_frozen_r10_launcher_accepts_only_exact_child_profile(self):
        r = self.request
        source = frozen("research/fresh_start_launch.py").decode()
        local = isolated_function(source, "bounded_timeout_command")
        remote_module = ast.parse(source)
        remote_string = next(
            node.value.value for node in remote_module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "REMOTE_SCRIPT"
                    for t in node.targets)
        )
        remote = isolated_function(remote_string, "bounded_timeout_command")
        for validator in (local, remote):
            self.assertTrue(validator(r["command"], r["host"], r["label"],
                                      r["remote"], r["environment"]))
            wrong = list(r["command"])
            wrong[0] = "timeout"
            self.assertFalse(validator(wrong, r["host"], r["label"],
                                       r["remote"], r["environment"]))
            wrong = list(r["command"])
            wrong.append("--extra")
            self.assertFalse(validator(wrong, r["host"], r["label"],
                                       r["remote"], r["environment"]))
            wrong_env = dict(r["environment"], PYTHONOPTIMIZE="1")
            self.assertFalse(validator(r["command"], r["host"], r["label"],
                                       r["remote"], wrong_env))
        proposal = read_json(ROOT / "research/v2/authorizations/20260925-trn22-c5c-a0-r2-proposal-ratification.json")
        train_argv = r["command"][r["command"].index("train.py") + 1:]
        self.assertEqual(train_argv, proposal["train_argv"])
        queue = read_json(ROOT / "research/v2/queue.json")
        self.assertEqual(shlex.split(queue["base_command"])[-len(train_argv):],
                         train_argv)
        self.assertEqual((r["command"][:7], r["command"][-2:]),
                         (["/usr/bin/timeout", "--foreground", "--signal=TERM",
                           "--kill-after=60s", "3600s", "/usr/bin/env", "-i"],
                          ["--matrix-lr", "0.015"]))
        self.assertEqual((r["environment"]["NEURON_LOGICAL_NC_CONFIG"],
                          r["environment"]["NEURON_COMPETITION_R1_SEED"],
                          r["environment"]["NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS"]),
                         ("2", "42", "1800"))

    def test_disk_and_fresh_public20m_eval(self):
        c, r, b, p, e = self.contract, self.request, self.base, self.pair, self.eval
        t = c["safety_thresholds_bytes"]
        self.assertEqual((r["min_free_bytes"], b["floor_free_bytes"],
                          t["immutable_operational_floor"]), (6979321856,) * 3)
        self.assertEqual((r["pre_launch_required_free_bytes"],
                          b["prelaunch_free_bytes"],
                          t["minimum_prelaunch_root_free"]), (10737418240,) * 3)
        self.assertEqual((b["stop_free_bytes"], p["stop_free_bytes"],
                          t["paired_monitor_stop"]), (9126805504,) * 3)
        self.assertLessEqual(b["poll_seconds"], t["monitor_poll_max_seconds"])
        self.assertTrue(r["external_boundary_observer"]["must_start_before_guard"])
        self.assertEqual(r["external_boundary_observer"]["command"][3], LABEL)
        self.assertEqual(
            digest((ROOT / "research/a0_boundary_observer.py").read_bytes()),
            r["external_boundary_observer"]["script_sha256"],
        )
        g1 = read_json(ROOT / "research/v2/g1.json")
        calibration = read_json(
            ROOT / "research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn22-r1/dispatch/dispatch.json"
        )["request"]
        expected = [
            x.replace("prepare.py", CHECKOUT + "/prepare.py")
             .replace("<absolute-train.py>", CHECKOUT + "/train.py")
             .replace("<absolute-final.pt>", e["checkpoint_path"])
            for x in g1["evaluation"]["command_template"]
        ]
        self.assertEqual(e["command"][:5],
                         ["/usr/bin/timeout", "--foreground", "--signal=TERM",
                          "--kill-after=60s", "1800s"])
        self.assertEqual(e["command"][5:], expected)
        env = dict(calibration["environment"])
        env["PYTHONPATH"] = CHECKOUT
        self.assertEqual(e["environment"], env)
        self.assertEqual((e["context_tokens"], e["eval_tokens"],
                          e["expected_num_bytes"]),
                         (1024, 20971520, 83462186))
        self.assertTrue(e["fresh_process"])
        self.assertEqual((e["expected_source_commit"], e["cwd"]), (COMMIT, CHECKOUT))
        self.assertEqual((e["train_sha256"], e["prepare_sha256"]),
                         (r["expected_sha256"]["train.py"],
                          r["expected_sha256"]["prepare.py"]))


if __name__ == "__main__":
    unittest.main()
