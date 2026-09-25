"""Offline adversarial checks. Every transport and remote action is synthetic."""

import copy
import datetime
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    item = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(item)
    return item


LOCAL = module("cpu_g2_final_local", HERE / "cpu_g2_invocation_final.py")
REMOTE = module("cpu_g2_final_remote", HERE / "remote_cpu_g2_once.py")


def reply(returncode=0, *, body=None, stderr=""):
    return SimpleNamespace(returncode=returncode,
                           stdout=json.dumps(body) if body is not None else "",
                           stderr=stderr)


class CPUOnlyFinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = LOCAL.build_plan()
        base = LOCAL.REPO / "research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-result"
        cls.stage = json.loads((base / "CHECKOUT_STAGE_RECEIPT_20260925T031732Z.json").read_text())
        cls.post = json.loads((base / "UNCONDITIONAL_POSTFLIGHT_20260925T031738Z.json").read_text())

    def snapshot(self, *, after=False, receipt_sha="0" * 64):
        old = copy.deepcopy(self.post)
        old["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        source = LOCAL.SOURCE
        checkout_stat = copy.deepcopy(self.stage["checkout"]["checkout_stat"])
        checkout_stat.update({"exists": True, "type": "directory"})
        train = copy.deepcopy(self.stage["checkout"]["source_files"]["train.py"])
        train.update({"exists": True, "type": "regular"})
        prepare = copy.deepcopy(self.stage["checkout"]["source_files"]["prepare.py"])
        prepare.update({"exists": True, "type": "regular"})
        scratch = {path: {"exists": False} for path in (
            REMOTE.ROOT, REMOTE.PROCESS_DIR, REMOTE.STEP_LOG, REMOTE.CPU_RECEIPT)}
        if after:
            scratch[REMOTE.ROOT] = {"exists": True, "type": "directory",
                                    "device": 66305, "uid": 1000,
                                    "mode": "0o700"}
            scratch[REMOTE.PROCESS_DIR] = {"exists": True, "type": "directory",
                                           "device": 66305, "uid": 1000,
                                           "mode": "0o700"}
            scratch[REMOTE.CPU_RECEIPT] = {"exists": True, "type": "regular",
                                           "device": 66305, "uid": 1000,
                                           "sha256": receipt_sha}
        return {
            "schema": "trn21-c5c-stage-r2-cpu-g2-full-readonly-v1",
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "old": old, "source_checkout": checkout_stat,
            "source_train": train, "source_prepare": prepare,
            "git": {"head": LOCAL.SOURCE_COMMIT + "\n", "count": "1\n",
                    "shallow": "true\n", "status": "", "remote": ""},
            "tools": {name: copy.deepcopy(self.post["targets"][
                self.plan["request"]["persistent_cpu_tools"][name + "_path"]])
                for name in ("preflight", "contained")},
            "scratch": scratch, "cpu_label_matches": [],
        }

    def test_offline_exact_plan_pins_cpu_only_commands_and_missing_authority(self):
        plan = self.plan
        self.assertEqual(plan["request"]["g2_options"],
                         ["--steps", "3", "--nproc", "4", "--timeout", "900", "--keep"])
        self.assertEqual(plan["request"]["cpu_label"], REMOTE.LABEL)
        self.assertEqual(plan["request"]["cpu_root"], REMOTE.ROOT)
        self.assertEqual(plan["binding"]["request_sha256"], plan["request_sha256"])
        self.assertEqual(LOCAL.argv_sha(plan["audit_argv"]),
                         plan["binding"]["full_audit_ssh_argv_sha256"])
        self.assertEqual(LOCAL.argv_sha(plan["run_argv"]),
                         plan["binding"]["one_cpu_run_ssh_argv_sha256"])
        self.assertEqual(LOCAL.sha(plan["stdin_payload"].encode()),
                         plan["binding"]["stdin_payload_sha256"])
        self.assertEqual(plan["audit_argv"][0:2], ["ssh", "-T"])
        self.assertIn("StrictHostKeyChecking=yes", plan["audit_argv"])
        self.assertFalse(plan["request"]["native_training_authorized"])
        self.assertFalse(plan["request"]["evaluation_authorized"])
        self.assertFalse(LOCAL.FINAL_REVIEW.exists())
        self.assertFalse(LOCAL.ASSIGNMENT.exists())

    def test_direct_execution_and_absent_actual_files_hold_without_marker_or_host(self):
        result = subprocess.run([sys.executable, str(HERE / "cpu_g2_invocation_final.py")],
                                capture_output=True, text=True, timeout=10,
                                check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DISPATCH HOLD", result.stderr)
        with self.assertRaisesRegex(RuntimeError, "actual review or assignment file absent"):
            LOCAL.authorize_actual_files(self.plan)

    def authority_fixture(self, directory):
        root = Path(directory)
        rel = Path("research/v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-final")
        path = root / rel
        path.mkdir(parents=True)
        review = path / "INDEPENDENT_FINAL_CPU_G2_REVIEW.PASS.json"
        assignment = path / "SUPERVISOR_CPU_ONLY_ASSIGNMENT.json"
        marker = root / "cpu-g2-one-attempt.jsonl"
        with patch.object(LOCAL, "MARKER", marker):
            review_raw = LOCAL.canonical(LOCAL.expected_review(self.plan)) + b"\n"
        review.write_bytes(review_raw)
        review.chmod(0o600)
        with patch.object(LOCAL, "MARKER", marker):
            assignment_raw = LOCAL.canonical(
                LOCAL.expected_assignment(self.plan, LOCAL.sha(review_raw))) + b"\n"
        assignment.write_bytes(assignment_raw)
        assignment.chmod(0o600)
        return root, review, assignment, marker

    def test_actual_file_chain_rejects_claimed_hash_forgery_and_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            root, review, assignment, marker = self.authority_fixture(directory)
            with patch.object(LOCAL, "REPO", root), \
                    patch.object(LOCAL, "FINAL_REVIEW", review), \
                    patch.object(LOCAL, "ASSIGNMENT", assignment), \
                    patch.object(LOCAL, "MARKER", marker):
                exact = LOCAL.authorize_actual_files(self.plan)
                self.assertEqual(exact["review_sha256"], LOCAL.sha(review.read_bytes()))
                bad = json.loads(assignment.read_text())
                bad["independent_final_review_sha256"] = "0" * 64
                assignment.write_bytes(LOCAL.canonical(bad) + b"\n")
                with self.assertRaisesRegex(RuntimeError, "assignment"):
                    LOCAL.authorize_actual_files(self.plan)
                bad["independent_final_review_sha256"] = LOCAL.sha(review.read_bytes())
                assignment.write_bytes(LOCAL.canonical(bad) + b"\n")
                review.unlink()
                with self.assertRaisesRegex(RuntimeError, "absent"):
                    LOCAL.authorize_actual_files(self.plan)

    def test_actual_review_wrong_verdict_or_symlink_never_authorizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, review, assignment, marker = self.authority_fixture(directory)
            with patch.object(LOCAL, "REPO", root), \
                    patch.object(LOCAL, "FINAL_REVIEW", review), \
                    patch.object(LOCAL, "ASSIGNMENT", assignment), \
                    patch.object(LOCAL, "MARKER", marker):
                data = json.loads(review.read_text())
                data["verdict"] = "HOLD"
                raw = LOCAL.canonical(data) + b"\n"
                review.write_bytes(raw)
                a = json.loads(assignment.read_text())
                a["independent_final_review_sha256"] = LOCAL.sha(raw)
                assignment.write_bytes(LOCAL.canonical(a) + b"\n")
                with self.assertRaisesRegex(RuntimeError, "not exact PASS"):
                    LOCAL.authorize_actual_files(self.plan)
                target = root / "fake-review.json"
                review.rename(target)
                review.symlink_to(target)
                with self.assertRaisesRegex(RuntimeError, "unsafe actual authority file"):
                    LOCAL.authorize_actual_files(self.plan)

    def test_fresh_snapshot_accepts_reviewed_source_and_rejects_drift(self):
        snapshot = self.snapshot()
        REMOTE.validate_snapshot(snapshot, self.plan["request"],
                                 self.plan["checkout_request"], phase="before")
        mutations = (
            ("source train hash", lambda s: s["source_train"].__setitem__("sha256", "0" * 64)),
            ("tool inode", lambda s: s["tools"]["preflight"].__setitem__("inode", 0)),
            ("boot B", lambda s: s["old"].__setitem__("boot_id", "different-boot")),
            ("host STOP", lambda s: s["old"]["stops"][REMOTE.STOP_PATHS[0]].__setitem__("exists", True)),
            ("new root", lambda s: s["scratch"][REMOTE.ROOT].__setitem__("exists", True)),
            ("CPU label", lambda s: s.__setitem__("cpu_label_matches", [REMOTE.LABEL])),
            ("disk floor", lambda s: s["old"].__setitem__("root_free_bytes", 1)),
            ("process", lambda s: s["old"]["processes"].__setitem__("hits", [{"pid": 7}])),
            ("checkpoint hash", lambda s: s["old"]["checkpoints"][0].__setitem__("sha256", "0" * 64)),
        )
        for title, mutate in mutations:
            with self.subTest(title=title):
                changed = copy.deepcopy(snapshot)
                mutate(changed)
                with self.assertRaises(RuntimeError):
                    REMOTE.validate_snapshot(changed, self.plan["request"],
                                             self.plan["checkout_request"],
                                             phase="before")

    def test_remote_payload_cannot_change_cpu_shape_or_enable_eval(self):
        payload = json.loads(self.plan["stdin_payload"])
        clean = REMOTE.check_payload(payload)
        self.assertEqual(clean["commands"]["launch_argv"][2], "launch")
        for target, field, value in (("request", "native_training_authorized", True),
                                     ("request", "g2_options", ["--steps", "4"]),
                                     ("commands", "launch_argv", ["python3", "train.py"])):
            with self.subTest(target=target, field=field):
                changed = copy.deepcopy(payload)
                if target == "request":
                    request = json.loads(changed["request_text"])
                    request[field] = value
                    changed["request_text"] = json.dumps(request)
                else:
                    changed[target][field] = value
                with self.assertRaises(RuntimeError):
                    REMOTE.check_payload(changed)

    def test_preexec_bridge_blocks_boot_or_stop_between_probe_and_spawn(self):
        request = self.plan["request"]
        checkout = self.plan["checkout_request"]
        with patch.object(REMOTE, "current_identity", side_effect=[
                ("boot-A", "mnt-A", "mount-A"), ("boot-B", "mnt-A", "mount-A")]), \
                patch.object(REMOTE, "stat_path", return_value={"exists": False}):
            with self.assertRaises(RuntimeError):
                REMOTE.bridge_gate(request, checkout, scratch_created=False)
        expected = ("boot-A", "mnt-A", "mount-A")
        def stat_for(path, *, hash_file=False):
            if path in REMOTE.STOP_PATHS:
                return {"exists": path == REMOTE.STOP_PATHS[0]}
            if path == REMOTE.SOURCE:
                return {"exists": True, "type": "directory", "device": 66305,
                        "inode": 3168319, "uid": 1000, "mode": "0o700"}
            if path.endswith("/train.py"):
                return {"exists": True, "type": "regular", "device": 66305,
                        "inode": 3174933, "uid": 1000,
                        "sha256": request["source_train_sha256"]}
            if path.endswith("/prepare.py"):
                return {"exists": True, "type": "regular", "device": 66305,
                        "inode": 3168417, "uid": 1000,
                        "sha256": request["source_prepare_sha256"]}
            for name in ("preflight", "contained"):
                tools = request["persistent_cpu_tools"]
                if path == tools[name + "_path"]:
                    return {"exists": True, "type": "regular", "device": 66305,
                            "inode": tools[name + "_inode"], "uid": 1000,
                            "mode": "0o600", "nlink": 1,
                            "bytes": tools[name + "_bytes"],
                            "sha256": tools[name + "_sha256"]}
            return {"exists": False}
        with patch.object(REMOTE, "current_identity", return_value=expected), \
                patch.object(REMOTE, "stat_path", side_effect=stat_for):
            with self.assertRaisesRegex(RuntimeError, "STOP"):
                REMOTE.bridge_gate(request, checkout, scratch_created=False)

    def test_one_attempt_marker_precedes_ssh_and_replay_is_impossible(self):
        with tempfile.TemporaryDirectory() as directory:
            root, review, assignment, marker = self.authority_fixture(directory)
            before = self.snapshot()
            after = self.snapshot(after=True)
            calls = []
            def transport(argv, *, input_text, timeout_seconds):
                self.assertTrue(marker.exists(), "one-attempt marker must precede SSH")
                calls.append(argv)
                if len(calls) == 2:
                    return reply(returncode=2, body={"status": "HOLD_PRESERVE_PARTIAL_NO_RETRY"})
                return reply(body={"status": "READ_ONLY_G2_AUDIT",
                                   "snapshot": before if len(calls) == 1 else after})
            with patch.object(LOCAL, "REPO", root), \
                    patch.object(LOCAL, "FINAL_REVIEW", review), \
                    patch.object(LOCAL, "ASSIGNMENT", assignment), \
                    patch.object(LOCAL, "MARKER", marker), \
                    patch.object(LOCAL, "build_plan", return_value=self.plan), \
                    patch.object(LOCAL, "local_breakers", return_value="clear"):
                result = LOCAL.operator_once(transport)
                self.assertEqual(result["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
                with self.assertRaises(FileExistsError):
                    LOCAL.operator_once(transport)
            self.assertEqual(calls, [self.plan["audit_argv"], self.plan["run_argv"],
                                     self.plan["audit_argv"]])
            self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
            events = [json.loads(line)["event"] for line in marker.read_text().splitlines()]
            self.assertEqual(events[0], "EXACT_CPU_ONLY_ASSIGNMENT_BOUND")
            self.assertIn("SSH_TRANSPORT", events)
            self.assertIn("SEPARATE_FULL_POSTFLIGHT_RECORDED", events)
            self.assertEqual(events[-1], "TERMINAL_HOLD")

    def test_local_controller_stop_and_breakers_block_even_with_clear_peer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "research/v2/state.json"
            control.parent.mkdir(parents=True)
            control.write_text(json.dumps({"schema": "trn-frontier-state-v2",
                                           "breakers": {
                                               "global_correctness": {"tripped": False},
                                               "trn21": {"tripped": False}}}))
            control.chmod(0o664)  # Existing controller state mode is 0664.
            with patch.object(LOCAL, "REPO", root), patch.object(LOCAL, "STATE", control):
                self.assertEqual(LOCAL.local_breakers(), LOCAL.sha(control.read_bytes()))
                stop = root / "research/v2/STOP"
                stop.write_text("STOP")
                with self.assertRaisesRegex(RuntimeError, "local STOP"):
                    LOCAL.local_breakers()
                stop.unlink()
                control.write_text(json.dumps({"schema": "trn-frontier-state-v2",
                                               "breakers": {
                                                   "global_correctness": {"tripped": False},
                                                   "trn21": {"tripped": True}}}))
                with self.assertRaisesRegex(RuntimeError, "breaker"):
                    LOCAL.local_breakers()

    def test_preflight_exception_still_attempts_separate_postflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root, review, assignment, marker = self.authority_fixture(directory)
            calls = []
            def transport(argv, *, input_text, timeout_seconds):
                self.assertTrue(marker.exists())
                calls.append(argv)
                if len(calls) == 1:
                    raise RuntimeError("synthetic preflight transport exception")
                return reply(body={"status": "READ_ONLY_G2_AUDIT",
                                   "snapshot": self.snapshot()})
            with patch.object(LOCAL, "REPO", root), \
                    patch.object(LOCAL, "FINAL_REVIEW", review), \
                    patch.object(LOCAL, "ASSIGNMENT", assignment), \
                    patch.object(LOCAL, "MARKER", marker), \
                    patch.object(LOCAL, "build_plan", return_value=self.plan), \
                    patch.object(LOCAL, "local_breakers", return_value="clear"):
                result = LOCAL.operator_once(transport)
            self.assertEqual(result["status"], "HOLD_PRESERVE_PARTIAL_NO_RETRY")
            self.assertEqual(calls, [self.plan["audit_argv"], self.plan["audit_argv"]])


if __name__ == "__main__":
    unittest.main()
