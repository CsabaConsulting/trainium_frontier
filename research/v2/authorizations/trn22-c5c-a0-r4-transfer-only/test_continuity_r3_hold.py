"""Controller-local synthetic safety tests; no SSH, host or key-content access."""

import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_continuity_r3_hold as hold
import stage_r4_remote as remote
import stage_r4_transport as transport
import test_stage_r4_transfer as fixtures


class ContinuityR3HoldTests(unittest.TestCase):
    def test_exact_rebindings_still_refuse_every_host_action(self):
        binding = hold.inspect_hold()
        self.assertFalse(binding["dispatch_authorized"])
        self.assertEqual(binding["remote_sha256"], hold.REMOTE_SHA256)
        self.assertEqual(binding["transport_sha256"], hold.TRANSPORT_SHA256)
        with (mock.patch.object(hold.core, "_run_ssh",
                                side_effect=AssertionError("SSH reached")),
              mock.patch.object(hold.core, "Journal",
                                side_effect=AssertionError("journal reached")),
              mock.patch.object(hold.transport, "checked_ssh_identity",
                                side_effect=AssertionError("key metadata reached")),
              mock.patch.object(hold.transport, "checked_local_source",
                                side_effect=AssertionError("source reached"))):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                hold.run_once()

    def test_request_or_code_drift_fails_before_any_dispatch_path(self):
        with mock.patch.object(hold, "REMOTE_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                hold.inspect_hold()
        with tempfile.TemporaryDirectory() as directory:
            forged = Path(directory) / "request.json"
            forged.write_bytes(hold.REQUEST.read_bytes().replace(
                b'"dispatch_authorized": false', b'"dispatch_authorized": true'))
            with mock.patch.object(hold, "REQUEST", forged):
                with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                    hold.inspect_hold()

    def test_file_backed_git_is_no_follow_and_fully_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "old-checkout"
            checkout.mkdir()
            git = checkout / ".git"
            git.write_text("gitdir: /unused/local-fixture\n", encoding="ascii")
            git_meta = remote._safe_path(str(git))
            head = remote._safe_path(str(git / "HEAD"))
            self.assertEqual(git_meta["type"], "regular")
            self.assertEqual(head, {"exists": False, "file_backed_git": git_meta})

        fixed_git = next(path for path in remote._old_paths()
                         if path.endswith("_r1/.git"))
        pre = fixtures.probe()
        pre["old_inventory"]["paths"][fixed_git] = git_meta
        pre["old_inventory"]["paths"][fixed_git + "/HEAD"] = head
        transport.validate_probe(pre, before_write=True)
        tampered = copy.deepcopy(pre)
        tampered["old_inventory"]["paths"][fixed_git + "/HEAD"][
            "file_backed_git"]["inode"] += 1
        with self.assertRaisesRegex(RuntimeError, "file-backed old .git inventory mismatch"):
            transport.validate_probe(tampered, before_write=True)
        unclassified = copy.deepcopy(pre)
        unclassified["old_inventory"]["paths"][fixed_git + "/HEAD"] = {
            "exists": False}
        with self.assertRaisesRegex(RuntimeError, "file-backed old .git inventory mismatch"):
            transport.validate_probe(unclassified, before_write=True)

    def test_symlink_and_unexpected_file_ancestors_do_not_hide_as_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "checkout"
            checkout.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (checkout / ".git").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "not a regular file"):
                remote._safe_path(str(checkout / ".git" / "HEAD"))
            (checkout / "out").write_text("not a directory", encoding="ascii")
            with self.assertRaisesRegex(RuntimeError, "non-directory inventory ancestor"):
                remote._safe_path(str(checkout / "out" / "unexpected"))

    def test_complete_synthetic_probe_emits_json_with_old_git_file(self):
        pre = fixtures.probe()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r1"
            checkout.mkdir()
            (checkout / ".git").write_text("gitdir: /unused/local-fixture\n",
                                            encoding="ascii")
            parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            real_safe_path = remote._safe_path

            def synthetic_safe_path(path):
                if path.startswith(str(root) + "/"):
                    return real_safe_path(path)
                return {"exists": False}

            def synthetic_fuser(path):
                key = ("checkpoint_fuser" if path == remote.CHECKPOINT else
                       "neuron0_fuser" if path == "/dev/neuron0" else
                       "shared_lock_fuser")
                return copy.deepcopy(pre[key])

            try:
                with (mock.patch.object(remote, "ROOT", str(root)),
                      mock.patch.object(remote, "_open_parent",
                                        side_effect=lambda: (os.dup(parent_fd),
                                                             pre["identity_after"]["ancestry"])),
                      mock.patch.object(remote, "_identity",
                                        return_value=pre["identity_after"]),
                      mock.patch.object(remote, "_safe_path",
                                        side_effect=synthetic_safe_path),
                      mock.patch.object(remote, "_protected",
                                        return_value=pre["protected"]),
                      mock.patch.object(remote, "_fuser", side_effect=synthetic_fuser),
                      mock.patch.object(remote, "_processes",
                                        return_value=pre["processes"]),
                      mock.patch.object(remote, "_neuron_ls",
                                        return_value=pre["neuron_ls"]),
                      mock.patch.object(remote, "_label_scan",
                                        return_value=pre["science_label_scan"]),
                      mock.patch.object(remote, "_process_identity",
                                        return_value=pre["process_identity"])):
                    receipt = remote.probe()
                old = receipt["old_inventory"]["paths"]
                self.assertTrue(receipt["old_inventory"]["catalog_complete"])
                self.assertEqual(old[str(checkout / ".git" / "HEAD")],
                                 {"exists": False,
                                  "file_backed_git": old[str(checkout / ".git")]})
                self.assertGreater(len(json.dumps(receipt)), 1000)
            finally:
                os.close(parent_fd)

    def test_all_late_busy_signals_block_the_sole_o_excl_create(self):
        pre = fixtures.probe()
        binding = transport.write_binding(pre)
        identity = pre["identity_after"]
        payload = b"local-synthetic-late-busy-fixture"
        for signal in ("checkpoint", "lock", "neuron0", "neuron_ls",
                       "label", "STOP", "process"):
            with self.subTest(signal=signal), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                target = parent / remote.TARGET_NAME
                busy = {"value": False}
                calls = {"identity": 0, "live_safety": 0}

                def identity_at_boundary(_parent_fd, _ancestry):
                    calls["identity"] += 1
                    if calls["identity"] == 2:
                        busy["value"] = True
                    return identity

                def path_state(path):
                    if path == remote.TARGET:
                        return {"exists": target.exists()}
                    return {"exists": bool(busy["value"] and signal == "STOP" and
                                             path == remote.STOP_PATHS[0])}

                def fuser(path):
                    key = ("checkpoint_fuser" if path == remote.CHECKPOINT else
                           "neuron0_fuser" if path == "/dev/neuron0" else
                           "shared_lock_fuser")
                    result = copy.deepcopy(pre[key])
                    if busy["value"] and signal == {
                            "checkpoint_fuser": "checkpoint",
                            "neuron0_fuser": "neuron0",
                            "shared_lock_fuser": "lock"}[key]:
                        result["pids"] = [777]
                        result["rc"] = 0
                    return result

                def neuron_ls():
                    result = copy.deepcopy(pre["neuron_ls"])
                    if busy["value"] and signal == "neuron_ls":
                        result["non_diagnostic_pids"] = [777]
                    return result

                def label_scan():
                    result = copy.deepcopy(pre["science_label_scan"])
                    if busy["value"] and signal == "label":
                        result["matches"] = [remote.ROOT + "/" + remote.LABEL]
                        result["match_count"] = 1
                    return result

                def processes():
                    result = {"matched": [], "unreadable_cmdlines": 0}
                    if busy["value"] and signal == "process":
                        result["matched"] = [{"pid": 777, "markers": ["train.py"]}]
                    return result

                real_live_safety = remote._live_prewrite_safety

                def count_live_safety():
                    calls["live_safety"] += 1
                    return real_live_safety()

                parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY |
                                    os.O_NOFOLLOW)
                try:
                    with (mock.patch.object(remote, "_open_parent",
                                            side_effect=lambda: (os.dup(parent_fd),
                                                                 identity["ancestry"])),
                          mock.patch.object(remote, "_identity",
                                            side_effect=identity_at_boundary),
                          mock.patch.object(remote, "_process_identity",
                                            return_value={"pid": 456,
                                                          "start_ticks": 789}),
                          mock.patch.object(remote, "_safe_path",
                                            side_effect=path_state),
                          mock.patch.object(remote, "_fuser", side_effect=fuser),
                          mock.patch.object(remote, "_neuron_ls",
                                            side_effect=neuron_ls),
                          mock.patch.object(remote, "_label_scan",
                                            side_effect=label_scan),
                          mock.patch.object(remote, "_processes",
                                            side_effect=processes),
                          mock.patch.object(remote, "_live_prewrite_safety",
                                            side_effect=count_live_safety)):
                        with self.assertRaises(RuntimeError):
                            remote.write(io.BytesIO(payload), binding)
                    self.assertTrue(busy["value"])
                    self.assertEqual(calls["live_safety"], 2)
                    self.assertFalse(target.exists(),
                                     "late " + signal + " activity created the target")
                finally:
                    os.close(parent_fd)

    def test_transport_requires_clean_final_live_snapshot_not_just_initial(self):
        pre = fixtures.probe()
        written = fixtures.receipt("write", pid=456, preflight=pre)
        transport.compare_preflight_write(pre, written)
        changed = copy.deepcopy(written)
        changed["prewrite_evidence"]["last_live_precreate_safety"][
            "neuron0_fuser"]["pids"] = [777]
        with self.assertRaisesRegex(RuntimeError, "Neuron/label safety receipt"):
            transport.compare_preflight_write(pre, changed)
        missing = copy.deepcopy(written)
        missing["prewrite_evidence"].pop("last_live_precreate_safety")
        with self.assertRaises(RuntimeError):
            transport.compare_preflight_write(pre, missing)


if __name__ == "__main__":
    unittest.main()
