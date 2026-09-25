"""Independent local-only checks of the disabled r3 transfer repair.

All filesystem fixtures are temporary; no SSH, host, AWS or key-content access.
"""

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_continuity_r3_hold as hold
import operator_transfer_continuity_final_r2 as old_final
import operator_transfer_final_once as marker_support
import stage_r4_remote as remote
import stage_r4_transport as transport
import test_stage_r4_transfer as fixtures


class IndependentContinuityR3StaticTests(unittest.TestCase):
    def test_host_observed_r1_directory_r2_git_file_r3_absence_forms_full_inventory(self):
        # The archived 04:21 UTC read-only snapshot saw these three forms.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "trainium_freshstart_20260925_v2_c5c_a0_control_trn22_"
            r1, r2 = root / (name + "r1"), root / (name + "r2")
            r1.mkdir()
            r2.mkdir()
            (r1 / ".git").mkdir()
            (r1 / ".git" / "HEAD").write_bytes(b"1" * 41)
            (r2 / ".git").write_bytes(b"2" * 148)
            parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with mock.patch.object(remote, "ROOT", str(root)):
                    old = remote._old_inventory(parent_fd)
                    paths = old["paths"]
                    self.assertEqual(len(paths), 41)
                    self.assertTrue(old["catalog_complete"])
                    self.assertEqual(set(old["top_level_matching_sources"]),
                                     {str(r1), str(r2)})
                    self.assertEqual(paths[str(r1 / ".git" / "HEAD")]["type"],
                                     "regular")
                    self.assertEqual(paths[str(r2 / ".git")]["bytes"], 148)
                    self.assertEqual(paths[str(r2 / ".git" / "HEAD")],
                                     {"exists": False,
                                      "file_backed_git": paths[str(r2 / ".git")]})
                    self.assertEqual(paths[str(root / (name + "r3") / ".git")],
                                     {"exists": False})
                    pre = fixtures.probe()
                    old["paths"][remote.CHECKPOINT] = {
                        key: value for key, value in
                        pre["protected"]["checkpoint"].items()
                        if key not in ("read_bytes", "sha256")}
                    old["paths"][remote.CACHE_ROOT] = pre["protected"]["cache_root"]
                    pre["old_inventory"] = old
                    with mock.patch.object(transport, "remote_constants",
                                           return_value=remote):
                        self.assertIs(transport.validate_probe(pre,
                                                               before_write=True), pre)
            finally:
                os.close(parent_fd)

    def test_only_file_backed_git_head_uses_the_nondirectory_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "old-checkout"
            checkout.mkdir()
            (checkout / ".git").write_text("gitdir: /unused/local\n",
                                            encoding="ascii")
            self.assertFalse(remote._safe_path(str(checkout / ".git" / "HEAD"))[
                "exists"])
            with self.assertRaisesRegex(RuntimeError,
                                        "non-directory inventory ancestor"):
                remote._safe_path(str(checkout / ".git" / "config"))
            (checkout / ".git").unlink()
            (checkout / ".git").symlink_to(Path(directory),
                                           target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "not a regular file"):
                remote._safe_path(str(checkout / ".git" / "HEAD"))
            (checkout / "out").write_text("ordinary file", encoding="ascii")
            with self.assertRaisesRegex(RuntimeError,
                                        "non-directory inventory ancestor"):
                remote._safe_path(str(checkout / "out" / "nested"))

    def test_final_live_receipt_rejects_every_busy_or_missing_signal(self):
        pre = fixtures.probe()
        write = fixtures.receipt("write", pid=1001, preflight=pre)
        self.assertEqual(transport.compare_preflight_write(pre, write)["status"],
                         "PREFLIGHT_TO_WRITE_IDENTITY_AND_LIVE_SAFETY_PASS")
        mutations = (
            ("checkpoint", lambda snap: snap["checkpoint_fuser"].update(
                {"rc": 0, "pids": [777]})),
            ("lock", lambda snap: snap["shared_lock_fuser"].update(
                {"rc": 0, "pids": [777]})),
            ("neuron0", lambda snap: snap["neuron0_fuser"].update(
                {"rc": 0, "pids": [777]})),
            ("neuron_ls", lambda snap: snap["neuron_ls"].update(
                {"non_diagnostic_pids": [777]})),
            ("label", lambda snap: snap["science_label_scan"].update(
                {"match_count": 1, "matches": ["synthetic"]})),
            ("STOP", lambda snap: snap["stops"][remote.STOP_PATHS[0]].update(
                {"exists": True})),
            ("process", lambda snap: snap["processes"].update(
                {"unreadable_cmdlines": 1})),
        )
        for name, change in mutations:
            with self.subTest(signal=name):
                altered = copy.deepcopy(write)
                change(altered["prewrite_evidence"]["last_live_precreate_safety"])
                with self.assertRaises(RuntimeError):
                    transport.compare_preflight_write(pre, altered)
        absent = copy.deepcopy(write)
        absent["prewrite_evidence"].pop("last_live_precreate_safety")
        with self.assertRaises(RuntimeError):
            transport.compare_preflight_write(pre, absent)

    def test_new_and_historical_wrappers_stop_before_marker_journal_or_ssh(self):
        binding = hold.inspect_hold()  # Pins historical PASS/HOLD and current bytes.
        self.assertFalse(binding["dispatch_authorized"])
        with (mock.patch.object(marker_support, "_consume_fixed_marker",
                                side_effect=AssertionError("marker reached")),
              mock.patch.object(hold.core, "Journal",
                                side_effect=AssertionError("journal reached")),
              mock.patch.object(hold.core, "_run_ssh",
                                side_effect=AssertionError("SSH reached")),
              mock.patch.object(hold.transport, "checked_ssh_identity",
                                side_effect=AssertionError("SSH identity reached"))):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                hold.run_once()
            with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                old_final.run_once()
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                hold.core.run_once()


if __name__ == "__main__":
    unittest.main()
