"""Independent local-only adversarial checks for the disabled trn21 r4 draft.

The synthetic writer uses private temporary files. No test contacts trn21,
opens SSH, reads PEM contents, or grants transfer authority.
"""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operator_transfer_once as operator
import stage_r4_remote as remote
import stage_r4_transport as transport
import test_stage_r4_transfer as fixtures


class IndependentTransferNegativeTests(unittest.TestCase):
    def test_frozen_hold_stops_before_marker_journal_or_ssh(self):
        request = json.loads(operator.REQUEST.read_bytes())
        self.assertIs(request["dispatch_authorized"], False)
        self.assertIs(request["source_checkout_authorized"], False)
        self.assertIs(request["native_run_authorized"], False)
        with (mock.patch.object(operator, "_consume_fixed_marker",
                                side_effect=AssertionError("marker reached")),
              mock.patch.object(operator, "Journal",
                                side_effect=AssertionError("journal reached")),
              mock.patch.object(operator.subprocess, "Popen",
                                side_effect=AssertionError("SSH reached"))):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()

    def test_literal_trn21_host_key_and_ssh_options_are_locally_pinned(self):
        identity = transport.checked_ssh_identity()
        self.assertEqual(identity["known_hosts_sha256"],
                         "e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3")
        self.assertEqual(identity["host_fingerprint"],
                         "SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc")
        argv = transport.build_ssh_argv("probe")
        self.assertEqual(argv[:4], ["ssh", "-F", "/dev/null", "-T"])
        self.assertEqual(argv[-2], transport.ENDPOINT)
        self.assertEqual(shlex.split(argv[-1])[-1], "probe")
        for option in ("StrictHostKeyChecking=yes", "IdentitiesOnly=yes",
                       "HostKeyAlgorithms=ssh-ed25519",
                       "GlobalKnownHostsFile=/dev/null",
                       "UserKnownHostsFile=" + str(transport.KNOWN_HOSTS)):
            self.assertIn(option, argv)
        with mock.patch.object(transport, "HOST_FINGERPRINT", "SHA256:wrong"):
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                transport.build_ssh_argv("probe")

    def test_pre_post_boot_a_cannot_mask_writer_boot_or_mount_b(self):
        pre = fixtures.probe(pid=101, offset=-3)
        archive, write, verify = fixtures.transfer_receipts(pre)
        post = fixtures.probe(pid=303, offset=1, archive=archive)
        self.assertEqual(transport.compare_postflight(pre, post)["status"],
                         "READ_ONLY_POSTFLIGHT_PASS")
        for key, value in (("boot_id", "boot-B"),
                           ("mount_namespace", "mnt:[other]")):
            with self.subTest(key=key):
                changed_write, changed_verify = copy.deepcopy((write, verify))
                writer_probe = changed_write["prewrite_probe"]
                for side in ("identity_before", "identity_after"):
                    writer_probe[side][key] = value
                    changed_write[side][key] = value
                    changed_verify[side][key] = value
                with self.assertRaisesRegex(RuntimeError, "preflight-to-write"):
                    transport.compare_sessions(pre, changed_write, changed_verify)

    def test_remote_boot_mismatch_blocks_before_parent_or_target_open(self):
        pre = fixtures.probe(pid=101, offset=-3)
        changed = fixtures.probe(pid=202, offset=-2, boot="boot-B")
        pin = remote._identity_digest(pre["identity_after"])
        with (mock.patch.object(remote, "probe", return_value=changed),
              mock.patch.object(remote, "_open_parent",
                                side_effect=AssertionError("parent opened"))):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                remote.write(io.BytesIO(b""), pin)

    def test_stop_process_and_neuron_appearing_after_full_probe_block_create(self):
        pre = fixtures.probe(pid=101, offset=-3)
        writer_probe = fixtures.probe(pid=202, offset=-2)
        current = copy.deepcopy(pre["identity_after"])
        pin = remote._identity_digest(current)
        clear_holder = {"path_exists": True, "rc": 1,
                        "pids": [], "stderr_empty": True}
        for case, expected in (("STOP", "STOP appeared at final"),
                               ("process", "process appeared at final"),
                               ("Neuron", "Neuron or shared lock appeared")):
            with self.subTest(case=case), tempfile.TemporaryDirectory(dir=HERE) as directory:
                active = {"late": False}
                target = Path(directory) / remote.TARGET_NAME

                def parent():
                    return os.open(directory, os.O_RDONLY | os.O_DIRECTORY), current["ancestry"]

                def path_metadata(path):
                    if (case == "STOP" and active["late"] and
                            path == remote.STOP_PATHS[0]):
                        return {"exists": True, "type": "regular"}
                    return {"exists": False}

                def processes():
                    return {"matched": ([{"pid": 777, "markers": ["train.py"]}]
                                        if case == "process" and active["late"] else []),
                            "unreadable_cmdlines": 0}

                def neuron():
                    active["late"] = True
                    if case == "Neuron":
                        return {**writer_probe["neuron_ls"],
                                "non_diagnostic_pids": [777]}
                    return writer_probe["neuron_ls"]

                with (mock.patch.object(remote, "probe", return_value=writer_probe),
                      mock.patch.object(remote, "_open_parent", side_effect=parent),
                      mock.patch.object(remote, "_identity", return_value=current),
                      mock.patch.object(remote, "_safe_path", side_effect=path_metadata),
                      mock.patch.object(remote, "_processes", side_effect=processes),
                      mock.patch.object(remote, "_fuser", return_value=clear_holder),
                      mock.patch.object(remote, "_neuron_ls", side_effect=neuron)):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        remote.write(io.BytesIO(b""), pin)
                self.assertFalse(target.exists())

    def test_exact_target_success_is_exclusive_and_partial_is_not_retried(self):
        current = fixtures.identity()
        pin = remote._identity_digest(current)
        payload = b"independent-local-payload"
        for partial in (False, True):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory(dir=HERE) as directory:
                target = Path(directory) / remote.TARGET_NAME

                def parent():
                    return os.open(directory, os.O_RDONLY | os.O_DIRECTORY), current["ancestry"]

                with (mock.patch.object(remote, "BYTES", len(payload) + int(partial)),
                      mock.patch.object(remote, "SHA256", hashlib.sha256(payload).hexdigest()),
                      mock.patch.object(remote, "probe", return_value={}),
                      mock.patch.object(remote, "_prewrite_safety",
                                        return_value={"identity_after": current}),
                      mock.patch.object(remote, "_open_parent", side_effect=parent),
                      mock.patch.object(remote, "_identity", return_value=current),
                      mock.patch.object(remote, "_immediate_activity_gate", return_value={}),
                      mock.patch.object(remote, "_safe_path",
                                        return_value={"exists": False})):
                    if partial:
                        with self.assertRaisesRegex(RuntimeError, "partial target retained"):
                            remote.write(io.BytesIO(payload), pin)
                    else:
                        self.assertEqual(remote.write(io.BytesIO(payload), pin)[
                            "readback_sha256"], hashlib.sha256(payload).hexdigest())
                    self.assertEqual(target.read_bytes(), payload)
                    self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
                    with self.assertRaises(FileExistsError):
                        remote.write(io.BytesIO(payload), pin)
                    self.assertEqual(target.read_bytes(), payload)

    def test_pinned_e65_checkpoints_old_files_and_caches_reject_drift(self):
        good = fixtures.probe()
        transport.validate_probe(good, before_write=True)
        mutations = []
        first_checkpoint = next(iter(remote.CHECKPOINTS))
        first_prior = next(iter(remote.E65_TRANSFER_FILES))
        mutations.append(("checkpoint", lambda p: p["protected"]["checkpoints"][
            first_checkpoint].__setitem__("sha256", "0" * 64)))
        mutations.append(("prior stage", lambda p: p["protected"]["e65_transfer_files"][
            first_prior].__setitem__("sha256", "0" * 64)))
        mutations.append(("e65 train", lambda p: p["protected"]["e65_source"][
            "train.py"].__setitem__("sha256", "0" * 64)))
        mutations.append(("e65 checkout", lambda p: p["protected"][
            "e65_checkout"].__setitem__("inode", 1)))
        mutations.append(("cache root", lambda p: p["protected"][
            "cache_root"].__setitem__("inode", 1)))
        mutations.append(("cache child", lambda p: p["protected"][
            "six_caches"]["compiler"].__setitem__("inode", 1)))
        for name, change in mutations:
            with self.subTest(name=name):
                bad = copy.deepcopy(good)
                change(bad)
                with self.assertRaises(RuntimeError):
                    transport.validate_probe(bad, before_write=True)

    def test_fixed_marker_and_journal_are_private_single_use(self):
        with tempfile.TemporaryDirectory(dir=operator.MARKER.parent) as directory:
            marker = Path(directory) / "fixed.marker"
            journal_path = Path(directory) / "fixed.jsonl"
            with (mock.patch.object(operator, "MARKER", marker),
                  mock.patch.object(operator, "JOURNAL", journal_path),
                  mock.patch.object(operator, "_read_evidence", return_value=b"fixture")):
                marker_sha = operator._consume_fixed_marker({"one_attempt": True})
                self.assertEqual(marker_sha, hashlib.sha256(marker.read_bytes()).hexdigest())
                self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
                with self.assertRaises(FileExistsError):
                    operator._consume_fixed_marker({"one_attempt": True})
                journal = operator.Journal()
                journal.emit("INDEPENDENT_LOCAL_TEST")
                journal.close()
                self.assertEqual(stat.S_IMODE(journal_path.stat().st_mode), 0o600)
                with self.assertRaises(FileExistsError):
                    operator.Journal()

    def test_failed_write_attempts_full_postflight_and_records_both_holds(self):
        pre = fixtures.probe(pid=101, offset=-3)
        post = fixtures.probe(pid=303, offset=1)
        post["protected"]["six_caches"]["neff"]["inode"] += 1
        calls, events = [], []

        class Journal:
            def emit(self, event, **details):
                events.append(event)

            def close(self):
                events.append("CLOSED")

        def offline_ssh(operation, journal, *, preflight=None):
            calls.append(operation)
            if len(calls) == 1:
                return pre
            if operation == "write":
                self.assertIs(preflight, pre)
                raise RuntimeError("synthetic remote partial write")
            return post

        with (mock.patch.object(operator, "authorize", return_value=(
                    {}, {"request_sha256": "b" * 64})),
              mock.patch.object(transport, "checked_local_source", return_value={}),
              mock.patch.object(transport, "checked_ssh_identity", return_value={}),
              mock.patch.object(operator, "_check_controller_breakers", return_value={}),
              mock.patch.object(operator, "_consume_fixed_marker", return_value="a" * 64),
              mock.patch.object(operator, "_pinned_marker", return_value=None),
              mock.patch.object(operator, "_read_evidence", return_value=b"fixture"),
              mock.patch.object(operator, "Journal", Journal),
              mock.patch.object(operator, "_run_ssh", side_effect=offline_ssh)):
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                operator.run_once()
        self.assertEqual(calls, ["probe", "write", "probe"])
        self.assertIn("TRANSFER_HOLD", events)
        self.assertIn("POSTFLIGHT_HOLD", events)
        self.assertNotIn("CROSS_SESSION_PERSISTENCE_PASS", events)
        self.assertEqual(events[-1], "CLOSED")


if __name__ == "__main__":
    unittest.main()
