"""Independent offline boundary witnesses for disabled trn22 r9 stage r3.

No SSH, AWS, Neuron device, checkpoint, or real host path is touched.  These
tests deliberately retain the candidate's real first-open/write and rollback
logic while replacing host probes with disposable synthetic observations.
"""

from contextlib import nullcontext
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "authorizations/trn22-c5c-a0-r9-incremental-stage-r3-hold"
SPEC = importlib.util.spec_from_file_location("independent_trn22_r9_r3", SUBJECT / "remote_incremental.py")
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)


class IndependentR3BoundaryTests(unittest.TestCase):
    def test_ordinary_empty_provisional_rollback_removes_only_own_name(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r3-rollback-") as temp:
            root = Path(temp)
            bundle = root / "own.bundle"
            parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            fd = os.open(bundle, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with mock.patch.object(remote, "BUNDLE", bundle):
                    remote._rollback_unpopulated_bundle(fd, parent)
                self.assertFalse(bundle.exists())
            finally:
                os.close(fd)
                os.close(parent)

    def test_nonempty_provisional_is_preserved_on_rollback_refusal(self):
        with tempfile.TemporaryDirectory(prefix="trn22-r3-nonempty-") as temp:
            root = Path(temp)
            bundle = root / "own.bundle"
            parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            fd = os.open(bundle, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd, b"partial")
                with mock.patch.object(remote, "BUNDLE", bundle):
                    with self.assertRaisesRegex(RuntimeError, "changed"):
                        remote._rollback_unpopulated_bundle(fd, parent)
                self.assertEqual(bundle.read_bytes(), b"partial")
            finally:
                os.close(fd)
                os.close(parent)

    def test_same_uid_name_swap_between_stat_and_unlink_deletes_replacement(self):
        """Witness: a different inode can be unlinked after identity validation.

        The original empty inode is renamed aside after the candidate's
        os.stat, then a separate nonempty inode takes its former name before
        the candidate's os.unlink.  The candidate deletes the replacement.
        This is inside a disposable directory; it is not a host action.
        """
        with tempfile.TemporaryDirectory(prefix="trn22-r3-swap-") as temp:
            root = Path(temp)
            bundle = root / "own.bundle"
            retained = root / "original-retained"
            parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            fd = os.open(bundle, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            real_unlink = os.unlink
            replacement_inode = []

            def swap_then_unlink(name, *, dir_fd=None):
                self.assertEqual(name, bundle.name)
                os.rename(bundle, retained)
                bundle.write_bytes(b"someone else's replacement")
                replacement_inode.append(os.lstat(bundle).st_ino)
                return real_unlink(name, dir_fd=dir_fd)

            try:
                with mock.patch.object(remote, "BUNDLE", bundle), \
                     mock.patch.object(remote.os, "unlink", side_effect=swap_then_unlink):
                    remote._rollback_unpopulated_bundle(fd, parent)
                self.assertTrue(retained.exists())
                self.assertFalse(bundle.exists())
                self.assertNotEqual(os.fstat(fd).st_ino, replacement_inode[0])
            finally:
                os.close(fd)
                os.close(parent)

    def _synthetic_write(self, *, arrival=None, delayed_open=False,
                         delayed_payload=False,
                         partial_payload=False):
        with tempfile.TemporaryDirectory(prefix="trn22-r3-write-") as temp:
            root = Path(temp)
            bundle = root / "delta.bundle"
            target = root / "new-checkout"
            label = root / ("late-" + remote.LABEL)
            data = b"independent exact synthetic bundle"
            snapshot = {"schema": "synthetic", "bundle_exists": False,
                        "target_exists": False, "source": {"commit": "frozen"}}
            events = {"finds": 0, "fusers": 0, "payload_delay": 0.0,
                      "open_delay": 0.0, "process": False,
                      "neuron": False, "disk_low": False}
            real_open = os.open
            real_write = os.write

            def open_file(path, flags, *args, **kwargs):
                fd = real_open(path, flags, *args, **kwargs)
                if path == bundle.name and flags & os.O_EXCL and delayed_open:
                    started = time.monotonic()
                    time.sleep(1.1)
                    events["open_delay"] = time.monotonic() - started
                return fd

            def inject(kind):
                if kind == "label":
                    label.touch()
                elif kind == "stop":
                    (root / "STOP").touch()
                elif kind == "process":
                    events["process"] = True
                elif kind == "neuron":
                    events["neuron"] = True
                elif kind == "disk":
                    events["disk_low"] = True

            def stop_and_process():
                if (root / "STOP").exists():
                    raise RuntimeError("synthetic STOP observed")
                if events["process"]:
                    raise RuntimeError("synthetic process observed")

            def space(_minimum):
                if events["disk_low"]:
                    raise RuntimeError("synthetic root disk floor observed")
                return {"root_free_bytes": remote.PRE_FREE + 4096,
                        "root_free_inodes": remote.MIN_INODES + 10}

            def run(argv, **_kwargs):
                if argv[0] == "/usr/bin/find":
                    events["finds"] += 1
                    output = (str(label) + "\n").encode() if label.exists() else b""
                    if events["finds"] == 2 and arrival and arrival.startswith("pre_"):
                        inject(arrival[4:])
                    return types.SimpleNamespace(returncode=0, stdout=output, stderr=b"")
                if argv[0] == "/usr/bin/fuser":
                    events["fusers"] += 1
                    if arrival and arrival.startswith("post_") and events["fusers"] == 3:
                        self.assertEqual(events["finds"], 3)
                        inject(arrival[5:])
                    return types.SimpleNamespace(returncode=1, stdout=b"", stderr=b"")
                if argv[0] == "neuron-ls":
                    processes = b'[{"neuron_processes":[{"command":"train.py"}]}]' \
                        if events["neuron"] else b"[]"
                    return types.SimpleNamespace(returncode=0, stdout=processes, stderr=b"")
                self.fail("unexpected synthetic subprocess: " + repr(argv))

            def write(fd, block):
                if delayed_payload:
                    started = time.monotonic()
                    time.sleep(1.1)
                    events["payload_delay"] = time.monotonic() - started
                if partial_payload:
                    if not events.get("partial_written"):
                        events["partial_written"] = True
                        return real_write(fd, block[:3])
                    raise OSError("synthetic disk error after three payload bytes")
                return real_write(fd, block)

            with mock.patch.object(remote, "ROOT", root), \
                 mock.patch.object(remote, "BUNDLE", bundle), \
                 mock.patch.object(remote, "TARGET", target), \
                 mock.patch.object(remote, "BUNDLE_BYTES", len(data)), \
                 mock.patch.object(remote, "BUNDLE_SHA256", hashlib.sha256(data).hexdigest()), \
                 mock.patch.object(remote, "STOP_PATHS", [root / "STOP"]), \
                 mock.patch.object(remote, "_host_lock", return_value=nullcontext({})), \
                 mock.patch.object(remote, "_probe", return_value=snapshot), \
                 mock.patch.object(remote, "_stable"), \
                 mock.patch.object(remote, "_stop_and_process", side_effect=stop_and_process), \
                 mock.patch.object(remote.subprocess, "run", side_effect=run), \
                 mock.patch.object(remote, "_space", side_effect=space), \
                 mock.patch.object(remote, "_checked_bundle", return_value={}), \
                 mock.patch.object(remote, "_reconstruct", return_value={}), \
                 mock.patch.object(remote, "_source", return_value=snapshot["source"]), \
                 mock.patch.object(remote.os, "open", side_effect=open_file), \
                 mock.patch.object(remote.os, "write", side_effect=write):
                try:
                    receipt = remote.perform("write", prior={"synthetic": True}, data=data)
                    error = None
                except Exception as exc:
                    receipt, error = None, type(exc).__name__ + ": " + str(exc)
            return {**events, "label_exists": label.exists(),
                    "bundle_bytes": bundle.read_bytes() if bundle.exists() else None,
                    "target_exists": target.exists(), "receipt": receipt, "error": error}

    def test_partial_payload_is_retained_and_checkout_is_not_created(self):
        result = self._synthetic_write(partial_payload=True)
        self.assertIn("synthetic disk error", result["error"])
        self.assertEqual(result["bundle_bytes"], b"ind")
        self.assertFalse(result["target_exists"])

    def test_payload_can_start_more_than_one_second_after_successful_age_gate(self):
        result = self._synthetic_write(delayed_payload=True)
        self.assertIsNone(result["error"])
        self.assertGreaterEqual(result["payload_delay"], 1.1)
        self.assertTrue(result["target_exists"])
        self.assertLess(result["receipt"]["final_no_write_continuity"]
                        ["post_open_space_age_seconds"], 1.0)

    def test_delayed_exclusive_open_is_detected_and_empty_inode_rolled_back(self):
        result = self._synthetic_write(delayed_open=True)
        self.assertIn("too old", result["error"])
        self.assertGreaterEqual(result["open_delay"], 1.1)
        self.assertIsNone(result["bundle_bytes"])
        self.assertFalse(result["target_exists"])

    def test_label_arriving_during_post_open_fuser_is_not_scanned_again(self):
        result = self._synthetic_write(arrival="post_label")
        self.assertIsNone(result["error"])
        self.assertTrue(result["label_exists"])
        self.assertTrue(result["target_exists"])
        self.assertEqual(result["finds"], 3)
        self.assertEqual(result["fusers"], 4)

    def test_pre_open_persistent_stop_process_neuron_and_disk_all_refuse(self):
        for kind in ("stop", "process", "neuron", "disk"):
            with self.subTest(kind=kind):
                result = self._synthetic_write(arrival="pre_" + kind)
                self.assertIsNotNone(result["error"])
                self.assertIsNone(result["bundle_bytes"])
                self.assertFalse(result["target_exists"])

    def test_post_open_persistent_stop_process_neuron_and_disk_roll_back(self):
        for kind in ("stop", "process", "neuron", "disk"):
            with self.subTest(kind=kind):
                result = self._synthetic_write(arrival="post_" + kind)
                self.assertIsNotNone(result["error"])
                self.assertIsNone(result["bundle_bytes"])
                self.assertFalse(result["target_exists"])


if __name__ == "__main__":
    unittest.main()
