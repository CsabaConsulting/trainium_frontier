"""Local-only counterexample for the trn21 r5 enabled transfer writer.

The r5 request reuses the exact r4 remote payload. This test deliberately
observes its unsafe behavior: a STOP introduced during the final label scan
is present before O_EXCL, but the writer still creates the target. The only
files written are disposable fixtures under /dev/shm.
"""

import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PACKAGE = Path(os.environ.get(
    "TRN21_R5_SUBJECT_PACKAGE",
    Path(__file__).resolve().parents[1] / "authorizations" /
    "trn21-c5c-a0-r4-transfer-only",
))
sys.path.insert(0, str(PACKAGE))
import stage_r4_remote_final_r4 as remote
import test_operator_transfer_final_r3_once as fixtures


MOUNT_PROOF = {
    "mount_namespace": "mnt:[synthetic]",
    "mountinfo_sha256": "a" * 64,
    "entry_count": 1,
    "root_mount_id": "27",
    "root_mount_point": "/",
    "root_mount_root": "/",
    "root_major_minor": "259:1",
    "root_filesystem": "ext4",
    "root_source": "/dev/root",
    "unsafe_aliases": [],
}


class StopDuringFinalLabelScanReproducer(unittest.TestCase):
    def test_persistent_stop_during_final_label_scan_still_creates_target(self):
        self.assertEqual(
            hashlib.sha256((PACKAGE / "stage_r4_remote_final_r4.py").read_bytes()).hexdigest(),
            "ad30d93b464175c6df1550654076bc868a8bd8cd652fedd08bd5bfe91f055e2b",
        )
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            stop_file = root / "late-STOP"
            activity = {"active": False, "kind": "STOP"}
            with (mock.patch.object(fixtures, "remote", remote),
                  mock.patch.object(remote, "_lock_mount_proof", return_value=MOUNT_PROOF)):
                with fixtures.simulated_writer(root, activity) as (
                        pin, lock_pin, target, _lock, _calls):
                    scan = remote._label_scan

                    def insert_stop_during_scan():
                        stop_file.write_text("STOP", encoding="ascii")
                        activity["active"] = True
                        return scan()

                    # Zero bytes let the local fixture complete the write and
                    # return a receipt without streaming the pinned real capsule.
                    with (mock.patch.object(remote, "_label_scan",
                                            side_effect=insert_stop_during_scan),
                          mock.patch.object(remote, "BYTES", 0),
                          mock.patch.object(remote, "SHA256",
                                            hashlib.sha256(b"").hexdigest())):
                        receipt = remote.write(io.BytesIO(b""), pin, lock_pin)
                    self.assertTrue(stop_file.is_file())
                    self.assertTrue(target.is_file())
                    self.assertEqual(receipt["status"],
                                     "HASH_VERIFIED_IN_THIS_SESSION_ONLY")
                    self.assertTrue(all(not item["exists"] for item in
                                        receipt["final_creation_gate"]["stops"].values()))
                    self.assertEqual(
                        receipt["final_creation_gate"]["science_label_scan"]["match_count"],
                        0,
                    )


if __name__ == "__main__":
    unittest.main()
