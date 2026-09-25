"""Verifier-only offline regressions against b80c69c4; no SSH or host action.

These tests state the intended safety gates. They are expected to fail on the
reviewed candidate, documenting why no final PASS receipt may be issued.
"""

import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_r4_remote as remote
import stage_r4_transport as transport
import test_stage_r4_transfer as fixtures


class FinalTransferNegativeTests(unittest.TestCase):
    def test_preflight_write_postflight_identity_chain_rejects_intermediate_boot(self):
        pre = fixtures.probe()
        write = fixtures.receipt("write", pid=1001)
        verify = fixtures.receipt("verify", pid=1002, utc_seconds=1)
        post = fixtures.probe(archive=write["target_stat"])

        transport.validate_probe(pre, before_write=True)
        transport.validate_probe(post, before_write=False)
        for receipt in (write, verify):
            for side in ("identity_before", "identity_after"):
                receipt[side]["boot_id"] = "different-boot"
        persistence = transport.compare_sessions(write, verify)

        # The valid pre/post probes are boot A while both transfer receipts
        # are boot B. Every adjacent session must be from the same boot.
        with self.assertRaisesRegex(RuntimeError, "boot|identity|session"):
            transport.compare_postflight(pre, post, persistence=persistence)

    def test_remote_stop_after_probe_blocks_target_creation(self):
        payload = b"offline verifier transfer fixture"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new-capsule.bin"
            stop = Path(directory) / "STOP"
            stop.write_text("STOP", encoding="ascii")
            parent_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            identity = {
                "hostname": remote.HOST, "uid": remote.UID,
                "boot_id": "fixture", "mount_namespace": "mnt:[fixture]",
                "mount": {"source": "/dev/root"}, "block_aliases": {},
                "ancestry": {"ubuntu": {"inode": 1}},
                "root_free_bytes": 30_000_000_000, "root_free_inodes": 100_000,
            }
            try:
                with (mock.patch.object(remote, "TARGET", str(target)),
                      mock.patch.object(remote, "TARGET_NAME", target.name),
                      mock.patch.object(remote, "STOP_PATHS", [str(stop)]),
                      mock.patch.object(remote, "BYTES", len(payload)),
                      mock.patch.object(remote, "SHA256", hashlib.sha256(payload).hexdigest()),
                      mock.patch.object(remote, "PRE_FREE", 0),
                      mock.patch.object(remote, "POST_FREE", 0),
                      mock.patch.object(remote, "_open_parent",
                                        side_effect=lambda: (os.dup(parent_fd), {})),
                      mock.patch.object(remote, "_identity", return_value=identity),
                      mock.patch.object(remote, "_safe_path",
                                        side_effect=lambda path: {"exists": os.path.lexists(path)})):
                    try:
                        remote.write(io.BytesIO(payload))
                    except RuntimeError as error:
                        self.assertIn("STOP", str(error))
                    else:
                        self.fail("remote STOP was present but writer created target=%s" %
                                  target.exists())
                    self.assertFalse(target.exists(), "STOP must prevent target creation")
            finally:
                os.close(parent_fd)


if __name__ == "__main__":
    unittest.main()
