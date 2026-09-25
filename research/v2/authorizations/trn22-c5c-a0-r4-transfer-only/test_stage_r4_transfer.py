"""Local synthetic tests only: never connect to trn22 or open the SSH key."""

import datetime
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
import stage_r4_remote as remote
import stage_r4_transport as transport
import operator_transfer_once as operator


def utc(offset_seconds=0):
    return (datetime.datetime.now(datetime.timezone.utc) +
            datetime.timedelta(seconds=offset_seconds)).isoformat()


def meta(kind="directory", *, device=66305, inode=123, uid=1000, mode="0o750",
         size=4096, nlink=1):
    return {"exists": True, "type": kind, "device": device, "inode": inode,
            "uid": uid, "gid": uid, "mode": mode, "nlink": nlink,
            "bytes": size}


def identity(free=30_000_000_000):
    return {"hostname": "ip-172-31-32-253", "uid": 1000,
            "boot_id": "synthetic-boot", "mount_namespace": "mnt:[synthetic]",
            "mount": {"filesystem": "ext4", "source": "/dev/root",
                      "mount_id": "27", "major_minor": "259:1", "point": "/"},
            "block_aliases": {
                "mount_major_minor": "259:1",
                "historical_names": {
                    "/dev/root": {"exists": False, "mount_source_string_only": True},
                    "/dev/nvme0n1p1": {"exists": True,
                                        "resolved_rdev_major_minor": "259:1"}},
                "sysfs_link": "../../devices/pci/nvme0n1p1",
                "sysfs_realpath": "/sys/devices/pci/nvme0n1p1"},
            "ancestry": {
                "root": {"type": "directory", "device": 66305, "inode": 2,
                         "uid": 0, "gid": 0, "mode": "0o755"},
                "home": {"type": "directory", "device": 66305, "inode": 1684,
                         "uid": 0, "gid": 0, "mode": "0o755"},
                "ubuntu": {"type": "directory", "device": 66305, "inode": 295711,
                           "uid": 1000, "gid": 1000, "mode": "0o750"},
            },
            "root_free_bytes": free, "root_free_inodes": 100_000}


def probe(*, archive=None):
    old_paths = {path: {"exists": False} for path in remote._old_paths()}
    old_paths[remote.CHECKPOINT] = meta("regular", inode=2417091, mode="0o664",
                                        size=469831009)
    old_paths[remote.CACHE_ROOT] = meta(inode=2430419, mode="0o775")
    target_paths = (remote.TARGET, remote.CHECKOUT, remote.MONITOR,
                    remote.CHECKOUT + "/out", remote.CHECKOUT + "/out/" + remote.LABEL,
                    remote.CHECKOUT + "/out/" + remote.LABEL + "-work",
                    remote.CHECKOUT + "/research/run-logs/" + remote.LABEL + ".log",
                    remote.CHECKOUT + "/research/run-logs/" + remote.LABEL + ".steps.jsonl",
                    remote.CHECKOUT + "/research/run-logs/" + remote.LABEL + ".process.json",
                    remote.CHECKOUT + "/research/run-logs/" + remote.LABEL + ".postflight.json",
                    remote.OBSERVER + "/" + remote.LABEL + ".observer.json",
                    remote.OBSERVER + "/" + remote.LABEL + ".observer.stderr.log")
    result = {
        "schema": remote.SCHEMA, "operation": "probe", "utc": utc(),
        "process_identity": {"pid": 123, "start_ticks": 456},
        "identity_before": identity(), "identity_after": identity(),
        "old_inventory": {"paths": old_paths, "top_level_matching_sources": {},
                          "catalog_complete": True},
        "targets": {path: {"exists": False} for path in target_paths},
        "stops": {path: {"exists": False} for path in remote.STOP_PATHS + [
            remote.CHECKOUT + "/research/agentic/STOP",
            remote.CHECKOUT + "/research/v2/STOP"]},
        "protected": {
            "checkpoint": {**old_paths[remote.CHECKPOINT],
                           "read_bytes": 469831009,
                           "sha256": remote.CHECKPOINT_SHA256},
            "cache_root": old_paths[remote.CACHE_ROOT],
            "six_caches": {name: meta(inode=inode, mode="0o775")
                           for name, inode in remote.CACHE_INODES.items()}},
        "processes": {"matched": [], "unreadable_cmdlines": 0},
        "science_label_scan": {"rc": 0, "match_count": 0, "matches": [],
                               "truncated": False, "stderr_empty": True},
        "checkpoint_fuser": {"path_exists": True, "rc": 1, "pids": [],
                             "stderr_empty": True},
        "shared_lock_fuser": {"path_exists": True, "rc": 1, "pids": [],
                              "stderr_empty": True},
        "neuron0_fuser": {"path_exists": True, "rc": 1, "pids": [],
                          "stderr_empty": True},
        "neuron_ls": {"rc": 0, "json_valid": True, "schema_valid": True,
                      "device_count": 1, "non_diagnostic_pids": [],
                      "stderr_empty": True},
    }
    if archive is not None:
        result["targets"][remote.TARGET] = archive
    return result


def receipt(operation, *, pid, utc_seconds=0):
    archive = meta("regular", inode=700777, mode="0o600", size=remote.BYTES)
    return {"schema": remote.SCHEMA, "operation": operation,
            "status": "HASH_VERIFIED_IN_THIS_SESSION_ONLY",
            "utc": utc(utc_seconds),
            "process_identity": {"pid": pid, "start_ticks": pid * 10},
            "path": remote.TARGET, "expected_bytes": remote.BYTES,
            "expected_sha256": remote.SHA256,
            "streamed_sha256": remote.SHA256 if operation == "write" else None,
            "readback_sha256": remote.SHA256, "target_stat": archive,
            "identity_before": identity(), "identity_after": identity()}


class TransferOnlyTests(unittest.TestCase):
    def test_frozen_request_refuses_before_ssh(self):
        with mock.patch.object(operator.subprocess, "Popen", side_effect=AssertionError("SSH opened")):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()

    def test_reviewed_source_and_exact_command(self):
        self.assertEqual(transport.REMOTE_SHA256,
                         transport.sha256(transport.REMOTE.read_bytes()))
        for operation in ("probe", "write", "verify"):
            argv = transport.build_ssh_argv(operation)
            self.assertEqual(argv[0:4], ["ssh", "-F", "/dev/null", "-T"])
            self.assertEqual(argv[-2], transport.ENDPOINT)
            self.assertIn("UserKnownHostsFile=" + str(transport.KNOWN_HOSTS), argv)
            self.assertIn("GlobalKnownHostsFile=/dev/null", argv)
            self.assertIn("HostKeyAlgorithms=ssh-ed25519", argv)
            self.assertEqual(shlex.split(argv[-1]),
                             ["exec", "python3", "-c", transport.reviewed_remote_source(),
                              operation])
        with self.assertRaises(ValueError):
            transport.build_ssh_argv("checkout")

    def test_local_capsule_exact_bytes_and_digest(self):
        source = transport.checked_local_source()
        self.assertEqual(source["bytes"], remote.BYTES)
        self.assertEqual(source["sha256"], remote.SHA256)

    def test_pinned_local_host_identity_and_hash_drift_hold(self):
        host = transport.checked_ssh_identity()
        self.assertEqual(host["host_fingerprint"], transport.HOST_FINGERPRINT)
        self.assertEqual(host["known_hosts_sha256"], transport.KNOWN_HOSTS_SHA256)
        with mock.patch.object(transport, "KNOWN_HOSTS_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "known-hosts bytes changed"):
                transport.build_ssh_argv("probe")

    def test_local_stop_blocks_clear_breakers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.json"
            state.write_text(json.dumps({"breakers": {
                "global_correctness": {"tripped": False},
                "trn22": {"tripped": False}}}), encoding="utf-8")
            (root / "STOP").write_text("local stop", encoding="utf-8")
            with mock.patch.object(operator, "CONTROL_STATE", state):
                with self.assertRaisesRegex(RuntimeError, "local STOP"):
                    operator._check_controller_breakers()

    def test_disabled_request_cannot_be_flag_flipped(self):
        old = json.loads(operator.REQUEST.read_text(encoding="utf-8"))
        old["dispatch_authorized"] = True
        with tempfile.TemporaryDirectory() as directory:
            forged = Path(directory) / "forged.json"
            forged.write_text(json.dumps(old), encoding="utf-8")
            with mock.patch.object(operator, "REQUEST", forged):
                with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                    operator.authorize()

    def test_fake_review_and_unbound_assignment_hold(self):
        request = json.loads(operator.REQUEST.read_text(encoding="utf-8"))
        request["dispatch_authorized"] = True
        request_bytes = json.dumps(request).encode("utf-8")
        fake_review = json.dumps({"verdict": "PASS", "reviewer_role": "independent_verifier"}).encode("utf-8")
        assignment = json.dumps({"dispatch_authorized": True, "one_attempt": True}).encode("utf-8")
        with self.assertRaisesRegex(RuntimeError, "structured PASS"):
            operator._validate_authority_bundle(request_bytes, fake_review, assignment)
        operator_sha = operator._sha(Path(operator.__file__).read_bytes())
        transport_sha = operator._sha(transport.REMOTE.parent.joinpath(
            "stage_r4_transport.py").read_bytes())
        bound_review = {"schema": "trn22-r4-independent-transfer-enable-review-v1",
                        "verdict": "PASS", "reviewer_role": "independent_verifier",
                        "request_sha256": operator._sha(request_bytes),
                        "operator_sha256": operator_sha,
                        "transport_sha256": transport_sha,
                        "remote_sha256": transport.REMOTE_SHA256,
                        "capsule_sha256": transport.SOURCE_SHA256,
                        "source_commit": request["source_commit"],
                        "host": "trn22", "scope": request["assignment_scope"],
                        "no_host_action_by_reviewer": True}
        with self.assertRaisesRegex(RuntimeError, "assignment mismatch"):
            operator._validate_authority_bundle(request_bytes,
                                                json.dumps(bound_review).encode("utf-8"),
                                                assignment)
        review_bytes = json.dumps(bound_review).encode("utf-8")
        bound_assignment = {
            "schema": "trn22-r4-supervisor-transfer-only-one-attempt-v1",
            "issuer_role": "supervisor", "dispatch_authorized": True,
            "scope": request["assignment_scope"], "host": "trn22",
            "request_sha256": operator._sha(request_bytes),
            "independent_review_sha256": operator._sha(review_bytes),
            "operator_sha256": operator_sha,
            "transport_sha256": transport_sha,
            "remote_sha256": transport.REMOTE_SHA256,
            "source_commit": request["source_commit"],
            "capsule_sha256": transport.SOURCE_SHA256,
            "target": transport.TARGET, "journal": str(operator.JOURNAL),
            "one_attempt": True,
        }
        self.assertEqual(operator._validate_authority_bundle(
            request_bytes, review_bytes, json.dumps(bound_assignment).encode("utf-8"))[1],
            bound_assignment)
        bound_assignment["independent_review_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "assignment mismatch"):
            operator._validate_authority_bundle(
                request_bytes, review_bytes, json.dumps(bound_assignment).encode("utf-8"))
        bound_review["request_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "structured PASS"):
            operator._validate_authority_bundle(
                request_bytes, json.dumps(bound_review).encode("utf-8"), assignment)

    def test_block_alias_mismatch_holds_preflight(self):
        bad = probe()
        bad["identity_before"]["block_aliases"]["historical_names"][
            "/dev/nvme0n1p1"]["resolved_rdev_major_minor"] = "259:2"
        bad["identity_after"] = bad["identity_before"]
        with self.assertRaisesRegex(RuntimeError, "aliases have not been reconciled"):
            transport.validate_probe(bad, before_write=True)

    def test_remote_alias_proof_uses_kernel_rdev_not_path_spelling(self):
        class Block:
            st_mode = stat.S_IFBLK | 0o600
            st_rdev = os.makedev(259, 1)
            st_dev = 5
            st_ino = 99
        def named(path):
            if path == "/dev/root":
                raise FileNotFoundError(path)
            return Block()
        with (mock.patch.object(remote.os, "lstat", side_effect=named),
              mock.patch.object(remote.os, "stat", return_value=Block()),
              mock.patch.object(remote.os, "readlink", return_value="../../devices/nvme0n1p1"),
              mock.patch.object(remote.os.path, "realpath",
                                return_value="/sys/devices/pci/nvme0n1p1")):
            reconciled = remote._block_aliases({"major_minor": "259:1"})
            self.assertEqual(reconciled["historical_names"]["/dev/nvme0n1p1"]
                             ["resolved_rdev_major_minor"], "259:1")
        class WrongBlock(Block):
            st_rdev = os.makedev(259, 2)
        with (mock.patch.object(remote.os, "lstat", side_effect=named),
              mock.patch.object(remote.os, "stat", return_value=WrongBlock())):
            with self.assertRaisesRegex(RuntimeError, "different block device"):
                remote._block_aliases({"major_minor": "259:1"})

    def test_synthetic_full_gate_and_cross_session(self):
        before = probe()
        archive = meta("regular", inode=700777, mode="0o600", size=remote.BYTES)
        after = probe(archive=archive)
        write, verify = receipt("write", pid=1001), receipt("verify", pid=1002, utc_seconds=1)
        transport.validate_probe(before, before_write=True)
        transport.validate_probe(after, before_write=False)
        persistence = transport.compare_sessions(write, verify)
        self.assertEqual(transport.compare_postflight(before, after, persistence=persistence)
                         ["status"], "READ_ONLY_POSTFLIGHT_PASS")

    def test_wrong_old_inventory_and_busy_host_hold(self):
        missing = probe()
        missing["old_inventory"]["paths"].pop(remote.CHECKPOINT)
        with self.assertRaisesRegex(RuntimeError, "inventory incomplete"):
            transport.validate_probe(missing, before_write=True)
        busy = probe()
        busy["neuron_ls"]["non_diagnostic_pids"] = [400]
        with self.assertRaisesRegex(RuntimeError, "Neuron busy"):
            transport.validate_probe(busy, before_write=True)
        occupied = probe()
        occupied["targets"][remote.TARGET] = meta("regular")
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            transport.validate_probe(occupied, before_write=True)

    def test_cross_session_must_be_distinct_and_pinned(self):
        write, verify = receipt("write", pid=1001), receipt("verify", pid=1001, utc_seconds=1)
        with self.assertRaisesRegex(RuntimeError, "distinct"):
            transport.compare_sessions(write, verify)
        verify["process_identity"]["pid"] = 1002
        verify["expected_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "exact capsule"):
            transport.compare_sessions(write, verify)

    def test_fixed_journal_is_one_attempt_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.jsonl"
            with mock.patch.object(operator, "JOURNAL", path):
                journal = operator.Journal()
                journal.emit("ONE_ATTEMPT_CONSUMED", assignment_sha256="a" * 64)
                journal.close()
                with self.assertRaises(FileExistsError):
                    operator.Journal()
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["event"], "ONE_ATTEMPT_CONSUMED")

    def test_failed_write_still_does_distinct_read_only_postflight(self):
        seen = []
        events = []
        class FakeJournal:
            def emit(self, event, **details):
                events.append(event)
            def close(self):
                events.append("CLOSED")
        pre = probe()
        post = probe()
        post["process_identity"] = {"pid": 999, "start_ticks": 9990}
        def fake_ssh(operation, journal):
            seen.append(operation)
            if len(seen) == 1:
                return pre
            if operation == "write":
                raise RuntimeError("synthetic partial write")
            return post
        with (mock.patch.object(operator, "authorize", return_value=(
                    {}, {"request_sha256": "b" * 64})),
              mock.patch.object(operator.transport, "checked_local_source",
                                return_value={"sha256": remote.SHA256}),
              mock.patch.object(operator, "_check_controller_breakers", return_value={}),
              mock.patch.object(operator, "_read_evidence", return_value=b"fixture-assignment"),
              mock.patch.object(operator, "Journal", FakeJournal),
              mock.patch.object(operator, "_run_ssh", side_effect=fake_ssh)):
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                operator.run_once()
        self.assertEqual(seen, ["probe", "write", "probe"])
        self.assertIn("FINAL_READ_ONLY_POSTFLIGHT", events)
        self.assertIn("TRANSFER_HOLD", events)

    def test_stop_appearing_after_preflight_prevents_write_and_postflights(self):
        calls = []
        checks = 0
        class FakeJournal:
            def emit(self, event, **details):
                pass
            def close(self):
                pass
        def breaker_check():
            nonlocal checks
            checks += 1
            if checks == 2:
                raise RuntimeError("local STOP appeared")
            return {}
        def ssh(operation, journal):
            calls.append(operation)
            return probe()
        with (mock.patch.object(operator, "authorize", return_value=(
                    {}, {"request_sha256": "b" * 64})),
              mock.patch.object(operator.transport, "checked_local_source",
                                return_value={"sha256": remote.SHA256}),
              mock.patch.object(operator, "_check_controller_breakers", side_effect=breaker_check),
              mock.patch.object(operator, "_read_evidence", return_value=b"fixture-assignment"),
              mock.patch.object(operator, "Journal", FakeJournal),
              mock.patch.object(operator, "_run_ssh", side_effect=ssh)):
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                operator.run_once()
        self.assertEqual(calls, ["probe", "probe"])


if __name__ == "__main__":
    unittest.main()
