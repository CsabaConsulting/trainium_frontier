"""Offline synthetic tests for the disabled trn21 r4 transfer package.

No test opens SSH, reads a private key, or contacts a Trainium host.
"""

import copy
import datetime
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
import stage_r4_remote as remote
import stage_r4_transport as transport
import operator_transfer_once as operator


def utc(offset=0):
    return (datetime.datetime.now(datetime.timezone.utc) +
            datetime.timedelta(seconds=offset)).isoformat()


def meta(kind="directory", *, inode=123, mode="0o750", size=4096, nlink=1,
         uid=1000, device=66305):
    return {"exists": True, "type": kind, "device": device, "inode": inode,
            "uid": uid, "gid": uid, "mode": mode, "nlink": nlink,
            "bytes": size}


def identity(*, boot="synthetic-boot", free=30_000_000_000):
    return {
        "hostname": "ip-172-31-34-85", "uid": 1000, "boot_id": boot,
        "mount_namespace": "mnt:[synthetic]",
        "mount": {"filesystem": "ext4", "source": "/dev/root", "mount_id": "27",
                  "major_minor": "259:1", "point": "/"},
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
                       "uid": 1000, "gid": 1000, "mode": "0o750"}},
        "root_free_bytes": free, "root_free_inodes": 100_000,
    }


def probe(*, pid=100, offset=-3, archive=None, boot="synthetic-boot"):
    old_paths = {path: {"exists": False} for path in remote._old_paths()}
    checkpoints = {}
    for path, (size, digest, inode) in remote.CHECKPOINTS.items():
        old_paths[path] = meta("regular", inode=inode, mode="0o664", size=size)
        checkpoints[path] = {**old_paths[path], "read_bytes": size, "sha256": digest}
    prior_files = {}
    for path, (size, digest, inode) in remote.E65_TRANSFER_FILES.items():
        old_paths[path] = meta("regular", inode=inode, mode="0o600", size=size)
        prior_files[path] = {**old_paths[path], "read_bytes": size, "sha256": digest}
    old_paths[remote.E65_CHECKOUT] = meta(inode=remote.E65_CHECKOUT_INODE,
                                          mode="0o700", nlink=10)
    e65_source = {}
    for name, (size, digest, inode) in remote.E65_SOURCE.items():
        path = remote.E65_CHECKOUT + "/" + name
        old_paths[path] = meta("regular", inode=inode,
                               mode="0o600", size=size)
        e65_source[name] = {**old_paths[path], "read_bytes": size, "sha256": digest}
    old_paths[remote.CACHE_ROOT] = meta(inode=remote.CACHE_ROOT_INODE,
                                        mode="0o775", nlink=8)
    current = identity(boot=boot)
    result = {
        "schema": remote.SCHEMA, "operation": "probe", "utc": utc(offset),
        "process_identity": {"pid": pid, "start_ticks": pid * 10},
        "identity_before": copy.deepcopy(current), "identity_after": copy.deepcopy(current),
        "old_inventory": {"paths": old_paths,
                          "top_level_matching_sources": {
                              remote.E65_CHECKOUT: old_paths[remote.E65_CHECKOUT]},
                          "catalog_complete": True},
        "targets": {path: {"exists": False} for path in remote._target_paths()},
        "stops": {path: {"exists": False} for path in remote.STOP_PATHS + [
            remote.CHECKOUT + "/research/agentic/STOP",
            remote.CHECKOUT + "/research/v2/STOP"]},
        "protected": {
            "checkpoints": checkpoints, "e65_transfer_files": prior_files,
            "e65_checkout": old_paths[remote.E65_CHECKOUT],
            "e65_source": e65_source,
            "cache_root": old_paths[remote.CACHE_ROOT],
            "six_caches": {name: meta(inode=inode, mode="0o775")
                           for name, inode in remote.CACHE_INODES.items()}},
        "processes": {"matched": [], "unreadable_cmdlines": 0},
        "science_label_scan": {"rc": 0, "matches": [], "match_count": 0,
                               "truncated": False, "stderr_empty": True},
        "checkpoint_fusers": {path: {"path_exists": True, "rc": 1,
                                    "pids": [], "stderr_empty": True}
                              for path in remote.CHECKPOINTS},
        "shared_lock_fuser": {"path_exists": True, "rc": 1,
                              "pids": [], "stderr_empty": True},
        "neuron0_fuser": {"path_exists": True, "rc": 1,
                          "pids": [], "stderr_empty": True},
        "neuron_ls": {"rc": 0, "json_valid": True, "schema_valid": True,
                      "device_count": 1, "non_diagnostic_pids": [],
                      "stderr_empty": True},
    }
    if archive is not None:
        result["targets"][remote.TARGET] = archive
    return result


def transfer_receipts(pre):
    archive = meta("regular", inode=700777, mode="0o600", size=remote.BYTES)
    prewrite = probe(pid=201, offset=-2)
    write = {
        "schema": remote.SCHEMA, "operation": "write",
        "status": "HASH_VERIFIED_IN_THIS_SESSION_ONLY", "utc": utc(-1),
        "process_identity": prewrite["process_identity"],
        "path": remote.TARGET, "expected_bytes": remote.BYTES,
        "expected_sha256": remote.SHA256, "streamed_sha256": remote.SHA256,
        "readback_sha256": remote.SHA256, "target_stat": archive,
        "identity_before": identity(), "identity_after": identity(),
        "prewrite_probe": prewrite,
        "preflight_identity_sha256": remote._identity_digest(pre["identity_after"]),
        "immediate_gate": {key: copy.deepcopy(prewrite[key]) for key in (
            "targets", "stops", "processes", "checkpoint_fusers", "shared_lock_fuser",
            "neuron0_fuser", "neuron_ls")},
    }
    verify = {**write, "operation": "verify", "utc": utc(0),
              "process_identity": {"pid": 202, "start_ticks": 2020},
              "streamed_sha256": None, "prewrite_probe": None,
              "preflight_identity_sha256": None}
    return archive, write, verify


class TransferOnlyTests(unittest.TestCase):
    def test_frozen_hold_and_no_ssh(self):
        request = json.loads(operator.REQUEST.read_text(encoding="utf-8"))
        self.assertEqual(request["host"], "trn21")
        self.assertEqual(request["target"], remote.TARGET)
        self.assertEqual(operator._sha(operator.REQUEST.read_bytes()),
                         operator.FROZEN_HOLD_REQUEST_SHA256)
        self.assertEqual(request["remote_sha256"],
                         hashlib.sha256(transport.REMOTE.read_bytes()).hexdigest())
        self.assertEqual(request["transport_sha256"],
                         hashlib.sha256((HERE / "stage_r4_transport.py").read_bytes()).hexdigest())
        self.assertIs(request["writer_preflight_identity_bridge_required"], True)
        self.assertIs(request["writer_immediate_activity_gate_required"], True)
        for key in ("dispatch_authorized", "source_checkout_authorized",
                    "cpu_g2_authorized", "native_run_authorized", "eval_authorized",
                    "monitor_operation_authorized", "cleanup_authorized",
                    "submission_authorized",
                    "same_path_overwrite_delete_rename_or_retry_authorized"):
            self.assertIs(request[key], False)
        with mock.patch.object(operator.subprocess, "Popen",
                               side_effect=AssertionError("SSH opened")):
            with self.assertRaisesRegex(RuntimeError, "DISPATCH HOLD"):
                operator.run_once()

    def test_frozen_source_and_plan_pins(self):
        source = transport.checked_local_source()
        self.assertEqual((source["bytes"], source["sha256"]),
                         (remote.BYTES, remote.SHA256))
        self.assertEqual(transport.REMOTE_SHA256,
                         hashlib.sha256(transport.REMOTE.read_bytes()).hexdigest())
        self.assertEqual(operator._sha(operator.TRN21_HOLD_PLAN.read_bytes()),
                         operator.TRN21_HOLD_PLAN_SHA256)
        self.assertEqual(operator._sha(operator.CAPSULE_REVIEW.read_bytes()),
                         operator.CAPSULE_REVIEW_SHA256)

    def test_trn21_host_key_and_literal_isolated_ssh_argv(self):
        found = transport.checked_ssh_identity()
        self.assertEqual(found["host_fingerprint"],
                         "SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc")
        pre = probe()
        for operation in ("probe", "write", "verify"):
            argv = transport.build_ssh_argv(operation,
                                            preflight=pre if operation == "write" else None)
            self.assertEqual(argv[:4], ["ssh", "-F", "/dev/null", "-T"])
            self.assertEqual(argv[-2], transport.ENDPOINT)
            self.assertIn("UserKnownHostsFile=" + str(transport.KNOWN_HOSTS), argv)
            self.assertIn("GlobalKnownHostsFile=/dev/null", argv)
            self.assertIn("HostKeyAlgorithms=ssh-ed25519", argv)
            words = ["exec", "python3", "-c", transport.reviewed_remote_source(),
                     operation]
            if operation == "write":
                words.append(remote._identity_digest(pre["identity_after"]))
            self.assertEqual(shlex.split(argv[-1]), words)
        with self.assertRaisesRegex(RuntimeError, "validated fresh preflight"):
            transport.build_ssh_argv("write")
        with self.assertRaises(ValueError):
            transport.build_ssh_argv("checkout")

    def test_known_hosts_hash_drift_fails_locally(self):
        with mock.patch.object(transport, "KNOWN_HOSTS_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "known-hosts bytes changed"):
                transport.build_ssh_argv("probe")

    def test_local_stop_and_breaker_mutations_fail(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            control = root / "control/research/v2/state.json"
            checkout = root / "worktree"
            control.parent.mkdir(parents=True)
            (control.parent.parent / "agentic").mkdir()
            (checkout / "research/v2").mkdir(parents=True)
            (checkout / "research/agentic").mkdir()
            clean = {"breakers": {"global_correctness": {"tripped": False},
                                  "trn21": {"tripped": False}}}
            control.write_text(json.dumps(clean), encoding="utf-8")
            with (mock.patch.object(operator, "CONTROL_STATE", control),
                  mock.patch.object(operator, "REPO", checkout)):
                self.assertTrue(operator._check_controller_breakers()["local_stops_clear"])
                for path in (control.parent / "STOP",
                             control.parent.parent / "agentic/STOP",
                             checkout / "research/v2/STOP",
                             checkout / "research/agentic/STOP"):
                    path.write_text("stop", encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "local STOP"):
                        operator._check_controller_breakers()
                    path.unlink()
                for breaker in ("global_correctness", "trn21"):
                    changed = copy.deepcopy(clean)
                    changed["breakers"][breaker]["tripped"] = True
                    control.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "breaker"):
                        operator._check_controller_breakers()

    def test_flag_flip_and_forged_review_fail(self):
        request = json.loads(operator.REQUEST.read_text(encoding="utf-8"))
        request["dispatch_authorized"] = True
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            forged = Path(directory) / "request.json"
            forged.write_text(json.dumps(request), encoding="utf-8")
            with mock.patch.object(operator, "REQUEST", forged):
                with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                    operator.authorize()
        request["remote_sha256"] = transport.REMOTE_SHA256
        request["transport_sha256"] = operator._sha((HERE / "stage_r4_transport.py").read_bytes())
        request_bytes = json.dumps(request).encode("utf-8")
        fake_review = json.dumps({"verdict": "PASS"}).encode("utf-8")
        fake_assignment = json.dumps({"dispatch_authorized": True}).encode("utf-8")
        with self.assertRaisesRegex(RuntimeError, "structured PASS"):
            operator._validate_authority_bundle(request_bytes, fake_review, fake_assignment)

    def test_alias_proof_and_protected_assets_gate(self):
        good = probe()
        transport.validate_probe(good, before_write=True)
        bad = copy.deepcopy(good)
        bad["identity_before"]["block_aliases"]["historical_names"][
            "/dev/nvme0n1p1"]["resolved_rdev_major_minor"] = "259:2"
        bad["identity_after"] = copy.deepcopy(bad["identity_before"])
        with self.assertRaisesRegex(RuntimeError, "aliases have not been reconciled"):
            transport.validate_probe(bad, before_write=True)
        bad = copy.deepcopy(good)
        first = next(iter(remote.CHECKPOINTS))
        bad["protected"]["checkpoints"][first]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "protected checkpoint"):
            transport.validate_probe(bad, before_write=True)
        bad = copy.deepcopy(good)
        bad["protected"]["e65_checkout"]["inode"] += 1
        with self.assertRaisesRegex(RuntimeError, "accepted e65 checkout"):
            transport.validate_probe(bad, before_write=True)

    def test_bridge_and_cross_session_positive_fixture(self):
        pre = probe()
        archive, write, verify = transfer_receipts(pre)
        persistence = transport.compare_sessions(pre, write, verify)
        post = probe(pid=300, offset=1, archive=archive)
        self.assertEqual(transport.compare_postflight(pre, post,
                                                       persistence=persistence)["status"],
                         "READ_ONLY_POSTFLIGHT_PASS")

    def test_preflight_to_write_boot_or_mount_drift_fails_before_creation(self):
        pre = probe()
        _, write, verify = transfer_receipts(pre)
        drift = probe(pid=201, offset=-2, boot="new-boot")
        write["prewrite_probe"] = drift
        write["identity_before"] = identity(boot="new-boot")
        write["identity_after"] = identity(boot="new-boot")
        with self.assertRaisesRegex(RuntimeError, "preflight-to-write"):
            transport.compare_sessions(pre, write, verify)
        with (mock.patch.object(remote, "probe", return_value=drift),
              mock.patch.object(remote, "_open_parent",
                                side_effect=AssertionError("target creation reached"))):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                remote.write(io.BytesIO(b""), remote._identity_digest(pre["identity_after"]))
        mount_drift = copy.deepcopy(drift)
        mount_drift["identity_before"]["mount_namespace"] = "mnt:[changed]"
        mount_drift["identity_after"]["mount_namespace"] = "mnt:[changed]"
        with (mock.patch.object(remote, "probe", return_value=mount_drift),
              mock.patch.object(remote, "_open_parent",
                                side_effect=AssertionError("target creation reached"))):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                remote.write(io.BytesIO(b""), remote._identity_digest(pre["identity_after"]))

    def test_remote_stop_and_process_gate_precedes_target_creation(self):
        pre = probe()
        for mutation, expected in (("STOP", "STOP"), ("process", "activity")):
            bad = probe(pid=201, offset=-2)
            if mutation == "STOP":
                first = next(iter(bad["stops"]))
                bad["stops"][first] = meta("regular", mode="0o600", size=1)
            else:
                bad["processes"]["matched"] = [{"pid": 44, "markers": ["train.py"]}]
            with (mock.patch.object(remote, "probe", return_value=bad),
                  mock.patch.object(remote, "_open_parent",
                                    side_effect=AssertionError("target creation reached"))):
                with self.assertRaisesRegex(RuntimeError, expected):
                    remote.write(io.BytesIO(b""),
                                 remote._identity_digest(pre["identity_after"]))

    def test_target_stop_or_process_appearing_after_clean_writer_probe_fails(self):
        pre = probe()
        writer_probe = probe(pid=201, offset=-2)
        pin = remote._identity_digest(pre["identity_after"])
        current = identity()
        for mutation, expected in (("target", "future target appeared"),
                                   ("STOP", "STOP appeared"),
                                   ("process", "process appeared")):
            with tempfile.TemporaryDirectory(dir=HERE) as directory:
                target = Path(directory) / remote.TARGET_NAME
                def parent():
                    return os.open(directory, os.O_RDONLY | os.O_DIRECTORY), current["ancestry"]
                def live_path(path):
                    if mutation == "target" and path == remote.CHECKOUT:
                        return meta(mode="0o700")
                    if mutation == "STOP" and path == remote.STOP_PATHS[0]:
                        return meta("regular", mode="0o600", size=1)
                    return {"exists": False}
                activity = ({"matched": [{"pid": 44, "markers": ["train.py"]}],
                             "unreadable_cmdlines": 0} if mutation == "process"
                            else {"matched": [], "unreadable_cmdlines": 0})
                with (mock.patch.object(remote, "probe", return_value=writer_probe),
                      mock.patch.object(remote, "_open_parent", side_effect=parent),
                      mock.patch.object(remote, "_identity", return_value=current),
                      mock.patch.object(remote, "_safe_path", side_effect=live_path),
                      mock.patch.object(remote, "_processes", return_value=activity)):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        remote.write(io.BytesIO(b""), pin)
                self.assertFalse(target.exists(), "live gate must precede O_EXCL creation")

    def test_activity_appearing_during_holder_checks_fails_final_live_gate(self):
        pre = probe()
        writer_probe = probe(pid=201, offset=-2)
        pin = remote._identity_digest(pre["identity_after"])
        current = identity()
        for mutation, expected in (("target", "future target appeared"),
                                   ("STOP", "STOP appeared"),
                                   ("process", "process appeared")):
            with tempfile.TemporaryDirectory(dir=HERE) as directory:
                active = {"now": False}
                target = Path(directory) / remote.TARGET_NAME
                def parent():
                    return os.open(directory, os.O_RDONLY | os.O_DIRECTORY), current["ancestry"]
                def live_path(path):
                    if active["now"] and mutation == "target" and path == remote.CHECKOUT:
                        return meta(mode="0o700")
                    if active["now"] and mutation == "STOP" and path == remote.STOP_PATHS[0]:
                        return meta("regular", mode="0o600", size=1)
                    return {"exists": False}
                def live_processes():
                    return {"matched": [{"pid": 44, "markers": ["train.py"]}]
                            if active["now"] and mutation == "process" else [],
                            "unreadable_cmdlines": 0}
                def live_neuron():
                    active["now"] = True
                    return writer_probe["neuron_ls"]
                clear_holder = {"path_exists": True, "rc": 1,
                                "pids": [], "stderr_empty": True}
                with (mock.patch.object(remote, "probe", return_value=writer_probe),
                      mock.patch.object(remote, "_open_parent", side_effect=parent),
                      mock.patch.object(remote, "_identity", return_value=current),
                      mock.patch.object(remote, "_safe_path", side_effect=live_path),
                      mock.patch.object(remote, "_processes", side_effect=live_processes),
                      mock.patch.object(remote, "_fuser", return_value=clear_holder),
                      mock.patch.object(remote, "_neuron_ls", side_effect=live_neuron)):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        remote.write(io.BytesIO(b""), pin)
                self.assertFalse(target.exists(), "late live gate must precede creation")

    def test_partial_target_is_preserved_and_same_path_refused(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            name = "sole-new-target.tar.gz"
            target = Path(directory) / name
            current = identity()
            pin = remote._identity_digest(current)
            def parent():
                return os.open(directory, os.O_RDONLY | os.O_DIRECTORY), current["ancestry"]
            def path_meta(path):
                return {"exists": Path(path).exists()}
            with (mock.patch.object(remote, "TARGET_NAME", name),
                  mock.patch.object(remote, "TARGET", str(target)),
                  mock.patch.object(remote, "BYTES", 5),
                  mock.patch.object(remote, "SHA256", hashlib.sha256(b"abcde").hexdigest()),
                  mock.patch.object(remote, "_open_parent", side_effect=parent),
                  mock.patch.object(remote, "_identity", return_value=current),
                  mock.patch.object(remote, "probe", return_value={}),
                  mock.patch.object(remote, "_prewrite_safety",
                                    return_value={"identity_after": current}),
                  mock.patch.object(remote, "_immediate_activity_gate", return_value={}),
                  mock.patch.object(remote, "_safe_path", side_effect=path_meta)):
                with self.assertRaisesRegex(RuntimeError, "partial target retained"):
                    remote.write(io.BytesIO(b"abc"), pin)
                self.assertEqual(target.read_bytes(), b"abc")
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
                with self.assertRaises(FileExistsError):
                    remote.write(io.BytesIO(b"abcde"), pin)
                self.assertEqual(target.read_bytes(), b"abc")

    def test_fixed_marker_and_journal_are_exclusive(self):
        with tempfile.TemporaryDirectory(dir=operator.MARKER.parent) as directory:
            marker = Path(directory) / "one-attempt.marker"
            journal_path = Path(directory) / "one-attempt.jsonl"
            with (mock.patch.object(operator, "MARKER", marker),
                  mock.patch.object(operator, "JOURNAL", journal_path),
                  mock.patch.object(operator, "_read_evidence", return_value=b"fixture")):
                digest = operator._consume_fixed_marker({"one_attempt": True})
                self.assertEqual(digest, operator._sha(marker.read_bytes()))
                self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
                with self.assertRaises(FileExistsError):
                    operator._consume_fixed_marker({"one_attempt": True})
                journal = operator.Journal()
                journal.emit("TEST", marker_sha256=digest)
                journal.close()
                self.assertEqual(stat.S_IMODE(journal_path.stat().st_mode), 0o600)
                with self.assertRaises(FileExistsError):
                    operator.Journal()

    def test_failed_write_still_attempts_distinct_full_postflight(self):
        seen, events = [], []
        pre = probe()
        post = probe(pid=301, offset=1)
        class FakeJournal:
            def emit(self, event, **details):
                events.append(event)
            def close(self):
                events.append("CLOSED")
        def fake_ssh(operation, journal, *, preflight=None):
            seen.append(operation)
            if len(seen) == 1:
                return pre
            if operation == "write":
                raise RuntimeError("synthetic partial write")
            return post
        with (mock.patch.object(operator, "authorize", return_value=(
                    {}, {"request_sha256": "b" * 64})),
              mock.patch.object(operator.transport, "checked_local_source", return_value={}),
              mock.patch.object(operator.transport, "checked_ssh_identity", return_value={}),
              mock.patch.object(operator, "_check_controller_breakers", return_value={}),
              mock.patch.object(operator, "_consume_fixed_marker", return_value="a" * 64),
              mock.patch.object(operator, "_pinned_marker", return_value=None),
              mock.patch.object(operator, "_read_evidence", return_value=b"fixture"),
              mock.patch.object(operator, "Journal", FakeJournal),
              mock.patch.object(operator, "_run_ssh", side_effect=fake_ssh)):
            with self.assertRaisesRegex(RuntimeError, "transfer HOLD"):
                operator.run_once()
        self.assertEqual(seen, ["probe", "write", "probe"])
        self.assertIn("FINAL_READ_ONLY_POSTFLIGHT", events)
        self.assertIn("TRANSFER_HOLD", events)


if __name__ == "__main__":
    unittest.main()
