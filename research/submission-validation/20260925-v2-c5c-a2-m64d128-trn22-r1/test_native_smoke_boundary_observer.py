"""Local-only regression checks for the A2 external STEP_LOG parent gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location(
    "a2_native_smoke_boundary_observer", HERE / "NATIVE_SMOKE_BOUNDARY_OBSERVER.py"
)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class StepParentGateTest(unittest.TestCase):
    label = "fs-20260925-v2-c5c-a2-m64d128-trn22-r1-smoke"

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="a2-step-parent-test-")
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.source = base / "source"
        self.scratch = base / "scratch"
        (self.source / "research" / "run-logs").mkdir(parents=True)
        guard = self.source / "research" / "fresh_start_guard.py"
        guard.write_text("# local test stub; never executed\n")
        self.parent = self.scratch / self.label
        self.parent.mkdir(parents=True)
        self.scratch.chmod(0o700)
        self.parent.chmod(0o700)
        self.probe = self.parent / f"{self.label}.write-probe"
        self.probe.write_text("positive same-UID scratch write probe\n")
        p = self.parent.stat()
        g = (self.source / "research" / "run-logs").stat()
        self.args = argparse.Namespace(
            label=self.label,
            source_checkout=self.source,
            scratch_root=self.scratch,
            base_observer=ROOT / "research" / "a0_boundary_observer.py",
            guard_script=guard,
            guard_script_sha256=hashlib.sha256(guard.read_bytes()).hexdigest(),
            process_json=self.source / "research" / "run-logs" / f"{self.label}.process.json",
            postflight_json=self.source / "research" / "run-logs" / f"{self.label}.postflight.json",
            boundary_file=self.parent / f"{self.label}.steps.jsonl",
            write_probe=self.probe,
            write_probe_sha256=hashlib.sha256(self.probe.read_bytes()).hexdigest(),
            expected_run_uid=os.geteuid(),
            parent_device=p.st_dev,
            parent_inode=p.st_ino,
            parent_uid=p.st_uid,
            parent_gid=p.st_gid,
            parent_mode=stat.S_IMODE(p.st_mode),
            guard_parent_device=g.st_dev,
            guard_parent_inode=g.st_ino,
            guard_parent_uid=g.st_uid,
            guard_parent_gid=g.st_gid,
            guard_parent_mode=stat.S_IMODE(g.st_mode),
        )

    def test_precreated_external_parent_and_absent_step_file_pass(self) -> None:
        receipt, reviewed_bytes = MOD.validate(self.args)
        self.assertEqual(receipt["step_parent"], str(self.parent))
        self.assertEqual(hashlib.sha256(reviewed_bytes).hexdigest(), MOD.BASE_OBSERVER_SHA256)

    def test_missing_parent_fails_before_observer(self) -> None:
        self.args.scratch_root = self.scratch / "missing"
        with self.assertRaises((ValueError, OSError)):
            MOD.validate(self.args)

    def test_precreated_step_file_is_rejected(self) -> None:
        self.args.boundary_file.write_text("")
        with self.assertRaisesRegex(ValueError, "STEP_LOG file already exists"):
            MOD.validate(self.args)

    def test_parent_identity_drift_is_rejected(self) -> None:
        self.args.parent_inode += 1
        with self.assertRaisesRegex(ValueError, "identity/ownership/mode changed"):
            MOD.validate(self.args)

    def test_missing_positive_write_probe_is_rejected(self) -> None:
        self.probe.unlink()
        with self.assertRaises(FileNotFoundError):
            MOD.validate(self.args)

    def test_missing_guard_receipt_parent_is_rejected(self) -> None:
        (self.source / "research" / "run-logs").rmdir()
        with self.assertRaises(FileNotFoundError):
            MOD.validate(self.args)

    def test_guard_receipt_parent_identity_drift_is_rejected(self) -> None:
        self.args.guard_parent_inode += 1
        with self.assertRaisesRegex(ValueError, "guard receipt parent identity/ownership/mode changed"):
            MOD.validate(self.args)

    def test_unwritable_parent_is_rejected(self) -> None:
        self.parent.chmod(0o500)
        with self.assertRaisesRegex(ValueError, "identity/ownership/mode changed|not writable"):
            MOD.validate(self.args)

    def test_symlinked_scratch_component_is_rejected(self) -> None:
        link = Path(self.temp.name) / "linked-scratch"
        link.symlink_to(self.scratch, target_is_directory=True)
        self.args.scratch_root = link
        self.args.boundary_file = link / self.label / f"{self.label}.steps.jsonl"
        self.args.write_probe = link / self.label / f"{self.label}.write-probe"
        with self.assertRaisesRegex(ValueError, "symlinked"):
            MOD.validate(self.args)

    def test_parent_inside_clean_source_is_rejected(self) -> None:
        self.args.scratch_root = self.source / "runtime"
        self.args.boundary_file = self.args.scratch_root / self.label / f"{self.label}.steps.jsonl"
        self.args.write_probe = self.args.scratch_root / self.label / f"{self.label}.write-probe"
        self.args.write_probe.parent.mkdir(parents=True)
        self.args.scratch_root.chmod(0o700)
        self.args.write_probe.parent.chmod(0o700)
        self.args.write_probe.write_text("test\n")
        self.args.write_probe_sha256 = hashlib.sha256(self.args.write_probe.read_bytes()).hexdigest()
        p = self.args.write_probe.parent.stat()
        self.args.parent_device = p.st_dev
        self.args.parent_inode = p.st_ino
        self.args.parent_uid = p.st_uid
        self.args.parent_gid = p.st_gid
        self.args.parent_mode = stat.S_IMODE(p.st_mode)
        with self.assertRaisesRegex(ValueError, "overlaps the clean source checkout"):
            MOD.validate(self.args)


if __name__ == "__main__":
    unittest.main()
