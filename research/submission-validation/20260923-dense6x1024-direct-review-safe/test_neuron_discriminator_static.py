#!/usr/bin/env python3
"""Local, hardware-free fail-closed tests for neuron_discriminator_direct.py."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("neuron_discriminator_direct", HERE / "neuron_discriminator_direct.py")
assert SPEC is not None and SPEC.loader is not None
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class DiscriminatorStaticTests(unittest.TestCase):
    @staticmethod
    def launch_args():
        return argparse.Namespace(
            run_label="fs-direct-discriminator-verifier-supplied",
            expected_commit="1" * 40,
            authorized_head="1" * 40,
            authorization_sha256="2" * 64,
            report=D.ROOT / "research/run-logs/fs-direct-discriminator-verifier-supplied-result.json",
            expected_hostname=D.EXPECTED_HOSTNAME,
        )

    def test_frozen_source_and_direct(self):
        result = D.static_source_checks()
        self.assertTrue(result["passed"], result)
        self.assertEqual(D.sha256(D.SOURCE), D.SOURCE_SHA256)
        self.assertEqual(D.sha256(D.DIRECT), D.DIRECT_SHA256)
        self.assertEqual(D.sha256(D.PREPARE), D.PREPARE_SHA256)

    def test_numerical_contract_is_literal(self):
        contract = D.expected_contract()
        self.assertEqual(contract["ce_rows"], [128, 512, 32768])
        self.assertEqual(contract["ce_repetitions"], 2)
        self.assertEqual(contract["optimizer_fixtures"], [
            "tiny_fixed", "tiny_changed", "production_width1024_fs_w62"
        ])
        self.assertEqual(contract["outer_timeout_argv"], [
            "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "5400s"
        ])
        self.assertEqual(contract["ce_child_environment"]["logical_singleton"], {
            "world_size": 1, "local_world_size": 1, "rank": 0, "local_rank": 0,
        })
        self.assertEqual(contract["optimizer_child_environment"]["retained_outer"], {
            "WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4",
        })
        self.assertEqual(contract["child_process_group"], {
            "start_new_session": True,
            "term_grace_seconds": 5.0,
            "kill_grace_seconds": 5.0,
            "poll_seconds": 0.05,
            "quiescence_required_before_snapshot": True,
            "pre_top_log_artifact_cache_revalidation": True,
        })
        self.assertEqual(contract["prepare_import_boundary"], {
            "repository_root_first": True,
            "explicit_canonical_module_load": True,
            "reject_preloaded_prepare": True,
            "reject_shadow_or_wrong_path": True,
            "regular_non_symlink_required": True,
            "authorized_sha256_required": D.PREPARE_SHA256,
        })

    def test_recompile_limit_warning_invalidates(self):
        self.assertTrue(D.scan_recompile_warning("WARNING: recompilation count exceeded limit"))
        self.assertTrue(D.scan_recompile_warning("too many recompiles; aborting"))
        self.assertFalse(D.scan_recompile_warning("compiled graph normally"))

    def test_result_writer_rejects_overwrite_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "result.json"
            D.exclusive_json(target, {"passed": True})
            with self.assertRaises(FileExistsError):
                D.exclusive_json(target, {"passed": False})
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(FileExistsError):
                D.exclusive_json(link, {"passed": False})

    def test_pending_request_cannot_be_full_authority(self):
        request = {
            "schema": D.AUTH_SCHEMA,
            "authorization_state": "pending_external_verifier",
            "purpose": D.EXPECTED_PURPOSE,
            "request_contract": D.expected_contract(),
            "expected_sha256": {
                name: D.sha256(D.ROOT / name) for name in D.PROTECTED_PATHS
                if (D.ROOT / name).exists()
            },
        }
        # Unit-test the trust-state branch without weakening the real fixed path.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            original = D.REQUEST
            try:
                D.REQUEST = path
                args = argparse.Namespace(
                    authorization_request=path,
                    static_only=False,
                    authorized_head="0" * 40,
                    authorization_sha256=D.sha256(path),
                )
                with mock.patch.object(D, "require_request_integrity", return_value={"tracked": True, "clean": True}):
                    with self.assertRaisesRegex(ValueError, "pending authorization request cannot launch"):
                        D.authenticate(args, full=True)
            finally:
                D.REQUEST = original

    def test_changed_contract_fails_before_fingerprints(self):
        request = {
            "schema": D.AUTH_SCHEMA,
            "authorization_state": "pending_external_verifier",
            "purpose": D.EXPECTED_PURPOSE,
            "request_contract": {**D.expected_contract(), "ce_rows": [128, 512]},
            "expected_sha256": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            original = D.REQUEST
            try:
                D.REQUEST = path
                args = argparse.Namespace(
                    authorization_request=path,
                    static_only=True,
                    authorized_head=D.PENDING_HEAD,
                    authorization_sha256=D.sha256(path),
                )
                with mock.patch.object(D, "require_request_integrity", return_value={"tracked": True, "clean": True}):
                    with self.assertRaisesRegex(ValueError, "contract differs"):
                        D.authenticate(args, full=False)
            finally:
                D.REQUEST = original

    def test_fingerprint_mutation_fails(self):
        hashes = {name: D.sha256(D.ROOT / name) for name in D.PROTECTED_PATHS}
        hashes[D.PROTECTED_PATHS[0]] = "f" * 64
        request = {
            "schema": D.AUTH_SCHEMA,
            "authorization_state": "pending_external_verifier",
            "purpose": D.EXPECTED_PURPOSE,
            "request_contract": D.expected_contract(),
            "expected_sha256": hashes,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            original = D.REQUEST
            try:
                D.REQUEST = path
                args = argparse.Namespace(
                    authorization_request=path, static_only=True,
                    authorized_head=D.PENDING_HEAD, authorization_sha256=D.sha256(path),
                )
                with mock.patch.object(D, "require_request_integrity", return_value={"tracked": True, "clean": True}):
                    with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                        D.authenticate(args, full=False)
            finally:
                D.REQUEST = original

    def test_descendant_head_fails_with_original_literals(self):
        request = {
            "schema": D.AUTH_SCHEMA,
            "authorization_state": "finalized_by_external_verifier",
            "purpose": D.EXPECTED_PURPOSE,
            "request_contract": D.expected_contract(),
            "expected_sha256": {name: D.sha256(D.ROOT / name) for name in D.PROTECTED_PATHS},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            original = D.REQUEST
            try:
                D.REQUEST = path
                args = argparse.Namespace(
                    authorization_request=path, static_only=False,
                    authorized_head="1" * 40, authorization_sha256=D.sha256(path),
                )
                with mock.patch.object(D, "require_request_integrity", return_value={"tracked": True, "clean": True}):
                    with mock.patch.object(D, "git", side_effect=["2" * 40, ""]):
                        with self.assertRaisesRegex(ValueError, "exact externally authorized commit"):
                            D.authenticate(args, full=True)
            finally:
                D.REQUEST = original

    def test_exact_launch_argv_rejects_timeout_substitution_by_construction(self):
        args = self.launch_args()
        request = D.exact_launch_request(args, {"request_sha256": "2" * 64})
        self.assertEqual(request["command"][:5], D.OUTER_TIMEOUT)
        self.assertNotIn("9999s", request["command"])
        self.assertEqual(request["command"][5], D.NATIVE_PYTHON)
        self.assertEqual(request["environment"], {
            **D.STATIC_ENVIRONMENT,
            "TRAINIUM_DISCRIMINATOR_LABEL": args.run_label,
            "TRAINIUM_DISCRIMINATOR_EXPECTED_COMMIT": args.expected_commit,
            "TRAINIUM_DISCRIMINATOR_AUTHORIZED_HEAD": args.authorized_head,
            "TRAINIUM_DISCRIMINATOR_AUTHORIZATION_SHA256": args.authorization_sha256,
        })
        self.assertEqual(request["environment"]["WORLD_SIZE"], "4")
        self.assertEqual(request["environment"]["LOCAL_WORLD_SIZE"], "4")

    def test_generated_environment_is_exactly_accepted_and_missing_world_rejected(self):
        args = self.launch_args()
        environment = D.expected_outer_environment(args)
        runtime = {**environment, "TRAINIUM_FRESH_START_RUN_ID": args.run_label}
        with mock.patch.dict(D.os.environ, runtime, clear=True):
            with mock.patch.object(D.socket, "gethostname", return_value=D.EXPECTED_HOSTNAME):
                with mock.patch.object(D.sys, "executable", D.NATIVE_PYTHON):
                    with mock.patch.object(D, "git", return_value=args.expected_commit):
                        D.validate_full_environment(args)
        for missing in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
            broken = dict(runtime)
            broken.pop(missing)
            with mock.patch.dict(D.os.environ, broken, clear=True):
                with self.assertRaisesRegex(ValueError, f"environment mismatch for {missing}"):
                    D.validate_full_environment(args)

    def test_launcher_default_hash_union_and_manifest_are_literal(self):
        args = self.launch_args()
        authorization = json.loads(D.REQUEST.read_text())
        authorized_request = D.exact_launch_request(args, {})
        values = dict(authorization["expected_sha256"])
        values[D.rel(D.REQUEST)] = "2" * 64
        required = {path: values[path] for path in D.launcher_fingerprint_paths()}
        self.assertEqual(
            set(D.launcher_fingerprint_paths()),
            set(D.LAUNCHER_DEFAULT_FINGERPRINT_PATHS) | set(D.PROTECTED_PATHS) | {D.rel(D.REQUEST)},
        )
        for required_default in ("train.py", "prepare.py", "research/fresh_start_backup.py"):
            self.assertIn(required_default, required)
        payload = {
            **authorized_request,
            "started_at": "2026-09-23T00:00:00+00:00",
            "hostname": D.EXPECTED_HOSTNAME,
            "git_commit": args.expected_commit,
            "source_dirty": False,
            "preflight_free_bytes": 9 * 1024**3,
            "source_sha256": required,
        }
        D.validate_launcher_payload(payload, authorized_request, required, args, "job.json")
        mutations = []
        changed_env = copy.deepcopy(payload)
        changed_env["environment"]["WORLD_SIZE"] = "999"
        mutations.append(changed_env)
        changed_timeout = copy.deepcopy(payload)
        changed_timeout["command"][4] = "9999s"
        mutations.append(changed_timeout)
        missing_default = copy.deepcopy(payload)
        missing_default["source_sha256"].pop("train.py")
        mutations.append(missing_default)
        extra_hash = copy.deepcopy(payload)
        extra_hash["source_sha256"]["extra.py"] = "0" * 64
        mutations.append(extra_hash)
        for mutation in mutations:
            with self.assertRaises(ValueError):
                D.validate_launcher_payload(mutation, authorized_request, required, args, "job.json")

    def test_authorization_schema_rejects_embedded_self_sha(self):
        request = json.loads(D.REQUEST.read_text())
        request["authorization_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            original = D.REQUEST
            try:
                D.REQUEST = path
                args = argparse.Namespace(
                    authorization_request=path, static_only=True,
                    authorized_head=D.PENDING_HEAD, authorization_sha256=D.sha256(path),
                )
                with mock.patch.object(D, "require_request_integrity", return_value={"tracked": True, "clean": True}):
                    with self.assertRaisesRegex(ValueError, "top-level key schema"):
                        D.authenticate(args, full=False)
            finally:
                D.REQUEST = original

    def test_authorization_request_integrity_rejects_symlink_untracked_and_dirty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("{}")
            link = root / "request.json"
            link.symlink_to(target)
            original = D.REQUEST
            try:
                D.REQUEST = link.absolute()
                with self.assertRaisesRegex(ValueError, "regular and non-symlink"):
                    D.require_request_integrity(link)
                D.REQUEST = target.absolute()
                with mock.patch.object(D, "rel", return_value="request.json"):
                    with mock.patch.object(D, "git", side_effect=D.subprocess.CalledProcessError(1, "git")):
                        with self.assertRaisesRegex(ValueError, "not tracked"):
                            D.require_request_integrity(target)
                    with mock.patch.object(D, "git", side_effect=["request.json", " M request.json"]):
                        with self.assertRaisesRegex(ValueError, "dirty/untracked"):
                            D.require_request_integrity(target)
            finally:
                D.REQUEST = original

    @staticmethod
    def distributed_environment():
        return {
            "PATH": "/usr/bin:/bin",
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "4",
            "RANK": "3",
            "LOCAL_RANK": "3",
            "GROUP_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29400",
            "TORCHELASTIC_RUN_ID": "stale",
            "RDZV_ENDPOINT": "stale:29400",
            "PMI_RANK": "3",
            "OMPI_COMM_WORLD_RANK": "3",
            "TRAINIUM_DISCRIMINATOR_PARENT_NONCE": "unit-test-nonce",
        }

    def test_ce_child_environment_is_explicit_singleton(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            child, identity = D.prepare_ce_environment(self.distributed_environment(), cache)
            self.assertFalse(any(D.is_distributed_environment_name(name) for name in child), child)
            self.assertEqual(child["NEURON_COMPILE_CACHE_DIR"], str(cache))
            self.assertEqual(identity["mode"], "ce-singleton-launch")
            self.assertEqual(
                (identity["world_size"], identity["local_world_size"], identity["rank"],
                 identity["local_rank"], identity["rendezvous_configured"]),
                (1, 1, 0, 0, False),
            )
            self.assertEqual(
                identity["removed_distributed_names"],
                sorted(name for name in self.distributed_environment()
                       if D.is_distributed_environment_name(name)),
            )
            worker_identity = D.ce_singleton_environment_identity(child)
            self.assertEqual(worker_identity["mode"], "ce-singleton")
            self.assertFalse(worker_identity["rendezvous_configured"])

    def test_ce_worker_rejects_every_distributed_environment_family(self):
        names = [*sorted(D.DISTRIBUTED_ENV_NAMES), *[prefix + "UNIT_TEST" for prefix in D.DISTRIBUTED_ENV_PREFIXES]]
        for name in names:
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, "inherited distributed/rendezvous"):
                    D.ce_singleton_environment_identity({name: "unexpected"})

    def test_optimizer_child_environment_preserves_only_outer_world_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            child, identity = D.prepare_optimizer_environment(self.distributed_environment(), cache)
            self.assertEqual(child["WORLD_SIZE"], "4")
            self.assertEqual(child["LOCAL_WORLD_SIZE"], "4")
            self.assertNotIn("RANK", child)
            self.assertNotIn("LOCAL_RANK", child)
            self.assertNotIn("MASTER_ADDR", child)
            self.assertNotIn("TORCHELASTIC_RUN_ID", child)
            self.assertEqual(identity["mode"], "optimizer-torchrun-four-rank-launch")
            self.assertTrue(identity["rank_injected_by_torchrun"])
            self.assertTrue(identity["rendezvous_injected_by_torchrun"])
            for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
                broken = self.distributed_environment()
                broken[key] = "1"
                with self.assertRaisesRegex(RuntimeError, "requires retained"):
                    D.prepare_optimizer_environment(broken, cache)

    def test_prepare_import_is_explicit_and_repository_root_first(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            empty_first = base / "empty-first"
            root.mkdir()
            empty_first.mkdir()
            prepare = root / "prepare.py"
            prepare.write_text("IMPORT_BOUNDARY_MARKER = 'canonical'\n")
            expected = D.sha256(prepare)
            original_path = list(D.sys.path)
            original_prepare = D.sys.modules.pop("prepare", None)
            try:
                D.sys.path[:] = [str(empty_first), str(root), *original_path]
                identity = D.load_pinned_prepare(root, expected, expected)
                self.assertEqual(D.path_from_sys_path(D.sys.path[0]), root.resolve())
                self.assertEqual(identity, {
                    "module": "prepare",
                    "path": str(root.absolute() / "prepare.py"),
                    "sha256": expected,
                    "repository_root": str(root.absolute()),
                    "repository_root_sys_path_index": 0,
                })
                self.assertEqual(D.sys.modules["prepare"].IMPORT_BOUNDARY_MARKER, "canonical")
                self.assertEqual(Path(D.sys.modules["prepare"].__file__).resolve(), prepare.resolve())
            finally:
                D.sys.modules.pop("prepare", None)
                if original_prepare is not None:
                    D.sys.modules["prepare"] = original_prepare
                D.sys.path[:] = original_path
                D.importlib.invalidate_caches()

    def test_prepare_import_rejects_shadow_and_any_preload(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            shadow = base / "shadow"
            root.mkdir()
            shadow.mkdir()
            prepare = root / "prepare.py"
            prepare.write_text("IDENTITY = 'canonical'\n")
            (shadow / "prepare.py").write_text("IDENTITY = 'shadow'\n")
            expected = D.sha256(prepare)
            original_path = list(D.sys.path)
            original_prepare = D.sys.modules.pop("prepare", None)
            try:
                D.sys.path[:] = [str(shadow), str(root), *original_path]
                with self.assertRaisesRegex(RuntimeError, "shadow/wrong-path"):
                    D.load_pinned_prepare(root, expected, expected)
                self.assertNotIn("prepare", D.sys.modules)
                D.sys.path[:] = [str(root), *original_path]
                for module_path in (shadow / "prepare.py", prepare):
                    with self.subTest(module_path=module_path):
                        preloaded = types.ModuleType("prepare")
                        preloaded.__file__ = str(module_path)
                        D.sys.modules["prepare"] = preloaded
                        with self.assertRaisesRegex(RuntimeError, "refusing preloaded"):
                            D.load_pinned_prepare(root, expected, expected)
                        self.assertIs(D.sys.modules["prepare"], preloaded)
                        D.sys.modules.pop("prepare")
            finally:
                D.sys.modules.pop("prepare", None)
                if original_prepare is not None:
                    D.sys.modules["prepare"] = original_prepare
                D.sys.path[:] = original_path
                D.importlib.invalidate_caches()

    def test_prepare_import_rejects_symlink_nonregular_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            original_path = list(D.sys.path)
            original_prepare = D.sys.modules.pop("prepare", None)
            try:
                symlink_root = base / "symlink-root"
                symlink_root.mkdir()
                target = base / "target.py"
                target.write_text("VALUE = 1\n")
                (symlink_root / "prepare.py").symlink_to(target)
                with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                    D.load_pinned_prepare(symlink_root, D.sha256(target), D.sha256(target))

                nonregular_root = base / "nonregular-root"
                nonregular_root.mkdir()
                (nonregular_root / "prepare.py").mkdir()
                with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                    D.load_pinned_prepare(nonregular_root, "0" * 64, "0" * 64)

                mismatch_root = base / "mismatch-root"
                mismatch_root.mkdir()
                mismatch = mismatch_root / "prepare.py"
                mismatch.write_text("VALUE = 'actual'\n")
                with self.assertRaisesRegex(ValueError, "differs from the authorized protected hash"):
                    D.load_pinned_prepare(mismatch_root, "0" * 64, "0" * 64)
                with self.assertRaisesRegex(ValueError, "authorized prepare SHA-256 differs"):
                    D.load_pinned_prepare(mismatch_root, D.sha256(mismatch), "f" * 64)
            finally:
                D.sys.modules.pop("prepare", None)
                if original_prepare is not None:
                    D.sys.modules["prepare"] = original_prepare
                D.sys.path[:] = original_path
                D.importlib.invalidate_caches()

    def test_train_dynamic_import_cannot_bypass_prepare_boundary(self):
        module_name = "must_not_execute_train_before_prepare_boundary"
        D.sys.modules.pop(module_name, None)
        with mock.patch.object(
            D, "load_pinned_prepare", side_effect=RuntimeError("prepare boundary rejected")
        ) as boundary:
            with self.assertRaisesRegex(RuntimeError, "prepare boundary rejected"):
                D.import_train(D.SOURCE, module_name)
        boundary.assert_called_once_with()
        self.assertNotIn(module_name, D.sys.modules)

    def run_local_child(self, root, name, code, timeout=5):
        cache = root / f"{name}-cache"
        cache.mkdir(exist_ok=True)
        artifact = root / f"{name}-artifact.txt"
        environment = {
            "PATH": "/usr/bin:/bin",
            "TRAINIUM_DISCRIMINATOR_PARENT_NONCE": "unit-test-nonce",
        }
        environment, identity = D.prepare_ce_environment(environment, cache)
        command = [sys.executable, "-c", code, str(artifact)]
        return D.run_child(
            name, command, environment, identity, timeout,
            root / f"{name}.log", root / f"{name}.json", cache,
            {"worker_artifact": artifact},
        )

    def test_child_success_record_has_exact_schema_complete_log_and_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = self.run_local_child(
                root, "success",
                "import pathlib,sys; print('stdout-complete', flush=True); "
                "print('stderr-complete', file=sys.stderr, flush=True); "
                "pathlib.Path(sys.argv[1]).write_text('artifact')",
            )
            self.assertTrue(process["passed"], process)
            self.assertEqual(set(process), D.CHILD_PROCESS_KEYS | {"record_receipt"})
            persisted = json.loads((root / "success.json").read_text())
            self.assertEqual(set(persisted), D.CHILD_PROCESS_KEYS)
            self.assertEqual(process["argv"], persisted["argv"])
            self.assertEqual(process["record_receipt"]["sha256"], D.sha256(root / "success.json"))
            log = (root / "success.log").read_text()
            self.assertIn("stdout-complete", log)
            self.assertIn("stderr-complete", log)
            self.assertEqual(process["log"]["bytes"], len((root / "success.log").read_bytes()))
            self.assertTrue(process["artifacts"]["worker_artifact"]["regular"])
            self.assertEqual(process["environment_identity"]["mode"], "ce-singleton-launch")
            self.assertTrue(process["process_group"]["start_new_session"])
            self.assertTrue(process["process_group"]["primary_reaped"])
            self.assertTrue(process["process_group"]["quiescent"])
            self.assertFalse(D.process_group_exists(process["process_group"]["process_group_id"]))
            D.validate_child_process_record(process, include_receipt=True)

    def test_child_error_timeout_and_warning_are_persisted_fail_closed(self):
        cases = {
            "error": ("import sys; print('before-error', flush=True); sys.exit(7)", 5),
            "timeout": ("import time; print('before-timeout', flush=True); time.sleep(3)", 1),
            "warning": ("print('torch._dynamo exceeded recompilation limit', flush=True)", 5),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, (code, timeout) in cases.items():
                with self.subTest(name=name):
                    process = self.run_local_child(root, name, code, timeout)
                    self.assertFalse(process["passed"])
                    self.assertTrue((root / f"{name}.json").is_file())
                    self.assertTrue((root / f"{name}.log").is_file())
                    persisted = json.loads((root / f"{name}.json").read_text())
                    self.assertEqual(persisted["argv"], process["argv"])
                    self.assertEqual(persisted["log"], process["log"])
                    if name == "error":
                        self.assertEqual(process["returncode"], 7)
                        self.assertIn("before-error", (root / "error.log").read_text())
                    elif name == "timeout":
                        self.assertTrue(process["timed_out"])
                        self.assertEqual(process["error"]["type"], "TimeoutExpired")
                        self.assertIn("before-timeout", (root / "timeout.log").read_text())
                    else:
                        self.assertTrue(process["recompile_limit_warnings"])

    def test_child_receipts_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = self.run_local_child(root, "once", "print('once', flush=True)")
            before_log = process["log"]["sha256"]
            before_record = process["record_receipt"]["sha256"]
            with self.assertRaisesRegex(FileExistsError, "replace child log"):
                self.run_local_child(root, "once", "print('twice', flush=True)")
            self.assertEqual(D.sha256(root / "once.log"), before_log)
            self.assertEqual(D.sha256(root / "once.json"), before_record)

    def test_child_record_schema_and_environment_mutations_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = self.run_local_child(root, "schema", "print('schema', flush=True)")
            mutations = []
            missing = copy.deepcopy(process)
            missing.pop("argv")
            mutations.append(missing)
            extra = copy.deepcopy(process)
            extra["unexpected"] = True
            mutations.append(extra)
            contaminated = copy.deepcopy(process)
            contaminated["environment_identity"]["retained"]["WORLD_SIZE"] = "4"
            mutations.append(contaminated)
            false_pass = copy.deepcopy(process)
            false_pass["returncode"] = 9
            mutations.append(false_pass)
            wrong_log = copy.deepcopy(process)
            wrong_log["log"]["sha256"] = "0" * 64
            mutations.append(wrong_log)
            wrong_record = copy.deepcopy(process)
            wrong_record["record_receipt"]["sha256"] = "0" * 64
            mutations.append(wrong_record)
            live_group = copy.deepcopy(process)
            live_group["process_group"]["quiescent"] = False
            mutations.append(live_group)
            for mutation in mutations:
                with self.assertRaises((ValueError, TypeError)):
                    D.validate_child_process_record(mutation, include_receipt=True)

    def test_spawn_error_persists_unlaunched_group_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "spawn-error-cache"
            cache.mkdir()
            environment, identity = D.prepare_ce_environment({
                "PATH": "/usr/bin:/bin",
                "TRAINIUM_DISCRIMINATOR_PARENT_NONCE": "unit-test-nonce",
            }, cache)
            process = D.run_child(
                "spawn-error", ["/definitely/not/a/real/executable"], environment, identity, 5,
                root / "spawn-error.log", root / "spawn-error.json", cache, {},
            )
            self.assertFalse(process["passed"])
            self.assertEqual(process["error"]["type"], "FileNotFoundError")
            self.assertFalse(process["process_group"]["launched"])
            self.assertTrue(process["process_group"]["quiescent"])
            D.validate_child_process_record(process, include_receipt=True)

    def test_late_writing_descendants_are_terminated_before_log_completion(self):
        cases = {"term": False, "kill": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, ignores_term in cases.items():
                descendant = (
                    "import signal,time;"
                    + ("signal.signal(signal.SIGTERM,signal.SIG_IGN);" if ignores_term else "")
                    + "print('descendant-ready',flush=True);time.sleep(2.0);"
                    + "print('late-descendant-output',flush=True);time.sleep(10)"
                )
                parent = (
                    "import subprocess,sys,time;"
                    f"subprocess.Popen([sys.executable,'-c',{descendant!r}]);"
                    "print('parent-ready',flush=True);time.sleep(10)"
                )
                with self.subTest(name=name):
                    with mock.patch.object(D, "PROCESS_GROUP_TERM_GRACE_SECONDS", 0.2), \
                            mock.patch.object(D, "PROCESS_GROUP_KILL_GRACE_SECONDS", 1.0):
                        process = self.run_local_child(root, f"late-{name}", parent, timeout=1)
                        self.assertFalse(process["passed"])
                        self.assertTrue(process["timed_out"])
                        self.assertTrue(process["process_group"]["term_sent"])
                        self.assertEqual(process["process_group"]["kill_sent"], ignores_term)
                        self.assertTrue(process["process_group"]["primary_reaped"])
                        self.assertTrue(process["process_group"]["quiescent"])
                        group_id = process["process_group"]["process_group_id"]
                        self.assertFalse(D.process_group_exists(group_id))
                        before = (process["log"]["bytes"], process["log"]["sha256"])
                        time.sleep(1.2)
                        after = (
                            (root / f"late-{name}.log").stat().st_size,
                            D.sha256(root / f"late-{name}.log"),
                        )
                        self.assertEqual(after, before)
                        self.assertNotIn("late-descendant-output", (root / f"late-{name}.log").read_text())
                        D.validate_child_process_record(process, include_receipt=True)
                        self.assertTrue(D.revalidate_process_evidence([process])["passed"])

    def test_prewrite_revalidation_rejects_late_log_artifact_and_cache_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mutations = ("log", "artifact", "cache")
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    name = f"prewrite-{mutation}"
                    base = self.run_local_child(
                        root, name,
                        "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('original'); print('done')",
                    )
                    self.assertTrue(D.revalidate_process_evidence([base])["passed"])
                    if mutation == "log":
                        with (root / f"{name}.log").open("ab") as output:
                            output.write(b"late\n")
                    elif mutation == "artifact":
                        (root / f"{name}-artifact.txt").write_text("changed")
                    else:
                        (root / f"{name}-cache" / "late-entry").write_text("changed")
                    revalidation = D.revalidate_process_evidence([base])
                    self.assertFalse(revalidation["passed"])
                    report = D.orchestrator_failure_report(
                        {"request_sha256": "a" * 64}, root / "results", [base], {},
                        RuntimeError("earlier failure"),
                    )
                    D.attach_prewrite_process_revalidation(report)
                    self.assertFalse(report["passed"])
                    self.assertFalse(report["process_evidence_revalidation"]["passed"])
                    self.assertEqual(report["evidence_failure"]["type"], "ProcessEvidenceMutation")

    def test_top_failure_report_retains_accumulated_child_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.run_local_child(root, "first", "print('first', flush=True)")
            second = self.run_local_child(root, "second", "import sys; print('second', flush=True); sys.exit(9)")
            report = D.orchestrator_failure_report(
                {"request_sha256": "a" * 64}, root / "results", [first, second],
                {"launch.json": {"sha256": "b" * 64}}, RuntimeError("second failed"),
            )
            self.assertFalse(report["passed"])
            self.assertEqual(report["processes"], [first, second])
            self.assertEqual(report["processes"][1]["returncode"], 9)
            self.assertEqual(report["processes"][1]["record_receipt"]["sha256"], D.sha256(root / "second.json"))
            self.assertEqual(report["failure"], {"type": "RuntimeError", "message": "second failed"})
            D.attach_prewrite_process_revalidation(report)
            self.assertTrue(report["process_evidence_revalidation"]["passed"])
            self.assertFalse(report["passed"])
            self.assertNotIn("evidence_failure", report)

    @staticmethod
    def tensor_record(label):
        return {
            "sha256": hashlib.sha256(label.encode()).hexdigest(),
            "shape": [1], "dtype": "torch.float32", "finite": True,
            "nonzero": True, "l2": 1.0, "max_abs": 1.0,
        }

    @classmethod
    def optimizer_matrix(cls):
        variants = {}
        for variant in ("raw", "source", "direct"):
            variants[variant] = []
            for rank in range(4):
                fixtures = []
                for fixture_name, spec in D.FIXTURE_SPECS.items():
                    count, total = spec["parameter_count"], spec["element_count"]
                    elements = [total - count + 1] + [1] * (count - 1)
                    entries = []
                    owner_elements = [0] * 4
                    owner_costs = [0.0] * 4
                    for index, amount in enumerate(elements):
                        owner = index % 4
                        kind = "adamw" if index % 2 == 0 else "muon"
                        cost = 1.0 + index / 1000
                        entries.append({
                            "index": index, "group": index, "position": 0, "kind": kind,
                            "shape": [amount], "elements": amount, "estimated_ms": cost,
                            "owner": owner, "offset": owner_elements[owner],
                        })
                        owner_elements[owner] += amount
                        owner_costs[owner] += cost
                    stride = (max(owner_elements) + 127) // 128 * 128
                    plan = {
                        "entries": entries,
                        "summary": {
                            "elements_by_owner": owner_elements,
                            "estimated_ms_by_owner": owner_costs,
                            "shard_elements": stride,
                            "total_padded_bytes": stride * 4 * 4,
                        },
                    }

                    def parameters(version):
                        return {
                            str(index): cls.tensor_record(f"{fixture_name}:parameter:{version}:{index}")
                            for index in range(count)
                        }

                    def state(step):
                        records = {}
                        for entry in entries:
                            if variant != "raw" and entry["owner"] != rank:
                                continue
                            key = f"g{entry['group']}:p0"
                            if entry["kind"] == "adamw":
                                records[key] = {
                                    "step": step,
                                    "exp_avg": cls.tensor_record(f"{fixture_name}:state:{step}:{key}:avg"),
                                    "exp_avg_sq": cls.tensor_record(f"{fixture_name}:state:{step}:{key}:sq"),
                                }
                            else:
                                records[key] = {
                                    "momentum_buffer": cls.tensor_record(f"{fixture_name}:state:{step}:{key}:momentum"),
                                    "second_momentum_buffer": cls.tensor_record(f"{fixture_name}:state:{step}:{key}:second"),
                                }
                        return records

                    snapshots = []
                    for step in range(1, spec["steps"] + 1):
                        parameter_version = 3 if fixture_name != "tiny_fixed" and step == 4 else step
                        parameter_records = parameters(parameter_version)
                        combined = hashlib.sha256(b"".join(
                            bytes.fromhex(parameter_records[str(index)]["sha256"])
                            for index in range(count)
                        )).hexdigest()
                        snapshot = {
                            "step": step,
                            "parameters": parameter_records,
                            "state": state(step),
                            "group_scalars": [
                                {
                                    "kind": entry["kind"],
                                    "lr": 0.0 if fixture_name != "tiny_fixed" and step == 4 else 0.01 / step,
                                    "momentum": 0.85 + 0.02 * step if entry["kind"] == "muon" else None,
                                }
                                for entry in entries
                            ],
                            "owner_calls": None if variant == "raw" else step,
                            "cross_rank_parameter_sha256": [combined] * 4,
                            "cross_rank_exact": True,
                        }
                        if fixture_name == "production_width1024_fs_w62":
                            output = cls.tensor_record(f"{fixture_name}:linear:{step}")
                            snapshot["linear_cache"] = {
                                "cache_exact": True, "output_exact": True,
                                "compiled_output": output, "eager_output": copy.deepcopy(output),
                            }
                        snapshots.append(snapshot)
                    fixtures.append({
                        "fixture": fixture_name,
                        "passed": True,
                        "parameter_count": count,
                        "element_count": total,
                        "plan": plan,
                        "initial_parameters": parameters(0),
                        "snapshots": snapshots,
                    })
                variants[variant].append({
                    "schema": D.WORKER_SCHEMA,
                    "worker": "optimizer",
                    "variant": variant,
                    "rank": rank,
                    "world_size": 4,
                    "passed": True,
                    "fixtures": fixtures,
                    "compile_cache": f"/tmp/cache-{variant}",
                })
        return variants

    @staticmethod
    def write_optimizer_matrix(root, matrix):
        for variant, reports in matrix.items():
            for rank, report in enumerate(reports):
                (root / f"optimizer-{variant}-rank{rank}.json").write_text(json.dumps(report))

    @staticmethod
    def make_first_step_parameter_noop(matrix):
        for reports in matrix.values():
            for report in reports:
                fixture = report["fixtures"][0]
                snapshot = fixture["snapshots"][0]
                snapshot["parameters"] = copy.deepcopy(fixture["initial_parameters"])
                combined = hashlib.sha256(b"".join(
                    bytes.fromhex(snapshot["parameters"][str(index)]["sha256"])
                    for index in range(fixture["parameter_count"])
                )).hexdigest()
                snapshot["cross_rank_parameter_sha256"] = [combined] * 4

    @staticmethod
    def make_second_step_state_noop(matrix):
        for reports in matrix.values():
            for report in reports:
                fixture = report["fixtures"][0]
                fixture["snapshots"][1]["state"] = copy.deepcopy(fixture["snapshots"][0]["state"])

    def test_optimizer_comparator_accepts_only_complete_exact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.optimizer_matrix()
            self.write_optimizer_matrix(root, matrix)
            comparison = D.compare_optimizer_results(root)
            self.assertTrue(comparison["passed"], comparison["failures"][:3])

    def test_optimizer_comparator_rejects_adversarial_evidence(self):
        mutations = {
            "empty_snapshots": lambda m: m["direct"][0]["fixtures"][0].__setitem__("snapshots", []),
            "extra_snapshot": lambda m: m["direct"][0]["fixtures"][0]["snapshots"].append(
                copy.deepcopy(m["direct"][0]["fixtures"][0]["snapshots"][-1])
            ),
            "nonfinite_tensor": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0]["parameters"]["0"].__setitem__("finite", False),
            "zero_tensor": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0]["parameters"]["0"].update(nonzero=False, l2=0.0, max_abs=0.0),
            "empty_state": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0].__setitem__("state", {}),
            "wrong_owner_calls": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0].__setitem__("owner_calls", 999),
            "wrong_schema": lambda m: m["direct"][0].__setitem__("schema", "wrong"),
            "wrong_rank": lambda m: m["direct"][0].__setitem__("rank", 3),
            "production_count_drift": lambda m: m["direct"][0]["fixtures"][2].__setitem__("parameter_count", 61),
            "nonfinite_scalar": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0]["group_scalars"][0].__setitem__("lr", float("nan")),
            "no_op_parameter_step": self.make_first_step_parameter_noop,
            "no_op_state_step": self.make_second_step_state_noop,
            "missing_step_key": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0].pop("step"),
            "bad_cross_rank_digest": lambda m: m["direct"][0]["fixtures"][0]["snapshots"][0]["cross_rank_parameter_sha256"].__setitem__(0, "0" * 64),
            "plan_drift": lambda m: m["direct"][0]["fixtures"][0]["plan"]["entries"][0].__setitem__("owner", 1),
        }
        base = self.optimizer_matrix()
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                matrix = copy.deepcopy(base)
                mutate(matrix)
                root = Path(directory)
                self.write_optimizer_matrix(root, matrix)
                comparison = D.compare_optimizer_results(root)
                self.assertFalse(comparison["passed"], name)


if __name__ == "__main__":
    unittest.main()
