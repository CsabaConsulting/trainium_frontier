#!/usr/bin/env python3
"""CPU/static adversarial tests for the Track-A confirmation lane."""

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import artifact_io as aio
import contracts as c
import launch_adapter as adapter
import process_lifecycle as lifecycle


class Reader:
    def __init__(self, values):
        self.values = values

    def read(self, name):
        if name not in self.values:
            raise SystemExit("fixture member missing")
        return self.values[name]


def metric(tokens, nbytes, bpb):
    return {
        "val_bpb": float(bpb), "num_tokens": tokens, "num_bytes": nbytes,
        "total_nats": float(bpb) * nbytes * math.log(2.0), "eval_seconds": 12.5,
        "eval_timeout": 1200, "eval_over_timeout": False,
    }


def clean_lifecycle(pgid=123):
    return {
        "pgid": pgid, "leader_returncode": 0, "cleanup_required": False,
        "term_sent": False, "kill_sent": False, "quiescent": True,
        "reaped_descendants": [], "term_grace_seconds": 15.0,
        "kill_grace_seconds": 10.0,
    }


def tar_bytes(values, names):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name in sorted(names):
            data = values[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def prior_fixture(key, bpb, control_bpb=1.0, mutate=None, completion_mutate=None):
    spec = c.PRIOR[key]
    checkpoint_sha = hashlib.sha256(key.encode()).hexdigest()
    checkpoint_path = str(Path(spec["remote"]) / "out" / f"{spec['label']}-work" / "sandbox/out/final.pt")
    screen = metric(c.SCREEN_TOKENS, c.SCREEN_BYTES, bpb)
    finite = {
        "finite": True, "weights_only": True, "tensor_count": 62,
        "checkpoint_sha256": checkpoint_sha, "checkpoint_bytes": 123456,
        "process_group": clean_lifecycle(124),
    }
    decision = None
    if spec["role"] == "a1":
        decision = {
            "decision": "NOMINATE_OTHER_HOST_FULL20M_REPLICATE", "screening_only": True,
            "keep_authorized": False, "adoption_authorized": False,
            "other_host_full20m_replicate_required": True,
            "control_screen2m_bpb": control_bpb, "candidate_screen2m_bpb": bpb,
        }
    result = {
        "schema": "track-a-result-v1", "arm": spec["arm"], "role": spec["role"],
        "label": spec["label"], "host": spec["host"], "hostname": spec["hostname"],
        "source_commit": c.SOURCE_COMMIT, "source_path": c.SOURCE_PATH,
        "source_sha256": c.SOURCE_SHA, "prepare_sha256": c.PREPARE_SHA,
        "launch_command": spec["launch"], "seed": 42, "world_size": 4, "lnc": 2,
        "max_train_seconds": 1800, "smoke": False, "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha, "checkpoint_bytes": 123456,
        "checkpoint_preserved": True, "checkpoint_prune_authorized": False,
        "cache_cleanup_authorized": False, "finite_scan": finite, "screen2m": screen,
        "screening_only": True, "keep_authorized": False, "adoption_authorized": False,
        "a1_decision": decision, "ready_to_retrieve": False,
    }
    receipt = {
        "schema": "track-a-screen2m-metrics-v1",
        "argv": c.expected_eval_argv(checkpoint_path, c.SCREEN_TOKENS), "metrics": screen,
        "wall_seconds": 12.6, "process_group": clean_lifecycle(125),
    }
    values = {name: (name + "\n").encode() for name in c.PRIOR_ARCHIVE_MEMBERS}
    values["training-process.json"] = c.canonical_json({
        "returncode": 0, "timed_out": False, "charged_within_budget": True,
        "seed": 42, "world_size": 4, "lnc": 2, "process_group": clean_lifecycle(126),
    })
    if mutate:
        mutate(result, receipt, values)
    values["result.json"] = c.canonical_json(result)
    values["screen2m.metrics.json"] = c.canonical_json(receipt)
    values["evidence.tar.gz"] = tar_bytes(values, c.PRIOR_ARCHIVE_MEMBERS)
    members = {name: aio.identity(data) for name, data in sorted(values.items())}
    manifest = {"schema": "track-a-artifact-manifest-v1", "label": spec["label"], "members": members}
    manifest_bytes = c.canonical_json(manifest)
    completion = {
        "schema": "track-a-completion-v1", "arm": spec["arm"], "role": spec["role"],
        "label": spec["label"], "host": spec["host"], "hostname": spec["hostname"],
        "finalization_head": c.VERIFIED_BASE_HEAD, "verified_core_commit": c.VERIFIED_LIFECYCLE_CORE,
        "authorization_sha256": "1" * 64, "request_sha256": "2" * 64,
        "runtime_job_sha256": "3" * 64, "source_sha256": c.SOURCE_SHA,
        "prepare_sha256": c.PREPARE_SHA, "artifact_manifest_sha256": aio.digest(manifest_bytes),
        "artifact_members": members, "archive_sha256": aio.digest(values["evidence.tar.gz"]),
        "archive_bytes": len(values["evidence.tar.gz"]), "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha, "checkpoint_bytes": 123456,
        "checkpoint_preserved": True, "checkpoint_prune_authorized": False,
        "cache_cleanup_authorized": False, "screen2m": screen, "ready_to_retrieve": True,
    }
    if completion_mutate:
        completion_mutate(completion)
    values["artifact-manifest.json"] = manifest_bytes
    values["ready-to-retrieve.json"] = c.canonical_json(completion)
    directory = f"research/run-logs/{spec['label']}/artifacts"
    identity = {
        "archive_commit": c.VERIFIED_BASE_HEAD, "directory": directory,
        "completion_sha256": aio.digest(values["ready-to-retrieve.json"]),
        "manifest_sha256": aio.digest(values["artifact-manifest.json"]),
        "result_sha256": aio.digest(values["result.json"]),
        "metrics_sha256": aio.digest(values["screen2m.metrics.json"]),
        "finalization_head": c.VERIFIED_BASE_HEAD,
        "verified_core_commit": c.VERIFIED_LIFECYCLE_CORE,
        "authorization_sha256": "1" * 64, "request_sha256": "2" * 64,
        "runtime_job_sha256": "3" * 64, "source_sha256": c.SOURCE_SHA,
        "prepare_sha256": c.PREPARE_SHA, "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha, "checkpoint_bytes": 123456,
    }
    return Reader(values), identity


def bundle_fixtures(mutate_key=None, mutate=None):
    readers = {}
    identities = {}
    bpbs = {"control-trn21": 1.0, "control-trn22": 1.01, "a1-trn21": 0.999}
    for key in c.PRIOR:
        reader, identity = prior_fixture(
            key, bpbs[key], 1.0,
            mutate if key == mutate_key else None,
        )
        readers[identity["directory"]] = reader
        identities[key] = identity
    return readers, identities


def reader_patch(readers):
    return mock.patch.object(c, "GitReader", side_effect=lambda identity, _kind: readers[identity["directory"]])


def full_receipt(label, checkpoint, bpb, pgid):
    spec = c.ARMS["pair-trn21"]
    return {
        "schema": "track-a-confirmation-metric-v1", "label": label,
        "checkpoint_path": checkpoint["checkpoint_path"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_bytes": checkpoint["checkpoint_bytes"], "host": spec["host"],
        "hostname": spec["hostname"], "source_sha256": c.SOURCE_SHA,
        "prepare_sha256": c.PREPARE_SHA,
        "argv": c.expected_eval_argv(checkpoint["checkpoint_path"], c.FULL_TOKENS),
        "metrics": metric(c.FULL_TOKENS, c.FULL_BYTES, bpb), "wall_seconds": 100.0,
        "process_group": clean_lifecycle(pgid),
    }


def confirmation_fixture(bundle, mutate=None, completion_mutate=None):
    spec = c.ARMS["pair-trn21"]
    runs = bundle["runs"]
    control = full_receipt(spec["label"], runs["control-trn21"], 1.0, 210)
    candidate = full_receipt(spec["label"], runs["a1-trn21"], 0.998, 211)
    metrics = {"control20m": control["metrics"], "a1_20m": candidate["metrics"]}
    decision = {
        "schema": "track-a-trn21-pair-decision-v1",
        "delta_definition": "candidate_minus_same_host_control",
        "metric_population": "fresh20m_seq1024_tokens20971520_bytes83462186",
        "trn21_delta": "-0.002", "strictly_negative": True,
        "verdict": "NOMINATE_TRN22_REPLICATE", "adoption_authorized": False,
    }
    summary_keys = (
        "archive_commit", "completion_sha256", "manifest_sha256", "result_sha256",
        "metrics_sha256", "finalization_head", "verified_core_commit", "authorization_sha256",
        "request_sha256", "runtime_job_sha256", "source_sha256", "prepare_sha256",
        "checkpoint_path", "checkpoint_sha256", "checkpoint_bytes",
    )
    prior = {key: {field: run[field] for field in summary_keys} for key, run in runs.items()}
    inputs = {"schema": "track-a-confirmation-input-v1", "prior_runs": prior,
              "screen_delta": bundle["screen_delta"], "trn21_confirmation": None}
    result = {
        "schema": "track-a-confirmation-result-v1", "arm": "pair-trn21",
        "role": spec["role"], "label": spec["label"], "host": spec["host"],
        "hostname": spec["hostname"], "source_sha256": c.SOURCE_SHA,
        "prepare_sha256": c.PREPARE_SHA, "input_evidence": inputs, "metrics": metrics,
        "decision": decision, "checkpoint_inputs_preserved": True,
        "candidate_checkpoint": None, "ready_to_retrieve": False,
    }
    values = {name: (name + "\n").encode() for name in c.OUTPUT_ARTIFACTS["pair-trn21"] - {"evidence.tar.gz"}}
    if mutate:
        mutate(result, control, candidate, values)
    values["input-evidence.json"] = c.canonical_json(inputs)
    values["control20m.metrics.json"] = c.canonical_json(control)
    values["a1-20m.metrics.json"] = c.canonical_json(candidate)
    values["decision.json"] = c.canonical_json(decision)
    values["result.json"] = c.canonical_json(result)
    values["evidence.tar.gz"] = tar_bytes(values, c.OUTPUT_ARTIFACTS["pair-trn21"] - {"evidence.tar.gz"})
    members = {name: aio.identity(data) for name, data in sorted(values.items())}
    manifest = {"schema": "track-a-confirmation-artifact-manifest-v1",
                "label": spec["label"], "members": members}
    manifest_bytes = c.canonical_json(manifest)
    completion = {
        "schema": "track-a-confirmation-completion-v1", "arm": "pair-trn21",
        "role": spec["role"], "label": spec["label"], "host": spec["host"],
        "hostname": spec["hostname"], "finalization_head": c.VERIFIED_BASE_HEAD,
        "verified_core_commit": c.VERIFIED_LIFECYCLE_CORE, "authorization_sha256": "1" * 64,
        "request_sha256": "2" * 64, "runtime_job_sha256": "3" * 64,
        "source_sha256": c.SOURCE_SHA, "prepare_sha256": c.PREPARE_SHA,
        "artifact_manifest_sha256": aio.digest(manifest_bytes), "artifact_members": members,
        "archive_sha256": aio.digest(values["evidence.tar.gz"]),
        "archive_bytes": len(values["evidence.tar.gz"]), "input_evidence": result["input_evidence"],
        "metrics": result["metrics"], "decision": result["decision"],
        "checkpoint_inputs_preserved": True, "candidate_checkpoint": None,
        "ready_to_retrieve": True,
    }
    if completion_mutate:
        completion_mutate(completion)
    values["artifact-manifest.json"] = manifest_bytes
    values["ready-to-retrieve.json"] = c.canonical_json(completion)
    directory = f"research/run-logs/{spec['label']}/artifacts"
    identity = {
        "archive_commit": c.VERIFIED_BASE_HEAD, "directory": directory,
        "completion_sha256": aio.digest(values["ready-to-retrieve.json"]),
        "manifest_sha256": aio.digest(values["artifact-manifest.json"]),
        "result_sha256": aio.digest(values["result.json"]),
        "control20m_sha256": aio.digest(values["control20m.metrics.json"]),
        "a120m_sha256": aio.digest(values["a1-20m.metrics.json"]),
        "finalization_head": c.VERIFIED_BASE_HEAD,
        "verified_core_commit": c.VERIFIED_LIFECYCLE_CORE,
        "authorization_sha256": "1" * 64, "request_sha256": "2" * 64,
        "runtime_job_sha256": "3" * 64, "source_sha256": c.SOURCE_SHA,
        "prepare_sha256": c.PREPARE_SHA,
    }
    return Reader(values), identity


def request_fixture(arm, core="a" * 40):
    rel = str(HERE.relative_to(ROOT))
    required = {
        c.SOURCE_PATH, "prepare.py", "README.md", "research/fresh_start_launch.py",
        "research/fresh_start_host.py", "research/fresh_start_guard.py",
        "research/fresh_start_backup.py", "research/fresh_start_sync.py",
        "research/run_experiment.py", f"{rel}/contracts.py", f"{rel}/artifact_io.py",
        f"{rel}/process_lifecycle.py", f"{rel}/confirmation_job.py",
        f"{rel}/launch_adapter.py", f"{rel}/EXPERIMENT_PROPOSAL.json",
        f"{rel}/RUNBOOK.md", f"{rel}/test_static.py",
        "research/submission-validation/20260924-v2-track-a-35c9-controls-a1/contracts.py",
        "research/submission-validation/20260924-v2-track-a-35c9-controls-a1/track_a_job.py",
    }
    source_map = {path: hashlib.sha256(path.encode()).hexdigest() for path in required}
    source_map[c.SOURCE_PATH] = c.SOURCE_SHA
    source_map["prepare.py"] = c.PREPARE_SHA
    source_map["README.md"] = c.RULES_SHA
    spec = c.ARMS[arm]
    return {
        "schema": "track-a-confirmation-request-v1", "arm": arm, "host": spec["host"],
        "expected_hostname": spec["hostname"], "remote": spec["remote"],
        "origin": "https://github.com/CsabaConsulting/trainium_frontier.git",
        "label": spec["label"], "purpose": f"authorized exact {spec['label']}",
        "min_free_bytes": c.MIN_FREE_BYTES, "environment": c.ENVIRONMENT,
        "command": c.expected_command(arm), "source_sha256": source_map,
        "request_path": f"{rel}/{arm}-request.json",
        "authorization_path": f"{rel}/{arm}-authorization.json",
        "verified_core_commit": core, "runner_owns_manifest": False, "non_overwrite": True,
    }


def pending_auth(arm, request, core="a" * 40):
    spec = c.ARMS[arm]
    return {
        "schema": "track-a-confirmation-authorization-v1",
        "state": "human_approved_pending_evidence_and_independent_verification",
        "arm": arm, "role": spec["role"], "host": spec["host"],
        "expected_hostname": spec["hostname"], "label": spec["label"],
        "source_commit": c.SOURCE_COMMIT, "source_path": c.SOURCE_PATH,
        "source_sha256": c.SOURCE_SHA, "prepare_sha256": c.PREPARE_SHA,
        "seed": 42, "max_train_seconds": 1800, "world_size": 4, "lnc": 2,
        "smoke_authorized": False, "screen_tokens": c.SCREEN_TOKENS,
        "screen_bytes": c.SCREEN_BYTES, "full_tokens": c.FULL_TOKENS,
        "full_bytes": c.FULL_BYTES, "checkpoint_prune_authorized": False,
        "cache_cleanup_authorized": False, "verified_base_head": c.VERIFIED_BASE_HEAD,
        "verified_lifecycle_core": c.VERIFIED_LIFECYCLE_CORE,
        "verified_core_commit": core, "request_path": request["request_path"],
        "request_sha256": c.digest(c.canonical_json(request)), "request_content": request,
        "prior_evidence": {key: "PENDING_GIT_BLOB_IDENTITY" for key in c.PRIOR},
        "trn21_confirmation_evidence": None if arm == "pair-trn21" else "PENDING_GIT_BLOB_IDENTITY",
    }


class SourceAndDecisionTests(unittest.TestCase):
    def test_source_prepare_and_rules_are_frozen(self):
        source = subprocess.check_output(["git", "show", f"{c.SOURCE_COMMIT}:{c.SOURCE_PATH}"], cwd=ROOT)
        self.assertEqual(hashlib.sha256(source).hexdigest(), c.SOURCE_SHA)
        self.assertEqual((ROOT / c.SOURCE_PATH).read_bytes(), source)
        self.assertEqual(hashlib.sha256((ROOT / "prepare.py").read_bytes()).hexdigest(), c.PREPARE_SHA)
        self.assertEqual(hashlib.sha256((ROOT / "README.md").read_bytes()).hexdigest(), c.RULES_SHA)

    def test_exact_a1_flags_and_no_source_edit(self):
        self.assertEqual(" ".join(c.train_argv()), "torchrun --standalone --nproc_per_node=4 train.py " + c.A1_TAIL)
        self.assertIn("--no-ngram-ve", c.CONTROL_LAUNCH)
        self.assertNotIn("--no-ngram-ve", c.A1_LAUNCH)
        self.assertEqual(c.ARMS["replicate-trn22"]["label"], "fs-20260924-v2-a1-replicate-trn22-r1")

    def test_screen_exact_boundary_and_nonfinite(self):
        self.assertEqual(c.screen_delta("1.000", "0.999"), c.Decimal("-0.001"))
        for bad in ("NaN", "Infinity", "0"):
            with self.assertRaises(SystemExit):
                c.screen_delta("1.0", bad)

    def test_adopt_requires_two_negative_and_decimal_mean(self):
        exact = c.final_decision("1.000", "0.999", "1.010", "1.009")
        self.assertEqual(exact["verdict"], "ADOPT")
        self.assertTrue(exact["adoption_authorized"])
        self.assertEqual(exact["mean_delta"], "-0.001")
        self.assertFalse(c.final_decision("1", "1", "1", "0.998")["adoption_authorized"])
        self.assertEqual(c.final_decision("1", "0.9991", "1", "0.9991")["verdict"],
                         "REPEAT_FULL20M_PAIR_ONCE")
        self.assertEqual(c.final_decision("1", "0.998", "1", "1.001")["verdict"], "DISCARD")

    def test_metric_and_disk_boundaries(self):
        self.assertEqual(c.disk_disposition(c.HARD_ABORT_BYTES - 1), "HARD_ABORT")
        self.assertEqual(c.disk_disposition(c.HARD_ABORT_BYTES), "BLOCKED")
        self.assertEqual(c.disk_disposition(c.MIN_FREE_BYTES), "ALLOWED")
        self.assertEqual(c.validate_metric(metric(c.FULL_TOKENS, c.FULL_BYTES, 1.0),
                                           c.FULL_TOKENS, c.FULL_BYTES)["num_bytes"], c.FULL_BYTES)
        attacked = metric(c.FULL_TOKENS, c.FULL_BYTES, 1.0)
        attacked["num_tokens"] -= 1
        with self.assertRaises(SystemExit):
            c.validate_metric(attacked, c.FULL_TOKENS, c.FULL_BYTES)


class LifecycleTests(unittest.TestCase):
    CHILD = (
        "import signal,sys,time\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "f=open(sys.argv[1],'ab',buffering=0)\n"
        "while True:\n f.write(b'x'); time.sleep(0.02)\n"
    )

    def run_fixture(self, timeout_parent):
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Path(directory)
            heartbeat = sandbox / "heartbeat"
            parent = (
                "import signal,subprocess,sys,time\n"
                "open('steps.jsonl','w').write('0\\n1\\n2\\n3\\n4\\n')\n"
                f"subprocess.Popen([sys.executable,'-c',{self.CHILD!r},sys.argv[1]],"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            )
            if timeout_parent:
                parent += "signal.signal(signal.SIGTERM,signal.SIG_IGN)\nwhile True: time.sleep(1)\n"
                timeout = 0.25
            else:
                parent += "time.sleep(0.2)\n"
                timeout = 2.0
            rc, _output, _wall, timed_out, observations, group = lifecycle.run_training(
                [sys.executable, "-c", parent, str(heartbeat)], sandbox, os.environ.copy(),
                timeout, term_grace=0.1, kill_grace=0.5,
            )
            size = heartbeat.stat().st_size
            time.sleep(0.1)
            self.assertEqual(heartbeat.stat().st_size, size)
            self.assertNotEqual(rc, 0)
            self.assertEqual(timed_out, timeout_parent)
            self.assertGreaterEqual(len(observations), 5)
            self.assertTrue(group["cleanup_required"])
            self.assertTrue(group["term_sent"])
            self.assertTrue(group["kill_sent"])
            self.assertTrue(group["quiescent"])
            self.assertFalse(lifecycle.group_exists(group["pgid"]))

    def test_rc0_leader_stubborn_descendant_fails_and_is_quiescent(self):
        self.run_fixture(False)

    def test_timeout_term_ignoring_group_fails_and_is_quiescent(self):
        self.run_fixture(True)


class EvidenceTests(unittest.TestCase):
    def test_all_three_authenticated_prior_archives_pass_and_nominate(self):
        readers, identities = bundle_fixtures()
        with reader_patch(readers):
            bundle = c.validate_prior_bundle(identities)
        self.assertEqual(bundle["screen_delta"], "-0.001")
        self.assertTrue(bundle["nominated"])

    def test_screen_shallower_than_threshold_rejects(self):
        readers, identities = bundle_fixtures()
        reader, identity = prior_fixture("a1-trn21", 0.9990001, 1.0)
        readers[identity["directory"]] = reader
        identities["a1-trn21"] = identity
        with reader_patch(readers), self.assertRaises(SystemExit):
            c.validate_prior_bundle(identities)

    def test_prior_host_checkpoint_and_count_mutations_reject(self):
        attacks = [
            lambda result, _receipt, _values: result.__setitem__("host", "trn22"),
            lambda result, _receipt, _values: result.__setitem__("checkpoint_path", "/tmp/other.pt"),
            lambda _result, receipt, _values: receipt["metrics"].__setitem__("num_bytes", 1),
            lambda result, _receipt, _values: result["finite_scan"]["process_group"].__setitem__("quiescent", False),
        ]
        for attack in attacks:
            readers, identities = bundle_fixtures("control-trn21", attack)
            with self.subTest(attack=attack), reader_patch(readers), self.assertRaises(SystemExit):
                c.validate_prior_bundle(identities)

    def test_prior_completion_provenance_is_externally_pinned(self):
        attacks = (
            ("finalization_head", "a" * 40), ("verified_core_commit", "b" * 40),
            ("authorization_sha256", "a" * 64), ("request_sha256", "b" * 64),
            ("runtime_job_sha256", "c" * 64), ("source_sha256", "d" * 64),
            ("checkpoint_sha256", "e" * 64), ("checkpoint_bytes", 999),
        )
        for field, bad in attacks:
            readers, identities = bundle_fixtures()
            reader, identity = prior_fixture(
                "control-trn21", 1.0,
                completion_mutate=lambda value, f=field, b=bad: value.__setitem__(f, b),
            )
            readers[identity["directory"]] = reader
            identities["control-trn21"] = identity
            with self.subTest(field=field), reader_patch(readers), self.assertRaises(SystemExit):
                c.validate_prior_bundle(identities)

    def test_external_commit_and_checkpoint_pins_reject_syntax_or_noncommit(self):
        readers, identities = bundle_fixtures()
        for field, bad in (("finalization_head", "NOT_A_COMMIT"),
                           ("verified_core_commit", "a" * 40),
                           ("archive_commit", "b" * 40),
                           ("checkpoint_path", "relative.pt")):
            attacked = copy.deepcopy(identities)
            attacked["control-trn21"][field] = bad
            with self.subTest(field=field), reader_patch(readers), self.assertRaises(SystemExit):
                c.validate_prior_bundle(attacked)

    def test_prior_screen_eval_argv_is_exact(self):
        readers, identities = bundle_fixtures(
            "control-trn21", lambda _result, receipt, _values: receipt.__setitem__("argv", ["fresh-eval"])
        )
        with reader_patch(readers), self.assertRaises(SystemExit):
            c.validate_prior_bundle(identities)

    def test_missing_late_appended_and_digest_mutations_reject(self):
        readers, identities = bundle_fixtures()
        readers[identities["control-trn21"]["directory"]].values["training.log"] += b"late"
        with reader_patch(readers), self.assertRaises(SystemExit):
            c.validate_prior_bundle(identities)
        readers, identities = bundle_fixtures()
        identities["control-trn22"]["result_sha256"] = "0" * 64
        with reader_patch(readers), self.assertRaises(SystemExit):
            c.validate_prior_bundle(identities)

    def test_authenticated_negative_trn21_confirmation_passes(self):
        readers, identities = bundle_fixtures()
        with reader_patch(readers):
            bundle = c.validate_prior_bundle(identities)
        confirmation_reader, confirmation_identity = confirmation_fixture(bundle)
        readers[confirmation_identity["directory"]] = confirmation_reader
        with reader_patch(readers):
            result = c.validate_trn21_confirmation(confirmation_identity, bundle)
        self.assertEqual(result["trn21_delta"], "-0.002")

    def test_confirmation_nonnegative_or_nonquiescent_or_mutated_rejects(self):
        readers, identities = bundle_fixtures()
        with reader_patch(readers):
            bundle = c.validate_prior_bundle(identities)
        attacks = [
            lambda result, _control, candidate, _values: (
                candidate["metrics"].update(metric(c.FULL_TOKENS, c.FULL_BYTES, 1.001)),
                result["metrics"]["a1_20m"].update(metric(c.FULL_TOKENS, c.FULL_BYTES, 1.001)),
            ),
            lambda _result, control, _candidate, _values: control["process_group"].__setitem__("quiescent", False),
            lambda result, _control, _candidate, _values: result["input_evidence"].__setitem__("screen_delta", "-9"),
        ]
        for attack in attacks:
            reader, identity = confirmation_fixture(bundle, attack)
            local = dict(readers)
            local[identity["directory"]] = reader
            with self.subTest(attack=attack), reader_patch(local), self.assertRaises(SystemExit):
                c.validate_trn21_confirmation(identity, bundle)

    def test_confirmation_completion_provenance_is_externally_pinned(self):
        readers, identities = bundle_fixtures()
        with reader_patch(readers):
            bundle = c.validate_prior_bundle(identities)
        attacks = (
            ("finalization_head", "NOT_A_COMMIT"),
            ("verified_core_commit", "NOT_A_CORE"),
            ("authorization_sha256", "a" * 64), ("request_sha256", "b" * 64),
            ("runtime_job_sha256", "c" * 64),
        )
        for field, bad in attacks:
            reader, identity = confirmation_fixture(
                bundle, completion_mutate=lambda value, f=field, b=bad: value.__setitem__(f, b)
            )
            local = dict(readers)
            local[identity["directory"]] = reader
            with self.subTest(field=field), reader_patch(local), self.assertRaises(SystemExit):
                c.validate_trn21_confirmation(identity, bundle)

    def test_confirmation_eval_argv_hash_host_and_process_are_exact(self):
        readers, identities = bundle_fixtures()
        with reader_patch(readers):
            bundle = c.validate_prior_bundle(identities)
        attacks = (
            lambda _result, control, _candidate, _values: control.__setitem__("argv", ["fresh-eval"]),
            lambda _result, control, _candidate, _values: control.__setitem__("source_sha256", "a" * 64),
            lambda _result, control, _candidate, _values: control.__setitem__("prepare_sha256", "b" * 64),
            lambda _result, control, _candidate, _values: control.__setitem__("hostname", "wrong-host"),
            lambda _result, control, _candidate, _values:
                control["process_group"].__setitem__("cleanup_required", True),
        )
        for attack in attacks:
            reader, identity = confirmation_fixture(bundle, attack)
            local = dict(readers)
            local[identity["directory"]] = reader
            with self.subTest(attack=attack), reader_patch(local), self.assertRaises(SystemExit):
                c.validate_trn21_confirmation(identity, bundle)

    def test_no_follow_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").write_bytes(b"x")
            (root / "link").symlink_to(root / "real")
            with self.assertRaises(SystemExit):
                aio.regular_bytes(root / "link")


class RequestRuntimeTests(unittest.TestCase):
    def test_exact_requests_and_pending_authorizations_fail_closed(self):
        for arm in c.ARMS:
            request = request_fixture(arm)
            self.assertRegex(c.validate_request(request, arm, "a" * 40), r"^[0-9a-f]{64}$")
            with self.assertRaises(SystemExit):
                c.validate_authorization(pending_auth(arm, request), request, arm, "a" * 40)

    def test_request_command_env_source_and_remote_mutations_reject(self):
        request = request_fixture("replicate-trn22")
        for attack in ("command", "environment", "source_sha256", "remote", "min_free_bytes"):
            value = copy.deepcopy(request)
            if attack == "command":
                value[attack][-1] = "changed"
            elif attack == "environment":
                value[attack]["NEURON_CC_FLAGS"] = "--optlevel=2"
            elif attack == "source_sha256":
                value[attack][c.SOURCE_PATH] = "f" * 64
            elif attack == "remote":
                value[attack] += "-other"
            else:
                value[attack] -= 1
            with self.subTest(attack=attack), self.assertRaises(SystemExit):
                c.validate_request(value, "replicate-trn22", "a" * 40)

    def test_concrete_and_runtime_exact_schema(self):
        request = request_fixture("pair-trn21")
        sha = c.digest(c.canonical_json(request))
        concrete = c.build_concrete(request, "b" * 64, sha, "c" * 40)
        adapter.validate_concrete(concrete, "pair-trn21")
        runtime = {**copy.deepcopy(concrete), "started_at": "now",
                   "hostname": c.ARMS["pair-trn21"]["hostname"], "git_commit": "c" * 40,
                   "source_dirty": False, "preflight_free_bytes": c.MIN_FREE_BYTES,
                   "source_sha256": copy.deepcopy(request["source_sha256"])}
        raw = json.dumps(runtime).encode()
        self.assertEqual(c.validate_runtime(runtime, concrete, raw), hashlib.sha256(raw).hexdigest())
        runtime["source_sha256"]["extra"] = "e" * 64
        with self.assertRaises(SystemExit):
            c.validate_runtime(runtime, concrete, json.dumps(runtime).encode())

    def test_exact_outer_timeouts_and_standard_schema(self):
        self.assertEqual(c.expected_command("pair-trn21")[:5],
                         ["timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "3000s"])
        self.assertEqual(c.expected_command("replicate-trn22")[:5],
                         ["timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "8400s"])

    def test_store_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            store = aio.Store(Path(directory), "label")
            store.create()
            store.write("x", b"one")
            with self.assertRaises(FileExistsError):
                store.write("x", b"two")


if __name__ == "__main__":
    unittest.main()
