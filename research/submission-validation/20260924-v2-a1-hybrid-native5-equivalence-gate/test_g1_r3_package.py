#!/usr/bin/env python3
"""Cross-boundary and fail-closed tests for the non-executable G1 r3 package."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
R1_LABEL = "fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1"
R2_LABEL = "fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2"
R3_LABEL = "fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3"
OLD_OBSERVER_SHA = "89ad30056bf3fe207c43ea61f8c8a94583232f9f21e5158a80fd37cce42d02e7"
NEW_OBSERVER_SHA = "25bc72a2c4a70948915e0449d6dbc6fbcfa35e519b06a82137b55d755c88e225"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FINAL = load("g1_r3_finalization_test", HERE / "g1_finalization.py")
OBSERVER = load("a1_native_gate_observer_r3_test", ROOT / "research/a1_native_gate_observer.py")
ADJUDICATOR = load("g1_r3_adjudicator_test", HERE / "adjudicate_g0_g1.py")


class FakeDist:
    def all_reduce(self, tensor, *args, **kwargs):
        return tensor

    def all_gather_into_tensor(self, output, input_tensor, *args, **kwargs):
        return output


def value_after(argv: list[str], flag: str) -> str:
    if argv.count(flag) != 1:
        raise AssertionError(f"expected one {flag}, found {argv.count(flag)}")
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise AssertionError(f"missing value after {flag}")
    return argv[index + 1]


def parsed_runtime_contract(template: dict) -> dict:
    argv = template["train_argv"]
    return {
        "arm": value_after(argv, "--native-gate-arm"),
        "detailed": "--native-gate-observation" in argv,
        "expected_mode": "hybrid",
        "label": value_after(argv, "--native-gate-label"),
        "host": value_after(argv, "--native-gate-host"),
        "report_dir": value_after(argv, "--native-gate-report-dir"),
        "config": {
            "seed": template["seed"],
            "num_steps": int(value_after(argv, "--num-steps")),
            "max_train_seconds": int(value_after(argv, "--max-train-seconds")),
            "n_embd": int(value_after(argv, "--n-embd")),
            "seq_len": int(value_after(argv, "--seq-len")),
            "pack_factor": int(value_after(argv, "--pack-factor")),
            "matrix_lr": float(value_after(argv, "--matrix-lr")),
            "clip_after_reduce": "--clip-after-reduce" in argv,
        },
    }


class G1R3BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_bytes = (HERE / "g1-request.template.json").read_bytes()
        cls.template = json.loads(cls.template_bytes)
        cls.pending = json.loads((HERE / "g1-r3-authorization.pending.json").read_text())
        cls.contract = parsed_runtime_contract(cls.template)

    def _construct(self, root: Path, rank: int, *, label: str = R3_LABEL,
                   host: str = "trn21"):
        previous = Path.cwd()
        os.chdir(root)
        try:
            with mock.patch.dict(os.environ, self.template["child_environment"], clear=True):
                observer = OBSERVER.NativeGateObserver(
                    arm=self.contract["arm"], detailed=self.contract["detailed"],
                    expected_mode=self.contract["expected_mode"], rank=rank,
                    world_size=self.template["world_size"],
                    report_dir=self.contract["report_dir"], dist_module=FakeDist(),
                    label=label, host=host, config=self.contract["config"],
                )
            os.fsync(observer._fd)
            os.close(observer._fd)
            observer._fd = -1
            return root / observer.path
        finally:
            os.chdir(previous)

    def test_exact_request_argv_and_runtime_env_construct_all_four_real_observers(self):
        FINAL.validate_template(self.template)
        self.assertEqual(self.contract["arm"], "G1")
        self.assertTrue(self.contract["detailed"])
        self.assertEqual(self.template["child_environment"]["TRAINIUM_FRESH_START_RUN_ID"],
                         self.contract["label"])
        self.assertEqual(self.template["child_environment"]["NEURON_LOGICAL_NC_CONFIG"], "2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            headers = []
            for rank in range(4):
                path = self._construct(root, rank)
                self.assertEqual(path.name, f"rank-{rank}.jsonl")
                records = path.read_text().splitlines()
                self.assertEqual(len(records), 1)
                headers.append(json.loads(records[0]))
            for rank, header in enumerate(headers):
                self.assertEqual(header, {
                    "arm": "G1", "config": self.contract["config"], "detailed": True,
                    "host": "trn21", "kind": "header", "label": R3_LABEL,
                    "optimizer_mode": "hybrid", "rank": rank,
                    "schema": OBSERVER.REPORT_SCHEMA, "world_size": 4,
                })

    def test_r1_r2_and_wrong_host_are_rejected_for_every_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in range(4):
                for label, host in ((R1_LABEL, "trn21"), (R2_LABEL, "trn21"),
                                    (R3_LABEL, "trn22")):
                    with self.subTest(rank=rank, label=label, host=host), \
                            self.assertRaisesRegex(ValueError, "host/label contract mismatch"):
                        self._construct(root, rank, label=label, host=host)
            self.assertFalse((root / self.contract["report_dir"]).exists())

    def test_independent_template_finalizer_and_observer_literals_agree_exactly(self):
        self.assertEqual(self.template["label"], R3_LABEL)
        self.assertEqual(FINAL.G1_LABEL, R3_LABEL)
        self.assertEqual(OBSERVER.ARM_LABELS["G1"], R3_LABEL)
        self.assertEqual(ADJUDICATOR.G1_LABEL, R3_LABEL)
        self.assertEqual(OBSERVER.ARM_LABELS["G0"],
                         "fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1")
        self.assertEqual(OBSERVER.ARM_LABELS["G2"],
                         "fs-20260924-v2-a1-native5-g2-ordinary-observation-on-trn21-r1")
        self.assertEqual(self.template["source_sha256"], FINAL.EXPECTED_SOURCE_SHA256)
        for relative, digest in self.template["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                             digest, relative)

    def test_observer_diff_is_exactly_one_g1_allowlist_literal(self):
        current = (ROOT / "research/a1_native_gate_observer.py").read_bytes()
        self.assertEqual(hashlib.sha256(current).hexdigest(), NEW_OBSERVER_SHA)
        self.assertEqual(current.count(R3_LABEL.encode()), 1)
        reconstructed = current.replace(R3_LABEL.encode(), R1_LABEL.encode(), 1)
        self.assertEqual(hashlib.sha256(reconstructed).hexdigest(), OLD_OBSERVER_SHA)

        old_manifest_sha = b"8119df41c8a9ecbfc3f9e69067c7e6ac162c2d5d8ba0a15635265304dcf865a2"
        new_manifest_sha = b"5fd3b2f075191210b7e373c8321c68b2d8f880db8cc7a788d60a90db4a382007"
        manifest = (HERE / "a1_ownership_manifest.json").read_bytes()
        self.assertEqual(hashlib.sha256(manifest).hexdigest(), new_manifest_sha.decode())
        self.assertEqual(hashlib.sha256(manifest.replace(
            NEW_OBSERVER_SHA.encode(), OLD_OBSERVER_SHA.encode(), 1)).hexdigest(),
            old_manifest_sha.decode())
        comparator = (HERE / "compare_gate_reports.py").read_bytes()
        self.assertEqual(hashlib.sha256(comparator).hexdigest(),
                         "4b199343a1cecfc6be4b86242da421127c39e8dcf9f67b378ef56f75927fdd32")
        self.assertEqual(hashlib.sha256(comparator.replace(
            new_manifest_sha, old_manifest_sha, 1)).hexdigest(),
            "00f1a7263bf7367fb7ab7568822396ff04839b726c9565bb1fd6d0d70ecca17c")

    def test_production_wrapper_forwards_parsed_values_without_transformation(self):
        tree = ast.parse((HERE / "train_gate_hybrid.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "NativeGateObserver"]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: ast.unparse(item.value) for item in calls[0].keywords}
        self.assertEqual(keywords["arm"], "args.native_gate_arm")
        self.assertEqual(keywords["detailed"], "args.native_gate_observation")
        self.assertEqual(keywords["label"], "args.native_gate_label")
        self.assertEqual(keywords["host"], "args.native_gate_host")
        self.assertEqual(keywords["report_dir"], "args.native_gate_report_dir")

    def test_adjudicator_accepts_only_r3_header_before_canonical_r1_translation(self):
        def record(label: str) -> bytes:
            return (json.dumps({"kind": "header", "label": label}, sort_keys=True,
                               separators=(",", ":")) + "\n").encode()
        with mock.patch.object(ADJUDICATOR.CMP, "_parse_rank_bytes",
                               return_value=[{"parsed": True}]) as parser:
            self.assertEqual(ADJUDICATOR._parse_g1_r3_rank_bytes(record(R3_LABEL), 0),
                             [{"parsed": True}])
            adapted = parser.call_args.args[0]
            self.assertEqual(json.loads(adapted)["label"],
                             ADJUDICATOR.CMP.LABELS["G1"])
        for label in (R1_LABEL, R2_LABEL, "fs-wrong"):
            with self.subTest(label=label), self.assertRaises(ADJUDICATOR.InvalidEvidence):
                ADJUDICATOR._parse_g1_r3_rank_bytes(record(label), 0)

    def test_r3_science_and_launch_flags_match_consumed_r2_except_label_and_paths(self):
        r2_bytes = subprocess_check_output([
            "git", "show",
            "4724f67d9a55781a0dc7ef0864c93e8db2ea0fa0:"
            "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/"
            "g1-request.template.json",
        ])
        r2 = json.loads(r2_bytes)
        fixed = ("world_size", "lnc", "seed", "optimizer_updates", "in_process_max_seconds",
                 "environment", "source", "outer_timeout_argv", "child_timeout_seconds",
                 "term_grace_seconds", "kill_grace_seconds", "min_free_bytes",
                 "hard_abort_bytes", "lifecycle", "downstream_locks", "g0_prerequisite")
        for key in fixed:
            self.assertEqual(self.template[key], r2[key], key)
        r2_argv = list(r2["train_argv"])
        r3_argv = list(self.template["train_argv"])
        for flag in ("--out-dir", "--native-gate-label", "--native-gate-report-dir"):
            r2_argv[r2_argv.index(flag) + 1] = f"<{flag}>"
            r3_argv[r3_argv.index(flag) + 1] = f"<{flag}>"
        self.assertEqual(r3_argv, r2_argv)
        metadata_rebound = {
            "research/a1_native_gate_observer.py",
            "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/"
            "a1_ownership_manifest.json",
            "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/"
            "compare_gate_reports.py",
        }
        for relative, digest in r2["source_sha256"].items():
            if relative not in metadata_rebound:
                self.assertEqual(self.template["source_sha256"].get(relative), digest, relative)

    def test_r3_paths_unique_pending_and_old_authorities_invalid(self):
        encoded = json.dumps(self.template, sort_keys=True)
        for old in (R1_LABEL, R2_LABEL):
            self.assertNotIn(old, self.template["label"])
            for value in (self.template["remote"], self.template["staging_bundle_remote_path"],
                          self.template["runtime_job_path"], *self.template["paths"].values()):
                self.assertNotIn(old, value)
        self.assertNotIn("--identity-file", encoded)
        self.assertFalse(self.template["executable"])
        self.assertFalse(self.pending["executable"])
        self.assertFalse((HERE / "g1-r3-authorization.json").exists())
        self.assertFalse(FINAL.MATERIALIZATION_OUTPUT.exists())
        for relative in (*[value for key, value in self.template["paths"].items()
                            if key != "compile_cache"], self.template["runtime_job_path"]):
            self.assertFalse((ROOT / relative).exists(), relative)
        request_sha = hashlib.sha256(self.template_bytes).hexdigest()
        for name in ("g1-authorization.json", "g1-r2-authorization.json"):
            with self.subTest(name=name), self.assertRaises(FINAL.ContractError):
                FINAL.validate_authorization(json.loads((HERE / name).read_text()),
                                             self.template, request_sha)
        self.assertIn(R1_LABEL, encoded)
        self.assertIn(R2_LABEL, encoded)

    def test_amendment_provenance_and_downstream_locks_are_fail_closed(self):
        amendment = json.loads((HERE / "G1_R3_CORRECTNESS_RETRY_AMENDMENT.json").read_text())
        self.assertEqual(amendment["observer_change"]["new_sha256"], NEW_OBSERVER_SHA)
        self.assertEqual(amendment["observer_change"]["rebound_g0_replay_sha256"],
                         "5e92240808e95b445a2bd0cb2f7c816c33d814e9a3a6cd0bff97504f013b6ce1")
        self.assertEqual(amendment["observer_change"]["rebound_g0_replay_bytes"], 326)
        self.assertFalse(amendment["observer_change"]["scientific_arithmetic_change"])
        self.assertTrue(amendment["r1_transport_failure"]["historical_consumed"])
        self.assertTrue(amendment["r2_correctness_failure"]["historical_consumed"])
        self.assertEqual(amendment["r2_correctness_failure"]["training_updates"], 0)
        self.assertTrue(all(amendment[key] is False for key in (
            "request_materialization_authorized", "host_access_authorized",
            "native_execution_authorized", "g2_authorized", "full_a1_authorized",
            "cleanup_authorized", "checkpoint_prune_authorized",
            "cache_cleanup_authorized", "submission_authorized")))
        locks = self.template["downstream_locks"]
        self.assertTrue(str(locks["G2"]).startswith("PENDING_"))
        self.assertTrue(all(locks[key] is False for key in (
            "full_a1", "cleanup", "checkpoint_prune", "cache_delete", "submission")))


def subprocess_check_output(argv: list[str]) -> bytes:
    import subprocess
    return subprocess.check_output(argv, cwd=ROOT)


if __name__ == "__main__":
    unittest.main()
