from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
IMMUTABLE_G0_REPORTS = (
    HERE / "operator-evidence"
    / "fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1"
    / "runtime/research/run-logs"
    / "fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1"
    / "native-gate"
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMPARE = load("a1_gate_compare", HERE / "compare_gate_reports.py")
MODEL_CHECK = load("a1_gate_model_check", HERE / "verify_model_checkpoints.py")
OBSERVER = load("a1_gate_observer", ROOT / "research/a1_native_gate_observer.py")
G0_FINALIZATION = load("a1_gate_g0_finalization", HERE / "g0_finalization.py")
OWNERSHIP_MANIFEST = COMPARE._load_ownership_manifest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def tensor(tag: str) -> dict:
    return {"sha256": digest(tag), "shape": [1], "dtype": "torch.float32",
            "numel": 1, "finite": True, "norm": 1.0}


def model_digest(tag: str) -> dict:
    return {"sha256": digest(tag), "tensor_count": 7}


def opt_item(name: str, kind: str, completed: int) -> dict:
    fields = {}
    if completed:
        if kind == "adamw":
            fields = {"step": completed, "exp_avg": tensor(f"{name}-avg-{completed}"),
                      "exp_avg_sq": tensor(f"{name}-sq-{completed}")}
        elif kind == "muon":
            fields = {"momentum_buffer": tensor(f"{name}-mom-{completed}"),
                      "second_momentum_buffer": tensor(f"{name}-mom2-{completed}")}
        else:
            fields = {"step": completed, "exp_avg_sq": tensor(f"{name}-sq-{completed}"),
                      "bias2": float(1.0 - 0.999 ** completed)}
    return {"name": name, "kind": kind, "fields": fields}


def optimizer_state(mode: str, rank: int, completed: int) -> list[dict]:
    items = [opt_item("ngram_embeds.b5.weight", "rmsprop", completed),
             opt_item("ngram_embeds.t5.weight", "rmsprop", completed)]
    ranks = [rank] if mode == "hybrid" else list(range(4))
    for owner in ranks:
        items.append(opt_item(f"adam.{owner}", "adamw", completed))
        items.append(opt_item(f"muon.{owner}", "muon", completed))
    return sorted(items, key=lambda item: item["name"])


def collective(mode: str) -> dict:
    sequence = []
    for step in range(5):
        sequence.append({"index": len(sequence), "op": "all_reduce", "reduce_op": "AVG",
                         "shape": [1], "dtype": "torch.float32", "numel": 1})
        if mode == "hybrid":
            stride = COMPARE.OWNERSHIP_SHARD_ELEMENTS
            sequence.append({"index": len(sequence), "op": "all_gather_into_tensor",
                             "output_shape": [4 * stride], "input_shape": [stride],
                             "output_dtype": "torch.float32", "input_dtype": "torch.float32",
                             "output_numel": 4 * stride, "input_numel": stride})
        sequence.append({"index": len(sequence), "op": "all_reduce", "reduce_op": "MAX",
                         "shape": [2], "dtype": "torch.int32", "numel": 2})
    raw = json.dumps(sequence, sort_keys=True, separators=(",", ":")).encode()
    return {"kind": "collectives", "count": len(sequence), "sequence": sequence,
            "sequence_sha256": hashlib.sha256(raw).hexdigest()}


def arm_records(arm: str, rank: int) -> list[dict]:
    detailed, mode = COMPARE.ARMS[arm]
    label = COMPARE.LABELS[arm]
    records = [{"kind": "header", "schema": COMPARE.SCHEMA, "arm": arm,
                "detailed": detailed, "optimizer_mode": mode, "rank": rank,
                "world_size": 4, "label": label, "host": "trn21",
                "config": {"clip_after_reduce": True, "matrix_lr": 0.015,
                           "max_train_seconds": 1800, "n_embd": 1024, "num_steps": 5,
                           "pack_factor": 32, "seed": 42, "seq_len": 1024}}]
    groups = copy.deepcopy(OWNERSHIP_MANIFEST["groups"])
    entries = [{key: value for key, value in entry.items()
                if key not in ("name", "dtype")}
               for entry in OWNERSHIP_MANIFEST["owned_entries"]]
    replicated = [{
        "group": OWNERSHIP_MANIFEST["replicated_rmsprop_parameters"][0]["group"],
        "parameters": [
            {key: (rank if key == "device_index" else value)
             for key, value in parameter.items() if key not in ("group", "name")}
            for parameter in OWNERSHIP_MANIFEST["replicated_rmsprop_parameters"]
        ],
    }]
    ownership = None if mode == "ordinary" else {
        "entries": entries,
        "summary": {"elements_by_owner": list(COMPARE.OWNERSHIP_ELEMENTS_BY_OWNER),
                    "estimated_ms_by_owner": list(COMPARE.OWNERSHIP_ESTIMATED_MS_BY_OWNER),
                    "shard_elements": COMPARE.OWNERSHIP_SHARD_ELEMENTS,
                    "total_padded_bytes": COMPARE.OWNERSHIP_PADDED_BYTES,
                    "replicated_rmsprop_groups": replicated},
        "replicated_rmsprop_groups": [replicated[0]["group"]],
    }
    records.append({"kind": "inventory", "groups": groups, "ownership": ownership,
                    "model_state": copy.deepcopy(
                        OWNERSHIP_MANIFEST["configuration"]["model_state"])})
    for step in range(5):
        value = float(5.0 - step / 10 + rank / 100)
        records.append({"kind": "loss", "step": step, "value": value,
                        "scalar_sha256": digest(f"loss-{rank}-{step}")})
        if detailed:
            for stage in ("local_pre_avg", "post_avg_preclip", "postclip"):
                suffix = f"{rank}-{step}" if stage == "local_pre_avg" else str(step)
                records.append({"kind": "gradients", "step": step, "stage": stage,
                                "tables": {name: tensor(f"{name}-{stage}-{suffix}")
                                           for name in sorted(COMPARE.TABLES)}})
            records.append({"kind": "clip", "step": step, "main_norm": 1.0,
                            "table_norm": 1.0, "max_norm": 1.0,
                            "scalar_bytes_sha256": digest(f"clip-{step}")})
        for phase, completed in (("pre", step), ("post", step + 1)):
            records.append({
                "kind": "update", "step": step, "phase": phase,
                "tables": {name: tensor(f"{name}-{step}-{phase}")
                           for name in sorted(COMPARE.TABLES)},
                "model_state": model_digest(f"model-{step}-{phase}"),
                "optimizer_state": optimizer_state(mode, rank, completed),
                "rms_call_count": completed,
            })
    records.append(collective(mode))
    records.append({"kind": "complete", "expected_steps": 5,
                    "final_model_state": model_digest("final")})
    return records


def write_fixture(root: Path, mutation=None) -> dict[str, Path]:
    paths = {}
    all_records = {arm: [arm_records(arm, rank) for rank in range(4)]
                   for arm in COMPARE.ARMS}
    if mutation:
        mutation(all_records)
    for arm, ranks in all_records.items():
        directory = root / arm
        directory.mkdir()
        paths[arm] = directory
        for rank, records in enumerate(ranks):
            data = b"".join(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode() + b"\n" for record in records)
            (directory / f"rank-{rank}.jsonl").write_bytes(data)
    return paths


class SourceContractTests(unittest.TestCase):
    def test_protected_hashes(self):
        self.assertEqual(sha(ROOT / "submissions/dense6x1024_public0992851/train.py"),
                         "9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae")
        self.assertEqual(sha(ROOT / "prepare.py"),
                         "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb")
        self.assertNotIn("a1_native_gate_observer", (ROOT / "submissions/dense6x1024_public0992851/train.py").read_text())

    def test_gate_diff_is_enumerated(self):
        candidate = ast.parse((ROOT / "submissions/dense6x1024_public0992851/train.py").read_text())
        hybrid = ast.parse((HERE / "train_gate_hybrid.py").read_text())
        ordinary = ast.parse((HERE / "train_gate_ordinary.py").read_text())
        def definitions(tree):
            return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        base_defs, hybrid_defs, ordinary_defs = map(definitions, (candidate, hybrid, ordinary))
        self.assertEqual(set(base_defs), set(hybrid_defs))
        for name in set(base_defs) - {"main", "clip_and_sync_gradients"}:
            self.assertEqual(base_defs[name], hybrid_defs[name], name)
        for name in set(hybrid_defs) - {"_install_submission_training"}:
            self.assertEqual(hybrid_defs[name], ordinary_defs[name], name)
        self.assertIn("_submission_optimizer_install", hybrid_defs["_install_submission_training"])
        self.assertNotIn("_submission_optimizer_install", ordinary_defs["_install_submission_training"])

    def test_pending_requests_are_unique_and_non_executable(self):
        request = json.loads((HERE / "requests.pending.json").read_text())
        self.assertIs(request["executable"], False)
        self.assertEqual(request["finalization_head"], "PENDING_EXTERNAL_VERIFIER")
        self.assertEqual(request["authorization_sha256"], "PENDING_EXTERNAL_VERIFIER")
        arms = request["arms"]
        for key in ("label", "remote", "compile_cache", "output", "report"):
            self.assertEqual(len({arm[key] for arm in arms}), 3)
        self.assertEqual([arm["arm"] for arm in arms], ["G0", "G1", "G2"])
        self.assertTrue(arms[0]["prerequisite"].startswith("HISTORICAL_G0_COMPLETED_"))
        self.assertTrue(all(arm["prerequisite"].startswith("PENDING_") for arm in arms[1:]))
        self.assertEqual(request["g0_comparator_correction_request"],
                         "g0-ownership-revalidation.pending.json")
        self.assertIn("--clip-after-reduce", request["common_train_argv"])
        self.assertEqual(request["common_train_argv"].count("--num-steps"), 1)
        self.assertEqual(request["common_train_argv"][:4],
                         ["--num-steps", "5", "--max-train-seconds", "1800"])

    def test_proposal_and_runbook_refuse_authorization(self):
        proposal = json.loads((HERE / "EXPERIMENT_PROPOSAL.json").read_text())
        self.assertTrue(proposal["gates"]["independent_verification"])
        self.assertFalse(proposal["gates"]["comparator_correction_independent_verification"])
        self.assertFalse(proposal["gates"]["native_execution_authorized"])
        self.assertTrue(proposal["gates"]["launched"])
        self.assertTrue(proposal["g0_comparator_correction"]["g1_remains_locked"])
        self.assertEqual(proposal["execution_contract"]["order"], ["G0", "G1", "G2"])
        for relative, expected in proposal["immutable_sources"]["gate_sources"].items():
            self.assertEqual(sha(ROOT / relative), expected, relative)
        runbook = (HERE / "RUNBOOK.md").read_text()
        self.assertIn("RMS_PASS_INTEGRATED_BLOCKED", runbook)
        self.assertIn("STOP; below", runbook)

    def test_comparator_correction_request_is_pinned_and_non_executable(self):
        request = json.loads((HERE / "g0-ownership-revalidation.pending.json").read_text())
        self.assertFalse(request["executable"])
        self.assertFalse(request["native_execution_authorized"])
        self.assertFalse(request["training_authorized"])
        self.assertFalse(request["g1_authorized"])
        self.assertEqual(request["implementation_head"], "PENDING_INDEPENDENT_VERIFIER")
        self.assertEqual(request["pins"]["ownership_manifest_sha256"],
                         sha(HERE / "a1_ownership_manifest.json"))
        self.assertEqual(request["pins"]["comparator_sha256"],
                         sha(HERE / "compare_gate_reports.py"))
        self.assertEqual(request["expected_contract"]["elements_by_owner"],
                         [27_405_059, 30_540_035, 30_408_963, 29_362_179])
        self.assertEqual(request["expected_contract"]["owner_exchange_output"],
                         [122_160_640])


class G0FinalizationTests(unittest.TestCase):
    def template(self):
        return json.loads((HERE / "g0-request.template.json").read_text())

    def authorized(self, template):
        return {
            "schema": "a1-native-five-g0-authorization-v1",
            "state": "independent-verifier-finalized-g0-only",
            "executable": True,
            "arm": "G0", "host": "trn21", "label": template["label"],
            "verified_core_commit": G0_FINALIZATION.CORE,
            "verifier_receipt_commit": G0_FINALIZATION.VERIFIER_COMMIT,
            "verifier_receipt_sha256": G0_FINALIZATION.VERIFIER_SHA,
            "request_template_path": template["request_template_path"],
            "request_sha256": sha(HERE / "g0-request.template.json"),
            "head_binding": "external-literals-only",
            "protected_finalization_sha256": {
                relative: sha(ROOT / relative) for relative in G0_FINALIZATION.FINALIZATION_PATHS
            },
            "source_sha256": template["source_sha256"],
            "native_execution_authorized": True,
            "g1_authorized": False, "g2_authorized": False,
            "checkpoint_prune_authorized": False, "cache_cleanup_authorized": False,
        }

    def test_g0_template_is_exact_and_only_g0_materializes(self):
        template = self.template()
        G0_FINALIZATION.validate_template(template)
        request_sha = sha(HERE / "g0-request.template.json")
        auth = self.authorized(template)
        with self.assertRaisesRegex(G0_FINALIZATION.ContractError,
                                    "protected hash mismatch"):
            G0_FINALIZATION.validate_authorization(auth, template, request_sha)
        concrete = G0_FINALIZATION.build_concrete_request(
            template, head="a" * 40, request_sha=request_sha,
            authorization_sha="b" * 64,
            authorization_path=template["authorization_path"],
        )
        self.assertEqual(concrete["arm"], "G0")
        self.assertEqual(concrete["host"], "trn21")
        self.assertEqual((template["seed"], template["world_size"], template["lnc"],
                          template["optimizer_updates"], template["in_process_max_seconds"]),
                         (42, 4, 2, 5, 1800))
        self.assertEqual(concrete["command"][:5],
                         ["timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "4200s"])
        self.assertIn("--no-native-gate-observation", template["train_argv"])
        self.assertNotIn("--native-gate-observation", template["train_argv"])
        self.assertEqual(template["child_environment"]["TRAINIUM_FRESH_START_RUN_ID"],
                         template["label"])
        self.assertEqual(template["train_argv"][:6],
                         ["--num-steps", "5", "--max-train-seconds", "1800", "--n-embd", "1024"])

    def test_g1_g2_are_locked_and_have_no_authorization(self):
        template = self.template()
        self.assertEqual(set(template["downstream_locks"]), {"G1", "G2"})
        self.assertTrue(all(value.startswith("PENDING_")
                            for value in template["downstream_locks"].values()))
        pending = json.loads((HERE / "requests.pending.json").read_text())
        by_arm = {arm["arm"]: arm for arm in pending["arms"]}
        self.assertIn("PENDING_AUTHENTICATED_G0", by_arm["G1"]["prerequisite"])
        self.assertIn("PENDING_AUTHENTICATED_G1", by_arm["G2"]["prerequisite"])
        historical_g0 = json.loads((HERE / "g0-authorization.json").read_text())
        self.assertTrue(historical_g0["executable"])
        self.assertFalse(historical_g0["g1_authorized"])
        self.assertFalse(historical_g0["g2_authorized"])
        self.assertFalse(any((HERE / name).exists() for name in
                             ("g1-authorization.json", "g2-authorization.json")))

    def test_pending_authorization_is_not_executable(self):
        template = self.template()
        pending = json.loads((HERE / "g0-authorization.pending.json").read_text())
        self.assertFalse(pending["executable"])
        self.assertFalse(pending["native_execution_authorized"])
        with self.assertRaises(G0_FINALIZATION.ContractError):
            G0_FINALIZATION.validate_authorization(
                pending, template, sha(HERE / "g0-request.template.json")
            )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "must-not-exist.json"
            result = subprocess.run([
                sys.executable, str(HERE / "g0_finalization.py"), "materialize",
                "--authorized-head", "a" * 40,
                "--request-sha256", sha(HERE / "g0-request.template.json"),
                "--authorization-sha256", sha(HERE / "g0-authorization.pending.json"),
                "--request-template", str((HERE / "g0-request.template.json").relative_to(ROOT)),
                "--authorization", str((HERE / "g0-authorization.pending.json").relative_to(ROOT)),
                "--output", str(output),
            ], cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertFalse(output.exists())
            self.assertIn("G0_FINALIZATION_INVALID", result.stderr)

    def test_exact_source_launcher_guard_and_verifier_hashes(self):
        template = self.template()
        for relative, expected in template["source_sha256"].items():
            actual = sha(ROOT / relative)
            if relative.endswith("compare_gate_reports.py"):
                self.assertEqual(expected,
                                 "aa2d7c084ee17c5c639bd5ff7b6bc0cbf6e9575d5f802a056328eb4e387f5ca4")
                self.assertNotEqual(actual, expected)
            else:
                self.assertEqual(actual, expected, relative)
        self.assertEqual(template["verified_core_commit"],
                         "7528ca8dd9d9915097929217c2a2b0d9cfc17107")
        self.assertEqual(template["verifier_receipt_commit"],
                         "aeb85c2cee3481901c827d7a2313b9178906fc84")
        self.assertEqual(template["verifier_receipt_sha256"],
                         "3e2ee9548202d134bc71b9d87b6bf9808857393aa45ec0fd9f6c65ff7fecbacf")
        correction = json.loads((HERE / "g0-ownership-revalidation.pending.json").read_text())
        self.assertEqual(correction["pins"]["comparator_sha256"],
                         sha(HERE / "compare_gate_reports.py"))

    def test_request_or_environment_mutation_fails_closed(self):
        attacks = []
        for mutate in (
            lambda value: value.update(arm="G1"),
            lambda value: value["train_argv"].__setitem__(1, "6"),
            lambda value: value["outer_timeout_argv"].__setitem__(-1, "9999s"),
            lambda value: value["child_environment"].update(NEURON_LOGICAL_NC_CONFIG="1"),
            lambda value: value["downstream_locks"].update(G1="authorized"),
            lambda value: value["paths"].update(output="out/attacker"),
            lambda value: value["source_sha256"].__setitem__("prepare.py", "0" * 64),
        ):
            attacked = copy.deepcopy(self.template())
            mutate(attacked)
            attacks.append(attacked)
        for attacked in attacks:
            with self.subTest(arm=attacked.get("arm"), argv=attacked.get("train_argv", [])[:2]):
                with self.assertRaises(G0_FINALIZATION.ContractError):
                    G0_FINALIZATION.validate_template(attacked)

    def test_lifecycle_source_contains_private_group_and_bounded_quiescence(self):
        source = (HERE / "g0_finalization.py").read_text()
        for required in (
            "start_new_session=True", "os.killpg", "signal.SIGTERM", "signal.SIGKILL",
            "_group_members", "os.O_EXCL", "os.O_NOFOLLOW", "os.fsync",
            "process_group_quiescent", "checkpoint_pruned", "cache_cleaned",
        ):
            self.assertIn(required, source)

    def test_safe_root_positive_create_hash_snapshot_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trusted"
            root.mkdir()
            output = root / "out" / "label"
            report = root / "research" / "run-logs" / "label" / "native-gate"
            result = root / "research" / "run-logs" / "label.runtime-result.json"
            G0_FINALIZATION._safe_mkdir(root, output)
            G0_FINALIZATION._safe_mkdir(root, report)
            rank = report / "rank-0.jsonl"
            G0_FINALIZATION._write_exclusive(root, rank, {"kind": "complete"})
            rows, identities = G0_FINALIZATION._artifact_snapshot([rank, report], root)
            self.assertEqual(set(rows), {
                "research/run-logs/label/native-gate/rank-0.jsonl",
            })
            G0_FINALIZATION._revalidate_snapshot(root, identities)
            sealed = G0_FINALIZATION._write_exclusive(root, result, {"complete": True})
            G0_FINALIZATION._assert_sealed_file(root, result, sealed)
            self.assertEqual(G0_FINALIZATION._load(result, root), {"complete": True})

    def test_realistic_child_artifact_growth_preserves_directory_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trusted"
            root.mkdir()
            cache = root / "cache" / "g0"
            output = root / "out" / "g0"
            report = root / "research" / "run-logs" / "g0" / "native-gate"
            child_log = root / "research" / "run-logs" / "g0.child.log"
            runtime_result = root / "research" / "run-logs" / "g0.runtime-result.json"
            for path in (cache, output, report):
                G0_FINALIZATION._safe_mkdir(root, path)
            identities = {
                path: G0_FINALIZATION._directory_identity(root, path)
                for path in (cache, output, report)
            }

            # Simulate legitimate child products, including a cache subdirectory that changes
            # parent link count as well as size/mtime.
            (cache / "MODULE_1").mkdir()
            (cache / "MODULE_1" / "graph.neff").write_bytes(b"compiled")
            (output / "final.pt").write_bytes(b"model-only-checkpoint")
            for rank in range(4):
                (report / f"rank-{rank}.jsonl").write_text('{"kind":"complete"}\n')
            log_seal = G0_FINALIZATION._write_exclusive(root, child_log, {"rc": 0})

            for path, identity in identities.items():
                G0_FINALIZATION._assert_directory_identity(root, path, identity)
            G0_FINALIZATION._validate_success_artifact_allowlist(output, report, root)
            rows, artifact_identities = G0_FINALIZATION._artifact_snapshot(
                [child_log, output / "final.pt", report], root,
            )
            self.assertEqual(set(rows), {
                "research/run-logs/g0.child.log", "out/g0/final.pt",
                "research/run-logs/g0/native-gate/rank-0.jsonl",
                "research/run-logs/g0/native-gate/rank-1.jsonl",
                "research/run-logs/g0/native-gate/rank-2.jsonl",
                "research/run-logs/g0/native-gate/rank-3.jsonl",
            })
            G0_FINALIZATION._assert_sealed_file(root, child_log, log_seal)
            G0_FINALIZATION._final_success_artifact_seal(
                output, report, artifact_identities, root,
            )
            result_seal = G0_FINALIZATION._write_exclusive(
                root, runtime_result, {"complete": True, "artifacts": rows},
            )
            G0_FINALIZATION._assert_sealed_file(root, runtime_result, result_seal)

    def test_exact_output_and_report_membership_attacks_are_rejected(self):
        for location in ("output", "report"):
            for attack in ("empty-directory", "nested-file", "extra-file", "post-hash-insertion"):
                with self.subTest(location=location, attack=attack), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "trusted"
                    root.mkdir()
                    output = root / "out" / "g0"
                    report = root / "research" / "run-logs" / "g0" / "native-gate"
                    G0_FINALIZATION._safe_mkdir(root, output)
                    G0_FINALIZATION._safe_mkdir(root, report)
                    (output / "final.pt").write_bytes(b"model-only-checkpoint")
                    for rank in range(4):
                        (report / f"rank-{rank}.jsonl").write_text(
                            '{"kind":"complete"}\n',
                        )
                    G0_FINALIZATION._validate_success_artifact_allowlist(
                        output, report, root,
                    )
                    target = output if location == "output" else report
                    identities = None
                    if attack == "post-hash-insertion":
                        _, identities = G0_FINALIZATION._artifact_snapshot(
                            [output / "final.pt", report], root,
                        )
                        (target / "post-hash-extra.bin").write_bytes(b"late")
                        # Existing file seals alone intentionally cannot see a new name.
                        G0_FINALIZATION._revalidate_snapshot(root, identities)
                    elif attack == "empty-directory":
                        (target / "unexpected-empty").mkdir()
                    elif attack == "nested-file":
                        nested = target / "unexpected-nested"
                        nested.mkdir()
                        (nested / "payload.bin").write_bytes(b"nested")
                    else:
                        (target / "unexpected.bin").write_bytes(b"extra")
                    with self.assertRaises(G0_FINALIZATION.ContractError):
                        G0_FINALIZATION._validate_success_artifact_allowlist(
                            output, report, root,
                        )

    def test_final_seal_rejects_mutation_between_every_subcheck(self):
        real_revalidate = G0_FINALIZATION._revalidate_snapshot
        real_bindings = G0_FINALIZATION._validate_success_artifact_bindings
        for target_kind in ("final.pt", "rank-0.jsonl"):
            for mutation in ("replacement", "insertion"):
                for interval in ("after-membership", "after-identities"):
                    with self.subTest(
                        target=target_kind, mutation=mutation, interval=interval,
                    ), tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp) / "trusted"
                        root.mkdir()
                        output = root / "out" / "g0"
                        report = root / "research" / "run-logs" / "g0" / "native-gate"
                        G0_FINALIZATION._safe_mkdir(root, output)
                        G0_FINALIZATION._safe_mkdir(root, report)
                        (output / "final.pt").write_bytes(b"model-only-checkpoint")
                        for rank in range(4):
                            (report / f"rank-{rank}.jsonl").write_text(
                                '{"kind":"complete"}\n',
                            )
                        _, identities = G0_FINALIZATION._artifact_snapshot(
                            [output / "final.pt", report], root,
                        )
                        target = (
                            output / "final.pt" if target_kind == "final.pt"
                            else report / "rank-0.jsonl"
                        )
                        attacked = False

                        def attack_once():
                            nonlocal attacked
                            if attacked:
                                return
                            attacked = True
                            if mutation == "replacement":
                                replacement = target.parent / ".replacement.tmp"
                                replacement.write_bytes(target.read_bytes())
                                os.replace(replacement, target)
                            else:
                                (target.parent / "post-seal-extra.bin").write_bytes(b"late")

                        def revalidate_after_attack(*args, **kwargs):
                            attack_once()
                            return real_revalidate(*args, **kwargs)

                        def bindings_after_attack(*args, **kwargs):
                            attack_once()
                            return real_bindings(*args, **kwargs)

                        patched_name = (
                            "_revalidate_snapshot" if interval == "after-membership"
                            else "_validate_success_artifact_bindings"
                        )
                        patched_call = (
                            revalidate_after_attack if interval == "after-membership"
                            else bindings_after_attack
                        )
                        with mock.patch.object(
                            G0_FINALIZATION, patched_name, side_effect=patched_call,
                        ):
                            with self.assertRaises(G0_FINALIZATION.ContractError):
                                G0_FINALIZATION._final_success_artifact_seal(
                                    output, report, identities, root,
                                )
                        self.assertTrue(attacked)

    def test_verifier_report_parent_symlink_attack_is_rejected(self):
        label = "fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trusted"
            outside = Path(tmp) / "outside"
            (root / "research" / "run-logs").mkdir(parents=True)
            outside.mkdir()
            (root / "research" / "run-logs" / label).symlink_to(
                outside, target_is_directory=True,
            )
            report = root / "research" / "run-logs" / label / "native-gate"
            with self.assertRaises(G0_FINALIZATION.ContractError):
                G0_FINALIZATION._validate_chain(root, report, "absent")
            with self.assertRaises(G0_FINALIZATION.ContractError):
                G0_FINALIZATION._artifact_snapshot([report], root)
            self.assertEqual(list(outside.iterdir()), [])

    def test_output_cache_and_materialization_parent_aliases_are_rejected(self):
        for name in ("output", "cache", "materialization"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "trusted"
                outside = Path(tmp) / "outside"
                root.mkdir()
                outside.mkdir()
                (root / name).symlink_to(outside, target_is_directory=True)
                target = root / name / "fixed-terminal"
                with self.assertRaises(G0_FINALIZATION.ContractError):
                    G0_FINALIZATION._safe_mkdir(root, target)
                with self.assertRaises(G0_FINALIZATION.ContractError):
                    G0_FINALIZATION._write_exclusive(root, target, {"attack": True})
                self.assertEqual(list(outside.iterdir()), [])

    def test_path_escape_non_directory_ancestor_and_hardlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trusted"
            root.mkdir()
            with self.assertRaises(G0_FINALIZATION.ContractError):
                G0_FINALIZATION._validate_chain(root, root / ".." / "escape", "absent")
            (root / "not-a-directory").write_text("x")
            with self.assertRaises(G0_FINALIZATION.ContractError):
                G0_FINALIZATION._validate_chain(
                    root, root / "not-a-directory" / "terminal", "absent",
                )
            original = root / "authority.json"
            alias = root / "authority-hardlink.json"
            original.write_text("{}\n")
            os.link(original, alias)
            with self.assertRaises(G0_FINALIZATION.ContractError):
                G0_FINALIZATION._load(original, root)

    def test_post_hash_identity_revalidation_rejects_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trusted"
            root.mkdir()
            artifact = root / "artifact.json"
            G0_FINALIZATION._write_exclusive(root, artifact, {"version": 1})
            _, identity = G0_FINALIZATION._sha_with_identity(artifact, root)
            artifact.unlink()
            artifact.write_text('{"version": 1}\n')
            with self.assertRaises(G0_FINALIZATION.ContractError):
                G0_FINALIZATION._assert_file_identity(root, artifact, identity)

    def test_post_close_materialization_and_result_replacements_are_rejected(self):
        names = (
            "fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1.request.json",
            "g0.runtime-result.json",
        )
        original_assert = G0_FINALIZATION._assert_sealed_file
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "trusted"
                root.mkdir()
                path = root / name
                attacked = False

                def replace_after_close(trusted_root, written_path, sealed):
                    nonlocal attacked
                    if not attacked:
                        attacked = True
                        written_path.unlink()
                        written_path.write_bytes(b'{"attacker":true}\n')
                    return original_assert(trusted_root, written_path, sealed)

                with mock.patch.object(
                    G0_FINALIZATION, "_assert_sealed_file",
                    side_effect=replace_after_close,
                ):
                    with self.assertRaises(G0_FINALIZATION.ContractError):
                        G0_FINALIZATION._write_exclusive(root, path, {"authorized": True})
                self.assertTrue(attacked)


class ComparatorTests(unittest.TestCase):
    def test_immutable_g0_reports_match_exact_a1_manifest(self):
        result = COMPARE.validate_immutable_g0(IMMUTABLE_G0_REPORTS)
        self.assertEqual(result, {
            "schema": "a1-native-five-g0-ownership-validation-v1",
            "status": "PASS",
            "manifest_sha256": COMPARE.OWNERSHIP_MANIFEST_SHA256,
            "owned_entry_count": 66,
            "elements_by_owner": [27_405_059, 30_540_035, 30_408_963, 29_362_179],
            "shard_elements": 30_540_160,
            "total_padded_elements": 122_160_640,
            "owner_exchanges_per_rank": 5,
        })

    def test_immutable_g0_cli_writes_exclusive_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "g0-validation.json"
            result = subprocess.run([
                sys.executable, str(HERE / "compare_gate_reports.py"),
                "--validate-g0-only", str(IMMUTABLE_G0_REPORTS),
                "--output", str(output),
            ], cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(output.read_text())["status"], "PASS")

    def test_manifest_binds_named_a1_parameters_tables_and_sources(self):
        manifest = COMPARE._load_ownership_manifest()
        entries = {entry["name"]: entry for entry in manifest["owned_entries"]}
        self.assertEqual(len(entries), 66)
        self.assertEqual(entries["ngram_proj.b5.weight"]["shape"], [1024, 128])
        self.assertEqual(entries["ngram_proj.t5.weight"]["elements"], 131_072)
        self.assertEqual(entries["transformer.h.5.attn.ve_gate_bi.weight"]["shape"], [8, 32])
        self.assertEqual(entries["transformer.h.5.attn.ve_gate_tri.weight"]["elements"], 256)
        self.assertTrue(all(entry["dtype"] == "torch.float32" for entry in entries.values()))
        self.assertTrue(COMPARE.TABLES.isdisjoint(entries))
        self.assertEqual({entry["name"] for entry in
                          manifest["replicated_rmsprop_parameters"]}, COMPARE.TABLES)
        self.assertEqual(manifest["summary"]["owned_elements"], 117_716_236)
        self.assertEqual(manifest["summary"]["total_padded_bytes"], 488_642_560)
        for relative, expected in manifest["source_pins"]["files"].items():
            self.assertEqual(sha(ROOT / relative), expected, relative)

    def test_immutable_report_or_manifest_digest_mutation_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "native-gate"
            copied.mkdir()
            for rank in range(4):
                source = IMMUTABLE_G0_REPORTS / f"rank-{rank}.jsonl"
                (copied / source.name).write_bytes(source.read_bytes())
            rank_zero = copied / "rank-0.jsonl"
            rank_zero.write_bytes(rank_zero.read_bytes() + b" ")
            with self.assertRaises(COMPARE.InvalidGate):
                COMPARE.validate_immutable_g0(copied)

            manifest = Path(tmp) / "manifest.json"
            manifest.write_bytes((HERE / "a1_ownership_manifest.json").read_bytes() + b" ")
            with mock.patch.object(COMPARE, "OWNERSHIP_MANIFEST_PATH", manifest):
                with self.assertRaises(COMPARE.InvalidGate):
                    COMPARE._load_ownership_manifest()

    def test_source_pin_mutation_is_invalid(self):
        original = COMPARE._file_sha

        def attacked(path):
            if path == ROOT / "submissions/dense6x1024_public0992851/train.py":
                return "0" * 64
            return original(path)

        with mock.patch.object(COMPARE, "_file_sha", side_effect=attacked):
            with self.assertRaises(COMPARE.InvalidGate):
                COMPARE._load_ownership_manifest()

    def test_exact_fixture_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_fixture(Path(tmp))
            result = COMPARE.compare(paths["G0"], paths["G1"], paths["G2"])
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["rms_repair_verdict"], "RMS_PASS")

    def test_table_difference_is_rms_fail(self):
        def mutate(records):
            gradient = next(r for r in records["G2"][0] if r["kind"] == "gradients")
            gradient["tables"]["ngram_embeds.b5.weight"]["sha256"] = digest("attack")
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_fixture(Path(tmp), mutate)
            result = COMPARE.compare(paths["G0"], paths["G1"], paths["G2"])
            self.assertEqual(result["status"], "RMS_FAIL")

    def test_non_table_difference_blocks_integrated_only(self):
        def mutate(records):
            update = next(r for r in records["G2"][0] if r["kind"] == "update")
            update["model_state"]["sha256"] = digest("integrated-attack")
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_fixture(Path(tmp), mutate)
            result = COMPARE.compare(paths["G0"], paths["G1"], paths["G2"])
            self.assertEqual(result["status"], "RMS_PASS_INTEGRATED_BLOCKED")
            self.assertEqual(result["rms_repair_verdict"], "RMS_PASS")

    def test_observation_perturbation_is_invalid(self):
        def mutate(records):
            loss = next(r for r in records["G1"][0] if r["kind"] == "loss")
            loss["scalar_sha256"] = digest("observer-perturbed")
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_fixture(Path(tmp), mutate)
            with self.assertRaises(COMPARE.InvalidGate):
                COMPARE.compare(paths["G0"], paths["G1"], paths["G2"])

    def test_missing_record_is_invalid(self):
        def mutate(records):
            records["G1"][2] = [r for r in records["G1"][2]
                                if not (r["kind"] == "clip" and r["step"] == 3)]
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_fixture(Path(tmp), mutate)
            with self.assertRaises(COMPARE.InvalidGate):
                COMPARE.compare(paths["G0"], paths["G1"], paths["G2"])

    def test_wrong_collective_digest_is_invalid(self):
        def mutate(records):
            record = next(r for r in records["G1"][1] if r["kind"] == "collectives")
            record["sequence_sha256"] = digest("wrong")
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_fixture(Path(tmp), mutate)
            with self.assertRaises(COMPARE.InvalidGate):
                COMPARE.compare(paths["G0"], paths["G1"], paths["G2"])

    def test_wrong_owner_all_gather_contract_is_invalid(self):
        """Five arbitrary gathers must not be normalized as the owner exchange."""
        def mutate(records):
            for arm in ("G0", "G1"):
                for rank_records in records[arm]:
                    record = next(r for r in rank_records if r["kind"] == "collectives")
                    for item in record["sequence"]:
                        if item["op"] == "all_gather_into_tensor":
                            item.update({
                                "output_shape": [28], "input_shape": [7],
                                "output_dtype": "torch.float64",
                                "input_dtype": "torch.float64",
                                "output_numel": 28, "input_numel": 7,
                            })
                    raw = json.dumps(record["sequence"], sort_keys=True,
                                     separators=(",", ":")).encode()
                    record["sequence_sha256"] = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_fixture(Path(tmp), mutate)
            with self.assertRaises(COMPARE.InvalidGate):
                COMPARE.compare(paths["G0"], paths["G1"], paths["G2"])

    def test_stale_dense_owner_contract_is_invalid_for_a1(self):
        def mutate(records):
            dense_stride = 30_414_976
            for arm in ("G0", "G1"):
                for rank_records in records[arm]:
                    inventory = next(r for r in rank_records if r["kind"] == "inventory")
                    inventory["ownership"]["summary"].update({
                        "elements_by_owner": [27_265_796, 30_414_851, 30_412_805, 29_360_128],
                        "shard_elements": dense_stride,
                        "total_padded_bytes": dense_stride * 4 * 4,
                    })
                    collective = next(r for r in rank_records if r["kind"] == "collectives")
                    for item in collective["sequence"]:
                        if item["op"] == "all_gather_into_tensor":
                            item.update({
                                "input_shape": [dense_stride], "input_numel": dense_stride,
                                "output_shape": [4 * dense_stride],
                                "output_numel": 4 * dense_stride,
                            })
                    raw = json.dumps(collective["sequence"], sort_keys=True,
                                     separators=(",", ":")).encode()
                    collective["sequence_sha256"] = hashlib.sha256(raw).hexdigest()

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_fixture(Path(tmp), mutate)
            with self.assertRaises(COMPARE.InvalidGate):
                COMPARE.compare(paths["G0"], paths["G1"], paths["G2"])

    def test_owner_all_gather_mutations_are_invalid(self):
        def reseal(record):
            for index, item in enumerate(record["sequence"]):
                item["index"] = index
            record["count"] = len(record["sequence"])
            raw = json.dumps(record["sequence"], sort_keys=True,
                             separators=(",", ":")).encode()
            record["sequence_sha256"] = hashlib.sha256(raw).hexdigest()

        def mutation(case):
            def mutate(records):
                for arm in ("G0", "G1"):
                    for rank_records in records[arm]:
                        record = next(r for r in rank_records if r["kind"] == "collectives")
                        sequence = record["sequence"]
                        gathers = [i for i, item in enumerate(sequence)
                                   if item["op"] == "all_gather_into_tensor"]
                        if case in ("stride", "inventory_extra"):
                            inventory = next(r for r in rank_records if r["kind"] == "inventory")
                            if case == "stride":
                                inventory["ownership"]["summary"]["shard_elements"] += 128
                            else:
                                inventory["ownership"]["summary"]["unexpected"] = True
                        elif case == "missing":
                            sequence.pop(gathers[2])
                        elif case == "extra":
                            sequence.insert(gathers[2], copy.deepcopy(sequence[gathers[2]]))
                        elif case == "order":
                            item = sequence.pop(gathers[2])
                            sequence.insert(gathers[2] - 1, item)
                        else:
                            item = sequence[gathers[2]]
                            if case == "shape":
                                item["input_shape"] = [COMPARE.OWNERSHIP_SHARD_ELEMENTS - 1]
                            elif case == "dtype":
                                item["input_dtype"] = "torch.bfloat16"
                            elif case == "numel":
                                item["output_numel"] -= 1
                            elif case == "output":
                                item["output_shape"] = [COMPARE.OWNERSHIP_PADDED_ELEMENTS - 1]
                        reseal(record)
            return mutate

        for case in ("missing", "extra", "order", "shape", "dtype", "numel", "output",
                     "stride", "inventory_extra"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                paths = write_fixture(Path(tmp), mutation(case))
                with self.assertRaises(COMPARE.InvalidGate):
                    COMPARE.compare(paths["G0"], paths["G1"], paths["G2"])

    def test_exact_named_a1_manifest_mutations_are_invalid(self):
        def mutation(case):
            def mutate(records):
                targets = [rank_records for arm in ("G0", "G1")
                           for rank_records in records[arm]]
                for rank_records in targets:
                    inventory = next(r for r in rank_records if r["kind"] == "inventory")
                    ownership = inventory["ownership"]
                    if case == "drop":
                        ownership["entries"].pop()
                    elif case == "rename":
                        inventory["groups"][7]["parameter_names"][0] = "ngram_proj.attacker.weight"
                    elif case == "resize":
                        ownership["entries"][28]["shape"] = [1024, 127]
                        ownership["entries"][28]["elements"] = 1024 * 127
                    elif case == "reassign":
                        ownership["entries"][28]["owner"] = 3
                    elif case == "group":
                        ownership["entries"][28]["group"] = 8
                    elif case == "table_inclusion":
                        ownership["entries"].append({
                            "index": 66, "group": 4, "position": 0, "kind": "rmsprop",
                            "shape": [524288, 128], "elements": 67108864,
                            "estimated_ms": 1.0, "owner": 0, "offset": 0,
                        })
                    elif case == "count":
                        ownership["summary"]["elements_by_owner"][0] += 1
                    elif case == "table_dtype":
                        ownership["summary"]["replicated_rmsprop_groups"][0][
                            "parameters"][0]["dtype"] = "torch.float32"
            return mutate

        for case in ("drop", "rename", "resize", "reassign", "group",
                     "table_inclusion", "count", "table_dtype"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                paths = write_fixture(Path(tmp), mutation(case))
                with self.assertRaises(COMPARE.InvalidGate):
                    COMPARE.compare(paths["G0"], paths["G1"], paths["G2"])

    def test_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.json"
            COMPARE._write_exclusive(output, {"status": "first"})
            with self.assertRaises(FileExistsError):
                COMPARE._write_exclusive(output, {"status": "overwrite"})


class ObserverAndCheckpointTests(unittest.TestCase):
    class FakeDist:
        class ReduceOp:
            AVG = "AVG"
        def all_reduce(self, tensor, *args, **kwargs):
            return tensor
        def all_gather_into_tensor(self, output, tensor, *args, **kwargs):
            output[:tensor.numel()].copy_(tensor.reshape(-1))
            return output

    class TinyTables(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ngram_embeds = torch.nn.ModuleDict({
                "b5": torch.nn.Embedding(2, 2), "t5": torch.nn.Embedding(2, 2)})

    def test_observer_arm_mode_and_exclusive_rank_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.TinyTables()
            optimizer = SimpleNamespace(param_groups=[{
                "kind": "rmsprop", "params": list(model.parameters()), "betas": (0.0, 0.999)}],
                state={}, _rmsprop_step=lambda group: None)
            ownership = SimpleNamespace(entries=[], summary={"replicated_rmsprop_groups": [{"group": 0}]},
                                        replicated_rmsprop_group_identity=(0,),
                                        replicated_rmsprop_calls=0, groups=[{"params": []}])
            observer = OBSERVER.NativeGateObserver(
                arm="G0", detailed=False, expected_mode="hybrid", rank=0, world_size=4,
                report_dir=tmp, dist_module=self.FakeDist(),
                label=OBSERVER.ARM_LABELS["G0"], host="trn21", config={})
            observer.bind(model, optimizer, ownership)
            with self.assertRaises(FileExistsError):
                OBSERVER.NativeGateObserver(
                    arm="G0", detailed=False, expected_mode="hybrid", rank=0, world_size=4,
                    report_dir=tmp, dist_module=self.FakeDist(),
                    label=OBSERVER.ARM_LABELS["G0"], host="trn21", config={})
            os.close(observer._fd)
        with self.assertRaises(ValueError):
            OBSERVER.NativeGateObserver(
                arm="G2", detailed=False, expected_mode="ordinary", rank=0, world_size=4,
                report_dir=tempfile.mkdtemp(), dist_module=self.FakeDist(),
                label=OBSERVER.ARM_LABELS["G2"], host="trn21", config={})

    def test_model_checkpoint_schema_rejects_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final.pt"
            payload = {"model_state": {"weight": torch.ones(2)}, "model_config": {},
                       "toggles": {}, "meta": {"step": 5}}
            torch.save(payload, path)
            loaded, _, state_sha = MODEL_CHECK._checkpoint(path)
            self.assertEqual(loaded["meta"]["step"], 5)
            self.assertEqual(len(state_sha), 64)
            payload["optimizer"] = {}
            attack = Path(tmp) / "attack.pt"
            torch.save(payload, attack)
            with self.assertRaises(ValueError):
                MODEL_CHECK._checkpoint(attack)


if __name__ == "__main__":
    unittest.main()
