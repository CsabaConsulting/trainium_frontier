from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMPARE = load("a1_gate_compare", HERE / "compare_gate_reports.py")
MODEL_CHECK = load("a1_gate_model_check", HERE / "verify_model_checkpoints.py")
OBSERVER = load("a1_gate_observer", ROOT / "research/a1_native_gate_observer.py")


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
            sequence.append({"index": len(sequence), "op": "all_gather_into_tensor",
                             "output_shape": [4], "input_shape": [1],
                             "output_dtype": "torch.float32", "input_dtype": "torch.float32",
                             "output_numel": 4, "input_numel": 1})
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
    groups = [{"index": 0, "kind": "adamw", "parameter_names": [f"adam.{rank}"]},
              {"index": 1, "kind": "muon", "parameter_names": [f"muon.{rank}"]},
              {"index": 2, "kind": "rmsprop",
               "parameter_names": sorted(COMPARE.TABLES)}]
    ownership = None if mode == "ordinary" else {
        "entries": [{"kind": "adamw"}, {"kind": "muon"}],
        "summary": {"replicated_rmsprop_groups": [{"group": 2}]},
        "replicated_rmsprop_groups": [2],
    }
    records.append({"kind": "inventory", "groups": groups, "ownership": ownership,
                    "model_state": model_digest("initial")})
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
        self.assertTrue(all(arm["prerequisite"].startswith("PENDING_") for arm in arms))
        self.assertIn("--clip-after-reduce", request["common_train_argv"])
        self.assertEqual(request["common_train_argv"].count("--num-steps"), 1)
        self.assertEqual(request["common_train_argv"][:4],
                         ["--num-steps", "5", "--max-train-seconds", "1800"])

    def test_proposal_and_runbook_refuse_authorization(self):
        proposal = json.loads((HERE / "EXPERIMENT_PROPOSAL.json").read_text())
        self.assertFalse(proposal["gates"]["independent_verification"])
        self.assertFalse(proposal["gates"]["native_execution_authorized"])
        self.assertFalse(proposal["gates"]["launched"])
        self.assertEqual(proposal["execution_contract"]["order"], ["G0", "G1", "G2"])
        for relative, expected in proposal["immutable_sources"]["gate_sources"].items():
            self.assertEqual(sha(ROOT / relative), expected, relative)
        runbook = (HERE / "RUNBOOK.md").read_text()
        self.assertIn("RMS_PASS_INTEGRATED_BLOCKED", runbook)
        self.assertIn("STOP; below", runbook)


class ComparatorTests(unittest.TestCase):
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
