from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


G1 = load("a1_g1_finalization", HERE / "g1_finalization.py")
ADJ = load("a1_g0_g1_adjudicator", HERE / "adjudicate_g0_g1.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
            "bytes": path.stat().st_size}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def tensor(tag: str) -> dict:
    return {"sha256": hashlib.sha256(tag.encode()).hexdigest(), "shape": [524_288, 128],
            "dtype": "torch.bfloat16", "numel": 67_108_864,
            "finite": True, "norm": 1.0}


class G1FinalizationContractTests(unittest.TestCase):
    def template(self):
        return json.loads((HERE / "g1-request.template.json").read_text())

    def test_template_is_exact_g1_only_and_pending(self):
        template = self.template()
        G1.validate_template(template)
        self.assertFalse(template["executable"])
        self.assertEqual(template["arm"], "G1")
        self.assertIn("--native-gate-observation", template["train_argv"])
        self.assertNotIn("--no-native-gate-observation", template["train_argv"])
        self.assertEqual(template["accepted_contract"]["r4_comparator_sha256"],
                         ADJ.COMPARATOR_SHA256)
        self.assertEqual(template["g0_prerequisite"]["corrected_replay_sha256"],
                         ADJ.G0_REPLAY_SHA256)
        self.assertEqual(template["downstream_locks"]["G2"].split("_")[0], "PENDING")
        self.assertTrue(all(template["downstream_locks"][key] is False for key in
                            ("full_a1", "cleanup", "checkpoint_prune",
                             "cache_delete", "submission")))

    def test_pending_authorization_cannot_materialize(self):
        template = self.template()
        pending = json.loads((HERE / "g1-authorization.pending.json").read_text())
        self.assertFalse(pending["executable"])
        self.assertFalse(pending["native_execution_authorized"])
        self.assertFalse(pending["g1_authorized"])
        self.assertFalse(pending["g2_authorized"])
        with self.assertRaises(G1.ContractError):
            G1.validate_authorization(pending, template, sha(HERE / "g1-request.template.json"))
        self.assertFalse((HERE / "g1-authorization.json").exists())
        self.assertFalse(Path("/tmp/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1.request.json").exists())

    def test_concrete_draft_has_exact_launcher_contract(self):
        template = self.template()
        concrete = G1.build_concrete_request(
            template, head="a" * 40, request_sha="b" * 64,
            authorization_sha="c" * 64,
            authorization_path=template["authorization_path"],
        )
        self.assertEqual(concrete["schema"], "a1-native-five-g1-concrete-request-v1")
        self.assertEqual(concrete["arm"], "G1")
        self.assertEqual(concrete["command"][:5],
                         ["timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "4200s"])
        self.assertEqual(template["child_timeout_seconds"], 3600)
        self.assertEqual(template["term_grace_seconds"], 60)
        self.assertEqual(template["min_free_bytes"], 6_979_321_856)
        self.assertEqual(template["hard_abort_bytes"], 5_368_709_120)

    def test_template_mutations_fail_closed(self):
        attacks = [
            lambda value: value.update(arm="G0"),
            lambda value: value["train_argv"].remove("--native-gate-observation"),
            lambda value: value["train_argv"].__setitem__(1, "6"),
            lambda value: value["outer_timeout_argv"].__setitem__(-1, "9999s"),
            lambda value: value["child_environment"].update(NEURON_LOGICAL_NC_CONFIG="1"),
            lambda value: value["paths"].update(report="research/run-logs/attack"),
            lambda value: value["accepted_contract"].update(r4_comparator_sha256="0" * 64),
            lambda value: value["g0_prerequisite"].update(corrected_replay_sha256="0" * 64),
            lambda value: value["downstream_locks"].update(G2="AUTHORIZED"),
            lambda value: value["downstream_locks"].update(cleanup=True),
        ]
        for attack in attacks:
            value = self.template()
            attack(value)
            with self.subTest(attack=attack), self.assertRaises(G1.ContractError):
                G1.validate_template(value)

    def test_pinned_inputs_match(self):
        template = self.template()
        for relative, expected in template["source_sha256"].items():
            self.assertEqual(sha(ROOT / relative), expected, relative)
        self.assertEqual(sha(HERE / "G1_DRAFTING_CHECKLIST_ANALYSIS.md"), G1.ANALYST_SHA)
        self.assertEqual(sha(HERE / "g0-comparator-r4-independent-verification.json"),
                         G1.VERIFIER_SHA)


class G0G1AdjudicationTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temporary.name)
        self.g1_dir = self.root / "research" / "run-logs" / ADJ.G1_LABEL / "native-gate"
        self.g1_dir.mkdir(parents=True)
        self.checkpoint = self.root / "out" / ADJ.G1_LABEL / "final.pt"
        self.checkpoint.parent.mkdir(parents=True)
        torch.save({"model_state": {"weight": torch.ones(2)}, "model_config": {},
                    "toggles": {}, "meta": {"step": 5}}, self.checkpoint)

    def tearDown(self):
        self.temporary.cleanup()

    def _g1_records(self, rank: int) -> list[dict]:
        path = ADJ.G0_REPORT_DIR / f"rank-{rank}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        header = records[0]
        header["arm"] = "G1"
        header["detailed"] = True
        header["label"] = ADJ.G1_LABEL
        detailed = []
        for step in range(5):
            for stage in ("local_pre_avg", "post_avg_preclip", "postclip"):
                suffix = f"{rank}-{step}" if stage == "local_pre_avg" else str(step)
                detailed.append({
                    "kind": "gradients", "step": step, "stage": stage,
                    "tables": {name: tensor(f"{name}-{stage}-{suffix}")
                               for name in sorted(ADJ.CMP.TABLES)},
                })
            detailed.append({"kind": "clip", "step": step, "main_norm": 1.0,
                             "table_norm": 1.0, "max_norm": 1.0,
                             "scalar_bytes_sha256": hashlib.sha256(
                                 f"clip-{step}".encode()).hexdigest()})
        return records[:-2] + detailed + records[-2:]

    def _write_reports(self, mutate=None) -> None:
        all_records = [self._g1_records(rank) for rank in range(4)]
        if mutate is not None:
            mutate(all_records)
        for rank, records in enumerate(all_records):
            path = self.g1_dir / f"rank-{rank}.jsonl"
            path.write_bytes(b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode() + b"\n" for row in records))

    def _manifest(self, *, report_mutate=None, manifest_mutate=None,
                  runtime_mutate=None, postflight_mutate=None, job_mutate=None,
                  process_mutate=None, checkpoint_payload=None):
        if checkpoint_payload is not None:
            torch.save(checkpoint_payload, self.checkpoint)
        self._write_reports(report_mutate)
        source = "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_hybrid.py"
        paths = {
            "remote": "/home/ubuntu/trainium_freshstart_20260924_v2_a1_native5_g1_hybrid_observation_on_trn21_r1",
            "compile_cache": "/home/ubuntu/neuron_cache_20260924_v2_a1_native5_g1_trn21_r1",
            "output": f"out/{ADJ.G1_LABEL}",
            "report": f"research/run-logs/{ADJ.G1_LABEL}/native-gate",
        }
        archived_logs = self.root / "research" / "run-logs"
        runtime_job_path = archived_logs / f"{ADJ.G1_LABEL}.job.json"
        process_path = archived_logs / f"{ADJ.G1_LABEL}.process.json"
        postflight = {
            "protocol_version": 2, "label": ADJ.G1_LABEL, "remote": paths["remote"],
            "stage": "postflight", "source_commit": "a" * 40,
            "command_returncode": 0, "guard_error": None, "interrupted_signals": [],
            "cleanup": {"actions": [], "remaining": [], "artifacts_deleted": [],
                        "caches_deleted": [], "driver_modified": False},
            "health": {"hostname": "ip-172-31-34-85", "neuron_process_count": 0,
                       "device_holders": {"stdout": ""}, "relevant_processes": []},
            "failures": [], "host_clean": True,
        }
        if postflight_mutate:
            postflight_mutate(postflight)
        postflight_path = archived_logs / f"{ADJ.G1_LABEL}.postflight.json"
        write_json(postflight_path, postflight)
        rank_bindings = {path.name: binding(path) for path in sorted(self.g1_dir.iterdir())}
        environment = ADJ._expected_environment(paths["compile_cache"])
        command = ADJ._expected_command(source, paths["output"], paths["report"])
        source_pins = dict(ADJ.EXPECTED_FIXED_SOURCES)
        runtime_sources = {path: "4" * 64 for path in ADJ.REQUIRED_RUNTIME_SOURCE_PATHS}
        runtime_sources.update(source_pins)
        runtime_sources.update({
            "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1_finalization.py": "1" * 64,
            "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-request.template.json": "2" * 64,
            "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-authorization.json": "3" * 64,
        })
        outer_environment = {key: value for key, value in environment.items()
                             if key not in {"NEURON_COMPILE_CACHE_DIR", "TRAINIUM_FRESH_START_RUN_ID"}}
        request_template = "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-request.template.json"
        authorization = "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1-authorization.json"
        driver = "research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g1_finalization.py"
        outer_command = [
            "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "4200s",
            "/home/ubuntu/workspace/native_venv/bin/python", driver, "run",
            "--authorized-head", "a" * 40, "--request-sha256", "b" * 64,
            "--authorization-sha256", "c" * 64, "--request-template", request_template,
            "--authorization", authorization, "--runtime-job",
            f"research/run-logs/{ADJ.G1_LABEL}.job.json",
        ]
        runtime_job = {
            "schema": "a1-native-five-g1-concrete-request-v1", "arm": "G1",
            "host": "trn21", "expected_hostname": "ip-172-31-34-85", "label": ADJ.G1_LABEL,
            "remote": paths["remote"], "expected_commit": "a" * 40,
            "origin": "https://github.com/CsabaConsulting/trainium_frontier.git",
            "purpose": "Human-approved G1-only A1 native five-update hybrid observation-on equivalence gate",
            "command": outer_command, "environment": outer_environment,
            "min_free_bytes": 6_979_321_856, "runner_owns_manifest": False,
            "non_overwrite": True, "verified_core_commit":
            "994d302b0036c56aaab63797f0561797bc4cf431",
            "verifier_receipt_commit": "8add2585d30b11ac2c50fb2701957657a1a6f30d",
            "request_template_path": request_template, "request_sha256": "b" * 64,
            "authorization_path": authorization, "authorization_sha256": "c" * 64,
            "started_at": "fixed", "hostname": "ip-172-31-34-85",
            "git_commit": "a" * 40, "source_dirty": False,
            "preflight_free_bytes": 7_000_000_000, "source_sha256": runtime_sources,
            "fingerprint_paths": sorted(runtime_sources),
        }
        if job_mutate:
            job_mutate(runtime_job)
        write_json(runtime_job_path, runtime_job)
        process = {"label": ADJ.G1_LABEL, "guard_pid": 10, "child_pid": 11,
                   "child_session": 11, "started_at": "fixed", "command": outer_command}
        if process_mutate:
            process_mutate(process)
        write_json(process_path, process)
        artifacts = {
            f"{paths['output']}/final.pt": {"bytes": self.checkpoint.stat().st_size,
                                            "sha256": sha(self.checkpoint), "mtime_ns": 1},
            f"research/run-logs/{ADJ.G1_LABEL}.child.log": {
                "bytes": 1, "sha256": "f" * 64, "mtime_ns": 1},
        }
        for name, row in rank_bindings.items():
            artifacts[f"{paths['report']}/{name}"] = {
                "bytes": row["bytes"], "sha256": row["sha256"], "mtime_ns": 1}
        runtime = {
            "schema": "a1-native-five-g1-runtime-result-v1", "arm": "G1",
            "status": "PASS", "complete": True, "label": ADJ.G1_LABEL, "host": "trn21",
            "authorized_head": "a" * 40, "request_sha256": "b" * 64,
            "authorization_sha256": "c" * 64, "runtime_job_sha256": sha(runtime_job_path),
            "command": command, "environment": environment, "returncode": 0,
            "timed_out": False, "interrupted_signals": [], "duration_seconds": 1000.0,
            "process_group_actions": [], "process_group_quiescent": True,
            "artifacts": artifacts, "compile_cache": {"path": paths["compile_cache"],
                                                       "preserved": True},
            "checkpoint_pruned": False, "cache_cleaned": False,
        }
        if runtime_mutate:
            runtime_mutate(runtime)
        runtime_path = archived_logs / f"{ADJ.G1_LABEL}.runtime-result.json"
        write_json(runtime_path, runtime)
        manifest = {
            "schema": ADJ.MANIFEST_SCHEMA, "state": "independent-evidence-finalized",
            "evidence_head": "a" * 40, "host": "trn21", "hostname": "ip-172-31-34-85",
            "label": ADJ.G1_LABEL, "source": source, "paths": paths,
            "source_sha256": source_pins, "runtime_source_sha256": runtime_sources,
            "request_sha256": "b" * 64, "authorization_sha256": "c" * 64,
            "runtime_job": binding(runtime_job_path), "runtime_result": binding(runtime_path),
            "postflight": binding(postflight_path), "process": binding(process_path),
            "checkpoint": binding(self.checkpoint), "rank_reports": rank_bindings,
            "g0_dependency": {
                "evidence_commit": "37aa3644e63e20a1427cfa69b44fd6de7e25cd81",
                "pass_receipt_commit": "8add2585d30b11ac2c50fb2701957657a1a6f30d",
                "pass_receipt_sha256": ADJ.G0_PASS_RECEIPT_SHA256,
                "corrected_replay_sha256": ADJ.G0_REPLAY_SHA256,
                "corrected_replay_bytes": 326,
                "checkpoint_sha256": ADJ.G0_CHECKPOINT_SHA256,
                "checkpoint_bytes": ADJ.G0_CHECKPOINT_BYTES,
            },
            "cleanup": False, "checkpoint_pruned": False, "cache_cleaned": False,
            "g2_authorized": False, "full_a1_authorized": False,
            "submission_authorized": False,
        }
        if manifest_mutate:
            manifest_mutate(manifest)
        manifest_path = self.root / "evidence-manifest.json"
        write_json(manifest_path, manifest)
        return manifest_path

    def _run(self, manifest_path: Path, *, verify_checkpoint=False):
        complete = next(json.loads(line) for line in
                        (ADJ.G0_REPORT_DIR / "rank-0.jsonl").read_text().splitlines()
                        if json.loads(line)["kind"] == "complete")["final_model_state"]
        checkpoint_result = {
            "sha256": sha(self.checkpoint), "bytes": self.checkpoint.stat().st_size,
            "model_state_sha256": complete["sha256"], "tensor_count": complete["tensor_count"],
        }
        checkpoint_patch = (mock.patch.object(ADJ, "_checkpoint", return_value=checkpoint_result)
                            if not verify_checkpoint else mock.patch.object(
                                ADJ, "_checkpoint", wraps=ADJ._checkpoint))
        with mock.patch.object(ADJ, "G1_ARCHIVE", self.root), \
                mock.patch.object(ADJ, "G1_REPORT_DIR", self.g1_dir), \
                mock.patch.object(ADJ, "G1_CHECKPOINT", self.checkpoint), checkpoint_patch:
            return ADJ.adjudicate(ADJ.G0_REPORT_DIR, self.g1_dir, manifest_path,
                                  sha(manifest_path), self.checkpoint)

    def test_positive_exact_g0_g1_nonperturbation(self):
        result = self._run(self._manifest())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["verdict"], "G1_OBSERVATION_NONPERTURBING")
        self.assertTrue(result["shared_records_exact"])
        self.assertTrue(result["g1_detailed_records_complete"])
        self.assertFalse(result["g2_authorized"])

    def test_shared_and_detailed_mutations_fail(self):
        def mutate_loss(rows):
            next(item for item in rows[0] if item["kind"] == "loss")["value"] += 1.0
        def drop_gradient(rows):
            rows[0].remove(next(item for item in rows[0] if item["kind"] == "gradients"))
        def mutate_postavg(rows):
            item = next(item for item in rows[1]
                        if item["kind"] == "gradients" and item["stage"] == "post_avg_preclip")
            item["tables"][sorted(ADJ.CMP.TABLES)[0]]["sha256"] = "0" * 64
        def table_owned(rows):
            next(item for item in rows[0] if item["kind"] == "inventory")["ownership"]["entries"][0]["kind"] = "rmsprop"
        def missing_state(rows):
            update = [item for item in rows[0] if item["kind"] == "update"][1]
            table = next(item for item in update["optimizer_state"]
                         if item["name"] in ADJ.CMP.TABLES)
            table["fields"].pop("exp_avg_sq")
        def gather_position(rows):
            item = next(item for item in rows[0] if item["kind"] == "collectives")
            next(row for row in item["sequence"] if row["op"] == "all_gather_into_tensor")["index"] += 1
        for attack in (mutate_loss, drop_gradient, mutate_postavg, table_owned,
                       missing_state, gather_position):
            with self.subTest(attack=attack):
                manifest = self._manifest(report_mutate=attack)
                with self.assertRaises((ADJ.InvalidEvidence, ADJ.CMP.InvalidGate)):
                    self._run(manifest)

    def test_same_fd_model_only_checkpoint_validation_positive(self):
        row = ADJ._checkpoint(self.checkpoint, sha(self.checkpoint), self.checkpoint.stat().st_size)
        self.assertEqual(row["tensor_count"], 1)
        self.assertRegex(row["model_state_sha256"], r"^[0-9a-f]{64}$")

    def test_manifest_lifecycle_and_checkpoint_mutations_fail(self):
        cases = [
            {"manifest_mutate": lambda value: value["g0_dependency"].update(
                corrected_replay_sha256="0" * 64)},
            {"manifest_mutate": lambda value: value["source_sha256"].update(
                {"prepare.py": "0" * 64})},
            {"manifest_mutate": lambda value: value.update(cleanup=True)},
            {"runtime_mutate": lambda value: value.update(timed_out=True)},
            {"runtime_mutate": lambda value: value.update(duration_seconds=3600.1)},
            {"runtime_mutate": lambda value: value.update(process_group_quiescent=False)},
            {"postflight_mutate": lambda value: value.update(host_clean=False)},
            {"job_mutate": lambda value: value["command"].__setitem__(-1, "attack.job.json")},
            {"job_mutate": lambda value: value["environment"].update(NEURON_LOGICAL_NC_CONFIG="1")},
            {"job_mutate": lambda value: value.update(git_commit="d" * 40)},
            {"job_mutate": lambda value: value["source_sha256"].update({"prepare.py": "0" * 64})},
            {"process_mutate": lambda value: value["command"].__setitem__(-1, "attack.job.json")},
            {"checkpoint_payload": {"model_state": {"weight": torch.ones(2)},
                                    "model_config": {}, "toggles": {}, "meta": {"step": 5},
                                    "optimizer": {}}},
        ]
        for case in cases:
            with self.subTest(case=case):
                manifest = self._manifest(**case)
                with self.assertRaises((ADJ.InvalidEvidence, ADJ.CMP.InvalidGate)):
                    self._run(manifest, verify_checkpoint="checkpoint_payload" in case)

    def test_external_manifest_hash_and_output_publication_fail_closed(self):
        manifest = self._manifest()
        with mock.patch.object(ADJ, "G1_ARCHIVE", self.root), \
                mock.patch.object(ADJ, "G1_REPORT_DIR", self.g1_dir), \
                mock.patch.object(ADJ, "G1_CHECKPOINT", self.checkpoint):
            with self.assertRaises(ADJ.CMP.InvalidGate):
                ADJ.adjudicate(ADJ.G0_REPORT_DIR, self.g1_dir, manifest,
                               "0" * 64, self.checkpoint)
        output = self.root / "verdict.json"
        payload = self._run(manifest)
        receipt = ADJ.CMP._write_exclusive(output, payload)
        ADJ.CMP._verify_publication_receipt(payload, receipt)
        with self.assertRaises(FileExistsError):
            ADJ.CMP._write_exclusive(output, payload)


if __name__ == "__main__":
    unittest.main()
