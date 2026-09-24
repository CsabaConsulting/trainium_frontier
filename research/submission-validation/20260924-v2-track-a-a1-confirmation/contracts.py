#!/usr/bin/env python3
"""Fail-closed contracts for the Track-A A1 promotion/confirmation lane."""

from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE_DIR = ROOT / "research/submission-validation/20260924-v2-track-a-35c9-controls-a1"
SOURCE_COMMIT = "c51f6fdc5ca77793a1524148a563cafac0ab22b0"
SOURCE_PATH = "submissions/dense6x1024_public0992851/train.py"
SOURCE_SHA = "35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2"
PREPARE_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
RULES_SHA = "4e4d39d5ef9d4bd69bc2048821292ce74f2ea99b684a823f14f3b01de2b11316"
VERIFIED_BASE_HEAD = "65155c668efaa215a2136f67617b44ed32d4c9f6"
VERIFIED_LIFECYCLE_CORE = "f58ab4a0d3fbfe8aa5acfb1c5b1cbdc15aaa2649"
BRANCH = "experiment/20260924-v2-track-a-a1-confirmation"
MIN_FREE_BYTES = 13 * 1024**3 // 2
HARD_ABORT_BYTES = 5 * 1024**3
SCREEN_TOKENS = 2_097_152
SCREEN_BYTES = 8_418_678
FULL_TOKENS = 20_971_520
FULL_BYTES = 83_462_186
CC_FLAGS = "--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
PYTHON = "/home/ubuntu/workspace/native_venv/bin/python"
PREFIX = "NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py "
CONTROL_TAIL = (
    "--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct "
    "--nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce "
    "--seq-len 1024 --no-eval-public --matrix-lr 0.015"
)
A1_TAIL = (
    "--n-embd 1024 --ngram-ve --ngram-ve-layers=-1 --ngram-ve-table-mult 64 "
    "--ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr "
    "--ngram-ve-bf16 --ngram-ve-clip-exclude --no-qk-shift --compile-sdpa-direct "
    "--nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce "
    "--seq-len 1024 --no-eval-public --matrix-lr 0.015"
)
CONTROL_LAUNCH = PREFIX + CONTROL_TAIL
A1_LAUNCH = PREFIX + A1_TAIL
ENVIRONMENT = {
    "HOME": "/home/ubuntu", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    "LOGNAME": "ubuntu", "NEURON_CC_FLAGS": CC_FLAGS,
    "NEURON_COMPETITION_R1_SEED": "42", "NEURON_LOGICAL_NC_CONFIG": "2",
    "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv",
    "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
    "PYTHONUNBUFFERED": "1", "TMPDIR": "/tmp", "USER": "ubuntu",
}
ARMS = {
    "pair-trn21": {
        "role": "paired_20m_eval", "host": "trn21", "hostname": "ip-172-31-34-85",
        "label": "fs-20260924-v2-a1-20m-pair-trn21-r1",
        "remote": "/home/ubuntu/trainium_freshstart_20260924_v2_a1_20m_pair_trn21_r1",
        "outer_timeout": "3000s",
    },
    "replicate-trn22": {
        "role": "other_host_replicate", "host": "trn22", "hostname": "ip-172-31-32-253",
        "label": "fs-20260924-v2-a1-replicate-trn22-r1",
        "remote": "/home/ubuntu/trainium_freshstart_20260924_v2_a1_replicate_trn22_r1",
        "outer_timeout": "8400s",
    },
}
PRIOR = {
    "control-trn21": {
        "arm": "control-trn21", "role": "control", "host": "trn21",
        "hostname": "ip-172-31-34-85", "label": "fs-20260924-v2-35c9-lr015-control-trn21-r1",
        "launch": CONTROL_LAUNCH,
        "remote": "/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1",
    },
    "control-trn22": {
        "arm": "control-trn22", "role": "control", "host": "trn22",
        "hostname": "ip-172-31-32-253", "label": "fs-20260924-v2-35c9-lr015-control-trn22-r1",
        "launch": CONTROL_LAUNCH,
        "remote": "/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1",
    },
    "a1-trn21": {
        "arm": "a1-trn21", "role": "a1", "host": "trn21",
        "hostname": "ip-172-31-34-85", "label": "fs-20260924-v2-a1-lastlayer-ngram64d128-trn21-r1",
        "launch": A1_LAUNCH,
        "remote": "/home/ubuntu/trainium_freshstart_20260924_v2_a1_lastlayer_ngram64d128_trn21_r1",
    },
}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
METRIC_KEYS = {"val_bpb", "num_tokens", "num_bytes", "total_nats", "eval_seconds", "eval_timeout", "eval_over_timeout"}
PRIOR_ARTIFACTS = {
    "contract-check.log", "contract-check-process.json", "training.log", "training-process.json",
    "training-steps.jsonl", "checkpoint-finite.log", "checkpoint-finite.json", "screen2m.log",
    "screen2m.metrics.json", "cache-observations.json", "result.json", "evidence.tar.gz",
}
PRIOR_ARCHIVE_MEMBERS = PRIOR_ARTIFACTS - {"evidence.tar.gz"}
OUTPUT_ARTIFACTS = {
    "pair-trn21": {
        "input-evidence.json", "control20m.log", "control20m.metrics.json",
        "a1-20m.log", "a1-20m.metrics.json", "cache-observations.json", "decision.json",
        "result.json", "evidence.tar.gz",
    },
    "replicate-trn22": {
        "input-evidence.json", "control20m.log", "control20m.metrics.json",
        "contract-check.log", "contract-check-process.json", "training.log",
        "training-process.json", "training-steps.jsonl", "checkpoint-finite.log",
        "checkpoint-finite.json", "a1-screen2m.log", "a1-screen2m.metrics.json",
        "a1-20m.log", "a1-20m.metrics.json", "cache-observations.json", "decision.json",
        "result.json", "evidence.tar.gz",
    },
}


def canonical_json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def valid_digest(value):
    return type(value) is str and HEX64.fullmatch(value) is not None and value != "0" * 64


def json_object(data, name, canonical=False):
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid {name} JSON") from error
    if type(value) is not dict or (canonical and canonical_json(value) != data):
        raise SystemExit(f"invalid {name} object/serialization")
    return value


def disk_disposition(free_bytes):
    if type(free_bytes) is not int or free_bytes < HARD_ABORT_BYTES:
        return "HARD_ABORT"
    if free_bytes < MIN_FREE_BYTES:
        return "BLOCKED"
    return "ALLOWED"


def validate_clean_lifecycle(value):
    keys = {"pgid", "leader_returncode", "cleanup_required", "term_sent", "kill_sent",
            "quiescent", "reaped_descendants", "term_grace_seconds", "kill_grace_seconds"}
    if type(value) is not dict or set(value) != keys:
        raise SystemExit("process lifecycle schema mismatch")
    if (type(value["pgid"]) is not int or value["pgid"] <= 0 or value["leader_returncode"] != 0
            or value["cleanup_required"] is not False or value["term_sent"] is not False
            or value["kill_sent"] is not False or value["quiescent"] is not True
            or value["reaped_descendants"] != [] or type(value["term_grace_seconds"]) is not float
            or value["term_grace_seconds"] <= 0 or type(value["kill_grace_seconds"]) is not float
            or value["kill_grace_seconds"] <= 0):
        raise SystemExit("successful subprocess was not independently quiescent")
    return value


def validate_metric(value, tokens, nbytes):
    if type(value) is not dict or set(value) != METRIC_KEYS:
        raise SystemExit("metric schema mismatch")
    for key in ("val_bpb", "total_nats", "eval_seconds"):
        if type(value[key]) is not float or value[key] <= 0 or not math.isfinite(value[key]):
            raise SystemExit(f"invalid metric {key}")
    if value["num_tokens"] != tokens or type(value["num_tokens"]) is not int:
        raise SystemExit("metric token count mismatch")
    if value["num_bytes"] != nbytes or type(value["num_bytes"]) is not int:
        raise SystemExit("metric byte count mismatch")
    if (value["eval_timeout"] != 1200 or type(value["eval_timeout"]) is not int
            or value["eval_over_timeout"] is not False or value["eval_seconds"] > 1200.0):
        raise SystemExit("metric timeout mismatch")
    expected = value["total_nats"] / (value["num_bytes"] * math.log(2.0))
    if not math.isclose(value["val_bpb"], expected, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit("metric arithmetic mismatch")
    return value


def parse_metric_output(text, tokens, nbytes):
    decoder = json.JSONDecoder()
    found = []
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and METRIC_KEYS.intersection(value):
            found.append(value)
    if len(found) != 1:
        raise SystemExit(f"expected one metric object, got {len(found)}")
    return validate_metric(found[0], tokens, nbytes)


def decimal_value(value, name):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise SystemExit(f"invalid decimal {name}") from error
    if not result.is_finite() or result <= 0:
        raise SystemExit(f"nonfinite/nonpositive decimal {name}")
    return result


def screen_delta(control_bpb, candidate_bpb):
    return decimal_value(candidate_bpb, "candidate screen") - decimal_value(control_bpb, "control screen")


def final_decision(control21, candidate21, control22, candidate22):
    delta21 = decimal_value(candidate21, "candidate trn21") - decimal_value(control21, "control trn21")
    delta22 = decimal_value(candidate22, "candidate trn22") - decimal_value(control22, "control trn22")
    mean = (delta21 + delta22) / Decimal(2)
    adopt = delta21 < 0 and delta22 < 0 and mean <= Decimal("-0.001")
    if adopt:
        verdict = "ADOPT"
    elif delta21 >= 0 or delta22 >= 0:
        verdict = "DISCARD"
    else:
        verdict = "REPEAT_FULL20M_PAIR_ONCE"
    return {
        "schema": "track-a-final-decision-v1", "delta_definition": "candidate_minus_same_host_control",
        "metric_population": "fresh20m_seq1024_tokens20971520_bytes83462186",
        "trn21_delta": str(delta21), "trn22_delta": str(delta22), "mean_delta": str(mean),
        "both_strictly_negative": delta21 < 0 and delta22 < 0,
        "mean_at_most_negative_0_001": mean <= Decimal("-0.001"),
        "other_host_alone_confirms": False, "verdict": verdict, "adoption_authorized": adopt,
    }


def train_argv():
    words = shlex.split(A1_LAUNCH)
    if words.pop(0) != "NEURON_LOGICAL_NC_CONFIG=2":
        raise SystemExit("A1 launch environment prefix mismatch")
    return words


def expected_eval_argv(checkpoint_path, tokens):
    return [
        PYTHON, "prepare.py", "eval-public", "--train-py", "train.py",
        "--checkpoint", checkpoint_path, "--device", "neuron", "--batch-size", "1",
        "--seq-len", "1024", "--eval-tokens", str(tokens), "--timeout-seconds", "1200",
    ]


def expected_command(arm):
    spec = ARMS[arm]
    rel = str(HERE.relative_to(ROOT))
    return [
        "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", spec["outer_timeout"],
        PYTHON, f"{rel}/confirmation_job.py", "--arm", arm,
        "--authorization", f"{rel}/{arm}-authorization.json",
        "--request", f"{rel}/{arm}-request.json",
        "--runtime-job", f"research/run-logs/{spec['label']}.job.json",
    ]


def validate_request(request, arm, core):
    spec = ARMS[arm]
    keys = {"schema", "arm", "host", "expected_hostname", "remote", "origin", "label", "purpose",
            "min_free_bytes", "environment", "command", "source_sha256", "request_path",
            "authorization_path", "verified_core_commit", "runner_owns_manifest", "non_overwrite"}
    if type(request) is not dict or set(request) != keys or request["schema"] != "track-a-confirmation-request-v1":
        raise SystemExit("request schema mismatch")
    rel = str(HERE.relative_to(ROOT))
    exact = {
        "arm": arm, "host": spec["host"], "expected_hostname": spec["hostname"],
        "remote": spec["remote"], "label": spec["label"], "min_free_bytes": MIN_FREE_BYTES,
        "environment": ENVIRONMENT, "command": expected_command(arm),
        "request_path": f"{rel}/{arm}-request.json",
        "authorization_path": f"{rel}/{arm}-authorization.json",
        "verified_core_commit": core, "runner_owns_manifest": False, "non_overwrite": True,
    }
    for key, expected in exact.items():
        if request.get(key) != expected:
            raise SystemExit(f"request mismatch: {key}")
    if type(request["origin"]) is not str or not request["origin"].startswith("https://github.com/"):
        raise SystemExit("request origin mismatch")
    if type(request["purpose"]) is not str or spec["label"] not in request["purpose"]:
        raise SystemExit("request purpose mismatch")
    mandatory = {SOURCE_PATH, "prepare.py", "README.md", "research/fresh_start_launch.py",
                 "research/fresh_start_host.py", "research/fresh_start_guard.py",
                 "research/fresh_start_backup.py", "research/fresh_start_sync.py",
                 "research/run_experiment.py", str(HERE.relative_to(ROOT) / "contracts.py"),
                 str(HERE.relative_to(ROOT) / "artifact_io.py"),
                 str(HERE.relative_to(ROOT) / "process_lifecycle.py"),
                 str(HERE.relative_to(ROOT) / "confirmation_job.py"),
                 str(HERE.relative_to(ROOT) / "launch_adapter.py"),
                 str(HERE.relative_to(ROOT) / "EXPERIMENT_PROPOSAL.json"),
                 str(HERE.relative_to(ROOT) / "RUNBOOK.md"),
                 str(HERE.relative_to(ROOT) / "test_static.py"),
                 "research/submission-validation/20260924-v2-track-a-35c9-controls-a1/contracts.py",
                 "research/submission-validation/20260924-v2-track-a-35c9-controls-a1/track_a_job.py"}
    if set(request["source_sha256"]) != mandatory:
        raise SystemExit("request protected source set mismatch")
    if any(not isinstance(path, str) or not valid_digest(value) for path, value in request["source_sha256"].items()):
        raise SystemExit("request protected source digest malformed")
    immutable = {SOURCE_PATH: SOURCE_SHA, "prepare.py": PREPARE_SHA, "README.md": RULES_SHA}
    if any(request["source_sha256"].get(path) != expected for path, expected in immutable.items()):
        raise SystemExit("immutable source/prepare/rules mismatch")
    return digest(canonical_json(request))


def validate_identity(value, kind):
    if kind == "prior":
        keys = {"archive_commit", "directory", "completion_sha256", "manifest_sha256",
                "result_sha256", "metrics_sha256", "finalization_head",
                "verified_core_commit", "authorization_sha256", "request_sha256",
                "runtime_job_sha256", "source_sha256", "prepare_sha256",
                "checkpoint_path", "checkpoint_sha256", "checkpoint_bytes"}
    elif kind == "confirmation":
        keys = {"archive_commit", "directory", "completion_sha256", "manifest_sha256",
                "result_sha256", "control20m_sha256", "a120m_sha256",
                "finalization_head", "verified_core_commit", "authorization_sha256",
                "request_sha256", "runtime_job_sha256", "source_sha256", "prepare_sha256"}
    else:
        raise SystemExit("unknown evidence identity kind")
    if type(value) is not dict or set(value) != keys or not HEX40.fullmatch(value.get("archive_commit", "")):
        raise SystemExit(f"{kind} evidence identity schema mismatch")
    directory = Path(value["directory"])
    if directory.is_absolute() or ".." in directory.parts or not directory.parts:
        raise SystemExit(f"{kind} evidence directory rejected")
    for key in keys - {"archive_commit", "directory", "finalization_head",
                       "verified_core_commit", "checkpoint_path", "checkpoint_bytes"}:
        if not valid_digest(value[key]):
            raise SystemExit(f"{kind} evidence digest invalid: {key}")
    for key in ("archive_commit", "finalization_head", "verified_core_commit"):
        commit = value[key]
        if (not HEX40.fullmatch(commit)
                or subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                                  cwd=ROOT, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL).returncode):
            raise SystemExit(f"{kind} evidence commit invalid: {key}")
    if kind == "prior":
        checkpoint = Path(value["checkpoint_path"])
        if (not checkpoint.is_absolute() or type(value["checkpoint_bytes"]) is not int
                or value["checkpoint_bytes"] <= 0):
            raise SystemExit("prior external checkpoint identity invalid")
    return value


class GitReader:
    def __init__(self, identity, kind):
        validate_identity(identity, kind)
        self.commit = identity["archive_commit"]
        self.directory = Path(identity["directory"])

    def read(self, name):
        if Path(name).name != name:
            raise SystemExit("evidence path escape")
        relative = str(self.directory / name)
        raw = subprocess.check_output(["git", "ls-tree", "-z", self.commit, "--", relative], cwd=ROOT)
        records = [record for record in raw.split(b"\0") if record]
        if len(records) != 1:
            raise SystemExit(f"evidence missing/ambiguous: {name}")
        metadata, observed = records[0].split(b"\t", 1)
        mode, kind, _object = metadata.split(b" ", 2)
        if mode not in {b"100644", b"100755"} or kind != b"blob" or observed.decode() != relative:
            raise SystemExit(f"evidence is not a regular Git blob: {name}")
        return subprocess.check_output(["git", "show", f"{self.commit}:{relative}"], cwd=ROOT)


def validate_prior_run(identity, key):
    validate_identity(identity, "prior")
    spec = PRIOR[key]
    if spec["label"] not in identity["directory"]:
        raise SystemExit("prior evidence directory does not bind label")
    reader = GitReader(identity, "prior")
    top_map = {
        "ready-to-retrieve.json": identity["completion_sha256"],
        "artifact-manifest.json": identity["manifest_sha256"],
        "result.json": identity["result_sha256"],
        "screen2m.metrics.json": identity["metrics_sha256"],
    }
    top = {name: reader.read(name) for name in top_map}
    for name, expected in top_map.items():
        if digest(top[name]) != expected:
            raise SystemExit(f"prior top digest mismatch: {name}")
    completion = json_object(top["ready-to-retrieve.json"], "prior completion", canonical=True)
    manifest = json_object(top["artifact-manifest.json"], "prior manifest", canonical=True)
    result = json_object(top["result.json"], "prior result", canonical=True)
    metric_receipt = json_object(top["screen2m.metrics.json"], "prior metric receipt", canonical=True)
    completion_keys = {
        "schema", "arm", "role", "label", "host", "hostname", "finalization_head",
        "verified_core_commit", "authorization_sha256", "request_sha256", "runtime_job_sha256",
        "source_sha256", "prepare_sha256", "artifact_manifest_sha256", "artifact_members",
        "archive_sha256", "archive_bytes", "checkpoint_path", "checkpoint_sha256",
        "checkpoint_bytes", "checkpoint_preserved", "checkpoint_prune_authorized",
        "cache_cleanup_authorized", "screen2m", "ready_to_retrieve",
    }
    if set(completion) != completion_keys or completion["schema"] != "track-a-completion-v1":
        raise SystemExit("prior completion schema mismatch")
    exact_completion = {
        "arm": spec["arm"], "role": spec["role"], "label": spec["label"],
        "host": spec["host"], "hostname": spec["hostname"], "source_sha256": SOURCE_SHA,
        "prepare_sha256": PREPARE_SHA, "checkpoint_preserved": True,
        "checkpoint_prune_authorized": False, "cache_cleanup_authorized": False,
        "ready_to_retrieve": True,
    }
    if any(completion.get(field) != expected for field, expected in exact_completion.items()):
        raise SystemExit("prior completion identity/state mismatch")
    external_completion = {
        field: identity[field] for field in (
            "finalization_head", "verified_core_commit", "authorization_sha256",
            "request_sha256", "runtime_job_sha256", "source_sha256", "prepare_sha256",
            "checkpoint_path", "checkpoint_sha256", "checkpoint_bytes",
        )
    }
    if any(completion.get(field) != expected for field, expected in external_completion.items()):
        raise SystemExit("prior completion differs from externally pinned identity")
    for field in ("authorization_sha256", "request_sha256", "runtime_job_sha256",
                  "artifact_manifest_sha256", "archive_sha256", "checkpoint_sha256"):
        if not valid_digest(completion[field]):
            raise SystemExit(f"prior completion digest invalid: {field}")
    if (not HEX40.fullmatch(completion["finalization_head"] or "")
            or not HEX40.fullmatch(completion["verified_core_commit"] or "")
            or type(completion["checkpoint_bytes"]) is not int or completion["checkpoint_bytes"] <= 0):
        raise SystemExit("prior completion commit/checkpoint identity invalid")
    if (set(manifest) != {"schema", "label", "members"}
            or manifest["schema"] != "track-a-artifact-manifest-v1"
            or manifest["label"] != spec["label"] or set(manifest["members"]) != PRIOR_ARTIFACTS
            or manifest["members"] != completion["artifact_members"]
            or digest(top["artifact-manifest.json"]) != completion["artifact_manifest_sha256"]):
        raise SystemExit("prior manifest mismatch")
    members = {}
    for name in sorted(PRIOR_ARTIFACTS):
        data = reader.read(name)
        members[name] = data
        if manifest["members"].get(name) != {"sha256": digest(data), "bytes": len(data)}:
            raise SystemExit(f"prior artifact mismatch: {name}")
    archive = members["evidence.tar.gz"]
    if digest(archive) != completion["archive_sha256"] or len(archive) != completion["archive_bytes"]:
        raise SystemExit("prior archive identity mismatch")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as handle:
        entries = handle.getmembers()
        if [entry.name for entry in entries] != sorted(PRIOR_ARCHIVE_MEMBERS) or any(not entry.isfile() for entry in entries):
            raise SystemExit("prior archive member schema mismatch")
        for entry in entries:
            if handle.extractfile(entry).read() != members[entry.name]:
                raise SystemExit(f"prior archive content mismatch: {entry.name}")
    result_keys = {
        "schema", "arm", "role", "label", "host", "hostname", "source_commit", "source_path",
        "source_sha256", "prepare_sha256", "launch_command", "seed", "world_size", "lnc",
        "max_train_seconds", "smoke", "checkpoint_path", "checkpoint_sha256", "checkpoint_bytes",
        "checkpoint_preserved", "checkpoint_prune_authorized", "cache_cleanup_authorized",
        "finite_scan", "screen2m", "screening_only", "keep_authorized", "adoption_authorized",
        "a1_decision", "ready_to_retrieve",
    }
    if set(result) != result_keys or result["schema"] != "track-a-result-v1":
        raise SystemExit("prior result schema mismatch")
    exact_result = {
        "arm": spec["arm"], "role": spec["role"], "label": spec["label"],
        "host": spec["host"], "hostname": spec["hostname"], "source_commit": SOURCE_COMMIT,
        "source_path": SOURCE_PATH, "source_sha256": SOURCE_SHA, "prepare_sha256": PREPARE_SHA,
        "launch_command": spec["launch"], "seed": 42, "world_size": 4, "lnc": 2,
        "max_train_seconds": 1800, "smoke": False, "checkpoint_sha256": completion["checkpoint_sha256"],
        "checkpoint_bytes": completion["checkpoint_bytes"], "checkpoint_preserved": True,
        "checkpoint_prune_authorized": False, "cache_cleanup_authorized": False,
        "screening_only": True, "keep_authorized": False, "adoption_authorized": False,
        "ready_to_retrieve": False,
    }
    if any(result.get(field) != expected for field, expected in exact_result.items()):
        raise SystemExit("prior result identity/state mismatch")
    expected_checkpoint = Path(spec["remote"]) / "out" / f"{spec['label']}-work" / "sandbox/out/final.pt"
    if (result["checkpoint_path"] != str(expected_checkpoint)
            or completion["checkpoint_path"] != str(expected_checkpoint)
            or identity["checkpoint_path"] != str(expected_checkpoint)
            or result["checkpoint_sha256"] != identity["checkpoint_sha256"]
            or result["checkpoint_bytes"] != identity["checkpoint_bytes"]):
        raise SystemExit("prior checkpoint path mismatch")
    finite = result["finite_scan"]
    if (type(finite) is not dict or finite.get("finite") is not True or finite.get("weights_only") is not True
            or type(finite.get("tensor_count")) is not int or finite["tensor_count"] <= 0
            or finite.get("checkpoint_sha256") != result["checkpoint_sha256"]
            or finite.get("checkpoint_bytes") != result["checkpoint_bytes"]):
        raise SystemExit("prior finite checkpoint receipt mismatch")
    validate_clean_lifecycle(finite.get("process_group"))
    if (set(metric_receipt) != {"schema", "argv", "metrics", "wall_seconds", "process_group"}
            or metric_receipt["schema"] != "track-a-screen2m-metrics-v1"
            or metric_receipt["argv"] != expected_eval_argv(identity["checkpoint_path"], SCREEN_TOKENS)
            or type(metric_receipt["wall_seconds"]) is not float or metric_receipt["wall_seconds"] <= 0
            or metric_receipt["metrics"] != result["screen2m"]):
        raise SystemExit("prior screen metric receipt mismatch")
    validate_clean_lifecycle(metric_receipt["process_group"])
    metrics = validate_metric(metric_receipt["metrics"], SCREEN_TOKENS, SCREEN_BYTES)
    if metrics != completion["screen2m"]:
        raise SystemExit("prior screen metric completion mismatch")
    training = json_object(members["training-process.json"], "prior training process", canonical=True)
    if (training.get("returncode") != 0 or training.get("timed_out") is not False
            or training.get("charged_within_budget") is not True or training.get("seed") != 42
            or training.get("world_size") != 4 or training.get("lnc") != 2):
        raise SystemExit("prior training process mismatch")
    validate_clean_lifecycle(training.get("process_group"))
    if spec["role"] == "control":
        if result["a1_decision"] is not None:
            raise SystemExit("prior control unexpectedly has A1 decision")
    else:
        decision = result["a1_decision"]
        if (type(decision) is not dict or decision.get("decision") != "NOMINATE_OTHER_HOST_FULL20M_REPLICATE"
                or decision.get("screening_only") is not True or decision.get("keep_authorized") is not False
                or decision.get("adoption_authorized") is not False
                or decision.get("other_host_full20m_replicate_required") is not True):
            raise SystemExit("prior A1 nomination receipt mismatch")
    return {
        "key": key, "label": spec["label"], "host": spec["host"],
        "checkpoint_path": result["checkpoint_path"], "checkpoint_sha256": result["checkpoint_sha256"],
        "checkpoint_bytes": result["checkpoint_bytes"], "screen2m": metrics,
        "archive_commit": identity["archive_commit"], "completion_sha256": identity["completion_sha256"],
        "manifest_sha256": identity["manifest_sha256"], "result_sha256": identity["result_sha256"],
        "metrics_sha256": identity["metrics_sha256"],
        "finalization_head": identity["finalization_head"],
        "verified_core_commit": identity["verified_core_commit"],
        "authorization_sha256": identity["authorization_sha256"],
        "request_sha256": identity["request_sha256"],
        "runtime_job_sha256": identity["runtime_job_sha256"],
        "source_sha256": identity["source_sha256"], "prepare_sha256": identity["prepare_sha256"],
        "a1_decision": result["a1_decision"],
    }


def validate_prior_bundle(value):
    if type(value) is not dict or set(value) != set(PRIOR):
        raise SystemExit("prior evidence bundle schema mismatch")
    runs = {key: validate_prior_run(value[key], key) for key in PRIOR}
    delta = screen_delta(runs["control-trn21"]["screen2m"]["val_bpb"],
                         runs["a1-trn21"]["screen2m"]["val_bpb"])
    if delta > Decimal("-0.001"):
        raise SystemExit("authenticated A1 same-host screen did not nominate")
    decision = runs["a1-trn21"]["a1_decision"]
    if (decimal_value(decision.get("control_screen2m_bpb"), "decision control")
            != decimal_value(runs["control-trn21"]["screen2m"]["val_bpb"], "control screen")
            or decimal_value(decision.get("candidate_screen2m_bpb"), "decision candidate")
            != decimal_value(runs["a1-trn21"]["screen2m"]["val_bpb"], "candidate screen")):
        raise SystemExit("A1 nomination is not bound to authenticated control/candidate metrics")
    return {"runs": runs, "screen_delta": str(delta), "nominated": True}


def validate_metric_receipt(raw, name, label, host, hostname, checkpoint, tokens, nbytes):
    receipt = json_object(raw, name, canonical=True)
    keys = {"schema", "label", "checkpoint_path", "checkpoint_sha256", "checkpoint_bytes",
            "host", "hostname", "source_sha256", "prepare_sha256", "argv", "metrics",
            "wall_seconds", "process_group"}
    if (set(receipt) != keys or receipt["schema"] != "track-a-confirmation-metric-v1"
            or receipt["label"] != label or receipt["checkpoint_path"] != checkpoint["checkpoint_path"]
            or receipt["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]
            or receipt["checkpoint_bytes"] != checkpoint["checkpoint_bytes"]
            or receipt["host"] != host or receipt["hostname"] != hostname
            or receipt["source_sha256"] != SOURCE_SHA or receipt["prepare_sha256"] != PREPARE_SHA
            or receipt["argv"] != expected_eval_argv(checkpoint["checkpoint_path"], tokens)
            or type(receipt["wall_seconds"]) is not float or receipt["wall_seconds"] <= 0):
        raise SystemExit(f"{name} receipt identity mismatch")
    validate_clean_lifecycle(receipt["process_group"])
    validate_metric(receipt["metrics"], tokens, nbytes)
    return receipt


def validate_trn21_confirmation(identity, prior_bundle):
    validate_identity(identity, "confirmation")
    spec = ARMS["pair-trn21"]
    if spec["label"] not in identity["directory"]:
        raise SystemExit("trn21 confirmation directory does not bind label")
    reader = GitReader(identity, "confirmation")
    top_map = {
        "ready-to-retrieve.json": identity["completion_sha256"],
        "artifact-manifest.json": identity["manifest_sha256"],
        "result.json": identity["result_sha256"],
        "control20m.metrics.json": identity["control20m_sha256"],
        "a1-20m.metrics.json": identity["a120m_sha256"],
    }
    top = {name: reader.read(name) for name in top_map}
    for name, expected in top_map.items():
        if digest(top[name]) != expected:
            raise SystemExit(f"trn21 confirmation digest mismatch: {name}")
    completion = json_object(top["ready-to-retrieve.json"], "trn21 confirmation completion", canonical=True)
    manifest = json_object(top["artifact-manifest.json"], "trn21 confirmation manifest", canonical=True)
    result = json_object(top["result.json"], "trn21 confirmation result", canonical=True)
    completion_keys = {
        "schema", "arm", "role", "label", "host", "hostname", "finalization_head",
        "verified_core_commit", "authorization_sha256", "request_sha256", "runtime_job_sha256",
        "source_sha256", "prepare_sha256", "artifact_manifest_sha256", "artifact_members",
        "archive_sha256", "archive_bytes", "input_evidence", "metrics", "decision",
        "checkpoint_inputs_preserved", "candidate_checkpoint", "ready_to_retrieve",
    }
    if set(completion) != completion_keys or completion["schema"] != "track-a-confirmation-completion-v1":
        raise SystemExit("trn21 confirmation completion schema mismatch")
    if (completion["arm"] != "pair-trn21" or completion["role"] != spec["role"]
            or completion["label"] != spec["label"] or completion["host"] != "trn21"
            or completion["hostname"] != spec["hostname"] or completion["source_sha256"] != SOURCE_SHA
            or completion["prepare_sha256"] != PREPARE_SHA or completion["checkpoint_inputs_preserved"] is not True
            or completion["candidate_checkpoint"] is not None or completion["ready_to_retrieve"] is not True):
        raise SystemExit("trn21 confirmation completion identity mismatch")
    external_completion = {
        field: identity[field] for field in (
            "finalization_head", "verified_core_commit", "authorization_sha256",
            "request_sha256", "runtime_job_sha256", "source_sha256", "prepare_sha256",
        )
    }
    if any(completion.get(field) != expected for field, expected in external_completion.items()):
        raise SystemExit("trn21 confirmation completion differs from externally pinned identity")
    for field in ("authorization_sha256", "request_sha256", "runtime_job_sha256",
                  "artifact_manifest_sha256", "archive_sha256"):
        if not valid_digest(completion[field]):
            raise SystemExit(f"trn21 confirmation completion digest invalid: {field}")
    expected_artifacts = OUTPUT_ARTIFACTS["pair-trn21"]
    if (set(manifest) != {"schema", "label", "members"}
            or manifest["schema"] != "track-a-confirmation-artifact-manifest-v1"
            or manifest["label"] != spec["label"] or set(manifest["members"]) != expected_artifacts
            or manifest["members"] != completion["artifact_members"]
            or digest(top["artifact-manifest.json"]) != completion["artifact_manifest_sha256"]):
        raise SystemExit("trn21 confirmation manifest mismatch")
    members = {}
    for name in sorted(expected_artifacts):
        data = reader.read(name)
        members[name] = data
        if manifest["members"].get(name) != {"sha256": digest(data), "bytes": len(data)}:
            raise SystemExit(f"trn21 confirmation artifact mismatch: {name}")
    archive = members["evidence.tar.gz"]
    if digest(archive) != completion["archive_sha256"] or len(archive) != completion["archive_bytes"]:
        raise SystemExit("trn21 confirmation archive identity mismatch")
    archive_members = expected_artifacts - {"evidence.tar.gz"}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as handle:
        entries = handle.getmembers()
        if [entry.name for entry in entries] != sorted(archive_members) or any(not entry.isfile() for entry in entries):
            raise SystemExit("trn21 confirmation archive schema mismatch")
        for entry in entries:
            if handle.extractfile(entry).read() != members[entry.name]:
                raise SystemExit(f"trn21 confirmation archive content mismatch: {entry.name}")
    result_keys = {"schema", "arm", "role", "label", "host", "hostname", "source_sha256",
                   "prepare_sha256", "input_evidence", "metrics", "decision",
                   "checkpoint_inputs_preserved", "candidate_checkpoint", "ready_to_retrieve"}
    if (set(result) != result_keys or result["schema"] != "track-a-confirmation-result-v1"
            or result["arm"] != "pair-trn21" or result["role"] != spec["role"]
            or result["label"] != spec["label"] or result["host"] != "trn21"
            or result["hostname"] != spec["hostname"] or result["source_sha256"] != SOURCE_SHA
            or result["prepare_sha256"] != PREPARE_SHA or result["checkpoint_inputs_preserved"] is not True
            or result["candidate_checkpoint"] is not None or result["ready_to_retrieve"] is not False):
        raise SystemExit("trn21 confirmation result mismatch")
    runs = prior_bundle["runs"]
    control_receipt = validate_metric_receipt(
        top["control20m.metrics.json"], "control20m", spec["label"], spec["host"],
        spec["hostname"], runs["control-trn21"],
        FULL_TOKENS, FULL_BYTES,
    )
    a1_receipt = validate_metric_receipt(
        top["a1-20m.metrics.json"], "a1-20m", spec["label"], spec["host"],
        spec["hostname"], runs["a1-trn21"],
        FULL_TOKENS, FULL_BYTES,
    )
    metrics = {"control20m": control_receipt["metrics"], "a1_20m": a1_receipt["metrics"]}
    delta = decimal_value(metrics["a1_20m"]["val_bpb"], "trn21 A1 20M") - decimal_value(
        metrics["control20m"]["val_bpb"], "trn21 control 20M"
    )
    expected_decision = {
        "schema": "track-a-trn21-pair-decision-v1",
        "delta_definition": "candidate_minus_same_host_control",
        "metric_population": "fresh20m_seq1024_tokens20971520_bytes83462186",
        "trn21_delta": str(delta), "strictly_negative": delta < 0,
        "verdict": "NOMINATE_TRN22_REPLICATE" if delta < 0 else "DISCARD",
        "adoption_authorized": False,
    }
    summary_keys = (
        "archive_commit", "completion_sha256", "manifest_sha256", "result_sha256",
        "metrics_sha256", "finalization_head", "verified_core_commit", "authorization_sha256",
        "request_sha256", "runtime_job_sha256", "source_sha256", "prepare_sha256",
        "checkpoint_path", "checkpoint_sha256", "checkpoint_bytes",
    )
    expected_prior = {key: {field: run[field] for field in summary_keys} for key, run in runs.items()}
    expected_inputs = {
        "schema": "track-a-confirmation-input-v1", "prior_runs": expected_prior,
        "screen_delta": prior_bundle["screen_delta"], "trn21_confirmation": None,
    }
    if (result["metrics"] != metrics or completion["metrics"] != metrics
            or result["decision"] != expected_decision or completion["decision"] != expected_decision
            or result["input_evidence"] != expected_inputs or completion["input_evidence"] != expected_inputs):
        raise SystemExit("trn21 confirmation metric/decision/input mismatch")
    if delta >= 0:
        raise SystemExit("trn21 confirmation did not nominate other-host replicate")
    return {
        "label": spec["label"], "archive_commit": identity["archive_commit"],
        "completion_sha256": identity["completion_sha256"], "manifest_sha256": identity["manifest_sha256"],
        "result_sha256": identity["result_sha256"], "control20m_sha256": identity["control20m_sha256"],
        "a120m_sha256": identity["a120m_sha256"], "metrics": metrics,
        "finalization_head": identity["finalization_head"],
        "verified_core_commit": identity["verified_core_commit"],
        "authorization_sha256": identity["authorization_sha256"],
        "request_sha256": identity["request_sha256"],
        "runtime_job_sha256": identity["runtime_job_sha256"],
        "source_sha256": identity["source_sha256"], "prepare_sha256": identity["prepare_sha256"],
        "trn21_delta": str(delta), "input_evidence": expected_inputs,
    }


def validate_authorization(auth, request, arm, core):
    spec = ARMS[arm]
    keys = {"schema", "state", "arm", "role", "host", "expected_hostname", "label",
            "source_commit", "source_path", "source_sha256", "prepare_sha256", "seed",
            "max_train_seconds", "world_size", "lnc", "smoke_authorized", "screen_tokens",
            "screen_bytes", "full_tokens", "full_bytes", "checkpoint_prune_authorized",
            "cache_cleanup_authorized", "verified_base_head", "verified_lifecycle_core",
            "verified_core_commit", "request_path", "request_sha256", "request_content",
            "prior_evidence", "trn21_confirmation_evidence"}
    if type(auth) is not dict or set(auth) != keys or auth["schema"] != "track-a-confirmation-authorization-v1":
        raise SystemExit("authorization schema mismatch")
    request_sha = validate_request(request, arm, core)
    exact = {
        "arm": arm, "role": spec["role"], "host": spec["host"],
        "expected_hostname": spec["hostname"], "label": spec["label"],
        "source_commit": SOURCE_COMMIT, "source_path": SOURCE_PATH, "source_sha256": SOURCE_SHA,
        "prepare_sha256": PREPARE_SHA, "seed": 42, "max_train_seconds": 1800,
        "world_size": 4, "lnc": 2, "smoke_authorized": False,
        "screen_tokens": SCREEN_TOKENS, "screen_bytes": SCREEN_BYTES,
        "full_tokens": FULL_TOKENS, "full_bytes": FULL_BYTES,
        "checkpoint_prune_authorized": False, "cache_cleanup_authorized": False,
        "verified_base_head": VERIFIED_BASE_HEAD, "verified_lifecycle_core": VERIFIED_LIFECYCLE_CORE,
        "verified_core_commit": core, "request_path": request["request_path"],
        "request_sha256": request_sha, "request_content": request,
    }
    for field, expected in exact.items():
        if auth.get(field) != expected:
            raise SystemExit(f"authorization mismatch: {field}")
    if auth["state"] != "human_approved_evidence_bound_pending_operator":
        raise SystemExit("authorization remains pending evidence/independent verification")
    prior = validate_prior_bundle(auth["prior_evidence"])
    confirmation = None
    if arm == "pair-trn21":
        if auth["trn21_confirmation_evidence"] is not None:
            raise SystemExit("pair arm cannot consume its own confirmation")
    else:
        confirmation = validate_trn21_confirmation(auth["trn21_confirmation_evidence"], prior)
    return {"request_sha256": request_sha, "prior": prior, "confirmation": confirmation}


def build_concrete(request, authorization_sha, request_sha, head):
    return {
        "schema": "fresh-start-request-v2", "arm": request["arm"], "host": request["host"],
        "expected_hostname": request["expected_hostname"], "remote": request["remote"],
        "expected_commit": head, "origin": request["origin"], "label": request["label"],
        "purpose": request["purpose"], "min_free_bytes": request["min_free_bytes"],
        "fingerprint_paths": sorted(request["source_sha256"]),
        "expected_sha256": request["source_sha256"], "environment": request["environment"],
        "command": request["command"], "runner_owns_manifest": False, "non_overwrite": True,
        "verified_core_commit": request["verified_core_commit"], "finalization_head": head,
        "request_path": request["request_path"], "request_sha256": request_sha,
        "authorization_path": request["authorization_path"], "authorization_sha256": authorization_sha,
    }


def validate_runtime(runtime, concrete, raw):
    additions = {"started_at", "hostname", "git_commit", "source_dirty", "preflight_free_bytes", "source_sha256"}
    if type(runtime) is not dict or set(runtime) != set(concrete) | additions:
        raise SystemExit("runtime job schema mismatch")
    for field, expected in concrete.items():
        if runtime[field] != expected:
            raise SystemExit(f"runtime differs from concrete request: {field}")
    if (runtime["hostname"] != concrete["expected_hostname"]
            or runtime["git_commit"] != concrete["finalization_head"]
            or runtime["source_dirty"] is not False
            or disk_disposition(runtime["preflight_free_bytes"]) != "ALLOWED"):
        raise SystemExit("runtime observation mismatch")
    if runtime["source_sha256"] != concrete["expected_sha256"]:
        raise SystemExit("runtime source hash map mismatch")
    return digest(raw)
