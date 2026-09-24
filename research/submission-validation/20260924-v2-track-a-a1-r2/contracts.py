#!/usr/bin/env python3
"""Immutable identities and validators for the infrastructure-only Track A A1 r2 retry."""

from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_COMMIT = "c51f6fdc5ca77793a1524148a563cafac0ab22b0"
SOURCE_PATH = "submissions/dense6x1024_public0992851/train.py"
SOURCE_SHA = "35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2"
PREPARE_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
RULES_SHA = "4e4d39d5ef9d4bd69bc2048821292ce74f2ea99b684a823f14f3b01de2b11316"
BRANCH = "experiment/20260924-v2-a1-lastlayer-ngram64d128-trn21-r2"
VERIFIED_R1_PARENT = "c873b603e767bd44177f64b12335aa85103d925b"
R1_FAILURE_COMMIT = "65d70f5956ad2c68c37560604c8f322cbbecf338"
R1_LABEL = "fs-20260924-v2-a1-lastlayer-ngram64d128-trn21-r1"
R1_EVIDENCE_DIRECTORY = (
    "research/submission-validation/20260924-v2-track-a-35c9-controls-a1/"
    f"operator-evidence/{R1_LABEL}"
)
RETRY_ARM = "a1-trn21-r2"
MIN_FREE_BYTES = 13 * 1024**3 // 2
HARD_ABORT_BYTES = 5 * 1024**3
SCREEN_TOKENS = 2_097_152
SCREEN_BYTES = 8_418_678
NOMINATE_DELTA = Decimal("0.001")
CC_FLAGS = "--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
BASE_TAIL = (
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
PREFIX = "NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py "
ARMS = {
    "control-trn21": {
        "role": "control", "host": "trn21", "hostname": "ip-172-31-34-85",
        "label": "fs-20260924-v2-35c9-lr015-control-trn21-r1", "launch": PREFIX + BASE_TAIL,
        "remote": "/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1",
    },
    "control-trn22": {
        "role": "control", "host": "trn22", "hostname": "ip-172-31-32-253",
        "label": "fs-20260924-v2-35c9-lr015-control-trn22-r1", "launch": PREFIX + BASE_TAIL,
        "remote": "/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1",
    },
    RETRY_ARM: {
        "role": "a1", "host": "trn21", "hostname": "ip-172-31-34-85",
        "label": "fs-20260924-v2-a1-lastlayer-ngram64d128-trn21-r2", "launch": PREFIX + A1_TAIL,
        "remote": "/home/ubuntu/trainium_freshstart_20260924_v2_a1_lastlayer_ngram64d128_trn21_r2",
    },
}
ENVIRONMENT = {
    "HOME": "/home/ubuntu", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    "LOGNAME": "ubuntu", "NEURON_CC_FLAGS": CC_FLAGS,
    "NEURON_COMPETITION_R1_SEED": "42", "NEURON_LOGICAL_NC_CONFIG": "2",
    "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv",
    "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
    "PYTHONUNBUFFERED": "1", "TMPDIR": "/tmp", "USER": "ubuntu",
}
METRIC_KEYS = {"val_bpb", "num_tokens", "num_bytes", "total_nats", "eval_seconds", "eval_timeout", "eval_over_timeout"}
ARTIFACTS = {
    "contract-check.log", "contract-check-process.json", "training.log", "training-process.json", "training-steps.jsonl",
    "checkpoint-finite.log", "checkpoint-finite.json", "screen2m.log", "screen2m.metrics.json",
    "cache-observations.json", "result.json", "evidence.tar.gz",
}
ARCHIVE_MEMBERS = ARTIFACTS - {"evidence.tar.gz"}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def canonical_json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def valid_digest(value):
    return type(value) is str and HEX64.fullmatch(value) is not None and value != "0" * 64


def validate_clean_lifecycle(value):
    required = {
        "pgid", "leader_returncode", "cleanup_required", "term_sent", "kill_sent",
        "quiescent", "reaped_descendants", "term_grace_seconds", "kill_grace_seconds",
    }
    if type(value) is not dict or set(value) != required:
        raise SystemExit("process-group lifecycle schema mismatch")
    if (type(value["pgid"]) is not int or value["pgid"] <= 0
            or value["leader_returncode"] != 0
            or value["cleanup_required"] is not False
            or value["term_sent"] is not False or value["kill_sent"] is not False
            or value["quiescent"] is not True or value["reaped_descendants"] != []
            or type(value["term_grace_seconds"]) is not float or value["term_grace_seconds"] <= 0
            or type(value["kill_grace_seconds"]) is not float or value["kill_grace_seconds"] <= 0):
        raise SystemExit("successful evidence did not have a clean process-group exit")
    return value


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


def train_argv(arm):
    words = shlex.split(ARMS[arm]["launch"])
    if words.pop(0) != "NEURON_LOGICAL_NC_CONFIG=2":
        raise SystemExit("launch environment prefix mismatch")
    return words


def validate_metric(value):
    if type(value) is not dict or set(value) != METRIC_KEYS:
        raise SystemExit("metric schema mismatch")
    for key in ("val_bpb", "total_nats", "eval_seconds"):
        if type(value[key]) is not float or value[key] <= 0 or not math.isfinite(value[key]):
            raise SystemExit(f"invalid metric {key}")
    if value["num_tokens"] != SCREEN_TOKENS or type(value["num_tokens"]) is not int:
        raise SystemExit("screen token count mismatch")
    if value["num_bytes"] != SCREEN_BYTES or type(value["num_bytes"]) is not int:
        raise SystemExit("screen byte count mismatch")
    if (type(value["eval_timeout"]) is not int or value["eval_timeout"] != 1200
            or value["eval_over_timeout"] is not False or value["eval_seconds"] > 1200.0):
        raise SystemExit("screen timeout mismatch")
    expected = value["total_nats"] / (value["num_bytes"] * math.log(2.0))
    if not math.isclose(value["val_bpb"], expected, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit("screen BPB arithmetic mismatch")
    return value


def parse_metric_output(text):
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
    return validate_metric(found[0])


def a1_decision(control_bpb, candidate_bpb):
    try:
        control = Decimal(str(control_bpb))
        candidate = Decimal(str(candidate_bpb))
    except (InvalidOperation, ValueError) as error:
        raise SystemExit("A1 BPB is not decimal") from error
    if not control.is_finite() or not candidate.is_finite() or control <= 0 or candidate <= 0:
        raise SystemExit("A1 BPB is nonfinite/nonpositive")
    improvement = control - candidate
    nominated = improvement >= NOMINATE_DELTA
    return {
        "control_screen2m_bpb": str(control), "candidate_screen2m_bpb": str(candidate),
        "improvement": str(improvement),
        "decision": "NOMINATE_OTHER_HOST_FULL20M_REPLICATE" if nominated else "DISCARD",
        "screening_only": True, "keep_authorized": False, "adoption_authorized": False,
        "other_host_full20m_replicate_required": nominated,
    }


def adoption_decision(control_trn21, candidate_trn21, control_trn22, candidate_trn22):
    """Downstream-only gate: delta is candidate minus matched control at full20M."""
    try:
        values = [Decimal(str(value)) for value in (
            control_trn21, candidate_trn21, control_trn22, candidate_trn22
        )]
    except (InvalidOperation, ValueError) as error:
        raise SystemExit("full20M BPB is not decimal") from error
    if any(not value.is_finite() or value <= 0 for value in values):
        raise SystemExit("full20M BPB is nonfinite/nonpositive")
    delta_trn21 = values[1] - values[0]
    delta_trn22 = values[3] - values[2]
    mean_delta = (delta_trn21 + delta_trn22) / Decimal(2)
    adopt = delta_trn21 < 0 and delta_trn22 < 0 and mean_delta <= Decimal("-0.001")
    return {
        "metric_population": "fresh20m_seq1024_tokens20971520_bytes83462186",
        "delta_definition": "candidate_bpb_minus_same_host_control_bpb",
        "trn21_delta": str(delta_trn21), "trn22_delta": str(delta_trn22),
        "mean_delta": str(mean_delta), "both_deltas_negative": delta_trn21 < 0 and delta_trn22 < 0,
        "mean_threshold": "-0.001", "adoption_authorized": adopt,
        "other_host_replicate_alone_confirms": False,
    }


def expected_command(arm):
    label = ARMS[arm]["label"]
    rel = str(HERE.relative_to(ROOT))
    return [
        "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "4200s",
        "/home/ubuntu/workspace/native_venv/bin/python", f"{rel}/track_a_job.py",
        "--arm", arm, "--authorization", f"{rel}/{arm}-authorization.json",
        "--request", f"{rel}/{arm}-request.json",
        "--runtime-job", f"research/run-logs/{label}.job.json",
    ]


def validate_request(request, arm, core):
    spec = ARMS[arm]
    required = {
        "schema", "arm", "host", "expected_hostname", "remote", "origin", "label", "purpose",
        "min_free_bytes", "environment", "command", "source_sha256", "request_path",
        "authorization_path", "verified_core_commit", "runner_owns_manifest", "non_overwrite",
    }
    if type(request) is not dict or set(request) != required or request["schema"] != "track-a-request-template-v1":
        raise SystemExit("request template schema mismatch")
    rel = str(HERE.relative_to(ROOT))
    exact = {
        "arm": arm, "host": spec["host"], "expected_hostname": spec["hostname"],
        "remote": spec["remote"], "label": spec["label"], "min_free_bytes": MIN_FREE_BYTES,
        "environment": ENVIRONMENT, "command": expected_command(arm),
        "request_path": f"{rel}/{arm}-request.json",
        "authorization_path": f"{rel}/{arm}-authorization.json", "verified_core_commit": core,
        "runner_owns_manifest": False, "non_overwrite": True,
    }
    for key, expected in exact.items():
        if request[key] != expected:
            raise SystemExit(f"request template mismatch: {key}")
    if type(request["origin"]) is not str or not request["origin"].startswith("https://github.com/"):
        raise SystemExit("request origin mismatch")
    if type(request["purpose"]) is not str or spec["label"] not in request["purpose"]:
        raise SystemExit("request purpose mismatch")
    for path, value in request["source_sha256"].items():
        if not isinstance(path, str) or not valid_digest(value):
            raise SystemExit("request source fingerprint malformed")
    mandatory = {SOURCE_PATH, "prepare.py", "README.md", f"{rel}/track_a_job.py", f"{rel}/contracts.py",
                 f"{rel}/artifact_io.py", f"{rel}/launch_adapter.py", f"{rel}/EXPERIMENT_PROPOSAL.json",
                 "research/fresh_start_launch.py", "research/fresh_start_host.py",
                 "research/fresh_start_guard.py", "research/fresh_start_backup.py",
                 "research/fresh_start_sync.py", "research/run_experiment.py"}
    mandatory |= {f"{rel}/stage_contract.py", f"{rel}/test_static.py", f"{rel}/RUNBOOK.md"}
    if set(request["source_sha256"]) != mandatory:
        raise SystemExit("request source fingerprint set incomplete")
    immutable = {SOURCE_PATH: SOURCE_SHA, "prepare.py": PREPARE_SHA, "README.md": RULES_SHA}
    if any(request["source_sha256"].get(path) != expected for path, expected in immutable.items()):
        raise SystemExit("request immutable source/rules fingerprint mismatch")
    return digest(canonical_json(request))


def validate_authorization(auth, request, arm, core):
    spec = ARMS[arm]
    required = {
        "schema", "state", "arm", "role", "host", "expected_hostname", "label",
        "source_commit", "source_path", "source_sha256", "prepare_sha256", "seed",
        "max_train_seconds", "world_size", "lnc", "smoke_authorized", "screen_tokens",
        "screen_bytes", "checkpoint_prune_authorized", "cache_cleanup_authorized",
        "verified_core_commit", "request_path", "request_sha256", "request_content",
        "control_evidence", "r1_failure_evidence",
    }
    if type(auth) is not dict or set(auth) != required or auth["schema"] != "track-a-authorization-v1":
        raise SystemExit("authorization schema mismatch")
    request_sha = validate_request(request, arm, core)
    exact = {
        "arm": arm, "role": spec["role"], "host": spec["host"],
        "expected_hostname": spec["hostname"], "label": spec["label"],
        "source_commit": SOURCE_COMMIT, "source_path": SOURCE_PATH, "source_sha256": SOURCE_SHA,
        "prepare_sha256": PREPARE_SHA, "seed": 42, "max_train_seconds": 1800,
        "world_size": 4, "lnc": 2, "smoke_authorized": False,
        "screen_tokens": SCREEN_TOKENS, "screen_bytes": SCREEN_BYTES,
        "checkpoint_prune_authorized": False, "cache_cleanup_authorized": False,
        "verified_core_commit": core, "request_path": request["request_path"],
        "request_sha256": request_sha, "request_content": request,
    }
    for key, expected in exact.items():
        if auth[key] != expected:
            raise SystemExit(f"authorization mismatch: {key}")
    if spec["role"] == "control":
        if auth["state"] != "human_approved_pending_operator" or auth["control_evidence"] is not None:
            raise SystemExit("control authorization state mismatch")
    else:
        validate_evidence_identity(auth["control_evidence"])
        validate_r1_failure_evidence(FailureGitReader(auth["r1_failure_evidence"]),
                                     auth["r1_failure_evidence"])
        if auth["state"] != "human_approved_control_and_r1_failure_bound_pending_operator":
            raise SystemExit("A1 retry is not independently verified and evidence-bound")
    return request_sha


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
    for key, expected in concrete.items():
        if runtime[key] != expected:
            raise SystemExit(f"runtime differs from concrete request: {key}")
    if (runtime["hostname"] != concrete["expected_hostname"]
            or runtime["git_commit"] != concrete["finalization_head"]
            or runtime["source_dirty"] is not False
            or disk_disposition(runtime["preflight_free_bytes"]) != "ALLOWED"):
        raise SystemExit("runtime observation mismatch")
    if not set(concrete["fingerprint_paths"]) <= set(runtime["source_sha256"]):
        raise SystemExit("runtime source fingerprints incomplete")
    for path, expected in concrete["expected_sha256"].items():
        if runtime["source_sha256"].get(path) != expected:
            raise SystemExit(f"runtime source hash mismatch: {path}")
    return digest(raw)


def validate_evidence_identity(value):
    required = {"archive_commit", "directory", "completion_sha256", "manifest_sha256", "result_sha256", "metrics_sha256"}
    if type(value) is not dict or set(value) != required or not HEX40.fullmatch(value["archive_commit"] or ""):
        raise SystemExit("control evidence identity schema mismatch")
    path = Path(value["directory"])
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit("control evidence directory rejected")
    for key in required - {"archive_commit", "directory"}:
        if not valid_digest(value[key]):
            raise SystemExit(f"control evidence digest invalid: {key}")


def validate_failure_identity(value):
    digest_keys = {
        "audit_sha256", "dispatch_sha256", "concrete_request_sha256", "runtime_job_sha256",
        "launch_sha256", "log_sha256", "postflight_sha256", "preflight_sha256", "process_sha256",
    }
    required = {"archive_commit", "directory"} | digest_keys
    if (type(value) is not dict or set(value) != required
            or value["archive_commit"] != R1_FAILURE_COMMIT
            or value["directory"] != R1_EVIDENCE_DIRECTORY):
        raise SystemExit("r1 failure evidence identity mismatch")
    if any(not valid_digest(value[key]) for key in digest_keys):
        raise SystemExit("r1 failure evidence digest invalid")


class GitReader:
    def __init__(self, identity):
        validate_evidence_identity(identity)
        self.commit = identity["archive_commit"]
        self.directory = Path(identity["directory"])

    def read(self, name):
        if Path(name).name != name:
            raise SystemExit("control evidence path escape")
        relative = str(self.directory / name)
        raw = subprocess.check_output(["git", "ls-tree", "-z", self.commit, "--", relative], cwd=ROOT)
        records = [record for record in raw.split(b"\0") if record]
        if len(records) != 1:
            raise SystemExit(f"control evidence missing/ambiguous: {name}")
        metadata, observed = records[0].split(b"\t", 1)
        mode, kind, _object = metadata.split(b" ", 2)
        if mode not in {b"100644", b"100755"} or kind != b"blob" or observed.decode() != relative:
            raise SystemExit(f"control evidence is not a regular blob: {name}")
        return subprocess.check_output(["git", "show", f"{self.commit}:{relative}"], cwd=ROOT)


class FailureGitReader:
    def __init__(self, identity):
        validate_failure_identity(identity)
        self.commit = identity["archive_commit"]
        self.directory = Path(identity["directory"])

    def names(self):
        prefix = str(self.directory) + "/"
        raw = subprocess.check_output(
            ["git", "ls-tree", "-r", "-z", "--name-only", self.commit, "--", str(self.directory)],
            cwd=ROOT,
        )
        names = []
        for record in raw.split(b"\0"):
            if record:
                path = record.decode()
                if not path.startswith(prefix):
                    raise SystemExit("r1 evidence path escaped directory")
                names.append(path[len(prefix):])
        return names

    def read(self, name):
        relative_name = Path(name)
        if relative_name.is_absolute() or ".." in relative_name.parts:
            raise SystemExit("r1 evidence path escape")
        relative = str(self.directory / relative_name)
        raw = subprocess.check_output(["git", "ls-tree", "-z", self.commit, "--", relative], cwd=ROOT)
        records = [record for record in raw.split(b"\0") if record]
        if len(records) != 1:
            raise SystemExit(f"r1 evidence missing/ambiguous: {name}")
        metadata, observed = records[0].split(b"\t", 1)
        mode, kind, _object = metadata.split(b" ", 2)
        if mode not in {b"100644", b"100755"} or kind != b"blob" or observed.decode() != relative:
            raise SystemExit(f"r1 evidence is not a regular blob: {name}")
        return subprocess.check_output(["git", "show", f"{self.commit}:{relative}"], cwd=ROOT)


def validate_r1_failure_evidence(reader, identity_value):
    """Authenticate the infrastructure-only r1 failure; never infer it from self-report alone."""
    validate_failure_identity(identity_value)
    label = R1_LABEL
    files = {
        "operator-failure-audit.json": "audit_sha256",
        f"dispatch/{label}.dispatch.json": "dispatch_sha256",
        f"dispatch/{label}.request.json": "concrete_request_sha256",
        f"runtime/run-logs/{label}.job.json": "runtime_job_sha256",
        f"runtime/run-logs/{label}.launch.json": "launch_sha256",
        f"runtime/run-logs/{label}.log": "log_sha256",
        f"runtime/run-logs/{label}.postflight.json": "postflight_sha256",
        f"runtime/run-logs/{label}.preflight.json": "preflight_sha256",
        f"runtime/run-logs/{label}.process.json": "process_sha256",
    }
    if set(reader.names()) != set(files):
        raise SystemExit("r1 evidence tree contains missing or unexpected artifacts")
    blobs = {name: reader.read(name) for name in files}
    for name, key in files.items():
        if digest(blobs[name]) != identity_value[key]:
            raise SystemExit(f"r1 evidence pinned digest mismatch: {name}")
    audit = json_object(blobs["operator-failure-audit.json"], "r1 failure audit")
    concrete = json_object(blobs[f"dispatch/{label}.request.json"], "r1 concrete request")
    runtime = json_object(blobs[f"runtime/run-logs/{label}.job.json"], "r1 runtime job")
    launch = json_object(blobs[f"runtime/run-logs/{label}.launch.json"], "r1 launch")
    dispatch = json_object(blobs[f"dispatch/{label}.dispatch.json"], "r1 dispatch")
    postflight = json_object(blobs[f"runtime/run-logs/{label}.postflight.json"], "r1 postflight")
    preflight = json_object(blobs[f"runtime/run-logs/{label}.preflight.json"], "r1 preflight")
    process = json_object(blobs[f"runtime/run-logs/{label}.process.json"], "r1 process")
    expected_audit_keys = {
        "schema", "label", "host", "hostname", "finalization_head", "finalization_tree",
        "source_sha256", "authorization_sha256", "template_request_sha256",
        "concrete_request_sha256", "control_evidence", "staging", "dispatch", "failure",
        "retrieval", "postflight", "audit_result",
    }
    if set(audit) != expected_audit_keys or audit["schema"] != "track-a-operator-failure-audit-v1":
        raise SystemExit("r1 failure audit schema mismatch")
    failure = audit["failure"]
    expected_failure = {
        "command_returncode": 1,
        "stage": "job identity check before training",
        "error": "git symbolic-ref --short HEAD returned exit status 128 because the staged checkout had detached HEAD",
        "log_sha256": identity_value["log_sha256"],
        "training_started": False,
        "evaluation_started": False,
        "output_directory_created": False,
        "checkpoint_created": False,
        "retry_performed": False,
    }
    if failure != expected_failure:
        raise SystemExit("r1 was not the pinned fail-before-training outcome")
    if (audit["label"] != label or audit["host"] != "trn21"
            or audit["hostname"] != ARMS[RETRY_ARM]["hostname"]
            or audit["finalization_head"] != VERIFIED_R1_PARENT
            or audit["source_sha256"] != SOURCE_SHA
            or audit["concrete_request_sha256"] != identity_value["concrete_request_sha256"]
            or audit["dispatch"].get("runtime_job_sha256") != identity_value["runtime_job_sha256"]
            or audit["staging"].get("checkout_was_detached") is not True
            or audit["retrieval"] != {"remote_local_hash_maps_match": True, "runtime_file_count": 6}
            or audit["postflight"].get("host_clean") is not True
            or audit["postflight"].get("device_holders") is not False
            or audit["postflight"].get("remaining_processes") != []
            or audit["audit_result"] != "FAILED_GATE_PRESERVED"):
        raise SystemExit("r1 failure audit identity/state mismatch")
    if (digest(canonical_json(concrete)) != identity_value["concrete_request_sha256"]
            or launch != runtime
            or dispatch.get("request") != concrete or dispatch.get("returncode") != 0
            or audit["dispatch"].get("launcher_returncode") != 0):
        raise SystemExit("r1 request/runtime/dispatch binding mismatch")
    if (concrete.get("label") != label or concrete.get("host") != "trn21"
            or concrete.get("expected_hostname") != ARMS[RETRY_ARM]["hostname"]
            or concrete.get("expected_commit") != VERIFIED_R1_PARENT
            or concrete.get("finalization_head") != VERIFIED_R1_PARENT
            or concrete.get("request_sha256") != audit["template_request_sha256"]
            or concrete.get("authorization_sha256") != audit["authorization_sha256"]
            or concrete.get("expected_sha256", {}).get(SOURCE_PATH) != SOURCE_SHA
            or runtime.get("git_commit") != VERIFIED_R1_PARENT
            or runtime.get("source_dirty") is not False
            or runtime.get("hostname") != ARMS[RETRY_ARM]["hostname"]):
        raise SystemExit("r1 runtime identity mismatch")
    log = blobs[f"runtime/run-logs/{label}.log"].decode("utf-8", "strict")
    if ("fatal: ref HEAD is not a symbolic ref" not in log
            or "returned non-zero exit status 128" not in log
            or "EXPERIMENT_POSTFLIGHT" not in log
            or "returncode\": 1" not in log):
        raise SystemExit("r1 log did not prove detached symbolic-ref failure")
    if (postflight.get("protocol_version") != 2 or postflight.get("label") != label
            or postflight.get("source_commit") != VERIFIED_R1_PARENT
            or postflight.get("command_returncode") != 1 or postflight.get("guard_error") is not None
            or postflight.get("interrupted_signals") != [] or postflight.get("failures") != []
            or postflight.get("host_clean") is not True
            or postflight.get("cleanup", {}).get("remaining") != []
            or postflight.get("health", {}).get("relevant_processes") != []
            or postflight.get("health", {}).get("neuron_process_count") != 0
            or postflight.get("artifacts_preserved_on_remote") is not True):
        raise SystemExit("r1 postflight was not clean and quiescent")
    if (preflight.get("hostname") != ARMS[RETRY_ARM]["hostname"]
            or preflight.get("disk", {}).get("free_bytes", 0) < MIN_FREE_BYTES):
        raise SystemExit("r1 preflight identity mismatch")
    old_package = "research/submission-validation/20260924-v2-track-a-35c9-controls-a1"
    old_command = [
        "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "4200s",
        "/home/ubuntu/workspace/native_venv/bin/python", f"{old_package}/track_a_job.py",
        "--arm", "a1-trn21", "--authorization", f"{old_package}/a1-trn21-authorization.json",
        "--request", f"{old_package}/a1-trn21-request.json", "--runtime-job",
        f"research/run-logs/{label}.job.json",
    ]
    if (process.get("label") != label or process.get("command") != old_command
            or any(type(process.get(key)) is not int or process[key] <= 0
                   for key in ("guard_pid", "child_pid", "child_session"))):
        raise SystemExit("r1 process identity mismatch")
    return {
        "archive_commit": identity_value["archive_commit"], "label": label,
        "classification": "infrastructure_fail_before_training_detached_head",
        "training_started": False, "evaluation_started": False, "checkpoint_created": False,
        "host_clean": True, "quiescent": True,
    }


def validate_control_evidence(reader, identity_value):
    validate_evidence_identity(identity_value)
    top = {
        "ready-to-retrieve.json": identity_value["completion_sha256"],
        "artifact-manifest.json": identity_value["manifest_sha256"],
        "result.json": identity_value["result_sha256"],
        "screen2m.metrics.json": identity_value["metrics_sha256"],
    }
    blobs = {name: reader.read(name) for name in top}
    for name, expected in top.items():
        if digest(blobs[name]) != expected:
            raise SystemExit(f"control evidence pinned digest mismatch: {name}")
    completion = json_object(blobs["ready-to-retrieve.json"], "control completion", canonical=True)
    manifest = json_object(blobs["artifact-manifest.json"], "control manifest", canonical=True)
    result = json_object(blobs["result.json"], "control result", canonical=True)
    receipt = json_object(blobs["screen2m.metrics.json"], "control metrics", canonical=True)
    completion_keys = {
        "schema", "arm", "role", "label", "host", "hostname", "finalization_head",
        "verified_core_commit", "authorization_sha256", "request_sha256", "runtime_job_sha256",
        "source_sha256", "prepare_sha256", "artifact_manifest_sha256", "artifact_members",
        "archive_sha256", "archive_bytes", "checkpoint_path", "checkpoint_sha256",
        "checkpoint_bytes", "checkpoint_preserved", "checkpoint_prune_authorized",
        "cache_cleanup_authorized", "screen2m", "ready_to_retrieve",
    }
    if set(completion) != completion_keys or completion["schema"] != "track-a-completion-v1":
        raise SystemExit("control completion schema mismatch")
    if (completion["role"] != "control" or completion["arm"] != "control-trn21"
            or completion["label"] != ARMS["control-trn21"]["label"]
            or completion["host"] != "trn21" or completion["hostname"] != ARMS["control-trn21"]["hostname"]
            or completion["source_sha256"] != SOURCE_SHA or completion["prepare_sha256"] != PREPARE_SHA
            or completion["checkpoint_preserved"] is not True
            or completion["checkpoint_prune_authorized"] is not False
            or completion["cache_cleanup_authorized"] is not False
            or completion["ready_to_retrieve"] is not True):
        raise SystemExit("control completion identity/state mismatch")
    for key in ("authorization_sha256", "request_sha256", "runtime_job_sha256", "artifact_manifest_sha256", "archive_sha256", "checkpoint_sha256"):
        if not valid_digest(completion[key]):
            raise SystemExit(f"control completion digest invalid: {key}")
    if type(completion["checkpoint_bytes"]) is not int or completion["checkpoint_bytes"] <= 0:
        raise SystemExit("control checkpoint size invalid")
    if (set(manifest) != {"schema", "label", "members"}
            or manifest["schema"] != "track-a-artifact-manifest-v1"
            or manifest["label"] != completion["label"] or set(manifest["members"]) != ARTIFACTS
            or manifest["members"] != completion["artifact_members"]
            or digest(blobs["artifact-manifest.json"]) != completion["artifact_manifest_sha256"]):
        raise SystemExit("control manifest mismatch")
    members = {}
    for name in sorted(ARTIFACTS):
        data = reader.read(name)
        members[name] = data
        expected = {"sha256": digest(data), "bytes": len(data)}
        if manifest["members"].get(name) != expected:
            raise SystemExit(f"control member mismatch: {name}")
    archive = members["evidence.tar.gz"]
    if digest(archive) != completion["archive_sha256"] or len(archive) != completion["archive_bytes"]:
        raise SystemExit("control archive mismatch")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        entries = tar.getmembers()
        if [entry.name for entry in entries] != sorted(ARCHIVE_MEMBERS) or any(not entry.isfile() for entry in entries):
            raise SystemExit("control archive schema mismatch")
        for entry in entries:
            if tar.extractfile(entry).read() != members[entry.name]:
                raise SystemExit(f"control archive content mismatch: {entry.name}")
    contract_process = json_object(members["contract-check-process.json"], "control contract process", canonical=True)
    if (set(contract_process) != {"schema", "argv", "returncode", "timed_out", "wall_seconds", "process_group"}
            or contract_process["schema"] != "track-a-contract-check-process-v1"
            or contract_process["returncode"] != 0 or contract_process["timed_out"] is not False
            or type(contract_process["wall_seconds"]) is not float or contract_process["wall_seconds"] <= 0):
        raise SystemExit("control contract process mismatch")
    validate_clean_lifecycle(contract_process["process_group"])
    training_process = json_object(members["training-process.json"], "control training process", canonical=True)
    training_keys = {
        "schema", "argv", "launch_command", "returncode", "timed_out", "wall_seconds",
        "startup_allowance_seconds", "charged_through_exit_seconds", "charged_within_budget",
        "observations", "seed", "world_size", "lnc", "process_group",
    }
    if (set(training_process) != training_keys or training_process["schema"] != "track-a-training-process-v1"
            or training_process["launch_command"] != ARMS["control-trn21"]["launch"]
            or training_process["returncode"] != 0 or training_process["timed_out"] is not False
            or training_process["charged_within_budget"] is not True
            or training_process["seed"] != 42 or training_process["world_size"] != 4
            or training_process["lnc"] != 2 or type(training_process["observations"]) is not list
            or len(training_process["observations"]) < 5):
        raise SystemExit("control training process mismatch")
    validate_clean_lifecycle(training_process["process_group"])
    result_keys = {
        "schema", "arm", "role", "label", "host", "hostname", "source_commit", "source_path",
        "source_sha256", "prepare_sha256", "launch_command", "seed", "world_size", "lnc",
        "max_train_seconds", "smoke", "checkpoint_path", "checkpoint_sha256", "checkpoint_bytes",
        "checkpoint_preserved", "checkpoint_prune_authorized", "cache_cleanup_authorized",
        "finite_scan", "screen2m", "screening_only", "keep_authorized", "adoption_authorized",
        "a1_decision", "ready_to_retrieve",
    }
    if set(result) != result_keys or result["schema"] != "track-a-result-v1":
        raise SystemExit("control result schema mismatch")
    if (result["arm"] != "control-trn21" or result["role"] != "control"
            or result["label"] != completion["label"] or result["host"] != "trn21"
            or result["hostname"] != ARMS["control-trn21"]["hostname"]
            or result["source_commit"] != SOURCE_COMMIT or result["source_path"] != SOURCE_PATH
            or result["source_sha256"] != SOURCE_SHA or result["prepare_sha256"] != PREPARE_SHA
            or result["launch_command"] != ARMS["control-trn21"]["launch"]
            or result["seed"] != 42 or result["world_size"] != 4 or result["lnc"] != 2
            or result["max_train_seconds"] != 1800 or result["smoke"] is not False
            or result["checkpoint_sha256"] != completion["checkpoint_sha256"]
            or result["checkpoint_bytes"] != completion["checkpoint_bytes"]
            or result["checkpoint_preserved"] is not True or result["checkpoint_prune_authorized"] is not False
            or result["cache_cleanup_authorized"] is not False or result["screening_only"] is not True
            or result["keep_authorized"] is not False or result["adoption_authorized"] is not False
            or result["a1_decision"] is not None or result["ready_to_retrieve"] is not False):
        raise SystemExit("control result identity/state mismatch")
    finite = result["finite_scan"]
    if (type(finite) is not dict or finite.get("finite") is not True or finite.get("weights_only") is not True
            or type(finite.get("tensor_count")) is not int or finite["tensor_count"] <= 0
            or finite.get("checkpoint_sha256") != result["checkpoint_sha256"]
            or finite.get("checkpoint_bytes") != result["checkpoint_bytes"]):
        raise SystemExit("control finite receipt mismatch")
    validate_clean_lifecycle(finite.get("process_group"))
    if (set(receipt) != {"schema", "argv", "metrics", "wall_seconds", "process_group"}
            or receipt["schema"] != "track-a-screen2m-metrics-v1"
            or receipt["metrics"] != result["screen2m"]):
        raise SystemExit("control metric receipt mismatch")
    validate_clean_lifecycle(receipt["process_group"])
    metrics = validate_metric(receipt["metrics"])
    if metrics != completion["screen2m"]:
        raise SystemExit("control completion metric mismatch")
    return {
        "label": result["label"], "host": result["host"], "archive_commit": identity_value["archive_commit"],
        "completion_sha256": identity_value["completion_sha256"], "manifest_sha256": identity_value["manifest_sha256"],
        "result_sha256": identity_value["result_sha256"], "metrics_sha256": identity_value["metrics_sha256"],
        "screen2m_val_bpb": metrics["val_bpb"], "checkpoint_sha256": result["checkpoint_sha256"],
    }
