"""Enabled exact checkout builder; final review and assignment are still absent.

Direct execution is HOLD. A later Operator may use the one-attempt flow only
after a separate Verifier PASS for these exact bytes and a matching Supervisor
checkout-only assignment. This module does no host work on import.
"""

import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TRANSFER = REPO / "research/v2/authorizations/trn21-c5c-a0-stage-r2"
EXTRACTION = REPO / "research/v2/authorizations/trn21-c5c-a0-stage-r2-extraction"
REQUEST = HERE / "CHECKOUT_REQUEST.FINAL_PENDING_REVIEW.json"
BINDING = HERE / "INVOCATION_BINDING.FINAL_PENDING_REVIEW.json"
REMOTE_ADAPTER = HERE / "remote_checkout_once.py"
EXTRACTION_CODE = EXTRACTION / "stage_r2_checkout.py"
TERMINAL = TRANSFER / "operator-transfer-only-terminal-receipt.json"
TRANSFER_REVIEW = TRANSFER / "INDEPENDENT_TRANSFER_RESULT_RECEIPT.md"
EXTRACTION_REVIEW = EXTRACTION / "INDEPENDENT_IMPLEMENTATION_REVIEW.md"
TERMINAL_SHA256 = "9ab90252e5e21812f5aad2af7fd0bfd1e71205e2f8f740f6f41faf6606fe6a2e"
TRANSFER_REVIEW_SHA256 = "bfafcb56ff449708dc31a1ebbe9023448018ea505e37a15847ae78bc3a3a2916"
EXTRACTION_REVIEW_SHA256 = "1a05c6dab4cbae6f49bd7e58760155d26562ba39df6a85524fe4a50289679df4"
EXTRACTION_CODE_SHA256 = "e66afb7a1e3368ed7854eaa679e2bad9a6b8529e3b793e0f8fff2c69c496785d"
PERSISTENCE_BUNDLE_SHA256 = "13db20bb2887eb6ff1c9feda31ea8453c9252ace18ff571021f866780e7cfe86"
TRANSFER_PREFIX = "research/v2/authorizations/trn21-c5c-a0-stage-r2/"
FLOOR = 10737418240
SOURCE_COMMIT = "e65a3097a5b00f6178188a935bb2acb9a5042c6c"
CHECKOUT_PATH = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2"
SOURCE_CAPSULE_SHA256 = "32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264"
TRAIN_SHA256 = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREPARE_SHA256 = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
ATTEMPT_MARKER_PATH = ("/home/poojith-devan/.local/state/"
                       "trainium-trn21-c5c-a0-stage-r2-checkout-attempt-once.jsonl")
FINAL_REVIEW_PATH = ("research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/"
                     "INDEPENDENT_FINAL_REQUEST_REVIEW.PASS.json")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def argv_sha256(argv):
    return sha256(json.dumps(argv, separators=(",", ":")).encode("utf-8"))


def pinned_bytes(path, expected):
    data = path.read_bytes()
    if sha256(data) != expected:
        raise RuntimeError("reviewed local evidence or code hash changed: " + str(path))
    return data


def referenced_json(pointer):
    path = pointer.get("path")
    if not isinstance(path, str) or not path.startswith(TRANSFER_PREFIX) or ".." in path:
        raise RuntimeError("transfer receipt pointer escaped the reviewed package")
    return json.loads(pinned_bytes(REPO / path, pointer["sha256"]))


def persistence_bundle(terminal):
    final_results = referenced_json(terminal["final_readonly_verify_results"])
    bundle = {}
    for artifact in ("capsule", "preflight", "contained"):
        group = terminal["artifacts"][artifact]
        if any(group[key] != 1 for key in (
                "write_attempts_by_this_operator", "first_verify_attempts_by_this_operator",
                "final_verify_attempts_by_this_operator")):
            raise RuntimeError("transfer artifact did not have exactly one attempt per phase")
        first_request = referenced_json(group["first_verify_request"])
        first_result = referenced_json(group["first_verify_result"])
        bundle[artifact] = {
            "write": first_request["write_receipt"],
            "verify": first_result["first_verify_receipt"],
            "final_verify": final_results["receipts"][artifact],
        }
    if sha256(canonical_json(bundle)) != PERSISTENCE_BUNDLE_SHA256:
        raise RuntimeError("complete nine-receipt bundle changed")
    # This is a local consistency check; the independent transfer-result
    # receipt separately authenticated raw SSH transports and their timing.
    source = pinned_bytes(EXTRACTION_CODE, EXTRACTION_CODE_SHA256)
    spec = importlib.util.spec_from_file_location("reviewed_stage_r2_checkout", EXTRACTION_CODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_persistence_bundle(bundle)
    return bundle, source


def build_plan():
    terminal = json.loads(pinned_bytes(TERMINAL, TERMINAL_SHA256))
    pinned_bytes(TRANSFER_REVIEW, TRANSFER_REVIEW_SHA256)
    pinned_bytes(EXTRACTION_REVIEW, EXTRACTION_REVIEW_SHA256)
    request_bytes = REQUEST.read_bytes()
    request = json.loads(request_bytes)
    pinned_bytes(EXTRACTION / "GATE_CONTRACT.json",
                 request["extraction_gate_contract_sha256"])
    bundle, extraction_source = persistence_bundle(terminal)
    remote_source = REMOTE_ADAPTER.read_bytes()
    if (request.get("schema") != "trn21-c5c-a0-checkout-only-request-v1" or
            request.get("checkout_authorized") is not True or
            request.get("operator_one_attempt_only") is not True or
            request.get("host") != "trn21" or
            request.get("checkout_path") != CHECKOUT_PATH or
            request.get("source_commit") != SOURCE_COMMIT or
            request.get("source_capsule_sha256") != SOURCE_CAPSULE_SHA256 or
            request.get("train_sha256") != TRAIN_SHA256 or
            request.get("prepare_sha256") != PREPARE_SHA256 or
            request["operator_terminal_receipt_sha256"] != TERMINAL_SHA256 or
            request["independent_transfer_result_sha256"] != TRANSFER_REVIEW_SHA256 or
            request["independent_extraction_implementation_review_sha256"] !=
            EXTRACTION_REVIEW_SHA256 or
            request["extraction_code_sha256"] != EXTRACTION_CODE_SHA256 or
            request["remote_adapter_sha256"] != sha256(remote_source) or
            request["operator_wrapper_sha256"] != sha256(Path(__file__).read_bytes()) or
            request["persistence_bundle_sha256"] != PERSISTENCE_BUNDLE_SHA256 or
            request.get("attempt_marker_path") != ATTEMPT_MARKER_PATH or
            request.get("attempt_marker_exclusive_fsync_before_any_host_contact") is not True or
            request.get("attempt_marker_parent_mode_required") != "0700" or
            request.get("independent_exact_request_review_path") != FINAL_REVIEW_PATH or
            request.get("independent_exact_request_review_sha256") is not None):
        raise RuntimeError("checkout request drifted from reviewed code or evidence")
    validate_full_snapshot(terminal["final_postflight_snapshot"], request,
                           before_stage=True, require_fresh=False)
    audit_argv = shlex.split(terminal["final_postflight_probe_exact_ssh_command"])
    if (len(audit_argv) < 12 or audit_argv[0:2] != ["ssh", "-T"] or
            audit_argv[-2] != "ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com" or
            "BatchMode=yes" not in audit_argv or "StrictHostKeyChecking=yes" not in audit_argv):
        raise RuntimeError("reviewed read-only host probe command changed")
    words = ["exec", "python3", "-c", remote_source.decode("utf-8"),
             sha256(request_bytes), EXTRACTION_CODE_SHA256, PERSISTENCE_BUNDLE_SHA256]
    remote_command = shlex.join(words)
    if shlex.split(remote_command) != words:
        raise RuntimeError("remote checkout command does not round-trip")
    stage_argv = audit_argv[:-1] + [remote_command]
    payload = canonical_json({
        "request_text": request_bytes.decode("utf-8"),
        "stage_source": extraction_source.decode("utf-8"),
        "persistence_bundle": bundle,
    }) + b"\n"
    binding_bytes = BINDING.read_bytes()
    binding = json.loads(binding_bytes)
    if (binding.get("dispatch_authorized") is not True or
            binding.get("one_attempt_only") is not True or
            binding.get("request_path") != "research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/CHECKOUT_REQUEST.FINAL_PENDING_REVIEW.json" or
            binding.get("operator_wrapper_path") != "research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/checkout_invocation_final.py" or
            binding["request_sha256"] != sha256(request_bytes) or
            binding["operator_wrapper_sha256"] != request["operator_wrapper_sha256"] or
            binding["remote_adapter_sha256"] != sha256(remote_source) or
            binding["extraction_code_sha256"] != EXTRACTION_CODE_SHA256 or
            binding["persistence_bundle_sha256"] != PERSISTENCE_BUNDLE_SHA256 or
            binding["audit_argv_sha256"] != argv_sha256(audit_argv) or
            binding["stage_argv_sha256"] != argv_sha256(stage_argv) or
            binding["stage_stdin_sha256"] != sha256(payload) or
            binding["stage_stdin_bytes"] != len(payload) or
            binding["host"] != request["host"] or
            binding["checkout_path"] != request["checkout_path"] or
            binding["source_commit"] != request["source_commit"] or
            binding.get("attempt_marker_path") != request["attempt_marker_path"] or
            binding.get("independent_exact_request_review_path") !=
            request["independent_exact_request_review_path"] or
            binding.get("independent_final_review_sha256_supplied_by_later_supervisor_assignment") is not True):
        raise RuntimeError("exact one-attempt argv or payload binding drifted")
    return {"request": request, "request_sha256": sha256(request_bytes),
            "binding": binding, "binding_sha256": sha256(binding_bytes),
            "audit_argv": audit_argv, "stage_argv": stage_argv,
            "stage_stdin": payload.decode("utf-8"), "terminal": terminal}


def validate_full_snapshot(snapshot, request, *, before_stage, require_fresh=True):
    """Interpret the archived full read-only probe format; never mutate host state."""
    host = request["expected_host_identity"]
    if (snapshot.get("schema") != "trn21-c5c-stage-r2-readonly-prewrite-v1" or
            snapshot.get("hostname") != host["hostname"] or
            snapshot.get("uid") != host["uid"] or
            snapshot.get("boot_id") != host["boot_id"] or
            snapshot.get("mount_namespace") != host["mount_namespace"] or
            snapshot.get("root_mount") != host["root_mount"] or
            snapshot.get("home_mount") != host["root_mount"] or
            snapshot.get("root_free_bytes", 0) < FLOOR):
        raise RuntimeError("fresh full host identity or free-space gate failed")
    when = datetime.datetime.fromisoformat(snapshot["utc"])
    now = datetime.datetime.now(datetime.timezone.utc)
    if when.tzinfo is None or (require_fresh and abs((now - when).total_seconds()) > 180):
        raise RuntimeError("read-only host snapshot is not fresh")
    for path, expected in host["ancestry"].items():
        observed = snapshot["ancestry"].get(path, {})
        if any(observed.get(key) != value for key, value in expected.items()):
            raise RuntimeError("unsafe or changed host ancestry: " + path)
    for path, expected in request["persistent_targets"].items():
        observed = snapshot["targets"].get(path, {})
        if any(observed.get(key) != value for key, value in expected.items()):
            raise RuntimeError("persistent artifact changed: " + path)
    for expected in request["protected_checkpoints"]:
        matching = [observed for observed in snapshot["checkpoints"]
                    if observed.get("path") == expected["path"]]
        if len(matching) != 1 or any(matching[0].get(key) != value
                                     for key, value in expected.items()):
            raise RuntimeError("protected checkpoint changed")
        holder = matching[0].get("fuser", {})
        if (holder.get("rc") != 1 or holder.get("pids") != [] or
                holder.get("stderr_empty") is not True):
            raise RuntimeError("protected checkpoint has an unknown holder")
    if any(snapshot["cache_root"].get(key) != value for key, value in
           request["warm_cache_root"].items()):
        raise RuntimeError("protected cache root changed")
    for name, expected in request["warm_cache_children"].items():
        observed = snapshot["caches"].get(name, {})
        if any(observed.get(key) != value for key, value in expected.items()):
            raise RuntimeError("protected warm cache changed: " + name)
    if (set(snapshot["stops"]) != set(request["required_stop_paths"]) or
            any(stop.get("exists") is not False for stop in snapshot["stops"].values()) or
            set(snapshot["science_and_stage_labels"]) !=
            {request["science_label_unconsumed"], request["stage_label"]} or
            any(label.get("rc") != 0 or label.get("matches") != [] or
                not label.get("stderr_empty") for label in
                snapshot["science_and_stage_labels"].values()) or
            snapshot["processes"].get("hits") != [] or
            snapshot["processes"].get("unreadable_cmdline_count") != 0 or
            any(snapshot[name].get("rc") != 1 or snapshot[name].get("pids") != [] or
                snapshot[name].get("stderr_empty") is not True for name in
                ("host_lock_fuser", "neuron0_fuser")) or
            snapshot["neuron_ls"].get("rc") != 0 or
            snapshot["neuron_ls"].get("non_diagnostic_pids") != [] or
            snapshot["neuron_ls"].get("json_valid") is not True):
        raise RuntimeError("STOP, occupied host, label or Neuron gate failed")
    for path in request["forbidden_paths_must_remain_absent"]:
        if snapshot["targets"].get(path, {}).get("exists") is not False:
            raise RuntimeError("future G2 or science-output path is present")
    checkout = snapshot["targets"].get(request["checkout_path"], {})
    if before_stage:
        if checkout.get("exists") is not False:
            raise RuntimeError("one-shot checkout target was already present")
    elif checkout.get("exists") is True:
        if (checkout.get("type") != "directory" or checkout.get("mode") != "0o700" or
                checkout.get("uid") != host["uid"] or
                checkout.get("device") != host["parent_device"]):
            raise RuntimeError("partial or complete checkout is not owner-private ext4")
    elif checkout.get("exists") is not False:
        raise RuntimeError("postflight cannot classify checkout target")
    return snapshot


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError("duplicate key in independent review receipt")
        value[key] = item
    return value


def verify_independent_request_review(plan, assignment):
    """Rehash the actual later Verifier PASS receipt before any host contact.

    The request fixes its path but deliberately does not embed its own future
    review hash. The later Supervisor assignment supplies that SHA-256, avoiding
    a request/review hash cycle while refusing a made-up digest or absent file.
    """
    relative = plan["request"]["independent_exact_request_review_path"]
    expected = assignment.get("independent_request_review_sha256")
    if (assignment.get("independent_request_review_path") != relative or
            not isinstance(expected, str) or
            re.fullmatch(r"[0-9a-f]{64}", expected) is None):
        raise RuntimeError("DISPATCH HOLD: assignment has no exact final review path and SHA-256")
    path = REPO / relative
    cursor = REPO
    for component in Path(relative).parts[:-1]:
        cursor = cursor / component
        info = os.lstat(cursor)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RuntimeError("DISPATCH HOLD: final review ancestry is not a directory")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or
                    before.st_nlink != 1 or before.st_mode & 0o022 or
                    not 0 < before.st_size <= 1_048_576):
                raise RuntimeError("DISPATCH HOLD: final review is not an intact local receipt")
            chunks = []
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if sum(map(len, chunks)) > 1_048_576:
                    raise RuntimeError("DISPATCH HOLD: final review exceeds size bound")
            after = os.fstat(fd)
        finally:
            os.close(fd)
    except OSError as error:
        raise RuntimeError("DISPATCH HOLD: independent final review receipt is absent or unsafe") from error
    data = b"".join(chunks)
    if (len(data) != before.st_size or
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or
            sha256(data) != expected):
        raise RuntimeError("DISPATCH HOLD: actual independent final review SHA-256 differs")
    try:
        review = json.loads(data, object_pairs_hook=_unique_json_object)
    except (UnicodeError, ValueError) as error:
        raise RuntimeError("DISPATCH HOLD: independent final review is not JSON") from error
    required = {
        "schema": "trn21-c5c-a0-checkout-independent-final-request-review-v1",
        "verdict": "PASS_EXACT_CHECKOUT_REQUEST_ONLY",
        "prepared_by_role": "verifier",
        "host": plan["request"]["host"],
        "checkout_path": plan["request"]["checkout_path"],
        "source_commit": plan["request"]["source_commit"],
        "source_capsule_sha256": plan["request"]["source_capsule_sha256"],
        "train_sha256": plan["request"]["train_sha256"],
        "prepare_sha256": plan["request"]["prepare_sha256"],
        "reviewed_request_sha256": plan["request_sha256"],
        "reviewed_binding_sha256": plan["binding_sha256"],
        "reviewed_operator_wrapper_sha256": plan["request"]["operator_wrapper_sha256"],
        "reviewed_remote_adapter_sha256": plan["request"]["remote_adapter_sha256"],
        "reviewed_extraction_code_sha256": plan["request"]["extraction_code_sha256"],
        "reviewed_persistence_bundle_sha256": plan["request"]["persistence_bundle_sha256"],
        "reviewed_audit_argv_sha256": plan["binding"]["audit_argv_sha256"],
        "reviewed_stage_argv_sha256": plan["binding"]["stage_argv_sha256"],
        "reviewed_stage_stdin_sha256": plan["binding"]["stage_stdin_sha256"],
        "attempt_marker_path": plan["request"]["attempt_marker_path"],
        "checkout_only": True,
        "cpu_g2_authorized": False,
        "native_training_authorized": False,
        "evaluation_authorized": False,
    }
    if not isinstance(review, dict) or any(review.get(key) != value
                                           for key, value in required.items()):
        raise RuntimeError("DISPATCH HOLD: actual review does not PASS these exact bytes")
    return {"path": relative, "sha256": expected}


class Journal:
    """Exclusive, fsynced append-only local evidence; no overwrite or cleanup."""

    def __init__(self, path):
        self.path = Path(path)
        parent = self.path.parent
        info = parent.stat(follow_symlinks=False)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or
                stat.S_IMODE(info.st_mode) != 0o700):
            raise RuntimeError("attempt-marker directory must be owner-private 0700")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        self.fd = os.open(self.path, flags, 0o600)
        try:
            self.parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            opened_parent = os.fstat(self.parent_fd)
            if (opened_parent.st_dev, opened_parent.st_ino) != (info.st_dev, info.st_ino):
                raise RuntimeError("attempt-marker parent changed during exclusive create")
            os.fsync(self.fd)
            os.fsync(self.parent_fd)
        except BaseException:
            if hasattr(self, "parent_fd"):
                os.close(self.parent_fd)
            os.close(self.fd)
            raise

    def emit(self, event, **details):
        payload = canonical_json({"utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(), "event": event, **details}) + b"\n"
        while payload:
            payload = payload[os.write(self.fd, payload):]
        os.fsync(self.fd)

    def close(self):
        os.fsync(self.fd)
        os.close(self.fd)
        os.fsync(self.parent_fd)
        os.close(self.parent_fd)


def _record_transport(journal, phase, argv, result):
    journal.emit("SSH_TRANSPORT", phase=phase, argv=argv,
                 argv_sha256=argv_sha256(argv), returncode=result.returncode,
                 stdout=result.stdout, stderr=result.stderr)


def pinned_ssh_transport(argv, *, input_text, timeout_seconds):
    """Future Operator transport: one bounded SSH process, no shell or retries."""
    if argv[0:2] != ["ssh", "-T"]:
        raise RuntimeError("non-pinned SSH transport command")
    return subprocess.run(argv, input=input_text, text=True, capture_output=True,
                          check=False, timeout=timeout_seconds)


def _run_with_unconditional_postflight(plan, journal, transport):
    """One stage attempt maximum; a distinct full read-only audit always follows."""
    request = plan["request"]
    stage_receipt = None
    failure = None
    postflight = None
    stage_attempted = False
    try:
        pre = transport(plan["audit_argv"], input_text=None, timeout_seconds=180)
        _record_transport(journal, "fresh_preflight", plan["audit_argv"], pre)
        if pre.returncode != 0 or pre.stderr.strip():
            raise RuntimeError("fresh full read-only preflight transport failed")
        validate_full_snapshot(json.loads(pre.stdout), request, before_stage=True)
        journal.emit("FRESH_FULL_PREFLIGHT_PASS")
        stage_attempted = True
        stage = transport(plan["stage_argv"], input_text=plan["stage_stdin"],
                          timeout_seconds=1800)
        _record_transport(journal, "one_checkout_attempt", plan["stage_argv"], stage)
        if stage.returncode != 0 or stage.stderr.strip():
            raise RuntimeError("exact checkout attempt failed or emitted stderr")
        stage_receipt = json.loads(stage.stdout)
        if (stage_receipt.get("status") !=
                "SOURCE_CHECKOUT_VERIFIED_FOR_INDEPENDENT_REVIEW_ONLY" or
                stage_receipt.get("cpu_g2_authorized") is not False or
                stage_receipt.get("native_training_authorized") is not False or
                stage_receipt.get("evaluation_authorized") is not False):
            raise RuntimeError("checkout receipt is not the exact review-only success")
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)[:500]}
        journal.emit("CHECKOUT_ATTEMPT_HOLD", stage_attempted=stage_attempted,
                     failure=failure, preserve_partial=True, retry_authorized=False)
    finally:
        # This is a NEW SSH call, not a check inside stage_fixed_checkout. It
        # also runs after a stage exception or nonzero exit; never clean/retry.
        try:
            post = transport(plan["audit_argv"], input_text=None,
                             timeout_seconds=180)
            _record_transport(journal, "unconditional_full_postflight",
                              plan["audit_argv"], post)
            if post.returncode != 0 or post.stderr.strip():
                raise RuntimeError("full read-only postflight transport failed")
            postflight = validate_full_snapshot(json.loads(post.stdout), request,
                                                before_stage=False)
            journal.emit("FULL_READ_ONLY_POSTFLIGHT_RECORDED",
                         checkout_target=postflight["targets"].get(request["checkout_path"]))
        except BaseException as error:
            journal.emit("FULL_POSTFLIGHT_HOLD", error_type=type(error).__name__,
                         error=str(error)[:500], preserve_partial=True,
                         retry_authorized=False)
            failure = failure or {"type": type(error).__name__,
                                  "message": "full postflight failed: " + str(error)[:400]}
    if failure or stage_receipt is None or postflight is None:
        journal.emit("TERMINAL_HOLD", failure=failure, stage_attempted=stage_attempted,
                     preserve_partial=True, retry_authorized=False)
        return {"status": "HOLD_PRESERVE_PARTIAL_NO_RETRY", "stage_attempted": stage_attempted}
    checkout = postflight["targets"][request["checkout_path"]]
    if checkout.get("exists") is not True:
        journal.emit("TERMINAL_HOLD", failure="success receipt but checkout absent",
                     preserve_partial=True, retry_authorized=False)
        return {"status": "HOLD_PRESERVE_PARTIAL_NO_RETRY", "stage_attempted": True}
    journal.emit("STAGE_EVIDENCE_READY_FOR_INDEPENDENT_REVIEW_ONLY",
                 stage_attempted=True, cpu_g2_authorized=False,
                 native_training_authorized=False, evaluation_authorized=False)
    return {"status": "STAGE_EVIDENCE_READY_FOR_INDEPENDENT_REVIEW_ONLY",
            "stage_attempted": True}


def operator_once(assignment, journal_path, transport=pinned_ssh_transport):
    """Future Operator API; one fixed fsynced marker is consumed before SSH."""
    plan = build_plan()
    request = plan["request"]
    if (request.get("checkout_authorized") is not True or
            plan["binding"].get("dispatch_authorized") is not True):
        raise RuntimeError("DISPATCH HOLD: exact checkout request is disabled")
    if (not isinstance(assignment, dict) or
            assignment.get("schema") != "trn21-c5c-a0-checkout-only-assignment-v1" or
            assignment.get("checkout_only_authorized") is not True or
            assignment.get("one_attempt_only") is not True or
            assignment.get("host") != request["host"] or
            assignment.get("checkout_path") != request["checkout_path"] or
            assignment.get("source_commit") != request["source_commit"] or
            assignment.get("source_capsule_sha256") != request["source_capsule_sha256"] or
            assignment.get("train_sha256") != request["train_sha256"] or
            assignment.get("prepare_sha256") != request["prepare_sha256"] or
            assignment.get("request_sha256") != plan["request_sha256"] or
            assignment.get("binding_sha256") != plan["binding_sha256"] or
            assignment.get("operator_wrapper_sha256") != request["operator_wrapper_sha256"] or
            assignment.get("remote_adapter_sha256") != request["remote_adapter_sha256"] or
            assignment.get("extraction_code_sha256") != request["extraction_code_sha256"] or
            assignment.get("stage_argv_sha256") != plan["binding"]["stage_argv_sha256"] or
            assignment.get("audit_argv_sha256") != plan["binding"]["audit_argv_sha256"] or
            assignment.get("stage_stdin_sha256") != plan["binding"]["stage_stdin_sha256"] or
            assignment.get("attempt_marker_path") != request["attempt_marker_path"] or
            assignment.get("independent_request_review_path") !=
            request["independent_exact_request_review_path"] or
            assignment.get("cpu_g2_authorized") is not False or
            assignment.get("native_training_authorized") is not False or
            assignment.get("evaluation_authorized") is not False or
            assignment.get("fresh_local_breaker_and_stop_gate") is not True):
        raise RuntimeError("DISPATCH HOLD: no matching one-attempt Supervisor assignment")
    if str(journal_path) != request["attempt_marker_path"]:
        raise RuntimeError("DISPATCH HOLD: caller path differs from fixed assignment-bound marker")
    review = verify_independent_request_review(plan, assignment)
    journal = Journal(request["attempt_marker_path"])
    try:
        journal.emit("EXACT_CHECKOUT_ASSIGNMENT_BOUND", assignment=assignment,
                     request_sha256=plan["request_sha256"],
                     binding_sha256=plan["binding_sha256"],
                     independent_request_review=review,
                     marker_path=request["attempt_marker_path"],
                     marker_consumed_before_host_contact=True)
        return _run_with_unconditional_postflight(plan, journal, transport)
    finally:
        journal.close()


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: request and Supervisor checkout-only assignment are absent")
