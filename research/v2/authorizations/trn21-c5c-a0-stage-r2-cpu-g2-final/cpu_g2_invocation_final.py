"""Exact trn21 CPU-only G2 enablement candidate; no authority exists yet.

The final Verifier PASS and Supervisor one-attempt assignment must be real
fixed-path files whose contents bind these exact bytes. Only then may an
Operator import ``operator_once``. Direct execution is intentionally HOLD.
No native training, public eval, cleanup, retry or AWS action is implemented.
"""

import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REQUEST = HERE / "CPU_G2_REQUEST.FINAL_PENDING_REVIEW.json"
BINDING = HERE / "INVOCATION_BINDING.FINAL_PENDING_REVIEW.json"
REMOTE = HERE / "remote_cpu_g2_once.py"
FINAL_REVIEW = HERE / "INDEPENDENT_FINAL_CPU_G2_REVIEW.PASS.json"
ASSIGNMENT = HERE / "SUPERVISOR_CPU_ONLY_ASSIGNMENT.json"
MARKER = Path("/home/poojith-devan/.local/state/"
              "trainium-trn21-c5c-a0-stage-r2-cpu-g2-attempt-once.jsonl")
STATE = Path("/home/poojith-devan/Desktop/Trainium/research/v2/state.json")
HOLD = REPO / "research/v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-request"
HOLD_REQUEST = HOLD / "CPU_G2_REQUEST.HOLD.json"
HOLD_BUILDER = HOLD / "build_cpu_g2_request.py"
HOLD_REVIEW = REPO / "research/v2/verification/20260925-trn21-c5c-a0-g2-request-independent-review.md"
CHECKOUT_REQUEST = REPO / "research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/CHECKOUT_REQUEST.FINAL_PENDING_REVIEW.json"
CHECKOUT_RESULT_REVIEW = REPO / "research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-result/INDEPENDENT_CHECKOUT_RESULT_REVIEW.PASS.json"
TERMINAL = REPO / "research/v2/authorizations/trn21-c5c-a0-stage-r2/operator-transfer-only-terminal-receipt.json"
HOLD_REQUEST_SHA256 = "20bb8c7349e7d98a2ff6dd05ff7570326c8e25da9b44871600ec321434bf1a78"
HOLD_BUILDER_SHA256 = "7ebeb249bc82021fc117711b61e4b779610ba54a632da0121ccc607ffb860dbb"
HOLD_REVIEW_SHA256 = "32c9a11c75515c22ba063ce2cd4ec364f120b8d4fae0c904701978781ac61aaf"
CHECKOUT_REQUEST_SHA256 = "be97460e0fa4cd20c00d29c2eefbe0923e511b9327998629f7962fee07c29010"
CHECKOUT_RESULT_REVIEW_SHA256 = "b3c82bd594ae312142925cc3888f1de532473854f246a17b4ac4251baf37b89c"
TERMINAL_SHA256 = "9ab90252e5e21812f5aad2af7fd0bfd1e71205e2f8f740f6f41faf6606fe6a2e"
PROBE_SOURCE_SHA256 = "58e2d76dbff0446f6db109253304628fa11e739c9c2a377fbfb6a7a830b8b9db"
SOURCE_COMMIT = "e65a3097a5b00f6178188a935bb2acb9a5042c6c"
SOURCE = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2"
ROOT = "/home/ubuntu/trainium-v2-g2-20260925-trn21-stage-r2"
LABEL = "fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2"
ENDPOINT = "ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def argv_sha(argv):
    return sha(json.dumps(argv, separators=(",", ":")).encode("utf-8"))


def unique_json(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise RuntimeError("duplicate JSON key: " + key)
        obj[key] = value
    return obj


def pinned(path, expected):
    data = path.read_bytes()
    if sha(data) != expected:
        raise RuntimeError("reviewed local file drift: " + str(path))
    return data


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_plan():
    """Rebuild frozen commands and SSH data offline; never contact a host."""
    for path, digest in ((HOLD_REQUEST, HOLD_REQUEST_SHA256),
                         (HOLD_BUILDER, HOLD_BUILDER_SHA256),
                         (HOLD_REVIEW, HOLD_REVIEW_SHA256),
                         (CHECKOUT_REQUEST, CHECKOUT_REQUEST_SHA256),
                         (CHECKOUT_RESULT_REVIEW, CHECKOUT_RESULT_REVIEW_SHA256),
                         (TERMINAL, TERMINAL_SHA256)):
        pinned(path, digest)
    hold = load_module(HOLD_BUILDER, "reviewed_cpu_g2_hold")
    old = hold.build_plan()
    request_bytes = REQUEST.read_bytes()
    request = json.loads(request_bytes, object_pairs_hook=unique_json)
    binding_bytes = BINDING.read_bytes()
    binding = json.loads(binding_bytes, object_pairs_hook=unique_json)
    checkout_bytes = CHECKOUT_REQUEST.read_bytes()
    checkout = json.loads(checkout_bytes, object_pairs_hook=unique_json)
    review = json.loads(CHECKOUT_RESULT_REVIEW.read_bytes(),
                        object_pairs_hook=unique_json)
    terminal = json.loads(TERMINAL.read_bytes(), object_pairs_hook=unique_json)
    remote_bytes = REMOTE.read_bytes()
    wrapper_sha = sha(Path(__file__).read_bytes())
    remote_sha = sha(remote_bytes)
    if (request.get("schema") != "trn21-c5c-a0-cpu-g2-final-request-v1" or
            request.get("status") !=
            "ENABLED_CPU_ONLY_INTENT_PENDING_INDEPENDENT_FINAL_REVIEW_AND_SUPERVISOR_ASSIGNMENT" or
            request.get("prepared_by_role") != "implementer" or
            request.get("cpu_g2_intent") is not True or
            request.get("one_attempt_only") is not True or
            request.get("dispatch_requires_actual_review_and_assignment") is not True or
            any(request.get(key) is not False for key in (
                "native_training_authorized", "evaluation_authorized",
                "submission_authorized", "aws_action_authorized",
                "cleanup_or_retry_authorized")) or
            request.get("host") != "trn21" or
            request.get("expected_hostname") != "ip-172-31-34-85" or
            request.get("source_checkout") != SOURCE or
            request.get("source_commit") != SOURCE_COMMIT or
            request.get("cpu_root") != ROOT or
            request.get("cpu_label") != LABEL or
            request.get("g2_options") !=
                ["--steps", "3", "--nproc", "4", "--timeout", "900", "--keep"] or
            request.get("synthetic_cpu_only") is not True or
            request.get("attempt_marker_path") != str(MARKER) or
            request.get("marker_o_excl_o_nofollow_file_and_parent_fsync_before_ssh") is not True or
            request.get("fresh_full_live_preflight_required") is not True or
            request.get("same_ssh_immediate_boot_mount_stop_bridge_required") is not True or
            request.get("unconditional_separate_full_readonly_postflight_required") is not True or
            request.get("minimum_root_free_bytes") != 10737418240 or
            request.get("host_endpoint") != ENDPOINT or
            request.get("origin_hold_request_sha256") != HOLD_REQUEST_SHA256 or
            request.get("origin_hold_independent_review_sha256") != HOLD_REVIEW_SHA256 or
            request.get("checkout_request_sha256") != CHECKOUT_REQUEST_SHA256 or
            request.get("checkout_result_independent_review_sha256") !=
                CHECKOUT_RESULT_REVIEW_SHA256 or
            request.get("archived_probe_source_sha256") != PROBE_SOURCE_SHA256 or
            request.get("remote_adapter_sha256") != remote_sha or
            request.get("operator_wrapper_sha256") != wrapper_sha or
            request.get("final_independent_review_path") !=
                str(FINAL_REVIEW.relative_to(REPO)) or
            request.get("supervisor_assignment_path") !=
                str(ASSIGNMENT.relative_to(REPO)) or
            any(key in request for key in (
                "final_independent_review_sha256", "supervisor_assignment_sha256"))):
        raise RuntimeError("CPU-only final request intent, source or code drift")
    old_request = old["request"]
    for key in ("source_checkout_required_identity", "source_train_sha256",
                "source_prepare_sha256", "source_train_inode",
                "source_prepare_inode", "persistent_cpu_tools"):
        if request.get(key) != old_request[key]:
            raise RuntimeError("reviewed source/tool binding drift: " + key)
    if (review["verdict"] != "PASS_EXACT_SOURCE_CHECKOUT_RESULT_ONLY" or
            review["checkout_path"] != SOURCE or
            review["source_commit"] != SOURCE_COMMIT or
            review["cpu_g2_authorized_by_this_receipt"] is not False or
            checkout["checkout_path"] != SOURCE or
            checkout["source_commit"] != SOURCE_COMMIT or
            checkout["cpu_g2_authorized"] is not False):
        raise RuntimeError("accepted checkout is not the reviewed source-only result")
    for request_key, old_key in (
        ("exact_launch_argv_sha256", "launch_argv"),
        ("exact_wait_argv_sha256", "wait_argv"),
        ("exact_stop_on_failure_argv_sha256", "stop_on_failure_argv"),
        ("exact_quiesce_argv_sha256", "quiesce_argv"),
    ):
        if (request.get(request_key) != old_request[request_key] or
                old_request[request_key] != argv_sha(old[old_key])):
            raise RuntimeError("frozen CPU command drift: " + request_key)
    # The reviewed full probe is re-used unmodified, then the new remote
    # adapter adds exact current checkout, tool and scratch-root observations.
    audit_argv_old = shlex.split(terminal["final_postflight_probe_exact_ssh_command"])
    if (argv_sha(audit_argv_old) != "cc1997adb5b815a739a12cbb09319f3869e08f7d8318899fea9496bee40f66a8" or
            audit_argv_old[0:2] != ["ssh", "-T"] or
            audit_argv_old[-2] != ENDPOINT or
            "StrictHostKeyChecking=yes" not in audit_argv_old or
            "BatchMode=yes" not in audit_argv_old):
        raise RuntimeError("reviewed strict SSH transport drift")
    probe_words = shlex.split(audit_argv_old[-1])
    if probe_words[0:3] != ["exec", "python3", "-c"] or len(probe_words) != 4:
        raise RuntimeError("archived full probe command changed")
    probe_source = probe_words[3]
    if sha(probe_source.encode("utf-8")) != PROBE_SOURCE_SHA256:
        raise RuntimeError("archived full read-only probe source drift")
    commands = {key: old[key] for key in (
        "launch_argv", "wait_argv", "stop_on_failure_argv", "quiesce_argv")}
    payload = canonical({"request_text": request_bytes.decode("utf-8"),
                         "checkout_request_text": checkout_bytes.decode("utf-8"),
                         "archived_probe_source": probe_source,
                         "commands": commands}) + b"\n"
    remote_source = remote_bytes.decode("utf-8")
    base_ssh = audit_argv_old[:-1]
    ssh = {}
    for mode in ("audit", "run"):
        words = ["exec", "python3", "-c", remote_source, mode, sha(payload)]
        command = shlex.join(words)
        if shlex.split(command) != words:
            raise RuntimeError("remote CPU command cannot round-trip")
        ssh[mode] = base_ssh + [command]
    if (binding.get("schema") != "trn21-c5c-a0-cpu-g2-final-invocation-v1" or
            binding.get("status") != "ENABLED_CPU_ONLY_INTENT_PENDING_REVIEW_AND_ASSIGNMENT" or
            binding.get("prepared_by_role") != "implementer" or
            binding.get("cpu_g2_intent") is not True or
            binding.get("one_attempt_only") is not True or
            any(binding.get(key) is not False for key in (
                "native_training_authorized", "evaluation_authorized",
                "submission_authorized", "cleanup_or_retry_authorized")) or
            binding.get("request_path") != str(REQUEST.relative_to(REPO)) or
            binding.get("request_sha256") != sha(request_bytes) or
            binding.get("operator_wrapper_path") != str(Path(__file__).relative_to(REPO)) or
            binding.get("operator_wrapper_sha256") != wrapper_sha or
            binding.get("remote_adapter_path") != str(REMOTE.relative_to(REPO)) or
            binding.get("remote_adapter_sha256") != remote_sha or
            binding.get("full_audit_ssh_argv_sha256") != argv_sha(ssh["audit"]) or
            binding.get("one_cpu_run_ssh_argv_sha256") != argv_sha(ssh["run"]) or
            binding.get("stdin_payload_sha256") != sha(payload) or
            binding.get("stdin_payload_bytes") != len(payload) or
            binding.get("attempt_marker_path") != str(MARKER) or
            binding.get("marker_consumed_and_fsynced_before_any_ssh") is not True or
            binding.get("full_live_preflight_and_immediate_same_ssh_bridge_required") is not True or
            binding.get("separate_full_postflight_even_on_failure_required") is not True or
            binding.get("final_review_and_assignment_files_absent_at_implementation") is not True or
            binding.get("final_independent_review_path") !=
                str(FINAL_REVIEW.relative_to(REPO)) or
            binding.get("supervisor_assignment_path") !=
                str(ASSIGNMENT.relative_to(REPO))):
        raise RuntimeError("exact CPU-only invocation binding drift")
    remote = load_module(REMOTE, "cpu_g2_remote_validator")
    return {"request": request, "request_sha256": sha(request_bytes),
            "binding": binding, "binding_sha256": sha(binding_bytes),
            "checkout_request": checkout, "remote": remote,
            "audit_argv": ssh["audit"], "run_argv": ssh["run"],
            "stdin_payload": payload.decode("utf-8")}


def read_actual_file(path, *, controller_state=False):
    """Read the actual fixed-path review/assignment, no symlinks or substitution."""
    try:
        named = path.lstat()
        # The existing controller state is mode 0664 in the user's own group;
        # authority receipts remain stricter and may not be group-writable.
        unsafe_mode = stat.S_IMODE(named.st_mode) & (0o002 if controller_state else 0o022)
        if (not stat.S_ISREG(named.st_mode) or named.st_uid != os.getuid() or
                named.st_nlink != 1 or unsafe_mode or
                (controller_state and named.st_gid != os.getgid()) or
                not 0 < named.st_size <= 1048576):
            raise RuntimeError("unsafe actual authority file")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            before = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns):
                raise RuntimeError("authority file changed at nofollow open")
            data = bytearray()
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > 1048576:
                    raise RuntimeError("authority file grew beyond size bound")
            after = os.fstat(fd)
        finally:
            os.close(fd)
        final = path.lstat()
        if (len(data) != named.st_size or
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)):
            raise RuntimeError("authority file changed during read")
        return bytes(data)
    except FileNotFoundError as error:
        raise RuntimeError("DISPATCH HOLD: actual review or assignment file absent") from error


def expected_review(plan):
    request = plan["request"]
    binding = plan["binding"]
    return {
        "schema": "trn21-c5c-a0-cpu-g2-independent-final-review-v1",
        "verdict": "PASS_EXACT_CPU_ONLY_ENABLEMENT",
        "prepared_by_role": "verifier",
        "host": "trn21", "source_checkout": SOURCE,
        "source_commit": SOURCE_COMMIT, "cpu_label": LABEL, "cpu_root": ROOT,
        "attempt_marker_path": str(MARKER),
        "reviewed_request_sha256": plan["request_sha256"],
        "reviewed_binding_sha256": plan["binding_sha256"],
        "reviewed_operator_wrapper_sha256": request["operator_wrapper_sha256"],
        "reviewed_remote_adapter_sha256": request["remote_adapter_sha256"],
        "reviewed_origin_hold_request_sha256": HOLD_REQUEST_SHA256,
        "reviewed_origin_hold_independent_review_sha256": HOLD_REVIEW_SHA256,
        "reviewed_checkout_result_sha256": CHECKOUT_RESULT_REVIEW_SHA256,
        "reviewed_audit_ssh_argv_sha256": binding["full_audit_ssh_argv_sha256"],
        "reviewed_run_ssh_argv_sha256": binding["one_cpu_run_ssh_argv_sha256"],
        "reviewed_stdin_payload_sha256": binding["stdin_payload_sha256"],
        "one_attempt_only": True, "cpu_only": True,
        "native_training_authorized": False,
        "evaluation_authorized": False, "submission_authorized": False,
        "no_host_action_by_reviewer": True,
    }


def expected_assignment(plan, actual_review_sha256):
    request = plan["request"]
    binding = plan["binding"]
    return {
        "schema": "trn21-c5c-a0-cpu-g2-supervisor-assignment-v1",
        "prepared_by_role": "supervisor",
        "host": "trn21", "source_checkout": SOURCE,
        "source_commit": SOURCE_COMMIT, "cpu_label": LABEL, "cpu_root": ROOT,
        "attempt_marker_path": str(MARKER),
        "request_sha256": plan["request_sha256"],
        "binding_sha256": plan["binding_sha256"],
        "operator_wrapper_sha256": request["operator_wrapper_sha256"],
        "remote_adapter_sha256": request["remote_adapter_sha256"],
        "independent_final_review_path": str(FINAL_REVIEW.relative_to(REPO)),
        "independent_final_review_sha256": actual_review_sha256,
        "full_audit_ssh_argv_sha256": binding["full_audit_ssh_argv_sha256"],
        "one_cpu_run_ssh_argv_sha256": binding["one_cpu_run_ssh_argv_sha256"],
        "stdin_payload_sha256": binding["stdin_payload_sha256"],
        "one_attempt_only": True, "cpu_g2_authorized": True,
        "native_training_authorized": False,
        "evaluation_authorized": False, "submission_authorized": False,
        "fresh_local_and_host_preflight_required": True,
        "unconditional_separate_full_postflight_required": True,
    }


def authorize_actual_files(plan):
    """One-way actual-byte chain; a JSON claim or absent file never suffices."""
    review_raw = read_actual_file(FINAL_REVIEW)
    assignment_raw = read_actual_file(ASSIGNMENT)
    review = json.loads(review_raw, object_pairs_hook=unique_json)
    assignment = json.loads(assignment_raw, object_pairs_hook=unique_json)
    if review != expected_review(plan):
        raise RuntimeError("DISPATCH HOLD: actual independent review is not exact PASS")
    if assignment != expected_assignment(plan, sha(review_raw)):
        raise RuntimeError("DISPATCH HOLD: actual Supervisor assignment does not bind PASS")
    return {"review_sha256": sha(review_raw),
            "assignment_sha256": sha(assignment_raw)}


def local_breakers():
    stops = {REPO / "research/agentic/STOP", REPO / "research/v2/STOP",
             STATE.parent / "STOP", STATE.parent.parent / "agentic/STOP"}
    for path in stops:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise RuntimeError("local STOP present: " + str(path))
    first = read_actual_file(STATE, controller_state=True)
    state = json.loads(first, object_pairs_hook=unique_json)
    if (state.get("schema") != "trn-frontier-state-v2" or
            any(state["breakers"][name].get("tripped") is not False
                for name in ("global_correctness", "trn21"))):
        raise RuntimeError("local correctness or trn21 breaker is tripped/unknown")
    if read_actual_file(STATE, controller_state=True) != first:
        raise RuntimeError("local breaker state raced during check")
    for path in stops:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise RuntimeError("local STOP appeared during breaker check: " + str(path))
    return sha(first)


class Journal:
    """One fixed O_EXCL|O_NOFOLLOW marker, durably consumed before any SSH."""

    def __init__(self):
        named = MARKER.parent.lstat()
        if (not stat.S_ISDIR(named.st_mode) or named.st_uid != os.getuid() or
                stat.S_IMODE(named.st_mode) != 0o700):
            raise RuntimeError("CPU attempt marker parent is not private 0700")
        self.parent_fd = os.open(MARKER.parent, os.O_RDONLY | os.O_DIRECTORY |
                                 os.O_NOFOLLOW | os.O_CLOEXEC)
        self.fd = None
        try:
            opened = os.fstat(self.parent_fd)
            if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                raise RuntimeError("CPU marker parent changed")
            self.fd = os.open(MARKER.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                              os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                              dir_fd=self.parent_fd)
            os.fsync(self.fd)
            os.fsync(self.parent_fd)
        except BaseException:
            if self.fd is not None:
                os.close(self.fd)
            os.close(self.parent_fd)
            raise

    def emit(self, event, **fields):
        line = canonical({"utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(), "event": event, **fields}) + b"\n"
        while line:
            written = os.write(self.fd, line)
            if written <= 0:
                raise RuntimeError("CPU G2 journal write failed")
            line = line[written:]
        os.fsync(self.fd)

    def close(self):
        os.fsync(self.fd)
        os.close(self.fd)
        os.fsync(self.parent_fd)
        os.close(self.parent_fd)


def ssh_transport(argv, *, input_text, timeout_seconds):
    if argv[0:2] != ["ssh", "-T"]:
        raise RuntimeError("non-pinned CPU G2 SSH transport")
    return subprocess.run(argv, input=input_text, text=True, capture_output=True,
                          check=False, timeout=timeout_seconds)


def record_transport(journal, phase, argv, result):
    journal.emit("SSH_TRANSPORT", phase=phase, argv_sha256=argv_sha(argv),
                 argv=argv, returncode=result.returncode,
                 stdout=result.stdout, stderr=result.stderr)


def run_with_postflight(plan, journal, transport, authority):
    """One CPU attempt maximum; separate full read-only postflight always."""
    remote = plan["remote"]
    request = plan["request"]
    checkout = plan["checkout_request"]
    pre = None
    raw_run = None
    post = None
    failure = None
    host_contact_started = False
    try:
        host_contact_started = True
        pre_transport = transport(plan["audit_argv"],
                                  input_text=plan["stdin_payload"],
                                  timeout_seconds=240)
        record_transport(journal, "fresh_full_preflight", plan["audit_argv"],
                         pre_transport)
        if pre_transport.returncode != 0 or pre_transport.stderr.strip():
            raise RuntimeError("fresh full read-only preflight transport failed")
        pre_result = json.loads(pre_transport.stdout, object_pairs_hook=unique_json)
        if pre_result.get("status") != "READ_ONLY_G2_AUDIT":
            raise RuntimeError("fresh full read-only preflight returned wrong status")
        pre = remote.validate_snapshot(pre_result["snapshot"], request, checkout,
                                       phase="before")
        journal.emit("FRESH_FULL_CPU_PREFLIGHT_PASS", snapshot=pre)
        if authorize_actual_files(plan) != authority:
            raise RuntimeError("review or assignment drifted after preflight")
        local_breakers()  # Fresh local STOP/breaker gate directly before run SSH.
        run_transport = transport(plan["run_argv"],
                                  input_text=plan["stdin_payload"],
                                  timeout_seconds=2100)
        record_transport(journal, "one_synthetic_cpu_g2_attempt", plan["run_argv"],
                         run_transport)
        if run_transport.returncode != 0 or run_transport.stderr.strip():
            raise RuntimeError("one CPU G2 transport failed or emitted stderr")
        raw_run = json.loads(run_transport.stdout, object_pairs_hook=unique_json)
        if (raw_run.get("status") !=
                "CPU_G2_RAW_EVIDENCE_PENDING_INDEPENDENT_NUMERIC_REVIEW" or
                raw_run.get("native_training_authorized") is not False or
                raw_run.get("evaluation_authorized") is not False):
            raise RuntimeError("CPU G2 result not exact review-only success")
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)[:500]}
        journal.emit("CPU_G2_HOLD", failure=failure, preserve_partial=True,
                     same_label_retry_authorized=False)
    finally:
        if host_contact_started:
            try:
                post_transport = transport(plan["audit_argv"],
                                           input_text=plan["stdin_payload"],
                                           timeout_seconds=240)
                record_transport(journal, "unconditional_separate_full_postflight",
                                 plan["audit_argv"], post_transport)
                if post_transport.returncode != 0 or post_transport.stderr.strip():
                    raise RuntimeError("separate full postflight transport failed")
                post_result = json.loads(post_transport.stdout,
                                         object_pairs_hook=unique_json)
                if post_result.get("status") != "READ_ONLY_G2_AUDIT":
                    raise RuntimeError("postflight returned wrong read-only status")
                post = remote.validate_snapshot(post_result["snapshot"], request,
                                                checkout, phase="after")
                journal.emit("SEPARATE_FULL_POSTFLIGHT_RECORDED", snapshot=post)
            except BaseException as error:
                journal.emit("POSTFLIGHT_HOLD", error_type=type(error).__name__,
                             error=str(error)[:500], preserve_partial=True,
                             same_label_retry_authorized=False)
                failure = failure or {"type": type(error).__name__,
                                      "message": "postflight failed: " + str(error)[:400]}
    if failure or raw_run is None or post is None:
        journal.emit("TERMINAL_HOLD", failure=failure, preserve_partial=True,
                     same_label_retry_authorized=False,
                     native_training_authorized=False, evaluation_authorized=False)
        return {"status": "HOLD_PRESERVE_PARTIAL_NO_RETRY"}
    receipt = raw_run["cpu_receipt"]["stat"]
    if (post["scratch"][remote.CPU_RECEIPT].get("sha256") !=
            receipt["sha256"] or
            post["scratch"][remote.ROOT].get("exists") is not True):
        journal.emit("TERMINAL_HOLD", failure="CPU receipt missing or changed after run",
                     preserve_partial=True, same_label_retry_authorized=False)
        return {"status": "HOLD_PRESERVE_PARTIAL_NO_RETRY"}
    journal.emit("CPU_G2_RAW_EVIDENCE_PENDING_INDEPENDENT_NUMERIC_REVIEW",
                 cpu_receipt_sha256=receipt["sha256"],
                 native_training_authorized=False, evaluation_authorized=False)
    return {"status": "CPU_G2_RAW_EVIDENCE_PENDING_INDEPENDENT_NUMERIC_REVIEW",
            "cpu_receipt_sha256": receipt["sha256"]}


def operator_once(transport=ssh_transport):
    """Only a future separately reviewed and assigned CPU-only Operator call."""
    plan = build_plan()
    authority = authorize_actual_files(plan)  # Both files are absent now.
    local_breakers()
    journal = Journal()  # Fixed marker is consumed and fsynced before SSH.
    try:
        journal.emit("EXACT_CPU_ONLY_ASSIGNMENT_BOUND", authority=authority,
                     request_sha256=plan["request_sha256"],
                     binding_sha256=plan["binding_sha256"],
                     marker_path=str(MARKER),
                     marker_consumed_before_host_contact=True)
        return run_with_postflight(plan, journal, transport, authority)
    finally:
        journal.close()


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: no actual final review or Supervisor assignment exists")
