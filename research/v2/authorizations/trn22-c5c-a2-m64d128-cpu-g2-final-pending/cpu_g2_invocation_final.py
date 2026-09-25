"""Exact A2 CPU-only G2 final-pending Operator wrapper; no authority exists.

The fixed-path independent final review and Supervisor assignment are absent.
Direct execution exits HOLD. Only a future distinct Operator may call
``operator_once`` after the actual reviewed byte-chain and local breakers pass.
No native training, evaluation, cleanup, retry or AWS action is implemented.
"""

from __future__ import annotations

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
GUARD = HERE / "scratch_guard.py"
FINAL_REVIEW = HERE / "INDEPENDENT_FINAL_CPU_G2_REVIEW.PASS.json"
ASSIGNMENT = HERE / "SUPERVISOR_CPU_ONLY_ASSIGNMENT.json"
MARKER = Path("/home/poojith-devan/.local/state/"
              "trainium-a2-c5c-m64d128-trn22-r4-cpu-g2-attempt-once.jsonl")
STATE = Path("/home/poojith-devan/Desktop/Trainium/research/v2/state.json")
R3 = REPO / "research/v2/g2-probes/fs-20260925-g2-a2-c5c-m64d128-trn22-r3-hold"
R3_REQUEST = R3 / "G2_REQUEST.HOLD.json"
R3_PLAN = R3 / "plan_g2.py"
R3_REVIEW = REPO / "research/v2/verification/20260925-a2-c5c-m64d128-trn22-r3-disabled-g2-independent-static-PASS.md"
PROBE = REPO / "research/v2/observations/20260925-dual-host-readonly-operator/remote_snapshot.py"
SOURCE = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2"
ROOT = "/home/ubuntu/trainium-v2-g2-a2-c5c-m64d128-20260925-trn22-r4"
LABEL = "fs-20260925-g2-a2-c5c-m64d128-trn22-r4"
ENDPOINT = "ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com"
R3_REQUEST_SHA = "1fe26626e5c5a0880637bb704818e0733e2a4d3aa94cb6ea1c502a9583a0dbd8"
R3_PLAN_SHA = "425133400f38ce4d6fa2f986b7e6357392365aace7e79d482f02812dca7498ae"
R3_REVIEW_SHA = "ed86b23dbc8f70122d56e7afa4f2adf28434c3336a12fb2a2e923bfbccdda68b"
PROBE_SHA = "4a8b8ed413240c229ce943dd9bd8d6781bbddf7c348aa76b5b496f5c5d14f70a"
GUARD_SHA = "d0f4a5de111f9d64bc66e784128f2b2b54adc1ac1f693bc2e7f74f1e80f4e0b4"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("duplicate JSON key: " + key)
        result[key] = value
    return result


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def pin(path: Path, digest: str) -> bytes:
    require(path.is_file() and not path.is_symlink(), "pinned file missing or symlinked")
    raw = path.read_bytes()
    require(sha(raw) == digest, "pinned bytes changed: " + str(path))
    return raw


def _r3_plan():
    pin(R3_REQUEST, R3_REQUEST_SHA)
    pin(R3_PLAN, R3_PLAN_SHA)
    pin(R3_REVIEW, R3_REVIEW_SHA)
    spec = importlib.util.spec_from_file_location("reviewed_a2_r3_plan", R3_PLAN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.build_plan()
    require(result["dispatch_authorized"] is False, "disabled A2 basis changed")
    return result


def _build_core() -> dict:
    """Construct exact command data offline; never connect or dispatch."""
    r3 = _r3_plan()
    raw_request = REQUEST.read_bytes()
    request = json.loads(raw_request, object_pairs_hook=unique_json)
    raw_probe = pin(PROBE, PROBE_SHA)
    raw_guard = pin(GUARD, GUARD_SHA)
    raw_remote = REMOTE.read_bytes()
    wrapper_sha = sha(Path(__file__).read_bytes())
    remote_sha = sha(raw_remote)
    authority = request["authority"]
    require(request["schema"] == "trn22-c5c-a2-m64d128-cpu-g2-final-pending-v1" and
            request["status"] ==
                "IMPLEMENTER_FINAL_PENDING_INDEPENDENT_REVIEW_FRESH_BINDING_AND_SUPERVISOR_ASSIGNMENT_NO_DISPATCH" and
            request["prepared_by_role"] == "implementer" and
            all(authority[k] is False for k in (
                "cpu_g2_dispatch_authorized", "host_contact_authorized",
                "source_stage_authorized", "native_training_authorized",
                "evaluation_authorized", "aws_action_authorized",
                "cleanup_authorized", "submission_authorized")) and
            all(authority[k] is None for k in (
                "independent_final_review", "supervisor_one_attempt_assignment",
                "fresh_host_binding", "operator_result")),
            "final-pending request authority changed")
    lineage = request["lineage"]
    require(lineage["disabled_r3_request_sha256"] == R3_REQUEST_SHA and
            lineage["disabled_r3_plan_sha256"] == R3_PLAN_SHA and
            lineage["independent_static_pass_sha256"] == R3_REVIEW_SHA and
            lineage["snapshot_is_not_fresh_preflight"] is True,
            "reviewed A2 lineage or snapshot scope changed")
    host = request["host"]
    require(host["name"] == "trn22" and host["endpoint"] == ENDPOINT and
            host["expected_hostname"] == "ip-172-31-32-253" and
            host["source_checkout"] == SOURCE and
            host["source_commit"] == "e65a3097a5b00f6178188a935bb2acb9a5042c6c" and
            (host["source_checkout_device"], host["source_checkout_inode"],
             host["source_checkout_uid"], host["source_checkout_mode"]) ==
                (66305, 2432738, 1000, "0775") and
            (host["git_pointer_inode"], host["git_pointer_mode"],
             host["git_pointer_sha256"], host["linked_gitdir_inode"],
             host["linked_gitdir_mode"]) ==
                (2432740, "0664",
                 "dc53f6887bc5f9c2c1bbdbbd60911cdbeb076f7c57bafd796ee5b96f69ecbf1b",
                 2432736, "0775") and
            host["same_uid_or_privileged_race_not_excluded"] is True and
            host["require_fresh_same_session_nofollow_source_and_gitdir_bridge"] is True,
            "linked host/source identity changed")
    science, source, cap = request["science"], request["source"], request["capacity"]
    require(science["label"] == LABEL and science["scratch_root"] == ROOT and
            science["train_argv"] == r3["train_argv"] and
            science["expected_table_keys"] == [
                "ngram_embeds.b1.weight", "ngram_embeds.t1.weight",
                "ngram_embeds.b5.weight", "ngram_embeds.t5.weight"] and
            all(science[k] is True for k in (
                "cpu_only", "synthetic_data", "real_prepare_import",
                "trace_rmsprop", "keep_raw_workdir_and_checkpoint")) and
            (science["seed"], science["nproc"], science["steps"],
             science["micro_rows"], science["micro_tokens"],
             science["g2_timeout_seconds"]) == (42, 4, 2, 2, 256, 3600),
            "A2 exact-size CPU science changed")
    require(source == {"train_py_sha256":
                     "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586",
                     "prepare_py_sha256": "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb",
                     "g2_tool_sha256": "6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8",
                     "contained_tool_sha256": "0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d",
                     "queue_sha256": "1e9dcc4fd5f38e62829dc19460d2337662184ecfbd4d21d8a20f15993be9dd00"},
            "source/tool hashes changed")
    require(cap == {"minimum_fresh_root_free_bytes": 34359738368,
                    "minimum_fresh_root_free_inodes": 10000,
                    "scratch_peak_projection_bytes": 25769803776,
                    "operational_root_floor_bytes": 6979321856,
                    "projected_peak_memory_bytes": 85899345920,
                    "minimum_fresh_memory_available_bytes": 103079215104,
                    "projection_is_conservative_policy_not_measurement": True,
                    "fresh_values_required_in_both_audit_and_run_session": True,
                    "fresh_host_values": None}, "capacity policy or fresh binding changed")
    execution = request["execution"]
    require(all(execution[k] is True for k in (
                "one_run_per_host", "unique_label_required",
                "marker_o_excl_nofollow_fsync_before_any_ssh",
                "strict_host_key_ssh_required",
                "local_and_host_stop_breaker_lock_device_gate_required",
                "fresh_full_readonly_audit_before_run_required",
                "repeat_full_audit_and_source_bridge_in_run_session_required",
                "exact_scratch_tmp_creation_after_marker_required",
                "contained_launch_wait_failure_stop_quiesce_required",
                "unconditional_distinct_session_readonly_postflight_required",
                "preserve_partial_on_any_failure",
                "raw_four_rank_and_eight_rms_records_required",
                "independent_raw_per_table_attribution_required")) and
            execution["same_label_retry_authorized"] is False and
            execution["fixed_attempt_marker_path"] == str(MARKER),
            "one-attempt protocol changed")
    bindings = request["bindings"]
    require(bindings["final_independent_review_path"] ==
                str(FINAL_REVIEW.relative_to(REPO)) and
            bindings["supervisor_assignment_path"] ==
                str(ASSIGNMENT.relative_to(REPO)) and
            all(bindings[key] is None for key in (
                "operator_runner_sha256", "remote_runner_sha256",
                "scratch_guard_sha256", "commands_sha256",
                "ssh_audit_argv_sha256", "ssh_run_argv_sha256",
                "stdin_payload_sha256")) and
            all(value is None for value in request["result"].values()),
            "result or authority binding injected")

    env = dict(r3["environment"])
    env.update(TMPDIR=ROOT + "/tmp", TRN_EXP_LABEL=LABEL,
               NEURON_COMPETITION_R1_STEP_LOG=ROOT + "/steps.jsonl")
    py = "/home/ubuntu/workspace/native_venv/bin/python"
    clean_env = ["/usr/bin/env", "-i", *[f"{k}={env[k]}" for k in sorted(env)]]
    guard_code = raw_guard.decode("utf-8")
    common = [ROOT, ROOT + "/tmp", "34359738368", "10000",
              "25769803776", "6979321856"]
    g2_args = ["--train-py", SOURCE + "/train.py", "--prepare-dir", SOURCE,
               "--steps", "2", "--nproc", "4", "--micro-rows", "2",
               "--micro-tokens", "256", "--timeout", "3600",
               "--trace-rmsprop", "--keep", "--out", ROOT + "/g2.json",
               "--", *r3["train_argv"]]
    prepare = [*clean_env, py, "-B", "-c", guard_code, "prepare", *common, "--end"]
    g2_run = [*clean_env, py, "-B", "-c", guard_code, "run", *common,
              SOURCE + "/workflow/tools/preflight_dryrun.py", *g2_args]
    contained = [py, SOURCE + "/workflow/tools/contained.py"]
    process = ROOT + "/process"
    commands = {
        "prepare_scratch_after_marker": prepare,
        "launch": [*contained, "launch", "--label", LABEL, "--dir", process,
                   "--cwd", SOURCE, "--cgroup", "--", *g2_run],
        "wait": [*contained, "wait", "--label", LABEL, "--dir", process,
                 "--timeout", "4200"],
        "stop_if_needed": [*contained, "stop", "--label", LABEL,
                           "--dir", process, "--grace", "60"],
        "quiesce": [*contained, "quiesce", "--label", LABEL, "--dir", process,
                    "--files", process + "/launch.json", process + "/run.log",
                    ROOT + "/g2.json"],
    }
    payload = canonical({"request": request, "commands": commands,
                         "probe_source": raw_probe.decode("utf-8")})
    require(len(payload) <= 2_000_000, "remote payload over bounded size")
    ssh_prefix = ["ssh", "-T", "-i", "/home/poojith-devan/Desktop/Trainium/trn2.pem",
                  "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                  "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=20",
                  ENDPOINT]
    ssh = {}
    for mode in ("audit", "run"):
        words = ["exec", "python3", "-B", "-c", raw_remote.decode("utf-8"),
                 mode, sha(payload)]
        command = shlex.join(words)
        require(shlex.split(command) == words, "SSH remote command round-trip failed")
        ssh[mode] = [*ssh_prefix, command]
    return {"request": request, "request_sha256": sha(raw_request),
            "remote_sha256": remote_sha, "guard_sha256": sha(raw_guard),
            "wrapper_sha256": wrapper_sha, "probe_sha256": sha(raw_probe),
            "commands": commands, "commands_sha256": sha(canonical(commands)),
            "stdin_payload": payload, "stdin_payload_sha256": sha(payload),
            "audit_argv": ssh["audit"], "run_argv": ssh["run"],
            "audit_argv_sha256": sha(canonical(ssh["audit"])),
            "run_argv_sha256": sha(canonical(ssh["run"])),
            "dispatch_authorized": False}


def expected_binding(plan: dict) -> dict:
    return {"schema": "trn22-a2-cpu-g2-final-invocation-binding-v1",
            "status": "FINAL_PENDING_INDEPENDENT_REVIEW_AND_SUPERVISOR_ASSIGNMENT_NO_DISPATCH",
            "prepared_by_role": "implementer", "host": "trn22", "label": LABEL,
            "request_path": str(REQUEST.relative_to(REPO)),
            "request_sha256": plan["request_sha256"],
            "operator_runner_path": str(Path(__file__).relative_to(REPO)),
            "operator_runner_sha256": plan["wrapper_sha256"],
            "remote_runner_path": str(REMOTE.relative_to(REPO)),
            "remote_runner_sha256": plan["remote_sha256"],
            "scratch_guard_path": str(GUARD.relative_to(REPO)),
            "scratch_guard_sha256": plan["guard_sha256"],
            "commands_sha256": plan["commands_sha256"],
            "audit_ssh_argv_sha256": plan["audit_argv_sha256"],
            "run_ssh_argv_sha256": plan["run_argv_sha256"],
            "stdin_payload_sha256": plan["stdin_payload_sha256"],
            "fixed_attempt_marker": str(MARKER),
            "one_attempt_only": True, "cpu_only": True,
            "native_training_authorized": False, "evaluation_authorized": False,
            "host_contact_authorized_by_binding": False,
            "final_review_and_assignment_absent_at_implementation": True}


def build_plan() -> dict:
    """The exact binding must exist and equal independently reconstructed bytes."""
    plan = _build_core()
    raw = BINDING.read_bytes()
    require(json.loads(raw, object_pairs_hook=unique_json) == expected_binding(plan),
            "final CPU G2 invocation binding changed")
    plan["binding_sha256"] = sha(raw)
    return plan


def expected_review(plan: dict) -> dict:
    return {"schema": "trn22-a2-cpu-g2-independent-final-review-v1",
            "verdict": "PASS_EXACT_CPU_ONLY_FINAL_PACKAGE",
            "prepared_by_role": "verifier", "host": "trn22", "label": LABEL,
            "source_checkout": SOURCE, "scratch_root": ROOT,
            "request_sha256": plan["request_sha256"],
            "binding_sha256": plan["binding_sha256"],
            "operator_runner_sha256": plan["wrapper_sha256"],
            "remote_runner_sha256": plan["remote_sha256"],
            "scratch_guard_sha256": plan["guard_sha256"],
            "read_only_probe_sha256": plan["probe_sha256"],
            "commands_sha256": plan["commands_sha256"],
            "audit_ssh_argv_sha256": plan["audit_argv_sha256"],
            "run_ssh_argv_sha256": plan["run_argv_sha256"],
            "stdin_payload_sha256": plan["stdin_payload_sha256"],
            "one_attempt_only": True, "cpu_only": True,
            "native_training_authorized": False,
            "evaluation_authorized": False,
            "aws_action_authorized": False,
            "submission_authorized": False,
            "no_host_action_by_reviewer": True}


def expected_assignment(plan: dict, review_sha256: str) -> dict:
    return {"schema": "trn22-a2-cpu-g2-supervisor-assignment-v1",
            "prepared_by_role": "supervisor", "host": "trn22", "label": LABEL,
            "source_checkout": SOURCE, "scratch_root": ROOT,
            "fixed_attempt_marker": str(MARKER),
            "request_sha256": plan["request_sha256"],
            "binding_sha256": plan["binding_sha256"],
            "independent_final_review_sha256": review_sha256,
            "operator_runner_sha256": plan["wrapper_sha256"],
            "remote_runner_sha256": plan["remote_sha256"],
            "commands_sha256": plan["commands_sha256"],
            "audit_ssh_argv_sha256": plan["audit_argv_sha256"],
            "run_ssh_argv_sha256": plan["run_argv_sha256"],
            "stdin_payload_sha256": plan["stdin_payload_sha256"],
            "cpu_only_one_attempt_authorized": True,
            "fresh_live_preflight_required": True,
            "unconditional_distinct_session_postflight_required": True,
            "native_training_authorized": False,
            "evaluation_authorized": False,
            "aws_action_authorized": False,
            "submission_authorized": False}


def read_actual(path: Path, *, state: bool = False) -> bytes:
    try:
        named = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError("DISPATCH HOLD: actual review/assignment/state absent") from error
    unsafe = 0o002 if state else 0o022
    require(stat.S_ISREG(named.st_mode) and named.st_uid == os.getuid() and
            named.st_nlink == 1 and not (stat.S_IMODE(named.st_mode) & unsafe) and
            0 < named.st_size <= 1_048_576, "unsafe authority/state file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) ==
                (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns),
                "authority/state identity raced")
        raw = os.read(fd, 1_048_577)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    final = path.lstat()
    require(len(raw) == named.st_size and
            (closed.st_dev, closed.st_ino, closed.st_size, closed.st_mtime_ns) ==
                (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns) and
            (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) ==
                (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns),
            "authority/state bytes raced")
    return raw


def authorize_actual_files(plan: dict) -> dict:
    review_raw = read_actual(FINAL_REVIEW)
    assignment_raw = read_actual(ASSIGNMENT)
    review = json.loads(review_raw, object_pairs_hook=unique_json)
    assignment = json.loads(assignment_raw, object_pairs_hook=unique_json)
    require(review == expected_review(plan), "DISPATCH HOLD: independent final review not exact")
    require(assignment == expected_assignment(plan, sha(review_raw)),
            "DISPATCH HOLD: Supervisor one-attempt assignment not exact")
    return {"review_sha256": sha(review_raw), "assignment_sha256": sha(assignment_raw)}


def local_breakers() -> str:
    stops = (REPO / "research/agentic/STOP", REPO / "research/v2/STOP",
             STATE.parent / "STOP", STATE.parent.parent / "agentic/STOP")
    require(all(not p.exists() and not p.is_symlink() for p in stops),
            "local STOP present")
    raw = read_actual(STATE, state=True)
    state = json.loads(raw, object_pairs_hook=unique_json)
    require(state.get("schema") == "trn-frontier-state-v2" and
            state["breakers"]["global_correctness"].get("tripped") is False and
            state["breakers"]["trn22"].get("tripped") is False and
            read_actual(STATE, state=True) == raw,
            "global/trn22 breaker changed or tripped")
    require(all(not p.exists() and not p.is_symlink() for p in stops),
            "local STOP appeared")
    return sha(raw)


class Journal:
    """The one fixed O_EXCL|O_NOFOLLOW marker is consumed before any SSH."""

    def __init__(self):
        parent = MARKER.parent.lstat()
        require(stat.S_ISDIR(parent.st_mode) and parent.st_uid == os.getuid() and
                stat.S_IMODE(parent.st_mode) == 0o700,
                "attempt marker parent must be owner-private mode 0700")
        self.parent_fd = os.open(MARKER.parent, os.O_RDONLY | os.O_DIRECTORY |
                                 os.O_NOFOLLOW | os.O_CLOEXEC)
        self.fd = None
        try:
            opened = os.fstat(self.parent_fd)
            require((opened.st_dev, opened.st_ino) == (parent.st_dev, parent.st_ino),
                    "attempt marker parent changed")
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

    def emit(self, event: str, **fields) -> None:
        raw = canonical({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         "event": event, **fields}) + b"\n"
        while raw:
            count = os.write(self.fd, raw)
            require(count > 0, "attempt journal write failed")
            raw = raw[count:]
        os.fsync(self.fd)

    def close(self) -> None:
        os.fsync(self.fd)
        os.close(self.fd)
        os.fsync(self.parent_fd)
        os.close(self.parent_fd)


def ssh_transport(argv, *, input_data: bytes, timeout_seconds: int):
    require(argv[:2] == ["ssh", "-T"] and "StrictHostKeyChecking=yes" in argv,
            "non-pinned SSH transport")
    return subprocess.run(argv, input=input_data, capture_output=True,
                          check=False, timeout=timeout_seconds)


def _transport_record(journal: Journal, phase: str, argv: list[str], result) -> None:
    require(len(result.stdout) <= 3_000_000 and len(result.stderr) <= 3_000_000,
            "SSH output too large")
    journal.emit("SSH_TRANSPORT", phase=phase,
                 argv_sha256=sha(canonical(argv)), returncode=result.returncode,
                 stdout=result.stdout.decode("utf-8", "replace"),
                 stderr=result.stderr.decode("utf-8", "replace"))


def run_with_postflight(plan: dict, journal: Journal, transport, authority: dict) -> dict:
    """One bounded CPU attempt, with an unconditional distinct SSH postflight."""
    failure = None
    raw_run = None
    post = None
    contact_started = False
    remote = importlib.util.spec_from_file_location("a2_remote_validator", REMOTE)
    module = importlib.util.module_from_spec(remote)
    remote.loader.exec_module(module)
    try:
        contact_started = True
        pre = transport(plan["audit_argv"], input_data=plan["stdin_payload"],
                        timeout_seconds=240)
        _transport_record(journal, "fresh_full_preflight", plan["audit_argv"], pre)
        require(pre.returncode == 0 and not pre.stderr, "preflight SSH failed")
        pre_json = json.loads(pre.stdout, object_pairs_hook=unique_json)
        require(pre_json.get("status") == "READ_ONLY_A2_G2_AUDIT", "wrong preflight status")
        module.validate_snapshot(pre_json["snapshot"], plan["request"], phase="before")
        journal.emit("FRESH_HOST_PREFLIGHT_PASS",
                     boot_id=pre_json["snapshot"]["base"]["identity"]["boot_id"])
        require(authorize_actual_files(plan) == authority,
                "actual review or assignment changed after preflight")
        local_breakers()
        run = transport(plan["run_argv"], input_data=plan["stdin_payload"],
                        timeout_seconds=5100)
        _transport_record(journal, "one_cpu_g2_attempt", plan["run_argv"], run)
        require(run.returncode == 0 and not run.stderr, "one CPU G2 SSH failed")
        raw_run = json.loads(run.stdout, object_pairs_hook=unique_json)
        require(raw_run.get("status") ==
                "CPU_G2_RAW_EVIDENCE_PENDING_INDEPENDENT_NUMERIC_REVIEW" and
                raw_run.get("native_training_authorized") is False and
                raw_run.get("evaluation_authorized") is False,
                "one CPU G2 result is not raw-review-only success")
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)[:500]}
        journal.emit("CPU_G2_HOLD", failure=failure, preserve_partial=True,
                     same_label_retry_authorized=False)
    finally:
        if contact_started:
            try:
                result = transport(plan["audit_argv"], input_data=plan["stdin_payload"],
                                   timeout_seconds=240)
                _transport_record(journal, "unconditional_distinct_session_postflight",
                                  plan["audit_argv"], result)
                require(result.returncode == 0 and not result.stderr,
                        "distinct postflight SSH failed")
                post_json = json.loads(result.stdout, object_pairs_hook=unique_json)
                require(post_json.get("status") == "READ_ONLY_A2_G2_AUDIT",
                        "wrong postflight status")
                post = module.validate_snapshot(post_json["snapshot"], plan["request"],
                                                phase="after")
                journal.emit("POSTFLIGHT_RECORDED",
                             boot_id=post["base"]["identity"]["boot_id"])
            except BaseException as error:
                failure = failure or {"type": type(error).__name__,
                                      "message": "postflight: " + str(error)[:450]}
                journal.emit("POSTFLIGHT_HOLD", failure=failure, preserve_partial=True,
                             same_label_retry_authorized=False)
    if failure or raw_run is None or post is None:
        journal.emit("TERMINAL_HOLD", failure=failure, preserve_partial=True,
                     native_training_authorized=False, evaluation_authorized=False)
        return {"status": "HOLD_PRESERVE_PARTIAL_NO_RETRY"}
    if post["scratch"][module.RECEIPT].get("sha256") != \
            raw_run["artifacts"]["g2_receipt_sha256"]:
        journal.emit("TERMINAL_HOLD", failure="postflight G2 receipt changed",
                     preserve_partial=True, same_label_retry_authorized=False,
                     native_training_authorized=False, evaluation_authorized=False)
        return {"status": "HOLD_PRESERVE_PARTIAL_NO_RETRY"}
    journal.emit("RAW_EVIDENCE_PENDING_INDEPENDENT_TABLE_ATTRIBUTION",
                 receipt_sha256=raw_run["artifacts"]["g2_receipt_sha256"],
                 rank_hashes={key: value["sha256"] for key, value in
                              raw_run["artifacts"]["rank_records"].items()},
                 native_training_authorized=False, evaluation_authorized=False)
    return {"status": "CPU_G2_RAW_EVIDENCE_PENDING_INDEPENDENT_NUMERIC_REVIEW",
            "g2_receipt_sha256": raw_run["artifacts"]["g2_receipt_sha256"],
            "numeric_gate": "HOLD_INDEPENDENT_PER_TABLE_PROOF"}


def operator_once(transport=ssh_transport) -> dict:
    """Future Operator entry point only after exact independent review/assignment."""
    plan = build_plan()
    authority = authorize_actual_files(plan)  # Absent in this Implementer package.
    local_breakers()
    journal = Journal()
    try:
        journal.emit("CPU_ONLY_ASSIGNMENT_BOUND", authority=authority,
                     request_sha256=plan["request_sha256"],
                     marker_consumed_before_ssh=True)
        return run_with_postflight(plan, journal, transport, authority)
    finally:
        journal.close()


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: independent final review and Supervisor assignment absent")
