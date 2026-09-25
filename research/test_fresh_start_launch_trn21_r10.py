#!/usr/bin/env python3
"""Offline trn21 r10 A0 profile tests; every process and host contact is fake."""

import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest

import fresh_start_launch as launcher
import test_fresh_start_launch_nonassert as fake


LABEL = "fs-20260925-v2-c5c-a0-control-trn21-r1"
REMOTE = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10"
HOSTNAME = "ip-172-31-34-85"
CACHE = ("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1"
         "/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train")
TORCHRUN = "/home/ubuntu/workspace/native_venv/bin/torchrun"
ASSIGNMENTS = (
    "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
    "NEURON_COMPETITION_R1_SEED", "NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS",
    "TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
    "TORCH_NEURONX_HLO_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
    "NEURON_COMPILE_CACHE_DIR", "TORCH_NEURONX_DEBUG_DIR",
)


def environment():
    return {
        "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
        "NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS": "1800",
        "NEURON_COMPETITION_R1_SEED": "42",
        "NEURON_LOGICAL_NC_CONFIG": "2",
        "NEURON_COMPILE_CACHE_DIR": CACHE + "/compiler",
        "TORCHINDUCTOR_CACHE_DIR": CACHE + "/inductor",
        "TORCH_NEURONX_DEBUG_DIR": CACHE + "/debug",
        "TORCH_NEURONX_HLO_CACHE_DIR": CACHE + "/hlo",
        "TORCH_NEURONX_NEFF_CACHE_DIR": CACHE + "/neff",
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": CACHE + "/local",
    }


def command_for(label=LABEL, remote=REMOTE, values=None):
    values = environment() if values is None else values
    return [
        "/usr/bin/timeout", "--foreground", "--signal=TERM",
        "--kill-after=60s", "3600s", "/usr/bin/env", "-i",
        "HOME=/home/ubuntu",
        "PATH=/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
        "NKI_VENV_PATH=/home/ubuntu/workspace/native_venv", "PYTHONUNBUFFERED=1",
        *[name + "=" + values[name] for name in ASSIGNMENTS],
        "NEURON_COMPETITION_R1_OUT_DIR=" + remote + "/out/" + label,
        "NEURON_COMPETITION_R1_STEP_LOG=" + remote + "/research/run-logs/" + label + ".steps.jsonl",
        "TRAINIUM_FRESH_START_RUN_ID=" + label,
        TORCHRUN, "--standalone", "--nproc_per_node=4", "train.py",
        "--n-embd", "1024", "--no-ngram-ve", "--no-qk-shift",
        "--compile-sdpa-direct", "--nki-local-conv", "--bf16-norm-output",
        "--pack-factor", "32", "--clip-after-reduce", "--seq-len", "1024",
        "--no-eval-public", "--matrix-lr", "0.015",
    ]


def request_template(monitor):
    request = fake.request_template(REMOTE, monitor)
    request.update(host="trn21", expected_hostname=HOSTNAME, label=LABEL,
                   remote=REMOTE, environment=environment(), command=command_for())
    return request


def probe_real_trn21_interpreter():
    """Run only within a child started with -O or inherited PYTHONOPTIMIZE."""
    optimize = sys.flags.optimize
    inherited = os.environ.get("PYTHONOPTIMIZE") if "PYTHONOPTIMIZE" in os.environ else None
    local = fake.LocalNonAssertTests("test_valid_unoptimized_request_reaches_only_fake_ssh")
    remote = fake.RemoteNonAssertTests("test_valid_unoptimized_remote_reaches_only_fake_guard")
    local_error, contacts, receipts = local.run_local(
        request_template("/home/ubuntu/fake-monitor"), compiled_optimize=optimize,
        runtime_optimize=optimize, pythonoptimize=inherited)
    remote_error, calls, state = remote.run_remote(
        request_factory=lambda _checkout, monitor: request_template(monitor),
        path_alias=REMOTE, hostname=HOSTNAME, compiled_optimize=optimize,
        runtime_optimize=optimize, pythonoptimize=inherited,
        real_science_source=True)
    if (not isinstance(local_error, RuntimeError)
            or "optimized Python" not in str(local_error)
            or contacts or receipts
            or not isinstance(remote_error, RuntimeError)
            or "optimized Python" not in str(remote_error)
            or calls["timeout_version"] or calls["git"] or calls["guard"]
            or state["host_dir"] or state["no_spawn"] is not None):
        raise RuntimeError("trn21 optimized interpreter reached a contact or write")
    print("TRN21_REAL_MODE_REFUSED")


class ExactTrn21R10Tests(unittest.TestCase):
    def setUp(self):
        self.local = fake.LocalNonAssertTests(
            "test_valid_unoptimized_request_reaches_only_fake_ssh")
        self.remote = fake.RemoteNonAssertTests(
            "test_valid_unoptimized_remote_reaches_only_fake_guard")

    def run_remote(self, **kwargs):
        return self.remote.run_remote(
            request_factory=lambda _checkout, monitor: request_template(monitor),
            path_alias=REMOTE, hostname=HOSTNAME, real_science_source=True,
            **kwargs)

    def assert_early_refusal(self, change):
        for compiled in (0, 1):
            with self.subTest(compiled=compiled):
                request = request_template("/home/ubuntu/fake-monitor")
                change(request)
                error, contacts, receipts = self.local.run_local(
                    request, compiled_optimize=compiled)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(contacts, [])
                self.assertEqual(receipts, [])
                error, calls, state = self.run_remote(
                    compiled_optimize=compiled,
                    mutation=lambda request, *_: change(request))
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(calls["timeout_version"], [])
                self.assertEqual(calls["git"], [])
                self.assertEqual(calls["guard"], [])
                self.assertFalse(state["host_dir"])
                self.assertIsNone(state["no_spawn"])
                self.assertFalse(state["manifest"] or state["job"] or state["log"])

    def test_exact_trn21_profile_reaches_only_fake_contacts(self):
        request = request_template("/home/ubuntu/fake-monitor")
        self.assertTrue(launcher.bounded_timeout_command(
            request["command"], request["host"], request["label"],
            request["remote"], request["environment"]))
        self.assertEqual(request["command"], command_for())
        for compiled in (0, 1):
            with self.subTest(compiled=compiled):
                error, contacts, receipts = self.local.run_local(
                    copy.deepcopy(request), compiled_optimize=compiled)
                self.assertIsNone(error)
                self.assertEqual(len(contacts), 1)
                self.assertEqual(contacts[0][0][0][:2], ["fake-ssh", "fake-host-21"])
                self.assertEqual(len(receipts), 1)
                error, calls, state = self.run_remote(
                    compiled_optimize=compiled, allow_fake_guard=True)
                self.assertIsNone(error)
                self.assertEqual(calls["timeout_version"],
                                 [["/usr/bin/timeout", "--version"]])
                self.assertEqual(len(calls["guard"]), 1)
                self.assertEqual(len(calls["interlock"]), 5)
                self.assertTrue(state["host_dir"] and state["manifest"] and state["job"])

    def test_disabled_request_refuses_before_any_contact(self):
        self.assert_early_refusal(lambda request: request.update(dispatch_authorized=False))

    def test_host_label_checkout_and_outputs_are_exact(self):
        old_checkout = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2"
        reserved_r4 = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r4"

        def retarget_label(request):
            request["label"] = "fs-20260925-v2-c5c-a0-control-trn21-r2"
            request["command"] = command_for(label=request["label"])

        def retarget_checkout(request, path):
            request["remote"] = path
            request["command"] = command_for(remote=path)

        changes = (
            lambda request: request.update(host="trn22"),
            lambda request: request.update(host="trn23"),
            lambda request: request.update(expected_hostname="ip-172-31-32-253"),
            retarget_label,
            lambda request: retarget_checkout(request, old_checkout),
            lambda request: retarget_checkout(request, reserved_r4),
            lambda request: request["command"].__setitem__(
                request["command"].index("NEURON_COMPETITION_R1_OUT_DIR=" + REMOTE + "/out/" + LABEL),
                "NEURON_COMPETITION_R1_OUT_DIR=" + REMOTE + "/out/other"),
            lambda request: request["command"].__setitem__(
                request["command"].index("NEURON_COMPETITION_R1_STEP_LOG=" + REMOTE + "/research/run-logs/" + LABEL + ".steps.jsonl"),
                "NEURON_COMPETITION_R1_STEP_LOG=" + REMOTE + "/research/run-logs/other.steps.jsonl"),
            lambda request: request["command"].__setitem__(
                request["command"].index("TRAINIUM_FRESH_START_RUN_ID=" + LABEL),
                "TRAINIUM_FRESH_START_RUN_ID=other"),
        )
        for change in changes:
            with self.subTest(change=change):
                self.assert_early_refusal(change)

    def test_timeout_env_argv_and_cache_mutations_refuse(self):
        mutations = (
            lambda request: request["command"].__setitem__(0, "timeout"),
            lambda request: request["command"].__setitem__(0, "/tmp/fake-timeout"),
            lambda request: request["command"].__setitem__(3, "--kill-after=0s"),
            lambda request: request["command"].__setitem__(4, "0s"),
            lambda request: request["command"].__setitem__(5, "env"),
            lambda request: request["command"].__setitem__(8, "PATH=/tmp/fake:/usr/bin"),
            lambda request: request["command"].insert(request["command"].index(TORCHRUN),
                                                       "PYTHONOPTIMIZE=1"),
            lambda request: request["command"].__setitem__(-1, "0.025"),
            lambda request: request["command"].__setitem__(
                request["command"].index(TORCHRUN), "/bin/sh"),
            lambda request: request["command"].__setitem__(
                request["command"].index("--nproc_per_node=4"), "--nproc_per_node=8"),
            lambda request: request["environment"].update(PATH="/tmp/fake:/usr/bin"),
            lambda request: request["environment"].update(PYTHONOPTIMIZE="1"),
            lambda request: request["environment"].update(PYTHONPATH="/tmp/fake"),
            lambda request: request["environment"].update(UNREVIEWED="1"),
            lambda request: request["environment"].update(NEURON_COMPETITION_R1_SEED="43"),
            lambda request: request["environment"].update(NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS="1801"),
            lambda request: request["environment"].update(NEURON_LOGICAL_NC_CONFIG="1"),
            lambda request: request["environment"].update(NEURON_CC_FLAGS="--optlevel=2"),
            lambda request: request["environment"].update(NEURON_COMPILE_CACHE_DIR="/tmp/cache"),
        )
        for change in mutations:
            with self.subTest(change=change):
                self.assert_early_refusal(change)

    def test_source_fingerprint_and_commit_gates_remain(self):
        for change in (
            lambda request: request["fingerprint_paths"].pop(),
            lambda request: request["fingerprint_paths"].append("research/agentic/private.json"),
            lambda request: request["fingerprint_paths"].__setitem__(0, "prepare.py"),
        ):
            with self.subTest(change=change):
                self.assert_early_refusal(change)
        for kwargs in ({"dirty": b"train.py\n"}, {"commit": "b" * 40},
                       {"source_corruption": "train.py"},
                       {"source_corruption": "prepare.py"}):
            with self.subTest(kwargs=kwargs):
                error, calls, state = self.run_remote(**kwargs)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(calls["guard"], [])
                self.assertEqual(calls["preflight"], [])
                self.assertEqual(state["no_spawn"]["stage"], "source_identity")

    def test_non_gnu_timeout_refuses_before_local_receipt_or_remote_checkout(self):
        banner = "FAKE_TIMEOUT_NO_DEADLINE\n"
        error, contacts, receipts = self.local.run_local(
            request_template("/home/ubuntu/fake-monitor"), timeout_banner=banner)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(contacts, [])
        self.assertEqual(receipts, [])
        error, calls, state = self.run_remote(timeout_banner=banner)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(calls["timeout_version"], [["/usr/bin/timeout", "--version"]])
        self.assertFalse(state["host_dir"])
        self.assertIsNone(state["no_spawn"])
        self.assertEqual(calls["git"], [])

    def test_real_optimized_interpreters_refuse_trn21(self):
        code = ("import test_fresh_start_launch_trn21_r10 as probe;"
                "probe.probe_real_trn21_interpreter()")
        clean = dict(os.environ)
        clean.pop("PYTHONOPTIMIZE", None)
        clean["PYTHONDONTWRITEBYTECODE"] = "1"
        cases = (("dash_o", ["-O"], None), ("dash_oo", ["-OO"], None),
                 ("env_1", [], "1"), ("env_0", [], "0"),
                 ("env_empty", [], ""))
        for name, flags, value in cases:
            with self.subTest(mode=name):
                env = dict(clean)
                if value is not None:
                    env["PYTHONOPTIMIZE"] = value
                result = subprocess.run(
                    [sys.executable, *flags, "-c", code], cwd=Path(__file__).parent,
                    env=env, capture_output=True, text=True, timeout=10, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("TRN21_REAL_MODE_REFUSED", result.stdout)


if __name__ == "__main__":
    unittest.main()
