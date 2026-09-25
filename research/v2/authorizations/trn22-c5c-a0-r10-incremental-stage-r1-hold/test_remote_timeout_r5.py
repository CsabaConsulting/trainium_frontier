"""No-host adversarial tests for the r5 remote timeout identity boundary."""

import importlib.util
import json
from pathlib import Path
import subprocess
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("trn22_r10_timeout_r5_operator", HERE / "operator_once.py")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


def result(stdout=b"", stderr=b"", returncode=0):
    return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class RemoteTimeoutIdentityTests(unittest.TestCase):
    def test_local_uutils_is_not_consulted_remote_gnu_keeps_fixed_deadline(self):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            if argv[:2] == ["/usr/bin/timeout", "--version"]:
                return result(b"timeout (uutils coreutils) 0.8.0\n")
            self.assertEqual(argv[0], "/usr/bin/ssh")
            if argv[-1] == "/usr/bin/env -i LC_ALL=C /usr/bin/timeout --version":
                return result(b"timeout (GNU coreutils) 9.1\nCopyright (C) GNU\n")
            return result(json.dumps({"phase": "preflight"}).encode())

        with (
            mock.patch.object(operator, "_local_stop_and_breakers") as stop,
            mock.patch.object(operator, "_ssh_identity") as identity,
            mock.patch.object(operator.subprocess, "run", side_effect=fake_run),
        ):
            receipt = operator._ssh_phase("preflight", None, b"", b"# inert source\n")
        self.assertEqual(receipt, {"phase": "preflight"})
        self.assertEqual(len(calls), 2)
        probe_argv, probe_kwargs = calls[0]
        phase_argv, phase_kwargs = calls[1]
        self.assertEqual(probe_argv[:-1], phase_argv[:-1])
        self.assertEqual(probe_argv[0], "/usr/bin/ssh")
        self.assertIn("StrictHostKeyChecking=yes", probe_argv)
        self.assertIn("UserKnownHostsFile=" + str(operator.KNOWN_HOSTS), probe_argv)
        self.assertIn("IdentitiesOnly=yes", probe_argv)
        self.assertEqual(probe_argv[-1],
                         "/usr/bin/env -i LC_ALL=C /usr/bin/timeout --version")
        self.assertEqual(probe_kwargs["timeout"], 25)
        self.assertEqual(phase_kwargs["timeout"], 350)
        self.assertIn("/usr/bin/timeout --foreground --signal=TERM ", phase_argv[-1])
        self.assertIn("--kill-after=15s 320s /usr/bin/env -i", phase_argv[-1])
        self.assertEqual(stop.call_count, 3)
        identity.assert_called_once_with()

    def test_remote_wrong_output_stderr_exit_or_excess_bytes_hold(self):
        cases = (
            result(b"timeout (uutils coreutils) 0.8.0\n"),
            result(b"fake timeout\n"),
            result(b"timeout (GNU coreutils) 9.1 injected\n"),
            result(b"timeout (GNU coreutils) 9.1\n", stderr=b"warning\n"),
            result(b"timeout (GNU coreutils) 9.1\n", returncode=1),
            result(b"timeout (GNU coreutils) 9.1\n" + b"x" * 4096),
        )
        for reply in cases:
            with self.subTest(reply=reply):
                with mock.patch.object(operator.subprocess, "run", return_value=reply) as run:
                    with self.assertRaisesRegex(RuntimeError,
                                                "remote absolute GNU timeout identity"):
                        operator._remote_timeout_identity()
                self.assertEqual(run.call_args.args[0][0], "/usr/bin/ssh")
                self.assertEqual(run.call_args.kwargs["timeout"], 25)

    def test_remote_probe_timeout_holds(self):
        with mock.patch.object(operator.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("pinned-ssh", 25)) as run:
            with self.assertRaisesRegex(RuntimeError, "remote absolute timeout identity probe timed out"):
                operator._remote_timeout_identity()
        self.assertEqual(run.call_args.args[0][0], "/usr/bin/ssh")
        self.assertEqual(run.call_args.kwargs["timeout"], 25)

    def test_identity_refusal_precedes_probe_marker_and_host_phase(self):
        remote = mock.Mock(side_effect=AssertionError("host phase attempted"))
        synthetic_sys = types.SimpleNamespace(flags=types.SimpleNamespace(optimize=0))
        with (
            mock.patch.object(operator, "sys", synthetic_sys),
            mock.patch.dict(operator.os.environ, {}, clear=True),
            mock.patch.object(operator, "_json", side_effect=[
                ({"dispatch_authorized": True}, "request"),
                ({}, "review"), ({}, "assignment")]),
            mock.patch.object(operator, "_source_bytes", return_value=(b"bundle", b"remote")),
            mock.patch.object(operator, "_authority"),
            mock.patch.object(operator, "_local_stop_and_breakers") as stop,
            mock.patch.object(operator, "_ssh_identity",
                              side_effect=RuntimeError("identity HOLD")) as identity,
            mock.patch.object(operator, "_remote_timeout_identity") as probe,
            mock.patch.object(operator, "_marker") as marker,
        ):
            with self.assertRaisesRegex(RuntimeError, "identity HOLD"):
                operator.stage_once(remote_call=remote)
            self.assertEqual(stop.call_count, 1)
            identity.assert_called_once_with()
            probe.assert_not_called()
            marker.assert_not_called()
            remote.assert_not_called()

    def test_phase_identity_refusal_precedes_probe_and_host_phase(self):
        with (
            mock.patch.object(operator, "_local_stop_and_breakers") as stop,
            mock.patch.object(operator, "_ssh_identity",
                              side_effect=RuntimeError("identity HOLD")) as identity,
            mock.patch.object(operator, "_remote_timeout_identity") as probe,
            mock.patch.object(operator.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "identity HOLD"):
                operator._ssh_phase("preflight", None, b"", b"# inert source\n")
            self.assertEqual(stop.call_count, 1)
            identity.assert_called_once_with()
            probe.assert_not_called()
            run.assert_not_called()

    def test_remote_probe_refusal_precedes_one_attempt_marker(self):
        remote = mock.Mock(side_effect=AssertionError("host phase attempted"))
        synthetic_sys = types.SimpleNamespace(flags=types.SimpleNamespace(optimize=0))
        with (
            mock.patch.object(operator, "sys", synthetic_sys),
            mock.patch.dict(operator.os.environ, {}, clear=True),
            mock.patch.object(operator, "_json", side_effect=[
                ({"dispatch_authorized": True}, "request"),
                ({}, "review"), ({}, "assignment")]),
            mock.patch.object(operator, "_source_bytes", return_value=(b"bundle", b"remote")),
            mock.patch.object(operator, "_authority"),
            mock.patch.object(operator, "_local_stop_and_breakers") as stop,
            mock.patch.object(operator, "_ssh_identity") as identity,
            mock.patch.object(operator, "_remote_timeout_identity",
                              side_effect=RuntimeError("remote timeout HOLD")) as probe,
            mock.patch.object(operator, "_marker") as marker,
        ):
            with self.assertRaisesRegex(RuntimeError, "remote timeout HOLD"):
                operator.stage_once(remote_call=remote)
            self.assertEqual(stop.call_count, 2)
            identity.assert_called_once_with()
            probe.assert_called_once_with()
            marker.assert_not_called()
            remote.assert_not_called()

    def test_late_stop_after_phase_probe_precedes_host_phase(self):
        with (
            mock.patch.object(operator, "_local_stop_and_breakers",
                              side_effect=[None, None, RuntimeError("late STOP")]) as stop,
            mock.patch.object(operator, "_ssh_identity") as identity,
            mock.patch.object(operator, "_remote_timeout_identity") as probe,
            mock.patch.object(operator.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "late STOP"):
                operator._ssh_phase("write", {}, b"payload", b"# inert source\n")
            self.assertEqual(stop.call_count, 3)
            identity.assert_called_once_with()
            probe.assert_called_once_with()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
