"""Mocked end-to-end and unit tests for research/agentic/autoloop.

Run from the repository root:

    python -m unittest discover -s research/agentic/tests -v

No network, no SSH, no git subprocess, no Bedrock. The fake transport runs the SAME argv guard as
the real SSH transport, so every remote command the controller would issue is exercised.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoloop import config as C  # noqa: E402
from autoloop.analyst import compare  # noqa: E402
from autoloop.controller import Controller  # noqa: E402
from autoloop.implementer import implement_code_proposal  # noqa: E402
from autoloop.manifest import ManifestError, Proposal, assert_manifest_legal, build_manifest, known_train_flags  # noqa: E402
from autoloop.models import BedrockClient, ModelError, ModelReply, parse_json_block  # noqa: E402
from autoloop.remote import (  # noqa: E402
    CommandResult,
    GuardError,
    LocalGit,
    RemoteError,
    RemoteOperator,
    check_key_path,
    guard_remote_argv,
)
from autoloop.state import Store  # noqa: E402
import autoloop_watchdog as W  # noqa: E402

COMMIT = C.AUTHORITATIVE_COMMIT
HOST = C.PINNED_HOSTS["trn22"]
TRAIN_PY = (
    'parser.add_argument("--weight-ema", action=argparse.BooleanOptionalAction, default=0)\n'
    'parser.add_argument("--weight-ema-decay", type=str, default="0.995")\n'
    'parser.add_argument("--cautious-wd", action=argparse.BooleanOptionalAction, default=0)\n'
    'parser.add_argument("--seed", type=int, default=42)\n'
    'parser.add_argument("--num-steps", type=int)\n'
)
T0 = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------------------------


class FakeHost:
    def __init__(self):
        self.hostname = "ip-10-0-0-22"
        self.files: dict[str, bytes] = {C.REMOTE_PYTHON: b"#!python"}
        self.dirs: set[str] = {C.REMOTE_DATA_CACHE, "/home/ubuntu/workspace/trainium_freshstart_20260918_wave195"}
        self.procs: dict[int, str] = {}
        self.free_kb = 30 * 1024 * 1024
        self.device_holder = ""
        self.repos: dict[str, dict] = {}
        self.dirty: dict[str, str] = {}
        self.launched: list[dict] = []
        self.checkout_override: str | None = None
        self._pid = 4000


class FakeTransport:
    def __init__(self, hosts: dict[str, FakeHost], train_py: str = TRAIN_PY):
        self.hosts = hosts
        self.train_py = train_py.encode()
        self.calls: list[tuple[str, list[str]]] = []

    # -- helpers --
    def _ok(self, argv, out=""):
        return CommandResult(argv, 0, out, "")

    def _fail(self, argv, rc=1, err=""):
        return CommandResult(argv, rc, "", err)

    def run(self, host, argv, timeout=60):
        if host not in C.PINNED_HOSTS.values():
            raise GuardError("host not pinned")
        guard_remote_argv(argv)  # mirror SSHTransport
        self.calls.append((host, list(argv)))
        h = self.hosts[host]
        p = argv[0]
        if p == "hostname":
            return self._ok(argv, h.hostname + "\n")
        if p == "ps":
            if argv[1] == "-p":
                return self._ok(argv, f"{argv[2]}\n") if int(argv[2]) in h.procs else self._fail(argv)
            lines = ["  PID COMMAND"] + [f"{pid} {cmd}" for pid, cmd in h.procs.items()]
            return self._ok(argv, "\n".join(lines) + "\n")
        if p == "fuser":
            return CommandResult(argv, 0, h.device_holder, "/dev/neuron0:") if h.device_holder else self._fail(argv, 1)
        if p == "df":
            return self._ok(argv, f"Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/root 1 1 {h.free_kb} 50% /\n")
        if p == "test":
            flag, path = argv[1], argv[2]
            if flag == "-e":
                return self._ok(argv) if (path in h.files or path in h.dirs) else self._fail(argv)
            if flag == "-d":
                return self._ok(argv) if path in h.dirs else self._fail(argv)
            return self._ok(argv) if path in h.files else self._fail(argv)
        if p == "mkdir":
            for path in [a for a in argv[1:] if a != "-p"]:
                if "-p" not in argv and path in h.dirs:
                    return self._fail(argv, 1, "exists")
                h.dirs.add(path)
            return self._ok(argv)
        if p == "rmdir":
            h.dirs.discard(argv[1])
            return self._ok(argv)
        if p == "rm":
            path = argv[2]
            if path not in h.files:
                return self._fail(argv, 1, "missing")
            del h.files[path]
            return self._ok(argv)
        if p == "cat":
            return self._ok(argv, h.files[argv[1]].decode()) if argv[1] in h.files else self._fail(argv)
        if p == "tail":
            path = argv[-1]
            return self._ok(argv, h.files[path].decode()[-2000:]) if path in h.files else self._fail(argv)
        if p == "sha256sum":
            path = argv[1]
            return self._ok(argv, f"{sha(h.files[path])}  {path}\n") if path in h.files else self._fail(argv)
        if p == "stat":
            path = argv[-1]
            return self._ok(argv, f"{len(h.files[path])}\n") if path in h.files else self._fail(argv)
        if p == "git":
            return self._git(h, argv)
        if p == C.REMOTE_PYTHON:
            return self._launch(h, argv)
        return self._fail(argv, 127, "unknown program")

    def _git(self, h: FakeHost, argv):
        if argv[1] == "clone":
            bundle, d = [a for a in argv[2:] if not a.startswith("-")]
            if bundle not in h.files or d in h.dirs:
                return self._fail(argv, 128, "clone refused")
            h.dirs.add(d)
            h.repos[d] = {"head": None, "commit": h.files[bundle].decode().split(":")[1]}
            return self._ok(argv)
        d, sub = argv[2], argv[3]
        repo = h.repos.get(d)
        if repo is None:
            return self._fail(argv, 128, "not a repo")
        if sub == "checkout":
            repo["head"] = h.checkout_override or argv[-1]
            h.files[f"{d}/train.py"] = self.train_py
            h.files[f"{d}/research/experiment-log.jsonl"] = b""
            return self._ok(argv)
        if sub == "rev-parse":
            return self._ok(argv, (repo["head"] or "") + "\n")
        if sub == "status":
            return self._ok(argv, h.dirty.get(d, ""))
        return self._fail(argv, 1, "unsupported")

    def _launch(self, h: FakeHost, argv):
        spec = json.loads(h.files[argv[2]])
        repo = h.repos.get(spec["cwd"])
        if repo is None or repo["head"] != spec["expect_cwd_head"]:
            return self._ok(argv, json.dumps({"error": "cwd HEAD mismatch"}))
        if h.dirty.get(spec["cwd"], ""):
            return self._ok(argv, json.dumps({"error": "cwd is dirty"}))
        if spec["pidfile"] in h.files or spec["donefile"] in h.files:
            return self._ok(argv, json.dumps({"error": "refusing to relaunch"}))
        h._pid += 1
        pid = h._pid
        h.procs[pid] = " ".join(spec["argv"])
        h.files[spec["pidfile"]] = f"{pid}\n".encode()
        h.files[spec["stdout"]] = b"started\n"
        spec["_pid"] = pid
        h.launched.append(spec)
        return self._ok(argv, json.dumps({"pid": pid}) + "\n")

    def put(self, host, local: Path, remote: str, timeout=300):
        if not remote.startswith(C.REMOTE_AUTOLOOP_DIR + "/"):
            raise GuardError("put outside scratch")
        self.hosts[host].files[remote] = Path(local).read_bytes()
        return CommandResult(["put", remote], 0, "", "")

    def get(self, host, remote: str, local: Path, timeout=300):
        h = self.hosts[host]
        if remote not in h.files:
            return CommandResult(["get", remote], 1, "", "missing")
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(h.files[remote])
        return CommandResult(["get", remote], 0, "", "")


def finish_run(h: FakeHost, spec: dict, *, bpb: float, steps: int, training_seconds: float, rc: int = 0,
               status: str = "keep", checkpoint: bool = True, causality: bool = True, extra: dict | None = None):
    pid = spec["_pid"]
    h.procs.pop(pid, None)
    argv = spec["argv"]
    label = argv[argv.index("--label") + 1]
    run_dir = spec["cwd"]
    max_train = int(argv[argv.index("--max-train-seconds") + 1])
    extra_args = argv[argv.index("--extra-args") + 1:]
    row = {
        "timestamp": "2026-09-22T13:00:00+00:00", "label": label, "solution": "autoloop", "description": "x",
        "launch_command": f"NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --num-steps 5000 --max-train-seconds {max_train} --out-dir out/{label} " + " ".join(extra_args),
        "git_commit": COMMIT[:7], "git_dirty": True, "status": status, "wall_elapsed_seconds": training_seconds + 40,
        "run_log": f"research/run-logs/{label}.log", "checkpoint_path": f"out/{label}/final.pt",
        "training_seconds": training_seconds, "budget_seconds": 1800, "max_train_seconds": max_train, "over_budget": False,
        "num_steps": steps, "world_size": 4, "lnc": 2, "seq_len": 2048, "num_params_M": 55.7,
        "causality_passed": causality, "public_val_bpb": bpb, "eval_over_timeout": False,
    }
    row.update(extra or {})
    h.files[spec["donefile"]] = json.dumps({"returncode": rc, "finished_at": "2026-09-22T13:00:00Z"}).encode()
    h.files[f"{run_dir}/research/run-logs/{label}.launch.json"] = json.dumps({
        "label": label,
        "launch_command": row["launch_command"],
        "git_commit": COMMIT,
        "source_dirty": False,
    }).encode()
    h.files[f"{run_dir}/research/run-logs/{label}.log"] = b"log line\npublic_val_bpb: %f\n" % bpb
    h.files[f"{run_dir}/research/experiment-log.jsonl"] += (json.dumps(row) + "\n").encode()
    h.files[spec["stdout"]] += b"--- logged\n"
    if checkpoint:
        h.files[f"{run_dir}/out/{label}/final.pt"] = b"\x00" * 64
    h.dirty[run_dir] = (
        " M research/experiment-log.jsonl\n"
        f"?? research/run-logs/{label}.launch.json\n"
        f"?? research/run-logs/{label}.log\n"
    )


class FakeModel:
    def __init__(self):
        self.replies: dict[str, list[str]] = {"researcher": [], "verifier": [], "analyst": []}
        self.calls: list[str] = []
        self.max_tokens: list[int] = []
        self.users: dict[str, list[str]] = {}
        self.systems: dict[str, list[str]] = {}

    def complete(self, *, role, model_id, system, user, max_tokens):
        self.calls.append(role)
        self.max_tokens.append(max_tokens)
        self.users.setdefault(role, []).append(user)
        self.systems.setdefault(role, []).append(system)
        if not self.replies.get(role):
            raise ModelError(f"no fake reply queued for {role}")
        return ModelReply(self.replies[role].pop(0), 1000, 300, "end_turn", model_id)


class FakeArchiver:
    def __init__(self, base: Path):
        self.base = base.resolve()
        self.commits: list[tuple[list[str], str]] = []

    def commit(self, paths, message):
        for p in paths:
            Path(p).resolve().relative_to(self.base)  # raises if outside research/agentic
        self.commits.append(([str(p) for p in paths], message))
        return f"sha{len(self.commits)}"


def fake_local_git(tmp: Path) -> LocalGit:
    def runner(argv):
        if argv[0] == "cat-file":
            return CommandResult(argv, 0 if argv[2].startswith(COMMIT) else 1, "", "")
        if argv[0] == "bundle" and argv[1] == "create":
            Path(argv[2]).parent.mkdir(parents=True, exist_ok=True)
            Path(argv[2]).write_bytes(f"bundle:{COMMIT}".encode())
        return CommandResult(argv, 0, "", "")

    return LocalGit(root=tmp, runner=runner, show_runner=lambda c, p: TRAIN_PY.encode() if c == COMMIT else None)


def researcher_json(pid="20260922-weight-ema-on", flags=("--weight-ema",)):
    return json.dumps({
        "proposal_id": pid,
        "hypothesis": "Weight EMA over the warmdown lowers final public_val_bpb at equal steps.",
        "single_causal_change": "enable --weight-ema; everything else identical",
        "flags": list(flags),
        "evidence": ["research/experiment-log.jsonl: T2.2 rows"],
        "duplicate_search_terms": ["weight ema", "ema"],
        "expected_effect": "public_val_bpb lower by >= 0.002",
        "abort_conditions": ["any correctness gate fails"],
    })


def code_researcher_json(pid="20260922-bf16-head"):
    return json.dumps({
        "proposal_id": pid,
        "hypothesis": "A bounded BF16 final projection should increase useful fixed-budget throughput without changing optimizer state.",
        "single_causal_change": "cast only the final projection operands to BF16 and immediately return logits to FP32",
        "flags": [],
        "evidence": ["research/experiment-log.jsonl", "train.py final projection"],
        "duplicate_search_terms": ["bf16 final projection", "bf16 lm head"],
        "expected_effect": "at least three percent more steps with non-worse smoke BPB",
        "abort_conditions": ["dtype boundary is not local", "causality gate fails"],
        "change_type": "code",
        "implementation_plan": "Edit only train.py at the final projection: cast its two operands to BF16, compute that projection, then immediately cast logits to FP32.",
        "files_allowed": ["train.py"],
    })


class Env:
    """Patch every module-level path constant into a temp dir and build the fakes."""

    def __init__(self, cfg_overrides: dict | None = None, boxes=("trn22",)):
        self.stack = ExitStack()
        self.tmp = Path(tempfile.mkdtemp())
        agent = self.tmp / "research" / "agentic"
        agent.mkdir(parents=True)
        (agent / "CURRENT_STATE.md").write_text("# state\nDo not reopen the parked attention redesign, width1152, or compiler-SBUF placement without new evidence.\n")
        (agent / "DECISION_LOG.md").write_text("| Date | Proposal | Commit | Control | Result | Verdict | Evidence |\n|---|---|---|---|---|---|---|\n")
        (self.tmp / "research").mkdir(exist_ok=True)
        ledger = self.tmp / "research" / "experiment-log.jsonl"
        ledger.write_text(json.dumps({"timestamp": "2026-09-08T00:00:00+00:00", "label": "old-row", "git_commit": "252fd6f", "status": "keep", "launch_command": "torchrun train.py --cautious-wd", "public_val_bpb": 1.05}) + "\n")
        for name, value in {
            "ROOT": self.tmp, "AGENT_DIR": agent, "RUNTIME_DIR": agent / "runtime", "EXPERIMENTS_DIR": agent / "experiments",
            "PROPOSALS_DIR": agent / "proposals", "STOP_FILE": agent / "STOP", "STATE_FILE": agent / "runtime" / "autoloop_state.json",
            "BUNDLE_DIR": agent / "runtime" / "bundles", "DECISION_LOG": agent / "DECISION_LOG.md", "LOCAL_LEDGER": ledger,
        }.items():
            self.stack.enter_context(mock.patch.object(C, name, value))
        self.agent = agent
        self.cfg = C.LoopConfig(researcher_model_id="us.anthropic.claude-fable-5-1", verifier_model_id="us.openai.gpt-6-astra", boxes=list(boxes))
        for k, v in (cfg_overrides or {}).items():
            if k == "caps":
                for ck, cv in v.items():
                    setattr(self.cfg.caps, ck, cv)
            else:
                setattr(self.cfg, k, v)
        self.hosts = {h: FakeHost() for h in C.PINNED_HOSTS.values()}
        self.transport = FakeTransport(self.hosts)
        self.model = FakeModel()
        self.archiver = FakeArchiver(agent)
        self.now = T0
        self.git = fake_local_git(self.tmp)

    def controller(self) -> Controller:
        store = Store(root=self.agent, state_path=C.STATE_FILE, experiments_dir=C.EXPERIMENTS_DIR)
        op = RemoteOperator(self.transport, self.git, self.cfg, store.controller_id, clock=lambda: self.now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        return Controller(cfg=self.cfg, store=store, operator=op, model=self.model, archiver=self.archiver, local_git=self.git,
                          stop_file=C.STOP_FILE, now=lambda: self.now, train_py_text=TRAIN_PY)

    def advance(self, seconds: int):
        self.now = self.now + timedelta(seconds=seconds)

    @property
    def h(self) -> FakeHost:
        return self.hosts[HOST]

    def close(self):
        self.stack.close()


# ---------------------------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------------------------


class GuardTests(unittest.TestCase):
    def test_rejects_forbidden_paths_and_programs(self):
        bad = [
            ["cat", "/home/ubuntu/workspace/trainium_freshstart_20260918_wave195/x"],
            ["aws", "ec2", "stop-instances"],
            ["rm", "-rf", "/home/ubuntu/autoloop"],
            ["git", "-C", f"{C.REMOTE_RUNS_DIR}/x", "clean", "-fdx"],
            ["git", "-C", "/home/ubuntu/workspace/other", "status", "--porcelain"],
            ["git", "push", "origin", "main"],
            ["bash", "-lc", "echo hi"],
            [C.REMOTE_PYTHON, "-c", "print(1)"],
            [C.REMOTE_PYTHON, "/home/ubuntu/evil.py", "/x"],
            ["cat", "/home/ubuntu/.ssh/authorized_keys"],
            ["cat", f"{C.REMOTE_AUTOLOOP_DIR}/x", "|", "sh"],
            ["tail", "-n", "5", f"{C.REMOTE_AUTOLOOP_DIR}/x;", "reboot"],
            ["fuser", "/dev/neuron1"],
            ["rmdir", f"{C.REMOTE_RUNS_DIR}/somerun"],
            ["rm", "-rf", f"{C.REMOTE_RUNS_DIR}/somerun"],
            ["rm", "--", f"{C.REMOTE_RUNS_DIR}/somerun/train.py"],
            ["rm", "--", f"{C.REMOTE_RUNS_DIR}/../other/out/x/final.pt"],
            ["mkdir", "/home/ubuntu/elsewhere"],
        ]
        for argv in bad:
            with self.assertRaises(GuardError, msg=argv):
                guard_remote_argv(argv)

    def test_allows_operator_command_shapes(self):
        run_dir = f"{C.REMOTE_RUNS_DIR}/abc"
        good = [
            ["hostname"], ["ps", "-eo", "pid,args"], ["fuser", "/dev/neuron0"], ["df", "-Pk", "/"],
            ["test", "-d", C.REMOTE_DATA_CACHE], ["test", "-x", C.REMOTE_PYTHON],
            ["mkdir", "-p", f"{C.REMOTE_AUTOLOOP_DIR}/locks"], ["rmdir", f"{C.REMOTE_AUTOLOOP_DIR}/locks/trn22.lock"],
            ["git", "clone", "--quiet", "--no-checkout", f"{C.REMOTE_AUTOLOOP_DIR}/bundles/b.bundle", run_dir],
            ["git", "-C", run_dir, "checkout", "--quiet", "--detach", COMMIT],
            ["git", "-C", run_dir, "rev-parse", "HEAD"], ["git", "-C", run_dir, "status", "--porcelain"],
            ["git", "-C", run_dir, "status", "--porcelain", "--untracked-files=all"],
            ["sha256sum", f"{run_dir}/train.py"], ["stat", "-c", "%s", f"{run_dir}/out/x/final.pt"],
            ["rm", "--", f"{run_dir}/out/al-20260922-test-arm-smoke-r1/final.pt"],
            [C.REMOTE_PYTHON, f"{C.REMOTE_AUTOLOOP_DIR}/launchers/l/remote_launcher.py", f"{C.REMOTE_AUTOLOOP_DIR}/launchers/l/launch.json"],
        ]
        for argv in good:
            guard_remote_argv(argv)

    def test_key_mode_check_never_reads(self):
        with tempfile.TemporaryDirectory() as d:
            key = Path(d) / "trn2.pem"
            key.write_text("NOT A REAL KEY")
            os.chmod(key, 0o644)
            with self.assertRaises(RemoteError):
                check_key_path(key)
            os.chmod(key, 0o600)
            with mock.patch("builtins.open", side_effect=AssertionError("key must not be opened")):
                check_key_path(key)


class ManifestTests(unittest.TestCase):
    def cfg(self, **kw):
        c = C.LoopConfig(researcher_model_id="a.anthropic.x", verifier_model_id="a.openai.y")
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def build(self, cfg, flags=("--weight-ema",), pid="20260922-test-arm", tier="smoke"):
        label = f"al-{pid}-{tier}-r1"
        return build_manifest(proposal_id=pid, tier=tier, box="trn22", remote_run_dir=f"{C.REMOTE_RUNS_DIR}/{COMMIT[:8]}--{pid}--{label}",
                              label=label, train_flags=list(flags), cfg=cfg, frozen_at="t", control_label=None, abort_conditions=[], description_note="n")

    def test_valid_manifest_has_protocol_shape(self):
        m = self.build(self.cfg())
        self.assertIn("--no-commit", m.argv)
        self.assertEqual(m.argv[m.argv.index("--seed") + 1], "42")
        self.assertEqual((m.lnc, m.nproc_per_node), (2, 4))
        self.assertIn("--optlevel=1", m.env["NEURON_CC_FLAGS"])
        self.assertIn("[trn22]", m.description)
        self.assertLessEqual(m.max_train_seconds, 1800)
        assert_manifest_legal(m)

    def test_rejects_non_full_chip_shape(self):
        with self.assertRaises(ManifestError):
            self.build(self.cfg(lnc=1, nproc_per_node=4))
        with self.assertRaises(ManifestError):
            self.build(self.cfg(lnc=2, nproc_per_node=1))

    def test_rejects_protocol_owned_or_unsafe_flags(self):
        for flags in (["--seed", "7"], ["--out-dir", "x"], ["--init-from", "ckpt"], ["--weight-ema", "0.9;rm"], ["value-first"], ["--weight-ema", "--no-weight-ema"]):
            with self.assertRaises(ManifestError, msg=flags):
                self.build(self.cfg(), flags=flags)

    def test_rejects_missing_cc_flag(self):
        with self.assertRaises(ManifestError):
            self.build(self.cfg(neuron_cc_flags="--auto-cast matmult"))

    def test_tampered_manifest_is_refused(self):
        m = self.build(self.cfg())
        m.argv[m.argv.index("--seed") + 1] = "43"
        with self.assertRaises(ManifestError):
            assert_manifest_legal(m)
        m = self.build(self.cfg())
        m.argv.remove("--no-commit")
        with self.assertRaises(ManifestError):
            assert_manifest_legal(m)
        m = self.build(self.cfg())
        m.commit = "0" * 40
        with self.assertRaises(ManifestError):
            assert_manifest_legal(m)

    def test_known_flags_include_no_forms(self):
        flags = known_train_flags(TRAIN_PY)
        self.assertIn("--weight-ema", flags)
        self.assertIn("--no-weight-ema", flags)
        self.assertIn("--weight-ema-decay", flags)

    def test_config_requires_independent_verifier_family(self):
        cfg = C.LoopConfig(researcher_model_id="us.anthropic.a", verifier_model_id="us.anthropic.b")
        self.assertTrue(any("different model family" in e for e in cfg.validate()))
        self.assertTrue(any("5 GiB" in e for e in C.LoopConfig(researcher_model_id="us.anthropic.a", verifier_model_id="us.openai.b", caps=C.Caps(min_free_gib=2)).validate()))

    def test_config_rejects_unbounded_bedrock_read_timeout(self):
        cfg = C.LoopConfig(
            researcher_model_id="us.anthropic.a",
            verifier_model_id="us.openai.b",
            bedrock_read_timeout_seconds=3601,
        )
        self.assertTrue(any("bedrock_read_timeout_seconds" in e for e in cfg.validate()))


class WatchdogTests(unittest.TestCase):
    @staticmethod
    def result(stdout="", returncode=0, stderr=""):
        return subprocess.CompletedProcess([], returncode, stdout, stderr)

    def test_healthy_service_and_closed_breaker_are_silent(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / "state.json"
            state.write_text('{"breaker": null}')
            with mock.patch.object(W, "STATE_FILE", state), mock.patch.object(W, "run", return_value=self.result("active\n")):
                self.assertIsNone(W.blocker_reason())

    def test_open_breaker_requires_attention(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / "state.json"
            state.write_text('{"breaker": {"reason": "worker gate"}}')
            with mock.patch.object(W, "STATE_FILE", state), mock.patch.object(W, "run", return_value=self.result("active\n")):
                self.assertIn("worker gate", W.blocker_reason())

    def test_existing_blocker_issue_is_deduplicated(self):
        with mock.patch.object(W, "run", return_value=self.result("7\n")) as invoked:
            W.notify("test blocker")
        self.assertEqual(invoked.call_count, 1)


class AnalystTests(unittest.TestCase):
    def row(self, bpb, steps, secs=1790.0, **kw):
        r = {"status": "keep", "over_budget": False, "causality_passed": True, "eval_over_timeout": False, "public_val_bpb": bpb,
             "num_steps": steps, "training_seconds": secs, "world_size": 4, "lnc": 2, "seq_len": 2048, "max_train_seconds": 1800, "git_dirty": True, "_dirt_explained": True}
        r.update(kw)
        return r

    def test_keep_discard_confounded_failed(self):
        caps = C.Caps()
        self.assertEqual(compare(tier="full", candidate=self.row(1.040, 3000), control=self.row(1.050, 3000), candidate_ckpt=True, control_ckpt=True, candidate_rc=0, caps=caps).decision, "KEEP")
        self.assertEqual(compare(tier="full", candidate=self.row(1.049, 3000), control=self.row(1.050, 3000), candidate_ckpt=True, control_ckpt=True, candidate_rc=0, caps=caps).decision, "DISCARD")
        self.assertEqual(compare(tier="full", candidate=self.row(1.000, 2000), control=self.row(1.050, 3000), candidate_ckpt=True, control_ckpt=True, candidate_rc=0, caps=caps).decision, "CONFOUNDED")
        self.assertEqual(compare(tier="full", candidate=self.row(1.000, 3000, causality_passed=False), control=self.row(1.050, 3000), candidate_ckpt=True, control_ckpt=True, candidate_rc=0, caps=caps).decision, "FAILED")
        self.assertEqual(compare(tier="full", candidate=self.row(1.000, 3000), control=self.row(1.050, 3000), candidate_ckpt=False, control_ckpt=True, candidate_rc=0, caps=caps).decision, "FAILED")

    def test_unexplained_post_run_dirt_fails_even_if_runner_reported_clean_source(self):
        caps = C.Caps()
        candidate = self.row(1.000, 3000, git_dirty=False, _dirt_explained=False)
        verdict = compare(
            tier="full",
            candidate=candidate,
            control=self.row(1.050, 3000),
            candidate_ckpt=True,
            control_ckpt=True,
            candidate_rc=0,
            caps=caps,
        )
        self.assertEqual(verdict.decision, "FAILED")
        self.assertIn("post-run worktree dirt", verdict.reasons[0])
        self.assertEqual(compare(tier="smoke", candidate=self.row(1.3, 380, 300.0), control=self.row(1.3, 400, 300.0), candidate_ckpt=True, control_ckpt=True, candidate_rc=0, caps=caps).decision, "DISCARD")
        self.assertEqual(compare(tier="smoke", candidate=self.row(1.3, 396, 300.0), control=self.row(1.3, 400, 300.0), candidate_ckpt=True, control_ckpt=True, candidate_rc=0, caps=caps).decision, "KEEP")

    def test_proposal_smoke_gate_cannot_be_weakened_by_global_tolerance(self):
        caps = C.Caps()
        verdict = compare(
            tier="smoke",
            candidate=self.row(1.167854, 330, 284.8),
            control=self.row(1.165604, 335, 284.5),
            candidate_ckpt=True,
            control_ckpt=True,
            candidate_rc=0,
            caps=caps,
            smoke_gate={"min_candidate_steps": 336, "min_step_rate_ratio": 1.0},
        )
        self.assertEqual(verdict.decision, "DISCARD")
        self.assertIn("proposal minimum 336", verdict.reasons[0])

    def test_proposal_smoke_gate_can_require_bpb_direction(self):
        caps = C.Caps()
        verdict = compare(
            tier="smoke",
            candidate=self.row(1.052, 410, 300.0),
            control=self.row(1.050, 400, 300.0),
            candidate_ckpt=True,
            control_ckpt=True,
            candidate_rc=0,
            caps=caps,
            smoke_gate={"max_bpb_delta": 0.0},
        )
        self.assertEqual(verdict.decision, "DISCARD")
        self.assertIn("exceeds proposal maximum", verdict.reasons[0])

    def test_parse_json_block(self):
        self.assertEqual(parse_json_block('text ```json\n{"a": 1}\n``` more')["a"], 1)
        self.assertEqual(parse_json_block('{"a": 2}')["a"], 2)
        with self.assertRaises(ModelError):
            parse_json_block("no json here")


class BedrockTransportTests(unittest.TestCase):
    def test_streams_text_and_collects_terminal_metadata(self):
        class FakeClient:
            def converse_stream(self, **kwargs):
                self.kwargs = kwargs
                return {"stream": iter([
                    {"messageStart": {"role": "assistant"}},
                    {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "{\"proposal"}}},
                    {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "_id\": \"p1\"}"}}},
                    {"messageStop": {"stopReason": "end_turn"}},
                    {"metadata": {"usage": {"inputTokens": 1200, "outputTokens": 42}}},
                ])}

        bedrock = BedrockClient.__new__(BedrockClient)
        bedrock._client = FakeClient()
        reply = bedrock.complete(
            role="researcher", model_id="us.anthropic.claude-fable-5-1",
            system="system", user="evidence", max_tokens=4096,
        )

        self.assertEqual(reply.text, '{"proposal_id": "p1"}')
        self.assertEqual(reply.input_tokens, 1200)
        self.assertEqual(reply.output_tokens, 42)
        self.assertEqual(reply.stop_reason, "end_turn")
        self.assertEqual(bedrock._client.kwargs["inferenceConfig"], {"maxTokens": 4096})

    def test_stream_error_event_is_a_model_error(self):
        class FakeClient:
            def converse_stream(self, **_kwargs):
                return {"stream": iter([
                    {"serviceUnavailableException": {"message": "retry later"}},
                ])}

        bedrock = BedrockClient.__new__(BedrockClient)
        bedrock._client = FakeClient()
        expected = "researcher call failed: serviceUnavailableException: retry later"
        with self.assertRaisesRegex(ModelError, expected):
            bedrock.complete(
                role="researcher", model_id="us.anthropic.claude-fable-5-1",
                system="system", user="evidence", max_tokens=4096,
            )

    def test_premature_stream_end_is_a_model_error(self):
        from urllib3.exceptions import ProtocolError

        class FakeClient:
            calls = 0

            def converse_stream(self, **_kwargs):
                self.calls += 1

                def events():
                    yield {"messageStart": {"role": "assistant"}}
                    raise ProtocolError("Response ended prematurely")

                return {"stream": events()}

        bedrock = BedrockClient.__new__(BedrockClient)
        bedrock._client = FakeClient()
        with self.assertRaisesRegex(ModelError, "researcher call failed after stream retry"):
            bedrock.complete(
                role="researcher", model_id="us.anthropic.claude-fable-5-1",
                system="system", user="evidence", max_tokens=4096,
            )
        self.assertEqual(bedrock._client.calls, 2)

    def test_premature_stream_end_retries_without_partial_text(self):
        from urllib3.exceptions import ProtocolError

        class FakeClient:
            calls = 0

            def converse_stream(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    def truncated():
                        yield {"contentBlockDelta": {"delta": {"text": "discard-me"}}}
                        raise ProtocolError("Response ended prematurely")

                    return {"stream": truncated()}
                return {"stream": iter([
                    {"contentBlockDelta": {"delta": {"text": "{\"proposal_id\":\"p2\"}"}}},
                    {"messageStop": {"stopReason": "end_turn"}},
                    {"metadata": {"usage": {"inputTokens": 50, "outputTokens": 10}}},
                ])}

        bedrock = BedrockClient.__new__(BedrockClient)
        bedrock._client = FakeClient()
        reply = bedrock.complete(
            role="researcher", model_id="us.anthropic.claude-fable-5-1",
            system="system", user="evidence", max_tokens=4096,
        )
        self.assertEqual(reply.text, '{"proposal_id":"p2"}')
        self.assertNotIn("discard-me", reply.text)
        self.assertEqual(bedrock._client.calls, 2)

    def test_stream_read_timeout_retries(self):
        from urllib3.exceptions import ReadTimeoutError

        class FakeClient:
            calls = 0

            def converse_stream(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    def timed_out():
                        raise ReadTimeoutError(None, None, "Read timed out")
                        yield  # pragma: no cover - makes this a lazy stream generator

                    return {"stream": timed_out()}
                return {"stream": iter([
                    {"contentBlockDelta": {"delta": {"text": "ok"}}},
                    {"messageStop": {"stopReason": "end_turn"}},
                    {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 1}}},
                ])}

        bedrock = BedrockClient.__new__(BedrockClient)
        bedrock._client = FakeClient()
        reply = bedrock.complete(
            role="researcher", model_id="us.anthropic.claude-fable-5-1",
            system="system", user="evidence", max_tokens=4096,
        )
        self.assertEqual(reply.text, "ok")
        self.assertEqual(bedrock._client.calls, 2)


class ImplementerTests(unittest.TestCase):
    def test_isolated_code_change_requires_independent_pass_and_commits_only_train_py(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "autoloop@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Autoloop Test"], check=True)
            (root / "train.py").write_text("def load_for_eval(path, device):\n    return None\n\nWIDTH = 640\n")
            subprocess.run(["git", "-C", str(root), "add", "train.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
            base = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            agent = root / "research" / "agentic"
            experiment = agent / "experiments" / "20260922-code-width"
            runtime = agent / "runtime"
            model = FakeModel()
            model.replies["implementer"] = [json.dumps({
                "summary": "change width",
                "replacements": [{"old": "WIDTH = 640", "new": "WIDTH = 672"}],
                "notes": [],
            })]
            model.replies["implementation_verifier"] = ['{"verdict":"PASS","findings":[]}']
            prop = Proposal.from_model_json({
                "proposal_id": "20260922-code-width",
                "hypothesis": "A slightly wider model should lower validation BPB at comparable trained tokens.",
                "single_causal_change": "change only the model width constant from 640 to 672",
                "flags": [],
                "evidence": ["research/experiment-log.jsonl"],
                "duplicate_search_terms": ["width 672"],
                "expected_effect": "lower BPB by at least 0.002",
                "change_type": "code",
                "implementation_plan": "Replace only the unique WIDTH constant in train.py and preserve every other setting.",
                "files_allowed": ["train.py"],
            }, "us.anthropic.claude-fable-5-1", "now")
            cfg = C.LoopConfig(
                researcher_model_id="us.anthropic.claude-fable-5-1",
                implementer_model_id="us.anthropic.claude-fable-5-1",
                verifier_model_id="us.openai.gpt-6-astra",
            )
            used = []
            with mock.patch.object(C, "ROOT", root), mock.patch.object(C, "RUNTIME_DIR", runtime), mock.patch.object(C, "AUTHORITATIVE_COMMIT", base):
                result = implement_code_proposal(
                    prop, experiment_dir=experiment, cfg=cfg, client=model, now="now",
                    can_call=lambda: (True, ""), record_usage=lambda role, reply: used.append(role),
                )
            self.assertTrue(result.passed)
            self.assertEqual(used, ["implementer", "implementation_verifier"])
            changed = subprocess.check_output(
                ["git", "-C", str(root), "diff-tree", "--no-commit-id", "--name-only", "-r", result.candidate_commit], text=True
            ).splitlines()
            self.assertEqual(changed, ["train.py"])
            self.assertIn("WIDTH = 672", subprocess.check_output(
                ["git", "-C", str(root), "show", f"{result.candidate_commit}:train.py"], text=True
            ))
            self.assertTrue((experiment / "implementation.patch").is_file())
            self.assertFalse((runtime / "worktrees" / prop.proposal_id).exists())
            verifier_user = model.users["implementation_verifier"][0]
            self.assertIn("COMPLETE CANDIDATE train.py", verifier_user)
            self.assertIn("WIDTH = 672", verifier_user)
            self.assertIn("do not require smoke", model.systems["implementation_verifier"][0])
            self.assertIn("named pre-edit artifact", model.systems["implementation_verifier"][0])


# ---------------------------------------------------------------------------------------------
# end-to-end with fakes
# ---------------------------------------------------------------------------------------------


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.env = Env()

    def tearDown(self):
        self.env.close()

    def run_controls(self, ctl):
        env = self.env
        rep = ctl.cycle()
        self.assertIsNone(rep.stopped_reason, rep.actions)
        self.assertEqual(len(env.h.launched), 1)
        finish_run(env.h, env.h.launched[0], bpb=1.30, steps=400, training_seconds=300.0)
        env.advance(600)
        rep = ctl.cycle()
        self.assertIsNone(rep.stopped_reason, rep.actions)
        self.assertEqual(len(env.h.launched), 2, rep.actions)
        finish_run(env.h, env.h.launched[1], bpb=1.050, steps=3000, training_seconds=1790.0)
        env.advance(2400)

    def test_full_ladder_control_then_candidate(self):
        env = self.env
        ctl = env.controller()
        self.run_controls(ctl)
        env.model.replies["researcher"].append(researcher_json())
        env.model.replies["verifier"].append('{"verdict": "PASS", "findings": ["single flag, evidence cited"]}')
        rep = ctl.cycle()  # collects control full, closes it, proposes, verifies, launches candidate smoke
        self.assertIsNone(rep.stopped_reason, rep.actions)
        self.assertEqual(len(env.h.launched), 3, rep.actions)
        smoke = env.h.launched[2]
        self.assertIn("--weight-ema", smoke["argv"])
        self.assertEqual(smoke["argv"][smoke["argv"].index("--seed") + 1], "42")
        self.assertIn("--no-commit", smoke["argv"])
        self.assertEqual(smoke["argv"][smoke["argv"].index("--nproc-per-node") + 1], "4")
        self.assertEqual(smoke["argv"][smoke["argv"].index("--lnc") + 1], "2")
        self.assertEqual(smoke["env"]["NEURON_CC_FLAGS"], "--optlevel=1")
        pdir = C.EXPERIMENTS_DIR / "20260922-weight-ema-on"
        self.assertTrue((pdir / "proposal.json").is_file())
        self.assertTrue(json.loads((pdir / "verification.json").read_text())["passed"])
        manifest = json.loads((pdir / "smoke-manifest.json").read_text())
        self.assertTrue(manifest["control_label"].startswith("al-20260922-control-trn22-lnc2x4"))
        finish_run(env.h, smoke, bpb=1.29, steps=398, training_seconds=300.0)
        env.advance(600)
        rep = ctl.cycle()
        self.assertEqual(len(env.h.launched), 4, rep.actions)
        full = env.h.launched[3]
        self.assertEqual(int(full["argv"][full["argv"].index("--max-train-seconds") + 1]), 1800)
        finish_run(env.h, full, bpb=1.045, steps=3010, training_seconds=1790.0)
        env.advance(2400)
        rep = ctl.cycle()
        self.assertIsNone(rep.stopped_reason, rep.actions)
        state = json.loads(C.STATE_FILE.read_text())
        self.assertEqual(state["proposals"]["20260922-weight-ema-on"]["stage"], "closed")
        self.assertEqual(state["proposals"]["20260922-weight-ema-on"]["verdict"], "KEEP")
        self.assertIn("20260922-weight-ema-on", C.DECISION_LOG.read_text())
        self.assertTrue((pdir / "artifacts" / "full" / "experiment-log.jsonl").is_file())
        full_result = json.loads((pdir / "full-result.json").read_text())
        self.assertTrue(full_result["collected"]["post_run_dirt_explained"])
        self.assertIn("run_launch_receipt", full_result["collected"]["files"])
        self.assertTrue(Path(full_result["collected"]["files"]["run_launch_receipt"]["local"]).is_file())
        self.assertTrue((pdir / "full-verdict.md").is_file())
        # every run dir was new and distinct; nothing touched wave195; every launch archived before the next
        cwds = [s["cwd"] for s in env.h.launched]
        self.assertEqual(len(set(cwds)), 4)
        self.assertTrue(all(c.startswith(C.REMOTE_RUNS_DIR + "/") for c in cwds))
        self.assertFalse(any("wave195" in " ".join(a) for _, a in env.transport.calls))
        self.assertGreaterEqual(len(env.archiver.commits), 4)
        self.assertEqual(len({s["argv"][s["argv"].index("--label") + 1] for s in env.h.launched}), 4)

    def test_restart_does_not_relaunch(self):
        env = self.env
        ctl = env.controller()
        ctl.cycle()
        self.assertEqual(len(env.h.launched), 1)
        ctl2 = env.controller()  # fresh Store from the same state file
        rep = ctl2.cycle()
        self.assertEqual(len(env.h.launched), 1, rep.actions)
        self.assertTrue(any("running" in a for a in rep.actions), rep.actions)
        finish_run(env.h, env.h.launched[0], bpb=1.3, steps=400, training_seconds=300.0)
        ctl3 = env.controller()
        rep = ctl3.cycle()
        self.assertEqual(len(env.h.launched), 2, rep.actions)

    def test_ready_work_on_later_box_precedes_new_research_on_first_box(self):
        env = Env(boxes=("trn22", "trn21"))
        self.addCleanup(env.close)
        ctl = env.controller()
        proposal = json.loads(researcher_json("20260922-ready-on-trn21"))
        ctl.store.add_proposal(proposal)
        ctl.store.set_stage(proposal["proposal_id"], "verified", box="trn21")
        advanced: list[str] = []

        def record_first_box(box, report):
            advanced.append(box)
            return True

        with (
            mock.patch.object(ctl, "_reconcile", return_value=set()),
            mock.patch.object(ctl, "_advance_box", side_effect=record_first_box),
        ):
            rep = ctl.cycle()
        self.assertIsNone(rep.stopped_reason)
        self.assertEqual(advanced, ["trn21"])
        self.assertNotIn("researcher", env.model.calls)

    def test_archived_smoke_checkpoint_is_hash_verified_then_pruned(self):
        env = self.env
        ctl = env.controller()
        ctl.cycle()
        spec = env.h.launched[0]
        finish_run(env.h, spec, bpb=1.3, steps=400, training_seconds=300.0)
        checkpoint = f"{spec['cwd']}/out/{spec['argv'][spec['argv'].index('--label') + 1]}/final.pt"
        self.assertIn(checkpoint, env.h.files)
        env.advance(600)
        rep = ctl.cycle()
        self.assertNotIn(checkpoint, env.h.files)
        pid = next(iter(ctl.store.proposals()))
        audit = ctl.store.read_artifact(pid, "smoke-checkpoint-pruned.json")
        self.assertEqual(audit["sha256"], sha(b"\x00" * 64))
        self.assertTrue(any("checkpoint hash-verified and pruned" in action for action in rep.actions))

    def test_stop_file_halts_before_any_remote_call(self):
        env = self.env
        C.STOP_FILE.write_text("stop")
        rep = env.controller().cycle()
        self.assertEqual(rep.stopped_reason, "STOP file present")
        self.assertEqual(env.transport.calls, [])

    def test_same_family_verifier_blocks_loop(self):
        env = self.env
        env.cfg.verifier_model_id = "us.anthropic.claude-other"
        rep = env.controller().cycle()
        self.assertIn("config invalid", rep.stopped_reason)
        self.assertEqual(env.transport.calls, [])

    def test_verifier_fail_closes_branch_but_loop_continues(self):
        env = self.env
        ctl = env.controller()
        self.run_controls(ctl)
        env.model.replies["researcher"] += [researcher_json("20260922-bad-idea"), researcher_json("20260922-good-idea")]
        env.model.replies["verifier"] += ['{"verdict": "FAIL", "findings": ["duplicate of closed direction"]}', '{"verdict": "PASS", "findings": []}']
        rep = ctl.cycle()
        self.assertEqual(len(env.h.launched), 2, rep.actions)
        state = json.loads(C.STATE_FILE.read_text())
        self.assertEqual(state["proposals"]["20260922-bad-idea"]["stage"], "rejected")
        feedback = ctl._recent_rejection_feedback()
        self.assertIn("20260922-bad-idea", feedback)
        self.assertIn("duplicate of closed direction", feedback)
        rep = ctl.cycle()
        self.assertEqual(len(env.h.launched), 3, rep.actions)
        self.assertIn("20260922-good-idea", env.h.launched[2]["cwd"])

    def test_fable_researcher_has_reasoning_headroom(self):
        env = self.env
        ctl = env.controller()
        self.run_controls(ctl)
        env.model.replies["researcher"].append(researcher_json())
        env.model.replies["verifier"].append('{"verdict":"PASS","findings":[]}')
        ctl.cycle()
        researcher_call = env.model.calls.index("researcher")
        self.assertEqual(env.model.max_tokens[researcher_call], 32000)
        self.assertIn("Do not require named pre-edit", env.model.systems["researcher"][0])
        self.assertIn("Authoritative measured/closed train.py evidence excerpts", env.model.users["researcher"][0])

    def test_researcher_source_index_exposes_closed_flag_evidence_and_is_bounded(self):
        ctl = self.env.controller()
        source = "\n".join([
            "# mechanism details",
            "# MEASURED 2026-09-15: EXACT NULL, stays OFF.",
            "USE_CAUTIOUS_WD = False",
            *[f"line_{i} = {i}" for i in range(100)],
        ])
        evidence = ctl._research_train_evidence_index(source, max_chars=1000)
        self.assertIn("EXACT NULL", evidence)
        self.assertIn("USE_CAUTIOUS_WD = False", evidence)
        self.assertLessEqual(len(evidence), 1000)

    def test_research_ledger_view_is_recent_and_bounded(self):
        env = self.env
        rows = []
        for i in range(100):
            rows.append(json.dumps({
                "timestamp": f"2026-09-22T{i // 60:02d}:{i % 60:02d}:00+00:00",
                "label": f"candidate-{i:03d}",
                "status": "ok",
                "public_val_bpb": 1.2 - i / 10000,
                "num_steps": 1000 + i,
                "launch_command": "torchrun train.py " + "x" * 5000,
                "description": "d" * 5000,
            }))
        C.LOCAL_LEDGER.write_text("\n".join(rows) + "\n")

        summary = env.controller()._ledger_summary(200)
        lines = summary.splitlines()
        self.assertEqual(len(lines), 60)
        self.assertLess(len(summary), 60000)
        self.assertIn("candidate-099", summary)
        self.assertNotIn("candidate-039", summary)

    def test_relevant_train_evidence_contains_flag_definition(self):
        ctl = self.env.controller()
        prop = Proposal.from_model_json(json.loads(researcher_json()), self.env.cfg.researcher_model_id, "now")
        evidence = ctl._train_evidence(prop, TRAIN_PY)
        self.assertIn('parser.add_argument("--weight-ema"', evidence)

    def test_proposal_verifier_receives_complete_authoritative_train_source(self):
        env = self.env
        ctl = env.controller()
        self.run_controls(ctl)
        env.model.replies["researcher"].append(researcher_json("20260922-source-grounded"))
        env.model.replies["verifier"].append('{"verdict":"PASS","findings":[]}')
        ctl.cycle()
        verifier_user = env.model.users["verifier"][0]
        self.assertIn("COMPLETE AUTHORITATIVE train.py", verifier_user)
        self.assertIn(TRAIN_PY, verifier_user)
        self.assertIn("pre-edit artifact", env.model.systems["verifier"][0])

    def test_deterministic_duplicate_rejected_without_llm_verifier(self):
        env = self.env
        ctl = env.controller()
        self.run_controls(ctl)
        C.LOCAL_LEDGER.write_text(json.dumps({"timestamp": "2026-09-20T00:00:00+00:00", "label": "prior-ema", "git_commit": COMMIT[:8], "status": "keep",
                                              "launch_command": "torchrun train.py --weight-ema", "public_val_bpb": 1.05}) + "\n")
        env.model.replies["researcher"].append(researcher_json("20260922-dup-ema"))
        env.model.replies["verifier"].append('{"verdict": "PASS", "findings": []}')
        rep = ctl.cycle()
        self.assertEqual(len(env.h.launched), 2, rep.actions)
        ver = json.loads((C.EXPERIMENTS_DIR / "20260922-dup-ema" / "verification.json").read_text())
        self.assertFalse(ver["passed"])
        self.assertTrue(any("identical flag set" in f for f in ver["deterministic_findings"]))
        self.assertNotIn("verifier", env.model.calls)  # deterministic failure short-circuits the LLM call

    def test_unknown_flag_rejected(self):
        env = self.env
        ctl = env.controller()
        self.run_controls(ctl)
        env.model.replies["researcher"].append(researcher_json("20260922-nope", flags=("--does-not-exist",)))
        rep = ctl.cycle()
        self.assertEqual(len(env.h.launched), 2, rep.actions)
        self.assertFalse(json.loads((C.EXPERIMENTS_DIR / "20260922-nope" / "verification.json").read_text())["passed"])

    def test_staging_fails_closed_on_head_mismatch(self):
        env = self.env
        env.h.checkout_override = "f" * 40
        rep = env.controller().cycle()
        self.assertEqual(len(env.h.launched), 0)
        self.assertIn("RemoteError", rep.stopped_reason)
        state = json.loads(C.STATE_FILE.read_text())
        self.assertEqual(state["breaker"]["reason"], "RemoteError")

    def test_busy_or_low_disk_box_never_launches(self):
        env = self.env
        env.h.procs[999] = "torchrun --standalone train.py"
        rep = env.controller().cycle()
        self.assertEqual(len(env.h.launched), 0)
        self.assertTrue(any("not clean/idle" in a for a in rep.actions), rep.actions)
        env.h.procs.clear()
        env.h.free_kb = 3 * 1024 * 1024
        ctl = env.controller()
        rep = ctl.cycle()
        self.assertEqual(len(env.h.launched), 0)
        self.assertTrue(json.loads(C.STATE_FILE.read_text())["breaker"])  # second consecutive infra failure

    def test_run_caps_enforced(self):
        env = self.env
        env.cfg.caps.max_runs_total = 1
        ctl = env.controller()
        ctl.cycle()
        finish_run(env.h, env.h.launched[0], bpb=1.3, steps=400, training_seconds=300.0)
        env.advance(600)
        rep = ctl.cycle()
        self.assertEqual(len(env.h.launched), 1)
        self.assertTrue(any("cap reached" in a for a in rep.actions), rep.actions)

    def test_llm_caps_enforced(self):
        env = self.env
        env.cfg.caps.max_llm_calls_per_day = 0
        ctl = env.controller()
        self.run_controls(ctl)
        env.model.replies["researcher"].append(researcher_json())
        rep = ctl.cycle()
        self.assertEqual(env.model.calls, [])
        self.assertTrue(any("cap reached" in a for a in rep.actions), rep.actions)

    def test_code_proposal_defers_until_implementer_and_verifier_slots_are_available(self):
        env = self.env
        env.cfg.caps.max_llm_calls_per_cycle = 2
        ctl = env.controller()
        self.run_controls(ctl)
        pid = "20260922-bf16-head"
        env.model.replies["researcher"].append(code_researcher_json(pid))
        env.model.replies["verifier"].append('{"verdict":"PASS","findings":[]}')

        with mock.patch.object(ctl, "_implement", wraps=ctl._implement) as implementation:
            rep = ctl.cycle()
            implementation.assert_not_called()
        self.assertEqual(ctl.store.proposals()[pid]["stage"], "proposal_verified")
        self.assertFalse((C.EXPERIMENTS_DIR / pid / "implementation.json").exists())
        self.assertTrue(any("deferred; requires two agent calls" in a for a in rep.actions), rep.actions)
        self.assertEqual(env.model.calls[-2:], ["researcher", "verifier"])

        def complete_implementation(prop, report):
            for role in ("implementer", "implementation_verifier"):
                ctl._record_usage(role, ModelReply("{}", 100, 20, "end_turn", env.cfg.verifier_model_id))
            ctl.store.set_stage(prop.proposal_id, "implemented", candidate_commit=COMMIT)
            report.add(f"implement: {prop.proposal_id} PASS commit={COMMIT[:12]}")
            return True

        def launch_after_implementation(proposal_id, tier, box, flags, report):
            ctl.store.set_stage(proposal_id, f"launched:{tier}", box=box)
            report.add(f"{box}: test launch {proposal_id}/{tier}")
            return True

        with (
            mock.patch.object(ctl, "_implement", side_effect=complete_implementation) as implementation,
            mock.patch.object(ctl, "_stage_and_launch", side_effect=launch_after_implementation) as launch,
        ):
            rep = ctl.cycle()
            implementation.assert_called_once()
            launch.assert_called_once()
        self.assertEqual(env.model.calls[-2:], ["researcher", "verifier"])
        self.assertEqual(ctl._cycle_llm_calls, 2)
        self.assertIsNone(rep.stopped_reason, (rep.actions, rep.stopped_reason))
        self.assertEqual(ctl.store.proposals()[pid]["stage"], "launched:smoke", rep.actions)
        self.assertEqual(len(env.h.launched), 2, rep.actions)

    def test_failed_candidate_run_closes_branch(self):
        env = self.env
        ctl = env.controller()
        self.run_controls(ctl)
        env.model.replies["researcher"].append(researcher_json())
        env.model.replies["verifier"].append('{"verdict": "PASS", "findings": []}')
        ctl.cycle()
        finish_run(env.h, env.h.launched[2], bpb=float("nan"), steps=0, training_seconds=1.0, rc=1, status="crash", checkpoint=False)
        env.advance(600)
        rep = ctl.cycle()
        state = json.loads(C.STATE_FILE.read_text())
        self.assertEqual(state["proposals"]["20260922-weight-ema-on"]["verdict"], "FAILED")
        self.assertEqual(state["proposals"]["20260922-weight-ema-on"]["stage"], "closed")
        self.assertEqual(len(env.h.launched), 3, rep.actions)  # no relaunch of a crashed label

    def test_archive_outside_agentic_is_refused(self):
        from autoloop.archive import ArchiveError, GitArchiver

        calls = []
        ga = GitArchiver(root=self.env.tmp, runner=lambda argv: (calls.append(argv), (0, "", ""))[1])
        with self.assertRaises(ArchiveError):
            ga.commit([self.env.tmp / "train.py"], "x")
        with self.assertRaises(ArchiveError):
            ga.commit([self.env.agent / "trn2.pem"], "x")
        self.assertEqual(calls, [])

    def test_cutoff_stops_loop(self):
        env = self.env
        env.now = C.COMPETITION_CUTOFF + timedelta(minutes=1)
        rep = env.controller().cycle()
        self.assertEqual(rep.stopped_reason, "competition cutoff reached")


if __name__ == "__main__":
    unittest.main()
