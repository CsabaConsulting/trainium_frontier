"""Local fake-process tests. No host contact, Neuron access, or model execution."""
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import host_queue as q


class FakeRunner(q.QueueRunner):
    def wait_idle(self, max_wait=None, *, starting=False):
        self.guards(starting=starting)
        return {"processes": [], "device_holders": []}

    def environment(self, job, directory, cache):
        return {"PATH": "/usr/bin:/bin"}

    def commands(self, job, directory):
        train = """import pathlib,time
p=pathlib.Path('out');p.mkdir();(p/'final.pt').write_bytes(b'checkpoint')
print('FAKE_STARTED',flush=True)
time.sleep(SLEEP)
print('step 0 | loss 2.5 | dt 0.5s | epoch 1')
print('---\\ntraining_seconds: 1790.0\\ncharged_seconds_est: 1791.0\\nstartup_allowance_est: 23.0\\nnum_steps: 3500\\nworld_size: 4\\nseq_len: 1024\\ntotal_tokens_M: 458.7\\nover_budget: False')
""".replace("SLEEP", str(self.cfg.get("fake_sleep", 0)))
        if job.get("fake_fail_train"):
            train += "raise SystemExit(17)\n"
        result = {"val_bpb": 0.990, "num_tokens": job.get("fake_eval_tokens", 20971520), "eval_over_timeout": False}
        return {"training": [sys.executable, "-c", train],
                "fresh-public20m": [sys.executable, "-c", "print(" + repr(json.dumps(result)) + ")"]}


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="trainium-host-queue-test-")
        self.root = Path(self.tmp.name)
        source = self.root / "source.py"
        prep = self.root / "prepare.py"
        source.write_text("# fake train source\n")
        prep.write_text("# fake prepare source\n")
        self.source_patch = patch.object(q, "SOURCE_SHA", q.sha(source))
        self.prep_patch = patch.object(q, "PREPARE_SHA", q.sha(prep))
        self.source_patch.start()
        self.prep_patch.start()
        self.cfg = {"schema": 1, "queue_id": "test", "host": "trn21", "armed": True,
                    "expected_hostname": socket.gethostname(), "freeze_utc": "2099-01-01T00:00:00+00:00",
                    "source": {"path": str(source), "sha256": q.SOURCE_SHA},
                    "prepare": {"path": str(prep), "sha256": q.PREPARE_SHA},
                    "python": sys.executable, "torchrun": sys.executable,
                    "train_cache": str(self.root / "cache"), "eval_cache": str(self.root / "eval-cache"),
                    "run_root": str(self.root / "runs"), "device_lock": str(self.root / "device.lock"),
                    "stop_paths": [str(self.root / "STOP")], "jobs": []}
        self.add_job("job-1")
        self.add_job("job-2")

    def tearDown(self):
        self.source_patch.stop()
        self.prep_patch.stop()
        self.tmp.cleanup()

    def add_job(self, label):
        job = {"label": label, "argv": ["--no-eval-public", "--seed", "42", "--max-train-seconds", "1800",
                                        "--num-steps", "1000000", "--out-dir", "out", "--seq-len", "1024"],
               "kind": "flag_arm", "arm_id": "F1", "base_sha256": q.SOURCE_SHA,
               "base_science_argv": ["--matrix-lr", "0.015"], "readiness": "READY_PENDING_HOST_PREFLIGHT", "armed": True,
               "same_host_control": {"run_label": "known-control", "public_val_bpb": 0.9915, "steps": 3600}}
        binding = {k: job[k] for k in ("label", "argv", "kind", "arm_id", "base_sha256")}
        binding.update(host=self.cfg["host"], source_sha256=q.SOURCE_SHA)
        path = self.root / (label + "-proposal.json")
        path.write_text(json.dumps({"status": "READY_PENDING_HOST_PREFLIGHT", "queue_binding": binding}))
        job["proposal"] = {"path": str(path), "sha256": q.sha(path)}
        self.cfg["jobs"].append(job)

    def runner(self):
        q.validate_manifest(self.cfg, run=True)
        return FakeRunner(self.cfg, q.fingerprint(self.cfg))

    def terminal(self, label):
        return json.loads((Path(self.cfg["run_root"]) / label / "terminal.json").read_text())

    def test_two_jobs_then_restart_does_not_duplicate(self):
        runner = self.runner()
        self.assertEqual(runner.run(), "queue_complete")
        first = self.terminal("job-1")
        second = self.terminal("job-2")
        self.assertEqual(first["status"], "evaluated_pending_review")
        self.assertLessEqual(first["utc"], second["started_utc"])
        self.assertEqual(runner.run(), "queue_complete")
        lines = (runner.root / "results.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(self.terminal("job-1"), first)

    def test_training_failure_preserves_error_and_does_not_eval_or_advance(self):
        self.cfg["jobs"][0]["fake_fail_train"] = True
        runner = self.runner()
        with self.assertRaisesRegex(RuntimeError, "TRAINING_EXIT_17"):
            runner.run()
        self.assertEqual(self.terminal("job-1")["error"], "TRAINING_EXIT_17")
        self.assertFalse((runner.root / "job-1/fresh-public20m-launch.json").exists())
        self.assertFalse((runner.root / "job-2").exists())
        with self.assertRaisesRegex(RuntimeError, "PREVIOUS_JOB_FAILED"):
            runner.run()

    def test_incomplete_eval_is_terminal_failure(self):
        self.cfg["jobs"][0]["fake_eval_tokens"] = 1024
        runner = self.runner()
        with self.assertRaisesRegex(RuntimeError, "EVAL_TOKEN_OR_TIMEOUT_FAILURE"):
            runner.run()
        self.assertEqual(self.terminal("job-1")["status"], "failed")
        self.assertFalse((runner.root / "job-2").exists())

    def test_stop_and_changed_bytes_prevent_launch(self):
        runner = self.runner()
        Path(self.cfg["stop_paths"][0]).touch()
        with self.assertRaisesRegex(RuntimeError, "STOP"):
            runner.run()
        Path(self.cfg["stop_paths"][0]).unlink()
        Path(self.cfg["source"]["path"]).write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "HASH_MISMATCH"):
            runner.run()
        self.assertFalse((runner.root / "job-1").exists())

    def test_unarmed_or_mismatched_proposal_prevents_launch(self):
        self.cfg["armed"] = False
        with self.assertRaisesRegex(RuntimeError, "QUEUE_UNARMED"):
            self.runner()
        self.cfg["armed"] = True
        p = Path(self.cfg["jobs"][0]["proposal"]["path"])
        p.write_text(json.dumps({"status": "READY_PENDING_HOST_PREFLIGHT", "queue_binding": {}}))
        self.cfg["jobs"][0]["proposal"]["sha256"] = q.sha(p)
        with self.assertRaisesRegex(RuntimeError, "PROPOSAL_DOES_NOT_BIND_EXACT_JOB"):
            self.runner().run()

    def test_pause_between_jobs_and_manifest_tail_replacement(self):
        runner = self.runner()
        runner.root.mkdir()
        (runner.root / "PAUSE_NEXT").touch()
        self.assertEqual(runner.run(), "paused_between_jobs")
        self.assertFalse((runner.root / "job-1").exists())

    def test_crossing_freeze_finishes_eval_but_does_not_start_next_job(self):
        self.cfg["fake_sleep"] = 0.5
        self.cfg["freeze_utc"] = (q.dt.datetime.now(q.dt.timezone.utc) + q.dt.timedelta(seconds=0.2)).isoformat()
        runner = self.runner()
        with self.assertRaisesRegex(RuntimeError, "RESEARCH_FREEZE"):
            runner.run()
        self.assertTrue((runner.root / "job-1/training-exit.json").exists())
        self.assertTrue((runner.root / "job-1/fresh-public20m-exit.json").exists())
        self.assertEqual(self.terminal("job-1")["status"], "evaluated_pending_review")
        self.assertFalse((runner.root / "job-2").exists())

    def test_killed_driver_never_relaunches_or_overlaps_surviving_phase(self):
        self.cfg["fake_sleep"] = 0.5
        runner = self.runner()
        process = multiprocessing.get_context("fork").Process(target=runner.run)
        process.start()
        receipt = runner.root / "job-1/training-process.json"
        deadline = time.monotonic() + 5
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(receipt.exists())
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)
        with open(self.cfg["device_lock"], "a") as lock:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        time.sleep(0.7)
        with self.assertRaisesRegex(RuntimeError, "STARTED_LABEL_WITHOUT_TERMINAL_RECEIPT"):
            runner.run()
        self.assertEqual(self.terminal("job-1")["status"], "interrupted")
        self.assertFalse((runner.root / "job-2").exists())
        log = (runner.root / "job-1/training.log").read_text()
        self.assertEqual(log.count("FAKE_STARTED"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
