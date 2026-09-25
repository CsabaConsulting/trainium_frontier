#!/usr/bin/env python3
"""Host-resident serial execution of a hash-pinned, explicitly armed flag-only queue.

Commands/environment are the proven 20260925 trn21 A1 manual driver. This adds
serialization and durable consumed labels; it makes no scientific decisions.
No SSH, downloads, cleanup, evaluator edits, arbitrary shell or automatic retry.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

SOURCE_SHA = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREPARE_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
CACHE_KEYS = {
    "TORCH_NEURONX_NEFF_CACHE_DIR": "neff", "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": "local",
    "TORCH_NEURONX_HLO_CACHE_DIR": "hlo", "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "NEURON_COMPILE_CACHE_DIR": "compiler", "TORCH_NEURONX_DEBUG_DIR": "debug",
}
LABEL_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159}\Z")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exclusive_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_json(path, value):
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def absolute(value):
    require(isinstance(value, str) and Path(value).is_absolute(), "PATH_MUST_BE_ABSOLUTE")
    return Path(value)


def checked_file(spec):
    path = absolute(spec["path"])
    require(path.is_file() and not path.is_symlink(), "MISSING_OR_SYMLINK_FILE: " + str(path))
    require(sha(path) == spec["sha256"], "HASH_MISMATCH: " + str(path))
    return path


def flag_value(argv, flag):
    values = []
    for i, token in enumerate(argv):
        if token == flag:
            require(i + 1 < len(argv), "FLAG_MISSING_VALUE: " + flag)
            values.append(argv[i + 1])
        elif token.startswith(flag + "="):
            values.append(token.split("=", 1)[1])
    require(len(values) == 1, "FLAG_MISSING_OR_DUPLICATE: " + flag)
    return values[0]


def validate_manifest(cfg, run=False):
    require(cfg.get("schema") == 1, "MANIFEST_SCHEMA")
    require(cfg.get("host") in ("trn21", "trn22"), "HOST_ALIAS")
    require(LABEL_RE.fullmatch(cfg.get("queue_id", "")), "QUEUE_ID")
    require(cfg["source"]["sha256"] == SOURCE_SHA, "ONLY_VERIFIED_C5C_SOURCE_SUPPORTED")
    require(cfg["prepare"]["sha256"] == PREPARE_SHA, "PREPARE_PIN")
    require(bool(cfg.get("expected_hostname")), "EXPECTED_HOSTNAME_REQUIRED")
    require(dt.datetime.fromisoformat(cfg["freeze_utc"]).tzinfo is not None, "FREEZE_TIMEZONE_REQUIRED")
    for key in ("python", "torchrun", "train_cache", "eval_cache", "run_root", "device_lock"):
        absolute(cfg[key])
    require(len(Path(cfg["run_root"]).parts) >= 4, "RUN_ROOT_TOO_BROAD")
    require(cfg.get("stop_paths"), "STOP_PATHS_REQUIRED")
    for path in cfg["stop_paths"]:
        absolute(path)
    require(isinstance(cfg.get("jobs"), list) and cfg["jobs"], "EMPTY_QUEUE")
    seen = set()
    for job in cfg["jobs"]:
        label, argv = job["label"], job["argv"]
        require(LABEL_RE.fullmatch(label) and label not in seen, "DUPLICATE_OR_BAD_LABEL")
        seen.add(label)
        require(job.get("readiness") == "READY_PENDING_HOST_PREFLIGHT", "JOB_NOT_READY")
        require(job.get("base_sha256") == SOURCE_SHA, "JOB_BASE_PIN")
        require(isinstance(argv, list) and all(isinstance(x, str) and "\0" not in x for x in argv), "ARGV_TYPE")
        require("--no-eval-public" in argv and "--eval-public" not in argv, "FRESH_EVAL_REQUIRED")
        for flag, value in (("--seed", "42"), ("--max-train-seconds", "1800"),
                            ("--num-steps", "1000000"), ("--out-dir", "out"), ("--seq-len", "1024")):
            require(flag_value(argv, flag) == value, "PINNED_FLAG: " + flag)
        require(job.get("same_host_control", {}).get("run_label"), "SAME_HOST_CONTROL_REQUIRED")
        require(isinstance(job.get("base_science_argv"), list) and job["base_science_argv"], "BASE_ARGV_REQUIRED")
        absolute(job["proposal"]["path"])
    if run:
        require(cfg.get("armed") is True, "QUEUE_UNARMED")
        require(all(job.get("armed") is True for job in cfg["jobs"]), "JOB_UNARMED")
    return cfg


def active_jobs():
    """Match executable/script tokens, not unrelated shell text or this runner's manifest."""
    hits = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            pid = int(path.parent.name)
            tokens = path.read_bytes().split(b"\0")
            names = [Path(x.decode(errors="replace")).name for x in tokens if x]
        except (OSError, ValueError):
            continue
        if pid == os.getpid():
            continue
        if any(x in names for x in ("torchrun", "train.py", "run_experiment.py", "submission_rehearsal.py", "host_driver.py")) or ("prepare.py" in names and "eval-public" in names):
            hits.append(pid)
    return hits


def device_holders():
    devices = sorted(str(p) for p in Path("/dev").glob("neuron[0-9]*"))
    require(devices, "NO_NEURON_DEVICE")
    probe = subprocess.run(["/usr/bin/fuser", *devices], capture_output=True, text=True)
    require(probe.returncode in (0, 1), "FUSER_FAILED: " + probe.stderr.strip())
    return probe.stdout.split() if probe.returncode == 0 else []


def training_metrics(path):
    text = path.read_text(errors="replace")
    out = {}
    for key in ("num_steps", "world_size", "seq_len"):
        matches = re.findall(r"^" + key + r":\s*(\d+)\s*$", text, re.M)
        require(len(matches) == 1, "MISSING_OR_DUPLICATE_SUMMARY: " + key)
        out[key] = int(matches[0])
    for key in ("training_seconds", "charged_seconds_est", "startup_allowance_est", "total_tokens_M"):
        matches = re.findall(r"^" + key + r":\s*(\S+)\s*$", text, re.M)
        require(len(matches) == 1 and math.isfinite(float(matches[0])), "INVALID_SUMMARY: " + key)
        out[key] = float(matches[0])
    require(re.findall(r"^over_budget:\s*(\S+)\s*$", text, re.M) == ["False"], "OVER_BUDGET_OR_MISSING_FLAG")
    require(out["num_steps"] > 0 and out["world_size"] == 4 and out["seq_len"] == 1024, "TRAIN_SHAPE_OR_ZERO_STEPS")
    require(0 < out["charged_seconds_est"] <= 1800, "CHARGED_BUDGET")
    losses = re.findall(r"\bstep\s+\d+\s*\|\s*loss\s+(\S+)", text)
    require(losses and all(math.isfinite(float(x)) and float(x) <= 100 for x in losses), "NONFINITE_OR_MISSING_LOSS")
    require("bad loss detected" not in text.lower(), "BAD_LOSS")
    out["over_budget"] = False
    return out


def evaluation_metrics(path):
    text, candidates = path.read_text(errors="replace"), []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and "val_bpb" in obj:
            candidates.append(obj)
    require(len(candidates) == 1, "MISSING_OR_DUPLICATE_EVAL_RESULT")
    result = candidates[0]
    require(math.isfinite(result["val_bpb"]) and result["val_bpb"] > 0, "INVALID_BPB")
    require(result.get("num_tokens") == 20971520 and result.get("eval_over_timeout") is False, "EVAL_TOKEN_OR_TIMEOUT_FAILURE")
    return result


class QueueRunner:
    def __init__(self, cfg, manifest_sha):
        self.cfg, self.manifest_sha = cfg, manifest_sha
        self.root = Path(cfg["run_root"])
        self.lock = None

    def guards(self, *, starting=False):
        require(socket.gethostname() == self.cfg["expected_hostname"], "HOSTNAME_MISMATCH")
        if starting:
            require(dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(self.cfg["freeze_utc"]), "RESEARCH_FREEZE")
        for value in self.cfg["stop_paths"]:
            require(not os.path.lexists(value), "STOP: " + value)
        require(not os.path.lexists(self.root / "STOP"), "QUEUE_STOP")
        require(shutil.disk_usage(self.root).free >= 6.5 * 1024**3, "DISK_BELOW_6_5_GIB")
        checked_file(self.cfg["source"])
        checked_file(self.cfg["prepare"])

    def wait_idle(self, max_wait=None, *, starting=False):
        started = time.monotonic()
        while True:
            self.guards(starting=starting)
            busy = {"processes": active_jobs(), "device_holders": device_holders()}
            if not any(busy.values()):
                return busy
            require(max_wait is None or time.monotonic() - started < max_wait, "POST_PHASE_NOT_QUIESCENT: " + json.dumps(busy))
            time.sleep(5)

    def environment(self, job, directory, cache):
        venv = str(Path(self.cfg["python"]).parent)
        env = {"HOME": "/home/ubuntu", "PATH": venv + ":/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
               "NKI_VENV_PATH": str(Path(venv).parent), "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": str(Path(self.cfg["prepare"]["path"]).parent),
               "NEURON_COMPETITION_R1_CACHE_DIR": "/home/ubuntu/.cache/neuron_competition_r1",
               "NEURON_LOGICAL_NC_CONFIG": "2", "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
               "NEURON_COMPETITION_R1_SEED": "42", "TRN_EXP_LABEL": job["label"]}
        for key, suffix in CACHE_KEYS.items():
            path = Path(cache) / suffix
            require(path.is_dir(), "WARM_CACHE_MISSING: " + str(path))
            env[key] = str(path)
        return env

    def commands(self, job, directory):
        return {
            "training": [self.cfg["torchrun"], "--standalone", "--nproc_per_node=4", "train.py", *job["argv"]],
            "fresh-public20m": [self.cfg["python"], self.cfg["prepare"]["path"], "eval-public", "--train-py", str(directory / "train.py"),
                                "--checkpoint", str(directory / "out/final.pt"), "--device", "neuron", "--batch-size", "1",
                                "--seq-len", "1024", "--eval-tokens", "20971520", "--timeout-seconds", "1200"],
        }

    def phase(self, job, directory, name, command, env):
        exclusive_json(directory / (name + "-launch.json"), {"utc": utc(), "label": job["label"], "command": command, "environment": env})
        limit = "5400s" if name == "training" else "1800s"
        with (directory / (name + ".log")).open("x") as log:
            child = subprocess.Popen(["/usr/bin/timeout", "--signal=TERM", "--kill-after=60s", limit, *command], cwd=directory, env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=(self.lock.fileno(),))
            exclusive_json(directory / (name + "-process.json"), {"pid": child.pid, "utc": utc()})
            code = child.wait()
        exclusive_json(directory / (name + "-exit.json"), {"exit_code": code, "utc": utc()})
        require(code == 0, name.upper() + "_EXIT_" + str(code))
        self.wait_idle(max_wait=30)

    def record_terminal(self, directory, record):
        exclusive_json(directory / "terminal.json", record)
        self.record_once(record)

    def record_once(self, record):
        ledger = self.root / "results.jsonl"
        if ledger.exists():
            seen = [json.loads(line) for line in ledger.read_text().splitlines() if line]
            if any(row["label"] == record["label"] for row in seen):
                return
        append_json(ledger, record)

    def execute(self, job):
        directory = self.root / job["label"]
        digest = fingerprint(job)
        if (directory / "terminal.json").exists():
            record = json.loads((directory / "terminal.json").read_text())
            require(record["job_sha256"] == digest, "CONSUMED_LABEL_CHANGED")
            self.record_once(record)
            require(record["status"] == "evaluated_pending_review", "PREVIOUS_JOB_FAILED: " + job["label"])
            return
        if directory.exists():
            # An attempt without a terminal receipt is only observed, never relaunched.
            self.wait_idle()
            record = {"label": job["label"], "status": "interrupted", "error": "STARTED_LABEL_WITHOUT_TERMINAL_RECEIPT",
                      "job_sha256": digest, "utc": utc(), "review_required": True}
            self.record_terminal(directory, record)
            raise RuntimeError(record["error"])
        self.wait_idle(starting=True)
        proposal = json.loads(checked_file(job["proposal"]).read_text())
        binding = {k: job[k] for k in ("label", "argv", "kind", "arm_id", "base_sha256")}
        binding.update(host=self.cfg["host"], source_sha256=SOURCE_SHA)
        require(proposal.get("queue_binding") == binding, "PROPOSAL_DOES_NOT_BIND_EXACT_JOB")
        require(proposal.get("status") == "READY_PENDING_HOST_PREFLIGHT", "PROPOSAL_NOT_READY")
        directory.mkdir()
        record = {"label": job["label"], "host": self.cfg["host"], "job_sha256": digest, "manifest_sha256": self.manifest_sha,
                  "train_py_sha256": SOURCE_SHA, "prepare_sha256": PREPARE_SHA, "proposal_sha256": job["proposal"]["sha256"],
                  "arm_id": job["arm_id"], "kind": job["kind"], "base_sha256": job["base_sha256"], "argv": job["argv"],
                  "base_science_argv": job["base_science_argv"], "base_argv_sha256": fingerprint(job["base_science_argv"]),
                  "same_host_control": job["same_host_control"],
                  "started_utc": utc(), "review_required": True}
        exclusive_json(directory / "attempt.started.json", record)
        try:
            shutil.copyfile(self.cfg["source"]["path"], directory / "train.py")
            require(sha(directory / "train.py") == SOURCE_SHA, "STAGED_SOURCE_MISMATCH")
            exclusive_json(directory / "proposal.json", proposal)
            commands = self.commands(job, directory)
            self.guards(starting=True)
            exclusive_json(directory / "preflight.json", {"utc": utc(), "disk_free_bytes": shutil.disk_usage(self.root).free,
                           "train_sha256": SOURCE_SHA, "prepare_sha256": PREPARE_SHA, "idle": True, "device_lock_held": True})
            self.phase(job, directory, "training", commands["training"], self.environment(job, directory, self.cfg["train_cache"]))
            metrics = training_metrics(directory / "training.log")
            require((directory / "out/final.pt").is_file(), "MISSING_CHECKPOINT")
            checkpoint_sha = sha(directory / "out/final.pt")
            self.guards()
            require(sha(directory / "train.py") == SOURCE_SHA, "SOURCE_CHANGED_DURING_TRAINING")
            self.phase(job, directory, "fresh-public20m", commands["fresh-public20m"], self.environment(job, directory, self.cfg["eval_cache"]))
            require(sha(directory / "train.py") == SOURCE_SHA, "SOURCE_CHANGED_DURING_EVAL")
            require(sha(directory / "out/final.pt") == checkpoint_sha, "CHECKPOINT_CHANGED_DURING_EVAL")
            result = evaluation_metrics(directory / "fresh-public20m.log")
            record.update(status="evaluated_pending_review", metrics=metrics, evaluation=result, checkpoint_sha256=checkpoint_sha,
                          checkpoint_uri=str(directory / "out/final.pt"), public_20m=result["val_bpb"], steps=metrics["num_steps"],
                          eval_tokens=20971520, eval_seq_len=1024, utc=utc())
            self.record_terminal(directory, record)
        except Exception as error:
            record.update(status="failed", error=str(error), utc=utc())
            if not (directory / "terminal.json").exists():
                self.record_terminal(directory, record)
            raise

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        require(not self.root.is_symlink(), "RUN_ROOT_SYMLINK")
        # The live manual A1/A2 drivers use this same lock over train AND eval.
        with open(self.cfg["device_lock"], "a") as self.lock:
            while True:
                self.guards(starting=True)
                if (self.root / "PAUSE_NEXT").exists():
                    return "paused_between_jobs"
                try:
                    fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(5)
            for job in self.cfg["jobs"]:
                if (self.root / "PAUSE_NEXT").exists():
                    return "paused_between_jobs"
                self.execute(job)
        return "queue_complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run", action="store_true", help="Operator only; default is validation without launch")
    args = parser.parse_args()
    require(sha(args.manifest) == args.manifest_sha256, "MANIFEST_HASH_MISMATCH")
    cfg = validate_manifest(json.loads(Path(args.manifest).read_text()), run=args.run)
    if not args.run:
        print(json.dumps({"status": "schema_valid", "armed": cfg.get("armed") is True, "jobs": len(cfg["jobs"])}))
        return 0
    runner = QueueRunner(cfg, args.manifest_sha256)
    try:
        status = runner.run()
        append_json(runner.root / "queue-events.jsonl", {"utc": utc(), "status": status, "manifest_sha256": args.manifest_sha256})
        print(status, flush=True)
        return 0
    except Exception as error:
        event = {"utc": utc(), "status": "queue_stopped", "error": str(error), "manifest_sha256": args.manifest_sha256}
        if runner.root.is_dir():
            append_json(runner.root / "queue-events.jsonl", event)
        print(json.dumps(event), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
