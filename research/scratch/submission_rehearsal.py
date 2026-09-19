#!/usr/bin/env python3
"""Run a submission in a directory containing only train.py and prepare.py.

The training command is parsed from the actual launch-command.txt, never
reconstructed through run_experiment.py. Only the output directory and step-log
path are injected using the organizer's documented environment variables.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--sandbox", type=Path, required=True)
    ap.add_argument("--mode", choices=("train", "eval", "check"), required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--eval-tokens", type=int, default=20971520)
    ap.add_argument("--fresh-caches", action="store_true",
                    help="Use empty, job-owned native SDK cache directories; keep shared caches intact")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    package = args.package.resolve()
    sandbox = args.sandbox.resolve()
    assert sandbox.is_relative_to(root / "out")
    sandbox.mkdir(exist_ok=False, parents=True)
    manifest = json.loads((package / "provenance.json").read_text())
    assert sha(package / "train.py") == manifest["train_sha256"]
    for src in (package / "train.py", root / "prepare.py"):
        shutil.copy2(src, sandbox / src.name)
        assert sha(src) == sha(sandbox / src.name)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    caches = {}
    if args.fresh_caches:
        for key, directory in (
            ("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"),
            ("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "local"),
            ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo"),
            ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
            ("NEURON_COMPILE_CACHE_DIR", "compiler"),
            ("TORCH_NEURONX_DEBUG_DIR", "debug"),
        ):
            cache = sandbox / "caches" / directory
            cache.mkdir(parents=True, exist_ok=False)
            assert not list(cache.iterdir())
            env[key] = str(cache)
            caches[key] = str(cache)
    launch = (package / "launch-command.txt").read_text().strip()
    assert launch == manifest["launch_command"]
    print("ISOLATED_SUBMISSION " + json.dumps(dict(
        mode=args.mode, sandbox=str(sandbox), files=sorted(p.name for p in sandbox.iterdir()),
        train_sha256=sha(sandbox / "train.py"), prepare_sha256=sha(sandbox / "prepare.py"),
        launch_command=launch, fresh_empty_sdk_caches=caches)), flush=True)
    subprocess.run([sys.executable, "prepare.py", "check", "--train-py", "train.py",
                    "--launch-command", launch], cwd=sandbox, env=env, check=True)
    if args.mode == "check":
        return
    if args.mode == "eval":
        assert args.checkpoint is not None and args.checkpoint.is_file()
        env["NEURON_CC_FLAGS"] = "--optlevel=1"
        command = [sys.executable, "prepare.py", "eval-public", "--train-py", "train.py",
                   "--checkpoint", str(args.checkpoint.resolve()), "--device", "neuron",
                   "--batch-size", "1", "--seq-len", "1024", "--eval-tokens", str(args.eval_tokens),
                   "--timeout-seconds", "1200"]
    else:
        command = shlex.split(launch)
        while command and "=" in command[0] and not command[0].startswith("-"):
            key, value = command.pop(0).split("=", 1)
            env[key] = value
        env["NEURON_COMPETITION_R1_OUT_DIR"] = str(sandbox / "out")
        env["NEURON_COMPETITION_R1_STEP_LOG"] = str(sandbox / "steps.jsonl")
    print("REHEARSAL_COMMAND " + json.dumps(command), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=sandbox, env=env)
    observed = []
    step_log = sandbox / "steps.jsonl"
    while process.poll() is None:
        if args.mode == "train" and step_log.exists():
            steps = step_log.read_text().splitlines()
            while len(observed) < len(steps):
                observed.append(dict(step=int(steps[len(observed)]),
                                     observed_seconds=time.monotonic() - started))
        time.sleep(.05)
    result = dict(returncode=process.returncode, wall_seconds=time.monotonic() - started,
                  step_boundaries=len(observed), mode=args.mode, fresh_empty_sdk_caches=caches)
    if args.mode == "train" and process.returncode == 0:
        assert len(observed) >= 5, "five official startup boundaries were not observed"
        checkpoint = sandbox / "out/final.pt"
        assert checkpoint.is_file()
        # The final checkpoint must exist by the deadline, independently of the
        # trainer's self-reported time. Process exit is a conservative upper bound.
        allowance = min(observed[4]["observed_seconds"], 900.0)
        result.update(startup_allowance_observed=allowance,
                      charged_through_process_exit=result["wall_seconds"] - allowance,
                      checkpoint_bytes=checkpoint.stat().st_size,
                      checkpoint_sha256=sha(checkpoint), observations=observed)
        result["charged_through_process_exit_within_budget"] = result["charged_through_process_exit"] <= 1800
    (sandbox / "rehearsal.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SUBMISSION_REHEARSAL " + json.dumps({k: v for k, v in result.items() if k != "observations"}), flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)
    if args.mode == "train":
        assert result["charged_through_process_exit_within_budget"], result


if __name__ == "__main__":
    main()
