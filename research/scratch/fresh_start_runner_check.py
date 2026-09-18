#!/usr/bin/env python3
"""Check that crash output survives and duplicate run labels cannot overwrite it."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


RUNNER = Path(__file__).resolve().parents[1] / "run_experiment.py"


def main():
    spec = importlib.util.spec_from_file_location("runner", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with tempfile.TemporaryDirectory(prefix="fresh-start-runner-") as directory:
        root = Path(directory)
        logfile = root / "crash.log"
        code = "import sys,time; print('live-before-exit',flush=True); time.sleep(1); print('failure',file=sys.stderr); sys.exit(7)"
        # A parent process checks the file BEFORE the child finishes, proving
        # incremental persistence rather than only checking the final file.
        invocation = (
            "import importlib.util,os,sys; from pathlib import Path; "
            "s=importlib.util.spec_from_file_location('r',sys.argv[1]); "
            "r=importlib.util.module_from_spec(s); s.loader.exec_module(r); "
            "sys.exit(r.run_to_log([sys.executable,'-u','-c',sys.argv[3]],"
            "Path(sys.argv[2]),cwd=Path(sys.argv[2]).parent,env=os.environ.copy()))"
        )
        process = subprocess.Popen([sys.executable, "-c", invocation,
                                    str(RUNNER), str(logfile), code])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if logfile.exists() and "live-before-exit" in logfile.read_text():
                break
            time.sleep(0.02)
        assert process.poll() is None, "log wasn't visible until process exit"
        assert process.wait(timeout=5) == 7
        before = logfile.read_bytes()
        assert b"failure" in before
        try:
            runner.run_to_log([sys.executable, "-c", "print('overwrite')"],
                              logfile, cwd=root, env=os.environ.copy())
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing log was not protected")
        assert logfile.read_bytes() == before
    summary = runner.parse_summary("public_val_bpb: 1.005\nnum_steps: 2104\ncausality_passed: True\n")
    assert summary == {"public_val_bpb": 1.005, "num_steps": 2104, "causality_passed": True}
    print("PASS: logs visible during run, crash output retained, overwrite refused, summary typed")


if __name__ == "__main__":
    main()
