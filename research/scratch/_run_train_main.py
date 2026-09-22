"""Import a train.py as a module (so its `__main__` adapter install does not run) and call main()."""
import importlib.util
import sys
import time

path, *argv = sys.argv[1:]
# Freeze the wall clock: the LR / weight-decay / Muon-momentum schedules are functions of elapsed
# seconds, and CPU step times differ between two processes, so a real clock would make the two
# runs diverge for reasons unrelated to the code under test. A constant clock pins every schedule
# at its t=0 value in both runs (lrm 1.0, wd_progress 0.0) and never trips the time budget.
time.monotonic = lambda: 1000.0
spec = importlib.util.spec_from_file_location("train_under_test", path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod  # dataclasses under `from __future__ import annotations` resolve the module by name
sys.argv = ["train.py", *argv]
spec.loader.exec_module(mod)
mod.main()
