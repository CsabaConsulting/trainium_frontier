#!/usr/bin/env python3
"""TEST-ONLY stand-in for a training run: prints step lines and the '---' summary block in the same
format as train.py, writes out/<label>/final.pt. Modes via FAKE_MODE: ok | badloss | escape."""
import os, sys, time, subprocess
out = sys.argv[sys.argv.index("--out-dir") + 1]
mode = os.environ.get("FAKE_MODE", "ok")
steps = int(os.environ.get("FAKE_STEPS", "3640"))
for s in range(0, 60, 10):
    ep = 1 if s < 40 else 2
    print(f"step {s:05d} | loss 3.1000 | lrm 1.000 | dt 0.49s | tok/s 267,000 | remaining_steps 1 | remaining_time 1000s | epoch {ep}", flush=True)
if mode == "badloss":
    print("RuntimeError: bad loss detected: nan", flush=True); sys.exit(1)
if mode == "escape":  # a descendant that leaves the session and keeps writing evidence
    subprocess.Popen(["setsid", "bash", "-c", f'trap "" TERM; while true; do echo x >> {out}/stray.txt; sleep 0.3; done'])
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "final.pt"), "wb") as f:
    f.write(os.urandom(4096))
print("---")
for k, v in [("checkpoint_path", f"{out}/final.pt"), ("training_seconds", "1794.8"), ("budget_seconds", "1800"),
             ("startup_allowance_est", "131.0"), ("charged_seconds_est", "1795.2"), ("over_budget", "False"),
             ("num_steps", str(steps)), ("total_tokens_M", "477.0"), ("world_size", "4"), ("seq_len", "1024"), ("num_params_M", "117.5")]:
    print(f"{k}: {v}")
