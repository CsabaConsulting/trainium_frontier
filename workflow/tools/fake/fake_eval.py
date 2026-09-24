#!/usr/bin/env python3
"""TEST-ONLY stand-in for the canonical `prepare.py eval-public` JSON."""
import argparse
import json
import math
import os
import time

p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["eval-public"])
p.add_argument("--train-py", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--device", required=True)
p.add_argument("--batch-size", type=int, required=True)
p.add_argument("--seq-len", type=int, required=True)
p.add_argument("--eval-tokens", type=int, required=True)
p.add_argument("--timeout-seconds", type=int, required=True)
a = p.parse_args()
bpb = float(os.environ.get("FAKE_BPB", "0.9920"))
nbytes = 4_000_000
row = {
    "val_bpb": bpb,
    "eval_seconds": 1.0,
    "eval_timeout": a.timeout_seconds,
    "eval_over_timeout": False,
    "num_tokens": a.eval_tokens,
    "num_bytes": nbytes,
    "total_nats": bpb * math.log(2) * nbytes,
}
mode = os.environ.get("FAKE_EVAL_MODE", "ok")
if mode == "missing":
    del row["num_tokens"]
elif mode == "wrong_tokens":
    row["num_tokens"] -= 1
elif mode == "nonfinite":
    row["val_bpb"] = float("nan")
print(json.dumps(row, indent=2, sort_keys=True))
if mode == "multiple":
    print(json.dumps(row))
elif mode == "hang_after_json":
    time.sleep(30)
