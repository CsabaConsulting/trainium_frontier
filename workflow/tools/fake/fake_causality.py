#!/usr/bin/env python3
"""TEST-ONLY checkpoint-bound causality result."""
import argparse
import hashlib
import json
import os


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


p = argparse.ArgumentParser()
p.add_argument("--train-py", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--seq-len", type=int, required=True)
a = p.parse_args()
passed = os.environ.get("FAKE_CAUSALITY", "pass") == "pass"
print(json.dumps({
    "schema": "checkpoint-causality-v1",
    "causality_passed": passed,
    "max_abs_diff": 0.0 if passed else 1.0,
    "tolerance": 0.01,
    "seq_len": a.seq_len,
    "train_py_sha256": digest(a.train_py),
    "checkpoint_sha256": digest(a.checkpoint),
}, indent=2, sort_keys=True))
