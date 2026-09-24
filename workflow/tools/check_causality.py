#!/usr/bin/env python3
"""Fresh-process causality check bound to exact train.py and checkpoint bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import prepare


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-py", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    args = parser.parse_args()
    train_py, checkpoint = Path(args.train_py).resolve(), Path(args.checkpoint).resolve()
    train_mod = prepare._import_train_module(str(train_py))
    if not hasattr(train_mod, "load_for_eval"):
        raise RuntimeError("train.py must define load_for_eval(checkpoint_path, device)")
    device = prepare._detect_eval_device(args.device)
    prepare._init_eval_distributed(device)
    try:
        tokenizer = prepare.ensure_tokenizer(build_if_missing=False)
        model = train_mod.load_for_eval(str(checkpoint), device)
        result = prepare.causality_check(
            model, tokenizer, args.batch_size, args.seq_len, device, tolerance=args.tolerance,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps({
                "schema": "checkpoint-causality-v1",
                "causality_passed": result["passed"],
                "max_abs_diff": result["max_abs_diff"],
                "tolerance": result["tolerance"],
                "seq_len": args.seq_len,
                "train_py_sha256": sha256(train_py),
                "checkpoint_sha256": sha256(checkpoint),
            }, indent=2, sort_keys=True))
    finally:
        prepare._cleanup_eval_distributed()


if __name__ == "__main__":
    main()
