#!/usr/bin/env python3
"""CPU-only training-row vocabulary/metadata cost audit with the fixed loader."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import prepare
from embedding_compact_probe import compact_metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, default=2048)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert args.batches > 0 and args.shard in prepare.PUBLIC_TRAIN_SHARDS
    torch.set_num_threads(4)
    tokenizer = prepare.ensure_tokenizer(build_if_missing=False)
    loader = prepare.make_dataloader(tokenizer, 1, 8192, "train", "cpu", shard_ids=[args.shard])
    counts, times, epochs = [], [], []
    digest = hashlib.sha256()
    start = time.perf_counter()
    for i in range(args.batches):
        x, _, state = next(loader)
        before = time.perf_counter()
        inverse, rows, unique = compact_metadata(x.reshape(-1), tokenizer.get_vocab_size())
        times.append((time.perf_counter() - before) * 1000)
        counts.append(unique)
        epochs.append(state["epoch"])
        digest.update(x.numpy().tobytes())
        if (i + 1) % 256 == 0:
            print(json.dumps(dict(batches=i + 1, max_unique=max(counts), elapsed=time.perf_counter()-start)), flush=True)
    q = [0, 50, 90, 95, 99, 100]
    result = dict(training_split_only=True, shard=args.shard, batches=args.batches,
                  tokens=args.batches * 8192, world_size=1, loader_batch_size=1,
                  loader_sequence_length=8192, tokenizer_sha256=prepare.TOKENIZER_PKL_SHA256,
                  prepare_sha256=prepare.sha256_file(Path(prepare.__file__)),
                  input_stream_sha256=digest.hexdigest(), epochs_seen=sorted(set(epochs)),
                  unique_percentiles=dict(zip(map(str, q), np.percentile(counts, q).tolist())),
                  metadata_cpu_ms_percentiles=dict(zip(map(str, q), np.percentile(times, q).tolist())),
                  bucket_counts={str(b): sum((b // 2 < n <= b) if b > 128 else n <= b for n in counts)
                                 for b in (128, 256, 512, 1024, 2048, 4096, 8192)},
                  unique_counts=counts, elapsed_seconds=time.perf_counter()-start,
                  caveat="One shard prefix and one rank; no assertion that all future batches fit the observed maximum. CPU timings are on the local host, not EC2.")
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'unique_counts'}), flush=True)


if __name__ == "__main__":
    main()
