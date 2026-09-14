"""Suspect (a) probe: is the data each rank trains on world-size-dependent beyond
'half as much per rank'?

train.py's loader is prepare.make_dataloader, which shards by ROW GROUP:
    _row_groups_for_split(split, start=rank, step=world_size)
  -> [all_groups[i] for i in range(rank, len(all_groups), world_size)]

Two things this can get wrong and that would scale differently with rank count:
  1. remainder handling -- a rank with fewer row groups exhausts its stream (and wraps
     to epoch 2, i.e. starts REPEATING data) earlier than its peers, and only rank 0's
     `epoch` is ever printed (print0), so an imbalance is invisible in the logs.
  2. packer waste -- make_dataloader's best-fit packer TRUNCATES (discards the tail of)
     the shortest buffered doc whenever nothing fits the remaining space. If that rate
     depended on the shard split it would bias what a smaller number of larger per-rank
     streams trains on.
"""
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, "/home/ubuntu/trainium_frontier")
import prepare  # noqa: E402

PATHS = [prepare.DATA_DIR / prepare.shard_filename(i) for i in prepare.PUBLIC_TRAIN_SHARDS]

# ---------------------------------------------------------------- part 1: balance
groups = []
for p in PATHS:
    pf = pq.ParquetFile(p)
    for rg in range(pf.num_row_groups):
        groups.append((p, rg))
print(f"train row groups: {len(groups)} over {len(PATHS)} shards")

print("\nmeasuring per-row-group document count and total text bytes ...")
rg_docs, rg_bytes = [], []
for p in PATHS:
    pf = pq.ParquetFile(p)
    for rg in range(pf.num_row_groups):
        col = pf.read_row_group(rg, columns=["text"]).column("text")
        rg_docs.append(len(col))
        rg_bytes.append(sum(len(v.as_py()) for v in col))
    print(f"  {p.name} done", flush=True)

for ws in (1, 2, 4, 8):
    d = [sum(rg_docs[i] for i in range(r, len(groups), ws)) for r in range(ws)]
    b = [sum(rg_bytes[i] for i in range(r, len(groups), ws)) for r in range(ws)]
    n = [len(range(r, len(groups), ws)) for r in range(ws)]
    print(f"ws={ws}: rgs/rank={n}")
    print(f"       docs/rank min={min(d)} max={max(d)} spread={(max(d)-min(d))/max(d)*100:.3f}%")
    print(f"       bytes/rank min={min(b)} max={max(b)} spread={(max(b)-min(b))/max(b)*100:.3f}%"
          f"  -> earliest rank wraps to epoch 2 at {min(b)/max(b)*100:.2f}% of the slowest rank's epoch")
    print(f"       total bytes={sum(b)} (identical across ws: {sum(b) == sum(rg_bytes)})")

# ------------------------------------------------------------- part 2: packer waste
import torch  # noqa: E402

tok = prepare.ensure_tokenizer(build_if_missing=False)
BOS = tok.get_bos_token_id()
SEQ = 4 * 2048          # PACK_FACTOR * SEQ_LEN, what train.py passes as loader_seq_len
NBATCH = 10


def packer_stats(rank, world_size, nbatch=NBATCH):
    """Re-implements make_dataloader's inner packing loop verbatim, counting the tokens
    the `else` branch throws away."""
    batches = prepare._document_batches("train", rank, world_size, 128)
    cap = SEQ + 1
    buf, buffer_min = [], 512
    kept = dropped = ndocs = ntrunc = 0
    for _ in range(nbatch):
        pos = 0
        while pos < cap:
            while len(buf) < buffer_min:
                docs, _ = next(batches)
                buf.extend(tok.encode(docs, prepend=BOS, num_threads=4))
            remaining = cap - pos
            best_idx, best_len = -1, 0
            for i, doc in enumerate(buf):
                n = len(doc)
                if n <= remaining and n > best_len:
                    best_idx, best_len = i, n
            if best_idx >= 0:
                doc = buf.pop(best_idx)
                kept += len(doc); ndocs += 1; pos += len(doc)
            else:
                si = min(range(len(buf)), key=lambda i: len(buf[i]))
                doc = buf.pop(si)
                kept += remaining
                dropped += len(doc) - remaining
                ndocs += 1; ntrunc += 1; pos += remaining
    return kept, dropped, ndocs, ntrunc


print(f"\npacker waste over {NBATCH} loader rows of {SEQ + 1} tokens (= {NBATCH} micro-batches):")
for ws in (4, 8):
    tot_k = tot_d = tot_n = tot_t = 0
    for r in range(ws):
        k, dr, nd, nt = packer_stats(r, ws)
        tot_k += k; tot_d += dr; tot_n += nd; tot_t += nt
    print(f"  ws={ws}: kept={tot_k:,} discarded={tot_d:,} "
          f"({tot_d / (tot_k + tot_d) * 100:.3f}% of tokens read) docs={tot_n:,} "
          f"truncations={tot_t} ({tot_t / tot_n * 100:.2f}% of docs)")
