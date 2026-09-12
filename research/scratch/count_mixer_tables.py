"""T4.1 step 2: tokenize train shards 0-3 and build the frozen count tables.

Part of the §6 offline protocol in research/ngram-drafter-and-count-mixer-2026-09-12.md.
Shards 0-3 build the tables; shard 4 is held out (it is stream (b) for gate fitting),
and the public val shard is never touched here.

Tokenization follows `count_train_tokens.py` / `prepare.make_dataloader` exactly:
every document is BOS-prepended and encoded with tiktoken's batch encoder on all
vCPUs. N-grams are counted *within* a document only, so no pair ever crosses a
document boundary (the alternative -- counting through boundaries, as the packer's
rows do -- is a `--cross-doc` flag).

Outputs, all under --out:
  unigram.npy      int64 [8192]
  bigram.npy       int32 [8192, 8192]        dense bigram counts (268 MB)
  trigram_keys.npy int64 [M]  sorted (prev2, prev, cur) keys   } exact trigram, no
  trigram_cnts.npy int32 [M]  matching counts                  } hashing (upper bound)
  tokens_shard*.npy int16 per-shard token stream + doc offsets (reused by other steps)

Pure CPU, does not touch the Neuron device.

Run: python research/scratch/count_mixer_tables.py --out ~/countmix
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import prepare as P

V = P.TOKENIZER_VOCAB_SIZE
BATCH = 512


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--cross-doc", action="store_true",
                    help="count n-grams through document boundaries too")
    ap.add_argument("--trigram", action="store_true", default=True)
    ap.add_argument("--no-trigram", dest="trigram", action="store_false")
    return ap.parse_args()


def tokenize_shard(tok, path, threads: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (tokens int16 [T], doc_starts int64 [D]) for one shard."""
    chunks: list[np.ndarray] = []
    lens: list[int] = []
    pf = P.pq.ParquetFile(path)
    for rg_idx in range(pf.num_row_groups):
        rows = pf.read_row_group(rg_idx).column("text").to_pylist()
        for i in range(0, len(rows), BATCH):
            for ids in tok.encode(rows[i:i + BATCH], prepend=P.BOS_TOKEN, num_threads=threads):
                chunks.append(np.asarray(ids, dtype=np.int16))
                lens.append(len(ids))
    tokens = np.concatenate(chunks)
    doc_starts = np.concatenate([[0], np.cumsum(np.asarray(lens, dtype=np.int64))[:-1]])
    return tokens, doc_starts


def main() -> None:
    args = parse_args()
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)
    tok = P.ensure_tokenizer(build_if_missing=False)

    unigram = np.zeros(V, dtype=np.int64)
    bigram = np.zeros(V * V, dtype=np.int32)
    tri_key_parts: list[np.ndarray] = []
    tri_cnt_parts: list[np.ndarray] = []
    total_tokens = 0
    started = time.monotonic()

    for shard in args.shards:
        t0 = time.monotonic()
        path = P.list_shard_paths([shard])[0]
        tokens, doc_starts = tokenize_shard(tok, path, args.threads)
        np.save(os.path.join(out_dir, f"tokens_shard{shard:05d}.npy"), tokens)
        np.save(os.path.join(out_dir, f"docstarts_shard{shard:05d}.npy"), doc_starts)
        u = tokens.astype(np.int64)
        total_tokens += u.size
        unigram += np.bincount(u, minlength=V)

        # Positions whose n-gram is entirely inside one document. A bigram at position i
        # uses (i-1, i); it is intra-doc iff i is not a doc start.
        if args.cross_doc:
            bi_ok = np.ones(u.size, dtype=bool)
            tri_ok = np.ones(u.size, dtype=bool)
            bi_ok[0] = tri_ok[0] = tri_ok[1] = False
        else:
            is_start = np.zeros(u.size, dtype=bool)
            is_start[doc_starts] = True
            bi_ok = ~is_start
            bi_ok[0] = False
            tri_ok = bi_ok.copy()
            tri_ok[1:] &= bi_ok[:-1]
            tri_ok[0] = False

        idx = np.nonzero(bi_ok)[0]
        keys = u[idx - 1] * V + u[idx]
        uk, uc = np.unique(keys, return_counts=True)
        bigram[uk] += uc.astype(np.int32)   # uk unique, so plain fancy-index add is exact
        del keys, uk, uc, idx

        if args.trigram:
            idx3 = np.nonzero(tri_ok)[0]
            k3 = (u[idx3 - 2] * V + u[idx3 - 1]) * V + u[idx3]
            uk3, uc3 = np.unique(k3, return_counts=True)
            tri_key_parts.append(uk3)
            tri_cnt_parts.append(uc3.astype(np.int32))
            del k3, idx3
        del u, tokens
        print(f"  shard {shard}: {total_tokens:,} tokens cumulative, {time.monotonic() - t0:.1f}s",
              flush=True)

    np.save(os.path.join(out_dir, "unigram.npy"), unigram)
    np.save(os.path.join(out_dir, "bigram.npy"), bigram.reshape(V, V))

    if args.trigram:
        t0 = time.monotonic()
        all_keys = np.concatenate(tri_key_parts)
        all_cnts = np.concatenate(tri_cnt_parts)
        del tri_key_parts, tri_cnt_parts
        order = np.argsort(all_keys)
        all_keys = all_keys[order]
        all_cnts = all_cnts[order]
        del order
        first = np.concatenate(([0], np.flatnonzero(all_keys[1:] != all_keys[:-1]) + 1))
        uk = all_keys[first]
        summed = np.add.reduceat(all_cnts.astype(np.int64), first)
        np.save(os.path.join(out_dir, "trigram_keys.npy"), uk)
        np.save(os.path.join(out_dir, "trigram_cnts.npy"), summed.astype(np.int32))
        print(f"  trigram merge: {uk.size:,} distinct trigrams, {time.monotonic() - t0:.1f}s")

    meta = {
        "shards": args.shards, "cross_doc": bool(args.cross_doc), "trigram": bool(args.trigram),
        "total_tokens": int(total_tokens), "seconds": time.monotonic() - started,
        "distinct_bigrams": int((bigram > 0).sum()),
    }
    with open(os.path.join(out_dir, "tables_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
