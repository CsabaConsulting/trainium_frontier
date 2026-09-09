"""C2 (day-1 plan item 2): count tokens across the 5 public train shards once, to confirm
`epoch` is still 1 at the current ~262M-tokens/run steady state.

Tokenizes every document in PUBLIC_TRAIN_SHARDS (BOS-prepended, exactly as
prepare.make_dataloader consumes them) with tiktoken's batch encoder across all
available CPUs, and reports the total token count. Pure CPU, does not touch the
device -- safe to run alongside a real training/eval job on the box.

Run: python research/scratch/count_train_tokens.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import prepare as P

BATCH = 512
THREADS = int(os.environ.get("NEURON_COMPETITION_R1_TOKENIZER_THREADS", "12"))


def main():
    tok = P.ensure_tokenizer(build_if_missing=False)
    paths = P.list_shard_paths(P.PUBLIC_TRAIN_SHARDS)
    print(f"shards: {[str(p) for p in paths]}")

    started = time.monotonic()
    total_tokens = 0
    total_docs = 0
    per_shard = {}
    for path in paths:
        shard_tokens = 0
        shard_docs = 0
        pf = P.pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            rows = rg.column("text").to_pylist()
            for i in range(0, len(rows), BATCH):
                batch = rows[i:i + BATCH]
                ids_batch = tok.encode(batch, prepend=P.BOS_TOKEN, num_threads=THREADS)
                for ids in ids_batch:
                    shard_tokens += len(ids)
                shard_docs += len(batch)
        per_shard[str(path.name)] = {"tokens": shard_tokens, "docs": shard_docs}
        total_tokens += shard_tokens
        total_docs += shard_docs
        print(f"  {path.name}: {shard_docs:,} docs, {shard_tokens:,} tokens "
              f"(running total {total_tokens:,})", flush=True)

    elapsed = time.monotonic() - started
    print(f"\nTOTAL: {total_docs:,} docs, {total_tokens:,} tokens across "
          f"{len(paths)} shards, {elapsed:.1f}s")
    print(f"per_shard={per_shard}")

    # Steady-state per-run consumption at the locked config: nkimuon-fullbudget's
    # 2043 steps x TOTAL_BATCH_SIZE=131072 = 267,780,096 tokens (the real headline run,
    # not a rounded guess).
    steady_state_tokens = 2043 * 131072
    n_epochs_per_run = steady_state_tokens / total_tokens if total_tokens else float("nan")
    print(f"\nnkimuon-fullbudget steady state (2043 steps x 131072 tok/step = "
          f"{steady_state_tokens:,} tokens/run): {n_epochs_per_run:.4f} epochs consumed per run "
          f"({'epoch stays 1' if n_epochs_per_run <= 1.0 else 'CROSSES INTO EPOCH 2'}).")


if __name__ == "__main__":
    main()
