#!/usr/bin/env python3
"""Why is an nn.Embedding row lookup so expensive on this stack, and is there a faster form?

Context (T3.1, 2026-09-11). Hashed n-gram value embeddings measure a real
quality-per-step win (-0.0311 bpb at a fixed 300 steps) but cost 2x the step time,
which more than eats it. research/scratch/phase_timer.py localized the whole cost to
the compiled forward+backward -- NOT the all-reduce, the clip or the optimizer -- and,
against expectation, showed it scales with the table's ROW COUNT and barely at all with
its WIDTH:

    5 tables, 65536 rows, dim 640, trainable   +191.6 ms/micro
    5 tables, 65536 rows, dim 640, FROZEN      + 91.6 ms/micro   (forward gather alone)
    5 tables, 65536 rows, dim 128, trainable   +179.2 ms/micro   (only 7% cheaper)
    5 tables,  8192 rows, dim 640, trainable   +104.0 ms/micro
    3 tables, 65536 rows, dim 640, trainable   +131.6 ms/micro

A row lookup's output size is rows-independent, so a cost that tracks `rows` means the
lowering is not a gather. This isolates the op away from the model and asks four
questions that decide whether T3.1 lives:

  A. SHAPE SCALING: fwd and fwd+bwd for the (rows x dim) grid, so the cost model is
     measured rather than inferred from five whole-model runs.
  B. FIXED COST PER GATHER: N separate gathers from N tables vs ONE gather from a
     single N*rows table with offset indices (which is a ~10-line change to
     _ngram_hash_indices, since the primes already differ per table). If most of the
     cost is per-gather overhead, merging is nearly free; if it is per-row, merging is
     neutral and this line is closed.
  C. ALTERNATIVE LOWERINGS: nn.Embedding vs index_select vs plain advanced indexing vs
     an explicit one-hot matmul (the last only where the one-hot fits).
  D. BASELINE EXPOSURE: the same measurement at the baseline's own unigram
     value-embedding shape (8192 x 640, three of them). If a lookup at that shape is
     also tens of ms, then the CURRENT locked model is already spending a large part of
     its step on value-embedding gathers and a faster form is a win with or without
     T3.1.

Run (4 ranks matches the real shape's per-rank work; 1 rank is enough for the ratios):
    NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=1 \
        research/scratch/embed_gather_bench.py
"""

import argparse
import os
import sys
import time

REPO = os.environ.get("PROF_REPO", "/home/ubuntu/trainium_frontier")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--rows-batch", type=int, default=4, help="micro-batch rows (train.py: 4)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    os.chdir(REPO)

    import torch
    import torch.nn as nn
    import train as T

    device_type = T.detect_device_type("")
    _ddp, rank, _lr, _ws, device = T.init_runtime(device_type)
    dtype = torch.bfloat16
    b, t = args.rows_batch, args.seq_len
    ntok = b * t

    def timeit(fn) -> float:
        for _ in range(args.warmup):
            fn()
        T.synchronize(device)
        best = float("inf")
        for _ in range(args.iters):
            t0 = time.monotonic()
            fn()
            T.synchronize(device)
            best = min(best, (time.monotonic() - t0) * 1e3)
        return best

    def make_idx(rows: int, n: int = 1) -> torch.Tensor:
        g = torch.Generator().manual_seed(0)
        return torch.randint(0, rows, (b, t * n), generator=g).to(torch.int32).to(device)

    def compiled(fn):
        if not args.compile:
            return fn
        return torch.compile(fn, backend="neuron" if device_type == "neuron" else "inductor",
                             dynamic=False)

    def report(tag: str, f_fwd, f_fwdbwd, rows: int, dim: int, ngather: int) -> None:
        """Time both callables, tolerating a lowering that simply does not compile."""
        out = {}
        for name, fn in (("fwd", f_fwd), ("fwd+bwd", f_fwdbwd)):
            try:
                out[name] = f"{timeit(fn):8.2f}ms"
            except Exception as e:  # an unsupported lowering is a RESULT here, not a crash
                out[name] = f"FAILED({type(e).__name__}: {str(e)[:60]})"
        if rank != 0:
            return
        bytes_out = ntok * ngather * dim * 2
        print(f"[eb] {tag:<44} rows={rows:>7} dim={dim:>4} gathers={ngather} "
              f"fwd={out['fwd']} fwd+bwd={out['fwd+bwd']} ({bytes_out / 1e6:6.1f} MB out)",
              flush=True)

    # ---- A/C/D: one table, several lowerings, over the shape grid --------------------
    grid = [(8192, 640), (8192, 128), (65536, 640), (65536, 128), (262144, 128)]
    for rows, dim in grid:
        idx = make_idx(rows)
        w = torch.zeros(rows, dim, dtype=dtype, device=device).normal_(std=0.02)

        emb = nn.Embedding(rows, dim, _weight=w.clone()).to(device)
        f_emb = compiled(lambda: emb(idx))

        def fb_emb():
            emb.zero_grad(set_to_none=True)
            emb(idx).float().sum().backward()
        fb_emb_c = compiled(fb_emb)
        report("nn.Embedding", f_emb, fb_emb_c, rows, dim, 1)

        p = torch.nn.Parameter(w.clone())
        flat = idx.reshape(-1).to(torch.int32)
        f_sel = compiled(lambda: torch.index_select(p, 0, flat))

        def fb_sel():
            if p.grad is not None:
                p.grad = None
            torch.index_select(p, 0, flat).float().sum().backward()
        report("index_select", f_sel, compiled(fb_sel), rows, dim, 1)

        f_adv = compiled(lambda: p[flat])

        def fb_adv():
            if p.grad is not None:
                p.grad = None
            p[flat].float().sum().backward()
        report("advanced indexing p[idx]", f_adv, compiled(fb_adv), rows, dim, 1)

        # One-hot matmul: only where the one-hot itself is not absurd (rows <= 8192 here
        # is already 8192x8192 = 134 MB in bf16; 65536 would be 1.07 GB).
        if rows <= 8192:
            def f_oh():
                oh = torch.nn.functional.one_hot(flat.long(), rows).to(dtype)
                return oh @ p

            def fb_oh():
                if p.grad is not None:
                    p.grad = None
                oh = torch.nn.functional.one_hot(flat.long(), rows).to(dtype)
                (oh @ p).float().sum().backward()
            report("one-hot matmul", compiled(f_oh), compiled(fb_oh), rows, dim, 1)

    # ---- B: N separate gathers vs one merged table -----------------------------------
    # This is the decisive one for T3.1: the five tables can trivially be merged into a
    # single [5*rows, dim] table, because each already uses its own prime pair and the
    # hash only needs `+ k*rows` added per table.
    for rows, dim, n in [(65536, 640, 5), (8192, 640, 5), (65536, 128, 5)]:
        tables = [nn.Embedding(rows, dim,
                               _weight=torch.zeros(rows, dim, dtype=dtype).normal_(std=0.02)).to(device)
                  for _ in range(n)]
        idxs = [make_idx(rows) for _ in range(n)]

        def f_sep():
            return [e(i) for e, i in zip(tables, idxs)]

        def fb_sep():
            for e in tables:
                e.zero_grad(set_to_none=True)
            out = sum(e(i).float().sum() for e, i in zip(tables, idxs))
            out.backward()
        report(f"{n}x separate nn.Embedding", compiled(f_sep), compiled(fb_sep), rows, dim, n)

        merged = nn.Embedding(n * rows, dim,
                              _weight=torch.zeros(n * rows, dim, dtype=dtype).normal_(std=0.02)).to(device)
        # [b, n*t] offset indices: table k occupies rows [k*rows, (k+1)*rows).
        moff = torch.cat([i + k * rows for k, i in enumerate(idxs)], dim=1)

        def f_merge():
            return merged(moff)

        def fb_merge():
            merged.zero_grad(set_to_none=True)
            merged(moff).float().sum().backward()
        report(f"1x merged nn.Embedding ({n} tables)", compiled(f_merge), compiled(fb_merge), n * rows, dim, n)

    if rank == 0:
        print("[eb] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
