#!/usr/bin/env python3
"""T3.1 follow-up: where is the ~15ms/table of UNEXPLAINED n-gram overhead?

Context (research/trn2-experiment-results.md, T3.1 section). After the `weight[idx]`
fix, the row gather+scatter for a (262144, 128) table measures 2.21-2.45ms fwd+bwd in
isolation, yet each n-gram table costs ~17.6ms of compiled micro time inside the model.
The residual (~15ms/table) is the hash + the dim-128 up-projection + the per-head gate +
the injection into V and their backwards -- which together move ~100MB (36us at 2.9TB/s)
and do ~1.3 GFLOP (~2us). Two named suspects:

  1. `torch.remainder(h, 262139)` -- the modulus is deliberately the largest PRIME below
     the (power-of-two) row count, so this is an int32 modulo by a non-power-of-two
     constant. Integer division is usually emulated; on a vector engine with an fp32
     datapath an EXACT 31-bit integer remainder cannot go through fp32 reciprocal at all
     (24-bit mantissa), so the compiler must either long-divide or fall back.
  2. The injection breaking a fusion the compiler was exploiting inside attention.

This file answers suspect 1 (and prices every other piece of the fragment) WITHOUT a
whole-model run, by rebuilding the exact fragment at the exact shapes and timing each
layer of it cumulatively. Suspect 2 is a whole-model question and is not answerable here
-- if the cumulative total lands far below ~15ms/table, the residual is suspect 2 and the
next step is a neuron-profile of the real compiled graph.

Arms (cumulative, each adds one piece; 3 bigram tables, the config the T3.1 grid picked):

  hash_prime   the 3 hashes only, `% 262139`               <- suspect 1 numerator
  hash_mask    the 3 hashes only, `& 262143` + xor-fold    <- suspect 1 denominator
  hash_none    indices are a precomputed constant buffer   <- floor for the hash phase
  +gather      ... plus weight[idx] on 3 (262144, 128) bf16 tables
  +proj        ... plus each table's Linear(128, 640)
  +gate        ... plus 2*sigmoid(Linear(32, 5)) per table
  +inject      ... plus v = v + gate.unsqueeze(-1) * ve   (the full fragment)

Each arm is one torch.compile'd callable, timed fwd-only and fwd+bwd, best-of-N.
`+inject` fwd+bwd minus `base` is the honest per-step price of the fragment.

Run (1 rank is enough for the ratios; the shapes are per-rank anyway):
    NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=1 \
        research/scratch/ngram_frag_bench.py
"""

import argparse
import json
import os
import sys
import time

REPO = os.environ.get("PROF_REPO", "/home/ubuntu/trainium_frontier")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rows-batch", type=int, default=4, help="micro-batch rows (train.py: 4)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--tables", type=int, default=3, help="bigram tables (T3.1 grid winner: 3)")
    ap.add_argument("--table-rows", type=int, default=262144)
    ap.add_argument("--table-dim", type=int, default=128)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--json-out", type=str, default="")
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
    n_embd, n_head, head_dim = 640, 5, 128
    gate_ch = T.VE_GATE_CHANNELS
    rows, dim = args.table_rows, args.table_dim
    mod = T._largest_prime_below(rows)
    mask = rows - 1
    assert rows & mask == 0, f"{rows} must be a power of two for the mask arm"
    ntab = args.tables

    if rank == 0:
        print(f"[nfb] b={b} t={t} tables={ntab} rows={rows} dim={dim} "
              f"prime_mod={mod} mask=0x{mask:x} compile={args.compile}", flush=True)

    # ---- inputs, all fixed so every arm sees identical data -------------------------
    g = torch.Generator().manual_seed(1234)
    idx_cpu = torch.randint(0, 8192, (b, t), generator=g)
    idx = idx_cpu.to(torch.int32).to(device)
    pad_id = 8192
    # x_v: the residual the gate reads (and, in the model, the thing c_v projects).
    x_v = torch.randn(b, t, n_embd, generator=g).to(dtype).to(device)
    # v: attention's value stream, the injection target.
    v_base = torch.randn(b, t, n_head, head_dim, generator=g).to(dtype).to(device)

    tables = [torch.randn(rows, dim, generator=g).to(dtype).to(device).requires_grad_(True)
              for _ in range(ntab)]
    projs = [torch.randn(n_embd, dim, generator=g).to(dtype).mul_(0.02).to(device).requires_grad_(True)
             for _ in range(ntab)]
    gates = [torch.randn(n_head, gate_ch, generator=g).to(dtype).mul_(0.02).to(device).requires_grad_(True)
             for _ in range(ntab)]
    primes = [T.ngram_bigram_primes(i) for i in range(ntab)]

    # Precomputed indices for the hash_none floor: same distribution, no hash math.
    const_idx = [torch.randint(0, mod, (b, t), generator=g).to(torch.int32).to(device)
                 for _ in range(ntab)]

    def shifted(cur):
        pad = torch.full_like(cur[:, :1], pad_id)
        return torch.cat((pad, cur[:, :-1]), dim=1)

    # ---- hash spellings --------------------------------------------------------------
    def hash_prime(cur):
        prev = shifted(cur)
        return [torch.remainder(torch.bitwise_xor(prev * p0, cur * p1), mod)
                for (p0, p1) in primes]

    def hash_mask(cur):
        """Power-of-two masked hash, no finalizer.

        `(a ^ b) & (2^k-1) == ((a & (2^k-1)) ^ (b & (2^k-1)))`, and for an ODD multiplier
        `p`, `id*p mod 2^18` is INJECTIVE over id < 2^14 -- so masking alone already gives
        the xor of two injective 18-bit hashes, not the "throw away all the mixing" case
        the prime modulus was chosen to avoid.
        """
        prev = shifted(cur)
        return [torch.bitwise_and(torch.bitwise_xor(prev * p0, cur * p1), mask)
                for (p0, p1) in primes]

    def hash_maskfold(cur):
        """... plus an xor-fold finalizer that carries h's discarded top 13 bits back in.

        `torch.bitwise_right_shift` does NOT legalize on this stack (Torch-MLIR ->
        StableHLO: "failed to legalize operation 'torch.operator'"), so the shift is
        spelled as a floor-division by a power of two, which is exact for h >= 0 (every
        product here is non-negative and < 2^31 by construction -- see NGRAM_HASH_PRIMES).
        """
        prev = shifted(cur)
        out = []
        for (p0, p1) in primes:
            h = torch.bitwise_xor(prev * p0, cur * p1)
            h = torch.bitwise_xor(h, torch.div(h, 1 << 18, rounding_mode="floor"))
            out.append(torch.bitwise_and(h, mask))
        return out

    def hash_none(_cur):
        return const_idx

    HASHES = {"prime": hash_prime, "mask": hash_mask,
              "maskfold": hash_maskfold, "none": hash_none}

    # ---- cumulative fragment stages ---------------------------------------------------
    def build(stage: str, hname: str):
        hfn = HASHES[hname]

        def fn():
            ids = hfn(idx)
            if stage == "hash":
                # int outputs have no grad; return a float scalar so fwd+bwd is well-formed
                # (the .float().sum() itself is one cheap reduce, identical across hash arms).
                return sum(i.float().sum() for i in ids)
            outs = [tables[k][ids[k]] for k in range(ntab)]
            if stage == "gather":
                return sum(o.float().sum() for o in outs)
            outs = [torch.nn.functional.linear(o, projs[k]) for k, o in enumerate(outs)]
            if stage == "proj":
                return sum(o.float().sum() for o in outs)
            gs = [2 * torch.sigmoid(torch.nn.functional.linear(
                x_v[..., k * gate_ch:(k + 1) * gate_ch], gates[k])) for k in range(ntab)]
            if stage == "gate":
                return sum(o.float().sum() for o in outs) + sum(gg.float().sum() for gg in gs)
            v = v_base
            for k in range(ntab):
                v = v + gs[k].unsqueeze(-1) * outs[k].view(b, t, n_head, head_dim)
            return v.float().sum()

        return torch.compile(fn, backend="neuron" if device_type == "neuron" else "inductor",
                             dynamic=False) if args.compile else fn

    def build_base():
        """Floor: just the reduce on v, so every arm's trailing .sum() is accounted for."""
        def fn():
            return v_base.float().sum()
        return torch.compile(fn, backend="neuron" if device_type == "neuron" else "inductor",
                             dynamic=False) if args.compile else fn

    params = tables + projs + gates

    def timeit(fn, bwd: bool) -> float:
        def once():
            out = fn()
            if bwd:
                for p in params:
                    p.grad = None
                out.backward()
        for _ in range(args.warmup):
            once()
        T.synchronize(device)
        best = float("inf")
        for _ in range(args.iters):
            t0 = time.monotonic()
            once()
            T.synchronize(device)
            best = min(best, (time.monotonic() - t0) * 1e3)
        return best

    results = {}

    def run(label, fn, bwd_ok=True):
        # One arm failing to legalize (or OOM'ing) must not abort the sweep -- the whole
        # point is to compare spellings, some of which are expected not to lower.
        rec = {}
        for key, bwd in (("fwd_ms", False), ("fwdbwd_ms", True)):
            if bwd and not bwd_ok:
                rec[key] = float("nan")
                continue
            try:
                rec[key] = timeit(fn, bwd)
            except Exception as exc:  # noqa: BLE001 -- deliberately broad, see above
                rec[key] = float("nan")
                rec.setdefault("errors", {})[key] = f"{type(exc).__name__}: {exc}"[:400]
        results[label] = rec
        if rank == 0:
            print(f"[nfb] {label:24s} fwd {rec['fwd_ms']:8.3f}ms   "
                  f"fwd+bwd {rec['fwdbwd_ms']:8.3f}ms", flush=True)
            for k, msg in rec.get("errors", {}).items():
                print(f"[nfb]   !! {k}: {msg}", flush=True)

    run("base(sum only)", build_base(), bwd_ok=False)
    # The hash is integer-only, so it has no backward in the model either -- fwd is the
    # whole cost and there is no grad_fn to call .backward() on.
    for hname in HASHES:
        run(f"hash-{hname}", build("hash", hname), bwd_ok=False)
    for stage in ("gather", "proj", "gate", "inject"):
        for hname in ("prime", "mask", "none"):
            run(f"{stage}-{hname}", build(stage, hname))

    if rank == 0:
        base = results["base(sum only)"]["fwd_ms"]
        print("\n[nfb] --- per-table fwd+bwd, minus the sum-only floor ---", flush=True)
        for label, r in results.items():
            if label.startswith("base"):
                continue
            print(f"[nfb] {label:24s} total {r['fwdbwd_ms'] - base:8.3f}ms  "
                  f"per-table {(r['fwdbwd_ms'] - base) / ntab:7.3f}ms", flush=True)
        dp = results["inject-prime"]["fwdbwd_ms"] - results["inject-mask"]["fwdbwd_ms"]
        hp = results["hash-prime"]["fwd_ms"] - results["hash-mask"]["fwd_ms"]
        print(f"[nfb] SUSPECT 1 (hash only, fwd): {hp:+.3f}ms total, {hp / ntab:+.3f}ms/table", flush=True)
        print(f"\n[nfb] SUSPECT 1 (remainder vs mask), full fragment fwd+bwd: "
              f"{dp:+.3f}ms total, {dp / ntab:+.3f}ms/table", flush=True)
        if args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump({"config": vars(args), "mod": mod, "results": results}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
