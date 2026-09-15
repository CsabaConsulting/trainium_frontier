#!/usr/bin/env python3
"""2026-09-15 RESHARD HUNT, option 2: price the STATIC one-hot matmul against the dynamic gather.

The dynamic-address gather `table[idx]` is what triggers the LNC=2 cross-physical-core reshard.
A one-hot matmul is the same mathematical operation with COMPILE-TIME KNOWN addresses:

    out[i, :] = table[idx[i], :]   ==   (onehot(idx) @ table)[i, :]

so it cannot trigger the mechanism at all. The question is only whether its FLOP and memory cost
is payable at the row count the quality result needs. The locked config is
NGRAM_VE_TABLE_MULT=128 -> R = 128 * 8192 = 1,048,576 rows x 128 wide, four tables (b1/t1/b5/t5).

WHAT THIS MEASURES, per (R, tile) point, on device, eager, bf16, at the real BT = 8192:
  dyn_fwd / dyn_fwdbwd    -- `table[idx]`, the shipped spelling
  oh_fwd  / oh_fwdbwd     -- tiled `(idx_tile[:, None] == arange(R)) @ table`

The one-hot is built by COMPARISON against `arange(R)`, not by `scatter_`/`F.one_hot`: a scatter
is itself a dynamic write and would defeat the entire purpose, so the comparison form is the only
honest implementation of this idea. Its cost is included in the timing.

Tiling over BT is mandatory, not an optimisation: a full [8192, 1048576] bf16 one-hot is 17.2 GB
against ~24 GB of per-logical-core HBM, and four tables would need 68.7 GB. The tile size is
therefore part of the design, and the script sweeps it.

Run:  NEURON_LOGICAL_NC_CONFIG=2 python research/scratch/onehot_gather_bench.py
(single process on purpose -- this prices an isolated operation, and nothing here depends on
world size. The RESHARD cost is measured separately at 4 ranks by reshard_probe.sh.)
"""
from __future__ import annotations

import argparse
import json
import time

import torch

DIM = 128
BT = 8192  # DEVICE_BATCH_SIZE(1) * PACK_FACTOR(4) * SEQ_LEN(2048), per rank


def pick_device() -> torch.device:
    try:
        import torch_neuronx  # noqa: F401
        return torch.device("neuron")
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "neuron":
        torch.neuron.synchronize() if hasattr(torch, "neuron") else None


def timeit(fn, dev, iters: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.monotonic()
    for _ in range(iters):
        fn()
    sync(dev)
    return (time.monotonic() - t0) / iters * 1e3  # ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=str, default="8192,32768,131072,1048576")
    ap.add_argument("--tiles", type=str, default="128,512")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    dev = pick_device()
    dtype = torch.bfloat16
    print(f"[oh] device={dev} dtype={dtype} BT={BT} DIM={DIM}", flush=True)
    gen = torch.Generator().manual_seed(7)
    results = []

    for R in (int(v) for v in args.rows.split(",")):
        idx = torch.randint(0, R, (BT,), generator=gen).to(dev)
        tbl = torch.randn(R, DIM, generator=gen).to(dtype).to(dev)

        def dyn_fwd(_t=tbl, _i=idx):
            return _t[_i]

        tblg = tbl.clone().requires_grad_(True)

        def dyn_fwdbwd(_t=tblg, _i=idx):
            out = _t[_i]
            out.sum().backward()
            _t.grad = None

        d_f = timeit(dyn_fwd, dev, args.iters)
        try:
            d_fb = timeit(dyn_fwdbwd, dev, args.iters)
        except Exception as exc:  # noqa: BLE001
            d_fb = float("nan")
            print(f"[oh] R={R} dyn fwd+bwd failed: {type(exc).__name__}: {exc}", flush=True)

        row = {"rows": R, "dyn_fwd_ms": d_f, "dyn_fwdbwd_ms": d_fb,
               "flops_fwd": 2 * BT * R * DIM,
               "onehot_full_bytes": BT * R * 2}
        print(f"[oh] R={R:>9,}  dyn fwd {d_f:8.3f} ms   fwd+bwd {d_fb:8.3f} ms   "
              f"onehot-matmul FLOPs/fwd {2 * BT * R * DIM:.3e}  full one-hot "
              f"{BT * R * 2 / 2**30:.2f} GiB", flush=True)

        ar = torch.arange(R, device=dev)
        for tile in (int(v) for v in args.tiles.split(",")):
            if tile > BT:
                continue
            oh_bytes = tile * R * 2

            def oh_fwd(_t=tbl, _i=idx, _ar=ar, _tile=tile, _R=R):
                outs = []
                for s in range(0, BT, _tile):
                    ib = _i[s:s + _tile]
                    oh = (ib.unsqueeze(1) == _ar.unsqueeze(0)).to(_t.dtype)
                    outs.append(oh @ _t)
                return torch.cat(outs, 0)

            try:
                o_f = timeit(oh_fwd, dev, max(1, args.iters // 2), warmup=1)
            except Exception as exc:  # noqa: BLE001
                o_f = float("nan")
                print(f"[oh]   tile={tile} one-hot fwd FAILED: "
                      f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
            row[f"oh_fwd_ms_tile{tile}"] = o_f
            row[f"oh_tile_bytes_tile{tile}"] = oh_bytes
            ratio = o_f / d_f if d_f and o_f == o_f else float("nan")
            print(f"[oh]   tile={tile:>5} one-hot fwd {o_f:9.3f} ms  "
                  f"({ratio:7.1f}x the dynamic gather)  tile one-hot "
                  f"{oh_bytes / 2**20:8.1f} MiB", flush=True)
        results.append(row)

    print("[oh] RESULTS " + json.dumps(results), flush=True)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
