#!/usr/bin/env python3
"""2026-09-15 RESHARD HUNT, option 1: a NKI indirect-gather kernel for the n-gram table read.

WHY A KERNEL AT ALL. The gather itself is not slow. `table[idx]` and this kernel both move the
same 2 MiB as 8192 x 256 B DMA descriptors -- that is a hardware floor for a [V, 128] bf16 table
(a gathered row IS 256 B contiguous, so it is one descriptor, and NeuronCore-v3 has no indirect
HWDGE so it is SWDGE with GpSimdE generating the descriptors). The kernel exists for exactly one
reason: to put the DYNAMIC ADDRESS INSIDE AN OPAQUE ISLAND so the surrounding neuronx-cc graph
never sees one, and therefore does not re-shard the residual/V stream across the two physical
cores of the LNC=2 logical core for every layer downstream of the injection point.

WHAT THE NKI API ALLOWS (nki 0.5.0, verified against the docs before writing this):
  FORWARD  -- expressible. `nisa.dma_copy(dst=sbuf_tile, src=table.ap(pattern=[[D, P], [1, D]],
              vector_offset=row_ids_sbuf, indirect_dim=0))`. `vector_offset` is one index per
              SBUF partition, so at most 128 rows per instruction => BT/128 = 64 instructions.
              Indirect access forces SWDGE (`trainium2_arch`: "NeuronCore-v3 hardware DGE
              currently does not support indirect DMA operations").
  BACKWARD -- NOT expressible. The n-gram backward is a scatter-add into a dense [V, D] grad with
              DUPLICATE indices (8192 hashed rows out of 1048576 collide, and must accumulate).
              Indirect `dma_copy` is documented as unique-indices-only; `dst_rmw_op` was REMOVED
              from `dma_copy` in NKI 0.3.0; its replacement `nisa.dma_compute` requires
              compile-time-known `dst` addresses (so it can gather-accumulate, never
              scatter-accumulate); and the Trn2 ISA has no atomic add reaching HBM (PSUM
              accumulation is TensorE-controlled, on-chip, statically addressed). Trn3's
              NeuronCore-v4 adds indirect scatter; Trn2 does not have it.
So this is a FORWARD-ONLY kernel, and the backward stays in PyTorch. That is not a compromise
imposed by laziness: the 2026-09-12 attribution put +47.0 ms of the +52.6 ms penalty in the
FORWARD NEFF and only +5.6 ms in the backward one, so a forward-only launder is where ~89% of
the theoretical recovery lives -- IF any of it still exists at four ranks.

WHAT THIS SCRIPT CHECKS, in order, so a failure localises itself:
  1. does the kernel compile and run at all on device (the `vector_offset` API is barely
     documented outside one MoE example, and its scaling convention is the likely first wall);
  2. is `vector_offset` in ROWS or in ELEMENTS -- tested with a table pre-filled so table[r, :]
     == r, which discriminates unambiguously;
  3. BIT-EXACTNESS of the forward against `table[idx]` at the real locked shape
     (V = 1048576, D = 128, BT = 8192, bf16). A row gather is a pure copy, so anything other
     than an exact match is a bug, not a rounding difference;
  4. bit-exactness of the autograd wrapper's BACKWARD against the reference `table[idx]`
     backward. The wrapper deliberately uses `index_put_(accumulate=True)` on a zeros tensor,
     which is the same aten call `index_backward` itself makes, so this must also be exact;
  5. wall-clock of the kernel vs the reference gather, in isolation (NOT the number that
     matters -- the whole point is the whole-graph resharding effect, which only
     reshard_probe.sh can see -- but a kernel that is 10x slower in isolation is dead anyway).

Run:  NEURON_LOGICAL_NC_CONFIG=2 python research/scratch/nki_ngram_gather_dev.py
"""
from __future__ import annotations

import argparse
import time

import torch

_NKI_PARTITION = 128

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    _HAS_NKI = True
except Exception as _exc:  # noqa: BLE001
    print(f"[nkg] NKI import failed: {_exc}")
    _HAS_NKI = False


if _HAS_NKI:

    @nki.jit
    def _ngram_gather_fwd(table, idx):
        """out[i, :] = table[idx[i], :].

        table: [V, D] HBM (bf16 in the real config). V may be ~1e6.
        idx:   [BT, 1] HBM int32, values in [0, V). BT must be a multiple of 128.
        out:   [BT, D] HBM, table's dtype.
        """
        _V, D = table.shape
        BT = idx.shape[0]
        P = _NKI_PARTITION
        out = nl.ndarray((BT, D), dtype=table.dtype, buffer=nl.shared_hbm)
        for t in nl.affine_range(BT // P):
            row0 = t * P
            # One index per SBUF partition -- vector_offset MUST live in SBUF.
            row_ids = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=row_ids, src=idx[nl.ds(row0, P), 0:1])
            # The indirect read. dim0's iteration index is REPLACED by vector_offset[p], and the
            # DMA engine scales it by dim0's stride (D), so partition p reads table[idx[p], 0:D].
            rows = nl.ndarray((P, D), dtype=table.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=rows,
                          src=table.ap(pattern=[[D, P], [1, D]], offset=0,
                                       vector_offset=row_ids, indirect_dim=0))
            nisa.dma_copy(dst=out[nl.ds(row0, P), 0:D], src=rows)
        return out

    @nki.jit
    def _ngram_gather_fwd_elemscale(table, idx_scaled):
        """Same kernel, for the case where vector_offset turns out to be in ELEMENTS.

        Then the caller must pass idx * D and the pattern's indirect dim needs stride 1 so the
        engine does not scale twice. Kept as a separate kernel rather than a branch so whichever
        convention is real is settled by which of the two matches `table[idx]`.
        """
        _V, D = table.shape
        BT = idx_scaled.shape[0]
        P = _NKI_PARTITION
        out = nl.ndarray((BT, D), dtype=table.dtype, buffer=nl.shared_hbm)
        for t in nl.affine_range(BT // P):
            row0 = t * P
            row_ids = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=row_ids, src=idx_scaled[nl.ds(row0, P), 0:1])
            rows = nl.ndarray((P, D), dtype=table.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=rows,
                          src=table.ap(pattern=[[1, P], [1, D]], offset=0,
                                       vector_offset=row_ids, indirect_dim=0))
            nisa.dma_copy(dst=out[nl.ds(row0, P), 0:D], src=rows)
        return out


class NgramGatherNKI(torch.autograd.Function):
    """`table[idx]` with the dynamic address hidden inside a NKI island.

    Backward is deliberately NOT a kernel: see the module docstring for the four independent
    reasons a duplicate-index scatter-add is inexpressible in NKI 0.5.0 on Trn2. It is instead
    the exact aten call `index_backward` itself makes -- `zeros.index_put_((idx,), g,
    accumulate=True)` -- so the gradient is bit-identical to the shipped path's, and the dynamic
    address it reintroduces lives only in the backward graph, which the 2026-09-12 attribution
    priced at +5.6 ms against the forward's +47.0 ms.
    """

    @staticmethod
    def forward(ctx, table, idx_flat):
        ctx.save_for_backward(idx_flat)
        ctx.table_shape = tuple(table.shape)
        ctx.table_dtype = table.dtype
        return _ngram_gather_fwd(table, idx_flat.reshape(-1, 1))

    @staticmethod
    def backward(ctx, grad_out):
        (idx_flat,) = ctx.saved_tensors
        g = torch.zeros(ctx.table_shape, dtype=grad_out.dtype, device=grad_out.device)
        g.index_put_((idx_flat.long(),), grad_out, accumulate=True)
        return g.to(ctx.table_dtype), None


def pick_device() -> torch.device:
    try:
        import torch_neuronx  # noqa: F401
        return torch.device("neuron")
    except Exception:
        return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1048576)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--bt", type=int, default=8192)
    ap.add_argument("--small-rows", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()
    if not _HAS_NKI:
        print("[nkg] FATAL: no NKI")
        return
    dev = pick_device()
    print(f"[nkg] device={dev} V={args.rows} D={args.dim} BT={args.bt}", flush=True)

    # --- CHECK 1+2: does it run, and is vector_offset in rows or elements? -------------
    # Small table, table[r, :] = r, so the returned VALUE names the row that was read.
    Vs, D, BT = args.small_rows, args.dim, _NKI_PARTITION * 2
    tbl_s = (torch.arange(Vs, dtype=torch.float32).unsqueeze(1)
             .expand(Vs, D).contiguous().to(torch.bfloat16).to(dev))
    idx_s = torch.randint(0, Vs, (BT,), dtype=torch.int32).to(dev)
    ref_s = tbl_s[idx_s]
    which = None
    for name, fn, ix in (("rows", _ngram_gather_fwd, idx_s),
                         ("elements", _ngram_gather_fwd_elemscale, idx_s * D)):
        try:
            got = fn(tbl_s, ix.reshape(-1, 1).contiguous())
            ok = torch.equal(got.float().cpu(), ref_s.float().cpu())
            print(f"[nkg] CHECK1/2 vector_offset={name:>8}: ran OK, exact={ok}", flush=True)
            if ok and which is None:
                which = name
        except Exception as exc:  # noqa: BLE001
            print(f"[nkg] CHECK1/2 vector_offset={name:>8}: FAILED "
                  f"{type(exc).__name__}: {str(exc)[:400]}", flush=True)
    if which is None:
        print("[nkg] WALL: neither offset convention reproduced table[idx]. "
              "Forward gather NOT expressible as written -- stop here.", flush=True)
        return
    print(f"[nkg] => vector_offset convention is {which.upper()}", flush=True)
    kern = _ngram_gather_fwd if which == "rows" else _ngram_gather_fwd_elemscale
    scale = 1 if which == "rows" else D

    # --- CHECK 3: bit-exact forward at the REAL locked shape ---------------------------
    V = args.rows
    tbl = torch.randn(V, D).to(torch.bfloat16).to(dev)
    idx = torch.randint(0, V, (args.bt,), dtype=torch.int32).to(dev)
    ref = tbl[idx]
    got = kern(tbl, (idx * scale).reshape(-1, 1).contiguous())
    exact = torch.equal(got.float().cpu(), ref.float().cpu())
    md = (got.float().cpu() - ref.float().cpu()).abs().max().item()
    print(f"[nkg] CHECK3 forward @ V={V} D={D} BT={args.bt}: exact={exact} max|diff|={md:.3e}",
          flush=True)

    # --- CHECK 4: bit-exact backward through the autograd wrapper ----------------------
    if scale != 1:
        print("[nkg] CHECK4 skipped: autograd wrapper assumes the ROWS convention", flush=True)
    else:
        g_out = torch.randn(args.bt, D).to(torch.bfloat16).to(dev)
        t_ref = tbl.clone().requires_grad_(True)
        (t_ref[idx] * g_out).sum().backward()
        t_nki = tbl.clone().requires_grad_(True)
        (NgramGatherNKI.apply(t_nki, idx) * g_out).sum().backward()
        bex = torch.equal(t_ref.grad.float().cpu(), t_nki.grad.float().cpu())
        bmd = (t_ref.grad.float().cpu() - t_nki.grad.float().cpu()).abs().max().item()
        print(f"[nkg] CHECK4 backward: exact={bex} max|diff|={bmd:.3e}", flush=True)

    # --- CHECK 5: isolated wall clock --------------------------------------------------
    idx_k = (idx * scale).reshape(-1, 1).contiguous()

    def t(fn, n=args.iters):
        for _ in range(2):
            fn()
        t0 = time.monotonic()
        for _ in range(n):
            fn()
        return (time.monotonic() - t0) / n * 1e3

    ms_ref = t(lambda: tbl[idx])
    ms_nki = t(lambda: kern(tbl, idx_k))
    print(f"[nkg] CHECK5 isolated fwd: reference table[idx] {ms_ref:7.3f} ms   "
          f"NKI kernel {ms_nki:7.3f} ms   ratio {ms_nki / ms_ref:.2f}x", flush=True)
    print("[nkg] done", flush=True)


if __name__ == "__main__":
    main()
