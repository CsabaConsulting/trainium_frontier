"""Fused causal attention BACKWARD for the P1 NKI port (train.py:2561-2568).

Math (per (batch, head), scale s, P = softmax(QK^T s), O = PV, D = rowsum(dO*O)):
  dV = P^T dO            dP = dO V^T
  dS = P * (dP - D)      dQ = dS K s        dK = dS^T Q s

Design: RECOMPUTE (flash-attention style) -- no change to the validated fwd
kernel, no [BH*T, T] P buffer in HBM. The qi-outer loop recomputes the fp32
softmax block with the verbatim fwd code path (same ops => bit-deterministic
vs fwd), then:
  dQ[q,:] is per-row independent -> PSUM-accumulate over kv chunks, store.
  dK / dV need cross-qi accumulation -> fp32 HBM scratch bufs (zeroed by the
  host), load-add-store per (qi, kv-chunk), downcast epilogue in-kernel.

Every op below is fwd-proven (dma/transpose/matmul/scalar/tensor/sum/exp/
reciprocal). ALL nc_transpose inputs are bf16 (fwd only ever transposed
bf16); dS/dP stay fp32 until the downcast copy that feeds a transpose or a
matmul stationary/moving operand. nl.store is never used for fp32->bf16
(explicit copy + dma, fwd lesson).

NKI 0.5.0 (box version): affine_range/sequential Python-range loops only.

Run on hardware (needs Neuron device):
  python3 research/scratch/nki_attn_bwd.py --selftest   # numpy block-math
  + hardware check when --hw is passed (imports nki).
"""

import argparse
import sys

import numpy as np

P = 128          # partition dim == head dim
KV_TILE = 512    # kv tiling (PSUM moving free-dim limit)
Q_TILE = 128     # q tiling (stationary free dim)


def build_causal_mask(t):
    """[T, T] bf16, 1.0 where kv <= query (keep), else 0.0."""
    m = (np.arange(t)[None, :] <= np.arange(t)[:, None]).astype(np.float32)
    return m.astype(np.float16)


def ref_bwd(q, k, v, do, scale, mask):
    """Numpy fp32 reference. q/k/v/do: [T, D]. Returns dQ,dK,dV [T, D]."""
    q32, k32, v32 = (x.astype(np.float32) for x in (q, k, v))
    s = (q32 @ k32.T) * scale + np.where(mask.astype(bool), 0, -1e30)
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    p = e / e.sum(axis=1, keepdims=True)
    o = p @ v32
    d = (do.astype(np.float32) * o).sum(axis=1, keepdims=True)
    dp = do.astype(np.float32) @ v32.T
    ds = p * (dp - d)
    return ds @ k32 * scale, ds.T @ q32 * scale, p.T @ do.astype(np.float32)


def numpy_block_math(q, k, v, do, scale, mask):
    """Mirror of the kernel loop structure in numpy: per-qi recompute, fp32
    HBM-style dK/dV accumulation, bf16 rounding at every downcast point the
    kernel has (matmul inputs, stored rows). Returns dQ,dK,dV."""
    T, D = q.shape
    b16 = lambda x: x.astype(np.float16).astype(np.float32)
    dk = np.zeros_like(q, dtype=np.float32)
    dv = np.zeros_like(q, dtype=np.float32)
    dq = np.zeros_like(q, dtype=np.float32)
    qt, kt = q.T.copy(), k.T.copy()  # [D, T] assembled tiles
    for r0 in range(0, T, Q_TILE):
        qs = qt[:, r0:r0 + Q_TILE]                    # [D, Q]
        qk = (b16(qs).T @ b16(kt)) * scale
        qk = np.where(mask[r0:r0 + Q_TILE], qk, -1e30)
        qk -= qk.max(axis=1, keepdims=True)
        e = np.exp(qk)
        e = np.where(mask[r0:r0 + Q_TILE], e, 0.0)
        ps = e / e.sum(axis=1, keepdims=True)          # fp32 P block
        do_b = do[r0:r0 + Q_TILE]
        o_b = ps @ v
        d = (do_b * o_b).sum(axis=1, keepdims=True)
        for c in range(0, T, P):
            v_c = v[c:c + P]                           # [P, D] native rows
            dv[c:c + P] += b16(ps[:, c:c + P]).T @ b16(do_b)
            dp = b16(do_b) @ b16(v_c).T
            ds = b16(ps[:, c:c + P]) * (dp - d)
            dq[r0:r0 + Q_TILE] += b16(ds).T @ b16(k[c:c + P]) * scale
            dk[c:c + P] += b16(ds) @ b16(q[r0:r0 + Q_TILE]) * scale
    return dq, dk, dv


try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    _HAS_NKI = True
except ImportError:  # numpy-only hosts: --selftest still works
    _HAS_NKI = False


if _HAS_NKI:
    @nki.jit
    # NOTE: T/D are REQUIRED params (no defaults), matching all train.py
    # NKI kernels; all callers pass them explicitly.
    def nki_attn_bwd(q, k, v, do, mask, scale, dq, dk, dv, dk_buf, dv_buf,
                     T, D):
        """q/k/v/do: [BH*T, D] bf16 HBM. mask: [T, T] bf16 HBM.
        dq/dk/dv: [BH*T, D] bf16 HBM outputs. dk_buf/dv_buf: [BH*T, D]
        fp32 HBM scratch, zeroed by host (cross-qi accumulation)."""
        BH = q.shape[0] // T
        for bh in nl.affine_range(BH):
            base = bh * T
            # --- assemble [P, T] Q^T / K^T (verbatim fwd path)
            q_sbuf = nl.ndarray((P, T), dtype=q.dtype, buffer=nl.sbuf)
            k_sbuf = nl.ndarray((P, T), dtype=q.dtype, buffer=nl.sbuf)
            for c in range(T // P):
                qc = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=qc, src=q[base + c * P:base + (c + 1) * P, :])
                qt = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=qt, data=qc)
                nisa.tensor_copy(dst=q_sbuf[0:P, c * P:(c + 1) * P], src=qt)
                kc = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=kc, src=k[base + c * P:base + (c + 1) * P, :])
                kt = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=kt, data=kc)
                nisa.tensor_copy(dst=k_sbuf[0:P, c * P:(c + 1) * P], src=kt)
            for qi in range(T // Q_TILE):
                r0 = qi * Q_TILE
                # --- recompute fp32 P block (verbatim fwd path)
                n_kv = T // KV_TILE
                qk_tiles = []
                for _i in range(n_kv):
                    qk_tiles.append(nl.ndarray((P, KV_TILE), dtype=nl.float32,
                                               buffer=nl.psum))
                for ki in range(n_kv):
                    nisa.nc_matmul(
                        dst=qk_tiles[ki],
                        stationary=q_sbuf[0:P, r0:r0 + Q_TILE],
                        moving=k_sbuf[0:P, ki * KV_TILE:(ki + 1) * KV_TILE])
                qk_full = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
                for ki in range(n_kv):
                    nisa.tensor_copy(
                        dst=qk_full[0:P, ki * KV_TILE:(ki + 1) * KV_TILE],
                        src=qk_tiles[ki])
                nisa.tensor_scalar(dst=qk_full, data=qk_full, op0=nl.multiply,
                                   operand0=scale)
                row_max = nl.max(qk_full, axis=1, keepdims=True)
                sub_full = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=sub_full, data=qk_full, op0=nl.subtract,
                                   operand0=row_max)
                exp_full = nl.exp(sub_full)
                mb = nl.ndarray((P, T), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=mb, src=mask[r0:r0 + Q_TILE, :])
                m = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=m, src=mb)
                nisa.tensor_tensor(dst=exp_full, data1=exp_full, data2=m,
                                   op=nl.multiply)
                sum_row = nl.sum(exp_full, axis=1, keepdims=True)
                inv = nl.reciprocal(sum_row)
                p_full = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=p_full, data=exp_full, op0=nl.multiply,
                                   operand0=inv)
                p16 = nl.ndarray((P, T), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=p16, src=p_full)
                # --- dO rows for this block
                do_b = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=do_b, src=do[base + r0:base + r0 + Q_TILE, :])
                # --- recompute O block (fwd AV pattern, fp32 PSUM accum)
                o_acc = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                for c in range(T // P):
                    st = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=st, data=p16[0:P, c * P:(c + 1) * P])
                    st_s = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=st_s, src=st)
                    vc = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=vc,
                                  src=v[base + c * P:base + (c + 1) * P, :])
                    nisa.nc_matmul(dst=o_acc, accumulate=(c > 0),
                                   stationary=st_s, moving=vc)
                o_b = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_b, src=o_acc)
                # --- D = rowsum(dO * O)
                do_f = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=do_f, src=do_b)
                do_o = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=do_o, data1=do_f, data2=o_b,
                                   op=nl.multiply)
                d_row = nl.sum(do_o, axis=1, keepdims=True)
                # --- Q rows for the dK contributions of this block
                q_b = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_b, src=q[base + r0:base + r0 + Q_TILE, :])
                # --- dQ accumulates over kv chunks in PSUM
                dq_acc = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                for c in range(T // P):
                    vc = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=vc,
                                  src=v[base + c * P:base + (c + 1) * P, :])
                    # dV contrib: P^T @ dO -> fp32 HBM scratch add.
                    # Contraction is over q (the shared partition dim),
                    # so stationary is the P block NATIVE [q, k] rows --
                    # a transpose here contracts k against q (silent
                    # garbage: hw FAIL dV 1.16, found 2026-09-17).
                    dv_c = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dv_c,
                                   stationary=p16[0:P, c * P:(c + 1) * P],
                                   moving=do_b)
                    dv_old = nl.ndarray((P, P), dtype=nl.float32,
                                        buffer=nl.sbuf)
                    nisa.dma_copy(dst=dv_old,
                                  src=dv_buf[base + c * P:base + (c + 1) * P, :])
                    dv_new = nl.ndarray((P, P), dtype=nl.float32,
                                        buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dv_new, src=dv_c)
                    nisa.tensor_tensor(dst=dv_new, data1=dv_new, data2=dv_old,
                                       op=nl.add)
                    nisa.dma_copy(dst=dv_buf[base + c * P:base + (c + 1) * P, :],
                                  src=dv_new)
                    # dP = dO @ V^T: contraction is over d, so V MUST
                    # be transposed to [d, k] rows (native [k, d] rows
                    # contract k against d: silent garbage poisons dS
                    # and through it dQ/dK, found 2026-09-17).
                    do_t = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=do_t, data=do_b)
                    do_ts = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=do_ts, src=do_t)
                    vc_t = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=vc_t, data=vc)
                    vc_ts = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=vc_ts, src=vc_t)
                    dp = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dp, stationary=do_ts, moving=vc_ts)
                    dp_s = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dp_s, src=dp)
                    # dS = P * (dP - D), pre-scaled by s (serves dQ and dK)
                    ds = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=ds, data=dp_s, op0=nl.subtract,
                                       operand0=d_row)
                    pblk = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=pblk,
                                     src=p_full[0:P, c * P:(c + 1) * P])
                    nisa.tensor_tensor(dst=ds, data1=ds, data2=pblk,
                                       op=nl.multiply)
                    nisa.tensor_scalar(dst=ds, data=ds, op0=nl.multiply,
                                       operand0=scale)
                    ds16 = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ds16, src=ds)
                    # dQ contrib: dS^T @ K
                    kc = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=kc,
                                  src=k[base + c * P:base + (c + 1) * P, :])
                    dst_t = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=dst_t, data=ds16)
                    dst_s = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dst_s, src=dst_t)
                    nisa.nc_matmul(dst=dq_acc, accumulate=(c > 0),
                                   stationary=dst_s, moving=kc)
                    # dK contrib: dS @ Q -> fp32 HBM scratch add
                    dk_c = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dk_c, stationary=ds16, moving=q_b)
                    dk_old = nl.ndarray((P, P), dtype=nl.float32,
                                        buffer=nl.sbuf)
                    nisa.dma_copy(dst=dk_old,
                                  src=dk_buf[base + c * P:base + (c + 1) * P, :])
                    dk_new = nl.ndarray((P, P), dtype=nl.float32,
                                        buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dk_new, src=dk_c)
                    nisa.tensor_tensor(dst=dk_new, data1=dk_new, data2=dk_old,
                                       op=nl.add)
                    nisa.dma_copy(dst=dk_buf[base + c * P:base + (c + 1) * P, :],
                                  src=dk_new)
                dq_s = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dq_s, src=dq_acc)
                dq16 = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dq16, src=dq_s)
                nisa.dma_copy(dst=dq[base + r0:base + r0 + Q_TILE, :], src=dq16)
            # --- epilogue: downcast fp32 scratch bufs to bf16 outputs
            # (unrolled x2: the compiler rejects tuple-destructuring loops
            # over kernel tensors with "expecting simple variable").
            for c in range(T // P):
                fblk = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=fblk, src=dk_buf[base + c * P:base + (c + 1) * P, :])
                hblk = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=hblk, src=fblk)
                nisa.dma_copy(
                    dst=dk[base + c * P:base + (c + 1) * P, :], src=hblk)
                nisa.dma_copy(
                    dst=fblk, src=dv_buf[base + c * P:base + (c + 1) * P, :])
                nisa.tensor_copy(dst=hblk, src=fblk)
                nisa.dma_copy(
                    dst=dv[base + c * P:base + (c + 1) * P, :], src=hblk)
        return dq, dk, dv


def _selftest():
    rng = np.random.default_rng(0)
    T, D, scale = 256, 128, 1.0 / np.sqrt(128)
    q = (rng.standard_normal((T, D)) / 8).astype(np.float16)
    k = (rng.standard_normal((T, D)) / 8).astype(np.float16)
    v = (rng.standard_normal((T, D)) / 8).astype(np.float16)
    do = (rng.standard_normal((T, D)) / 8).astype(np.float16)
    mask = build_causal_mask(T)
    r = ref_bwd(q, k, v, do, scale, mask)
    b = numpy_block_math(q.astype(np.float32), k.astype(np.float32),
                         v.astype(np.float32), do.astype(np.float32),
                         scale, mask)
    names = ("dQ", "dK", "dV")
    ok = True
    for n, x, y in zip(names, r, b):
        err = np.abs(x - y).max()
        print(f"{n}: block-math vs fp32-ref max-abs {err:.6f}")
        ok &= err < 0.05
    print("BLOCK-MATH", "PASS" if ok else "FAIL")
    return ok


def _hwtest():
    """On-Neuron check: kernel dQ/dK/dV vs fp32 numpy ref from the same
    quantized inputs (bh-major rows, like the fwd harness)."""
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nki_attn_equiv import check
    from ml_dtypes import bfloat16
    if not _HAS_NKI:
        raise SystemExit("nki import failed; --hw needs a Neuron host")
    rng = np.random.default_rng(1)
    allok = True
    for dt in (bfloat16, np.float32):
        for (B, T, H) in [(1, 2048, 1), (1, 2048, 5)]:
            D = 128
            gen = lambda: (rng.standard_normal((B, T, H, D)) * 0.5)
            qb, kb, vb, db = gen(), gen(), gen(), gen() * 0.5
            flat = lambda x: x.transpose(0, 2, 1, 3).reshape(
                B * H * T, D).astype(dt)
            q, k, v, do = flat(qb), flat(kb), flat(vb), flat(db)
            mask = build_causal_mask(T).astype(
                dt if dt is not np.float32 else np.float32)
            z16 = np.zeros((B * H * T, D), dtype=dt)
            z32 = np.zeros((B * H * T, D), dtype=np.float32)
            got = nki_attn_bwd(q, k, v, do, mask, 1.0 / np.sqrt(D),
                               z16.copy(), z16.copy(), z16.copy(),
                               z32.copy(), z32.copy(), T, D)
            qf = lambda x: x.reshape(B, H, T, D).transpose(
                0, 2, 1, 3).astype(np.float32)
            # per-head fp32 ref from the same quantized inputs
            want = [ref_bwd(qf(q)[b, :, h], qf(k)[b, :, h], qf(v)[b, :, h],
                            qf(do)[b, :, h], 1.0 / np.sqrt(D),
                            mask.astype(np.float32))
                    for b in range(B) for h in range(H)]
            for name, gi in zip(("dQ", "dK", "dV"), range(3)):
                g = np.array(got[gi]).reshape(B, H, T, D).transpose(0, 2, 1, 3)
                w = np.stack([[want[b * H + h][gi] for h in range(H)]
                              for b in range(B)]).transpose(0, 2, 1, 3)
                tag = f"nki-bwd-vs-ref {name} {np.dtype(dt).name} B{B}T{T}H{H}"
                allok &= check(g, w, tag, rel_gate=(dt is np.float32))
    print("HWTEST PASS" if allok else "HWTEST FAIL", flush=True)
    return allok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--hw", action="store_true",
                    help="run the NKI kernel on Neuron hardware")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if args.hw:
        raise SystemExit(0 if _hwtest() else 1)
    print("pass --selftest (numpy) or --hw (needs Neuron device)", flush=True)
