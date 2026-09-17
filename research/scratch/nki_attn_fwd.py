"""Fused causal attention forward for the P1 NKI port (train.py:2561-2568).

Spec: per (batch, head) group, Q/K/V are [T=2048, D=128] bf16 (row-major in
HBM as one [BH*T, D] matrix). Computes causal softmax(Q @ K.T * scale) @ V
in bf16, matching F.scaled_dot_product_attention(is_causal=True).

Layout follows attention_fwd_v7 (single-head [128, S] tiles, P=d_head):
inputs are assembled on-chip from [T, D] row chunks via load + nc_transpose.
Softmax is v1-style explicit fp32 (max / subtract / exp / mask-mul / sum /
normalize) -- no fused-bias semantics to misread. Causal mask is an explicit
[T, T] bf16 HBM input (1.0 keep / 0.0 drop); exp(x)*0 == exp(-inf) == 0, so
no -inf handling is needed anywhere.

NKI 0.5.0 (box version): affine_range/sequential Python-range loops only,
tensor_scalar accepts tensor operands with broadcast (v1-proven).

Run on hardware (needs Neuron device):
  python3 research/scratch/nki_attn_fwd.py --selftest   # numpy vs ref_attn
"""

import argparse
import sys

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

P = 128          # partition dim == head dim
KV_TILE = 512    # kv tiling (PSUM moving free-dim limit)
Q_TILE = 128     # q tiling (stationary free dim)


def build_causal_mask(t):
    """[T, T] bf16, 1.0 where kv <= query (keep), else 0.0."""
    m = (np.arange(t)[None, :] <= np.arange(t)[:, None]).astype(np.float32)
    return m.astype(np.float16)


@nki.jit
def nki_attn_fwd(q, k, v, mask, scale, T=2048, D=128):
    """q/k/v: [BH*T, D] bf16 HBM. mask: [T, T] bf16 HBM. Returns [BH*T, D]."""
    BH = q.shape[0] // T
    out = nl.ndarray((BH * T, D), dtype=q.dtype, buffer=nl.shared_hbm)

    for bh in nl.affine_range(BH):
        base = bh * T
        # --- assemble [P, T] tiles (P == D == 128) from [T, D] row chunks
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
            # --- QK tiles: [P, KV_TILE] fp32 PSUM
            n_kv = T // KV_TILE
            qk_tiles = []
            for _i in range(n_kv):
                qk_tiles.append(nl.ndarray((P, KV_TILE), dtype=nl.float32,
                                           buffer=nl.psum))
            for ki in range(n_kv):
                # each dst written ONCE: default accumulate=False (first write
                # initializes PSUM); the flag is only for same-dst K-loops.
                nisa.nc_matmul(
                    dst=qk_tiles[ki],
                    stationary=q_sbuf[0:P, r0:r0 + Q_TILE],
                    moving=k_sbuf[0:P, ki * KV_TILE:(ki + 1) * KV_TILE])
            qk_full = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            for ki in range(n_kv):
                nisa.tensor_copy(
                    dst=qk_full[0:P, ki * KV_TILE:(ki + 1) * KV_TILE],
                    src=qk_tiles[ki])
            # --- scale + softmax (v1-style explicit fp32)
            nisa.tensor_scalar(dst=qk_full, data=qk_full, op0=nl.multiply,
                               operand0=scale)
            row_max = nl.max(qk_full, axis=1, keepdims=True)
            # separate dst (NOT in-place): tensor operands broadcast, and the
            # v1 reference uses a distinct output for the subtract.
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
            norm_full = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=norm_full, data=exp_full, op0=nl.multiply,
                               operand0=inv)
            scores = nl.ndarray((P, T), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=scores, src=norm_full)
            # --- AV over kv chunks of 128. scores chunk transposed on-chip;
            # V chunk DMA'd straight from HBM rows (native [kv, d] order) --
            # delibrately NO 3D SBUF buffer: indexed-3D SBUF slices
            # miscompile (partition descriptor), found 2026-09-17.
            acc = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
            for c in range(T // P):
                st = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=st, data=scores[0:P, c * P:(c + 1) * P])
                st_s = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=st_s, src=st)
                vc = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=vc, src=v[base + c * P:base + (c + 1) * P, :])
                nisa.nc_matmul(dst=acc, accumulate=(c > 0),
                               stationary=st_s, moving=vc)
            ob = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ob, src=acc)
            # NOTE: nl.store does NOT downcast fp32->bf16 on store (garbage
            # out, found 2026-09-17): explicit downcast copy + dma instead.
            ob16 = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ob16, src=ob)
            nisa.dma_copy(dst=out[base + r0:base + r0 + Q_TILE, :], src=ob16)
    return out


def _selftest():
    from nki_attn_equiv import ref_attn, check
    rng = np.random.default_rng(0)
    from ml_dtypes import bfloat16
    allok = True
    for dt in (np.float32, bfloat16):
        for (B, T, H) in [(1, 512, 2), (2, 1024, 2), (1, 2048, 5)]:
            D = 128
            # kernel rows are bh-major: flatten (b,h) blocks
            qb = (rng.standard_normal((B, T, H, D)) * 0.5)
            kb = (rng.standard_normal((B, T, H, D)) * 0.5)
            vb = (rng.standard_normal((B, T, H, D)) * 0.5)
            q = qb.transpose(0, 2, 1, 3).reshape(B * H * T, D).astype(dt)
            k = kb.transpose(0, 2, 1, 3).reshape(B * H * T, D).astype(dt)
            v = vb.transpose(0, 2, 1, 3).reshape(B * H * T, D).astype(dt)
            mask = build_causal_mask(T).astype(
                dt if dt is not np.float32 else np.float32)
            got = np.array(nki_attn_fwd(q, k, v, mask, 1.0 / np.sqrt(D), T, D))
            # reference takes [b, t, h, d] float, computed FROM the same
            # quantized inputs (isolates kernel-internal noise from IO noise)
            qf = q.reshape(B, H, T, D).transpose(0, 2, 1, 3).astype(np.float32)
            kf = k.reshape(B, H, T, D).transpose(0, 2, 1, 3).astype(np.float32)
            vf = v.reshape(B, H, T, D).transpose(0, 2, 1, 3).astype(np.float32)
            want = ref_attn(qf, kf, vf)
            got_b = got.reshape(B, H, T, D).transpose(0, 2, 1, 3)
            tag = f"nki-fwd-vs-ref {np.dtype(dt).name} B{B}T{T}H{H}"
            allok &= check(got_b, want, tag,
                           rel_gate=(dt is np.float32))
    ok = allok
    print("SELFTEST PASS" if ok else "SELFTEST FAIL", flush=True)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, __import__("os").path.dirname(__file__))
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    print("pass --selftest (needs Neuron device)", flush=True)
