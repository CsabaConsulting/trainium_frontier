"""Hand-written NKI kernel for the Mamba2 chunked SSD scan -- FORWARD.

Development/equivalence harness. The kernel here is checked on-device against
train.py's eager `ssd_chunked` (itself verified exact against a naive sequential
recurrence in mamba2_ssd_ref.py), then moved into train.py once green.

WHY a kernel at all: the eager form is ~25 separate tensor ops per Mamba2 layer,
and this repo's native-eager Neuron backend lowers each op to its own NEFF (see
train.py's rope-kernel comment: ~30 NEFFs for one rope+norm). One NEFF for the
whole scan removes that dispatch cost.

LAYOUT / ALGEBRA NOTES (why the kernel looks like this)

`nisa.nc_matmul(dst, stationary, moving)` computes `dst = stationary.T @ moving`
with the CONTRACTION dimension on the partition axis of BOTH operands. So the
whole design is chosen to make every contraction land on a partition axis with
no avoidable transposes.

Per (batch row b, head h, chunk c), with Q = chunk len, P = head_dim, N = d_state:

  dtA  [Q,1]      = dt * A_h                                            (<= 0)
  ell  [Q,1]      = inclusive cumsum of dtA = mm(UI, dtA), UI[r,t]=1 iff r<=t
  ellb [Q_s,Q_t]  = ell_t broadcast along s = mm(onesQ, UI*dtA)
  seg  [Q_s,Q_t]  = ell_t - ell_s = ellb - ell        (ell is the per-partition
                                                       operand of tensor_scalar)
  MT   [Q_s,Q_t]  = exp(min(seg,0) + intra_mask) * (B_s . C_t)
  Y    [Q_t,P]   += mm(MT, dx)                        (contraction over s)
  Bbar [Q_s,N]    = B * exp(min(de - ell, 0))
  Gst  [N,P]      = mm(Bbar, dx)                      (contraction over s)
  Y    [Q_t,P]   += mm(CbarT, S)                      (contraction over N)
  S           <-   exp(min(de,0)) * S + Gst

MT is built with s (the SOURCE token) on the PARTITION axis, not t. That is
deliberate: it makes MT itself the `stationary` operand of the Y_intra matmul, so
the [Q,Q] score matrix never needs transposing. Only B and C are transposed
(2 x nc_transpose per chunk).

THREE THINGS THAT LOOK LIKE THEY NEED A MATMUL AND DON'T. Compute engines cannot
broadcast along the partition axis (only DMA or a matmul can), which naively
costs extra matmuls for `de` and for `Cbar^T`. But `ellb[s,t] = ell_t` is ALREADY
constant along s, so:

  de replicated on every partition          = ellb[:, Q-1]
  exp(min(ell_t,0)) broadcast on partitions = exp(min(ellb,0))[0:N, 0:Q]   (N<=Q)
  Cbar^T[n,q] = C^T[n,q] * exp(min(ell_q,0)) -> elementwise with the tile above,
                so Cbar needs NO transpose of its own.

That removes 3 of the 11 tensor-engine ops per chunk and, more importantly, 4 of
the PSUM tiles -- which matters because PSUM is only 8 banks.

PSUM ACCUMULATION IS IMPLICIT. Multiple `nc_matmul` writes to the same PSUM tile
form an accumulation group and SUM (nki-writing/references/common-patterns.md:
"Multiple writes to the same PSUM buffer trigger hardware accumulation
automatically"). So a PSUM tile may NOT be recycled as scratch for an unrelated
matmul -- each distinct result needs its own tile. This kernel uses 7:
  ps_ell [Q,1]  ps_ellb [Q,Q]  ps_bT [N,Q]  ps_cT [N,Q]  ps_cb [Q,Q]
  ps_y [Q,P]  ps_g [N,P]
and deliberately EXPLOITS the accumulation for ps_y, into which both the
intra-chunk and the inter-chunk matmul write (Y = Y_intra + Y_inter for free).

MASK BEFORE EXP. `min(seg, 0)` and the additive -30000 causal mask are applied
BEFORE `exp`. For s > t the raw exponent ell_t - ell_s is POSITIVE and grows like
|A|*dt*(s-t) (measured up to +520 at T=384), so exp() would overflow to +inf and
inf*0 = NaN -- the latent bug in the sibling repo's kernel
(../state-space-models-neuron/mamba2/mamba2_kernel.py does nl.exp then
nisa.affine_select). After the clamp every exponent is <= 0 and exp <= 1. Every
LIVE (s <= t) entry has a provably <= 0 exponent (A < 0, dt > 0), so the clamp is
exact, not an approximation. This mirrors the eager reference op-for-op:
`exp(seg.clamp(max=0) + intra_add)`.

Run on a Trn2 box:  python research/scratch/ssd_nki_dev.py
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch

import nki
import nki.isa as nisa
import nki.language as nl

_PMAX = 128
# exp(-30000) is exactly 0.0 in fp32; added to an already-clamped (<=0) exponent
# it can never produce NaN, unlike masking after exp (which would do inf * 0).
_NEG = -30000.0


@nki.jit
def ssd_fwd_nki(x2, dt2, B2, C2, Ab, Db, T, Q, G):
    """Chunked SSD forward.

    x2  [BT, H*P]   fp32, token-major (BT = b*T), head-major within the row
    dt2 [BT, H]     fp32, > 0
    B2  [BT, G*N]   fp32
    C2  [BT, G*N]   fp32
    Ab  [128, H]    fp32, A_h (< 0) replicated on every partition
    Db  [128, H]    fp32, D_h replicated on every partition
    T, Q, G         python ints (compile-time constants)
    -> y2 [BT, H*P] fp32
    """
    BT, HP = x2.shape
    H = dt2.shape[1]
    P = HP // H
    N = B2.shape[1] // G
    nb = BT // T
    CPB = T // Q          # chunks per batch row
    hpg = H // G          # heads per B/C group
    fp = nl.float32

    # N <= Q is what lets ellb double as the partition-broadcast of exp(ell).
    assert N <= Q <= _PMAX and P <= 512 and T % Q == 0 and H % G == 0

    y2 = nl.ndarray((BT, HP), dtype=fp, buffer=nl.shared_hbm)

    # ---- constants, built once -----------------------------------------------
    ri = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
    nisa.iota(dst=ri, pattern=[[0, Q]], offset=0, channel_multiplier=1)  # row index
    ci = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
    nisa.iota(dst=ci, pattern=[[1, Q]], offset=0, channel_multiplier=0)  # col index
    UI = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=UI, data1=ri, data2=ci, op=nl.less_equal)     # r <= t
    onesQ = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
    nisa.memset(dst=onesQ, value=1.0)
    # additive causal mask: 0 where s <= t, -30000 where s > t
    MASK = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=MASK, data=UI, op0=nl.subtract, operand0=1.0)  # 0 / -1
    nisa.tensor_scalar(dst=MASK, data=MASK, op0=nl.multiply, operand0=-_NEG)

    ab = nl.ndarray((Q, H), dtype=fp, buffer=nl.sbuf)
    nisa.dma_copy(dst=ab, src=Ab[0:Q, 0:H])
    db = nl.ndarray((Q, H), dtype=fp, buffer=nl.sbuf)
    nisa.dma_copy(dst=db, src=Db[0:Q, 0:H])

    # Loop over (batch row, group, head-in-group) so the B/C group index is an
    # affine expression of the loop variables rather than a floor-div of one.
    for bi in range(nb):
      for g in range(G):
        for hh in range(hpg):
            h = g * hpg + hh
            S = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
            nisa.memset(dst=S, value=0.0)

            for c in range(CPB):
                r0 = bi * T + c * Q

                xt = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                nisa.dma_copy(dst=xt, src=x2[r0:r0 + Q, h * P:h * P + P])
                dtt = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.dma_copy(dst=dtt, src=dt2[r0:r0 + Q, h:h + 1])
                Bt = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                nisa.dma_copy(dst=Bt, src=B2[r0:r0 + Q, g * N:g * N + N])
                Ct = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                nisa.dma_copy(dst=Ct, src=C2[r0:r0 + Q, g * N:g * N + N])

                dtA = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)     # dt*A, <= 0
                nisa.tensor_scalar(dst=dtA, data=dtt, op0=nl.multiply,
                                   operand0=ab[0:Q, h:h + 1])
                dx = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)      # dt*x
                nisa.tensor_scalar(dst=dx, data=xt, op0=nl.multiply, operand0=dtt)

                # --- ell: inclusive cumsum of dtA, as a matmul against UI ------
                ps_ell = nl.ndarray((Q, 1), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=ps_ell, stationary=UI, moving=dtA,
                               is_stationary_onezero=True)
                ell = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ell, src=ps_ell)

                # --- ellb[s,t] = ell_t, i.e. ell broadcast along partitions ---
                UIdtA = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=UIdtA, data=UI, op0=nl.multiply, operand0=dtA)
                ps_ellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=ps_ellb, stationary=onesQ, moving=UIdtA,
                               is_stationary_onezero=True)
                ellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ellb, src=ps_ellb)

                # exp(min(ell_t, 0)) broadcast along partitions (free because ellb
                # is already constant along s). Used for Cbar and, at column
                # Q-1, for the chunk decay exp(min(de,0)).
                eellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=eellb, data=ellb, op0=nl.minimum, operand0=0.0)
                nisa.activation(dst=eellb, op=nl.exp, data=eellb)

                # --- intra-chunk decay x score matrix, s on the partition axis -
                seg = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=seg, data=ellb, op0=nl.subtract, operand0=ell,
                                   op1=nl.minimum, operand1=0.0)
                nisa.tensor_tensor(dst=seg, data1=seg, data2=MASK, op=nl.add)
                L = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.activation(dst=L, op=nl.exp, data=seg)

                ps_bT = nl.ndarray((N, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_transpose(dst=ps_bT, data=Bt)
                BTs = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=BTs, src=ps_bT)
                ps_cT = nl.ndarray((N, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_transpose(dst=ps_cT, data=Ct)
                CTs = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=CTs, src=ps_cT)

                ps_cb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=ps_cb, stationary=BTs, moving=CTs)
                CBs = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=CBs, src=ps_cb)
                MT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=MT, data1=L, data2=CBs, op=nl.multiply)

                # --- Y = Y_intra + Y_inter, summed IN PSUM (two writes, one tile)
                ps_y = nl.ndarray((Q, P), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=ps_y, stationary=MT, moving=dx)
                if c > 0:
                    # Cbar^T[n,q] = C^T[n,q] * exp(min(ell_q,0)); no transpose of
                    # Cbar needed, the eellb rows ARE that broadcast (N <= Q).
                    CbarT = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=CbarT, data1=CTs, data2=eellb[0:N, 0:Q],
                                       op=nl.multiply)
                    nisa.nc_matmul(dst=ps_y, stationary=CbarT, moving=S)

                yb = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=yb, src=ps_y)
                dxx = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=dxx, data=xt, op0=nl.multiply,
                                   operand0=db[0:Q, h:h + 1])
                nisa.tensor_tensor(dst=yb, data1=yb, data2=dxx, op=nl.add)
                nisa.dma_copy(dst=y2[r0:r0 + Q, h * P:h * P + P], src=yb)

                # --- chunk state: S <- exp(min(de,0)) * S + Bbar^T (dt x) ------
                if c + 1 < CPB:
                    # de = ell_{Q-1}, already replicated on every partition
                    rev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=rev, data1=ellb[0:Q, Q - 1:Q], data2=ell,
                                       op=nl.subtract)
                    nisa.tensor_scalar(dst=rev, data=rev, op0=nl.minimum, operand0=0.0)
                    erev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.activation(dst=erev, op=nl.exp, data=rev)
                    Bbar = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=Bbar, data=Bt, op0=nl.multiply,
                                       operand0=erev)
                    ps_g = nl.ndarray((N, P), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=ps_g, stationary=Bbar, moving=dx)
                    Gs = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=Gs, src=ps_g)
                    nisa.tensor_scalar(dst=S, data=S, op0=nl.multiply,
                                       operand0=eellb[0:N, Q - 1:Q])
                    nisa.tensor_tensor(dst=S, data1=S, data2=Gs, op=nl.add)

    return y2


@nki.jit
def ssd_bwd_nki(gy2, x2, dt2, B2, C2, Ab, Db, T, Q, G):
    """Chunked SSD backward. Same layouts as ssd_fwd_nki.

    -> dx2 [BT,H*P], ddt2 [BT,H], dAo [H,1], dB2 [BT,G*N], dC2 [BT,G*N], dDo [H,1]

    The gradient algebra (and the reason no clamp-indicator tensors are needed) is
    derived and verified in ssd_nki_sim.py; this is a transcription of the
    `ssd_bwd_sim` there, which agrees with autograd through the independent naive
    sequential scan to ~1e-14 in float64.

    TWO STRUCTURAL CHOICES WORTH THE COMMENT:

    1. BOTH ORIENTATIONS OF THE [Q,Q] TENSORS FOR FREE. dB contracts over t while
       dC contracts over s, so the score/decay matrices are needed both ways.
       Transposing them would cost nc_transpose + a PSUM tile each, and PSUM is
       the binding constraint. They can be had without one because every [Q,Q]
       tensor here is either an outer product (transpose = swap the two matmul
       operands, free) or a function of ell alone. For the latter,
       `tensor_scalar(ellb, subtract, ell)` gives [i,j] = ell_j - ell_i and the
       same call with `reverse0=True` gives [i,j] = ell_i - ell_j, which read at
       (i,j)=(t,s) IS seg[s,t]. So two elementwise ops plus two constant masks
       (MASK, MASKT) give L and L^T with no tensor-engine work at all.
       (seg is NOT symmetric -- reading ONE raw tile both ways silently negates
       the exponent. That bug cost a hardware round trip; the simulator now
       mirrors this exact dataflow so it cannot recur.)

    2. dB/dC ARE ACCUMULATED IN-KERNEL ACROSS HEADS. With n_groups < n_heads (the
       real config is G=1, H=10) B and C are shared, so their gradients sum over
       the hpg heads of a group. Rather than spilling per-head partials to HBM and
       reducing them with an extra eager op, the kernel keeps a [Q, CPB*N]
       accumulator per group covering every chunk (8 KB/partition at the real
       geometry) and writes it once per group.

    The forward state S_c entering each chunk is RECOMPUTED (pass 1) rather than
    saved by the forward kernel: saving it would be CPB x [N,P] x H x nb = 21 MB
    per layer of extra saved-tensor memory and an extra HBM write in the forward,
    versus ~3 extra matmuls per chunk here.
    """
    BT, HP = x2.shape
    H = dt2.shape[1]
    P = HP // H
    N = B2.shape[1] // G
    nb = BT // T
    CPB = T // Q
    hpg = H // G
    fp = nl.float32

    assert N <= Q <= _PMAX and P <= 512 and T % Q == 0 and H % G == 0

    dx2 = nl.ndarray((BT, HP), dtype=fp, buffer=nl.shared_hbm)
    ddt2 = nl.ndarray((BT, H), dtype=fp, buffer=nl.shared_hbm)
    dB2 = nl.ndarray((BT, G * N), dtype=fp, buffer=nl.shared_hbm)
    dC2 = nl.ndarray((BT, G * N), dtype=fp, buffer=nl.shared_hbm)
    dAo = nl.ndarray((H, 1), dtype=fp, buffer=nl.shared_hbm)
    dDo = nl.ndarray((H, 1), dtype=fp, buffer=nl.shared_hbm)

    # ---- constants ------------------------------------------------------------
    ri = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
    nisa.iota(dst=ri, pattern=[[0, Q]], offset=0, channel_multiplier=1)
    ci = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
    nisa.iota(dst=ci, pattern=[[1, Q]], offset=0, channel_multiplier=0)
    UI = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=UI, data1=ri, data2=ci, op=nl.less_equal)   # [r,t] r<=t
    LI = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=LI, data1=ri, data2=ci, op=nl.greater_equal)  # [t,r] r<=t
    onesQ = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
    nisa.memset(dst=onesQ, value=1.0)
    MASK = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=MASK, data=UI, op0=nl.subtract, operand0=1.0)
    nisa.tensor_scalar(dst=MASK, data=MASK, op0=nl.multiply, operand0=-_NEG)
    MASKT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=MASKT, data=LI, op0=nl.subtract, operand0=1.0)
    nisa.tensor_scalar(dst=MASKT, data=MASKT, op0=nl.multiply, operand0=-_NEG)

    ab = nl.ndarray((Q, H), dtype=fp, buffer=nl.sbuf)
    nisa.dma_copy(dst=ab, src=Ab[0:Q, 0:H])
    db = nl.ndarray((Q, H), dtype=fp, buffer=nl.sbuf)
    nisa.dma_copy(dst=db, src=Db[0:Q, 0:H])

    # dA_h / dD_h are sums over every token; accumulate elementwise into a [Q,H]
    # tile and do ONE partition-axis reduction (a single matmul) at the very end.
    accA = nl.ndarray((Q, H), dtype=fp, buffer=nl.sbuf)
    nisa.memset(dst=accA, value=0.0)
    accD = nl.ndarray((Q, H), dtype=fp, buffer=nl.sbuf)
    nisa.memset(dst=accD, value=0.0)

    for bi in range(nb):
      for g in range(G):
        aB = nl.ndarray((Q, CPB * N), dtype=fp, buffer=nl.sbuf)
        nisa.memset(dst=aB, value=0.0)
        aC = nl.ndarray((Q, CPB * N), dtype=fp, buffer=nl.sbuf)
        nisa.memset(dst=aC, value=0.0)

        for hh in range(hpg):
            h = g * hpg + hh

            # ---- pass 1: the forward state entering each chunk ----------------
            Sall = nl.ndarray((N, CPB * P), dtype=fp, buffer=nl.sbuf)
            S = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
            nisa.memset(dst=S, value=0.0)
            for c in range(CPB):
                nisa.tensor_copy(dst=Sall[0:N, c * P:c * P + P], src=S)
                if c + 1 == CPB:
                    break
                r0 = bi * T + c * Q
                xt = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                nisa.dma_copy(dst=xt, src=x2[r0:r0 + Q, h * P:h * P + P])
                dtt = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.dma_copy(dst=dtt, src=dt2[r0:r0 + Q, h:h + 1])
                Bt = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                nisa.dma_copy(dst=Bt, src=B2[r0:r0 + Q, g * N:g * N + N])

                dtA = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=dtA, data=dtt, op0=nl.multiply,
                                   operand0=ab[0:Q, h:h + 1])
                u = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=u, data=xt, op0=nl.multiply, operand0=dtt)

                p1 = nl.ndarray((Q, 1), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=p1, stationary=UI, moving=dtA,
                               is_stationary_onezero=True)
                ell = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ell, src=p1)
                UIdtA = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=UIdtA, data=UI, op0=nl.multiply, operand0=dtA)
                p2 = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=p2, stationary=onesQ, moving=UIdtA,
                               is_stationary_onezero=True)
                ellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ellb, src=p2)
                eellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=eellb, data=ellb, op0=nl.minimum, operand0=0.0)
                nisa.activation(dst=eellb, op=nl.exp, data=eellb)

                rev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=rev, data1=ellb[0:Q, Q - 1:Q], data2=ell,
                                   op=nl.subtract)
                nisa.tensor_scalar(dst=rev, data=rev, op0=nl.minimum, operand0=0.0)
                erev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.activation(dst=erev, op=nl.exp, data=rev)
                Bbar = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=Bbar, data=Bt, op0=nl.multiply, operand0=erev)
                p3 = nl.ndarray((N, P), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=p3, stationary=Bbar, moving=u)
                Gs = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=Gs, src=p3)
                nisa.tensor_scalar(dst=S, data=S, op0=nl.multiply,
                                   operand0=eellb[0:N, Q - 1:Q])
                nisa.tensor_tensor(dst=S, data1=S, data2=Gs, op=nl.add)

            # ---- pass 2: reverse over chunks ---------------------------------
            dS = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
            nisa.memset(dst=dS, value=0.0)

            for c in range(CPB - 1, -1, -1):
                r0 = bi * T + c * Q
                xt = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                nisa.dma_copy(dst=xt, src=x2[r0:r0 + Q, h * P:h * P + P])
                dtt = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.dma_copy(dst=dtt, src=dt2[r0:r0 + Q, h:h + 1])
                Bt = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                nisa.dma_copy(dst=Bt, src=B2[r0:r0 + Q, g * N:g * N + N])
                Ct = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                nisa.dma_copy(dst=Ct, src=C2[r0:r0 + Q, g * N:g * N + N])
                gy = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                nisa.dma_copy(dst=gy, src=gy2[r0:r0 + Q, h * P:h * P + P])

                # --- recompute the forward intermediates
                dtA = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=dtA, data=dtt, op0=nl.multiply,
                                   operand0=ab[0:Q, h:h + 1])
                u = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=u, data=xt, op0=nl.multiply, operand0=dtt)

                q1 = nl.ndarray((Q, 1), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=q1, stationary=UI, moving=dtA,
                               is_stationary_onezero=True)
                ell = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ell, src=q1)
                UIdtA = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=UIdtA, data=UI, op0=nl.multiply, operand0=dtA)
                q2 = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=q2, stationary=onesQ, moving=UIdtA,
                               is_stationary_onezero=True)
                ellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ellb, src=q2)
                eellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=eellb, data=ellb, op0=nl.minimum, operand0=0.0)
                nisa.activation(dst=eellb, op=nl.exp, data=eellb)
                eellc = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=eellc, data=ell, op0=nl.minimum, operand0=0.0)
                nisa.activation(dst=eellc, op=nl.exp, data=eellc)

                # L and L^T without a transpose (see the docstring). NOTE the two
                # raw tiles: seg is NOT symmetric, so reading one tile at [t,s]
                # gives ell_s - ell_t, the NEGATION of what L^T needs. reverse0=True
                # produces the other orientation directly.
                raw = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=raw, data=ellb, op0=nl.subtract, operand0=ell,
                                   op1=nl.minimum, operand1=0.0)
                tmp = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=tmp, data1=raw, data2=MASK, op=nl.add)
                L = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.activation(dst=L, op=nl.exp, data=tmp)
                nisa.tensor_scalar(dst=raw, data=ellb, op0=nl.subtract, operand0=ell,
                                   reverse0=True, op1=nl.minimum, operand1=0.0)
                nisa.tensor_tensor(dst=tmp, data1=raw, data2=MASKT, op=nl.add)
                LT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.activation(dst=LT, op=nl.exp, data=tmp)

                q3 = nl.ndarray((N, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_transpose(dst=q3, data=Bt)
                BTs = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=BTs, src=q3)
                q4 = nl.ndarray((N, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_transpose(dst=q4, data=Ct)
                CTs = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=CTs, src=q4)

                q5 = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=q5, stationary=BTs, moving=CTs)   # CB[s,t]
                MT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=MT, data1=L, data2=q5, op=nl.multiply)
                q6 = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=q6, stationary=CTs, moving=BTs)   # CB^T[t,s]
                MTT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=MTT, data1=LT, data2=q6, op=nl.multiply)

                q7 = nl.ndarray((P, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_transpose(dst=q7, data=gy)
                gyT = nl.ndarray((P, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gyT, src=q7)
                q8 = nl.ndarray((P, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_transpose(dst=q8, data=u)
                uT = nl.ndarray((P, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=uT, src=q8)

                # --- dD_h += <gy_t, x_t>
                gx = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=gx, data1=gy, data2=xt, op=nl.multiply)
                rsum = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=rsum, op=nl.add, data=gx, axis=(1,))
                nisa.tensor_tensor(dst=accD[0:Q, h:h + 1], data1=accD[0:Q, h:h + 1],
                                   data2=rsum, op=nl.add)

                # --- intra-chunk
                q9 = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=q9, stationary=uT, moving=gyT)      # dMT[s,t]
                dMT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dMT, src=q9)
                qa = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=qa, stationary=gyT, moving=uT)      # dMT^T[t,s]
                dMTT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dMTT, src=qa)

                dseg = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=dseg, data1=dMT, data2=MT, op=nl.multiply)
                dCB = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=dCB, data1=dMT, data2=L, op=nl.multiply)
                dCBT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=dCBT, data1=dMTT, data2=LT, op=nl.multiply)

                # dell_j = sum_s dseg[s,j] (partition reduce -> matmul) - sum_t dseg[j,t]
                qb = nl.ndarray((Q, 1), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=qb, stationary=dseg, moving=onesQ[0:Q, 0:1],
                               is_moving_onezero=True)
                dell = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dell, src=qb)
                nisa.tensor_reduce(dst=rsum, op=nl.add, data=dseg, axis=(1,))
                nisa.tensor_tensor(dst=dell, data1=dell, data2=rsum, op=nl.subtract)

                qc = nl.ndarray((Q, N), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=qc, stationary=dCBT, moving=Ct)     # dB[s,n]
                dB = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dB, src=qc)
                qd = nl.ndarray((Q, N), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=qd, stationary=dCB, moving=Bt)      # dC[t,n]
                dC = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dC, src=qd)

                # du accumulates two matmuls -> deliberately one PSUM tile
                qe = nl.ndarray((Q, P), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=qe, stationary=MTT, moving=gy)      # du[s,p]

                # --- inter-chunk output term (S_0 == 0, nothing to do at c == 0)
                if c > 0:
                    qf = nl.ndarray((P, N), dtype=fp, buffer=nl.psum)
                    nisa.nc_transpose(dst=qf, data=Sall[0:N, c * P:c * P + P])
                    STs = nl.ndarray((P, N), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=STs, src=qf)
                    qg = nl.ndarray((Q, N), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=qg, stationary=gyT, moving=STs)  # dCbar[t,n]
                    dCbar = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dCbar, src=qg)
                    scr = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=scr, data=dCbar, op0=nl.multiply,
                                       operand0=eellc)
                    nisa.tensor_tensor(dst=dC, data1=dC, data2=scr, op=nl.add)
                    nisa.tensor_tensor(dst=scr, data1=dCbar, data2=Ct, op=nl.multiply)
                    nisa.tensor_reduce(dst=rsum, op=nl.add, data=scr, axis=(1,))
                    nisa.tensor_tensor(dst=rsum, data1=rsum, data2=eellc,
                                       op=nl.multiply)
                    nisa.tensor_tensor(dst=dell, data1=dell, data2=rsum, op=nl.add)

                # --- the carry into the next chunk's state (dS is dS_{c+1} here)
                have_dde = c + 1 < CPB
                if have_dde:
                    qh = nl.ndarray((P, N), dtype=fp, buffer=nl.psum)
                    nisa.nc_transpose(dst=qh, data=dS)
                    dST = nl.ndarray((P, N), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dST, src=qh)
                    qi = nl.ndarray((Q, N), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=qi, stationary=uT, moving=dST)  # dBbar[s,n]
                    dBbar = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dBbar, src=qi)

                    rev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=rev, data1=ellb[0:Q, Q - 1:Q], data2=ell,
                                       op=nl.subtract)
                    nisa.tensor_scalar(dst=rev, data=rev, op0=nl.minimum, operand0=0.0)
                    erev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.activation(dst=erev, op=nl.exp, data=rev)
                    # Bbar^T = B^T * erev broadcast on partitions; erev_s =
                    # exp(min(de-ell_s,0)) and eellb[p,s] = exp(min(ell_s,0)) with de
                    # at column Q-1, so the broadcast comes straight out of eellb.
                    revb = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=revb, data=ellb[0:N, 0:Q], op0=nl.subtract,
                                       operand0=ellb[0:N, Q - 1:Q], reverse0=True,
                                       op1=nl.minimum, operand1=0.0)
                    nisa.activation(dst=revb, op=nl.exp, data=revb)
                    BbarT = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=BbarT, data1=BTs, data2=revb,
                                       op=nl.multiply)
                    nisa.nc_matmul(dst=qe, stationary=BbarT, moving=dS)  # du += ...

                    scr2 = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=scr2, data=dBbar, op0=nl.multiply,
                                       operand0=erev)
                    nisa.tensor_tensor(dst=dB, data1=dB, data2=scr2, op=nl.add)
                    nisa.tensor_tensor(dst=scr2, data1=dBbar, data2=Bt,
                                       op=nl.multiply)
                    drev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=drev, op=nl.add, data=scr2, axis=(1,))
                    nisa.tensor_tensor(dst=drev, data1=drev, data2=erev,
                                       op=nl.multiply)
                    nisa.tensor_tensor(dst=dell, data1=dell, data2=drev,
                                       op=nl.subtract)

                    # dde = sum_s drev_s + <dS_{c+1}, S_c> * ac. Both are
                    # partition-axis reductions, so both are matmuls against a ones
                    # column -- accumulated into one [1,1] PSUM tile.
                    qj = nl.ndarray((1, 1), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=qj, stationary=drev, moving=onesQ[0:Q, 0:1],
                                   is_moving_onezero=True)
                    pr0 = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=pr0, data1=dS,
                                       data2=Sall[0:N, c * P:c * P + P],
                                       op=nl.multiply)
                    pr1 = nl.ndarray((N, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=pr1, op=nl.add, data=pr0, axis=(1,))
                    nisa.tensor_tensor(dst=pr1, data1=pr1, data2=eellb[0:N, Q - 1:Q],
                                       op=nl.multiply)
                    nisa.nc_matmul(dst=qj, stationary=pr1, moving=onesQ[0:N, 0:1],
                                   is_moving_onezero=True)
                    dde = nl.ndarray((1, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dde, src=qj)

                # --- dell -> ddtA (reverse inclusive cumsum), then dt / A / x
                qk = nl.ndarray((Q, 1), dtype=fp, buffer=nl.psum)
                nisa.nc_matmul(dst=qk, stationary=LI, moving=dell,
                               is_stationary_onezero=True)
                if have_dde:
                    # de = ell_{Q-1}, and ddtA_r = sum_{t>=r} dell_t, so the dde
                    # contribution reaches EVERY r -- a broadcast add, not a
                    # one-hot injection. Broadcasting [1,1] onto Q partitions is a
                    # matmul with a ones row, and it accumulates into the same tile.
                    nisa.nc_matmul(dst=qk, stationary=onesQ[0:1, 0:Q], moving=dde,
                                   is_stationary_onezero=True)
                ddtA = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ddtA, src=qk)

                du = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_copy(dst=du, src=qe)
                ddt = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=ddt, data=ddtA, op0=nl.multiply,
                                   operand0=ab[0:Q, h:h + 1])
                nisa.tensor_tensor(dst=gx, data1=du, data2=xt, op=nl.multiply)
                nisa.tensor_reduce(dst=rsum, op=nl.add, data=gx, axis=(1,))
                nisa.tensor_tensor(dst=ddt, data1=ddt, data2=rsum, op=nl.add)
                nisa.tensor_tensor(dst=rsum, data1=ddtA, data2=dtt, op=nl.multiply)
                nisa.tensor_tensor(dst=accA[0:Q, h:h + 1], data1=accA[0:Q, h:h + 1],
                                   data2=rsum, op=nl.add)

                dxt = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=dxt, data=du, op0=nl.multiply, operand0=dtt)
                nisa.tensor_scalar(dst=gx, data=gy, op0=nl.multiply,
                                   operand0=db[0:Q, h:h + 1])
                nisa.tensor_tensor(dst=dxt, data1=dxt, data2=gx, op=nl.add)

                nisa.dma_copy(dst=dx2[r0:r0 + Q, h * P:h * P + P], src=dxt)
                nisa.dma_copy(dst=ddt2[r0:r0 + Q, h:h + 1], src=ddt)
                nisa.tensor_tensor(dst=aB[0:Q, c * N:c * N + N],
                                   data1=aB[0:Q, c * N:c * N + N], data2=dB,
                                   op=nl.add)
                nisa.tensor_tensor(dst=aC[0:Q, c * N:c * N + N],
                                   data1=aC[0:Q, c * N:c * N + N], data2=dC,
                                   op=nl.add)

                # --- dS_c = ac_c dS_{c+1} + (inter term of chunk c)
                if c > 0:
                    Cbar = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=Cbar, data=Ct, op0=nl.multiply,
                                       operand0=eellc)
                    ql = nl.ndarray((N, P), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=ql, stationary=Cbar, moving=gy)
                    dSy = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dSy, src=ql)
                    nisa.tensor_scalar(dst=dS, data=dS, op0=nl.multiply,
                                       operand0=eellb[0:N, Q - 1:Q])
                    nisa.tensor_tensor(dst=dS, data1=dS, data2=dSy, op=nl.add)

        for c in range(CPB):
            r0 = bi * T + c * Q
            nisa.dma_copy(dst=dB2[r0:r0 + Q, g * N:g * N + N],
                          src=aB[0:Q, c * N:c * N + N])
            nisa.dma_copy(dst=dC2[r0:r0 + Q, g * N:g * N + N],
                          src=aC[0:Q, c * N:c * N + N])

    # ---- dA / dD: one partition-axis reduction for all H heads at once --------
    pa = nl.ndarray((H, 1), dtype=fp, buffer=nl.psum)
    nisa.nc_matmul(dst=pa, stationary=accA, moving=onesQ[0:Q, 0:1],
                   is_moving_onezero=True)
    sa = nl.ndarray((H, 1), dtype=fp, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sa, src=pa)
    nisa.dma_copy(dst=dAo, src=sa)
    pd = nl.ndarray((H, 1), dtype=fp, buffer=nl.psum)
    nisa.nc_matmul(dst=pd, stationary=accD, moving=onesQ[0:Q, 0:1],
                   is_moving_onezero=True)
    sd = nl.ndarray((H, 1), dtype=fp, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sd, src=pd)
    nisa.dma_copy(dst=dDo, src=sd)

    return dx2, ddt2, dAo, dB2, dC2, dDo


# -----------------------------------------------------------------------------
# eager reference (a copy of train.py's ssd_chunked, kept standalone so this file
# can run without importing the whole training script)


def _masks(Q, NC, device, dtype):
    qi = torch.arange(Q, device=device)
    zero = torch.zeros((), device=device, dtype=dtype)
    neg = torch.full((), _NEG, device=device, dtype=dtype)
    tril_q = qi[:, None] >= qi[None, :]
    ci = torch.arange(NC, device=device)
    return (tril_q.to(dtype), torch.where(tril_q, zero, neg),
            torch.where(ci[:, None] > ci[None, :], zero, neg),
            (ci[:, None] > ci[None, :]).to(dtype), (ci[:, None] >= ci[None, :]).to(dtype))


def ssd_chunked_eager(x, dt, A, B, C, D, Q, masks):
    b, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    NC = T // Q
    hpg = H // G
    cumsum_op, intra_add, inter_add, excl_op, incl_op = masks
    xc = x.reshape(b, NC, Q, H, P).permute(0, 1, 3, 2, 4)
    dtc = dt.reshape(b, NC, Q, H).permute(0, 1, 3, 2)
    Bf = B.reshape(b, NC, Q, G, N).permute(0, 1, 3, 2, 4)
    Cf = C.reshape(b, NC, Q, G, N).permute(0, 1, 3, 2, 4)
    if hpg > 1:
        Bf = Bf.repeat_interleave(hpg, dim=2)
        Cf = Cf.repeat_interleave(hpg, dim=2)
    dx = dtc.unsqueeze(-1) * xc
    ell = torch.matmul(dtc * A.reshape(1, 1, H, 1), cumsum_op.transpose(0, 1))
    seg = ell.unsqueeze(-1) - ell.unsqueeze(-2)
    Lc = torch.exp(seg.clamp(max=0.0) + intra_add)
    Y = torch.matmul(Lc * torch.matmul(Cf, Bf.transpose(-1, -2)), dx)
    de = ell[..., -1]
    Bbar = Bf * torch.exp((de.unsqueeze(-1) - ell).clamp(max=0.0)).unsqueeze(-1)
    Gst = torch.matmul(Bbar.transpose(-1, -2), dx)
    deh = de.permute(0, 2, 1)
    Pex = torch.matmul(deh, excl_op.transpose(0, 1))
    Pin = torch.matmul(deh, incl_op.transpose(0, 1))
    L2 = torch.exp((Pex.unsqueeze(-1) - Pin.unsqueeze(-2)).clamp(max=0.0) + inter_add)
    S_in = torch.matmul(L2, Gst.permute(0, 2, 1, 3, 4).reshape(b, H, NC, N * P))
    S_in = S_in.reshape(b, H, NC, N, P).permute(0, 2, 1, 3, 4)
    Cbar = Cf * torch.exp(ell.clamp(max=0.0)).unsqueeze(-1)
    if hpg == 1 and G == 1:
        Cbar = Cbar.expand(b, NC, H, Q, N)
    Y = Y + torch.matmul(Cbar, S_in)
    y = Y.permute(0, 1, 3, 2, 4).reshape(b, T, H, P)
    return y + D.reshape(1, 1, H, 1) * x


def call_nki(x, dt, A, B, C, D, Q, G):
    b, T, H, P = x.shape
    N = B.shape[3]
    x2 = x.reshape(b * T, H * P).contiguous()
    dt2 = dt.reshape(b * T, H).contiguous()
    B2 = B.reshape(b * T, G * N).contiguous()
    C2 = C.reshape(b * T, G * N).contiguous()
    Ab = A.reshape(1, H).expand(_PMAX, H).contiguous()
    Db = D.reshape(1, H).expand(_PMAX, H).contiguous()
    y2 = ssd_fwd_nki(x2, dt2, B2, C2, Ab, Db, T, Q, G)
    return y2.reshape(b, T, H, P)


def call_nki_bwd(gy, x, dt, A, B, C, D, Q, G):
    b, T, H, P = x.shape
    N = B.shape[3]
    outs = ssd_bwd_nki(gy.reshape(b * T, H * P).contiguous(),
                       x.reshape(b * T, H * P).contiguous(),
                       dt.reshape(b * T, H).contiguous(),
                       B.reshape(b * T, G * N).contiguous(),
                       C.reshape(b * T, G * N).contiguous(),
                       A.reshape(1, H).expand(_PMAX, H).contiguous(),
                       D.reshape(1, H).expand(_PMAX, H).contiguous(),
                       T, Q, G)
    dx2, ddt2, dAo, dB2, dC2, dDo = outs
    return (dx2.reshape(b, T, H, P), ddt2.reshape(b, T, H), dAo.reshape(H),
            dB2.reshape(b, T, G, N), dC2.reshape(b, T, G, N), dDo.reshape(H))


CASES = [
    # (b, T, H, P, G, N, Q)
    (1, 128, 2, 128, 1, 128, 128),    # one chunk, smallest real shape
    (1, 256, 2, 128, 1, 128, 128),    # two chunks -> exercises the state carry
    (2, 512, 4, 128, 1, 128, 128),    # multi-batch, multi-head
    (1, 512, 4, 128, 2, 128, 128),    # n_groups=2 -> dB/dC sum over 2 heads each
    (2, 2048, 10, 128, 1, 128, 128),  # the competition shape
]


def _inputs(b, T, H, P, G, N, seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(b, T, H, P, generator=gen)
    dt = torch.nn.functional.softplus(torch.randn(b, T, H, generator=gen) * 0.5 - 3.0)
    A = -torch.exp(torch.linspace(0.0, 1.5, H))
    B = torch.randn(b, T, G, N, generator=gen) * 0.3
    C = torch.randn(b, T, G, N, generator=gen) * 0.3
    D = torch.rand(H, generator=gen) + 0.5
    return x, dt, A, B, C, D


def main():
    dev = torch.device("neuron")
    only = os.environ.get("SSD_CASES")
    do_bwd = os.environ.get("SSD_BWD", "1") == "1"
    cases = [CASES[int(i)] for i in only.split(",")] if only else CASES
    ok = True
    for (b, T, H, P, G, N, Q) in cases:
        tag = f"b={b} T={T:5d} H={H:2d} G={G} Q={Q}"
        x, dt, A, B, C, D = _inputs(b, T, H, P, G, N, seed=T + H + G)
        masks = _masks(Q, T // Q, x.device, torch.float32)

        ref = ssd_chunked_eager(x, dt, A, B, C, D, Q, masks)
        args = [t.to(dev) for t in (x, dt, A, B, C, D)]
        got = call_nki(*args, Q, G).cpu()
        denom = ref.abs().max().clamp(min=1e-30)
        r = ((got - ref).abs().max() / denom).item()
        fin = bool(torch.isfinite(got).all())
        good = r < 2e-5 and fin
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] fwd {tag} max_rel={r:.3e} "
              f"all_finite={fin}", flush=True)

        if not do_bwd:
            continue
        # reference gradients: autograd through the eager chunked form
        leaves = [t.clone().requires_grad_(True) for t in (x, dt, A, B, C, D)]
        gsrc = torch.randn(b, T, H, P, generator=torch.Generator().manual_seed(T * 2))
        (ssd_chunked_eager(*leaves, Q, masks) * gsrc).sum().backward()
        gargs = [t.to(dev) for t in (gsrc, x, dt, A, B, C, D)]
        gots = [t.cpu() for t in call_nki_bwd(*gargs, Q, G)]
        worst, det = 0.0, []
        for nm, g, lf in zip(("x", "dt", "A", "B", "C", "D"), gots, leaves):
            gr = lf.grad
            rr_ = ((g - gr).abs().max() / gr.abs().max().clamp(min=1e-30)).item()
            worst = max(worst, rr_)
            det.append(f"d{nm}={rr_:.1e}")
        good = worst < 5e-4 and all(bool(torch.isfinite(t).all()) for t in gots)
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] bwd {tag} {' '.join(det)}", flush=True)
    if os.environ.get("SSD_BENCH", "0") == "1":
        bench(dev)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


def bench(dev, reps=20):
    """NKI kernel vs the eager chunked form, at the real per-layer shape.

    Why this matters more than the flop count: the eager path lowers to ~25
    separate NEFFs per Mamba2 layer (one per aten op over [b,T,H,P] tensors),
    each with its own launch + HBM round trip, while the kernel is ONE NEFF that
    keeps every intermediate in SBUF. The win is launch count and HBM traffic,
    not arithmetic, so it can only be measured on the device.
    """
    b, T, H, P, G, N, Q = 1, 2048, 10, 128, 1, 128, 128
    x, dt, A, B, C, D = _inputs(b, T, H, P, G, N, seed=7)
    masks = _masks(Q, T // Q, torch.device("cpu"), torch.float32)
    dargs = [t.to(dev) for t in (x, dt, A, B, C, D)]
    dmask = tuple(m.to(dev) for m in masks) if isinstance(masks, tuple) else masks.to(dev)
    gsrc = torch.randn(b, T, H, P, generator=torch.Generator().manual_seed(3)).to(dev)

    def timed(fn, label):
        fn(); torch.neuron.synchronize() if hasattr(torch, "neuron") else None
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if hasattr(torch, "neuron"):
            torch.neuron.synchronize()
        ms = (time.perf_counter() - t0) * 1e3 / reps
        print(f"  {label:34s} {ms:8.3f} ms/call", flush=True)
        return ms

    print(f"\n== bench at b={b} T={T} H={H} P={P} N={N} Q={Q} (per Mamba2 layer) ==")
    n_ms = timed(lambda: call_nki(*dargs, Q, G), "NKI fwd")
    e_ms = timed(lambda: ssd_chunked_eager(*dargs, Q, dmask), "eager fwd")
    print(f"  fwd speedup: {e_ms / max(n_ms, 1e-9):.2f}x")
    nb_ms = timed(lambda: call_nki_bwd(gsrc, *dargs, Q, G), "NKI bwd")

    def eager_bwd():
        lv = [t.detach().clone().requires_grad_(True) for t in dargs]
        (ssd_chunked_eager(*lv, Q, dmask) * gsrc).sum().backward()

    eb_ms = timed(eager_bwd, "eager fwd+bwd (autograd)")
    print(f"  fwd+bwd: NKI {n_ms + nb_ms:.3f} ms vs eager {eb_ms:.3f} ms "
          f"-> {eb_ms / max(n_ms + nb_ms, 1e-9):.2f}x")


if __name__ == "__main__":
    sys.exit(main())
