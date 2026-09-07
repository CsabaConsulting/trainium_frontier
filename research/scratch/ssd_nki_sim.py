"""CPU simulator of the NKI SSD kernel's dataflow, op for op.

WHY THIS EXISTS. `ssd_nki_dev.py` cannot even be imported without a Neuron
install, so on a laptop there is no way to test it -- and hardware slots on the
shared Trn2 box are scarce. But almost every bug I can make in that kernel is an
ALGEBRA bug (wrong operand order in `stationary.T @ moving`, a broadcast that
silently uses the wrong axis, an index off by one in the `ellb[:, Q-1]` trick),
not a NKI-API bug. Those are all reproducible on CPU if the simulator is
faithful to the hardware's semantics rather than to what I meant.

So this file re-implements the kernel with a tiny set of primitives that mimic
the ISA:

  mm(stationary, moving)   -> stationary.T @ moving, contraction on axis 0 (the
                              partition axis) of BOTH operands. This is the one
                              that catches transposed-operand mistakes.
  ts(data, op, operand)    -> tensor_scalar: `operand` is either a python scalar
                              or a [P,1] column. A [1,F] row is REJECTED, because
                              compute engines cannot broadcast along partitions.
  tt(a, b, op)             -> tensor_tensor: shapes must match EXACTLY. No numpy
                              broadcasting, because the hardware has none.
  tr(data)                 -> nc_transpose.

Then it walks the same loop nest as the kernel (batch row, group, head, chunk)
with the same explicit SBUF/PSUM tiles, and compares against the independent
eager reference. Anything that passes here and still fails on device is a real
NKI issue, which is a much smaller search space.

Run: python research/scratch/ssd_nki_sim.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from mamba2_ssd_ref import make_inputs, ssd_recurrent

torch.set_num_threads(1)

_NEG = -30000.0
_PMAX = 128
FAILS = []


# ---------------------------------------------------------------------------
# ISA-faithful primitives. Every one of these asserts the hardware's shape rules
# so that a mistake fails here rather than silently working via broadcasting.

def mm(stationary, moving):
    """nisa.nc_matmul: dst = stationary.T @ moving, contraction on axis 0."""
    K, M = stationary.shape
    K2, Nf = moving.shape
    assert K == K2, f"contraction mismatch {stationary.shape} x {moving.shape}"
    assert K <= _PMAX, f"matmul K={K} > 128"
    assert M <= _PMAX, f"stationary free {M} > 128"
    assert Nf <= 512, f"moving free {Nf} > 512"
    return stationary.transpose(0, 1) @ moving          # [M, Nf] in PSUM


def tr(data):
    """nisa.nc_transpose: [P,F] -> [F,P], both dims <= 128."""
    P, F = data.shape
    assert P <= _PMAX and F <= _PMAX, f"transpose {data.shape} exceeds 128x128"
    return data.transpose(0, 1).contiguous()


def _chk_operand(data, operand):
    if torch.is_tensor(operand):
        assert operand.dim() == 2 and operand.shape[0] == data.shape[0] \
            and operand.shape[1] == 1, (
            f"tensor_scalar operand must be [{data.shape[0]},1], got "
            f"{tuple(operand.shape)} -- a [1,F] row would need a partition-axis "
            f"broadcast, which compute engines cannot do")


def ts(data, op, operand, op1=None, operand1=None, reverse0=False):
    """nisa.tensor_scalar: (data op operand) op1 operand1, or with reverse0=True
    (operand op data) op1 operand1."""
    _chk_operand(data, operand)
    out = op(operand, data) if reverse0 else op(data, operand)
    if op1 is not None:
        _chk_operand(data, operand1)
        out = op1(out, operand1)
    return out


def tt(a, b, op):
    """nisa.tensor_tensor: shapes must match exactly (no hardware broadcast)."""
    assert a.shape == b.shape, f"tensor_tensor shape mismatch {a.shape} {b.shape}"
    return op(a, b)


def _min0(t, _):
    return t.clamp(max=0.0)


def rr(data):
    """nisa.tensor_reduce(op=add, axis=free): [P,F] -> [P,1]. Free axis only --
    a partition-axis reduction is NOT a reduce op on this hardware, it is a
    matmul against a ones column."""
    return data.sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------

def ssd_fwd_sim(x, dt, A, B, C, D, Q, G):
    """Replay of ssd_fwd_nki. x [b,T,H,P] dt [b,T,H] B/C [b,T,G,N]."""
    b, T, H, P = x.shape
    N = B.shape[3]
    CPB = T // Q
    hpg = H // G
    assert N <= Q <= _PMAX and P <= 512 and T % Q == 0 and H % G == 0

    # --- flatten to the kernel's HBM layout ---------------------------------
    x2 = x.reshape(b * T, H * P)
    dt2 = dt.reshape(b * T, H)
    B2 = B.reshape(b * T, G * N)
    C2 = C.reshape(b * T, G * N)
    dtype = x.dtype
    y2 = torch.zeros(b * T, H * P, dtype=dtype)

    # --- constants ----------------------------------------------------------
    ri = torch.arange(Q, dtype=dtype)[:, None].expand(Q, Q)
    ci = torch.arange(Q, dtype=dtype)[None, :].expand(Q, Q)
    UI = (ri <= ci).to(dtype)                                    # [r,t] r<=t
    onesQ = torch.ones(Q, Q, dtype=dtype)
    MASK = (UI - 1.0) * (-_NEG)                                  # 0 / -30000

    for bi in range(b):
        for g in range(G):
            for hh in range(hpg):
                h = g * hpg + hh
                # A_h / D_h reach the kernel as a [128,H] tile replicated on every
                # partition (Ab/Db), so a [Q,1] column is the faithful model. Do NOT
                # use float(A[h]) -- that detaches, and then autograd cannot check dA.
                a_h = A[h].reshape(1, 1).expand(Q, 1)
                d_h = D[h].reshape(1, 1).expand(Q, 1)
                S = torch.zeros(N, P, dtype=dtype)               # SBUF state

                for c in range(CPB):
                    r0 = bi * T + c * Q
                    xt = x2[r0:r0 + Q, h * P:h * P + P]
                    dtt = dt2[r0:r0 + Q, h:h + 1]
                    Bt = B2[r0:r0 + Q, g * N:g * N + N]
                    Ct = C2[r0:r0 + Q, g * N:g * N + N]

                    dtA = ts(dtt, torch.mul, a_h)
                    dx = ts(xt, torch.mul, dtt)

                    ell = mm(UI, dtA)                            # [Q,1]
                    UIdtA = ts(UI, torch.mul, dtA)
                    ellb = mm(onesQ, UIdtA)                      # [Q_s,Q_t]=ell_t
                    eellb = ts(ellb, _min0, 0.0).exp()

                    seg = ts(ellb, torch.sub, ell, op1=_min0, operand1=0.0)
                    seg = tt(seg, MASK, torch.add)
                    L = seg.exp()

                    BTs = tr(Bt)                                 # [N,Q]
                    CTs = tr(Ct)
                    CBs = mm(BTs, CTs)                           # [Q_s,Q_t]
                    MT = tt(L, CBs, torch.mul)

                    ps_y = mm(MT, dx)                            # [Q_t,P]
                    if c > 0:
                        CbarT = tt(CTs, eellb[0:N, 0:Q], torch.mul)
                        ps_y = ps_y + mm(CbarT, S)               # PSUM accumulate

                    dxx = ts(xt, torch.mul, d_h)
                    y2[r0:r0 + Q, h * P:h * P + P] = tt(ps_y, dxx, torch.add)

                    if c + 1 < CPB:
                        rev = tt(ellb[0:Q, Q - 1:Q], ell, torch.sub)
                        rev = ts(rev, _min0, 0.0)
                        erev = rev.exp()
                        Bbar = ts(Bt, torch.mul, erev)
                        Gs = mm(Bbar, dx)                        # [N,P]
                        S = ts(S, torch.mul, eellb[0:N, Q - 1:Q])
                        S = tt(S, Gs, torch.add)

    return y2.reshape(b, T, H, P)


# ---------------------------------------------------------------------------
# BACKWARD
#
# Derivation, with u_s = dt_s x_s ("dx" in the forward), seg_{s,t} = ell_t - ell_s,
# L = exp(clamp(seg,0) + mask), MT = L * <B_s,C_t>:
#
#   dD_h    = sum_t <gy_t, x_t>                        dx_t += D_h gy_t
#   dMT_{s,t} = <u_s, gy_t>                            (contraction over P)
#   dseg    = dMT * MT        <-- because dL = dMT*CB and dseg = dL*L, so
#                                 dseg = dMT*CB*L = dMT*MT. Masked entries have
#                                 MT == 0, so this ALSO applies the clamp/mask
#                                 subgradient for free (see note on clamps below).
#   dCB     = dMT * L
#   dell_j  = sum_s dseg_{s,j} - sum_t dseg_{j,t}      (j appears as both t and s)
#   dB_{s,n} += sum_t dCB_{s,t} C_{t,n}   dC_{t,n} += sum_s dCB_{s,t} B_{s,n}
#   du_s    += sum_t MT_{s,t} gy_t
#   [inter, c>0]  dCbar_{t,n} = sum_p gy_{t,p} S_c[n,p]
#                 dC_t += dCbar_t eell_t ;  dell_t += <dCbar_t, C_t> * eell_t
#                 dS_c += sum_t Cbar_{t,n} gy_{t,p}
#   [carry, c<CPB-1]  dGst = dS_{c+1}
#                 dBbar_{s,n} = sum_p u_{s,p} dGst[n,p]
#                 du_s += sum_n Bbar_{s,n} dGst[n,:]
#                 dB_s += dBbar_s erev_s
#                 drev_s = <dBbar_s, B_s> erev_s ;  dell_s -= drev_s ; dde += sum_s drev_s
#                 dac = <dS_{c+1}, S_c> ;           dde += dac * ac
#   dS_c    = ac_c dS_{c+1} + (inter term of chunk c)
#   de = ell_{Q-1}, and ddtA_r = sum_{t>=r} dell_t, so a dde contribution lands on
#   EVERY r -- no need to inject it into a single slot of dell (which would need an
#   awkward one-hot); just add the broadcast dde to ddtA.
#   ddt_s   = ddtA_s A_h + <du_s, x_s>     dA_h = sum_s ddtA_s dt_s
#   dx_s    = dt_s du_s + D_h gy_s
#
# ON THE CLAMPS. Every clamp(., max=0) in the forward sits on an argument that is
# provably <= 0 (A<0, dt>0 so ell is strictly decreasing): seg for s<=t, ell_t,
# de-ell_s, de. torch's clamp_max backward passes gradient where `input <= max`,
# so the subgradient is exactly 1 on all of them, including the t==s / s==Q-1
# boundary cases where the argument is exactly 0. The only entries where the clamp
# is active are s>t, and those are killed by the -30000 additive mask anyway
# (MT == 0 there). So NO clamp indicator tensors are needed; the algebra above is
# the exact autograd, not an approximation. Verified by t_bwd_vs_autograd.
#
# ON dB/dC WITH n_groups < n_heads. B and C are shared by hpg = H/G heads, so
# their gradients SUM over the heads of a group. The kernel keeps a per-group SBUF
# accumulator spanning all chunks -- shape [Q, CPB*N], 8 KB/partition at the real
# geometry -- so the reduction happens in-kernel and no extra eager op is needed.
# The simulator models exactly that accumulator.

def ssd_bwd_sim(gy_in, x, dt, A, B, C, D, Q, G):
    """Replay of the planned ssd_bwd_nki. Returns dx, ddt, dA, dB, dC, dD."""
    b, T, H, P = x.shape
    N = B.shape[3]
    CPB = T // Q
    hpg = H // G
    dtype = x.dtype

    x2 = x.reshape(b * T, H * P)
    dt2 = dt.reshape(b * T, H)
    B2 = B.reshape(b * T, G * N)
    C2 = C.reshape(b * T, G * N)
    gy2 = gy_in.reshape(b * T, H * P)

    dx2 = torch.zeros(b * T, H * P, dtype=dtype)
    ddt2 = torch.zeros(b * T, H, dtype=dtype)
    dB2 = torch.zeros(b * T, G * N, dtype=dtype)
    dC2 = torch.zeros(b * T, G * N, dtype=dtype)
    # accA/accD are [Q,H] elementwise accumulators; the final dA/dD is one
    # partition-axis reduction (a single matmul against a ones column) at the end.
    accA = torch.zeros(Q, H, dtype=dtype)
    accD = torch.zeros(Q, H, dtype=dtype)

    ri = torch.arange(Q, dtype=dtype)[:, None].expand(Q, Q)
    ci = torch.arange(Q, dtype=dtype)[None, :].expand(Q, Q)
    UI = (ri <= ci).to(dtype)
    LI = (ri >= ci).to(dtype)          # LI[t,r]=1 iff r<=t: reverse cumsum operator
    onesQ = torch.ones(Q, Q, dtype=dtype)
    onesQ1 = torch.ones(Q, 1, dtype=dtype)
    onesN1 = torch.ones(N, 1, dtype=dtype)
    ones1Q = torch.ones(1, Q, dtype=dtype)
    MASK = (UI - 1.0) * (-_NEG)
    MASKT = (LI - 1.0) * (-_NEG)

    for bi in range(b):
        for g in range(G):
            # per-group, all-chunk dB/dC accumulators (the [Q, CPB*N] SBUF tile)
            aB = torch.zeros(Q, CPB * N, dtype=dtype)
            aC = torch.zeros(Q, CPB * N, dtype=dtype)

            for hh in range(hpg):
                h = g * hpg + hh
                aA = A[h].reshape(1, 1).expand(Q, 1)
                aD = D[h].reshape(1, 1).expand(Q, 1)

                # ---- pass 1: recompute the forward state entering each chunk.
                # Cheaper than making the forward kernel spill CPB x [N,P] states
                # to HBM (21 MB/layer at the real geometry) and hold them as saved
                # tensors; costs ~3 matmuls per chunk instead.
                Sall = [torch.zeros(N, P, dtype=dtype) for _ in range(CPB)]
                S = torch.zeros(N, P, dtype=dtype)
                for c in range(CPB):
                    Sall[c] = S
                    if c + 1 == CPB:
                        break
                    r0 = bi * T + c * Q
                    xt = x2[r0:r0 + Q, h * P:h * P + P]
                    dtt = dt2[r0:r0 + Q, h:h + 1]
                    Bt = B2[r0:r0 + Q, g * N:g * N + N]
                    dtA = ts(dtt, torch.mul, aA)
                    u = ts(xt, torch.mul, dtt)
                    ell = mm(UI, dtA)
                    ellb = mm(onesQ, ts(UI, torch.mul, dtA))
                    eellb = ts(ellb, _min0, 0.0).exp()
                    erevc = ts(tt(ellb[0:Q, Q - 1:Q], ell, torch.sub),
                               _min0, 0.0).exp()
                    Gs = mm(ts(Bt, torch.mul, erevc), u)
                    S = tt(ts(S, torch.mul, eellb[0:N, Q - 1:Q]), Gs, torch.add)

                # ---- pass 2: reverse over chunks
                dSnext = torch.zeros(N, P, dtype=dtype)
                for c in range(CPB - 1, -1, -1):
                    r0 = bi * T + c * Q
                    xt = x2[r0:r0 + Q, h * P:h * P + P]
                    dtt = dt2[r0:r0 + Q, h:h + 1]
                    Bt = B2[r0:r0 + Q, g * N:g * N + N]
                    Ct = C2[r0:r0 + Q, g * N:g * N + N]
                    gy = gy2[r0:r0 + Q, h * P:h * P + P]

                    # --- recompute forward intermediates
                    dtA = ts(dtt, torch.mul, aA)
                    u = ts(xt, torch.mul, dtt)
                    ell = mm(UI, dtA)
                    ellb = mm(onesQ, ts(UI, torch.mul, dtA))
                    eellb = ts(ellb, _min0, 0.0).exp()
                    eellc = ts(ell, _min0, 0.0).exp()               # [Q,1]
                    seg = tt(ts(ellb, torch.sub, ell, op1=_min0, operand1=0.0),
                             MASK, torch.add)
                    L = seg.exp()
                    # BOTH ORIENTATIONS WITHOUT A TRANSPOSE. seg is NOT symmetric --
                    # raw[i,j] = min(ell_j - ell_i, 0), so raw read at [t,s] is
                    # ell_s - ell_t, the NEGATION of what L^T needs. The other
                    # orientation therefore needs its own tile, built with the
                    # subtraction reversed (reverse0=True in nisa.tensor_scalar):
                    # raw2[i,j] = min(ell_i - ell_j, 0), so raw2[t,s] = min(ell_t -
                    # ell_s, 0) = seg[s,t]. Plus the transposed causal mask MASKT.
                    segT = tt(ts(ellb, torch.sub, ell, reverse0=True,
                                 op1=_min0, operand1=0.0), MASKT, torch.add)
                    LT = segT.exp()
                    BTs = tr(Bt)
                    CTs = tr(Ct)
                    CB = mm(BTs, CTs)                                # CB[s,t]
                    MT = tt(L, CB, torch.mul)
                    MTT = tt(LT, mm(CTs, BTs), torch.mul)            # MTT[t,s]=MT[s,t]
                    gyT = tr(gy)
                    uT = tr(u)

                    # --- D / skip
                    accD[:, h:h + 1] = tt(accD[:, h:h + 1], rr(tt(gy, xt, torch.mul)),
                                          torch.add)

                    # --- intra-chunk
                    dMT = mm(uT, gyT)                                # [Q_s,Q_t]
                    dMTT = mm(gyT, uT)                               # [Q_t,Q_s]
                    dseg = tt(dMT, MT, torch.mul)
                    dCB = tt(dMT, L, torch.mul)
                    dCBT = tt(dMTT, LT, torch.mul)                   # = dCB^T
                    dell = tt(mm(dseg, onesQ1), rr(dseg), torch.sub)  # [Q,1]
                    dB = mm(dCBT, Ct)                                # [Q_s,N]
                    dC = mm(dCB, Bt)                                 # [Q_t,N]
                    du = mm(MTT, gy)                                 # [Q_s,P]

                    # --- inter-chunk output (S_0 == 0, so nothing for c == 0)
                    Cbar = ts(Ct, torch.mul, eellc)
                    if c > 0:
                        ST = tr(Sall[c])                             # [P,N]
                        dCbar = mm(gyT, ST)                          # [Q_t,N]
                        dC = tt(dC, ts(dCbar, torch.mul, eellc), torch.add)
                        deell = rr(tt(dCbar, Ct, torch.mul))         # [Q,1]
                        dell = tt(dell, tt(deell, eellc, torch.mul), torch.add)

                    # --- carry into the next chunk's state
                    dde = torch.zeros(1, 1, dtype=dtype)
                    if c + 1 < CPB:
                        dGst = dSnext
                        dBbar = mm(uT, tr(dGst))                     # [Q_s,N]
                        erevc = ts(tt(ellb[0:Q, Q - 1:Q], ell, torch.sub),
                                   _min0, 0.0).exp()
                        # Bbar^T with no transpose: erev_s = exp(min(de - ell_s, 0))
                        # and ellb[:,j] = ell_j on every partition, so the whole
                        # [N,Q] broadcast of erev falls out of ellb itself.
                        revb = ts(ellb[0:N, 0:Q], torch.sub,
                                  ellb[0:N, Q - 1:Q], reverse0=True,
                                  op1=_min0, operand1=0.0).exp()
                        BbarT = tt(BTs, revb, torch.mul)
                        du = tt(du, mm(BbarT, dGst), torch.add)
                        dB = tt(dB, ts(dBbar, torch.mul, erevc), torch.add)
                        drev = tt(rr(tt(dBbar, Bt, torch.mul)), erevc, torch.mul)
                        dell = tt(dell, drev, torch.sub)
                        dde = tt(dde, mm(drev, onesQ1), torch.add)
                        # dac = <dS_{c+1}, S_c>, then dde += dac * ac
                        pr = rr(tt(dSnext, Sall[c], torch.mul))      # [N,1]
                        dde = tt(dde, mm(ts(pr, torch.mul, eellb[0:N, Q - 1:Q]),
                                         onesN1), torch.add)

                    # --- dell -> ddtA (reverse inclusive cumsum) -> dt, A
                    ddtA = tt(mm(LI, dell), mm(ones1Q, dde), torch.add)
                    ddt = tt(ts(ddtA, torch.mul, aA), rr(tt(du, xt, torch.mul)),
                             torch.add)
                    accA[:, h:h + 1] = tt(accA[:, h:h + 1],
                                          tt(ddtA, dtt, torch.mul), torch.add)
                    dxt = tt(ts(du, torch.mul, dtt), ts(gy, torch.mul, aD), torch.add)

                    dx2[r0:r0 + Q, h * P:h * P + P] = dxt
                    ddt2[r0:r0 + Q, h:h + 1] = ddt
                    aB[:, c * N:(c + 1) * N] = tt(aB[:, c * N:(c + 1) * N], dB,
                                                  torch.add)
                    aC[:, c * N:(c + 1) * N] = tt(aC[:, c * N:(c + 1) * N], dC,
                                                  torch.add)

                    # --- dS_c = ac_c dS_{c+1} + (inter term of chunk c)
                    dSnext = tt(ts(dSnext, torch.mul, eellb[0:N, Q - 1:Q]),
                                mm(Cbar, gy), torch.add)

            for c in range(CPB):
                r0 = bi * T + c * Q
                dB2[r0:r0 + Q, g * N:g * N + N] = aB[:, c * N:(c + 1) * N]
                dC2[r0:r0 + Q, g * N:g * N + N] = aC[:, c * N:(c + 1) * N]

    dA = mm(accA, onesQ1).reshape(H)     # partition reduce, one matmul for all H
    dD = mm(accD, onesQ1).reshape(H)
    return (dx2.reshape(b, T, H, P), ddt2.reshape(b, T, H), dA,
            dB2.reshape(b, T, G, N), dC2.reshape(b, T, G, N), dD)


# ---------------------------------------------------------------------------

def rel(a, r):
    return ((a - r).abs().max() / r.abs().max().clamp(min=1e-30)).item()


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def t_vs_recurrent():
    print("== kernel dataflow (simulated) vs naive sequential recurrence, float64 ==")
    for cfg, Q in [(dict(bsz=1, T=64, H=2, P=8, G=1, N=4), 64),     # single chunk
                   (dict(bsz=1, T=128, H=2, P=8, G=1, N=4), 64),    # two chunks
                   (dict(bsz=2, T=192, H=6, P=16, G=3, N=8), 64),   # groups, hpg=2
                   (dict(bsz=2, T=256, H=4, P=32, G=1, N=32), 32),  # N<Q, 8 chunks
                   (dict(bsz=1, T=384, H=3, P=128, G=1, N=128), 128)]:  # N==Q==128
        inp = make_inputs(**cfg, seed=cfg["T"] * 7 + cfg["H"])
        inp.pop("Q")
        y_ref = ssd_recurrent(**inp)
        y_sim = ssd_fwd_sim(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"],
                            inp["D"], Q, cfg["G"])
        r = rel(y_sim, y_ref)
        check(f"T={cfg['T']:4d} H={cfg['H']:2d} P={cfg['P']:3d} G={cfg['G']} "
              f"N={cfg['N']:3d} Q={Q:3d}", r < 1e-12, f"max_rel={r:.3e}")


def t_degenerate():
    print("\n== degenerate inputs ==")
    inp = make_inputs(bsz=1, T=128, H=2, P=8, G=1, N=8, seed=3)
    inp.pop("Q")
    # dt -> 0: y must collapse to the skip term D*x
    z = dict(inp)
    z["dt"] = torch.full_like(inp["dt"], 1e-12)
    y = ssd_fwd_sim(z["x"], z["dt"], z["A"], z["B"], z["C"], z["D"], 64, 1)
    tgt = z["D"].reshape(1, 1, -1, 1) * z["x"]
    check("dt->0 collapses to D*x", rel(y, tgt) < 1e-8, f"max_rel={rel(y, tgt):.3e}")
    # D=0 and B=0: no state ever accumulates, output must be exactly 0
    z = dict(inp)
    z["B"] = torch.zeros_like(inp["B"])
    z["D"] = torch.zeros_like(inp["D"])
    y = ssd_fwd_sim(z["x"], z["dt"], z["A"], z["B"], z["C"], z["D"], 64, 1)
    check("B=0,D=0 gives exactly 0", bool((y == 0).all()),
          f"max|y|={y.abs().max().item():.3e}")


def t_no_overflow():
    print("\n== masked-exponent overflow regression (the sibling-repo bug) ==")
    # Large |A| * dt * T makes the s>t exponents huge and positive. Masking AFTER
    # exp would do inf * 0 = NaN here; masking before exp must stay finite.
    inp = make_inputs(bsz=1, T=384, H=2, P=8, G=1, N=8, seed=5)
    inp.pop("Q")
    inp["A"] = torch.full_like(inp["A"], -12.0)
    inp["dt"] = torch.full_like(inp["dt"], 0.4)
    y = ssd_fwd_sim(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], inp["D"],
                    128, 1)
    y_ref = ssd_recurrent(**inp)
    check("|A|*dt*Q ~ 615: all finite", bool(torch.isfinite(y).all()))
    check("|A|*dt*Q ~ 615: matches recurrence", rel(y, y_ref) < 1e-10,
          f"max_rel={rel(y, y_ref):.3e}")


def t_causality():
    print("\n== strict causality of the simulated kernel ==")
    inp = make_inputs(bsz=1, T=128, H=2, P=8, G=1, N=8, seed=9, requires_grad=True)
    inp.pop("Q")
    worst = 0.0
    for tp in (0, 1, 5, 63, 64, 100, 127):
        for v in inp.values():
            v.grad = None
        y = ssd_fwd_sim(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"],
                        inp["D"], 64, 1)
        y[0, tp].sum().backward()
        gf = inp["x"].grad[0, tp + 1:]
        worst = max(worst, gf.abs().max().item() if gf.numel() else 0.0)
    check("no gradient to any later input", worst == 0.0, f"worst={worst:.3e}")


def t_fp32_real_shape():
    print("\n== fp32 at the real per-layer shape (H=10 P=128 N=128 Q=128) ==")
    inp = make_inputs(bsz=1, T=512, H=10, P=128, G=1, N=128, dtype=torch.float32,
                      seed=21)
    inp.pop("Q")
    i64 = {k: (v.double() if torch.is_tensor(v) else v) for k, v in inp.items()}
    y64 = ssd_recurrent(**i64)
    y32 = ssd_fwd_sim(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], inp["D"],
                      128, 1)
    r = rel(y32.double(), y64)
    check("fp32 sim vs fp64 recurrence", r < 1e-4 and bool(torch.isfinite(y32).all()),
          f"max_rel={r:.3e}")


def t_bwd_vs_autograd():
    print("\n== backward dataflow (simulated) vs autograd of the forward, float64 ==")
    for cfg, Q in [(dict(bsz=1, T=64, H=2, P=8, G=1, N=4), 64),      # single chunk
                   (dict(bsz=1, T=128, H=2, P=8, G=1, N=4), 64),     # two chunks
                   (dict(bsz=2, T=192, H=6, P=16, G=3, N=8), 64),    # hpg=2 -> dB/dC
                   (dict(bsz=2, T=256, H=4, P=32, G=1, N=32), 32),   # 8 chunks, N<Q
                   (dict(bsz=1, T=256, H=4, P=16, G=1, N=8), 64)]:   # G=1, hpg=4
        inp = make_inputs(**cfg, seed=cfg["T"] * 3 + cfg["N"], requires_grad=True)
        inp.pop("Q")
        gsrc = torch.randn(inp["x"].shape, dtype=torch.float64,
                           generator=torch.Generator().manual_seed(cfg["T"]))
        for v in inp.values():
            v.grad = None
        y = ssd_fwd_sim(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], inp["D"],
                        Q, cfg["G"])
        (y * gsrc).sum().backward()
        got = ssd_bwd_sim(gsrc, inp["x"].detach(), inp["dt"].detach(),
                          inp["A"].detach(), inp["B"].detach(), inp["C"].detach(),
                          inp["D"].detach(), Q, cfg["G"])
        worst = 0.0
        detail = []
        for name, g in zip(("x", "dt", "A", "B", "C", "D"), got):
            r = rel(g, inp[name].grad)
            worst = max(worst, r)
            detail.append(f"d{name}={r:.1e}")
        check(f"T={cfg['T']:4d} H={cfg['H']:2d} P={cfg['P']:3d} G={cfg['G']} "
              f"N={cfg['N']:3d} Q={Q:3d}", worst < 1e-11, " ".join(detail))


def t_bwd_vs_recurrent_autograd():
    print("\n== backward vs autograd of the NAIVE SEQUENTIAL scan (independent) ==")
    # The strongest check: the simulated backward is compared to autograd through
    # ssd_recurrent, which shares no code with the chunked algorithm at all.
    cfg, Q = dict(bsz=2, T=128, H=4, P=16, G=2, N=8), 32
    inp = make_inputs(**cfg, seed=77, requires_grad=True)
    inp.pop("Q")
    gsrc = torch.randn(inp["x"].shape, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(4))
    for v in inp.values():
        v.grad = None
    (ssd_recurrent(**inp) * gsrc).sum().backward()
    got = ssd_bwd_sim(gsrc, inp["x"].detach(), inp["dt"].detach(), inp["A"].detach(),
                      inp["B"].detach(), inp["C"].detach(), inp["D"].detach(),
                      Q, cfg["G"])
    for name, g in zip(("x", "dt", "A", "B", "C", "D"), got):
        r = rel(g, inp[name].grad)
        check(f"d{name} vs sequential-scan autograd", r < 1e-10, f"max_rel={r:.3e}")


def t_bwd_fp32_real_shape():
    print("\n== backward in fp32 at the real per-layer shape ==")
    inp = make_inputs(bsz=1, T=512, H=10, P=128, G=1, N=128, dtype=torch.float32,
                      seed=31)
    inp.pop("Q")
    gsrc = torch.randn(inp["x"].shape, generator=torch.Generator().manual_seed(8))
    i64 = {k: v.double().requires_grad_(True) for k, v in inp.items()}
    (ssd_recurrent(**i64) * gsrc.double()).sum().backward()
    got = ssd_bwd_sim(gsrc, inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"],
                      inp["D"], 128, 1)
    for name, g in zip(("x", "dt", "A", "B", "C", "D"), got):
        r = rel(g.double(), i64[name].grad)
        check(f"fp32 d{name} vs fp64 sequential autograd", r < 5e-4,
              f"max_rel={r:.3e}")


if __name__ == "__main__":
    t_vs_recurrent()
    t_degenerate()
    t_no_overflow()
    t_causality()
    t_fp32_real_shape()
    t_bwd_vs_autograd()
    t_bwd_vs_recurrent_autograd()
    t_bwd_fp32_real_shape()
    print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
    sys.exit(1 if FAILS else 0)
