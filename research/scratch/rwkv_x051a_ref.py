"""Independent references for the RWKV "x051a" time-mixing operator.

Three implementations of the SAME operator, written deliberately differently so
they can cross-check each other in float64:

  1. `rwkv_recurrent`  -- naive per-token sequential recurrence with an explicit
     [N,N] state. O(T) python steps. This is the DEFINITION we hold everything
     else to; it is written from the recurrence, not from the chunked form.
  2. `rwkv_quadratic`  -- naive O(T^2) "attention-like" form: build the full
     [T,T] decay matrix directly from its closed form and contract. Written
     from the closed-form solution, independently of (1).
  3. `rwkv_reference_chunked` -- a LITERAL transcription of nanoRWKV's
     `RWKV_TimeMix_x051a.forward` chunk loop (the tile/flip/view decay-matrix
     construction included, verbatim), so that train.py's rewritten
     construction can be checked against the upstream code it came from and not
     merely against my own algebra.

The operator, per head (r,k,v are [T,N], w and u are scalars per head):

    y_t = u * (r_t . k_t) * v_t  +  sum_{s<t} w^(t-s-1) * (r_t . k_s) * v_s

which is strictly causal: position t reads only s <= t.

Run: python research/scratch/rwkv_x051a_ref.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def make_inputs(bsz: int, T: int, H: int, N: int, seed: int = 0, dtype=torch.float64):
    """Random r/k/v plus a decay/bonus pair in the same parameterisation as x051a.

    `time_decay` is the raw parameter: w = exp(-exp(time_decay)), so w is in (0,1)
    for any real time_decay. Draws it over roughly the init range (-6..-1) plus a
    couple of extreme values so the checks also exercise near-1 and small w.
    """
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(bsz, T, H, N, generator=g, dtype=dtype) * 0.5
    k = torch.randn(bsz, T, H, N, generator=g, dtype=dtype) * 0.5
    v = torch.randn(bsz, T, H, N, generator=g, dtype=dtype)
    time_decay = (torch.rand(H, 1, generator=g, dtype=dtype) * 5.0 - 6.0)
    if H >= 2:
        time_decay[0, 0] = -6.0      # w ~ 0.99752 (very slow decay)
        time_decay[-1, 0] = 1.0      # w ~ 0.06599 (very fast decay)
    time_faaaa = torch.randn(H, 1, generator=g, dtype=dtype) * 0.5
    return dict(r=r, k=k, v=v, time_decay=time_decay, time_faaaa=time_faaaa)


def rwkv_recurrent(r, k, v, time_decay, time_faaaa):
    """Definition: per-token sequential recurrence with an explicit [N,N] state.

    S_0 = 0;  y_t = r_t @ S_t + u * (r_t . k_t) * v_t;  S_{t+1} = w * S_t + k_t (x) v_t

    S_t = sum_{s<t} w^(t-1-s) k_s (x) v_s, so r_t @ S_t is exactly the history term.
    """
    b, T, H, N = r.shape
    w = torch.exp(-torch.exp(time_decay)).reshape(1, H, 1, 1)   # (0,1)
    u = time_faaaa.reshape(1, H, 1, 1)
    state = torch.zeros(b, H, N, N, dtype=r.dtype)
    out = torch.zeros(b, T, H, N, dtype=r.dtype)
    for t in range(T):
        rt = r[:, t].reshape(b, H, 1, N)          # [b,H,1,N]
        kt = k[:, t].reshape(b, H, N, 1)
        vt = v[:, t].reshape(b, H, 1, N)
        hist = torch.matmul(rt, state)            # [b,H,1,N]
        bonus = u * torch.matmul(rt, kt) * vt     # (r.k) * v, scaled by u
        out[:, t] = (hist + bonus).reshape(b, H, N)
        state = w * state + torch.matmul(kt, vt)
    return out


def rwkv_quadratic(r, k, v, time_decay, time_faaaa):
    """Closed form: the full [T,T] decay matrix, built directly from its definition.

    D[t,s] = u if s == t, w^(t-s-1) if s < t, 0 if s > t.
    y = (D * (r @ k^T)) @ v
    """
    b, T, H, N = r.shape
    w = torch.exp(-torch.exp(time_decay)).reshape(H, 1, 1)
    u = time_faaaa.reshape(H, 1, 1)
    ti = torch.arange(T).reshape(T, 1)
    si = torch.arange(T).reshape(1, T)
    D = torch.zeros(H, T, T, dtype=r.dtype)
    lower = (si < ti)
    exponent = (ti - si - 1).clamp(min=0).to(r.dtype)
    D = torch.where(lower, w ** exponent, torch.zeros((), dtype=r.dtype))
    D = D + u * torch.eye(T, dtype=r.dtype)
    rh = r.permute(0, 2, 1, 3)                     # [b,H,T,N]
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    y = torch.matmul(D.unsqueeze(0) * torch.matmul(rh, kh.transpose(-1, -2)), vh)
    return y.permute(0, 2, 1, 3)


def rwkv_reference_chunked(r_in, k_in, v_in, time_decay, time_faaaa, Q):
    """Verbatim transcription of nanoRWKV RWKV_TimeMix_x051a.forward's chunk loop.

    Copied from /nanoRWKV/model.py with only the projections/time-shift removed
    (r/k/v arrive already projected) so the operator itself is untouched --
    including the torch.tile / .flip() / .view() decay-matrix construction, which
    is exactly what train.py's rewritten construction has to reproduce.
    """
    b, T, H, N = r_in.shape
    assert T % Q == 0
    r = r_in.transpose(1, 2)                       # [B,H,T,N]
    k = k_in.permute(0, 2, 3, 1)                   # [B,H,N,T]
    v = v_in.transpose(1, 2)                       # [B,H,T,N]

    w = torch.exp(-torch.exp(time_decay))          # [H,1]
    u = time_faaaa

    ws = w.pow(Q).view(1, H, 1, 1)

    ind = torch.arange(Q - 1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
    w = w.repeat(1, Q).pow(ind)

    wk = w.view(1, H, 1, Q)
    wb = wk.transpose(-2, -1).flip(2)

    w = torch.cat([w[:, 1:], u], dim=1)
    w = F.pad(w, (0, Q))
    w = torch.tile(w, [Q])
    w = w[:, :-Q].view(-1, Q, 2 * Q - 1)
    w = w[:, :, Q - 1:].view(1, H, Q, Q)

    w = w.to(dtype=r.dtype)
    wk = wk.to(dtype=r.dtype)
    wb = wb.to(dtype=r.dtype)
    ws = ws.to(dtype=r.dtype)

    state = torch.zeros(b, H, N, N, device=r.device, dtype=r.dtype)
    y = torch.empty(b, H, T, N, device=r.device, dtype=r.dtype)
    for i in range(T // Q):
        rr = r[:, :, i * Q:i * Q + Q, :]
        kk = k[:, :, :, i * Q:i * Q + Q]
        vv = v[:, :, i * Q:i * Q + Q, :]
        y[:, :, i * Q:i * Q + Q, :] = ((rr @ kk) * w) @ vv + (rr @ state) * wb
        state = ws * state + (kk * wk) @ vv
    return y.transpose(1, 2)


def _relmax(a, b):
    denom = b.abs().max()
    return ((a - b).abs().max() / denom).item() if denom > 0 else (a - b).abs().max().item()


def main() -> None:
    torch.set_num_threads(1)
    fails = []

    def check(name, cond, detail=""):
        tag = "PASS" if cond else "FAIL"
        if not cond:
            fails.append(name)
        print(f"[{tag}] {name}{('  ' + detail) if detail else ''}")

    print("== the three independent references agree (float64) ==")
    for cfg, Q in [(dict(bsz=2, T=64, H=3, N=8), 16),
                   (dict(bsz=1, T=128, H=4, N=16), 32),
                   (dict(bsz=1, T=96, H=2, N=8), 96),
                   (dict(bsz=2, T=48, H=5, N=4), 16)]:
        inp = make_inputs(**cfg, seed=cfg["T"] * 7 + cfg["H"])
        y_rec = rwkv_recurrent(**inp)
        y_quad = rwkv_quadratic(**inp)
        y_ch = rwkv_reference_chunked(inp["r"], inp["k"], inp["v"],
                                      inp["time_decay"], inp["time_faaaa"], Q)
        tag = f"T={cfg['T']} H={cfg['H']} N={cfg['N']} Q={Q}"
        check(f"recurrent == quadratic  {tag}", _relmax(y_quad, y_rec) < 1e-12,
              f"max_rel={_relmax(y_quad, y_rec):.3e}")
        check(f"recurrent == nanoRWKV chunked  {tag}", _relmax(y_ch, y_rec) < 1e-12,
              f"max_rel={_relmax(y_ch, y_rec):.3e}")

    print("\n== strict causality of the operator itself ==")
    # Perturb one input position and confirm no earlier output moves.
    inp = make_inputs(bsz=1, T=32, H=2, N=6, seed=3)
    base = rwkv_recurrent(**inp)
    worst_before = 0.0
    for tp in (1, 7, 16, 31):
        pert = {kk: vv.clone() for kk, vv in inp.items()}
        pert["r"][:, tp] += 3.0
        pert["k"][:, tp] += 3.0
        pert["v"][:, tp] += 3.0
        out = rwkv_recurrent(**pert)
        d = (out[:, :tp] - base[:, :tp]).abs().max().item() if tp > 0 else 0.0
        worst_before = max(worst_before, d)
        print(f"       perturb t={tp:2d}: max|delta| at positions < t = {d:.3e}")
    check("perturbing position t never moves any output < t", worst_before == 0.0,
          f"worst={worst_before:.3e}")

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
