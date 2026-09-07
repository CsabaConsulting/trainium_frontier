"""Verify the loop-free eager Mamba2 block against the sequential-scan ground truth."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba2_ssd_ref import (ssd_recurrent, ssd_chunked, grouped_causal_conv1d,
                            make_inputs)
from mamba2_block_eager import (ssd_chunked_matmul, causal_conv1d, Mamba2Mixer,
                                rms_norm_gated)

torch.set_num_threads(1)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-30)).item()


def t1_loopfree_vs_recurrent():
    print("[1] loop-free chunked SSD vs sequential recurrence (float64, must be EXACT)")
    ok = True
    for cfg, Q in [(dict(bsz=2, T=256, H=4, P=16, G=2, N=8), 64),
                   (dict(bsz=1, T=128, H=1, P=8, G=1, N=8), 32),
                   (dict(bsz=3, T=384, H=6, P=32, G=3, N=16), 128),
                   (dict(bsz=1, T=512, H=10, P=128, G=1, N=128), 128),   # competition shape
                   (dict(bsz=1, T=64, H=8, P=4, G=1, N=4), 64)]:        # single chunk
        inp = make_inputs(**cfg, seed=sum(cfg.values()) if all(
            isinstance(v, int) for v in cfg.values()) else 0)
        inp.pop("Q")
        y_ref = ssd_recurrent(**inp)
        y_lf = ssd_chunked_matmul(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"],
                                  inp["D"], Q, acc_dtype=torch.float64)
        r = rel(y_lf, y_ref)
        print(f"  T={cfg['T']:4d} H={cfg['H']:2d} P={cfg['P']:3d} G={cfg['G']} "
              f"N={cfg['N']:3d} Q={Q:3d}  max_rel={r:.3e}")
        ok &= r < 1e-12
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def t2_loopfree_vs_loop_chunked():
    print("[2] loop-free vs the loop-based chunked reference (same algorithm, no loop)")
    inp = make_inputs(bsz=2, T=384, H=6, P=16, G=2, N=8, seed=7)
    inp.pop("Q")
    ok = True
    for Q in (32, 64, 96, 128, 192, 384):
        y_loop = ssd_chunked(**inp, chunk_size=Q)
        y_lf = ssd_chunked_matmul(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"],
                                  inp["D"], Q, acc_dtype=torch.float64)
        r = rel(y_lf, y_loop)
        print(f"  Q={Q:4d}  max_rel={r:.3e}")
        ok &= r < 1e-12
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def t3_conv_equiv():
    print("[3] shift-based causal_conv1d vs F.conv1d reference")
    g = torch.Generator().manual_seed(17)
    u = torch.randn(2, 40, 12, dtype=torch.float64, generator=g)
    w = torch.randn(12, 4, dtype=torch.float64, generator=g)
    r = rel(causal_conv1d(u, w), grouped_causal_conv1d(u, w))
    print(f"  max_rel={r:.3e}")
    # causality
    u.requires_grad_(True)
    worst = 0.0
    for tp in (0, 1, 3, 20, 39):
        u.grad = None
        causal_conv1d(u, w)[:, tp].sum().backward()
        gf = u.grad[:, tp + 1:]
        worst = max(worst, gf.abs().max().item() if gf.numel() else 0.0)
    print(f"  causality: worst future |grad| = {worst:.3e}")
    ok = r < 1e-13 and worst == 0.0
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def t4_grads():
    print("[4] gradients of loop-free vs sequential recurrence (float64)")
    inp = make_inputs(bsz=2, T=96, H=3, P=8, G=1, N=8, seed=9, requires_grad=True)
    inp.pop("Q")
    gsrc = torch.randn(2, 96, 3, 8, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(42))
    out = {}
    for name, fn in (("rec", lambda: ssd_recurrent(**inp)),
                     ("lf", lambda: ssd_chunked_matmul(
                         inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], inp["D"],
                         24, acc_dtype=torch.float64))):
        for v in inp.values():
            v.grad = None
        (fn() * gsrc).sum().backward()
        out[name] = {k: v.grad.clone() for k, v in inp.items()}
    ok = True
    for k in inp:
        r = rel(out["lf"][k], out["rec"][k])
        print(f"  d/d{k:2s}  max_rel={r:.3e}")
        ok &= r < 1e-11
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def t5_mixer_causality_and_shapes():
    print("[5] full Mamba2Mixer: shapes, causality, grad flow to every parameter")
    torch.manual_seed(0)
    m = Mamba2Mixer(n_embd=64, d_state=16, head_dim=16, expand=2, n_groups=1,
                    chunk_size=8).double()
    u = torch.randn(2, 32, 64, dtype=torch.float64, requires_grad=True)
    y = m(u)
    print(f"  in {tuple(u.shape)} -> out {tuple(y.shape)}")
    ok = y.shape == u.shape

    # causality of the WHOLE mixer (conv + scan + gate + norm + projections)
    worst = 0.0
    for tp in (0, 1, 7, 8, 15, 31):
        u.grad = None
        m.zero_grad()
        m(u)[:, tp].sum().backward()
        gf = u.grad[:, tp + 1:]
        worst = max(worst, gf.abs().max().item() if gf.numel() else 0.0)
    print(f"  causality: worst future |grad| = {worst:.3e} (must be exactly 0)")
    ok &= worst == 0.0

    # every parameter must receive a gradient (no dead weights -- the exact failure
    # mode train.py's own CausalConv was just found to have)
    m.zero_grad()
    u.grad = None
    m(u).sum().backward()
    dead = [n for n, p in m.named_parameters()
            if p.grad is None or p.grad.abs().max().item() == 0.0]
    print(f"  parameters with zero/None grad: {dead if dead else 'none'}")
    ok &= not dead
    for n, p in m.named_parameters():
        print(f"    {n:16s} shape={str(tuple(p.shape)):>14s}  "
              f"|grad|max={p.grad.abs().max():.3e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def t6_mixer_dt_and_A_sanity():
    print("[6] Mamba2Mixer init sanity: dt in [dt_min, dt_max], A < 0")
    m = Mamba2Mixer(n_embd=640, d_state=128, head_dim=128, expand=2, n_groups=1,
                    chunk_size=128)
    dt0 = F.softplus(m.dt_bias.detach())
    A = -torch.exp(m.A_log.detach())
    print(f"  n_heads={m.n_heads} d_inner={m.d_inner} d_conv={m.d_conv}")
    print(f"  softplus(dt_bias) in [{dt0.min():.5f}, {dt0.max():.5f}] "
          f"(target [{m.dt_min}, {m.dt_max}])")
    print(f"  A in [{A.min():.3f}, {A.max():.3f}] (must all be < 0)")
    ok = bool((dt0 >= m.dt_min * 0.5).all() and (dt0 <= m.dt_max * 1.5).all()
              and (A < 0).all())
    nparam = sum(p.numel() for p in m.parameters())
    print(f"  params per mixer = {nparam:,}")
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def t7_fp32_competition_shape():
    print("[7] fp32 accuracy at the real competition shape (b=1 T=2048 H=10 P=128 N=128)")
    inp = make_inputs(bsz=1, T=2048, H=10, P=128, G=1, N=128, dtype=torch.float32, seed=21)
    inp.pop("Q")
    i64 = {k: (v.double() if torch.is_tensor(v) else v) for k, v in inp.items()}
    y64 = ssd_recurrent(**i64)
    ok = True
    for Q in (64, 128, 256):
        y32 = ssd_chunked_matmul(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"],
                                 inp["D"], Q)
        r = rel(y32.double(), y64)
        finite = bool(torch.isfinite(y32).all())
        print(f"  Q={Q:3d}  fp32 vs fp64-sequential max_rel={r:.3e}  all_finite={finite}")
        ok &= (r < 1e-4 and finite)
    # bf16 inputs, fp32 accumulate (what ships)
    ib = {k: (v.bfloat16() if torch.is_tensor(v) and v.dtype == torch.float32 else v)
          for k, v in inp.items()}
    yb = ssd_chunked_matmul(ib["x"].float(), ib["dt"].float(), ib["A"].float(),
                            ib["B"].float(), ib["C"].float(), ib["D"].float(), 128)
    print(f"  bf16-rounded inputs, fp32 accum: max_rel vs fp64 = {rel(yb.double(), y64):.3e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


if __name__ == "__main__":
    res = {
        "loopfree_vs_recurrent": t1_loopfree_vs_recurrent(),
        "loopfree_vs_loop": t2_loopfree_vs_loop_chunked(),
        "conv_equiv": t3_conv_equiv(),
        "grads": t4_grads(),
        "mixer": t5_mixer_causality_and_shapes(),
        "init_sanity": t6_mixer_dt_and_A_sanity(),
        "fp32_shape": t7_fp32_competition_shape(),
    }
    print("=" * 70)
    for k, v in res.items():
        print(f"  {k:24s} {'PASS' if v else 'FAIL'}")
    print("ALL PASS" if all(res.values()) else "SOME FAILED")
