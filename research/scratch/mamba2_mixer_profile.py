"""Where does the Mamba2 mixer's time actually go, per component, on device?

CONTEXT (all measured, 4 ranks, lnc=2, DEPTH=6, SEQ_LEN=2048, per-rank micro
batch b=2 x T=2048, grad_accum=8):

    transformer   eager 2.18 s/step  ->  compiled 0.92 s/step   (2.37x)
    Mamba2 MMAMMM eager 1.96 s/step  ->  compiled 2.17 s/step   (0.90x, WORSE)
    fixed 300 steps: Mamba2 1.270639 bpb vs transformer 1.269458 bpb (dead heat,
    at 45.0M vs 55.7M params)

Dynamo produces ONE graph with zero breaks for both architectures, so the missing
2.37x is not a graph break -- the neuron backend just compiles this graph badly.
That leaves "shrink the graph with more kernels" as the only lever I control, and
whether that lever is long enough is a question about op-level cost, not opinion:
a Mamba2 step has 8 grad_accum x 5 Mamba2 layers = 40 mixer invocations, so the
whole 1.96 s budget is 49 ms per mixer call (fwd+bwd). The NKI SSD scan already
accounts for ~6.3 ms of that. This script measures the rest.

Run on the device box:  python research/scratch/mamba2_mixer_profile.py
"""

import os
import sys
import time

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import torch
import torch.nn.functional as F

import train as T_

REPS = 10


def sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def timed(fn, label, out):
    fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
    sync()
    ms = (time.perf_counter() - t0) * 1e3 / REPS
    out.append((label, ms))
    print(f"  {label:36s} {ms:9.3f} ms", flush=True)
    return ms


def main():
    dev = torch.device("neuron")
    b, T = 2, T_.SEQ_LEN
    T_.USE_MAMBA2 = True
    T_.USE_NKI_SSD = T_._HAS_NKI
    cfg = T_.build_config()          # the locked shape: n_embd=640, n_head=5
    n_embd = cfg.n_embd
    mx = T_.Mamba2Mixer(cfg).to(dev)
    H, P, N, G, Q = mx.n_heads, mx.head_dim, mx.d_state, mx.n_groups, mx.chunk
    print(f"mixer: n_embd={n_embd} d_inner={mx.d_inner} H={H} P={P} N={N} G={G} "
          f"d_conv={mx.d_conv} Q={Q}   micro batch b={b} T={T}")
    print(f"nki_ssd={T_.USE_NKI_SSD}  has_nki={T_._HAS_NKI}\n")

    u = torch.randn(b, T, n_embd, device=dev)
    gsrc = torch.randn(b, T, n_embd, device=dev)
    res = []

    # --- the whole mixer, forward and forward+backward
    print("== whole mixer ==")
    with torch.no_grad():
        f_ms = timed(lambda: mx(u), "mixer forward (no_grad)", res)

    def fwd_bwd():
        mx.zero_grad(set_to_none=True)
        ui = u.detach().clone().requires_grad_(True)
        (mx(ui) * gsrc).sum().backward()

    fb_ms = timed(fwd_bwd, "mixer forward+backward", res)

    # --- components, forward only, at the exact shapes the mixer uses
    print("\n== components (forward only, no_grad) ==")
    with torch.no_grad():
        proj = mx.in_proj(u)
        xBC = proj[..., mx.d_inner:mx.d_inner + mx.d_conv].contiguous()
        cw = mx.conv_weight.to(xBC.dtype)
        timed(lambda: mx.in_proj(u), "in_proj  [b,T,E]->[b,T,2E+d_conv+H]", res)
        timed(lambda: T_.ssd_causal_conv1d(xBC, cw), "causal_conv1d (k=4, depthwise)", res)
        conv = T_.ssd_causal_conv1d(xBC, cw)
        timed(lambda: F.silu(conv), "silu on [b,T,d_conv]", res)
        x, Bm, Cm = torch.split(F.silu(conv), [mx.d_inner, G * N, G * N], dim=-1)
        dt_raw = proj[..., -H:].contiguous()
        timed(lambda: F.softplus(dt_raw.float() + mx.dt_bias.float()), "softplus dt", res)
        dt = F.softplus(dt_raw.float() + mx.dt_bias.float())
        A = -torch.exp(mx.A_log.float())
        masks = T_._ssd_masks(Q, T // Q, dev, torch.float32)
        xr = x.reshape(b, T, H, P).float().contiguous()
        Br = Bm.reshape(b, T, G, N).float().contiguous()
        Cr = Cm.reshape(b, T, G, N).float().contiguous()
        Dm = mx.D.float()
        timed(lambda: T_.ssd_chunked(xr, dt, A, Br, Cr, Dm, Q, masks),
              "SSD scan, EAGER", res)
        if T_._HAS_NKI:
            timed(lambda: T_._SSDNKI.apply(xr, dt, A, Br, Cr, Dm, Q),
                  "SSD scan, NKI kernel", res)
        y = T_.ssd_chunked(xr, dt, A, Br, Cr, Dm, Q, masks)
        z = proj[..., :mx.d_inner].contiguous()

        def gated_norm():
            h = y.reshape(b, T, mx.d_inner) * F.silu(z.float())
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
            return h * mx.norm_weight.float()

        timed(gated_norm, "gated RMSNorm (y*silu(z), rsqrt)", res)
        hn = gated_norm()
        timed(lambda: mx.out_proj(hn.to(u.dtype)), "out_proj [b,T,d_inner]->[b,T,E]", res)

    # --- what a whole step costs, and what is left to win
    print("\n== budget ==")
    per_step_mixers = 8 * 5          # grad_accum x Mamba2 layers in MMAMMM
    print(f"  measured mixer fwd+bwd            {fb_ms:9.3f} ms")
    print(f"  x {per_step_mixers} mixer calls per optimizer step  "
          f"{fb_ms * per_step_mixers / 1e3:9.3f} s of the 1.96 s eager step")
    print(f"  -> mixers are {fb_ms * per_step_mixers / 1960.0 * 100:.1f}% of the step; "
          f"the rest is MLPs, the attention layer, embedding/lm_head and the optimizer")
    print("\n  component shares of the mixer FORWARD:")
    tot = sum(ms for lb, ms in res if lb not in
              ("mixer forward (no_grad)", "mixer forward+backward", "SSD scan, EAGER"))
    for lb, ms in res:
        if lb in ("mixer forward (no_grad)", "mixer forward+backward"):
            continue
        print(f"    {lb:36s} {ms:8.3f} ms  {ms / max(tot, 1e-9) * 100:5.1f}%")


if __name__ == "__main__":
    sys.exit(main())
