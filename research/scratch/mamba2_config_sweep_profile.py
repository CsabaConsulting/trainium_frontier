"""Per-config, per-component cost of the Mamba2 mixer on real device (eager).

Lever 1 of the "close the Mamba2 speed gap" investigation. Measured context, all
verified against research/experiment-log.jsonl:

    xf-fixed300            300 steps  272.1 s train  ->  0.907 s/step  1.269458 bpb  55.7M
    m2hybrid-nki-fixed300  300 steps  585.1 s train  ->  1.950 s/step  1.270639 bpb  45.0M

Dead heat on quality-per-step, 2.15x slower per step. The mixer config
(expand=2 / d_state=128 / head_dim=128 / n_groups=1 / conv_kernel=4) is the
Mamba2 paper's default and was never tuned for this budget. This script measures
what each cheaper config actually costs per mixer call (fwd and fwd+bwd) at the
real per-rank shape, so the 300-step A/Bs are spent only on configs that could
plausibly close a 1.04 s/step gap.

The projection is deliberately explicit: 5 Mamba2 layers x GRAD_ACCUM micro
steps = N mixer calls per optimizer step, so
    predicted step = (xf step) + N * (mixer fwd+bwd) - (attention cost it replaces)
is not computed here (too many unknowns); instead the script reports the RATIO of
each config's mixer fwd+bwd against the baseline config's, which is the only
number needed to rank candidates.

Run on the device box:  python research/scratch/mamba2_config_sweep_profile.py
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

REPS = 8

# (label, expand, d_state, head_dim, n_groups, conv_kernel)
CONFIGS = [
    ("baseline e2 n128 p128 g1 k4", 2, 128, 128, 1, 4),
    ("e1        n128 p128 g1 k4", 1, 128, 128, 1, 4),
    ("e2         n64 p128 g1 k4", 2, 64, 128, 1, 4),
    ("e1         n64 p128 g1 k4", 1, 64, 128, 1, 4),
    ("e2         n32 p128 g1 k4", 2, 32, 128, 1, 4),
    ("e1         n64 p128 g1 k2", 1, 64, 128, 1, 2),
    ("e1        n128  p64 g1 k4", 1, 128, 64, 1, 4),
]


def sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def timed(fn, reps=REPS):
    fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t0) * 1e3 / reps


def measure(label, expand, d_state, head_dim, n_groups, conv_kernel, b, T, cfg, dev):
    T_.MAMBA2_EXPAND = expand
    T_.MAMBA2_D_STATE = d_state
    T_.MAMBA2_HEAD_DIM = head_dim
    T_.MAMBA2_N_GROUPS = n_groups
    T_.MAMBA2_CONV_KERNEL = conv_kernel
    mx = T_.Mamba2Mixer(cfg).to(dev)
    H, P, N, G, Q = mx.n_heads, mx.head_dim, mx.d_state, mx.n_groups, mx.chunk
    n_par = sum(p.numel() for p in mx.parameters())
    u = torch.randn(b, T, cfg.n_embd, device=dev)
    gsrc = torch.randn(b, T, cfg.n_embd, device=dev)

    with torch.no_grad():
        f_ms = timed(lambda: mx(u))

    def fwd_bwd():
        mx.zero_grad(set_to_none=True)
        ui = u.detach().clone().requires_grad_(True)
        (mx(ui) * gsrc).sum().backward()

    fb_ms = timed(fwd_bwd)

    # component breakdown, forward only
    comps = {}
    with torch.no_grad():
        proj = mx.in_proj(u)
        xBC = proj[..., mx.d_inner:mx.d_inner + mx.d_conv].contiguous()
        cw = mx.conv_weight.to(xBC.dtype)
        comps["in_proj"] = timed(lambda: mx.in_proj(u))
        comps["conv1d"] = timed(lambda: T_.ssd_causal_conv1d(xBC, cw))
        conv = T_.ssd_causal_conv1d(xBC, cw)
        comps["silu"] = timed(lambda: F.silu(conv))
        x, Bm, Cm = torch.split(F.silu(conv), [mx.d_inner, G * N, G * N], dim=-1)
        dt_raw = proj[..., -H:].contiguous()
        dt = F.softplus(dt_raw.float() + mx.dt_bias.float())
        comps["softplus_dt"] = timed(lambda: F.softplus(dt_raw.float() + mx.dt_bias.float()))
        A = -torch.exp(mx.A_log.float())
        xr = x.reshape(b, T, H, P).float().contiguous()
        Br = Bm.reshape(b, T, G, N).float().contiguous()
        Cr = Cm.reshape(b, T, G, N).float().contiguous()
        Dm = mx.D.float()
        if T_._HAS_NKI:
            comps["scan_nki"] = timed(lambda: T_._SSDNKI.apply(xr, dt, A, Br, Cr, Dm, Q))
        masks = T_._ssd_masks(Q, T // Q, dev, torch.float32)
        comps["scan_eager"] = timed(lambda: T_.ssd_chunked(xr, dt, A, Br, Cr, Dm, Q, masks))
        y = T_.ssd_chunked(xr, dt, A, Br, Cr, Dm, Q, masks)
        z = proj[..., :mx.d_inner].contiguous()

        def gated_norm():
            h = y.reshape(b, T, mx.d_inner) * F.silu(z.float())
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
            return h * mx.norm_weight.float()

        comps["gated_rmsnorm"] = timed(gated_norm)
        hn = gated_norm()
        comps["out_proj"] = timed(lambda: mx.out_proj(hn.to(u.dtype)))

    print(f"\n--- {label}")
    print(f"    d_inner={mx.d_inner} H={H} P={P} N={N} G={G} d_conv={mx.d_conv} "
          f"k={mx.conv_kernel} Q={Q}  params={n_par / 1e6:.3f}M")
    print(f"    mixer fwd          {f_ms:8.3f} ms")
    print(f"    mixer fwd+bwd      {fb_ms:8.3f} ms")
    for k, v in comps.items():
        print(f"      {k:16s} {v:8.3f} ms")
    return dict(label=label, fwd=f_ms, fwd_bwd=fb_ms, params=n_par, comps=comps,
                geom=(mx.d_inner, H, P, N, G, mx.d_conv))


def main():
    dev = torch.device("neuron")
    T_.USE_MAMBA2 = True
    T_.USE_NKI_SSD = T_._HAS_NKI
    cfg = T_.build_config()
    b, T = 2, T_.SEQ_LEN
    print(f"locked shape: n_embd={cfg.n_embd} n_layer={cfg.n_layer} SEQ_LEN={T}  "
          f"per-rank micro batch b={b}")
    print(f"nki_ssd={T_.USE_NKI_SSD} has_nki={T_._HAS_NKI}  REPS={REPS}")

    rows = []
    for c in CONFIGS:
        try:
            rows.append(measure(c[0], *c[1:], b, T, cfg, dev))
        except Exception as e:  # a geometry the kernel or the asserts reject
            print(f"\n--- {c[0]}\n    SKIPPED: {type(e).__name__}: {e}")

    base = rows[0]
    # The real submission shape: 4 ranks, DEVICE_BATCH_SIZE*PACK_FACTOR=2 rows of
    # SEQ_LEN per rank -> world_tokens_per_micro=16384, grad_accum=8.
    grad_accum = T_.TOTAL_BATCH_SIZE // (T_.DEVICE_BATCH_SIZE * T_.PACK_FACTOR * T_.SEQ_LEN * 4)
    n_mixer_layers = sum(T_.mamba2_layer_flags(cfg.n_layer))
    calls = n_mixer_layers * grad_accum
    print("\n" + "=" * 100)
    print(f"# {n_mixer_layers} Mamba2 layers x grad_accum={grad_accum} = {calls} mixer "
          f"fwd+bwd calls per optimizer step")
    print(f"{'config':30s} {'params':>9s} {'fwd ms':>9s} {'fwd+bwd':>9s} "
          f"{'ratio':>7s} {'mixer s/step':>13s} {'pred s/step':>12s}")
    base_mixer_s = base["fwd_bwd"] * calls / 1e3
    for r in rows:
        ratio = r["fwd_bwd"] / base["fwd_bwd"]
        mixer_s = r["fwd_bwd"] * calls / 1e3
        # 1.950 s/step measured for the baseline config; the non-mixer remainder is
        # assumed unchanged across configs (same MLPs, same attention layer, same
        # embedding/lm_head, same optimizer -- only the mixer geometry moves).
        pred = 1.950 - (base_mixer_s - mixer_s)
        print(f"{r['label']:30s} {r['params'] / 1e6:8.3f}M {r['fwd']:9.3f} "
              f"{r['fwd_bwd']:9.3f} {ratio:7.3f} {mixer_s:12.3f}s {pred:11.3f}s")
    print("\nMeasured targets: mamba baseline 1.950 s/step, transformer 0.907 s/step "
          "(xf-fixed300).")
    print("Caveat: this is a single-mixer microbenchmark on ONE core with no DDP "
          "all-reduce and no\ncontention from 3 sibling ranks. Treat 'pred s/step' as a "
          "RANKING signal, not a forecast.")


if __name__ == "__main__":
    sys.exit(main())
