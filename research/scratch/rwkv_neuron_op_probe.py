"""Does every op the RWKV-x051a port needs actually work on the Neuron device?

Run this BEFORE spending real box time on a training run. Each op is exercised at
(or near) the real per-layer shape in eager mode on the Neuron device, and its
result is compared against the identical computation on CPU. An op that silently
returns wrong numbers is the failure mode that matters here, so every probe is a
numeric comparison, not just "it did not throw".

Also probes, for the record, the three ops the port deliberately AVOIDS
(nn.GroupNorm, torch.tile, Tensor.flip) so the decision to rewrite them is
recorded as measured rather than assumed.

Usage (on the box, inside the venv):
    NEURON_RT_VISIBLE_CORES=0 python research/scratch/rwkv_neuron_op_probe.py
    python research/scratch/rwkv_neuron_op_probe.py --cpu-only   # sanity-check the harness
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
os.environ.setdefault("NEURON_COMPILE_CACHE_DIR", "/tmp/neuron_cache")

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOT the real locked shape (B=2, T=2048, C=640, H=5, N=128, Q=128): every probe
# here is a separate NEFF compile on the native-eager backend, and at the real shape
# ~20 of them is a long wall-clock bill for a question that is about the OP, not the
# size. The real shape is exercised end-to-end by the train.py smoke test instead;
# this probe answers "does the op exist and return the right numbers at all", so it
# runs at 1/8 the sequence and 1/5 the width, keeping the same 5-D chunk structure
# (NC>1, so the inter-chunk path is live) and the same head_dim=32-style geometry.
B, T, C, H, N, Q = 1, 256, 128, 4, 32, 32
NC = T // Q

RESULTS: list[tuple[str, str, str]] = []   # (name, verdict, detail)


def probe(name: str, fn, dev: torch.device, tol: float = 2e-2, *shapes):
    """Run fn on CPU and on dev with identical inputs; compare."""
    gen = torch.Generator().manual_seed(hash(name) % (2**31))
    args_cpu = [torch.randn(*s, generator=gen, dtype=torch.float32) for s in shapes]
    try:
        out_cpu = fn(*args_cpu)
    except Exception as e:
        RESULTS.append((name, "CPU-ERROR", f"{type(e).__name__}: {e}"))
        print(f"[CPU-ERROR] {name}: {type(e).__name__}: {e}")
        return
    if dev.type == "cpu":
        RESULTS.append((name, "CPU-ONLY", "no device to compare against"))
        print(f"[CPU-ONLY] {name}  out={tuple(out_cpu.shape)}", flush=True)
        return
    t0 = time.monotonic()
    try:
        args_dev = [a.to(dev) for a in args_cpu]
        out_dev = fn(*args_dev)
        got = out_dev.float().cpu()
    except Exception as e:
        RESULTS.append((name, "DEVICE-ERROR", f"{type(e).__name__}: {e}"))
        print(f"[DEVICE-ERROR] {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(limit=3)
        return
    dt = time.monotonic() - t0
    if not torch.isfinite(got).all():
        RESULTS.append((name, "NONFINITE", "device output has inf/nan"))
        print(f"[NONFINITE] {name}", flush=True)
        return
    denom = out_cpu.abs().max().clamp(min=1e-6)
    rel = ((got - out_cpu).abs().max() / denom).item()
    ok = rel < tol
    RESULTS.append((name, "OK" if ok else "MISMATCH", f"max_rel={rel:.3e} {dt:.1f}s"))
    print(f"[{'OK' if ok else 'MISMATCH'}] {name:38s} out={tuple(out_cpu.shape)} "
          f"max_rel={rel:.3e} first_call={dt:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cpu")
    if not args.cpu_only:
        try:
            # NO `import torch_neuronx`: this repo runs the NATIVE torch-neuron eager
            # backend, which registers the 'neuron' device on plain `import torch`
            # (train.py never imports torch_neuronx either). Importing the XLA-flavoured
            # package instead is what hung the first attempt at this probe for an hour
            # with zero CPU burned and no output.
            dev = torch.device("neuron")
            # touch the device once so a failure to initialise surfaces here, not
            # buried inside the first probe
            _ = torch.ones(4, device=dev).cpu()
            print(f"device = {dev} (torch {torch.__version__})", flush=True)
        except Exception as e:
            print(f"!! could not open the neuron device: {type(e).__name__}: {e}")
            print("!! falling back to CPU-only -- this run proves NOTHING about Neuron")
            dev = torch.device("cpu")
    else:
        print("cpu-only harness sanity check")

    print("\n-- ops the port actually uses --", flush=True)
    # 1. the time_shift pattern: slice + left F.pad on the T axis
    probe("F.pad time_shift [B,T,C]", lambda x: F.pad(x[:, :-1, :], (0, 0, 1, 0)),
          dev, 2e-2, (B, T, C))
    probe("time_shift - x (the maa input)",
          lambda x: F.pad(x[:, :-1, :], (0, 0, 1, 0)) - x, dev, 2e-2, (B, T, C))
    # 2. exp of a scaled exponent grid + additive mask (the decay matrices)
    qi = torch.arange(Q)
    e_intra_cpu = (qi[:, None] - qi[None, :] - 1).clamp(min=0).float()
    m_intra_cpu = torch.where(qi[:, None] > qi[None, :], 0.0, -30000.0)

    def decay(lw):
        e = e_intra_cpu.to(lw.device)
        m = m_intra_cpu.to(lw.device)
        return torch.exp(-torch.exp(lw).reshape(H, 1, 1) * e + m)

    probe("exp(-exp(p)*grid + mask) [H,Q,Q]", decay, dev, 2e-2, (H, 1))
    probe("torch.exp on [H,1]", lambda p: torch.exp(-torch.exp(p)), dev, 2e-2, (H, 1))
    probe(".pow() on a [1,1,C] ratio",
          lambda p: 1.0 - torch.pow(p.sigmoid(), 0.7), dev, 2e-2, (1, 1, C))
    # 3. the 5-D batched matmuls the chunked operator is built from
    probe("5D matmul [b,NC,H,Q,N]@[b,NC,H,N,Q]",
          lambda a, b_: torch.matmul(a, b_.transpose(-1, -2)),
          dev, 5e-2, (B, NC, H, Q, N), (B, NC, H, Q, N))
    probe("5D matmul [b,NC,H,Q,Q]@[b,NC,H,Q,N]",
          lambda a, b_: torch.matmul(a, b_), dev, 5e-2,
          (B, NC, H, Q, Q), (B, NC, H, Q, N))
    probe("5D matmul state build [b,NC,H,N,Q]@[b,NC,H,Q,N]",
          lambda a, b_: torch.matmul(a.transpose(-1, -2), b_), dev, 5e-2,
          (B, NC, H, Q, N), (B, NC, H, Q, N))
    probe("chunk-decay matmul [1,H,NC,NC]@[b,H,NC,N*N]",
          lambda a, b_: torch.matmul(a, b_), dev, 5e-2,
          (1, H, NC, NC), (B, H, NC, N * N))
    # 4. the reshape/permute plumbing (5-D permute is the part most likely to be slow)
    probe("reshape+permute [b,T,H,N]->[b,NC,H,Q,N]",
          lambda x: x.reshape(B, NC, Q, H, N).permute(0, 1, 3, 2, 4).contiguous(),
          dev, 2e-2, (B, T, H, N))
    probe("permute+reshape [b,NC,H,N,N]->[b,H,NC,N*N]",
          lambda x: x.permute(0, 2, 1, 3, 4).reshape(B, H, NC, N * N).contiguous(),
          dev, 2e-2, (B, NC, H, N, N))
    # 5. the pointwise/reduce tail
    probe("F.silu [B,T,C]", F.silu, dev, 2e-2, (B, T, C))
    probe("torch.sigmoid [B,T,C]", torch.sigmoid, dev, 2e-2, (B, T, C))
    probe("relu(x)**2 [B,T,3C]", lambda x: F.relu(x).pow(2), dev, 2e-2, (B, T, 3 * C))
    probe("mean/var/rsqrt over last dim [B,T,H,N]",
          lambda y: ((y - y.mean(-1, keepdim=True))
                     * torch.rsqrt((y - y.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True) + 6.4e-4)),
          dev, 5e-2, (B, T, H, N))
    probe("eye/arange/clamp built ON device",
          lambda x: (torch.eye(Q, device=x.device)
                     + (torch.arange(Q, device=x.device)[:, None]
                        - torch.arange(Q, device=x.device)[None, :] - 1).clamp(min=0).float()) * x,
          dev, 2e-2, (1, 1))

    print("\n-- ops the port deliberately avoids (recorded, not required) --", flush=True)
    gn = nn.GroupNorm(H, C, eps=6.4e-4)
    probe("nn.GroupNorm(H, C)  [AVOIDED]",
          lambda x: gn.to(x.device)(x), dev, 5e-2, (B * 8, C))
    probe("torch.tile  [AVOIDED]", lambda x: torch.tile(x, [Q]), dev, 2e-2, (H, 2 * Q))
    probe("Tensor.flip  [AVOIDED]", lambda x: x.flip(2), dev, 2e-2, (1, H, Q, 1))
    probe("F.conv1d grouped  [AVOIDED, for reference]",
          lambda x, w: F.conv1d(x, w.reshape(C, 1, 4), groups=C), dev, 5e-2,
          (B, C, 256), (C, 4))

    print("\n===== summary =====")
    bad = [r for r in RESULTS if r[1] not in ("OK", "CPU-ONLY")]
    required_bad = [r for r in bad if "AVOIDED" not in r[0]]
    for name, verdict, detail in RESULTS:
        print(f"{verdict:14s} {name:44s} {detail}")
    print(f"\n{len(RESULTS)} probes, {len(bad)} not-OK, "
          f"{len(required_bad)} of them on ops the port REQUIRES")
    if required_bad:
        print("REQUIRED OPS FAILED: " + ", ".join(n for n, _, _ in required_bad))
    sys.exit(1 if required_bad else 0)


if __name__ == "__main__":
    main()
