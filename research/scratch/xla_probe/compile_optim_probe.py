"""T1.7 re-check: can torch.compile wrap the Muon optimizer step once
torch._foreach_* is out of the way?

Two backends, same pure-tensor function:
  --backend native  torch.device("neuron"), torch.compile(backend="neuron")
  --backend xla     xm.xla_device(),         torch.compile(backend="openxla")

The function is the starter kit's `_muon_step` body reduced to tensors only
(no optimizer-state dict, no Python-object mutation), which is the shape a
compiled optimizer step would have to take anyway.
"""

from __future__ import annotations

import argparse
import time
import traceback

import torch

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def muon_step(w, g, momentum_buffer, second, momentum, lr, wd):
    """Starter-kit _muon_step, tensors in / tensors out. Shapes [n, m, k]."""
    momentum_buffer = momentum_buffer.lerp(g, 1 - momentum)
    g = g.lerp(momentum_buffer, momentum)
    x = g
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if x.shape[-2] > x.shape[-1]:
        for a, b, c in POLAR_EXPRESS_COEFFS[:5]:
            xtx = x.mT @ x
            x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:5]:
            xxt = x @ x.mT
            x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    g = x
    red_dim = -1 if w.shape[-2] >= w.shape[-1] else -2
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second = second.lerp(v_mean, 1 - 0.90)
    step_size = second.clamp_min(1e-10).rsqrt()
    scaled = (v_mean * red_dim_size) * step_size.square()
    v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
    mask = (g * w) >= 0
    w = w - (lr * g + lr * wd * w * mask)
    return w, momentum_buffer, second


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["native", "xla"], default="native")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--n", type=int, default=6, help="params stacked per shape group")
    ap.add_argument("--dim", type=int, default=640)
    args = ap.parse_args()

    xm = None
    if args.backend == "xla":
        import torch_xla.core.xla_model as _xm

        xm = _xm
        dev = xm.xla_device()
        compile_backend = "openxla"
    else:
        dev = torch.device("neuron")
        compile_backend = "neuron"

    def sync():
        if xm is not None:
            xm.mark_step()
            xm.wait_device_ops()
        elif hasattr(torch, "neuron"):
            torch.neuron.synchronize()

    n, d = args.n, args.dim
    w = torch.randn(n, d, d, device=dev, dtype=torch.bfloat16)
    g = torch.randn(n, d, d, device=dev, dtype=torch.bfloat16)
    mb = torch.zeros(n, d, d, device=dev, dtype=torch.bfloat16)
    second = torch.zeros(n, d, 1, device=dev, dtype=torch.float32)
    sync()

    def timeit(fn, label):
        try:
            t0 = time.monotonic()
            out = fn(w, g, mb, second, 0.95, 0.025, 0.10)
            sync()
            first = time.monotonic() - t0
            t0 = time.monotonic()
            for _ in range(args.iters):
                out = fn(w, g, mb, second, 0.95, 0.025, 0.10)
            sync()
            per = (time.monotonic() - t0) / args.iters
            chk = float(out[0].float().sum().cpu())
            print(f"OK   {label:12s} first={first:.2f}s per_iter={per * 1e3:.2f}ms checksum={chk:.4f}", flush=True)
            return per
        except Exception as exc:  # noqa: BLE001
            msg = " | ".join(l.strip() for l in str(exc).strip().splitlines() if l.strip())[:500]
            print(f"FAIL {label:12s} {type(exc).__name__}: {msg}", flush=True)
            traceback.print_exc()
            return None

    print(f"backend={args.backend} device={dev} compile_backend={compile_backend} "
          f"shape=[{n},{d},{d}]", flush=True)
    eager = timeit(muon_step, "eager")
    compiled_fn = torch.compile(muon_step, backend=compile_backend, dynamic=False)
    comp = timeit(compiled_fn, "compiled")
    if eager and comp:
        print(f"RATIO eager/compiled = {eager / comp:.3f}x", flush=True)

    # Also: does _foreach_copy_ still break torch.compile on this backend?
    def with_foreach(w_, g_, mb_, second_, momentum, lr, wd):
        out = muon_step(w_, g_, mb_, second_, momentum, lr, wd)
        params = list(torch.zeros_like(w_).unbind(0))
        torch._foreach_copy_(params, list(out[0].unbind(0)))
        return (torch.stack(params),) + out[1:]

    timeit(torch.compile(with_foreach, backend=compile_backend, dynamic=False), "compiled+fe")


if __name__ == "__main__":
    main()
