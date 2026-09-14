"""Stage-by-stage numeric comparison of the starter kit's Muon step across
backends (cpu reference / native-eager neuron / torch_xla), on identical inputs.

    python muon_numerics.py --device cpu
    NEURON_LOGICAL_NC_CONFIG=2 python muon_numerics.py --device neuron
    PJRT_DEVICE=NEURON NEURON_LOGICAL_NC_CONFIG=2 python muon_numerics.py --device xla

Prints one line per intermediate so the first diverging stage is obvious.
"""

from __future__ import annotations

import argparse

import torch

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def run_adamw(dev, flush, show, dtype: torch.dtype) -> None:
    """Starter kit _adamw_step for the embedding group (betas 0.8/0.995, eps 1e-10).

    The gradient deliberately spans the dynamic range a vocab-8192 embedding sees:
    exact zeros for unseen rows plus magnitudes from 1e-3 down to 1e-20, which is
    where grad.square() underflows in bf16.
    """
    torch.manual_seed(99)
    rows, cols = 8192, 640
    p_cpu = (torch.randn(rows, cols, dtype=torch.float32) * 0.87).to(dtype)
    g_cpu = torch.randn(rows, cols, dtype=torch.float32)
    decades = torch.tensor([0.0, 1e-3, 1e-5, 1e-8, 1e-12, 1e-20], dtype=torch.float32)
    g_cpu = g_cpu * decades[torch.randint(0, len(decades), (rows, 1))]
    g_cpu = g_cpu.to(dtype)

    p = p_cpu.to(dev)
    grad = g_cpu.to(dev)
    exp_avg = torch.zeros_like(p)
    exp_avg_sq = torch.zeros_like(p)
    beta1, beta2, eps, lr, wd = 0.8, 0.995, 1e-10, 0.3, 0.001

    show(f"adamw[{dtype}].p_in", p)
    show(f"adamw[{dtype}].grad", grad)
    p.mul_(1 - lr * wd)
    exp_avg.lerp_(grad, 1 - beta1)
    show(f"adamw[{dtype}].exp_avg", exp_avg)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    show(f"adamw[{dtype}].exp_avg_sq", exp_avg_sq)
    denom = exp_avg_sq.sqrt() + eps
    show(f"adamw[{dtype}].denom", denom)
    update = exp_avg / 1.0 / denom
    show(f"adamw[{dtype}].update", update)
    p.add_(update, alpha=-lr)
    show(f"adamw[{dtype}].p_out", p)


def run_embedding(dev, flush, show, xm, accum_steps: int) -> None:
    """Embedding backward: duplicate-index scatter-add, and accumulation across
    micro-steps (with a mark_step between them on XLA, as the probe does).

    grad = sum over positions of coef[position], scattered into row idx[position],
    so the exact answer is computable and duplicate handling is observable.
    """
    torch.manual_seed(7)
    rows, cols, tokens = 8192, 640, 512
    w_cpu = torch.randn(rows, cols, dtype=torch.float32) * 0.02
    idx_cpu = [torch.randint(0, rows, (1, tokens), dtype=torch.int32) for _ in range(accum_steps)]
    coef_cpu = [torch.randn(1, tokens, cols, dtype=torch.float32) for _ in range(accum_steps)]

    # exact reference on cpu fp32 with index_add
    ref = torch.zeros(rows, cols, dtype=torch.float32)
    for i, c in zip(idx_cpu, coef_cpu):
        ref.index_add_(0, i.flatten().long(), c.reshape(-1, cols))
    print(f"reference(cpu fp32 index_add) l2 {float(ref.pow(2).sum().sqrt()):.6f} "
          f"sum {float(ref.sum()):+.6f}", flush=True)

    w = torch.nn.Parameter(w_cpu.to(dev))
    for i, c in zip(idx_cpu, coef_cpu):
        out = torch.nn.functional.embedding(i.to(dev).long(), w)
        (out * c.to(dev)).sum().backward()
        if xm is not None:
            xm.mark_step()
    show(f"embed_grad(accum={accum_steps})", w.grad)

    # duplicate-index-only check: every position hits the same row
    w2 = torch.nn.Parameter(w_cpu.to(dev))
    dup = torch.zeros(1, tokens, dtype=torch.int32) + 5
    out = torch.nn.functional.embedding(dup.to(dev).long(), w2)
    (out * coef_cpu[0].to(dev)).sum().backward()
    flush()
    got = float(w2.grad.float().pow(2).sum().sqrt().cpu())
    want = float(coef_cpu[0].reshape(-1, cols).sum(0).pow(2).sum().sqrt())
    print(f"dup_index_scatter_add l2 got {got:.6f} want {want:.6f} "
          f"ratio {got / want:.6f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["muon", "adamw", "embedding"], default="muon")
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--device", choices=["cpu", "neuron", "xla"], default="cpu")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--rows", type=int, default=640)
    ap.add_argument("--cols", type=int, default=640)
    ap.add_argument("--scale", type=float, default=1e-3, help="gradient scale (Muon sees tiny grads)")
    ap.add_argument("--ns-steps", type=int, default=5)
    args = ap.parse_args()

    xm = None
    if args.device == "xla":
        import torch_xla.core.xla_model as _xm

        xm = _xm
        dev = _xm.xla_device()
    else:
        dev = torch.device(args.device)

    def flush():
        if xm is not None:
            xm.mark_step()
            xm.wait_device_ops()

    def show(tag, t):
        flush()
        f = t.float().cpu()
        print(f"{tag:24s} sum {float(f.sum()):+.6e}  absmax {float(f.abs().max()):.6e}  "
              f"l2 {float(f.pow(2).sum().sqrt()):.6e}", flush=True)

    if args.op == "embedding":
        run_embedding(dev, flush, show, xm, args.accum_steps)
        return

    if args.op == "adamw":
        for dt in (torch.bfloat16, torch.float32):
            run_adamw(dev, flush, show, dt)
        return

    torch.manual_seed(1234)
    n, r, c = args.n, args.rows, args.cols
    g_cpu = (torch.randn(n, r, c, dtype=torch.float32) * args.scale).bfloat16()
    w_cpu = (torch.randn(n, r, c, dtype=torch.float32) * 0.02).bfloat16()

    g = g_cpu.to(dev)
    w = w_cpu.to(dev)
    momentum_buffer = torch.zeros(n, r, c, dtype=torch.bfloat16, device=dev)
    second = torch.zeros(n, r, 1, dtype=torch.float32, device=dev)
    momentum, beta2, lr, wd = 0.95, 0.90, 0.025, 0.10

    show("in.g", g)
    show("in.w", w)

    momentum_buffer.lerp_(g, 1 - momentum)
    show("momentum_buffer", momentum_buffer)
    g = g.lerp_(momentum_buffer, momentum)
    show("g_after_lerp", g)

    x = g.bfloat16()
    nrm = x.norm(dim=(-2, -1), keepdim=True)
    show("x.norm", nrm)
    x = x / (nrm * 1.01 + 1e-6)
    show("x_normalized", x)

    if x.shape[-2] > x.shape[-1]:
        for i, (a, b, cc) in enumerate(POLAR_EXPRESS_COEFFS[: args.ns_steps]):
            xtx = x.mT @ x
            x = a * x + x @ (b * xtx + cc * (xtx @ xtx))
            show(f"ns{i}", x)
    else:
        for i, (a, b, cc) in enumerate(POLAR_EXPRESS_COEFFS[: args.ns_steps]):
            xxt = x @ x.mT
            x = a * x + (b * xxt + cc * (xxt @ xxt)) @ x
            show(f"ns{i}", x)
    g = x

    red_dim = -1 if r >= c else -2
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    show("v_mean", v_mean)
    red_dim_size = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    show("v_norm", v_norm)
    second.lerp_(v_mean, 1 - beta2)
    show("second", second)
    step_size = second.clamp_min(1e-10).rsqrt()
    show("step_size", step_size)
    scaled = (v_mean * red_dim_size) * step_size.square()
    v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
    show("v_norm_new", v_norm_new)
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
    show("g_final", g)

    mask = (g * w) >= 0
    show("mask", mask)
    w.sub_(lr * g + lr * wd * w * mask)
    show("w_out", w)


if __name__ == "__main__":
    main()
