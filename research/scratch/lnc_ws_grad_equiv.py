"""Suspect (b) probe: does the world-size split change what the OPTIMIZER sees?

train.py calls sync_gradients() (all_reduce AVG over every parameter) BEFORE
optimizer.step(), so Muon's Newton-Schulz never sees a per-rank batch: it sees the
average over the whole TOTAL_BATCH_SIZE. This reproduces the two grouping paths
exactly and measures the difference.

  ws=4 : grad_accum=4  -> each rank sums 4 micro-grads scaled 1/4, then AVG over 4 ranks
  ws=8 : grad_accum=2  -> each rank sums 2 micro-grads scaled 1/2, then AVG over 8 ranks

Both are (1/16) * sum(g_0..g_15) mathematically. Reported for fp32 (the live-path
gradient dtype for Muon/AdamW params) and bf16 (NGRAM_VE_BF16=True tables).
"""
import torch

torch.manual_seed(1234)
NMICRO = 16


def accum(gs, ws, dtype):
    """Reproduce train.py's loop: per-rank in-place accumulation of loss/grad_accum
    gradients, then dist.all_reduce(AVG) across ranks."""
    ga = NMICRO // ws
    locals_ = []
    for r in range(ws):
        acc = torch.zeros_like(gs[0], dtype=dtype)
        for i in range(r * ga, (r + 1) * ga):
            acc += (gs[i] / ga).to(dtype)          # backward() accumulating loss/grad_accum
        locals_.append(acc)
    tot = torch.zeros_like(locals_[0], dtype=torch.float32)
    for l in locals_:
        tot += l.to(torch.float32)                  # AVG = sum/ws
    return (tot / ws).to(dtype)


for shape, name in (((4480, 640), "muon matrix (c_fc-like)"), ((640,), "scalar/vector"),
                    ((8192, 640), "wte")):
    for dtype in (torch.float32, torch.bfloat16):
        gs = [torch.randn(shape, dtype=torch.float32) * 3e-3 for _ in range(NMICRO)]
        exact = torch.stack(gs).sum(0) / NMICRO
        a = accum(gs, 4, dtype).float()
        b = accum(gs, 8, dtype).float()
        den = exact.abs().mean().item()
        print(f"{name:24s} {str(dtype):16s} "
              f"max|ws4-ws8|={(a - b).abs().max().item():.3e}  "
              f"rel_to_mean|g|={(a - b).abs().max().item() / den:.3e}  "
              f"max|ws4-exact|={(a - exact).abs().max().item():.3e}  "
              f"max|ws8-exact|={(b - exact).abs().max().item():.3e}")

# Now push both through the real Newton-Schulz body and compare the UPDATE.
import importlib.util
import sys
sys.path.insert(0, "/home/ubuntu/trainium_frontier")
import sys
spec = importlib.util.spec_from_file_location("tp", "/home/ubuntu/trainium_frontier/train.py")
tp = importlib.util.module_from_spec(spec)
sys.modules["tp"] = tp
spec.loader.exec_module(tp)

shape = (4480, 640)
gs = [torch.randn(shape, dtype=torch.float32) * 3e-3 for _ in range(NMICRO)]
g4 = accum(gs, 4, torch.float32).unsqueeze(0)
g8 = accum(gs, 8, torch.float32).unsqueeze(0)
print(f"\ninput to Muon: max|g_ws4 - g_ws8| = {(g4 - g8).abs().max().item():.3e}")
outs = []
w0 = torch.randn((1,) + shape) * 0.02   # SAME initial weight for both arms
for g in (g4, g8):
    w = w0
    mb = torch.zeros_like(w)
    sm = torch.zeros_like(w)
    ratio = max(1.0, shape[-2] / shape[-1]) ** 0.5
    red_dim = -1 if shape[-2] >= shape[-1] else -2
    ns = min(5, len(tp.POLAR_EXPRESS_COEFFS))
    w2, _, _ = tp._muon_math(
        w.clone(), g.clone(), mb, sm,
        torch.tensor(0.95), torch.tensor(0.05),
        torch.tensor(0.02 * ratio), torch.tensor(0.02 * 0.0),
        shape[-2] > shape[-1], ns, red_dim, g.size(red_dim), 1 - 0.90, True)
    outs.append(w2)
d = (outs[0] - outs[1]).abs()
print(f"Muon update after NS: max|w_ws4 - w_ws8| = {d.max().item():.3e}  "
      f"mean = {d.mean().item():.3e}  ||w|| = {outs[0].norm().item():.4f}")
