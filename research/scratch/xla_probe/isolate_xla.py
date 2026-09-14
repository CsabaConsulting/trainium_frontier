"""Bisect which starter-kit op breaks the neuronx-cc XLA (PJRT NEURON) path.

One stage per process (a failed compile poisons the XLA tensor state, so stages
cannot share an interpreter):

    for s in $(python isolate_xla.py --list); do
        PJRT_DEVICE=NEURON python isolate_xla.py --stage $s
    done
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch
import torch.nn.functional as F

B, T, C, H = 1, 1024, 128, 1
HD = C // H
V = 8192
BF = torch.bfloat16


def _dev():
    import torch_xla.core.xla_model as xm

    return xm.xla_device()


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


# --- stages ------------------------------------------------------------------

def t01_matmul(d):
    a = torch.randn(512, 512, device=d, dtype=BF)
    b = torch.randn(512, 512, device=d, dtype=BF)
    return a @ b


def t02_randn_only(d):
    return torch.randn(B, T, C, device=d)


def t03_arange_outer_cos(d):
    freqs = torch.arange(0, HD, 2, dtype=torch.float32, device=d)
    inv = 1.0 / (100000 ** (freqs / HD))
    tt = torch.arange(T, dtype=torch.float32, device=d)
    tab = torch.outer(tt, inv)
    return tab.cos().to(BF), tab.sin().to(BF)


def t04_cos_from_cpu(d):
    freqs = torch.arange(0, HD, 2, dtype=torch.float32)
    inv = 1.0 / (100000 ** (freqs / HD))
    tab = torch.outer(torch.arange(T, dtype=torch.float32), inv)
    return tab.cos().to(BF).to(d), tab.sin().to(BF).to(d)


def t05_randint_int32(d):
    return torch.randint(0, V, (B, T), dtype=torch.int32, device=d)


def t06_int32_to_long(d):
    idx = torch.randint(0, V, (B, T), dtype=torch.int32, device=d)
    return idx.long() + 1


def t07_embedding_int32_idx(d):
    idx = torch.randint(0, V, (B, T), dtype=torch.int32).to(d)
    wte = torch.randn(V, C).to(d)
    return F.embedding(idx.long(), wte)


def t08_embedding_long_idx(d):
    idx = torch.randint(0, V, (B, T), dtype=torch.long).to(d)
    wte = torch.randn(V, C).to(d)
    return F.embedding(idx, wte)


def t09_index_select(d):
    idx = torch.randint(0, V, (B * T,), dtype=torch.int32).to(d)
    wte = torch.randn(V, C).to(d)
    return torch.index_select(wte, 0, idx)


def t10_rmsnorm(d):
    x = torch.randn(B, T, C, device=d, dtype=BF)
    return norm(x)


def t11_sdpa_causal(d):
    q = torch.randn(B, H, T, HD, device=d, dtype=BF)
    k = torch.randn(B, H, T, HD, device=d, dtype=BF)
    v = torch.randn(B, H, T, HD, device=d, dtype=BF)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def t12_manual_causal(d):
    q = torch.randn(B, H, T, HD, device=d, dtype=BF)
    k = torch.randn(B, H, T, HD, device=d, dtype=BF)
    v = torch.randn(B, H, T, HD, device=d, dtype=BF)
    m = torch.full((T, T), float("-inf"), device=d, dtype=BF).triu(1)
    att = (q @ k.transpose(-2, -1)) * (HD ** -0.5) + m
    return att.softmax(dim=-1) @ v


def t13_relu_sq(d):
    x = torch.randn(B, T, C, device=d, dtype=BF)
    return F.relu(x).square()


def t14_tanh_softcap(d):
    x = torch.randn(B, T, V, device=d, dtype=torch.float32)
    return 15.0 * torch.tanh(x / 15.0)


def t15_cross_entropy(d):
    logits = torch.randn(B * T, V, device=d, dtype=torch.float32)
    tgt = torch.randint(0, V, (B * T,), dtype=torch.long).to(d)
    return F.cross_entropy(logits, tgt, ignore_index=-1)


def t16_backward(d):
    w = torch.randn(C, C, device=d, dtype=BF).requires_grad_(True)
    x = torch.randn(B, T, C, device=d, dtype=BF)
    y = F.linear(x, w).sum()
    y.backward()
    return w.grad


def t17_muon_polar(d):
    g = torch.randn(2, 512, 128, device=d, dtype=BF)
    x = g / (g.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in [(8.15, -22.48, 15.87), (4.04, -2.80, 0.50)]:
        xtx = x.mT @ x
        x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    return x


def t18_bool_mask_update(d):
    g = torch.randn(2, 512, 128, device=d, dtype=BF)
    w = torch.randn(2, 512, 128, device=d, dtype=BF)
    mask = (g * w) >= 0
    w.sub_(0.01 * g + 0.001 * w * mask)
    return w


def t19_foreach_copy(d):
    params = [torch.randn(512, 128, device=d, dtype=BF) for _ in range(2)]
    stacked = torch.randn(2, 512, 128, device=d, dtype=BF)
    torch._foreach_copy_(params, list(stacked.unbind(0)))
    return params[0]


def t20_lerp_(d):
    a = torch.randn(2, 512, 128, device=d, dtype=BF)
    b = torch.randn(2, 512, 128, device=d, dtype=BF)
    a.lerp_(b, 0.05)
    return a


STAGES = {name: fn for name, fn in sorted(globals().items()) if name.startswith("t") and name[1:3].isdigit()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, default="")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print(" ".join(STAGES))
        return
    fn = STAGES[args.stage]
    import torch_xla.core.xla_model as xm

    d = _dev()
    try:
        out = fn(d)
        xm.mark_step()
        xm.wait_device_ops()
        outs = out if isinstance(out, tuple) else (out,)
        vals = [float(o.float().sum().cpu()) for o in outs if isinstance(o, torch.Tensor)]
        print(f"OK    {args.stage}  {vals}", flush=True)
    except Exception as exc:  # noqa: BLE001
        msg = " | ".join(l.strip() for l in str(exc).strip().splitlines() if l.strip())[:400]
        print(f"FAIL  {args.stage}  {type(exc).__name__}: {msg}", flush=True)
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
