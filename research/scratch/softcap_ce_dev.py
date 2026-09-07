"""Dev scratchpad for the fused logit-softcap + cross-entropy NKI kernel.

Standalone (does NOT import train.py) so the compile/iterate loop is fast. Once the
primitives here are verified, the kernel bodies get pasted into train.py verbatim and
research/scratch/softcap_ce_equiv.py becomes the real harness (importing from train.py).
"""
import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch
import torch.nn.functional as F

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
S = 15.0


@nki.jit
def fwd(z, tgt):
    """(loss, lse) for loss = logsumexp_j(S*tanh(z_j/S)) - S*tanh(z_c/S)."""
    N, V = z.shape
    loss = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    lse = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    idx = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=idx, pattern=[[1, V]], offset=0, channel_multiplier=0)
    negS = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=negS, value=-S)

    for i in nl.affine_range(N // P):
        rows = nl.ds(i * P, P)
        zb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=zb, src=z[rows])
        cb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cb, src=tgt[rows])

        tb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=tb, op=nl.tanh, data=zb, scale=1.0 / S)

        # d = sum_j exp(y_j - S), y = S*tanh(z/S) in (-S, S) so every term is in
        # (exp(-2S), 1]: a constant shift is provably safe, no row-max pass needed.
        d = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=zb, op=nl.exp, data=tb, scale=S, bias=negS,
                        reduce_op=nl.add, reduce_res=d,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)

        mb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mb, data=idx, op0=nl.equal, operand0=cb)
        nisa.tensor_tensor(dst=mb, data1=mb, data2=tb, op=nl.multiply)
        tc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tc, op=nl.add, data=mb, axis=(1,))

        lb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=lb, op=nl.log, data=d)
        nisa.tensor_scalar(dst=lb, data=lb, op0=nl.add, operand0=S)   # lse = S + log d
        ob = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ob, data=tc, op0=nl.multiply, operand0=S)  # y_c
        nisa.tensor_tensor(dst=ob, data1=lb, data2=ob, op=nl.subtract)

        nisa.dma_copy(dst=lse[rows], src=lb)
        nisa.dma_copy(dst=loss[rows], src=ob)
    return loss, lse


@nki.jit
def bwd(z, tgt, lse, g):
    """dz_j = g * (softmax(y)_j - [j==c]) * (1 - tanh(z_j/S)**2)."""
    N, V = z.shape
    dz = nl.ndarray((N, V), dtype=nl.float32, buffer=nl.shared_hbm)

    idx = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=idx, pattern=[[1, V]], offset=0, channel_multiplier=0)

    for i in nl.affine_range(N // P):
        rows = nl.ds(i * P, P)
        zb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=zb, src=z[rows])
        cb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cb, src=tgt[rows])
        gb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gb, src=g[rows])
        nb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=nb, src=lse[rows])
        nisa.tensor_scalar(dst=nb, data=nb, op0=nl.multiply, operand0=-1.0)

        tb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=tb, op=nl.tanh, data=zb, scale=1.0 / S)
        nisa.activation(dst=zb, op=nl.exp, data=tb, scale=S, bias=nb)  # p = softmax(y)

        mb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mb, data=idx, op0=nl.equal, operand0=cb,
                           op1=nl.multiply, operand1=gb)                 # [j==c]*g
        nisa.scalar_tensor_tensor(dst=zb, data=zb, op0=nl.multiply, operand0=gb,
                                  op1=nl.subtract, operand1=mb)          # p*g - [j==c]*g
        nisa.activation(dst=mb, op=nl.square, data=tb)                   # t^2
        nisa.scalar_tensor_tensor(dst=tb, data=mb, op0=nl.subtract, operand0=1.0,
                                  reverse0=True, op1=nl.multiply, operand1=zb)
        nisa.dma_copy(dst=dz[rows], src=tb)
    return dz


def ref(z, tgt):
    y = S * torch.tanh(z / S)
    return F.cross_entropy(y, tgt, reduction="none")


def run(N, V, seed=0, scale=3.0):
    torch.manual_seed(seed)
    z = (torch.randn(N, V) * scale).float()
    tgt = torch.randint(0, V, (N,))
    g = (torch.randn(N) * 0.5).float()

    zr = z.clone().requires_grad_(True)
    lr = ref(zr, tgt)
    lr.backward(g)
    ref_loss, ref_dz = lr.detach(), zr.grad.detach()

    zd = z.to("neuron")
    td = tgt.reshape(-1, 1).to(torch.float32).to("neuron")
    loss_d, lse_d = fwd(zd, td)
    dz_d = bwd(zd, td, lse_d, g.reshape(-1, 1).to("neuron"))

    got_loss = loss_d.cpu().reshape(-1)
    got_dz = dz_d.cpu()

    for tag, got, r in (("fwd", got_loss, ref_loss), ("bwd", got_dz, ref_dz)):
        d = (got - r).abs()
        rel = d.max().item() / max(r.abs().max().item(), 1e-30)
        print(f"  {tag}: max_abs={d.max().item():.3e} rel_to_max={rel:.3e} "
              f"exact={100.0 * (got == r).float().mean().item():.2f}%")


print("=== small shape (256, 512) ===")
run(256, 512)
print("=== real vocab (256, 8192) ===")
run(256, 8192)
print("DONE")
