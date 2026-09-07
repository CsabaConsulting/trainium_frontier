"""Is the fused softcap+CE lse error from the Activation engine's exp, or from the
Scalar-engine reduce_regs accumulator? Compare three ways of getting d = sum_j exp(...):
  d_fused : nisa.activation(..., reduce_op=nl.add, reduce_res=d)   (free, 0 extra passes)
  d_vec   : nisa.tensor_reduce(nl.add) on the Vector engine over the same exp output
  e_out   : the raw exp tensor, so exp's own elementwise accuracy is measurable
"""
import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
S = 15.0


@nki.jit
def probe(t_in):
    """t_in is a *given* tanh value tile, so tanh error is excluded by construction."""
    N, V = t_in.shape
    e_out = nl.ndarray((N, V), dtype=nl.float32, buffer=nl.shared_hbm)
    d_fused = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    d_vec = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    negS = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=negS, value=-S)
    for i in nl.affine_range(N // P):
        rows = nl.ds(i * P, P)
        tb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tb, src=t_in[rows])
        eb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        df = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=eb, op=nl.exp, data=tb, scale=S, bias=negS,
                        reduce_op=nl.add, reduce_res=df,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        dv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=dv, op=nl.add, data=eb, axis=(1,))
        nisa.dma_copy(dst=e_out[rows], src=eb)
        nisa.dma_copy(dst=d_fused[rows], src=df)
        nisa.dma_copy(dst=d_vec[rows], src=dv)
    return e_out, d_fused, d_vec


def diag(N, V, scale):
    torch.manual_seed(0)
    t = torch.tanh((torch.randn(N, V) * scale) / S).float()
    e_hw, df, dv = probe(t.to("neuron"))
    e_hw, df, dv = e_hw.cpu(), df.cpu().reshape(-1), dv.cpu().reshape(-1)

    arg = (S * t - S)                                   # exact fp32 exp argument
    e_ref64 = torch.exp(arg.double())
    e_ref32 = torch.exp(arg)
    rel_hw = ((e_hw.double() - e_ref64).abs() / e_ref64).max()
    rel_torch = ((e_ref32.double() - e_ref64).abs() / e_ref64).max()

    d_ref64 = e_ref64.sum(dim=1)                        # exact sum of exact exps
    d_of_hw_e = e_hw.double().sum(dim=1)                # exact sum of the hw exps
    print(f"  N={N} V={V} scale={scale}")
    print(f"    exp elementwise rel err: hw={rel_hw:.3e}  torch={rel_torch:.3e}")
    print(f"    log(d) abs err vs fp64-of-exact-exp:  fused={(torch.log(df.double()) - torch.log(d_ref64)).abs().max():.3e}"
          f"  vector={(torch.log(dv.double()) - torch.log(d_ref64)).abs().max():.3e}")
    print(f"    log(d) abs err vs fp64-sum-of-the-SAME-hw-exps (isolates accumulator): "
          f"fused={(torch.log(df.double()) - torch.log(d_of_hw_e)).abs().max():.3e}"
          f"  vector={(torch.log(dv.double()) - torch.log(d_of_hw_e)).abs().max():.3e}")
    print(f"    exp-only contribution to log(d): "
          f"{(torch.log(d_of_hw_e) - torch.log(d_ref64)).abs().max():.3e}")


for args in ((256, 512, 3.0), (256, 8192, 3.0), (256, 8192, 30.0)):
    diag(*args)
print("DONE")
