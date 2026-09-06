"""Probe round 2: rsqrt via activation, fused activation-reduce on 3D, divide, in-place."""
import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch
import nki
import nki.language as nl
import nki.isa as nisa

P, H, D = 128, 3, 128
Dh = D // 2
BC = [[H, P], [1, H], [0, D]]  # [P,H,1] -> [P,H,D] stride-0 broadcast


@nki.jit
def p_rsqrt(x):
    """activation(op=nl.rsqrt) on a [P,H,1] tile."""
    out = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    xb = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:P])
    s = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s, op=nl.add, data=xb, axis=(2,))
    r = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=r, op=nl.rsqrt, data=s)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def p_act_reduce(x):
    """activation(op=nl.square, reduce_op=add, reduce_res=[P,H,1]) fused on 3D."""
    sq_out = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.shared_hbm)
    ss_out = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    xb = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:P])
    sq = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=xb, reduce_op=nl.add, reduce_res=ss)
    nisa.dma_copy(dst=sq_out, src=sq)
    nisa.dma_copy(dst=ss_out, src=ss)
    return sq_out, ss_out


@nki.jit
def p_ts_chain(x):
    """tensor_scalar with op0/op1 chain on a [P,H,1] tile: s*(1/D) + eps."""
    out = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    xb = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:P])
    s = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s, op=nl.add, data=xb, axis=(2,))
    r = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r, data=s, op0=nl.multiply, operand0=1.0 / D,
                       op1=nl.add, operand1=0.5)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def p_divide(x, y):
    """tensor_tensor with op=nl.divide on [P,H,1]."""
    out = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    xb = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    yb = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:P])
    nisa.dma_copy(dst=yb, src=y[0:P])
    a = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
    b = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=a, op=nl.add, data=xb, axis=(2,))
    nisa.tensor_reduce(dst=b, op=nl.add, data=yb, axis=(2,))
    r = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=r, data1=a, data2=b, op=nl.divide)
    nisa.dma_copy(dst=out, src=r)
    return out


@nki.jit
def p_inplace(x):
    """in-place tensor_tensor where dst aliases data1, on a strided 3D slice."""
    out = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.shared_hbm)
    xb = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:P])
    z = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=z[0:P, 0:H, 0:Dh], data1=xb[0:P, 0:H, 0:Dh],
                       data2=xb[0:P, 0:H, Dh:D], op=nl.multiply)
    nisa.tensor_tensor(dst=z[0:P, 0:H, 0:Dh], data1=z[0:P, 0:H, 0:Dh],
                       data2=xb[0:P, 0:H, Dh:D], op=nl.add)
    nisa.tensor_tensor(dst=z[0:P, 0:H, Dh:D], data1=xb[0:P, 0:H, Dh:D],
                       data2=xb[0:P, 0:H, 0:Dh], op=nl.subtract)
    nisa.dma_copy(dst=out, src=z)
    return out


@nki.jit
def p_stt_scalar(x, y):
    """scalar_tensor_tensor(dst, data, op0, operand0=[P,H,1], op1, operand1=tensor)."""
    out = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.shared_hbm)
    xb = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    yb = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:P])
    nisa.dma_copy(dst=yb, src=y[0:P])
    s = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s, op=nl.add, data=xb, axis=(2,))
    r = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=r, data=xb, op0=nl.multiply,
                              operand0=s.ap(pattern=BC), op1=nl.add, operand1=yb)
    nisa.dma_copy(dst=out, src=r)
    return out


def run(name, fn, *args):
    print(f"\n=== {name} ===")
    try:
        r = fn(*args)
        if isinstance(r, tuple):
            r = tuple(t.cpu() for t in r)
            print(f"  OK shapes={[tuple(t.shape) for t in r]}")
        else:
            r = r.cpu()
            print(f"  OK shape={tuple(r.shape)} finite={bool(torch.isfinite(r).all())}")
        return r
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL: {type(e).__name__}: {str(e)[:600]}")
        return None


torch.manual_seed(0)
x = torch.rand(P, H, D, dtype=torch.float32) + 0.5
y = torch.rand(P, H, D, dtype=torch.float32) + 0.5
xd, yd = x.to("neuron"), y.to("neuron")

r = run("p_rsqrt", p_rsqrt, xd)
if r is not None:
    print(f"  maxdiff={(r - torch.rsqrt(x.sum(-1, keepdim=True))).abs().max().item():.3e}")

r = run("p_act_reduce", p_act_reduce, xd)
if r is not None:
    print(f"  sq maxdiff={(r[0] - x * x).abs().max().item():.3e}")
    print(f"  ss maxdiff={(r[1] - (x * x).sum(-1, keepdim=True)).abs().max().item():.3e}")

r = run("p_ts_chain", p_ts_chain, xd)
if r is not None:
    print(f"  maxdiff={(r - (x.sum(-1, keepdim=True) / D + 0.5)).abs().max().item():.3e}")

r = run("p_divide", p_divide, xd, yd)
if r is not None:
    print(f"  maxdiff={(r - x.sum(-1, keepdim=True) / y.sum(-1, keepdim=True)).abs().max().item():.3e}")

r = run("p_inplace", p_inplace, xd)
if r is not None:
    ref = torch.cat([x[..., :Dh] * x[..., Dh:] + x[..., Dh:], x[..., Dh:] - x[..., :Dh]], dim=-1)
    print(f"  maxdiff={(r - ref).abs().max().item():.3e}")

r = run("p_stt_scalar", p_stt_scalar, xd, yd)
if r is not None:
    print(f"  maxdiff={(r - (x * x.sum(-1, keepdim=True) + y)).abs().max().item():.3e}")

print("\nDONE")
