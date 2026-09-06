"""Probe NKI 0.5.0 primitives needed for the fused rope+rmsnorm kernel.

Each probe is an independent try/except so one failure doesn't hide the rest.
Run on the Trn2 box: python nki_probe.py
"""
import os
import inspect

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch
import nki
import nki.language as nl
import nki.isa as nisa

P = 128
H = 3
D = 128
Dh = D // 2


def sig(fn_name, mod=nisa):
    try:
        print(f"  SIG {fn_name}: {inspect.signature(getattr(mod, fn_name))}")
    except Exception as e:  # noqa: BLE001
        print(f"  SIG {fn_name}: FAILED {e}")


print("=== signatures ===")
for name in ["tensor_tensor", "tensor_scalar", "tensor_reduce", "scalar_tensor_tensor",
             "activation", "tensor_copy", "rsqrt", "reciprocal", "dma_copy"]:
    sig(name)
sig("rms_norm", nl)


# ---------------------------------------------------------------- probe 1: 3D strided tensor_tensor
@nki.jit
def probe_3d_tt(x, cos_t):
    """xb[:, :, 0:64] * cos broadcast -> tests 3D strided operands."""
    BT = x.shape[0]
    out = nl.ndarray((BT, H, Dh), dtype=nl.float32, buffer=nl.shared_hbm)
    xb = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:P])
    cb = nl.ndarray((P, H, Dh), dtype=nl.float32, buffer=nl.sbuf)
    for h in nl.static_range(H):
        nisa.dma_copy(dst=cb[0:P, h, 0:Dh], src=cos_t[0:P, 0:Dh])
    res = nl.ndarray((P, H, Dh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=res, data1=xb[0:P, 0:H, 0:Dh], data2=cb, op=nl.multiply)
    nisa.dma_copy(dst=out[0:P], src=res)
    return out


# ---------------------------------------------------------------- probe 2: 3D reduce axis=(2,)
@nki.jit
def probe_3d_reduce(x):
    out = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    xb = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:P])
    s = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s, data=xb, op=nl.add, axis=(2,))
    nisa.dma_copy(dst=out, src=s)
    return out


# ---------------------------------------------------------------- probe 3: tensor_scalar with [P,1] operand
@nki.jit
def probe_ts_p1(x):
    out = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.shared_hbm)
    xb = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:P, 0:D])
    s = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s, data=xb, op=nl.add, axis=(1,))
    r = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.rsqrt(dst=r, data=s)
    res = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=res, data=xb, op0=nl.multiply, operand0=r)
    nisa.dma_copy(dst=out, src=res)
    return out


# ---------------------------------------------------------------- probe 4: stride-0 broadcast via .ap()
@nki.jit
def probe_ap_bcast(x):
    """[P,H,1] scalar tile broadcast to [P,H,D] via stride-0 access pattern."""
    out = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.shared_hbm)
    xb = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:P])
    s = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s, data=xb, op=nl.add, axis=(2,))
    res = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=res, data1=xb,
                       data2=s.ap(pattern=[[H, P], [1, H], [0, D]]),
                       op=nl.multiply)
    nisa.dma_copy(dst=out, src=res)
    return out


# ---------------------------------------------------------------- probe 5: HBM .ap() stride-0 broadcast load
@nki.jit
def probe_hbm_ap_bcast(cos_t):
    out = nl.ndarray((P, H, Dh), dtype=nl.float32, buffer=nl.shared_hbm)
    cb = nl.ndarray((P, H, Dh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cb, src=cos_t.ap(pattern=[[Dh, P], [0, H], [1, Dh]], offset=0))
    nisa.dma_copy(dst=out, src=cb)
    return out


# ---------------------------------------------------------------- probe 6: scalar_tensor_tensor with [P,1] scalar
@nki.jit
def probe_stt(x, y):
    out = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.shared_hbm)
    xb = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    yb = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:P, 0:D])
    nisa.dma_copy(dst=yb, src=y[0:P, 0:D])
    s = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s, data=xb, op=nl.add, axis=(1,))
    res = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=res, data=xb, op0=nl.multiply, scalar=s,
                              op1=nl.add, operand0=yb)
    nisa.dma_copy(dst=out, src=res)
    return out


# ---------------------------------------------------------------- probe 7: bf16 in / fp32 compute / bf16 out
@nki.jit
def probe_mixed(x):
    BT = x.shape[0]
    out = nl.ndarray((BT, D), dtype=x.dtype, buffer=nl.shared_hbm)
    xb = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:P, 0:D])  # bf16 HBM -> fp32 SBUF
    res = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=res, data1=xb, data2=xb, op=nl.multiply)
    ob = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ob, src=res)
    nisa.dma_copy(dst=out[0:P, 0:D], src=ob)
    return out


def run(name, fn, *args):
    print(f"\n=== {name} ===")
    try:
        r = fn(*args)
        r = r.cpu()
        print(f"  OK shape={tuple(r.shape)} dtype={r.dtype} finite={bool(torch.isfinite(r).all())}")
        return r
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print(f"  FAIL: {type(e).__name__}: {msg[:1200]}")
        return None


dev = "neuron"
torch.manual_seed(0)
x32 = torch.randn(P, H, D, dtype=torch.float32)
x2d = torch.randn(P, D, dtype=torch.float32)
y2d = torch.randn(P, D, dtype=torch.float32)
cos2d = torch.randn(P, Dh, dtype=torch.float32)
xbf = torch.randn(P, D, dtype=torch.bfloat16)

x32_d = x32.to(dev)
x2d_d = x2d.to(dev)
y2d_d = y2d.to(dev)
cos2d_d = cos2d.to(dev)
xbf_d = xbf.to(dev)

r1 = run("probe_3d_tt (3D strided tensor_tensor)", probe_3d_tt, x32_d, cos2d_d)
if r1 is not None:
    ref = x32[:, :, :Dh] * cos2d[:, None, :]
    print(f"  maxdiff={(r1 - ref).abs().max().item():.3e}")

r2 = run("probe_3d_reduce (axis=(2,))", probe_3d_reduce, x32_d)
if r2 is not None:
    ref = x32.sum(-1, keepdim=True)
    print(f"  maxdiff={(r2 - ref).abs().max().item():.3e}")

r3 = run("probe_ts_p1 (tensor_scalar [P,1] operand)", probe_ts_p1, x2d_d)
if r3 is not None:
    ref = x2d * torch.rsqrt(x2d.sum(-1, keepdim=True).abs())
    print(f"  (sign-sensitive ref, just check finite/shape) maxdiff={(r3 - ref).abs().max().item():.3e}")

r4 = run("probe_ap_bcast (SBUF stride-0 .ap broadcast)", probe_ap_bcast, x32_d)
if r4 is not None:
    ref = x32 * x32.sum(-1, keepdim=True)
    print(f"  maxdiff={(r4 - ref).abs().max().item():.3e}")

r5 = run("probe_hbm_ap_bcast (HBM stride-0 .ap broadcast)", probe_hbm_ap_bcast, cos2d_d)
if r5 is not None:
    ref = cos2d[:, None, :].expand(P, H, Dh)
    print(f"  maxdiff={(r5 - ref).abs().max().item():.3e}")

r6 = run("probe_stt (scalar_tensor_tensor [P,1] scalar)", probe_stt, x2d_d, y2d_d)
if r6 is not None:
    ref = x2d * x2d.sum(-1, keepdim=True) + y2d
    print(f"  maxdiff={(r6 - ref).abs().max().item():.3e}")

r7 = run("probe_mixed (bf16 HBM <-> fp32 SBUF)", probe_mixed, xbf_d)
if r7 is not None:
    ref = (xbf.float() * xbf.float()).bfloat16()
    print(f"  maxdiff={(r7.float() - ref.float()).abs().max().item():.3e}")

print("\nDONE")
