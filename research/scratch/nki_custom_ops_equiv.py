"""Verify the torch.library custom-op wrapping of the two NKI kernels.

Three independent questions, in order, each answered with a real measurement:

  A. NUMERICS. Does routing the kernels through `tfc::rope_norm` /
     `tfc::softcap_ce` instead of `torch.autograd.Function.apply` change any
     output or gradient? It must not -- the kernels, arguments and backward math
     are identical, only the dispatch path differs. Checked bitwise.

  B. TRACING (the whole point). Does Dynamo graph-break on each path? Measured
     with `backend="eager"` so this costs no neuronx-cc time: the question is
     purely "can Dynamo trace it", and `fullgraph=True` turns a graph break into
     a hard error, which is an unambiguous signal either way.

  C. REAL LOWERING. Does `backend="neuron"` actually compile a graph containing
     the custom op, and produce the same answer? (Slow -- one compile per case.)

Run on the box:
    source ~/workspace/native_venv/bin/activate
    NEURON_RT_VISIBLE_CORES=0 python3 research/scratch/nki_custom_ops_equiv.py
Add --skip-c to run only A and B (fast, no neuronx-cc).
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

import torch

import train as T

# Real locked shapes: n_embd = ceil(DEPTH*ASPECT_RATIO / HEAD_DIM)*HEAD_DIM = 640,
# n_head = 5, B rows = DEVICE_BATCH_SIZE*PACK_FACTOR = 2, T = 2048, V = 8192.
B, TT, H, D = 2, T.SEQ_LEN, 5, T.HEAD_DIM
V = T.TOKENIZER_VOCAB_SIZE


def banner(msg: str) -> None:
    print(f"\n{'=' * 78}\n{msg}\n{'=' * 78}", flush=True)


def make_rope_inputs(device):
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(B, TT, H, D, generator=g, dtype=torch.float32).to(torch.bfloat16)
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    ang = torch.arange(TT, dtype=torch.float32)[:, None] * inv[None, :]
    cos = ang.cos().to(torch.bfloat16).view(1, TT, 1, D // 2)
    sin = ang.sin().to(torch.bfloat16).view(1, TT, 1, D // 2)
    gout = torch.randn(B, TT, H, D, generator=g, dtype=torch.float32).to(torch.bfloat16)
    return (x.to(device), cos.to(device), sin.to(device), gout.to(device))


def make_ce_inputs(device):
    g = torch.Generator().manual_seed(4321)
    logits = (torch.randn(B * TT, V, generator=g, dtype=torch.float32) * 6.0).contiguous()
    tgt = torch.randint(0, V, (B, TT), generator=g, dtype=torch.long)
    return logits.to(device), tgt.to(device)


def run_rope(x, cos, sin, gout, custom_ops: bool):
    T.USE_NKI_CUSTOM_OPS = custom_ops
    xr = x.detach().clone().requires_grad_(True)
    y = T.rope_norm(xr, cos, sin)
    y.backward(gout)
    return y.detach(), xr.grad.detach()


def run_ce(logits, tgt, custom_ops: bool):
    T.USE_NKI_CUSTOM_OPS = custom_ops
    z = logits.detach().clone().requires_grad_(True)
    loss = T.softcap_cross_entropy(z, tgt, "mean")
    loss.backward()
    return loss.detach(), z.grad.detach()


def compare(name, ref, new):
    ref_c, new_c = ref.float().cpu(), new.float().cpu()
    same = torch.equal(ref_c, new_c)
    diff = (ref_c - new_c).abs()
    n_diff = int((diff > 0).sum())
    print(f"  {name:28s} bitwise_identical={same!s:5s} "
          f"max_abs_diff={diff.max().item():.3e} n_differing={n_diff}/{diff.numel()}")
    return same


def part_a(device):
    banner("A. NUMERICAL EQUIVALENCE (eager, bitwise) -- must be identical")
    ok = True

    x, cos, sin, gout = make_rope_inputs(device)
    y_ref, gx_ref = run_rope(x, cos, sin, gout, custom_ops=False)
    y_new, gx_new = run_rope(x, cos, sin, gout, custom_ops=True)
    print("rope+QK-norm:")
    ok &= compare("forward y", y_ref, y_new)
    ok &= compare("backward dx", gx_ref, gx_new)

    logits, tgt = make_ce_inputs(device)
    l_ref, gz_ref = run_ce(logits, tgt, custom_ops=False)
    l_new, gz_new = run_ce(logits, tgt, custom_ops=True)
    print("softcap+CE:")
    ok &= compare("forward loss", l_ref, l_new)
    ok &= compare("backward dz", gz_ref, gz_new)
    print(f"  (loss values: autograd.Function={l_ref.item():.10f} custom_op={l_new.item():.10f})")

    print(f"\nPART A VERDICT: {'PASS (bit-identical)' if ok else 'FAIL'}")
    return ok


def trace_probe(label, fn, args, custom_ops):
    """Return (n_graph_breaks, fullgraph_ok, err) for Dynamo-only tracing."""
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    T.USE_NKI_CUSTOM_OPS = custom_ops

    dynamo.reset()
    counters.clear()
    breaks = 0
    try:
        compiled = torch.compile(fn, backend="eager", dynamic=False)
        compiled(*args)
        breaks = sum(counters["graph_break"].values())
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:34s} non-fullgraph trace RAISED {type(exc).__name__}: {exc}")
        return None, False, f"{type(exc).__name__}"

    dynamo.reset()
    counters.clear()
    try:
        compiled = torch.compile(fn, backend="eager", dynamic=False, fullgraph=True)
        compiled(*args)
        full_ok, err = True, ""
    except Exception as exc:  # noqa: BLE001
        full_ok, err = False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:130]}"

    print(f"  {label:34s} graph_breaks={breaks:<3d} fullgraph_ok={full_ok!s:5s} {err}")
    return breaks, full_ok, err


def part_b(device):
    banner("B. DYNAMO TRACING (backend='eager', no neuronx-cc) -- fewer breaks is the win")

    x, cos, sin, gout = make_rope_inputs(device)
    logits, tgt = make_ce_inputs(device)

    def rope_fwd_bwd(x_, cos_, sin_, g_):
        y = T.rope_norm(x_, cos_, sin_)
        return (y * g_).sum()

    def ce_fwd(z_, t_):
        return T.softcap_cross_entropy(z_, t_, "mean")

    xr = x.detach().clone().requires_grad_(True)
    zr = logits.detach().clone().requires_grad_(True)

    print("rope+QK-norm:")
    r_old = trace_probe("autograd.Function (current)", rope_fwd_bwd, (xr, cos, sin, gout), False)
    r_new = trace_probe("torch.library custom op (new)", rope_fwd_bwd, (xr, cos, sin, gout), True)
    print("softcap+CE:")
    c_old = trace_probe("autograd.Function (current)", ce_fwd, (zr, tgt), False)
    c_new = trace_probe("torch.library custom op (new)", ce_fwd, (zr, tgt), True)

    improved = []
    for nm, old, new in (("rope_norm", r_old, r_new), ("softcap_ce", c_old, c_new)):
        ob, ofull, _ = old
        nb, nfull, _ = new
        better = (nfull and not ofull) or (ob is not None and nb is not None and nb < ob)
        improved.append(better)
        print(f"  -> {nm}: fullgraph {ofull} -> {nfull}, breaks {ob} -> {nb} "
              f"({'IMPROVED' if better else 'no improvement'})")
    print(f"\nPART B VERDICT: {'PASS' if all(improved) else 'PARTIAL/FAIL'}")
    return all(improved)


def part_c(device):
    banner("C. REAL backend='neuron' LOWERING (slow: one neuronx-cc compile per case)")
    import torch._dynamo as dynamo

    x, cos, sin, gout = make_rope_inputs(device)
    logits, tgt = make_ce_inputs(device)
    ok = True

    for custom_ops in (False, True):
        tag = "custom_op" if custom_ops else "autograd.Function"
        T.USE_NKI_CUSTOM_OPS = custom_ops

        def combined(x_, cos_, sin_, g_, z_, t_):
            y = T.rope_norm(x_, cos_, sin_)
            return (y * g_).sum() + T.softcap_cross_entropy(z_, t_, "mean")

        dynamo.reset()
        xr = x.detach().clone().requires_grad_(True)
        zr = logits.detach().clone().requires_grad_(True)
        try:
            compiled = torch.compile(combined, backend="neuron", dynamic=False)
            out = compiled(xr, cos, sin, gout, zr, tgt)
            out.backward()
            print(f"  {tag:22s} compiled OK  out={out.detach().float().cpu().item():.8f}  "
                  f"grad_x_norm={xr.grad.float().norm().item():.8f}  "
                  f"grad_z_norm={zr.grad.float().norm().item():.8f}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  {tag:22s} FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
    print(f"\nPART C VERDICT: {'both paths compile' if ok else 'a path FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-c", action="store_true", help="skip the slow real-compile stage")
    args = ap.parse_args()

    print(f"torch={torch.__version__}")
    print(f"_HAS_NKI={T._HAS_NKI} _HAS_NKI_CUSTOM_OPS={T._HAS_NKI_CUSTOM_OPS} "
          f"err={T._NKI_CUSTOM_OPS_ERR!r}")
    if not T._HAS_NKI_CUSTOM_OPS:
        raise SystemExit("custom-op registration failed -- nothing to test")
    print(f"registered ops: {[n for n in dir(torch.ops.tfc)]}")
    print(f"shapes: rope x={(B, TT, H, D)}  ce logits={(B * TT, V)}")

    device = torch.device("neuron")
    results = {"A": part_a(device), "B": part_b(device)}
    if not args.skip_c:
        results["C"] = part_c(device)

    banner("SUMMARY")
    for k, v in results.items():
        print(f"  part {k}: {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
