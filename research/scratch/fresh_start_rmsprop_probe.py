#!/usr/bin/env python3
"""Probe the table optimizer's state precision, then a fused FP32-state NKI step.

This is an isolated numerical/performance experiment, not a training default.
BF16 parameters and gradients remain BF16; second moments and optimizer arithmetic
use FP32. Thus this changes more than the storage of a rounded BF16 moment alone.
The reference is independently computed on CPU. No model/data/eval code is edited.

Run CPU checks while training is active with --device cpu. Use --device neuron only
on an idle host; restrict NEURON_RT_VISIBLE_CORES externally. --bench-elements can
be 134217728 for one real 1M-by-128 table; smaller correctness cases run first.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch


def reference(p, grad, moment, lr, bias2, beta2=0.999, eps=1e-10):
    """FP32 arithmetic, BF16 stored weights; no device kernel used as reference."""
    g = grad.float()
    v = torch.lerp(moment.float(), g.square(), 1 - beta2)
    w = (p.float() - lr * g / ((v / bias2).sqrt() + eps)).to(p.dtype)
    return w, v


def cpu_decay_check():
    values = torch.logspace(-30, 30, 1001).bfloat16()
    bf, fp = values.clone(), values.float()
    torch._foreach_lerp_([bf], [torch.zeros_like(bf)], 0.001)
    torch._foreach_lerp_([fp], [torch.zeros_like(fp)], 0.001)
    assert torch.equal(bf, values), "Unexpected BF16 CPU rounding semantics"
    assert not torch.any(fp == values.float())
    a, b = torch.ones(128, dtype=torch.bfloat16), torch.ones(128)
    za, zb = torch.zeros_like(a), torch.zeros_like(b)
    for _ in range(1000):
        torch._foreach_lerp_([a], [za], 0.001)
        torch._foreach_lerp_([b], [zb], 0.001)
    torch.testing.assert_close(b, torch.full_like(b, 0.999 ** 1000), rtol=1e-5, atol=0)
    assert torch.all(a == 1)
    report = dict(check="cpu_decay", unchanged_bf16=1001, tested=1001,
                  zero_steps=1000, bf16_value=float(a[0]), fp32_value=float(b[0]))
    print(json.dumps(report), flush=True)


def build_kernel():
    # NKI 0.5 resolves names from module globals, not this factory's closure.
    global nisa, nl, nt
    import nki
    import nki.isa as nisa
    import nki.language as nl
    import neuronxcc.nki.typing as nt

    @nki.jit
    def fused_table_rmsprop(p: nt.tensor, grad, moment: nt.tensor, scalars,
                           lanes: int = 1, inplace: bool = False):
        """Return an updated BF16 table and FP32 moment, with optional aliasing.

        Args: p/grad [M,F] BF16 HBM, moment [M,F] FP32 HBM,
              scalars [128,2] FP32 HBM (lr and inverse bias correction).
        Constraints: M divisible by 128*lanes; F <= 4096; all arrays contiguous.
        Each element: v=lerp(v,g*g,.001); p=BF16(p-lr*g/(sqrt(v/bias)+eps)).
        One tile retains optimizer intermediates in SBUF; there are no intermediate
        full-table HBM writes. Compare explicit copy-back with a mutable-input
        variant; benchmark native alias handling rather than assuming it is free.
        """
        m, f = p.shape
        assert lanes in (1, 2) and m % (128 * lanes) == 0 and f <= 4096
        assert grad.shape == p.shape and moment.shape == p.shape
        if inplace:
            out_p, out_v = p, moment
        else:
            out_p = nl.ndarray(p.shape, dtype=p.dtype, buffer=nl.shared_hbm)
            out_v = nl.ndarray(moment.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        lr = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        inv_bias = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lr, src=scalars[:, 0:1])
        nisa.dma_copy(dst=inv_bias, src=scalars[:, 1:2])
        lane = nl.program_id(0)
        for tile in nl.affine_range(m // (128 * lanes)):
            start = (tile * lanes + lane) * 128
            ps = nl.ndarray((128, f), dtype=p.dtype, buffer=nl.sbuf)
            gs = nl.ndarray((128, f), dtype=grad.dtype, buffer=nl.sbuf)
            vs = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ps, src=p[start:start + 128, :])
            nisa.dma_copy(dst=gs, src=grad[start:start + 128, :])
            nisa.dma_copy(dst=vs, src=moment[start:start + 128, :])
            gg = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gg, data1=gs, data2=gs, op=nl.multiply)
            dv = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dv, data1=gg, data2=vs, op=nl.subtract)
            nv = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=nv, data=dv, op0=nl.multiply,
                                      operand0=0.001, op1=nl.add, operand1=vs)
            debiased = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=debiased, data=nv, op0=nl.multiply, operand0=inv_bias)
            denom = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=denom, data=debiased, op=nl.sqrt)
            nisa.tensor_scalar(dst=denom, data=denom, op0=nl.add, operand0=1e-10)
            inv_denom = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv_denom, data=denom)
            update = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=update, data1=gs, data2=inv_denom, op=nl.multiply)
            nisa.tensor_scalar(dst=update, data=update, op0=nl.multiply, operand0=lr)
            np = nl.ndarray((128, f), dtype=p.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=np, data1=ps, data2=update, op=nl.subtract)
            nisa.dma_copy(dst=out_p[start:start + 128, :], src=np)
            nisa.dma_copy(dst=out_v[start:start + 128, :], src=nv)
        return out_p, out_v

    return fused_table_rmsprop


def synchronize():
    torch.neuron.synchronize()


def scalar_tensor(lr, bias, device):
    return torch.tensor([lr, 1.0 / bias], dtype=torch.float32).repeat(128, 1).to(device)


def hardware_checks(kernel, device):
    torch.manual_seed(927)
    cases = 0
    for rows, cols in ((128, 1024), (256, 2048)):
        for case in ("zero", "sparse", "dense", "late"):
            p = (torch.randn(rows, cols) * 0.01).bfloat16()
            g = (torch.randn(rows, cols) * 0.003).bfloat16()
            v = torch.rand(rows, cols) * 1e-5
            if case == "zero":
                g.zero_()
                v[0].zero_()  # zero / epsilon must be finite and stay zero
            if case == "sparse":
                g[torch.rand(rows, cols) < 0.97] = 0
                v.zero_()
            if case == "late":
                g *= 0.001
            lr, bias = (0.012, 0.001) if case == "sparse" else (0.001, 0.865)
            expected_p, expected_v = reference(p, g, v, lr, bias)
            for lanes in ((1, 2) if rows % 256 == 0 else (1,)):
                for inplace in (False, True):
                    pd, gd, vd = p.to(device), g.to(device), v.to(device)
                    actual_pd, actual_vd = kernel[lanes](
                        pd, gd, vd, scalar_tensor(lr, bias, device), lanes, inplace)
                    actual_p, actual_v = actual_pd.cpu(), actual_vd.cpu()
                    torch.testing.assert_close(actual_v, expected_v, rtol=2e-5, atol=1e-14)
                    # One output BF16 ULP can differ at a rounding boundary after reciprocal.
                    # Measure exact-match fraction too; do not claim bit-exact equivalence.
                    torch.testing.assert_close(actual_p, expected_p, rtol=0.009, atol=2e-7)
                    assert torch.isfinite(actual_p).all() and torch.isfinite(actual_v).all()
                    if inplace:
                        assert torch.equal(pd.cpu(), actual_p) and torch.equal(vd.cpu(), actual_v)
                    else:
                        assert torch.equal(pd.cpu(), p) and torch.equal(vd.cpu(), v), "input mutation"
                    assert torch.equal(gd.cpu(), g), "gradient mutation"
                    if case == "zero":
                        assert torch.equal(actual_p, p)
                        assert (actual_v[1:] < v[1:]).all()
                    print(json.dumps(dict(check="nki_correctness", rows=rows, cols=cols, case=case,
                                          lanes=lanes, inplace=inplace,
                                          weight_exact_fraction=float((actual_p == expected_p).float().mean()),
                                          max_weight_abs=float((actual_p.float() - expected_p.float()).abs().max()),
                                          max_moment_abs=float((actual_v - expected_v).abs().max()))), flush=True)
                    cases += 1
    # The exact legacy native operation, not the new NKI kernel, must be checked too.
    b = torch.ones(128, dtype=torch.bfloat16, device=device)
    f = torch.ones(128, dtype=torch.float32, device=device)
    bz, fz = torch.zeros_like(b), torch.zeros_like(f)
    for _ in range(1000):
        torch._foreach_lerp_([b], [bz], 0.001)
        torch._foreach_lerp_([f], [fz], 0.001)
    bc, fc = b.cpu(), f.cpu()
    assert torch.all(bc == 1)
    torch.testing.assert_close(fc, torch.full_like(fc, 0.999 ** 1000), rtol=2e-4, atol=0)
    print(json.dumps(dict(check="native_decay", zero_steps=1000,
                          bf16_value=float(bc[0]), fp32_value=float(fc[0]),
                          nki_cases=cases)), flush=True)


def benchmark(kernel, device, elements, steps):
    cols = 2048
    assert elements % (128 * cols) == 0
    shape = (elements // cols, cols)
    p = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    g = torch.full(shape, 0.001, dtype=torch.bfloat16, device=device)
    v = torch.full(shape, 1e-6, dtype=torch.float32, device=device)
    vb = v.bfloat16()
    scalars = scalar_tensor(0.001, 0.865, device)

    def nki_step(lanes, inplace):
        pn, vn = kernel[lanes](p, g, v, scalars, lanes, inplace)
        if not inplace:
            p.copy_(pn)
            v.copy_(vn)

    def fp32_step():
        gf = g.float()
        torch._foreach_lerp_([v], torch._foreach_mul([gf], [gf]), 0.001)
        d = torch._foreach_div([v], [0.865])
        torch._foreach_sqrt_(d)
        torch._foreach_add_(d, 1e-10)
        u = torch._foreach_div([gf], d)
        torch._foreach_add_([p], u, alpha=-0.001)

    def legacy_step():
        torch._foreach_lerp_([vb], torch._foreach_mul([g], [g]), 0.001)
        d = torch._foreach_div([vb], [0.865])
        torch._foreach_sqrt_(d)
        torch._foreach_add_(d, 1e-10)
        u = torch._foreach_div([g], d)
        torch._foreach_add_([p], u, alpha=-0.001)

    arms = {"legacy_bf16": legacy_step, "fp32_math": fp32_step,
            "nki_1_copyback": lambda: nki_step(1, False),
            "nki_2_copyback": lambda: nki_step(2, False),
            "nki_1_inplace": lambda: nki_step(1, True),
            "nki_2_inplace": lambda: nki_step(2, True)}
    for fn in arms.values():
        for _ in range(3):
            fn()
        synchronize()
    samples = {name: [] for name in arms}
    for rep in range(3):
        order = list(arms) if rep % 2 == 0 else list(reversed(arms))
        for name in order:
            synchronize()
            started = time.perf_counter()
            for _ in range(steps):
                arms[name]()
            synchronize()
            samples[name].append(1000 * (time.perf_counter() - started) / steps)
    print(json.dumps(dict(check="optimizer_benchmark", elements=elements,
                          logical_cores=os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
                          visible_cores=os.environ.get("NEURON_RT_VISIBLE_CORES"),
                          samples_ms=samples,
                          median_ms={name: statistics.median(s) for name, s in samples.items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "neuron"), default="cpu")
    parser.add_argument("--bench-elements", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cpu_decay_check()
    if args.device == "neuron":
        import torch_neuronx  # noqa: F401 -- register native backend
        device = torch.device("neuron")
        kernel = build_kernel()
        hardware_checks(kernel, device)
        if args.bench_elements:
            benchmark(kernel, device, args.bench_elements, args.steps)
    print("PASS: requested RMSProp probes completed", flush=True)


if __name__ == "__main__":
    main()
