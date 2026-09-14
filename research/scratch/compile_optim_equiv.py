#!/usr/bin/env python3
"""T1.7 equivalence gate: is the compiled Muon step (train.py USE_COMPILE_MUON /
--compile-muon) equivalent to the in-tree eager and NKI Muon steps?

Two separate questions, deliberately kept apart, because they have different standards:

  1. **The SOURCE REWRITE** -- `_muon_math` is a foreach-free, allocation-functional,
     device-scalar re-expression of `_muon_step`'s eager body. Run UNCOMPILED
     (--eager-math), this must be BIT-EXACT against the eager body: 0 differing
     elements. Anything else is a bug in the rewrite, and it can be checked on CPU
     with no Neuron device at all.
  2. **THE COMPILER** -- with torch.compile(backend="neuron") actually on, neuronx-cc
     is free to fuse and re-associate, so the standard is the same one the NKI kernels
     that are NOT bit-exact were held to: agreement to a few bf16 ULP, reported as a
     real number rather than asserted.

Three arms over identically-initialised parameter sets, fed identical synthetic grads
and a per-step-varying lr/momentum/weight_decay schedule (the compiled path carries all
three as runtime device operands, so a static schedule would not exercise the one thing
most likely to be wrong about them):

    ref   eager loop body,   USE_NKI_MUON=False, USE_COMPILE_MUON=False
    nki   NKI fusion,        USE_NKI_MUON=True   <- the current shipped default
    comp  compiled body,     USE_COMPILE_MUON=True

Every Muon parameter AND every Muon optimizer-state tensor is compared element by
element after every step, so divergence shows up immediately rather than averaged away.

Also checks, up front and on its own, the one API-level assumption the rewrite makes:
`torch.lerp(a, b, w_tensor)` (lerp.Tensor, what the compiled body uses) versus
`torch.lerp(a, b, w_float)` (lerp.Scalar, what the eager body uses) must be bitwise
identical for the fp32/bf16 shapes involved.

Usage:
    # source-rewrite gate, no device needed
    python research/scratch/compile_optim_equiv.py --device cpu --eager-math --steps 6
    # full gate on hardware
    NEURON_RT_VISIBLE_CORES=0 python research/scratch/compile_optim_equiv.py --steps 6
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

REPO = os.environ.get("PROF_REPO", os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch

import train as T

# `stacked_grads` is deliberately NOT gated. It is pure scratch -- `_muon_step` refills
# every row of it from the live `p.grad` at the top of every step -- and the eager body
# happens to reuse it as the destination of its in-place `g.lerp_(momentum_buffer, mom)`
# while the compiled body (which must not mutate a graph input) leaves it holding the raw
# gradient. So it differs by construction and that difference is unobservable: nothing
# reads it before the next refill. It is still compared and reported, just not gated.
STATE_KEYS = ("momentum_buffer", "second_momentum_buffer", "stacked_params")
SCRATCH_KEYS = ("stacked_grads",)


def build_arm(device, seed=42):
    config = T.build_config()
    torch.manual_seed(seed)
    model = T.GPT(config)
    model.init_weights()
    model.to(device)
    opt = model.setup_optimizer()
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return model, opt


def cmp_exact(a: torch.Tensor, b: torch.Tensor):
    x, y = a.detach().float().cpu(), b.detach().float().cpu()
    ne = int((x != y).sum())
    md = float((x - y).abs().max()) if x.numel() else 0.0
    return ne, md, x.numel()


def set_schedule(opt, step):
    lrm = 1.0 - 0.017 * step
    wd_progress = min(1.0, step / 7.0)
    for group in opt.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = 0.95 - 0.01 * step
            group["weight_decay"] = T.WEIGHT_DECAY * (1.0 - wd_progress)


def lerp_overload_check(device):
    """lerp.Tensor (0-dim fp32 weight) vs lerp.Scalar (Python float weight).

    Only the fp32 result gates: _muon_math lerps `momentum_buffer`/`g` and
    `second`/`v_mean`, all four of which are fp32 (the compiled branch is gated on
    `w.dtype == torch.float32`). The bf16 row is reported anyway because it is a real
    and non-obvious hazard for any FUTURE use of a tensor lerp weight -- lerp.Scalar
    computes bf16 in fp32 opmath and rounds once, lerp.Tensor rounds the intermediate,
    so they disagree by a full bf16 ULP.
    """
    bad = 0
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(7)
        a = torch.randn(257, 63, dtype=dtype, device=device)
        b = torch.randn(257, 63, dtype=dtype, device=device)
        for wv in (0.05, 0.95, 0.09999999999999998, 0.8500000000000001):
            wt = torch.tensor(wv, dtype=torch.float32, device=device)
            ne, md, _ = cmp_exact(torch.lerp(a, b, wv), torch.lerp(a, b, wt))
            if dtype == torch.float32:
                bad += ne
            if ne:
                print(f"  lerp {dtype} w={wv!r}: {ne} differ, max|diff| {md:.3e}"
                      + ("" if dtype == torch.float32 else "   (informational, not gated)"))
    print(f"lerp overload check (fp32, gated): {'IDENTICAL' if bad == 0 else f'{bad} DIFFERING'}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--eager-math", action="store_true",
                    help="call _muon_math directly instead of torch.compile-ing it, to "
                         "separate the source rewrite from the compiler")
    args = ap.parse_args()

    T.USE_TENSOR_LR_SCALARS = True
    T.USE_FOREACH_OPTIM = True
    device = torch.device(args.device)
    T.COMPUTE_DTYPE = T.compute_dtype_for(device)
    print(f"device={device} compute_dtype={T.COMPUTE_DTYPE} has_nki={T._HAS_NKI} "
          f"eager_math={args.eager_math}")

    if args.eager_math:
        # keep the exact same call site, minus the compiler (a non-neuron device already
        # takes this path by itself -- see the comment in _muon_step)
        class _PassThrough(dict):
            def get(self, key, default=None):  # noqa: D102
                return T._muon_math

        T._MUON_COMPILED = _PassThrough()

    bad_lerp = lerp_overload_check(device)

    arms = {}
    for name in ("ref", "nki", "comp"):
        if name == "nki" and not (T._HAS_NKI and device.type == "neuron"):
            continue
        arms[name] = build_arm(device, args.seed)
    muon = {k: [g for g in o.param_groups if g["kind"] == "muon"]
            for k, (_m, o) in arms.items()}
    print("muon groups: " + str([(tuple(g["params"][0].shape), len(g["params"]),
                                  str(g["params"][0].dtype)) for g in muon["ref"]]))
    n_fp32_groups = sum(1 for g in muon["ref"] if g["params"][0].dtype == torch.float32)
    print(f"muon groups eligible for the compiled branch (fp32): {n_fp32_groups}")
    assert n_fp32_groups > 0, "no fp32 Muon group -- the compiled branch would never fire"

    ref_model = arms["ref"][0]
    for name, (m, _o) in arms.items():
        if name == "ref":
            continue
        for pr, pn in zip(ref_model.parameters(), m.parameters()):
            ne, md, _ = cmp_exact(pr, pn)
            assert ne == 0, f"arm {name} differs at init by {ne} elements (max {md})"

    def run(name, flags):
        T.USE_NKI_MUON, T.USE_COMPILE_MUON = flags
        set_schedule(arms[name][1], step)
        for g in muon[name]:
            arms[name][1]._muon_step(g)
        T.USE_NKI_MUON, T.USE_COMPILE_MUON = False, False

    worst = {k: 0.0 for k in arms if k != "ref"}
    bad = {k: 0 for k in arms if k != "ref"}
    scratch = {k: 0 for k in arms if k != "ref"}
    for step in range(args.steps):
        grads = {}
        for name, p in ref_model.named_parameters():
            grads[name] = (torch.randn(p.shape, dtype=torch.float32,
                                       generator=torch.Generator().manual_seed(
                                           hash((step, name)) % (2 ** 31))) * 0.01)
        for _m, _o in arms.values():
            for name, p in _m.named_parameters():
                p.grad = grads[name].clone().to(device=device, dtype=p.dtype)

        run("ref", (False, False))
        if "nki" in arms:
            run("nki", (True, False))
        run("comp", (False, True))
        T.synchronize(device)

        for name in worst:
            for gi, (gr, gx) in enumerate(zip(muon["ref"], muon[name])):
                shape = tuple(gr["params"][0].shape)
                for pi, (pr, px) in enumerate(zip(gr["params"], gx["params"])):
                    ne, md, n = cmp_exact(pr, px)
                    worst[name] = max(worst[name], md)
                    bad[name] += ne
                    if ne and pi == 0:
                        print(f"  [step {step}] {name} group{gi}{shape} param{pi}: "
                              f"{ne}/{n} differ, max|diff| {md:.3e}")
                sr = arms["ref"][1].state[gr["params"][0]]
                sx = arms[name][1].state[gx["params"][0]]
                for key in STATE_KEYS + SCRATCH_KEYS:
                    if key not in sr or key not in sx:
                        continue
                    ne, md, n = cmp_exact(sr[key], sx[key])
                    if key in SCRATCH_KEYS:
                        scratch[name] += ne
                        continue
                    worst[name] = max(worst[name], md)
                    bad[name] += ne
                    if ne:
                        print(f"  [step {step}] {name} group{gi}{shape} {key}: "
                              f"{ne}/{n} differ, max|diff| {md:.3e}")
        print(f"step {step}: " + "  ".join(
            f"{k}: {'EXACT' if bad[k] == 0 else f'{bad[k]} diff'} (worst {worst[k]:.3e})"
            for k in worst), flush=True)

    fired = len(T._MUON_COMPILED) if not args.eager_math else n_fp32_groups
    print("\n=== T1.7 COMPILED-MUON EQUIVALENCE ===")
    print(f"compiled specialisations built: {fired} (expected {n_fp32_groups})")
    print(f"steps compared      {args.steps}")
    print(f"lerp overload       {'IDENTICAL' if bad_lerp == 0 else 'DIFFERS'}")
    for k in worst:
        print(f"{k:5s} vs ref        differing={bad[k]}  worst max|diff|={worst[k]:.3e}"
              f"   [ungated scratch stacked_grads: {scratch[k]}]")
    ok = bad_lerp == 0 and bad.get("comp", 1) == 0
    if args.eager_math:
        print("VERDICT (source rewrite): " + ("BIT-EXACT" if ok else "NOT EXACT -- bug in _muon_math"))
        return 0 if ok else 1
    print("VERDICT (compiled): " + ("BIT-EXACT" if ok else
                                    f"not bit-exact, max|diff| {worst.get('comp', 0.0):.3e} "
                                    "-- judge against bf16 ULP"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
