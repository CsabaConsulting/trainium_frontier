#!/usr/bin/env python3
"""Size the Muon elementwise-fusion prize BEFORE writing any NKI kernel.

The idea: never mind what a fused kernel computes -- what matters on this backend is
how many NEFF launches it costs, because ~54% of the optimizer window is pure
inter-launch gap. So emulate the *launch count* an ideal fusion would have, using
ordinary aten ops chosen so each fused region collapses to exactly one dispatch, and
time it against the real `_muon_step`.

The emulation is mathematically WRONG on purpose (it is timing-only) and its data
traffic is <= the real fused kernel's, so the measured delta is a genuine UPPER
BOUND on what an NKI fusion could recover. If the upper bound is small, no kernel
needs to be written.

Fusion plan being emulated (per Muon group, real launch counts from
research/scratch/muon_dispatch_census.py):

  region                                    now   fused
  read-in (2x _foreach_copy_)                 2      2
  momentum lerps + bf16 + normalize           7      1   (K1)
  Newton-Schulz: 15 bmm + 25 elementwise     40     25   (3 bmm + K2 + K3) x5
  second-moment v_mean from g                 3      1   (K4)
  second-moment scalar chain (tiny tensors)  15      1   (K5)
  final scale + mask + weight-decay update   10      1   (K6)
  device lr scalar copy                       1      1
  write-back (_foreach_copy_)                 1      1
  TOTAL                                      79     33

Usage (on the box, 4 ranks / LNC=2 to match the real shape):

  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 \
      research/scratch/muon_fusion_sizing.py --steps 6 --warmup 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

REPO = os.environ.get("PROF_REPO", os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch

import train as T


ALL_REGIONS = ("momnorm", "ns", "vmean", "scalars", "final")


@torch.no_grad()
def muon_step_fused_emul(opt, group, regions=ALL_REGIONS):
    """Launch-count emulation of the fused Muon step. Same op *shapes* and same
    matmuls; each region named in `regions` collapsed to one dispatch, every other
    region left at its real launch count so the marginal value of each candidate
    kernel is measurable on its own."""
    params = group["params"]
    if not params:
        return
    p0 = params[0]
    state = opt.state[p0]
    shape = p0.shape
    g = state["stacked_grads"]
    w = state["stacked_params"]

    # read-in: unchanged, 2 launches
    g_rows = list(g.unbind(0))
    grads = [p.grad for p in params]
    torch._foreach_copy_(g_rows, grads)
    torch._foreach_copy_(list(w.unbind(0)), params)

    momentum_buffer = state["momentum_buffer"]
    second = state["second_momentum_buffer"]

    # K1: momentum lerps + bf16 cast + norm-normalize  -> 1 launch (was 7)
    if "momnorm" in regions:
        x = g.to(torch.bfloat16)
    else:
        momentum_buffer.lerp_(g, 1 - group["momentum"])
        g = g.lerp_(momentum_buffer, group["momentum"])
        x = g.bfloat16() if T.COMPUTE_DTYPE == torch.bfloat16 else g
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)

    # Newton-Schulz: matmuls untouched, 5 elementwise ops per iter -> 2 (K2, K3)
    fuse_ns = "ns" in regions
    tall = x.shape[-2] > x.shape[-1]
    for a, b, c in T.POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
        if tall:
            xtx = x.mT @ x
            if fuse_ns:
                m = torch.add(xtx @ xtx, xtx)             # K2
                x = torch.add(x @ m, x)                   # K3
            else:
                x = a * x + x @ (b * xtx + c * (xtx @ xtx))
        else:
            xxt = x @ x.mT
            if fuse_ns:
                m = torch.add(xxt @ xxt, xxt)             # K2
                x = torch.add(m @ x, x)                   # K3
            else:
                x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    g = x

    red_dim = -1 if shape[-2] >= shape[-1] else -2
    red_dim_size = g.size(red_dim)
    # K4: float+square+mean over the reduced axis -> 1 launch (was 3)
    if "vmean" in regions:
        v_mean = g.mean(dim=red_dim, keepdim=True, dtype=torch.float32)
    else:
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)

    # K5: the entire tiny-tensor scalar chain -> 1 launch (was 15)
    if "scalars" in regions:
        scale = second.rsqrt()
    else:
        v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
        second.lerp_(v_mean, 1 - group["beta2"])
        step_size = second.clamp_min(1e-10).rsqrt()
        scaled = (v_mean * red_dim_size) * step_size.square()
        v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
        scale = (step_size * (v_norm / v_norm_new.clamp_min(1e-10)))

    lr_base = group["lr"]
    if T.USE_TENSOR_LR_SCALARS:
        lr_base = opt._device_scalar(group, "lr", lr_base, p0)   # 1 launch
    # K6: g*scale, mask, lr/wd update, w.sub_  -> 1 launch (was 10)
    if "final" in regions:
        w.addcmul_(g.to(w.dtype), scale.to(w.dtype), value=-1.0)
    else:
        g = g * scale.to(g.dtype)
        lr = lr_base * max(1.0, shape[-2] / shape[-1]) ** 0.5
        wd = group["weight_decay"]
        mask = (g * w) >= 0
        w.sub_(lr * g + lr * wd * w * mask)

    torch._foreach_copy_(params, list(w.unbind(0)))


def build(args):
    T.USE_TENSOR_LR_SCALARS = True
    T.USE_FOREACH_OPTIM = True

    device_type = T.detect_device_type("")
    ddp, rank, local_rank, world_size, device = T.init_runtime(device_type)
    T.COMPUTE_DTYPE = T.compute_dtype_for(device)

    config = T.build_config()
    torch.manual_seed(42)
    model = T.GPT(config)
    model.init_weights()
    model.to(device)
    opt = model.setup_optimizer()
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

    # Real grads are irrelevant to launch-gap timing; skipping fwd/bwd isolates the
    # optimizer step and makes the measurement cheap and repeatable.
    for p in model.parameters():
        p.grad = torch.randn(p.shape, dtype=p.dtype).to(device) * 0.01
    return ddp, rank, world_size, device, opt


def run_mode(opt, device, regions, steps, warmup):
    """regions=None means the real _muon_step; otherwise a tuple of fused regions."""
    muon = [g for g in opt.param_groups if g["kind"] == "muon"]
    adamw = [g for g in opt.param_groups if g["kind"] == "adamw"]

    def muon_pass():
        for g in muon:
            if regions is None:
                opt._muon_step(g)
            else:
                muon_step_fused_emul(opt, g, regions)

    for _ in range(warmup):
        for g in adamw:
            opt._adamw_step_foreach(g)
        muon_pass()
        T.synchronize(device)

    muon_only = []
    for _ in range(steps):
        for g in adamw:
            opt._adamw_step_foreach(g)
        T.synchronize(device)
        t = time.monotonic()
        muon_pass()
        T.synchronize(device)
        muon_only.append((time.monotonic() - t) * 1e3)
    return muon_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2, help="interleaved repeats per mode")
    args = ap.parse_args()

    ddp, rank, world_size, device, opt = build(args)

    # baseline, then each region alone, then all together -- interleaved so a slow
    # patch of the box cannot land entirely on one arm.
    modes = [("baseline", None)]
    modes += [(r, (r,)) for r in ALL_REGIONS]
    modes.append(("all", ALL_REGIONS))

    acc = {name: [] for name, _ in modes}
    for _rep in range(args.reps):
        for name, regions in modes:
            acc[name] += run_mode(opt, device, regions, args.steps, args.warmup)
            if rank == 0:
                print(f"[sz] rep{_rep} {name:<9} muon_only "
                      f"{statistics.median(acc[name][-args.steps:]):7.2f}ms", flush=True)

    if rank == 0:
        base = statistics.median(acc["baseline"])
        # launches saved per region, from muon_dispatch_census.py, x4 Muon groups
        saved = {"momnorm": 6 * 4, "ns": 15 * 4, "vmean": 2 * 4,
                 "scalars": 14 * 4, "final": 9 * 4, "all": 46 * 4}
        print(f"\n=== MUON ELEMENTWISE-FUSION SIZING (upper bound), 4 ranks LNC=2 ===")
        print(f"{'region':<10} {'muon_ms':>9} {'delta_ms':>9} {'launches':>9} "
              f"{'ms/launch':>10} {'min':>7} {'max':>7}")
        print(f"{'baseline':<10} {base:>9.2f} {'--':>9} {'320':>9} {'--':>10} "
              f"{min(acc['baseline']):>7.2f} {max(acc['baseline']):>7.2f}")
        for name, _ in modes[1:]:
            m = statistics.median(acc[name])
            d = base - m
            n = saved[name]
            print(f"{name:<10} {m:>9.2f} {d:>+9.2f} {n:>9} {d/n:>10.3f} "
                  f"{min(acc[name]):>7.2f} {max(acc[name]):>7.2f}")
        print("SIZING_JSON " + json.dumps(dict(
            world_size=world_size, lnc=os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
            samples=acc)))

    if ddp:
        try:
            import torch.distributed as dist
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
