#!/usr/bin/env python3
"""Real launch count and real wall-clock for train.py's USE_NKI_MUON fusion.

Two measurements on the same process, so nothing about the box or the build differs
between arms:

  1. LAUNCH COUNT -- a TorchDispatchMode census of the aten ops _muon_step actually
     dispatches with the flag off vs on, plus a count of the NKI kernel calls (which
     are not aten ops and so are invisible to the dispatcher; they are counted by
     wrapping the three kernel symbols). Total launches = aten non-view ops + NKI calls.

  2. WALL CLOCK -- the Muon-only window, arms interleaved across repeats so a slow
     patch of the box cannot land entirely on one arm. This box has shown sustained
     ~2x slowdowns, so interleaving is not optional.

Forward/backward are skipped and grads are synthetic: the quantity under test is
optimizer dispatch overhead, which does not care what the gradients contain.

Usage (4 ranks / LNC=2 to match the real locked shape):
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 \
      research/scratch/muon_nki_bench.py --steps 6 --warmup 3 --reps 3
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
from torch.utils._python_dispatch import TorchDispatchMode

import train as T

# identical classification to research/scratch/muon_dispatch_census.py so the numbers
# are directly comparable to the 320-launch baseline recorded there
VIEW_OPS = {
    "aten.view", "aten.reshape", "aten._unsafe_view", "aten.transpose",
    "aten.permute", "aten.t", "aten.expand", "aten.unbind", "aten.select",
    "aten.slice", "aten.detach", "aten.alias", "aten.squeeze", "aten.unsqueeze",
    "aten.as_strided", "aten.is_same_size", "aten.contiguous", "aten.numpy_T",
}


class Census(TorchDispatchMode):
    def __init__(self):
        self.n_real = 0
        self.n_view = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = ".".join(str(func).split(".")[:2])
        if name in VIEW_OPS:
            self.n_view += 1
        else:
            self.n_real += 1
        return func(*args, **(kwargs or {}))


class KernelCounter:
    """Counts NKI kernel invocations, which never reach the aten dispatcher."""

    NAMES = ("_muon_ns_poly", "_muon_ns_axpy", "_muon_update")

    def __init__(self):
        self.n = 0
        self._orig = {}

    def __enter__(self):
        for nm in self.NAMES:
            fn = getattr(T, nm)
            self._orig[nm] = fn

            def wrap(*a, _fn=fn, **k):
                self.n += 1
                return _fn(*a, **k)
            setattr(T, nm, wrap)
        return self

    def __exit__(self, *exc):
        for nm, fn in self._orig.items():
            setattr(T, nm, fn)
        return False


def build():
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
    for p in model.parameters():
        p.grad = torch.randn(p.shape, dtype=p.dtype).to(device) * 0.01
    return ddp, rank, world_size, device, opt


def run_arm(opt, device, use_nki, steps, warmup):
    muon = [g for g in opt.param_groups if g["kind"] == "muon"]
    adamw = [g for g in opt.param_groups if g["kind"] == "adamw"]
    T.USE_NKI_MUON = use_nki

    def muon_pass():
        for g in muon:
            opt._muon_step(g)

    for _ in range(warmup):
        for g in adamw:
            opt._adamw_step_foreach(g)
        muon_pass()
        T.synchronize(device)

    out = []
    for _ in range(steps):
        for g in adamw:
            opt._adamw_step_foreach(g)
        T.synchronize(device)
        t = time.monotonic()
        muon_pass()
        T.synchronize(device)
        out.append((time.monotonic() - t) * 1e3)
    T.USE_NKI_MUON = False
    return out


def census_arm(opt, use_nki):
    muon = [g for g in opt.param_groups if g["kind"] == "muon"]
    T.USE_NKI_MUON = use_nki
    c = Census()
    with KernelCounter() as kc, c:
        for g in muon:
            opt._muon_step(g)
    T.USE_NKI_MUON = False
    return c.n_real, c.n_view, kc.n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    ddp, rank, world_size, device, opt = build()

    # warm both arms first so neither census nor timing pays a compile
    run_arm(opt, device, False, 1, 1)
    run_arm(opt, device, True, 1, 2)

    cen = {}
    for name, flag in (("eager", False), ("nki", True)):
        cen[name] = census_arm(opt, flag)

    acc = {"eager": [], "nki": []}
    for rep in range(args.reps):
        for name, flag in (("eager", False), ("nki", True)):
            acc[name] += run_arm(opt, device, flag, args.steps, args.warmup)
            if rank == 0:
                print(f"[bn] rep{rep} {name:<6} muon_only "
                      f"{statistics.median(acc[name][-args.steps:]):7.2f}ms", flush=True)

    if rank == 0:
        e, n = statistics.median(acc["eager"]), statistics.median(acc["nki"])
        print("\n=== USE_NKI_MUON: real launch count and real Muon window ===")
        print(f"world_size={world_size} lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG')}")
        print(f"{'arm':<7} {'aten':>6} {'nki':>5} {'launches':>9} {'views':>7} "
              f"{'muon_ms':>9} {'min':>7} {'max':>7}")
        for name in ("eager", "nki"):
            ar, vw, nk = cen[name]
            ms = statistics.median(acc[name])
            print(f"{name:<7} {ar:>6} {nk:>5} {ar + nk:>9} {vw:>7} {ms:>9.2f} "
                  f"{min(acc[name]):>7.2f} {max(acc[name]):>7.2f}")
        le = cen["eager"][0] + cen["eager"][2]
        ln = cen["nki"][0] + cen["nki"][2]
        print(f"\nlaunches   {le} -> {ln}  ({le - ln} removed, {100*(le-ln)/le:.1f}%)")
        print(f"muon_ms    {e:.2f} -> {n:.2f}  ({e - n:+.2f} ms, {100*(e-n)/e:+.1f}%)")
        if le != ln:
            print(f"ms/launch  {(e - n)/(le - ln):.3f}")
        print("BENCH_JSON " + json.dumps(dict(
            world_size=world_size, lnc=os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
            census=cen, samples=acc)))

    if ddp:
        try:
            import torch.distributed as dist
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
