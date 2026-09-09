"""Bit-equality harness for T1.2 (train.py's --grad-bucket / build_flat_grad_buffer /
sync_gradients_flat / clip_grad_norm_flat_).

The claim being tested: replacing "one dist.all_reduce per parameter, then
torch.nn.utils.clip_grad_norm_ over model.parameters()" with "one flat fp32 buffer,
one dist.all_reduce, one norm+mul clip on the buffer" must not change a single bit of
the resulting (post-clip, post-all_reduce) gradients. This is exercised against the
REAL collective (torch.distributed, gloo backend, CPU, world_size=2 local processes)
rather than assumed from "elementwise ops don't care about buffer packing" -- that
assumption is exactly the kind of thing this competition's notebook has been burned by
before (see the NKI-Muon and foreach-optimizer equivalence harnesses), so it gets
checked directly instead of trusted.

Two independent checks:
  1. Synthetic tensors of the real model's actual parameter shapes (locked config),
     random per-rank values, world_size=2. Old path: per-tensor dist.all_reduce(AVG).
     New path: concatenate into one flat buffer, one dist.all_reduce(AVG), unflatten.
     Asserts torch.equal on every tensor.
  2. Same, but with clip_grad_norm_ in front (old: torch.nn.utils.clip_grad_norm_ over
     the per-rank LOCAL tensors before the all_reduce; new: clip_grad_norm_flat_ over
     the per-rank LOCAL flat buffer before the all_reduce) -- matches the training
     loop's actual clip-then-sync order in both the old and new code paths.

Run: python research/scratch/grad_bucket_equiv.py
"""
import os
import sys

os.environ.setdefault("NEURON_COMPILE_CACHE_DIR", "/tmp/neuron_cache")
sys.argv = ["equiv"]  # keep train.py's argparse out of this

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import train as T

WORLD_SIZE = 2
GRAD_CLIP = 1.0


def real_param_shapes():
    """Shapes from the real locked-config model -- not toy shapes."""
    torch.manual_seed(0)
    cfg = T.build_config()
    m = T.GPT(cfg)
    shapes = [p.shape for p in m.parameters() if p.requires_grad]
    del m
    return shapes


def make_local_grads(shapes, rank, seed_base):
    g = torch.Generator().manual_seed(seed_base + rank)
    return [torch.randn(s, generator=g, dtype=torch.float32) * 0.01 for s in shapes]


def old_path(grads, clip):
    """Per-tensor clip_grad_norm_ (matching torch.nn.utils.clip_grad_norm_'s math
    directly, since these are plain tensors, not nn.Parameters with .grad) then
    per-tensor dist.all_reduce(AVG). Returns the resulting list of tensors."""
    local = [g.clone() for g in grads]
    if clip > 0:
        total_norm = torch.sqrt(sum((t.float() ** 2).sum() for t in local))
        clip_coef = torch.clamp(clip / (total_norm + 1e-6), max=1.0)
        for t in local:
            t.mul_(clip_coef)
    for t in local:
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return local


def new_path(grads, clip):
    """Flatten into one buffer, clip_grad_norm_flat_-equivalent, one all_reduce,
    unflatten. Mirrors build_flat_grad_buffer/clip_grad_norm_flat_/sync_gradients_flat
    exactly (same formula, same order), just against plain tensors instead of an
    nn.Module's .grad attributes (spawn's per-process model would need real device
    setup; the flattening arithmetic under test does not depend on that)."""
    flat = torch.cat([g.reshape(-1).clone() for g in grads])
    if clip > 0:
        T.clip_grad_norm_flat_(flat, clip)
    T.sync_gradients_flat(flat)
    out = []
    offset = 0
    for g in grads:
        n = g.numel()
        out.append(flat[offset:offset + n].view(g.shape).clone())
        offset += n
    return out


def worker(rank, world_size, shapes, results_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    for check_name, clip in (("no_clip", 0.0), ("with_clip", GRAD_CLIP)):
        grads = make_local_grads(shapes, rank, seed_base=1000 if clip == 0 else 2000)
        old = old_path(grads, clip)
        # new_path calls dist.all_reduce again -- must feed it the SAME local grads,
        # so rebuild them identically (deterministic seed) rather than reusing `old`.
        grads2 = make_local_grads(shapes, rank, seed_base=1000 if clip == 0 else 2000)
        new = new_path(grads2, clip)
        max_abs = 0.0
        all_equal = True
        for a, b in zip(old, new):
            all_equal = all_equal and torch.equal(a, b)
            max_abs = max(max_abs, (a - b).abs().max().item())
        torch.save({"all_equal": all_equal, "max_abs": max_abs}, os.path.join(results_dir, f"{check_name}_rank{rank}.pt"))

    dist.destroy_process_group()


def main():
    shapes = real_param_shapes()
    print(f"model params: {len(shapes)} tensors, shapes from build_config()={T.build_config()}")
    results_dir = "/tmp/grad_bucket_equiv_results"
    os.makedirs(results_dir, exist_ok=True)
    for f in os.listdir(results_dir):
        os.remove(os.path.join(results_dir, f))

    mp.spawn(worker, args=(WORLD_SIZE, shapes, results_dir), nprocs=WORLD_SIZE, join=True)

    fails = []
    for check_name in ("no_clip", "with_clip"):
        for rank in range(WORLD_SIZE):
            r = torch.load(os.path.join(results_dir, f"{check_name}_rank{rank}.pt"))
            tag = "PASS" if r["all_equal"] else "FAIL"
            print(f"[{tag}] {check_name} rank={rank} all_equal={r['all_equal']} max_abs_diff={r['max_abs']:.3e}")
            if not r["all_equal"]:
                fails.append((check_name, rank))

    print("\n" + ("ALL CHECKS PASSED -- flat-buffer path is bit-identical to per-tensor path"
                  if not fails else f"FAILURES: {fails}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
