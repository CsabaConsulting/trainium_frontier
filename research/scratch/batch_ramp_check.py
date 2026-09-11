"""T2.1 correctness harness for the batch-size ramp (CPU only).

The ramp's risk is not its arithmetic, it is that it introduces a per-step-changing quantity
into a loop that was carefully purged of them (USE_TENSOR_LR_SCALARS). So the checks are:

  1. OFF IS IDENTITY: with BATCH_RAMP=False, batch_ramp_grad_accum() returns the reference
     grad_accum and an lr scale of exactly 1.0 at every progress value -- so the default path
     is bit-identical to the locked baseline, not merely equivalent to it.
  2. SCHEDULE: stage boundaries land exactly on BATCH_RAMP_BOUNDS, stages are monotonically
     non-decreasing in batch size, and the lr scale is exactly (B/B_ref)**BATCH_RAMP_LR_EXP.
  3. LOSS SCALING EQUIVALENCE: accumulating N micro-batches with `loss * inv_gas_dev` (the
     device-scalar path the ramp uses) gives BIT-IDENTICAL gradients to the historical
     `loss / N` Python-float path, for every N the ramp can select.
  4. NO NEW DEVICE SCALARS: stepping the optimizer at three different lr scales reuses the SAME
     cached scalar tensors -- i.e. the lr scale rides the existing _device_scalar copy_ and adds
     no per-value allocation (and, on Neuron, no per-value graph constant).
  5. STAGE-B CONVENTION: the middle stage reproduces today's TOTAL_BATCH_SIZE exactly, so the
     ramp is a perturbation around the locked batch size rather than a different one.

Run: python research/scratch/batch_ramp_check.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train as T  # noqa: E402


def main() -> int:
    T.COMPUTE_DTYPE = torch.float32
    failures = []
    ref = 4  # grad_accum at the locked shape: TOTAL_BATCH_SIZE / (1 * 4 * 2048 * 4)

    # ---- 1. off is the identity ------------------------------------------------------
    T.BATCH_RAMP = False
    probes = [i / 20.0 for i in range(21)]
    off = [T.batch_ramp_grad_accum(p, ref) for p in probes]
    ok_off = all(g == ref and s == 1.0 for g, s in off)
    print(f"[1] BATCH_RAMP=False is the identity at {len(probes)} progress values: {ok_off}")
    if not ok_off:
        failures.append("BATCH_RAMP=False changed grad_accum or the lr scale")

    # ---- 2. schedule ----------------------------------------------------------------
    T.BATCH_RAMP = True
    lo, hi = T.BATCH_RAMP_BOUNDS
    gA, gB, gC = T.BATCH_RAMP_GRAD_ACCUM
    cases = [(0.0, gA), (lo - 1e-9, gA), (lo, gB), (hi - 1e-9, gB), (hi, gC), (1.0, gC)]
    bad = [(p, T.batch_ramp_grad_accum(p, ref)[0], want) for p, want in cases
           if T.batch_ramp_grad_accum(p, ref)[0] != want]
    scales = {g: T.batch_ramp_grad_accum(p, ref)[1] for p, g in [(0.0, gA), (lo, gB), (hi, gC)]}
    want_scales = {g: (g / ref) ** T.BATCH_RAMP_LR_EXP for g in (gA, gB, gC)}
    ok_sched = not bad and scales == want_scales and gA <= gB <= gC
    print(f"[2] boundaries {T.BATCH_RAMP_BOUNDS} grad_accum {T.BATCH_RAMP_GRAD_ACCUM} "
          f"lr_scales={ {k: round(v, 6) for k, v in scales.items()} } -> {ok_sched} bad={bad}")
    if not ok_sched:
        failures.append("stage schedule or lr scale is wrong")

    # ---- 3. loss-scaling equivalence -------------------------------------------------
    # Same model, same deterministic micro-batches, two spellings of the 1/N scaling.
    b, t = 2, 128
    torch.manual_seed(0)
    xs = [torch.randint(0, T.TOKENIZER_VOCAB_SIZE, (b, t), dtype=torch.int32) for _ in range(8)]
    ys = [torch.randint(0, T.TOKENIZER_VOCAB_SIZE, (b, t), dtype=torch.long) for _ in range(8)]

    def grads_for(n: int, use_device_scalar: bool) -> list[torch.Tensor]:
        torch.manual_seed(1234)
        m = T.GPT(T.build_config())
        m.init_weights()
        m.eval()
        m.zero_grad(set_to_none=True)
        inv = torch.tensor(1.0 / n, dtype=torch.float32)
        for i in range(n):
            loss = m(xs[i], ys[i])
            (loss * inv if use_device_scalar else loss / n).backward()
        return [p.grad.clone() for p in m.parameters() if p.grad is not None]

    for n in sorted(set(T.BATCH_RAMP_GRAD_ACCUM)):
        a, c = grads_for(n, False), grads_for(n, True)
        same = len(a) == len(c) and all(torch.equal(x, y) for x, y in zip(a, c))
        worst = max((x - y).abs().max().item() for x, y in zip(a, c)) if a else 0.0
        print(f"[3] grad_accum={n}: `loss/N` vs `loss*inv_tensor` bit-identical={same} "
              f"max_abs_diff={worst:.3e} ({len(a)} tensors)")
        if not same:
            failures.append(f"device-scalar loss scaling is not bit-identical at N={n}")

    # ---- 4. no new device scalars per lr value ---------------------------------------
    torch.manual_seed(1234)
    m = T.GPT(T.build_config())
    m.init_weights()
    opt = m.setup_optimizer()
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    loss = m(xs[0], ys[0])
    loss.backward()
    ids_seen = []
    for scale in [(gA / ref) ** 0.5, 1.0, (gC / ref) ** 0.5]:
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * scale
        opt.step()
        ids_seen.append({(k, key, id(t0), id(t1))
                         for k, d in opt._scalar_cache.items() for key, (t0, t1) in d.items()})
    ok_ids = len(ids_seen[0]) > 0 and all(s == ids_seen[0] for s in ids_seen[1:])
    print(f"[4] three different lr scales reuse the same {len(ids_seen[0])} cached scalar "
          f"tensors: {ok_ids}")
    if not ok_ids:
        failures.append("a new lr scale allocated new device scalars (would recompile on Neuron)")

    # ---- 5. stage B is today's batch size --------------------------------------------
    micro = T.DEVICE_BATCH_SIZE * T.PACK_FACTOR * T.SEQ_LEN * 4  # 4 ranks at the locked shape
    ok_b = gB * micro == T.TOTAL_BATCH_SIZE
    print(f"[5] stage B tokens={gB * micro:,} vs TOTAL_BATCH_SIZE={T.TOTAL_BATCH_SIZE:,} -> {ok_b}"
          f"  (A={gA * micro:,} C={gC * micro:,})")
    if not ok_b:
        failures.append("stage B does not reproduce TOTAL_BATCH_SIZE")

    T.BATCH_RAMP = False
    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
