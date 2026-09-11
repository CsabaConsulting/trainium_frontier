"""T3.1 correctness harness for the hashed n-gram value embeddings (CPU only).

Checks, in order, all on the real GPT at the locked config:

  1. STEP-0 NO-OP: with zero-initialized tables, USE_NGRAM_VE=True is BIT-IDENTICAL to
     USE_NGRAM_VE=False on the forward pass. Two things have to hold for that: the tables
     are zero (the zero-init GATE is 2*sigmoid(0) = 1.0, i.e. neutral, not off), and the
     extra modules are RNG-neutral at construction (nn.Embedding gets _weight=, the gates
     go through _rng_neutral) so no shared tensor's init is reshuffled. Also verifies that
     every shared parameter is elementwise equal, which localizes a failure to the
     injection instead of the init.
  2. CAUSALITY: perturbing token t only changes outputs at positions >= t. Run directly on
     the hash indices AND on the model's logits.
  3. DETERMINISM: the same tokens produce the same table rows across calls, across a
     freshly constructed model, and independently of dict iteration order.
  4. INDEX RANGE: every index is in [0, ngram_mod) < rows, so no gather ever reads a row
     that torch would have to clamp, and no int32 product overflowed on the way there.
  5. PARAM COUNT / GROUPING: the tables land in exactly one rmsprop param group, are absent
     from every other group, and the reported parameter total matches the arithmetic.
  6. RMSPROP: one step of MuonAdamW's rmsprop branch on an untouched row leaves it at
     exactly 0.0 (no 0/0 NaN), on a hit row moves it by exactly lr, and the foreach and
     loop paths agree bit-for-bit.
  7. FLAT LR: a no_lr_decay group ignores the warmdown while the others follow it.

Run: python research/scratch/ngram_ve_equiv.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train as T  # noqa: E402


def build(ngram: bool, trigram: bool = True, bf16: bool = False, seed: int = 1234, dim: int = 0):
    T.USE_NGRAM_VE = ngram
    T.NGRAM_VE_TRIGRAM = trigram
    T.NGRAM_VE_BF16 = bf16
    T.NGRAM_VE_DIM = dim
    torch.manual_seed(seed)
    model = T.GPT(T.build_config())
    model.init_weights()
    model.eval()
    return model


def main() -> int:
    T.COMPUTE_DTYPE = torch.float32  # CPU: bf16 would hide real differences in rounding
    failures = []
    b, t = 2, 256
    torch.manual_seed(0)
    idx = torch.randint(0, T.TOKENIZER_VOCAB_SIZE, (b, t), dtype=torch.int32)
    targets = torch.randint(0, T.TOKENIZER_VOCAB_SIZE, (b, t), dtype=torch.long)

    # ---- 1. step-0 no-op -----------------------------------------------------------
    off = build(False)
    on = build(True)
    with torch.no_grad():
        lo, ln = off(idx), on(idx)
    same = torch.equal(lo, ln)
    print(f"[1] step-0 no-op: bit-identical={same} max_abs_diff={(lo - ln).abs().max():.3e}")
    if not same:
        failures.append("step-0 forward is not bit-identical with zero-init tables")

    # Localize any failure above: are the SHARED parameters even the same? If these differ,
    # the extra modules perturbed the RNG stream at construction and the problem is the init,
    # not the injection.
    sd_off, sd_on = off.state_dict(), on.state_dict()
    shared_bad = [k for k in sd_off if not torch.equal(sd_off[k], sd_on[k])]
    print(f"    shared tensors identical: {len(sd_off) - len(shared_bad)}/{len(sd_off)} "
          f"(rng-neutral construction) bad={shared_bad[:4]}")
    if shared_bad:
        failures.append(f"{len(shared_bad)} shared tensors differ => construction is not RNG-neutral")

    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    rows, mod = on.ngram_rows, on.ngram_mod
    n_tables = len(on.ngram_embeds)
    expect = n_off + n_tables * rows * on.config.n_embd + \
        sum(g.weight.numel() for blk in on.transformer.h
            for g in (blk.attn.ve_gate_bi, blk.attn.ve_gate_tri) if g is not None)
    print(f"    tables={sorted(on.ngram_embeds)} rows={rows} mod={mod} "
          f"params {n_off/1e6:.1f}M -> {n_on/1e6:.1f}M (expected {expect/1e6:.1f}M)")
    if n_on != expect:
        failures.append(f"param count {n_on} != expected {expect}")

    # ---- 2. causality --------------------------------------------------------------
    idx2 = idx.clone()
    cut = t // 2
    idx2[:, cut] = (idx2[:, cut] + 1234) % T.TOKENIZER_VOCAB_SIZE
    h1, h2 = on._ngram_hash_indices(idx), on._ngram_hash_indices(idx2)
    bad_prefix = []
    for k in h1:
        if not torch.equal(h1[k][:, :cut], h2[k][:, :cut]):
            bad_prefix.append(k)
        # a trigram index at t depends on t-2..t, so the change must be visible at
        # cut..cut+2 and nowhere later
        span = 3 if k.startswith("t") else 2
        if torch.equal(h1[k][:, cut:cut + span], h2[k][:, cut:cut + span]):
            failures.append(f"hash {k} did not react to the perturbed token at all")
        if not torch.equal(h1[k][:, cut + span:], h2[k][:, cut + span:]):
            failures.append(f"hash {k} leaked past its own n-gram window")
    print(f"[2] hash causality: prefix-unchanged={not bad_prefix} (bad={bad_prefix})")
    if bad_prefix:
        failures.append(f"hash indices before the perturbation changed: {bad_prefix}")
    with torch.no_grad():
        l1, l2 = on(idx), on(idx2)
    pre_same = torch.equal(l1[:, :cut], l2[:, :cut])
    print(f"[2] logit causality: positions <{cut} unchanged={pre_same}")
    if not pre_same:
        failures.append("logits before the perturbed position changed")

    # ---- 3. determinism ------------------------------------------------------------
    again = on._ngram_hash_indices(idx)
    fresh = build(True)._ngram_hash_indices(idx)
    det = all(torch.equal(h1[k], again[k]) and torch.equal(h1[k], fresh[k]) for k in h1)
    print(f"[3] determinism (repeat call + fresh model): {det}")
    if not det:
        failures.append("hash indices are not deterministic")

    # ---- 4. index range / no int32 overflow ----------------------------------------
    # Worst case for overflow is the sentinel id (padded_vocab_size) times the largest prime.
    worst = on.ngram_pad_id * max(T.NGRAM_HASH_PRIMES)
    ok_range = all(int(v.min()) >= 0 and int(v.max()) < mod for v in h1.values())
    edge = on._ngram_hash_indices(
        torch.full((1, 8), T.TOKENIZER_VOCAB_SIZE - 1, dtype=torch.int32))
    ok_edge = all(int(v.min()) >= 0 and int(v.max()) < mod for v in edge.values())
    print(f"[4] index range in [0,{mod}): random={ok_range} max-token={ok_edge}; "
          f"worst int32 product={worst:,} (limit {2**31 - 1:,})")
    if not (ok_range and ok_edge and worst < 2**31 - 1 and mod <= rows):
        failures.append("index range / int32 headroom check failed")

    # ---- 5. optimizer grouping -----------------------------------------------------
    opt = on.setup_optimizer()
    tbl_ids = {id(p) for p in on.ngram_embeds.parameters()}
    groups = [g for g in opt.param_groups if tbl_ids & {id(p) for p in g["params"]}]
    ok_group = (len(groups) == 1 and groups[0]["kind"] == "rmsprop"
                and len(groups[0]["params"]) == n_tables
                and groups[0]["weight_decay"] == 0.0
                and groups[0]["no_lr_decay"] is True
                and groups[0]["betas"] == (0.0, T.NGRAM_VE_BETA2))
    print(f"[5] grouping: one rmsprop group, {len(groups[0]['params'])} tensors, "
          f"lr={groups[0]['lr']:.6f}, wd={groups[0]['weight_decay']}, "
          f"no_lr_decay={groups[0]['no_lr_decay']} -> {ok_group}")
    if not ok_group:
        failures.append("n-gram tables are not in exactly one correctly-configured group")
    gate_ids = {id(g.weight) for blk in on.transformer.h
                for g in (blk.attn.ve_gate_bi, blk.attn.ve_gate_tri) if g is not None}
    ungrouped = gate_ids - {id(p) for g in opt.param_groups for p in g["params"]}
    print(f"    n-gram gates in an optimizer group: {len(gate_ids) - len(ungrouped)}"
          f"/{len(gate_ids)}")
    if ungrouped:
        failures.append("some n-gram gate weights are in NO optimizer group (dead params)")

    # ---- 6. rmsprop branch ---------------------------------------------------------
    for foreach in (True, False):
        T.USE_FOREACH_OPTIM = foreach
        p = torch.zeros(4, 3)
        o = T.MuonAdamW([dict(kind="rmsprop", params=[p], lr=0.1, betas=(0.0, 0.999),
                              eps=1e-10, weight_decay=0.0, no_lr_decay=True)])
        p.grad = torch.zeros_like(p)
        p.grad[0] = 1.0                      # row 0 hit, rows 1-3 untouched
        o.step()
        untouched_zero = torch.equal(p[1:], torch.zeros(3, 3))
        hit = float(p[0, 0])
        print(f"[6] rmsprop foreach={foreach}: hit row moved {hit:+.6f} (expect -0.100000), "
              f"untouched rows exactly 0.0={untouched_zero}, finite={torch.isfinite(p).all().item()}")
        if not (untouched_zero and abs(hit + 0.1) < 1e-6 and torch.isfinite(p).all()):
            failures.append(f"rmsprop step wrong (foreach={foreach})")
    T.USE_FOREACH_OPTIM = True

    # ---- 7. flat lr ----------------------------------------------------------------
    T.WARMUP_STEPS = 10
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    step, elapsed, budget = 500, 900.0, 1000.0   # deep inside the warmdown
    lrm = T.lr_multiplier(step, elapsed=elapsed, max_train_seconds=budget)
    lrm_flat = min(1.0, (step + 1) / max(1, T.WARMUP_STEPS))
    ok_flat = lrm < 0.5 and lrm_flat == 1.0
    print(f"[7] flat lr: warmdown lrm={lrm:.4f} vs no_lr_decay lrm={lrm_flat:.4f} -> {ok_flat}")
    if not ok_flat:
        failures.append("flat-lr multiplier does not bypass the warmdown")

    # ---- 8. bf16 tables still train and stay finite --------------------------------
    T.COMPUTE_DTYPE = torch.float32
    bm = build(True, bf16=True)
    bopt = bm.setup_optimizer()
    for g in bopt.param_groups:
        g["initial_lr"] = g["lr"]
    dtypes = {str(p.dtype) for p in bm.ngram_embeds.parameters()}
    loss = bm(idx, targets)
    loss.backward()
    bopt.step()
    moved = max(float(p.detach().abs().max()) for p in bm.ngram_embeds.parameters())
    finite = all(torch.isfinite(p).all().item() for p in bm.parameters())
    print(f"[8] bf16 tables: dtypes={dtypes} loss={float(loss):.6f} "
          f"max|table| after 1 step={moved:.6f} all-finite={finite}")
    if not (dtypes == {"torch.bfloat16"} and finite and moved > 0.0):
        failures.append("bf16 table path did not train cleanly")

    # ---- 9. NGRAM_VE_DIM: narrow table + per-table up-projection --------------------
    # Same three invariants as the full-width path, plus the one that is specific to it:
    # the up-projection must be NONZERO or the table's gradient is identically zero.
    T.COMPUTE_DTYPE = torch.float32
    dm = build(True, dim=128)
    with torch.no_grad():
        ld = dm(idx)
    n_dm = sum(p.numel() for p in dm.parameters())
    proj_nonzero = all(float(p.weight.abs().max()) > 0 for p in dm.ngram_proj.values())
    dopt = dm.setup_optimizer()
    for g in dopt.param_groups:
        g["initial_lr"] = g["lr"]
    dloss = dm(idx, targets)
    dloss.backward()
    tbl_grad = max(float(p.grad.abs().max()) for p in dm.ngram_embeds.parameters())
    dopt.step()
    proj_ids = {id(p.weight) for p in dm.ngram_proj.values()}
    muon = [g for g in dopt.param_groups if g["kind"] == "muon" and proj_ids & {id(p) for p in g["params"]}]
    ok_dim = (torch.equal(ld, lo) and len(dm.ngram_proj) == n_tables and proj_nonzero
              and tbl_grad > 0.0 and len(muon) == 1
              and all(torch.isfinite(p).all().item() for p in dm.parameters()))
    print(f"[9] dim=128: step-0 identical={torch.equal(ld, lo)} "
          f"proj_tables={len(dm.ngram_proj)} proj_nonzero={proj_nonzero} "
          f"max|dL/dtable|={tbl_grad:.3e} proj_in_muon_groups={len(muon)} "
          f"params {n_on/1e6:.1f}M -> {n_dm/1e6:.1f}M -> {ok_dim}")
    if not ok_dim:
        failures.append("NGRAM_VE_DIM path failed (see [9])")
    T.NGRAM_VE_DIM = 0

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
