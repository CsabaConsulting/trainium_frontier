#!/usr/bin/env python3
"""T1.6 equivalence gate: is USE_TRANSPOSED_LINEAR (--transposed-linear) bit-exact?

T1.6 stores every `Linear` weight as [in, out] and computes `x @ weight` instead of
`F.linear(x, weight)`'s `x @ weight.T`. That is advertised as a pure storage/compute
reformulation with zero quality effect, so the standard is BIT-EXACTNESS -- 0 differing
elements, not "close" -- and this harness is what has to establish it before any speed
number from the flag is worth reading.

Run on CPU. That is not a fallback, it is the point: bf16 matmuls on device are
tile-order-dependent and a device run cannot separate "the reformulation is wrong" from
"the compiler re-associated a reduction". CPU fp32 removes that ambiguity entirely, and it
is the same standard the dead-code cleanup was held to (CPU fingerprint probe: identical
loss, all 80 param md5s, all 74 grad md5s).

Three things are checked, in the order they can fail:

  1. **Init.** Every transposed weight must equal the reference's TRANSPOSE, element for
     element. This is the one the implementation could most easily get wrong: `uniform_`
     fills in memory order and does NOT commute with a transpose, and `init_weights` also
     row-slices `c_in.weight[:n]`. transpose_linear_weights() is called at the END of
     init_weights precisely so this holds; if it were called at construction time this
     check would fail on every non-square matrix.
  2. **Forward + backward.** Identical loss (to the last bit) and identical gradients
     (compared against the reference's transpose) on identical synthetic batches.
  3. **Optimizer.** N full steps, comparing every parameter after every step. This is the
     check that catches the real trap: Muon's `ratio = max(1, rows/cols)**0.5` is a
     ONE-SIDED LR scale and is NOT transpose-invariant -- read off a transposed [640, 3200]
     tensor it collapses from sqrt(5) to 1.0, a silent 2.24x LR change on every non-square
     matrix. _muon_step carries a `transposed` group key to correct for exactly this, and
     step 1 of this loop is what proves the correction works. Deliberately run for several
     steps with a VARYING lr/momentum/wd schedule so the second-moment buffer (whose
     reduction axis also flips) is exercised past its zero init.

Usage:
    python research/scratch/transposed_linear_equiv.py --steps 6
    python research/scratch/transposed_linear_equiv.py --steps 6 --small   # fast smoke
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

import torch  # noqa: E402

import train as T  # noqa: E402


def build_arm(transposed: bool, device, seed: int, small: bool):
    """Build + init a model with USE_TRANSPOSED_LINEAR forced either way.

    The toggle is set BEFORE GPT(config)/init_weights() because init_weights is where the
    conversion happens. Same seed both arms, and the seed is set immediately before
    construction so the two arms draw the identical RNG stream -- the whole claim is that
    T1.6 does not perturb initialization, so the harness must not perturb it either.
    """
    T.USE_TRANSPOSED_LINEAR = transposed
    if small:
        # Shrink the hashed n-gram tables too, or the "small" arm still allocates the real
        # [1048576, mult] tables and is not small at all. Set before build_config() because
        # the table geometry is derived there.
        T.NGRAM_VE_TABLE_MULT = 1
    config = T.build_config()
    if small:
        # A small shape exercises every code path (both tall and wide Muon groups, the
        # gates, the n-gram projections) in seconds instead of minutes. The full locked
        # shape is the default because that is the one being submitted.
        # n_embd=256/n_head=2 is the smallest LEGAL shape: CausalSelfAttention asserts
        # n_embd >= 3 * VE_GATE_CHANNELS (96), and head_dim wants to stay at 128.
        config.n_layer, config.n_head, config.n_embd = 2, 2, 256
        config.sequence_len = 128
    torch.manual_seed(seed)
    model = T.GPT(config)
    model.init_weights()
    model.to(device)
    opt = model.setup_optimizer()
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return model, opt


def cmp_exact(a: torch.Tensor, b: torch.Tensor):
    """(n_differing, max_abs_diff, numel) in fp64 so the comparison itself cannot round."""
    x, y = a.detach().double().cpu(), b.detach().double().cpu()
    ne = int((x != y).sum())
    md = float((x - y).abs().max()) if x.numel() else 0.0
    return ne, md, x.numel()


def transposed_names(model) -> set[str]:
    """Fully-qualified '<module>.weight' names of every converted Linear."""
    out = set()
    for name, m in model.named_modules():
        if isinstance(m, T.Linear) and m.weight_transposed:
            out.add(f"{name}.weight" if name else "weight")
    return out


def compare_states(ref_model, tr_model, tnames, label, report):
    """Compare every parameter, transposing the ones T1.6 converted."""
    ref = dict(ref_model.named_parameters())
    tr = dict(tr_model.named_parameters())
    if set(ref) != set(tr):
        report(f"  {label}: PARAMETER NAME SETS DIFFER "
               f"(ref-only={sorted(set(ref) - set(tr))} tr-only={sorted(set(tr) - set(ref))})")
        return 1, 0.0
    bad, worst = 0, 0.0
    for name in sorted(ref):
        a, b = ref[name], tr[name]
        if name in tnames:
            if b.shape != (a.shape[1], a.shape[0]):
                report(f"  {label}: {name} shape {tuple(b.shape)} is not the transpose "
                       f"of {tuple(a.shape)}")
                bad += 1
                continue
            b = b.t()
        elif b.shape != a.shape:
            report(f"  {label}: {name} shape {tuple(b.shape)} != {tuple(a.shape)} "
                   f"(and it is not a converted Linear)")
            bad += 1
            continue
        ne, md, _ = cmp_exact(a, b)
        bad += ne
        worst = max(worst, md)
        if ne:
            report(f"  {label}: {name} {ne} differing, max|diff| {md:.3e}")
    return bad, worst


def compare_grads(ref_model, tr_model, tnames, label, report):
    ref = dict(ref_model.named_parameters())
    tr = dict(tr_model.named_parameters())
    bad, worst, missing = 0, 0.0, 0
    for name in sorted(ref):
        ga, gb = ref[name].grad, tr[name].grad
        if ga is None or gb is None:
            # A parameter with no gradient on BOTH sides is fine (frozen qk_shift_beta is
            # exactly this); one-sided is a real failure.
            if (ga is None) != (gb is None):
                report(f"  {label}: {name} grad present on only one side "
                       f"(ref={ga is not None} tr={gb is not None})")
                bad += 1
            else:
                missing += 1
            continue
        if name in tnames:
            gb = gb.t()
        ne, md, _ = cmp_exact(ga, gb)
        bad += ne
        worst = max(worst, md)
        if ne:
            report(f"  {label}: grad {name} {ne} differing, max|diff| {md:.3e}")
    return bad, worst, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--small", action="store_true", help="tiny shape for a fast smoke")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    failures = []

    def report(msg):
        print(msg)
        failures.append(msg)

    device = torch.device(args.device)
    T.COMPUTE_DTYPE = T.compute_dtype_for(device)
    print(f"device={device} compute_dtype={T.COMPUTE_DTYPE} steps={args.steps} small={args.small}")

    ref_model, ref_opt = build_arm(False, device, args.seed, args.small)
    tr_model, tr_opt = build_arm(True, device, args.seed, args.small)

    tnames = transposed_names(tr_model)
    print(f"converted Linears: {len(tnames)}")
    if not tnames:
        report("FAIL: the transposed arm converted ZERO Linears -- the flag did nothing, so "
               "every 'IDENTICAL' below would be vacuous")

    # A null-test guard, same idea as nki_custom_ops_dispatch_check.py: because this whole
    # harness expects bit-identity, a flag that silently did nothing would produce a clean
    # pass. Assert the shapes REALLY flipped for at least one non-square matrix.
    nonsq = [n for n in tnames
             if dict(tr_model.named_parameters())[n].shape[0]
             != dict(tr_model.named_parameters())[n].shape[1]]
    if not nonsq:
        report("FAIL: no non-square Linear was converted -- the Muon `ratio` trap this "
               "harness exists to catch would not be exercised")
    else:
        print(f"non-square converted Linears: {len(nonsq)} (e.g. {nonsq[0]} "
              f"{tuple(dict(tr_model.named_parameters())[nonsq[0]].shape)})")

    n_ref = sum(p.numel() for p in ref_model.parameters())
    n_tr = sum(p.numel() for p in tr_model.parameters())
    print(f"n_params: ref={n_ref} transposed={n_tr} "
          f"{'MATCH' if n_ref == n_tr else 'MISMATCH'}")
    if n_ref != n_tr:
        report(f"FAIL: parameter count changed {n_ref} -> {n_tr}")

    # Muon group structure: same number of groups, same kinds, same member counts. The
    # transposed arm's shapes differ by construction, so shapes are NOT compared -- what
    # matters is that no parameter fell out of Muon into a different group, or got merged
    # with a group it does not belong to.
    def group_sig(opt):
        return [(g["kind"], len(g["params"])) for g in opt.param_groups]
    ref_sig, tr_sig = group_sig(ref_opt), group_sig(tr_opt)
    print(f"optimizer groups: ref={len(ref_sig)} transposed={len(tr_sig)}")
    if sorted(ref_sig) != sorted(tr_sig):
        report(f"FAIL: optimizer group structure changed\n    ref={sorted(ref_sig)}\n"
               f"    tr ={sorted(tr_sig)}")
    tr_muon_transposed = [g.get("transposed", False) for g in tr_opt.param_groups
                          if g["kind"] == "muon"]
    print(f"muon groups flagged transposed: {sum(tr_muon_transposed)}/{len(tr_muon_transposed)}")
    if not all(tr_muon_transposed):
        report("FAIL: some Muon group was NOT flagged transposed -- its `ratio` LR scale "
               "would be read off the wrong orientation")

    # ---- 1. init ----
    bad, worst = compare_states(ref_model, tr_model, tnames, "init", report)
    print(f"[1] init weights: {'IDENTICAL' if bad == 0 else f'{bad} DIFFERING'} "
          f"(max|diff| {worst:.3e})")
    if bad:
        report(f"FAIL: init is not a pure transpose ({bad} differing elements)")

    # ---- 2/3. forward, backward, optimizer steps ----
    vocab = ref_model.config.vocab_size
    seq = ref_model.config.sequence_len
    for step in range(args.steps):
        torch.manual_seed(1000 + step)
        idx = torch.randint(0, vocab, (args.batch, seq), dtype=torch.int64, device=device)
        tgt = torch.randint(0, vocab, (args.batch, seq), dtype=torch.int64, device=device)

        # Vary the schedule so the second-moment buffer and the wd mask are exercised, not
        # just their zero-init state.
        lrm = 1.0 - 0.017 * step
        for opt in (ref_opt, tr_opt):
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lrm
                if g["kind"] == "muon":
                    g["momentum"] = 0.95 - 0.01 * step
                    g["weight_decay"] = T.WEIGHT_DECAY * (1.0 - min(1.0, step / 7.0))

        losses = {}
        for tag, model, opt in (("ref", ref_model, ref_opt), ("tr", tr_model, tr_opt)):
            # The flag is read by Linear.forward per-instance, but n-gram/eval paths read
            # module globals, so keep the global consistent with the arm being run.
            T.USE_TRANSPOSED_LINEAR = (tag == "tr")
            opt.zero_grad(set_to_none=True)
            loss = model(idx, tgt)
            loss.backward()
            losses[tag] = loss

        ne, md, _ = cmp_exact(losses["ref"], losses["tr"])
        lref = float(losses["ref"])
        ltr = float(losses["tr"])
        ok_loss = (ne == 0)
        print(f"[2] step {step} loss: ref={lref!r} tr={ltr!r} "
              f"{'IDENTICAL' if ok_loss else f'DIFFER by {md:.3e}'}")
        if not ok_loss:
            report(f"FAIL: step {step} loss differs by {md:.3e}")

        gbad, gworst, gmissing = compare_grads(ref_model, tr_model, tnames, f"step {step}", report)
        print(f"[2] step {step} grads: {'IDENTICAL' if gbad == 0 else f'{gbad} DIFFERING'} "
              f"(max|diff| {gworst:.3e}, {gmissing} params with no grad on either side)")
        if gbad:
            report(f"FAIL: step {step} gradients differ ({gbad} elements)")

        for opt in (ref_opt, tr_opt):
            opt.step()

        pbad, pworst = compare_states(ref_model, tr_model, tnames, f"step {step} post", report)
        print(f"[3] step {step} params after opt.step: "
              f"{'IDENTICAL' if pbad == 0 else f'{pbad} DIFFERING'} (max|diff| {pworst:.3e})")
        if pbad:
            report(f"FAIL: step {step} post-optimizer params differ ({pbad} elements)")

    print()
    if failures:
        print(f"RESULT: FAIL -- {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"RESULT: PASS -- USE_TRANSPOSED_LINEAR is BIT-EXACT over {args.steps} full "
          f"forward/backward/optimizer steps ({len(tnames)} Linears converted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
