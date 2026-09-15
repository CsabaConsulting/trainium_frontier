"""CPU sanity check for --muon-lr-fp32 / USE_MUON_LR_FP32 (see the _muon_lr_g docstring).

Two things to prove, per the task:
  (1) flag OFF reproduces the pre-existing (bf16-quantised-lr) behaviour byte for byte -- no
      regression against the original 2-argument `_muon_lr_g`.
  (2) flag ON changes the `w` update by a small nonzero amount consistent with recovering
      ~0.4% of lr precision -- i.e. it is not a no-op.

Modelled on research/scratch/sched_tricks_probe.py's approach (intercept `_muon_math`, run the
real GPT at the locked config on CPU for a couple of steps, diff every operand). COMPUTE_DTYPE is
forced to bfloat16 even though the device is CPU: that is the one knob controlling whether the
Newton-Schulz iterate `g` handed to `_muon_lr_g` is bf16 (the locked Neuron config) or fp32 (what
CPU would otherwise give by compute_dtype_for) -- the bug only exists in the bf16-`g` case, so the
CPU check has to force it to be reproduced at all.

Run: python research/scratch/muon_lr_fp32_equiv.py
"""
import os
import sys

os.environ.setdefault("NEURON_COMPILE_CACHE_DIR", "/tmp/neuron_cache")
sys.argv = ["probe"]

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import train as T

STEPS = 3
SEED = 1234


# --- Part 1: unit-level check directly on _muon_lr_g ------------------------------------

def unit_check():
    print("=" * 78)
    print("PART 1: unit check of _muon_lr_g")
    print("=" * 78)
    torch.manual_seed(0)
    g = (torch.randn(64, 96) * 0.7).to(torch.bfloat16)
    lr = torch.tensor(0.0123457, dtype=torch.float32)  # 0-dim, fp32 -- the locked-default shape

    def _original_muon_lr_g(lr, g):
        """The pre-existing 2-arg implementation, verbatim, as the regression oracle."""
        if isinstance(lr, torch.Tensor) and lr.dim() > 0 and lr.dtype != g.dtype:
            lr = lr.to(g.dtype)
        return lr * g

    ref_off = _original_muon_lr_g(lr, g)
    new_off = T._muon_lr_g(lr, g, False)
    off_identical = torch.equal(ref_off, new_off)
    print(f"flag OFF vs original _muon_lr_g: torch.equal={off_identical} "
          f"dtype={new_off.dtype}")
    assert off_identical, "REGRESSION: flag-off path no longer matches the original function"

    new_on = T._muon_lr_g(lr, g, True)
    print(f"flag ON: dtype={new_on.dtype} (expect float32, was {ref_off.dtype})")
    assert new_on.dtype == torch.float32

    # Exact full-precision product, as the ceiling _muon_lr_g(..., True) should hit exactly.
    full = lr * g.float()
    on_matches_full = torch.equal(new_on, full)
    print(f"flag ON == lr(fp32) * g.float() exactly: {on_matches_full}")
    assert on_matches_full

    diff = (new_on.double() - ref_off.double()).abs()
    rel = (diff / new_on.double().abs().clamp_min(1e-12))
    # bf16 has a 7-bit mantissa => worst-case relative rounding error of a value is 2^-8 (0.39%),
    # and lr getting rounded to bf16 before the multiply is exactly one such rounding.
    print(f"max abs diff (ON - OFF): {diff.max().item():.6e}")
    print(f"max relative diff: {rel.max().item()*100:.4f}%  (bf16 ULP ceiling: {100*2**-8:.4f}%)")
    assert diff.max().item() > 0, "flag ON produced a no-op -- it did not change anything"
    assert rel.max().item() <= 2 ** -7, "difference is larger than a plausible bf16-rounding fix"
    print("PART 1: PASS -- flag OFF is byte-identical to the original, flag ON differs by a "
          "small nonzero amount within the bf16-rounding envelope.\n")


# --- Part 2: integration check through the real _muon_math on the real GPT -----------------

_ORIG_MATH = T._muon_math


def instrument(sink):
    def wrapped(w, g, momentum_buffer, second, mom, one_mom, lr, lrwd,
                tall, ns_steps, red_dim, red_dim_size, one_beta2, to_bf16, lr_fp32=False):
        key = tuple(g.shape)
        rec = {"mom": mom.clone() if isinstance(mom, torch.Tensor) else mom,
               "one_mom": one_mom.clone() if isinstance(one_mom, torch.Tensor) else one_mom,
               "lr": lr.clone() if isinstance(lr, torch.Tensor) else lr,
               "lrwd": lrwd.clone() if isinstance(lrwd, torch.Tensor) else lrwd,
               "one_beta2": one_beta2, "g_in": g.clone(), "w_in": w.clone(),
               "mbuf_in": momentum_buffer.clone(), "to_bf16": to_bf16, "lr_fp32": lr_fp32}
        out = _ORIG_MATH(w, g, momentum_buffer, second, mom, one_mom, lr, lrwd,
                          tall, ns_steps, red_dim, red_dim_size, one_beta2, to_bf16, lr_fp32)
        rec["w_out"], rec["mbuf_out"], rec["second_out"] = (t.clone() for t in out)
        sink.setdefault(key, []).append(rec)
        return out
    return wrapped


def run(lr_fp32_flag):
    torch.manual_seed(SEED)
    T.COMPUTE_DTYPE = torch.bfloat16  # force the bf16-g case even though device is CPU
    T.USE_COMPILE_MUON = True
    T.USE_TENSOR_LR_SCALARS = True
    T.USE_MUON_LR_FP32 = lr_fp32_flag
    model = T.GPT(T.build_config())
    sink = {}
    # _MUON_COMPILED caches the function object per (shape, tall, ns_steps, red_dim); without
    # clearing it a second run would keep calling the FIRST run's instrumented wrapper.
    T._MUON_COMPILED.clear()
    T._muon_math = instrument(sink)
    try:
        opt = model.setup_optimizer()
        named = dict(model.named_parameters())
        for s in range(STEPS):
            gg = torch.Generator().manual_seed(9000 + s)
            for n, p in named.items():
                p.grad = (torch.randn(p.shape, generator=gg, dtype=torch.float32) * 0.01).to(p.dtype)
            prog = s / max(1, STEPS - 1)
            for grp in opt.param_groups:
                grp["lr"] = grp["initial_lr"] * (1.0 - 0.3 * prog)
                if grp["kind"] == "muon":
                    grp["momentum"] = T.muon_momentum(s, prog)
            opt.step()
    finally:
        T._muon_math = _ORIG_MATH
        T._MUON_COMPILED.clear()
    return sink


def integration_check():
    print("=" * 78)
    print("PART 2: integration check through the real GPT / _muon_math, CPU, bf16-g forced")
    print("=" * 78)
    off = run(False)
    on = run(True)
    assert sorted(off) == sorted(on)

    total_w_diffs = 0
    max_rel_to_update = 0.0
    bad_side_channel = 0
    for key in sorted(off):
        for i, (a, b) in enumerate(zip(off[key], on[key])):
            # Every scalar operand, and every tensor computed UPSTREAM of the lr*g term, must
            # be identical -- the flag must not perturb anything but the final w update. `w_in`
            # is only checked at step 0 (identical fresh params): from step 1 on, `w_in` is last
            # step's `w_out`, which the flag has already (correctly) perturbed by design, so it
            # is expected -- not a bug -- for it to differ downstream.
            names = ("mom", "one_mom", "lr", "lrwd", "one_beta2", "g_in", "mbuf_in")
            if i == 0:
                names = names + ("w_in",)
            for name in names:
                va, vb = a[name], b[name]
                same = torch.equal(va, vb) if isinstance(va, torch.Tensor) else va == vb
                if not same:
                    bad_side_channel += 1
                    print(f"{key} step{i} {name}: DIFFERS between OFF/ON (should be identical input)")
            # momentum_buffer / second are computed upstream of the lr*g term and must be
            # bit-identical regardless of the flag, every step (they never read w).
            for name in ("mbuf_out", "second_out"):
                if not torch.equal(a[name], b[name]):
                    bad_side_channel += 1
                    print(f"{key} step{i} {name}: DIFFERS between OFF/ON (should be identical)")
            d = (a["w_out"].double() - b["w_out"].double()).abs()
            md = d.max().item()
            # The meaningful scale to compare against is the per-step UPDATE (w_in - w_out), not
            # w_out itself -- w is dominated by the pre-existing weight value the tiny update
            # rides on top of, so a diff-over-w_out ratio is dominated by noise near w's zero
            # crossings and means nothing.
            update_mag = (a["w_in"].double() - a["w_out"].double()).abs()
            rel_to_update = (d / update_mag.clamp_min(1e-12)).median().item()
            max_rel_to_update = max(max_rel_to_update, rel_to_update)
            if md > 0:
                total_w_diffs += 1
            print(f"{key} step{i}: w_out max|diff|={md:.3e}  "
                  f"median(diff / per-step update size)={rel_to_update*100:.4f}%")

    print(f"\nside-channel mismatches (should be 0): {bad_side_channel}")
    print(f"steps with a nonzero w_out difference (should be > 0): {total_w_diffs}")
    print(f"overall max median(diff/update) across steps (should be > 0 and on the order of the "
          f"bf16 ULP, {100*2**-8:.3f}%-{100*2**-6:.3f}%): {max_rel_to_update*100:.4f}%")
    assert bad_side_channel == 0, "flag perturbed something other than the final w update"
    assert total_w_diffs > 0, "flag ON is a no-op end to end -- it never changed w"
    assert 0.0 < max_rel_to_update <= 0.05, \
        "w_out difference (relative to the per-step update size) is outside the plausible " \
        "bf16-rounding range"
    print("PART 2: PASS -- flag only changes w_out, by a small nonzero amount consistent with "
          "recovering bf16-rounded lr precision, and nothing else.\n")


if __name__ == "__main__":
    unit_check()
    integration_check()
    print("ALL CHECKS PASSED")
