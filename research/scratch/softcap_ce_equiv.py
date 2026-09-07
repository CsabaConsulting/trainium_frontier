"""Numerical-equivalence sweep for the fused logit-softcap + cross-entropy NKI kernel.

Compares the kernel (forward AND backward) against the eager chain it replaces,

    y = LOGIT_SOFTCAP * torch.tanh(z / LOGIT_SOFTCAP)
    F.cross_entropy(y, targets, reduction="none")

across input scales and the degenerate cases that actually stress the algorithm:
small logits near zero, logits deep in tanh saturation, all-equal logits, and
extremely-confident-correct / extremely-confident-wrong predictions. Reference is
computed on CPU, so only the two NKI kernels generate a NEFF.

The kernel under test is imported straight out of train.py, so this harness can
never drift from what actually ships.

Also asserts the claim the kernel's design rests on -- that y is bounded to (-S, S)
by tanh, hence the shifted exponent y_j - y_c is bounded to (-2S, 2S) and can neither
overflow nor underflow fp32, hence no row-max-subtraction pass is needed -- by
measuring the real exp range on every case rather than taking it on faith.

Judges the kernel against an fp64 ground truth as well as against the eager fp32
chain, because on ill-conditioned rows (confidently-correct tokens, where the loss
is a ~1e-3 difference of two ~15s) the eager fp32 chain is itself visibly wrong and
is not a valid yardstick.
"""
import os
import sys

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 1")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F

import train as T_

SoftcapCE = T_._SoftcapCrossEntropyNKI
S = T_.LOGIT_SOFTCAP

BF16_ULP = 2.0 ** -8   # bf16 relative resolution
FP32_ULP = 2.0 ** -24  # fp32 relative resolution

WORST = {"fwd_abs": 0.0, "fwd_rel": 0.0, "bwd_abs": 0.0, "bwd_rel": 0.0,
         "fwd_bf16_ulp": 0.0, "fwd_fp32_ulp": 0.0,
         "bwd_bf16_ulp": 0.0, "bwd_fp32_ulp": 0.0,
         "fwd_vs_fp64": 0.0, "eager_vs_fp64": 0.0}


def eager_ref(z, targets):
    y = S * torch.tanh(z / S)
    return F.cross_entropy(y, targets, reduction="none")


def fp64_ref(z, targets):
    """Ground truth. Everything -- tanh, logsumexp, the gather -- in fp64, and the
    loss written as the cancellation-free log(sum_j exp(y_j - y_c)) so the reference
    itself is not the thing under suspicion."""
    y = S * torch.tanh(z.double() / S)
    yc = y.gather(1, targets.reshape(-1, 1).long()).reshape(-1, 1)
    return torch.log(torch.exp(y - yc).sum(dim=1))


def ulps(got, ref, unit):
    """Deviation in ULPs of `ref`, floored at 1e-3 of the tensor max so that
    near-zero reference elements cannot manufacture astronomical ratios."""
    g32, r32 = got.float(), ref.float()
    floor = max(r32.abs().max().item(), 1e-30) * 1e-3
    scale = torch.clamp(r32.abs(), min=floor) * unit
    return ((g32 - r32).abs() / scale).max().item()


def check(name, z, targets, g=None):
    n, v = z.shape
    if g is None:
        torch.manual_seed(99)
        g = (torch.randn(n) * 0.5).float()

    zr = z.clone().requires_grad_(True)
    lr = eager_ref(zr, targets)
    lr.backward(g)
    ref_loss, ref_dz = lr.detach(), zr.grad.detach()

    # The stability claim the design rests on, measured not assumed: y is bounded to
    # (-S, S) by tanh, so the shifted exponent y_j - y_c is bounded to (-2S, 2S) with
    # NO row-max pass, every exp term lands in (e^-30, e^30), and their sum over V
    # cannot leave fp32 -- while d >= 1 always, since the j == c term is exp(0).
    y = (S * torch.tanh(z / S))
    yc = y.gather(1, targets.reshape(-1, 1).long())
    e = torch.exp((y - yc).double())
    dsum = e.sum(dim=1)

    zd = z.to("neuron").detach().clone().requires_grad_(True)
    tgt_d = targets.reshape(-1, 1).to(torch.float32).to("neuron")
    got_loss = SoftcapCE.apply(zd, tgt_d)
    got_loss.backward(g.to("neuron"))
    got_loss_c, got_dz_c = got_loss.detach().cpu(), zd.grad.detach().cpu()

    print(f"{name}  [{n}x{v}]")
    print(f"   y in [{y.min():+.4f}, {y.max():+.4f}]  exp(y-y_c) in "
          f"[{e.min():.3e}, {e.max():.3e}]  sum_j in [{dsum.min():.3e}, {dsum.max():.3e}]"
          f"  (fp32 min normal 1.18e-38, max 3.40e+38)")
    e32 = e.float()
    assert torch.isfinite(e32).all() and (e32 > 0).all(), "exp(y-y_c) left fp32 range"
    assert torch.isfinite(dsum.float()).all(), "sum_j exp(y-y_c) overflowed fp32"
    assert dsum.min() >= 1.0 - 1e-9, f"d < 1, so log(d) < 0: {dsum.min()}"

    for tag, got, ref in (("fwd", got_loss_c, ref_loss), ("bwd", got_dz_c, ref_dz)):
        d = (got.float() - ref.float()).abs()
        denom = max(ref.float().abs().max().item(), 1e-30)
        rel = d.max().item() / denom
        ub, uf = ulps(got, ref, BF16_ULP), ulps(got, ref, FP32_ULP)
        # fp64 so the metric itself cannot report cos > 1 on long vectors
        a, b = got.double().flatten(), ref.double().flatten()
        cs = (a @ b / (a.norm() * b.norm() + 1e-300)).item()
        print(f"   {tag}: max_abs={d.max().item():.3e}  rel_to_max={rel:.3e}  "
              f"bf16_ulp={ub:.4f}  fp32_ulp={uf:8.1f}  cos={cs:.9f}")
        WORST[f"{tag}_abs"] = max(WORST[f"{tag}_abs"], d.max().item())
        WORST[f"{tag}_rel"] = max(WORST[f"{tag}_rel"], rel)
        WORST[f"{tag}_bf16_ulp"] = max(WORST[f"{tag}_bf16_ulp"], ub)
        WORST[f"{tag}_fp32_ulp"] = max(WORST[f"{tag}_fp32_ulp"], uf)

    # Who is actually closer to the truth? The eager fp32 chain is the thing being
    # replaced, not the ground truth, and on ill-conditioned rows it has real error
    # of its own. Judge both against fp64.
    t64 = fp64_ref(z, targets)
    e_kernel = (got_loss_c.double() - t64).abs().max().item()
    e_eager = (ref_loss.double() - t64).abs().max().item()
    print(f"   fwd vs fp64 truth: kernel={e_kernel:.3e}  eager_fp32={e_eager:.3e}  "
          f"(loss range [{t64.min():.3e}, {t64.max():.3e}])")
    WORST["fwd_vs_fp64"] = max(WORST["fwd_vs_fp64"], e_kernel)
    WORST["eager_vs_fp64"] = max(WORST["eager_vs_fp64"], e_eager)

    # Structural: sum_j dz_j is NOT ~0 in general (the sech^2 factor is inside the
    # sum, so sum_j (p_j - onehot_j) * sech^2_j != 0 unless sech^2 is ~constant).
    # The meaningful check is that the kernel's row sums track the reference's.
    rs_got = got_dz_c.double().sum(dim=1)
    rs_ref = ref_dz.double().sum(dim=1)
    print(f"   row-sum agreement: max|sum dz - sum ref_dz|={(rs_got - rs_ref).abs().max():.3e}"
          f"  (ref row sums span [{rs_ref.min():+.3e}, {rs_ref.max():+.3e}])")


def randt(n, v, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, v, (n,), generator=gen)


# ---------------------------------------------------------------- fast iteration
for N, V in ((256, 512), (512, 2048)):
    for seed, scale in ((0, 1.0), (1, 6.0), (7, 30.0)):
        torch.manual_seed(seed)
        check(f"random scale={scale}", (torch.randn(N, V) * scale).float(), randt(N, V, seed))

N, V = 256, 8192
torch.manual_seed(11)
check("tiny logits (x1e-3, softcap ~linear)", (torch.randn(N, V) * 1e-3).float(), randt(N, V, 11))
torch.manual_seed(12)
check("deep tanh saturation (x1e3)", (torch.randn(N, V) * 1e3).float(), randt(N, V, 12))
check("all-equal logits (uniform softmax)", torch.full((N, V), 0.7), randt(N, V, 13))
check("all-zero logits", torch.zeros(N, V), randt(N, V, 14))

tgt = randt(N, V, 15)
zc = torch.zeros(N, V)
zc[torch.arange(N), tgt] = 500.0            # far past tanh saturation
check("extremely-confident-CORRECT", zc, tgt)

zw = torch.zeros(N, V)
wrong = (tgt + 1) % V
zw[torch.arange(N), wrong] = 500.0
zw[torch.arange(N), tgt] = -500.0
check("extremely-confident-WRONG", zw, tgt)

torch.manual_seed(16)
zm = (torch.randn(N, V) * torch.logspace(-3, 3, N)[:, None]).float()
check("per-row scale sweep 1e-3..1e3", zm, randt(N, V, 16))

torch.manual_seed(17)
check("grad all zeros", (torch.randn(N, V) * 3).float(), randt(N, V, 17),
      g=torch.zeros(N))

# realistic: logits from a small random unembedding, the shape the kernel really sees
torch.manual_seed(18)
h = torch.randn(N, 768) * 0.6
w = torch.randn(V, 768) * 0.001
check("realistic lm_head logits", (h @ w.T).float(), randt(N, V, 18))

# ------------------------------------------------------------------- real shape
N, V = 4096, 8192
torch.manual_seed(20)
h = torch.randn(N, 768) * 0.6
w = torch.randn(V, 768) * 0.001
check("REAL SHAPE realistic lm_head logits", (h @ w.T).float(), randt(N, V, 20))
torch.manual_seed(21)
check("REAL SHAPE random scale=6", (torch.randn(N, V) * 6).float(), randt(N, V, 21))

# ------------------------------------- wrapper: reduction semantics must be intact
print("\nwrapper reduction check (softcap_cross_entropy, mean and none)")
N, V, T = 256, 8192, 128
torch.manual_seed(30)
logits2 = (torch.randn(2, N // 2, V) * 4).float()
targets2 = randt(N, V, 30).reshape(2, N // 2)
ref_mean = eager_ref(logits2.reshape(-1, V), targets2.reshape(-1)).mean()
ref_none = eager_ref(logits2.reshape(-1, V), targets2.reshape(-1)).view_as(targets2)
T_.USE_NKI_SOFTCAP_CE = True
ld, td = logits2.to("neuron"), targets2.to("neuron")
for red, ref in (("mean", ref_mean), ("none", ref_none)):
    got = T_.softcap_cross_entropy(ld, td, red).cpu()
    assert got.shape == ref.shape, f"{red}: shape {got.shape} != {ref.shape}"
    print(f"   reduction={red:5s} shape={tuple(got.shape)} "
          f"max_abs={(got - ref).abs().max().item():.3e}")
T_.USE_NKI_SOFTCAP_CE = False
# Knob OFF must reproduce the ORIGINAL chain bit for bit. That chain is
# F.cross_entropy(capped, t, reduction=red) -- NOT reduction="none" then .mean(),
# which differs in summation order, so ref_mean above is the wrong thing to compare to.
for red in ("mean", "none"):
    capped = S * torch.tanh(logits2 / S)
    orig = F.cross_entropy(capped.reshape(-1, V), targets2.reshape(-1), reduction=red)
    if red == "none":
        orig = orig.view_as(targets2)
    got_off = T_.softcap_cross_entropy(logits2, targets2, red)
    same = bool(torch.equal(got_off, orig))
    print(f"   knob OFF, reduction={red:5s}: bit-identical to original chain: {same}")
    assert same, f"eager fallback drifted from the original chain for reduction={red}"

print("\nWORST CASE ACROSS ALL CASES")
for k in ("fwd", "bwd"):
    print(f"   {k}: max_abs={WORST[k + '_abs']:.3e}  max_rel_to_max={WORST[k + '_rel']:.3e}  "
          f"max_bf16_ulp={WORST[k + '_bf16_ulp']:.4f}  max_fp32_ulp={WORST[k + '_fp32_ulp']:.1f}")
print(f"   vs fp64 truth: kernel={WORST['fwd_vs_fp64']:.3e}  "
      f"eager_fp32={WORST['eager_vs_fp64']:.3e}")
print("\nDONE")
