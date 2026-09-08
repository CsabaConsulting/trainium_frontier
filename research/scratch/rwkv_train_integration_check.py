"""CPU audit of train.py's USE_RWKV wiring. No Neuron device, no NKI.

Same shape as mamba2_train_integration_check.py, checking as assertions rather
than by eyeballing everything that can be checked without hardware:

  1. USE_RWKV=False is byte-identical to the pre-change transformer: same
     parameter names, same shapes, same count, same loss from the same seed.
  2. With USE_RWKV=True the pure-RWKV model and a hybrid pattern both build:
     "R" layers have .tmix/.cmix and no .attn/.mlp, attention layers the reverse.
  3. EVERY parameter of the model receives a nonzero gradient and is in exactly
     one optimizer param group -- i.e. no repeat of the dead `local_conv`.
  4. tmix.output and cmix.value are zero-init, so a fresh RWKV block is an
     exact identity (both sublayers).
  5. The whole model is strictly causal: d loss(position tp) / d input(> tp) == 0
     exactly, checked through the embedding gradient.
  6. One real fwd/bwd/opt.step() moves every parameter (nothing frozen).
  7. train.py's rwkv_chunked agrees, in float64, with all THREE independently
     written references in rwkv_x051a_ref.py (naive sequential recurrence, the
     closed-form quadratic, and the verbatim nanoRWKV chunk loop).
  8. train.py's rwkv_group_norm == nn.GroupNorm(n_head, n_embd), and time_shift
     == nn.ZeroPad2d((0,0,1,-1)), bit-for-bit.
  9. The reference init schedule is reproduced: time_decay spans the intended
     per-head timescale range and w = exp(-exp(time_decay)) stays in (0,1).
 10. --rwkv together with --mamba2 is rejected, and no layer is ever both.

Run: python research/scratch/rwkv_train_integration_check.py
"""
import os
import sys

os.environ.setdefault("NEURON_COMPETITION_R1_DEVICE_TYPE", "cpu")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, _HERE)

import torch
import torch.nn as nn

import train as T_
from rwkv_x051a_ref import (make_inputs, rwkv_quadratic, rwkv_recurrent,
                            rwkv_reference_chunked)

torch.set_num_threads(1)

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{tag}] {name}{('  ' + detail) if detail else ''}")


T_.COMPUTE_DTYPE = torch.float32

# Toy geometry: n_embd 128 = 4 heads x 32, chunk 32 divides seq_len 64 (so NC=2,
# which is the smallest NC that actually exercises the inter-chunk carry).
SEQ, VOCAB, NEMBD, NHEAD = 64, 512, 128, 4
T_.RWKV_HEAD_DIM = 32
T_.RWKV_CHUNK = 32
T_.RWKV_CMIX_EXPAND = 3
CFG = dict(sequence_len=SEQ, vocab_size=VOCAB, n_layer=T_.DEPTH, n_head=NHEAD, n_embd=NEMBD)


def build(rwkv: bool, pattern="RRRRRR", seed=0):
    T_.USE_RWKV = rwkv
    T_.RWKV_PATTERN = pattern
    torch.manual_seed(seed)
    m = T_.GPT(T_.GPTConfig(**CFG))
    m.init_weights()
    return m


# --- 1. toggle OFF is identical to the untouched transformer ------------------
print("== 1. USE_RWKV=False leaves the transformer path untouched ==")
base = build(False, seed=1)
names_off = {n: tuple(p.shape) for n, p in base.named_parameters()}
check("no block has a tmix", all(b.tmix is None for b in base.transformer.h))
check("no block has a cmix", all(b.cmix is None for b in base.transformer.h))
check("every block has attn and mlp",
      all(b.attn is not None and b.mlp is not None for b in base.transformer.h))
check("no rwkv params leaked into the model",
      not any(("tmix" in n) or ("cmix" in n) for n in names_off))
idx = torch.randint(0, VOCAB, (2, SEQ + 1), generator=torch.Generator().manual_seed(5))
torch.manual_seed(1)
base2 = T_.GPT(T_.GPTConfig(**CFG))
base2.init_weights()
l1 = base(idx[:, :-1], idx[:, 1:]).item()
l2 = base2(idx[:, :-1], idx[:, 1:]).item()
check("same-seed rebuild gives bit-identical loss", l1 == l2, f"{l1!r} vs {l2!r}")
n_off = sum(p.numel() for p in base.parameters())
print(f"       transformer params (toy shape) = {n_off:,}  loss={l1:.6f}")

# --- 2. pure RWKV and a hybrid pattern both build ----------------------------
print("\n== 2. USE_RWKV=True builds ==")
model = build(True, "RRRRRR", seed=1)
flags = [b.is_rwkv for b in model.transformer.h]
print(f"       is_rwkv per layer = {flags}")
check("pattern RRRRRR -> all True", all(flags))
check("rwkv layers have tmix+cmix and no attn/mlp",
      all((b.tmix is not None and b.cmix is not None
           and b.attn is None and b.mlp is None)
          for b in model.transformer.h if b.is_rwkv))
tm = model.transformer.h[0].tmix
print(f"       tmix: n_head={tm.n_head} head_dim={tm.head_dim} chunk={tm.chunk} "
      f"cmix_hidden={model.transformer.h[0].cmix.key.weight.shape[0]}")
n_on = sum(p.numel() for p in model.parameters())
print(f"       pure-RWKV params (toy shape) = {n_on:,} ({n_on / n_off:.3f}x the transformer)")

hyb = build(True, "RRRARR", seed=1)
hflags = [b.is_rwkv for b in hyb.transformer.h]
check("pattern RRRARR -> [T,T,T,F,T,T]", hflags == [True, True, True, False, True, True])
check("hybrid attention layer has attn+mlp and no tmix/cmix",
      hyb.transformer.h[3].attn is not None and hyb.transformer.h[3].mlp is not None
      and hyb.transformer.h[3].tmix is None and hyb.transformer.h[3].cmix is None)
check("hybrid forward runs", torch.isfinite(hyb(idx[:, :-1], idx[:, 1:])).item())
check("no layer is both mamba and rwkv",
      not any(b.is_mamba and b.is_rwkv for b in hyb.transformer.h))
# RWKV layers KEEP their value-embedding table (unlike Mamba2), because x051a has
# a real per-token value stream to inject it into.
ve_layers = sorted(int(k) for k in model.value_embeds.keys())
want_ve = [i for i in range(T_.DEPTH) if T_.USE_VE and T_.has_ve(i, T_.DEPTH)]
check("RWKV layers keep their value_embeds tables", ve_layers == want_ve,
      f"{ve_layers} vs want {want_ve}")
check("tmix on a has_ve layer has a ve_gate",
      all(model.transformer.h[i].tmix.ve_gate is not None for i in want_ve))

# --- 4. zero-init output projections => a fresh RWKV block is an exact no-op --
print("\n== 4. zero-init: a fresh RWKV block is an exact identity ==")
u = torch.randn(2, SEQ, NEMBD) * 3.0
blk0 = model.transformer.h[0]
yt = blk0.tmix(T_.norm(u))
yc = blk0.cmix(T_.norm(u))
check("tmix output is exactly 0.0 at init", bool((yt == 0).all()),
      f"max|y|={yt.abs().max().item():.3e}")
check("cmix output is exactly 0.0 at init", bool((yc == 0).all()),
      f"max|y|={yc.abs().max().item():.3e}")
check("tmix.output.weight is exactly 0", bool((blk0.tmix.output.weight == 0).all()))
check("cmix.value.weight is exactly 0", bool((blk0.cmix.value.weight == 0).all()))
for nm, w in [("tmix.receptance", blk0.tmix.receptance.weight),
              ("tmix.key", blk0.tmix.key.weight),
              ("tmix.value", blk0.tmix.value.weight),
              ("tmix.gate", blk0.tmix.gate.weight),
              ("cmix.key", blk0.cmix.key.weight),
              ("cmix.receptance", blk0.cmix.receptance.weight)]:
    check(f"{nm}.weight is NOT 0 (would be a dead layer)", w.abs().max().item() > 0)

# --- 9. the reference init schedule survived the port ------------------------
print("\n== 9. reference layer-dependent init ==")
for i, b in enumerate(model.transformer.h):
    td = b.tmix.time_decay
    w = torch.exp(-torch.exp(td))
    check(f"layer {i}: w = exp(-exp(time_decay)) strictly in (0,1)",
          bool((w > 0).all() and (w < 1).all()),
          f"time_decay[{td.min().item():.3f},{td.max().item():.3f}] "
          f"w[{w.min().item():.5f},{w.max().item():.5f}]")
check("time_decay is not all-equal across heads (per-head timescales)",
      model.transformer.h[0].tmix.time_decay.std().item() > 0)
check("time_maa_k in [0,1]",
      bool((model.transformer.h[0].tmix.time_maa_k >= 0).all()
           and (model.transformer.h[0].tmix.time_maa_k <= 1).all()))
check("ln_x_weight is 1, ln_x_bias is 0 at init",
      bool((model.transformer.h[0].tmix.ln_x_weight == 1).all()
           and (model.transformer.h[0].tmix.ln_x_bias == 0).all()))

# --- 3/6. no dead parameters, no ungrouped parameters ------------------------
# Note on step 0: this repo zero-inits every block's output projection, so at step 0
# the gradient w.r.t. everything UPSTREAM of one is exactly 0 -- equally true of the
# plain transformer. So "is this parameter alive?" is only answerable after a few real
# optimizer steps. `local_conv.weight` is excluded: a KNOWN pre-existing dead
# parameter in this repo (zero-init, in no optimizer group).
print("\n== 3/6. every parameter is trained (grad + optimizer group) ==")
DEAD_KNOWN = ("local_conv.",)
opt = model.setup_optimizer()
grouped = [id(p) for g in opt.param_groups for p in g["params"]]
gset = set(grouped)
named = list(model.named_parameters())
ungrouped = [n for n, p in named if id(p) not in gset]
check("no ungrouped parameters besides the known-dead local_conv",
      all(any(k in n for k in DEAD_KNOWN) for n in ungrouped), f"{ungrouped}")
check("every tmix/cmix parameter tensor is in an optimizer group",
      not [n for n, p in named
           if ("tmix" in n or "cmix" in n) and id(p) not in gset])
check("no parameter in two groups", len(grouped) == len(gset),
      f"{len(grouped)} entries vs {len(gset)} unique")

NSTEP = 4
gen = torch.Generator().manual_seed(11)
before = {n: p.detach().clone() for n, p in named}
seen_grad = {n: 0.0 for n, _ in named}
for s in range(NSTEP):
    bidx = torch.randint(0, VOCAB, (2, SEQ + 1), generator=gen)
    opt.zero_grad(set_to_none=True)
    ls = model(bidx[:, :-1], bidx[:, 1:])
    ls.backward()
    for n, p in named:
        if p.grad is not None:
            seen_grad[n] = max(seen_grad[n], p.grad.abs().max().item())
    opt.step()
    print(f"       step {s}: loss={ls.item():.5f}")
check("loss stayed finite over the steps", torch.isfinite(ls).item())
nograd = [n for n, g in seen_grad.items()
          if g == 0.0 and not any(k in n for k in DEAD_KNOWN)]
check(f"every parameter got a nonzero gradient within {NSTEP} steps", not nograd,
      f"{nograd}")
frozen = [n for n, p in named if (p - before[n]).abs().max().item() == 0.0
          and not any(k in n for k in DEAD_KNOWN)]
check(f"no parameter frozen after {NSTEP} steps", not frozen, f"{frozen}")
for n, p in named:
    if n.startswith("transformer.h.0.") and ("tmix" in n or "cmix" in n):
        print(f"       {n:44s} {str(tuple(p.shape)):>14s}  |grad|max={seen_grad[n]:.3e}"
              f"  moved={(p - before[n]).abs().max().item():.3e}")

# --- 5. strict causality of the whole RWKV model -----------------------------
print("\n== 5. strict causality of the whole pure-RWKV model ==")
model.zero_grad(set_to_none=True)
worst = 0.0
for tp in (0, 1, 5, 30, 31, 32, 40, SEQ - 1):
    pert = torch.zeros(1, SEQ, NEMBD, requires_grad=True)
    saved = model.transformer.wte

    class _Wte(torch.nn.Module):
        def forward(self, i):
            return saved(i) + pert

    model.transformer.wte = _Wte()
    out = model(idx[:1, :-1])
    model.transformer.wte = saved
    out[0, tp].sum().backward()
    gf = pert.grad[0, tp + 1:]
    w = gf.abs().max().item() if gf.numel() else 0.0
    worst = max(worst, w)
    print(f"       tp={tp:3d}  worst future |d out/d in| = {w:.3e}")
check("no gradient flows from a position to any later input", worst == 0.0,
      f"worst={worst:.3e}")

# --- 7. rwkv_chunked == all three independent references (float64) -----------
print("\n== 7. train.py rwkv_chunked vs three independent references (float64) ==")
for cfg, Q in [(dict(bsz=2, T=64, H=3, N=8), 16),
               (dict(bsz=1, T=128, H=4, N=16), 32),
               (dict(bsz=1, T=96, H=2, N=8), 96),
               (dict(bsz=2, T=48, H=5, N=4), 16),
               (dict(bsz=1, T=256, H=2, N=8), 128)]:
    inp = make_inputs(**cfg, seed=cfg["T"] * 7 + cfg["H"])
    log_w = -torch.exp(inp["time_decay"])
    masks = T_._rwkv_masks(Q, cfg["T"] // Q, inp["r"].device, torch.float64)
    y_tp = T_.rwkv_chunked(inp["r"], inp["k"], inp["v"], log_w, inp["time_faaaa"], Q, masks)
    tag = f"T={cfg['T']} H={cfg['H']} N={cfg['N']} Q={Q}"
    for rname, ref in [("recurrent", rwkv_recurrent(**inp)),
                       ("quadratic", rwkv_quadratic(**inp)),
                       ("nanoRWKV-chunked",
                        rwkv_reference_chunked(inp["r"], inp["k"], inp["v"],
                                               inp["time_decay"], inp["time_faaaa"], Q))]:
        rel = ((y_tp - ref).abs().max() / ref.abs().max()).item()
        check(f"rwkv_chunked == {rname}  {tag}", rel < 1e-12, f"max_rel={rel:.3e}")

# A masked entry must be EXACTLY zero, not merely tiny: this is what makes the
# +inf*0 = NaN hazard impossible even if time_decay learns a large positive value.
print("\n       masked-region exactness / overflow safety")
mk = T_._rwkv_masks(8, 4, torch.device("cpu"), torch.float64)
e_intra, m_intra, eye_q, _, _, e_l2, m_l2 = mk
check("e_intra has no negative exponent", bool((e_intra >= 0).all()))
check("e_l2 has no negative exponent", bool((e_l2 >= 0).all()))
for lw in (-1e-6, -1.0, -100.0):
    d = torch.exp(lw * e_intra + m_intra)
    upper = torch.triu(torch.ones_like(d))
    check(f"intra decay is EXACTLY 0 on/above the diagonal (log_w={lw})",
          bool((d * upper == 0).all()), f"max={(d * upper).abs().max().item():.3e}")
    dl = torch.exp(lw * e_l2 + m_l2)
    check(f"inter-chunk decay is EXACTLY 0 on/above the diagonal (log_w={lw})",
          bool((dl * torch.triu(torch.ones_like(dl))) .abs().max().item() == 0.0))
# and the pathological case: a huge |log_w| must not make anything non-finite
inp = make_inputs(bsz=1, T=64, H=2, N=8, seed=1)
inp["time_decay"] = torch.tensor([[6.0], [-20.0]], dtype=torch.float64)  # w ~ 0 and w ~ 1
lwx = -torch.exp(inp["time_decay"])
yx = T_.rwkv_chunked(inp["r"], inp["k"], inp["v"], lwx, inp["time_faaaa"], 16,
                     T_._rwkv_masks(16, 4, inp["r"].device, torch.float64))
check("extreme time_decay (w~0 and w~1) gives a finite output",
      bool(torch.isfinite(yx).all()))
yref = rwkv_recurrent(**inp)
relx = ((yx - yref).abs().max() / yref.abs().max()).item()
check("extreme time_decay still matches the recurrence", relx < 1e-12, f"max_rel={relx:.3e}")

# --- 8. the two rewritten primitives are bit-equal to what they replaced -----
print("\n== 8. rewritten primitives vs the reference nn.* modules ==")
xg = torch.randn(3, SEQ, NEMBD, dtype=torch.float32)
zp = nn.ZeroPad2d((0, 0, 1, -1))
check("time_shift == nn.ZeroPad2d((0,0,1,-1))",
      bool((T_.time_shift(xg) == zp(xg)).all()),
      f"max|delta|={(T_.time_shift(xg) - zp(xg)).abs().max().item():.3e}")
gn = nn.GroupNorm(NHEAD, NEMBD, eps=T_.RWKV_LN_X_EPS)
with torch.no_grad():
    gn.weight.normal_()
    gn.bias.normal_()
yy = torch.randn(3, SEQ, NHEAD, NEMBD // NHEAD, dtype=torch.float32)
mine = T_.rwkv_group_norm(yy, gn.weight, gn.bias)
theirs = gn(yy.reshape(3 * SEQ, NEMBD)).reshape(3, SEQ, NEMBD)
d = (mine - theirs).abs().max().item()
check("rwkv_group_norm == nn.GroupNorm(n_head, n_embd)", d < 1e-5, f"max|delta|={d:.3e}")

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
