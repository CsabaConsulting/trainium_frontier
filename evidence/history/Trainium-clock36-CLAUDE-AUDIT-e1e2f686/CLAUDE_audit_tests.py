"""Independent CPU reference tests for candidate train.py (SHA e1e2f686...).

Read-only use of the candidate: imported as a module (not __main__), CPU only, stub prepare.
No data, no training run, no evaluator. Tests the eager paths and re-derives the NKI kernel
math (rope+norm fwd/bwd, two-tap conv fwd/bwd, softcap-CE fwd/bwd) in plain torch.
"""
import math, os, sys, tempfile
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candidate_train as T

torch.manual_seed(0)
results = []
def report(name, ok, detail):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

print("module:", T.__file__, "HAS_NKI:", T._HAS_NKI)
print("NEURON_CC_FLAGS set at import:", os.environ.get("NEURON_CC_FLAGS"),
      "| NEURON_COMPILE_CACHE_DIR:", os.environ.get("NEURON_COMPILE_CACHE_DIR"))

# ---------- build the exact submitted architecture (launch flags: n_embd 1024, seq 1024) ----------
T.COMPUTE_DTYPE = torch.float32          # what load_for_eval selects on CPU
T.SEQ_LEN, T.N_EMBD = 1024, 1024
cfg = T.build_config()
print("config:", cfg)
model = T.GPT(cfg)
model.init_weights()
nparams = sum(p.numel() for p in model.parameters())
print(f"params: {nparams:,}")
attn_layers = [i for i, b in enumerate(model.transformer.h) if b.attn is not None]
print("attention layers:", attn_layers, "| value-embed layers:", list(model.value_embeds.keys()),
      "| unet gates:", len(model.unet_skip_gates))
# Perturb every parameter so zero-initialised pieces (c_proj, gates, conv, ve_gate) are exercised.
with torch.no_grad():
    for p in model.parameters():
        p.add_(torch.randn_like(p) * 0.02)
model.eval()

B, Tn = 2, 256
idx = torch.randint(0, 8192, (B, Tn))

# T1 causality: changing tokens at positions >= k must not change logits at positions < k.
with torch.no_grad():
    base = model(idx)
    k = 100
    idx2 = idx.clone(); idx2[:, k:] = torch.randint(0, 8192, (B, Tn - k))
    alt = model(idx2)
d = (base[:, :k] - alt[:, :k]).abs().max().item()
dk = (base[:, k:] - alt[:, k:]).abs().max().item()
report("T1 causality (eval, full 8-layer U-Net)", d == 0.0 and dk > 0, f"max|dlogits[<k]|={d:.3e}, future positions changed by {dk:.3e}")

# T2 batch independence: row 0 alone vs inside a batch of 3 with different companions.
with torch.no_grad():
    solo = model(idx[:1])
    other = torch.randint(0, 8192, (2, Tn))
    batch3 = model(torch.cat([idx[:1], other], 0))
d = (solo[0] - batch3[0]).abs().max().item()
report("T2 batch independence (B1 vs B3, CPU)", d < 1e-5, f"max|diff|={d:.3e}")

# T3 train-mode (_FusedMLP autograd path) vs eval-mode logits, same weights.
model.train()
with torch.no_grad():
    tr = model(idx)
model.eval()
d = (tr - base).abs().max().item()
report("T3 train-mode vs eval-mode forward (MLP routes differ)", d < 1e-4, f"max|diff|={d:.3e}")

# T4 _FusedMLP backward vs plain autograd (bf16, as on device).
x = torch.randn(512, 1024, dtype=torch.bfloat16, requires_grad=True)
up = (torch.randn(4096, 1024) * 0.02).bfloat16().requires_grad_(True)
down = (torch.randn(1024, 4096) * 0.02).bfloat16().requires_grad_(True)
r = torch.randn(512, 1024, dtype=torch.bfloat16)
out_f = T._FusedMLP.apply(x, up, down)
gx, gu, gd = torch.autograd.grad((out_f.float() * r.float()).sum(), (x, up, down))
x2, up2, down2 = [t.detach().clone().requires_grad_(True) for t in (x, up, down)]
out_r = F.linear(F.relu(F.linear(x2, up2)).square(), down2)
rx, ru, rd = torch.autograd.grad((out_r.float() * r.float()).sum(), (x2, up2, down2))
def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()
fw = rel(out_f, out_r); ex, eu, ed = rel(gx, rx), rel(gu, ru), rel(gd, rd)
report("T4 _FusedMLP fwd/bwd vs autograd (bf16)", fw == 0.0 and max(ex, eu, ed) < 2e-2,
       f"fwd rel={fw:.2e}; grad rel x={ex:.2e} up={eu:.2e} down={ed:.2e}")

# T5 checkpoint round trip through the real save_checkpoint / load_for_eval.
with tempfile.TemporaryDirectory() as td:
    from pathlib import Path
    path = Path(td) / "final.pt"
    T.save_checkpoint(path, model, {"step": 0})
    loaded = T.load_for_eval(str(path), torch.device("cpu"))
    with torch.no_grad():
        lo = loaded(idx)
    d = (lo - base).abs().max().item()
    frozen = all(not p.requires_grad for p in loaded.parameters())
    hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in loaded.modules())
    report("T5 save_checkpoint -> load_for_eval (weights_only) identical logits", d == 0.0 and frozen and not loaded.training and hooks == 0,
           f"max|diff|={d:.3e}, eval={not loaded.training}, frozen={frozen}, hooks={hooks}, seq_len={loaded.config.sequence_len}")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    print("   checkpoint keys:", sorted(raw.keys()), "| model_config:", raw["model_config"])

# T6 two-tap causal conv: re-derive the NKI kernel's tap mapping and backward in plain torch.
C, L, Bc = 256, 384, 2
xc = torch.randn(Bc, L, C, requires_grad=True)
w = torch.randn(C, 1, 2, requires_grad=True)
conv = T.CausalConv(C, 2); conv.weight.data.copy_(w.data)
y_ref = conv(xc)                                         # eager fallback (F.conv1d, left pad 1)
wt = w[:, 0, :].T                                        # kernel layout: row0 = prev tap, row1 = current tap
prev = torch.cat([torch.zeros(Bc, 1, C), xc[:, :-1]], 1)
y_k = prev * wt[0] + xc * wt[1]                          # _conv2_fwd_tile formula
g = torch.randn_like(y_ref)
gn = torch.cat([g[:, 1:], torch.zeros(Bc, 1, C)], 1)
dx_k = g * wt[1] + gn * wt[0]                            # _conv2_bwd_tile dx formula
dw_k = torch.stack([(prev * g).sum((0, 1)), (xc * g).sum((0, 1))], 0).T.unsqueeze(1)  # partial taps
dx_ref, dw_ref = torch.autograd.grad((y_ref * g).sum(), (xc, conv.weight))
report("T6 conv kernel math vs eager conv1d (fwd, dx, dw, causal)",
       rel(y_k, y_ref) < 1e-6 and rel(dx_k, dx_ref) < 1e-6 and rel(dw_k, dw_ref) < 1e-6,
       f"fwd rel={rel(y_k, y_ref):.1e} dx rel={rel(dx_k, dx_ref):.1e} dw rel={rel(dw_k, dw_ref):.1e}")

# T7 fused rope+norm kernel math (fwd and hand-written bwd) vs eager autograd, fp32.
Bq, Tq, H, D = 2, 256, 8, 128
xq = torch.randn(Bq, Tq, H, D, requires_grad=True)
cos, sin = [t.float() for t in model._precompute_rotary_embeddings(Tq, D)]
def eager(x):
    return F.rms_norm(T.apply_rotary_emb(x, cos, sin), (D,), eps=T._RMS_EPS)
ye = eager(xq)
gq = torch.randn_like(ye)
dref, = torch.autograd.grad((ye * gq).sum(), (xq,))
with torch.no_grad():
    c, s = cos, sin
    x1, x2 = xq[..., :D // 2], xq[..., D // 2:]
    z = torch.cat([x1 * c + x2 * s, x2 * c - x1 * s], -1)          # kernel z1/z2
    ms = (z * z).mean(-1, keepdim=True) + T._RMS_EPS
    rr = ms.rsqrt(); r2 = 1.0 / ms
    yk = z * rr
    qq = (gq * z).mean(-1, keepdim=True) * r2
    dz = (gq - z * qq) * rr
    d1, d2 = dz[..., :D // 2], dz[..., D // 2:]
    dxk = torch.cat([d1 * c - d2 * s, d2 * c + d1 * s], -1)          # kernel o1/o2
report("T7 rope+norm kernel math fwd/bwd vs eager autograd",
       rel(yk, ye) < 1e-5 and rel(dxk, dref) < 1e-5, f"fwd rel={rel(yk, ye):.1e} bwd rel={rel(dxk, dref):.1e}")

# T8 softcap-CE kernel math (training loss only) vs eager.
N, V = 256, 8192
zl = (torch.randn(N, V) * 5).requires_grad_(True)
tg = torch.randint(0, V, (N,))
le = F.cross_entropy(15 * torch.tanh(zl / 15), tg, reduction="none")
gl = torch.randn(N)
dze, = torch.autograd.grad((le * gl).sum(), (zl,))
with torch.no_grad():
    cu = torch.tanh(zl / 15)
    tv = 15 * cu.gather(1, tg[:, None])
    den = torch.exp(15 * cu - tv).sum(1, keepdim=True)
    lk = torch.log(den).squeeze(1)
    lse = tv + torch.log(den)
    sm = torch.exp(15 * cu - lse)
    oh = F.one_hot(tg, V).float() * gl[:, None]
    dzk = (1 - cu * cu) * (sm * gl[:, None] - oh)
report("T8 softcap-CE kernel math fwd/bwd vs eager", rel(lk, le) < 1e-5 and rel(dzk, dze) < 1e-5,
       f"loss rel={rel(lk, le):.1e} grad rel={rel(dzk, dze):.1e}")

# T9 schedule/accounting helpers.
T._PROCESS_STARTED = 0.0
a = T.startup_adjusted_schedule_elapsed(2, elapsed=860.0, budget_started=0.0)   # 860 s in, under cap-30
b = T.startup_adjusted_schedule_elapsed(2, elapsed=1000.0, budget_started=0.0)  # startup overran
c_ = T.startup_adjusted_schedule_elapsed(7, elapsed=123.0, budget_started=0.0)  # after boundary 5
report("T9 schedule clock helper (LR only; stop clock separate)", a == 0.0 and abs(b - 130.0) < 1e-9 and c_ == 123.0,
       f"startup 860s->{a}, startup 1000s->{b} (charged excess), post-boundary->{c_}")

print()
print(f"SUMMARY: {sum(ok for _, ok in results)}/{len(results)} PASS")
