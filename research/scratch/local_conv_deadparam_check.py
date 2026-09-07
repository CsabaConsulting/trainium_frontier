"""Is `CausalConv` (train.py's `local_conv`) actually live? CPU-only audit.

Written while starting work on a fused NKI kernel for the depthwise causal conv1d
(the profile's largest non-GEMM exposed-time target, 5.4% of the compiled window).
Before optimizing it, check what it computes. Three independent facts, each
asserted rather than eyeballed:

  1. `CausalConv.weight` is zero-initialized in `__init__` and `GPT.init_weights`
     never touches it.
  2. It is in NO optimizer param group: `GPT.setup_optimizer` collects
     `block.attn.parameters()` + `block.mlp.parameters()` into `matrix_params` and
     resid/x0/v lambdas into `scalar_params`; `block.local_conv.parameters()` is
     collected nowhere, so `MuonAdamW` never sees it and it stays zero for the
     entire run.
  3. Therefore `local_conv(norm(x))` is exactly 0.0 and `x = x + 0.0` is a
     bit-exact no-op -- the whole branch is dead compute in forward AND backward.

Also checks the one channel through which a dead-but-differentiated parameter
could still perturb training (grad-norm clipping over `model.parameters()`), and
whether `GRAD_CLIP` currently enables it.

Run: python research/scratch/local_conv_deadparam_check.py
"""
import os
import sys

os.environ.setdefault("NEURON_COMPETITION_R1_DEVICE_TYPE", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F

import train as T_

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{tag}] {name}{('  ' + detail) if detail else ''}")


T_.COMPUTE_DTYPE = torch.float32  # CPU has no bf16 matmul kernels worth using here
torch.manual_seed(0)

cfg = T_.GPTConfig(sequence_len=64, vocab_size=512, n_layer=T_.DEPTH, n_head=4, n_embd=128)
model = T_.GPT(cfg)
model.init_weights()

print(f"config={cfg}\nUSE_LOCAL_CONV={T_.USE_LOCAL_CONV} LOCAL_CONV_KERNEL={T_.LOCAL_CONV_KERNEL} "
      f"GRAD_CLIP={T_.GRAD_CLIP}\n")

convs = [b.local_conv for b in model.transformer.h]
check("local_conv exists on every block", all(c is not None for c in convs))

# --- 1. weight is exactly zero after construction + init_weights -------------
allzero = all((c.weight == 0).all().item() for c in convs)
check("weight == 0 exactly after init_weights", allzero,
      f"max|w|={max(c.weight.abs().max().item() for c in convs):.3e}")

# --- 2. weight is in no optimizer param group --------------------------------
opt = model.setup_optimizer()
grouped = {id(p) for g in opt.param_groups for p in g["params"]}
conv_ids = {id(c.weight) for c in convs}
n_missing = len(conv_ids - grouped)
check("local_conv.weight is in NO optimizer param group", n_missing == len(conv_ids),
      f"{n_missing}/{len(conv_ids)} conv weights ungrouped")
n_params_total = sum(1 for _ in model.parameters())
print(f"       params in model={n_params_total}  params in optimizer={len(grouped)}  "
      f"ungrouped={n_params_total - len(grouped)}")
ungrouped_named = [n for n, p in model.named_parameters() if id(p) not in grouped]
print(f"       ungrouped names: {ungrouped_named}")

# --- 3. the branch is a bit-exact no-op in the forward -----------------------
for dt in (torch.float32, torch.bfloat16):
    x = (torch.randn(2, cfg.sequence_len, cfg.n_embd) * 3.0).to(dt)
    conv = convs[0]
    y = conv(T_.norm(x).to(dt))
    check(f"conv output is exactly 0.0 ({dt})", bool((y == 0).all()),
          f"max|y|={y.abs().max().item():.3e}")
    check(f"x + conv(norm(x)) is bit-exact x ({dt})",
          bool(((x + y) == x).all()) and (x + y).dtype == x.dtype)

# --- 4. a real training step: does anything move the weight? -----------------
idx = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len + 1))
before = [c.weight.detach().clone() for c in convs]
loss = model(idx[:, :-1], idx[:, 1:])
loss.backward()
gnorms = [c.weight.grad.norm().item() if c.weight.grad is not None else None for c in convs]
print(f"       dweight norms: {['%.3e' % g if g is not None else 'None' for g in gnorms]}")
check("dweight IS computed (backward pays for it)",
      all(g is not None and g > 0 for g in gnorms))
opt.step()
opt.zero_grad(set_to_none=True)
moved = max((c.weight - b).abs().max().item() for c, b in zip(convs, before))
check("weight unchanged after a full fwd/bwd/opt.step()", moved == 0.0, f"max move={moved:.3e}")

# --- 5. the only path a dead-but-differentiated param could still matter -----
check("GRAD_CLIP disabled, so dweight cannot leak into training via grad-norm",
      T_.GRAD_CLIP == 0.0, f"GRAD_CLIP={T_.GRAD_CLIP}")

# --- 6. what deleting the branch would cost/save in reported numbers ---------
conv_params = sum(c.weight.numel() for c in convs)
print(f"\nconv params (never trained, but counted in num_params_M and all-reduced "
      f"every step): {conv_params} ({conv_params / 1e6:.4f}M) at this toy size; "
      f"{T_.DEPTH * 640 * T_.LOCAL_CONV_KERNEL} ({T_.DEPTH * 640 * T_.LOCAL_CONV_KERNEL / 1e6:.4f}M) "
      f"at the locked shape (C=640)")

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
