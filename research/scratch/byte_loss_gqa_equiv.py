"""CPU equivalence checks for the sub-1.0 plan P2/P4 diffs (GQA + byte-weighted loss).

Convention: research/scratch/*_equiv.py numeric checks, runnable on CPU with the
repo's torch. Run:  python3 research/scratch/byte_loss_gqa_equiv.py
Covers:
  1. LOSS_BYTE_WEIGHT=OFF (or weights None) is exactly the historical mean/none.
  2. LOSS_BYTE_WEIGHT=ON equals manual sum(w*l)/sum(w) through the EAGER path.
  3. GQA/MQA expand path is exact: an MQA module matches a dense module whose
     K/V rows are the tiled MQA rows, end to end (proj -> repeat -> rope -> SDPA).
  4. GQA refuses fused-QKV/coalesced combos and non-divisor kv counts.
  5. build_config validates --n-kv-heads.
Hardware still must confirm: NKI softcap+CE kernel with rerouted "none", and the
rope+QK-norm kernel firing on repeat_interleave+contiguous K/V.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(f"FAILED: {name}")


torch.manual_seed(0)

# --- 1+2: byte-weighted loss through the eager path (CPU => NKI guarded off) ---
B, Tm, V = 2, 16, 64
logits = torch.randn(B, Tm, V, dtype=torch.float32)
targets = torch.randint(0, V, (B, Tm))
nbytes = torch.randint(1, 6, (V,)).to(torch.float32)  # synthetic id->bytes

T.LOSS_BYTE_WEIGHT = False
T.BYTE_LOSS_W = None
ref_mean = T.softcap_cross_entropy(logits, targets, "mean")
ref_none = T.softcap_cross_entropy(logits, targets, "none")
capped = T.LOGIT_SOFTCAP * torch.tanh(logits / T.LOGIT_SOFTCAP)
manual_mean = F.cross_entropy(capped.reshape(-1, V), targets.reshape(-1), reduction="mean")
manual_none = F.cross_entropy(capped.reshape(-1, V), targets.reshape(-1), reduction="none")
check("flag-OFF mean == manual CE mean", torch.equal(ref_mean, manual_mean))
check("flag-OFF none == manual CE none", torch.equal(ref_none, manual_none.view_as(targets)))

T.LOSS_BYTE_WEIGHT = True
T.BYTE_LOSS_W = None  # flag on but no table => historical fallback
check("flag-ON weights-None == manual mean", torch.equal(
    T.softcap_cross_entropy(logits, targets, "mean"), manual_mean))

w = nbytes / nbytes.mean()
T.BYTE_LOSS_W = w
got = T.softcap_cross_entropy(logits, targets, "mean")
per = manual_none
expected = (per * w[targets.reshape(-1)]).sum() / w[targets.reshape(-1)].sum()
check("flag-ON mean == manual sum(w*l)/sum(w)", torch.allclose(got, expected, atol=1e-6))
check("flag-ON none still raw per-token", torch.equal(
    T.softcap_cross_entropy(logits, targets, "none"), manual_none.view_as(targets)))
T.LOSS_BYTE_WEIGHT = False
T.BYTE_LOSS_W = None

# --- 3: GQA expand-path exactness at the attention-module level ---
n_head, n_kv, hd, d = 4, 2, 8, 4 * 8
cfg_mqa = T.GPTConfig(sequence_len=32, vocab_size=V, n_layer=2, n_head=n_head,
                      n_embd=d, n_kv_heads=n_kv)
attn = T.CausalSelfAttention(cfg_mqa, layer_idx=0)
check("MQA c_k/c_v rows == n_kv*hd",
      attn.c_k.weight.shape == (n_kv * hd, d) and attn.c_v.weight.shape == (n_kv * hd, d))
check("kv_repeat == n_head/n_kv", attn.kv_repeat == n_head // n_kv == 2)

# Dense reference with tiled rows: must match the MQA module bit-exactly.
cfg_dense = T.GPTConfig(sequence_len=32, vocab_size=V, n_layer=2, n_head=n_head, n_embd=d)
ref = T.CausalSelfAttention(cfg_dense, layer_idx=0)
def _tile_heads(w_small, n_kv, hd, repeat):
    # Repeat whole head-blocks (n_kv,hd,d) -> (n_head,hd,d), NOT flat rows: MQA head h
    # reads small-head h//repeat, so the dense reference must tile head-blocks.
    return w_small.view(n_kv, hd, -1).repeat_interleave(repeat, dim=0).reshape(-1, w_small.shape[-1])


with torch.no_grad():
    ref.c_q.weight.copy_(attn.c_q.weight)
    ref.c_k.weight.copy_(_tile_heads(attn.c_k.weight, n_kv, hd, attn.kv_repeat))
    ref.c_v.weight.copy_(_tile_heads(attn.c_v.weight, n_kv, hd, attn.kv_repeat))
    ref.c_proj.weight.copy_(attn.c_proj.weight)
    if ref.ve_gate is not None and attn.ve_gate is not None:
        ref.ve_gate.weight.copy_(attn.ve_gate.weight)
    if getattr(ref, "head_gate", None) is not None and getattr(attn, "head_gate", None) is not None:
        ref.head_gate.weight.copy_(attn.head_gate.weight)
        ref.head_gate_bias.copy_(attn.head_gate_bias)

x = torch.randn(B, Tm, d)
cos = torch.randn(1, Tm, 1, hd // 2)
sin = torch.randn(1, Tm, 1, hd // 2)
out_mqa = attn(x, x, x, None, cos, sin, None)
out_dense = ref(x, x, x, None, cos, sin, None)
check("MQA output == tiled-dense output (exact expand path)",
      torch.equal(out_mqa, out_dense))

# Dense default unchanged: n_kv_heads=0 => full projections, no repeat op.
attn0 = T.CausalSelfAttention(cfg_dense, layer_idx=0)
check("default n_kv_heads=0 => dense shapes",
      attn0.n_kv_heads == n_head and attn0.c_k.weight.shape == (d, d))

# --- 4: refusals ---
T.USE_FUSED_QKV = True
try:
    T.CausalSelfAttention(cfg_mqa, layer_idx=0)
    check("GQA+fused-QKV refused", False)
except ValueError:
    check("GQA+fused-QKV refused", True)
finally:
    T.USE_FUSED_QKV = False
try:
    T.CausalSelfAttention(cfg_mqa, layer_idx=0, coalesced="all")
    check("GQA+coalesce refused", False)
except ValueError:
    check("GQA+coalesce refused", True)
bad = T.GPTConfig(sequence_len=32, vocab_size=V, n_layer=2, n_head=5, n_embd=5 * hd,
                  n_kv_heads=2)
try:
    T.CausalSelfAttention(bad, layer_idx=0)
    check("non-divisor kv refused", False)
except ValueError:
    check("non-divisor kv refused", True)

# --- 5: build_config validation ---
old_depth, old_nkv, old_hd = T.DEPTH, T.N_KV_HEADS, T.HEAD_DIM
T.HEAD_DIM = hd
try:
    T.N_KV_HEADS = 1
    c = T.build_config(depth=2)
    check("build_config carries n_kv_heads", c.n_kv_heads == 1)
    T.N_KV_HEADS = 7
    try:
        T.build_config(depth=2)
        check("build_config rejects kv>n_head", False)
    except ValueError:
        check("build_config rejects kv>n_head", True)
finally:
    T.DEPTH, T.N_KV_HEADS, T.HEAD_DIM = old_depth, old_nkv, old_hd

print("ALL EQUIV CHECKS PASSED")
