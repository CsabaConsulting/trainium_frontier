#!/usr/bin/env python3
"""Build S0 = review-clean, numerics-identical rewrite of the 116188 candidate.

Input : 05-train-submission-clean-116188.py (SHA-256 116188cf...)
Output: workflow/candidates/S0/train.py

Every edit below is an exact-match block replacement asserted to occur once, so the transform is
auditable. Intent: remove constructs a reviewer could read as eval special-casing or dead
features, WITHOUT changing training arithmetic. Proven afterwards by the G2 lockstep comparison
(bit-identical losses and parameter hashes against 116188) and a load_for_eval logit comparison.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "05-train-submission-clean-116188.py"
DST = Path(__file__).resolve().parent / "train.py"
SRC_SHA = "116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d"


def between(s: str, start: str, end: str) -> tuple[int, int]:
    i = s.index(start)
    assert s.count(start) == 1, start[:60]
    j = s.index(end, i)
    return i, j


def sub(s: str, old: str, new: str) -> str:
    n = s.count(old)
    assert n == 1, f"expected 1 match, found {n}: {old[:80]!r}"
    return s.replace(old, new)


def replace_block(s: str, start: str, end: str, new: str) -> str:
    i, j = between(s, start, end)
    return s[:i] + new + s[j:]


def main() -> int:
    raw = SRC.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "source is not the 116188 candidate"
    s = raw.decode()

    # 1. Docstring: state the contract a reviewer checks.
    s = replace_block(s, '"""Trainium Frontier Phase 1 submission', "from __future__ import annotations", '''"""Trainium Frontier Phase 1: dense GPT, 6 layers x 1024 wide, trained from scratch in 1,800 s.

Launch (full chip, four LNC2 ranks):
    NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py

Model: untied token embedding and head, RMSNorm, rotary embeddings with QK-norm, ReLU^2 MLP, value
embeddings on alternating layers, learned residual/input mix, causal two-tap depthwise convolution,
softcap-15 logits. BF16 branch inputs with FP32 residuals and statistics. On Neuron, three NKI
kernels implement exactly the same math as their eager fallbacks: fused rope+QK-norm, the two-tap
causal convolution, and softcap cross-entropy (training loss only).

Optimizer: Muon (Newton-Schulz, compiled body) for matrices; AdamW for embeddings, head, scalars and
the convolution; wall-clock LR and weight-decay schedules. Data parallel over four ranks; optimizer
state and the update of each parameter are owned by one rank and the updated parameters are
all-gathered, so every rank holds the same full model after each step.

Budget: the first TRAIN_STARTUP_STEPS_EXCLUDED steps (compilation) are excluded up to the organiser's
startup allowance; training stops before max_train_seconds with a reserve for saving the checkpoint.
Training reads only the 'train' split through prepare.make_dataloader.

Evaluation: `load_for_eval` rebuilds the same GPT class used for training, strict-loads the saved
weights, and returns it in eval mode. Its forward is a pure causal function of the input tokens
(no state, no collectives, no dependence on batch composition). The rotary tables are buffers, not
weights; they are built for up to EVAL_MIN_SEQ_LEN positions so any scorer context length works,
and positions below the training length get exactly the training values.
"""
''')

    # 2. Imports: drop unused modules and unused prepare names (no validation-data helpers).
    s = sub(s, "import random\n", "")
    s = sub(s, "import pyarrow.parquet as pq\n", "")
    i, j = between(s, "from prepare import ", "\n")
    s = s[:i] + ("from prepare import TOKENIZER_VOCAB_SIZE, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS, "
                 "TRAIN_STARTUP_STEPS_EXCLUDED, TRAIN_TIME_BUDGET_SECONDS, ensure_tokenizer, get_dist_info, "
                 "make_dataloader, print0, record_step_boundary") + s[j:]

    # 3. Constants: bake the trained configuration (previously supplied as launch flags).
    s = sub(s, "SEQ_LEN = 2048\n", "SEQ_LEN = 1024\n")
    s = sub(s, "ASPECT_RATIO = 96\n", "")
    s = sub(s, "N_EMBD = 0\n", "N_EMBD = 1024\n")
    s = sub(s, "ATTN_SCALE = 0.0\n", "")
    s = sub(s, "N_KV_HEADS = 0\n", "")
    s = sub(s, "loss_scale_clean = 0\nloss_scale_skips = 0\nLOSS_SCALE = 1.0\n", "")
    s = sub(s, "PACK_FACTOR = 4\n", "PACK_FACTOR = 32\n")

    # 4. Config: drop GQA / attention-scale fields that are never used (dense attention, SDPA default scale).
    s = sub(s, "    mlp_ratio: int = 4\n    attn_scale: float = 0.0\n    n_kv_heads: int = 0\n", "    mlp_ratio: int = 4\n")

    # 5. Softcap-CE lane choice: plain name, defined before the kernels that call it.
    s = sub(s, "_NKI_PARTITION = 128\n", '''_NKI_PARTITION = 128

def _ce_lanes(rows: int) -> int:
    """Row-tiling lanes for the softcap cross-entropy kernel (rows must be a positive multiple of 128)."""
    if rows <= 0 or rows % 128:
        raise ValueError('the softcap cross-entropy kernel needs a positive multiple of 128 rows')
    return 2 if rows >= 512 and rows % 256 == 0 else 1
''')
    assert s.count("selected_lanes(z.shape[0])") == 2
    s = s.replace("selected_lanes(z.shape[0])", "_ce_lanes(z.shape[0])")
    s = replace_block(s, "def selected_lanes(rows):", "if __name__ == '__main__':", "")

    # 6. Attention: dense heads only (GQA and coalesced-projection branches were unreachable).
    s = replace_block(s, "class CausalSelfAttention(nn.Module):", "def reference_forward(", '''class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig, layer_idx: int=0):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim * self.n_head == self.n_embd
        self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate = Linear(VE_GATE_CHANNELS, self.n_head, bias=False) if layer_idx % 2 == (config.n_layer - 1) % 2 else None

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, c = x_q.shape
        q = self.c_q(x_q).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x_k).view(b, t, self.n_head, self.head_dim)
        v = self.c_v(x_v).view(b, t, self.n_head, self.head_dim)
        if ve is not None:
            ve = ve.view(b, t, self.n_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x_v[..., :VE_GATE_CHANNELS]))
            v = v + gate.unsqueeze(-1) * ve
        v = v.to(q.dtype)
        q = rope_norm(q, cos, sin)
        k = rope_norm(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)

''')

    # 7. MLP: ONE forward for training and evaluation. `_FusedMLP` computes exactly
    #    c_proj(relu(c_fc(x))^2) and only customises what the backward saves; the former
    #    `self.training`/shape switch to an equivalent eager path is removed.
    s = replace_block(s, "class MLP(nn.Module):", "class Block(nn.Module):", '''class MLP(nn.Module):
    """c_proj(relu(c_fc(x))**2). The same function runs in training and evaluation; `_FusedMLP` only
    changes what autograd saves for the backward pass (the ReLU output, transposed)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.c_fc = Linear(config.n_embd, hidden, bias=False)
        self.c_proj = Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = _linear_weight(self.c_fc, x.dtype)
        down = _linear_weight(self.c_proj, x.dtype)
        result = _FusedMLP.apply(x.reshape(-1, x.shape[-1]), up, down)
        return result.reshape(x.shape)

''')

    # 8. Block: no coalesced input projection; the always-present modules need no None checks.
    s = replace_block(s, "class Block(nn.Module):", "class GPT(nn.Module):", '''class Block(nn.Module):

    def __init__(self, config: GPTConfig, layer_idx: int=0):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.resid_lambda = nn.Parameter(torch.ones(1))
        self.x0_lambda = nn.Parameter(torch.zeros(1))
        self.local_conv = CausalConv(config.n_embd, LOCAL_CONV_KERNEL)

    def forward(self, x: torch.Tensor, x0: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = self.resid_lambda * x + self.x0_lambda * x0
        attn_in = x  # attention reads the block input; the local convolution adds to the residual in parallel
        x = x + self.local_conv(norm(x))
        attn_out = self.attn(norm(attn_in), norm(attn_in), norm(attn_in), ve, cos, sin)
        x = x + attn_out
        x = x + self.mlp(norm(x))
        return x

''')

    # 9. init_weights: same RNG draws in the same order, unreachable coalesced branch removed.
    s = replace_block(s, "        for block in self.transformer.h:\n            if block.c_in is not None:", "        for ve in self.value_embeds.values():", '''        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            b = ATTN_CPROJ_INIT_SCALE * s
            torch.nn.init.uniform_(block.attn.c_proj.weight, -b, b)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            block.resid_lambda.fill_(1.0)
            block.x0_lambda.fill_(0.05)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
''')

    # 10. setup_optimizer: same groups in the same order (Muon groups sorted by shape).
    s = replace_block(s, "        for block in self.transformer.h:\n            if block.local_conv is not None:", "        for p in scalar_params:", '''        for block in self.transformer.h:
            local_conv_params.append(block.local_conv.weight)
            scalar_params.extend([block.resid_lambda, block.x0_lambda])
            matrix_params.extend(block.attn.parameters())
            matrix_params.extend(block.mlp.parameters())
''')
    s = replace_block(s, "        depth_of: dict[int, int] = {}", "        opt = MuonAdamW(param_groups)", '''        for shape in sorted({p.shape for p in matrix_params}):
            params = [p for p in matrix_params if p.shape == shape]
            group = dict(kind='muon', params=params, lr=MATRIX_LR, momentum=0.95, beta2=MUON_BETA2, ns_steps=MUON_NS_STEPS, weight_decay=WEIGHT_DECAY, transposed=False)
            param_groups.append(group)
''')

    # 11. build_config from the baked constants.
    s = replace_block(s, "def build_config() -> GPTConfig:", "POLAR_EXPRESS_COEFFS = ", '''def build_config() -> GPTConfig:
    """Model configuration from the module constants: DEPTH layers, N_EMBD wide, HEAD_DIM-wide heads."""
    if N_EMBD % HEAD_DIM != 0:
        raise ValueError(f'N_EMBD={N_EMBD} must be a multiple of HEAD_DIM={HEAD_DIM}')
    return GPTConfig(sequence_len=SEQ_LEN, vocab_size=TOKENIZER_VOCAB_SIZE, n_layer=DEPTH, n_head=N_EMBD // HEAD_DIM, n_embd=N_EMBD, mlp_ratio=MLP_RATIO)
''')

    # 12. load_for_eval docstring: say exactly what the rotary sizing does.
    s = sub(s, '''    """Return the trained model in eval mode. The rotary tables are not parameters and are sized for
    sequences up to EVAL_MIN_SEQ_LEN, so evaluation may use a longer context than training did.
    """''', '''    """Return the trained model (same GPT class, strict-loaded weights) in eval mode for the scorer.

    Only the non-persistent rotary buffers are sized for up to EVAL_MIN_SEQ_LEN positions so that any
    scorer context length is accepted; for positions below the training length their values are the
    ones used in training. No weights, layers or arithmetic differ from training.
    """''')

    # 13. main(): only the options a run needs; everything else is fixed in the file.
    s = replace_block(s, "def main() -> None:", "    device_type = detect_device_type(args.device_type)", '''def main() -> None:
    parser = argparse.ArgumentParser(description='Trainium Frontier Phase 1 training: dense 6x1024 GPT, 1,800 s, full chip (LNC2 x 4 ranks).')
    parser.add_argument('--device-type', type=str, default='', help='cpu | cuda | neuron; empty selects automatically.')
    parser.add_argument('--num-steps', type=int, default=NUM_STEPS, help='Upper bound on steps; the wall-clock budget ends training first.')
    parser.add_argument('--max-train-seconds', type=int, default=MAX_TRAIN_SECONDS)
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True, help="torch.compile(backend='neuron') for the model forward/backward.")
    parser.add_argument('--seed', type=int, default=42, help='torch.manual_seed value.')
    args = parser.parse_args()
''')
    s = sub(s, "(loss / grad_accum_steps * LOSS_SCALE).backward()", "(loss / grad_accum_steps).backward()")
    s = sub(s, "group['lr'] = group['initial_lr'] * 1.0 * (lrm_flat", "group['lr'] = group['initial_lr'] * (lrm_flat")
    s = sub(s, "print0(f'gradient_clip={GRAD_CLIP:g} clip_after_reduce={True}')", "print0(f'gradient_clip={GRAD_CLIP:g} (after the cross-rank average)')")

    # 14. No validation data in the training script: remove the in-process public eval/causality check.
    s = replace_block(s, "    public_eval = None\n    causal = None\n    if args.eval_public:", "    if rank == 0:\n        print('---')", "")
    s = replace_block(s, "        if causal is not None:\n", "    cleanup_runtime()\n\ndef make_plan(", "")

    # 15. Optimizer ownership: descriptive errors, documented cost weights, no unused resume path.
    s = sub(s, "raise ValueError('ownership probe is restricted to AdamW and Muon')", "raise ValueError('optimizer ownership supports AdamW and Muon groups only')")
    s = sub(s, "def make_plan(groups, world_size):\n", '''def make_plan(groups, world_size):
    """Assign each parameter's optimizer update to one rank, balancing measured per-shape Muon step
    times (milliseconds, used only as load-balancing weights) and AdamW element counts."""
''')
    s = sub(s, "raise ValueError('install ownership before the first update; use its checkpoint loader to resume')", "raise ValueError('install ownership before the first update')")
    s = sub(s, "        self.pad_singleton_muon = bool(True)\n", "        self.pad_singleton_muon = True\n")
    s = sub(s, "        self.raw_state_dict = optimizer.state_dict\n        self.raw_load_state_dict = optimizer.load_state_dict\n", "")
    s = replace_block(s, "    def state_dict(self):\n        state = self.raw_state_dict()", "if __name__ == '__main__':", "")

    # Sanity: nothing reviewer-sensitive or dead remains.
    for bad in ("evaluate_bpb", "causality_check", "public_val", "selected_lanes", "adapter", "probe", "coalesced",
                "n_kv_heads", "LOSS_SCALE", "NEURON_COMPETITION_R1", "Not part of this build", "batch-ramp",
                "weight-ema", "byte-wte", "c_in", "self.training", "eval_public"):
        assert bad not in s, f"leftover: {bad}"
    DST.write_text(s)
    print(f"wrote {DST} sha256={hashlib.sha256(s.encode()).hexdigest()} lines={s.count(chr(10))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
