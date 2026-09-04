"""
Neuron Competition R1 baseline training script.

Participants edit this file. It is intentionally self-contained and readable.
The fixed data/tokenizer/eval utilities live in prepare.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    EVAL_TIMEOUT_SECONDS,
    TOKENIZER_VOCAB_SIZE,
    TRAIN_TIME_BUDGET_SECONDS,
    Tokenizer,
    causality_check,
    ensure_tokenizer,
    evaluate_bpb,
    get_dist_info,
    make_dataloader,
    print0,
)


# -----------------------------------------------------------------------------
# Baseline hyperparameters. Model-related config lives here, not in prepare.py.

SEQ_LEN = 1024
DEPTH = 10
ASPECT_RATIO = 64
HEAD_DIM = 128

DEVICE_BATCH_SIZE = 1
TOTAL_BATCH_SIZE = 65536
NUM_STEPS = 1000
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS - 60

LOGIT_SOFTCAP = 15.0

EMBEDDING_LR = 0.30
UNEMBEDDING_LR = 0.006
MATRIX_LR = 0.025
SCALAR_LR = 0.20
WEIGHT_DECAY = 0.10
WARMUP_STEPS = 0
GRAD_CLIP = 0.0
WARMDOWN_RATIO = 0.75   # fraction of max_train_seconds spent on LR cooldown (0 = old flat-after-warmup)
FINAL_LR_FRAC = 0.05    # LR floor as a fraction of peak LR at the end of cooldown (1.0 = no floor effect)
WD_DECAY_TO_ZERO = True  # decay Muon's weight_decay linearly to 0 over the run instead of holding it constant
MUON_MOMENTUM_WARMUP_STEPS = 300  # steps to ramp Muon momentum 0.85 -> 0.95
INIT_SCALE = 0.68  # multiplier on the attn/mlp uniform-init bound (1.0 = original baseline scale)
PACK_FACTOR = 8  # dataloader packing-bin multiplier: fetch PACK_FACTOR*SEQ_LEN rows, reshape to PACK_FACTOR*batch rows of SEQ_LEN
USE_VE = True  # value embeddings (ResFormer-style): alternating layers, input-gated mix into attention's V stream
VE_GATE_CHANNELS = 32


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Alternating layers get a value embedding; the last layer always does."""
    return layer_idx % 2 == (n_layer - 1) % 2


WINDOW_PATTERN = "L"    # per-layer attention window pattern, cycled to n_layer length; 'L' = full context, 'S' = short window
SHORT_WINDOW_FRAC = 0.5  # short window size as a fraction of SEQ_LEN (only matters for 'S' layers)
ROPE_BASE = 100000       # rotary embedding base frequency
USE_LOCAL_CONV = True    # cheap depthwise causal conv1d before attention in each block, RWKV/Mamba/Hyena-style
LOCAL_CONV_KERNEL = 4
USE_GATED_MLP = False    # SwiGLU-style: c_fc(x) * silu(c_gate(x)) instead of relu(c_fc(x))**2; adds one weight matrix
USE_WEIGHT_CACHE = True  # cache each Linear's dtype-cast weight once per optimizer step instead of every forward call


class CausalConv(nn.Module):
    """Depthwise causal conv1d: zero-initialized, so the residual branch it's added into starts as a true no-op."""

    def __init__(self, n_embd: int, kernel_size: int):
        super().__init__()
        self.n_embd = n_embd
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(n_embd, 1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        xt = F.pad(xt, (self.kernel_size - 1, 0))
        y = F.conv1d(xt, self.weight.to(dtype=x.dtype), groups=self.n_embd)
        return y.transpose(1, 2)


def _build_window_mask(seq_len: int, window: int, dtype: torch.dtype) -> torch.Tensor:
    """Additive attention mask: 0 where token j is visible to query i (causal + within window), -inf elsewhere."""
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    visible = (j <= i) & (j > i - window)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype)
    mask.masked_fill_(~visible, float("-inf"))
    return mask[None, None, :, :]


# -----------------------------------------------------------------------------
# Device / distributed setup

def neuron_available() -> bool:
    try:
        torch.device("neuron")
        return True
    except Exception:
        return False


def detect_device_type(requested: str = "") -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if neuron_available():
        return "neuron"
    return "cpu"


def init_runtime(device_type: str):
    assert device_type in {"cpu", "cuda", "neuron"}
    torch.manual_seed(42)
    ddp, rank, local_rank, world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    elif ddp and device_type == "neuron":
        dist.init_process_group("neuron")
        device = torch.device("neuron")
    elif ddp:
        dist.init_process_group("gloo")
        device = torch.device("cpu")
    else:
        device = torch.device(device_type)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return ddp, rank, local_rank, world_size, device


def cleanup_runtime() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "neuron" and hasattr(torch, "neuron"):
        torch.neuron.synchronize()


def compute_dtype_for(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "neuron":
        return torch.bfloat16
    return torch.float32


# -----------------------------------------------------------------------------
# Model

@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int


COMPUTE_DTYPE: torch.dtype = torch.bfloat16


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


# -----------------------------------------------------------------------------
# Example: a custom NKI kernel (optional, default OFF)
#
# This is an ILLUSTRATION to get you started writing your own Trainium kernels.
# It implements the MLP activation `relu(x)**2` as a fused NKI kernel. The eager
# baseline does not use it; enable it with `--nki-relu2` (Neuron only).
#
# Three things worth copying from this example:
#   1. The kernel body: load a tile from HBM into SBUF, compute, store back.
#   2. Tiling over the 128-partition limit: a NeuronCore processes at most 128
#      rows (the "partition" dim) at once, so we loop over chunks of 128 rows.
#   3. Autograd: a raw NKI kernel is NOT differentiable. To use one inside the
#      training forward pass you must wrap it in a torch.autograd.Function and
#      provide the backward yourself, or gradients silently stop flowing.
#
# To write your own: change the per-tile math in `_relu_sq_fwd` (and the matching
# derivative in `_relu_sq_bwd`), or fuse a different elementwise op.

_NKI_PARTITION = 128  # NeuronCore SBUF partition limit (== nki.language.tile_size.pmax)

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def _relu_sq_fwd(x):
        """Fused y = relu(x)**2 for a 2-D [M, N] tensor, tiled over rows."""
        M, N = x.shape
        out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
        for t in nl.affine_range((M + _NKI_PARTITION - 1) // _NKI_PARTITION):
            rows = nl.ds(t * _NKI_PARTITION, _NKI_PARTITION)
            xb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[rows])
            rb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rb, data=xb, op0=nl.maximum, operand0=0.0)  # relu
            yb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yb, data1=rb, data2=rb, op=nl.multiply)     # square
            nisa.dma_copy(dst=out[rows], src=yb)
        return out

    @nki.jit
    def _relu_sq_bwd(x, grad_out):
        """Backward of relu(x)**2: grad_in = grad_out * 2 * relu(x)."""
        M, N = x.shape
        out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
        for t in nl.affine_range((M + _NKI_PARTITION - 1) // _NKI_PARTITION):
            rows = nl.ds(t * _NKI_PARTITION, _NKI_PARTITION)
            xb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[rows])
            gb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=gb, src=grad_out[rows])
            rb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rb, data=xb, op0=nl.maximum, operand0=0.0)   # relu(x)
            two_r = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=two_r, data=rb, op0=nl.multiply, operand0=2.0)
            yb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yb, data1=two_r, data2=gb, op=nl.multiply)
            nisa.dma_copy(dst=out[rows], src=yb)
        return out

    class _ReluSquaredNKI(torch.autograd.Function):
        """Makes the NKI relu**2 kernel differentiable so it can train."""

        @staticmethod
        def forward(ctx, x):
            y = _relu_sq_fwd(x.reshape(-1, x.shape[-1])).reshape(x.shape)
            ctx.save_for_backward(x)
            return y

        @staticmethod
        def backward(ctx, grad_out):
            (x,) = ctx.saved_tensors
            flat_x = x.reshape(-1, x.shape[-1])
            flat_g = grad_out.reshape(-1, x.shape[-1]).contiguous()
            return _relu_sq_bwd(flat_x, flat_g).reshape(x.shape)

    _HAS_NKI = True
except Exception:  # nki unavailable (e.g. CPU/GPU dev box) — eager path still works
    _HAS_NKI = False

# Toggled by the --nki-relu2 flag in main(). Default OFF: the shipped baseline is
# numerically identical to the plain eager `relu(x)**2`.
USE_NKI_RELU2 = False


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """relu(x)**2, via the example NKI kernel when enabled, else eager."""
    # The example kernel tiles rows in blocks of 128, so it needs a row count that
    # is a multiple of 128 (true for the baseline: B*T = 1*1024). Otherwise fall
    # back to eager so the knob is always safe to flip.
    if USE_NKI_RELU2 and _HAS_NKI and x.device.type == "neuron" and (x.numel() // x.shape[-1]) % _NKI_PARTITION == 0:
        return _ReluSquaredNKI.apply(x)
    return F.relu(x).square()


class Linear(nn.Linear):
    """Linear without bias that casts weight to input dtype (allows bf16 input with f32 weight).

    When USE_WEIGHT_CACHE is on, the cast copy is refreshed once per optimizer step
    (see refresh_weight_cache, called from the training loop) instead of being
    recomputed on every forward call across the grad-accum inner loop.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_weight: torch.Tensor | None = None

    def refresh_weight_cache(self, dtype: torch.dtype) -> None:
        self._cached_weight = self.weight.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if USE_WEIGHT_CACHE and self._cached_weight is not None and self._cached_weight.dtype == x.dtype:
            return F.linear(x, self._cached_weight, None)
        return F.linear(x, self.weight.to(dtype=x.dtype), None)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim * self.n_head == self.n_embd
        self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate = Linear(VE_GATE_CHANNELS, self.n_head, bias=False) if USE_VE and has_ve(layer_idx, config.n_layer) else None

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_head, self.head_dim)
        v = self.c_v(x).view(b, t, self.n_head, self.head_dim)
        if ve is not None:
            ve = ve.view(b, t, self.n_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :VE_GATE_CHANNELS]))
            v = v + gate.unsqueeze(-1) * ve
        q = norm(apply_rotary_emb(q, cos, sin))
        k = norm(apply_rotary_emb(k, cos, sin))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[..., :t, :t])
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_gate = Linear(config.n_embd, 4 * config.n_embd, bias=False) if USE_GATED_MLP else None
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c_gate is not None:
            return self.c_proj(self.c_fc(x) * F.silu(self.c_gate(x)))
        return self.c_proj(relu_squared(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.resid_lambda = nn.Parameter(torch.ones(1))
        # x0 skip: mixes initial embedding into each layer's input (stabilizes deep nets)
        self.x0_lambda = nn.Parameter(torch.zeros(1))
        self.local_conv = CausalConv(config.n_embd, LOCAL_CONV_KERNEL) if USE_LOCAL_CONV else None

    def forward(self, x: torch.Tensor, x0: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.resid_lambda * x + self.x0_lambda * x0
        if self.local_conv is not None:
            x = x + self.local_conv(norm(x))
        x = x + self.attn(norm(x), ve, cos, sin, attn_mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, config.n_embd)
            for i in range(config.n_layer) if USE_VE and has_ve(i, config.n_layer)
        })
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, config.n_embd // config.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        pattern = (WINDOW_PATTERN * (config.n_layer // max(1, len(WINDOW_PATTERN)) + 1))[:config.n_layer]
        self.layer_is_short = [c == "S" for c in pattern]
        short_window = max(1, int(config.sequence_len * SHORT_WINDOW_FRAC))
        self.register_buffer("mask_full", _build_window_mask(config.sequence_len, config.sequence_len, torch.float32), persistent=False)
        self.register_buffer("mask_short", _build_window_mask(config.sequence_len, short_window, torch.float32), persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = INIT_SCALE * math.sqrt(3.0) * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
            if block.mlp.c_gate is not None:
                torch.nn.init.uniform_(block.mlp.c_gate.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            block.resid_lambda.fill_(1.0)
            block.x0_lambda.fill_(0.05)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, self.config.n_embd // self.config.n_head)
        self.cos.copy_(cos)
        self.sin.copy_(sin)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = ROPE_BASE):
        device = self.transformer.wte.weight.device
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (freqs / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        table = torch.outer(t, inv_freq)
        cos = table.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = table.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        return cos, sin

    def get_device(self) -> torch.device:
        return self.transformer.wte.weight.device

    def estimate_flops(self) -> int:
        """Estimate FLOPs per token (forward + backward). Attention QK^T and AV are O(T) per token."""
        nparams = sum(p.numel() for p in self.parameters())
        non_matmul = self.transformer.wte.weight.numel() + 2 * self.config.n_layer  # per-block scalars
        h = self.config.n_head
        d = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        # QK^T: 2*h*d*T per token, AV: 2*h*d*T per token, x3 for fwd+bwd
        attn_flops = self.config.n_layer * 12 * h * d * t
        return 6 * (nparams - non_matmul) + attn_flops

    def setup_optimizer(self):
        scalar_params = []
        matrix_params = []
        for block in self.transformer.h:
            scalar_params.extend([block.resid_lambda, block.x0_lambda])
            matrix_params.extend(block.attn.parameters())
            matrix_params.extend(block.mlp.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        value_embed_params = list(self.value_embeds.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=scalar_params, lr=SCALAR_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if value_embed_params:
            param_groups.append(dict(kind="adamw", params=value_embed_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.0))
        for shape in sorted({p.shape for p in matrix_params}):
            params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind="muon", params=params, lr=MATRIX_LR, momentum=0.95, beta2=0.90, ns_steps=5, weight_decay=WEIGHT_DECAY))
        opt = MuonAdamW(param_groups)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        return opt

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean"):
        _b, t = idx.shape
        assert t <= self.config.sequence_len
        cos, sin = self.cos[:, :t], self.sin[:, :t]
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](idx).to(COMPUTE_DTYPE) if str(i) in self.value_embeds else None
            attn_mask = self.mask_short if self.layer_is_short[i] else self.mask_full
            x = block(x, x0, ve, cos, sin, attn_mask)
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        if loss_reduction == "none":
            return loss.view_as(targets)
        return loss


def build_config(depth: int = DEPTH) -> GPTConfig:
    base_dim = depth * ASPECT_RATIO
    n_embd = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    n_head = n_embd // HEAD_DIM
    return GPTConfig(
        sequence_len=SEQ_LEN,
        vocab_size=TOKENIZER_VOCAB_SIZE,
        n_layer=depth,
        n_head=n_head,
        n_embd=n_embd,
    )


# -----------------------------------------------------------------------------
# Optimizer

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def _adamw_step(self, group):
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            p.mul_(1 - group["lr"] * group["weight_decay"])
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            update = exp_avg / bias1 / ((exp_avg_sq / bias2).sqrt() + group["eps"])
            p.add_(update, alpha=-group["lr"])

    @torch.no_grad()
    def _muon_step(self, group):
        params = group["params"]
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        shape = p0.shape
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if "second_momentum_buffer" not in state:
            state_shape = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=torch.float32, device=p0.device)
        if "stacked_grads" not in state:
            state["stacked_grads"] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if "stacked_params" not in state:
            state["stacked_params"] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)

        g = state["stacked_grads"]
        w = state["stacked_params"]
        for i, p in enumerate(params):
            if p.grad is None:
                g[i].zero_()
            else:
                g[i].copy_(p.grad)
            w[i].copy_(p)
        momentum_buffer = state["momentum_buffer"]
        second = state["second_momentum_buffer"]

        momentum_buffer.lerp_(g, 1 - group["momentum"])
        g = g.lerp_(momentum_buffer, group["momentum"])
        x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        if x.shape[-2] > x.shape[-1]:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                xtx = x.mT @ x
                x = a * x + x @ (b * xtx + c * (xtx @ xtx))
        else:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                xxt = x @ x.mT
                x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
        g = x

        red_dim = -1 if shape[-2] >= shape[-1] else -2
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = g.size(red_dim)
        v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
        second.lerp_(v_mean, 1 - group["beta2"])
        step_size = second.clamp_min(1e-10).rsqrt()
        scaled = (v_mean * red_dim_size) * step_size.square()
        v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
        g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)

        lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
        wd = group["weight_decay"]
        mask = (g * w) >= 0
        w.sub_(lr * g + lr * wd * w * mask)
        torch._foreach_copy_(params, list(w.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._adamw_step(group)
            elif group["kind"] == "muon":
                self._muon_step(group)
            else:
                raise ValueError(f"unknown optimizer kind: {group['kind']}")


def sync_gradients(model: nn.Module) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


# -----------------------------------------------------------------------------
# Checkpoint / scoring hook

def save_checkpoint(path: Path, model: GPT, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": asdict(model.config),
        "meta": meta,
    }
    torch.save(payload, path)


def torch_load_weights(path: str | Path, map_location):
    return torch.load(path, map_location=map_location, weights_only=True)


def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    payload = torch_load_weights(checkpoint_path, map_location="cpu")
    config = GPTConfig(**payload["model_config"])
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Training

def lr_multiplier(step: int, elapsed: float = 0.0, max_train_seconds: float = 1.0) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    if WARMDOWN_RATIO <= 0.0:
        return 1.0
    progress = min(1.0, elapsed / max(1e-6, max_train_seconds))
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def muon_momentum(step: int) -> float:
    frac = min(step / MUON_MOMENTUM_WARMUP_STEPS, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuron Competition R1 baseline")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_NUM_STEPS", NUM_STEPS)))
    parser.add_argument("--max-train-seconds", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS", MAX_TRAIN_SECONDS)))
    parser.add_argument("--out-dir", type=str, default=os.environ.get("NEURON_COMPETITION_R1_OUT_DIR", "out"))
    parser.add_argument("--eval-public", action=argparse.BooleanOptionalAction, default=os.environ.get("NEURON_COMPETITION_R1_EVAL_PUBLIC", "1") != "0")
    parser.add_argument("--eval-tokens", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_EVAL_TOKENS", str(4 * 524288))))
    parser.add_argument("--compile", action="store_true", default=os.environ.get("NEURON_COMPETITION_R1_COMPILE", "0") != "0",
                        help="enable torch.compile on the full model")
    parser.add_argument("--nki-relu2", action="store_true", default=os.environ.get("NEURON_COMPETITION_R1_NKI_RELU2", "0") != "0",
                        help="use the example fused NKI relu(x)**2 kernel in the MLP (Neuron only; illustration)")
    args = parser.parse_args()

    global USE_NKI_RELU2
    USE_NKI_RELU2 = args.nki_relu2

    budget_started = time.monotonic()
    device_type = detect_device_type(args.device_type)
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)

    print0(f"device={device} dtype={COMPUTE_DTYPE} world_size={world_size}")
    print0(f"training target={args.num_steps} steps max_seconds={args.max_train_seconds} official_budget={TRAIN_TIME_BUDGET_SECONDS}s")

    tokenizer = ensure_tokenizer(build_if_missing=False)
    config = build_config()
    print0(f"model_config={json.dumps(asdict(config), indent=2)}")
    model = GPT(config)
    model.init_weights()
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    flops_per_token = model.estimate_flops()
    print0(f"num_params={num_params:,} flops_per_token={flops_per_token:e}")

    orig_model = model
    if args.compile:
        compile_backend = "neuron" if device_type == "neuron" else "inductor"
        model = torch.compile(model, backend=compile_backend, dynamic=False)
        print0(f"torch.compile enabled (backend={compile_backend})")

    world_tokens_per_micro = DEVICE_BATCH_SIZE * PACK_FACTOR * SEQ_LEN * world_size
    if TOTAL_BATCH_SIZE % world_tokens_per_micro != 0:
        raise ValueError(f"TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} must divide micro tokens={world_tokens_per_micro}")
    grad_accum_steps = TOTAL_BATCH_SIZE // world_tokens_per_micro
    print0(f"micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps} pack_factor={PACK_FACTOR}")

    loader_seq_len = PACK_FACTOR * SEQ_LEN
    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, "train", device)

    def fetch():
        xb, yb, state = next(train_loader)
        if PACK_FACTOR > 1:
            xb = xb.reshape(DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN)
            yb = yb.reshape(DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN)
        return xb, yb, state

    x, y, loader_state = fetch()
    optimizer = model.setup_optimizer()

    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0

    while step < args.num_steps:
        elapsed = time.monotonic() - budget_started
        if elapsed >= args.max_train_seconds:
            break

        synchronize(device)
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        for _ in range(grad_accum_steps):
            loss = model(x, y)
            last_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, loader_state = fetch()

        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        sync_gradients(model)

        lrm = lr_multiplier(step, elapsed=elapsed, max_train_seconds=args.max_train_seconds)
        wd_progress = min(1.0, elapsed / max(1e-6, args.max_train_seconds))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum(step)
                group["weight_decay"] = WEIGHT_DECAY * (1.0 - wd_progress) if WD_DECAY_TO_ZERO else WEIGHT_DECAY
        optimizer.step()
        if USE_WEIGHT_CACHE:
            for m in model.modules():
                if isinstance(m, Linear):
                    m.refresh_weight_cache(COMPUTE_DTYPE)

        synchronize(device)
        dt = time.monotonic() - t0
        train_loss = float(last_loss.item())
        bad = torch.tensor(1 if (math.isnan(train_loss) or train_loss > 100) else 0, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if int(bad.item()) != 0:
            raise RuntimeError(f"bad loss detected: {train_loss}")

        total_train_tokens += TOTAL_BATCH_SIZE
        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        tok_per_sec = TOTAL_BATCH_SIZE / max(dt, 1e-6)
        if rank == 0 and (step < 5 or step % 10 == 0):
            remaining_steps = args.num_steps - step - 1
            remaining_time = max(0.0, args.max_train_seconds - (time.monotonic() - budget_started))
            print(
                f"step {step:05d} | loss {debiased:.4f} | lrm {lrm:.3f} | "
                f"dt {dt:.2f}s | tok/s {tok_per_sec:,.0f} | remaining_steps {remaining_steps} | "
                f"remaining_time {remaining_time:.0f}s | "
                f"epoch {loader_state['epoch']}",
                flush=True,
            )

        if step == 0:
            # After first step all persistent objects are allocated; freeze to avoid GC pauses
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1

    checkpoint_path = Path(args.out_dir) / "final.pt"
    budget_elapsed_before_save = time.monotonic() - budget_started
    if rank == 0:
        save_checkpoint(checkpoint_path, orig_model, {
            "step": step,
            "total_train_tokens": total_train_tokens,
            "budget_elapsed_before_save": budget_elapsed_before_save,
            "world_size": world_size,
            "compiled": args.compile,
        })
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    budget_elapsed_after_save = time.monotonic() - budget_started

    public_eval = None
    causal = None
    if args.eval_public:
        causal = causality_check(orig_model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, device)
        public_eval = evaluate_bpb(
            orig_model,
            tokenizer,
            DEVICE_BATCH_SIZE,
            SEQ_LEN,
            device,
            split="public_val",
            max_tokens=args.eval_tokens,
            timeout_seconds=EVAL_TIMEOUT_SECONDS,
        )

    if rank == 0:
        print("---")
        print(f"checkpoint_path:    {checkpoint_path}")
        print(f"training_seconds:  {budget_elapsed_before_save:.1f}")
        print(f"budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}")
        print(f"target_steps:      {args.num_steps}")
        print(f"max_train_seconds: {args.max_train_seconds}")
        print(f"over_budget:       {budget_elapsed_after_save > TRAIN_TIME_BUDGET_SECONDS}")
        print(f"total_seconds:     {budget_elapsed_after_save:.1f}")
        print(f"compiled:          {args.compile}")
        print(f"nki_relu2:         {args.nki_relu2}")
        print(f"num_steps:         {step}")
        print(f"total_tokens_M:    {total_train_tokens / 1e6:.1f}")
        print(f"world_size:        {world_size}")
        print(f"lnc:               {os.environ.get('NEURON_LOGICAL_NC_CONFIG', '')}")
        print(f"seq_len:           {SEQ_LEN}")
        print(f"num_params_M:      {num_params / 1e6:.1f}")
        if causal is not None:
            print(f"causality_passed:  {causal['passed']}")
            print(f"causality_max_abs: {causal['max_abs_diff']:.6f}")
        if public_eval is not None:
            print(f"public_val_bpb:    {public_eval['val_bpb']:.6f}")
            print(f"eval_seconds:      {public_eval['eval_seconds']:.1f}")
            print(f"eval_timeout:      {public_eval['eval_timeout']}")
            print(f"eval_over_timeout: {public_eval['eval_over_timeout']}")

    cleanup_runtime()


if __name__ == "__main__":
    main()
