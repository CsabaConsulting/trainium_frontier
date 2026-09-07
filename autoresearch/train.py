"""
Autoresearch: 5-minute single-core training for rapid iteration.
Edit hyperparameters below, run `python train.py`, compare val_bpb.
"""

import os
os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --optlevel=3 --enable-saturate-infinity --enable-fast-loading-neuron-binaries"
# Stochastic rounding turned OFF (was unconditionally on): the main train.py found this
# permanently disqualified for a real submission -- causality_check's fixed tolerance has
# a structural blind spot for SR's own hardware forward-noise, a false positive but a real
# submission would hit the identical risk regardless of any speed benefit. Leaving it on
# here would make any "improvement" this loop finds misleading (wouldn't transfer).

import gc
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TOKENIZER_VOCAB_SIZE,
    ensure_tokenizer,
    evaluate_bpb,
    make_dataloader,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

SEQ_LEN = 2048  # locked, see ../research/trn2-experiment-results.md (real interior optimum, robust to torch.compile per the Tier-1 spot-check)
DEPTH = 6  # locked, real interior optimum, robust to torch.compile. Reverted back from an UNCONFIRMED DEPTH=4 5-min-proxy "keep" (see results.tsv) -- that capacity-affecting change was never rechecked at a longer budget before the session paused, so it should not be left in place confounding new (non-capacity) experiments.
ASPECT_RATIO = 96  # locked, real interior optimum
HEAD_DIM = 128

DEVICE_BATCH_SIZE = 16
TOTAL_BATCH_SIZE = 131072  # locked, real optimum (re-validated at n_embd=640)
TIME_BUDGET = 300

LOGIT_SOFTCAP = 11.0
ROPE_BASE = 2_000_000

EMBEDDING_LR = 0.2
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.3
WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.75, 0.99)
MUON_MOMENTUM = 0.85
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.0
FINAL_LR_FRAC = 0.02

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 8192
    n_layer: int = 10
    n_head: int = 5
    n_embd: int = 640


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos, sin, v0=None, v0_lambda=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v_raw = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        # EXPERIMENT: value-residual learning (Zhou et al.) -- inject layer-0's raw value
        # stream into every later layer's V via a learned per-layer scalar, zero-init so the
        # model starts identical to baseline. Cheap: one elementwise add of an
        # already-computed tensor, no extra matmuls, so unlike the other structural
        # experiments this shouldn't cost step throughput in the fixed-time budget.
        v = v_raw if v0 is None else v_raw + v0_lambda * v0
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), v_raw


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x, h0=None, h0_lambda=None):
        h_raw = self.c_fc(x)
        # EXPERIMENT: hidden-residual learning for the MLP, mirroring the confirmed
        # value-residual keep (val_bpb 1.405->1.371) -- inject layer-0's raw MLP
        # pre-activation into every later layer's MLP via a learned per-layer scalar,
        # zero-init. Same "free" pattern: reuses an already-computed tensor, no new matmuls.
        h = h_raw if h0 is None else h_raw + h0_lambda * h0
        return self.c_proj(F.relu(h).square()), h_raw


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, v0=None, v0_lambda=None, h0=None, h0_lambda=None):
        attn_out, v_raw = self.attn(norm(x), cos, sin, v0, v0_lambda)
        x = x + attn_out
        mlp_out, h_raw = self.mlp(norm(x), h0, h0_lambda)
        x = x + mlp_out
        return x, v_raw, h_raw


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.v_lambda = nn.Parameter(torch.zeros(config.n_layer))  # value-residual learning, index 0 unused
        self.h_lambda = nn.Parameter(torch.zeros(config.n_layer))  # MLP hidden-residual learning, index 0 unused
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary(config.sequence_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, device=None):
        if device is None:
            device = 'cpu'
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (ROPE_BASE ** (freqs / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        table = torch.outer(t, inv_freq)
        cos = table.cos().bfloat16()[None, :, None, :]
        sin = table.sin().bfloat16()[None, :, None, :]
        return cos.to(device), sin.to(device)

    @torch.no_grad()
    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        device = self.transformer.wte.weight.device
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.5)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for block in self.transformer.h:
            for name in ['c_q', 'c_k', 'c_v']:
                torch.nn.init.uniform_(getattr(block.attn, name).weight, -s, s)
            torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=0.01)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.data.fill_(1.0)
        self.x0_lambdas.data.fill_(0.05)

    def setup_optimizer(self):
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        scalar_params = [self.resid_lambdas, self.x0_lambdas, self.v_lambda, self.h_lambda]
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        param_groups = [
            dict(kind='adamw', params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=scalar_params, lr=SCALAR_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=MATRIX_LR,
                momentum=MUON_MOMENTUM, ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY,
            ))
        opt = MuonAdamW(param_groups)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        return opt

    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x0 = x
        v0, h0 = None, None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x, v_raw, h_raw = block(
                x, cos, sin,
                v0, self.v_lambda[i] if i > 0 else None,
                h0, self.h_lambda[i] if i > 0 else None,
            )
            if v0 is None:
                v0, h0 = v_raw, h_raw
        x = norm(x)
        logits = self.lm_head(x)
        logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

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
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._adamw_step(group)
            elif group['kind'] == 'muon':
                self._muon_step(group)

    def _adamw_step(self, group):
        beta1, beta2 = group['betas']
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
            state['step'] += 1
            grad = p.grad.float()
            p.mul_(1 - group['lr'] * group['weight_decay'])
            state['exp_avg'].lerp_(grad, 1 - beta1)
            state['exp_avg_sq'].lerp_(grad.square(), 1 - beta2)
            bias1 = 1 - beta1 ** state['step']
            bias2 = 1 - beta2 ** state['step']
            update = state['exp_avg'] / bias1 / ((state['exp_avg_sq'] / bias2).sqrt() + group['eps'])
            p.add_(update.to(p.dtype), alpha=-group['lr'])

    def _muon_step(self, group):
        params = group['params']
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        shape = p0.shape
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros(len(params), *shape, dtype=torch.float32, device=p0.device)
            state_shape = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state['second_momentum_buffer'] = torch.zeros(state_shape, dtype=torch.float32, device=p0.device)

        g = torch.stack([p.grad for p in params]).float()
        w = torch.stack(list(params))
        mb = state['momentum_buffer']
        second = state['second_momentum_buffer']

        mb.lerp_(g, 1 - group['momentum'])
        g = g.lerp_(mb, group['momentum'])
        x = g.bfloat16()
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        if x.shape[-2] > x.shape[-1]:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group['ns_steps']]:
                xtx = x.mT @ x
                x = a * x + x @ (b * xtx + c * (xtx @ xtx))
        else:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group['ns_steps']]:
                xxt = x @ x.mT
                x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
        g = x

        red_dim = -1 if shape[-2] >= shape[-1] else -2
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = g.size(red_dim)
        v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
        second.lerp_(v_mean, 1 - group['beta2'])
        step_size = second.clamp_min(1e-10).rsqrt()
        scaled = (v_mean * red_dim_size) * step_size.square()
        v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
        g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)

        lr = group['lr'] * max(1.0, shape[-2] / shape[-1]) ** 0.5
        wd = group['weight_decay']
        mask = (g * w) >= 0
        w.sub_(lr * g + lr * wd * w * mask)
        torch._foreach_copy_(params, list(w.unbind(0)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    device = torch.device("neuron")

    tokenizer = ensure_tokenizer(build_if_missing=False)
    config = GPTConfig(
        sequence_len=SEQ_LEN,
        vocab_size=TOKENIZER_VOCAB_SIZE,
        n_layer=DEPTH,
        n_head=((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM),
        n_embd=((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM,
    )
    print(f"config: {asdict(config)}")

    model = GPT(config)
    model.init_weights()
    model.to(device).to(torch.bfloat16)
    torch.neuron.synchronize()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"params: {num_params:,}")

    tokens_per_micro = DEVICE_BATCH_SIZE * SEQ_LEN
    grad_accum_steps = max(1, TOTAL_BATCH_SIZE // tokens_per_micro)
    print(f"grad_accum_steps: {grad_accum_steps}, effective batch: {grad_accum_steps * tokens_per_micro}")

    optimizer = model.setup_optimizer()
    loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, "train", device)
    x, y, _ = next(loader)

    training_time = 0
    step = 0

    while True:
        torch.neuron.synchronize()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            loss = model(x, y)
            (loss / grad_accum_steps).backward()
            x, y, _ = next(loader)

        progress = min(training_time / TIME_BUDGET, 1.0)
        warmdown_start = max(WARMUP_RATIO, 1.0 - WARMDOWN_RATIO)
        if progress < WARMUP_RATIO:
            lrm = progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        elif progress > warmdown_start:
            cooldown = (1.0 - progress) / (1.0 - warmdown_start) if warmdown_start < 1.0 else 1.0
            lrm = cooldown + (1 - cooldown) * FINAL_LR_FRAC
        else:
            lrm = 1.0
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * lrm

        optimizer.step()

        torch.neuron.synchronize()
        dt = time.time() - t0

        if step > 5:
            training_time += dt

        if step % 50 == 0:
            print(f"step {step:04d} | loss {loss.item():.4f} | dt {dt*1000:.0f}ms | remaining {max(0, TIME_BUDGET - training_time):.0f}s")

        if step == 0:
            gc.collect()
            gc.disable()
        step += 1

        if step > 5 and training_time >= TIME_BUDGET:
            break

    print(f"\nTraining done: {step} steps, {training_time:.1f}s")

    model.eval()
    del x, y, optimizer
    gc.enable()
    gc.collect()
    torch.neuron.synchronize()

    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, device, split="public_val", max_tokens=4*524288)
    print("---")
    print(f"val_bpb: {val_bpb['val_bpb']:.6f}")
    print(f"steps:   {step}")
    print(f"params:  {num_params:,}")


if __name__ == "__main__":
    main()
