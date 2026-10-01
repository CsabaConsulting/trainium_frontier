"""
One-hot-matmul embedding backward (pure PyTorch, no NKI).

Forward is the normal gather. Backward replaces the scatter-add with
    dW = onehot(idx)^T @ grad
as chunked TensorEngine matmuls, accumulated in fp32.

Tables that share the same indices (wte + value embeddings) go through ONE one-hot
build and ONE wider matmul (grads concatenated along D).

Env flags
  ONEHOT_EMB_BWD = 0 | fn | op   0  = stock F.embedding (default; flag-off identity)
                                 fn = torch.autograd.Function -- Dynamo traces THROUGH it, so the
                                      Neuron backend sees plain compare/cast/matmul. Preferred.
                                 op = torch.library.custom_op + register_autograd -- opaque to the
                                      compiler; on the Neuron backend expect a graph break / fallback.
  ONEHOT_CHUNK   = 8192          tokens per chunk (one-hot chunk = V x chunk bf16 = 128 MiB at V=8192)
  ONEHOT_DTYPE   = bf16 | fp32   matmul input dtype. fp32 = exactness reference, bf16 = fast path.

Usage in train.py:
    tok, ve = embed(idx, self.wte.weight, self.ve.weight)   # tables sharing idx -> tuple
    (tok,)  = embed(idx, self.wte.weight)
    (big,)  = embed(bigram_idx, self.bigram.weight)          # different idx -> separate call
"""
import os
from typing import List, Sequence

import torch
import torch.nn.functional as F

MODE = os.environ.get("ONEHOT_EMB_BWD", "0")
CHUNK = int(os.environ.get("ONEHOT_CHUNK", "8192"))
MM_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("ONEHOT_DTYPE", "bf16")]


def onehot_grad(idx: torch.Tensor, grads: Sequence[torch.Tensor], num_embeddings: int,
                chunk: int = None, mm_dtype: torch.dtype = None) -> List[torch.Tensor]:
    """idx: (...) int; grads: tensors (..., D_i). Returns one fp32 (V, D_i) per grad."""
    chunk = chunk or CHUNK
    mm_dtype = mm_dtype or MM_DTYPE
    flat = idx.reshape(-1)
    T = flat.numel()
    widths = [g.shape[-1] for g in grads]
    g = torch.cat([gi.reshape(T, -1) for gi in grads], dim=1)              # (T, sum D)
    # One-hot built already transposed, (V, c): no transpose in the graph, no int64 F.one_hot.
    vocab = torch.arange(num_embeddings, device=flat.device, dtype=flat.dtype).unsqueeze(1)
    acc = torch.zeros(num_embeddings, g.shape[1], device=g.device, dtype=torch.float32)
    for s in range(0, T, chunk):                                            # static trip count -> unrolled
        e = min(s + chunk, T)
        oh_t = (vocab == flat[s:e].unsqueeze(0)).to(mm_dtype)               # (V, c)
        acc = acc + torch.mm(oh_t, g[s:e].to(mm_dtype)).float()             # out-of-place, compile-friendly
    return list(acc.split(widths, dim=1))


# ---------------------------------------------------------------- autograd.Function path
class _OneHotEmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, idx, *weights):
        ctx.save_for_backward(idx)
        ctx.V = weights[0].shape[0]
        ctx.meta = [(w.dtype, w.shape[1]) for w in weights]
        return tuple(F.embedding(idx, w) for w in weights)

    @staticmethod
    def backward(ctx, *gouts):
        (idx,) = ctx.saved_tensors
        gouts = [g if g is not None else torch.zeros(*idx.shape, d, device=idx.device)
                 for g, (_, d) in zip(gouts, ctx.meta)]
        dws = onehot_grad(idx, gouts, ctx.V)
        return (None, *[dw.to(dt) for dw, (dt, _) in zip(dws, ctx.meta)])


# ---------------------------------------------------------------- custom_op path (one table per call)
@torch.library.custom_op("onehot::embed_bwd", mutates_args=())
def _embed_bwd_op(idx: torch.Tensor, grad: torch.Tensor, num_embeddings: int) -> torch.Tensor:
    return onehot_grad(idx, [grad], num_embeddings)[0]


@_embed_bwd_op.register_fake
def _(idx, grad, num_embeddings):
    return grad.new_empty(num_embeddings, grad.shape[-1], dtype=torch.float32)


@torch.library.custom_op("onehot::embed", mutates_args=())
def _embed_op(idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.embedding(idx, weight)


@_embed_op.register_fake
def _(idx, weight):
    return weight.new_empty(*idx.shape, weight.shape[1])


def _setup(ctx, inputs, output):
    idx, weight = inputs
    ctx.save_for_backward(idx)
    ctx.V, ctx.wdtype = weight.shape[0], weight.dtype


def _bwd(ctx, grad):
    (idx,) = ctx.saved_tensors
    return None, _embed_bwd_op(idx, grad, ctx.V).to(ctx.wdtype)


_embed_op.register_autograd(_bwd, setup_context=_setup)


# ---------------------------------------------------------------- entry point
def embed(idx: torch.Tensor, *weights: torch.Tensor, mode: str = None):
    mode = MODE if mode is None else mode
    if mode == "0":
        return tuple(F.embedding(idx, w) for w in weights)
    if mode == "fn":
        return _OneHotEmbeddingFn.apply(idx, *weights)
    if mode == "op":
        return tuple(_embed_op(idx, w) for w in weights)
    raise ValueError(f"ONEHOT_EMB_BWD={mode!r}")