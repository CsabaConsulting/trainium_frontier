"""Trainium Frontier Phase 1: dense GPT, 6 layers x 1024 wide, trained from scratch in 1,800 s.

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
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
_PROCESS_STARTED = time.monotonic()
if __name__ == '__main__':
    os.environ.setdefault('NEURON_CC_FLAGS', '--optlevel=1 --auto-cast matmult --auto-cast-type bf16')
os.environ.setdefault('NEURON_COMPILE_CACHE_DIR', '/tmp/neuron_cache')
os.environ.setdefault('NEURON_CC_FLAGS', '--optlevel=1')
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from prepare import TOKENIZER_VOCAB_SIZE, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS, TRAIN_STARTUP_STEPS_EXCLUDED, TRAIN_TIME_BUDGET_SECONDS, ensure_tokenizer, get_dist_info, make_dataloader, print0, record_step_boundary
SEQ_LEN = 1024
DEPTH = 6
HEAD_DIM = 128
N_EMBD = 1024
MLP_RATIO = 4
DEVICE_BATCH_SIZE = 1
TOTAL_BATCH_SIZE = 131072
NUM_STEPS = 1000000
MAX_TRAIN_SECONDS = 1800
CHECKPOINT_RESERVE_SECONDS = 15.0
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0
LOGIT_SOFTCAP = 15.0
EMBEDDING_LR = 0.3
UNEMBEDDING_LR = 0.006
MATRIX_LR = 0.015
SCALAR_LR = 0.2
SCALAR_BETA1 = 0.9
SCALAR_BETA2 = 0.95
WEIGHT_DECAY = 0.2
EMBED_ADAMW_BETA2 = 0.95
WARMUP_STEPS = 0
GRAD_CLIP = 1.0
WARMDOWN_RATIO = 0.75
FINAL_LR_FRAC = 0.05
MUON_MOMENTUM_WARMUP_STEPS = 300
MUON_MOMENTUM_COOLDOWN_FRAC = 0.045
MUON_MOMENTUM_FINAL = 0.85
MUON_NS_STEPS = 5
MUON_BETA2 = 0.9
INIT_SCALE = 0.5
ATTN_CPROJ_INIT_SCALE = 0.1
PACK_FACTOR = 32
VE_GATE_CHANNELS = 32
ROPE_BASE = 100000
LOCAL_CONV_KERNEL = 2
BOS_TOKEN_ID = -1

class CausalConv(nn.Module):
    """Depthwise causal conv1d: zero-initialized, so the residual branch it's added into starts as a true no-op."""

    def __init__(self, n_embd: int, kernel_size: int):
        super().__init__()
        self.n_embd = n_embd
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(n_embd, 1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == 'neuron' and x.dtype in (torch.float32, torch.bfloat16) and (self.kernel_size == 2) and (x.shape[1] % 128 == 0) and (x.shape[2] % 128 == 0):
            if not _HAS_NKI:
                raise RuntimeError('--nki-local-conv requires the installed native NKI stack')
            return _Conv2NKI.apply(x, self.weight)
        xt = x.transpose(1, 2)
        xt = F.pad(xt, (self.kernel_size - 1, 0))
        y = F.conv1d(xt, self.weight.to(dtype=x.dtype), groups=self.n_embd)
        return y.transpose(1, 2)

def neuron_available() -> bool:
    try:
        torch.device('neuron')
        return True
    except Exception:
        return False

def detect_device_type(requested: str='') -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if neuron_available():
        return 'neuron'
    return 'cpu'

def init_runtime(device_type: str, seed: int=42):
    assert device_type in {'cpu', 'cuda', 'neuron'}
    torch.manual_seed(seed)
    ddp, rank, local_rank, world_size = get_dist_info()
    if ddp and device_type == 'cuda':
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group('nccl', device_id=device)
    elif ddp and device_type == 'neuron':
        dist.init_process_group('neuron')
        device = torch.device('neuron')
    elif ddp:
        dist.init_process_group('gloo')
        device = torch.device('cpu')
    else:
        device = torch.device(device_type)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return (ddp, rank, local_rank, world_size, device)

def cleanup_runtime() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'neuron' and hasattr(torch, 'neuron'):
        torch.neuron.synchronize()

def compute_dtype_for(device: torch.device) -> torch.dtype:
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == 'neuron':
        return torch.bfloat16
    return torch.float32

@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    mlp_ratio: int = 4
COMPUTE_DTYPE: torch.dtype = torch.bfloat16

def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x.float(), (x.size(-1),), eps=_RMS_EPS).to(torch.bfloat16)
_NKI_PARTITION = 128

def _ce_lanes(rows: int) -> int:
    """Row-tiling lanes for the softcap cross-entropy kernel (rows must be a positive multiple of 128)."""
    if rows <= 0 or rows % 128:
        raise ValueError('the softcap cross-entropy kernel needs a positive multiple of 128 rows')
    return 2 if rows >= 512 and rows % 256 == 0 else 1
_RMS_EPS = 1.1920928955078125e-07
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def _rope_norm_fwd(x, cos, sin):
        """Fused rms_norm(rope(x)) for x [BT, H, D], cos/sin [T, D//2]."""
        BT, H, D = x.shape
        T = cos.shape[0]
        Dh = D // 2
        dt = x.dtype
        P = _NKI_PARTITION
        BC = [[H, P], [1, H], [0, D]]
        CS = [[Dh, P], [0, H], [1, Dh]]
        out = nl.ndarray((BT, H, D), dtype=dt, buffer=nl.shared_hbm)
        for i in nl.affine_range(BT // P):
            r0 = i * P
            t0 = i * P % T
            xb = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[r0:r0 + P])
            cb = nl.ndarray((P, H, Dh), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=cb, src=cos.ap(pattern=CS, offset=t0 * Dh))
            sb = nl.ndarray((P, H, Dh), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=sb, src=sin.ap(pattern=CS, offset=t0 * Dh))
            x1, x2 = (xb[0:P, 0:H, 0:Dh], xb[0:P, 0:H, Dh:D])
            zb = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            tb = nl.ndarray((P, H, Dh), dtype=dt, buffer=nl.sbuf)
            z1, z2 = (zb[0:P, 0:H, 0:Dh], zb[0:P, 0:H, Dh:D])
            nisa.tensor_tensor(dst=z1, data1=x1, data2=cb, op=nl.multiply)
            nisa.tensor_tensor(dst=tb, data1=x2, data2=sb, op=nl.multiply)
            nisa.tensor_tensor(dst=z1, data1=z1, data2=tb, op=nl.add)
            nisa.tensor_tensor(dst=z2, data1=x2, data2=cb, op=nl.multiply)
            nisa.tensor_tensor(dst=tb, data1=x1, data2=sb, op=nl.multiply)
            nisa.tensor_tensor(dst=z2, data1=z2, data2=tb, op=nl.subtract)
            zf = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=zf, src=zb)
            sq = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=zf, data2=zf, op=nl.multiply)
            ss = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=(2,))
            ms = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ms, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=_RMS_EPS)
            r2 = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=r2, data=ms)
            rr = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rr, op=nl.sqrt, data=r2)
            ob = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=ob, data1=zf, data2=rr.ap(pattern=BC), op=nl.multiply)
            nisa.dma_copy(dst=out[r0:r0 + P], src=ob)
        return out

    @nki.jit
    def _rope_norm_bwd(x, g, cos, sin):
        """Gradient of _rope_norm_fwd w.r.t. x (cos/sin are constants)."""
        BT, H, D = x.shape
        T = cos.shape[0]
        Dh = D // 2
        dt = x.dtype
        P = _NKI_PARTITION
        BC = [[H, P], [1, H], [0, D]]
        CS = [[Dh, P], [0, H], [1, Dh]]
        dx = nl.ndarray((BT, H, D), dtype=dt, buffer=nl.shared_hbm)
        for i in nl.affine_range(BT // P):
            r0 = i * P
            t0 = i * P % T
            xb = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[r0:r0 + P])
            gf = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gf, src=g[r0:r0 + P])
            cb = nl.ndarray((P, H, Dh), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=cb, src=cos.ap(pattern=CS, offset=t0 * Dh))
            sb = nl.ndarray((P, H, Dh), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=sb, src=sin.ap(pattern=CS, offset=t0 * Dh))
            x1, x2 = (xb[0:P, 0:H, 0:Dh], xb[0:P, 0:H, Dh:D])
            zb = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            tb = nl.ndarray((P, H, Dh), dtype=dt, buffer=nl.sbuf)
            z1, z2 = (zb[0:P, 0:H, 0:Dh], zb[0:P, 0:H, Dh:D])
            nisa.tensor_tensor(dst=z1, data1=x1, data2=cb, op=nl.multiply)
            nisa.tensor_tensor(dst=tb, data1=x2, data2=sb, op=nl.multiply)
            nisa.tensor_tensor(dst=z1, data1=z1, data2=tb, op=nl.add)
            nisa.tensor_tensor(dst=z2, data1=x2, data2=cb, op=nl.multiply)
            nisa.tensor_tensor(dst=tb, data1=x1, data2=sb, op=nl.multiply)
            nisa.tensor_tensor(dst=z2, data1=z2, data2=tb, op=nl.subtract)
            zf = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=zf, src=zb)
            acc = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=acc, data1=zf, data2=zf, op=nl.multiply)
            ss = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ss, op=nl.add, data=acc, axis=(2,))
            ms = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ms, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=_RMS_EPS)
            r2 = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=r2, data=ms)
            rr = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rr, op=nl.sqrt, data=r2)
            nisa.tensor_tensor(dst=acc, data1=gf, data2=zf, op=nl.multiply)
            pp = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=pp, op=nl.add, data=acc, axis=(2,))
            qq = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=qq, data=pp, op0=nl.multiply, operand0=1.0 / D)
            nisa.tensor_tensor(dst=qq, data1=qq, data2=r2, op=nl.multiply)
            nisa.tensor_tensor(dst=acc, data1=zf, data2=qq.ap(pattern=BC), op=nl.multiply)
            nisa.tensor_tensor(dst=acc, data1=gf, data2=acc, op=nl.subtract)
            dzb = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dzb, data1=acc, data2=rr.ap(pattern=BC), op=nl.multiply)
            d1, d2 = (dzb[0:P, 0:H, 0:Dh], dzb[0:P, 0:H, Dh:D])
            ob = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            o1, o2 = (ob[0:P, 0:H, 0:Dh], ob[0:P, 0:H, Dh:D])
            nisa.tensor_tensor(dst=o1, data1=d1, data2=cb, op=nl.multiply)
            nisa.tensor_tensor(dst=tb, data1=d2, data2=sb, op=nl.multiply)
            nisa.tensor_tensor(dst=o1, data1=o1, data2=tb, op=nl.subtract)
            nisa.tensor_tensor(dst=o2, data1=d2, data2=cb, op=nl.multiply)
            nisa.tensor_tensor(dst=tb, data1=d1, data2=sb, op=nl.multiply)
            nisa.tensor_tensor(dst=o2, data1=o2, data2=tb, op=nl.add)
            nisa.dma_copy(dst=dx[r0:r0 + P], src=ob)
        return dx

    class _RopeNormNKI(torch.autograd.Function):
        """Makes the fused rope+norm kernel differentiable so it can train."""

        @staticmethod
        def forward(ctx, x, cos, sin):
            b, t, h, d = x.shape
            x3 = x.reshape(b * t, h, d)
            cos2 = cos.reshape(-1, d // 2)
            sin2 = sin.reshape(-1, d // 2)
            y = _rope_norm_fwd(x3, cos2, sin2).reshape(b, t, h, d)
            ctx.save_for_backward(x3, cos2, sin2)
            ctx.x_shape = (b, t, h, d)
            return y

        @staticmethod
        def backward(ctx, grad_out):
            x3, cos2, sin2 = ctx.saved_tensors
            b, t, h, d = ctx.x_shape
            g3 = grad_out.reshape(b * t, h, d).contiguous()
            return (_rope_norm_bwd(x3, g3, cos2, sin2).reshape(b, t, h, d), None, None)

    @nki.jit
    def _softcap_ce_fwd(z, tgt, lanes: int=2, target_gather: bool=False):
        """Compute unreduced loss and saved logsumexp with unchanged softcap15.

        Args:
            z: Contiguous FP32 or BF16 [N, V] logits. BF16 is widened in
                SBUF; all loss arithmetic and saved statistics remain FP32.
            tgt: FP32 [N, 1] class indices, exact integers in [0, V).
            lanes: Physical cores in the matching launch grid.
            target_gather: Gather one SBUF element per row instead of mask/sum.
        Returns:
            FP32 [N, 1] loss and logsumexp, as in the original kernel.
        """
        rows_count, vocab = z.shape
        partitions = 128
        softcap = 15.0
        assert lanes in (1, 2) and rows_count % (partitions * lanes) == 0
        assert z.dtype in (nl.float32, nl.bfloat16) and tgt.dtype == nl.float32
        assert tgt.shape == (rows_count, 1) and vocab % 512 == 0
        loss = nl.ndarray((rows_count, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        lse = nl.ndarray((rows_count, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        if not target_gather:
            indices = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            nisa.iota(dst=indices, pattern=[[1, vocab]], offset=0, channel_multiplier=0)
        chunk = 512
        chunks = vocab // chunk
        for tile in nl.affine_range(rows_count // (partitions * lanes)):
            first = (tile * lanes + nl.program_id(0)) * partitions
            rows = nl.ds(first, partitions)
            logits = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            labels = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            if z.dtype == nl.bfloat16:
                source = nl.ndarray((partitions, vocab), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=source, src=z[rows])
                nisa.tensor_copy(dst=logits, src=source)
            else:
                nisa.dma_copy(dst=logits, src=z[rows])
            nisa.dma_copy(dst=labels, src=tgt[rows])
            capped_unit = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=capped_unit, op=nl.tanh, data=logits, scale=1.0 / softcap)
            target_unit = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            if target_gather:
                gather_indices = nl.ndarray((partitions, 1), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gather_indices, src=labels)
                nisa.nc_n_gather(dst=target_unit, data=capped_unit, indices=gather_indices)
            else:
                mask = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=mask, data=indices, op0=nl.equal, operand0=labels)
                nisa.tensor_tensor(dst=mask, data1=mask, data2=capped_unit, op=nl.multiply)
                nisa.tensor_reduce(dst=target_unit, op=nl.add, data=mask, axis=(1,))
            target_value = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            negative_target = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=target_value, data=target_unit, op0=nl.multiply, operand0=softcap)
            nisa.tensor_scalar(dst=negative_target, data=target_unit, op0=nl.multiply, operand0=-softcap)
            partial_sums = nl.ndarray((partitions, chunks), dtype=nl.float32, buffer=nl.sbuf)
            for part in nl.affine_range(chunks):
                columns = nl.ds(part * chunk, chunk)
                nisa.activation(dst=logits[:, columns], op=nl.exp, data=capped_unit[:, columns], scale=softcap, bias=negative_target, reduce_op=nl.add, reduce_res=partial_sums[:, nl.ds(part, 1)], reduce_cmd=nisa.reduce_cmd.reset_reduce)
            denominator = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=denominator, op=nl.add, data=partial_sums, axis=(1,))
            log_denominator = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=log_denominator, op=nl.log, data=denominator)
            logsumexp = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=logsumexp, data1=target_value, data2=log_denominator, op=nl.add)
            nisa.dma_copy(dst=loss[rows], src=log_denominator)
            nisa.dma_copy(dst=lse[rows], src=logsumexp)
        return (loss, lse)

    @nki.jit
    def _softcap_ce_bwd(z, tgt, lse, g, lanes: int=2, round_bf16_output: bool=False):
        """Return [N, V] gradients, normally in the logits dtype.

        Args:
            z, tgt, lse: Same inputs and saved logsumexp as forward.
            g: Arbitrary FP32 [N, 1] upstream loss gradient.
            lanes: Physical cores in the matching launch grid.
            round_bf16_output: Separately gated head-only contract. Preserve
                FP32 logits/arithmetic, but round the output to BF16 for a
                consumer that already has that backward cast boundary.
        Returns:
            Original FP32 softcap15/CE derivative, rounded at the output when
            the input was BF16 or round_bf16_output is enabled. The latter
            requires a matching consumer boundary; it is not exact for an
            arbitrary FP32-logit leaf's gradient.
        """
        rows_count, vocab = z.shape
        partitions = 128
        softcap = 15.0
        assert lanes in (1, 2) and rows_count % (partitions * lanes) == 0
        assert z.dtype in (nl.float32, nl.bfloat16) and tgt.dtype == nl.float32
        assert lse.dtype == nl.float32 and g.dtype == nl.float32
        assert tgt.shape == (rows_count, 1) and lse.shape == (rows_count, 1)
        assert g.shape == (rows_count, 1)
        output_dtype = nl.bfloat16 if round_bf16_output else z.dtype
        dz = nl.ndarray(z.shape, dtype=output_dtype, buffer=nl.shared_hbm)
        indices = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=indices, pattern=[[1, vocab]], offset=0, channel_multiplier=0)
        for tile in nl.affine_range(rows_count // (partitions * lanes)):
            rows = nl.ds((tile * lanes + nl.program_id(0)) * partitions, partitions)
            logits = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            labels = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            gradient = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            negative_lse = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            if z.dtype == nl.bfloat16:
                source = nl.ndarray((partitions, vocab), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=source, src=z[rows])
                nisa.tensor_copy(dst=logits, src=source)
            else:
                nisa.dma_copy(dst=logits, src=z[rows])
            nisa.dma_copy(dst=labels, src=tgt[rows])
            nisa.dma_copy(dst=gradient, src=g[rows])
            nisa.dma_copy(dst=negative_lse, src=lse[rows])
            nisa.tensor_scalar(dst=negative_lse, data=negative_lse, op0=nl.multiply, operand0=-1.0)
            capped_unit = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=capped_unit, op=nl.tanh, data=logits, scale=1.0 / softcap)
            nisa.activation(dst=logits, op=nl.exp, data=capped_unit, scale=softcap, bias=negative_lse)
            temporary = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=temporary, data=indices, op0=nl.equal, operand0=labels, op1=nl.multiply, operand1=gradient)
            nisa.scalar_tensor_tensor(dst=logits, data=logits, op0=nl.multiply, operand0=gradient, op1=nl.subtract, operand1=temporary)
            nisa.activation(dst=temporary, op=nl.square, data=capped_unit)
            nisa.scalar_tensor_tensor(dst=capped_unit, data=temporary, op0=nl.subtract, operand0=1.0, reverse0=True, op1=nl.multiply, operand1=logits)
            if output_dtype == nl.bfloat16:
                rounded = nl.ndarray((partitions, vocab), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rounded, src=capped_unit)
                nisa.dma_copy(dst=dz[rows], src=rounded)
            else:
                nisa.dma_copy(dst=dz[rows], src=capped_unit)
        return dz
    from torch_neuronx import wrap_nki as _wrap_softcap_ce_nki
    _softcap_ce_fwd_native = _wrap_softcap_ce_nki(_softcap_ce_fwd)
    _softcap_ce_bwd_native = _wrap_softcap_ce_nki(_softcap_ce_bwd)

    class _SoftcapCrossEntropyNKI(torch.autograd.Function):
        """Makes the fused softcap+cross-entropy kernel differentiable.

        Forward returns the UNREDUCED per-token loss, exactly like
        F.cross_entropy(..., reduction="none"); mean/none handling stays in the
        caller so this changes no reduction semantics.
        """

        @staticmethod
        def forward(ctx, z, tgt):
            lanes = _ce_lanes(z.shape[0])
            loss, lse = _softcap_ce_fwd_native[lanes](z, tgt, lanes, True)
            ctx.save_for_backward(z, tgt, lse)
            return loss.reshape(-1)

        @staticmethod
        def backward(ctx, grad_out):
            z, tgt, lse = ctx.saved_tensors
            g = grad_out.reshape(-1, 1).float().contiguous()
            lanes = _ce_lanes(z.shape[0])
            return (_softcap_ce_bwd_native[lanes](z, tgt, lse, g, lanes), None)
    from torch_neuronx import wrap_nki as _wrap_conv2_nki

    def _conv2_taps(wt, channels):
        prev = nl.ndarray((128, channels), dtype=wt.dtype, buffer=nl.sbuf)
        curr = nl.ndarray((128, channels), dtype=wt.dtype, buffer=nl.sbuf)
        pattern = [[0, 128], [1, channels]]
        nisa.dma_copy(dst=prev, src=wt.ap(pattern=pattern, offset=0))
        nisa.dma_copy(dst=curr, src=wt.ap(pattern=pattern, offset=channels))
        return (prev, curr)

    def _conv2_fwd_tile(x, out, prev_w, curr_w, batch, start, channels, first):
        prev = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
        curr = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=curr, src=x[batch, start:start + 128, :])
        if first:
            nisa.memset(dst=prev[0:1, :], value=0.0)
            nisa.dma_copy(dst=prev[1:128, :], src=x[batch, 0:127, :])
        else:
            nisa.dma_copy(dst=prev, src=x[batch, start - 1:start + 127, :])
        acc = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
        prod = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
        result = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=acc, data1=prev, data2=prev_w, op=nl.multiply)
        nisa.tensor_tensor(dst=prod, data1=curr, data2=curr_w, op=nl.multiply)
        nisa.tensor_tensor(dst=result, data1=acc, data2=prod, op=nl.add)
        nisa.dma_copy(dst=out[batch, start:start + 128, :], src=result)

    @nki.jit
    def _conv2_forward(x, wt, lanes: int=1):
        batch, length, channels = x.shape
        assert length % 128 == 0 and channels % 128 == 0
        assert wt.shape == (2, channels)
        assert lanes in (1, 2) and batch % lanes == 0
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        prev_w, curr_w = _conv2_taps(wt, channels)
        lane = nl.program_id(0)
        for b in nl.affine_range(batch // lanes):
            bi = b * lanes + lane
            _conv2_fwd_tile(x, out, prev_w, curr_w, bi, 0, channels, True)
            if length > 128:
                for t in nl.affine_range(length // 128 - 1):
                    _conv2_fwd_tile(x, out, prev_w, curr_w, bi, (t + 1) * 128, channels, False)
        return out

    def _conv2_dw_partial(xpart, grad, ones, partials, batch, tile, tap, channels):
        prod = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=xpart, data2=grad, op=nl.multiply)
        for c in nl.affine_range(channels // 128):
            summed = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
            result = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.nc_matmul(dst=summed, stationary=prod[:, c * 128:(c + 1) * 128], moving=ones, accumulate=False)
            nisa.tensor_copy(dst=result, src=summed)
            nisa.dma_copy(dst=partials[batch, tile, tap, c * 128:(c + 1) * 128, :], src=result)

    def _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones, batch, tile, length, channels, first, last):
        start = tile * 128
        xp = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
        xc = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
        gc = nl.ndarray((128, channels), dtype=grad.dtype, buffer=nl.sbuf)
        gn = nl.ndarray((128, channels), dtype=grad.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xc, src=x[batch, start:start + 128, :])
        nisa.dma_copy(dst=gc, src=grad[batch, start:start + 128, :])
        if first:
            nisa.memset(dst=xp[0:1, :], value=0.0)
            nisa.dma_copy(dst=xp[1:128, :], src=x[batch, 0:127, :])
        else:
            nisa.dma_copy(dst=xp, src=x[batch, start - 1:start + 127, :])
        if last:
            nisa.memset(dst=gn, value=0.0)
            nisa.dma_copy(dst=gn[0:127, :], src=grad[batch, start + 1:length, :])
        else:
            nisa.dma_copy(dst=gn, src=grad[batch, start + 1:start + 129, :])
        acc = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
        prod = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
        result = nl.ndarray((128, channels), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=acc, data1=gc, data2=curr_w, op=nl.multiply)
        nisa.tensor_tensor(dst=prod, data1=gn, data2=prev_w, op=nl.multiply)
        nisa.tensor_tensor(dst=result, data1=acc, data2=prod, op=nl.add)
        nisa.dma_copy(dst=dx[batch, start:start + 128, :], src=result)
        _conv2_dw_partial(xp, gc, ones, partials, batch, tile, 0, channels)
        _conv2_dw_partial(xc, gc, ones, partials, batch, tile, 1, channels)

    @nki.jit
    def _conv2_backward(x, grad, wt, lanes: int=1):
        batch, length, channels = x.shape
        assert length % 128 == 0 and channels % 128 == 0
        assert grad.shape == x.shape and wt.shape == (2, channels)
        assert lanes in (1, 2) and batch % lanes == 0
        tiles = length // 128
        dx = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        partials = nl.ndarray((batch, tiles, 2, channels, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        prev_w, curr_w = _conv2_taps(wt, channels)
        ones = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        lane = nl.program_id(0)
        for b in nl.affine_range(batch // lanes):
            bi = b * lanes + lane
            if tiles == 1:
                _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones, bi, 0, length, channels, True, True)
            else:
                _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones, bi, 0, length, channels, True, False)
                if tiles > 2:
                    for t in nl.affine_range(tiles - 2):
                        _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones, bi, t + 1, length, channels, False, False)
                _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones, bi, tiles - 1, length, channels, False, True)
        return (dx, partials)
    _conv2_forward_native = _wrap_conv2_nki(_conv2_forward)
    _conv2_backward_native = _wrap_conv2_nki(_conv2_backward)

    class _Conv2NKI(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, weight):
            assert weight.shape == (x.shape[-1], 1, 2)
            lanes = 2 if x.shape[0] % 2 == 0 else 1
            wt = weight[:, 0, :].T.contiguous().to(x.dtype)
            xx = x.contiguous()
            y = _conv2_forward_native[lanes](xx, wt, lanes)
            ctx.save_for_backward(xx, wt)
            ctx.lanes = lanes
            ctx.weight_dtype = weight.dtype
            return y

        @staticmethod
        def backward(ctx, grad):
            x, wt = ctx.saved_tensors
            dx, partials = _conv2_backward_native[ctx.lanes](x, grad.contiguous(), wt, ctx.lanes)
            dw = partials.sum(dim=(0, 1)).squeeze(-1).to(wt.dtype)
            return (dx, dw.T.unsqueeze(1).to(ctx.weight_dtype))
    _HAS_NKI = True
except Exception:
    _HAS_NKI = False

def softcap_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, loss_reduction: str) -> torch.Tensor:
    """softcap(z) = LOGIT_SOFTCAP * tanh(z / LOGIT_SOFTCAP) followed by cross-entropy. On Neuron, contiguous
    fp32 logits with a row count divisible by 128 use the fused NKI kernel pair, which returns the
    unreduced per-token loss; the mean/none reduction is applied here exactly as F.cross_entropy
    would apply it. Other inputs use the eager chain.
    """
    if _HAS_NKI and logits.device.type == 'neuron' and (loss_reduction in ('mean', 'none')) and (logits.dtype == torch.float32) and logits.is_contiguous():
        z = logits.reshape(-1, logits.shape[-1])
        if z.shape[0] % _NKI_PARTITION == 0:
            tgt = targets.reshape(-1, 1).to(torch.float32)
            per_token = _SoftcapCrossEntropyNKI.apply(z, tgt)
            if loss_reduction == 'none':
                out = per_token.view_as(targets)
            else:
                out = per_token.mean()
            return out
    capped = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
    loss = F.cross_entropy(capped.reshape(-1, capped.size(-1)), targets.reshape(-1), reduction=loss_reduction)
    if loss_reduction == 'none':
        out = loss.view_as(targets)
    else:
        out = loss
    return out

class Linear(nn.Linear):
    """Linear without bias that casts the weight to the input dtype. With USE_WEIGHT_CACHE the cast copy is
    refreshed once per optimizer step (refresh_weight_cache) instead of on every forward call.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_weight: torch.Tensor | None = None

    def refresh_weight_cache(self, dtype: torch.dtype) -> None:
        self._cached_weight = self.weight.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cached_weight is not None and self._cached_weight.dtype == x.dtype:
            w = self._cached_weight
        else:
            w = self.weight.to(dtype=x.dtype)
        return F.linear(x, w, None)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * -sin + x2 * cos
    return torch.cat((y1, y2), dim=-1)

def rope_norm(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """norm(apply_rotary_emb(x, cos, sin)), fused into one NKI kernel on Neuron when B*T and T are multiples
    of 128 (x is [B, T, H, D]; cos/sin are [1, T, 1, D//2]); otherwise the eager composition.
    """
    if _HAS_NKI and x.device.type == 'neuron' and (x.ndim == 4):
        b, t, _, d = x.shape
        if b * t % _NKI_PARTITION == 0 and t % _NKI_PARTITION == 0 and (d % 2 == 0) and (cos.shape[1] == t) and (cos.shape[-1] == d // 2) and x.is_contiguous():
            return _RopeNormNKI.apply(x, cos, sin)
    return norm(apply_rotary_emb(x, cos, sin))

class CausalSelfAttention(nn.Module):

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

def reference_forward(x, up, down):
    relu = F.relu(F.linear(x, up.to(x.dtype)))
    return (F.linear(relu.square(), down.to(x.dtype)), relu)

def _fused_mlp_kernel(x, up, down):
    result, relu = reference_forward(x, up, down)
    return (result, relu.T.contiguous())

def _linear_weight(linear, dtype):
    cached = linear._cached_weight
    if cached is not None and cached.dtype == dtype:
        return cached
    return linear.weight.to(dtype=dtype)

class _FusedMLP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, up, down, up_packed=None, down_packed=None):
        result, relu = _fused_mlp_kernel(x, up, down)
        ctx.save_for_backward(x, up, down, relu)
        return result

    @staticmethod
    def backward(ctx, grad):
        x, up, down, relu = ctx.saved_tensors
        activated_t = relu.square()
        dhidden_t = down.to(x.dtype).T @ grad.T
        dpre_t = dhidden_t * (relu * 2.0)
        dx = (up.to(x.dtype).T @ dpre_t).T
        dup = (dpre_t @ x).to(up.dtype)
        ddown = (grad.T @ activated_t.T).to(down.dtype)
        return (dx, dup, ddown)

class MLP(nn.Module):
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

class Block(nn.Module):

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

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        padded_vocab_size = (config.vocab_size + 63) // 64 * 64
        self.transformer = nn.ModuleDict({'wte': nn.Embedding(padded_vocab_size, config.n_embd), 'h': nn.ModuleList([Block(config, i) for i in range(config.n_layer)])})
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, config.n_embd) for i in range(config.n_layer) if i % 2 == (config.n_layer - 1) % 2})
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, config.n_embd // config.n_head)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = INIT_SCALE * math.sqrt(3.0) * self.config.n_embd ** (-0.5)
        for block in self.transformer.h:
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
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, self.config.n_embd // self.config.n_head)
        self.cos.copy_(cos)
        self.sin.copy_(sin)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int=ROPE_BASE):
        device = self.transformer.wte.weight.device
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / base ** (freqs / head_dim)
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        table = torch.outer(t, inv_freq)
        cos = table.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = table.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        return (cos, sin)

    def estimate_flops(self) -> int:
        """Estimate FLOPs per token (forward + backward). Attention QK^T and AV are O(T) per token."""
        nparams = sum((p.numel() for p in self.parameters()))
        non_matmul = self.transformer.wte.weight.numel() + 2 * self.config.n_layer
        h = self.config.n_head
        d = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = self.config.n_layer * 12 * h * d * t
        return 6 * (nparams - non_matmul) + attn_flops

    def setup_optimizer(self):
        scalar_params = []
        matrix_params = []
        local_conv_params = []
        for block in self.transformer.h:
            local_conv_params.append(block.local_conv.weight)
            scalar_params.extend([block.resid_lambda, block.x0_lambda])
            matrix_params.extend(block.attn.parameters())
            matrix_params.extend(block.mlp.parameters())
        for p in scalar_params:
            if p.ndim > 1:
                raise ValueError(f'scalar_params must hold only scalars/1-D vectors (SCALAR_LR={SCALAR_LR} is tuned for those); got a {tuple(p.shape)} tensor. Route it to gate_params or to Muon instead.')
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        value_embed_params = list(self.value_embeds.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** (-0.5)
        param_groups = [dict(kind='adamw', params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, EMBED_ADAMW_BETA2), eps=1e-10, weight_decay=0.001), dict(kind='adamw', params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01), dict(kind='adamw', params=scalar_params, lr=SCALAR_LR, betas=(SCALAR_BETA1, SCALAR_BETA2), eps=1e-10, weight_decay=0.0)]
        if value_embed_params:
            param_groups.append(dict(kind='adamw', params=value_embed_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, EMBED_ADAMW_BETA2), eps=1e-10, weight_decay=0.0))
        if local_conv_params:
            param_groups.append(dict(kind='adamw', params=local_conv_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        for shape in sorted({p.shape for p in matrix_params}):
            params = [p for p in matrix_params if p.shape == shape]
            group = dict(kind='muon', params=params, lr=MATRIX_LR, momentum=0.95, beta2=MUON_BETA2, ns_steps=MUON_NS_STEPS, weight_decay=WEIGHT_DECAY, transposed=False)
            param_groups.append(group)
        opt = MuonAdamW(param_groups)
        for group in opt.param_groups:
            group['initial_lr'] = group['lr']
            if group.get('betas') is not None:
                group['initial_betas'] = tuple(group['betas'])
        return opt

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None=None, loss_reduction: str='mean'):
        _b, t = idx.shape
        assert t <= self.config.sequence_len
        cos = self.cos[:, :t]
        sin = self.sin[:, :t]
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](idx).to(COMPUTE_DTYPE) if str(i) in self.value_embeds else None
            x = block(x, x0, ve, cos, sin)
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        if targets is None:
            return LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        return softcap_cross_entropy(logits, targets, loss_reduction)

def build_config() -> GPTConfig:
    """Model configuration from the module constants: DEPTH layers, N_EMBD wide, HEAD_DIM-wide heads."""
    if N_EMBD % HEAD_DIM != 0:
        raise ValueError(f'N_EMBD={N_EMBD} must be a multiple of HEAD_DIM={HEAD_DIM}')
    return GPTConfig(sequence_len=SEQ_LEN, vocab_size=TOKENIZER_VOCAB_SIZE, n_layer=DEPTH, n_head=N_EMBD // HEAD_DIM, n_embd=N_EMBD, mlp_ratio=MLP_RATIO)
POLAR_EXPRESS_COEFFS = [(8.156554524902461, -22.48329292557795, 15.878769915207462), (4.042929935166739, -2.808917465908714, 0.5000178451051316), (3.8916678022926607, -2.772484153217685, 0.5060648178503393), (3.285753657755655, -2.3681294933425376, 0.46449024233003106), (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)]
_MUON_COMPILED: dict[tuple, object] = {}

def _muon_math(w, g, momentum_buffer, second, mom, one_mom, lr, lrwd, tall: bool, ns_steps: int, red_dim: int, red_dim_size: int, one_beta2, to_bf16: bool, lr_fp32: bool=False):
    """Muon update for one parameter group as a pure function (stacked weights, gradients, momentum and
    second-moment buffers in; updated buffers out), written so torch.compile(backend='neuron') can
    trace it: no torch._foreach_* ops, no in-place ops on inputs, and every per-step scalar (lr,
    lr*wd, momentum, 1-momentum, optionally 1-beta2) arrives as a 0-dim fp32 device tensor so the
    schedule never changes the traced graph. Newton-Schulz orthogonalisation with the Polar Express
    coefficients, per-row/column second-moment normalisation, and weight decay masked to the
    entries where gradient and weight agree in sign.
    """
    momentum_buffer = torch.lerp(momentum_buffer, g, one_mom)
    g = torch.lerp(g, momentum_buffer, mom)
    x = g.bfloat16() if to_bf16 else g
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-06)
    if tall:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            xtx = x.mT @ x
            x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            xxt = x @ x.mT
            x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    g = x
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second = torch.lerp(second, v_mean, one_beta2)
    step_size = second.clamp_min(1e-10).rsqrt()
    scaled = v_mean * red_dim_size * step_size.square()
    v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
    mask = g * w >= 0
    w = w - (_muon_lr_g(lr, g, lr_fp32) + lrwd * w * mask)
    return (w, momentum_buffer, second)

def _muon_lr_g(lr, g, lr_fp32: bool=False):
    """`lr * g` with the dtype behaviour of a 0-dim lr: a dimensioned lr is cast to g's dtype before the
    multiply; with lr_fp32 the product is formed in fp32 instead.
    """
    if isinstance(lr, torch.Tensor) and lr.dim() > 0 and (lr.dtype != g.dtype):
        lr = lr.to(g.dtype)
        return lr * g
    if lr_fp32 and isinstance(lr, torch.Tensor) and (lr.dim() == 0) and (lr.dtype != g.dtype):
        return lr * g.float()
    return lr * g

class MuonAdamW(torch.optim.Optimizer):

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._scalar_cache: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}

    def _device_scalar(self, group, key: str, value: float, ref: torch.Tensor) -> torch.Tensor:
        """Carry a per-step hyperparameter in a persistent device tensor rather than a Python float, so the
        traced Neuron graph is identical on every step.
        """
        cache = self._scalar_cache.setdefault(id(group), {})
        entry = cache.get(key)
        if entry is None:
            entry = (torch.empty((), dtype=torch.float32), torch.zeros((), dtype=torch.float32, device=ref.device))
            cache[key] = entry
        host, dev = entry
        host.fill_(float(value))
        dev.copy_(host)
        return dev

    def _muon_depth_dev(self, group, ref: torch.Tensor):
        """Per-parameter (lr, momentum) multipliers of a group as [K, 1, 1] tensors, built once per group.
        Returns None when the group has none (the case in this configuration).
        """
        mults = group.get('depth_mults')
        if mults is None:
            return None
        cache = self._scalar_cache.setdefault(id(group), {})
        entry = cache.get('depth_mults')
        if entry is None:
            lr_mult, mom_mult = mults
            host_lr = torch.tensor(lr_mult, dtype=torch.float32).view(-1, 1, 1)
            host_mom = torch.tensor(mom_mult, dtype=torch.float64).view(-1, 1, 1)
            entry = (host_lr, host_mom, host_lr.to(ref.device))
            cache['depth_mults'] = entry
        return entry

    @staticmethod
    def _fill_depth_momentum(host_mom64, mom: float, out_mom, out_one_mom) -> None:
        """Write the per-parameter (momentum, 1-momentum) pair into two fp32 [K, 1, 1] rows; both products are
        formed in float64 and rounded once, clamped to a valid lerp weight.
        """
        one = (host_mom64 * (1.0 - mom)).clamp_(0.0, 1.0)
        out_one_mom.copy_(one)
        out_mom.copy_(1.0 - one)

    def _muon_depth_mom(self, group, ref: torch.Tensor):
        """(momentum, 1-momentum) per parameter as two [K, 1, 1] fp32 device views behind one host->device copy;
        None when the group has no multipliers.
        """
        depth = self._muon_depth_dev(group, ref)
        if depth is None:
            return None
        host_mom = depth[1]
        cache = self._scalar_cache.setdefault(id(group), {})
        entry = cache.get('depth_mom_step')
        if entry is None:
            k = host_mom.shape[0]
            entry = (torch.empty(2, k, 1, 1, dtype=torch.float32), torch.zeros(2, k, 1, 1, dtype=torch.float32, device=ref.device))
            cache['depth_mom_step'] = entry
        host, dev = entry
        self._fill_depth_momentum(host_mom, float(group['momentum']), host[0], host[1])
        dev.copy_(host)
        return (dev[0], dev[1])

    def _muon_compile_scalars(self, group, ref: torch.Tensor, ratio: float):
        """(lr*ratio, lr*ratio*wd, momentum, 1-momentum, 1-beta2) as five 0-dim fp32 device views behind one
        host->device copy. The lr products are formed in fp32 and 1-momentum in float64 so they match the
        eager path bit for bit.
        """
        depth = self._muon_depth_dev(group, ref)
        cache = self._scalar_cache.setdefault(id(group), {})
        if depth is not None:
            host_lr, host_mom, _dev_lr = depth
            entry = cache.get('muon_compile_depth')
            if entry is None:
                k = host_lr.shape[0]
                entry = (torch.empty(5, k, 1, 1, dtype=torch.float32), torch.zeros(5, k, 1, 1, dtype=torch.float32, device=ref.device), torch.empty((), dtype=torch.float32))
                cache['muon_compile_depth'] = entry
            host, dev, tmp = entry
            tmp.fill_(float(group['lr']))
            tmp.mul_(ratio)
            torch.mul(host_lr, float(tmp), out=host[0])
            torch.mul(host[0], float(group['weight_decay']), out=host[1])
            self._fill_depth_momentum(host_mom, float(group['momentum']), host[2], host[3])
            host[4].fill_(1.0 - float(group['beta2']))
            dev.copy_(host)
            return (dev[0], dev[1], dev[2], dev[3], dev[4])
        entry = cache.get('muon_compile')
        if entry is None:
            entry = (torch.empty(5, dtype=torch.float32), torch.zeros(5, dtype=torch.float32, device=ref.device), torch.empty((), dtype=torch.float32))
            cache['muon_compile'] = entry
        host, dev, tmp = entry
        tmp.fill_(float(group['lr']))
        tmp.mul_(ratio)
        host[0].fill_(float(tmp))
        tmp.mul_(float(group['weight_decay']))
        host[1].fill_(float(tmp))
        mom = float(group['momentum'])
        host[2].fill_(mom)
        host[3].fill_(1.0 - mom)
        host[4].fill_(1.0 - float(group['beta2']))
        dev.copy_(host)
        return (dev[0], dev[1], dev[2], dev[3], dev[4])

    def _adamw_state(self, group):
        """Shared prologue: parameters that have a gradient this step, lazily created state, per-parameter step
        counters, and the per-parameter bias corrections as scalar lists.
        """
        beta1, beta2 = group['betas']
        params = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        bias1 = []
        bias2 = []
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            params.append(p)
            grads.append(p.grad)
            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])
            bias1.append(1 - beta1 ** state['step'])
            bias2.append(1 - beta2 ** state['step'])
        return (params, grads, exp_avgs, exp_avg_sqs, bias1, bias2)

    @torch.no_grad()
    def _adamw_step_foreach(self, group):
        """AdamW for one group with batched torch._foreach_* calls (one launch per stage instead of one per
        parameter). Bit-identical to the per-parameter form: grad.square() as _foreach_mul(g, g),
        per-parameter bias corrections as scalar lists, and the weight-decay multiply skipped when
        weight_decay == 0 (an exact no-op).
        """
        params, grads, exp_avgs, exp_avg_sqs, bias1, bias2 = self._adamw_state(group)
        if not params:
            return
        lr = group['lr']
        lr = self._device_scalar(group, 'lr', lr, params[0])
        beta1, beta2 = group['betas']
        wd = group['weight_decay']
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), 1 - beta2)
        denom = torch._foreach_div(exp_avg_sqs, bias2)
        torch._foreach_sqrt_(denom)
        torch._foreach_add_(denom, group['eps'])
        update = torch._foreach_div(exp_avgs, bias1)
        torch._foreach_div_(update, denom)
        if wd != 0:
            torch._foreach_mul_(params, 1 - lr * wd)
        torch._foreach_mul_(update, lr)
        torch._foreach_sub_(params, update)

    @torch.no_grad()
    def _muon_step(self, group):
        params = group['params']
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        shape = p0.shape
        transposed = group.get('transposed', False)
        lshape = (shape[-1], shape[-2]) if transposed else shape
        red_dim = -1 if lshape[-2] >= lshape[-1] else -2
        if transposed:
            red_dim = -2 if red_dim == -1 else -1
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if 'second_momentum_buffer' not in state:
            state_shape = (len(params), shape[-2], 1) if red_dim == -1 else (len(params), 1, shape[-1])
            state['second_momentum_buffer'] = torch.zeros(state_shape, dtype=torch.float32, device=p0.device)
        if 'stacked_grads' not in state:
            state['stacked_grads'] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if 'stacked_params' not in state:
            state['stacked_params'] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)
        g = state['stacked_grads']
        w = state['stacked_params']
        g_rows = list(g.unbind(0))
        grads = [p.grad for p in params]
        if all((gr is not None for gr in grads)):
            torch._foreach_copy_(g_rows, grads)
        else:
            for i, gr in enumerate(grads):
                if gr is None:
                    g_rows[i].zero_()
                else:
                    g_rows[i].copy_(gr)
        torch._foreach_copy_(list(w.unbind(0)), params)
        momentum_buffer = state['momentum_buffer']
        second = state['second_momentum_buffer']
        if w.dtype == torch.float32:
            ratio = max(1.0, lshape[-2] / lshape[-1]) ** 0.5
            ns_steps = min(group['ns_steps'], len(POLAR_EXPRESS_COEFFS))
            key = (g.shape[-2] > g.shape[-1], ns_steps, red_dim, tuple(g.shape))
            fn = _MUON_COMPILED.get(key)
            if fn is None:
                fn = torch.compile(_muon_math, backend='neuron', dynamic=False) if p0.device.type == 'neuron' else _muon_math
                _MUON_COMPILED[key] = fn
                print0(f'muon step: COMPILED body for {tuple(g.shape)} (tall={key[0]} ns_steps={ns_steps} red_dim={red_dim}) [specialisations so far: {len(_MUON_COMPILED)}]')
            lr_t, lrwd_t, mom_t, one_mom_t, one_b2_t = self._muon_compile_scalars(group, p0, ratio)
            one_beta2 = 1 - group['beta2']
            w, momentum_buffer, second = fn(w, g, momentum_buffer, second, mom_t, one_mom_t, lr_t, lrwd_t, key[0], ns_steps, red_dim, g.size(red_dim), one_beta2, COMPUTE_DTYPE == torch.bfloat16, False)
            state['stacked_params'] = w
            state['momentum_buffer'] = momentum_buffer
            state['second_momentum_buffer'] = second
            torch._foreach_copy_(params, list(w.unbind(0)))
            return
        depth = self._muon_depth_dev(group, p0)
        if depth is None:
            momentum_buffer.lerp_(g, 1 - group['momentum'])
            g = g.lerp_(momentum_buffer, group['momentum'])
        else:
            mom_d, one_mom_d = self._muon_depth_mom(group, p0)
            momentum_buffer.lerp_(g, one_mom_d)
            g = g.lerp_(momentum_buffer, mom_d)
        x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-06)
        if x.shape[-2] > x.shape[-1]:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group['ns_steps']]:
                xtx = x.mT @ x
                x = a * x + x @ (b * xtx + c * (xtx @ xtx))
        else:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group['ns_steps']]:
                xxt = x @ x.mT
                x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
        g = x
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = g.size(red_dim)
        v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
        second.lerp_(v_mean, 1 - group['beta2'])
        step_size = second.clamp_min(1e-10).rsqrt()
        scaled = v_mean * red_dim_size * step_size.square()
        v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
        g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
        ratio = max(1.0, lshape[-2] / lshape[-1]) ** 0.5
        lr_base = group['lr']
        lr_base = self._device_scalar(group, 'lr', lr_base, p0)
        lr = lr_base * ratio
        if depth is not None:
            lr = lr * depth[2]
        wd = group['weight_decay']
        mask = g * w >= 0
        w.sub_(_muon_lr_g(lr, g, False) + lr * wd * w * mask)
        torch._foreach_copy_(params, list(w.unbind(0)))

    @torch.no_grad()
    def step(self, update_adamw: bool=True):
        """update_adamw=False runs only the Muon groups and leaves the AdamW groups' gradients accumulating."""
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                if not update_adamw:
                    continue
                self._adamw_step_foreach(group)
            elif group['kind'] == 'muon':
                self._muon_step(group)
            else:
                raise ValueError(f"unknown optimizer kind: {group['kind']}")

def sync_gradients(model: nn.Module) -> None:
    """All-reduce (average) every gradient. skip_param_ids defers the listed parameters' reduction to a
    later step; ReduceOp.AVG is linear, so deferring is exact.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

def clip_and_sync_gradients(model: nn.Module, clip_main_params: list[torch.nn.Parameter], *, max_norm: float) -> None:
    """Average the accumulated gradients across ranks, then clip them to a global norm of max_norm."""
    sync_gradients(model)
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(clip_main_params, max_norm, foreach=None)

def save_checkpoint(path: Path, model: GPT, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'model_config': asdict(model.config), 'meta': meta}
    torch.save(payload, path)
EVAL_MIN_SEQ_LEN = 8192

def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Return the trained model (same GPT class, strict-loaded weights) in eval mode for the scorer.

    Only the non-persistent rotary buffers are sized for up to EVAL_MIN_SEQ_LEN positions so that any
    scorer context length is accepted; for positions below the training length their values are the
    ones used in training. No weights, layers or arithmetic differ from training.
    """
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = GPTConfig(**payload['model_config'])
    if config.sequence_len < EVAL_MIN_SEQ_LEN:
        config.sequence_len = EVAL_MIN_SEQ_LEN
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(payload['model_state'], strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model

def lr_multiplier(step: int, elapsed: float=0.0, max_train_seconds: float=1.0) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    progress = min(1.0, elapsed / max(1e-06, max_train_seconds))
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    lrm = cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
    return lrm

def muon_momentum(step: int, progress: float=0.0) -> float:
    """Muon momentum: 0.85 -> 0.95 over MUON_MOMENTUM_WARMUP_STEPS, then a cooldown towards
    MUON_MOMENTUM_FINAL over the last MUON_MOMENTUM_COOLDOWN_FRAC of the wall-clock budget.
    """
    frac = min(step / MUON_MOMENTUM_WARMUP_STEPS, 1.0)
    m = (1 - frac) * 0.85 + frac * 0.95
    cd = min(1.0, max(0.0, (progress - (1.0 - MUON_MOMENTUM_COOLDOWN_FRAC)) / MUON_MOMENTUM_COOLDOWN_FRAC))
    m = (1 - cd) * m + cd * MUON_MOMENTUM_FINAL
    return m

def main() -> None:
    parser = argparse.ArgumentParser(description='Trainium Frontier Phase 1 training: dense 6x1024 GPT, 1,800 s, full chip (LNC2 x 4 ranks).')
    parser.add_argument('--device-type', type=str, default='', help='cpu | cuda | neuron; empty selects automatically.')
    parser.add_argument('--num-steps', type=int, default=NUM_STEPS, help='Upper bound on steps; the wall-clock budget ends training first.')
    parser.add_argument('--max-train-seconds', type=int, default=MAX_TRAIN_SECONDS)
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True, help="torch.compile(backend='neuron') for the model forward/backward.")
    parser.add_argument('--seed', type=int, default=42, help='torch.manual_seed value.')
    args = parser.parse_args()
    device_type = detect_device_type(args.device_type)
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type, seed=args.seed)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    print0(f'device={device} dtype={COMPUTE_DTYPE} world_size={world_size}')
    print0(f'training target={args.num_steps} steps max_seconds={args.max_train_seconds} official_budget={TRAIN_TIME_BUDGET_SECONDS}s')
    print0(f'gradient_clip={GRAD_CLIP:g} (after the cross-rank average)')
    tokenizer = ensure_tokenizer(build_if_missing=False)
    global BOS_TOKEN_ID
    BOS_TOKEN_ID = tokenizer.get_bos_token_id()
    config = build_config()
    print0(f'model_config={json.dumps(asdict(config), indent=2)}')
    model = GPT(config)
    model.init_weights()
    model.to(device)
    num_params = sum((p.numel() for p in model.parameters()))
    flops_per_token = model.estimate_flops()
    print0(f'num_params={num_params:,} flops_per_token={flops_per_token:e}')
    orig_model = model
    if args.compile:
        compile_backend = 'neuron' if device_type == 'neuron' else 'inductor'
        model = torch.compile(model, backend=compile_backend, dynamic=False)
        print0(f'torch.compile enabled (backend={compile_backend})')
    world_tokens_per_micro = DEVICE_BATCH_SIZE * PACK_FACTOR * SEQ_LEN * world_size
    if TOTAL_BATCH_SIZE % world_tokens_per_micro != 0:
        raise ValueError(f'TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} must divide micro tokens={world_tokens_per_micro}')
    grad_accum_steps = TOTAL_BATCH_SIZE // world_tokens_per_micro
    print0(f'micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps} pack_factor={PACK_FACTOR}')
    loader_seq_len = PACK_FACTOR * SEQ_LEN
    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, 'train', device)

    def fetch():
        xb, yb, state = next(train_loader)
        if PACK_FACTOR > 1:
            xb = xb.reshape(DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN)
            yb = yb.reshape(DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN)
        return (xb, yb, state)
    x, y, loader_state = fetch()
    optimizer = model.setup_optimizer()
    assert dist.is_initialized() and dist.get_world_size() == 4
    owned_optimizer = OwnedOptimizer(optimizer)
    cached_linears = [m for m in model.modules() if isinstance(m, Linear)]
    clip_main_params = list(orig_model.parameters())
    loop_started = time.monotonic()
    budget_started = loop_started
    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0
    recent_step_times: list[float] = []
    startup_allowance_est = 0.0
    grad_accum_now = grad_accum_steps
    while step < args.num_steps:
        elapsed = time.monotonic() - budget_started
        grad_accum_now = grad_accum_steps
        synchronize(device)
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        for micro_index in range(grad_accum_now):
            loss = model(x, y)
            last_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, loader_state = fetch()
        clip_and_sync_gradients(model, clip_main_params, max_norm=GRAD_CLIP)
        lrm = lr_multiplier(step, elapsed=elapsed, max_train_seconds=args.max_train_seconds)
        lrm_flat = min(1.0, (step + 1) / max(1, WARMUP_STEPS))
        wd_progress = min(1.0, elapsed / max(1e-06, args.max_train_seconds))
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * (lrm_flat if group.get('no_lr_decay') else lrm)
            if group['kind'] == 'muon':
                group['momentum'] = muon_momentum(step, wd_progress)
                group['weight_decay'] = WEIGHT_DECAY * (1.0 - wd_progress)
                group['beta2'] = 0.9
        owned_optimizer.step(update_adamw=True)
        for m in cached_linears:
            m.refresh_weight_cache(COMPUTE_DTYPE)
        synchronize(device)
        record_step_boundary(step)
        dt = time.monotonic() - t0
        recent_step_times.append(dt)
        train_loss = float(last_loss.item())
        if step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED:
            now = time.monotonic()
            observed_startup = now - _PROCESS_STARTED
            startup_allowance_est = min(observed_startup, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            excess = max(0.0, observed_startup + STARTUP_LAUNCH_MARGIN_SECONDS - TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            budget_started = now - excess
        elapsed_after_step = time.monotonic() - budget_started
        reached_min_steps = step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED
        step_cap_hit = step + 1 >= args.num_steps
        time_budget_hit = reached_min_steps and (elapsed_after_step >= args.max_train_seconds or elapsed_after_step + max(recent_step_times[-20:]) + CHECKPOINT_RESERVE_SECONDS >= args.max_train_seconds)
        stop_after_step = step_cap_hit or time_budget_hit
        control = torch.tensor([1 if math.isnan(train_loss) or train_loss > 100 else 0, 1 if stop_after_step else 0], device=device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(control, op=dist.ReduceOp.MAX)
        if int(control[0].item()) != 0:
            raise RuntimeError(f'bad loss detected: {train_loss}')
        step_tokens = grad_accum_now * world_tokens_per_micro
        total_train_tokens += step_tokens
        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        tok_per_sec = step_tokens / max(dt, 1e-06)
        if rank == 0 and (step < 5 or step % 10 == 0):
            remaining_steps = args.num_steps - step - 1
            remaining_time = max(0.0, args.max_train_seconds - (time.monotonic() - budget_started))
            print(f"step {step:05d} | loss {debiased:.4f} | lrm {lrm:.3f} | dt {dt:.2f}s | tok/s {tok_per_sec:,.0f} | remaining_steps {remaining_steps} | remaining_time {remaining_time:.0f}s | epoch {loader_state['epoch']}", flush=True)
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1
        if int(control[1].item()) != 0:
            break
    checkpoint_path = Path(args.out_dir) / 'final.pt'
    budget_elapsed_before_save = time.monotonic() - budget_started
    if rank == 0:
        save_checkpoint(checkpoint_path, orig_model, {'step': step, 'total_train_tokens': total_train_tokens, 'budget_elapsed_before_save': budget_elapsed_before_save, 'world_size': world_size, 'compiled': args.compile})
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    budget_elapsed_after_save = time.monotonic() - budget_started
    if rank == 0:
        print('---')
        print(f'checkpoint_path:    {checkpoint_path}')
        print(f'training_seconds:  {budget_elapsed_before_save:.1f}')
        print(f'budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}')
        print(f'target_steps:      {args.num_steps}')
        print(f'max_train_seconds: {args.max_train_seconds}')
        print(f'startup_allowance_est: {startup_allowance_est:.1f}')
        print(f'charged_seconds_est:   {budget_elapsed_after_save:.1f}')
        print(f'over_budget:       {budget_elapsed_after_save > TRAIN_TIME_BUDGET_SECONDS}')
        print(f'total_seconds:     {time.monotonic() - loop_started:.1f}')
        print(f'compiled:          {args.compile}')
        print(f'num_steps:         {step}')
        print(f'total_tokens_M:    {total_train_tokens / 1000000.0:.1f}')
        print(f'world_size:        {world_size}')
        print(f"lnc:               {os.environ.get('NEURON_LOGICAL_NC_CONFIG', '')}")
        print(f'seq_len:           {SEQ_LEN}')
        print(f'num_params_M:      {num_params / 1000000.0:.1f}')
    cleanup_runtime()

def make_plan(groups, world_size):
    """Assign each parameter's optimizer update to one rank, balancing measured per-shape Muon step
    times (milliseconds, used only as load-balancing weights) and AdamW element counts."""
    muon_ms = {(1024, 1024): 12.54413 / 24, (1024, 4096): 9.76316 / 6, (4096, 1024): 8.98251 / 6, (8, 32): 0.23355 / 3}
    entries = []
    parameters = []
    seen = set()
    for group_index, group in enumerate(groups):
        if group['kind'] not in ('adamw', 'muon'):
            raise ValueError('optimizer ownership supports AdamW and Muon groups only')
        if 'depth_mults' in group:
            raise ValueError('per-depth optimizer multipliers require a separate ownership gate')
        for position, p in enumerate(group['params']):
            if id(p) in seen or p.dtype != torch.float32 or (not p.is_contiguous()):
                raise ValueError('ownership requires unique contiguous FP32 Parameters')
            seen.add(id(p))
            cost = muon_ms.get(tuple(p.shape), 0.01) if group['kind'] == 'muon' else 4.00487945 * p.numel() / (8192 * 1024) + 0.01
            entries.append(dict(index=len(entries), group=group_index, position=position, kind=group['kind'], shape=list(p.shape), elements=p.numel(), estimated_ms=cost))
            parameters.append(p)
    if not parameters:
        raise ValueError('empty optimizer')
    counts = [0] * world_size
    costs = [0.0] * world_size
    for entry in sorted(entries, key=lambda e: (-e['estimated_ms'], -e['elements'], e['index'])):
        owner = min(range(world_size), key=lambda r: (costs[r], counts[r], r))
        entry.update(owner=owner, offset=counts[owner])
        counts[owner] += entry['elements']
        costs[owner] += entry['estimated_ms']
    stride = (max(counts) + 127) // 128 * 128
    return (entries, parameters, dict(elements_by_owner=counts, estimated_ms_by_owner=costs, shard_elements=stride, total_padded_bytes=stride * world_size * 4))

class OwnedOptimizer:

    def __init__(self, optimizer):
        if not dist.is_initialized():
            raise ValueError('optimizer ownership requires an initialized process group')
        if optimizer.state:
            raise ValueError('install ownership before the first update')
        self.optimizer = optimizer
        self.pad_singleton_muon = True
        self.rank, self.world_size = (dist.get_rank(), dist.get_world_size())
        self.entries, self.parameters, self.summary = make_plan(optimizer.param_groups, self.world_size)
        self.parameter_ids = [id(p) for p in self.parameters]
        self.groups, self.positions, self.padding = ([], [], {})
        for group_index, source in enumerate(optimizer.param_groups):
            positions = [e['position'] for e in self.entries if e['group'] == group_index and e['owner'] == self.rank]
            self.positions.append(positions)
            self.groups.append({**source, 'params': [source['params'][i] for i in positions]})
            if self.pad_singleton_muon and source['kind'] == 'muon' and (len(positions) == 1):
                dummy = torch.nn.Parameter(torch.zeros_like(self.groups[-1]['params'][0]))
                dummy.grad = torch.zeros_like(dummy)
                self.groups[-1]['params'].append(dummy)
                self.padding[group_index] = dummy
        device = self.parameters[0].device
        if any((p.device != device for p in self.parameters)):
            raise ValueError('all Parameters must share a device')
        stride = self.summary['shard_elements']
        self.send = torch.zeros(stride, dtype=torch.float32, device=device)
        self.received = torch.empty(stride * self.world_size, dtype=torch.float32, device=device)
        self.owned, self.send_views, self.receive_views = ([], [], [])
        for entry, p in zip(self.entries, self.parameters):
            self.receive_views.append(self.received.narrow(0, entry['owner'] * stride + entry['offset'], entry['elements']).view_as(p))
            if entry['owner'] == self.rank:
                self.owned.append(p)
                self.send_views.append(self.send.narrow(0, entry['offset'], entry['elements']).view_as(p))
        self.calls = 0

    @torch.no_grad()
    def step(self, update_adamw=True):
        opt = self.optimizer
        for source, owned in zip(opt.param_groups, self.groups):
            owned.update({key: value for key, value in source.items() if key != 'params'})
            if not owned['params']:
                continue
            if owned['kind'] == 'adamw':
                if update_adamw:
                    opt._adamw_step_foreach(owned)
            else:
                opt._muon_step(owned)
        if self.owned:
            torch._foreach_copy_(self.send_views, self.owned)
        dist.all_gather_into_tensor(self.received, self.send)
        torch._foreach_copy_(self.parameters, self.receive_views)
        self.calls += 1

if __name__ == '__main__':
    main()
