"""Trainium Frontier Phase 1: the fixed CLEAN-GATEFIX training recipe.

Eight 1024-wide blocks use a U-Net residual stack, with causal attention and
two-tap causal depthwise convolution in blocks 0, 2, 4, 6 and 7. All blocks use
ReLU-squared MLPs; block 7 has a gated token-value embedding. Untied token and
output embeddings, rotary Q/K normalization, learned residual/input mixing and
softcap-15 logits are shared by training and the saved-weight evaluation hook.

Native NKI kernels implement rotary RMS normalization, causal convolution and
softcap cross-entropy, including their backward passes. The MLP's custom
autograd uses PyTorch matrix operations. Base parameters and optimizer state are
FP32; the table and its RMS state are BF16; normalized branch inputs and the residual stream are BF16 on Neuron,
with explicit FP32 normalization/loss/optimizer statistics as implemented below.
Muon updates matrix weights; AdamW updates embeddings, scalar gates, the 2-D
value gate and convolutions. Four LNC2 ranks average then clip gradients and
share optimizer ownership. Training uses the provided loader, context 1024,
32 rows per rank, a 131072-to-262144 token batch ramp, the existing wall-clock
schedules and the parameter-free next-two-token auxiliary objective.

The main entrypoint trains from seed 42 and saves out/final.pt (CLI options:
--out-dir and --tbl_freeze_tokens). It never reads validation data. The scorer
calls load_for_eval to load the same causal model with strict saved weights.
Compiler defaults below preserve the verified gate-fix compilation policy;
they do not select a different model or inspect scoring inputs.
SPMD grids follow the documented LNC contract: a [2] grid requests two physical cores, so the same kernels run under the trainer's LNC=2 and any evaluator core configuration.
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
_NKI_LANES = 2 if os.environ.get('NEURON_LOGICAL_NC_CONFIG') == '2' else 1
if __name__ == '__main__':
    os.environ.setdefault('NEURON_CC_FLAGS', '--optlevel=3 --auto-cast matmult --auto-cast-type bf16')
os.environ.setdefault('NEURON_COMPILE_CACHE_DIR', '/tmp/neuron_cache')
os.environ.setdefault('NEURON_CC_FLAGS', '--optlevel=3')
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F
from prepare import TOKENIZER_VOCAB_SIZE, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS, TRAIN_STARTUP_STEPS_EXCLUDED, TRAIN_TIME_BUDGET_SECONDS, ensure_tokenizer, get_dist_info, make_dataloader, print0, record_step_boundary
SEQ_LEN = 1024
DEPTH = 8
HEAD_DIM = 128
N_EMBD = 1024
MLP_RATIO = 4
ATTN_SCALE = 0.0
LANE_B = True
N_KV_HEADS = 0
DEVICE_BATCH_SIZE = 1
TOTAL_BATCH_SIZE = 262144
NUM_STEPS = 1000000
MAX_TRAIN_SECONDS = 1800
CHECKPOINT_RESERVE_SECONDS = 15.0
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0
LOGIT_SOFTCAP = 15.0
EMBEDDING_LR = 0.3
UNEMBEDDING_LR = 0.006
MATRIX_LR = 0.021
SCALAR_LR = 0.2
SCALAR_BETA1 = 0.9
SCALAR_BETA2 = 0.95
WEIGHT_DECAY = 0.2
EMBED_ADAMW_BETA2 = 0.95
WARMUP_STEPS = 0
GRAD_CLIP = 1.0
# FLATCACHE: flat fp32 master + flat bf16 weight cache + flat eager tail (False = the per-parameter path).
FLAT_WEIGHTS = True
LOSS_SCALE = 1.0
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
ATTENTION_LAYERS = (0, 2, 4, 6, 7)
# Z-bundle switches (OP_INVENTORY.md, "Zero-engineering fixes"). Every switch is
# math-identical in forward; Z1 and Z3 change only the BF16 accumulation order of
# one gradient sum. Flip one to False to bisect; all default ON.
Z1_NORM_ONCE = True
Z3_ONE_GATE_SLICE = True
Z5_SKIP_VOCAB_SLICE = True
Z6_TABLE_REBIND = True
Z7_PERSISTENT_SCALARS = True
Z10_ROPE_WRAP_NKI = True
Z11_FULLGRAPH_CORE = True

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
                raise RuntimeError('causal convolution requires the installed native NKI stack')
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
    attn_scale: float = 0.0
    n_kv_heads: int = 0
    mtp: bool = True
    d1_detached: bool = False
    d1_injection: str = 'embedding_pre_norm'
    tbl_every: int = 2
    d2_two_site: bool = False
    tbl_freeze_tokens: int = 0
    lane_b: bool = False
COMPUTE_DTYPE: torch.dtype = torch.bfloat16

def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=_RMS_EPS).to(torch.bfloat16)
_NKI_PARTITION = 128
_RMS_EPS = 1.1920928955078125e-07
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def _rope_norm_fwd(x, cos, sin, lanes: int=1):
        """Fused rms_norm(rope(x)) for x [BT, H, D], cos/sin [T, D//2].

        lanes: physical cores in the matching launch grid (Z10). Row tiles are
        interleaved over nl.program_id(0) exactly as _softcap_ce_fwd does; every
        tile is independent, so the per-lane assignment changes no arithmetic.
        With lanes == 1 the loop is the original single-core loop.
        """
        BT, H, D = x.shape
        T = cos.shape[0]
        Dh = D // 2
        dt = x.dtype
        P = _NKI_PARTITION
        assert lanes in (1, 2) and BT % (P * lanes) == 0 and T % (P * lanes) == 0
        BC = [[H, P], [1, H], [0, D]]
        CS = [[Dh, P], [0, H], [1, Dh]]
        out = nl.ndarray((BT, H, D), dtype=dt, buffer=nl.shared_hbm)
        lane = nl.program_id(0) if lanes > 1 else 0
        tiles_per_seq = T // (P * lanes)
        for i in nl.affine_range(BT // (P * lanes)):
            r0 = (i * lanes + lane) * P
            t0 = ((i % tiles_per_seq) * lanes + lane) * P
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
    def _rope_norm_bwd(x, g, cos, sin, lanes: int=1):
        """Gradient of _rope_norm_fwd w.r.t. x (cos/sin are constants).

        lanes: same launch-grid sharding as _rope_norm_fwd (Z10).
        """
        BT, H, D = x.shape
        T = cos.shape[0]
        Dh = D // 2
        dt = x.dtype
        P = _NKI_PARTITION
        assert lanes in (1, 2) and BT % (P * lanes) == 0 and T % (P * lanes) == 0
        BC = [[H, P], [1, H], [0, D]]
        CS = [[Dh, P], [0, H], [1, Dh]]
        dx = nl.ndarray((BT, H, D), dtype=dt, buffer=nl.shared_hbm)
        lane = nl.program_id(0) if lanes > 1 else 0
        tiles_per_seq = T // (P * lanes)
        for i in nl.affine_range(BT // (P * lanes)):
            r0 = (i * lanes + lane) * P
            t0 = ((i % tiles_per_seq) * lanes + lane) * P
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

    from torch_neuronx import wrap_nki as _wrap_rope_norm_nki
    _rope_norm_fwd_native = _wrap_rope_norm_nki(_rope_norm_fwd)
    _rope_norm_bwd_native = _wrap_rope_norm_nki(_rope_norm_bwd)

    def _rope_lanes(rows: int, t: int) -> int:
        """Launch grid for the rope kernels: selected_lanes on the row count, and the
        sequence length must also split into whole 128-row tiles per lane."""
        lanes = selected_lanes(rows)
        return min(lanes if t % (_NKI_PARTITION * lanes) == 0 else 1, _NKI_LANES)

    class _RopeNormNKI(torch.autograd.Function):
        """Makes the fused rope+norm kernel differentiable so it can train."""

        @staticmethod
        def forward(ctx, x, cos, sin):
            b, t, h, d = x.shape
            x3 = x.reshape(b * t, h, d)
            cos2 = cos.reshape(-1, d // 2)
            sin2 = sin.reshape(-1, d // 2)
            if Z10_ROPE_WRAP_NKI:
                # Z10: wrap_nki custom op on a [lanes] grid, as the CE and conv kernels.
                lanes = _rope_lanes(b * t, t)
                y = _rope_norm_fwd_native[lanes](x3, cos2, sin2, lanes).reshape(b, t, h, d)
            else:
                lanes = 1
                y = _rope_norm_fwd(x3, cos2, sin2).reshape(b, t, h, d)
            ctx.save_for_backward(x3, cos2, sin2)
            ctx.x_shape = (b, t, h, d)
            ctx.lanes = lanes
            return y

        @staticmethod
        def backward(ctx, grad_out):
            x3, cos2, sin2 = ctx.saved_tensors
            b, t, h, d = ctx.x_shape
            g3 = grad_out.reshape(b * t, h, d).contiguous()
            if Z10_ROPE_WRAP_NKI:
                dx3 = _rope_norm_bwd_native[ctx.lanes](x3, g3, cos2, sin2, ctx.lanes)
            else:
                dx3 = _rope_norm_bwd(x3, g3, cos2, sin2)
            return (dx3.reshape(b, t, h, d), None, None)

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

    @nki.jit
    def _softcap_ce2_fwd(z, tgt1, tgt2, lanes: int=2, target_gather: bool=False):
        """Two-target softcap15 cross-entropy forward sharing one softmax pass.

        Same tiling, sharding, softcap and precision as _softcap_ce_fwd: the
        exp/sum pass is stabilised by the first target's softcapped logit, so
        loss1 and the saved logsumexp come from the same instructions as the
        single-target kernel; loss2 = logsumexp - softcap(z[tgt2]) is one extra
        [128, 1] gather and subtract per tile. Used by the MTP objective, whose
        two targets share every logit-sized quantity.

        Args:
            z: Contiguous FP32 or BF16 [N, V] logits. BF16 is widened in
                SBUF; all loss arithmetic and saved statistics remain FP32.
            tgt1, tgt2: FP32 [N, 1] class indices, exact integers in [0, V).
            lanes: Physical cores in the matching launch grid.
            target_gather: Gather one SBUF element per row instead of mask/sum.
        Returns:
            FP32 [N, 1] loss1, loss2 and logsumexp.
        """
        rows_count, vocab = z.shape
        partitions = 128
        softcap = 15.0
        assert lanes in (1, 2) and rows_count % (partitions * lanes) == 0
        assert z.dtype in (nl.float32, nl.bfloat16)
        assert tgt1.dtype == nl.float32 and tgt2.dtype == nl.float32
        assert tgt1.shape == (rows_count, 1) and tgt2.shape == (rows_count, 1) and vocab % 512 == 0
        loss1 = nl.ndarray((rows_count, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        loss2 = nl.ndarray((rows_count, 1), dtype=nl.float32, buffer=nl.shared_hbm)
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
            labels1 = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            labels2 = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            if z.dtype == nl.bfloat16:
                source = nl.ndarray((partitions, vocab), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=source, src=z[rows])
                nisa.tensor_copy(dst=logits, src=source)
            else:
                nisa.dma_copy(dst=logits, src=z[rows])
            nisa.dma_copy(dst=labels1, src=tgt1[rows])
            nisa.dma_copy(dst=labels2, src=tgt2[rows])
            capped_unit = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=capped_unit, op=nl.tanh, data=logits, scale=1.0 / softcap)
            target_unit1 = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            target_unit2 = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            if target_gather:
                gather_indices1 = nl.ndarray((partitions, 1), dtype=nl.uint32, buffer=nl.sbuf)
                gather_indices2 = nl.ndarray((partitions, 1), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gather_indices1, src=labels1)
                nisa.tensor_copy(dst=gather_indices2, src=labels2)
                nisa.nc_n_gather(dst=target_unit1, data=capped_unit, indices=gather_indices1)
                nisa.nc_n_gather(dst=target_unit2, data=capped_unit, indices=gather_indices2)
            else:
                mask = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=mask, data=indices, op0=nl.equal, operand0=labels1)
                nisa.tensor_tensor(dst=mask, data1=mask, data2=capped_unit, op=nl.multiply)
                nisa.tensor_reduce(dst=target_unit1, op=nl.add, data=mask, axis=(1,))
                nisa.tensor_scalar(dst=mask, data=indices, op0=nl.equal, operand0=labels2)
                nisa.tensor_tensor(dst=mask, data1=mask, data2=capped_unit, op=nl.multiply)
                nisa.tensor_reduce(dst=target_unit2, op=nl.add, data=mask, axis=(1,))
            target_value1 = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            negative_target1 = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            target_value2 = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=target_value1, data=target_unit1, op0=nl.multiply, operand0=softcap)
            nisa.tensor_scalar(dst=negative_target1, data=target_unit1, op0=nl.multiply, operand0=-softcap)
            nisa.tensor_scalar(dst=target_value2, data=target_unit2, op0=nl.multiply, operand0=softcap)
            partial_sums = nl.ndarray((partitions, chunks), dtype=nl.float32, buffer=nl.sbuf)
            for part in nl.affine_range(chunks):
                columns = nl.ds(part * chunk, chunk)
                nisa.activation(dst=logits[:, columns], op=nl.exp, data=capped_unit[:, columns], scale=softcap, bias=negative_target1, reduce_op=nl.add, reduce_res=partial_sums[:, nl.ds(part, 1)], reduce_cmd=nisa.reduce_cmd.reset_reduce)
            denominator = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=denominator, op=nl.add, data=partial_sums, axis=(1,))
            log_denominator = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=log_denominator, op=nl.log, data=denominator)
            logsumexp = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=logsumexp, data1=target_value1, data2=log_denominator, op=nl.add)
            second_loss = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=second_loss, data1=logsumexp, data2=target_value2, op=nl.subtract)
            nisa.dma_copy(dst=loss1[rows], src=log_denominator)
            nisa.dma_copy(dst=loss2[rows], src=second_loss)
            nisa.dma_copy(dst=lse[rows], src=logsumexp)
        return (loss1, loss2, lse)

    @nki.jit
    def _softcap_ce2_bwd(z, tgt1, tgt2, lse, g1, g2, lanes: int=2, round_bf16_output: bool=False):
        """Return ONE [N, V] gradient for both targets, normally in the logits dtype.

        Per row, with p = softmax(softcap(z)) rebuilt from the saved logsumexp:
        dz = ((g1 + g2) * p - g1 * onehot(tgt1) - g2 * onehot(tgt2)) * (1 - tanh(z / 15)^2),
        i.e. the sum of the two _softcap_ce_bwd gradients autograd used to add
        on [N, V]. The instruction sequence is _softcap_ce_bwd's with the
        combined coefficient (g1 + g2) and one extra one-hot subtraction.

        Args:
            z, tgt1, tgt2, lse: Same inputs and saved logsumexp as forward.
            g1, g2: Arbitrary FP32 [N, 1] upstream gradients of loss1 and loss2.
            lanes: Physical cores in the matching launch grid.
            round_bf16_output: Same head-only contract as _softcap_ce_bwd.
        Returns:
            FP32 softcap15/CE derivative, rounded at the output when the input
            was BF16 or round_bf16_output is enabled (as _softcap_ce_bwd).
        """
        rows_count, vocab = z.shape
        partitions = 128
        softcap = 15.0
        assert lanes in (1, 2) and rows_count % (partitions * lanes) == 0
        assert z.dtype in (nl.float32, nl.bfloat16)
        assert tgt1.dtype == nl.float32 and tgt2.dtype == nl.float32
        assert lse.dtype == nl.float32 and g1.dtype == nl.float32 and g2.dtype == nl.float32
        assert tgt1.shape == (rows_count, 1) and tgt2.shape == (rows_count, 1)
        assert lse.shape == (rows_count, 1) and g1.shape == (rows_count, 1) and g2.shape == (rows_count, 1)
        output_dtype = nl.bfloat16 if round_bf16_output else z.dtype
        dz = nl.ndarray(z.shape, dtype=output_dtype, buffer=nl.shared_hbm)
        indices = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=indices, pattern=[[1, vocab]], offset=0, channel_multiplier=0)
        for tile in nl.affine_range(rows_count // (partitions * lanes)):
            rows = nl.ds((tile * lanes + nl.program_id(0)) * partitions, partitions)
            logits = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            labels1 = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            labels2 = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            gradient1 = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            gradient2 = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            gradient_sum = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            negative_lse = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            if z.dtype == nl.bfloat16:
                source = nl.ndarray((partitions, vocab), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=source, src=z[rows])
                nisa.tensor_copy(dst=logits, src=source)
            else:
                nisa.dma_copy(dst=logits, src=z[rows])
            nisa.dma_copy(dst=labels1, src=tgt1[rows])
            nisa.dma_copy(dst=labels2, src=tgt2[rows])
            nisa.dma_copy(dst=gradient1, src=g1[rows])
            nisa.dma_copy(dst=gradient2, src=g2[rows])
            nisa.dma_copy(dst=negative_lse, src=lse[rows])
            nisa.tensor_scalar(dst=negative_lse, data=negative_lse, op0=nl.multiply, operand0=-1.0)
            nisa.tensor_tensor(dst=gradient_sum, data1=gradient1, data2=gradient2, op=nl.add)
            capped_unit = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=capped_unit, op=nl.tanh, data=logits, scale=1.0 / softcap)
            nisa.activation(dst=logits, op=nl.exp, data=capped_unit, scale=softcap, bias=negative_lse)
            temporary = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=temporary, data=indices, op0=nl.equal, operand0=labels1, op1=nl.multiply, operand1=gradient1)
            nisa.scalar_tensor_tensor(dst=logits, data=logits, op0=nl.multiply, operand0=gradient_sum, op1=nl.subtract, operand1=temporary)
            nisa.tensor_scalar(dst=temporary, data=indices, op0=nl.equal, operand0=labels2, op1=nl.multiply, operand1=gradient2)
            nisa.tensor_tensor(dst=logits, data1=logits, data2=temporary, op=nl.subtract)
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
    _softcap_ce2_fwd_native = _wrap_softcap_ce_nki(_softcap_ce2_fwd)
    _softcap_ce2_bwd_native = _wrap_softcap_ce_nki(_softcap_ce2_bwd)

    class _SoftcapCrossEntropyNKI(torch.autograd.Function):
        """Makes the fused softcap+cross-entropy kernel differentiable.

        Forward returns the UNREDUCED per-token loss, exactly like
        F.cross_entropy(..., reduction="none"); mean/none handling stays in the
        caller so this changes no reduction semantics.
        """

        @staticmethod
        def forward(ctx, z, tgt):
            lanes = selected_lanes(z.shape[0])
            loss, lse = _softcap_ce_fwd_native[lanes](z, tgt, lanes, True)
            ctx.save_for_backward(z, tgt, lse)
            return loss.reshape(-1)

        @staticmethod
        def backward(ctx, grad_out):
            z, tgt, lse = ctx.saved_tensors
            g = grad_out.reshape(-1, 1).float().contiguous()
            lanes = selected_lanes(z.shape[0])
            return (_softcap_ce_bwd_native[lanes](z, tgt, lse, g, lanes), None)

    class _SoftcapCrossEntropy2NKI(torch.autograd.Function):
        """Two-target softcap+cross-entropy sharing one softmax pass (the MTP objective).

        Forward returns the two UNREDUCED per-token losses, exactly like two
        F.cross_entropy(..., reduction="none") calls on the same softcapped
        logits; mask/weight/mean handling stays in the caller. Backward runs
        one kernel that writes the combined [N, V] gradient, replacing the two
        single-target backward passes and the [N, V] add autograd used to emit.
        """

        @staticmethod
        def forward(ctx, z, tgt1, tgt2):
            lanes = selected_lanes(z.shape[0])
            loss1, loss2, lse = _softcap_ce2_fwd_native[lanes](z, tgt1, tgt2, lanes, True)
            ctx.save_for_backward(z, tgt1, tgt2, lse)
            return (loss1.reshape(-1), loss2.reshape(-1))

        @staticmethod
        def backward(ctx, grad1, grad2):
            z, tgt1, tgt2, lse = ctx.saved_tensors
            g1 = grad1.reshape(-1, 1).float().contiguous()
            g2 = grad2.reshape(-1, 1).float().contiguous()
            lanes = selected_lanes(z.shape[0])
            return (_softcap_ce2_bwd_native[lanes](z, tgt1, tgt2, lse, g1, g2, lanes), None, None)
    from torch_neuronx import wrap_nki as _wrap_conv2_nki

    def _conv2_taps(wt, channels):
        """Broadcast the previous/current [2, C] taps to [128, C] SBUF tiles."""
        prev = nl.ndarray((128, channels), dtype=wt.dtype, buffer=nl.sbuf)
        curr = nl.ndarray((128, channels), dtype=wt.dtype, buffer=nl.sbuf)
        pattern = [[0, 128], [1, channels]]
        nisa.dma_copy(dst=prev, src=wt.ap(pattern=pattern, offset=0))
        nisa.dma_copy(dst=curr, src=wt.ap(pattern=pattern, offset=channels))
        return (prev, curr)

    def _conv2_fwd_tile(x, out, prev_w, curr_w, batch, start, channels, first):
        """Write one causal 128-token tile, zero-padding only the first previous row."""
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
        """Return causal two-tap [B, T, C] outputs with disjoint batch ownership per lane."""
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
        """Store unchanged FP32 128-token dW reductions in one contiguous channel row."""
        prod = nl.ndarray((128, channels), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=xpart, data2=grad, op=nl.multiply)
        packed = nl.ndarray((1, channels), dtype=nl.float32, buffer=nl.sbuf)
        for c in nl.affine_range(channels // 128):
            summed = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
            result = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.nc_matmul(dst=summed, stationary=prod[:, c * 128:(c + 1) * 128], moving=ones, accumulate=False)
            nisa.tensor_copy(dst=result, src=summed)
            # Repack the unchanged FP32 reduction; no new arithmetic or HBM scratch.
            transposed = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=transposed, data=result, engine=nisa.engine.tensor)
            nisa.tensor_copy(dst=packed[:, c * 128:(c + 1) * 128], src=transposed, engine=nisa.engine.vector)
        offset = ((batch * partials.shape[1] + tile) * 2 + tap) * channels
        nisa.dma_copy(dst=partials.ap(pattern=[[channels, 1], [1, channels]], offset=offset), src=packed)

    def _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones, batch, tile, length, channels, first, last):
        """Write one dX tile and both FP32 dW partials, respecting sequence boundaries."""
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
        """Return dX and FP32 [B, T/128, 2, C, 1] partials for the later sum/casts."""
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
            lanes = 2 if x.shape[0] % 2 == 0 and _NKI_LANES == 2 else 1
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
    if _HAS_NKI and logits.device.type == 'neuron' and (loss_reduction in ('mean', 'none')) and (logits.dtype in (torch.float32, torch.bfloat16)) and logits.is_contiguous():
        z = logits.reshape(-1, logits.shape[-1])
        if z.shape[0] % _NKI_PARTITION == 0:
            tgt = targets.reshape(-1, 1).to(torch.float32)
            per_token = _SoftcapCrossEntropyNKI.apply(z, tgt)
            if loss_reduction == 'none':
                out = per_token.view_as(targets)
            else:
                out = per_token.mean()
            return out
    capped = LOGIT_SOFTCAP * torch.tanh(logits.float() / LOGIT_SOFTCAP)
    loss = F.cross_entropy(capped.reshape(-1, capped.size(-1)), targets.reshape(-1), reduction=loss_reduction)
    if loss_reduction == 'none':
        out = loss.view_as(targets)
    else:
        out = loss
    return out
# One fused two-target CE kernel pair for the MTP objective instead of two single-target pairs plus a
# [N, V] gradient add. False restores the original two-call path (bisect switch; identical math).
MTP_FUSED_CE = True

def softcap_cross_entropy2(logits: torch.Tensor, targets1: torch.Tensor, targets2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unreduced softcap15 cross-entropy of the same logits against two target tensors of one shape,
    sharing the softcap and the softmax. On Neuron, contiguous fp32/bf16 logits with a row count
    divisible by 128 use the fused two-target NKI kernel pair (one forward pass, one combined
    gradient); other inputs use the eager chain: one softcap, F.cross_entropy twice. Returns
    (first, second) shaped like the targets, exactly what two softcap_cross_entropy(..., 'none')
    calls return; any mask/weight/mean reduction stays with the caller.
    """
    if _HAS_NKI and logits.device.type == 'neuron' and (logits.dtype in (torch.float32, torch.bfloat16)) and logits.is_contiguous():
        z = logits.reshape(-1, logits.shape[-1])
        if z.shape[0] % _NKI_PARTITION == 0:
            tgt1 = targets1.reshape(-1, 1).to(torch.float32)
            tgt2 = targets2.reshape(-1, 1).to(torch.float32)
            first, second = _SoftcapCrossEntropy2NKI.apply(z, tgt1, tgt2)
            return (first.view_as(targets1), second.view_as(targets2))
    capped = LOGIT_SOFTCAP * torch.tanh(logits.float() / LOGIT_SOFTCAP)
    flat = capped.reshape(-1, capped.size(-1))
    first = F.cross_entropy(flat, targets1.reshape(-1), reduction='none').view_as(targets1)
    second = F.cross_entropy(flat, targets2.reshape(-1), reduction='none').view_as(targets2)
    return (first, second)


def mtp_auxiliary_weight(progress: float) -> float:
    # Host-side schedule only; every model invocation still receives a[2] tensor.
    return 0.5 * max(0.0, 1.0 - max(0.0, progress) / (2.0 / 3.0))

def mtp_cross_entropy(logits, targets, weights, reduction):
    """Parameter-free k1/k2 objective, inspired by modded-nanogpt PR178.

    https://github.com/KellerJordan/modded-nanogpt/pull/178 (MIT).
    Independently implemented. No extra head or eval change. Do not branch on
    weights: both same-shape CE calls remain when k2 weight reaches zero.
    Per-row last-position k2 has no valid target, and is masked, never wrapped.
    """
    assert targets.ndim == 2 and targets.shape[1] >= 2
    assert weights.shape == (2,)
    shifted = torch.cat((targets[:, 1:], torch.zeros_like(targets[:, :1])), dim=1)
    if MTP_FUSED_CE:
        # One softcap/softmax pass and one combined logits gradient for both targets.
        first, second = softcap_cross_entropy2(logits, targets, shifted)
    else:
        first = softcap_cross_entropy(logits, targets, 'none')
        second = softcap_cross_entropy(logits, shifted, 'none')
    mask = torch.cat((torch.ones_like(second[:, :-1]), torch.zeros_like(second[:, :1])), dim=1)
    second = second * mask * (targets.shape[1] / (targets.shape[1] - 1))
    per_token = (weights[0] * first + weights[1] * second) / weights.sum()
    if reduction == 'none':
        return per_token
    if reduction == 'mean':
        return per_token.mean()
    raise ValueError('MTP supports mean/none only')

class Linear(nn.Linear):
    """Bias-free linear with a differentiable cast cache refreshed after each update."""

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

# R113: the native-tested R110 wrapped channel-major pair, not the older address adapter.
# No CPU/timing/environment switch selects a different train/eval graph on Neuron.
try:
    from torch_neuronx import wrap_nki as _wrap_laneb_nki
    try:
        from nkilib.core.attention.attention_cte import attention_cte as _laneb_cte_kernel
        from nkilib.core.attention.attention_bwd import attention_bwd as _laneb_bwd_kernel
    except ImportError:
        from nkilib_src.nkilib.core.attention.attention_cte import attention_cte as _laneb_cte_kernel
        from nkilib_src.nkilib.core.attention.attention_bwd import attention_bwd as _laneb_bwd_kernel
    _LANEB_CTE = _wrap_laneb_nki(_laneb_cte_kernel)[_NKI_LANES]
    _LANEB_BWD = _wrap_laneb_nki(_laneb_bwd_kernel)[_NKI_LANES]
except ImportError:
    _LANEB_CTE = None
    _LANEB_BWD = None


class _LaneBWrappedAttention(torch.autograd.Function):
    """R110 native-tested channel-major Function; copies are explicit model costs."""
    @staticmethod
    def forward(ctx, q_c, k_c, v_c, scale):
        b, h, d, t = v_c.shape
        v_t = v_c.reshape(b * h, d, t).transpose(1, 2).contiguous()
        o, neg_max, sum_recip = _LANEB_CTE(
            q_c, k_c, v_t, scale=scale, causal_mask=True, tp_q=False,
            tp_k=False, tp_out=True, cache_softmax=True)
        lse = -(neg_max.float() + torch.log(sum_recip.float()))
        lse = lse.reshape(b, h, 128, t // 128).contiguous()
        ctx.save_for_backward(q_c, k_c, v_c, o, lse)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, d_o):
        q_c, k_c, v_c, o, lse = ctx.saved_tensors
        b, h, d, t = v_c.shape
        outs = _LANEB_BWD(
            q_c.view(b, h, d, t), k_c.view(b, h, d, t), v_c,
            o.view(b, h, d, t), d_o.contiguous().view(b, h, d, t), lse,
            use_causal_mask=True, mixed_precision=True, softmax_scale=ctx.scale)
        return outs[0].view(b * h, d, t), outs[1].view(b * h, d, t), outs[2], None


def _lane_b_native_attention(q, k, v, scale):
    """B,H,T,D in/out; same native path for training and targets=None evaluation."""
    if _LANEB_CTE is None or _LANEB_BWD is None:
        raise RuntimeError('LANEB_NATIVE_LIBRARY_UNAVAILABLE')
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError('LANEB_EQUAL_BHTD_REQUIRED')
    b, h, t, d = q.shape
    if b < 1 or h < 1 or t < 512 or t % 512 or d < 1 or d > 128:
        raise ValueError('LANEB_NATIVE_SHAPE_UNSUPPORTED')
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('LANEB_BF16_REQUIRED')
    if q.device != k.device or q.device != v.device:
        raise ValueError('LANEB_SAME_DEVICE_REQUIRED')
    scale = d ** -0.5 if scale is None else scale
    q_c = q.transpose(2, 3).reshape(b * h, d, t).contiguous()
    k_c = k.transpose(2, 3).reshape(b * h, d, t).contiguous()
    v_c = v.transpose(2, 3).contiguous()
    o = _LaneBWrappedAttention.apply(q_c, k_c, v_c, scale)
    return o.view(b, h, d, t).transpose(2, 3)


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig, layer_idx: int=0):
        super().__init__()
        self.n_head = config.n_head
        self.lane_b = config.lane_b
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim * self.n_head == self.n_embd
        self.n_kv_heads = config.n_kv_heads or config.n_head
        self.kv_repeat = 1
        if self.n_kv_heads != self.n_head:
            if self.n_head % self.n_kv_heads != 0:
                raise ValueError(f'n_kv_heads={self.n_kv_heads} must divide n_head={self.n_head}')
            self.kv_repeat = self.n_head // self.n_kv_heads
        self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.c_v = Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_scale = config.attn_scale if config.attn_scale > 0.0 else None
        self.d1_injection = config.d1_injection
        self.d2_two_site = config.d1_detached and config.d2_two_site and layer_idx == 7
        if self.d2_two_site:
            with torch.random.fork_rng(devices=[]):
                self.bigram_gate = Linear(32, self.n_head, bias=False)
        self.ve_gate = Linear(VE_GATE_CHANNELS, self.n_head, bias=False) if layer_idx % 2 == (config.n_layer - 1) % 2 else None

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, bigram_ve: torch.Tensor | None=None) -> torch.Tensor:
        b, t, c = x_q.shape
        q = self.c_q(x_q).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x_k).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.c_v(x_v)
        ve_gate_input = None
        bigram_gate_input = None
        if Z3_ONE_GATE_SLICE and ve is not None and bigram_ve is not None and self.d2_two_site:
            # Z3: one contiguous [.., :64] slice feeds both gates. The two sub-slices
            # hold the same elements as x_v[..., :32] and x_v[..., 32:64]; in backward
            # the two [BT, 64] slice gradients occupy disjoint channels, so their sum is
            # exact and only one full-width slice_backward zero-fill remains.
            gate_input = x_v[..., :2 * VE_GATE_CHANNELS]
            ve_gate_input = gate_input[..., :VE_GATE_CHANNELS]
            bigram_gate_input = gate_input[..., 32:64]
        if bigram_ve is not None and (self.d2_two_site or self.d1_injection in ('value_pre_view', 'value_pre_view_contiguous_gate')):
            v = v + self.bigram_delta(x_v, bigram_ve, bigram_gate_input)
        v = v.view(b, t, self.n_kv_heads, self.head_dim)
        if self.n_kv_heads != self.n_head:
            k = k.repeat_interleave(self.kv_repeat, dim=2).contiguous()
            v = v.repeat_interleave(self.kv_repeat, dim=2).contiguous()
        if ve is not None:
            ve = ve.view(b, t, self.n_head, self.head_dim)
            if ve_gate_input is None:
                ve_gate_input = x_v[..., :VE_GATE_CHANNELS]
            gate = 2 * torch.sigmoid(self.ve_gate(ve_gate_input))
            v = v + gate.unsqueeze(-1) * ve
        v = v.to(q.dtype)
        q = rope_norm(q, cos, sin)
        k = rope_norm(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.lane_b and q.device.type == 'neuron':
            y = _lane_b_native_attention(q, k, v, self.attn_scale)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        if bigram_ve is not None and self.d1_injection == 'attention_pre_projection':
            y = y + self.bigram_delta(x_v, bigram_ve)
        return self.c_proj(y)

    def bigram_delta(self, x_v: torch.Tensor, bigram_ve: torch.Tensor, gate_input: torch.Tensor | None=None) -> torch.Tensor:
        if gate_input is None:
            gate_input = x_v[..., 32:64]
        if self.d1_injection == 'value_pre_view_contiguous_gate':
            gate_input = gate_input.contiguous()
        gate = 2 * torch.sigmoid(self.bigram_gate(gate_input))
        return gate.to(bigram_ve.dtype).repeat_interleave(self.head_dim, dim=-1) * bigram_ve

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

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.c_fc = Linear(config.n_embd, hidden, bias=False)
        self.c_proj = Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.numel() // x.shape[-1]
        tile = min(512, tokens // 2)
        # Keep the eager fallback for non-BF16 inputs or unsupported token tiling.
        if x.dtype != torch.bfloat16 or (tile < 128) or tile % 128 or tokens % (2 * tile):
            h = self.c_fc(x)
            return self.c_proj(F.relu(h).square())
        up = _linear_weight(self.c_fc, x.dtype)
        down = _linear_weight(self.c_proj, x.dtype)
        result = _FusedMLP.apply(x.reshape(tokens, x.shape[-1]), up, down)
        return result.reshape(x.shape)

class Block(nn.Module):

    def __init__(self, config: GPTConfig, layer_idx: int=0):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx) if layer_idx in ATTENTION_LAYERS else None
        self.mlp = MLP(config)
        self.resid_lambda = nn.Parameter(torch.ones(1))
        self.x0_lambda = nn.Parameter(torch.zeros(1))
        self.local_conv = CausalConv(config.n_embd, LOCAL_CONV_KERNEL) if self.attn is not None else None

    def forward(self, x: torch.Tensor, x0: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, bigram_ve: torch.Tensor | None=None) -> torch.Tensor:
        x = self.resid_lambda.to(x.dtype) * x + self.x0_lambda.to(x.dtype) * x0
        if self.attn is None:
            return x + self.mlp(norm(x))
        if Z1_NORM_ONCE:
            # Z1: one norm of the pre-conv stream feeds the conv input and q/k/v.
            # Forward values are bit-identical (same function, same input); the
            # backward sums the four branch gradients before one norm backward
            # instead of after four, which only changes BF16 accumulation order.
            xn = norm(x)
            if self.local_conv is not None:
                x = x + self.local_conv(xn)
            v_norm = xn
            attn_out = self.attn(xn, xn, xn, ve, cos, sin, bigram_ve)
        else:
            q_in = k_in = v_in = x
            if self.local_conv is not None:
                x = x + self.local_conv(norm(x))
            if bigram_ve is None:
                # Exact parent evaluation order matters for BF16 backward accumulation.
                attn_out = self.attn(norm(q_in), norm(k_in), norm(v_in), ve, cos, sin)
            else:
                q_norm = norm(q_in)
                k_norm = norm(k_in)
                v_norm = norm(v_in)
                attn_out = self.attn(q_norm, k_norm, v_norm, ve, cos, sin, bigram_ve)
        x = x + attn_out
        if bigram_ve is not None and self.attn.d1_injection == 'residual_post_attention':
            x = x + self.attn.bigram_delta(v_norm, bigram_ve)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_layer != 8:
            raise ValueError('this submission requires exactly eight layers')
        self.config = config
        if config.d1_injection != 'embedding_pre_norm':
            raise ValueError('unsupported pinned D1 injection site')
        if config.tbl_every not in (1, 2):
            raise ValueError('tbl_every must be explicitly selected as 1 or 2')
        if type(config.d2_two_site) is not bool or type(config.tbl_freeze_tokens) is not int or config.tbl_freeze_tokens < 0:
            raise ValueError('D2 boolean and nonnegative integer token-freeze configuration required')
        if config.d2_two_site and (not config.d1_detached or config.n_embd < 64 or config.n_kv_heads not in (0, config.n_head)):
            raise ValueError('D2 requires enabled D1, width>=64 and full query/value heads')
        if config.d2_two_site and config.n_embd != 128 * config.n_head:
            raise ValueError('D2 NOPROJ-ID requires 128 channels per head')
        self.__dict__['_d1_compiled_core'] = None
        padded_vocab_size = (config.vocab_size + 63) // 64 * 64
        self.transformer = nn.ModuleDict({'wte': nn.Embedding(padded_vocab_size, config.n_embd), 'h': nn.ModuleList([Block(config, i) for i in range(config.n_layer)])})
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, config.n_embd) for i in range(config.n_layer) if i % 2 == (config.n_layer - 1) % 2 and i in ATTENTION_LAYERS})
        # Architecture inspiration: Ritwika Kancharla, Trainium of Thought (Sep 2026),
        # https://ritwikareddykancharla.github.io/projects/trainium-of-thought.html
        # Encoder i feeds input n-1-i; zero-initialized FP32 scalar gates.
        self.unet_n_enc = config.n_layer // 2
        self.unet_skip_gates = nn.ParameterList([nn.Parameter(torch.zeros(1, dtype=torch.float32))
                                               for _ in range(self.unet_n_enc)])
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, config.n_embd // config.n_head)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
        if config.d1_detached:
            table_shape = (262144, 256) if config.d2_two_site else (524288, 128)
            self.register_buffer('value_bigram_table', torch.zeros(*table_shape, dtype=torch.bfloat16), persistent=True)
            self.value_bigram_gate_logit = nn.Parameter(torch.zeros((), dtype=torch.float32))
            with torch.random.fork_rng(devices=[]):
                self.value_bigram_proj = Linear(128, config.n_embd, bias=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = INIT_SCALE * math.sqrt(3.0) * self.config.n_embd ** (-0.5)
        for block in self.transformer.h:
            if block.attn is not None:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            if block.attn is not None:
                b = ATTN_CPROJ_INIT_SCALE * s
                torch.nn.init.uniform_(block.attn.c_proj.weight, -b, b)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.resid_lambda is not None:
                block.resid_lambda.fill_(1.0)
                block.x0_lambda.fill_(0.05)
            if block.attn is not None and block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for gate in self.unet_skip_gates:
            torch.nn.init.zeros_(gate)
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, self.config.n_embd // self.config.n_head)
        self.cos.copy_(cos)
        self.sin.copy_(sin)
        if self.config.d1_detached:
            self.value_bigram_table.zero_()
            with torch.random.fork_rng(devices=[]):
                torch.nn.init.uniform_(self.value_bigram_proj.weight, -s, s)
                if self.config.d2_two_site:
                    self.transformer.h[7].attn.bigram_gate.weight.zero_()
            self.value_bigram_gate_logit.zero_()

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
        attn_flops = sum((block.attn is not None for block in self.transformer.h)) * 12 * h * d * t
        return 6 * (nparams - non_matmul) + attn_flops

    def setup_optimizer(self):
        scalar_params = []
        matrix_params = []
        local_conv_params = []
        for block in self.transformer.h:
            if block.local_conv is not None:
                local_conv_params.append(block.local_conv.weight)
            if block.resid_lambda is not None:
                scalar_params.extend([block.resid_lambda, block.x0_lambda])
            if block.attn is not None:
                matrix_params.extend(block.attn.parameters())
            matrix_params.extend(block.mlp.parameters())
        if self.config.d1_detached:
            matrix_params.append(self.value_bigram_proj.weight)
        gate_params = [p for name, p in self.named_parameters() if p.ndim == 2 and 'gate' in name]
        if self.config.d1_detached:
            gate_params.append(self.value_bigram_gate_logit)
        gate_ids = {id(p) for p in gate_params}
        matrix_params = [p for p in matrix_params if id(p) not in gate_ids]
        scalar_params.extend(self.unet_skip_gates)
        for p in scalar_params:
            if p.ndim > 1:
                raise ValueError(f'scalar_params must hold only scalars/1-D vectors (SCALAR_LR={SCALAR_LR} is tuned for those); got a {tuple(p.shape)} tensor. Route it to gate_params or to Muon instead.')
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        value_embed_params = list(self.value_embeds.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** (-0.5)
        param_groups = [dict(kind='adamw', params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, EMBED_ADAMW_BETA2), eps=1e-10, weight_decay=0.001), dict(kind='adamw', params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01), dict(kind='adamw', params=scalar_params, lr=SCALAR_LR, betas=(SCALAR_BETA1, SCALAR_BETA2), eps=1e-10, weight_decay=0.0)]
        if gate_params:
            param_groups.append(dict(kind='adamw', params=gate_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if value_embed_params:
            param_groups.append(dict(kind='adamw', params=value_embed_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, EMBED_ADAMW_BETA2), eps=1e-10, weight_decay=0.0))
        if local_conv_params:
            param_groups.append(dict(kind='adamw', params=local_conv_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        for shape, transposed in sorted({(p.shape, False) for p in matrix_params}, key=lambda st: (st[0], st[1])):
            params = [p for p in matrix_params if p.shape == shape and False == transposed]
            group = dict(kind='muon', params=params, lr=MATRIX_LR, momentum=0.95, beta2=MUON_BETA2, ns_steps=MUON_NS_STEPS, weight_decay=WEIGHT_DECAY, transposed=transposed)
            param_groups.append(group)
        opt = MuonAdamW(param_groups)
        for group in opt.param_groups:
            group['initial_lr'] = group['lr']
            if group.get('betas') is not None:
                group['initial_betas'] = tuple(group['betas'])
        return opt

    def bigram_hash(self, idx: torch.Tensor) -> torch.Tensor:
        """R54: authoritative row-local self-at-t0 int32 hash, called by forward."""
        cur = idx.to(torch.int32)
        prev = torch.cat((cur[:, :1], cur[:, :-1]), dim=1)
        return torch.remainder(torch.bitwise_xor(prev * 120193, cur * 128201), self.value_bigram_table.shape[0] - 1)

    def compile_core(self, backend: str) -> None:
        # A plain callable, not a registered submodule/checkpoint namespace.
        # Z11: fullgraph=True turns any silent graph break in the core into a
        # compile-time error (an assertion that Z10 holds), never a math change.
        self.__dict__['_d1_compiled_core'] = torch.compile(self.forward_core, backend=backend, dynamic=False, fullgraph=Z11_FULLGRAPH_CORE)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None=None, loss_reduction: str='mean', mtp_weights: torch.Tensor | None=None, table_context: dict | None=None):
        rows = None
        if self.config.d1_detached:
            hash_ids = self.bigram_hash(idx)
            rows = self.value_bigram_table[hash_ids]
            if table_context is not None:
                if table_context or not torch.is_grad_enabled():
                    raise ValueError('new empty training context with gradients enabled required')
                rows = rows.detach().requires_grad_(True)
                # Keep this microbatch's IDs independent of reusable loader storage.
                table_context.update(indices=hash_ids.detach().clone(), leaf=rows, tokens=idx.numel())
        elif table_context is not None:
            raise ValueError('table context supplied while D1 is disabled')
        core = self.__dict__.get('_d1_compiled_core')
        if core is None:
            core = self.forward_core
        return core(idx, rows, targets, loss_reduction, mtp_weights)

    def forward_core(self, idx: torch.Tensor, rows: torch.Tensor | None, targets: torch.Tensor | None=None, loss_reduction: str='mean', mtp_weights: torch.Tensor | None=None):
        _b, t = idx.shape
        assert t <= self.config.sequence_len
        cos = self.cos[:, :t]
        sin = self.sin[:, :t]
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        bigram_ve7 = None
        if self.config.d1_detached:
            if rows is None:
                raise ValueError('D1 core requires the shared-forward lookup values')
            rows0 = rows[..., :128] if self.config.d2_two_site else rows
            projected = self.value_bigram_proj(rows0)
            if self.config.d2_two_site:
                bigram_ve7 = rows[..., 128:256].repeat(1, 1, self.config.n_head)
            gate = (2 * torch.sigmoid(self.value_bigram_gate_logit)).to(x.dtype)
            x = x + gate * projected
        x = norm(x)
        x0 = x
        unet_skips: list[torch.Tensor] = []
        for i, block in enumerate(self.transformer.h):
            if i >= self.unet_n_enc:
                x = x + self.unet_skip_gates[i - self.unet_n_enc].to(x.dtype) * unet_skips.pop()
            ve = self.value_embeds[str(i)](idx).to(COMPUTE_DTYPE) if block.attn is not None and str(i) in self.value_embeds else None
            if self.config.d2_two_site and i == 7:
                x = block(x, x0, ve, cos, sin, bigram_ve7)
            else:
                x = block(x, x0, ve, cos, sin)
            if i < self.unet_n_enc:
                unet_skips.append(x)
        x = norm(x)
        logits = self.lm_head(x)
        if not (Z5_SKIP_VOCAB_SLICE and logits.shape[-1] == self.config.vocab_size):
            # Z5: the full-range slice is an alias on torch 2.14, but an older SDK
            # torch may trace it as slice + slice_backward (a [BT, V] zero-fill and
            # copy), so it is only taken when the padded vocab really is wider.
            logits = logits[..., :self.config.vocab_size]
        if targets is None:
            return LOGIT_SOFTCAP * torch.tanh(logits.float() / LOGIT_SOFTCAP)
        if self.config.mtp:
            if mtp_weights is None:
                mtp_weights = torch.tensor([1.0, 0.0], dtype=torch.float32, device=logits.device)
            return mtp_cross_entropy(logits, targets, mtp_weights, loss_reduction)
        return softcap_cross_entropy(logits, targets, loss_reduction)

def build_config(*, d1_detached: bool=True, d1_injection: str='embedding_pre_norm', tbl_every: int=2, d2_two_site: bool=True, tbl_freeze_tokens: int=286000000) -> GPTConfig:
    """Bare candidate training is ON; legacy saved GPTConfig dictionaries default OFF."""
    return GPTConfig(sequence_len=SEQ_LEN, vocab_size=TOKENIZER_VOCAB_SIZE, n_layer=DEPTH, n_head=N_EMBD // HEAD_DIM, n_embd=N_EMBD, mlp_ratio=MLP_RATIO, attn_scale=ATTN_SCALE, n_kv_heads=N_KV_HEADS, lane_b=LANE_B, d1_detached=d1_detached, d1_injection=d1_injection, tbl_every=tbl_every, d2_two_site=d2_two_site, tbl_freeze_tokens=tbl_freeze_tokens)
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

def d1_table_grad(gbuf: torch.Tensor, indices: torch.Tensor, leaf_gradient: torch.Tensor, objective_weight: torch.Tensor) -> torch.Tensor:
    """Duplicate-safe FP32 accumulation; explicit int64 index_put adapter only."""
    rows = indices.reshape(-1).to(torch.int64)
    values = leaf_gradient.reshape(-1, gbuf.shape[1]).float() * objective_weight
    return torch.index_put(gbuf, (rows,), values, accumulate=True)

def d1_rms_step(table: torch.Tensor, gradient: torch.Tensor, square_avg: torch.Tensor, lr: torch.Tensor, correction: torch.Tensor):
    """BF16 historical operation stages; dynamic FP32 0-D LR/correction inputs."""
    squared = (gradient * gradient).to(torch.bfloat16)
    second = torch.lerp(square_avg, squared, 0.001).to(torch.bfloat16)
    denom = (second / correction).to(torch.bfloat16).sqrt().to(torch.bfloat16)
    denom = (denom + 1e-10).to(torch.bfloat16)
    update = (gradient / denom).to(torch.bfloat16)
    updated = (table.float() - lr * update.float()).to(torch.bfloat16)
    return updated, second, torch.zeros_like(gradient, dtype=torch.float32)

@torch.no_grad()
def d2_table_rank_max_abs_diff(table: torch.Tensor) -> float:
    """End-run diagnostic on a clone; never changes a trained table or master."""
    if not (dist.is_available() and dist.is_initialized()):
        return 0.0
    reference = table.detach().clone()
    dist.broadcast(reference, src=0)
    difference = table.detach().cpu().float()
    difference.sub_(reference.cpu().float()).abs_()
    maximum = torch.tensor(float(difference.max().item()), dtype=torch.float32, device=table.device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())

class _PersistentScalar:
    """Z7: a 0-dim FP32 device scalar refilled in place from a host buffer (the
    MuonAdamW._device_scalar pattern) instead of a fresh torch.tensor(...)
    allocation plus H2D copy on every call. The copy is skipped when the value
    is unchanged (the compiled consumers only read the scalar). fill_ rounds a
    Python number to FP32 exactly as torch.tensor(value, dtype=float32) does.
    """

    def __init__(self, device: torch.device):
        self.host = torch.empty((), dtype=torch.float32)
        self.dev = torch.zeros((), dtype=torch.float32, device=device)
        self.value = None

    def get(self, value) -> torch.Tensor:
        if self.value != value:
            self.host.fill_(float(value))
            self.dev.copy_(self.host)
            self.value = value
        return self.dev

class DetachedTableUpdate:
    """Training-only replicated state; never a model Parameter/module/checkpoint key.

    Each incoming leaf gradient already includes LOSS_SCALE/grad_accum_now.
    Undo only grad_accum_now and weight by microbatch token count; divide by
    all pending tokens once, so every2 retains a token-weighted mean of BOTH
    steps even across the batch-ramp boundary. No extra world-size division.
    """
    def __init__(self, model: GPT, *, compile_backend: str | None=None):
        if not model.config.d1_detached:
            raise ValueError('D1 updater requires enabled table')
        self.model = model
        self.table = model.value_bigram_table
        if self.table.dtype != torch.bfloat16 or self.table.requires_grad:
            raise ValueError('persistent BF16 non-autograd table buffer required')
        self.weight_scalar = _PersistentScalar(self.table.device)
        self.lr_scalar = _PersistentScalar(self.table.device)
        self.correction_scalar = _PersistentScalar(self.table.device)
        self.every = model.config.tbl_every
        self.d2_two_site = model.config.d2_two_site
        self.freeze_tokens = model.config.tbl_freeze_tokens if self.d2_two_site else 0
        self.frozen = False
        self.initial_lr = EMBEDDING_LR * (model.config.n_embd / 768) ** (-0.5)
        self.gbuf = torch.zeros_like(self.table, dtype=torch.float32)
        self.square_avg = torch.zeros_like(self.table, dtype=torch.bfloat16)
        self.pending_tokens = 0
        self.pending_steps = 0
        self.update_count = 0
        self.grad_function = d1_table_grad
        self.rms_function = d1_rms_step
        if compile_backend is not None:
            self.grad_function = torch.compile(d1_table_grad, backend=compile_backend, fullgraph=True, dynamic=False)
            self.rms_function = torch.compile(d1_rms_step, backend=compile_backend, fullgraph=True, dynamic=False)

    @torch.no_grad()
    def accumulate(self, context: dict, *, accumulation_steps: int) -> None:
        if self.frozen:
            # Retain the same compiled core/leaf backward graph; skip dense scatter.
            context.clear()
            return
        leaf = context['leaf']
        indices = context['indices']
        tokens = int(context['tokens'])
        if leaf.grad is None or leaf.grad.shape[:-1] != indices.shape or indices.dtype != torch.int32:
            raise RuntimeError('missing/mismatched D1 leaf gradient or int32 indices')
        if accumulation_steps < 1 or tokens <= 0:
            raise ValueError('positive objective/token normalization required')
        if Z7_PERSISTENT_SCALARS:
            # Z7: no allocation and no H2D copy right after loss.backward() unless
            # the value changes (once per run, at the batch-ramp boundary).
            weight = self.weight_scalar.get(accumulation_steps * tokens)
        else:
            weight = torch.tensor(accumulation_steps * tokens, dtype=torch.float32, device=self.table.device)
        self.gbuf = self.grad_function(self.gbuf, indices, leaf.grad, weight)
        self.pending_tokens += tokens
        context.clear()

    @torch.no_grad()
    def finish_step(self, *, loader_epoch: int, force: bool=False, abort: bool=False, freeze: bool=False) -> bool:
        if abort:
            raise RuntimeError('all-rank table control refused the charged training deadline')
        if self.frozen:
            return False
        self.pending_steps += 1
        if self.pending_steps < self.every and not (force or freeze):
            return False
        self._update(loader_epoch=loader_epoch)
        if freeze:
            self.frozen = True
        return True

    @torch.no_grad()
    def _update(self, *, loader_epoch: int) -> None:
        if self.pending_tokens <= 0:
            raise RuntimeError('table update without accumulated training tokens')
        # One cast only, at the explicitly selected dense BF16 AVG/RMS boundary.
        gradient = (self.gbuf / self.pending_tokens).to(torch.bfloat16)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(gradient, op=dist.ReduceOp.AVG)
        self.update_count += 1
        rate = self.initial_lr * 1.0 if self.d2_two_site else (0.0 if loader_epoch >= 2 else self.initial_lr * 1.0)
        if Z7_PERSISTENT_SCALARS:
            lr = self.lr_scalar.get(rate)
            correction = self.correction_scalar.get(1.0 - 0.999 ** self.update_count)
        else:
            lr = torch.tensor(rate, dtype=torch.float32, device=self.table.device)
            correction = torch.tensor(1.0 - 0.999 ** self.update_count, dtype=torch.float32, device=self.table.device)
        updated, second, cleared = self.rms_function(self.table, gradient, self.square_avg, lr, correction)
        if Z6_TABLE_REBIND:
            # Z6: the compiled core never sees the table (the gather in GPT.forward is
            # eager and reads the buffer attribute on every call), so re-registering
            # the fresh outputs under the same persistent buffer name replaces two
            # eager 128 MiB copies with pointer rebinds. state_dict() keeps the key
            # value_bigram_table; self.table tracks the model's buffer. Same rebind
            # pattern as self.gbuf = cleared below and the Muon stacked buffers.
            self.model.register_buffer('value_bigram_table', updated, persistent=True)
            self.table = self.model.value_bigram_table
            self.square_avg = second
        else:
            self.table.copy_(updated)
            self.square_avg.copy_(second)
        self.gbuf = cleared
        self.pending_tokens = 0
        self.pending_steps = 0

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
    """Average every available gradient across the training ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

def clip_and_sync_gradients(model: nn.Module, clip_main_params: list[torch.nn.Parameter], *, max_norm: float) -> None:
    """Average all dense gradients once; R50 table is excluded from every clip group."""
    sync_gradients(model)
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(clip_main_params, max_norm, foreach=None)

def synchronized_table_control(loader_epoch: int, previous_epoch: int, force: bool, expired: bool, device, freeze: bool=False):
    """One all-rank decision BEFORE branching into or refusing table AVG/RMS.

    Local clock estimates are advisory inputs only. MAX makes epoch freeze,
    force-partial and deadline refusal identical on every rank. No rank-local
    deadline exception may occur between this control and the table collective.
    """
    control = torch.tensor([max(loader_epoch, previous_epoch), int(force), int(expired), int(freeze)],
                           dtype=torch.int32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(control, op=dist.ReduceOp.MAX)
    return int(control[0].item()), bool(control[1].item()), bool(control[2].item()), bool(control[3].item())

def save_checkpoint(path: Path, model: GPT, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'model_config': asdict(model.config), 'meta': meta}
    torch.save(payload, path)
EVAL_MIN_SEQ_LEN = 8192

def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Return the trained model in eval mode. The rotary tables are not parameters and are sized for
    sequences up to EVAL_MIN_SEQ_LEN, so evaluation may use a longer context than training did.
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

def startup_adjusted_schedule_elapsed(step: int, elapsed: float, budget_started: float) -> float:
    """Optimizer progress only; never change the actual budget/stop clock.

    Reconstruct process age from the existing step-start sample, without a new
    clock read. Preserve charged excess above the same capped startup allowance.
    After the fifth boundary, the unchanged budget reset already accounts for it.
    """
    if step >= TRAIN_STARTUP_STEPS_EXCLUDED:
        return elapsed
    process_age = elapsed + (budget_started - _PROCESS_STARTED)
    return max(0.0, process_age + STARTUP_LAUNCH_MARGIN_SECONDS - TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)

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

def _batch_ramp_accum(progress: float, previous: int=1) -> int:
    """One-way 1->2 accumulation transition; never changes microbatch shapes."""
    return max(previous, 1 if progress < (1.0 / 3.0) else 2)


def _synchronized_batch_ramp_accum(progress: float, previous: int, device) -> int:
    # Rank zero's charged-time sample is authoritative. No rank may independently
    # change the number of forward/backward calls or gradient collectives.
    proposed = _batch_ramp_accum(progress, previous) if dist.get_rank() == 0 else 0
    phase = torch.tensor(proposed, dtype=torch.int32, device=device)
    dist.broadcast(phase, src=0)
    result = int(phase.item())
    if result not in (1, 2):
        raise RuntimeError('invalid synchronized batch-ramp phase')
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='CLEAN-GATEFIX fixed Phase 1 training recipe')
    parser.add_argument('--out-dir', type=str, default=os.environ.get('NEURON_COMPETITION_R1_OUT_DIR', 'out'), help='Checkpoint output directory.')
    parser.add_argument('--tbl_freeze_tokens', type=int, default=0, help='Freeze D2 table after the completed global-token boundary; 0 continues updates.')
    args = parser.parse_args()
    device_type = detect_device_type()
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type, seed=46)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    print0(f'device={device} dtype={COMPUTE_DTYPE} world_size={world_size}')
    print0(f'training target={NUM_STEPS} steps max_seconds={MAX_TRAIN_SECONDS} official_budget={TRAIN_TIME_BUDGET_SECONDS}s')
    print0(f'gradient_clip={GRAD_CLIP:g} clip_after_reduce={True}')
    tokenizer = ensure_tokenizer(build_if_missing=False)
    config = build_config(tbl_freeze_tokens=args.tbl_freeze_tokens)
    print0(f'model_config={json.dumps(asdict(config), indent=2)}')
    model = GPT(config)
    model.init_weights()
    model.to(device)
    num_params = sum((p.numel() for p in model.parameters()))
    flops_per_token = model.estimate_flops()
    print0(f'num_params={num_params:,} flops_per_token={flops_per_token:e}')
    orig_model = model
    compile_backend = 'neuron' if device_type == 'neuron' else 'inductor'
    if config.d1_detached:
        model.compile_core(compile_backend)
    else:
        model = torch.compile(model, backend=compile_backend, dynamic=False)
    print0(f'torch.compile enabled (backend={compile_backend})')
    world_tokens_per_micro = DEVICE_BATCH_SIZE * PACK_FACTOR * SEQ_LEN * world_size
    if TOTAL_BATCH_SIZE % world_tokens_per_micro != 0:
        raise ValueError(f'TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} must divide micro tokens={world_tokens_per_micro}')
    grad_accum_steps = TOTAL_BATCH_SIZE // world_tokens_per_micro
    if world_tokens_per_micro != 131072 or grad_accum_steps != 2 or MAX_TRAIN_SECONDS <= 0:
        raise ValueError('the fixed batch ramp requires micro131072 / total262144 and a positive time budget')
    ramp_accum = 1
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
    table_update = DetachedTableUpdate(orig_model, compile_backend=compile_backend) if config.d1_detached else None
    assert dist.is_initialized() and dist.get_world_size() == 4
    if FLAT_WEIGHTS:
        # Installed after setup_optimizer and before the first forward (the compiled core traces the flat views).
        flat_weights = FlatWeights(orig_model, optimizer, max_norm=GRAD_CLIP, compute_dtype=COMPUTE_DTYPE)
        owned_optimizer = None
        print0(f'flat weights: {len(flat_weights.parameters)} parameters in one fp32 buffer of {flat_weights.flat.numel():,} elements (stride {flat_weights.stride:,}), {len(flat_weights.linear_slots)} Linear bf16 views, {len(flat_weights.cat_plan)} cat pieces, reduce_mode={flat_weights.reduce_mode}')
    else:
        flat_weights = None
        owned_optimizer = OwnedOptimizer(optimizer)
    cached_linears = [m for m in model.modules() if isinstance(m, Linear)]
    table_epoch = 1
    # Persistent table buffer is absent from parameters, main AVG, clip and ownership.
    clip_main_params = list(orig_model.parameters())
    loop_started = time.monotonic()
    budget_started = loop_started
    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0
    recent_step_times: list[float] = []
    startup_allowance_est = 0.0
    grad_accum_now = grad_accum_steps
    while step < NUM_STEPS:
        elapsed = time.monotonic() - budget_started
        grad_accum_now = grad_accum_steps
        synchronize(device)
        t0 = time.monotonic()
        ramp_progress = startup_adjusted_schedule_elapsed(step, elapsed, budget_started) / MAX_TRAIN_SECONDS
        ramp_accum = _synchronized_batch_ramp_accum(ramp_progress, ramp_accum, device)
        grad_accum_now = ramp_accum
        # Fixed-shape runtime backward seed, not Python-scalar graph specialization.
        ramp_backward_seed = torch.tensor(LOSS_SCALE / grad_accum_now, dtype=torch.float32, device=device)
        if flat_weights is not None:
            if step == 0:
                flat_weights.refresh()          # one bf16 cast; every Linear cache is a view of it
            flat_weights.zero_grad()            # one launch: the flat fp32 gradient buffer
        else:
            if step == 0:
                for m in cached_linears:
                    m.refresh_weight_cache(COMPUTE_DTYPE)
            optimizer.zero_grad(set_to_none=True)
        last_loss = None
        mtp_progress = startup_adjusted_schedule_elapsed(step, elapsed, budget_started) / MAX_TRAIN_SECONDS
        mtp_weights = torch.tensor([1.0, mtp_auxiliary_weight(mtp_progress)], dtype=torch.float32, device=device)
        for micro_index in range(grad_accum_now):
            table_context = {} if table_update is not None else None
            if table_context is None:
                loss = model(x, y, mtp_weights=mtp_weights)
            else:
                loss = model(x, y, mtp_weights=mtp_weights, table_context=table_context)
            last_loss = loss.detach()
            loss.backward(gradient=ramp_backward_seed)
            if table_update is not None:
                table_update.accumulate(table_context, accumulation_steps=grad_accum_now)
            x, y, loader_state = fetch()
        if flat_weights is None:
            clip_and_sync_gradients(model, clip_main_params, max_norm=GRAD_CLIP)
        schedule_elapsed = startup_adjusted_schedule_elapsed(step, elapsed, budget_started)
        lrm = lr_multiplier(step, elapsed=schedule_elapsed, max_train_seconds=MAX_TRAIN_SECONDS)
        lrm_flat = min(1.0, (step + 1) / max(1, WARMUP_STEPS))
        wd_progress = min(1.0, schedule_elapsed / max(1e-06, MAX_TRAIN_SECONDS))
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * 1.0 * (lrm_flat if group.get('no_lr_decay') else lrm)
            if group['kind'] == 'muon':
                group['momentum'] = muon_momentum(step, wd_progress)
                group['weight_decay'] = WEIGHT_DECAY * (1.0 - wd_progress)
                group['beta2'] = 0.9
        if flat_weights is not None:
            flat_weights.step(update_adamw=True)   # reduce-scatter + clip + Muon/AdamW + all-gather + bf16 refresh
        else:
            owned_optimizer.step(update_adamw=True)
        if table_update is not None:
            # Local estimates cannot independently select a replicated-table collective.
            remaining = MAX_TRAIN_SECONDS - (time.monotonic() - budget_started)
            estimated_step = max(recent_step_times[-20:], default=0.0)
            local_force = step + 1 >= NUM_STEPS or (step >= TRAIN_STARTUP_STEPS_EXCLUDED and remaining <= CHECKPOINT_RESERVE_SECONDS + 2 * estimated_step)
            completed_tokens = total_train_tokens + grad_accum_now * world_tokens_per_micro
            local_freeze = table_update.freeze_tokens > 0 and completed_tokens >= table_update.freeze_tokens and not table_update.frozen
            next_epoch, force_table, abort_table, freeze_table = synchronized_table_control(int(loader_state['epoch']), table_epoch, local_force, remaining <= 0.0, device, local_freeze)
            table_epoch = next_epoch
            table_update.finish_step(loader_epoch=table_epoch, force=force_table, abort=abort_table, freeze=freeze_table)
            if freeze_table:
                print0(f'D2_TABLE_FROZEN step={step} global_tokens={completed_tokens} threshold={table_update.freeze_tokens} updates={table_update.update_count}; gate/projections continue')
        if flat_weights is None:
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
            recent_step_times.clear()
        elapsed_after_step = time.monotonic() - budget_started
        reached_min_steps = step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED
        step_cap_hit = step + 1 >= NUM_STEPS
        time_budget_hit = reached_min_steps and (elapsed_after_step >= MAX_TRAIN_SECONDS or elapsed_after_step + max(recent_step_times[-20:], default=0.0) + CHECKPOINT_RESERVE_SECONDS >= MAX_TRAIN_SECONDS)
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
            remaining_steps = NUM_STEPS - step - 1
            remaining_time = max(0.0, MAX_TRAIN_SECONDS - (time.monotonic() - budget_started))
            print(f"step {step:05d} | loss {debiased:.4f} | lrm {lrm:.3f} | dt {dt:.2f}s | tok/s {tok_per_sec:,.0f} | remaining_steps {remaining_steps} | remaining_time {remaining_time:.0f}s | epoch {loader_state['epoch']}", flush=True)
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1
        if int(control[1].item()) != 0:
            break
    if table_update is not None:
        if table_update.pending_tokens:
            raise RuntimeError('unfinished every2 group; refuse dropping gradients or post-loop/post-budget update')
        print0(f'D1_TABLE_COMPLETE updates={table_update.update_count} pending_tokens=0; all updates inside recorded steps')
        if table_update.d2_two_site:
            rank_check_started = time.monotonic()
            table_max_abs_diff = d2_table_rank_max_abs_diff(table_update.table)
            rank_check_seconds = time.monotonic() - rank_check_started
            print0(f'D2_TABLE_RANK_CONSISTENCY world_size={world_size} max_abs_diff_vs_rank0={table_max_abs_diff:.9g} method=cloned_rank0_broadcast_host_fp32_max_scalar_allreduce diagnostic_seconds={rank_check_seconds:.6f}')
            if not math.isfinite(table_max_abs_diff):
                raise RuntimeError('nonfinite end-run table rank-consistency diagnostic')
    checkpoint_path = Path(args.out_dir) / 'final.pt'
    budget_elapsed_before_save = time.monotonic() - budget_started
    if rank == 0:
        save_checkpoint(checkpoint_path, orig_model, {'step': step, 'total_train_tokens': total_train_tokens, 'budget_elapsed_before_save': budget_elapsed_before_save, 'world_size': world_size, 'compiled': True})
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    budget_elapsed_after_save = time.monotonic() - budget_started
    if rank == 0:
        print('---')
        print(f'checkpoint_path:    {checkpoint_path}')
        print(f'training_seconds:  {budget_elapsed_before_save:.1f}')
        print(f'budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}')
        print(f'target_steps:      {NUM_STEPS}')
        print(f'max_train_seconds: {MAX_TRAIN_SECONDS}')
        print(f'startup_allowance_est: {startup_allowance_est:.1f}')
        print(f'charged_seconds_est:   {budget_elapsed_after_save:.1f}')
        print(f'over_budget:       {budget_elapsed_after_save > TRAIN_TIME_BUDGET_SECONDS}')
        print(f'total_seconds:     {time.monotonic() - loop_started:.1f}')
        print(f'compiled:          {True}')
        print(f'num_steps:         {step}')
        print(f'total_tokens_M:    {total_train_tokens / 1000000.0:.1f}')
        print(f'world_size:        {world_size}')
        print(f"lnc:               {os.environ.get('NEURON_LOGICAL_NC_CONFIG', '')}")
        print(f'seq_len:           {SEQ_LEN}')
        print(f'num_params_M:      {num_params / 1000000.0:.1f}')
    cleanup_runtime()

def make_plan(groups, world_size):
    muon_ms = {(1024, 1024): 12.54413 / 24, (1024, 4096): 9.76316 / 6, (4096, 1024): 8.98251 / 6, (8, 32): 0.23355 / 3}
    entries = []
    parameters = []
    seen = set()
    for group_index, group in enumerate(groups):
        if group['kind'] not in ('adamw', 'muon'):
            raise ValueError('optimizer ownership supports AdamW and Muon only')
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
            raise ValueError('install ownership before the first update; use its checkpoint loader to resume')
        self.optimizer = optimizer
        self.pad_singleton_muon = bool(True)
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
        self.raw_state_dict = optimizer.state_dict
        self.raw_load_state_dict = optimizer.load_state_dict
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

    def state_dict(self):
        state = self.raw_state_dict()
        state['state'] = {key: {name: value for name, value in fields.items() if name not in ('stacked_params', 'stacked_grads')} for key, fields in state['state'].items()}
        return dict(schema='fresh-start-owned-optimizer-v1', rank=self.rank, world_size=self.world_size, entries=self.entries, pad_singleton_muon=self.pad_singleton_muon, adamw_math_mode=getattr(self.optimizer, '_fresh_start_adamw_math_mode', None), optimizer=state)

    def load_state_dict(self, payload):
        if payload.get('schema') != 'fresh-start-owned-optimizer-v1' or payload.get('rank') != self.rank or payload.get('world_size') != self.world_size or (payload.get('pad_singleton_muon') != self.pad_singleton_muon) or (payload.get('adamw_math_mode') != getattr(self.optimizer, '_fresh_start_adamw_math_mode', None)) or (payload.get('entries') != self.entries):
            raise ValueError('optimizer checkpoint ownership/schema mismatch')
        self.raw_load_state_dict(payload['optimizer'])
        self.optimizer._scalar_cache.clear()

_FLAT_REDUCE_SCATTER = getattr(funcol, 'reduce_scatter_single', None) or funcol.reduce_scatter_tensor

def _muon_body_for(key, p0):
    """The per-shape Muon body _muon_step uses (the same _MUON_COMPILED entry, so the same NEFF on the device)."""
    fn = _MUON_COMPILED.get(key)
    if fn is None:
        fn = torch.compile(_muon_math, backend='neuron', dynamic=False) if p0.device.type == 'neuron' else _muon_math
        _MUON_COMPILED[key] = fn
        print0(f'muon step: COMPILED body for {tuple(key[3])} (tall={key[0]} ns_steps={key[1]} red_dim={key[2]}) [specialisations so far: {len(_MUON_COMPILED)}]')
    return fn

# ---------------------------------------------------------------------------------------------------------------
# FLATCACHE (FLAT_WEIGHTS=True): see FLATCACHE.md. Three flat buffers replace every per-parameter launch of the
# weight/gradient path: `flat` (fp32 master, owner-major exactly like OwnedOptimizer.received; every dense
# Parameter is a view of it), `flat_grad` (fp32 gradient accumulator, same layout; non-Linear parameters have
# p.grad pre-bound to their slot so AccumulateGrad adds in place), `flat_bf16` (ONE cast per step; every
# Linear._cached_weight is a view of it; its backward is ONE cat of the bf16 weight gradients and ONE fp32
# accumulate). The tail is eager: one reduce-scatter (AVG), the global-norm clip from the owned shard, Muon on
# the contiguous owned stacked views through the same compiled _muon_math bodies, AdamW with the same
# torch._foreach_* sequence on the owned views, one all_gather_into_tensor into `flat`, then the bf16 cast.
# ---------------------------------------------------------------------------------------------------------------


class _FlatCast(torch.autograd.Function):
    """flat fp32 -> flat bf16 once per step; returns the per-Linear bf16 views. Backward: one cat + one fp32 add."""

    @staticmethod
    def forward(ctx, flat, dummy, fw):
        ctx.fw = fw
        ctx.set_materialize_grads(False)
        flat_bf16 = flat.to(fw.compute_dtype)  # ONE cast per step (today: 40 refresh_weight_cache casts)
        fw.flat_bf16 = flat_bf16
        return tuple(flat_bf16.narrow(0, slot, numel).view(shape) for slot, numel, shape in fw.linear_slots)

    @staticmethod
    def backward(ctx, *grads):
        ctx.fw._accumulate_linear_grads(grads)
        return (None, None, None)


class FlatWeights:
    """Install after model.to(device) and setup_optimizer(), before compile_core and before the first
    refresh; replaces OwnedOptimizer, refresh_weight_cache, optimizer.zero_grad and clip_and_sync_gradients.
    """

    REDUCE_MODES = ('reduce_scatter_avg', 'all_reduce_avg')

    def __init__(self, model, optimizer, *, max_norm: float, compute_dtype: torch.dtype, linear_cls=None, muon_body_for=None, make_plan=make_plan, pad_singleton_muon: bool = True, reduce_mode: str = 'reduce_scatter_avg', group=None):
        linear_cls = Linear if linear_cls is None else linear_cls
        muon_body_for = _muon_body_for if muon_body_for is None else muon_body_for
        if not dist.is_initialized():
            raise ValueError('flat weights require an initialized process group')
        if optimizer.state:
            raise ValueError('install flat weights before the first update')
        if reduce_mode not in self.REDUCE_MODES:
            raise ValueError(f'reduce_mode must be one of {self.REDUCE_MODES}')
        self.optimizer = optimizer
        self.model = model
        self.max_norm = float(max_norm)
        self.compute_dtype = compute_dtype
        self.muon_body_for = muon_body_for
        self.reduce_mode = reduce_mode
        self.group = group if group is not None else dist.group.WORLD
        try:
            self.group_name = funcol._resolve_group_name(self.group)
        except Exception:
            self.group_name = self.group.group_name
        self.pad_singleton_muon = bool(pad_singleton_muon)
        self.rank, self.world_size = dist.get_rank(self.group), dist.get_world_size(self.group)
        # Same plan, same owners, same offsets, same 128-padded stride as OwnedOptimizer.
        self.entries, self.parameters, self.summary = make_plan(optimizer.param_groups, self.world_size)
        self.parameter_ids = [id(p) for p in self.parameters]
        device = self.parameters[0].device
        if any((p.device != device for p in self.parameters)):
            raise ValueError('all Parameters must share a device')
        self.device = device
        model_params = {id(p) for p in model.parameters()}
        if model_params != set(self.parameter_ids):
            raise ValueError('the optimizer parameter set must be exactly model.parameters() (the clip set of the original path)')
        stride = self.stride = self.summary['shard_elements']
        total = stride * self.world_size
        self.flat = torch.zeros(total, dtype=torch.float32, device=device)
        self.flat_grad = torch.zeros(total, dtype=torch.float32, device=device)
        self.send = torch.zeros(stride, dtype=torch.float32, device=device)
        self.flat_bf16 = None
        self.dummy = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
        # (1) FLAT MASTER: copy every parameter into its slot once, then rebind p.data to the slot view.
        linears = [m for m in model.modules() if isinstance(m, linear_cls)]
        weight_to_linear = {}
        for m in linears:
            if m.bias is not None:
                raise ValueError('flat weights support bias-free Linear only')
            weight_to_linear[id(m.weight)] = m
        self.slots = []                    # (slot, numel) per plan entry
        self.linear_slots = []             # (slot, numel, shape) per Linear, in plan order
        self.linear_modules = []
        self.linear_index_of_entry = {}
        with torch.no_grad():
            for e, p in zip(self.entries, self.parameters):
                slot = e['owner'] * stride + e['offset']
                n = e['elements']
                self.flat.narrow(0, slot, n).view_as(p).copy_(p)
                p.data = self.flat.narrow(0, slot, n).view(p.shape)
                self.slots.append((slot, n))
                m = weight_to_linear.get(id(p))
                if m is not None:
                    self.linear_index_of_entry[e['index']] = len(self.linear_slots)
                    self.linear_slots.append((slot, n, tuple(p.shape)))
                    self.linear_modules.append(m)
                    p.grad = None
                else:
                    # (2) fp32 in-place accumulation for the non-Linear parameters: AccumulateGrad does
                    # `p.grad += new_grad` when p.grad is defined and grad mode is off (the backward engine).
                    p.grad = self.flat_grad.narrow(0, slot, n).view(p.shape)
        if len(self.linear_modules) != len(linears):
            raise ValueError('every Linear weight must be in the optimizer plan')
        # Cat plan for the backward: the whole buffer in slot order as Linear pieces and zero gap pieces.
        order = sorted(range(len(self.linear_slots)), key=lambda i: self.linear_slots[i][0])
        self.cat_plan = []                 # (linear index or -1, length)
        pos = 0
        max_gap = 0
        for i in order:
            slot, n, _shape = self.linear_slots[i]
            if slot < pos:
                raise ValueError('overlapping slots')
            if slot > pos:
                self.cat_plan.append((-1, slot - pos))
                max_gap = max(max_gap, slot - pos)
            self.cat_plan.append((i, n))
            pos = slot + n
        if pos < total:
            self.cat_plan.append((-1, total - pos))
            max_gap = max(max_gap, total - pos)
        max_lin = max((n for _s, n, _sh in self.linear_slots), default=1)
        self.zero_buf = torch.zeros(max(max_gap, max_lin, 1), dtype=compute_dtype, device=device)
        # Owned groups exactly like OwnedOptimizer.groups (with the zero dummy for a singleton Muon group) plus
        # the layout of each owned group inside the owned shard.
        self.groups, self.positions, self.padding = ([], [], {})
        self.muon_layout = {}              # group index -> (base offset in shard, k_real, k, numel, shape)
        self.adamw_layout = {}             # group index -> list of (offset, numel, shape)
        for gi, source in enumerate(optimizer.param_groups):
            owned = [e for e in self.entries if e['group'] == gi and e['owner'] == self.rank]
            positions = [e['position'] for e in owned]
            self.positions.append(positions)
            self.groups.append({**source, 'params': [source['params'][i] for i in positions]})
            if not owned:
                continue
            if source['kind'] == 'muon':
                numel = owned[0]['elements']
                shape = tuple(owned[0]['shape'])
                base = owned[0]['offset']
                for j, e in enumerate(owned):
                    if e['offset'] != base + j * numel or tuple(e['shape']) != shape:
                        raise ValueError('make_plan gave a Muon group non-contiguous owned slots; the stacked view needs contiguity')
                k_real = len(owned)
                k = k_real
                if self.pad_singleton_muon and k_real == 1:
                    dummy = torch.nn.Parameter(torch.zeros_like(self.groups[-1]['params'][0]))
                    self.groups[-1]['params'].append(dummy)
                    self.padding[gi] = dummy
                    k = 2
                self.muon_layout[gi] = (base, k_real, k, numel, shape)
            else:
                self.adamw_layout[gi] = [(e['offset'], e['elements'], tuple(e['shape'])) for e in owned]
        self.calls = 0
        self.backward_calls = 0
        self.raw_state_dict = optimizer.state_dict
        self.raw_load_state_dict = optimizer.load_state_dict

    # ------------------------------------------------------------------ per-micro-step pieces

    def zero_grad(self) -> None:
        """One launch instead of optimizer.zero_grad over every parameter."""
        self.flat_grad.zero_()

    def refresh(self) -> None:
        """One bf16 cast of the flat master; every Linear._cached_weight becomes a view of it (no launches)."""
        with torch.enable_grad():
            views = _FlatCast.apply(self.flat, self.dummy, self)
        for m, v in zip(self.linear_modules, views):
            m._cached_weight = v

    def _accumulate_linear_grads(self, grads) -> None:
        """Backward of _FlatCast: ONE cat of the 40 bf16 weight gradients (zero views fill the non-Linear and
        padding slots) and ONE fp32 accumulate. Adding the exact fp32 value of each bf16 gradient to the fp32
        slot is today's ToCopyBackward cast followed by the fp32 AccumulateGrad add; the zero pieces add
        exactly 0 to the non-Linear slots, whose own fp32 gradients arrive through their pre-bound .grad views.
        """
        zero = self.zero_buf
        pieces = []
        for i, length in self.cat_plan:
            g = grads[i] if i >= 0 else None
            pieces.append(zero.narrow(0, 0, length) if g is None else g.reshape(-1))
        flat_bf16_grad = torch.cat(pieces)
        self.flat_grad.add_(flat_bf16_grad)   # fp32 += bf16 (promoted to fp32 inside the kernel)
        self.backward_calls += 1

    # ------------------------------------------------------------------ the tail

    def _reduce_scatter(self) -> torch.Tensor:
        if self.reduce_mode == 'reduce_scatter_avg':
            shard = _FLAT_REDUCE_SCATTER(self.flat_grad, 'avg', 0, self.group_name)
            if isinstance(shard, funcol.AsyncCollectiveTensor):
                shard = shard.wait()
            return shard
        dist.all_reduce(self.flat_grad, op=dist.ReduceOp.AVG, group=self.group)
        return self.flat_grad.narrow(0, self.rank * self.stride, self.stride)

    def _clip_(self, shard: torch.Tensor) -> None:
        """clip_grad_norm_ over all dense gradients, from the owned shard: local sum of squares, one scalar
        all_reduce, coefficient max_norm / (norm + 1e-6) clamped to 1, one multiply. No .item(), no branch.
        """
        if self.max_norm <= 0:
            return
        sq = shard.square().sum()
        dist.all_reduce(sq, op=dist.ReduceOp.SUM, group=self.group)
        total_norm = sq.sqrt()
        clip_coef = self.max_norm / (total_norm + 1e-06)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        shard.mul_(clip_coef_clamped)

    @torch.no_grad()
    def _muon_group(self, gi: int, owned: dict, shard: torch.Tensor) -> None:
        """MuonAdamW._muon_step on the owned stacked views: the stacked gradient is a view of the reduced
        shard and the stacked weights a view of `send` (the owned master shard); the unchanged _muon_math body
        (the same compiled specialisation key) returns the new stack, which is copied into `send` once.
        """
        opt = self.optimizer
        params = owned['params']
        p0 = params[0]
        state = opt.state[p0]
        base, k_real, k, numel, shape = self.muon_layout[gi]
        transposed = owned.get('transposed', False)
        lshape = (shape[-1], shape[-2]) if transposed else shape
        red_dim = -1 if lshape[-2] >= lshape[-1] else -2
        if transposed:
            red_dim = -2 if red_dim == -1 else -1
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if 'second_momentum_buffer' not in state:
            state_shape = (len(params), shape[-2], 1) if red_dim == -1 else (len(params), 1, shape[-1])
            state['second_momentum_buffer'] = torch.zeros(state_shape, dtype=torch.float32, device=p0.device)
        w_view = self.send.narrow(0, base, k_real * numel).view(k_real, *shape)
        g = shard.narrow(0, base, k_real * numel).view(k_real, *shape)
        w = w_view
        if k > k_real:
            # pad_singleton_muon: the original appends a zero dummy parameter with a zero gradient; every op of
            # _muon_math is per-row, so the real row is unchanged and the compiled key (k=2) is the same.
            pad = torch.zeros(k - k_real, *shape, dtype=p0.dtype, device=p0.device)
            w = torch.cat((w, pad), 0)
            g = torch.cat((g, pad), 0)
        momentum_buffer = state['momentum_buffer']
        second = state['second_momentum_buffer']
        ratio = max(1.0, lshape[-2] / lshape[-1]) ** 0.5
        ns_steps = min(owned['ns_steps'], 5)
        key = (g.shape[-2] > g.shape[-1], ns_steps, red_dim, tuple(g.shape))
        fn = self.muon_body_for(key, p0)
        lr_t, lrwd_t, mom_t, one_mom_t, one_b2_t = opt._muon_compile_scalars(owned, p0, ratio)
        one_beta2 = 1 - owned['beta2']
        w_new, momentum_buffer, second = fn(w, g, momentum_buffer, second, mom_t, one_mom_t, lr_t, lrwd_t, key[0], ns_steps, red_dim, g.size(red_dim), one_beta2, self.compute_dtype == torch.bfloat16, False)
        state['momentum_buffer'] = momentum_buffer
        state['second_momentum_buffer'] = second
        w_view.copy_(w_new[:k_real] if k > k_real else w_new)

    @torch.no_grad()
    def _adamw_group(self, gi: int, owned: dict, shard: torch.Tensor) -> None:
        """MuonAdamW._adamw_step_foreach with the parameter/gradient lists replaced by views of `send` and of
        the reduced shard; the same expression sequence, state keyed by the real Parameters.
        """
        opt = self.optimizer
        layout = self.adamw_layout[gi]
        params = [self.send.narrow(0, off, n).view(shape) for off, n, shape in layout]
        grads = [shard.narrow(0, off, n).view(shape) for off, n, shape in layout]
        beta1, beta2 = owned['betas']
        exp_avgs, exp_avg_sqs, bias1, bias2 = [], [], [], []
        for p in owned['params']:
            state = opt.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])
            bias1.append(1 - beta1 ** state['step'])
            bias2.append(1 - beta2 ** state['step'])
        lr = opt._device_scalar(owned, 'lr', owned['lr'], params[0])
        wd = owned['weight_decay']
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), 1 - beta2)
        denom = torch._foreach_div(exp_avg_sqs, bias2)
        torch._foreach_sqrt_(denom)
        torch._foreach_add_(denom, owned['eps'])
        update = torch._foreach_div(exp_avgs, bias1)
        torch._foreach_div_(update, denom)
        if wd != 0:
            torch._foreach_mul_(params, 1 - lr * wd)
        torch._foreach_mul_(update, lr)
        torch._foreach_sub_(params, update)

    @torch.no_grad()
    def step(self, update_adamw: bool = True) -> None:
        """clip_and_sync_gradients + OwnedOptimizer.step + refresh_weight_cache, flat."""
        opt = self.optimizer
        shard = self._reduce_scatter()
        self._clip_(shard)
        self.send.copy_(self.flat.narrow(0, self.rank * self.stride, self.stride))
        for gi, (source, owned) in enumerate(zip(opt.param_groups, self.groups)):
            owned.update({key: value for key, value in source.items() if key != 'params'})
            if not owned['params']:
                continue
            if owned['kind'] == 'adamw':
                if update_adamw:
                    self._adamw_group(gi, owned, shard)
            else:
                self._muon_group(gi, owned, shard)
        dist.all_gather_into_tensor(self.flat, self.send, group=self.group)   # the parameters are views: done
        self.calls += 1
        self.refresh()

    # ------------------------------------------------------------------ checkpointing (OwnedOptimizer schema)

    def state_dict(self):
        state = self.raw_state_dict()
        state['state'] = {key: {name: value for name, value in fields.items() if name not in ('stacked_params', 'stacked_grads')} for key, fields in state['state'].items()}
        return dict(schema='fresh-start-owned-optimizer-v1', rank=self.rank, world_size=self.world_size, entries=self.entries, pad_singleton_muon=self.pad_singleton_muon, adamw_math_mode=getattr(self.optimizer, '_fresh_start_adamw_math_mode', None), optimizer=state)

    def load_state_dict(self, payload):
        if payload.get('schema') != 'fresh-start-owned-optimizer-v1' or payload.get('rank') != self.rank or payload.get('world_size') != self.world_size or (payload.get('pad_singleton_muon') != self.pad_singleton_muon) or (payload.get('adamw_math_mode') != getattr(self.optimizer, '_fresh_start_adamw_math_mode', None)) or (payload.get('entries') != self.entries):
            raise ValueError('optimizer checkpoint ownership/schema mismatch')
        self.raw_load_state_dict(payload['optimizer'])
        self.optimizer._scalar_cache.clear()
        # Parameters restored by the caller into the views are already in `flat`; nothing else to sync.

def selected_lanes(rows):
    if rows <= 0 or rows % 128:
        raise ValueError('the NKI CE adapter requires positive multiples of128 rows')
    return min(2 if rows >= 512 and rows % 256 == 0 else 1, _NKI_LANES)
if __name__ == '__main__':
    main()
