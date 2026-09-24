"""Trainium Frontier submission: dense 6x1024, seed42, T1024, pack32.

Frozen from source 1da6c0714c0755fd3b078e972fbb0aefbb43bf6d.
Verified source checkpoint: public val_bpb=0.9928513116311253 over
20,971,520 tokens; this is not a private leaderboard result.
All required kernels and training adapters are included in this file.
The organizer supplies the unchanged prepare.py and fixed assets.
Use the accompanying launch-command.txt; train from fresh initialization."""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
_PROCESS_STARTED = time.monotonic()
if __name__ == '__main__':
    os.environ.setdefault('NEURON_CC_FLAGS', '--optlevel=1 --auto-cast matmult --auto-cast-type bf16')
USE_TRN2_COMPILER_FLAGS = False
os.environ.setdefault('NEURON_COMPILE_CACHE_DIR', '/tmp/neuron_cache')
if USE_TRN2_COMPILER_FLAGS:
    os.environ.setdefault('NEURON_FUSE_SOFTMAX', '1')
    os.environ.setdefault('NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS', '3')
    os.environ.setdefault('NEURON_CC_FLAGS', f"--model-type=transformer --distribution-strategy=llm-training --lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG', '1')}")
else:
    os.environ.setdefault('NEURON_CC_FLAGS', '--optlevel=1')
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from prepare import EVAL_TIMEOUT_SECONDS, TOKENIZER_VOCAB_SIZE, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS, TRAIN_STARTUP_STEPS_EXCLUDED, TRAIN_TIME_BUDGET_SECONDS, Tokenizer, _document_batches, _row_groups_for_split, causality_check, ensure_tokenizer, evaluate_bpb, get_dist_info, get_token_bytes, make_dataloader, print0, record_step_boundary

def make_dataloader_requeue(tokenizer: Tokenizer, batch_size: int, seq_len: int, split: str, device: torch.device | str, tokenizer_threads: int=4, tokenizer_batch_size: int=128, shard_ids=None, _document_batches_fn=_document_batches):
    assert batch_size > 0 and seq_len > 1
    ddp, rank, _local_rank, world_size = get_dist_info()
    batches = _document_batches_fn(split, rank, world_size, tokenizer_batch_size, shard_ids=shard_ids)
    bos = tokenizer.get_bos_token_id()
    row_capacity = seq_len + 1
    doc_buffer: list[list[int]] = []
    buffer_min = 512
    epoch = 0
    dev = torch.device(device)
    pin_memory = dev.type == 'cuda'
    row_buffer = torch.empty((batch_size, row_capacity), dtype=torch.long)
    cpu_inputs = torch.empty((batch_size, seq_len), dtype=torch.int32, pin_memory=pin_memory)
    cpu_targets = torch.empty((batch_size, seq_len), dtype=torch.long, pin_memory=pin_memory)
    inputs = torch.empty((batch_size, seq_len), dtype=torch.int32, device=dev)
    targets = torch.empty((batch_size, seq_len), dtype=torch.long, device=dev)

    def refill() -> None:
        nonlocal epoch
        docs, epoch = next(batches)
        token_lists = tokenizer.encode(docs, prepend=bos, num_threads=tokenizer_threads)
        doc_buffer.extend(token_lists)
    while True:
        for row_idx in range(batch_size):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_min:
                    refill()
                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    n = len(doc)
                    if n <= remaining and n > best_len:
                        best_idx, best_len = (i, n)
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    n = len(doc)
                    row_buffer[row_idx, pos:pos + n] = torch.tensor(doc, dtype=torch.long)
                    pos += n
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
        cpu_inputs.copy_(row_buffer[:, :-1].to(dtype=torch.int32))
        cpu_targets.copy_(row_buffer[:, 1:])
        inputs.copy_(cpu_inputs, non_blocking=pin_memory)
        targets.copy_(cpu_targets, non_blocking=pin_memory)
        yield (inputs, targets, {'epoch': epoch})

def _document_batches_epoch_shuffle(split, start, step, tokenizer_batch_size, shard_ids=None):
    """prepare._document_batches, but the row-group order is reshuffled from epoch 2 onward.

    prepare's generator walks `groups` in the same order every epoch, so a run that crosses the
    epoch boundary replays the corpus's OPENING documents -- and it does so during the lowest-LR
    cooldown steps, which shape the final weights most. Measured cost of that at the locked 1800s
    config: see the 2026-09-13 "data wall" section of research/trn2-experiment-results.md.

    Epoch 1 is byte-for-byte prepare's order (no shuffle, no RNG draw), so any run that never
    reaches epoch 2 -- which includes the locked submission config at ~2140 steps -- is bit-identical
    to the baseline. The RNG is a local random.Random, never the global one, so it cannot perturb
    weight init on any rank. Every rank shuffles its own disjoint group list with the same seed
    stream; ranks never share groups, so no coordination is needed.
    """
    groups = _row_groups_for_split(split, start, step, shard_ids=shard_ids)
    epoch = 0
    while True:
        epoch += 1
        order = list(groups)
        if epoch > 1:
            random.Random(EPOCH_SHUFFLE_SEED + epoch).shuffle(order)
        for path, rg_idx in order:
            rg = pq.ParquetFile(path).read_row_group(rg_idx)
            rows = rg.column('text').to_pylist()
            for i in range(0, len(rows), tokenizer_batch_size):
                yield (rows[i:i + tokenizer_batch_size], epoch)
SEQ_LEN = 2048
DEPTH = 6
ASPECT_RATIO = 96
HEAD_DIM = 128
N_EMBD = 0
MLP_RATIO = 4
ATTN_SCALE = 0.0
N_KV_HEADS = 0
DEVICE_BATCH_SIZE = 1
TOTAL_BATCH_SIZE = 131072
NUM_STEPS = 1000000
STEP_CAP_GUARD_MIN_PROGRESS = 0.5
STEP_CAP_GUARD_MAX_PROGRESS = 0.98
BATCH_RAMP = False
BATCH_RAMP_GRAD_ACCUM = (2, 4, 8)
BATCH_RAMP_BOUNDS = (0.35, 0.75)
BATCH_RAMP_LR_EXP = 0.5
USE_WEIGHT_EMA = False
WEIGHT_EMA_DECAYS = (0.995,)
WEIGHT_EMA_START_FRAC = 0.75
EVAL_PREFER_EMA = os.environ.get('NEURON_COMPETITION_R1_EVAL_PREFER_EMA', '1') != '0'
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
CHECKPOINT_RESERVE_SECONDS = 15.0
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0
LOGIT_SOFTCAP = 15.0
LOSS_BYTE_WEIGHT = False
BYTE_LOSS_W: torch.Tensor | None = None
EMBEDDING_LR = 0.3
UNEMBEDDING_LR = 0.006
MATRIX_LR = 0.02
SCALAR_LR = 0.2
SCALAR_BETA1 = 0.9
SCALAR_BETA2 = 0.95
USE_CAUTIOUS_WD = False
WEIGHT_DECAY = 0.2
VE_WD = 0.0
EMBED_ADAMW_BETA2 = 0.95
WARMUP_STEPS = 0
GRAD_CLIP = 1.0
CLIP_AFTER_REDUCE = False
LOSS_SCALE_DYNAMIC = False
LOSS_SCALE_GROWTH_INTERVAL = 500
LOSS_SCALE_MAX = 8192.0
loss_scale_clean = 0
loss_scale_skips = 0
LOSS_SCALE = 1.0
WARMDOWN_RATIO = 0.75
FINAL_LR_FRAC = 0.05
USE_TENSOR_LR_SCALARS = True
LRM_QUANTUM = 0
USE_FOREACH_OPTIM = True
WD_DECAY_TO_ZERO = True
MUON_MOMENTUM_WARMUP_STEPS = 300
MUON_MOMENTUM_COOLDOWN_FRAC = 0.045
MUON_MOMENTUM_FINAL = 0.85
MUON_NS_STEPS = 5
DEMON_BETA1 = False
DEMON_BETA1_FINAL = 0.55
DEMON_BETA1_REF = 0.8
DEMON_BETA1_START_FRAC = -1.0
MUON_BETA2 = 0.9
MUON_BETA2_FINAL = 0.0
MUON_DEPTH_LR_BOTTOM = 1.0
MUON_DEPTH_LR_TOP = 1.0
MUON_DEPTH_MOM_BOTTOM = 0.0
MUON_DEPTH_MOM_TOP = 0.0
MUON_DEPTH_MOM_REF = 0.95
ADAM_EVERY_N = 1
ADAM_EVERY_N_LR_COMP = False
INIT_SCALE = 0.5
ATTN_CPROJ_INIT_SCALE = 0.1
USE_BYTE_WTE_INIT = False
BYTE_WTE_INIT_MIX = 1.0
BYTE_WTE_INIT_NGRAM = 3
BYTE_WTE_INIT_STD = 0.8
PACK_FACTOR = 4
USE_EPOCH_SHUFFLE = False
EPOCH_SHUFFLE_SEED = 1234
USE_VE = True
VE_GATE_CHANNELS = 32

def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Alternating layers get a value embedding; the last layer always does."""
    return layer_idx % 2 == (n_layer - 1) % 2
USE_NGRAM_VE = True
NGRAM_VE_LAYERS = '1,-1'
NGRAM_VE_TABLE_MULT = 128
NGRAM_VE_TRIGRAM = True
NGRAM_VE_DIM = 128
NGRAM_VE_LR = 0.0
NGRAM_VE_BETA2 = 0.999
NGRAM_VE_FLAT_LR = True
NGRAM_VE_OPT = 'rmsprop'
NGRAM_VE_BF16 = True
NGRAM_RMS_FP32 = False
USE_NKI_NGRAM_RMS = False
NGRAM_VE_CLIP_EXCLUDE = True
NGRAM_VE_INDEX_GATHER = True
USE_ENGRAM = False
ENGRAM_SITE = '2'
ENGRAM_ORDERS = '2,3'
ENGRAM_HEADS = 1
ENGRAM_TABLE_MULT = 32
ENGRAM_MEM_DIM = 128
ENGRAM_KEY_DIM = 0
ENGRAM_CONV_KERNEL = 4
ENGRAM_SHARE_NGRAM_TABLES = False
ENGRAM_BF16 = True
USE_OUT_BIGRAM = False
OUT_BIGRAM_BF16 = True
OUT_BIGRAM_GATE_BIAS = -2.0
OUT_BIGRAM_LR = 0.0
RELU2_TAU = 0.0
USE_MLP_SANDWICH_NORM = False
USE_HEAD_GATE = False
HEAD_GATE_NORM = True
HEAD_GATE_CHANNELS = 32
HEAD_GATE_MUL = 2.0
HEAD_GATE_BIAS = 0.0
USE_OUT_POOL = False
OUT_POOL_LAYERS = 3
USE_QK_SHIFT = True
QK_SHIFT_BETA = 0.0
QK_SHIFT_FREEZE = True
USE_BLOCK_NUDGE = False
BLOCK_NUDGE_SITE = 'qk'
USE_PARALLEL_BLOCK = False
PARALLEL_BLOCK_COALESCE = 'none'
USE_QKV_NORM_CSE = False
USE_X0_GATE = False

def head_gate_slice(n_embd: int, n_layer: int) -> tuple[int, int]:
    """(start, stop) residual channels the T3.6 head gate reads.

    Placed after every value-embedding gate's slice so no two gates share channels. The VE gates
    claim 0:32 (unigram), 32:64 (bigram) and 64:96 (trigram) whenever those tables exist on ANY
    layer, and the offset is computed layer-INDEPENDENTLY on purpose: every layer's head gate must
    read the same slice, or the A/B would also be testing a per-layer channel permutation.
    """
    claimed = 0
    if USE_VE:
        claimed = VE_GATE_CHANNELS
        if any((has_ngram_bigram(i, n_layer) for i in range(n_layer))):
            claimed = 2 * VE_GATE_CHANNELS
        if any((has_ngram_trigram(i, n_layer) for i in range(n_layer))):
            claimed = 3 * VE_GATE_CHANNELS
    start = claimed
    stop = start + HEAD_GATE_CHANNELS
    assert stop <= n_embd, (start, stop, n_embd)
    return (start, stop)

def causal_shift_1(x: torch.Tensor) -> torch.Tensor:
    """x shifted one position later along dim 1, zero-padded at t=0.

    Result[:, t] == x[:, t-1] (and 0 at t == 0), so anything built from it reads strictly into the
    past. Written as a slice + cat rather than torch.roll because roll WRAPS the last position
    into position 0, which would be an exact causality violation.
    """
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)

def _next_prime(n: int) -> int:
    """Smallest prime >= n. Used once at import to build the hash multipliers."""

    def is_prime(m: int) -> bool:
        if m < 2:
            return False
        if m % 2 == 0:
            return m == 2
        f = 3
        while f * f <= m:
            if m % f == 0:
                return False
            f += 2
        return True
    while not is_prime(n):
        n += 1
    return n
NGRAM_HASH_PRIMES = tuple((_next_prime(8192 + 8000 * k) for k in range(30)))

def _largest_prime_below(n: int) -> int:
    """Largest prime <= n. Used for the hash modulus (see GPT.ngram_mod)."""
    m = n
    while _next_prime(m) != m:
        m -= 1
    return m

def _rng_neutral(build):
    """Construct a module WITHOUT consuming draws from the global CPU generator.

    Only correct for modules that init_weights() overwrites unconditionally -- here the
    T3.1 n-gram gates, which it zeroes. The point is that USE_NGRAM_VE must not perturb
    every other tensor's initial values: nn.Linear.reset_parameters() draws from the
    default generator at CONSTRUCTION time, so adding five tiny gates would shift the RNG
    stream for everything init_weights() draws afterwards and make the A/B a comparison of
    two different random inits plus the feature, instead of just the feature. The model is
    built on CPU and only then .to(device)'d (see main()), so the CPU generator is the only
    one involved and save/restore is exact.
    """
    state = torch.random.get_rng_state()
    try:
        return build()
    finally:
        torch.random.set_rng_state(state)

def ngram_bigram_primes(layer_idx: int) -> tuple[int, int]:
    """(p_prev, p_cur) for this layer's bigram hash. Different pair per layer so
    collisions decorrelate across layers (Recursive's stated reason)."""
    i = 2 * layer_idx % (len(NGRAM_HASH_PRIMES) - 1)
    return (NGRAM_HASH_PRIMES[i], NGRAM_HASH_PRIMES[i + 1])

def ngram_trigram_primes(layer_idx: int) -> tuple[int, int, int]:
    """(p_prev2, p_prev, p_cur) for this layer's trigram hash. Offset into the prime
    table so it never coincides with any layer's bigram pair."""
    i = 12 + 3 * layer_idx % (len(NGRAM_HASH_PRIMES) - 14)
    return (NGRAM_HASH_PRIMES[i], NGRAM_HASH_PRIMES[i + 1], NGRAM_HASH_PRIMES[i + 2])

def ngram_layer_set(n_layer: int) -> frozenset[int]:
    """NGRAM_VE_LAYERS parsed to absolute indices. Empty set means "no restriction".

    Negative indices wrap (so "-1" is the last layer whatever DEPTH is), which is what makes
    the winning `NGRAM_VE_LAYERS="-1"` configuration survive a depth change instead of
    silently pointing at the wrong layer.
    """
    if not NGRAM_VE_LAYERS:
        return frozenset()
    return frozenset((int(tok) % n_layer for tok in NGRAM_VE_LAYERS.split(',') if tok.strip()))

def has_ngram_bigram(layer_idx: int, n_layer: int) -> bool:
    """Bigram table on every VE layer (the same has_ve() rule the unigram VE uses), further
    restricted to NGRAM_VE_LAYERS when that is set."""
    if not (USE_NGRAM_VE and USE_VE and has_ve(layer_idx, n_layer)):
        return False
    keep = ngram_layer_set(n_layer)
    return not keep or layer_idx in keep

def has_ngram_trigram(layer_idx: int, n_layer: int) -> bool:
    """Trigram table on the FIRST and LAST VE layer only (Recursive's allocation)."""
    if not (NGRAM_VE_TRIGRAM and has_ngram_bigram(layer_idx, n_layer)):
        return False
    ve = [i for i in range(n_layer) if has_ve(i, n_layer)]
    return layer_idx in (ve[0], ve[-1])
ENGRAM_HASH_PRIMES = tuple((_next_prime(20011 + 7919 * k) for k in range(24)))

def engram_orders() -> tuple[int, ...]:
    """ENGRAM_ORDERS parsed. Only 2 and 3 are supported (Engram's own choice; 4-grams were
    reported "slightly suboptimal" and the index shift would need a third pad column)."""
    orders = tuple((int(o) for o in ENGRAM_ORDERS.split(',') if o.strip()))
    assert orders and all((o in (2, 3) for o in orders)), f'ENGRAM_ORDERS={ENGRAM_ORDERS!r}'
    return orders

def engram_hash_primes(head_idx: int, order: int) -> tuple[int, ...]:
    """The `order` multipliers for hash head `head_idx` of that order, oldest token first.

    Orders get disjoint 12-prime regions so no head of order 2 can ever share a multiplier with
    a head of order 3; within a region each head takes its own consecutive slice.
    """
    base = (order - 2) * 12 + head_idx * order
    assert base + order <= len(ENGRAM_HASH_PRIMES), f'ENGRAM_HEADS={ENGRAM_HEADS} needs more than {len(ENGRAM_HASH_PRIMES)} hash primes'
    return tuple((ENGRAM_HASH_PRIMES[base + j] for j in range(order)))

def engram_site_index(n_layer: int) -> int:
    """ENGRAM_SITE resolved to an absolute block index (negatives count from the end)."""
    return int(ENGRAM_SITE) % n_layer
WINDOW_PATTERN = 'L'
SHORT_WINDOW_FRAC = 0.5
ROPE_BASE = 100000
USE_LOCAL_CONV = True
LOCAL_CONV_KERNEL = 2
TRAIN_LOCAL_CONV = True
USE_GATED_MLP = False
USE_COALESCED_SWIGLU = False
COALESCED_SWIGLU_HIDDEN = 0
USE_WEIGHT_CACHE = True
USE_TRANSPOSED_LINEAR = False
USE_DOC_MASK = False
USE_SELECTIVE_GATE = False
SELECTIVE_GATE_CHANNELS = 32
USE_FUSED_QKV = False
USE_CAUSAL_FASTPATH = True

def doc_boundary_mask(idx: torch.Tensor) -> torch.Tensor:
    """Additive mask: 0 where i,j fall in the same packed document (by BOS-delimited segment), -inf otherwise.
    Combine (add) with the existing static causal/window mask -- both are additive float masks."""
    is_bos = idx == BOS_TOKEN_ID
    doc_id = torch.cumsum(is_bos.long(), dim=1)
    same_doc = doc_id.unsqueeze(2) == doc_id.unsqueeze(1)
    mask = torch.zeros(idx.shape[0], 1, idx.shape[1], idx.shape[1], dtype=torch.float32, device=idx.device)
    mask.masked_fill_(~same_doc.unsqueeze(1), float('-inf'))
    return mask
BOS_TOKEN_ID = -1

class CausalConv(nn.Module):
    """Depthwise causal conv1d: zero-initialized, so the residual branch it's added into starts as a true no-op."""

    def __init__(self, n_embd: int, kernel_size: int):
        super().__init__()
        self.n_embd = n_embd
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(n_embd, 1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if USE_NKI_LOCAL_CONV and x.device.type == 'neuron' and (x.dtype in (torch.float32, torch.bfloat16)) and (self.kernel_size == 2) and (x.shape[1] % 128 == 0) and (x.shape[2] % 128 == 0):
            if not _HAS_NKI:
                raise RuntimeError('--nki-local-conv requires the installed native NKI stack')
            return _Conv2NKI.apply(x, self.weight)
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
    mask.masked_fill_(~visible, float('-inf'))
    return mask[None, None, :, :]

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
COMPUTE_DTYPE: torch.dtype = torch.bfloat16
USE_BF16_NORM_OUTPUT = False
USE_BF16_SDPA_INPUT = False

def norm(x: torch.Tensor) -> torch.Tensor:
    if USE_BF16_NORM_OUTPUT:
        return F.rms_norm(x.float(), (x.size(-1),), eps=_RMS_EPS).to(torch.bfloat16)
    return F.rms_norm(x, (x.size(-1),))
_NKI_PARTITION = 128
_NKI_ATTN_P = 128
_NKI_ATTN_KV_TILE = 512
_NKI_ATTN_Q_TILE = 128
_RMS_EPS = 1.1920928955078125e-07
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
            nisa.tensor_scalar(dst=rb, data=xb, op0=nl.maximum, operand0=0.0)
            yb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yb, data1=rb, data2=rb, op=nl.multiply)
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
            nisa.tensor_scalar(dst=rb, data=xb, op0=nl.maximum, operand0=0.0)
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
            x, = ctx.saved_tensors
            flat_x = x.reshape(-1, x.shape[-1])
            flat_g = grad_out.reshape(-1, x.shape[-1]).contiguous()
            return _relu_sq_bwd(flat_x, flat_g).reshape(x.shape)

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
    def _softcap_ce_fwd(z, tgt):
        """Per-token softcap+cross-entropy forward for z [N, V] fp32.

        tgt is [N, 1] fp32 target class indices (exact for V < 2**24). Returns
        (loss [N, 1], lse [N, 1]), both fp32; lse is saved for the backward.
        """
        N, V = z.shape
        P = _NKI_PARTITION
        S = LOGIT_SOFTCAP
        loss = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        lse = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        idx = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=idx, pattern=[[1, V]], offset=0, channel_multiplier=0)
        C = 512 if V > 512 and V % 512 == 0 else V
        R = V // C
        for i in nl.affine_range(N // P):
            rows = nl.ds(i * P, P)
            zb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=zb, src=z[rows])
            cb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cb, src=tgt[rows])
            tb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=tb, op=nl.tanh, data=zb, scale=1.0 / S)
            mb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mb, data=idx, op0=nl.equal, operand0=cb)
            nisa.tensor_tensor(dst=mb, data1=mb, data2=tb, op=nl.multiply)
            tc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tc, op=nl.add, data=mb, axis=(1,))
            yc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=yc, data=tc, op0=nl.multiply, operand0=S)
            nyc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=nyc, data=tc, op0=nl.multiply, operand0=-S)
            parts = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
            for k in nl.affine_range(R):
                cols = nl.ds(k * C, C)
                nisa.activation(dst=zb[:, cols], op=nl.exp, data=tb[:, cols], scale=S, bias=nyc, reduce_op=nl.add, reduce_res=parts[:, nl.ds(k, 1)], reduce_cmd=nisa.reduce_cmd.reset_reduce)
            d = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=d, op=nl.add, data=parts, axis=(1,))
            lo = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=lo, op=nl.log, data=d)
            lb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=lb, data1=yc, data2=lo, op=nl.add)
            nisa.dma_copy(dst=lse[rows], src=lb)
            nisa.dma_copy(dst=loss[rows], src=lo)
        return (loss, lse)

    @nki.jit
    def _softcap_ce_bwd(z, tgt, lse, g):
        """Gradient of _softcap_ce_fwd w.r.t. the raw (pre-softcap) logits z.

        g is the [N, 1] fp32 upstream gradient of the per-token loss vector, so
        this implements the reduction="none" backward; mean/sum scaling is left
        to autograd outside the kernel.
        """
        N, V = z.shape
        P = _NKI_PARTITION
        S = LOGIT_SOFTCAP
        dz = nl.ndarray((N, V), dtype=nl.float32, buffer=nl.shared_hbm)
        idx = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=idx, pattern=[[1, V]], offset=0, channel_multiplier=0)
        for i in nl.affine_range(N // P):
            rows = nl.ds(i * P, P)
            zb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=zb, src=z[rows])
            cb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cb, src=tgt[rows])
            gb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gb, src=g[rows])
            nb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=nb, src=lse[rows])
            nisa.tensor_scalar(dst=nb, data=nb, op0=nl.multiply, operand0=-1.0)
            tb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=tb, op=nl.tanh, data=zb, scale=1.0 / S)
            nisa.activation(dst=zb, op=nl.exp, data=tb, scale=S, bias=nb)
            mb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mb, data=idx, op0=nl.equal, operand0=cb, op1=nl.multiply, operand1=gb)
            nisa.scalar_tensor_tensor(dst=zb, data=zb, op0=nl.multiply, operand0=gb, op1=nl.subtract, operand1=mb)
            nisa.activation(dst=mb, op=nl.square, data=tb)
            nisa.scalar_tensor_tensor(dst=tb, data=mb, op0=nl.subtract, operand0=1.0, reverse0=True, op1=nl.multiply, operand1=zb)
            nisa.dma_copy(dst=dz[rows], src=tb)
        return dz

    class _SoftcapCrossEntropyNKI(torch.autograd.Function):
        """Makes the fused softcap+cross-entropy kernel differentiable.

        Forward returns the UNREDUCED per-token loss, exactly like
        F.cross_entropy(..., reduction="none"); mean/none handling stays in the
        caller so this changes no reduction semantics.
        """

        @staticmethod
        def forward(ctx, z, tgt):
            loss, lse = _softcap_ce_fwd(z, tgt)
            ctx.save_for_backward(z, tgt, lse)
            return loss.reshape(-1)

        @staticmethod
        def backward(ctx, grad_out):
            z, tgt, lse = ctx.saved_tensors
            g = grad_out.reshape(-1, 1).float().contiguous()
            return (_softcap_ce_bwd(z, tgt, lse, g), None)

    @nki.jit
    def _muon_ns_poly(t1, t2, cb, cc):
        """b*t1 + c*t2 for a 2-D [M, N] bf16 pair -- 3 aten launches become 1.

        This is the Newton-Schulz polynomial `b*xtx + c*(xtx@xtx)`. The two matmuls
        feeding it stay in aten. `cb`/`cc` are [128,1] fp32 runtime operands.

        M is a compile-time constant (nki.jit specialises per shape), so both the
        128-partition tiling and its remainder resolve at trace time -- the real Muon
        groups include a 15-row flattened shape, so the remainder case is load-bearing
        and a bare affine_range over ceil(M/128) would read out of bounds.
        """
        M, N = t1.shape
        out = nl.ndarray((M, N), dtype=t1.dtype, buffer=nl.shared_hbm)
        n_full = M // _NKI_PARTITION
        rem = M - n_full * _NKI_PARTITION
        for i in range(n_full + (1 if rem else 0)):
            start = i * _NKI_PARTITION
            size = _NKI_PARTITION if i < n_full else rem
            a1 = nl.ndarray((size, N), dtype=t1.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a1, src=t1[start:start + size, 0:N])
            a2 = nl.ndarray((size, N), dtype=t2.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a2, src=t2[start:start + size, 0:N])
            kb = nl.ndarray((size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=kb, src=cb[0:size, 0:1])
            kc = nl.ndarray((size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=kc, src=cc[0:size, 0:1])
            m1 = nl.ndarray((size, N), dtype=t1.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m1, data=a1, op0=nl.multiply, operand0=kb)
            m2 = nl.ndarray((size, N), dtype=t1.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m2, data=a2, op0=nl.multiply, operand0=kc)
            s = nl.ndarray((size, N), dtype=t1.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=s, data1=m1, data2=m2, op=nl.add)
            nisa.dma_copy(dst=out[start:start + size, 0:N], src=s)
        return out

    @nki.jit
    def _muon_ns_axpy(x, pm, ca):
        """a*x + pm for a 2-D [M, N] bf16 pair -- 2 aten launches become 1.

        The Newton-Schulz update `x = a*x + x @ (...)`, with the matmul left in aten
        and `ca` a [128,1] fp32 runtime operand. Addition is commutative in IEEE, so
        computing a*x first and adding the (already-materialised) matmul result gives
        exactly eager's `add(a*x, x@m)`.
        """
        M, N = x.shape
        out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
        n_full = M // _NKI_PARTITION
        rem = M - n_full * _NKI_PARTITION
        for i in range(n_full + (1 if rem else 0)):
            start = i * _NKI_PARTITION
            size = _NKI_PARTITION if i < n_full else rem
            xb = nl.ndarray((size, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[start:start + size, 0:N])
            pb = nl.ndarray((size, N), dtype=pm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=pb, src=pm[start:start + size, 0:N])
            ka = nl.ndarray((size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ka, src=ca[0:size, 0:1])
            ax = nl.ndarray((size, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ax, data=xb, op0=nl.multiply, operand0=ka)
            yb = nl.ndarray((size, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yb, data1=ax, data2=pb, op=nl.add)
            nisa.dma_copy(dst=out[start:start + size, 0:N], src=yb)
        return out

    @nki.jit
    def _muon_update(g, w, klr, klw):
        """The whole sign-gated Muon weight update -- 9 aten launches become 1.

            mask = (g*w) >= 0
            out  = w - (lr*g + (lr*wd)*w*mask)

        `g` is bf16 [M, N] (already scaled), `w` fp32 [M, N]; `klr` and `klw` are
        [128,1] fp32 runtime operands holding lr and lr*wd, both formed on the host in
        float32 so they equal the device's own fp32 products bit for bit. Returning a
        fresh tensor instead of updating `w` in place is free here: `stacked_params` is
        a scratch buffer refilled from the live params at the top of every step, so the
        caller just rebinds it.

        The mask is materialised as 0.0/1.0 and multiplied in, which is what eager's
        `fp32 * bool` does; the multiply is exact either way, so no rounding question
        arises from the gate itself.
        """
        M, N = g.shape
        out = nl.ndarray((M, N), dtype=w.dtype, buffer=nl.shared_hbm)
        n_full = M // _NKI_PARTITION
        rem = M - n_full * _NKI_PARTITION
        for i in range(n_full + (1 if rem else 0)):
            start = i * _NKI_PARTITION
            size = _NKI_PARTITION if i < n_full else rem
            gb = nl.ndarray((size, N), dtype=g.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=gb, src=g[start:start + size, 0:N])
            wb = nl.ndarray((size, N), dtype=w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wb, src=w[start:start + size, 0:N])
            a_lr = nl.ndarray((size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=a_lr, src=klr[0:size, 0:1])
            a_lw = nl.ndarray((size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=a_lw, src=klw[0:size, 0:1])
            f32 = nl.float32
            s1 = nl.ndarray((size, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=s1, data1=gb, data2=wb, op=nl.multiply)
            s2 = nl.ndarray((size, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=s2, data=s1, op0=nl.greater_equal, operand0=0.0)
            s3 = nl.ndarray((size, N), dtype=g.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=s3, data=gb, op0=nl.multiply, operand0=a_lr)
            s4 = nl.ndarray((size, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=s4, data=wb, op0=nl.multiply, operand0=a_lw)
            s5 = nl.ndarray((size, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=s5, data1=s4, data2=s2, op=nl.multiply)
            nisa.tensor_tensor(dst=s1, data1=s3, data2=s5, op=nl.add)
            nisa.tensor_tensor(dst=s2, data1=wb, data2=s1, op=nl.subtract)
            nisa.dma_copy(dst=out[start:start + size, 0:N], src=s2)
        return out

    @nki.jit
    def _ngram_rmsprop_fp32_nki(p, grad, moment, scalars, beta2: float, eps: float, lanes: int):
        """BF16 table/gradient, FP32 moment and arithmetic; explicit copy-back outside.

        Inputs are contiguous [M,2048] views, with M divisible by 128*lanes.
        scalars[128,2] holds runtime LR and inverse bias correction. Separate
        output buffers were faster than mutable-input alias handling on the
        installed native backend. Two programs use the two physical cores of LNC2.
        """
        m, f = p.shape
        assert lanes in (1, 2) and m % (128 * lanes) == 0 and (f == 2048)
        out_p = nl.ndarray(p.shape, dtype=p.dtype, buffer=nl.shared_hbm)
        out_v = nl.ndarray(moment.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        lr = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        inv_bias = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lr, src=scalars[:, 0:1])
        nisa.dma_copy(dst=inv_bias, src=scalars[:, 1:2])
        lane = nl.program_id(0)
        for tile in nl.affine_range(m // (128 * lanes)):
            start = (tile * lanes + lane) * 128
            ps = nl.ndarray((128, f), dtype=p.dtype, buffer=nl.sbuf)
            gs = nl.ndarray((128, f), dtype=grad.dtype, buffer=nl.sbuf)
            vs = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ps, src=p[start:start + 128, :])
            nisa.dma_copy(dst=gs, src=grad[start:start + 128, :])
            nisa.dma_copy(dst=vs, src=moment[start:start + 128, :])
            gg = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gg, data1=gs, data2=gs, op=nl.multiply)
            dv = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dv, data1=gg, data2=vs, op=nl.subtract)
            nv = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=nv, data=dv, op0=nl.multiply, operand0=1.0 - beta2, op1=nl.add, operand1=vs)
            debiased = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=debiased, data=nv, op0=nl.multiply, operand0=inv_bias)
            denom = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=denom, data=debiased, op=nl.sqrt)
            nisa.tensor_scalar(dst=denom, data=denom, op0=nl.add, operand0=eps)
            inv_denom = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv_denom, data=denom)
            update = nl.ndarray((128, f), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=update, data1=gs, data2=inv_denom, op=nl.multiply)
            nisa.tensor_scalar(dst=update, data=update, op0=nl.multiply, operand0=lr)
            np = nl.ndarray((128, f), dtype=p.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=np, data1=ps, data2=update, op=nl.subtract)
            nisa.dma_copy(dst=out_p[start:start + 128, :], src=np)
            nisa.dma_copy(dst=out_v[start:start + 128, :], src=nv)
        return (out_p, out_v)

    @nki.jit
    def _attn_nki_fwd(q, k, v, mask, scale, T, D):
        """q/k/v: [BH*T, D] bf16 HBM. mask: [T, T] bf16 HBM. Returns [BH*T, D]."""
        BH = q.shape[0] // T
        out = nl.ndarray((BH * T, D), dtype=q.dtype, buffer=nl.shared_hbm)
        for bh in nl.affine_range(BH):
            base = bh * T
            q_sbuf = nl.ndarray((_NKI_ATTN_P, T), dtype=q.dtype, buffer=nl.sbuf)
            k_sbuf = nl.ndarray((_NKI_ATTN_P, T), dtype=q.dtype, buffer=nl.sbuf)
            for c in range(T // _NKI_ATTN_P):
                qc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=qc, src=q[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                qt = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=qt, data=qc)
                nisa.tensor_copy(dst=q_sbuf[0:_NKI_ATTN_P, c * _NKI_ATTN_P:(c + 1) * _NKI_ATTN_P], src=qt)
                kc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=kc, src=k[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                kt = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=kt, data=kc)
                nisa.tensor_copy(dst=k_sbuf[0:_NKI_ATTN_P, c * _NKI_ATTN_P:(c + 1) * _NKI_ATTN_P], src=kt)
            for qi in range(T // _NKI_ATTN_Q_TILE):
                r0 = qi * _NKI_ATTN_Q_TILE
                n_kv = T // _NKI_ATTN_KV_TILE
                qk_tiles = []
                for _i in range(n_kv):
                    qk_tiles.append(nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_KV_TILE), dtype=nl.float32, buffer=nl.psum))
                for ki in range(n_kv):
                    nisa.nc_matmul(dst=qk_tiles[ki], stationary=q_sbuf[0:_NKI_ATTN_P, r0:r0 + _NKI_ATTN_Q_TILE], moving=k_sbuf[0:_NKI_ATTN_P, ki * _NKI_ATTN_KV_TILE:(ki + 1) * _NKI_ATTN_KV_TILE])
                qk_full = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                for ki in range(n_kv):
                    nisa.tensor_copy(dst=qk_full[0:_NKI_ATTN_P, ki * _NKI_ATTN_KV_TILE:(ki + 1) * _NKI_ATTN_KV_TILE], src=qk_tiles[ki])
                nisa.tensor_scalar(dst=qk_full, data=qk_full, op0=nl.multiply, operand0=scale)
                row_max = nl.max(qk_full, axis=1, keepdims=True)
                sub_full = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=sub_full, data=qk_full, op0=nl.subtract, operand0=row_max)
                exp_full = nl.exp(sub_full)
                mb = nl.ndarray((_NKI_ATTN_P, T), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=mb, src=mask[r0:r0 + _NKI_ATTN_Q_TILE, :])
                m = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=m, src=mb)
                nisa.tensor_tensor(dst=exp_full, data1=exp_full, data2=m, op=nl.multiply)
                sum_row = nl.sum(exp_full, axis=1, keepdims=True)
                inv = nl.reciprocal(sum_row)
                norm_full = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=norm_full, data=exp_full, op0=nl.multiply, operand0=inv)
                scores = nl.ndarray((_NKI_ATTN_P, T), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=scores, src=norm_full)
                acc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.psum)
                for c in range(T // _NKI_ATTN_P):
                    st = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=st, data=scores[0:_NKI_ATTN_P, c * _NKI_ATTN_P:(c + 1) * _NKI_ATTN_P])
                    st_s = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=st_s, src=st)
                    vc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=vc, src=v[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                    nisa.nc_matmul(dst=acc, accumulate=c > 0, stationary=st_s, moving=vc)
                ob = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ob, src=acc)
                ob16 = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ob16, src=ob)
                nisa.dma_copy(dst=out[base + r0:base + r0 + _NKI_ATTN_Q_TILE, :], src=ob16)
        return out

    @nki.jit
    def _attn_nki_bwd(q, k, v, do, mask, scale, dq, dk, dv, dk_buf, dv_buf, T, D):
        """q/k/v/do: [BH*T, D] bf16 HBM. mask: [T, T] bf16 HBM.
        dq/dk/dv: [BH*T, D] bf16 HBM outputs. dk_buf/dv_buf: [BH*T, D]
        fp32 HBM scratch, zeroed by host (cross-qi accumulation)."""
        BH = q.shape[0] // T
        for bh in nl.affine_range(BH):
            base = bh * T
            q_sbuf = nl.ndarray((_NKI_ATTN_P, T), dtype=q.dtype, buffer=nl.sbuf)
            k_sbuf = nl.ndarray((_NKI_ATTN_P, T), dtype=q.dtype, buffer=nl.sbuf)
            for c in range(T // _NKI_ATTN_P):
                qc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=qc, src=q[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                qt = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=qt, data=qc)
                nisa.tensor_copy(dst=q_sbuf[0:_NKI_ATTN_P, c * _NKI_ATTN_P:(c + 1) * _NKI_ATTN_P], src=qt)
                kc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=kc, src=k[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                kt = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=kt, data=kc)
                nisa.tensor_copy(dst=k_sbuf[0:_NKI_ATTN_P, c * _NKI_ATTN_P:(c + 1) * _NKI_ATTN_P], src=kt)
            for qi in range(T // _NKI_ATTN_Q_TILE):
                r0 = qi * _NKI_ATTN_Q_TILE
                n_kv = T // _NKI_ATTN_KV_TILE
                qk_tiles = []
                for _i in range(n_kv):
                    qk_tiles.append(nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_KV_TILE), dtype=nl.float32, buffer=nl.psum))
                for ki in range(n_kv):
                    nisa.nc_matmul(dst=qk_tiles[ki], stationary=q_sbuf[0:_NKI_ATTN_P, r0:r0 + _NKI_ATTN_Q_TILE], moving=k_sbuf[0:_NKI_ATTN_P, ki * _NKI_ATTN_KV_TILE:(ki + 1) * _NKI_ATTN_KV_TILE])
                qk_full = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                for ki in range(n_kv):
                    nisa.tensor_copy(dst=qk_full[0:_NKI_ATTN_P, ki * _NKI_ATTN_KV_TILE:(ki + 1) * _NKI_ATTN_KV_TILE], src=qk_tiles[ki])
                nisa.tensor_scalar(dst=qk_full, data=qk_full, op0=nl.multiply, operand0=scale)
                row_max = nl.max(qk_full, axis=1, keepdims=True)
                sub_full = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=sub_full, data=qk_full, op0=nl.subtract, operand0=row_max)
                exp_full = nl.exp(sub_full)
                mb = nl.ndarray((_NKI_ATTN_P, T), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=mb, src=mask[r0:r0 + _NKI_ATTN_Q_TILE, :])
                m = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=m, src=mb)
                nisa.tensor_tensor(dst=exp_full, data1=exp_full, data2=m, op=nl.multiply)
                sum_row = nl.sum(exp_full, axis=1, keepdims=True)
                inv = nl.reciprocal(sum_row)
                p_full = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=p_full, data=exp_full, op0=nl.multiply, operand0=inv)
                p16 = nl.ndarray((_NKI_ATTN_P, T), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=p16, src=p_full)
                do_b = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=do_b, src=do[base + r0:base + r0 + _NKI_ATTN_Q_TILE, :])
                o_acc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.psum)
                for c in range(T // _NKI_ATTN_P):
                    st = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=st, data=p16[0:_NKI_ATTN_P, c * _NKI_ATTN_P:(c + 1) * _NKI_ATTN_P])
                    st_s = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=st_s, src=st)
                    vc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=vc, src=v[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                    nisa.nc_matmul(dst=o_acc, accumulate=c > 0, stationary=st_s, moving=vc)
                o_b = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_b, src=o_acc)
                do_f = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=do_f, src=do_b)
                do_o = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=do_o, data1=do_f, data2=o_b, op=nl.multiply)
                d_row = nl.sum(do_o, axis=1, keepdims=True)
                q_b = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_b, src=q[base + r0:base + r0 + _NKI_ATTN_Q_TILE, :])
                dq_acc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.psum)
                for c in range(T // _NKI_ATTN_P):
                    vc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=vc, src=v[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                    dv_c = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dv_c, stationary=p16[0:_NKI_ATTN_P, c * _NKI_ATTN_P:(c + 1) * _NKI_ATTN_P], moving=do_b)
                    dv_old = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(dst=dv_old, src=dv_buf[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                    dv_new = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dv_new, src=dv_c)
                    nisa.tensor_tensor(dst=dv_new, data1=dv_new, data2=dv_old, op=nl.add)
                    nisa.dma_copy(dst=dv_buf[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :], src=dv_new)
                    do_t = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=do_t, data=do_b)
                    do_ts = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=do_ts, src=do_t)
                    vc_t = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=vc_t, data=vc)
                    vc_ts = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=vc_ts, src=vc_t)
                    dp = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dp, stationary=do_ts, moving=vc_ts)
                    dp_s = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dp_s, src=dp)
                    ds = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=ds, data=dp_s, op0=nl.subtract, operand0=d_row)
                    pblk = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=pblk, src=p_full[0:_NKI_ATTN_P, c * _NKI_ATTN_P:(c + 1) * _NKI_ATTN_P])
                    nisa.tensor_tensor(dst=ds, data1=ds, data2=pblk, op=nl.multiply)
                    nisa.tensor_scalar(dst=ds, data=ds, op0=nl.multiply, operand0=scale)
                    ds16 = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ds16, src=ds)
                    kc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=kc, src=k[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                    dst_t = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=dst_t, data=ds16)
                    dst_s = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dst_s, src=dst_t)
                    nisa.nc_matmul(dst=dq_acc, accumulate=c > 0, stationary=dst_s, moving=kc)
                    dk_c = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dk_c, stationary=ds16, moving=q_b)
                    dk_old = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(dst=dk_old, src=dk_buf[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                    dk_new = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dk_new, src=dk_c)
                    nisa.tensor_tensor(dst=dk_new, data1=dk_new, data2=dk_old, op=nl.add)
                    nisa.dma_copy(dst=dk_buf[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :], src=dk_new)
                dq_s = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dq_s, src=dq_acc)
                dq16 = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dq16, src=dq_s)
                nisa.dma_copy(dst=dq[base + r0:base + r0 + _NKI_ATTN_Q_TILE, :], src=dq16)
            for c in range(T // _NKI_ATTN_P):
                fblk = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=fblk, src=dk_buf[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                hblk = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=hblk, src=fblk)
                nisa.dma_copy(dst=dk[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :], src=hblk)
                nisa.dma_copy(dst=fblk, src=dv_buf[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                nisa.tensor_copy(dst=hblk, src=fblk)
                nisa.dma_copy(dst=dv[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :], src=hblk)
        return (dq, dk, dv)

    class _AttnNKI(torch.autograd.Function):
        """Causal SDPA via the fused NKI fwd/bwd pair above.

        q/k/v are [B, H, T, D] (transpose views OK -- reshape materializes
        bh-major row order); returns y [B, H, T, D]. Validated dtypes
        bf16/fp32, head dim 128, T a multiple of 512 (checked by the caller).
        """

        @staticmethod
        def forward(ctx, q, k, v, scale):
            b, h, t, d = q.shape
            q3 = q.reshape(b * h * t, d).contiguous()
            k3 = k.reshape(b * h * t, d).contiguous()
            v3 = v.reshape(b * h * t, d).contiguous()
            mask = _attn_nki_mask(t, q.device, q.dtype)
            y3 = _attn_nki_fwd(q3, k3, v3, mask, scale, t, d)
            ctx.save_for_backward(q3, k3, v3)
            ctx.bhtd = (b, h, t, d)
            ctx.scale = scale
            return y3.view(b, h, t, d)

        @staticmethod
        def backward(ctx, grad_out):
            q3, k3, v3 = ctx.saved_tensors
            b, h, t, d = ctx.bhtd
            scale = ctx.scale
            g3 = grad_out.reshape(b * h * t, d).contiguous()
            mask = _attn_nki_mask(t, q3.device, q3.dtype)
            n = b * h * t
            dev = q3.device
            dq3 = torch.zeros((n, d), dtype=q3.dtype, device=dev)
            dk3 = torch.zeros((n, d), dtype=q3.dtype, device=dev)
            dv3 = torch.zeros((n, d), dtype=q3.dtype, device=dev)
            dk_buf = torch.zeros((n, d), dtype=torch.float32, device=dev)
            dv_buf = torch.zeros((n, d), dtype=torch.float32, device=dev)
            dq3, dk3, dv3 = _attn_nki_bwd(q3, k3, v3, g3, mask, scale, dq3, dk3, dv3, dk_buf, dv_buf, t, d)
            return (dq3.view(b, h, t, d), dk3.view(b, h, t, d), dv3.view(b, h, t, d), None)
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
USE_NKI_RELU2 = False
USE_NKI_ROPE_NORM = True
USE_NKI_SOFTCAP_CE = True
USE_NKI_MUON = True
USE_NKI_ATTN = False
USE_NKI_LOCAL_CONV = False
COMPILE_SDPA_DIRECT = False
_NKI_ATTN_MASK_CACHE: dict = {}

def _attn_nki_mask(t, device, dtype):
    """[T, T] causal mask (1.0 keep / 0.0 drop), cached per (T, device, dtype)."""
    key = (t, str(device), str(dtype))
    m = _NKI_ATTN_MASK_CACHE.get(key)
    if m is None:
        keep = torch.arange(t).unsqueeze(1) >= torch.arange(t).unsqueeze(0)
        m = torch.where(keep, torch.tensor(1.0), torch.tensor(0.0)).to(dtype).to(device)
        _NKI_ATTN_MASK_CACHE[key] = m
    return m

def softcap_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, loss_reduction: str) -> torch.Tensor:
    """`LOGIT_SOFTCAP*tanh(z/S)` then cross-entropy, fused into one NKI kernel pair.

    Replaces only the softcap+CE math. The kernel produces the unreduced per-token
    loss, so the mean/none branching below is identical to what F.cross_entropy's
    own `reduction` argument did -- reduction semantics are unchanged either way,
    and the eager branch is the original chain verbatim so flipping the knob off
    restores the previous numerics exactly.

    LOSS_BYTE_WEIGHT (default OFF): when on and loss_reduction == "mean", the
    per-token losses are combined as sum(w_i * l_i)/sum(w_i) with w = the token's
    byte length (mean-normalized in main(), so the loss scale -- and hence every
    LR -- is unchanged). Works through BOTH the NKI and eager paths: the flag
    only reroutes "mean" to per-token + weighting; "none" still returns the raw
    per-token losses unweighted.
    """
    weight = None
    if loss_reduction == 'mean' and LOSS_BYTE_WEIGHT and (BYTE_LOSS_W is not None):
        weight = BYTE_LOSS_W.to(device=logits.device, dtype=torch.float32)[targets.reshape(-1)]
        loss_reduction = 'none'
    if USE_NKI_SOFTCAP_CE and _HAS_NKI and (logits.device.type == 'neuron') and (loss_reduction in ('mean', 'none')) and (logits.dtype == torch.float32) and logits.is_contiguous():
        z = logits.reshape(-1, logits.shape[-1])
        if z.shape[0] % _NKI_PARTITION == 0:
            tgt = targets.reshape(-1, 1).to(torch.float32)
            per_token = _SoftcapCrossEntropyNKI.apply(z, tgt)
            if loss_reduction == 'none':
                out = per_token.view_as(targets)
            else:
                out = per_token.mean()
            if weight is not None:
                return (out.reshape(-1) * weight).sum() / weight.sum()
            return out
    capped = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
    loss = F.cross_entropy(capped.reshape(-1, capped.size(-1)), targets.reshape(-1), reduction=loss_reduction)
    if loss_reduction == 'none':
        out = loss.view_as(targets)
    else:
        out = loss
    if weight is not None:
        return (out.reshape(-1) * weight).sum() / weight.sum()
    return out

def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """relu(x - RELU2_TAU)**2, via the example NKI kernel when enabled, else eager.

    T3.9: RELU2_TAU is Recursive's "shifted ReLU^2" (tau=0.5 in their recipe, 0.0 = the
    historical activation). The shift is a compile-time Python float folded into one subtraction,
    NOT a per-step-changing scalar, so it is baked into the graph exactly once -- the rule about
    new scalars needing to be device tensors applies to values that CHANGE per step.
    """
    if USE_NKI_RELU2 and _HAS_NKI and (x.device.type == 'neuron') and (x.numel() // x.shape[-1] % _NKI_PARTITION == 0):
        return _ReluSquaredNKI.apply(x)
    if RELU2_TAU != 0.0:
        return F.relu(x - RELU2_TAU).square()
    return F.relu(x).square()

class Linear(nn.Linear):
    """Linear without bias that casts weight to input dtype (allows bf16 input with f32 weight).

    When USE_WEIGHT_CACHE is on, the cast copy is refreshed once per optimizer step
    (see refresh_weight_cache, called from the training loop) instead of being
    recomputed on every forward call across the grad-accum inner loop.

    T1.6: when `weight_transposed` is set (by transpose_linear_weights(), driven by
    USE_TRANSPOSED_LINEAR), `self.weight` holds [in_features, out_features] instead of
    nn.Linear's [out_features, in_features] and forward does a plain `x @ weight`. The flag
    is per-instance rather than read off the global in forward() so that a module which was
    never converted still works while the global is on -- forward must describe the tensor it
    actually holds, never what the global wishes it held.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_weight: torch.Tensor | None = None
        self.weight_transposed = False

    def refresh_weight_cache(self, dtype: torch.dtype) -> None:
        self._cached_weight = self.weight.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if USE_WEIGHT_CACHE and self._cached_weight is not None and (self._cached_weight.dtype == x.dtype):
            w = self._cached_weight
        else:
            w = self.weight.to(dtype=x.dtype)
        if self.weight_transposed:
            return x @ w
        return F.linear(x, w, None)
_FNV64_OFFSET = 14695981039346656037
_FNV64_PRIME = 1099511628211
_FNV64_MASK = 18446744073709551615

def _fnv1a64(data: bytes) -> int:
    """FNV-1a 64. Deliberately NOT Python's builtin hash(): that is randomized per process for
    bytes/str (PYTHONHASHSEED), which would make the init non-reproducible across runs -- the one
    property an init function must never lack. Hand-rolled rather than hashlib so it stays in the
    same style as NGRAM_HASH_PRIMES and costs no new import."""
    h = _FNV64_OFFSET
    for b in data:
        h = (h ^ b) * _FNV64_PRIME & _FNV64_MASK
    return h

def byte_feature_wte_init(wte: torch.Tensor, tokenizer, vocab_size: int) -> tuple[int, int, float]:
    """Plan 11.1: overwrite `wte`'s first `vocab_size` rows with a hashed byte-n-gram
    representation of each token's own byte string. Returns (rows_written, rows_skipped,
    mean_features_per_token) so the caller can print a null test.

    THE FEATURE MAP. For token id i with bytes `raw`, the feature string is
    `b"\\x02" + raw + b"\\x03"` (0x02/0x03 are STX/ETX, which cannot appear inside a UTF-8 byte
    string, so they are unambiguous boundary markers) and the features are every contiguous
    substring of length 1..BYTE_WTE_INIT_NGRAM. The markers are what make " the" and "the "
    different feature sets, i.e. what lets the init encode word-initial vs word-final position --
    which for a BPE vocabulary of mostly space-prefixed word pieces is most of the signal there is.

    THE PROJECTION is the hashing trick (signed feature hashing, as in fastText's subword buckets):
    each feature's 64-bit hash gives a column index (low bits) and a sign (bit 40, chosen away from
    the bits the modulo consumes so index and sign are independent). Accumulating +-1 into a
    length-n_embd vector is an unbiased random projection of the feature indicator vector, so two
    tokens' rows have expected inner product proportional to their shared-n-gram count. That is the
    whole mechanism: no learned component, no data, no external artifact.

    CENTER, THEN SCALE. The columns are first centered over the written rows (features present in
    every token would otherwise give every row the same large shared component -- measured +0.224
    mean random-pair cosine before centering, ~0.000 after), then each row is rescaled to a fixed
    RMS of BYTE_WTE_INIT_STD, both to match the normal_(0, 0.8) init the control uses and so that a
    long token does not start with a systematically larger embedding than a short one (AdamW is
    scale-invariant in the gradient but not in the parameter, so a 3x row-scale spread would be a 3x
    spread in effective relative LR).

    SPECIAL/PADDED ROWS keep their random init: a special token has no meaningful surface form (and
    prepare.py's own token_bytes table records 0 bytes for exactly these ids), and rows at
    index >= vocab_size are the vocab padding, never gathered by any real token id.
    """
    n_embd = wte.shape[-1]
    enc = tokenizer.enc
    special_ids = set((getattr(enc, '_special_tokens', None) or {}).values())
    special_ids.add(tokenizer.get_bos_token_id())
    feat = torch.zeros(vocab_size, n_embd, dtype=torch.float32)
    touched = torch.zeros(vocab_size, dtype=torch.bool)
    written = skipped = 0
    total_features = 0
    max_n = max(1, int(BYTE_WTE_INIT_NGRAM))
    for tid in range(vocab_size):
        if tid in special_ids:
            skipped += 1
            continue
        try:
            raw = enc.decode_single_token_bytes(tid)
        except Exception:
            skipped += 1
            continue
        if not raw:
            skipped += 1
            continue
        s = b'\x02' + raw + b'\x03'
        row = feat[tid]
        n_feat = 0
        for n in range(1, max_n + 1):
            for j in range(len(s) - n + 1):
                h = _fnv1a64(s[j:j + n])
                col = h % n_embd
                row[col] += 1.0 if h >> 40 & 1 else -1.0
                n_feat += 1
        touched[tid] = True
        written += 1
        total_features += n_feat
    rows = feat[touched]
    rows -= rows.mean(dim=0, keepdim=True)
    rms = rows.pow(2).mean(dim=-1, keepdim=True).sqrt()
    rows *= BYTE_WTE_INIT_STD / rms.clamp_min(1e-12)
    feat[touched] = rows
    degenerate = touched & (feat.abs().sum(dim=-1) == 0.0)
    if bool(degenerate.any()):
        n_deg = int(degenerate.sum())
        touched &= ~degenerate
        written -= n_deg
        skipped += n_deg
    mix = float(BYTE_WTE_INIT_MIX)
    keep = math.sqrt(max(0.0, 1.0 - mix * mix))
    with torch.no_grad():
        target = wte[:vocab_size]
        blended = mix * feat.to(target.dtype) + keep * target
        target.copy_(torch.where(touched.unsqueeze(-1), blended, target))
    return (written, skipped, total_features / written if written else 0.0)

def transpose_linear_weights(root: nn.Module) -> int:
    """T1.6: replace every Linear's [out, in] weight with its [in, out] transpose, in place.

    Returns the number of modules converted (0 when the flag is off, which callers print so a
    silent no-op can never masquerade as a measured arm).

    Called at the END of GPT.init_weights(), which is the single point both main() and
    load_for_eval() go through, so the training model and the eval model always agree on the
    layout. `state_dict` keys are UNCHANGED (still ".weight"); only the shape flips. That makes
    USE_TRANSPOSED_LINEAR shape-critical, hence its entry in SHAPE_CRITICAL_TOGGLES: a
    checkpoint written with it on will not strict-load with it off, and vice versa. That is
    load_for_eval() restores the saved, explicitly whitelisted architecture
    toggles before construction, so a CLI-selected layout also survives scoring
    in a fresh process without repeating the training command line.

    `.t().contiguous()` and not just `.t()`: a non-contiguous parameter would leave the compiler
    to materialize the transpose anyway, which is the entire cost this change exists to remove.
    """
    n = 0
    for m in root.modules():
        if isinstance(m, Linear) and (not m.weight_transposed):
            with torch.no_grad():
                wt = m.weight.detach().t().contiguous()
            m.weight = nn.Parameter(wt, requires_grad=m.weight.requires_grad)
            m.weight_transposed = True
            m._cached_weight = None
            n += 1
    return n

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = (x[..., :d], x[..., d:])
    y1 = x1 * cos + x2 * sin
    y2 = x1 * -sin + x2 * cos
    return torch.cat((y1, y2), dim=-1)

def rope_norm(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """norm(apply_rotary_emb(x, cos, sin)), fused into one NKI kernel when enabled.
    x is [B, T, H, D]; cos/sin are [1, T, 1, D//2] broadcast over batch and heads.
    The kernel needs B*T and T to be multiples of 128 (both true for the baseline:
    B*T = 2*1024, T = 1024) so that each 128-row tile maps to one contiguous run of
    positions within a single batch row. Anything else falls back to eager, so the
    knob is always safe to flip.
    """
    if USE_NKI_ROPE_NORM and _HAS_NKI and (x.device.type == 'neuron') and (x.ndim == 4):
        b, t, _, d = x.shape
        if b * t % _NKI_PARTITION == 0 and t % _NKI_PARTITION == 0 and (d % 2 == 0) and (cos.shape[1] == t) and (cos.shape[-1] == d // 2) and x.is_contiguous():
            return _RopeNormNKI.apply(x, cos, sin)
    return norm(apply_rotary_emb(x, cos, sin))

@torch._dynamo.disable
def nki_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None) -> torch.Tensor:
    """Causal SDPA on post-rope [B, H, T, D] q/k/v. With --nki-attn on Neuron
    hardware (and only when the validated kernel geometry holds: bf16/fp32,
    head dim 128, T a multiple of 512, q/k/v same shape), runs the fused NKI
    fwd/bwd pair; otherwise the original eager SDPA line verbatim, so the
    knob is always safe to flip.
    Dynamo is disabled around this node on purpose: tracing through the
    functorch autograd.Function wrapper poisoned a neighboring (rope)
    kernel's lowering (take-2 crash). Opaque node, kernels still on device.
    """
    if USE_NKI_ATTN and _HAS_NKI and (q.device.type == 'neuron') and (q.ndim == 4) and (q.shape == k.shape == v.shape) and (q.dtype in (torch.bfloat16, torch.float32)) and (q.shape[-1] == 128) and (q.shape[2] % 512 == 0):
        s = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
        return _AttnNKI.apply(q, k, v, s)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig, layer_idx: int=0, coalesced: str='none'):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim * self.n_head == self.n_embd
        self.n_kv_heads = config.n_kv_heads or config.n_head
        self.kv_repeat = 1
        if self.n_kv_heads != self.n_head:
            if USE_FUSED_QKV or coalesced != 'none':
                raise ValueError('GQA (n_kv_heads != n_head) needs the standard separate q/k/v projections: incompatible with USE_FUSED_QKV (one fused 3-way matmul) and with parallel-block coalescing (one c_in matmul).')
            if self.n_head % self.n_kv_heads != 0:
                raise ValueError(f'n_kv_heads={self.n_kv_heads} must divide n_head={self.n_head}')
            self.kv_repeat = self.n_head // self.n_kv_heads
        self.coalesced = coalesced
        if coalesced == 'all':
            pass
        elif coalesced == 'vfc':
            self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
            self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        elif USE_FUSED_QKV:
            self.c_qkv = Linear(config.n_embd, 3 * config.n_embd, bias=False)
        else:
            self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
            self.c_k = Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self.c_v = Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_scale = config.attn_scale if config.attn_scale > 0.0 else None
        self.ve_gate = Linear(VE_GATE_CHANNELS, self.n_head, bias=False) if USE_VE and has_ve(layer_idx, config.n_layer) else None
        self.ve_gate_bi = _rng_neutral(lambda: Linear(VE_GATE_CHANNELS, self.n_head, bias=False)) if has_ngram_bigram(layer_idx, config.n_layer) else None
        self.ve_gate_tri = _rng_neutral(lambda: Linear(VE_GATE_CHANNELS, self.n_head, bias=False)) if has_ngram_trigram(layer_idx, config.n_layer) else None
        if self.ve_gate_tri is not None:
            assert config.n_embd >= 3 * VE_GATE_CHANNELS, (config.n_embd, VE_GATE_CHANNELS)
        elif self.ve_gate_bi is not None:
            assert config.n_embd >= 2 * VE_GATE_CHANNELS, (config.n_embd, VE_GATE_CHANNELS)
        if USE_HEAD_GATE:
            self.head_gate_lo, self.head_gate_hi = head_gate_slice(config.n_embd, config.n_layer)
            self.head_gate = _rng_neutral(lambda: Linear(HEAD_GATE_CHANNELS, self.n_head, bias=False))
            self.head_gate_bias = nn.Parameter(torch.full((self.n_head,), float(HEAD_GATE_BIAS)))
        else:
            self.head_gate_lo = self.head_gate_hi = 0
            self.head_gate = None

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None, ve_bi: torch.Tensor | None=None, ve_tri: torch.Tensor | None=None, qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None=None) -> torch.Tensor:
        b, t, c = x_q.shape
        if qkv is not None:
            q, k, v = qkv
            if q is None:
                q = self.c_q(x_q).view(b, t, self.n_head, self.head_dim)
                k = self.c_k(x_k).view(b, t, self.n_head, self.head_dim)
        elif USE_FUSED_QKV:
            q, k, v = self.c_qkv(x_q).split(self.n_embd, dim=-1)
            q = q.view(b, t, self.n_head, self.head_dim)
            k = k.view(b, t, self.n_head, self.head_dim)
            v = v.view(b, t, self.n_head, self.head_dim)
        else:
            q = self.c_q(x_q).view(b, t, self.n_head, self.head_dim)
            k = self.c_k(x_k).view(b, t, self.n_kv_heads, self.head_dim)
            v = self.c_v(x_v).view(b, t, self.n_kv_heads, self.head_dim)
            if self.n_kv_heads != self.n_head:
                k = k.repeat_interleave(self.kv_repeat, dim=2).contiguous()
                v = v.repeat_interleave(self.kv_repeat, dim=2).contiguous()
        if ve is not None:
            ve = ve.view(b, t, self.n_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x_v[..., :VE_GATE_CHANNELS]))
            v = v + gate.unsqueeze(-1) * ve
        if ve_bi is not None:
            g_bi = 2 * torch.sigmoid(self.ve_gate_bi(x_v[..., VE_GATE_CHANNELS:2 * VE_GATE_CHANNELS]))
            v = v + g_bi.unsqueeze(-1) * ve_bi.view(b, t, self.n_head, self.head_dim)
        if ve_tri is not None:
            g_tri = 2 * torch.sigmoid(self.ve_gate_tri(x_v[..., 2 * VE_GATE_CHANNELS:3 * VE_GATE_CHANNELS]))
            v = v + g_tri.unsqueeze(-1) * ve_tri.view(b, t, self.n_head, self.head_dim)
        if USE_BF16_NORM_OUTPUT:
            v = v.to(q.dtype)
        q = rope_norm(q, cos, sin)
        k = rope_norm(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        sdpa_output_dtype = q.dtype
        if USE_BF16_SDPA_INPUT and (not USE_NKI_ATTN):
            q, k, v = (q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16))
        if attn_mask is None:
            if COMPILE_SDPA_DIRECT and (not USE_NKI_ATTN):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
            else:
                y = nki_attention(q, k, v, self.attn_scale)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[..., :t, :t], scale=self.attn_scale)
        if USE_BF16_SDPA_INPUT and (not USE_NKI_ATTN):
            y = y.to(sdpa_output_dtype)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        if self.head_gate is not None:
            yh = y.view(b, t, self.n_head, self.head_dim)
            if HEAD_GATE_NORM:
                yh = norm(yh)
            g = self.head_gate(x_q[..., self.head_gate_lo:self.head_gate_hi])
            g = HEAD_GATE_MUL * torch.sigmoid(g + self.head_gate_bias.to(g.dtype))
            y = (yh * g.unsqueeze(-1)).view(b, t, c)
        return self.c_proj(y)

def coalesced_swiglu_hidden(n_embd: int, mlp_ratio: int) -> int:
    """Parameter-matched SwiGLU hidden width: the h that makes 3*d*h == 2*d*(mlp_ratio*d), i.e.
    h = (2/3)*mlp_ratio*d (= 8/3*d at the locked mlp_ratio 4), rounded to a multiple of 128.

    Reads mlp_ratio from the CONFIG, not the global: load_for_eval rebuilds the model from the
    saved model_config, so a checkpoint written with --mlp-ratio must reproduce its own width."""
    if COALESCED_SWIGLU_HIDDEN > 0:
        return int(COALESCED_SWIGLU_HIDDEN)
    exact = 2.0 / 3.0 * mlp_ratio * n_embd
    return max(128, int(round(exact / 128.0)) * 128)

class MLP(nn.Module):

    def __init__(self, config: GPTConfig, coalesced: str='none'):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.swiglu_hidden = coalesced_swiglu_hidden(config.n_embd, config.mlp_ratio) if USE_COALESCED_SWIGLU else 0
        if self.swiglu_hidden:
            hidden = self.swiglu_hidden
        fc_out = 2 * hidden if self.swiglu_hidden else hidden
        self.c_fc = None if coalesced in ('all', 'vfc') else Linear(config.n_embd, fc_out, bias=False)
        self.c_gate = Linear(config.n_embd, hidden, bias=False) if USE_GATED_MLP else None
        self.c_proj = Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, fc: torch.Tensor | None=None) -> torch.Tensor:
        h = self.c_fc(x) if fc is None else fc
        if self.swiglu_hidden:
            gate = h[..., :self.swiglu_hidden]
            up = h[..., self.swiglu_hidden:]
            return self.c_proj(F.silu(gate) * up)
        if self.c_gate is not None:
            return self.c_proj(h * F.silu(self.c_gate(x)))
        return self.c_proj(relu_squared(h))

class Block(nn.Module):

    def __init__(self, config: GPTConfig, layer_idx: int=0):
        super().__init__()
        self.coalesced = PARALLEL_BLOCK_COALESCE if USE_PARALLEL_BLOCK else 'none'
        self.attn = CausalSelfAttention(config, layer_idx, coalesced=self.coalesced)
        self.mlp = MLP(config, coalesced=self.coalesced)
        if self.coalesced == 'all':
            self.c_in = Linear(config.n_embd, 3 * config.n_embd + config.mlp_ratio * config.n_embd, bias=False)
        elif self.coalesced == 'vfc':
            self.c_in = Linear(config.n_embd, config.n_embd + config.mlp_ratio * config.n_embd, bias=False)
        else:
            self.c_in = None
        self.resid_lambda = nn.Parameter(torch.ones(1))
        self.x0_lambda = nn.Parameter(torch.zeros(1))
        self.x0_gate_s = nn.Parameter(torch.zeros(1)) if USE_X0_GATE else None
        self.block_nudge = nn.Parameter(torch.zeros(1), requires_grad=False) if USE_BLOCK_NUDGE else None
        if USE_PARALLEL_BLOCK:
            self.par_lambda_a = nn.Parameter(torch.ones(1))
            self.par_lambda_m = nn.Parameter(torch.ones(1))
        else:
            self.par_lambda_a = self.par_lambda_m = None
        self.local_conv = CausalConv(config.n_embd, LOCAL_CONV_KERNEL) if USE_LOCAL_CONV else None
        self.scalar_gate = Linear(SELECTIVE_GATE_CHANNELS, 2, bias=False) if USE_SELECTIVE_GATE else None
        self.qk_shift_beta = nn.Parameter(torch.full((config.n_embd,), float(QK_SHIFT_BETA)), requires_grad=not QK_SHIFT_FREEZE) if USE_QK_SHIFT else None

    def forward(self, x: torch.Tensor, x0: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None, ve_bi: torch.Tensor | None=None, ve_tri: torch.Tensor | None=None) -> torch.Tensor:
        if self.scalar_gate is not None:
            g = self.scalar_gate(x[..., :SELECTIVE_GATE_CHANNELS])
            resid_lambda = self.resid_lambda * (1.0 + g[..., 0:1])
            x0_lambda = self.x0_lambda * (1.0 + g[..., 1:2])
            x = resid_lambda * x + x0_lambda * x0
        elif self.x0_gate_s is not None:
            gate0 = 2.0 * torch.sigmoid(self.x0_gate_s * x.mean(dim=-1, keepdim=True))
            x = self.resid_lambda * x + self.x0_lambda * (gate0 * x0)
        else:
            x = self.resid_lambda * x + self.x0_lambda * x0
        if self.block_nudge is not None and BLOCK_NUDGE_SITE == 'resid':
            x = x + self.block_nudge * x
        q_in = k_in = v_in = x
        if self.block_nudge is not None and BLOCK_NUDGE_SITE == 'qk':
            q_in = k_in = x + self.block_nudge * x
        if self.qk_shift_beta is not None:
            q_in = k_in = x + self.qk_shift_beta * causal_shift_1(x)
        if self.local_conv is not None:
            x = x + self.local_conv(norm(x))
        if self.par_lambda_a is not None:
            h = norm(x)
            qkv = fc = None
            if self.c_in is not None:
                b, t, c = h.shape
                nh, hd = (self.attn.n_head, self.attn.head_dim)
                proj = self.c_in(h)
                if self.coalesced == 'all':
                    q = proj[..., :c].contiguous().view(b, t, nh, hd)
                    k = proj[..., c:2 * c].contiguous().view(b, t, nh, hd)
                    v = proj[..., 2 * c:3 * c].view(b, t, nh, hd)
                    fc = proj[..., 3 * c:]
                else:
                    q = k = None
                    v = proj[..., :c].view(b, t, nh, hd)
                    fc = proj[..., c:]
                qkv = (q, k, v)
            attn_out = self.attn(h, h, h, ve, cos, sin, attn_mask, ve_bi, ve_tri, qkv=qkv)
            x = x + self.par_lambda_a * attn_out + self.par_lambda_m * self.mlp(h, fc=fc)
            return x
        if USE_QKV_NORM_CSE and q_in is k_in and (k_in is v_in):
            qkv_n = norm(q_in)
            attn_out = self.attn(qkv_n, qkv_n, qkv_n, ve, cos, sin, attn_mask, ve_bi, ve_tri)
            x = x + attn_out
        else:
            attn_out = self.attn(norm(q_in), norm(k_in), norm(v_in), ve, cos, sin, attn_mask, ve_bi, ve_tri)
            x = x + attn_out
        if USE_MLP_SANDWICH_NORM:
            x = x + norm(self.mlp(norm(x)))
        else:
            x = x + self.mlp(norm(x))
        return x

class EngramLite(nn.Module):
    """T3.2 Engram-lite: hashed n-gram memory read through a hidden-state-dependent gate.

    See the USE_ENGRAM block for the full spec, the causality argument and the step-0 no-op
    argument. This class owns everything except the hash indices themselves, which are computed
    once per forward alongside T3.1's in GPT._ngram_hash_indices (the prev/prev2 shifts are
    shared), and except the gathered rows in share mode, which GPT.forward hands over.

    `share_e_dim` is the width of the e_t that GPT.forward will supply in share mode; it is None
    in own-table mode, where this module builds and gathers its own tables.
    """

    def __init__(self, config: GPTConfig, padded_vocab_size: int, share_e_dim: int | None=None):
        super().__init__()
        self.orders = engram_orders()
        self.heads = int(ENGRAM_HEADS)
        assert self.heads >= 1, ENGRAM_HEADS
        self.share = share_e_dim is not None
        self.mem_dim = int(ENGRAM_MEM_DIM)
        self.rows = int(ENGRAM_TABLE_MULT) * padded_vocab_size
        self.mod = _largest_prime_below(self.rows)
        dtype = torch.bfloat16 if ENGRAM_BF16 else torch.float32
        self.table_keys = tuple((f'e{h}o{o}' for o in self.orders for h in range(self.heads)))
        if self.share:
            self.tables = nn.ModuleDict()
            self.e_dim = share_e_dim
        else:
            self.tables = nn.ModuleDict({k: nn.Embedding(self.rows, self.mem_dim, _weight=torch.zeros(self.rows, self.mem_dim, dtype=dtype)) for k in self.table_keys})
            self.e_dim = self.mem_dim * len(self.table_keys)
        self.key_dim = int(ENGRAM_KEY_DIM) if int(ENGRAM_KEY_DIM) > 0 else config.n_embd
        self.w_k = _rng_neutral(lambda: Linear(self.e_dim, self.key_dim, bias=False))
        self.w_v = _rng_neutral(lambda: Linear(self.e_dim, config.n_embd, bias=False))
        self.w_q = _rng_neutral(lambda: Linear(config.n_embd, self.key_dim, bias=False)) if self.key_dim != config.n_embd else None
        self.conv = CausalConv(config.n_embd, int(ENGRAM_CONV_KERNEL)) if int(ENGRAM_CONV_KERNEL) > 0 else None
        self._scale = self.key_dim ** (-0.5)

    def gather(self, ngram_idx: dict[str, torch.Tensor]) -> torch.Tensor:
        """e_t in own-table mode: concatenate this module's own table rows, heads-within-order."""
        parts = [self.tables[k].weight[ngram_idx[k]].to(COMPUTE_DTYPE) for k in self.table_keys]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Y for `H <- H + Y`. `h` is the residual at the injection site, `e` is e_t."""
        k = self.w_k(e)
        v = self.w_v(e)
        q = self.w_q(h) if self.w_q is not None else h
        alpha = torch.sigmoid((norm(q) * norm(k)).sum(dim=-1, keepdim=True) * self._scale)
        vt = alpha * v
        if self.conv is None:
            return vt
        return F.silu(self.conv(norm(vt))) + vt

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        padded_vocab_size = (config.vocab_size + 63) // 64 * 64
        self.transformer = nn.ModuleDict({'wte': nn.Embedding(padded_vocab_size, config.n_embd), 'h': nn.ModuleList([Block(config, i) for i in range(config.n_layer)])})
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, config.n_embd) for i in range(config.n_layer) if USE_VE and has_ve(i, config.n_layer)})
        self.ngram_rows = NGRAM_VE_TABLE_MULT * padded_vocab_size
        self.ngram_mod = _largest_prime_below(self.ngram_rows)
        ngram_dtype = torch.bfloat16 if NGRAM_VE_BF16 else torch.float32
        self.ngram_dim = NGRAM_VE_DIM if NGRAM_VE_DIM > 0 else config.n_embd

        def _table() -> nn.Embedding:
            return nn.Embedding(self.ngram_rows, self.ngram_dim, _weight=torch.zeros(self.ngram_rows, self.ngram_dim, dtype=ngram_dtype))
        tables: dict[str, nn.Module] = {}
        for i in range(config.n_layer):
            if has_ngram_bigram(i, config.n_layer):
                tables[f'b{i}'] = _table()
            if has_ngram_trigram(i, config.n_layer):
                tables[f't{i}'] = _table()
        self.ngram_embeds = nn.ModuleDict(tables)
        self.ngram_proj = nn.ModuleDict({k: _rng_neutral(lambda: Linear(self.ngram_dim, config.n_embd, bias=False)) for k in tables} if self.ngram_dim != config.n_embd else {})
        self.ngram_pad_id = padded_vocab_size
        self.engram_site = engram_site_index(config.n_layer) if USE_ENGRAM else -1
        self.engram_share_keys: tuple[str, ...] = ()
        if USE_ENGRAM:
            share_e_dim = None
            if ENGRAM_SHARE_NGRAM_TABLES:
                self.engram_share_keys = tuple((k for k in self.ngram_embeds if int(k[1:]) <= self.engram_site))
                assert self.engram_share_keys, f'ENGRAM_SHARE_NGRAM_TABLES needs an n-gram table at or before layer {self.engram_site}; have {list(self.ngram_embeds)}'
                share_e_dim = self.ngram_dim * len(self.engram_share_keys)
            self.engram = EngramLite(config, padded_vocab_size, share_e_dim)
        else:
            self.engram = None
        if USE_OUT_BIGRAM:
            ob_dtype = torch.bfloat16 if OUT_BIGRAM_BF16 else torch.float32
            self.out_bigram = nn.Embedding(padded_vocab_size, padded_vocab_size, _weight=torch.zeros(padded_vocab_size, padded_vocab_size, dtype=ob_dtype))
            self.out_bigram_gate = _rng_neutral(lambda: Linear(config.n_embd, 1, bias=False))
            self.out_bigram_gate_bias = nn.Parameter(torch.full((1,), float(OUT_BIGRAM_GATE_BIAS)))
        else:
            self.out_bigram = None
            self.out_bigram_gate = None
            self.out_bigram_gate_bias = None
        self.out_pool_lambdas = nn.Parameter(torch.zeros(min(OUT_POOL_LAYERS, config.n_layer))) if USE_OUT_POOL else None
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, config.n_embd // config.n_head)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
        pattern = (WINDOW_PATTERN * (config.n_layer // max(1, len(WINDOW_PATTERN)) + 1))[:config.n_layer]
        self.layer_is_short = [c == 'S' for c in pattern]
        short_window = max(1, int(config.sequence_len * SHORT_WINDOW_FRAC))
        self.register_buffer('mask_full', _build_window_mask(config.sequence_len, config.sequence_len, torch.float32), persistent=False)
        self.register_buffer('mask_short', _build_window_mask(config.sequence_len, short_window, torch.float32), persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = INIT_SCALE * math.sqrt(3.0) * self.config.n_embd ** (-0.5)
        for block in self.transformer.h:
            if block.c_in is not None:
                n = self.config.n_embd
                if block.coalesced == 'all':
                    torch.nn.init.uniform_(block.c_in.weight[:n], -s, s)
                    torch.nn.init.uniform_(block.c_in.weight[n:2 * n], -s, s)
                    torch.nn.init.uniform_(block.c_in.weight[2 * n:3 * n], -s, s)
                    fc_rows = block.c_in.weight[3 * n:]
                else:
                    torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                    torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                    torch.nn.init.uniform_(block.c_in.weight[:n], -s, s)
                    fc_rows = block.c_in.weight[n:]
            elif USE_FUSED_QKV:
                n = self.config.n_embd
                torch.nn.init.uniform_(block.attn.c_qkv.weight[:n], -s, s)
                torch.nn.init.uniform_(block.attn.c_qkv.weight[n:2 * n], -s, s)
                torch.nn.init.uniform_(block.attn.c_qkv.weight[2 * n:], -s, s)
            else:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            if ATTN_CPROJ_INIT_SCALE > 0.0:
                b = ATTN_CPROJ_INIT_SCALE * s
                torch.nn.init.uniform_(block.attn.c_proj.weight, -b, b)
            else:
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight if block.c_in is None else fc_rows, -0.4 * s, 0.4 * s)
            if block.mlp.c_gate is not None:
                torch.nn.init.uniform_(block.mlp.c_gate.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.resid_lambda is not None:
                block.resid_lambda.fill_(1.0)
                block.x0_lambda.fill_(0.05)
            if block.par_lambda_a is not None:
                block.par_lambda_a.fill_(1.0)
                block.par_lambda_m.fill_(1.0)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            for g in (block.attn.ve_gate_bi, block.attn.ve_gate_tri):
                if g is not None:
                    torch.nn.init.zeros_(g.weight)
            if block.scalar_gate is not None:
                torch.nn.init.zeros_(block.scalar_gate.weight)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for tbl in self.ngram_embeds.values():
            torch.nn.init.zeros_(tbl.weight)
        for proj in self.ngram_proj.values():
            torch.nn.init.uniform_(proj.weight, -s, s)
        if self.engram is not None:
            for tbl in self.engram.tables.values():
                torch.nn.init.zeros_(tbl.weight)
            for lin in (self.engram.w_k, self.engram.w_v, self.engram.w_q):
                if lin is not None:
                    torch.nn.init.uniform_(lin.weight, -s, s)
            if self.engram.conv is not None:
                torch.nn.init.zeros_(self.engram.conv.weight)
        if self.out_bigram is not None:
            torch.nn.init.zeros_(self.out_bigram.weight)
            torch.nn.init.zeros_(self.out_bigram_gate.weight)
        if self.out_pool_lambdas is not None:
            torch.nn.init.zeros_(self.out_pool_lambdas)
        for block in self.transformer.h:
            if block.attn is not None and block.attn.head_gate is not None:
                torch.nn.init.zeros_(block.attn.head_gate.weight)
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, self.config.n_embd // self.config.n_head)
        self.cos.copy_(cos)
        self.sin.copy_(sin)
        if USE_TRANSPOSED_LINEAR:
            n_t = transpose_linear_weights(self)
            print0(f'T1.6: transposed {n_t} Linear weights to [in, out] layout')

    def _ngram_hash_indices(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """T3.1: per-table gather indices for the hashed n-gram value embeddings.

        STRICTLY CAUSAL: index[b, t] is a function of idx[b, t], idx[b, t-1] and (for the
        trigram tables) idx[b, t-2] only, so no position can ever read a later one --
        which is why causality_check stays at exactly 0.0 with this on.

        DETERMINISTIC: a pure function of the token ids and of compile-time prime
        constants (NGRAM_HASH_PRIMES, built once at import by a sieve, and self.ngram_mod).
        No RNG, no dependence on dict/iteration order, no per-step-changing scalar -- the
        same token pair maps to the same row on every step, every rank and every process.

        Computed on DEVICE rather than on the host in fetch(): every operand is int32 and
        every op (mul by a compile-time constant, bitwise_xor, remainder) is elementwise on
        a [b, T] tensor, i.e. ~5 tiny launches per table against the 168 MB gather the row
        lookup itself costs. The plan's host-side fallback (numpy int64 in fetch() plus an
        extra H2D copy per micro-step) exists if these int ops ever stop compiling, but it
        also has to thread an extra tensor through forward()/eval/load_for_eval, so it is
        only worth paying for if measured to be necessary.
        """
        cur = idx.to(torch.int32)
        pad = torch.full_like(cur[:, :1], self.ngram_pad_id)
        prev = torch.cat((pad, cur[:, :-1]), dim=1)
        need_trigram = any((k.startswith('t') for k in self.ngram_embeds))
        engram_keys = tuple(self.engram.tables) if self.engram is not None else ()
        need_trigram = need_trigram or any((k.endswith('o3') for k in engram_keys))
        prev2 = torch.cat((pad, pad, cur[:, :-2]), dim=1) if need_trigram else None
        out: dict[str, torch.Tensor] = {}
        for key in engram_keys:
            head_idx, order = (int(key[1:key.index('o')]), int(key[key.index('o') + 1:]))
            primes = engram_hash_primes(head_idx, order)
            if order == 2:
                h = torch.bitwise_xor(prev * primes[0], cur * primes[1])
            else:
                h = torch.bitwise_xor(torch.bitwise_xor(prev2 * primes[0], prev * primes[1]), cur * primes[2])
            out[key] = torch.remainder(h, self.engram.mod)
        for key in self.ngram_embeds:
            layer = int(key[1:])
            if key[0] == 'b':
                p_prev, p_cur = ngram_bigram_primes(layer)
                h = torch.bitwise_xor(prev * p_prev, cur * p_cur)
            else:
                p_prev2, p_prev, p_cur = ngram_trigram_primes(layer)
                h = torch.bitwise_xor(torch.bitwise_xor(prev2 * p_prev2, prev * p_prev), cur * p_cur)
            out[key] = torch.remainder(h, self.ngram_mod)
        return out

    def _ngram_lookup(self, key: str, rows: torch.Tensor, raw_out: dict[str, torch.Tensor] | None=None) -> torch.Tensor:
        """One table's contribution to v, at full n_embd width, in COMPUTE_DTYPE.

        With NGRAM_VE_DIM=0 (default) this is the plain gather. With a narrow table it is
        the gather followed by that table's own up-projection -- kept in one place so the
        forward loop reads the same either way and the two paths cannot drift apart.

        NGRAM_VE_INDEX_GATHER picks HOW the gather is spelled. The two spellings are
        numerically identical (both are a row gather whose backward scatter-adds into a
        dense [rows, dim] grad), but they lower to completely different backward kernels on
        this stack, and the difference decides whether T3.1 is affordable at all
        (research/run-logs/embed-gather-bench.log, best-of-5 fwd+bwd, b=4 t=2048):

            rows    dim   nn.Embedding   index_select   weight[idx]
             8192   640        3.18ms         5.12ms       5.20ms
            65536   640       19.39ms         5.20ms       5.21ms
            65536   128       19.09ms         1.88ms       1.80ms
           262144   128       72.49ms        72.43ms       2.21ms

        The forward is ~0.25ms at EVERY shape, so `aten::embedding_dense_backward` is the
        entire cost, and it is linear in the table's ROW COUNT while being independent of
        its width -- i.e. it touches the whole table, not just the rows that were read.
        Advanced indexing is the only spelling that stays cheap at every shape.
        """
        tbl = self.ngram_embeds[key].weight
        out = (tbl[rows] if NGRAM_VE_INDEX_GATHER else self.ngram_embeds[key](rows)).to(COMPUTE_DTYPE)
        if raw_out is not None:
            raw_out[key] = out
        if key in self.ngram_proj:
            out = self.ngram_proj[key](out)
        return out

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int=ROPE_BASE):
        device = self.transformer.wte.weight.device
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / base ** (freqs / head_dim)
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        table = torch.outer(t, inv_freq)
        cos = table.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = table.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        return (cos, sin)

    def get_device(self) -> torch.device:
        return self.transformer.wte.weight.device

    def estimate_flops(self) -> int:
        """Estimate FLOPs per token (forward + backward). Attention QK^T and AV are O(T) per token."""
        nparams = sum((p.numel() for p in self.parameters()))
        non_matmul = self.transformer.wte.weight.numel() + 2 * self.config.n_layer
        non_matmul += sum((p.numel() for p in self.ngram_embeds.parameters()))
        if self.engram is not None:
            non_matmul += sum((p.numel() for p in self.engram.tables.parameters()))
        if self.out_bigram is not None:
            non_matmul += self.out_bigram.weight.numel()
        h = self.config.n_head
        d = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = self.config.n_layer * 12 * h * d * t
        return 6 * (nparams - non_matmul) + attn_flops

    def setup_optimizer(self):
        scalar_params = []
        matrix_params = []
        local_conv_params = []
        gate_params = []
        for block in self.transformer.h:
            if block.local_conv is not None:
                local_conv_params.append(block.local_conv.weight)
            if block.resid_lambda is not None:
                scalar_params.extend([block.resid_lambda, block.x0_lambda])
            if block.x0_gate_s is not None:
                scalar_params.append(block.x0_gate_s)
            if block.par_lambda_a is not None:
                scalar_params.extend([block.par_lambda_a, block.par_lambda_m])
            if block.qk_shift_beta is not None and block.qk_shift_beta.requires_grad:
                gate_params.append(block.qk_shift_beta)
            if block.attn.head_gate is not None:
                gate_params.append(block.attn.head_gate_bias)
            matrix_params.extend((p for name, p in block.attn.named_parameters() if name != 'head_gate_bias'))
            matrix_params.extend(block.mlp.parameters())
            if block.c_in is not None:
                matrix_params.append(block.c_in.weight)
        matrix_params.extend((p.weight for p in self.ngram_proj.values()))
        engram_conv_params = []
        if self.engram is not None:
            matrix_params.extend((lin.weight for lin in (self.engram.w_k, self.engram.w_v, self.engram.w_q) if lin is not None))
            if self.engram.conv is not None:
                engram_conv_params.append(self.engram.conv.weight)
        if self.out_pool_lambdas is not None:
            scalar_params.append(self.out_pool_lambdas)
        if self.out_bigram is not None:
            gate_params.extend([self.out_bigram_gate.weight, self.out_bigram_gate_bias])
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
            param_groups.append(dict(kind='adamw', params=value_embed_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, EMBED_ADAMW_BETA2), eps=1e-10, weight_decay=VE_WD))
        ngram_params = list(self.ngram_embeds.parameters())
        if self.engram is not None:
            ngram_params.extend(self.engram.tables.parameters())
        if ngram_params:
            ngram_lr = (NGRAM_VE_LR if NGRAM_VE_LR > 0.0 else EMBEDDING_LR) * dmodel_lr_scale
            rmsprop = NGRAM_VE_OPT == 'rmsprop'
            param_groups.append(dict(kind='rmsprop' if rmsprop else 'adamw', params=ngram_params, lr=ngram_lr, betas=(0.0, NGRAM_VE_BETA2) if rmsprop else (0.8, EMBED_ADAMW_BETA2), eps=1e-10, weight_decay=0.0, no_lr_decay=NGRAM_VE_FLAT_LR))
            if rmsprop:
                table_dtypes = sorted({str(p.dtype) for p in ngram_params})
                math_dtypes = ['torch.float32'] if NGRAM_RMS_FP32 else table_dtypes
                print0(f'ngram_rmsprop: fp32_state_math={NGRAM_RMS_FP32} nki={USE_NKI_NGRAM_RMS} parameter_gradient_dtypes={table_dtypes} state_math_dtypes={math_dtypes}')
        if self.out_bigram is not None:
            param_groups.append(dict(kind='adamw', params=[self.out_bigram.weight], lr=(OUT_BIGRAM_LR if OUT_BIGRAM_LR > 0.0 else UNEMBEDDING_LR) * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.0))
        if local_conv_params and TRAIN_LOCAL_CONV:
            param_groups.append(dict(kind='adamw', params=local_conv_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if engram_conv_params:
            param_groups.append(dict(kind='adamw', params=engram_conv_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        transposed_ids = {id(m.weight) for m in self.modules() if isinstance(m, Linear) and m.weight_transposed}
        depth_of: dict[int, int] = {}
        for _i, _block in enumerate(self.transformer.h):
            for _p in _block.parameters():
                depth_of[id(_p)] = _i
        depth_on = any(muon_depth_enabled())
        for shape, transposed in sorted({(p.shape, id(p) in transposed_ids) for p in matrix_params}, key=lambda st: (st[0], st[1])):
            params = [p for p in matrix_params if p.shape == shape and (id(p) in transposed_ids) == transposed]
            group = dict(kind='muon', params=params, lr=MATRIX_LR, momentum=0.95, beta2=MUON_BETA2, ns_steps=MUON_NS_STEPS, weight_decay=WEIGHT_DECAY, transposed=transposed)
            if depth_on:
                group['depth_mults'] = muon_depth_mults([depth_of.get(id(p)) for p in params], self.config.n_layer)
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
        cos, sin = (self.cos[:, :t], self.sin[:, :t])
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
        doc_mask = doc_boundary_mask(idx) if USE_DOC_MASK else None
        ngram_idx = self._ngram_hash_indices(idx) if len(self.ngram_embeds) or self.engram is not None else None
        engram_raw: dict[str, torch.Tensor] | None = {} if self.engram is not None and self.engram.share else None
        pool_from = self.config.n_layer - self.out_pool_lambdas.numel() if self.out_pool_lambdas is not None else -1
        pool_buf: list[torch.Tensor] = []
        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](idx).to(COMPUTE_DTYPE) if str(i) in self.value_embeds else None
            ve_bi = ve_tri = None
            if ngram_idx is not None:
                if f'b{i}' in self.ngram_embeds:
                    ve_bi = self._ngram_lookup(f'b{i}', ngram_idx[f'b{i}'], engram_raw)
                if f't{i}' in self.ngram_embeds:
                    ve_tri = self._ngram_lookup(f't{i}', ngram_idx[f't{i}'], engram_raw)
            if USE_CAUSAL_FASTPATH and doc_mask is None and (not self.layer_is_short[i]):
                attn_mask = None
            else:
                static_mask = (self.mask_short if self.layer_is_short[i] else self.mask_full)[..., :t, :t]
                attn_mask = static_mask + doc_mask if doc_mask is not None else static_mask
            x = block(x, x0, ve, cos, sin, attn_mask, ve_bi, ve_tri)
            if self.engram is not None and i == self.engram_site:
                if engram_raw is not None:
                    parts = [engram_raw[k] for k in self.engram_share_keys]
                    e_t = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
                else:
                    e_t = self.engram.gather(ngram_idx)
                x = x + self.engram(x, e_t)
            if i >= pool_from:
                pool_buf.append(x)
        if self.out_pool_lambdas is not None:
            for k, h in enumerate(pool_buf):
                x = x + self.out_pool_lambdas[k] * h
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        if self.out_bigram is not None:
            g = torch.sigmoid(self.out_bigram_gate(x) + self.out_bigram_gate_bias.to(x.dtype))
            row = self.out_bigram.weight[idx][..., :self.config.vocab_size]
            logits = logits + (g * row).float()
        if targets is None:
            return LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        return softcap_cross_entropy(logits, targets, loss_reduction)

def build_config(depth: int | None=None) -> GPTConfig:
    """T0.2: depth and width are now independently settable.

    `depth=None` reads the module-level DEPTH *at call time* -- deliberately not
    `depth: int = DEPTH`, which would freeze the pre-flag value at def time and make
    `--depth` silently inert (the same class of bug as the NUM_STEPS default).
    N_EMBD == 0 reproduces the historical derivation exactly.
    """
    depth = DEPTH if depth is None else depth
    if N_EMBD > 0:
        n_embd = N_EMBD
        if n_embd % HEAD_DIM != 0:
            raise ValueError(f'N_EMBD={N_EMBD} must be a multiple of HEAD_DIM={HEAD_DIM}')
    else:
        base_dim = depth * ASPECT_RATIO
        n_embd = (base_dim + HEAD_DIM - 1) // HEAD_DIM * HEAD_DIM
    n_head = n_embd // HEAD_DIM
    if N_KV_HEADS < 0 or N_KV_HEADS > n_head:
        raise ValueError(f'N_KV_HEADS={N_KV_HEADS} must be 0 (dense) or in [1, n_head={n_head}]')
    if N_KV_HEADS > 0 and n_head % N_KV_HEADS != 0:
        raise ValueError(f'N_KV_HEADS={N_KV_HEADS} must divide n_head={n_head} (locked width: n_head=5, so only 1/MQA or 5/dense)')
    return GPTConfig(sequence_len=SEQ_LEN, vocab_size=TOKENIZER_VOCAB_SIZE, n_layer=depth, n_head=n_head, n_embd=n_embd, mlp_ratio=MLP_RATIO, attn_scale=ATTN_SCALE, n_kv_heads=N_KV_HEADS)
POLAR_EXPRESS_COEFFS = [(8.156554524902461, -22.48329292557795, 15.878769915207462), (4.042929935166739, -2.808917465908714, 0.5000178451051316), (3.8916678022926607, -2.772484153217685, 0.5060648178503393), (3.285753657755655, -2.3681294933425376, 0.46449024233003106), (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)]
ADAM_LIKE_KINDS = ('adamw', 'rmsprop')
USE_COMPILE_MUON = True
USE_MUON_LR_FP32 = False
_MUON_COMPILED: dict[tuple, object] = {}

def _muon_math(w, g, momentum_buffer, second, mom, one_mom, lr, lrwd, tall: bool, ns_steps: int, red_dim: int, red_dim_size: int, one_beta2, to_bf16: bool, lr_fp32: bool=False):
    """Muon's per-group math body, as a pure tensors-in/tensors-out function.

    This is the ONLY form of the Muon step that `torch.compile(backend="neuron")` can
    take, and every deviation from `_muon_step`'s eager body is forced by that:

    * **No `torch._foreach_*` anywhere.** `torch._foreach_copy_` is what blocked T1.7:
      the neuron backend fails Torch-MLIR legalization on it
      ("failed to legalize operation 'torch.operator'"). The stacked read-in and the
      write-back to the live parameters keep using `_foreach_copy_`, but they stay
      OUTSIDE this function, in eager, where foreach is legal and cheap (1 launch a side).
    * **No in-place ops on graph inputs.** `lerp` not `lerp_`, and `w` is returned
      rather than mutated. That costs nothing: `stacked_params` is already scratch that
      `_muon_step` refills from the live params every step, and `momentum_buffer` /
      `second_momentum_buffer` are rebound in the state dict, which is a Python-level
      assignment with no device work (the same trick the NKI path already uses for
      `stacked_params`).
    * **No per-step Python float reaches a device op.** `lr`, `lr*wd`, `momentum` and
      `1-momentum` come in as 0-dim fp32 DEVICE tensors, because under
      `dynamic=False` a Python float argument is a Dynamo guard: the LR schedule alone
      would force a fresh neuronx-cc compile on every single step. This is the
      USE_TENSOR_LR_SCALARS lesson applied to a compiled region, where it is much worse
      than in eager -- eager only bakes `alpha=`-style scalars, Dynamo bakes all of them.
      `tall` / `ns_steps` / `red_dim` / `red_dim_size` / `to_bf16` / `lr_fp32` ARE
      Python constants on purpose: they are fixed per group for the whole run, so
      guarding on them is free and it keeps the traced graph straight-line.
      `one_beta2` is in that class only while the plan-§11.1 NorMuon ramp is OFF, which is
      the default; when MUON_BETA2_FINAL > 0 it changes every step and arrives as a 0-dim
      fp32 device tensor instead, for exactly the reason `lr` does. `torch.lerp` takes
      either overload, so the body is written once. Under plan-§11.1 per-depth scaling
      `mom` / `one_mom` / `lr` / `lrwd` arrive as [K, 1, 1] instead of 0-dim and broadcast
      over the stacked [K, rows, cols] buffers; nothing in this body needs to know.

    Bit-exactness against the eager body is the standard, and these are the three places
    it could have been lost:

    1. `1-momentum` is formed on the HOST in float64 and uploaded, not computed on
       device from an fp32 `momentum`. `fp32(1.0 - 0.95)` in float64-then-round is
       0.05f, while `1.0f - 0.95f` on device is 0.0500000119 -- a real difference.
       `_muon_compile_scalars` does the float64 subtraction, exactly like the Python
       float the eager path hands to `lerp_.Scalar`.
    2. `lr*ratio` and `lr*ratio*wd` are formed on the host in float32 (NOT float64), so
       they equal the device's own fp32 products bit for bit -- the identical argument
       `_muon_lrwd` already makes for the NKI path.
    3. `lerp`'s weight arrives as a 0-dim tensor here and as a Python scalar in eager,
       i.e. the `lerp.Tensor` overload versus `lerp.Scalar`. Verified equal, not
       assumed: research/scratch/compile_optim_equiv.py compares both overloads
       directly and then compares 6 full optimizer steps element by element.
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
    """`lr * g`, computed the way a 0-dim `lr` computes it whatever `lr`'s shape.

    Plan §11.1 item 3 makes `lr` a [K, 1, 1] operand instead of a 0-dim one, and that flips
    PyTorch's type promotion. `g` is bf16 here (the Newton-Schulz iterate) while `lr` is fp32:

      * 0-dim `lr`  -> same-category, so the 0-dim operand's dtype loses. Common dtype is bf16, and
        TensorIterator CASTS lr to bf16 before multiplying. The product is bf16 of a bf16 lr.
      * [K,1,1] `lr` -> both dimensioned, so common dtype is fp32: lr keeps its full precision and
        the product comes out fp32.

    Left alone, the per-depth arm would therefore also carry a silent precision UPGRADE of the
    weight update -- roughly 0.4% of relative lr accuracy recovered -- riding along with the change
    being measured, on the arm least able to afford a confound. Measured, not theorised:
    research/scratch/sched_tricks_probe.py intercepts `_muon_math` and shows every scalar operand
    bit-identical and only `w_out` differing, by ~3e-5 on fp32 weights, with neutral multipliers.

    So the cast happens BEFORE the multiply, not after: rounding the fp32 product back to bf16 is
    NOT the same number, because the 0-dim path rounds lr first and the product second. This is what
    makes the neutral-multiplier case in research/scratch/sched_tricks_equiv.py bitwise.

    Recorded while here, as a separate matter: the locked path really does round lr to bf16 before
    scaling the update, i.e. every Muon group's effective lr carries up to ~0.4% quantisation error.
    It is uniform per group per step, so it acts as tiny lr jitter rather than as noise on the
    direction -- but keeping lr in fp32 is a candidate free win in its own right, and it must be its
    own flag and its own A/B, not something smuggled in under per-depth scaling.

    `lr_fp32` (USE_MUON_LR_FP32 / --muon-lr-fp32) is that flag, and it is that A/B. Default False
    reproduces the paragraph above byte for byte -- this branch is only reached when the dimensioned
    (`lr.dim() > 0`) branch above was NOT taken, which is exactly the 0-dim locked-default case, so
    it can never touch the per-depth path's intentional bf16-matching behaviour. When on, instead of
    letting the multiply's own type promotion cast the fp32 `lr` down to `g`'s dtype first, `g` is
    upcast to fp32 and the product is formed and returned in fp32. Both call sites add that product
    to another fp32 term (`lrwd * w * mask`) and subtract the fp32 sum from an fp32 `w`, so the
    lower-precision cast that used to happen to `lr` before the multiply now simply does not happen
    anywhere before the final fp32 subtract -- there is no dtype for `g.float() * lr` to be cast
    down to here, unlike the `lr.dim() > 0` branch above where the cast is the point.
    """
    if isinstance(lr, torch.Tensor) and lr.dim() > 0 and (lr.dtype != g.dtype):
        lr = lr.to(g.dtype)
        return lr * g
    if lr_fp32 and isinstance(lr, torch.Tensor) and (lr.dim() == 0) and (lr.dtype != g.dtype):
        return lr * g.float()
    return lr * g

def _cautious_weight_decay_(params, update, lr, wd) -> None:
    """T2.3: decoupled AdamW weight decay applied only where `update * p > 0`, in place.

    `update` is the un-lr-scaled Adam direction (`p` moves by `-lr*update`), one entry per param.
    Masking on that product means decay is applied only where the gradient is already carrying the
    element toward zero, so decay never fights an element the optimizer is actively driving
    outward. A positive `lr` cannot change the sign of the product, so the pre-lr-scaled `update`
    gives the same mask as the post-scaled one.

    Convention note: `_muon_math` has carried this same trick for the Muon groups all along, with
    `>=` rather than `>`. The plan's T2.3 spec writes `> 0` and that is what is implemented here;
    the two differ only where the product is EXACTLY zero (and there either `update` is zero, so
    the element is not moving, or `p` is zero, so the decay factor is moot), i.e. not measurably.

    Not a `torch._foreach_*` chain on purpose: only groups with wd != 0 reach it, which in the
    locked config is two tensors (wte, lm_head), so batching would save ~4 launches while adding a
    dependency on `_foreach_sign_`/`_foreach_clamp_min_` being present and eager-lowerable here.
    The plain loop is ~12 extra eager launches out of the optimizer step's ~400.
    """
    for p, u in zip(params, update):
        p.mul_(1 - lr * wd * (u * p > 0).to(p.dtype))

class MuonAdamW(torch.optim.Optimizer):

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._scalar_cache: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        self._ns_coef_cache: dict[torch.device, torch.Tensor] = {}

    def _device_scalar(self, group, key: str, value: float, ref: torch.Tensor) -> torch.Tensor:
        """Carry a per-step-changing hyperparameter in a persistent DEVICE tensor
        instead of a Python float. Measured on Trn2: a Python-float lr is baked into
        the Neuron graph as a constant, so a new value every step forces a fresh
        neuronx-cc compile every step -- dt 0.90s -> 2.2-4.3s for the whole LR
        cooldown, i.e. the last WARMDOWN_RATIO of every run. Writing the value into a
        pre-allocated tensor keeps the graph shape-and-constant identical forever."""
        cache = self._scalar_cache.setdefault(id(group), {})
        entry = cache.get(key)
        if entry is None:
            entry = (torch.empty((), dtype=torch.float32), torch.zeros((), dtype=torch.float32, device=ref.device))
            cache[key] = entry
        host, dev = entry
        host.fill_(float(value))
        dev.copy_(host)
        return dev

    def _ns_coef(self, ref: torch.Tensor) -> torch.Tensor:
        """The Polar-Express (a, b, c) triples as runtime [128, 1] fp32 operands.

        Shape (n_coeffs, 3, 128, 1) so `coef[i, j]` is a *contiguous* [128, 1] view --
        NKI wants contiguous operands, and a column slice of a wider tensor would not
        be. Built once per device and never touched again, so it costs zero launches
        per step, and it is the reason the Newton-Schulz kernels compile once per shape
        instead of once per coefficient value.

        Each value is stored as float32, which is exactly what eager does with these
        Python floats: multiplying a bf16 tensor by a Python scalar uses float32 as the
        opmath type, so the scalar is rounded to fp32 there too.
        """
        cached = self._ns_coef_cache.get(ref.device)
        if cached is None:
            host = torch.empty((len(POLAR_EXPRESS_COEFFS), 3, _NKI_PARTITION, 1), dtype=torch.float32)
            for i, abc in enumerate(POLAR_EXPRESS_COEFFS):
                for j, v in enumerate(abc):
                    host[i, j].fill_(v)
            cached = host.to(ref.device)
            self._ns_coef_cache[ref.device] = cached
        return cached

    def _muon_lrwd(self, group, ref: torch.Tensor, ratio: float):
        """lr and lr*wd as two persistent [128, 1] fp32 device operands, one copy_.

        Replaces `_device_scalar("lr")` for the fused path at the same cost -- one
        host->device copy per step -- while additionally absorbing the two 0-dim device
        multiplies (`lr_base * ratio` and `lr * wd`) that the eager path spends a whole
        launch on each.

        Both products are formed on the host in float32 rather than float64 so they are
        bit-identical to the device's own fp32 arithmetic: eager computes
        fp32(fp32(lr) * fp32(ratio)) because `ratio` is a Python float wrapped into the
        fp32 opmath type, and doing the same multiply on an fp32 CPU tensor reproduces
        it exactly.
        """
        cache = self._scalar_cache.setdefault(id(group), {})
        entry = cache.get('muon_lrwd')
        if entry is None:
            entry = (torch.empty((2, _NKI_PARTITION, 1), dtype=torch.float32), torch.zeros((2, _NKI_PARTITION, 1), dtype=torch.float32, device=ref.device), torch.empty((), dtype=torch.float32))
            cache['muon_lrwd'] = entry
        host, dev, tmp = entry
        tmp.fill_(float(group['lr']))
        tmp.mul_(ratio)
        host[0].fill_(float(tmp))
        tmp.mul_(float(group['weight_decay']))
        host[1].fill_(float(tmp))
        dev.copy_(host)
        return (dev[0], dev[1])

    def _muon_depth_dev(self, group, ref: torch.Tensor):
        """Plan §11.1 item 3: the group's per-parameter multipliers, built once.

        Returns `(host_lr_f32, host_mom_f64, dev_lr_f32)`, all [K, 1, 1]. [K, 1, 1] is the shape
        that broadcasts over the group's stacked [K, rows, cols] buffers, so a per-LAYER multiplier
        costs nothing structural: the (shape, transposed) grouping, the batched Newton-Schulz and
        the one-compiled-graph-per-shape all stay exactly as they are. Returns None when the
        feature is off, which is the signal every call site branches on.

        The MOMENTUM multiplier is kept in float64 on purpose. The momentum operands have to be
        derived as `1 - (1-m)*c` and `(1-m)*c`, and doing that subtraction in fp32 costs up to one
        ulp -- which would mean a multiplier of exactly 1.0 did NOT reproduce the 0-dim path
        bit-for-bit, and the neutral-multiplier equivalence check (the one thing that proves this
        plumbing correct) would fail for a reason that has nothing to do with per-depth scaling.
        In float64 it is exact: c == 1.0 gives back `fp32(1.0 - m)` and `fp32(m)`, which is
        literally what the OFF path fills. The LR multiplier needs no such care: it enters as an
        fp32 multiply by 1.0, which is exact.
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
        """Write the per-parameter (momentum, 1-momentum) pair into two fp32 [K, 1, 1] rows.

        Both products are formed in float64 and rounded once, so a multiplier of exactly 1.0 gives
        `fp32(mom)` / `fp32(1.0 - mom)` -- the same two floats the non-depth path fills, and the
        same two the eager `lerp_.Scalar` weight would have been. Clamped before the round because
        the momentum a layer ends up at must stay a valid lerp weight for any multiplier a sweep
        might try, not just the ones that happen to be safe."""
        one = (host_mom64 * (1.0 - mom)).clamp_(0.0, 1.0)
        out_one_mom.copy_(one)
        out_mom.copy_(1.0 - one)

    def _muon_depth_mom(self, group, ref: torch.Tensor):
        """Plan §11.1 item 3, eager path: (momentum, 1-momentum) per parameter as two [K, 1, 1]
        fp32 device views behind ONE host->device copy, exactly like _muon_compile_scalars does for
        the compiled path.

        Formed on the HOST rather than as `dev_mult * (1 - group["momentum"])` on device, for the
        USE_TENSOR_LR_SCALARS reason: `1 - momentum` changes every step of the warmup and the
        cooldown, and a Python float multiplied into a device tensor is the exact construct that
        gets baked into the Neuron graph as a constant. Staging it means the graph never sees it.
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
        """T1.7: (lr*ratio, lr*ratio*wd, momentum, 1-momentum, 1-beta2) as five 0-dim fp32
        device views behind ONE host->device copy of a (5,) staging tensor.

        Replaces `_device_scalar("lr")` for the compiled path at strictly lower cost --
        one copy instead of one per scalar -- and it is the reason the compiled graph has
        no per-step constants at all. Slicing `dev[i]` is a pure view: no device work, no
        launch, so four operands cost the same as one.

        The two products are formed in float32 (see `_muon_lrwd`) and `1-momentum` in
        float64 (see `_muon_math`), each matching what the eager path's arithmetic does
        at that point. This distinction is load-bearing, not stylistic.

        Plan §11.1 items 2-3 extend this without changing its cost:
          * a FIFTH slot carries `1 - beta2`, used only when MUON_BETA2_FINAL > 0. With the ramp
            off the caller keeps passing the Python float, so the compiled graph is unchanged; with
            it on the value changes every step and would otherwise be a Dynamo guard, i.e. a full
            recompile per step. Staging one extra float costs nothing -- it is the same single
            host->device copy either way.
          * with per-depth scaling on, the staging tensor is [5, K, 1, 1] instead of (5,) and the
            first four rows carry the per-parameter multipliers pre-applied. Still ONE copy, and
            the returned operands broadcast over the group's stacked [K, rows, cols] buffers.
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
        """Shared prologue for both AdamW paths: filter to params that actually have a
        grad this step, lazily create their state, advance the per-parameter step
        counter, and return the parallel lists the update needs.

        The step counter is per-parameter and NOT per-group because a parameter whose
        grad was None on some earlier step legitimately lags the rest of its group; the
        bias corrections are therefore returned as per-parameter scalar lists.
        """
        beta1, beta2 = group['betas']
        params, grads, exp_avgs, exp_avg_sqs, bias1, bias2 = ([], [], [], [], [], [])
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
        """Batched-dispatch equivalent of _adamw_step. Same update math, restructured so
        each stage is ONE torch._foreach_* call over the whole group instead of one aten
        op per parameter.

        Motivation is pure launch overhead, not arithmetic: the native TorchNeuron eager
        backend compiles one NEFF per aten op, so a K-parameter group costs ~8*K tiny
        launches in the loop form, each one paying a full host dispatch gap with every
        engine idle. A profile of the locked config measured MuonAdamW.step() at 10% of
        step time (89.6ms) across 647 launches / 171 un-fused NEFFs, 73.2% of which was
        dispatch gap.

        This is required to be BIT-IDENTICAL to _adamw_step, not merely close:
          * grad.square() is emitted as _foreach_mul(g, g) -- verified bitwise identical
            to .square() for both fp32 and bf16 (research/scratch/foreach_api_probe.py).
          * bias1/bias2 go in as ScalarLists, one entry per parameter, preserving the
            per-parameter step counter exactly (see _adamw_state).
          * the weight-decay multiply is skipped when weight_decay == 0. That is exactly,
            not approximately, a no-op: the original computes p.mul_(1 - lr*0) =
            p.mul_(1.0), and multiplying a finite IEEE754 value by exactly 1.0 (fp32,
            then rounded back to the parameter dtype) returns it unchanged. Four of the
            locked config's AdamW groups have weight_decay=0.0, so this is also where a
            good chunk of the saved launches come from.
        foreach ops require same dtype/device but NOT the same shape, so the ragged
        scalar group batches fine -- confirmed against this installed torch (2.11.0) in
        research/scratch/foreach_api_probe.py rather than assumed from a newer API.
        """
        params, grads, exp_avgs, exp_avg_sqs, bias1, bias2 = self._adamw_state(group)
        if not params:
            return
        lr = group['lr']
        if USE_TENSOR_LR_SCALARS:
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
            if USE_CAUTIOUS_WD:
                _cautious_weight_decay_(params, update, lr, wd)
            else:
                torch._foreach_mul_(params, 1 - lr * wd)
        if USE_TENSOR_LR_SCALARS:
            torch._foreach_mul_(update, lr)
            torch._foreach_sub_(params, update)
        else:
            torch._foreach_add_(params, update, alpha=-lr)

    @torch.no_grad()
    def _adamw_step(self, group):
        beta1, beta2 = group['betas']
        lr = group['lr']
        if USE_TENSOR_LR_SCALARS and group['params']:
            lr = self._device_scalar(group, 'lr', lr, group['params'][0])
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)
            bias1 = 1 - beta1 ** state['step']
            bias2 = 1 - beta2 ** state['step']
            update = exp_avg / bias1 / ((exp_avg_sq / bias2).sqrt() + group['eps'])
            if USE_CAUTIOUS_WD and group['weight_decay'] != 0:
                _cautious_weight_decay_([p], [update], lr, group['weight_decay'])
            else:
                p.mul_(1 - lr * group['weight_decay'])
            if USE_TENSOR_LR_SCALARS:
                p.sub_(update * lr)
            else:
                p.add_(update, alpha=-lr)

    def _rmsprop_state(self, group, *, cast_grads=True):
        """Prologue for the RMSProp path -- the same contract as _adamw_state, minus the
        first moment (that is the whole point of this kind) and minus its bias correction."""
        beta2 = group['betas'][1]
        params, grads, exp_avg_sqs, bias2 = ([], [], [], [])
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            state_dtype = torch.float32 if NGRAM_RMS_FP32 else p.dtype
            if not state:
                state['step'] = 0
                state['exp_avg_sq'] = torch.zeros_like(p, dtype=state_dtype)
            elif state['exp_avg_sq'].dtype != state_dtype:
                raise ValueError(f"RMSProp second-moment dtype {state['exp_avg_sq'].dtype} does not match NGRAM_RMS_FP32={NGRAM_RMS_FP32} (expected {state_dtype}). Start with fresh optimizer state or restore its original precision; load_state_dict may have cast FP32 state to the parameter dtype.")
            state['step'] += 1
            params.append(p)
            grads.append(p.grad.float() if NGRAM_RMS_FP32 and cast_grads else p.grad)
            exp_avg_sqs.append(state['exp_avg_sq'])
            bias2.append(1 - beta2 ** state['step'])
        return (params, grads, exp_avg_sqs, bias2)

    @torch.no_grad()
    def _rmsprop_step_nki(self, group):
        if not NGRAM_RMS_FP32 or not _HAS_NKI:
            raise ValueError('NKI table RMSProp requires FP32 moments and an available NKI backend')
        lanes = 2 if os.environ.get('NEURON_LOGICAL_NC_CONFIG', '1') == '2' else 1
        for p in group['params']:
            if p.grad is None:
                continue
            if p.device.type != 'neuron' or p.dtype != torch.bfloat16 or (not p.is_contiguous()) or (not p.grad.is_contiguous()) or (p.numel() % (2048 * 128 * lanes) != 0):
                raise ValueError('NKI table RMSProp requires contiguous Neuron BF16 tables/grads with element count divisible by 2048*128*LNC')
        params, grads, moments, biases = self._rmsprop_state(group, cast_grads=False)
        for p, grad, moment, bias in zip(params, grads, moments, biases):
            state = self.state[p]
            if 'nki_rms_scalars_cpu' not in state:
                state['nki_rms_scalars_cpu'] = torch.empty((128, 2), dtype=torch.float32)
                state['nki_rms_scalars_device'] = torch.empty((128, 2), dtype=torch.float32, device=p.device)
            scalars_cpu = state['nki_rms_scalars_cpu']
            scalars_cpu[:, 0].fill_(group['lr'])
            scalars_cpu[:, 1].fill_(1.0 / bias)
            scalars = state['nki_rms_scalars_device']
            scalars.copy_(scalars_cpu)
            p_view, grad_view, v_view = (x.view(-1, 2048) for x in (p, grad, moment))
            next_p, next_v = _ngram_rmsprop_fp32_nki[lanes](p_view, grad_view, v_view, scalars, group['betas'][1], group['eps'], lanes)
            p_view.copy_(next_p)
            v_view.copy_(next_v)

    @torch.no_grad()
    def _rmsprop_step(self, group):
        """T3.1: RMSProp with no first moment, for the hashed n-gram value-embedding tables.

        `update = grad / (sqrt(exp_avg_sq / bias2) + eps)`, i.e. exactly Adam with beta1 = 0.
        The bias correction is NOT optional here: classic un-corrected RMSProp starts from
        exp_avg_sq = 0, so step 1 gives `g / sqrt((1-beta2) g^2)` = 31.6 x sign(g) at
        beta2 = 0.999 -- a 31x over-sized first update into a zero-initialized table. With
        the correction the first update is exactly sign(g) * lr, like Adam's.

        weight_decay is asserted 0 rather than implemented: decaying a table whose rows
        mostly have no gradient this step just erases the rows that are not being trained.

        No NaN on an untouched row: its grad and exp_avg_sq are both exactly 0.0, and eps
        (1e-10) is representable in bf16 as well as fp32 -- bf16 shares fp32's 8-bit
        exponent -- so the denominator is 1e-10, not 0, and 0/1e-10 = 0.

        NGRAM_RMS_FP32 changes BOTH state storage and optimizer arithmetic. The state
        helper casts grads before their square, leaving the original p.grad unchanged.
        All subsequent operands are FP32 through the denominator and LR multiplication;
        subtracting that FP32 update into p rounds to p.dtype only at the final write.
        Both foreach and scalar-tensor paths use the same promotion contract.
        """
        assert group['weight_decay'] == 0.0, 'rmsprop groups are weight-decay free by design'
        if USE_NKI_NGRAM_RMS:
            self._rmsprop_step_nki(group)
            return
        params, grads, exp_avg_sqs, bias2 = self._rmsprop_state(group)
        if not params:
            return
        lr = group['lr']
        if USE_TENSOR_LR_SCALARS:
            lr = self._device_scalar(group, 'lr', lr, params[0])
        beta2 = group['betas'][1]
        if USE_FOREACH_OPTIM:
            torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), 1 - beta2)
            denom = torch._foreach_div(exp_avg_sqs, bias2)
            torch._foreach_sqrt_(denom)
            torch._foreach_add_(denom, group['eps'])
            update = torch._foreach_div(grads, denom)
            if USE_TENSOR_LR_SCALARS:
                torch._foreach_mul_(update, lr)
                torch._foreach_sub_(params, update)
            else:
                torch._foreach_add_(params, update, alpha=-lr)
        else:
            for p, grad, exp_avg_sq, b2 in zip(params, grads, exp_avg_sqs, bias2):
                exp_avg_sq.lerp_(grad.square(), 1 - beta2)
                update = grad / ((exp_avg_sq / b2).sqrt() + group['eps'])
                if USE_TENSOR_LR_SCALARS:
                    p.sub_(update * lr)
                else:
                    p.add_(update, alpha=-lr)

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
        if USE_FOREACH_OPTIM:
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
        else:
            for i, p in enumerate(params):
                if p.grad is None:
                    g[i].zero_()
                else:
                    g[i].copy_(p.grad)
                w[i].copy_(p)
        momentum_buffer = state['momentum_buffer']
        second = state['second_momentum_buffer']
        if USE_COMPILE_MUON and USE_TENSOR_LR_SCALARS and (w.dtype == torch.float32):
            ratio = max(1.0, lshape[-2] / lshape[-1]) ** 0.5
            ns_steps = min(group['ns_steps'], len(POLAR_EXPRESS_COEFFS))
            key = (g.shape[-2] > g.shape[-1], ns_steps, red_dim, tuple(g.shape))
            fn = _MUON_COMPILED.get(key)
            if fn is None:
                fn = torch.compile(_muon_math, backend='neuron', dynamic=False) if p0.device.type == 'neuron' else _muon_math
                _MUON_COMPILED[key] = fn
                print0(f'muon step: COMPILED body for {tuple(g.shape)} (tall={key[0]} ns_steps={ns_steps} red_dim={red_dim}) [specialisations so far: {len(_MUON_COMPILED)}]')
            lr_t, lrwd_t, mom_t, one_mom_t, one_b2_t = self._muon_compile_scalars(group, p0, ratio)
            one_beta2 = one_b2_t if MUON_BETA2_FINAL > 0.0 else 1 - group['beta2']
            w, momentum_buffer, second = fn(w, g, momentum_buffer, second, mom_t, one_mom_t, lr_t, lrwd_t, key[0], ns_steps, red_dim, g.size(red_dim), one_beta2, COMPUTE_DTYPE == torch.bfloat16, USE_MUON_LR_FP32)
            state['stacked_params'] = w
            state['momentum_buffer'] = momentum_buffer
            state['second_momentum_buffer'] = second
            torch._foreach_copy_(params, list(w.unbind(0)))
            return
        depth = self._muon_depth_dev(group, p0)
        use_nki = USE_NKI_MUON and _HAS_NKI and (p0.device.type == 'neuron') and USE_TENSOR_LR_SCALARS and (COMPUTE_DTYPE == torch.bfloat16) and (w.dtype == torch.float32) and (shape[-1] <= 8192) and (min(shape[-2], shape[-1]) <= 8192) and (depth is None)
        if depth is None:
            momentum_buffer.lerp_(g, 1 - group['momentum'])
            g = g.lerp_(momentum_buffer, group['momentum'])
        else:
            mom_d, one_mom_d = self._muon_depth_mom(group, p0)
            momentum_buffer.lerp_(g, one_mom_d)
            g = g.lerp_(momentum_buffer, mom_d)
        x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-06)
        if use_nki:
            coef = self._ns_coef(p0)
            tall = x.shape[-2] > x.shape[-1]
            for i in range(min(group['ns_steps'], len(POLAR_EXPRESS_COEFFS))):
                ca, cb, cc = (coef[i, 0], coef[i, 1], coef[i, 2])
                sym = x.mT @ x if tall else x @ x.mT
                k = sym.shape[-1]
                m = _muon_ns_poly(sym.reshape(-1, k), (sym @ sym).reshape(-1, k), cb, cc).view_as(sym)
                prod = x @ m if tall else m @ x
                n = x.shape[-1]
                x = _muon_ns_axpy(x.reshape(-1, n), prod.reshape(-1, n), ca).view_as(x)
        elif x.shape[-2] > x.shape[-1]:
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
        if use_nki:
            klr, klw = self._muon_lrwd(group, p0, ratio)
            n = g.shape[-1]
            w = _muon_update(g.reshape(-1, n), w.reshape(-1, n), klr, klw).view_as(w)
            state['stacked_params'] = w
        else:
            lr_base = group['lr']
            if USE_TENSOR_LR_SCALARS:
                lr_base = self._device_scalar(group, 'lr', lr_base, p0)
            lr = lr_base * ratio
            if depth is not None:
                lr = lr * depth[2]
            wd = group['weight_decay']
            mask = g * w >= 0
            w.sub_(_muon_lr_g(lr, g, USE_MUON_LR_FP32) + lr * wd * w * mask)
        torch._foreach_copy_(params, list(w.unbind(0)))

    @torch.no_grad()
    def step(self, update_adamw: bool=True):
        """`update_adamw=False` (T1.4) runs only the Muon groups this step and leaves the AdamW
        groups' gradients untouched, so they keep accumulating into the same .grad tensors."""
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                if not update_adamw:
                    continue
                if USE_FOREACH_OPTIM:
                    self._adamw_step_foreach(group)
                else:
                    self._adamw_step(group)
            elif group['kind'] == 'rmsprop':
                if not update_adamw:
                    continue
                self._rmsprop_step(group)
            elif group['kind'] == 'muon':
                self._muon_step(group)
            else:
                raise ValueError(f"unknown optimizer kind: {group['kind']}")

    @torch.no_grad()
    def zero_grad_selective(self, zero_adamw: bool=True) -> None:
        """T1.4: like `zero_grad(set_to_none=True)`, but can leave the AdamW groups' gradients in
        place so they accumulate across a skipped update. Never severs an accumulation by
        accident -- the AdamW branch is skipped entirely rather than zeroed with a different
        mechanism."""
        for group in self.param_groups:
            if group['kind'] in ADAM_LIKE_KINDS and (not zero_adamw):
                continue
            for p in group['params']:
                p.grad = None

    def adamw_param_ids(self) -> frozenset[int]:
        """T1.4: identities of every parameter in an AdamW (or RMSProp) group, so the per-step
        all_reduce can skip exactly those on a non-update step."""
        return frozenset((id(p) for g in self.param_groups if g['kind'] in ADAM_LIKE_KINDS for p in g['params']))

def split_clip_params(model: 'GPT') -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """(main, ngram) parameter lists for clip_grad_norm_ -- see NGRAM_VE_CLIP_EXCLUDE.

    GRAD_CLIP is a GLOBAL norm over every gradient at once, so folding the ~200M-parameter
    hashed n-gram tables into it would raise the total norm, shrink clip_coef, and thereby
    change the effective learning rate of every OTHER tensor in the model -- a confound with
    nothing to do with whether n-gram tables help. Splitting them off leaves the main clip
    bit-identical to the historical `clip_grad_norm_(model.parameters(), ...)` (same tensors,
    in the same order) whenever there are no n-gram tables, and still norm-bounds the tables.
    """
    ngram_ids = {id(p) for p in model.ngram_embeds.parameters()} if NGRAM_VE_CLIP_EXCLUDE else set()
    if NGRAM_VE_CLIP_EXCLUDE and model.out_bigram is not None:
        ngram_ids = ngram_ids | {id(model.out_bigram.weight)}
    if NGRAM_VE_CLIP_EXCLUDE and model.engram is not None:
        ngram_ids = ngram_ids | {id(p) for p in model.engram.tables.parameters()}
    if not ngram_ids:
        return (list(model.parameters()), [])
    main, ngram = ([], [])
    for p in model.parameters():
        (ngram if id(p) in ngram_ids else main).append(p)
    return (main, ngram)

def sync_gradients(model: nn.Module, skip_param_ids: frozenset[int] | None=None) -> None:
    """`skip_param_ids` (T1.4) defers these parameters' all_reduce to a later step. Exact, not an
    approximation: ReduceOp.AVG is linear, so averaging the sum of K locally-accumulated steps
    equals summing K averaged steps."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in model.parameters():
        if p.grad is not None and (not (skip_param_ids and id(p) in skip_param_ids)):
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

def build_flat_grad_buffer(model: nn.Module) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Returns (flat_buffer, params) and leaves every params[i].grad pointing at a
    zeroed view into flat_buffer. Call once, before the first backward() of the run."""
    params = [p for p in model.parameters() if p.requires_grad]
    device = params[0].device if params else torch.device('cpu')
    total = sum((p.numel() for p in params))
    flat = torch.zeros(total, dtype=torch.float32, device=device)
    offset = 0
    for p in params:
        n = p.numel()
        view = flat[offset:offset + n].view_as(p)
        p.grad = view
        offset += n
    assert offset == total
    return (flat, params)

def sync_gradients_flat(flat_grad_buffer: torch.Tensor) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    dist.all_reduce(flat_grad_buffer, op=dist.ReduceOp.AVG)

def clip_and_sync_gradients(model: nn.Module, clip_main_params: list[torch.nn.Parameter], clip_ngram_params: list[torch.nn.Parameter], *, max_norm: float, clip_after_reduce: bool=False, grad_flat_buffer: torch.Tensor | None=None, foreach: bool | None=None, skip_param_ids: frozenset[int] | None=None) -> None:
    """Apply the selected clipping order, preserving the separate n-gram norm.

    Call after every micro-batch has accumulated and loss scaling has been undone.
    With clip_after_reduce=True every included gradient must be averaged this step;
    deferred Adam gradients would mix local and averaged quantities in the same norm.
    """
    if clip_after_reduce and skip_param_ids:
        raise ValueError('clip-after-reduce requires all gradients to be reduced every step')
    if grad_flat_buffer is not None and clip_ngram_params:
        raise ValueError('a flat gradient buffer cannot represent separate n-gram clipping')
    if grad_flat_buffer is not None:
        if clip_after_reduce:
            sync_gradients_flat(grad_flat_buffer)
        if max_norm > 0:
            clip_grad_norm_flat_(grad_flat_buffer, max_norm)
        if not clip_after_reduce:
            sync_gradients_flat(grad_flat_buffer)
    else:
        if clip_after_reduce:
            sync_gradients(model)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(clip_main_params, max_norm, foreach=foreach)
            if clip_ngram_params:
                torch.nn.utils.clip_grad_norm_(clip_ngram_params, max_norm, foreach=foreach)
        if not clip_after_reduce:
            sync_gradients(model, skip_param_ids=skip_param_ids)

def loss_scale_transition(scale: float, clean: int, finite: bool, growth_interval: int, max_scale: float):
    """One dynamic-loss-scaling step. Pure function (CPU-tested).

    Overflow (non-finite scaled grads): halve (floor 1.0), reset streak, skip.
    Clean: bump streak; double (cap max_scale) every growth_interval steps.
    Returns (new_scale, new_clean, skip: bool).
    """
    if not finite:
        return (max(1.0, scale / 2.0), 0, True)
    clean += 1
    if clean >= growth_interval and scale < max_scale:
        return (min(max_scale, scale * 2.0), 0, False)
    return (scale, clean, False)

def grads_finite_(model: nn.Module, grad_flat_buffer: torch.Tensor | None) -> bool:
    """True iff every accumulated (still scaled) grad is finite. One scalar
    sync in the buffer path; used by dynamic loss scaling to detect overflow
    BEFORE unscale/clip (standard placement: scaled grads overflow first)."""
    if grad_flat_buffer is not None:
        return bool(torch.isfinite(grad_flat_buffer).all().item())
    for p in model.parameters():
        g = p.grad
        if g is not None and (not bool(torch.isfinite(g).all().item())):
            return False
    return True

def unscale_grads_(model: nn.Module, grad_flat_buffer: torch.Tensor | None, scale: float) -> None:
    """Divide accumulated grads by the LOSS_SCALE factor (static loss scaling).

    Call once per optimizer step, after the grad-accum loop and BEFORE clip /
    sync / step, so clipping and the update see true units. No-op (returns
    before touching anything) when scale == 1.0, which keeps the default path
    literally the old code. With the flat buffer the division is one op;
    otherwise every p.grad is divided in place (covers the main + ngram clip
    param lists, which are p.grad views).
    """
    if scale == 1.0:
        return
    if grad_flat_buffer is not None:
        grad_flat_buffer.div_(scale)
        return
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(scale)

def clip_grad_norm_flat_(flat_grad_buffer: torch.Tensor, max_norm: float) -> torch.Tensor:
    """T1.3: the flat-buffer equivalent of torch.nn.utils.clip_grad_norm_(params, max_norm) --
    one norm + one mul instead of one norm + one mul PER TENSOR. Mathematically identical to the
    default (norm_type=2, foreach) path: the global L2 norm of a concatenation of tensors equals
    the L2 norm of the concatenated flat vector, and clipping is a single elementwise scale
    applied uniformly, so grouping changes nothing about which numbers are combined -- only NNs
    where the reduction algorithm's summation order depends on how values are chunked (the
    cross-rank all_reduce, handled by sync_gradients_flat, not this local per-rank op) could ever
    perturb bit-equality; the local norm/mul here is a single-rank, single-tensor float32
    reduction verified bit-equal in research/scratch/grad_bucket_equiv.py."""
    total_norm = flat_grad_buffer.norm(2)
    clip_coef = max_norm / (total_norm + 1e-06)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    flat_grad_buffer.mul_(clip_coef_clamped)
    return total_norm
ARCH_TOGGLE_NAMES = ['PACK_FACTOR', 'INIT_SCALE', 'USE_VE', 'WINDOW_PATTERN', 'SHORT_WINDOW_FRAC', 'ROPE_BASE', 'USE_LOCAL_CONV', 'LOCAL_CONV_KERNEL', 'TRAIN_LOCAL_CONV', 'USE_GATED_MLP', 'USE_WEIGHT_CACHE', 'USE_TRANSPOSED_LINEAR', 'WARMDOWN_RATIO', 'FINAL_LR_FRAC', 'WARMUP_STEPS', 'WD_DECAY_TO_ZERO', 'MUON_MOMENTUM_WARMUP_STEPS', 'USE_DOC_MASK', 'VE_WD', 'USE_SELECTIVE_GATE', 'SELECTIVE_GATE_CHANNELS', 'USE_FUSED_QKV', 'USE_CAUSAL_FASTPATH', 'USE_TRN2_COMPILER_FLAGS', 'COMPILE_SDPA_DIRECT', 'USE_NKI_ATTN', 'USE_NKI_LOCAL_CONV', 'USE_BF16_NORM_OUTPUT', 'USE_BF16_SDPA_INPUT', 'MATRIX_LR', 'EMBEDDING_LR', 'UNEMBEDDING_LR', 'SCALAR_LR', 'WEIGHT_DECAY', 'GRAD_CLIP', 'CLIP_AFTER_REDUCE', 'LOGIT_SOFTCAP', 'LOSS_BYTE_WEIGHT', 'LOSS_SCALE', 'LOSS_SCALE_DYNAMIC', 'ATTN_CPROJ_INIT_SCALE', 'MUON_NS_STEPS', 'MUON_MOMENTUM_COOLDOWN_FRAC', 'MUON_MOMENTUM_FINAL', 'ADAM_EVERY_N', 'ADAM_EVERY_N_LR_COMP', 'DEMON_BETA1', 'DEMON_BETA1_FINAL', 'DEMON_BETA1_REF', 'DEMON_BETA1_START_FRAC', 'MUON_BETA2', 'MUON_BETA2_FINAL', 'MUON_DEPTH_LR_BOTTOM', 'MUON_DEPTH_LR_TOP', 'MUON_DEPTH_MOM_BOTTOM', 'MUON_DEPTH_MOM_TOP', 'MUON_DEPTH_MOM_REF', 'DEPTH', 'N_EMBD', 'HEAD_DIM', 'MLP_RATIO', 'ATTN_SCALE', 'N_KV_HEADS', 'USE_NGRAM_VE', 'NGRAM_VE_LAYERS', 'NGRAM_VE_TABLE_MULT', 'NGRAM_VE_TRIGRAM', 'NGRAM_VE_DIM', 'NGRAM_VE_BF16', 'NGRAM_RMS_FP32', 'USE_NKI_NGRAM_RMS', 'NGRAM_VE_LR', 'NGRAM_VE_BETA2', 'NGRAM_VE_FLAT_LR', 'NGRAM_VE_OPT', 'NGRAM_VE_CLIP_EXCLUDE', 'USE_OUT_BIGRAM', 'OUT_BIGRAM_BF16', 'OUT_BIGRAM_GATE_BIAS', 'OUT_BIGRAM_LR', 'RELU2_TAU', 'USE_MLP_SANDWICH_NORM', 'USE_HEAD_GATE', 'HEAD_GATE_NORM', 'HEAD_GATE_CHANNELS', 'HEAD_GATE_MUL', 'HEAD_GATE_BIAS', 'USE_OUT_POOL', 'OUT_POOL_LAYERS', 'USE_QK_SHIFT', 'QK_SHIFT_BETA', 'QK_SHIFT_FREEZE', 'USE_X0_GATE', 'USE_BLOCK_NUDGE', 'BLOCK_NUDGE_SITE', 'USE_QKV_NORM_CSE', 'USE_PARALLEL_BLOCK', 'PARALLEL_BLOCK_COALESCE', 'USE_ENGRAM', 'ENGRAM_SITE', 'ENGRAM_ORDERS', 'ENGRAM_HEADS', 'ENGRAM_TABLE_MULT', 'ENGRAM_MEM_DIM', 'ENGRAM_KEY_DIM', 'ENGRAM_CONV_KERNEL', 'ENGRAM_SHARE_NGRAM_TABLES', 'ENGRAM_BF16', 'USE_EPOCH_SHUFFLE', 'EPOCH_SHUFFLE_SEED', 'BATCH_RAMP', 'BATCH_RAMP_GRAD_ACCUM', 'BATCH_RAMP_BOUNDS', 'BATCH_RAMP_LR_EXP', 'USE_WEIGHT_EMA', 'WEIGHT_EMA_DECAYS', 'WEIGHT_EMA_START_FRAC', 'USE_BYTE_WTE_INIT', 'BYTE_WTE_INIT_MIX', 'BYTE_WTE_INIT_NGRAM', 'BYTE_WTE_INIT_STD', 'USE_COALESCED_SWIGLU', 'COALESCED_SWIGLU_HIDDEN', 'USE_CAUTIOUS_WD', 'SCALAR_BETA1', 'SCALAR_BETA2']
SHAPE_CRITICAL_TOGGLES = {'USE_VE', 'USE_LOCAL_CONV', 'USE_GATED_MLP', 'USE_SELECTIVE_GATE', 'USE_FUSED_QKV', 'USE_TRANSPOSED_LINEAR', 'USE_NGRAM_VE', 'NGRAM_VE_TRIGRAM', 'NGRAM_VE_TABLE_MULT', 'NGRAM_VE_DIM', 'USE_OUT_BIGRAM', 'USE_HEAD_GATE', 'HEAD_GATE_CHANNELS', 'USE_OUT_POOL', 'OUT_POOL_LAYERS', 'USE_QK_SHIFT', 'USE_X0_GATE', 'USE_BLOCK_NUDGE', 'USE_PARALLEL_BLOCK', 'PARALLEL_BLOCK_COALESCE', 'USE_ENGRAM', 'ENGRAM_ORDERS', 'ENGRAM_HEADS', 'ENGRAM_TABLE_MULT', 'ENGRAM_MEM_DIM', 'ENGRAM_KEY_DIM', 'ENGRAM_CONV_KERNEL', 'ENGRAM_SHARE_NGRAM_TABLES', 'N_KV_HEADS', 'USE_COALESCED_SWIGLU', 'COALESCED_SWIGLU_HIDDEN'}

def current_toggles() -> dict:
    g = globals()
    return {name: g[name] for name in ARCH_TOGGLE_NAMES}
_EVAL_TOGGLE_DEFAULTS = current_toggles()

def restore_eval_toggles(payload: dict) -> None:
    """Restore only known model/training settings from a weights-only checkpoint.

    Architecture and arithmetic flags are part of the frozen model definition;
    the standalone scorer supplies no training CLI arguments. Unknown payload
    keys cannot replace arbitrary module globals. No data-dependent adaptation
    or evaluation state is introduced.
    """
    saved = payload.get('toggles', {})
    if not isinstance(saved, dict):
        raise ValueError('checkpoint toggles must be a dictionary')
    for name in ARCH_TOGGLE_NAMES:
        globals()[name] = saved.get(name, _EVAL_TOGGLE_DEFAULTS[name])

def save_checkpoint(path: Path, model: GPT, meta: dict, ema_state: dict | None=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'model_config': asdict(model.config), 'toggles': current_toggles(), 'meta': meta}
    if ema_state is not None:
        payload['ema_state'] = {k: v.detach().cpu() for k, v in ema_state.items()}
    torch.save(payload, path)

def warm_start_from(path: str, model: GPT) -> dict:
    """Load a checkpoint's weights into a freshly-built model (fresh optimizer, fresh step=0).
    Not a bit-exact resume -- no optimizer/RNG/dataloader state. For dev-iteration only; never
    for a scored submission (each submission must be a fresh run from scratch, per the rules).
    """
    payload = torch_load_weights(path, map_location='cpu')
    saved_toggles = payload.get('toggles', {})
    live_toggles = current_toggles()
    mismatches = {k: (saved_toggles[k], live_toggles[k]) for k in saved_toggles if k in live_toggles and saved_toggles[k] != live_toggles[k]}
    if mismatches:
        for name, (saved, live) in mismatches.items():
            tag = 'SHAPE-CRITICAL' if name in SHAPE_CRITICAL_TOGGLES else 'silent-behavior'
            print0(f'WARNING [{tag}]: toggle {name} mismatch -- checkpoint={saved!r} current={live!r}')
    model.load_state_dict(payload['model_state'], strict=True)
    meta = payload.get('meta', {})
    print0(f"warm-started from {path}: checkpoint step={meta.get('step')} total_train_tokens={meta.get('total_train_tokens')}")
    return meta

def torch_load_weights(path: str | Path, map_location):
    return torch.load(path, map_location=map_location, weights_only=True)
EVAL_MIN_SEQ_LEN = 8192

def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Build+load a model for eval-only use, robust to any probing seq_len up to EVAL_MIN_SEQ_LEN.

    prepare.py's eval-public (and the official scorer, whose context length is not published --
    see C1 in the research plan) call this with a fixed 2-argument signature and then run
    model(x) at whatever --seq-len they choose, which may exceed the checkpoint's own trained
    SEQ_LEN. config.sequence_len only controls three things here: the size of the cos/sin
    rotary tables, the size of mask_full/mask_short, and the `t <= config.sequence_len` assert in
    GPT.forward -- none of which are trained parameters, and cos/sin/mask_full/mask_short are all
    registered with persistent=False, so they are never part of the saved state_dict and widening
    sequence_len before construction cannot break strict-loading or change a single weight shape.
    Zero training-cost change: this function is eval-only, never called from the training loop.
    """
    global COMPUTE_DTYPE, BOS_TOKEN_ID
    COMPUTE_DTYPE = compute_dtype_for(device)
    payload = torch_load_weights(checkpoint_path, map_location='cpu')
    restore_eval_toggles(payload)
    if USE_DOC_MASK:
        BOS_TOKEN_ID = ensure_tokenizer(build_if_missing=False).get_bos_token_id()
    config = GPTConfig(**payload['model_config'])
    if config.sequence_len < EVAL_MIN_SEQ_LEN:
        config.sequence_len = EVAL_MIN_SEQ_LEN
    model = _SubmissionEvalGPT(config)
    model.init_weights()
    state = payload['model_state']
    if EVAL_PREFER_EMA and 'ema_state' in payload:
        state = payload['ema_state']
        print(f"load_for_eval: using EMA weights from {checkpoint_path} (raw weights still available under 'model_state')")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model

def lr_multiplier(step: int, elapsed: float=0.0, max_train_seconds: float=1.0) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    if WARMDOWN_RATIO <= 0.0:
        return 1.0
    progress = min(1.0, elapsed / max(1e-06, max_train_seconds))
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    lrm = cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
    if LRM_QUANTUM > 0:
        lrm = round(lrm * LRM_QUANTUM) / LRM_QUANTUM
    return lrm

def batch_ramp_stage(progress: float) -> int:
    """Which T2.1 stage the run is in, from the wall-clock fraction. 0 = A, 1 = B, 2 = C.

    Separate from batch_ramp_grad_accum() only so the stage index can be logged and asserted on
    without duplicating the boundary comparison.
    """
    lo, hi = BATCH_RAMP_BOUNDS
    if progress < lo:
        return 0
    if progress < hi:
        return 1
    return 2

def batch_ramp_grad_accum(progress: float, grad_accum_ref: int) -> tuple[int, float]:
    """(micro-batches to accumulate, lr scale) for this point in the budget.

    With BATCH_RAMP off this is `(grad_accum_ref, 1.0)` for every progress value, which is what
    makes the default path bit-identical to the locked baseline rather than merely equivalent.

    The lr scale is (B/B_ref)**BATCH_RAMP_LR_EXP and, because the micro shape is fixed, B/B_ref is
    exactly grad_accum/grad_accum_ref -- no token arithmetic needed, and no dependence on
    world_size or PACK_FACTOR that could silently drift from the loop's own accounting.
    """
    if not BATCH_RAMP:
        return (grad_accum_ref, 1.0)
    gas = BATCH_RAMP_GRAD_ACCUM[batch_ramp_stage(progress)]
    return (gas, (gas / grad_accum_ref) ** BATCH_RAMP_LR_EXP)

def weight_ema_state(model: GPT, names: list[str], shadow: list[torch.Tensor]) -> dict:
    """A full state_dict with the EMA copy substituted for every PARAMETER.

    Built from model.state_dict() rather than from `shadow` alone so the result has exactly the
    same key set as "model_state" and still load_state_dict(strict=True)s into the same model.
    Non-parameter entries (any persistent buffer) are carried over verbatim -- they are not
    trained, so averaging them would be meaningless, and dropping them would break strict load.
    """
    state = dict(model.state_dict())
    by_name = dict(zip(names, shadow))
    missing = [n for n in by_name if n not in state]
    if missing:
        raise KeyError(f'EMA tracks parameters absent from state_dict: {missing}')
    for n, t in by_name.items():
        state[n] = t
    return state

def muon_momentum(step: int, progress: float=0.0) -> float:
    """Muon momentum: 0.85 -> 0.95 over MUON_MOMENTUM_WARMUP_STEPS, then optionally back down.

    `progress` is elapsed/max_train_seconds (the same 0..1 wall-clock fraction wd_progress uses).
    The cooldown is off by default (MUON_MOMENTUM_COOLDOWN_FRAC == 0.0), in which case this is
    bit-identical to the original two-line warmup.

    Both branches return a plain Python float, which is safe here for a reason worth recording:
    the warmup already feeds 300 DISTINCT float momentum values into `momentum_buffer.lerp_(g, 1 -
    momentum)` on every historical run, and the logged dt is flat at ~0.86s from step 2 onwards --
    so `lerp_.Scalar`'s weight is NOT baked into the Neuron graph as a constant the way `alpha=`
    and a float `lr` were (the USE_TENSOR_LR_SCALARS lesson). A per-step-changing momentum
    therefore does not need the _device_scalar treatment. Verified again by watching dt across the
    cooldown window of the A/B run itself, not just inferred from history.
    """
    frac = min(step / MUON_MOMENTUM_WARMUP_STEPS, 1.0)
    m = (1 - frac) * 0.85 + frac * 0.95
    if MUON_MOMENTUM_COOLDOWN_FRAC > 0.0:
        cd = min(1.0, max(0.0, (progress - (1.0 - MUON_MOMENTUM_COOLDOWN_FRAC)) / MUON_MOMENTUM_COOLDOWN_FRAC))
        m = (1 - cd) * m + cd * MUON_MOMENTUM_FINAL
    return m

def demon_beta1_scale(progress: float) -> float:
    """Plan §11.1 item 1: the multiplicative shrink applied to every AdamW group's INITIAL beta1.

    `progress` is elapsed/max_train_seconds, the same 0..1 wall-clock fraction the LR schedule,
    wd_progress and muon_momentum's cooldown all key on. Returns exactly 1.0 when DEMON_BETA1 is
    off, so the caller's `betas` tuple is left untouched and the AdamW step sees the same floats it
    always has.

    Linear in progress across [start, 1], where start defaults to the LR warmdown's own start
    (1 - WARMDOWN_RATIO). Recursive quotes "beta1 0.8 -> 0.55 across its warmdown"; the shrink form
    means a group already at 0.8 lands exactly on DEMON_BETA1_FINAL while the 0.9 groups (scalars,
    gates, the conv groups) move proportionally rather than being left out or over-shrunk.

    NOT routed through _device_scalar, and that is a measured call rather than an oversight: the
    only place beta1 reaches a device op in the AdamW path is `_foreach_lerp_`'s / `lerp_`'s weight
    (`1 - beta1`) and the `_foreach_div` bias-correction ScalarLists, and BOTH already take
    per-step-CHANGING values on every historical run -- the 300-step momentum warmup feeds 300
    distinct `lerp_.Scalar` weights with dt flat at ~0.86s, and `1 - beta**step` is a different
    float every step until it saturates. The ops that do bake are `add_(alpha=)` and
    `mul_(1 - lr*wd)`, neither of which touches beta1. Confirmed by watching dt over a device smoke
    with the schedule on, not inferred: see research/scratch/schedule_tricks_smoke.sh.
    """
    if not DEMON_BETA1:
        return 1.0
    start = DEMON_BETA1_START_FRAC if DEMON_BETA1_START_FRAC >= 0.0 else 1.0 - WARMDOWN_RATIO
    start = min(max(start, 0.0), 0.999)
    frac = min(1.0, max(0.0, (progress - start) / max(1e-06, 1.0 - start)))
    return 1.0 + frac * (DEMON_BETA1_FINAL / DEMON_BETA1_REF - 1.0)

def muon_beta2(progress: float) -> float:
    """Plan §11.1 item 2: the Muon/NorMuon second-moment beta2, ramped over the run.

    Returns MUON_BETA2 unchanged when the ramp is off (MUON_BETA2_FINAL <= 0), which is what keeps
    the default path passing the identical Python float it always has -- important because this
    value is consumed INSIDE the compiled Muon body, where a per-step Python float is a Dynamo
    guard and therefore a per-step recompile of the whole optimizer graph. When the ramp IS on,
    _muon_step routes `1 - beta2` through _muon_compile_scalars as a device tensor instead.
    """
    if MUON_BETA2_FINAL <= 0.0:
        return MUON_BETA2
    p = min(1.0, max(0.0, progress))
    return MUON_BETA2 + p * (MUON_BETA2_FINAL - MUON_BETA2)

def muon_depth_enabled() -> tuple[bool, bool]:
    """Plan §11.1 item 3: (per-depth LR on, per-depth momentum on).

    Both halves are independently gateable so the two axes Recursive changed together can be
    measured apart -- LR x1.15->x0.85 and momentum 0.90->0.97 are not the same claim.
    """
    lr_on = not (MUON_DEPTH_LR_BOTTOM == 1.0 and MUON_DEPTH_LR_TOP == 1.0)
    mom_on = MUON_DEPTH_MOM_BOTTOM > 0.0 and MUON_DEPTH_MOM_TOP > 0.0
    return (lr_on, mom_on)

def muon_depth_mults(depths: list[int | None], n_layer: int) -> tuple[list[float], list[float]]:
    """Per-parameter (lr multiplier, one-minus-momentum multiplier) for one Muon group.

    `depths[i]` is the transformer-block index of the group's i-th parameter, or None for a
    parameter that lives outside the block stack (the n-gram up-projections, Engram's projections)
    and therefore has no depth -- those get a neutral 1.0 on both axes rather than being forced to
    an arbitrary end of the ramp.

    The momentum multiplier is on (1 - momentum), see the MUON_DEPTH_MOM_* comments: scaling
    momentum itself would push it past 1.0 and diverge, and would also flatten the global
    warmup/cooldown schedule this rides on top of.
    """
    lr_on, mom_on = muon_depth_enabled()
    span = max(1, n_layer - 1)
    lr_mult, mom_mult = ([], [])
    for d in depths:
        if d is None or not lr_on:
            lr_mult.append(1.0)
        else:
            f = d / span
            lr_mult.append(MUON_DEPTH_LR_BOTTOM + f * (MUON_DEPTH_LR_TOP - MUON_DEPTH_LR_BOTTOM))
        if d is None or not mom_on:
            mom_mult.append(1.0)
        else:
            f = d / span
            target = MUON_DEPTH_MOM_BOTTOM + f * (MUON_DEPTH_MOM_TOP - MUON_DEPTH_MOM_BOTTOM)
            mom_mult.append((1.0 - target) / max(1e-06, 1.0 - MUON_DEPTH_MOM_REF))
    return (lr_mult, mom_mult)

def main() -> None:
    global USE_LOCAL_CONV, LRM_QUANTUM, USE_TENSOR_LR_SCALARS, USE_FOREACH_OPTIM
    global LOCAL_CONV_KERNEL, TRAIN_LOCAL_CONV, USE_NKI_LOCAL_CONV
    global USE_VE
    global MATRIX_LR, EMBEDDING_LR, UNEMBEDDING_LR, SCALAR_LR, WEIGHT_DECAY, WD_DECAY_TO_ZERO
    global FINAL_LR_FRAC, WARMDOWN_RATIO, INIT_SCALE, ATTN_CPROJ_INIT_SCALE, LOGIT_SOFTCAP
    global MUON_NS_STEPS, MUON_MOMENTUM_COOLDOWN_FRAC, MUON_MOMENTUM_FINAL
    global ADAM_EVERY_N, ADAM_EVERY_N_LR_COMP, PACK_FACTOR, GRAD_CLIP, LOSS_SCALE
    global CLIP_AFTER_REDUCE
    global COMPILE_SDPA_DIRECT, USE_BF16_NORM_OUTPUT, USE_BF16_SDPA_INPUT
    global LOSS_SCALE_DYNAMIC, loss_scale_clean, loss_scale_skips
    global DEPTH, N_EMBD, HEAD_DIM, MLP_RATIO, ATTN_SCALE, SEQ_LEN, N_KV_HEADS
    global USE_NGRAM_VE, NGRAM_VE_TABLE_MULT, NGRAM_VE_TRIGRAM, NGRAM_VE_BF16, NGRAM_VE_DIM
    global NGRAM_VE_LAYERS
    global NGRAM_RMS_FP32, USE_NKI_NGRAM_RMS
    global NGRAM_VE_LR, NGRAM_VE_BETA2, NGRAM_VE_FLAT_LR, NGRAM_VE_OPT, NGRAM_VE_CLIP_EXCLUDE
    global USE_OUT_BIGRAM, OUT_BIGRAM_BF16, OUT_BIGRAM_GATE_BIAS, OUT_BIGRAM_LR
    global RELU2_TAU, USE_MLP_SANDWICH_NORM
    global USE_HEAD_GATE, HEAD_GATE_NORM, HEAD_GATE_CHANNELS, HEAD_GATE_MUL, HEAD_GATE_BIAS
    global USE_OUT_POOL, OUT_POOL_LAYERS, USE_QK_SHIFT, QK_SHIFT_BETA, QK_SHIFT_FREEZE
    global USE_X0_GATE, USE_BLOCK_NUDGE, BLOCK_NUDGE_SITE, USE_QKV_NORM_CSE, USE_PARALLEL_BLOCK
    global PARALLEL_BLOCK_COALESCE, USE_EPOCH_SHUFFLE, EPOCH_SHUFFLE_SEED
    global USE_ENGRAM, ENGRAM_SITE, ENGRAM_ORDERS, ENGRAM_HEADS, ENGRAM_TABLE_MULT
    global ENGRAM_MEM_DIM, ENGRAM_KEY_DIM, ENGRAM_CONV_KERNEL, ENGRAM_SHARE_NGRAM_TABLES
    global ENGRAM_BF16
    global BATCH_RAMP, BATCH_RAMP_GRAD_ACCUM, BATCH_RAMP_BOUNDS, BATCH_RAMP_LR_EXP
    global USE_WEIGHT_EMA, WEIGHT_EMA_DECAYS, WEIGHT_EMA_START_FRAC
    global USE_COMPILE_MUON
    global USE_TRANSPOSED_LINEAR
    global USE_BYTE_WTE_INIT, BYTE_WTE_INIT_MIX, BYTE_WTE_INIT_NGRAM, BYTE_WTE_INIT_STD
    global USE_COALESCED_SWIGLU, COALESCED_SWIGLU_HIDDEN
    global USE_CAUTIOUS_WD, SCALAR_BETA1, SCALAR_BETA2
    global DEMON_BETA1, DEMON_BETA1_FINAL, DEMON_BETA1_REF, DEMON_BETA1_START_FRAC
    global MUON_BETA2, MUON_BETA2_FINAL
    global MUON_DEPTH_LR_BOTTOM, MUON_DEPTH_LR_TOP
    global MUON_DEPTH_MOM_BOTTOM, MUON_DEPTH_MOM_TOP, MUON_DEPTH_MOM_REF
    parser = argparse.ArgumentParser(description='Neuron Competition R1 baseline')
    parser.add_argument('--device-type', type=str, default='')
    parser.add_argument('--num-steps', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_NUM_STEPS', NUM_STEPS)))
    parser.add_argument('--debug-numerics-steps', type=int, default=0, help='Diagnostic only: synchronize and inspect loss/parameters/gradients at each stage of the first N steps. Invalidates speed measurements.')
    parser.add_argument('--max-train-seconds', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS', MAX_TRAIN_SECONDS)))
    parser.add_argument('--out-dir', type=str, default=os.environ.get('NEURON_COMPETITION_R1_OUT_DIR', 'out'))
    parser.add_argument('--eval-public', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_EVAL_PUBLIC', '1') != '0')
    parser.add_argument('--eval-tokens', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_EVAL_TOKENS', str(4 * 524288))))
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_COMPILE', '1') != '0', help="enable torch.compile(backend='neuron') on the full model. Default ON (Bucket E, real full-1800s-confirmed win: 827 steps/1.113006 vs eager's 746 steps/1.122271, causality exact) -- pass --no-compile for the plain eager comparison path.")
    parser.add_argument('--compile-only', action='store_true', help='run one synthetic optimizer step to warm the compile cache, then exit')
    parser.add_argument('--nki-relu2', action='store_true', default=os.environ.get('NEURON_COMPETITION_R1_NKI_RELU2', '0') != '0', help='use the example fused NKI relu(x)**2 kernel in the MLP (Neuron only; illustration)')
    parser.add_argument('--nki-rope-norm', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_NKI_ROPE_NORM', '1') != '0', help='fuse norm(apply_rotary_emb(x)) into one NKI kernel for q/k (Neuron only). Default ON (Solution 12, real ~15-18%% step-time win, verified to 1-bf16-ULP against eager) -- pass --no-nki-rope-norm for the plain eager comparison path.')
    parser.add_argument('--nki-softcap-ce', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_NKI_SOFTCAP_CE', '1') != '0', help='fuse the logit softcap and cross-entropy (fwd+bwd) into one NKI kernel pair (Neuron only), collapsing ~8 full-[B*T,V] NEFF launches into 2. Default ON: verified numerically offline (research/scratch/softcap_ce_equiv.py) and confirmed on real hardware at the final locked shape -- pass --no-nki-softcap-ce for the plain eager comparison path.')
    parser.add_argument('--byte-loss-weight', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_BYTE_LOSS_WEIGHT', '0') != '0', help="Sub-1.0 plan P4: weight each token's training loss by its byte length (sum(w*l)/sum(w), mean-normalized so the loss scale is unchanged), aligning the objective with bpb. Weights come from the provided tokenizer artifact only. Default OFF = plain mean, bit-identical.")
    parser.add_argument('--nki-muon', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_NKI_MUON', '1') != '0', help="fuse the elementwise ops wrapping Muon's Newton-Schulz matmuls, plus the sign-gated weight update, into three NKI kernels (Neuron only). The matmuls themselves stay in aten. Default ON: removes 92 of Muon's 320 per-step device launches (-25.5%% off the Muon window) on a step whose optimizer window is ~54%% inter-launch gap, with BIT-IDENTICAL update math (research/scratch/muon_nki_equiv.py: 0 differing elements). Real paired 900s A/B: 1019 steps/1.089758 vs 1004/1.090677 -- pass --no-nki-muon for the plain eager comparison path.")
    parser.add_argument('--nki-attn', action='store_true', default=os.environ.get('NEURON_COMPETITION_R1_NKI_ATTN', '0') != '0', help='run causal SDPA through the fused NKI fwd/bwd pair (Neuron only). Default OFF.')
    parser.add_argument('--compile-sdpa-direct', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_COMPILE_SDPA_DIRECT', '1' if COMPILE_SDPA_DIRECT else '0') != '0', help='Call causal SDPA directly when --nki-attn is off, allowing Dynamo to trace it instead of entering the disabled attention wrapper. Default OFF preserves the historical graph partitioning. The custom NKI attention path retains its wrapper when enabled.')
    parser.add_argument('--muon-lr-fp32', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_MUON_LR_FP32', '0') != '0', help="see the _muon_lr_g docstring. Default OFF (the locked, current behaviour): a 0-dim fp32 lr multiplied against the bf16 Newton-Schulz iterate g gets CAST DOWN to bf16 before the multiply, by PyTorch's own 0-dim type-promotion rule -- quantising every Muon group's effective lr by up to ~0.4%% relative error, every step. ON keeps that product in fp32 instead (g upcast, lr untouched, cast down to w's dtype only at the final subtract). Zero-cost precision fix, no expected throughput impact; does not touch the per-depth [K,1,1] lr path's intentional bf16-matching behaviour. Its own flag and its own A/B, per that docstring.")
    parser.add_argument('--compile-muon', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_COMPILE_MUON', '1' if USE_COMPILE_MUON else '0') != '0', help="T1.7: run Muon's per-group math body inside torch.compile(backend='neuron') instead of ~80 eager aten launches (Neuron only). Mutually exclusive with --nki-muon by construction -- a compiled region cannot contain an nki.jit call -- so this trades the NKI elementwise fusion for whole-body fusion. See _muon_math.")
    parser.add_argument('--transposed-linear', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_TRANSPOSED_LINEAR', '1' if USE_TRANSPOSED_LINEAR else '0') != '0', help="T1.6: store every Linear weight as [in, out] and compute x @ weight instead of F.linear's x @ weight.T, on the theory that TensorE wants the contraction dim on partitions and the [out, in] storage costs a transpose per matmul. Bit-exact by construction (weights are initialized in the native layout and transposed afterwards; Muon is told the logical orientation so its one-sided `ratio` LR scale is unchanged). SHAPE-CRITICAL: flips every Linear weight's state_dict shape.")
    parser.add_argument('--local-conv', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_LOCAL_CONV', '1' if USE_LOCAL_CONV else '0') != '0', help="depthwise causal conv1d before attention in each block. NOT dead compute any more: T0.1 put local_conv.weight in its own AdamW group (TRAIN_LOCAL_CONV=True), so --no-local-conv now REMOVES 7,680 TRAINED parameters and is a real capacity ablation, not the bit-identical no-op this help text used to claim. Measured on the promoted stack (120-step instrument, throughput->capacity track): --no-local-conv is +2.65%% faster (161,075 vs 156,917 tok/s) but its step-110 loss is 4.2054 vs the control's 3.9380 -- a huge quality cost for 2.65%%, so DO NOT ablate it for speed. (This also reverses the older 'turning the compute off measured 6.9x-compiled-graph SLOWER' note in the USE_LOCAL_CONV comment: that was the pre-NKI eager stack.) Kept as a flag so an A/B runs from one identical binary.")
    parser.add_argument('--nki-local-conv', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_NKI_LOCAL_CONV', '1' if USE_NKI_LOCAL_CONV else '0') != '0', help='Use the gated BTC-layout native NKI two-tap convolution for FP32 residuals with time/channels divisible by 128. Other shapes and precisions retain the original causal convolution.')
    parser.add_argument('--bf16-norm-output', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_BF16_NORM_OUTPUT', '1' if USE_BF16_NORM_OUTPUT else '0') != '0', help='Keep FP32 RMS statistics/epsilon and residuals, but store normalized branch inputs in BF16, including the final head. This is an opt-in arithmetic change, saved for standalone eval.')
    parser.add_argument('--bf16-sdpa-input', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_BF16_SDPA_INPUT', '1' if USE_BF16_SDPA_INPUT else '0') != '0', help='Cast Q/K/V only at the native SDPA boundary, restoring its output dtype. Independent of --bf16-norm-output; does not modify the alternative --nki-attn implementation.')
    parser.add_argument('--ve', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_VE', '1' if USE_VE else '0') != '0', help="ResFormer-style value embeddings on alternating layers (has_ve), input-gated into attention's V stream. Default ON (the locked value); exposed for the throughput->capacity track's plain-block ablate-down arm. NOTE --no-ve also removes the n-gram value embeddings, because has_ngram_bigram() requires USE_VE and has_ve(layer) -- so --no-ve is strictly a step-time diagnostic, not a candidate config.")
    parser.add_argument('--tensor-lr-scalars', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_TENSOR_LR_SCALARS', '1' if USE_TENSOR_LR_SCALARS else '0') != '0', help='hold the per-step lr in a device tensor rather than a Python float, so a changing LR schedule stops forcing a fresh neuronx-cc compile every step')
    parser.add_argument('--foreach-optim', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_FOREACH_OPTIM', '1' if USE_FOREACH_OPTIM else '0') != '0', help="batch the optimizer's per-parameter work into torch._foreach_* calls instead of a Python loop of single-tensor ops. Bit-identical update math; targets the optimizer step's launch/dispatch overhead only.")
    parser.add_argument('--ngram-ve', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_NGRAM_VE', '1' if USE_NGRAM_VE else '0') != '0', help="T3.1: hashed bigram (all VE layers) and trigram (first/last VE layer) VALUE embeddings, mixed into attention's V through their own zero-init per-head gates on disjoint residual channels (unigram 0:32, bigram 32:64, trigram 64:96). Tables are zero-init, so this is an exact no-op at step 0 and the A/B runs from one identical binary. Strictly causal: a position's table row is a hash of tokens t-2..t only.")
    parser.add_argument('--ngram-ve-layers', type=str, default=os.environ.get('NEURON_COMPETITION_R1_NGRAM_VE_LAYERS', NGRAM_VE_LAYERS), help="which layers get a bigram table, comma-separated, negatives counting from the end ('-1' = last layer at any DEPTH). Empty = every VE layer, Recursive's own allocation. The per-step cost of these tables depends on WHICH layer holds them and not on their size: one table on the last layer is FASTER than no table at all, while any second table anywhere costs ~50ms/micro-step. See NGRAM_VE_LAYERS for the measurement.")
    parser.add_argument('--ngram-ve-table-mult', type=int, default=NGRAM_VE_TABLE_MULT, help='rows per n-gram table = this x padded_vocab_size (8 => 65536 rows, 41.9M params per table at n_embd=640). Recursive went to 64x; start at 8x.')
    parser.add_argument('--ngram-ve-trigram', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_NGRAM_VE_TRIGRAM', '1' if NGRAM_VE_TRIGRAM else '0') != '0', help='add a trigram table on the first and last VE layer as well as the bigram table every VE layer gets. --no-ngram-ve-trigram is the bigram-only arm.')
    parser.add_argument('--ngram-ve-dim', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_NGRAM_VE_DIM', NGRAM_VE_DIM)), help='n-gram table WIDTH (0 = n_embd, gathered straight into v). A throughput knob: the measured cost of these tables scales with rows x width in BOTH the forward gather and the dense embedding backward, so e.g. 128 is ~5x less table to read, scatter into, all-reduce and update, at the price of one [tokens,dim]x[dim,n_embd] matmul per table. See NGRAM_VE_DIM.')
    parser.add_argument('--ngram-ve-opt', type=str, default=NGRAM_VE_OPT, choices=('rmsprop', 'adamw'), help="optimizer for the n-gram tables. rmsprop (default, Recursive's choice) drops the first moment so a rarely-hit row still gets a full-rate update; adamw is the fallback arm and matches the unigram VE group.")
    parser.add_argument('--ngram-ve-lr', type=float, default=NGRAM_VE_LR, help='lr for the n-gram tables before the dmodel scale (0 = use --embedding-lr).')
    parser.add_argument('--ngram-ve-beta2', type=float, default=NGRAM_VE_BETA2, help='second-moment decay for the n-gram table optimizer.')
    parser.add_argument('--ngram-ve-flat-lr', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_NGRAM_VE_FLAT_LR', '1' if NGRAM_VE_FLAT_LR else '0') != '0', help='hold the n-gram tables\' lr FLAT (warmup only, no warmdown), per Recursive\'s "sparse tables benefit from full-rate training". --no-ngram-ve-flat-lr puts them back on the normal cooldown.')
    parser.add_argument('--ngram-ve-bf16', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_NGRAM_VE_BF16', '1' if NGRAM_VE_BF16 else '0') != '0', help='store n-gram tables and gradients in bf16 rather than fp32. Optimizer state follows that dtype unless --ngram-rms-fp32 is on.')
    parser.add_argument('--ngram-rms-fp32', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_NGRAM_RMS_FP32', '1' if NGRAM_RMS_FP32 else '0') != '0', help='Use FP32 second moments AND FP32 RMSProp update arithmetic for n-gram tables, retaining their parameter/gradient dtype. Casts grads before squaring and rounds at the final parameter write. Requires --ngram-ve-opt rmsprop. Default OFF.')
    parser.add_argument('--nki-ngram-rms', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_NKI_NGRAM_RMS', '1' if USE_NKI_NGRAM_RMS else '0') != '0', help='Fuse FP32 table RMSProp in NKI, using both cores under LNC2. Requires --ngram-rms-fp32 and contiguous BF16 Neuron tables. Default OFF; test quality and whole-run time before promotion.')
    parser.add_argument('--ngram-ve-clip-exclude', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_NGRAM_VE_CLIP_EXCLUDE', '1' if NGRAM_VE_CLIP_EXCLUDE else '0') != '0', help='keep the n-gram tables out of the GLOBAL clip_grad_norm_ and clip them as their own group, so adding them cannot change the effective lr of every other tensor through clip_coef. See split_clip_params.')

    def _hp(flag: str, default, typ, helptext: str) -> None:
        env = 'NEURON_COMPETITION_R1_' + flag.lstrip('-').replace('-', '_').upper()
        parser.add_argument(flag, type=typ, default=typ(os.environ.get(env, default)), help=helptext)
    parser.add_argument('--byte-wte-init', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_BYTE_WTE_INIT', '1' if USE_BYTE_WTE_INIT else '0') != '0', help="Plan 11.1 (Recursive's from-vanilla run): initialize wte from a hashed bag of byte n-grams of each token's own byte string (from the provided tokenizer artifact's id->bytes map -- no external data) instead of normal_(0, 0.8), so tokens with similar surface form start correlated. Host-side and one-shot: zero ops added to the training graph, and no torch RNG consumed, so every OTHER tensor is initialized identically to the control arm. See byte_feature_wte_init.")
    _hp('--byte-wte-init-mix', BYTE_WTE_INIT_MIX, float, 'Plan 11.1: 1.0 = pure byte features; mix<1 blends mix*feat + sqrt(1-mix^2)*random, which keeps the row RMS and guarantees no two rows can coincide on a hash collision.')
    _hp('--byte-wte-init-ngram', BYTE_WTE_INIT_NGRAM, int, 'Plan 11.1: highest byte-n-gram order hashed into each row (1..N).')
    _hp('--byte-wte-init-std', BYTE_WTE_INIT_STD, float, "Plan 11.1: target per-row RMS of the byte-feature rows. Must track init_weights' own normal_ std for wte (0.8) or the arm also becomes an embedding-scale test.")
    parser.add_argument('--coalesced-swiglu', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_COALESCED_SWIGLU', '1' if USE_COALESCED_SWIGLU else '0') != '0', help="parameter-matched coalesced SwiGLU MLP: one Linear(n_embd, 2*hidden) split into gate and up halves, silu(gate)*up, at hidden = 8/3*n_embd (1664) so the three matrices cost the same as the baseline's two. Distinct from --gated-mlp/USE_GATED_MLP, which keeps hidden=4*n_embd and so adds 50%% MLP params and FLOPs. SHAPE-CRITICAL. Prior ~15-20%%: the ReLU^2 >= SwiGLU consensus at this scale is hardware-independent.")
    _hp('--coalesced-swiglu-hidden', COALESCED_SWIGLU_HIDDEN, int, 'force the coalesced-SwiGLU hidden width (must be a multiple of 128). 0 = auto, i.e. the parameter-matched round(2/3*mlp_ratio*n_embd / 128)*128. SHAPE-CRITICAL.')
    parser.add_argument('--cautious-wd', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_CAUTIOUS_WD', '1' if USE_CAUTIOUS_WD else '0') != '0', help='T2.3 (modded #50): in the AdamW groups, apply weight decay only where update*p > 0, so decay never fights an element the gradient is driving outward. Affects only the groups with wd != 0 -- wte (0.001) and lm_head (0.01). The Muon groups already do this (see _muon_math).')
    _hp('--scalar-beta1', SCALAR_BETA1, float, 'T2.4 (modded #52): Adam beta1 for the scalar/skip group. Default 0.9 = historical.')
    _hp('--scalar-beta2', SCALAR_BETA2, float, "T2.4 (modded #52): Adam beta2 for the scalar/skip group. Default 0.95 = historical; record #52's 'smoothed scalars' uses 0.99, usually with a lower --scalar-lr.")
    parser.add_argument('--demon-beta1', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_DEMON_BETA1', '1' if DEMON_BETA1 else '0') != '0', help="plan 11.1 (Recursive): anneal the AdamW groups' beta1 down across the LR warmdown ('Demon'), as a multiplicative shrink on each group's own initial beta1 calibrated so a 0.8 group lands on --demon-beta1-final. The rmsprop n-gram group has no first moment and is unaffected.")
    _hp('--demon-beta1-final', DEMON_BETA1_FINAL, float, 'plan 11.1: beta1 at the end of the run for a group whose initial beta1 is --demon-beta1-ref. Recursive uses 0.55 from 0.8. Only read with --demon-beta1.')
    _hp('--demon-beta1-ref', DEMON_BETA1_REF, float, "plan 11.1: the initial beta1 that --demon-beta1-final is quoted against (Recursive: 0.8, which is also this repo's embedding-family beta1).")
    _hp('--demon-beta1-start-frac', DEMON_BETA1_START_FRAC, float, "plan 11.1: wall-clock progress fraction at which the beta1 anneal starts. Negative (default) = start where the LR warmdown starts, i.e. 1 - WARMDOWN_RATIO = 0.25, which is Recursive's 'across its warmdown'. 0.0 = anneal across the whole run.")
    _hp('--muon-beta2', MUON_BETA2, float, "plan 11.1 / NorMuon: the Muon groups' second-moment beta2, i.e. the decay of the per-row second moment of the ORTHOGONALIZED update that rescales it. Default 0.90 = the tuned historical literal. Recursive's variant sits at 0.95->0.97; SkyPilot's autoresearch found 0.95->0.98 worth ~0.001 at their setting -- neither transfers as a VALUE here, only as a direction, so sweep it.")
    _hp('--muon-beta2-final', MUON_BETA2_FINAL, float, 'plan 11.1 / NorMuon: ramp the Muon beta2 linearly from --muon-beta2 to this over elapsed/max_train_seconds. 0.0 (default) = OFF, and specifically = keep handing the compiled Muon body `1-beta2` as the Python float it is today, so the traced graph is byte-identical. Recursive ramps 0.95 -> 0.97.')
    _hp('--muon-depth-lr-bottom', MUON_DEPTH_LR_BOTTOM, float, "plan 11.1: Muon LR multiplier for layer 0's matrices, ramped linearly to --muon-depth-lr-top at the last layer. Recursive: 1.15 -> 0.85. Both 1.0 (default) = OFF. Carried as a per-parameter [K,1,1] operand inside the existing per-shape groups, so the batched Newton-Schulz and the compiled-graph count are unchanged.")
    _hp('--muon-depth-lr-top', MUON_DEPTH_LR_TOP, float, "plan 11.1: Muon LR multiplier for the LAST layer's matrices. Recursive: 0.85.")
    _hp('--muon-depth-mom-bottom', MUON_DEPTH_MOM_BOTTOM, float, 'plan 11.1: target Muon momentum for layer 0 when the global scheduled momentum is at --muon-depth-mom-ref. Recursive: 0.90 (bottom) -> 0.97 (top). Applied as a multiplier on (1 - momentum) so it rides ON TOP of the global warmup/cooldown schedule and can never push momentum past 1. 0.0 (default, either endpoint) = OFF.')
    _hp('--muon-depth-mom-top', MUON_DEPTH_MOM_TOP, float, 'plan 11.1: target Muon momentum for the LAST layer. Recursive: 0.97.')
    _hp('--muon-depth-mom-ref', MUON_DEPTH_MOM_REF, float, "plan 11.1: the global momentum the two --muon-depth-mom-* targets are quoted against. 0.95 (default) = muon_momentum()'s warmed-up plateau.")
    parser.add_argument('--out-bigram', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_OUT_BIGRAM', '1' if USE_OUT_BIGRAM else '0') != '0', help='T4.2: a gradient-trained [vocab, vocab] table indexed by the CURRENT token whose row is added to the next-token logits, gated per token by sigmoid(Linear(n_embd->1)). Zero-init table => exact no-op at step 0. 67.1M bf16 parameters. Distinct from the frozen count-based mixer that was closed as a negative on 2026-09-12. See USE_OUT_BIGRAM.')
    parser.add_argument('--out-bigram-bf16', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_OUT_BIGRAM_BF16', '1' if OUT_BIGRAM_BF16 else '0') != '0', help="store the [vocab, vocab] output bigram table in bf16 (Aster's choice).")
    _hp('--out-bigram-gate-bias', OUT_BIGRAM_GATE_BIAS, float, "T4.2: bias init of the output bigram table's per-token gate. Aster specifies -2.0.")
    _hp('--out-bigram-lr', OUT_BIGRAM_LR, float, "T4.2: lr for the output bigram table's own AdamW group. 0 => UNEMBEDDING_LR.")
    _hp('--relu2-tau', RELU2_TAU, float, "T3.9: shifted ReLU^2 -- the MLP activation becomes relu(h - tau)^2. 0.0 = historical. Recursive: 'tau=0.5 confirmed optimal'.")
    parser.add_argument('--mlp-sandwich-norm', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_MLP_SANDWICH_NORM', '1' if USE_MLP_SANDWICH_NORM else '0') != '0', help='T3.9: x = x + norm(mlp(norm(x))) instead of x = x + mlp(norm(x)).')
    parser.add_argument('--head-gate', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_HEAD_GATE', '1' if USE_HEAD_GATE else '0') != '0', help='T3.6/T3.9: per-head gate on the attention output, read from a residual slice that no value-embedding gate has claimed (see head_gate_slice).')
    parser.add_argument('--head-gate-norm', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_HEAD_GATE_NORM', '1' if HEAD_GATE_NORM else '0') != '0', help="RMS-norm each head's attention output before gating (Recursive's form). Pass --no-head-gate-norm for the gate-only variant, which is an exact no-op at step 0.")
    _hp('--head-gate-channels', HEAD_GATE_CHANNELS, int, 'T3.6/T3.9: width of the residual slice the head gate reads. SHAPE-CRITICAL.')
    _hp('--head-gate-mul', HEAD_GATE_MUL, float, "T3.6/T3.9: gate = this * sigmoid(...). 2.0 makes zero-init neutral; 1.0 is the plan's sigmoid-only T3.6 form.")
    _hp('--head-gate-bias', HEAD_GATE_BIAS, float, 'T3.6/T3.9: bias init of the head gate. 0.0 is neutral at --head-gate-mul 2.')
    parser.add_argument('--out-pool', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_OUT_POOL', '1' if USE_OUT_POOL else '0') != '0', help='T3.9: zero-init pooling of the last --out-pool-layers block outputs into the final residual. Exact no-op at step 0.')
    _hp('--out-pool-layers', OUT_POOL_LAYERS, int, 'T3.9: how many trailing block outputs to pool. Recursive uses 3. SHAPE-CRITICAL.')
    parser.add_argument('--qk-shift', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_QK_SHIFT', '1' if USE_QK_SHIFT else '0') != '0', help='T3.9: causal token shift on the Q/K inputs only (q_in = k_in = x + beta*x_prev, per-dim learned beta, V unshifted). Different mechanism from --local-conv.')
    _hp('--qk-shift-beta', QK_SHIFT_BETA, float, "T3.9: per-dim init of the Q/K shift's beta. Recursive uses 0.3.")
    parser.add_argument('--qk-shift-freeze', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_QK_SHIFT_FREEZE', '1' if QK_SHIFT_FREEZE else '0') != '0', help="freeze the Q/K shift's beta (requires_grad=False, no optimizer group) so it stays at --qk-shift-beta forever. With --qk-shift-beta 0.0 this is a mathematical identity that still keeps the elementwise ops in the compiled graph -- the decisive probe for the block-body compiler-scheduling speedup.")
    parser.add_argument('--block-nudge', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_BLOCK_NUDGE', '1' if USE_BLOCK_NUDGE else '0') != '0', help='add `x = x + nudge*x` with one frozen zero scalar per block -- a mathematical identity whose only effect is on the compiled graph. The cheap, general form of the +10%% throughput the frozen qk-shift found.')
    parser.add_argument('--block-nudge-site', type=str, choices=('qk', 'resid'), default=os.environ.get('NEURON_COMPETITION_R1_BLOCK_NUDGE_SITE', BLOCK_NUDGE_SITE), help="where --block-nudge lands: the Q/K input ('qk', where --qk-shift put it) or the residual stream after the resid/x0 mix ('resid').")
    parser.add_argument('--parallel-block', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_PARALLEL_BLOCK', '1' if USE_PARALLEL_BLOCK else '0') != '0', help='GPT-J/PaLM parallel block: attention and MLP both read one norm(x) and are summed into the residual together (x + la*attn + lm*mlp). Removes 3 of 4 norms per block and unserialises the two branches.')
    parser.add_argument('--parallel-block-coalesce', type=str, choices=('none', 'vfc', 'all'), default=os.environ.get('NEURON_COMPETITION_R1_PARALLEL_BLOCK_COALESCE', PARALLEL_BLOCK_COALESCE), help="merge the parallel block's input projections into one matmul: 'vfc' (c_v+c_fc, no copies) or 'all' (q,k,v,c_fc, needs contiguous copies of q/k to keep the rope+QK-norm NKI kernel). Needs --parallel-block.")
    parser.add_argument('--qkv-norm-cse', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_QKV_NORM_CSE', '1' if USE_QKV_NORM_CSE else '0') != '0', help='hoist the three identical norm(q_in)/norm(k_in)/norm(v_in) calls into one when they share the same input tensor. Bit-exact identity; the only effect is 2 fewer ops per block in a dispatch-bound graph.')
    parser.add_argument('--engram', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_ENGRAM', '1' if USE_ENGRAM else '0') != '0', help='T3.2 Engram-lite: hashed n-gram memory read through a gate on the hidden state (alpha = sigmoid(norm(h).norm(k)/sqrt(d))), then Y = SiLU(dwconv(norm(alpha*v))) + alpha*v added to the residual. The context-aware superset of the n-gram value embeddings.')
    parser.add_argument('--engram-site', type=str, default=os.environ.get('NEURON_COMPETITION_R1_ENGRAM_SITE', ENGRAM_SITE), help="block index whose OUTPUT residual Engram writes into; negative counts from the end. Default 2 (the paper's mid-stack site): T3.1's LNC=2 downstream-resharding penalty was expected to make this unaffordable and measurably does NOT (1128 vs 1129 steps at 900s), while being 0.0016 bpb better than the last-layer placement.")
    parser.add_argument('--engram-orders', type=str, default=os.environ.get('NEURON_COMPETITION_R1_ENGRAM_ORDERS', ENGRAM_ORDERS), help="comma-separated n-gram orders for Engram's hash heads; only 2 and 3 are supported (Engram reports 4-grams slightly suboptimal).")
    parser.add_argument('--engram-heads', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_ENGRAM_HEADS', ENGRAM_HEADS)), help="hash heads PER ORDER. Each head is its own table with its own primes, so K heads is K independent hashings of the same n-gram (Engram's own collision-mitigation trick). Total tables = heads * len(orders).")
    parser.add_argument('--engram-table-mult', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_ENGRAM_TABLE_MULT', ENGRAM_TABLE_MULT)), help='rows per Engram table, in units of the padded vocab (8192). 32 => 262144 rows; the modulus used is the largest prime <= that.')
    parser.add_argument('--engram-mem-dim', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_ENGRAM_MEM_DIM', ENGRAM_MEM_DIM)), help='d_mem: width of each Engram table row. e_t is the concatenation over all tables, so w_k/w_v read heads*len(orders)*d_mem.')
    parser.add_argument('--engram-key-dim', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_ENGRAM_KEY_DIM', ENGRAM_KEY_DIM)), help="width of the gate's key/query space. 0 (default) means n_embd, which lets the gate dot norm(h) directly and needs no query projection.")
    parser.add_argument('--engram-conv-kernel', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_ENGRAM_CONV_KERNEL', ENGRAM_CONV_KERNEL)), help='depthwise causal conv kernel applied to norm(alpha*v) before the SiLU. 0 disables the conv branch entirely (Y = alpha*v), which is the ablation arm. Zero-init, so it starts as a no-op either way.')
    parser.add_argument('--engram-share-ngram-tables', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_ENGRAM_SHARE_NGRAM_TABLES', '1' if ENGRAM_SHARE_NGRAM_TABLES else '0') != '0', help="read T3.1's ALREADY-GATHERED n-gram rows instead of Engram's own tables. Isolates the MECHANISM (gate + conv + residual write) from the added capacity: no new gather, no new table parameters, no extra gradient all-reduce. Needs an n-gram table at or before --engram-site.")
    parser.add_argument('--engram-bf16', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_ENGRAM_BF16', '1' if ENGRAM_BF16 else '0') != '0', help="hold Engram's tables in bf16 (halves 67M params of memory and the gather's bandwidth). Same trade as --ngram-ve-bf16.")
    parser.add_argument('--epoch-shuffle', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_EPOCH_SHUFFLE', '1' if USE_EPOCH_SHUFFLE else '0') != '0', help="reshuffle the row-group order from epoch 2 onward. Epoch 1 keeps prepare's exact order, so this is a no-op for any run that never crosses the epoch boundary. Only matters above ~2195 steps at the locked shape -- see the 2026-09-13 data-wall section of research/trn2-experiment-results.md.")
    parser.add_argument('--epoch-shuffle-seed', type=int, default=EPOCH_SHUFFLE_SEED, help='seed base for --epoch-shuffle; epoch e is shuffled with seed+e.')
    parser.add_argument('--x0-gate', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_X0_GATE', '1' if USE_X0_GATE else '0') != '0', help="Recursive's input-dependent x0 gate: scale each block's x0 skip by 2*sigmoid(s*mean(x)) with one zero-init learned scalar s per layer. Bit-exact no-op at step 0. Mutually exclusive with USE_SELECTIVE_GATE, which owns the same line of Block.forward.")
    parser.add_argument('--batch-ramp', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_BATCH_RAMP', '1' if BATCH_RAMP else '0') != '0', help='T2.1: ramp the effective batch size across the run by varying grad_accum_steps on wall-clock-fraction boundaries, holding the micro shape (and therefore every compiled graph) fixed. Off by default.')
    parser.add_argument('--batch-ramp-grad-accum', type=str, default=os.environ.get('NEURON_COMPETITION_R1_BATCH_RAMP_GRAD_ACCUM', ','.join((str(g) for g in BATCH_RAMP_GRAD_ACCUM))), help="T2.1: comma-separated micro-batch counts for stages A,B,C. Stage B should normally be TOTAL_BATCH_SIZE's own grad_accum (4 at the locked shape) so the ramp is symmetric around today's batch size.")
    parser.add_argument('--batch-ramp-bounds', type=str, default=os.environ.get('NEURON_COMPETITION_R1_BATCH_RAMP_BOUNDS', ','.join((str(b) for b in BATCH_RAMP_BOUNDS))), help='T2.1: comma-separated elapsed/max_train_seconds boundaries A->B,B->C. The second should normally equal WARMDOWN_RATIO so the largest batch covers exactly the lr cooldown.')
    parser.add_argument('--batch-ramp-lr-exp', type=float, default=float(os.environ.get('NEURON_COMPETITION_R1_BATCH_RAMP_LR_EXP', BATCH_RAMP_LR_EXP)), help='T2.1: lr scales as (B/B_ref)**this. 0.5 = sqrt (gradient-noise matched), 1.0 = linear scaling, 0.0 = leave the lr alone.')
    parser.add_argument('--weight-ema', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_WEIGHT_EMA', '1' if USE_WEIGHT_EMA else '0') != '0', help='T2.2: keep an exponential moving average of the weights over the cooldown tail, save it alongside the raw weights, and evaluate both. Off by default.')
    parser.add_argument('--weight-ema-decay', type=str, default=os.environ.get('NEURON_COMPETITION_R1_WEIGHT_EMA_DECAY', ','.join((str(d) for d in WEIGHT_EMA_DECAYS))), help="T2.2: comma-separated per-step EMA decays -- one shadow each, all updated from the same weights every step, so they are a zero-noise comparison. The averaging window is ~1/(1-decay) steps (0.98 ~ 50, 0.995 ~ 200, 0.999 ~ 1000). The FIRST is the primary: it is the one saved into the checkpoint's 'ema_state'.")
    parser.add_argument('--weight-ema-start-frac', type=float, default=float(os.environ.get('NEURON_COMPETITION_R1_WEIGHT_EMA_START_FRAC', WEIGHT_EMA_START_FRAC)), help='T2.2: elapsed/max_train_seconds at which the shadow is seeded from the live weights. Defaults to WARMDOWN_RATIO (average the cooldown).')
    parser.add_argument('--lrm-quantum', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_LRM_QUANTUM', LRM_QUANTUM)), help='snap the LR multiplier onto a 1/N grid (0 = continuous). Each distinct lr value costs one neuronx-cc compile, so a continuous cooldown recompiles the optimizer every step.')
    parser.add_argument('--init-from', type=str, default='', help='warm-start model weights from a prior checkpoint (fresh optimizer/step/dataloader; dev-iteration only, never for a scored submission)')
    parser.add_argument('--grad-bucket', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_GRAD_BUCKET', '0') != '0', help="T1.2: point every parameter's .grad at a view into one flat fp32 buffer and all_reduce it once per step instead of once per parameter. Also switches grad clipping to the flat-buffer one-norm/one-mul form (T1.3). Verified bit-equal to the per-tensor path in research/scratch/grad_bucket_equiv.py. Default OFF pending a real hardware A/B (~1%% expected win).")
    parser.add_argument('--clip-foreach', action=argparse.BooleanOptionalAction, default=None if 'NEURON_COMPETITION_R1_CLIP_FOREACH' not in os.environ else os.environ['NEURON_COMPETITION_R1_CLIP_FOREACH'] != '0', help='T1.3: force foreach=True/False on torch.nn.utils.clip_grad_norm_ instead of letting it auto-detect (the default when the flag is absent). Numerically identical; only matters when --grad-bucket is off. Measured NEUTRAL at LNC=2 (clip_ms 6.2-7.7 vs 5.5-7.7).')
    parser.add_argument('--seed', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_SEED', '42')), help='torch.manual_seed value (init_runtime). Default 42, unchanged from the long-standing hardcoded value -- exposed as a flag purely so noise/variance can be estimated with paired seeds without editing the file.')
    parser.add_argument('--local-conv-kernel', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_LOCAL_CONV_KERNEL', str(LOCAL_CONV_KERNEL))), help="depthwise causal conv1d kernel size for --local-conv (T0.1 sweep knob). Only matters once the conv's weight is actually trained (see the USE_LOCAL_CONV comment); a shape-critical toggle already covered by SHAPE_CRITICAL_TOGGLES via LOCAL_CONV_KERNEL.")
    parser.add_argument('--train-local-conv', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_TRAIN_LOCAL_CONV', '1' if TRAIN_LOCAL_CONV else '0') != '0', help='T0.1: register local_conv.weight in its own AdamW group (lr=MATRIX_LR, betas=(0.9,0.95), wd=0.0) instead of leaving it permanently zero/dead. Still zero-initialized, so step 0 is unchanged either way. Default OFF (bit-identical to every prior locked-baseline run) -- pass --train-local-conv for the A/B; only matters when --local-conv is on.')
    _hp('--seq-len', SEQ_LEN, int, 'Training context length. Previously only settable by editing the constant. Currently 2048. NOTE: the in-process eval runs at this same length, so a --seq-len A/B is apples-to-oranges unless --eval-seq-len pins both arms to one eval length -- which matters a lot now that the scorer looks like it evaluates at 1024 (see the 2026-09-11 section of trn2-experiment-results.md).')
    parser.add_argument('--eval-seq-len', type=int, default=int(os.environ.get('NEURON_COMPETITION_R1_EVAL_SEQ_LEN', 0)), help="Run a SECOND in-process public_val eval at this context length and report it as public_val_bpb_alt. 0 (default) = off, zero cost. Must be <= --seq-len and a multiple of 128: the live model's rotary/mask tables are sized to --seq-len (a LONGER eval needs prepare.py eval-public, whose load_for_eval widens them to EVAL_MIN_SEQ_LEN). Post-training only, so it costs wall clock and never a charged training second.")
    _hp('--depth', DEPTH, int, 'T0.2: n_layer. Previously only settable by editing the constant, so every historical depth sweep also moved the width (n_embd is derived from DEPTH via ASPECT_RATIO unless --n-embd is given). Currently 6.')
    _hp('--n-embd', N_EMBD, int, 'T0.2: force n_embd (must be a multiple of --head-dim), decoupling width from DEPTH*ASPECT_RATIO. 0 (default) = historical derived width, bit-identical.')
    _hp('--n-kv-heads', N_KV_HEADS, int, 'Sub-1.0 plan P2: grouped-query attention KV-head count. 0 (default) = dense, bit-identical. Must divide n_head; at the locked width (n_head=5) the only nontrivial choice is 1 (MQA). Shrinks the K/V projections; QK^T stays full-width.')
    _hp('--head-dim', HEAD_DIM, int, 'T0.2: per-head width; n_head = n_embd / head_dim. Also the rounding granularity of the derived n_embd. Currently 128. The fused rope+QK-norm NKI kernel is written generically in D (it only requires D even), so 64 stays on the kernel.')
    _hp('--mlp-ratio', MLP_RATIO, int, 'T0.3: MLP hidden width as a multiple of n_embd. Currently 4.')
    _hp('--attn-scale', ATTN_SCALE, float, "T0.3: explicit SDPA softmax scale. 0.0 (default) = SDPA's own 1/sqrt(head_dim) (0.0884 at 128). QK-norm already fixes the q/k norms, so the logit temperature is a free knob here rather than a redundant init reparameterisation.")
    _hp('--matrix-lr', MATRIX_LR, float, "T0.4: Muon groups' base lr (also the AdamW lr for the gate/local_conv groups). Currently 0.025.")
    _hp('--embedding-lr', EMBEDDING_LR, float, 'T0.4: wte (and value-embedding) AdamW lr before the (n_embd/768)**-0.5 scale. Currently 0.30.')
    _hp('--unembedding-lr', UNEMBEDDING_LR, float, 'T0.4: lm_head AdamW lr before the d_model scale. Currently 0.006.')
    _hp('--scalar-lr', SCALAR_LR, float, 'T0.4: AdamW lr for the scalar/skip params. Currently 0.20.')
    _hp('--weight-decay', WEIGHT_DECAY, float, "T0.4: Muon groups' weight decay (decayed to zero over the run unless --no-wd-decay-to-zero). Currently 0.10.")
    _hp('--final-lr-frac', FINAL_LR_FRAC, float, 'T0.4: LR floor at the end of cooldown as a fraction of peak. Currently 0.05.')
    _hp('--warmdown-ratio', WARMDOWN_RATIO, float, 'T0.4: fraction of the wall-clock budget spent on LR cooldown. Currently 0.75.')
    _hp('--init-scale', INIT_SCALE, float, 'T0.4: multiplier on the attn/mlp uniform-init bound. Currently 0.68.')
    _hp('--attn-cproj-init', ATTN_CPROJ_INIT_SCALE, float, 'T0.4: attn.c_proj init bound as a multiple of the shared bound s; 0.0 (default) keeps the historical exact-zero init.')
    _hp('--logit-softcap', LOGIT_SOFTCAP, float, 'T0.4: tanh logit softcap. Currently 15.0.')
    _hp('--muon-ns-steps', MUON_NS_STEPS, int, 'T0.4: Newton-Schulz iterations per Muon step (3 matmuls each). Currently 5.')
    _hp('--muon-momentum-cooldown-frac', MUON_MOMENTUM_COOLDOWN_FRAC, float, 'T0.4: ramp Muon momentum from its warmed-up 0.95 down to --muon-momentum-final over this trailing fraction of the wall-clock budget. 0.0 (default) = no cooldown, bit-identical to the old schedule.')
    _hp('--muon-momentum-final', MUON_MOMENTUM_FINAL, float, 'T0.4: momentum at the very end of the cooldown. Only read when --muon-momentum-cooldown-frac > 0.')
    parser.add_argument('--wd-decay-to-zero', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_WD_DECAY_TO_ZERO', '1' if WD_DECAY_TO_ZERO else '0') != '0', help="T0.4: decay Muon's weight_decay linearly to 0 over the run instead of holding it constant. Currently ON.")
    _hp('--grad-clip', GRAD_CLIP, float, "Gradient-norm clip threshold. By default clips each rank's local gradients before all_reduce(AVG); --clip-after-reduce clips the averaged global-batch gradients. 0 disables clipping. Separate n-gram clipping is preserved.")
    parser.add_argument('--clip-after-reduce', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_CLIP_AFTER_REDUCE', '1' if CLIP_AFTER_REDUCE else '0') != '0', help='Average all accumulated gradients before clipping. Default OFF preserves the historical local-clip-then-average update. Requires --adam-every-n 1; supports separate n-gram clipping.')
    _hp('--loss-scale', LOSS_SCALE, float, 'Sub-1.0 plan P2: static loss-scale factor. Multiplies the training loss before backward and divides grads back before clip/sync/step, so backward grads stay representable under compiler fp8 auto-cast (full-model grads underflow fp8_e4m3 otherwise). 1.0 (default) = off, bit-identical. Use powers of 2 (128/1024).')
    parser.add_argument('--loss-scale-dynamic', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_LOSS_SCALE_DYNAMIC', '0') != '0', help='Dynamic loss scaling for fp8 auto-cast: skip+halve the update on grad overflow (all ranks agree via all-reduce), double the scale every 500 clean steps (cap 8192). Starts from --loss-scale. Default OFF.')
    _hp('--pack-factor', PACK_FACTOR, int, 'T1.5: dataloader packing-bin multiplier. The loader fetches DEVICE_BATCH_SIZE rows of PACK_FACTOR*SEQ_LEN tokens and reshapes them to DEVICE_BATCH_SIZE*PACK_FACTOR rows of SEQ_LEN, so this is the micro-batch ROW COUNT and grad_accum_steps falls as it rises (TOTAL_BATCH_SIZE is held fixed). Only ever set by editing this constant before; a flag so the T1.5 re-test is reproducible from the launch command alone.')
    _hp('--adam-every-n', ADAM_EVERY_N, int, 'T1.4: update the AdamW parameter groups (wte, lm_head, value-embedding tables, scalars, local_conv) only every Nth step, accumulating their gradients in between and skipping their all_reduce on the other steps. Muon groups still update every step. 1 (default) = historical behaviour, bit-identical.')
    parser.add_argument('--adam-every-n-lr-comp', action=argparse.BooleanOptionalAction, default=os.environ.get('NEURON_COMPETITION_R1_ADAM_EVERY_N_LR_COMP', '1' if ADAM_EVERY_N_LR_COMP else '0') != '0', help="T1.4: multiply the AdamW groups' lr by --adam-every-n. Adam's update is scale-invariant in the gradient, so updating N times less often otherwise moves those tensors ~N times less far in total; this separates 'less often' from 'less far'. Default OFF.")
    args = parser.parse_args()
    if args.ngram_rms_fp32 and args.ngram_ve_opt != 'rmsprop':
        parser.error('--ngram-rms-fp32 requires --ngram-ve-opt rmsprop')
    if args.nki_ngram_rms and (not args.ngram_rms_fp32):
        parser.error('--nki-ngram-rms requires --ngram-rms-fp32')
    if args.eval_seq_len:
        if args.eval_seq_len > args.seq_len:
            parser.error(f"--eval-seq-len {args.eval_seq_len} > --seq-len {args.seq_len}: the live model's rotary/mask tables are only built out to --seq-len. Use `prepare.py eval-public --seq-len N` on the saved checkpoint instead.")
        if args.eval_seq_len % 128 != 0:
            parser.error('--eval-seq-len must be a multiple of 128 (NKI tiling).')
    if args.adam_every_n < 1:
        parser.error('--adam-every-n must be >= 1')
    if args.clip_after_reduce and args.adam_every_n > 1:
        parser.error('--clip-after-reduce requires --adam-every-n 1: deferred Adam gradients would mix unreduced local gradients with averaged gradients in the norm.')
    if args.adam_every_n > 1 and args.grad_bucket:
        parser.error("--adam-every-n > 1 is incompatible with --grad-bucket: the flat buffer is zeroed wholesale every step, so it cannot leave the AdamW groups' gradients standing to accumulate. (--grad-bucket is also a known hang on this hardware; see trn2-experiment-results.md.)")
    global USE_NKI_RELU2, USE_NKI_ROPE_NORM, USE_NKI_SOFTCAP_CE, USE_NKI_MUON, USE_MUON_LR_FP32, USE_NKI_ATTN
    global LOSS_BYTE_WEIGHT, BYTE_LOSS_W
    ADAM_EVERY_N = args.adam_every_n
    ADAM_EVERY_N_LR_COMP = args.adam_every_n_lr_comp
    GRAD_CLIP = args.grad_clip
    CLIP_AFTER_REDUCE = args.clip_after_reduce
    LOSS_SCALE = args.loss_scale
    if not LOSS_SCALE > 0:
        raise ValueError(f'--loss-scale must be positive, got {LOSS_SCALE}')
    LOSS_SCALE_DYNAMIC = args.loss_scale_dynamic
    loss_scale_clean = 0
    loss_scale_skips = 0
    PACK_FACTOR = args.pack_factor
    SEQ_LEN = args.seq_len
    DEPTH = args.depth
    N_EMBD = args.n_embd
    HEAD_DIM = args.head_dim
    MLP_RATIO = args.mlp_ratio
    ATTN_SCALE = args.attn_scale
    N_KV_HEADS = args.n_kv_heads
    MATRIX_LR = args.matrix_lr
    EMBEDDING_LR = args.embedding_lr
    UNEMBEDDING_LR = args.unembedding_lr
    SCALAR_LR = args.scalar_lr
    WEIGHT_DECAY = args.weight_decay
    WD_DECAY_TO_ZERO = args.wd_decay_to_zero
    FINAL_LR_FRAC = args.final_lr_frac
    WARMDOWN_RATIO = args.warmdown_ratio
    INIT_SCALE = args.init_scale
    ATTN_CPROJ_INIT_SCALE = args.attn_cproj_init
    LOGIT_SOFTCAP = args.logit_softcap
    MUON_NS_STEPS = args.muon_ns_steps
    MUON_MOMENTUM_COOLDOWN_FRAC = args.muon_momentum_cooldown_frac
    MUON_MOMENTUM_FINAL = args.muon_momentum_final
    USE_VE = args.ve
    USE_LOCAL_CONV = args.local_conv
    USE_NKI_LOCAL_CONV = args.nki_local_conv
    LOCAL_CONV_KERNEL = args.local_conv_kernel
    TRAIN_LOCAL_CONV = args.train_local_conv
    USE_NGRAM_VE = args.ngram_ve
    NGRAM_VE_LAYERS = args.ngram_ve_layers
    NGRAM_VE_TABLE_MULT = args.ngram_ve_table_mult
    NGRAM_VE_DIM = args.ngram_ve_dim
    NGRAM_VE_TRIGRAM = args.ngram_ve_trigram
    NGRAM_VE_BF16 = args.ngram_ve_bf16
    NGRAM_RMS_FP32 = args.ngram_rms_fp32
    USE_NKI_NGRAM_RMS = args.nki_ngram_rms
    NGRAM_VE_OPT = args.ngram_ve_opt
    NGRAM_VE_LR = args.ngram_ve_lr
    NGRAM_VE_BETA2 = args.ngram_ve_beta2
    NGRAM_VE_FLAT_LR = args.ngram_ve_flat_lr
    NGRAM_VE_CLIP_EXCLUDE = args.ngram_ve_clip_exclude
    USE_BYTE_WTE_INIT = args.byte_wte_init
    BYTE_WTE_INIT_MIX = args.byte_wte_init_mix
    BYTE_WTE_INIT_NGRAM = args.byte_wte_init_ngram
    BYTE_WTE_INIT_STD = args.byte_wte_init_std
    if not 0.0 <= BYTE_WTE_INIT_MIX <= 1.0:
        parser.error(f'--byte-wte-init-mix must be in [0, 1], got {BYTE_WTE_INIT_MIX}')
    if BYTE_WTE_INIT_NGRAM < 1:
        parser.error(f'--byte-wte-init-ngram must be >= 1, got {BYTE_WTE_INIT_NGRAM}')
    USE_COALESCED_SWIGLU = args.coalesced_swiglu
    COALESCED_SWIGLU_HIDDEN = args.coalesced_swiglu_hidden
    if USE_COALESCED_SWIGLU:
        if USE_GATED_MLP:
            parser.error('--coalesced-swiglu and USE_GATED_MLP are two different SwiGLU MLPs and both own mlp.c_fc/c_gate; pick one (USE_GATED_MLP has no flag, so this can only fire if the module constant was edited). --coalesced-swiglu is the parameter-matched, single-matmul form.')
        if args.parallel_block and args.parallel_block_coalesce != 'none':
            parser.error(f'--coalesced-swiglu is incompatible with --parallel-block-coalesce {args.parallel_block_coalesce}: block.c_in is sized mlp_ratio*n_embd and hands mlp.forward a single pre-activation, not a gate/up pair.')
        if COALESCED_SWIGLU_HIDDEN % 128 != 0:
            parser.error(f'--coalesced-swiglu-hidden must be a multiple of 128 (0 = auto), got {COALESCED_SWIGLU_HIDDEN}')
    USE_CAUTIOUS_WD = args.cautious_wd
    SCALAR_BETA1 = args.scalar_beta1
    SCALAR_BETA2 = args.scalar_beta2
    DEMON_BETA1 = args.demon_beta1
    DEMON_BETA1_FINAL = args.demon_beta1_final
    DEMON_BETA1_REF = args.demon_beta1_ref
    DEMON_BETA1_START_FRAC = args.demon_beta1_start_frac
    MUON_BETA2 = args.muon_beta2
    MUON_BETA2_FINAL = args.muon_beta2_final
    MUON_DEPTH_LR_BOTTOM = args.muon_depth_lr_bottom
    MUON_DEPTH_LR_TOP = args.muon_depth_lr_top
    MUON_DEPTH_MOM_BOTTOM = args.muon_depth_mom_bottom
    MUON_DEPTH_MOM_TOP = args.muon_depth_mom_top
    MUON_DEPTH_MOM_REF = args.muon_depth_mom_ref
    if MUON_BETA2_FINAL > 0.0 and (not 0.0 < MUON_BETA2_FINAL < 1.0):
        parser.error(f'--muon-beta2-final must be in (0, 1) when set, got {MUON_BETA2_FINAL}')
    if (MUON_DEPTH_MOM_BOTTOM > 0.0) != (MUON_DEPTH_MOM_TOP > 0.0):
        parser.error('--muon-depth-mom-bottom and --muon-depth-mom-top must both be set or both left at 0.0 (the OFF sentinel is the pair, not either one)')
    USE_OUT_BIGRAM = args.out_bigram
    OUT_BIGRAM_BF16 = args.out_bigram_bf16
    OUT_BIGRAM_GATE_BIAS = args.out_bigram_gate_bias
    OUT_BIGRAM_LR = args.out_bigram_lr
    RELU2_TAU = args.relu2_tau
    USE_MLP_SANDWICH_NORM = args.mlp_sandwich_norm
    USE_HEAD_GATE = args.head_gate
    HEAD_GATE_NORM = args.head_gate_norm
    HEAD_GATE_CHANNELS = args.head_gate_channels
    HEAD_GATE_MUL = args.head_gate_mul
    HEAD_GATE_BIAS = args.head_gate_bias
    USE_OUT_POOL = args.out_pool
    OUT_POOL_LAYERS = args.out_pool_layers
    USE_QK_SHIFT = args.qk_shift
    QK_SHIFT_BETA = args.qk_shift_beta
    QK_SHIFT_FREEZE = args.qk_shift_freeze
    USE_X0_GATE = args.x0_gate
    USE_BLOCK_NUDGE = args.block_nudge
    BLOCK_NUDGE_SITE = args.block_nudge_site
    USE_QKV_NORM_CSE = args.qkv_norm_cse
    USE_ENGRAM = args.engram
    ENGRAM_SITE = args.engram_site
    ENGRAM_ORDERS = args.engram_orders
    ENGRAM_HEADS = args.engram_heads
    ENGRAM_TABLE_MULT = args.engram_table_mult
    ENGRAM_MEM_DIM = args.engram_mem_dim
    ENGRAM_KEY_DIM = args.engram_key_dim
    ENGRAM_CONV_KERNEL = args.engram_conv_kernel
    ENGRAM_SHARE_NGRAM_TABLES = args.engram_share_ngram_tables
    ENGRAM_BF16 = args.engram_bf16
    if USE_ENGRAM:
        engram_orders()
        if ENGRAM_HEADS < 1:
            raise ValueError(f'--engram-heads must be >= 1, got {ENGRAM_HEADS}')
        if ENGRAM_TABLE_MULT < 1 or ENGRAM_MEM_DIM < 1:
            raise ValueError(f'--engram-table-mult and --engram-mem-dim must both be >= 1, got {ENGRAM_TABLE_MULT} and {ENGRAM_MEM_DIM}')
        if ENGRAM_CONV_KERNEL < 0:
            raise ValueError(f'--engram-conv-kernel must be >= 0 (0 disables), got {ENGRAM_CONV_KERNEL}')
        if ENGRAM_SHARE_NGRAM_TABLES and (not USE_NGRAM_VE):
            raise ValueError('--engram-share-ngram-tables needs --ngram-ve: there are no gathered n-gram rows to share otherwise.')
    USE_EPOCH_SHUFFLE = args.epoch_shuffle
    EPOCH_SHUFFLE_SEED = args.epoch_shuffle_seed
    USE_PARALLEL_BLOCK = args.parallel_block
    PARALLEL_BLOCK_COALESCE = args.parallel_block_coalesce
    if PARALLEL_BLOCK_COALESCE != 'none' and (not USE_PARALLEL_BLOCK):
        raise ValueError('--parallel-block-coalesce needs --parallel-block: the coalesced matmul is only valid when all four input projections read the same normed residual, which is exactly what the parallel block makes true.')
    if PARALLEL_BLOCK_COALESCE != 'none' and (USE_FUSED_QKV or USE_GATED_MLP):
        raise ValueError('--parallel-block-coalesce is incompatible with USE_FUSED_QKV (it is the same idea, done differently) and with USE_GATED_MLP (c_gate would still need its own matmul over the same input, which defeats the coalescing).')
    if USE_PARALLEL_BLOCK:
        for name, on in (('--qk-shift', USE_QK_SHIFT), ('--block-nudge', USE_BLOCK_NUDGE), ('--mlp-sandwich-norm', USE_MLP_SANDWICH_NORM)):
            if on:
                raise ValueError(f'--parallel-block and {name} are incompatible: the parallel block replaces the whole serial attn->MLP body that {name} modifies, so {name} would be silently ignored.')
    if USE_BLOCK_NUDGE and USE_QK_SHIFT and (BLOCK_NUDGE_SITE == 'qk'):
        raise ValueError('--block-nudge --block-nudge-site qk and --qk-shift both rewrite q_in/k_in; the nudge exists to REPLACE the frozen qk-shift, so pick one.')
    if USE_X0_GATE and USE_SELECTIVE_GATE:
        raise ValueError('--x0-gate and USE_SELECTIVE_GATE both rewrite the same resid/x0 mix line in Block.forward; pick one.')
    if RELU2_TAU != 0.0 and args.nki_relu2:
        raise ValueError('--relu2-tau is not implemented by the example NKI relu2 kernel; run it with --no-nki-relu2 (the default).')
    if OUT_POOL_LAYERS < 1:
        raise ValueError(f'--out-pool-layers must be >= 1, got {OUT_POOL_LAYERS}')
    if HEAD_GATE_CHANNELS < 1:
        raise ValueError(f'--head-gate-channels must be >= 1, got {HEAD_GATE_CHANNELS}')
    BATCH_RAMP = args.batch_ramp
    BATCH_RAMP_GRAD_ACCUM = tuple((int(v) for v in args.batch_ramp_grad_accum.split(',')))
    BATCH_RAMP_BOUNDS = tuple((float(v) for v in args.batch_ramp_bounds.split(',')))
    BATCH_RAMP_LR_EXP = args.batch_ramp_lr_exp
    if len(BATCH_RAMP_GRAD_ACCUM) != 3 or min(BATCH_RAMP_GRAD_ACCUM) < 1:
        raise ValueError(f'--batch-ramp-grad-accum needs three counts >= 1, got {BATCH_RAMP_GRAD_ACCUM}')
    if len(BATCH_RAMP_BOUNDS) != 2 or not 0.0 < BATCH_RAMP_BOUNDS[0] <= BATCH_RAMP_BOUNDS[1] < 1.0:
        raise ValueError(f'--batch-ramp-bounds needs 0 < A->B <= B->C < 1, got {BATCH_RAMP_BOUNDS}')
    USE_WEIGHT_EMA = args.weight_ema
    WEIGHT_EMA_DECAYS = tuple((float(v) for v in args.weight_ema_decay.split(',')))
    WEIGHT_EMA_START_FRAC = args.weight_ema_start_frac
    if not WEIGHT_EMA_DECAYS or not all((0.0 < d < 1.0 for d in WEIGHT_EMA_DECAYS)):
        raise ValueError(f'--weight-ema-decay needs one or more values in (0,1), got {WEIGHT_EMA_DECAYS}')
    if not 0.0 <= WEIGHT_EMA_START_FRAC < 1.0:
        raise ValueError(f'--weight-ema-start-frac must be in [0,1), got {WEIGHT_EMA_START_FRAC}')
    LRM_QUANTUM = args.lrm_quantum
    USE_TENSOR_LR_SCALARS = args.tensor_lr_scalars
    USE_FOREACH_OPTIM = args.foreach_optim
    USE_NKI_RELU2 = args.nki_relu2
    USE_NKI_ROPE_NORM = args.nki_rope_norm
    USE_NKI_SOFTCAP_CE = args.nki_softcap_ce
    LOSS_BYTE_WEIGHT = args.byte_loss_weight
    USE_NKI_MUON = args.nki_muon
    USE_NKI_ATTN = args.nki_attn
    COMPILE_SDPA_DIRECT = args.compile_sdpa_direct
    USE_BF16_NORM_OUTPUT = args.bf16_norm_output
    USE_BF16_SDPA_INPUT = args.bf16_sdpa_input
    USE_MUON_LR_FP32 = args.muon_lr_fp32
    USE_TRANSPOSED_LINEAR = args.transposed_linear
    USE_COMPILE_MUON = args.compile_muon
    if USE_COMPILE_MUON and USE_NKI_MUON:
        USE_NKI_MUON = False
        print0('muon step: --compile-muon supersedes --nki-muon (a compiled region cannot contain an nki.jit call); NKI Muon disabled for this run')
    device_type = detect_device_type(args.device_type)
    if USE_NKI_NGRAM_RMS and (device_type != 'neuron' or not _HAS_NKI):
        parser.error('--nki-ngram-rms requires a Neuron device and available NKI backend')
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type, seed=args.seed)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    print0(f'device={device} dtype={COMPUTE_DTYPE} world_size={world_size}')
    print0(f'training target={args.num_steps} steps max_seconds={args.max_train_seconds} official_budget={TRAIN_TIME_BUDGET_SECONDS}s')
    print0(f'gradient_clip={GRAD_CLIP:g} clip_after_reduce={CLIP_AFTER_REDUCE}')
    print0(f'attention: nki_attn={USE_NKI_ATTN} compile_sdpa_direct={COMPILE_SDPA_DIRECT} bf16_sdpa_input={USE_BF16_SDPA_INPUT}')
    print0(f'branch_precision: bf16_norm_output={USE_BF16_NORM_OUTPUT} nki_local_conv={USE_NKI_LOCAL_CONV}')
    if args.num_steps < args.max_train_seconds:
        print0(f'WARNING: --num-steps {args.num_steps} is below max_train_seconds={args.max_train_seconds}s and will almost certainly stop training before the time budget (the LR schedule would end without cooldown). For a scored submission leave --num-steps at its default ({NUM_STEPS}).')
    tokenizer = ensure_tokenizer(build_if_missing=False)
    global BOS_TOKEN_ID
    BOS_TOKEN_ID = tokenizer.get_bos_token_id()
    config = build_config()
    print0(f'model_config={json.dumps(asdict(config), indent=2)}')
    model = GPT(config)
    model.init_weights()
    if USE_BYTE_WTE_INIT:
        n_ok, n_skip, feats = byte_feature_wte_init(model.transformer.wte.weight, tokenizer, config.vocab_size)
        print0(f'byte-wte-init: {n_ok} rows from byte features, {n_skip} left random (mix={BYTE_WTE_INIT_MIX} max_n={BYTE_WTE_INIT_NGRAM} std={BYTE_WTE_INIT_STD} mean_features_per_token={feats:.1f})')
    if args.init_from:
        warm_start_from(args.init_from, model)
    model.to(device)
    if LOSS_BYTE_WEIGHT:
        tb = get_token_bytes('cpu').to(torch.float32)
        tb = torch.clamp(tb, min=1.0)
        BYTE_LOSS_W = tb / tb.mean()
        print0(f'byte-loss-weight: mean_bytes={tb.mean().item():.3f} max_bytes={tb.max().item():.0f} weight_range=({BYTE_LOSS_W.min().item():.3f}, {BYTE_LOSS_W.max().item():.3f})')
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
    if args.compile_only:
        synthetic_x = torch.zeros((DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN), dtype=torch.int32).to(device)
        synthetic_y = torch.zeros((DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN), dtype=torch.long).to(device)
        optimizer = model.setup_optimizer()
        model.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            loss = model(synthetic_x, synthetic_y)
            (loss / grad_accum_steps).backward()
        clip_main, clip_ngram = split_clip_params(orig_model)
        clip_and_sync_gradients(model, clip_main, clip_ngram, max_norm=GRAD_CLIP, clip_after_reduce=CLIP_AFTER_REDUCE, foreach=args.clip_foreach)
        optimizer.step()
        synchronize(device)
        print0('compile-only synthetic training step completed')
        cleanup_runtime()
        return
    loader_seq_len = PACK_FACTOR * SEQ_LEN
    if USE_EPOCH_SHUFFLE:
        train_loader = make_dataloader_requeue(tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, 'train', device, _document_batches_fn=_document_batches_epoch_shuffle)
    else:
        train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, 'train', device)

    def fetch():
        xb, yb, state = next(train_loader)
        if PACK_FACTOR > 1:
            xb = xb.reshape(DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN)
            yb = yb.reshape(DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN)
        return (xb, yb, state)
    x, y, loader_state = fetch()
    optimizer = model.setup_optimizer()
    grad_flat_buffer = None
    if args.grad_bucket:
        grad_flat_buffer, _grad_bucket_params = build_flat_grad_buffer(model)
        print0(f'grad_bucket enabled: {len(_grad_bucket_params)} params, {grad_flat_buffer.numel():,} floats in one flat fp32 buffer')
    adamw_skip_ids: frozenset[int] = frozenset()
    if ADAM_EVERY_N > 1:
        adamw_skip_ids = optimizer.adamw_param_ids()
        if ADAM_EVERY_N_LR_COMP:
            for group in optimizer.param_groups:
                if group['kind'] == 'adamw':
                    group['initial_lr'] = group['initial_lr'] * ADAM_EVERY_N
        print0(f'adam_every_n={ADAM_EVERY_N} lr_comp={ADAM_EVERY_N_LR_COMP}: {len(adamw_skip_ids)} AdamW params update every {ADAM_EVERY_N} steps')
    cached_linears = [m for m in model.modules() if isinstance(m, Linear)] if USE_WEIGHT_CACHE else []
    clip_main_params, clip_ngram_params = split_clip_params(orig_model)
    assert not (grad_flat_buffer is not None and clip_ngram_params), '--grad-bucket clips one flat buffer and cannot exclude the n-gram tables; use --no-ngram-ve-clip-exclude to combine them'

    def debug_numerics(stage, step_number, loss, inspect_tensors=False):
        if step_number >= args.debug_numerics_steps:
            return
        synchronize(device)
        record = dict(rank=rank, step=step_number, stage=stage, loss=None if loss is None else float(loss.item()), timing_valid=False)
        if inspect_tensors:
            named = [(n, p) for n, p in orig_model.named_parameters() if p.requires_grad]
            names = ['param:' + n for n, _ in named]
            tensors = [p.detach() for _, p in named]
            names += ['grad:' + n for n, p in named if p.grad is not None]
            tensors += [p.grad.detach() for _, p in named if p.grad is not None]
            flags = torch.stack([torch.isfinite(t).all() for t in tensors]).cpu().tolist()
            record['nonfinite'] = [n for n, ok in zip(names, flags) if not ok]
            record['missing_gradients'] = [n for n, p in named if p.grad is None]
            record['checked_tensors'] = len(tensors)
        print('NUMERICS_JSON ' + json.dumps(record), flush=True)
    loop_started = time.monotonic()
    budget_started = loop_started
    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0
    recent_step_times: list[float] = []
    startup_allowance_est = 0.0
    inv_gas_host = torch.empty((), dtype=torch.float32)
    inv_gas_dev = torch.zeros((), dtype=torch.float32, device=device)
    grad_accum_now = grad_accum_steps
    ramp_stage_logged = -1
    ema_names = [n for n, _ in orig_model.named_parameters()]
    ema_live = [p for _, p in orig_model.named_parameters()]
    ema_shadows: list[list[torch.Tensor]] | None = None
    ema_updates = 0
    ema_lerp_ws = [1.0 - d for d in WEIGHT_EMA_DECAYS]
    while step < args.num_steps:
        elapsed = time.monotonic() - budget_started
        ramp_progress = min(1.0, elapsed / max(1e-06, args.max_train_seconds))
        grad_accum_now, batch_lr_scale = batch_ramp_grad_accum(ramp_progress, grad_accum_steps)
        if BATCH_RAMP:
            inv_gas_host.fill_(1.0 / grad_accum_now)
            inv_gas_dev.copy_(inv_gas_host)
            stage = batch_ramp_stage(ramp_progress)
            if stage != ramp_stage_logged:
                ramp_stage_logged = stage
                print0(f"batch_ramp: step {step} progress {ramp_progress:.3f} -> stage {'ABC'[stage]} grad_accum={grad_accum_now} tokens={grad_accum_now * world_tokens_per_micro:,} lr_scale={batch_lr_scale:.4f}")
        synchronize(device)
        t0 = time.monotonic()
        adam_step_now = (step + 1) % ADAM_EVERY_N == 0
        zero_adamw_now = step % ADAM_EVERY_N == 0
        if grad_flat_buffer is not None:
            grad_flat_buffer.zero_()
        elif zero_adamw_now:
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad_selective(zero_adamw=False)
        last_loss = None
        for micro_index in range(grad_accum_now):
            loss = model(x, y)
            last_loss = loss.detach()
            debug_numerics(f'micro{micro_index}_forward', step, last_loss)
            ((loss * inv_gas_dev if BATCH_RAMP else loss / grad_accum_steps) * LOSS_SCALE).backward()
            debug_numerics(f'micro{micro_index}_backward', step, last_loss, inspect_tensors=micro_index + 1 == grad_accum_now)
            x, y, loader_state = fetch()
        skip_step = False
        unscale_with = LOSS_SCALE
        if LOSS_SCALE_DYNAMIC:
            finite = grads_finite_(model, grad_flat_buffer)
            ov = torch.tensor([0.0 if finite else 1.0], device=device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(ov, op=dist.ReduceOp.MAX)
            new_scale, new_clean, skip_step = loss_scale_transition(LOSS_SCALE, loss_scale_clean, bool(ov[0].item() == 0.0), LOSS_SCALE_GROWTH_INTERVAL, LOSS_SCALE_MAX)
            if skip_step:
                loss_scale_skips += 1
                print0(f'loss-scale: overflow at step {step}, skipping update (scale {LOSS_SCALE:g} -> {new_scale:g}, skips {loss_scale_skips})', flush=True)
                if grad_flat_buffer is not None:
                    grad_flat_buffer.zero_()
                else:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.detach().zero_()
            LOSS_SCALE, loss_scale_clean = (new_scale, new_clean)
        unscale_grads_(model, grad_flat_buffer, unscale_with)
        clip_and_sync_gradients(model, clip_main_params, clip_ngram_params, max_norm=GRAD_CLIP, clip_after_reduce=CLIP_AFTER_REDUCE, grad_flat_buffer=grad_flat_buffer, foreach=args.clip_foreach, skip_param_ids=None if adam_step_now else adamw_skip_ids)
        debug_numerics('clip_and_reduce', step, last_loss, inspect_tensors=True)
        lrm = lr_multiplier(step, elapsed=elapsed, max_train_seconds=args.max_train_seconds)
        lrm_flat = min(1.0, (step + 1) / max(1, WARMUP_STEPS))
        wd_progress = min(1.0, elapsed / max(1e-06, args.max_train_seconds))
        demon_b1 = demon_beta1_scale(wd_progress)
        muon_b2 = muon_beta2(wd_progress)
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * batch_lr_scale * (lrm_flat if group.get('no_lr_decay') else lrm)
            if group['kind'] == 'muon':
                group['momentum'] = muon_momentum(step, wd_progress)
                group['weight_decay'] = WEIGHT_DECAY * (1.0 - wd_progress) if WD_DECAY_TO_ZERO else WEIGHT_DECAY
                group['beta2'] = muon_b2
            elif group['kind'] == 'adamw' and DEMON_BETA1:
                _b1, _b2 = group['initial_betas']
                group['betas'] = (_b1 * demon_b1, _b2)
        if not skip_step:
            optimizer.step(update_adamw=adam_step_now)
        debug_numerics('optimizer', step, last_loss, inspect_tensors=True)
        if USE_WEIGHT_EMA and ramp_progress >= WEIGHT_EMA_START_FRAC:
            with torch.no_grad():
                if ema_shadows is None:
                    ema_shadows = [[p.detach().clone() for p in ema_live] for _ in ema_lerp_ws]
                    print0(f'weight_ema: seeded at step {step} progress {ramp_progress:.3f} decays={WEIGHT_EMA_DECAYS} tensors={len(ema_live)} each')
                else:
                    for shadow, w in zip(ema_shadows, ema_lerp_ws):
                        torch._foreach_lerp_(shadow, ema_live, w)
                    ema_updates += 1
        for m in cached_linears:
            m.refresh_weight_cache(COMPUTE_DTYPE)
        debug_numerics('weight_cache', step, last_loss)
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
        if step_cap_hit and (not time_budget_hit) and (rank == 0):
            progress_at_exit = elapsed_after_step / max(1e-06, args.max_train_seconds)
            if STEP_CAP_GUARD_MIN_PROGRESS <= progress_at_exit < STEP_CAP_GUARD_MAX_PROGRESS:
                print(f'WARNING: --num-steps {args.num_steps} step cap stopped training at step {step + 1}, elapsed={elapsed_after_step:.0f}s -- only {progress_at_exit:.1%} of max_train_seconds={args.max_train_seconds:.0f}s. The wall-clock LR cooldown (lr_multiplier) did NOT run to completion. This is the 2026-09-10 submission-gap failure mode (research/submission-gap-analysis-2026-09-10.md) -- if this is meant to be a real/scored run, raise --num-steps (default {NUM_STEPS}); if this is an intentional short/smoke run, ignore.', flush=True)
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
        if step == 50 and rank == 0:
            window = sorted(recent_step_times[-20:])
            median_dt = window[len(window) // 2]
            time_to_step_cap = (args.num_steps - step - 1) * median_dt
            time_left = args.max_train_seconds - (time.monotonic() - budget_started)
            if time_to_step_cap < time_left:
                print(f'WARNING: at {median_dt:.2f}s/step the --num-steps {args.num_steps} cap will stop training ~{time_left - time_to_step_cap:.0f}s BEFORE max_train_seconds={args.max_train_seconds}s; the wall-clock LR schedule will end without cooldown. Intentional for fixed-step sweeps, wrong for a scored submission.', flush=True)
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
        save_checkpoint(checkpoint_path, orig_model, {'step': step, 'total_train_tokens': total_train_tokens, 'budget_elapsed_before_save': budget_elapsed_before_save, 'world_size': world_size, 'compiled': args.compile, 'weight_ema_updates': ema_updates, 'weight_ema_decay': WEIGHT_EMA_DECAYS[0] if ema_shadows is not None else None}, ema_state=weight_ema_state(orig_model, ema_names, ema_shadows[0]) if ema_shadows is not None else None)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    budget_elapsed_after_save = time.monotonic() - budget_started
    public_eval = None
    public_eval_alt = None
    causal = None
    if args.eval_public:
        causal = causality_check(orig_model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, device)
        public_eval = evaluate_bpb(orig_model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, device, split='public_val', max_tokens=args.eval_tokens, timeout_seconds=EVAL_TIMEOUT_SECONDS)
        if args.eval_seq_len:
            public_eval_alt = evaluate_bpb(orig_model, tokenizer, DEVICE_BATCH_SIZE, args.eval_seq_len, device, split='public_val', max_tokens=args.eval_tokens, timeout_seconds=EVAL_TIMEOUT_SECONDS)
    ema_evals: list[tuple[float, dict, dict | None]] = []
    ema_eval_seconds = 0.0
    if args.eval_public and ema_shadows is not None:
        ema_eval_started = time.monotonic()
        for decay, shadow in zip(WEIGHT_EMA_DECAYS, ema_shadows):
            with torch.no_grad():
                for p, s in zip(ema_live, shadow):
                    p.copy_(s)
            for m in cached_linears:
                m.refresh_weight_cache(COMPUTE_DTYPE)
            main = evaluate_bpb(orig_model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, device, split='public_val', max_tokens=args.eval_tokens, timeout_seconds=EVAL_TIMEOUT_SECONDS)
            alt = None
            if args.eval_seq_len:
                alt = evaluate_bpb(orig_model, tokenizer, DEVICE_BATCH_SIZE, args.eval_seq_len, device, split='public_val', max_tokens=args.eval_tokens, timeout_seconds=EVAL_TIMEOUT_SECONDS)
            ema_evals.append((decay, main, alt))
        ema_eval_seconds = time.monotonic() - ema_eval_started
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
        print(f'nki_relu2:         {args.nki_relu2}')
        print(f'nki_rope_norm:     {args.nki_rope_norm}')
        print(f'nki_softcap_ce:    {args.nki_softcap_ce}')
        print(f'transposed_linear: {USE_TRANSPOSED_LINEAR}')
        print(f'nki_muon:          {USE_NKI_MUON}')
        print(f'compile_muon:      {USE_COMPILE_MUON}')
        print(f'muon_lr_fp32:      {USE_MUON_LR_FP32}')
        print(f'ngram_ve:          {args.ngram_ve} tables={len(orig_model.ngram_embeds)} rows={orig_model.ngram_rows} dim={orig_model.ngram_dim} index_gather={NGRAM_VE_INDEX_GATHER} trigram={args.ngram_ve_trigram} opt={args.ngram_ve_opt} flat_lr={args.ngram_ve_flat_lr} bf16={args.ngram_ve_bf16} rms_fp32_state_math={NGRAM_RMS_FP32} nki_rms={USE_NKI_NGRAM_RMS}')
        print(f'out_bigram:        {USE_OUT_BIGRAM} bf16={OUT_BIGRAM_BF16} gate_bias={OUT_BIGRAM_GATE_BIAS} lr={OUT_BIGRAM_LR}')
        print(f'oneliners:         relu2_tau={RELU2_TAU} mlp_sandwich={USE_MLP_SANDWICH_NORM} head_gate={USE_HEAD_GATE} hg_norm={HEAD_GATE_NORM} hg_ch={HEAD_GATE_CHANNELS} hg_mul={HEAD_GATE_MUL} hg_bias={HEAD_GATE_BIAS} out_pool={USE_OUT_POOL} pool_layers={OUT_POOL_LAYERS} qk_shift={USE_QK_SHIFT} qk_beta={QK_SHIFT_BETA} qk_freeze={QK_SHIFT_FREEZE} x0_gate={USE_X0_GATE} block_nudge={USE_BLOCK_NUDGE} nudge_site={BLOCK_NUDGE_SITE} qkv_norm_cse={USE_QKV_NORM_CSE} parallel_block={USE_PARALLEL_BLOCK} par_coalesce={PARALLEL_BLOCK_COALESCE} epoch_shuffle={USE_EPOCH_SHUFFLE} epoch_shuffle_seed={EPOCH_SHUFFLE_SEED}')
        print(f'engram:            {USE_ENGRAM} site={orig_model.engram_site} orders={ENGRAM_ORDERS} heads={ENGRAM_HEADS} tables={(0 if orig_model.engram is None else len(orig_model.engram.tables))} mult={ENGRAM_TABLE_MULT} mem_dim={ENGRAM_MEM_DIM} e_dim={(0 if orig_model.engram is None else orig_model.engram.e_dim)} key_dim={(0 if orig_model.engram is None else orig_model.engram.key_dim)} conv_k={ENGRAM_CONV_KERNEL} share={ENGRAM_SHARE_NGRAM_TABLES} bf16={ENGRAM_BF16}')
        print(f'batch_ramp:        {BATCH_RAMP} grad_accum={BATCH_RAMP_GRAD_ACCUM} bounds={BATCH_RAMP_BOUNDS} lr_exp={BATCH_RAMP_LR_EXP} ref_grad_accum={grad_accum_steps} last_grad_accum={grad_accum_now}')
        print(f'weight_ema:        {USE_WEIGHT_EMA} decays={WEIGHT_EMA_DECAYS} start_frac={WEIGHT_EMA_START_FRAC} seeded={ema_shadows is not None} updates={ema_updates}')
        print(f'sched_tricks:      demon_b1={DEMON_BETA1} demon_final={DEMON_BETA1_FINAL} demon_ref={DEMON_BETA1_REF} demon_start={DEMON_BETA1_START_FRAC} muon_b2={MUON_BETA2} muon_b2_final={MUON_BETA2_FINAL} depth_lr={MUON_DEPTH_LR_BOTTOM}->{MUON_DEPTH_LR_TOP} depth_mom={MUON_DEPTH_MOM_BOTTOM}->{MUON_DEPTH_MOM_TOP} depth_mom_ref={MUON_DEPTH_MOM_REF}')
        print(f'num_steps:         {step}')
        print(f'total_tokens_M:    {total_train_tokens / 1000000.0:.1f}')
        print(f'world_size:        {world_size}')
        print(f"lnc:               {os.environ.get('NEURON_LOGICAL_NC_CONFIG', '')}")
        print(f'seq_len:           {SEQ_LEN}')
        print(f'num_params_M:      {num_params / 1000000.0:.1f}')
        if causal is not None:
            print(f"causality_passed:  {causal['passed']}")
            print(f"causality_max_abs: {causal['max_abs_diff']:.6f}")
        if public_eval is not None:
            print(f"public_val_bpb:    {public_eval['val_bpb']:.6f}")
            print(f"eval_seconds:      {public_eval['eval_seconds']:.1f}")
            print(f"eval_timeout:      {public_eval['eval_timeout']}")
            print(f"eval_over_timeout: {public_eval['eval_over_timeout']}")
        if public_eval_alt is not None:
            print(f'eval_alt_seq_len:  {args.eval_seq_len}')
            print(f"public_val_bpb_alt: {public_eval_alt['val_bpb']:.6f}")
            print(f"eval_alt_seconds:  {public_eval_alt['eval_seconds']:.1f}")
        if ema_evals:
            primary_decay, primary, primary_alt = ema_evals[0]
            print(f"public_val_bpb_ema: {primary['val_bpb']:.6f}")
            print(f"ema_delta:         {primary['val_bpb'] - public_eval['val_bpb']:+.6f}")
            print(f'eval_ema_seconds:  {ema_eval_seconds:.1f}')
            if primary_alt is not None:
                print(f"public_val_bpb_ema_alt: {primary_alt['val_bpb']:.6f}")
                print(f"ema_delta_alt:     {primary_alt['val_bpb'] - public_eval_alt['val_bpb']:+.6f}")
            arms = ' '.join((f"{d}={m['val_bpb']:.6f}/{m['val_bpb'] - public_eval['val_bpb']:+.6f}" + (f"|{a['val_bpb']:.6f}/{a['val_bpb'] - public_eval_alt['val_bpb']:+.6f}" if a is not None else '') for d, m, a in ema_evals))
            print(f"weight_ema_arms:   raw={public_eval['val_bpb']:.6f}" + (f"|{public_eval_alt['val_bpb']:.6f}" if public_eval_alt is not None else '') + f' {arms}')
    cleanup_runtime()
import torch
import torch.nn.functional as F

def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def fused_mlp(x, up, down, lanes: int=2, token_tile: int=512, contiguous_weights: bool=False, tiled_io: bool=False, transposed_relu: bool=False, grouped_io: bool=False, weight_mode: int=0, fp8_weight_cache: bool=False, up_scales=None, down_scales=None):
        tokens, channels = x.shape
        if weight_mode == 3:
            hidden = up.shape[1] * 128
            assert up.shape == (128, hidden // 128, channels // 128, 128)
            assert down.shape == (128, channels // 128, hidden // 128, 128)
        else:
            hidden, up_channels = up.shape
            assert up_channels == channels and down.shape == (channels, hidden)
        assert x.dtype == nl.bfloat16 and up.dtype in (nl.float32, nl.bfloat16)
        assert down.dtype == up.dtype
        assert channels % 128 == 0 and hidden % 128 == 0 and (lanes in (1, 2))
        assert token_tile <= 512 and token_tile % 128 == 0
        assert tokens % (token_tile * lanes) == 0
        assert weight_mode in (0, 1, 2, 3)
        if weight_mode:
            assert grouped_io
        if grouped_io:
            assert contiguous_weights and tiled_io and transposed_relu
            assert up.dtype == nl.bfloat16
        if fp8_weight_cache:
            assert grouped_io and weight_mode == 0
            assert up_scales.shape == (128, 3) and down_scales.shape == (128, 3)
            up_meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
            down_meta = nl.ndarray((128, 3), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=up_meta, src=up_scales)
            nisa.dma_copy(dst=down_meta, src=down_scales)
        cache_dtype = nl.float8_e4m3 if fp8_weight_cache else nl.bfloat16
        nc, nh = (channels // 128, hidden // 128)
        weight_bytes = 2 if fp8_weight_cache else 4
        live_bytes = weight_bytes * channels * hidden + 2 * (channels + hidden) * token_tile
        if weight_mode:
            live_bytes = 2 * (channels + hidden) * token_tile + 4 * 128 * (channels + hidden)
            if weight_mode == 1:
                live_bytes += 2 * channels * hidden
        if grouped_io:
            live_bytes += 2 * channels * token_tile
        assert live_bytes + 2 * 1024 ** 2 <= nl.tile_size.sbuf_size_bytes
        result = nl.ndarray((tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        saved_shape = (hidden, tokens) if transposed_relu else (tokens, hidden)
        saved_relu = nl.ndarray(saved_shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        if weight_mode < 2:
            up_cache = nl.ndarray((128, nh, nc, 128), dtype=cache_dtype, buffer=nl.sbuf)
        if weight_mode == 0:
            down_cache = nl.ndarray((128, nc, nh, 128), dtype=cache_dtype, buffer=nl.sbuf)
        if grouped_io:
            if weight_mode < 2:
                for hi in nl.affine_range(nh):
                    up_rows = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_copy(dst=up_rows, src=up[hi * 128:(hi + 1) * 128, :])
                    for ci in nl.affine_range(nc):
                        up_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=up_t, data=up_rows[:, ci * 128:(ci + 1) * 128])
                        if fp8_weight_cache:
                            nisa.tensor_scalar(dst=up_cache[:, hi, ci, :], data=up_t, op0=nl.multiply, operand0=up_meta[:, 0:1])
                        else:
                            nisa.tensor_copy(dst=up_cache[:, hi, ci, :], src=up_t)
            if weight_mode == 0:
                for ci in nl.affine_range(nc):
                    down_rows = nl.ndarray((128, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_copy(dst=down_rows, src=down[ci * 128:(ci + 1) * 128, :])
                    for hi in nl.affine_range(nh):
                        down_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=down_t, data=down_rows[:, hi * 128:(hi + 1) * 128])
                        if fp8_weight_cache:
                            nisa.tensor_scalar(dst=down_cache[:, ci, hi, :], data=down_t, op0=nl.multiply, operand0=down_meta[:, 0:1])
                        else:
                            nisa.tensor_copy(dst=down_cache[:, ci, hi, :], src=down_t)
        if not grouped_io:
            for hi in nl.affine_range(nh):
                for ci in nl.affine_range(nc):
                    if contiguous_weights:
                        up_raw = nl.ndarray((128, 128), dtype=up.dtype, buffer=nl.sbuf)
                        down_raw = nl.ndarray((128, 128), dtype=down.dtype, buffer=nl.sbuf)
                        up_t = nl.ndarray((128, 128), dtype=up.dtype, buffer=nl.psum)
                        down_t = nl.ndarray((128, 128), dtype=down.dtype, buffer=nl.psum)
                        nisa.dma_copy(dst=up_raw, src=up[hi * 128:(hi + 1) * 128, ci * 128:(ci + 1) * 128])
                        nisa.dma_copy(dst=down_raw, src=down[ci * 128:(ci + 1) * 128, hi * 128:(hi + 1) * 128])
                        nisa.nc_transpose(dst=up_t, data=up_raw)
                        nisa.nc_transpose(dst=down_t, data=down_raw)
                        nisa.tensor_copy(dst=up_cache[:, hi, ci, :], src=up_t)
                        nisa.tensor_copy(dst=down_cache[:, ci, hi, :], src=down_t)
                    else:
                        nisa.dma_copy(dst=up_cache[:, hi, ci, :], src=up.ap(pattern=[[1, 128], [channels, 128]], offset=hi * 128 * channels + ci * 128))
                        nisa.dma_copy(dst=down_cache[:, ci, hi, :], src=down.ap(pattern=[[1, 128], [hidden, 128]], offset=ci * 128 * hidden + hi * 128))
        for mi in nl.affine_range(tokens // (token_tile * lanes)):
            m0 = (mi * lanes + nl.program_id(0)) * token_tile
            inputs = nl.ndarray((128, nc, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
            activated = nl.ndarray((128, nh, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
            if grouped_io:
                for ti in nl.affine_range(token_tile // 128):
                    input_wide = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_copy(dst=input_wide, src=x[m0 + ti * 128:m0 + (ti + 1) * 128, :])
                    for ci in nl.affine_range(nc):
                        input_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=input_t, data=input_wide[:, ci * 128:(ci + 1) * 128])
                        nisa.tensor_copy(dst=inputs[:, ci, ti * 128:(ti + 1) * 128], src=input_t)
            if not grouped_io:
                for ci in nl.affine_range(nc):
                    if tiled_io:
                        for ti in nl.affine_range(token_tile // 128):
                            input_rows = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                            input_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                            nisa.dma_copy(dst=input_rows, src=x[m0 + ti * 128:m0 + (ti + 1) * 128, ci * 128:(ci + 1) * 128])
                            nisa.nc_transpose(dst=input_t, data=input_rows)
                            nisa.tensor_copy(dst=inputs[:, ci, ti * 128:(ti + 1) * 128], src=input_t)
                    else:
                        nisa.dma_copy(dst=inputs[:, ci, :], src=x.ap(pattern=[[1, 128], [channels, token_tile]], offset=m0 * channels + ci * 128))
            for hi in nl.affine_range(nh):
                if weight_mode >= 2:
                    up_block = nl.ndarray((128, nc, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                    if weight_mode == 3:
                        nisa.dma_copy(dst=up_block, src=up[:, hi, :, :])
                    else:
                        up_stream_rows = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.dma_copy(dst=up_stream_rows, src=up[hi * 128:(hi + 1) * 128, :])
                        for ci in nl.affine_range(nc):
                            up_stream_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                            nisa.nc_transpose(dst=up_stream_t, data=up_stream_rows[:, ci * 128:(ci + 1) * 128])
                            nisa.tensor_copy(dst=up_block[:, ci, :], src=up_stream_t)
                up_acc = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.psum)
                for ci in nl.affine_range(nc):
                    if weight_mode >= 2:
                        nisa.nc_matmul(dst=up_acc, stationary=up_block[:, ci, :], moving=inputs[:, ci, :], accumulate=ci > 0)
                    elif fp8_weight_cache:
                        up_dequantized = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=up_dequantized, data=up_cache[:, hi, ci, :], op0=nl.multiply, operand0=up_meta[:, 1:2])
                        nisa.nc_matmul(dst=up_acc, stationary=up_dequantized, moving=inputs[:, ci, :], accumulate=ci > 0)
                    else:
                        nisa.nc_matmul(dst=up_acc, stationary=up_cache[:, hi, ci, :], moving=inputs[:, ci, :], accumulate=ci > 0)
                rounded = nl.ndarray((128, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
                relu = nl.ndarray((128, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rounded, src=up_acc)
                nisa.tensor_scalar(dst=relu, data=rounded, op0=nl.maximum, operand0=0.0)
                nisa.tensor_tensor(dst=activated[:, hi, :], data1=relu, data2=relu, op=nl.multiply)
                if transposed_relu:
                    nisa.dma_copy(dst=saved_relu[hi * 128:(hi + 1) * 128, m0:m0 + token_tile], src=relu)
                elif tiled_io:
                    for ti in nl.affine_range(token_tile // 128):
                        relu_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                        relu_rows = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.nc_transpose(dst=relu_t, data=relu[:, ti * 128:(ti + 1) * 128])
                        nisa.tensor_copy(dst=relu_rows, src=relu_t)
                        nisa.dma_copy(dst=saved_relu[m0 + ti * 128:m0 + (ti + 1) * 128, hi * 128:(hi + 1) * 128], src=relu_rows)
                else:
                    nisa.dma_copy(dst=saved_relu.ap(pattern=[[1, 128], [hidden, token_tile]], offset=m0 * hidden + hi * 128), src=relu)
            if grouped_io:
                wide_output_rows = nl.ndarray((128, token_tile // 128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
            for ci in nl.affine_range(nc):
                if weight_mode:
                    down_block = nl.ndarray((128, nh, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                    if weight_mode == 3:
                        nisa.dma_copy(dst=down_block, src=down[:, ci, :, :])
                    else:
                        down_stream_rows = nl.ndarray((128, hidden), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.dma_copy(dst=down_stream_rows, src=down[ci * 128:(ci + 1) * 128, :])
                        for hi in nl.affine_range(nh):
                            down_stream_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                            nisa.nc_transpose(dst=down_stream_t, data=down_stream_rows[:, hi * 128:(hi + 1) * 128])
                            nisa.tensor_copy(dst=down_block[:, hi, :], src=down_stream_t)
                down_acc = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.psum)
                for hi in nl.affine_range(nh):
                    if weight_mode:
                        nisa.nc_matmul(dst=down_acc, stationary=down_block[:, hi, :], moving=activated[:, hi, :], accumulate=hi > 0)
                    elif fp8_weight_cache:
                        down_dequantized = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=down_dequantized, data=down_cache[:, ci, hi, :], op0=nl.multiply, operand0=down_meta[:, 1:2])
                        nisa.nc_matmul(dst=down_acc, stationary=down_dequantized, moving=activated[:, hi, :], accumulate=hi > 0)
                    else:
                        nisa.nc_matmul(dst=down_acc, stationary=down_cache[:, ci, hi, :], moving=activated[:, hi, :], accumulate=hi > 0)
                output = nl.ndarray((128, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=output, src=down_acc)
                if grouped_io:
                    for ti in nl.affine_range(token_tile // 128):
                        output_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.nc_transpose(dst=output_t, data=output[:, ti * 128:(ti + 1) * 128])
                        nisa.tensor_copy(dst=wide_output_rows[:, ti, ci * 128:(ci + 1) * 128], src=output_t)
                elif tiled_io:
                    for ti in nl.affine_range(token_tile // 128):
                        output_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                        output_rows = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.nc_transpose(dst=output_t, data=output[:, ti * 128:(ti + 1) * 128])
                        nisa.tensor_copy(dst=output_rows, src=output_t)
                        nisa.dma_copy(dst=result[m0 + ti * 128:m0 + (ti + 1) * 128, ci * 128:(ci + 1) * 128], src=output_rows)
                else:
                    nisa.dma_copy(dst=result.ap(pattern=[[1, 128], [channels, token_tile]], offset=m0 * channels + ci * 128), src=output)
            if grouped_io:
                for ti in nl.affine_range(token_tile // 128):
                    nisa.dma_copy(dst=result[m0 + ti * 128:m0 + (ti + 1) * 128, :], src=wide_output_rows[:, ti, :])
        return (result, saved_relu)
    return fused_mlp

def reference_forward(x, up, down):
    relu = F.relu(F.linear(x, up.to(x.dtype)))
    return (F.linear(relu.square(), down.to(x.dtype)), relu)

def prepack_weight(weight):
    """Lossless BF16 layout cache; call after every master-weight update.

    Moving this outside the microbatch is an optimization only when the caller
    measures refresh cost. It creates no new Parameter or checkpoint entry.
    """
    out_features, in_features = weight.shape
    assert out_features % 128 == 0 and in_features % 128 == 0
    return weight.reshape(out_features // 128, 128, in_features // 128, 128).permute(3, 0, 2, 1).contiguous()

def unpack_weight(weight):
    return weight.permute(1, 3, 2, 0).reshape(weight.shape[1] * 128, weight.shape[2] * 128)

def make_autograd(kernel_call, transposed_relu=False, normal_input_gradient=False, packed_weight_inputs=False):

    class FusedMLP(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, up, down, up_packed=None, down_packed=None):
            if packed_weight_inputs:
                assert up_packed is not None and down_packed is not None
                result, relu = kernel_call(x, up_packed, down_packed)
            else:
                result, relu = kernel_call(x, up, down)
            ctx.save_for_backward(x, up, down, relu)
            return result

        @staticmethod
        def backward(ctx, grad):
            x, up, down, relu = ctx.saved_tensors
            if transposed_relu:
                activated_t = relu.square()
                dhidden_t = down.to(x.dtype).T @ grad.T
                dpre_t = dhidden_t * (relu * 2.0)
                if normal_input_gradient:
                    dx = dpre_t.T @ up.to(x.dtype)
                else:
                    dx = (up.to(x.dtype).T @ dpre_t).T
                dup = (dpre_t @ x).to(up.dtype)
                ddown = (grad.T @ activated_t.T).to(down.dtype)
            else:
                activated = relu.square()
                dhidden = grad @ down.to(x.dtype)
                dpre = dhidden * (relu * 2.0)
                dx = dpre @ up.to(x.dtype)
                dup = (dpre.T @ x).to(up.dtype)
                ddown = (grad.T @ activated).to(down.dtype)
            if packed_weight_inputs:
                return (dx, dup, ddown, None, None)
            return (dx, dup, ddown)
    return FusedMLP.apply
import types
import torch

@torch.compiler.disable(recursive=False)
def call_separate_region(compiled_function, *inputs):
    """A deliberate graph boundary; the inner training function still compiles."""
    return compiled_function(*inputs)

def make_forward(device, native_forward, use_weight_cache, normal_input_gradient=False, token_tile=512, grouped_io=False, weight_mode=0, forward_backend='nki', separate_region=False):
    assert forward_backend in ('nki', 'native')
    if forward_backend == 'native':
        assert weight_mode == 0

        def kernel_call(x, up, down):
            result, relu = reference_forward(x, up, down)
            return (result, relu.T.contiguous())
    elif str(device).startswith('neuron'):
        from torch_neuronx import wrap_nki
        if weight_mode == 4:
            raise ValueError('Unsubmitted MLP variant; use native forward')
            kernel = wrap_nki(build_fixed_kernel())
        else:
            kernel = wrap_nki(build_kernel())

        def kernel_call(x, up, down):
            tile = min(token_tile, x.shape[0] // 2)
            return kernel[2](x, up, down, 2, tile, True, True, True, grouped_io, weight_mode)
    else:

        def kernel_call(x, up, down):
            if weight_mode >= 3:
                up, down = (unpack_weight(up), unpack_weight(down))
            result, relu = reference_forward(x, up, down)
            return (result, relu.T)
    fused = make_autograd(kernel_call, transposed_relu=True, normal_input_gradient=normal_input_gradient, packed_weight_inputs=weight_mode >= 3)
    if separate_region:

        def training_call(x, up, down, up_packed=None, down_packed=None):
            if weight_mode >= 3:
                return fused(x, up, down, up_packed, down_packed)
            return fused(x, up, down)
        compiled_training = torch.compile(training_call, backend='neuron' if str(device).startswith('neuron') else 'inductor', fullgraph=True, dynamic=False)

        def apply_fused(*inputs):
            return call_separate_region(compiled_training, *inputs)
    else:
        apply_fused = fused

    def weight(linear, dtype):
        cached = linear._cached_weight
        if use_weight_cache and cached is not None and (cached.dtype == dtype):
            return cached
        return linear.weight.to(dtype=dtype)

    def forward(self, x, fc=None):
        tokens = x.numel() // x.shape[-1]
        tile = min(token_tile, tokens // 2)
        if not self.training or fc is not None or x.dtype != torch.bfloat16 or (tile < 128) or tile % 128 or tokens % (2 * tile):
            return native_forward(self, x, fc)
        up, down = (weight(self.c_fc, x.dtype), weight(self.c_proj, x.dtype))
        if weight_mode >= 3:
            if use_weight_cache and self.c_fc._cached_weight is not None and (self.c_proj._cached_weight is not None) and (self.c_fc._cached_weight.dtype == x.dtype) and (self.c_proj._cached_weight.dtype == x.dtype):
                up_packed = self.c_fc._nki_packed_weight
                down_packed = self.c_proj._nki_packed_weight
            else:
                up_packed, down_packed = (prepack_weight(up), prepack_weight(down))
            result = apply_fused(x.reshape(tokens, x.shape[-1]), up, down, up_packed, down_packed)
        else:
            result = apply_fused(x.reshape(tokens, x.shape[-1]), up, down)
        return result.reshape(x.shape)
    return forward

def _submission_mlp_install(model, device, train_module, normal_input_gradient=False, token_tile=512, grouped_io=False, weight_mode=0, forward_backend='nki', layer_indices=None, separate_region=False):
    assert train_module.RELU2_TAU == 0.0
    assert token_tile in (128, 256, 512)
    assert weight_mode in (0, 1, 2, 3, 4) and (weight_mode == 0 or grouped_io)
    assert forward_backend in ('nki', 'native')
    assert forward_backend != 'native' or weight_mode == 0
    keys = tuple(model.state_dict())
    parameter_ids = tuple((id(p) for p in model.parameters()))
    selected = tuple(range(len(model.transformer.h))) if layer_indices is None else tuple(layer_indices)
    assert selected and len(set(selected)) == len(selected)
    assert all((0 <= i < len(model.transformer.h) for i in selected))
    shared_forwards = {}
    for index, block in enumerate(model.transformer.h):
        if index not in selected:
            continue
        module = block.mlp
        assert module.c_fc is not None and module.c_gate is None and (not module.swiglu_hidden)
        assert module.c_fc.bias is None and module.c_proj.bias is None
        assert not module.c_fc.weight_transposed and (not module.c_proj.weight_transposed)
        hidden, channels = module.c_fc.weight.shape
        assert module.c_proj.weight.shape == (channels, hidden)
        assert channels % 128 == 0 and hidden % 128 == 0
        live_bytes = 4 * channels * hidden + 2 * (channels + hidden) * token_tile
        if weight_mode:
            live_bytes = 2 * (channels + hidden) * token_tile + 4 * 128 * (channels + hidden)
            if weight_mode == 1:
                live_bytes += 2 * channels * hidden
        if grouped_io:
            live_bytes += 2 * channels * token_tile
        assert live_bytes + 2 * 1024 ** 2 <= 24 * 1024 ** 2
        if weight_mode >= 3:
            for linear in (module.c_fc, module.c_proj):
                native_refresh = type(linear).refresh_weight_cache

                def refresh(self, dtype, _native_refresh=native_refresh):
                    _native_refresh(self, dtype)
                    self._nki_packed_weight = prepack_weight(self._cached_weight)
                linear.refresh_weight_cache = types.MethodType(refresh, linear)
                linear._nki_packed_weight = prepack_weight(linear._cached_weight) if linear._cached_weight is not None else None
        if separate_region and type(module) in shared_forwards:
            forward = shared_forwards[type(module)]
        else:
            forward = make_forward(device, type(module).forward, train_module.USE_WEIGHT_CACHE, normal_input_gradient, token_tile, grouped_io, weight_mode, forward_backend, separate_region)
            if separate_region:
                shared_forwards[type(module)] = forward
        module.forward = types.MethodType(forward, module)
    assert tuple(model.state_dict()) == keys
    assert tuple((id(p) for p in model.parameters())) == parameter_ids
from types import MethodType
import torch
import torch.distributed as dist

def make_plan(groups, world_size):
    muon_ms = {(1024, 1024): 12.54413 / 24, (1024, 4096): 9.76316 / 6, (4096, 1024): 8.98251 / 6, (8, 32): 0.23355 / 3}
    entries, parameters, seen = ([], [], set())
    for group_index, group in enumerate(groups):
        if group['kind'] not in ('adamw', 'muon'):
            raise ValueError('ownership probe is restricted to AdamW and Muon')
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
    counts, costs = ([0] * world_size, [0.0] * world_size)
    for entry in sorted(entries, key=lambda e: (-e['estimated_ms'], -e['elements'], e['index'])):
        owner = min(range(world_size), key=lambda r: (costs[r], counts[r], r))
        entry.update(owner=owner, offset=counts[owner])
        counts[owner] += entry['elements']
        costs[owner] += entry['estimated_ms']
    stride = (max(counts) + 127) // 128 * 128
    return (entries, parameters, dict(elements_by_owner=counts, estimated_ms_by_owner=costs, shard_elements=stride, total_padded_bytes=stride * world_size * 4))

class OwnedOptimizer:

    def __init__(self, optimizer, pad_singleton_muon=False):
        if not dist.is_initialized():
            raise ValueError('optimizer ownership requires an initialized process group')
        if optimizer.state:
            raise ValueError('install ownership before the first update; use its checkpoint loader to resume')
        self.optimizer = optimizer
        self.pad_singleton_muon = bool(pad_singleton_muon)
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

def _submission_optimizer_install(optimizer, train_module, pad_singleton_muon=False):
    if not train_module.USE_FOREACH_OPTIM or not train_module.USE_TENSOR_LR_SCALARS:
        raise ValueError('ownership gate requires the current foreach/runtime-scalar policy')
    if hasattr(optimizer, '_fresh_start_owned'):
        raise ValueError('ownership already installed')
    adapter = OwnedOptimizer(optimizer, pad_singleton_muon=pad_singleton_muon)
    optimizer._fresh_start_owned = adapter
    optimizer.step = MethodType(lambda self, update_adamw=True: adapter.step(update_adamw), optimizer)
    optimizer.state_dict = MethodType(lambda self: adapter.state_dict(), optimizer)
    optimizer.load_state_dict = MethodType(lambda self, payload: adapter.load_state_dict(payload), optimizer)
    return adapter

def build_kernels():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def forward(z, tgt, lanes: int=2, target_gather: bool=False):
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
    def backward(z, tgt, lse, g, lanes: int=2, round_bf16_output: bool=False):
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
    return (forward, backward)

def selected_lanes(rows):
    if rows <= 0 or rows % 128:
        raise ValueError('the NKI CE adapter requires positive multiples of128 rows')
    return 2 if rows >= 512 and rows % 256 == 0 else 1

def _submission_ce_install(train_module):
    """Replace the two primitive calls used by the existing CE autograd class.

    Returns the original pair so an isolated diagnostic can restore it. Model
    parameters, loss reductions, scoring and checkpoint formats are unchanged.
    Installation belongs before compilation, once per process.
    """
    if not train_module._HAS_NKI or train_module.LOGIT_SOFTCAP != 15.0:
        raise ValueError('adaptive CE is gated only for NKI with softcap15')
    if not train_module.USE_NKI_SOFTCAP_CE:
        raise ValueError('adaptive CE requires the existing NKI loss path')
    if getattr(train_module, '_fresh_start_adaptive_ce_installed', False):
        raise RuntimeError('adaptive CE is already installed in this process')
    from torch_neuronx import wrap_nki
    forward, backward = (wrap_nki(kernel) for kernel in build_kernels())
    originals = (train_module._softcap_ce_fwd, train_module._softcap_ce_bwd)

    def adaptive_forward(z, target):
        lanes = selected_lanes(z.shape[0])
        return forward[lanes](z, target, lanes, True)

    def adaptive_backward(z, target, lse, gradient):
        lanes = selected_lanes(z.shape[0])
        return backward[lanes](z, target, lse, gradient, lanes)
    train_module._softcap_ce_fwd = adaptive_forward
    train_module._softcap_ce_bwd = adaptive_backward
    train_module._fresh_start_adaptive_ce_installed = True
    return originals

class _SubmissionEvalGPT(GPT):
    """Use identical per-sequence arithmetic for every scorer batch size.

    BF16 backend kernels can select different rounding paths when the batch
    dimension changes. Evaluating independent rows with batch dimension one
    preserves the original B1 logits and makes batching a pure concatenation.
    The parameters, sequence computation and training model remain unchanged.
    """

    def forward(self, idx):
        if idx.ndim != 2 or idx.shape[0] < 1:
            raise ValueError('Expected a nonempty batch of token sequences')
        single_forward = super().forward
        if idx.shape[0] == 1:
            return single_forward(idx)
        return torch.cat([single_forward(idx[i:i + 1]) for i in range(idx.shape[0])], dim=0)

def _install_submission_training():
    """Training-only adapters; load_for_eval uses the native frozen model."""
    import sys
    module = sys.modules[__name__]
    original_init = GPT.init_weights
    original_setup_optimizer = GPT.setup_optimizer
    ce_installed = False

    def initialize(model):
        nonlocal ce_installed
        original_init(model)
        assert USE_BF16_NORM_OUTPUT and COMPUTE_DTYPE == torch.bfloat16
        before_rng = torch.get_rng_state().clone()
        _submission_mlp_install(model, 'neuron', module, forward_backend='native')
        if not ce_installed:
            _submission_ce_install(module)
            ce_installed = True
        assert torch.equal(before_rng, torch.get_rng_state())
        print('SUBMISSION_TRAIN_INTEGRATION ' + json.dumps(dict(source_commit='1da6c0714c0755fd3b078e972fbb0aefbb43bf6d', native_forward_custom_mlp_backward=True, optimizer_ownership=True, adaptive_nki_ce=True, loader_and_timing_hooks_unchanged=True)), flush=True)

    def setup_optimizer(model, *args, **kwargs):
        optimizer = original_setup_optimizer(model, *args, **kwargs)
        assert USE_FOREACH_OPTIM and USE_TENSOR_LR_SCALARS
        assert not USE_CAUTIOUS_WD
        assert dist.is_initialized() and dist.get_world_size() == 4
        ownership = _submission_optimizer_install(optimizer, module, pad_singleton_muon=True)
        print('SUBMISSION_OPTIMIZER_INTEGRATION ' + json.dumps(dict(owner_partitioning=True, rank=dist.get_rank(), **ownership.summary)), flush=True)
        return optimizer
    GPT.init_weights = initialize
    GPT.setup_optimizer = setup_optimizer
if __name__ == '__main__':
    _install_submission_training()
    main()
