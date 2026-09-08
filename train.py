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

# Proxy for process launch, captured before the heavy imports below. Used to
# align the self-managed budget clock with the runner's startup allowance.
_PROCESS_STARTED = time.monotonic()

# Trn2 compiler/runtime config wins (Solution 13a/13b) -- must be set before the
# torch/torch_neuronx imports below to take effect. Both toggles default OFF and
# must be smoke-tested individually before trusting them: two attempts at 13a
# broke neuronx-cc compilation outright ("hlo2penguin: Did you mean
# '--remat=true'?"), taking down dist.init_process_group's own RNG-state barrier
# kernel. First attempt: a quoted --internal-hlo2tensorizer-options value (bad
# env-var quoting). Second attempt, after fixing the quoting: still broke --
# `neuronx-cc compile --help` on this installed SDK (2.26.6360.0) confirms
# --internal-hlo2tensorizer-options/--fast-relayout do not exist as flags at
# all here. That claim came from the Gemini Deep Research cross-check
# (trn2-optimization-research.md Source 4) and did not hold on real hardware --
# exactly the "verify before trusting" caution already flagged there. Dropped.
# --model-type/--distribution-strategy/--lnc are confirmed present in --help.
# NEURON_COMPILE_CACHE_DIR is left unconditional: a plain filesystem path, not
# compiler-flag syntax, independently verified safe across every run so far.
USE_TRN2_COMPILER_FLAGS = False  # A/B'd at final locked shape: real small win alone (246/1.264601 vs baseline 240/1.269318), but combined with USE_CAUSAL_FASTPATH it's a net-negative (262/1.254491 vs USE_CAUSAL_FASTPATH alone's 265/1.252675) -- doesn't compose, left off. See trn2-experiment-results.md
USE_STOCHASTIC_ROUNDING = False  # discarded: fails causality_check (max_abs_diff 0.021) at final shape -- confirmed a false positive (SR's own forward-to-forward rounding noise alone is 0.0113, already exceeding the harness's 1e-2 tolerance), not a real leak -- but the real submission's own compliance checks would hit the same false-positive risk, so not safe to use regardless of any speed benefit. See trn2-experiment-results.md
os.environ.setdefault("NEURON_COMPILE_CACHE_DIR", "/tmp/neuron_cache")
if USE_TRN2_COMPILER_FLAGS:
    os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
    os.environ.setdefault("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", "3")
    os.environ.setdefault(
        "NEURON_CC_FLAGS",
        f"--model-type=transformer --distribution-strategy=llm-training "
        f"--lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG', '1')}",
    )
if USE_STOCHASTIC_ROUNDING:
    os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    EVAL_TIMEOUT_SECONDS,
    TOKENIZER_VOCAB_SIZE,
    TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,
    TRAIN_STARTUP_STEPS_EXCLUDED,
    TRAIN_TIME_BUDGET_SECONDS,
    Tokenizer,
    _document_batches,
    causality_check,
    ensure_tokenizer,
    evaluate_bpb,
    get_dist_info,
    make_dataloader,
    print0,
    record_step_boundary,
)


# -----------------------------------------------------------------------------
# Bucket B (post-submission round): remainder-requeue packer. A "data sampling
# strategy" change per README (train.py may freely change this; prepare.py itself
# is untouched here -- reuses its private _document_batches generator and
# get_dist_info). prepare.py's make_dataloader() permanently crops+discards a
# document's remainder when it doesn't fit the current row's remaining space
# (picks the globally-shortest buffered doc, takes doc[:remaining], drops
# doc[remaining:] forever). This is a byte-for-byte copy of that function except
# the crop branch requeues the leftover back into doc_buffer instead of dropping
# it, so a long document gets spread losslessly across multiple rows over time
# instead of losing everything past the first crop point. The requeued leftover
# carries no new BOS token (it isn't a new document, just a continuation) --
# correct, since BOS is only prepended once, at encode() time, before any
# cropping ever happens. _document_batches_fn is an injectable seam purely for
# the offline correctness harness (research/scratch/requeue_packer_equiv.py);
# real runs always use the real prepare._document_batches default.
def make_dataloader_requeue(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    device: torch.device | str,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    shard_ids=None,
    _document_batches_fn=_document_batches,
):
    assert batch_size > 0 and seq_len > 1
    ddp, rank, _local_rank, world_size = get_dist_info()
    batches = _document_batches_fn(split, rank, world_size, tokenizer_batch_size, shard_ids=shard_ids)
    bos = tokenizer.get_bos_token_id()
    row_capacity = seq_len + 1
    doc_buffer: list[list[int]] = []
    buffer_min = 512
    epoch = 0

    dev = torch.device(device)
    pin_memory = dev.type == "cuda"
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
                        best_idx, best_len = i, n
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    n = len(doc)
                    row_buffer[row_idx, pos:pos + n] = torch.tensor(doc, dtype=torch.long)
                    pos += n
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    leftover = doc[remaining:]
                    if leftover:
                        doc_buffer.append(leftover)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1].to(dtype=torch.int32))
        cpu_targets.copy_(row_buffer[:, 1:])
        inputs.copy_(cpu_inputs, non_blocking=pin_memory)
        targets.copy_(cpu_targets, non_blocking=pin_memory)
        yield inputs, targets, {"epoch": epoch}


# -----------------------------------------------------------------------------
# Baseline hyperparameters. Model-related config lives here, not in prepare.py.

SEQ_LEN = 2048  # Solution 11 (Trn2 real-hardware context-length resweep, fixed DEPTH=6/ASPECT_RATIO=96/TOTAL_BATCH_SIZE=131072): real interior optimum -- 2048 beat 512/1024/4096 (U-shape, see trn2-experiment-results.md). Not yet confirmed at full 1800s budget.
DEPTH = 6  # Tier-1 spot-check under compile: DEPTH=5 won decisively at 900s (560/1.176155 vs 427/1.180633) but LOST at full 1800s (1082/1.117264 vs DEPTH=6's pre-beta2 827/1.113006) -- budget-dependent-optimum pattern, DEPTH=5 hits its capacity floor before 1800s is up. DEPTH=6 kept; see trn2-experiment-results.md. Confirming DEPTH=6+beta2=0.95 at full budget next for a fair apples-to-apples number.
ASPECT_RATIO = 96  # Solution 11 (Trn2 real-hardware width-only resweep at fixed DEPTH=6): real interior optimum -- 640 beat 256/384/512/768 (U-shape, see trn2-experiment-results.md)
HEAD_DIM = 128

DEVICE_BATCH_SIZE = 1
TOTAL_BATCH_SIZE = 131072  # Solution 11 (Trn2 real-hardware batch resweep, re-validated at n_embd=640): same winner as at n_embd=384 -- 131072 beat 65536/262144 at the new width too (900s: 1.269318 vs 1.279503/1.307378). See trn2-experiment-results.md
NUM_STEPS = 1000
# The official runner charges wall time only after the first five completed
# optimizer steps (the startup/compilation allowance, capped). The baseline
# restarts its budget clock at that boundary so it can use the full charged
# budget even when compilation makes the first steps slow.
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
CHECKPOINT_RESERVE_SECONDS = 5.0
# Covers the launch -> module-import gap this process cannot observe when
# checking its startup time against the allowance cap.
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0

LOGIT_SOFTCAP = 15.0

EMBEDDING_LR = 0.30
UNEMBEDDING_LR = 0.006
MATRIX_LR = 0.025
SCALAR_LR = 0.20
WEIGHT_DECAY = 0.10
VE_WD = 0.0  # weight decay for the value_embeds AdamW group (separate from the shared embedding_params group)
EMBED_ADAMW_BETA2 = 0.95  # Tier-1 real win at final locked shape (900s: 427/1.180633 vs 0.995's 406/1.190086, both dimensions better) -- HLAT (7B/70B pretrain) used beta2=0.95, deferred pending Solution 11, now confirmed standalone. See trn2-experiment-results.md
WARMUP_STEPS = 0
GRAD_CLIP = 0.0
WARMDOWN_RATIO = 0.75   # fraction of max_train_seconds spent on LR cooldown (0 = old flat-after-warmup)
FINAL_LR_FRAC = 0.05    # LR floor as a fraction of peak LR at the end of cooldown (1.0 = no floor effect)
USE_TENSOR_LR_SCALARS = True   # LOCKED ON. Carry the per-step lr in a device tensor instead of a Python float so the optimizer graph is compiled once, not once per lr value (see MuonAdamW._device_scalar). Biggest single win measured on Trn2 so far: dt stays 0.90s through the whole cooldown instead of 2.2-4.3s, giving 1995 steps/1.043489 at 1800s vs 847/1.108697, and 995/1.094327 at 900s vs 427/1.180633. Model math unchanged; causality exact.
LRM_QUANTUM = 0         # >0: snap the LR multiplier onto a 1/LRM_QUANTUM grid, so the cooldown uses at most this many distinct lr values (see lr_multiplier); 0 = continuous, one new value per step
WD_DECAY_TO_ZERO = True  # decay Muon's weight_decay linearly to 0 over the run instead of holding it constant
MUON_MOMENTUM_WARMUP_STEPS = 300  # steps to ramp Muon momentum 0.85 -> 0.95
INIT_SCALE = 0.68  # multiplier on the attn/mlp uniform-init bound (1.0 = original baseline scale)
PACK_FACTOR = 2  # dataloader packing-bin multiplier: fetch PACK_FACTOR*SEQ_LEN rows, reshape to PACK_FACTOR*batch rows of SEQ_LEN
USE_REQUEUE_PACKER = False  # Bucket B: implemented + verified correct (offline harness, research/scratch/requeue_packer_equiv.py, zero corruption over a 20k-doc synthetic corpus), but the real hardware A/B at the final locked shape was a small real negative (332 steps/1.235799 vs baseline 330/1.234514) -- not kept. See trn2-experiment-results.md
USE_VE = True  # value embeddings (ResFormer-style): alternating layers, input-gated mix into attention's V stream
VE_GATE_CHANNELS = 32


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Alternating layers get a value embedding; the last layer always does."""
    return layer_idx % 2 == (n_layer - 1) % 2


WINDOW_PATTERN = "L"    # per-layer attention window pattern, cycled to n_layer length; 'L' = full context, 'S' = short window
SHORT_WINDOW_FRAC = 0.5  # short window size as a fraction of SEQ_LEN (only matters for 'S' layers)
ROPE_BASE = 100000       # rotary embedding base frequency
USE_LOCAL_CONV = True    # depthwise causal conv1d before attention in each block, RWKV/Mamba/Hyena-style.
# NOTE: as wired this is DEAD COMPUTE -- CausalConv.weight is zero-initialised, init_weights never
# touches it, and setup_optimizer() collects only attn/mlp/lambda params, so it is in no optimizer
# group and stays exactly zero for the whole run: conv(norm(x)) is exactly 0.0 and x + 0.0 is
# bit-exact x (GRAD_CLIP == 0.0 closes the one path -- global grad-norm -- by which its
# computed-then-discarded dweight could still perturb training). Asserted in
# research/scratch/local_conv_deadparam_check.py. It is nevertheless left ON because turning it off
# measured SLOWER on real hardware, not faster; drive the A/B with --local-conv / --no-local-conv.
LOCAL_CONV_KERNEL = 4
USE_GATED_MLP = False    # SwiGLU-style: c_fc(x) * silu(c_gate(x)) instead of relu(c_fc(x))**2; adds one weight matrix
USE_WEIGHT_CACHE = True  # cache each Linear's dtype-cast weight once per optimizer step instead of every forward call
USE_DOC_MASK = False  # block attention across document boundaries within a packed row (BOS-derived, data-dependent)
USE_SELECTIVE_GATE = False  # make resid_lambda/x0_lambda data-dependent (Mamba-style selectivity) instead of static scalars
SELECTIVE_GATE_CHANNELS = 32
USE_XSA = False  # A/B'd at final locked shape: real net negative (368 steps/1.208657 vs baseline 384/1.198478, both dimensions worse) -- consistent with the paper's own scale-dependent results (their smallest tested model, 0.7B, got only a small gain; ours is ~13x smaller). Bucket D: Exclusive Self Attention (arXiv:2603.09078). See trn2-experiment-results.md
_XSA_EPS = 1e-6  # numerical floor for the ||v_i||^2 denominator, avoids div-by-zero at init/degenerate v
USE_VALUE_RESIDUAL = False  # Value Residual Learning / ResFormer (arXiv:2410.17897, Zhou et al., ICLR 2025): inject layer-0's own value stream into every later layer's V via a learned per-layer scalar (zero-init). Real win at the autoresearch proxy scale (1.405->1.368 val_bpb, eager, no compile, no kernels) but a real, clean NEGATIVE at final locked shape (388/1.196654 vs baseline 427/1.180633, both dimensions worse) -- a new instance of the scale-transfer trap (matches XSA/SLAY), here likely from keeping an extra full-sized activation (v0) alive across the whole network depth for fwd+bwd, interacting badly with torch.compile's memory/fusion assumptions. See trn2-experiment-results.md
USE_FUSED_QKV = False  # A/B'd at final locked shape: net negative as expected (206 steps/1.312086 vs baseline 240/1.269318) -- disables the rope+QK-norm kernel fusion, fewer steps and worse quality. See trn2-experiment-results.md
USE_CAUSAL_FASTPATH = True  # Solution 13 winner, A/B'd at final locked shape: real win (265 steps/1.252675 vs baseline 240/1.269318, +0.0166) -- the best of all four Solution-13 toggles, doesn't compose further with USE_TRN2_COMPILER_FLAGS. See trn2-experiment-results.md
USE_MUDD = False  # MUDDFormer (arXiv:2502.12170, Xiao et al., ICML'25 -- real, verified directly against the arxiv HTML, not just the prior research note): Multiway dynamic DENSE connections replace this block's single learned resid_lambda/x0_lambda scalar with FOUR independent, per-token dynamic weighted sums (one each for the block's Q/K/V input streams and its residual-stream base) drawn from ALL earlier hidden states (embedding + every prior block's output), not just x_prev/x0. Causally trivial: the weight generator (Block.mudd) and the weighted sum both operate per-token-position independently -- it only recombines information already available at that position from earlier LAYERS, never a different position. Zero-init no-op: the generator's second-stage projection (MuddCompose.w2) is zero-initialized so at step 0 the dynamic weights collapse to the constant bias, which is itself set to reproduce the exact resid_lambda=1.0/x0_lambda=0.05 formula for all 4 streams -- USE_MUDD=True is bit-identical to USE_MUDD=False at init. CAVEAT the paper's own depth-scaling ablation (Fig. 4/Table 2) only tests LM depths >=24 layers (up to 42) and a 12-layer ViT; it never tests anything as shallow as this repo's locked DEPTH=6, and states Transformer++ "shows diminishing returns beyond 24 layers" while MUDD "maintains gains up to 42" -- i.e. the demonstrated advantage is framed entirely in terms of depths this repo doesn't reach. Real A/B at the locked shape (900s, DEPTH=6): 646 steps/1.146435 vs baseline 995/1.094327 -- a real, clean NEGATIVE on both dimensions (fewer steps from the O(L) dense-connection overhead, ~1.40s/step vs baseline's ~0.90s/step, AND worse quality even so). Consistent with the paper's own framing: this repo's DEPTH=6 is well below anything MUDD was shown to help at. See research/experiment-log.jsonl (mudd-900s-v2) and research/run-logs/mudd-900s-v2.log.
USE_DCMHA = False  # Dynamically Composable Multi-Head Attention (arXiv:2405.08553, Xiao et al., ICML'24 oral -- verified directly against the reference PyTorch implementation, github.com/Caiyun-AI/DCFormer). A learned Compose function recombines the raw [B,H,T,T] attention SCORE matrix (pre-softmax) and WEIGHT matrix (post-softmax) across the HEAD dimension, with per-position weights generated from each token's own hidden state. Forces CausalSelfAttention off the fused F.scaled_dot_product_attention kernel and onto an explicit QK^T -> mask -> compose -> softmax -> compose -> @V sequence (SDPA never exposes the raw score/weight matrices Compose needs) -- profiling already found SDPA a near-peak-efficient fused kernel (~10.6-10.8% of the step, ~98.5% TensorE-occupied on backward), so this risks a real throughput regression independent of whether Compose helps quality. MEASURED, not assumed, and it is a real, decisive NEGATIVE: at the locked shape (SEQ_LEN=2048) torch.compile(backend="neuron") FAILS to compile outright ("NCC_EXTP004: Instructions generated by compiler 6560191 exceeds the typical limit of 5000000") -- the unfused sequence is simply too large a graph at this sequence length. Eager mode (--no-compile) does run, but at ~116-223s/step vs baseline eager's ~1.2s/step (~100x slower) -- see research/run-logs/dcmha-compile-fail.log and dcmha-eager-smoke.log. Causally safe by construction: Compose only mixes OTHER HEADS' values at the SAME (query,key) cell (never a different position pair), and the reference's own Compose (_cross_head_proj, verified by reading github.com/Caiyun-AI/DCFormer/pytorch/dcformer/modeling_dcformer.py) is exactly linear/homogeneous in its input with no additive bias term -- so at any masked cell, where the causal mask has already zeroed every head's score/weight uniformly, Compose(0)=0 for every head there too and cannot resurrect a masked position (confirmed empirically, not just argued, in research/scratch/mudd_dcmha_equiv.py and on real hardware: causality_max_abs=0.0). Zero-init no-op: the generator's second-stage weights (DCMHACompose's per-branch linear layers) are zero-initialized, so at step 0 Compose(x)=x exactly and USE_DCMHA=True is bit-identical to USE_DCMHA=False.
_DCMHA_RANK_DIVISOR = 2  # low-rank divisor for the head-mixing hypernetwork, matches the reference's dynamic_squeeze_ratio=n_head//2 default
USE_MAMBA2 = False  # Mamba2 / SSD hybrid (arXiv:2405.21060, Dao & Gu): replace SOME blocks' CausalSelfAttention with a real Mamba2Mixer (in_proj -> depthwise causal conv1d -> selective state-space scan -> gated RMSNorm -> out_proj), keeping the MLP, the resid/x0 skip and the local_conv untouched, so MAMBA2_PATTERN picks which layers are which. The SSD scan is the chunked "structured state-space duality" form, verified exact (float64, max_rel < 1e-12) against an independently written naive sequential recurrence in research/scratch/mamba2_ssd_ref.py and re-checked by research/scratch/mamba2_train_integration_check.py on every commit. Causally safe by construction AND measured (causality_max_abs=0.0 on real hardware): the conv1d is left-padded so position t only ever reads t-3..t, and the scan's two decay matrices are masked BEFORE exp -- the intra-chunk mask is lower-triangular (s <= t) and the inter-chunk mask STRICTLY lower-triangular (chunk c reads only chunks c' < c), with the mask added as -30000 to an exponent already clamped to <= 0, so exp() gives exactly 0.0 there and no +inf * 0 = NaN is possible (the latent bug in the sibling repo's kernel, which masks after exp). Zero-init no-op: mixer.out_proj is zero-initialized like every other block output projection, so a fresh Mamba2 layer is an exact identity. MEASURED at the locked shape and it is a real NEGATIVE on wall clock, not a win: at 45.0M params vs the transformer's 55.7M it is a dead heat on quality at a fixed 300 steps (1.270639 vs 1.269458 bpb) but torch.compile makes it SLOWER not faster (2.17 vs 2.18 s/step eager, i.e. 0.90x, while the transformer gets 2.37x), so at a fixed 1800s wall-clock budget it only reaches 908 steps against the transformer's 1995 and loses 1.109065 to 1.043489 bpb. Kept off by default and kept in-tree because the scan kernel itself is a real 4.5x win over eager and the arch may pay off if the compiler improves. See research/experiment-log.jsonl rows m2hybrid-nki-fixed300 / m2hybrid-nki-fullbudget (and their xf-fixed300 / tensorlr-fullbudget baselines), plus research/run-logs/m2hybrid-*.log.
MAMBA2_PATTERN = "MMAMMM"  # per-layer mixer selection, cycled to n_layer like WINDOW_PATTERN: "M" = Mamba2 mixer, anything else = ordinary attention. "MMAMMM" (5 Mamba2 + 1 attention at layer 2) is the measured best of the hybrid patterns tried -- a pure-Mamba2 stack loses on quality, and one full-context attention layer is enough to restore exact long-range retrieval. Layers that become Mamba2 also lose their value-embedding (there is no attention V to inject one into), so the ve_gate/value_embeds tables for those layers are not created at all.
MAMBA2_HEAD_DIM = 128  # P, the per-head state width of the scan. 128 == the SBUF partition limit, which is what lets one head's chunk tile fit a single NKI matmul.
MAMBA2_D_STATE = 128  # N, the SSM state dimension per group. The kernel requires N <= MAMBA2_CHUNK.
MAMBA2_CHUNK = 128  # Q, the SSD chunk length. Must divide sequence_len and satisfy N <= Q <= 128 (the partition limit) for the NKI kernel; anything else silently falls back to the eager scan.
MAMBA2_EXPAND = 2  # d_inner = MAMBA2_EXPAND * n_embd, the standard Mamba/Mamba2 expansion factor
MAMBA2_N_GROUPS = 1  # G, B/C groups shared across heads (G=1 == "all heads share one B/C", the Mamba2 default)
MAMBA2_CONV_KERNEL = 4  # k for the depthwise causal conv1d, the Mamba2 reference default
MAMBA2_DT_MIN = 0.001  # dt_bias is initialized to softplus^-1 of a log-uniform draw in [MAMBA2_DT_MIN, MAMBA2_DT_MAX]
MAMBA2_DT_MAX = 0.1
MAMBA2_DT_INIT_FLOOR = 1e-4


def doc_boundary_mask(idx: torch.Tensor) -> torch.Tensor:
    """Additive mask: 0 where i,j fall in the same packed document (by BOS-delimited segment), -inf otherwise.
    Combine (add) with the existing static causal/window mask -- both are additive float masks."""
    is_bos = idx == BOS_TOKEN_ID
    doc_id = torch.cumsum(is_bos.long(), dim=1)
    same_doc = doc_id.unsqueeze(2) == doc_id.unsqueeze(1)
    mask = torch.zeros(idx.shape[0], 1, idx.shape[1], idx.shape[1], dtype=torch.float32, device=idx.device)
    mask.masked_fill_(~same_doc.unsqueeze(1), float("-inf"))
    return mask


BOS_TOKEN_ID = -1  # set once in main()/load_for_eval() from the tokenizer, before any forward call that needs USE_DOC_MASK


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
# torch's F.rms_norm default eps: measured empirically to be float32 eps (the
# accumulation dtype's epsilon), NOT double eps -- see research notes. Getting
# this wrong is invisible for O(1) inputs and a 6% error for small ones.
_RMS_EPS = 1.1920928955078125e-07
# Additive mask for the SSD decay matrices. exp(-30000) is exactly 0.0 in
# fp32/bf16/fp64, and adding it to an exponent that has ALREADY been clamped to
# <= 0 can never produce a NaN -- unlike masking after exp, which multiplies a
# possible +inf by 0. Shared by the eager scan and the NKI kernels so both mask
# at exactly the same point in the algebra.
_SSD_NEG = -30000.0

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

    # -------------------------------------------------------------------------
    # Fused rotary-embedding + QK-norm kernel (forward AND backward).
    #
    # Replaces `norm(apply_rotary_emb(x, cos, sin))`, which the eager backend
    # lowers to ~30 separate one-op NEFFs per call (2 slices, 4 muls, 1 neg,
    # 2 adds, 1 cat, 1 rms_norm forward, plus the whole backward chain). The
    # device profile (research/trn2-experiment-results.md, Solution 12) showed
    # rope at 12.63 ms and norm at 6.00 ms of a 91.4 ms window, essentially all
    # of it launch/dispatch serialization rather than real compute. One NEFF for
    # the whole fused chain removes that dispatch cost.
    #
    # Math (x: [.., D], cos/sin: [.., D/2], D even, no learnable gamma):
    #   z1 = x1*cos + x2*sin ; z2 = x2*cos - x1*sin       (rope, an orthogonal
    #                                                      rotation per (x1,x2))
    #   r  = rsqrt(mean(z^2, -1) + eps) ; out = z * r      (parameter-free RMS)
    # backward (derived here; no forward+backward reference exists upstream):
    #   dz = r * (g - (<g,z> * r^2 / D) * z)
    #   dx1 = dz1*cos - dz2*sin ; dx2 = dz1*sin + dz2*cos  (rope^T == rope^-1)
    #
    # Rounding is deliberately matched to the eager chain: the rope products and
    # sums are rounded to x.dtype (bf16) exactly where the eager elementwise ops
    # would round, and only the RMS statistic / scaling run in fp32 (which is
    # what F.rms_norm does internally). Measured on-device against eager:
    # 99.9% of elements bit-identical, worst case 1 bf16 ULP (see
    # research/scratch/rope_norm_equiv.py).

    @nki.jit
    def _rope_norm_fwd(x, cos, sin):
        """Fused rms_norm(rope(x)) for x [BT, H, D], cos/sin [T, D//2]."""
        BT, H, D = x.shape
        T = cos.shape[0]
        Dh = D // 2
        dt = x.dtype
        P = _NKI_PARTITION
        # [P,H,1] -> [P,H,D] broadcast: stride 0 over the last axis.
        BC = [[H, P], [1, H], [0, D]]
        CS = [[Dh, P], [0, H], [1, Dh]]  # [T,Dh] -> [P,H,Dh], broadcast over heads
        out = nl.ndarray((BT, H, D), dtype=dt, buffer=nl.shared_hbm)

        for i in nl.affine_range(BT // P):
            r0 = i * P
            t0 = (i * P) % T  # rows are (b,t)-major, T % P == 0, so t0 is exact

            xb = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[r0:r0 + P])
            cb = nl.ndarray((P, H, Dh), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=cb, src=cos.ap(pattern=CS, offset=t0 * Dh))
            sb = nl.ndarray((P, H, Dh), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=sb, src=sin.ap(pattern=CS, offset=t0 * Dh))

            x1, x2 = xb[0:P, 0:H, 0:Dh], xb[0:P, 0:H, Dh:D]
            zb = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            tb = nl.ndarray((P, H, Dh), dtype=dt, buffer=nl.sbuf)
            z1, z2 = zb[0:P, 0:H, 0:Dh], zb[0:P, 0:H, Dh:D]

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
            nisa.tensor_scalar(dst=ms, data=ss, op0=nl.multiply, operand0=1.0 / D,
                               op1=nl.add, operand1=_RMS_EPS)
            # sqrt(1/m) rather than activation(rsqrt): measured 5.8e-7 vs 4.3e-5
            # relative error on this SDK (nki_probe3.py), i.e. ~75x tighter and
            # well under a bf16 ULP, which minimises rounding-tie flips vs eager.
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
            t0 = (i * P) % T

            xb = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[r0:r0 + P])
            gf = nl.ndarray((P, H, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gf, src=g[r0:r0 + P])
            cb = nl.ndarray((P, H, Dh), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=cb, src=cos.ap(pattern=CS, offset=t0 * Dh))
            sb = nl.ndarray((P, H, Dh), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=sb, src=sin.ap(pattern=CS, offset=t0 * Dh))

            # recompute z = rope(x) (cheaper than saving it for backward)
            x1, x2 = xb[0:P, 0:H, 0:Dh], xb[0:P, 0:H, Dh:D]
            zb = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            tb = nl.ndarray((P, H, Dh), dtype=dt, buffer=nl.sbuf)
            z1, z2 = zb[0:P, 0:H, 0:Dh], zb[0:P, 0:H, Dh:D]
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
            nisa.tensor_scalar(dst=ms, data=ss, op0=nl.multiply, operand0=1.0 / D,
                               op1=nl.add, operand1=_RMS_EPS)
            r2 = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=r2, data=ms)              # r^2 == 1/(mean+eps)
            rr = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rr, op=nl.sqrt, data=r2)  # r, see fwd for why not rsqrt

            # q = <g,z> * r^2 / D
            nisa.tensor_tensor(dst=acc, data1=gf, data2=zf, op=nl.multiply)
            pp = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=pp, op=nl.add, data=acc, axis=(2,))
            qq = nl.ndarray((P, H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=qq, data=pp, op0=nl.multiply, operand0=1.0 / D)
            nisa.tensor_tensor(dst=qq, data1=qq, data2=r2, op=nl.multiply)

            # dz = r * (g - q*z), rounded to dt like eager's rms_norm backward
            nisa.tensor_tensor(dst=acc, data1=zf, data2=qq.ap(pattern=BC), op=nl.multiply)
            nisa.tensor_tensor(dst=acc, data1=gf, data2=acc, op=nl.subtract)
            dzb = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dzb, data1=acc, data2=rr.ap(pattern=BC), op=nl.multiply)

            # dx = rope^T(dz)
            d1, d2 = dzb[0:P, 0:H, 0:Dh], dzb[0:P, 0:H, Dh:D]
            ob = nl.ndarray((P, H, D), dtype=dt, buffer=nl.sbuf)
            o1, o2 = ob[0:P, 0:H, 0:Dh], ob[0:P, 0:H, Dh:D]
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
            return _rope_norm_bwd(x3, g3, cos2, sin2).reshape(b, t, h, d), None, None

    # -------------------------------------------------------------------------
    # Fused logit-softcap + cross-entropy kernel (forward AND backward).
    #
    # Replaces `LOGIT_SOFTCAP*tanh(z/S)` followed by `F.cross_entropy(...)`. The
    # eager backend lowers the softcap alone to 6 separate NEFFs over the full
    # [B*T, V] logits tensor (div.out x2, tanh, mul.Tensor x2, tanh_backward),
    # plus cross_entropy_loss / _backward_. A device profile at the locked shape
    # ([4096, 8192] fp32) measured that whole span at 18.957 ms = 11.53% of the
    # window, and an ablation that deleted just the softcap step measured a real
    # -3.2 to -3.5% step time. This kernel collapses the span to two NEFFs whose
    # only unavoidable cost is the HBM traffic (read z once in fwd, read z +
    # write dz in bwd).
    #
    # Math (S = LOGIT_SOFTCAP, c = target class, derived and verified from
    # scratch in research/scratch/softcap_ce_equiv.py, same discipline as the
    # rope+norm backward above):
    #   t = tanh(z/S) ; y = S*t                          (softcap; y in (-S, S))
    #   loss = log sum_j exp(y_j - y_c)                  (== lse - y_c, see below)
    #   dy_j/dz_j  = 1 - tanh(z_j/S)^2 = 1 - t_j^2       (== sech^2(z_j/S))
    #   dloss/dz_j = g * (softmax(y)_j - [j == c]) * (1 - t_j^2)
    # softmax(y)_j is recomputed in the backward as exp(y_j - lse) from the saved
    # lse, so the only tensor carried between the passes is [N] wide, not [N, V].
    #
    # Three deliberate deviations from PyTorch's log_softmax algorithm:
    #  * No row-max subtraction. y is bounded to (-S, S) = (-15, 15) by tanh's
    #    range *by construction*, so shifting by y_c (also in (-S, S)) puts every
    #    exponent in (-2S, 2S) = (-30, 30), i.e. every term in (9.4e-14, 1.1e13)
    #    and sum_j in [1, 8.7e16] for V = 8192: fp32 can neither overflow nor
    #    underflow here for any input z whatsoever, so the row-max reduction pass
    #    is unnecessary rather than merely usually-unnecessary. Verified
    #    empirically over input scales 1e-3..1e3 plus all-equal / confidently-
    #    right / confidently-wrong logits, not just asserted.
    #  * The shift is y_c, not a constant. Shifting by the target logit makes the
    #    j == c term exactly exp(0) = 1, so d >= 1 and loss = log(d) is a single
    #    log with no subtraction. Writing it as lse - y_c instead is catastrophic
    #    on exactly the rows that get most confident during training: a token with
    #    p_correct ~ 0.9975 has lse ~ y_c ~ 15 and a true loss of ~2.5e-3, so the
    #    difference throws away 4 of fp32's 7 digits. Costs nothing: the gather
    #    that produces y_c is needed anyway, only reordered before the exp.
    #  * sum_j exp(...) rides along on the Activation-engine exp instructions via
    #    reduce_op/reduce_res instead of a separate nisa.tensor_reduce (measured
    #    bit-identical to the Vector-engine reduce, softcap_ce_diag2.py), but in
    #    R chunks rather than one sweep -- see the d= comment in the forward.
    #
    # Accuracy, measured not assumed (softcap_ce_equiv.py, softcap_ce_diag3.py).
    # Two Activation-engine LUT limits set the floor, the same class of finding as
    # the rsqrt note in the rope+norm kernel above:
    #   exp: ~1.1e-5 relative, flat in the argument, vs torch's 5.9e-8 (~190x).
    #        Crucially it is *systematic* -- a deterministic function of the
    #        argument -- so the V per-term errors add coherently instead of
    #        averaging out, and d's relative error does not shrink with V.
    #   log: a few ULPs of its own output, 3.4e-6 at log(d) ~ 9, 4.2e-5 at ~30.
    # log(d) error == relative error of d, so both land 1:1 on the loss. Net, at
    # the shape and logit distribution that actually ships ([4096, 8192], logits
    # from a real lm_head): fwd 4.3e-5 absolute = 0.0012 bf16 ULP, bwd 2.4e-7 =
    # 0.0015 bf16 ULP, i.e. ~1000x inside one bf16 ULP. Worst case over synthetic
    # stress inputs (uniform-random logits at scale 6, y filling most of the tanh
    # range): fwd 4.7e-4 = 0.0072 bf16 ULP, bwd 8.8e-6 = 0.22 bf16 ULP. The only
    # input that exceeds one bf16 ULP is the confidently-correct token, at 2.4
    # bf16 ULP of a 2.5e-3 loss -- and there the eager fp32 chain being replaced is
    # equally far from an fp64 reference (2.641e-5 vs this kernel's 2.653e-5), so
    # that case is an fp32 limit, not a regression introduced here. Gradients feed
    # a bf16 matmul whose own resolution is 4e-3.

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

        # Vocab index ramp, identical on every partition and hoisted out of the row
        # loop: the target one-hot then costs one compare per tile rather than a
        # fresh iota, and gathering y_c by mask beats 128 single-element indirect
        # DMAs per tile (which is what nkilib's cross_entropy_forward does).
        idx = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=idx, pattern=[[1, V]], offset=0, channel_multiplier=0)

        # Chunk width for the exp reduction (see the d= comment below). Python-level
        # constants, resolved at trace time.
        C = 512 if (V > 512 and V % 512 == 0) else V
        R = V // C

        for i in nl.affine_range(N // P):
            rows = nl.ds(i * P, P)
            zb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=zb, src=z[rows])
            cb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cb, src=tgt[rows])

            tb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=tb, op=nl.tanh, data=zb, scale=1.0 / S)

            # t_c = sum_j t_j * [j == c]: exactly one nonzero term, so exact.
            mb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mb, data=idx, op0=nl.equal, operand0=cb)
            nisa.tensor_tensor(dst=mb, data1=mb, data2=tb, op=nl.multiply)
            tc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tc, op=nl.add, data=mb, axis=(1,))
            yc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=yc, data=tc, op0=nl.multiply, operand0=S)
            nyc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=nyc, data=tc, op0=nl.multiply, operand0=-S)

            # d = sum_j exp(y_j - y_c), reduced for free on the exp instructions
            # themselves (reduce_res). zb is dead once tb exists, so the exp output
            # overwrites it.
            #
            # Reduced in R chunks of C rather than one sweep of V, because a single
            # sequential fp32 accumulator over V = 8192 carries O(V * eps) = 4.9e-4
            # rounding error, and that error lands directly on the loss (log(d) error
            # == relative error of d). It is worst exactly where the accumulator is
            # dominated by one term: a confidently-correct token has d = 1 + 3e-3, so
            # 8191 tiny addends each round against a running value of 1.0 and the
            # 2.5e-3 loss came out 16% wrong (4.2e-4 absolute). Summing R partials
            # instead costs O((C + R) * eps) and took that case to 2.65e-5, which is
            # parity with torch's own fp32 pairwise sum (2.64e-5) and leaves it
            # limited by the exp/log LUTs rather than by the reduction. Cases whose
            # d is dominated by a few large terms were never accumulator-limited and
            # are unchanged. Cost is R instruction issues instead of 1 over the same
            # total elements, plus one [P, R] reduce.
            parts = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
            for k in nl.affine_range(R):
                cols = nl.ds(k * C, C)
                nisa.activation(dst=zb[:, cols], op=nl.exp, data=tb[:, cols],
                                scale=S, bias=nyc, reduce_op=nl.add,
                                reduce_res=parts[:, nl.ds(k, 1)],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
            d = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=d, op=nl.add, data=parts, axis=(1,))

            # loss = log(d) directly: shifting by y_c rather than by a constant makes
            # the j == c term exactly exp(0) == 1, so d >= 1 and the loss is a single
            # log instead of the difference lse - y_c, which is ill-conditioned for a
            # confident token (lse ~ y_c ~ 15, loss ~ 2.5e-3).
            lo = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=lo, op=nl.log, data=d)
            lb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=lb, data1=yc, data2=lo, op=nl.add)  # lse = y_c + log d

            nisa.dma_copy(dst=lse[rows], src=lb)
            nisa.dma_copy(dst=loss[rows], src=lo)

        return loss, lse

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

            # t = tanh(z/S), then p = softmax(y) = exp(S*t - lse) straight off the
            # saved lse -- no second pass over the vocab to renormalise.
            tb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=tb, op=nl.tanh, data=zb, scale=1.0 / S)
            nisa.activation(dst=zb, op=nl.exp, data=tb, scale=S, bias=nb)

            # u = (p - onehot) * g, fused as (p*g) - ([j==c]*g): the one-hot and
            # its g scaling ride on one tensor_scalar, the subtract and the p*g on
            # one scalar_tensor_tensor.
            mb = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mb, data=idx, op0=nl.equal, operand0=cb,
                               op1=nl.multiply, operand1=gb)
            nisa.scalar_tensor_tensor(dst=zb, data=zb, op0=nl.multiply, operand0=gb,
                                      op1=nl.subtract, operand1=mb)

            # dz = (1 - t^2) * u; reverse0 gives `operand0 - data` so the 1 - t^2
            # and the multiply are a single Vector-engine instruction.
            nisa.activation(dst=mb, op=nl.square, data=tb)
            nisa.scalar_tensor_tensor(dst=tb, data=mb, op0=nl.subtract, operand0=1.0,
                                      reverse0=True, op1=nl.multiply, operand1=zb)
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
            return _softcap_ce_bwd(z, tgt, lse, g), None

    # ---------------------------------------------------------------------
    # Mamba2 chunked SSD scan (forward + backward), one NEFF each.
    #
    # WHY a kernel at all: the eager form below (`ssd_chunked`) is ~25 separate
    # tensor ops per Mamba2 layer, and this repo's native-eager Neuron backend
    # lowers each aten op to its own NEFF (same reason the rope+norm kernel
    # exists: ~30 NEFFs for one rope+norm). One NEFF for the whole scan removes
    # that dispatch cost -- measured 4.53x forward and 3.83x fwd+bwd over eager
    # at the real per-layer shape (b=1 T=2048 H=10 P=128 N=128 Q=128).
    #
    # LAYOUT / ALGEBRA. `nisa.nc_matmul(dst, stationary, moving)` computes
    # `dst = stationary.T @ moving` with the CONTRACTION dimension on the
    # partition axis of BOTH operands, so the whole design is chosen to make
    # every contraction land on a partition axis with no avoidable transposes.
    # Per (batch row b, head h, chunk c), with Q = chunk len, P = head_dim,
    # N = d_state:
    #
    #   dtA  [Q,1]      = dt * A_h                                      (<= 0)
    #   ell  [Q,1]      = inclusive cumsum of dtA = mm(UI, dtA), UI[r,t]=1 iff r<=t
    #   ellb [Q_s,Q_t]  = ell_t broadcast along s = mm(onesQ, UI*dtA)
    #   seg  [Q_s,Q_t]  = ell_t - ell_s
    #   MT   [Q_s,Q_t]  = exp(min(seg,0) + intra_mask) * (B_s . C_t)
    #   Y    [Q_t,P]   += mm(MT, dx)                      (contraction over s)
    #   Bbar [Q_s,N]    = B * exp(min(de - ell, 0))
    #   Gst  [N,P]      = mm(Bbar, dx)                    (contraction over s)
    #   Y    [Q_t,P]   += mm(CbarT, S)                    (contraction over N)
    #   S           <-   exp(min(de,0)) * S + Gst
    #
    # MT is built with s (the SOURCE token) on the PARTITION axis, not t: that
    # makes MT itself the `stationary` operand of the Y_intra matmul, so the
    # [Q,Q] score matrix never needs transposing. Only B and C are transposed.
    #
    # THREE THINGS THAT LOOK LIKE THEY NEED A MATMUL AND DON'T. Compute engines
    # cannot broadcast along the partition axis (only DMA or a matmul can),
    # which naively costs extra matmuls for `de` and for `Cbar^T`. But
    # `ellb[s,t] = ell_t` is ALREADY constant along s, so
    #   de replicated on every partition          = ellb[:, Q-1]
    #   exp(min(ell_t,0)) broadcast on partitions = exp(min(ellb,0))[0:N, 0:Q]  (N<=Q)
    #   Cbar^T[n,q] = C^T[n,q] * exp(min(ell_q,0)) -> elementwise with that tile,
    # so Cbar needs no transpose of its own. That removes 3 of the 11
    # tensor-engine ops per chunk and 4 of the PSUM tiles, which matters because
    # PSUM is only 8 banks.
    #
    # PSUM ACCUMULATION IS IMPLICIT: multiple nc_matmul writes to the same PSUM
    # tile form an accumulation group and SUM. So a PSUM tile may NOT be recycled
    # as scratch for an unrelated matmul -- each distinct result needs its own
    # tile (7 here). The forward deliberately EXPLOITS the accumulation for
    # ps_y, into which both the intra- and inter-chunk matmul write, giving
    # Y = Y_intra + Y_inter for free.
    #
    # MASK BEFORE EXP. For s > t the raw exponent ell_t - ell_s is POSITIVE and
    # grows like |A|*dt*(s-t) (measured up to +520 at T=384), so a bare exp()
    # would overflow to +inf and inf*0 = NaN. min(seg,0) and the additive
    # _SSD_NEG mask are therefore applied BEFORE exp. Every LIVE (s <= t) entry
    # has a provably <= 0 exponent (A < 0, dt > 0), so the clamp is exact rather
    # than an approximation, and this mirrors the eager reference op-for-op.
    #
    # Developed and equivalence-checked on device in
    # research/scratch/ssd_nki_dev.py (fwd max_rel < 2e-5, bwd < 5e-4 vs eager,
    # across 5 shapes up to the competition shape) with the gradient algebra
    # derived and verified in float64 in research/scratch/ssd_nki_sim.py.
    # ---------------------------------------------------------------------

    @nki.jit
    def ssd_fwd_nki(x2, dt2, B2, C2, Ab, Db, T, Q, G):
        """Chunked SSD forward.

        x2  [BT, H*P]   fp32, token-major (BT = b*T), head-major within the row
        dt2 [BT, H]     fp32, > 0
        B2  [BT, G*N]   fp32
        C2  [BT, G*N]   fp32
        Ab  [128, H]    fp32, A_h (< 0) replicated on every partition
        Db  [128, H]    fp32, D_h replicated on every partition
        T, Q, G         python ints (compile-time constants)
        -> y2 [BT, H*P] fp32
        """
        BT, HP = x2.shape
        H = dt2.shape[1]
        P = HP // H
        N = B2.shape[1] // G
        nb = BT // T
        CPB = T // Q          # chunks per batch row
        hpg = H // G          # heads per B/C group
        fp = nl.float32

        # N <= Q is what lets ellb double as the partition-broadcast of exp(ell).
        assert N <= Q <= _NKI_PARTITION and P <= 512 and T % Q == 0 and H % G == 0

        y2 = nl.ndarray((BT, HP), dtype=fp, buffer=nl.shared_hbm)

        # ---- constants, built once ---------------------------------------
        ri = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
        nisa.iota(dst=ri, pattern=[[0, Q]], offset=0, channel_multiplier=1)  # row index
        ci = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
        nisa.iota(dst=ci, pattern=[[1, Q]], offset=0, channel_multiplier=0)  # col index
        UI = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=UI, data1=ri, data2=ci, op=nl.less_equal)     # r <= t
        onesQ = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
        nisa.memset(dst=onesQ, value=1.0)
        # additive causal mask: 0 where s <= t, -30000 where s > t
        MASK = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=MASK, data=UI, op0=nl.subtract, operand0=1.0)  # 0 / -1
        nisa.tensor_scalar(dst=MASK, data=MASK, op0=nl.multiply, operand0=-_SSD_NEG)

        ab = nl.ndarray((Q, H), dtype=fp, buffer=nl.sbuf)
        nisa.dma_copy(dst=ab, src=Ab[0:Q, 0:H])
        db = nl.ndarray((Q, H), dtype=fp, buffer=nl.sbuf)
        nisa.dma_copy(dst=db, src=Db[0:Q, 0:H])

        # Loop over (batch row, group, head-in-group) so the B/C group index is
        # an affine expression of the loop variables rather than a floor-div.
        for bi in range(nb):
          for g in range(G):
            for hh in range(hpg):
                h = g * hpg + hh
                S = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
                nisa.memset(dst=S, value=0.0)

                for c in range(CPB):
                    r0 = bi * T + c * Q

                    xt = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                    nisa.dma_copy(dst=xt, src=x2[r0:r0 + Q, h * P:h * P + P])
                    dtt = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.dma_copy(dst=dtt, src=dt2[r0:r0 + Q, h:h + 1])
                    Bt = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.dma_copy(dst=Bt, src=B2[r0:r0 + Q, g * N:g * N + N])
                    Ct = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.dma_copy(dst=Ct, src=C2[r0:r0 + Q, g * N:g * N + N])

                    dtA = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)   # dt*A, <= 0
                    nisa.tensor_scalar(dst=dtA, data=dtt, op0=nl.multiply,
                                       operand0=ab[0:Q, h:h + 1])
                    dx = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)    # dt*x
                    nisa.tensor_scalar(dst=dx, data=xt, op0=nl.multiply, operand0=dtt)

                    # --- ell: inclusive cumsum of dtA, as a matmul against UI
                    ps_ell = nl.ndarray((Q, 1), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=ps_ell, stationary=UI, moving=dtA,
                                   is_stationary_onezero=True)
                    ell = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ell, src=ps_ell)

                    # --- ellb[s,t] = ell_t, i.e. ell broadcast along partitions
                    UIdtA = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=UIdtA, data=UI, op0=nl.multiply, operand0=dtA)
                    ps_ellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=ps_ellb, stationary=onesQ, moving=UIdtA,
                                   is_stationary_onezero=True)
                    ellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ellb, src=ps_ellb)

                    # exp(min(ell_t, 0)) broadcast along partitions (free because
                    # ellb is already constant along s). Used for Cbar and, at
                    # column Q-1, for the chunk decay exp(min(de,0)).
                    eellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=eellb, data=ellb, op0=nl.minimum, operand0=0.0)
                    nisa.activation(dst=eellb, op=nl.exp, data=eellb)

                    # --- intra-chunk decay x score matrix, s on the partition axis
                    seg = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=seg, data=ellb, op0=nl.subtract, operand0=ell,
                                       op1=nl.minimum, operand1=0.0)
                    nisa.tensor_tensor(dst=seg, data1=seg, data2=MASK, op=nl.add)
                    L = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.activation(dst=L, op=nl.exp, data=seg)

                    ps_bT = nl.ndarray((N, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_transpose(dst=ps_bT, data=Bt)
                    BTs = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=BTs, src=ps_bT)
                    ps_cT = nl.ndarray((N, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_transpose(dst=ps_cT, data=Ct)
                    CTs = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=CTs, src=ps_cT)

                    ps_cb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=ps_cb, stationary=BTs, moving=CTs)
                    CBs = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=CBs, src=ps_cb)
                    MT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=MT, data1=L, data2=CBs, op=nl.multiply)

                    # --- Y = Y_intra + Y_inter, summed IN PSUM (two writes, one tile)
                    ps_y = nl.ndarray((Q, P), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=ps_y, stationary=MT, moving=dx)
                    if c > 0:
                        # Cbar^T[n,q] = C^T[n,q] * exp(min(ell_q,0)); no transpose
                        # of Cbar needed, the eellb rows ARE that broadcast (N<=Q).
                        CbarT = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=CbarT, data1=CTs, data2=eellb[0:N, 0:Q],
                                           op=nl.multiply)
                        nisa.nc_matmul(dst=ps_y, stationary=CbarT, moving=S)

                    yb = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=yb, src=ps_y)
                    dxx = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=dxx, data=xt, op0=nl.multiply,
                                       operand0=db[0:Q, h:h + 1])
                    nisa.tensor_tensor(dst=yb, data1=yb, data2=dxx, op=nl.add)
                    nisa.dma_copy(dst=y2[r0:r0 + Q, h * P:h * P + P], src=yb)

                    # --- chunk state: S <- exp(min(de,0)) * S + Bbar^T (dt x)
                    if c + 1 < CPB:
                        # de = ell_{Q-1}, already replicated on every partition
                        rev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=rev, data1=ellb[0:Q, Q - 1:Q], data2=ell,
                                           op=nl.subtract)
                        nisa.tensor_scalar(dst=rev, data=rev, op0=nl.minimum, operand0=0.0)
                        erev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                        nisa.activation(dst=erev, op=nl.exp, data=rev)
                        Bbar = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=Bbar, data=Bt, op0=nl.multiply,
                                           operand0=erev)
                        ps_g = nl.ndarray((N, P), dtype=fp, buffer=nl.psum)
                        nisa.nc_matmul(dst=ps_g, stationary=Bbar, moving=dx)
                        Gs = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=Gs, src=ps_g)
                        nisa.tensor_scalar(dst=S, data=S, op0=nl.multiply,
                                           operand0=eellb[0:N, Q - 1:Q])
                        nisa.tensor_tensor(dst=S, data1=S, data2=Gs, op=nl.add)

        return y2

    @nki.jit
    def ssd_bwd_nki(gy2, x2, dt2, B2, C2, Ab, Db, T, Q, G):
        """Chunked SSD backward. Same layouts as ssd_fwd_nki.

        -> dx2 [BT,H*P], ddt2 [BT,H], dAo [H,1], dB2 [BT,G*N], dC2 [BT,G*N], dDo [H,1]

        TWO STRUCTURAL CHOICES WORTH THE COMMENT:

        1. BOTH ORIENTATIONS OF THE [Q,Q] TENSORS FOR FREE. dB contracts over t
           while dC contracts over s, so the score/decay matrices are needed both
           ways. Transposing them would cost an nc_transpose + a PSUM tile each,
           and PSUM is the binding constraint. They can be had without one because
           every [Q,Q] tensor here is either an outer product (transpose = swap
           the two matmul operands, free) or a function of ell alone. For the
           latter, `tensor_scalar(ellb, subtract, ell)` gives [i,j] = ell_j - ell_i
           and the same call with `reverse0=True` gives [i,j] = ell_i - ell_j,
           which read at (i,j)=(t,s) IS seg[s,t]. So two elementwise ops plus two
           constant masks (MASK, MASKT) give L and L^T with no tensor-engine work.
           (seg is NOT symmetric -- reading ONE raw tile both ways silently
           negates the exponent. That bug cost a hardware round trip; the float64
           simulator in ssd_nki_sim.py now mirrors this exact dataflow so it
           cannot recur.)

        2. dB/dC ARE ACCUMULATED IN-KERNEL ACROSS HEADS. With n_groups < n_heads
           (the real config is G=1, H=10) B and C are shared, so their gradients
           sum over the hpg heads of a group. Rather than spilling per-head
           partials to HBM and reducing them with an extra eager op, the kernel
           keeps a [Q, CPB*N] accumulator per group covering every chunk (8 KB per
           partition at the real geometry) and writes it once per group.

        The forward state S_c entering each chunk is RECOMPUTED (pass 1) rather
        than saved by the forward kernel: saving it would be CPB x [N,P] x H x nb
        = 21 MB per layer of extra saved-tensor memory plus an extra HBM write in
        the forward, versus ~3 extra matmuls per chunk here.
        """
        BT, HP = x2.shape
        H = dt2.shape[1]
        P = HP // H
        N = B2.shape[1] // G
        nb = BT // T
        CPB = T // Q
        hpg = H // G
        fp = nl.float32

        assert N <= Q <= _NKI_PARTITION and P <= 512 and T % Q == 0 and H % G == 0

        dx2 = nl.ndarray((BT, HP), dtype=fp, buffer=nl.shared_hbm)
        ddt2 = nl.ndarray((BT, H), dtype=fp, buffer=nl.shared_hbm)
        dB2 = nl.ndarray((BT, G * N), dtype=fp, buffer=nl.shared_hbm)
        dC2 = nl.ndarray((BT, G * N), dtype=fp, buffer=nl.shared_hbm)
        dAo = nl.ndarray((H, 1), dtype=fp, buffer=nl.shared_hbm)
        dDo = nl.ndarray((H, 1), dtype=fp, buffer=nl.shared_hbm)

        # ---- constants ---------------------------------------------------
        ri = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
        nisa.iota(dst=ri, pattern=[[0, Q]], offset=0, channel_multiplier=1)
        ci = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
        nisa.iota(dst=ci, pattern=[[1, Q]], offset=0, channel_multiplier=0)
        UI = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=UI, data1=ri, data2=ci, op=nl.less_equal)   # [r,t] r<=t
        LI = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=LI, data1=ri, data2=ci, op=nl.greater_equal)  # [t,r] r<=t
        onesQ = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
        nisa.memset(dst=onesQ, value=1.0)
        MASK = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=MASK, data=UI, op0=nl.subtract, operand0=1.0)
        nisa.tensor_scalar(dst=MASK, data=MASK, op0=nl.multiply, operand0=-_SSD_NEG)
        MASKT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=MASKT, data=LI, op0=nl.subtract, operand0=1.0)
        nisa.tensor_scalar(dst=MASKT, data=MASKT, op0=nl.multiply, operand0=-_SSD_NEG)

        ab = nl.ndarray((Q, H), dtype=fp, buffer=nl.sbuf)
        nisa.dma_copy(dst=ab, src=Ab[0:Q, 0:H])
        db = nl.ndarray((Q, H), dtype=fp, buffer=nl.sbuf)
        nisa.dma_copy(dst=db, src=Db[0:Q, 0:H])

        # dA_h / dD_h are sums over every token; accumulate elementwise into a
        # [Q,H] tile and do ONE partition-axis reduction (a single matmul) at the
        # very end.
        accA = nl.ndarray((Q, H), dtype=fp, buffer=nl.sbuf)
        nisa.memset(dst=accA, value=0.0)
        accD = nl.ndarray((Q, H), dtype=fp, buffer=nl.sbuf)
        nisa.memset(dst=accD, value=0.0)

        for bi in range(nb):
          for g in range(G):
            aB = nl.ndarray((Q, CPB * N), dtype=fp, buffer=nl.sbuf)
            nisa.memset(dst=aB, value=0.0)
            aC = nl.ndarray((Q, CPB * N), dtype=fp, buffer=nl.sbuf)
            nisa.memset(dst=aC, value=0.0)

            for hh in range(hpg):
                h = g * hpg + hh

                # ---- pass 1: the forward state entering each chunk --------
                Sall = nl.ndarray((N, CPB * P), dtype=fp, buffer=nl.sbuf)
                S = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
                nisa.memset(dst=S, value=0.0)
                for c in range(CPB):
                    nisa.tensor_copy(dst=Sall[0:N, c * P:c * P + P], src=S)
                    if c + 1 == CPB:
                        break
                    r0 = bi * T + c * Q
                    xt = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                    nisa.dma_copy(dst=xt, src=x2[r0:r0 + Q, h * P:h * P + P])
                    dtt = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.dma_copy(dst=dtt, src=dt2[r0:r0 + Q, h:h + 1])
                    Bt = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.dma_copy(dst=Bt, src=B2[r0:r0 + Q, g * N:g * N + N])

                    dtA = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=dtA, data=dtt, op0=nl.multiply,
                                       operand0=ab[0:Q, h:h + 1])
                    u = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=u, data=xt, op0=nl.multiply, operand0=dtt)

                    p1 = nl.ndarray((Q, 1), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=p1, stationary=UI, moving=dtA,
                                   is_stationary_onezero=True)
                    ell = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ell, src=p1)
                    UIdtA = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=UIdtA, data=UI, op0=nl.multiply, operand0=dtA)
                    p2 = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=p2, stationary=onesQ, moving=UIdtA,
                                   is_stationary_onezero=True)
                    ellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ellb, src=p2)
                    eellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=eellb, data=ellb, op0=nl.minimum, operand0=0.0)
                    nisa.activation(dst=eellb, op=nl.exp, data=eellb)

                    rev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=rev, data1=ellb[0:Q, Q - 1:Q], data2=ell,
                                       op=nl.subtract)
                    nisa.tensor_scalar(dst=rev, data=rev, op0=nl.minimum, operand0=0.0)
                    erev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.activation(dst=erev, op=nl.exp, data=rev)
                    Bbar = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=Bbar, data=Bt, op0=nl.multiply, operand0=erev)
                    p3 = nl.ndarray((N, P), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=p3, stationary=Bbar, moving=u)
                    Gs = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=Gs, src=p3)
                    nisa.tensor_scalar(dst=S, data=S, op0=nl.multiply,
                                       operand0=eellb[0:N, Q - 1:Q])
                    nisa.tensor_tensor(dst=S, data1=S, data2=Gs, op=nl.add)

                # ---- pass 2: reverse over chunks -------------------------
                dS = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
                nisa.memset(dst=dS, value=0.0)

                for c in range(CPB - 1, -1, -1):
                    r0 = bi * T + c * Q
                    xt = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                    nisa.dma_copy(dst=xt, src=x2[r0:r0 + Q, h * P:h * P + P])
                    dtt = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.dma_copy(dst=dtt, src=dt2[r0:r0 + Q, h:h + 1])
                    Bt = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.dma_copy(dst=Bt, src=B2[r0:r0 + Q, g * N:g * N + N])
                    Ct = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.dma_copy(dst=Ct, src=C2[r0:r0 + Q, g * N:g * N + N])
                    gy = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                    nisa.dma_copy(dst=gy, src=gy2[r0:r0 + Q, h * P:h * P + P])

                    # --- recompute the forward intermediates
                    dtA = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=dtA, data=dtt, op0=nl.multiply,
                                       operand0=ab[0:Q, h:h + 1])
                    u = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=u, data=xt, op0=nl.multiply, operand0=dtt)

                    q1 = nl.ndarray((Q, 1), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=q1, stationary=UI, moving=dtA,
                                   is_stationary_onezero=True)
                    ell = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ell, src=q1)
                    UIdtA = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=UIdtA, data=UI, op0=nl.multiply, operand0=dtA)
                    q2 = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=q2, stationary=onesQ, moving=UIdtA,
                                   is_stationary_onezero=True)
                    ellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ellb, src=q2)
                    eellb = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=eellb, data=ellb, op0=nl.minimum, operand0=0.0)
                    nisa.activation(dst=eellb, op=nl.exp, data=eellb)
                    eellc = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=eellc, data=ell, op0=nl.minimum, operand0=0.0)
                    nisa.activation(dst=eellc, op=nl.exp, data=eellc)

                    # L and L^T without a transpose (see the docstring). NOTE the
                    # two raw tiles: seg is NOT symmetric, so reading one tile at
                    # [t,s] gives ell_s - ell_t, the NEGATION of what L^T needs.
                    # reverse0=True produces the other orientation directly.
                    raw = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=raw, data=ellb, op0=nl.subtract, operand0=ell,
                                       op1=nl.minimum, operand1=0.0)
                    tmp = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=tmp, data1=raw, data2=MASK, op=nl.add)
                    L = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.activation(dst=L, op=nl.exp, data=tmp)
                    nisa.tensor_scalar(dst=raw, data=ellb, op0=nl.subtract, operand0=ell,
                                       reverse0=True, op1=nl.minimum, operand1=0.0)
                    nisa.tensor_tensor(dst=tmp, data1=raw, data2=MASKT, op=nl.add)
                    LT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.activation(dst=LT, op=nl.exp, data=tmp)

                    q3 = nl.ndarray((N, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_transpose(dst=q3, data=Bt)
                    BTs = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=BTs, src=q3)
                    q4 = nl.ndarray((N, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_transpose(dst=q4, data=Ct)
                    CTs = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=CTs, src=q4)

                    q5 = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=q5, stationary=BTs, moving=CTs)   # CB[s,t]
                    MT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=MT, data1=L, data2=q5, op=nl.multiply)
                    q6 = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=q6, stationary=CTs, moving=BTs)   # CB^T[t,s]
                    MTT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=MTT, data1=LT, data2=q6, op=nl.multiply)

                    q7 = nl.ndarray((P, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_transpose(dst=q7, data=gy)
                    gyT = nl.ndarray((P, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=gyT, src=q7)
                    q8 = nl.ndarray((P, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_transpose(dst=q8, data=u)
                    uT = nl.ndarray((P, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=uT, src=q8)

                    # --- dD_h += <gy_t, x_t>
                    gx = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=gx, data1=gy, data2=xt, op=nl.multiply)
                    rsum = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=rsum, op=nl.add, data=gx, axis=(1,))
                    nisa.tensor_tensor(dst=accD[0:Q, h:h + 1], data1=accD[0:Q, h:h + 1],
                                       data2=rsum, op=nl.add)

                    # --- intra-chunk
                    q9 = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=q9, stationary=uT, moving=gyT)      # dMT[s,t]
                    dMT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dMT, src=q9)
                    qa = nl.ndarray((Q, Q), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=qa, stationary=gyT, moving=uT)      # dMT^T[t,s]
                    dMTT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dMTT, src=qa)

                    dseg = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=dseg, data1=dMT, data2=MT, op=nl.multiply)
                    dCB = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=dCB, data1=dMT, data2=L, op=nl.multiply)
                    dCBT = nl.ndarray((Q, Q), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=dCBT, data1=dMTT, data2=LT, op=nl.multiply)

                    # dell_j = sum_s dseg[s,j] (partition reduce -> matmul)
                    #          - sum_t dseg[j,t]
                    qb = nl.ndarray((Q, 1), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=qb, stationary=dseg, moving=onesQ[0:Q, 0:1],
                                   is_moving_onezero=True)
                    dell = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dell, src=qb)
                    nisa.tensor_reduce(dst=rsum, op=nl.add, data=dseg, axis=(1,))
                    nisa.tensor_tensor(dst=dell, data1=dell, data2=rsum, op=nl.subtract)

                    qc = nl.ndarray((Q, N), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=qc, stationary=dCBT, moving=Ct)     # dB[s,n]
                    dB = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dB, src=qc)
                    qd = nl.ndarray((Q, N), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=qd, stationary=dCB, moving=Bt)      # dC[t,n]
                    dC = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dC, src=qd)

                    # du accumulates two matmuls -> deliberately one PSUM tile
                    qe = nl.ndarray((Q, P), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=qe, stationary=MTT, moving=gy)      # du[s,p]

                    # --- inter-chunk output term (S_0 == 0, nothing at c == 0)
                    if c > 0:
                        qf = nl.ndarray((P, N), dtype=fp, buffer=nl.psum)
                        nisa.nc_transpose(dst=qf, data=Sall[0:N, c * P:c * P + P])
                        STs = nl.ndarray((P, N), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=STs, src=qf)
                        qg = nl.ndarray((Q, N), dtype=fp, buffer=nl.psum)
                        nisa.nc_matmul(dst=qg, stationary=gyT, moving=STs)  # dCbar[t,n]
                        dCbar = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dCbar, src=qg)
                        scr = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=scr, data=dCbar, op0=nl.multiply,
                                           operand0=eellc)
                        nisa.tensor_tensor(dst=dC, data1=dC, data2=scr, op=nl.add)
                        nisa.tensor_tensor(dst=scr, data1=dCbar, data2=Ct, op=nl.multiply)
                        nisa.tensor_reduce(dst=rsum, op=nl.add, data=scr, axis=(1,))
                        nisa.tensor_tensor(dst=rsum, data1=rsum, data2=eellc,
                                           op=nl.multiply)
                        nisa.tensor_tensor(dst=dell, data1=dell, data2=rsum, op=nl.add)

                    # --- the carry into the next chunk's state (dS is dS_{c+1})
                    have_dde = c + 1 < CPB
                    if have_dde:
                        qh = nl.ndarray((P, N), dtype=fp, buffer=nl.psum)
                        nisa.nc_transpose(dst=qh, data=dS)
                        dST = nl.ndarray((P, N), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dST, src=qh)
                        qi = nl.ndarray((Q, N), dtype=fp, buffer=nl.psum)
                        nisa.nc_matmul(dst=qi, stationary=uT, moving=dST)  # dBbar[s,n]
                        dBbar = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dBbar, src=qi)

                        rev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=rev, data1=ellb[0:Q, Q - 1:Q], data2=ell,
                                           op=nl.subtract)
                        nisa.tensor_scalar(dst=rev, data=rev, op0=nl.minimum, operand0=0.0)
                        erev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                        nisa.activation(dst=erev, op=nl.exp, data=rev)
                        # Bbar^T = B^T * erev broadcast on partitions; erev_s =
                        # exp(min(de-ell_s,0)) and eellb[p,s] = exp(min(ell_s,0))
                        # with de at column Q-1, so the broadcast comes straight
                        # out of eellb.
                        revb = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=revb, data=ellb[0:N, 0:Q], op0=nl.subtract,
                                           operand0=ellb[0:N, Q - 1:Q], reverse0=True,
                                           op1=nl.minimum, operand1=0.0)
                        nisa.activation(dst=revb, op=nl.exp, data=revb)
                        BbarT = nl.ndarray((N, Q), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=BbarT, data1=BTs, data2=revb,
                                           op=nl.multiply)
                        nisa.nc_matmul(dst=qe, stationary=BbarT, moving=dS)  # du += ...

                        scr2 = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=scr2, data=dBbar, op0=nl.multiply,
                                           operand0=erev)
                        nisa.tensor_tensor(dst=dB, data1=dB, data2=scr2, op=nl.add)
                        nisa.tensor_tensor(dst=scr2, data1=dBbar, data2=Bt,
                                           op=nl.multiply)
                        drev = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_reduce(dst=drev, op=nl.add, data=scr2, axis=(1,))
                        nisa.tensor_tensor(dst=drev, data1=drev, data2=erev,
                                           op=nl.multiply)
                        nisa.tensor_tensor(dst=dell, data1=dell, data2=drev,
                                           op=nl.subtract)

                        # dde = sum_s drev_s + <dS_{c+1}, S_c> * ac. Both are
                        # partition-axis reductions, so both are matmuls against a
                        # ones column -- accumulated into one [1,1] PSUM tile.
                        qj = nl.ndarray((1, 1), dtype=fp, buffer=nl.psum)
                        nisa.nc_matmul(dst=qj, stationary=drev, moving=onesQ[0:Q, 0:1],
                                       is_moving_onezero=True)
                        pr0 = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=pr0, data1=dS,
                                           data2=Sall[0:N, c * P:c * P + P],
                                           op=nl.multiply)
                        pr1 = nl.ndarray((N, 1), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_reduce(dst=pr1, op=nl.add, data=pr0, axis=(1,))
                        nisa.tensor_tensor(dst=pr1, data1=pr1, data2=eellb[0:N, Q - 1:Q],
                                           op=nl.multiply)
                        nisa.nc_matmul(dst=qj, stationary=pr1, moving=onesQ[0:N, 0:1],
                                       is_moving_onezero=True)
                        dde = nl.ndarray((1, 1), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dde, src=qj)

                    # --- dell -> ddtA (reverse inclusive cumsum), then dt / A / x
                    qk = nl.ndarray((Q, 1), dtype=fp, buffer=nl.psum)
                    nisa.nc_matmul(dst=qk, stationary=LI, moving=dell,
                                   is_stationary_onezero=True)
                    if have_dde:
                        # de = ell_{Q-1}, and ddtA_r = sum_{t>=r} dell_t, so the
                        # dde contribution reaches EVERY r -- a broadcast add, not
                        # a one-hot injection. Broadcasting [1,1] onto Q partitions
                        # is a matmul with a ones row, accumulating into the same
                        # tile.
                        nisa.nc_matmul(dst=qk, stationary=onesQ[0:1, 0:Q], moving=dde,
                                       is_stationary_onezero=True)
                    ddtA = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ddtA, src=qk)

                    du = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=du, src=qe)
                    ddt = nl.ndarray((Q, 1), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=ddt, data=ddtA, op0=nl.multiply,
                                       operand0=ab[0:Q, h:h + 1])
                    nisa.tensor_tensor(dst=gx, data1=du, data2=xt, op=nl.multiply)
                    nisa.tensor_reduce(dst=rsum, op=nl.add, data=gx, axis=(1,))
                    nisa.tensor_tensor(dst=ddt, data1=ddt, data2=rsum, op=nl.add)
                    nisa.tensor_tensor(dst=rsum, data1=ddtA, data2=dtt, op=nl.multiply)
                    nisa.tensor_tensor(dst=accA[0:Q, h:h + 1], data1=accA[0:Q, h:h + 1],
                                       data2=rsum, op=nl.add)

                    dxt = nl.ndarray((Q, P), dtype=fp, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=dxt, data=du, op0=nl.multiply, operand0=dtt)
                    nisa.tensor_scalar(dst=gx, data=gy, op0=nl.multiply,
                                       operand0=db[0:Q, h:h + 1])
                    nisa.tensor_tensor(dst=dxt, data1=dxt, data2=gx, op=nl.add)

                    nisa.dma_copy(dst=dx2[r0:r0 + Q, h * P:h * P + P], src=dxt)
                    nisa.dma_copy(dst=ddt2[r0:r0 + Q, h:h + 1], src=ddt)
                    nisa.tensor_tensor(dst=aB[0:Q, c * N:c * N + N],
                                       data1=aB[0:Q, c * N:c * N + N], data2=dB,
                                       op=nl.add)
                    nisa.tensor_tensor(dst=aC[0:Q, c * N:c * N + N],
                                       data1=aC[0:Q, c * N:c * N + N], data2=dC,
                                       op=nl.add)

                    # --- dS_c = ac_c dS_{c+1} + (inter term of chunk c)
                    if c > 0:
                        Cbar = nl.ndarray((Q, N), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=Cbar, data=Ct, op0=nl.multiply,
                                           operand0=eellc)
                        ql = nl.ndarray((N, P), dtype=fp, buffer=nl.psum)
                        nisa.nc_matmul(dst=ql, stationary=Cbar, moving=gy)
                        dSy = nl.ndarray((N, P), dtype=fp, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dSy, src=ql)
                        nisa.tensor_scalar(dst=dS, data=dS, op0=nl.multiply,
                                           operand0=eellb[0:N, Q - 1:Q])
                        nisa.tensor_tensor(dst=dS, data1=dS, data2=dSy, op=nl.add)

            for c in range(CPB):
                r0 = bi * T + c * Q
                nisa.dma_copy(dst=dB2[r0:r0 + Q, g * N:g * N + N],
                              src=aB[0:Q, c * N:c * N + N])
                nisa.dma_copy(dst=dC2[r0:r0 + Q, g * N:g * N + N],
                              src=aC[0:Q, c * N:c * N + N])

        # ---- dA / dD: one partition-axis reduction for all H heads at once
        pa = nl.ndarray((H, 1), dtype=fp, buffer=nl.psum)
        nisa.nc_matmul(dst=pa, stationary=accA, moving=onesQ[0:Q, 0:1],
                       is_moving_onezero=True)
        sa = nl.ndarray((H, 1), dtype=fp, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sa, src=pa)
        nisa.dma_copy(dst=dAo, src=sa)
        pd = nl.ndarray((H, 1), dtype=fp, buffer=nl.psum)
        nisa.nc_matmul(dst=pd, stationary=accD, moving=onesQ[0:Q, 0:1],
                       is_moving_onezero=True)
        sd = nl.ndarray((H, 1), dtype=fp, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sd, src=pd)
        nisa.dma_copy(dst=dDo, src=sd)

        return dx2, ddt2, dAo, dB2, dC2, dDo

    class _SSDNKI(torch.autograd.Function):
        """Differentiable wrapper around the SSD kernel pair.

        Both kernels want the same flat [b*T, ...] token-major layout, and A/D
        replicated across all 128 partitions (a compute engine cannot broadcast
        along the partition axis, so the replication is done host-side once per
        call -- it is H floats wide). Nothing but the flat views is saved for
        backward: the kernel recomputes every per-chunk intermediate itself.
        """

        @staticmethod
        def forward(ctx, x, dt, A, B, C, D, Q):
            b, T, H, P = x.shape
            G, N = B.shape[2], B.shape[3]
            x2 = x.reshape(b * T, H * P).contiguous()
            dt2 = dt.reshape(b * T, H).contiguous()
            B2 = B.reshape(b * T, G * N).contiguous()
            C2 = C.reshape(b * T, G * N).contiguous()
            Ab = A.reshape(1, H).expand(_NKI_PARTITION, H).contiguous()
            Db = D.reshape(1, H).expand(_NKI_PARTITION, H).contiguous()
            y2 = ssd_fwd_nki(x2, dt2, B2, C2, Ab, Db, T, Q, G)
            ctx.save_for_backward(x2, dt2, B2, C2, Ab, Db)
            ctx.ssd_dims = (b, T, H, P, G, N, Q)
            return y2.reshape(b, T, H, P)

        @staticmethod
        def backward(ctx, grad_y):
            x2, dt2, B2, C2, Ab, Db = ctx.saved_tensors
            b, T, H, P, G, N, Q = ctx.ssd_dims
            gy2 = grad_y.reshape(b * T, H * P).float().contiguous()
            dx2, ddt2, dAo, dB2, dC2, dDo = ssd_bwd_nki(
                gy2, x2, dt2, B2, C2, Ab, Db, T, Q, G)
            return (dx2.reshape(b, T, H, P), ddt2.reshape(b, T, H), dAo.reshape(H),
                    dB2.reshape(b, T, G, N), dC2.reshape(b, T, G, N), dDo.reshape(H),
                    None)

    _HAS_NKI = True
except Exception:  # nki unavailable (e.g. CPU/GPU dev box) — eager path still works
    _HAS_NKI = False

# Toggled by the --nki-relu2 flag in main(). Default OFF: the shipped baseline is
# numerically identical to the plain eager `relu(x)**2`.
USE_NKI_RELU2 = False
# Toggled by the --nki-rope-norm flag in main(); can also be turned off via that
# flag/env var if you need the bit-for-bit-reproducible eager
# `norm(apply_rotary_emb(...))` chain for comparison. Default ON: Solution 12,
# numerically verified to 1-bf16-ULP against eager (fwd+bwd, research/scratch/
# rope_norm_equiv.py) and confirmed at the real submission shape (8-way,
# TOTAL_BATCH_SIZE=131072): ~15-18% real per-step speedup, causality exact,
# quality difference is noise-level (loss trajectories match to 4 decimals). See
# trn2-experiment-results.md.
USE_NKI_ROPE_NORM = True
# Toggled by the --nki-softcap-ce flag in main(); default ON (the rope+norm pattern):
# verified numerically offline (research/scratch/softcap_ce_equiv.py, worst case 2.44
# bf16 ULP fwd / 0.22 bf16 ULP bwd, an fp32-limited edge case -- realistic lm_head-scale
# logits are ~0.001-0.002 bf16 ULP) and confirmed on real hardware at the final locked
# shape: 384 steps/1.198478 vs the same-lineage baseline's 366/1.206231 (real win) and a
# near-tie against an older baseline missing a separate free fix -- no comparison shows
# it worse than the best number on record. See trn2-experiment-results.md.
USE_NKI_SOFTCAP_CE = True
# Toggled by the --nki-ssd flag in main(); only ever consulted when USE_MAMBA2 is
# on. Default ON: the kernel pair is a real measured win over the eager scan at
# the per-layer shape (b=1 T=2048 H=10 P=128 N=128 Q=128) -- 4.53x forward,
# 3.83x fwd+bwd -- because the eager form lowers to ~25 separate NEFFs while the
# kernel is one NEFF that keeps every intermediate in SBUF. Equivalence to the
# eager scan is checked on device by research/scratch/ssd_nki_dev.py (fwd
# max_rel < 2e-5, bwd < 5e-4, across 5 shapes) and the eager scan is itself
# checked exactly against a naive sequential recurrence on CPU. Flip it off to
# get the bit-reproducible eager scan back for comparison.
USE_NKI_SSD = True


def softcap_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                          loss_reduction: str) -> torch.Tensor:
    """`LOGIT_SOFTCAP*tanh(z/S)` then cross-entropy, fused into one NKI kernel pair.

    Replaces only the softcap+CE math. The kernel produces the unreduced per-token
    loss, so the mean/none branching below is identical to what F.cross_entropy's
    own `reduction` argument did -- reduction semantics are unchanged either way,
    and the eager branch is the original chain verbatim so flipping the knob off
    restores the previous numerics exactly.
    """
    if USE_NKI_SOFTCAP_CE and _HAS_NKI and logits.device.type == "neuron" \
            and loss_reduction in ("mean", "none") and logits.dtype == torch.float32 \
            and logits.is_contiguous():
        z = logits.reshape(-1, logits.shape[-1])
        # The kernel tiles rows in blocks of 128 (true for the locked shape:
        # B*T = 4096); anything else falls back to eager so the knob is always safe.
        if z.shape[0] % _NKI_PARTITION == 0:
            tgt = targets.reshape(-1, 1).to(torch.float32)
            per_token = _SoftcapCrossEntropyNKI.apply(z, tgt)
            if loss_reduction == "none":
                return per_token.view_as(targets)
            return per_token.mean()
    capped = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
    loss = F.cross_entropy(
        capped.reshape(-1, capped.size(-1)),
        targets.reshape(-1),
        reduction=loss_reduction,
    )
    if loss_reduction == "none":
        return loss.view_as(targets)
    return loss


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


def rope_norm(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """norm(apply_rotary_emb(x, cos, sin)), fused into one NKI kernel when enabled.

    x is [B, T, H, D]; cos/sin are [1, T, 1, D//2] broadcast over batch and heads.
    The kernel needs B*T and T to be multiples of 128 (both true for the baseline:
    B*T = 2*1024, T = 1024) so that each 128-row tile maps to one contiguous run of
    positions within a single batch row. Anything else falls back to eager, so the
    knob is always safe to flip.
    """
    if USE_NKI_ROPE_NORM and _HAS_NKI and x.device.type == "neuron" and x.ndim == 4:
        b, t, _, d = x.shape
        if (b * t) % _NKI_PARTITION == 0 and t % _NKI_PARTITION == 0 and d % 2 == 0 \
                and cos.shape[1] == t and cos.shape[-1] == d // 2 and x.is_contiguous():
            return _RopeNormNKI.apply(x, cos, sin)
    return norm(apply_rotary_emb(x, cos, sin))


# -----------------------------------------------------------------------------
# Mamba2 / SSD (arXiv:2405.21060). Only reached when USE_MAMBA2 is on.


def _ssd_masks(Q: int, NC: int, device, dtype):
    """The five constant masks the chunked scan needs, built once per (T, device).

    cumsum_op   [Q,Q]   lower-triangular ones -- the within-chunk inclusive cumsum
                        expressed as a matmul (no torch.cumsum: this repo's
                        native-eager backend handles matmul far better, and it is
                        exactly what the NKI kernel does internally).
    intra_add   [Q,Q]   0 where s <= t, _SSD_NEG otherwise (the causal mask).
    inter_add   [NC,NC] 0 where c' <  c, _SSD_NEG otherwise -- STRICTLY lower
                        triangular, because chunk c's incoming state may only see
                        chunks that finished before it.
    excl_op     [NC,NC] exclusive cumsum over chunks, as a matmul.
    incl_op     [NC,NC] inclusive cumsum over chunks, as a matmul.
    """
    qi = torch.arange(Q, device=device)
    zero = torch.zeros((), device=device, dtype=dtype)
    neg = torch.full((), _SSD_NEG, device=device, dtype=dtype)
    tril_q = qi[:, None] >= qi[None, :]
    ci = torch.arange(NC, device=device)
    return (tril_q.to(dtype), torch.where(tril_q, zero, neg),
            torch.where(ci[:, None] > ci[None, :], zero, neg),
            (ci[:, None] > ci[None, :]).to(dtype), (ci[:, None] >= ci[None, :]).to(dtype))


def ssd_chunked(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                C: torch.Tensor, D: torch.Tensor, Q: int, masks) -> torch.Tensor:
    """Loop-free chunked SSD scan (the "structured state-space duality" form).

    x: [b,T,H,P]  dt: [b,T,H] (>0)  A: [H] (<0)  B,C: [b,T,G,N]  D: [H]
    -> y: [b,T,H,P]

    NO python loop over chunks and no cumsum. The inter-chunk state recurrence
    S_{c+1} = a_c S_c + G_c has a SCALAR a_c, so it collapses to the same
    masked-decay matmul form as the intra-chunk term, just at chunk granularity
    (NC x NC instead of Q x Q) -- two levels of one trick, zero sequential steps.
    That matters because the native-eager backend lowers each op to its own NEFF,
    so a 16-step loop over chunks would cost 16x the dispatch.

    Deliberately dtype-transparent (no internal cast): the caller decides the
    accumulation dtype. Mamba2Mixer feeds it fp32; the CPU equivalence harness
    feeds it float64 and gets agreement with a naive sequential recurrence to
    max_rel < 1e-12, which only works if nothing here silently downcasts.

    Every exponent is clamped to <= 0 BEFORE exp and the mask is ADDITIVE, so
    masked entries are exactly exp(-30000) == 0.0 and no +inf * 0 can arise. Every
    live entry provably has a <= 0 exponent (A < 0, dt > 0), so the clamp is
    exact rather than an approximation.
    """
    b, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    NC = T // Q
    hpg = H // G
    cumsum_op, intra_add, inter_add, excl_op, incl_op = masks
    # -> [b, NC, H, Q, ...]: chunk and head become batch dims of the matmuls
    xc = x.reshape(b, NC, Q, H, P).permute(0, 1, 3, 2, 4)
    dtc = dt.reshape(b, NC, Q, H).permute(0, 1, 3, 2)
    Bf = B.reshape(b, NC, Q, G, N).permute(0, 1, 3, 2, 4)
    Cf = C.reshape(b, NC, Q, G, N).permute(0, 1, 3, 2, 4)
    if hpg > 1:
        Bf = Bf.repeat_interleave(hpg, dim=2)          # groups -> heads
        Cf = Cf.repeat_interleave(hpg, dim=2)
    dx = dtc.unsqueeze(-1) * xc
    # within-chunk inclusive cumsum of dt*A, as a matmul: ell[...,t] = sum_{s<=t}
    ell = torch.matmul(dtc * A.reshape(1, 1, H, 1), cumsum_op.transpose(0, 1))
    seg = ell.unsqueeze(-1) - ell.unsqueeze(-2)        # [b,NC,H,Qt,Qs]
    Lc = torch.exp(seg.clamp(max=0.0) + intra_add)     # masked BEFORE exp
    Y = torch.matmul(Lc * torch.matmul(Cf, Bf.transpose(-1, -2)), dx)
    # per-chunk state contribution G_c
    de = ell[..., -1]                                  # [b,NC,H], <= 0
    Bbar = Bf * torch.exp((de.unsqueeze(-1) - ell).clamp(max=0.0)).unsqueeze(-1)
    Gst = torch.matmul(Bbar.transpose(-1, -2), dx)     # [b,NC,H,N,P]
    # inter-chunk carry in closed form:
    # S_in[c] = sum_{c'<c} exp(Pexcl[c] - Pincl[c']) G_{c'}, exponent <= 0
    deh = de.permute(0, 2, 1)
    Pex = torch.matmul(deh, excl_op.transpose(0, 1))
    Pin = torch.matmul(deh, incl_op.transpose(0, 1))
    L2 = torch.exp((Pex.unsqueeze(-1) - Pin.unsqueeze(-2)).clamp(max=0.0) + inter_add)
    S_in = torch.matmul(L2, Gst.permute(0, 2, 1, 3, 4).reshape(b, H, NC, N * P))
    S_in = S_in.reshape(b, H, NC, N, P).permute(0, 2, 1, 3, 4)
    Cbar = Cf * torch.exp(ell.clamp(max=0.0)).unsqueeze(-1)
    if hpg == 1 and G == 1:
        Cbar = Cbar.expand(b, NC, H, Q, N)
    Y = Y + torch.matmul(Cbar, S_in)
    y = Y.permute(0, 1, 3, 2, 4).reshape(b, T, H, P)
    return y + D.reshape(1, 1, H, 1) * x


def ssd_scan(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
             C: torch.Tensor, D: torch.Tensor, Q: int, masks) -> torch.Tensor:
    """The chunked SSD scan, fused into one NKI kernel pair when enabled.

    Same fallback contract as rope_norm/softcap_cross_entropy: the kernel needs
    fp32 contiguous inputs and N <= Q <= 128 (the SBUF partition limit, which is
    also what lets the kernel broadcast exp(ell) along partitions for free),
    P <= 512 (the PSUM free-dim limit) and T % Q == 0. All four hold at the locked
    shape (P=128, N=128, Q=128, T=2048); anything else falls back to the eager
    scan, so the knob is always safe to flip.
    """
    if USE_NKI_SSD and _HAS_NKI and x.device.type == "neuron" and x.ndim == 4:
        b, T, H, P = x.shape
        G, N = B.shape[2], B.shape[3]
        if N <= Q <= _NKI_PARTITION and P <= 512 and T % Q == 0 and H % G == 0 \
                and x.dtype == torch.float32 and dt.dtype == torch.float32 \
                and B.dtype == torch.float32 and C.dtype == torch.float32 \
                and x.is_contiguous() and dt.is_contiguous() \
                and B.is_contiguous() and C.is_contiguous():
            return _SSDNKI.apply(x, dt, A, B, C, D, Q)
    return ssd_chunked(x, dt, A, B, C, D, Q, masks)


def ssd_causal_conv1d(u: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv1d over the k shifted copies of u.

    Written as k shifts + k muls + adds rather than F.conv1d: on the native-eager
    Neuron backend a depthwise grouped conv1d is not a well-supported primitive
    while pad/slice/mul/add are, and k == 4 so this is only 4 terms. Exactly
    equal to F.conv1d(pad(u, (k-1, 0)), w[:, None, :], groups=C).

    Causal by construction: the pad is on the LEFT, so output position t reads
    only u[t-k+1 .. t] and the first k-1 positions see zeros.
    """
    k = weight.shape[-1]
    out = None
    for j in range(k):
        shift = k - 1 - j                                  # u[t - shift]
        us = u if shift == 0 else F.pad(u[:, :-shift, :], (0, 0, shift, 0))
        term = us * weight[:, j]
        out = term if out is None else out + term
    return out


def ssd_gated_rms_norm(y: torch.Tensor, z: torch.Tensor, weight: torch.Tensor,
                       eps: float = 1e-5) -> torch.Tensor:
    """Mamba2's gated RMSNorm: rmsnorm(y * silu(z)) * weight over the d_inner axis."""
    h = y.float() * F.silu(z.float())
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)) * weight.float()


class Mamba2Mixer(nn.Module):
    """A real Mamba2 mixer, used in place of CausalSelfAttention on "M" layers.

    in_proj -> depthwise causal conv1d -> silu -> selective SSD scan -> gated
    RMSNorm -> out_proj. Follows the reference Mamba2 block (state-spaces/mamba)
    but with this repo's `Linear` (bias-free, dtype-casting, weight-cache aware)
    instead of tensor-parallel layers, and the loop-free chunked scan above
    instead of an XLA kernel.

    One projection produces all three streams at once, as in real Mamba2:
        [ z (gate, d_inner) | xBC (conv'd, d_conv) | dt (n_heads) ]
    with d_conv = d_inner + 2*n_groups*d_state.

    Geometry comes from the MAMBA2_* module constants rather than the config so
    that the toggles can be swept from the CLI without touching GPTConfig.

    Numerics: the scan runs in fp32 regardless of COMPUTE_DTYPE. dt is a softplus
    output that gets exponentiated inside the scan, so bf16's 8-bit mantissa there
    would change the decay by percent-level amounts; the cast is at the call site
    rather than inside ssd_chunked so the CPU harness can still exercise the scan
    in float64.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        n_embd = config.n_embd
        self.n_embd = n_embd
        self.d_state = MAMBA2_D_STATE
        self.head_dim = MAMBA2_HEAD_DIM
        self.d_inner = MAMBA2_EXPAND * n_embd
        assert self.d_inner % self.head_dim == 0, (self.d_inner, self.head_dim)
        self.n_heads = self.d_inner // self.head_dim
        self.n_groups = MAMBA2_N_GROUPS
        assert self.n_heads % self.n_groups == 0
        self.conv_kernel = MAMBA2_CONV_KERNEL
        self.chunk = MAMBA2_CHUNK
        self.d_conv = self.d_inner + 2 * self.n_groups * self.d_state
        self.in_proj = Linear(n_embd, self.d_inner + self.d_conv + self.n_heads, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(self.d_conv, self.conv_kernel))
        self.dt_bias = nn.Parameter(torch.empty(self.n_heads))
        self.A_log = nn.Parameter(torch.empty(self.n_heads))
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.norm_weight = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = Linear(self.d_inner, n_embd, bias=False)
        self._masks = None
        self.reset_ssm_parameters()

    @torch.no_grad()
    def reset_ssm_parameters(self) -> None:
        """The SSM-specific init, done here so a bare Mamba2Mixer is already sane.

        GPT.init_weights only touches in_proj/out_proj (the two ordinary matrices),
        exactly as it does for attn/mlp.
        """
        H = self.n_heads
        # dt_bias = softplus^-1(dt) for a LOG-uniform dt in [dt_min, dt_max]: the
        # real Mamba/Mamba2 init. Guarantees the discretization step starts in a
        # sane range instead of wherever softplus(0) lands.
        dt = torch.exp(torch.rand(H) * (math.log(MAMBA2_DT_MAX) - math.log(MAMBA2_DT_MIN))
                       + math.log(MAMBA2_DT_MIN)).clamp(min=MAMBA2_DT_INIT_FLOOR)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        # A = -exp(A_log) with A_log = log(1..H): the "S4D-real" scalar-per-head
        # init, giving each head a different decay timescale. A < 0 always, which
        # is what makes every scan exponent <= 0 and the clamp exact.
        self.A_log.copy_(torch.log(torch.arange(1, H + 1, dtype=torch.float32)))
        self.D.fill_(1.0)
        self.norm_weight.fill_(1.0)
        # depthwise causal conv: uniform with fan_in = kernel_size, matching
        # nn.Conv1d's default for groups == channels. NOT zero-init -- that was the
        # local_conv dead-parameter failure mode (a zero-init depthwise conv has
        # zero gradient forever because its own weight is the only path).
        bound = 1.0 / math.sqrt(self.conv_kernel)
        self.conv_weight.uniform_(-bound, bound)

    def _get_masks(self, T: int, device, dtype):
        key = (T, device, dtype)
        if self._masks is None or self._masks[0] != key:
            self._masks = (key, _ssd_masks(self.chunk, T // self.chunk, device, dtype))
        return self._masks[1]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: [b, T, n_embd] -> [b, T, n_embd]"""
        b, T, _ = u.shape
        H, P, N, G = self.n_heads, self.head_dim, self.d_state, self.n_groups
        proj = self.in_proj(u)
        z, xBC, dt_raw = torch.split(proj, [self.d_inner, self.d_conv, H], dim=-1)
        xBC = F.silu(ssd_causal_conv1d(xBC, self.conv_weight.to(xBC.dtype)))
        x, Bm, Cm = torch.split(xBC, [self.d_inner, G * N, G * N], dim=-1)
        dt = F.softplus(dt_raw.float() + self.dt_bias.float())            # [b,T,H] > 0
        A = -torch.exp(self.A_log.float())                                # [H] < 0
        y = ssd_scan(x.reshape(b, T, H, P).float().contiguous(), dt, A,
                     Bm.reshape(b, T, G, N).float().contiguous(),
                     Cm.reshape(b, T, G, N).float().contiguous(),
                     self.D.float(), self.chunk,
                     self._get_masks(T, u.device, torch.float32))
        h = ssd_gated_rms_norm(y.reshape(b, T, self.d_inner), z, self.norm_weight)
        return self.out_proj(h.to(u.dtype))


def mamba2_layer_flags(n_layer: int) -> list[bool]:
    """Which layers are Mamba2 mixers, from MAMBA2_PATTERN cycled to n_layer."""
    if not USE_MAMBA2:
        return [False] * n_layer
    pattern = MAMBA2_PATTERN or "M"
    cycled = (pattern * (n_layer // len(pattern) + 1))[:n_layer]
    return [c == "M" for c in cycled]


class _RMSNormNoScale(nn.Module):
    """Parameter-free RMS norm over the last dim (DCMHA's dw1_norm)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype)


class _DCMHAWeightGen(nn.Module):
    """One of DCMHA's four dynamic-weight generators (pre-Q, pre-K, post-Q, post-K).

    Produces a per-position, per-head low-rank pair (w1, w2) and a per-position,
    per-head diagonal/gating term (dd) from that position's own hidden state x --
    exactly the qw1/qw2/qdd (or kw1/kw2/kdd) triple in the reference implementation
    (DynamicWeightProjection, github.com/Caiyun-AI/DCFormer). Zero-init no-op:
    `dd` is fully zero-initialized (dd=tanh(0)=0 regardless of x) and `l2`'s
    SECOND half (the w2 output rows) is zero-initialized -- so the low-rank term
    `qhidden @ w2` is exactly 0 at step 0 regardless of w1 -- Compose(inputs)=inputs
    exactly (see _cross_head_compose). Only zeroing w2, not w1 too, matters: w1 and
    w2 both feed one bilinear product downstream (qhidden = inputs @ w1; term =
    qhidden @ w2), and if BOTH factors of a product are zero simultaneously the
    gradient into EITHER factor vanishes identically (verified empirically in
    research/scratch/mudd_dcmha_equiv.py -- an earlier version of this class zeroed
    all of l2 and silently had zero gradient into l2.weight forever, a dead
    mechanism that would never move away from its no-op point). Zeroing only w2
    keeps w1 at its normal small-random init, so d(loss)/d(w2) = qhidden (nonzero,
    since w1 is nonzero) -- gradient flows into w2 from step 1, and once w2 moves
    off zero, w1's own gradient (which flows via w2) becomes live too.
    """

    def __init__(self, n_embd: int, n_head: int, rank: int, hidden: int):
        super().__init__()
        self.rank = rank
        self.n_head = n_head
        self.l1 = Linear(n_embd, hidden, bias=False)
        self.l2 = Linear(hidden, 2 * rank * n_head, bias=False)
        self.dd = Linear(n_embd, n_head, bias=False)
        self.norm = _RMSNormNoScale()
        with torch.no_grad():
            self.l2.weight[rank * n_head:].zero_()  # w2 half only -- see class docstring
            self.dd.weight.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.gelu(self.l1(x))
        w1, w2 = self.l2(h).chunk(2, dim=-1)
        w1 = self.norm(w1.view(*w1.shape[:-1], self.rank, self.n_head))
        w2 = w2.view(*w2.shape[:-1], self.rank, self.n_head)
        dd = torch.tanh(self.dd(x))
        return w1, w2, dd


def _cross_head_compose(inputs: torch.Tensor, qw1: torch.Tensor, qw2: torch.Tensor, kw1: torch.Tensor,
                         kw2: torch.Tensor, qdd: torch.Tensor, kdd: torch.Tensor) -> torch.Tensor:
    """DCMHA's Compose function (arXiv:2405.08553 Eq. 7 / reference `_cross_head_proj`).

    inputs: [B,H,T,S] (the raw score matrix pre-softmax, or the weight matrix
    post-softmax). qw1/qw2/qdd: [B,T,R,H]/[B,T,H] (generated from the QUERY
    position's own hidden state). kw1/kw2/kdd: [B,S,R,H]/[B,S,H] (from the KEY
    position's own hidden state). Every term below is `inputs` multiplied by a
    factor that depends only on the fixed (t,s) cell being computed and the
    already-existing head dimension -- no cell (t,s) ever reads a value from a
    different cell (t',s'), and the whole function is linear/homogeneous in
    `inputs` (no additive bias independent of `inputs`). That homogeneity is
    exactly why this is causally safe post-softmax: at any masked cell the
    causal mask has already zeroed `inputs` for EVERY head uniformly, so
    Compose(0)=0 there too and cannot resurrect a masked position (see USE_DCMHA
    comment above for the full argument).
    """
    out = inputs
    qhidden = torch.einsum('bhts,btrh->brts', inputs, qw1)
    out = out + torch.einsum('brts,btrh->bhts', qhidden, qw2)
    khidden = torch.einsum('bhts,bsrh->brts', inputs, kw1)
    out = out + torch.einsum('brts,bsrh->bhts', khidden, kw2)
    out = out + torch.einsum('bhts,bth->bhts', inputs, qdd)
    out = out + torch.einsum('bhts,bsh->bhts', inputs, kdd)
    return out


class DCMHACompose(nn.Module):
    """DCMHA (arXiv:2405.08553): generates the pre-softmax and post-softmax
    Compose arguments for one attention layer from its (already-normed) input x."""

    def __init__(self, n_embd: int, n_head: int, layer_idx: int = 0):
        super().__init__()
        rank = max(1, n_head // _DCMHA_RANK_DIVISOR)
        hidden = max(4, n_head * 4)
        self.pre_q = _DCMHAWeightGen(n_embd, n_head, rank, hidden)
        self.pre_k = _DCMHAWeightGen(n_embd, n_head, rank, hidden)
        self.post_q = _DCMHAWeightGen(n_embd, n_head, rank, hidden)
        self.post_k = _DCMHAWeightGen(n_embd, n_head, rank, hidden)

    def forward(self, x: torch.Tensor):
        # Always compute in fp32, regardless of x's incoming dtype: the caller's
        # `logits`/`probs` this feeds into are also fp32 (matching the reference's
        # own fp32-for-softmax-stability choice), and x's own dtype is NOT reliably
        # bf16 -- this repo's resid_lambda/x0_lambda are fp32 params, so
        # resid_lambda*x upcasts the residual stream to fp32 on every non-MUDD
        # layer, while MUDD's stream (no such fp32 scalar) stays bf16 -- decoupling
        # here avoids a dtype-mismatch crash when the two toggles combine.
        x = x.float()
        qw1, qw2, qdd = self.pre_q(x)
        kw1, kw2, kdd = self.pre_k(x)
        pw1, pw2, pdd = self.post_q(x)
        rw1, rw2, rdd = self.post_k(x)
        return (qw1, qw2, kw1, kw2, qdd, kdd), (pw1, pw2, rw1, rw2, pdd, rdd)


def _dcmha_keep_mask(t: int, attn_mask: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    """Boolean [1,1,T,T]-or-[B,1,T,T] keep mask (True = visible), matching the
    semantics `attn_mask is None` already has elsewhere in this file (full causal,
    no doc/window masking) plus the additive-float-mask convention (0 = visible,
    -inf = masked) used when attn_mask is not None."""
    if attn_mask is None:
        return torch.ones(t, t, dtype=torch.bool, device=device).tril()[None, None]
    return attn_mask[..., :t, :t] == 0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim * self.n_head == self.n_embd
        if USE_FUSED_QKV:
            self.c_qkv = Linear(config.n_embd, 3 * config.n_embd, bias=False)
        else:
            self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
            self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
            self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate = Linear(VE_GATE_CHANNELS, self.n_head, bias=False) if USE_VE and has_ve(layer_idx, config.n_layer) else None
        self.dcmha = DCMHACompose(config.n_embd, self.n_head, layer_idx) if USE_DCMHA else None

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None, v0: torch.Tensor | None = None, v0_lambda: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c = x_q.shape
        if USE_FUSED_QKV:
            # Fused QKV needs one shared input; MUDD's separate Q/K/V streams collapse
            # to x_q in this combination (untested/discouraged -- both default OFF).
            q, k, v = self.c_qkv(x_q).split(self.n_embd, dim=-1)
            q = q.view(b, t, self.n_head, self.head_dim)
            k = k.view(b, t, self.n_head, self.head_dim)
            v = v.view(b, t, self.n_head, self.head_dim)
        else:
            q = self.c_q(x_q).view(b, t, self.n_head, self.head_dim)
            k = self.c_k(x_k).view(b, t, self.n_head, self.head_dim)
            v = self.c_v(x_v).view(b, t, self.n_head, self.head_dim)
        if ve is not None:
            ve = ve.view(b, t, self.n_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x_v[..., :VE_GATE_CHANNELS]))
            v = v + gate.unsqueeze(-1) * ve
        v_raw = v
        if v0 is not None:
            # Value Residual Learning (arXiv:2410.17897): v0 is layer-0's own v_raw, same
            # shape [b,t,n_head,head_dim], causally trivial (both are pure per-position
            # functions of x, no cross-position mixing beyond what SDPA already does).
            v = v + v0_lambda * v0
        q = rope_norm(q, cos, sin)
        k = rope_norm(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.dcmha is not None:
            # Unfused path: DCMHA's Compose needs the raw [B,H,T,T] score/weight
            # matrices, which F.scaled_dot_product_attention never materializes.
            pre_args, post_args = self.dcmha(x_q)
            scale = 1.0 / math.sqrt(self.head_dim)
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            logits = _cross_head_compose(logits, *pre_args)
            keep = _dcmha_keep_mask(t, attn_mask, x_q.device)
            logits = logits.masked_fill(~keep, torch.finfo(torch.float32).min)
            probs = logits.softmax(dim=-1)
            probs = _cross_head_compose(probs, *post_args).to(v.dtype)
            y = torch.matmul(probs, v)
        elif attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[..., :t, :t])
        if USE_XSA:
            # Exclusive Self Attention (arXiv:2603.09078, Zhai/Apple): project out the
            # component of the attention output along the token's OWN value vector --
            # z_i = y_i - (y_i . v_i / ||v_i||^2) * v_i. Causally safe by construction:
            # z_i depends only on y_i and v_i, both already functions of x_1..i only.
            # y, v are [B, H, T, D] here (post-transpose, pre-reshape).
            proj_coef = (y * v).sum(dim=-1, keepdim=True) / (v.pow(2).sum(dim=-1, keepdim=True) + _XSA_EPS)
            y = y - proj_coef * v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y), v_raw


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


class MuddCompose(nn.Module):
    """MUDDFormer's multiway dynamic dense connections (arXiv:2502.12170, Eq. 5-8).

    Replaces the single learned resid_lambda/x0_lambda skip with FOUR independent
    per-token dynamic weighted sums over ALL earlier hidden states (embedding +
    every prior block's output) -- one each for this block's Q input, K input,
    V input, and residual-stream base. `num_sources` = layer_idx + 1 is the count
    of hidden states available to this layer (X_0 = the embedding, X_1..X_{i}
    = the outputs of blocks 0..i-1), matching the paper's j=0..i summation.

    Causally trivial: `w1`/`w2` project the layer's OWN current hidden state
    (per-position, no cross-position mixing -- an RMSNorm+matmul is a pure
    per-token map) into per-position weights, and the weighted sum
    `sum_j A[...,j] * X_j` only recombines information already present at that
    SAME token position from earlier LAYERS -- it never reads a different
    position, so it is exactly as safe as the resid_lambda/x0_lambda scalar it
    replaces.

    Zero-init no-op: `w2` is zero-initialized, so at step 0 the dynamic term
    (GELU(RMSNorm(X_i) @ w1) @ w2) is exactly 0 regardless of input, leaving
    A == bias. `bias` is set to reproduce the exact resid_lambda=1.0/x0_lambda=0.05
    formula (weight 1.0 on this layer's own immediate input X_i, weight 0.05 on
    the initial embedding X_0, 0 elsewhere) for all 4 streams identically --
    USE_MUDD=True is bit-identical to USE_MUDD=False at init.
    """

    def __init__(self, n_embd: int, num_sources: int, layer_idx: int):
        super().__init__()
        self.num_sources = num_sources
        self.w1 = nn.Parameter(torch.empty(n_embd, 4, num_sources))
        torch.nn.init.normal_(self.w1, std=n_embd**-0.5)
        self.w2 = nn.Parameter(torch.zeros(4, num_sources, num_sources))
        target = torch.zeros(4, num_sources)
        target[:, layer_idx] += 1.0
        target[:, 0] += 0.05
        self.bias = nn.Parameter(target)

    def forward(self, hidden_states: list[torch.Tensor], query_state: torch.Tensor):
        h = F.gelu(torch.einsum('btd,dsj->btsj', norm(query_state).float(), self.w1.float()))
        a = torch.einsum('btsj,sjk->btsk', h, self.w2.float()) + self.bias.float()
        stacked = torch.stack(hidden_states[: self.num_sources], dim=0)  # [J,B,T,D]
        out = torch.einsum('btsj,jbtd->sbtd', a.to(stacked.dtype), stacked)
        return out[0], out[1], out[2], out[3]  # q_in, k_in, v_in, resid_in


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        # MAMBA2_PATTERN decides, per layer, whether this block's sequence mixer is
        # attention or a Mamba2 SSD mixer. Exactly one of attn/mixer exists, so
        # neither path ever carries the other's dead parameters.
        self.is_mamba = mamba2_layer_flags(config.n_layer)[layer_idx]
        self.attn = None if self.is_mamba else CausalSelfAttention(config, layer_idx)
        self.mixer = Mamba2Mixer(config) if self.is_mamba else None
        self.mlp = MLP(config)
        self.mudd = MuddCompose(config.n_embd, layer_idx + 1, layer_idx) if USE_MUDD else None
        self.resid_lambda = nn.Parameter(torch.ones(1)) if self.mudd is None else None
        # x0 skip: mixes initial embedding into each layer's input (stabilizes deep nets)
        self.x0_lambda = nn.Parameter(torch.zeros(1)) if self.mudd is None else None
        self.local_conv = CausalConv(config.n_embd, LOCAL_CONV_KERNEL) if USE_LOCAL_CONV else None
        self.scalar_gate = Linear(SELECTIVE_GATE_CHANNELS, 2, bias=False) if USE_SELECTIVE_GATE else None
        # Value Residual Learning (arXiv:2410.17897): layer 0 has no v0 to consume yet (it
        # produces the v0 every later layer will use), so it doesn't need its own lambda.
        # A Mamba2 layer has no attention V to blend, so it gets no v_lambda either.
        self.v_lambda = nn.Parameter(torch.zeros(1)) \
            if USE_VALUE_RESIDUAL and layer_idx > 0 and not self.is_mamba else None

    def forward(self, x: torch.Tensor, x0: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None, v0: torch.Tensor | None = None, hidden_states: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mudd is not None:
            q_in, k_in, v_in, x = self.mudd(hidden_states, hidden_states[-1])
        else:
            if self.scalar_gate is not None:
                g = self.scalar_gate(x[..., :SELECTIVE_GATE_CHANNELS])
                resid_lambda = self.resid_lambda * (1.0 + g[..., 0:1])
                x0_lambda = self.x0_lambda * (1.0 + g[..., 1:2])
                x = resid_lambda * x + x0_lambda * x0
            else:
                x = self.resid_lambda * x + self.x0_lambda * x0
            q_in = k_in = v_in = x
        if self.local_conv is not None:
            x = x + self.local_conv(norm(x))
        if self.mixer is not None:
            # A Mamba2 layer produces no v_raw: there is no attention V here, so it
            # cannot seed the value residual. GPT.forward keeps v0 at None until the
            # first ATTENTION layer, which is the only correct thing to do anyway --
            # a Mamba2 layer's state is not a per-token value vector.
            x = x + self.mixer(norm(q_in))
            v_raw = None
        else:
            attn_out, v_raw = self.attn(norm(q_in), norm(k_in), norm(v_in), ve, cos, sin, attn_mask, v0, self.v_lambda)
            x = x + attn_out
        x = x + self.mlp(norm(x))
        return x, v_raw


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
        self.layer_is_mamba = [blk.is_mamba for blk in self.transformer.h]
        # A value embedding is injected into ATTENTION's V projection, so a Mamba2
        # layer has nowhere to put one. Skipping the table entirely (rather than
        # building it and never reading it) keeps the parameter count honest -- at
        # the locked shape with MMAMMM every has_ve layer happens to be Mamba2, so
        # this is the difference between 45.0M and 60.7M parameters, all of it dead.
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, config.n_embd)
            for i in range(config.n_layer)
            if USE_VE and has_ve(i, config.n_layer) and not self.layer_is_mamba[i]
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
            if block.mixer is not None:
                # Same treatment as attention/MLP: the input projection gets the
                # shared uniform init, the output projection is zeroed so the whole
                # block starts as an exact identity. Everything SSM-specific
                # (conv_weight, dt_bias, A_log, D, norm_weight) was already set by
                # Mamba2Mixer.reset_ssm_parameters and must NOT be clobbered here.
                torch.nn.init.uniform_(block.mixer.in_proj.weight, -s, s)
                torch.nn.init.zeros_(block.mixer.out_proj.weight)
                if block.resid_lambda is not None:
                    block.resid_lambda.fill_(1.0)
                    block.x0_lambda.fill_(0.05)
                if block.scalar_gate is not None:
                    torch.nn.init.zeros_(block.scalar_gate.weight)
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
                if block.mlp.c_gate is not None:
                    torch.nn.init.uniform_(block.mlp.c_gate.weight, -0.4 * s, 0.4 * s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
                continue
            if USE_FUSED_QKV:
                # Slice-wise, not one uniform_ over the whole fused tensor: keeps each
                # third's per-tensor distribution identical to the unfused baseline (RNG
                # draw order differs -- expected, not a bug -- see Solution 13c).
                n = self.config.n_embd
                torch.nn.init.uniform_(block.attn.c_qkv.weight[:n], -s, s)
                torch.nn.init.uniform_(block.attn.c_qkv.weight[n:2 * n], -s, s)
                torch.nn.init.uniform_(block.attn.c_qkv.weight[2 * n:], -s, s)
            else:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
            if block.mlp.c_gate is not None:
                torch.nn.init.uniform_(block.mlp.c_gate.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.resid_lambda is not None:
                block.resid_lambda.fill_(1.0)
                block.x0_lambda.fill_(0.05)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.scalar_gate is not None:
                torch.nn.init.zeros_(block.scalar_gate.weight)
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
        ssm_params = []  # Mamba2's non-matrix state-space tensors, see below
        hypernet_params = []  # MUDD/DCMHA's own generator weights: real matrices, not
        # true scalars like resid_lambda -- SCALAR_LR=0.20 is tuned for a single learned
        # scalar's limited blast radius. A real hardware A/B first put MuddCompose.w1
        # (a genuine [n_embd,4,num_sources] projection) in the SCALAR_LR group and hit
        # a real NaN divergence around step ~270-280 (loss climbing 4.7->6+ then NaN,
        # while the same optimizer/schedule trains the baseline stably past 900+ steps)
        # -- a plausible cause given AdamW's adaptive per-parameter scaling can amplify
        # an 8x-too-aggressive LR (0.20 vs MATRIX_LR=0.025) on a real weight matrix over
        # a few hundred steps. Not a fit for Muon's same-shape orthogonalization groups
        # either (tiny, varies in shape per layer/branch), so it gets its own AdamW group
        # at MATRIX_LR instead.
        for block in self.transformer.h:
            if block.resid_lambda is not None:
                scalar_params.extend([block.resid_lambda, block.x0_lambda])
            if block.mudd is not None:
                hypernet_params.extend([block.mudd.w1, block.mudd.w2, block.mudd.bias])
            if block.v_lambda is not None:
                scalar_params.append(block.v_lambda)
            if block.mixer is not None:
                # in_proj / out_proj are ordinary weight matrices -> Muon, grouped
                # by shape like every other block matrix.
                matrix_params.extend([block.mixer.in_proj.weight,
                                      block.mixer.out_proj.weight])
                # The five SSM tensors are NOT Muon material: dt_bias/A_log/D are
                # per-head 1-D vectors (H = 10 elements) and norm_weight is a 1-D
                # RMSNorm gain, none of which has a second axis to orthogonalize,
                # and conv_weight is a DEPTHWISE conv kernel [d_conv, k=4] whose
                # rows are independent channels -- orthogonalizing across channels
                # would be meaningless (and near-degenerate at k=4). They also must
                # not go in the SCALAR_LR=0.20 group: that LR is tuned for a single
                # learned scalar's blast radius and already caused a real NaN
                # divergence when a genuine weight matrix was put there (see the
                # hypernet_params comment above). So they get AdamW at MATRIX_LR
                # with weight_decay=0.0, which also matches the reference Mamba2's
                # no_weight_decay set for A_log/D/dt_bias.
                ssm_params.extend([block.mixer.conv_weight, block.mixer.dt_bias,
                                   block.mixer.A_log, block.mixer.D,
                                   block.mixer.norm_weight])
                matrix_params.extend(block.mlp.parameters())
                continue
            if block.attn.dcmha is not None:
                # DCMHA's hypernetwork weights are tiny and vary in shape per
                # layer/branch -- not a fit for Muon's same-shape orthogonalization
                # groups, so route them through AdamW alongside the scalar/skip
                # params (excluded from block.attn.parameters() below by name).
                hypernet_params.extend(block.attn.dcmha.parameters())
            matrix_params.extend(p for name, p in block.attn.named_parameters() if not name.startswith("dcmha."))
            matrix_params.extend(block.mlp.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        value_embed_params = list(self.value_embeds.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, EMBED_ADAMW_BETA2), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=scalar_params, lr=SCALAR_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if hypernet_params:
            param_groups.append(dict(kind="adamw", params=hypernet_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if ssm_params:
            param_groups.append(dict(kind="adamw", params=ssm_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if value_embed_params:
            param_groups.append(dict(kind="adamw", params=value_embed_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, EMBED_ADAMW_BETA2), eps=1e-10, weight_decay=VE_WD))
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
        v0 = None
        # MUDDFormer (arXiv:2502.12170): each block's dense-connection generator
        # (Block.mudd) needs every earlier hidden state (the embedding plus every
        # prior block's output), not just x0/x_prev -- kept only when USE_MUDD so
        # the non-MUDD path pays no extra memory for activations it never uses.
        hidden_states = [x0] if USE_MUDD else None
        doc_mask = doc_boundary_mask(idx) if USE_DOC_MASK else None
        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](idx).to(COMPUTE_DTYPE) if str(i) in self.value_embeds else None
            if self.layer_is_mamba[i]:
                # No attention here, so don't materialize an additive mask this
                # layer will never read. The Mamba2 mixer is causal structurally
                # (left-padded conv + masked-before-exp decay), not by mask.
                attn_mask = None
            elif USE_CAUSAL_FASTPATH and doc_mask is None and not self.layer_is_short[i]:
                # Solution 13d: this layer's static mask is purely causal (full context,
                # no doc-boundary masking) -- let SDPA take its is_causal fast path
                # instead of materializing the additive mask tensor.
                attn_mask = None
            else:
                static_mask = (self.mask_short if self.layer_is_short[i] else self.mask_full)[..., :t, :t]
                attn_mask = static_mask + doc_mask if doc_mask is not None else static_mask
            x, v_raw = block(x, x0, ve, cos, sin, attn_mask, v0, hidden_states)
            if USE_MUDD:
                hidden_states.append(x)
            if USE_VALUE_RESIDUAL and v0 is None:
                v0 = v_raw
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        if targets is None:
            return LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        return softcap_cross_entropy(logits, targets, loss_reduction)


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
        self._scalar_cache: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}

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
            entry = (torch.empty((), dtype=torch.float32),
                     torch.zeros((), dtype=torch.float32, device=ref.device))
            cache[key] = entry
        host, dev = entry
        host.fill_(float(value))
        dev.copy_(host)
        return dev

    @torch.no_grad()
    def _adamw_step(self, group):
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        if USE_TENSOR_LR_SCALARS and group["params"]:
            lr = self._device_scalar(group, "lr", lr, group["params"][0])
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
            p.mul_(1 - lr * group["weight_decay"])
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            update = exp_avg / bias1 / ((exp_avg_sq / bias2).sqrt() + group["eps"])
            if USE_TENSOR_LR_SCALARS:
                # alpha= is the one place a Python float becomes a baked graph
                # constant; as a tensor multiply it stays a runtime operand.
                p.sub_(update * lr)
            else:
                p.add_(update, alpha=-lr)

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

        lr_base = group["lr"]
        if USE_TENSOR_LR_SCALARS:
            lr_base = self._device_scalar(group, "lr", lr_base, p0)
        lr = lr_base * max(1.0, shape[-2] / shape[-1]) ** 0.5
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

ARCH_TOGGLE_NAMES = [
    "PACK_FACTOR", "INIT_SCALE", "USE_VE", "WINDOW_PATTERN", "SHORT_WINDOW_FRAC",
    "ROPE_BASE", "USE_LOCAL_CONV", "LOCAL_CONV_KERNEL", "USE_GATED_MLP",
    "USE_WEIGHT_CACHE", "WARMDOWN_RATIO", "FINAL_LR_FRAC", "WARMUP_STEPS",
    "WD_DECAY_TO_ZERO", "MUON_MOMENTUM_WARMUP_STEPS", "USE_DOC_MASK", "VE_WD",
    "USE_SELECTIVE_GATE", "SELECTIVE_GATE_CHANNELS", "USE_FUSED_QKV",
    "USE_CAUSAL_FASTPATH", "USE_STOCHASTIC_ROUNDING", "USE_TRN2_COMPILER_FLAGS",
    "USE_REQUEUE_PACKER", "USE_XSA", "USE_VALUE_RESIDUAL", "USE_MUDD", "USE_DCMHA",
    "USE_MAMBA2", "MAMBA2_PATTERN", "MAMBA2_HEAD_DIM", "MAMBA2_D_STATE",
    "MAMBA2_CHUNK", "MAMBA2_EXPAND", "MAMBA2_N_GROUPS", "MAMBA2_CONV_KERNEL",
]
# Toggles whose mismatch corrupts the state_dict shape (load_state_dict(strict=True) will
# error loudly). Everything else in ARCH_TOGGLE_NAMES lives in non-persistent buffers or
# plain Python constants excluded from state_dict() entirely -- a mismatch there loads
# silently with no crash and quietly wrong behavior, which is exactly why we snapshot and
# check all of them, not just the shape-critical ones.
SHAPE_CRITICAL_TOGGLES = {"USE_VE", "USE_LOCAL_CONV", "USE_GATED_MLP", "USE_SELECTIVE_GATE", "USE_FUSED_QKV", "USE_VALUE_RESIDUAL", "USE_MUDD", "USE_DCMHA",
                          # Every MAMBA2_* below changes which parameter tensors exist
                          # or their shapes: the pattern swaps attn <-> mixer per layer
                          # (and drops that layer's value_embeds table), and head_dim /
                          # d_state / expand / n_groups / conv_kernel all resize
                          # in_proj, conv_weight and the per-head vectors. MAMBA2_CHUNK
                          # is the one that is NOT shape-critical -- it only changes how
                          # the same math is blocked -- so it stays out of this set.
                          "USE_MAMBA2", "MAMBA2_PATTERN", "MAMBA2_HEAD_DIM",
                          "MAMBA2_D_STATE", "MAMBA2_EXPAND", "MAMBA2_N_GROUPS",
                          "MAMBA2_CONV_KERNEL"}


def current_toggles() -> dict:
    g = globals()
    return {name: g[name] for name in ARCH_TOGGLE_NAMES}


def save_checkpoint(path: Path, model: GPT, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": asdict(model.config),
        "toggles": current_toggles(),
        "meta": meta,
    }
    torch.save(payload, path)


def warm_start_from(path: str, model: GPT) -> dict:
    """Load a checkpoint's weights into a freshly-built model (fresh optimizer, fresh step=0).
    Not a bit-exact resume -- no optimizer/RNG/dataloader state. For dev-iteration only; never
    for a scored submission (each submission must be a fresh run from scratch, per the rules).
    """
    payload = torch_load_weights(path, map_location="cpu")
    saved_toggles = payload.get("toggles", {})
    live_toggles = current_toggles()
    mismatches = {k: (saved_toggles[k], live_toggles[k]) for k in saved_toggles if k in live_toggles and saved_toggles[k] != live_toggles[k]}
    if mismatches:
        for name, (saved, live) in mismatches.items():
            tag = "SHAPE-CRITICAL" if name in SHAPE_CRITICAL_TOGGLES else "silent-behavior"
            print0(f"WARNING [{tag}]: toggle {name} mismatch -- checkpoint={saved!r} current={live!r}")
    model.load_state_dict(payload["model_state"], strict=True)
    meta = payload.get("meta", {})
    print0(f"warm-started from {path}: checkpoint step={meta.get('step')} total_train_tokens={meta.get('total_train_tokens')}")
    return meta


def torch_load_weights(path: str | Path, map_location):
    return torch.load(path, map_location=map_location, weights_only=True)


def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    global COMPUTE_DTYPE, BOS_TOKEN_ID
    COMPUTE_DTYPE = compute_dtype_for(device)
    if USE_DOC_MASK:
        BOS_TOKEN_ID = ensure_tokenizer(build_if_missing=False).get_bos_token_id()
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
    lrm = cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
    if LRM_QUANTUM > 0:
        # Snap the multiplier onto a coarse grid. Measured on real hardware: the
        # optimizer's per-step Python-float lr becomes a baked constant in the
        # Neuron graph, so a NEW lr value every step means a fresh neuronx-cc
        # compile every step -- dt goes 0.90s -> 2.2-4.3s the moment the cooldown
        # starts (i.e. for the last WARMDOWN_RATIO of every run). Quantising the
        # schedule caps the number of distinct lr values, so only the first step
        # at each level pays a compile and the rest reuse it.
        lrm = round(lrm * LRM_QUANTUM) / LRM_QUANTUM
    return lrm


def muon_momentum(step: int) -> float:
    frac = min(step / MUON_MOMENTUM_WARMUP_STEPS, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


def main() -> None:
    global USE_LOCAL_CONV, LRM_QUANTUM, USE_TENSOR_LR_SCALARS, USE_MUDD, USE_DCMHA, USE_MAMBA2, MAMBA2_PATTERN, USE_NKI_SSD  # flags below default off these constants
    global MAMBA2_EXPAND, MAMBA2_D_STATE, MAMBA2_HEAD_DIM, MAMBA2_CHUNK, MAMBA2_N_GROUPS, MAMBA2_CONV_KERNEL
    parser = argparse.ArgumentParser(description="Neuron Competition R1 baseline")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_NUM_STEPS", NUM_STEPS)))
    parser.add_argument("--max-train-seconds", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS", MAX_TRAIN_SECONDS)))
    parser.add_argument("--out-dir", type=str, default=os.environ.get("NEURON_COMPETITION_R1_OUT_DIR", "out"))
    parser.add_argument("--eval-public", action=argparse.BooleanOptionalAction, default=os.environ.get("NEURON_COMPETITION_R1_EVAL_PUBLIC", "1") != "0")
    parser.add_argument("--eval-tokens", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_EVAL_TOKENS", str(4 * 524288))))
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_COMPILE", "1") != "0",
                        help="enable torch.compile(backend='neuron') on the full model. Default ON "
                             "(Bucket E, real full-1800s-confirmed win: 827 steps/1.113006 vs eager's "
                             "746 steps/1.122271, causality exact) -- pass --no-compile for the plain "
                             "eager comparison path.")
    parser.add_argument("--compile-only", action="store_true",
                        help="run one synthetic optimizer step to warm the compile cache, then exit")
    parser.add_argument("--nki-relu2", action="store_true", default=os.environ.get("NEURON_COMPETITION_R1_NKI_RELU2", "0") != "0",
                        help="use the example fused NKI relu(x)**2 kernel in the MLP (Neuron only; illustration)")
    parser.add_argument("--nki-rope-norm", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_NKI_ROPE_NORM", "1") != "0",
                        help="fuse norm(apply_rotary_emb(x)) into one NKI kernel for q/k (Neuron only). "
                             "Default ON (Solution 12, real ~15-18% step-time win, verified to 1-bf16-ULP "
                             "against eager) -- pass --no-nki-rope-norm for the plain eager comparison path.")
    parser.add_argument("--nki-softcap-ce", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_NKI_SOFTCAP_CE", "1") != "0",
                        help="fuse the logit softcap and cross-entropy (fwd+bwd) into one NKI kernel pair "
                             "(Neuron only), collapsing ~8 full-[B*T,V] NEFF launches into 2. Default ON: "
                             "verified numerically offline (research/scratch/softcap_ce_equiv.py) and confirmed "
                             "on real hardware at the final locked shape -- pass --no-nki-softcap-ce for the "
                             "plain eager comparison path.")
    parser.add_argument("--local-conv", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_LOCAL_CONV", "1" if USE_LOCAL_CONV else "0") != "0",
                        help="depthwise causal conv1d before attention in each block. Provably dead "
                             "compute as wired (zero weight, in no optimizer group -- see the comment on "
                             "USE_LOCAL_CONV), so --no-local-conv is a bit-identical model; exposed as a "
                             "flag so the A/B runs from one identical binary.")
    parser.add_argument("--tensor-lr-scalars", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_TENSOR_LR_SCALARS", "1" if USE_TENSOR_LR_SCALARS else "0") != "0",
                        help="hold the per-step lr in a device tensor rather than a Python float, so a "
                             "changing LR schedule stops forcing a fresh neuronx-cc compile every step")
    parser.add_argument("--lrm-quantum", type=int,
                        default=int(os.environ.get("NEURON_COMPETITION_R1_LRM_QUANTUM", LRM_QUANTUM)),
                        help="snap the LR multiplier onto a 1/N grid (0 = continuous). Each distinct "
                             "lr value costs one neuronx-cc compile, so a continuous cooldown "
                             "recompiles the optimizer every step.")
    parser.add_argument("--mudd", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_MUDD", "1" if USE_MUDD else "0") != "0",
                        help="MUDDFormer multiway dynamic dense connections (arXiv:2502.12170) in place of "
                             "the resid_lambda/x0_lambda skip -- zero-init no-op at step 0, exposed as a "
                             "flag so the A/B runs from one identical binary.")
    parser.add_argument("--dcmha", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_DCMHA", "1" if USE_DCMHA else "0") != "0",
                        help="Dynamically Composable Multi-Head Attention (arXiv:2405.08553) -- unfuses "
                             "F.scaled_dot_product_attention into an explicit QK^T/mask/compose/softmax/"
                             "compose/@V sequence so Compose can see the raw [B,H,T,T] matrices. Zero-init "
                             "no-op at step 0, exposed as a flag so the A/B runs from one identical binary.")
    parser.add_argument("--mamba2", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_MAMBA2", "1" if USE_MAMBA2 else "0") != "0",
                        help="Mamba2/SSD hybrid (arXiv:2405.21060): replace the layers marked \"M\" in "
                             "--mamba2-pattern with a Mamba2Mixer (causal conv1d + selective state-space "
                             "scan) instead of attention, keeping the MLP and skips. Zero-init no-op at "
                             "step 0, exposed as a flag so the A/B runs from one identical binary. Known "
                             "real negative on wall clock at the locked shape -- see the USE_MAMBA2 comment.")
    parser.add_argument("--mamba2-pattern", type=str, default=MAMBA2_PATTERN,
                        help="per-layer mixer pattern cycled to n_layer, \"M\" = Mamba2 mixer and anything "
                             "else = attention (default MMAMMM: 5 Mamba2 layers + one full attention layer "
                             "at index 2). Only read when --mamba2 is on.")
    parser.add_argument("--mamba2-expand", type=int, default=MAMBA2_EXPAND,
                        help="d_inner = this x n_embd. The Mamba2 paper default is 2 (d_inner=1280 "
                             "at the locked n_embd=640, H=10 heads); 1 halves every projection AND "
                             "the scan's head count. Only read when --mamba2 is on.")
    parser.add_argument("--mamba2-d-state", type=int, default=MAMBA2_D_STATE,
                        help="N, the SSM state width per B/C group (paper default 128). Must satisfy "
                             "N <= --mamba2-chunk for the NKI kernel. Only read when --mamba2 is on.")
    parser.add_argument("--mamba2-head-dim", type=int, default=MAMBA2_HEAD_DIM,
                        help="P, the per-head state width; must divide d_inner and be <= 512 (the "
                             "PSUM free-dim limit). Only read when --mamba2 is on.")
    parser.add_argument("--mamba2-chunk", type=int, default=MAMBA2_CHUNK,
                        help="Q, the SSD chunk length; must divide sequence_len and satisfy "
                             "N <= Q <= 128 for the NKI kernel. Only read when --mamba2 is on.")
    parser.add_argument("--mamba2-n-groups", type=int, default=MAMBA2_N_GROUPS,
                        help="G, the number of B/C groups (paper default 1 = all heads share one "
                             "B/C). Must divide n_heads. Only read when --mamba2 is on.")
    parser.add_argument("--mamba2-conv-kernel", type=int, default=MAMBA2_CONV_KERNEL,
                        help="k for the depthwise causal conv1d (paper default 4). Costs k shifted "
                             "mul-adds over [b,T,d_conv] per mixer. Only read when --mamba2 is on.")
    parser.add_argument("--nki-ssd", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_NKI_SSD", "1" if USE_NKI_SSD else "0") != "0",
                        help="run the Mamba2 chunked SSD scan through the hand-written NKI kernel pair "
                             "(4.53x fwd / 3.83x fwd+bwd over the eager scan at the per-layer shape). "
                             "Falls back to eager automatically off-device or when the shape does not "
                             "conform; pass --no-nki-ssd for the bit-reproducible eager scan.")
    parser.add_argument("--init-from", type=str, default="",
                        help="warm-start model weights from a prior checkpoint (fresh optimizer/step/dataloader; dev-iteration only, never for a scored submission)")
    args = parser.parse_args()

    global USE_NKI_RELU2, USE_NKI_ROPE_NORM, USE_NKI_SOFTCAP_CE
    USE_LOCAL_CONV = args.local_conv         # read by Block.__init__, i.e. before GPT(config) below
    LRM_QUANTUM = args.lrm_quantum
    USE_TENSOR_LR_SCALARS = args.tensor_lr_scalars
    USE_MUDD = args.mudd                      # read by Block.__init__, i.e. before GPT(config) below
    USE_DCMHA = args.dcmha                    # read by CausalSelfAttention.__init__, i.e. before GPT(config) below
    USE_NKI_RELU2 = args.nki_relu2
    USE_NKI_ROPE_NORM = args.nki_rope_norm
    USE_NKI_SOFTCAP_CE = args.nki_softcap_ce
    USE_MAMBA2 = args.mamba2                  # read by Block.__init__, i.e. before GPT(config) below
    MAMBA2_PATTERN = args.mamba2_pattern      # ditto (mamba2_layer_flags reads it per block)
    # Mixer geometry: all read by Mamba2Mixer.__init__, i.e. before GPT(config) below.
    MAMBA2_EXPAND = args.mamba2_expand
    MAMBA2_D_STATE = args.mamba2_d_state
    MAMBA2_HEAD_DIM = args.mamba2_head_dim
    MAMBA2_CHUNK = args.mamba2_chunk
    MAMBA2_N_GROUPS = args.mamba2_n_groups
    MAMBA2_CONV_KERNEL = args.mamba2_conv_kernel
    USE_NKI_SSD = args.nki_ssd

    device_type = detect_device_type(args.device_type)
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)

    print0(f"device={device} dtype={COMPUTE_DTYPE} world_size={world_size}")
    print0(f"training target={args.num_steps} steps max_seconds={args.max_train_seconds} official_budget={TRAIN_TIME_BUDGET_SECONDS}s")

    tokenizer = ensure_tokenizer(build_if_missing=False)
    global BOS_TOKEN_ID
    BOS_TOKEN_ID = tokenizer.get_bos_token_id()
    config = build_config()
    print0(f"model_config={json.dumps(asdict(config), indent=2)}")
    model = GPT(config)
    model.init_weights()
    if args.init_from:
        warm_start_from(args.init_from, model)
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

    if args.compile_only:
        synthetic_x = torch.zeros((DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN), dtype=torch.int32).to(device)
        synthetic_y = torch.zeros((DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN), dtype=torch.long).to(device)
        optimizer = model.setup_optimizer()
        model.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            loss = model(synthetic_x, synthetic_y)
            (loss / grad_accum_steps).backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        sync_gradients(model)
        optimizer.step()
        synchronize(device)
        print0("compile-only synthetic training step completed")
        cleanup_runtime()
        return

    loader_seq_len = PACK_FACTOR * SEQ_LEN
    _loader_fn = make_dataloader_requeue if USE_REQUEUE_PACKER else make_dataloader
    train_loader = _loader_fn(tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, "train", device)

    def fetch():
        xb, yb, state = next(train_loader)
        if PACK_FACTOR > 1:
            xb = xb.reshape(DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN)
            yb = yb.reshape(DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN)
        return xb, yb, state

    x, y, loader_state = fetch()
    optimizer = model.setup_optimizer()
    loop_started = time.monotonic()
    budget_started = loop_started

    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0
    recent_step_times: list[float] = []
    startup_allowance_est = 0.0

    while step < args.num_steps:
        elapsed = time.monotonic() - budget_started

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
        record_step_boundary(step)
        dt = time.monotonic() - t0
        recent_step_times.append(dt)
        train_loss = float(last_loss.item())
        if step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED:
            # The runner excludes launch through this boundary (capped), so
            # restart the budget clock here to use the full charged budget.
            # If startup exceeded the cap, the excess is already charged:
            # back-date the clock by that excess, with a margin for the
            # launch -> import gap this process cannot observe.
            now = time.monotonic()
            observed_startup = now - _PROCESS_STARTED
            startup_allowance_est = min(observed_startup, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            excess = max(
                0.0,
                observed_startup + STARTUP_LAUNCH_MARGIN_SECONDS - TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,
            )
            budget_started = now - excess
        elapsed_after_step = time.monotonic() - budget_started
        # Fold the deadline decision into the existing health collective. A
        # local wall-clock check can otherwise make one rank begin another
        # optimizer step while its peers head for the checkpoint barrier.
        # The budget clock is only meaningful once it has been reset at the
        # excluded-startup boundary; before that, elapsed_after_step is measured
        # from loop start and a slow first-step compile could trip the predictive
        # stop and break before five steps complete -- leaving the run unscorable.
        # Gate the time-based stop behind reaching the excluded-step count so the
        # run is always scorable; an explicit num_steps budget still stops early.
        reached_min_steps = step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED
        stop_after_step = (
            step + 1 >= args.num_steps
            or (
                reached_min_steps
                and (
                    elapsed_after_step >= args.max_train_seconds
                    or elapsed_after_step
                    + max(recent_step_times[-20:])
                    + CHECKPOINT_RESERVE_SECONDS
                    >= args.max_train_seconds
                )
            )
        )
        control = torch.tensor(
            [
                1 if (math.isnan(train_loss) or train_loss > 100) else 0,
                1 if stop_after_step else 0,
            ],
            device=device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(control, op=dist.ReduceOp.MAX)
        if int(control[0].item()) != 0:
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
        if int(control[1].item()) != 0:
            break

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
        # The budget clock restarts at the fifth completed step to mirror the
        # runner's charged window, so post-restart elapsed time IS the charged
        # estimate (conservative: the runner also excludes pre-loop startup).
        print(f"training_seconds:  {budget_elapsed_before_save:.1f}")
        print(f"budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}")
        print(f"target_steps:      {args.num_steps}")
        print(f"max_train_seconds: {args.max_train_seconds}")
        print(f"startup_allowance_est: {startup_allowance_est:.1f}")
        print(f"charged_seconds_est:   {budget_elapsed_after_save:.1f}")
        print(f"over_budget:       {budget_elapsed_after_save > TRAIN_TIME_BUDGET_SECONDS}")
        print(f"total_seconds:     {time.monotonic() - loop_started:.1f}")
        print(f"compiled:          {args.compile}")
        print(f"nki_relu2:         {args.nki_relu2}")
        print(f"nki_rope_norm:     {args.nki_rope_norm}")
        print(f"nki_softcap_ce:    {args.nki_softcap_ce}")
        print(f"mudd:              {args.mudd}")
        print(f"dcmha:             {args.dcmha}")
        print(f"mamba2:            {args.mamba2}")
        print(f"mamba2_pattern:    {args.mamba2_pattern}")
        print(f"nki_ssd:           {args.nki_ssd}")
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
