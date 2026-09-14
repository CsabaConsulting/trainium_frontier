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
import random
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
else:
    # T0.6 (Day 2) WINNER: --optlevel=1 beats the compiler's default (2). Counter-intuitive but
    # measured, and measured twice.
    #   phase_timer, 4 ranks / LNC=2 / PACK_FACTOR=4 / real dataloader, median step_total of 6
    #   steady-state steps: no flags 837.6ms | --optlevel=2 (explicit) 836.2 | --optlevel=1 816.4
    #   | --optlevel=3 834.8 | --model-type=transformer 835.6 | transformer+optlevel=3 834.9.
    #   All six --optlevel=1 steps (812.7-818.4ms) sit below all six baseline steps (832.9-839.6),
    #   i.e. non-overlapping distributions, not a mean shifted inside the noise.
    #   Real 900s runs, paired: optlevel=1 1097/1.081696 and 1098/1.082876 (0.8155/0.8150 s/step)
    #   vs default 1077/1.083610 and 1080/1.083759 (0.8306/0.8283 s/step) = +19 steps (+1.8%) and
    #   -0.001399 bpb, with both optlevel=1 runs below both default runs.
    # Numerics are unchanged in any way that matters: the first five logged training losses match
    # the default-optlevel run to 4 decimals (9.0099 / 8.7969 / 8.3958 / 8.3060 vs 8.3059 / 8.0681
    # -- one 1e-4 drift at step 3 from different fusion), and causality_passed True /
    # causality_max_abs 0.0 on both runs.
    # Why it has to live HERE rather than in the launch command: the Phase 1 submission command is
    # fixed at `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py ...`,
    # so a compiler flag exported outside the process would simply not be applied at submission
    # time. setdefault, so an external NEURON_CC_FLAGS still wins for A/B work.
    # --optlevel=2 explicit reproducing the no-flags number is the control that proves the env var
    # is actually reaching neuronx-cc, i.e. that 2 really is the default being beaten.
    os.environ.setdefault("NEURON_CC_FLAGS", "--optlevel=1")
if USE_STOCHASTIC_ROUNDING:
    os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")

import pyarrow.parquet as pq
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
    _row_groups_for_split,
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
#
# `requeue=False` makes this function reproduce prepare.make_dataloader()
# EXACTLY -- the only textual difference between the two is the
# `if requeue:` guard around the leftover append, so with it False the crop
# branch is prepare's line-for-line (pop the shortest doc, take doc[:remaining],
# drop the rest). That mode exists so USE_EPOCH_SHUFFLE below can inject a
# different _document_batches_fn without also changing the packer, i.e. without
# confounding a data-ORDER experiment with a data-PACKING one.
def make_dataloader_requeue(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    device: torch.device | str,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    shard_ids=None,
    requeue: bool = True,
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
                    if requeue:
                        leftover = doc[remaining:]
                        if leftover:
                            doc_buffer.append(leftover)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1].to(dtype=torch.int32))
        cpu_targets.copy_(row_buffer[:, 1:])
        inputs.copy_(cpu_inputs, non_blocking=pin_memory)
        targets.copy_(cpu_targets, non_blocking=pin_memory)
        yield inputs, targets, {"epoch": epoch}


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
            rows = rg.column("text").to_pylist()
            for i in range(0, len(rows), tokenizer_batch_size):
                yield rows[i:i + tokenizer_batch_size], epoch


# -----------------------------------------------------------------------------
# Baseline hyperparameters. Model-related config lives here, not in prepare.py.

SEQ_LEN = 2048  # Real interior optimum, robust to torch.compile (see trn2-experiment-results.md). 4096's quality drop-off tested and ruled out as a RoPE-extrapolation artifact (scaled ROPE_BASE made no difference) -- more likely a genuine capacity/signal-density limit at this model scale.
DEPTH = 6  # Tier-0 resweep confirmed DEPTH=8 loses clearly at full 1800s (1217/1.052406 vs 1995/1.043489) -- DEPTH=6 stays locked. See trn2-experiment-results.md
ASPECT_RATIO = 96  # Tier-0 resweep confirmed width=640 still correct at full 1800s: 512 (2433/1.05416) and 768 (1623/1.04415, close but still behind) both lose. See trn2-experiment-results.md
HEAD_DIM = 128
N_EMBD = 0  # T0.2 depth/width decoupling: explicit width override. 0 (default) = keep the historical
# derivation n_embd = round_up(DEPTH * ASPECT_RATIO, HEAD_DIM), which is bit-identical to every run
# before this flag existed. Set > 0 (must be a multiple of HEAD_DIM) to force a specific
# (DEPTH, n_embd) pair -- DEPTH had never been tested independently of the width it happens to
# imply through ASPECT_RATIO, which is exactly what --depth/--n-embd exist to separate.
MLP_RATIO = 4  # T0.3: MLP hidden width = MLP_RATIO * n_embd. 4 is the historical (and starter-kit)
# value, so the default is bit-identical to every previous run. Carried in GPTConfig so a checkpoint
# is self-describing and load_for_eval rebuilds the right shape without needing the flag.
ATTN_SCALE = 0.0  # T0.3: explicit softmax scale for F.scaled_dot_product_attention. 0.0 (default) =
# pass scale=None, i.e. SDPA's own 1/sqrt(head_dim) == 0.0884 at HEAD_DIM=128 -- bit-identical to
# every previous run. QK-norm (rope_norm on both q and k) already fixes |q|,|k| to sqrt(D) per head,
# which decouples the useful logit temperature from the 1/sqrt(D) convention, so this is a real free
# knob rather than a redundant reparameterisation of the init scale.

DEVICE_BATCH_SIZE = 1
TOTAL_BATCH_SIZE = 131072  # Tier-0 resweep confirmed still correct at full 1800s: 65536 (3544/1.048832) and 262144 (1038/1.051031) both lose. See trn2-experiment-results.md
NUM_STEPS = 1_000_000  # Deliberately unreachable: the wall-clock budget (MAX_TRAIN_SECONDS, self-enforced with a
# checkpoint reserve) is the ONLY control that may stop a scored run. The starter kit's 1000 silently capped the
# 2026-09-10 submission at step 1000 (~815s, lrm ~0.74, no cooldown) -> 1.1495 on the board vs 1.034 locally,
# because the bare launch command passes no --num-steps while every local run passed --num-steps 5000 via
# research/run_experiment.py. See research/submission-gap-analysis-2026-09-10.md. Pass --num-steps N for smoke tests.
# The official runner charges wall time only after the first five completed
# optimizer steps (the startup/compilation allowance, capped). The baseline
# restarts its budget clock at that boundary so it can use the full charged
# budget even when compilation makes the first steps slow.

# T2.1 batch-size ramp. TOTAL_BATCH_SIZE above is a single fixed value for the whole run, and
# the Tier-0 resweep showed both neighbours lose -- but that sweep could only ask "which ONE
# batch size is best on average", and the answer to that is a compromise. Early training is
# gradient-noise-tolerant (any descent direction is roughly right, so steps are worth more than
# precision) while the cooldown is not (a large-batch, low-noise gradient is what a small lr can
# actually exploit). A ramp buys both: more cheap steps early, cleaner steps late.
#
# The MICRO shape never changes (DEVICE_BATCH_SIZE * PACK_FACTOR * SEQ_LEN), only how many
# micro-batches are accumulated, so every compiled graph is shape-identical in all three stages
# and no stage boundary can trigger a recompile. Boundaries are fractions of the WALL-CLOCK
# budget, matching the LR schedule, which is time-driven too -- BATCH_RAMP_BOUNDS[1] is
# deliberately WARMDOWN_RATIO so stage C is exactly the cooldown.
BATCH_RAMP = False
BATCH_RAMP_GRAD_ACCUM = (2, 4, 8)   # micro-batches per stage; 4 == today's TOTAL_BATCH_SIZE
BATCH_RAMP_BOUNDS = (0.35, 0.75)    # elapsed/max_train_seconds boundaries A->B and B->C
BATCH_RAMP_LR_EXP = 0.5             # lr scales as (B/B_ref)**this. 0.5 = sqrt, the standard
# gradient-noise scaling (SGD-noise-matched); 1.0 would be linear scaling and 0.0 no scaling.
# The scale is folded straight into group["lr"], which ALREADY goes through
# MuonAdamW._device_scalar every step, so this adds no new device constant and cannot recompile
# anything -- see the USE_TENSOR_LR_SCALARS note. The 1/grad_accum loss scaling is the one place
# a changing Python float WOULD reach a device op, so under BATCH_RAMP it is carried in a
# preallocated device tensor updated with copy_ instead (see the training loop).

# T2.2 EMA of the final weights. The last weights a run happens to land on are one sample from a
# distribution the optimizer is still bouncing around inside; averaging over the tail of the
# trajectory cancels part of that bounce for free (no extra gradient, no extra step). It is only
# credible over the COOLDOWN, which is why WEIGHT_EMA_START_FRAC defaults to WARMDOWN_RATIO: the
# shadow is initialized from the live weights at that point, so the averaged region is the low-lr
# tail and nothing from the high-lr phase is dragged in.
#
# Cost: one torch._foreach_lerp_ over the 62 parameters per step per decay. That is ~3 x 223MB of
# HBM traffic = ~0.23ms at 2.9TB/s, i.e. <0.03% of an 816ms step, plus 223MB of extra HBM per
# shadow (of 96GB). EAGER ONLY -- it sits next to optimizer.step(), which is already outside every
# compiled region, because torch._foreach_* cannot be traced by backend="neuron".
# The lerp weight is a fixed Python constant (1 - decay), never a per-step-changing one, so it
# cannot cause the per-step recompile that USE_TENSOR_LR_SCALARS exists to avoid.
#
# WEIGHT_EMA_DECAYS is a TUPLE because the shadows are so cheap that carrying several at once is
# strictly better than running several jobs: every decay then sees bit-identical weights at every
# step, so comparing them has no run-to-run noise in it at all. Whichever is listed FIRST is the
# primary -- it is the one written into the checkpoint's "ema_state" (and therefore the one
# load_for_eval would hand the scorer); the rest are evaluated in-process and reported only,
# which keeps the checkpoint at 2x the weights instead of (1+N)x and its save inside
# CHECKPOINT_RESERVE_SECONDS.
#
# The averaging window is ~1/(1-decay) steps and the EMA's weights are, to first order, the
# trajectory's weights from ~that many steps ago. That lag is the entire risk: it only pays for
# itself once the trajectory has stopped descending and is merely bouncing. Measured at a 400s
# budget (474 steps, seeded at 0.75, decay 0.995 -> a 200-step window over a 103-step tail) the
# EMA was +0.0168 bpb WORSE than the raw weights -- consistent with it being "the weights from
# ~100 steps ago" and nothing else. Hence the short windows are the interesting ones here.
USE_WEIGHT_EMA = False
WEIGHT_EMA_DECAYS = (0.995,)    # per-step decays; 0.98 ~ 50 steps, 0.995 ~ 200, 0.999 ~ 1000
WEIGHT_EMA_START_FRAC = 0.75    # elapsed/max_train_seconds at which the shadows are seeded
# When on, the checkpoint carries BOTH weight sets ("model_state" = raw, "ema_state" = averaged)
# and the run evaluates both, so a single run is its own A/B with zero run-to-run noise.
# load_for_eval's 2-argument contract is untouched: it prefers "ema_state" only when the key is
# actually present, which can only happen if this toggle was on during training.
EVAL_PREFER_EMA = os.environ.get("NEURON_COMPETITION_R1_EVAL_PREFER_EMA", "1") != "0"

MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
CHECKPOINT_RESERVE_SECONDS = 5.0
# Covers the launch -> module-import gap this process cannot observe when
# checking its startup time against the allowance cap.
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0

LOGIT_SOFTCAP = 15.0

EMBEDDING_LR = 0.30
UNEMBEDDING_LR = 0.006
MATRIX_LR = 0.020  # T0.4 resweep (Day 3): part of the combined stack below. Single-axis at 900s: 0.020 -0.000835/-0.000106 (paired), 0.015 +0.002728, 0.030 +0.000552, 0.035 +0.000990 -- an asymmetric curve whose only better-than-baseline point is 0.020, sub-noise on its own, adopted as part of the confirmed stack.
SCALAR_LR = 0.20
WEIGHT_DECAY = 0.20  # T0.4 resweep (Day 3): part of the combined stack below. Single-axis at 900s: 0.20 -0.000557/-0.000533 (paired, spread 0.000024 -- the tightest pair in the log), 0.35 -0.000458, 0.05 -0.000019. Flat-but-reproducible axis; 0.20 chosen as the middle of the better region. NOTE: combining 0.20 with WD_DECAY_TO_ZERO=False came in +0.001828, i.e. WORSE than either alone -- they are effectively the same knob (total integrated decay), so WD_DECAY_TO_ZERO stays True.
VE_WD = 0.0  # weight decay for the value_embeds AdamW group (separate from the shared embedding_params group)
EMBED_ADAMW_BETA2 = 0.95  # Tier-0 resweep confirms still correct at the new ~2000-step regime: 0.95 (995/1.094327) beats 0.995 (977/1.097342) again. See trn2-experiment-results.md
WARMUP_STEPS = 0
GRAD_CLIP = 1.0  # Scavenger hunt real win, locked: never tested on Trn2 before ("deferred, not GPU-retested"). HLAT (real 7B/70B Trainium pretrain) used clip=1.0. Real repeat-confirmed win at 900s (999/1.0912, 1002/1.090864 vs 0.0-baseline cluster 1002-1004/1.09367-1.09399) and full 1800s (2000/1.042278 vs prior best 1987/1.043651). See trn2-experiment-results.md
WARMDOWN_RATIO = 0.75   # fraction of max_train_seconds spent on LR cooldown (0 = old flat-after-warmup)
FINAL_LR_FRAC = 0.05    # LR floor as a fraction of peak LR at the end of cooldown (1.0 = no floor effect)
USE_TENSOR_LR_SCALARS = True   # LOCKED ON. Carry the per-step lr in a device tensor instead of a Python float so the optimizer graph is compiled once, not once per lr value (see MuonAdamW._device_scalar). Biggest single win measured on Trn2 so far: dt stays 0.90s through the whole cooldown instead of 2.2-4.3s, giving 1995 steps/1.043489 at 1800s vs 847/1.108697, and 995/1.094327 at 900s vs 427/1.180633. Model math unchanged; causality exact.
LRM_QUANTUM = 0         # >0: snap the LR multiplier onto a 1/LRM_QUANTUM grid, so the cooldown uses at most this many distinct lr values (see lr_multiplier); 0 = continuous, one new value per step
USE_FOREACH_OPTIM = True  # batch the optimizer's per-parameter work into torch._foreach_* calls instead of a Python loop of single-tensor ops (see MuonAdamW._adamw_step_foreach and the batched read-in in _muon_step). Pure dispatch-overhead change: the update math is BIT-IDENTICAL, not an approximation. Motivation: on this native-eager backend every aten op is its own NEFF, so the optimizer step ran 647 launches over 171 separate NEFFs with 73.9ms of the 123.8ms window being pure inter-launch gap -- 10.0% of step time, every engine idle for three quarters of it. MEASURED, all four checks: (1) EQUIVALENCE -- research/scratch/foreach_optim_equiv.py builds the real GPT at the locked config on CPU, runs 5 steps of loop-vs-foreach on identical deterministic grads and gets 0.000e+00 max abs diff (torch.equal True) on all 62 params AND all 42 optimizer-state tensors, for both USE_TENSOR_LR_SCALARS=True and False. (2) DISPATCH COUNT (CPU, TorchDispatchMode) 1023 -> 708 aten calls per step; the AdamW groups collapse 271 -> 54 (5.02x, they are 17 tiny tensors of which 12 are shape-(1,) scalars each costing ~8 launches for no compute) while Muon only gets 816 -> 652 (1.25x, its bulk is real batched Newton-Schulz matmul). (3) REAL PROFILE of the optimizer-step window (4 ranks, LNC=2, neuron-explorer): launches 647 -> 402 (-37.9%), inter-launch gap 73.9 -> 50.5ms (-31.6%), gap share of the window 59.8% -> 54.4%. (4) REAL TIMING at the submission shape (research/scratch/phase_timer.py, 4 ranks, LNC=2, compile on): opt_step_ms 90.05 -> 66.99 (-23.1ms, -25.6%, non-overlapping ranges over 4 measurements a side), step_total_ms 891.4 -> 876.3 (-1.7%). Smoke test single-device 15 steps: causality_passed True, causality_max_abs 0.0. END-TO-END, paired same-day 900s A/B at the locked shape (this is the number that decided the default, because the week's older 900s rows span 976-1004 steps across different knob settings and a cross-day absolute comparison is not trustworthy): ON 1002/1004/1003 steps (mean 1003.0) at 1.093909/1.093671/1.093991 bpb (mean 1.093857) vs OFF 974/968 (mean 971.0) at 1.096378/1.096809 (mean 1.096594) -- +32.0 steps (+3.3%) and -0.0027 bpb, with the two groups' step counts 28 apart and non-overlapping. Quality moves the right way for the mechanical reason (same update math, more steps in the same wall clock). CONFIRMED at the real full 1800s budget, also paired same-day: ON 1987 steps / 1.043651 bpb vs OFF 1934 / 1.045602 -- +53 steps (+2.7%) and -0.0020 bpb. See rows foreach-optim-900s{,-v2,-v3}, foreach-off-900s-v{1,2}, foreach-optim-fullbudget and foreach-off-fullbudget in research/experiment-log.jsonl and the "optimizer dispatch overhead" section of research/trn2-experiment-results.md. NOTE torch._foreach_* is EAGER-ONLY on this stack: torch.compile(backend="neuron") on the optimizer step dies with 'failed to legalize operation torch.operator' on _foreach_copy_, which is also why the pre-existing _foreach_copy_ write-back at the bottom of _muon_step makes compiling the optimizer step impossible either way.
WD_DECAY_TO_ZERO = True  # decay Muon's weight_decay linearly to 0 over the run instead of holding it constant
MUON_MOMENTUM_WARMUP_STEPS = 300  # Tier-0 resweep, confirmed still correct despite the step-count regime shift: 700 (979/1.097288) and 100 (976/1.098676) both worse than 300's baseline (995/1.094327) at 900s -- real signal both directions lose, not stale. See trn2-experiment-results.md
# T0.4: symmetric counterpart to the warmup above -- ramp Muon's momentum back DOWN over the
# tail of the run (modded-nanogpt does this; the intuition is that a lower momentum near the
# end lets the update track the sharply-decaying LR instead of averaging over a stale window).
# Expressed as a FRACTION OF THE WALL-CLOCK BUDGET, not a step count, because every run here is
# time-budgeted and the achieved step count is not known in advance (0.024 of 1800s at ~0.86s/step
# is ~50 steps). 0.0 = off, i.e. bit-identical to the pre-T0.4 schedule.
# T0.4 resweep (Day 3): ON at 0.045 of the budget (~50 steps at 1800s), part of the confirmed
# combined stack. Single-axis at 900s: -0.000493 with step count unchanged (1097 vs ~1097), which is
# also the evidence that the schedule does NOT recompile -- the momentum is written into the param
# groups as a DEVICE TENSOR updated in place (see MuonAdamW/set_lr), never as a fresh Python float,
# because a new Python scalar reaching a device op recompiles the whole graph every step (the
# USE_TENSOR_LR_SCALARS lesson).
MUON_MOMENTUM_COOLDOWN_FRAC = 0.045
MUON_MOMENTUM_FINAL = 0.85  # momentum at the very end of the cooldown (only read when FRAC > 0)
MUON_NS_STEPS = 5  # Newton-Schulz iterations in Muon's orthogonalization (3 matmuls each)
# T1.4: run the AdamW parameter groups (wte, lm_head, value-embedding tables, scalars and the T0.1
# local_conv group) only every Nth optimizer step, letting their gradients ACCUMULATE across the
# skipped steps rather than being zeroed. 1 = every step (the historical behaviour, bit-identical).
# Rationale: the AdamW groups are 17 small tensors, of which 12 are shape-(1,) scalars -- they cost
# a disproportionate share of optimizer dispatch and of the per-step all_reduce COUNT while carrying
# almost no gradient information per step. The Muon groups (the real matmul weights) still update
# every step, untouched.
#
# The deferral is EXACT on the collective side: skipping dist.all_reduce(AVG) on a non-update step
# and reducing the 2-step local sum on the update step gives the same number as reducing each step
# and summing, because AVG is linear. It is NOT exact on the training-dynamics side, and deliberately
# so: (a) Adam's update is scale-invariant in the gradient, so summing two steps' grads and taking
# ONE step moves those tensors about half as far in total as two steps would -- an effective LR
# halving for those groups, which is why --adam-every-n-lr-comp exists to multiply their lr by N and
# separate "less often" from "less far"; and (b) GRAD_CLIP is applied over model.parameters() every
# step, so an accumulating AdamW gradient can be scaled by the clip coefficient more than once
# before it is consumed. Both are real changes, judged on bpb like any other, not equivalences.
ADAM_EVERY_N = 1
ADAM_EVERY_N_LR_COMP = False
INIT_SCALE = 0.50  # multiplier on the attn/mlp uniform-init bound (1.0 = original baseline scale). T0.4 resweep (Day 3), part of the combined stack: 0.50 -0.000906, 0.85 +0.000061 at 900s -- points weakly but consistently at a smaller init.
# T0.4: attn.c_proj is zero-initialised (below, in init_weights) so a fresh block is an exact
# identity. Zero-init output projections are known to delay early learning, which matters more in a
# ~2000-step run than in a normal pretrain, so this makes the attention output projection's init
# bound a swept multiple of the shared bound `s`: 0.0 keeps the historical exact-zero behaviour,
# 0.1 gives uniform(-0.1*s, 0.1*s), etc. mlp.c_proj is deliberately NOT covered -- the task scopes
# this to attention only, so the two output projections can be separated if it wins.
# T0.4 resweep (Day 3): 0.10 adopted as part of the confirmed combined stack. Single-axis at 900s:
# 0.10 -0.000472, 0.25 +0.000805 -- a small nonzero bound helps, a large one hurts. mlp.c_proj stays
# exactly zero.
ATTN_CPROJ_INIT_SCALE = 0.10
# ---------------------------------------------------------------------------------------------
# T0.4 COMBINED STACK (Day 3), confirmed at the real full 1800s budget. The five constants above
# (MATRIX_LR 0.025->0.020, WEIGHT_DECAY 0.10->0.20, INIT_SCALE 0.68->0.50, ATTN_CPROJ_INIT_SCALE
# 0.0->0.10, MUON_MOMENTUM_COOLDOWN_FRAC 0.0->0.045) were adopted TOGETHER, not individually: none
# of them cleared the 0.002 single-axis lock threshold at 900s, but all five pointed the same way,
# and the stack was then measured rather than assumed (which mattered -- the one other "stack two
# sub-noise wins" test in this sweep, WEIGHT_DECAY 0.20 + WD_DECAY_TO_ZERO off, came out WORSE).
#   900s, paired:  stack 1097/1.081226 and 1096/1.081464 (mean 1.081345)
#                  defaults 1097/1.081696, 1098/1.082876, 1102/1.082271 (mean 1.082281)
#                  -> -0.000936, and min(defaults) > max(stack), i.e. non-overlapping
#   1800s, paired: stack 2205/1.034005 and 2199/1.034101 (mean 1.034053)
#                  defaults 2205/1.035073 and 2204/1.035816 (mean 1.035445)
#                  -> -0.001392, again non-overlapping, at identical step counts (so this is pure
#                     optimisation quality, no throughput confound)
# The rest of the T0.4 axes were re-tested and CONFIRMED unchanged: FINAL_LR_FRAC 0.05,
# WARMDOWN_RATIO 0.75, EMBEDDING_LR 0.30, UNEMBEDDING_LR 0.006, LOGIT_SOFTCAP 15.0, MUON_NS_STEPS 5,
# WD_DECAY_TO_ZERO True. See the T0.4 section of research/trn2-experiment-results.md for all 27 rows.
# ---------------------------------------------------------------------------------------------
# Dataloader packing-bin multiplier: fetch DEVICE_BATCH_SIZE rows of PACK_FACTOR*SEQ_LEN tokens and
# reshape to DEVICE_BATCH_SIZE*PACK_FACTOR rows of SEQ_LEN, so this is the micro-batch ROW COUNT and
# grad_accum_steps falls as it rises (TOTAL_BATCH_SIZE is held fixed either way).
# T1.5 (Day 2): 4 is the winner, paired same-day same-commit A/B, 900s:
#   PF=4  1077/1.083610 and 1080/1.083759  (mean 1.083685, 0.829-0.831 s/step)
#   PF=2  1049/1.085529 and 1054/1.085307  (mean 1.085418, 0.849-0.853 s/step)
# = -0.001734 bpb and +27 steps (+2.6%). Both PF=4 runs beat both PF=2 runs and the within-pair
# spreads (0.000149 / 0.000222) are ~8x smaller than the between-group gap, so this clears the
# "<0.003 needs a paired repeat" bar. The step-count half is corroborated mechanically by a
# synthetic-data fwd+bwd microbenchmark on the real 4-rank/LNC=2 compiled shape: us/token by
# micro-batch rows = 38.325 (1) / 22.322 (2) / 21.673 (4) / 23.800 (8), i.e. 4 rows is the
# per-token optimum and 8 is already past it -- which is why PACK_FACTOR=8 is NOT worth an A/B.
# HISTORY: the old comment here claimed "4 lost decisively (776 steps/1.113876)". That run
# (`packfactor4-900s`, commit 9dc9873) is a STALE-COMMIT ARTIFACT, not a property of PACK_FACTOR:
# re-run unchanged on current code it gives 1077 steps, not 776. Its logged launch_command carried
# no --pack-factor flag (the constant had to be edited by hand back then), so it was not
# reproducible from its own log -- hence the flag now exists. See trn2-experiment-results.md.
PACK_FACTOR = 4
USE_REQUEUE_PACKER = False  # Bucket B: implemented + verified correct (offline harness, research/scratch/requeue_packer_equiv.py, zero corruption over a 20k-doc synthetic corpus), but the real hardware A/B at the final locked shape was a small real negative (332 steps/1.235799 vs baseline 330/1.234514) -- not kept. See trn2-experiment-results.md
USE_EPOCH_SHUFFLE = False  # DISCARDED, kept as the recorded negative: at 1800s with the frozen
# qk-shift (+9.2% steps) it measured 2341/1.023799 against 2354/1.021278 unshuffled and 2149/1.018967
# step-matched -- i.e. reordering does NOT recover the throughput and makes the epoch crossing WORSE.
# Mechanism: prepare's identical-order replay hands epoch 2 the corpus's OPENING row groups, the ones
# the model last saw ~2100 steps ago and has had the most time to forget; a uniform shuffle mixes in
# groups seen only tens of steps earlier, which is a much sharper overfit. prepare's "naive" order is
# already the recency-MAXIMIZING one, so no reordering variant is worth trying.
# Reshuffle the row-group order from epoch 2 onward (epoch 1 untouched, so
# this is a literal no-op for any run that never crosses the epoch boundary -- including the locked
# submission config at ~2140 steps). Exists because the 1800s budget is DATA-bound, not
# compute-bound: one epoch is ~2195 steps / ~288M tokens at the locked shape, and a mathematically
# IDENTICAL arm that ran 9.2% faster (2354 steps) came out +0.00231 bpb WORSE than the same arm
# step-matched at 2149, instead of the -0.0062 the compute-scaling slope predicts. prepare's
# generator replays the corpus in identical order every epoch, so those extra steps re-trained on a
# contiguous ~7% slice of the corpus during the lowest-LR cooldown -- a narrow fine-tune, not extra
# data. See the 2026-09-13 "data wall" section of trn2-experiment-results.md.
# "Data sampling strategy" is explicitly permitted in train.py by the README; prepare.py is untouched
# (this reuses its private _row_groups_for_split, exactly as USE_REQUEUE_PACKER reuses
# _document_batches).
EPOCH_SHUFFLE_SEED = 1234  # seed base for the per-epoch row-group shuffle; epoch e uses SEED + e.
USE_VE = True  # value embeddings (ResFormer-style): alternating layers, input-gated mix into attention's V stream
VE_GATE_CHANNELS = 32


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Alternating layers get a value embedding; the last layer always does."""
    return layer_idx % 2 == (n_layer - 1) % 2


# -----------------------------------------------------------------------------
# T3.1 -- hashed bigram/trigram VALUE embeddings (Recursive's recipe, research plan
# §5 T3.1 + §11.1). The two SOTA 5-minute-budget solutions both credit hashed n-gram
# tables as their single largest gap-closer, and Recursive's variant rides exactly on
# the code path this file already has: on every has_ve() layer, add a table indexed by
# a HASH of (prev, cur) -- and, on the first and last VE layer, of (prev2, prev, cur) --
# and mix it into the same attention V stream the unigram value embedding already feeds,
# through its own zero-init per-head gate reading a DISJOINT slice of the residual
# (unigram 0:32, bigram 32:64, trigram 64:96).
#
# Strictly causal by construction: position t's index is a function of tokens t, t-1,
# t-2 only -- never a later position -- so causality_check must stay at exactly 0.0.
# Exact no-op at step 0: the tables are ZERO-initialized (unlike the unigram VE, whose
# table is uniform(-s, s)), so `v + gate * 0 == v` bit-exactly even though the zero-init
# gate evaluates to the neutral 2*sigmoid(0) = 1.0. That is what makes USE_NGRAM_VE=True
# bit-identical to False at init and the A/B honest.
# ON since 2026-09-12: with NGRAM_VE_LAYERS="-1" this is the single largest measured win in the
# repo. Paired 1800s A/B, same box/session/commit, --eval-seq-len 1024: control 1.034199 bpb /
# 2191 steps vs 1.018985 / 2139 with bigram+trigram tables (524288 rows x 128) on the LAST layer
# only, i.e. -0.0152 bpb. It was "exactly break-even" for two days only because the tables were
# on every VE layer, which is a ~+50ms/step whole-graph scheduling penalty that has nothing to do
# with the tables' size -- see NGRAM_VE_LAYERS below.
USE_NGRAM_VE = True
NGRAM_VE_LAYERS = "-1"    # WHICH layers get a bigram table. "" = every has_ve() layer, which is
# Recursive's own allocation and what the original T3.1 grid measured. A comma-separated list of
# layer indices (negatives count from the end, so "-1" is the last layer at any DEPTH) restricts
# it. This exists because the per-step cost of these tables turned out to depend almost entirely
# on WHICH layer holds them and not at all on how big they are. Measured 2026-09-12 with
# research/scratch/prof_driver.py at the locked shape (1 rank, LNC=2, compile on, foreach
# optimizer, 262144 rows, dim 128), best-of-3 micro fwd+bwd, baseline 172.9 ms:
#   layers          micro dt    delta      step dt
#   (none)           172.9 ms      --       2.862 s
#   {5} only         164.1 ms    -8.8       2.768 s   <- FREE (see the -8.8 caveat below)
#   {1} only         178.2 ms    +5.3       2.991 s
#   {3} only         220.6 ms   +47.7       3.662 s
#   {1,5}            223.1 ms   +50.2       3.711 s
#   {3,5}            222.7 ms   +49.8       3.711 s
#   {1,3}            222.5 ms   +49.6       3.711 s
#   {1,3,5} (all)    225.7 ms   +52.8       3.768 s
# and the cost is INVARIANT to table geometry -- 8192 rows costs the same +51.9 as 262144 rows,
# dim 64 the same as dim 128 -- so this is not a bandwidth or a backward-scatter cost at all.
# ROOT CAUSE, from a device profile of the forward NEFF (neuron-explorer event data, 2026-09-12):
# under LNC=2 a logical core spans two PHYSICAL cores, and one dynamic-address gather makes the
# compiler re-shard the residual/V stream across those two pcores for every layer DOWNSTREAM of
# the injection. Each downstream layer boundary then pays a cross-pcore SBUF->SBUF reduce-add
# whose descriptors are 1 partition wide fanned into 128, i.e. ~30 BYTES PER DMA PACKET. The
# layer-3 arm issues 42,031,968 cross-pcore packets (the layer-5 and no-table arms issue exactly
# ZERO) for 38 MB of payload: 108x the packets of the baseline for 1.7x the bytes, ~94 ms of pure
# descriptor issue, and the Tensor engine sits idle 23.7 ms of the +43.8 ms with the data DMA
# queue busy. It is descriptor-rate bound, which is exactly why table size and width do not
# matter. The packets arrive in 3 bursts for an injection at layer 3 of DEPTH 6 -- one per layer
# at-or-after the injection point -- so the cost counts DOWNSTREAM LAYER BOUNDARIES, not gathers.
# That is why the LAST layer is free: it has no downstream boundaries to re-shard. Same reason two
# tables on ONE layer are free while two on DIFFERENT layers cost ~50 ms.
# CAVEAT on the -8.8 ms above: the profile shows the layer-5 arm has zero cross-pcore packets and
# pays only ~0.40 ms for the injected ops, so it is not genuinely faster than no table at all --
# its recompile happened to pick a cheaper matmul schedule (243,140 vs 263,258 MATMUL instrs).
# Treat last-layer injection as step-NEUTRAL, not as a speedup; the real 1800s runs bear that out
# (2202 steps vs the control's 2191 for one table, 2139 for two).
# NOT fixable by moving the op: merging all gathers into one, hoisting them to the top of the
# forward, injecting after attention instead of into V, and changing the hash are all measured
# null results, because the trigger is the DYNAMIC ADDRESS anywhere in the compiled region, not
# the op's position. Making layers 1/3 cheap would require taking the gather out of the compiled
# graph entirely (pass the gathered rows in as an ordinary input), re-expressing it as a static
# one-hot matmul (only viable at small row counts), or dropping to LNC=1 (no peer SBUF to reduce
# into). See the 2026-09-12 sections of research/trn2-experiment-results.md for the full ablation
# (the gate, the slice offset, the up-projection and the injection into V were each ruled out
# individually) and for the profile tables.
NGRAM_VE_TABLE_MULT = 64  # rows per table = this x padded_vocab_size (64 x 8192 = 524288).
# Chosen by the 2026-09-12 1800s sweep with NGRAM_VE_LAYERS="-1", bigram+trigram, dim 128
# (bpb / steps / params): mult 32 -> 1.020516 / 2173 / 123.0M, mult 64 -> 1.018985 / 2139 /
# 190.1M, mult 128 -> 1.019171 / 2067 / 324.3M. Capacity saturates at 64: doubling again is
# 0.0002 WORSE for 72 fewer steps and 134M more parameters. dim 256 at mult 64 measured
# 1.018483 / 2069 / 324.5M, which is 0.0005 better -- inside the noise band and at 1.7x the
# parameters, so dim stays 128 (the "fewer total parameters" tie-break is real).
NGRAM_VE_TRIGRAM = True   # also add a trigram table on the FIRST and LAST VE layer.
# With NGRAM_VE_LAYERS="-1" the first and last VE layer are the same layer, so this adds exactly
# one extra table, on that layer, and two tables on ONE layer is still free (measured). Worth
# -0.0017 at 1800s (1.020516 -> 1.018985 is mult, but bigram-only at mult 64 is 1.022205 vs
# bigram+trigram at mult 32 = 1.020516 at equal parameter count).
NGRAM_VE_DIM = 128        # table WIDTH. 0 => n_embd, i.e. the table is gathered straight into v.
# >0 makes each table [rows, NGRAM_VE_DIM] followed by its own Linear(NGRAM_VE_DIM, n_embd)
# up-projection, which is a THROUGHPUT knob, not a quality one, and it exists because the
# measured cost of the full-width version is the whole problem with T3.1 on this stack.
# MEASURED with research/scratch/phase_timer.py at the locked shape (4 ranks, LNC=2, compile
# on, foreach optimizer), micro-step mean fwd+bwd, baseline 182.3 ms:
#   5 tables x 65536 rows, trainable      373.9 ms  (+191.6)  -> step 1610 ms vs base 808
#   5 tables x 65536 rows, tables FROZEN  273.9 ms  (+ 91.6)  -> the FORWARD gather alone is
#                                                   half the cost; it is not just the dense
#                                                   embedding backward
#   3 tables x 65536 rows, trainable      313.9 ms  (+131.6)  -> ~30 ms per table
#   5 tables x  8192 rows, trainable      286.3 ms  (+104.0)  -> ~1.53e-3 ms per row (5 tables),
#                                                   i.e. cost scales with ROWS even in the
#                                                   forward, which a plain gather should not do
# Both terms are proportional to rows x width, so narrowing the table is the only lever that
# touches all of them at once: NGRAM_VE_DIM=128 is 5x less table to read, scatter into,
# all-reduce and update, at the price of one tiny [tokens,128]x[128,640] matmul per table.
# Step-0 no-op is preserved: the TABLE is zero-init, so the product is zero whatever the
# up-projection holds (and the projection must NOT be zero-init, or the table would receive
# no gradient at all and the whole thing would be dead weight -- see init_weights).
NGRAM_VE_LR = 0.0         # 0 => EMBEDDING_LR (Recursive uses the embedding LR for the tables)
NGRAM_VE_BETA2 = 0.999    # RMSProp second-moment decay
NGRAM_VE_FLAT_LR = True   # no warmdown for these tables ("sparse tables benefit from full-rate
                          # training"). Warmup still applies -- only the cooldown is skipped.
NGRAM_VE_OPT = "rmsprop"  # "rmsprop" (no first moment, Recursive's choice) or "adamw"
NGRAM_VE_BF16 = True      # store the tables (and their optimizer state) in bf16
NGRAM_VE_CLIP_EXCLUDE = True  # keep the tables OUT of the global clip_grad_norm_ and clip them as
# their own group instead. GRAD_CLIP=1.0 is a GLOBAL norm, so folding ~200M extra parameters'
# gradients into it would shrink clip_coef and thereby change the effective LR of every OTHER
# tensor in the model -- a confound that has nothing to do with whether n-gram tables help. With
# this on, the baseline arm's clip is bit-identical (same tensors, same order) and the tables are
# still norm-bounded, just separately.
NGRAM_VE_INDEX_GATHER = True  # gather rows as `weight[idx]` instead of `nn.Embedding.__call__`.
# Numerically identical; only the lowering differs. `aten::embedding_dense_backward` on this stack
# costs time proportional to the table's ROW COUNT (19.4ms at 65536 rows, 72.5ms at 262144,
# regardless of width) while advanced indexing's backward stays at 1.8-5.2ms. Kept as a toggle
# purely so the two lowerings remain A/B-able from research/scratch/phase_timer.py; there is no
# known reason to turn it off. See _ngram_lookup for the full measurement table.

# -----------------------------------------------------------------------------
# T3.2 -- ENGRAM-LITE: gated, context-aware conditional memory (research plan §5 T3.2, i.e.
# DeepSeek's Engram (arXiv 2601.07372v2) scaled down). This is the architectural SUPERSET of
# T3.1: the same hashed n-gram statistics, but read through a gate that depends on the model's
# own HIDDEN STATE rather than on a fixed 32-channel residual slice, and injected into the
# RESIDUAL STREAM rather than into attention's V. The plan gates it on "only if T3.1 shows a
# clear win", which T3.1 did (-0.0152 at 1800s, the largest measured win in the repo).
#
# The mechanism, exactly as the plan's T3.2 entry specifies it:
#     e_t   = concat_{h,n} table[h,n][ hash_{h,n}(tokens t-n+1 .. t) ]     (K heads per order)
#     k_t   = W_K e_t ,   v_t = W_V e_t
#     a_t   = sigmoid( <RMSNorm(q_t), RMSNorm(k_t)> / sqrt(d) ) ,  q_t = h_t (or W_Q h_t)
#     vt_t  = a_t * v_t
#     Y     = SiLU(Conv1D_dw(RMSNorm(vt))) + vt          (the conv is ZERO-init)
#     H    <- H + Y                                       (at ENGRAM_SITE)
#
# STRICTLY CAUSAL twice over, and structurally rather than by mask: the hash at position t reads
# tokens t-n+1..t only (identical argument to T3.1's), and the depthwise mixer is this file's
# CausalConv, which LEFT-pads by kernel_size-1 and therefore reads positions t-K+1..t only.
# causality_max_abs must stay exactly 0.0.
#
# EXACT step-0 no-op: the tables are ZERO-init, so e_t = 0 => k_t = v_t = 0 => vt = 0 =>
# RMSNorm(0) = 0/sqrt(0+eps) = 0 => Conv(0) = 0 => SiLU(0) = 0*sigmoid(0) = 0 => Y = 0,
# bit-exactly, for any value of W_K/W_V/conv. W_V must NOT be zero-init -- the same trap
# NGRAM_VE_DIM's up-projection documents, because dL/de_t = W_V^T dL/dv_t + W_K^T dL/dk_t, so a
# zero W_V (with a_t's own branch also starting at v=0) would leave the tables with identically
# zero gradient forever and the whole module would be dead weight. The CONV is the one thing
# that is zero-init, which makes Y == vt from step 1 and lets the conv branch switch itself on
# only if it earns its way -- the plan's own "conv zero-init".
#
# WHY ENGRAM_SITE DEFAULTS TO MID-STACK, AND WHY THAT WAS A SURPRISE (2026-09-13).
# T3.1's LNC=2 root-cause work established that a logical NeuronCore spans two PHYSICAL cores and
# that ONE dynamic-address gather makes the compiler re-shard the residual/V stream across both
# pcores at every layer boundary DOWNSTREAM of the injection, at ~30 bytes per DMA packet:
# ~+48 ms per micro-step for an n-gram injection at layer 3 of 6 versus ZERO at the last layer,
# INVARIANT to table geometry (see the NGRAM_VE_LAYERS block above). This module was therefore
# built with ENGRAM_SITE defaulting to -1, on the reasoning that a smaller d_mem cannot escape a
# descriptor-rate-bound cost and that the last layer is the only site with no downstream boundary.
#
# THAT DID NOT TRANSFER, and it was measured rather than assumed. Two paired 900s runs, identical
# except for the site, at 8.4M of Engram tables plus the frozen qk-shift:
#
#   site 5 (last)   1129 steps   val_bpb 1.063967
#   site 2 (mid)    1128 steps   val_bpb 1.062342
#
# ONE step apart -- mid-stack placement costs nothing here -- and 0.0016 bpb better. The
# difference from T3.1 is presumably WHAT CONSUMES the gathered rows: T3.1 injects into
# attention's V *inside* the block, so the gathered tensor is consumed by that layer's attention
# subgraph, whereas Engram consumes it in two dense matmuls plus an elementwise chain and adds one
# [B, T, C] tensor onto the residual AT a layer boundary. That mechanism is a hypothesis; the
# throughput equality is the measurement. Either way the conclusion is the practical one: the
# paper's mid-stack site is affordable, so the default is the paper's site, and the last-layer
# placement stays expressible for contrast (it is the arm that measures WORSE).
#
# Why the site matters at all for quality: injecting at the last block puts Y one norm and one
# matmul away from the logits, which makes Engram-lite mostly an additive output path -- the same
# thing T4.2's output bigram table was, and that was measured negative. Mid-stack leaves three
# more blocks to process the memory read, which is the point of the mechanism.
USE_ENGRAM = False
ENGRAM_SITE = "2"          # 0-based index of the block AFTER whose output Y is added to the
                           # residual; negative counts from the end. "2" is the paper's mid-stack
                           # placement and is both free and better here (see above). "-1" is the
                           # final residual just before the final norm.
                           # NOTE: ENGRAM_SHARE_NGRAM_TABLES needs an n-gram table at or BEFORE
                           # this site, and NGRAM_VE_LAYERS is locked at "-1", so share mode
                           # requires an explicit --engram-site -1 (or a wider --ngram-ve-layers).
ENGRAM_ORDERS = "2,3"      # which n-gram orders feed e_t. Engram uses n=2,3 and reports 4-grams
                           # "slightly suboptimal"; T3.1 independently found bigram+trigram worth
                           # -0.0017 over bigram-only at equal parameter count.
ENGRAM_HEADS = 1           # K independent hash heads PER ORDER, each with its own table and its
                           # own prime triple, so collisions decorrelate across heads (the same
                           # reason T3.1 gives each layer a different prime pair). K=1 is the
                           # cheap starting point: e_t is then 2 x ENGRAM_MEM_DIM wide.
ENGRAM_TABLE_MULT = 32     # rows per table = this x padded_vocab_size (32 x 8192 = 262144). The
                           # modulus is the largest prime <= that, for the reason GPT.ngram_mod
                           # documents. T3.1's own sweep found capacity SATURATED at 64x with
                           # dim 128 (128x was 0.0002 worse for 72 fewer steps), so there is no
                           # reason to start higher here: 32x x 128 x 2 tables = 67.1M new
                           # parameters, which is already 35% on top of the locked 190.1M.
ENGRAM_MEM_DIM = 128       # d_mem, the width of each table's row. The plan suggests 256 per head;
                           # 128 is T3.1's measured-best width on this stack and halves the
                           # gather/scatter/all-reduce traffic. e_t width = heads x orders x this.
ENGRAM_KEY_DIM = 0         # d, the width the gate's dot product is taken in. 0 => n_embd, which
                           # means q_t is the raw hidden state and NO extra projection is needed;
                           # >0 adds a W_Q Linear(n_embd, d) on the query side. 0 is both cheaper
                           # and the more faithful reading of "RMSNorm(h_t) . RMSNorm(k_t)".
ENGRAM_CONV_KERNEL = 4     # depthwise CAUSAL conv width on the gated memory (0 disables the conv
                           # branch entirely, leaving Y = vt). Zero-init, so it is a no-op until
                           # trained. The Engram paper's own ablation calls the conv the least
                           # important of its three ingredients ("tokenizer canonicalization and
                           # context-aware gating matter more than the conv"), which is why it is
                           # separately switchable.
ENGRAM_SHARE_NGRAM_TABLES = False
                           # reuse T3.1's ALREADY-GATHERED n-gram rows as e_t instead of building
                           # new tables. This is the zero-marginal-gather, zero-new-parameter arm:
                           # it adds no table, no extra dynamic-address gather, no extra gradient
                           # all-reduce and no extra optimizer state, so it isolates the Engram
                           # MECHANISM (hidden-state-dependent gating + residual injection + conv)
                           # from the added CAPACITY. Requires every shared table to live at or
                           # before ENGRAM_SITE, which the locked NGRAM_VE_LAYERS="-1" satisfies
                           # exactly when the site is the last layer.
ENGRAM_BF16 = True         # store the tables (and their optimizer state) in bf16, like T3.1's.
# The tables ride T3.1's optimizer group (RMSProp, no first moment, weight_decay 0, flat LR) for
# the reasons NGRAM_VE_OPT/NGRAM_VE_FLAT_LR document -- they are the same kind of object, and
# giving them their own differently-tuned group would confound the arm with an optimizer change.
# W_K/W_V (and W_Q when ENGRAM_KEY_DIM > 0) are ordinary matrices and go to Muon; the conv kernel
# is depthwise and goes to AdamW at MATRIX_LR with wd=0, exactly like block.local_conv.

# ENGRAM_HASH_PRIMES / engram_hash_primes() / engram_site_index() live further down, next to
# NGRAM_HASH_PRIMES, because they need _next_prime() which is defined there.

# -----------------------------------------------------------------------------
# T4.2: Aster's OUTPUT bigram logit table (research/frontier-research-plan-2026-09-09.md §11.1,
# change (2) of the three Aster describes). A [V, V] table indexed by the CURRENT token whose row
# is added straight to the next-token logits, gated per token by sigmoid(Linear(n_embd -> 1)) with
# a negative bias. Strictly causal by construction: row = table[idx[b, t]] and the gate reads
# x[b, t], so position t's logit bias is a function of tokens <= t only, exactly like lm_head's
# own output.
#
# This is NOT the T4.1 count-based mixer that was closed as a negative on 2026-09-12: that one was
# a FROZEN, corpus-counted, Kneser-Ney-smoothed distribution mixed into the softmax after training
# (+0.0006, and 75% of even that was already absorbed by T3.1). This one is gradient-trained
# jointly with the model and can learn residual CORRECTIONS to the neural logits rather than being
# pinned to the corpus's conditional distribution. Prior after T4.1 and T3.1 (which already banked
# -0.0152 from the same bigram statistics on the attention-value side): ten-thousandths, not
# thousandths. Measured result recorded in research/trn2-experiment-results.md.
USE_OUT_BIGRAM = False
OUT_BIGRAM_BF16 = True     # store the [V, V] table in bf16 (Aster's choice): 8192^2 x 2 B = 134 MB
                           # of parameters and 67.1M on the tie-break's parameter count.
OUT_BIGRAM_GATE_BIAS = -2.0  # bias init of the per-token gate. Aster specifies -2, i.e. the gate
                           # starts at sigmoid(-2) = 0.119 rather than neutral. The step-0 no-op
                           # comes from the ZERO-init TABLE (see init_weights), so it holds for any
                           # bias value; the bias only sets how fast the mechanism can turn on.
OUT_BIGRAM_LR = 0.0        # 0 => UNEMBEDDING_LR. The table is an additive bias on the logits, so
                           # lm_head's own LR is the right scale reference, and it gets lm_head's
                           # AdamW betas in its own group (a [V, V] embedding is Muon-ineligible
                           # for the same reason wte is: it is read by a gather, not a matmul).

# -----------------------------------------------------------------------------
# T3.9: the cheap architectural one-liners from Recursive's optimized_from_karpathy.py
# (research/frontier-research-plan-2026-09-09.md §11.1). Each is independently toggleable because
# the whole point of the item is to find out which of them individually clears the 0.002 bar --
# Recursive ships them as a bundle and gives no per-item attribution.
RELU2_TAU = 0.0            # shifted ReLU^2: the MLP activation becomes relu(h - RELU2_TAU)^2.
                           # 0.0 = the historical relu(h)^2. Recursive: "tau=0.5 confirmed
                           # optimal". Nonzero forces the eager activation (the example NKI relu2
                           # kernel implements the unshifted form only) -- USE_NKI_RELU2 is off by
                           # default anyway, and relu_squared asserts the combination is not used.
USE_MLP_SANDWICH_NORM = False  # x = x + norm(mlp(norm(x))) instead of x = x + mlp(norm(x)).
                           # Attention is already pre-norm-only here too, but Recursive applies
                           # the sandwich to the MLP only, which is what this reproduces.
USE_HEAD_GATE = False      # T3.6: per-head gate on the attention output, read from a residual
                           # slice. Recursive's form is y = norm(y) * 2*sigmoid(head_gate(x[..., :32]))
                           # per head ("head-level MoE"); the plan's own T3.6 entry is the
                           # gate-only variant y = y * sigmoid(Linear(12, n_head)(x[..., :12])).
                           # HEAD_GATE_NORM picks between them.
HEAD_GATE_NORM = True      # RMSNorm each head's attention output before gating (Recursive's form).
                           # NOTE this is the one item here that is NOT a step-0 no-op: it rescales
                           # every head to unit RMS at init. With HEAD_GATE_NORM=False the arm is
                           # an exact identity at step 0 (zero-init weight, HEAD_GATE_BIAS=0 =>
                           # 2*sigmoid(0) = 1.0).
HEAD_GATE_CHANNELS = 32    # width of the residual slice the gate reads. The slice OFFSET is
                           # computed to sit after every value-embedding gate's slice (see
                           # head_gate_slice) so the gates never compete for the same channels:
                           # unigram VE has 0:32, bigram 32:64, trigram 64:96, so this takes
                           # 96:128 at the locked config.
HEAD_GATE_MUL = 2.0        # gate = HEAD_GATE_MUL * sigmoid(...). 2.0 makes zero-init NEUTRAL
                           # (gate == 1.0), matching every other gate in this file; 1.0 gives the
                           # plan's T3.6 sigmoid-only form, whose neutral point is a positive bias.
HEAD_GATE_BIAS = 0.0       # bias init of the head gate. With HEAD_GATE_MUL=2 the neutral value is
                           # 0.0; with HEAD_GATE_MUL=1 the plan asks for sigma ~= 0.5-0.75, i.e. a
                           # bias in [0.0, 1.1].
USE_OUT_POOL = False       # zero-init pooling of the last OUT_POOL_LAYERS block outputs into the
                           # final residual, before the final norm: x = x + sum_k lam_k * h_k with
                           # every lam_k zero-init, so step 0 is an exact no-op.
OUT_POOL_LAYERS = 3        # how many trailing block outputs to pool (Recursive uses 3).
USE_QK_SHIFT = False       # causal token shift on the Q/K inputs only: q_in = k_in = x + beta*x_prev
                           # with a learned PER-DIM beta, V left unshifted. Different mechanism from
                           # USE_LOCAL_CONV (T0.1), which is a learned depthwise conv applied to the
                           # residual stream itself and feeds all three of Q/K/V.
QK_SHIFT_BETA = 0.3        # per-dim beta init (Recursive: 0.3). Not a step-0 no-op by design.
QK_SHIFT_FREEZE = False    # T3.9-followup / T1.x throughput probe: keep beta OUT of every optimizer
                           # group and set requires_grad=False, so it stays at QK_SHIFT_BETA forever.
                           # With --qk-shift-beta 0.0 that makes the whole shift a MATHEMATICAL
                           # IDENTITY at every step (x + 0.0*x_prev == x bit-exactly in IEEE fp, and
                           # the backward contribution through the shifted branch is scaled by the
                           # same 0.0), while the elementwise ops STAY in the compiled graph. That is
                           # the decisive probe for the observed anomaly that adding almost any
                           # elementwise op to the block body makes a step 8-9% FASTER: if the frozen
                           # arm keeps the speedup with bit-identical losses, the +8% is free.
USE_BLOCK_NUDGE = False    # The harvestable form of the block-body compiler-scheduling anomaly, once
                           # --qk-shift-freeze had shown the +10% throughput survives with the
                           # mathematics removed: add `x = x + nudge * x` with ONE nudge scalar per
                           # block, frozen at exactly 0.0 and in no optimizer group. That is a
                           # mathematical identity forever (x + 0.0*x == x bit-exactly for finite x,
                           # and the backward contribution through the extra branch is scaled by the
                           # same 0.0), so it can only ever change the compiled graph -- which on
                           # this stack is the entire point. Strictly cheaper than the frozen
                           # qk-shift: a broadcast scalar mul plus an add, with no causal_shift_1
                           # pad/slice and no per-dim vector.
BLOCK_NUDGE_SITE = "qk"    # where the nudge lands. "qk" puts it exactly where --qk-shift put it (the
                           # Q/K input, V untouched); "resid" puts it on the residual stream right
                           # after the resid/x0 mix, so all of Q/K/V and the MLP see it. Both are
                           # identities; they differ only in which compiled region gets perturbed.
USE_PARALLEL_BLOCK = False # DISCARDED: a large real negative at the full budget. Step-matched at
                           # ~2140 steps (a SHORTENED 1630s budget so its +10.5% throughput stays
                           # below the ~2195-step data wall): parvfc-1630s 2143/1.032908 and
                           # parall-1630s 2128/1.033200 vs the same-day control's 2155/1.018303, i.e.
                           # +0.0146 to +0.0149, parameter-matched (+12 lambda scalars) with
                           # causality_max_abs 0.0 and the n-gram VE confirmed still on. The two
                           # coalesce modes agree to 0.0003, so the regression is the PARALLEL
                           # STRUCTURE, not the coalescing. It is also a warning about early loss: the
                           # arm is AHEAD at step 110 (3.9445 vs 3.9640 at lrm 1.000 on identical
                           # data) and 0.047 behind by step 1000. At DEPTH=6 there are only six
                           # chances for the MLP to consume what attention just produced and this
                           # removes all of them; the architecture comes from 4-8x deeper models.
                           # The coalescing half IS worth +10.5% and its recipe (contiguous q/k slices
                           # keep the rope+QK-norm NKI kernel alive -- which is the real explanation of
                           # USE_FUSED_QKV's -0.0428) carries over to the serial block, but under the
                           # data wall throughput above ~2.6% is worth ~0 bpb. See
                           # trn2-experiment-results.md.
                           # GPT-J / PaLM-style parallel block: attention and the MLP both read the
                           # SAME norm(x) and their outputs are summed into the residual together,
                           # `x = x + la*attn(norm(x)) + lm*mlp(norm(x))`, instead of the MLP reading
                           # the post-attention residual. Removes three of the four norm() calls per
                           # block and one residual add, and makes the attention and MLP matmuls
                           # mutually independent instead of serialised through the attention output
                           # -- both of which matter far more here than the FLOPs, because this stack
                           # is dispatch- and serialisation-bound (~28% serialisation loss at ~8% MFU).
                           # The quality cost is that the MLP no longer sees what attention just did;
                           # published results at this scale call that roughly neutral, which is
                           # exactly the kind of claim that only a paired A/B on this box can settle.
                           # Bit-identical at step 0 (mlp.c_proj is zero-init, so the MLP contributes
                           # exactly 0 whichever input it reads, and la starts at 1.0).
PARALLEL_BLOCK_COALESCE = "none"
                           # Only read when USE_PARALLEL_BLOCK is on. Which of the block's four
                           # input projections (c_q, c_k, c_v, mlp.c_fc -- all four take the SAME
                           # normed residual in a parallel block, and all four have in_features =
                           # n_embd) are merged into one wide matmul on the Block itself:
                           #   "none" -- four separate Linears, as today.
                           #   "vfc"  -- c_v and c_fc merged into Linear(640, 3200); c_q/c_k stay
                           #             separate. Three matmuls instead of four and NO extra copy:
                           #             a [b,t,640] last-dim-contiguous slice still .view()s to
                           #             [b,t,10,64], and v never reaches the rope+QK-norm kernel.
                           #   "all"  -- one Linear(640, 4480) for q,k,v,c_fc. Needs .contiguous()
                           #             on the q and k slices, because rope_norm() falls back to
                           #             eager on a non-contiguous input -- which is exactly the
                           #             mechanism that made USE_FUSED_QKV a -0.0428 regression
                           #             (see trn2-experiment-results.md, Solution 13c). So this
                           #             mode trades 2 copies for 3 saved matmul launches.
                           # Bit-identical to "none" at step 0 either way: init_weights fills the
                           # slices in the baseline's order with the baseline's per-slice bounds.
                           # NOT identical afterwards, and not only because of the copies -- Muon
                           # now orthogonalizes one [4480,640] matrix instead of four smaller ones,
                           # which is a real optimizer change and part of what the arm measures.
USE_QKV_NORM_CSE = False   # Block.forward calls norm() THREE times on q_in/k_in/v_in, which in the
                           # locked config are the SAME tensor object -- three F.rms_norm nodes with
                           # identical inputs. This hoists them into one call whenever the three
                           # inputs are literally the same object (`q_in is k_in is v_in`), which is
                           # a bit-exact identity: F.rms_norm is deterministic, so three calls on one
                           # tensor return three equal results, and the backward just accumulates the
                           # three incoming grads into one node instead of separately. Falls back to
                           # the three-call form automatically whenever the inputs really do differ
                           # (MUDD, --qk-shift, --block-nudge --block-nudge-site qk), so it composes
                           # with everything. Worth probing on its own because this stack is
                           # dispatch-bound (~67 NEFF launches per micro-step): 2 saved launches per
                           # block times 10 blocks is a bigger fraction of a step than of the FLOPs.
USE_X0_GATE = False        # DISCARDED at 900s: per-step NEUTRAL (-0.00050 residual, inside the
                           # <0.001 noise floor) and 13% SLOWER per step, which is the entire delta.
                           # Paired 900s A/B, fresh same-commit control: 951 steps/1.074042 vs
                           # 1071/1.065348; the -0.077/ln(steps) throughput term alone predicts
                           # +0.00919 against an observed +0.00869. The gate does do something early
                           # (step-100 loss 3.9594 vs 4.0002, both at lrm 1.000 on identical data) but
                           # it has washed out by convergence. Four extra elementwise ops per block --
                           # one of them a channel-mean REDUCTION -- cost 13% in this dispatch-bound
                           # graph, which also falsifies the idea that any small block-body edit
                           # collects the frozen-qk-shift speedup. See trn2-experiment-results.md.
                           # Recursive's INPUT-DEPENDENT x0 gate, the rehabilitated form of the
                           # GPU-discarded USE_SELECTIVE_GATE (which was +0.0092 there): scale this
                           # block's x0 skip by `2*sigmoid(s * mean(x))` with ONE learned scalar s
                           # per layer and `mean` taken over the residual CHANNELS of each token
                           # independently (so it is causally trivial -- position t reads only
                           # position t, same argument as MuddCompose). s is zero-initialized, and
                           # 2*sigmoid(0) == 1.0 exactly in IEEE fp, so the gate is a BIT-EXACT
                           # no-op at step 0 against the plain `x0_lambda * x0` skip. Differs from
                           # USE_SELECTIVE_GATE in three ways that matter on this stack: it touches
                           # only the x0 term (not resid_lambda), its input is a free per-token
                           # reduction rather than a Linear over 32 residual channels, and it is
                           # multiplicative-around-1 rather than (1 + g).


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
        if any(has_ngram_bigram(i, n_layer) for i in range(n_layer)):
            claimed = 2 * VE_GATE_CHANNELS
        if any(has_ngram_trigram(i, n_layer) for i in range(n_layer)):
            claimed = 3 * VE_GATE_CHANNELS
    start = claimed
    stop = start + HEAD_GATE_CHANNELS
    assert stop <= n_embd, (start, stop, n_embd)
    return start, stop


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


# 30 well-spaced primes, generated deterministically so a given (layer, order) always
# hashes the same way for the life of this code -- and provably prime, rather than a
# hand-typed list that might quietly contain a composite. The upper bound matters: the
# hash multiplies a token id (<= NGRAM_PAD_ID = padded_vocab_size = 8192) by one of
# these and the product must stay inside int32, so max prime x 8192 < 2^31 requires
# every prime < 262144. 8192 + 8000*29 = 240192 leaves real headroom.
NGRAM_HASH_PRIMES = tuple(_next_prime(8192 + 8000 * k) for k in range(30))


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
    i = (2 * layer_idx) % (len(NGRAM_HASH_PRIMES) - 1)
    return NGRAM_HASH_PRIMES[i], NGRAM_HASH_PRIMES[i + 1]


def ngram_trigram_primes(layer_idx: int) -> tuple[int, int, int]:
    """(p_prev2, p_prev, p_cur) for this layer's trigram hash. Offset into the prime
    table so it never coincides with any layer's bigram pair."""
    i = 12 + (3 * layer_idx) % (len(NGRAM_HASH_PRIMES) - 14)
    return NGRAM_HASH_PRIMES[i], NGRAM_HASH_PRIMES[i + 1], NGRAM_HASH_PRIMES[i + 2]


def ngram_layer_set(n_layer: int) -> frozenset[int]:
    """NGRAM_VE_LAYERS parsed to absolute indices. Empty set means "no restriction".

    Negative indices wrap (so "-1" is the last layer whatever DEPTH is), which is what makes
    the winning `NGRAM_VE_LAYERS="-1"` configuration survive a depth change instead of
    silently pointing at the wrong layer.
    """
    if not NGRAM_VE_LAYERS:
        return frozenset()
    return frozenset(int(tok) % n_layer for tok in NGRAM_VE_LAYERS.split(",") if tok.strip())


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


# T3.2 Engram-lite hash primes. Deliberately DISJOINT from NGRAM_HASH_PRIMES: in share mode the
# two mechanisms read the same table, but in own-table mode their hashes must not coincide, or an
# Engram table would be a renamed copy of a T3.1 table. Same int32 bound as NGRAM_HASH_PRIMES
# (max prime x 8192 < 2^31 requires every prime < 262144; 20011 + 7919*23 = 202148 has headroom).
ENGRAM_HASH_PRIMES = tuple(_next_prime(20011 + 7919 * k) for k in range(24))


def engram_orders() -> tuple[int, ...]:
    """ENGRAM_ORDERS parsed. Only 2 and 3 are supported (Engram's own choice; 4-grams were
    reported "slightly suboptimal" and the index shift would need a third pad column)."""
    orders = tuple(int(o) for o in ENGRAM_ORDERS.split(",") if o.strip())
    assert orders and all(o in (2, 3) for o in orders), f"ENGRAM_ORDERS={ENGRAM_ORDERS!r}"
    return orders


def engram_hash_primes(head_idx: int, order: int) -> tuple[int, ...]:
    """The `order` multipliers for hash head `head_idx` of that order, oldest token first.

    Orders get disjoint 12-prime regions so no head of order 2 can ever share a multiplier with
    a head of order 3; within a region each head takes its own consecutive slice.
    """
    base = (order - 2) * 12 + head_idx * order
    assert base + order <= len(ENGRAM_HASH_PRIMES), \
        f"ENGRAM_HEADS={ENGRAM_HEADS} needs more than {len(ENGRAM_HASH_PRIMES)} hash primes"
    return tuple(ENGRAM_HASH_PRIMES[base + j] for j in range(order))


def engram_site_index(n_layer: int) -> int:
    """ENGRAM_SITE resolved to an absolute block index (negatives count from the end)."""
    return int(ENGRAM_SITE) % n_layer


WINDOW_PATTERN = "L"    # per-layer attention window pattern, cycled to n_layer length; 'L' = full context, 'S' = short window
SHORT_WINDOW_FRAC = 0.5  # short window size as a fraction of SEQ_LEN (only matters for 'S' layers)
ROPE_BASE = 100000       # rotary embedding base frequency. Scaling up to 500000 at SEQ_LEN=4096 made no difference (848/1.101979 vs 849/1.101243) -- rules out RoPE-extrapolation as the cause of 4096's quality drop-off. See trn2-experiment-results.md
USE_LOCAL_CONV = True    # depthwise causal conv1d before attention in each block, RWKV/Mamba/Hyena-style.
# HISTORY (T0.1, day-1 frontier-plan item, fixed this): for most of the competition this was DEAD
# COMPUTE -- CausalConv.weight was zero-initialised, init_weights never touched it, and
# setup_optimizer() collected only attn/mlp/lambda/ssm/rwkv/ve params, so local_conv.weight was in
# NO optimizer group and stayed exactly zero for the whole run (conv(norm(x)) exactly 0.0, x + 0.0
# bit-exact x). Asserted by research/scratch/local_conv_deadparam_check.py, which is the reference
# for what "dead" meant before this fix. setup_optimizer() now DOES register local_conv.weight in
# its own AdamW group (lr=MATRIX_LR, betas=(0.9,0.95), wd=0.0) -- see the T0.1 comment there -- so
# it trains like every other tensor; check local_conv_deadparam_check.py's live output (not this
# comment) for the current true ungrouped-param count. Init is still exactly zero, so step 0 remains
# a bit-exact no-op regardless of whether the weight then moves. Left ON (rather than removed)
# independent of whether training it helps: turning the COMPUTE off measured SLOWER on real
# hardware (6.9x compiled-graph slowdown, unexplained compiler effect). Training the weight IS a
# real win and is now the default -- see TRAIN_LOCAL_CONV below for the numbers; drive the A/B for
# THAT fix specifically with --train-local-conv / --no-train-local-conv (see setup_optimizer).
LOCAL_CONV_KERNEL = 2  # T0.1 SWEEP WINNER, now the locked default (was 4). 900s sweep with
# TRAIN_LOCAL_CONV on: kernel=2 1049 steps/1.085529, kernel=4 1009/1.085591, kernel=8 999/1.088411 --
# kernel=2 wins on BOTH axes (most steps AND best bpb; a smaller depthwise kernel is less compute per
# step and cost nothing in quality here). Confirmed at full 1800s as part of the winning pair below.
# See trn2-experiment-results.md.
TRAIN_LOCAL_CONV = True  # T0.1 WINNER, now the locked default (was False). Whether local_conv.weight
# is registered with the optimizer (see setup_optimizer's local_conv_params group and the
# USE_LOCAL_CONV comment above). Kept a SEPARATE toggle from USE_LOCAL_CONV/--local-conv on purpose:
# USE_LOCAL_CONV controls whether the conv COMPUTE happens at all (a shape-critical toggle --
# removing the compute changes num_params and was independently measured 6.9x-compiled-graph-slower,
# so it must stay ON), while TRAIN_LOCAL_CONV controls only whether its (zero-init) weight is
# optimizer-visible. For most of the competition this was OFF, which made the conv DEAD COMPUTE.
# REAL WIN, paired same-day A/B from one identical binary: 900s 1009/1.085591 (on, kernel=4) vs
# 1012/1.090389 (off) = -0.004798 bpb at equal step count; full 1800s 2100 steps/1.038386 (on,
# kernel=2) vs 2037/1.040879 (off) = +63 steps (+3.1%) and -0.002493 bpb. That 2100/1.038386 is the
# best-known local config and the baseline every subsequent experiment is measured against.
# --no-train-local-conv reproduces the old (dead-conv) behaviour for provenance.
USE_GATED_MLP = False    # SwiGLU-style: c_fc(x) * silu(c_gate(x)) instead of relu(c_fc(x))**2; adds one weight matrix
USE_WEIGHT_CACHE = True  # cache each Linear's dtype-cast weight once per optimizer step instead of every forward call
USE_FP8_LINEAR = False  # run the MLP's c_fc/c_proj FORWARD matmul in torch.float8_e4m3fn (current-scaling), backward left in bf16/fp32. Scope deliberately limited to the MLP (the widest matmuls, matching modded-nanogpt's own FP8-MLP records #84/#89); attention's c_q/c_k/c_v/c_proj and lm_head are NOT touched, because those feed the rope/softcap NKI kernels and would compound quantization noise into paths this repo has verified to 1-bf16-ULP. Design (see the "FP8 training-path precision" section of research/trn2-experiment-results.md): CURRENT-scaling, i.e. each tensor's own amax every call, NOT delayed-scaling amax-history -- that sidesteps Transformer Engine's own documented 50-200-step warmup instability, which would eat a far larger fraction of this competition's ~2000-step budget than of a normal long pretrain. FORWARD-only fp8: _Fp8LinearFn casts both operands, and its backward recomputes grads against the saved FULL-PRECISION tensors, so gradient quantization (a separate, unvalidated risk) is never taken on. DEFAULT OFF, and it is a real, cleanly-measured DOUBLE negative on this specific Trainium2 native-PyTorch stack -- both axes lose, so there is no configuration in which this pays. (1) SPEED, and this is the decisive one: the only NUMERICALLY CORRECT fp8 GEMM on this stack, torch._scaled_mm, measures 46366 ms/call against bf16 F.linear's 0.340 ms at the real c_fc shape [2048,640]x[640,2560] -- 136521x SLOWER, matched methodology, a real host-visible sync per iteration on both arms, and ~46.4 s on EVERY call (46180/47053/46798/46500/46217/45950) not just the first, so it is a per-call cost that never amortizes and is NOT compilation. Controls isolate it precisely to the GEMM: the bf16->fp8 CAST alone is 3.096 ms and an fp8->bf16 upcast followed by an ordinary bf16 matmul is 5.217 ms, both sane -- fp8 as a DTYPE is fine here, there is simply no fp8 Tensor-Engine GEMM path exposed through this stack. CORRECTS an earlier research note in trn2-experiment-results.md which recorded that "torch.matmul on direct float8_e4m3fn tensors works natively on the neuron device": it does not raise, but it is SILENTLY WRONG -- it returns a float8_e4m3fn (not fp32-accumulated) result that is all-NaN at this shape (rel_l2=nan, absmax=nan), which is exactly why this implementation uses torch._scaled_mm and never torch.matmul on fp8 operands. (2) QUALITY: fp8 costs 13x the per-matmul error of the bf16 it replaces -- forward rel_l2 0.0374 vs bf16's 0.00287 against an fp64 CPU reference, and the real hardware number reproduces the offline harness's CPU-simulated prediction to 4 decimals (0.0374 vs 0.0374), so research/scratch/fp8_linear_equiv.py is a trustworthy model of the hardware. Because 46.4 s/GEMM x 12 GEMMs (2 per MLP x DEPTH=6) is ~557 s of forward per step, a real dozens-to-100-step loss-trajectory run is ~10 min/step and physically infeasible on the shared box, so the open compounding question was instead answered by the harness's DEPTH=6 additive-residual simulation. Kept in-tree, off by default, per this session's standing pattern for honest negatives; --fp8-linear runs the A/B from one identical binary. Read by MLP.__init__, i.e. before GPT(config).
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
USE_MAMBA2 = False  # Mamba2 / SSD hybrid (arXiv:2405.21060, Dao & Gu): replace SOME blocks' CausalSelfAttention with a real Mamba2Mixer (in_proj -> depthwise causal conv1d -> selective state-space scan -> gated RMSNorm -> out_proj), keeping the MLP, the resid/x0 skip and the local_conv untouched, so MAMBA2_PATTERN picks which layers are which. The SSD scan is the chunked "structured state-space duality" form, verified exact (float64, max_rel < 1e-12) against an independently written naive sequential recurrence in research/scratch/mamba2_ssd_ref.py and re-checked by research/scratch/mamba2_train_integration_check.py on every commit. Causally safe by construction AND measured (causality_max_abs=0.0 on real hardware): the conv1d is left-padded so position t only ever reads t-3..t, and the scan's two decay matrices are masked BEFORE exp -- the intra-chunk mask is lower-triangular (s <= t) and the inter-chunk mask STRICTLY lower-triangular (chunk c reads only chunks c' < c), with the mask added as -30000 to an exponent already clamped to <= 0, so exp() gives exactly 0.0 there and no +inf * 0 = NaN is possible (the latent bug in the sibling repo's kernel, which masks after exp). Zero-init no-op: mixer.out_proj is zero-initialized like every other block output projection, so a fresh Mamba2 layer is an exact identity. MEASURED at the locked shape and it is a real NEGATIVE on wall clock, not a win: at 45.0M params vs the transformer's 55.7M it is a dead heat on quality at a fixed 300 steps (1.270639 vs 1.269458 bpb), but it is slower per step, so at a fixed 1800s budget it only reached 908 steps against the transformer's 1995 and lost 1.109065 to 1.043489 bpb. TWO CORRECTIONS to the first round of that measurement, both re-measured directly (2x3 steady-state-dt isolate at the real 4-rank/lnc=2 shape, research/run-logs/l2iso-*.log): (1) `m2hybrid-nki-fixed300` and `m2hybrid-nki-fullbudget` both ran at dt 1.98-2.00 s/step, which is exactly the measured `--no-nki-ssd` speed (2.00) and not the kernel's real 1.53 -- the NKI scan was evidently not active in them despite `--nki-ssd`, which is why ssd_scan now announces the branch it actually took. (2) torch.compile is not a 2.37x-vs-0.90x story: on current code (post `--tensor-lr-scalars`) the transformer only gets 0.97 -> 0.92 s/step from compile, and the Mamba2 path measures 2.00 (eager+eager-scan) / 2.20 (compile+eager-scan) / 1.53 (eager+NKI) / 4.71 (compile+NKI). The best real Mamba2 configuration is therefore eager + the NKI kernel at 1.53 s/step against the transformer's 0.92 -- a 1.66x gap, not 2.15x. CLOSED OUT, and the answer is still no: the cheapest mixer geometry that keeps the model working (--mamba2-expand 1 --mamba2-d-state 64 --mamba2-conv-kernel 2) gets a real 1.36x to 1.11 s/step, but a fixed-300-step sweep shows it gives up the quality-per-step tie that motivated the whole line (1.288748 vs the paper geometry's 1.273513 and the transformer's 1.269458), and at a real 900s budget it lands 795 steps / 1.128185 against the transformer's 1004 / 1.093785 -- still +0.034 bpb. That residual is out of reach of any further speed work: fitting bpb ~ a - 0.077*ln(steps) on current-code transformer runs, closing 0.034 bpb on step count alone needs 1.57x more steps, i.e. 0.72 s/step, FASTER than the transformer itself, so even a zero-cost mixer would not do it -- and a zero-cost mixer only buys a tie anyway, since the dead heat held for the paper geometry and not for the cheap one this needs. The mixer COUNT then turned out to be the axis nobody had measured (see MAMBA2_PATTERN): one Mamba2 layer of six is the interior optimum, not five, and the best configuration the whole search produced -- one mixer at the cheap geometry, `--mamba2 --nki-ssd --no-compile --mamba2-pattern AMAAAA --mamba2-expand 1 --mamba2-d-state 64 --mamba2-conv-kernel 2` -- lands 887 steps / 1.106694 at a real 900s budget against the transformer's 1004 / 1.093785. That is a 4x narrowing of the deficit (+0.0508 at M=5 paper -> +0.0344 at M=5 cheap -> +0.0129 here) and still a loss, and the decomposition is what closes the question: of that +0.0129 only +0.0095 is step count, while +0.0034 is quality-per-step, so a mixer that cost ZERO time would still be behind. The trajectory also converges on the baseline from above monotonically -- every move toward less Mamba2 helped, and the endpoint of that trajectory is M=0, the pure transformer -- so there is no interior optimum that wins. Kept off by default and kept in-tree as a complete, working, exactness-verified implementation with its geometry and pattern on the CLI. See research/experiment-log.jsonl rows m2-g{0..3}-*-300, m2-pat{1,2,3}-*-300, m2-g{0,3}-*-900s and m2-pat1-cheap-900s (baselines xf-fixed300 / lrmq25-900s / tensorlr-fullbudget), research/run-logs/m2-*.log and l2iso-*.log, and the "Mamba2/SSD hybrid" section of research/trn2-experiment-results.md.
MAMBA2_PATTERN = "MMAMMM"  # per-layer mixer selection, cycled to n_layer like WINDOW_PATTERN: "M" = Mamba2 mixer, anything else = ordinary attention. Layers that become Mamba2 lose their value-embedding (there is no attention V to inject one into), so the ve_gate/value_embeds tables for those layers are not created at all -- which is itself a reason a HIGH Mamba2 count is costly, since "MMAMMM" throws away 5 of 6 of those tables. This default is kept only because every recorded Mamba2 run used it; it is NOT the best point on this axis. The mixer-count axis was unmeasured until it was swept directly (300 fixed steps, paper geometry, vs the transformer's 0.907 s/step / 1.269458 bpb): M=1 "AMAAAA" 1.080 s/step / 1.267050, M=2 "AAMAAM" 1.201 / 1.268748, M=3 "AMAMAM" 1.250 / 1.282111, M=5 "MMAMMM" 1.509 / 1.273513. So ONE mixer is the interior optimum and actually beats pure attention on quality-per-step, while five is both far slower and worse. It still does not pay: regressing total = base + n_attn*a + n_m*m over those points plus the geometry sweep prices one mixer at +0.114..+0.173 s/step over the attention layer it replaces (+0.041 at the cheap geometry), against a break-even price of 0.077*ln(1.080/0.907) = 0.0134 bpb for the 0.0024 it earns -- 5.6x short. See the USE_MAMBA2 comment for the 900s numbers.
MAMBA2_HEAD_DIM = 128  # P, the per-head state width of the scan. 128 == the SBUF partition limit, which is what lets one head's chunk tile fit a single NKI matmul.
MAMBA2_D_STATE = 128  # N, the SSM state dimension per group. The kernel requires N <= MAMBA2_CHUNK.
MAMBA2_CHUNK = 128  # Q, the SSD chunk length. Must divide sequence_len and satisfy N <= Q <= 128 (the partition limit) for the NKI kernel; anything else silently falls back to the eager scan.
MAMBA2_EXPAND = 2  # d_inner = MAMBA2_EXPAND * n_embd, the standard Mamba/Mamba2 expansion factor
MAMBA2_N_GROUPS = 1  # G, B/C groups shared across heads (G=1 == "all heads share one B/C", the Mamba2 default)
MAMBA2_CONV_KERNEL = 4  # k for the depthwise causal conv1d, the Mamba2 reference default
MAMBA2_DT_MIN = 0.001  # dt_bias is initialized to softplus^-1 of a log-uniform draw in [MAMBA2_DT_MIN, MAMBA2_DT_MAX]
MAMBA2_DT_MAX = 0.1
MAMBA2_DT_INIT_FLOOR = 1e-4

USE_RWKV = False  # RWKV "x051a" (BlinkDL, github.com/BlinkDL/RWKV-LM; ported from the nanoRWKV
# reference at ../nanoRWKV/model.py, RWKV_TimeMix_x051a + RWKV_ChannelMix_x051a). Replaces the
# layers marked "R" in RWKV_PATTERN with an RWKV time-mix (instead of attention) AND an RWKV
# channel-mix (instead of the MLP), keeping this repo's pre-norm, resid/x0 skip, local_conv and
# value-embedding wiring. x051a is the formulation specifically designed to train in plain
# PyTorch with no custom CUDA kernel, which is why it is the right first cut here.
#
# The operator, per head (r/k/v are [T,N], w = exp(-exp(time_decay)) in (0,1), u = time_faaaa):
#     y_t = u * (r_t . k_t) * v_t  +  sum_{s<t} w^(t-s-1) * (r_t . k_s) * v_s
# i.e. a linear-attention/decay kernel. Strictly causal by construction (only s <= t appears),
# verified as a real measurement rather than an assumption: research/scratch/rwkv_x051a_ref.py
# cross-checks three independently written implementations (per-token sequential recurrence,
# closed-form O(T^2) decay matrix, and a VERBATIM transcription of nanoRWKV's chunk loop) to
# float64 round-off, and perturbing position t provably moves nothing at positions < t.
#
# The chunked evaluation is LOOP-FREE here, unlike the reference's `for i in range(T//Q)`. The
# inter-chunk state recurrence S_{c+1} = w^Q * S_c + G_c has a per-head SCALAR decay w^Q, so it
# collapses to the same masked-decay matmul at chunk granularity that the intra-chunk term uses
# at token granularity -- exactly the trick ssd_chunked already uses for Mamba2, and for the
# same reason (the native-eager backend lowers each op to its own NEFF, so T/Q = 16 sequential
# chunk steps would cost 16x the dispatch). See rwkv_chunked.
#
# The decay-matrix construction is also rewritten: the reference builds it with
# w.pow(ind)/torch.tile/.flip()/.view() gymnastics through an [H, 2Q*Q] intermediate; here every
# decay tensor is exp(log_w * <constant integer exponent> + <additive causal mask>) with the
# exponent clamped to >= 0 BEFORE the multiply, so (a) no reshape gymnastics reach the device,
# (b) masked entries are exactly exp(-30000) == 0.0, and (c) no live exponent can ever be
# positive, hence no +inf that a 0 mask would turn into NaN -- the same masked-before-exp
# discipline as the SSD scan. Verified equal to the verbatim reference construction to float64
# round-off (research/scratch/rwkv_train_integration_check.py).
#
# MEASURED VERDICT: OFF, and closed rather than parked -- a clean double negative on real
# hardware (4 ranks, lnc=2, T=2048, 55.75M params, parameter-matched against the transformer
# so the A/B is mixer-vs-mixer):
#   rwkv-pure-900s   480 steps  1.226041  (1.863 s/step)   <- real 900s budget
#   tensorlr-900s    995 steps  1.094327  (0.899 s/step)   <- transformer, same budget
#   xf-fixed480      480 steps  1.205611  <- equal-STEPS control: RWKV also loses per step
#   rwkv-fixed995    995 steps  1.144032  <- steelman: RWKV given the baseline's step count
# 1. Speed: 2.05x slower per step than the transformer, and the cost is NOT the chunked
#    matmuls -- RWKV_CHUNK in {64,128,256,512} all land within 1.86-1.98 s/step
#    (research/scratch/rwkv_chunk_sweep.sh), i.e. it is per-op dispatch/elementwise overhead
#    spread over the block, not FLOPs, so blocking cannot fix it. torch.compile does not
#    either: 1.82 s/step behind a 505 s compile that would eat half a 900 s budget.
# 2. Quality: worse at EQUAL steps too, and the gap WIDENS with training (+0.0204 bpb at 480
#    steps, +0.0497 at 995). That is what closes the line: even a from-scratch NKI kernel that
#    made the mixer free could at best buy the transformer's step rate, and RWKV is still
#    0.0497 behind there -- so no kernel work is justified, which is exactly what the cheap
#    eager port was built to find out before paying for one.
# Same shape of result as USE_MAMBA2 (see research/trn2-experiment-results.md): at this scale
# and budget a linear-attention/decay mixer is both slower AND worse than plain quadratic
# attention over 2048 tokens. Kept in-tree, fully wired and verified, because the finding is
# only reusable if the code is.
RWKV_PATTERN = "RRRRRR"  # per-layer mixer selection, cycled to n_layer like WINDOW_PATTERN/MAMBA2_PATTERN:
# "R" = RWKV time-mix + channel-mix, anything else = ordinary attention + MLP. Default is a PURE
# RWKV model (every layer), the simpler first cut; a hybrid is a follow-up if pure RWKV shows
# promise (same phased-ambition pattern used for Mamba2). Mutually exclusive with MAMBA2_PATTERN
# on any given layer -- main() rejects --rwkv together with --mamba2.
RWKV_HEAD_DIM = 128  # N, the per-head width. n_embd / RWKV_HEAD_DIM heads (5 at the locked
# n_embd=640), matching the reference's head_size = n_embd // n_head. The carried state is
# [B,H,N,N] and the r @ state matmul costs T*N*C MACs, so halving N halves that term.
RWKV_CHUNK = 128  # Q, the chunk length. Must divide sequence_len. The reference warns against
# going below 128 in bf16 (w^(Q-1) loses accuracy); 128 and 256 both divide the locked T=2048.
# NOT shape-critical -- it only changes how the same math is blocked.
RWKV_CMIX_EXPAND = 3  # channel-mix hidden width = this x n_embd (the x051a reference default is
# 3, vs this repo's MLP at 4). With 3, an RWKV layer's parameter count (5*C^2 time-mix +
# 7*C^2 channel-mix = 12*C^2) exactly matches the transformer layer it replaces (4*C^2 attention
# + 8*C^2 MLP = 12*C^2), which makes the A/B a clean mixer-vs-mixer comparison.
RWKV_LN_X_EPS = 1e-5 * 64.0  # eps for the time-mix's per-head GroupNorm. The reference uses
# eps=(1e-5)*64 (RWKV's head_size_divisor**2 = 8**2 convention), which is much larger than a
# normal LayerNorm eps and is load-bearing for RWKV's scale, so it is kept verbatim.


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


def init_runtime(device_type: str, seed: int = 42):
    assert device_type in {"cpu", "cuda", "neuron"}
    torch.manual_seed(seed)
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
    # T0.3: MLP hidden = mlp_ratio * n_embd. Defaulted (and appended last) so that
    # GPTConfig(**payload["model_config"]) still loads every checkpoint written before this
    # field existed -- those were all ratio 4, which is exactly this default.
    mlp_ratio: int = 4
    # T0.3: SDPA softmax scale; 0.0 => SDPA's own 1/sqrt(head_dim). Also carried in the config
    # (same reasoning as mlp_ratio) because it changes what the trained weights MEAN without
    # changing any shape, so a checkpoint evaluated with the wrong value would fail silently.
    attn_scale: float = 0.0


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

    # ---------------- Muon elementwise fusion (MuonAdamW._muon_step) ----------------
    # The optimizer step on this backend is dispatch-gap bound: ~54% of the optimizer
    # window is pure inter-launch gap, and a census of the real op stream
    # (research/scratch/muon_dispatch_census.py) shows Muon dispatches 320 real
    # launches per step of which 260 (81.2%) are elementwise/reduce ops wrapped
    # around 60 matmuls. The matmuls are exactly what the Tensor Engine is for and
    # are deliberately left in aten; these kernels fuse ONLY the elementwise
    # wrappers, which is where the launch count actually lives.
    #
    # Two design rules make this safe:
    #
    # 1. Every scalar (the Polar-Express a/b/c, lr, lr*wd) arrives as a runtime
    #    [128,1] fp32 SBUF operand, NEVER as a compile-time constant. nki.jit
    #    specialises its compile cache on shape+dtype only, so runtime operands mean
    #    one compile per shape for the whole run; a baked-in coefficient or lr would
    #    force a fresh neuronx-cc compile every time the value changed -- the exact
    #    trap USE_TENSOR_LR_SCALARS exists to avoid.
    #
    # 2. Rounding is NOT fused tighter than the eager chain. Every eager intermediate
    #    keeps its own explicitly-typed dst here, so the fp32->bf16 rounding points
    #    land in the same places. nisa.tensor_scalar/tensor_tensor compute in fp32
    #    internally and cast to dst.dtype for free, so e.g. `b*xtx + c*xtx2` is three
    #    instructions rather than one fused expression on purpose: collapsing it would
    #    skip the bf16 rounding of `b*xtx` that eager performs. This is what makes the
    #    fusion bit-exact rather than merely close.
    #
    # Instruction count inside a kernel is free; only launch count costs. Verified
    # bit-identical to the eager chain by research/scratch/muon_nki_equiv.py.

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
            # separate bf16 dsts: eager rounds b*t1 and c*t2 to bf16 before adding.
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
            nisa.tensor_tensor(dst=s1, data1=gb, data2=wb, op=nl.multiply)     # g*w
            s2 = nl.ndarray((size, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=s2, data=s1, op0=nl.greater_equal, operand0=0.0)
            # bf16, NOT fp32, and this is load-bearing: eager's `lr * g` has a 0-dim
            # fp32 `lr` against a dimensioned bf16 `g`, and PyTorch treats a 0-dim
            # tensor as a scalar of its category, so the dimensioned bf16 dtype wins the
            # promotion and the product is rounded to bf16. Computing it in fp32 here
            # instead is a ~2e-5 difference in every updated weight -- the one thing the
            # equivalence harness caught.
            s3 = nl.ndarray((size, N), dtype=g.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=s3, data=gb, op0=nl.multiply, operand0=a_lr)  # lr*g
            s4 = nl.ndarray((size, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=s4, data=wb, op0=nl.multiply, operand0=a_lw)  # lr*wd*w
            s5 = nl.ndarray((size, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=s5, data1=s4, data2=s2, op=nl.multiply)       # *mask
            # s1/s2 are dead from here, so reuse them rather than growing the SBUF
            # footprint (N is up to 2560 fp32 = 10 KB per partition per tile).
            nisa.tensor_tensor(dst=s1, data1=s3, data2=s5, op=nl.add)
            nisa.tensor_tensor(dst=s2, data1=wb, data2=s1, op=nl.subtract)
            nisa.dma_copy(dst=out[start:start + size, 0:N], src=s2)
        return out

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
#
# END-TO-END, at the real 4-rank/lnc=2 locked shape, steady-state dt (2x3 isolate,
# research/run-logs/l2iso-*.log):
#     Mamba2 eager   + this kernel  1.53 s/step   <- best Mamba2 configuration
#     Mamba2 eager   + eager scan   2.00
#     Mamba2 compile + eager scan   2.20
#     Mamba2 compile + this kernel  4.71          <- see the torch.compile warning
#     transformer compile 0.92 | transformer eager 0.97
# So the kernel is a real 1.31x end-to-end win IN EAGER and a real 2.14x
# end-to-end LOSS under torch.compile: `.apply` on a torch.autograd.Function is
# opaque to Dynamo, so the compiled graph splits around all five mixers' kernel
# calls and the fragments compile far worse than the unsplit graph does. Use
# --no-compile whenever --mamba2 --nki-ssd are on.
USE_NKI_SSD = True
# Toggled by the --nki-muon flag in main(). Fuses the elementwise ops that wrap
# Muon's Newton-Schulz matmuls (the matmuls themselves stay in aten) plus the whole
# sign-gated weight update, via _muon_ns_poly/_muon_ns_axpy/_muon_update.
#
# Sized BEFORE it was written (research/scratch/muon_fusion_sizing.py, a launch-count
# emulation using cheap aten ops, which gives a genuine upper bound because its data
# traffic is <= a real kernel's): the "652 Muon dispatches" in the USE_FOREACH_OPTIM
# note above is not a launch count -- a TorchDispatchMode census
# (research/scratch/muon_dispatch_census.py) reproduces 652 exactly as 15 matmul + 65
# elementwise + 83 pure VIEW ops per group x4 groups, and views cost nothing. The real
# figure is 320 device launches, 260 of them (81.2%) elementwise. Collapsing every
# elementwise region would take Muon's window from 50.09ms to 19.71ms at the real
# 4-rank/LNC=2 shape; this flag implements the two regions of that plan where exactness
# is achievable, removing 92 of the 320.
#
# The three regions left in eager on purpose (`scalars` 5.59ms, `momnorm` 3.82ms,
# `vmean` 0.93ms of that upper bound) contain ATen reductions -- linalg_vector_norm,
# mean, sum -- whose accumulation order is not reproducible in NKI. Fusing them would
# trade the exact-match standard the rest of this file's kernels are held to for ~1ms
# each. Recorded as a decision, not an omission.
#
# MEASURED, all four checks:
# (1) EQUIVALENCE -- research/scratch/muon_nki_equiv.py runs 5 steps of fused-vs-eager on
#     identically-initialised parameter sets with identical synthetic grads and a
#     per-step-varying lr/momentum/weight_decay schedule, comparing every Muon param AND
#     every optimizer state tensor element by element on all 4 real group shapes:
#     0 differing elements, max|diff| 0.000e+00. BIT-EXACT, which is the right standard
#     here (a lossless re-expression of existing elementwise ops, like the rope+QK-norm
#     and softcap+CE kernels -- not the FP8 kernel's inherent-precision-loss standard).
# (2) LAUNCH COUNT + (3) REAL TIMING (research/scratch/muon_nki_bench.py, 4 ranks LNC=2,
#     arms interleaved across 3 repeats so a slow patch of the box cannot land on one
#     arm): device launches 316 -> 224 (180 aten + 44 NKI), -92 = -29.1%; Muon window
#     52.17ms -> 38.86ms, -13.32ms = -25.5%, ranges non-overlapping (49.84-53.45 vs
#     37.57-43.71), 0.145ms per avoided launch -- inside the sizing script's independently
#     measured 0.100-0.206 band. Counting launches naively says only "24 removed": NKI
#     kernels never reach the aten dispatcher but their output allocation does
#     (aten.empty x44), and the host-side scalar staging adds CPU-only ops.
#     Smoke test single-device 15 steps: causality_passed True, causality_max_abs 0.0.
# (4) END-TO-END, paired same-day 900s A/B at the same commit and the locked shape:
#     ON 1019 steps / 1.089758 bpb vs OFF 1004 / 1.090677 -- +15 steps (+1.49%) and
#     -0.000919 bpb, with the OFF arm landing squarely in the historical baseline cluster
#     (gradclip1-900s 999/1.0912, -v2 1002/1.090864) so the box was in a normal state.
#     That is ~1.5% of step time and about half the USE_FOREACH_OPTIM win -- an honestly
#     modest number, and it matched the pre-run prediction from (3) almost exactly
#     (+15 steps, -0.0011 bpb predicted). Because the math is bit-exact the quality
#     question collapses to pure step count: the trajectory is identical to the last bit,
#     so more steps in the same wall clock cannot be anything but better.
# (5) CONFIRMED at the real full 1800s budget, also paired same-day and same-commit:
#     ON 2043 steps / 1.041139 bpb vs OFF 2011 / 1.042049 -- +32 steps (+1.59%) and
#     -0.000910 bpb, a clean reproduction of the 900s proxy (the effect is pure step
#     count, so it scales with the budget rather than washing out). This is a NEW BEST
#     confirmed number; the previous best on record, gradclip1-fullbudget 2000/1.042278,
#     is itself reconfirmed by the OFF control at 2011/1.042049.
# See rows nkimuon-900s / nkimuon-off-900s / nkimuon-fullbudget / nkimuon-off-fullbudget
# in research/experiment-log.jsonl and the "Muon's elementwise wrapper ops in NKI"
# section of research/trn2-experiment-results.md.
USE_NKI_MUON = True


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
    """relu(x - RELU2_TAU)**2, via the example NKI kernel when enabled, else eager.

    T3.9: RELU2_TAU is Recursive's "shifted ReLU^2" (tau=0.5 in their recipe, 0.0 = the
    historical activation). The shift is a compile-time Python float folded into one subtraction,
    NOT a per-step-changing scalar, so it is baked into the graph exactly once -- the rule about
    new scalars needing to be device tensors applies to values that CHANGE per step.
    """
    # The example kernel tiles rows in blocks of 128, so it needs a row count that
    # is a multiple of 128 (true for the baseline: B*T = 1*1024). Otherwise fall
    # back to eager so the knob is always safe to flip.
    if USE_NKI_RELU2 and _HAS_NKI and x.device.type == "neuron" and (x.numel() // x.shape[-1]) % _NKI_PARTITION == 0:
        # The kernel implements the UNSHIFTED form only; main() refuses the combination.
        return _ReluSquaredNKI.apply(x)
    if RELU2_TAU != 0.0:
        return F.relu(x - RELU2_TAU).square()
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


_FP8_E4M3_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max, verified on this SDK (torch 2.11.0)
_FP8_AMAX_EPS = 1e-12  # floor on the amax denominator: an all-zero tensor must not give inf scale


def _fp8_current_scale(t: torch.Tensor) -> torch.Tensor:
    """Current-scaling factor (Transformer Engine's Float8CurrentScaling): map this
    tensor's OWN amax onto the top of the E4M3 range. No amax history, no delay --
    that is the whole point of the design (see the USE_FP8_LINEAR comment). Returned
    as a 0-dim device tensor so nothing forces a host sync mid-forward.
    """
    return _FP8_E4M3_MAX / t.detach().abs().amax().clamp_min(_FP8_AMAX_EPS)


def _fp8_can_run(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """fp8 GEMM is only attempted where it is known real; everything else falls back to
    the ordinary bf16 F.linear, so the toggle is always safe to flip.

    torch._scaled_mm is the ONLY numerically correct fp8 matmul on this stack --
    torch.matmul on fp8 operands returns an all-NaN fp8 result without raising (measured;
    see USE_FP8_LINEAR). It is 2-D only, hence the reshape in _fp8_linear_forward.
    """
    return (
        hasattr(torch, "_scaled_mm")
        and hasattr(torch, "float8_e4m3fn")
        and x.device.type == "neuron"
        and weight.device.type == "neuron"
        and x.ndim >= 2
        and x.shape[-1] == weight.shape[1]
    )


def _fp8_linear_forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """One forward-only-fp8 matmul: cast both operands with their own current-scale,
    GEMM in fp8 with fp32 accumulation, and let _scaled_mm undo both scales via its
    scale_a/scale_b arguments (so the rescale costs nothing extra).
    """
    x_2d = x.reshape(-1, x.shape[-1])
    x_scale = _fp8_current_scale(x_2d)
    w_scale = _fp8_current_scale(weight)
    x_fp8 = (x_2d.float() * x_scale).to(torch.float8_e4m3fn)
    w_fp8 = (weight.float() * w_scale).to(torch.float8_e4m3fn)
    y = torch._scaled_mm(
        x_fp8,
        w_fp8.t().contiguous(),  # _scaled_mm wants the [K, N] orientation
        scale_a=1.0 / x_scale,
        scale_b=1.0 / w_scale,
        out_dtype=x.dtype,
    )
    return y.reshape(*x.shape[:-1], weight.shape[0])


class _Fp8LinearFn(torch.autograd.Function):
    """Forward in fp8, backward in the input dtype (bf16) against the SAVED
    full-precision operands.

    This asymmetry is the deliberate de-risking: only the forward matmul ever sees an
    fp8 rounding, and dgrad/wgrad are computed from the original x and weight, so no
    gradient is ever quantized. For a single isolated matmul that makes the backward
    numerically EXACT versus an ordinary bf16 Linear (verified, 0.0 relative error);
    through a whole MLP it is near-exact rather than exact, because c_proj's wgrad is
    taken against the relu^2 of an activation that the fp8 forward did perturb.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        return _fp8_linear_forward(x, weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_x = grad_out @ weight  # [..., O] @ [O, I] -> [..., I]
        grad_w = grad_out.reshape(-1, grad_out.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
        return grad_x, grad_w.to(weight.dtype)


class Fp8Linear(Linear):
    """Linear whose forward matmul runs in fp8 when USE_FP8_LINEAR is on.

    Deliberately a subclass of Linear, not a parallel implementation: it inherits the
    exact same single `weight` parameter (so the state_dict is byte-for-byte the same
    shape either way -- which is why USE_FP8_LINEAR is NOT shape-critical) and the same
    USE_WEIGHT_CACHE / _cached_weight refresh pattern. The cached tensor is the one fed
    to autograd, exactly as in Linear.forward, so grads still flow back through the
    once-per-optimizer-step dtype cast to self.weight.

    With USE_FP8_LINEAR off this is bit-identical to Linear -- it takes the same
    super().forward(x) path, no fp8 code runs at all.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not USE_FP8_LINEAR:
            return super().forward(x)
        if USE_WEIGHT_CACHE and self._cached_weight is not None and self._cached_weight.dtype == x.dtype:
            weight = self._cached_weight
        else:
            weight = self.weight.to(dtype=x.dtype)
        if not _fp8_can_run(x, weight):
            return F.linear(x, weight, None)
        return _Fp8LinearFn.apply(x, weight)


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

    The fallback is ANNOUNCED, once, loudly. It used to be silent, and that cost a
    real measurement: `m2hybrid-nki-fixed300` / `m2hybrid-nki-fullbudget` both
    passed `--nki-ssd`, both printed `nki_ssd: True` (which only echoes the flag,
    not the guard), and both ran at dt 1.98-2.00 s/step -- exactly the measured
    `--no-nki-ssd` speed (2.00) and nowhere near the kernel's real 1.53. A
    per-rank one-shot print of which branch was actually taken makes that class
    of mistake impossible to repeat.
    """
    if USE_NKI_SSD and _HAS_NKI and x.device.type == "neuron" and x.ndim == 4:
        b, T, H, P = x.shape
        G, N = B.shape[2], B.shape[3]
        if N <= Q <= _NKI_PARTITION and P <= 512 and T % Q == 0 and H % G == 0 \
                and x.dtype == torch.float32 and dt.dtype == torch.float32 \
                and B.dtype == torch.float32 and C.dtype == torch.float32 \
                and x.is_contiguous() and dt.is_contiguous() \
                and B.is_contiguous() and C.is_contiguous():
            _announce_ssd_path("NKI kernel", x, dt, B, C, Q)
            return _SSDNKI.apply(x, dt, A, B, C, D, Q)
    _announce_ssd_path("eager ssd_chunked", x, dt, B, C, Q)
    return ssd_chunked(x, dt, A, B, C, D, Q, masks)


_SSD_PATH_ANNOUNCED = False


def _announce_ssd_path(which: str, x, dt, B, C, Q: int) -> None:
    """Print, once per process, which SSD branch the shape guard actually chose."""
    global _SSD_PATH_ANNOUNCED
    if _SSD_PATH_ANNOUNCED:
        return
    _SSD_PATH_ANNOUNCED = True
    b, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    detail = (f"b={b} T={T} H={H} P={P} G={G} N={N} Q={Q} dev={x.device.type} "
              f"dtypes=({x.dtype},{dt.dtype},{B.dtype},{C.dtype}) "
              f"contig=({x.is_contiguous()},{dt.is_contiguous()},"
              f"{B.is_contiguous()},{C.is_contiguous()})")
    print0(f"ssd_scan path: {which}  [USE_NKI_SSD={USE_NKI_SSD} _HAS_NKI={_HAS_NKI}]  {detail}")


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


# -----------------------------------------------------------------------------
# RWKV "x051a" (BlinkDL). Only reached when USE_RWKV is on. See the USE_RWKV comment.


def time_shift(x: torch.Tensor) -> torch.Tensor:
    """x[..., t-1, :], with position 0 reading zeros -- RWKV's 1-step token shift.

    The reference uses `nn.ZeroPad2d((0, 0, 1, -1))`, which on a [B,T,C] tensor pads
    the T axis with one row of zeros at the front and crops one off the back. Written
    here as an explicit slice + left F.pad instead, which is (a) exactly the same
    tensor and (b) the pad/slice pattern this repo has already established as working
    on the native-eager Neuron backend (see ssd_causal_conv1d, which deliberately
    avoids F.conv1d for the same reason).

    Causal by construction: the pad is on the LEFT, so position t reads only t-1.
    """
    return F.pad(x[:, :-1, :], (0, 0, 1, 0))


def _rwkv_masks(Q: int, NC: int, device, dtype):
    """The five constants the loop-free chunked RWKV operator needs, per (Q, NC).

    Every one is an integer exponent grid or an additive causal mask; the actual
    decay values are exp(log_w * grid + mask), computed per forward so that
    time_decay/time_faaaa stay differentiable.

    e_intra [Q,Q]   max(t-s-1, 0) -- the within-chunk decay exponent. CLAMPED at 0 so
                    that the masked (s >= t) region can never produce a POSITIVE
                    exponent, hence never a +inf that the mask would turn into NaN.
    m_intra [Q,Q]   0 where s < t, _SSD_NEG otherwise. Additive, applied BEFORE exp,
                    so masked entries are exactly exp(-30000) == 0.0.
    eye_q   [Q,Q]   identity, carrying the u = time_faaaa "current token" bonus.
    e_wk    [Q,1]   Q-1-j -- the decay applied to key position j when it is folded
                    into the chunk's outgoing state.
    e_wb    [Q,1]   t -- the decay applied to the incoming state at chunk offset t.
    e_l2    [NC,NC] Q * max(c-c'-1, 0), the same thing one level up: chunk c's
                    incoming state decays chunk c''s contribution by w^(Q*(c-c'-1)).
    m_l2    [NC,NC] 0 where c' < c, _SSD_NEG otherwise -- STRICTLY lower triangular,
                    because chunk c may only see chunks that finished before it.
    """
    qi = torch.arange(Q, device=device)
    ci = torch.arange(NC, device=device)
    zero = torch.zeros((), device=device, dtype=dtype)
    neg = torch.full((), _SSD_NEG, device=device, dtype=dtype)
    tril_q = qi[:, None] > qi[None, :]                      # strictly lower: s < t
    tril_c = ci[:, None] > ci[None, :]
    return (
        (qi[:, None] - qi[None, :] - 1).clamp(min=0).to(dtype),      # e_intra
        torch.where(tril_q, zero, neg),                              # m_intra
        torch.eye(Q, device=device, dtype=dtype),                    # eye_q
        (Q - 1 - qi).reshape(Q, 1).to(dtype),                        # e_wk
        qi.reshape(Q, 1).to(dtype),                                  # e_wb
        (Q * (ci[:, None] - ci[None, :] - 1).clamp(min=0)).to(dtype),  # e_l2
        torch.where(tril_c, zero, neg),                              # m_l2
    )


def rwkv_chunked(r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 log_w: torch.Tensor, u: torch.Tensor, Q: int, masks) -> torch.Tensor:
    """Loop-free chunked RWKV-x051a time mixing.

    r, k, v: [b,T,H,N]   log_w: [H,1] (<= 0, == -exp(time_decay))   u: [H,1]
    -> y: [b,T,H,N]

    Computes, per head:  y_t = u*(r_t.k_t)*v_t + sum_{s<t} w^(t-s-1) * (r_t.k_s) * v_s
    with w = exp(log_w) in (0,1).

    NO python loop over chunks, unlike the reference. Within a chunk the operator is
    an ordinary masked quadratic form (r @ k^T) * D @ v; across chunks the carried
    [N,N] state obeys S_{c+1} = w^Q * S_c + G_c, whose decay is a per-head SCALAR, so
    the whole carry collapses to one more masked-decay matmul at chunk granularity:
        S_in[c] = sum_{c'<c} w^(Q*(c-c'-1)) * G_{c'}
    Two levels of the same trick, zero sequential steps -- the same structure
    ssd_chunked uses for Mamba2, and for the same dispatch-cost reason.

    Deliberately dtype-transparent: the decay tensors are built in log_w's dtype and
    cast to r's, so the caller decides precision. RWKVTimeMix passes fp32 log_w with
    bf16 r/k/v (matching the reference, which computes w in float and casts to r.dtype);
    the offline harness passes float64 throughout and gets agreement with an
    independently written sequential recurrence to float64 round-off, which only works
    because nothing here silently downcasts.

    Every exponent is >= 0 by construction (the grids are clamped) and log_w <= 0, so
    every live exponent is <= 0 and exp() is in (0,1]; the causal masks are ADDITIVE
    _SSD_NEG applied BEFORE exp, so masked entries are exactly 0.0 and no +inf * 0
    can arise.
    """
    b, T, H, N = r.shape
    NC = T // Q
    e_intra, m_intra, eye_q, e_wk, e_wb, e_l2, m_l2 = masks
    lw5 = log_w.reshape(1, 1, H, 1, 1)
    # decay tensors, in log_w's dtype (fp32 in training), then cast down to r's dtype
    d_intra = (torch.exp(lw5 * e_intra + m_intra) + u.reshape(1, 1, H, 1, 1) * eye_q).to(r.dtype)
    d_wk = torch.exp(lw5 * e_wk).to(r.dtype)                # [1,1,H,Q,1]
    d_wb = torch.exp(lw5 * e_wb).to(r.dtype)                # [1,1,H,Q,1]
    d_l2 = torch.exp(log_w.reshape(1, H, 1, 1) * e_l2 + m_l2).to(r.dtype)   # [1,H,NC,NC]
    # -> [b, NC, H, Q, N]: chunk and head become batch dims of the matmuls
    rc = r.reshape(b, NC, Q, H, N).permute(0, 1, 3, 2, 4)
    kc = k.reshape(b, NC, Q, H, N).permute(0, 1, 3, 2, 4)
    vc = v.reshape(b, NC, Q, H, N).permute(0, 1, 3, 2, 4)
    # intra-chunk: the masked quadratic form
    y = torch.matmul(torch.matmul(rc, kc.transpose(-1, -2)) * d_intra, vc)
    # each chunk's own outgoing contribution G_c = sum_j w^(Q-1-j) k_j (x) v_j
    Gst = torch.matmul((kc * d_wk).transpose(-1, -2), vc)             # [b,NC,H,N,N]
    # inter-chunk carry in closed form (no sequential step)
    S_in = torch.matmul(d_l2, Gst.permute(0, 2, 1, 3, 4).reshape(b, H, NC, N * N))
    S_in = S_in.reshape(b, H, NC, N, N).permute(0, 2, 1, 3, 4)
    y = y + torch.matmul(rc, S_in) * d_wb
    return y.permute(0, 1, 3, 2, 4).reshape(b, T, H, N)


def rwkv_group_norm(y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                    eps: float = RWKV_LN_X_EPS) -> torch.Tensor:
    """The time-mix's `ln_x`: nn.GroupNorm(n_head, n_embd) over the head axis.

    y is [b,T,H,N]; GroupNorm(H, H*N) applied to a [b*T, H*N] row normalises each
    group of N channels independently and then applies a PER-CHANNEL affine -- which
    on the [b,T,H,N] view is exactly a mean/var normalisation over the last axis
    followed by a [H*N] scale/shift. Written out rather than calling nn.GroupNorm so
    the op set stays inside the pad/slice/reduce/mul primitives this repo has already
    validated on the native-eager Neuron backend (checked against nn.GroupNorm itself
    in research/scratch/rwkv_train_integration_check.py).

    Runs in fp32: this is the only normalisation inside the mixer and the eps is huge
    (6.4e-4) by design, so bf16 variance accumulation over N=128 channels is the one
    place a real precision loss could hide.
    """
    b, t, h, n = y.shape
    yf = y.float()
    mu = yf.mean(dim=-1, keepdim=True)
    centered = yf - mu
    var = centered.pow(2).mean(dim=-1, keepdim=True)
    out = centered * torch.rsqrt(var + eps)
    return out.reshape(b, t, h * n) * weight.float() + bias.float()


class RWKVTimeMix(nn.Module):
    """RWKV-x051a time mixing, used in place of CausalSelfAttention on "R" layers.

    Faithful to nanoRWKV's RWKV_TimeMix_x051a with four deliberate, documented
    deviations, none of which changes the operator:

      1. The chunk loop is replaced by the loop-free form (see rwkv_chunked).
      2. The decay-matrix construction is replaced by exp(log_w * grid + mask)
         (see rwkv_chunked) -- same values, no tile/flip/view gymnastics.
      3. `nn.ZeroPad2d` -> `time_shift` (slice + left pad) and `nn.GroupNorm` ->
         `rwkv_group_norm`, both bit-equivalent, both to stay on primitives this
         repo has already exercised on Neuron.
      4. The projections are this repo's `Linear` (bias-free, dtype-casting,
         weight-cache aware) instead of nn.Linear with config.bias, and the value
         embedding (USE_VE) is injected into the `v` stream exactly as
         CausalSelfAttention injects it -- x051a genuinely has a per-token value
         stream, so unlike a Mamba2 layer this one CAN carry a value embedding, and
         not carrying it would handicap the A/B on an axis unrelated to the mixer.

    Everything RWKV-specific (time_maa_*, time_decay, time_faaaa, ln_x) is
    initialised by reset_rwkv_parameters from the reference's own layer-dependent
    schedule; GPT.init_weights only touches the five ordinary projection matrices.
    """

    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        n_embd = config.n_embd
        self.n_embd = n_embd
        self.head_dim = RWKV_HEAD_DIM
        assert n_embd % self.head_dim == 0, (n_embd, self.head_dim)
        self.n_head = n_embd // self.head_dim
        self.chunk = RWKV_CHUNK
        self.layer_idx = layer_idx
        self.n_layer = config.n_layer
        self.receptance = Linear(n_embd, n_embd, bias=False)
        self.key = Linear(n_embd, n_embd, bias=False)
        self.value = Linear(n_embd, n_embd, bias=False)
        self.gate = Linear(n_embd, n_embd, bias=False)
        self.output = Linear(n_embd, n_embd, bias=False)
        self.time_maa_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_v = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_r = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_g = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_decay = nn.Parameter(torch.zeros(self.n_head, 1))
        self.time_faaaa = nn.Parameter(torch.zeros(self.n_head, 1))
        self.ln_x_weight = nn.Parameter(torch.ones(n_embd))
        self.ln_x_bias = nn.Parameter(torch.zeros(n_embd))
        self.ve_gate = Linear(VE_GATE_CHANNELS, self.n_head, bias=False) \
            if USE_VE and has_ve(layer_idx, config.n_layer) else None
        self._masks = None
        self.reset_rwkv_parameters()

    @torch.no_grad()
    def reset_rwkv_parameters(self) -> None:
        """The reference's layer-dependent init for the RWKV-specific tensors.

        Verbatim from RWKV_TimeMix_x051a.__init__ except for the H == 1 guard (the
        reference divides by n_head - 1 and would crash on a single-head config).
        """
        C, H, L, i = self.n_embd, self.n_head, self.n_layer, self.layer_idx
        ratio_0_to_1 = i / max(1, L - 1)
        ratio_1_to_almost0 = 1.0 - (i / L)
        ddd = (torch.arange(C, dtype=torch.float32) / C).reshape(1, 1, C)
        self.time_maa_k.copy_(1.0 - torch.pow(ddd, ratio_1_to_almost0))
        self.time_maa_v.copy_(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
        self.time_maa_r.copy_(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
        self.time_maa_g.copy_(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
        hh = torch.arange(H, dtype=torch.float32) / max(1, H - 1)
        # decay_speed = -6 + 5 * (h/(H-1))^(0.7 + 1.3*ratio_0_to_1): every head gets a
        # different timescale, from w = exp(-exp(-6)) ~ 0.9975 to w = exp(-exp(-1)) ~ 0.69.
        self.time_decay.copy_((-6.0 + 5.0 * hh.pow(0.7 + 1.3 * ratio_0_to_1)).reshape(H, 1))
        self.time_faaaa.copy_((ratio_0_to_1 * (1.0 - hh)).reshape(H, 1))
        self.ln_x_weight.fill_(1.0)
        self.ln_x_bias.zero_()

    def _get_masks(self, T: int, device, dtype):
        """Cached per (T, device, dtype). Built on CPU, then moved once.

        Deliberately NOT built on the device: every one of these is a pure constant
        (arange / compare / clamp / where / eye), and on the native-eager Neuron
        backend each distinct op+shape is its own HLO compile. Building them on the
        device cost ~8 extra first-call compilations for tensors whose values never
        change, so they are built on CPU and moved once per (T, device, dtype).
        """
        key = (T, str(device), dtype)
        if self._masks is None or self._masks[0] != key:
            cpu = _rwkv_masks(self.chunk, T // self.chunk, torch.device("cpu"), dtype)
            self._masks = (key, tuple(m.to(device) for m in cpu))
        return self._masks[1]

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None = None) -> torch.Tensor:
        """x: [b, T, n_embd] -> [b, T, n_embd]"""
        b, T, C = x.shape
        H, N = self.n_head, self.head_dim
        assert T % self.chunk == 0, (T, self.chunk)
        # data-dependent shift mixing: every stream sees its own blend of x[t] and x[t-1]
        xx = time_shift(x) - x
        xr = x + xx * self.time_maa_r.to(x.dtype)
        xk = x + xx * self.time_maa_k.to(x.dtype)
        xv = x + xx * self.time_maa_v.to(x.dtype)
        xg = x + xx * self.time_maa_g.to(x.dtype)
        r = self.receptance(xr).view(b, T, H, N)
        k = self.key(xk).view(b, T, H, N)
        v = self.value(xv).view(b, T, H, N)
        if ve is not None:
            # Same input-gated value-embedding mix as CausalSelfAttention, on the same
            # stream (V) and with the same 2*sigmoid gate. Purely per-position.
            gate = 2 * torch.sigmoid(self.ve_gate(xv[..., :VE_GATE_CHANNELS]))
            v = v + gate.unsqueeze(-1) * ve.view(b, T, H, N)
        g = F.silu(self.gate(xg))
        log_w = -torch.exp(self.time_decay.float())          # [H,1] < 0
        u = self.time_faaaa.float()                          # [H,1]
        y = rwkv_chunked(r, k, v, log_w, u, self.chunk,
                         self._get_masks(T, x.device, torch.float32))
        y = rwkv_group_norm(y, self.ln_x_weight, self.ln_x_bias).to(x.dtype) * g
        return self.output(y)


class RWKVChannelMix(nn.Module):
    """RWKV-x051a channel mixing, used in place of MLP on "R" layers.

    A relu-squared gated MLP with the same 1-step shift mixing as the time mix:
        h = relu(key(x + (x[t-1]-x[t]) * maa_k))^2
        y = sigmoid(receptance(x + (x[t-1]-x[t]) * maa_r)) * value(h)
    Structurally very close to this repo's MLP (which is also relu-squared), with two
    differences: the sigmoid receptance gate, and the shift mixing. `relu_squared` is
    this repo's own helper, so the NKI relu2 kernel applies here too if enabled.
    """

    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        n_embd = config.n_embd
        self.n_embd = n_embd
        self.layer_idx = layer_idx
        self.n_layer = config.n_layer
        hidden = RWKV_CMIX_EXPAND * n_embd
        self.key = Linear(n_embd, hidden, bias=False)
        self.value = Linear(hidden, n_embd, bias=False)
        self.receptance = Linear(n_embd, n_embd, bias=False)
        self.time_maa_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_r = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.reset_rwkv_parameters()

    @torch.no_grad()
    def reset_rwkv_parameters(self) -> None:
        """Verbatim from RWKV_ChannelMix_x051a.__init__."""
        C, L, i = self.n_embd, self.n_layer, self.layer_idx
        ratio_1_to_almost0 = 1.0 - (i / L)
        ddd = (torch.arange(C, dtype=torch.float32) / C).reshape(1, 1, C)
        self.time_maa_k.copy_(1.0 - torch.pow(ddd, ratio_1_to_almost0))
        self.time_maa_r.copy_(1.0 - torch.pow(ddd, ratio_1_to_almost0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = time_shift(x) - x
        xk = x + xx * self.time_maa_k.to(x.dtype)
        xr = x + xx * self.time_maa_r.to(x.dtype)
        h = self.value(relu_squared(self.key(xk)))
        return torch.sigmoid(self.receptance(xr)) * h


def rwkv_layer_flags(n_layer: int) -> list[bool]:
    """Which layers are RWKV (time-mix + channel-mix), from RWKV_PATTERN cycled to n_layer."""
    if not USE_RWKV:
        return [False] * n_layer
    pattern = RWKV_PATTERN or "R"
    cycled = (pattern * (n_layer // len(pattern) + 1))[:n_layer]
    return [c == "R" for c in cycled]


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
    def __init__(self, config: GPTConfig, layer_idx: int = 0, coalesced: str = "none"):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim * self.n_head == self.n_embd
        # `coalesced` (parallel block only): which of this module's input projections live in the
        # OWNING BLOCK's single c_in matmul instead of here. "all" = q, k and v; "vfc" = v only
        # (q and k stay separate so their outputs are contiguous and the rope+QK-norm NKI kernel
        # still fires -- see PARALLEL_BLOCK_COALESCE). The replaced Linears are not built at all,
        # so no path ever carries a dead parameter.
        self.coalesced = coalesced
        if coalesced == "all":
            pass
        elif coalesced == "vfc":
            self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
            self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        elif USE_FUSED_QKV:
            self.c_qkv = Linear(config.n_embd, 3 * config.n_embd, bias=False)
        else:
            self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
            self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
            self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        # T0.3: None => SDPA's own 1/sqrt(head_dim). Snapshotted here (not read from the global
        # per forward) so it is a single compile-time constant in the graph, identical every step.
        self.attn_scale = config.attn_scale if config.attn_scale > 0.0 else None
        self.ve_gate = Linear(VE_GATE_CHANNELS, self.n_head, bias=False) if USE_VE and has_ve(layer_idx, config.n_layer) else None
        # T3.1: one extra per-head gate per hashed n-gram order, each reading a DISJOINT
        # 32-channel slice of the V-stream residual so the three value-embedding gates
        # never compete for the same channels (unigram 0:32, bigram 32:64, trigram 64:96).
        # Both are built RNG-neutrally (init_weights zeroes them anyway) so that flipping
        # USE_NGRAM_VE does not reshuffle every other tensor's init -- see _rng_neutral.
        self.ve_gate_bi = _rng_neutral(lambda: Linear(VE_GATE_CHANNELS, self.n_head, bias=False)) \
            if has_ngram_bigram(layer_idx, config.n_layer) else None
        self.ve_gate_tri = _rng_neutral(lambda: Linear(VE_GATE_CHANNELS, self.n_head, bias=False)) \
            if has_ngram_trigram(layer_idx, config.n_layer) else None
        if self.ve_gate_tri is not None:
            assert config.n_embd >= 3 * VE_GATE_CHANNELS, (config.n_embd, VE_GATE_CHANNELS)
        elif self.ve_gate_bi is not None:
            assert config.n_embd >= 2 * VE_GATE_CHANNELS, (config.n_embd, VE_GATE_CHANNELS)
        self.dcmha = DCMHACompose(config.n_embd, self.n_head, layer_idx) if USE_DCMHA else None
        # T3.6 / T3.9: per-head gate on the attention output, reading a residual slice that no
        # value-embedding gate has claimed. Built RNG-neutrally and re-initialized by
        # init_weights for exactly the reason the n-gram gates are (see _rng_neutral): flipping
        # USE_HEAD_GATE must not reshuffle every other tensor's init.
        # The bias is a separate nn.Parameter rather than Linear(bias=True) because this file's
        # Linear subclass is bias-free by design (it casts the weight to the input dtype), and
        # because a 1-D bias must be kept OUT of the Muon groups that block.attn's named
        # parameters otherwise flow into -- setup_optimizer routes it to the scalar group by name.
        if USE_HEAD_GATE:
            self.head_gate_lo, self.head_gate_hi = head_gate_slice(config.n_embd, config.n_layer)
            self.head_gate = _rng_neutral(
                lambda: Linear(HEAD_GATE_CHANNELS, self.n_head, bias=False))
            self.head_gate_bias = nn.Parameter(torch.full((self.n_head,), float(HEAD_GATE_BIAS)))
        else:
            self.head_gate_lo = self.head_gate_hi = 0
            self.head_gate = None

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None, v0: torch.Tensor | None = None, v0_lambda: torch.Tensor | None = None, ve_bi: torch.Tensor | None = None, ve_tri: torch.Tensor | None = None, qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c = x_q.shape
        if qkv is not None:
            # Parallel block with a coalesced input projection: the block already did the matmul
            # and handed over q/k/v as [b, t, n_head, head_dim], including whatever contiguity the
            # rope+QK-norm kernel needs. x_q/x_k/x_v are still the normed residual, which is what
            # the value-embedding and head gates below read. Under "vfc" only v is coalesced, so
            # q and k arrive as None and are projected here as usual -- that is the whole point of
            # that mode: their outputs stay contiguous for the NKI kernel.
            q, k, v = qkv
            if q is None:
                q = self.c_q(x_q).view(b, t, self.n_head, self.head_dim)
                k = self.c_k(x_k).view(b, t, self.n_head, self.head_dim)
        elif USE_FUSED_QKV:
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
        if ve_bi is not None:
            # T3.1. Purely per-position, like the unigram VE above: ve_bi[t] was gathered
            # with a hash of tokens (t-1, t) and the gate reads position t's own residual,
            # so nothing here can see a later position.
            g_bi = 2 * torch.sigmoid(self.ve_gate_bi(x_v[..., VE_GATE_CHANNELS:2 * VE_GATE_CHANNELS]))
            v = v + g_bi.unsqueeze(-1) * ve_bi.view(b, t, self.n_head, self.head_dim)
        if ve_tri is not None:
            g_tri = 2 * torch.sigmoid(self.ve_gate_tri(x_v[..., 2 * VE_GATE_CHANNELS:3 * VE_GATE_CHANNELS]))
            v = v + g_tri.unsqueeze(-1) * ve_tri.view(b, t, self.n_head, self.head_dim)
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
            scale = self.attn_scale if self.attn_scale is not None else 1.0 / math.sqrt(self.head_dim)
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            logits = _cross_head_compose(logits, *pre_args)
            keep = _dcmha_keep_mask(t, attn_mask, x_q.device)
            logits = logits.masked_fill(~keep, torch.finfo(torch.float32).min)
            probs = logits.softmax(dim=-1)
            probs = _cross_head_compose(probs, *post_args).to(v.dtype)
            y = torch.matmul(probs, v)
        elif attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[..., :t, :t], scale=self.attn_scale)
        if USE_XSA:
            # Exclusive Self Attention (arXiv:2603.09078, Zhai/Apple): project out the
            # component of the attention output along the token's OWN value vector --
            # z_i = y_i - (y_i . v_i / ||v_i||^2) * v_i. Causally safe by construction:
            # z_i depends only on y_i and v_i, both already functions of x_1..i only.
            # y, v are [B, H, T, D] here (post-transpose, pre-reshape).
            proj_coef = (y * v).sum(dim=-1, keepdim=True) / (v.pow(2).sum(dim=-1, keepdim=True) + _XSA_EPS)
            y = y - proj_coef * v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        if self.head_gate is not None:
            # T3.6 / T3.9. Per-head, per-POSITION: reshape to [b, t, H, D], optionally RMS-norm
            # each head's D-vector on its own, then scale head h at position t by
            # HEAD_GATE_MUL * sigmoid(w_h . x_q[b, t, lo:hi] + bias_h). Nothing crosses positions,
            # so causality is untouched (x_q is already position t's own normed residual).
            yh = y.view(b, t, self.n_head, self.head_dim)
            if HEAD_GATE_NORM:
                yh = norm(yh)
            g = self.head_gate(x_q[..., self.head_gate_lo:self.head_gate_hi])
            g = HEAD_GATE_MUL * torch.sigmoid(g + self.head_gate_bias.to(g.dtype))
            y = (yh * g.unsqueeze(-1)).view(b, t, c)
        return self.c_proj(y), v_raw


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, coalesced: str = "none"):
        super().__init__()
        # c_fc/c_proj are the two widest matmuls in the model, which is why they are the
        # ones USE_FP8_LINEAR targets (matching modded-nanogpt's own FP8-MLP records).
        # Fp8Linear is a Linear subclass with an identical parameter set, so picking the
        # class here leaves the state_dict unchanged and makes the OFF path literally the
        # pre-existing code. c_gate is left on plain Linear: USE_GATED_MLP is itself off
        # by default, so an fp8 variant of it would be an unmeasured combination.
        mlp_linear = Fp8Linear if USE_FP8_LINEAR else Linear
        hidden = config.mlp_ratio * config.n_embd
        # `coalesced` != "none" (parallel block only): c_fc lives in the owning block's single c_in
        # matmul, so it is not built here and forward takes the pre-activation as an argument.
        self.c_fc = None if coalesced in ("all", "vfc") else mlp_linear(config.n_embd, hidden, bias=False)
        self.c_gate = Linear(config.n_embd, hidden, bias=False) if USE_GATED_MLP else None
        self.c_proj = mlp_linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, fc: torch.Tensor | None = None) -> torch.Tensor:
        h = self.c_fc(x) if fc is None else fc
        if self.c_gate is not None:
            return self.c_proj(h * F.silu(self.c_gate(x)))
        return self.c_proj(relu_squared(h))


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
        # RWKV_PATTERN does the same for RWKV, except that an RWKV layer replaces BOTH
        # sublayers: time-mix instead of attention AND channel-mix instead of the MLP,
        # which is how the reference block is structured. Exactly one of
        # attn / mixer / tmix exists and exactly one of mlp / cmix exists, so no path
        # ever carries another's dead parameters.
        self.is_rwkv = rwkv_layer_flags(config.n_layer)[layer_idx]
        assert not (self.is_mamba and self.is_rwkv), "a layer cannot be both Mamba2 and RWKV"
        # Coalescing is decided PER BLOCK, not globally: a Mamba2/RWKV layer keeps its serial form
        # and its own MLP, so it must keep its own c_fc. See PARALLEL_BLOCK_COALESCE.
        self.coalesced = PARALLEL_BLOCK_COALESCE \
            if (USE_PARALLEL_BLOCK and not (self.is_mamba or self.is_rwkv)) else "none"
        self.attn = None if (self.is_mamba or self.is_rwkv) \
            else CausalSelfAttention(config, layer_idx, coalesced=self.coalesced)
        self.mixer = Mamba2Mixer(config) if self.is_mamba else None
        self.tmix = RWKVTimeMix(config, layer_idx) if self.is_rwkv else None
        self.cmix = RWKVChannelMix(config, layer_idx) if self.is_rwkv else None
        self.mlp = None if self.is_rwkv else MLP(config, coalesced=self.coalesced)
        # The coalesced input projection. Built HERE, immediately after the two modules whose
        # Linears it absorbs, and deliberately NOT via _rng_neutral: the absorbed Linears all have
        # in_features == n_embd and the same total numel as c_in, so this consumes exactly as many
        # RNG draws as the baseline's block does, leaving the stream position -- and therefore every
        # LATER tensor's initialization -- unchanged. init_weights then fills the slices in the
        # baseline's own order and with the baseline's own per-slice bounds, which makes the
        # coalesced arm bit-identical to the uncoalesced parallel block at step 0.
        if self.coalesced == "all":
            self.c_in = Linear(config.n_embd, 3 * config.n_embd + config.mlp_ratio * config.n_embd,
                               bias=False)
        elif self.coalesced == "vfc":
            self.c_in = Linear(config.n_embd, config.n_embd + config.mlp_ratio * config.n_embd,
                               bias=False)
        else:
            self.c_in = None
        self.mudd = MuddCompose(config.n_embd, layer_idx + 1, layer_idx) if USE_MUDD else None
        self.resid_lambda = nn.Parameter(torch.ones(1)) if self.mudd is None else None
        # x0 skip: mixes initial embedding into each layer's input (stabilizes deep nets)
        self.x0_lambda = nn.Parameter(torch.zeros(1)) if self.mudd is None else None
        # Recursive's input-dependent x0 gate: one scalar per layer, zero-init so
        # 2*sigmoid(0) == 1.0 makes it a bit-exact no-op against the static x0_lambda skip.
        self.x0_gate_s = nn.Parameter(torch.zeros(1)) if (USE_X0_GATE and self.mudd is None) else None
        # Frozen zero nudge: never trained, never in an optimizer group. See USE_BLOCK_NUDGE.
        self.block_nudge = nn.Parameter(torch.zeros(1), requires_grad=False) if USE_BLOCK_NUDGE else None
        # T-parallel: the two branch weights of the parallel attn+MLP block. Only built where
        # BOTH sublayers exist (a Mamba2 or RWKV layer keeps its serial form), so a mixed model
        # still works -- the branch in forward is guarded on par_lambda_a being present.
        # nn.Parameter(ones) draws no RNG, so building these cannot shift any other tensor's init.
        if USE_PARALLEL_BLOCK and self.attn is not None and self.mlp is not None:
            self.par_lambda_a = nn.Parameter(torch.ones(1))
            self.par_lambda_m = nn.Parameter(torch.ones(1))
        else:
            self.par_lambda_a = self.par_lambda_m = None
        self.local_conv = CausalConv(config.n_embd, LOCAL_CONV_KERNEL) if USE_LOCAL_CONV else None
        self.scalar_gate = Linear(SELECTIVE_GATE_CHANNELS, 2, bias=False) if USE_SELECTIVE_GATE else None
        # Value Residual Learning (arXiv:2410.17897): layer 0 has no v0 to consume yet (it
        # produces the v0 every later layer will use), so it doesn't need its own lambda.
        # A Mamba2 layer has no attention V to blend, so it gets no v_lambda either.
        self.v_lambda = nn.Parameter(torch.zeros(1)) \
            if USE_VALUE_RESIDUAL and layer_idx > 0 and not (self.is_mamba or self.is_rwkv) else None
        # T3.9: causal Q/K token shift (Recursive's from-vanilla recipe). Per-DIM learned beta, so
        # each residual channel decides for itself how much of the previous token to mix into the
        # query/key view; V is deliberately left unshifted. Only meaningful where there is an
        # attention module to feed.
        self.qk_shift_beta = nn.Parameter(torch.full((config.n_embd,), float(QK_SHIFT_BETA)),
                                          requires_grad=not QK_SHIFT_FREEZE) \
            if USE_QK_SHIFT and not (self.is_mamba or self.is_rwkv) else None

    def forward(self, x: torch.Tensor, x0: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None, v0: torch.Tensor | None = None, hidden_states: list[torch.Tensor] | None = None, ve_bi: torch.Tensor | None = None, ve_tri: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mudd is not None:
            q_in, k_in, v_in, x = self.mudd(hidden_states, hidden_states[-1])
        else:
            if self.scalar_gate is not None:
                g = self.scalar_gate(x[..., :SELECTIVE_GATE_CHANNELS])
                resid_lambda = self.resid_lambda * (1.0 + g[..., 0:1])
                x0_lambda = self.x0_lambda * (1.0 + g[..., 1:2])
                x = resid_lambda * x + x0_lambda * x0
            elif self.x0_gate_s is not None:
                # Recursive's input-dependent x0 gate. mean() is over the LAST axis only, i.e. each
                # token's own 640 residual channels -- no cross-position reduction, so causality is
                # structural here, not something the mask has to enforce. The grouping
                # `x0_lambda * (gate * x0)` is deliberate: at init gate is exactly 1.0, so this is
                # bit-identical to `x0_lambda * x0`, which is what makes the A/B start from the same
                # point. 2*sigmoid keeps the neutral value at 1.0 instead of 0.5, matching every
                # other gate in this file (ve_gate, head_gate).
                gate0 = 2.0 * torch.sigmoid(self.x0_gate_s * x.mean(dim=-1, keepdim=True))
                x = self.resid_lambda * x + self.x0_lambda * (gate0 * x0)
            else:
                x = self.resid_lambda * x + self.x0_lambda * x0
            if self.block_nudge is not None and BLOCK_NUDGE_SITE == "resid":
                x = x + self.block_nudge * x
            q_in = k_in = v_in = x
            if self.block_nudge is not None and BLOCK_NUDGE_SITE == "qk":
                # Exactly where --qk-shift put its perturbation (Q/K only, V untouched), minus the
                # causal shift and minus the trainable parameter. An identity; see USE_BLOCK_NUDGE.
                q_in = k_in = x + self.block_nudge * x
            if self.qk_shift_beta is not None:
                # T3.9: q/k see x + beta*x_prev, v sees plain x. causal_shift_1 zero-pads at t=0
                # (it does NOT wrap), so position t reads only positions t and t-1.
                q_in = k_in = x + self.qk_shift_beta * causal_shift_1(x)
        if self.local_conv is not None:
            x = x + self.local_conv(norm(x))
        if self.par_lambda_a is not None:
            # T-parallel: attention and the MLP both read the SAME normed residual and their
            # outputs are summed into it once, instead of the MLP reading the post-attention
            # residual. One norm() instead of four, one residual add instead of two, and the
            # attn/MLP matmuls become independent (they can be co-scheduled rather than
            # serialised through the attention output) -- which is the point on a stack that is
            # dispatch- and serialisation-bound.
            #
            # Bit-identical at step 0 despite being a real architecture change: mlp.c_proj is
            # zero-initialized, so mlp_out is exactly 0 at step 0 whichever input it reads, and
            # par_lambda_a starts at exactly 1.0. So the step-0 loss must match the baseline's,
            # which is what makes the A/B start from a common point (and makes a step-0
            # mismatch a real bug signal, not an expected difference).
            h = norm(x)
            qkv = fc = None
            if self.c_in is not None:
                b, t, c = h.shape
                nh, hd = self.attn.n_head, self.attn.head_dim
                proj = self.c_in(h)
                if self.coalesced == "all":
                    # A last-dim slice of proj is NOT contiguous, and rope_norm() silently falls
                    # back to eager on a non-contiguous input -- exactly the trap that made
                    # USE_FUSED_QKV a -0.0428 regression. So q and k are copied; the copy is two
                    # [b,t,640] elementwise kernels against a ~15-18% kernel speedup, which is the
                    # trade this mode exists to measure. v needs no copy: its last dim is still
                    # contiguous, so [b,t,640] -> [b,t,nh,hd] is a real view, and v never reaches
                    # the kernel.
                    q = proj[..., :c].contiguous().view(b, t, nh, hd)
                    k = proj[..., c:2 * c].contiguous().view(b, t, nh, hd)
                    v = proj[..., 2 * c:3 * c].view(b, t, nh, hd)
                    fc = proj[..., 3 * c:]
                else:  # "vfc": only v and c_fc are coalesced, so no copy is needed anywhere
                    q = k = None
                    v = proj[..., :c].view(b, t, nh, hd)
                    fc = proj[..., c:]
                qkv = (q, k, v)
            attn_out, v_raw = self.attn(h, h, h, ve, cos, sin, attn_mask, v0, self.v_lambda,
                                        ve_bi, ve_tri, qkv=qkv)
            x = x + self.par_lambda_a * attn_out + self.par_lambda_m * self.mlp(h, fc=fc)
            return x, v_raw
        if self.mixer is not None:
            # A Mamba2 layer produces no v_raw: there is no attention V here, so it
            # cannot seed the value residual. GPT.forward keeps v0 at None until the
            # first ATTENTION layer, which is the only correct thing to do anyway --
            # a Mamba2 layer's state is not a per-token value vector.
            x = x + self.mixer(norm(q_in))
            v_raw = None
        elif self.tmix is not None:
            # RWKV time mixing. No rope and no attn_mask: the operator is causal
            # structurally (1-step time_shift + a decay matrix that is exactly 0 above
            # the diagonal), and RWKV carries no positional encoding by design -- the
            # per-head decay IS the position signal. v_raw stays None: x051a's own value
            # stream is not what the value-residual toggle means by v0 (that is defined
            # as layer-0 ATTENTION's V), so no v0 is seeded from here.
            x = x + self.tmix(norm(q_in), ve)
            v_raw = None
        elif USE_QKV_NORM_CSE and q_in is k_in and k_in is v_in:
            # See USE_QKV_NORM_CSE. Guarded on object identity, not equality, so this branch can
            # only ever be taken when the three norm() calls it replaces had identical inputs.
            qkv_n = norm(q_in)
            attn_out, v_raw = self.attn(qkv_n, qkv_n, qkv_n, ve, cos, sin, attn_mask, v0, self.v_lambda, ve_bi, ve_tri)
            x = x + attn_out
        else:
            attn_out, v_raw = self.attn(norm(q_in), norm(k_in), norm(v_in), ve, cos, sin, attn_mask, v0, self.v_lambda, ve_bi, ve_tri)
            x = x + attn_out
        if self.cmix is not None:
            x = x + self.cmix(norm(x))
        elif USE_MLP_SANDWICH_NORM:
            # T3.9: sandwich norm on the MLP branch only (Recursive). norm() is parameter-free
            # F.rms_norm with the default eps, so the zero-init c_proj still gives exactly 0 here
            # at step 0 (0 / sqrt(0 + eps) = 0) -- the step-0 no-op survives, and the arm only
            # starts to differ once c_proj leaves the origin.
            x = x + norm(self.mlp(norm(x)))
        else:
            x = x + self.mlp(norm(x))
        return x, v_raw


class EngramLite(nn.Module):
    """T3.2 Engram-lite: hashed n-gram memory read through a hidden-state-dependent gate.

    See the USE_ENGRAM block for the full spec, the causality argument and the step-0 no-op
    argument. This class owns everything except the hash indices themselves, which are computed
    once per forward alongside T3.1's in GPT._ngram_hash_indices (the prev/prev2 shifts are
    shared), and except the gathered rows in share mode, which GPT.forward hands over.

    `share_e_dim` is the width of the e_t that GPT.forward will supply in share mode; it is None
    in own-table mode, where this module builds and gathers its own tables.
    """

    def __init__(self, config: GPTConfig, padded_vocab_size: int, share_e_dim: int | None = None):
        super().__init__()
        self.orders = engram_orders()
        self.heads = int(ENGRAM_HEADS)
        assert self.heads >= 1, ENGRAM_HEADS
        self.share = share_e_dim is not None
        self.mem_dim = int(ENGRAM_MEM_DIM)
        self.rows = int(ENGRAM_TABLE_MULT) * padded_vocab_size
        # Same argument as GPT.ngram_mod: `rows` is a power of two, and `h % 2^k` keeps only h's
        # low k bits, discarding every bit of mixing the primes put in the high half. A prime
        # modulus uses all of h; the few rows above it are simply never indexed.
        self.mod = _largest_prime_below(self.rows)
        dtype = torch.bfloat16 if ENGRAM_BF16 else torch.float32
        # Keys are stable strings so a state_dict makes the (heads, orders) allocation visible and
        # strict=True catches a toggle mismatch loudly.
        self.table_keys = tuple(f"e{h}o{o}" for o in self.orders for h in range(self.heads))
        if self.share:
            self.tables = nn.ModuleDict()
            self.e_dim = share_e_dim
        else:
            # `_weight=` for the same two reasons T3.1's _table() gives: nn.Embedding's own
            # reset_parameters would draw millions of normals per table from the global generator
            # only for init_weights to zero them, and that draw would shift the RNG stream for
            # every tensor initialized afterwards -- turning the A/B into a comparison of two
            # different random inits plus the feature.
            self.tables = nn.ModuleDict({
                k: nn.Embedding(self.rows, self.mem_dim,
                                _weight=torch.zeros(self.rows, self.mem_dim, dtype=dtype))
                for k in self.table_keys})
            self.e_dim = self.mem_dim * len(self.table_keys)
        self.key_dim = int(ENGRAM_KEY_DIM) if int(ENGRAM_KEY_DIM) > 0 else config.n_embd
        # _rng_neutral for the same reason the T3.1 gates and up-projections use it: init_weights
        # overwrites all three unconditionally, so their construction must not move the RNG stream.
        self.w_k = _rng_neutral(lambda: Linear(self.e_dim, self.key_dim, bias=False))
        self.w_v = _rng_neutral(lambda: Linear(self.e_dim, config.n_embd, bias=False))
        self.w_q = _rng_neutral(lambda: Linear(config.n_embd, self.key_dim, bias=False)) \
            if self.key_dim != config.n_embd else None
        self.conv = CausalConv(config.n_embd, int(ENGRAM_CONV_KERNEL)) \
            if int(ENGRAM_CONV_KERNEL) > 0 else None
        self._scale = self.key_dim ** -0.5

    def gather(self, ngram_idx: dict[str, torch.Tensor]) -> torch.Tensor:
        """e_t in own-table mode: concatenate this module's own table rows, heads-within-order."""
        parts = [self.tables[k].weight[ngram_idx[k]].to(COMPUTE_DTYPE) for k in self.table_keys]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Y for `H <- H + Y`. `h` is the residual at the injection site, `e` is e_t."""
        k = self.w_k(e)
        v = self.w_v(e)
        q = self.w_q(h) if self.w_q is not None else h
        # The gate. A per-token reduction over the key dimension only -- never across positions --
        # so causality is structural here, exactly as for the x0 gate and MuddCompose. There is
        # ONE such reduction in the whole model (the x0 gate's 13% cost came from paying one per
        # block, six times over, in a dispatch-bound graph).
        alpha = torch.sigmoid((norm(q) * norm(k)).sum(dim=-1, keepdim=True) * self._scale)
        vt = alpha * v
        if self.conv is None:
            return vt
        # CausalConv left-pads by kernel_size-1, so position t reads only t-K+1..t. Zero-init
        # weight => SiLU(0) == 0 => Y == vt until the conv has learned something.
        return F.silu(self.conv(norm(vt))) + vt


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
        # RWKV layers DO keep their value-embedding table: x051a has a real per-token
        # value stream (the `value` projection feeding the decay operator), so ve is
        # injected there exactly as it is into attention's V, with the same
        # zero-initialized ve_gate. That is deliberate -- it keeps a USE_RWKV A/B a
        # clean mixer-vs-mixer comparison instead of also silently deleting 5 of 6
        # value-embedding tables the way MAMBA2_PATTERN="MMAMMM" does.
        self.layer_is_rwkv = [blk.is_rwkv for blk in self.transformer.h]
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
        # T3.1: hashed n-gram VALUE embeddings. Keys are "b<layer>" / "t<layer>" so a
        # bigram-only and a bigram+trigram run have visibly different state_dict keys
        # (strict=True then catches a toggle mismatch loudly -- USE_NGRAM_VE and
        # NGRAM_VE_TRIGRAM are both in SHAPE_CRITICAL_TOGGLES for that reason).
        # A Mamba2 layer has no attention V, exactly as for the unigram tables above.
        self.ngram_rows = NGRAM_VE_TABLE_MULT * padded_vocab_size
        # The MODULUS is the largest prime <= rows, not `rows` itself. `rows` is a power of
        # two here, and `h % 2^k` keeps only h's low k bits, which throws away every bit of
        # mixing the primes put in the high half (modded-nanogpt uses `mod (R-1)` for the
        # same reason). A prime modulus uses all of h. The few rows above the modulus are
        # simply never indexed -- kept so the row count stays a multiple of 64.
        self.ngram_mod = _largest_prime_below(self.ngram_rows)
        ngram_dtype = torch.bfloat16 if NGRAM_VE_BF16 else torch.float32
        self.ngram_dim = NGRAM_VE_DIM if NGRAM_VE_DIM > 0 else config.n_embd

        def _table() -> nn.Embedding:
            # `_weight=` skips nn.Embedding.reset_parameters(), which would otherwise draw
            # 41.9M normals PER TABLE (210M total) from the global generator only for
            # init_weights to immediately zero them -- wasted work AND, worse, an RNG-stream
            # shift that would change every other tensor's init. Zero is the real init here
            # (see init_weights): it makes the step-0 forward bit-identical to USE_NGRAM_VE
            # off, because the zero-init gate evaluates to the NEUTRAL 2*sigmoid(0) = 1.0
            # rather than 0, so only a zero TABLE makes the injection a true no-op.
            return nn.Embedding(self.ngram_rows, self.ngram_dim,
                                _weight=torch.zeros(self.ngram_rows, self.ngram_dim, dtype=ngram_dtype))

        tables: dict[str, nn.Module] = {}
        for i in range(config.n_layer):
            if self.layer_is_mamba[i]:
                continue
            if has_ngram_bigram(i, config.n_layer):
                tables[f"b{i}"] = _table()
            if has_ngram_trigram(i, config.n_layer):
                tables[f"t{i}"] = _table()
        self.ngram_embeds = nn.ModuleDict(tables)
        # NGRAM_VE_DIM: per-table up-projection to n_embd. Deliberately per-table rather than
        # one shared matrix -- a shared one would force every layer's table into the same
        # 128-dimensional subspace of v, which is exactly the cross-layer decorrelation the
        # per-layer prime pairs exist to create. Uses the repo's own Linear (bf16 weight cache,
        # Muon-eligible), and is empty when the tables are already full width so the
        # NGRAM_VE_DIM=0 path adds no module and no state_dict key at all.
        # _rng_neutral for the same reason as the gates: init_weights re-initializes these
        # unconditionally, so their construction-time draws must not shift the RNG stream and
        # change every other tensor's init between the two arms of the A/B.
        self.ngram_proj = nn.ModuleDict(
            {k: _rng_neutral(lambda: Linear(self.ngram_dim, config.n_embd, bias=False))
             for k in tables}
            if self.ngram_dim != config.n_embd else {})
        # The n-gram gates live in CausalSelfAttention, so an RWKV layer (which has a
        # tmix instead of an attn) would carry a table nothing ever reads. Refuse the
        # combination loudly rather than silently pay ~40M dead parameters for it.
        assert not (tables and any(self.layer_is_rwkv)), \
            "USE_NGRAM_VE is not wired into RWKVTimeMix; use --no-rwkv"
        # Sentinel for the missing history at positions 0 (and 1): a token id one past the
        # padded vocabulary, so it can never be confused with a real token. NGRAM_HASH_PRIMES
        # is bounded so that this id times any multiplier still fits in int32.
        self.ngram_pad_id = padded_vocab_size
        # T3.2 Engram-lite. Built last among the table-owning modules so it cannot shift the RNG
        # stream of anything above it (its own construction is RNG-neutral either way).
        self.engram_site = engram_site_index(config.n_layer) if USE_ENGRAM else -1
        self.engram_share_keys: tuple[str, ...] = ()
        if USE_ENGRAM:
            share_e_dim = None
            if ENGRAM_SHARE_NGRAM_TABLES:
                # Share mode reuses the rows T3.1 has ALREADY gathered, so the mechanism (gate +
                # conv + residual write) can be measured with zero new gathers, zero new
                # parameters in the tables and zero extra gradient all-reduce. Only tables at or
                # before the injection site qualify: a later layer's rows do not exist yet.
                self.engram_share_keys = tuple(
                    k for k in self.ngram_embeds if int(k[1:]) <= self.engram_site)
                assert self.engram_share_keys, (
                    f"ENGRAM_SHARE_NGRAM_TABLES needs an n-gram table at or before layer "
                    f"{self.engram_site}; have {list(self.ngram_embeds)}")
                share_e_dim = self.ngram_dim * len(self.engram_share_keys)
            self.engram = EngramLite(config, padded_vocab_size, share_e_dim)
        else:
            self.engram = None
        # T4.2: Aster's output bigram logit table. [padded_vocab, padded_vocab] so a row can be
        # gathered by raw token id and sliced to the real vocab on the way into the logits, which
        # keeps the gather's index space identical to wte's. Built RNG-neutrally and zero-init (see
        # init_weights) for the same two reasons as the n-gram tables: an exact step-0 no-op, and
        # no shift of the RNG stream that every other tensor's init depends on.
        if USE_OUT_BIGRAM:
            ob_dtype = torch.bfloat16 if OUT_BIGRAM_BF16 else torch.float32
            self.out_bigram = nn.Embedding(
                padded_vocab_size, padded_vocab_size,
                _weight=torch.zeros(padded_vocab_size, padded_vocab_size, dtype=ob_dtype))
            self.out_bigram_gate = _rng_neutral(lambda: Linear(config.n_embd, 1, bias=False))
            self.out_bigram_gate_bias = nn.Parameter(torch.full((1,), float(OUT_BIGRAM_GATE_BIAS)))
        else:
            self.out_bigram = None
            self.out_bigram_gate = None
            self.out_bigram_gate_bias = None
        # T3.9: zero-init pooling of the last OUT_POOL_LAYERS block outputs into the final
        # residual. One learned scalar per pooled layer, all zero at init, so step 0 is an exact
        # no-op and the arm can only ever add what it learns to be worth adding.
        self.out_pool_lambdas = nn.Parameter(torch.zeros(min(OUT_POOL_LAYERS, config.n_layer))) \
            if USE_OUT_POOL else None
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
            if block.tmix is not None:
                # Same discipline as attention/Mamba2: input projections get the shared
                # uniform init, output projections are zeroed so a fresh RWKV block is
                # an exact identity (both sublayers: tmix.output and cmix.value). The
                # x051a-specific tensors (time_maa_*, time_decay, time_faaaa, ln_x gain
                # and bias) were set by reset_rwkv_parameters with the reference's
                # layer-dependent schedule and must NOT be clobbered here.
                for lin in (block.tmix.receptance, block.tmix.key, block.tmix.value, block.tmix.gate):
                    torch.nn.init.uniform_(lin.weight, -s, s)
                torch.nn.init.zeros_(block.tmix.output.weight)
                if block.tmix.ve_gate is not None:
                    torch.nn.init.zeros_(block.tmix.ve_gate.weight)
                # cmix.key is the 3x-wide "c_fc" analogue -> the same 0.4*s the MLP uses.
                torch.nn.init.uniform_(block.cmix.key.weight, -0.4 * s, 0.4 * s)
                torch.nn.init.uniform_(block.cmix.receptance.weight, -s, s)
                torch.nn.init.zeros_(block.cmix.value.weight)
                if block.resid_lambda is not None:
                    block.resid_lambda.fill_(1.0)
                    block.x0_lambda.fill_(0.05)
                if block.scalar_gate is not None:
                    torch.nn.init.zeros_(block.scalar_gate.weight)
                continue
            if block.c_in is not None:
                # Parallel block, coalesced input projection. Filled slice by slice IN THE
                # BASELINE'S ORDER and with the baseline's per-slice bounds (q/k/v at +-s, the
                # c_fc slice at +-0.4s), each row range being exactly the weight matrix it
                # replaces. Because uniform_ advances the RNG by numel regardless of the bound,
                # this consumes the same draws in the same order as the uncoalesced arm, so the
                # two are bit-identical at init -- which is what lets one A/B isolate the
                # coalescing from the architecture change.
                n = self.config.n_embd
                if block.coalesced == "all":
                    torch.nn.init.uniform_(block.c_in.weight[:n], -s, s)
                    torch.nn.init.uniform_(block.c_in.weight[n:2 * n], -s, s)
                    torch.nn.init.uniform_(block.c_in.weight[2 * n:3 * n], -s, s)
                    fc_rows = block.c_in.weight[3 * n:]
                else:  # "vfc": q/k are still real Linears, initialized in the branch below
                    torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                    torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                    torch.nn.init.uniform_(block.c_in.weight[:n], -s, s)
                    fc_rows = block.c_in.weight[n:]
            elif USE_FUSED_QKV:
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
            if ATTN_CPROJ_INIT_SCALE > 0.0:
                # T0.4: nonzero attn output-projection init (see ATTN_CPROJ_INIT_SCALE).
                b = ATTN_CPROJ_INIT_SCALE * s
                torch.nn.init.uniform_(block.attn.c_proj.weight, -b, b)
            else:
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            # The c_fc slice of a coalesced c_in comes LAST, matching the baseline's order
            # (q, k, v, attn.c_proj, c_fc) so the RNG stream is identical -- see block.c_in above.
            torch.nn.init.uniform_(block.mlp.c_fc.weight if block.c_in is None else fc_rows,
                                   -0.4 * s, 0.4 * s)
            if block.mlp.c_gate is not None:
                torch.nn.init.uniform_(block.mlp.c_gate.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.resid_lambda is not None:
                block.resid_lambda.fill_(1.0)
                block.x0_lambda.fill_(0.05)
            if block.par_lambda_a is not None:
                # Explicit, not left to nn.Parameter(ones): every other scalar in this loop is set
                # here, and la=1.0 is exactly what makes the parallel block a step-0 no-op.
                block.par_lambda_a.fill_(1.0)
                block.par_lambda_m.fill_(1.0)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            for g in (block.attn.ve_gate_bi, block.attn.ve_gate_tri):
                if g is not None:
                    torch.nn.init.zeros_(g.weight)   # T3.1: neutral gate = 2*sigmoid(0) = 1.0
            if block.scalar_gate is not None:
                torch.nn.init.zeros_(block.scalar_gate.weight)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # T3.1: ZERO-init, deliberately unlike the unigram tables above. The gate is neutral
        # (1.0) rather than closed at init, so a zero TABLE is the only thing that makes
        # USE_NGRAM_VE=True bit-identical to USE_NGRAM_VE=False at step 0. It is also what
        # modded-nanogpt #62 does with its bigram table.
        for tbl in self.ngram_embeds.values():
            torch.nn.init.zeros_(tbl.weight)
        # NGRAM_VE_DIM's up-projections must NOT be zero-init, which is the opposite of the rule
        # every other output projection in this file follows. The table is already zero, so the
        # injected value is zero either way and the step-0 no-op holds regardless -- but the
        # TABLE's gradient is (dL/dv) @ proj, so a zero projection would make it identically
        # zero and neither tensor would ever leave the origin. Same uniform(-s, s) as the
        # unigram value-embedding tables, since this matrix is what plays their role here.
        for proj in self.ngram_proj.values():
            torch.nn.init.uniform_(proj.weight, -s, s)
        # T3.2 Engram-lite. Zero TABLES + zero CONV, and W_K/W_V/W_Q emphatically NOT zero.
        # The chain that makes step 0 exact: e_t = 0 => k_t = v_t = 0 => alpha = sigmoid(0) = 0.5
        # (a nonzero constant, so the gate alone would NOT do it) => vt = 0.5 * 0 = 0 =>
        # RMSNorm(0) = 0 => Conv(0) = 0 => SiLU(0) = 0 => Y = 0 => H unchanged, bit-exactly.
        # And W_V must stay nonzero for the same reason ngram_proj must: the table's gradient is
        # dL/de = W_V^T dL/dv + W_K^T dL/dk, which a zero W_V/W_K would pin at the origin forever.
        if self.engram is not None:
            for tbl in self.engram.tables.values():
                torch.nn.init.zeros_(tbl.weight)
            for lin in (self.engram.w_k, self.engram.w_v, self.engram.w_q):
                if lin is not None:
                    torch.nn.init.uniform_(lin.weight, -s, s)
            if self.engram.conv is not None:
                torch.nn.init.zeros_(self.engram.conv.weight)
        # T4.2: zero TABLE + zero gate WEIGHT, and the gate bias keeps its constructed value
        # (OUT_BIGRAM_GATE_BIAS). The zero table is what makes step 0 exact: sigmoid(bias) is a
        # positive constant, so a nonzero table would shift every logit at init.
        if self.out_bigram is not None:
            torch.nn.init.zeros_(self.out_bigram.weight)
            torch.nn.init.zeros_(self.out_bigram_gate.weight)
        # T3.9: zero pooling weights => exact step-0 no-op (they are constructed zero; this line
        # exists so a future non-zero construction cannot silently change that).
        if self.out_pool_lambdas is not None:
            torch.nn.init.zeros_(self.out_pool_lambdas)
        # T3.6 / T3.9: zero head-gate weight, bias left at HEAD_GATE_BIAS. With
        # HEAD_GATE_MUL * sigmoid(HEAD_GATE_BIAS) == 1.0 (the default 2.0 / 0.0 pair) and
        # HEAD_GATE_NORM off, this is an exact identity at step 0.
        for block in self.transformer.h:
            if block.attn is not None and block.attn.head_gate is not None:
                torch.nn.init.zeros_(block.attn.head_gate.weight)
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, self.config.n_embd // self.config.n_head)
        self.cos.copy_(cos)
        self.sin.copy_(sin)

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
        # Missing history at t = 0 (and t = 1) gets a sentinel id one past the padded
        # vocabulary, so "no previous token" is its own hash input rather than colliding
        # with real token 0.
        pad = torch.full_like(cur[:, :1], self.ngram_pad_id)
        prev = torch.cat((pad, cur[:, :-1]), dim=1)
        need_trigram = any(k.startswith("t") for k in self.ngram_embeds)
        # T3.2's own-mode tables share the prev/prev2 shifts computed here rather than recomputing
        # them: the shifts are the only part of the index math that is per-position rather than
        # per-table, so doing it once keeps the launch count down in a dispatch-bound graph.
        engram_keys = tuple(self.engram.tables) if self.engram is not None else ()
        need_trigram = need_trigram or any(k.endswith("o3") for k in engram_keys)
        prev2 = torch.cat((pad, pad, cur[:, :-2]), dim=1) if need_trigram else None
        out: dict[str, torch.Tensor] = {}
        for key in engram_keys:
            head_idx, order = int(key[1:key.index("o")]), int(key[key.index("o") + 1:])
            primes = engram_hash_primes(head_idx, order)
            if order == 2:
                h = torch.bitwise_xor(prev * primes[0], cur * primes[1])
            else:
                h = torch.bitwise_xor(torch.bitwise_xor(prev2 * primes[0], prev * primes[1]),
                                      cur * primes[2])
            out[key] = torch.remainder(h, self.engram.mod)
        for key in self.ngram_embeds:
            layer = int(key[1:])
            if key[0] == "b":
                p_prev, p_cur = ngram_bigram_primes(layer)
                h = torch.bitwise_xor(prev * p_prev, cur * p_cur)
            else:
                p_prev2, p_prev, p_cur = ngram_trigram_primes(layer)
                h = torch.bitwise_xor(torch.bitwise_xor(prev2 * p_prev2, prev * p_prev),
                                      cur * p_cur)
            out[key] = torch.remainder(h, self.ngram_mod)
        return out

    def _ngram_lookup(self, key: str, rows: torch.Tensor,
                      raw_out: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
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
        # T3.2 share mode taps the PRE-projection rows: they are what an Engram table would hold,
        # and reading them here means the shared arm adds no gather of its own. `raw_out` is None
        # on the default path, so that path's ops are byte-identical to before.
        if raw_out is not None:
            raw_out[key] = out
        if key in self.ngram_proj:
            out = self.ngram_proj[key](out)
        return out

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
        # Same reasoning as wte: an n-gram table is READ BY A GATHER, not a matmul, so
        # pricing its up-to-210M parameters at 6 FLOPs/param/token would report a fictitious
        # ~1.3 GFLOP/token and make the MFU number meaningless. Deliberately NOT extended to
        # the (much smaller) unigram value_embeds tables, which have the same character but
        # whose exclusion would silently move every historical MFU number in the log.
        non_matmul += sum(p.numel() for p in self.ngram_embeds.parameters())
        # T3.2: same reasoning -- Engram's tables are read by a gather, never multiplied.
        if self.engram is not None:
            non_matmul += sum(p.numel() for p in self.engram.tables.parameters())
        # T4.2: same reasoning -- the [V, V] output bigram table is read by a gather and added,
        # never multiplied, so pricing its 67M parameters at 6 FLOPs/param/token would report a
        # fictitious ~0.4 GFLOP/token and make every MFU number in this run meaningless.
        if self.out_bigram is not None:
            non_matmul += self.out_bigram.weight.numel()
        h = self.config.n_head
        d = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        # QK^T: 2*h*d*T per token, AV: 2*h*d*T per token, x3 for fwd+bwd.
        # Only TRUE attention layers pay this: a Mamba2 or RWKV layer has no
        # T-by-T contraction. RWKV's sequence-mixing cost is quadratic only inside
        # a chunk (Q, not T) plus a chunk-granularity pass, so it is counted here at
        # its own scale rather than being silently priced as full attention.
        n_attn = sum(1 for i in range(self.config.n_layer)
                     if not (self.layer_is_mamba[i] or self.layer_is_rwkv[i]))
        attn_flops = n_attn * 12 * h * d * t
        n_rwkv = sum(1 for i in range(self.config.n_layer) if self.layer_is_rwkv[i])
        if n_rwkv:
            hr, nr = self.config.n_embd // RWKV_HEAD_DIM, RWKV_HEAD_DIM
            q = min(RWKV_CHUNK, t)
            # per token: intra-chunk (r@k^T then @v) 4*hr*nr*Q, plus the state path
            # (k^T@v build, chunk-decay mix, r@S) ~6*hr*nr*nr/Q*Q -> 6*hr*nr*nr per
            # chunk amortized over Q tokens; x3 for fwd+bwd.
            attn_flops += n_rwkv * 3 * (4 * hr * nr * q + 6 * hr * nr * nr // max(1, q))
        return 6 * (nparams - non_matmul) + attn_flops

    def setup_optimizer(self):
        scalar_params = []
        matrix_params = []
        local_conv_params = []  # T0.1: local_conv.weight, previously dead compute -- see below
        ssm_params = []  # Mamba2's non-matrix state-space tensors, see below
        rwkv_params = []  # RWKV x051a's non-matrix per-channel/per-head tensors, see below
        gate_params = []  # T3.9/T4.2: small 1-D or single-row gate tensors (attention head-gate
        # biases, the output-bigram gate's weight+bias, per-dim Q/K shift betas). None of them is
        # Muon material -- a 1-D vector has no second axis to orthogonalize and a [1, n_embd] row
        # would come out of Newton-Schulz as a unit-norm direction, i.e. a wildly oversized update
        # -- and none of them belongs in the SCALAR_LR=0.20 group either, which is tuned for a
        # single learned scalar's blast radius and has already caused a real NaN divergence when a
        # genuine tensor was put there (see the hypernet_params comment below). AdamW at MATRIX_LR,
        # weight_decay 0, exactly like ssm_params/rwkv_params/local_conv_params.
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
            if block.local_conv is not None:
                # T0.1: register the depthwise causal conv1d weight so it actually
                # trains. Not Muon material (depthwise: each output channel's kernel
                # is an independent [1, k] row with no second axis to orthogonalize,
                # and k is tiny -- same reasoning as ssm_params/rwkv_params below), and
                # not the SCALAR_LR=0.20 group either (that LR is tuned for a single
                # learned scalar's blast radius, not a real weight tensor -- see the
                # hypernet_params comment). Own AdamW group at MATRIX_LR, wd=0.0,
                # matching the plan's T0.1 spec. Kept zero-initialized (CausalConv.__init__)
                # so step 0 is still an exact no-op regardless of this change.
                local_conv_params.append(block.local_conv.weight)
            if block.resid_lambda is not None:
                scalar_params.extend([block.resid_lambda, block.x0_lambda])
            if block.x0_gate_s is not None:
                # A single learned scalar per layer, exactly the class SCALAR_LR was tuned for --
                # same group as the resid_lambda/x0_lambda it gates. It also NEEDS the larger LR:
                # its input is mean(x) over 640 channels, which is ~1/sqrt(640) of a channel's own
                # scale, so the gradient reaching s is correspondingly small.
                scalar_params.append(block.x0_gate_s)
            if block.par_lambda_a is not None:
                # Two learned scalars per layer, same class as resid_lambda/x0_lambda: one
                # multiplies a whole branch output, which is what SCALAR_LR=0.20 is tuned for.
                scalar_params.extend([block.par_lambda_a, block.par_lambda_m])
            if block.mudd is not None:
                hypernet_params.extend([block.mudd.w1, block.mudd.w2, block.mudd.bias])
            if block.v_lambda is not None:
                scalar_params.append(block.v_lambda)
            if block.qk_shift_beta is not None and block.qk_shift_beta.requires_grad:
                # QK_SHIFT_FREEZE leaves beta out of EVERY optimizer group on purpose (see the
                # QK_SHIFT_FREEZE comment): the throughput probe needs the ops in the graph but the
                # parameter mathematically inert, and an AdamW group would move it off its init.
                gate_params.append(block.qk_shift_beta)
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
            if block.tmix is not None:
                # The five/three projections of each sublayer are ordinary weight
                # matrices -> Muon, grouped by shape like every other block matrix.
                for lin in (block.tmix.receptance, block.tmix.key, block.tmix.value,
                            block.tmix.gate, block.tmix.output,
                            block.cmix.key, block.cmix.value, block.cmix.receptance):
                    matrix_params.append(lin.weight)
                if block.tmix.ve_gate is not None:
                    matrix_params.append(block.tmix.ve_gate.weight)
                # Everything else RWKV owns is a 1-D per-channel or per-head vector with
                # no second axis to orthogonalize (time_maa_* are [1,1,C] mixing ratios,
                # time_decay/time_faaaa are [H,1], ln_x_weight/ln_x_bias are the per-head
                # GroupNorm affine). Same reasoning as ssm_params: not Muon material, and
                # emphatically not SCALAR_LR=0.20 material either -- that LR already
                # caused a real NaN divergence when a genuine tensor was put there (see
                # the hypernet_params comment). AdamW at MATRIX_LR, weight_decay=0.0,
                # which also matches upstream RWKV's no-weight-decay set for these.
                rwkv_params.extend([block.tmix.time_maa_k, block.tmix.time_maa_v,
                                    block.tmix.time_maa_r, block.tmix.time_maa_g,
                                    block.tmix.time_decay, block.tmix.time_faaaa,
                                    block.tmix.ln_x_weight, block.tmix.ln_x_bias,
                                    block.cmix.time_maa_k, block.cmix.time_maa_r])
                continue
            if block.attn.dcmha is not None:
                # DCMHA's hypernetwork weights are tiny and vary in shape per
                # layer/branch -- not a fit for Muon's same-shape orthogonalization
                # groups, so route them through AdamW alongside the scalar/skip
                # params (excluded from block.attn.parameters() below by name).
                hypernet_params.extend(block.attn.dcmha.parameters())
            if block.attn.head_gate is not None:
                # The [n_head, HEAD_GATE_CHANNELS] weight goes to Muon with every other per-head
                # gate matrix (ve_gate/ve_gate_bi/ve_gate_tri have exactly this shape); only the
                # 1-D bias has to be pulled out by name.
                gate_params.append(block.attn.head_gate_bias)
            matrix_params.extend(p for name, p in block.attn.named_parameters()
                                 if not name.startswith("dcmha.") and name != "head_gate_bias")
            matrix_params.extend(block.mlp.parameters())
            if block.c_in is not None:
                # The coalesced input projection is an ordinary weight matrix, so it goes to Muon
                # with every other block matrix. It lives on the Block rather than inside attn/mlp
                # (it spans both), which is why it needs its own line here. Its [4480, 640] shape is
                # its own Muon shape group -- which is itself part of what the arm measures, since
                # Muon's per-shape orthogonalization now sees one big matrix instead of four.
                matrix_params.append(block.c_in.weight)
        # NGRAM_VE_DIM's up-projections are ordinary [n_embd, ngram_dim] weight matrices, so
        # they belong in Muon with every other matrix, grouped by shape like the rest. They
        # live on GPT rather than inside a Block, which is why they need an explicit line here
        # -- the loop above only walks self.transformer.h.
        matrix_params.extend(p.weight for p in self.ngram_proj.values())
        # T3.2: W_K / W_V / W_Q are ordinary dense 2-D weight matrices with no shape pathology, so
        # they go to Muon like every other matrix here. The conv kernel does NOT (depthwise, k=4 --
        # same argument as block.local_conv/ssm_params) and gets its own AdamW group below.
        engram_conv_params = []
        if self.engram is not None:
            matrix_params.extend(lin.weight for lin in
                                 (self.engram.w_k, self.engram.w_v, self.engram.w_q)
                                 if lin is not None)
            if self.engram.conv is not None:
                engram_conv_params.append(self.engram.conv.weight)
        if self.out_pool_lambdas is not None:
            # T3.9: OUT_POOL_LAYERS learned scalars, exactly the same character as
            # resid_lambda/x0_lambda -- the group SCALAR_LR was actually tuned for.
            scalar_params.append(self.out_pool_lambdas)
        if self.out_bigram is not None:
            gate_params.extend([self.out_bigram_gate.weight, self.out_bigram_gate_bias])
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        value_embed_params = list(self.value_embeds.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, EMBED_ADAMW_BETA2), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=scalar_params, lr=SCALAR_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if gate_params:
            param_groups.append(dict(kind="adamw", params=gate_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if hypernet_params:
            param_groups.append(dict(kind="adamw", params=hypernet_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if ssm_params:
            param_groups.append(dict(kind="adamw", params=ssm_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if rwkv_params:
            param_groups.append(dict(kind="adamw", params=rwkv_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if value_embed_params:
            param_groups.append(dict(kind="adamw", params=value_embed_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, EMBED_ADAMW_BETA2), eps=1e-10, weight_decay=VE_WD))
        ngram_params = list(self.ngram_embeds.parameters())
        # T3.2's own-mode tables ride T3.1's group deliberately: they are the same object (a hashed
        # n-gram table read by a row gather), so every one of the three reasons below applies
        # verbatim, and a separate group would only add an untuned LR to sweep.
        if self.engram is not None:
            ngram_params.extend(self.engram.tables.parameters())
        if ngram_params:
            # T3.1. Its own group for three reasons, all from Recursive's recipe:
            #  * RMSProp (no first moment) rather than Adam. A hashed n-gram row is touched
            #    a fraction of a step on average, and Adam's m/sqrt(v) DAMPS a rare update
            #    by exactly the frequency factor it divides out of v; dropping the first
            #    moment gives every row a full-rate update whenever it is hit, which is
            #    the stated point ("sparse tables benefit from full-rate training").
            #  * weight_decay 0 -- decaying a table 99.95% of whose rows have no gradient
            #    this step is pure signal destruction.
            #  * no_lr_decay -- these tables keep a FLAT lr for the whole run instead of
            #    following the warmdown (the training loop honours this key; warmup still
            #    applies, only the cooldown is skipped).
            # NGRAM_VE_OPT="adamw" is the fallback arm: identical group, but the same
            # (0.8, EMBED_ADAMW_BETA2) Adam the unigram value-embedding tables use, so the
            # optimizer choice can be A/B'd on its own without touching anything else.
            ngram_lr = (NGRAM_VE_LR if NGRAM_VE_LR > 0.0 else EMBEDDING_LR) * dmodel_lr_scale
            rmsprop = NGRAM_VE_OPT == "rmsprop"
            param_groups.append(dict(
                kind="rmsprop" if rmsprop else "adamw", params=ngram_params, lr=ngram_lr,
                betas=(0.0, NGRAM_VE_BETA2) if rmsprop else (0.8, EMBED_ADAMW_BETA2),
                eps=1e-10, weight_decay=0.0, no_lr_decay=NGRAM_VE_FLAT_LR))
        if self.out_bigram is not None:
            # T4.2. Its own AdamW group mirroring lm_head's, which is the right reference: this
            # table's output is an additive term on the same logits lm_head produces, so its
            # gradient has lm_head's scale and UNEMBEDDING_LR is the LR that was swept against it
            # (and found to be a sharp optimum -- +0.0017/+0.0014 for -42%/+67%). NOT Muon: a
            # [V, V] table read by a row gather is Muon-ineligible for exactly the reason wte is.
            # weight_decay 0 rather than lm_head's 0.01: most rows see gradient from only the
            # handful of positions holding that token this step, so decay would shrink rare rows
            # far harder than common ones -- the same argument the n-gram group above makes.
            param_groups.append(dict(
                kind="adamw", params=[self.out_bigram.weight],
                lr=(OUT_BIGRAM_LR if OUT_BIGRAM_LR > 0.0 else UNEMBEDDING_LR) * dmodel_lr_scale,
                betas=(0.8, 0.96), eps=1e-10, weight_decay=0.0))
        if local_conv_params and TRAIN_LOCAL_CONV:
            param_groups.append(dict(kind="adamw", params=local_conv_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        if engram_conv_params:
            # T3.2: deliberately NOT folded into the local_conv group, which is gated on
            # TRAIN_LOCAL_CONV -- flipping that debug toggle must not silently freeze Engram's conv
            # (a frozen zero conv would make Y == vt forever and quietly delete half the mechanism).
            param_groups.append(dict(kind="adamw", params=engram_conv_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
        for shape in sorted({p.shape for p in matrix_params}):
            params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind="muon", params=params, lr=MATRIX_LR, momentum=0.95, beta2=0.90, ns_steps=MUON_NS_STEPS, weight_decay=WEIGHT_DECAY))
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
        # T3.1: all hash indices in one place (each is a [b, T] int32 tensor, ~32 KB), so
        # the shared prev/prev2 shifts are computed once rather than once per layer.
        ngram_idx = self._ngram_hash_indices(idx) \
            if (len(self.ngram_embeds) or self.engram is not None) else None
        # T3.2 share mode: the pre-projection rows T3.1 gathers, keyed by table, accumulated as the
        # loop passes each table's layer. Stays None otherwise so the default path is unchanged.
        engram_raw: dict[str, torch.Tensor] | None = \
            {} if (self.engram is not None and self.engram.share) else None
        # T3.9: the trailing block outputs the output pooling reads. Only materialized when the
        # toggle is on, so the default path keeps exactly one live residual as before.
        pool_from = self.config.n_layer - self.out_pool_lambdas.numel() if self.out_pool_lambdas is not None else -1
        pool_buf: list[torch.Tensor] = []
        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](idx).to(COMPUTE_DTYPE) if str(i) in self.value_embeds else None
            ve_bi = ve_tri = None
            if ngram_idx is not None:
                if f"b{i}" in self.ngram_embeds:
                    ve_bi = self._ngram_lookup(f"b{i}", ngram_idx[f"b{i}"], engram_raw)
                if f"t{i}" in self.ngram_embeds:
                    ve_tri = self._ngram_lookup(f"t{i}", ngram_idx[f"t{i}"], engram_raw)
            if self.layer_is_mamba[i] or self.layer_is_rwkv[i]:
                # No attention here, so don't materialize an additive mask this
                # layer will never read. Both alternatives are causal structurally
                # rather than by mask: Mamba2 via left-padded conv + masked-before-exp
                # decay, RWKV via a 1-step time_shift + a decay matrix that is exactly
                # 0.0 above the diagonal (also masked before exp).
                attn_mask = None
            elif USE_CAUSAL_FASTPATH and doc_mask is None and not self.layer_is_short[i]:
                # Solution 13d: this layer's static mask is purely causal (full context,
                # no doc-boundary masking) -- let SDPA take its is_causal fast path
                # instead of materializing the additive mask tensor.
                attn_mask = None
            else:
                static_mask = (self.mask_short if self.layer_is_short[i] else self.mask_full)[..., :t, :t]
                attn_mask = static_mask + doc_mask if doc_mask is not None else static_mask
            x, v_raw = block(x, x0, ve, cos, sin, attn_mask, v0, hidden_states, ve_bi, ve_tri)
            if self.engram is not None and i == self.engram_site:
                # H <- H + Y, on the residual leaving this block. Placed at the LAST layer by
                # default: the gather's dynamic address forces the LNC=2 compiler to re-shard the
                # residual stream at every DOWNSTREAM layer boundary (~+48ms/micro-step for a
                # layer-3 injection vs 0 for layer 5), and that cost is invariant to d_mem.
                if engram_raw is not None:
                    parts = [engram_raw[k] for k in self.engram_share_keys]
                    e_t = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
                else:
                    e_t = self.engram.gather(ngram_idx)
                x = x + self.engram(x, e_t)
            if USE_MUDD:
                hidden_states.append(x)
            if USE_VALUE_RESIDUAL and v0 is None:
                v0 = v_raw
            if i >= pool_from:
                pool_buf.append(x)
        if self.out_pool_lambdas is not None:
            # T3.9: zero-init pooling of the last few layers' outputs into the final residual. The
            # last entry of pool_buf IS x, so its term is a redundant rescale of x -- kept anyway
            # because that is Recursive's own "pool the last three layers" and dropping it would
            # silently make this a 2-layer pool.
            for k, h in enumerate(pool_buf):
                x = x + self.out_pool_lambdas[k] * h
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        if self.out_bigram is not None:
            # T4.2: bias[b, t, :] = sigmoid(w . x[b,t] + bias) * table[idx[b, t], :]. Added to the
            # RAW logits, i.e. BEFORE the softcap -- both the eager tanh below and the fused NKI
            # softcap+CE kernel take pre-softcap logits and apply LOGIT_SOFTCAP*tanh(z/S)
            # themselves, so the bias has to land here to be capped along with everything else
            # (adding it after the cap could push a logit outside (-S, S) and would also change
            # what the kernel's own backward is the gradient OF).
            # The gather+multiply stay in COMPUTE_DTYPE and only the [.., vocab] result is upcast,
            # which is Aster's "added before the float upcast" and halves the intermediate.
            g = torch.sigmoid(self.out_bigram_gate(x) + self.out_bigram_gate_bias.to(x.dtype))
            row = self.out_bigram.weight[idx][..., :self.config.vocab_size]
            logits = logits + (g * row).float()
        if targets is None:
            return LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        return softcap_cross_entropy(logits, targets, loss_reduction)


def build_config(depth: int | None = None) -> GPTConfig:
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
            raise ValueError(f"N_EMBD={N_EMBD} must be a multiple of HEAD_DIM={HEAD_DIM}")
    else:
        base_dim = depth * ASPECT_RATIO
        n_embd = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    n_head = n_embd // HEAD_DIM
    return GPTConfig(
        sequence_len=SEQ_LEN,
        vocab_size=TOKENIZER_VOCAB_SIZE,
        n_layer=depth,
        n_head=n_head,
        n_embd=n_embd,
        mlp_ratio=MLP_RATIO,
        attn_scale=ATTN_SCALE,
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


# Kinds that T1.4's every-N-steps machinery treats alike: both are diagonal, per-element
# optimizers whose gradients accumulate in place between updates, unlike Muon's stacked
# momentum buffers. See MuonAdamW.step / zero_grad_selective / adamw_param_ids.
ADAM_LIKE_KINDS = ("adamw", "rmsprop")


# T1.7: toggled by --compile-muon in main(). Wrap Muon's per-group math body in
# torch.compile(backend="neuron") so the whole elementwise+matmul chain becomes ONE
# NEFF per group instead of ~80 aten launches. Default OFF until measured on this
# config; see the _muon_math docstring for why this was blocked and what unblocked it.
USE_COMPILE_MUON = False

# One compiled callable per (tall, ns_steps, red_dim) specialisation, built lazily.
# Dynamo re-guards on the tensor shapes by itself, so a single entry can still hold
# several NEFFs -- this cache only exists so the wrapper object is not rebuilt per step.
_MUON_COMPILED: dict[tuple, object] = {}


def _muon_math(w, g, momentum_buffer, second, mom, one_mom, lr, lrwd,
               tall: bool, ns_steps: int, red_dim: int, red_dim_size: int,
               one_beta2: float, to_bf16: bool):
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
      `tall` / `ns_steps` / `red_dim` / `red_dim_size` / `one_beta2` / `to_bf16` ARE
      Python constants on purpose: they are fixed per group for the whole run, so
      guarding on them is free and it keeps the traced graph straight-line.

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
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
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
    scaled = (v_mean * red_dim_size) * step_size.square()
    v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
    mask = (g * w) >= 0
    w = w - (lr * g + lrwd * w * mask)
    return w, momentum_buffer, second


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
            entry = (torch.empty((), dtype=torch.float32),
                     torch.zeros((), dtype=torch.float32, device=ref.device))
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
            host = torch.empty((len(POLAR_EXPRESS_COEFFS), 3, _NKI_PARTITION, 1),
                               dtype=torch.float32)
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
        entry = cache.get("muon_lrwd")
        if entry is None:
            entry = (torch.empty((2, _NKI_PARTITION, 1), dtype=torch.float32),
                     torch.zeros((2, _NKI_PARTITION, 1), dtype=torch.float32,
                                 device=ref.device),
                     torch.empty((), dtype=torch.float32))
            cache["muon_lrwd"] = entry
        host, dev, tmp = entry
        tmp.fill_(float(group["lr"]))
        tmp.mul_(ratio)                      # fp32(fp32(lr) * fp32(ratio)) == device
        host[0].fill_(float(tmp))
        tmp.mul_(float(group["weight_decay"]))
        host[1].fill_(float(tmp))
        dev.copy_(host)
        return dev[0], dev[1]

    def _muon_compile_scalars(self, group, ref: torch.Tensor, ratio: float):
        """T1.7: (lr*ratio, lr*ratio*wd, momentum, 1-momentum) as four 0-dim fp32 device
        views behind ONE host->device copy of a (4,) staging tensor.

        Replaces `_device_scalar("lr")` for the compiled path at strictly lower cost --
        one copy instead of one per scalar -- and it is the reason the compiled graph has
        no per-step constants at all. Slicing `dev[i]` is a pure view: no device work, no
        launch, so four operands cost the same as one.

        The two products are formed in float32 (see `_muon_lrwd`) and `1-momentum` in
        float64 (see `_muon_math`), each matching what the eager path's arithmetic does
        at that point. This distinction is load-bearing, not stylistic.
        """
        cache = self._scalar_cache.setdefault(id(group), {})
        entry = cache.get("muon_compile")
        if entry is None:
            entry = (torch.empty(4, dtype=torch.float32),
                     torch.zeros(4, dtype=torch.float32, device=ref.device),
                     torch.empty((), dtype=torch.float32))
            cache["muon_compile"] = entry
        host, dev, tmp = entry
        tmp.fill_(float(group["lr"]))
        tmp.mul_(ratio)                      # fp32(fp32(lr) * fp32(ratio)) == device
        host[0].fill_(float(tmp))
        tmp.mul_(float(group["weight_decay"]))
        host[1].fill_(float(tmp))
        mom = float(group["momentum"])
        host[2].fill_(mom)
        host[3].fill_(1.0 - mom)             # float64 subtraction, like lerp_.Scalar's
        dev.copy_(host)
        return dev[0], dev[1], dev[2], dev[3]

    def _adamw_state(self, group):
        """Shared prologue for both AdamW paths: filter to params that actually have a
        grad this step, lazily create their state, advance the per-parameter step
        counter, and return the parallel lists the update needs.

        The step counter is per-parameter and NOT per-group because a parameter whose
        grad was None on some earlier step legitimately lags the rest of its group; the
        bias corrections are therefore returned as per-parameter scalar lists.
        """
        beta1, beta2 = group["betas"]
        params, grads, exp_avgs, exp_avg_sqs, bias1, bias2 = [], [], [], [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            params.append(p)
            grads.append(p.grad)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            bias1.append(1 - beta1 ** state["step"])
            bias2.append(1 - beta2 ** state["step"])
        return params, grads, exp_avgs, exp_avg_sqs, bias1, bias2

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
        lr = group["lr"]
        if USE_TENSOR_LR_SCALARS:
            lr = self._device_scalar(group, "lr", lr, params[0])
        beta1, beta2 = group["betas"]
        wd = group["weight_decay"]
        if wd != 0:
            torch._foreach_mul_(params, 1 - lr * wd)
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), 1 - beta2)
        denom = torch._foreach_div(exp_avg_sqs, bias2)
        torch._foreach_sqrt_(denom)
        torch._foreach_add_(denom, group["eps"])
        update = torch._foreach_div(exp_avgs, bias1)
        torch._foreach_div_(update, denom)
        if USE_TENSOR_LR_SCALARS:
            # Same reason as the loop path: alpha= would bake the lr into the graph as a
            # constant, a tensor multiply keeps it a runtime operand.
            torch._foreach_mul_(update, lr)
            torch._foreach_sub_(params, update)
        else:
            torch._foreach_add_(params, update, alpha=-lr)

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

    def _rmsprop_state(self, group):
        """Prologue for the RMSProp path -- the same contract as _adamw_state, minus the
        first moment (that is the whole point of this kind) and minus its bias correction."""
        beta2 = group["betas"][1]
        params, grads, exp_avg_sqs, bias2 = [], [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            params.append(p)
            grads.append(p.grad)
            exp_avg_sqs.append(state["exp_avg_sq"])
            bias2.append(1 - beta2 ** state["step"])
        return params, grads, exp_avg_sqs, bias2

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
        """
        assert group["weight_decay"] == 0.0, "rmsprop groups are weight-decay free by design"
        params, grads, exp_avg_sqs, bias2 = self._rmsprop_state(group)
        if not params:
            return
        lr = group["lr"]
        if USE_TENSOR_LR_SCALARS:
            lr = self._device_scalar(group, "lr", lr, params[0])
        beta2 = group["betas"][1]
        if USE_FOREACH_OPTIM:
            torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), 1 - beta2)
            denom = torch._foreach_div(exp_avg_sqs, bias2)
            torch._foreach_sqrt_(denom)
            torch._foreach_add_(denom, group["eps"])
            update = torch._foreach_div(grads, denom)
            if USE_TENSOR_LR_SCALARS:
                torch._foreach_mul_(update, lr)
                torch._foreach_sub_(params, update)
            else:
                torch._foreach_add_(params, update, alpha=-lr)
        else:
            for p, grad, exp_avg_sq, b2 in zip(params, grads, exp_avg_sqs, bias2):
                exp_avg_sq.lerp_(grad.square(), 1 - beta2)
                update = grad / ((exp_avg_sq / b2).sqrt() + group["eps"])
                if USE_TENSOR_LR_SCALARS:
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
        # Read-in into the stacked buffers. The loop form costs 2 launches per parameter
        # (one grad copy, one param copy) purely to gather the group into the contiguous
        # buffer the batched Newton-Schulz below already wants; _foreach_copy_ does each
        # side in one. unbind(0) is a pure view op with no device work, and this is the
        # same construct the write-back at the bottom of this method already relies on.
        if USE_FOREACH_OPTIM:
            g_rows = list(g.unbind(0))
            grads = [p.grad for p in params]
            if all(gr is not None for gr in grads):
                torch._foreach_copy_(g_rows, grads)
            else:
                # Mixed case: keep the exact per-parameter semantics (zero the row for a
                # missing grad) rather than batching something subtly different.
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
        momentum_buffer = state["momentum_buffer"]
        second = state["second_momentum_buffer"]

        # T1.7: the whole math body as ONE compiled graph. Checked before the NKI path
        # because the two are alternatives, not layers -- a compiled region cannot
        # contain an nki.jit call (Dynamo treats it as opaque and splits the graph around
        # it, which is exactly how the Mamba2 SSD kernel lost 2.14x under compile), so
        # --compile-muon deliberately gives up the NKI fusion in exchange for whole-body
        # fusion. Gated on USE_TENSOR_LR_SCALARS for the same reason use_nki is: the
        # compiled body always carries lr as an fp32 device operand.
        # On a non-neuron device the body runs UNCOMPILED, which is not a fallback but
        # the thing the equivalence gate needs: it isolates the source rewrite (foreach-
        # free / functional / device-scalar) from the compiler, and that half is required
        # to be bit-exact. research/scratch/compile_optim_equiv.py --device cpu.
        if (USE_COMPILE_MUON and USE_TENSOR_LR_SCALARS and w.dtype == torch.float32):
            ratio = max(1.0, shape[-2] / shape[-1]) ** 0.5
            red_dim = -1 if shape[-2] >= shape[-1] else -2
            ns_steps = min(group["ns_steps"], len(POLAR_EXPRESS_COEFFS))
            key = (g.shape[-2] > g.shape[-1], ns_steps, red_dim, tuple(g.shape))
            fn = _MUON_COMPILED.get(key)
            if fn is None:
                fn = (torch.compile(_muon_math, backend="neuron", dynamic=False)
                      if p0.device.type == "neuron" else _muon_math)
                _MUON_COMPILED[key] = fn
                # Announce the branch actually taken, for the same reason ssd_scan does:
                # a silent fallback to the eager/NKI body would make an A/B measure nothing.
                print0(f"muon step: COMPILED body for {tuple(g.shape)} "
                       f"(tall={key[0]} ns_steps={ns_steps} red_dim={red_dim}) "
                       f"[specialisations so far: {len(_MUON_COMPILED)}]")
            lr_t, lrwd_t, mom_t, one_mom_t = self._muon_compile_scalars(group, p0, ratio)
            w, momentum_buffer, second = fn(
                w, g, momentum_buffer, second, mom_t, one_mom_t, lr_t, lrwd_t,
                key[0], ns_steps, red_dim, g.size(red_dim),
                1 - group["beta2"], COMPUTE_DTYPE == torch.bfloat16)
            # All three are scratch or state the caller owns, so rebinding is free --
            # no copy back (see the _muon_math docstring).
            state["stacked_params"] = w
            state["momentum_buffer"] = momentum_buffer
            state["second_momentum_buffer"] = second
            torch._foreach_copy_(params, list(w.unbind(0)))
            return

        # Fused-elementwise path: same math, fewer launches. Gated on
        # USE_TENSOR_LR_SCALARS because the fused update always carries lr as an fp32
        # device operand, which is the tensor-lr semantics; with the flag off eager
        # would fold lr*ratio in float64 instead and the two would not be bit-equal.
        use_nki = (USE_NKI_MUON and _HAS_NKI and p0.device.type == "neuron"
                   and USE_TENSOR_LR_SCALARS and COMPUTE_DTYPE == torch.bfloat16
                   and w.dtype == torch.float32 and shape[-1] <= 8192
                   and min(shape[-2], shape[-1]) <= 8192)

        momentum_buffer.lerp_(g, 1 - group["momentum"])
        g = g.lerp_(momentum_buffer, group["momentum"])
        x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        if use_nki:
            # Newton-Schulz with the three matmuls untouched and the five elementwise
            # ops per iteration collapsed into two kernels: 8 launches -> 5. Every
            # tensor crossing a kernel boundary is a contiguous matmul or kernel output,
            # so `reshape(-1, last)` is a pure view and the flattening is free.
            coef = self._ns_coef(p0)
            tall = x.shape[-2] > x.shape[-1]
            for i in range(min(group["ns_steps"], len(POLAR_EXPRESS_COEFFS))):
                ca, cb, cc = coef[i, 0], coef[i, 1], coef[i, 2]
                sym = x.mT @ x if tall else x @ x.mT
                k = sym.shape[-1]
                m = _muon_ns_poly(sym.reshape(-1, k), (sym @ sym).reshape(-1, k),
                                  cb, cc).view_as(sym)
                prod = x @ m if tall else m @ x
                n = x.shape[-1]
                x = _muon_ns_axpy(x.reshape(-1, n), prod.reshape(-1, n), ca).view_as(x)
        elif x.shape[-2] > x.shape[-1]:
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

        ratio = max(1.0, shape[-2] / shape[-1]) ** 0.5
        if use_nki:
            # 10 launches -> 2 (one scalar upload, one kernel). The upload replaces the
            # _device_scalar copy the eager path already pays for, so it is not new.
            klr, klw = self._muon_lrwd(group, p0, ratio)
            n = g.shape[-1]
            w = _muon_update(g.reshape(-1, n), w.reshape(-1, n), klr, klw).view_as(w)
            # stacked_params is scratch -- fully refilled from the live params at the top
            # of every step -- so rebinding it to the kernel's output is free and needs
            # no copy back.
            state["stacked_params"] = w
        else:
            lr_base = group["lr"]
            if USE_TENSOR_LR_SCALARS:
                lr_base = self._device_scalar(group, "lr", lr_base, p0)
            lr = lr_base * ratio
            wd = group["weight_decay"]
            mask = (g * w) >= 0
            w.sub_(lr * g + lr * wd * w * mask)
        torch._foreach_copy_(params, list(w.unbind(0)))

    @torch.no_grad()
    def step(self, update_adamw: bool = True):
        """`update_adamw=False` (T1.4) runs only the Muon groups this step and leaves the AdamW
        groups' gradients untouched, so they keep accumulating into the same .grad tensors."""
        for group in self.param_groups:
            if group["kind"] == "adamw":
                if not update_adamw:
                    continue
                if USE_FOREACH_OPTIM:
                    self._adamw_step_foreach(group)
                else:
                    self._adamw_step(group)
            elif group["kind"] == "rmsprop":
                # T3.1: gated by the same update_adamw flag as the AdamW groups. Both are
                # diagonal/elementwise optimizers over gradients that keep accumulating in
                # place, so T1.4's "skip the cheap groups on some steps" transformation is
                # exactly as valid here, and treating them alike keeps zero_grad_selective /
                # adamw_param_ids from ever leaving one of the two half-updated.
                if not update_adamw:
                    continue
                self._rmsprop_step(group)
            elif group["kind"] == "muon":
                self._muon_step(group)
            else:
                raise ValueError(f"unknown optimizer kind: {group['kind']}")

    @torch.no_grad()
    def zero_grad_selective(self, zero_adamw: bool = True) -> None:
        """T1.4: like `zero_grad(set_to_none=True)`, but can leave the AdamW groups' gradients in
        place so they accumulate across a skipped update. Never severs an accumulation by
        accident -- the AdamW branch is skipped entirely rather than zeroed with a different
        mechanism."""
        for group in self.param_groups:
            if group["kind"] in ADAM_LIKE_KINDS and not zero_adamw:
                continue
            for p in group["params"]:
                p.grad = None

    def adamw_param_ids(self) -> frozenset[int]:
        """T1.4: identities of every parameter in an AdamW (or RMSProp) group, so the per-step
        all_reduce can skip exactly those on a non-update step."""
        return frozenset(id(p) for g in self.param_groups if g["kind"] in ADAM_LIKE_KINDS
                         for p in g["params"])


def split_clip_params(model: "GPT") -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """(main, ngram) parameter lists for clip_grad_norm_ -- see NGRAM_VE_CLIP_EXCLUDE.

    GRAD_CLIP is a GLOBAL norm over every gradient at once, so folding the ~200M-parameter
    hashed n-gram tables into it would raise the total norm, shrink clip_coef, and thereby
    change the effective learning rate of every OTHER tensor in the model -- a confound with
    nothing to do with whether n-gram tables help. Splitting them off leaves the main clip
    bit-identical to the historical `clip_grad_norm_(model.parameters(), ...)` (same tensors,
    in the same order) whenever there are no n-gram tables, and still norm-bounds the tables.
    """
    ngram_ids = {id(p) for p in model.ngram_embeds.parameters()} if NGRAM_VE_CLIP_EXCLUDE else set()
    # T4.2: the [V, V] output bigram table is 67M more parameters with exactly the same problem --
    # a big, sparse-per-step gradient that would drag the global clip coefficient down and silently
    # rescale every other tensor's update. Rides the same toggle, since it is the same argument.
    if NGRAM_VE_CLIP_EXCLUDE and model.out_bigram is not None:
        ngram_ids = ngram_ids | {id(model.out_bigram.weight)}
    # T3.2: same argument again for Engram's own tables (67M at the default mult=32, 2 tables). The
    # projections and the conv stay in the MAIN clip -- they are small dense matrices whose gradient
    # is dense every step, i.e. exactly the population GRAD_CLIP was calibrated on.
    if NGRAM_VE_CLIP_EXCLUDE and model.engram is not None:
        ngram_ids = ngram_ids | {id(p) for p in model.engram.tables.parameters()}
    if not ngram_ids:
        return list(model.parameters()), []
    main, ngram = [], []
    for p in model.parameters():
        (ngram if id(p) in ngram_ids else main).append(p)
    return main, ngram


def sync_gradients(model: nn.Module, skip_param_ids: frozenset[int] | None = None) -> None:
    """`skip_param_ids` (T1.4) defers these parameters' all_reduce to a later step. Exact, not an
    approximation: ReduceOp.AVG is linear, so averaging the sum of K locally-accumulated steps
    equals summing K averaged steps."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in model.parameters():
        if p.grad is not None and not (skip_param_ids and id(p) in skip_param_ids):
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


# -----------------------------------------------------------------------------
# T1.2 -- flat gradient buffer: one all_reduce/step instead of one per parameter.
#
# sync_gradients() above issues one dist.all_reduce per parameter (62 at the locked
# shape). This builds a single flat fp32 buffer sized to hold every trainable
# parameter's gradient, points each p.grad at a VIEW into that buffer (so autograd's
# normal in-place accumulation -- "if p.grad is not None: p.grad += new_grad" --
# writes directly into the shared storage), and all-reduces the whole buffer once.
# Must be built exactly once, before the first backward() of the run (grad-accum
# micro-steps within one optimizer step then keep accumulating into the same views),
# and the training loop must use optimizer.zero_grad(set_to_none=False) so the zero
# between optimizer steps zeroes the existing view tensors in place instead of
# replacing them with None (which would sever the view and silently fall back to
# fresh per-tensor grad tensors on the next backward). Verified bit-equal against the
# per-tensor path in research/scratch/grad_bucket_equiv.py; only wired in behind
# --grad-bucket (default OFF) until a real hardware A/B decides whether the ~1%
# expected win is real. See the T1.2 section of trn2-experiment-results.md.
def build_flat_grad_buffer(model: nn.Module) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Returns (flat_buffer, params) and leaves every params[i].grad pointing at a
    zeroed view into flat_buffer. Call once, before the first backward() of the run."""
    params = [p for p in model.parameters() if p.requires_grad]
    device = params[0].device if params else torch.device("cpu")
    total = sum(p.numel() for p in params)
    flat = torch.zeros(total, dtype=torch.float32, device=device)
    offset = 0
    for p in params:
        n = p.numel()
        view = flat[offset:offset + n].view_as(p)
        p.grad = view
        offset += n
    assert offset == total
    return flat, params


def sync_gradients_flat(flat_grad_buffer: torch.Tensor) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    dist.all_reduce(flat_grad_buffer, op=dist.ReduceOp.AVG)


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
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    flat_grad_buffer.mul_(clip_coef_clamped)
    return total_norm


# -----------------------------------------------------------------------------
# Checkpoint / scoring hook

ARCH_TOGGLE_NAMES = [
    "PACK_FACTOR", "INIT_SCALE", "USE_VE", "WINDOW_PATTERN", "SHORT_WINDOW_FRAC",
    "ROPE_BASE", "USE_LOCAL_CONV", "LOCAL_CONV_KERNEL", "TRAIN_LOCAL_CONV", "USE_GATED_MLP",
    "USE_WEIGHT_CACHE", "WARMDOWN_RATIO", "FINAL_LR_FRAC", "WARMUP_STEPS",
    "WD_DECAY_TO_ZERO", "MUON_MOMENTUM_WARMUP_STEPS", "USE_DOC_MASK", "VE_WD",
    "USE_SELECTIVE_GATE", "SELECTIVE_GATE_CHANNELS", "USE_FUSED_QKV",
    "USE_CAUSAL_FASTPATH", "USE_STOCHASTIC_ROUNDING", "USE_TRN2_COMPILER_FLAGS",
    "USE_REQUEUE_PACKER", "USE_XSA", "USE_VALUE_RESIDUAL", "USE_MUDD", "USE_DCMHA",
    "USE_MAMBA2", "MAMBA2_PATTERN", "MAMBA2_HEAD_DIM", "MAMBA2_D_STATE",
    "MAMBA2_CHUNK", "MAMBA2_EXPAND", "MAMBA2_N_GROUPS", "MAMBA2_CONV_KERNEL",
    "USE_RWKV", "RWKV_PATTERN", "RWKV_HEAD_DIM", "RWKV_CHUNK", "RWKV_CMIX_EXPAND",
    "USE_FP8_LINEAR",
    # T0.4 resweep knobs. None of these changes a tensor's name or shape, so every one of them
    # is exactly the "loads silently with quietly wrong behavior" class this snapshot exists to
    # catch -- ATTN_CPROJ_INIT_SCALE and LOGIT_SOFTCAP especially, since they change what the
    # weights MEAN without changing what they look like.
    "MATRIX_LR", "EMBEDDING_LR", "UNEMBEDDING_LR", "SCALAR_LR", "WEIGHT_DECAY", "GRAD_CLIP",
    "LOGIT_SOFTCAP", "ATTN_CPROJ_INIT_SCALE", "MUON_NS_STEPS",
    "MUON_MOMENTUM_COOLDOWN_FRAC", "MUON_MOMENTUM_FINAL",
    "ADAM_EVERY_N", "ADAM_EVERY_N_LR_COMP",
    # T0.2/T0.3. DEPTH/N_EMBD/HEAD_DIM/MLP_RATIO are all *also* recoverable from
    # model_config (n_layer / n_embd / n_embd//n_head / mlp_ratio), which is what
    # load_for_eval actually rebuilds from -- they are listed here only so a mismatch
    # is reported by name rather than as a wall of shape errors. ATTN_SCALE is the one
    # that is NOT in model_config and changes what the weights mean without changing
    # any shape, i.e. exactly the silent class this snapshot exists for.
    "DEPTH", "N_EMBD", "HEAD_DIM", "MLP_RATIO", "ATTN_SCALE",
    # T3.1 hashed n-gram value embeddings. NGRAM_VE_BF16 is the interesting one to list
    # here: load_state_dict silently CASTS a mismatched dtype rather than erroring, so a
    # bf16-vs-fp32 mismatch is exactly the quiet class this snapshot exists for. The
    # optimizer-side knobs (LR/BETA2/FLAT_LR/OPT/CLIP_EXCLUDE) leave no trace in the
    # weights at all, which is the other half of the same reason.
    "USE_NGRAM_VE", "NGRAM_VE_LAYERS", "NGRAM_VE_TABLE_MULT", "NGRAM_VE_TRIGRAM", "NGRAM_VE_DIM",
    "NGRAM_VE_BF16",
    "NGRAM_VE_LR", "NGRAM_VE_BETA2", "NGRAM_VE_FLAT_LR", "NGRAM_VE_OPT",
    "NGRAM_VE_CLIP_EXCLUDE",
    # T4.2 Aster output bigram logit table, T3.9 Recursive one-liners. RELU2_TAU,
    # USE_MLP_SANDWICH_NORM, HEAD_GATE_NORM/MUL/BIAS and QK_SHIFT_BETA are all pure
    # silent-behavior knobs -- they change what the same-shaped weights MEAN, which is exactly
    # the class this snapshot exists to catch.
    "USE_OUT_BIGRAM", "OUT_BIGRAM_BF16", "OUT_BIGRAM_GATE_BIAS", "OUT_BIGRAM_LR",
    "RELU2_TAU", "USE_MLP_SANDWICH_NORM",
    "USE_HEAD_GATE", "HEAD_GATE_NORM", "HEAD_GATE_CHANNELS", "HEAD_GATE_MUL", "HEAD_GATE_BIAS",
    "USE_OUT_POOL", "OUT_POOL_LAYERS", "USE_QK_SHIFT", "QK_SHIFT_BETA", "QK_SHIFT_FREEZE",
    "USE_X0_GATE", "USE_BLOCK_NUDGE", "BLOCK_NUDGE_SITE", "USE_QKV_NORM_CSE",
    "USE_PARALLEL_BLOCK", "PARALLEL_BLOCK_COALESCE",
    # T3.2 Engram-lite. ENGRAM_SITE / ENGRAM_ORDERS / ENGRAM_HEADS / ENGRAM_TABLE_MULT /
    # ENGRAM_MEM_DIM / ENGRAM_KEY_DIM / ENGRAM_CONV_KERNEL all change a tensor's shape or its
    # existence; ENGRAM_SHARE_NGRAM_TABLES changes which tensors exist at all; ENGRAM_BF16 is the
    # silently-CAST-on-load class, exactly like NGRAM_VE_BF16 above.
    "USE_ENGRAM", "ENGRAM_SITE", "ENGRAM_ORDERS", "ENGRAM_HEADS", "ENGRAM_TABLE_MULT",
    "ENGRAM_MEM_DIM", "ENGRAM_KEY_DIM", "ENGRAM_CONV_KERNEL", "ENGRAM_SHARE_NGRAM_TABLES",
    "ENGRAM_BF16",
    # Data ORDER, not a tensor -- but it changes which tokens the cooldown steps saw, which is
    # exactly the kind of thing that must be recoverable from a checkpoint after the fact.
    "USE_EPOCH_SHUFFLE", "EPOCH_SHUFFLE_SEED",
    # T2.1 batch-size ramp. Pure training-schedule knobs -- no tensor changes name, shape or
    # dtype -- but they change how many tokens each recorded step saw and at what lr, so a
    # checkpoint trained with a ramp and reloaded without one is silently mis-described.
    "BATCH_RAMP", "BATCH_RAMP_GRAD_ACCUM", "BATCH_RAMP_BOUNDS", "BATCH_RAMP_LR_EXP",
    # T2.2 weight EMA. Also shape-neutral, but it decides whether the checkpoint has an
    # "ema_state" key at all -- i.e. WHICH weights load_for_eval will hand the scorer.
    "USE_WEIGHT_EMA", "WEIGHT_EMA_DECAYS", "WEIGHT_EMA_START_FRAC",
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
                          "MAMBA2_CONV_KERNEL",
                          # Same story for RWKV: the pattern swaps attn+mlp <-> tmix+cmix
                          # per layer, head_dim resizes time_decay/time_faaaa (H = n_embd
                          # / RWKV_HEAD_DIM) and the ve_gate output width, and cmix_expand
                          # resizes cmix.key/cmix.value. RWKV_CHUNK is the exception, like
                          # MAMBA2_CHUNK: pure blocking of identical math, no tensor
                          # changes shape, so it is deliberately NOT in this set.
                          "USE_RWKV", "RWKV_PATTERN", "RWKV_HEAD_DIM",
                          "RWKV_CMIX_EXPAND",
                          # T3.1: USE_NGRAM_VE adds/removes the ngram_embeds.* tables AND the
                          # attn.ve_gate_bi/ve_gate_tri weights, NGRAM_VE_TRIGRAM adds/removes
                          # the "t<layer>" tables and their gates, and NGRAM_VE_TABLE_MULT
                          # resizes every table's row count -- all three break strict loading.
                          "USE_NGRAM_VE", "NGRAM_VE_TRIGRAM", "NGRAM_VE_TABLE_MULT",
                          "NGRAM_VE_DIM",
                          # T4.2/T3.9: each of these adds, removes or resizes a parameter tensor --
                          # out_bigram.weight + its gate, attn.head_gate.weight/bias (whose input
                          # width IS HEAD_GATE_CHANNELS), out_pool_lambdas (length OUT_POOL_LAYERS)
                          # and the per-block qk_shift_beta. OUT_BIGRAM_BF16 is deliberately not
                          # here for the same reason NGRAM_VE_BF16 is not: load_state_dict casts a
                          # dtype mismatch silently instead of erroring.
                          "USE_OUT_BIGRAM", "USE_HEAD_GATE", "HEAD_GATE_CHANNELS",
                          "USE_OUT_POOL", "OUT_POOL_LAYERS", "USE_QK_SHIFT",
                          # USE_X0_GATE adds one x0_gate_s scalar per block; USE_BLOCK_NUDGE adds
                          # one frozen block_nudge scalar per block. Both are state_dict keys even
                          # though the nudge never trains, so a mismatch is a load-time error.
                          "USE_X0_GATE", "USE_BLOCK_NUDGE",
                          # USE_PARALLEL_BLOCK adds par_lambda_a/par_lambda_m per block;
                          # PARALLEL_BLOCK_COALESCE replaces c_q/c_k/c_v/c_fc with one c_in.
                          "USE_PARALLEL_BLOCK", "PARALLEL_BLOCK_COALESCE",
                          # T3.2: USE_ENGRAM adds engram.*; ORDERS/HEADS decide how many tables
                          # exist and hence w_k/w_v's input width; TABLE_MULT and MEM_DIM resize
                          # the tables (MEM_DIM also resizes w_k/w_v's input); KEY_DIM resizes
                          # w_k's output and adds/removes w_q; CONV_KERNEL resizes conv.weight
                          # (and 0 removes it); SHARE_NGRAM_TABLES removes the tables entirely and
                          # changes w_k/w_v's input width to the shared rows'. ENGRAM_BF16 is
                          # excluded for the same reason NGRAM_VE_BF16 is: a silent cast, not an
                          # error. ENGRAM_SITE is excluded too -- it moves WHERE the same-shaped
                          # module is read, so it is silent-behavior, not shape-critical.
                          "USE_ENGRAM", "ENGRAM_ORDERS", "ENGRAM_HEADS", "ENGRAM_TABLE_MULT",
                          "ENGRAM_MEM_DIM", "ENGRAM_KEY_DIM", "ENGRAM_CONV_KERNEL",
                          "ENGRAM_SHARE_NGRAM_TABLES"}
# USE_FP8_LINEAR is deliberately NOT in the set above: Fp8Linear is a Linear subclass with
# the identical single `weight` parameter, so flipping it changes only the arithmetic used
# inside MLP.forward, never a tensor's name or shape. That makes it exactly the
# "silent-behavior" class of mismatch the snapshot above exists to catch, which is why it
# still belongs in ARCH_TOGGLE_NAMES.


def current_toggles() -> dict:
    g = globals()
    return {name: g[name] for name in ARCH_TOGGLE_NAMES}


def save_checkpoint(path: Path, model: GPT, meta: dict, ema_state: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": asdict(model.config),
        "toggles": current_toggles(),
        "meta": meta,
    }
    # T2.2: BOTH weight sets in one file, so nothing downstream has to guess which checkpoint is
    # which and the raw weights are never destroyed by the averaging. The key is absent entirely
    # when USE_WEIGHT_EMA is off, which is what keeps load_for_eval's behaviour unchanged.
    if ema_state is not None:
        payload["ema_state"] = {k: v.detach().cpu() for k, v in ema_state.items()}
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


EVAL_MIN_SEQ_LEN = 8192  # C1/T0.7: floor for load_for_eval's rotary/window-mask tables, see below.


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
    if USE_DOC_MASK:
        BOS_TOKEN_ID = ensure_tokenizer(build_if_missing=False).get_bos_token_id()
    payload = torch_load_weights(checkpoint_path, map_location="cpu")
    config = GPTConfig(**payload["model_config"])
    if config.sequence_len < EVAL_MIN_SEQ_LEN:
        config.sequence_len = EVAL_MIN_SEQ_LEN
    model = GPT(config)
    model.init_weights()
    # T2.2: prefer the averaged weights when the checkpoint carries them. Same signature, same
    # return type, same strict load -- "ema_state" only exists if USE_WEIGHT_EMA was on for that
    # run, so for every checkpoint written before T2.2 this is exactly the old single line.
    state = payload["model_state"]
    if EVAL_PREFER_EMA and "ema_state" in payload:
        state = payload["ema_state"]
        print(f"load_for_eval: using EMA weights from {checkpoint_path} "
              f"(raw weights still available under 'model_state')")
    model.load_state_dict(state, strict=True)
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
        return grad_accum_ref, 1.0
    gas = BATCH_RAMP_GRAD_ACCUM[batch_ramp_stage(progress)]
    return gas, (gas / grad_accum_ref) ** BATCH_RAMP_LR_EXP


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
        raise KeyError(f"EMA tracks parameters absent from state_dict: {missing}")
    for n, t in by_name.items():
        state[n] = t
    return state


def muon_momentum(step: int, progress: float = 0.0) -> float:
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
        # 0 while progress <= 1 - FRAC, ramping to 1 at progress == 1.
        cd = min(1.0, max(0.0, (progress - (1.0 - MUON_MOMENTUM_COOLDOWN_FRAC))
                          / MUON_MOMENTUM_COOLDOWN_FRAC))
        m = (1 - cd) * m + cd * MUON_MOMENTUM_FINAL
    return m


def main() -> None:
    global USE_LOCAL_CONV, LRM_QUANTUM, USE_TENSOR_LR_SCALARS, USE_FOREACH_OPTIM, USE_MUDD, USE_DCMHA, USE_MAMBA2, MAMBA2_PATTERN, USE_NKI_SSD  # flags below default off these constants
    global USE_FP8_LINEAR
    global MAMBA2_EXPAND, MAMBA2_D_STATE, MAMBA2_HEAD_DIM, MAMBA2_CHUNK, MAMBA2_N_GROUPS, MAMBA2_CONV_KERNEL
    global USE_RWKV, RWKV_PATTERN, RWKV_HEAD_DIM, RWKV_CHUNK, RWKV_CMIX_EXPAND
    global LOCAL_CONV_KERNEL, TRAIN_LOCAL_CONV
    # Throughput->capacity track, Probe 3: USE_VE needs to be reachable from the CLI so the
    # "maximally plain block" ablate-down arm can be measured. Default is the locked True, so
    # passing nothing changes nothing. Declared here because --ve's argparse default reads it.
    global USE_VE
    # T0.4/T1.4 resweep knobs. These must be declared global HERE, at the top of main(), not next
    # to their assignments further down: the argparse defaults below READ them, and Python rejects
    # a `global` statement that follows any use of the name in the same function
    # ("SyntaxError: name 'MATRIX_LR' is used prior to global declaration").
    global MATRIX_LR, EMBEDDING_LR, UNEMBEDDING_LR, SCALAR_LR, WEIGHT_DECAY, WD_DECAY_TO_ZERO
    global FINAL_LR_FRAC, WARMDOWN_RATIO, INIT_SCALE, ATTN_CPROJ_INIT_SCALE, LOGIT_SOFTCAP
    global MUON_NS_STEPS, MUON_MOMENTUM_COOLDOWN_FRAC, MUON_MOMENTUM_FINAL
    global ADAM_EVERY_N, ADAM_EVERY_N_LR_COMP, PACK_FACTOR, GRAD_CLIP
    # T0.2/T0.3 capacity-shape knobs. All read inside build_config()/GPT(config), i.e. well before
    # the first step, so none of them can add a per-step graph constant.
    global DEPTH, N_EMBD, HEAD_DIM, MLP_RATIO, ATTN_SCALE, SEQ_LEN
    # T3.1. USE_NGRAM_VE / NGRAM_VE_TABLE_MULT / NGRAM_VE_TRIGRAM / NGRAM_VE_BF16 are read by
    # GPT.__init__ and CausalSelfAttention.__init__ (before GPT(config)); the rest are read by
    # setup_optimizer() or once per step in the training loop, all well before any device op
    # that could bake them in as a per-step graph constant.
    global USE_NGRAM_VE, NGRAM_VE_TABLE_MULT, NGRAM_VE_TRIGRAM, NGRAM_VE_BF16, NGRAM_VE_DIM
    global NGRAM_VE_LAYERS
    global NGRAM_VE_LR, NGRAM_VE_BETA2, NGRAM_VE_FLAT_LR, NGRAM_VE_OPT, NGRAM_VE_CLIP_EXCLUDE
    # T4.2 / T3.9. USE_OUT_BIGRAM, OUT_BIGRAM_BF16, OUT_BIGRAM_GATE_BIAS, USE_HEAD_GATE,
    # HEAD_GATE_CHANNELS, HEAD_GATE_BIAS, USE_OUT_POOL, OUT_POOL_LAYERS, USE_QK_SHIFT and
    # QK_SHIFT_BETA are read by GPT.__init__ / Block.__init__ / CausalSelfAttention.__init__, i.e.
    # before GPT(config); OUT_BIGRAM_LR by setup_optimizer(). RELU2_TAU, USE_MLP_SANDWICH_NORM,
    # HEAD_GATE_NORM and HEAD_GATE_MUL are read inside forward() -- as COMPILE-TIME Python
    # constants that never change across steps, so they are baked into the graph exactly once
    # (the "no new per-step Python scalar" rule is about values that CHANGE per step).
    global USE_OUT_BIGRAM, OUT_BIGRAM_BF16, OUT_BIGRAM_GATE_BIAS, OUT_BIGRAM_LR
    global RELU2_TAU, USE_MLP_SANDWICH_NORM
    global USE_HEAD_GATE, HEAD_GATE_NORM, HEAD_GATE_CHANNELS, HEAD_GATE_MUL, HEAD_GATE_BIAS
    global USE_OUT_POOL, OUT_POOL_LAYERS, USE_QK_SHIFT, QK_SHIFT_BETA, QK_SHIFT_FREEZE
    global USE_X0_GATE, USE_BLOCK_NUDGE, BLOCK_NUDGE_SITE, USE_QKV_NORM_CSE, USE_PARALLEL_BLOCK
    global PARALLEL_BLOCK_COALESCE, USE_EPOCH_SHUFFLE, EPOCH_SHUFFLE_SEED
    # T3.2. Every one of these is read at MODULE-CONSTRUCTION time (or, for ENGRAM_SITE, once per
    # forward as a Python int compared against the loop index), so none of them is a per-step
    # changing scalar reaching a device op.
    global USE_ENGRAM, ENGRAM_SITE, ENGRAM_ORDERS, ENGRAM_HEADS, ENGRAM_TABLE_MULT
    global ENGRAM_MEM_DIM, ENGRAM_KEY_DIM, ENGRAM_CONV_KERNEL, ENGRAM_SHARE_NGRAM_TABLES
    global ENGRAM_BF16
    # T2.1. Read once per step by batch_ramp_grad_accum() to pick a Python int loop count and a
    # Python float lr scale -- neither reaches a device op as a constant (the loss scaling, which
    # would, goes through inv_gas_dev instead).
    global BATCH_RAMP, BATCH_RAMP_GRAD_ACCUM, BATCH_RAMP_BOUNDS, BATCH_RAMP_LR_EXP
    global USE_WEIGHT_EMA, WEIGHT_EMA_DECAYS, WEIGHT_EMA_START_FRAC
    # T1.7. Declared HERE, not next to its assignment further down, because --compile-muon's
    # argparse default READS it ("1" if USE_COMPILE_MUON else "0"), and Python forbids a `global`
    # statement that follows any use of the name in the same function. Same rule that put
    # MATRIX_LR et al. in this block; --nki-muon dodged it by hardcoding a literal default instead.
    global USE_COMPILE_MUON
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
    parser.add_argument("--nki-muon", action=argparse.BooleanOptionalAction,
                        # literal default, like the other --nki-* flags: this function
                        # rebinds USE_NKI_MUON as a global, so it cannot also read it here.
                        default=os.environ.get("NEURON_COMPETITION_R1_NKI_MUON", "1") != "0",
                        help="fuse the elementwise ops wrapping Muon's Newton-Schulz matmuls, plus the "
                             "sign-gated weight update, into three NKI kernels (Neuron only). The matmuls "
                             "themselves stay in aten. Default ON: removes 92 of Muon's 320 per-step device "
                             "launches (-25.5%% off the Muon window) on a step whose optimizer window is "
                             "~54%% inter-launch gap, with BIT-IDENTICAL update math "
                             "(research/scratch/muon_nki_equiv.py: 0 differing elements). Real paired 900s "
                             "A/B: 1019 steps/1.089758 vs 1004/1.090677 -- pass --no-nki-muon for the plain "
                             "eager comparison path.")
    parser.add_argument("--compile-muon", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_COMPILE_MUON",
                                               "1" if USE_COMPILE_MUON else "0") != "0",
                        help="T1.7: run Muon's per-group math body inside "
                             "torch.compile(backend='neuron') instead of ~80 eager aten launches "
                             "(Neuron only). Mutually exclusive with --nki-muon by construction -- a "
                             "compiled region cannot contain an nki.jit call -- so this trades the NKI "
                             "elementwise fusion for whole-body fusion. See _muon_math.")
    parser.add_argument("--local-conv", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_LOCAL_CONV", "1" if USE_LOCAL_CONV else "0") != "0",
                        help="depthwise causal conv1d before attention in each block. Provably dead "
                             "compute as wired (zero weight, in no optimizer group -- see the comment on "
                             "USE_LOCAL_CONV), so --no-local-conv is a bit-identical model; exposed as a "
                             "flag so the A/B runs from one identical binary.")
    parser.add_argument("--ve", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_VE", "1" if USE_VE else "0") != "0",
                        help="ResFormer-style value embeddings on alternating layers (has_ve), "
                             "input-gated into attention's V stream. Default ON (the locked value); "
                             "exposed for the throughput->capacity track's plain-block ablate-down "
                             "arm. NOTE --no-ve also removes the n-gram value embeddings, because "
                             "has_ngram_bigram() requires USE_VE and has_ve(layer) -- so --no-ve "
                             "is strictly a step-time diagnostic, not a candidate config.")
    parser.add_argument("--tensor-lr-scalars", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_TENSOR_LR_SCALARS", "1" if USE_TENSOR_LR_SCALARS else "0") != "0",
                        help="hold the per-step lr in a device tensor rather than a Python float, so a "
                             "changing LR schedule stops forcing a fresh neuronx-cc compile every step")
    parser.add_argument("--foreach-optim", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_FOREACH_OPTIM", "1" if USE_FOREACH_OPTIM else "0") != "0",
                        help="batch the optimizer's per-parameter work into torch._foreach_* calls "
                             "instead of a Python loop of single-tensor ops. Bit-identical update "
                             "math; targets the optimizer step's launch/dispatch overhead only.")
    parser.add_argument("--fp8-linear", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_FP8_LINEAR", "1" if USE_FP8_LINEAR else "0") != "0",
                        help="run the MLP's c_fc/c_proj FORWARD matmul in float8_e4m3fn with "
                             "current-scaling (backward stays bf16/fp32). Default OFF -- a real, "
                             "measured DOUBLE negative on this stack: torch._scaled_mm, the only "
                             "numerically correct fp8 GEMM here, is 136521x slower than bf16 "
                             "F.linear (46366 vs 0.340 ms/call, per-call, never amortizes) AND fp8 "
                             "costs 13x the per-matmul error of the bf16 it replaces (0.0374 vs "
                             "0.00287). See the USE_FP8_LINEAR comment; exposed as a flag so the "
                             "A/B runs from one identical binary.")
    parser.add_argument("--ngram-ve", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_NGRAM_VE", "1" if USE_NGRAM_VE else "0") != "0",
                        help="T3.1: hashed bigram (all VE layers) and trigram (first/last VE layer) "
                             "VALUE embeddings, mixed into attention's V through their own zero-init "
                             "per-head gates on disjoint residual channels (unigram 0:32, bigram 32:64, "
                             "trigram 64:96). Tables are zero-init, so this is an exact no-op at step 0 "
                             "and the A/B runs from one identical binary. Strictly causal: a position's "
                             "table row is a hash of tokens t-2..t only.")
    parser.add_argument("--ngram-ve-layers", type=str,
                        default=os.environ.get("NEURON_COMPETITION_R1_NGRAM_VE_LAYERS", NGRAM_VE_LAYERS),
                        help="which layers get a bigram table, comma-separated, negatives counting "
                             "from the end ('-1' = last layer at any DEPTH). Empty = every VE layer, "
                             "Recursive's own allocation. The per-step cost of these tables depends "
                             "on WHICH layer holds them and not on their size: one table on the last "
                             "layer is FASTER than no table at all, while any second table anywhere "
                             "costs ~50ms/micro-step. See NGRAM_VE_LAYERS for the measurement.")
    parser.add_argument("--ngram-ve-table-mult", type=int, default=NGRAM_VE_TABLE_MULT,
                        help="rows per n-gram table = this x padded_vocab_size (8 => 65536 rows, "
                             "41.9M params per table at n_embd=640). Recursive went to 64x; start at 8x.")
    parser.add_argument("--ngram-ve-trigram", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_NGRAM_VE_TRIGRAM", "1" if NGRAM_VE_TRIGRAM else "0") != "0",
                        help="add a trigram table on the first and last VE layer as well as the "
                             "bigram table every VE layer gets. --no-ngram-ve-trigram is the "
                             "bigram-only arm.")
    parser.add_argument("--ngram-ve-dim", type=int,
                        default=int(os.environ.get("NEURON_COMPETITION_R1_NGRAM_VE_DIM", NGRAM_VE_DIM)),
                        help="n-gram table WIDTH (0 = n_embd, gathered straight into v). A throughput "
                             "knob: the measured cost of these tables scales with rows x width in BOTH "
                             "the forward gather and the dense embedding backward, so e.g. 128 is ~5x "
                             "less table to read, scatter into, all-reduce and update, at the price of "
                             "one [tokens,dim]x[dim,n_embd] matmul per table. See NGRAM_VE_DIM.")
    parser.add_argument("--ngram-ve-opt", type=str, default=NGRAM_VE_OPT, choices=("rmsprop", "adamw"),
                        help="optimizer for the n-gram tables. rmsprop (default, Recursive's choice) "
                             "drops the first moment so a rarely-hit row still gets a full-rate "
                             "update; adamw is the fallback arm and matches the unigram VE group.")
    parser.add_argument("--ngram-ve-lr", type=float, default=NGRAM_VE_LR,
                        help="lr for the n-gram tables before the dmodel scale (0 = use --embedding-lr).")
    parser.add_argument("--ngram-ve-beta2", type=float, default=NGRAM_VE_BETA2,
                        help="second-moment decay for the n-gram table optimizer.")
    parser.add_argument("--ngram-ve-flat-lr", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_NGRAM_VE_FLAT_LR", "1" if NGRAM_VE_FLAT_LR else "0") != "0",
                        help="hold the n-gram tables' lr FLAT (warmup only, no warmdown), per "
                             "Recursive's \"sparse tables benefit from full-rate training\". "
                             "--no-ngram-ve-flat-lr puts them back on the normal cooldown.")
    parser.add_argument("--ngram-ve-bf16", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_NGRAM_VE_BF16", "1" if NGRAM_VE_BF16 else "0") != "0",
                        help="store the n-gram tables (and their optimizer state) in bf16 rather "
                             "than fp32 -- halves their HBM footprint and all-reduce volume.")
    parser.add_argument("--ngram-ve-clip-exclude", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_NGRAM_VE_CLIP_EXCLUDE", "1" if NGRAM_VE_CLIP_EXCLUDE else "0") != "0",
                        help="keep the n-gram tables out of the GLOBAL clip_grad_norm_ and clip them "
                             "as their own group, so adding them cannot change the effective lr of "
                             "every other tensor through clip_coef. See split_clip_params.")
    # Shorthand for the many scalar knobs below: one flag whose default is the module-level
    # constant, overridable by NEURON_COMPETITION_R1_<FLAG>. Defined here rather than next to the
    # T0.4 block it was written for because it is a local of main(): every call must come after
    # this `def` statement executes, or Python raises UnboundLocalError.
    def _hp(flag: str, default, typ, helptext: str) -> None:
        env = "NEURON_COMPETITION_R1_" + flag.lstrip("-").replace("-", "_").upper()
        parser.add_argument(flag, type=typ, default=typ(os.environ.get(env, default)), help=helptext)

    # T4.2: Aster's output bigram logit table.
    parser.add_argument("--out-bigram", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_OUT_BIGRAM", "1" if USE_OUT_BIGRAM else "0") != "0",
                        help="T4.2: a gradient-trained [vocab, vocab] table indexed by the CURRENT "
                             "token whose row is added to the next-token logits, gated per token by "
                             "sigmoid(Linear(n_embd->1)). Zero-init table => exact no-op at step 0. "
                             "67.1M bf16 parameters. Distinct from the frozen count-based mixer that "
                             "was closed as a negative on 2026-09-12. See USE_OUT_BIGRAM.")
    parser.add_argument("--out-bigram-bf16", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_OUT_BIGRAM_BF16", "1" if OUT_BIGRAM_BF16 else "0") != "0",
                        help="store the [vocab, vocab] output bigram table in bf16 (Aster's choice).")
    _hp("--out-bigram-gate-bias", OUT_BIGRAM_GATE_BIAS, float,
        "T4.2: bias init of the output bigram table's per-token gate. Aster specifies -2.0.")
    _hp("--out-bigram-lr", OUT_BIGRAM_LR, float,
        "T4.2: lr for the output bigram table's own AdamW group. 0 => UNEMBEDDING_LR.")
    # T3.9: Recursive's cheap architectural one-liners, each independently toggleable.
    _hp("--relu2-tau", RELU2_TAU, float,
        "T3.9: shifted ReLU^2 -- the MLP activation becomes relu(h - tau)^2. 0.0 = historical. "
        "Recursive: 'tau=0.5 confirmed optimal'.")
    parser.add_argument("--mlp-sandwich-norm", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_MLP_SANDWICH_NORM", "1" if USE_MLP_SANDWICH_NORM else "0") != "0",
                        help="T3.9: x = x + norm(mlp(norm(x))) instead of x = x + mlp(norm(x)).")
    parser.add_argument("--head-gate", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_HEAD_GATE", "1" if USE_HEAD_GATE else "0") != "0",
                        help="T3.6/T3.9: per-head gate on the attention output, read from a residual "
                             "slice that no value-embedding gate has claimed (see head_gate_slice).")
    parser.add_argument("--head-gate-norm", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_HEAD_GATE_NORM", "1" if HEAD_GATE_NORM else "0") != "0",
                        help="RMS-norm each head's attention output before gating (Recursive's form). "
                             "Pass --no-head-gate-norm for the gate-only variant, which is an exact "
                             "no-op at step 0.")
    _hp("--head-gate-channels", HEAD_GATE_CHANNELS, int,
        "T3.6/T3.9: width of the residual slice the head gate reads. SHAPE-CRITICAL.")
    _hp("--head-gate-mul", HEAD_GATE_MUL, float,
        "T3.6/T3.9: gate = this * sigmoid(...). 2.0 makes zero-init neutral; 1.0 is the plan's "
        "sigmoid-only T3.6 form.")
    _hp("--head-gate-bias", HEAD_GATE_BIAS, float,
        "T3.6/T3.9: bias init of the head gate. 0.0 is neutral at --head-gate-mul 2.")
    parser.add_argument("--out-pool", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_OUT_POOL", "1" if USE_OUT_POOL else "0") != "0",
                        help="T3.9: zero-init pooling of the last --out-pool-layers block outputs "
                             "into the final residual. Exact no-op at step 0.")
    _hp("--out-pool-layers", OUT_POOL_LAYERS, int,
        "T3.9: how many trailing block outputs to pool. Recursive uses 3. SHAPE-CRITICAL.")
    parser.add_argument("--qk-shift", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_QK_SHIFT", "1" if USE_QK_SHIFT else "0") != "0",
                        help="T3.9: causal token shift on the Q/K inputs only (q_in = k_in = "
                             "x + beta*x_prev, per-dim learned beta, V unshifted). Different "
                             "mechanism from --local-conv.")
    _hp("--qk-shift-beta", QK_SHIFT_BETA, float,
        "T3.9: per-dim init of the Q/K shift's beta. Recursive uses 0.3.")
    parser.add_argument("--qk-shift-freeze", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_QK_SHIFT_FREEZE", "1" if QK_SHIFT_FREEZE else "0") != "0",
                        help="freeze the Q/K shift's beta (requires_grad=False, no optimizer group) "
                             "so it stays at --qk-shift-beta forever. With --qk-shift-beta 0.0 this "
                             "is a mathematical identity that still keeps the elementwise ops in the "
                             "compiled graph -- the decisive probe for the block-body "
                             "compiler-scheduling speedup.")
    parser.add_argument("--block-nudge", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_BLOCK_NUDGE", "1" if USE_BLOCK_NUDGE else "0") != "0",
                        help="add `x = x + nudge*x` with one frozen zero scalar per block -- a "
                             "mathematical identity whose only effect is on the compiled graph. The "
                             "cheap, general form of the +10%% throughput the frozen qk-shift found.")
    parser.add_argument("--block-nudge-site", type=str, choices=("qk", "resid"),
                        default=os.environ.get("NEURON_COMPETITION_R1_BLOCK_NUDGE_SITE", BLOCK_NUDGE_SITE),
                        help="where --block-nudge lands: the Q/K input ('qk', where --qk-shift put "
                             "it) or the residual stream after the resid/x0 mix ('resid').")
    parser.add_argument("--parallel-block", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_PARALLEL_BLOCK", "1" if USE_PARALLEL_BLOCK else "0") != "0",
                        help="GPT-J/PaLM parallel block: attention and MLP both read one norm(x) and "
                             "are summed into the residual together (x + la*attn + lm*mlp). Removes "
                             "3 of 4 norms per block and unserialises the two branches.")
    parser.add_argument("--parallel-block-coalesce", type=str, choices=("none", "vfc", "all"),
                        default=os.environ.get("NEURON_COMPETITION_R1_PARALLEL_BLOCK_COALESCE", PARALLEL_BLOCK_COALESCE),
                        help="merge the parallel block's input projections into one matmul: 'vfc' "
                             "(c_v+c_fc, no copies) or 'all' (q,k,v,c_fc, needs contiguous copies "
                             "of q/k to keep the rope+QK-norm NKI kernel). Needs --parallel-block.")
    parser.add_argument("--qkv-norm-cse", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_QKV_NORM_CSE", "1" if USE_QKV_NORM_CSE else "0") != "0",
                        help="hoist the three identical norm(q_in)/norm(k_in)/norm(v_in) calls into "
                             "one when they share the same input tensor. Bit-exact identity; the "
                             "only effect is 2 fewer ops per block in a dispatch-bound graph.")
    parser.add_argument("--engram", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_ENGRAM", "1" if USE_ENGRAM else "0") != "0",
                        help="T3.2 Engram-lite: hashed n-gram memory read through a gate on the "
                             "hidden state (alpha = sigmoid(norm(h).norm(k)/sqrt(d))), then "
                             "Y = SiLU(dwconv(norm(alpha*v))) + alpha*v added to the residual. The "
                             "context-aware superset of the n-gram value embeddings.")
    parser.add_argument("--engram-site", type=str,
                        default=os.environ.get("NEURON_COMPETITION_R1_ENGRAM_SITE", ENGRAM_SITE),
                        help="block index whose OUTPUT residual Engram writes into; negative counts "
                             "from the end. Default 2 (the paper's mid-stack site): T3.1's LNC=2 "
                             "downstream-resharding penalty was expected to make this unaffordable "
                             "and measurably does NOT (1128 vs 1129 steps at 900s), while being "
                             "0.0016 bpb better than the last-layer placement.")
    parser.add_argument("--engram-orders", type=str,
                        default=os.environ.get("NEURON_COMPETITION_R1_ENGRAM_ORDERS", ENGRAM_ORDERS),
                        help="comma-separated n-gram orders for Engram's hash heads; only 2 and 3 "
                             "are supported (Engram reports 4-grams slightly suboptimal).")
    parser.add_argument("--engram-heads", type=int,
                        default=int(os.environ.get("NEURON_COMPETITION_R1_ENGRAM_HEADS", ENGRAM_HEADS)),
                        help="hash heads PER ORDER. Each head is its own table with its own primes, "
                             "so K heads is K independent hashings of the same n-gram (Engram's own "
                             "collision-mitigation trick). Total tables = heads * len(orders).")
    parser.add_argument("--engram-table-mult", type=int,
                        default=int(os.environ.get("NEURON_COMPETITION_R1_ENGRAM_TABLE_MULT", ENGRAM_TABLE_MULT)),
                        help="rows per Engram table, in units of the padded vocab (8192). 32 => "
                             "262144 rows; the modulus used is the largest prime <= that.")
    parser.add_argument("--engram-mem-dim", type=int,
                        default=int(os.environ.get("NEURON_COMPETITION_R1_ENGRAM_MEM_DIM", ENGRAM_MEM_DIM)),
                        help="d_mem: width of each Engram table row. e_t is the concatenation over "
                             "all tables, so w_k/w_v read heads*len(orders)*d_mem.")
    parser.add_argument("--engram-key-dim", type=int,
                        default=int(os.environ.get("NEURON_COMPETITION_R1_ENGRAM_KEY_DIM", ENGRAM_KEY_DIM)),
                        help="width of the gate's key/query space. 0 (default) means n_embd, which "
                             "lets the gate dot norm(h) directly and needs no query projection.")
    parser.add_argument("--engram-conv-kernel", type=int,
                        default=int(os.environ.get("NEURON_COMPETITION_R1_ENGRAM_CONV_KERNEL", ENGRAM_CONV_KERNEL)),
                        help="depthwise causal conv kernel applied to norm(alpha*v) before the "
                             "SiLU. 0 disables the conv branch entirely (Y = alpha*v), which is the "
                             "ablation arm. Zero-init, so it starts as a no-op either way.")
    parser.add_argument("--engram-share-ngram-tables", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_ENGRAM_SHARE_NGRAM_TABLES", "1" if ENGRAM_SHARE_NGRAM_TABLES else "0") != "0",
                        help="read T3.1's ALREADY-GATHERED n-gram rows instead of Engram's own "
                             "tables. Isolates the MECHANISM (gate + conv + residual write) from "
                             "the added capacity: no new gather, no new table parameters, no extra "
                             "gradient all-reduce. Needs an n-gram table at or before --engram-site.")
    parser.add_argument("--engram-bf16", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_ENGRAM_BF16", "1" if ENGRAM_BF16 else "0") != "0",
                        help="hold Engram's tables in bf16 (halves 67M params of memory and the "
                             "gather's bandwidth). Same trade as --ngram-ve-bf16.")
    parser.add_argument("--epoch-shuffle", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_EPOCH_SHUFFLE", "1" if USE_EPOCH_SHUFFLE else "0") != "0",
                        help="reshuffle the row-group order from epoch 2 onward. Epoch 1 keeps "
                             "prepare's exact order, so this is a no-op for any run that never "
                             "crosses the epoch boundary. Only matters above ~2195 steps at the "
                             "locked shape -- see the 2026-09-13 data-wall section of "
                             "research/trn2-experiment-results.md.")
    parser.add_argument("--epoch-shuffle-seed", type=int, default=EPOCH_SHUFFLE_SEED,
                        help="seed base for --epoch-shuffle; epoch e is shuffled with seed+e.")
    parser.add_argument("--x0-gate", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_X0_GATE", "1" if USE_X0_GATE else "0") != "0",
                        help="Recursive's input-dependent x0 gate: scale each block's x0 skip by "
                             "2*sigmoid(s*mean(x)) with one zero-init learned scalar s per layer. "
                             "Bit-exact no-op at step 0. Mutually exclusive with USE_SELECTIVE_GATE, "
                             "which owns the same line of Block.forward.")
    parser.add_argument("--batch-ramp", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_BATCH_RAMP", "1" if BATCH_RAMP else "0") != "0",
                        help="T2.1: ramp the effective batch size across the run by varying "
                             "grad_accum_steps on wall-clock-fraction boundaries, holding the micro "
                             "shape (and therefore every compiled graph) fixed. Off by default.")
    parser.add_argument("--batch-ramp-grad-accum", type=str,
                        default=os.environ.get("NEURON_COMPETITION_R1_BATCH_RAMP_GRAD_ACCUM",
                                               ",".join(str(g) for g in BATCH_RAMP_GRAD_ACCUM)),
                        help="T2.1: comma-separated micro-batch counts for stages A,B,C. Stage B "
                             "should normally be TOTAL_BATCH_SIZE's own grad_accum (4 at the locked "
                             "shape) so the ramp is symmetric around today's batch size.")
    parser.add_argument("--batch-ramp-bounds", type=str,
                        default=os.environ.get("NEURON_COMPETITION_R1_BATCH_RAMP_BOUNDS",
                                               ",".join(str(b) for b in BATCH_RAMP_BOUNDS)),
                        help="T2.1: comma-separated elapsed/max_train_seconds boundaries A->B,B->C. "
                             "The second should normally equal WARMDOWN_RATIO so the largest batch "
                             "covers exactly the lr cooldown.")
    parser.add_argument("--batch-ramp-lr-exp", type=float,
                        default=float(os.environ.get("NEURON_COMPETITION_R1_BATCH_RAMP_LR_EXP", BATCH_RAMP_LR_EXP)),
                        help="T2.1: lr scales as (B/B_ref)**this. 0.5 = sqrt (gradient-noise "
                             "matched), 1.0 = linear scaling, 0.0 = leave the lr alone.")
    parser.add_argument("--weight-ema", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_WEIGHT_EMA", "1" if USE_WEIGHT_EMA else "0") != "0",
                        help="T2.2: keep an exponential moving average of the weights over the "
                             "cooldown tail, save it alongside the raw weights, and evaluate both. "
                             "Off by default.")
    parser.add_argument("--weight-ema-decay", type=str,
                        default=os.environ.get("NEURON_COMPETITION_R1_WEIGHT_EMA_DECAY",
                                               ",".join(str(d) for d in WEIGHT_EMA_DECAYS)),
                        help="T2.2: comma-separated per-step EMA decays -- one shadow each, all "
                             "updated from the same weights every step, so they are a zero-noise "
                             "comparison. The averaging window is ~1/(1-decay) steps (0.98 ~ 50, "
                             "0.995 ~ 200, 0.999 ~ 1000). The FIRST is the primary: it is the one "
                             "saved into the checkpoint's 'ema_state'.")
    parser.add_argument("--weight-ema-start-frac", type=float,
                        default=float(os.environ.get("NEURON_COMPETITION_R1_WEIGHT_EMA_START_FRAC", WEIGHT_EMA_START_FRAC)),
                        help="T2.2: elapsed/max_train_seconds at which the shadow is seeded from "
                             "the live weights. Defaults to WARMDOWN_RATIO (average the cooldown).")
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
    parser.add_argument("--rwkv", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_RWKV", "1" if USE_RWKV else "0") != "0",
                        help="replace the layers selected by --rwkv-pattern with an RWKV x051a "
                             "time-mix + channel-mix pair instead of attention + MLP (linear-attention "
                             "decay operator, no rope, no attn mask, evaluated in a loop-free chunked "
                             "form). Mutually exclusive with --mamba2. Default OFF -- see the USE_RWKV "
                             "comment for the measured verdict.")
    parser.add_argument("--rwkv-pattern", type=str, default=RWKV_PATTERN,
                        help="per-layer selection cycled to n_layer: 'R' = RWKV block, anything else "
                             "= ordinary attention + MLP (e.g. 'RRRARR' keeps one attention layer "
                             "at index 3). Only read when --rwkv is on.")
    parser.add_argument("--rwkv-chunk", type=int, default=RWKV_CHUNK,
                        help="Q, the chunk length the decay operator is blocked at; must divide "
                             "sequence_len. Pure blocking of identical math -- NOT shape-critical. "
                             "Only read when --rwkv is on.")
    parser.add_argument("--rwkv-head-dim", type=int, default=RWKV_HEAD_DIM,
                        help="N, the per-head width; head count is n_embd / N (5 at n_embd=640). The "
                             "carried state is [N,N] per head, so this is the dominant cost knob. "
                             "Only read when --rwkv is on.")
    parser.add_argument("--rwkv-cmix-expand", type=int, default=RWKV_CMIX_EXPAND,
                        help="channel-mix hidden width as a multiple of n_embd (upstream x051a uses "
                             "3, which with the 5 time-mix projections matches the transformer "
                             "layer's 12*C^2). Only read when --rwkv is on.")
    parser.add_argument("--init-from", type=str, default="",
                        help="warm-start model weights from a prior checkpoint (fresh optimizer/step/dataloader; dev-iteration only, never for a scored submission)")
    parser.add_argument("--grad-bucket", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_GRAD_BUCKET", "0") != "0",
                        help="T1.2: point every parameter's .grad at a view into one flat fp32 "
                             "buffer and all_reduce it once per step instead of once per parameter. "
                             "Also switches grad clipping to the flat-buffer one-norm/one-mul form "
                             "(T1.3). Verified bit-equal to the per-tensor path in "
                             "research/scratch/grad_bucket_equiv.py. Default OFF pending a real "
                             "hardware A/B (~1%% expected win).")
    # Tri-state on purpose: absent -> None (let clip_grad_norm_ auto-detect, the historical
    # default), --clip-foreach -> True, --no-clip-foreach -> False. The False leg is not cosmetic:
    # at LNC=1 the auto-detected `_foreach_norm` path routes into the SDK's
    # nkilib.experimental.foreach.foreach_norm l2_norm_kernel_lnc0, which fails to COMPILE on Trn2
    # ("[NCC_IBIR533] CollectiveKind::SendRecv is not a valid enum value for field
    # CollectiveCompute.Kind on arch Trn2"), so --no-clip-foreach is the only way to get any
    # LNC=1 number at all with GRAD_CLIP > 0. See trn2-experiment-results.md T0.6.
    parser.add_argument("--clip-foreach", action=argparse.BooleanOptionalAction,
                        default=(None if "NEURON_COMPETITION_R1_CLIP_FOREACH" not in os.environ
                                 else os.environ["NEURON_COMPETITION_R1_CLIP_FOREACH"] != "0"),
                        help="T1.3: force foreach=True/False on torch.nn.utils.clip_grad_norm_ "
                             "instead of letting it auto-detect (the default when the flag is "
                             "absent). Numerically identical; only matters when --grad-bucket is "
                             "off. Measured NEUTRAL at LNC=2 (clip_ms 6.2-7.7 vs 5.5-7.7).")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("NEURON_COMPETITION_R1_SEED", "42")),
                        help="torch.manual_seed value (init_runtime). Default 42, unchanged from the "
                             "long-standing hardcoded value -- exposed as a flag purely so noise/variance "
                             "can be estimated with paired seeds without editing the file.")
    parser.add_argument("--local-conv-kernel", type=int,
                        default=int(os.environ.get("NEURON_COMPETITION_R1_LOCAL_CONV_KERNEL", str(LOCAL_CONV_KERNEL))),
                        help="depthwise causal conv1d kernel size for --local-conv (T0.1 sweep knob). "
                             "Only matters once the conv's weight is actually trained (see the "
                             "USE_LOCAL_CONV comment); a shape-critical toggle already covered by "
                             "SHAPE_CRITICAL_TOGGLES via LOCAL_CONV_KERNEL.")
    parser.add_argument("--train-local-conv", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_TRAIN_LOCAL_CONV", "1" if TRAIN_LOCAL_CONV else "0") != "0",
                        help="T0.1: register local_conv.weight in its own AdamW group (lr=MATRIX_LR, "
                             "betas=(0.9,0.95), wd=0.0) instead of leaving it permanently zero/dead. "
                             "Still zero-initialized, so step 0 is unchanged either way. Default OFF "
                             "(bit-identical to every prior locked-baseline run) -- pass "
                             "--train-local-conv for the A/B; only matters when --local-conv is on.")
    # ---- T0.4 hyperparameter-resweep knobs -----------------------------------------------
    # Every one of these defaults to the module-level constant, so passing none of them is
    # bit-identical to the locked baseline; they exist so a coordinate-descent sweep runs from
    # ONE identical binary instead of a series of edited files (the same discipline every other
    # A/B in this repo follows). All are float/int scalars read either before GPT(config) (init
    # scales) or inside setup_optimizer/lr_multiplier (LRs, decays), never mid-step, so none of
    # them can introduce a per-step graph constant.
    # (_hp itself is defined further up, just above the T4.2 block, because that block was the
    # first caller and `_hp` is a local of main() -- calling it before its `def` statement runs
    # raises UnboundLocalError, which is exactly what happened on 2026-09-12.)
    _hp("--seq-len", SEQ_LEN, int, "Training context length. Previously only settable by editing the "
        "constant. Currently 2048. NOTE: the in-process eval runs at this same length, so a "
        "--seq-len A/B is apples-to-oranges unless --eval-seq-len pins both arms to one eval "
        "length -- which matters a lot now that the scorer looks like it evaluates at 1024 "
        "(see the 2026-09-11 section of trn2-experiment-results.md).")
    parser.add_argument("--eval-seq-len", type=int,
                        default=int(os.environ.get("NEURON_COMPETITION_R1_EVAL_SEQ_LEN", 0)),
                        help="Run a SECOND in-process public_val eval at this context length and "
                             "report it as public_val_bpb_alt. 0 (default) = off, zero cost. Must be "
                             "<= --seq-len and a multiple of 128: the live model's rotary/mask tables "
                             "are sized to --seq-len (a LONGER eval needs prepare.py eval-public, "
                             "whose load_for_eval widens them to EVAL_MIN_SEQ_LEN). Post-training "
                             "only, so it costs wall clock and never a charged training second.")
    _hp("--depth", DEPTH, int, "T0.2: n_layer. Previously only settable by editing the constant, so "
        "every historical depth sweep also moved the width (n_embd is derived from DEPTH via "
        "ASPECT_RATIO unless --n-embd is given). Currently 6.")
    _hp("--n-embd", N_EMBD, int, "T0.2: force n_embd (must be a multiple of --head-dim), decoupling "
        "width from DEPTH*ASPECT_RATIO. 0 (default) = historical derived width, bit-identical.")
    _hp("--head-dim", HEAD_DIM, int, "T0.2: per-head width; n_head = n_embd / head_dim. Also the "
        "rounding granularity of the derived n_embd. Currently 128. The fused rope+QK-norm NKI "
        "kernel is written generically in D (it only requires D even), so 64 stays on the kernel.")
    _hp("--mlp-ratio", MLP_RATIO, int, "T0.3: MLP hidden width as a multiple of n_embd. Currently 4.")
    _hp("--attn-scale", ATTN_SCALE, float, "T0.3: explicit SDPA softmax scale. 0.0 (default) = SDPA's "
        "own 1/sqrt(head_dim) (0.0884 at 128). QK-norm already fixes the q/k norms, so the logit "
        "temperature is a free knob here rather than a redundant init reparameterisation.")
    _hp("--matrix-lr", MATRIX_LR, float, "T0.4: Muon groups' base lr (also the AdamW lr for the "
        "ssm/rwkv/hypernet/local_conv groups). Currently 0.025.")
    _hp("--embedding-lr", EMBEDDING_LR, float, "T0.4: wte (and value-embedding) AdamW lr before the "
        "(n_embd/768)**-0.5 scale. Currently 0.30.")
    _hp("--unembedding-lr", UNEMBEDDING_LR, float, "T0.4: lm_head AdamW lr before the d_model scale. "
        "Currently 0.006.")
    _hp("--scalar-lr", SCALAR_LR, float, "T0.4: AdamW lr for the scalar/skip params. Currently 0.20.")
    _hp("--weight-decay", WEIGHT_DECAY, float, "T0.4: Muon groups' weight decay (decayed to zero over "
        "the run unless --no-wd-decay-to-zero). Currently 0.10.")
    _hp("--final-lr-frac", FINAL_LR_FRAC, float, "T0.4: LR floor at the end of cooldown as a fraction "
        "of peak. Currently 0.05.")
    _hp("--warmdown-ratio", WARMDOWN_RATIO, float, "T0.4: fraction of the wall-clock budget spent on "
        "LR cooldown. Currently 0.75.")
    _hp("--init-scale", INIT_SCALE, float, "T0.4: multiplier on the attn/mlp uniform-init bound. "
        "Currently 0.68.")
    _hp("--attn-cproj-init", ATTN_CPROJ_INIT_SCALE, float, "T0.4: attn.c_proj init bound as a multiple "
        "of the shared bound s; 0.0 (default) keeps the historical exact-zero init.")
    _hp("--logit-softcap", LOGIT_SOFTCAP, float, "T0.4: tanh logit softcap. Currently 15.0.")
    _hp("--muon-ns-steps", MUON_NS_STEPS, int, "T0.4: Newton-Schulz iterations per Muon step "
        "(3 matmuls each). Currently 5.")
    _hp("--muon-momentum-cooldown-frac", MUON_MOMENTUM_COOLDOWN_FRAC, float, "T0.4: ramp Muon momentum "
        "from its warmed-up 0.95 down to --muon-momentum-final over this trailing fraction of the "
        "wall-clock budget. 0.0 (default) = no cooldown, bit-identical to the old schedule.")
    _hp("--muon-momentum-final", MUON_MOMENTUM_FINAL, float, "T0.4: momentum at the very end of the "
        "cooldown. Only read when --muon-momentum-cooldown-frac > 0.")
    parser.add_argument("--wd-decay-to-zero", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_WD_DECAY_TO_ZERO",
                                               "1" if WD_DECAY_TO_ZERO else "0") != "0",
                        help="T0.4: decay Muon's weight_decay linearly to 0 over the run instead of "
                             "holding it constant. Currently ON.")
    _hp("--grad-clip", GRAD_CLIP, float, "Global gradient-norm clip, applied to each rank's LOCAL "
        "gradients BEFORE sync_gradients' all_reduce(AVG) -- so its effective strength depends on "
        "the per-rank token count, and therefore on world_size. 0 disables it. Currently 1.0; a "
        "flag so the T0.6 LNC=1 quality-cliff mechanism (clip-then-average vs average-then-clip) "
        "is testable without editing the constant.")
    _hp("--pack-factor", PACK_FACTOR, int, "T1.5: dataloader packing-bin multiplier. The loader "
        "fetches DEVICE_BATCH_SIZE rows of PACK_FACTOR*SEQ_LEN tokens and reshapes them to "
        "DEVICE_BATCH_SIZE*PACK_FACTOR rows of SEQ_LEN, so this is the micro-batch ROW COUNT and "
        "grad_accum_steps falls as it rises (TOTAL_BATCH_SIZE is held fixed). Only ever set by "
        "editing this constant before; a flag so the T1.5 re-test is reproducible from the "
        "launch command alone.")
    _hp("--adam-every-n", ADAM_EVERY_N, int, "T1.4: update the AdamW parameter groups (wte, lm_head, "
        "value-embedding tables, scalars, local_conv) only every Nth step, accumulating their "
        "gradients in between and skipping their all_reduce on the other steps. Muon groups still "
        "update every step. 1 (default) = historical behaviour, bit-identical.")
    parser.add_argument("--adam-every-n-lr-comp", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_ADAM_EVERY_N_LR_COMP",
                                               "1" if ADAM_EVERY_N_LR_COMP else "0") != "0",
                        help="T1.4: multiply the AdamW groups' lr by --adam-every-n. Adam's update is "
                             "scale-invariant in the gradient, so updating N times less often "
                             "otherwise moves those tensors ~N times less far in total; this "
                             "separates 'less often' from 'less far'. Default OFF.")
    args = parser.parse_args()
    if args.eval_seq_len:
        if args.eval_seq_len > args.seq_len:
            parser.error(f"--eval-seq-len {args.eval_seq_len} > --seq-len {args.seq_len}: the live "
                         "model's rotary/mask tables are only built out to --seq-len. Use "
                         "`prepare.py eval-public --seq-len N` on the saved checkpoint instead.")
        if args.eval_seq_len % 128 != 0:
            parser.error("--eval-seq-len must be a multiple of 128 (NKI tiling).")
    if args.adam_every_n < 1:
        parser.error("--adam-every-n must be >= 1")
    if args.adam_every_n > 1 and args.grad_bucket:
        parser.error("--adam-every-n > 1 is incompatible with --grad-bucket: the flat buffer is "
                     "zeroed wholesale every step, so it cannot leave the AdamW groups' gradients "
                     "standing to accumulate. (--grad-bucket is also a known hang on this hardware; "
                     "see trn2-experiment-results.md.)")
    if args.rwkv and args.mamba2:
        parser.error("--rwkv and --mamba2 are mutually exclusive: a layer cannot be both a Mamba2 "
                     "mixer and an RWKV block, and no hybrid-of-hybrids pattern language exists. "
                     "Pick one, or use its pattern flag to keep some attention layers.")

    global USE_NKI_RELU2, USE_NKI_ROPE_NORM, USE_NKI_SOFTCAP_CE, USE_NKI_MUON
    # (USE_COMPILE_MUON's `global` is at the top of main(); see the note there.)
    # T0.4/T1.4 resweep knobs (see the _hp block above; declared global at the top of main()).
    # Assigned here, i.e. before GPT(config) and before setup_optimizer(), which is where every
    # one of them is read.
    ADAM_EVERY_N = args.adam_every_n
    ADAM_EVERY_N_LR_COMP = args.adam_every_n_lr_comp
    GRAD_CLIP = args.grad_clip
    PACK_FACTOR = args.pack_factor
    # T0.2/T0.3: all four are read by build_config()/MLP.__init__/CausalSelfAttention.__init__ below.
    SEQ_LEN = args.seq_len
    DEPTH = args.depth
    N_EMBD = args.n_embd
    HEAD_DIM = args.head_dim
    MLP_RATIO = args.mlp_ratio
    ATTN_SCALE = args.attn_scale
    MATRIX_LR = args.matrix_lr
    EMBEDDING_LR = args.embedding_lr
    UNEMBEDDING_LR = args.unembedding_lr
    SCALAR_LR = args.scalar_lr
    WEIGHT_DECAY = args.weight_decay
    WD_DECAY_TO_ZERO = args.wd_decay_to_zero
    FINAL_LR_FRAC = args.final_lr_frac
    WARMDOWN_RATIO = args.warmdown_ratio
    INIT_SCALE = args.init_scale               # read by GPT.init_weights, i.e. inside GPT(config)
    ATTN_CPROJ_INIT_SCALE = args.attn_cproj_init  # ditto
    LOGIT_SOFTCAP = args.logit_softcap
    MUON_NS_STEPS = args.muon_ns_steps
    MUON_MOMENTUM_COOLDOWN_FRAC = args.muon_momentum_cooldown_frac
    MUON_MOMENTUM_FINAL = args.muon_momentum_final
    USE_VE = args.ve                         # read by Block/CausalSelfAttention/GPT.__init__, i.e. before GPT(config) below
    USE_LOCAL_CONV = args.local_conv         # read by Block.__init__, i.e. before GPT(config) below
    LOCAL_CONV_KERNEL = args.local_conv_kernel  # ditto (CausalConv.__init__ reads it via the global default)
    TRAIN_LOCAL_CONV = args.train_local_conv  # read by GPT.setup_optimizer(), called after GPT(config) below
    # T3.1: the first four are read by GPT.__init__/CausalSelfAttention.__init__ below, the
    # rest by setup_optimizer() / the training loop.
    USE_NGRAM_VE = args.ngram_ve
    NGRAM_VE_LAYERS = args.ngram_ve_layers
    NGRAM_VE_TABLE_MULT = args.ngram_ve_table_mult
    NGRAM_VE_DIM = args.ngram_ve_dim
    NGRAM_VE_TRIGRAM = args.ngram_ve_trigram
    NGRAM_VE_BF16 = args.ngram_ve_bf16
    NGRAM_VE_OPT = args.ngram_ve_opt
    NGRAM_VE_LR = args.ngram_ve_lr
    NGRAM_VE_BETA2 = args.ngram_ve_beta2
    NGRAM_VE_FLAT_LR = args.ngram_ve_flat_lr
    NGRAM_VE_CLIP_EXCLUDE = args.ngram_ve_clip_exclude
    # T4.2 / T3.9. All read by GPT/Block/CausalSelfAttention.__init__ or setup_optimizer() below,
    # i.e. before the first device op.
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
    # T3.2. All read by EngramLite.__init__/GPT.__init__, i.e. before GPT(config) below.
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
        engram_orders()  # validates ENGRAM_ORDERS, raising here rather than mid-construction
        if ENGRAM_HEADS < 1:
            raise ValueError(f"--engram-heads must be >= 1, got {ENGRAM_HEADS}")
        if ENGRAM_TABLE_MULT < 1 or ENGRAM_MEM_DIM < 1:
            raise ValueError("--engram-table-mult and --engram-mem-dim must both be >= 1, got "
                             f"{ENGRAM_TABLE_MULT} and {ENGRAM_MEM_DIM}")
        if ENGRAM_CONV_KERNEL < 0:
            raise ValueError(f"--engram-conv-kernel must be >= 0 (0 disables), got {ENGRAM_CONV_KERNEL}")
        if ENGRAM_SHARE_NGRAM_TABLES and not USE_NGRAM_VE:
            raise ValueError("--engram-share-ngram-tables needs --ngram-ve: there are no gathered "
                             "n-gram rows to share otherwise.")
    USE_EPOCH_SHUFFLE = args.epoch_shuffle
    EPOCH_SHUFFLE_SEED = args.epoch_shuffle_seed
    USE_PARALLEL_BLOCK = args.parallel_block
    PARALLEL_BLOCK_COALESCE = args.parallel_block_coalesce
    if PARALLEL_BLOCK_COALESCE != "none" and not USE_PARALLEL_BLOCK:
        raise ValueError("--parallel-block-coalesce needs --parallel-block: the coalesced matmul is "
                         "only valid when all four input projections read the same normed residual, "
                         "which is exactly what the parallel block makes true.")
    if PARALLEL_BLOCK_COALESCE != "none" and (USE_FUSED_QKV or USE_GATED_MLP):
        raise ValueError("--parallel-block-coalesce is incompatible with USE_FUSED_QKV (it is the "
                         "same idea, done differently) and with USE_GATED_MLP (c_gate would still "
                         "need its own matmul over the same input, which defeats the coalescing).")
    if USE_PARALLEL_BLOCK:
        # The parallel block reads ONE norm(x) and ignores q_in/k_in/v_in entirely, so anything
        # that exists only to make those three differ would be silently dropped. Fail loudly.
        for name, on in (("--qk-shift", USE_QK_SHIFT), ("--mudd", USE_MUDD),
                         ("--block-nudge", USE_BLOCK_NUDGE),
                         ("--mlp-sandwich-norm", USE_MLP_SANDWICH_NORM)):
            if on:
                raise ValueError(f"--parallel-block and {name} are incompatible: the parallel block "
                                 "replaces the whole serial attn->MLP body that "
                                 f"{name} modifies, so {name} would be silently ignored.")
    if USE_BLOCK_NUDGE and USE_QK_SHIFT and BLOCK_NUDGE_SITE == "qk":
        raise ValueError("--block-nudge --block-nudge-site qk and --qk-shift both rewrite q_in/k_in; "
                         "the nudge exists to REPLACE the frozen qk-shift, so pick one.")
    if USE_X0_GATE and USE_SELECTIVE_GATE:
        raise ValueError("--x0-gate and USE_SELECTIVE_GATE both rewrite the same resid/x0 mix line "
                         "in Block.forward; pick one.")
    if RELU2_TAU != 0.0 and args.nki_relu2:
        raise ValueError("--relu2-tau is not implemented by the example NKI relu2 kernel; "
                         "run it with --no-nki-relu2 (the default).")
    if OUT_POOL_LAYERS < 1:
        raise ValueError(f"--out-pool-layers must be >= 1, got {OUT_POOL_LAYERS}")
    if HEAD_GATE_CHANNELS < 1:
        raise ValueError(f"--head-gate-channels must be >= 1, got {HEAD_GATE_CHANNELS}")
    BATCH_RAMP = args.batch_ramp
    BATCH_RAMP_GRAD_ACCUM = tuple(int(v) for v in args.batch_ramp_grad_accum.split(","))
    BATCH_RAMP_BOUNDS = tuple(float(v) for v in args.batch_ramp_bounds.split(","))
    BATCH_RAMP_LR_EXP = args.batch_ramp_lr_exp
    if len(BATCH_RAMP_GRAD_ACCUM) != 3 or min(BATCH_RAMP_GRAD_ACCUM) < 1:
        raise ValueError(f"--batch-ramp-grad-accum needs three counts >= 1, got {BATCH_RAMP_GRAD_ACCUM}")
    if len(BATCH_RAMP_BOUNDS) != 2 or not 0.0 < BATCH_RAMP_BOUNDS[0] <= BATCH_RAMP_BOUNDS[1] < 1.0:
        raise ValueError(f"--batch-ramp-bounds needs 0 < A->B <= B->C < 1, got {BATCH_RAMP_BOUNDS}")
    USE_WEIGHT_EMA = args.weight_ema
    WEIGHT_EMA_DECAYS = tuple(float(v) for v in args.weight_ema_decay.split(","))
    WEIGHT_EMA_START_FRAC = args.weight_ema_start_frac
    if not WEIGHT_EMA_DECAYS or not all(0.0 < d < 1.0 for d in WEIGHT_EMA_DECAYS):
        raise ValueError(f"--weight-ema-decay needs one or more values in (0,1), got {WEIGHT_EMA_DECAYS}")
    if not 0.0 <= WEIGHT_EMA_START_FRAC < 1.0:
        raise ValueError(f"--weight-ema-start-frac must be in [0,1), got {WEIGHT_EMA_START_FRAC}")
    LRM_QUANTUM = args.lrm_quantum
    USE_TENSOR_LR_SCALARS = args.tensor_lr_scalars
    USE_FOREACH_OPTIM = args.foreach_optim
    USE_FP8_LINEAR = args.fp8_linear           # read by MLP.__init__, i.e. before GPT(config) below
    USE_MUDD = args.mudd                      # read by Block.__init__, i.e. before GPT(config) below
    USE_DCMHA = args.dcmha                    # read by CausalSelfAttention.__init__, i.e. before GPT(config) below
    USE_NKI_RELU2 = args.nki_relu2
    USE_NKI_ROPE_NORM = args.nki_rope_norm
    USE_NKI_SOFTCAP_CE = args.nki_softcap_ce
    USE_NKI_MUON = args.nki_muon
    USE_COMPILE_MUON = args.compile_muon
    USE_MAMBA2 = args.mamba2                # read by Block.__init__, i.e. before GPT(config) below
    MAMBA2_PATTERN = args.mamba2_pattern      # ditto (mamba2_layer_flags reads it per block)
    # Mixer geometry: all read by Mamba2Mixer.__init__, i.e. before GPT(config) below.
    MAMBA2_EXPAND = args.mamba2_expand
    MAMBA2_D_STATE = args.mamba2_d_state
    MAMBA2_HEAD_DIM = args.mamba2_head_dim
    MAMBA2_CHUNK = args.mamba2_chunk
    MAMBA2_N_GROUPS = args.mamba2_n_groups
    MAMBA2_CONV_KERNEL = args.mamba2_conv_kernel
    USE_NKI_SSD = args.nki_ssd
    USE_RWKV = args.rwkv                      # read by Block.__init__, i.e. before GPT(config) below
    RWKV_PATTERN = args.rwkv_pattern          # ditto (rwkv_layer_flags reads it per block)
    RWKV_HEAD_DIM = args.rwkv_head_dim        # read by RWKVTimeMix.__init__, ditto
    RWKV_CHUNK = args.rwkv_chunk
    RWKV_CMIX_EXPAND = args.rwkv_cmix_expand

    device_type = detect_device_type(args.device_type)
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type, seed=args.seed)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)

    print0(f"device={device} dtype={COMPUTE_DTYPE} world_size={world_size}")
    print0(f"training target={args.num_steps} steps max_seconds={args.max_train_seconds} official_budget={TRAIN_TIME_BUDGET_SECONDS}s")
    # Guard against the 2026-09-10 failure mode (a step cap that binds before the time budget leaves the
    # wall-clock LR schedule uncooled). Coarse check here; the precise projection runs at step 50 below.
    if args.num_steps < args.max_train_seconds:
        print0(f"WARNING: --num-steps {args.num_steps} is below max_train_seconds={args.max_train_seconds}s and will "
               f"almost certainly stop training before the time budget (the LR schedule would end without cooldown). "
               f"For a scored submission leave --num-steps at its default ({NUM_STEPS}).")

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
            clip_main, clip_ngram = split_clip_params(orig_model)
            torch.nn.utils.clip_grad_norm_(clip_main, GRAD_CLIP, foreach=args.clip_foreach)
            if clip_ngram:
                torch.nn.utils.clip_grad_norm_(clip_ngram, GRAD_CLIP, foreach=args.clip_foreach)
        sync_gradients(model)
        optimizer.step()
        synchronize(device)
        print0("compile-only synthetic training step completed")
        cleanup_runtime()
        return

    loader_seq_len = PACK_FACTOR * SEQ_LEN
    if USE_REQUEUE_PACKER or USE_EPOCH_SHUFFLE:
        # train.py's own copy of the packer. With requeue=USE_REQUEUE_PACKER=False it is
        # prepare.make_dataloader line-for-line, so --epoch-shuffle changes the data ORDER and
        # nothing else.
        train_loader = make_dataloader_requeue(
            tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, "train", device,
            requeue=USE_REQUEUE_PACKER,
            _document_batches_fn=(_document_batches_epoch_shuffle if USE_EPOCH_SHUFFLE
                                  else _document_batches),
        )
    else:
        train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, "train", device)

    def fetch():
        xb, yb, state = next(train_loader)
        if PACK_FACTOR > 1:
            xb = xb.reshape(DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN)
            yb = yb.reshape(DEVICE_BATCH_SIZE * PACK_FACTOR, SEQ_LEN)
        return xb, yb, state

    x, y, loader_state = fetch()
    optimizer = model.setup_optimizer()
    grad_flat_buffer = None
    if args.grad_bucket:
        # T1.2: build once, before the first backward() of the run -- see the
        # build_flat_grad_buffer docstring. Uses model.parameters() (every trainable
        # tensor), same as sync_gradients(), so it also covers local_conv.weight
        # regardless of --train-local-conv.
        grad_flat_buffer, _grad_bucket_params = build_flat_grad_buffer(model)
        print0(f"grad_bucket enabled: {len(_grad_bucket_params)} params, {grad_flat_buffer.numel():,} floats in one flat fp32 buffer")
    # T1.4: identify the AdamW-group parameters once, outside the loop, so the per-step decision
    # costs nothing. Mutually exclusive with --grad-bucket, whose flat buffer is zeroed wholesale
    # and cannot leave one subset of gradients standing (asserted at flag-parse time).
    adamw_skip_ids: frozenset[int] = frozenset()
    if ADAM_EVERY_N > 1:
        adamw_skip_ids = optimizer.adamw_param_ids()
        if ADAM_EVERY_N_LR_COMP:
            for group in optimizer.param_groups:
                if group["kind"] == "adamw":
                    group["initial_lr"] = group["initial_lr"] * ADAM_EVERY_N
        print0(f"adam_every_n={ADAM_EVERY_N} lr_comp={ADAM_EVERY_N_LR_COMP}: "
               f"{len(adamw_skip_ids)} AdamW params update every {ADAM_EVERY_N} steps")
    # T1.10: hoist the Linear lookup out of the step loop. The refresh below used to walk the whole
    # `model.modules()` tree and isinstance-test every module on EVERY step to find the same ~40
    # Linears; the set never changes after construction. Pure removal of per-step Python work, no
    # numerics change, no new code path. (The measured alternative -- one `torch._foreach_copy_`
    # into preallocated bf16 buffers instead of ~40 separate `weight.to(dtype)` casts -- is a real
    # but tiny 2.65ms -> 1.20ms per step, i.e. 0.17% of an 837ms step, so it was NOT adopted: it
    # needs a second code path and a duplicate set of buffers to buy less than the run-to-run
    # noise on step_total. See trn2-experiment-results.md T1.10.)
    cached_linears = [m for m in model.modules() if isinstance(m, Linear)] if USE_WEIGHT_CACHE else []
    # T3.1: split once, outside the loop (the sets never change after construction), for the
    # same reason cached_linears is hoisted -- and so the main clip's tensor list is provably
    # identical to the historical model.parameters() order. See split_clip_params.
    clip_main_params, clip_ngram_params = split_clip_params(orig_model)
    assert not (grad_flat_buffer is not None and clip_ngram_params), \
        "--grad-bucket clips one flat buffer and cannot exclude the n-gram tables; " \
        "use --no-ngram-ve-clip-exclude to combine them"

    loop_started = time.monotonic()
    budget_started = loop_started

    smooth_loss = 0.0
    total_train_tokens = 0
    step = 0
    recent_step_times: list[float] = []
    startup_allowance_est = 0.0
    # T2.1: the loss scaling is 1/grad_accum, and under BATCH_RAMP that changes mid-run. A Python
    # float multiplying a device tensor is baked into the Neuron graph as a constant, so a new
    # value would force a fresh compile (the USE_TENSOR_LR_SCALARS failure mode). Carry it in a
    # preallocated device scalar written with copy_ instead. Allocated unconditionally (it is 4
    # bytes) but only USED when BATCH_RAMP is on, so the default path keeps the literal
    # `loss / grad_accum_steps` and stays bit-identical to every previous run.
    inv_gas_host = torch.empty((), dtype=torch.float32)
    inv_gas_dev = torch.zeros((), dtype=torch.float32, device=device)
    grad_accum_now = grad_accum_steps
    ramp_stage_logged = -1
    # T2.2: EMA of the final weights. `ema_names`/`ema_live` are fixed lists (the parameter set
    # never changes after construction) so the per-step work is one torch._foreach_lerp_ over
    # preallocated tensors and no Python walk of the module tree. `ema_shadows` stays None until
    # WEIGHT_EMA_START_FRAC of the budget has elapsed -- both to keep the high-lr phase out of the
    # average and so a run that never reaches that point allocates nothing.
    ema_names = [n for n, _ in orig_model.named_parameters()]
    ema_live = [p for _, p in orig_model.named_parameters()]
    ema_shadows: list[list[torch.Tensor]] | None = None   # one shadow set per decay
    ema_updates = 0
    ema_lerp_ws = [1.0 - d for d in WEIGHT_EMA_DECAYS]  # fixed for the run, see the T2.2 note

    while step < args.num_steps:
        elapsed = time.monotonic() - budget_started
        # T2.1: recomputed every step from the wall clock, exactly like lrm and wd_progress, so
        # the three schedules cannot drift apart. Only the ACCUMULATION COUNT changes -- the micro
        # shape is fixed, so no graph is invalidated at a stage boundary. Deliberately shares the
        # pre-reset `elapsed` quirk with lrm/wd_progress: for steps 1-4 the clock still includes
        # the compile, so a short budget can briefly report a later stage before budget_started is
        # reset at TRAIN_STARTUP_STEPS_EXCLUDED. Matching the existing schedules is worth more than
        # special-casing four steps out of ~2200.
        ramp_progress = min(1.0, elapsed / max(1e-6, args.max_train_seconds))
        grad_accum_now, batch_lr_scale = batch_ramp_grad_accum(ramp_progress, grad_accum_steps)
        if BATCH_RAMP:
            inv_gas_host.fill_(1.0 / grad_accum_now)
            inv_gas_dev.copy_(inv_gas_host)
            stage = batch_ramp_stage(ramp_progress)
            if stage != ramp_stage_logged:
                ramp_stage_logged = stage
                print0(f"batch_ramp: step {step} progress {ramp_progress:.3f} -> stage "
                       f"{'ABC'[stage]} grad_accum={grad_accum_now} "
                       f"tokens={grad_accum_now * world_tokens_per_micro:,} "
                       f"lr_scale={batch_lr_scale:.4f}")

        synchronize(device)
        t0 = time.monotonic()
        # T1.4: with ADAM_EVERY_N > 1 the AdamW groups update on the LAST step of each
        # N-step window ((step+1) % N == 0, i.e. odd steps when N=2) and their gradients are
        # zeroed only at the START of a window (step % N == 0). With N == 1 both are always
        # True and every branch below is the historical one.
        adam_step_now = (step + 1) % ADAM_EVERY_N == 0
        zero_adamw_now = step % ADAM_EVERY_N == 0
        if grad_flat_buffer is not None:
            # zero_(set_to_none=False)-equivalent: keeps every p.grad pointed at its
            # view into the buffer rather than severing the link with a fresh None.
            grad_flat_buffer.zero_()
        elif zero_adamw_now:
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad_selective(zero_adamw=False)
        last_loss = None
        for _ in range(grad_accum_now):
            loss = model(x, y)
            last_loss = loss.detach()
            # See inv_gas_dev above: device scalar under BATCH_RAMP, literal float otherwise.
            (loss * inv_gas_dev if BATCH_RAMP else loss / grad_accum_steps).backward()
            x, y, loader_state = fetch()

        if grad_flat_buffer is not None:
            if GRAD_CLIP > 0:
                clip_grad_norm_flat_(grad_flat_buffer, GRAD_CLIP)
            sync_gradients_flat(grad_flat_buffer)
        else:
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(clip_main_params, GRAD_CLIP,
                                               foreach=args.clip_foreach)
                if clip_ngram_params:
                    torch.nn.utils.clip_grad_norm_(clip_ngram_params, GRAD_CLIP,
                                                   foreach=args.clip_foreach)
            sync_gradients(model, skip_param_ids=None if adam_step_now else adamw_skip_ids)

        lrm = lr_multiplier(step, elapsed=elapsed, max_train_seconds=args.max_train_seconds)
        # T3.1: warmup-only multiplier for the groups that opt out of the warmdown (the
        # hashed n-gram tables). Identical to lr_multiplier() during warmup and pinned at
        # 1.0 afterwards, so those groups hold a FLAT lr for the whole run. It settles on
        # exactly one value after WARMUP_STEPS, so it costs no extra graph constants.
        lrm_flat = min(1.0, (step + 1) / max(1, WARMUP_STEPS))
        wd_progress = min(1.0, elapsed / max(1e-6, args.max_train_seconds))
        for group in optimizer.param_groups:
            # T2.1: batch_lr_scale is 1.0 unless BATCH_RAMP is on, so this multiplication is the
            # identity on the default path. It costs nothing even when it is not: group["lr"] is
            # already handed to _device_scalar, which writes whatever float it is given into a
            # persistent device tensor, so the graph never sees the value.
            group["lr"] = group["initial_lr"] * batch_lr_scale * (lrm_flat if group.get("no_lr_decay") else lrm)
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum(step, wd_progress)
                group["weight_decay"] = WEIGHT_DECAY * (1.0 - wd_progress) if WD_DECAY_TO_ZERO else WEIGHT_DECAY
        optimizer.step(update_adamw=adam_step_now)
        # T2.2: after the step, so the average is over post-update weights. Eager by construction
        # (this is the same region as optimizer.step, which torch.compile never sees) because
        # torch._foreach_* cannot be lowered by backend="neuron".
        if USE_WEIGHT_EMA and ramp_progress >= WEIGHT_EMA_START_FRAC:
            # no_grad is load-bearing, not decoration: the live tensors are leaf Parameters with
            # requires_grad=True, so an in-place lerp_ into a non-requiring shadow OUTSIDE no_grad
            # succeeds but makes the shadow a graph node holding a reference to the parameters --
            # a new autograd graph retained every step.
            with torch.no_grad():
                if ema_shadows is None:
                    ema_shadows = [[p.detach().clone() for p in ema_live] for _ in ema_lerp_ws]
                    print0(f"weight_ema: seeded at step {step} progress {ramp_progress:.3f} "
                           f"decays={WEIGHT_EMA_DECAYS} tensors={len(ema_live)} each")
                else:
                    for shadow, w in zip(ema_shadows, ema_lerp_ws):
                        torch._foreach_lerp_(shadow, ema_live, w)
                    ema_updates += 1
        for m in cached_linears:  # T1.10: prebuilt list; empty when USE_WEIGHT_CACHE is off
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

        # T2.1: the step's ACTUAL token count, which is TOTAL_BATCH_SIZE only when the ramp is off.
        step_tokens = grad_accum_now * world_tokens_per_micro
        total_train_tokens += step_tokens
        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        tok_per_sec = step_tokens / max(dt, 1e-6)
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

        if step == 50 and rank == 0:
            # Host-only projection (no device op): will the step cap fire before the time budget does?
            window = sorted(recent_step_times[-20:])
            median_dt = window[len(window) // 2]
            time_to_step_cap = (args.num_steps - step - 1) * median_dt
            time_left = args.max_train_seconds - (time.monotonic() - budget_started)
            if time_to_step_cap < time_left:
                print(f"WARNING: at {median_dt:.2f}s/step the --num-steps {args.num_steps} cap will stop training "
                      f"~{time_left - time_to_step_cap:.0f}s BEFORE max_train_seconds={args.max_train_seconds}s; "
                      f"the wall-clock LR schedule will end without cooldown. Intentional for fixed-step sweeps, "
                      f"wrong for a scored submission.", flush=True)
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
            "weight_ema_updates": ema_updates,
            "weight_ema_decay": WEIGHT_EMA_DECAYS[0] if ema_shadows is not None else None,
        }, ema_state=(weight_ema_state(orig_model, ema_names, ema_shadows[0])
                      if ema_shadows is not None else None))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    budget_elapsed_after_save = time.monotonic() - budget_started

    public_eval = None
    public_eval_alt = None
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
        if args.eval_seq_len:
            # Second eval of the SAME weights at a shorter context. Purely diagnostic: the
            # scorer's own context length is not published, and the 2026-09-11 calibration
            # (same checkpoint: 1.033421 at 2048 vs 1.049535 at 1024, board 1.0548) says the
            # length we are graded at is probably NOT the one we train at.
            public_eval_alt = evaluate_bpb(
                orig_model,
                tokenizer,
                DEVICE_BATCH_SIZE,
                args.eval_seq_len,
                device,
                split="public_val",
                max_tokens=args.eval_tokens,
                timeout_seconds=EVAL_TIMEOUT_SECONDS,
            )

    # T2.2: evaluate the EMA weights too, in the SAME process, on the SAME eval window. This is
    # the whole reason the shadow is kept on device instead of only written to the checkpoint: raw
    # and averaged differ by nothing except the averaging, so the comparison has zero run-to-run
    # noise and needs no paired run. Safe to clobber the live weights here because the checkpoint
    # (which holds the raw set) is already written and training is over. The bf16 weight cache has
    # to be refreshed or the Linears would keep serving the pre-swap copies.
    ema_evals: list[tuple[float, dict, dict | None]] = []   # (decay, eval@SEQ_LEN, eval@alt)
    ema_eval_seconds = 0.0
    if args.eval_public and ema_shadows is not None:
        ema_eval_started = time.monotonic()
        for decay, shadow in zip(WEIGHT_EMA_DECAYS, ema_shadows):
            with torch.no_grad():
                for p, s in zip(ema_live, shadow):
                    p.copy_(s)
            for m in cached_linears:
                m.refresh_weight_cache(COMPUTE_DTYPE)
            main = evaluate_bpb(
                orig_model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, device,
                split="public_val", max_tokens=args.eval_tokens,
                timeout_seconds=EVAL_TIMEOUT_SECONDS,
            )
            alt = None
            if args.eval_seq_len:
                alt = evaluate_bpb(
                    orig_model, tokenizer, DEVICE_BATCH_SIZE, args.eval_seq_len, device,
                    split="public_val", max_tokens=args.eval_tokens,
                    timeout_seconds=EVAL_TIMEOUT_SECONDS,
                )
            ema_evals.append((decay, main, alt))
        ema_eval_seconds = time.monotonic() - ema_eval_started

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
        print(f"nki_muon:          {args.nki_muon}")
        print(f"compile_muon:      {args.compile_muon}")
        print(f"mudd:              {args.mudd}")
        print(f"dcmha:             {args.dcmha}")
        print(f"mamba2:            {args.mamba2}")
        print(f"mamba2_pattern:    {args.mamba2_pattern}")
        print(f"nki_ssd:           {args.nki_ssd}")
        print(f"rwkv:              {args.rwkv}")
        print(f"rwkv_pattern:      {args.rwkv_pattern}")
        print(f"fp8_linear:        {args.fp8_linear}")
        # T3.1. Printed as one line so the experiment log can tell two n-gram arms apart
        # (SUMMARY_KEYS in research/run_experiment.py has the matching entry).
        print(f"ngram_ve:          {args.ngram_ve} tables={len(orig_model.ngram_embeds)} "
              f"rows={orig_model.ngram_rows} dim={orig_model.ngram_dim} "
              f"index_gather={NGRAM_VE_INDEX_GATHER} "
              f"trigram={args.ngram_ve_trigram} opt={args.ngram_ve_opt} "
              f"flat_lr={args.ngram_ve_flat_lr} bf16={args.ngram_ve_bf16}")
        # T4.2 / T3.9, same reason: one line carrying every knob of this block, so two arms are
        # distinguishable in experiment-log.jsonl without re-reading the run log
        # (SUMMARY_KEYS in research/run_experiment.py has the matching entries).
        print(f"out_bigram:        {USE_OUT_BIGRAM} bf16={OUT_BIGRAM_BF16} "
              f"gate_bias={OUT_BIGRAM_GATE_BIAS} lr={OUT_BIGRAM_LR}")
        print(f"oneliners:         relu2_tau={RELU2_TAU} mlp_sandwich={USE_MLP_SANDWICH_NORM} "
              f"head_gate={USE_HEAD_GATE} hg_norm={HEAD_GATE_NORM} hg_ch={HEAD_GATE_CHANNELS} "
              f"hg_mul={HEAD_GATE_MUL} hg_bias={HEAD_GATE_BIAS} "
              f"out_pool={USE_OUT_POOL} pool_layers={OUT_POOL_LAYERS} "
              f"qk_shift={USE_QK_SHIFT} qk_beta={QK_SHIFT_BETA} qk_freeze={QK_SHIFT_FREEZE} "
              f"x0_gate={USE_X0_GATE} "
              f"block_nudge={USE_BLOCK_NUDGE} nudge_site={BLOCK_NUDGE_SITE} "
              f"qkv_norm_cse={USE_QKV_NORM_CSE} parallel_block={USE_PARALLEL_BLOCK} "
              f"par_coalesce={PARALLEL_BLOCK_COALESCE} "
              f"epoch_shuffle={USE_EPOCH_SHUFFLE} epoch_shuffle_seed={EPOCH_SHUFFLE_SEED}")
        # T3.2, same one-line convention. `tables` and `site` are read off the built model rather
        # than off the flags, so a share-mode run visibly reports 0 own tables and the resolved
        # absolute block index rather than the raw "-1".
        print(f"engram:            {USE_ENGRAM} site={orig_model.engram_site} "
              f"orders={ENGRAM_ORDERS} heads={ENGRAM_HEADS} "
              f"tables={0 if orig_model.engram is None else len(orig_model.engram.tables)} "
              f"mult={ENGRAM_TABLE_MULT} mem_dim={ENGRAM_MEM_DIM} "
              f"e_dim={0 if orig_model.engram is None else orig_model.engram.e_dim} "
              f"key_dim={0 if orig_model.engram is None else orig_model.engram.key_dim} "
              f"conv_k={ENGRAM_CONV_KERNEL} share={ENGRAM_SHARE_NGRAM_TABLES} "
              f"bf16={ENGRAM_BF16}")
        # T2.1, same reason: one line carrying every ramp knob, so two ramp arms are
        # distinguishable in experiment-log.jsonl without re-reading the run log.
        print(f"batch_ramp:        {BATCH_RAMP} grad_accum={BATCH_RAMP_GRAD_ACCUM} "
              f"bounds={BATCH_RAMP_BOUNDS} lr_exp={BATCH_RAMP_LR_EXP} "
              f"ref_grad_accum={grad_accum_steps} last_grad_accum={grad_accum_now}")
        # T2.2, same one-line convention.
        print(f"weight_ema:        {USE_WEIGHT_EMA} decays={WEIGHT_EMA_DECAYS} "
              f"start_frac={WEIGHT_EMA_START_FRAC} seeded={ema_shadows is not None} "
              f"updates={ema_updates}")
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
        if public_eval_alt is not None:
            print(f"eval_alt_seq_len:  {args.eval_seq_len}")
            print(f"public_val_bpb_alt: {public_eval_alt['val_bpb']:.6f}")
            print(f"eval_alt_seconds:  {public_eval_alt['eval_seconds']:.1f}")
        # T2.2: same weights-except-averaging, same eval window, so the delta against
        # public_val_bpb / public_val_bpb_alt above IS the effect of the EMA, with no run-to-run
        # noise in it at all.
        if ema_evals:
            # The PRIMARY decay (the one in the checkpoint) gets its own keys; every arm also
            # appears on one summary line so a multi-decay run is one log row, not several.
            primary_decay, primary, primary_alt = ema_evals[0]
            print(f"public_val_bpb_ema: {primary['val_bpb']:.6f}")
            print(f"ema_delta:         {primary['val_bpb'] - public_eval['val_bpb']:+.6f}")
            print(f"eval_ema_seconds:  {ema_eval_seconds:.1f}")
            if primary_alt is not None:
                print(f"public_val_bpb_ema_alt: {primary_alt['val_bpb']:.6f}")
                print(f"ema_delta_alt:     "
                      f"{primary_alt['val_bpb'] - public_eval_alt['val_bpb']:+.6f}")
            arms = " ".join(
                f"{d}={m['val_bpb']:.6f}/{m['val_bpb'] - public_eval['val_bpb']:+.6f}"
                + (f"|{a['val_bpb']:.6f}/{a['val_bpb'] - public_eval_alt['val_bpb']:+.6f}"
                   if a is not None else "")
                for d, m, a in ema_evals
            )
            print(f"weight_ema_arms:   raw={public_eval['val_bpb']:.6f}"
                  + (f"|{public_eval_alt['val_bpb']:.6f}" if public_eval_alt is not None else "")
                  + f" {arms}")

    cleanup_runtime()


if __name__ == "__main__":
    main()
