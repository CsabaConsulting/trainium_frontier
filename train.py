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
    get_token_bytes,
    make_dataloader,
    print0,
    record_step_boundary,
)


# -----------------------------------------------------------------------------
# train.py's own copy of prepare.make_dataloader(), line-for-line identical in
# behaviour (it reuses prepare's private _document_batches generator and
# get_dist_info). The only difference is the injectable _document_batches_fn
# seam, which lets USE_EPOCH_SHUFFLE below change the data ORDER without
# touching the packer -- i.e. without confounding a data-ORDER experiment with
# a data-PACKING one. Changing the data sampling strategy in train.py is
# explicitly permitted by the README; prepare.py itself is untouched.
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
N_KV_HEADS = 0  # Sub-1.0 plan P2 (2026-09-16): grouped-query attention KV-head count. 0 (default) =
# dense attention (n_kv_heads = n_head), bit-identical to every previous run. Set to a divisor of
# n_head to shrink the K/V projections (the QK^T + softmax work stays full-width by design, see
# CausalSelfAttention.forward). NOTE the granularity limit: at the locked width (n_head=5) the
# only nontrivial choice is 1 (MQA); richer choices (2, 4) need a width with a composite n_head
# (e.g. n_embd=768 -> 6 heads -> kv 2/3; n_embd=1024 -> 8 heads -> kv 2/4).

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

# Defense-in-depth for the 2026-09-10 failure mode above, in case a future edit reintroduces a
# step cap that binds before the clock does: below MIN, a step-cap exit is treated as an
# intentional short/smoke run (this week's research/scratch/*.sh runs stay under ~20%
# elapsed/max_train_seconds) and is not warned about; at/above MAX it is close enough to the
# deadline to count as an on-time exit rather than the clock never getting to run its cooldown.
# See the step_cap_hit/time_budget_hit check in the training loop below.
STEP_CAP_GUARD_MIN_PROGRESS = 0.5
STEP_CAP_GUARD_MAX_PROGRESS = 0.98

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
# CHECKPOINT_RESERVE_SECONDS raised from 5.0 on 2026-09-14, when the locked config's checkpoint went
# from 0.49 GB (190.1M params) to 1.30 GB (592.9M, n-gram VE tables at mult 128 on two layers, four
# tables of 1048576 x 128). torch.save of that file measured
# 1.0 s in one run and 9.9 s in the next -- same file, same box, minutes apart, so it is write-back
# and page-cache state, not size alone. The 9.9 s one overshot the old 5.0 s reserve and printed
# `over_budget: True` at charged_seconds_est 1804.4 for a 1800 s budget: scorable, but 4.4 s into a
# margin that does not exist. 15.0 s covers the measured worst case with 50% headroom and costs ~18
# steps, i.e. ~0.0007 bpb at the -0.077/ln(steps) slope. Cheap insurance; revisit if the checkpoint
# grows again, and note that USE_WEIGHT_EMA would double it.
CHECKPOINT_RESERVE_SECONDS = 15.0
# Covers the launch -> module-import gap this process cannot observe when
# checking its startup time against the allowance cap.
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0

LOGIT_SOFTCAP = 15.0
LOSS_BYTE_WEIGHT = False  # Sub-1.0 plan P4 (2026-09-16): weight each token's training loss by
# its byte length, aligning the objective with the bpb metric (rare long tokens dominate bpb but
# not mean CE). Default OFF = plain mean, bit-identical to every previous run. Read-only w.r.t.
# the rules: the weights come from the provided tokenizer artifact's id->bytes map (same source
# as prepare.py's token_bytes.pt), no external data, no extra state, eval path untouched.
BYTE_LOSS_W: torch.Tensor | None = None  # mean-normalized float32 weights, built once in main()

EMBEDDING_LR = 0.30
UNEMBEDDING_LR = 0.006
MATRIX_LR = 0.020  # T0.4 resweep (Day 3): part of the combined stack below. Single-axis at 900s: 0.020 -0.000835/-0.000106 (paired), 0.015 +0.002728, 0.030 +0.000552, 0.035 +0.000990 -- an asymmetric curve whose only better-than-baseline point is 0.020, sub-noise on its own, adopted as part of the confirmed stack.
SCALAR_LR = 0.20
# T2.4 (modded-nanogpt record #52), "smoothed scalars": the scalar/skip group's own Adam betas.
# Defaults are exactly the (0.9, 0.95) the group has always used, so passing neither flag is
# bit-identical to every prior run. The idea is that a single learned scalar's gradient is far
# noisier per step than a matrix's (one number absorbing the whole model's disagreement), so a
# longer second-moment window (0.99) plus a smaller step may beat the current aggressive setting.
#
# The historical NaN this group is famous for (see the gate_params comment in setup_optimizer) came
# from a real PROJECTION MATRIX landing here and being driven at 8x MATRIX_LR. That is now
# structurally impossible: gate_params exists precisely to catch 1-D/single-row gate tensors, and
# setup_optimizer asserts the scalar group holds nothing with ndim > 1. Raising beta2 and/or
# lowering the LR both move AWAY from that failure, not toward it.
#
# MEASURED 2026-09-15, CLOSED NEGATIVE, defaults unchanged. beta2 0.99 at the current SCALAR_LR 0.20
# is a null (per-step +0.000215 against a control pair whose spread is 0.000148), and lowering the LR
# on top of it is monotonically worse: 0.10 +0.001190, 0.05 +0.001827. That closes the axis from both
# ends and re-confirms SCALAR_LR 0.20 as at or past the top of the curve.
SCALAR_BETA1 = 0.9
SCALAR_BETA2 = 0.95
# T2.3 (modded-nanogpt record #50), "cautious weight decay": apply weight decay only to the
# elements where the Adam update and the parameter itself agree in sign, i.e.
# `mask = (update * p) > 0`, so decay never fights a parameter that the gradient is already
# pushing away from zero. Only the AdamW groups with a nonzero weight_decay are affected, which in
# the locked config is exactly two tensors -- wte (wd 0.001) and lm_head (wd 0.01) -- so the cost
# is ~12 extra eager launches out of ~400 in the optimizer step.
#
# IMPLEMENTATION NOTE (this is why the flag is safe to add to a locked default): both AdamW paths
# were restructured so the decay multiply happens AFTER `update` is computed instead of before.
# That is exactly, not approximately, equivalent with the flag off -- the update depends only on
# exp_avg/exp_avg_sq/grad and never reads p, and the decay reads the same pre-decay p either way.
# research/scratch/cautious_wd_equiv.py checks it bitwise on the real model.
#
# MEASURED 2026-09-15: EXACT NULL, stays OFF. Paired 900s, per-step -0.000087 against a control pair
# whose own spread is 0.000148 -- not a regression, not a win, and on a locked default the tie goes to
# the incumbent. The mask does real work (~50% of elements spared at these shapes); it just does not
# matter at wd 0.001/0.01 over ~2100 steps. Would be worth re-measuring on a config with a large
# embedding weight decay, where the whole decay term is not already near-irrelevant.
USE_CAUTIOUS_WD = False
WEIGHT_DECAY = 0.20  # T0.4 resweep (Day 3): part of the combined stack below. Single-axis at 900s: 0.20 -0.000557/-0.000533 (paired, spread 0.000024 -- the tightest pair in the log), 0.35 -0.000458, 0.05 -0.000019. Flat-but-reproducible axis; 0.20 chosen as the middle of the better region. NOTE: combining 0.20 with WD_DECAY_TO_ZERO=False came in +0.001828, i.e. WORSE than either alone -- they are effectively the same knob (total integrated decay), so WD_DECAY_TO_ZERO stays True.
VE_WD = 0.0  # weight decay for the value_embeds AdamW group (separate from the shared embedding_params group)
EMBED_ADAMW_BETA2 = 0.95  # Tier-0 resweep confirms still correct at the new ~2000-step regime: 0.95 (995/1.094327) beats 0.995 (977/1.097342) again. See trn2-experiment-results.md
WARMUP_STEPS = 0
GRAD_CLIP = 1.0  # Scavenger hunt real win, locked: never tested on Trn2 before ("deferred, not GPU-retested"). HLAT (real 7B/70B Trainium pretrain) used clip=1.0. Real repeat-confirmed win at 900s (999/1.0912, 1002/1.090864 vs 0.0-baseline cluster 1002-1004/1.09367-1.09399) and full 1800s (2000/1.042278 vs prior best 1987/1.043651). See trn2-experiment-results.md
CLIP_AFTER_REDUCE = False  # Fresh-start audit: opt-in global-batch clipping after gradient AVG.
# Default keeps the historical per-rank clip-before-AVG update. These are different algorithms:
# averaging locally clipped gradients can change direction, not just the effective learning rate.
# The new order is an experiment, not an assumed quality win; compare at the full 1800s budget.
LOSS_SCALE_DYNAMIC = False  # Sub-1.0 plan P2: DYNAMIC loss scaling companion.
# Static --loss-scale unfreezes fp8 (9.01->7.71 by step 10) but a fixed scale
# eventually overflows (NaN guard trips). Dynamic mode adapts: skip+halve on
# overflow, double every LOSS_SCALE_GROWTH_INTERVAL clean steps (cap MAX).
# Default OFF; static --loss-scale path untouched when off.
LOSS_SCALE_GROWTH_INTERVAL = 500
LOSS_SCALE_MAX = 8192.0
loss_scale_clean = 0
loss_scale_skips = 0
LOSS_SCALE = 1.0  # Sub-1.0 plan P2 (2026-09-17): STATIC loss scaling for compiler
# fp8 auto-cast. Full-model grads (~1e-6/elt over 593M params) underflow fp8_e4m3
# (min normal ~0.0156) without it; the tiny-probe grads survived only because few
# params concentrate the norm. 1.0 (default) = unscaled, bit-identical to every
# previous run (x*1.0 and x/1.0 are IEEE-exact, and the unscale call is skipped
# entirely at 1.0). Use power-of-2 values (128/1024): exact in fp32, no rounding.
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
# ---------------------------------------------------------------------------
# Plan §11.1: three SCHEDULE-LEVEL tricks from Recursive's optimized_from_karpathy.py (0.9109 on
# the nanochat autoresearch task, the strongest external evidence in this competition). All three
# are pure optimizer-schedule changes -- they add no new tensors, no new ops, and no new parameters
# -- so their throughput term is expected to be exactly zero and any delta they produce is a
# per-step-quality delta. All three default to EXACTLY the current behaviour (see each knob).
#
# The shared implementation discipline for all three is the USE_TENSOR_LR_SCALARS lesson: a per-step
# CHANGING Python float that reaches a device op is baked into the Neuron graph as a constant and
# forces a fresh neuronx-cc compile every step. Which ops actually bake is op-specific and MEASURED,
# not assumed (see the muon_momentum docstring: `lerp_.Scalar`'s weight demonstrably does NOT bake,
# while `add_(alpha=)` and `mul_(1 - lr*wd)` do). Inside a COMPILED region (USE_COMPILE_MUON) the
# rule is absolute: under dynamic=False Dynamo guards on every Python float argument, so anything
# that changes per step must arrive as a device tensor. That is why (2) below routes `1 - beta2`
# through _muon_compile_scalars instead of passing it as the Python float it is today.
#
# (1) Adam "Demon" beta1 decay. Recursive anneals Adam's beta1 from 0.8 down to 0.55 across the
# run's warmdown. Our AdamW groups already start at beta1 = 0.8 for the embedding family (wte,
# lm_head, value embeddings, the adamw fallback for the n-gram tables) and 0.9 for the scalar and
# gate/conv groups, so the anneal is expressed as a MULTIPLICATIVE shrink on each group's own
# initial beta1, calibrated so a group at DEMON_BETA1_REF lands exactly on DEMON_BETA1_FINAL. The
# rmsprop group (n-gram tables) has no first moment at all and is excluded by its `kind`.
DEMON_BETA1 = False
DEMON_BETA1_FINAL = 0.55  # target beta1 at progress == 1 for a group whose initial beta1 is REF
DEMON_BETA1_REF = 0.8     # the initial beta1 the FINAL above is quoted against (Recursive's value)
# Progress fraction at which the anneal starts. Negative (the default) means "start where the LR
# warmdown starts", i.e. 1 - WARMDOWN_RATIO, which is Recursive's "across its warmdown". Set 0.0 to
# anneal across the whole run instead.
DEMON_BETA1_START_FRAC = -1.0
#
# (2) NorMuon beta2 ramp. Recursive ramps their Muon variant's second-moment beta2 from 0.95 to
# 0.97 across the run. This repo's Muon ALREADY carries exactly that mechanism: `beta2` in the muon
# groups is the decay of `second_momentum_buffer`, a per-row second moment of the ORTHOGONALIZED
# update used to rescale it -- that is NorMuon's neuron-wise adaptive normalization, not Adam's
# beta2 on the raw gradient. So the port is 1:1 in shape. It is NOT 1:1 in value: our tuned
# constant is 0.90, not 0.95, so what transfers is "ramp beta2 UP over the run", and the endpoints
# have to be swept here rather than copied. MUON_BETA2 replaces what used to be a `beta2=0.90`
# literal in setup_optimizer (pure refactor, same value).
MUON_BETA2 = 0.90
MUON_BETA2_FINAL = 0.0  # 0.0 (default) = OFF: hold MUON_BETA2 flat, and keep passing `1 - beta2`
                        # to the compiled body as the Python float it is today. > 0 ramps linearly
                        # from MUON_BETA2 to this over elapsed/max_train_seconds.
#
# (3) Per-depth LR / momentum scaling. Recursive scales LR by x1.15 at the bottom (earliest) layer
# down to x0.85 at the top, and momentum from 0.90 at the bottom to 0.97 at the top. Our Muon param
# groups are keyed by (shape, transposed) and STACK every param of that shape across layers into one
# [K, rows, cols] buffer, which is what makes the batched Newton-Schulz worth having -- splitting
# them per layer would give 30-odd single-parameter groups and throw that away. So the multiplier is
# carried as a per-parameter [K, 1, 1] fp32 device tensor that broadcasts over the stacked buffer:
# the group structure, the launch count and the compiled graph count are all unchanged, and the only
# added device work is that lr/momentum arrive as [K,1,1] instead of 0-dim operands.
#
# LR is scaled multiplicatively (BOTTOM at layer 0, TOP at layer n_layer-1, linear in between).
# Momentum CANNOT be: 0.95 x 1.15 > 1 diverges, and the momentum here is itself scheduled (0.85 ->
# 0.95 warmup, then optionally back down). So per-depth momentum is expressed as a multiplier on
# (1 - momentum), i.e. on the update's weight, derived from the two target momenta at the reference
# global momentum: mult = (1 - target) / (1 - REF). Recursive's 0.90/0.97 at REF 0.95 gives 2.0 at
# the bottom and 0.6 at the top, and the whole global schedule keeps its shape underneath.
# Params outside a transformer block (the n-gram up-projections, Engram's W_K/W_V/W_Q) have no
# depth and get multiplier 1.0.
MUON_DEPTH_LR_BOTTOM = 1.0   # LR multiplier at layer 0. Recursive: 1.15. 1.0 = OFF.
MUON_DEPTH_LR_TOP = 1.0      # LR multiplier at the last layer. Recursive: 0.85. 1.0 = OFF.
MUON_DEPTH_MOM_BOTTOM = 0.0  # target momentum at layer 0 when global momentum == REF. Recursive:
                             # 0.90. 0.0 = OFF (both BOTTOM and TOP must be > 0 to enable).
MUON_DEPTH_MOM_TOP = 0.0     # ... at the last layer. Recursive: 0.97. 0.0 = OFF.
MUON_DEPTH_MOM_REF = 0.95    # the global momentum the two targets above are quoted against; this is
                             # muon_momentum()'s warmed-up plateau value.
# ---------------------------------------------------------------------------
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
# Plan 11.1 (Recursive's from-vanilla run): BYTE-FEATURE `wte` INIT. Instead of
# normal_(0, 0.8), seed each token's embedding row from a hashed bag of byte n-grams of the
# token's OWN byte string, so tokens with similar surface form start with correlated
# embeddings and the model does not have to rediscover orthography from data alone.
#
# LEGALITY: the only input is `tokenizer.enc.decode_single_token_bytes(id)`, i.e. the fixed,
# provided tokenizer artifact's own id->bytes mapping. No external data, no pretrained vectors,
# no corpus statistics.
#
# ZERO throughput cost by construction: this runs once, on the host, between init_weights() and
# model.to(device), and changes no op in the forward/backward graph. It also draws NO torch RNG,
# so every other tensor in the model is initialized bit-identically to the control arm -- the
# only difference between the two arms is the contents of wte.weight.
#
# NOTE `GPT.forward` does `x = norm(wte(idx))`, so the row SCALE is invisible to the forward
# pass; it matters only through AdamW's effective relative step size (and wd=0.001) on that
# tensor. The features are therefore rescaled to the same per-row RMS the normal_(0, 0.8) init
# produces, not left at their raw counting scale.
#
# MEASURED 2026-09-15, CLOSED NEGATIVE, stays OFF. Paired 900s block, control pair spread 0.000148:
# per-step +0.002188 at mix 1.0 (1.054655 / 1055 steps vs control mean 1.052504 / 1054.5), and the
# throughput term is 0.000 as designed. Confirmed by a monotone DOSE-RESPONSE on the mix rather than
# a bare repeat: mix 0.25 +0.001238, mix 0.50 +0.001470, mix 1.00 +0.002188. Already harmful at mix
# 0.25, so it is the correlation STRUCTURE, not the amplitude. Reading: a structured prior on wte is
# a constraint the optimizer pays to escape, and at 8192 rows x ~2100 dense-update steps this table
# is the best-trained tensor in the model -- it does not need one. See the 2026-09-15 section of
# research/trn2-experiment-results.md.
USE_BYTE_WTE_INIT = False
BYTE_WTE_INIT_MIX = 1.0    # 1.0 = pure byte features; 0 <= mix < 1 blends with the random init as
# mix*feat + sqrt(1-mix^2)*random, which keeps the row RMS at ~0.8 for any mix because the two
# components are near-orthogonal in 640 dimensions. A blend hedges the one real risk of a pure
# feature init: two tokens whose hashed feature sets collide would start life as the SAME row,
# and a random component guarantees every row stays distinct.
BYTE_WTE_INIT_NGRAM = 3    # highest byte-n-gram order hashed (1..N, plus \x02/\x03 boundary
# markers around the token's bytes so a prefix feature is distinguishable from a suffix one).
BYTE_WTE_INIT_STD = 0.8    # target per-row RMS; MUST track the normal_ std used in init_weights.
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
# (this reuses its private _row_groups_for_split, exactly as make_dataloader_requeue reuses
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
NGRAM_VE_LAYERS = "1,-1"  # WHICH layers get a bigram table. "" = every has_ve() layer, which is
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
#
# LAYER 1 IS BACK ON, since 2026-09-14, and it is the largest single win since the tables themselves.
# The +34% figure above ({5} -> {1,5}, 2.768 -> 3.711 s/step) was measured at ONE RANK, and it does
# not survive the move to the 4-rank shape the competition actually runs: with the residual stream
# already sharded four ways the cross-pcore reduce-add is far cheaper, and the real cost of adding
# layer 1 is -5.86% throughput (177394 -> 167008 tok/s, 120-step instrument, 4 ranks, LNC=2). The
# +13.67% budget from --compile-muon (T1.7, +3.38%) plus the frozen qk-shift (+9.50%) pays for it
# with room to spare. Step-matched 1800s arms, same day, same commit, causality_max_abs 0.0:
#   layers "-1"   (control)  2141 steps  1.019126   190.1M     <- reproduces the locked 1.018985
#   layers "1,-1"            2139 steps  1.007376   324.5M     <- -0.0118 bpb at an EXACT step match
#   layers "1,-1" @2187      2187 steps  1.005973   324.5M     <- +46 steps of leftover budget
#   layers "1,-1" mult 128   2135 steps  1.004363   592.9M     <- LOCKED (see NGRAM_VE_TABLE_MULT)
# layers "3,-1" and "1,3,-1" are both WORSE than the control on the 120-step proxy (3.9973 and
# 3.9855 vs 3.9608), so layer 3 is not a budget question, it is harmful at this depth. And the
# position space is CLOSED: has_ve() is `i % 2 == (n_layer-1) % 2`, so at DEPTH 6 the only VE layers
# are {1,3,5} -- "0,-1", "2,-1", "1,2,-1" parse fine and then silently do nothing (measured: exactly
# 190.1M params and the control's tok/s). Any even-layer attempt must change has_ve() first.
NGRAM_VE_TABLE_MULT = 128 # rows per table = this x padded_vocab_size (128 x 8192 = 1048576).
# The 2026-09-12 1800s sweep with NGRAM_VE_LAYERS="-1", bigram+trigram, dim 128 (bpb / steps /
# params) said capacity saturates at 64: mult 32 -> 1.020516 / 2173 / 123.0M, mult 64 -> 1.018985 /
# 2139 / 190.1M, mult 128 -> 1.019171 / 2067 / 324.3M, i.e. doubling was 0.0002 WORSE for 72 fewer
# steps. **That saturation is a property of the ONE-LAYER configuration, and it reverses once layer 1
# is also injecting** (2026-09-14, both arms at a legal <=1800s budget, same day/commit):
#   layers "1,-1" mult  64   2187 steps  1.005973   324.5M
#   layers "1,-1" mult 128   2135 steps  1.004363   592.9M   <- -0.0016 as run, -0.0035 step-matched
# despite 52 fewer steps, so the extra capacity outruns the 0.077/ln(steps) penalty it costs. Two
# injection sites make more table rows useful; one site cannot use them. Going further is arithmetic
# that does NOT close: dim 256 on top of layer 1 is 0.8427 s/step (2136 steps) with a WORSE 120-step
# proxy than mult 128 (3.9377 vs 3.9343), and dim 256 + mult 128 together cost 24.3% (1736 steps),
# whose +0.0177 step penalty no measured capacity gain repays.
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
NGRAM_VE_BF16 = True      # store tables/gradients in bf16; state follows unless overridden below
NGRAM_RMS_FP32 = False   # opt-in FP32 RMSProp STATE AND ARITHMETIC; BF16 tables/grads stay BF16.
USE_NKI_NGRAM_RMS = False  # opt-in fused FP32 table update; requires NGRAM_RMS_FP32.
# With beta2=.999, a zero-gradient step's 0.1% moment decay can round away in BF16.
# Cast gradients BEFORE squaring, keep second moments/denominators/updates in FP32,
# and round only when subtracting the FP32 update into the existing BF16 parameter.
# This changes arithmetic as well as state storage and requires a quality/time A/B.
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
USE_QK_SHIFT = True        # causal token shift on the Q/K inputs only: q_in = k_in = x + beta*x_prev
                           # with a learned PER-DIM beta, V left unshifted. Different mechanism from
                           # USE_LOCAL_CONV (T0.1), which is a learned depthwise conv applied to the
                           # residual stream itself and feeds all three of Q/K/V.
                           # ON since 2026-09-14, but ONLY as the FROZEN beta=0.0 identity below --
                           # i.e. it is on for its +9.50% throughput, not for its mathematics. The
                           # learned form (beta 0.3, trainable) is NOT what is enabled here.
QK_SHIFT_BETA = 0.0        # per-dim beta init (Recursive: 0.3). Not a step-0 no-op by design.
                           # 0.0 since 2026-09-14: with QK_SHIFT_FREEZE this is a bit-exact identity
                           # (see below), which is the whole point -- the win is the compiled graph.
QK_SHIFT_FREEZE = True     # T3.9-followup / T1.x throughput probe: keep beta OUT of every optimizer
                           # group and set requires_grad=False, so it stays at QK_SHIFT_BETA forever.
                           # With --qk-shift-beta 0.0 that makes the whole shift a MATHEMATICAL
                           # IDENTITY at every step (x + 0.0*x_prev == x bit-exactly in IEEE fp, and
                           # the backward contribution through the shifted branch is scaled by the
                           # same 0.0), while the elementwise ops STAY in the compiled graph. That is
                           # the decisive probe for the observed anomaly that adding almost any
                           # elementwise op to the block body makes a step 8-9% FASTER: if the frozen
                           # arm keeps the speedup with bit-identical losses, the +8% is free.
                           # It did, twice: +9.50% at the locked config (2026-09-14 quadrant) with the
                           # loss curve bit-identical, and the step-matched control at ~2145 steps
                           # reproduced the locked 1.018985 to 3e-6 with this ON. Free throughput.
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
                           # (--qk-shift, --block-nudge --block-nudge-site qk), so it composes
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
                           # position t). s is zero-initialized, and
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
# setup_optimizer() collected only attn/mlp/lambda/ve params, so local_conv.weight was in
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
# gpu-closed-techniques-trn2-revisit-2026-09-12.md §3.2: the PARAMETER-MATCHED, COALESCED form of
# the SwiGLU above, which is the only version of it worth a second hearing on Trn2.
#
# What is different from USE_GATED_MLP, and why each difference matters here:
#  * PARAMETER-MATCHED. USE_GATED_MLP bolts a third n_embd x 4*n_embd matrix onto the existing two,
#    i.e. +50% MLP parameters and +50% MLP FLOPs -- which is why the GPU measurement closed it as
#    "worse AND slower AND bigger" (+0.0039 bpb, +19% step time, 85.9M -> 102.3M params). Here the
#    hidden width is dropped to ~8/3*n_embd instead of 4*n_embd so the three matrices cost the same
#    as the baseline's two: 3*640*1664 = 3,194,880 vs 2*640*2560 = 3,276,800 per layer (-2.5% MLP,
#    -0.08% of the whole 592.9M model). That makes this a test of the ACTIVATION, not of capacity.
#  * COALESCED. gate and up come out of ONE Linear(n_embd, 2*hidden) rather than two, so the block
#    keeps its launch count instead of adding one matmul per layer. On a stack that is
#    dispatch/serialisation-bound with ~67 NEFF launches per micro-step, and where wider matmuls
#    cost ~0.65x proportional time while extra launches cost ~1.0x, folding work into an existing
#    launch is the only shape of change that is ever throughput-neutral.
#
# PRIOR IS LOW AND SAID SO UP FRONT (~15-20%): the small-model consensus (modded-nanogpt, nanochat,
# Recursive, Parameter Golf #340/#344) is ReLU^2 >= SwiGLU at this scale, and that half of the
# closure is hardware-independent. A negative here is the expected outcome, not a failure.
#
# Step-0 no-op is preserved for free: mlp.c_proj is zero-init, so the MLP contributes exactly 0 at
# step 0 whatever the activation is. It is NOT bit-identical to the control beyond that, because
# c_fc changes numel and therefore shifts the init RNG stream for every tensor built after it --
# unavoidable for any real shape change, and the same caveat USE_NGRAM_VE/USE_GATED_MLP carry.
#
# MEASURED 2026-09-15, CLOSED NEGATIVE on BOTH axes, stays OFF. Paired 900s (control pair spread
# 0.000148): 1.057505 / 1043 steps vs control mean 1.052504 / 1054.5, i.e. per-step +0.004157 AND
# 1.09% SLOWER -- despite the identical launch count and 0.7% FEWER flops/token (3.962e8 vs 3.992e8).
# The 15-20% prior was called correctly; the ReLU^2 >= SwiGLU consensus holds on Trn2 too. The
# slowdown is the more transferable half: "same launch count, fewer FLOPs" is NOT a throughput
# guarantee here. Unseparated suspects (not worth device time on a discarded arm): the two
# non-contiguous last-dim slices, and 3328 being a worse output tile than 2560.
USE_COALESCED_SWIGLU = False
COALESCED_SWIGLU_HIDDEN = 0  # 0 = auto: round(8/3 * n_embd / 128) * 128, which is 1664 at n_embd=640
# (the parameter-matched width, since 3*d*h == 2*d*(4*d) gives h = 8/3*d). Set > 0 to force a width
# (must be a multiple of 128 so the matmuls stay on TensorE's tiling).
USE_WEIGHT_CACHE = True  # cache each Linear's dtype-cast weight once per optimizer step instead of every forward call
# T1.6: store every Linear's weight TRANSPOSED, as [in_features, out_features], and compute
# `x @ weight` instead of `F.linear(x, weight)` (which is `x @ weight.T`). The premise is that
# TensorE wants the contraction dimension on partitions, so the [out, in] storage forces the
# compiled graph to carry a transpose per matmul. See transpose_linear_weights() for why the
# conversion happens AFTER init_weights() rather than at construction (bit-exact init), and
# _muon_step for why Muon has to be told the LOGICAL orientation (its `ratio` LR scale is NOT
# transpose-invariant -- the one real trap in this change).
USE_TRANSPOSED_LINEAR = False
USE_DOC_MASK = False  # block attention across document boundaries within a packed row (BOS-derived, data-dependent)
USE_SELECTIVE_GATE = False  # make resid_lambda/x0_lambda data-dependent (Mamba-style selectivity) instead of static scalars
SELECTIVE_GATE_CHANNELS = 32
USE_FUSED_QKV = False  # A/B'd at final locked shape: net negative as expected (206 steps/1.312086 vs baseline 240/1.269318) -- disables the rope+QK-norm kernel fusion, fewer steps and worse quality. See trn2-experiment-results.md
USE_CAUSAL_FASTPATH = True  # Solution 13 winner, A/B'd at final locked shape: real win (265 steps/1.252675 vs baseline 240/1.269318, +0.0166) -- the best of all four Solution-13 toggles, doesn't compose further with USE_TRN2_COMPILER_FLAGS. See trn2-experiment-results.md


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
        # Keep arbitrary scorer sequence lengths and other precisions on the
        # original path. Only the gated FP32 two-tap geometry enters this kernel.
        if (USE_NKI_LOCAL_CONV and x.device.type == "neuron"
                and x.dtype == torch.float32 and self.kernel_size == 2
                and x.shape[1] % 128 == 0 and x.shape[2] % 128 == 0):
            if not _HAS_NKI:
                raise RuntimeError("--nki-local-conv requires the installed native NKI stack")
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
    # Sub-1.0 plan P2 (2026-09-16): grouped-query attention KV-head count. 0 (default) =
    # dense (n_kv_heads = n_head), bit-identical to every previous run. Defaulted (and
    # appended last) so GPTConfig(**payload["model_config"]) still loads every checkpoint
    # written before this field existed -- same pattern as mlp_ratio above.
    n_kv_heads: int = 0


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
# Fused-attention tile sizes (P1 NKI port). Column-0 on purpose: kernel bodies
# reference these and the NKI tracer only resolves module-top-level globals.
_NKI_ATTN_P = 128  # == head dim; also the SBUF partition dim
_NKI_ATTN_KV_TILE = 512
_NKI_ATTN_Q_TILE = 128
# torch's F.rms_norm default eps: measured empirically to be float32 eps (the
# accumulation dtype's epsilon), NOT double eps -- see research notes. Getting
# this wrong is invisible for O(1) inputs and a 6% error for small ones.
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

    @nki.jit
    def _ngram_rmsprop_fp32_nki(p, grad, moment, scalars,
                               beta2: float, eps: float, lanes: int):
        """BF16 table/gradient, FP32 moment and arithmetic; explicit copy-back outside.

        Inputs are contiguous [M,2048] views, with M divisible by 128*lanes.
        scalars[128,2] holds runtime LR and inverse bias correction. Separate
        output buffers were faster than mutable-input alias handling on the
        installed native backend. Two programs use the two physical cores of LNC2.
        """
        m, f = p.shape
        assert lanes in (1, 2) and m % (128 * lanes) == 0 and f == 2048
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
            nisa.scalar_tensor_tensor(dst=nv, data=dv, op0=nl.multiply,
                                      operand0=1.0 - beta2, op1=nl.add, operand1=vs)
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
        return out_p, out_v

    # ---- Fused causal SDPA (P1 NKI port, research/scratch/nki_attn_fwd.py +
    # ---- nki_attn_bwd.py, hw-validated 2026-09-17: fp32 ~1e-6, bf16 fwd
    # ---- format-floor / bwd dQ/dK ~4e-4, dV ~2.5e-3). T/D required params.
    # ---- Tile sizes live as column-0 module constants below (next to
    # ---- _NKI_PARTITION): the NKI tracer does not resolve try-block-level
    # ---- globals (take-2 crash: unbound _NKI_ATTN_Q_TILE).
    @nki.jit
    def _attn_nki_fwd(q, k, v, mask, scale, T, D):
        """q/k/v: [BH*T, D] bf16 HBM. mask: [T, T] bf16 HBM. Returns [BH*T, D]."""
        BH = q.shape[0] // T
        out = nl.ndarray((BH * T, D), dtype=q.dtype, buffer=nl.shared_hbm)

        for bh in nl.affine_range(BH):
            base = bh * T
            # --- assemble [_NKI_ATTN_P, T] tiles (_NKI_ATTN_P == D == 128) from [T, D] row chunks
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
                # --- QK tiles: [_NKI_ATTN_P, _NKI_ATTN_KV_TILE] fp32 PSUM
                n_kv = T // _NKI_ATTN_KV_TILE
                qk_tiles = []
                for _i in range(n_kv):
                    qk_tiles.append(nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_KV_TILE), dtype=nl.float32,
                                               buffer=nl.psum))
                for ki in range(n_kv):
                    # each dst written ONCE: default accumulate=False (first write
                    # initializes PSUM); the flag is only for same-dst K-loops.
                    nisa.nc_matmul(
                        dst=qk_tiles[ki],
                        stationary=q_sbuf[0:_NKI_ATTN_P, r0:r0 + _NKI_ATTN_Q_TILE],
                        moving=k_sbuf[0:_NKI_ATTN_P, ki * _NKI_ATTN_KV_TILE:(ki + 1) * _NKI_ATTN_KV_TILE])
                qk_full = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                for ki in range(n_kv):
                    nisa.tensor_copy(
                        dst=qk_full[0:_NKI_ATTN_P, ki * _NKI_ATTN_KV_TILE:(ki + 1) * _NKI_ATTN_KV_TILE],
                        src=qk_tiles[ki])
                # --- scale + softmax (v1-style explicit fp32)
                nisa.tensor_scalar(dst=qk_full, data=qk_full, op0=nl.multiply,
                                   operand0=scale)
                row_max = nl.max(qk_full, axis=1, keepdims=True)
                # separate dst (NOT in-place): tensor operands broadcast, and the
                # v1 reference uses a distinct output for the subtract.
                sub_full = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=sub_full, data=qk_full, op0=nl.subtract,
                                   operand0=row_max)
                exp_full = nl.exp(sub_full)
                mb = nl.ndarray((_NKI_ATTN_P, T), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=mb, src=mask[r0:r0 + _NKI_ATTN_Q_TILE, :])
                m = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=m, src=mb)
                nisa.tensor_tensor(dst=exp_full, data1=exp_full, data2=m,
                                   op=nl.multiply)
                sum_row = nl.sum(exp_full, axis=1, keepdims=True)
                inv = nl.reciprocal(sum_row)
                norm_full = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=norm_full, data=exp_full, op0=nl.multiply,
                                   operand0=inv)
                scores = nl.ndarray((_NKI_ATTN_P, T), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=scores, src=norm_full)
                # --- AV over kv chunks of 128. scores chunk transposed on-chip;
                # V chunk DMA'd straight from HBM rows (native [kv, d] order) --
                # delibrately NO 3D SBUF buffer: indexed-3D SBUF slices
                # miscompile (partition descriptor), found 2026-09-17.
                acc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.psum)
                for c in range(T // _NKI_ATTN_P):
                    st = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=st, data=scores[0:_NKI_ATTN_P, c * _NKI_ATTN_P:(c + 1) * _NKI_ATTN_P])
                    st_s = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=st_s, src=st)
                    vc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=vc, src=v[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                    nisa.nc_matmul(dst=acc, accumulate=(c > 0),
                                   stationary=st_s, moving=vc)
                ob = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ob, src=acc)
                # NOTE: nl.store does NOT downcast fp32->bf16 on store (garbage
                # out, found 2026-09-17): explicit downcast copy + dma instead.
                ob16 = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ob16, src=ob)
                nisa.dma_copy(dst=out[base + r0:base + r0 + _NKI_ATTN_Q_TILE, :], src=ob16)
        return out

    @nki.jit
    def _attn_nki_bwd(q, k, v, do, mask, scale, dq, dk, dv, dk_buf, dv_buf,
                     T, D):
        """q/k/v/do: [BH*T, D] bf16 HBM. mask: [T, T] bf16 HBM.
        dq/dk/dv: [BH*T, D] bf16 HBM outputs. dk_buf/dv_buf: [BH*T, D]
        fp32 HBM scratch, zeroed by host (cross-qi accumulation)."""
        BH = q.shape[0] // T
        for bh in nl.affine_range(BH):
            base = bh * T
            # --- assemble [_NKI_ATTN_P, T] Q^T / K^T (verbatim fwd path)
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
                # --- recompute fp32 _NKI_ATTN_P block (verbatim fwd path)
                n_kv = T // _NKI_ATTN_KV_TILE
                qk_tiles = []
                for _i in range(n_kv):
                    qk_tiles.append(nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_KV_TILE), dtype=nl.float32,
                                               buffer=nl.psum))
                for ki in range(n_kv):
                    nisa.nc_matmul(
                        dst=qk_tiles[ki],
                        stationary=q_sbuf[0:_NKI_ATTN_P, r0:r0 + _NKI_ATTN_Q_TILE],
                        moving=k_sbuf[0:_NKI_ATTN_P, ki * _NKI_ATTN_KV_TILE:(ki + 1) * _NKI_ATTN_KV_TILE])
                qk_full = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                for ki in range(n_kv):
                    nisa.tensor_copy(
                        dst=qk_full[0:_NKI_ATTN_P, ki * _NKI_ATTN_KV_TILE:(ki + 1) * _NKI_ATTN_KV_TILE],
                        src=qk_tiles[ki])
                nisa.tensor_scalar(dst=qk_full, data=qk_full, op0=nl.multiply,
                                   operand0=scale)
                row_max = nl.max(qk_full, axis=1, keepdims=True)
                sub_full = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=sub_full, data=qk_full, op0=nl.subtract,
                                   operand0=row_max)
                exp_full = nl.exp(sub_full)
                mb = nl.ndarray((_NKI_ATTN_P, T), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=mb, src=mask[r0:r0 + _NKI_ATTN_Q_TILE, :])
                m = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=m, src=mb)
                nisa.tensor_tensor(dst=exp_full, data1=exp_full, data2=m,
                                   op=nl.multiply)
                sum_row = nl.sum(exp_full, axis=1, keepdims=True)
                inv = nl.reciprocal(sum_row)
                p_full = nl.ndarray((_NKI_ATTN_P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=p_full, data=exp_full, op0=nl.multiply,
                                   operand0=inv)
                p16 = nl.ndarray((_NKI_ATTN_P, T), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=p16, src=p_full)
                # --- dO rows for this block
                do_b = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=do_b, src=do[base + r0:base + r0 + _NKI_ATTN_Q_TILE, :])
                # --- recompute O block (fwd AV pattern, fp32 PSUM accum)
                o_acc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.psum)
                for c in range(T // _NKI_ATTN_P):
                    st = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=st, data=p16[0:_NKI_ATTN_P, c * _NKI_ATTN_P:(c + 1) * _NKI_ATTN_P])
                    st_s = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=st_s, src=st)
                    vc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=vc,
                                  src=v[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                    nisa.nc_matmul(dst=o_acc, accumulate=(c > 0),
                                   stationary=st_s, moving=vc)
                o_b = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_b, src=o_acc)
                # --- D = rowsum(dO * O)
                do_f = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=do_f, src=do_b)
                do_o = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=do_o, data1=do_f, data2=o_b,
                                   op=nl.multiply)
                d_row = nl.sum(do_o, axis=1, keepdims=True)
                # --- Q rows for the dK contributions of this block
                q_b = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_b, src=q[base + r0:base + r0 + _NKI_ATTN_Q_TILE, :])
                # --- dQ accumulates over kv chunks in PSUM
                dq_acc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.psum)
                for c in range(T // _NKI_ATTN_P):
                    vc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=vc,
                                  src=v[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                    # dV contrib: _NKI_ATTN_P^T @ dO -> fp32 HBM scratch add.
                    # Contraction is over q (the shared partition dim),
                    # so stationary is the _NKI_ATTN_P block NATIVE [q, k] rows --
                    # a transpose here contracts k against q (silent
                    # garbage: hw FAIL dV 1.16, found 2026-09-17).
                    dv_c = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dv_c,
                                   stationary=p16[0:_NKI_ATTN_P, c * _NKI_ATTN_P:(c + 1) * _NKI_ATTN_P],
                                   moving=do_b)
                    dv_old = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32,
                                        buffer=nl.sbuf)
                    nisa.dma_copy(dst=dv_old,
                                  src=dv_buf[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                    dv_new = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32,
                                        buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dv_new, src=dv_c)
                    nisa.tensor_tensor(dst=dv_new, data1=dv_new, data2=dv_old,
                                       op=nl.add)
                    nisa.dma_copy(dst=dv_buf[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :],
                                  src=dv_new)
                    # dP = dO @ V^T: contraction is over d, so V MUST
                    # be transposed to [d, k] rows (native [k, d] rows
                    # contract k against d: silent garbage poisons dS
                    # and through it dQ/dK, found 2026-09-17).
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
                    # dS = _NKI_ATTN_P * (dP - D), pre-scaled by s (serves dQ and dK)
                    ds = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=ds, data=dp_s, op0=nl.subtract,
                                       operand0=d_row)
                    pblk = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=pblk,
                                     src=p_full[0:_NKI_ATTN_P, c * _NKI_ATTN_P:(c + 1) * _NKI_ATTN_P])
                    nisa.tensor_tensor(dst=ds, data1=ds, data2=pblk,
                                       op=nl.multiply)
                    nisa.tensor_scalar(dst=ds, data=ds, op0=nl.multiply,
                                       operand0=scale)
                    ds16 = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ds16, src=ds)
                    # dQ contrib: dS^T @ K
                    kc = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=kc,
                                  src=k[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                    dst_t = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=dst_t, data=ds16)
                    dst_s = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dst_s, src=dst_t)
                    nisa.nc_matmul(dst=dq_acc, accumulate=(c > 0),
                                   stationary=dst_s, moving=kc)
                    # dK contrib: dS @ Q -> fp32 HBM scratch add
                    dk_c = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dk_c, stationary=ds16, moving=q_b)
                    dk_old = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32,
                                        buffer=nl.sbuf)
                    nisa.dma_copy(dst=dk_old,
                                  src=dk_buf[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                    dk_new = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32,
                                        buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dk_new, src=dk_c)
                    nisa.tensor_tensor(dst=dk_new, data1=dk_new, data2=dk_old,
                                       op=nl.add)
                    nisa.dma_copy(dst=dk_buf[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :],
                                  src=dk_new)
                dq_s = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dq_s, src=dq_acc)
                dq16 = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dq16, src=dq_s)
                nisa.dma_copy(dst=dq[base + r0:base + r0 + _NKI_ATTN_Q_TILE, :], src=dq16)
            # --- epilogue: downcast fp32 scratch bufs to bf16 outputs
            # (unrolled x2: the compiler rejects tuple-destructuring loops
            # over kernel tensors with "expecting simple variable").
            for c in range(T // _NKI_ATTN_P):
                fblk = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=fblk, src=dk_buf[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                hblk = nl.ndarray((_NKI_ATTN_P, _NKI_ATTN_P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=hblk, src=fblk)
                nisa.dma_copy(
                    dst=dk[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :], src=hblk)
                nisa.dma_copy(
                    dst=fblk, src=dv_buf[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :])
                nisa.tensor_copy(dst=hblk, src=fblk)
                nisa.dma_copy(
                    dst=dv[base + c * _NKI_ATTN_P:base + (c + 1) * _NKI_ATTN_P, :], src=hblk)
        return dq, dk, dv

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
            dq3, dk3, dv3 = _attn_nki_bwd(q3, k3, v3, g3, mask, scale,
                                          dq3, dk3, dv3, dk_buf, dv_buf, t, d)
            return (dq3.view(b, h, t, d), dk3.view(b, h, t, d),
                    dv3.view(b, h, t, d), None)


    # Native BTC-layout two-tap convolution. Same trained parameters; FP32-input
    # forward/dX/dW validated against FP64, then timed in the full dense model.
    from torch_neuronx import wrap_nki as _wrap_conv2_nki

    def _conv2_taps(wt, channels):
        prev = nl.ndarray((128, channels), dtype=wt.dtype, buffer=nl.sbuf)
        curr = nl.ndarray((128, channels), dtype=wt.dtype, buffer=nl.sbuf)
        pattern = [[0, 128], [1, channels]]
        nisa.dma_copy(dst=prev, src=wt.ap(pattern=pattern, offset=0))
        nisa.dma_copy(dst=curr, src=wt.ap(pattern=pattern, offset=channels))
        return prev, curr


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
    def _conv2_forward(x, wt, lanes: int = 1):
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
            nisa.nc_matmul(dst=summed, stationary=prod[:, c * 128:(c + 1) * 128],
                           moving=ones, accumulate=False)
            nisa.tensor_copy(dst=result, src=summed)
            nisa.dma_copy(dst=partials[batch, tile, tap, c * 128:(c + 1) * 128, :], src=result)


    def _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones,
                  batch, tile, length, channels, first, last):
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
            # A memset confined to partition 127 failed SBUF allocation in both write
            # orders on the installed compiler. Initialize the full tile, then fill
            # the prefix. Only its final row remains zero; no gradient crosses B/T.
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
    def _conv2_backward(x, grad, wt, lanes: int = 1):
        batch, length, channels = x.shape
        assert length % 128 == 0 and channels % 128 == 0
        assert grad.shape == x.shape and wt.shape == (2, channels)
        assert lanes in (1, 2) and batch % lanes == 0
        tiles = length // 128
        dx = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        partials = nl.ndarray((batch, tiles, 2, channels, 1), dtype=nl.float32,
                              buffer=nl.shared_hbm)
        prev_w, curr_w = _conv2_taps(wt, channels)
        ones = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        lane = nl.program_id(0)
        for b in nl.affine_range(batch // lanes):
            bi = b * lanes + lane
            if tiles == 1:
                _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones,
                          bi, 0, length, channels, True, True)
            else:
                _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones,
                          bi, 0, length, channels, True, False)
                if tiles > 2:
                    for t in nl.affine_range(tiles - 2):
                        _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones,
                                  bi, t + 1, length, channels, False, False)
                _conv2_bwd_tile(x, grad, dx, partials, prev_w, curr_w, ones,
                          bi, tiles - 1, length, channels, False, True)
        return dx, partials


    # Resolve the native higher-order operator before Dynamo enters autograd.forward.
    # Constructing a raw nki.Kernel grid inside the traced function hits the installed
    # NKI 0.5 / Torch 2.11 dataclass-hash path before that kernel has its func field.
    # wrap_nki is the installed backend's explicit, graph-capturable interface.
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
            return dx, dw.T.unsqueeze(1).to(ctx.weight_dtype)

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
#     and softcap+CE kernels).
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
# Toggled by the --nki-attn flag in main(). Default OFF: fused causal SDPA
# (research/scratch/nki_attn_fwd.py + nki_attn_bwd.py, hw-validated
# 2026-09-17 -- fp32 ~1e-6, bf16 fwd format floor, bwd dQ/dK ~4e-4) for q/k/v.
USE_NKI_ATTN = False
# Full-model native gate and timing: fs-w7-conv2-nki-native-gate-r5 and
# fs-w7-dense1024-noqk-nki-conv-phase-r1. Default off for controlled comparisons.
USE_NKI_LOCAL_CONV = False
# Opt-in restoration of traceable SDPA when the custom attention kernel is OFF.
# The historical nki_attention wrapper disables Dynamo even on its plain-SDPA fallback,
# which changes graph partitioning. Keep that path as the default for reproducible A/Bs;
# this flag tests the direct call without changing the custom kernel's guarded wrapper.
COMPILE_SDPA_DIRECT = False
# Pure-torch helper, intentionally outside the nki try-block so CPU imports
# (and this file's scratch equiv test) can reach it without a device.
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


def softcap_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                          loss_reduction: str) -> torch.Tensor:
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
    if loss_reduction == "mean" and LOSS_BYTE_WEIGHT and BYTE_LOSS_W is not None:
        # .to() on an already-placed tensor is a no-op returning self, so this is
        # free after the first step's host-to-device copy. No ignore_index exists on
        # this path (the calls below never pass one), so every target is a valid id.
        weight = BYTE_LOSS_W.to(device=logits.device, dtype=torch.float32)[targets.reshape(-1)]
        loss_reduction = "none"
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
                out = per_token.view_as(targets)
            else:
                out = per_token.mean()
            if weight is not None:
                return (out.reshape(-1) * weight).sum() / weight.sum()
            return out
    capped = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
    loss = F.cross_entropy(
        capped.reshape(-1, capped.size(-1)),
        targets.reshape(-1),
        reduction=loss_reduction,
    )
    if loss_reduction == "none":
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
        if USE_WEIGHT_CACHE and self._cached_weight is not None and self._cached_weight.dtype == x.dtype:
            w = self._cached_weight
        else:
            w = self.weight.to(dtype=x.dtype)
        if self.weight_transposed:
            return x @ w
        return F.linear(x, w, None)


_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_FNV64_MASK = 0xFFFFFFFFFFFFFFFF


def _fnv1a64(data: bytes) -> int:
    """FNV-1a 64. Deliberately NOT Python's builtin hash(): that is randomized per process for
    bytes/str (PYTHONHASHSEED), which would make the init non-reproducible across runs -- the one
    property an init function must never lack. Hand-rolled rather than hashlib so it stays in the
    same style as NGRAM_HASH_PRIMES and costs no new import."""
    h = _FNV64_OFFSET
    for b in data:
        h = ((h ^ b) * _FNV64_PRIME) & _FNV64_MASK
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
    # tiktoken exposes the specials as a name->id dict; take the IDS, which is the same set
    # prepare.py's _save_tokenizer records 0 bytes for. The `or {}` covers an encoding with none.
    special_ids = set((getattr(enc, "_special_tokens", None) or {}).values())
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
        s = b"\x02" + raw + b"\x03"
        row = feat[tid]
        n_feat = 0
        for n in range(1, max_n + 1):
            for j in range(len(s) - n + 1):
                h = _fnv1a64(s[j:j + n])
                col = h % n_embd
                row[col] += 1.0 if (h >> 40) & 1 else -1.0
                n_feat += 1
        touched[tid] = True
        written += 1
        total_features += n_feat
    # CENTERING, and it is not cosmetic. Some features occur in EVERY token -- the \x02/\x03 markers
    # most obviously, but also the high-frequency letter unigrams -- and each of those lands in the
    # same column with the same sign in every row, so the raw feature matrix has a large shared DC
    # direction. Measured on the real 8192-token vocabulary: mean cosine over 2000 random token
    # pairs was +0.224 before centering (i.e. ~22% of every row was the same vector) and ~0.000
    # after, while the contrast that matters -- pairs sharing a 3-byte prefix -- stays far above it.
    # A shared component in wte is close to useless anyway: it adds the same vector to every
    # position, which is a bias the model already has, and it eats row norm that should be carrying
    # the token's identity.
    rows = feat[touched]
    rows -= rows.mean(dim=0, keepdim=True)
    # Per-row rescale AFTER centering: the raw accumulator is a count whose magnitude tracks token
    # byte length, and a 3x row-scale spread would be a 3x spread in AdamW's effective relative LR.
    rms = rows.pow(2).mean(dim=-1, keepdim=True).sqrt()
    rows *= BYTE_WTE_INIT_STD / rms.clamp_min(1e-12)
    feat[touched] = rows
    # A row that centered to exactly zero (astronomically unlikely, not impossible) keeps its random
    # init rather than being written as a zero vector.
    degenerate = touched & (feat.abs().sum(dim=-1) == 0.0)
    if bool(degenerate.any()):
        n_deg = int(degenerate.sum())
        touched &= ~degenerate
        written -= n_deg
        skipped += n_deg
    mix = float(BYTE_WTE_INIT_MIX)
    keep = math.sqrt(max(0.0, 1.0 - mix * mix))
    with torch.no_grad():
        target = wte[:vocab_size]                  # False rows of `touched` keep their normal_()
        blended = mix * feat.to(target.dtype) + keep * target
        target.copy_(torch.where(touched.unsqueeze(-1), blended, target))
    return written, skipped, (total_features / written if written else 0.0)


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
        if isinstance(m, Linear) and not m.weight_transposed:
            with torch.no_grad():
                wt = m.weight.detach().t().contiguous()
            m.weight = nn.Parameter(wt, requires_grad=m.weight.requires_grad)
            m.weight_transposed = True
            m._cached_weight = None
            n += 1
    return n


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


@torch._dynamo.disable
def nki_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  scale: float | None) -> torch.Tensor:
    """Causal SDPA on post-rope [B, H, T, D] q/k/v. With --nki-attn on Neuron
    hardware (and only when the validated kernel geometry holds: bf16/fp32,
    head dim 128, T a multiple of 512, q/k/v same shape), runs the fused NKI
    fwd/bwd pair; otherwise the original eager SDPA line verbatim, so the
    knob is always safe to flip.
    Dynamo is disabled around this node on purpose: tracing through the
    functorch autograd.Function wrapper poisoned a neighboring (rope)
    kernel's lowering (take-2 crash). Opaque node, kernels still on device.
    """
    if (USE_NKI_ATTN and _HAS_NKI and q.device.type == "neuron" and q.ndim == 4
            and q.shape == k.shape == v.shape
            and q.dtype in (torch.bfloat16, torch.float32)
            and q.shape[-1] == 128 and q.shape[2] % 512 == 0):
        s = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
        return _AttnNKI.apply(q, k, v, s)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0, coalesced: str = "none"):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim * self.n_head == self.n_embd
        # Sub-1.0 plan P2: grouped-query attention. 0/None => dense (n_kv = n_head).
        self.n_kv_heads = config.n_kv_heads or config.n_head
        self.kv_repeat = 1
        if self.n_kv_heads != self.n_head:
            if USE_FUSED_QKV or coalesced != "none":
                raise ValueError("GQA (n_kv_heads != n_head) needs the standard separate q/k/v "
                                 "projections: incompatible with USE_FUSED_QKV (one fused 3-way "
                                 "matmul) and with parallel-block coalescing (one c_in matmul).")
            if self.n_head % self.n_kv_heads != 0:
                raise ValueError(f"n_kv_heads={self.n_kv_heads} must divide "
                                 f"n_head={self.n_head}")
            self.kv_repeat = self.n_head // self.n_kv_heads
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
            # GQA: K/V project to n_kv_heads * head_dim (== n_embd when dense, so the
            # default shapes -- and inits -- are exactly the historical ones).
            self.c_k = Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self.c_v = Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
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

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None, ve_bi: torch.Tensor | None = None, ve_tri: torch.Tensor | None = None, qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
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
            # Fused QKV needs one shared input; separate Q/K/V streams collapse
            # to x_q in this combination (untested/discouraged -- both default OFF).
            q, k, v = self.c_qkv(x_q).split(self.n_embd, dim=-1)
            q = q.view(b, t, self.n_head, self.head_dim)
            k = k.view(b, t, self.n_head, self.head_dim)
            v = v.view(b, t, self.n_head, self.head_dim)
        else:
            q = self.c_q(x_q).view(b, t, self.n_head, self.head_dim)
            k = self.c_k(x_k).view(b, t, self.n_kv_heads, self.head_dim)
            v = self.c_v(x_v).view(b, t, self.n_kv_heads, self.head_dim)
            if self.n_kv_heads != self.n_head:
                # GQA: expand KV heads to full n_head BEFORE rope_norm, so the fused
                # rope+QK-norm NKI kernel sees exactly the shapes it was built for and
                # every downstream consumer (VE views, SDPA, head gate) is untouched.
                # The matmul saving (smaller c_k/c_v) is preserved; only rope and QK^T
                # stay at full width, as in the dense baseline. .contiguous() because
                # repeat_interleave yields a strided view and the NKI kernel needs
                # contiguous inputs (same requirement the "vfc" coalesce mode serves).
                k = k.repeat_interleave(self.kv_repeat, dim=2).contiguous()
                v = v.repeat_interleave(self.kv_repeat, dim=2).contiguous()
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
        q = rope_norm(q, cos, sin)
        k = rope_norm(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if attn_mask is None:
            if COMPILE_SDPA_DIRECT and not USE_NKI_ATTN:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
            else:
                y = nki_attention(q, k, v, self.attn_scale)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[..., :t, :t], scale=self.attn_scale)
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
        return self.c_proj(y)


def coalesced_swiglu_hidden(n_embd: int, mlp_ratio: int) -> int:
    """Parameter-matched SwiGLU hidden width: the h that makes 3*d*h == 2*d*(mlp_ratio*d), i.e.
    h = (2/3)*mlp_ratio*d (= 8/3*d at the locked mlp_ratio 4), rounded to a multiple of 128.

    Reads mlp_ratio from the CONFIG, not the global: load_for_eval rebuilds the model from the
    saved model_config, so a checkpoint written with --mlp-ratio must reproduce its own width."""
    if COALESCED_SWIGLU_HIDDEN > 0:
        return int(COALESCED_SWIGLU_HIDDEN)
    exact = (2.0 / 3.0) * mlp_ratio * n_embd
    return max(128, int(round(exact / 128.0)) * 128)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, coalesced: str = "none"):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        # Parameter-matched coalesced SwiGLU: one Linear(n_embd, 2*hidden) whose two halves are the
        # gate and the up projection, at the narrower hidden width that keeps the three matrices'
        # total size equal to the baseline's two. `swiglu_hidden` doubles as the forward-path flag
        # so a module built without the toggle can never take the gated branch.
        self.swiglu_hidden = coalesced_swiglu_hidden(config.n_embd, config.mlp_ratio) \
            if USE_COALESCED_SWIGLU else 0
        if self.swiglu_hidden:
            hidden = self.swiglu_hidden
        # `coalesced` != "none" (parallel block only): c_fc lives in the owning block's single c_in
        # matmul, so it is not built here and forward takes the pre-activation as an argument.
        fc_out = 2 * hidden if self.swiglu_hidden else hidden
        self.c_fc = None if coalesced in ("all", "vfc") else Linear(config.n_embd, fc_out, bias=False)
        self.c_gate = Linear(config.n_embd, hidden, bias=False) if USE_GATED_MLP else None
        self.c_proj = Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, fc: torch.Tensor | None = None) -> torch.Tensor:
        h = self.c_fc(x) if fc is None else fc
        if self.swiglu_hidden:
            # silu(gate) * up from the two halves of the single projection. The last-dim slices are
            # non-contiguous, which is fine HERE (unlike the parallel block's q/k slices, which had
            # to be copied because rope_norm silently falls back to eager on a strided input): both
            # operands only ever reach elementwise ops, and the product they produce is a fresh
            # contiguous tensor before it reaches c_proj's matmul.
            gate = h[..., :self.swiglu_hidden]
            up = h[..., self.swiglu_hidden:]
            return self.c_proj(F.silu(gate) * up)
        if self.c_gate is not None:
            return self.c_proj(h * F.silu(self.c_gate(x)))
        return self.c_proj(relu_squared(h))



class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.coalesced = PARALLEL_BLOCK_COALESCE if USE_PARALLEL_BLOCK else "none"
        self.attn = CausalSelfAttention(config, layer_idx, coalesced=self.coalesced)
        self.mlp = MLP(config, coalesced=self.coalesced)
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
        self.resid_lambda = nn.Parameter(torch.ones(1))
        # x0 skip: mixes initial embedding into each layer's input (stabilizes deep nets)
        self.x0_lambda = nn.Parameter(torch.zeros(1))
        # Recursive's input-dependent x0 gate: one scalar per layer, zero-init so
        # 2*sigmoid(0) == 1.0 makes it a bit-exact no-op against the static x0_lambda skip.
        self.x0_gate_s = nn.Parameter(torch.zeros(1)) if USE_X0_GATE else None
        # Frozen zero nudge: never trained, never in an optimizer group. See USE_BLOCK_NUDGE.
        self.block_nudge = nn.Parameter(torch.zeros(1), requires_grad=False) if USE_BLOCK_NUDGE else None
        # T-parallel: the two branch weights of the parallel attn+MLP block. The branch in
        # forward is guarded on par_lambda_a being present.
        # nn.Parameter(ones) draws no RNG, so building these cannot shift any other tensor's init.
        if USE_PARALLEL_BLOCK:
            self.par_lambda_a = nn.Parameter(torch.ones(1))
            self.par_lambda_m = nn.Parameter(torch.ones(1))
        else:
            self.par_lambda_a = self.par_lambda_m = None
        self.local_conv = CausalConv(config.n_embd, LOCAL_CONV_KERNEL) if USE_LOCAL_CONV else None
        self.scalar_gate = Linear(SELECTIVE_GATE_CHANNELS, 2, bias=False) if USE_SELECTIVE_GATE else None
        # T3.9: causal Q/K token shift (Recursive's from-vanilla recipe). Per-DIM learned beta, so
        # each residual channel decides for itself how much of the previous token to mix into the
        # query/key view; V is deliberately left unshifted.
        self.qk_shift_beta = nn.Parameter(torch.full((config.n_embd,), float(QK_SHIFT_BETA)),
                                          requires_grad=not QK_SHIFT_FREEZE) \
            if USE_QK_SHIFT else None

    def forward(self, x: torch.Tensor, x0: torch.Tensor, ve: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None, ve_bi: torch.Tensor | None = None, ve_tri: torch.Tensor | None = None) -> torch.Tensor:
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
            attn_out = self.attn(h, h, h, ve, cos, sin, attn_mask, ve_bi, ve_tri, qkv=qkv)
            x = x + self.par_lambda_a * attn_out + self.par_lambda_m * self.mlp(h, fc=fc)
            return x
        if USE_QKV_NORM_CSE and q_in is k_in and k_in is v_in:
            # See USE_QKV_NORM_CSE. Guarded on object identity, not equality, so this branch can
            # only ever be taken when the three norm() calls it replaces had identical inputs.
            qkv_n = norm(q_in)
            attn_out = self.attn(qkv_n, qkv_n, qkv_n, ve, cos, sin, attn_mask, ve_bi, ve_tri)
            x = x + attn_out
        else:
            attn_out = self.attn(norm(q_in), norm(k_in), norm(v_in), ve, cos, sin, attn_mask, ve_bi, ve_tri)
            x = x + attn_out
        if USE_MLP_SANDWICH_NORM:
            # T3.9: sandwich norm on the MLP branch only (Recursive). norm() is parameter-free
            # F.rms_norm with the default eps, so the zero-init c_proj still gives exactly 0 here
            # at step 0 (0 / sqrt(0 + eps) = 0) -- the step-0 no-op survives, and the arm only
            # starts to differ once c_proj leaves the origin.
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
        # so causality is structural here, exactly as for the x0 gate. There is
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
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, config.n_embd)
            for i in range(config.n_layer)
            if USE_VE and has_ve(i, config.n_layer)
        })
        # T3.1: hashed n-gram VALUE embeddings. Keys are "b<layer>" / "t<layer>" so a
        # bigram-only and a bigram+trigram run have visibly different state_dict keys
        # (strict=True then catches a toggle mismatch loudly -- USE_NGRAM_VE and
        # NGRAM_VE_TRIGRAM are both in SHAPE_CRITICAL_TOGGLES for that reason).
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
            # USE_COALESCED_SWIGLU: this one call covers BOTH halves of the [2*hidden, n_embd]
            # projection at the same +-0.4s bound, which is what USE_GATED_MLP's separate c_gate
            # line does too -- canonical SwiGLU gives gate and up the same init scale.
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
        # T1.6 MUST be the last thing init_weights does. Every initializer above addresses
        # weights in nn.Linear's native [out, in] layout -- `c_in.weight[:n]` row slices, the
        # per-slice uniform_ bounds that keep the coalesced arm bit-identical to the unfused
        # one, and `uniform_` itself, which fills in MEMORY order and so does not commute with
        # a transpose. Initializing first and transposing after is what makes
        # USE_TRANSPOSED_LINEAR bit-exact rather than "the same distribution".
        if USE_TRANSPOSED_LINEAR:
            n_t = transpose_linear_weights(self)
            # Announce the count: this arm's whole effect is invisible in the loss (it is
            # bit-exact), so a flag that silently converted nothing would look exactly like a
            # flag that worked. Same reason --compile-muon prints the branch it took.
            print0(f"T1.6: transposed {n_t} Linear weights to [in, out] layout")

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
        attn_flops = self.config.n_layer * 12 * h * d * t
        return 6 * (nparams - non_matmul) + attn_flops

    def setup_optimizer(self):
        scalar_params = []
        matrix_params = []
        local_conv_params = []  # T0.1: local_conv.weight, previously dead compute -- see below
        gate_params = []  # T3.9/T4.2: small 1-D or single-row gate tensors (attention head-gate
        # biases, the output-bigram gate's weight+bias, per-dim Q/K shift betas). None of them is
        # Muon material -- a 1-D vector has no second axis to orthogonalize and a [1, n_embd] row
        # would come out of Newton-Schulz as a unit-norm direction, i.e. a wildly oversized update
        # -- and none of them belongs in the SCALAR_LR=0.20 group either. THIS IS A MEASURED
        # CONSTRAINT, not a style preference: SCALAR_LR is tuned for a single learned scalar's
        # limited blast radius, and a real hardware A/B that put a genuine projection matrix in
        # the SCALAR_LR group hit a real NaN divergence around step ~270-280 (loss climbing
        # 4.7->6+ then NaN, while the same optimizer/schedule trains the baseline stably past
        # 900+ steps) -- a plausible cause given AdamW's adaptive per-parameter scaling can
        # amplify an 8x-too-aggressive LR (0.20 vs MATRIX_LR=0.025) on a real weight tensor over
        # a few hundred steps. So: AdamW at MATRIX_LR, weight_decay 0, exactly like
        # local_conv_params.
        for block in self.transformer.h:
            if block.local_conv is not None:
                # T0.1: register the depthwise causal conv1d weight so it actually
                # trains. Not Muon material (depthwise: each output channel's kernel
                # is an independent [1, k] row with no second axis to orthogonalize,
                # and k is tiny), and
                # not the SCALAR_LR=0.20 group either (that LR is tuned for a single
                # learned scalar's blast radius, not a real weight tensor -- see the
                # gate_params comment above). Own AdamW group at MATRIX_LR, wd=0.0,
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
            if block.qk_shift_beta is not None and block.qk_shift_beta.requires_grad:
                # QK_SHIFT_FREEZE leaves beta out of EVERY optimizer group on purpose (see the
                # QK_SHIFT_FREEZE comment): the throughput probe needs the ops in the graph but the
                # parameter mathematically inert, and an AdamW group would move it off its init.
                gate_params.append(block.qk_shift_beta)
            if block.attn.head_gate is not None:
                # The [n_head, HEAD_GATE_CHANNELS] weight goes to Muon with every other per-head
                # gate matrix (ve_gate/ve_gate_bi/ve_gate_tri have exactly this shape); only the
                # 1-D bias has to be pulled out by name.
                gate_params.append(block.attn.head_gate_bias)
            matrix_params.extend(p for name, p in block.attn.named_parameters()
                                 if name != "head_gate_bias")
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
        # same argument as block.local_conv) and gets its own AdamW group below.
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
        # T2.4 guard, and the answer to "is the old NaN still a live risk?". The scalar group runs at
        # SCALAR_LR = 8-10x MATRIX_LR, which is safe for a single learned scalar and NOT safe for a
        # real weight tensor -- a projection matrix that landed here once diverged to NaN around step
        # ~270. Every gate-shaped tensor added since (ve_gate/ve_gate_bi/ve_gate_tri, head_gate,
        # out_bigram_gate, qk_shift_beta) is routed to Muon or to gate_params instead, so the
        # invariant "this group holds only scalars and 1-D lambda vectors" is currently true; assert
        # it so a future addition cannot quietly break it and blame the LR.
        for p in scalar_params:
            if p.ndim > 1:
                raise ValueError(f"scalar_params must hold only scalars/1-D vectors (SCALAR_LR="
                                 f"{SCALAR_LR} is tuned for those); got a {tuple(p.shape)} tensor. "
                                 "Route it to gate_params or to Muon instead.")
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        value_embed_params = list(self.value_embeds.parameters())
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=(0.8, EMBED_ADAMW_BETA2), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            # T2.4: SCALAR_BETA1/2 default to the historical (0.9, 0.95).
            dict(kind="adamw", params=scalar_params, lr=SCALAR_LR,
                 betas=(SCALAR_BETA1, SCALAR_BETA2), eps=1e-10, weight_decay=0.0),
        ]
        if gate_params:
            param_groups.append(dict(kind="adamw", params=gate_params, lr=MATRIX_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0))
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
            if rmsprop:
                table_dtypes = sorted({str(p.dtype) for p in ngram_params})
                math_dtypes = ["torch.float32"] if NGRAM_RMS_FP32 else table_dtypes
                print0(f"ngram_rmsprop: fp32_state_math={NGRAM_RMS_FP32} nki={USE_NKI_NGRAM_RMS} "
                       f"parameter_gradient_dtypes={table_dtypes} state_math_dtypes={math_dtypes}")
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
        # T1.6: a transposed weight is a DIFFERENT Muon group from a same-shaped untransposed one,
        # because `transposed` changes the logical orientation the group's math is derived from
        # (see _muon_step). Keyed on identity rather than on the global toggle so a partially
        # converted model can never be silently mis-grouped. With USE_TRANSPOSED_LINEAR off the
        # set is empty and the grouping is byte-for-byte the previous `sorted({p.shape})`.
        transposed_ids = {id(m.weight) for m in self.modules()
                          if isinstance(m, Linear) and m.weight_transposed}
        # Plan §11.1 item 3: parameter -> transformer-block index, so a Muon group that stacks one
        # shape across layers can still carry a PER-LAYER lr/momentum multiplier. Built by walking
        # the block list rather than parsing names, so it cannot drift from the grouping above.
        # Parameters outside the block stack (ngram_proj, Engram's projections) are simply absent
        # and get a neutral multiplier -- see muon_depth_mults.
        depth_of: dict[int, int] = {}
        for _i, _block in enumerate(self.transformer.h):
            for _p in _block.parameters():
                depth_of[id(_p)] = _i
        depth_on = any(muon_depth_enabled())
        for shape, transposed in sorted({(p.shape, id(p) in transposed_ids) for p in matrix_params},
                                        key=lambda st: (st[0], st[1])):
            params = [p for p in matrix_params
                      if p.shape == shape and (id(p) in transposed_ids) == transposed]
            group = dict(kind="muon", params=params, lr=MATRIX_LR, momentum=0.95, beta2=MUON_BETA2,
                         ns_steps=MUON_NS_STEPS, weight_decay=WEIGHT_DECAY,
                         transposed=transposed)
            if depth_on:
                # Only set when the feature is on, so with it off the group dicts -- and therefore
                # everything keyed on them in MuonAdamW -- are byte-for-byte what they were.
                group["depth_mults"] = muon_depth_mults(
                    [depth_of.get(id(p)) for p in params], self.config.n_layer)
            param_groups.append(group)
        opt = MuonAdamW(param_groups)
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
            # Plan §11.1 item 1: Demon anneals beta1 relative to the group's ORIGINAL value, so
            # that value has to survive the per-step rewrite of group["betas"]. Muon groups have
            # no "betas" key at all (they carry `momentum` and `beta2` separately), hence .get.
            if group.get("betas") is not None:
                group["initial_betas"] = tuple(group["betas"])
        return opt

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_reduction: str = "mean"):
        _b, t = idx.shape
        assert t <= self.config.sequence_len
        cos, sin = self.cos[:, :t], self.sin[:, :t]
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
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
            if USE_CAUSAL_FASTPATH and doc_mask is None and not self.layer_is_short[i]:
                # Solution 13d: this layer's static mask is purely causal (full context,
                # no doc-boundary masking) -- let SDPA take its is_causal fast path
                # instead of materializing the additive mask tensor.
                attn_mask = None
            else:
                static_mask = (self.mask_short if self.layer_is_short[i] else self.mask_full)[..., :t, :t]
                attn_mask = static_mask + doc_mask if doc_mask is not None else static_mask
            x = block(x, x0, ve, cos, sin, attn_mask, ve_bi, ve_tri)
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
    if N_KV_HEADS < 0 or N_KV_HEADS > n_head:
        raise ValueError(f"N_KV_HEADS={N_KV_HEADS} must be 0 (dense) or in [1, n_head={n_head}]")
    if N_KV_HEADS > 0 and n_head % N_KV_HEADS != 0:
        raise ValueError(f"N_KV_HEADS={N_KV_HEADS} must divide n_head={n_head} "
                         f"(locked width: n_head=5, so only 1/MQA or 5/dense)")
    return GPTConfig(
        sequence_len=SEQ_LEN,
        vocab_size=TOKENIZER_VOCAB_SIZE,
        n_layer=depth,
        n_head=n_head,
        n_embd=n_embd,
        mlp_ratio=MLP_RATIO,
        attn_scale=ATTN_SCALE,
        n_kv_heads=N_KV_HEADS,
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
# NEFF per group instead of ~80 aten launches. See the _muon_math docstring for why this
# was blocked and what unblocked it.
# ON since 2026-09-14: +3.38% end-to-end at the locked config (156063 -> 161342 tok/s, 120-step
# instrument, 4 ranks, LNC=2), and it stacks slightly SUPER-additively with the frozen qk-shift
# (+13.67% together vs a 13.19% multiplicative prediction). Not the ~6% the isolated 8.18x
# microbenchmark predicted, because that was priced against a non-NKI optimizer and the shipped
# path already had the NKI Muon fusion. It REPLACES that fusion: a compiled region cannot contain
# an nki.jit call, so --compile-muon and --nki-muon are mutually exclusive by construction.
# Numerics: the source rewrite is BIT-EXACT against the eager body (0 differing elements, CPU,
# 5 steps, per-step-varying schedule). The COMPILED execution is not -- worst max|diff| 3.25e-03
# accumulated over 3 steps, bf16-ULP-scale drift in the bf16 Newton-Schulz chain, the same order as
# the rope+norm (1 ULP) and softcap+CE (2.44 ULP) kernels already shipped. Checked where it
# matters: the step-matched 1800s control with this ON scored 1.019126/2141, which normalises to
# 1.018982 against the locked config's 1.018985 -- per-step neutral to 3e-6.
USE_COMPILE_MUON = True

# Candidate free win flagged (not adopted) in the _muon_lr_g docstring: the locked default
# path multiplies a 0-dim fp32 `lr` by a bf16 Newton-Schulz iterate `g`, and PyTorch's type
# promotion rule for a 0-dim operand against a dimensioned one casts `lr` DOWN to bf16 before
# the multiply -- quantising every Muon group's effective lr by up to ~0.4% relative error,
# uniformly per group per step, every step. ON here keeps that product in fp32 instead (see
# _muon_lr_g). Its own flag and its own A/B, per that docstring -- never smuggled in under the
# per-depth-lr work that surfaced it, and never touching that work's intentional bf16-matching
# behaviour for `lr.dim() > 0`. Default OFF = current shipped behaviour, unchanged.
USE_MUON_LR_FP32 = False

# One compiled callable per (tall, ns_steps, red_dim) specialisation, built lazily.
# Dynamo re-guards on the tensor shapes by itself, so a single entry can still hold
# several NEFFs -- this cache only exists so the wrapper object is not rebuilt per step.
_MUON_COMPILED: dict[tuple, object] = {}


def _muon_math(w, g, momentum_buffer, second, mom, one_mom, lr, lrwd,
               tall: bool, ns_steps: int, red_dim: int, red_dim_size: int,
               one_beta2, to_bf16: bool, lr_fp32: bool = False):
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
    w = w - (_muon_lr_g(lr, g, lr_fp32) + lrwd * w * mask)
    return w, momentum_buffer, second


def _muon_lr_g(lr, g, lr_fp32: bool = False):
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
    if isinstance(lr, torch.Tensor) and lr.dim() > 0 and lr.dtype != g.dtype:
        lr = lr.to(g.dtype)
        return lr * g
    if lr_fp32 and isinstance(lr, torch.Tensor) and lr.dim() == 0 and lr.dtype != g.dtype:
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
        p.mul_(1 - lr * wd * ((u * p) > 0).to(p.dtype))


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
        mults = group.get("depth_mults")
        if mults is None:
            return None
        cache = self._scalar_cache.setdefault(id(group), {})
        entry = cache.get("depth_mults")
        if entry is None:
            lr_mult, mom_mult = mults
            host_lr = torch.tensor(lr_mult, dtype=torch.float32).view(-1, 1, 1)
            host_mom = torch.tensor(mom_mult, dtype=torch.float64).view(-1, 1, 1)
            entry = (host_lr, host_mom, host_lr.to(ref.device))
            cache["depth_mults"] = entry
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
        entry = cache.get("depth_mom_step")
        if entry is None:
            k = host_mom.shape[0]
            entry = (torch.empty(2, k, 1, 1, dtype=torch.float32),
                     torch.zeros(2, k, 1, 1, dtype=torch.float32, device=ref.device))
            cache["depth_mom_step"] = entry
        host, dev = entry
        self._fill_depth_momentum(host_mom, float(group["momentum"]), host[0], host[1])
        dev.copy_(host)
        return dev[0], dev[1]

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
            entry = cache.get("muon_compile_depth")
            if entry is None:
                k = host_lr.shape[0]
                entry = (torch.empty(5, k, 1, 1, dtype=torch.float32),
                         torch.zeros(5, k, 1, 1, dtype=torch.float32, device=ref.device),
                         torch.empty((), dtype=torch.float32))
                cache["muon_compile_depth"] = entry
            host, dev, tmp = entry
            tmp.fill_(float(group["lr"]))
            tmp.mul_(ratio)                  # fp32(fp32(lr) * fp32(ratio)) == device
            torch.mul(host_lr, float(tmp), out=host[0])
            torch.mul(host[0], float(group["weight_decay"]), out=host[1])
            # (1 - momentum) * per-depth multiplier in float64, then back to the momentum the
            # buffer keeps -- the two operands stay exactly complementary, no layer's momentum can
            # drift outside (0, 1), and a multiplier of 1.0 reproduces the 0-dim path bit for bit.
            self._fill_depth_momentum(host_mom, float(group["momentum"]), host[2], host[3])
            host[4].fill_(1.0 - float(group["beta2"]))
            dev.copy_(host)
            return dev[0], dev[1], dev[2], dev[3], dev[4]
        entry = cache.get("muon_compile")
        if entry is None:
            entry = (torch.empty(5, dtype=torch.float32),
                     torch.zeros(5, dtype=torch.float32, device=ref.device),
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
        host[4].fill_(1.0 - float(group["beta2"]))
        dev.copy_(host)
        return dev[0], dev[1], dev[2], dev[3], dev[4]

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
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), 1 - beta2)
        denom = torch._foreach_div(exp_avg_sqs, bias2)
        torch._foreach_sqrt_(denom)
        torch._foreach_add_(denom, group["eps"])
        update = torch._foreach_div(exp_avgs, bias1)
        torch._foreach_div_(update, denom)
        # T2.3: the decay multiply sits HERE rather than before the two lerps, and that move is
        # exactly (not approximately) equivalence-preserving: `update` reads only exp_avg,
        # exp_avg_sq and grad -- never p -- and p is still at its pre-decay value when the mask and
        # the multiply read it, so with USE_CAUTIOUS_WD off this computes the same two operations in
        # the same order on the same operands as before. It has to be here because the cautious mask
        # needs `update`. (Bitwise-checked in research/scratch/cautious_wd_equiv.py.)
        if wd != 0:
            if USE_CAUTIOUS_WD:
                _cautious_weight_decay_(params, update, lr, wd)
            else:
                torch._foreach_mul_(params, 1 - lr * wd)
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
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            update = exp_avg / bias1 / ((exp_avg_sq / bias2).sqrt() + group["eps"])
            # T2.3: moved down from above the lerps for the same reason, and with the same
            # equivalence argument, as in _adamw_step_foreach. The else branch keeps the original
            # single-tensor `p.mul_(1 - lr*wd)` verbatim -- including the wd==0 no-op the foreach
            # docstring documents -- so this path stays the literal reference implementation the
            # foreach one is bitwise-checked against.
            # The wd == 0 groups keep the literal `p.mul_(1 - lr*0)`, which is an exact no-op on a
            # finite IEEE754 value (see the foreach docstring), rather than paying for a mask that
            # would be multiplied by zero anyway.
            if USE_CAUTIOUS_WD and group["weight_decay"] != 0:
                _cautious_weight_decay_([p], [update], lr, group["weight_decay"])
            else:
                p.mul_(1 - lr * group["weight_decay"])
            if USE_TENSOR_LR_SCALARS:
                # alpha= is the one place a Python float becomes a baked graph
                # constant; as a tensor multiply it stays a runtime operand.
                p.sub_(update * lr)
            else:
                p.add_(update, alpha=-lr)

    def _rmsprop_state(self, group, *, cast_grads=True):
        """Prologue for the RMSProp path -- the same contract as _adamw_state, minus the
        first moment (that is the whole point of this kind) and minus its bias correction."""
        beta2 = group["betas"][1]
        params, grads, exp_avg_sqs, bias2 = [], [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            state_dtype = torch.float32 if NGRAM_RMS_FP32 else p.dtype
            if not state:
                state["step"] = 0
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=state_dtype)
            elif state["exp_avg_sq"].dtype != state_dtype:
                # Optimizer.load_state_dict may cast state to the parameter dtype.
                # Reject lost precision or a mid-run mode change before any update;
                # scored training starts fresh, so no lossy migration is required.
                raise ValueError(
                    f"RMSProp second-moment dtype {state['exp_avg_sq'].dtype} does not match "
                    f"NGRAM_RMS_FP32={NGRAM_RMS_FP32} (expected {state_dtype}). "
                    "Start with fresh optimizer state or restore its original precision; "
                    "load_state_dict may have cast FP32 state to the parameter dtype."
                )
            state["step"] += 1
            params.append(p)
            grads.append(p.grad.float() if NGRAM_RMS_FP32 and cast_grads else p.grad)
            exp_avg_sqs.append(state["exp_avg_sq"])
            bias2.append(1 - beta2 ** state["step"])
        return params, grads, exp_avg_sqs, bias2

    @torch.no_grad()
    def _rmsprop_step_nki(self, group):
        if not NGRAM_RMS_FP32 or not _HAS_NKI:
            raise ValueError("NKI table RMSProp requires FP32 moments and an available NKI backend")
        lanes = 2 if os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1") == "2" else 1
        for p in group["params"]:
            if p.grad is None:
                continue
            if (p.device.type != "neuron" or p.dtype != torch.bfloat16
                    or not p.is_contiguous() or not p.grad.is_contiguous()
                    or p.numel() % (2048 * 128 * lanes) != 0):
                raise ValueError("NKI table RMSProp requires contiguous Neuron BF16 tables/grads "
                                 "with element count divisible by 2048*128*LNC")
        # The kernel promotes in SBUF. Casting entire gradients to FP32 here
        # would add full-table HBM traffic and undo the benefit of fusion.
        params, grads, moments, biases = self._rmsprop_state(group, cast_grads=False)
        for p, grad, moment, bias in zip(params, grads, moments, biases):
            state = self.state[p]
            if "nki_rms_scalars_cpu" not in state:
                state["nki_rms_scalars_cpu"] = torch.empty((128, 2), dtype=torch.float32)
                state["nki_rms_scalars_device"] = torch.empty((128, 2), dtype=torch.float32,
                                                             device=p.device)
            scalars_cpu = state["nki_rms_scalars_cpu"]
            scalars_cpu[:, 0].fill_(group["lr"])
            scalars_cpu[:, 1].fill_(1.0 / bias)
            scalars = state["nki_rms_scalars_device"]
            scalars.copy_(scalars_cpu)
            p_view, grad_view, v_view = (x.view(-1, 2048) for x in (p, grad, moment))
            next_p, next_v = _ngram_rmsprop_fp32_nki[lanes](
                p_view, grad_view, v_view, scalars, group["betas"][1], group["eps"], lanes)
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
        assert group["weight_decay"] == 0.0, "rmsprop groups are weight-decay free by design"
        if USE_NKI_NGRAM_RMS:
            self._rmsprop_step_nki(group)
            return
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
        # T1.6: Muon's math is NOT transpose-invariant, so a transposed parameter has to be
        # described to it in its LOGICAL [out, in] orientation. Two things depend on it:
        #
        #  * `ratio = max(1, rows/cols)**0.5` is a one-sided LR scale. For the [3200, 640] MLP
        #    c_fc it is sqrt(5) = 2.236; read off the transposed [640, 3200] tensor it collapses
        #    to 1.0. Getting this wrong is a silent 2.24x LR change on every non-square matrix --
        #    i.e. USE_TRANSPOSED_LINEAR would NOT have been the "zero quality effect" storage
        #    reformulation it is advertised as. This is the whole reason `transposed` exists.
        #  * `red_dim` is the RMS-normalization axis. Here the fix is a flip rather than a
        #    correction: reducing the transposed tensor over the OPPOSITE axis is exactly the
        #    transpose of reducing the original, and everything downstream of `v_mean` is
        #    elementwise with broadcasting, so the whole tail comes out transposed and correct.
        #
        # The `tall` flag driving the Newton-Schulz branch is deliberately left on the ACTUAL
        # shape: the two branches contract over the smaller side either way, so they cost the
        # same and each is the transpose of the other (`x.mT @ x` is bit-exactly symmetric, so
        # the symmetric polynomial commutes with the transpose).
        transposed = group.get("transposed", False)
        lshape = (shape[-1], shape[-2]) if transposed else shape
        red_dim = -1 if lshape[-2] >= lshape[-1] else -2
        if transposed:
            red_dim = -2 if red_dim == -1 else -1
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if "second_momentum_buffer" not in state:
            # Derived from red_dim rather than re-deriving the shape comparison, so the buffer
            # can never disagree with the axis the reduction actually keeps.
            state_shape = (len(params), shape[-2], 1) if red_dim == -1 else (len(params), 1, shape[-1])
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
        # it, which measured a 2.14x regression when a hand-written scan kernel was
        # placed inside a compiled region), so
        # --compile-muon deliberately gives up the NKI fusion in exchange for whole-body
        # fusion. Gated on USE_TENSOR_LR_SCALARS for the same reason use_nki is: the
        # compiled body always carries lr as an fp32 device operand.
        # On a non-neuron device the body runs UNCOMPILED, which is not a fallback but
        # the thing the equivalence gate needs: it isolates the source rewrite (foreach-
        # free / functional / device-scalar) from the compiler, and that half is required
        # to be bit-exact. research/scratch/compile_optim_equiv.py --device cpu.
        if (USE_COMPILE_MUON and USE_TENSOR_LR_SCALARS and w.dtype == torch.float32):
            ratio = max(1.0, lshape[-2] / lshape[-1]) ** 0.5  # T1.6: logical orientation
            ns_steps = min(group["ns_steps"], len(POLAR_EXPRESS_COEFFS))
            key = (g.shape[-2] > g.shape[-1], ns_steps, red_dim, tuple(g.shape))
            fn = _MUON_COMPILED.get(key)
            if fn is None:
                fn = (torch.compile(_muon_math, backend="neuron", dynamic=False)
                      if p0.device.type == "neuron" else _muon_math)
                _MUON_COMPILED[key] = fn
                # Announce the branch actually taken: a silent fallback to the eager/NKI
                # body would make an A/B measure nothing.
                print0(f"muon step: COMPILED body for {tuple(g.shape)} "
                       f"(tall={key[0]} ns_steps={ns_steps} red_dim={red_dim}) "
                       f"[specialisations so far: {len(_MUON_COMPILED)}]")
            lr_t, lrwd_t, mom_t, one_mom_t, one_b2_t = \
                self._muon_compile_scalars(group, p0, ratio)
            # Plan §11.1 item 2: with the beta2 ramp OFF this stays the Python float it has always
            # been (so the traced graph and its single specialisation are unchanged); with it ON the
            # value changes every step, and under dynamic=False a Python float is a Dynamo guard --
            # i.e. a full neuronx-cc recompile of the whole Muon body per step. The device tensor
            # is the same operand `mom`/`one_mom` already are.
            one_beta2 = one_b2_t if MUON_BETA2_FINAL > 0.0 else 1 - group["beta2"]
            w, momentum_buffer, second = fn(
                w, g, momentum_buffer, second, mom_t, one_mom_t, lr_t, lrwd_t,
                key[0], ns_steps, red_dim, g.size(red_dim),
                one_beta2, COMPUTE_DTYPE == torch.bfloat16, USE_MUON_LR_FP32)
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
        # Plan §11.1 item 3: the NKI update kernel flattens the stacked buffer to [K*rows, cols]
        # and takes lr as a [128, 1] broadcast operand, so a PER-PARAMETER lr cannot be expressed
        # through it. Excluded rather than silently ignored -- a silent fall-through would make the
        # per-depth arm measure the baseline. (In the locked config the compiled path above already
        # takes precedence over NKI, so this only matters with --no-compile-muon.)
        depth = self._muon_depth_dev(group, p0)
        use_nki = (USE_NKI_MUON and _HAS_NKI and p0.device.type == "neuron"
                   and USE_TENSOR_LR_SCALARS and COMPUTE_DTYPE == torch.bfloat16
                   and w.dtype == torch.float32 and shape[-1] <= 8192
                   and min(shape[-2], shape[-1]) <= 8192
                   and depth is None)

        if depth is None:
            momentum_buffer.lerp_(g, 1 - group["momentum"])
            g = g.lerp_(momentum_buffer, group["momentum"])
        else:
            # [K, 1, 1] weights broadcast over the stacked buffers; `lerp_.Tensor` instead of
            # `lerp_.Scalar`, verified equal for a uniform weight in compile_optim_equiv.py.
            mom_d, one_mom_d = self._muon_depth_mom(group, p0)
            momentum_buffer.lerp_(g, one_mom_d)
            g = g.lerp_(momentum_buffer, mom_d)
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

        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = g.size(red_dim)
        v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
        second.lerp_(v_mean, 1 - group["beta2"])
        step_size = second.clamp_min(1e-10).rsqrt()
        scaled = (v_mean * red_dim_size) * step_size.square()
        v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
        g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)

        ratio = max(1.0, lshape[-2] / lshape[-1]) ** 0.5  # T1.6: logical orientation
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
            if depth is not None:
                # Plan §11.1 item 3. [K, 1, 1] broadcast over the stacked [K, rows, cols] update,
                # so the wd term below scales with lr exactly as it already does.
                lr = lr * depth[2]
            wd = group["weight_decay"]
            mask = (g * w) >= 0
            # _muon_lr_g, not `lr * g`: see its docstring. With per-depth scaling off `lr` is 0-dim
            # and the helper returns the identical tensor when USE_MUON_LR_FP32 is off (the
            # locked default), so this line is unchanged unless that flag is on.
            w.sub_(_muon_lr_g(lr, g, USE_MUON_LR_FP32) + lr * wd * w * mask)
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


def clip_and_sync_gradients(
    model: nn.Module,
    clip_main_params: list[torch.nn.Parameter],
    clip_ngram_params: list[torch.nn.Parameter],
    *,
    max_norm: float,
    clip_after_reduce: bool = False,
    grad_flat_buffer: torch.Tensor | None = None,
    foreach: bool | None = None,
    skip_param_ids: frozenset[int] | None = None,
) -> None:
    """Apply the selected clipping order, preserving the separate n-gram norm.

    Call after every micro-batch has accumulated and loss scaling has been undone.
    With clip_after_reduce=True every included gradient must be averaged this step;
    deferred Adam gradients would mix local and averaged quantities in the same norm.
    """
    if clip_after_reduce and skip_param_ids:
        raise ValueError("clip-after-reduce requires all gradients to be reduced every step")
    if grad_flat_buffer is not None and clip_ngram_params:
        raise ValueError("a flat gradient buffer cannot represent separate n-gram clipping")

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


def loss_scale_transition(scale: float, clean: int, finite: bool,
                          growth_interval: int, max_scale: float):
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
        if g is not None and not bool(torch.isfinite(g).all().item()):
            return False
    return True


def unscale_grads_(model: nn.Module, grad_flat_buffer: torch.Tensor | None,
                   scale: float) -> None:
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
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    flat_grad_buffer.mul_(clip_coef_clamped)
    return total_norm


# -----------------------------------------------------------------------------
# Checkpoint / scoring hook

ARCH_TOGGLE_NAMES = [
    "PACK_FACTOR", "INIT_SCALE", "USE_VE", "WINDOW_PATTERN", "SHORT_WINDOW_FRAC",
    "ROPE_BASE", "USE_LOCAL_CONV", "LOCAL_CONV_KERNEL", "TRAIN_LOCAL_CONV", "USE_GATED_MLP",
    "USE_WEIGHT_CACHE", "USE_TRANSPOSED_LINEAR", "WARMDOWN_RATIO", "FINAL_LR_FRAC", "WARMUP_STEPS",
    "WD_DECAY_TO_ZERO", "MUON_MOMENTUM_WARMUP_STEPS", "USE_DOC_MASK", "VE_WD",
    "USE_SELECTIVE_GATE", "SELECTIVE_GATE_CHANNELS", "USE_FUSED_QKV",
    "USE_CAUSAL_FASTPATH", "USE_TRN2_COMPILER_FLAGS",
    "COMPILE_SDPA_DIRECT", "USE_NKI_ATTN", "USE_NKI_LOCAL_CONV",
    # T0.4 resweep knobs. None of these changes a tensor's name or shape, so every one of them
    # is exactly the "loads silently with quietly wrong behavior" class this snapshot exists to
    # catch -- ATTN_CPROJ_INIT_SCALE and LOGIT_SOFTCAP especially, since they change what the
    # weights MEAN without changing what they look like.
    "MATRIX_LR", "EMBEDDING_LR", "UNEMBEDDING_LR", "SCALAR_LR", "WEIGHT_DECAY", "GRAD_CLIP",
    "CLIP_AFTER_REDUCE",
    "LOGIT_SOFTCAP", "LOSS_BYTE_WEIGHT", "LOSS_SCALE", "LOSS_SCALE_DYNAMIC",
    "ATTN_CPROJ_INIT_SCALE", "MUON_NS_STEPS",
    "MUON_MOMENTUM_COOLDOWN_FRAC", "MUON_MOMENTUM_FINAL",
    "ADAM_EVERY_N", "ADAM_EVERY_N_LR_COMP",
    # Plan §11.1 items 1-3 (Recursive's schedule tricks). Exactly the silent class this snapshot
    # exists for: none of them changes a tensor's name, shape or dtype, so a checkpoint trained
    # with a different beta1 anneal / beta2 ramp / per-depth LR loads without a murmur.
    "DEMON_BETA1", "DEMON_BETA1_FINAL", "DEMON_BETA1_REF", "DEMON_BETA1_START_FRAC",
    "MUON_BETA2", "MUON_BETA2_FINAL",
    "MUON_DEPTH_LR_BOTTOM", "MUON_DEPTH_LR_TOP",
    "MUON_DEPTH_MOM_BOTTOM", "MUON_DEPTH_MOM_TOP", "MUON_DEPTH_MOM_REF",
    # T0.2/T0.3. DEPTH/N_EMBD/HEAD_DIM/MLP_RATIO are all *also* recoverable from
    # model_config (n_layer / n_embd / n_embd//n_head / mlp_ratio), which is what
    # load_for_eval actually rebuilds from -- they are listed here only so a mismatch
    # is reported by name rather than as a wall of shape errors. ATTN_SCALE is the one
    # that is NOT in model_config and changes what the weights mean without changing
    # any shape, i.e. exactly the silent class this snapshot exists for.
    "DEPTH", "N_EMBD", "HEAD_DIM", "MLP_RATIO", "ATTN_SCALE", "N_KV_HEADS",
    # T3.1 hashed n-gram value embeddings. NGRAM_VE_BF16 is the interesting one to list
    # here: load_state_dict silently CASTS a mismatched dtype rather than erroring, so a
    # bf16-vs-fp32 mismatch is exactly the quiet class this snapshot exists for. The
    # optimizer-side knobs (LR/BETA2/FLAT_LR/OPT/CLIP_EXCLUDE) leave no trace in the
    # weights at all, which is the other half of the same reason.
    "USE_NGRAM_VE", "NGRAM_VE_LAYERS", "NGRAM_VE_TABLE_MULT", "NGRAM_VE_TRIGRAM", "NGRAM_VE_DIM",
    "NGRAM_VE_BF16", "NGRAM_RMS_FP32", "USE_NKI_NGRAM_RMS",
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
    # Plan 11.1 byte-feature wte init. Nothing here changes a shape -- it changes what the
    # SAME-shaped wte rows CONTAIN at step 0, which after a run is unrecoverable from the
    # weights themselves. That is the purest form of the silent class this snapshot exists for.
    "USE_BYTE_WTE_INIT", "BYTE_WTE_INIT_MIX", "BYTE_WTE_INIT_NGRAM", "BYTE_WTE_INIT_STD",
    # gpu-closed-techniques §3.2 coalesced SwiGLU: both are shape-critical (see below).
    "USE_COALESCED_SWIGLU", "COALESCED_SWIGLU_HIDDEN",
    # T2.3/T2.4. Pure optimizer knobs -- no tensor changes name, shape or dtype, and they leave
    # no trace at all in the saved weights, same reason the NGRAM_VE_* optimizer knobs are listed.
    "USE_CAUTIOUS_WD", "SCALAR_BETA1", "SCALAR_BETA2",
]
# Toggles whose mismatch corrupts the state_dict shape (load_state_dict(strict=True) will
# error loudly). Everything else in ARCH_TOGGLE_NAMES lives in non-persistent buffers or
# plain Python constants excluded from state_dict() entirely -- a mismatch there loads
# silently with no crash and quietly wrong behavior, which is exactly why we snapshot and
# check all of them, not just the shape-critical ones.
SHAPE_CRITICAL_TOGGLES = {"USE_VE", "USE_LOCAL_CONV", "USE_GATED_MLP", "USE_SELECTIVE_GATE", "USE_FUSED_QKV",
                          # T1.6: transposes every Linear weight in place. Same state_dict KEYS,
                          # flipped shapes, so a cross-setting strict load errors loudly -- which
                          # is the desired behaviour, see transpose_linear_weights().
                          "USE_TRANSPOSED_LINEAR",
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
                          "ENGRAM_SHARE_NGRAM_TABLES",
                          # Sub-1.0 P2: N_KV_HEADS resizes attn.c_k/c_v (n_embd -> n_kv*head_dim),
                          # so a cross-setting strict load errors loudly, which is what we want.
                          "N_KV_HEADS",
                          # Coalesced SwiGLU resizes mlp.c_fc's output (hidden -> 2*hidden) and
                          # mlp.c_proj's input (2560 -> COALESCED_SWIGLU_HIDDEN), so a cross-setting
                          # strict load errors loudly, which is what we want. HIDDEN is listed for
                          # the same reason NGRAM_VE_TABLE_MULT is: it resizes both of them again.
                          "USE_COALESCED_SWIGLU", "COALESCED_SWIGLU_HIDDEN"}


def current_toggles() -> dict:
    g = globals()
    return {name: g[name] for name in ARCH_TOGGLE_NAMES}


# Capture import-time defaults once. A legacy checkpoint lacking a new toggle
# must not inherit it from a different checkpoint loaded earlier in this process.
_EVAL_TOGGLE_DEFAULTS = current_toggles()


def restore_eval_toggles(payload: dict) -> None:
    """Restore only known model/training settings from a weights-only checkpoint.

    Architecture and arithmetic flags are part of the frozen model definition;
    the standalone scorer supplies no training CLI arguments. Unknown payload
    keys cannot replace arbitrary module globals. No data-dependent adaptation
    or evaluation state is introduced.
    """
    saved = payload.get("toggles", {})
    if not isinstance(saved, dict):
        raise ValueError("checkpoint toggles must be a dictionary")
    for name in ARCH_TOGGLE_NAMES:
        globals()[name] = saved.get(name, _EVAL_TOGGLE_DEFAULTS[name])


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
    payload = torch_load_weights(checkpoint_path, map_location="cpu")
    restore_eval_toggles(payload)
    if USE_DOC_MASK:
        BOS_TOKEN_ID = ensure_tokenizer(build_if_missing=False).get_bos_token_id()
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
    start = DEMON_BETA1_START_FRAC if DEMON_BETA1_START_FRAC >= 0.0 else (1.0 - WARMDOWN_RATIO)
    start = min(max(start, 0.0), 0.999)
    frac = min(1.0, max(0.0, (progress - start) / max(1e-6, 1.0 - start)))
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
    return lr_on, mom_on


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
    lr_mult, mom_mult = [], []
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
            mom_mult.append((1.0 - target) / max(1e-6, 1.0 - MUON_DEPTH_MOM_REF))
    return lr_mult, mom_mult


def main() -> None:
    global USE_LOCAL_CONV, LRM_QUANTUM, USE_TENSOR_LR_SCALARS, USE_FOREACH_OPTIM  # flags below default off these constants
    global LOCAL_CONV_KERNEL, TRAIN_LOCAL_CONV, USE_NKI_LOCAL_CONV
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
    global ADAM_EVERY_N, ADAM_EVERY_N_LR_COMP, PACK_FACTOR, GRAD_CLIP, LOSS_SCALE
    global CLIP_AFTER_REDUCE
    global COMPILE_SDPA_DIRECT
    global LOSS_SCALE_DYNAMIC, loss_scale_clean, loss_scale_skips
    # T0.2/T0.3 capacity-shape knobs. All read inside build_config()/GPT(config), i.e. well before
    # the first step, so none of them can add a per-step graph constant.
    global DEPTH, N_EMBD, HEAD_DIM, MLP_RATIO, ATTN_SCALE, SEQ_LEN, N_KV_HEADS
    # T3.1. USE_NGRAM_VE / NGRAM_VE_TABLE_MULT / NGRAM_VE_TRIGRAM / NGRAM_VE_BF16 are read by
    # GPT.__init__ and CausalSelfAttention.__init__ (before GPT(config)); the rest are read by
    # setup_optimizer() or once per step in the training loop, all well before any device op
    # that could bake them in as a per-step graph constant.
    global USE_NGRAM_VE, NGRAM_VE_TABLE_MULT, NGRAM_VE_TRIGRAM, NGRAM_VE_BF16, NGRAM_VE_DIM
    global NGRAM_VE_LAYERS
    global NGRAM_RMS_FP32, USE_NKI_NGRAM_RMS
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
    global USE_TRANSPOSED_LINEAR  # T1.6, same reason as USE_COMPILE_MUON above
    # Plan 11.1 / gpu-closed-techniques §3.2 / T2.3 / T2.4 -- same rule again: every one of these is
    # read by its own argparse default line below, so the `global` has to precede the parser.
    global USE_BYTE_WTE_INIT, BYTE_WTE_INIT_MIX, BYTE_WTE_INIT_NGRAM, BYTE_WTE_INIT_STD
    global USE_COALESCED_SWIGLU, COALESCED_SWIGLU_HIDDEN
    global USE_CAUTIOUS_WD, SCALAR_BETA1, SCALAR_BETA2
    # Plan §11.1 items 1-3 (Recursive's schedule tricks). Same rule as the block above: every one
    # of these is read by its own argparse default line, so the `global` must precede the parser.
    # All of them are read either by setup_optimizer() or once per step in the training loop, so
    # none of them can add a per-step graph constant (see each knob's comment for the discipline).
    global DEMON_BETA1, DEMON_BETA1_FINAL, DEMON_BETA1_REF, DEMON_BETA1_START_FRAC
    global MUON_BETA2, MUON_BETA2_FINAL
    global MUON_DEPTH_LR_BOTTOM, MUON_DEPTH_LR_TOP
    global MUON_DEPTH_MOM_BOTTOM, MUON_DEPTH_MOM_TOP, MUON_DEPTH_MOM_REF
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
                             "Default ON (Solution 12, real ~15-18%% step-time win, verified to 1-bf16-ULP "
                             "against eager) -- pass --no-nki-rope-norm for the plain eager comparison path.")
    parser.add_argument("--nki-softcap-ce", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_NKI_SOFTCAP_CE", "1") != "0",
                        help="fuse the logit softcap and cross-entropy (fwd+bwd) into one NKI kernel pair "
                             "(Neuron only), collapsing ~8 full-[B*T,V] NEFF launches into 2. Default ON: "
                             "verified numerically offline (research/scratch/softcap_ce_equiv.py) and confirmed "
                             "on real hardware at the final locked shape -- pass --no-nki-softcap-ce for the "
                             "plain eager comparison path.")
    parser.add_argument("--byte-loss-weight", action=argparse.BooleanOptionalAction,
                        # Literal default, like the other --nki-* flags: main() rebinds
                        # LOSS_BYTE_WEIGHT as a global below, so it cannot also read it here.
                        default=os.environ.get("NEURON_COMPETITION_R1_BYTE_LOSS_WEIGHT",
                                               "0") != "0",
                        help="Sub-1.0 plan P4: weight each token's training loss by its byte length "
                             "(sum(w*l)/sum(w), mean-normalized so the loss scale is unchanged), "
                             "aligning the objective with bpb. Weights come from the provided "
                             "tokenizer artifact only. Default OFF = plain mean, bit-identical.")
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
    parser.add_argument("--nki-attn", action="store_true",
                        default=os.environ.get("NEURON_COMPETITION_R1_NKI_ATTN", "0") != "0",
                        help="run causal SDPA through the fused NKI fwd/bwd pair (Neuron only). "
                             "Default OFF.")
    parser.add_argument("--compile-sdpa-direct", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_COMPILE_SDPA_DIRECT",
                                               "1" if COMPILE_SDPA_DIRECT else "0") != "0",
                        help="Call causal SDPA directly when --nki-attn is off, allowing Dynamo "
                             "to trace it instead of entering the disabled attention wrapper. "
                             "Default OFF preserves the historical graph partitioning. The "
                             "custom NKI attention path retains its wrapper when enabled.")
    parser.add_argument("--muon-lr-fp32", action=argparse.BooleanOptionalAction,
                        # literal default, like --nki-muon above: this function rebinds
                        # USE_MUON_LR_FP32 as a global, so it cannot also read it here.
                        default=os.environ.get("NEURON_COMPETITION_R1_MUON_LR_FP32", "0") != "0",
                        help="see the _muon_lr_g docstring. Default OFF (the locked, current "
                             "behaviour): a 0-dim fp32 lr multiplied against the bf16 Newton-Schulz "
                             "iterate g gets CAST DOWN to bf16 before the multiply, by PyTorch's own "
                             "0-dim type-promotion rule -- quantising every Muon group's effective lr "
                             "by up to ~0.4%% relative error, every step. ON keeps that product in "
                             "fp32 instead (g upcast, lr untouched, cast down to w's dtype only at the "
                             "final subtract). Zero-cost precision fix, no expected throughput impact; "
                             "does not touch the per-depth [K,1,1] lr path's intentional bf16-matching "
                             "behaviour. Its own flag and its own A/B, per that docstring.")
    parser.add_argument("--compile-muon", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_COMPILE_MUON",
                                               "1" if USE_COMPILE_MUON else "0") != "0",
                        help="T1.7: run Muon's per-group math body inside "
                             "torch.compile(backend='neuron') instead of ~80 eager aten launches "
                             "(Neuron only). Mutually exclusive with --nki-muon by construction -- a "
                             "compiled region cannot contain an nki.jit call -- so this trades the NKI "
                             "elementwise fusion for whole-body fusion. See _muon_math.")
    parser.add_argument("--transposed-linear", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_TRANSPOSED_LINEAR",
                                               "1" if USE_TRANSPOSED_LINEAR else "0") != "0",
                        help="T1.6: store every Linear weight as [in, out] and compute x @ weight "
                             "instead of F.linear's x @ weight.T, on the theory that TensorE wants the "
                             "contraction dim on partitions and the [out, in] storage costs a transpose "
                             "per matmul. Bit-exact by construction (weights are initialized in the "
                             "native layout and transposed afterwards; Muon is told the logical "
                             "orientation so its one-sided `ratio` LR scale is unchanged). "
                             "SHAPE-CRITICAL: flips every Linear weight's state_dict shape.")
    parser.add_argument("--local-conv", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_LOCAL_CONV", "1" if USE_LOCAL_CONV else "0") != "0",
                        help="depthwise causal conv1d before attention in each block. NOT dead compute "
                             "any more: T0.1 put local_conv.weight in its own AdamW group "
                             "(TRAIN_LOCAL_CONV=True), so --no-local-conv now REMOVES 7,680 TRAINED "
                             "parameters and is a real capacity ablation, not the bit-identical no-op "
                             "this help text used to claim. Measured on the promoted stack (120-step "
                             "instrument, throughput->capacity track): --no-local-conv is +2.65%% faster "
                             "(161,075 vs 156,917 tok/s) but its step-110 loss is 4.2054 vs the "
                             "control's 3.9380 -- a huge quality cost for 2.65%%, so DO NOT ablate it "
                             "for speed. (This also reverses the older 'turning the compute off "
                             "measured 6.9x-compiled-graph SLOWER' note in the USE_LOCAL_CONV comment: "
                             "that was the pre-NKI eager stack.) Kept as a flag so an A/B runs from "
                             "one identical binary.")
    parser.add_argument("--nki-local-conv", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_NKI_LOCAL_CONV",
                                               "1" if USE_NKI_LOCAL_CONV else "0") != "0",
                        help="Use the gated BTC-layout native NKI two-tap convolution for "
                             "FP32 residuals with time/channels divisible by 128. Other "
                             "shapes and precisions retain the original causal convolution.")
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
                        help="store n-gram tables and gradients in bf16 rather than fp32. "
                             "Optimizer state follows that dtype unless --ngram-rms-fp32 is on.")
    parser.add_argument("--ngram-rms-fp32", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_NGRAM_RMS_FP32",
                                               "1" if NGRAM_RMS_FP32 else "0") != "0",
                        help="Use FP32 second moments AND FP32 RMSProp update arithmetic for "
                             "n-gram tables, retaining their parameter/gradient dtype. Casts "
                             "grads before squaring and rounds at the final parameter write. "
                             "Requires --ngram-ve-opt rmsprop. Default OFF.")
    parser.add_argument("--nki-ngram-rms", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_NKI_NGRAM_RMS",
                                               "1" if USE_NKI_NGRAM_RMS else "0") != "0",
                        help="Fuse FP32 table RMSProp in NKI, using both cores under LNC2. "
                             "Requires --ngram-rms-fp32 and contiguous BF16 Neuron tables. "
                             "Default OFF; test quality and whole-run time before promotion.")
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

    # Plan 11.1: byte-feature wte init.
    parser.add_argument("--byte-wte-init", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_BYTE_WTE_INIT",
                                               "1" if USE_BYTE_WTE_INIT else "0") != "0",
                        help="Plan 11.1 (Recursive's from-vanilla run): initialize wte from a hashed "
                             "bag of byte n-grams of each token's own byte string (from the provided "
                             "tokenizer artifact's id->bytes map -- no external data) instead of "
                             "normal_(0, 0.8), so tokens with similar surface form start correlated. "
                             "Host-side and one-shot: zero ops added to the training graph, and no "
                             "torch RNG consumed, so every OTHER tensor is initialized identically to "
                             "the control arm. See byte_feature_wte_init.")
    _hp("--byte-wte-init-mix", BYTE_WTE_INIT_MIX, float,
        "Plan 11.1: 1.0 = pure byte features; mix<1 blends mix*feat + sqrt(1-mix^2)*random, which "
        "keeps the row RMS and guarantees no two rows can coincide on a hash collision.")
    _hp("--byte-wte-init-ngram", BYTE_WTE_INIT_NGRAM, int,
        "Plan 11.1: highest byte-n-gram order hashed into each row (1..N).")
    _hp("--byte-wte-init-std", BYTE_WTE_INIT_STD, float,
        "Plan 11.1: target per-row RMS of the byte-feature rows. Must track init_weights' own "
        "normal_ std for wte (0.8) or the arm also becomes an embedding-scale test.")
    # gpu-closed-techniques §3.2: parameter-matched, coalesced SwiGLU.
    parser.add_argument("--coalesced-swiglu", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_COALESCED_SWIGLU",
                                               "1" if USE_COALESCED_SWIGLU else "0") != "0",
                        help="parameter-matched coalesced SwiGLU MLP: one Linear(n_embd, 2*hidden) "
                             "split into gate and up halves, silu(gate)*up, at hidden = 8/3*n_embd "
                             "(1664) so the three matrices cost the same as the baseline's two. "
                             "Distinct from --gated-mlp/USE_GATED_MLP, which keeps hidden=4*n_embd "
                             "and so adds 50%% MLP params and FLOPs. SHAPE-CRITICAL. Prior ~15-20%%: "
                             "the ReLU^2 >= SwiGLU consensus at this scale is hardware-independent.")
    _hp("--coalesced-swiglu-hidden", COALESCED_SWIGLU_HIDDEN, int,
        "force the coalesced-SwiGLU hidden width (must be a multiple of 128). 0 = auto, i.e. the "
        "parameter-matched round(2/3*mlp_ratio*n_embd / 128)*128. SHAPE-CRITICAL.")
    # T2.3 / T2.4: cautious weight decay on the Adam groups, and the scalar group's own betas.
    parser.add_argument("--cautious-wd", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_CAUTIOUS_WD",
                                               "1" if USE_CAUTIOUS_WD else "0") != "0",
                        help="T2.3 (modded #50): in the AdamW groups, apply weight decay only where "
                             "update*p > 0, so decay never fights an element the gradient is driving "
                             "outward. Affects only the groups with wd != 0 -- wte (0.001) and "
                             "lm_head (0.01). The Muon groups already do this (see _muon_math).")
    _hp("--scalar-beta1", SCALAR_BETA1, float,
        "T2.4 (modded #52): Adam beta1 for the scalar/skip group. Default 0.9 = historical.")
    _hp("--scalar-beta2", SCALAR_BETA2, float,
        "T2.4 (modded #52): Adam beta2 for the scalar/skip group. Default 0.95 = historical; "
        "record #52's 'smoothed scalars' uses 0.99, usually with a lower --scalar-lr.")
    # Plan §11.1 items 1-3: Recursive's three schedule-level tricks. All zero-new-op, all default
    # to exactly the current behaviour.
    parser.add_argument("--demon-beta1", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_DEMON_BETA1",
                                               "1" if DEMON_BETA1 else "0") != "0",
                        help="plan 11.1 (Recursive): anneal the AdamW groups' beta1 down across the "
                             "LR warmdown ('Demon'), as a multiplicative shrink on each group's own "
                             "initial beta1 calibrated so a 0.8 group lands on --demon-beta1-final. "
                             "The rmsprop n-gram group has no first moment and is unaffected.")
    _hp("--demon-beta1-final", DEMON_BETA1_FINAL, float,
        "plan 11.1: beta1 at the end of the run for a group whose initial beta1 is "
        "--demon-beta1-ref. Recursive uses 0.55 from 0.8. Only read with --demon-beta1.")
    _hp("--demon-beta1-ref", DEMON_BETA1_REF, float,
        "plan 11.1: the initial beta1 that --demon-beta1-final is quoted against (Recursive: 0.8, "
        "which is also this repo's embedding-family beta1).")
    _hp("--demon-beta1-start-frac", DEMON_BETA1_START_FRAC, float,
        "plan 11.1: wall-clock progress fraction at which the beta1 anneal starts. Negative "
        "(default) = start where the LR warmdown starts, i.e. 1 - WARMDOWN_RATIO = 0.25, which is "
        "Recursive's 'across its warmdown'. 0.0 = anneal across the whole run.")
    _hp("--muon-beta2", MUON_BETA2, float,
        "plan 11.1 / NorMuon: the Muon groups' second-moment beta2, i.e. the decay of the per-row "
        "second moment of the ORTHOGONALIZED update that rescales it. Default 0.90 = the tuned "
        "historical literal. Recursive's variant sits at 0.95->0.97; SkyPilot's autoresearch found "
        "0.95->0.98 worth ~0.001 at their setting -- neither transfers as a VALUE here, only as a "
        "direction, so sweep it.")
    _hp("--muon-beta2-final", MUON_BETA2_FINAL, float,
        "plan 11.1 / NorMuon: ramp the Muon beta2 linearly from --muon-beta2 to this over "
        "elapsed/max_train_seconds. 0.0 (default) = OFF, and specifically = keep handing the "
        "compiled Muon body `1-beta2` as the Python float it is today, so the traced graph is "
        "byte-identical. Recursive ramps 0.95 -> 0.97.")
    _hp("--muon-depth-lr-bottom", MUON_DEPTH_LR_BOTTOM, float,
        "plan 11.1: Muon LR multiplier for layer 0's matrices, ramped linearly to "
        "--muon-depth-lr-top at the last layer. Recursive: 1.15 -> 0.85. Both 1.0 (default) = OFF. "
        "Carried as a per-parameter [K,1,1] operand inside the existing per-shape groups, so the "
        "batched Newton-Schulz and the compiled-graph count are unchanged.")
    _hp("--muon-depth-lr-top", MUON_DEPTH_LR_TOP, float,
        "plan 11.1: Muon LR multiplier for the LAST layer's matrices. Recursive: 0.85.")
    _hp("--muon-depth-mom-bottom", MUON_DEPTH_MOM_BOTTOM, float,
        "plan 11.1: target Muon momentum for layer 0 when the global scheduled momentum is at "
        "--muon-depth-mom-ref. Recursive: 0.90 (bottom) -> 0.97 (top). Applied as a multiplier on "
        "(1 - momentum) so it rides ON TOP of the global warmup/cooldown schedule and can never "
        "push momentum past 1. 0.0 (default, either endpoint) = OFF.")
    _hp("--muon-depth-mom-top", MUON_DEPTH_MOM_TOP, float,
        "plan 11.1: target Muon momentum for the LAST layer. Recursive: 0.97.")
    _hp("--muon-depth-mom-ref", MUON_DEPTH_MOM_REF, float,
        "plan 11.1: the global momentum the two --muon-depth-mom-* targets are quoted against. "
        "0.95 (default) = muon_momentum()'s warmed-up plateau.")
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
    _hp("--n-kv-heads", N_KV_HEADS, int, "Sub-1.0 plan P2: grouped-query attention KV-head count. "
        "0 (default) = dense, bit-identical. Must divide n_head; at the locked width (n_head=5) "
        "the only nontrivial choice is 1 (MQA). Shrinks the K/V projections; QK^T stays full-width.")
    _hp("--head-dim", HEAD_DIM, int, "T0.2: per-head width; n_head = n_embd / head_dim. Also the "
        "rounding granularity of the derived n_embd. Currently 128. The fused rope+QK-norm NKI "
        "kernel is written generically in D (it only requires D even), so 64 stays on the kernel.")
    _hp("--mlp-ratio", MLP_RATIO, int, "T0.3: MLP hidden width as a multiple of n_embd. Currently 4.")
    _hp("--attn-scale", ATTN_SCALE, float, "T0.3: explicit SDPA softmax scale. 0.0 (default) = SDPA's "
        "own 1/sqrt(head_dim) (0.0884 at 128). QK-norm already fixes the q/k norms, so the logit "
        "temperature is a free knob here rather than a redundant init reparameterisation.")
    _hp("--matrix-lr", MATRIX_LR, float, "T0.4: Muon groups' base lr (also the AdamW lr for the "
        "gate/local_conv groups). Currently 0.025.")
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
    _hp("--grad-clip", GRAD_CLIP, float, "Gradient-norm clip threshold. By default clips each "
        "rank's local gradients before all_reduce(AVG); --clip-after-reduce clips the averaged "
        "global-batch gradients. 0 disables clipping. Separate n-gram clipping is preserved.")
    parser.add_argument("--clip-after-reduce", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("NEURON_COMPETITION_R1_CLIP_AFTER_REDUCE",
                                               "1" if CLIP_AFTER_REDUCE else "0") != "0",
                        help="Average all accumulated gradients before clipping. Default OFF "
                             "preserves the historical local-clip-then-average update. Requires "
                             "--adam-every-n 1; supports separate n-gram clipping.")
    _hp("--loss-scale", LOSS_SCALE, float, "Sub-1.0 plan P2: static loss-scale factor. "
        "Multiplies the training loss before backward and divides grads back before "
        "clip/sync/step, so backward grads stay representable under compiler fp8 "
        "auto-cast (full-model grads underflow fp8_e4m3 otherwise). 1.0 (default) = "
        "off, bit-identical. Use powers of 2 (128/1024).")
    parser.add_argument("--loss-scale-dynamic", action=argparse.BooleanOptionalAction,
                        # Literal default: main() rebinds LOSS_SCALE_DYNAMIC as a global below,
                        # so it cannot also read it here (same pattern as --byte-loss-weight).
                        default=os.environ.get("NEURON_COMPETITION_R1_LOSS_SCALE_DYNAMIC",
                                               "0") != "0",
                        help="Dynamic loss scaling for fp8 auto-cast: skip+halve the update on "
                             "grad overflow (all ranks agree via all-reduce), double the scale "
                             "every 500 clean steps (cap 8192). Starts from --loss-scale. "
                             "Default OFF.")
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
    if args.ngram_rms_fp32 and args.ngram_ve_opt != "rmsprop":
        parser.error("--ngram-rms-fp32 requires --ngram-ve-opt rmsprop")
    if args.nki_ngram_rms and not args.ngram_rms_fp32:
        parser.error("--nki-ngram-rms requires --ngram-rms-fp32")
    if args.eval_seq_len:
        if args.eval_seq_len > args.seq_len:
            parser.error(f"--eval-seq-len {args.eval_seq_len} > --seq-len {args.seq_len}: the live "
                         "model's rotary/mask tables are only built out to --seq-len. Use "
                         "`prepare.py eval-public --seq-len N` on the saved checkpoint instead.")
        if args.eval_seq_len % 128 != 0:
            parser.error("--eval-seq-len must be a multiple of 128 (NKI tiling).")
    if args.adam_every_n < 1:
        parser.error("--adam-every-n must be >= 1")
    if args.clip_after_reduce and args.adam_every_n > 1:
        parser.error("--clip-after-reduce requires --adam-every-n 1: deferred Adam gradients "
                     "would mix unreduced local gradients with averaged gradients in the norm.")
    if args.adam_every_n > 1 and args.grad_bucket:
        parser.error("--adam-every-n > 1 is incompatible with --grad-bucket: the flat buffer is "
                     "zeroed wholesale every step, so it cannot leave the AdamW groups' gradients "
                     "standing to accumulate. (--grad-bucket is also a known hang on this hardware; "
                     "see trn2-experiment-results.md.)")

    global USE_NKI_RELU2, USE_NKI_ROPE_NORM, USE_NKI_SOFTCAP_CE, USE_NKI_MUON, USE_MUON_LR_FP32, USE_NKI_ATTN
    global LOSS_BYTE_WEIGHT, BYTE_LOSS_W
    # (USE_COMPILE_MUON's `global` is at the top of main(); see the note there.)
    # T0.4/T1.4 resweep knobs (see the _hp block above; declared global at the top of main()).
    # Assigned here, i.e. before GPT(config) and before setup_optimizer(), which is where every
    # one of them is read.
    ADAM_EVERY_N = args.adam_every_n
    ADAM_EVERY_N_LR_COMP = args.adam_every_n_lr_comp
    GRAD_CLIP = args.grad_clip
    CLIP_AFTER_REDUCE = args.clip_after_reduce
    LOSS_SCALE = args.loss_scale
    if not (LOSS_SCALE > 0):
        raise ValueError(f"--loss-scale must be positive, got {LOSS_SCALE}")
    LOSS_SCALE_DYNAMIC = args.loss_scale_dynamic
    loss_scale_clean = 0
    loss_scale_skips = 0
    PACK_FACTOR = args.pack_factor
    # T0.2/T0.3: all four are read by build_config()/MLP.__init__/CausalSelfAttention.__init__ below.
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
    INIT_SCALE = args.init_scale               # read by GPT.init_weights, i.e. inside GPT(config)
    ATTN_CPROJ_INIT_SCALE = args.attn_cproj_init  # ditto
    LOGIT_SOFTCAP = args.logit_softcap
    MUON_NS_STEPS = args.muon_ns_steps
    MUON_MOMENTUM_COOLDOWN_FRAC = args.muon_momentum_cooldown_frac
    MUON_MOMENTUM_FINAL = args.muon_momentum_final
    USE_VE = args.ve                         # read by Block/CausalSelfAttention/GPT.__init__, i.e. before GPT(config) below
    USE_LOCAL_CONV = args.local_conv         # read by Block.__init__, i.e. before GPT(config) below
    USE_NKI_LOCAL_CONV = args.nki_local_conv
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
    NGRAM_RMS_FP32 = args.ngram_rms_fp32
    USE_NKI_NGRAM_RMS = args.nki_ngram_rms
    NGRAM_VE_OPT = args.ngram_ve_opt
    NGRAM_VE_LR = args.ngram_ve_lr
    NGRAM_VE_BETA2 = args.ngram_ve_beta2
    NGRAM_VE_FLAT_LR = args.ngram_ve_flat_lr
    NGRAM_VE_CLIP_EXCLUDE = args.ngram_ve_clip_exclude
    # Plan 11.1 byte-feature wte init: read after model.init_weights(), on the host.
    USE_BYTE_WTE_INIT = args.byte_wte_init
    BYTE_WTE_INIT_MIX = args.byte_wte_init_mix
    BYTE_WTE_INIT_NGRAM = args.byte_wte_init_ngram
    BYTE_WTE_INIT_STD = args.byte_wte_init_std
    if not 0.0 <= BYTE_WTE_INIT_MIX <= 1.0:
        parser.error(f"--byte-wte-init-mix must be in [0, 1], got {BYTE_WTE_INIT_MIX}")
    if BYTE_WTE_INIT_NGRAM < 1:
        parser.error(f"--byte-wte-init-ngram must be >= 1, got {BYTE_WTE_INIT_NGRAM}")
    # gpu-closed-techniques §3.2 coalesced SwiGLU: read by MLP.__init__, i.e. inside GPT(config).
    USE_COALESCED_SWIGLU = args.coalesced_swiglu
    COALESCED_SWIGLU_HIDDEN = args.coalesced_swiglu_hidden
    if USE_COALESCED_SWIGLU:
        if USE_GATED_MLP:
            parser.error("--coalesced-swiglu and USE_GATED_MLP are two different SwiGLU MLPs and "
                         "both own mlp.c_fc/c_gate; pick one (USE_GATED_MLP has no flag, so this "
                         "can only fire if the module constant was edited). --coalesced-swiglu is "
                         "the parameter-matched, single-matmul form.")
        if args.parallel_block and args.parallel_block_coalesce != "none":
            parser.error("--coalesced-swiglu is incompatible with --parallel-block-coalesce "
                         f"{args.parallel_block_coalesce}: block.c_in is sized mlp_ratio*n_embd and "
                         "hands mlp.forward a single pre-activation, not a gate/up pair.")
        if COALESCED_SWIGLU_HIDDEN % 128 != 0:
            parser.error(f"--coalesced-swiglu-hidden must be a multiple of 128 (0 = auto), got "
                         f"{COALESCED_SWIGLU_HIDDEN}")
    # T2.3 / T2.4: read by setup_optimizer() and by the AdamW step.
    USE_CAUTIOUS_WD = args.cautious_wd
    SCALAR_BETA1 = args.scalar_beta1
    SCALAR_BETA2 = args.scalar_beta2
    # Plan §11.1 items 1-3. MUON_BETA2 and the MUON_DEPTH_* knobs are read by setup_optimizer();
    # DEMON_* and MUON_BETA2_FINAL are read once per step by the schedule block in the loop.
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
    if MUON_BETA2_FINAL > 0.0 and not (0.0 < MUON_BETA2_FINAL < 1.0):
        parser.error(f"--muon-beta2-final must be in (0, 1) when set, got {MUON_BETA2_FINAL}")
    if (MUON_DEPTH_MOM_BOTTOM > 0.0) != (MUON_DEPTH_MOM_TOP > 0.0):
        parser.error("--muon-depth-mom-bottom and --muon-depth-mom-top must both be set or "
                     "both left at 0.0 (the OFF sentinel is the pair, not either one)")
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
        for name, on in (("--qk-shift", USE_QK_SHIFT),
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
    USE_NKI_RELU2 = args.nki_relu2
    USE_NKI_ROPE_NORM = args.nki_rope_norm
    USE_NKI_SOFTCAP_CE = args.nki_softcap_ce
    LOSS_BYTE_WEIGHT = args.byte_loss_weight
    USE_NKI_MUON = args.nki_muon
    USE_NKI_ATTN = args.nki_attn
    COMPILE_SDPA_DIRECT = args.compile_sdpa_direct
    USE_MUON_LR_FP32 = args.muon_lr_fp32
    USE_TRANSPOSED_LINEAR = args.transposed_linear
    USE_COMPILE_MUON = args.compile_muon
    if USE_COMPILE_MUON and USE_NKI_MUON:
        # Not an error, because both are ON by default now and --compile-muon is the faster of the
        # two (+3.38% vs the NKI path at the locked config). Silent precedence inside _muon_step
        # would be worse than saying so: a reader looking at `nki_muon: True` in the config dump has
        # to know that the compiled branch is checked first. Turn it off explicitly instead, so the
        # dump tells the truth about which Muon step actually ran.
        USE_NKI_MUON = False
        print0("muon step: --compile-muon supersedes --nki-muon (a compiled region cannot contain "
               "an nki.jit call); NKI Muon disabled for this run")

    device_type = detect_device_type(args.device_type)
    if USE_NKI_NGRAM_RMS and (device_type != "neuron" or not _HAS_NKI):
        parser.error("--nki-ngram-rms requires a Neuron device and available NKI backend")
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type, seed=args.seed)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)

    print0(f"device={device} dtype={COMPUTE_DTYPE} world_size={world_size}")
    print0(f"training target={args.num_steps} steps max_seconds={args.max_train_seconds} official_budget={TRAIN_TIME_BUDGET_SECONDS}s")
    print0(f"gradient_clip={GRAD_CLIP:g} clip_after_reduce={CLIP_AFTER_REDUCE}")
    print0(f"attention: nki_attn={USE_NKI_ATTN} compile_sdpa_direct={COMPILE_SDPA_DIRECT}")
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
    if USE_BYTE_WTE_INIT:
        # Plan 11.1. AFTER init_weights (so every other tensor's draws are untouched -- wte's own
        # normal_ still runs and is then overwritten, which is what keeps the RNG stream identical
        # to the control) and BEFORE warm_start_from/model.to(device): host-side, one-shot, zero
        # ops in the training graph. The print is the arm's null test.
        n_ok, n_skip, feats = byte_feature_wte_init(model.transformer.wte.weight, tokenizer,
                                                    config.vocab_size)
        print0(f"byte-wte-init: {n_ok} rows from byte features, {n_skip} left random "
               f"(mix={BYTE_WTE_INIT_MIX} max_n={BYTE_WTE_INIT_NGRAM} std={BYTE_WTE_INIT_STD} "
               f"mean_features_per_token={feats:.1f})")
    if args.init_from:
        warm_start_from(args.init_from, model)
    model.to(device)
    if LOSS_BYTE_WEIGHT:
        # Build once, host-side, before the first step: per-token byte lengths from the
        # provided tokenizer artifact (prepare.get_token_bytes == token_bytes.pt). Specials
        # record 0 bytes -- clamp to 1 so BOS/specials keep ~mean-CE weight instead of
        # dropping out of the loss entirely. Mean-normalized so the loss (and every LR
        # tuned against its scale) is unchanged in expectation.
        tb = get_token_bytes("cpu").to(torch.float32)
        tb = torch.clamp(tb, min=1.0)
        BYTE_LOSS_W = tb / tb.mean()
        print0(f"byte-loss-weight: mean_bytes={tb.mean().item():.3f} max_bytes={tb.max().item():.0f} "
               f"weight_range=({BYTE_LOSS_W.min().item():.3f}, {BYTE_LOSS_W.max().item():.3f})")
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
        clip_main, clip_ngram = split_clip_params(orig_model)
        clip_and_sync_gradients(model, clip_main, clip_ngram, max_norm=GRAD_CLIP,
                                clip_after_reduce=CLIP_AFTER_REDUCE, foreach=args.clip_foreach)
        optimizer.step()
        synchronize(device)
        print0("compile-only synthetic training step completed")
        cleanup_runtime()
        return

    loader_seq_len = PACK_FACTOR * SEQ_LEN
    if USE_EPOCH_SHUFFLE:
        # train.py's own copy of the packer: prepare.make_dataloader line-for-line, so
        # --epoch-shuffle changes the data ORDER and nothing else.
        train_loader = make_dataloader_requeue(
            tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, "train", device,
            _document_batches_fn=_document_batches_epoch_shuffle,
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
            # LOSS_SCALE (default 1.0, IEEE-exact no-op): static loss scaling so backward
            # grads stay representable under fp8 auto-cast; unscaled below before clip.
            ((loss * inv_gas_dev if BATCH_RAMP else loss / grad_accum_steps) * LOSS_SCALE).backward()
            x, y, loader_state = fetch()

        # Loss scaling: static unscale, plus the dynamic controller (default off).
        # The overflow check runs on the SCALED grads (they overflow first), and
        # all ranks agree on the skip via MAX all-reduce (same pattern as the
        # bad-loss collective below) so no rank steps while its peers skip.
        # Growth takes effect NEXT step: unscale uses the scale the grads were
        # built with, never the just-grown one.
        skip_step = False
        unscale_with = LOSS_SCALE
        if LOSS_SCALE_DYNAMIC:
            finite = grads_finite_(model, grad_flat_buffer)
            ov = torch.tensor([0.0 if finite else 1.0], device=device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(ov, op=dist.ReduceOp.MAX)
            new_scale, new_clean, skip_step = loss_scale_transition(
                LOSS_SCALE, loss_scale_clean, bool(ov[0].item() == 0.0),
                LOSS_SCALE_GROWTH_INTERVAL, LOSS_SCALE_MAX)
            if skip_step:
                loss_scale_skips += 1
                print0(f"loss-scale: overflow at step {step}, skipping update "
                       f"(scale {LOSS_SCALE:g} -> {new_scale:g}, "
                       f"skips {loss_scale_skips})", flush=True)
                if grad_flat_buffer is not None:
                    grad_flat_buffer.zero_()
                else:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.detach().zero_()
            LOSS_SCALE, loss_scale_clean = new_scale, new_clean
        unscale_grads_(model, grad_flat_buffer, unscale_with)
        clip_and_sync_gradients(
            model, clip_main_params, clip_ngram_params, max_norm=GRAD_CLIP,
            clip_after_reduce=CLIP_AFTER_REDUCE, grad_flat_buffer=grad_flat_buffer,
            foreach=args.clip_foreach,
            skip_param_ids=None if adam_step_now else adamw_skip_ids,
        )

        lrm = lr_multiplier(step, elapsed=elapsed, max_train_seconds=args.max_train_seconds)
        # T3.1: warmup-only multiplier for the groups that opt out of the warmdown (the
        # hashed n-gram tables). Identical to lr_multiplier() during warmup and pinned at
        # 1.0 afterwards, so those groups hold a FLAT lr for the whole run. It settles on
        # exactly one value after WARMUP_STEPS, so it costs no extra graph constants.
        lrm_flat = min(1.0, (step + 1) / max(1, WARMUP_STEPS))
        wd_progress = min(1.0, elapsed / max(1e-6, args.max_train_seconds))
        # Plan §11.1 items 1-2. Both are keyed on exactly the same wall-clock progress fraction the
        # LR warmdown, the WD decay and the Muon momentum cooldown already use, and both are
        # computed ONCE per step here rather than per group. demon_beta1_scale returns exactly 1.0
        # and muon_beta2 returns exactly MUON_BETA2 when the respective feature is off, so the
        # assignments below are then provably the identity on the default path.
        demon_b1 = demon_beta1_scale(wd_progress)
        muon_b2 = muon_beta2(wd_progress)
        for group in optimizer.param_groups:
            # T2.1: batch_lr_scale is 1.0 unless BATCH_RAMP is on, so this multiplication is the
            # identity on the default path. It costs nothing even when it is not: group["lr"] is
            # already handed to _device_scalar, which writes whatever float it is given into a
            # persistent device tensor, so the graph never sees the value.
            group["lr"] = group["initial_lr"] * batch_lr_scale * (lrm_flat if group.get("no_lr_decay") else lrm)
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum(step, wd_progress)
                group["weight_decay"] = WEIGHT_DECAY * (1.0 - wd_progress) if WD_DECAY_TO_ZERO else WEIGHT_DECAY
                group["beta2"] = muon_b2
            elif group["kind"] == "adamw" and DEMON_BETA1:
                # Plan §11.1 item 1. Relative to the group's INITIAL beta1, so the 0.8 embedding
                # family lands on DEMON_BETA1_FINAL and the 0.9 groups move proportionally. The
                # rmsprop group is excluded by this `kind` test, which is correct rather than
                # incidental: it has no first moment for beta1 to decay.
                _b1, _b2 = group["initial_betas"]
                group["betas"] = (_b1 * demon_b1, _b2)
        if not skip_step:
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
        step_cap_hit = step + 1 >= args.num_steps
        time_budget_hit = reached_min_steps and (
            elapsed_after_step >= args.max_train_seconds
            or elapsed_after_step
            + max(recent_step_times[-20:])
            + CHECKPOINT_RESERVE_SECONDS
            >= args.max_train_seconds
        )
        stop_after_step = step_cap_hit or time_budget_hit
        if step_cap_hit and not time_budget_hit and rank == 0:
            # Defense-in-depth for the exact 2026-09-10 failure mode (NUM_STEPS default of 1000
            # silently capped the bare-launch submission at step 1000, ~815s into an 1800s budget,
            # before lr_multiplier's wall-clock cooldown ever ran -- see
            # research/submission-gap-analysis-2026-09-10.md, fixed in ba37f86 by making NUM_STEPS
            # unreachable). NUM_STEPS is unreachable today, so this only fires if some future edit
            # reintroduces a step cap (a lowered NUM_STEPS, a stray --num-steps on a real launch,
            # or a new bound entirely) that binds before the clock does.
            #
            # Banded on elapsed/max_train_seconds so it does NOT fire on this week's intentional
            # short --num-steps runs (research/scratch/*.sh): those stay well under
            # STEP_CAP_GUARD_MIN_PROGRESS (fixed-step probes are ~120/1800=6.7%, calibration sweeps
            # top out around 300/1500=20%). A step cap that instead binds deep into a run that was
            # clearly aiming for its full max_train_seconds budget -- exactly what happened on
            # 2026-09-10 -- lands above the band and is not a normal on-time exit (that's
            # time_budget_hit above), so it is the specific case worth a loud warning.
            progress_at_exit = elapsed_after_step / max(1e-6, args.max_train_seconds)
            if STEP_CAP_GUARD_MIN_PROGRESS <= progress_at_exit < STEP_CAP_GUARD_MAX_PROGRESS:
                print(
                    f"WARNING: --num-steps {args.num_steps} step cap stopped training at step "
                    f"{step + 1}, elapsed={elapsed_after_step:.0f}s -- only "
                    f"{progress_at_exit:.1%} of max_train_seconds={args.max_train_seconds:.0f}s. "
                    f"The wall-clock LR cooldown (lr_multiplier) did NOT run to completion. This "
                    f"is the 2026-09-10 submission-gap failure mode "
                    f"(research/submission-gap-analysis-2026-09-10.md) -- if this is meant to be "
                    f"a real/scored run, raise --num-steps (default {NUM_STEPS}); if this is an "
                    f"intentional short/smoke run, ignore.",
                    flush=True,
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
        print(f"transposed_linear: {USE_TRANSPOSED_LINEAR}")
        # USE_NKI_MUON, not args.nki_muon: --compile-muon supersedes it (see main()), and the dump
        # should say which Muon step actually ran rather than which one was requested.
        print(f"nki_muon:          {USE_NKI_MUON}")
        print(f"compile_muon:      {USE_COMPILE_MUON}")
        print(f"muon_lr_fp32:      {USE_MUON_LR_FP32}")
        # T3.1. Printed as one line so the experiment log can tell two n-gram arms apart
        # (SUMMARY_KEYS in research/run_experiment.py has the matching entry).
        print(f"ngram_ve:          {args.ngram_ve} tables={len(orig_model.ngram_embeds)} "
              f"rows={orig_model.ngram_rows} dim={orig_model.ngram_dim} "
              f"index_gather={NGRAM_VE_INDEX_GATHER} "
              f"trigram={args.ngram_ve_trigram} opt={args.ngram_ve_opt} "
              f"flat_lr={args.ngram_ve_flat_lr} bf16={args.ngram_ve_bf16} "
              f"rms_fp32_state_math={NGRAM_RMS_FP32} nki_rms={USE_NKI_NGRAM_RMS}")
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
        # Plan §11.1 items 1-3, same one-line convention: every knob of the three schedule tricks
        # on one line so two arms are distinguishable in experiment-log.jsonl without re-reading
        # the run log (SUMMARY_KEYS in research/run_experiment.py has the matching entry).
        print(f"sched_tricks:      demon_b1={DEMON_BETA1} demon_final={DEMON_BETA1_FINAL} "
              f"demon_ref={DEMON_BETA1_REF} demon_start={DEMON_BETA1_START_FRAC} "
              f"muon_b2={MUON_BETA2} muon_b2_final={MUON_BETA2_FINAL} "
              f"depth_lr={MUON_DEPTH_LR_BOTTOM}->{MUON_DEPTH_LR_TOP} "
              f"depth_mom={MUON_DEPTH_MOM_BOTTOM}->{MUON_DEPTH_MOM_TOP} "
              f"depth_mom_ref={MUON_DEPTH_MOM_REF}")
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
