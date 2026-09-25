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
