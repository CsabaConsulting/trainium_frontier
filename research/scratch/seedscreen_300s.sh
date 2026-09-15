#!/usr/bin/env bash
# Preliminary/exploratory WIDE seed screen, CHEAP short-budget proxy.
#
# Context: the 2026-09-15 3-seed variance check (seeds 42/43/44, full 1800s
# budget each) found real weight-init-driven public_val_bpb spread of
# max-min=0.006136 / sample std~=0.0032 on the current locked default config,
# bigger than this week's informal noise assumption. This script is NOT that
# check -- it does not draw a conclusion and adopts no seed. It is a WIDER
# (8 new seeds) but CHEAPER (fixed 300 steps, not full 1800s) screen, purely
# to characterize the SHAPE of the seed-noise distribution (fat tail? is 42
# an outlier?) before any full-budget confirmation is spent on it.
#
# --seed only varies torch.manual_seed at init_runtime (train.py:1335-1337),
# consumed once before model construction / before dist.init_process_group.
# This model has no nn.Dropout; every stochastic op is a weight-init call
# (normal_/uniform_/zeros_) drawing from the seeded global RNG -- so, exactly
# as in the 3-seed check, the entire measured spread below is weight-init
# noise alone, at byte-identical data order (USE_EPOCH_SHUFFLE=False).
#
# Fixed-step (not fixed-wall-clock) convention, same idea as the existing
# *_fixed300 / lever1 300-step probes in this directory (e.g.
# mamba2_lever1_sweep.sh, mamba2_lowm_cheap.sh): --num-steps 300 with a
# --max-train-seconds ceiling (3600s) that is never actually reached (300
# steps at the locked config's measured ~0.83 s/step is ~250s of training),
# so every run is stopped by the step cap, not the clock, and lands at the
# SAME lr-schedule position: elapsed/max_train_seconds ~7%, i.e. lrm=1.0 the
# whole probe, well before WARMDOWN_RATIO=0.75's cooldown kicks in. This
# isolates early-training loss variance across seeds from any
# budget/schedule-position confound -- it is explicitly NOT a full-budget
# bpb and must not be compared numerically to the 1800s numbers, only
# qualitatively (shape of spread).
#
# Seeds: 8 new values, all distinct from the already-measured 42/43/44 (and
# from 7, used elsewhere this week).
#
# Run on the box from the repo root:
#   bash research/scratch/seedscreen_300s.sh
set -u

go() {
  local seed="$1"
  local label="seedscreen-s${seed}-300"
  echo "######## ${label} ########"
  python research/run_experiment.py --label "$label" --solution seedscreen-prelim \
    --description "Preliminary/exploratory CHEAP short-budget seed screen (companion to the 2026-09-15 3-seed full-1800s variance check, seeds 42/43/44 -> max-min 0.006136 / sample std ~0.0032). Fixed 300 steps, locked default config otherwise, seed=${seed} only. NOT a full-budget bpb, no seed adopted." \
    --num-steps 300 --max-train-seconds 3600 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --seed "$seed"
}

for s in 1 2 3 5 8 13 21 100; do
  go "$s"
done

echo
echo "===== summary (seedscreen-prelim, fixed 300 steps) ====="
python - <<'PY'
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("research/experiment-log.jsonl").read_text().splitlines() if l.strip()]
rows = [r for r in rows if r.get("solution") == "seedscreen-prelim"]
for r in rows[-8:]:
    print(f"{r.get('label'):24s} steps={r.get('num_steps')} bpb={r.get('public_val_bpb')} "
          f"params={r.get('num_params_M')}M train_s={r.get('training_seconds')} status={r.get('status')}")
vals = [r.get("public_val_bpb") for r in rows[-8:] if r.get("public_val_bpb") is not None]
if vals:
    import statistics
    print(f"n={len(vals)} min={min(vals):.6f} max={max(vals):.6f} range={max(vals)-min(vals):.6f} "
          f"mean={statistics.mean(vals):.6f} pstdev={statistics.pstdev(vals):.6f} stdev={statistics.stdev(vals):.6f}")
PY
