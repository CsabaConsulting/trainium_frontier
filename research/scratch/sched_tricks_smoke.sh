#!/usr/bin/env bash
# Device smoke + THROUGHPUT/RECOMPILE probe for plan §11.1's three schedule tricks (Demon beta1,
# NorMuon beta2 ramp, per-depth Muon LR/momentum), before any 900s device time is spent.
#
# WHY a probe and not just a smoke: the CPU harness (research/scratch/sched_tricks_equiv.py) proves
# the MATH. It cannot prove (a) that the compiled Muon body still builds when `1-beta2` becomes a
# device tensor and when lr/momentum become [K,1,1] instead of 0-dim operands, or (b) the one thing
# these three items live or die on -- that none of them turns a per-step-changing value into a
# per-step neuronx-cc RECOMPILE. That failure mode does not error; it just takes dt from ~0.86s to
# 2-4s for the rest of the run, which is how USE_TENSOR_LR_SCALARS was found in the first place.
#
# The budget is deliberately COMPRESSED (--max-train-seconds 180 with a 200-step cap) so that
# elapsed/max_train_seconds sweeps ~0 -> ~1 inside the probe. At the real 1800s budget a 200-step
# probe would only reach progress ~0.1, i.e. before the Demon anneal even starts (it begins at
# 1 - WARMDOWN_RATIO = 0.25) and with the beta2 ramp barely moved -- it would measure nothing.
#
# COLD-CACHE CAVEAT, learned the hard way on the first pass: with an empty neuronx-cc cache step 0
# costs ~206 s and step 1 ~161 s, both of which land in `recent_step_times`. train.py's predictive
# stop is `elapsed + max(recent_step_times[-20:]) + CHECKPOINT_RESERVE >= max_train_seconds`, so on a
# 180 s budget the arm halts at 5 steps with nothing to measure. Run any COLD arm second, or re-run
# it once the cache is warm -- the compressed budget and a cold cache are incompatible by
# construction, and the symptom is a num_steps of 5, not an error.
#
# PASS for every arm: run completes, causality_passed True and causality_max_abs 0.000000, no
# Traceback, num_params unchanged at 592,916,844 (all three items are parameter-neutral by
# construction), exactly ONE "muon step: COMPILED body" line per Muon shape group (a per-step
# recompile shows up as dt, not as extra lines, but a *specialisation explosion* shows up here),
# and dt flat over steps 40-190 at the control's value.
set -u
cd "$(dirname "$0")/../.." 2>/dev/null || cd ~/trainium_frontier

run() {
  local name="$1"; shift
  echo "=== $(date -u +%H:%M:%S) $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps 200 --max-train-seconds 180 --out-dir "out/schedsmoke-$name" \
    --eval-tokens 262144 --eval-seq-len 1024 "$@" \
    > "research/run-logs/schedsmoke-$name.log" 2>&1
  local log="research/run-logs/schedsmoke-$name.log"
  grep -E "num_params=|sched_tricks:|causality_passed|causality_max_abs|public_val_bpb:|num_steps:" \
    "$log" | head -8
  echo -n "   COMPILED body lines: "; grep -c "muon step: COMPILED body" "$log"
  # dt over the window where the schedules are actually moving. A recompile-per-step regression is
  # a 2-4x jump here, not a subtle one.
  # $11 is the dt value ($10 is the literal "dt" label) -- checked against a real line:
  # "step 00080 | loss 4.3165 | lrm 0.872 | dt 0.84s | tok/s 155,557 | ..."
  awk '/^step /{v=$11; gsub("s","",v); s=$2+0; if (s>=40 && s<=190){n++; t+=v; if(v>mx)mx=v}}
       END{if(n)printf "   steps 40-190: n=%d  mean dt %.3fs  max dt %.3fs\n", n, t/n, mx}' "$log"
  grep -iE "Traceback|Error|out of memory|nan|RECOMPIL" "$log" | head -3
}

run ctrl
run demon      --demon-beta1
run demonfull  --demon-beta1 --demon-beta1-start-frac 0.0
run b2ramp     --muon-beta2-final 0.97
run depthlr    --muon-depth-lr-bottom 1.15 --muon-depth-lr-top 0.85
run depthmom   --muon-depth-mom-bottom 0.90 --muon-depth-mom-top 0.97
run depthboth  --muon-depth-lr-bottom 1.15 --muon-depth-lr-top 0.85 \
               --muon-depth-mom-bottom 0.90 --muon-depth-mom-top 0.97
run allthree   --demon-beta1 --muon-beta2-final 0.97 \
               --muon-depth-lr-bottom 1.15 --muon-depth-lr-top 0.85 \
               --muon-depth-mom-bottom 0.90 --muon-depth-mom-top 0.97
echo "=== $(date -u +%H:%M:%S) done ==="
