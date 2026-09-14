#!/usr/bin/env bash
# T1.6: does storing Linear weights transposed ([in, out], `x @ w`) actually buy throughput on the
# CURRENT locked config?
#
# WHY THIS HAS TO BE RE-MEASURED RATHER THAN ASSUMED. The plan's T1.6 premise -- "the compiled graph
# carries +75% transpose instructions vs eager and F.linear is 48% of the window" -- comes from a
# profile taken at **nproc=1 / LNC=1**, and the results doc's own note on that profile says the
# compiled step was measured SLOWER there than eager and that the finding "doesn't replicate the real
# submission's win", which is at 4 ranks / LNC=2. The same doc also records `aten::linear` /
# `linear_backward` already hitting **80.4% of peak TensorE FLOPS**, i.e. explicitly "nothing to fix
# there via kernel work". So the number motivating T1.6 was never measured on the stack that gets
# submitted. This project has also just been burned twice by exactly this: the rope+norm kernel
# measured +15-18% / +7.92% / -1.34% on three configurations of the same model, and two `local_conv`
# comments were stale in OPPOSITE directions. A kernel/layout number expires when the stack changes,
# and the sign can flip.
#
# INSTRUMENT: the 120-step screen used by Probe 3, whose s/step reproducibility was independently
# established at 0.27% (two accidentally-identical depth-8 arms). tok/s is the 8-point mean of the
# printed steps 40-110; step 0 carries the cold compile and MUST be excluded (including it
# contaminated the first pass of Probe 3's numbers).
#
# ARMS ARE INTERLEAVED (full, t16, full, t16, full, t16) rather than run back-to-back, so a slow
# drift in box state cannot masquerade as the effect -- same discipline as Probe 3's ab-full/ab-norope
# triples, where the control turned out to spread 0.74% while the treatment held to 0.06%.
#
# QUALITY IS NOT READ FROM THIS SCRIPT. --transposed-linear is bit-exact through forward and backward
# but NOT through Muon's Newton-Schulz (research/scratch/transposed_linear_equiv.py: step-0 loss and
# all gradients identical, post-optimizer params differ at ~1e-8 and up), because the NS iteration's
# two algebraically-equivalent forms `q(X X^T) @ X` and `X @ q(X^T X)` round differently and which
# one runs depends on the tensor's actual orientation. So the step-110 loss is printed for sanity
# (it must stay at the ~3.94 control level, i.e. nothing is catastrophically broken) but any real
# quality claim needs a paired 900s A/B, and that is only worth spending if the throughput lands.
#
# All arms carry the locked config EXPLICITLY rather than relying on defaults, so they survive the
# other box promoting something else mid-run (which happened twice during Probe 3).
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
BEST=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze
      --ngram-ve-layers "1,-1" --ngram-ve-table-mult 128)
run() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/t16-$name" \
    --eval-seq-len 1024 "${BEST[@]}" "$@" > "research/run-logs/t16-$name.log" 2>&1
  grep -E "num_params=" "research/run-logs/t16-$name.log" | head -1
  # Null-test guard: the arm's whole effect is invisible in the loss, so a flag that silently did
  # nothing would look exactly like a working one. This line MUST appear in every t16 arm and MUST
  # be absent from every control arm.
  grep -E "^T1.6: transposed" "research/run-logs/t16-$name.log" | head -1
  grep -E "^step 000[4-9]0 |^step 001[01]0 " "research/run-logs/t16-$name.log" | tail -8
  grep -iE "Traceback|Error|out of memory" "research/run-logs/t16-$name.log" | head -3
}
run t16-ctrl-r1
run t16-tr-r1    --transposed-linear
run t16-ctrl-r2
run t16-tr-r2    --transposed-linear
run t16-ctrl-r3
run t16-tr-r3    --transposed-linear
echo "=== done ==="
