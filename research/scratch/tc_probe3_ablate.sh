#!/usr/bin/env bash
# Track: throughput->capacity, Probe 3 steps 1 and 3 -- the plain block's step-time FLOOR, then the
# real marginal s/step cost of each feature ON THE CURRENT STACK.
#
# WHY THIS MATTERS NOW, concretely. Part B measured the exchange rate the whole track exists to find:
# width 768 is worth **-0.0055 bpb at equal steps** but costs 14.65% of the throughput, and the
# -0.077 bpb/ln(steps) slope prices that shortfall at +0.0125 -- so width loses by ~0.007 at the real
# 1800 s budget. Inverting the slope: 768x6 BREAKS EVEN at 1990 steps, i.e. at 0.9045 s/step against
# its measured 0.99075. **A 9.5% throughput gain on the 768 shape turns width from a loss into a
# win, and everything past 9.5% is profit.** This script asks the only remaining question: is 9.5%
# anywhere in the block? Probe 1 (XLA graph mode) and Probe 2 (launch count) both closed negative, so
# the block body is the last place left to look.
#
# Two kinds of arm, deliberately separated:
#   - `ab-nokernels`: the two NKI kernel islands off, model IDENTICAL. This is the clean measurement
#     of what hand-written kernels are still worth on the current stack, at constant capacity. It is
#     the arm that could plausibly hand back throughput, because the compiler might now fuse the
#     plain ops better than it could when the kernels were written (T1.7 changed the optimizer, the
#     promoted layer-1 n-gram changed the resharding pattern -- neither existed then).
#   - `ab-nofeat` / `ab-plain`: capacity removed as well, so they are NOT candidate configs and their
#     s/step is a FLOOR, not an offer. Quality is expected to be much worse. With the newly promoted
#     defaults `--no-ngram-ve` alone drops 402.8M of 592.9M parameters.
#
# NOTE on --no-ve: it transitively removes the n-gram value embeddings (has_ngram_bigram requires
# USE_VE and has_ve), so `ab-plain` is ngram-free by construction; --no-ngram-ve is passed anyway so
# the arm is explicit rather than relying on that coupling.
# NOTE on "lambdas fixed" (the track doc's phrasing): resid_lambda/x0_lambda are 2 learned SCALARS
# per block, 12 tensors total, in the AdamW scalar group. Freezing them cannot move s/step at this
# instrument's 0.27% resolution and has no flag, so it is deliberately not an arm.
#
# All arms run on the promoted defaults (n-gram VE layers "1,-1" at mult 128, compiled Muon, frozen
# qk-shift), passed explicitly so the arms survive a further defaults change on the other box.
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
BEST=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze
      --ngram-ve-layers "1,-1" --ngram-ve-table-mult 128)
run() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/tc-probe-$name" \
    --eval-seq-len 1024 "${BEST[@]}" "$@" > "research/run-logs/tc-probe-$name.log" 2>&1
  grep -E "num_params=" "research/run-logs/tc-probe-$name.log" | head -1
  grep -E "^step 000[4-9]0 |^step 001[01]0 " "research/run-logs/tc-probe-$name.log" | tail -8
  grep -iE "Traceback|Error|out of memory" "research/run-logs/tc-probe-$name.log" | head -3
}
# --- constant-capacity arms: the only ones that could yield an adoptable win ---
run ab-full                                                    # same-day control
run ab-nokernels  --no-nki-rope-norm --no-nki-softcap-ce
run ab-norope     --no-nki-rope-norm
run ab-noce       --no-nki-softcap-ce
# NOT a constant-capacity arm after all: --local-conv's help text claimed "provably dead compute,
# bit-identical", which was true before T0.1 but is now STALE -- TRAIN_LOCAL_CONV=True puts
# local_conv.weight in its own AdamW group, so this drops 7,680 TRAINED params. Left in the
# constant-capacity block as run, with the result read as a capacity ablation. Help text fixed.
run ab-noconv     --no-local-conv
# --- floor arms: capacity removed, diagnostic only ---
run ab-nofeat     --no-local-conv --no-ve --no-ngram-ve
run ab-plain      --no-local-conv --no-ve --no-ngram-ve --no-nki-rope-norm --no-nki-softcap-ce
echo "=== done ==="
