#!/usr/bin/env bash
# Track: throughput->capacity, Probe 3 steps 1 and 3 -- the plain block's step-time FLOOR, then the
# real marginal s/step cost of re-adding each feature ON THAT FLOOR.
#
# Why this is not just re-reading the old per-feature costs: every one of them was measured on a
# DIFFERENT stack (the rope kernel's ~15-18% was a launch-count win on the pre-NKI, pre-compile-muon
# eager stack), and T3.2's finding is that op count is the binding constraint -- which means feature
# costs are not additive and do not transfer across stacks. The point of arm `plain` is to find how
# much of the step is irreducible, i.e. how much a maximally-plain block could ever hand back.
#
# All arms carry the other box's +13.67% budget flags, because that is the current fast stack and
# there is no reason to measure a floor under a stack nobody would run.
#
# QUALITY IS EXPECTED TO BE WORSE IN EVERY ABLATED ARM. This script measures s/step ONLY (120-step
# fixed-step instrument, steady-state dt flat from ~step 5). The step-110 loss is printed as a sanity
# proxy, not as a verdict -- 120 steps is far too short to price a capacity change.
#
# NOTE on --no-ve: it transitively removes the n-gram value embeddings (has_ngram_bigram requires
# USE_VE and has_ve), so `plain` is ve-free and ngram-free by construction; --no-ngram-ve is passed
# anyway so the arm is explicit rather than relying on that coupling.
# NOTE on "lambdas fixed" (the doc's phrasing): resid_lambda/x0_lambda are 2 learned SCALARS per
# block, 12 tensors total, living in the AdamW scalar group. Freezing them cannot move s/step at
# this resolution and has no flag, so it is deliberately not an arm here.
set -u
cd "$(dirname "$0")/../.." || exit 1
STEPS=${STEPS:-120}
BUDGET=(--compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze)
PLAIN=(--no-local-conv --no-ve --no-ngram-ve --no-nki-rope-norm --no-nki-softcap-ce)
run() {
  local name="$1"; shift
  echo "=== $name : $* ==="
  NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py \
    --num-steps "$STEPS" --max-train-seconds 1800 --out-dir "out/tc-probe-$name" \
    --eval-seq-len 1024 "${BUDGET[@]}" "$@" > "research/run-logs/tc-probe-$name.log" 2>&1
  grep -E "num_params" "research/run-logs/tc-probe-$name.log" | head -2
  grep -E "^step 00(0[1-9]0|1[01]0) " "research/run-logs/tc-probe-$name.log" | tail -12
  grep -iE "Traceback|Error|out of memory" "research/run-logs/tc-probe-$name.log" | head -3
}
# --- step 1: the floor, and the two halves of it separately ---
run ab-full                                                    # = the budget control, re-run same-day
run ab-plain     "${PLAIN[@]}"                                 # maximally plain block
run ab-nokernels --no-nki-rope-norm --no-nki-softcap-ce        # kernels off, features on
run ab-nofeat    --no-local-conv --no-ve --no-ngram-ve         # features off, kernels on
# --- step 3: re-add one at a time ON THE PLAIN FLOOR ---
run ab-plain-p-rope   --no-local-conv --no-ve --no-ngram-ve --no-nki-softcap-ce
run ab-plain-p-ce     --no-local-conv --no-ve --no-ngram-ve --no-nki-rope-norm
run ab-plain-p-conv   --no-ve --no-ngram-ve --no-nki-rope-norm --no-nki-softcap-ce
run ab-plain-p-ve     --no-local-conv --no-ngram-ve --no-nki-rope-norm --no-nki-softcap-ce
run ab-plain-p-vengram --no-local-conv --no-nki-rope-norm --no-nki-softcap-ce
echo "=== done ==="
