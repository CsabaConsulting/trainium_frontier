#!/usr/bin/env bash
# Paired 900s spot-check for --ngram-ve-lr, the lr applied to the n-gram value-embedding
# tables (NGRAM_VE_LR, train.py:699, locked default 0.0 == sentinel for "use EMBEDDING_LR
# instead", currently EMBEDDING_LR=0.30, train.py:308). These tables are 537M of the model's
# 592.9M total params -- the dominant mass -- and have NEVER had their own lr tuned; they just
# silently inherit whatever the much smaller token/positional embedding tables use. Exact
# precedent: EMBED_ADAMW_BETA2 (another inherited hyperparameter on an embedding-style param
# group) was tested standalone for the first time this week and 0.95 beat the 0.995 default by
# +21 steps (+5.2%) and 0.009453 bpb (see research/trn2-experiment-results.md, "EMBED_ADAMW_BETA2"
# around line 493). A concurrent agent on box1 is testing the analogous NGRAM_VE_BETA2 axis
# (research/scratch/ngram_ve_beta2_900s.sh); this script tests the distinct NGRAM_VE_LR axis.
#
# This is a PURE hyperparameter change -- same param count, same architecture, same ops, only the
# optimizer's lr constant differs (NGRAM_VE_FLAT_LR=True means it is applied flat throughout, no
# warmdown schedule to complicate the comparison) -- so unlike the ngram-ve-layers/table-mult/dim
# axes (which changed tensor shapes or injection sites and needed step-matching via a
# calibration), no throughput difference is expected here. Same discipline as
# muon_lr_fp32_900s.sh / ngram_ve_beta2_900s.sh (also pure-hyperparameter changes with no new
# tensors): flat 900s for every arm, bracketed by a fresh same-commit control before and after,
# and THROUGHPUT IS VERIFIED RATHER THAN ASSUMED (per the task brief) -- if the arms don't land
# within the controls' own step-count spread, something is wrong and needs investigating before
# trusting the bpb numbers.
#
# Two arms bracketing the current implicit value meaningfully: half (0.15) and double (0.60) of
# the locked default (0.30, via the sentinel). Locked config otherwise: DEPTH=6, ASPECT_RATIO=96,
# --ngram-ve-layers 1,-1 (default), --ngram-ve-table-mult 128 (default), --ngram-ve-beta2 0.999
# (default, box1's axis, not touched here) -- none touched, only --ngram-ve-lr varies.
#
# Noise floor per task brief: |delta| < 0.001 bpb is noise; < 0.003 needs a same-day paired repeat
# before trusting; a real win >= noise floor earns a confirming 1800s pair before adoption.
set -u
cd "$(dirname "$0")/../.." || exit 1
run() {
  local label="$1"; local desc="$2"; shift 2
  echo "=== $(date -u +%H:%M:%S) $label: $* ==="
  python research/run_experiment.py --label "$label" --solution T3.1 \
    --description "$desc" --num-steps 5000 --max-train-seconds 900 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}

run ngmlr-ctrl-900s-r1 \
  "fresh same-commit 900s control for the --ngram-ve-lr isolation A/B, flag at locked default (lr sentinel 0.0 -> EMBEDDING_LR=0.30, layers 1,-1, mult 128, beta2 0.999)"

run ngmlr-arm-015-900s \
  "--ngram-ve-lr 0.15 (half the locked implicit 0.30; pure hyperparameter change, no new tensors, no throughput effect expected)" \
  --ngram-ve-lr 0.15

run ngmlr-arm-060-900s \
  "--ngram-ve-lr 0.60 (double the locked implicit 0.30; pure hyperparameter change, no new tensors, no throughput effect expected)" \
  --ngram-ve-lr 0.60

run ngmlr-ctrl-900s-r2 \
  "closing same-commit 900s control: measures this block's own drift/noise band"
echo "=== $(date -u +%H:%M:%S) done ==="
