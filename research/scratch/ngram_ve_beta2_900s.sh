#!/usr/bin/env bash
# Paired 900s spot-check for --ngram-ve-beta2, the RMSProp second-moment decay for the n-gram
# value-embedding tables (NGRAM_VE_BETA2, train.py:700, locked default 0.999). Never tuned
# standalone for this shape -- inherited ("Recursive's choice"). Exact precedent: EMBED_ADAMW_BETA2
# (another inherited beta2 on an embedding-style param group) was tested standalone for the first
# time this week and 0.95 beat the 0.995 default by +21 steps (+5.2%) and 0.009453 bpb (see
# research/trn2-experiment-results.md, "EMBED_ADAMW_BETA2" around line 493). NGRAM_VE_BETA2 governs
# far more parameter mass (537M of 592.9M total) so the stakes are larger if mistuned.
#
# This is a PURE hyperparameter change -- same param count, same architecture, same ops, only the
# optimizer's beta2 constant differs -- so unlike the ngram-ve-layers/table-mult/dim axes (which
# changed tensor shapes or injection sites and needed step-matching via a calibration), no
# throughput difference is expected here. Same discipline as muon_lr_fp32_900s.sh (also a
# pure-hyperparameter/precision change with no new tensors): flat 900s for every arm, bracketed by
# a fresh same-commit control before and after, and THROUGHPUT IS VERIFIED RATHER THAN ASSUMED
# (per the task brief) -- if the arms don't land within the controls' own step-count spread,
# something is wrong and needs investigating before trusting the bpb numbers.
#
# Two most-informative arms, mirroring the EMBED_ADAMW_BETA2 precedent: 0.95 (its winning value)
# and 0.99 (a midpoint between 0.95 and the locked 0.999). Locked config otherwise: DEPTH=6,
# ASPECT_RATIO=96, --ngram-ve-layers 1,-1 (default), --ngram-ve-table-mult 128 (default) -- none
# touched, only --ngram-ve-beta2 varies.
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

run ngmbeta2-ctrl-900s-r1 \
  "fresh same-commit 900s control for the --ngram-ve-beta2 isolation A/B, flag at locked default (beta2=0.999, layers 1,-1, mult 128)"

run ngmbeta2-arm-095-900s \
  "--ngram-ve-beta2 0.95 (mirrors EMBED_ADAMW_BETA2's winning value; pure hyperparameter change, no new tensors, no throughput effect expected)" \
  --ngram-ve-beta2 0.95

run ngmbeta2-arm-099-900s \
  "--ngram-ve-beta2 0.99 (midpoint between 0.95 and the locked 0.999; pure hyperparameter change, no new tensors, no throughput effect expected)" \
  --ngram-ve-beta2 0.99

run ngmbeta2-ctrl-900s-r2 \
  "closing same-commit 900s control: measures this block's own drift/noise band"
echo "=== $(date -u +%H:%M:%S) done ==="
