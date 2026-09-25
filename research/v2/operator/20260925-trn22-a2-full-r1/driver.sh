#!/bin/bash
set -u
umask 077
ROOT=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1
PKG="$ROOT/out/fs-20260925-v2-c5c-a2-m64d128-trn22-r1-full-source"
WORK="$ROOT/out/fs-20260925-v2-c5c-a2-m64d128-trn22-r1-full-work"
PY=/home/ubuntu/workspace/native_venv/bin/python
HARNESS="$ROOT/research/scratch/submission_rehearsal.py"
exec 9>/home/ubuntu/.cache/trainium_freshstart/device0.lock
/usr/bin/flock -n 9 || { printf 'FAILED_DEVICE_LOCK\n'; exit 71; }
if test -e "$PKG/attempt.started"; then printf 'FAILED_ALREADY_STARTED\n'; exit 72; fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$PKG/attempt.started"
printf 'TRAIN_DRIVER_STARTED_'; date -u +%Y-%m-%dT%H:%M:%SZ
/usr/bin/timeout -k 60s 4200s "$PY" "$HARNESS" --package "$PKG" --sandbox "$WORK/train" --mode train > "$PKG/train.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$PKG/train.exitcode"
printf 'TRAIN_FINISHED_rc=%s_' "$rc"; date -u +%Y-%m-%dT%H:%M:%SZ
if test "$rc" -ne 0; then
  printf 'TRAIN_FAILED_LAST_LINES\n'
  tail -n 45 "$PKG/train.log"
  exit "$rc"
fi
printf 'EVAL_DRIVER_STARTED_'; date -u +%Y-%m-%dT%H:%M:%SZ
/usr/bin/timeout -k 60s 1800s "$PY" "$HARNESS" --package "$PKG" --sandbox "$WORK/eval" --mode eval --checkpoint "$WORK/train/out/final.pt" --eval-tokens 20971520 > "$PKG/eval.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$PKG/eval.exitcode"
printf 'EVAL_FINISHED_rc=%s_' "$rc"; date -u +%Y-%m-%dT%H:%M:%SZ
if test "$rc" -ne 0; then tail -n 45 "$PKG/eval.log"; fi
exit "$rc"
