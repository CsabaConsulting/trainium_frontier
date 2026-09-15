#!/usr/bin/env bash
# RECOVERY block for the two plan-§11.1 per-depth arms the main 10-arm block never reached.
#
# Box 1 (ec2-16-50-61-34) lost network 7 arms into research/scratch/sched_tricks_900s.sh, with
# st-depthmom-900s in flight and st-depthboth-900s + the closing control unrun. Those three cannot
# be spliced in from another box: a 900 s number is only comparable INSIDE its own block, so this
# runs the two missing arms with their OWN bracketing control pair and is read entirely against
# those two controls -- never against box 1's numbers.
#
# Same four labels the report script already understands (st-*), so
# research/scratch/sched_tricks_report.py decomposes this block without changes. Distinct labels
# from the box-1 arms so the two blocks stay separable in experiment-log.jsonl.
set -u
cd "$(dirname "$0")/../.." 2>/dev/null || cd ~/trainium_frontier
run() {
  local label="$1"; local desc="$2"; shift 2
  echo "=== $(date -u +%H:%M:%S) $label : $* ==="
  python research/run_experiment.py --label "$label" --solution R1.2 \
    --description "$desc" --num-steps 5000 --max-train-seconds 900 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args --eval-seq-len 1024 "$@"
}

run st-ctrl-b2-r1 "box-2 recovery block: opening same-commit 900s control for the two per-depth Muon arms"

run st-depthmom-b2 \
  "plan 11.1 item 3b, per-depth Muon momentum only: target 0.90 at layer 0 -> 0.97 at layer 5, applied as a multiplier on (1 - momentum) at reference momentum 0.95 so it rides on top of the global warmup/cooldown schedule" \
  --muon-depth-mom-bottom 0.90 --muon-depth-mom-top 0.97

run st-depthboth-b2 \
  "plan 11.1 item 3, per-depth LR and momentum together, exactly as Recursive pairs them (LR x1.15->x0.85, momentum 0.90->0.97)" \
  --muon-depth-lr-bottom 1.15 --muon-depth-lr-top 0.85 \
  --muon-depth-mom-bottom 0.90 --muon-depth-mom-top 0.97

run st-ctrl-b2-r2 "box-2 recovery block: closing same-commit 900s control, i.e. this block's own noise band"
echo "=== $(date -u +%H:%M:%S) done ==="
