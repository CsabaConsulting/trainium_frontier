#!/usr/bin/env bash
# Lever 1, part 2: how many Mamba2 layers, if any, actually help quality-per-step?
#
# WHY THIS IS NOT ALREADY ANSWERED. MAMBA2_PATTERN's comment calls "MMAMMM" (5 of
# 6 layers Mamba2) "the measured best of the hybrid patterns tried", but there is
# NOT ONE pattern-variant row in research/experiment-log.jsonl -- every Mamba2 row
# ever recorded is MMAMMM. So the mixer-count axis is untested, and it is the one
# axis the geometry sweep cannot speak to: that sweep shows 5/6 Mamba2 is worse
# than pure attention on quality-per-step, and says nothing at all about 1/6 or
# 2/6, which is the ratio the literature's hybrid claim is actually about.
#
# Two reasons a LOW Mamba2 fraction is the favourable regime:
#   1. Speed. A paper-geometry mixer costs ~0.093 s/step (fwd+bwd, 8 grad-accum
#      microsteps) against ~0.065 s for the attention layer it replaces, so each
#      swap is only ~+0.028 s. 1-2 mixers should land ~0.94-0.96 s/step against
#      the transformer's 0.907 -- nearly free, versus 1.51 for 5 mixers.
#   2. Quality. Mamba2 layers forfeit their value-embedding (no attention V to
#      inject one into), so MMAMMM throws away 5 of 6 value-embedding tables.
#      At 1-2 mixers almost all of them survive.
#
# Run at the PAPER geometry, which has the better quality-per-step of the two
# (1.273513 vs the cheap geometry's 1.288748) and whose speed penalty only
# matters when there are many mixers -- which is exactly what this sweep avoids.
#
# Fixed 300 steps so quality-per-step is isolated. Reference: xf-fixed300 =
# 0.907 s/step / 1.269458 bpb / 55.7M (i.e. the M=0 point of this very sweep).
#
# The decisive read is the SIGN OF THE TREND across M=1,2,3,5:
#   - if bpb is monotonically worse as M grows, the optimum is M=0 and the whole
#     architecture is dead on this task, no speed work can save it;
#   - if M=1 or M=2 beats 1.269458, there is a real win available, because the
#     speed cost at that ratio is small enough to keep nearly all the steps.
set -u
COMMON="--mamba2 --nki-ssd --no-compile"

go() {
  local label="$1"; local desc="$2"; shift 2
  echo "######## $label ########"
  echo "extra flags: [$*]"
  python research/run_experiment.py --label "$label" --solution Mamba2Pattern \
    --description "$desc" --num-steps 300 --max-train-seconds 1500 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args $COMMON "$@"
}

go m2-pat1-AMAAAA-300 "Pattern sweep at paper geometry: ONE Mamba2 layer of 6 (AMAAAA, layer 1). Nearly free on speed (~+0.028 s/step) and keeps 5 of 6 value-embedding tables. Reference xf-fixed300 = 1.269458 at 0.907 s/step" \
  --mamba2-pattern AMAAAA

go m2-pat2-AAMAAM-300 "Pattern sweep at paper geometry: TWO Mamba2 layers of 6 (AAMAAM)" \
  --mamba2-pattern AAMAAM

go m2-pat3-AMAMAM-300 "Pattern sweep at paper geometry: THREE Mamba2 layers of 6 (AMAMAM), the midpoint that fixes the sign of the quality-vs-mixer-count trend together with M=1,2 and the existing M=5 (MMAMMM, 1.273513)" \
  --mamba2-pattern AMAMAM

echo
echo "===== summary (M=0 reference: xf-fixed300 300 / 1.269458 / 55.7M / 272.1s) ====="
python - <<'PY'
import json, pathlib
rows=[json.loads(l) for l in pathlib.Path("research/experiment-log.jsonl").read_text().splitlines() if l.strip()]
for r in rows[-3:]:
    ts=r.get("training_seconds"); ns=r.get("num_steps")
    dt=f"{ts/ns:.3f}" if ts and ns else "NA"
    print(f"{r.get('label'):22s} steps={ns} bpb={r.get('public_val_bpb')} "
          f"params={r.get('num_params_M')}M train_s={ts} s/step={dt}")
PY
