#!/usr/bin/env bash
# Lever 1, part 3: the actual best candidate the first two sweeps imply.
#
# The pattern sweep found a real interior optimum in quality-per-step at ONE
# Mamba2 layer of 6 (300 fixed steps, paper geometry):
#   M=0 1.269458 @0.907 s/step | M=1 1.267050 @1.080 | M=2 1.268748 @1.201
#   M=3 1.282111 @1.250        | M=5 1.273513 @1.509
# So a single mixer genuinely helps quality-per-step (-0.0024 bpb vs pure
# attention) -- it just cost +0.173 s/step at the paper geometry, and the
# break-even price of a mixer at this budget is only 0.077*ln(1.080/0.907) =
# 0.0134 bpb, 5.6x more than it earns. Wrong geometry, not a wrong idea.
#
# Solving total = base + n_attn*a + n_m*m across all five points above plus the
# geometry sweep prices a mixer as (s/step over the attention layer it replaces):
#   paper geometry  +0.114 .. +0.173   (M=3, M=5, M=1 estimates)
#   cheap geometry  +0.041             ((1.110 - 0.907) / 5, from m2-g3-*-300)
# i.e. the cheap geometry makes a mixer ~3-4x cheaper to add. Meanwhile the cheap
# geometry only cost ~0.003 bpb PER MIXER in quality (its +0.0152 at M=5), so at
# M=1 the trade should be nearly free -- predicted ~0.95 s/step, ~950 of the
# transformer's 1004 steps, and a step penalty of only 0.077*ln(1004/950) =
# 0.004 bpb. That is the first configuration in this whole line predicted to land
# near break-even rather than 0.03+ behind.
#
# Fixed 300 steps, so this measures the one unknown: what M=1 and M=2 cost in
# quality-per-step once the mixer is also cheap. Reference xf-fixed300 = 1.269458.
set -u
COMMON="--mamba2 --nki-ssd --no-compile --mamba2-expand 1 --mamba2-d-state 64 --mamba2-conv-kernel 2"

go() {
  local label="$1"; local desc="$2"; shift 2
  echo "######## $label ########"
  echo "extra flags: [$*]"
  python research/run_experiment.py --label "$label" --solution Mamba2LowM \
    --description "$desc" --num-steps 300 --max-train-seconds 1500 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args $COMMON "$@"
}

go m2-pat1-cheap-300 "Best candidate: ONE Mamba2 layer of 6 (AMAAAA) at the cheap geometry (expand=1 d_state=64 conv_kernel=2). A cheap mixer prices at only +0.041 s/step over the attention layer it replaces, vs +0.12-0.17 at the paper geometry, while M=1 paper already BEAT pure attention on quality-per-step (1.267050 vs 1.269458). Reference xf-fixed300 = 1.269458 @ 0.907 s/step" \
  --mamba2-pattern AMAAAA

go m2-pat2-cheap-300 "Same at TWO Mamba2 layers of 6 (AAMAAM), to see whether the second mixer still pays once mixers are cheap (at the paper geometry it did not: 1.268748 @1.201)" \
  --mamba2-pattern AAMAAM

echo
echo "===== summary (M=0 reference: xf-fixed300 300 / 1.269458 / 55.7M / 272.1s = 0.907 s/step) ====="
python - <<'PY'
import json, pathlib
rows=[json.loads(l) for l in pathlib.Path("research/experiment-log.jsonl").read_text().splitlines() if l.strip()]
for r in rows[-2:]:
    ts=r.get("training_seconds"); ns=r.get("num_steps")
    dt=ts/ns if ts and ns else None
    proj=""
    if dt:
        import math
        # predicted 900s bpb vs the verified lrmq25-900s baseline (1004 / 1.093785),
        # using the current-code fit bpb ~ a - 0.077*ln(steps)
        steps900 = 894.0 / dt
        delta_q = (r.get("public_val_bpb") or 0) - 1.269458
        pred = 1.093785 + 0.077*math.log(1004/steps900) + delta_q
        proj = f" -> 900s proj: {steps900:.0f} steps / {pred:.6f} (baseline 1.093785)"
    print(f"{r.get('label'):20s} steps={ns} bpb={r.get('public_val_bpb')} "
          f"params={r.get('num_params_M')}M s/step={dt:.3f}{proj}" if dt else r.get("label"))
PY
