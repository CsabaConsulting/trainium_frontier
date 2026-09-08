#!/usr/bin/env bash
# Lever 1, escalation: the two most interesting Mamba2 configs at a REAL 900s
# wall-clock budget, against the verified transformer baseline
# (tensorlr-900s: 995 steps / public_val_bpb 1.094327 / 55.7M params).
#
# Fixed-300-step results that motivate these two (all eager + NKI SSD kernel,
# 4 ranks, lnc=2; xf-fixed300 reference 0.907 s/step / 1.269458 / 55.7M):
#   g0 e2 n128 k4 (paper)  1.509 s/step  45.0M  1.273513   <- best quality/step
#   g1 e1 n128 k4          1.267 s/step  38.8M  1.290404
#   g2 e1  n64 k4          1.257 s/step  38.4M  1.293135
#   g3 e1  n64 k2          1.110 s/step  38.4M  1.288748   <- fastest
#
# g0 and g3 bracket the speed/quality trade, so they bracket the 900s answer too.
set -u
COMMON="--mamba2 --mamba2-pattern MMAMMM --nki-ssd --no-compile"

go() {
  local label="$1"; local desc="$2"; shift 2
  echo "######## $label ########"
  python research/run_experiment.py --label "$label" --solution Mamba2Lever1 \
    --description "$desc" --num-steps 5000 --max-train-seconds 900 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args $COMMON "$@"
}

go m2-g3-e1n64k2-900s "Lever 1 at the real 900s budget: cheapest Mamba2 geometry (expand=1 d_state=64 conv_kernel=2, 1.110 s/step vs the paper geometry's 1.509), eager + NKI SSD kernel. Baseline to beat: tensorlr-900s 995 steps / 1.094327"

go m2-g0-e2n128k4-900s "Lever 1 at the real 900s budget: paper Mamba2 geometry with the NKI scan VERIFIABLY active (1.509 s/step; the earlier m2hybrid-nki-* runs silently fell back to the 2.00 s/step eager scan). Best quality-per-step of the family. Baseline to beat: tensorlr-900s 995 steps / 1.094327"

echo
echo "===== summary ====="
python - <<'PY'
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("research/experiment-log.jsonl").read_text().splitlines() if l.strip()]
for r in rows[-2:]:
    print(f"{r.get('label'):24s} steps={r.get('num_steps')} bpb={r.get('public_val_bpb')} "
          f"params={r.get('num_params_M')}M train_s={r.get('training_seconds')} status={r.get('status')}")
PY
