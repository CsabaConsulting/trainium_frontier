#!/usr/bin/env bash
# Lever 1: quality-per-step of cheaper Mamba2 mixer configs, at a FIXED 300 steps
# so quality and speed are separated (the same control design as the existing
# m2hybrid-nki-fixed300 / xf-fixed300 pair, xf reference = 1.269458 bpb).
#
# All runs are eager (--no-compile) with the NKI SSD kernel ON, because the 2x2
# isolate measured that as the fastest Mamba2 configuration by a wide margin:
#   m2 eager + NKI 1.53 | m2 eager + eager-scan 2.00 | m2 compile + eager-scan 2.20
#   | m2 compile + NKI 4.71   (xf compile 0.92, xf eager 0.97)
#
# Run on the box from the repo root:
#   bash research/scratch/mamba2_lever1_sweep.sh
set -u
COMMON="--mamba2 --mamba2-pattern MMAMMM --nki-ssd --no-compile"

go() {
  local label="$1"; local desc="$2"; shift 2
  echo "######## $label ########"
  python research/run_experiment.py --label "$label" --solution Mamba2Lever1 \
    --description "$desc" --num-steps 300 --max-train-seconds 1500 \
    --nproc-per-node 4 --lnc 2 --no-commit --extra-args $COMMON "$@"
}

# 1. Corrected baseline: the paper-default geometry with the kernel VERIFIABLY on
#    (the recorded m2hybrid-nki-fixed300 ran at 1.98 s/step = the eager-scan
#    speed, so its 1.270639 bpb is a valid quality point but its throughput was
#    not the kernel's).
go m2-g0-e2n128k4-300 "Lever 1 baseline re-measure: paper geometry expand=2 d_state=128 head_dim=128 k=4, eager + NKI SSD kernel, kernel path now announced by ssd_scan"

# 2. expand=1 only -- the single biggest cheap lever (24% off mixer fwd+bwd)
go m2-g1-e1n128k4-300 "Lever 1: expand=1 (d_inner 1280->640, H 10->5), everything else paper default" \
  --mamba2-expand 1

# 3. expand=1 + d_state=64
go m2-g2-e1n64k4-300 "Lever 1: expand=1 + d_state=64 (mixer fwd+bwd 0.707x of baseline in the microbenchmark)" \
  --mamba2-expand 1 --mamba2-d-state 64

# 4. cheapest measured: expand=1 + d_state=64 + conv_kernel=2
go m2-g3-e1n64k2-300 "Lever 1: cheapest measured config -- expand=1 + d_state=64 + conv_kernel=2 (0.567x mixer fwd+bwd); the k-shift conv1d was the largest single forward component at k=4" \
  --mamba2-expand 1 --mamba2-d-state 64 --mamba2-conv-kernel 2

echo
echo "===== summary ====="
python - <<'PY'
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("research/experiment-log.jsonl").read_text().splitlines() if l.strip()]
for r in rows[-4:]:
    print(f"{r.get('label'):24s} steps={r.get('num_steps')} bpb={r.get('public_val_bpb')} "
          f"params={r.get('num_params_M')}M train_s={r.get('training_seconds')} status={r.get('status')}")
PY
