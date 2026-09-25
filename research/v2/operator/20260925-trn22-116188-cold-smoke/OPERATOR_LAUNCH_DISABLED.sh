#!/usr/bin/env bash
# One-use Operator procedure. LOCAL SYNTAX/PACKET PREPARATION ONLY until the
# exact controller assignment is independently reviewed and enabled.
set -euo pipefail

if [[ "${1:-}" != "--reviewed-one-use-run" || "$#" -ne 1 ]]; then
  echo "HOLD: pass --reviewed-one-use-run only after the Supervisor confirms the exact assignment" >&2
  exit 77
fi

repo=/home/poojith-devan/Desktop/Trainium
assignment="$repo/research/v2/assignments/20260925-trn22-116188-cold-compile-smoke-r1.json"
source_file="$repo/submissions/dense6x1024_direct_review_matrixlr015/train.py"
key=/home/poojith-devan/Desktop/Trainium/trn2.pem
host=ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com
root=/home/ubuntu/trainium_116188_cold_smoke_20260925_trn22_r1
trusted=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1
expected_source=116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d
expected_prepare=5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb

python3 - "$assignment" "$expected_source" "$root" <<'PY'
import json, pathlib, sys
p, source_sha, root = sys.argv[1:]
a = json.loads(pathlib.Path(p).read_text())
assert a['schema'] == 'trainium-one-use-supervisor-assignment-v1'
assert a['scope'] == 'TRN22_KNOWN_GOOD_116188_TRUE_COLD_120_SECOND_COMPILE_SMOKE_DIAGNOSTIC_ONLY'
assert a['label'] == 'fs-20260925-116188-cold-compile-smoke-trn22-r1'
assert a['host'] == 'trn22' and a['run_root'] == root
assert a['train_sha256'] == source_sha
assert a['step_log_path'] == root + '/receipts/step-boundaries-rank0.log'
assert a['assignment_status'] == 'ASSIGNED' and a['launch_authorized'] is True
assert a['submission_authorized'] is False
g2 = json.loads((pathlib.Path(p).parents[3] / a['cpu_g2_base_pass']).read_text())
assert g2['verdict'] == 'PASS' and g2['nproc'] == 4
assert a['training_command'] == ('NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone '
    '--nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift '
    '--compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 '
    '--clip-after-reduce --seq-len 1024 --no-eval-public --max-train-seconds 120')
PY

[[ $(sha256sum "$source_file" | awk '{print $1}') == "$expected_source" ]]
[[ ! -e "$repo/research/agentic/STOP" && ! -e "$repo/research/v2/STOP" ]]
ssh_opts=(-i "$key" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10)

# Read-only fresh preflight; an A0 eval or any device/lock holder is a HOLD.
ssh "${ssh_opts[@]}" "$host" /bin/bash -s -- "$root" "$trusted" "$expected_prepare" <<'REMOTE_PREFLIGHT'
set -euo pipefail
root=$1; trusted=$2; expected_prepare=$3
[[ ! -e "$root" ]]
[[ -f "$trusted/out/fs-20260925-v2-c5c-a0-control-trn22-r10-manual-source-r1/driver.log" ]]
grep -q '^EVALUATED_' "$trusted/out/fs-20260925-v2-c5c-a0-control-trn22-r10-manual-source-r1/driver.log"
[[ $(sha256sum "$trusted/prepare.py" | awk '{print $1}') == "$expected_prepare" ]]
[[ ! -e "$trusted/research/agentic/STOP" && ! -e "$trusted/research/v2/STOP" ]]
[[ -e /dev/neuron0 ]]
if ps -eo comm=,args= | awk '$1 ~ /^(python|python3|torchrun)$/ && $0 ~ /(train.py|torchrun|prepare.py|submission_rehearsal.py)/ {print; found=1} END {exit found?0:1}'; then
  echo 'HOLD: training/eval process still active' >&2; exit 81
fi
if fuser -s /dev/neuron0; then echo 'HOLD: Neuron device holder' >&2; exit 82; fi
if ! flock -n /home/ubuntu/.cache/trainium_freshstart/device0.lock /bin/true; then echo 'HOLD: host device lock held' >&2; exit 83; fi
free=$(df -B1 / | awk 'NR==2 {print $4}')
[[ "$free" =~ ^[0-9]+$ && "$free" -ge 6979321856 ]]
for suffix in 00000 00001 00002 00003 00004 06542; do
  [[ -s "/home/ubuntu/.cache/neuron_competition_r1/data/shard_${suffix}.parquet" ]]
done
[[ -s /home/ubuntu/.cache/neuron_competition_r1/tokenizer/tokenizer.pkl ]]
[[ -s /home/ubuntu/.cache/neuron_competition_r1/tokenizer/token_bytes.pt ]]
[[ $(sha256sum /home/ubuntu/.cache/neuron_competition_r1/tokenizer/tokenizer.pkl | awk '{print $1}') == aacd6c6a838eaec2c662a7d0cba0b5412f930a6ddee95b9ecb2c8fac53f3dafc ]]
[[ $(sha256sum /home/ubuntu/.cache/neuron_competition_r1/tokenizer/token_bytes.pt | awk '{print $1}') == 3de03cf6f8e16a9397fecd0cebd04f435f11994b215a5c785787fcc15adb0e15 ]]
date -u '+PREFLIGHT_%Y-%m-%dT%H:%M:%SZ'
printf 'free_bytes=%s\n' "$free"
REMOTE_PREFLIGHT

# The root and seven cache paths are absent until this precise one-use stage.
ssh "${ssh_opts[@]}" "$host" /bin/bash -s -- "$root" <<'REMOTE_STAGE'
set -euo pipefail
root=$1
[[ ! -e "$root" ]]
umask 077
mkdir -m 700 "$root"
mkdir -m 700 "$root/cache" "$root/receipts"
for name in persistent compiler neff neff_local hlo debug inductor; do
  mkdir -m 700 "$root/cache/$name"
  [[ -z $(find "$root/cache/$name" -mindepth 1 -print -quit) ]]
done
[[ ! -e "$root/receipts/step-boundaries-rank0.log" ]]
set -C
: > "$root/receipts/step-boundaries-rank0.log"
set +C
chmod 600 "$root/receipts/step-boundaries-rank0.log"
[[ ! -s "$root/receipts/step-boundaries-rank0.log" ]]
stat -c 'step_log_inode=%i mode=%a bytes=%s' "$root/receipts/step-boundaries-rank0.log"
date -u '+CACHE_STAGE_%Y-%m-%dT%H:%M:%SZ'
REMOTE_STAGE

scp "${ssh_opts[@]}" "$source_file" "$host:$root/train.py"

# Recheck identity, imported evaluator, source, cache emptiness, device/lock,
# then launch exact one-use smoke. No public eval occurs for this diagnostic.
ssh "${ssh_opts[@]}" "$host" /bin/bash -s -- "$root" "$trusted" "$expected_source" "$expected_prepare" <<'REMOTE_LAUNCH'
set -euo pipefail
root=$1; trusted=$2; expected_source=$3; expected_prepare=$4
[[ $(sha256sum "$root/train.py" | awk '{print $1}') == "$expected_source" ]]
[[ $(sha256sum "$trusted/prepare.py" | awk '{print $1}') == "$expected_prepare" ]]
[[ ! -e "$trusted/research/agentic/STOP" && ! -e "$trusted/research/v2/STOP" ]]
[[ ! -s "$root/receipts/step-boundaries-rank0.log" ]]
for name in persistent compiler neff neff_local hlo debug inductor; do
  [[ -z $(find "$root/cache/$name" -mindepth 1 -print -quit) ]]
done
[[ $(df -B1 / | awk 'NR==2 {print $4}') -ge 6979321856 ]]
if ps -eo comm=,args= | awk '$1 ~ /^(python|python3|torchrun)$/ && $0 ~ /(train.py|torchrun|prepare.py|submission_rehearsal.py)/ {print; found=1} END {exit found?0:1}'; then
  echo 'HOLD: training/eval process at final seam' >&2; exit 81
fi
if fuser -s /dev/neuron0; then echo 'HOLD: Neuron device holder at final seam' >&2; exit 82; fi
if ! flock -n /home/ubuntu/.cache/trainium_freshstart/device0.lock /bin/true; then echo 'HOLD: host lock at final seam' >&2; exit 83; fi
cd "$root"
env -i HOME=/home/ubuntu PATH=/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin PYTHONPATH="$trusted" \
  /home/ubuntu/workspace/native_venv/bin/python -c 'import hashlib,pathlib,prepare,sys; p=pathlib.Path(prepare.__file__).resolve(); h=hashlib.sha256(p.read_bytes()).hexdigest(); print("imported_prepare",p,h); sys.exit(0 if p==pathlib.Path(sys.argv[1]).resolve() and h==sys.argv[2] else 1)' "$trusted/prepare.py" "$expected_prepare"
nohup /bin/bash -c '
  set +e
  root=$1; trusted=$2
  cd "$root" || exit 96
  date -u "+TRAIN_STARTED_%Y-%m-%dT%H:%M:%SZ"
  /usr/bin/flock -n /home/ubuntu/.cache/trainium_freshstart/device0.lock \
    /usr/bin/timeout -k 60s 2100s /usr/bin/env -i \
      HOME=/home/ubuntu \
      PATH=/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin \
      PYTHONPATH="$trusted" PYTHONUNBUFFERED=1 NKI_VENV_PATH=/home/ubuntu/workspace/native_venv \
      NEURON_LOGICAL_NC_CONFIG=2 \
      NEURON_COMPETITION_R1_STEP_LOG="$root/receipts/step-boundaries-rank0.log" \
      NEURON_COMPILE_CACHE_URL="$root/cache/persistent" \
      NEURON_COMPILE_CACHE_DIR="$root/cache/compiler" \
      TORCH_NEURONX_NEFF_CACHE_DIR="$root/cache/neff" \
      TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR="$root/cache/neff_local" \
      TORCH_NEURONX_HLO_CACHE_DIR="$root/cache/hlo" \
      TORCH_NEURONX_DEBUG_DIR="$root/cache/debug" \
      TORCHINDUCTOR_CACHE_DIR="$root/cache/inductor" \
      /home/ubuntu/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=4 train.py \
        --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv \
        --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 \
        --no-eval-public --max-train-seconds 120 > "$root/train.log" 2>&1
  rc=$?
  /usr/bin/printf "%s\n" "$rc" > "$root/receipts/train.exitcode"
  date -u "+TRAIN_FINISHED_%Y-%m-%dT%H:%M:%SZ"
  exit "$rc"
' _ "$root" "$trusted" > "$root/driver.log" 2>&1 < /dev/null &
pid=$!
sleep 3
kill -0 "$pid"
date -u '+LAUNCHED_%Y-%m-%dT%H:%M:%SZ'
printf 'driver_pid=%s\n' "$pid"
REMOTE_LAUNCH
