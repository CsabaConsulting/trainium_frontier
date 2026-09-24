#!/usr/bin/env bash
# Operator-reviewed CPU-only G2 batch. This script does not train on Neuron.
# It must run from the exact private tmpfs staging checkout named below.
set -Eeuo pipefail
umask 077

readonly EXPECTED_HOST="ip-172-31-32-253"
readonly ROOT="/dev/shm/trainium-v2-g2-20260925-trn22-r1"
readonly PYTHON="/home/ubuntu/workspace/native_venv/bin/python"
readonly HOST_LOCK="/home/ubuntu/.cache/trainium_freshstart/device0.lock"
readonly BATCH_LABEL="fs-20260925-v2-g2-baseline-s0-trn22-r1"
readonly RUN_ROOT="${ROOT}/research/v2/g2-runs/${BATCH_LABEL}"
readonly CONTAINED="${ROOT}/workflow/tools/contained.py"
readonly HARNESS="${ROOT}/workflow/tools/preflight_dryrun.py"
readonly PREPARE="${ROOT}/prepare.py"
readonly WAIT_TIMEOUT_SECONDS=3660

readonly SHA_PREPARE="5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
readonly SHA_CONTAINED="0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d"
readonly SHA_HARNESS="6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8"
readonly SHA_35C9="35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2"
readonly SHA_RESEARCH_V2="c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
readonly SHA_116188="116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d"
readonly SHA_S0="ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb"

readonly LABEL_35C9="fs-20260925-g2-a0-35c9-trn22-r1"
readonly LABEL_RESEARCH_V2="fs-20260925-g2-a0-research-v2-trn22-r1"
readonly LABEL_116188="fs-20260925-g2-s0-116188-trn22-r1"
readonly LABEL_S0="fs-20260925-g2-s0-bare-trn22-r1"

readonly TRAIN_35C9="${ROOT}/04-train-research-35c9.py"
readonly TRAIN_RESEARCH_V2="${ROOT}/submissions/research_v2/train.py"
readonly TRAIN_116188="${ROOT}/05-train-submission-clean-116188.py"
readonly TRAIN_S0="${ROOT}/submissions/candidates/S0/train.py"

readonly -a FLAGS_35C9=(
  --n-embd 1024
  --no-ngram-ve
  --no-qk-shift
  --compile-sdpa-direct
  --nki-local-conv
  --bf16-norm-output
  --pack-factor 32
  --clip-after-reduce
  --seq-len 1024
  --no-eval-public
  --matrix-lr 0.015
)
readonly -a FLAGS_116188=(
  --n-embd 1024
  --no-ngram-ve
  --no-qk-shift
  --compile-sdpa-direct
  --nki-local-conv
  --bf16-norm-output
  --pack-factor 32
  --clip-after-reduce
  --seq-len 1024
  --no-eval-public
)

ACTIVE_LABEL=""
ACTIVE_DIR=""
ACTIVE_RECEIPT=""

die() {
  echo "G2 batch refused: $*" >&2
  exit 1
}

sha256_file() {
  /usr/bin/sha256sum -- "$1" | /usr/bin/awk '{print $1}'
}

require_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || die "not a regular non-symlink file: $path"
  [[ "$(sha256_file "$path")" == "$expected" ]] || die "SHA-256 mismatch: $path"
}

quiesce_active() {
  local -a files=("${ACTIVE_DIR}/run.log")
  if [[ -f "$ACTIVE_RECEIPT" ]]; then
    files+=("$ACTIVE_RECEIPT")
  fi
  "$PYTHON" "$CONTAINED" quiesce \
    --label "$ACTIVE_LABEL" --dir "$ACTIVE_DIR" --window 10 --files "${files[@]}"
}

stop_and_quiesce_active() {
  [[ -n "$ACTIVE_LABEL" ]] || return 0
  "$PYTHON" "$CONTAINED" stop --label "$ACTIVE_LABEL" --dir "$ACTIVE_DIR" --grace 30 || true
  quiesce_active
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if (( rc != 0 )) && [[ -n "$ACTIVE_LABEL" ]]; then
    stop_and_quiesce_active || echo "BREAKER: failed to quiesce $ACTIVE_LABEL" >&2
  fi
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

validate_receipt() {
  local receipt="$1" train="$2" expected_sha="$3"
  shift 3
  "$PYTHON" - "$receipt" "$train" "$expected_sha" "$SHA_PREPARE" "$ROOT" "$@" <<'PY'
import json
import pathlib
import sys

receipt_path, train_path, train_sha, prepare_sha, prepare_dir, *expected_argv = sys.argv[1:]
r = json.loads(pathlib.Path(receipt_path).read_text())
assert r.get("schema") == "g2-preflight-v1"
assert r.get("verdict") == "PASS"
assert r.get("train_py") == train_path
assert r.get("train_py_sha256") == train_sha
assert r.get("argv") == expected_argv
assert r.get("steps") == 3
assert r.get("nproc") == 4
assert r.get("micro_batch") == [2, 256]
assert r.get("timed_out") is False
assert r.get("torchrun_rc") == 0
assert r.get("prepare", {}).get("dir") == prepare_dir
assert r.get("prepare", {}).get("sha256") == prepare_sha
assert r.get("prepare", {}).get("stub") is False
assert len(r.get("ranks", [])) == 4
assert [x.get("rank") for x in r["ranks"]] == [0, 1, 2, 3]
assert all(x.get("status") == "PASS" for x in r["ranks"])
assert r["ranks"][0].get("checkpoint_check", {}).get("ok") is True
assert all(not x.get("cross_rank_mismatches") for x in r["ranks"])
PY
}

run_probe() {
  local label="$1" train="$2" expected_sha="$3"
  shift 3
  local -a train_argv=("$@")
  local dir="${RUN_ROOT}/processes/${label}"
  local receipt="${RUN_ROOT}/receipts/${label}.json"
  local tmpdir="${RUN_ROOT}/tmp/${label}"
  local wait_rc=0 status_rc=0

  [[ ! -e "$dir" && ! -L "$dir" ]] || die "process directory already exists: $dir"
  [[ ! -e "$receipt" && ! -L "$receipt" ]] || die "receipt already exists: $receipt"
  [[ ! -e "$tmpdir" && ! -L "$tmpdir" ]] || die "TMPDIR already exists: $tmpdir"
  /usr/bin/install -d -m 700 -- "$dir" "$tmpdir"

  ACTIVE_LABEL="$label"
  ACTIVE_DIR="$dir"
  ACTIVE_RECEIPT="$receipt"

  "$PYTHON" "$CONTAINED" launch --label "$label" --dir "$dir" --cwd "$ROOT" --cgroup -- \
    /usr/bin/env TMPDIR="$tmpdir" OMP_NUM_THREADS=1 \
    "$PYTHON" "$HARNESS" \
      --train-py "$train" --prepare-dir "$ROOT" --steps 3 --nproc 4 --timeout 3600 \
      --out "$receipt" --keep -- "${train_argv[@]}"

  "$PYTHON" "$CONTAINED" wait --label "$label" --dir "$dir" --timeout "$WAIT_TIMEOUT_SECONDS" || wait_rc=$?
  if (( wait_rc != 0 )); then
    stop_and_quiesce_active || die "timeout/error and quiescence failed for $label"
    ACTIVE_LABEL=""
    ACTIVE_DIR=""
    ACTIVE_RECEIPT=""
    die "contained wait failed for $label with rc=$wait_rc"
  fi

  "$PYTHON" "$CONTAINED" status --label "$label" >"${dir}/status-after-wait.json" || status_rc=$?
  if (( status_rc == 0 )); then
    stop_and_quiesce_active || die "survivor cleanup failed for $label"
    ACTIVE_LABEL=""
    ACTIVE_DIR=""
    ACTIVE_RECEIPT=""
    die "tagged descendants survived the leader for $label"
  fi
  (( status_rc == 1 )) || die "contained status failed for $label with rc=$status_rc"

  quiesce_active
  ACTIVE_LABEL=""
  ACTIVE_DIR=""
  ACTIVE_RECEIPT=""
  validate_receipt "$receipt" "$train" "$expected_sha" "${train_argv[@]}"
}

compare_receipts() {
  local name="$1" a="$2" b="$3" sha_a="$4" sha_b="$5"
  local out="${RUN_ROOT}/comparisons/${name}.json"
  local pending="${out}.pending"
  [[ ! -e "$out" && ! -L "$out" && ! -e "$pending" && ! -L "$pending" ]] || \
    die "comparison output already exists: $out"
  "$PYTHON" "$HARNESS" --compare "$a" "$b" >"$pending" || die "comparison failed: $name"
  "$PYTHON" - "$pending" "$sha_a" "$sha_b" <<'PY'
import json
import pathlib
import sys

p, sha_a, sha_b = sys.argv[1:]
r = json.loads(pathlib.Path(p).read_text())
assert r.get("a_sha256") == sha_a
assert r.get("b_sha256") == sha_b
assert r.get("plan_equal") is True
assert r.get("losses_identical") is True
assert r.get("state_keys_only_in_one") == []
assert r.get("param_digest_differences") == []
assert r.get("n_params_compared", 0) > 0
assert r.get("identical") is True
PY
  /usr/bin/mv -- "$pending" "$out"
}

[[ "$(/usr/bin/hostname -s)" == "$EXPECTED_HOST" ]] || die "wrong host"
[[ "$(/usr/bin/readlink -f -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")" == "$ROOT" ]] || \
  die "script is not executing from the exact staged repository"
[[ -x "$PYTHON" && -f "$PYTHON" ]] || die "native Python identity invalid"
[[ "$(git -C "$ROOT" status --porcelain --untracked-files=all)" == "" ]] || die "staged repository is dirty"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || die "batch output already exists: $RUN_ROOT"

require_hash "$PREPARE" "$SHA_PREPARE"
require_hash "$CONTAINED" "$SHA_CONTAINED"
require_hash "$HARNESS" "$SHA_HARNESS"
require_hash "$TRAIN_35C9" "$SHA_35C9"
require_hash "$TRAIN_RESEARCH_V2" "$SHA_RESEARCH_V2"
require_hash "$TRAIN_116188" "$SHA_116188"
require_hash "$TRAIN_S0" "$SHA_S0"

/usr/bin/install -d -m 700 -- \
  "${RUN_ROOT}/processes" "${RUN_ROOT}/receipts" "${RUN_ROOT}/comparisons" "${RUN_ROOT}/tmp"

unset WORLD_SIZE LOCAL_WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
unset NEURON_CC_FLAGS NEURON_COMPILE_CACHE_DIR NEURON_LOGICAL_NC_CONFIG
unset PYTHONOPTIMIZE PYTHONPATH

[[ -f "$HOST_LOCK" && ! -L "$HOST_LOCK" ]] || die "shared device lock identity invalid"
exec 9>>"$HOST_LOCK"
/usr/bin/flock --exclusive --nonblock 9 || die "shared device lock is busy"
if /usr/bin/flock --exclusive --nonblock "$HOST_LOCK" /usr/bin/true; then
  die "shared device lock exclusivity check failed"
fi

run_probe "$LABEL_35C9" "$TRAIN_35C9" "$SHA_35C9" "${FLAGS_35C9[@]}"
run_probe "$LABEL_RESEARCH_V2" "$TRAIN_RESEARCH_V2" "$SHA_RESEARCH_V2" "${FLAGS_35C9[@]}"
run_probe "$LABEL_116188" "$TRAIN_116188" "$SHA_116188" "${FLAGS_116188[@]}"
run_probe "$LABEL_S0" "$TRAIN_S0" "$SHA_S0"

readonly RECEIPT_35C9="${RUN_ROOT}/receipts/${LABEL_35C9}.json"
readonly RECEIPT_RESEARCH_V2="${RUN_ROOT}/receipts/${LABEL_RESEARCH_V2}.json"
readonly RECEIPT_116188="${RUN_ROOT}/receipts/${LABEL_116188}.json"
readonly RECEIPT_S0="${RUN_ROOT}/receipts/${LABEL_S0}.json"

compare_receipts "35c9-vs-research-v2-default-off" \
  "$RECEIPT_35C9" "$RECEIPT_RESEARCH_V2" "$SHA_35C9" "$SHA_RESEARCH_V2"
compare_receipts "116188-vs-s0" \
  "$RECEIPT_116188" "$RECEIPT_S0" "$SHA_116188" "$SHA_S0"
compare_receipts "35c9-dense-flags-vs-116188" \
  "$RECEIPT_35C9" "$RECEIPT_116188" "$SHA_35C9" "$SHA_116188"

readonly BATCH_RECEIPT="${RUN_ROOT}/batch-receipt.json"
"$PYTHON" - "$BATCH_RECEIPT" "$RUN_ROOT" "$(/usr/bin/git -C "$ROOT" rev-parse HEAD)" \
  "$EXPECTED_HOST" "$BATCH_LABEL" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import time

out = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
head, host, label = sys.argv[3:]
artifacts = sorted(root.glob("receipts/*.json")) + sorted(root.glob("comparisons/*.json"))
assert len(artifacts) == 7
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema": "workflow-v2-cpu-g2-batch-v1",
    "label": label,
    "host": host,
    "git_head": head,
    "synthetic_cpu_only": True,
    "steps_per_probe": 3,
    "nproc_per_probe": 4,
    "probe_count": 4,
    "comparison_count": 3,
    "all_passed": True,
    "lock": {
        "path": "/home/ubuntu/.cache/trainium_freshstart/device0.lock",
        "held_through_all_probe_quiescence": True,
        "second_nonblocking_flock_rejected": True,
    },
    "artifacts": {str(p.relative_to(root)): digest(p) for p in artifacts},
    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError("short write")
        view = view[n:]
    os.fsync(fd)
finally:
    os.close(fd)
dirfd = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(dirfd)
finally:
    os.close(dirfd)
assert json.loads(out.read_text())["all_passed"] is True
PY

trap - EXIT INT TERM HUP
echo "PASS: ${BATCH_RECEIPT}"
