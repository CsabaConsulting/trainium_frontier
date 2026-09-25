# trn21 c5c A0 source stage and exact-environment CPU G2

Status: **HOLD — Implementer-prepared, not an Operator assignment or dispatch
request.** G1 independently PASSed only the old retained-checkpoint public-20M
evaluator calibration. This runbook prepares the *first* trn21 c5c A0 source
stage and a separate, CPU-only four-rank probe. Neither a successful stage nor
G2 authorizes native training, evaluation, adoption or submission.

## Provenance and remaining decisions

- Science proposal and conditional Supervisor ratification:
  `research/submission-validation/20260925-v2-c5c-a0-control-trn21-r1/`
  and `research/v2/authorizations/20260925-trn21-c5c-a0-r1-proposal-ratification.json`.
  Their hashes are bound in [GATE_CONTRACT.json](GATE_CONTRACT.json).
- Independent trn21 G1 PASS review:
  `research/agentic/verification/20260925-g1-trn21-r2-calibration-result-review.md`,
  SHA-256 `2c0674fe4eb0e3ad64cf07bbcf6a6d24bbc89595d6e5d1b488c93b0e06f84ed0`.
  Its reproduced 0.9918353324459732 is **not** a new c5c control result.
- The Operator's read-only A0 feasibility receipt is
  `research/v2/authorizations/fs-20260925-v2-c5c-a0-control-trn21-r1.readonly-feasibility.json`,
  SHA-256 `90b77b5f220da9e72dbe1932b27b87b953ba85865793ea9ebc3318997f330df1`.
  It reported the e65 source commit absent from 144 one-level trn21 Git
  repositories, 58,676,293,632 B root free, an idle/clean host and no A0
  science-label collision. These are *historical observations*, not this
  package's live stage/G2 preflight; its independent review is still required.
- This whole package needs an independent Verifier review, a separate
  Supervisor stage/CPU-G2 assignment, and a fresh Operator preflight. An open
  or uncertain breaker is a HOLD for specific human disposition. No agent may
  infer authority from the conditional science ratification or G1 PASS.

## Exact source transfer; no host fetch or substitution

The pinned source is commit
`e65a3097a5b00f6178188a935bb2acb9a5042c6c`. Its **root** `train.py`
must SHA-256 to
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
and unchanged `prepare.py` to
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The model file is already the exact c5c bytes in e65; do not copy a different
`train.py` over it. An Implementer-built, one-commit **shallow Git tar capsule**
is available locally at
`/dev/shm/trainium-c5c-e65-one-commit-20260925.tar.gz` (104,005,787 B,
mode `0600`, SHA-256
`32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264`).
Local extraction produced the exact HEAD, one reachable commit, a clean tree,
empty remotes, no object alternates and a passing Git connectivity check.
The source tree had no symlinks. This local test does **not** verify the later
host transfer. The capsule is in tmpfs, not Git; if absent or hash/size-drifted,
stop for a new capsule and independent review. A Git bundle is deliberately
not used: a shallow bundle failed the local extraction test because it omitted
the parent boundary object.

Only after the package review and stage assignment, the Operator uses the
existing pinned trn21 SSH identity/endpoint with `BatchMode`, `IdentitiesOnly`
and strict host-key checking to transfer **that exact file** to the previously
absent host path
`/dev/shm/trainium-c5c-e65-one-commit-20260925.trn21.tar.gz`.
Do not print key contents or create a new credential. The Operator verifies
the file's 104,005,787-byte size and SHA-256 *both before and after transfer*,
checks its tar members are relative/no traversal and contain no symlinks, and
archives the transfer evidence. Do not use a host `git fetch`, an unreviewed
remote branch, a visually similar file, or another machine's working tree.

The sole new science checkout path is
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r1`.
It must be absent before the Operator atomically creates it mode `0700` and
extracts the verified capsule with no same-owner, no same-permission and no
overwrite semantics. The following is a **stage recipe**, not permission to
run it now; the Operator must capture the resulting receipt before G2:

```bash
set -euo pipefail
source_tar=/dev/shm/trainium-c5c-e65-one-commit-20260925.trn21.tar.gz
source_checkout=/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r1
test ! -e "$source_checkout"
test "$(stat -c %s -- "$source_tar")" = 104005787
test "$(sha256sum -- "$source_tar" | cut -d ' ' -f 1)" = 32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264
mkdir -m 700 -- "$source_checkout"
tar -xzf "$source_tar" --no-same-owner --no-same-permissions --no-overwrite-dir -C "$source_checkout"
test "$(git -C "$source_checkout" rev-parse HEAD)" = e65a3097a5b00f6178188a935bb2acb9a5042c6c
test "$(git -C "$source_checkout" rev-list --count HEAD)" = 1
test "$(git -C "$source_checkout" rev-parse --is-shallow-repository)" = true
test -z "$(git -C "$source_checkout" remote)"
test ! -e "$source_checkout/.git/objects/info/alternates"
test -z "$(git -C "$source_checkout" status --porcelain=v1 --untracked-files=all)"
git -C "$source_checkout" fsck --no-reflogs --connectivity-only --no-progress
test "$(sha256sum -- "$source_checkout/train.py" | cut -d ' ' -f 1)" = c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586
test "$(sha256sum -- "$source_checkout/prepare.py" | cut -d ' ' -f 1)" = 5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb
test ! -e "$source_checkout/out/fs-20260925-v2-c5c-a0-control-trn21-r1"
```

Before extraction, independently check the archive member list and local Git
config for unsafe hooks, filters, remotes, credential helpers, path traversal
or symlinks; after extraction recheck the stage tree and those config keys.
If a target already exists, archive/check fails, stage becomes dirty, or any
identity differs, record **FAIL/HOLD** and leave the partial target for
diagnosis. Do not retry the same stage path by overwriting or deleting it.
The stage receipt must bind source/host archive hashes, exact commit and
worktree state, tool hashes, host identity, fresh disk bytes, retained cache
device/inode identities, protected checkpoint size/hash, and absent labels/
science outputs. A different Verifier reviews that receipt before G2.

## Separate CPU-tool stage and live preflight

The e65 source capsule intentionally contains **no** workflow G2 tools.
Stage only these two existing local files to the new, otherwise absent
`/dev/shm/trainium-v2-g2-tools-20260925-trn21-r1/workflow/tools/` tree, using
the same pinned transport; the root is mode `0700`, and the files must match
these hashes on both sides:

| File | Required SHA-256 |
| --- | --- |
| `workflow/tools/preflight_dryrun.py` | `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8` |
| `workflow/tools/contained.py` | `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d` |

These are outside the clean science checkout. No model, `prepare.py`, data,
evaluator or scoring file is edited. Tool transfer alone is not CPU-G2
authority; archive and independently review its hashes with the source stage.

Immediately before a separately assigned G2, the Operator reauthenticates:

1. trn21 hostname `ip-172-31-34-85`, exact clean e65 HEAD, two source hashes,
   tool hashes, no inherited rank/distributed/Python/Neuron override env,
   and no open global/trn21 stop.
2. Root free space at least **10,737,418,240 B**, `/dev/shm` capacity,
   unchanged two protected checkpoint hashes from the contract, and each of
   the six historical warm-cache paths' device/inode identity. No cache
   cleanup, copying, redirection or cold fallback.
3. No train/eval/G2/guard process, shared lock or Neuron holder; science A0
   label/checkout output/step/log absent and CPU G2 label/root absent. The
   new G2 root must not be created before this absence check.

The earlier read-only receipt is supporting evidence, not a substitute for
these live checks. The retained 35c9 checkpoint and 116188 G1 checkpoint are
not warm starts for this A0 control. Any drift or occupied host is HOLD.

## Exact four-rank CPU G2 shape — still on HOLD

Use new **CPU-only** label
`fs-20260925-g2-a0-c5c-exactenv-trn21-r1`, isolated root
`/dev/shm/trainium-v2-g2-20260925-trn21-r1`, mode `0700`. Its step-log parent
is the new root and must exist **before** launching; the step file itself must
not. This avoids the failed trn22 r2 probe, whose rank 0 could not append to
an absent `research/run-logs` parent. Do not create anything inside the clean
checkout to make the probe pass. The G2 tool appends only `--device-type cpu
--no-compile --num-steps 3 --out-dir <isolated-G2-workdir>/out` to the exact
Track-A training suffix; this is synthetic CPU data, no native training or
public evaluation.

The following is a reviewed **candidate command shape** for the Operator's
later CPU-only assignment, not an executable authorization from this file:

```bash
set -euo pipefail
g2_label=fs-20260925-g2-a0-c5c-exactenv-trn21-r1
g2_root=/dev/shm/trainium-v2-g2-20260925-trn21-r1
g2_checkout=/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r1
g2_tools=/dev/shm/trainium-v2-g2-tools-20260925-trn21-r1/workflow/tools
g2_proc_dir="$g2_root/process-$g2_label"
g2_step_log="$g2_root/$g2_label.steps.jsonl"
g2_science_label=fs-20260925-v2-c5c-a0-control-trn21-r1
test ! -e "$g2_root"
mkdir -m 700 -- "$g2_root"
test -d "$g2_root"
test ! -e "$g2_step_log"
/home/ubuntu/workspace/native_venv/bin/python "$g2_tools/contained.py" launch \
  --label "$g2_label" --dir "$g2_proc_dir" --cwd "$g2_checkout" --cgroup -- \
  /usr/bin/env -i \
  HOME=/home/ubuntu \
  PATH=/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin \
  NKI_VENV_PATH=/home/ubuntu/workspace/native_venv \
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
  TMPDIR="$g2_root" TRN_EXP_LABEL="$g2_label" \
  NEURON_LOGICAL_NC_CONFIG=2 \
  'NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16' \
  NEURON_COMPETITION_R1_SEED=42 NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800 \
  TORCH_NEURONX_NEFF_CACHE_DIR=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/neff \
  TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/local \
  TORCH_NEURONX_HLO_CACHE_DIR=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/hlo \
  TORCHINDUCTOR_CACHE_DIR=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/inductor \
  NEURON_COMPILE_CACHE_DIR=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/compiler \
  TORCH_NEURONX_DEBUG_DIR=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/debug \
  NEURON_COMPETITION_R1_OUT_DIR="$g2_checkout/out/$g2_science_label" \
  NEURON_COMPETITION_R1_STEP_LOG="$g2_step_log" \
  TRAINIUM_FRESH_START_RUN_ID="$g2_science_label" \
  /home/ubuntu/workspace/native_venv/bin/python "$g2_tools/preflight_dryrun.py" \
  --train-py "$g2_checkout/train.py" --prepare-dir "$g2_checkout" \
  --steps 3 --nproc 4 --timeout 900 --out "$g2_root/c5c-g2.json" --keep -- \
  --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct \
  --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce \
  --seq-len 1024 --no-eval-public --matrix-lr 0.015
```

After launch, wait on that exact contained label/dir with an outer bound of
1,200 seconds. If the probe times out or fails, authenticate the owned label
then stop **only** it; do not launch another job. In all cases call
`contained.py quiesce --label "$g2_label" --dir "$g2_proc_dir" --files
"$g2_proc_dir/launch.json" "$g2_proc_dir/run.log"`; those two files are
expected even if the G2 receipt or step file is absent. A failed quiescence
is a stop/breaker investigation, not a retry. Inspect the G2 receipt itself;
`contained.py wait` returning zero is not a G2 PASS.

The Operator archives the full `c5c-g2.json`, four rank JSONs, torchrun log,
contained launch/run/quiescence/optional stop receipts, isolated step log and
SHA-256 manifest. Require `verdict=PASS`, all four ranks PASS, exact source
and `prepare.py` hashes, exact training argv, three finite losses per rank,
`steps=3`, `nproc=4`, micro-batch `[2,256]`, stable ownership plan, no
cross-rank parameter mismatch and model-only checkpoint/load-for-eval PASS.
The appended CPU `--out-dir` must reside under the isolated G2 root, and the
step log must contain exactly `0`, `1`, `2` in order. Recheck source clean,
checkpoints, cache identities, quiescence/Neuron state, disk and absent
science output/step/log paths afterward. Failure is recorded under this G2
label; never silently reuse it.

A **different Verifier** then compares the new PASS receipt with the archived
c5c CPU PASS anchor
`research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/remote-run/receipts/fs-20260925-g2-a0-research-v2-trn22-r1.json`
(SHA-256 `17ce1ec84ddc0d585ecd103237ecfcd86d3820cdd9aaf1c4fe8b60ed308b90e3`)
using `preflight_dryrun.py --compare A.json B.json`. The output must report
`identical: true` for plan hash, per-rank losses, state keys and final
parameter digests. Cross-host bit identity is a **test**, not an assumption.

Native A0 remains **HOLD** even if the source stage and CPU G2 pass. A later
separately reviewed native launcher/request may need changed monitoring
mechanics; it must rebind its own exact tool and source hashes, run a
read-only external optimizer-boundary observer, continuously sample root
free bytes during compile/training and stop the authenticated owned guard
before the **6,979,321,856-B (6.5-GiB)** floor. Immediately pre-dispatch root
free must still be at least **10,737,418,240 B**. Only a fresh host preflight,
independent final review and one-attempt Supervisor authorization can permit
that native label. No claimed BPB gain or submission readiness follows from
this Implementer package.
