# Pending trn22 c5c A0 r2 exact-environment CPU G2 retry

Status: **HOLD — independent safety classification and package review required.**
Prepared by the Implementer only. This file is not a host-dispatch instruction,
native-run authorization, or reset of any breaker. The separate science A0 label
`fs-20260925-v2-c5c-a0-control-trn22-r2` remains unconsumed.

## Diagnosis and breaker ambiguity

The [archived r2 CPU G2 receipt](../fs-20260925-v2-c5c-a0-control-trn22-r2/operator-stage-hold.json)
shows that source staging passed but rank 0 failed at its first
`prepare.record_step_boundary(0)`: `NEURON_COMPETITION_R1_STEP_LOG` pointed to
`research/run-logs/...` inside the new clean checkout, and that parent did not
exist. The other three Gloo failures followed rank 0's exit. The CPU process was
quiescent afterward; it performed **no native training or evaluation**, produced
no public BPB, and did not consume the science A0 r2 label. The CPU G2 **r2**
label is consumed and must never be rerun.

The workflow also blocks a host after two consecutive launch failures and
pauses code arms after a correctness failure. The native A0 r1 disk stop was
one failed trn22 launch; whether this contained CPU preflight failure is a
second launch failure or a correctness-breaker event has **not** been
independently classified. Do not infer that the human's earlier re-arm of the
*named disk breaker* clears either possibility. A separate Safety Analyst or
Verifier must decide this from both archived receipts and current state. If
either breaker applies or the disposition is uncertain, keep this retry on
HOLD and request a *specific human acknowledgement/reset*. The Implementer
does not reset any breaker.

## Why this correction is input-neutral

The only failed path was an observer artifact. In unchanged `prepare.py` lines
43–45 and 96–111, the step-log environment variable is read into a path at
import; after an optimizer step, **rank 0 only** appends the step number and
flushes it. The function never reads the file or supplies model input. The
unchanged c5c `train.py` calls it at line 4010 after step synchronization.
The G2 tool appends `--device-type cpu --no-compile --num-steps 3 --out-dir
<temporary-G2-workdir>/out` to the otherwise exact Track-A training argv.
Thus this retry changes only the CPU probe's label, temporary artifact root,
and step-log destination. The four science-effective environment values,
model and `prepare.py` bytes, six cache paths, science OUT_DIR default,
science RUN_ID, and Track-A argv stay identical to the failed G2. This is
not a claim of native equivalence, throughput, or BPB.

## Immutable inputs and destination

The existing host checkout is
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2`,
detached at `e65a3097a5b00f6178188a935bb2acb9a5042c6c`. Its root
`train.py` must hash to `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `prepare.py` to
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The retained G1 checkpoint and all six warm-cache device/inode identities
must remain as recorded in the failed Operator receipt. No source copy, edit,
cache cleanup, checkpoint prune, or worktree cleanup is part of this retry.

Use a *new* CPU-only label:
`fs-20260925-g2-a0-c5c-exactenv-trn22-r3`. The new isolated root is
`/dev/shm/trainium-v2-g2-20260925-trn22-r3`, mode `0700`. Its step-log path is
`/dev/shm/trainium-v2-g2-20260925-trn22-r3/fs-20260925-g2-a0-c5c-exactenv-trn22-r3.steps.jsonl`.
Before launch, the root must be absent, then atomically created; the parent
must exist and the step-log file itself must still be absent. Never create
`research/run-logs` or any output inside the clean source checkout to make a
probe pass. If the r3 root or label already exists, stop and assign a new
label only after a new diagnosis and review; never overwrite it.

## Operator-only gate sequence — not yet approved

1. Independent safety classification of the two breaker rules, independent
   static PASS on [GATE_CONTRACT.json](GATE_CONTRACT.json), and a separate
   Supervisor assignment for **CPU G2 only** are all required before SSH.
   `research/v2/state.json` must have no applicable active stop; no prior
   human acknowledgement may be reused for a newly tripped breaker.
2. Reauthenticate the exact hostname, pinned clean checkout/commit and source
   hashes, hash-verified G2/containment tools, retained checkpoint and six
   warm-cache identities. Check no train/eval/guard process, shared lock,
   Neuron holder, or existing r3 label. Check root free space is at least
   10,737,418,240 B and inspect `/dev/shm` free capacity. Verify science r2
   output, step-log and run-log paths remain absent. Stop on any mismatch.
3. Atomically create only the new `/dev/shm` root with mode `0700`. Verify
   it is owned by the Operator UID and the isolated step log is absent. Do
   not create anything under the checkout. The exact read-only tool hashes
   are `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`
   for `preflight_dryrun.py` and
   `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d`
   for `contained.py`, both under
   `/dev/shm/trainium-v2-g2-20260925-trn22-r1/workflow/tools/`.
4. With those gates PASS, the following is the **one reviewed command shape**.
   The variable names below refer only to the new CPU probe, not to AWS or
   the scored run. The process is contained; the child uses `env -i`.

```bash
set -euo pipefail
g2_label=fs-20260925-g2-a0-c5c-exactenv-trn22-r3
g2_root=/dev/shm/trainium-v2-g2-20260925-trn22-r3
g2_checkout=/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2
g2_tools=/dev/shm/trainium-v2-g2-20260925-trn22-r1/workflow/tools
g2_proc_dir="$g2_root/process-$g2_label"
g2_step_log="$g2_root/$g2_label.steps.jsonl"
g2_science_label=fs-20260925-v2-c5c-a0-control-trn22-r2
test ! -e "$g2_root"
mkdir -m 700 "$g2_root"
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
  TORCH_NEURONX_NEFF_CACHE_DIR=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train/neff \
  TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train/local \
  TORCH_NEURONX_HLO_CACHE_DIR=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train/hlo \
  TORCHINDUCTOR_CACHE_DIR=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train/inductor \
  NEURON_COMPILE_CACHE_DIR=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train/compiler \
  TORCH_NEURONX_DEBUG_DIR=/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train/debug \
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

5. Wait with the same contained label and directory (outer wait bound 1,200
   seconds). **Even if it times out or the probe fails**, stop only this
   authenticated label as needed, then call `contained.py quiesce` on the
   receipt and contained run log. Do not start another probe if quiescence
   fails. Inspect `c5c-g2.json` for the tool verdict; a process exit alone
   is not a PASS.
6. Require `verdict=PASS`, all four rank reports PASS, source/prepare SHA,
   exact training argv, `steps=3`, `nproc=4`, `micro_batch=[2,256]`, finite
   per-rank losses, no cross-rank parameter mismatch, stable ownership-plan
   hash, and successful model-only checkpoint/load-for-eval check. The G2
   receipt's appended argv must point its `--out-dir` into its own
   `/dev/shm/.../g2-*/out`; the science output directory must still be
   absent. The isolated step log must contain exactly three lines, `0`, `1`,
   `2`, in that order. No native throughput or BPB conclusion follows.
7. Archive the complete G2 receipt, four rank JSON files, torchrun log,
   contained launch/run/quiescence receipts, isolated step log, and a SHA-256
   manifest. Recheck clean source tree, unchanged retained checkpoint and
   warm-cache identity, idle/Neuron state, root disk, and absent science r2
   output/log/step paths. If any check fails, record FAIL/HOLD under this r3
   label; do not silently retry it.
8. Only after retrieval, an **independent** Verifier runs the built-in
   `preflight_dryrun.py --compare` on the new PASS receipt and the archived
   prior c5c PASS receipt at
   `research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/remote-run/receipts/fs-20260925-g2-a0-research-v2-trn22-r1.json`
   (SHA-256 `17ce1ec84ddc0d585ecd103237ecfcd86d3820cdd9aaf1c4fe8b60ed308b90e3`).
   The output must say `identical: true` for plan hash, all per-rank losses,
   state keys and final parameter digests. Any difference is a HOLD for
   diagnosis, not a science result.

Even a fully passing CPU G2 leaves native A0 r2 on HOLD until its own
independently reviewed final request, fresh host/disk/cache/source checks,
external boundary observer, and separate one-attempt Supervisor authority.
The workflow-v2 `launch_authorized` flag remains false. No edit to
`prepare.py`, `train.py`, scoring, data, the decision engine, or submission
candidate is proposed here.
