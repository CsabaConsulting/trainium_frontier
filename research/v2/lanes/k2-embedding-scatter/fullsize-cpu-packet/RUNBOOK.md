# Exact117M CPU G2 packet — preparation only

Do not run on the current laptop. Source review is also HOLD: root reports six
cross-tile cancellation failures from the fresh reviewer's own NKI-algorithm
emulation. Wait for its archived reproducer; no candidate edit or threshold
change is authorized here. The CPU ON function uses `index_add_`, not the native
kernel, so an eventual G2 PASS cannot erase that distinct failed review.

Candidate25b731f7/e4d93babb4ce is untouched. This branch adds only this packet.
No host access, tensor execution, canonical import, queue or ledger mutation.

## Resource finding

At19:43:37UTC, laptop MemAvailable was6,073,999,360bytes; swap already used
1,464,233,984bytes. The archived real ownership plan proves117,453,580 FP32
parameters. Params+grads and ownership send/receive storage alone total
**6,191,712,640bytes across4 ranks**. This excludes optimizer state, BF16 caches,
activations, dummy Muon padding, scratch, runtime and checkpoint-reload model.
The available-memory shortfall therefore exists before uncertain costs.
Current cgroup has no hard memory limit. There is no defensible safe local
upper-RSS or runtime bound. Do not rely on swap or close other applications.

The existing reviewed full-size host driver required64GiB MemAvailable and
6.5GiB disk. Retain that known-safe resource guard until independent measured
evidence supports another configuration. That floor is not measured peak usage.
Prior exact6ef7 host runs took20.0s (RecursiveON) and19.9s (RecursiveOFF), with
about121GiB available. They have no peakRSS measurement. Three sequential arms
may take about one minute on the same adequately provisioned environment, but
this is context, not a promised upper bound or timing benchmark.

Laptop `/tmp` is tmpfs, with5.04GB available. Each retained full checkpoint is
roughly470MB; using that tmpfs worsens RAM pressure. Existing host CPU route used
task-owned disk TMPDIR; preserve this. Root has44.39GB disk available locally,
but disk does not fix RAM. Local Python3.14/Torch2.13 runtime lacks requests and
PyArrow. The earlier tiny tests explicitly used offline stubs/Gloo compatibility;
the unchanged real prepare route cannot currently import there. The offline
prepare stub excludes2 startup steps, whereas real prepare excludes5. It must
not be silently substituted for the exact prior host gate or native accounting.
Do not install dependencies or relax shapes in this preparation task.

## Reused evidence and machinery

- `workflow/tools/preflight_dryrun.py` SHA
  `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`.
- Exact real prepare SHA
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Prior `research/v2/operator/20260925-trn21-l3-timer-dense-g2-r1/g2.json`
  SHA `48ff468c7fc9131fb70a3319158695176c5d0f54a3ef1a09dddad96687bbe162`.
- Prior schedule-OFF counterpart under
  `20260925-trn21-l3-timer-dense-off-g2-r1/g2.json`, SHA
  `6813cf265a125782c6792dd3eca5f6ab47629e8d90e59fc05b87cc5b9c1028ff`.
- Prior `...dense-g2-r1/cpu_driver.py` SHA
  `cd48f01b2a1a4fb28d785d780361bf55d2ff8c2cd0dba929032cfe42db121a02`.

The old driver has historical hardcoded paths and lacks the latest shared-lock
guard. It is evidence for the CPU command/resource floor, **not a launcher to
restart or copy unmodified**. Use the current reviewed manual route/guard.
No new launcher or approval mechanism is supplied.

## Exact execution contract, after both HOLDs resolve

Supervisor binds the existing memory-rich host and unused label/path to the
current Operator. No new instance/resource authority. Acquire the same remote
exclusive `/home/ubuntu/.cache/trainium_freshstart/device0.lock`; fresh `ps` AND
`neuron-ls` must prove idle under the lock. Fail closed/log if any check fails;
retain lock through every child, final receipt, and process-exit verification.
These are CPU-only jobs, but still obey the one-host-job rule and cannot overlap
P/CT/native work. Do not let packet preparation block a ready native job.

Stage base from exact commit a4aa4351789115a2d0aa947d9ca00ff689f9bd0b and candidate
from25b731f7638672aa8d03a7cd15b6a02b8f5776ea. Authenticate each source SHA against
REQUEST.json. Reference is genuine6ef7, not a reconstructed or flag-mutated file.
Keep all62 tensors/117,453,580 parameters, depth6, width1024 and ordinary8192-row
vocabulary. No depth/width/MLP/tokenizer reduction and no table/architecture flags.

Operator creates one new disk-backed packet directory (below called PACKET),
with `reference/train.py`, `candidate/train.py`, the trusted G2 driver, `tmp`,
and output locations; PREPARE denotes its verified unchanged organizer directory.
These are explicit binding placeholders, not permission to reuse an existing
output path. Bind actual paths before executing. Native venv Python is the same
existing2.11.0+cpu runtime as the prior host receipt.

Each of these three commands runs **sequentially**, with one of the substitutions
in the table; all other argv are identical. Common environment:
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
PYTHONDONTWRITEBYTECODE=1, NEURON_LOGICAL_NC_CONFIG=2,
NEURON_CC_FLAGS='--optlevel=1 --auto-cast matmult --auto-cast-type bf16',
NEURON_COMPETITION_R1_SEED=42, NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800,
TMPDIR=PACKET/tmp. Do not deploy laptop import stubs.

| Arm | SOURCE | RECEIPT | EXTRA |
|---|---|---|---|
| reference | PACKET/reference/train.py | PACKET/reference-g2.json | empty |
| explicit OFF | PACKET/candidate/train.py | PACKET/off-g2.json | --no-embedding-scatter |
| ON | PACKET/candidate/train.py | PACKET/on-g2.json | --embedding-scatter |

```text
/home/ubuntu/workspace/native_venv/bin/python -B PACKET/preflight_dryrun.py
 --train-py SOURCE --prepare-dir PREPARE
 --steps 3 --nproc 4 --micro-rows 2 --micro-tokens 256
 --timeout 3600 --trace-rmsprop --keep --out RECEIPT --
 --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct
 --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce
 --seq-len 1024 --no-eval-public --matrix-lr 0.015 --recursive-schedule
 --seed 42 --max-train-seconds 1800 --num-steps 1000000 --out-dir out EXTRA
```

Harness timeout3600 is inherited unchanged, not a predicted runtime or a safe
laptop guard. Its subprocess timeout kills the launcher directly, without an
explicit process-group cleanup in that file. On failure/timeout the Operator
must verify all job-owned children exited before proceeding; do not interpret a
launcher exit as proof of quiescence or run another arm with surviving children.
The current reviewed outer lifecycle/idle checks remain necessary. No group
termination or other host action is taken by this packet.

After all arms pass, run the existing `--compare reference-g2.json off-g2.json`
and `--compare reference-g2.json on-g2.json` and archive exact results. Expected
plan SHA `b89d62227c844eb13706fbe81bddfb9b8b7db5fefcd57b4e7d0ea3213dca5031`,
62 parameter digests, three losses on each of4 ranks, checkpoint strict reload
and deterministic finite logits. Missing/rankFAIL/exitERROR means no PASS.
Do not compare exact loss bits with an old different runtime; fresh reference
and candidate share the same runtime and test shims.

The model is full size, but G2 slices each training forward to2x256:512 tokens
per rank,2048 across4 ranks, not production32768/rank or131072/update. CPU ON
uses the candidate's CPU reference; NKI never executes. Native gradient
correctness (including the reported review failures), cold startup, whole-step
speed, official budget and quality are all separate pending gates.

## Decision and undo

2026-09-26 01:18 IST /2026-09-25 19:48UTC: retain preparation-only status and
require the existing memory-rich host resource profile. Reason: mandatory tensor
floor exceeds laptop available memory; source independently reported HOLD.
Undo: discard this packet branch; no run, production change or host state exists
to reverse. Supervisor may assign execution only after the source HOLD and
resource/preflight conditions are resolved. No human question is needed overnight.
