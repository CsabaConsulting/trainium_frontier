# Independent post-launch review: c5c A0 controls and S0 r1

Verifier scope: evidence committed through controller `b64a249d` (including
trn22 Operator commit `f6319e53`). This is a **post-launch receipt review**, not
a training-result verdict, native correctness proof, score, or authorization to
launch another run. I did not access either host. The human's 25 September
break-glass clarification waived independent pre-launch review for these S0/A0
runs only; it did not waive this review before using their results.

## trn21 A0: PASS for launch receipt; HOLD for result use

- Assignment:
  `research/v2/assignments/20260925-trn21-a0-breakglass-r1.json`; completed
  proposal and independent exact-byte/argv CPU G2 are cited there. One-use
  label `fs-20260925-v2-c5c-a0-control-trn21-r1`, seed 42, 1800 s training
  budget, full-chip `torchrun --standalone --nproc_per_node=4`.
- The assigned and archived `host-launch.json` argv match exactly, in order:
  `--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct
  --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce
  --seq-len 1024 --no-eval-public --matrix-lr 0.015`. The source SHA-256 is
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  the trusted `prepare.py` SHA-256 is
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- The raw preflight/stage/poststage/launch records in
  `research/v2/observations/20260925-trn21-a0-manual-r1/` show an initially
  absent unique destination, host source rehash, clean pinned e65 dependency
  checkout, no STOP, idle Neuron/host lock, and 57.8 GB free (above the
  6.5 GiB break-glass floor). The poststage host check at 13:30:56 UTC
  reports `source_match:true`, `idle:true`, and matching warm-cache inodes.
  The actual launch receipt reports `STARTED` at 13:31:07 UTC.
- Independent read-only host capture
  `research/v2/observations/20260925-dual-host-readonly-operator/OPERATOR_RAW_TRN21_READONLY_20260925T133116Z.json`
  has successful authenticated transport, four Neuron worker PIDs
  2866521–2866524, `torchrun` PID 2866506, device and host lock holders,
  57,824,366,592 bytes free, and clean pinned source hashes at 13:31:21 UTC.
  The earlier 13:24:40 post-S0 capture showed no job/device/lock holder, so
  there is no evidence of two simultaneous trn21 jobs at A0 launch.
- Controller diff `b18fb358..b64a249d` does not edit `train.py`, `prepare.py`,
  `submissions/research_v2/train.py`, or `workflow/tools/decide.py`.

**Limit:** process start is not completed training. Do not feed this control to
`decide.py` or compare A1 against it until the run exit, log, steps, checkpoint,
fresh-process public-20M/context-1024 evaluation, and ledger row are retrieved
and independently checked. The global workflow-v2 `launch_authorized:false`
state is not converted to `true` by this one-off break-glass assignment.

## trn22 A0: HOLD for independent post-launch proof and result use

- The one-use assignment
  `research/v2/assignments/20260925-trn22-a0-breakglass-r10.json` names label
  `fs-20260925-v2-c5c-a0-control-trn22-r10`, exact c5c source SHA above,
  and archived exact-argv four-rank CPU G2 receipt SHA-256
  `38dd39a02d86d639f6333228048f8bdb9dcc77735f03c2d45963e1057bf945b9`.
  `research/v2/operator/20260925-trn22-a0-r10/preflight.json` reports the
  c5c/prepare hashes, 58,452,512,768 free bytes, no job PIDs, clear Neuron,
  seed 42, 1800 s, LNC2, and the same A0 science flags. It records the
  manual harness and fresh train/eval sandbox paths. `attempt.json` records
  one launch attempt at 13:19:56 UTC, before assignment expiry.
- `LAUNCH_RECEIPT.md` attests a detached driver, `torchrun`, and four worker
  PIDs at 13:20:10 UTC, host package rehash, and exact inner command. However
  `f6319e53` archives only that narrative, `preflight.json`, and
  `attempt.json`. It does **not** archive the raw post-launch process/Neuron
  snapshot, staged `launch-command.txt`/provenance, or initial `train.log`
  needed to independently corroborate the four workers and the inner
  `torchrun` argv on the host. The latest committed independent trn22 host
  snapshot predates this launch. Therefore the Operator's claim is plausible
  but **not independently proven by the fixed packet**. Do not interrupt the
  running job for this documentation HOLD.
- The historical harness checkout had six pre-existing untracked protected
  logs but no tracked source edits. That is not evidence of altered training
  bytes; verify the fresh sandbox's exact source/evaluator hashes and effective
  dirty-tree status from raw provenance before declaring the eventual result
  comparable. Controller diff `b18fb358..b64a249d` has no science/evaluator
  source edit.

**Minimum to clear:** archive raw authenticated post-launch worker/Neuron/lock
observation (or equivalent host process receipt), staged package hashes and
exact inner launch command, and initial live log/provenance. Then separately
check the completed exit, steps, checkpoint, fresh-process public-20M/context-
1024 eval, and ledger row. Neither an A0 score nor an A2 comparison is yet
approved by this review.

## trn21 S0 r1: failed gate, no score, cause not established

The distinct one-use S0 label `fs-20260925-s0-cold-replay-trn21-r1` used
SHA-256 `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb`
and bare four-rank `torchrun`. Archived `training.log` reports rank 3's
Neuron `nki_kernel_global` compilation error, specifically failure reading a
`module.neff` under the private debug directory, at initialization before the
first training step; torchrun exited 1 and no checkpoint or public BPB exists.
This is the observed failure, **not** proof of a model bug, disk exhaustion,
compiler-cache root cause, or even a genuinely cold persistent compiler
cache. `launch.json` sets `NEURON_COMPILE_CACHE_DIR` and local debug/cache
directories, but contains no recorded `NEURON_COMPILE_CACHE_URL` or explicit
compiler `--cache_dir` binding. A descriptive “cold” label does not establish
cache isolation. The failure label is consumed; any diagnostic needs a new
label and separate authority. No scored S0 submission gate passed.
