# Independent Verifier: trn22 G2 r3 → frozen r4 source identity

Reviewed 2026-09-25 02:24 UTC, without SSH, host/AWS action, training,
evaluation, or source implementation. I did not operate or implement the
trn22 G2 r3 probe.

**Verdict: PASS_CARRY_FORWARD_NUMERIC_ONLY for the exact c5c `train.py` and
`prepare.py` bytes, Track-A argv, and four-rank CPU G2 conditions.** This is
the separate source-identity disposition allowed by the r4 HOLD runbook; it
does not relabel the old host receipt as an execution from commit `446b6576`,
prove that a new r4 checkout exists, validate the new safety launcher on a
host, or authorize source transfer, CPU probing, native dispatch, evaluation,
adoption, or submission. If a future staged r4 checkout or native request
differs on any compared byte, argv, or effective value below, this disposition
is void and G2 must be rerun under a new reviewed request.

## Frozen evidence and independent checks

- Prior exact-env G2 r3 used source commit
  `e65a3097a5b00f6178188a935bb2acb9a5042c6c`; its result is
  `research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn22-r3/remote-run/c5c-g2.json`,
  SHA-256 `38dd39a02d86d639f6333228048f8bdb9dcc77735f03c2d45963e1057bf945b9`.
  The Operator receipt SHA-256 is
  `c8d73cc447a4fb99c9aba1869f3df47accee550ea2e04d704b9a43cd1bd8e116`.
  Its separate independent numeric review SHA-256 is
  `e76654c7882c2ede9e382914aebb672b10841a083fc0eb2271e15b39965a1de4`.
  I re-ran the ten-entry retrieved `SHA256SUMS` manifest: all ten OK.
- Frozen r4 source is `446b657695f086f9b484fa601c9a3b61c2495912`,
  a descendant of e65; `git diff --check e65..446b` passed. The disabled r4
  request is SHA-256
  `ece18831f81ff8460f923165bb758c6aa02435ffe01e08398ccda891dd8baac0`.
  Its independently reviewed HOLD-only package remains `dispatch_authorized=false`.
- Direct Git blob byte checks at **both** e65 and 446b give `train.py`
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `prepare.py`
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  `git diff --exit-code e65..446b -- train.py prepare.py` passed. These
  match both fields in the retrieved G2 receipt and r4 request. The scored
  evaluator, tokenizer, data, and model arithmetic are not changed by the
  e65→446b diff.
- The CPU harness and containment tools were staged separately and are **not
  Git paths in either frozen source commit**. The prior G2 Operator preflight
  authenticated their host SHA-256 values; the current repository copies
  rehash identically: `workflow/tools/preflight_dryrun.py`
  `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`,
  `workflow/tools/contained.py`
  `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d`.
  This is not a claim about their present host bytes; those require fresh
  authentication if a new action ever uses them.
- A direct JSON comparison of the G2 Operator receipt with the disabled r4
  request returned true for the **ordered, exact Track-A training argv**:
  `--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct
  --nki-local-conv --bf16-norm-output --pack-factor 32
  --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015`.
  The four effective values also match exactly: LNC `2`,
  `NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16`,
  seed `42`, and maximum training seconds `1800`. The native request has
  `--nproc_per_node=4`; the G2 receipt ran four ranks. All six warm-cache
  environment path values match exactly.
- The G2 receipt reports `PASS`, torchrun exit 0, four PASS ranks, three
  optimizer steps, CPU 2×256 micro-batch, plan SHA-256
  `b89d62227c844eb13706fbe81bddfb9b8b7db5fefcd57b4e7d0ea3213dca5031`,
  and the separate step log is exactly `0`, `1`, `2`. I reran the pinned
  `preflight_dryrun.py --compare` against the prior c5c anchor: exit 0,
  `identical=true`, `plan_equal=true`, `losses_identical=true`, no state-key
  or parameter-digest differences, 62 parameters compared.

## Source-diff and path boundary

The only changed or added production Python modules from e65 to 446b are
`research/fresh_start_guard.py`, `research/fresh_start_host.py`,
`research/fresh_start_launch.py`,
`research/fresh_start_monitor_interlock.py`, and the base and paired root-free
monitors. The CPU G2 harness imports the exact `train.py` and `prepare.py`
files, and those files do not import these changed `research` modules. Thus
none of these non-scored safety/orchestration changes changes what the old
CPU G2 actually tested. Conversely, G2 never tested the r4 guard, launcher,
monitor, source transfer, Neuron compile, 30-minute runtime, or public BPB.
They retain their independent safety and live-host gates.

The old G2's checkout and temporary CPU `--out-dir` paths necessarily differ
from a future r4 checkout. The harness appends its CPU-only
`--device-type cpu --no-compile --num-steps 3 --out-dir ...` flags, so the
scored train argv remains the ordered Track-A vector above. `prepare.py`
uses `NEURON_COMPETITION_R1_STEP_LOG` only to append a completed step; the
file is never model input. `train.py`'s `NEURON_COMPETITION_R1_OUT_DIR` is
only a default overridden by the G2 `--out-dir`. Label/TMPDIR/step-log and
science output-path substitutions are artifact/containment differences, not
evidence of a new numeric recipe. They still need correct live parent/path
checks; the earlier failed G2 r2 is proof that such checks matter.

## Unchanged HOLD gates

The prior G2 receipt was executed from the e65 r2 checkout, **not** a 446b
r4 checkout. Before the r4 science label can launch, an independently
reviewed exact 446b transfer and clean checkout must exist, with fresh host
hashes for `train.py`, `prepare.py`, CPU tools if used, and every requested
source fingerprint. The new request/bindings, paired root-free monitors,
boundary observer, protected assets, disk/process/lock/STOP checks, and
one-label one-attempt Supervisor authorization still require their own
independent gates. This carry-forward is a CPU-numeric source-identity
disposition only; it does not set `research/v2/state.json` launch permission
or convert the disabled r4 package into a dispatchable request.
