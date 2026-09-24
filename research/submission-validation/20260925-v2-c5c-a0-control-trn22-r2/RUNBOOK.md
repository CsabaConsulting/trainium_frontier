# Pending c5c A0 trn22 retry — safety runbook

Status: **draft only; dispatch not authorized**. This is an Implementer-prepared
retry package for `fs-20260925-v2-c5c-a0-control-trn22-r2`. It does not replace a
Researcher/Supervisor decision, independent Verifier receipt, finalized immutable
request, Operator preflight, or the human's breaker reset. Do not SSH or launch from
this document.

## Why a new attempt is necessary

The one-shot `r1` authorization was consumed. Its [terminal receipt](../../v2/incidents/fs-20260925-v2-c5c-a0-control-trn22-r1/operator-terminal.json)
and [independent safety review](../../agentic/verification/20260925-a0-trn22-disk-safety-review.md)
show a root-disk-floor stop during native compilation **before step 0**:
7,949,975,552 free bytes at preflight, 6,449,438,720 at the first below-floor
observation, and 6,449,033,216 after postflight. That is 1,500,942,336 bytes
of preflight-to-postflight consumption. The observer recorded zero optimizer
boundaries; there was no checkpoint, fresh public20M score, or scientific verdict.
The postflight had no owned processes or Neuron holder, but `host_clean=false`
solely because disk remained below the 6.5-GiB floor. The `r1` label cannot be
reused and is not a c5c control.

## Immutable science and effective environment

The intended model bytes are the existing
`submissions/research_v2/train.py` SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
`prepare.py` remains
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The future stage must copy those model bytes to the new checkout's root `train.py`
and bind a clean source commit. This package does not edit either file.

The Track-A training suffix is exactly:

```text
--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct
--nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce
--seq-len 1024 --no-eval-public --matrix-lr 0.015
```

The later finalized request must use `env -i` and capture the **effective**
`NEURON_LOGICAL_NC_CONFIG=2`, `NEURON_CC_FLAGS=--optlevel=1 --auto-cast
matmult --auto-cast-type bf16`, `NEURON_COMPETITION_R1_SEED=42`, and
`NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800` assignments. The `train.py`
parser reads those last two as the `--seed` and `--max-train-seconds` defaults;
`prepare.py` fixes the budget at `30 * 60`. No extra seed, budget, or step-limit
CLI flag is proposed. This explicit environment is semantically intended to
match `r1`'s authenticated defaults, but it changes the literal environment:
the old G2 receipt does **not** prove this exact command. A new exact-env,
four-rank G2 and independent comparison must pass before any authorization.

Use full-chip LNC2/four-rank `torchrun`, a 1,800-second training target, and
the known guard with a 3,600-second outer bound. Retain only the six existing
35c9 trn22 warm train-cache directories listed in
[GATE_CONTRACT.json](GATE_CONTRACT.json); reauthenticate their path/device/inode
identity after the volume operation. Do not clear, copy, redirect or delete
caches. A cold fallback is not approved. The future unique checkout, output,
step file and logs must contain `-r2`, must be absent before staging, and must
not overwrite `r1`.

## Provisional disk projection

The immutable operational floor is **6,979,321,856 bytes (6.5 GiB)**. The
observed `r1` compile-only growth was **1,500,942,336 bytes** before any
checkpoint existed. A retained dense checkpoint example is **469,831,009
bytes**. An independent prior dense package reported a **1,854,160,896-byte**
complete-run footprint. The provisional arithmetic is:

```text
6,979,321,856
+ max(1,854,160,896, 1,500,942,336 + 469,831,009)
+ 1,073,741,824  (unknown/log reserve)
= 10,023,837,025 bytes
```

Round **up** to a required fresh pre-dispatch root free space of **10 GiB =
10,737,418,240 bytes**. This leaves 713,581,215 bytes beyond that provisional
formula. Neither observation proves a maximum compile or training footprint;
the 10-GiB threshold is a conservative *gate*, not a guarantee. The Operator
must monitor free space throughout compile and training, and preemptively stop
only the authenticated owned guard if space approaches the 6.5-GiB floor.
Any below-floor sample is a failed safety gate. Never lower either threshold to
increase utilization.

The [disk decision packet](../../v2/DISK_DECISION_PACKET_2026-09-25.md) asks
Poojith to expand both existing root volumes from 200 to 250 GiB and verify the
guest partition/filesystem. **Only the human takes AWS/resize actions.** The
`trn22` breaker then requires the human's explicit, recorded acknowledgement
and reset after the reviewed diagnosis; volume expansion alone is insufficient.
Do not infer reset from an edited `state.json` field without human provenance.

## Gate order; each missing item is a HOLD

1. Human volume/guest resize is confirmed. A fresh read-only `df -B1 /`
   report shows at least 10,737,418,240 free bytes; verify root device and
   filesystem, and preserve the receipt.
2. Human explicitly acknowledges the `r1` incident and resets the
   `trn22` breaker. `research/v2/state.json` must agree with that recorded
   human action. No agent resets the breaker.
3. A different Verifier reviews the new proposal, disk arithmetic, unchanged
   science bytes, previous incident, cache policy and role boundaries.
4. Stage a **new** clean source checkout at an exact commit and rehash root
   `train.py`, `prepare.py`, launcher/tool files and proposal/contract.
   Materialize a new request only after staging; it must have
   `dispatch_authorized=false` until a separate controller decision binds
   its complete SHA and one-attempt authority. Never recycle `r1`'s request.
5. Rebind trn22 G1 evaluator calibration and run a new G2 against the exact
   `env -i` assignments plus training argv above. Re-run CPU numerical
   comparison; the previous G2 is background evidence, not an exact-env pass.
6. The Operator's live preflight verifies hostname, clean commit, all hashes,
   six existing warm caches, absent unique r2 paths, idle process/device,
   shared lock, source/effective environment and at least 10 GiB free. Start
   the separately staged read-only boundary observer *before* the guard;
   prove it is alive and the step file is absent.
7. Only after those gates may a Supervisor issue a separately recorded,
   exact-request, one-attempt authorization. An Operator alone may execute it.
   A failed r2 run can only be followed by diagnosis and a new `-r3` label.

## If a separately authorized r2 eventually runs

The Operator preserves every log, guard/process/postflight receipt, cache
inventory, observer output and checkpoint; proves quiescence; and verifies
five consecutive integer optimizer boundaries beginning at 0. The boundary
file contains **no loss or timestamp**; use the training stdout log for the
step-0/1 loss anchor `9.0112`/`8.7822`, and the external observer plus
budget summary for timing. Require a finite, hash-identified model-only
checkpoint and no budget breach. Any crash, NaN, missing artifact, causality
failure or below-floor disk reading is a failed gate, not a near-win.

Fresh `prepare.py eval-public` at context 1024 over 20,971,520 tokens,
causality checking and any checkpoint retention decision require their own
reviewed post-training gate. The predeclared global numerical-equivalence
band is `[0.99095, 0.99295]`; it is **not** a historical trn22 public20M
control value. A passing A0 establishes a new c5c control; it does not itself
show a BPB improvement, authorize A1/A2, adoption, or submission.

No host action, AWS action, protected deletion, breaker reset, launch,
fresh evaluation, or leaderboard submission has been performed by this
Implementer package.
