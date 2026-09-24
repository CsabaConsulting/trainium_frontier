# Pending c5c A0 trn21 control — safety runbook

Status: **content-complete proposal; conditional dispatch HOLD**. The planned
label is fs-20260925-v2-c5c-a0-control-trn21-r1. This package is
Implementer-prepared for the Supervisor-selected A0 control. It is not a
final run request, Verifier receipt, human volume receipt, Operator preflight
or one-shot launch authorization. No host was contacted to prepare it.

## Purpose and evidence boundary

A trn21 A0 run on the pinned research-v2 source is needed before later
Track-A trn21 quality decisions can use a same-host c5c public20M control.
The archived trn21 35c9 LR0.015 control completed 3,636 steps and screened
at 0.9866973734526626 public2M BPB. Its step-0/1 training losses were
9.0112 and 8.7822. Evidence is on the historical trn21 control branch at
b0b23cbc. That run has no public20M result and used different source bytes;
it is a numerical anchor, not the missing c5c control.

The trn22 c5c r1 attempt stopped during compile before optimizer step 0.
Its terminal receipt records no checkpoint or BPB. The separate
[trn22 r2 package](../20260925-v2-c5c-a0-control-trn22-r2/GATE_CONTRACT.json)
is also pending. This trn21 package uses the **same immutable science** and
a first, locally unused trn21 label. Before drafting, the exact label was
absent from the local ledgers, decision log, research/workflow/submission/
report/run records, archived host evidence, filenames, local commit messages
and matching branch names. No live trn21 checkout, output, step or log path
was inspected; the Operator must prove those paths absent before staging.

## Immutable 1,800-second science contract

The intended source is submissions/research_v2/train.py, SHA-256
c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586.
The trusted prepare.py remains SHA-256
5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb.
A future, separately reviewed stage must copy the source bytes exactly to
the new checkout's root train.py, pin the clean source commit and rehash
the launcher files. This package edits no source, data or scoring file.

The Track-A training suffix is exactly:

~~~text
--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct
--nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce
--seq-len 1024 --no-eval-public --matrix-lr 0.015
~~~

The future finalized request must use a fully captured env -i environment
with NEURON_LOGICAL_NC_CONFIG=2, NEURON_CC_FLAGS="--optlevel=1 --auto-cast
matmult --auto-cast-type bf16", NEURON_COMPETITION_R1_SEED=42 and
NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800. The pinned train.py parses
the last two variables as the seed and max-train-seconds defaults. The
future process uses full-chip LNC2/four-rank torchrun and the established
guard with a 3,600-second outer bound. No step-limited or 900-second smoke
is substituted. An old G2 receipt does not authenticate this exact clean
environment or this host; a new exact-env, four-rank G2 and numerical
comparison are required.

The historically observed warm training-cache root on trn21 is:

~~~text
/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train
~~~

Its six recorded children are neff, local, hlo, inductor, compiler and
debug. Their current existence, device/inode identity, lock state and
readiness for c5c are **unknown**. The Operator must reauthenticate each
without cleaning, copying, redirecting or deleting it. A cold-cache
fallback requires a newly reviewed plan. Both retained trn21 LR0.015
checkpoints listed in [GATE_CONTRACT.json](GATE_CONTRACT.json) are historical
references to preserve, not training warm starts. Their current existence
and hashes have not been checked for this package. The new A0 output must
use a unique path and must not overwrite either checkpoint.

## Conservative provisional disk gate

The latest archived trn21 inventory had only 6,901,788,672 bytes free,
below the immutable 6.5-GiB operational floor of 6,979,321,856 bytes.
The archived trn21 35c9 control's preflight-to-postflight root-space
decrease was 1,846,603,776 bytes. A historical complete dense footprint
from the other host was 1,854,160,896 bytes. The trn22 c5c r1 compile alone
used 1,500,942,336 bytes before any optimizer step, and a retained dense
checkpoint example is 469,831,009 bytes. The provisional calculation is:

~~~text
6,979,321,856
+ max(1,846,603,776, 1,854,160,896, 1,500,942,336 + 469,831,009)
+ 1,073,741,824  (unknown/log reserve)
= 10,023,837,025 bytes
~~~

Round **up** to a fresh pre-dispatch root-free requirement of **10 GiB =
10,737,418,240 bytes**, leaving 713,581,215 bytes above the formula.
This is equal to the pending trn22 r2 gate, never lower. Neither the old
trn21 35c9 run nor the trn22 c5c failed compile proves a maximum trn21
c5c footprint. Ten GiB is a conservative provisional gate, not a
guarantee. The Operator must monitor root free space during compile and
training and preemptively stop only the authenticated owned guard if it
approaches 6.5 GiB. Any below-floor sample is a failed safety gate.

The [disk decision packet](../../v2/DISK_DECISION_PACKET_2026-09-25.md)
asks the human to expand both existing root volumes from 200 to 250 GiB.
Only the human takes AWS and guest resize actions. A fresh read-only root
device, filesystem and df -B1 / receipt is necessary after that action.
No cleanup, checkpoint prune, cache deletion or threshold reduction is
authorized.

## Gate order; a missing gate means HOLD

1. Human volume and guest-filesystem expansion is complete. Fresh read-only
   evidence confirms the expected trn21 root device/filesystem and at least
   10,737,418,240 bytes free before dispatch. Check the current global and
   trn21 breaker states; any future breaker reset is human-only.
2. The separate [trn21 G1 calibration plan](../../v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r1/RUNBOOK_PENDING.md)
   completes with its own Operator receipt and independent PASS review.
   The trn22 G1 pass does not transfer to trn21. G1 calibration is not a
   new training result or a c5c control.
3. A Supervisor ratifies this conditional proposal. A different Verifier
   reviews the three package files, compares the immutable science with
   trn22 r2, checks the 10-GiB arithmetic and confirms unchanged c5c and
   prepare.py hashes. Implementer self-checks cannot supply that receipt.
4. Stage a **new** clean trn21 checkout at an exact pinned commit. Copy the
   existing c5c source to root train.py byte for byte. Rehash source,
   prepare.py, launcher/tool files and package documents. Prove the unique
   label's checkout, output, step and log paths absent live before creating
   them. Only then materialize a new immutable, hash-bound request with
   dispatch_authorized=false.
5. Run four-rank G2 on the exact env -i assignments and training argv
   above, followed by the declared numerical comparison. Record G2 PASS,
   source/argv/environment identities and independent review. No CPU or
   native training is run by this package.
6. The Operator's live preflight confirms hostname, clean commit and
   complete source hashes; both protected checkpoints stay untouched;
   six historical warm-cache paths and their identities are authenticated;
   the shared lock is free; no train, eval, G2 or guard process or Neuron
   holder is active; all breakers are closed; the stop sentinel is checked;
   and root free space still meets 10 GiB.
7. Stage and start the read-only external optimizer-boundary observer
   **before** any guard launch. Prove it is alive and the new step file is
   absent. A separate Supervisor one-shot authorization must then bind the
   finalized request SHA, human volume evidence, trn21 G1, exact-env G2,
   independent Verifier receipt, stage receipt, fresh Operator preflight
   and complete launch command. Only the Operator may execute that one
   attempt. The existing run_job.py remains pilot-only until separately
   proven; changing launch machinery is not implied here.

## If a separately authorized run eventually occurs

Retain the complete guard, process, training log, disk samples, cache
inventory, observer, postflight and checkpoint evidence. Prove stable
quiescence and at least five consecutive integer external optimizer
boundaries beginning at 0. The boundary file contains no losses or
timestamps; use training stdout for the step-0/1 loss comparison and the
external observer with the budget summary for timing. Require a finite,
hash-identified model-only checkpoint, no NaN or budget breach and no
below-floor disk sample. A crash, missing checkpoint, boundary failure or
dirty postflight is a failed gate and consumes this label.

A separately authorized fresh-process prepare.py eval-public at context
1024 over exactly 20,971,520 tokens, causality check and independent
comparison must follow before the A0 can enter the c5c control set. The
predeclared global numerical-equivalence interval is [0.99095, 0.99295]
BPB; it is not a historical trn21 c5c score. A successful A0 is a
baseline, not an improvement, A1/A2 authorization, adoption or submission.
Any failed attempt requires diagnosis and a new unique label.

This Implementer package performed no SSH, AWS, training, final staging,
protected deletion, breaker reset, dispatch authorization or submission.
