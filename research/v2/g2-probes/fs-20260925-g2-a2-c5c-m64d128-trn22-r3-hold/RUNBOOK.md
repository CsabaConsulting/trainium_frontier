# A2 exact-size CPU G2 on trn22 — disabled r3 repair request

**Status: HOLD.** This is an Implementer-only, offline request and runbook. It
does not authorize SSH, source staging, CPU execution, Neuron training,
evaluation, AWS action, cleanup or submission. `plan_g2.py` constructs command
*data* for review and refuses direct execution. This repairs the independent
[r2 STATIC HOLD](../../verification/20260925-a2-c5c-m64d128-trn22-r2-disabled-g2-independent-HOLD.md),
not the original science proposal. This r3 package has no independent
Verifier receipt. No trn22 contact, source transfer or G2 result was made here.

## Why this is a new gate, not another A2 experiment

The existing [A2 proposal](../../../submission-validation/20260925-v2-c5c-a2-m64d128-trn22-r1/EXPERIMENT_PROPOSAL.json)
and [static PASS](../../../agentic/verification/20260925-v2-c5c-a2-m64d128-trn22-r1-static-r2-review.md)
already define the four-table m64/d128 profile and its 27-token ordered argv.
They have no full-size, four-rank, two-step CPU/RMSProp receipt or contained
request. This package names a **new** CPU-only label
`fs-20260925-g2-a2-c5c-m64d128-trn22-r3`; it does not re-use either r1/r2
label or change the science proposal. The G2 uses
synthetic tokens while importing the real, unchanged `prepare.py`; it is not
training on public or private validation data, and it produces no BPB.

The planned source is the existing clean trn22 e65 A0-r2 checkout at
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2`.
At 07:33 UTC it held exact c5c `train.py` SHA-256 `c5c410af…` and unchanged
`prepare.py` SHA-256 `5807c46e…`; a separate bounded independent review
accepted that observation. The root source is a linked Git worktree whose
directory and `.git` pointer are mode 0775/0664. No alternate local GID-1000
member or ACL grant was found, but same-UID, external NSS and privileged races
were not excluded. Therefore that old observation is **not** a fresh source
lock, preflight, or permission to run. Its e65 *native launcher* is older than
the reviewed safety interlock and remains barred from native training; this
request is strictly for the independently reviewed CPU G2 tool.

## Frozen intended payload

`G2_REQUEST.HOLD.json` pins the old proposal/static review, queue, source,
evaluator and CPU/containment tool hashes. `plan_g2.build_plan()` independently
derives the A2 argv by replacing the single `--no-ngram-ve` token in the
unchanged queue *in place*, compares it to the reviewed A2 contract, and
constructs a deterministic command-array digest. No shell expansion or
`--real-data`/stub prepare is allowed. The only planned CPU command is the
checkout's pinned `workflow/tools/preflight_dryrun.py` with:

```text
--steps 2 --nproc 4 --micro-rows 2 --micro-tokens 256
--timeout 3600 --trace-rmsprop --keep
--out /home/ubuntu/trainium-v2-g2-a2-c5c-m64d128-20260925-trn22-r3/g2.json
-- <exact reviewed A2 argv>
```

The planned `TMPDIR` is the **new** ext4 scratch root's `tmp/`, so the G2
tool's random `g2-*` work directory, four rank records and `out/final.pt`
survive for independent inspection. A unique two-boundary step file and the
contained process directory are under that same new root. The command data
uses a clean `env -i`, fixed Python/`PATH`, seed 42, 1800-second science
budget, LNC2, four ranks, the required optlevel-1 compiler string, and
`TRN_EXP_LABEL` even after `env -i` so descendant containment keeps working.
`--no-compile` and `--device-type cpu` are appended by the G2 tool, not by
an edited `train.py`. The source checkout, tool files, output root and env
remain **future host bindings**, not proven by this offline plan.

### The r2 scratch and pin defects are repaired fail-closed

The pinned G2 tool uses `tempfile.mkdtemp(prefix="g2-")`. If `TMPDIR` is
missing, Python silently falls back to another filesystem. Merely setting
TMPDIR in the environment does not prevent that. r3 adds a separate
`prepare_scratch_after_marker` command-data operation: after a fresh
preflight and an fsynced one-attempt marker, it must create the exact absent
r3 scratch root and `tmp/` with mode 0700. Its no-follow file-descriptor
checks bind inode, device, owner and mode, and it rejects a tempfile fallback.
It also checks fresh free bytes, at least 10,000 free inodes, and the reviewed
peak projection plus the protected operational floor. A partial root is
preserved after any failure; there is no automatic cleanup or retry.

The contained launch carries those same pinned `scratch_guard.py` bytes as
`python -B -c <source>` **in the G2 launcher's own interpreter**. It rechecks
the exact prepared empty directories, pins `tempfile.tempdir` to the chosen
TMPDIR, rehashes c5c/prepare/G2, then invokes the unchanged pinned G2 tool
with the unchanged A2 argv. Thus the actual `mkdtemp` cannot silently
fall back. The future worker ranks inherit that exact TMPDIR. Both command
arrays currently contain `UNBOUND_PROJECTION`, which the guard rejects:
**this draft cannot prepare scratch or launch**. A separate enabled request
must bind a positive independently reviewed projection after fresh preflight
and recompute the final command digest. No default projection is invented.

`build_plan()` also authenticates the nonzero free-inode floor and the full
observed checkout identity, including `.git` pointer SHA-256/mode 0664 and
resolved gitdir mode 0775. Those observations remain stale and never replace
same-session rechecks.

## Gates before a different role may request CPU execution

1. A separate Verifier reviews this exact package, its hashes, queue-derived
   argv, CPU-only environment, fail-closed validator and tests. A static PASS
   covers the draft only. `research/v2/state.json` remains authoritative;
   both `launch_authorized` and trn22 `launch_allowed` are false in the
   archived handoff.
2. The Supervisor produces a **separate enabled CPU-only request** with an
   exact file/command SHA, actual independent review path/hash and fixed
   fsynced one-attempt marker. It must never infer authority from this JSON,
   standing development consent, an idle snapshot, or the A2 proposal alone.
3. The Operator's fresh trn22 preflight checks hostname, boot/mount identity,
   full 250-GiB root filesystem, free bytes/inodes, `MemAvailable`, no
   train/eval/G2 process, no `/dev/neuron*` holder or host lock, local and
   host STOP/breaker state, unique unused label and absent scratch/TMPDIR/
   receipt/process/step paths. Root free must be at least 32 GiB, free inodes
   at least 10,000, and RAM
   available at least 64 GiB **and** a separately reviewed scratch-peak
   projection plus the unchanged 6.5-GiB floor must fit. The 07:33 UTC
   observation of ~54 GiB free and older ~122-GiB RAM reports are not this
   preflight.
4. After those checks, a positive reviewed scratch-peak projection and the
   fsynced one-attempt marker, use only the exact scratch preparation
   operation. Its receipt must show the requested root and TMPDIR, device,
   owner, mode 0700, distinct inodes and `tempfile_root == TMPDIR`. Any
   pre-existing or unexpected path, failed capacity check or fallback is HOLD.
   Preserve partial scratch and do not retry this label. Do not silently
   replace `UNBOUND_PROJECTION` or run the disabled commands.
5. In the actual run session, no-follow rehash the linked checkout, `.git`
   pointer and resolved gitdir identities; require clean e65 HEAD and exact
   c5c `train.py`, unchanged `prepare.py`, G2 and containment tool bytes.
   Recheck the same identities immediately before the contained child exec;
   any source/boot/mount/STOP change is HOLD. Preserve the two retained
   checkpoints and six warm-cache identities. A simple check in an earlier
   SSH session cannot bridge this mutable checkout into a later run.
6. Operator alone uses the separately reviewed contained launch/wait/failure
   stop/quiesce sequence, one CPU job on trn22. On every attempted preflight
   or run, including errors, take a **distinct-session full read-only
   postflight**. Preserve partial scratch and the attempt marker; no same-
   label retry or cleanup. A containment survivor/device holder trips the
   correctness gate.

## Required numeric and artifact review after a future run

The aggregate G2 receipt must be `PASS`, exact c5c/prepare hashes and ordered
argv, four CPU/gloo ranks, two finite losses per rank and a common ownership
plan digest and identical full ownership summaries. Each rank must report
exactly tables `b1,t1,b5,t5`, all changed, no cross-rank mismatch, two
RMSProp group calls and eight update checks: four at step 1 and four at step
2, 67,108,864 elements per check, 1–524,288 touched rows, unchanged
untouched rows, finite medians, and recomputation
error divided by LR at most 0.03. Rank 0 must strictly load the produced
model checkpoint and return deterministic finite logits of shape
`[1,256,8192]` with final digests for all four tables. The isolated step
file must contain exactly `0\n1\n`.

`plan_g2.validate_g2_receipt()` and `validate_step_log()` fail closed on those
structural/numeric checks. **Even an otherwise clean receipt returns
`numeric_gate: HOLD`.** The pinned G2 tool's RMS check records lack table
identifiers. Four identical-size checks per step do not prove that b1, t1,
b5 and t5 were each updated; a forged `table_name` field is rejected. A
later independent reviewer must establish a raw, source-anchored mapping of
trace order to each of the four tables for each rank and step, with raw rank
file hashes and an explicit independent proof. If the mapping cannot be
established from raw traces and pinned source, the CPU numeric gate stays
HOLD pending a separately proposed and reviewed tool revision. No implementer
may grant its own numeric PASS.

The validators are **not** a Verifier: the later
independent reviewer must rehash the retrieved raw G2 receipt and all four
rank JSON files, prove each raw rank record equals the aggregate, hash the
retained `out/final.pt` and raw run/step logs, check the contained launch and
quiescence receipts, and reconcile the full postflight/protected assets.
The aggregate tool verdict alone is insufficient. No exact-size A2 CPU PASS,
Neuron smoke, valid same-host A0 control, A2 quality result or BPB gain exists
yet. A later native smoke and full run each need their own reviewed request
and one-shot assignment; this CPU plan cannot authorize either.
