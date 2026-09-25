# trn22 c5c A2 m64/d128 CPU G2 — final-pending one-attempt package

**Status: HOLD, no dispatch.** This is a distinct Implementer package for an
exact-size, CPU-only G2. It does not authorize SSH, source transfer, synthetic
CPU execution, native Neuron training, public evaluation, AWS action,
cleanup, retry or submission. Neither fixed-path independent final review
nor Supervisor assignment exists. The request, binding and direct wrapper
entry point all say no dispatch. The package was built and tested offline;
no host was contacted.

## Frozen basis and new label

The [r3 disabled A2 plan](../../g2-probes/fs-20260925-g2-a2-c5c-m64d128-trn22-r3-hold/RUNBOOK.md)
received an independent static PASS, but **only** for its offline safety
repair. This package derives the same ordered 27-token A2 argv from that
pinned plan: width 1024, bigram+trigram, layers `1,-1`, table multiplier 64,
dimension 128, replicated RMSProp, matrix LR 0.015 and the frozen c5c flags.
It uses a **new**, unused label
`fs-20260925-g2-a2-c5c-m64d128-trn22-r4` and private scratch root
`/home/ubuntu/trainium-v2-g2-a2-c5c-m64d128-20260925-trn22-r4`.
The intended source is the existing trn22 clean e65 linked checkout; there
is no source copy. The c5c `train.py`, unchanged `prepare.py`, queue, G2 tool
and containment tool are SHA-pinned. The exact environment is clean `env -i`,
seed 42, LNC2, four ranks, two synthetic CPU steps, optlevel-1 compiler
flags, real `prepare.py` import, `--trace-rmsprop`, `--keep`, and no
`--real-data`. No BPB score can result from this diagnostic.

`INVOCATION_BINDING.FINAL_PENDING_REVIEW.json` fixes the request, runner,
remote adapter, scratch guard, command arrays, strict-host-key SSH arrays,
payload and one-attempt marker by SHA-256. `cpu_g2_invocation_final.py`
recomputes every binding offline. An independent Verifier must review these
exact bytes and write an exact fixed-path PASS receipt; only a Supervisor may
then write the separate exact one-attempt CPU-only assignment. Neither
future file is made by this Implementer. The current `research/v2/state.json`
still has general launch disabled; this package does not change it.

## Capacity policy and unknown live values

The positive conservative planning bounds are 24 GiB scratch peak, 6.5 GiB
protected root floor, at least 32 GiB fresh root free, at least 10,000 fresh
free inodes, 80 GiB projected peak RAM and at least 96 GiB fresh
`MemAvailable`. These are **policy bounds, not measured A2 peaks**. The
source's 8,192-token padded vocabulary gives four m64/d128 bf16 tables of
`4 × (64 × 8192) × 128 × 2 = 536,870,912` table-weight bytes per rank;
the four-rank CPU trace also keeps dense gradients, fp32 RMS state and cloned
pre-update tensors. The bounds allow far more than those table tensors and a
retained rank-0 checkpoint, but an exact-size measurement remains the purpose
of G2 and may disprove the projection. The
07:33 UTC snapshot is historical and does not establish today's free space,
memory, boot/mount, source cleanliness or occupancy. `fresh_host_values`
remains null in the request. The full read-only audit obtains and validates
fresh values; the run SSH process repeats those gates immediately before
any scratch creation. If measured or independently reviewed evidence says a
bound is inadequate, stop and revise/review a new request—do not lower a
floor during an attempt.

## Future one-attempt sequence, only after actual review and assignment

1. The Operator imports the exact wrapper; its direct command-line entry
   point always exits `DISPATCH HOLD`. `operator_once()` first no-follow reads
   the fixed independent review and Supervisor assignment files, requires
   exact schema/byte-chain equality and checks local STOPs plus global/trn22
   breakers. Missing or altered files stop before any marker or SSH.
2. It creates exactly one owner-private marker at the fixed path with
   `O_EXCL|O_NOFOLLOW`, fsyncs file and parent, and never removes or retries
   that label. No other run may occupy trn22. The first strict-host-key SSH
   process runs a full **read-only** audit: hostname/boot/mount/250-GiB root,
   free bytes/inodes/RAM, all STOPs, process scan, host lock, Neuron device,
   label absence, exact scratch absence, clean e65 source checkout, `.git`
   pointer SHA/mode/inode, resolved linked-gitdir identity, train/prepare
   and G2/containment tool hashes. Same-UID and privileged races are not
   assumed away.
3. After rechecking actual authority and local breakers, one second SSH
   process repeats the full audit. It creates only the new root and `tmp/`
   via the pinned no-follow scratch guard. That guard rejects Python's
   default tempfile fallback. A second fresh source/STOP/occupancy audit
   follows in the same SSH process. A no-subprocess pre-exec bridge checks
   boot, mount, STOP, linked Git and source/tool bytes before the contained
   launch. Inside the G2 launcher's own interpreter, the guard checks and
   pins exact TMPDIR before invoking the unchanged G2 tool. The Operator
   uses only the frozen contained launch, wait, failure-stop and quiesce
   arrays; no native/eval command exists.
4. A **different SSH process** performs the full read-only postflight
   unconditionally after attempted host contact, including failed audit,
   transport, scratch, run, timeout or parse paths. The marker, partial
   scratch and logs remain on failure. No same-label retry or cleanup is
   authorized. The local journal durably records raw SSH outputs and the
   decision path.

## Evidence and later independent decision

The CPU helper retains a single `g2-*` workdir and `final.pt`, records the
aggregate G2 receipt, all four raw rank JSON records, all **eight ordered
RMSProp records per rank**, the exact `0\n1\n` step log, run logs and
checkpoint hash. The aggregate must contain the same four raw rank records
and a CPU-only two-step, four-rank PASS to be kept as raw evidence. Even then
the package returns `numeric_gate: HOLD_INDEPENDENT_PER_TABLE_PROOF`:
the unchanged G2 tool's RMS records lack table IDs. A different independent
reviewer must map trace order to b1/t1/b5/t5 for every rank and both steps
using pinned source and raw files, or a separately reviewed tool change is
required. No result here proves Neuron correctness, A2 quality, a valid A0
control, a BPB improvement or submission readiness.

Offline tests (no transport) run with:

`python -B -m unittest research/v2/authorizations/trn22-c5c-a2-m64d128-cpu-g2-final-pending/test_cpu_g2_final.py -v`
