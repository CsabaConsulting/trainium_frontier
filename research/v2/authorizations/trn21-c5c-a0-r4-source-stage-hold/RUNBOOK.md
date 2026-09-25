# trn21 clean 446b safety-source stage — Implementer HOLD plan

**No host contact or launch is authorized by this package.** It supplies
trn21-specific targets and immutable source/protected-asset pins, not a
transfer writer, SSH command, checkout invocation, Verifier PASS, Supervisor
assignment, or Operator preflight. The accepted e65 checkout and its future
CPU G2 remain separate and untouched.

## Why a new checkout is necessary

The accepted e65 checkout is correct for the c5c A0 **science**, but its
launcher/guard/host safety code predates the reviewed paired disk-monitor
interlock. The e65 launcher does not gate `dispatch_authorized` or live
monitors; its guard has the old process-group signal route. The 446b source
has the same `train.py`, `prepare.py`, `fresh_start_sync.py` and
`run_experiment.py` bytes, and changes only the three safety files plus
three added interlock/monitor files listed in `GATE_CONTRACT.json`. A
reviewed *new* clean 446b checkout is preferable to editing e65 in place;
the latter would dirty an independently accepted source checkout and break
its existing G2/request bindings. No native A0 run should target e65.

## Frozen reusable local source; no new build needed

The existing one-commit capsule at
`/home/poojith-devan/Desktop/Trainium-source-artifacts/trainium-c5c-r4-446b-one-commit-20260925.tar.gz`
is 104,154,875 bytes, SHA-256
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`,
mode 0600, single-link and owner 1000. Its manifest is the already-reviewed
1,229,617-byte `trn22-c5c-a0-r4-source-stage/MEMBERS.jsonl`, SHA-256
`7ec8798c3e6d0284a4514a15a0caeb17bc5eb2cb5df935fb7dd6e8f3aabee42a`.
The independent capsule scan covered all 6,368 archive members and exact
detached HEAD `446b657695f086f9b484fa601c9a3b61c2495912`. That PASS is
local-artifact evidence, **not** trn21 transfer/checkout authority. A fresh
independent trn21 plan review must rebind the capsule and host-specific
targets. The package commit is not the frozen source commit.

Proposed, never-created trn21 targets are:

- Persistent root-ext4 capsule:
  `/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn21-stage-r1.tar.gz`
- New private checkout:
  `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r4`
- Later separately staged monitor directory:
  `/home/ubuntu/trainium_a0_disk_monitor_20260925_trn21_r4`

The previously accepted e65 checkout is
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`
(device 66305/inode 3168319). It must remain unmodified. The old failed
`/dev/shm` stage path is never reused.

## Gate 1 — separately reviewed transfer-only operation

Before an enabled transfer request is even considered, a distinct Operator
must capture a fresh, full, read-only trn21 preflight: pinned SSH host key,
hostname/UID, boot and mount namespace, root ext4 device 66305/`259:1`,
safe nonsymlink `/home/ubuntu` ancestry, all three proposed targets absent,
science label absent, no STOP/open breaker/process/Neuron/lock holder, both
protected checkpoints and six warm caches unchanged. Its minimum root free
is **13,177,973,911 bytes** (10 GiB post-stage floor + capsule + unpacked
regular files + 2 GiB uncertainty reserve), with at least 10,000 free
inodes. The 03:17:38 UTC checkout postflight is historical, not this
preflight. No cleanup/resize/breaker reset is implied.

Only after separate host-specific implementation and independent review may
a Supervisor assign **one transfer attempt**. Reusing trn22's transfer
writer design requires new literal trn21 SSH identity/argv, request/review/
assignment hash chain, fixed private O_EXCL/fsynced local journal, exact
capsule SHA/bytes, no-follow exclusive ext4 mode-0600 target, file and parent
fsync, same-descriptor SHA readback, a different SSH session's readback, and
an unconditional different-session full read-only postflight even on failure.
Archive raw argv/exit/stdout/stderr, PID/birth, target inode/device and
host/boot/mount identities. A partial target is consumed and preserved;
never retry, rename, replace, or delete it. A different Verifier must
accept the retrieved transfer result. The trn22 wrapper is *not* directly
dispatchable on trn21.

## Gate 2 — separate checkout-only operation

After transfer result PASS, another Implementer may adapt the previously
reviewed trn21 descriptor-relative `stage_r2_checkout.py` to the frozen
446b capsule and its **different** 6,368-member manifest. It must scan the
complete gzip/raw tar stream and member hashes before creation, reject
links/devices/PAX/traversal/duplicates/unsafe Git metadata, create only the
new absent 0700 checkout with O_EXCL|O_NOFOLLOW files, and rehash the same
persistent capsule descriptor afterward. The result must prove exact
one-commit shallow detached HEAD 446b, no remote/alternates, `git fsck`,
clean tracked/untracked status, and all pinned train/prepare/safety hashes.
The two checkpoints, six caches and e65 checkout must have identical
before/after identities. A full read-only postflight is unconditional even
if extraction fails; partial checkout remains in place, no same-path retry.
This stage needs its own exact request, independent review, fresh preflight,
one-shot Supervisor assignment, Operator receipt, and independent result
review. This package supplies none of them.

## CPU G2 and native remain later gates

The e65 four-rank CPU G2 is currently pending. If it later passes, a
separate Verifier can consider a **source-identity carryforward** to the
new clean 446b checkout, because `train.py`/`prepare.py` bytes match. The
Verifier must also compare exact ordered training argv, seed, LNC/world,
compiler/effective environment, CPU harness bytes and scored-code import
surface. If any differ, run a new four-rank G2 on the 446b checkout and
independently review its numeric result. Neither source equality nor this
plan grants G2 acceptance or native authorization.

Only after that may a final, independently reviewed trn21 446b native
request bind a fresh source checkout, a new request→base→pair monitor chain,
an authenticated paired monitor and read-only optimizer-boundary observer
already live before guard, 10-GiB prelaunch/6.5-GiB floor/8.5-GiB stop
thresholds, protected assets, no STOP/breaker/holder, and a single Supervisor
one-label/one-attempt assignment. Training, fresh public20M evaluation,
adoption and submission are separate later decisions.

The companion local tests verify source/capsule identities and disabled
fields only. They do not touch trn21 or independently approve this plan.
