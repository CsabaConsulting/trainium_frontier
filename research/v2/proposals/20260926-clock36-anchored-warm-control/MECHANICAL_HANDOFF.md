# Disabled anchored exact36 control: mechanical handoff

Status: IMPLEMENTED PREPARATION ONLY. Independent mechanical review and a
separate sole-Operator release remain required. No host is reserved, staged or
launched. Clean submission qualification retains priority. This packet does not
approve its source, result, control eligibility, cache reuse or submission.

The completed Researcher design is preserved byte-for-byte from16d14c8f.
The new binding freezes the whole parent and proposed argv plus both complete
training/evaluation environments. Training argv uses **zero-based script index3**:
that alone changes from relative train.py to the authenticated completed parent's
absolute train.py. Compiler flags, scientific flags, all six retained training
cache roots and separate evaluation cache roots are unchanged. The only
environment differences are the new TRN_EXP_LABEL in each phase and the new
TORCH_NEURONX_METRICS_DIR in training. Full harness vectors differ only by the
new package/train/eval/checkpoint root. All vectors and allowed differences are
explicit data in ANCHORED_BINDING.disabled.json, not inferred at launch.

## Composition and unchanged code

host_driver.py composes the existing full-control preflight() and phase()
functions unchanged (AST identity tested against the completed trn21 exact36
full driver336de641). The original guardadafb54a, dispatcherc71474bc,
harness7a55cb87, prepare5807c46e and training source36a86274 are pinned and are
not edited or replaced. identity_checks.py is an unchanged copy of W1 helper
85eef9549. Only identity() and no_symlink() are used. W1 inputs(),
validate_phase() and capture_inputs() are NOT called.

The new composition adds explicitly reviewed path/file snapshots, an exact
hostname/original-lock check, a full-run through-exit accounting check, a
non-authoritative closed graph-file inventory, and final boundary receipts.
It is not a byte-identical whole driver. driver-vs-parent.patch is its complete
zero-context diff against the exact parent (full source is included separately). No new scheduler, automatic successor,
compiler-input manipulation, source wrapper or import shim is added.

The unchanged full preflight contains Python assertions. The frozen invocation
uses normal /usr/bin/python3 -I -B, and this composition explicitly refuses an
optimized interpreter before any child. Own test execution under -O still
compiles the pinned normal-interpreter driver at optimize=0 to test the actual
launch semantics and separately checks optimized-runtime refusal.

## Exact present-time anchor contract

The raw03:25:58UTC observation pins train0444 and prepare0600 by
dev/inode/mode/uid/size/mtime_ns/ctime_ns/SHA. This is not W1's readonly-directory
contract. No chmod, rename, rewrite, copy-over, repair or historical directory
write is permitted. At runtime the unchanged individual-file operation adds
gid/nlink, rejects symlinks through every ancestor, rejects hardlinks, and
authenticates opened-file before/after identity. Every returned file field must
remain equal at pretrain, posttrain, preeval, posteval and final snapshots.
Reading may change atime; it is not an observed/invariant field.

The historical script directory and its old packet parent must retain ALL
captured directory fields: dev,ino,mode,uid,gid,nlink,mtime_ns,ctime_ns.
For every other source ancestor, dev,ino,mode,uid,gid must remain equal.
Their nlink/mtime_ns/ctime_ns are recorded but may change because the shared
out/project/home ancestors can acquire new packet/normal entries. No other
difference is discretionary. The binding lists these field sets explicitly.

Static script-directory resolution must resolve prepare to the exact colocated
prepare.py using PathFinder. Any other importable module/package matching the
source's complete AST-derived top-level import-root set, or sitecustomize or
usercustomize, is refused. A prepare package/wrong origin is refused. No
PYTHONPATH/PYTHONHOME and no bytecode writes are allowed. This establishes only
the ordinary source-file script-directory expectation; actual worker import
origin, installed import hooks and absence of transient malicious writes are
not observed/proven. prepare0600 is not OS-immutable. The original proposal
requires this distinct operational no-writer contract to be independently
accepted; otherwise the packet is blocked, not repaired.

## Future staging and execution contract — not authority

Only the sole Operator after a separate release may use the exact disabled
binding's dispatch_command/outer_command. Preserve the original lifetime
device0 lock66305:596823; the unchanged guard authenticates the inherited FD and
fresh ps AND neuron-ls at initial, pretrain, preeval and final boundaries.
Busy/unknown/error refuses. STOP/disk>=10.5GiB/cache-directory checks are reused.
One synchronous training child2730s+60s, then one attached eval1800s+60s only
after successful training, identity, checkpoint and accounting checks.
No automatic other job, second population, W1/S2 rearm, retry or full-run chain.

A future Operator must authenticate the absent unique packet root before
staging; exact old anchor/raw identity; runtime versions; clean tracked source;
retired launcher absence and unresolved breakers; all manifest/source/helper/
guard/dispatcher/harness pins and hostname. Create only new package/metrics
files/directories, with package files born readonly by the existing staging
procedure; do not change old file modes. Stage train.py from exact7419927c blob,
not the canonical research checkout. Stage unchanged idle_guard.py and
dispatch_driver.py from the common-host-idle-guard directory. Stage unchanged
proposal/audit/helper files pinned by support. Root train/eval sandboxes must
still be absent and new metrics directory empty when the driver starts.

The rehearsal creates byte-identical train/prepare sandbox copies and leaves
literal --out-dir out and step-log injection in the fresh train cwd. Training
executes the old absolute script; attached eval runs its ordinary fresh sandbox
copy with unchanged fresh20971520 tokens/B1/context1024. No actual training
source is committed into this packet or modified.

The common final idle guard is an immediate check, not a proof about every
previous descendant. The Operator must retain actual dispatch/driver/timeout/
harness/torchrun/rank/compiler/evaluator PID,starttime,PPID,PGID,SID evidence and
explicit all-observed-identity absence, device idle and original-lock release
before result use or another job. The driver does not invent a new process
controller or claim that parent exit proves quiescence. Terminal status remains
COMPLETE_PENDING_INDEPENDENT_RESULT_REVIEW, never native/source/quality PASS.
Unavailable graph files are disclosed and leave evidence completeness pending.

## Accounting and target

The exact unchanged source and harness retain1800 official charged seconds,
capped900 startup allowance, existing stopping/checkpoint reserves, and full
numsteps1000000. The driver recomputes actual process-wall minus capped observed
fifth through exit and rejects contradictions/nonfinite/budget failures.
Operational uncapped fifth<=180 is separately WARM_TARGET_MET; >180 is
WARM_TARGET_MISSED but does NOT stop an otherwise valid full run or its attached
evaluation. A low fifth is not a BPB win. Historical W1 remains a miss of120s.
The fifth observer is not source step-start time; no update/LR/WD identity is
claimed between warm and cold runs.

No native cache-hit, compile reuse, forward/backward graph labeling, or
pathname-only causality is established. SDK writes in the retained cache roots
are allowed; no cache walk, seeding, clearing, deletion or alteration is added.
Closed graph rows and raw logs are evidence for later analysis only.

## Own bounded checks and remaining gates

Run locally (stdlib only):
    /usr/bin/python3 -B research/v2/proposals/20260926-clock36-anchored-warm-control/test_packet.py
    /usr/bin/python3 -O -B research/v2/proposals/20260926-clock36-anchored-warm-control/test_packet.py

Seven test methods with many negative subcases pass both modes. Real tiny
filesystem fixtures test0444/0600, opened identity, wrong stat/hash, symlink,
hardlink and module shadowing. Pure fixtures test precise directory invariants,
inclusive180/1800 boundaries, budget/NaN failures, inherited preflight failures,
pass_fds, and13 mocked success/failure compositions. Mocks never execute any
child or device command. No test failures occurred; raw normal/-O output is
retained. No tensor/model/SDK import, data read, host operation or actual source
execution was performed.

These are Implementer tests, not independent review. Pending: fresh independent
mechanical review, distinct anchor-contract acceptance, source/runtime gate
authentication, higher-priority clean qualification, separate release, actual
preflight, full run/eval and retrieved all-process postflight. No actual cache
or filesystem availability is implied by this local preparation.
