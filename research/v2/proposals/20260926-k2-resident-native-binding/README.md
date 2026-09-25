# Disabled K2 resident-consumer native binding

Preparation only. Independent exact-source, genuinely fresh reviewer and
mechanical binding gates remain pending, followed by a separate one-shot
Supervisor assignment to the sole Operator. No host is reserved; architecture
decision-tree/CT work may preempt this diagnostic. No launch, staging, tensors,
real data, training, eval, queue mutation or submission action occurred here.

## Exact frozen workload

Candidate module commit: 077481d0476cebfd79ddc751f8f63e61c1fbf82f.
The three original files are extracted unchanged from
research/v2/lanes/k2-resident-consumer/:

- consumer_nki.py: 57b5b118e86eb9d78e26a011b276904aed9065868b8a12f39169e1257c3a1843
- check_native.py: 17e2d2ad417407c12327e1446aed857f273d445946839357102e93dea46d4435
- reference.py: 14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f

No arithmetic, source-checker exactness, kernel code or checker arguments
changed. The experiment remains a supplied-synthetic-metadata consumer,
V256/C128/T256, not production routing or an actual model. e4d93 stays REFUSE.
Any native unsupported operation or byte difference still fails; no tolerance
or eager-production fallback is added.

New rank_entry.py only records a whitelist of each worker's declared environment
before imports and then runs the exact checker as __main__, with original argv0
and identical arguments. It does not reconstruct/modify the kernel or reference.
It retains its own worker PID and terminal code. These receipts are not proof
that the SDK honors every cache variable; actual compiler/cache evidence remains
an Operator retrieval and independent-review requirement.

## Derivation and minimal future route

The host driver derives from the already-working guarded P diagnostic driver,
pinned in NATIVE_BINDING.disabled.json. The full delta is archived. Shared
idle_guard.py and dispatch_driver.py stay byte-identical. This packet adds no
queue, authority service, retry engine or general launcher.

Proposed label: fs-20260926-k2-resident-consumer-trn22-r1.
Only proposed host: trn22 / ip-172-31-32-253.
Root:
 /home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-k2-resident-consumer-trn22-r1-packet

After every pending gate and explicit assignment, the sole Operator uses the
existing manual immutable staging/dispatch route:

1. Preserve active jobs. Check hostname, historical queues/launchers stopped,
   STOP/breaker state, disk (driver retains 10.5 GiB headroom threshold), unique
   absent root and all file/command hashes. Never overwrite a prior attempt.
2. Stage exact frozen three source files, this packet's driver/rank entry/
   completion checker, and current unchanged shared guard/dispatch. The
   manifest and binding provide pins; the original source is not duplicated.
3. Invoke the bound dispatch command only after assignment:
   /usr/bin/python3 -I -B <root>/dispatch_driver.py.
   It invokes the SAME persistent lock:
   /usr/bin/flock -n /home/ubuntu/.cache/trainium_freshstart/device0.lock /usr/bin/python3 -I -B <root>/host_driver.py.
   Never replace that lock inode or launch bound-command.txt directly.
4. Under the inherited exclusive lock, the driver verifies host/STOP/disk/pins
   and absent caches/metrics/receipts. It creates six new empty private cache
   directories, recording paths/device/inodes and initial emptiness. Compiler
   URL equals compiler DIR. No data cache is touched.
5. The unchanged guard authenticates the inherited FD/outer command and performs
   fresh ps AND neuron-ls immediately before the single child. Unknown, busy,
   failed or malformed checks refuse and log. The same FD is passed to the
   synchronous timeout child; there is no check/spawn release/reacquire gap.
6. Run exactly one four-rank torchrun diagnostic, bounded by timeout 600 seconds
   and a 60-second TERM grace. The environment is a clean allowlist, LNC2,
   seed42, optlevel1 and the original BF16 matmul autocast flags, CPU threads1.
   It enables per-rank graph_stats under the new root. No train.py/prepare.py,
   actual data, evaluator or quality checkpoint is used.

The new Python driver uses explicit exceptions; it is also tested under -O.
The preserved outer argv remains -I -B. No change to launch interpreter mode
or environment may be inferred from a local optimized-Python test.

## Completion is narrower than correctness acceptance

A zero child exit triggers the packet-local completion checker. It requires,
on every rank:

- Correct source hashes/label/argv/world/shape/seed and exactness criterion.
- All 20 prescribed case/dtype pairs, including empty direct zero-trip cases.
- Every expected comparison: 150 exact, finite byte comparisons per rank.
- All four consumer timing entries, three finite positive samples each.
- Exact captured allowlisted environment and clean wrapper exit.
- No missing/STARTED/refused case, missing gradient comparison, nonfinite,
  changed dtype/hash, contradictory max error or accepted-speedup assertion.

Only then is terminal status
TINY_DIAGNOSTIC_COMPLETE_PENDING_INDEPENDENT_ARTIFACT_REVIEW.
This is not source approval, actual-model/native G2, routing support, cold
training qualification, a speed win or adoption. Source/fresh/binding approval
must precede any launch and independent artifact interpretation still follows.

Component timing excludes metadata construction/upload. The original checker
reports its local diagnostic clock; the driver separately records complete
timeout-wrapped child wall and UTC boundaries. Neither is a 1800-second quality
run or official training-budget receipt. Compilation inventory/coldness cannot
be inferred from zero metric compile times or environment strings.

Retrieve raw dispatch/lock/preflight/cold-cache/launch/log/child-exit/terminal,
all rank environment/wrapper/checker receipts, and all per-rank graph_stats/
compiler logs. Missing runtime cache evidence or graph inventory remains
visible and unproved, not fabricated by the mechanical completion checker.

After timeout/failure, the sole Operator must prove actual worker/compiler/
device quiescence before any later label, including these recorded worker PIDs
and rank_entry.py/check_native.py descendants. The inherited FD is not a proof
that arbitrary deeper orphaned descendants have exited. No automatic retry,
cleanup, full run, eval or next job is implemented or authorized.

## Local preparation checks

83 stdlib checks pass normal and -O. They pin source/guard/dispatch/support
bytes and command/environment equivalence, exercise refusal cases against
synthetic in-memory receipts, and exercise driver control flow with ALL
filesystem/process/guard effects faked. No real driver process or native import
was executed. The original test initially tried to read source files from this
older worktree rather than their explicitly pinned commit; both failed lookup
attempts are retained, then fixed to read the frozen Git blobs. No scientific
test or host attempt failed.

Night-mode decision (20:53:12 UTC): prepare an isolated reversible disabled
binding while hosts remain busy; reason is avoiding future packet delay without
waiving gates. Undo by declining packet. Supervisor owns the canonical log.
