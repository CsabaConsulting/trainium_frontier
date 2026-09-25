# Base initial-cache cold-five binding — DISABLED

Preparation only. No staging, host contact or launch occurred. Exact source
acceptance, full117M four-rank CPU identity, fresh independent source acceptance,
mechanical binding review and a separate Operator assignment remain pending.
Source bcbd is not adopted. No evaluator/full run follows automatically.
Active architecture/seed jobs and historical stopped queues are untouched.

## Frozen identities and exact profile

Candidate b635c8b0cf2a10e1bcdd983e86192672ed6ad56f,
workflow/proposals/research-v2/train.py:
bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522.
Reference a4aa4351789115a2d0aa947d9ca00ff689f9bd0b, same source path:
6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3.

Real prepare remains5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb.
Rehearsal is unchanged00f2ca7f79696e65ce52458f074dd13b871da90c,
research/scratch/submission_rehearsal.py:
7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b.
Later revisions have different hashes; never substitute HEAD.

Existing reviewed guard/dispatch remain
adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737 /
c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a.
New driver:
befa411d78abfe5cc2fc88086245cd20fce0e796ae53cb4d4c3c418abdc640ad.

Only label/root/source literals differ from the working guarded3e79 cold
driver03ba9c6b. Its control flow is unchanged; no older unguarded base/CPU driver
is reused. The complete delta is in driver-vs-architecture.diff.
Package training argv is exactly prior6ef7 cold baseline argv plus
--initial-cache-refresh, with no architecture/attention-layer flags:

    /home/ubuntu/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015 --recursive-schedule --seed 42 --max-train-seconds 1800 --num-steps 5 --out-dir out --initial-cache-refresh

Full6x1024/117453580 parameters, context1024, pack32, global131072 tokens,
grad accumulation1, full-chip LNC2/world4, seed42. Other defaults are frozen
6ef7/bcbd under the bound clean environment. Compiler flags stay
--optlevel=1 --auto-cast matmult --auto-cast-type bf16; adding optlevel1 is not
the experiment. Graph statistics stay enabled as in the cold control.
The unchanged real loader/data are used; no synthetic clock or CPU shim.

Frozen refresh executes under normal grad mode after first-step t0 and before
first forward; post-update refresh/timers remain unchanged. The test records
all args.num_steps references. Fifth stop-control follows record_step_boundary;
the limit is not a model/update/LR formula input. This is static evidence, not
native graph/update identity. Real elapsed time still controls LR/WD/momentum/
beta2, so changed startup may change those values. Fixed-clock CPU identity
does not promise native first-five identity. Cooldown deliberately is incomplete.

## Future Operator route — not an execution instruction now

Proposed host trn22, hostname ip-172-31-32-253; not reserved.
Unique label fs-20260926-ct-basebcbd-cold-trn22-r1.
Root:
 /home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-basebcbd-cold-trn22-r1-packet

After all gates and explicit assignment, use the existing manual staging/
dispatch route, not a new queue or approval tool:

1. Confirm host, STOP/breaker state, stopped historical launchers/paused queue,
   disk>=6.5GiB plus compile/checkpoint headroom. Preserve active jobs. New root
   must be absent and exclusively created; never reuse this label/root.
2. Extract exact candidate Git blob to package/train.py. Stage supplied package
   metadata and host_driver.py plus unchanged guard/dispatch. Verify all pins,
   including launch text/provenance. Driver checks source/prepare/harness/guard;
   package metadata hash binding is an Operator preflight duty. Candidate source
   is not duplicated in this packet.
3. Keep train/ absent. Invoke only the bound detached command after assignment:
   /usr/bin/python3 -I -B <root>/dispatch_driver.py.
   It invokes the exact outer argv:
   /usr/bin/flock -n /home/ubuntu/.cache/trainium_freshstart/device0.lock /usr/bin/python3 -I -B <root>/host_driver.py.
   Never replace the original persistent lock inode.
4. The unchanged guard authenticates parent argv and already-exclusive inherited
   descriptor/inode, then obtains fresh ps AND neuron-ls under that lock before
   the one synchronous job. Failed/unknown/busy checks refuse and log. Driver
   passes that FD to timeout/job; dispatch records outer failures even when
   contention prevents entry. No retry, alternate lock or automatic next job.
5. Archive all four worker identities/effective cache environments as in the
   prior guarded cold run, actual compiler arguments, per-rank graph_stats and
   cache creation/contents proof. Paths alone do not establish effective coldness.

Driver/harness contain historical Python assertions. Bound launch interpreters
are NOT -O; outer -I ignores PYTHONOPTIMIZE and clean child env omits it.
Running this packet's test with -O never authorizes changed launch argv.
The test never imports or executes the driver.

## Fresh caches and accounting

Unchanged harness creates six new absent directories with exist_ok=False and
checks them empty before training: neff/local/hlo/inductor/compiler/debug under
new train/caches/. NEURON_COMPILE_CACHE_URL equals the new compiler DIR.
No shared SDK cache is copied or deleted. Existing data/tokenizer cache stays.
Creation and ordinary static prepare check precede torchrun; neither is a
precompile/prewarm operation.

Clock starts immediately before torchrun Popen. Unchanged rank0 step log emits
0..4; **uncapped startup = observations[4].observed_seconds**, not the capped
allowance or trainer estimate. These are first-observation elapsed timestamps:
50ms nominal polling plus scheduler delay, not exact hardware event times.
Require five sequential lines and all four clean workers.

Record separately:

- Each observation and uncapped fifth boundary.
- Capped allowance=min(fifth,900).
- Torchrun wall through exit, including checkpoint/exit work.
- Charged through exit=wall-allowance, finite/nonnegative and<=1800.
- Source startup_allowance_est/charged_seconds_est under separate labels.
- Outer UTC times/timeout; worker/cache/graph evidence.

Observer excludes pre-torchrun preparation; do not relabel its charge as outer
driver wall. A final log line missed between polls remains incomplete evidence.
Missing boundary/checkpoint, failure/timeout or invalid budget evidence is HOLD/
failure, never inferred acceptance. Outer timeout1800s plus60s TERM grace is
diagnostic containment, not permission to exceed official charged1800 or to
call this a30-minute quality run.

## Interpretation and retrieval

Target<=600seconds against previous same-host6ef7 cold observation855.071994s.
That is a historical startup reference, not a fresh paired quality control.
A target miss stays a miss; a hit only informs a separate Supervisor decision.
No automatic full/eval/BPB/ledger row/adoption. Above870 on base needs the
existing compile-offset assessment before any later full run; preserve the900
allowance cap and other CT rules. Five steps leave too few steady samples:
report intervals/printed throughput with that limitation, not a robust speed
PASS. Zero compiler metric duration is not proof of a cache hit.

Retrieve dispatch/outer-flock logs and exit; under-lock-idle.json; training-launch,
training.log, run-exit-code and terminal; train/rehearsal.json plus steps.jsonl;
checkpoint size/hash/status; all-rank identity/effective-env/cold-cache proof;
per-rank compiler graph_stats. Retain checkpoint on host, no automatic deletion
or large transfer.

Inherited-FD protection covers the immediate job boundary, not arbitrary deeper
descendant deaths: unchanged inner torchrun closes inherited FDs. After failure,
sole Operator establishes actual process/compiler/device quiescence before any
new label. A driver exiting does not prove children are gone.

Night-mode decision (2026-09-25 20:33:55 UTC / September 26 02:03:55 IST):
prepare a reversible disabled binding while hosts stay busy.
Reason: remove future packet delay without waiving gates. Undo: decline packet;
no host/source state changed. Supervisor owns the canonical decision log.
