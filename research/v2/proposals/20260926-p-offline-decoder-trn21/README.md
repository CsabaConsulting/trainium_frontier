# Disabled fixed P offline decoder — trn21

Preparation only. No trace or native file was read here, no SDK/model was imported, no host was contacted, and no current job or cache/checkpoint was touched. Proposal-first commit: 3bd0ba36. Independent mechanics review and a separate sole Operator assignment remain required.

## One existing trace, one attempt

The exact source is /home/ubuntu/trainium_profile_recursive6ef7_20260926_trn21_r1/profile/rank0/native. The frozen archived inventory supplies all86 files (4,310,546,731 bytes), including the complete rank0 session and all80 NEFFs (167,863,163 bytes). The proposal preserves every relative path; no arbitrary model pairing, timestamp-leaf-only input, NEFF flattening, new capture or training. Prior inventory sizes are NOT prior content authentication: future assigned copying records every source/destination SHA and identity.

Installed CLI evidence at cd298eba supports the exact directory-mode command:

```text
/opt/aws/neuron/bin/neuron-explorer view --session-dir /home/ubuntu/trainium_p_offline_summary_20260926_trn21_r1/input/native --output-format summary-json --ignore-nc-buf-usage --ignore-instruction-hierarchy
```

It documents summary-json console output without a database. This is not proof that this installed binary ingests the actual nested tree. Runtime records the exact non-symlink binary hash/identity and requires its archived version verbatim; any mismatch/unsupported layout is a failure, not permission to rewrite inputs or retry. No neuron-profile shim, database/UI/server, installation, JSON/Parquet expansion or SDK modification.

## Containment and preserved originals

The original shared guard and dispatcher are reused unchanged. host_driver.py is one fixed offline job, derived from the existing guarded checkpoint/P pattern; decoder_support.py adds fixed copy/hash/pipe/resource checks, not a scheduler or general CLI. Exact inputs are independently copied, hash-checked and made0444/0555; never hardlinked, moved or edited in place. Source and copy hashes/metadata are rechecked after a successful decode. Failure or exhausted time may leave posthash unknown; it never fabricates unchanged evidence. All bounded partial artifacts remain; no cleanup or deletion is authorized.

Hold the existing /home/ubuntu/.cache/trainium_freshstart/device0.lock from fresh in-lock ps AND neuron-ls preflight through all assembly/decode/cleanup/postflight. The inherited FD is retained by children. Stop on a busy/unknown check; do not start until current full75b9 and its attached eval have finished. Require exact hostname ip-172-31-34-85, no STOP, >=24 GiB available memory and >=12 GiB free scratch disk.

The limits are ceilings, not targets:

- 600 seconds overall, work alarm/deadline 580 seconds with cleanup reserve; decoder <=300 seconds, owned TERM then <=10 seconds grace then KILL/reap. If insufficient bounded time remains for the two final idle checks, mark quiescence unknown/REFUSE and require separate Operator confirmation before any later job.
- One inherited CPU affinity and GOMAXPROCS=1. Parent soft CPU alarm 55, hard 60 after decoder fork; version probe hard 5; decoder receives floor(300-60-32-actual previous child CPU), capped 235 and normally at most 207/208 seconds. 32 CPU seconds are reserved for the two final 15-second guard calls. This is deliberately stricter than the 300-total ceiling, not a limit increase. Observed combined CPU is also checked.
- Parent AS soft 512 MiB/hard 16 GiB, allowing child AS hard 16 GiB; decoder file-size hard 64 MiB and core dumps disabled. Virtual address space is not a cgroup total-RSS guarantee. A Go reservation failure remains a diagnostic failure; no larger retry follows.
- Combined retained version+decoder stdout/stderr <=16 MiB, with byte-exact truncation at the limit and process termination. Generated scratch stop threshold 256 MiB, sampled every 0.1 seconds and excluding only authenticated fixed-size read-only inputs; maximum 4096 entries. Record overshoot: this is not a disk quota. Inspection stays in the unique scratch root, not a broad cache/data scan.
- Any unexpected descendant is unsupported and stops the job. Linux CHILD_SUBREAPER is required before spawn so fast double-forks reparent to this fixed single-child wrapper; unsupported prctl refuses. PID/start-time authentication prevents signaling a reused unrelated PID; reparented/new-session descendants are terminated and reaped. This is not a cgroup memory guarantee. Fresh shared postflight/Operator process evidence remains mandatory, with no next job if unknown.

Only the newly owned decoder group/identified descendants can be signaled, never current jobs. The first version process is a bounded capability preflight, not a second decode. New child pipe readers are closed on every exit; all final stdout/stderr remain capped.

## Future staging only after review and separate assignment

New absent root: /home/ubuntu/trainium_p_offline_summary_20260926_trn21_r1. Sole Operator stages only this frozen proposal, host_driver.py, decoder_support.py, binding/manifest, and byte-identical shared idle_guard.py/dispatch_driver.py. No input/home/tmp/output paths may exist. Do not copy profile bytes during staging: the timed, locked driver does the complete bounded assembly. Verify all hashes before exact dispatch_command in OFFLINE_BINDING.disabled.json. Do not use an optimized production interpreter or add arguments.

The disabled metadata is not a runtime release token; assignment and packet review are external mandatory gates. The exact host/root/hash/STOP/resource/idle checks are runtime fail-closed. The driver invokes no train.py, prepare.py, evaluator, runtime capture or network/provider API.

Retain input-identities, exact binary/version, argv/environment/resource limits, guard/lock receipts, raw bounded streams, exit/signal/CPU/RSS/scratch peaks, readonly postflight if complete and terminal. After driver exit, Operator must record the original lock is free and process identity/quiescence before any next work. No fallback decode, host queue alteration or automatic follow-on.

## Interpretation and local evidence

Exit0 is labeled PROCESS_EXIT_ZERO_COVERAGE_UNKNOWN, never successful all-model attribution. Supplying80 NEFFs does not prove all80 were used. Independent analysis must identify actual decoded models and unmatched/ambiguous mappings; if summary output lacks coverage fields, retain unknown rather than expand output automatically. Only emitted decoded device measurements support engine/kernel conclusions. Rank0 capture skew remains; this is not all-rank/whole-step timing, a K1/K2 speedup, internal matmul/SDPA dtype proof or MFU. Buffer fullness and instruction hierarchy are intentionally absent.

Sixteen own tiny stdlib tests pass normal/-O: exact archive/argv, layout, copy hashes/modes, missing/extra/symlink/deadline negatives, actual bounded stream/wall/CPU/file-size/scratch stops, detached-child and fast-double-fork termination, PID-reuse refusal, and pins. No tensor slot or actual profile bytes are used. Initial 13-test runs passed but exposed unclosed pipe ResourceWarnings; those raw outputs are preserved and readers were explicitly closed before the final 16-test runs. These are Implementer tests, not independent review or installed-runtime success.

Reversible preparation decision at 2026-09-25T23:18:54Z: prepare exactly one bounded CPU-only decode for future idle use. Undo: leave this packet unassigned/unstaged; originals remain untouched.
