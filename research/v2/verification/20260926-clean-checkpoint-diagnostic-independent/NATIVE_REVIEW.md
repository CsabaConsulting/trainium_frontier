# Separate native diagnostic mechanics disposition

**Conditional PASS for disabled native diagnostic binding mechanics only.** Reviewed unchanged `d524de5cb974aa77883037cdf858fcfddba3113d` / canonical135b19e8. The real-checkpoint CPU outcome is still pending independent acceptance. Native remains unassigned/unstaged/launch-disabled until that evidence is accepted and the sole Operator receives a separate exact release plus fresh preflight. No CPU result, native execution, source-gate reversal, full-training eligibility or upload permission is supplied here.

`native_mechanics.py` ran **42 independent stdlib checks**, normal and `-O`, without Torch imports, tensors or host actions. It authenticates the native driver and compact-JSON argv hashes, exact source/checkpoint/support identity, actual AST identity of all CPU/native driver functions, native constants, four-rank scalar validation, mocked runtime initialization and exact finally cleanup. The prior CPU receipt's151 normal/optimized tests remain applicable to the identical helper/control functions; this new result is42 additional checks, not42 native executions.

## Exact bounded route

- Native binding SHA `72223bf612b17756843106c637cd3ec2f07380eb8cf0d86a2a6000f5f5d60da6`; driver `46a80ff186497f877f7268f864a47f8a4b1af7b9ca848878540b2079ce529f7e`.
- Same probe `d21d35c68440e6a2b62abf5338603f0c8a4891a454bb61d3df0fdc420d7682a1`, validator `9565ad654d75a897bc2d8c3f39872c4af54763996af827beca87b0b8e34a5c19`, candidate3f5035, research3e79 and retained checkpoint068749c2559f... as the CPU review.
- Exactly one `timeout --foreground --signal=TERM --kill-after=15s 900s` invocation of pinned native-venv `torchrun --standalone --nproc_per_node=4 probe.py --backend neuron`, with LNC2 and one-thread CPU helpers. No retry, training, public evaluation or later stage is launched.
- Probe requires WORLD_SIZE4/LNC2, initializes `dist.init_process_group('neuron')`, uses `torch.device('neuron')` and the unchanged actual loader. This matches unchanged prepare.py's native evaluation backend/mapping pattern; no adapter or local-rank-conditioned forward is introduced. Native synchronization is required before copied output observation. Missing synchronization API refuses as UNRESOLVED; this review does not claim that the installed native API or these model paths were executed locally.
- Four workers receive deterministic identical synthetic tokens. Each runs reference then candidate with the fixed nine-call matrix and25 cases. Cross-rank equality is checked from all four retained CPU scalar/hash receipts outside forward, not by a collective inside model forward.

## Fail-closed validation and cleanup

The independent four-rank fixture passes only with ranks0–3/world4 and every case present. Missing rank, wrong rank/world, a missing or duplicate case, an explicit failed/nonfinite case, divergent rank output hash, divergent rank input bytes, cleanup-unresolved status or attempted source-refusal erasure all reject. Byte parity/finiteness is produced by the pinned probe's comparison code, tested independently in the CPU review; the scalar validator does not recompute logits from hashes. Actual retrieved error fields/checks must still be audited, and no toy receipt is native correctness evidence.

The exact finally block destroys an initialized process group after success or a caught error. A cleanup exception changes status to UNRESOLVED. An uninitialized group is not destroyed. Source/call exceptions leave partial case evidence and cannot pass the required exact status/all25 checks. Forced termination can preclude finally/receipt writing; missing receipts and nonzero child exit remain failure, not permission to continue.

The native driver uses the unchanged common `/home/ubuntu/.cache/trainium_freshstart/device0.lock`, exact authenticated outer parent/FD, fresh ps AND neuron-ls before spawn and inherited FD in the immediate timeout child. Normal synchronous driver lifetime remains locked. The same STOP, host, source/support/checkpoint pins,64GiB available RAM and6.5GiB free disk floors are enforced. Six job-private absent/empty caches and matching compiler URL/DIR do not prove runtime coldness/performance; no existing cache is reset.

The900s child watchdog covers imports, distributed init, compilation, loading, matrix calls and orderly cleanup. Driver preflight/hash/cache-creation time is outside that child watchdog and separately included in driver elapsed reporting. The15s kill grace is not extra accepted diagnostic work. Per-rank receipt cap16MiB and global emitted artifact cap256MiB remain strict acceptance conditions; the driver polls every250ms, sends TERM to the owned process group then KILL after15s if needed, and final validation rechecks size. Polling may overshoot. SDK caches are explicitly excluded from emitted artifact size, so their physical size is constrained by host disk/preflight and later observation rather than a256MiB hard quota.

As already documented, timeout or parent/group termination is **not proof that every torchrun/compiler descendant is gone** or that lock FD inheritance survives every descendant's FD handling.900+15 is the configured watchdog, not an experimentally proved upper bound on arbitrary orphan activity. No automatic follow-on exists; the sole Operator must retain fresh final ps/neuron-ls quiescence and any failed timeout/cleanup receipts before the host can be reused. This receipt does not waive that existing requirement or claim an untested process-tree guarantee.

## Gate scope

Mechanical acceptance is conditional only; CPU accepted evidence is an external persisted workflow gate, not an executable approval document checked by this driver. This disabled packet must not be staged merely because its code can run. Successful future native receipts would establish only tested syntheticT1024/B1/B2 behaviour across ranks0–3/world4, not all contexts/world sizes, a public BPB score, unchanged-class equivalence, cold/full training qualification or submission acceptance. The fresh structural `EVAL_BATCH_WRAPPER_OMITTED` **REFUSE is preserved verbatim and remains true**.

Reproduce locally, stdlib only:

```text
python3 research/v2/verification/20260926-clean-checkpoint-diagnostic-independent/native_mechanics.py
python3 -O research/v2/verification/20260926-clean-checkpoint-diagnostic-independent/native_mechanics.py
```
