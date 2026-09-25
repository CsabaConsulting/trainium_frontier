# Exact 93ae native architecture — live

Operator launched `fs-20260925-l2-unet8x1024-93ae-trn22-full-r1`
on trn22 at **2026-09-25 17:41:38 UTC / 23:11:38 IST** after independent
full-size CPU/dual-OFF acceptance `02ed1896` and fresh machine checks.

- Detached flock PID 3079487; driver 3079488; torchrun 3079490;
  four workers 3079505–3079508. Exact source `93ae1f92…`, driver `da58b422…`.
- Host root:
  `/home/ubuntu/trainium_l2_unet8x1024_93ae_20260925_trn22_r1/native/fs-20260925-l2-unet8x1024-93ae-trn22-full-r1`.
- Log verifies eight 1024-wide layers, five attention layers, four U-Net skips,
  value embedding on layer 7, no tables, 113,256,724 parameters, four ranks.
  Recursive schedule and FP8 are not enabled. Comparator stays c5c A0
  `fs-20260925-v2-c5c-a0-control-trn22-r10`, public BPB 0.991513958330512.
- Exact 1800-second budget. Final fresh public20M/context1024 runs automatically
  after training. Initial status is compilation; no quality or cold claim.
- Fresh preflight found 46,126,444,544 disk bytes available, no STOP/breaker,
  no prior workload/device holder, free common lock, and matching source,
  evaluator, driver and CPU receipt hashes. Completed proposal is in commit
  `2542a5a9`; raw launch/current workers are in `b2cd4670`.
- Existing warm train/eval caches are reused. Existing SDK graph_stats export
  is training-only. Exact fifth-step timestamp is unavailable from this proven
  driver; retain its source-native startup estimate and process wall separately.
  Genuine cold validation is mandatory before submission.
- The old serial queue is exited; owned PAUSE_NEXT and six unstarted fallbacks
  remain intact. After manual training AND eval fully end, inspect quiescence,
  then hand back serially to Supervisor's updated tail or the existing queue.
  Never run both launchers or interrupt this job to reorder the queue.

At 17:43:12 UTC, one already-created live compiler input was retrieved read-only
before SDK cleanup: 16,658-byte module.mlir, SHA256
`927f008f870ad145057d17cdb33f66dfeee965cd8392db47cf7063ff4b7d838d`.
`live-compiler-input/` includes the exact input, first 64 KiB of the existing
compiler log, timestamps and observed compiler commands. Graph role and cache
key remain unknown; this is not evidence of a miss or a matched-run comparison.
No flags, SDK, model or host cache content were changed for this observation.
