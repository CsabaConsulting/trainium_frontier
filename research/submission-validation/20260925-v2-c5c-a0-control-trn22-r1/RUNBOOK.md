# Warm c5c A0 control — pending runbook

This package prepares one warm trn22 A0 control. It does not authorize dispatch. The request is
bound to the immutable science-core commit and its exact fingerprint map, but its pending state and
`dispatch_authorized=false` are intentional fail-closed policy sentinels. Independent verification
and a separate Operator authorization decision are required before use.

## Immutable science contract

- Host: `trn22` / `ip-172-31-32-253`; seed 42, 1800 seconds, world 4, LNC2 are exact source/default
  semantics. Do not add `--seed`, `--max-train-seconds`, or `--num-steps` to the G2-tested argv.
- Root `train.py` SHA256: `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
- `prepare.py` SHA256: `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Compiler flags: `--optlevel=1 --auto-cast matmult --auto-cast-type bf16`.
- Warm caches only: reuse the six existing `caches/train/{neff,local,hlo,inductor,compiler,debug}`
  directories under `/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/`, mapped respectively to
  `TORCH_NEURONX_NEFF_CACHE_DIR`, `TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR`,
  `TORCH_NEURONX_HLO_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `NEURON_COMPILE_CACHE_DIR`, and
  `TORCH_NEURONX_DEBUG_DIR`. These are the exact train-cache paths from the same-host 35c9
  LR0.015 control. The six directories must already exist. Bounded misses may add entries within
  those paths; do not clear, create, copy, or redirect a cache. `/tmp/neuron_cache`,
  `/tmp/neff_cache`, and `/var/neuron-cache/research` are out of scope for this launch.
- Exact workload argv is the suffix after `train.py` in the request template.

## Required order

1. Obtain durable PASS receipts for retained-checkpoint G1 calibration and the exact four c5c G2
   probes/comparisons. A missing or non-PASS receipt stops this package.
2. An independent Verifier checks the request's core commit and every fingerprint. Any later
   authorization record must bind this request's exact SHA256 and must not alter command,
   environment, label, paths, floors, or policies.
3. Stage the reviewed core into the unique absent checkout
   `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r1`. Do not edit or reuse any
   retained checkout. Verify clean HEAD, origin, all hashes, hostname, unique path/label absence,
   idle device, quiescence, and the canonical device lock.
4. Snapshot all six retained warm caches by anchored path/device/inventory without modifying them. Require at
   least `7,623,566,951` free bytes before dispatch: the legacy guard's exact 6.5-GiB floor plus a
   conservative 0.6-GiB dense-output allowance. Refuse if the projection is unsafe.
5. Dispatch the finalized request once through the unchanged generic `fresh_start_launch.py`. Its
   child command is bounded at 3600 seconds; return code 124 is failure. No cleanup or deletion is
   authorized.
6. After quiescence, retrieve and hash the launch/job/process/log/postflight files, the step-boundary
   file, and `out/.../final.pt`. The boundary file is **not JSONL** and contains no loss or timestamp:
   `prepare.record_step_boundary` appends one fsynced decimal integer line after each completed
   optimizer step. Require a nonempty consecutive monotonic sequence beginning at 0, no duplicate
   lines, and at least five boundaries. Use the training stdout log—not the boundary file—to compare
   step-0/1 losses exactly with `9.0112` and `8.7822`. Later warm/cold timing-dependent losses are not
   an equality gate. Validate the printed training budget summary together with the external
   runtime observer's boundary-arrival timestamps; neither evidence source is sufficient alone.
7. Seal a parsed checkpoint receipt (weights-only load, finite tensors, tensor count and exact
   SHA/bytes), then run causality and a fresh public20M evaluation in separate later-authorized,
   bounded guarded jobs. The historical 499f753c control has no authenticated public20M value; its
   only quality metric is a 2M screen. Compare the new public20M result with the separately declared
   global numerical-equivalence band `[0.99095, 0.99295]`. This first c5c baseline cannot be called a
   quality win. A value outside the band opens a breaker and requires a human alert.
8. Archive all evidence before any verdict. Preserve the checkpoint, caches, checkout, and logs.
   There is no prune, cache cleanup, cold run, deletion, submission, or leaderboard authority here.

## Known boundary

The generic launcher records runtime source hashes but does not compare `expected_sha256` itself.
Therefore staging verification and post-retrieval manifest comparison are mandatory independent
gates. This package deliberately does not add a new driver or modify the known working launcher.
