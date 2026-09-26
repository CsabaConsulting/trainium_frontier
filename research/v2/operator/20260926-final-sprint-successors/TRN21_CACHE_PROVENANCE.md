# trn21 discarded-run cache check — local provenance only

26 September 2026, approximately08:27 UTC. No host call, scan, staging, hash of
remote data, deletion or signal. Checked archived bindings only. No eligible
positive run-private cache candidate was established among the runs below;
reclaimable bytes are therefore **unproved**, not a claimed zero-size census.

## clock36 LR.012: not private, exclude

Its exact six training cache directories are:

```text
/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/compiler
/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/neff
/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/local
/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/hlo
/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/inductor
/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train/debug
```

These are the shared Sep24 research namespace, not children owned solely by
the discarded LR run. Archived actual A1 and matrixLR.0175 training launches,
plus 3e79/75b9/bcbd full bindings and ledger environments, reference this same
set. The prepared genuine e1e2 trn21 control also still references it. Thus
there is positive evidence of sharing, not merely absence of private proof.

Its eval caches are the six compiler/neff/local/hlo/inductor/debug children of
`/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/eval/caches`.
The **currently running batch.021's attached20M evaluation uses that exact
namespace**, as does the prepared e1e2 control. Preserve it intact.

Reference bindings and SHA256:

- `research/v2/proposals/20260926-clock36-lr012-trn21/NATIVE_BINDING.disabled.json`:
  `11e4470243ae8d4b0364d1b6f7b58231052c02f1e25733ab9ae16570cde6082a`.
- `research/v2/operator/20260925-trn21-a1-full-final/training-launch.json`:
  `c0455ab1cc1611b2b648ce28dd435a07bd03603d0fa1e3bf52529a7c7e2a5866`.
- `research/v2/operator/20260925-trn21-scalar-matrixlr0175-r1/completed-training/training-launch.json`:
  `36dbf21584baf168d47bcc52490f2406e5c7380e4a71f723232fd841062c81e3`.
- `research/v2/proposals/20260926-base-cache-full-trn21/NATIVE_BINDING.disabled.json`:
  `bc492cf2fcf863a615835008dd7598cf03b58fa2f5c157661a2e5022434d6909`.
- `research/v2/proposals/20260926-architecture-recursive-full-trn21/NATIVE_BINDING.disabled.json`:
  `c8503bf7aeb06cc4f59d289eba73def9ebd4a59e46dd804a8a16b74ebc425035`.
- Current binding: `research/v2/operator/20260926-final-sprint-successors/trn21-batch-lr14-bound-packet.json`.
- Prepared fallback: `research/v2/operator/20260926-e1e2-prospective-controls/trn21-prepared.json`.

## Other bounded exclusions

- The current P0 schedule's six private cold-cache children are actively reused
  by batch.021 training, and P0 remains INSUFFICIENT, not discarded. Exclude.
- The 75b9 cold-five namespace is a distinct unscored diagnostic; its scored
  full run used the shared Sep24 namespace, not those cold-five caches. It
  cannot acquire scored-owner eligibility merely from source identity. Exclude.
- Archived failed K2-symmetric and ordinary-subnormal private caches have
  allocated-size receipts of692,224B and380,928B respectively, but are unscored
  diagnostics. Exclude under the current already-scored-only cleanup rule.
- Previously authorized discarded checkpoints (including LR.012) were already
  removed; do not count them again. Checkpoints/logs/results/current/shared
  caches stay protected under this local-only task.

At the prior08:20 snapshot, trn21 free13,633,781,760B was861,732,864B below
the next13.5-GiB initial floor, before its pending checkpoint. This note does
not authorize reducing that floor, altering P2's larger reserve, deleting
shared caches, or launching an under-resourced job. Actual post-eval free
space remains the eventual gate. trn22 K2-ON's separately assigned live-parent
comparison and13.5-GiB floor are unchanged.
