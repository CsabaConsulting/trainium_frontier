# Independent Verifier: c5c A2 native-30 smoke package

Reviewed Implementer commit `436d2e1ef331e5e627a8c604689336d193da2d4f`
on 2026-09-25 IST. This review is independent of the package author and did
not access a host, stage a checkout, execute G2, launch Neuron training, or
evaluate a checkpoint.

**Verdict: STATIC FAIL; staging and dispatch remain HOLD.** The proposed
source and ordered argv are correct, but the package omits a positive
STEP_LOG-parent preflight. That omission matches a newly observed zero-step
infrastructure failure on the same host. The Implementer must repair the
package and a different Verifier must review the repaired bytes before an
Operator may stage or launch this smoke. This receipt is not a correctness
breaker reset, one-shot authorization, A2 result, or scored-run approval.

## Checks that passed

- The three added files are only
  `NATIVE_SMOKE_GATE_CONTRACT.json`, `NATIVE_SMOKE_RUNBOOK.md`, and
  `NATIVE_SMOKE_IMPLEMENTATION.md` under the A2 package. No `train.py`,
  `prepare.py`, data, tokenizer, evaluator, launcher, queue, or decision-rule
  bytes changed. Both package JSON objects parse and `git diff --check`
  passed. The additive contract SHA-256 is
  `d91bb0c6dc117a15240e5c51ea7fe9638819f4d7f3e50d2cd08551c22ee64382`.
- Recomputed identities on the reviewed branch match the package:
  research-v2 `train.py` is
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
  unchanged `prepare.py` is
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
  `research/v2/queue.json` is
  `1e9dcc4fd5f38e62829dc19460d2337662184ecfbd4d21d8a20f15993be9dd00`,
  the original A2 contract is
  `a40723151099e0fec912d670b702ef9f2774fe14853025660a9ffc116d2fc364`,
  its independent static-r2 receipt is
  `c1f4665335c3af3201593ff3c1770b21c6430abf821be078422e890f4a16d85d`,
  and the read-only boundary observer is
  `a6f437fe8a9d0a14ec3025bc822f1afa12e3f355fca730e4ae0db7ae00263592`.
- Independent ordered-array comparison returned exact equality between the
  new 35-token smoke command and original A2
  `source_command_argv_prefix + a2_argv + argv_extension`; the original
  27-token A2 argv is also the queue's literal in-place replacement of
  `--no-ngram-ve`. It retains m64/d128 bigram+trigram tables on layers
  `1,-1`, RMSProp, BF16, clipping, matrix LR 0.015, LNC2/four ranks,
  seed 42, 1,800-second parser budget, `--num-steps 30`, and
  `--no-eval-public`. `train.py` saves `final.pt` after the step cap and
  calls unchanged `record_step_boundary(step)` once per completed step.
- The package is explicitly non-dispatchable: stage/guard/env/cache/STEP_LOG
  values, A2-specific disk/HBM projection, exact-size G2 receipt, native
  one-shot authorization, and Operator postflight are all unset. The proposed
  smoke label was found only in the pending A2 package, not in local ledgers
  or archived run logs. That does **not** prove the host label/output paths
  are unused; they must be checked live before any stage.

## Blocking defect: STEP_LOG parent is not authenticated

`prepare.py` lines 96-111 appends to `NEURON_COMPETITION_R1_STEP_LOG` with
`open(path, "a")`; it does not create the parent directory. The new contract
requires the *file* to be absent before launch, and the runbook repeats that
absence check, but neither requires the parent to exist or be a trusted,
writable directory outside the clean pinned source checkout. The Operator
could satisfy every listed absence check while pointing at a nonexistent
`research/run-logs` parent. This is not theoretical: the archived trn22 A0 r2
CPU G2 [Operator HOLD](../../v2/g2-probes/fs-20260925-v2-c5c-a0-control-trn22-r2/operator-stage-hold.json)
at 22:48:11 UTC reports `FAIL_INFRA_STEP_LOG_PARENT_ABSENT` from this same
`record_step_boundary` append, with zero reported boundaries. Its G2 label
was consumed and no native science ran. A native A2 smoke with the same
path mistake would waste a unique label and compile time.

Required repair, without changing `prepare.py` or any model bytes:

1. State that the STEP_LOG is in a unique, trusted absolute directory **outside
   the clean source checkout**. Before observer/guard launch, the Operator
   must verify that parent and all traversed components exist, are directories,
   are not symlinks, and are writable by the exact run UID; record resolved
   path plus parent device/inode/ownership/permissions. Create a dedicated
   label directory only under a reviewed scratch root if needed, never by
   dirtying the pinned checkout or deleting protected material.
2. Keep the STEP_LOG file itself absent before launch, prove the observer
   witnessed that absence, and bind the identical canonical file path in
   the clean `env -i` command, observer request, guard manifest, and immutable
   one-shot request. Do not satisfy writability by precreating the step file.
3. Fail the stage if any parent/UID/path binding is missing, if an output
   path is reused, or if the final request omits the fresh preflight evidence.

## Further HOLDs, not static passes

- The current A2 table G2 is still unrun at the exact m64/d128 size. Its
  four-rank, two-step RMS trace, all rank records, containment and
  quiescence must receive independent review before native smoke. The old
  m8/stub or different-source evidence cannot substitute.
- The seeded queue's IMPL-1 acceptance explicitly says **50-step** Neuron
  smoke with equal per-rank table hashes; the newer master runbook and this
  package say **30 steps**. The Implementer correctly disclosed that conflict.
  A 30-step result must not be relabeled as the 50-step acceptance. Before
  full A2 promotion, obtain a separate new-label 50-step proof or a durable
  human-approved supersession of that criterion.
- Exact c5c normal logs do not prove per-rank table/moment equality. The
  package correctly says the native gate remains HOLD if that evidence is
  missing; a separately reviewed, exact-source diagnostic would then be
  needed. A rank-0 checkpoint or finite rank-0 loss is insufficient.
- trn22 G1 calibration is not a same-host A0 control. The latest controller
  state still has `launch_authorized=false`, `hosts.trn22.launch_allowed=false`,
  no valid c5c A0 public20M control, and no A2-specific disk/HBM projection.
  The human-authorized disk-breaker re-arm alone does not authorize a run.
  A non-scored smoke may precede A0, but a full scored A2 cannot.

Only after the narrow package repair and independent re-review may the
Supervisor consider **Operator staging only**. Native dispatch needs its
own exact-size G2 review, fresh host/observer/disk/containment proof,
immutable request and one-attempt authorization. No host action occurred in
this review.
