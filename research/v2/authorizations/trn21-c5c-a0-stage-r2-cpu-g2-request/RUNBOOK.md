# trn21 exact-source four-rank CPU G2 request — HOLD

This Implementer package binds a proposed **synthetic CPU-only** G2 attempt to the
independently accepted trn21 stage-r2 checkout. It is not a dispatch artifact.
`CPU_G2_REQUEST.HOLD.json` and `INVOCATION_BINDING.HOLD.json` have all run
authority set to `false`; direct execution of `build_cpu_g2_request.py` exits
`DISPATCH HOLD`. No host contact, CPU G2, native training, evaluation, or
submission occurred in preparing this package.

## Frozen lineage

- Accepted checkout result: `research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-result/INDEPENDENT_CHECKOUT_RESULT_REVIEW.PASS.json`, SHA-256
  `b3c82bd594ae312142925cc3888f1de532473854f246a17b4ac4251baf37b89c`.
  That PASS accepts the **checkout only**, not CPU G2.
- Checkout Operator terminal receipt: SHA-256
  `1e11e59f1fe4ceecd393c43552d0f4105961e3d5fcc5e03709f07abc46d3dc3d`.
  The raw checkout-stage and unconditional-postflight receipts are separately
  hash-bound in the request and rechecked by the offline builder.
- Reviewed hardened, still-disabled G2 package:
  `research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2-final-pending/`.
  Its independent static review has SHA-256
  `ad19dc6a648977fbad4ce0b8d57de05256a68da0723160114fa5ac23ea60d455`.
- Host/source: trn21 `ip-172-31-34-85`, checkout
  `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`,
  one-commit clean shallow commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c`.
  `train.py` SHA-256 is
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  `prepare.py` SHA-256 is
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- The proposed G2 label is
  `fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2` and the **new** private
  scratch root is `/home/ubuntu/trainium-v2-g2-20260925-trn21-stage-r2`.
  One attempt is allowed. The frozen command arrays are review data, not
  permission to execute them.

## Required sequence; do not collapse roles

1. A separate Verifier reviews the exact request, invocation binding, builder,
   tests, and pinned source/result/static-review bytes. Archive an independent
   **static** verdict with full file hashes. A static PASS is not run authority.
2. An Implementer may then prepare a **new**, separately named enabled final
   request/binding frozen to that verdict. Do not mutate these HOLD files in
   place. An independent Verifier must review those exact final bytes. The
   Supervisor alone may issue a narrow, one-attempt CPU-only assignment after
   reviewing the final package. None of those downstream objects exists here.
3. Before any assigned attempt, an Operator must perform the full fresh live
   preflight in the request: exact host, source Git and file identity, no-follow
   CPU-tool SHA/inode/UID/mode/link checks, both protected checkpoints, all six
   warm caches, boot/mount/ext4 identity, STOP/breakers, root free space,
   processes/host lock/Neuron holder, science output paths, and exact new G2
   root/label/step/log/process absence. The old checkout postflight checked an
   **older** G2 root, not this new `...trn21-stage-r2` root. Its PASS does not
   prove that the proposed scratch root is absent now.
4. Only after all separate gates pass may the assigned Operator make the one
   synthetic four-rank CPU attempt, with the exact frozen command arrays and
   `--steps 3 --nproc 4 --timeout 900 --keep`. The Operator must retrieve raw
   evidence, prove quiescence, and perform a separate, unconditional full
   read-only postflight even on failure. Do not clean or retry the same label.
5. A different Verifier must compare the CPU numeric result against the pinned
   c5c anchor. A CPU PASS would not itself authorize native training or eval.

For local inspection only, run the offline tests with
`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-request/test_cpu_g2_request.py -v`.
The tests mutate in-memory copies to prove that forged review, checkout,
tool, root, command, or authority fields fail closed. They never dispatch.
