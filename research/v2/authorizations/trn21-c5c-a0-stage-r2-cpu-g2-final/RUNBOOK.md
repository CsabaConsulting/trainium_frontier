# trn21 stage-r2 c5c synthetic CPU G2 — final enablement candidate

**Current state: HOLD.** This Implementer package is a CPU-only *intent*, not
an independent final PASS or Supervisor assignment. Neither future authority
file exists at implementation time. Direct execution of the local wrapper
exits `DISPATCH HOLD`; no host connection, CPU run, native training, evaluation,
AWS action, cleanup, breaker reset, retry, or submission was performed while
preparing it. The reviewed disabled request remains unchanged.

## Immutable basis and permitted experiment

The actual accepted trn21 checkout-result review is
`research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-result/INDEPENDENT_CHECKOUT_RESULT_REVIEW.PASS.json`
(SHA-256 `b3c82bd594ae312142925cc3888f1de532473854f246a17b4ac4251baf37b89c`).
The independently reviewed disabled CPU request is
`research/v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-request/CPU_G2_REQUEST.HOLD.json`
(SHA-256 `20bb8c7349e7d98a2ff6dd05ff7570326c8e25da9b44871600ec321434bf1a78`),
with separate static-review SHA-256
`32c9a11c75515c22ba063ce2cd4ec364f120b8d4fae0c904701978781ac61aaf`.
The accepted source is exactly clean one-commit shallow `e65a3097…` on trn21,
with `train.py` SHA-256 `c5c410af…` and unchanged `prepare.py` SHA-256
`5807c46e…`. The frozen four-rank, three-step, synthetic CPU command remains
`--steps 3 --nproc 4 --timeout 900 --keep`, with `--no-eval-public` and without
`--real-data`. All four original compact-JSON argv hashes are pinned in the
request and re-derived from the reviewed offline builder. Nothing here enables
native Neuron training, public evaluation, or a science run.

The G2 label is
`fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2`, on a **new** private ext4
scratch root `/home/ubuntu/trainium-v2-g2-20260925-trn21-stage-r2`. The
checkout postflight inspected a different older root (`…trn21-r2`), not this
one. Therefore the exact new root, process directory, step log, receipt,
label and matching processes must all be proved absent in the fresh live
preflight. Existing source, two checkpoints and six warm caches are protected.

## Authority sequence

1. A separate Verifier statically reviews the exact five files in this
   directory: final request, binding, local wrapper, transported remote
   adapter, and tests (plus this runbook). Archive a structured exact-byte
   `INDEPENDENT_FINAL_CPU_G2_REVIEW.PASS.json` at the fixed path named in the
   request **only on actual PASS**. It must equal
   `cpu_g2_invocation_final.expected_review(build_plan())`; that object binds
   the exact request, binding, wrapper, remote code, source, command digests,
   fixed marker and CPU-only scope. The Verifier does not contact a host.
2. Only the Supervisor may then create the actual fixed-path
   `SUPERVISOR_CPU_ONLY_ASSIGNMENT.json`. It must equal
   `cpu_g2_invocation_final.expected_assignment(build_plan(),
   SHA256(actual_final_review_bytes))`. This is a one-way byte chain; merely
   writing a claimed review hash or passing a dict to the Operator is not
   authority. These files must be regular, owned locally, single-link,
   non-group/world-writable and read with `O_NOFOLLOW`.
3. The Operator must use `operator_once()` from the exact reviewed local
   wrapper with no alternate marker, label, root, operation or argv. Before
   any SSH, it checks local controller/worktree STOPs and both global/trn21
   breakers, then creates the fixed owner-private journal/one-attempt marker
   with `O_EXCL|O_NOFOLLOW`, fsyncs file and parent, and never deletes it.
   A failure after this point still consumes the only attempt.
4. The first strict-host-key SSH call is a full fresh **read-only** audit. It
   authenticates host boot/mount/namespace and 10-GiB free-space floor;
   source Git, train/prepare hash+inode; no-follow CPU helper hash+inode/mode;
   two protected checkpoint hashes and holders; six cache identities; host
   STOPs, labels, processes, lock and Neuron occupancy; and exact new G2
   scratch absence. The wrapper rechecks actual authority files and local
   STOP/breakers after that audit.
5. The second, single run SSH call independently repeats the full audit in
   its own process. It bridges boot/mount/source/tools/STOP to the frozen
   `contained.py launch` in the **same** SSH process, creates only the new
   private scratch root, and performs another bridge in the launch child's
   `preexec_fn` immediately before exec. A changed boot, mount or newly
   appearing STOP is HOLD, not a launch. It invokes only the frozen launch,
   wait, failure-stop and quiesce arrays. The CPU receipt must be a comparable
   three-step, four-rank synthetic PASS; no BPB conclusion is made here.
6. A **new, separate** SSH process performs the full read-only postflight
   unconditionally after an attempted preflight/run, including on transport,
   preflight, run or parsing failure. Raw SSH stdout/stderr and decisions are
   fsynced into the fixed journal. Preserve any partial CPU root and marker;
   no cleanup or same-label retry. A different Verifier must compare CPU
   numerics and quiescence before the Supervisor considers any later native
   experiment. A CPU PASS itself grants no native/eval authority.

The supplied offline suite uses synthetic snapshots and mocked transports;
it never SSHes. Run it with:

`python3 -B -m unittest research/v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-final/test_cpu_g2_final.py -v`
