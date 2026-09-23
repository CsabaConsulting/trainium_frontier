# Source/direct Neuron discriminator authorization protocol

This is an implementation artifact for the already approved direct-review-safe proposal. It does
not authorize hardware, training, submission, or a scientific verdict. The candidate remains
`submissions/dense6x1024_direct_review/train.py` at SHA-256
`c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537`; the historical source is
`submissions/dense6x1024_lean_review/train.py` at SHA-256
`9173f62617789b058083c3a4722a906e2b165f4986716281f1915db3fb9fa344`.

## Trust root and current state

The checkout cannot authorize itself. Full mode requires literal `--authorized-head` and
`--authorization-sha256` values supplied in an independent Verifier's out-of-band final report.
The harness requires exact HEAD (not ancestry), exact authorization-request bytes, a clean tracked
checkout for every hard-coded protected path, and every protected file's independently reviewed
SHA-256. A descendant commit is rejected with the original literals. Supplying a descendant's own
newly computed values is not an authorized command and this runbook never emits one.

This corrected Implementer-authored request is `pending_external_verifier`; full mode rejects that
state. A later independent state-only finalization may make the request
`finalized_by_external_verifier`. There is no SHA sentinel: even static mode requires the literal
SHA-256 of the actual request bytes. This deliberately non-executable block confers no authority in
either state.

```bash
echo 'No hardware authority. Obtain the exact request and command from the independent Verifier.' >&2
exit 64
```

Neither this runbook nor the harness computes or prints an authorized HEAD/request SHA. The final
Verifier is the sole source of those two literals, the unique run label, and the exact manifested
fresh-start request. An Operator must copy that report verbatim; no runtime substitution is valid.
The authorization file intentionally does not embed its future commit SHA (which would be a Git
self-reference); exact HEAD authority exists only in the Verifier's external literal and the
matching CLI/environment/launch/job fields checked against `git rev-parse HEAD`.

## Controlled prepare import boundary

The r2 hardware attempt stopped before Neuron execution because an absolute dynamic import of the
frozen train module could not resolve root `prepare.py`. This correction does not change either
train module or any numerical gate. Before dynamically executing source or direct `train.py`, each
worker now requires the authenticated protected-map value for `prepare.py` to equal the hard-coded
SHA-256 `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` and the actual root file
bytes. Root and `prepare.py` must be a directory/file respectively, both non-symlinks.

The boundary rejects any already loaded `prepare` module, including one claiming the correct path,
and rejects any current module-search result whose origin is not the pinned root file. It removes
duplicate root entries, makes the repository root the first `sys.path` location, verifies canonical
resolution again, then reads and hashes the canonical source bytes once, compiles those exact bytes
with the canonical filename, and executes the resulting code in a fresh canonical module
`prepare`. It does not ask `SourceFileLoader` to execute the module, does not read or write a
`__pycache__`/`.pyc`, and sets module cache metadata to `None`. Immediately after execution it
re-reads the source, requires byte-for-byte equality and the authorized SHA-256, and rechecks module
name/origin/file/loader/cache metadata plus root-first precedence before returning to the frozen
train import. A timestamp-valid or hash-based malicious `.pyc` therefore cannot affect executed
definitions. A missing authorization value, shadow directory, wrong origin, symlink, non-regular
path, changed bytes, preload, or identity drift fails before train execution.

## Decisive workload

The top-level process sequentially creates five fresh, non-existing compile-cache directories and
runs five isolated subprocesses: CE source, CE direct, optimizer raw, optimizer source, optimizer
direct. A child cannot run without the top-level nonce. Any nonzero exit or output line indicating
a recompile-limit warning invalidates the gate.

The outer manifested environment remains exact and contains `WORLD_SIZE=LOCAL_WORLD_SIZE=4` for
the four-rank optimizer contract. Before either plain-Python CE child starts, the harness removes
`WORLD_SIZE`, `LOCAL_WORLD_SIZE`, all rank/group/role/node variables, `MASTER_ADDR`, `MASTER_PORT`,
and every `TORCHELASTIC_`, `TORCHRUN_`, `PET_`, `RDZV_`, `RENDEZVOUS_`, `PMI_`, `PMIX_`, and
`OMPI_COMM_WORLD_` variable. The CE worker independently rejects any such inherited variable before
importing Torch/Neuron and records logical singleton identity (world/local world 1, rank/local rank
0, no rendezvous). In contrast, each optimizer launch retains exactly the outer world/local-world
shape of four, removes stale rank/rendezvous values, and lets the pinned four-rank `torchrun` inject
the per-worker ranks and rendezvous settings.

Before the parent judges a launched child, it writes two label-derived, no-overwrite receipts under
`research/run-logs`: `LABEL.child-NN-NAME.log` is the complete combined stdout/stderr stream, and
`LABEL.child-NN-NAME.json` records the exact argv, sanitized environment identity, return code,
duration, timeout/error state, all recompile-limit matches, log byte count/SHA-256, expected worker
artifact observations, and compile-cache observation. Both use `O_EXCL`/no-follow semantics and are
fsynced; an existing path fails closed. The top result includes each record and its record SHA-256,
including on the first child error, timeout, warning, missing artifact, malformed worker report, or
later comparison failure. The Operator must retrieve these child logs/records with the other
receipts; a missing or overwritten child receipt invalidates the gate.

Every child is launched with `start_new_session=True`, making its PID the dedicated session and
process-group ID. The direct child is reaped. On timeout, wait error, or a direct-child exit that
leaves descendants behind, the harness sends `SIGTERM` to the entire group, polls for at most five
seconds, then sends `SIGKILL` to the entire group and polls for at most another five seconds. It
requires `killpg(PGID, 0)` to report no such group after the direct child is reaped. Only after this
quiescence proof does it fsync and hash the complete combined log, observe worker artifacts and the
compile cache, and write the immutable child record. Each child record includes whether TERM/KILL
was sent, group/session identity, direct-child reap state, poll count, and the final no-group proof.

Immediately before the top result is exclusively written, the harness reopens and revalidates every
accumulated child log byte count/SHA-256, child-record bytes/SHA-256, expected artifact snapshot,
compile-cache snapshot, and absence of the recorded process group. A late append or artifact/cache
change makes the top result fail closed with `ProcessEvidenceMutation`; it cannot retain a passing
gate. This is the final check before the result fsync/write.

- CE runs 128, 512, and launch-relevant 32768 rows, twice each. Source and direct loss and logit
  gradient must pass actual `torch.equal` checks with max-absolute difference exactly zero. Forward
  and backward lane/call traces must be identical (one lane at 128; two at 512/32768). An independent
  PyTorch softcap-15 CE comparison is recorded only as a diagnostic and cannot veto or rescue the
  source/direct numerical comparison. Its successful execution is operationally required: an
  exception, timeout, nonfinite record or recompile-limit warning still fails the worker/process.
- Optimizer variants run under isolated four-rank LNC2 `torchrun` processes. Every step snapshots
  all parameters, persistent momentum/second-moment/AdamW state, group scalars, ownership plan and
  cross-rank parameter digests. Source and direct snapshots must be exactly equal on every rank.
  Merged direct-owner state and every direct parameter/scalar snapshot must equal replicated raw.
- Diagnostics include a two-step tiny fixed schedule and the four-step changed schedule. The
  production gate uses the fs-w62 width-1024 inventory (62 parameter tensors and 117,453,580
  elements), with its None-grad, zero-grad, skipped-AdamW and changed-lr/momentum/betas pattern.

There are no numerical tolerances in the decisive source/direct or owner/raw gates: equality is
byte exact. The PyTorch CE values are clearly labeled non-gating.

## Independently generated request requirements

The Verifier must review and then finalize only a request whose `request_contract` equals the
hard-coded harness contract. It pins:

- host `trn22`, hostname `ip-172-31-32-253`, remote checkout
  `/home/ubuntu/trainium_freshstart_submission_validation_20260923_direct_discriminator`;
- seed 42, four host threads, `WORLD_SIZE=LOCAL_WORLD_SIZE=4`, four optimizer-worker ranks and
  `NEURON_LOGICAL_NC_CONFIG=2`;
- `NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16` and the complete fixed
  PATH/NKI/thread/unbuffered/no-fallback environment;
- outer argv exactly `timeout --foreground --signal=TERM --kill-after=60s 5400s`;
- native Python `/home/ubuntu/workspace/native_venv/bin/python` and optimizer prefix exactly
  `/home/ubuntu/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=4`;
- the complete ordered protected-path hash map, including harness, test, this runbook, proposal,
  source/direct candidates, launcher defaults (`train.py`, `prepare.py`,
  `research/fresh_start_backup.py`), fresh-start tools, fs-w62 provenance, the prior failed
  component result, the earlier discriminator verifier, and
  `independent-neuron-discriminator-fix-harness-verifier.json` at the FAIL commit that required this
  process-group correction, plus `independent-analyst-neuron-discriminator-r2.json` establishing the
  later import-boundary-only correction and
  `independent-neuron-discriminator-prepare-boundary-verifier.json` establishing the cached-bytecode
  bypass corrected here.

The manifested parsed command array is exactly:

```text
timeout --foreground --signal=TERM --kill-after=60s 5400s \
/home/ubuntu/workspace/native_venv/bin/python \
/home/ubuntu/trainium_freshstart_submission_validation_20260923_direct_discriminator/research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron_discriminator_direct.py \
--authorization-request /home/ubuntu/trainium_freshstart_submission_validation_20260923_direct_discriminator/research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron-discriminator-authorization-request.json \
--authorized-head <EXTERNAL_LITERAL> \
--authorization-sha256 <EXTERNAL_LITERAL> \
--expected-commit <SAME_EXTERNAL_HEAD_LITERAL> \
--run-label <VERIFIER_SUPPLIED_UNIQUE_LABEL> \
--report /home/ubuntu/trainium_freshstart_submission_validation_20260923_direct_discriminator/research/run-logs/<SAME_LABEL>-result.json \
--expected-hostname ip-172-31-32-253
```

The final Verifier must replace only the three explicitly marked external literal/label fields in
the parsed array and must wrap it
in the repository's existing `research/fresh_start_launch.py` request schema with: exact commit and
dirty-tree rejection; `trn22` host lock; `fresh_start_guard.py` preflight; unique label/cache/report
paths; and postflight/receipt retrieval through `fresh_start_sync.py`. The final report must publish
the complete request bytes/SHA and exact `python research/fresh_start_launch.py <request>` command.
There are intentionally no executable placeholders or request-generation commands here.
At process entry the harness reconstructs that complete request as parsed JSON and requires exact
equality with both launch and job receipts, including ordered command argv and all environment
key/value pairs. It also requires the matching passing preflight receipt and guard run identity.

## Verification sequence

1. Run local static tests only (command below). Review source and every protected hash.
2. Independently finalize the authorization request, commit that exact authorization-only change,
   and record final HEAD and request SHA out of band. Do not add a descendant commit.
3. Generate and review the exact fresh-start request using the literals, command, environment,
   host lock, preflight, timeout and postflight above. Publish it in the final Verifier report.
4. Operator checks out exact HEAD and runs only the report's literal launch command. No training.
5. Retrieve request/launch/job/process/log/result/postflight receipts. Missing receipts, a dirty path,
   missing child log/record, pre/postflight failure, recompile-limit warning, worker failure or any
   inequality is a failed gate.

Local implementation check (safe; no torch/Neuron import):

```bash
python -m unittest research/submission-validation/20260923-dense6x1024-direct-review-safe/test_neuron_discriminator_static.py
python -m py_compile research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron_discriminator_direct.py
```
