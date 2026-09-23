# Source/direct Neuron discriminator (pending independent authorization)

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

The checked-in request is deliberately `pending_external_verifier`; full mode rejects it. The
sentinels below work only with `--static-only` and cannot reach a worker or Neuron import.

```bash
AUTHORIZED_HEAD='PENDING_EXTERNAL_VERIFIER_HEAD'
AUTHORIZATION_SHA256='PENDING_EXTERNAL_VERIFIER_REQUEST_SHA256'
echo 'No hardware authority. Obtain the exact request and command from the independent Verifier.' >&2
exit 64
```

Neither this runbook nor the harness computes or prints an authorized HEAD/request SHA. The final
Verifier is the sole source of those two literals, the unique run label, and the exact manifested
fresh-start request. An Operator must copy that report verbatim; no runtime substitution is valid.

## Decisive workload

The top-level process sequentially creates five fresh, non-existing compile-cache directories and
runs five isolated subprocesses: CE source, CE direct, optimizer raw, optimizer source, optimizer
direct. A child cannot run without the top-level nonce. Any nonzero exit or output line indicating
a recompile-limit warning invalidates the gate.

- CE runs 128, 512, and launch-relevant 32768 rows, twice each. Source and direct loss and logit
  gradient must pass actual `torch.equal` checks with max-absolute difference exactly zero. Forward
  and backward lane/call traces must be identical (one lane at 128; two at 512/32768). An independent
  PyTorch softcap-15 CE comparison is recorded only as a diagnostic and cannot veto or rescue the
  source/direct comparison.
- Optimizer variants run under isolated four-rank LNC2 `torchrun` processes. Every step snapshots
  all parameters, persistent momentum/second-moment/AdamW state, group scalars, ownership plan and
  cross-rank parameter digests. Source and direct snapshots must be exactly equal on every rank.
  Merged direct-owner state and every direct parameter/scalar snapshot must equal replicated raw.
- Diagnostics include a two-step tiny fixed schedule and the four-step changed schedule. The
  production gate uses the fs-w62 width-1024 inventory (117,453,580 parameters), with its None-grad,
  zero-grad, skipped-AdamW and changed-lr/momentum/betas pattern.

There are no numerical tolerances in the decisive source/direct or owner/raw gates: equality is
byte exact. The PyTorch CE values are clearly labeled non-gating.

## Independently generated request requirements

The Verifier must review and then finalize only a request whose `request_contract` equals the
hard-coded harness contract. It pins:

- host `trn22`, hostname `ip-172-31-32-253`, remote checkout
  `/home/ubuntu/trainium_freshstart_submission_validation_20260923_direct_discriminator`;
- seed 42, four host threads, four ranks and `NEURON_LOGICAL_NC_CONFIG=2`;
- `NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16` and the complete fixed
  PATH/NKI/thread/unbuffered/no-fallback environment;
- outer argv exactly `timeout --foreground --signal=TERM --kill-after=60s 5400s`;
- native Python `/home/ubuntu/workspace/native_venv/bin/python` and optimizer prefix exactly
  `/home/ubuntu/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=4`;
- the complete ordered protected-path hash map, including harness, test, this runbook, proposal,
  source/direct candidates, fresh-start tools, fs-w62 provenance, and the prior failed component
  result.

The manifested inner workload must be exactly:

```text
/home/ubuntu/workspace/native_venv/bin/python \
research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron_discriminator_direct.py \
--authorization-request research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron-discriminator-authorization-request.json \
--authorized-head <EXTERNAL_LITERAL> \
--authorization-sha256 <EXTERNAL_LITERAL> \
--expected-commit <SAME_EXTERNAL_HEAD_LITERAL> \
--run-label <VERIFIER_SUPPLIED_UNIQUE_LABEL> \
--report research/run-logs/<SAME_LABEL>-result.json \
--expected-hostname ip-172-31-32-253
```

The whole inner argv is prefixed by the fixed 5400-second timeout. The final Verifier must wrap it
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
   pre/postflight failure, recompile-limit warning, worker failure or any inequality is a failed gate.

Local implementation check (safe; no torch/Neuron import):

```bash
python -m unittest research/submission-validation/20260923-dense6x1024-direct-review-safe/test_neuron_discriminator_static.py
python -m py_compile research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron_discriminator_direct.py
```
