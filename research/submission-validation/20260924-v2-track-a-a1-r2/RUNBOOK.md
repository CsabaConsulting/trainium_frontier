# Track-A A1 r2 infrastructure retry

This package is preparation only. Its authorization remains pending independent verification and
must reject launch materialization. It does not authorize SSH, host cleanup, training or a push.

The Operator must use the unique label
`fs-20260924-v2-a1-lastlayer-ngram64d128-trn21-r2` and remote directory
`/home/ubuntu/trainium_freshstart_20260924_v2_a1_lastlayer_ngram64d128_trn21_r2`.
The r1 remote, logs, dispatch receipt and evidence must never be overwritten or deleted.

After cloning the final approved Git bundle, the checkout must be materialized as the exact
symbolic branch `experiment/20260924-v2-a1-lastlayer-ngram64d128-trn21-r2` at the exact
Verifier-approved finalization HEAD. A detached HEAD is not acceptable even when its commit bytes
match. The Operator must create/switch the branch without `-f`, reset, or history rewriting, then run:

```text
python research/submission-validation/20260924-v2-track-a-a1-r2/stage_contract.py \
  --repo /home/ubuntu/trainium_freshstart_20260924_v2_a1_lastlayer_ngram64d128_trn21_r2 \
  --expected-head EXACT_VERIFIER_FINALIZATION_HEAD \
  --request /home/ubuntu/trainium_freshstart_20260924_v2_a1_lastlayer_ngram64d128_trn21_r2/research/submission-validation/20260924-v2-track-a-a1-r2/a1-trn21-r2-request.json
```

Only a clean result binding the symbolic branch, exact HEAD, two-file finalization diff and every
protected working/tree hash may proceed to the standard launcher. The exact source, A1 flags,
seed42, full1800 budget, LNC2/world4, authenticated b0b23 trn21 control and fresh2M population are
unchanged from r1. The r1 failure-evidence identity remains pending until supplied and independently
verified; no pending template is executable.
