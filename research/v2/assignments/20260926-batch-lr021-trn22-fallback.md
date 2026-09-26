# Selected next: existing batch/LR.021 cross-host check

2026-09-26 15:02 IST. Supervisor release under FINAL SPRINT.
Sole remote Operator: trn22_queue_operator. This selects an existing reviewed
flag-only arm; it does not change candidate bytes, gates or launch mechanics.

Launch fs-20260926-sprint-clean-batch-lr14-trn22-r1 on trn22 after fresh guards.
Completed proposal: research/v2/proposals/20260926-sprint-p0-clean-batch/EXPERIMENT_PROPOSAL.json
SHA256 785bfe4ea42bd8c82cacfeabf874de39bb7f253bc5bf1b23e2f7b112e6db8aef.
Prepared packet: research/v2/operator/20260926-final-sprint-p0/batch-lr14-trn22-successor-prepared.json
SHA256 bb5d0a7977783b04568cb9bd4b54971002cd1121487a925a7136298e10d8bfdc.
Source SHA256 8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee.

Exact command, seed42, full chip four ranks, fixed1800 official seconds:
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --total-batch-size 262144 --matrix-lr 0.021

One fresh independent P0 review c28e7724 already accepted this exact source
and variant. The same source/argv completed on trn21:1921steps, public20M
0.9869854734569898, charged1788.753648s; this is not yet an adopted win.
Compare the new trn22 result against its actual e1e2 same-host control
0.9879041000701606. The trn21 matched control is still running, protected.
This is the prescribed joint batch/LR arm, not an isolated LR attribution.
Use unchanged decide.py only once actual comparable rows exist.

K2's one diagnosed retry failed beforestep0 at09:25:28.443006UTC with
unbound variable 'tokens'. Complete53-identity/device/fuser/original832589
quiescence was observed09:26:13.421855; archive8b65da0d. K2 is parked;
no third K2 attempt or dependent table launch follows this release.

Prove label unused and hold ORIGINAL device66305/inode832589 lifetime lock.
Fresh in-lock STOP, ps, neuron-ls/fuser, exact source/argv and protected
evaluator/harness identities, disk>=14495514624B, runtime>=11274289152B,
RAM>=64GiB. Retain existing original guardian through training, attached
fresh-process public20971520/context1024 eval and safe descendant release.
Reuse only the prepared retained same-host research caches after their
owner is terminal; do not call reuse a cache hit or a cold start.

Operator may mechanically bind this release and the actual failed K2
predecessor to the proven stage_next/manual transport. No new source or
launcher semantics, extra review, user approval, deletion or AWS action.
Target dispatch before09:35:28UTC (ten-minute idle limit); if a fresh guard
fails, preserve the exact error and do not bypass it. Record actual launch,
all failures and complete raw results. No automatic adoption or upload.
