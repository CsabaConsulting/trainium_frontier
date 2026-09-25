# S-A against the Final Push L3 brief — independent HOLD

Reviewed 2026-09-25 by a Verifier, without changing model code, contacting a
host, launching training or evaluating a checkpoint.

**HOLD as the newly requested S-A / Recursive schedule implementation.**
Commit `61d05fd1dffe0f48fb8b4be025024be8f5fd8644` adds only four metadata
files under `research/v2/stacks/20260925-s-a-a2-l4/`. It makes no `train.py`
change. The candidate is still exact research-v2 SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
This is a scope mismatch, not a failed numerical experiment or a table verdict.

## What matches, and what is missing

The A2 component matches the declared layers `1,-1`, multiplier 64, dimension
128, bigram+trigram, BF16 RMSProp, flat table LR and separate table clipping.
The newer exact A2 CPU evidence is independently reviewed in the canonical
controller's `research/v2/verification/20260925-trn22-a2-exact-cpu-g2-r2-independent-NUMERIC-PASS.md`.

The old S-A command selects shared warmdown `0.9`, linear momentum cooldown
to `0.8`, embedding LR `0.45`, and the existing final LR floor `0.05`.
It does not provide `--recursive-schedule`, independent Muon `0.95` / Adam
`0.65` warmdowns, quadratic momentum cooldown to `0.79`, or the NorMuon
beta2 `0.95 -> 0.97` warmdown ramp. The code has only one `lrm` for all
non-flat groups at research-v2 `train.py:3980-3986`.

There is also an undeclared difference relative to newest L3: changing
`--embedding-lr` to `0.45` increases the flat n-gram table LR by 50% because
`NGRAM_VE_LR` defaults to zero and falls back to `EMBEDDING_LR` at
`train.py:2186-2188`. At width 1024 the table LR changes from approximately
`0.2598076211` to `0.3897114317`. Newest L3 does not request this change.

## Exact bounded implementation needed

1. Add one default-false `--recursive-schedule` switch, its module setting,
   parser assignment and saved toggle metadata. Keep the existing off path's
   arithmetic, optimizer groups and model construction untouched.
2. At the current schedule helpers (`train.py:3242`, `3301`, `3352`) add
   flag-on behavior using the existing elapsed/max-train-seconds convention:
   Muon warmdown 0.95, Adam warmdown 0.65, both with floor 0.05. Muon
   momentum keeps the 300-step 0.85-to-0.95 warmup, then follows the source's
   quadratic warmdown to 0.79. NorMuon beta2 stays 0.95 before Muon warmdown
   and ramps linearly to 0.97 during it.
3. At the optimizer-group assignment (`train.py:3980-3994`) choose the
   multiplier by `group['kind']`, preserving `no_lr_decay` precedence and
   `batch_lr_scale`. Existing flat RMSProp table groups retain their current
   schedule and LR. Preserve current WD behavior; do not add WD pulses,
   Adam beta1 decay, table beta2 ramps or other upstream features not requested.
4. Reuse the existing compiled Muon tensor-scalar path (`train.py:2599-2639`,
   `2914`). Enabling the ramp must select its dynamic `one_beta2` tensor;
   computing changing Python beta2 values while leaving
   `MUON_BETA2_FINAL == 0` would select the old constant path. A flag-on
   assignment of the ramp endpoints before optimizer construction is one
   straightforward way to preserve the existing dispatch.
5. New S-A argv should be exact A2 plus `--recursive-schedule`; remove the
   old extra schedule flags and `--embedding-lr 0.45`. If the Supervisor
   deliberately retains that LR arm, name and document it separately.
   Add an Apache-2.0 attribution comment for adapted schedule code.

Primary reference checked during this review:
[Recursive optimized_from_karpathy.py](https://github.com/recursive-org/first-steps-toward-automated-ai-research/blob/main/nanochat_autoresearch/solutions/optimized_from_karpathy.py).
Relevant one-based source lines: 834-837 (warmdowns/floor), 943-971
(momentum and beta2), 1049-1068 (time-based loop implementation).
These are source facts, not evidence the recipe improves our Trainium BPB.

## Proportional verification for the new patch

Review only that bounded diff; run flag-on CPU G2 with exact proposed A2
argv; compare the new file flag-off against exact c5c over three real CPU
training steps using deterministic G2 receipts and parameter digests.
Add focused schedule boundary checks at progress 0, 0.05, 0.35, 0.5 and 1,
including flat-table precedence, to cover the late schedule that the first
three training steps cannot exercise. No optimizer resume-format review is
needed. No CPU training was run for this metadata-only packet, and no
three-step identity or new G2 PASS is claimed. The clear missing-code finding
makes such a run unnecessary before the repair.

The old command remains a separately named, previously proposed flags-only
experiment if the Supervisor still wants it; this review does not block the
current A1/A2 host runs or require a new prelaunch review ceremony for them.
The corrected L3 patch can be prepared tonight; run-readiness requires its
actual diff and CPU evidence, not this plan alone.
