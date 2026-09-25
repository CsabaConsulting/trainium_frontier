# Independent trn21 matrix-LR0.0175 result audit

**PASS for this archived research-result audit; unchanged selector verdict:
DISCARD.** Candidate public-20M BPB `0.992122757617654` versus same-host c5c
LR0.015 control `0.9914739939934344`; delta `+0.0006487636242196171` (worse).
This is not a6ef7/Recursive result, nomination, adoption, or official timing/cold
certification. No ledger row, old classification, queue, candidate or host was
modified by this reviewer.

## Evidence and comparison

- Candidate: `fs-20260925-fp-c5c-matrixlr0175-trn21-r1`, arm `FP-c5c-F1`,
  `flag_arm`. Training archive `da4b94b7`; final eval archive `14f8cd6f`, under
  `research/v2/operator/20260925-trn21-scalar-matrixlr0175-r1/`.
- Exact unchanged source SHA256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  unchanged prepare SHA256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  These match local committed science bytes, preflight, attempt and terminal.
  This does not attest whole-host repository cleanliness.
- Exact proposal/launch argv agree. Sole science change from dense old-schedule
  control is `--matrix-lr 0.015` to `0.0175`. Four ranks, LNC2, seed42,
  width1024/context1024/pack32, compiler policy and no-table/no-qk-shift profile
  match. Explicit1800-second budget, high step cap and isolated output are
  operational restatements. No Recursive, FP8, architecture or batch change.
- Comparison is only to `fs-20260925-v2-c5c-a0-control-trn21-r1`, the valid
  same-host/same-base control in the current ledger. It is not compared against
  adopted6ef7 ON or the other host. Source/control association stays historical.
- All13 files in the two retrieval manifests match their recorded sizes and
  SHA256 values. Raw training:3669steps,480.9M rounded tokens,1784.8 source
  training seconds,1785.2 source charged estimate, `over_budget: False`.
  Logged sampled losses are finite; no traceback. Training and eval exited0.
  Control3675steps gives ratio0.9984; step count is diagnostic, not a quality gate.
- Eval is a separate process after training exit, exact unchanged
  `prepare.py eval-public`, checkpoint-bound, batch1/context1024,
  20,971,520tokens,1200-second limit. Raw result:57,395,868nats /83,462,186bytes;
  `57,395,868 / (83,462,186 * ln(2))` independently reproduces the BPB above.
  Eval696.1463568019681s, `eval_over_timeout: false`; terminal timestamp
  `2026-09-25T18:25:17.252702+00:00`.
- Checkpoint SHA256
  `f722109658e944dc6fcb27fc0e838759f07733e4b116d1bf98e8ce1e61870187`, retained
  at terminal `checkpoint_uri`. The runner hashes before/after evaluation;
  successful fresh evaluation demonstrates loadability. No checkpoint was
  downloaded or deleted here. Binary byte count is not supplied in this receipt.
- Control raw eval SHA256 independently rechecked; its57,358,336nats and same
  byte count reproduce `0.9914739939934344`. Its existing result review remains
  unchanged. No checkpoint-specific causality test is present in these scalar
  logs; unchanged eval-public does not itself run one. Keep causality null,
  consistent with the unchanged selector, rather than inventing a PASS.

## Timing limitations retained

Training launch UTC17:30:02.143813 to exit18:13:28.188309 spans
**2606.044496seconds** on the driver's wall clock, including process return.
The source prints startup allowance estimate813.5 and charged estimate1785.2.
There is no independent fifth-boundary observer in the clean launch environment.
Therefore actual startup and external charged-through-exit remain **unknown**;
do not subtract the source-native813.5 from an external-origin duration and call
the result measured, nor assume a900-second allowance merely because it is the
cap. This differs from architecture93ae, whose first-five lower bound already
proves saturation at900. No actual overrun is established by these scalar logs.

The result is accepted within the existing warm research/selector evidence
scope, not proof of official process-clock compliance. Both candidate and
same-host manual control have source-timer evidence and lack an independently
observed exact allowance. The reused cache paths are not a cold-start test.
These limitations cannot turn the measured quality regression into a win.

## Unchanged selector query

Query uses all10 current historical ledger rows plus one independently reviewed
candidate constructed **in memory only**. No row is edited/relabelled; no new
ledger file is installed. The complete input candidate and output are in
`AUDIT_RESULT.json`, with null external charged/startup fields retained.

- Ledger snapshot: canonical `d95ac9036aca7e705e4f67e227c211a41dd92513`,
  SHA256 `a6ea564fd159ab165673286ba6806ef224a9f8373d18e60189981cbecf3069fa`;
  canonical hash was rechecked unchanged after the query.
- Unchanged `workflow/tools/decide.py` SHA256
  `fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac`, rules
  `v2-2026-09-25-general-fixed-budget`.
- `invalid_runs: {}`; one c5c/trn21 control; rounded delta `+0.0006488`;
  **DISCARD — single-host delta >= +0.0005**.
- Diagnostics preserve `measured_startup_missing` and
  `cold_startup_unproven_submission_gate`. These do not alter the BPB verdict.

`audit.py` passed51 stdlib checks in normal and `python3 -O` modes; no tensors,
model imports or host work. One initial reviewer-only parser attempt rejected
the old control log's trailing footer as extra JSON; using the JSON decoder's
single-object parser fixed that local audit helper. It was not a run failure.
Audit script SHA256
`6579fc723475ec74c66b6e5f649da0dcebe087d0fece3552dfd63cfd891515e5`;
result SHA256
`bebea136e225390d64ef9b6b299c9cb6f3f5db658ff2d3536330e7236ab7a973`.

Disposition: retain the negative old-base result and its limitations. Do not
adopt LR0.0175, relabel it as a Recursive-base experiment, or infer replication
or launch authority from this receipt.
