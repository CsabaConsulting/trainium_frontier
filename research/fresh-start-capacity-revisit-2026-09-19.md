# Capacity checks after the kernel gains

This is a source-audit follow-up, not a new BPB result. The frozen standalone
submission remains source129, reproducing source98 public BPB0.9928513116311253.
The cold training and evaluation rehearsal passed. The caller-profile diagnosis
remains active; capacity preparation proceeds independently of the frozen package.

Current2026-09-19 09:51UTC: the width1152 phase is COMPLETE/clean with all four
model postchecks. Median587.591986ms versus nearby same-source width1024
487.434961ms, a20.55% step-time increase for21.54% more parameters. It passes
the predeclared700ms threshold below. The single seed42 quality run
`fs-w146-dense1152-context1024-pack32-owned-ce-1800-r1` started09:45:14.368755
with clean preflight; finite losses and about0.59s/step are observed after
compilation. Training, checkpoint preservation and matched public evaluation
are still pending. The diagnostic phase's final loss is not a quality forecast.

The research installer now scopes the resident-SBUF assertion to the NKI forward
backend. Native installation can reach its actual compiler at width1152/1280;
all module/parameter/shape/cache assertions are retained. The existing full
Linear integration gate accepts `--real-channels` (default1024), so the same
cached/uncached, accumulated-gradient and evaluation/checkpoint checks can be
applied at a new width.

Source143 (`2f826dc2e8ae244f67e91e128e6692b52b99cddc`) passed the actual
pack32/T1024 gate for both1024 and1152 channels. Both runs assert compiled
execution and cover cached/uncached accumulated FP32 master gradients, two
updates, exact native evaluation and strict checkpoint restoration. They retain
the existing numerical bounds; no tolerance was changed. The combined job
`fs-w143-native-mlp-widths1024-1152-integration-r1` completed command0 with clean
postflight at08:35:45.433370UTC. This establishes supported MLP behavior, not
whole-model speed or BPB. Actual optimizer-group measurements are still required
before assigning owners for a wider model.

`scratch/optimizer_group_cost_probe.py` now derives its complete parameter-group
inventory from an exact recorded `PROFILE_CONFIG`, changing only the requested
width. A device invocation measures one actual group or an explicit prefix,
including native stacking/copies/update and optional singleton padding. The
synthetic fixed-gradient measurements do not change the ownership planner or
replace its exact numerical/state/collective gates. Hardware cost measurements
completed on trn21 source144, job
`fs-w144-width1152-optimizer-group-costs-r1`, clean postflight09:02:50.813903.
Inventory finds142,752,108 parameters and13 measurement cases across nine
actual groups. Its input is the hash-verified source144
mirror of the trn22 native capture log, not an assumed cross-host file path.
The prepared planner adds only these four new Muon shape costs; D1024 costs
and the assignment algorithm are unchanged. The per-matrix estimates use the
representative owner count, not the old0.01ms unknown-shape fallback:

| Muon matrix shape | Measured count | Median group ms | Group range ms |
| --- | ---: | ---: | ---: |
|1152x1152|6|5.493694|5.460872–5.508509|
|1152x4608|2|5.106647|5.091937–5.115605|
|4608x1152|2|5.528477|5.500950–5.568338|
|9x32|2|0.660949|0.497489–0.665520|

These are amortized planning estimates; grouped cost has launch overhead and
need not scale linearly. Whole-group controls and all AdamW measurements remain
in `run-logs/fs-w144-width1152-optimizer-group-costs-r1.summary.json`.
The existing ownership checker now accepts the exact recorded configuration
and width1152. It must pass replicated/owned parameter and state equality,
None/zero gradients, schedules, cache refresh, padding, cross-rank hashes and
checkpoint continuation before the new plan is used for a complete model phase.
Source146 (`1c70da11`) passed that gate at09:11:01.480307 with clean postflight:
all28 comparisons across four ranks and seven stages are exact, including the
after-timing state. Owned optimizer plus exchange measured31.90–31.92ms versus
replicated68.67–69.33ms. This is not a comparison against the former unknown-
shape ownership plan or a complete-model speed measurement. A nearby same-host
source146 D1024 control is COMPLETE/clean09:23:29.405196: median487.434961ms,
all four model postchecks pass, last loss5.126208782196045. Staged archival
maintenance restores disk headroom before the D1152 whole-model phase.

One fixed-seed42 full-budget request is prepared for after that phase. Its
predeclared cutoff is700ms per step with all numerical checks and clean
postflight. This deliberately permits a slower larger model: it tests the
capacity/time tradeoff, with roughly337million target tokens in1800seconds
at the cutoff before accounting for finalization. It is not a pure speed
comparison. The phase harness clips before averaging and uses its diagnostic
schedule; the quality trainer retains the winning clip-after-average policy
and protected wall-clock schedule. Phase loss is not a quality prediction.
Only a matched20,971,520-token public evaluation will support a BPB comparison.

## Two implementation restrictions can bias a future width comparison

The original `research/scratch/mlp_fused_integration.py:install` applied its 24MiB SBUF
estimate to both `forward_backend="nki"` and `forward_backend="native"`.
The native path calls the ordinary Torch forward and the custom backward; it
does not require both complete weight matrices to reside in NKI SBUF. At a
512-token kernel tile and MLP ratio4, the original estimate plus reserve is:

| Width | Estimate plus reserve | Original install assertion |
| --- | --- | --- |
| 1024 | 23MiB | passes |
| 1152 | 27.875MiB | rejects even native forward |
| 1280 | 33.25MiB | rejects even native forward |

The asserted capacity remains appropriate to the resident NKI variant. The
native width experiment must independently check actual compilation, memory
and numerical behavior after the guard is scoped to that variant. This does not
prove that a wider model is faster or better; it identifies a misleading early
rejection that would prevent the experiment.

`research/scratch/optimizer_ownership.py:make_plan` has measured Muon cost
entries for D1024 matrices and `(8,32)`, with a0.01ms fallback for every other
Muon shape. AdamW cost continues scaling with the number of elements. At a new
width, the planner can therefore treat nearly all matrix optimization as free,
putting it on the two ranks that do not own the large embedding/head matrices.
A metadata-only illustration with24 square/12 MLP matrices and two token/head
matrices shows this concentration. It is not a full GPT or a device benchmark;
the input template and output counts are preserved in
`run-logs/fs-capacity-native-guard-ownership-audit-20260919.json`.

Do not reuse those fallback estimates to decide that width1152/1280 is slow.
Profile the actual grouped optimizer shapes and rebuild the ownership plan,
including collective padding and native batch-one rounding behavior. Recheck
the existing exact optimizer/state-consolidation contract after any change.
Interpolation from cubic work can suggest a starting plan; it is not a measured
cost and must be labeled as such.

## Bounded follow-up

1. Keep the submitted D1024 recipe unchanged. Scope the native installation
   guard in a separate research variant and gate both1024 and the new width.
2. Measure the new Muon groups before changing ownership. Use the real model's
   parameter inventory, not the illustrative template above.
3. Gate the complete optimizer update and complete model step on one host,
   using the winning T1024/pack32 geometry and all current precision choices.
4. Only then compare30-minute quality with a fixed seed and unchanged data.
   Record throughput, updates, exposure and LR trace alongside parameter count.
   This revisits capacity jointly with the new kernel/ownership implementation;
   the earlier depth8 result is still valid for its recorded configuration.

No width sweep is queued ahead of the active FP8 caller diagnosis. Both
observations come from local code, not assumptions about the
other teams' implementations.
