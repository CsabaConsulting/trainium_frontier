# Capacity checks after the kernel gains

This is a source-audit follow-up, not a new BPB result. The frozen standalone
submission remains source129, reproducing source98 public BPB0.9928513116311253.
The cold training and evaluation rehearsal passed. The caller-profile diagnosis
remains active; capacity preparation proceeds independently of the frozen package.

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
are pending; no new width costs have been inserted into the planner.

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
