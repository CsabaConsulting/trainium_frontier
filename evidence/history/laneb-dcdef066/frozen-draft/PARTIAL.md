# PARTIAL — paused for R137 T2 FULL22 preparation

Local evidence discovery only. No package acceptance, independent audit, Claude audit,
export, upload, launch, model execution or host access. Source copy, eight-item
AUDIT_BRIEF, final manifest and reproducible local checker are NOT YET BUILT.

Canonical repository: `/home/poojith-devan/Desktop/Trainium`.
Exact source: `research/v2/proposals/20260930-r113-laneb-wrapped-ce2-flat-z/train.py`,
SHA256 `dcdef066d8d0827f4ecdf0a80f8517e5348af28eea7acc06e03df4db49d4e90d`.
Exact command (unchanged):
`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0`.

## Read-only diagnostic findings (not an independent audit)

Pinned FULL verifier `research/v2/verification/20260930-r132-current-full-receipts/verify.py`
SHA256 `ace0b4db184c9269d9d030b82a38314f7a7acbe4384a93c13c37e23575863b37`
and its frozen manifest were authenticated. Its first unmodified attempt on FULL21
refused with `KeyError: R98_ROOT_SELECTION.json`: the actual legacy packet contains
`R113_ROOT_SELECTION.json`. A diagnostic in-memory replacement of exactly that member
name, with every other verifier byte/check unchanged, authenticated the FULL21 gates.
This adapter has NOT yet been saved/reviewed as a package checker. Repeat22 passed
the unmodified pinned helper. No original verifier or packet was edited.

| Research receipt | Absolute canonical archive suffix under repository | Archive SHA256 | Public-20M BPB | Cold seconds | Charged-through-exit seconds |
| --- | --- | --- | --- | --- | --- |
| FULL21 | research/v2/operator/20260930-r113-laneb-wrapped-full-trn21-preparation/artifacts/results-bundle.json | 05a764083999229bb92dc7a9c3b177480dcdb3d0f526cdc78bdb884c73645bac | 0.9663196934983302 | 663.9014979940839 | 1793.414417522028 |
| Repeat22 | research/v2/operator/20260930-r127-laneb-samebyte-repeat-full-trn22-preparation/artifacts/results-bundle.json | defb95ea5df8ac81778e4f0c637ab44846b779ceda006b3b6b16de748e962b38 | 0.9659050462155375 | 645.364225326106 | 1792.7041728659533 |

Mean: 0.9661123698569338. Both seed42; not independent seeds and not official scores.
Both raw receipts passed original terminal/Guardian, exact source/command,
six-empty-caches, stable checkpoint, fresh20971520/context1024, finite recorded losses,
four-rank table zero-difference and pending0 gates in the diagnostic metadata calls.
FULL21 checkpoint d1fb7a623a3a537fe4d559d723831626a0989a7dbe2d1c1cb78fa579dbc6253b;
repeat22 checkpoint 8334b60bdb693d03b72bbb259892ee3af24a16ab5f99322d9acdd81c80cb4c53;
both 587788862 bytes. No checkpoint copies/backups made or claimed. Present remote
retention is UNKNOWN (no host calls).

## Exact parent and outstanding checklist findings

Direct parent: `research/v2/proposals/20260929-r109-ce2-flat-zbundle/train.py`,
SHA256 e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310.
Existing exact SOURCE.diff SHA894a76cc54f31fa870af864cdd239c6ecdae69d01492848eaa4818f414e0e910.
Parent FULL21 archive088637f6f4db8ce993f5cc52e80ccc95f5e3441aacb5cc06ce88e59af2be1a0f,
packet28d29a6ea0325e320164672e46ea29bb3d4104a58cd8df94279dccce396a5584,
at `research/v2/operator/20260929-r109-ce2-flat-z-full-trn21-preparation/artifacts/`.
Same diagnostic member-only R98->R100 adapter passed raw parent FULL gates,
public0.9672630039667548. This is research evidence, NOT proof of a Claude-audited
ancestor. Last Claude-audited ancestor remains UNKNOWN.

Primary checklist: workflow/REJECTION_CHECKLIST_2026-09-28_0315.md.
Unknowns must be FAIL pending evidence, not presumed PASS. Missing: pristine clean-box
dependencies/minimal-environment evidence; exhaustive native whole-model train/eval
numerical equivalence; cumulative bytes written for checkpoints/logs/caches; current
leftover inventory; fresh independent package review; Claude audit. Archive raw_bytes,
checkpoint sizes and free disk are not cumulative write measurements.

Static findings requiring disclosure in the eventual eight rows:
source lines36–39 retain main/import compiler-default difference; main CLI line1989
defaults tbl_freeze_tokens to0 (build_config line1419 legacy default286000000 remains);
no AST `.training` attribute exists; load_for_eval lines1920–1936 strictly restores
weights and freezes model, rotary capacity8192 does not change actual evalcontext1024;
MLP lines1153–1163 keeps dtype/tiling fallback and custom _FusedMLP/_fused_mlp_kernel
lack docstrings; Z switches and dormant D1 injection branches remain. No source edits.
Timing-based batch/schedule, table force/deadline and stop/checkpoint guards must be
explicitly discussed, not represented as bit-identical repeat trajectories.

Existing scoped SOURCE_CPU review:
research/v2/verification/20260930-r113-laneb-fresh-scored-review/REVIEW.json
SHA c5ebb3320e5c5b329beff5dcecac61eb80a6a98a23592d74c5e105caeac841f8.
Author archive fbd77a18d2683ac09b3ec8e45c0c24a2498b4fb9014de2f4f5dc4faef1940fe1
has10groups; own95cc6f1bafd0e50790f215121230cf387a46436f1ac9774c086b052ce0d3cae8
has8groups/125metrics. Review scope explicitly CPU surrogates, not native model/eval
parity or submission acceptance. These were inspected/hash-referenced, not rerun.
