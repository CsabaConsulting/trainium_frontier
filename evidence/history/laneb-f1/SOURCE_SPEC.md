# R158 LaneB-F1 LNCFIX — Variant A, proof pending

Candidate `train.py`: `02c758a9bd980819333efeb48557e26e44c680c4e683f8eaf491782574d92c38`.
Parent `../20260930-r113-laneb-wrapped-ce2-flat-z/train.py`:
`dcdef066d8d0827f4ecdf0a80f8517e5348af28eea7acc06e03df4db49d4e90d`.
SOURCE.diff: `aebce799a141af2fc2ab9594e3ac045921f0e70735e721cf2485d1d753a6374b`.
Authority R158 snapshot: `6bb0d978c907191836691aab82cd7d5e04523ba6f978cb3d6ff6be1872242a3e`.

Exactly seven reversible edits: the requested single header sentence; one import-time
`_NKI_LANES` assignment (2 only when `NEURON_LOGICAL_NC_CONFIG` equals literal
string `2`, otherwise 1); cap the original RoPE choice by this value; Conv2 lanes2
only for even batches AND cap2; cap the original selected_lanes rule; replace the
two LaneB forward/backward attention handle grids `[2]` with `[_NKI_LANES]`.
Z10 remains True. No kernel body, tensor operation, parameter, initialization,
optimizer, data, scorer, checkpoint, budget, command or other source byte changes.
The exact inverse recovers the parent. Only `_rope_lanes`, `_Conv2NKI.forward` and
`selected_lanes` function ASTs change; attention changes are module handle binding
expressions. All other function ASTs are identical. Shared train/eval dispatch only.

Under the unchanged command
`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0`,
all original grid choices are preserved, including both LaneB attention grids2.
Unset,1 or any value other than literal2 selects cap1 and the same original kernels.
This is a source-level dispatch identity, not measured native graph/timing/score.

Root reports actual R157(c) archive
`649154454091811b6b861ce6670f7cb1f45edd00b8f78c98f0db010192f986de`
at `operator/20260930-r157-laneb-rope-attention-eval-diagnostic-trn21-preparation/cases/c-laneb-rope-attention-one/collect-20260930T083652635337Z.json`:
original LaneB source with process-only RoPE1/attention1, LNC1, B1/T1024/public2M,
returned `0.9608059277535552` exactly, exit0 and Guardian released. It enables this
source build per R158; it is not a run of this new source. Conditional case(d) is
not enabled by the successful(c).

Author instance-CPU proof is bounded stdlib selector AST/source identity only, not
a model/Torch/SDK import. It must additionally test both actual attention binding
expressions using inert handles, cap1 versus explicit2 and exact parent identity.
Independent source/own-instance-CPU review remains separate. Actual author, own,
final review, exact-source diagnostic/native/FULL evidence remains NULL.

This source does not independently select a host or waive proof. Root binds the
required actual exact-source scorer-condition proof and matched native evidence
under current R157/R158 authority, and the FULL only after all applicable gates.
One valid fresh20M/T1024 FULL package receipt, seed42/reserve15, cold900/charged1800,
original locks/Guardian/25GiB entry/22GiB runtime remain mandatory. No numeric
within-noise cutoff is invented. No automatic adoption, package acceptance or upload.
