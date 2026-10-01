# R158 e1bd-F1 LNCFIX — Variant A, proof pending

Parent: `../20260929-r109-ce2-flat-zbundle/train.py`, SHA256
`e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310`.
Candidate `train.py`: `25c5aec53ad7cc20f4dadf437d3ebe6dfa4f4be0214ab39d638a45df5ec7f936`.
Diff: `7a8f733effaf815a25c1e001fc51aaaeaaba486749b0d86814ac08b02cbd2b80`.
Authority R158 snapshot: `6bb0d978c907191836691aab82cd7d5e04523ba6f978cb3d6ff6be1872242a3e`.

Exactly five reversible edits: the requested single header sentence; one import-time
`_NKI_LANES` assignment (2 only when `NEURON_LOGICAL_NC_CONFIG` equals literal
string `2`, otherwise 1); cap the existing RoPE choice by this value; allow Conv2
lanes2 only for even batches AND cap2; cap the existing selected_lanes rule.
Z10 remains True. No kernel body, tensor operation, weight, initialization,
optimizer, data, scoring, checkpoint, budget, source command or other source byte
changes. Exact inverse edits recover the original parent bytes. AST changes are
only `_rope_lanes`, `_Conv2NKI.forward` and `selected_lanes`; all other functions
are identical. The cap is shared by training and evaluation, not an eval-only path.

Under the immutable original command
`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0`,
all existing grid choices are unchanged. This is a source-level equivalence claim,
not a measured native graph/timing/score claim. With unset,1 or any value other
than literal2, the same kernels use at most one lane. Kernel loops remain original.

Root reports actual R157(b) raw archive4cf149f0795599cffb19118ee3f6cbd09a9befdcedfd99a59ae931d942fc8f9e
under `operator/20260930-r157-e1bd-rope-eval-diagnostic-trn21-preparation/cases/b-e1bd-rope-one/collect-20260930T082523328036Z.json`:
process-only RoPE1 on original source returned0.9617131518505532 exactly. That
receipt chooses Variant A; it is NOT an actual run of this new source.

Author worker is instance-only, stdlib-only: authenticate candidate/base, exact
inverse edits and unchanged function ASTs; evaluate only the three selector
expressions extracted from their authenticated AST, never import the model or
SDK. Exercise unset,1,2,empty,0,3 and02 environments; valid/invalid row counts,
sequence tile divisibility and odd/even Conv2 batches; assert original choice
identity under explicit2 and cap1 elsewhere. No mathematical kernels or model
CPU loop is justified by this source-only dispatch change. This is author evidence,
not independent review or native proof. One CPU nice19, CPU45/50s, wall60s,
AS8GiB, sampled RSS2GiB, stdout/stderr1MiB each, FSIZE1MiB, no cores; original
private CPU mutex, safe warm admission and owned-descendant reaping are external.

Required subsequent proofs, in R158 order on trn21 after diagnostics:
1. Exact candidate loading unchanged protected checkpoint3a0004d915883002980c884fa16a07aec76eed812cb70c683d96cec807ce683f,
   B1/T1024/public2097152, LNC1 then unset. Both must equal0.9617131518505532
   exactly using unchanged prepare.evaluate_bpb. No process-only source override.
2. Same-host60-step timing under original explicitLNC2 command versus exact e1bd
   control, equal within noise. R158 does not specify a numeric noise tolerance;
   Root/Claude must resolve the comparison from actual matched receipts, not
   invent a threshold. Original cold<=900/charged<=1800 safeguards stay.
3. One FULL21 package receipt, seed42, reserve15s, fresh public20M/T1024, original
   validity/locks/disks/Guardian. No FULL authority exists in this source packet.

Independent source/own-instance-CPU review is separate and pending. All actual F1
CPU/diagnostic/timing/FULL receipts remain NULL. No automatic acceptance/adoption.
R158 permits describing the original e1bd receipts as the second receipt only
after proving unchanged explicitLNC2 training and obtaining F1's required proofs;
that eventual audit statement is not made as completed here. No upload/export.
LaneB-F1 remains separate and must await actual R157(c).
