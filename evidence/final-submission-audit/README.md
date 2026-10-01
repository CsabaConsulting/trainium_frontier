# T3 exact-source audit package — UPLOAD_READY

**UPLOAD_READY under Claude Round191.** Claude accepted exact source2b4b and both
FULL receipts; the remaining finite-unset condition is now satisfied. Both own-
checkpoint public2M diagnostics passed: LNC1 and unset each returned exactly
`0.9590707374259858` (delta0), B1/T1024/2097152 tokens on the same protected T3
checkpoint `029510d8f1eba40e9e81f4e93b12665fea158785945b6fda33bfd3b85306baab`.

Upload payload: **train.py**, SHA256 `2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894`.
Current complete file index: **FINALIZATION_MANIFEST_R191.json** (all current
files, excluding the manifest itself). Read READINESS_R191.json,
INTEGRITY_REVIEW_R191.json and AUTHORITY_R191.txt for exact evidence and authority.
No source/evaluator/tokenizer/command change was made by finalization.

```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0
```

Leave caller NEURON_CC_FLAGS unset to reproduce the tested source-owned main
O3+BF16-auto-cast and import/eval O3 defaults. Two research fresh public20M/T1024
FULL scores: `0.964447246764078` (trn21) and `0.9642283417620233` (trn22); exact
decimal mean `0.96433779426305065`. Both seed42; not independent-seed evidence or
an official platform score. Both52-member FULL verifiers and both diagnostic
readers were replayed without hosts or tensor/model execution. Independent
byte/AST checking confirms only the two O1→O3 default strings changed from
LaneB-F1; all model/training/eval function ASTs and post-import-Torch bytes match.

Historical provenance: EXPORT_MANIFEST.json remains the unchanged R186 export
manifest, **not** the current file index. Its original README and a duplicate
of that manifest are preserved verbatim in `history/r186-pre-r191/`; PRESERVATION_MAP.json maps
all29 original members. Every original proof/raw/checklist and the R188_R2
diagnostic append remain byte-identical. Original pending markers are historical;
READINESS_R191.json and CHECKLIST_ADDENDUM_R191.md record their later resolution.
No existing Claude verdict was overwritten or manufactured.

Pristine organiser-image sufficiency, unrecorded cumulative/transient I/O, complete
current leftovers/backup state and current checkpoint presence remain UNKNOWN.
They are not silently marked PASS or made new gates. No credentials/checkpoint
tensors/new data/SDK installation/host action were added. **No upload was
performed. Poojith retains the upload action.**
