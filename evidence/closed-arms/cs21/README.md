# CS21 exact-source early audit — READY_FOR_CLAUDE_AUDIT

R191 early platform-eligibility evidence is available: independent source/CPU
acceptance plus actual native training3018 steps, cold606.0083909828681s (<900).
**Claude's CS21 source verdict is pending; this export does not claim UPLOAD_READY.**
Fresh research public20M, canonical FULL validity and official score remain NULL.

Candidate train.py SHA256: `e056552274ac7ed3ec977589cda03428984ec373d1dc724e42f2698841645f82`.
Parent T3 SHA256: `2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894`. Only the auxiliary training schedule amplitude
changes0.5→0.75. Model/eval/kernel source and command otherwise match T3 exactly.

```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0
```

Caller NEURON_CC_FLAGS stays unset as in the released job; source-owned O3 main
and import defaults are unchanged. Read AUDIT_BRIEF.md, source-cpu-review.json,
SOURCE_BYTE_AST_PROOF.json and EARLY_NATIVE_EVIDENCE.json. EXPORT_MANIFEST.json
indexes all files other than itself. FULL_RESULT_PENDING.json records explicit
future NULLs; the actual status is NOT a canonical FULL receipt or score.

The8192-byte train/rehearsal.json tail is preserved as a fragment, not silently
repaired or called complete. Raw status/packet/CPU envelopes preserve exact bytes.
No source/evaluator/tokenizer edits, host calls, Torch/model execution, checkpoint
tensor copies or upload were performed. Unknown AMI/I/O/backup/current-retention
telemetry is not converted to PASS or introduced as a new audit prerequisite.
Claude audits this copy; Poojith alone performs any later approved upload.
