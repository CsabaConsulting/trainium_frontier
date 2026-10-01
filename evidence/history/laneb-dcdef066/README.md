# REVIEW REQUEST ONLY / NOT UPLOAD-READY

This folder is a local Claude audit handoff, not a submission, approval or upload.
Root separately authorized this NEW Downloads copy for review only. No host calls,
Torch/model execution, scientific edits, shared-state edits or upload occurred.

Start with [the eight-item audit brief](frozen-draft/AUDIT_BRIEF.md), then
[the integrity report](integrity-review/REVIEW.md) and
[its scoped result](integrity-review/RESULT.json).

Only checklist item4 (the exact research receipts' measured cold/budget gates)
passes. Items1,2,3,5,6,7,8 remain FAIL/UNKNOWN or require audit adjudication:

- Clean scoring-box imports/dependencies and minimal-environment sufficiency.
- Whole-model native train/evaluation numerical identity and runtime/cache isolation.
- Cumulative checkpoint/log/cache bytes written and current large-leftover inventory.
- Inherited timing-dependent table/stop behavior.
- Retained disabled/compatibility paths and kernel/header documentation gaps.

The report's verdict is PACKAGE_INTEGRITY_ONLY. Its reviewer reused the isolated
session `/root/r136_sb_l2_isolated_verifier` because fresh spawn failed with
`collab spawn failed: agent thread limit reached`. It is NOT a fresh-session
review. It records586 checks, independently hashes13 allowed package members
and all raw archive members in scope; eight package members were deliberately
unread, listed in RESULT.json. It grants no source/numerical, native-equivalence,
clean-box, Claude-compliance or submission acceptance. Export byte verification
below does not broaden that review scope.

Exact source remains dcdef066d8d0827f4ecdf0a80f8517e5348af28eea7acc06e03df4db49d4e90d.
The top-level train.py and launch-command.txt are byte-identical convenience copies
of their frozen-draft counterparts. The exact command remains:

`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0`

Do NOT run the training command as part of this handoff. The two public research
scores are0.9663196934983302 and0.9659050462155375, mean0.9661123698569338. Both use
seed42; they are not official/private scores or independent-seed evidence.
No checkpoint bytes or large raw archives are copied. Their canonical absolute
paths and hashes remain in frozen-draft/REFERENCES.json; this folder alone is not
a self-contained raw-evidence export.

The original draft, including every FAIL/UNKNOWN, legacy-schema diagnostic,
prior no-export prose/flags and paused-work history, is preserved byte-for-byte
under frozen-draft/. Those flags describe the earlier frozen draft; this separate
Root-authorized audit-handoff copy does not turn them into acceptance. The original
integrity report is also preserved unchanged. No synthetic selection or prior
failure was rewritten.

## Read-only verification

EXPORT_MANIFEST.json hashes every copied/generated file except itself; its own
SHA256 is supplied with the handoff. Supply that exact hash to the stdlib checker:

`python3 -I -B verify_export.py --manifest-sha256 <HANDOFF_MANIFEST_SHA256>`

Add `--compare-canonical` on the originating laptop to compare all25 frozen-draft
and review files, plus the two source/command convenience copies, to their canonical
originals. No model, original package helper or host code is executed by this checker.
The original package checkers/builders under frozen-draft/ are retained audit
material tied to canonical repository paths; do not run relocated builders.
Full raw-evidence replays remain the canonical commands documented in the reports.

Claude acceptance: UNKNOWN/pending. Fresh independent package/source/compliance
review: not supplied by the reused-session integrity report. Upload authorization:
FALSE. Request review of the explicit gaps; do not describe this as upload-ready.
