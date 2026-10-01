# LaneB exact-source two-receipt audit draft — repository only

This is an evidence package for Root review, then a fresh separate verifier and
Claude audit. It is NOT an accepted or submission-ready package. No export,
Downloads write, upload, host call, launch, model execution or scientific change.
R132's 23:00 tier-1 audit milestone is preparation context, not permission to
assume that any checklist item passes. `PARTIAL.md` is the preserved earlier
discovery/paused-work record; this README describes the completed local draft.

The sole proposed model payload is `train.py`, copied with apply_patch from the
canonical exact scored source. All audit helpers and JSON files are local review
material, not model dependencies or proposed upload payloads.

- Source SHA256: `dcdef066d8d0827f4ecdf0a80f8517e5348af28eea7acc06e03df4db49d4e90d`.
- Command: `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0`.
- Public research receipts: 0.9663196934983302 on trn21 and 0.9659050462155375 on
  trn22; mean0.9661123698569338. Both seed42, not independent seeds or official scores.
- `AUDIT_BRIEF.md` contains the mandatory eight checklist rows. Unknowns are FAIL
  pending evidence; no blanket acceptance is issued.

`cold-evidence.json` records the exact raw source/command, terminal/Guardian,
empty training caches, cold<=900, charged-through-exit<=1800, stable checkpoint,
fresh20971520/context1024 and four-rank table diagnostics. It also explicitly
records UNKNOWN cumulative writes and present checkpoint/leftover inventory.
The evaluation cache is separate but historically reused; it is NOT claimed empty.

`DIFF_AND_MANIFEST.md` and `SOURCE.diff` compare against the exact direct parent
e1bd539f, whose raw research FULL receipt is authenticated in ancestor-evidence.json.
Last Claude-audited ancestor is UNKNOWN. This is not a claim that the parent or
current source has passed a submission audit.

No giant raw archive copies are included. `REFERENCES.json` pins absolute canonical
file paths, SHA256 and byte sizes. Each receipt also indexes every raw member by
name/hash/size. Exact paths are dependencies of this local draft; copying this
folder alone does not export a self-contained evidence archive.

The pinned ace0b4db FULL verifier is preserved. Its first unmodified call refused
the legacy FULL21 selection-member name; LEGACY_SCHEMA_DIAGNOSTIC.json preserves
that refusal. The only in-memory adaptation is the exact real legacy member name
R113_ROOT_SELECTION.json (R100_ROOT_SELECTION.json for the direct parent). The
original packet/archive bytes are never changed, no selection is synthesized,
and all original verification gates remain. Exact adapter diffs are included.

Run the read-only local check from the canonical repository:

```text
python -I -B research/v2/submission-prep/20260930-r136-laneb-two-receipt-audit/verify_local.py
```

It verifies the manifest, external pins, exact copy/command/diff, and reauthenticates
the fixed actual receipts and existing scoped sourceCPU evidence. It imports only
stdlib metadata helpers, never train.py/Torch. Successful metadata verification
does not promote the package to PASS or authorize submission. Do not run the
training command or numerical workers as part of this task.

Root must review the stated gaps, obtain fresh separate package verification and
Claude audit, and resolve the human-only export/upload decision. Missing evidence
must not be filled with inferred PASS, invented totals, claimed backups, or source
cleanup disguised as packaging. Any later source edit requires a new source SHA
and the applicable review/receipt workflow; this draft changes no defaults.
