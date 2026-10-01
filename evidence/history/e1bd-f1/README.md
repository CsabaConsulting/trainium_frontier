# R163 source audit handoff — FULL pending

READY FOR SOURCE REVIEW ONLY, under Round 163. This copy is intentionally
exported before the running FULL completes. Source, exact diff, independent
CPU evidence, both exact-source public2M diagnostics, matched60 evidence,
and unchanged launch command are included. No upload authorization.

The older draft status/checklist below describes missing final evidence;
it does not block this source review. Actual FULL score, cold/budget receipt
and checkpoint evidence will be added as separate files when collected.
Claude may review now and append the FULL check later. Unmeasured telemetry
is explicitly unknown, not an additional instrumentation requirement.

Only train.py is the proposed submitted payload. All other files are audit
evidence. Parent receipts remain historical under the R158 rationale.

## Preserved draft notes

# e1bd-F1 export draft

Status: EXPORT_DRAFT_NOT_READY. This directory is local factual packaging preparation only. `train.py` is the sole proposed model payload. Neither this directory nor the historical parent package is an acceptance or upload authorization.

Read AUDIT_BRIEF.md, the exact human REJECTION_CHECKLIST.md, EVIDENCE.json and the raw receipt index. Future FULL fields intentionally remain NULL. All copied inputs are byte-identical to hash-bound existing local artifacts; checkpoints/model tensors, caches, private data and credentials are not copied. Existing Downloads packages and live state are untouched.

Raw JSON archives without a final newline are stored in explicit `exact-raw-bytes-envelope-v1` containers: base64 decoding `raw_bytes_b64` recovers every original byte and the declared original hash. They are never silently newline-normalized. These wrappers are listed in EXPORT_MANIFEST.json; source and ordinary copied text remain direct byte-identical files.

The local builder performs only stdlib metadata/hash/JSON/AST checks and additive apply_patch writes. It has no host, model or numerical execution path. A later actual FULL receipt requires a separately reviewed additive update/new package; the builder does not watch, launch, accept or submit anything.
