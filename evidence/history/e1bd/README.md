# REVIEW REQUEST ONLY — e1bd CE2 + FLATCACHE + ZBUNDLE

Start with AUDIT_BRIEF.md and its eight-item checklist. Package audit PENDING; not upload-ready. Exact source e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310; 2 authenticated public research FULL receipt(s), mean 0.9671172862498154. Historical source review is not a fallback-package Claude verdict. R151 LaneB official scoring failure and current diagnostic uncertainty are disclosed. No acceptance, upload or run authorization is inferred.

Only train.py is proposed for a future submission. audited_parent.py, verifier and all evidence files are reviewer material, not model dependencies. Source and command are unchanged. All prior packages remain untouched.

From this folder, verify using the separately supplied exact manifest pin:

`python3 -I -B verify_export.py --manifest-sha256 HANDOFF_MANIFEST_SHA256`

Use --compare-canonical only on the originating machine to check pinned repository metadata. No model import or execution is performed. Original source/CPU/full gate metadata can be reauthenticated from the repository with `python -B research/v2/submission-prep/20260930-r151-fallback-packages/build_packages.py --check-only`; this does not export or launch anything.
