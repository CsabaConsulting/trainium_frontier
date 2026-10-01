# A3 conditional upload preparation — NOT PACKAGE READY

Authority: R73 received07:53:47 IST. Export only after ALL THREE exact-c5ef
full cold/fresh20M receipts qualify and arithmetic mean<=0.97225. Do not select
the best two or call this an independent-seed result: every run uses seed42.

Scored source c5ef8414e2df998a9b52eeb2e249b66c0110a3c37869e676719e93975a3432b4.
Prepared upload derivative29e5cd2cfd35bb52e5a955057a1661786239d0812d9a797c85dc0df7fe2d7446.
Only three replacements: argparse default freeze286M->0; correct BF16 table/RMS
header claim; correct CLI-options header sentence. Direct build_config default
and all model/training/evaluation code stay unchanged. No scored source changed.
Independent structural review PASSED, receipt813bb4bf193738f86c98286a147ba3f445fffa06462e541ebc72dce12f2e5b7a
in research/v2/verification/20260929-r73-a3-upload-structural/RECEIPT.json.
Root reran the independent stdlib checker08:03; exact parser/config/normalized
AST checks PASS. This is not a new full run or Claude compliance ACCEPT.
No files exported to Downloads yet.

## Exact run receipts

1. trn21-r71-a3-full-r1: archive
   research/v2/operator/20260929-r71-a3-full-trn21-preparation/artifacts/results-bundle.json
   d38a9edd82b3c451a2f01825a404281a87c3c1fd9a3c03a1b73b4b9656c55e4b.
   Public20M0.9719336457209543; cold676.520512s; charged1792.683746s.
2. trn21-r72-a3-best-single-filler-full-r1: currently live; future archive
   research/v2/operator/20260929-r72-a3-best-single-filler-full-trn21-preparation/artifacts/results-bundle.json.
3. trn22-r72-a3-filler22-retry2-full-r1: currently live; future archive
   research/v2/operator/20260929-r72-a3-filler22-retry2-full-trn22-preparation/artifacts/results-bundle.json.

Authenticate full member hashes, source/argv/seed/config, terminal return codes,
six fresh empty caches, official startup and charged-through-exit accounting,
strict checkpoint/hash stability, exact-zero finite four-rank table diagnostic,
fresh20971520-token/context1024 evaluation and original Guardian release for ALL
three. Missing or failed evidence is not a score qualification.

Scored command:
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0
Upload command, after verified equivalent default0:
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py

## Export contents if adopted

NEW Downloads/Trainium-A3-CLAUDE-AUDIT-20260929-29e5cd2c/: train.py (upload
derivative), scored_train_c5ef8414.py, SOURCE.diff vs exact scored source,
DIFF_AND_MANIFEST.md, AUDIT_BRIEF.md, cold-evidence.json, all three source/run
receipts and independent source/config proofs. Include scored-source diff vs
D2 and explicit distinction between scoredSHA and uploadSHA. No checkpoints,
datasets or SDK files need to enter the upload. Preserve D2 package untouched.

Checklist in final AUDIT_BRIEF: each of the eight rejection items PASS/FAIL with
exact evidence: single-file/dependencies/command; same trained causal train/eval
path; no forbidden reads/network/weights/local paths/cache dependency; cold and
charged budgets; artifact bytes/disk evidence (unknown cumulative IO stays
unknown, never inferred from retained size); standalone defaults; deterministic
seed/time-schedule; readable source and kernel docstrings. Do not substitute a
structural precheck for Claude's compliance audit or label a pending check PASS.

Required README framing:
"hashed bigram embedding table (modded-nanogpt / nanochat / Engram lineage),
zero-initialised, learned from the provided training data inside the timed loop
by RMSprop with gradient accumulation over two steps, saved with the model,
identical train/eval forward, a pure function of token ids at positions <= t;
harness untouched."

A3 replaces the learned layer7 128->1024 projection with a fixed repeated128
map to all eight heads, retaining each existing learned per-head gate. Embedding
site/table/every2/FP32 sink remain D2. Other two receipts and the ALL3 mean are
pending; no adoption, upload or promised official gain.
