# Human upload package checklist

- [x] `train.py` is exactly 85,557 bytes and SHA-256 `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.
- [x] `launch-command.txt` contains one bare command and has SHA-256 `b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd`.
- [x] The package does not contain root `train.py`, `prepare.py`, evaluator/tokenizer files, checkpoints, credentials, or S0 private bytes.
- [x] Two cold seed-42 full-budget runs and their exact fresh public20M metrics are bound in `EVIDENCE_MANIFEST.json`.
- [x] Existing reviewer, CPU/load-for-eval, checkpoint-finite, rehearsal, metrics, result, and operator receipts are hash-bound.
- [x] Known authoritative rejection fragments and the missing full-text limitation are recorded without inference.
- [x] Offline packager checks are recorded in `OFFLINE_CHECKS.md`.
- [ ] An independent verifier has audited this exact staged package and issued a final package receipt.
- [ ] The human has explicitly approved one rate-limited leaderboard upload.

`submission_approved` is **false**. This directory is staged for review only and must not be uploaded yet.
