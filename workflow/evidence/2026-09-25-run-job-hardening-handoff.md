# run_job mechanical-correctness hardening — Implementer handoff

Status: local implementation and self-test PASS. This is a one-host pilot candidate, not authorization
to replace the working legacy launcher, access a host, launch training, or delete stored artifacts.

Base: `8b9fc841ed817eac263c7087c3bce7c9f008bda1`.

Bounded changes:

- uses the legacy shared host/device lock
  `/home/ubuntu/.cache/trainium_freshstart/device0.lock` and validates the actual hostname;
- refuses every launched/claimed label while permitting a clean preflight-only retry;
- checks the merged launch environment, full Git cleanliness, immutable source/prepare/G2 pins, and
  the complete production G2 schema;
- accepts public-20M facts only from one exact organizer JSON result and recomputes BPB;
- runs a separate unchanged-`prepare.causality_check` process bound to the exact source/checkpoint;
- seals the checkpoint through eval/causality, archives with no-overwrite/fsync/readback, then durably
  records, appends the locked ledger, writes the verdict, and only then prunes an unprotected exact
  inode/path/hash;
- writes non-overwriting stage receipts and rejects path traversal, symlink roots/checkpoints, dirty
  source, duplicate ledger labels, malformed summaries, and missing numerical evidence.

Local checks:

- `python -m py_compile workflow/tools/run_job.py workflow/tools/check_causality.py workflow/tools/fake/*.py`
- `bash -n workflow/tools/test_run_job.sh`
- `git diff --check`
- `bash workflow/tools/test_run_job.sh` — PASS, seven ledger rows, zero leaked tagged processes.
- `workflow/tools/decide.py` remains byte-identical at
  `a47bedd258697b0791f48dcac3098d9071b5aaf00a7664d9e39987b4052adc66`.

Required next boundary: an independent Verifier must review the diff and rerun the local suite. Only
after PASS may an Operator stage the tool on one approved host and run its local self-test there. A
production job additionally needs the new immutable job/template fields and full non-stub G2 receipt.
