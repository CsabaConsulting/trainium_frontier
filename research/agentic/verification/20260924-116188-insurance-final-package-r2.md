# Final insurance-package re-verification

Verdict: **PASS for technical packaging only** at commit
`7e0ab71b8eefbdc4908b73ccb6f7e20218378c0f`. This supersedes the metadata
HOLD recorded at verifier commit `4506b96b4ac55055e415a9442bcb61b3a38a7216`.

The packaged `train.py` remains exactly 85,557 bytes with SHA-256
`116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.
It is byte-identical to the canonical candidate and its pinned evidence object. The
launch file remains exactly 246 bytes with SHA-256
`b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd`;
the corrected documentation accurately reports one trailing newline and no embedded
newline.

`SHA256SUMS` verifies and covers all nine other files in the package, intentionally
excluding only itself. Independent manifest replay verified 20 exact path/size/SHA
bindings: all 13 prior receipts, the repair-basis HOLD receipt, and all six result,
runtime-job, and postflight JSON files for the two cold runs. There were no missing or
extra inventory entries.

Fresh offline checks passed: the official `prepare.py check` using the documented
dependency-complete Python 3.12 environment; the confirmation-r2 static/Decimal test;
Python compilation; and a synthetic CPU checkpoint round trip with strict state load,
exact logits, frozen/eval state, no hooks, and state identity. AST review found exactly
one ordinary `load_for_eval`, one `torch.load` with `map_location='cpu'` and
`weights_only=True`, and no hidden eval wrapper, hook, monkeypatch, alternate weights,
pretrained source, or external data path. The three import-time `os.environ.setdefault`
calls are accurately disclosed: they provide compiler/cache defaults when absent and
do not create a separate evaluation model or forward path. Secret and prohibited-file
scans passed.

The package remains deliberately unapproved: `submission_approved` and
`human_upload_authorized` are both `false`. Technical packaging is complete, but the
full text of the prior portal rejection is still unavailable. The remaining decision
is human-only: recover that explanation, or explicitly accept the residual uncertainty
that it could contain a reviewer concern not visible in the truncated records, and
then separately authorize one human-controlled upload. This receipt neither makes that
risk decision nor authorizes or performs a submission.
