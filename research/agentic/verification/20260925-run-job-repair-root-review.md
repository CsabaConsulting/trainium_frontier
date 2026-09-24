# Independent repair verification: new run_job pilot only

Reviewer: root, independent of Implementer. Time: 2026-09-24 20:55 UTC.
Frozen branch `fix/20260925-run-job-mechanical-correctness`, commit
`d7733fbae5d89d968ad2b6937e4b81defe309442`.

Result: local synthetic regression PASS; one-host synthetic pilot still required.
This does **not** authorize production migration or a scored run.

Root reran `bash workflow/tools/test_run_job.sh` against the frozen clean tree. Process exit 0;
final `PASS rows=7 no_leaked_processes`, scratch `/tmp/tmp.0RFEBiwoYw`. This final run supersedes
an earlier non-authoritative run while the Implementer was editing the test tree.

Reviewed repairs to the separately reported three failures: post-Popen receipt failure now
stops/quiesces the tagged child; evaluator/causality outer timeouts cannot accept printed success;
extra checkpoint or model-source arguments cannot redirect evaluation. Root additionally reproduced
an argparse abbreviation bypass in the previous repair (`--checkp`, `--train-p`). Final code uses
an exact five-option evaluation schema and final regression rejects both full and abbreviated forms.
The final suite also covers duplicate labels, source drift, effective environment, dirty source,
host-lock contention, malformed measured eval JSON, causality identity and timeouts.

Current limitations: explicit seed/time argv is required by this pilot and so it cannot execute
the exact already-tested default-argv A0 or bare S0 without a separately reviewed compatibility
change. Production checkpoints are protected pending human approval. No decide.py changes.
Do not switch launchers merely because synthetic tests pass. Continue the separately reviewed
legacy warm A0 path; this repair met the local-test timebox without blocking that path.
