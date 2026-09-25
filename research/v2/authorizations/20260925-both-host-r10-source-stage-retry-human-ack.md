# Human decision — one new r10 source-stage attempt per host, conditional

Recorded 2026-09-25 12:22 UTC from the user's direct reply to the
Supervisor's question in this task. The question described that both
machines' one-use source-staging attempts failed without creating a checkout
and asked whether, **if read-only diagnosis identifies a fix and independent
safety review passes**, the team may make one new, uniquely labelled staging
attempt on each machine, without starting training, deleting files, or
changing AWS resources. The user's answer was: **“Yes, one safe attempt per
machine.”**

This is limited human approval for at most one *new* trn21 and one *new*
trn22 source-stage attempt under those conditions. It does not authorize
reusing either consumed r5 label/marker, retrying before a reviewed diagnosis,
cleaning a partial artifact, a breaker reset, native training, evaluation,
AWS actions, candidate adoption, or leaderboard submission. Each possible
attempt still requires its own distinct completed request, independent final
verification, fresh full local/host preflight, and exact one-attempt
Supervisor assignment under the active repository contract. A failed gate
remains HOLD.
