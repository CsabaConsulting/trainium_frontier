# Sole Operator recovery, 26 September 2026

Supervisor transferred the role to `/root/operator_recovery` after the old
Operator's two platform quota failures. The platform errors are not host
failures. Local tools succeeded immediately. No model or billing change,
host mutation, job restart, or new training launch was performed.

Canonical authority is assignment `20260926-operator-recovery.md` at
`d1b060584`, current root AGENTS and `FINAL_SPRINT_ACTIVE_GOAL.md`.
This isolated archive does not change Supervisor state or the old worktree.

The old finite monitor PID 2171819 was present at the initial local check,
then absent at 10:58:38 UTC, before its planned 11:03 end. The exact monitor
and observer process search found no survivors. Last old receipts are from
10:58:05 UTC. Its early exit is a local monitoring interruption; no host
terminal or transport error was emitted in those receipts. Supervisor
explicitly released immediate read-only recovery polling after this absence
check. The proven fixed transport and remote observer body are unchanged.

## Evidence recovered before renewed host contact

All 18 new receipts from 10:52 through 10:58 are copied byte-for-byte into
`recovered/`; `recovered-manifest.json` records every SHA-256 and source path.
Nanosecond integers in the raw files were not parsed and rewritten.

trn21 had 3 completed boundaries at the observation starting
10:52:04.878443 UTC and 51 by 10:52:49.899401 UTC. This is a conservative
fifth-boundary polling bracket, not an exact official startup or startup
charge. At 10:58:05.068751 UTC it had 389 complete boundaries, latest logged
step 380, loss 3.2003, 279489 tokens/s. Guardian 3283774/start 95946276 held
the original lock 66305:596823. Exact official rehearsal was not yet emitted.

trn22 at 10:58:05.060822 UTC had no completed step file or fifth boundary.
Guardian 3400752/start 103355943 held original lock 66305:832589. Its known
training start was 10:45:00.321461 UTC. No terminal, fresh-eval launch, or
fatal line was present. This observation does not prove a cache hit or a
cold start. Both hosts' receipts have transport return code 0 and empty stderr.

Both existing full1800 jobs and attached fresh public20M evaluations remain
protected. No BPB, final checkpoint, terminal budget result, or win is claimed.
The recovery observer is bounded at 11:15:01 UTC and samples every 45 seconds.
