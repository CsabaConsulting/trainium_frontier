# Initial workflow-v2 backlog

No experiment is ready or launch-authorized. Queue ordering is preserved exactly in `queue.json`; this
file records only the blocking installation handoff.

1. **G1 — pending in main working context.** Produce and independently review the real `prepare.py`,
   evaluator-CLI, shard, startup-budget, and host-layout receipt. Do not import an observation as a PASS.
2. **G2 — pending independent Verifier.** Review the installed bytes and receipts, reproduce the exact
   build identities, and issue a receipt without changing `workflow/tools/decide.py` or its thresholds.
3. **Native gates — pending.** Research-v2 and S0 have no native clearance from this installation.
4. **Only after gates and authority checks:** materialize jobs from the seeded queue. A queue entry is
   not launch authorization.

The prior selection/control launcher is inactive. Do not start workflow v2 until the authoritative
state changes from `launch_authorized: false`, and never run two launchers concurrently.
