# Trn21 cleanup versus Trn22 native-gate migration

Date: 2026-09-25  
Role: Analyst  
Primary recommendation: **prepare Option B (fresh same-host G0/G1/G2 on trn22)**  
Execution status: **HOLD — neither option is presently authorized or storage-complete**

This report is analysis only. It authorizes no implementation, host access, cleanup, deletion,
request materialization, gate run, training, G2 advancement, AWS lifecycle action, or submission.

## Evidence reviewed

- G1 r3 disk-failure evidence commit
  `5e45a564bba0d04b052345a7c9c1812ddb71ce51`, receipt SHA-256
  `2325354ee0660fe713a64b23e102c2b53b9c8cab4ca6a5ab081f8850cb322946`.
- Disk-failure analysis commit
  `83e533546fb8b53f5c8268d84f0ccd2b5c90e1a1`.
- Cleanup implementation r1 and verifier FAIL:
  `e4f94585a988972d9428d0d665ea9fa3ba8aa035` and
  `e1152cb26c78dbdc68b5c57f5b0749760dcade82`; verifier receipt SHA-256
  `c874891cbc754943eb4ca2d1f5f37b9f7a994200cd87c563a81133edd5867051`.
- Cleanup implementation r2 and verifier FAIL:
  `fb0e1a03ec18bab699893d2ba6d79b1a88ee765c` and
  `5af3dd88d3e312cfda9e771ad67f93a4f89e26e0`; verifier receipt SHA-256
  `8db725310ba9dcfabc62904891ca3b54affef6830ad6d8382270cc6e04103f20`.
- Historical trn21 G0 evidence commit
  `37aa3644e63e20a1427cfa69b44fd6de7e25cd81`; corrected replay verifier
  `8add2585d30b11ac2c50fb2701957657a1a6f30d`.
- Root contract, `README.md`, `research/experiment-protocol.md`, standing authorization/mission,
  current A1 proposal/runbook, and the two cleanup proposals/FAIL receipts.

The following trn22 facts were supplied as a read-only observation for this decision; this Analyst
did not access either host:

```text
free before fresh staging              8,010,760,192
free after representative staging      7,818,592,256
staging debit                             192,167,936
margin over 6.5-GiB floor                 839,270,400
processes / holders / workflow lock      none / none / free
```

They must be freshly re-proved by an Operator before any authority or execution decision.

## Option A — continue cleanup r3 on trn21

Potential advantage: if cleanup succeeded with sufficient capacity, the authenticated trn21 G0
could remain the same-host observation-off arm, leaving only G1 and G2 native gates. The historical
G0 took `1,066.4868` seconds, so avoiding a new G0 saves roughly 18 minutes of device time.

The critical-path risk is much larger than one normal repair cycle. Independent cleanup review has
already failed twice:

- r1 could delete an unauthorized path replacement, delete post-inventory content drift, and lose
  durable evidence after partial deletion;
- r2 still deleted a replacement at the final unlink boundary, accepted hardlink/same-inode
  mutation, misreported an unlink that took effect before an exception, and could seal unparsable
  partial journal bytes.

The remaining blocker is at the kernel name-operation boundary: POSIX unlink/rmdir acts on a name,
not an already authenticated file descriptor. Proving that a same-UID adversary cannot replace or
mutate the name/inode at that exact syscall boundary is difficult and may require a different
mechanism or a consciously redefined threat contract. It must not be papered over by moving test
hooks, trusting a prior stat/hash, weakening the verifier, or calling a partial delete non-destructive.

Even a successful cleanup package is not by itself an end-to-end disk solution. Its frozen target
is `7,286,616,064` post-cleanup bytes. The prior G0 runtime moved free space from
`9,630,490,624` to `7,487,733,760`, a retained debit of `2,142,756,864` bytes including a
`739,322,371`-byte checkpoint. A G1 run beginning near the cleanup target could approach or cross
the hard floor and would leave G2 blocked unless its own artifacts/caches were safely archived and
removed or substantially more space were recovered. That lifecycle has not been authorized or
proved.

Option A is therefore shorter only under optimistic assumptions: cleanup r3 passes immediately,
the exact target recovery is achieved, G1 stays within safe capacity, and a second cleanup is not
needed before G2. Current evidence does not support those assumptions.

## Option B — migrate the complete equivalence gate to trn22

Advantages:

- it avoids making the currently failed destructive cleanup implementation the immediate critical
  dependency;
- the supplied inventory is idle/unheld and a fresh staging leaves `839,270,400` bytes above the
  frozen 6.5-GiB operational floor;
- the work is a bounded gate-host/identity migration. Candidate, optimizer, model, data, flags,
  update count and exact numerical criteria remain unchanged;
- three serial cold five-step gates are operationally predictable. Using historical G0 duration as
  a rough observation gives about `3 x 1,066.49 = 3,199.46` seconds (53.3 minutes) of device time,
  excluding staging, archive, review and possible compiler variance. This is an estimate, not an
  ETA or authority.

Required scientific discipline: the old trn21 G0 cannot be used as the causal control for trn22.
Host effects matter, and host is part of the observer/report identity. Trn22 must run a fresh G0,
then G1, then G2, in separate processes with separate initially absent paths and cold/unique caches.
Only same-host trn22 reports may establish observation non-perturbation and hybrid-versus-ordinary
equivalence. The trn21 G0 remains historical regression evidence only.

### Storage blocker that must not be hidden

The supplied trn22 capacity is sufficient to stage and start one arm, not proof that all three arms
fit while retained. The arithmetic is already tight:

```text
post-staging free                         7,818,592,256
minus one known checkpoint                  739,322,371
= checkpoint-only lower projection        7,079,269,885
margin over next-arm floor                   99,948,029
next representative staging debit           192,167,936
```

Thus, even ignoring compiler/cache growth, retaining one checkpoint and staging the next arm would
fall below the floor. The historical G0 preflight-to-postflight debit was `2,142,756,864`; applying
that observation to supplied trn22 post-staging space projects `5,675,835,392`, which is
`1,303,486,464` below the operational floor and only `307,126,272` above the hard floor. Cache and
host state may differ, but favorable assumptions are forbidden.

Therefore Option B is the safer/faster **primary preparation lane**, but it is not READY_TO_RUN.
Before G0 authorization, its proposal must choose and independently verify one of these storage
lifecycle solutions:

1. prove by fresh conservative inventory that enough capacity exists to retain all three complete
   arm archives, checkpoints and unique caches through final adjudication; or
2. predeclare separate between-arm archival and cleanup/prune gates for exact gate-created artifacts,
   with explicit authority, complete retrieval/hash proof, independently reviewed deletion
   semantics and a fresh disk check before each next arm.

The current observations do not satisfy option 1, and option 2 reintroduces cleanup work. However,
deleting a newly created, fully archived single checkpoint or arm-specific cache can be scoped more
narrowly than recursively deleting multiple historical worktrees, provided the authority and
destructive primitive independently pass. No current receipt grants it.

## Exact trn22 package scope

Recommended fresh identities:

```text
G0 fs-20260925-v2-a1-native5-g0-hybrid-observation-off-trn22-r1
G1 fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn22-r1
G2 fs-20260925-v2-a1-native5-g2-ordinary-observation-on-trn22-r1
host trn22
expected hostname ip-172-31-32-253
```

Each arm requires distinct trn22 bundle, checkout, compile-cache, output, report, child-log, job,
result, dispatch and local request paths derived from its exact label. Reuse of any trn21 or consumed
r1/r2/r3 G1 identity is forbidden.

The host migration amendment must enumerate only gate-infrastructure identity changes:

- observer host allowlist `trn21 -> trn22` and all three exact trn22 arm labels;
- proposal, runbook, request/finalization/adjudication host, hostname, label, schema and path bindings;
- ownership-manifest observer pin and comparator manifest pin mechanically rebound;
- trn22-specific evidence manifests and same-host comparator identity.

It must keep these scientific bytes/settings fixed:

- production candidate SHA-256
  `9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae`;
- `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`;
- hybrid/ordinary gate sources
  `908431d7e1da8a9110e6e6005de5ecbe3dba13d798bf2f37f2443404dd8d7d94` and
  `6ed69d2356c28e0dd8f88f494d988c0b87f7f86426a001945bfd958ed6f8f6d0`;
- checkpoint verifier
  `e1ead481f9786ec02a3c60d88e72820f084274200bccd1f7eeb16da108cb6e2e`;
- seed 42, world size 4, LNC2, exact five updates, optimizer/model/data arithmetic, complete argv/
  environment, exact-byte acceptance criteria, checkpoint/causality requirements, operational and
  hard disk floors, and every downstream lock.

No numeric tolerance, record cardinality, equality requirement, disk threshold, or G0/G1/G2 order
may be relaxed. Any observer/manifest/comparator hash change must be disclosed and independently
proved to be host/identity plumbing only.

## Required Option B gates and authority boundaries

1. **Supervisor selection, no authority.** Record trn22 migration as the selected preparation lane,
   r3 as consumed, cleanup r2 as FAIL, and all execution/cleanup/G2/full-A1/submission booleans false.
2. **Storage-lifecycle design before implementation finalization.** Freeze conservative per-arm
   capacity accounting, retention requirements and any between-arm cleanup targets. No G0 request
   may be authorized based only on the supplied first-stage margin.
3. **Implementer package.** Make only the enumerated host/identity/pin changes in a fresh branch.
   Do not edit the production candidate, prepare/evaluator/tokenizer/scoring, optimizer arithmetic or
   thresholds.
4. **Independent package verification.** Review exact diffs/hashes; construct the real observer for
   all ranks/arms with exact request argv; accept only exact trn22 host/labels and reject trn21 and
   consumed labels; verify production forwarding, ownership plan, G0 local replay mechanics,
   comparator logic, checkpoint/causality gates, all tests and path freshness.
5. **Per-arm state-only authority.** G0 alone first. A separate Verifier must approve its exact
   authorization; then one O_EXCL request is materialized and independently authenticated. G1 and
   G2 remain false.
6. **Operator G0.** Freshly prove host identity, idle/lock/holders, exact source/head, paths absent,
   and pre-/post-staging floors. Execute exactly once, retrieve/hash/archive complete evidence and
   postflight. No automatic next-arm launch.
7. **G0 evidence and storage gate.** Independent evidence verification must establish valid same-host
   G0 and exact retained disk usage. If free space is insufficient for G1, execute only a separately
   verified archival/cleanup authority or STOP. Do not silently prune.
8. **Repeat distinct authority chains for G1 then G2.** Each depends on authenticated prior-arm
   evidence, clean postflight, sufficient disk after staging, unique identities and its own request
   verifier. A failure never advances order.
9. **Final adjudication.** Authenticate all trn22 reports/checkpoints/lifecycle evidence and run the
   exact comparator. A PASS is evidence for the defined gate only; full A1 training and submission
   remain separate decisions.

## Parallel preparation

Preparation can proceed in parallel without authorizing actions:

- primary lane: draft the trn22 migration/storage proposal and independently review its static
  package in isolated worktrees;
- fallback lane: research whether cleanup r3 has a genuinely sound destructive primitive and extend
  temporary-filesystem tests for every r2 blocker.

Parallel work must remain documentation, implementation and verification only. Do not create an
executable authorization, materialize a request, inventory a host, delete data or reserve both hosts.
Only one selected lane should cross into state-only authorization after its blockers pass. Role
separation and independent review remain mandatory; one agent must not propose, implement, verify
and operate the same action.

## Decision

Choose **Option B as the primary preparation path**. Its additional cold G0 costs roughly one known
gate duration, while cleanup A has already failed two independent adversarial reviews at a difficult
destructive syscall boundary. B better protects existing evidence and removes cleanup correctness
from the immediate gate-science path.

This recommendation is conditional, not a run approval: trn22 currently proves capacity for one
staged arm only. The trn22 proposal must solve and independently verify the complete three-arm
storage lifecycle before G0 authorization. Keep cleanup r3 as a non-authorized fallback research
lane. No present evidence authorizes cleanup, host migration, G0, G1 or G2 execution.
