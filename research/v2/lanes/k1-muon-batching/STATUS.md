# K1 Muon batching — source audit, 26 September 2026

## CLOSED by Final Sprint, 26 September12:17 IST

Literal same-shape Muon batching and rank ownership already exist in audited
e1e2. Do not allocate another implementation/review/native slot to this lane.
This closure is the human's priority decision, NOT proof that all optimizer
cost is below5%. Preserve the measured profile and all earlier findings below.
Any later reopening needs a distinct measured bottleneck and current policy.

## 02:12 IST evidence update

The completed P trace is now correlated against the exact OwnedOptimizer tail.
Full parameter writeback spans6.8–8.2ms across ranks; these are overlapping
runtime intervals, not exclusive instruction time or a causal speedup bound.
Omitting only the owner's approximately quarter-copy lacks evidence for the
required5%whole-step gain, so that residual proposal is not selected for a
native slot. Literal batching remains already present. This does not close
other K1 possibilities. Analysis and reproducible SHA/AST checks are at
`research/v2/analysis/20260926-k1-writeback-runtime/`.

There is no active K1 model Implementer at this moment: the local Implementer
is preparing the base-cache cold binding. Earlier ownership/discovery notes
below are historical, not evidence of a second racing agent. Installed device
decoder capability is identified; device-instruction cost attribution pending.

Done: exact adopted6ef7 already performs the requested same-shape batched
Newton–Schulz and four-rank owner-compute plus one all-gather. No production
patch or new default-OFF no-op flag was made. This is a duplicate-mechanism
finding, not a newly measured speedup or a conclusion that all optimizer
improvements are exhausted.

Actual lane ownership: one local GPT-6 Astra Implementer
`/root/fp8_local_implementer`. No second model-family builder was launched by
this agent; Supervisor is separately discovering the Bedrock route. No native
host access or tensor workload in this audit. Independent review is pending.

Blocked: no missing implementation of the literal requested batching/sharding
mechanism was found. Reimplementing it would not be a new factor. Prior native
evidence also prohibits casually removing singleton padding: doing so changed
tall-matrix updates. No technical error or unavailable host is invented.

Next: use P's measured phase/rank profile to select a distinct remaining
optimizer cost. Shape-aware ownership pairing is a possible proposal (not
selected or implemented): avoid two dummy tall matrices without using native
singleton batches, while measuring critical-rank balance and exchange padding.
It changes the owner plan and needs its own default-OFF flag, CPU G2/OFF identity,
per-matrix/state BF16-tolerance checks, different-family review, and >=5%
whole-step timing before a full run. No current >=5% gain is predicted as fact.

See `AUDIT.md` and reproducible stdlib `audit_plan.py`/`plan.json`. Update status
again at the required two-hour boundary if this lane remains active; this file
does not create an autonomous host launcher or a scheduling mechanism.
