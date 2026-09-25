# Independent FP8 timing-packet review: HOLD for proportional revision

Reviewed packet `17bc5748d892b58f2295bedf33a6b99cebcebf6c`, at
`research/v2/proposals/20260925-l1-fp8-whole-step/`, on 25 September 2026.
Reviewer `/root/fp8_independent_verifier` did not author the source or packet.
This is a packet scope/feasibility HOLD, not a new FP8 source defect or a failed
native experiment. No host contact, source edits, tensor tests or training occurred.

## Checks that pass

- Independently verified all six manifest file hashes. The exact candidate and
  OFF control are the same `921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858`
  bytes from source commit `50a901374ae6f76b35b8556d32fd72b1ece28fc6`.
  The c5c CPU comparison source at `submissions/research_v2/train.py` also hashes
  to the stated `c5c410af...`. Packet changes are only four new proposal files;
  `git diff --check` passes.
- Root's independent timer-composition review is completed at `e9109dec`, in
  `research/v2/verification/20260925-lane-timer-ports-and-gather-review.md`.
  Attach this receipt directly; the packet's statement that it is underway is
  stale. Original source/reduced CPU review `20ec99c2` remains correctly scoped.
- Native training argv uses legal full-chip LNC2/world4, dense6x1024, context1024,
  pack32, seed42 and no tables/schedule/architecture stack. The only causal flag
  difference is explicit FP8 OFF/ON. Both retain `--max-train-seconds 1800`.
  A 100-step diagnostic cap is explicit, not a claim of full-budget quality.
- The CPU recipe retains exact parameter dimensions and uses the known G2
  three-step/synthetic 1x16-token shim. Appended CPU/three-step arguments have
  the harness's normal precedence. Exact-size ON/OFF G2 and c5c OFF identity
  still need actual evidence; no local full-size run was attempted.
- Whole-step boundaries 19..99 correctly span 80 complete intervals after 20
  warmup steps. Optimizer, cache refresh, device synchronization, data fetching
  and between-boundary controls are included. The polling-conservative 3%
  step-time reduction formula is appropriate. No performance evidence exists.
- The current packet explicitly disclaims self-approval, full quality and
  submission authorization. Startup charging, cap900, margin30 and checkpoint
  reserve are preserved. Retain the **1800 charged-second training cap through
  save/exit for every native training run** in any revision.

## Required narrow revision

The newest human instruction is finish verification, then one matched whole-step
test at the first free slot, with >=3% before a full quality run. The Supervisor
clarified this review's scope to minimum sufficient gates. The current packet
instead mandates two 30-step debug runs, four 100-step timed processes and repeated
fresh native checkpoint/causality processes before a decision. Their compilation
cost is unmeasured and can consume the scarce free slot without answering the
requested first comparison. The debug flag also builds diagnostic work that is
excluded from the actual timing path.

Implementer should revise the proposal and runbook, without changing train.py:

1. Keep exact-source review and exact-size CPU ON/OFF/default-OFF identity.
   Keep one supplied native FP8 component correctness check, because legacy
   packed bytes and native forward/BF16 VJP have never been executed.
2. Make the initial matched block **one OFF 100-step run, then one ON 100-step
   run**, same host/source/environment, fixed 20 warmup plus all 80 measured
   intervals. Remove the two mandatory standalone debug30 runs. The timed runs
   themselves exercise full native compilation and training at the real shape.
3. Bind finite training losses, all four clean exits, quiescence, exact source,
   finite checkpoint state and charged-budget compliance to these same timed
   runs. Bind strict load and the frozen causal/batch-independent forward check
   to the ON timed checkpoint after the pair. Prior CPU causality/source review
   and native component VJP evidence remain part of the chain. Do not add debug
   tensor inspections inside the measured window or duplicate the training run
   solely to obtain a checkpoint. Native failures still fail the applicable gate.
4. The unchanged comparator accepts exactly four receipts, so a two-run packet
   must supply a small packet-local pair calculation using the identical
   `1-(ON_mean+ON_error)/(OFF_mean-OFF_error) >= 0.03` rule. Do not pretend the
   existing ABBA command accepts two inputs. Preserve raw intervals/distributions;
   add a matched repeat only for ambiguous polling uncertainty, observed drift,
   compile contamination or other concrete comparability uncertainty. Do not
   average a valid sub-3% first result into a promotion.
5. Remove the automatic telemetry-unavailable HOLD and arbitrary mandatory 1%
   same-arm drift guard, which requires the extra pair. Record available policy,
   contention/throttle observations and unavailable sensors candidly. Known
   differential contention/throttling can invalidate a comparison; unavailable
   optional sensors alone do not block this authorized diagnostic.
6. Record measured startup, capped allowance, charged excess and schedules;
   remove the newly invented `startup <=870s` eligibility gate for this warm
   research diagnostic. Never forgive charged excess or change the 1800 cap.
   Startup can affect schedule phase and must be disclosed: LR is initially flat,
   but default weight decay varies continuously with charged time, so flat LR
   alone does not prove matched phases. This test measures throughput, not model
   quality. Cold submission feasibility remains separate and is not an extra
   cold-replay requirement for this timing packet.

The current host source registry does not itself establish dispatch support for
921271. The Operator/Supervisor must use the existing checked route that can bind
these exact bytes (or the proven manual path) and preserve the common lock; no
new launcher or independent approval ceremony is requested by this review.

Outcome: **HOLD_PACKET_SCOPE_REVISION**. The revisions above are assigned to the
Implementer; this Verifier has not edited the packet. After a bounded revised
packet review, exact-size CPU/native evidence remains pending until actually run.
No whole-step speedup, quality gain or submission readiness is certified.
