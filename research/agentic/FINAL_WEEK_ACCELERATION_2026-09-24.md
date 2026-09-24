# Final-week acceleration reconciliation

This is the Supervisor's reconciliation of the human-approved final-week plan with the repository
contract and measured evidence. It is an operating plan, not permission to submit or alter an EC2
instance.

## Immediate critical path

1. Preserve and independently verify canonical root commit `e12d0dee`, whose `train.py` SHA-256 is
   `116188cf…`. Keep the already confirmed file immutable as the insurance candidate.
2. Restore disk headroom first through Operator-owned, archived cleanup. The final-week operational
   gate is 6.5 GiB, retaining 1.5 GiB above the hard 5 GiB protocol floor; never lower the hard
   validator or delete an artifact whose proposal says pruning is unauthorized.
3. Establish one current-base 20M control on each host. Existing LR0.015 results are useful trn21
   evidence, but do not replace a trn22 control.
4. Run constant-only arms first: matrix LR 0.0125 and embedding LR 0.20 in parallel, followed by
   embedding LR 0.45, unembedding LR 0.004/0.009, the 1080-second epoch diagnostic, and warmdown
   0.85. Each arm needs its own proposal, implementation commit, independent receipt, unique label,
   and fresh preflight.
5. Use a 2M fresh-process public evaluation only for nomination. A nominated arm is re-evaluated or
   replicated with the exact 20M population before any KEEP/adoption verdict.

## Faster and safer than the supplied plan

- Do not wait for an autonomous Researcher before executing the approved constant queue. Researcher
  output fills the queue only after the measured arms and controls are staged.
- Do not bake ten launch flags into the only confirmed file before preserving it. A bare-launch
  variant is a separate implementation and must prove identical training configuration and clean
  fresh-process loading; it must not block research on the confirmed bytes.
- Reuse compile caches only for research throughput after a hash-bound cache experiment demonstrates
  identical source/compiler/environment identity. Maintain a cold-cache finalist replay because the
  scorer may compile cold and the observed startup is close to the 900-second allowance.
- Keep host-local controls. Cross-host BPB and throughput are diagnostic, not interchangeable.
- Stop a tuning family after three null arms. Combine changes only after each wins alone, then demand
  that the combined stack retains at least half the sum of isolated gains.

## Decision thresholds

- Screen delta at 2M: `<= -0.0005` nominates a 20M confirmation; `>= +0.0005` discards; the middle
  is discarded unless it is a predeclared adjacent sweep point.
- Adoption: two 20M runs both negative with mean delta `<= -0.0008`, plus current same-host controls.
- A sub-0.97 public result is presumed invalid until independently replayed and audited.

## Non-negotiable separation

The Supervisor selects; the Implementer changes one factor; the Verifier independently reviews;
the Operator alone launches and retrieves; the Analyst compares against the declared control.
Leaderboard upload remains a human action.
