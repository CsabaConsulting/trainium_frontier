# Independent analyst verdict: Track-A 35c9 + matrix-LR 0.015 controls

Date: 2026-09-24  
Role: Analyst (read-only evidence adjudication; no training, SSH, source edit, push, or submission)

## Verdict

**VALID AS TWO SEPARATE HOST-LOCAL 2M SCREENING BASELINES. DO NOT POOL THEM AS A CAUSAL CONTROL.**

Both runs satisfy the Track-A control contract: the exact same source, launch,
seed, evaluator population, and finalized protocol were used; preflight passed;
training returned zero inside budget; the process groups were quiescent; finite
checkpoints and complete sealed artifacts were retrieved; and postflight was
clean. The observed comparison is nevertheless one run on each of two hosts.
Host and ordinary cold-run effects are therefore confounded and cannot be
decomposed from these two observations.

The trn22 score is `0.0006521422933748` BPB higher (worse) than trn21. That is
`65.21422933748%` of the frozen `0.001` 2M nomination margin. This makes
same-host pairing material. The arithmetic mean, `0.9870234445993500`, is
reported only as a descriptive summary and must not replace either host's
control.

## Authenticated evidence

Common identity:

- Finalized Track-A protocol commit: `65155c668efaa215a2136f67617b44ed32d4c9f6`.
- Source commit/path: `c51f6fdc5ca77793a1524148a563cafac0ab22b0`,
  `submissions/dense6x1024_public0992851/train.py`.
- Source SHA-256: `35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2`.
- `prepare.py` SHA-256: `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Seed `42`, world size `4`, LNC `2`, maximum charged training time `1800 s`.
- Exact fresh-process screen: sequence length `1024`, `2,097,152` tokens,
  `8,418,678` bytes, evaluator timeout `1200 s`.

| Field | trn21 control | trn22 control |
|---|---:|---:|
| Evidence commit | `b0b23cbc2e6d16e8bc8d88be9cb042140494206d` | `499f753ca554c5c732c35ab4eab7e96488d5e355` |
| Label | `fs-20260924-v2-35c9-lr015-control-trn21-r1` | `fs-20260924-v2-35c9-lr015-control-trn22-r1` |
| Hostname | `ip-172-31-34-85` | `ip-172-31-32-253` |
| Steps | `3636` | `3634` |
| Logged training seconds | `1784.9` | `1784.8` |
| Exact charged-through-exit seconds | `1790.407160185976` | `1790.5283933380852` |
| Screen total nats | `5,757,757.0` | `5,761,562.5` |
| 2M BPB | `0.9866973734526626` | `0.9873495157460374` |
| Eval seconds | `99.05535452801269` | `100.98319244803861` |
| Checkpoint SHA-256 | `aaa4b7c0ff58c56df564954c21f8abdd786930bfaa1cc447f6fb8e5b0da83535` | `db92ca7bc7fd428aab1f5f53d2868d791cd8271d707087f3f764459f02974c0f` |
| Artifact-manifest SHA-256 | `d067f57d83301c27f35a4aea63d508eb230dc0ef9d656839d71e08b2c48af625` | `684ec566cf0efc612f962a92d4492d37c529a2988fb4fc03afa68356ffe14644` |
| Archive SHA-256 | `f981f233f39ff82ee33059ec9157c969fd9aed6489b40c1cff6a11afe3ed35e6` | `a93faed2ecbba89b018f2327ef06eab47b5bcfa14e125c909c61f67652ab00fc` |
| Ready receipt SHA-256 | `41a27cd7c7a0747a302c0c5e034f9fe6a302d0650efbe91624291ef284d3aacb` | `a3cdb39726a9154f4586b3041a1bb640d4d2f7f93cd20cac68ba8b9ad70bfc6a` |

Independent local verification recomputed the byte count and SHA-256 of every
manifest member, the manifest, archive, and ready receipt. Both archives contain
the same required 11 payload member names, and all checks matched the sealed
receipts. Each checkpoint is `469,834,849` bytes, finite under
`torch.load(..., map_location="cpu", weights_only=True)`, and contains 62
tensors. Both preflights report `passed: true` with no failures; both postflights
report return code zero, no failures, and `host_clean: true`. The committed
operator audits are PASS.

Primary evidence locations (relative to repository root):

- trn21: `research/submission-validation/20260924-v2-track-a-35c9-controls-a1/operator-evidence/fs-20260924-v2-35c9-lr015-control-trn21-r1/`
  at commit `b0b23cbc2e6d16e8bc8d88be9cb042140494206d`.
- trn22: `research/submission-validation/20260924-v2-track-a-35c9-controls-a1/operator-evidence/fs-20260924-v2-35c9-lr015-control-trn22-r1/`
  at commit `499f753ca554c5c732c35ab4eab7e96488d5e355`.

## Exact cross-host comparison

Using Decimal arithmetic on the committed values:

- BPB delta, trn22 minus trn21: `+0.0006521422933748` (trn21 is lower).
- Descriptive mean: `0.9870234445993500`; half-spread:
  `0.0003260711466874`.
- Total-nats delta on the identical evaluator population: `+3,805.5`.
- Step delta, trn22 minus trn21: `-2` steps (`-0.05500550055%` versus trn21),
  equal to `-262,144` training tokens at 131,072 tokens/step.
- From logged training seconds: trn21 `2.03708891254412` steps/s versus
  trn22 `2.03608247422680` steps/s, a trn22 delta of
  `-0.00100643831732` steps/s (`-0.04940571377%`).
- From exact charged-through-exit seconds: trn21 `2.03082297750771`
  steps/s versus trn22 `2.02956848577259` steps/s (`-0.06177257934%`).

The throughput and step-count differences are operationally small. They do not
explain away the BPB observation, but neither do these two runs establish a
stable host effect: there is no within-host repeat with which to separate host
from cold-run variation. Process-wall differences are dominated by different
startup/compile allowances and are not a valid cross-host throughput claim.

## A1/A2 screening rule

Keep the predeclared gate unchanged and apply it to the exact same-host control:

- An arm screened on **trn21** nominates only if
  `0.9866973734526626 - candidate_BPB >= 0.001`, equivalently
  `candidate_BPB <= 0.9856973734526626`.
- An arm screened on **trn22** nominates only if
  `0.9873495157460374 - candidate_BPB >= 0.001`, equivalently
  `candidate_BPB <= 0.9863495157460374`.

The finalized proposal explicitly binds A1 to the trn21 control. If A2 is
assigned to trn22, it must use the trn22 control; if it runs on trn21, it must
use the trn21 control. Host assignment must be frozen before reading the arm's
score. Using the descriptive mean would move the cutoff by
`0.0003260711466874`: it would be too loose for trn21 and too strict for trn22.

An A1/A2 2M result can only **NOMINATE** or **DISCARD**. It cannot KEEP/adopt a
candidate. It also does not support a direct cross-host ranking of A1 versus A2.
Downstream adoption remains subject to the proposal's fresh full-20M matched
trn21 and trn22 deltas: both strictly negative, with their Decimal arithmetic
mean at most `-0.001`. The present controls are valid 2M screening baselines,
not full-20M causal controls.

## Residual limitation

The fixed 2M evaluator population makes the two checkpoint evaluations exactly
comparable as measurements of those checkpoints; it does not remove training
run variability or the representativeness limitation of a 2M screen. With one
control per host, no numerical host correction or run-to-run noise band is
identified. The scientifically conservative action is therefore the frozen
same-host delta gate followed by the already-required matched full-20M
confirmation, not a pooled baseline or a post-hoc threshold adjustment.
