# c5c/LR0.015 controls: ledger reconciliation

Two independently accepted controls are appended to `research/v2/ledger.jsonl`.
The earlier failed A0 and S0 rows remain byte-for-byte unchanged. Both new rows
use exact source SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
seed 42, LNC2/four ranks and fresh-process public-20M/context-1024 evaluation.

| Host | Control label | Steps | Public-20M | Used for |
| --- | --- | ---: | ---: | --- |
| trn21 | `fs-20260925-v2-c5c-a0-control-trn21-r1` | 3675 | 0.9914739939934344 | A1 and later c5c arms on trn21 |
| trn22 | `fs-20260925-v2-c5c-a0-control-trn22-r10` | 3633 | 0.991513958330512 | A2 and later c5c arms on trn22 |

Use each candidate's same-host control. The between-host score difference is
0.000039964337077647, descriptive only; it is not a noise estimate from
replicates or a reason to pool the hosts. These records establish baselines,
not a new adopted win or an official score.

## Evidence semantics and limitations

The row `git_commit` identifies `8b9fc841ed817eac263c7087c3bce7c9f008bda1`,
whose committed `submissions/research_v2/train.py` independently hashes to c5c.
It is not a claim about the harness checkout HEAD or the non-Git run directory.
`git_dirty=false` applies to those exact committed science bytes copied into
the isolated run directory. Both rows state that scope explicitly.

For trn22, the separate Sep 24 harness checkout was **not globally clean**:
six pre-existing, protected untracked run logs remained; tracked files were
unchanged. `preflight.json` records `tracked_source_clean=true` and
`harness_checkout_untracked_historical_logs=true`. The ledger retains that
distinction in structured fields and notes. The source came from the clean
e65 checkout, and the harness copied the verified candidate and evaluator to
fresh train/eval sandboxes. The independently accepted review covers this run.

trn22's postflight is an Operator attestation in `final/FINAL_RESULT.md`,
not a separate raw transcript, so `quiescence_receipt_sha256` is null and the
attestation is separately linked/hashed. trn21 has a separately archived raw
postflight. No explicit causality-test result, exact epoch boundary, or median
step time is inferred; those fields remain null. Checkpoint hashes/status are
from the preserved host/evaluator receipts, not a new live host check.

Native rounded training/startup times are separate from trn22's externally
observed startup/charged-through-exit timings. Existing research cache reuse
is recorded without claiming measured cache hits or verified effective
persistent-cache identity.

## Validation performed

- Append-only prefix preserved, four unique labels, two new valid controls.
- Every linked receipt/log/review hash in the new rows recomputed successfully.
- Both scores independently recomputed from their raw nats/bytes using ln(2).
- `decide.py` accepts one valid c5c control on each host; older failures remain
  invalid. Its SHA-256 remains
  `a47bedd258697b0791f48dcac3098d9071b5aaf00a7664d9e39987b4052adc66`.
- `decide.py --drift`: no alarms. Only one valid control per host exists, so
  this cannot establish within-host drift stability.
- `decide.py --arm A1` and `--arm A2`: `INSUFFICIENT`, because no evaluated
  candidate arm is in this ledger yet. No adoption was made.
- `git diff --check` passed. No host contact or model/evaluator changes.
