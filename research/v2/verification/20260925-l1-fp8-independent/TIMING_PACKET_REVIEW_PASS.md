# Independent simplified FP8 timing-packet review: PASS

Verdict: **PASS_PACKET_STATIC_AND_SYNTHETIC_ONLY** for packet
`30177711824b6d82feb8e4706eb57e41784f0414`. This supersedes the packet-scope
HOLD in `0a8247e7737d185f901eca94cadc66f3cbf66335` for this revised packet only.
It does not certify native execution, exact-size G2 results, speedup, BPB or
submission readiness, and does not itself change any launch flag.

Reviewer: `/root/fp8_independent_verifier`, distinct from the Implementer.
Reviewed the complete diff from `17bc5748`, the complete new pair comparator
and tests, and the referenced source/CPU receipt chain on 25 September 2026.
No packet/source edits, hardware access, host writes, tensor allocations,
training, cleanup or external API calls occurred during this review.

## Scope HOLD resolved

- The initial native sequence is now one existing component check, one OFF
  100-step run, one ON 100-step run, then one frozen-forward/checkpoint check
  using the ON timed checkpoint. Each timed run uses the fixed 20 warmup plus
  all 80 measured steps. The two debug30 runs, mandatory ABBA and 1% same-arm
  drift requirement are removed.
- Finite logged training losses, finite checkpoint states, all four clean
  worker exits, quiescence and the budget gate attach to those same runs.
  The packet does not falsely claim every training gradient was inspected.
  Native component VJP checks and the ON checkpoint causal/batch-independent
  forward check supply their stated bounded evidence.
- Unavailable optional sensors and startup above 870 seconds no longer create
  automatic research-diagnostic holds. Actual contention, compiler activity,
  drift and charged-time schedule differences remain recorded and interpreted.
  No startup charge is forgiven. The **1800 charged-second cap through save/exit**,
  official startup cap900, margin30 and checkpoint reserve remain unchanged.
- A repeat requires concrete uncertainty and new labels. A clearly failing
  valid pair is not averaged into a later promotion. A first unambiguous accepted
  PASS requires no automatic second pair or extra cold replay.
- Root's completed source/timer review is correctly attached as
  `e9109dec72e41f0d481bd55df49aaedf7655bb6a`. The original `20ec99c2` reduced-CPU
  review is not misrepresented as an exact-size or native result. The source
  registry limitation is disclosed with the existing checked/manual route as
  the solution; no launcher or additional approval mechanism is added.

## Comparator review and independent checks

The new read-only `compare_pair.py` consumes exactly OFF/ON receipts. It pins
source SHA, receipt schema, 20/80 window, 100 contiguous raw boundaries, one
observed event per measured boundary, finite positive measured intervals, common
host/source path, distinct labels and OFF-before-ON boundary chronology. It
recomputes the mean and polling error from retained events/bound and checks the
supplied summaries. Every interval contributes to the reported distributions
and eight consecutive 10-step block means; no outlier is removed.

For means O/N and endpoint polling errors eO/eN, the bounds are:

```text
lower = 1 - (N + eN) / (O - eO)
upper = 1 - (N - eN) / (O + eO)
PASS         if lower >= 0.03
FAIL         if upper < 0.03
INCONCLUSIVE otherwise
```

The denominators are checked positive. Exactly 3% observed improvement with
nonzero polling error is correctly inconclusive. This is a 3% step-time
reduction, with polling bounds only, not a statistical confidence interval or
proof of repeatability. The result explicitly requires the separate actual
argv/environment/cache/boot/process-nonoverlap/budget/terminal/native evidence.
It cannot create a standalone performance or BPB acceptance claim.

Independently executed in the packet worktree:

```sh
python3 -m unittest discover -s research/v2/proposals/20260925-l1-fp8-whole-step -p 'test_compare_pair.py' -v
python3 -O -m unittest discover -s research/v2/proposals/20260925-l1-fp8-whole-step -p 'test_compare_pair.py' -v
git diff 17bc5748 30177711 --check
```

Results: **10/10 PASS** under normal Python (0.010s) and **10/10 PASS** under
optimized Python (0.006s), plus whitespace PASS. Tests cover pass/fail/borderline
polling, slow intervals, malformed identity/window, forged numeric summaries,
missing/duplicate/coalesced/reversed/nonfinite events and chronology.

An additional independent stdlib `Fraction` oracle checked 36 synthetic cases:
OFF means {0.5,1,2}s, true reductions {-10%,0%,2%,3%,4%,50%}, polling gaps
{0.005,0.1}s. Exact rational lower/upper formulas agreed with every comparator
classification; lower bounds agreed to absolute tolerance 1e-12. No model,
tensor or hardware library was imported for these checks.

Independently rehashed all **eight** manifest entries (six unchanged source/support
files plus both new packet tools): all match. Comparator SHA is
`eaf0c1fa53545e4eca40057e7c2e9b9b38d23ae3c527776da6f614df806cc6d0`;
test SHA is `0b2d5eb31d94d067654478a80b6feecf651313ee53ac641438d3de1e8f66a958`.
The exact training source remains
`921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858`.
Only the six packet files changed in this revision; production source, existing
observer/checkers, prepare.py and evaluator are unchanged. Packet worktree was
clean after the checks.

## Remaining evidence, without added ceremony

No new blocker was found. Exact-parameter-size CPU ON/OFF and default-OFF
identity remain pending until their retrieved evidence is accepted; source
staging alone is not a CPU or native PASS. The Operator still applies the
existing idle/disk/STOP/lock/exact-source/full-chip protocol. Component native
correctness, actual 100-step runs, ON frozen-forward checkpoint check and
retrieved result review remain unexecuted gates of this packet. A valid accepted
>=3% timing result can support the separate full-budget quality proposal
already described by the Supervisor; none is claimed or launched here.
