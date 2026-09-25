# Independent L3 Recursive schedule review: source and tiny CPU PASS

Recorded 2026-09-25 15:58:17 UTC by
`/root/recursive_schedule_readonly_verifier`, independently of the Implementer.

**PASS:** diff review, inspected tiny four-rank CPU numeric evidence and
three-step default-OFF identity. **Exact-size production-argv CPU G2 remains
PENDING.** This is not Neuron/full-run/quality/submission approval.

The machine-readable receipt beside this file binds source SHA
`f9b6a9cbb1a1760fb8db422ffd79ffb91e9280899c40aa86028d935acf355f67`
at `workflow/proposals/research-v2/train.py`, against exact c5c SHA
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
Implementation `e273b97c`, handoff clarification `6a2ae2c1`, and conditional
profiles were inspected on controller `70e53553d92dc04bcdf2a7747c6e4efe4833e209`.
The latest 21:11 IST correction controls profile selection; no table winner
is assumed by this receipt.

## Findings

- No blocking source defect found. Muon/Adam warmdowns 0.95/0.65 and floor
  0.05 use existing charged elapsed/budget semantics (`train.py:3248`, `4011`).
- Source-matching Muon momentum and NorMuon beta2 reach real optimizer
  groups (`3331`, `3384`, `4017`). Dynamic beta2 uses the existing fifth
  tensor scalar (`2915`); a changing Python constant is not injected into
  the compiled body.
- Flat-table LR precedence remains intact (`2196`, `4018`). No WD pulses,
  table-beta2 ramp, hidden embedding-LR increase or forward/eval change.
- Default-OFF computation is unchanged. Checkpoint metadata gains an
  explicit false toggle; serialized checkpoint-byte identity is not claimed.
- Direct comparison of existing c5c and OFF receipts confirms all four
  ranks' three losses, rank-0's 74 parameter digests and ownership plans
  match. Every rank separately reports no cross-rank parameter discrepancy.
- Tiny ON receipt has four ranks, three finite steps, 74/74 changed
  parameters, all four tables changing, deterministic finite checkpoint
  logits, and worst normalized RMSProp error `0.025894049 < 0.03`.
- Packaging whitespace is fixed; `git show --format= --check e273b97c`
  exits successfully. The reviewed candidate hash did not change.

## Evidence limits and minimum next step

Tiny tests use width 128, table multiplier 1, dimension 16 and micro-batch
1x128 under CPU Torch 2.6, with documented synthetic/Gloo/PyArrow shims.
They are not exact-size or Neuron tests. Their fake clock does not reach
cooldown; the separate inspected schedule test covers boundary formulas and
the actual group-assignment block. No BPB or new causality claim follows.

The JSON receipt may satisfy runner-registry `diff_review` and
`flag_off_identity`. **Do not register it as production `cpu_g2`.** Obtain
the outstanding exact-source, exact-final-argv CPU G2 for the selected
profile; then normal Operator preflight and a full research run are the
next useful steps. No new review-of-review or short native-smoke requirement
is imposed.

Inherited `sched_tricks` logging still prints the old 0.9 beta2 constant;
the active L3 schedule starts at 0.95. This is a nonblocking observability
note, not an incorrect update formula.

No host contacted, no new training test run, no candidate/source/evaluator
file changed by this reviewer. Only these review artifacts were written.
