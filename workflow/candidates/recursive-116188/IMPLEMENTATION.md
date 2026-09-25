# Recursive schedule on the proven 116188 vehicle

Status: **IMPLEMENTER SELF-CHECK PASS; INDEPENDENT REVIEW / EXACT-SIZE G2 /
TRUE-COLD FULL-BUDGET REPLAY PENDING. NOT SUBMISSION ELIGIBLE.**

Parent authorization: port only the already-adopted Recursive schedule and
reviewed predictor reset, with the documented dense launch profile specialized
for bare invocation. This is a new isolated candidate, not a replacement for
any existing submission or Downloads artifact. No host was contacted.

Candidate: `workflow/candidates/recursive-116188/train.py`, SHA-256
`4083da774ca0f8e26b69309203b07daceca1b559a1418eacc8371f2ff5021409`.
Branch: `codex/20260925-recursive-116188-submission-port`, based on canonical
`5d907edad6105e763acb9636daaf70f6445946e5`.

## Exact scope

`vs-116188.patch` is the full source diff against preserved
`submissions/dense6x1024_direct_review_matrixlr015/train.py` (SHA
`116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`).
The adopted reference is commit `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`,
`workflow/proposals/research-v2/train.py`, SHA
`6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.

- The three schedule helper bodies are exactly the reference's Recursive-ON
  AST, including floating-point expression order: Muon/AdamW warmdowns .95/.65,
  final LR fraction .05, momentum .85→.95 warmup then quadratic cooldown to .79,
  and Muon beta2 .95→.97. Attribution is retained.
- The existing compiled Muon body's beta2 input uses its existing persistent
  tensor scalar. The optimizer math itself is unchanged; no new optimizer,
  graph family, ownership plan, weights, model, table or FP8 mechanism is added.
- The reviewed predictor reset clears the five excluded startup durations and
  uses `max(..., default=0.0)` afterwards. The actual timer-boundary AST matches
  the reviewed prospective-116188 patch. Startup cap900, excluded boundaries5,
  launch margin30, budget1800, checkpoint reserve15, synchronization, boundary
  reporting and charged elapsed arithmetic are unchanged. No timing conflict
  was found. Weight decay still follows the same charged-budget progress,
  including its pre-existing startup phase; no new timing phase was introduced.
- Bare-launch specialization changes only defaults `N_EMBD: 0→1024`,
  `SEQ_LEN: 2048→1024`, `PACK_FACTOR: 4→32`, and `eval-public: ON→OFF`.
  Under a clean environment, **every parsed argument**, not merely these four,
  matches the proven 116188 documented launch. Matrix LR remains .015;
  depth6, heads8, MLP ratio4, full-chip LNC2×4, seed42, step cap1000000 and
  budget1800 are unchanged. The explicit proven true/false kernel and dense
  flags already equal the source defaults. Existing environment overrides
  still exist and must be checked by the Operator; they are not permission
  to alter the approved profile.

No `--recursive-schedule` option or OFF branch is introduced: this is the
requested baked Porter candidate. An OFF-equivalence test is **inapplicable**;
the relevant test is identity to adopted ON, plus unchanged unaffected code.
Old unused schedule constants are retained from 116188 rather than bundling
unrelated cleanup. All model classes and other helpers, `load_for_eval`,
`weights_only=True`, checkpoint schema, input/data/scoring code and protected
`prepare.py` are unchanged. No runtime scorer/import-path shim is in train.py.

## Local evidence and limitations

`python3 -B workflow/candidates/recursive-116188/test_port.py` and the same
command with `-O` pass **16 tests** (1.225s / 1.096s recorded). Tests compare
exact schedule AST, >1000 progress points including adjacent floats at each
boundary, actual group assignments, all effective bare arguments, unchanged
model/helper AST, the optimizer's sole input-transport change, and ten inherited
or additional timer cases. The same A2 trace that previously stopped at five
continues without changing its charged startup excess. These are self-checks.

The tiny CPU G2 pair uses unchanged `workflow/tools/preflight_dryrun.py`, Python
`/home/poojith-devan/venv/bin/python`, Torch `2.13.0+cpu`, four ranks, three steps,
six layers at **width128**, synthetic **1×16** token micro-inputs, one thread
per OMP/MKL/OpenBLAS, and deterministic fake-clock/BF16 CPU shims recorded in
each receipt. Research argv is the adopted dense argv with width128; candidate
argv is only `--n-embd 128`. G2 appends CPU/no-compile/3steps/private-output.
Thus this is neither production-size nor exact bare-argv proof.

The first attempts (`tiny-g2-research-on.json`, `tiny-g2-port.json`) both failed
during import because the laptop runtime lacks `requests` required by real
prepare. No training step ran; these FAIL receipts are retained. Retrying as
new `-r2` receipts used the existing test-only `--stub-prepare` facility, plus
`test_support/pyarrow` (import-only, every real Parquet read fails closed), with
no package install. First attempts borrowed the identical-purpose L2 PyArrow
stub; r2 uses this packet's stub. **Never deploy either stub to native tests.**

`tiny-g2-research-on-r2.json` and `tiny-g2-port-r2.json` pass in 6.1s / 5.0s.
`tiny-lockstep.json` reports bit-identical per-rank three-step losses, all 62
state digests, and ownership plan
`806be765aee5bf7597ada264f58f60162a46dc5b9b7d098dc13a9e40c2bbfa57`.
Both receipts also record strict checkpoint reload/deterministic-logit checks
and cross-rank consistency. This does not prove native BF16/NKI numerics,
causality at full size, cold cache routing, runtime, budget or BPB.

## Native handoff

`proposal.json` is pending, **not launch authorization**. `OPERATOR_RECIPE.md`
reuses the existing G2/contained/verified-runner workflow; it does not create a
launcher or queue entry. First run independent exact-size CPU G2 on an idle
large-memory host, compare against exact6ef7 Recursive ON, then a separately
bound genuine-cold full1800/public20M replay. Record actual startup-to-fifth
boundary, capped allowance, source estimates and charged-through-exit time as
separate fields. Require independently reviewed cache-routing proof: the prior
116188 cold smoke's empty persistent URL directory did **not** prove cold.

The existing research adoption's two-host mean .9901674436550496 is evidence
for choosing the component, not a measured result for this new source. No
speedup/BPB win/submission claim is made. A second host cold replay and final
independent submission audit/human approval remain required; do not merge into
the submission branch or overwrite a prior package.
