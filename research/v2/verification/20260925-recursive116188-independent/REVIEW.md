# Independent clean Recursive port review

2026-09-25. **PASS_SOURCE_AND_AUDITED_REDUCED_CPU_EVIDENCE_ONLY** for commit
`abe15ac29cc5ec277f9fe1c96012fd8cb5eaa04a`, candidate
`workflow/candidates/recursive-116188/train.py`, SHA-256
`4083da774ca0f8e26b69309203b07daceca1b559a1418eacc8371f2ff5021409`.

No concrete port defect found. This is a faithful narrow port of the already
adopted schedule into the preserved submission vehicle, not a new scientific
adoption, native result, cold proof, or submission approval. Exact-size CPU G2,
native trained-forward/causality/finite/budget verification and genuine cold
validation remain pending. No additional experiment or acceptance threshold is
introduced by this review. No candidate, host, submission package, Downloads,
data or evaluator was modified.

## Exact source and arithmetic review

The complete diff is **32 additions/20 deletions** against preserved116188
(`116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`).
Reference6ef7 is pinned to `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`, SHA-256
`6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`, matching the
durable two-host research adoption. The reviewed changes are only:

- Baked Recursive-ON helper bodies and group assignment: Muon/AdamW warmdowns
  .95/.65, final LR fraction .05, momentum warmup .85→.95 then quadratic
  cooldown to .79, and beta2 .95→.97. All helper arguments and operation ordering
  are AST-identical to the adopted ON branches; the floating boundary expressions
  are preserved rather than simplified to decimal thresholds.
- The compiled Muon call receives the existing `one_b2_t` device scalar, as
  reference6ef7 does when Recursive is ON. Its persistent scalar update already
  writes `1-float(group['beta2'])` every step. No change to `_muon_math`, state
  layout, parameter ownership, communication, or AdamW math was introduced.
- The existing fifth startup boundary clears `recent_step_times`, and the
  next-step maximum uses `default=0.0`. Timer statements match the previously
  reviewed prospective116188 reset. Startup cap900, excluded boundaries5,
  margin30, budget1800, checkpoint reserve15, synchronization, actual charged
  excess and weight-decay progress remain unchanged. This removes predictor
  contamination; it does not forgive startup or provide another allowance.
- Four launch specializations: width1024, context1024, pack32, in-training public
  eval OFF. All other source bodies/defaults remain116188. No FP8, U-Net,
  tables, batch change or new kernel was imported. Attribution remains present.

The schedule is intentionally baked ON. An OFF-toggle equivalence gate would
not describe this candidate; identity to adopted ON plus unaffected-code review
is the relevant port check.

## Complete bare-launch profile and evaluation semantics

Independent audit derives the proven argv from the preserved package's
`launch-command.txt` and cross-checks its `provenance.json`, then executes only
the parser/config AST under an empty environment. **All17 parsed arguments**
match that proven explicit invocation. The full mapping is durable in
`evidence-audit.json`, not merely an assertion about four edited constants.

The resulting model config is depth6, width1024, heads8/head-dim128,
context1024, vocab8192, MLP ratio4, dense KV heads. Other effective values include
batch131072, pack32, seed42, step cap1000000, budget1800, matrixLR.015,
compile/SDPA-direct/NKI-local-conv/BF16-norm/clip-after-reduce ON, ngram/QK-shift
OFF, and output `out`. At world4 this is131072 tokens per micro-step and one
accumulation. LNC2/world4 remain required launch environment/topology, not
something the bare Python file independently provisions.

`eval_tokens=2097152` remains an inactive legacy default because
`eval_public=false`; it does not replace the separately specified fresh
public20971520/context1024 evaluation. Existing environment overrides still
exist. The profile equality is under a clean/checked environment, so the
Operator must retain the packet's rejection of conflicting inherited overrides.
No environment override mechanism was added.

All model classes, forward arithmetic, initialization, `build_config`, loss/data
paths, save schema and `load_for_eval` are unchanged from116188. The loader
continues `torch.load(..., weights_only=True)`, constructs from saved config,
extends only non-persistent evaluation context buffers, uses strict state load,
then eval mode and `requires_grad_(False)`. No scorer-dependent update or
alternate schedule-dependent inference was added. Training-only schedules need
no new saved architecture toggle in this baked model.

## Independent tests and raw receipt audit

Executed in the isolated review worktree:

```sh
python3 -B workflow/candidates/recursive-116188/test_port.py
python3 -O -B workflow/candidates/recursive-116188/test_port.py
sha256sum -c workflow/candidates/recursive-116188/SHA256SUMS
python3 -B research/v2/verification/20260925-recursive116188-independent/audit_evidence.py
```

Both standard-library test runs pass **16 tests**, exit0 (normal1.175s,
optimized1.082s). These execute actual schedule/group/timer statements,
including >1000 progress points and adjacent floating-point boundary values,
warmup cases, all optimizer group kinds and the previously recorded five-step
startup failure trace. Source/model/helper AST comparisons and complete
bare-parser equality pass. All14 packet manifest hashes pass.

I independently parsed both successful raw reduced G2 receipts rather than
relying on `tiny-lockstep.json`, and reran the unchanged harness comparator:

- Exact source hashes6ef7 and4083; four distinct ranks, three finite losses each.
- All **12 loss values** match exactly, as do **all62 parameter digests** and
  ownership plan `806be765aee5bf7597ada264f58f60162a46dc5b9b7d098dc13a9e40c2bbfa57`.
- Every rank records all62 parameters updated, no cross-rank mismatches, no
  unchanged parameters/tables; rank0 records checkpoint existence, strict-load
  success and deterministic `[1,16,8192]` logits.
- Research argv is exactly the adoption argv with width changed to128. Port
  argv is only `--n-embd 128`; the harness appends CPU/no-compile/3steps/private
  output. This audits the recorded runs; I did not claim a new G2 execution.

Both initial `tiny-g2-*.json` FAIL receipts are retained. All four ranks in each
failed at import with `ModuleNotFoundError: No module named 'requests'`, before
any recorded loss or training step. These are environment/dependency diagnostics,
not model failures and not passes. The separately labelled r2 runs explicitly
use stub-prepare and fail-closed PyArrow support; no install or production change
was hidden. Such stubs must not enter a native/cold assignment.

The independent audit initially combined constant assignments and extracted
`global` declarations in one synthetic Python module, causing a verifier-helper
SyntaxError. Executing those two AST pieces separately corrected the helper;
no candidate or supplied evidence was changed. Its final exit0 result is
`evidence-audit.json`.

## Limits and handoff

The recorded numerical runs are width128/depth6, synthetic1x16 inputs, four CPU
ranks, BF16/fake-clock shims and only three updates. They neither exercise the
full native microbatch nor prove exact-size memory, native BF16/NKI behavior,
full-size causality, cold routing, runtime, budget or final BPB. This review
allocated no tensors and contacted no host. Stdlib arithmetic/timer tests are
not cold/native measurements; original research BPB is not this source's score.

The proposal is truthfully pending its exact-source host binding and existing
verification/cold workflow. It retains bare argv, fixed1800, trusted prepare,
fresh20M evaluation, measured startup versus capped allowance distinctions and
human-only submission. Prior empty-cache-directory evidence remains inconclusive.
No repeat scientific adoption, new threshold, launcher or arbitrary additional
gate is required by this source review.
