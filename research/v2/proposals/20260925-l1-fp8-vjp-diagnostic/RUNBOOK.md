# First-shape FP8 VJP diagnosis — not a component PASS

The original source921271 / checker7ba967b0 failure in `c1ac162c` remains
authoritative. This proposal does not reopen timing or training. It adds only an
opt-in mode to that checker; `train.py`, the default component path, all four
original shapes, and its 1% RMS gate are unchanged.

One unique label: `fs-20260925-l1-921271-vjpdiag-r1`. Supervisor/Operator select
one next free host after its current work. No host was contacted by this builder.
Independent review and exact Operator binding are pending.

## Operator recipe

Use the existing detached manual component driver from `a26b7414` with a new
immutable binding/run directory and label. Do not reuse its consumed phase,
overwrite its files, or run its old hard-coded component-PASS receipt predicate.
Operator's mechanical binding must pin the new checker hash/review, exact source
and existing prepare hash, new command/label/receipt directory, 300-second wall
timeout (existing 60-second kill grace), and diagnostic-only receipt/terminal
names below. Retain its existing owned-between-job handoff, STOP, disk,
idle/device-holder, unique-label, exact-byte, exit and quiescence checks. This
packet supplies no launcher and makes no driver, queue or environment edits.

```sh
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 <staged-check_native.py> --train-py <exact921271-train.py> --out-dir <fresh-diagnostic-receipts> --diagnose-first-case --label fs-20260925-l1-921271-vjpdiag-r1
```

Record full absolute argv, checker/source/support hashes, SDK versions, boot,
compiler/cache environment and start/end wall time. Keep the previous checked
settings; no cold-cache, SDK, power or topology change. Four ranks are mandatory.
The checker checks the exact candidate hash before import. Reuse the established
source staging/prepare path; do not add production shims.

Expected files are `diagnostic-rank0.json` through `diagnostic-rank3.json`, each
with `verdict=DIAGNOSTIC_ONLY_NOT_COMPONENT_PASS`, the unique label, exact source
and checker hashes, and the matching rank/world4. A completed diagnostic may
have comparison failures; exit0 means observations were collected, not that the
component passed. Operator terminal status must be
`DIAGNOSTIC_COMPLETE_NOT_COMPONENT_PASS`, never the original component PASS.
Exceptions/timeout/missing receipts are `DIAGNOSTIC_INCOMPLETE`; retrieve raw
logs and terminal/postflight regardless. Do not automatically retry or launch
OFF/ON timing. No quality ledger row, optimizer steps or training-budget claim.

## Smallest retained contrasts

Only the failed seed42 `[256,256,256]` case is used, including zero, tiny and
outlier rows. No full model, data, optimizer, checkpoint or table is allocated.
There are two compiler entry points (original bound apply and enclosing Python
wrapper), one eager custom-function call, one replay of the existing bound
callable, and four BF16 GEMMs. Compilation may specialize more than once; the
wall timeout bounds unexpected behavior and does not make an incomplete run a
pass. There are no 1024-wide or full-component shape sweeps.

1. Original compiled bound-apply runs first, with the same direct preparation,
   CPU packed-byte comparison, forward, supplied gradient and backward as the
   failed path. No saved-tensor inspection occurs before this backward.
2. An enclosing compiled wrapper calls `.apply` inside its Python frame, as
   production `Linear.forward` does. Independent fresh leaf inputs prevent
   gradient accumulation from the preceding case.
3. Eager `.apply` inspects its actual saved context before backward.
4. The already compiled original callable is replayed with saved-context
   inspection before backward. This observation synchronizes/materializes
   tensors and may change allocation/execution behavior; it is deliberately
   separate from the first, uninspected result.
5. Manual native BF16 VJPs use the direct kernel's effective operands. A second
   BF16 control uses CPU reference effective operands transferred to the native
   device, distinguishing operand handoff from ordinary GEMM/upstream failure.

Actual gradients and upstream round trips record finite state, nonzero count,
RMS, L2 norm and actual maximum; `error_max_abs` is separately named. References
are recorded alongside each comparison. Saved context traversal is bounded to
16 nodes/tensors and reports unavailable inspection/errors, never inferred
success. It does not assume saved-tensor ordering: each tensor is compared
against same-shaped named input, master, effective-operand and upstream
references. `direct_operands` remains explicitly separate from actual compiled
context. All numeric flags retain the original 0.01 RMS ceiling.

## Interpretation (independent review required)

- Upstream round-trip or ordinary BF16 control failure: first investigate
  transfer/native BF16 execution; do not blame row-quantization rounding.
- Manual VJP and eager custom VJP agree, compiled paths fail: localize to
  compiled custom-autograd/saved-output handling.
- Enclosing wrapper agrees but original bound apply fails: checker entry-point
  distinction is supported; production correctness still needs its full gate.
- Both compiled paths fail with bad/missing saved operands: investigate saved
  output preservation. A working separate direct call is not evidence about
  compiled saved tensors.
- Only inspected replay agrees: observation/materialization/lifetime effect;
  never replace the failing first result with the inspected one.
- All custom paths fail while manual VJP agrees: investigate custom backward
  dispatch/gradient routing. A candidate registration repair would require a
  separate reviewed source patch and the original gates.
- All comparisons agree: record non-reproduction, not a retroactive PASS.
  The other shapes and production model remain unchecked by this diagnostic.

## Local checks

`test_native_diagnostic_cpu.py` extracts the exact candidate CPU formulas and
checker helpers. Explicit CPU-only packing/project adapters exercise the
diagnostic plumbing with `aot_eager`; they are not native emulation evidence.
Normal and optimized Python checks cover original-path retention, exact seed
oracles, actual/error magnitude separation, saved-context capture, missing/NaN
gradients, and deliberate zero-VJP injection. They cannot prove the host's
torch2.11/Neuron/NKI behavior. The test uses one CPU thread and no model.
