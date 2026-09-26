# P2 local-constants candidate — READY FOR SEQUENTIAL TESTS, NOT PASS

## Actual sequential checks — 26 September, approximately15:28 IST

Supervisor granted the local process slot. FP32 reference and both int32/int64
CPU suites passed, including OFF3 identity and interpreted ON2 table/RMS
behavior. Exact production embedding parser passed. The exact table parser
FAILED before backend at line257 `first = group * partition`:

```
'mul' expected (int, int) or (float, float) or (list, int) or (int, list)
or (tuple, int) or (int, tuple) or (str, int) or (int, str), got (object, int)
```

`nl.dynamic_range` returns a runtime VirtualRegister, not a Python arithmetic
integer. CPU interpretation did not model that compiler type restriction.
`parser-result-table.json` retains the actual failure; no backend, NEFF,
simulator or native table PASS is claimed. Source4079 remains frozen and is
NOT LAUNCHABLE. Supervisor authorized one minimal derivative using the SDK's
explicit dynamic HBM selection, without unrolling4096 groups or changing
FP32 arithmetic. That work is separate; this failure is not erased.

26 September 2026, Implementer packaging only.

Source SHA-256:
`4079e8e6bff1f8438e0430a8b8b2099085da5d6f908149268ac17e8b318fdbff`.

Two local-geometry scope moves only, one in existing K2 embedding backward
and one in its table counterpart. FP32 masters/gradient accumulation/RMS,
BF16 gathered activations, original hashes and all operations are unchanged.
Old sources remain frozen. Exact production geometries and limits are in
`SPEC.md`; sequential commands are in `TEST_COMMANDS.md`.

No Python, Torch, NKI compiler or simulator was executed for this packaging.
The Supervisor assigned the sole heavy-process slot to K2 full local compile.
CPU, real parser/backend, available simulator and native receipts for these
new bytes are pending. No copied old receipt is presented as a fresh result.

K2 timing qualification, table correctness/native admission, exact proposals,
fresh original-lock/resource checks and actual control selection remain.
No host, shared queue/state, proposal, goal, evaluation, data or submission
file was changed. This candidate is not ready to launch merely because its
source and harness are packaged.

## Compiler diagnostic correction, approximately15:23 IST

The separate compiler lane's first K2 BIR check used raw `CompileOptions`
with a different allocator from the actual JIT defaults and encountered an
SBUF allocation error. That mismatched-options check is NOT a demonstrated
production-source failure. A production-options rerun is now pending; its
outcome must govern, not the earlier diagnostic. The compiler lane reports
K2 production-shape simulator numerics passed with zero error. These are
external K2 reports, not executed table tests or table native qualification.

The table source has not been compiled. No speculative allocation repair is
included. Keep this source frozen while the matched K2 compiler evidence is
resolved. Any later change requires a distinct source/diff and receipts.
