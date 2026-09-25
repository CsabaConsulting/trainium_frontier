# K2 embedding scatter — 26 September 2026, 01:26 IST

Candidate25b731f7/e4d93 remains isolated, default OFF, not merged or adopted.
One local Astra implementation exists; no second-family implementation ran.

Fresh independent Astra session01a0da15-9827-7cc0-8845-a39138c52543 completed
19:47:21UTC with **REFUSE**. It received only full-context diff, neutral lane
spec and raw receipts. Its own test passed708/714 checks: all696 actual CPU,
default/OFF/integration checks pass, but six intended-native-algorithm CPU
emulation exactness comparisons fail. Native code was NOT executed.

Concrete failure: duplicate-ID BF16 gradients [2^24,1,-2^24] at positions
0,128,129 give dense FP32 result0 and tile-grouped result1. Smaller gradients
[1,2^-24,-1] similarly differ by2^-24. Per-tile regrouping changes FP32
rounding, so FP32 accumulation alone does not satisfy the required exactness.
This is not evidence of missing dequantization, a measured native error or a
performance loss. Preserve the REFUSE and counterexamples; no relaxed pass.

Evidence: `research/v2/verification/20260926-fresh-astra-k2a/`, including the
unmodified reviewer report/test/results and a labelled operator-derived input/
error supplement. Supervisor reran95 archive-integrity checks normal/-O.

Exact117M reference/OFF/ON CPU packet is prepared in `fullsize-cpu-packet/`,
but held by this source finding and laptop resources. A6.192GB tensor floor
already exceeded6.074GB available before optimizer/caches/activations. Do not
run it locally, substitute reduced shapes, or use a CPU ON pass to erase the
native-algorithm emulation failure.

Next: seek an accumulation algorithm satisfying the agreed dense semantics;
independent review plus genuine full-size CPU/native gates and>=5% whole-step
speed still apply. Current work does not authorize a host launch or adoption.
