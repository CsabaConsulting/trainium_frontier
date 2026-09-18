# Stored-state rounding in the AdamW accuracy gate

2026-09-18, 20:56 UTC. Kernel and optimizer arithmetic are unchanged by this
gate correction. Wave57 failed, and wave58 deliberately reproduced that failure;
both original logs remain preserved. This is a correction to the numerical
contract, not a claim that the original whole-chain bound passed.

For an update with FP32 stored moments, first compare each new moment against
the independent FP64 formula. Then compare the weight arithmetic against FP64
using those actual stored moments. In symbols, with `F(p,m,v)` the weight-update
formula and `m*,v*` the ideal moments:

`w_actual - F(p,m*,v*) = [w_actual - F(p,m_fp32,v_fp32)] +
                       [F(p,m_fp32,v_fp32) - F(p,m*,v*)]`.

The original weight envelope bounded the first term but did not propagate the
second. Cancellation when forming a near-zero first moment can make the second
term larger than that envelope even for correctly rounded FP32 operations.

At iteration6/parameter1, the two flagged elements are reproduced by fresh
native foreach updates from the exact same inputs. CPU ordered FP32 arithmetic
also reproduces the second exactly and the first within one FP32 ULP. The
original first/second moment gates both pass. Wave58 records:

| Flat index | Original weight error | Original bound | Weight effect of stored-state rounding | Remaining weight arithmetic error |
| --- | ---: | ---: | ---: | ---: |
| 2542192 | 3.08822e-9 | 2.85147e-9 | 3.11478e-9 | 2.65589e-11 |
| 6021746 | 4.91953e-9 | 1.23917e-9 | 4.91460e-9 | 4.92778e-12 |

`adamw_stage_accuracy.py` therefore checks both moment stages independently,
then the conditional weight stage. The original `4*gamma(32)` constants,
subnormal floor, and `2e-6` relative RMS limits remain. The whole-update RMS
against ideal FP64 and the separate accumulated-trajectory check also remain.
The original unpropagated failure count is still reported, not hidden.

Two CPU fixtures reproduce the original failed contract before testing the
corrected one. Negative controls corrupt a moment (and consistently recompute
its weight) or corrupt the weight alone; the corrected gate must reject both.
Conditioning on actual moments cannot excuse an inaccurate moment because its
independent state check is enforced first within the same gate.

The unrelated native first-step square-root approximation from wave55 remains
a real numerical deviation: approximately 7.8e-6 relative error in sqrt of the
actual FP32 variance. Stored-state propagation does not explain that case.
The NKI GpSIMD inverse-square-root/Vector reciprocal path passed those original
component gates.

Wave59 stopped in the two-element CPU fixture, before any optimizer integration
update. Its first-moment relative RMS is 5.85e-5 because this fixture deliberately
selects only two near-cancellation outputs; that is not a representative sample
of the 8M-element table, whose observed first-moment RMS passed. The fixture now
explicitly checks the original and corrected elementwise bounds and reports its
RMS without treating it as a full-table sample. Its two negative controls must
fail the relevant elementwise state/weight bound. Every full-table stage and
trajectory RMS limit remains enforced without a bypass parameter. This fixture
correction does not change kernel, optimizer or full-table gate arithmetic.

Evidence: `run-logs/fs-w58-adamw-rounding-diagnosis-r1.log`, especially
`ADAMW_FAILED_ELEMENT_DIAG`; clean postflight despite the intentional assertion
failure. Next requirement: all repeated-update, cache, skip, restore and
Parameter-identity gates must pass before full-model timing.
