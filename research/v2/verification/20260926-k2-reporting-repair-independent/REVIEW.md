# K2 nonfinite reporting successor — PASS for the narrow repair only

Frozen successor `18b9365aabb63fa85e1720fdefc11177ca8e7853`, checker SHA256
`89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850`.
This is an independent regression review, not a fresh-session retry or a native
source/binding acceptance. Original `077481d0` fresh **REFUSE** remains intact.

I wrote and ran `audit.py` independently of the Implementer's test. **34 checks
pass normal and `-O`**, with actual 1×1 FP32 CPU tensors, Torch2.13.0+cpu and
one intra/inter-op thread. No model, native runtime/kernel, host or full-size
allocation was used. The local tensor slot is released. Reproduce with:

`env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python research/v2/verification/20260926-k2-reporting-repair-independent/audit.py`

Repeat with `python -O`. Both exact result files are retained. The preliminary
fixture was corrected to use an actual finite-mismatch record for its scenario
named `finite`; the original comparison tests already included finite controls.
No candidate code, expected value or threshold was changed.

The verbatim old/new comparison helpers were extracted without entering native
`main()`. Inputs +Inf (`7f800000`), -Inf (`ff800000`) and quiet NaN (`7fc00000`)
against +0 reproduce the original strict-JSON failure. Every comparison remains
`exact:false, finite:false`. Successor records encode fully, retaining tensor
hashes, dtype, shape, nonzero count and rejection; only nonfinite `max_abs`
becomes JSON null with `max_abs_nonfinite` spelling `inf` or `nan`. Null is not
interpreted as zero error or a tolerance. Finite exact, finite mismatch, a small
finite difference and signed-zero records are unchanged. Signed-zero remains
a byte mismatch even when absolute error is zero.

Executing the exact original finalizer on a local +Inf refusal fixture also
reproduces its partial JSON file and skipped process-group cleanup. Both raw
partial contents and exception are retained in the result JSON. The exact
successor finalizer writes a valid refusal and reaches cleanup. Independent
fault injections confirm cleanup on strict serialization, exclusive-file,
disk-write and stdout failures; invalid serialization happens before file
creation, existing files remain untouched, and a stdout failure does not erase
an already valid receipt. File-write failures can still leave partial files and
remain failed evidence, not a successful receipt.

Whole-checker AST comparison after replacing only the changed comparison
reporting body/finalizer and removing the new `math` import is identical to
`077481d0`. Kernel `57b5b118…`, reference `14572563…` and arithmetic tests are
byte-unchanged. Shape/dtype/byte acceptance, main case-status logic and zero
tolerances are unchanged. Default invocation returned `DISABLED` with both
native flags false. No source or evaluator was edited.

The repair resolves the reproduced reporting regression in its stated scope.
It does **not** cover setup/import/distributed-init/synchronize-API failures
before the original main try, prove native dynamic-loop/gather legality,
gradient correctness, routing, performance or quality. Cleanup itself may
raise; no successful receipt is inferred from that. The old `8555da97` native
binding remains disabled and pins the original checker `17e2d2ad…`, so it must
not be used for this successor. Any future native diagnostic still needs exact
rebinding, the applicable independent gates and sole-Operator assignment.
