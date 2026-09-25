# Narrow failure-reporting repair — pending independent review

The original 077481d0 checker remains **REFUSE** in fresh review 90db28e0.
Its verbatim NaN/+Inf/-Inf reproducer and partial-output artifacts are unchanged.
This successor repairs diagnostic reporting only. It is not a native result,
source acceptance, arithmetic repair or launch authorization.

## Exact change

The comparison still decides identical dtype/shape/bytes and records raw tensor
SHA-256, finite status and nonzero count exactly as before. Finite max_abs values
are unchanged. A nonfinite maximum is now JSON null plus max_abs_nonfinite set
to its explicit Python spelling (nan or inf). Thus a nonfinite actual output
against the finite fixture remains FAIL_EXACTNESS and the outer receipt remains
REFUSE_OR_UNSUPPORTED, without strict JSON encoding failing on the error metric.
This field is descriptive only; it is not a replacement tolerance or zero error.

Final receipts are strictly encoded before exclusively creating rankN.json;
an unrelated invalid metric cannot leave a newly opened partial JSON file.
Local process-group cleanup now runs in a finally around final reporting,
including encoding, exclusive-file and stdout errors. No end barrier was added.
Existing artifacts are never overwritten. A disk write failure can still leave
an incomplete file; it cannot be reported as successful evidence. Setup/import,
distributed initialization and synchronize-API errors before the original main
try retain their existing reporting limitation; this patch does not expand scope.

## Evidence and limits

test_reporting.py extracts the exact reporting helpers/finalizer without entering
native main. Its default mode uses at most one scalar with stdlib byte arithmetic,
not Torch/NKI simulation. Five test groups pass normal and -O: the original
nonfinite failure remains reproducible; fixed cases encode complete refusal
receipts; finite/signed-zero records remain identical; serialization/file/stdout
failures always reach cleanup; the rest of the checker AST, kernel, reference
and arithmetic tests are unchanged; default OFF imports no native runtime.
The optional K2_REPORT_TEST_REAL_TORCH=1 mode uses only 1x1 CPU tensors and
must wait for the shared tiny CPU slot. Its outcome is recorded separately,
not implied by the scalar result. Neither test is a native legality/correctness
or whole-step performance claim. Earlier checks-*.json remain historical original
packet receipts; reporting-repair checks are named separately.

## Binding and authority

The old 8555da97 native host binding is unchanged and disabled, and its checker
hash is stale for this successor. Do not run it. Independent repair review,
separate exact rebinding, all four-rank/all-case gates and sole Operator assignment
remain required. No host calls, routing producer, production train.py, data,
prepare/evaluator, threshold or queue change occurred. Kernel 57b5b118 and
reference 14572563 are byte-unchanged. Rejected full candidate e4d93 remains REFUSE.

Night-mode decision at 21:57 UTC: address only the independently reproduced R1
reporting defect. Reason: preserve failed native evidence before any future
consumer feasibility probe. Undo: decline this successor; original evidence and
disabled binding remain intact. Supervisor owns canonical decisions and gates.
