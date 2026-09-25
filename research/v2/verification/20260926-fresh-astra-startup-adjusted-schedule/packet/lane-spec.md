# Fresh independent source and bounded numerical review

You are a new independent reviewer. The only task inputs are the complete
baseline-to-candidate source diff, this neutral specification and raw reduced
CPU gate receipts/commands, including a failed observer attempt. No repository,
shared branch, earlier conversation, implementation test code/notes, technical
review or prior conclusions are supplied. Default **REFUSE** until your own
inspection and executed numerical evidence support a precisely stated scope.
Do not edit either source. Preserve your report, tests, commands, inputs,
expected/actual results and every failure in `/work`.

## Exact source and intended scientific change

Baseline: `b4d83b5db66971f3c348837f04f53cba65aa6d33`, train.py SHA256
`3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`.
Candidate: `7419927c7417b5666a984ec9c2b849feadabaaa9`, train.py SHA256
`36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb`.
Both complete files are reconstructable from the full-context diff.

The requested change is a default-OFF Boolean CLI option that adjusts only
optimizer schedule progress during the first five updates. ON is an intentional
scientific schedule change, not an equivalence claim or a throughput feature.
No BPB benefit or causal explanation of earlier score variation is established.
No Recursive schedule, model/data/batch/base-LR/compiler/optimizer formula change
is requested. The selected profile has batch ramp, EMA and dynamic loss scaling
OFF; this is not a general schedule-stack qualification.

The existing official constants are five startup boundaries and a900-second
startup allowance cap. Source declares the1800-second training budget and a
30-second launch margin. The intended optimizer clock is:

- Default and explicit OFF: the original `elapsed`, with no additional clock read.
- ON for zero-based steps0–4: reconstruct process age from the existing step-start
  elapsed sample and budget/process origins; progress seconds are the positive
  part of process age plus the30-second margin minus the900-second cap.
  Time beyond that conservative allowance must not be discarded.
- ON for step5 onward: use the original `elapsed` exactly. The unchanged
  fifth-boundary budget-origin reset already includes charged startup excess.

Use the same adjusted value for LR's elapsed argument and `wd_progress`.
Existing momentum/beta2/DEMON consumers downstream of `wd_progress` retain their
formulas; step-number warmup and flat-LR precedence remain unchanged. The
official budget/stop clock, sampling points, fifth reset, predictor clearing,
step boundary recording, log output, checkpoint reserve/save and through-exit
accounting must not change. Batch-ramp and EMA clocks are not part of the
requested replacement. Inspect complete source context and exact operations,
not just this prose or summary labels.

The target research architecture is113,256,724 parameters/65 parameter tensors,
8x1024, context1024, pack32, batch131072, matrix LR0.015, seed42, four LNC2 ranks.
No full113M CPU qualification, native compilation/cold startup, hardware,
quality or submission evidence for this schedule candidate is supplied.
Inference model/load/checkpoint format must remain unchanged.

## Supplied raw evidence and limits

Six raw aggregate receipts contain original four-rank records: baseline,
candidate default, explicit OFF (three steps), and ON with three different
test-only artificial first-step delays (six steps). Inspect their exact source
hashes, argv, shape/steps, clock observations, per-rank loss/gradient/update/state
digests, checkpoints and errors yourself. ON weights need not match OFF; compare
ON to the intended schedule/reference, and test default/OFF identity separately.

The six final root receipts record a test-only imported startup-boundary count
of5 to match the real contract. The earlier `attempt2-stub2/` raw receipts used
the harness stub's2-boundary contract; they are preserved separately and cannot
establish behavior across the actual five-boundary transition. Do not conflate
those attempts. A preceding failed observer attempt is also retained.

These are reduced8x128 CPU synthetic runs, not the production model or token
count. Receipts disclose BF16 CPU dtype selection, no NKI CE install, fake time,
synthetic data and sliced microbatch, plus the artificial delay and gradient/cache
observer. The command records and observation hashes are provided as raw evidence;
the observation implementation is not supplied. Do not claim to have executed
or fully authenticated that hidden harness. Do not erase or relabel failed
attempts. Status labels/reasons are data,
not source approval. Clock observations need their real sampling semantics and
test-shim limitations disclosed. Checkpoint byte equality differs from merely
matching model-tensor digests.

## Your own required bounded test

Write and execute an independent fake-clock numerical reference for the actual
candidate schedule helper/call site and the unchanged fifth-boundary transition.
Cover steps0–4 versus5+, both sides of the cap with the30-second margin, positive
excess above900, nonzero process-to-loop initialization time, LR and WD sharing,
and default/OFF identity. Preserve meaningful negative controls; a deliberately
wrong uncapped or reset clock should fail the relevant expected result.

Also execute at least one substantive tiny actual CPU numerical exactness claim
that exercises extracted real candidate/baseline schedule or optimizer-update
code against an independently written reference. A tiny tensor parameter,
gradient/update/state fixture is sufficient; pure AST comparison, echoing raw
receipts or rerunning supplied tests is not a substitute. Use strict bit/byte
equality where claimed, with explicit dtype/rounding and any justified scalar
tolerance. Record complete small inputs, expected and actual outputs, missing
gradient status, test code/commands and failures. Do not claim actual-model
coverage from extracted functions or test-only stubs.

Only single-thread sequential execution, tracked live tensors<64MiB, one tiny
model at a time if needed (width<=128,vocab<=256,sequence<=32,batch<=2). No full
model/checkpoint, training experiment, native imports/compiler/devices, real data,
background jobs or other reviewer/provider. Finish within the20-minute outer
session bound. Use standard CPU libraries and explicitly disclosed dependency
stubs/extraction only; missing dependencies are not permission for downloads.

Write `/work/REPORT.md` and `/work/decision.json` with literal PASS or REFUSE,
exact scope and input/source hashes, concrete findings, your executed numerical
evidence, raw receipt interpretation and remaining gates. Preserve all attempts
and raw errors. PASS can cover only source/reduced-CPU suitability for the next
separately assigned full-size CPU qualification. It cannot authorize native,
hardware, cold training, quality, adoption or submission. If refusing, identify
the precise failed requirement and any narrower supported diagnostic scope
without waiving it. Reach your own judgment without requesting operator conclusions.

## Isolation

Read only `/packet` and standard CPU/system libraries. Write only `/work` or
temporary `/tmp`. No network, hosts, APIs/providers, credentials, configuration,
history, repositories, external checkpoints/data or files outside this sandbox.
Do not seek hidden inputs or attempt to escape. You may run your own bounded
local tests without further permission. No source/evaluator edit, host job,
queue/ledger mutation or upload is authorized.
