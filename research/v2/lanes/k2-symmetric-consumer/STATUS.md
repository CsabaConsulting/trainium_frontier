# K2 symmetric grid2 — 26 September 05:17 IST

## 06:38 IST: new registration prototype selected and implementation active

Researcher proposal8bab89e1/canonicald86e3045 is complete; Supervisor selection
a6ebb4c2 authorizes distinct Implementer CPU-only work. One registered forward
per dtype plus register_autograd, same automatic backward and immutable kernel
2c8295/reference145725. No explicit wrap or second consumer op. Tiny eager/AOT
tests must show the registered consumer is actually used, with wrong-consumer
negative controls as well as strict bytes/changed tangent and metadata replay.
No native/production activation; fresh independent source review still pending.
The prior compiled-zero-gradient failure is not reversed by a proposal or CPU
surrogate test. Both running host quality jobs remain separate and protected.

## 06:27 IST: first-fixture failure authenticated; registration design next

Source-neutral FP32-first diagnostic ontrn22 ran00:33:35–00:34:04UTC.
All4 ranks again produce zero custom-compiled gradients versus32709 expected
nonzeros/maxerror8.0; direct/eager and ordinary compiled controls match. All
recorded inputs unchanged. Independent461 checks including own binary32
reconstruction and root normal/-O replays pass696730ff/26a03fb8. This is a
FAILED diagnostic, not native qualification. Prior BF16/zero-case history is
not needed to reproduce it; precise compiler/runtime cause remains unknown.
Raw015bacec retains31artifacts and9PID/256row explicit device/lock quiescence.

Installed SDK source-only observations0770f22a/7ff42226/9284abc8 resolve the
wrapper API without executing it: automatic Kernel.__call__ already constructs
a HOP using its LNC, __getitem__ replaces LNC, nki_op exposes explicit autograd
registration and functional decomposition. Do not run a redundant guessed
wrap-only patch. A new minimal registered-forward/autograd proposal is being
prepared, keeping exact2c8295 kernel,145725 oracle and automatic backward
callable. No implementation/native release yet; original failed receipts and
full20/all4 strict byte gate remain. Current full-quality jobs are separate.

## 05:45 IST: native exactness failed; parked

Actual native run00:08:48–00:09:18UTC completed13/20cases on each of four ranks.
All four fail `EXACTNESS_FALSIFIED: unique/fp32`: compiled enclosing backward
produces zero gradients versus32709 nonzeros/max_abs8.0, while the same-case
direct and eager paths are exact. This is a correctness failure, not a
compiler error, accepted kernel, speedup or quality result. No retry assigned.
Rawc087cbb7/canonical7f0fbbde contains31files556014bytes rehashed by root.
00:10:44 explicit postflight records all8 observed identities absent, full
261-row census without related workers/compiler, deviceidle/original596823
lockfree. Short-lived compiler/timeout PIDs were not observed; no IDs invented.
Independent diagnosis is read-only. Separate qualifiedbcbd full fallback is
released ontrn21 with fresh checks; this failed kernel is never part of it.

## 05:38 IST: new-evidence source gate passed; native actually dispatched

Fresh input-only Astra with NEW versioned primary API documentation plus exact
unchanged checker/reference completed00:06:01UTC. Source/CPU-onlyPASS,48cases,
17negativecontrols,6,291,456 outputbytes exact; no nativeclaim. Root816checks
normal/-O authenticate419files and rawcases. Reportarchive6fd5f333 at
research/v2/verification/20260926-fresh-astra-k2-symmetric-api/.
The original missing-evidence REFUSE below is preserved, not silentlyoverridden.

Separate one-shot native release6fd5f333, exact710ea677 mechanical127checks
replayed normal/-O. P r2 terminal/quiescence authenticated; actual trn21dispatch
00:08:48.310143UTC PID3101403, originalflock/freshchecks, tiny600+60s4rank/LNC2.
Fourworker/underlock evidence retrieving; no consumer output/timing yet.
All20cases150bytecomparisons/rank and explicit descendantpostflight required.
No fullmodel/autograd-production/speed/BPB/adoption/uploadauthority.

05:19IST update (correcting the earlier rounded05:20 label): fresh review
completed05:17:21IST, archived5ac75f7b. **REFUSE**
because packet lacked version-specific API support evidence for grid/address
semantics; no new source defect demonstrated. Independent16-case CPU model
compared2,097,152bytes exactly, but cannot establish native lowering. Root
authenticated all129archivefiles and per-case raw output/ownership bytes.
No native authority. Official-versioned documentation evidence is being
collected for a later independently assessed supplemental packet, not a
reworded retry or override. Exact source remains unchanged.

Native binding mechanics alone passedf4aeca71/canonical8f9aed56; root read
full review/runtime and replayed127checks normal/-O. This does not satisfy
the separate source gate, descendant postflight or future Operator release.

Distinct source9ce2e6a0/kernel2c8295a4 partitions the tiny supplied-metadata
consumer over two explicit programs with symmetric row-loop schedules.
Old namespace-only route remains PARKED after NCC_IXGM002; this is a new
proposal-first supported design, not a renamed retry or production patch.

Fresh input-only isolated Astra began05:12:28IST, archivefddfbe0c. Own CPU
reference work is in progress; no final verdict/native result yet. New trn21
native packet710ea677 is disabled and mechanically under independent review.
It must follow full75b9+eval and P decoder terminal/retrieval/quiescence, with
separate Supervisor release. Strict byte oracle, originallock596823,600+60s,
all4rank receipts and explicit descendant postflight are unchanged. Component
timing cannot establish whole-step speed, routing cost, quality or adoption.
