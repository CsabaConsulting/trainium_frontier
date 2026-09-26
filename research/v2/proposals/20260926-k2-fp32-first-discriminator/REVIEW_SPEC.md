# FP32-first diagnostic specification

This separate disabled diagnostic tests one question after the original four-rank
`unique/fp32` compiled-custom-backward failure: does that failure require preceding
BF16 or zero-expected cancellation fixtures? It does not repair or qualify the
kernel. Original failure receipts and strict twenty-case gate remain unchanged.

Preserve kernel SHA256 `2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae`
and reference SHA256 `14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f`.
The custom wrapper must retain AST identity to `make_wrapper` in original checker
`89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850`; no math,
clone, dtype or registration fix is selected.

Only V256/C128/T256, seed42, FP32 unique first. Execute direct consumer, eager
custom autograd, compiled custom autograd, then separately named ordinary compiled
embedding. All routes must be attempted and retained after a numerical mismatch.
Each route records pre/post upstream, ids, weight and four metadata tensors, with
shape/dtype/finiteness/nonzero/hash; these host observations are not measurements
inside a compiled VJP and may affect execution. Compare output/gradient bytes with
the dense CPU oracle, strict zero tolerance, preserving missing/nonfinite/error
records. Any mismatch, mutation or exception gives a nonzero diagnostic exit.
All-exact completion is only an observation about this fixture, not native
qualification, quality, production routing, timing or adoption.

Review must default REFUSE until evidence supports this restricted diagnostic
scope and independently execute a small CPU reference/exactness or negative
control. Native/NKI behavior cannot be inferred from substituted CPU backends.
No host access, full model, source edits, protected data or provider retry.

Future host packet only: trn22, unique label
`fs-20260926-k2-fp32-first-trn22-d1`, after current full/eval and W1 terminal,
retrieved and quiescent. Four ranks/LNC2, 600+60-second child timeout, original
lifetime lock and fresh in-lock ps plus neuron-ls, 10.5GiB free disk, six private
initially-empty caches, explicit descendant/compiler/device postflight. Disabled
until separate independent binding review and Supervisor/Operator assignment.
