# K2 one-line native namespace repair — native disabled

The only executable candidate change is `global nl, nisa` immediately inside
`build_kernel`, matching the existing 3e79 factory at lines 4238–4242. Proposal
commit **378d7f4c** preceded implementation. The candidate kernel SHA-256 is
`4939265bf73b3d5f42276e72e48f022df338226a40a9be5b79ed63bde6a4f5b8`.

The old native attempt remains **REFUSE**. On all four ranks, the old consumer
failed at its first `nl.bfloat16` lookup; no consumer arithmetic or latency
measurement completed. This repair follows the independently archived
`d5c668b2` audit. It does not prove the installed frontend uses only globals,
that this resolves its error, or that later dynamic loops, indirect gathers,
layout, SBUF allocation, rounding and autograd will compile or pass.

## Exact scope and own evidence

- The complete decorated nested function and trailing factory return are
  byte-identical to 18b9365a, with tail SHA-256
  `6ef9cbebdcce813f07fe017da710a776f76a6f665da0302e50266c370791720e`.
- The checker remains `89189024`; scalar reference `14572563` and its original
  test `972cb83f` are unchanged. Dtypes, dimensions, exact-byte criterion,
  signed zeros, nonfinite reporting, API shape and cases are unchanged.
- Own 32-check stdlib fixture passes normally and under `-O`: original aliases
  use closure cells/LOAD_DEREF, repaired aliases use module globals/LOAD_GLOBAL.
  Fake NKI modules only capture the undecorated Python function; no kernel
  body executes. Explicit `compile(..., optimize=0)` preserves source asserts
  for both fixture invocations. Real NKI and Torch are never imported.
- The unchanged 20-case scalar reference passes normally and under `-O`;
  peaks are 7,603,477 and 7,591,445 traced Python bytes, below its existing
  8 MiB bound. Both cross-tile cancellation counterexamples remain rejected
  for the old grouped arithmetic. This is emulation, not device/Torch parity.
- Default and explicit OFF still refuse before all native imports. No
  production train.py, prepare.py, evaluator, dataset, guard or launcher changes.

Exact local commands/results are in `LOCAL_RECEIPT.json`; the full-context
one-line diff is `kernel-full-context.diff`. One read-only missing-filename
lookup is recorded; all executed scope/scalar checks passed. No tensor jobs,
full model, host actions or native retries occurred.

## Fresh review and later native gate remain separate

`REVIEW_INPUTS.json` identifies a compact neutral spec, full kernel diff and
five byte-identical original native receipts. They are extracted without
modification from authenticated bundle 94fb52c4 (113,871 raw bytes overall),
not rewritten summaries. Own tests, this note and prior conclusions are not
fresh-review inputs. The Supervisor/Verifier assembles the isolated review.

Old 8555/df483 native binding files, lane proposal and historical manifests
remain frozen to their original source pins. **They cannot launch this new
source.** Only a separate reviewed, new-label binding plus explicit sole
Operator assignment could permit one diagnosed retry. It must retain the
original lifetime lock, fresh in-lock ps/neuron-ls, STOP/resource/pin checks,
all four complete rank receipts and every original strict exactness case.
No automatic retry, production adoption, training, evaluator call, quality
decision or performance claim follows from this local result.

Module-global aliases are the intended scope change and are populated only
when explicitly enabled. They refer to the same imported NKI modules on
repeated factory calls. This tiny module has no other factory alias writer;
future integration would require a fresh namespace-collision review.

Undo: decline the isolated branch. Original bytes/refusals stay durable;
no remote or production state needs reverting. Supervisor owns the canonical
night-mode decision log.
