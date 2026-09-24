# Track B2 touched-row n-gram RMSProp feasibility result

## Decision

No `train.py` candidate was materialized. The mathematical design is sound, but the
repository does not establish a legal, compiler-supported way to obtain and update the
touched-row records on the pinned native TorchNeuron SDK. Guessing here risks silent wrong
gradients, a compiler failure, or a slower full-budget run.

The intended control remains byte-identical at
`submissions/dense6x1024_public0992851/train.py` (SHA-256
`35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2`). Neither
`prepare.py` nor any evaluation/data/scoring path was changed.

## Audited path and exact semantics

The control uses four BF16 n-gram embedding tables. Direct indexed lookup produces a dense
parameter gradient (`aten::gather.out(... sparse_grad=False ...)` in preserved native logs).
`clip_and_sync_gradients` passes all four n-gram tensors to one `clip_grad_norm_` call on
each rank, so they share one joint rank-local coefficient before `ReduceOp.AVG`; then
`MuonAdamW._rmsprop_step` applies bias-corrected RMSProp. Therefore an
equivalent touched-row path must, in order:

1. accumulate every occurrence gradient across local microbatches;
2. deterministically coalesce duplicate row indices;
3. compute one joint local norm and clip coefficient across all four tables;
4. exchange the clipped row records across all ranks;
5. deterministically reduce duplicates and divide by world size;
6. lazily materialize `v = beta2**delta * v`, then apply the current gradient square and
   the table-global bias correction; and
7. write identical updated rows on every replica.

Dense RMSProp decays moments on untouched rows every global table step. Consequently, the
requested bitwise-unchanged *physical* untouched state is equivalent only if each row also
has a last-touch step and deferred decay is materialized before its next use. Simply skipping
untouched state is wrong.

## Blocking SDK gaps

- The present backward exposes only the full dense `p.grad`. Finding touched rows afterward
  requires a full-table scan and does not remove backward allocation/traffic.
- `nn.Embedding(..., sparse=True)` would require sparse COO autograd, sparse accumulation,
  sparse distributed exchange, and sparse optimizer support. No preserved pinned-SDK run or
  primary installed documentation establishes that chain on native Trainium.
- A custom autograd function or output hook would need to export runtime occurrence gradients
  from a compiled backward without Python side effects. The repository has no successful
  native proof of that contract.
- Fixed-capacity all-gather can avoid variable collective sizes, but it does not solve how
  occurrence gradients are obtained, and it still requires supported deterministic
  unique/sort/segment reduction plus indexed row/state writes.
- The preserved XLA experiment demonstrates that repeated `index_add_` across a graph boundary
  silently retained only the last accumulation. That is not proof of the native behavior, but
  it makes an unprobed scatter-based implementation unacceptable.
- NKI 0.5 can implement row math after records exist, but no in-repository evidence provides
  an inter-rank collective inside that kernel or an exact compiled-autograd handoff.

These are compiler/runtime feasibility gaps, not a finding that sparse row updates can never
work on Trainium.

## Offline result and bounded next step

`touched_row_reference.py` uses the production geometry (four named tables, each
1,048,576 rows x 128 columns) with sparse fixture storage. It proves duplicate reduction,
four simulated ranks, one joint per-rank clip coefficient, AVG, global-step bias correction,
deferred moment decay, untouched physical rows, deterministic replay, and
serialize/reload/**next-update** equivalence. It covers BF16 parameters with FP32 and BF16 state,
rejects invalid indices/shapes/dtypes, and statically binds the actual 35c9 model-state/
strict-load evaluation contract and table keys. Its BF16 mode is an explicitly defined software
emulation: occurrence additions, clipped records, collective accumulation, and AVG are rounded at
declared boundaries and matches its dense oracle bit-for-bit. Native Trainium may choose a different reduction tree or accumulation order,
so native BF16 equivalence remains an explicit hardware probe gate rather than an offline claim.

The source check establishes only that the approved 35c9 `train.py`, `prepare.py`, model-state
save expression, and strict-load expression are unchanged. It does **not** construct a synthetic
checkpoint and does not claim checkpoint-key, table-byte, or logit identity. The future component
probe must use one authenticated real checkpoint, strict-load it through the exact control and
candidate evaluation paths, and compare the complete key set, all four n-gram table bytes, and
predeclared-input logits exactly. Failure or absence of that evidence blocks any training run.

The safer next action is a separately authorized component-only native probe with tiny fixed
shapes. It must test, independently, sparse/occurrence-gradient capture, multi-rank fixed-shape
exchange, deterministic coalescing, and indexed BF16 parameter/FP32-state writes across repeated
step boundaries. It must compare complete parameter and logical-state tensors to the dense
reference before any full model candidate is written. A component pass would justify reopening
this proposal; a host fallback, unsupported operator, wrong accumulation, or less than a credible
3% whole-step bound closes it.

The already measured exact owner-table optimizer path (source98) is a safer bounded systems
alternative: it preserved dense math and measured a 1.12814% whole-step gain, or 3.16362% in its
separate pack32 composition. Those measurements do not authorize transplanting or launching it
here.

## Attribution

No third-party implementation was adapted in this lane. The control's existing Recursive
attribution remains pinned to Apache-2.0 source commit `a962ec43` and cited file SHA-256
`f92ecc81`; no additional upstream MIT code was copied.
