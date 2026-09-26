# P2 compiler-local geometry packaging

Implementer-only delivery, 26 September 2026. This is a new frozen candidate,
not a modification to previously tested or failed sources. No host action,
launch permission, native correctness, throughput or BPB claim.

## Exact source boundary

- Previous P2 source: `0127d02dcaae0e6fdb70db693697caf02233db31d926dddd4649a1d496e6b543`.
- New source: `4079e8e6bff1f8438e0430a8b8b2099085da5d6f908149268ac17e8b318fdbff`.
- New K2-only comparison parent: `662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00`.

`SOURCE.diff` is the complete production change: move the two immutable
geometry tuple assignments and their comment from each outer factory into
its decorated `resident_consumer` body. The embedding factory change is the
exact K2-only local-constants patch. Static inspection found the same captured
names in the table factory, so the identical scope repair is applied there.
Every later instruction in both kernel bodies remains unchanged.

The real local NKI parser reproduced the original K2 `unbound variable
'tokens'` failure, with five further captured geometry names exposed in
sequence. A Python interpreter preserves closures and cannot establish NKI
parser legality. Local constants remove that particular dependency; this
new table source still needs its own real compiler evidence. Moving constants
does not establish legality of dynamic table-group addressing or allocation.

## Arithmetic and integration unchanged

Only the lexical scope of constants changes. No hash, table layout, loop,
DMA, BF16/FP32 conversion, gradient accumulation, autograd, optimizer,
clipping, RNG, timing/budget, checkpoint, evaluation or launch flag changes.
The int32/int64 token guard remains as previously repaired. Default tables
remain OFF; the scientific table arm is `--ngram-ve` on qualified K2.

Tables retain FP32 master parameters, FP32 dense gradients and microstep
accumulation, FP32 AVG reduction and RMS state. The exact ordinary reference
is `F.embedding(rows, fp32_master_table).to(torch.bfloat16)`; do not cast the
whole table before gathering and do not compare to the historical BF16
collision-rounding path. The compiled lookup's forward remains that ordinary
Torch gather/cast. Custom NKI is only the registered backward consumer.
No fictitious standalone NKI forward kernel is introduced for testing.

## Exact production geometries

Both kernels target Trn2, LNC2, two feature-owner programs, BF16 incoming
tangents and FP32 dense outputs. IDs/metadata cover 32,768 tokens per rank.

| Input / output | Embedding backward | Each table backward |
| --- | --- | --- |
| gradient | BF16 `[32768,1024]` | BF16 `[32768,128]` |
| counts, starts | int32 `[8192,1]` each | int32 `[524288,1]` each |
| positions | int32 `[32768,1]` | int32 `[32768,1]` |
| group maximums | int32 `[64,1]` | int32 `[4096,1]` |
| output | FP32 `[8192,1024]` | FP32 `[524288,128]` |
| row group / owner features | 128 / 512 | 128 / 64 |

The K2 launch guard requires pack32/context1024/width1024, so the actual
model microbatch shape is 32 sequences of length 1024 (not the inactive
module defaults before launch flags).
Ordinary embedding forward gathers FP32 `[8192,1024]` weights using int32
token IDs `[32,1024]`, returning BF16 `[32,1024,1024]`. Each table forward
gathers FP32 `[524288,128]` weights using hashed int64 rows `[32,1024]`,
returning BF16 `[32,1024,128]`. Stable sort/search metadata remains ordinary
Torch. The two table calls share geometry but have separate metadata and
weights. Metadata preparation and gather must be included in full training
graph compilation and native timing; a backward-only NKI compile does not
prove those paths.

## Packaged tests and limits

`test_tables.py`, `test_fp32_reference.py` and `cpu_instruction_model.py`
are byte-identical copies of the previous P2 suite. The interpreter walks
the AST and replaces the same two geometry tuples whether outside or inside
the decorated body; no harness adjustment to accommodate the repair is made.
Use the new K2-only parent for the existing exact shared-helper AST check.

`test_parser.py` checks one selected actual factory per invocation using the
version-matched real NKI frontend and exact production input metadata. It
does not import the training module, rewrite the factory body, allocate
production tensors, or claim backend/NEFF/simulator success. It writes a
separate embedding/table parser receipt; unsupported compiler behavior fails.

No tests have been run on this new source by this packaging delivery. Earlier
P2 receipts are historical, not fresh PASS receipts for these bytes. The
unchanged suite's limits remain: reduced CPU geometry, mocked collectives,
OFF3 with K2 disabled and SGD, ON2 with interpreted backward and actual table
RMS, same-shape causality, and the retained cross-shape prefix discrepancy.

## Remaining gates

1. Wait for the Supervisor's laptop process slot. Only one Torch/NKI/compiler
   or simulator process at a time; no hidden background test or parallel pool.
2. Run sequential exact-source CPU int32/int64, FP32 reference, both real
   production-shape parser checks, full available compiler/backend checks,
   and simulator numerics if available. Record unavailable stages explicitly.
3. Preserve one independent review of the precise scope delta and evidence.
   Existing review composition may avoid redundant reviews, not missing tests.
4. K2 whole-step timing must actually pass. Table native admission additionally
   needs its exact completed proposal, original host lock and resource checks.
5. Measure the table patch's native correctness, startup and whole-step cost;
   table quality is final fresh-process public-20M/context-1024 at official
   1800 seconds, not a 3% table throughput threshold. Preserve an actual
   matched same-host K2 control for the eventual table comparison.

The next host slot remains K2-first. Packaging does not authorize tables to
preempt its timing or to bypass table correctness, official accounting,
existing selector or submission checks.
