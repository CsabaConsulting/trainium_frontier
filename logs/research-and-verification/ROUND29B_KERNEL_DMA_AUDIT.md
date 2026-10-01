# Round29b — explicit own-kernel DMA source audit

2026-09-28; analysis only under [Round29b](../../workflow/ROUND29B_2026-09-28_0240.md).
No kernel edit, model execution, compiler, host call or new verification gate.
Root owns SDK/profile verification and selection. Safe CLEAN remains unchanged.

Source: [CLEAN train.py](proposals/20260928-clean-gatefix/train.py), SHA256
`881a4f491d256fb89af33246a3b568fc3ffead9616070eea1860d7d8f7f9f749`.
Every line number below refers to that exact file. A stdlib AST enumeration
finds **35 `nisa.dma_copy` source sites**, all listed below; no `dma_transpose`,
`nl.load` or `nl.store` sites occur in these kernels. Alternative dtype/boundary
branches are listed separately, not assumed to run simultaneously.

## Scope and interpretation

Production per rank: B32, T1024, C1024, H8 attention heads, D128, Dh64,
N=B*T=32768, vocabulary V8192. BF16 is2 bytes, FP32 is4 bytes.
P128 is the usual SBUF partition axis. `B/part` below means **logical HBM
payload associated with one active SBUF partition in the source operation**,
not per physical DMA engine, descriptor, packet or observed transfer.
Broadcast views distinguish repeated logical elements from unique addresses.
Compiler splitting, coalescing, casting, spills and implicit framework copies
are not derivable from this table. These are not measured traffic totals.

AWS recommends at least2 KiB/partition and reports near saturation around
4 KiB/partition with broad partition participation. Reducing active partitions
to make each row larger may reduce engine parallelism; crossing2 KiB is not a
speed guarantee. [AWS DMA overview](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.30.0/nki/get-started/about/nki-dma-overview.html)

| Requested component | Current execution route | Historical500f comparison |
| --- | --- | --- |
| RoPE + RMSNorm | Active NKI fwd/bwd, lines179–311; Q and K at719–720 in five attention blocks. Plain `norm` elsewhere at170–171 is framework RMSNorm. | Same kernel ASTs, not removed/dead. |
| MLP forward | **Plain PyTorch**, `reference_forward`728–730 and misleadingly named `_fused_mlp_kernel`732–734; `F.linear`, ReLU, square, `F.linear`, saved `relu.T.contiguous()`. | Also plain PyTorch in500f. No requested active NKI MLP-forward kernel was removed by CLEAN. |
| Softcap CE | Active NKI fwd/bwd,313–465, for supported contiguous logits. MTP invokes it twice638–640, including when k2 weight is zero. | Same kernel ASTs. Active target gather=true; generic mask and FP32 branches remain, not separate active experimental recipes. |
| Two-tap conv | Active NKI fwd/bwd,468–594, in blocks0/2/4/6/7. Batch32 chooses two lanes; each owns16 batch rows. | Same kernel/helper ASTs, not removed/dead. |

Historical protected file:
`/home/poojith-devan/Downloads/Trainium-500f-COLD-PAIR-AUDIT-20260928-r1/train.py`,
SHA256 `500f80fec9eac2a42d6f0eae646b0e91643d46c4524ee3722e332434751fd368`.
This audit compared ASTs for all ten listed kernel/helper definitions: all
equal after excluding source-location metadata. CLEAN removed disabled flags
and unused coalescing construction paths, **not these NKI kernels**; see
[cleanup scope](proposals/20260928-clean-gatefix/SPEC.md). An archived older
experimental MLP forward must not be counted as today's custom DMA.

## RoPE + RMSNorm: all9 sites

Here dt=x.dtype, active BF16. Cos/sin are BF16 from857–858. `CS` at188/235
is `[[Dh,P],[0,H],[1,Dh]]`: time advances64 elements; the head stride is zero.
The destination includes eight logical copies of the same64 source elements.
`r0=i*128`, `t0=(i*128)%T`; T is divisible by128, so a tile does not cross
the table boundary. No explicit DMA mask or data-dependent address gather.
The backward wrapper makes G contiguous at310; source BF16 G is widened by
the DMA into FP32 `gf`, which doubles SBUF bytes, not HBM source bytes.

| Line | Direction / source→destination | SBUF tile; active partitions | HBM B/part at production | Logical tile / caveat |
| --- | --- | --- | --- | --- |
| 194 | H→S, x→xb | `[128,8,128]` BF16;128 | 2048 | 256 KiB, contiguous input slice. |
| 196 | H→S, cos AP→cb | `[128,8,64]` BF16;128 | 1024 logical; **128 unique** | 128 KiB logical destination;16 KiB unique HBM addresses due head broadcast. |
| 198 | H→S, sin AP→sb | `[128,8,64]` BF16;128 | 1024 logical; **128 unique** | Same zero-head-stride broadcast. |
| 223 | S→H, ob→out | `[128,8,128]` BF16;128 | 2048 | 256 KiB, same x/out layout. |
| 241 | H→S, x→xb | `[128,8,128]` BF16;128 | 2048 | 256 KiB contiguous. |
| 243 | H→S, g→gf | `[128,8,128]` BF16→FP32;128 | 2048 source | 256 KiB HBM payload;512 KiB destination. If actual G is FP32, source is4096 B/part instead. |
| 245 | H→S, cos AP→cb | `[128,8,64]` BF16;128 | 1024 logical; **128 unique** | Same broadcast as196. |
| 247 | H→S, sin AP→sb | `[128,8,64]` BF16;128 | 1024 logical; **128 unique** | Same broadcast as198. |
| 289 | S→H, ob→dx | `[128,8,128]` BF16;128 | 2048 | 256 KiB contiguous output slice. |

For generic FP32 x/output paths, the dt-dependent entries double; count cos/sin
HBM bytes from their actual dtype, not merely the destination dtype. All
`tensor_copy`, reductions, reciprocal/sqrt and BF16 intermediate writes here
are on chip. They are not extra explicit HBM transfers. The direct `nki.jit`
call has no source-level `program_id` grid; do not invent a two-lane source
partition for this kernel from the global LNC2 setting.

## Softcap CE: all12 sites

Rows are128 contiguous logits rows selected by `nl.ds`; two explicit lanes at
production, `first=(tile*lanes+program_id)*128`, hence disjoint row ownership.
Each logits row has8192 entries. BF16 input/output is the ordinary current
head path; FP32 alternatives below remain supported. All CE arithmetic/loss,
labels, saved lse and incoming per-token gradient are FP32. `target_gather`
is true at456; `nc_n_gather`357 is **SBUF-local**, not an HBM gather. The CE
forward branch's equality mask is inactive and introduces no extra DMA sites.
No explicit DMA mask; MTP's last-position target mask is applied outside CE.

| Line | Direction / source→destination | SBUF tile; active partitions | HBM B/part | Logical tile / condition |
| --- | --- | --- | --- | --- |
| 346 | H→S, z→source | `[128,8192]` BF16;128 | 16384 | 2 MiB, active BF16 branch; widening347 is SBUF-only. |
| 349 | H→S, z→logits | `[128,8192]` FP32;128 | 32768 | 4 MiB, alternate FP32 branch. |
| 350 | H→S, tgt→labels | `[128,1]` FP32;128 | **4** | 512 B contiguous scalar vector. |
| 377 | S→H, log_denominator→loss | `[128,1]` FP32;128 | **4** | 512 B contiguous output vector. |
| 378 | S→H, logsumexp→lse | `[128,1]` FP32;128 | **4** | 512 B, distinct allocation from loss. |
| 418 | H→S, z→source | `[128,8192]` BF16;128 | 16384 | 2 MiB, active BF16 branch;419 widening is local. |
| 421 | H→S, z→logits | `[128,8192]` FP32;128 | 32768 | 4 MiB, alternate FP32 branch. |
| 422 | H→S, tgt→labels | `[128,1]` FP32;128 | **4** | 512 B. |
| 423 | H→S, g→gradient | `[128,1]` FP32;128 | **4** | 512 B; wrapper's463 `.float().contiguous()` occurs outside kernel. |
| 424 | H→S, lse→negative_lse | `[128,1]` FP32;128 | **4** | 512 B; sign inversion425 is local. |
| 437 | S→H, rounded→dz | `[128,8192]` BF16;128 | 16384 | 2 MiB, BF16 input or enabled generic round flag. |
| 439 | S→H, capped_unit→dz | `[128,8192]` FP32;128 | 32768 | 4 MiB, FP32 input and round flag=false. |

The current adapter leaves `round_bf16_output` atfalse; it is not an active
head-only override. For BF16 z,437 still applies because output dtype equals
z.dtype. Vocab chunk512 at337/370 is an on-chip exp/reduction loop, **not**
512-element HBM loads. Mean/none loss reduction and incoming-gradient
materialization are framework work outside this explicit source inventory.

## Two-tap convolution: all14 sites

Active x/grad/wt are BF16, C1024; the adapter581 packs/casts weights to
contiguous `[2,C]`, xx is contiguous, and backward makes grad contiguous.
The kernel permits FP32 inputs: replace2 by4 in corresponding payloads.
Products/accumulations are FP32, output rounded to x.dtype; partial dW is
FP32. `wt` loads broadcast the same C elements into all128 partitions using
`[[0,128],[1,C]]`, so the whole call has only C unique source elements.

| Line | Direction / source→destination | SBUF tile; active partitions | HBM B/active part | Logical tile / condition |
| --- | --- | --- | --- | --- |
| 472 | H→S, wt tap0→prev_w | `[128,1024]` BF16;128 | 2048 logical | 256 KiB destination, only2 KiB unique source across all partitions; once per kernel instance, reused across its batches. |
| 473 | H→S, wt tap1→curr_w | `[128,1024]` BF16;128 | 2048 logical | Same broadcast, offsetC; both forward and backward call this helper. |
| 479 | H→S, x current→curr | `[128,1024]` BF16;128 | 2048 | 256 KiB. |
| 482 | H→S, x[0:127]→prev[1:128] | `[127,1024]` BF16;127 | 2048 | 254 KiB, first tile only; partition0 is zeroed locally. |
| 484 | H→S, x[start-1:start+127]→prev | `[128,1024]` BF16;128 | 2048 | 256 KiB, other tiles; one-token shift within same batch. |
| 491 | S→H, result→out | `[128,1024]` BF16;128 | 2048 | 256 KiB. |
| 518 | S→H, result→partials[b,t,tap,c:c+128,:] | `[128,1]` FP32;128 | **4** | 512 B;8 channel blocks per tap,2 taps per time tile. |
| 526 | H→S, x current→xc | `[128,1024]` BF16;128 | 2048 | 256 KiB. |
| 527 | H→S, grad current→gc | `[128,1024]` BF16;128 | 2048 | 256 KiB; use grad.dtype for generic payload. |
| 530 | H→S, x[0:127]→xp[1:128] | `[127,1024]` BF16;127 | 2048 | 254 KiB, first tile only; xp partition0 zeroed. |
| 532 | H→S, x shifted previous→xp | `[128,1024]` BF16;128 | 2048 | 256 KiB, non-first tiles. |
| 535 | H→S, grad[start+1:length]→gn[0:127] | `[127,1024]` BF16;127 | 2048 | 254 KiB, last tile; gn memset zero preserves final-position boundary. |
| 537 | H→S, grad[start+1:start+129]→gn | `[128,1024]` BF16;128 | 2048 | 256 KiB, non-last tiles; same-batch lookahead only. |
| 544 | S→H, result→dx | `[128,1024]` BF16;128 | 2048 | 256 KiB. |

First/last decisions are static tile boundaries, not data masks. No transfer
wraps into the next batch. At518 the partition dimension now denotes channels,
not token rows: `nc_matmul`516 reduces128 token products, then517 copies PSUM
to SBUF. Both are on chip. `partials` is a real returned output, not hidden
scratch. Framework593 subsequently sums batch/time dimensions and rounds to
BF16, then594 casts to FP32 master weight dtype; those operations cannot be
counted as explicit source DMA here and must remain unchanged.

Source invocation arithmetic for518:32 batches ×8 time tiles ×2 taps ×8
channel blocks =4096 stores per conv backward, across its two lanes; five
conv layers give20480 source store invocations/rank/microstep. This is **not**
a measured descriptor/transfer count or proof the compiler leaves them split.

## Batched equivalents worth considering — no implementation or speed claim

1. **Conv dW stores, most concrete candidate.** Keep every existing FP32
   product and128-token `nc_matmul` reduction unchanged. Repack each finished
   `[128,1]` result on chip into a `[1,128]` bit-preserving transpose, collect
   all eight channel blocks into a contiguous `[1,1024]` FP32 staging row,
   and issue one4 KiB HBM store to the same complete `partials[b,t,tap,:,0]`
   vector. Alternatively two `[512]` rows give2 KiB per active partition.
   This replaces eight512 B source stores with one4 KiB source store per tap
   and leaves partial tensor shape, element order, reduction order and later
   BF16 rounding unchanged. Cost: eight small on-chip transposes/copies and
   fewer active DMA partitions; speed is unknown. Root must verify exact
   installed transpose/copy/AP legality. Do not fuse the batch/time dW sum or
   alter its FP32 accumulation order as an alleged transfer-only patch.
2. **RoPE cos/sin reuse, not extra precision or a new forward.** Each128 KiB
   table is reused across32 batches. Hoist each complete unchanged BF16 table
   once per kernel invocation and select each128-token window locally,
   broadcasting heads only in SBUF. A natural `[128,8,64]` time-block layout
   batches eight windows but has only1 KiB/partition and strided source blocks;
   it does **not** satisfy2 KiB/partition merely because total size is128 KiB.
   A contiguous `[32,2048]` staging layout gives4 KiB/partition but requires
   proven bit-preserving on-chip repartitioning before existing arithmetic.
   This is conditional layout work, not a free reshape. Both tables add
   256 KiB persistent SBUF. The compiler may already hoist repeated loads;
   profile/timing must establish any benefit. Keep every BF16 multiply/add
   boundary, table values and modulo-T address unchanged. Do not precompute
   differently rounded rotations or combine normalization reductions.
3. **CE scalar streams only: batch temporally, not entire logits.** Hoist a
   block of labels/g/lse before processing its successive128-row tiles, and
   buffer loss/lse scalars for a later grouped store to their original arrays.
   A `[128,K]` FP32 staging layout gives only `4*K` bytes/partition;2 KiB would require
   K512, i.e.65536 scalar rows, larger than this invocation's32768 rows.
   Simply enlarging that free axis or counting aggregate bytes is not a valid
   fix. Repacking512 consecutive scalars as `[1,512]` yields2 KiB/partition,
   but moving between that layout and the existing128-partition CE rows
   needs explicit on-chip repartitioning; fewer DMA partitions and transpose
   overhead may lose. Keeping two-lane ownership unchanged also creates gaps
   between each lane's alternating128-row blocks. A grouped AP must retain
   those gaps, not overwrite the other lane. Read-only loads could overfetch
   both lanes' scalars but duplicate traffic; write ownership cannot do so.
   Treat this as lower-confidence research until a legal cheap layout is shown.
   Separate loss/lse HBM allocations cannot be combined into one contiguous
   store without changing the interface/alias layout; do not assume adjacency.

CE memory limit: one `[128,8192]` FP32 array is4 MiB. Forward already declares
multiple such arrays; backward declares logits/indices/capped/temporary plus
dtype staging. Processing multiple128-row logits tiles simultaneously can
multiply that footprint beyond the28 MiB per-core SBUF ceiling. Keep the original
128-row compute loop and batch only small scalar metadata if pursued. No
"vectorize all CE rows" proposal: a full FP32 logits tensor is1 GiB/rank.
[AWS Trn2 SBUF capacity](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/nki/guides/architecture/trainium2_arch.html)

Current conv activations and RoPE x/output rows are **exactly**2 KiB/partition,
not below it; their duplicated causal-neighbor reads are a different reuse
hypothesis. Altering those stencils, combining batches, or making full MLP
forward/backward kernels is outside a scalar-transfer batching patch.

## Limits and handoff

The Round29 profile has incomplete DMA metadata and no framework attribution;
this source table does not repair it or satisfy the full-backward profile gate.
Root's SDK decision is separate; standalone imports are visibly `import nki`,
`nki.language`, `nki.isa` at175–177, not legacy `neuronxcc.nki`. No version or
hardware was queried here. Round29b supersedes the previous full-backward
integration request; this audit does not select or revive that build.

Any later transfer-only candidate must preserve source outputs, BF16 casts,
causal edges and all training/eval behavior, pass actual CPU equality and the
existing same-host native60 comparison. Round29b explicitly sets>=1.5% for
this math-identical batching arm; it does not authorize an SDK upgrade or
weaken unrelated full-backward gates. Root reviews/selects any change; this
report is source analysis, not another process gate or a completed proposal.
