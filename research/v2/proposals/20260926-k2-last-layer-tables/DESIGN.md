# P2: exact last-layer A1 tables on audited eight-layer + K2

Status: **FP32 implementation authorized off-host, 26 September 13:26 IST**.
The user explicitly decided FP32 table-gradient accumulation. Develop it in
parallel now; native/full execution remains conditional on K2's qualified
whole-step timing and the ordinary patch checks. One prospective package:
the fixed last-layer A1 layout with FP32 master tables/gradients/RMS state on
the eight-layer K2 stack. This deliberately changes historical BF16 table
arithmetic. Do not add earlier-layer tables, a larger table, a new gate
mechanism or a freeze schedule. No host launch follows this design alone.

The base is audited `e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43`, official **0.9878** reported by Poojith; its cold public20M is **0.9883890652263234**. K2 candidate `research/v2/lanes/k2-final-sprint/train.py` hashes to **`63925519ac6c0aab1a1d598e96fef3f4413fbd083d9b6d177901a2e972060db5`**. Its CPU checks and one review have passed according to the Supervisor; native timing is not established at design time. P0 adoption/stack choice remains the Supervisor's decision, not an extra factor selected here.

## Fixed A1 specification

Port the following subset from **`submissions/research_v2/train.py`**, SHA256 **`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`**. The executed A1 argv/result is in `research/v2/ledger.jsonl` under `fs-20260925-v2-c5c-a1-m64d128-trn21-r1-full`; explicit layout also appears in `research/v2/proposals/20260925-final-push-overnight/fs-20260925-fp-c5c-a1-crosshost-trn22-r1/EXPERIMENT_PROPOSAL.json`.

| Item | Exact proposed setting |
|---|---|
| Placement | Last layer only: absolute layer **7**, already an attention/unigram-VE layer in e1e2. Keep the existing unigram VE7. Add keys `b7` and `t7`. |
| Storage | Two zero-initialized **FP32 [524288,128] master Parameters**: `64 × padded_vocab8192` rows, width128. Gather in FP32, then cast gathered activations to BF16. No sparse parameter representation or hidden master copy. |
| Hashes | Use the unchanged layer-indexed A1 functions below, modulus **524287**, padding ID **8192**. |
| Projections | One bias-free **128→1024** `Linear` per table, FP32 parameters with the existing BF16 weight-cache path. Initialize uniform `[-s,s]`, `s=0.5*sqrt(3)/sqrt(1024)`. |
| Existing A1 gates | Separate zero-initialized bias-free **32→8** FP32 gate matrices. Bigram consumes normalized V-input channels32:64; trigram64:96. Each gate is `2*sigmoid(linear(...))`, initially1. These are the existing A1 injection gates, not a new gate experiment. |
| Injection | Preserve order: ordinary projected V + gated unigram VE, then gated projected bigram, then gated projected trigram; cast V to Q dtype before the unchanged RoPE/SDPA path. |
| Table optimizer | Bias-corrected RMSProp: beta1=0, beta2=.999, epsilon1e-10, weight decay0. **FP32 master parameter/gradient/second moment and FP32 RMS arithmetic**. FP32 across repeated rows, accumulation microsteps, distributed AVG and optimizer input. This is the explicit precision amendment, not a separate sweep. |
| Table LR | `.3*(1024/768)**(-.5)` = **0.2598076211353316**, flat after warmup; warmup0 makes the multiplier1. Set `no_lr_decay=True`. |
| Projection/gate optimizer | Existing Muon matrix groups and schedule, not the RMSProp table group. Include projections in cached-Linear refresh. |
| Reduction/clip | AVG-reduce all accumulated gradients. Clip ordinary parameters/projections/gates together at norm1; clip **both tables jointly in a separate group** at norm1. |
| Ownership | Keep table parameters/state replicated. Exclude RMSProp groups from owner partitioning and parameter all-gather; update that full group once on every rank after AVG/clip. Preserve existing FP32 owned AdamW/Muon behavior. |

Source anchors in c5c: hash helpers331–394; gates1668–1670/1705–1710; construction1914–1929; initialization2004–2014; lookup2109–2114; optimizer2163–2195/2757–2855; clip3026–3046/3088–3093; ownership4528–4529/4593–4611. The retrieved A1 `research/v2/operator/20260925-trn21-a1-full-final/training.log:66` confirms BF16 state/math in the actual run.

Hash contract, computed independently for each input sequence:

```text
P[k] = smallest_prime_at_least(8192 + 8000*k), k=0..29
bigram primes at layer7 = (P[14],P[15]) = (120193,128201)
trigram primes at layer7 = (P[17],P[18],P[19]) = (144203,152197,160201)
cur = idx.int32()
prev  = [8192, cur[:-1]]
prev2 = [8192, 8192, cur[:-2]]
b7 = remainder((prev*120193) XOR (cur*128201), 524287)
t7 = remainder((prev2*144203) XOR (prev*152197) XOR (cur*160201), 524287)
```

Preserve int32 multiply/XOR/remainder semantics, then convert the final nonnegative row IDs to int64 for K2 metadata. Layer7 uses layer7's multipliers, rather than retaining historical six-layer `b5/t5` multipliers. There is **no BOS/document reset** in A1; padding is the sequence-boundary sentinel, not the tokenizer BOS token. Handle length1/2 safely while matching the formula at ordinary lengths. No tokenizer/data changes or cross-call state are needed.

## K2 extension is required, not a flag-only port

Current K2 lines88–101 fix **V8192/T32768/C1024**, two512-channel owners, FP32 dense output. Its registered lookup at lines193–216 rejects non-FP32 weights and accepts only BF16 tangents. It **cannot** directly service A1's V524288/C128/BF16 parameters. Do not weaken those existing WTE/VE contracts or pad a 128-wide table to1024.

Add one separate registered table lookup/kernel specialization, ordinary forward
**`F.embedding(fp32_master_table, rows).to(BF16)`**, FP32 table-gradient output,
and stable token-order FP32 accumulation. Never cast the whole table to BF16
before gather; that would reintroduce BF16 collision accumulation in backward.
Keep current token WTE/VE lookup unchanged. The existing metadata producer
accepts a vocab argument; reuse its clone/stable-sort/searchsorted algorithm
with vocab524288 separately for b7 and t7. Their hash IDs differ and cannot
share WTE/VE metadata. Fresh snapshots per forward/microstep, no global cache.

Suggested fixed geometry is128 rows ×64 features per owner, two symmetric feature owners. There are now4096 row groups, so do **not** mechanically expand the existing64 static group bodies into4096. Use a bounded reusable body with runtime traversal of row groups and occurrence loops; installed SDK support for those dynamic offsets is a native unknown. Preserve disjoint output ownership, zero stores for unused rows, and original token order within every row. No atomics, touched-row optimizer approximation or reassociated sum is part of this proposal.

**The FP32 contract is decided.** Current K2's FP32 accumulation principle is
the required reference for the new geometry. Compare against ordinary Torch
embedding on FP32 master Parameters, followed by BF16 activation conversion.
The archived BF16 cancellation witness is historical evidence explaining this
choice, NOT a gate requiring preservation of BF16 rounding. Confirm FP32
Parameter.grad and optimizer state, including separate microsteps. Returning
FP32 from a backward attached to BF16 Parameters is insufficient because
autograd can cast the stored gradient back to the Parameter dtype.

## Clean integration and evaluation

Use the existing default-false `--ngram-ve` switch as a **fixed A1 preset**; remove only its current “unsupported” rejection. Do not expose an A2/wider-table sweep. Persist one default-false enable field in `GPTConfig` (the remaining preset is fixed source), so `save_checkpoint`'s existing `asdict` and `load_for_eval` reconstruct the same modules without training CLI/global-state assumptions. Old e1e2 checkpoints remain loadable through that false default. Preserve strict state loading.

In K2 source, extend GPT construction/init/setup/forward at lines927–1075; pass the two optional VE contributions through Block to layer7's attention. Add only the A1 RMSProp methods, group routing, separate clipping and RMS ownership bypass missing at lines1007–1047,1577–1618,1697–1779. Use RNG-neutral gate/projection constructors and initialize new projections after existing randomized tensors, preserving common-parameter initialization. With tables off, construct no new parameters and consume no new RNG.

Evaluation must perform the **same hashes, BF16 table lookup, projections and gates** with ordinary Torch. K2 stays training/grad-enabled only; `model.eval()`/`requires_grad_(False)` avoids native imports or metadata work. Longer evaluation contexts and different batch sizes must work through ordinary lookup. Each output position depends only on current/past tokens in its own sequence; confirm prefix causality, independent-row batching, repeated-call purity and strict checkpoint reload in the same small CPU suite. Do not change `prepare.py`, the evaluator, data or scorer.

## Minimal validation and the cheapest falsifier

1. **CPU FP32 exactness, first:** exercise duplicate rows with BF16 tangents
   and FP32 master tables/accumulators. Include `[1,2^-8,-1]`, cancellation,
   zero/unused rows, interleaved duplicates, both tables and two microsteps.
   Compare the proposed producer/consumer to
   `F.embedding(fp32_master_table, rows).to(BF16)` and its FP32 gradients;
   include hash boundaries/collisions and FP32 RMSProp updates. Tiny fixtures
   first, no full-table CPU allocation. Historical BF16 equivalence is not required.
2. **Three OFF steps:** new source with tables off versus exact qualified K2 parent, same seed/input/fixed clocks, byte-equal losses, gradients and updates. Also compare a small tables-ON model/update against a plain-Torch A1 reference, including one replicated RMS update per rank and the separate clip groups. Use width128 or greater for this model test: the original K2 width64 fixture cannot represent the trigram gate's channels64:96. One independent reviewer, ≤30min, covering this single patch; no additional review unless a test fails.
3. **One short native timing check:** sole Operator, ordinary unique-label proposal/preflight/lifetime lock; four ranks/LNC2, pack32/context1024/width1024, seed42, 45 complete steps (five startup, five settling,35 steady), with tables and K2 on. Compare against the same-host qualified K2 control; obtain a flag-only OFF control if a comparable one is unavailable. Report cold fifth boundary, complete-step tok/s, finite/nonzero table updates and terminal/checkpoint status. Include both metadata producers, table backward, dense AVG, clipping, replicated RMS and cache refresh. No consumer-only speed claim or extra route matrix. Table addition itself is **not** required to gain3%: that bar unlocks K2; P2's final judgment is full-budget fresh public20M on the resulting stack under unchanged `decide.py` rules.

## Unknowns and cost limits

K2 native qualification remains the execution prerequisite. Unknowns are
table-geometry dynamic-loop/indirect-DMA lowering, metadata cost for524288
search rows per table, dense FP32 collective/RMS cost, and cold compile time.
Two tables add134,217,728 parameters: **512MiB** FP32 weights, 512MiB dense
gradients, and 512MiB second moment—**1536MiB/rank** before temporaries,
gathered BF16 activations and projections. Checkpoint growth is at least the
512MiB master weights if optimizer state is not saved. These are size estimates,
not measured peak memory, disk or native timing. Do not infer resource admission.

Historical A1 scored **0.9986312057378084** versus0.9914739939934344, delta+0.007157211744374,3051 steps. It was a valid fixed-budget loss. Its attribution to slow scatter is a hypothesis motivating this new stack, not a measured causal result. No new BPB estimate is presented as a measurement. After a successful short check, the Supervisor can issue the one full1800s P2 run against its current same-host K2-stack control; implementation and timing do not imply adoption or submission readiness.

Authority: `workflow/FINAL_SPRINT_GOAL_2026-09-26_1326_IST.md` and the user's
explicit FP32 amendment. The earlier BF16 design is retained in git history.
Supervisor assigned a separate Implementer; no native result or adoption follows.
