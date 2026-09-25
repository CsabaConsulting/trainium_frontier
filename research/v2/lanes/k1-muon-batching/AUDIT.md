# K1: batching and owner-compute are already on the adopted base

Audited source: `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`,
`workflow/proposals/research-v2/train.py`, SHA256
`6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.
Adopted launch uses Recursive schedule ON; this audit does not change it.

## Concrete call trace

- `GPT.setup_optimizer`, lines2148–2224: collects block matrices separately from
  embeddings/head/scalars; lines2211–2218 group by exact `(shape, transposed)`.
  Each shape group has one parameter list, not a per-layer optimizer call.
- `_muon_step`, lines2861–2921: persistent `[K,rows,cols]` gradient/weight stacks,
  batched foreach staging and one compiled `_muon_math` invocation per owned
  group. `_MUON_COMPILED` keys include full batch shape, orientation, reduction
  dimension and iteration count. Compile-Muon is already default ON (line2298).
- `_muon_math`, lines2302–2373: BF16 iterate with independent per-matrix norm;
  `x.mT @ x` or `x @ x.mT` and polynomial matrix products operate on rank3
  tensors. The loop is over five NS iterations, not individual matrices.
  This is already the requested batched-matmul form, subject to actual backend
  lowering/efficiency that P must measure.
- `_muon_compile_scalars`, lines2586–2642: all five changing scalar inputs share
  one host-to-device staging copy per group; no scalar batching gap is presumed.
- `make_plan`, lines4558–4584: deterministic cost-balanced whole-parameter
  ownership. `OwnedOptimizer`, lines4586–4650, builds rank-local subsets,
  executes only those updates, packs once, calls `all_gather_into_tensor` once,
  then copies full FP32 parameters back. This is whole-optimizer ownership,
  not merely NS sharding with extra per-shape collectives.
- Training installation, lines4878–4885: exact world4 required; ownership is
  installed unconditionally with `pad_singleton_muon=True`. `_muon_step` in
  isolation would miss this wrapper and incorrectly suggest redundant work.

## Exact current owner inventory, no model allocation

`audit_plan.py` executes only the exact source's `make_plan` AST against dtype,
shape, element-count and contiguous metadata. It allocates no torch tensors.
The reconstructed plan hash is
`b89d62227c844eb13706fbe81bddfb9b8b7db5fefcd57b4e7d0ea3213dca5031`, exactly
matching all four raw rank receipts and aggregated full-size6ef7 G2 in
`research/v2/operator/20260925-trn21-recursive116188-g2-r1/reference-on/results/`.
Those historical CPU receipts exercise three updates on117,453,580 parameters,
with synthetic2x256 inputs and other explicit G2 shims; not native performance.

| Owned matrix shape | rank0 | rank1 | rank2 | rank3 |
| --- | ---: | ---: | ---: | ---: |
| 8x32 VE gate | 3 | 0 | 0 | 0 |
| 1024x1024 attention | 6 | 5 | 5 | 8 |
| 1024x4096 MLP output | 0 | 2 | 2 | 2 |
| 4096x1024 MLP input | 1 (+1 dummy) | 2 | 2 | 1 (+1 dummy) |

All62 live parameters are assigned exactly once. The real model has39 Muon
matrices; native Muon processes41 rows across owners because of two padding
rows. Rank costs estimated by old profile constants are12.9474/12.9397/
12.9488/12.9477ms; these are planner weights, NOT current measured times.
FP32 elements by owner:27,265,796/30,414,851/30,412,805/29,360,128.
The single padded gather reconstructs486,639,616bytes per rank versus
469,814,320bytes of live parameters. Do not infer >=5% whole-step savings from
either dummy count or these memory volumes.

## Prior trials that materially constrain a new proposal

1. Existing foreach optimizer batching was already adopted. Historical
   `research/trn2-experiment-results.md:1131` reports23.1ms optimizer reduction
   and paired same-day+3.3% fixed-budget steps on the old stack. This is not
   an incremental gain still available to exact6ef7.
2. T1.7 already compiled the whole batched Muon math body; the same results
   file at5314 records CPU bit-exact rewrite, native BF16-scale drift up to
   0.003254maxabs, and historical+3.38% whole-step throughput on a different
   table-heavy model. A reported8.18x isolated-body improvement became3.38%
   whole-step; it cannot be reused as a new6ef7 forecast.
3. Original owner-compute integration is documented in
   `c18305ed:research/fresh-start-optimizer-ownership-2026-09-18.md` and
   `4a07e5b7:research/fresh-start-optimizer-ownership-2026-09-18.md`.
   Unpadded native singleton tall Muon groups on ranks0/3 changed weights and
   second moments. Batch2 matched; adding zero dummy rows repaired strict
   equality. Wave62's raw summary at
   `4a07e5b7:research/run-logs/fs-w62-optimizer-ownership-padded-neuron-gate-r1.summary.json`
   records28 exactness observations, zero dummy state, cache/None/zero/skip/
   checkpoint checks, and25.35–25.38ms owned optimizer versus52.29–52.84ms
   replicated. That is optimizer-only, not a new whole-model result.
4. The later same-source matched full-step ownership result is recorded in
   `785c294e:research/fresh-start-optimizer-ownership-2026-09-18.md`:
   554.8731335→526.7829345ms,5.06245%, atT2048/pack8/GA2. Ownership from that
   campaign is already present in6ef7. Its subsequent ledger quality row uses
   an old evaluation prefix/context; do not turn that into current20M evidence.
5. The separate20260922 NS-only splitting trial on the old592.9M `9a4caa7b`
   vehicle failed:325steps versus335,1.1432 versus1.1775steps/s (~2.9% slower),
   smoke BPB1.173345 versus1.165604. See
   `research/agentic/experiments/20260922-muon-owner-ns-split-r4/smoke-verdict.md`
   and its raw archive. It split the compiled body into pre/NS/post and added
   per-shape all-gathers; current whole-optimizer ownership is a different
   implementation. It is evidence against blindly layering the split again,
   not proof all owner-planner changes will fail.

## Smallest informative next step

P should measure the current6ef7 steady critical-rank optimizer window:
per-shape batched NS, stack input/output copies, AdamW and the single gather,
including actual execution and inter-launch gaps. Compare8x1024 separately;
do not use old planner constants as measured elapsed or zero compile_us as
cache-hit evidence. If remaining optimizer cost is below5% of the complete
step, optimizer-only K1 cannot meet the required5% total-step reduction even
with a zero-cost replacement (Amdahl upper bound).

One narrower *unselected* hypothesis is pairing whole tall matrices during
ownership planning so their6real matrices form three batch2groups instead of
four batch2groups with two dummies. It must preserve all gradients, live
Parameter identities/FP32 masters, missing-gradient semantics, schedule and
per-matrix moment state, and may worsen load balance/padded gather. Another
possibility is reducing stack/copy overhead if P identifies it, but persistent
view/alias redesign is materially broader and not inferred as authorized.
Do not remove singleton padding, change state precision, split an individual
matrix, change LR/schedule/NS iterations, or add a new collective under the
name of missing batching.

No production candidate was implemented from this audit. The exact existing
mechanism was found, so a no-op flag would waste verification and host slots.
New source requires a distinct selected factor and the full newest gates:
default OFF, CPU G2 ON,3step OFF identity, CPU batched/per-matrix numerical
comparison including state/None/zero, adversarial different-family review,
then >=5% measured whole-step improvement before any full quality run.

Local checks: `python3 research/v2/lanes/k1-muon-batching/audit_plan.py` and the
same command with `python3 -O` both exited0 and produced the exact archived
plan hash. These are stdlib metadata checks, not CPU optimizer equivalence
tests. `git diff --check` passed; production source stayed byte-identical.
