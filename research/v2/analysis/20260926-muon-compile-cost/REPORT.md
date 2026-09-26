# Accepted e1e2 Muon compile-cost analysis — 26 September 2026

**Finding:** the exact accepted base's cold replay reports **59.817 seconds**
across eight unique 194-node inference graphs consistent with Muon. This is
approximately the human's 60-second investigation trigger; it does not establish
more than 60 seconds, or 59.817 seconds of removable startup. The same-source
trn21 control with reused research caches reports 43.077 seconds. No candidate
implementation or host action was performed by this lane.

## Measured counters and attribution

The cold source is
`workflow/candidates/confirmed-clock36-116188/train.py`, SHA-256
`e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43`.
Cold raw bundle:
`research/v2/operator/20260926-trn22-eval-boundary-0557/completed/results-bundle.json`,
SHA-256 `20e1e5f920dff44a4d0a047790e02397ed916945e975a6822d6385efe2f9151d`.
Each CSV is embedded at `files["metrics-training/metric_table_graph_stats_rankN.csv"].text`.
The exact remote run root is
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet`.

| Inferred matrix shape | Owner-local batch K / graph prefix / counter rank / seconds | Shape total |
|---|---|---:|
| 1024×1024 attention | K6 / `561e61ab` / r2 / 9.588; K3 / `aee6ba1f` / r1 / 5.787; K5 / `acae4190` / r3 / 8.254 | 23.629 s |
| 4096×1024 `c_fc` | K2 / `50cab471` / r3 / 6.680; K3 / `6a741cc4` / r1 / 6.012 | 12.692 s |
| 1024×4096 `c_proj` | K2 / `ea28caef` / r2 / 6.101; K4 / `c778b3a6` / r3 / 12.592 | 18.693 s |
| 8×32 `ve_gate` | K2 / `3d178ec2` / r1 / 4.803 | 4.803 s |
| **All four shapes** | **8 unique graphs; 13 rank occurrences** | **59.817 s** |

All eight graphs have exactly one nonzero `torch_neuronx_compile_us` entry.
The other rank occurrences are not summed twice. Full IDs, per-CSV hashes,
record timestamps and the inferred ownership map are in `result.json`.

Counter arithmetic/source identity confidence is high. Per-shape attribution
is strong **inference**, not direct IR observation: the pinned source groups
Muons in sorted shape order, calls only owned groups in that order, pads
singleton groups to K2, and keys compilation on the complete `[K,M,N]` shape,
orientation, reduction dimension and NS iterations. Exact source `make_plan`
executed on shape-only metadata gives the following call sequences. Matching
graph IDs across ranks receive the same inferred signature. The rank0 log
explicitly prints its three corresponding Muon signatures.

| Rank | 8×32 | 1024×1024 | 1024×4096 | 4096×1024 |
|---|---:|---:|---:|---:|
| 0 | absent | 6 | 2 | 2 (1 real) |
| 1 | 2 (1 real) | 3 | 2 (1 real) | 3 |
| 2 | absent | 6 | 2 (1 real) | 2 |
| 3 | absent | 5 | 4 | 2 |

The same accepted e1e2 control on trn21 gives square14.715 s,
`c_fc`12.160 s, `c_proj`12.653 s and gate3.549 s, total43.077 s. Its raw bundle
is `research/v2/operator/20260926-final-sprint-successors/trn21-e1e2-control-completed/results-bundle.json`,
SHA-256 `9289bf8a94f164dccedbc1898408e7b8a02a4e3ed7cb927a44ad501b203e22d9`;
the copy read was in `/home/poojith-devan/Desktop/Trainium-trn22-queue-operator/`.
This control reused historical research cache roots. Neither lower counters
nor zero counters prove a persistent cache hit.

## Counter sum is not concurrent elapsed time

The 59.817 s is a sum of reported compiler-duration counters over distinct
graphs. It is neither measured CPU-seconds of compiler work nor a union of
compiler start/end intervals. Multiple ranks invoke their owned groups, and
overlap, waits, lowering, tracing and cache coordination can affect elapsed
startup. The raw CSV timestamp is a graph-record timestamp; its exact
relationship to compiler start/end has not been authenticated here. Therefore
**Muon elapsed compile time and compiler CPU work remain unknown**.

The whole cold replay's actual fifth-step startup is859.473523823 s, measured
independently by its harness; charged startup excess is0 s. That number is
not used to infer any per-graph duration. The fresh empty SDK caches are
recorded by that run. Its model forward/backward compile counters are
204.994/479.732 s, distinct from Muon's counters.

The sole Operator's bounded follow-up at10:20:51UTC found the exact completed
cold run's `train/caches/debug/neuronx-cc` directory **empty**. HLO root had
more than the128-entry bound and was refused without expanding the search;
the NEFF root had86 two-hex shard directories and no direct graph files.
No individual Muon IR or compiler start/end log was recovered. Receipt:
`research/v2/operator/20260926-k2-native201-handoff/base-cold-muon-cache-headers.json`,
SHA-256 `49080ddbdcb2354496b745d4a3fdf77b53b2ed40570aab18ac6eeac34f1a96e0`,
Operator commit `1526d0ccf`. Missing timings stay unknown. No further broad
cache scan or remote execution was performed by this lane.

## Transpose and whole-update constraints

Exact source anchors: `setup_optimizer` lines866–906, `_muon_math`950–981,
`_muon_compile_scalars`1059–1100, `_muon_step`1155–1210,
`make_plan`1540–1568 and `OwnedOptimizer`1571–1631.

1. The optimizer retains a `transposed` group option, but accepted e1e2
   `setup_optimizer` sets it False for every matrix. The option changes
   logical aspect ratio and reduction-axis selection; it does not transpose
   parameters, gradients or state. Clean `Linear.forward` uses `F.linear`
   directly, and the custom MLP backward consumes native `[out,in]` weights.
   The old research file's general transposed-linear facility is absent from
   this accepted clean caller. Merely flipping the group option is invalid.

2. For rectangular c_proj, transpose `w`, gradient and first-moment state;
   transpose its second-moment state from `[K,1,4096]` to `[K,4096,1]`.
   Its physical shape then matches c_fc `[K,4096,1024]`, with `tall=True`,
   `red_dim=-1`, reduction size1024. The NS polynomial is transpose-equivariant
   in exact arithmetic. The full update additionally requires matching
   momentum interpolation, variance normalization, BF16 casts, zero/None
   gradient behavior and sign-masked decay.

3. Preserve **logical** scale: c_fc uses
   `sqrt(max(1,4096/1024))=2`; c_proj uses1 even after optimizer-scratch
   transposition. Both LR and LR×weight-decay carry that ratio. A mixed stack
   therefore needs per-matrix LR/decay tensors, or another exact split of its
   update. The current group's one boolean and scalar ratio cannot describe
   both. FP32 staging order and the scalar-versus-dimensional LR promotion
   in `_muon_lr_g` must remain explicit. The clean FP32 branch uses
   `lr_fp32=False`; this investigation does not change that precision policy.

4. Keeping actual model Parameters in native layout and transposing only
   optimizer scratch is a possible design, but requires implemented copy-back,
   momentum/variance and ownership mapping. Physical Parameter transposition
   instead affects model forward, custom backward, cached weights and saved
   state shapes, so it has a larger caller surface. Neither was implemented.

5. Two matrix dimensions do **not** imply two compiled graphs. K is in the
   key. With current ownership held fixed, sharing transposed c_proj with
   c_fc as separate groups would reduce eight signatures to seven: their
   common K2 can share a physical graph, while K3 and K4 remain distinct.
   Truly merging each owner's real c_fc/c_proj rows yields K3/K4/K3/K6,
   before any new ownership/padding design, and still leaves three square
   K values plus the gate. Changing group order, planner costs or gate kind
   can change ownership and must be recomputed, not assumed.

6. Moving the single256-element ve_gate to AdamW is a scientific optimizer
   change. It removes its dedicated Muon shape but changes update/state,
   LR, betas and decay policy and may change ownership. It is never an
   equivalence claim; keep this causal factor separately identifiable in any
   selected proposal and full-budget result. Its observed cold counter is
   only4.803 s, with removable wall time unknown.

## Bounded CPU feasibility experiment

After the Supervisor explicitly assigned the sole local-heavy slot,
`transpose_feasibility.py` ran one single-thread CPU process under6GiB address
space,60CPU-second and90wall-second caps. It exited0 in1.04 script seconds;
the slot was then released. It extracts the two arithmetic functions from
pinned e1e2, never imports the full training module, and uses seed42.

Eight updates test `[2,8,32]` and `[2,32,8]` transposition and a prospective
merged `[4,32,8]` scratch stack with ratios `[2,2,1,1]`. Comparisons include
weights, first momentum and second moment, changing LR/momentum/decay,
zero-initialized output projection and zero gradients (including the source's
None-to-zero arithmetic behavior). It does not exercise a new wrapper's
handling of actual None objects or missing distributed gradients.

| Iterate dtype | Result over the small cases |
|---|---|
| BF16, with FP32 weights/moments | All72 state comparisons bit-identical |
| FP32 | All72 finite; not bit-identical; max absolute5.7742e-8, max relative L2 1.24131e-6 |

Raw comparisons are in `transpose-result.json`. These are observed residuals,
not a newly approved production tolerance. They support feasibility only:
no production1024/4096 matrix test, actual implementation, model OFF identity,
native compiler/whole-step timing or quality equivalence has passed here.
The proposed gate-to-AdamW change was not tested as an identity.

## Recommendation

Retain compile reduction as a bounded diagnostic lane, with the cold counter
total recorded as approximately60 s. Do not prioritize a production optimizer
change solely from this near-threshold sum. Exact elapsed savings remain
unmeasured, and the accepted base is already below the900 s startup allowance;
compile reduction alone gives it no extra net-token budget under the new
formula. It may provide useful startup margin for a larger candidate, but
that benefit needs actual candidate cold startup and steady rate.

If selected, first pursue an optimizer-scratch transpose design preserving
logical scales and existing model shapes. Keep gate-to-AdamW an explicit
separate scientific factor. Judge the implemented candidate with the agreed
CPU exactness/OFF identity/one review/one native timing bar, then the required
full official-budget quality run; do not reopen general K1 racing or claim
two graphs/performance from shape arithmetic. No new gate or launch authority
is created by this report.

Reproduction: `python3 analyze.py --warm-bundle <exact warm bundle>` from this
directory (or the repository-relative script path) authenticates both bundles
and source using only stdlib. Normal and `python3 -O` both produced59.817/
43.077 s. Numerical command was
`timeout 90s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python research/v2/analysis/20260926-muon-compile-cost/transpose_feasibility.py`.
