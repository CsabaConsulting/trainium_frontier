# Fresh-start rules, hardware evidence, and research priorities — 2026-09-18

This is a source audit and experiment proposal, not a record of new training results. It uses the live Devpost pages, the local organizer-derived `README.md` and immutable `prepare.py`, official Neuron documentation, and the repository's prior experiments. The organizer's GitHub URL returned 404 to anonymous extraction; its currently authenticated remote contents were therefore not verified. The local fork is the available technical contract.

## 1. What is established about the competition

The objective is **minimize** validation bits per byte, not maximize it. Phase 1 uses one complete Trainium2 chip for 30 minutes. The formal rules describe a pinned validation shard of approximately 20 million tokens and exclude startup/compilation. The top ten advance. The Devpost rules introduction still refers to an unrelated Qwen3 MoE inference task; its explicit Round 1 criteria and the landing page instead agree on language-model training. Treat the stale introduction as an editorial inconsistency, not a different task. Sources: [Devpost landing page](https://trainium-frontier.devpost.com/), [formal rules](https://trainium-frontier.devpost.com/rules).

The operational timing mechanism is particularly important. In local `prepare.py:38–43`, the constants are:

| Item | Value |
| --- | --- |
| Charged training budget | 1,800 seconds |
| Startup steps excluded | First five completed optimizer steps |
| Maximum startup allowance | 900 seconds |
| Evaluation timeout | 1,200 seconds |

`record_step_boundary` in `prepare.py:101` must be called immediately after each real optimizer step and synchronization. The official runner records the arrival time of each boundary itself; participant-written timestamps are not the clock. A run without five boundaries is not scorable. Local README language that compilation counts is broader than the formal rules and this explicit mechanism: do not interpret it as permission to hide arbitrary later compilation, preparation, training, or checkpoint work. Use the real fifth-boundary protocol, fresh initialization, and a safety margin for checkpoint writing. Any deliberate delay before that boundary to accumulate extra training would violate the purpose of the budget.

The authoritative evaluation is local `prepare.py:evaluate_bpb`, starting at line 596. It calls `model(x)`, slices logits to the fixed vocabulary, converts logits to FP32, and recomputes cross entropy independently. It never trusts `model(x,y)` loss. `load_for_eval` must return a consolidated full model on one device, with a frozen, pure, stateless, causal forward. Outputs must not depend on other rows in a batch, rank, world size, or prior evaluation calls. Parallelism belongs in training, and the saved artifact must already be consolidated. Source: local `README.md:181–246`, linked upstream location [organizer repository](https://github.com/aws-neuron/trainium-frontier).

### Allowed, forbidden, and ambiguous

These distinctions follow the local README's explicit permission/prohibition sections, not speculation about what an automated reviewer might accept.

| Direction | Rule status and usable interpretation |
| --- | --- |
| Architecture, width/depth, optimizer, schedule, batch size, training loss | Explicitly allowed within `train.py`; no parameter cap |
| NKI attention, matmul, normalization, optimizer, loss | Explicitly allowed; custom kernels are encouraged, not mandatory |
| Precision | No BF16-only restriction stated; numerical correctness and causal deterministic evaluation still required |
| Data sampling | Explicitly allowed |
| Change `prepare.py`, tokenizer, data pipeline, scoring code | Explicitly forbidden |
| More ClimbMix shards beyond provided training assets | Not allowed merely because their URLs are public; they would be external training data |
| Pretrained weights/embeddings or precomputed learned retrieval tables | Forbidden; learned assets must be produced by this submitted training run on allowed data |
| Packages beyond the provided environment | Explicitly forbidden for submissions; research must not assume an upgrade is deployable |
| Train on or adapt to scoring/private validation data | Forbidden |
| Cross-run checkpoint averaging or ensembles of separately budgeted runs | Not a permitted way to inherit extra training; do not use |
| EMA/SWA/snapshot ensemble produced entirely inside one budget | No explicit ban found; architecture freedom suggests plausibility, but the final causal/frozen contract, total training time, saved full model, and evaluation timeout still apply. This is an inference, not organizer confirmation |
| Replace trusted loader with a new packer/streaming implementation | Ambiguous: data sampling is allowed but data-pipeline modification is forbidden. Do not label it unconditionally compliant just because `prepare.py` stays unchanged |

A conservative route for data experiments is to consume allowed training batches from the trusted loader and sample/reorder those batches. A read-only CPU coverage audit can investigate the original loader without altering any submission component. Epoch row-group reordering that preserves packing exactly is more defensible as sampling than a packing rewrite, but still deserves a precise documented rationale. No organizer was contacted in this research pass.

## 2. Corpus and metric facts that change experiment design

`prepare.py:52–57` fixes training shards to **0–4** and public validation to **6542** of `karpathy/climbmix-400b-shuffle`. The fixed BPE tokenizer has **8,192** entries and SHA256-checked artifacts. The full dataset's name does not authorize training on its remaining shards. There is no canonical token-count constant of 288M in `prepare.py`.

The earlier “288M-token data wall” was inferred from the printed epoch counter at approximately step 2,195 with 131,072 tokens per step. That counter increments while reading raw document batches, before buffered best-fit packing consumes all of them. Its meaning is not an exact count of unique trained tokens. `make_dataloader` also discards the remainder of a document when no buffered document fits the residual row space. These are observations of existing code, not proof that changing the packer would help or be legal. Source: `prepare.py:435–521`; historical inference and its supporting runs are in `research/trn2-experiment-results.md:4444`.

Consequently, report all three quantities separately: actual optimizer-token presentations, raw documents/tokenized tokens encountered, and estimated unique content presented. “One epoch” alone is not sufficient to conclude that the useful data is exhausted. The old notes do identify a serious confound: a short prefix of the same corpus is revisited during final learning-rate decay. Their stronger assertion that speed becomes worthless above 2,195 steps should remain a testable hypothesis. Same-time scoring is the competition objective; step-matched controls isolate mechanisms but cannot replace full-budget comparisons.

There is also a precise objective mismatch worth a cheap isolated control. Trusted evaluation uses:

```
numerator = sum(CE_i for valid targets whose token_bytes_i > 0)
denominator = log(2) * sum(token_bytes_i for valid targets)
```

This is **not** `sum(token_bytes_i * CE_i)`. For a fixed sampled dataset, the denominator is parameter-independent: unweighted token CE on positive-byte targets is already proportional to the BPB objective. Therefore the existing `--byte-loss-weight` rationale should not be interpreted as exact BPB alignment. A distinct, cheap hypothesis is to mask only BOS/reserved zero-byte targets in the training loss and normalize over scored tokens. This preserves the tokenizer and loader, and tests exactly the part of CE that the scorer excludes. No isolated zero-byte-mask result was found in the examined historical notes. Expected gain is unknown and may be negligible if these targets are rare or useful as an auxiliary task.

## 3. What “real FP8 on Trainium2” means

AWS documents native double-throughput FP8 on NeuronCore-v3. It is an explicit `nki.isa.nc_matmul` mode with FP32 accumulation. The contraction dimension is packed across `[128, 2, ...]`; the documented largest operation uses stationary `[128,2,128]`, moving `[128,2,512]`, and output `[128,512]`. `perf_mode=nisa.matmul_perf_mode.double_row` enables it. The ideal factor of two is for the relevant Tensor Engine matmul work, not the entire model. It does not double optimizer, communication, packing, launch, or scaling throughput. Source: [Trainium2 NKI architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html).

The exact format distinction is essential. Current official `nc_matmul` documentation supports legacy **`float8_e4m3` / `float8_e5m2`** for Trainium2 double-row mode. Its OCP **`float8_e4m3fn`** support begins at **NeuronCore-v4**, and cannot be mixed with legacy E4M3 in the same operation. MXFP8/MXFP4, `_x4` packed formats, and `nc_matmul_mx` concern the Trainium3 microscaling path. A page's generic “relevant for” badge is less specific than its architecture and operand restrictions. Source: [nc_matmul API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html), [MXFP tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.0/nki/deep-dives/mxfp-matmul.html).

Before writing a kernel, verify the **installed** SDK's types, function signature, allocation API, and native-PyTorch custom-autograd integration. New online documentation is not evidence that a newer package may be installed under competition rules. A safe first prototype accepts BF16/FP32 tensors, applies explicit scales and casts inside NKI to supported native FP8, performs double-row matmul, and returns BF16 with FP32 accumulation. Casting PyTorch tensors directly to OCP `torch.float8_e4m3fn` is not an adequate test of this path.

### What historical failures do and do not establish

The 09-12 precision note measured a disastrously slow Torch `_scaled_mm`/emulation route and incorrect direct FP8 `torch.matmul`. The 09-16/17 plan separately tested global compiler FP8 autocasting, with frozen learning, divergent scaled training, and failures across multiple static/dynamic loss scales. These close those implementations. They do **not** establish that an explicitly scaled, forward-only NKI MLP matmul fails. Conversely, the existence of hardware FP8 support does not establish an end-to-end speed or quality win.

A full “FP8 pipeline” is a poor first experiment: it entangles quantization of sensitive reductions, gradients, Muon orthogonalization, and forward weights. Keep masters, optimizer, norms, softmax, CE, and initially all backward matmuls in their proven precision. Start with one measured expensive forward MLP projection. Use BF16 controls with the same layout and dispatch boundaries so that launch changes are not mistaken for precision effects.

For an affected end-to-end fraction `f`, even an ideal doubling gives `speedup <= 1/(1 - f/2)` before scale, transpose, and dispatch overhead. A forward-only operation is only a fraction of the usual forward/input-gradient/weight-gradient matmul work. This is an analytical upper bound, not a measured fraction on either host.

## 4. Kernel roadmap grounded in the actual native stack

The current model calls `F.scaled_dot_product_attention` through the default branch of `train.py:nki_attention`, around line 2808. There is already a flag-gated custom NKI forward/backward pair, default off, so “implement NKI attention” is not a new experiment. The most recent recorded corrected-mask run (`sub1-p1-nkiattn-smoke2`) passed causality and tracked SDPA's loss but ran approximately 1.9 times slower: 1.61–1.62 seconds/step versus 0.85–0.86. Earlier apparent near-zero BPB was an upper-triangular-mask causality bug, not a quality result. Sources: `research/plan-sub1-2026-09-16.md:45–143`, existing `_attn_nki_fwd` / `_attn_nki_bwd` / `_AttnNKI` implementation.

The SDPA Python operator name does not identify the compiled backend implementation or prove that attention is materializing a quadratic matrix. Historical dispatch-bound eager measurements also do not prove that today's full-model compiled graph has the same bottleneck. Before another attention redesign, obtain the current compiled forward/backward operator or instruction profile at the record shape and account for host gaps, matmuls, transposes, activation materialization, embedding scatter, and optimizer time. A single-pass online-softmax redesign is a materially different kernel hypothesis; it still must beat the native baseline including integration boundaries and backward.

This research does **not** recommend inserting XLA `xm.mark_step()`, importing `torch_xla` into the native `torch.device('neuron')` path, or copying old XLA training flags. The active experiment reports identify a native PyTorch editable installation and separate unused XLA environments. Native controls, existing `torch.neuron.synchronize()`, installed compiler support, and explicit environment capture are the relevant interfaces. The existing protocol's requirement to include `--optlevel=1` is supported by the documented 09-17 slowdown incident.

Prioritize a kernel by demonstrated removable cost:

1. **Embedding gather/backward and optimizer traffic for n-gram tables.** The current model has hundreds of millions of table parameters despite a much smaller dense transformer. If profiling identifies repeated dense zeroing, scatter, whole-table AdamW, or excessive indirect DMA, reducing that traffic may dominate attention or FP8. Preserve exact update semantics first; sparse updates with weight decay/momentum require careful treatment of unvisited rows.
2. **Fused MLP forward/backward or residual-normalization boundaries.** Useful when measured HBM intermediates or launch boundaries dominate. Benchmark the complete differentiated operation, not only an attractive forward kernel. Layout can matter more than changing dtype.
3. **Explicit scaled FP8 forward matmul.** Run only after an isolated correctly tiled kernel beats the same-shape native BF16 operation after casts/scales are included, and after gradients/short learning trajectories pass. Extend to backward only with a separately calibrated design.
4. **Attention replacement.** Run only if the installed SDPA forward/backward demonstrably leaves a substantial gap. A training kernel needs both directions, exact scaling/QK-normalization/RoPE compatibility, causal masking, and batch-independent evaluation.

Each is a hypothesis ranked by expected information value. None is a claim that rewriting everything in NKI is necessary. The unifying test is full-budget BPB after recovered time is spent productively.

## 5. Practical first experimental slate

The two machines should have distinct roles while retaining one active device job per machine. One produces same-seed, full-budget quality controls; the other runs short profiling/correctness/kernel gates and promotes only concrete candidates. Persist launch text, compiler environment, script checksum, package versions, host identity, timing, and results locally between runs; synchronize completed logs before the next risky run.

| Priority | Experiment | Discriminator / stop condition |
| --- | --- | --- |
| P0 | Reproduce the current record with a clean captured environment | Establish present throughput and same-seed BPB before interpreting new wins |
| P0 | Read-only corpus/packing coverage audit | Test whether epoch counter, effective distinct tokens, and printed throughput describe the same thing |
| P1 | Only mask zero-byte targets in CE | Exact objective-alignment control; reject if overhead or loss trajectory negates any gain |
| P1 | Trusted-batch reordering/buffered replay, especially final cooldown | Separate a narrow late-data repeat from an intrinsic inability to benefit from repeat data |
| P1 | Current compiled-step profile at the actual winning shape | Select one kernel whose removable cost can plausibly move full-budget quality |
| P2 | BF16 fused operation chosen by that profile | Require forward/backward correctness plus complete-operation speedup |
| P2 | Native-format scaled FP8 MLP-forward prototype | Reject if casts/transposes absorb the gain or short-run quality degrades beyond the speed budget |
| P3 | Within-run snapshot/SWA variants or auxiliary prefix loss | Only after checking earlier negative evidence and isolating a materially different schedule; no cross-run learned assets |

The earlier “data wall” makes it especially important to measure speed and learning independently. A faster kernel may pay by training a wider dense network at the same token count, by fitting more updates at a smaller global batch, or by sampling a more representative final training segment. Those are separate hypotheses, requiring their own controls; elapsed time, optimizer steps, and token presentations must all be logged.

## Source and tooling notes

- Live Devpost pages were extracted with the already-installed Tavily CLI at `/home/csaba/.local/bin/tvly`; no packages were installed. The initial PATH omission was resolved with its absolute path.
- Broad Tavily queries containing several `site:` constraints returned irrelevant results in this pass. Those results were discarded. Official hardware statements were independently verified against AWS documentation and the installed `neuron-nki-docs` reference files.
- Exact local rules and evaluator behavior above come from this checkout's `README.md` and `prepare.py`, not an invented private-runner implementation.
- Relevant historical evidence: `research/engineering-stance-parallelism-precision-2026-09-12.md`; `research/plan-sub1-2026-09-16.md`; `research/experiment-protocol.md`; and the “data wall” section in `research/trn2-experiment-results.md`.
- Potential follow-up primary source supplied by the main research agent: [Gram Newton–Schulz](https://tridao.me/blog/2026/gram-newton-schulz/). It has not yet been evaluated in this document; optimizer speed is only valuable in proportion to its measured current-step share.
