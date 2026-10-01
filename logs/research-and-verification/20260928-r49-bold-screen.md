# Round49 bold plain-PyTorch screen — no qualifying new arm

Date: 2026-09-28. Evidence cutoff: 10:58 UTC / 16:28 IST.
Role: bounded read-only Researcher. Advisory only; not independent verification, a completed experiment proposal, queue entry, or launch authority.

## Decision

**NO QUALIFYING NEW IDEA FOUND. Zero new arms proposed.**

The search covered primary modded-nanogpt records/PRs, nanochat's own experiment summary, and the original conditioned/mimetic initialization papers. One genuinely different mechanism received a closer screen: conditioned attention initialization. Its transferable gain remains unknown and it does not clear Round49's credible >=0.005 public-BPB requirement. This is a bounded negative research finding, not a claim that no such technique exists.

[Round49](/home/poojith-devan/Desktop/Trainium/workflow/ROUND49_2026-09-28.md) controls: do not resume cancelled warmdown/ramp/LR/MTP variants, duplicate the active MUDD-lite experiment, or invent a stability arm from unsupported spike correlations. This report does not change MUDD, cleanup, the fallback package, or host priorities.

## Baseline and evidence scale

Exact source inspected and SHA-256 verified:
[cleanup train.py](/home/poojith-devan/Desktop/Trainium/research/v2/proposals/20260928-clean-conv-eval-path/train.py),
`6497d163d930275dfbc0bb494fbb546456478e3751119dcc1baf918f5638f9d3`.

It retains eight 1024-wide blocks, attention at 0/2/4/6/7, 128-wide heads, 4096-wide ReLU² MLPs, the established Muon/AdamW path, and shared training/evaluation model behavior. Its initialization already uses `INIT_SCALE=0.5` and a nonzero attention output projection initialized uniformly within `±0.1*s` (lines 77–78, 837–848).

[Current research brief](/home/poojith-devan/Desktop/Trainium/research/v2/RESEARCH_BRIEF.md) records the base eight-cold public20M mean as **0.9766558936725349**. Human official score **0.9768** and tenth **0.9700** are human-reported in [Round49](/home/poojith-devan/Desktop/Trainium/workflow/ROUND49_2026-09-28.md), not a fresh portal observation. The base means and logs were not recomputed in this research task.

A 0.005 gain means a public target of **0.9716558936725349** relative to that mean. Using the established public evaluation totals of 20,971,520 tokens and 83,462,186 bytes, it corresponds to **0.01379289124 fewer nats per evaluated token**:

`0.005 * ln(2) * 83,462,186 / 20,971,520`.

That is only the units conversion for this evaluator. A nats/token improvement on a different tokenizer or dataset is not a Trainium BPB prediction.

## Closest new lead: conditioned Q/K/V initialization — ineligible

**Author-reported measurement:** the original paper's 472M GPT-2 WikiText-103 experiment reports perplexity 44.2 → 42.7. Its recipe initializes Q/K independently semi-orthogonally per head, V as the same rectangular identity per head, and preserves the output projection. [Paper, §3.3 and Appendices A.1.1/A.4](https://arxiv.org/html/2609.07086v1).

**Exact diff proposal: NONE selected.** Counterfactual footprint: only replace Q/K/V initialization inside `GPT.init_weights` for the five attention blocks; the forward, evaluator and optimizer stay as supplied. No such patch was written.

**Eligibility inference:** `ln(44.2/42.7)=0.03452586885`, numerically larger than the required local nats change. Transferring at least 39.95% would be needed even under a naive equal-unit comparison. There is no supporting local measurement for that transfer across model depth, data, tokenizer, Muon, and the short training horizon. The paper's unit-scale initialization also differs from the base's 0.5-scale scheme; a scale-matched port would be another unmeasured variant.

**Risk inference:** initialization adds no recurring forward operations; native steady-step overhead is therefore expected to be negligible, not measured. Startup QR/SVD placement and repeated per-head V subspaces require careful implementation. No compile, numerical, timing, or quality gate was attempted. **Do not enqueue.**

## Other screened leads and explicit rejects

These are explanations of rejection, not alternate proposals. External figures below are primary authors' reported measurements; this task did not reproduce their runs.

| Lead | Primary evidence | Why it does not qualify here |
|---|---|---|
| Paired-head attention | [modded-nanogpt PR191](https://github.com/KellerJordan/modded-nanogpt/pull/191) reports five losses averaging about 3.2788 and time 108.7958 s versus a prior-record retiming of 112.262 s. It bundles paired heads with a RoPE implementation improvement and uses 65 fewer steps. | No isolated BPB-sized effect. From the stated reshape, pairing changes H heads of length T into H/2 heads of length 2T: dense score elements double in each changed layer. That cost is an algebraic inference for our dense context-1024 path; the upstream short-window result cannot establish a safe fixed-budget transfer. No port proposed. |
| Learnable XSA | [PR264](https://github.com/KellerJordan/modded-nanogpt/pull/264) reports matched-1410-step H200 loss 3.28316 → 3.27865, a 0.00451-nat gain, with zero-initialized learned gates. | The external gain is below the local 0.01379-nat target before transfer assumptions. Local fixed XSA previously lost **+0.010179 BPB** and 4.2% of steps: [local result](/home/poojith-devan/Desktop/Trainium/research/trn2-experiment-results.md:427). Gating is a real distinction, but neither it nor the PR's unarchived XSA+MUDD anecdote supports reopening this lane for a >=0.005 prediction. |
| Lightweight dynamically composable attention | [PR315](https://github.com/KellerJordan/modded-nanogpt/pull/315) reports combined matched-step gain 0.00763 nats; its DC-only attribution is about 0.0045, with the remainder attributed to MUDD gates/cleanup. Its fast correction uses a custom Triton path alongside FA3. | No isolated >=0.005-BPB support and a plain-PyTorch port lacks the reported execution path. The local full DCMHA implementation previously failed with **NCC_EXTP004**; eager was roughly 100x slower: [local history](/home/poojith-devan/Desktop/Trainium/research/trn2-experiment-results.md:5763), section “Two measured verdicts that existed ONLY in train.py comments.” The lite variant is not identical to that failure, but no compiler-safe, low-overhead port is demonstrated. It also overlaps the current MUDD experiment. |
| Mimetic V/O initialization | [PR214](https://github.com/KellerJordan/modded-nanogpt/pull/214) initially reports 20 fewer steps and ~1.16 s improvement. Its later discussion says positive/negative identity give similar results and ordinary nonzero random attention initialization recovers the gain. | The initialization mechanism is not isolated by the headline. The exact6497 base already uses small nonzero attention output initialization. Historical local initialization-scale improvements were below 0.001 and the corresponding old log entries are dirty-tree evidence, not valid current wins. No basis for >=0.005 remains. |
| Smear / additional attention gates | [nanochat's own discussion481](https://github.com/karpathy/nanochat/discussions/481) reports negligible gains for these in that experiment series, while reporting 0.003–0.01 BPB for residual/x0 scalars. | Residual/x0 scalars already exist here. The exact base already has causal local convolution. An input-smear gate would be different, but this source supplies no evidence for a >=0.005 improvement on top of this stack. No local-convolution or gate variant proposed. |

The older [mimetic paper](https://proceedings.mlr.press/v202/trockman23a/trockman23a.pdf) reports WikiText-103 perplexity 28.87 → 28.21 and character-PTB 1.233 → 1.210 BPC. Those are real reported task-specific results, not interchangeable with this competition's public BPB or proof of the speedrun's original causal explanation.

## Operational disposition

- New experiment proposals: **0**. Selected exact diff: **none**. Expected new-arm BPB gain: **unknown**.
- The [ten-log instability analysis](/home/poojith-devan/Desktop/Trainium/research/v2/research_cycles/20260928-round48-instability.md) remains the relevant local evidence; no new stability inference is drawn from the initialization papers.
- Host/device contacts, training, model loads, numerical tests, candidate edits, Bedrock/AWS calls, queue/state mutations, submissions, and cleanup actions in this task: **none**.
- Only this report and its identical Downloads copy were created. No host slot was reserved and no MUDD or cleanup dependency was introduced.
