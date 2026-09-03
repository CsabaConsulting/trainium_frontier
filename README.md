# The AWS Trainium Frontier

### AI-Assisted Model and Kernel Co-Design on AWS Trainium

[![Hardware](https://img.shields.io/badge/Hardware-AWS%20Trainium-orange)]()
[![Interface](https://img.shields.io/badge/Kernels-NKI-blue)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-green)]()

Train a language model from scratch on AWS Trainium under a fixed time budget, and
push it as far as you can across two axes: how low can you drive **validation
bits-per-byte (`val_bpb`)**, and — in Phase 2 — how capable a model can you train as
measured by **CORE**. You may optimize anything: the architecture, the optimizer, the
training loop, the data sampling, and even custom hardware kernels written in
the [**Neuron Kernel Interface (NKI)**](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html).

The starter kit ships a strong, readable baseline and a development loop tuned for
rapid iteration — by hand, with AI agents, or any mix of the two.

FOR GENERAL QUESTIONS, PLEASE REFER TO THIS [LINK](https://trainium-frontier.devpost.com/forum_topics)
---

## Table of Contents

- [Why This Competition](#why-this-competition)
- [Quickstart](#quickstart)
- [The Task](#the-task)
- [What You May and May Not Modify](#what-you-may-and-may-not-modify)
- [Submission Format](#submission-format)
- [Scoring](#scoring)
- [The Baseline](#the-baseline)
- [Technical Environment](#technical-environment)
- [AI-Assisted Research with autoresearch](#ai-assisted-research-with-autoresearch)
- [Timeline](#timeline)
- [Prizes](#prizes)
- [Eligibility](#eligibility)
- [Finalist Deliverables](#finalist-deliverables)
- [Contact](#contact)

---

## Why This Competition

Modern LLM architectures have co-evolved within a single hardware family — the shapes
of our attention mechanisms, MLPs, numerical formats, and parallelism strategies have
all been implicitly shaped by the constraints of the chips they run on. When the
hardware changes, the efficient frontier of model architectures changes with it. This
competition asks you to rediscover what *efficient* means on genuinely new silicon.

AWS Trainium offers a different surface area for developer control: a 128×128 systolic
TensorEngine, a 128-lane VectorEngine, explicit on-chip SBUF scratchpads managed in
software, and a Python-native kernel interface (NKI) that compiles custom kernels in
seconds and integrates them as standard PyTorch ops. The entire NKI API surface fits in
a weekend — equally accessible to a human writing kernels by hand and to an AI agent
generating them under human direction.

There is a direct tradeoff at the heart of the challenge: a better architecture needs
fewer steps to reach a given loss, while faster kernels fit more steps into the same
budget. The winning solution finds the best point on that frontier — and because the
frontier is shaped by Trainium's specific silicon, the solutions will look different
from those tuned for other hardware.

**Who this is for:** ML architecture/training researchers, ML systems researchers
interested in hardware-aware optimization and custom kernels, and teams building
AI-assisted research workflows. No prior Trainium experience is required; familiarity
with PyTorch and transformer training is expected, and CUDA/Triton experience transfers
directly to NKI.

---

## Quickstart

### 1. Environment

Start from the provided AWS Trainium training DLAMI (id: ami-0d5703e99c1589f8d venv: neurips-torchneuronx-native-beta-2.11.3-20260824):

```bash
pip install -r requirements.txt
```

### 2. Download data and the tokenizer

`prepare.py` downloads the training/validation shards and unpacks the fixed BPE
tokenizer artifact:

```bash
python prepare.py setup
```

(Use `--num-train-shards N` to fetch fewer shards while iterating;
`--build-tokenizer-if-missing` rebuilds the tokenizer from scratch instead of using the
shipped artifact — not needed for normal runs.)

### 3. Train the baseline

```bash
NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py
```

The world size is detected from `torchrun`. You can also run with `lnc=2`:
`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`. Training
writes a checkpoint to `out/final.pt`.

### 4. Score your model locally

Run the **exact** scoring metric against your own checkpoint, any time you like, using
the public validation shard:

```bash
python prepare.py eval-public --train-py train.py --checkpoint out/final.pt
```

This calls the same trusted `evaluate_bpb()` function the official scorer uses (see
[Scoring](#scoring)), so the number you see locally is computed identically to the one
on the leaderboard — only the validation data differs.

### 5. (Optional) Try the example NKI kernel

`train.py` ships one inline NKI kernel (a fused `relu(x)²`), **off by default**. Enable
it to see a custom kernel run end-to-end:

```bash
NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py --nki-relu2
```

It is numerically identical to the eager baseline and is written for clarity, not
speed — a starting template for your own kernels (e.g. fused RMSNorm or attention).

---

## The Task

The competition runs in two phases.

| Phase | Hardware | Teams | Training budget | Scoring |
| --- | --- | --- | --- | --- |
| **Phase 1** | Single Trn2 chip (`trn2.3xlarge`) | All registered (up to 100) | 30 minutes | `val_bpb` |
| **Phase 2** | Full Trn2 server (`trn2.48xlarge`, 16 chips) | Top 10 from Phase 1 | 4 hours | Weighted `val_bpb` + CORE |

- **Phase 1:** Achieve the lowest `val_bpb` after exactly 30 minutes of training on a
  single Trn2 chip. Advancement to Phase 2 is determined purely by Phase 1 leaderboard
  position (lowest `val_bpb`).
- **Phase 2:** Achieve the best weighted score across `val_bpb` (training efficiency)
  and CORE (capability) after exactly 4 hours of training on a full server. You also
  provide an `inference.py` wrapper so the scoring infrastructure can run CORE against
  your trained model.

Because the chip architecture is consistent across phases, Phase 1 learnings transfer
directly to Phase 2 — the new surface area in Phase 2 is scale, collective
communication, and training a model that performs well on downstream in-context
learning, not a different chip to re-learn.

**Model size is fully open.** Choose your own depth, width, and parameter count. The
`val_bpb` metric is vocabulary-size-independent and there are no parameter caps. Ties
are broken by (1) fewer total parameters, then (2) simpler code.

---

## What You May and May Not Modify

**Permitted:**

- Modify `train.py` freely: architecture, optimizer, hyperparameters, training loop,
  batch size, model size, data sampling strategy, and multi-chip parallelism strategy in Phase 2.
- Modify `inference.py` in Phase 2: inference implementation, KV-cache strategy, custom
  inference kernels.
- Write custom NKI kernels that replace standard PyTorch operators (attention,
  normalization, matmul, optimizer steps, loss, collective communication in Phase 2).
- Combine ML-level and systems-level optimizations in any proportion.
- Use AI agents, the provided `autoresearch` framework, or other automated tools at any
  stage of development.
- Run the scoring metric against your own model during development, as often as you
  like.

**Not permitted:**

- Modifying `prepare.py`, the evaluation harness, the data pipeline, the tokenizer, or
  the scoring functions (`evaluate_bpb()`).
- Using pre-trained weights, embeddings, tokenizers, or any external training data — you
  may only use assets produced within your own training loop on the provided data.
- Training on, or otherwise incorporating, the CORE eval bundle, CORE task data, answer
  keys, or any derivative thereof.
- Hardcoding CORE-specific heuristics, answer lookups, or task detectors in the
  inference path.
- Reading, mirroring, or inferring the validation data used for scoring.
- Installing packages beyond the provided environment.
- Exceeding the training-time budget for the phase (30 min / 4 hr).
- Artificially restricting training to a subset of NeuronCores — submissions must run on
  the full chip (Phase 1) and the full server (Phase 2).
- In Phase 2: shipping an inference path that is inconsistent with the trained model
  (e.g. evaluating a different model than was trained).

Submissions are screened by an automated pipeline (static analysis + an LLM-based code
reviewer prompted with these rules) and, for top entries, by manual review. The intent
is to keep the leaderboard reflecting genuine progress in training efficient, capable
models — not exploitation of the evaluation infrastructure.

---

## Submission Format

You submit:

1. Your edited **`train.py`**.
2. A **launch-command string**, e.g.
   `NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py`.
   Valid shapes: `nproc_per_node ∈ {1, 4, 8}`, `NEURON_LOGICAL_NC_CONFIG ∈ {1, 2}`, no
   shell pipes/redirects/chaining/background.
3. (Optional) custom NKI kernels referenced by your `train.py`.
4. (Phase 2 only) your **`inference.py`** wrapper.

Your `train.py` **must define a `load_for_eval` hook** that the scorer uses to load your
trained model:

```python
def load_for_eval(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Return an eval-mode model loaded from checkpoint_path."""
```

Contract: it must return a **consolidated, single-device, replicated full model** with a
**pure, stateless, causal forward** — logits must be a function of the input tokens
only (independent of batch size, world size, rank, and batch content; no collectives in
forward; frozen, no hooks). Tensor/pipeline/FSDP parallelism is training-only;
consolidate before saving, because the scorer does not merge shards.

**Rate limits:** 5 submissions per team per week in Phase 1, 3 per week in Phase 2; your
best score counts. A submission must complete training within budget and produce a
loadable checkpoint with no NaNs or crashes — there are no retries.

---

## Scoring

### Phase 1 — `val_bpb`

The metric is **validation bits-per-byte**, byte-weighted across the validation data:

```
val_bpb = total_nats / (log(2) · total_bytes)      # lower is better
```

The score is computed by the trusted `evaluate_bpb()` function in `prepare.py`, which
**recomputes cross-entropy directly from your model's logits** — a participant-reported
loss is never trusted.

> **The scoring methodology is fully public.** The official leaderboard runs
> `evaluate_bpb()` against a pinned, private validation shard under a time budget, but
> the *function itself lives in the `prepare.py` you already have*. You can reproduce
> the exact metric locally on the public dev shard at any time:
>
> ```bash
> python prepare.py eval-public --train-py train.py --checkpoint out/final.pt
> ```
>
> The only difference between your local number and the leaderboard number is the
> validation data: the official shards are held out and disjoint from anything you can
> train or self-evaluate on, so memorizing the public dev shard only inflates your own
> local metric — it cannot move the leaderboard.

### Phase 2 — Weighted `val_bpb` + CORE

Phase 2 combines a training-efficiency metric and a capability metric, 50/50:

- **`val_bpb` (50%)** — as above, on the final trained model. Lower is better.
- **CORE (50%)** — the CORE metric from the nanochat evaluation suite: an aggregate
  across a pinned battery of in-context-learning tasks (reading comprehension,
  commonsense reasoning, language understanding, world knowledge, symbolic
  problem-solving). Per-task accuracy is centered against the task's random baseline,
  then averaged. CORE is evaluated in scoring mode (log-likelihood over candidate
  completions, no autoregressive generation). Higher is better.

Both metrics are min-max normalized across the Phase 2 submission pool, directions
aligned, then averaged. **The CORE eval bundle is public and pinned** — you can download
it and evaluate your own models during development. A 60-minute CORE evaluation timeout
applies on the full server (excluding one-time compilation); treat inference efficiency
as a first-class design constraint from the start of Phase 2.

### Correctness gate (both phases)

Every submission must complete training within budget, produce a valid score (no
NaN/crash), run on the provided environment without modifying the harness, use the full
chip/server, and use only the provided training data. In Phase 2 the inference path must
be consistent with the trained model and complete CORE within the timeout.

---

## The Baseline

The shipped `train.py` is a complete, readable nanochat-derived pipeline that is a
credible starting point but leaves substantial headroom on both the architecture and
kernel axes:

- **Model:** GPT-style dense LLM (~59.6M params at depth 10 for the Phase 1 starting
  point): untied embeddings/head, RMSNorm, rotary positional embeddings, QK norm, causal
  attention, ReLU² MLP, logit softcapping.
- **Optimizer:** Muon (Polar Express orthogonalization) for matrix parameters; AdamW for
  embeddings, head, and scalars.
- **Schedule:** warmup/warmdown tuned for the training budget.
- **Parallelism:** utilizes all NeuronCores on the chip (Phase 1) / all chips on the
  server (Phase 2) out of the box.
- **Kernels:** one inline example NKI kernel (fused `relu(x)²`, default off) as a
  starting template.

The baseline is eager-mode; `torch.compile` / Neuron trace are allowed at your own risk,
and compile time counts against the budget.

---

## Technical Environment

**Phase 1 — single Trn2 chip (`trn2.3xlarge`):** NeuronCore-v3 architecture (128×128
systolic TensorEngine, 128-lane VectorEngine), 24 MiB SBUF per NeuronCore, 32 GB HBM per
bank, 820 GB/s HBM bandwidth.

**Phase 2 — full Trn2 server (`trn2.48xlarge`):** 16 Trainium chips connected via
NeuronLink; you manage the parallelism strategy (tensor / pipeline / data parallel) at
the framework and/or kernel level. A working multi-chip baseline is provided.

**Software:** PyTorch native for Neuron, the latest stable Neuron SDK and NKI releases
at competition launch, and the nanochat-derived training pipeline + CORE evaluation
harness.

**Data:** the ClimbMix dataset (publicly available, shuffled web text). The tokenizer is
a fixed BPE tokenizer (vocab size 8,192), identical for all participants and shipped as a
SHA256-verified artifact. The validation shard used for `val_bpb` is pinned, held out,
and not part of training data.

---

## AI-Assisted Research with autoresearch

This competition treats AI-accelerated research as a first-class mode of participation.
The kit includes `autoresearch/`, a Trainium-optimized version of the open-source
human-in-the-loop agent-driven experimentation framework. Its narrow single-file
modification surface, scalar reward signal, and tight feedback loop are designed for
directing AI agents to explore architectures, generate kernels, tune hyperparameters,
and run experiments under human supervision.

It runs a single-core, fixed 5-minute training loop for rapid iteration:

```bash
cd autoresearch
bash run.sh
```

`autoresearch/` is a **development tool only** — it is not part of your submission or the
evaluation path. The leaderboard treats all submissions identically, whether fully
human-driven, fully agent-driven, or anywhere in between.

---

## Timeline

| Date | Milestone |
| --- | --- |
| August 31, 2026 | Phase 1 opens; Trn2 leaderboard goes live |
| September 30, 2026 | Phase 1 closes; top 10 teams announced |
| October 7, 2026 | Phase 2 opens; top 10 teams receive `trn2.48xlarge` access; CORE bundle pinned (SHA published) |
| November 4, 2026 | Phase 2 closes |
| November 11, 2026 | Finalists selected |
| December 6–12, 2026 | Finalist presentations at the competition workshop |

---

## Prizes

- **First Place:** $25,000 USD

- **Second Place:** $10,000 USD

- **Third Place:** $5,000 USD
- **Top 10 Phase 2 finalists:** receive exclusive Neuron team jackets and finalist swag packs.
- **Top 3 Phase 2 finalists:** one month of dedicated single-chip **Trn3** access for
  research use — explore your winning approach on next-generation silicon (MXFP8/FP4 data
  types, larger SBUF, higher HBM bandwidth).

AWS credits are provided to academic teams in both rounds, including Trainium and LLMs on
Amazon Bedrock.

---

## Eligibility

- **Team size:** 1–4 members per team.
- **Maximum teams:** 100 (first-come, first-served registration).
- **Open to all**, except Amazon employees, interns, and scholars (2025 or 2026) and
  their immediate family members, who are ineligible. Contributors to the `nanochat` and
  `autoresearch` open-source projects are explicitly welcome.

---

## Finalist Deliverables

Phase 2 winners (top 3 teams) provide, by the close of the competition workshop:

- **Open-source release:** complete submission code and final trained model weights under
  Apache 2.0.
- **Technical report (4–8 pages):** architectural choices, kernel implementations,
  ablations across `val_bpb` and CORE, and lessons learned.
- **Presentation:** a 15-minute talk at the competition workshop (at least one team
  member attending in person).

All winning techniques are published and reproducible — a durable contribution to the ML
systems community and a head start for the next wave of hardware-aware, AI-accelerated
architecture research.

---

## Contact

Questions? Reach the organizing team at the address published with the competition
announcement.
