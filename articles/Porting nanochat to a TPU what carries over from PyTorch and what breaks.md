Karpathy's nanochat normally runs on an 8×H100 GPU node, and several ports of it to JAX already exist. Among them, my aim was to keep the config and architecture as close to nanochat as possible ([parity](https://github.com/tucan9389/nanochat-jax/blob/main/dev/PARITY.md)) while catching up on both model quality and training performance — the quality (its CORE score) reproduced cleanly, and the performance came only partway. If nanochat is new to you: it's Karpathy's full-stack LLM project — tokenizer training, pretraining, SFT, and RL in a single repository — where about four hours and roughly $100 on one 8×H100 node gets you your own chatbot (nanochat is [often called the "$100 speedrun"](https://github.com/karpathy/nanochat); those numbers are for d20). For reference, stopping at the GPT-2-grade base model (d24) takes about two hours and roughly $48 on the same node. This post is a record of that port: what carried over unchanged from PyTorch, and what broke on the TPU.

## 1. Reproduction results

nanochat-jax currently provides a speedrun.sh script that covers base model training and SFT (upstream nanochat goes all the way to RL; this reproduction stops at SFT). The claim above — that the quality reproduced — is based on the CORE score. [CORE](https://arxiv.org/abs/2406.11794) is the average of accuracies over 22 evaluation tasks, each rescaled so that random guessing scores 0 and a perfect score is 1. Within the same evaluation harness, it lets you compare models against each other, which is why nanochat uses it to judge what counts as "GPT-2 grade". What we reproduced is recipe 4 from the official nanochat [LEADERBOARD](https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md) (R4 from here on; it's also called d24, since depth 24 ≈ 1.4B parameters). The band (0.2512–0.2677) is the score distribution from Karpathy running the same R4 recipe 7 times; this run's 0.2695 lands just above it.

On performance, there's still a gap. MFU is about 24% (d24) — half of Karpathy's measured H100 numbers (47–48% at d20).

Below, we run each script of speedrun.sh on a TPU v6e-8 and check whether the quality Karpathy reported actually comes out.

| Model               | Chip              | CORE (base) | Train time      | Train cost      | Total cost        |
|---------------------|-------------------|-------------|-----------------|-----------------|-------------------|
| GPT-2 (2019, 1.5B)  | TPU v3 x32 (est.) | 0.2565      | 168 h           | ~$43,000 (est.) | -                 |
| nanochat R4 (d24)   | H100 x8           | 0.2571      | ~2 h            | ~$48 (est.)     | -                 |
| nanochat-jax (ours) | TPU v6e-8 (spot)  | 0.2695      | 5.29 h (6.02 w) | $30 ($130 od)   | $60.8 (~$263 od)  |

*CORE scores and the band are from the nanochat [LEADERBOARD](https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md). w = wall clock including checkpointing and compilation; od = on-demand list price.[^costnote]*

If you spot anything wrong or unclear in this post, or have a question, please let me know at tucan.dev@gmail.com or in the comments — feedback is always welcome 🙂

---

## 2. TPU basics

This run used **8 v6e chips** (a single-host slice). Per-chip specs across generations:

| | v5p | **v6e (Trillium)** | 7x (Ironwood) |
|---|---:|---:|---:|
| HBM capacity | 95 GB | **32 GB** | 192 GB |
| HBM bandwidth | 2,765 GB/s | 1,638 GB/s | 7,380 GB/s |
| bf16 compute | 459 TFLOPs | **918 TFLOPs** | 2,307 TFLOPs |
| Low-precision compute | Int8 918 TOPs | **Int8 1,836 TOPs** | FP8 4,614 TFLOPs |
| MXU (matrix-multiply unit) | 128×128 | **256×256** | 256×256 |
| Chips per host | 4 | **8** | 4 |
| 1-host topology | 2×2×1 | **2×4** | 2×2×1 |
| Max pod | 8,960 chips | 256 chips | 9,216 chips |

*Sources: [Cloud TPU System Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture), [v6e](https://docs.cloud.google.com/tpu/docs/v6e), [tpu7x (Ironwood)](https://docs.cloud.google.com/tpu/docs/tpu7x). bf16, Int8, and HBM numbers are from each generation's official spec sheet. Low-precision compute is roughly 2× bf16; v5p and v6e hardware-accelerate only up to Int8, and native FP8 support starts with Ironwood.*

The v6e's MXU grew to 256×256, from the 128×128 of every generation up to v5p — if a tensor dimension isn't a multiple of 256, XLA pads it with zeros and part of the unit is wasted (more on this in insight 5 of section 4). Meanwhile, HBM is 32GB per chip — a third of v5p's 95GB — while compute is 2×: a compute-heavy, memory-lean design.

Our v6e-8 slice adds up to 7.34 bf16 PFLOPS and costs \~$4–5/hour on **spot** (us-central1). Spot is heavily discounted against on-demand, in exchange for GCP being able to reclaim (preempt) it at any time. This run cost $60.8 on spot over 12.19 hours total (\~$263 at on-demand rates), with one preemption and recovery along the way. You're billed for as long as the node exists (training or not) — forget to delete it, and at $5/h for an 8-chip slice, a day is ≈ $120.

<img width="598" height="340" alt="The JAX AI Stack: hardware (CPU/GPU/TPU), the XLA compiler, JAX, and the library layer including Flax" src="https://github.com/user-attachments/assets/4072e6ad-7949-401f-9ab7-ddbc03920033" />

**Figure 1.** The JAX AI Stack — hardware (CPU/GPU/**TPU**) → the **XLA** compiler → the **JAX** core → the library layer on top (**Flax** and others). (Source: [jaxstack.ai](https://jaxstack.ai/), © JAX team)

Of this stack, nanochat-jax uses three software layers — XLA, JAX, and Flax — plus Pallas, which isn't in the figure.

| Role | nanochat-jax | PyTorch equivalent |
|---|---|---|
| Neural network modules | Flax NNX | `torch.nn` |
| Computation + compilation | JAX + XLA (`jit`·`grad`·`vmap`) | eager + `autograd` + `torch.compile` |
| Custom kernel | Pallas (Splash Attention) | Triton |
| Optimizer | JAX (own implementation, no `torch` dependency) | `torch.optim` |
| Data loading | NumPy (no `torch` dependency) | `torch.utils.data` |
| Checkpoint | PyTorch `.pt` (for nanochat compatibility) | PyTorch `.pt` |

---

## 3. Speedrun verification — from the tokenizer to the report card

The pipeline runs tokenizer (5.3m) → base (6.02h + eval 44.5m) → SFT (68.7m + eval \~3.5h), and we run the scripts that speedrun.sh executes, one stage at a time (to run everything in one shot, see Appendix C). For a deep understanding of each stage, see Karpathy's [walkthrough](https://github.com/karpathy/nanochat/discussions/1) post (his is d20, ours is d24); this post focuses on quickly checking measured values against the reference numbers. The midtraining from the walkthrough era (an intermediate stage that pre-taught conversation format and tool use) no longer exists as a separate stage upstream (its data was folded into the SFT mixture), so we also go straight from base to SFT. Setup starts in Appendix A.

<img width="684" height="134" alt="The nanochat speedrun pipeline: train tokenizer → train base model → train SFT → train RL (dashed)" src="https://github.com/user-attachments/assets/f4540502-c4a9-4b95-8330-42ab700a5399" />

**Figure 2.** The nanochat speedrun pipeline — this reproduction runs tokenizer, base, and SFT (the solid boxes) and skips RL (dashed).

### Step 1. Tokenizer (5.3m)

```bash
python -m nanochat_jax.dataset -n 170     # ClimbMix train 170 + val 1 shards
python -m scripts.tok_train --max-chars 100000000 --doc-cap 10000 --vocab-size 32768
python -m scripts.tok_eval
```

We train a vocab-32768[^vocab] tokenizer on 100M characters of [ClimbMix](https://huggingface.co/datasets/nvidia/ClimbMix), then compare against GPT-2 how many tokens the same text takes. On the training data (the `train` row), ours uses **1.5% fewer** tokens than the GPT-2 tokenizer. Fewer tokens means more text learned for the same budget — a compression advantage — and Karpathy's tokenizer shows the same pattern, a signal that the reproduction is on track (for Korean, the training data contains no Hangul, so ours spends about 2× the tokens of GPT-4; the full per-domain table is in Appendix D). **The number to check: `train` +1.5%.**

### Step 2. Base model (6.02h + 44.5m)

```bash
# --recipe through --use-real-data: Karpathy's R4 config / --attn-impl through --splash-*: v6e-8 TPU-specific
python -m scripts.base_train \
  --recipe=324e69c --depth=24 --seq-len=2048 --vocab-size=32768 \
  --target-param-data-ratio=9.5 --total-batch-size=1048576 \
  --device-batch-size=2 --grad-accum-steps=32 --grad-accum-impl=fused \
  --warmup-steps=0 --warmdown-ratio=0.5 --final-lr-frac=0.0 \
  --weight-decay=0.2 --matrix-lr=0.02 --embedding-lr=0.3 \
  --unembedding-lr=0.004 --scalar-lr=0.5 \
  --bf16 --cast-embeddings-bf16 --use-real-data \
  --attn-impl=splash --splash-block-q=512 --splash-block-kv=512 --splash-block-kv-compute=256 \
  --matmul-precision=default --lm-head-precision=highest --ve-grad-impl=onehot \
  --checkpoint-every=200 --keep-last-checkpoints=2 \
  --model-tag=d24_speedrun_r4 --no-final-eval
```

The top half of these arguments is the R4 recipe as-is (730M scaling parameters × 9.5 ≈ 6.9B tokens)[^ratio]; the only ones you may need to touch are the `--splash-*` kernel block sizes.[^splash]

```text
[step  100] loss=4.517054 ... mfu=24.5%
[step 1000] loss=2.786205 ... mfu=24.7%
```

Against Karpathy R4's 2-hour base train loop (8×H100), ours takes 5.29h (6.02h wall clock including checkpointing and compilation), at about 24% MFU[^mfu].[^scaling] (The two lines above are an excerpt; I've uploaded the [full training log](https://github.com/tucan9389/nanochat-jax/releases/download/row2/31_base_train.log) of this run — the header also prints flops/token and peak TFLOPS.)

```bash
RESULTS_DIR=$HOME/.cache/nanochat-jax/results/d24_speedrun_r4 && mkdir -p $RESULTS_DIR

python -m scripts.base_eval --source base --model-tag d24_speedrun_r4 \
    --eval bpb,core --eval-steps 20 --device-batch-size 1 --max-per-task -1 \
    --out $RESULTS_DIR/base_eval.json --partial-out $RESULTS_DIR/base_eval_core_partial.json --bf16
```

CORE (the 22-task metric from section 1) comes out to **0.2695** — above GPT-2's 0.2565, and in the same class as the R4 band (0.2512–0.2677). The val bpb of 0.7343 is a reference number only, since tokenizer differences mix into it (R4: 0.7185).

<img width="2300" height="500" alt="Three base-training metric charts" src="https://github.com/user-attachments/assets/c84067d8-217c-4b27-bbe1-fcc34eedd7ac" />

**Figure 3.** The dashed lines in the middle and right panels are the R4 baselines — by the end of training, both metrics reach the baselines, landing at R4 level (the curves are from an earlier reproduction run).

### Step 3. SFT (68.7m + eval \~3.5h)

```bash
# SFT training (68.7m)
python -m scripts.chat_sft --model-tag d24_speedrun_r4 --output-model-tag d24_speedrun_r4_sft \
    --num-iterations -1 --device-batch-size 1 --max-seq-len 2048 --total-batch-size 524288 \
    --sft-scope full --eval-every 200 --eval-steps 4 \
    --chatcore-every -1 --chatcore-max-cat -1 --chatcore-max-sample 24 \
    --load-optimizer 1 --log-every 10 --bf16
# scoring — the 3 multiple-choice tasks in one go, the 3 generative tasks one by one
python -m scripts.chat_eval -i sft -g d24_speedrun_r4_sft --max-new-tokens 512 --batch-size 8 \
    --dtype bfloat16 --out $RESULTS_DIR/chat_eval_sft.json --task-name "ARC-Easy|ARC-Challenge|MMLU"
for task in SpellingBee HumanEval GSM8K; do
  python -m scripts.chat_eval -i sft -g d24_speedrun_r4_sft --task-name "$task" \
      --max-new-tokens 512 --dtype bfloat16 --num-samples 1 --temperature 0 --top-k 50 \
      --jit-gen 1 --out $RESULTS_DIR/chat_eval_$(echo "$task" | tr 'A-Z' 'a-z').json
done
```

Train for 68.7 minutes on a mixture (\~1.07M rows) of SmolTalk, MMLU, GSM8K, SpellingBee[^abilities], and 1,000 identity conversations[^identity], and the base model that could only repeat the question back becomes a model that answers it. ChatCORE (the centered average over 6 chat tasks) is **0.3733** — the comparison line here is the base model itself. On the same 6 tasks, base scored essentially 0 on the generative ones, so most of the generative score is ability SFT added: SpellingBee 0.9961, GSM8K 0.1008 (math is RL's job). I don't compare 1:1 against the report card in Karpathy's walkthrough — that one is d20 and ran on the older midtraining pipeline.

<img width="3000" height="680" alt="Three SFT metric charts" src="https://github.com/user-attachments/assets/bbd9c545-7abb-4bd6-a5e3-ea33f0a65176" />

**Figure 4.** Right panel: SFT (blue) is ahead of base (gray) on every task — you can see the generative tasks, which scored 0 on base, now score above zero.

Beyond the scores, checking with actual prompts:

```bash
JAX_PLATFORMS=cpu NANOCHAT_JAX_BASE_DIR=<checkpoint dir> PYTHONPATH=<nanochat-jax dir> \
python scripts/chat_cli.py -i sft -t 0.0 -p "How do I make coffee?"
```

Feed the same prompt to base and to SFT, and the difference is easy to see.

```text
Prompt: How do I make coffee?
  BASE : How do I make coffee? How do I make coffee? ...          (repeats the question)
  SFT  : Making coffee is a simple process... you'll need a coffee maker, ...

Prompt: Spell the word 'banana' letter by letter.
  BASE : The word 'banana' is made up of two letters. ...          (fails to spell)
  SFT  : banana:b,a,n,a,n,a
```

### Report card

```bash
python -m nanochat_jax.report generate
```

```text
| Metric          | BASE     | SFT      | RL |
|-----------------|----------|----------|----|
| CORE metric     | 0.269486 | -        | -  |
| ARC-Challenge   | -        | 0.509386 | -  |
| ARC-Easy        | -        | 0.622896 | -  |
| GSM8K           | -        | 0.100834 | -  |
| HumanEval       | -        | 0.140244 | -  |
| MMLU            | -        | 0.369534 | -  |
| SpellingBee     | -        | 0.996094 | -  |
| train bpb       | 0.731504 | -        | -  |
| val bpb         | 0.734295 | -        | -  |
| ChatCORE metric | -        | 0.373265 | -  |
```

End to end: 12h14m wall clock (12.19h billed), **$60.8** on spot (\~$263 on-demand), one preemption with a 17-minute recovery (RL is out of scope, so its column stays empty — kept to match nanochat's format). The CORE reproduction is done. The next two sections cover the missteps on the way there, and what they cost.

---

## 4. Five things I learned from the TPU/JAX port

The most important thing the port taught me: if you can avoid it, don't. I decided to do it anyway, so here are the problems I ran into and some insights of my own, trimmed down to five. Each item is tagged with which side it hurt: quality (CORE) or performance (MFU). The first is the bug that fooled us the longest.

**Insight 1. [Quality] Some bugs give wrong results without raising an error — data corruption from two lines of NumPy**

When quality collapsed, the first thing I suspected was the chip, and the second was the bf16 embeddings. But what had quietly gone wrong was not the model — it was **how the training loop fed the data**: no error, no NaN, no shape mismatch, the kind of bug that shows up only as lower quality. Below is the code in base_train.py that collects the grad-accum microbatches.

```python
idx_chunks, target_chunks = [], []
for _ in range(grad_accum_steps):
    inputs_np, targets_np = next(train_loader)
    idx_chunks.append(np.asarray(inputs_np))        # ← collects the views as-is (the bug)
    target_chunks.append(np.asarray(targets_np))
idx_all = jnp.asarray(np.stack(idx_chunks, axis=0))         # stacked later
targets_all = jnp.asarray(np.stack(target_chunks, axis=0))
```

Follow `train_loader` into dataloader.py, and you find `row_buffer` (line 112), which **reuses a single buffer** for speed and hands out **views** (zero-copy slices) of it. But the path above collects the microbatches in a list and calls `np.stack` **later**. Since `np.asarray` doesn't copy something that is already an array,[^numpygotcha] every collected microbatch ends up pointing at the **last contents** of the reused buffer — the batch for one step effectively becomes **N copies of the last microbatch**. That is data corruption: the token count stays the same, only the contents are wrong.

The original PyTorch code draws one microbatch, **immediately runs forward/backward to accumulate it into the gradients**, then draws the next — so the data is consumed before the buffer is reused, and the trap doesn't exist there. We, on the other hand, **collect all the microbatches and hand them to a single compiled step (`lax.scan`)** — and while they were being collected, the buffer got overwritten (why we needed this new path in the first place is explained by insight 2 below). The fix is one line: copy at capture time.

```python
    idx_chunks.append(np.array(inputs_np, copy=True))       # copy at capture time
    target_chunks.append(np.array(targets_np, copy=True))
```

The v6e runs from the period with this bug were stuck in a CORE band of 0.08–0.13; the run right after the fix scored 0.274 (that 0.274 is the published canonical run, separate from the speedrun's 0.2695 — both are R4-level). Also, the fix is already in main, so the commands you'd run from the speedrun section above are safe. (This data-feeding code actually had one more "sibling bug".[^siblingbug])

**If a dataloader can hand out views of a reused buffer, then on any path that stacks them later, the safe first move is to check whether they get copied.**

**Insight 2. [Performance] It's better to merge the training step into a single jit**

```python
compute_grads = jax.jit(compute_grads)          # compile unit 1
accumulate    = jax.jit(accumulate)             # compile unit 2
apply_update  = jax.jit(apply_update)           # compile unit 3

acc = zero_grads
for mb in microbatches:                         # DON'T — three programs chained by a Python loop
    acc = accumulate(acc, compute_grads(state, mb))
state = apply_update(state, acc)

@jax.jit                                        # DO — the whole step is one program
def train_step(state, microbatches):
    def micro_step(acc, mb):                    # same logic as above — only the boundary moves inside
        return accumulate(acc, compute_grads(state, mb)), None
    acc = lax.scan(micro_step, zero_grads, microbatches)[0]
    return apply_update(state, acc)
```

Port the PyTorch code as-is and you get the DON'T above. On GPU that structure is fine — asynchronous CUDA and `torch.compile` hide the host-side dispatch cost. On TPU, an A/B of the two structures moves MFU from 16% to 22%.[^probe]

The cause is the boundaries. XLA optimizes only within one jit program, so splitting the step into three programs leaves the device idle between programs (in our first implementation, idle time reached 26.7% of the step), and the compiler can't overlap operations across them either. The [MaxText docs](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/jax_xla_and_pallas.html) call wrapping the whole step in a single jit the "primary mechanism" behind their performance. Meanwhile, the first (manual) method in the [official optax grad-accum example](https://optax.readthedocs.io/en/stable/_collections/examples/gradient_accumulation_and_microbatching.html) shows this structure with no warning, so it's easy to copy if you miss the scan alternative further down the page.

**Rule: one compiled program per step. When performance is low, look at device idle time before you look at kernels (it shows in the [XProf](https://openxla.org/xprof) timeline) — if the idle time stands out, suspect the structure (the jit boundaries) first.**

**Insight 3. [Performance] The Splash attention kernel's defaults need tuning**

```python
# jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py
# TODO(apaszke,sharadmv): Select better parameters based on a heuristic.
block_q: int = 128          # TRAP — all 8 tile fields default to 128
block_kv: int = 128
block_q_dkv: int = 128      #   ...5 of them are backward-only
```
```bash
python scripts/base_train.py ... \
  --splash-block-q=512 --splash-block-kv=512 --splash-block-kv-compute=256   # DO — the values we use at seq 2048
```

Splash is the TPU flash-attention kernel that ships inside JAX. Eight tile-size fields decide its performance — open the [source](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py) and the defaults are all 128, with the authors' TODO still sitting right above them.

In an A/B that changed only the tiles, MFU went from 22.4% to 41.2%,[^probe] and the loss matched to six decimal places — exactly what the docstring says: tiles barely touch the numerics and mostly move performance. Most of the runtime sat in the backward pass. Why flash-attention's backward costs more than its forward is covered in the [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691); the trade-offs of Splash tiles are in [this write-up](https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas). [MaxText also uses its own 512 instead of the JAX defaults](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml).

The best values depend on the sequence length — in our seq-4096 experiments the optimum was 1024, and the published run (seq 2048) uses 512. [The sweep from tpuchat, an independent port,](https://github.com/vorushin/tpuchat/blob/master/pallas/CLAUDE.md) converged to the same 1024 for its shapes — and 2048 tiles were actually slower. Bigger is not automatically better.
**Since the loss isn't sensitive to the tiles (see the docstring), a tile sweep even before profiling is a reasonable first move.**

**Insight 4. [Quality] By default, TPUs compute fp32 matmuls in bf16**

Our training command carries a flag called `--lm-head-precision highest`. For a while I believed it was protecting quality; in reality, it does nothing. How that happened is tied to a TPU-specific behavior.

Request an fp32 matmul on a TPU, and even though the dtype stays fp32, under the default settings the multiplication [runs in bf16](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/_src/lax/lax.py#L2016). This is not hidden behavior — it's a documented default — and it's a trap well known enough that a Google researcher [wrote in an issue](https://github.com/jax-ml/jax/issues/7010) that his own team had been bitten by it for roughly the fourth time.

```python
a = np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32)
ref = a.astype(np.float64) @ a.T.astype(np.float64)
out = jax.jit(jnp.dot)(a, a.T)                 # fp32 inputs, default precision
assert np.abs(np.asarray(out, np.float64) - ref).max() / np.abs(ref).max() < 1e-5
# expected — passes on CPU; on TPU we expect it to fail, because of the default above
```

We actually hit this early in the port. One vocab projection amplified a 1.67e-6 input error by 197×, and a per-op `precision=highest` on that operation stopped the divergence. Up to that point, it was a genuine fix. The mistake came after — we carried the flag over into bf16 training as well, but precision settings act only on fp32 operations ([a maintainer's answer to exactly this question](https://github.com/jax-ml/jax/discussions/18938)). Our layers cast the weights to bf16 before the multiply, so there is no fp32 operation left for the setting to act on.

```python
x = jnp.asarray(a[:8], jnp.bfloat16)
w = jnp.asarray(a, jnp.bfloat16)               # weights get cast to bf16 before the multiply
assert jnp.all(jnp.dot(x, w, precision="default")
            == jnp.dot(x, w, precision="highest"))
# TRAP — this passes. The two paths are bit-identical = evidence the setting is a no-op
```

The quality gap happened to close in the same period, so the flag took the credit — but when I later retraced the commits, the most cleanly isolated cause was a different fix that landed alongside it: the missing grad-accum accumulation loop (the "sibling bug" from insight 1's footnote), not the precision flag. The flag stays in the command only to keep reproduction matched with the archived logs, and it's scheduled for removal in the next refactoring.
**For work that does need fp32 precision (reference comparisons, fp32 evaluation, and so on), this default can be a problem. See: [Cloud TPU bfloat16 docs](https://docs.cloud.google.com/tpu/docs/bfloat16), [PyTorch/XLA precision tutorial](https://docs.pytorch.org/xla/release/r2.8/tutorials/precision_tutorial.html).**

**Insight 5. [Performance] The best head_dim depends on the TPU generation — on v6e, multiples of 256 are recommended**

```python
# nanocode/configs.py — d24: n_embed 2048, n_head 8 → head_dim 256
# "adjusted for head_dim=256 to better feed the 256x256 tpuv6e systolic arrays"
```

Our model uses head_dim=128. [Parity with nanochat](https://github.com/tucan9389/nanochat-jax/blob/main/dev/PARITY.md) was the goal, so we kept the reference value as-is. But how fast that value runs depends on the chip generation. The TPU matrix-multiply unit (MXU) is [256×256 on v6e and Ironwood, 128×128 on the generations before](https://docs.cloud.google.com/tpu/docs/performance-guide), and for a matmul to reach peak throughput, [the dimensions need to be at least the MXU size](https://jax-ml.github.io/scaling-book/tpus/). Accordingly, [Google's model-design guide recommends sizing head_dim in multiples of 256 on Trillium (v6e)](https://maxtext.readthedocs.io/en/latest/guides/optimization/custom_model.html). head_dim=128 was an exact fit up through v5p; on v6e, it no longer fills the MXU.

We measured MFU holding the parameter count equal (729.8M) and changing only the head decomposition, running each config for a short 12 steps (MFU stabilizes within a few steps). Read the difference between the two rows rather than the absolute values[^probe]:

| shape | MFU |
|---|---:|
| 12 heads × 128 — nanochat as-is (our setting) | 30.1% |
| 6 heads × 256 — the v6e-recommended alignment | **35.2%** |

That works out to about 17% faster at the same parameter count. Whether all of that difference comes from MXU alignment, though, we haven't isolated at the level of individual operations — the accurate statement stops at "a result consistent with feeding the MXU better". [nanocode](https://github.com/salmanmohammadi/nanocode/blob/main/nanocode/configs.py#L8), a nanochat-based JAX/TPU project, chose 256 for the same reason (the snippet above).

For now we keep 128 for parity, but the architecture is open for later optimization work. One check remains first: we haven't A/B-tested whether quality (CORE) holds at the same parameter count — the speed is measured, the quality isn't.
**The same config can align with the MXU differently depending on the chip generation.**

Next, let me add up what all of this actually cost — missteps and trial and error included.

---

## 5. Why it cost about $2,500

Starting with almost no LLM pretraining experience, I spent three months and about $2,500 (TPU compute \~$2,350 + storage) getting to the $60.8 speedrun reproduction above. Even with coding agents doing a lot of the work, a bill this size honestly surprised me. The breakdown by category:

| Where it went | ~Cost | What went wrong |
|---|---:|---|
| Quality reproduction — the v5p period | ~$730 | Ran the quality reproduction on a [v5p-32, a slice that costs about 3× a v6e-8](https://cloud.google.com/tpu/pricing) — for each candidate missing recipe element (the grad-accum bug, matmul precision, and so on), a full train to confirm or reject, over and over |
| Quality regression after moving to v6e | ~$275 | The training loop rewritten for the v6e move (the fused structure of section 4, insight 2) carried the NumPy bug from insight 1, so every step's batch collapsed into copies of the last microbatch and CORE dropped sharply; I misattributed it twice — to the chip, then to bf16 embeddings — and it took ten days to find |
| Five operational incidents | ~$585 | Lost checkpoints, overfitting from too little data, billing on a dead node, a zero-byte ckpt crash, and a multi-host launcher incident — five in total; prevention went into code (atomic saves) and operating rules (Appendix C) |
| Performance (MFU) optimization | ~$375 | The work behind the published run's ~24% MFU (insights 2 and 3 of section 4 are its results). The big money went into two full trains launched for speed alone, without a quality gate |
| Publication rehearsal and regression hunting | ~$165 | Pre-release end-to-end runs fixed 3 problems that only show up in a full training run (a disk blow-up, eval attention, SFT OOM) |
| Finishing the final reproduction | ~$180 | The first v6e reproduction, ChatEval, and several final clean-clone runs — the last $60.8 speedrun reached CORE 0.2695, the level of nanochat's R4 |
| Port correctness checks and misc. | ~$40 | This went into checking that runs split across 1–16 chips produce the same results, plus evaluation; the PyTorch tensor comparisons finished on a laptop CPU, at no TPU cost |

In short: the port itself (the bottom row) cost $40, and the rest went into reproducing the reference CORE with that port. Early on, I believed this recipe only ran on v5p. The v5p settings as-is hit OOM on v6e-8's HBM (32GB per chip vs v5p's 95GB); my current guess is that the microbatch size was the cause — at the time I changed the batch, the embeddings, and the training loop all at once during the migration, so I never isolated which change was decisive. After seeing [nanocode's report of training a model of the same scale on a single v6e-8](https://github.com/salmanmohammadi/nanocode), I moved the batch and settings over to v6e, and one base training run dropped from \~$200 in the v5p days to \~$50, then down to \~$30 with the MFU optimizations (insights 2 and 3 of section 4). Today's $60.8 full run is that \~$30 base training plus tokenizer, SFT, and evaluation.

---

## 6. Wrapping up, and next steps

Thanks for following along this far. Quality has reached the same class on the CORE score; what remains is performance. MFU is up to about 24% (insights 2 and 3 of section 4 were the levers). Which of jit idle time, Splash tiles, head_dim alignment, or the attention HBM bottleneck dominates the remaining gap to Karpathy's 47–48% — that takes a roofline/XProf breakdown to answer, and that's the next piece of work. If you've dealt with this kind of problem before, I'd welcome your comments (MFU numbers from comparable cases are collected in the footnote[^mfu]).

This is a starting point. Change the depth and build your own model, swap in a different tokenizer and dataset to teach it the knowledge you want, or pick up the optimization where we stopped. The $300 credit on a new GCP account covers three or four full runs on spot at about $61 each — but for a new account, requesting v6e quota comes first (Appendix B). If you'd rather play with the model before training anything, the base and SFT checkpoints from this run are up on [Hugging Face](https://huggingface.co/tucan9389/nanochat-jax).

---

## 7. Further reading

**nanochat & LLM training basics**
- [Let's reproduce GPT-2 (124M) — Karpathy (video)](https://www.youtube.com/watch?v=l8pRSuU81PU) — trains GPT-2 (124M) from an empty file, through data loading, optimization, and multi-GPU.
- [Neural Networks: Zero to Hero — Karpathy (course)](https://karpathy.ai/zero-to-hero.html) — Karpathy's free course, including build GPT, the GPT Tokenizer, and makemore.
- [Transformers / Attention — 3Blue1Brown (chapters 5–6)](https://www.youtube.com/watch?v=wjZofJX0v4M) — a visual explanation of attention and transformers.
- [nanochat walkthrough — Discussion #1](https://github.com/karpathy/nanochat/discussions/1) — Karpathy's step-by-step writeup of the d20 speedrun.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — a \~300-line train.py that trains GPT; nanochat's simpler cousin (Karpathy's word).
- [modded-nanogpt — Keller Jordan](https://github.com/KellerJordan/modded-nanogpt) — where Muon came from, and the source of the squared ReLU, QK-norm, value embeddings, and logit softcap that nanochat uses.
- [Chinchilla paper](https://arxiv.org/abs/2203.15556) — argues for scaling model size and token count together under a compute budget; the framework our tokens:params ratio ([^ratio]) follows (our 9.5 is measured, not Chinchilla's 20).

**TPU & JAX**
- [scaling-book — Google DeepMind](https://jax-ml.github.io/scaling-book/) — a textbook on TPU rooflines, MFU, and parallelism.
- [TPU Deep Dive — Henry Ko](https://henryhmko.github.io/posts/tpu/tpu.html) — explains the 128×128 systolic MXU, the memory hierarchy, and multi-chip setups for people coming from GPUs.
- [Making DL Go Brrrr — Horace He](https://horace.io/brrr_intro.html) — how to tell whether a computation is compute-, memory-, or overhead-bound.
- [JAX AI Stack](https://jaxstack.ai/) — the official page for the stack (JAX, Flax, and others); source of Figure 1.
- [Pallas docs](https://docs.jax.dev/en/latest/pallas/index.html) · [Flax NNX docs](https://flax.readthedocs.io/en/latest/nnx_basics.html) — Pallas is the language for writing TPU kernels; Flax NNX is the neural-network API the model code uses.
- [When XLA Isn't Enough — Toulme](https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas) — follows the path from XLA down to a Pallas kernel, through TPU Splash attention.

**Neighboring projects**
- [nanocode](https://github.com/salmanmohammadi/nanocode) — a coding agent (a Claude Code clone) trained in pure JAX on a v6e-8. Same depth 24 as ours, but a head_dim 256 / seq 4096 setup, so its reported **52.5% MFU** is measured under different conditions than our 24% ([^probe]).
- [tpuchat](https://github.com/vorushin/tpuchat) + [the author's "$50" writeup](https://vorushin.github.io/blog/llm-pretraining-tpu-v6e-50usd) — trains **depth 8 / 164M** — about **1/8 our parameter count** — on a single **v6e-1** in Colab. **46.5% MFU**.
- [nor — modded-nanogpt in JAX/TPU](https://nor-blog.pages.dev/posts/2025-08-21-modded-nanogpt-jax/) — reports **15→23% MFU on a v6e-8**. In the modded-nanogpt GPT-2 speedrun line, d_model 1024 (depth and parameter count not stated).
- [rkstgr — d24 FP8](https://huggingface.co/rkstgr/nanochat-d24-speedrun) — the same **d24 / 1.4B**, run on **16×H100 with FP8** (GPU). Base CORE 0.247, **47.6% MFU**.

**Repository links**
- [nanochat-jax](https://github.com/tucan9389/nanochat-jax) — the code for this reproduction.
- [dev/PARITY.md](https://github.com/tucan9389/nanochat-jax/blob/main/dev/PARITY.md) — a parity table **comparing this port against nanochat R1–R6, axis by axis**.
- [dev/LEADERBOARD.md](https://github.com/tucan9389/nanochat-jax/blob/main/dev/LEADERBOARD.md) — a table of scores and costs for our runs.
- [Full training log](https://github.com/tucan9389/nanochat-jax/releases/download/row2/31_base_train.log) — the training log the numbers in this post come from; the header includes flops/token and peak TFLOPS.
- [Hugging Face checkpoints](https://huggingface.co/tucan9389/nanochat-jax) — the base and SFT weights (load and use them without training anything yourself).

---

*Thanks to [Jaewook Kang](https://www.linkedin.com/in/jaewook-kang-3a4217b9/), [Joosung Yoon](https://www.linkedin.com/in/joosung-yoon/), [Taekmin Kim](https://www.linkedin.com/in/taekminkim/), [Xing Liu](https://www.linkedin.com/in/xingliu14/), and one anonymous friend for reviewing this — and to my wife, for the budget and the time.*

## Appendix

<details>

<summary>Appendix A) Setup — from TPU allocation to environment</summary>

Allocate a v6e-8 TPU. `--spot` requests a spot instance.

```bash
gcloud alpha compute tpus tpu-vm create <N> \
  --zone=<Z> --accelerator-type=v6e-8 --version=v6e-ubuntu-2404 --spot
```

Once `READY` appears, you can connect. It normally shows up within a few minutes, and you'll know about a failure within a minute.

From `READY` on, you're being billed \~$5/hour. Connect right away. (a) is an interactive shell; (b) and (c) are for one-shot commands and background runs.

> A new project may have a TPU quota of 0. Check and request v6e quota for your target region on the console's Quotas page first — as Appendix B shows, quota was the #1 cause of failed allocations.

```bash
# (a) interactive shell — for quick commands
gcloud compute tpus tpu-vm ssh <N> --zone=<Z>

# (b) one-shot command — put multi-line scripts in the REMOTE variable via a quoted heredoc
REMOTE=$(cat <<'EOF'
<script to run on the VM>
EOF
)
gcloud compute tpus tpu-vm ssh <N> --zone=<Z> --ssh-flag="-oConnectTimeout=30" --command="$REMOTE"

# (c) long-running command — detach from the session with setsid nohup and run in the background
REMOTE=$(cat <<'EOF'
setsid nohup bash -c '<script to run on the VM>' > ~/logs/command0.log 2>&1 &
echo LAUNCHED
EOF
)
gcloud compute tpus tpu-vm ssh <N> --zone=<Z> --ssh-flag="-oConnectTimeout=30" --command="$REMOTE"
```

Once you're on the machine, clone the nanochat-jax repository and install the dependencies.

```bash
sudo rm -f /tmp/libtpu_lockfile   # required before the first jax process
git clone https://github.com/tucan9389/nanochat-jax.git
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 ~/venv && source ~/venv/bin/activate
uv pip install "torch~=2.11.0" --index-url https://download.pytorch.org/whl/cpu
cd ~/nanochat-jax && uv pip install -e ".[tpu,dev]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -c "import jax; print(jax.devices())"
```

If you see eight TPU v6 lite devices, setup is done.

</details>

<details>

<summary>Appendix B) History and statistics of trying to allocate TPU spot instances</summary>

TPU spot attempt statistics by region, April–July 2026. This is an excerpt of the main chip–region combinations, so the rows add up to less than the full statistics (2,885 creation attempts, 1,632 failures).

| Chip | Region | Attempts→successes | Failure causes | Slice price (spot, May/Jul) | Actual usage |
| --- | --- | --- | --- | --- | --- |
| v6e-8 | us-east4 | 763→**0** | quota 0 ×738 | $3.39/$2.90·h | — |
| v6e-8 | **us-central1** | 575→**208 (36%)** | capacity ×321, quota ×34 | $4.12/$4.99·h | **150.1h $659, 92 preemptions** |
| v6e-8 | asia-southeast1 | 222→**0** | capacity ×176, zone unsupported ×46 | **$2.14·h (cheapest)** | — |
| v6e-8 | us-west1 | 110→0 | capacity ×105 | $4.12·h | — |
| v6e-8 | **asia-northeast1** | 88→**85 (97%)** | capacity ×1 | **$8.74·h (2.1×)** | 20.0h $175, 3 preemptions |
| v6e-8 | us-east5 | 29→0 | capacity ×19, quota ×10 | $10.21·h | — |
| v6e-8 | europe-west4 and others | 28→0 | mixed | $11.10·h | — |
| v6e-32 | all 6 regions | 107→**0** | **all quota 0** | ($8.55+·h) | tried on May 3 and May 5, then gave up |
| v5p-32 | us-east5 | 58→10 (17%) | quota ×40, capacity ×8 | **$25.54·h** | 27.0h **$691**, 1 preemption |
| v5p-32 | **us-central1** | 23→**23 (100%)** | — | **$13.24·h** | 16.3h $216, 2 preemptions |
| v5p-32 | europe-west4 | 4→2 | zone unsupported ×2 | $20.50·h | 7.4h $153 |
| v5p-32 | us-east1/us-east4 | 15→0 | quota / zone unsupported | — | — |
| v5p-8 | us-east5 | 27→**27 (100%)** | — | $6.38·h | 19.2h $123, 3 preemptions |
| v5p-8 | europe-west4 | 4→3 |  | $5.12·h | 33.8h $173 (31.4h, the longest run) |
| v5p-16 | us-east5 | 13→4 | quota ×5, capacity ×3 | $12.77·h | 4.7h $60 |
| v6e-1 | europe-west4 | 17→**17 (100%)** | — | $1.39·h | 6.5h $9, **6 preemptions** |
| v6e-1 | us-east5 | 22→6 | capacity ×11, quota ×5 | $1.28·h | 0.7h $1 |
| v6e-16 | asia-northeast1/ew4 | 8→6 | quota ×2 | $17.5–22.2·h | 1.8h $36 |

</details>

<details>

<summary>Appendix C) Cheatsheet of GCP commands</summary>

> Placeholders: `<N>` = TPU name, `<Z>` = zone (e.g. us-central1-b), `<BUCKET>` = GCS bucket, `<REGION>` = bucket region (same region as the TPU zone). In a non-interactive shell on a Mac, run `export PATH=/Users/<you>/google-cloud-sdk/bin:$PATH` first.

**In your local machine (Mac)**

```bash
# allocate TPU — sweep zones until you land on one with capacity (success rates and prices: Appendix B)
for z in us-central1-b us-central1-a us-central1-c us-east4-a asia-southeast1-b; do
  gcloud alpha compute tpus tpu-vm create <N> \
    --zone=$z --accelerator-type=v6e-8 --version=v6e-ubuntu-2404 --spot --quiet && break
done

# check the status of machine
gcloud compute tpus tpu-vm describe <N> --zone=<Z> --format='value(state)'   # READY / PREEMPTED
gcloud compute tpus tpu-vm list --zone=<Z>

# run a command on the TPU machine
gcloud compute tpus tpu-vm ssh <N> --zone=<Z>                    # interactive shell
gcloud compute tpus tpu-vm ssh <N> --zone=<Z> --command="<CMD>"  # one-shot command (for multi-line scripts, see heredoc patterns (b)(c) in Appendix A)
gcloud compute tpus tpu-vm scp <N>:~/logs/31_base_train.log ./logs/ --zone=<Z>   # retrieve logs

# deallocate the TPU machine (after deleting, sweep every zone you tried for zombie instances — create is async)
gcloud compute tpus tpu-vm delete <N> --zone=<Z> --quiet
for z in us-central1-b us-central1-a us-central1-c us-east4-a asia-southeast1-b; do
  gcloud compute tpus tpu-vm list --zone=$z 2>&1 | grep -v "Listed 0"
done

# anything else?
gcloud storage buckets create gs://<BUCKET> --location=<REGION>                        # same region as the TPU zone
gcloud storage rsync -r gs://<BUCKET>/base_checkpoints /path/to/ssd/base_checkpoints   # permanent mirror of the results
gcloud auth login; gcloud config set project <PROJECT-ID>                              # first time only
```

**In your TPU instance (VM)**

```bash
# core command within speedrun.sh
sudo rm -f /tmp/libtpu_lockfile            # required before the first jax process
bash runs/speedrun.sh                       # the production pipeline in one shot
bash runs/runcpu.sh                         # d2/CPU wiring check (only checks it runs to the end)
# the stages speedrun.sh runs in order (same as Steps 1-3 in the main text):
python -m nanochat_jax.dataset -n 8                                                    # B-1 for the tokenizer
python -m nanochat_jax.dataset -n 170 &                                                # B-2 for training (background)
python -m scripts.tok_train --max-chars 100000000 --doc-cap 10000 --vocab-size 32768   # B-3
python -m scripts.tok_eval                                                             # B-4
python -m scripts.base_train  <Step 2 arguments from the main text>                    # B-5
python -m scripts.base_eval --source base --model-tag d24_speedrun_r4 --eval bpb,core ...   # B-6
python -m scripts.chat_sft    <Step 3 arguments from the main text>                    # B-7
python -m scripts.chat_eval -i sft -g d24_speedrun_r4_sft ...                          # B-8
python -m nanochat_jax.report generate                                                 # B-9

# monitoring base training — the GCS checks and preemption recovery below only work if you mirror checkpoints to <BUCKET> (periodic gcloud storage rsync recommended)
tail -5 ~/logs/31_base_train.log | grep -E 'step|bpb'   # last step lines
ps -ef | grep '[b]ase_train'                            # process alive ([b] avoids grep matching itself)
df -h / | tail -1                                        # disk
gcloud storage ls -l gs://<BUCKET>/base_checkpoints/d24_speedrun_r4/ | tail -8   # latest ckpt in GCS

# monitoring SFT training
tail -5 ~/logs/60_sft_chain.log | grep -E 'step|val_bpb'
ps -ef | grep '[c]hat_sft'

# anything else? — preemption recovery (restore from GCS, then continue)
# training: python -m scripts.base_train ... --resume-from-step <N>
# eval: python -m scripts.chat_eval  ... --start-index <number of completed problems>
```

</details>

<details>

<summary>Appendix D) Full tok_eval compression tables</summary>

diff % = (baseline tokens − our tokens) ÷ baseline tokens. Positive means we fit the same bytes into fewer tokens (a compression advantage). ratio = bytes/token.

**vs GPT-2**

```text
| Text    |   Bytes | GPT-2 tok | ratio | Ours tok | ratio |  diff % |
|---------|---------|-----------|-------|----------|-------|---------|
| news    |    1803 |       392 |  4.60 |      406 |  4.44 |   -3.6% |
| korean  |     893 |       745 |  1.20 |      729 |  1.22 |   +2.1% |
| code    |    1259 |       576 |  2.19 |      410 |  3.07 |  +28.8% |
| math    |     690 |       348 |  1.98 |      338 |  2.04 |   +2.9% |
| science |    1098 |       253 |  4.34 |      238 |  4.61 |   +5.9% |
| train   | 2948778 |    631304 |  4.67 |   621881 |  4.74 |   +1.5% |
| val     | 3024593 |    653067 |  4.63 |   645862 |  4.68 |   +1.1% |
```

**vs GPT-4**

```text
| Text    |   Bytes | GPT-4 tok | ratio | Ours tok | ratio |  diff % |
|---------|---------|-----------|-------|----------|-------|---------|
| news    |    1803 |       387 |  4.66 |      406 |  4.44 |   -4.9% |
| korean  |     893 |       364 |  2.45 |      729 |  1.22 | -100.3% |
| code    |    1259 |       309 |  4.07 |      410 |  3.07 |  -32.7% |
| math    |     690 |       295 |  2.34 |      338 |  2.04 |  -14.6% |
| science |    1098 |       245 |  4.48 |      238 |  4.61 |   +2.9% |
| train   | 2948778 |    611619 |  4.82 |   621881 |  4.74 |   -1.7% |
| val     | 3024593 |    631183 |  4.79 |   645862 |  4.68 |   -2.3% |
```

</details>

---

[^ratio]: The \~6.9B tokens come from the tokens:params ratio nanochat uses. It follows the compute-optimal ratio concept established by Chinchilla (the params–tokens combination that gives the lowest loss for a given compute budget), but the value itself is nanochat's own measurement, not Chinchilla's 20: an early scaling fit of \~8 ([miniseries #420](https://github.com/karpathy/nanochat/discussions/420)) was later re-measured to a compute-optimal \~10.5, which made the script default 12 ([#481](https://github.com/karpathy/nanochat/discussions/481)), and R4 uses 9.5, slightly undertrained relative to that. Karpathy's guess for why the value comes out below 20 is the influence of the Muon optimizer, among other things (#420). The ratio multiplies the non-embedding (scaling) parameters (d24 ≈ 730M): 730M × 9.5 ≈ 6.9B tokens. Of the [ClimbMix shards](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle), training uses about 150; 171 is the download count with some margin (170 train + 1 validation).

[^vocab]: The common belief that "rounding the vocab up to a hardware-friendly multiple makes it faster" comes from a GPU-era tweet by Karpathy ([50257→50304, rounded up to a multiple of 64, \~25% faster via kernel occupancy](https://x.com/karpathy/status/1621578354024677377)). On TPU, XLA automatically pads matrix dimensions to multiples of the MXU tile (128 on v5p, 256 on v6e) ([Cloud TPU performance guide](https://cloud.google.com/tpu/docs/performance-guide)), and 32768 is already such a multiple, so nothing is wasted on padding. On how vocab size relates to model scale, the vocab scaling law ([Tao et al. 2024](https://arxiv.org/abs/2407.13623): larger models deserve larger vocabularies, but sublinearly) is a good read.

[^splash]: Block sizes are for performance and have almost no effect on numerics (Pallas `BlockSizes` docstring). The hard constraints mostly live in the Pallas splash kernel (JAX 0.10.0, the version we use), not in our code — `block_kv` must be an integer multiple of `block_kv_compute`, and `block_kv_compute` must be a multiple of 128 (=`NUM_LANES`) ([kernel L967–970](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L967-L970)); `block_q` and `block_kv` must also be multiples of 128 ([L667](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L667) and [L673](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L673)). Blocks must be ≤ `seq_len` ([L1446–1449](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py#L1446-L1449)) and must fit in on-chip VMEM (exceeding it gives a compile/allocation error — splash itself doesn't check this). The only constraint our wrapper enforces directly is `block_kv_compute % 128` ([attention.py#L82-L85](https://github.com/tucan9389/nanochat-jax/blob/c01dd82e8e8bdaae616322e6fc02a33001fc370d/nanochat_jax/attention.py#L82-L85)). [MaxText's default](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml#L1085) is also 512 (our starting value), and larger models raise only `block_q` ([gemma2-9b on v6e uses `sa_block_q=2048`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/tpu/v6e/gemma2_9b.sh); `block_kv` stays at 512). Tune with [XProf](https://openxla.org/xprof) — the splash kernel tags its block sizes into the trace as XProf metadata.

[^mfu]: In his homelab d24 reproduction ([#710](https://github.com/karpathy/nanochat/discussions/710)), matt-langston reports his DGX Spark at 29.6% MFU and puts the general H100 band at 50–65% (no depth given; the same thread also explains the difference between GPU utilization and MFU). Karpathy's own measured anchors are \~47–48% for the d20 speedrun and 47.40% for d34. [nor](https://nor-blog.pages.dev/posts/2025-08-21-modded-nanogpt-jax/), who ported modded-nanogpt to the same v6e-8, also landed at 15→23%, and the GPT-3 training MFU in the [PaLM paper](https://arxiv.org/pdf/2204.02311)'s table is 21.3%. So our 24% is not far outside this distribution. That said, these numbers differ in seq_len, model architecture, and how FLOPs are counted, so they can't be compared head-on — MFU was originally a metric for comparing system efficiency on the same model ([PaLM §4.1](https://arxiv.org/abs/2204.02311)).

[^scaling]: For scaling and rooflines on TPU, Google DeepMind's [scaling-book](https://jax-ml.github.io/scaling-book/) is a textbook-level resource.

[^probe]: The MFU comparisons in this section were measured on a comparison config aligned with nanocode (head_dim 256, seq_len 4096, value embeddings removed), to isolate just the structure and tile differences — which is why the absolute values differ from the published run (seq 2048, head 128, \~24% MFU). On this config, MFU climbed loop 16.2% → fused 22.4% → Splash tile tuning 41.2%, and the tiles (insight 3) were the bigger lever. The two structures in insight 2 can be reproduced with `--grad-accum-impl loop|fused` in our repo. The 52.5% nanocode reports on the same chip adds call-path and layout optimizations on top of this config, so it isn't directly comparable with our 24% ([nanocode](https://github.com/salmanmohammadi/nanocode/discussions/1)). The two MFU values in insight 5 (30.1% and 35.2%) were measured at seq 2048, the same as the published run.

[^costnote]: GPT-2's \~$43k is Karpathy's retrospective estimate (TPU v3 ×32 × 168h). The R4 row's \~2h and \~$48 are my estimate: a 2-hour base train loop times \~$24/hour for 8×H100. The \~$73 total is from Karpathy's [2026 tweet](https://x.com/karpathy/status/2017703360393318587) (GPT-2-grade d24, 8×H100, 3.04h) — that run is the default d24 (data-ratio 12), so it's the same d24 as our R4 (ratio 9.5) but a separate recipe. Our precise CORE value is 0.269486 (see the report card); the text uses 0.2695.

[^identity]: Karpathy, "Guide: infusing identity to your nanochat" ([nanochat #139](https://github.com/karpathy/nanochat/discussions/139)) — mixing 1,000 synthetic identity conversations into the midtraining/SFT mixture.

[^abilities]: The general way to add narrow abilities like this via midtraining/SFT is laid out in Karpathy's "counting r in strawberry (and how to add abilities generally)" ([nanochat #164](https://github.com/karpathy/nanochat/discussions/164)).

[^numpygotcha]: `np.asarray` not copying an input that is already an ndarray is [NumPy's documented behavior](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html) — a slice is a view sharing the same buffer, and changes through a view show up in the original. NumPy inside data pipelines is a well-known source of silent bugs: Tanel Pärnamaa downloaded and audited 100K+ GitHub repositories that import PyTorch, and reported that among the repos combining a custom dataset, NumPy's random number generator, and multi-process loading, over 95% had a related bug (PyTorch's official tutorials, OpenAI, and NVIDIA code included; [post](https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/)). Not exactly the same bug as ours, but the same family.

[^siblingbug]: An earlier version of the port had a "sibling bug", too — the grad-accum accumulation loop was missing entirely, so the model trained on only **1/N of the configured tokens** while the LR schedule decayed as if full batches were being consumed (silent under-training — the early loss looks normal, and only the final quality suffers). Upstream nanochat **derives** `grad_accum_steps` from `total_batch_size` and **asserts** the two divide evenly, which blocks this kind of mismatch at the source. Grad-accum is just an easy place for things to silently go wrong: HuggingFace and Unsloth also shipped grad-accum bugs (different mechanism — loss normalization) for a while in 2024 before fixing them ([Unsloth](https://unsloth.ai/blog/gradient) and [HuggingFace](https://huggingface.co/blog/gradient_accumulation)), and it's a known point of confusion in nanoGPT as well ([#340](https://github.com/karpathy/nanoGPT/issues/340)).
