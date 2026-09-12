# From n-gram speculative drafters to a count-based n-gram mixer — what the "flash scan" idea is worth here

Research pass on 2026-09-12 (local box, no hardware). Prompted by the user's line of thought:
shrink a speculative draft model until it becomes a Markov chain / n-gram table built by one
streaming pass over the corpus. The pasted Google AI Mode answer describes model-free n-gram
speculative decoding (prompt-lookup style). This document covers (A) that technique and why
it is worth exactly zero for val_bpb, and (B) the one form in which the same corpus statistics
*can* move val_bpb: a frozen count-based n-gram distribution mixed into the scored forward,
which the plan lists as T4.1 and nobody has tested. Along the way it corrects the evidence base
the plan cited for T4.1.

## 0. Short answer

1. **As a drafter, an n-gram table is irrelevant.** Prompt-lookup / n-gram speculative decoding
   (vLLM `ngram`, SGLang, ANPD, Token Recycling, SuffixDecoding) accelerates *generation* by
   proposing tokens that the same model then verifies. Verification leaves the model's
   distribution unchanged by construction, and `evaluate_bpb` never generates. Zero effect on
   the metric, same as DSpark, Medusa and EAGLE (§1).
2. **As a mixer, the same statistics are a legitimate and untested bpb lever.** A frozen bigram
   (and possibly hashed trigram) table counted from the training shards *inside* train.py,
   stored in `final.pt`, and blended into the output distribution in `forward`, satisfies every
   rule I can find: the forward stays pure, stateless and causal; the table is an asset
   "produced within your own training loop on the provided data"; nothing about the validation
   data is read (§2).
3. **The evidence the plan leaned on for this is contaminated and must be downgraded.**
   Parameter Golf's spectacular n-gram results (0.40–0.55 bpb) came from *eval-time caches
   that update on the scored text* (illegal here: stateless forward, no inference from
   validation data) **and** a scoring bug: hashed caches normalised only the correct token's
   count. The honest rerun (#978) shows a properly normalised n-gram alone at 1.51 bpb, worse
   than the neural model; 33+ PRs were invalidated on 2026-03-27. The "+0.080 within frontier"
   figure in the *What Really Works* paper is for those eval-time methods and never
   distinguishes frozen training-only tables (§3.1).
4. **The clean evidence that remains is still positive but modest.** kNN-LM: retrieving from
   the model's *own* training set cuts WikiText-103 perplexity 18.65 → 16.12 (−13.5%) for a
   247M model that had already trained on that data for many epochs. Infini-gram: GPT-Neo 125M,
   pretrained on the Pile, still gains 22% perplexity from a Pile-train ∞-gram (with a 360B-token
   index, ~1000× our corpus). This repo's own T3.1: *learned* hashed bigram tables gave a real
   −0.020 to −0.024 bpb per-step quality gain that was eaten entirely by their device-side
   training cost. A count-based mixer has the same source of gain (n-gram memory the 56M
   backbone cannot store after 0.9 epochs) with the cost moved to the host CPU (§3.2–3.3, §5).
5. **Expected value:** +0.005 to +0.015 bpb improvement if the offline test in §6 succeeds,
   at ~0 device training cost, a few ms per eval row, and 134 MB (bigram, fp16) in the
   checkpoint. Failure mode is a clean zero. This is worth more than any remaining Tier 3 item
   and is cheap to falsify offline. Ask the organisers the T4.1 rules question in parallel;
   do not spend a submission before an answer (§6–§7).

## 1. Part A — n-gram speculative decoding (the thing the AI Mode answer describes)

- **Prompt-lookup decoding** (Saxena, 2023) and its production forms: vLLM's `ngram`
  speculative method (`num_speculative_tokens` ≈ 5, `prompt_lookup_max` ≈ 4), SGLang's
  n-gram speculator. Draft tokens are copied from the longest n-gram match in the prompt and
  the text generated so far; the target model verifies them in one parallel pass. Reported
  acceptance ≈ 65–70% on mixed workloads, 80%+ on repetitive text; throughput gains of
  1.1–1.3× on general chat, several × on summarisation/code editing (vLLM issue #16258,
  Snowflake Arctic Inference post).
- **Corpus-statistics variants**: ANPD (adaptive n-gram parallel decoding), Token Recycling
  (tree of previously seen continuations), SuffixDecoding (NeurIPS 2025; suffix tree over
  prior outputs). All model-free, all verified by the target model.
- The AI Mode answer's numbers ("under 5,000 parameters", "3–4 orders of magnitude smaller
  than a 4.8M neural companion") are about drafter storage. They are plausible for a trimmed
  bigram cache and irrelevant to us.

Why it is worth zero here: speculative decoding is *lossless by design*. Accepted drafts are
exactly the tokens the target would have sampled; rejected ones are corrected. The target's
`P(next | context)` never changes, and `evaluate_bpb` scores exactly that quantity from a
single forward on the given text. There is no generation step in Phase 1 scoring, and Phase 2's
CORE eval also runs through the same `load_for_eval` forward.

## 2. Part B — the same statistics inside the scored forward: rules check

Relevant rule text (README, byte-identical to upstream):

- "pure, stateless, causal forward — logits must be a function of the input tokens only
  (independent of batch size, world size, rank, and batch content; no collectives in forward;
  frozen, no hooks)". A frozen count table read by `forward` is a function of the input tokens
  only. **Any table that updates during evaluation is forbidden** — this is what rules out the
  Parameter Golf cache style.
- "you may only use assets produced within your own training loop on the provided data". A
  table counted from the training shards during the 30-minute run qualifies. A table built from
  anything else does not.
- "Reading, mirroring, or inferring the validation data used for scoring" is forbidden; a
  training-shard table reads none of it.
- "Model size is fully open ... Ties are broken by (1) fewer total parameters, then (2) simpler
  code." A 8192×8192 table is 67M entries; whether counts in a buffer are "parameters" is the
  organisers' call, and it only matters on an exact tie.
- `evaluate_bpb` applies `F.cross_entropy` to whatever `model(x)` returns (truncated to the
  vocab, cast to fp32). If `forward` returns `log(P_mix)`, `log_softmax` leaves it unchanged
  because the mixture already sums to one. So the mixer is expressed purely as a different
  logits function, no harness change.
- `causality_check` flips tokens after the midpoint and compares earlier logits; a mixer that
  uses only tokens ≤ t passes trivially.

Open question to put to the organisers (plan T4.1 already lists it): is a hybrid neural +
count-based n-gram language model, with tables built from the training shards inside the
budget and frozen at eval, an acceptable "model"? Phrase it that way and cite kNN-LM /
Infini-gram. Also ask whether counting in the excluded startup window is acceptable (T4.3);
the design below does not need it.

## 3. Evidence on how much a frozen n-gram mixer helps a small neural LM

### 3.1 Parameter Golf — downgrade this evidence

- Issue #140 "Tier 5: N-gram Cache (Invalidated Mar 27)": "hashed implementations scored
  only the correct token without full-vocabulary normalization. #978 proved correctly
  normalized n-gram achieves only 1.51 BPP (worse than neural baseline). 33+ PRs closed".
  PR #978's rerun: neural sliding-window 1.147 bpb; the "corrected" n-gram path 1.513.
- The surviving top entries (#1094 at 0.4027, #803 at 0.4416) are "causal sequential chunk
  scoring (score-first, update-after)" caches: tables *learn the validation text as it is
  scored*. Legal under PG's backward-looking rule, illegal here (stateless forward; no
  inference from validation data). Their gains say nothing about a frozen train-only table.
- *What Really Works?* (arXiv 2607.01517) reports N-gram Backoff (Eval) at +0.080 within the
  frontier but "never specifies whether the backoff tables are built from training data or
  updated from validation text during scoring" and does not mention the normalisation
  invalidation. The plan's T4.1 bullet that cites this figure should be read as "eval-time
  adaptive caches help", not as evidence for our setting.
- "Complementary Training" (#803: down-weight tokens a bigram predicts well so the neural
  model specialises) is a real idea, but its payoff there came from a strong eval-time cache
  taking 20–75% of the mixture weight. With a weak frozen table it would mostly make the
  neural model worse on exactly the tokens the table handles poorly. Not for now.

### 3.2 Clean, comparable evidence

| source | neural model | n-gram/memory source | effect | relevance |
|---|---|---|---|---|
| kNN-LM (Khandelwal et al., ICLR 2020) | 247M Transformer, WikiText-103 | its **own 103M-token training set**, λ≈0.25 | PPL 18.65 → 16.12 (−13.5%) | closest analogue: memory of the training data the model already saw, added at eval |
| Infini-gram (Liu et al., 2024) Table 1 | GPT-2 117M (not Pile-trained) | Pile-train ∞-gram, 360B tokens | PPL 22.82 → 13.71 (−42%) | index 1000× larger than ours and off-distribution for the model; upper bound |
| Infini-gram Table 1 | GPT-Neo 125M (**Pile-trained**) | same Pile-train index | 13.50 → 10.76 (−22%) | model already trained on the index data and still gains; gains shrink with model size (GPT-2 42% → 33% at 1.6B) |
| N-Grammer (Roy et al., 2022) | C4 Transformer | learned latent n-grams | ~10–15% PPL | learned, not counted; same mechanism |
| *N-gram Is Back* (Li et al., EMNLP 2022) | Transformer-XL class | fixed n-gram LM as residual base | consistent gains on LM/MT/summarisation | fixed table + neural residual is exactly the frozen-mixer structure |
| **This repo, T3.1** (notebook 2026-09-11/12) | the locked 56M model | learned hashed bigram VE tables, 3×262144 rows | −0.0235 bpb per step at 300 steps, −0.0217 at 844; extrapolated −0.020 at 2200; **cost −22.9% steps → net −0.0004** | the gain is real and slowly decaying; the failure was throughput, not quality |

Common thread: the gain is "memory the parameters cannot hold", largest for small,
undertrained models, and it does not require the memory source to be new data. Our model has
56M parameters and one pass over 316M tokens; it is the archetype.

### 3.3 Translating to this model

- The T3.1 per-table arithmetic gives the size of the n-gram signal on *this* data with *this*
  model: a bigram table was worth 0.0078 bpb per table (three tables, diminishing) and a trigram
  table 0.0022, before cost. An output-side mixer cannot be assumed to capture all of what an
  attention-side value embedding captures, but it also carries none of the training cost.
- Aster's best-known result on this task (0.9098) explicitly includes an **output bigram logit
  table**, `nn.Embedding(V, V)` indexed by the current token, zero-init and per-token gated,
  added to the logits (plan §11.1). That is the *learned* version of exactly this component,
  credited by a team at the top of the equivalent leaderboard. This repo tested VE tables (T3.1)
  but never the output table.
- Working estimate: 0.005–0.015 bpb from a bigram-only frozen mixer with a learned or
  entropy-adaptive gate; trigram adds less and costs memory. Compare: the scorer's 1024-context
  penalty is 0.016, the entire Days 2–4 block was 0.0044, Tier 3 items are 0.002–0.006.

## 4. Design — count on the host, store in the checkpoint, mix in `forward`

**Counting (zero device cost).** train.py already owns the packer (`make_dataloader_requeue`,
rows land in `cpu_inputs`/`row_buffer` on the host before the device copy). Per step, on each
rank, `np.bincount(prev*8192 + cur, minlength=8192**2)` over the rank's 32,768 tokens costs
~1 ms and overlaps device compute. Doc boundaries: skip pairs that cross a BOS if
`USE_DOC_MASK` semantics are wanted, or count through them; try both offline. At the end,
one `all_reduce` of an int32 `[8192, 8192]` tensor (268 MB, once) merges the four ranks, or
`gather_object` on the host. Store as fp16 log-probabilities after smoothing (134 MB) in
`final.pt`; with 0.9 epochs, the counts cover essentially the whole corpus.

**Smoothing.** Bigram: absolute discounting / interpolated Kneser–Ney against the unigram;
both are a few numpy lines on the dense table. Trigram (optional, later): hashed contexts to
2^22 buckets with top-K=16 continuations, stupid backoff to the bigram; build from `np.unique`
on 64-bit keys in a background thread during the last minutes or skip entirely for v1.

**Mixing in `forward` (eval path; also usable in training if the gate is learned).**
`p_n = table[x_t]` (a `[T, V]` gather), `p_mix = (1−g)·softmax(logits) + g·p_n`, return
`log(p_mix)`. Gate options, in order of preference:
1. *Learned per-token gate* (Aster style): `g = σ(w·h_t + b)`, `b` init −2 so the mixer starts
   near off, trained jointly with the model so the neural part learns to rely on it. Adds one
   `d→1` linear (641 params). Requires the table to exist during training: either count in
   the first ~100 steps and freeze (the counts converge fast at the bigram level), or use
   the previous run's table — the latter is external data and **not allowed**. First option.
2. *Entropy-adaptive fixed rule*: `g = f(H(softmax(logits)), count(x_t))`, no training; fit
   `f` offline (§6) on train-shard-held-out text, never on the public val shard.
3. *Global λ*: simplest; kNN-LM got most of its gain with a fixed λ≈0.25.

**Eval cost.** Gather + two elementwise passes over `[T, 8192]` fp32 per row; on the Trn2
eval of 20M tokens this is seconds, far inside the 20-minute eval timeout. Checkpoint grows
by 134 MB; `load_for_eval` loads it as a buffer (`persistent=True` this time).

**What not to build.** Anything that updates the table at eval time (illegal). Anything hashed
that normalises only the correct token (the PG bug: the mixture must be a proper distribution
over all 8192 tokens, which the dense bigram table gives for free). Within-row "token tilt"
caches (PG measured −0.00014 once made causal; attention already does this).

## 5. Why this is different from T3.1, which was discarded

| | T3.1 hashed n-gram VE (discarded) | count-based output mixer (proposed) |
|---|---|---|
| where the n-gram enters | attention values, every VE layer | output distribution only |
| how it is learned | gradient descent on 3×262144×128 bf16 tables | counting on the host |
| device training cost | −22.9% steps (embedding_dense_backward O(rows)) | ~0 (a `[T,V]` gather if the gate is learned; nothing if not) |
| checkpoint growth | +100M params | +67M counts (134 MB fp16) |
| measured/expected quality | −0.020 to −0.024 per step (measured) | unknown; 0.005–0.015 (estimate) |
| tie-break exposure | +100M "parameters" | +67M table entries, status to confirm |

The two are also complementary in principle: T3.1 helps the model *use* n-gram context
internally; the mixer fixes the *output* calibration on memorisable continuations. But T3.1
is priced out on this hardware regardless.

## 5a. Update, 2026-09-12 evening — T3.1 is no longer discarded (other agent's result)

While this document was being written, the box-side agent found the T3.1 overhead's cause
and reversed the verdict (commits `1eede36`, `39c5e0a`; notebook "T3.1 overhead hunt, part
2"). The ~15 ms/table cost was a **layer-position** effect of the whole-graph schedule, not
the table size, gather, modulus, gate or up-projection: a table on the *last* layer only
(`NGRAM_VE_LAYERS=-1`) makes the step *faster* than baseline (r=0.967–0.983) at any table
size, while tables on layers 1 or 3 cost +5 to +48 ms. Paired 900 s A/B, same box, same day:

| arm | bpb | steps | Δ |
|---|---|---|---|
| control | 1.081306 | 1094 | — |
| bigram, layer −1, 524288 rows × 128 | 1.068101 | 1099 | **−0.0132** |
| bigram + trigram, layer −1, 262144 rows each | 1.066917 | 1084 | **−0.0144** |

Causality exact on all arms; 1800 s escalation pending. Two consequences for this document:

1. **The learned hashed n-gram value embedding is now the leading n-gram lever**, not a
   discarded one, and it is rules-safe today (plain parameters trained by gradient descent;
   T4.2). §5's comparison table should be read with T3.1's cost column at ≈0 and its measured
   quality at −0.013 to −0.014 per step at 900 s.
2. **The count-based output mixer becomes a complement, not a substitute.** T3.1 injects
   n-gram context into attention values in the last layer; the mixer recalibrates the output
   distribution with exact corpus counts. They may overlap (both draw on the same bigram
   statistics) or stack (Aster used both an input-side hashed n-gram embedding and an output
   bigram logit table). The offline test in §6 should therefore be run on the **new T3.1
   checkpoint**, not the plain baseline: if the residual gain of a frozen bigram mixer on top
   of last-layer n-gram VE is < 0.003, drop the mixer and skip the rules question; if ≥ 0.005,
   it is still worth asking.

Also relevant to §8's startup-window discussion: the agent measured that the larger tables
add ~145 s of compile time which is entirely uncharged under the 900 s allowance
(`startup_allowance_est` 121 → 266 s, `charged_seconds_est` unchanged). Compiling more
graphs in the excluded window is the uncontroversial use of it; counting data there remains
the question to ask.

## 6. Offline test first (CPU on the box, zero device time while other A/Bs run)

This is the plan's T4.1 protocol, made concrete:

1. From the best checkpoint, dump per-token neural log-probs on two token streams through
   `prepare.py`'s eval loader at `seq_len 1024` (the scorer's length): (a) the public val
   shard's first 2M tokens, (b) ~2M tokens of **train shard 4 held out** from the count
   tables. Store `log p_neural(target)` and the top-64 `(id, logp)` per position, fp16.
2. Tokenise shards 0–3 once (`count_train_tokens.py` already does this in ~50 s on 12 vCPUs),
   build the dense bigram count table and a hashed trigram top-16 table; smooth both ways.
3. Fit gates on stream (b) only: global λ; per-order λ; a 3-feature logistic gate
   `(neural entropy proxy from top-64, log count(x_t), max order matched)`. Report bpb of the
   mixture vs neural alone on (a) and (b).
4. **Go/no-go**: ≥ 0.005 bpb on (a) with gates fit on (b) → implement §4 on the device path
   and run the standard paired 900 s A/B (expect the step count to be unchanged, so the
   decision is on bpb alone). < 0.003 → drop and record.

Approximation caveat: the offline mixer uses top-64 neural probabilities; the residual mass
must be handled (assign the remainder to the n-gram distribution's tail or re-dump full
`[N, 8192]` fp16 for a 500k-token slice to check the error). Also confirm the gain survives
at the scorer's 1024 context, where the neural model is weaker and the mixer should help
*more*.

## 7. Recommendation

- **Part A (drafters):** file with the other inference-only ideas; nothing to do.
- **Part B (frozen mixer):** promote from "Tier 4 stand-out bet" to the next CPU-side task.
  It costs no device time to falsify, has a measured quality signal on this exact model from
  T3.1, and its cost profile is the opposite of T3.1's. Run §6 while the box is busy with
  other items; post the rules question on Devpost the same day.
- **Fix the plan text:** T4.1's evidence bullet should cite kNN-LM/Infini-gram and this repo's
  T3.1 numbers, and should note that Parameter Golf's n-gram figures are eval-time caches and
  were partly invalidated for a normalisation bug. A pointer to this document has been added
  under T4.1.
- If the organisers rule the mixer out, Aster's *learned* output bigram table (8192 rows, so
  ~32× less exposure to the T3.1 embedding-backward pathology than the 262144-row VE tables)
  is the rules-safe fallback and can be measured with the usual phase_timer-then-900s pipeline.

## 8. Can the counting run on the host CPU alongside the Trainium cores? (added same day)

**Bigram counting: yes, comfortably.** The box is a `trn2.3xlarge` (12 vCPUs, 128 GB host
RAM, one Trainium2 chip); the 12 vCPUs are already serving four rank processes (Python dispatch, tokenizer threads, the best-fit
packer, NEFF launches), with the real dataloader's `fetch()` costing 20–30 ms of *exposed*
time per ~0.81 s step (2.5–3.3%, notebook T1.1). Per step each rank handles 32,768 tokens.
The cheap way to count is not `np.bincount(..., minlength=8192**2)` (that allocates a fresh
268 MB output every call, ~50–100 ms); it is

```
keys = prev.astype(np.int64) * 8192 + cur           # 32k int64, from the host-side row_buffer
u, c = np.unique(keys, return_counts=True)          # ~1–2 ms
table[u] += c                                       # fancy-index add into a persistent int32 [8192*8192]
```

≈ 2–3 ms of numpy per rank per step, GIL released inside the C loops, so it overlaps the
device. Batching keys for 50 steps and doing one `unique` per 1.6M keys (~50–100 ms, in the
existing loader thread) amortises it further. Memory: 268 MB int32 per rank, 1.1 GB total,
fine on the host. Merge once at the end with a device `all_reduce` of the int32 table
(268 MB, sub-second on-chip) or `gather_object`; convert to fp16 log-probs (134 MB) and
save. The acceptance test is the usual one: `phase_timer` `step_total_ms` with counting on
vs off must differ by < 1% (≈ 8 ms), because each 1% of step time is 0.0008 bpb; if it
does not, move the counting to a background thread fed by a queue, or count on rank 0 only
from a host-side pass over the shards (the 50 s tokenisation is the bottleneck, not the
counting).

**Hashed trigram: possible, second phase.** Same `unique`+add pattern on 64-bit keys
`hash(prev2, prev) * 8192 + cur` folded to a 2^24-bucket int32 table (64 MB). The
complication is not counting but *reading*: a dense `[2^24, 8192]` table is impossible, so
the eval path needs top-K continuations per context bucket, which means a pass over the
counts at the end (`np.unique` on the accumulated keys, then per-bucket top-16) — tens of
seconds of CPU that must fit before the checkpoint is written and inside the budget. Do the
bigram first; add the trigram only if the offline test (§6) says the increment is worth it.

**kNN-LM: no, but not for memory reasons.** A full kNN-LM datastore is one d=640 key per
training token: 288M × 640 × 2 B ≈ 370 GB, which does not fit 128 GB; a product-quantised
store (64-byte codes) is ≈ 18 GB and *would* fit. The blockers are elsewhere: (1) the keys are
the pre-`lm_head` hidden states, so every step would ship 32,768 × 640 bf16 per rank
device→host (≈ 200 MB/s sustained) and PQ-encode ~160k vectors/step on the same 12 vCPUs,
a host load comparable to the entire dataloader; (2) evaluation needs ~20M nearest-neighbour
queries against 288M codes inside `forward`, and IVF-PQ on 12 cores at ~10–20k QPS is
1000–2000 s against a 1200 s eval timeout — on the runner's machine, whose RAM we do not
control; (3) `final.pt` would grow to ~20 GB and the scorer loads it as one consolidated
model. The paper's 13.5% gain also came from the *full* datastore; sampled stores lose most
of it. The n-gram table is the kNN-LM idea with the "key" reduced to the exact previous
token(s) — no keys to ship, no index to build, a `[T, V]` gather at eval — which is what makes
it fit this box and the scorer.

**Startup window (the user's question: do the counting inside the excluded launch→step-5
window so it costs no charged time?).**

*Mechanics: yes, for n-grams.* The runner excludes wall time from launch through the fifth
`record_step_boundary` call, capped at 900 s (`prepare.py`, PR #6). Our own startup is
45–120 s (compile), so ~780 s of host time before the fifth boundary would be excluded from
the charge, and `train.py`'s self-managed clock also restarts at that boundary, so it would
not eat our own budget either. A full-corpus bigram count needs ~50 s to tokenise the five
shards on 12 threads (`count_train_tokens.py`) plus a few seconds of counting — it fits with
room to spare. Two practical caveats: neuronx-cc is itself CPU-heavy during steps 0–1, so
the count would contend with compilation on the same 12 vCPUs (both are excluded, so this
only matters against the 900 s cap); and the count must be done by one rank or split by shard
and merged, not repeated four times.

*What it would buy.* Only two things over counting during training: the table exists
before step 0, which is what a **learned** gate (§4, option 1) needs so the neural model can
train against the mixer from the start; and coverage is 100% of the corpus instead of the
~91% one epoch sees. Neither is worth much: a fixed or entropy-adaptive gate needs no table
during training, and a bigram table converges long before 91% of the data.

*Rules.* This is exactly the T4.3 question and it is not settled. The exclusion was
introduced for *compilation* ("Exclude compilation time from training budget", PR #6; forum
threads 45039/45077), and the README bans "exploitation of the evaluation infrastructure".
Counting n-grams is model-building work — it produces a component that the scored forward
uses — done in a window meant for compile. Nothing in the runner would detect it, but
`train.py` is screened by an LLM reviewer and, for top entries, by humans, and the intent is
what they will judge. Recommendation: **do not rely on it without an explicit yes.** Ask (it
is question 3 in §9). If the answer is no or silent, count during training on the host as in
§4; the cost is ~2–3 ms/step overlapped with the device, i.e. effectively zero, and the design
loses nothing that matters.

*kNN-LM cannot use the window at all.* Its keys are hidden states of the *trained* model; at
step 0 the model is random and any datastore built then is useless. The datastore is
inherently post-training work, so the excluded window does not help, and the eval-time
search cost in the previous paragraph stands.

*Drafter.* Irrelevant either way (Part A): it cannot change val_bpb wherever it is built.

## 9. Draft question for the Devpost forum

Post as a new thread under a neutral title such as "Hybrid neural + count-based n-gram
model: allowed?" and keep it to the facts the organisers need:

> We are considering a hybrid language model for Phase 1: a standard transformer plus a
> frozen n-gram component. During the 30-minute training run, `train.py` counts bigram (and
> possibly trigram) statistics **from the provided training shards only**, on the host CPU,
> and saves the resulting count table inside `out/final.pt` together with the weights.
> `load_for_eval` returns a single `nn.Module` whose `forward(x)` is a pure, stateless,
> causal function of the input tokens: it computes the transformer's distribution, looks up
> the n-gram distribution for each position from the frozen table, mixes them with a fixed or
> learned per-token weight, and returns the log of the mixture as logits. No collectives, no
> hooks, nothing is updated at evaluation time, and nothing about the validation data is read
> or inferred — the table is built entirely from the training shards within the budget.
>
> Three questions:
> 1. Is this an acceptable submission under "you may only use assets produced within your own
>    training loop on the provided data" and the "pure, stateless, causal forward" contract?
> 2. For the tie-break on "fewer total parameters", do frozen count-table entries stored as a
>    buffer count as parameters?
> 3. Separately: is host-side data preprocessing (for example tokenising or counting over the
>    training shards) during the excluded launch-to-fifth-step window acceptable, or should
>    all such work happen after the fifth step? We do not plan to run any optimizer steps in
>    that window.
>
> For reference, this is the same idea as kNN-LM (Khandelwal et al., ICLR 2020) and
> Infini-gram (Liu et al., 2024) — a neural LM interpolated with a memory built from its own
> training data — with the memory reduced to exact n-gram counts so that it fits the budget.

Do not mention Parameter Golf in the post; its n-gram track was invalidated for eval-time
adaptation and a normalisation bug, and citing it would invite the wrong comparison.

## 10. Sources

- Speculative drafters: Saxena, "Prompt Lookup Decoding" (2023, github.com/apoorvumang/prompt-lookup-decoding);
  vLLM issue #16258 and speculative-decoding docs; Snowflake Arctic Inference blog; ANPD
  (arXiv 2404.08698); Token Recycling (arXiv 2408.08696); SuffixDecoding (arXiv 2411.04975, NeurIPS 2025).
- kNN-LM: Khandelwal et al., ICLR 2020, arXiv 1911.00172. Infini-gram: Liu et al., 2024,
  arXiv 2401.17377 (Table 1). N-Grammer: Roy et al., arXiv 2207.06366. *N-gram Is Back*: Li et
  al., Findings of EMNLP 2022, arXiv 2210.14431.
- Parameter Golf: issue #140 (Tier 5 invalidation, Mar 27 closure, #803/#1094 descriptions),
  PR #978 (normalised rerun: n-gram path 1.513 bpb), issue #677 (mass closure); *Parameter
  Golf: What Really Works?*, arXiv 2607.01517.
- Aster preliminary results (output bigram logit table), via plan §11.1.
- This repo: README ("pure, stateless, causal forward"; assets produced within the training
  loop; tie-breaks), `prepare.py` (`evaluate_bpb`, `causality_check`), `train.py`
  (`make_dataloader_requeue`, host-side `cpu_inputs`/`row_buffer`),
  `research/trn2-experiment-results.md` (T3.1 sections 2026-09-11/12),
  `research/scratch/count_train_tokens.py`, `research/frontier-research-plan-2026-09-09.md`
  §6 T4.1–T4.3, §11.1.
