# Fresh-start evidence audit — 2026-09-18

Lower BPB is better. This is a local-evidence audit, not an official-score claim.
Sources are the committed raw run logs, the 401-row experiment index,
`trn2-experiment-results.md`, `plan-sub1-2026-09-16.md`, current code, and relevant
September 15–16 memory summaries. Raw logs take precedence over retrospective
mechanism claims. The index is incomplete: some seed runs, width diagnostics and
dress rehearsals exist only as raw logs. No remote experiments were launched by
this audit.

## What is reproducible

| Run / source in `run-logs/` | BPB | Definition | Steps | Charged training / exposure |
|---|---:|---|---:|---|
| `seedvar-s42-1800s-v2.log` | 1.004355 | in-process public evaluation, sequence 2048, raw weights | 2139 | 1784.9 s training, in budget |
| `t17-l1-m128.log` | 1.004363 / 1.019229 | same checkpoint, in-process sequence 2048 / 1024 | 2135 | 1794.3 s training, 279.8M tokens |
| `tc-sh-640x6-2135.log` | 1.004380 | width-study control, public sequence 2048 | 2138 | 1795.5 s charged, other-box reproduction |
| `sub1-record-repro-full.log` | 1.005436 / 1.020435 | raw, in-process sequence 2048 / 1024 | 2094 | 1784.7 s training, 274.5M tokens |
| `sub1-p42-nonautocast-full.log` | 1.005781 / 1.020452 | raw, in-process sequence 2048 / 1024 | 2104 | 1784.9 s training, 275.8M tokens |
| `finalwrap-dress-rehearsal-20260916-evalpublic.log` | 1.003045 | standalone `prepare.py eval-public`, sequence 2048, 2,097,152 tokens | 2131 training steps | 1784.4 s training; separate eval 45.2 s |
| `tc-sh-768x6-2135.log` | 0.998894 | diagnostic width 768, sequence 2048 | 2135 | **over 1800 s**, not submittable as run |
| `tc-sh-896x6-2135.log` | 0.987716 | diagnostic width 896, sequence 2048 | 2137 | **2488.7 s charged**, not submittable as run |

The 1.003045 standalone score is explicitly documented as sequence 2048 in
`trn2-experiment-results.md` around line 9798. It is not evidence for EMA or a
sequence-1024 improvement. No current private official score is established by
these local logs. `prepare.py eval-public` defaults to sequence 1024, whereas the
in-process primary metric uses the training length. The approximately 0.015 BPB
context effect is materially larger than the remaining public gap.

The locked model is 6 layers × width 640, 5 attention heads of dimension 128,
sequence 2048, MLP ratio 4, global batch 131072 tokens, LNC=2, four ranks, four
accumulation microsteps, pack factor 4. There are 592,916,844 parameters; about
536.9M are four learned token bigram/trigram tables of shape 1048576×128, injected
at layers 1 and 5. Tables and RMSProp state are BF16, beta2=.999, flat LR=.30.
Dense Muon and AdamW settings retain their existing defaults; compiled Muon,
frozen zero Q/K shift, fused RoPE/norm and fused softcap/CE are enabled. EMA is
**off**. These are learned token tables, not frozen byte-count tables.

An explicit baseline launch, using the current baseline defaults for other
settings, is:

```sh
NEURON_CC_FLAGS='--optlevel=1' NEURON_LOGICAL_NC_CONFIG=2 \
torchrun --standalone --nproc_per_node=4 train.py \
  --num-steps 5000 --max-train-seconds 1800 --seed 42 \
  --depth 6 --n-embd 640 --eval-seq-len 1024 \
  --compile-muon --qk-shift --qk-shift-beta 0.0 --qk-shift-freeze \
  --ngram-ve-layers 1,-1 --ngram-ve-table-mult 128 \
  --out-dir out/NEW-UNIQUE-LABEL
```

Preserve the code hash and expanded configuration, since default-off flags can
otherwise change a reconstruction. Seed 42 is explicitly confirmed in the final
rehearsal notes; the recent explicit-seed versus default-seed comparison should
not be described as an established cross-seed experiment.

## The strongest missed interaction: throughput × width × exposure

The width payoff study already proved useful additional dense capacity:
768 gives roughly -0.0055 BPB and 896 roughly -0.0167 BPB at approximately equal
2135 steps and normalized cooldown. Its closure was an economic statement:
without faster execution, insufficient tokens fit within 1800 seconds. It was
not a failure to learn at larger width. See
`scratch/tc_probe3_shape_payoff.sh` and `trn2-experiment-results.md` around 5934.

The old width-896 launch used `--max-train-seconds 2493 --n-embd 896` with the
baseline flags above, no explicit step cap override, and no auto-cast. It built
631,850,284 parameters, seven heads, depth six. The four large tables did not
grow with width. A 28% reduction in its historical charged time would put its
2137-step training trajectory near 1800 seconds, if the improvement transferred
without a quality cost. Less speed can still help: the old slope-based analysis
estimated about +11.6% throughput was needed merely to beat the 640 baseline.

Recovered BF16 auto-cast is precisely a new interaction worth testing. With
`NEURON_CC_FLAGS='--optlevel=1 --auto-cast matmult --auto-cast-type bf16'`, the
640 model reached 2719 steps / 356.4M tokens in budget, but 1.015549 BPB
(`sub1-p2b-autocast-bf16-fullv2.log`). This is a valid negative for that exact
run, not proof that the faster compiler mode cannot fund a wider model.

The claim that auto-cast numerical drift is established “by elimination” is too
strong. LR reductions to .77 and .65 still ran 2700 / 2688 steps and repeated
part of the corpus. They did not match data exposure, the LR value on each
document, nor the table-update trajectory. At step 2100, fullv2 still logged
`lrm=.347`; the normal run is finishing near `.06`. A same-exposure, compressed
cooldown control is needed before attributing the entire gap to precision.

The corpus comprises five training shards, 415 row groups, 424,960 documents,
1,260,004,142 source text bytes (measured historical data probe). At the locked
packing/batch shape, the first epoch wraps near 2195 steps, approximately 288M
tokens; record runs consume roughly 95–98% of it. The repeated-data penalty is
an empirical result for this training setup, not a universal theorem that a
second epoch cannot help. The flat-LR ngram optimizer is a possible interaction.

## Closures that need narrower wording

1. **Shuffle.** The recent `sub1-p4-shuffle-full` ended at 2126 steps. Its
   implementation deliberately preserves epoch 1 byte-for-byte, so this is a
   no-op control. It cannot establish that data order is irrelevant. The older
   `epochshuf-frozen-1800s` did enter epoch 2 and lost (1.023799 versus 1.021278
   unshuffled, older single-site configuration). Retain that measured negative;
   “identical ordering is optimal among every possible data strategy” is not
   demonstrated.
2. **EMA.** The only measured in-process multi-decay sweep,
   `t22-ema-1800s`, used 55.7M parameters with no ngram tables. Decay .98 was
   essentially neutral (-.000110), .995 slightly worse, .999 substantially worse.
   The rationale that cooldown removes the value of averaging does not transfer
   directly to the current 536.9M table parameters, which never cool their LR.
   Cross-seed checkpoint averaging losing .037 is a separate question from
   same-trajectory averaging. Test short table-only or whole-model EMA in FP32
   if its cost is acceptable; compare raw and EMA from the same run.
3. **BF16 table optimizer precision.** `zeros_like(p)` makes RMSProp moments
   BF16 with the table; beta2=.999 requests .1% decay, below half an ULP for many
   positive normal BF16 values. Infrequently touched rows can therefore retain
   moments instead of decaying. Beta2=.95/.99 sweeps changed the intended
   optimizer and did not isolate moment storage precision. A FP32-moment-only
   arm preserving BF16 tables is a concrete untested hypothesis; the historical
   search and local numerical check below support prioritizing it.
4. **Stochastic rounding.** The global runtime toggle failed causality because
   two identical forwards differed by .011341 (tolerance .01), as documented
   near line 310 and `scratch/sr_causality_falsepositive_check.py`. This says
   nothing about training-only stochastic parameter rounding followed by a
   deterministic, static eval model. Such an optimizer change needs numerical
   and rules checks; it is not precluded by the old forward-randomness failure.
5. **Architecture smoke gates.** A universal demand to beat quality at 300
   steps can discard changes whose purpose is late convergence or sufficient
   training of larger capacity. Use smoke for causality, loss sanity and cost;
   use a prespecified exposure/schedule experiment for those quality hypotheses.

Do not revive byte-length-weighted CE on the old rationale. The scorer sums
ordinary per-token NLL and divides by a data-dependent byte total. At a fixed
dataset that denominator is constant with respect to model parameters, so the
ordinary NLL already optimizes the metric (apart from zero-byte special-token
mask details). The linear byte-weighting run's +.048 loss is unsurprising.
`ideas-sub1-2026-09-16.md` also incorrectly says a frozen count/logit mixer is
already enabled; the logged output-bigram toggle is false.

## Five suggested next runs, in priority order

| Run | Change | Question and decision |
|---|---|---|
| 1. Exposure discriminator | Baseline 640 + BF16 matmult auto-cast/O1; shorten wall budget using a pilot to finish near 2100 steps; high step cap | Does its quality recover when it avoids epoch 2 and completes cooldown? Time-based schedule matching is approximate; record actual steps and per-step `lrm`. Current first-wave telemetry is nearer .70s/step than historical .66, so a 1395s budget may end near 1980 steps rather than 2100. Do not call that exact matching. |
| 2. Width896 × BF16 | Baseline with `--n-embd 896 --depth 6`, BF16 matmult auto-cast/O1, full 1800s | Convert previously demonstrated sub-1 capacity into a legal run. Current first-wave early telemetry around .88s/step suggests about 2000 steps, below epoch 2. Compare both eval lengths, causality and actual budget. |
| 3. Average then clip | Baseline no-auto-cast; synchronize gradients before clipping, preserve clipping thresholds/groups | The existing local-clip-before-average order is not mathematically equivalent to clipping the averaged global gradient. Run the corrected-order arm only after CPU/distributed equivalence checks and a hardware smoke; log clipping frequency and gradient norms. |
| 4. RMSProp FP32 moment | Keep BF16 tables, same beta2/LR/epsilon and parameter update rounding; only `exp_avg_sq` FP32 | Is BF16 state creating an unintended cumulative-sum optimizer on rare ngrams? First prove the decay failure and correctness on synthetic sparse rows; time the implementation. A fused NKI update may make the added precision affordable. |
| 5. Targeted short EMA | Current best full-budget model; FP32 shadows over only the table weights, short averaging window, same-run raw evaluation | Does the current flat-LR parameter majority benefit where the old cooled dense model did not? Treat as a paired in-run comparison; track memory, copying/checkpoint overhead and decay precision. |

Width768×BF16 is the immediate reserve if 896 cannot realize enough throughput
or exposes precision sensitivity. Optimizer-scope precision isolation follows
if run 1 remains worse after an adequate schedule/exposure match. Keep whole
FP8 and a full attention rewrite behind correctness and speed evidence: latest
FP8 compiler smokes either freeze/diverge, while corrected NKI attention
`sub1-p1-nkiattn-smoke2` was causally correct and loss-equivalent but about
1.9× slower per step. The false 0.001543 BPB take was future-token leakage and
must never enter score tables.

## Recording cautions

The apparent two-box speed collapse was caused by losing `--optlevel=1` when
exporting `NEURON_CC_FLAGS`, not demonstrated hardware deterioration. Preserve
the full compiler flag string in each manifest. Historical `git_dirty=True`
often includes the logger's own append; preserve source hashes and file diffs
instead of using that boolean as evidence that code was uncontrolled. Some
historical git hashes in the imported index are absent from the current object
database, making the raw configuration print and saved source especially useful.
The old logger writes a run log only at process completion, so a reused label
can show stale output while a new process is live. Use unique labels, live
stdout files, one device owner per instance, and incremental local retrieval.

Memory retrieval attempted the installed `memsearch` CLI, whose current option
is `--collection` rather than the skill's `--default-collection`; its Milvus
lock could not be created under the then-active read-only sandbox. Relevant
source memory markdown was read directly, per the skill fallback. No memory
database changes were made.

## Follow-up: RMSProp moment precision was not isolated

Searched historical markdown, memory sources, scratch files, and every indexed
label/description/command for FP32 moments/state, BF16 RMSProp precision,
master weights, `--no-ngram-ve-bf16`, and stochastic rounding. No experiment
isolating FP32 RMSProp moments while retaining BF16 table parameters was found.
This is an evidence-availability statement, not proof no unlogged probe ran.

Relevant distinct precedents:

- `trn2-experiment-results.md` around 8236–8373: beta2=.95 and .99 were
  intentional hyperparameter changes, both with BF16 tables/state unchanged;
  .95 hurt reproducibly by roughly .002 BPB, .99 was neutral. The explanation
  that .999 moments smoothly decay between rare visits was assumed, not measured.
- `mlrfp32-arm-900s` and its paired controls tested casting the **Muon LR
  product** to FP32, not RMSProp state. Its neutral result cannot close this arm.
- Global SR (`sol13b-final-shape-ab`) failed the eval self-consistency check;
  no isolated optimizer-only rounding experiment was found.
- Older reports discuss FP32 master weights/BF16 compute for the dense model;
  this does not describe the later explicitly BF16 hashed tables.

Direct local CPU check, PyTorch 2.11.0, 2026-09-18:

```python
v = torch.logspace(-30, 30, 1001).to(torch.bfloat16)
before = v.clone()
v.lerp_(torch.zeros_like(v), 0.001)
# (v == before).sum() == 1001; FP32 equivalent has 0 unchanged.

# Starting at 1.0, repeating this zero-gradient update 1000 times:
# BF16: 1.0; FP32: 0.3676954209804535.
```

For normal positive BF16 values, a .1% decrement is smaller than half the
spacing to the previous representable value, even at the tightest part of a
binade. Round-to-nearest therefore erases the decrement; this is not only a
tiny-number underflow issue. The training runtime's original foreach path
still needs its own hardware check, especially with global runtime rounding
flags recorded.

Correct state precision is not automatically a quality gain: the unintended
long-lived moments may be acting as regularization, and the old LR was selected
with that behavior in place. Separate the mechanism test from the full-budget
quality verdict. Preserve all other arithmetic as closely as possible, compare
matched steps first, then retune beta2/LR only if the mechanism is demonstrated.

## Follow-up: the NKI-attention OFF path changed compiler boundaries

**Current default flags do not reproduce the historical compiled attention
path.** Commit `5e9043e` adds an unconditional `@torch._dynamo.disable` decorator
to `nki_attention`, and changes the ordinary causal branch from direct
`F.scaled_dot_product_attention(...)` to an unconditional call to that wrapper.
The `USE_NKI_ATTN` test is inside the disabled wrapper. Therefore even with
`--nki-attn` absent, the ordinary SDPA is outside the enclosing Dynamo graph.
Eager mathematical equality does not establish identical compiled scheduling,
fusion, auto-cast placement, or training numerics.

The source-history evidence is exact:

| Source | `train.py` Git blob | Source checksum / behavior |
|---|---|---|
| `5e9043e^` | `95455ce8afef971bacd9867aa55fc212dde87a0e` | SHA256 `52ac9e6560a40592255e3499f4d42ac6e7073ea82da1b428e03dc984e6d4baa8`; direct causal SDPA, no `nki_attention` wrapper |
| `5e9043e` | `23802e54de688871e6c1bd81a3ce335ee4bdc261` | SHA256 `0c238a0c0ae2b8c3ad9502b29e4f8dd6442d66e8440e4004bfecdf0b83d21cff`; disabled wrapper for both ON and OFF |
| `dffa5ee` rehearsal-era source | `212cbbe5b1564b20eec899618c8bef7d99f98e78` | MD5 `24cce227d3bcf9750f6ea07f9d58a8d1`, matches the explicitly recorded September 16 dress-rehearsal source MD5; direct SDPA |

Commit time is later than live editing: `5e9043e` committed the whole port on
2026-09-18 at 06:26:47 UTC. Contemporaneous note commit `bfd77de`, at 02:16:52
UTC, first reports the disable workaround already applied for take 4, after
the presence control completed. `f313810` (00:18:09 UTC) says the port was
integrated, but only committed scratch files/notes; it did not preserve that
intermediate `train.py` source. This limits source-hash reconstruction of the
live presence-control version.

| Historical measurement | Time evidence | Relationship to disable edit |
|---|---|---|
| `t17-l1-m128` | indexed completion September 14, 11:57 UTC | before |
| `tc-sh-896x6-2135` | raw log begins September 14, 14:46 UTC | before |
| `seedvar-s42-1800s-v2` | raw log begins September 15, 13:23 UTC | before |
| `sub1-p2b-autocast-bf16-fullv2` | indexed completion September 17, 13:15 UTC | before |
| `sub1-p42-nonautocast-full` / record repro | September 17, 15:26 / 15:29 UTC | before |
| `sub1-p1-nkiattn-off-smoke` | September 18, 02:00 UTC completion; note says disable added afterward | pre-disable according to contemporaneous notes; no saved per-run source hash |
| `sub1-p1-nkiattn-smoke` / `smoke2` | September 18, 02:28 / 02:57 UTC completion | disabled wrapper, **NKI ON** |

No recorded benchmark of **disabled-wrapper OFF versus direct-SDPA OFF** was
found. The earlier OFF presence control's 252.9 seconds / 300 steps cannot be
used as a measurement of today's disabled OFF path. The CPU fallback test in
`scratch/nki_attn_train_equiv.py` tests eager output equality and guards; it
does not test compiled graph equivalence or performance.

The current first-wave manifest `fs-w640-bf16-exposure1395-r1.launch.json`
records source SHA256
`55e22919d688f5d32f2fe38f1fe709551fb9713715d28e338fd999ab2975df39` at
`ca63567`, descended from the disabled-wrapper source. Consequently its
comparison with historical fullv2 changes both training exposure and compiled
graph boundaries. Its result remains useful as a candidate, but cannot alone
identify either cause. A fresh same-source paired smoke with direct SDPA when
the flag is OFF is the required isolation; preserve the first-wave outcomes as
measurements of the source actually launched.
