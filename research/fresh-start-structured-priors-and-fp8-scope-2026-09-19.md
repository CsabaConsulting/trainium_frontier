# FP8 coverage, fast functions, and structured prediction ideas

Research response2026-09-19. The current objective is lower Phase1 BPB in1800
charged training seconds. This note distinguishes measured code, bounded
hypotheses and generation-only techniques. No new prior is installed in the
submission, and no extra packages or language assets have been imported.

## FP8 coverage and why the broader route remains open

The current full-FP8 **MLP** includes both up/down forward projections and all
four input/weight-gradient matrix products. It is not a full-transformer FP8
implementation. Attention Q/K/V/output projections, attention itself and the
vocabulary projection remain outside this FP8 region. ReLU derivatives/residual
activations use BF16; master weights, master gradients and sensitive reductions
retain their declared higher precision. Evaluation uses the original native MLP.

An individual stage can lose while a larger FP8 region wins by amortizing
calibration, cache preparation and layout conversions. We already observed
precision-specific layout rankings, so a BF16-only timing is not a veto. Nor
does the current negative full-model integration close all FP8 implementations.
The actual source145 whole-step comparison is native488.702ms, original FP8
MLP568.354ms, compact FP8 MLP562.902ms. These measurements include the complete
model, refresh, communication and optimizer; they are not just a slow isolated
stage. The current composition still has to recover roughly74.20ms per step
to match native before it can claim a training-throughput win.

Source140 profiles observe smaller128x32/32x128 BF16 tiles around the FP8 region,
but missing HLO/layer metadata and dropped notifications prevent assigning that
cost to a specific projection or proving its cause. The next controls remain:

1. Finish same-source resident/streamed packed-weight consumers. Source147's
   eleven independent cases and real-Linear lifecycle checks PASS, clean
   09:59:18UTC. The resident whole-model control completed clean13:20:28UTC,
   all four numerical postchecks:562.968ms, reproducing source145 within0.012%
   with identical final diagnostic loss. The streamed comparison started
   13:22:11UTC with clean preflight after checking every prerequisite.
2. Test the separately documented output-layout boundary if required. Attribute
   changes with a caller profile only after an unprofiled whole-step improvement
   or a result that meaningfully distinguishes the memory/layout hypotheses.
3. Keep broader FP8 projection coverage open: a separately labelled Q/K/V or
   attention-output projection control can test whether extending the region
   removes costly boundaries. Preserve independent Parameter/Muon identities,
   FP32 masters, per-token causal scaling and timed packing. Retain existing
   normalization/softmax/CE precision; indiscriminately casting every operation
   is not necessary to obtain FP8 TensorEngine throughput.
4. Require actual input/weight-gradient accumulation, cache refresh, causality,
   row independence, full-step timing and full-budget quality at each promotion.
   A numerically stable FP8 implementation can still lose final BPB through
   quantization error, and a faster isolated kernel can still slow its caller.

## What happened to the retro numerical ideas

| Technique | Measured result | Current use |
| --- | --- | --- |
| Scalar reciprocal seed plus Newton correction |16.77% faster8M-element AdamW component;1.128% whole-step gain when composed with ownership|Gated optional code; absent from frozen best|
| Stronger inverse-square-root corrections |Original34 numerical gates pass for two variants; complete component timing loses|Retain diagnosis, no promotion|
| Range-reduced degree5/7 exponential |Accuracy gates pass;3.57x/4.19x original latency|Diagnostic only|
| BF16-domain joint-function lookup table |All finite BF16 values pass declared bounds;47.20ms versus2.48ms primitive|This tested layout loses; no training integration|
| Exponent-bit power-of-two scales |Avoids log2/floor/exp2; independently checked raw FP8 bytes and metadata|Used in current FP8 prototypes|
| Reuse saved FP8 bytes, eliminate redundant hidden min/abs work, fuse preparation |Numerical and component gates recorded in kernel queue|Used in current FP8 prototypes|
| Exact loss target gather and adaptive one/two-core scheduling |3.6625% measured whole-step gain|Included in frozen best recipe|
| RoPE sine/cosine tables |Already precomputed by baseline|Existing feature, not a new gain|

These results implement the broader idea of replacing/reusing/eliminating work,
not only a sqrt shortcut. Small interpolated/mantissa tables and composed
rational functions remain candidates only where a profile justifies their total
load/gather/instruction cost. Do not extrapolate a CPU-era speedup past NKI's
existing hardware function engines or call a standalone accuracy win a speed win.
Detailed arithmetic counterexamples, bounds and logs are in
`fresh-start-retro-numerics-2026-09-18.md`.

## Why speculative generation is not the Phase1 BPB route

The trusted `prepare.py:evaluate_bpb` calls `model(x)` on supplied token rows
and computes full-vocabulary cross entropy against the following tokens. It
does not call a generator. An exact speculative drafter changes the execution
of sampling while preserving the target distribution; therefore it cannot
change this BPB score by itself. This follows directly from the local scorer
and the [original speculative-decoding paper](https://arxiv.org/abs/2211.17192).

A corpus trie can serve either as a drafter or as part of a different probability
model. Only the latter can directly improve BPB. The fixed tokenizer already
defines8192 token IDs. A corpus word dictionary would be an additional learned
asset over combinations of these IDs; it cannot replace the tokenizer. Unknown
words, proper names, identifiers and incomplete fragments need full support.

Our earlier hashed bigram/trigram **value embeddings** are gradient-trained
vectors selected by recent token IDs. They are not a word dictionary or a
speculative decoder. A separate count-based mixer was actually tested on
September12: best honestly fitted gain0.000618BPB on the then-current model,
0.002489 on its paired control without learned n-grams, both below that study's
0.003 advancement threshold. The original estimate was substantially too large.
Those old numbers are not measurements on the stronger current model. Revisit
only with a materially different hypothesis and a cheap held-out-training
pilot. See `ngram-drafter-and-count-mixer-2026-09-12.md`, section6 and the original
countmix logs; do not repeat its invalid top-K probability approximation.

Corpus-derived assets must be built within charged training and checkpointed.
The repository records an explicit organizer answer on September14 forbidding
prefilling count assets in the excluded first-five-step startup window
(`frontier-research-plan-2026-09-09.md`, T4.3). This is more specific than a
guess about whether scanning the corpus during init might be free. Ordinary
compilation/cache warmup is distinct from obtaining additional learned assets.

## Corpus and dependency facts checked today

The [publisher's dataset card](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)
labels ClimbMix English. The competition uses its shuffled text derivative and
five permitted training shards. English-labelled web text can include foreign
fragments, names, malformed examples, code and structured snippets; it is not a
promise that every sequence is grammatical English or valid source code.
The container format (Parquet/JSONL in upstream releases) does not imply that
the actual training text is JSON.

`scratch/structured_corpus_audit.py` ran read-only on4000 documents from each
permitted shard:20,000 documents/58,656,551 UTF8 bytes. It found73 explicitly
labelled Python fences in14 documents;65 parsed as complete Python ASTs,8 did
not. Those fence bodies total10,878bytes,0.01855% of sampled bytes. Nine
documents contain the JSON-key heuristic; no explicitly JSON-labelled fence
or complete JSON document was recognized in this sample. Heuristics miss
unfenced code and have false positives. These are coverage observations, not a
complete code/language census, a proof about validation, or a quality estimate.
No source text was executed or stored in the audit output. Shard/script hashes
and content-free example locators are in
`run-logs/fs-training-structure-coverage-20260919.json`.

Read-only package metadata on both active native venvs finds no XGrammar,
Outlines/Outlines-core, LM Format Enforcer, interegular or lark. The checked
common dictionary packages and `/usr/share/dict/words` are also absent. This
is not an exhaustive inventory of every embedded resource or proof of the
organizer's final environment. Python `ast` and `json` are standard-library
modules, so absence of distributions with those names is immaterial.
Receipt: `run-logs/fs-decoder-package-metadata-20260919.json`.

The follow-up traditional-NLP inventory reads all installed distribution
metadata (81 distributions on trn21,82 on trn22) and checks selected NLP,
stemming, spelling, trie, language-identification and count-LM packages.
Only regex2026.7.19, tokenizers0.22.2, tiktoken0.13.0 and
transformers5.14.1 from the selected set are installed. NLTK, spaCy, TextBlob,
Gensim, Stanza, KenLM, SnowballStemmer, SymSpell, WordFreq and the other
selected specialized packages are absent. Declared dependency strings include
optional extras; the receipt deliberately does not call them resolved active
dependency edges. No package was installed, target module imported, device
used, or external corpus/model loaded. Probe and exact results:
`scratch/nlp_dependency_inventory.py` and
`run-logs/fs-traditional-nlp-package-inventory-20260919.json`.

One concrete embedded implementation was found: Transformers' Whisper
`english_normalizer.py` contains number/contraction normalization rules.
Read-only source inspection confirms that its spelling normalizer takes
`english_spelling_mapping` as a constructor argument; that file does not
provide a general spelling dictionary or conditional probability predictor.
Source SHA256 and class signatures are recorded in
`run-logs/fs-whisper-normalizer-source-audit-20260919.json`. A filename hint
or installed Transformers library is not evidence that pretrained weights,
lexicons or task-specific NLP models are bundled and usable.

Even if a dependency bundled an English dictionary, availability would not
authorize using that external language asset. The local README limits learned
assets to the submitted training loop on supplied data, and the current
[competition rules](https://trainium-frontier.devpost.com/rules) require only
provided training data without external data/pretrained assets. A generic
parser algorithm is different from importing a wordlist, pretrained model,
corpus or CORE-specific knowledge. No extra packages were installed here.

Reusable generic algorithms could support byte/word-boundary, digit, case or
punctuation features, or a smoothed count predictor learned from the allowed
corpus. These would need a causal probability-model role to affect BPB.
Stemming, tagging or normalization alone does not provide next-token
probabilities, and normalizing scored targets would violate the fixed scorer.
Learned auxiliary features must still predict the original token sequence.

Do not call spelling-based features wholly untried. The older byte-feature
embedding initialization was a measured negative: three mixture strengths
produced progressively worse BPB, roughly+0.00124 to+0.00219 relative to the
step-adjusted control. It hashed only the provided tokenizer's byte1..3-grams;
see `trn2-experiment-results.md`, Item1 near line6775. Persistent, learned,
gated features would be a different hypothesis, but this negative and the old
count-mixer results argue for a cheap residual/coverage pilot before a device
training trial. No classic-NLP package bypass or init-time learned asset is
queued as a competition shortcut.

## What transfers from grammar-constrained generation

[XGrammar](https://github.com/mlc-ai/xgrammar),
[Outlines](https://github.com/dottxt-ai/outlines) and
[LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer) constrain
token choices using a supplied format/grammar; they do not automatically supply
better semantic predictions or infer that arbitrary web text must be JSON.
The useful idea is the intersection of token-byte continuations and a causal
parser state, not a requirement to install a particular framework.

If a grammar permits a set S containing the true next token, renormalizing
onto S increases that token's probability. If the actual token is outside S,
a hard zero makes its loss infinite. Truncated contexts, examples of invalid
syntax, quote/Markdown boundaries and uncertainty about the text's language
make universal JSON/Python enforcement unsuitable for this corpus.

Python's [`ast.parse`](https://docs.python.org/3/library/ast.html#ast.parse)
expects a parseable program or expression. A legal incomplete prefix such as
`def f(` fails that test. A prefix aid needs an incremental lexer/parser with
indentation, escaping and possible continuations, rather than repeatedly
accepting/rejecting complete ASTs. Parsing the whole scoring row to guide
earlier positions would also leak future tokens. Every position must use only
its own prefix; each row/call reconstructs any ephemeral state independently.

The narrow transferable candidate is a **soft, trained syntax prior** or
syntax features with an ordinary-language fallback. A normalized mixture
`q = (1-g)*p_model + g*p_prior`, with g learned only on allowed training data
and capped below1, can preserve support when the parser is wrong. Its frozen
causal forward must return full-vocabulary logits for the trusted scorer.
All counting, fitting, parser computation and extra model cost must be charged.
This is a changed model requiring training/quality gates, not a free decoder.

A broader prior for identifiers/names repeated earlier in the same context is
another design candidate, but attention already copies and historical notes
warn of weak within-row cache gains. Any experiment must use only already-seen
continuations from strictly earlier matches, never later occurrences or state
accumulated across validation calls. It is lower priority than the active
kernel controls until a held-out-training residual analysis supports it.

We already tested a related tokenizer-only canonical-continuation constraint:
0.996184154 to0.996108324 on one paired2.1M-token public check, a0.00007583BPB
gain. Its narrow ASCII seam rule and empirical falsification checks are recorded
in `fresh-start-deep-research-2026-09-18.md`. That small result motivates careful
coverage and probability-mass measurement before implementing another mask.

## A training-time interpretation of vocabulary organization

A separate architectural candidate is a hierarchical/adaptive output head over
the same8192 token IDs. Predict a cluster and then a token conditional on that
cluster: `p(token|x) = p(cluster(token)|x) * p(token|cluster(token),x)`.
During training, only the target's conditional head is required to compute its
exact loss under this **new model**. At evaluation, expand every cluster into
all8192 normalized log probabilities, in the original token order, for the
unchanged trusted scorer. This is not speculative decoding and does not
preserve the current flat-head model's probabilities or promise equal quality.

The [original adaptive-softmax paper](https://arxiv.org/abs/1609.04309) uses
frequency-dependent output clusters to reduce large-vocabulary training cost.
[PyTorch's API documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html)
describes target-selected clusters and a full `log_prob` interface. This is a
concept reference, not evidence that the installed torch2.11/Neuron compiler
supports its dynamic implementation. No new package is needed to express the
factorization, but a Neuron-efficient implementation would need separate work.

This hypothesis has several reasons to fail here:8192 is a small vocabulary,
the current dense head is efficient, routing/bucketing/padding can erase the
saved GEMMs, and lower-rank rare-token heads can reduce capacity. Computing all
conditional heads for all training tokens would also sacrifice much of the
proposed saving. The output head's measured whole-step share bounds the total
return; component FLOP counts alone do not justify implementation or a run.

Bounded feasibility plan: first audit current output-head cost, allowed-data
token frequencies and static routing shapes. Charge any corpus-derived cluster
construction; do not precompute it in the excluded startup window. Check a CPU
reference's normalization, original-ID mapping, exact target loss/gradients,
causality and full-support expansion. Advance to a device prototype only if
static shapes predict a meaningful whole-step saving after routing overhead,
then require timed full-model and held-out-quality tests. Search of current
research/article Markdown found no earlier adaptive-softmax experiment, but
that is not a proof that every older branch/log was covered. Design only.

## Bounded next steps

- Keep the streamed/resident FP8 whole-model comparison and matched width1152
  evidence. The width checkpoint is independent-hash verified locally and
  GitHub-asset verified. Its matched20M evaluation completed clean13:24:40UTC:
  0.9938092802993084, worse than frozen0.9928513116311253. No promotion.
- Keep wider FP8 matmul coverage on the kernel queue. Do not close it because
  the current MLP composition loses, and do not predict a win without measuring
  the complete region and quality.
- For grammar ideas, first measure prefix-recognizable coverage and removable
  probability mass on held-out training text, with gate fitting separated from
  its test split. The completed document audit is only the first triage step.
  Include malformed/truncated/narrative counterexamples and full-support output.
- Advance a prior only for meaningful held-out CE/BPB improvement after its
  runtime and construction costs. Keep the historical count mixer and hard
  dictionary restriction out of the main training queue in their tested forms.
- Keep hierarchical output factorization as a separate feasibility candidate;
  it translates vocabulary organization into possible training savings without
  relying on a generation loop. It does not displace the active FP8 controls.
