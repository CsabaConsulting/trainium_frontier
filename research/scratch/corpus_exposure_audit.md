# Corpus-exposure audit

`corpus_exposure_audit.py` diagnoses the **existing** training loader without
altering `prepare.py`, `train.py`, tokenization, sampling, or any production
loader. It only reads the five provided training shards. It does not initialize
Neuron, train a model, read validation text, download assets, or derive a
replacement dataset.

Run the synthetic check locally:

```sh
python3 research/scratch/corpus_exposure_audit.py --self-test
```

Verified locally on 2026-09-18 in approximately 2.8 seconds: a hand-sized
conservation case and exact CPU input/target/epoch matching against the original
`prepare.make_dataloader` for four ranks × eight rows at the locked geometry.
The original function is executed verbatim in a diagnostic-only namespace with
synthetic input globals; the imported module and source files are not patched.

For a later real audit, on an **otherwise idle** instance with cached competition
assets and the existing environment:

```sh
cd /home/ubuntu/trainium_frontier
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/ubuntu/workspace/native_venv/bin/python \
  /home/ubuntu/trainium_frontier/research/scratch/corpus_exposure_audit.py \
  --steps 2100,2135,2195,2800 --tokenizer-threads 1 \
  --output research/run-logs/corpus-exposure-20260918.json
```

Use the actual synced repository path if it differs. `--output` refuses an
existing file. The script verifies the cached tokenizer through `prepare`'s
verification functions; it deliberately does not call `ensure_data` or
`ensure_tokenizer`, which could refresh or download assets. Cache/tokenizer
location environment overrides are the same as `prepare.py`.

This requires **one full encoding pass through the training corpus**, not one
pass per requested checkpoint. Rank streams run serially and cache only document
lengths/IDs/UTF-8 byte counts, not the full tokenized corpus. At the default four
checkpoints it then simulates up to 11,201 loader rows per rank. Budget several
CPU minutes plus disk reads; actual wall time is intentionally not promised
without a measured corpus pass. Per-row-group progress prints to stderr, and
the JSON reports elapsed time. CPU, disk and tokenizer contention can bias a
concurrent training run, so the full audit has **not** been launched on either
active training instance.

## Geometry and exactness

The locked configuration is `B=1`, pack factor 4, model context length 2048,
world size 4, total batch 131072. The loader receives length **8192**, packs a
row of **8193** tokens, and returns two shifted arrays of 8192. Training reshapes
each array to four independent 2048-token contexts. One global microbatch
therefore has 32768 target positions, with four accumulation microsteps per
optimizer update. Context splits do not discard target positions, but they do
remove cross-context attention; this audit does not measure that quality cost.

The diagnostic preserves:

- Training shards 0–4, their original row-group order, rank striding by row group.
- Original document order within each row group; tokenization batches stop at
  row-group boundaries and contain up to 128 documents.
- Refill before **every** packing selection while fewer than 512 buffered
  documents remain. Initial fill and later fills follow the same rule.
- Longest fitting document, first index on ties; if none fits, the first
  shortest document is cropped to the remaining row space and its tail dropped.
- Buffer contents across source-epoch wraps. No epoch shuffle or batch ramp.

The packing decision depends only on document lengths, so real-data simulation
stores lengths after calling the same fixed tokenizer with the same prepended
BOS. The CPU synthetic comparison additionally verifies actual token ordering
against the original loader, including ties and cropping.

## What the counters mean

`raw_corpus_once` is the full encoded rank-assigned training corpus, including
one BOS per document; `text_tokens_excluding_bos` removes those artificial
tokens. `utf8_bytes` is actual encoded UTF-8 length. The older
`lnc_ws_data_probe.py` used `len(text)` while labeling its output bytes, which
counts Unicode characters instead.

Every copied document is tagged with its **source epoch**, and every snapshot
checks, separately for each epoch:

```text
loaded encoded tokens = retained packed tokens + dropped text tail + buffered tokens
raw corpus tokens = retained + dropped + buffered + not-yet-loaded tokens
loaded documents = selected documents + buffered documents
```

`retained_encoded_tokens` counts the union of the packed x/y row, not
`input_positions + target_positions` (which would double count almost every
token). Each packed row's first token is BOS and is input-only; all retained
ordinary text tokens are targets. Internal document BOS markers are also
training targets and are reported separately. Thus:

```text
target positions = retained ordinary text tokens + internal BOS target tokens
```

`dropped_text_tokens` counts cropped suffixes permanently discarded from that
specific occurrence of a document. `pending_encoded_tokens` counts complete
documents still in the reservoir; these must not be mislabeled as dropped.
`not_yet_loaded_encoded_tokens` is the remainder of that source epoch that has
not reached the reservoir at all. Repeated source epochs count repeated
occurrences rather than novel corpus coverage.

The logged loader `epoch` is the source epoch of the **most recent refill**, not
proof that every earlier-epoch token was trained. The audit records separately
the first refill row and first retained row for each source epoch, plus pending
long documents. A tiny rank corpus can refill through multiple epochs before
emitting its first row, which the synthetic check exercises.

At each requested completed-step count, `trained_prefix` includes exactly the
microbatches that feed backward. `loader_after_untrained_prefetch` also includes
the one extra batch fetched after the final backward. The actual train log
prints the latter batch's loader state. Counters from that extra batch describe
loader work, not additional training exposure. The five startup-excluded steps
still count as actual exposure: startup timing exclusion does not remove their
data from the trained model.

The hand case is a capacity-6 row with documents of BOS-inclusive lengths
4, 7, 9, 10 and a toy refill threshold of three. It packs length4 plus a
length2 prefix of length7, drops the other five tokens of that document, and
leaves lengths9+10 buffered: **30 loaded = 6 retained + 5 dropped + 19 pending**.
Its five target positions contain four ordinary tokens and one internal BOS.

## Rules and interpretation

This is an offline read-only diagnostic over authorized training data, not a
submission change. The README permits changes to data sampling strategy in
`train.py` while also forbidding modifications to the official data pipeline
and `prepare.py`; any proposed future repacking implementation needs its own
rules review. This audit grants no authority to modify either. It does not
reconstruct or inspect validation samples and does not create additional data.

Report actual counts before concluding “one full corpus” or a hard repetition
ceiling. A displayed epoch-2 refill can coexist with unread/untrained tails of
epoch-1 documents. Conversely, a discarded suffix really is absent from that
epoch's training even if all documents were nominally visited. No BPB effect
follows from a token-count discrepancy alone.
