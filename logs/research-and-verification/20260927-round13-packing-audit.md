# Round13 training packing audit — empirical addendum 2026-09-27 14:05 IST

**Measured crop waste is0.0978341%, below the3% gate. Do not promote a zero-waste loader proposal under that gate.** The separate coverage result is that all90 documents longer than32769 tokens were read but never selected at c239's horizon; they account for1.53576% of corpus tokens. This is greedy-buffer selection bias, not cropped-token waste and not evidence of a BPB improvement from changing the loader.

This addendum supersedes the13:52 source-only UNKNOWN statements retained below. Parent confirmed all six training-only transfers complete and remote/local SHA-matched before any copied file was opened. Files live only in /home/poojith-devan/.cache/trainium-round13-audit-MU1oWL. No validation shard, text or token-value dump was read/printed/committed; generated19.5MB document-length manifest stays in that cache. The parent installed the isolated local pyarrow/tiktoken environment; no host SDK/package actions were taken by this Researcher.

## Authentic input and exact horizon

Extractor04bec6475d9d6c6def47c7c38c9a617ce8a9b703c4641b4f44061d54ba220c38 verified the official tokenizer SHA before unpickling and rehashed all five shards against research/v2/analysis/round13-training-sha.json. Exact encode_ordinary_batch, one BOS added to each length,128document batches, two tokenizer threads and non-threaded parquet reads. Complete424,960documents,315,947,189BOS-inclusive tokens (315,522,229text tokens),415row groups. Extraction took56.7098seconds. Full manifest SHA fdd050bffe898100179a922898a52f3be2cb980d93dbfd3b52077a4d1350e0cd.

The authenticated c239 training log reports531.8M tokens with .1f formatting and131072tokens/microstep. Only4057 integer microsteps fit that rounding interval, giving531,759,104exact training target positions globally;2687optimizer steps are not4057microsteps. Source fetches once initially and once after every backward, hence4058fetched rows/rank. This is exact source/log-format inference, not direct checkpoint metadata. See round13-c239-horizon.json with archive/log hashes. Earlier4058trained/4059fetched assumption is rejected.

## Source read versus useful emission

Counts aggregate four ranks; row slots include one extra target-alignment token per row. D/(E+D) is crop waste; buffered P is not crop waste.

| Horizon per rank | Read S | Emitted row slots E | Crop discarded D | Still buffered P | Crop waste |
|---|---:|---:|---:|---:|---:|
|4057 trained rows|541,000,456|531,775,332|520,767|8,704,357|0.0978341%|
|4058 fetched rows|541,087,521|531,906,408|520,811|8,660,302|0.0978183%|

Both satisfy S=E+D+P exactly. The fetched horizon includes the final unused prefetched row; its531,890,176target slots must NOT be called trained tokens. Replaying4057rows models tokens required for actual backward;4058models actual final fetch/read-ahead state. Nothing drops during the32x1024reshape.

Per-rank trained crop rates: rank0 .0981616%, rank1 .0980250%, rank2 .0986435%, rank3 .0965062%. Every rank is far below3%.

## Document-length coverage and repeated epochs

Bins below are non-overlapping, with lengths including BOS. Unique coverage means stable (document ID, prefix-position) exposure in inputs/targets, not deduplicated textual content across separate documents.

| Length bin | Corpus docs | Corpus tokens | Docs emitted at least once | Unique positions exposed |
|---|---:|---:|---:|---:|
|1..1024|376,665|194,226,290|375,805|193,897,375|
|1025..8192|46,651|96,416,935|46,651|96,416,935|
|8193..32769|1,554|20,451,778|1,554|20,451,778|
|>32769|90|4,852,186|0|0|

All corpus document IDs were read at least once on their assigned ranks. Unique exposed positions total310,766,088 (98.36014% of the corpus). Of5,181,101unexposed positions,4,852,186belong to the90 oversized documents;328,915are in the shortest bin. Thus the long documents were not actually cropped in this run—they stayed unselected while smaller documents continued filling rows. The source's prefix-only crop theorem for a selected oversized document is true, but it never fired for these90 documents at this horizon.

There were221,009,244repeated prefix-position emissions (41.56064% of emitted slots), across302,219repeat document emissions. This is expected repeated-epoch work once training exceeds one corpus pass, not221M discarded tokens. The buffer retains earlier-epoch documents and refills deterministic source order, so emitted rows can mix epochs. In the4058fetched replay, ranks0..3 respectively had176/159/201/184mixed-epoch rows. No data shuffle or suffix rotation occurs at restart. Full-length documents may be repeated; tiny cropped prefixes can be repeated with the same stable ID. No claim about semantic duplicates across different IDs is made.

## Verification, artifacts and decision

The source-derived oracle547d6e2179d48da0dd679fa127b8ac0afb17f97808f0c4ba5bdf144a18084ad9 was checked against actual unchanged make_dataloader AST-extracted from pinned prepare.py, using real CPU torch and synthetic unique-position token IDs. Every input/target position and exact token-read count agreed on three trajectories, including repeated-epoch long-doc crops and production seq_len32768. The check excludes prepare's import/setup/data/evaluation code. This validates the reference implementation on those cases; independent review of the empirical receipts is still requested.

Durable evidence:
- round13-packing-empirical-results.json SHA e04b3ab4f81f7b7dc78dae029159f8c8b0a6aa94019580993cccf4323cf4d7a9: both complete per-rank horizons.
- round13-train-length-extraction-receipt.json SHA a4f77d582fc1b2b6ce8250c6f705ff77d641c40836eab76d71129ee0df921e1e: train file hashes, row/document/token counts, manifest hash; repetitive per-row-group counts remain in the cache receipt.
- round13-c239-horizon.json SHA fd3ecd3c1d0af449c63007a490e39d6030a1d106a8858e3ae37bd3579bd1006e.
- round13-real-loader-synthetic-result.json and round13_verify_real_loader.py: real-loader CPU check. After coverage-only reporting additions, the same three exact-identity cases were rerun PASS at13:59:53IST.

Reproduce using existing cache artifacts:
```text
python3 research/v2/analysis/round13_packing_lengths.py --manifest /home/poojith-devan/.cache/trainium-round13-audit-MU1oWL/train-lengths-r1.json --prepare prepare.py --rows 4057
python3 research/v2/analysis/round13_packing_lengths.py --manifest /home/poojith-devan/.cache/trainium-round13-audit-MU1oWL/train-lengths-r1.json --prepare prepare.py --rows 4058
```

Verdict: the>3%waste condition is false, so no zero-waste loader arm is proposed/released. Long-document coverage is a separate measured observation to retain, not new authority to alter the training pipeline. No model, evaluator, tokenizer, queue, ledger, host/device, or13:45research-cycle changes and no new training/BPB result.

---

## Historical source-only audit — 2026-09-27 13:52 IST


Status: READ-ONLY SOURCE AUDIT; empirical training-corpus waste UNKNOWN. No train.py, prepare.py, evaluator, tokenizer, queue, launcher, ledger or13:45research-cycle changes. No host/device/SSH/cloud/package calls and no validation data accessed.

## Finding

The current training loader definitely drops document suffixes. Its actual production row capacity is32769 tokens, not1025: c239 passes DEVICE_BATCH_SIZE=1 and PACK_FACTOR32*SEQ_LEN1024=32768 to prepare.make_dataloader, which allocates seq_len+1. On failure to find a whole document fitting the remaining row, it removes the shortest buffered document, emits only doc[:remaining], and never stores the suffix. Thus documents longer than32769 tokens including BOS can only ever contribute a prefix on each occurrence; at least length-32769 tokens are discarded each time such a document is selected. This is established from source, not an observed3% waste result.

Empirical source tokens read, tokens emitted, discarded percentage, length-bin coverage, repeated-prefix volume and epochs reached are all UNKNOWN without training-only metadata. Do not release a zero-waste training-loader experiment merely because cropping exists. The requested>3% empirical gate is NOT established.

## Exact evidence

- Canonical inspected parent: research/v2/operator/20260927-c239-qualification/stack-ramp-bare/snapshot/train.py, SHA256 c23935404ac44314e333ddf3608a6c671f4526263a3348eb901bb3f8d4ca05b0. Training call/reshape at1464–1473; microloop fetch at1507–1514.
- Canonical prepare.py equals that packet's snapshot/prepare.py, both SHA2565807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb. _row_groups_for_split at423–432, _document_batches at435–444, make_dataloader at456–518; crop branch at505–510. Tokenizer.encode prepends one BOS, no EOS, at220–237.
- Public training shard IDs are0..4. Rank r receives global ordered row-group indices r,r+4,..., not shuffled documents or token-balanced partitions. Row groups and document order repeat unchanged each epoch.
- Checked local default /home/poojith-devan/.cache/neuron_competition_r1/{data,tokenizer}; cache directories and shard_00000..00004.parquet absent. Bounded rg filename search under Desktop/Documents/defaultcache found no matching training shards or tokenizer.pkl. Local NEURON_COMPETITION_R1_CACHE_DIR is unset. Packet points to remote /home/ubuntu/.cache/neuron_competition_r1; it was NOT accessed. This is not a claim that no training files exist anywhere outside the searched roots.

## Exact loader invariants and consequences

1. Tokenizer batches up to128 documents. Before each packing decision the buffer is refilled until it has at least512 documents. Greedy selection takes the longest document no longer than remaining space; equal-length ties use earliest buffered index. Otherwise shortest wins, earliest tie, and its tail is discarded. No padding and no continuation cursor exist.
2. Each emitted rank microbatch contains32769 row-token slots,32768 input positions and32768 target positions. Reshaping into32x1024 changes attention context boundaries but discards ZERO additional slots. All original target positions1..32768 remain. Only the first of the32 chunks is guaranteed to begin at the row's BOS; there is no32-fold per-document truncation introduced by pack32.
3. For R fetched rows, emitted row slots=32769R and training-target slots=32768R. The one-slot difference is target alignment, not evidence of dropped document suffixes. Do not add it to crop waste or count input+target views as65536 new tokens. Each selected document occurrence contributes its BOS at least once; the crop drops only its suffix.
4. Conservation for tokenized source occurrences is S=E+D+P, where S includes all read-ahead documents, E emitted row slots, D discarded crop slots, P tokens in the still-buffered documents. Crop waste over consumed source is D/(E+D). D/S is a separate read-ahead-sensitive fraction. Reporting1-E/S as waste would falsely count pending tokens. The loop prefetches the next row even after the last trained microstep; distinguish fetched/emitted rows from rows actually used in backward.
5. Documents longer than32769 cannot fit any row, so they are never fully covered on a single occurrence. Repeated epochs reread them from offset0, not a rotating suffix. Documents below the capacity can also lose suffixes when they are the shortest remaining non-fitting choice. Greedy buffering can delay long documents within a finite budget; length-bin statistics must include pending/unselected documents, not only selected averages.
6. _document_batches increments epoch when restarting the SAME rank's ordered row groups. doc_buffer is not cleared, so emitted rows can mix documents from different source epochs. The yielded epoch is the most recently fetched batch's epoch, not a guarantee that every emitted token belongs to that epoch. Four ranks can have different corpus sizes/restart times. No new shuffle occurs at epoch restart.
7. Given each occurrence always emits doc[:k], cumulative unique positions exposed from a stable document ID are max(k across occurrences), not sum(k). Subsequent epochs can repeat already-seen prefixes while its tail remains unseen. Text duplicate detection across distinct documents is a different question; it requires authenticated content/token hashes and is not answered by lengths alone.

## Reproducible bounded checker

research/v2/analysis/round13_packing_lengths.py is a standard-library, CPU-only length replay of the pinned algorithm. It does not import prepare, initialize distributed/device libraries, open parquet, read text/token values or write files. It accepts only explicit split=train and shard IDs0..4; it checks prepare SHA and preserves rank row-group ownership,128-document batching,512minimum buffer, tie rules, cropping and deterministic epochs. It emits aggregate read/emitted/discarded/pending counts, per-length-bin consumption, unique prefix positions and repeated document emissions. It is a source-derived reference, not independent execution of the real loader and not proof that an input manifest is complete/authentic.

Command: python3 research/v2/analysis/round13_packing_lengths.py --self-test

Actual13:51:26 IST result:5 synthetic cases PASS. At miniature capacity33, exact33-token documents have0% crop waste;66-token documents have50%;10-token documents incur17.5% (three full10-token docs plus3-token prefix then7-token discarded tail per row). Repeated66-token document IDs expose only33 unique prefix positions after four rows despite four emissions. These are mathematical synthetic examples, NOT ClimbMix measurements. They prove waste can exceed3%, not that this run does.

Checker SHA256ef639bc4355fabf021c794aecdabee5b7ecaaa9b1b1b2d354b14aa16e4cc03c3.

## Minimal safe evidence needed from the sole Operator

Preferred: a small TRAIN-ONLY length manifest, produced by an approved read-only CPU process using the existing pinned tokenizer, covering all documents of training shards0..4 in exact shard,row-group,row order. Supply each document's stable opaque ID (e.g. shard/rowgroup/row), exact tokens_with_bos, shard and row_group; include shard byte hashes, tokenizer SHA, row-group counts and row counts separately to authenticate completeness. No document text or token payload is needed. Hashing only provided training files is sufficient; never enumerate/read validation shard6542. No download/install is requested.

Manifest schema: {"split":"train","groups":[{"shard":0,"row_group":0,"documents":[{"id":"s0-rg0-row0","tokens_with_bos":123}]}]}.

Replay: python3 research/v2/analysis/round13_packing_lengths.py --manifest /explicit/train-lengths.json --prepare prepare.py --rows N

Use authenticated actual rank microstep/fetch counts for N (reported checkpoint/log evidence), rather than optimizer-step count alone because accumulation ramps1→2. The utility accepts common N for all ranks; evaluate each actual rank horizon if histories differ. At least one complete training-epoch manifest per rank is needed to characterize epoch repeats; a sample can estimate length frequencies but cannot reproduce the512buffer trajectory reliably. If local approved training shards/tokenizer become available, an independently reviewed train-only manifest extractor can run entirely on CPU without touching the model or evaluation pipeline.

## Decision

No empirical percentage can currently be claimed; zero-waste proposal gate remains UNKNOWN. Preserve all current jobs and Round12. Once measured consumed-source crop waste>3% is independently verified, a separate train.py-only training iterator proposal can retain unconsumed suffixes while keeping exact dataset/tokenizer/rank coverage and untouched evaluator; its model-quality effect and altered training ordering require a full controlled experiment. This conditional description is not implementation or launch authority.
