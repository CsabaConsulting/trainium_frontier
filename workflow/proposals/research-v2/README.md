# research-v2: one research file with IMPL-1, IMPL-2 and IMPL-3 (every new flag default-off)

| Stage | Change | SHA-256 |
|---|---|---|
| 0 | exact 35c9 (`submissions/dense6x1024_public0992851/train.py`) | `35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2` |
| 1 | + IMPL-1: tables bypass optimizer ownership (`../IMPL-1/`) | `9e5dce344674b8217d8749ca3e2ce8c1961a73a9c10c1c6759f645a62ed66783` |
| 2 | + IMPL-2 `--ngram-ve-freeze-epoch N` (`impl2.patch`) | `5f89c65434a77e02424b55204720407015c6445839bfd20cb10f02aa59aa8a21` |
| 3 | + IMPL-3 `--fresh-tail-frac F`, `--requeue-identity` (`impl3.patch`) = **`train.py`** | **`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`** |

Rebuild and check it with `python workflow/proposals/research-v2/make_v2.py`. It asserts the stage-0 and stage-1 hashes.

Deploy it as a new research path (e.g. `submissions/research_v2/train.py`) next to the organizer `prepare.py`. Leave 35c9 untouched.
Use it for **every** Track-A arm from now on: all arms and controls then share identical bytes, and only argv
differs. Re-run one same-host control on these bytes before comparing against 35c9-era controls. Every
new flag is proven bit-identical at default (T1), so this is only a formality.

## IMPL-2: epoch-2 table freeze (`--ngram-ve-freeze-epoch N`, 0 = off)
- **Why:** on the 640 table stack, steps taken after the data wall (loader epoch 2, low LR) cost +0.0023 to
  +0.0048 BPB. Dense runs about 1.7 epochs, so tables may memorize repeated n-grams during the cooldown.
- **What:** from the first step whose batch comes from loader epoch ≥ N, the n-gram table gradients are
  dropped before the all-reduce. There is then no table update, no table all-reduce and no RMSProp state
  change. Table lookups continue.
- **Why dropping gradients rather than `requires_grad`:** the compiled graph is unchanged, so there is no
  mid-run recompile. The backward still computes the table gradients, which costs a little.
- **Caveat:** the loader's `epoch` label reflects the latest buffer refill. It can lead actual consumption
  by up to about 512 documents (≈0.1% of an epoch).

## IMPL-3: fresh-tail data order (`--fresh-tail-frac F`, 0 = off; control `--requeue-identity`)
- **Why:** with the natural order, the last ~41% of a dense run (all low-LR cooldown) replays the
  *opening* documents. The fresh-tail order walks each rank's row groups as A (the first 1−F) twice, then
  B (the last F, unseen). With F≈0.25–0.3, the lowest-LR tail trains on new documents.
- **What:** the same data, the same per-rank partition (`start=rank, step=world`), a different order.
  It uses the in-file `make_dataloader_requeue` (a copy of the organizer loader).
- **Controls:** compare **only** against `--requeue-identity`, which is the same loader with the natural
  order. The requeue loader's `buffer_min=512` may differ from `prepare.make_dataloader`.
- **Guard:** at start the file logs `fresh tail: rank r walks |A| row groups twice, then |B| unseen
  ones`. It refuses to run if B would be empty on a rank.
- **Sizing:** tokens needed ≈ 1.7 epochs. With F=0.25 the stream holds 2×0.75 + 0.25 = 1.75 epochs before
  cycling, so most of B is reached. With F=0.3 it holds exactly 1.7.

## CPU verification (gloo, 4 ranks, the file's real `main()`; receipts in `workflow/evidence/`)
| Test | Result | Receipt |
|---|---|---|
| T1: all new flags at default, 3 steps against exact 35c9 | **identical** (plan, losses, 62/62 parameter hashes) | `lockstep3-35c9-vs-v2.json` |
| T2: A2 tables with `--ngram-ve-freeze-epoch 2` (not reached) against IMPL-1 A2 | **identical** (74/74) | `impl2-untriggered-vs-impl1.json` |
| T3: A2 tables with `--ngram-ve-freeze-epoch 1` (freeze from step 0), 2 steps | all 4 tables unchanged; 0 table updates; ranks identical; freeze logged once. The only other unchanged tensors are the zero-init table gates, whose gradient is exactly 0 while the tables are zero | `g2-a2m8-v2-freeze1.json` |
| T4: flag plumbing | `--fresh-tail-frac 0.25` and `--requeue-identity` PASS and log their data order; conflicting orders and F=0.7 are rejected by argparse | `g2-v2-*.json` |
| T5: the data order itself, on 32 real Parquet row groups over 4 ranks | each rank walks A twice, then its unseen B, then full passes; ranks disjoint; an empty B is rejected | `impl3-fresh-tail-order-test.txt` |

## Remaining gates before scored arms
1. **Verifier receipt** for stages 1–3 (checklist B). Re-run T1–T5 on your own machine against the
   real `prepare.py`.
2. **Neuron smoke** of the A2 argv (50 steps). Record the median/percentiles of |table grad| at steps
   1/50 for A-eps, the step time, and `SUBMISSION_OPTIMIZER_INTEGRATION`.
3. **Host check for IMPL-3.** Hash the first 20 batches of `--requeue-identity` against the organizer
   loader (the 35c9 default) on the real shards. If they differ, the identity control remains mandatory
   for FT arms, which is already the rule. Record the per-rank `|A|/|B|` log line.
