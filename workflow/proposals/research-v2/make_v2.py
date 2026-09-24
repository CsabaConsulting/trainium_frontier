#!/usr/bin/env python3
"""Build the v2 research file: exact 35c9 + IMPL-1 + IMPL-2 + IMPL-3, every new flag default-off.

  stage 0: 04-train-research-35c9.py                         sha256 35c9ef79...
  stage 1: + IMPL-1 (tables bypass optimizer ownership)     sha256 9e5dce34...  (asserted)
  stage 2: + IMPL-2 --ngram-ve-freeze-epoch N (0 = off)
  stage 3: + IMPL-3 --fresh-tail-frac F (0 = off), --requeue-identity (control on the same loader)

Writes train.py (stage 3) and impl2.patch / impl3.patch (unified diffs stage1->2, stage2->3).
With all new flags at their defaults the file must be bit-identical in training to 35c9 (checked by
the G2 lockstep, see README.md).
"""
from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "04-train-research-35c9.py"
SHA0 = "35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2"
SHA1 = "9e5dce344674b8217d8749ca3e2ce8c1961a73a9c10c1c6759f645a62ed66783"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sub(s: str, old: str, new: str) -> str:
    n = s.count(old)
    assert n == 1, f"expected 1 match, found {n}: {old[:90]!r}"
    return s.replace(old, new)


def impl1(s: str) -> str:
    s = sub(s, """    for group_index, group in enumerate(groups):
        if group['kind'] not in ('adamw', 'muon'):
            raise ValueError('ownership probe is restricted to AdamW and Muon')""", """    for group_index, group in enumerate(groups):
        if group['kind'] == 'rmsprop':
            continue  # IMPL-1: n-gram tables stay replicated; OwnedOptimizer.step updates them on every rank
        if group['kind'] not in ('adamw', 'muon'):
            raise ValueError('ownership probe is restricted to AdamW and Muon')""")
    s = sub(s, """        opt = self.optimizer
        for source, owned in zip(opt.param_groups, self.groups):
            owned.update({key: value for key, value in source.items() if key != 'params'})""", """        opt = self.optimizer
        for source, owned in zip(opt.param_groups, self.groups):
            if source['kind'] == 'rmsprop':
                # IMPL-1: replicated table group. Its gradients were all-reduced (AVG) before this
                # call, so every rank applies the identical update, gated like MuonAdamW.step.
                if update_adamw:
                    opt._rmsprop_step(source)
                continue
            owned.update({key: value for key, value in source.items() if key != 'params'})""")
    return s


def impl2(s: str) -> str:
    s = sub(s, "EPOCH_SHUFFLE_SEED = 1234\n", """EPOCH_SHUFFLE_SEED = 1234
NGRAM_VE_FREEZE_EPOCH = 0
""")
    s = sub(s, "def main() -> None:\n", """def main() -> None:
    global NGRAM_VE_FREEZE_EPOCH, FRESH_TAIL_FRAC, REQUEUE_IDENTITY
""")
    s = sub(s, "    parser.add_argument('--epoch-shuffle-seed',", """    parser.add_argument('--ngram-ve-freeze-epoch', type=int, default=NGRAM_VE_FREEZE_EPOCH, help='IMPL-2: from the first step whose batch comes from loader epoch >= N, stop updating the n-gram tables (their gradients are dropped before the all-reduce; lookups continue). 0 (default) = off, bit-identical. The compiled graph is unchanged, so there is no recompilation.')
    parser.add_argument('--epoch-shuffle-seed',""")
    s = sub(s, "    USE_EPOCH_SHUFFLE = args.epoch_shuffle\n", """    USE_EPOCH_SHUFFLE = args.epoch_shuffle
    NGRAM_VE_FREEZE_EPOCH = args.ngram_ve_freeze_epoch
    if NGRAM_VE_FREEZE_EPOCH < 0:
        parser.error('--ngram-ve-freeze-epoch must be >= 0 (0 = off)')
    if NGRAM_VE_FREEZE_EPOCH and args.grad_bucket:
        parser.error('--ngram-ve-freeze-epoch cannot drop table gradients from a flat --grad-bucket buffer')
""")
    s = sub(s, "    while step < args.num_steps:\n        elapsed = time.monotonic() - budget_started\n", """    tables_frozen_logged = False
    while step < args.num_steps:
        step_epoch = loader_state['epoch']
        elapsed = time.monotonic() - budget_started
""")
    s = sub(s, """            x, y, loader_state = fetch()
        skip_step = False
""", """            x, y, loader_state = fetch()
        if NGRAM_VE_FREEZE_EPOCH and step_epoch >= NGRAM_VE_FREEZE_EPOCH:
            for p in orig_model.ngram_embeds.parameters():
                p.grad = None
            if not tables_frozen_logged:
                tables_frozen_logged = True
                print0(f'ngram_ve: tables frozen from step {step} (loader epoch {step_epoch} >= {NGRAM_VE_FREEZE_EPOCH})', flush=True)
        skip_step = False
""")
    return s


def impl3(s: str) -> str:
    s = sub(s, "NGRAM_VE_FREEZE_EPOCH = 0\n", """NGRAM_VE_FREEZE_EPOCH = 0
FRESH_TAIL_FRAC = 0.0
REQUEUE_IDENTITY = False
""")
    s = sub(s, "def _document_batches_epoch_shuffle(", '''def _read_row_group_batches(path, rg_idx, tokenizer_batch_size, epoch):
    rg = pq.ParquetFile(path).read_row_group(rg_idx)
    rows = rg.column('text').to_pylist()
    for i in range(0, len(rows), tokenizer_batch_size):
        yield (rows[i:i + tokenizer_batch_size], epoch)

def _document_batches_identity(split, start, step, tokenizer_batch_size, shard_ids=None):
    """IMPL-3 control: this rank's row groups in their natural order every epoch (same walk as the
    fresh-tail order uses), so a --fresh-tail-frac arm is compared on the SAME loader implementation."""
    groups = list(_row_groups_for_split(split, start, step, shard_ids=shard_ids))
    epoch = 0
    while True:
        epoch += 1
        for path, rg_idx in groups:
            yield from _read_row_group_batches(path, rg_idx, tokenizer_batch_size, epoch)

def _document_batches_fresh_tail(split, start, step, tokenizer_batch_size, shard_ids=None):
    """IMPL-3 fresh-tail order. Split this rank's row groups into A (first 1-f) and B (last f) and walk
    A, A, B, then the full list once per further epoch. A run that ends inside B spends its lowest-LR
    tail on documents it has never seen instead of replaying the opening documents. Same data, same
    per-rank partition, different order. Epoch labels: A=1, A=2, B=2, then 3, 4, ...
    """
    groups = list(_row_groups_for_split(split, start, step, shard_ids=shard_ids))
    k = int(round(len(groups) * (1.0 - FRESH_TAIL_FRAC)))
    head, tail = groups[:k], groups[k:]
    print(f'fresh tail: rank {start} walks {len(head)} row groups twice, then {len(tail)} unseen ones (frac={FRESH_TAIL_FRAC})', flush=True)
    if not tail:
        raise ValueError(f'--fresh-tail-frac {FRESH_TAIL_FRAC} leaves no tail row groups on rank {start} ({len(groups)} groups); raise the fraction')
    for order, epoch in ((head, 1), (head, 2), (tail, 2)):
        for path, rg_idx in order:
            yield from _read_row_group_batches(path, rg_idx, tokenizer_batch_size, epoch)
    epoch = 2
    while True:
        epoch += 1
        for path, rg_idx in groups:
            yield from _read_row_group_batches(path, rg_idx, tokenizer_batch_size, epoch)

def _document_batches_epoch_shuffle(''')
    s = sub(s, "    parser.add_argument('--epoch-shuffle-seed',", """    parser.add_argument('--fresh-tail-frac', type=float, default=FRESH_TAIL_FRAC, help='IMPL-3: walk each rank\\'s row groups as A, A, B where B is the last FRAC of them, so the end of a ~1.7-epoch run trains on unseen documents. 0 (default) = off (organizer loader, bit-identical). Uses the in-file requeue loader; compare against --requeue-identity.')
    parser.add_argument('--requeue-identity', action=argparse.BooleanOptionalAction, default=REQUEUE_IDENTITY, help='IMPL-3 control: the same in-file requeue loader as --fresh-tail-frac, natural row-group order. Default off.')
    parser.add_argument('--epoch-shuffle-seed',""")
    s = sub(s, "    NGRAM_VE_FREEZE_EPOCH = args.ngram_ve_freeze_epoch\n", """    NGRAM_VE_FREEZE_EPOCH = args.ngram_ve_freeze_epoch
    FRESH_TAIL_FRAC = args.fresh_tail_frac
    REQUEUE_IDENTITY = args.requeue_identity
    if not 0.0 <= FRESH_TAIL_FRAC <= 0.5:
        parser.error('--fresh-tail-frac must be in [0, 0.5]')
    if sum((bool(FRESH_TAIL_FRAC), REQUEUE_IDENTITY, USE_EPOCH_SHUFFLE)) > 1:
        parser.error('--fresh-tail-frac, --requeue-identity and --epoch-shuffle are mutually exclusive data orders')
""")
    s = sub(s, """    if USE_EPOCH_SHUFFLE:
        train_loader = make_dataloader_requeue(tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, 'train', device, _document_batches_fn=_document_batches_epoch_shuffle)
    else:""", """    if USE_EPOCH_SHUFFLE:
        train_loader = make_dataloader_requeue(tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, 'train', device, _document_batches_fn=_document_batches_epoch_shuffle)
    elif FRESH_TAIL_FRAC > 0.0:
        print0(f'data order: fresh tail, frac={FRESH_TAIL_FRAC}')
        train_loader = make_dataloader_requeue(tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, 'train', device, _document_batches_fn=_document_batches_fresh_tail)
    elif REQUEUE_IDENTITY:
        print0('data order: identity (requeue loader control)')
        train_loader = make_dataloader_requeue(tokenizer, DEVICE_BATCH_SIZE, loader_seq_len, 'train', device, _document_batches_fn=_document_batches_identity)
    else:""")
    return s


def udiff(a: str, b: str) -> str:
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), "a/train.py", "b/train.py", n=3))


def main() -> int:
    s0 = SRC.read_text()
    assert sha(s0) == SHA0, "source is not the exact 35c9 file"
    s1 = impl1(s0)
    assert sha(s1) == SHA1, f"IMPL-1 stage hash mismatch: {sha(s1)}"
    s2 = impl2(s1)
    s3 = impl3(s2)
    (HERE / "train.py").write_text(s3)
    (HERE / "impl2.patch").write_text(udiff(s1, s2))
    (HERE / "impl3.patch").write_text(udiff(s2, s3))
    for name, s in (("stage1 (IMPL-1)", s1), ("stage2 (+IMPL-2)", s2), ("stage3 (+IMPL-3) = train.py", s3)):
        print(f"{name:32s} sha256 {sha(s)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
