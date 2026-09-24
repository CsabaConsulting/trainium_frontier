"""TEST-ONLY stand-in for the organizer's prepare.py.

It exists so the G2 preflight harness and the IMPL-1 tests can run on a machine that has neither
the real prepare.py nor the data shards (for example a CI box or this repository's sandbox).
It provides the names that train.py imports, with synthetic data and no scoring semantics.

Never place this file next to a train.py that is being trained or scored. On the Trn2 hosts
and the controller workstation, the preflight harness must import the organizer's real
prepare.py (the default); `--stub-prepare` is an explicit opt-in for offline checks only.
"""
from __future__ import annotations

import os
import random

import torch
import torch.distributed as dist

EVAL_TIMEOUT_SECONDS = 1200
TOKENIZER_VOCAB_SIZE = 8192
TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS = 900
TRAIN_STARTUP_STEPS_EXCLUDED = 2
TRAIN_TIME_BUDGET_SECONDS = 1800
STUB = True


def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return True, int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 0, 1


def print0(*args, **kwargs):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


def record_step_boundary(step):
    return None


class Tokenizer:
    """Synthetic tokenizer: ids in [1, vocab); BOS is id 0."""

    def __init__(self, vocab_size: int = TOKENIZER_VOCAB_SIZE):
        self.vocab_size = vocab_size

    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def encode(self, docs, prepend=None, num_threads=1):
        out = []
        for d in docs:
            rng = random.Random(hash(d) & 0xFFFFFFFF)
            ids = [rng.randrange(1, self.vocab_size) for _ in range(rng.randrange(8, 400))]
            if prepend is not None:
                ids.insert(0, prepend if isinstance(prepend, int) else 0)
            out.append(ids)
        return out


def ensure_tokenizer(build_if_missing: bool = False) -> Tokenizer:
    return Tokenizer()


def get_token_bytes(device="cpu"):
    tb = torch.full((TOKENIZER_VOCAB_SIZE,), 4, dtype=torch.int32, device=device)
    tb[0] = 0
    return tb


def _row_groups_for_split(split, start, step, shard_ids=None):
    return [(f"synthetic-{split}", i) for i in range(start, 64, step)]


def _document_batches(split, start, step, tokenizer_batch_size=128, shard_ids=None):
    epoch = 0
    while True:
        epoch += 1
        for _, rg in _row_groups_for_split(split, start, step, shard_ids):
            docs = [f"{split}-{rg}-{i}" for i in range(tokenizer_batch_size)]
            yield docs, epoch


def make_dataloader(tokenizer, batch_size, seq_len, split, device, **_):
    g = torch.Generator().manual_seed(1234 + (dist.get_rank() if dist.is_initialized() else 0))
    while True:
        x = torch.randint(1, TOKENIZER_VOCAB_SIZE, (batch_size, seq_len), generator=g, dtype=torch.int64)
        y = torch.randint(1, TOKENIZER_VOCAB_SIZE, (batch_size, seq_len), generator=g, dtype=torch.int64)
        yield x.to(torch.int32).to(device), y.to(device), {"epoch": 1}


def causality_check(*args, **kwargs):
    return {"passed": True, "max_abs_diff": 0.0}


def evaluate_bpb(*args, **kwargs):
    raise RuntimeError("stub prepare.py: evaluate_bpb is not available offline")
