#!/usr/bin/env python3
"""Read-only training-corpus packing audit; never opens validation shards.

The length-only simulator preserves prepare.make_dataloader's document order,
128-document refill batches, 512-document low-water mark, first-best-fit tie
break, shortest-document fallback, and permanent tail dropping. It is a
diagnostic, not a replacement loader. --self-test needs CPU torch only; real
audits additionally need prepare.py's existing dependencies and cached assets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("loaded_documents", "loaded_encoded_tokens", "retained_documents",
          "retained_encoded_tokens", "retained_text_tokens", "truncated_documents",
          "dropped_text_tokens", "target_bos_tokens", "input_text_tokens")


@dataclass(frozen=True)
class Document:
    uid: int
    encoded_length: int  # encode_ordinary(document) plus exactly one BOS
    utf8_bytes: int = 0


def document_batches(groups, batch_size=128):
    """Same row-group and within-group batching order as _document_batches."""
    if not groups or not any(groups):
        raise ValueError("rank has no documents")
    for epoch in itertools.count(1):
        for docs in groups:
            for start in range(0, len(docs), batch_size):
                yield docs[start:start + batch_size], epoch


class PackingAudit:
    def __init__(self, batches, loader_seq_len, buffer_min=512):
        self.batches = iter(batches)
        self.capacity = loader_seq_len + 1
        self.buffer_min = buffer_min
        self.buffer = []
        self.by_epoch = defaultdict(Counter)
        self.rows = 0
        self.refills = 0
        self.refill_epoch = 0
        self.first_refill_row = {}
        self.first_retained_row = {}

    def pack_row(self, capture=False):
        pos = 0
        segments = [] if capture else None
        while pos < self.capacity:
            while len(self.buffer) < self.buffer_min:
                docs, epoch = next(self.batches)
                if not docs or any(d.encoded_length < 1 for d in docs):
                    raise ValueError("nonempty batches and BOS-inclusive positive lengths required")
                self.refill_epoch = epoch
                self.first_refill_row.setdefault(epoch, self.rows + 1)
                self.refills += 1
                self.buffer.extend((doc, epoch) for doc in docs)
                counts = self.by_epoch[epoch]
                counts["loaded_documents"] += len(docs)
                counts["loaded_encoded_tokens"] += sum(d.encoded_length for d in docs)
            remaining = self.capacity - pos
            best_idx, best_len = -1, 0
            for i, (doc, _epoch) in enumerate(self.buffer):
                n = doc.encoded_length
                if n <= remaining and n > best_len:
                    best_idx, best_len = i, n
            if best_idx >= 0:
                doc, epoch = self.buffer.pop(best_idx)
                take = doc.encoded_length
            else:
                shortest_idx = min(range(len(self.buffer)),
                                   key=lambda i: self.buffer[i][0].encoded_length)
                doc, epoch = self.buffer.pop(shortest_idx)
                take = remaining
            self.first_retained_row.setdefault(epoch, self.rows + 1)
            counts = self.by_epoch[epoch]
            counts["retained_documents"] += 1
            counts["retained_encoded_tokens"] += take
            counts["retained_text_tokens"] += take - 1
            counts["target_bos_tokens"] += int(pos > 0)
            # x omits the final packed token, y omits the initial BOS.
            counts["input_text_tokens"] += take - 1 - int(pos + take == self.capacity and take > 1)
            if take < doc.encoded_length:
                counts["truncated_documents"] += 1
                counts["dropped_text_tokens"] += doc.encoded_length - take
            if capture:
                segments.append((doc, epoch, take))
            pos += take
        self.rows += 1
        return segments

    def snapshot(self, raw_documents, raw_encoded_tokens):
        pending = defaultdict(Counter)
        for doc, epoch in self.buffer:
            pending[epoch]["pending_documents"] += 1
            pending[epoch]["pending_encoded_tokens"] += doc.encoded_length
        per_epoch = {}
        for epoch in sorted(set(self.by_epoch) | set(pending)):
            counts = {key: self.by_epoch[epoch][key] for key in FIELDS}
            counts.update({key: pending[epoch][key]
                           for key in ("pending_documents", "pending_encoded_tokens")})
            assert counts["loaded_encoded_tokens"] == (
                counts["retained_encoded_tokens"] + counts["dropped_text_tokens"]
                + counts["pending_encoded_tokens"])
            assert counts["loaded_documents"] == (
                counts["retained_documents"] + counts["pending_documents"])
            counts["not_yet_loaded_documents"] = raw_documents - counts["loaded_documents"]
            counts["not_yet_loaded_encoded_tokens"] = raw_encoded_tokens - counts["loaded_encoded_tokens"]
            assert counts["not_yet_loaded_documents"] >= 0
            assert counts["not_yet_loaded_encoded_tokens"] >= 0
            counts["pending_documents_longer_than_loader_bin"] = sum(
                doc.encoded_length > self.capacity for doc, source_epoch in self.buffer
                if source_epoch == epoch)
            counts["pending_max_document_encoded_length"] = max(
                (doc.encoded_length for doc, source_epoch in self.buffer if source_epoch == epoch),
                default=0)
            per_epoch[str(epoch)] = counts
        total = {key: sum(counts[key] for counts in per_epoch.values()) for key in FIELDS}
        assert total["retained_encoded_tokens"] == self.rows * self.capacity
        targets = self.rows * (self.capacity - 1)
        assert total["retained_text_tokens"] + total["target_bos_tokens"] == targets
        return {
            "loader_rows_produced": self.rows,
            "packed_tokens_union_of_x_and_y": total["retained_encoded_tokens"],
            "input_positions": targets,
            "target_positions": targets,
            "refill_batches": self.refills,
            "most_recent_refill_epoch": self.refill_epoch,
            "first_refill_row_1indexed": dict(self.first_refill_row),
            "first_retained_row_1indexed": dict(self.first_retained_row),
            "counts_over_all_source_epoch_occurrences": total,
            "by_source_epoch": per_epoch,
        }


def simulate(groups, steps, batch_size=1, pack_factor=4, seq_len=2048,
             world_size=4, total_batch_tokens=131072):
    global_micro = batch_size * pack_factor * seq_len * world_size
    if total_batch_tokens % global_micro:
        raise ValueError("global batch must divide the microbatch token count exactly")
    accum = total_batch_tokens // global_micro
    rows_per_step = accum * batch_size
    raw_docs = sum(len(group) for group in groups)
    raw_tokens = sum(doc.encoded_length for group in groups for doc in group)
    audit = PackingAudit(document_batches(groups), pack_factor * seq_len)
    # train.py primes one loader microbatch and fetches the next after EACH
    # backward, including the final backward. That last batch is not trained.
    wanted = {step * rows_per_step for step in steps}
    wanted |= {step * rows_per_step + batch_size for step in steps}
    snapshots = {}
    for row in range(1, max(wanted) + 1):
        audit.pack_row()
        if row in wanted:
            snapshots[row] = audit.snapshot(raw_docs, raw_tokens)
    return {
        "raw_corpus_once": {
            "documents": raw_docs, "encoded_tokens_including_bos": raw_tokens,
            "text_tokens_excluding_bos": raw_tokens - raw_docs,
            "utf8_bytes": sum(doc.utf8_bytes for group in groups for doc in group),
        },
        "checkpoints": {
            str(step): {
                "trained_prefix": snapshots[step * rows_per_step],
                "loader_after_untrained_prefetch": snapshots[step * rows_per_step + batch_size],
            } for step in steps
        },
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prepare():
    sys.path.insert(0, str(ROOT))
    import prepare
    return prepare


def self_test():
    # Tiny hand example: cap=6; choose length4, then no doc fits2. Refill
    # first (the low-water rule), truncate length7 to2 and permanently drop5.
    docs = [Document(i, length) for i, length in enumerate((4, 7, 9, 10))]
    audit = PackingAudit(iter(((docs[:3], 1), (docs[3:], 1))), 5, buffer_min=3)
    segments = audit.pack_row(capture=True)
    assert [(doc.uid, take) for doc, _epoch, take in segments] == [(0, 4), (1, 2)]
    snap = audit.snapshot(4, 30)
    one = snap["by_source_epoch"]["1"]
    assert (one["loaded_encoded_tokens"], one["retained_encoded_tokens"],
            one["dropped_text_tokens"], one["pending_encoded_tokens"]) == (30, 6, 5, 19)
    assert snap["target_positions"] == 5 and one["retained_text_tokens"] == 4
    assert one["target_bos_tokens"] == 1
    print("hand conservation: loaded30 = packed6 + dropped_tail5 + buffered19; targets5 = text4 + BOS1")

    # Execute the ORIGINAL loader function in an isolated namespace with fake
    # input globals. No monkeypatch, no production source edit, no real data.
    import torch
    prepare = load_prepare()
    bos = 8191
    lengths = (2, 3, 7, 17, 2049, 8201, 16390)
    groups = [[Document(g * 32 + i, lengths[(g * 32 + i) % len(lengths)])
               for i in range(32)] for g in range(24)]

    class FakeTokenizer:
        def get_bos_token_id(self):
            return bos

        def encode(self, docs, prepend, num_threads):
            return [[prepend] + list(range(10000 + doc.uid * 20000,
                                            10000 + doc.uid * 20000 + doc.encoded_length - 1))
                    for doc in docs]

    def batches_for_rank(split, start, step, tokenizer_batch_size, shard_ids=None):
        assert split == "train" and shard_ids is None
        return document_batches(groups[start::step], tokenizer_batch_size)

    for rank in range(4):
        namespace = {
            "torch": torch, "Tokenizer": FakeTokenizer, "Iterable": object,
            "get_dist_info": lambda rank=rank: (True, rank, rank, 4),
            "_document_batches": batches_for_rank,
        }
        exec("from __future__ import annotations\n" + inspect.getsource(prepare.make_dataloader), namespace)
        original = namespace["make_dataloader"](FakeTokenizer(), 1, 4 * 2048, "train", "cpu")
        mine = PackingAudit(document_batches(groups[rank::4]), 4 * 2048)
        for _ in range(8):
            pieces = mine.pack_row(capture=True)
            expected = []
            for doc, _epoch, take in pieces:
                expected += [bos] + list(range(10000 + doc.uid * 20000,
                                               10000 + doc.uid * 20000 + take - 1))
            x, y, state = next(original)
            assert x.tolist() == [expected[:-1]]
            assert y.tolist() == [expected[1:]]
            assert state["epoch"] == mine.refill_epoch
            mine.snapshot(sum(len(g) for g in groups[rank::4]),
                          sum(d.encoded_length for g in groups[rank::4] for d in g))
        # The tiny rank corpus has <512 documents; original refill runs across
        # multiple epochs before the first row is emitted. This catches the
        # common mistake of treating refill epoch as completed presentation.
        assert mine.first_refill_row[2] == 1
        assert mine.by_epoch[1]["loaded_documents"] > mine.by_epoch[1]["retained_documents"]
    result = simulate(groups[0::4], [1, 2])
    assert result["checkpoints"]["1"]["trained_prefix"]["target_positions"] == 4 * 8192
    assert result["checkpoints"]["1"]["loader_after_untrained_prefetch"]["target_positions"] == 5 * 8192
    print("PASS: original prepare.make_dataloader CPU outputs/epoch state match exactly for 4 ranks x 8 locked-geometry rows")
    print("PASS: source-epoch conservation, repeated refills, stable selection, tail dropping, BOS accounting, and untrained prefetch")


def real_audit(args):
    prepare = load_prepare()
    # No ensure_data/ensure_tokenizer calls: these can download, delete, or
    # refresh assets. This diagnostic requires existing organizer assets.
    prepare.verify_tokenizer_cache(prepare.TOKENIZER_DIR)
    tokenizer = prepare.Tokenizer.from_directory(prepare.TOKENIZER_DIR)
    paths = prepare.list_shard_paths(prepare.PUBLIC_TRAIN_SHARDS)
    if prepare.PUBLIC_TRAIN_SHARDS != list(range(5)):
        raise RuntimeError("review changed public-training-shard contract before auditing")
    result = {
        "mode": "training_corpus_length_only_audit",
        "definitions": {
            "raw_encoded": "encode_ordinary(text) + one BOS per document, counted once per source epoch",
            "retained": "union of packed input and target positions; first BOS is input-only",
            "dropped": "text suffix permanently discarded by shortest-document fallback",
            "pending": "encoded documents in the live >=512-document refill reservoir",
            "unread": "documents of that source epoch not yet pulled into the reservoir",
            "epoch": "latest minibatch refill source epoch, NOT fully consumed corpus",
            "trained_prefix": "first steps*grad_accum loader microbatches actually used by backward",
            "prefetch": "one additional batch fetched after the final backward, not trained",
            "utf8_bytes": "actual UTF-8 byte length; not Python Unicode character count",
        },
        "geometry": {
            "device_batch_size": args.batch_size, "pack_factor": args.pack_factor,
            "model_seq_len": args.seq_len, "loader_row_capacity": args.pack_factor * args.seq_len + 1,
            "world_size": args.world_size, "total_batch_tokens": args.total_batch_tokens,
            "tokenizer_batch_size": 128, "buffer_min": 512,
            "tokenizer_threads": args.tokenizer_threads, "step_checkpoints": args.steps,
            "epoch_shuffle": False, "batch_ramp": False,
        },
        "source_sha256": {name: sha256(ROOT / name) for name in ("prepare.py", "train.py")},
        "tokenizer_sha256": sha256(prepare.TOKENIZER_DIR / "tokenizer.pkl"),
        "training_shards": [{"name": p.name, "bytes_on_disk": p.stat().st_size,
                            "mtime_ns": p.stat().st_mtime_ns} for p in paths],
        "ranks": {},
    }
    started = time.monotonic()
    for rank in range(args.world_size):
        row_groups = prepare._row_groups_for_split("train", rank, args.world_size)
        groups = []
        encoded = 0
        for ordinal, (path, rg_idx) in enumerate(row_groups):
            rows = prepare.pq.ParquetFile(path).read_row_group(rg_idx, columns=["text"]).column("text").to_pylist()
            docs = []
            for start in range(0, len(rows), 128):
                texts = rows[start:start + 128]
                token_lists = tokenizer.encode(texts, prepend=tokenizer.get_bos_token_id(),
                                               num_threads=args.tokenizer_threads)
                docs.extend(Document((ordinal << 32) + start + i, len(tokens), len(text.encode("utf-8")))
                            for i, (text, tokens) in enumerate(zip(texts, token_lists)))
                encoded += sum(map(len, token_lists))
            groups.append(docs)
            print(f"rank={rank} row_group={ordinal+1}/{len(row_groups)} raw_encoded={encoded:,} "
                  f"elapsed={time.monotonic()-started:.1f}s", file=sys.stderr, flush=True)
        result["ranks"][str(rank)] = simulate(groups, args.steps, args.batch_size, args.pack_factor,
                                              args.seq_len, args.world_size, args.total_batch_tokens)
        result["ranks"][str(rank)]["assigned_row_groups"] = [
            {"shard": path.name, "row_group": rg_idx} for path, rg_idx in row_groups]
    result["wall_seconds"] = time.monotonic() - started
    result["all_ranks_raw_corpus_once"] = {
        key: sum(rank["raw_corpus_once"][key] for rank in result["ranks"].values())
        for key in ("documents", "encoded_tokens_including_bos", "text_tokens_excluding_bos", "utf8_bytes")}
    result["all_ranks_exposure"] = {}
    for step in args.steps:
        result["all_ranks_exposure"][str(step)] = {}
        for kind in ("trained_prefix", "loader_after_untrained_prefetch"):
            rank_snaps = [rank["checkpoints"][str(step)][kind] for rank in result["ranks"].values()]
            totals = {key: sum(snap["counts_over_all_source_epoch_occurrences"][key]
                               for snap in rank_snaps) for key in FIELDS}
            totals["target_positions"] = sum(snap["target_positions"] for snap in rank_snaps)
            first_epoch_fields = (*FIELDS, "pending_documents", "pending_encoded_tokens",
                                  "not_yet_loaded_documents", "not_yet_loaded_encoded_tokens")
            totals["first_source_epoch"] = {
                key: sum(snap["by_source_epoch"]["1"][key] for snap in rank_snaps)
                for key in first_epoch_fields}
            result["all_ranks_exposure"][str(step)][kind] = totals
        assert result["all_ranks_exposure"][str(step)]["trained_prefix"]["target_positions"] == (
            step * args.total_batch_tokens)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--steps", default="2100,2135,2195,2800")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pack-factor", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--total-batch-tokens", type=int, default=131072)
    parser.add_argument("--tokenizer-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, help="new JSON result path; refuses overwrite")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    args.steps = sorted({int(step) for step in args.steps.split(",")})
    if not args.steps or min(args.steps) < 1:
        parser.error("--steps must contain positive completed-step counts")
    if min(args.batch_size, args.pack_factor, args.world_size, args.tokenizer_threads,
           args.total_batch_tokens) < 1 or args.seq_len < 2:
        parser.error("geometry/thread counts must be positive, sequence length at least two")
    if args.total_batch_tokens % (args.batch_size * args.pack_factor * args.seq_len * args.world_size):
        parser.error("global batch must divide the microbatch token count exactly")
    if args.output and args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = real_audit(args)
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
