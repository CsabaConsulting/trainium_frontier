#!/usr/bin/env python3
"""Build and falsify a conservative tokenizer-derived continuation mask on CPU.

This is a research probe, not a change to the tokenizer, trainer, or scorer.
Only an ASCII-letter token of length >=3 followed by another ASCII-letter token
is considered. The ascii-word extension also permits a single ASCII non-letter/
non-number prefix, excluding apostrophe and CR/LF. Under the fixed pretokenizer
these pairs must be in one letter run:
the previous token cannot be a suffix of a contraction (at most two letters).
The substring property of ordinary BPE then permits a two-token re-encoding test.
Unicode, numeric, special, ambiguous short-left, and other unproven boundary cases
remain unrestricted. Empirical checks supplement, not replace, that argument.

Motivation: Vieira et al., https://arxiv.org/abs/2506.07956 (ICML 2025).
The general GPT-2 mask in modded-nanogpt PR350 is not copied: its pretokenizer
differs from this competition's fixed tokenizer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import prepare


def build_mask(tokenizer, scope="ascii-long"):
    enc = tokenizer.enc
    if enc._pat_str != prepare.SPLIT_PATTERN:
        raise ValueError("the conservative seam argument requires the fixed split pattern")
    letters = [(b.decode("ascii"), tid) for b, tid in enc._mergeable_ranks.items()
               if b.isascii() and b.isalpha()]
    previous = [(s, tid) for s, tid in letters if len(s) >= 3]
    if scope == "ascii-word":
        # One ASCII non-letter/non-number prefix is permitted by the exact
        # letter-run regex. Exclude apostrophes and newlines altogether rather
        # than attempting to model contextual contraction/boundary behavior.
        for b, tid in enc._mergeable_ranks.items():
            if (b.isascii() and len(b) >= 4 and b[1:].isalpha()
                    and not b[:1].isalnum() and b[:1] not in (b"'", b"\r", b"\n")):
                previous.append((b.decode("ascii"), tid))
    elif scope != "ascii-long":
        raise ValueError(scope)
    for s, tid in letters + previous:
        assert enc.encode_ordinary(s) == [tid], (s, tid)
    mask = np.zeros((enc.n_vocab, enc.n_vocab), dtype=np.bool_)
    encode = enc.encode_ordinary
    started = time.perf_counter()
    for i, (left, left_id) in enumerate(previous):
        for right, right_id in letters:
            mask[left_id, right_id] = encode(left + right) != [left_id, right_id]
        if (i + 1) % 500 == 0:
            print("CANONICAL_BUILD_PROGRESS " + json.dumps(dict(
                rows=i + 1, total_rows=len(previous), elapsed=time.perf_counter() - started)),
                flush=True)
    return mask, dict(previous_tokens=len(previous), following_tokens=len(letters),
                      tested_pairs=len(previous) * len(letters), forbidden_pairs=int(mask.sum()),
                      build_seconds=time.perf_counter() - started)


def check_texts(tokenizer, mask, texts, source):
    encoded = tokenizer.encode(texts, prepend=tokenizer.bos_token_id, num_threads=4)
    count = 0
    for doc_number, ids in enumerate(encoded):
        a = np.asarray(ids, dtype=np.int64)
        bad = np.flatnonzero(mask[a[:-1], a[1:]])
        if len(bad):
            at = int(bad[0])
            raise AssertionError(dict(source=source, document=doc_number,
                                      token_offset=at, forbidden_valid_pair=ids[at:at + 2]))
        count += len(ids)
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scope", choices=["ascii-long", "ascii-word"], default="ascii-long",
                    help="ascii-word also permits one unambiguous non-letter/non-digit prefix")
    ap.add_argument("--parquet", type=Path, action="append", default=[])
    ap.add_argument("--max-training-tokens", type=int, default=5_000_000)
    ap.add_argument("--fuzz-documents", type=int, default=50_000)
    args = ap.parse_args()
    if args.output.suffix != ".npy":
        ap.error("--output must end in .npy")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        ap.error("refusing to overwrite existing probe artifacts")
    allowed_names = {prepare.shard_filename(i) for i in prepare.PUBLIC_TRAIN_SHARDS}
    if any(p.name not in allowed_names for p in args.parquet):
        ap.error("this probe accepts only filenames from the provided training shards")
    tokenizer = prepare.ensure_tokenizer(build_if_missing=False)
    mask, result = build_mask(tokenizer, args.scope)
    special_ids = list(tokenizer.enc._special_tokens.values())
    assert not mask[special_ids].any() and not mask[:, special_ids].any()
    rng = random.Random(42)
    alphabetic = [b.decode("ascii") for b in tokenizer.enc._mergeable_ranks
                  if b.isascii() and b.isalpha()]
    prefixes = ["", "'", "a'", ".'", "...", " ", "\n", "99", "Ж", "\u0301", "\x00"]
    suffixes = ["", "'", "bar", "\n", "123", "🙂", " ", "Z", "\u0301"]
    fixed = ["can'treally", "i'lllook", "a'Silly", "WE'REthere", "foo1234bar",
             "a\n\nb", "\\.\n\n", "x'reference", "l'llook", "\x00reader",
             "Жreaders🙂", "<|reserved_0|>alphabet"]
    result["fuzz_tokens"] = check_texts(tokenizer, mask, fixed, "fixed adversarial contexts")
    for start in range(0, args.fuzz_documents, 1000):
        texts = [rng.choice(prefixes) + rng.choice(alphabetic) + rng.choice(alphabetic)
                 + rng.choice(suffixes) for _ in range(min(1000, args.fuzz_documents - start))]
        result["fuzz_tokens"] += check_texts(tokenizer, mask, texts, "synthetic context fuzz")
    result["fuzz_documents"] = args.fuzz_documents + len(fixed)
    result["training_tokens_checked"] = result["training_documents_checked"] = 0
    result["training_sources"] = []
    if args.parquet:
        import pyarrow.parquet as pq
        for path in args.parquet:
            entry = dict(path=str(path), sha256=prepare.sha256_file(path), documents=0, tokens=0)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text"]):
                texts = batch.column(0).to_pylist()
                count = check_texts(tokenizer, mask, texts, str(path))
                entry["tokens"] += count
                entry["documents"] += len(texts)
                result["training_tokens_checked"] += count
                result["training_documents_checked"] += len(texts)
                if result["training_tokens_checked"] >= args.max_training_tokens:
                    break
            result["training_sources"].append(entry)
            if result["training_tokens_checked"] >= args.max_training_tokens:
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A deterministic, tokenizer-derived research artifact, not a learned lookup.
    np.save(args.output, mask, allow_pickle=False)
    result.update(tokenizer_sha256=prepare.TOKENIZER_PKL_SHA256,
                  tokenizer_pattern=tokenizer.enc._pat_str, bos_id=tokenizer.bos_token_id,
                  mask_shape=list(mask.shape), mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                  forbidden_valid_pairs=0, quality_measured=False,
                  scope=args.scope,
                  previous_rule="at least three ASCII letters; ascii-word optionally permits one "
                                "non-alphanumeric prefix except apostrophe/CR/LF",
                  following_rule="one or more ASCII letters")
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print("CANONICAL_PROBE_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
