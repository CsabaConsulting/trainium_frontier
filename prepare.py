"""
Fixed Neuron Competition R1 preparation and evaluation utilities.

Public participants can read this file and use it during development. Official
scoring should use the organizer copy of this file, not a submitted copy.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import pickle
import shlex
import shutil
import sys
import time
import urllib.request
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import requests
import tiktoken
import torch
import torch.distributed as dist
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Fixed public constants

TRAIN_TIME_BUDGET_SECONDS = 30 * 60
EVAL_TIMEOUT_SECONDS = 20 * 60

CACHE_DIR = Path(os.environ.get("NEURON_COMPETITION_R1_CACHE_DIR", Path.home() / ".cache" / "neuron_competition_r1"))
DATA_DIR = CACHE_DIR / "data"
TOKENIZER_DIR = CACHE_DIR / "tokenizer"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
PUBLIC_TRAIN_SHARDS = list(range(5))
PUBLIC_VAL_SHARDS = [MAX_SHARD]

TOKENIZER_VOCAB_SIZE = 8192
TOKENIZER_ARTIFACT_PATH = Path(os.environ.get("NEURON_COMPETITION_R1_TOKENIZER_PATH", ARTIFACTS_DIR / "tokenizer_8192.zip"))
TOKENIZER_ARTIFACT_URL = os.environ.get("NEURON_COMPETITION_R1_TOKENIZER_URL", "")
TOKENIZER_ARTIFACT_SHA256 = os.environ.get(
    "NEURON_COMPETITION_R1_TOKENIZER_SHA256",
    "12fa2c4a56d9f26d46e8ea0a57f69d78145930eecb0a765c9bfc085176af8a54",
)
TOKENIZER_PKL_SHA256 = os.environ.get(
    "NEURON_COMPETITION_R1_TOKENIZER_PKL_SHA256",
    "aacd6c6a838eaec2c662a7d0cba0b5412f930a6ddee95b9ecb2c8fac53f3dafc",
)
TOKEN_BYTES_SHA256 = os.environ.get(
    "NEURON_COMPETITION_R1_TOKEN_BYTES_SHA256",
    "3de03cf6f8e16a9397fecd0cebd04f435f11994b215a5c785787fcc15adb0e15",
)

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"


# -----------------------------------------------------------------------------
# Small runtime helpers

def shard_filename(index: int) -> str:
    return f"shard_{index:05d}.parquet"


def print0(*args, **kwargs) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


def get_dist_info() -> tuple[bool, int, int, int]:
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return True, int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 0, 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def torch_load_weights(path: Path, map_location):
    return torch.load(path, map_location=map_location, weights_only=True)


# -----------------------------------------------------------------------------
# Data download

def download_single_shard(index: int) -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = shard_filename(index)
    path = DATA_DIR / filename
    if path.exists():
        return True

    url = f"{BASE_URL}/{filename}"
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(1, 6):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(path)
            print(f"downloaded {filename}")
            return True
        except Exception as exc:
            print(f"attempt {attempt}/5 failed for {filename}: {exc}")
            for p in (tmp, path):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            if attempt < 5:
                time.sleep(2**attempt)
    return False


def ensure_data(num_train_shards: int = len(PUBLIC_TRAIN_SHARDS), workers: int = 8) -> None:
    """Download train shards [0, num_train_shards) plus public dev val shards."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = min(max(num_train_shards, 1), len(PUBLIC_TRAIN_SHARDS))
    shard_ids = PUBLIC_TRAIN_SHARDS[:n]
    for val_id in PUBLIC_VAL_SHARDS:
        if val_id not in shard_ids:
            shard_ids.append(val_id)

    missing = [i for i in shard_ids if not (DATA_DIR / shard_filename(i)).exists()]
    if not missing:
        print(f"data ready: {len(shard_ids)} shards at {DATA_DIR}")
        return

    print(f"downloading {len(missing)} missing shards to {DATA_DIR}")
    with Pool(processes=max(1, min(workers, len(missing)))) as pool:
        results = pool.map(download_single_shard, missing)
    if not all(results):
        raise RuntimeError("failed to download all requested shards")


def list_shard_paths(shard_ids: Iterable[int] | None = None, require_all: bool = True) -> list[Path]:
    if shard_ids is None:
        paths = sorted(DATA_DIR.glob("shard_*.parquet"))
    else:
        paths = [DATA_DIR / shard_filename(i) for i in list(shard_ids)]
        missing = [p for p in paths if not p.exists()]
        if missing and require_all:
            names = ", ".join(p.name for p in missing[:8])
            if len(missing) > 8:
                names += f", ... ({len(missing)} missing)"
            raise FileNotFoundError(f"missing required shard(s): {names}; run `python prepare.py setup`")
    return [p for p in paths if p.exists()]


# -----------------------------------------------------------------------------
# Tokenizer artifact / development fallback

class Tokenizer:
    def __init__(self, enc: tiktoken.Encoding):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(
        cls, directory: Path = TOKENIZER_DIR, *, expected_sha: str = TOKENIZER_PKL_SHA256
    ) -> "Tokenizer":
        tok_path = directory / "tokenizer.pkl"
        if not expected_sha:
            raise RuntimeError(
                f"refusing to load {tok_path}: no expected SHA256 provided "
                "(integrity verification cannot be skipped)"
            )
        actual = sha256_file(tok_path)
        if actual != expected_sha:
            raise RuntimeError(
                f"tokenizer.pkl SHA mismatch at {tok_path}: expected {expected_sha}, got {actual}"
            )
        with tok_path.open("rb") as f:
            enc = pickle.load(f)
        tok = cls(enc)
        if tok.get_vocab_size() != TOKENIZER_VOCAB_SIZE:
            raise ValueError(f"expected vocab size {TOKENIZER_VOCAB_SIZE}, got {tok.get_vocab_size()}")
        return tok

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads: int = 8):
        prepend_id = None
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            return ids
        if isinstance(text, list):
            rows = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend_id is not None:
                for row in rows:
                    row.insert(0, prepend_id)
            return rows
        raise TypeError(f"unsupported text type: {type(text)}")

    def decode(self, ids) -> str:
        return self.enc.decode(ids)


def _save_tokenizer(enc: tiktoken.Encoding) -> None:
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    with (TOKENIZER_DIR / "tokenizer.pkl").open("wb") as f:
        pickle.dump(enc, f)

    special_ids = {enc.encode_single_token(token) for token in SPECIAL_TOKENS}
    token_bytes = []
    for token_id in range(enc.n_vocab):
        if token_id in special_ids:
            token_bytes.append(0)
        else:
            token_bytes.append(len(enc.decode_single_token_bytes(token_id)))
    torch.save(torch.tensor(token_bytes, dtype=torch.int32), TOKENIZER_DIR / "token_bytes.pt")


def _verify_token_bytes(enc: tiktoken.Encoding, token_bytes: torch.Tensor) -> None:
    if token_bytes.numel() != enc.n_vocab:
        raise RuntimeError(f"token_bytes length mismatch: expected {enc.n_vocab}, got {token_bytes.numel()}")
    special_ids = {enc.encode_single_token(token) for token in SPECIAL_TOKENS}
    expected = []
    for token_id in range(enc.n_vocab):
        if token_id in special_ids:
            expected.append(0)
        else:
            expected.append(len(enc.decode_single_token_bytes(token_id)))
    expected_tensor = torch.tensor(expected, dtype=token_bytes.dtype)
    diff = (token_bytes.cpu() != expected_tensor).nonzero().flatten()
    if diff.numel():
        bad = int(diff[0])
        raise RuntimeError(
            f"token_bytes mismatch at token {bad}: expected {int(expected_tensor[bad])}, "
            f"got {int(token_bytes.cpu()[bad])}"
        )


def verify_tokenizer_cache(directory: Path = TOKENIZER_DIR) -> None:
    tok_path = directory / "tokenizer.pkl"
    bytes_path = directory / "token_bytes.pt"
    if not tok_path.exists() or not bytes_path.exists():
        raise FileNotFoundError(f"missing tokenizer files under {directory}")
    if TOKEN_BYTES_SHA256 and sha256_file(bytes_path) != TOKEN_BYTES_SHA256:
        raise RuntimeError(f"token_bytes.pt SHA mismatch at {bytes_path}")
    tok = Tokenizer.from_directory(directory)
    token_bytes = torch_load_weights(bytes_path, map_location="cpu")
    _verify_token_bytes(tok.enc, token_bytes)


def _download_tokenizer_artifact() -> bool:
    local_path = TOKENIZER_ARTIFACT_PATH
    artifact_path = TOKENIZER_DIR / "tokenizer_artifact.zip"

    if local_path.exists():
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = local_path
    elif TOKENIZER_ARTIFACT_URL:
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading tokenizer artifact from {TOKENIZER_ARTIFACT_URL}")
        urllib.request.urlretrieve(TOKENIZER_ARTIFACT_URL, artifact_path)
    else:
        return False

    if TOKENIZER_ARTIFACT_SHA256:
        actual = sha256_file(artifact_path)
        if actual != TOKENIZER_ARTIFACT_SHA256:
            raise RuntimeError(f"tokenizer artifact SHA mismatch: expected {TOKENIZER_ARTIFACT_SHA256}, got {actual}")

    shutil.unpack_archive(str(artifact_path), str(TOKENIZER_DIR))
    return True


def _tokenizer_text_iterator(max_chars: int, doc_cap: int):
    nchars = 0
    train_paths = list_shard_paths(PUBLIC_TRAIN_SHARDS)
    if not train_paths:
        raise RuntimeError("no train shards available for tokenizer fallback")
    for path in train_paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap]
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def build_tokenizer_fallback(max_chars: int = 1_000_000_000, doc_cap: int = 10_000) -> None:
    """Development fallback. Official starter should use a fixed artifact."""
    import rustbpe  # dev-only: lazy import so normal train/eval does not require rustbpe
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"building development tokenizer fallback: vocab={TOKENIZER_VOCAB_SIZE}, max_chars={max_chars:,}")
    bpe = rustbpe.Tokenizer()
    bpe.train_from_iterator(
        _tokenizer_text_iterator(max_chars=max_chars, doc_cap=doc_cap),
        TOKENIZER_VOCAB_SIZE - len(SPECIAL_TOKENS),
        pattern=SPLIT_PATTERN,
    )
    mergeable_ranks = {bytes(k): v for k, v in bpe.get_mergeable_ranks()}
    offset = len(mergeable_ranks)
    special_tokens = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="neuron_competition_r1_bpe",
        pat_str=bpe.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    _save_tokenizer(enc)

    test = "Hello world! Numbers: 123. Unicode: hello."
    assert enc.decode(enc.encode_ordinary(test)) == test
    print(f"tokenizer saved to {TOKENIZER_DIR}")


def ensure_tokenizer(build_if_missing: bool = False) -> Tokenizer:
    tok_path = TOKENIZER_DIR / "tokenizer.pkl"
    bytes_path = TOKENIZER_DIR / "token_bytes.pt"
    if tok_path.exists() and bytes_path.exists():
        try:
            verify_tokenizer_cache(TOKENIZER_DIR)
            return Tokenizer.from_directory(TOKENIZER_DIR)
        except Exception as exc:
            print(f"cached tokenizer invalid, refreshing from artifact: {exc}")
            for path in (tok_path, bytes_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
    if not _download_tokenizer_artifact():
        if not build_if_missing:
            raise FileNotFoundError(
                f"fixed tokenizer not found at {TOKENIZER_DIR}; run setup with --build-tokenizer-if-missing "
                "for development, or set NEURON_COMPETITION_R1_TOKENIZER_URL"
            )
        # Dev fallback: a self-built tokenizer cannot match the hardcoded official
        # hashes, so verify it against the bytes we just wrote rather than skipping
        # verification. Every load path still passes a real SHA check.
        build_tokenizer_fallback()
        tok = Tokenizer.from_directory(TOKENIZER_DIR, expected_sha=sha256_file(tok_path))
        _verify_token_bytes(tok.enc, torch_load_weights(bytes_path, map_location="cpu"))
        return tok
    verify_tokenizer_cache(TOKENIZER_DIR)
    return Tokenizer.from_directory(TOKENIZER_DIR)


def get_token_bytes(device: torch.device | str = "cpu") -> torch.Tensor:
    path = TOKENIZER_DIR / "token_bytes.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing token bytes: {path}")
    return torch_load_weights(path, map_location=device)


# -----------------------------------------------------------------------------
# Dataloader

def _paths_for_split(split: str, shard_ids: Iterable[int] | None = None) -> list[Path]:
    if shard_ids is not None:
        paths = list_shard_paths(shard_ids)
    elif split == "train":
        paths = list_shard_paths(PUBLIC_TRAIN_SHARDS)
    elif split == "public_val":
        paths = list_shard_paths(PUBLIC_VAL_SHARDS)
    else:
        raise ValueError(f"unknown split {split!r}; pass shard_ids for private eval")
    if not paths:
        raise RuntimeError(f"no parquet files found for split={split}")
    return paths


def _row_groups_for_split(split: str, start: int, step: int, shard_ids=None) -> list[tuple[Path, int]]:
    paths = _paths_for_split(split, shard_ids=shard_ids)
    all_groups: list[tuple[Path, int]] = []
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            all_groups.append((path, rg_idx))
    if not all_groups:
        raise RuntimeError(f"no row groups for split={split}")
    return [all_groups[i] for i in range(start, len(all_groups), step)]


def _document_batches(split: str, start: int, step: int, tokenizer_batch_size: int, shard_ids=None):
    groups = _row_groups_for_split(split, start, step, shard_ids=shard_ids)
    epoch = 0
    while True:
        epoch += 1
        for path, rg_idx in groups:
            rg = pq.ParquetFile(path).read_row_group(rg_idx)
            rows = rg.column("text").to_pylist()
            for i in range(0, len(rows), tokenizer_batch_size):
                yield rows[i:i + tokenizer_batch_size], epoch


def _document_batches_once(split: str, start: int, step: int, tokenizer_batch_size: int, shard_ids=None):
    groups = _row_groups_for_split(split, start, step, shard_ids=shard_ids)
    for path, rg_idx in groups:
        rg = pq.ParquetFile(path).read_row_group(rg_idx)
        rows = rg.column("text").to_pylist()
        for i in range(0, len(rows), tokenizer_batch_size):
            yield rows[i:i + tokenizer_batch_size]


def make_dataloader(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    device: torch.device | str,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    shard_ids: Iterable[int] | None = None,
):
    """BOS-aligned best-fit packed token dataloader."""
    assert batch_size > 0 and seq_len > 1
    ddp, rank, _local_rank, world_size = get_dist_info()
    batches = _document_batches(split, rank, world_size, tokenizer_batch_size, shard_ids=shard_ids)
    bos = tokenizer.get_bos_token_id()
    row_capacity = seq_len + 1
    doc_buffer: list[list[int]] = []
    buffer_min = 512
    epoch = 0

    dev = torch.device(device)
    pin_memory = dev.type == "cuda"
    row_buffer = torch.empty((batch_size, row_capacity), dtype=torch.long)
    cpu_inputs = torch.empty((batch_size, seq_len), dtype=torch.int32, pin_memory=pin_memory)
    cpu_targets = torch.empty((batch_size, seq_len), dtype=torch.long, pin_memory=pin_memory)
    inputs = torch.empty((batch_size, seq_len), dtype=torch.int32, device=dev)
    targets = torch.empty((batch_size, seq_len), dtype=torch.long, device=dev)

    def refill() -> None:
        nonlocal epoch
        docs, epoch = next(batches)
        token_lists = tokenizer.encode(docs, prepend=bos, num_threads=tokenizer_threads)
        doc_buffer.extend(token_lists)

    while True:
        for row_idx in range(batch_size):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_min:
                    refill()
                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    n = len(doc)
                    if n <= remaining and n > best_len:
                        best_idx, best_len = i, n
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    n = len(doc)
                    row_buffer[row_idx, pos:pos + n] = torch.tensor(doc, dtype=torch.long)
                    pos += n
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1].to(dtype=torch.int32))
        cpu_targets.copy_(row_buffer[:, 1:])
        inputs.copy_(cpu_inputs, non_blocking=pin_memory)
        targets.copy_(cpu_targets, non_blocking=pin_memory)
        yield inputs, targets, {"epoch": epoch}


def make_eval_dataloader(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    device: torch.device | str,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    shard_ids: Iterable[int] | None = None,
):
    """Finite validation dataloader that covers each selected document once."""
    assert batch_size > 0 and seq_len > 1
    ddp, rank, _local_rank, world_size = get_dist_info()
    docs_iter = _document_batches_once(split, rank, world_size, tokenizer_batch_size, shard_ids=shard_ids)
    bos = tokenizer.get_bos_token_id()
    row_capacity = seq_len + 1
    dev = torch.device(device)

    stream: list[int] = []
    batch_x: list[list[int]] = []
    batch_y: list[list[int]] = []
    doc_count = 0

    def append_row(row: list[int], valid_targets: int) -> None:
        x = row[:-1]
        y = row[1:]
        if valid_targets < seq_len:
            y = y[:valid_targets] + [-1] * (seq_len - valid_targets)
        batch_x.append(x)
        batch_y.append(y)

    def pop_full_rows() -> None:
        nonlocal stream
        while len(stream) >= row_capacity:
            append_row(stream[:row_capacity], seq_len)
            stream = stream[seq_len:]

    def emit_batch(pad_final: bool = False):
        if pad_final:
            while batch_x and len(batch_x) < batch_size:
                batch_x.append([bos] * seq_len)
                batch_y.append([-1] * seq_len)
        if len(batch_x) < batch_size:
            return None
        x = torch.tensor(batch_x[:batch_size], dtype=torch.int32, device=dev)
        y = torch.tensor(batch_y[:batch_size], dtype=torch.long, device=dev)
        del batch_x[:batch_size]
        del batch_y[:batch_size]
        return x, y, {"documents": doc_count}

    for docs in docs_iter:
        token_lists = tokenizer.encode(docs, prepend=bos, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_count += 1
            stream.extend(tokens)
            pop_full_rows()
            while len(batch_x) >= batch_size:
                yield emit_batch()

    if len(stream) > 1:
        valid_targets = len(stream) - 1
        row = stream + [bos] * (row_capacity - len(stream))
        append_row(row, valid_targets)
    while batch_x:
        yield emit_batch(pad_final=True)


# -----------------------------------------------------------------------------
# Trusted evaluation

class EvalTimeoutError(TimeoutError):
    pass


@torch.no_grad()
def evaluate_bpb(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    device: torch.device | str,
    split: str = "public_val",
    max_tokens: int | None = None,
    timeout_seconds: int = EVAL_TIMEOUT_SECONDS,
    shard_ids: Iterable[int] | None = None,
) -> dict:
    """Evaluate model with trusted CE from logits. Does not call model(x, y)."""
    started = time.monotonic()
    dev = torch.device(device)
    token_bytes = get_token_bytes(dev)
    loader = make_eval_dataloader(tokenizer, batch_size, seq_len, split, dev, shard_ids=shard_ids)
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    local_token_limit = None if max_tokens is None else max(1, math.ceil(max_tokens / max(world_size, 1)))

    model.eval()
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=dev)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=dev)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=dev)
    local_tokens = 0

    for x, y, _state in loader:
        if time.monotonic() - started > timeout_seconds:
            raise EvalTimeoutError(f"evaluation exceeded {timeout_seconds}s")
        logits = model(x)
        if logits.size(-1) < TOKENIZER_VOCAB_SIZE:
            raise RuntimeError(f"model vocab too small: {logits.size(-1)} < {TOKENIZER_VOCAB_SIZE}")
        logits = logits[..., :TOKENIZER_VOCAB_SIZE].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1, reduction="none")
        y_flat = y.reshape(-1)
        valid = y_flat >= 0
        safe_y = y_flat.clamp_min(0)
        nbytes = token_bytes[safe_y]
        mask = valid & (nbytes > 0)
        total_nats += (loss * mask).sum()
        total_bytes += (nbytes * valid).sum()
        batch_tokens = valid.sum()
        total_tokens += batch_tokens
        local_tokens += int(batch_tokens.item())
        if local_token_limit is not None and local_tokens >= local_token_limit:
            break

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    elapsed = time.monotonic() - started
    nats = total_nats.item()
    nbytes = int(total_bytes.item())
    val_bpb = nats / (math.log(2) * nbytes) if nbytes > 0 else float("inf")
    return {
        "val_bpb": float(val_bpb),
        "eval_seconds": float(elapsed),
        "eval_timeout": int(timeout_seconds),
        "eval_over_timeout": bool(elapsed > timeout_seconds),
        "num_tokens": int(total_tokens.item()),
        "num_bytes": nbytes,
        "total_nats": float(nats),
    }


@torch.no_grad()
def causality_check(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    device: torch.device | str,
    tolerance: float = 1e-2,
    split: str = "public_val",
    shard_ids: Iterable[int] | None = None,
) -> dict:
    """Smoke test that future tokens do not affect earlier logits."""
    dev = torch.device(device)
    loader = make_eval_dataloader(tokenizer, batch_size, seq_len, split, dev, shard_ids=shard_ids)
    x, _y, _ = next(loader)
    cut = max(2, seq_len // 2)
    x2 = x.clone()
    x2[:, cut:] = torch.flip(x2[:, cut:], dims=[1])
    model.eval()
    a = model(x)[:, :cut - 1].float()
    b = model(x2)[:, :cut - 1].float()
    max_abs = (a - b).abs().max().item()
    return {"passed": bool(max_abs <= tolerance), "max_abs_diff": float(max_abs), "tolerance": tolerance}


# -----------------------------------------------------------------------------
# Basic static checks

FORBIDDEN_IMPORT_ROOTS = {"socket", "subprocess", "requests", "urllib", "http", "ftplib"}
FORBIDDEN_CALLS = {
    ("os", "system"),
    ("os", "popen"),
    ("shutil", "rmtree"),
    ("subprocess", "run"),
    ("subprocess", "Popen"),
    ("subprocess", "check_call"),
    ("subprocess", "check_output"),
}
# Direct builtin calls parse as ast.Name (not ast.Attribute), so they bypass
# FORBIDDEN_CALLS. These dynamic-execution builtins also defeat the import
# blocklist (e.g. __import__("socket")) and have no place in an honest train.py.
FORBIDDEN_NAMES = {"eval", "exec", "__import__", "compile"}


def _launch_command_warnings(launch_command: str) -> list[str]:
    warnings: list[str] = []
    bad_tokens = ["|", ">", "<", "&&", "||", ";", "&", "`", "$("]
    for tok in bad_tokens:
        if tok in launch_command:
            warnings.append(f"launch command contains disallowed shell token: {tok}")
    try:
        tokens = shlex.split(launch_command)
    except ValueError as exc:
        return warnings + [f"launch command parse error: {exc}"]
    if not tokens:
        return warnings + ["launch command is empty"]

    env = {}
    idx = 0
    while idx < len(tokens) and "=" in tokens[idx] and not tokens[idx].startswith("-"):
        key, value = tokens[idx].split("=", 1)
        env[key] = value
        idx += 1
    if idx >= len(tokens):
        return warnings + ["launch command missing executable"]

    lnc = env.get("NEURON_LOGICAL_NC_CONFIG")
    if lnc is not None and lnc not in {"1", "2"}:
        warnings.append("NEURON_LOGICAL_NC_CONFIG must be 1 or 2")

    exe = Path(tokens[idx]).name
    args = tokens[idx + 1:]
    if exe not in {"torchrun", "python", "python3"}:
        warnings.append("launch executable must be torchrun, python, or python3")
        return warnings

    if exe == "torchrun":
        nproc = None
        for i, arg in enumerate(args):
            if arg.startswith("--nproc_per_node="):
                nproc = arg.split("=", 1)[1]
            elif arg == "--nproc_per_node" and i + 1 < len(args):
                nproc = args[i + 1]
        if nproc not in {"1", "4", "8"}:
            warnings.append("torchrun --nproc_per_node must be 1, 4, or 8")
        if not any(Path(arg).name == "train.py" for arg in args):
            warnings.append("torchrun launch command must target train.py")
    else:
        script_args = [arg for arg in args if not arg.startswith("-")]
        if not script_args or Path(script_args[0]).name != "train.py":
            warnings.append("python launch command must target train.py")
    return warnings


def check_submission(train_py_path: str, launch_command: str | None = None) -> list[str]:
    warnings: list[str] = []
    path = Path(train_py_path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    warnings.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                warnings.append(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            if isinstance(value, ast.Name) and (value.id, node.func.attr) in FORBIDDEN_CALLS:
                warnings.append(f"forbidden call: {value.id}.{node.func.attr}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_NAMES:
                warnings.append(f"forbidden call: {node.func.id}()")

    if launch_command:
        warnings.extend(_launch_command_warnings(launch_command))
    return warnings


# -----------------------------------------------------------------------------
# CLI

def _import_train_module(train_py: str):
    path = Path(train_py).resolve()
    spec = importlib.util.spec_from_file_location("neuron_competition_r1_train_for_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _detect_eval_device(requested: str) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        torch.device("neuron")
        return torch.device("neuron")
    except Exception:
        return torch.device("cpu")


def _init_eval_distributed(device: torch.device) -> None:
    ddp, _rank, local_rank, _world_size = get_dist_info()
    if not ddp or (dist.is_available() and dist.is_initialized()):
        return
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    elif device.type == "neuron":
        dist.init_process_group("neuron")
    else:
        dist.init_process_group("gloo")


def _cleanup_eval_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def main() -> None:
    parser = argparse.ArgumentParser(description="Neuron Competition R1 prep utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    setup = sub.add_parser("setup")
    setup.add_argument("--num-train-shards", type=int, default=len(PUBLIC_TRAIN_SHARDS))
    setup.add_argument("--workers", type=int, default=8)
    setup.add_argument("--build-tokenizer-if-missing", action="store_true")

    check = sub.add_parser("check")
    check.add_argument("--train-py", type=str, required=True)
    check.add_argument("--launch-command", type=str, default=None)

    eval_public = sub.add_parser("eval-public")
    eval_public.add_argument("--train-py", type=str, required=True)
    eval_public.add_argument("--checkpoint", type=str, required=True)
    eval_public.add_argument("--device", type=str, default="")
    eval_public.add_argument("--batch-size", type=int, default=1)
    eval_public.add_argument("--seq-len", type=int, default=1024)
    eval_public.add_argument("--eval-tokens", type=int, default=4 * 524288, help="0 means evaluate the full public val shard")
    eval_public.add_argument("--timeout-seconds", type=int, default=EVAL_TIMEOUT_SECONDS)

    args = parser.parse_args()
    if args.cmd == "setup":
        ensure_data(args.num_train_shards, args.workers)
        ensure_tokenizer(build_if_missing=args.build_tokenizer_if_missing)
        print("setup complete")
    elif args.cmd == "check":
        warnings = check_submission(args.train_py, args.launch_command)
        if warnings:
            for w in warnings:
                print(f"WARNING: {w}")
            raise SystemExit(1)
        print("check passed")
    elif args.cmd == "eval-public":
        train_mod = _import_train_module(args.train_py)
        if not hasattr(train_mod, "load_for_eval"):
            raise RuntimeError("train.py must define load_for_eval(checkpoint_path, device)")
        device = _detect_eval_device(args.device)
        _init_eval_distributed(device)
        try:
            tokenizer = ensure_tokenizer(build_if_missing=False)
            model = train_mod.load_for_eval(args.checkpoint, device)
            result = evaluate_bpb(
                model,
                tokenizer,
                args.batch_size,
                args.seq_len,
                device,
                split="public_val",
                max_tokens=None if args.eval_tokens <= 0 else args.eval_tokens,
                timeout_seconds=args.timeout_seconds,
            )
            if int(os.environ.get("RANK", "0")) == 0:
                print(json.dumps(result, indent=2, sort_keys=True))
        finally:
            _cleanup_eval_distributed()


if __name__ == "__main__":
    main()
