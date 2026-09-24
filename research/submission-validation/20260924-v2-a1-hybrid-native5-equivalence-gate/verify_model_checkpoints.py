#!/usr/bin/env python3
"""Strict model-only checkpoint and eval-logit identity check for the native gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"not one regular unlinked file: {path}")


def _load_module(path: Path, name: str):
    _regular(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load gate source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _raw(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    if not bool(torch.isfinite(cpu).all().item()):
        raise ValueError("nonfinite model/checkpoint tensor")
    return cpu.view(torch.uint8).numpy().tobytes()


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(_raw(tensor))
    return digest.hexdigest()


def _checkpoint(path: Path) -> tuple[dict, str, str]:
    _regular(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if set(payload) != {"model_state", "model_config", "toggles", "meta"}:
        raise ValueError("checkpoint is not the exact model-only schema")
    if not isinstance(payload["model_state"], dict) or not payload["model_state"]:
        raise ValueError("empty checkpoint model state")
    return payload, hashlib.sha256(path.read_bytes()).hexdigest(), _state_digest(payload["model_state"])


def _evaluate(source: Path, checkpoint: Path, module_name: str) -> tuple[str, str]:
    module = _load_module(source, module_name)
    model = module.load_for_eval(str(checkpoint), torch.device("cpu"))
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(42)
    first = torch.randint(0, module.TOKENIZER_VOCAB_SIZE, (2, 1024),
                          generator=generator, dtype=torch.int32)
    second = first.clone()
    second[:, 513:] = torch.randint(0, module.TOKENIZER_VOCAB_SIZE,
                                    second[:, 513:].shape, generator=generator,
                                    dtype=torch.int32)
    with torch.no_grad():
        logits_a = model(first)
        logits_b = model(second)
    if not torch.equal(logits_a[:, :513], logits_b[:, :513]):
        raise ValueError("causality/prefix identity failed")
    live_digest = _state_digest(model.state_dict())
    logits_digest = hashlib.sha256(_raw(logits_a)).hexdigest()
    del model, logits_a, logits_b
    return live_digest, logits_digest


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                allow_nan=False).encode() + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-source", type=Path, required=True)
    parser.add_argument("--ordinary-source", type=Path, required=True)
    parser.add_argument("--g0-checkpoint", type=Path, required=True)
    parser.add_argument("--g1-checkpoint", type=Path, required=True)
    parser.add_argument("--g2-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = []
        sources = [args.hybrid_source, args.hybrid_source, args.ordinary_source]
        checkpoints = [args.g0_checkpoint, args.g1_checkpoint, args.g2_checkpoint]
        for index, (source, checkpoint) in enumerate(zip(sources, checkpoints)):
            payload, checkpoint_sha, state_sha = _checkpoint(checkpoint)
            live_sha, logits_sha = _evaluate(source, checkpoint, f"a1_gate_eval_{index}")
            if live_sha != state_sha:
                raise ValueError(f"G{index} strict-loaded state differs from checkpoint")
            records.append({"arm": f"G{index}", "checkpoint_sha256": checkpoint_sha,
                            "model_state_sha256": state_sha,
                            "eval_logits_sha256": logits_sha,
                            "meta": payload["meta"]})
        if len({record["model_state_sha256"] for record in records}) != 1:
            raise ValueError("model-only checkpoints differ")
        if len({record["eval_logits_sha256"] for record in records}) != 1:
            raise ValueError("eval-mode logits differ")
        result = {"schema": "a1-native-five-model-check-v1", "status": "PASS",
                  "causality_exact": True, "records": records}
        rc = 0
    except Exception as exc:
        result = {"schema": "a1-native-five-model-check-v1", "status": "INVALID",
                  "error": f"{type(exc).__name__}: {exc}"}
        rc = 2
    _write(args.output, result)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
