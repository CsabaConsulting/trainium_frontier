#!/usr/bin/env python3
"""Paired diagnostic for a fixed, tokenizer-derived reserved-output mask.

The trusted scorer excludes targets whose fixed token-byte count is zero.
Remove probability mass from those four output classes using a constant rule,
then call the unchanged scorer twice on one frozen checkpoint. No fitting,
validation-dependent selection, stored validation logits, or training claim.
A training candidate would also exclude zero-byte targets from its own loss.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import prepare
import train as T


class ReservedOutputMask(nn.Module):
    def __init__(self, model, zero_byte):
        super().__init__()
        self.model = model
        self.register_buffer("zero_byte", zero_byte, persistent=False)

    def forward(self, tokens):
        logits = self.model(tokens)
        # Mask after the existing softcap. A finite sentinel is essential:
        # prepare.py computes all token losses before multiplying by its mask.
        return logits.masked_fill(self.zero_byte, -10000.0)


@torch.no_grad()
def gate(model, vocab_size, device, zero_byte):
    gen = torch.Generator().manual_seed(314159)
    x = torch.randint(vocab_size, (2, 128), generator=gen).to(device)
    raw = model.model(x).float().cpu()
    expected = raw.masked_fill(zero_byte, -10000.0)
    actual = model(x).float().cpu()
    future = x.clone()
    future[:, 64:] = (future[:, 64:] + 17) % vocab_size
    changed_future = model(future).float().cpu()
    other = x.clone()
    other[1] = (other[1] + 31) % vocab_size
    changed_other = model(other).float().cpu()
    repeated = model(x).float().cpu()
    single = model(x[:1]).float().cpu()
    positive_ids = (~zero_byte).nonzero().flatten()
    targets = positive_ids[torch.randint(len(positive_ids), x.shape, generator=gen)]
    base_ce = F.cross_entropy(raw.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")
    mask_ce = F.cross_entropy(actual.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")
    reserved = zero_byte.nonzero().flatten()[0].expand(x.numel())
    excluded_loss = F.cross_entropy(actual.reshape(-1, vocab_size), reserved, reduction="none")
    result = dict(base_logits_finite=bool(torch.isfinite(raw).all()),
                  logits_finite=bool(torch.isfinite(actual).all()),
                  mask_reference_max_abs=float((expected - actual).abs().max()),
                  future_max_abs=float((actual[:, :64] - changed_future[:, :64]).abs().max()),
                  other_batch_max_abs=float((actual[0] - changed_other[0]).abs().max()),
                  repeated_call_max_abs=float((actual - repeated).abs().max()),
                  batch_size_max_abs=float((actual[:1] - single).abs().max()),
                  positive_target_nll_max_increase=float((mask_ce - base_ce).max()),
                  excluded_targets_finite_before_mask=bool(torch.isfinite(excluded_loss).all()),
                  excluded_targets_zero_after_mask=bool((excluded_loss * 0 == 0).all()),
                  synthetic_inputs_only=True)
    result["passed"] = (result["base_logits_finite"] and result["logits_finite"]
                        and all(result[k] == 0 for k in ("mask_reference_max_abs", "future_max_abs",
                                                        "other_batch_max_abs", "repeated_call_max_abs"))
                        and result["batch_size_max_abs"] <= 1e-2
                        and result["positive_target_nll_max_increase"] <= 2e-5
                        and result["excluded_targets_finite_before_mask"]
                        and result["excluded_targets_zero_after_mask"])
    print("ZERO_BYTE_GATE_JSON " + json.dumps(result), flush=True)
    if not result["passed"]:
        raise AssertionError("zero-byte mask contract gate failed")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "neuron"), default="neuron")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--eval-tokens", type=int, default=2_097_152)
    ap.add_argument("--timeout-seconds", type=int, default=1200)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4 if args.device == "cpu" else 1)
    _, _, _, world_size, device = T.init_runtime(args.device, seed=42)
    if world_size != 1:
        ap.error("This paired diagnostic requires one evaluation rank")
    try:
        tokenizer = prepare.ensure_tokenizer(build_if_missing=False)
        zero_byte = prepare.get_token_bytes("cpu") == 0
        assert len(zero_byte) == tokenizer.get_vocab_size() and zero_byte.any() and (~zero_byte).any()
        model = T.load_for_eval(str(args.checkpoint), device)
        wrapped = ReservedOutputMask(model, zero_byte.to(device)).eval()
        config = dict(checkpoint=str(args.checkpoint),
                      checkpoint_sha256=prepare.sha256_file(args.checkpoint),
                      tokenizer_sha256=prepare.TOKENIZER_PKL_SHA256,
                      zero_byte_ids=zero_byte.nonzero().flatten().tolist(),
                      mask_sha256=hashlib.sha256(zero_byte.numpy().tobytes()).hexdigest(),
                      train_source_sha256=prepare.sha256_file(Path(T.__file__)),
                      seq_len=args.seq_len, batch_size=1, world_size=world_size,
                      eval_tokens=args.eval_tokens, inference_diagnostic_only=True,
                      construction_uses_validation=False)
        print("ZERO_BYTE_EVAL_CONFIG " + json.dumps(config), flush=True)
        checks = gate(wrapped, tokenizer.get_vocab_size(), device, zero_byte)
        if args.check_only:
            return
        kwargs = dict(batch_size=1, seq_len=args.seq_len, device=device,
                      split="public_val", max_tokens=args.eval_tokens,
                      timeout_seconds=args.timeout_seconds)
        started = time.perf_counter()
        baseline = prepare.evaluate_bpb(model, tokenizer, **kwargs)
        print("ZERO_BYTE_BASELINE_JSON " + json.dumps(baseline), flush=True)
        constrained = prepare.evaluate_bpb(wrapped, tokenizer, **kwargs)
        assert (baseline["num_tokens"], baseline["num_bytes"]) == (constrained["num_tokens"], constrained["num_bytes"])
        assert all(math.isfinite(r["val_bpb"]) for r in (baseline, constrained))
        print("ZERO_BYTE_EVAL_RESULT " + json.dumps(dict(
            config=config, checks=checks, baseline=baseline, constrained=constrained,
            bpb_delta=constrained["val_bpb"] - baseline["val_bpb"],
            paired_eval_seconds=time.perf_counter() - started)), flush=True)
    finally:
        T.cleanup_runtime()


if __name__ == "__main__":
    main()
