#!/usr/bin/env python3
"""Paired inference diagnostic using the unchanged trusted public BPB scorer.

The checkpoint is frozen. A conservative mask is derived afresh from the fixed
tokenizer, never from validation observations. This probe does not create a
submission: any promising constraint must subsequently be trained and evaluated
consistently within a budgeted run. No validation tokens or logits are saved.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import prepare
import train as T
from canonical_token_mask_probe import build_mask


class CanonicalContinuation(nn.Module):
    def __init__(self, model, forbidden):
        super().__init__()
        self.model = model
        self.register_buffer("forbidden", forbidden, persistent=False)

    def forward(self, tokens):
        logits = self.model(tokens)
        # The model's existing softcap is +/-15. Finite -10000 leaves negligible
        # forbidden mass and preserves the evaluator's finite-logit contract.
        return logits.masked_fill(self.forbidden[tokens], -10000.0)


@torch.no_grad()
def gate(model, vocab_size, device, forbidden_cpu):
    # Entirely synthetic inputs; no validation observations are used for gates.
    gen = torch.Generator().manual_seed(314159)
    x = torch.randint(vocab_size, (2, 128), generator=gen).to(device)
    raw = model.model(x).float().cpu()
    expected = raw.masked_fill(torch.from_numpy(forbidden_cpu)[x.cpu()], -10000.0)
    a = model(x).float().cpu()
    future = x.clone()
    future[:, 64:] = (future[:, 64:] + 17) % vocab_size
    b = model(future).float().cpu()
    other = x.clone()
    other[1] = (other[1] + 31) % vocab_size
    c = model(other).float().cpu()
    repeated = model(x).float().cpu()
    single = model(x[:1]).float().cpu()
    result = dict(logits_finite=bool(torch.isfinite(a).all()),
                  base_logits_finite=bool(torch.isfinite(raw).all()),
                  device_mask_reference_max_abs=float((expected - a).abs().max()),
                  future_max_abs=float((a[:, :64] - b[:, :64]).abs().max()),
                  other_batch_max_abs=float((a[0] - c[0]).abs().max()),
                  repeated_call_max_abs=float((a - repeated).abs().max()),
                  batch_size_max_abs=float((a[:1] - single).abs().max()),
                  batch_size_tolerance=1e-2)
    result["passed"] = (result["logits_finite"] and result["base_logits_finite"]
                        and result["device_mask_reference_max_abs"] == 0
                        and result["future_max_abs"] == 0
                        and result["other_batch_max_abs"] == 0
                        and result["repeated_call_max_abs"] == 0
                        and result["batch_size_max_abs"] <= result["batch_size_tolerance"])
    print("CANONICAL_MASK_GATE_JSON " + json.dumps(result), flush=True)
    if not result["passed"]:
        raise AssertionError("canonical-mask evaluation gate failed")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--scope", choices=["ascii-long", "ascii-word"], default="ascii-word")
    ap.add_argument("--device", choices=["cpu", "neuron"], default="neuron")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--eval-tokens", type=int, default=2_097_152)
    ap.add_argument("--timeout-seconds", type=int, default=1200)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8 if args.device == "cpu" else 1)
    _, rank, _, world_size, device = T.init_runtime(args.device, seed=42)
    if world_size != 1:
        ap.error("this paired diagnostic uses fixed single-rank evaluation geometry")
    try:
        tokenizer = prepare.ensure_tokenizer(build_if_missing=False)
        forbidden, construction = build_mask(tokenizer, args.scope)
        model = T.load_for_eval(str(args.checkpoint), device)
        wrapped = CanonicalContinuation(model, torch.from_numpy(forbidden).to(device)).eval()
        config = dict(checkpoint=str(args.checkpoint),
                      checkpoint_sha256=prepare.sha256_file(args.checkpoint),
                      tokenizer_sha256=prepare.TOKENIZER_PKL_SHA256,
                      mask_sha256=hashlib.sha256(forbidden.tobytes()).hexdigest(),
                      source_sha256=prepare.sha256_file(Path(T.__file__)),
                      scope=args.scope, construction=construction,
                      seq_len=args.seq_len, batch_size=1, world_size=world_size,
                      max_tokens=args.eval_tokens, inference_diagnostic_only=True)
        print("CANONICAL_EVAL_CONFIG " + json.dumps(config), flush=True)
        checks = gate(wrapped, tokenizer.get_vocab_size(), device, forbidden)
        if args.check_only:
            return
        started = time.perf_counter()
        kwargs = dict(batch_size=1, seq_len=args.seq_len, device=device,
                      split="public_val", max_tokens=args.eval_tokens,
                      timeout_seconds=args.timeout_seconds)
        with torch.no_grad():
            baseline = prepare.evaluate_bpb(model, tokenizer, **kwargs)
            print("CANONICAL_BASELINE_JSON " + json.dumps(baseline), flush=True)
            constrained = prepare.evaluate_bpb(wrapped, tokenizer, **kwargs)
        if ((baseline["num_tokens"], baseline["num_bytes"])
                != (constrained["num_tokens"], constrained["num_bytes"])):
            raise AssertionError("paired scorer runs did not cover identical geometry")
        if not all(math.isfinite(r["val_bpb"]) for r in (baseline, constrained)):
            raise AssertionError("paired scorer run returned a nonfinite BPB")
        result = dict(config=config, checks=checks, baseline=baseline,
                      constrained=constrained,
                      bpb_delta=constrained["val_bpb"] - baseline["val_bpb"],
                      paired_eval_seconds=time.perf_counter() - started)
        print("CANONICAL_EVAL_RESULT " + json.dumps(result), flush=True)
    finally:
        T.cleanup_runtime()


if __name__ == "__main__":
    main()
