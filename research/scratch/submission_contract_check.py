#!/usr/bin/env python3
"""Compare a standalone submission hook with the source winning checkpoint.

No validation text is used. Synthetic tokens exercise strict checkpoint loading,
frozen parameters, causality, batch independence, repeatability and state purity.
Run under the campaign host guard; this is a development check, not a submission.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import prepare
import train as baseline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-py", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "neuron"), default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.device == "neuron":
        prepare._init_eval_distributed(device)
    try:
        candidate = prepare._import_train_module(str(args.train_py))
        reference = baseline.load_for_eval(str(args.checkpoint), device)
        model = candidate.load_for_eval(str(args.checkpoint), device)
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())
        assert all(not m._forward_hooks and not m._forward_pre_hooks
                   and not m._backward_hooks for m in model.modules())
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        torch.manual_seed(102398)
        tokens = torch.randint(8192, (2, 128), dtype=torch.int32, device="cpu").to(device)
        changed = tokens.clone()
        changed[:, 65:] = (changed[:, 65:] + 37) % 8192
        def close(a, b, exact=False):
            a, b = a.detach().float().cpu(), b.detach().float().cpu()
            torch.testing.assert_close(a, b, rtol=0, atol=0 if exact else 1e-5)
            return float((a - b).abs().max())
        with torch.inference_mode():
            actual = model(tokens)
            report = dict(
                source_checkpoint_logits_max_abs=close(actual, reference(tokens), exact=True),
                repeat_max_abs=close(actual, model(tokens), exact=True),
                causality_max_abs=close(actual[:, :65], model(changed)[:, :65]),
                batch_independence_max_abs=close(actual[:1], model(tokens[:1])),
                batch_permutation_max_abs=close(actual.flip(0), model(tokens.flip(0))),
            )
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v.detach().cpu(), state[k], rtol=0, atol=0)
        for k, v in model.named_buffers():
            torch.testing.assert_close(v.detach().cpu(), buffers[k], rtol=0, atol=0)
        assert not hasattr(candidate, "_fresh_start_adaptive_ce_installed")
        assert not any("_fresh_start_owned" in vars(m) for m in model.modules())
        report.update(passed=True, device=args.device, frozen=True, no_hooks=True,
                      state_and_buffers_unchanged=True, no_training_adapters_at_import=True,
                      num_params=sum(p.numel() for p in model.parameters()))
        print("SUBMISSION_CONTRACT " + json.dumps(report), flush=True)
    finally:
        if args.device == "neuron":
            prepare._cleanup_eval_distributed()


if __name__ == "__main__":
    main()
