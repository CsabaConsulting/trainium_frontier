#!/usr/bin/env python3
"""Fail-closed contract check for the frozen review-clean submission artifact.

This uses only synthetic tokens. It extends the historical submission contract check with an odd
batch, a second load, and complete metric reporting before returning failure. The caller must name
the reference train.py explicitly: use the repository root train.py for historical full-feature
checkpoints and the review-clean train.py for fresh lean checkpoints.
"""

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-py", type=Path, required=True)
    parser.add_argument(
        "--reference-train-py",
        type=Path,
        required=True,
        help=(
            "Explicit load_for_eval implementation used for exact-logit comparison. Use root "
            "train.py for historical full-feature checkpoints or repeat --train-py for fresh "
            "lean checkpoints."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "neuron"), default="cpu")
    args = parser.parse_args()
    train_py = args.train_py.resolve()
    reference_train_py = args.reference_train_py.resolve()
    checkpoint = args.checkpoint.resolve()
    if not train_py.is_file():
        parser.error(f"--train-py is not a file: {train_py}")
    if not reference_train_py.is_file():
        parser.error(f"--reference-train-py is not a file: {reference_train_py}")
    if not checkpoint.is_file():
        parser.error(f"--checkpoint is not a file: {checkpoint}")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.device == "neuron":
        prepare._init_eval_distributed(device)
    try:
        reference_module = prepare._import_train_module(str(reference_train_py))
        reference_module_independent = reference_train_py != train_py
        candidate = (
            prepare._import_train_module(str(train_py))
            if reference_module_independent
            else reference_module
        )
        if reference_module_independent:
            assert reference_module is not candidate
        assert callable(getattr(reference_module, "load_for_eval", None))
        assert callable(getattr(candidate, "load_for_eval", None))
        reference = reference_module.load_for_eval(str(checkpoint), device)
        model = candidate.load_for_eval(str(checkpoint), device)
        assert reference is not model
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        assert all(
            not module._forward_hooks
            and not module._forward_pre_hooks
            and not module._backward_hooks
            for module in model.modules()
        )
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        buffers = {key: value.detach().cpu().clone() for key, value in model.named_buffers()}

        torch.manual_seed(102398)
        tokens = torch.randint(8192, (3, 128), dtype=torch.int32, device="cpu").to(device)
        changed_future = tokens.clone()
        changed_future[:, 65:] = (changed_future[:, 65:] + 37) % 8192
        altered_other_rows = tokens.clone()
        altered_other_rows[1:] = (altered_other_rows[1:] + 139) % 8192

        def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
            left = left.detach().float().cpu()
            right = right.detach().float().cpu()
            return float((left - right).abs().max())

        with torch.inference_mode():
            actual = model(tokens)
            rows = torch.cat([model(tokens[index : index + 1]) for index in range(3)], dim=0)
            reference_rows = torch.cat(
                [reference(tokens[index : index + 1]) for index in range(3)], dim=0
            )
            second_model = candidate.load_for_eval(str(checkpoint), device)
            assert second_model is not model
            second = second_model(tokens[:1])
            first_after_second_load = model(tokens[:1])
            metrics = {
                "source_checkpoint_batch1_logits_max_abs": max_abs(
                    model(tokens[:1]), reference(tokens[:1])
                ),
                "source_checkpoint_rowwise_logits_max_abs": max_abs(rows, reference_rows),
                "repeat_max_abs": max_abs(actual, model(tokens)),
                "causality_max_abs": max_abs(actual[:, :65], model(changed_future)[:, :65]),
                "batch_independence_even_max_abs": max_abs(
                    model(tokens[:2])[:1], model(tokens[:1])
                ),
                "batch_independence_odd_max_abs": max_abs(actual, rows),
                "batch_permutation_max_abs": max_abs(actual.flip(0), model(tokens.flip(0))),
                "other_row_content_max_abs": max_abs(
                    actual[:1], model(altered_other_rows)[:1]
                ),
                "second_load_logits_max_abs": max_abs(second, model(tokens[:1])),
                "first_model_after_second_load_max_abs": max_abs(
                    first_after_second_load, model(tokens[:1])
                ),
            }

        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.detach().cpu(), state[key], rtol=0, atol=0)
        for key, value in model.named_buffers():
            torch.testing.assert_close(value.detach().cpu(), buffers[key], rtol=0, atol=0)
        no_training_adapters = not hasattr(candidate, "_fresh_start_adaptive_ce_installed") and not any(
            "_fresh_start_owned" in vars(module) for module in model.modules()
        )
        passed = all(value == 0.0 for value in metrics.values()) and no_training_adapters
        report = {
            **metrics,
            "passed": passed,
            "device": args.device,
            "frozen": True,
            "no_hooks": True,
            "state_and_buffers_unchanged": True,
            "no_training_adapters_at_import": no_training_adapters,
            "num_params": sum(parameter.numel() for parameter in model.parameters()),
            "train_py": str(train_py),
            "reference_train_py": str(reference_train_py),
            "reference_module_independent": reference_module_independent,
            "reference_model_independent": True,
            "second_model_independent": True,
        }
        print("REVIEW_CLEAN_SUBMISSION_CONTRACT " + json.dumps(report), flush=True)
        if not passed:
            raise SystemExit(1)
    finally:
        if args.device == "neuron":
            prepare._cleanup_eval_distributed()


if __name__ == "__main__":
    main()
