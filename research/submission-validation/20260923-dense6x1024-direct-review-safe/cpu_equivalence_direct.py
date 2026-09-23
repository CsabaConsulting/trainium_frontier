#!/usr/bin/env python3
"""CPU equivalence checks for the direct reviewer-safe integration.

This is implementation evidence, not an independent review and not a Trainium gate.  It checks
the executable CPU portions of the three direct integrations against the hash-pinned source and
records the hardware-only NKI limitation explicitly.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types


sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "submissions/dense6x1024_lean_review/train.py"
DIRECT = ROOT / "submissions/dense6x1024_direct_review/train.py"
LAUNCH = ROOT / "submissions/dense6x1024_direct_review/launch-command.txt"
REPORT = ROOT / "submissions/dense6x1024_direct_review/cpu-equivalence.json"
SOURCE_SHA256 = "9173f62617789b058083c3a4722a906e2b165f4986716281f1915db3fb9fa344"
EXPECTED_LAUNCH = (
    "NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py "
    "--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct "
    "--nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce "
    "--seq-len 1024 --no-eval-public"
)
TRIGGERS = (
    "_submission_",
    "_install_submission_training",
    "MethodType",
    "types.MethodType",
    "SUBMISSION_",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_train(name: str, path: Path):
    # prepare.py imports tiktoken, but none of these checks constructs a tokenizer.  Keeping this
    # tiny import-only stub makes the CPU evidence independent of that optional local package.
    if "tiktoken" not in sys.modules:
        stub = types.ModuleType("tiktoken")
        stub.Encoding = object
        sys.modules["tiktoken"] = stub
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensors_equal(left, right) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and bool(left.equal(right))


def state_equal(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(tensors_equal(left[key], right[key]) for key in left)


def nested_equal(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    if hasattr(left, "equal") and hasattr(right, "equal"):
        return tensors_equal(left, right)
    return left == right


def optimizer_worker(source_path: Path, direct_path: Path) -> None:
    import torch
    import torch.distributed as dist

    source = load_train("owned_source_worker", source_path)
    direct = load_train("owned_direct_worker", direct_path)
    dist.init_process_group("gloo")
    try:
        torch.manual_seed(4201)
        source_parameters = [
            torch.nn.Parameter(torch.randn(17)),
            torch.nn.Parameter(torch.randn(11)),
            torch.nn.Parameter(torch.randn(8, 8)),
            torch.nn.Parameter(torch.randn(8, 8)),
        ]
        direct_parameters = [torch.nn.Parameter(parameter.detach().clone()) for parameter in source_parameters]

        def make_optimizer(module, parameters):
            return module.MuonAdamW(
                [
                    dict(
                        kind="adamw",
                        params=parameters[:2],
                        lr=0.003,
                        betas=(0.8, 0.95),
                        eps=1e-10,
                        weight_decay=0.01,
                    ),
                    dict(
                        kind="muon",
                        params=parameters[2:],
                        lr=0.02,
                        momentum=0.9,
                        beta2=0.9,
                        ns_steps=3,
                        weight_decay=0.2,
                        transposed=False,
                    ),
                ]
            )

        source_optimizer = make_optimizer(source, source_parameters)
        direct_optimizer = make_optimizer(direct, direct_parameters)
        source_owner = source._submission_optimizer_install(source_optimizer, source)
        direct_owner = direct.OwnedOptimizer(direct_optimizer)
        assert source_owner.entries == direct_owner.entries
        assert source_owner.summary == direct_owner.summary
        for step in range(3):
            for index, (source_parameter, direct_parameter) in enumerate(
                zip(source_parameters, direct_parameters)
            ):
                gradient = torch.arange(source_parameter.numel(), dtype=torch.float32).reshape_as(
                    source_parameter
                )
                gradient = (gradient + 1 + 7 * step + index) / (29 + index)
                source_parameter.grad = gradient.clone()
                direct_parameter.grad = gradient.clone()
            source_optimizer.step(update_adamw=True)
            direct_owner.step(update_adamw=True)
        local_ok = all(
            tensors_equal(source_parameter, direct_parameter)
            for source_parameter, direct_parameter in zip(source_parameters, direct_parameters)
        )
        local_ok = local_ok and source_owner.calls == direct_owner.calls == 3
        local_ok = local_ok and nested_equal(source_optimizer.state_dict(), direct_owner.state_dict())
        status = torch.tensor(int(local_ok), dtype=torch.int32)
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        if not bool(status.item()):
            raise AssertionError("source adapter and explicit OwnedOptimizer updates differ")
        if dist.get_rank() == 0:
            print("OWNED_OPTIMIZER_EQUIVALENT", flush=True)
    finally:
        dist.destroy_process_group()


def normalized(node: ast.FunctionDef, name: str = "normalized") -> str:
    copied = deepcopy(node)
    copied.name = name
    return ast.dump(copied, include_attributes=False)


def named_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {name}, found {len(matches)}")
    return matches[0]


def adaptive_ce_checks(source_text: str, direct_text: str) -> dict[str, bool]:
    source_tree = ast.parse(source_text)
    direct_tree = ast.parse(direct_text)
    build = named_function(source_tree, "build_kernels")
    source_forward = named_function(build, "forward")
    source_backward = named_function(build, "backward")
    direct_forward = named_function(direct_tree, "_softcap_ce_fwd")
    direct_backward = named_function(direct_tree, "_softcap_ce_bwd")
    ce_class = next(
        node
        for node in ast.walk(direct_tree)
        if isinstance(node, ast.ClassDef) and node.name == "_SoftcapCrossEntropyNKI"
    )
    class_text = ast.unparse(ce_class)
    return {
        "forward_kernel_ast_equal": normalized(source_forward) == normalized(direct_forward),
        "backward_kernel_ast_equal": normalized(source_backward) == normalized(direct_backward),
        "class_selects_lanes": "selected_lanes" in class_text,
        "class_calls_wrapped_forward": "_softcap_ce_fwd_native[lanes]" in class_text,
        "class_calls_wrapped_backward": "_softcap_ce_bwd_native[lanes]" in class_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owned-worker", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--direct", type=Path, default=DIRECT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    if args.owned_worker:
        optimizer_worker(args.source.resolve(), args.direct.resolve())
        return

    import torch
    import torch.nn as nn

    source_path = args.source.resolve()
    direct_path = args.direct.resolve()
    if sha256(source_path) != SOURCE_SHA256:
        raise AssertionError("hash-pinned source changed")
    source_text = source_path.read_text()
    direct_text = direct_path.read_text()
    source = load_train("equivalence_source", source_path)
    direct = load_train("equivalence_direct", direct_path)

    config_values = dict(
        sequence_len=128,
        vocab_size=64,
        n_layer=1,
        n_head=1,
        n_embd=128,
        mlp_ratio=4,
        attn_scale=0.0,
        n_kv_heads=0,
    )
    torch.manual_seed(42)
    source_model = source.GPT(source.GPTConfig(**config_values))
    source_model.init_weights()
    source_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    direct_model = direct.GPT(direct.GPTConfig(**config_values))
    direct_model.init_weights()
    direct_rng = torch.get_rng_state().clone()
    initialization = {
        "state_dict_exact": state_equal(source_model.state_dict(), direct_model.state_dict()),
        "rng_exact": tensors_equal(source_rng, direct_rng),
    }

    with tempfile.TemporaryDirectory(prefix="direct-review-cpu-") as temporary:
        checkpoint = Path(temporary) / "model.pt"
        source.save_checkpoint(checkpoint, source_model, {"evidence": "cpu-only"})
        source_eval = source.load_for_eval(str(checkpoint), torch.device("cpu"))
        direct_eval = direct.load_for_eval(str(checkpoint), torch.device("cpu"))
        tokens = torch.tensor(
            [[1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30]],
            dtype=torch.long,
        )
        with torch.no_grad():
            source_logits = source_eval(tokens)
            direct_logits = direct_eval(tokens)
            changed_future = tokens.clone()
            changed_future[:, 5:] = torch.tensor([[31, 33, 35], [37, 39, 41]])
            direct_changed = direct_eval(changed_future)
        evaluation = {
            "source_checkpoint_loads_in_direct": True,
            "logits_exact": tensors_equal(source_logits, direct_logits),
            "max_abs_logit_difference": float((source_logits - direct_logits).abs().max()),
            "causal_prefix_exact": tensors_equal(direct_logits[:, :5], direct_changed[:, :5]),
        }

    class BlockHolder(nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp

    class TransformerHolder(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.h = nn.ModuleList([block])

    class ModelHolder(nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.transformer = TransformerHolder(BlockHolder(mlp))

    source_mlp = source.MLP(source.GPTConfig(**config_values))
    direct_mlp = direct.MLP(direct.GPTConfig(**config_values))
    direct_mlp.load_state_dict(source_mlp.state_dict())
    source_holder = ModelHolder(source_mlp)
    source._submission_mlp_install(source_holder, "cpu", source)
    source_mlp.train()
    direct_mlp.train()
    torch.manual_seed(4202)
    source_input = torch.randn(2, 128, 128, dtype=torch.bfloat16, requires_grad=True)
    direct_input = source_input.detach().clone().requires_grad_(True)
    source_output = source_mlp(source_input)
    direct_output = direct_mlp(direct_input)
    source_output.float().square().mean().backward()
    direct_output.float().square().mean().backward()
    fused_mlp = {
        "output_exact": tensors_equal(source_output, direct_output),
        "input_gradient_exact": tensors_equal(source_input.grad, direct_input.grad),
        "up_gradient_exact": tensors_equal(source_mlp.c_fc.weight.grad, direct_mlp.c_fc.weight.grad),
        "down_gradient_exact": tensors_equal(
            source_mlp.c_proj.weight.grad, direct_mlp.c_proj.weight.grad
        ),
    }

    worker_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(Path(__file__).resolve()),
        "--owned-worker",
        "--source",
        str(source_path),
        "--direct",
        str(direct_path),
    ]
    worker = subprocess.run(
        worker_command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    optimizer = {
        "two_rank_gloo_exit_code": worker.returncode,
        "source_adapter_matches_explicit_owner": (
            worker.returncode == 0 and "OWNED_OPTIMIZER_EQUIVALENT" in worker.stdout
        ),
    }
    if worker.returncode:
        optimizer["stderr"] = worker.stderr[-4000:]

    ce = adaptive_ce_checks(source_text, direct_text)
    static = {
        "source_sha256_pinned": sha256(source_path) == SOURCE_SHA256,
        "launch_command_exact": LAUNCH.read_text().strip() == EXPECTED_LAUNCH,
        "review_trigger_hits": [trigger for trigger in TRIGGERS if trigger in direct_text],
        "runtime_forward_assignment_absent": ".forward =" not in direct_text,
        "runtime_step_assignment_absent": ".step =" not in direct_text,
        "runtime_ce_global_assignment_absent": (
            "train_module._softcap_ce_fwd =" not in direct_text
            and "train_module._softcap_ce_bwd =" not in direct_text
        ),
    }
    sections = [initialization, evaluation, fused_mlp, optimizer, ce]
    all_passed = all(all(value is True or value == 0.0 for value in section.values()) for section in sections)
    all_passed = all_passed and all(
        value is True or value == [] for key, value in static.items() if key != "source_sha256_pinned"
    )
    all_passed = all_passed and static["source_sha256_pinned"]
    report = {
        "schema": "dense6x1024-direct-review-cpu-equivalence-v1",
        "scope": "local CPU/static implementation evidence; no Trainium execution and no independent approval",
        "source": {"path": str(source_path.relative_to(ROOT)), "sha256": sha256(source_path)},
        "candidate": {"path": str(direct_path.relative_to(ROOT)), "sha256": sha256(direct_path)},
        "initialization": initialization,
        "evaluation": evaluation,
        "fused_mlp": fused_mlp,
        "owned_optimizer": optimizer,
        "adaptive_nki_ce_static": ce,
        "static": static,
        "hardware_residual": (
            "Adaptive CE launch wrappers, NKI compilation, full four-rank LNC2 ownership, timing, "
            "and fresh-run quality require the independent Trainium validation gates."
        ),
        "all_cpu_static_checks_passed": all_passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
