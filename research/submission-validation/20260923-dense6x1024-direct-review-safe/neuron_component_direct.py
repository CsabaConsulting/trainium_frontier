#!/usr/bin/env python3
"""Fail-closed Neuron component gate for the direct-review dense6x1024 package.

The full mode must run under four-rank LNC2 ``torchrun`` on real Neuron.  It exercises the
candidate's direct fused-MLP autograd path, both adaptive CE grids, and explicit optimizer
ownership.  It does not load data, train the model, score a checkpoint, or make a quality claim.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_TRAIN_SHA256 = "c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537"
EXPECTED_WORLD_SIZE = 4
EXPECTED_LNC = "2"
VOCAB_SIZE = 8192
MLP_SHAPE = (32, 1024, 1024)
MLP_HIDDEN = 4096

# Declared before execution.  The fused MLP uses BF16 matmuls; its FP32 master-weight gradients
# are BF16-rounded at the custom-autograd boundary.  CE keeps logits and arithmetic in FP32.
TOLERANCES = {
    "mlp_output": {"atol": 0.0625, "rtol": 0.02},
    "mlp_input_gradient": {"atol": 0.002, "rtol": 0.03},
    "mlp_up_gradient": {"atol": 0.02, "rtol": 0.03},
    "mlp_down_gradient": {"atol": 0.02, "rtol": 0.03},
    "ce_loss": {"atol": 0.0002, "rtol": 0.0002},
    "ce_gradient": {"atol": 0.00005, "rtol": 0.0005},
    "optimizer_parameter": {"atol": 0.000001, "rtol": 0.000001},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def functions_named(tree: ast.AST, name: str) -> list[ast.FunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]


def classes_named(tree: ast.AST, name: str) -> list[ast.ClassDef]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name]


def static_wiring_checks(train_py: Path) -> dict[str, Any]:
    text = train_py.read_text()
    tree = ast.parse(text)
    mlp = classes_named(tree, "MLP")
    ce = classes_named(tree, "_SoftcapCrossEntropyNKI")
    main = functions_named(tree, "main")
    if len(mlp) != 1 or len(ce) != 1 or len(main) != 1:
        raise AssertionError("expected one MLP, _SoftcapCrossEntropyNKI, and main definition")
    mlp_text = ast.unparse(mlp[0])
    ce_text = ast.unparse(ce[0])
    main_text = ast.unparse(main[0])
    trigger_hits = [
        trigger
        for trigger in ("_submission_", "_install_submission_training", "MethodType", "SUBMISSION_")
        if trigger in text
    ]
    checks = {
        "candidate_sha256": sha256(train_py),
        "candidate_hash_pinned": sha256(train_py) == EXPECTED_TRAIN_SHA256,
        "mlp_calls_direct_fused_autograd": "_FusedMLP.apply" in mlp_text,
        "ce_selects_lanes": ce_text.count("selected_lanes") == 2,
        "ce_calls_wrapped_forward": "_softcap_ce_fwd_native[lanes]" in ce_text,
        "ce_calls_wrapped_backward": "_softcap_ce_bwd_native[lanes]" in ce_text,
        "main_constructs_explicit_owner": "owned_optimizer = OwnedOptimizer(optimizer)" in main_text,
        "main_steps_explicit_owner": "owned_optimizer.step(update_adamw=True)" in main_text,
        "review_trigger_hits": trigger_hits,
        "runtime_forward_assignment_absent": ".forward =" not in text,
        "runtime_step_assignment_absent": ".step =" not in text,
        "runtime_ce_global_assignment_absent": (
            "train_module._softcap_ce_fwd =" not in text
            and "train_module._softcap_ce_bwd =" not in text
        ),
    }
    checks["passed"] = all(
        value is True or value == [] or key in {"candidate_sha256", "passed"}
        for key, value in checks.items()
    )
    return checks


def import_candidate(train_py: Path):
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("dense6x1024_direct_neuron_component", train_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {train_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synchronize(torch, device) -> None:
    if not hasattr(torch, "neuron"):
        raise RuntimeError("torch.neuron is unavailable")
    torch.neuron.synchronize()


def graph_node_names(tensor) -> list[str]:
    pending = [tensor.grad_fn]
    seen: set[int] = set()
    names: list[str] = []
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        names.append(type(node).__name__)
        pending.extend(parent for parent, _index in getattr(node, "next_functions", ()))
    return names


def tensor_metrics(torch, actual, reference, tolerance: dict[str, float]) -> dict[str, Any]:
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        return {
            "passed": False,
            "shape_equal": actual.shape == reference.shape,
            "dtype_equal": actual.dtype == reference.dtype,
            "actual_shape": list(actual.shape),
            "reference_shape": list(reference.shape),
            "actual_dtype": str(actual.dtype),
            "reference_dtype": str(reference.dtype),
            **tolerance,
        }
    left = actual.detach().float()
    right = reference.detach().float()
    finite = bool((torch.isfinite(left).all() & torch.isfinite(right).all()).cpu().item())
    difference = (left - right).abs()
    maximum = float(difference.max().cpu().item())
    mean = float(difference.mean().cpu().item())
    limit = tolerance["atol"] + tolerance["rtol"] * right.abs()
    within = bool((difference <= limit).all().cpu().item())
    return {
        "passed": finite and within,
        "finite": finite,
        "shape_equal": True,
        "dtype_equal": True,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_abs": maximum,
        "mean_abs": mean,
        **tolerance,
    }


def nested_exact(torch, left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(nested_exact(torch, left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(nested_exact(torch, a, b) for a, b in zip(left, right))
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and bool(left.detach().cpu().equal(right.detach().cpu()))
        )
    return type(left) is type(right) and left == right


def nested_finite(torch, value) -> bool:
    if isinstance(value, dict):
        return all(nested_finite(torch, item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(nested_finite(torch, item) for item in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().cpu().item())
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def fused_mlp_gate(torch, candidate, device, seed: int) -> dict[str, Any]:
    import torch.nn.functional as F

    batch, sequence, channels = MLP_SHAPE
    tokens = batch * sequence
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    x_host = torch.randn(MLP_SHAPE, generator=generator, dtype=torch.float32).mul_(0.5).bfloat16()
    up_host = torch.randn(MLP_HIDDEN, channels, generator=generator).mul_(0.02)
    down_host = torch.randn(channels, MLP_HIDDEN, generator=generator).mul_(0.02)
    upstream_host = torch.randn(MLP_SHAPE, generator=generator).mul_(0.25).bfloat16()
    config = candidate.GPTConfig(
        sequence_len=sequence,
        vocab_size=VOCAB_SIZE,
        n_layer=1,
        n_head=channels // 128,
        n_embd=channels,
        mlp_ratio=4,
        attn_scale=0.0,
        n_kv_heads=0,
    )
    mlp = candidate.MLP(config)
    with torch.no_grad():
        mlp.c_fc.weight.copy_(up_host)
        mlp.c_proj.weight.copy_(down_host)
    mlp.to(device)
    mlp.train()
    mlp.c_fc.refresh_weight_cache(torch.bfloat16)
    mlp.c_proj.refresh_weight_cache(torch.bfloat16)
    x = x_host.to(device).requires_grad_(True)
    upstream = upstream_host.to(device)
    output = mlp(x)
    node_names = graph_node_names(output)
    fused_node_names = [name for name in node_names if "FusedMLP" in name]
    loss = (output.float() * upstream.float()).sum() / tokens
    loss.backward()
    synchronize(torch, device)
    actual = {
        "output": output.detach(),
        "input_gradient": x.grad.detach(),
        "up_gradient": mlp.c_fc.weight.grad.detach(),
        "down_gradient": mlp.c_proj.weight.grad.detach(),
    }

    # Independent eager/autograd expression: no candidate helper, custom Function, or saved layout.
    reference_x = x_host.to(device).requires_grad_(True)
    reference_up = up_host.to(device).requires_grad_(True)
    reference_down = down_host.to(device).requires_grad_(True)
    hidden = F.relu(F.linear(reference_x, reference_up.to(torch.bfloat16)))
    reference_output = F.linear(hidden.square(), reference_down.to(torch.bfloat16))
    reference_loss = (reference_output.float() * upstream.float()).sum() / tokens
    reference_loss.backward()
    synchronize(torch, device)
    reference = {
        "output": reference_output.detach(),
        "input_gradient": reference_x.grad.detach(),
        "up_gradient": reference_up.grad.detach(),
        "down_gradient": reference_down.grad.detach(),
    }
    metrics = {
        name: tensor_metrics(torch, actual[name], reference[name], TOLERANCES[f"mlp_{name}"])
        for name in actual
    }
    branch = {
        "training": mlp.training,
        "input_dtype": str(x.dtype),
        "tokens": tokens,
        "tile": min(512, tokens // 2),
        "candidate_trigger_shape": list(MLP_SHAPE),
        "cached_bf16_weights": (
            mlp.c_fc._cached_weight is not None
            and mlp.c_proj._cached_weight is not None
            and mlp.c_fc._cached_weight.dtype == torch.bfloat16
            and mlp.c_proj._cached_weight.dtype == torch.bfloat16
        ),
        "fused_autograd_nodes": fused_node_names,
        "fused_branch_exercised": bool(fused_node_names),
    }
    passed = all(metric["passed"] for metric in metrics.values()) and all(
        (
            branch["training"],
            branch["input_dtype"] == "torch.bfloat16",
            branch["tokens"] == 32768,
            branch["tile"] == 512,
            branch["cached_bf16_weights"],
            branch["fused_branch_exercised"],
        )
    )
    result = {"passed": passed, "branch": branch, "metrics": metrics}
    del mlp, x, output, loss, reference_x, reference_up, reference_down, hidden, reference_output
    del reference_loss, upstream, actual, reference
    gc.collect()
    synchronize(torch, device)
    return result


class GridProbe:
    """Record grid selection while forwarding every call to the real wrapped NKI kernel."""

    def __init__(self, wrapped, label: str):
        self.wrapped = wrapped
        self.label = label
        self.calls: list[dict[str, Any]] = []

    def __getitem__(self, lanes: int):
        real_call = self.wrapped[lanes]

        def invoke(*args, **kwargs):
            self.calls.append(
                {
                    "label": self.label,
                    "lanes": int(lanes),
                    "shapes": [list(value.shape) for value in args if hasattr(value, "shape")],
                }
            )
            return real_call(*args, **kwargs)

        return invoke


def adaptive_ce_case(torch, candidate, device, seed: int, rows: int) -> dict[str, Any]:
    import torch.nn.functional as F

    expected_lanes = 1 if rows == 128 else 2
    if candidate.selected_lanes(rows) != expected_lanes:
        raise AssertionError(f"selected_lanes({rows}) did not return {expected_lanes}")
    generator = torch.Generator(device="cpu").manual_seed(seed + 200 + rows)
    logits_host = torch.randn(rows, VOCAB_SIZE, generator=generator).mul_(6.0)
    targets_host = torch.randint(VOCAB_SIZE, (rows,), generator=generator)
    targets_host[:4] = torch.tensor([0, VOCAB_SIZE - 1, VOCAB_SIZE // 2 - 1, VOCAB_SIZE // 2])
    upstream_host = torch.randn(rows, generator=generator).mul_(0.25)
    original_forward = candidate._softcap_ce_fwd_native
    original_backward = candidate._softcap_ce_bwd_native
    forward_probe = GridProbe(original_forward, "forward")
    backward_probe = GridProbe(original_backward, "backward")
    candidate._softcap_ce_fwd_native = forward_probe
    candidate._softcap_ce_bwd_native = backward_probe
    try:
        logits = logits_host.to(device).requires_grad_(True)
        targets = targets_host.to(device)
        upstream = upstream_host.to(device)
        loss = candidate.softcap_cross_entropy(logits, targets, "none")
        node_names = graph_node_names(loss)
        direct_nodes = [name for name in node_names if "SoftcapCrossEntropyNKI" in name]
        gradient, = torch.autograd.grad(loss, logits, grad_outputs=upstream)
        synchronize(torch, device)
    finally:
        candidate._softcap_ce_fwd_native = original_forward
        candidate._softcap_ce_bwd_native = original_backward

    # Independent native PyTorch softcap15 CE/autograd reference.
    reference_logits = logits_host.to(device).requires_grad_(True)
    reference_targets = targets_host.to(device)
    capped = 15.0 * torch.tanh(reference_logits / 15.0)
    reference_loss = F.cross_entropy(capped, reference_targets, reduction="none")
    reference_gradient, = torch.autograd.grad(
        reference_loss,
        reference_logits,
        grad_outputs=upstream_host.to(device),
    )
    synchronize(torch, device)
    metrics = {
        "loss": tensor_metrics(torch, loss.detach(), reference_loss.detach(), TOLERANCES["ce_loss"]),
        "gradient": tensor_metrics(
            torch, gradient.detach(), reference_gradient.detach(), TOLERANCES["ce_gradient"]
        ),
    }
    wiring = {
        "rows": rows,
        "vocab": VOCAB_SIZE,
        "expected_lanes": expected_lanes,
        "selected_lanes": candidate.selected_lanes(rows),
        "direct_autograd_nodes": direct_nodes,
        "direct_autograd_exercised": bool(direct_nodes),
        "wrapped_forward_calls": forward_probe.calls,
        "wrapped_backward_calls": backward_probe.calls,
        "wrapped_forward_exercised_once": len(forward_probe.calls) == 1,
        "wrapped_backward_exercised_once": len(backward_probe.calls) == 1,
        "wrapped_forward_selected_expected_grid": [call["lanes"] for call in forward_probe.calls]
        == [expected_lanes],
        "wrapped_backward_selected_expected_grid": [call["lanes"] for call in backward_probe.calls]
        == [expected_lanes],
    }
    passed = all(metric["passed"] for metric in metrics.values()) and all(
        (
            wiring["selected_lanes"] == expected_lanes,
            wiring["direct_autograd_exercised"],
            wiring["wrapped_forward_exercised_once"],
            wiring["wrapped_backward_exercised_once"],
            wiring["wrapped_forward_selected_expected_grid"],
            wiring["wrapped_backward_selected_expected_grid"],
        )
    )
    result = {"passed": passed, "wiring": wiring, "metrics": metrics}
    del logits, targets, upstream, loss, gradient, reference_logits, reference_targets, capped
    del reference_loss, reference_gradient
    gc.collect()
    synchronize(torch, device)
    return result


def adaptive_ce_gate(torch, candidate, device, seed: int) -> dict[str, Any]:
    if not candidate._HAS_NKI:
        raise RuntimeError("candidate imported without the native NKI stack")
    cases = {str(rows): adaptive_ce_case(torch, candidate, device, seed, rows) for rows in (128, 512)}
    return {"passed": all(case["passed"] for case in cases.values()), "cases": cases}


def optimizer_state_coverage(torch, owner) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    passed = True
    for group_index, (source, positions, owned) in enumerate(
        zip(owner.optimizer.param_groups, owner.positions, owner.groups)
    ):
        actual_parameters = [source["params"][position] for position in positions]
        prefix_matches = all(a is b for a, b in zip(actual_parameters, owned["params"]))
        state_ok = True
        detail: dict[str, Any] = {}
        if positions and source["kind"] == "adamw":
            state_ok = all(
                parameter in owner.optimizer.state
                and {"step", "exp_avg", "exp_avg_sq"}.issubset(owner.optimizer.state[parameter])
                for parameter in actual_parameters
            )
            detail["adamw_state_parameters"] = len(actual_parameters)
        elif positions and source["kind"] == "muon":
            state = owner.optimizer.state[owned["params"][0]]
            required = {
                "momentum_buffer",
                "second_momentum_buffer",
                "stacked_grads",
                "stacked_params",
            }
            state_ok = required.issubset(state) and all(
                state[name].shape[0] == len(owned["params"])
                for name in required
            )
            detail.update(
                muon_actual_parameters=len(actual_parameters),
                muon_stacked_rows=len(owned["params"]),
                singleton_padding=group_index in owner.padding,
            )
        check = {
            "group": group_index,
            "kind": source["kind"],
            "positions": positions,
            "prefix_matches_source": prefix_matches,
            "state_complete": state_ok,
            **detail,
        }
        checks.append(check)
        passed = passed and prefix_matches and state_ok
    return {"passed": passed, "groups": checks}


def optimizer_gate(torch, candidate, device, seed: int, rank: int, world_size: int) -> dict[str, Any]:
    config = candidate.GPTConfig(
        sequence_len=128,
        vocab_size=VOCAB_SIZE,
        n_layer=2,
        n_head=1,
        n_embd=128,
        mlp_ratio=4,
        attn_scale=0.0,
        n_kv_heads=0,
    )
    torch.manual_seed(seed + 301)
    reference_model = candidate.GPT(config)
    reference_model.init_weights()
    torch.manual_seed(seed + 301)
    owned_model = candidate.GPT(config)
    owned_model.init_weights()
    if not all(
        left.equal(right)
        for left, right in zip(reference_model.state_dict().values(), owned_model.state_dict().values())
    ):
        raise AssertionError("optimizer fixtures did not initialize identically")
    reference_model.to(device)
    owned_model.to(device)
    candidate.COMPUTE_DTYPE = torch.bfloat16
    reference_optimizer = reference_model.setup_optimizer()
    base_optimizer = owned_model.setup_optimizer()
    owner = candidate.OwnedOptimizer(base_optimizer)
    api = {
        "base_param_groups_retained": hasattr(base_optimizer, "param_groups"),
        "base_zero_grad_callable": callable(getattr(base_optimizer, "zero_grad", None)),
        "owner_has_no_param_groups_api": not hasattr(owner, "param_groups"),
        "owner_has_no_zero_grad_api": not hasattr(owner, "zero_grad"),
        "explicit_owner_is_distinct_from_base": owner is not base_optimizer,
        "base_state_empty_before_first_update": not base_optimizer.state,
    }
    expected_slots = {
        (group_index, position)
        for group_index, group in enumerate(base_optimizer.param_groups)
        for position in range(len(group["params"]))
    }
    actual_slots = {(entry["group"], entry["position"]) for entry in owner.entries}
    plan = {
        "entries": len(owner.entries),
        "parameters": len(owner.parameters),
        "expected_slots": len(expected_slots),
        "slot_set_exact": actual_slots == expected_slots,
        "entry_indices_exact": [entry["index"] for entry in owner.entries]
        == list(range(len(owner.entries))),
        "parameter_ids_exact": owner.parameter_ids == [id(parameter) for parameter in owner.parameters],
        "all_four_owners_present": {entry["owner"] for entry in owner.entries} == set(range(world_size)),
        "local_positions_exact": all(
            positions
            == [
                entry["position"]
                for entry in owner.entries
                if entry["group"] == group_index and entry["owner"] == rank
            ]
            for group_index, positions in enumerate(owner.positions)
        ),
        "digest": hashlib.sha256(
            json.dumps(owner.entries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "summary": owner.summary,
    }
    reference_named = list(reference_model.named_parameters())
    owned_named = list(owned_model.named_parameters())
    if [name for name, _parameter in reference_named] != [name for name, _parameter in owned_named]:
        raise AssertionError("optimizer fixture parameter names differ")
    gradient_generator = torch.Generator(device="cpu").manual_seed(seed + 302)
    for (_name_a, reference_parameter), (_name_b, owned_parameter) in zip(reference_named, owned_named):
        host_gradient = torch.randn(
            reference_parameter.shape,
            generator=gradient_generator,
            dtype=torch.float32,
        ).mul_(0.01)
        device_gradient = host_gradient.to(device)
        reference_parameter.grad = device_gradient
        owned_parameter.grad = device_gradient.clone()
    for reference_group, owned_group in zip(reference_optimizer.param_groups, base_optimizer.param_groups):
        for group in (reference_group, owned_group):
            group["lr"] = group["initial_lr"] * 0.5
            if group["kind"] == "muon":
                group["momentum"] = 0.9
                group["weight_decay"] = 0.1
                group["beta2"] = 0.9
    reference_optimizer.step(update_adamw=True)
    owner.step(update_adamw=True)
    synchronize(torch, device)
    parameter_metrics = {
        name_a: tensor_metrics(
            torch,
            owned_parameter,
            reference_parameter,
            TOLERANCES["optimizer_parameter"],
        )
        for (name_a, reference_parameter), (_name_b, owned_parameter) in zip(
            reference_named, owned_named
        )
    }
    worst_parameter = max(parameter_metrics, key=lambda name: parameter_metrics[name]["max_abs"])
    parameter_summary = {
        "passed": all(metric["passed"] for metric in parameter_metrics.values()),
        "count": len(parameter_metrics),
        "worst_parameter": worst_parameter,
        "max_abs": parameter_metrics[worst_parameter]["max_abs"],
        **TOLERANCES["optimizer_parameter"],
    }
    schedule_propagated = all(
        all(owned_group[key] == source[key] for key in source if key != "params")
        for source, owned_group in zip(base_optimizer.param_groups, owner.groups)
    )
    coverage = optimizer_state_coverage(torch, owner)
    payload = owner.state_dict()
    schema = {
        "schema": payload.get("schema"),
        "schema_exact": payload.get("schema") == "fresh-start-owned-optimizer-v1",
        "rank_exact": payload.get("rank") == rank,
        "world_size_exact": payload.get("world_size") == world_size,
        "entries_exact": payload.get("entries") == owner.entries,
        "pad_singleton_muon_exact": payload.get("pad_singleton_muon") is True,
        "optimizer_state_finite": nested_finite(torch, payload["optimizer"]),
        "stacked_transient_fields_absent": all(
            name not in fields
            for fields in payload["optimizer"]["state"].values()
            for name in ("stacked_params", "stacked_grads")
        ),
    }
    torch.manual_seed(seed + 303)
    reload_model = candidate.GPT(config)
    reload_model.init_weights()
    reload_model.to(device)
    reload_optimizer = reload_model.setup_optimizer()
    reload_owner = candidate.OwnedOptimizer(reload_optimizer)
    reload_owner.load_state_dict(payload)
    roundtrip_exact = nested_exact(torch, payload, reload_owner.state_dict())
    bad_schema_rejected = False
    try:
        reload_owner.load_state_dict({**payload, "schema": "invalid-schema"})
    except ValueError:
        bad_schema_rejected = True
    state = {
        "coverage": coverage,
        "schema": schema,
        "roundtrip_exact": roundtrip_exact,
        "bad_schema_rejected": bad_schema_rejected,
        "owner_calls": owner.calls,
        "schedule_propagated": schedule_propagated,
        "send_finite": bool(torch.isfinite(owner.send).all().cpu().item()),
        "received_finite": bool(torch.isfinite(owner.received).all().cpu().item()),
    }
    passed = (
        all(api.values())
        and all(value is True or key in {"entries", "parameters", "expected_slots", "digest", "summary"}
                for key, value in plan.items())
        and parameter_summary["passed"]
        and coverage["passed"]
        and all(value is True or key == "schema" for key, value in schema.items())
        and roundtrip_exact
        and bad_schema_rejected
        and owner.calls == 1
        and schedule_propagated
        and state["send_finite"]
        and state["received_finite"]
    )
    result = {
        "passed": passed,
        "api_split": api,
        "ownership_plan": plan,
        "parameter_reference": parameter_summary,
        "state": state,
    }
    del reference_model, owned_model, reload_model, reference_optimizer, base_optimizer
    del owner, reload_optimizer, reload_owner, payload, parameter_metrics
    gc.collect()
    synchronize(torch, device)
    return result


def guarded_gate(name: str, function) -> dict[str, Any]:
    try:
        result = function()
        if not isinstance(result, dict) or not isinstance(result.get("passed"), bool):
            raise TypeError(f"{name} did not return a boolean passed field")
        return result
    except Exception as error:  # Fail closed while allowing every rank to reach global reduction.
        return {
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=12),
        }
    finally:
        gc.collect()


def numeric_metric(result: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = result
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def run_neuron(args, train_py: Path, static: dict[str, Any]) -> tuple[dict[str, Any], int]:
    import torch
    import torch.distributed as dist

    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != EXPECTED_LNC:
        raise RuntimeError(f"NEURON_LOGICAL_NC_CONFIG must be {EXPECTED_LNC}")
    if int(os.environ.get("WORLD_SIZE", "0")) != EXPECTED_WORLD_SIZE:
        raise RuntimeError("launch must use torchrun --nproc_per_node=4")
    if dist.is_initialized():
        raise RuntimeError("process group was already initialized")
    torch.set_num_threads(args.host_threads)
    candidate = import_candidate(train_py)
    if not getattr(candidate, "_HAS_NKI", False):
        raise RuntimeError("candidate did not import the native NKI stack")
    initialized = False
    cleanup_succeeded = False
    rank = int(os.environ.get("RANK", "-1"))
    local_report: dict[str, Any] = {}
    global_report: dict[str, Any] = {}
    try:
        dist.init_process_group("neuron")
        initialized = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != EXPECTED_WORLD_SIZE:
            raise RuntimeError(f"expected {EXPECTED_WORLD_SIZE} ranks, got {world_size}")
        device = torch.device("neuron")
        torch.manual_seed(args.seed)
        environment = {
            "rank": rank,
            "world_size": world_size,
            "device": str(device),
            "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
            "torch_version": torch.__version__,
            "torch_neuron_available": hasattr(torch, "neuron"),
            "candidate_has_nki": candidate._HAS_NKI,
            "seed": args.seed,
        }
        mlp = guarded_gate("fused_mlp", lambda: fused_mlp_gate(torch, candidate, device, args.seed))
        ce = guarded_gate("adaptive_ce", lambda: adaptive_ce_gate(torch, candidate, device, args.seed))
        optimizer = guarded_gate(
            "owned_optimizer",
            lambda: optimizer_gate(torch, candidate, device, args.seed, rank, world_size),
        )
        local_report = {
            "rank": rank,
            "environment": environment,
            "fused_mlp": mlp,
            "adaptive_ce": ce,
            "owned_optimizer": optimizer,
            "passed": mlp["passed"] and ce["passed"] and optimizer["passed"],
        }
        stage_names = ("fused_mlp", "adaptive_ce", "owned_optimizer")
        status = torch.tensor(
            [int(local_report[name]["passed"]) for name in stage_names],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(status, op=dist.ReduceOp.SUM)
        status_counts = [int(value) for value in status.cpu().tolist()]
        plan_digest = optimizer.get("ownership_plan", {}).get("digest")
        plan_available = isinstance(plan_digest, str) and len(plan_digest) == 64
        plan_bytes = bytes.fromhex(plan_digest) if plan_available else bytes(32)
        plan_input = torch.tensor(list(plan_bytes), dtype=torch.float32, device=device)
        gathered_plan = torch.empty(32 * world_size, dtype=torch.float32, device=device)
        dist.all_gather_into_tensor(gathered_plan, plan_input)
        gathered_plan_rows = gathered_plan.cpu().reshape(world_size, 32)
        plan_digests_agree = plan_available and all(
            bool(gathered_plan_rows[index].equal(gathered_plan_rows[0]))
            for index in range(1, world_size)
        )
        metric_paths = (
            ("fused_mlp", "metrics", "output", "max_abs"),
            ("fused_mlp", "metrics", "input_gradient", "max_abs"),
            ("fused_mlp", "metrics", "up_gradient", "max_abs"),
            ("fused_mlp", "metrics", "down_gradient", "max_abs"),
            ("adaptive_ce", "cases", "128", "metrics", "loss", "max_abs"),
            ("adaptive_ce", "cases", "128", "metrics", "gradient", "max_abs"),
            ("adaptive_ce", "cases", "512", "metrics", "loss", "max_abs"),
            ("adaptive_ce", "cases", "512", "metrics", "gradient", "max_abs"),
            ("owned_optimizer", "parameter_reference", "max_abs"),
        )
        values = [numeric_metric(local_report, path) for path in metric_paths]
        available = torch.tensor(
            [int(value is not None) for value in values], dtype=torch.float32, device=device
        )
        maxima = torch.tensor(
            [0.0 if value is None else value for value in values],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(available, op=dist.ReduceOp.SUM)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        metric_names = [".".join(path) for path in metric_paths]
        global_report = {
            "rank_pass_counts": dict(zip(stage_names, status_counts)),
            "required_pass_count_per_stage": world_size,
            "max_abs_across_ranks": {
                name: float(value)
                for name, value, count in zip(metric_names, maxima.cpu().tolist(), available.cpu().tolist())
                if int(count) == world_size
            },
            "metrics_available_on_all_ranks": all(int(count) == world_size for count in available.cpu().tolist()),
            "ownership_plan_digest_all_ranks": plan_digests_agree,
            "all_ranks_passed": (
                all(count == world_size for count in status_counts)
                and all(int(count) == world_size for count in available.cpu().tolist())
                and plan_digests_agree
            ),
        }
        print(
            "DIRECT_NEURON_COMPONENT_RANK "
            + json.dumps(
                {
                    "rank": rank,
                    "passed": local_report["passed"],
                    "stages": {name: local_report[name]["passed"] for name in stage_names},
                    "errors": {
                        name: {
                            "type": local_report[name].get("error_type"),
                            "error": local_report[name].get("error"),
                        }
                        for name in stage_names
                        if not local_report[name]["passed"]
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
        dist.barrier()
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()
            cleanup_succeeded = not dist.is_initialized()
    passed = bool(global_report.get("all_ranks_passed")) and cleanup_succeeded
    report = {
        "schema": "dense6x1024-direct-neuron-component-v1",
        "scope": "standalone real-Neuron component gate; no data, training, scoring, or quality verdict",
        "candidate": {"path": str(train_py), "sha256": sha256(train_py)},
        "static_wiring": static,
        "tolerances": TOLERANCES,
        "expected_environment": {
            "world_size": EXPECTED_WORLD_SIZE,
            "NEURON_LOGICAL_NC_CONFIG": EXPECTED_LNC,
            "device": "neuron",
        },
        "distributed_initialized": initialized,
        "distributed_cleanup_succeeded": cleanup_succeeded,
        "rank0": local_report if rank == 0 else {"rank": rank},
        "global": global_report,
        "passed": passed,
    }
    return report, 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-py", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host-threads", type=int, default=4)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Validate the pinned source/direct wiring without importing torch or touching hardware.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_py = args.train_py.resolve()
    if not train_py.is_file():
        raise FileNotFoundError(train_py)
    static = static_wiring_checks(train_py)
    if not static["passed"]:
        report = {
            "schema": "dense6x1024-direct-neuron-component-v1",
            "mode": "static-only" if args.static_only else "neuron",
            "static_wiring": static,
            "passed": False,
        }
        write_report(args.report.resolve(), report)
        print("DIRECT_NEURON_COMPONENT " + json.dumps(report, sort_keys=True), flush=True)
        return 1
    if args.static_only:
        report = {
            "schema": "dense6x1024-direct-neuron-component-v1",
            "mode": "static-only",
            "candidate": {"path": str(train_py), "sha256": sha256(train_py)},
            "static_wiring": static,
            "tolerances": TOLERANCES,
            "passed": True,
            "qualification": "No Neuron operation was executed.",
        }
        write_report(args.report.resolve(), report)
        print("DIRECT_NEURON_COMPONENT " + json.dumps(report, sort_keys=True), flush=True)
        return 0
    report, exit_code = run_neuron(args, train_py, static)
    if int(os.environ.get("RANK", "-1")) == 0:
        write_report(args.report.resolve(), report)
        print("DIRECT_NEURON_COMPONENT " + json.dumps(report, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        failure = {
            "schema": "dense6x1024-direct-neuron-component-v1",
            "rank": int(os.environ.get("RANK", "-1")),
            "passed": False,
            "fatal_error_type": type(error).__name__,
            "fatal_error": str(error),
            "traceback": traceback.format_exc(limit=16),
        }
        if failure["rank"] in (-1, 0) and "--report" in sys.argv:
            try:
                report_index = sys.argv.index("--report") + 1
                write_report(Path(sys.argv[report_index]).resolve(), failure)
            except Exception as report_error:
                failure["report_write_error"] = f"{type(report_error).__name__}: {report_error}"
        print("DIRECT_NEURON_COMPONENT_FATAL " + json.dumps(failure, sort_keys=True), flush=True)
        raise SystemExit(2)
