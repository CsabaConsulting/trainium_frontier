#!/usr/bin/env python3
"""Diagnose a failed MLP region-boundary gate without relaxing its tolerances.

Keep the forward/backward implementations uninstrumented. Compare returned
outputs and gradients after execution, repeat changed/zero inputs, and use
independent FP64 GEMMs with explicit BF16 rounding as a numerical reference.
This is a diagnostic: it records gate failures and never authorizes promotion.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import torch

from mlp_fused_integration import install
from mlp_fused_integration_check import Probe, T, refresh


def autograd_nodes(tensor):
    pending, seen, counts = [tensor.grad_fn], set(), Counter()
    while pending and len(seen) < 100:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        counts[type(node).__name__] += 1
        pending.extend(n for n, _ in node.next_functions)
    return dict(counts)


def metrics(actual, expected):
    a, b = actual.double(), expected.double()
    error = (a - b).abs()
    rms = b.square().mean().sqrt()
    relative = float(error.square().mean().sqrt() / rms.clamp_min(1e-30))
    failures = int((error > .03 * b.abs() + .008 * rms + 1e-30).sum())
    return dict(finite=bool(torch.isfinite(a).all()), exact=bool(torch.equal(a, b)),
                bound_failures=failures, relative_rms=relative,
                max_abs=float(error.max()), reference_rms=float(rms),
                original_gate_passed=bool(torch.isfinite(a).all()) and not failures and relative < .004)


def rounded_reference(x, up, down, grad, round_hidden=True, round_pre=True):
    """Exact sums of BF16 operands; quantization locations are explicit."""
    def bf(value):
        return value.bfloat16().double()

    x, up, down, grad = (v.double() for v in (x, up, down, grad))
    relu = bf(x @ up.T).clamp_min(0)
    activated = bf(relu.square())
    output = bf(activated @ down.T)
    hidden = grad @ down
    if round_hidden:
        hidden = bf(hidden)
    pre = hidden * (2 * relu)
    if round_pre:
        pre = bf(pre)
    return dict(output=output, dx=bf(pre @ up), dup=bf(pre.T @ x),
                ddown=bf(grad.T @ activated))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--normal-dx", action="store_true")
    ap.add_argument("--accumulate", action=argparse.BooleanOptionalAction, default=True,
                    help="Retain parameter gradients between cases, matching the failing second microbatch.")
    ap.add_argument("--variants", default="native,native_autograd_joined,native_autograd_separate,nki_joined,nki_separate",
                    help="Comma-separated variants, in execution order. A single variant tests cross-model interference.")
    args = ap.parse_args()
    import torch_neuronx  # noqa:F401
    from torch_neuronx.neuron_dynamo_backend.cache_artifact import ArtifactType
    from torch_neuronx.neuron_dynamo_backend.compile import get_compiled_cache_keys
    torch.set_num_threads(4)
    # Five models times two cache states intentionally need more guard variants
    # than the usual default. Keep compilation enabled; do not suppress errors.
    torch._dynamo.config.recompile_limit = 32
    torch.manual_seed(1023)
    T.USE_WEIGHT_CACHE = True
    T.USE_GATED_MLP = T.USE_COALESCED_SWIGLU = T.USE_NKI_RELU2 = False
    T.RELU2_TAU = 0.0
    original = Probe(256).to("neuron")
    prepared = {"native": original}
    specifications = (("native_autograd_joined", "native", False),
                      ("native_autograd_separate", "native", True),
                      ("nki_joined", "nki", False),
                      ("nki_separate", "nki", True))
    names = args.variants.split(",")
    allowed = {"native", *(name for name, _, _ in specifications)}
    if not names or len(set(names)) != len(names) or not set(names) <= allowed:
        ap.error("--variants needs unique valid names")
    before_rng = torch.get_rng_state().clone()
    for name, backend, separate in specifications:
        if name not in names:
            continue
        model = copy.deepcopy(original)
        install(model, "neuron", T, normal_input_gradient=args.normal_dx,
                grouped_io=backend == "nki", weight_mode=3 if backend == "nki" else 0,
                forward_backend=backend, separate_region=separate)
        prepared[name] = model
    assert torch.equal(before_rng, torch.get_rng_state())
    models = {name: prepared[name] for name in names}
    compiled = {name: torch.compile(model, backend="neuron", dynamic=False,
                                     fullgraph="separate" not in name)
                for name, model in models.items()}
    x = torch.randn(4, 64, 256).bfloat16().to("neuron")
    grad = (torch.randn(4, 64, 256) * .01).bfloat16().to("neuron")
    original_up = original.transformer.h[0].mlp.c_fc.weight.detach().cpu().clone()
    original_down = original.transformer.h[0].mlp.c_proj.weight.detach().cpu().clone()
    up, down = original_up.bfloat16(), original_down.bfloat16()
    cpu_grad = grad.detach().cpu().reshape(256, 256)
    state = dict(purpose=__doc__, shape=[4, 64, 256], normal_input_gradient=args.normal_dx,
                 numerical_tolerance_changed=False, promotion_allowed=False,
                 diagnostic_recompile_limit=32, accumulate=args.accumulate, variants=names,
                 cases=[], repeats=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for cached in (False, True):
        if cached:
            for model in models.values():
                refresh(model)
        for model in models.values():
            model.zero_grad(set_to_none=True)
        previous = {}
        reference_gradient_sums = {}
        for case in ("base", "changed", "base_again", "changed_again", "zero", "changed_after_zero"):
            kind = "base" if case.startswith("base") else "zero" if case == "zero" else "changed"
            xx = torch.zeros_like(x) if kind == "zero" else x + (0.03125 if kind == "changed" else 0.0)
            actual = {}
            execution_info = {}
            for name, model in models.items():
                if not args.accumulate:
                    model.zero_grad(set_to_none=True)
                leaf = xx.detach().clone().requires_grad_()
                output = compiled[name](leaf)
                nodes = autograd_nodes(output)
                output.backward(grad)
                mlp = model.transformer.h[0].mlp
                actual[name] = dict(output=output.detach().cpu().reshape(256, 256).clone(),
                                    dx=leaf.grad.detach().cpu().reshape(256, 256).clone(),
                                    dup=mlp.c_fc.weight.grad.detach().cpu().clone(),
                                    ddown=mlp.c_proj.weight.grad.detach().cpu().clone())
                execution_info[name] = dict(
                    autograd_nodes=nodes,
                    compiled_cache_keys=sorted(get_compiled_cache_keys(ArtifactType.NEURON_NEFF)),
                    master_weights_unchanged=bool(
                        torch.equal(mlp.c_fc.weight.detach().cpu(), original_up)
                        and torch.equal(mlp.c_proj.weight.detach().cpu(), original_down)))
            cpu_x = xx.detach().cpu().reshape(256, 256)
            references = {
                "strict_bf16": rounded_reference(cpu_x, up, down, cpu_grad),
                "unrounded_hidden": rounded_reference(cpu_x, up, down, cpu_grad, False, True),
                "unrounded_pre": rounded_reference(cpu_x, up, down, cpu_grad, True, False),
                "unrounded_both": rounded_reference(cpu_x, up, down, cpu_grad, False, False),
            }
            if args.accumulate:
                for ref, values in references.items():
                    previous_sums = reference_gradient_sums.setdefault(ref, {})
                    for key in ("dup", "ddown"):
                        if key in previous_sums:
                            values[key] = (values[key].float() + previous_sums[key].float()).double()
                        previous_sums[key] = values[key]
            for name, values in actual.items():
                record = dict(cached=cached, case=case, variant=name,
                              execution=execution_info[name],
                              versus_native=({k: metrics(v, actual["native"][k]) for k, v in values.items()}
                                             if "native" in actual else None),
                              versus_cpu={ref: {k: metrics(v, expected[k]) for k, v in values.items()}
                                          for ref, expected in references.items()})
                state["cases"].append(record)
                print("MLP_BOUNDARY_DIAG " + json.dumps(record), flush=True)
                key = (name, kind)
                if key in previous:
                    # Parameter gradients intentionally accumulate; only their
                    # native/reference comparisons above apply in that mode.
                    repeated = dict(cached=cached, case=case, variant=name,
                                    exact={k: torch.equal(v, previous[key][k]) for k, v in values.items()
                                           if not args.accumulate or k in ("output", "dx")})
                    state["repeats"].append(repeated)
                    print("MLP_BOUNDARY_REPEAT " + json.dumps(repeated), flush=True)
                else:
                    previous[key] = values
            args.output.write_text(json.dumps(state, indent=2) + "\n")
    state["all_repeats_exact"] = all(all(r["exact"].values()) for r in state["repeats"])
    args.output.write_text(json.dumps(state, indent=2) + "\n")
    print("MLP_BOUNDARY_DIAG_COMPLETE " + json.dumps(dict(
        all_repeats_exact=state["all_repeats_exact"], promotion_allowed=False,
        output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
