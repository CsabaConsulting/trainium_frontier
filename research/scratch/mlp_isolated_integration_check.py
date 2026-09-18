#!/usr/bin/env python3
"""Compare actual MLP training against native tensors from a separate process.

The alternating-model diagnostic changes the native reference's backward
rounding after another model is compiled. Isolate the reference without changing
the existing numerical bound. Capture/compare must be separate guarded jobs.
Both exercise FP32 masters, BF16 caches, two accumulated microbatches, reference
updates, evaluation, and strict loading into the ordinary native model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from mlp_fused_bf16_probe import compare
from mlp_fused_integration import install
from mlp_fused_integration_check import Probe, T, refresh
from mlp_separate_diagnose import autograd_nodes


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args, tokens, channels):
    path = args.reference_directory / f"native-{tokens}-{channels}.pt"
    capture = args.action == "capture"
    if capture:
        assert not path.exists(), f"Do not overwrite a reference: {path}"
    else:
        reference = torch.load(path, map_location="cpu", weights_only=True)
        assert reference["shape"] == [4, tokens // 4, channels]
        assert reference["schema"] == 1 and reference["seed"] == 1023
        print("ISOLATED_MLP_REFERENCE " + json.dumps(dict(
            path=str(path), bytes=path.stat().st_size, sha256=digest(path))), flush=True)

    torch.manual_seed(1023)
    T.USE_WEIGHT_CACHE = True
    T.USE_GATED_MLP = T.USE_COALESCED_SWIGLU = T.USE_NKI_RELU2 = False
    T.RELU2_TAU = 0.0
    model = Probe(channels).to("neuron")
    x = torch.randn(4, tokens // 4, channels).bfloat16().to("neuron")
    grad = (torch.randn(4, tokens // 4, channels) * .01).bfloat16().to("neuron")
    if capture:
        reference = dict(schema=1, seed=1023, shape=list(x.shape),
                         initial_state=cpu_state(model), input=x.cpu(), gradient=grad.cpu(),
                         steps=[])
    else:
        model.load_state_dict(reference["initial_state"], strict=True)
        torch.testing.assert_close(x.cpu(), reference["input"], rtol=0, atol=0)
        torch.testing.assert_close(grad.cpu(), reference["gradient"], rtol=0, atol=0)
        before_rng = torch.get_rng_state().clone()
        install(model, "neuron", T, grouped_io=args.forward_backend == "nki",
                weight_mode=3 if args.forward_backend == "nki" else 0,
                forward_backend=args.forward_backend, separate_region=True)
        assert torch.equal(before_rng, torch.get_rng_state())

    compiled = torch.compile(model, backend="neuron", dynamic=False, fullgraph=capture)
    for cached in (False, True):
        for step in range(2):
            step_index = int(cached) * 2 + step
            expected_step = None if capture else reference["steps"][step_index]
            if not capture:
                assert expected_step["cached"] == cached and expected_step["step"] == step
            model.zero_grad(set_to_none=True)
            if cached:
                refresh(model)
            record = dict(cached=cached, step=step, micros=[])
            for micro in range(2):
                leaf = (x + micro * .03125).detach().clone().requires_grad_()
                output = compiled(leaf)
                nodes = autograd_nodes(output)
                assert nodes.get("CompiledFunctionBackward", 0) >= 1, nodes
                print("ISOLATED_MLP_EXECUTION " + json.dumps(dict(
                    action=args.action, tokens=tokens, channels=channels,
                    cached=cached, step=step, micro=micro, autograd_nodes=nodes)), flush=True)
                output.backward(grad)
                actual = dict(output=output.detach().cpu().clone(),
                              dx=leaf.grad.detach().cpu().clone())
                assert all(bool(torch.isfinite(value).all()) for value in actual.values())
                if capture:
                    record["micros"].append(actual)
                else:
                    for key, value in actual.items():
                        compare(f"isolated_{key}:{tokens}:{channels}:{cached}:{step}:{micro}",
                                value, expected_step["micros"][micro][key])
            gradients = {}
            for name, parameter in model.named_parameters():
                assert parameter.dtype == torch.float32
                assert parameter.grad is not None and parameter.grad.dtype == torch.float32
                gradients[name] = parameter.grad.detach().cpu().clone()
                assert bool(torch.isfinite(gradients[name]).all())
                if not capture:
                    compare(f"isolated_accumulated_gradient:{tokens}:{channels}:{cached}:{step}:{name}",
                            gradients[name], expected_step["gradients"][name])
            if capture:
                record["gradients"] = gradients
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(parameter.grad, alpha=-.001)
                record["next_state"] = cpu_state(model)
                reference["steps"].append(record)
            else:
                # Identical reference updates isolate cache refresh from drift.
                model.load_state_dict(expected_step["next_state"], strict=True)
                for name, value in cpu_state(model).items():
                    assert torch.equal(value, expected_step["next_state"][name])

    model.eval()
    refresh(model)
    with torch.no_grad():
        evaluated = model(x[:2, :31]).detach().cpu()
    if capture:
        reference["evaluation"] = evaluated
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reference, path)
        print("ISOLATED_MLP_CAPTURE " + json.dumps(dict(
            path=str(path), bytes=path.stat().st_size, sha256=digest(path),
            tokens=tokens, channels=channels, native_only=True)), flush=True)
    else:
        torch.testing.assert_close(evaluated, reference["evaluation"], rtol=0, atol=0)
        restored = Probe(channels).to("neuron")
        restored.load_state_dict(model.state_dict(), strict=True)
        restored.eval()
        refresh(restored)
        with torch.no_grad():
            torch.testing.assert_close(restored(x[:2, :31]).cpu(), evaluated, rtol=0, atol=0)
        print("ISOLATED_MLP_INTEGRATION_GATE " + json.dumps(dict(
            tokens=tokens, channels=channels, passed=True, numerical_tolerance_changed=False,
            forward_backend=args.forward_backend,
            reference_from_separate_process=True, fp32_master_gradients=True,
            cached_and_uncached_paths=True, accumulated_microbatches=2,
            updates_per_cache_state=2, eval_exact=True, strict_native_checkpoint_load=True)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action", choices=("capture", "compare"), required=True)
    ap.add_argument("--reference-directory", type=Path, required=True)
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--forward-backend", choices=("nki", "native"), default="nki",
                    help="Native control retains the same region boundary and custom backward.")
    args = ap.parse_args()
    import torch_neuronx  # noqa:F401
    torch.set_num_threads(4)
    run(args, 256, 256)
    if args.real_shape:
        run(args, 8192, 1024)
    print("PASS: isolated MLP " + args.action, flush=True)


if __name__ == "__main__":
    main()
