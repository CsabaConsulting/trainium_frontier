#!/usr/bin/env python3
"""Gate actual Linear master weights/caches and several accumulated microbatches."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T

from mlp_fused_bf16_probe import compare
from mlp_fused_integration import install


class Probe(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        config = T.GPTConfig(sequence_len=2048, vocab_size=8192, n_layer=1,
                             n_head=channels // 128, n_embd=channels, mlp_ratio=4)
        block = torch.nn.Module()
        block.mlp = T.MLP(config)
        self.transformer = torch.nn.ModuleDict({"h": torch.nn.ModuleList([block])})

    def forward(self, x):
        return self.transformer.h[0].mlp(x)


def refresh(model):
    for module in model.modules():
        if isinstance(module, T.Linear):
            module.refresh_weight_cache(torch.bfloat16)


def run(device, tokens, channels, normal_input_gradient=False, token_tile=512, grouped_io=False,
        weight_mode=0):
    torch.manual_seed(1023)
    T.USE_WEIGHT_CACHE = True
    T.USE_GATED_MLP = T.USE_COALESCED_SWIGLU = T.USE_NKI_RELU2 = False
    T.RELU2_TAU = 0.0
    baseline = Probe(channels).to(device)
    candidate = copy.deepcopy(baseline)
    before_rng = torch.get_rng_state().clone()
    install(candidate, device, T, normal_input_gradient=normal_input_gradient,
            token_tile=token_tile, grouped_io=grouped_io, weight_mode=weight_mode)
    assert torch.equal(before_rng, torch.get_rng_state())
    if device == "neuron":
        native = torch.compile(baseline, backend="neuron", fullgraph=True, dynamic=False)
        optimized = torch.compile(candidate, backend="neuron", fullgraph=True, dynamic=False)
    else:
        native, optimized = baseline, candidate
    x = torch.randn(4, tokens // 4, channels).bfloat16().to(device)
    g = (torch.randn(4, tokens // 4, channels) * .01).bfloat16().to(device)
    # Refresh is differentiable and every microbatch must accumulate into the
    # original FP32 master Parameter, including after a real parameter update.
    for cached in (False, True):
        for step in range(2):
            baseline.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            if cached:
                refresh(baseline)
                refresh(candidate)
            for micro in range(2):
                xx = x + (micro * .03125)
                bx = xx.detach().clone().requires_grad_()
                cx = xx.detach().clone().requires_grad_()
                expected, output = native(bx), optimized(cx)
                compare(f"integration_output:{tokens}:{channels}:{cached}:{step}:{micro}", output, expected)
                expected.backward(g)
                output.backward(g)
                compare(f"integration_dx:{tokens}:{channels}:{cached}:{step}:{micro}", cx.grad, bx.grad)
            for (name, p), (other_name, q) in zip(candidate.named_parameters(), baseline.named_parameters()):
                assert name == other_name and p.dtype == q.dtype == torch.float32
                assert p.grad is not None and p.grad.dtype == torch.float32
                compare(f"integration_accumulated_gradient:{tokens}:{channels}:{cached}:{step}:{name}", p.grad, q.grad)
            # Apply the SAME reference update to both copies so the next gate
            # isolates cache refresh/backward rather than compounded drift.
            with torch.no_grad():
                for p, q in zip(candidate.parameters(), baseline.parameters()):
                    q.add_(q.grad, alpha=-.001)
                    p.copy_(q)
    candidate.eval()
    baseline.eval()
    refresh(candidate)
    refresh(baseline)
    with torch.no_grad():
        xx = x[:2, :31]
        torch.testing.assert_close(candidate(xx).cpu(), baseline(xx).cpu(), rtol=0, atol=0)
    restored = Probe(channels).to(device)
    restored.load_state_dict(candidate.state_dict(), strict=True)
    restored.eval()
    refresh(restored)
    with torch.no_grad():
        torch.testing.assert_close(restored(xx).cpu(), candidate(xx).cpu(), rtol=0, atol=0)
    print("FUSED_MLP_INTEGRATION_GATE " + json.dumps(dict(
        tokens=tokens, channels=channels, passed=True, fp32_master_gradients=True,
        normal_input_gradient=normal_input_gradient,
        token_tile=token_tile, grouped_io=grouped_io, weight_mode=weight_mode,
        cached_and_uncached_paths=True, accumulation_and_refresh=True,
        eval_exact=True, strict_native_checkpoint_load_passed=True)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "neuron"), default="neuron")
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--normal-dx", action="store_true")
    ap.add_argument("--token-tile", type=int, choices=(128, 256, 512), default=512)
    ap.add_argument("--grouped-io", action="store_true")
    ap.add_argument("--weight-mode", type=int, choices=(0, 1, 2), default=0,
                    help="0: cache both;1: stream down;2: stream both weight matrices.")
    args = ap.parse_args()
    if args.weight_mode and not args.grouped_io:
        ap.error("streaming weight modes require --grouped-io")
    torch.set_num_threads(4)
    if args.device == "neuron":
        import torch_neuronx  # noqa:F401
    run(args.device, 256, 256, args.normal_dx, args.token_tile, args.grouped_io, args.weight_mode)
    if args.real_shape:
        run(args.device, 8192, 1024, args.normal_dx, args.token_tile, args.grouped_io, args.weight_mode)
    print("PASS: fused MLP master parameter, cache, accumulation and checkpoint gates", flush=True)


if __name__ == "__main__":
    main()
