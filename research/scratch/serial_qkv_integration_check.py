#!/usr/bin/env python3
"""Gate the real attention adapter, master/cache lifecycle and causal behavior.

Independent projection FP64 gates are a prerequisite. This uses the campaign's
existing, unchanged paired-native BF16 bounds for the complete nonlinear
attention path; exact equality remains required for causality, repeatability,
native evaluation and strict checkpoint restoration. No tolerance is fitted to
candidate results. No whole-model speed or quality claim follows.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
from mlp_fused_bf16_probe import compare
from serial_qkv_integration import install


class Probe(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        config = T.GPTConfig(sequence_len=128, vocab_size=8192, n_layer=1,
                             n_head=channels // 128, n_embd=channels, mlp_ratio=4)
        block = torch.nn.Module()
        block.attn = T.CausalSelfAttention(config, 0)
        self.transformer = torch.nn.ModuleDict(h=torch.nn.ModuleList([block]))

    def forward(self, q, k, v, ve, cos, sin):
        return self.transformer.h[0].attn(q, k, v, ve, cos, sin, None)


def refresh(model):
    for module in model.modules():
        if isinstance(module, T.Linear): module.refresh_weight_cache(torch.bfloat16)


def exact(name, actual, expected):
    a, b = actual.detach().cpu(), expected.detach().cpu()
    passed = bool(torch.isfinite(a).all()) and torch.equal(a, b)
    print('SERIAL_QKV_ATTENTION_EXACT ' + json.dumps(dict(name=name, passed=passed)), flush=True)
    assert passed, name


def run(channels, layout):
    torch.manual_seed(151031 + channels)
    baseline = Probe(channels).to('neuron')
    candidate = copy.deepcopy(baseline)
    contract = install(candidate, 'neuron', T, layout)
    native = torch.compile(baseline, backend='neuron', fullgraph=True, dynamic=False)
    optimized = torch.compile(candidate, backend='neuron', fullgraph=True, dynamic=False)
    host_inputs = [torch.nn.functional.rms_norm(torch.randn(4, 128, channels), (channels,)).bfloat16()
                   for _ in range(3)]
    host_inputs.append((torch.randn(4, 128, channels) * .1).bfloat16())
    angles = torch.outer(torch.arange(128).float(),
                         100000.0**(-torch.arange(0, 128, 2).float() / 128))
    cos, sin = [v.reshape(1, 128, 1, 64).to('neuron') for v in (angles.cos(), angles.sin())]
    gradient = (torch.randn(4, 128, channels) * .01).bfloat16().to('neuron')
    for cached in (False, True):
        for step in range(2):
            baseline.zero_grad(set_to_none=True); candidate.zero_grad(set_to_none=True)
            if cached:
                refresh(baseline); refresh(candidate)
            for micro in range(2):
                inputs = [x + micro * .03125 for x in host_inputs]
                bx = [x.to('neuron').detach().clone().requires_grad_() for x in inputs]
                cx = [x.to('neuron').detach().clone().requires_grad_() for x in inputs]
                expected, output = native(*bx, cos, sin), optimized(*cx, cos, sin)
                assert type(expected.grad_fn).__name__ == type(output.grad_fn).__name__ == 'CompiledFunctionBackward'
                tag = f'qkv_attention:{channels}:{cached}:{step}:{micro}'
                compare(tag + ':output', output, expected)
                expected.backward(gradient); output.backward(gradient)
                for index, (actual, truth) in enumerate(zip(cx, bx)):
                    compare(tag + f':input_gradient{index}', actual.grad, truth.grad)
            for (name, actual), (other_name, truth) in zip(candidate.named_parameters(), baseline.named_parameters()):
                assert name == other_name and actual.dtype == truth.dtype == torch.float32
                if truth.grad is None:
                    assert actual.grad is None
                else:
                    assert actual.grad is not None and actual.grad.dtype == torch.float32
                    compare(f'qkv_attention:{channels}:{cached}:{step}:accumulated:{name}', actual.grad, truth.grad)
            # Same reference update isolates refreshed-cache behavior from
            # accumulated numerical trajectory differences across optimizers.
            with torch.no_grad():
                for actual, truth in zip(candidate.parameters(), baseline.parameters()):
                    if truth.grad is not None: truth.add_(truth.grad, alpha=-.002)
                    actual.copy_(truth)
    refresh(candidate); refresh(baseline)
    with torch.no_grad():
        inputs = [x.to('neuron') for x in host_inputs]
        first = optimized(*inputs, cos, sin)
        exact(f'{channels}:training_repeat', optimized(*inputs, cos, sin), first)
        changed = [x.clone() for x in inputs]
        for value in changed: value[0, 64:].add_(.5)
        result = optimized(*changed, cos, sin)
        exact(f'{channels}:future_independence', result[0, :64], first[0, :64])
        exact(f'{channels}:other_batch_independence', result[1:], first[1:])
        # Three batches require projection-row padding, while keeping the
        # original attention length and the RoPE fast-path geometry unchanged.
        short = [x[:3].contiguous() for x in inputs]
        compare(f'{channels}:padded_training_rows', optimized(*short, cos, sin), native(*short, cos, sin))
    candidate.eval(); baseline.eval()
    refresh(candidate); refresh(baseline)
    with torch.no_grad():
        exact(f'{channels}:native_evaluation', candidate(*inputs, cos, sin), baseline(*inputs, cos, sin))
    restored = Probe(channels).to('neuron')
    restored.load_state_dict(candidate.state_dict(), strict=True)
    restored.eval(); refresh(restored)
    with torch.no_grad():
        exact(f'{channels}:strict_native_restore', restored(*inputs, cos, sin), candidate(*inputs, cos, sin))
    print('SERIAL_QKV_ATTENTION_GATE ' + json.dumps(dict(
        channels=channels, layout=layout, passed=True, compiled_forward_backward=True,
        cached_and_uncached=True, original_separate_parameters=True,
        accumulation_and_refresh=True, distinct_input_gradients=True,
        causal_and_batch_independence_exact=True, padded_projection_rows=True,
        native_evaluation_and_restore_exact=True, contract=contract)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', choices=('tokens', 'channels'), required=True)
    args = parser.parse_args()
    import torch_neuronx  # noqa: F401
    torch.set_num_threads(4)
    T.USE_WEIGHT_CACHE = T.USE_BF16_NORM_OUTPUT = T.USE_NKI_ROPE_NORM = T.COMPILE_SDPA_DIRECT = True
    T.USE_FUSED_QKV = T.USE_PARALLEL_BLOCK = T.USE_NGRAM_VE = T.USE_HEAD_GATE = T.USE_NKI_ATTN = False
    T.USE_VE = True
    assert T._HAS_NKI
    run(256, args.layout)
    torch._dynamo.reset()
    print('SERIAL_QKV_ATTENTION_GUARD_RESET ' + json.dumps(dict(
        completed_width=256, next_width=1024, shared_device_cache_purged=False)), flush=True)
    run(1024, args.layout)
    print('SERIAL_QKV_INTEGRATION_RESULT ' + json.dumps(dict(
        layout=args.layout, widths=[256, 1024], passed=True,
        paired_native_bounds_unchanged=True, native_rope_geometry_preserved=True,
        whole_model_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
